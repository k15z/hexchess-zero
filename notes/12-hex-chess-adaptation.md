# 12 — Adapting AlphaZero to a small hex-board variant

Notes for porting AlphaZero techniques to Glinski hexagonal chess (91 hex cells).
Numbers below are empirically measured against the actual engine where noted.

## What's different from chess and Go

| Aspect | Go 19x19 | Chess | Glinski hex chess |
|---|---|---|---|
| Board cells | 361 | 64 | 91 |
| Branching factor | ~250 | ~35 | **mean 39.8, median 40** (random play, n=75k positions) |
| Initial-position legal moves | ~360 | 20 | **51** (verified) |
| Average game length (plies) | ~250 | ~80 | **unknown — depends on player strength** (random play caps at 400+ via shuffles; minimax-d3 with softmax also shuffles to 200+; real MCTS games TBD) |
| Draws | very rare (komi) | ~50% top-level | **non-trivial** (random play: ~9% material/fifty draws + 2% stalemate; expect higher under engine play due to shuffle propensity) |
| Repetition rules | superko | threefold | threefold (verified — `is_game_over()` returns `draw_repetition`) |
| Move encoding size | ~362 | ~4672 | **4206** (pinned by golden hash test) |
| Symmetries | 8 (D4) | none | **central inversion only** — `(q,r) → (-q,-r)` (see §9 below) |

## Implications

### 1. Input encoding

Embed the hex grid in the smallest enclosing rectangle (11x11 for a radius-5 hex board). Add a
**validity mask plane** so the network can ignore invalid cells. Zero-pad invalid cells in all
other planes.

This is exactly the approach KataGo uses for variable-size Go boards (zero-pad to max size,
mask).

### 2. Move encoding

Use a deterministic (from-cell, to-cell, promotion) bijection. Aim for ~4000 indices, far less
than `91 × 91 = 8281`, by only enumerating moves that *could* be legal for some piece.

Critical: the bijection must be **stable forever**. Any change invalidates all training data.
Hash-test it as part of CI.

### 3. Branching factor → Dirichlet `α`

Heuristic `α ≈ 10 / branching_factor`:
- **Measured Glinski branching: mean 39.8, median 40** (random play, ~75k positions). Confirmed.
- `α = 10 / 40 = 0.25`. Compare: chess uses 0.3, Go uses 0.03.
- The empirical distribution is wide (stdev 21, p5=6, p95=73) — endgame positions can have very few moves; midgame can have 70+. The mean is what matters for noise calibration.

Use `α = 0.25`. **Confirmed empirically.**

### 4. MCTS sims

CLAUDE.md notes: ≥800 sims is the production minimum. This matches KataGo / Lc0 / AlphaZero. Do
not be tempted to evaluate at 100 sims to "save time" — it produces misleading rankings because
the NN's prior advantage compounds with depth.

For training games under PCR: full=800, fast=160, p_full=0.25.

### 5. Network sizing

Small game → small net is fine. Suggested starter:
- 8–12 residual blocks
- 128–192 filters
- SE in every block
- Global pool bias in 2 blocks
- WDL value head + (optional) MLH

This is ~2–4M params and trains comfortably on a single consumer GPU. Scale up later if you
plateau.

### 6. Replay window

KataGo's formula with a smaller `c`:
```
N_window = c · (1 + β · ((N_total / c)^α − 1) / α)
c = 25_000, α = 0.75, β = 0.4
```

For comparison: KataGo used `c = 250k` for Go; you have ~10x fewer positions per game and
faster game generation, so scaling `c` down by ~10x is reasonable.

### 7. Game length and resignation

**Game length is unproven and highly player-strength-dependent.**
- Random self-play: 78% timeout at 400 plies (random doesn't push for mate); minority that finished did so in 50–250 plies.
- Minimax depth-2 deterministic self-play: 48 plies (one repeated game; deterministic pathology).
- Minimax depth-3 + softmax sampling: timed out at 200 plies on every game I sampled — both sides oscillate when the position is "balanced". Glinski's dense piece interaction makes shuffles very easy.

Implications:
- **Don't trust the original "~80–120 plies" guess.** Plan for game-length p95 in the 200+ range until measured.
- MLH is critical to break shuffles (already in chunk 3). Without MLH, expect timeouts to dominate.
- Resignation is genuinely useful here, not optional — many "drawn-by-shuffle" games could be resigned earlier if one side has a meaningful disadvantage.
- Per-game move-limit cap should be > 250 plies in self-play config to avoid truncating real games.

### 8. Draws and repetition

Draws are **common** in Glinski:
- Random play: 9% material/fifty-move + 2% stalemate = ~11% formal draws; another ~78% are 400-ply timeouts that would mostly become draws under a stricter shuffle detector.
- Minimax sampling: virtually all games shuffle to the move limit.

WDL value head is **mandatory** (chunk 3 ✓). MLH is **mandatory** (chunk 3 ✓), not optional. Without MLH the policy will shuffle in plus-equal positions and the trainer will see degenerate target distributions. The plan's "draw rate explosion" failure-mode detector (notes/10 §3) should be wired with a generous threshold (Glinski may sit at 30–50% draws even when healthy).

### 9. Symmetries — central inversion, not horizontal mirror

**The only symmetry that survives is `(q, r) → (-q, -r)` (central inversion / 180° rotation around the board center).** The naive "horizontal mirror across the q=0 axis" `(q, r) → (-q-r, r)` preserves `is_valid()` but does NOT preserve the set of pawn-promotion pairs because Glinski's promotion edges aren't symmetric under that transform.

Central inversion works because:
1. It's an involution: `inv(inv(c)) == c`.
2. It maps white pawn directions to black pawn directions (white's "+r" advance becomes black's "-r" advance).
3. It maps white promotion cells to black promotion cells, so promo-pair entries in the move table close cleanly under the transform.
4. `max(|q|, |r|, |q+r|)` is trivially preserved.

This is exhaustively verified in `engine/src/serialization.rs::test_mirror_symmetry_exhaustive`. The Rust mirror table is exposed to Python as `hexchess.mirror_indices_array()` and consumed by `training/data_v2.mirror_batch` (chunk B of the Python TODO pass).

In CHW board encoding (`col = q+5, row = r+5`), central inversion is **flip both axes**: `out[c, r, col] = in[c, 10-r, 10-col]`.

Free 2× data multiplier — applied with `p=0.5` per loaded file in the v2 trainer loader. No 6× rotational augmentation is possible because pawn direction kills all rotations except the 180° one.

### 10. Bootstrap from minimax

Glinski hex chess has a viable handcrafted minimax baseline (material + mobility + pawn
structure + king safety). Use it for imitation bootstrap (a day or so of compute) before
switching to pure self-play. This dodges the "everyone moves randomly" early phase.

### 11. Anchor opponents for Elo

The hex-chess project uses minimax at multiple depths as anchors. Also include:
- Random
- Material-only depth-1
- Material+mobility depth-3
- Material+mobility depth-6
- A frozen first NN snapshot

These give you a stable rating ladder that doesn't drift.

### 12. Threefold and 50-move rule

Detect both in the engine and treat as immediate draws in self-play. Encode the halfmove clock
and repetition count in input planes so the network can reason about them.

## A reasonable starting config (summarized)

| Parameter | Value |
|---|---|
| Network | 8 blocks × 144 filters (~6.3M params), SE, global pool bias |
| Heads | Policy (4206 outputs), WDL value, MLH, STV, aux opponent-reply |
| MCTS | 800 sims (PCR: full=800/fast=160, p=0.25) |
| `c_puct` | 2.5 / `c_puct_root=3.5` |
| Dirichlet `α` | 0.25, `ε` = 0.25, shaped (50% broad + 50% top-10 peaked) |
| FPU | reduction 0.0 (training), 0.2 (eval) |
| Temperature | smooth decay τ_max=1.0 → τ_min=0.1, halflife=20, hard-greedy at ply 60 |
| Replay window | sublinear, c=25k, α=0.75, β=0.4 |
| Sample reuse | 4 |
| LR | 1e-3 constant + 2000-step linear warmup, SGD momentum 0.9, wd=3e-5 |
| Batch size | 256 |
| SWA | EMA over 4 snapshots, every 250k samples |
| Gating | first 5 promotions (≥50% over 200 games, 3-failure escape) |
| Promotion | every 500k samples |
| Mirror augmentation | central inversion, p=0.5 |

## What to read next from the corpus

1. KataGo paper end-to-end ([arXiv:1902.10565](https://arxiv.org/abs/1902.10565)).
2. KataGo follow-on methods doc
   ([`KataGoMethods.md`](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)).
3. Lc0 WDL post ([blog](https://lczero.org/blog/2020/04/wdl-head/)).
4. Lc0 project history ([wiki](https://lczero.org/dev/wiki/project-history/)).
5. Oracle "Lessons from AlphaZero" series on Medium (Aditya Prasad / Anthony Young).
