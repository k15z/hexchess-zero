# 12 — Adapting AlphaZero to a small hex-board variant

Notes for porting AlphaZero techniques to Glinski hexagonal chess (91 hex cells).
Numbers below are empirically measured against the actual engine where noted.

## What's different from chess and Go

| Aspect | Go 19x19 | Chess | Glinski hex chess |
|---|---|---|---|
| Board cells | 361 | 64 | 91 |
| Branching factor | ~250 | ~35 | **random: mean 40, median 40** / **MCTS-200 self-play: mean 27, median 24** (MCTS visits more constrained positions) |
| Initial-position legal moves | ~360 | 20 | **51** (verified) |
| Average game length (plies) | ~250 | ~80 | **MCTS-200 self-play: mean 244, median 247, p95 = 500-cap** (n=30 games, heuristic eval). Games are LONG, not the originally-guessed 80–120. |
| Draws | very rare (komi) | ~50% top-level | **MCTS-200 self-play: 47% formal draws** (27% fifty-move + 20% material), ~7% timeout. Expect 30–50% even under stronger engine play. |
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

**Empirical numbers from MCTS-200 self-play (heuristic eval, no NN, n=30 games):**
- Mean game length: **244 plies**, median 247, max 500 (timeout cap)
- p5 = 20 plies, p95 = 500 plies
- 47% formal draws (27% fifty-move + 20% material), 47% checkmates, 7% timeout

The original "~80–120 plies" guess was wrong by 2-3×. **Plan for self-play move-limit cap ≥ 500 plies; budget compute accordingly.**

Implications:
- MLH is critical to break shuffles (already in chunk 3). Without MLH, expect even more timeouts/draws.
- Resignation is genuinely useful — at heuristic strength, ~half of games drift into 200+ ply shuffle zones that resign should clip.
- A trained NN should produce sharper games and shorter average length (Lc0/KataGo evidence) but baseline starting from heuristic is what bootstrap will do, so prepare for the long-game regime.
- Per-game move-limit cap should be ≥ 500 plies in self-play config (we currently default to 200 in some places — needs audit).

### 8. Draws and repetition

Draws are **very common** in Glinski:
- MCTS-200 self-play (heuristic eval): **47% formal draws** (27% fifty-move + 20% insufficient material).
- Random play: ~11% formal + dominant timeout-by-shuffle.

WDL value head is **mandatory** (chunk 3 ✓). MLH is **mandatory** (chunk 3 ✓), not optional. Without MLH the policy will shuffle in plus-equal positions and the trainer will see degenerate target distributions. The plan's "draw rate explosion" failure-mode detector (notes/10 §3) should be **calibrated for a 30–50% baseline draw rate** — the default 75% threshold is fine, but anything below 25% might indicate the engine has stopped exploring drawn lines (also a problem).

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
