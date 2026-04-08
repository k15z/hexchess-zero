# 12 — Adapting AlphaZero to a small hex-board variant

Notes specifically for porting AlphaZero techniques to a small-board chess-like game (e.g.
Glinski hexagonal chess: 91 hex cells, branching ~30–50, average game length ~80–120 plies,
draws possible).

## What's different from chess and Go

| Aspect | Go 19x19 | Chess | Glinski hex chess |
|---|---|---|---|
| Board cells | 361 | 64 | 91 |
| Branching factor | ~250 | ~35 | ~30–50 |
| Average game length (plies) | ~250 | ~80 | ~80–120 |
| Draws | very rare (komi) | ~50% top-level | possible, frequency unknown |
| Repetition rules | superko | threefold | threefold (assumed) |
| Move encoding size | ~362 | ~4672 | ~4000 |
| Symmetries | 8 (D4) | none | 6 (hex rotations) — do you exploit them? |

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
- Glinski branching ~30–50 → `α ≈ 0.2–0.3`.
- Compare: chess uses 0.3, Go uses 0.03.

Probably start at `α = 0.25`.

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

Glinski games are short (~80–120 plies) compared to Go. This means:
- Self-play throughput is high — easier to bootstrap.
- Each game contributes proportionally more terminal value signal.
- Resignation is less critical (games end fast anyway), but still useful at ~10–20% of games.

### 8. Draws and repetition

If draws are common, **WDL value head is mandatory**. Add a moves-left head to encourage
making progress; otherwise the network may shuffle in won positions (the standard Lc0 / KataGo
draw-rate-explosion failure mode).

### 9. Symmetries

Glinski hex chess has some hex-board symmetries (rotations, possibly reflections), but pawn
direction breaks most of them. If a useful symmetry exists, you can **augment training data**
by feeding both the original and its symmetric image with the corresponding policy mirroring.
Free 2x or 6x data multiplier.

For Go, KataGo augments with all 8 D4 symmetries during training. Worth checking if any
symmetry survives in your specific variant.

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
| Network | 10 blocks × 192 filters, SE, global pool bias |
| Heads | Policy (~4000 outputs), WDL value, optional MLH |
| MCTS | 800 sims (PCR: full=800/fast=160, p=0.25) |
| `c_puct` | 2.5 |
| Dirichlet `α` | 0.25, `ε` = 0.25 |
| FPU | reduction 0 (training), 0.3 (eval) |
| Temperature | τ=1 for first 15 plies, decay to 0 by ply 30 |
| Replay window | sublinear, c=25k, α=0.75, β=0.4 |
| Sample reuse | 4 |
| LR | 1e-3 constant, drop to 1e-4 at end |
| Batch size | 512 |
| SWA | EMA(0.75), 4 snapshots |
| Gating | off (use Elo service for safety) |
| Promotion | every 500k samples |
| Selfplay:trainer compute | ~8:1 |

## What to read next from the corpus

1. KataGo paper end-to-end ([arXiv:1902.10565](https://arxiv.org/abs/1902.10565)).
2. KataGo follow-on methods doc
   ([`KataGoMethods.md`](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)).
3. Lc0 WDL post ([blog](https://lczero.org/blog/2020/04/wdl-head/)).
4. Lc0 project history ([wiki](https://lczero.org/dev/wiki/project-history/)).
5. Oracle "Lessons from AlphaZero" series on Medium (Aditya Prasad / Anthony Young).
