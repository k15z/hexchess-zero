# 08 — Data efficiency: how much self-play do you actually need?

## Reference points

| System | Game | Compute | Self-play games | Time to surpass best handcrafted |
|---|---|---|---|---|
| AlphaGo Zero | Go 19x19 | 4 TPUs (play) + 64 GPUs train-equiv, 3 days (20b) | ~4.9M (reported) | ~36 h to beat AlphaGo Lee |
| AlphaZero (Go) | Go 19x19 | 5000 1st-gen TPUs selfplay + 64 2nd-gen TPUs train, 34 h | 21M (Table S3) | ~8 h to beat AlphaGo Lee |
| AlphaZero (chess) | Chess | same, 9 h | 44M (Table S3) | ~4 h (300k steps) to surpass Stockfish 8 |
| AlphaZero (shogi) | Shogi | same, 12 h | 24M (Table S3) | <2 h (110k steps) to surpass Elmo |
| ELF-OpenGo | Go 19x19 | ~2000 V100s, 13–14 days (Wu 2019 §1) | not precisely reported | — |
| **KataGo main run** | Go 19x19 | **~27 V100s avg, 19 days (≈1.4 GPU-yr)** | **4.2M** | surpasses ELF final model by end of run |
| Lc0 T40 | Chess | community-distributed (hundreds of GPUs over months) | hundreds of millions | months |

## The KataGo "50x" claim

Wu (2019) claims KataGo reaches AlphaZero/ELF-level Go strength in **~50x less compute**. The
breakdown:
- ~9x from the techniques in the paper (PCR, forced playouts, global pooling, aux targets, etc.)
- Additional gains from architecture (SE blocks), longer training, larger nets, etc.

The takeaway: **the original AlphaZero recipe is enormously wasteful**. With modern techniques,
2026-era hardware (a single 8x H100 node) can probably reproduce AlphaZero-strength chess in
days, not the original 5000-TPU days.

## What drives data efficiency

In rough order of importance:

1. **Search-policy/value tension management** (PCR): the single biggest free win. ~37%.
2. **Network architecture quality** (SE, global pooling, depth/width balance): ~60% combined.
3. **Auxiliary supervision** (aux policy, ownership/score, MLH, short-term value): ~30–50%
   combined depending on game.
4. **Replay window sizing** (sublinear): subtle but important — wrong window size can wreck a
   run that would otherwise have worked.
5. **WDL value head** (vs scalar): hard to attribute exactly but Lc0 reports it as a "major"
   improvement.
6. **Tuned MCTS exploration** (cpuct, FPU, Dirichlet α): another 10–30%.

Resignation was removed from this project after an S3 audit showed that the
calibration sample produced too many false positives to trust the value labels.

Multiplicatively, these stack to easily an order of magnitude on top of vanilla AZ.

## Bootstrap from a baseline

If you have a strong handcrafted evaluator (minimax, material+mobility, etc.), you can save
significant time by:

1. Generating ~100k–1M positions of (minimax-search-best-move, position) pairs.
2. Training the network in **supervised mode** (cross-entropy on best move, MSE on minimax
   value) for a few epochs.
3. Switching to AlphaZero-style self-play once the policy is non-uniform.

This skips the long "everyone is random" early phase. KataGo and Lc0 don't do this (they're
pure-zero by design), but for a new project where you're not going for the "from zero" claim,
imitation bootstrap is worth a day of compute.

## When can you expect to surpass a strong baseline?

For a small game (~91 cells, branching ~30–50, average game length ~80–120 plies), with a
modern stack (PCR + SE + global pooling + WDL + sublinear window):

| Milestone | Approximate self-play volume |
|---|---|
| Beats random | ~1k games |
| Beats material-only minimax depth-1 | ~10k games |
| Beats material+mobility minimax depth-3 | ~50–200k games |
| Beats material+mobility minimax depth-6 | ~500k–2M games |
| Beats handcrafted "expert" | ~5–20M games |
| Plateaus | ~50–200M games |

These are order-of-magnitude estimates extrapolating from KataGo's Go progression and the
Oracle Connect Four / Chain Reaction reproductions. For your specific game and search budget,
**actually measure** by anchoring against your minimax baseline at multiple depths.

## Sample efficiency vs game count

Counts in "games" hide a lot. The right unit is **positions seen by the trainer**:

- AlphaZero chess: ~44M games × ~80 plies/game ≈ 3.5B positions generated.
- KataGo main run: 4.2M games → 241M training samples (the paper reports the sample count
  directly; the ~4:1 "plies-per-sample" ratio is consistent with playout cap randomization only
  recording the ~25% full-search positions but is not explicitly stated as such).
- A small game with PCR: maybe ~100M positions to reach plateau.

The trainer **does not see all positions** — only the "full search" positions under PCR.

## Diminishing returns

Empirically, Elo gain is roughly logarithmic in compute beyond some point. KataGo's self-play
Elo curve shows steep gains in the first 5 days, then a long slow climb. AlphaZero's chess Elo
curve shows similar shape: most of the gain in the first ~4 hours.

**Practical implication:** your run will probably stop being interesting before it stops gaining
Elo. Set a budget, hit it, and move on to architecture/algorithm improvements rather than just
running longer.
