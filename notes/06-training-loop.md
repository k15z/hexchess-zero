# 06 — The training loop

## Anatomy

```
[Self-play workers]      generate (s, π, z) tuples → write to shared storage
        │
        ▼
[Replay buffer]          sliding window of recent samples
        │
        ▼
[Trainer]                samples mini-batches, takes SGD steps, exports new model
        │
        ▼
[Promotion]              new model becomes current, workers pick it up
        │
        └──── (loop) ────┘

[Eval / Elo service]     in parallel, ranks model versions
```

The whole system is a producer/consumer with a sliding window in the middle. Two failure modes:

- **Too much trainer relative to selfplay** → trainer overfits to a narrow recent window, model
  quality stalls or regresses.
- **Too much selfplay relative to trainer** → wasted compute, data goes stale before being used.

KataGo's [`SelfplayTraining.md`](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md)
recommends spending **4x to 40x more compute on selfplay than on training**. AlphaZero's TPU
ratios were similar — most of the cluster was self-playing.

## The replay window

The single most important hyperparameter you've probably never heard of.

**KataGo's moving window formula** ([Wu 2019, Appendix C](https://arxiv.org/abs/1902.10565)):

```
N_window(N_total) = c · (1 + β · ((N_total / c)^α − 1) / α)
```

with `c = 250,000`, `α = 0.75`, `β = 0.4`. The paper notes this is simply the sublinear curve
`f(n) = n^α` rescaled so that `f(c) = c` and `f'(c) = β`.

Worked examples (computed from the formula above):
- At `N_total = 250k`: window = `250k` (by construction).
- At `N_total = 1M`: window ≈ `494k`.
- At `N_total = 10M`: window ≈ `2.2M`.
- At `N_total = 100M`: window ≈ `12M`.
- At `N_total ≈ 241M` (end of KataGo's main run): window ≈ `22M`, matching the paper's quoted
  "~22 million by the end of the main run."

The window grows **sublinearly** so it stays well-mixed even when production has been running
for weeks, starting at 250k samples.

Why sublinear matters: linear growth means very old (weak-policy) games dominate the buffer
forever; constant size means every recent burst of self-play immediately washes out the buffer
and you train on a noisy moving target. Sublinear is the sweet spot.

**For a small game**, scale `c` down — try `c = 25k–50k` initially.

## Sample reuse rate

KataGo uses `-max-train-bucket-per-new-data 4`, meaning **at most 4 SGD samples per generated
self-play sample**. They describe this as "conservative" — values up to ~8 are fine, but >10
risks overfitting.

For comparison:
- AlphaZero: each position seen by the trainer approximately once (effectively reuse=1).
- Lc0: typically reuse ≈ 1–2.
- KataGo: reuse ≈ 2–4.

Higher reuse means less self-play wasted, but more risk of overfitting to the window.

## Batch sizes and learning rate

| System | Batch size | LR | Notes |
|---|---|---|---|
| AlphaZero | 4096 | 0.2 → 0.02 → 0.002 → 0.0002 | stepwise |
| KataGo | 256 | 6e-5 per-sample | constant most of run |
| Lc0 | 4096 (typical) | varies, ~0.1 cosine decay | |

KataGo's "per-sample" LR gives a per-batch rate of `256 × 6e-5 ≈ 0.0154`. For the first 5M
samples of each net size, this was lowered by a factor of 3 (i.e. `2e-5` per sample) to reduce
early-training instability (Wu 2019, Appendix C).

LR drops were rare — KataGo dropped from `6e-5` to `6e-6` per sample at ~day 17.5 of the 19-day
main run to squeeze out final strength.

**Practical recipe:**
1. Start with a small constant LR (~0.001 effective).
2. Watch loss curves; if they're stable for a long time, you can drop LR by 10x at the end of the
   run for a final boost.
3. Don't bother with cosine annealing — it's mostly a vision-training trick that doesn't pay
   off for self-play training, where the data distribution itself is non-stationary.

## Weight averaging (SWA)

KataGo's SWA scheme ([Wu 2019, Appendix C](https://arxiv.org/abs/1902.10565)):

- Save a weight snapshot every ~250k training samples.
- Every 4 snapshots, produce a new candidate net by **exponential moving average** of the
  snapshots with decay `0.75` (four-snapshot lookback).
- The EMA net is what gets promoted, not the raw last-step weights.

This costs nothing at inference time (it's just averaging) and consistently helps stability and
final strength. **Use SWA.**

## Gating: yes or no?

**AlphaGo Zero** required new candidate nets to beat the current net 55% in a 400-game match
before promoting. AlphaZero **dropped this** in favor of continuous promotion.

KataGo's verdict ([SelfplayTraining](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md)):
gating is *optional*. Disabling it saves compute. Enabling it can be useful for debugging or
for very long-running production systems where occasional regressions are unacceptable.

**Recommendation:**
- During development / first few weeks: gate. Catches bugs that turn the network into garbage.
- In steady-state production: don't gate. The continuous improvement is faster.
- Always have a separate Elo service that catches regressions (see `09-evaluation-and-elo.md`).

## Promotion / model sync

Self-play workers should poll for new models. KataGo and Lc0 both promote a new net every few
hundred thousand samples — frequent enough that workers always have a recent model, infrequent
enough that workers don't spend more time downloading than playing.

Reasonable default: **promote every ~500k–1M training samples**, or every ~30 minutes of trainer
wall time, whichever comes first.

## Bootstrap / cold start

Random initialization → random self-play → can take a long time to escape "everyone moves
randomly" attractor. Options:

1. **Just wait** (AlphaZero): with enough compute, escapes within a few hours.
2. **Imitation bootstrap**: pretrain on minimax / handcrafted-evaluator self-play games, then
   switch to NN-guided MCTS once the policy is non-random. Faster, slightly compromises the
   "from zero" purity.
3. **Lower MCTS sims early**: cheap & weak search early, scale up as net improves.

For a small game with a known minimax baseline, option 2 is worth ~1 day of compute and is
strongly recommended.

## Resignation calibration

Removed from the training loop. The calibration sample showed too many false
positives, so self-play now runs to natural termination and does not maintain
resignation telemetry.

## Operational summary

A reasonable initial config for a small board game:

| Knob | Value |
|---|---|
| Net | 8 blocks × 128 filters + SE |
| MCTS sims (training) | 800 (with PCR: full=800 p=0.25, fast=160 p=0.75) |
| Replay window | sublinear, c=25k, α=0.75, β=0.4 |
| Batch size | 512 |
| Sample reuse | 4 |
| LR | 0.001 constant |
| Optimizer | SGD momentum 0.9, weight decay 1e-4 |
| SWA | EMA(0.75), 4-snapshot lookback |
| Gating | off (with separate Elo service) |
| Promote every | 500k samples |
| Selfplay:trainer compute | 8:1 |
