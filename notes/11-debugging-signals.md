# 11 — Debugging signals: what to log

A short list of metrics that, between them, will let you debug ~80% of training problems.

## Tier 1 — must log every step

### Policy entropy

```
H(p) = − Σ_a p(a) · log p(a)
```

Computed on the network's prediction (not the target). Track both:
- **Mean entropy** across the batch.
- **Entropy histogram** weekly so you can see distributional changes.

Healthy curve: starts near `log(num_legal_moves)` (uniform), decays smoothly, plateaus at some
value strictly above zero (the network should never be 100% confident).

**Warning signs:**
- Entropy crashes to ~0 in the first few hours → policy collapse.
- Entropy stays at the maximum forever → network isn't learning.
- Entropy fluctuates wildly → instability, LR likely too high.

### Value MAE / WDL accuracy

```
Value MAE = mean(|v_pred − z|)        # for scalar
WDL acc = mean(argmax(WDL_pred) == argmax(WDL_target))
```

Healthy: starts high (MAE ≈ 0.5–0.7, WDL acc ≈ 0.33), decays to ~0.2 / ~0.6 over time.

### Loss components

Always log policy loss, value loss, and any aux losses **separately** (not just total). Equal-
weighted total losses obscure which head is failing.

### Learning rate, gradient norm

If you have grad clipping, also log the pre-clip grad norm. Sudden spikes are early warnings of
instability.

## Tier 2 — log per game / per evaluation cycle

### Game length distribution

Histogram of plies-per-game. Healthy:
- Mean: should be close to the natural game length for the game (chess: ~80 plies; Go 19x19:
  ~250; small hex chess: ~80–120).
- Tail: a heavy tail at the move limit (300+ plies) means too many shuffle-loop draws.

### Outcome distribution (W/D/L %)

Healthy distribution depends on the game. For chess at top level, ~50% draws is normal. For Go,
near-zero draws because of komi.

**Warning signs:**
- Draws → 100% over time: see failure mode 3.
- Wins by white >> wins by black (or vice versa): possible symmetry / orientation bug.

### NN vs MCTS agreement

Fraction of moves where `argmax(NN_policy) == argmax(MCTS_visits)` at search time. Healthy:
- Early training: <30% (search disagrees a lot, that's the whole point).
- Mid: ~60–80%.
- Late: ~85–95%.

If this stays at 100% from very early, MCTS isn't doing useful work — `c_puct` is too low or
sims are too low.

If it stays at <50% forever, the network isn't learning from search.

### KL(old_policy || new_policy)

Periodically, evaluate the *previous* model's policy on the *current* model's training batches
and compute `KL(p_old || p_new)`. This is your training-step size in policy space.

- Very small KL → trainer is barely moving (LR too low or replay reuse too high).
- Very large KL → unstable, overfitting or LR too high.

KataGo's policy surprise weighting uses a related quantity.

### Resignation calibration

Removed. Self-play no longer resigns, so this signal is intentionally absent.

## Tier 3 — debugging signals to enable when something's wrong

### Per-channel input statistics

Compute mean and std of every input plane across a batch. Useful for catching:
- A channel that's all zeros (forgot to encode something).
- A channel that's mostly NaN.
- Boundary mask correctly zero on invalid hex cells.

### Move-visit distribution dump

For a few "canonical" positions (opening, midgame, won endgame, lost endgame), dump the full
MCTS visit distribution every N training steps. Watch them evolve. Useful for catching:
- A position where the "obviously best move" never gets visited.
- Visits going to clearly illegal moves (encoding bug).

### NN vs MCTS Q correlation

Plot `NN_value(s)` vs `MCTS_root_Q(s)` across many positions. They should be strongly
correlated (~0.9+ once trained). If they diverge systematically, the network has biases that
search is correcting — investigate.

### Per-version loss and Elo

Store the mean policy/value loss seen by the trainer at the time each model version was
exported. Plot loss vs Elo across versions. Confirms (or disproves) that loss correlates with
strength for your run. They often diverge — see `07-loss-functions.md`.

## What healthy looks like, in one chart

If you only have one dashboard, plot these 6 lines together over wall time:

1. Trainer policy loss (left axis)
2. Trainer value loss (left axis)
3. Policy entropy (right axis, normalized)
4. Anchor Elo vs frozen baseline (right axis)
5. Self-play games per hour (separate axis)
6. Mean game length (separate axis)

Things you should see:
- Losses decreasing smoothly.
- Entropy decreasing smoothly but not crashing.
- Anchor Elo increasing, ideally monotonically.
- Game throughput stable (drops indicate worker issues).
- Game length stable or slowly increasing (decisive games tend to get longer as both sides get
  better).

## Hexchess Zero specific signals

For a small variant game, also worth logging:

- **Distinct openings played per 100 games**: should stay >20. If it drops, your opening
  exploration has collapsed.
- **Distribution of game-end reasons**: checkmate, stalemate, threefold, 50-move, insufficient
  material. Sudden shifts indicate strategy changes.
- **Average legal moves per position**: a sanity check on game progression and movegen
  correctness.
