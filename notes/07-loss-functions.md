# 07 — Loss functions

## The standard AlphaZero loss

```
L = L_value + L_policy + c · L_reg

L_value  = (z − v)²                         # MSE on scalar
L_policy = − Σ_a π(a) · log p(a)            # cross-entropy
L_reg    = ||θ||²                            # L2 weight decay
```

Where:
- `z ∈ {−1, 0, +1}` is the actual game outcome from side-to-move's perspective.
- `v` is the network's value prediction.
- `π` is the MCTS visit-count distribution at the root (the search-improved policy).
- `p` is the network's policy.
- `c ≈ 1e-4` for AlphaZero; KataGo uses `c_L2 = 3e-5`.

The two head losses are typically combined with **equal weight**. AlphaZero uses 1.0 vs 1.0.
KataGo (Wu 2019 §2) uses a value-loss scaling coefficient `c_g = 1.5` — i.e. the game-outcome
value loss is upweighted 1.5x relative to the policy cross-entropy, not the other way around.

## WDL value loss (preferred)

For any game with draws, replace MSE with cross-entropy on a 3-class softmax:

```
L_value = − [w_z · log P(W) + d_z · log P(D) + l_z · log P(L)]
```

where `(w_z, d_z, l_z)` is one-hot for the actual outcome.

Equivalent in expectation to MSE on the scalar `P(W) − P(L)` for win/loss-only games, but
strictly more informative when draws exist.

Reasons WDL > MSE-scalar:
1. **Calibration** — softmax outputs are probabilities, you can read off P(draw) directly.
2. **No saturation pathology** — `tanh(v)` saturates near ±1, gradients vanish in clearly
   won/lost positions. Softmax cross-entropy doesn't saturate the same way.
3. **Downstream uses** — contempt, drawishness reasoning, search aggregation all want a
   distribution, not a scalar.

See [Lc0 WDL blog](https://lczero.org/blog/2020/04/wdl-head/) for full discussion.

## Policy loss details

Cross-entropy against the **MCTS visit distribution**, not against the actual move played:

```
π(a) = N(s,a) / Σ_b N(s,b)
```

The visit distribution is the policy improvement target — the whole reason AlphaZero works is
that MCTS turns the network's policy into a slightly better one.

**Mask illegal moves** by setting their logits to `−∞` before softmax in the network output. The
target `π` already has zero mass on illegal moves, so cross-entropy naturally ignores them, but
you must mask in the predicted distribution to prevent gradients from flowing into illegal
actions.

## Auxiliary losses

If you have aux heads, just add their losses with small weights:

```
L_total = L_value + L_policy + λ_mlh · L_mlh + λ_aux · L_aux + ...
```

Typical weights:
- Moves-left head (Lc0): `λ ≈ 0.5–1.0` of value loss.
- Ownership (KataGo): `λ ≈ 1.5` (per-cell, summed).
- Auxiliary policy (next-opponent-move): `λ ≈ 0.15` of main policy.
- Short-term value heads: `λ ≈ 0.1–0.3` each.

The aux losses regularize the trunk; the absolute weights don't matter much as long as no single
head dominates.

## Healthy loss curves

What you should see, on a typical small-game training run after the bootstrap phase:

| Metric | Initial | Steady-state | Late-stage |
|---|---|---|---|
| Policy loss | 5–7 (≈ log num_moves) | 1.5–3.0 | 1.0–2.0 |
| Value loss (MSE) | 0.6–0.8 (close to var(z)) | 0.3–0.5 | 0.15–0.30 |
| Value loss (WDL CE) | 1.0–1.1 (≈ log 3) | 0.6–0.9 | 0.4–0.6 |
| Policy entropy | log(num_legal) | gradually decreasing | low but nonzero |

The policy loss starts near `log(num_legal_moves)` because the random network outputs uniform.
For Glinski hex chess with ~30–50 legal moves, that's ~3.4–3.9.

The value loss for WDL starts near `log(3) ≈ 1.099` because the random network outputs uniform
over W/D/L.

**Both losses should decrease monotonically** in expectation (with noise) for a healthy run.
Sudden jumps usually indicate a bug or a destabilizing parameter change.

## Common loss-related bugs

- **Loss starts at the wrong value**: your loss formula or label encoding is wrong. Double-check
  with a single forward pass on a known position.
- **Loss decreases then explodes**: LR too high, no gradient clipping, or numerical instability
  in WDL softmax (check for NaN inputs).
- **Loss drops fast then plateaus very early**: model is too small, or replay buffer is too
  small and overfitting.
- **Policy loss decreases but value loss doesn't**: value head is broken, or you're feeding it
  uninformative game outcomes (e.g. all draws).
- **Value loss decreases but policy loss doesn't**: policy head is broken, or your move
  encoding is buggy and the policy target rarely matches what the network can output.

See `10-common-failure-modes.md` for the full bug runbook.

## Loss magnitudes during KataGo training

Approximate from KataGo training logs (Go 19x19, Wu 2019):

- Policy loss: starts ~6.0, plateaus around 1.5–2.0 after a few days.
- Value MSE: starts ~0.7, plateaus around 0.3 by mid-run.
- Aux ownership loss: starts ~0.4, plateaus around 0.2.
- Total combined loss: starts ~9, ends ~3.5.

For a small game, expect smaller absolute values for the policy loss (since `log(num_moves)` is
smaller) but similar shapes.

## Loss vs. Elo

A critical lesson: **lower loss does NOT always mean higher Elo**. The loss is computed against
*current* MCTS visits — but if the network improves, MCTS produces different visits, and the loss
target shifts. You can have flat loss curves with rising Elo, or even slightly increasing loss
with rising Elo (the network has gotten better, so the targets are harder).

**Always evaluate with an external Elo measurement, not loss.** See `09-evaluation-and-elo.md`.
