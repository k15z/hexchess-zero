# Architecture Experiment Results

## Setup

- **Data**: 1.04M imitation learning positions (minimax depth-3 targets), 140 .npz files
- **Split**: 90% train (~930K positions), 10% val (~125K positions) by filename hash
- **Training**: 50K steps, batch 256, Adam (lr=1e-3, wd=1e-4)
- **Device**: Apple MPS (shared across parallel runs)
- **Eval**: Val policy CE loss, val value (WDL) CE loss, top-1 policy accuracy (50 val batches)
- **Date**: 2026-03-31

## Tier 1 Results

All variants use the same 6-block 128-filter ResNet body unless noted.

| Rank | Variant | Val Policy | Val Value | Val Top-1 | Params | Time |
|---|---|---|---|---|---|---|
| **#1** | **se_blocks** | **3.5241** | 0.7485 | 11.8% | 2,924,595 | 2.8hr |
| #2 | kitchen_sink | 3.5587 | 0.7752 | 11.9% | 2,901,233 | 2.9hr |
| #3 | baseline | 3.5735 | 0.7661 | 11.6% | 2,849,139 | 2.4hr |
| #4 | validity_masking | 3.5932 | 0.7429 | 11.3% | 2,849,139 | 2.7hr |

### Variant descriptions

- **baseline**: Current model. 6 residual blocks, 128 filters, conv policy head (2-ch conv -> flatten -> FC 242->4206), shallow value head (1-ch conv -> flatten -> FC 121->256->3).
- **se_blocks**: Baseline + Squeeze-and-Excitation in every residual block. Leela-style scale+bias variant (128->32->256, split into scale/bias). +2.7% params.
- **validity_masking**: Baseline + binary mask (91 valid hex cells) applied after every convolution to zero out the 30 invalid cells in the 11x11 grid. Zero extra params.
- **kitchen_sink**: SE blocks + validity masking + improved value head (32-ch conv -> global avg pool -> FC 32->128->3 instead of 1-ch conv -> flatten -> FC 121->256->3).

### Key findings

1. **SE blocks give the largest single improvement.** Best policy loss (3.5241 vs 3.5735 baseline, -1.4%) with negligible parameter increase. The channel-attention mechanism helps the network condition on global board context for move prediction.

2. **Validity masking helps value, not policy.** Best value loss (0.7429) but worst policy loss. Zeroing invalid cells gives the value head a cleaner signal for position evaluation, but doesn't help move prediction — possibly because the policy head's FC layer can already learn to ignore invalid-cell features.

3. **Kitchen sink converges slowly but finishes strong.** Lagged badly in early training (6.4% top-1 at step 2K vs 9.0% for SE) but caught up to tie for best top-1 (11.9%) by completion. The three combined modifications interact in ways that require more training to resolve.

4. **Top-1 accuracy is tightly clustered** (11.3-11.9%). The real differentiation is in loss quality — better calibrated probability distributions over moves, not just getting the argmax right.

5. **Value loss and policy loss don't correlate.** The best policy variant (SE) has middling value loss; the best value variant (masking) has the worst policy loss. This motivates testing architectural changes that target each head independently.

## Tier 2 Results

| Rank | Variant | Val Policy | Val Value | Val Top-1 | Params | Time |
|---|---|---|---|---|---|---|
| **#1** | **tf_hybrid** | **3.4525** | **0.7330** | **14.1%** | 20,236,006 | 3.2hr |
| **#2** | **global_pool_se** | 3.5062 | 0.7507 | 13.8% | 3,023,283 | 2.0hr |
| #3 | kitchen_sink_attn | 3.7266 | 0.7423 | 9.6% | 1,895,302 | 3.4hr |

### Variant descriptions

- **tf_hybrid**: 6-block 128-filter conv body + 2 transformer encoder layers (4 heads, 512-dim FFN) operating on the 91 valid hex cells as tokens. Learned positional embeddings. Policy via per-cell projection + FC. Value via mean-pooled tokens + MLP.
- **global_pool_se**: SE in every block + KataGo-style global pooling in blocks 1 and 4. Global pooling computes per-channel mean/scaled-mean/max, projects to a bias vector added to the main conv path.
- **kitchen_sink_attn**: SE + validity masking + improved value head + attention policy head (Q/K dot-product per move instead of FC). Uses a placeholder move table (not the real engine mapping).

### Key findings

6. **tf_hybrid is the overall winner.** Best on all three metrics — 14.1% top-1 is a 21% relative improvement over baseline. The 2 transformer layers capture long-range hex relationships (bishop/rook lines spanning the board) that conv layers need many blocks to approximate. The cost: 7x more parameters and 2x training time.

7. **global_pool_se is the efficiency champion.** 13.8% top-1 with only 3.0M params — nearly matches tf_hybrid at 1/7th the parameter count. The combination of SE (channel attention) + global pooling (explicit board-wide context) is the best bang-for-buck modification found.

8. **The attention policy head underperformed.** 9.6% top-1 is worse than plain baseline (11.6%). The placeholder move table (sequential cell pairs instead of real engine move mapping) likely hurt — the Q/K structure can't learn meaningful from/to relationships when the index mapping is arbitrary. Worth revisiting with the real `hexchess.move_to_index()` table.

## Combined Rankings (All 7 Experiments)

| Rank | Variant | Val Policy | Val Value | Val Top-1 | Params |
|---|---|---|---|---|---|
| **#1** | **tf_hybrid** | **3.4525** | **0.7330** | **14.1%** | 20.2M |
| **#2** | **global_pool_se** | 3.5062 | 0.7507 | **13.8%** | 3.0M |
| #3 | se_blocks | 3.5241 | 0.7485 | 11.8% | 2.9M |
| #4 | kitchen_sink | 3.5587 | 0.7752 | 11.9% | 2.9M |
| #5 | baseline | 3.5735 | 0.7661 | 11.6% | 2.8M |
| #6 | validity_masking | 3.5932 | 0.7429 | 11.3% | 2.8M |
| #7 | kitchen_sink_attn | 3.7266 | 0.7423 | 9.6% | 1.9M |

## Decision

**Adopted `global_pool_se` as the production architecture** (implemented in `training/model.py`).

- Best cost/quality tradeoff: 13.8% top-1 with only 3.0M params
- Small enough for fast MCTS inference (critical for self-play throughput)
- Fastest training time of any non-baseline variant (2.0hr vs 2.4hr baseline)
- tf_hybrid is strictly better on quality but 7x params makes it impractical for self-play

### Future directions

- **Revisit attention policy head with real move table.** The placeholder mapping sabotaged E6. With the engine's actual `move_to_index()` mapping, the Q/K structure should learn meaningful piece-movement patterns.
- **tf_hybrid as analysis model.** Consider for offline evaluation where inference speed matters less.
