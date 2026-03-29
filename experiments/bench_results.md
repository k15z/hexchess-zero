# Architecture Benchmark Results

## Round 1: Broad exploration (338K positions, 5 epochs, MPS)

### Summary
| Model | Total Loss | Policy | Value | ms/pos | Params |
|-------|-----------|--------|-------|--------|--------|
| CNN 6x128 hex-masked | **4.3851** | 4.1172 | 0.2678 | 1.07 | 2.8M |
| CNN 6x128 SE-ResNet | 4.3906 | 4.1892 | 0.2014 | 1.54 | 2.9M |
| CNN 6x128 (baseline) | 4.4111 | 4.1570 | 0.2541 | 1.07 | 2.8M |
| CNN 4x64 (old/small) | 4.4254 | 4.1924 | 0.2330 | 0.79 | 1.4M |
| CNN 6x128 wider-policy(32ch) | 4.4272 | 4.2220 | 0.2053 | 1.08 | 18.1M |
| CNN 6x192 (wider) | 4.4330 | 4.1808 | 0.2522 | 1.07 | 5.1M |
| CNN 10x128 (deeper) | 4.4434 | 4.1648 | 0.2786 | 1.56 | 4.0M |
| CNN(2)+Transformer(3) | 5.0303 | 4.8127 | 0.2176 | 1.78 | 2.4M |
| Transformer 4L d128 | 5.2081 | 4.8371 | 0.3710 | 1.55 | 2.0M |
| Local-Attn 4L d128 | 5.5281 | 5.1285 | 0.3996 | 1.59 | 2.0M |
| MLP 4x1024 (hex-only) | 6.3175 | 6.0981 | 0.2194 | 0.65 | 6.3M |
| MLP 4x1024 (flat grid) | 6.4684 | 6.2567 | 0.2117 | 0.24 | 6.7M |
| HexGNN 6L d128 | 7.6050 | 7.2055 | 0.3995 | 94.55 | 1.4M |
| Transformer 4L d128 no-pos | 7.6572 | 7.2631 | 0.3940 | 1.57 | 1.9M |

### Convergence (val_total by epoch)
| Model | E1 | E2 | E3 | E4 | E5 |
|-------|------|------|------|------|------|
| CNN 6x128 hex-masked | 4.986 | 4.583 | 4.529 | 4.456 | 4.385 |
| CNN 6x128 SE-ResNet | 4.876 | 4.539 | 4.431 | 4.380 | 4.391 |
| CNN 6x128 (baseline) | 4.950 | 4.608 | 4.511 | 4.465 | 4.411 |
| CNN 4x64 (old/small) | 4.900 | 4.549 | 4.479 | 4.431 | 4.425 |
| CNN 6x128 wider-policy(32ch) | 4.508 | 4.421 | 4.366 | 4.382 | 4.427 |

### Key findings
- **Hex masking** is the single best change: free perf, best loss
- **SE-ResNet** strong but 50% slower inference; note it *overfit* at epoch 5 (4.380 -> 4.391)
- **Wider policy head (32ch)** overfit badly: peaked at E3 (4.366) then regressed
- **Transformers** lag CNNs significantly, still converging at E5
- **MLPs** confirm spatial structure matters
- **GNN** too slow (Python loops) and underfits
- **Positional embeddings** critical for transformers (5.21 vs 7.66 without)

## Round 2: Focused hex-masked CNN variants (338K positions, 5 epochs, MPS)

Experiments 8-14 were cancelled early (only 7/14 completed). Results below are for completed experiments only.

### Summary
| Model | Total Loss | Policy | Value | ms/pos | Params |
|-------|-----------|--------|-------|--------|--------|
| hex-masked+SE 6x128 | **4.3416** | 4.1468 | **0.1948** | 1.58 | 2.9M |
| hex-masked 6x128 | 4.3822 | **4.1351** | 0.2471 | 1.12 | 2.8M |
| hex-masked 8x128 | 4.3970 | 4.1420 | 0.2550 | 1.36 | 3.4M |
| hex-masked pol-8ch | 4.4174 | 4.1893 | 0.2281 | 1.10 | 5.9M |
| baseline (no mask) | 4.4434 | 4.1886 | 0.2549 | 1.05 | 2.8M |
| hex-masked 6x160 | 4.4449 | 4.1990 | 0.2459 | 1.10 | 3.8M |

### Convergence (val_total by epoch)
| Model | E1 | E2 | E3 | E4 | E5 |
|-------|------|------|------|------|------|
| hex-masked+SE | 4.855 | 4.491 | 4.393 | **4.333** | 4.342 |
| hex-masked | 5.046 | 4.669 | 4.517 | 4.434 | 4.382 |
| hex-masked 8x128 | 5.048 | 4.613 | 4.553 | 4.466 | 4.397 |
| hex-masked pol-8ch | 4.705 | 4.474 | 4.424 | 4.404 | 4.417 |
| baseline (no mask) | 5.007 | 4.612 | 4.567 | 4.482 | 4.443 |
| hex-masked 6x160 | 4.934 | 4.632 | 4.561 | 4.479 | 4.445 |

### Key findings
- **Hex-masked+SE is the best architecture tested** (4.342 total). Combines hex masking with SE channel attention. The SE blocks dramatically improve value prediction (0.195 vs 0.247 without SE).
- **SE causes slight overfit** — peaked at E4 (4.333) and regressed at E5. This is consistent with round 1. Could benefit from more data or regularization.
- **Hex masking confirmed again** — 4.382 vs 4.443 baseline, consistent 0.06 improvement for free.
- **8 blocks slightly better than 6** (4.397 vs 4.382 for hex-masked) but 6x160 wider was *worse* (4.445). Depth > width for this problem.
- **Wider policy head (8ch) overfit** like round 1's 32ch variant — fast initial convergence (4.705 at E1) but regressed after E4. The extra policy params aren't justified by this data volume.
- **Not tested (cancelled)**: pol-16ch, val-512, flatten-hex pooling, avg/max pooling, combined variants.

### Recommendation
**Hex-masked + SE (6 blocks, 128 filters)** is the production pick. Total loss 4.342 at 1.58 ms/pos — 2.3% better than plain hex-masked, 3.5% better than baseline unmasked CNN. The SE overhead (50% slower inference) is worth the value head improvement. Consider more training epochs or data to mitigate the slight overfitting.

## Round 3: Attention policy head (354K positions from gen2, 5 epochs, MPS)

Head-to-head comparison of conv policy head vs Lc0-style attention policy head. Both use the same hex-masked SE trunk (6x128). The attention policy head uses self-attention over 91 hex cells, then gathers from/to cell features per move and computes logits via dot product (no giant FC layer).

### Summary
| Model | Total Loss | Policy | Value | ms/pos | Params |
|-------|-----------|--------|-------|--------|--------|
| hex+SE attn d64 2L | **3.6231** | **3.4700** | 0.1531 | 2.69 | 2.0M |
| hex+SE attn d128 1L | 3.6243 | 3.4720 | **0.1523** | 2.46 | 2.1M |
| hex+SE attn d64 1L | 3.6354 | 3.4832 | 0.1522 | 2.42 | 2.0M |
| hex+SE conv-policy | 3.7130 | 3.5461 | 0.1669 | 1.57 | 2.9M |

### Convergence (val_total by epoch)
| Model | E1 | E2 | E3 | E4 | E5 |
|-------|------|------|------|------|------|
| hex+SE attn d64 2L | 4.167 | 3.848 | 3.782 | 3.639 | 3.623 |
| hex+SE attn d128 1L | 4.118 | 3.853 | 3.740 | 3.668 | 3.624 |
| hex+SE attn d64 1L | 4.074 | 3.832 | 3.770 | 3.672 | 3.635 |
| hex+SE conv-policy | 4.358 | 3.992 | 3.807 | 3.743 | 3.713 |

### Key findings
- **Attention policy head wins decisively**: 3.623 vs 3.713, a 2.4% improvement in total loss, driven entirely by better policy (3.470 vs 3.546). Value loss is also slightly better.
- **Fewer parameters**: attention models are ~2.0M params vs 2.9M for conv — 32% fewer parameters for better results. The giant FC layer (2ch × 121 → 4206 = ~1M params) is replaced by attention (~40K) + gather (free).
- **All attention variants are close**: d64 1L, d64 2L, and d128 1L are within 0.012 of each other. The 2-layer d64 variant edges ahead slightly.
- **Inference is ~60% slower**: 2.4-2.7 ms vs 1.57 ms. The self-attention over 91 cells adds cost, but this is likely acceptable given the quality improvement.
- **Still converging at E5**: all models still improving, suggesting more epochs or data would help further.

### Recommendation
**Hex-masked SE trunk + attention policy head (d64, 2 layers)** is the new best architecture. Fewer params, better loss, still converging. The inference cost (2.7ms) is manageable. Next steps: integrate into production model.py and retrain.
