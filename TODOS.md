# TODOs

## Neural network architecture

- **Attention policy head** — Replace conv→flatten→FC policy head with attention layer mapping spatial features to move logits. Lc0's attention policy was +270 Elo over conv policy. More parameter-efficient for 4206-dim move output.

- **Smolgen-style dynamic position encoding** — Lc0's chess-specific innovation: compress board state into a vector, generate supplemental attention logits per head. +50% effective capacity at 10% throughput cost.

## MCTS optimizations

- Tree reuse across moves — keep the subtree rooted at the chosen move instead of rebuilding from scratch

## Training pipeline

- **Surprise weighting for replay buffer sampling** — KataGo and Lc0 both upweight positions where the net's raw evaluation disagrees with the MCTS search result (policy KL divergence, value squared error). The intuition: these are positions where the net has the most to learn. KataGo stores per-position `targetWeight` in `.npz` files during self-play (`policySurpriseDataWeight`, `valueSurpriseDataWeight`). Lc0 calls it "diff-focus" (`diff_focus_q_weight`, `diff_focus_pol_scale`). Implementation: compute surprise weights in the worker (raw net output vs MCTS result), store as `sample_weights` in `.npz`, use in `ReplayBuffer` sampling.
- **Training rate limiting** — KataGo uses a "train bucket" mechanism (`-max-train-bucket-per-new-data 4`) that limits how many training steps can be taken per new data row, preventing overfitting when self-play throughput is low relative to training speed. Implementation: track how many times each batch/file has been sampled, skip or deprioritize files that have been trained on more than N times. This naturally throttles the trainer to stay in sync with worker output.
- Scale up network size (more residual blocks / filters) once baseline model quality stabilizes

## Web UI

- Bundle a trained ONNX model (via tract) so the browser opponent uses the NN instead of random rollouts

## Packaging / distribution

- npm package for WASM bindings
- PyPI package for Python bindings

## CI

- GitHub Actions: Rust tests + clippy, Python bindings build + test, WASM build
- Training pipeline tests (pytest) and linting (ruff)

## Testing

- Property-based tests (fuzz movegen, verify apply/undo round-trips)
- Perft test suite — no published perft tables exist for Glinski, so cross-validate against `scottbedard/hexchess` at shallow depths and encode confirmed-correct behavior into regression tests

## Evaluator API cleanup

- The proposal specifies a zero-alloc `Policy` buffer API (`evaluate(&self, state, &mut Policy) -> f32`) but the current implementation returns `(Vec<f32>, f32)`. Consider switching to the buffer API to reduce allocations in the hot path.
