# TODOs

## Neural network architecture

- **Attention policy head** — Replace conv→flatten→FC policy head with attention layer mapping spatial features to move logits. Lc0's attention policy was +270 Elo over conv policy. More parameter-efficient for 4206-dim move output.

- **Smolgen-style dynamic position encoding** — Lc0's chess-specific innovation: compress board state into a vector, generate supplemental attention logits per head. +50% effective capacity at 10% throughput cost.

## MCTS optimizations

- Tree reuse across moves — keep the subtree rooted at the chosen move instead of rebuilding from scratch

## Training pipeline

- **Surprise weighting for replay buffer sampling** — KataGo and Lc0 both upweight positions where the net's raw evaluation disagrees with the MCTS search result (policy KL divergence, value squared error). The intuition: these are positions where the net has the most to learn. KataGo stores per-position `targetWeight` in `.npz` files during self-play (`policySurpriseDataWeight`, `valueSurpriseDataWeight`). Lc0 calls it "diff-focus" (`diff_focus_q_weight`, `diff_focus_pol_scale`). Implementation: compute surprise weights in the worker (raw net output vs MCTS result), store as `sample_weights` in `.npz`, use in `ReplayBuffer` sampling.

## Packaging / distribution

- npm package for WASM bindings
- PyPI package for Python bindings

## Testing

- Property-based tests (fuzz movegen, verify apply/undo round-trips)

## Evaluator API cleanup

- The proposal specifies a zero-alloc `Policy` buffer API (`evaluate(&self, state, &mut Policy) -> f32`) but the current implementation returns `(Vec<f32>, f32)`. Consider switching to the buffer API to reduce allocations in the hot path.
