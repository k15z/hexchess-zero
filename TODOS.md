# TODOs

## Neural network architecture

- **WDL value head** — Replace tanh scalar with 3-class Win/Draw/Loss softmax. Confirmed +33 Elo in AlphaZero-FX, core improvement in Lc0. Requires changes to model.py, trainer.py (loss fn), self_play.py (store discrete W/D/L labels), and Rust inference to interpret 3-class output.

- **Attention policy head** — Replace conv→flatten→FC policy head with attention layer mapping spatial features to move logits. Lc0's attention policy was +270 Elo over conv policy. More parameter-efficient for 4206-dim move output.

- **Smolgen-style dynamic position encoding** — Lc0's chess-specific innovation: compress board state into a vector, generate supplemental attention logits per head. +50% effective capacity at 10% throughput cost.

## Neural network input encoding

- Add repetition count plane to board encoding (0/1/2 times current position has been seen). Without this, the network is blind to repetitions and can't learn to avoid/seek draws. AlphaZero uses last 8 board positions as history planes; LeelaZero uses a simpler repetition count. Either approach would help reduce the high `draw_repetition` rate in self-play.

- **Richer input features** — Add derived feature planes to board encoding (currently 16ch). Candidates from AlphaZero-FX (+97 Elo): material count per side, material difference, checking pieces, piece mobility/attack maps. Requires changes to serialization.rs, model input channels, and TENSOR_SHAPE.

## MCTS optimizations

- Tree reuse across moves — keep the subtree rooted at the chosen move instead of rebuilding from scratch

## Training pipeline

- Scale up network size (more residual blocks / filters) once baseline model quality stabilizes
- **Bump shuffle buffer** — Current 10K is small relative to ~400K positions/gen. Lc0 uses 200K-2M. Consider 100K+ for better batch decorrelation.
- **Continuous training** — Move from discrete generations to async self-play + continuous training (true AlphaZero style). Eliminates stale-model problem.

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
