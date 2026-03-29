# TODOs

## Neural network input encoding

- Add repetition count plane to board encoding (0/1/2 times current position has been seen). Without this, the network is blind to repetitions and can't learn to avoid/seek draws. AlphaZero uses last 8 board positions as history planes; LeelaZero uses a simpler repetition count. Either approach would help reduce the high `draw_repetition` rate in self-play.

## MCTS optimizations

- Tree reuse across moves — keep the subtree rooted at the chosen move instead of rebuilding from scratch

## Training pipeline

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
