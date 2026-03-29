# Hexchess

Hexagonal chess engine (Glinski variant) in Rust with AlphaZero-style self-play training.

## Structure

- **`engine/`** —Rust engine: board representation (91-cell hex grid, axial coordinates), move generation, MCTS search, and neural network inference
- **`training/`** —Python AlphaZero loop: self-play, training, ONNX export, arena evaluation
- **`bindings/wasm/`** —WASM bindings for browser play (uses tract for inference)
- **`bindings/python/`** —PyO3 bindings for the training pipeline (uses ONNX Runtime)
- **`web/`** —Browser UI (vanilla JS + SVG)

## Quick Start

```bash
# Run engine tests
cargo test

# Build Python bindings (for training)
cd bindings/python && maturin develop && cd ../..

# Run a full training loop
python -m training loop --generations 10

# Build WASM bindings (for web UI)
wasm-pack build --target web bindings/wasm
```
