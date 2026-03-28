# Hexchess: Hexagonal Chess Engine + AI

## Overview

A high-performance hexagonal chess engine written in Rust, paired with an AlphaZero-style self-play training pipeline. The engine exposes bindings for both JavaScript (via WASM) and Python, enabling use in web UIs, notebooks, and training loops.

We implement **Glinski's hexagonal chess** — the most widely played hex chess variant (91 cells, flat-topped hex grid, standard piece set with 3 bishops). Glinski was chosen for its large existing player base and well-documented rules, which gives us the best shot at finding reference games and human opponents for evaluation.

---

## Architecture

```
hexchess/
├── Cargo.toml               # Workspace root — ties engine + bindings together
├── engine/                   # Rust core (the single source of truth for game rules)
│   ├── src/
│   │   ├── board.rs          # Board representation, coordinate system
│   │   ├── movegen.rs        # Legal move generation
│   │   ├── game.rs           # Game state, apply/undo, outcome detection
│   │   ├── eval.rs           # Fast heuristic eval (material, mobility — for debugging/baselines)
│   │   ├── mcts.rs           # Monte Carlo Tree Search (policy+value guided)
│   │   ├── inference.rs      # Neural network evaluation (ONNX via ort natively, tract for WASM — single-threaded)
│   │   ├── serialization.rs  # Board/move encoding for NN input, training data export
│   │   └── lib.rs
│   ├── benches/              # Criterion benchmarks (movegen, MCTS throughput)
│   ├── tests/
│   │   └── perft.rs          # Perft test suite — node counts at depth N for known positions
│   └── Cargo.toml
│
├── bindings/
│   ├── wasm/                 # wasm-bindgen JS/TS package
│   │   ├── src/lib.rs        # Thin wrapper exposing engine to JS
│   │   └── Cargo.toml
│   └── python/               # PyO3 bindings
│       ├── src/lib.rs
│       └── Cargo.toml
│
├── web/                      # Minimal browser UI for playing against the engine
│   ├── index.html
│   ├── style.css
│   └── main.js              # Loads WASM, renders hex board, handles game loop vs AI
│
├── training/                 # Python — AlphaZero-style training pipeline
│   ├── self_play.py          # Orchestrates self-play game generation (calls engine via Python bindings)
│   ├── trainer.py            # Neural network training loop (PyTorch)
│   ├── model.py              # Policy+value network definition
│   ├── export.py             # PyTorch -> ONNX export
│   ├── arena.py              # Pit new model vs. current best
│   └── config.py             # Hyperparameters, paths
│
└── models/                   # Checked-in ONNX artifacts (small) + .gitignore for large checkpoints
    └── .gitkeep
```

---

## Core Engine (`engine/`)

### Board Representation

Hexagonal chess is played on a hex grid. We'll use **axial coordinates** `(q, r)` which map cleanly to a flat-topped hexagonal grid and allow efficient neighbor lookup via fixed offset tables.

Board state is stored as a compact struct:
- Piece placement: array indexed by cell ID (91 cells)
- Side to move, half-move clock (for 50-move rule), full-move counter, game phase
- En passant target cell (if a pawn just advanced two squares)
- Zobrist hash for transposition detection
- Position history (list of Zobrist hashes) for threefold repetition detection

Note: Glinski has **no castling** (no rooks-in-corners setup), so castling rights are not tracked.

Draw conditions: stalemate, threefold repetition (via position history), and 50-move rule (via half-move clock).

### Move Generation

Legal move generation is the performance-critical path (MCTS does millions of rollouts).

Key optimizations:
- **Precomputed ray/step tables** — for each cell, precompute hex-direction rays (for sliding pieces) and knight-jump destinations at init time
- **Incremental update** — apply/undo moves mutate board state in-place with minimal work
- **No allocation in the hot path** — moves written into a reusable stack-allocated buffer

### MCTS

The MCTS implementation follows the AlphaZero variant:
1. **Select** — walk tree using PUCT (balances prior policy + visit count)
2. **Expand + Evaluate** — at leaf, call neural network to get `(policy, value)`
3. **Backpropagate** — update visit counts and value estimates up the tree

The engine owns the tree search logic. Neural network inference is pluggable:

```rust
/// Fixed-size policy array over all possible (from, to) pairs on the 91-cell board.
/// Indices are deterministic: see `serialization.rs` for the mapping.
pub type Policy = [f32; NUM_MOVE_INDICES];

pub trait Evaluator: Send + Sync {
    /// Given a board state, write move priors into `policy` and return the value estimate.
    /// Only legal move indices need to be populated; others are ignored by MCTS.
    fn evaluate(&self, state: &GameState, policy: &mut Policy) -> f32;
}
```

The evaluator writes into a caller-owned `Policy` buffer to avoid allocation in the hot path. MCTS reuses a single buffer across evaluations.

Two built-in implementations:
- `OnnxEvaluator` — loads an ONNX model via `ort` (ONNX Runtime Rust bindings) for native play
- `TractEvaluator` — loads an ONNX model via `tract`, a pure-Rust inference library that compiles to WASM
- `RandomEvaluator` — uniform policy, random value; useful for testing

### Serialization / NN Encoding

The engine provides canonical encodings:

- **Board -> tensor**: The 91-cell hex grid is embedded into an **11x11 rectangular array** with invalid cells masked to zero. Each cell gets multiple feature planes (one per piece type per color, plus side-to-move, move count, etc.), producing a tensor of shape `(C, 11, 11)`. The masking approach is simple and lets us use standard 2D convolutions — the network learns to ignore padding cells. This is the same strategy used by Leela Chess Zero for standard chess (8x8 board embedded directly).

- **Move -> index**: bijection between all possible (from, to, promotion?) tuples on the 91-cell board and a fixed-size policy vector. Pawn promotions get **distinct indices per promotion piece type** (queen, rook, bishop, knight), so the network can express promotion preferences. Non-promotion moves use `promotion = None`. This is a sparse mapping (most from-to pairs are unreachable) but keeps the index computation trivial. The engine provides a `move_to_index()` / `index_to_move()` pair and a constant `NUM_MOVE_INDICES`.

- **Training record**: `(board_tensor, policy_target, value_target)` serialized as flat binary records of fixed size. Each record is: `[f32 x C*11*11] [f32 x NUM_MOVE_INDICES] [f32 x 1]`. Files are written as flat sequences of records (no framing or headers), memory-mappable for fast random access in the PyTorch dataloader via `np.memmap`.

---

## Bindings

### WASM (`bindings/wasm/`)

Exposes the engine to JavaScript/TypeScript via `wasm-bindgen`:
- `Game` class: `new()`, `legal_moves()`, `apply_move()`, `is_game_over()`, `status()`
- `AiPlayer` class: wraps MCTS + `TractEvaluator`, model loaded from a `Uint8Array`
- Board state serializable to/from JSON for UI integration

NN inference in WASM uses `tract` (pure Rust, compiles to `wasm32-unknown-unknown` without issue). We avoid `ort` in the WASM target since ONNX Runtime requires native C++ and doesn't cross-compile to WASM cleanly. The ONNX model format is still the interchange — `tract` reads `.onnx` files directly.

Published as an npm package. Built with `wasm-pack`. Enables building a web UI that runs the full engine + AI client-side.

### Python (`bindings/python/`)

Exposes the engine to Python via PyO3, published as a pip-installable package:
- `Game`, `Board`, `Move` classes mirroring the Rust API
- `MctsSearch` class for running search from Python (used in self-play)
- Efficient batch interface: `encode_batch(states) -> np.ndarray` for vectorized NN input
- Zero-copy numpy integration where possible via `numpy` feature of PyO3

This is the bridge between the fast Rust engine and the Python training loop.

---

## Training Pipeline (`training/`)

Implements a self-play reinforcement learning loop modeled after AlphaZero.

### Network Architecture (`model.py`)

A ResNet-style policy+value network:
- **Input**: spatial encoding of the hex board (multi-channel tensor)
- **Backbone**: residual tower (configurable depth/width — start small, scale up)
- **Policy head**: outputs logits over all possible moves; masked to legal moves during search
- **Value head**: scalar in `[-1, 1]` predicting game outcome from current player's perspective

### Self-Play (`self_play.py`)

```
loop:
    1. Load current best ONNX model
    2. Play N games of self-play using MCTS (engine via Python bindings)
       - At each position, run MCTS for K simulations
       - Sample move from visit count distribution (with temperature)
       - Record (state, policy_target, outcome) tuples
    3. Write game records to training data store
```

Self-play is parallelizable: launch multiple processes, each running independent games. The Rust engine handles the heavy lifting (move generation + MCTS); Python just orchestrates.

### Training (`trainer.py`)

```
loop:
    1. Sample minibatches from replay buffer of recent self-play games
    2. Train network to minimize:
       loss = cross_entropy(predicted_policy, mcts_policy)
            + mse(predicted_value, game_outcome)
            + l2_regularization
    3. Periodically export checkpoint to ONNX (export.py)
```

**Replay buffer**: a sliding window over the most recent N games (configurable, default ~500k positions). Training data files are memory-mapped (`np.memmap`) for fast random access without loading everything into RAM. Old files are pruned once the buffer exceeds the configured window size.

### Model Evaluation (`arena.py`)

After training a new checkpoint:
1. Export to ONNX
2. Play a match: new model vs. current best (both using MCTS with their respective models)
3. If new model wins >55% of games, promote it to current best
4. Repeat

### Export (`export.py`)

`torch.onnx.export()` with fixed input shapes matching the engine's encoding. The resulting `.onnx` file can be loaded by:
- The Rust engine (via `ort`) for native play
- The WASM build (via `tract`) for browser play
- Python directly for evaluation/debugging

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Rust core, not Python** | MCTS throughput is everything — need millions of simulations/sec. Rust gives us zero-cost abstractions, no GC pauses, and easy WASM compilation. |
| **ONNX as the model interchange format** | Decouples training framework (PyTorch) from inference runtime. ONNX runs everywhere: native, WASM, mobile. |
| **Engine owns MCTS, not Python** | The search loop is tight and latency-sensitive. Crossing the Python/Rust boundary per tree node would kill performance. Python just calls `engine.search(state, num_simulations)` and gets back a policy. |
| **Axial hex coordinates** | Clean arithmetic for neighbor computation, established convention in hex grid literature, minimal wasted space vs. offset or cube coords. |
| **Monorepo** | Engine, bindings, and training are tightly coupled (encoding formats, move indices). Keeping them together prevents version skew. |
| **`ort` natively, `tract` for WASM** | ONNX Runtime (`ort`) gives best native performance with hardware acceleration (CUDA, CoreML). `tract` is pure Rust and compiles to WASM without issue. Both read the same `.onnx` files, so one model artifact serves all targets. |
| **11x11 rectangular embedding for hex grid** | Lets us use standard 2D convolutions with no custom ops. The network learns to ignore the ~30 masked padding cells. Simple to implement, well-understood, and compatible with standard NN tooling. |

---

## Development Phases

### Phase 1 — Engine fundamentals
- Board representation and coordinate system
- Full legal move generation with tests against known positions
- Game state management (apply/undo, check/checkmate/stalemate/draw detection)
- Perft test suite — no published perft tables exist for Glinski, so we cross-validate against `scottbedard/hexchess` at shallow depths (depth 1 = 50 moves from start is confirmed). Rely on manual QA first, then encode confirmed-correct behavior into regression tests once confident.
- Property-based tests (fuzz movegen, verify apply/undo round-trips)
- Benchmarks for movegen throughput

### Phase 2 — Bindings + playable browser demo
- MCTS with `RandomEvaluator` (uniform policy, random value) as a baseline opponent
- WASM bindings (wasm-pack) — engine + MCTS exposed to JS
- Python bindings (PyO3) — enough to play a game from a Python script
- **Playable web UI**: minimal browser app where a human plays against the MCTS random-rollout opponent. This is the primary validation milestone — manual QA to confirm the engine handles all piece movements, captures, promotions, check, checkmate, stalemate, and draws correctly in real games. The UI needs to show the hex board, highlight legal moves on click, and let the player make moves against the AI.

### Phase 3 — Neural network integration
- Define NN input encoding and policy output mapping in the engine
- Implement `OnnxEvaluator` in Rust
- Train a small initial model via supervised learning on random self-play data (bootstrap)
- Wire MCTS to use the NN evaluator

### Phase 4 — Self-play training loop
- Self-play data generation pipeline
- Training loop with replay buffer
- ONNX export and model promotion (arena)
- Iterate: generate data -> train -> evaluate -> promote -> repeat

### Phase 5 — Polish + scale
- Optimize MCTS (tree reuse, batched NN inference with virtual loss for parallel search)
- Scale up network size and training compute
- Upgrade web UI to bundle a trained ONNX model (via tract) so the browser opponent uses the NN instead of random rollouts
- npm + PyPI package publishing

---

## CI / Tooling

- **Rust**: `cargo test` (unit + perft), `cargo bench` (criterion), `cargo clippy`, `cargo fmt --check`
- **Python bindings**: built with `maturin` (`maturin develop` for local install, `maturin build --release` for wheels)
- **WASM bindings**: built with `wasm-pack build --target web`
- **Training**: `pytest` for training pipeline tests, `ruff` for linting
- **GitHub Actions**: CI runs on push/PR — Rust tests + clippy, Python bindings build + test, WASM build, training unit tests
