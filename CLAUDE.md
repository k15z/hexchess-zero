# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Hexagonal chess engine (Glinski variant) in Rust with AlphaZero-style self-play training. Monorepo: Rust engine is the single source of truth for game rules, with WASM bindings for browser play and Python bindings for training.

## Build & Test Commands

```bash
# Engine (Rust)
cargo test                              # run all unit tests (~95 tests)
cargo test board::tests::test_name      # run a single test by name
cargo test -p hexchess-engine           # test only the engine crate
cargo bench --bench movegen             # criterion benchmark for move generation
cargo clippy                            # lint
cargo fmt --check                       # format check

# WASM bindings
wasm-pack build --target web bindings/wasm

# Python bindings (local dev install)
cd bindings/python && maturin develop

# Training pipeline (async distributed)
python -m training worker                 # run self-play worker loop
python -m training trainer                # run continuous trainer loop
python -m training status                 # show cluster status
python -m training progress               # show training progress table
python -m training elo-service             # run continuous Elo rating service
```

## Architecture

**Workspace crates:** `engine/`, `bindings/wasm/`, `bindings/python/`

### Engine (`engine/src/`)

- **board.rs** — 91-cell hex grid with axial coordinates `(q, r)`. Validity: `max(|q|, |r|, |q+r|) <= 5`. Cells indexed 0-90 via const lookup tables. Zobrist hashing for transposition/repetition detection.
- **movegen.rs** — Pseudo-legal then legal move generation. Precomputed ray tables (sliding pieces) and knight tables lazily initialized via `OnceLock`. `MoveList` is stack-allocated (256-entry array, no heap in hot path).
- **game.rs** — `GameState` with apply/undo (in-place mutation + `UndoInfo` stack). Status detection: checkmate, stalemate, threefold repetition, 50-move rule, insufficient material.
- **mcts.rs** — AlphaZero PUCT search. Arena-allocated nodes (`Vec<MctsNode>` indexed by `usize`). Transposition table caches NN evaluations. Dirichlet noise at root. Temperature-based move selection.
- **serialization.rs** — Board-to-tensor encoding: `(19, 11, 11)` CHW layout, hex grid embedded in 11x11 rect (invalid cells zero-padded). Channels 0-11: piece planes, 12: side to move, 13: fullmove, 14: halfmove clock, 15: en passant, 16: repetition count, 17: validity mask, 18: in-check. Deterministic move-to-index bijection (~4000 entries) for policy vector; pawn promotions get 4 separate indices. `encode_board` takes `&GameState` (not `&Board`) to access repetition history.
- **inference.rs** — `Evaluator` trait returns `(policy_logits, value)`. NN outputs WDL logits (Win/Draw/Loss); evaluators convert to scalar `W - L` for MCTS. Implementations: `OnnxEvaluator` (ORT, feature-gated behind `onnx`), `TractEvaluator` (pure Rust, used in WASM), `HeuristicEvaluator` (material-based baseline).
- **eval.rs** — Material heuristic: centipawn sum through `tanh(cp/400)`.

### Bindings

- **WASM** (`bindings/wasm/`) — wasm-bindgen. Exports `Game` and `AiPlayer` classes. Uses `TractEvaluator` (no native ONNX in WASM). Serde for JS interop. Optimized for size (LTO, `opt-level = "s"`).
- **Python** (`bindings/python/`) — PyO3. Exports `Game`, `MctsSearch`, `encode_board()`, `move_to_index()`. Uses `OnnxEvaluator` (engine compiled with `onnx` feature). NumPy integration for tensor passing.

### Training (`training/`)

Async distributed AlphaZero loop: workers generate self-play data continuously, trainer promotes every model unconditionally after each cycle. Elo service tracks strength independently.

- **worker.py** — Continuous self-play loop. Polls for model updates, plays games using MCTS + latest model, flushes `.npz` batches to shared storage.
- **trainer_loop.py** — Continuous trainer loop. Samples from a recency-weighted replay buffer (5M positions, 3-hour half-life), trains for N steps, exports ONNX, promotes unconditionally. Buffer reloaded every 1K steps to pick up fresh worker data. Saves versioned snapshots (`models/vN.onnx`).
- **model.py** — `HexChessNet`: conv input → 6 residual blocks (128 filters) → policy head + WDL value head. Input `(19, 11, 11)`, policy output size = `num_move_indices()`, WDL output = 3 logits (Win/Draw/Loss).
- **export.py** — PyTorch → ONNX. Softmax applied to policy logits at inference time in evaluator, not in the model.
- **elo.py** — Shared Elo types: Player protocol, MinimaxPlayer, MctsPlayer, game play, MLE Elo computation.
- **elo_service.py** — Continuous Elo rating service (k8s Deployment). Uncertainty-based matchmaking: picks the least-played pair, plays one game, updates Elo. Persists state to `elo_state.json`. LRU-caches player objects to bound memory.
- **metrics.py** — Reads trainer logs and displays progress summary table.
- **config.py** — All hyperparameters and derived paths (`AsyncConfig`).

### Web (`web/`)

Vanilla JS + SVG. Loads WASM package, renders flat-topped hex board, supports human vs AI play.

## Glinski Hex Chess Rules (key differences from standard chess)

- 91 hexagonal cells, 3 bishops per side, 9 pawns per side, no castling
- 12 directions: 6 cardinal (rook), 6 diagonal (bishop), 12 knight jumps
- White pawns advance in +r direction, black in -r direction
- Promotion cells: 11 edge cells per color along the far rank
- Starting position has 51 legal moves (king has 2 moves from start, not 1)

## Data Layout

Training artifacts are stored in `.data/` (gitignored), shared via NFS on k8s:
```
.data/
  models/          # best.onnx, best.pt, best.meta.json, v{N}.onnx snapshots
  training_data/   # version-tagged selfplay .npz files
  logs/            # trainer.jsonl, worker-*.jsonl
  elo_state.json      # Continuous Elo service state (pair results, ratings)
  elo_rankings.jsonl  # Elo ranking history (appended by both batch and service)
```

## Workflow

- Always run `/review` on a PR **before** merging it. Address any correctness or performance issues before merge.

## Key Conventions

- Coordinates are always axial `(q, r)` — never use doubled or offset coordinates
- The `onnx` cargo feature gates all ONNX Runtime dependencies; Python bindings enable it, WASM does not
- Move indices must stay deterministic — any change to the move table breaks all existing training data and models
- Board tensor embedding: `col = q + 5, row = r + 5` maps hex cells into the 11x11 grid
