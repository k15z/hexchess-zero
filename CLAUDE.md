# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Hexagonal chess engine (Glinski variant) in Rust with AlphaZero-style self-play training. Monorepo: Rust engine is the single source of truth for game rules, with WASM bindings for browser play and Python bindings for training.

## Build & Test Commands

```bash
# Setup (one-time)
make setup                              # uv sync + build Python bindings

# Training pipeline
make worker                             # run self-play worker
make trainer                            # run continuous trainer
make elo                                # run Elo rating service
make dashboard                          # run training dashboard
make status                             # show training status

# Engine (Rust)
make test                               # cargo test (~95 tests)
make bench                              # criterion benchmark for move generation
make lint                               # clippy + fmt check
cargo test board::tests::test_name      # run a single test by name

# Web frontend
make web-dev                            # WASM build + Vite dev server
make web-build                          # production build

# Docker
make docker-up                          # start all training services
make docker-down                        # stop all services
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
- **eval.rs** — Material heuristic: centipawn sum through `tanh(cp/750)`.

### Bindings

- **WASM** (`bindings/wasm/`) — wasm-bindgen. Exports `Game` and `AiPlayer` classes. Uses `TractEvaluator` (no native ONNX in WASM). Serde for JS interop. Optimized for size (LTO, `opt-level = "s"`).
- **Python** (`bindings/python/`) — PyO3. Exports `Game`, `MctsSearch`, `encode_board()`, `move_to_index()`. Uses `OnnxEvaluator` (engine compiled with `onnx` feature). NumPy integration for tensor passing.

### Training (`training/`)

AlphaZero-style self-play training loop. All shared state lives in S3 (DigitalOcean Spaces / Cloudflare R2). Workers can run anywhere — local machines, cloud VMs, Docker.

- **storage.py** — S3 abstraction layer. Key constants, upload/download helpers, position counting from filenames. All shared I/O goes through this module.
- **worker.py** — Continuous self-play loop. Polls S3 for model updates, plays games using MCTS, flushes `.npz` batches to S3. During bootstrap (no model), generates minimax imitation data with process-pool parallelism.
- **trainer_loop.py** — Continuous trainer. Samples from a sliding-window replay buffer (downloads from S3), trains for N steps per cycle, exports ONNX, promotes to S3. KataGo-style token bucket throttles training to match data production rate.
- **model.py** — `HexChessNet`: SE residual blocks (128 filters) with KataGo-style global pooling → policy head + WDL value head. Input `(19, 11, 11)`, policy output size = `num_move_indices()`, WDL output = 3 logits.
- **export.py** — PyTorch → ONNX export.
- **elo.py** — Shared Elo types: Player protocol, MinimaxPlayer, MctsPlayer, game play with per-player timing, MLE Elo computation with Bayesian prior.
- **elo_service.py** — Continuous Elo rating service. Uncertainty-based matchmaking, persists state to S3 (`state/elo.json`). LRU-caches player objects to bound memory.
- **config.py** — Hyperparameters. Local `.cache/` directory for downloaded S3 objects.
- **dashboard.py** + **dashboard.html** — Lightweight web dashboard reading from S3.

### Web (`web/`)

React + TypeScript. Loads WASM package, renders flat-topped hex board, supports human vs AI play.

## Glinski Hex Chess Rules (key differences from standard chess)

- 91 hexagonal cells, 3 bishops per side, 9 pawns per side, no castling
- 12 directions: 6 cardinal (rook), 6 diagonal (bishop), 12 knight jumps
- White pawns advance in +r direction, black in -r direction
- Promotion cells: 11 edge cells per color along the far rank
- Starting position has 51 legal moves (king has 2 moves from start, not 1)

## S3 Bucket Layout

All training artifacts are stored in S3 (configured via `.env`):
```
models/
  latest.onnx              # current model for inference
  latest.meta.json         # {"version": N, "timestamp": "..."}
  checkpoint.pt            # PyTorch training checkpoint
  versions/{N}.onnx        # immutable version snapshots

data/
  selfplay/v{N}/{ts}_{rand}_n{count}.npz
  imitation/{ts}_{rand}_n{count}.npz

state/
  elo.json                 # Elo ratings + pair results + player timing

heartbeats/
  {hostname}.json          # worker liveness + stats
```

Position count is encoded in each `.npz` filename (`_n{count}`) so the trainer can count positions via S3 LIST without opening files.

## Workflow

- Always run `/review` on a PR **before** merging it. Address any correctness or performance issues before merge.

## Key Conventions

- Coordinates are always axial `(q, r)` — never use doubled or offset coordinates
- The `onnx` cargo feature gates all ONNX Runtime dependencies; Python bindings enable it, WASM does not
- Move indices must stay deterministic — any change to the move table breaks all existing training data and models
- Board tensor embedding: `col = q + 5, row = r + 5` maps hex cells into the 11x11 grid
- S3 key constants are defined in `training/storage.py` — never use raw string keys elsewhere
- `.env` file contains S3 credentials (BUCKET_NAME, ACCESS_KEY, SECRET_KEY, ENDPOINT) — gitignored, never committed
- MCTS evaluations and tournaments must use ≥500 simulations (the production default). Lower sim counts (e.g. 200) produce misleading results because the NN's learned policy advantage only compounds with sufficient search depth
- The k3s cluster nodes (`kevz-infra-*`) are Hetzner servers. **Never delete Hetzner servers without filtering by name** — always exclude `kevz-infra-*` to avoid destroying the cluster
