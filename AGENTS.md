# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

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

# Documentation
make docs-dev                           # WASM build + Fumadocs dev server
make docs-build                         # production build

# Docker
make docker-up                          # start all training services
make docker-down                        # stop all services
```

## Architecture

**Workspace crates:** `engine/`, `bindings/wasm/`, `bindings/python/`

### Engine (`engine/src/`)

- **board.rs** - 91-cell hex grid with axial coordinates `(q, r)`. Validity: `max(|q|, |r|, |q+r|) <= 5`. Cells indexed 0-90 via const lookup tables. Zobrist hashing for transposition/repetition detection.
- **movegen.rs** - Pseudo-legal then legal move generation. Precomputed ray tables (sliding pieces) and knight tables lazily initialized via `OnceLock`. `MoveList` is stack-allocated (256-entry array, no heap in hot path).
- **game.rs** - `GameState` with apply/undo (in-place mutation + `UndoInfo` stack). Status detection: checkmate, stalemate, threefold repetition, 50-move rule, insufficient material.
- **mcts.rs** - AlphaZero PUCT search. Arena-allocated nodes (`Vec<MctsNode>` indexed by `usize`). Transposition table caches NN evaluations. Dirichlet noise at root. Temperature-based move selection.
- **serialization.rs** - Board-to-tensor encoding: `(22, 11, 11)` CHW layout in **side-to-move frame**, hex grid embedded in 11x11 rect (invalid cells zero-padded). When black is to move, the board is centrally inverted `(q,r)->(-q,-r)` and piece channels are color-swapped so that channels 0-5 are always "my pieces" and 6-11 are always "opponent pieces" - the network never sees an absolute-color view. Channel 12 is the absolute side-to-move (redundant but kept), 13-14 fullmove/halfmove, 15 en passant (STM-framed), 16 repetition, 17 validity mask, 18 in-check, 19-20 last-move from/to (STM-framed), 21 plies-since-pawn. Deterministic move-to-index bijection (~4000 entries); pawn promotions get 4 separate indices. Policy vectors are also in STM frame - use `stm_policy_index(mv, stm)` (Rust) or `game.policy_index(...)` (Python) to index a move, which applies the mirror remap for black. `encode_board` takes `&GameState` (not `&Board`) to access repetition history.
- **inference.rs** - `Evaluator` trait returns `(policy_logits, value)`. NN outputs WDL logits (Win/Draw/Loss); evaluators convert to scalar `W - L` for MCTS. Implementations: `OnnxEvaluator` (ORT, feature-gated behind `onnx`), `TractEvaluator` (pure Rust, used in WASM), `HeuristicEvaluator` (one-ply lookahead with the same `EvalWeights` as minimax).
- **eval.rs** - Material heuristic: centipawn sum through `tanh(cp/750)`.
- **minimax.rs** - Alpha-beta search with configurable `EvalWeights` (material, mobility, pawn structure, king safety, ...). Used for imitation bootstrap and exposed via the Python `minimax_search*` functions.

Glinski algebraic notation (`f6`, `b1-b2`, `f10-f11=Q`) is implemented on `HexCoord` / `Move` directly in `board.rs` and `movegen.rs` - there is no separate `notation.rs`. Move indices are deterministic and unrelated to notation strings.

### Bindings

- **WASM** (`bindings/wasm/`) - wasm-bindgen. Exports `Game` and `AiPlayer` classes. Uses `TractEvaluator` (no native ONNX in WASM). Serde for JS interop. Optimized for size (LTO, `opt-level = "s"`).
- **Python** (`bindings/python/`) - PyO3. Exports `Game`, `MctsSearch`, `encode_board()`, `move_to_index()` (absolute-frame; prefer `Game.policy_index()` for STM-frame targets paired with `encode_board`). Uses `OnnxEvaluator` (engine compiled with `onnx` feature). NumPy integration for tensor passing.

### Training (`training/`)

AlphaZero-style self-play training loop. All shared state lives in S3 (DigitalOcean Spaces / Cloudflare R2). Workers can run anywhere - local machines, cloud VMs, Docker.

- **storage.py** - S3 abstraction layer. Key constants, upload/download helpers, position counting from filenames. All shared I/O goes through this module.
- **worker.py** - Continuous self-play loop. Polls S3 for model updates, plays games using MCTS, flushes `.npz` batches to S3. During bootstrap (no model), generates minimax imitation data with process-pool parallelism.
- **trainer_loop.py** - Continuous trainer. Samples from a sliding-window replay buffer (downloads from S3), emits periodic trainer summaries for observability, and promotes as soon as the fresh-position threshold is met at a reload/poll boundary. KataGo-style token bucket throttles training to match data production rate.
- **model.py** - `HexChessNet`: 8 SE residual blocks (144 filters) with KataGo-style global pooling at blocks 2 and 5. Input `(22, 11, 11)` in STM frame. Heads: main policy, terminal WDL value, moves-left (MLH), short-term value (STV), and auxiliary opponent policy.
- **export.py** - PyTorch -> ONNX export.
- **elo.py** - Shared Elo types: `Player` protocol, `MinimaxPlayer`, `MctsPlayer`, game play with per-player timing, OpenSkill (Plackett-Luce / Weng-Lin) rating math rescaled to look Elo-like, `predict_draw`, table formatting.
- **elo_service.py** - Continuous Elo rating service. `predict_draw`-driven matchmaking with placement matches for new models. Scales horizontally: each replica plays one game at a time and writes per-game immutable objects under `state/elo_games/`; the `state/elo.json` projection is rebuilt from the game log (last-writer-wins, idempotent). LRU-caches loaded ONNX sessions to bound memory.
- **imitation.py** - Minimax self-play games for the bootstrap phase, parallelized with `ProcessPoolExecutor`.
- **dashboard.py** + **dashboard_store.py** + **dashboard.html** - HTTP server backed by an in-memory store that incrementally syncs from S3 in a background thread (ETag caches + LIST diffs on `state/elo_games/`), so HTTP requests never block on S3.
- **config.py** - Hyperparameters. Local `.cache/` directory for downloaded S3 objects.

### Documentation (`docs/`)

Fumadocs (Next.js) documentation site with an interactive playground. The playground loads the WASM package to let users play hexagonal chess in the browser. Built via `make docs-dev`.

## Glinski Hex Chess Rules (key differences from standard chess)

- 91 hexagonal cells, 3 bishops per side, 9 pawns per side, no castling
- 12 directions: 6 cardinal (rook), 6 diagonal (bishop), 12 knight jumps
- White pawns advance in +r direction, black in -r direction
- Promotion cells: 11 edge cells per color along the far rank
- Starting position has 51 legal moves (king has 2 moves from start, not 1)

## S3 Bucket Layout

All training artifacts are stored in S3 (configured via `.env`):
```text
models/
  latest.onnx              # current model for inference
  latest.meta.json         # {"version": N, "timestamp": "...", "positions_at_promote": M}
  checkpoint.pt            # PyTorch training checkpoint
  versions/{N}.onnx        # immutable version snapshots

data/
  selfplay/v{N}/{ts}_{rand}_n{count}.npz
  selfplay_traces/v{N}/{game_id}.json
  imitation/{ts}_{rand}_n{count}.npz

state/
  elo.json                 # derived Elo projection (rebuilt from elo_games/)
  elo_games/{ts}_{rand}.json  # one immutable object per played game
  trainer_metrics.json     # latest trainer telemetry for the dashboard

benchmarks/
  results/{version}.json   # benchmark snapshots consumed by the dashboard

heartbeats/
  {hostname}.json          # worker liveness + stats
```

Position count is encoded in each `.npz` filename (`_n{count}`) so the trainer can count positions via S3 LIST without opening files.

## Workflow

- Always run `codex review` on a PR before merging it. Use `codex review --base <branch>` for branch diffs or `codex review --uncommitted` for local changes.
- Do not merge feature branches directly into local or remote `main`. If a task says "merge the PR", push the branch and create or update a GitHub PR instead of advancing `main` locally.
- When merging a PR, use GitHub's `squash` or `rebase` merge modes. Do not create merge commits unless the user explicitly asks for that history shape.
- When asked to assess training health, consult `ai/training-health.md` for the data collection and interpretation checklist.

## Deployment

Two k8s clusters: `kevz-infra` (public Hetzner k3s) and `kevz-gpu` (private, behind Tailscale). Manifests live in `k8s/`.

**Image tags:** The Docker workflow publishes both floating tags (`cpu`, `cuda`) and SHA-pinned tags using 7-char short SHAs (`cpu-<sha7>`, `cuda-<sha7>`). The manifests in `k8s/` use the floating tags as defaults, but deploys should pin to a specific SHA tag at apply time for reproducibility:
```bash
# Find available SHA tags
gh api /users/k15z/packages/container/hexchess-zero%2Ftraining/versions \
  --jq '.[].metadata.container.tags[]' | grep -E '^(cpu|cuda)-'

# Deploy with a specific SHA tag
TAG=47e2984
kubectl set image deployment/trainer trainer=ghcr.io/k15z/hexchess-zero/training:cuda-$TAG -n hexchess
kubectl set image deployment/worker worker=ghcr.io/k15z/hexchess-zero/training:cpu-$TAG -n hexchess
kubectl set image deployment/dashboard dashboard=ghcr.io/k15z/hexchess-zero/training:cpu-$TAG \
  tunnel=ghcr.io/k15z/hexchess-zero/training:cpu-$TAG -n hexchess
kubectl set image deployment/elo-service elo=ghcr.io/k15z/hexchess-zero/training:cpu-$TAG -n hexchess
```

## Key Conventions

- Coordinates are always axial `(q, r)` - never use doubled or offset coordinates
- The `onnx` cargo feature gates all ONNX Runtime dependencies; Python bindings enable it, WASM does not
- Move indices must stay deterministic - any change to the move table breaks all existing training data and models
- Board tensor embedding: `col = q + 5, row = r + 5` maps hex cells into the 11x11 grid
- S3 key constants are defined in `training/storage.py` - never use raw string keys elsewhere
- `.env` file contains S3 credentials (BUCKET_NAME, ACCESS_KEY, SECRET_KEY, ENDPOINT) - gitignored, never committed
- MCTS evaluations and tournaments must use >=800 simulations (the production default, matching AlphaZero/Lc0). Lower sim counts produce misleading results because the NN's learned policy advantage only compounds with sufficient search depth
- The k3s cluster nodes (`kevz-infra-*`) are Hetzner servers. Never delete Hetzner servers without filtering by name - always exclude `kevz-infra-*` to avoid destroying the cluster
