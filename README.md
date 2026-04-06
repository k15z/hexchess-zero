# Hexchess

Hexagonal chess engine (Glinski variant) in Rust with AlphaZero-style self-play training.

## Install

Prebuilt packages are published to PyPI and npm under the name **`hexchess-zero`**:

```bash
pip install hexchess-zero       # Python (imports as `hexchess`)
npm install hexchess-zero       # JavaScript / WASM
```

Python wheels are available for Linux, macOS, and Windows (3.9–3.13); the npm package bundles the WASM binary. See [`docs/content/docs/usage/`](docs/content/docs/usage/) for API documentation, or the published docs site for interactive examples.

## Structure

- **`engine/`** — Rust engine: board representation (91-cell hex grid, axial coordinates), move generation, MCTS search, and neural network inference
- **`training/`** — Async distributed AlphaZero loop: self-play workers, continuous trainer, Elo rating service
- **`bindings/wasm/`** — WASM bindings for browser play (uses tract for inference)
- **`bindings/python/`** — PyO3 bindings for the training pipeline (uses ONNX Runtime)
- **`docs/`** — Documentation site with interactive playground (Fumadocs)

## Quick Start

```bash
# Run engine tests
cargo test

# Build Python bindings (for training)
cd bindings/python && maturin develop && cd ../..

# Start training (run in separate terminals)
python -m training worker      # self-play worker (run N of these)
python -m training trainer     # continuous trainer (run 1)
python -m training elo-service # Elo rating service (run 1)

# Monitor
python -m training status      # cluster status
python -m training progress    # training progress table

# Run documentation site (includes interactive playground)
make docs-dev
```

## Training Pipeline

The pipeline has three async components that communicate through shared storage (`.data/`):

1. **Workers** generate self-play games using MCTS + the latest model, flush `.npz` batches to `.data/training_data/`
2. **Trainer** samples uniformly from the most recent positions, trains, exports ONNX, and promotes a new model version every cycle
3. **Elo service** runs continuous round-robin matches between model versions to track strength

On first run (no model exists), the trainer bootstraps by training on minimax-supervised imitation data before switching to self-play.

### Configuration

All parameters live in `training/config.py`. Here's how they interact:

#### MCTS & Self-Play

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_simulations` | 500 | MCTS simulations per move. Higher = stronger play but slower data generation. Directly controls worker throughput (positions/hour). |
| `temperature_threshold` | 60 | Move number after which temperature drops to `temperature_low`. Controls exploration vs exploitation in self-play games. |
| `temperature_high` | 1.0 | Temperature for early-game moves (before threshold). Higher = more diverse openings in training data. |
| `temperature_low` | 0.35 | Temperature for late-game moves (after threshold). Lc0-style floor ensures policy targets retain gradient signal. |
| `dirichlet_alpha` | 0.3 | Dirichlet noise concentration at MCTS root. Encourages exploration of moves the net hasn't learned yet. |
| `dirichlet_epsilon` | 0.25 | Mixing weight for Dirichlet noise (0 = no noise, 1 = all noise). |
| `worker_batch_size` | 5 | Games per `.npz` flush. Smaller = fresher data available to trainer sooner, but more filesystem overhead. |

#### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 256 | Training batch size. Interacts with `replay_buffer_size` — larger batches relative to buffer = more repeated samples per epoch. |
| `learning_rate` | 0.001 | Adam learning rate. |
| `l2_regularization` | 1e-4 | Weight decay. Prevents overfitting, especially important when the replay buffer is small relative to training steps. |
| `replay_buffer_size` | 5,000,000 | Max positions in the sliding window. The trainer selects the most recent `.npz` files up to this limit and samples uniformly. Larger = more data diversity but older positions stay longer. |
| `steps_per_cycle` | 5,000 | Training steps before promoting a new model version. Each version = one cycle. Controls how often workers get an updated model. Shorter cycles = faster model turnover but less training per version. |
| `reload_interval` | 1,000 | Re-scan `.data/training_data/` every N steps within a cycle, picking up fresh worker output. Without this, the trainer would use a stale snapshot for the entire cycle. Should be < `steps_per_cycle`. |
| `min_positions_to_start` | 1,000,000 | Bootstrap gate: self-play training won't start until this many positions exist. Prevents training on too little data early on. |

#### Bootstrap (Imitation)

These only apply to the initial bootstrap phase when no model exists yet:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `imitation_depth` | 3 | Minimax search depth for generating imitation targets. Deeper = better targets but much slower generation. |
| `imitation_random_plies` | 8 | Random opening moves per imitation game. Creates position diversity so the net doesn't just memorize one opening. |
| `imitation_temperature` | 200.0 | Softmax temperature for converting centipawn scores to policy targets. Higher = softer policy (more weight on suboptimal moves). |
| `bootstrap_steps` | 50,000 | Training steps for imitation bootstrap. Must be enough to beat the heuristic evaluator, then self-play takes over. |

#### Network Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_residual_blocks` | 6 | Depth of the residual tower. More blocks = more capacity but slower inference (affects worker throughput). |
| `num_filters` | 128 | Width of convolutional layers. More filters = more capacity but slower inference. |
| `board_channels` | 19 | Input tensor channels (piece planes + metadata). Must match `serialization.rs`. |
| `board_height` / `board_width` | 11 | Hex grid embedded in 11x11 rectangle. Fixed by the coordinate system. |

### Key Dynamics

**Worker throughput vs training speed:** Workers produce positions at a rate determined by `num_simulations` and hardware. The trainer consumes them at a rate determined by `steps_per_cycle`, `batch_size`, and GPU speed. If the trainer is much faster than workers, it overtrains on stale data. If workers are much faster, data goes untrained.

**Buffer size vs cycle length:** With `replay_buffer_size = 5M` and `steps_per_cycle = 5000` at `batch_size = 256`, each cycle trains on ~1.28M samples — roughly 25% of the buffer. Positions near the edge of the window get fewer passes than recent ones simply because they'll age out sooner.

**Reload interval:** With `reload_interval = 1000`, fresh worker data enters the training distribution 5 times per cycle. This matters when workers are producing data fast — without reloads, the trainer would miss an entire cycle's worth of fresh data.

**Model version turnover:** Every `steps_per_cycle` steps, the trainer exports a new version. Workers poll for updates and switch. The lag between a new version appearing and workers using it depends on how often workers check (currently every `worker_batch_size` games).

## Data Layout

```
.data/
  models/           # best.onnx, best.pt, best.meta.json, v{N}.onnx snapshots
  training_data/    # version-tagged self-play .npz files (sp_v5_*.npz)
  logs/             # trainer.jsonl, worker-*.jsonl
  elo_state.json    # Elo service state (pair results, ratings)
  elo_rankings.jsonl # Elo ranking history
```
