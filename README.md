# Hexchess Zero

Hexagonal chess engine (Glinski variant) in Rust with AlphaZero-style self-play training.

## Install

Prebuilt packages are published to PyPI and npm under the name **`hexchess-zero`**:

```bash
pip install hexchess-zero       # Python (imports as `hexchess`)
npm install hexchess-zero       # JavaScript / WASM
```

Python wheels are available for Linux, macOS, and Windows (3.9–3.13); the npm package bundles the WASM binary. See [`docs/content/docs/usage/`](docs/content/docs/usage/) for API documentation, or the published docs site for interactive examples.

## Structure

- **`engine/`** — Rust engine: board representation (91-cell hex grid, axial coordinates), move generation, MCTS search, minimax search, and neural network inference
- **`training/`** — Async distributed AlphaZero loop: self-play workers, continuous trainer, Elo rating service, dashboard
- **`bindings/wasm/`** — WASM bindings for browser play (uses tract for inference)
- **`bindings/python/`** — PyO3 bindings for the training pipeline (uses ONNX Runtime)
- **`docs/`** — Documentation site with interactive playground (Fumadocs)
- **`k8s/`** — Kubernetes manifests for the production training cluster

## Quick Start

```bash
# Run engine tests
make test

# One-time setup: uv sync + build Python bindings
make setup

# Start training (run in separate terminals — or `make docker-up`)
make worker      # self-play worker (run N of these)
make trainer     # continuous trainer (run 1)
make elo         # Elo rating service (scale via replicas)
make dashboard   # status dashboard

# Quick CLI status
make status

# Run documentation site (includes interactive playground)
make docs-dev
```

## Training Pipeline

The pipeline has three asynchronous components that coordinate purely through **S3** (DigitalOcean Spaces / Cloudflare R2 / any S3-compatible store). Workers can run anywhere with credentials.

1. **Workers** generate self-play games using MCTS + the latest model and flush `.npz` batches to S3.
2. **Trainer** maintains a sliding-window replay buffer over recent self-play data, trains a multi-head network, exports ONNX, and promotes a new model version once enough fresh data has accumulated.
3. **Elo service** plays continuous matches between model versions and baselines, persisting per-game results to S3 and rating models with OpenSkill (Weng-Lin / Plackett-Luce).

On first run (no model exists), the trainer bootstraps by training on minimax-supervised imitation data before switching to self-play. The production model consumes a **22-channel side-to-move tensor** and predicts five heads: main policy, terminal WDL value, moves-left, short-term value, and an auxiliary opponent-policy target.

### Configuration

All parameters live in `training/config.py`. Key dials:

#### MCTS & Self-Play

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_simulations` | 800 | MCTS simulations per move. Higher = stronger play but slower data generation. |
| `temperature_threshold` | 60 | Move number after which temperature drops to `temperature_low`. |
| `temperature_high` | 1.0 | Temperature for early-game moves (before threshold). Higher = more diverse openings. |
| `temperature_low` | 0.35 | Late-game temperature. Lc0-style floor — anything near zero produced one-hot policy targets in 65–70% of positions, killing gradient signal. |
| `dirichlet_alpha` | 0.3 | Dirichlet noise concentration at the MCTS root. |
| `dirichlet_epsilon` | 0.25 | Mixing weight for Dirichlet noise. |
| `worker_batch_size` | 2 | Games per `.npz` flush. |

#### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 256 | Training batch size. |
| `learning_rate` | 5e-4 | SGD learning rate after warmup. |
| `l2_regularization` | 3e-5 | Weight decay. |
| `window_c` / `window_alpha` / `window_beta` | 25,000 / 0.75 / 0.4 | KataGo-style sublinear replay-window parameters. |
| `steps_per_cycle` | 3,000 | Trainer steps per cycle before reloading metrics, buffers, and promotion state. |
| `reload_interval` | 1,000 | Re-scan S3 every N steps within a cycle to pick up fresh worker output. |
| `max_train_steps_per_new_data` | 4.0 | KataGo-style token bucket: target training passes per new data point. Throttles the trainer when workers fall behind. |
| `min_positions_to_start` | 200,000 | Bootstrap gate: self-play training won't start until this many positions exist. |
| `promote_every_new_positions` | 300,000 | Minimum fresh self-play positions required between promotions. |
| `imitation_mix_start` / `imitation_mix_end` / `imitation_mix_decay_end_version` | 0.3 / 0.0 / 5 | Early versions mix in minimax imitation data, then decay to pure self-play. |

#### Bootstrap (Imitation)

These only apply to the initial bootstrap phase when no model exists yet:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `imitation_depth` | 3 | Minimax search depth for generating imitation targets. |
| `imitation_exploration_plies` | 30 | Plies that use softmax sampling (rather than greedy) for opening diversity. |
| `imitation_temperature` | 200.0 | Softmax temperature for converting centipawn scores to policy targets. |
| `imitation_wdl_lambda` | 0.5 | Blend between sigmoid(eval) and final game outcome for the WDL value target. |
| `bootstrap_steps` | 50,000 | Training steps for imitation bootstrap. |
| `bootstrap_learning_rate` | 0.003 | Higher LR for the clean supervised signal (3× self-play LR). |

#### Network Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_residual_blocks` | 8 | Depth of the residual tower. |
| `num_filters` | 144 | Width of convolutional layers. |
| `se_channels` | 32 | Squeeze-and-excitation bottleneck width. |
| `global_pool_channels` | 32 | KataGo-style global pooling width. |
| `global_pool_blocks` | (2, 5) | Which residual blocks get global pooling. |
| `policy_channels` / `aux_policy_channels` / `value_channels` | 4 / 2 / 32 | Conv channels in the main policy, auxiliary policy, and value heads. |
| `board_channels` | 22 | Input tensor channels (piece planes + metadata). Must match `serialization.rs`. |
| `board_height` / `board_width` | 11 | Hex grid embedded in 11×11 rectangle. Fixed by the coordinate system. |

### Key Dynamics

**Worker throughput vs training speed.** Workers produce positions at a rate determined by `num_simulations` and CPU count. The trainer consumes them at a rate determined by `steps_per_cycle`, `batch_size`, and GPU speed. The token bucket (`max_train_steps_per_new_data`) keeps them in lockstep — if the trainer outpaces data production, it sleeps instead of overfitting on stale data.

**Replay window vs cycle length.** The trainer no longer uses a fixed-size replay buffer. Instead it computes a KataGo-style sublinear window from the cumulative self-play count, so early training emphasizes recency while later training keeps more historical coverage without growing linearly forever.

**Model version turnover.** The trainer only promotes after both conditions hold: it has completed a training cycle, and at least `promote_every_new_positions` new self-play positions have landed since the last promotion. Early versions can also be gated against the incumbent before `models/latest.meta.json` is advanced.

## S3 Layout

```
models/
  latest.onnx              # current model for inference
  latest.meta.json         # {"version": N, "timestamp": "...", "positions_at_promote": M}
  checkpoint.pt            # PyTorch training checkpoint
  versions/{N}.onnx        # immutable version snapshots

data/
  selfplay/v{N}/{ts}_{rand}_n{count}.npz   # self-play batches (position count in filename)
  selfplay_traces/v{N}/{game_id}.json      # per-game search trace sidecars
  imitation/{ts}_{rand}_n{count}.npz       # bootstrap minimax batches

state/
  elo.json                 # Elo projection (rebuilt from elo_games/)
  elo_games/{ts}_{rand}.json  # one immutable object per played game (race-free writes)
  trainer_metrics.json     # latest trainer telemetry for the dashboard

heartbeats/
  {hostname}.json          # worker liveness + stats for the dashboard

benchmarks/
  results/{version}.json   # benchmark snapshots consumed by the dashboard
```

Position counts are encoded in each `.npz` filename so the trainer can compute buffer size from an S3 LIST without opening any files.

S3 credentials live in `.env` (gitignored): `BUCKET_NAME`, `ACCESS_KEY`, `SECRET_KEY`, `ENDPOINT`.

## Documentation

Full docs (engine internals, training details, bindings reference, deployment, interactive playground) live in `docs/` and are published from the Fumadocs site. Run `make docs-dev` to view locally.
