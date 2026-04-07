"""Hyperparameters and paths for the training pipeline."""

from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    """Return the project root (parent of training/)."""
    return Path(__file__).resolve().parent.parent


def _cache_root() -> Path:
    """Local cache for downloaded S3 objects (models, .npz files)."""
    return _project_root() / ".cache"


@dataclass
class _BaseConfig:
    """Shared hyperparameters for training and model architecture."""

    # --- MCTS ---
    num_simulations: int = 800

    # --- Self-play ---
    temperature_threshold: int = 60  # after this many moves, temperature drops to temperature_low
    temperature_high: float = 1.0
    temperature_low: float = 0.35  # Lc0-style floor — 0.01 produced one-hot targets in 65-70% of positions
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # --- Training ---
    batch_size: int = 256
    learning_rate: float = 0.001
    momentum: float = 0.9
    l2_regularization: float = 1e-4
    replay_buffer_size: int = 1_000_000

    # --- Network architecture ---
    num_residual_blocks: int = 10
    num_filters: int = 192
    se_channels: int = 48  # SE bottleneck width (Leela-style)
    global_pool_channels: int = 32  # KataGo-style global pooling width
    global_pool_blocks: tuple[int, ...] = (3, 6)  # which blocks get global pooling
    policy_channels: int = 8  # conv channels in policy head (wider = better move prediction)
    value_channels: int = 32  # conv channels in value head (pooled via global avg)
    board_channels: int = 19
    board_height: int = 11
    board_width: int = 11


@dataclass
class AsyncConfig(_BaseConfig):
    """Configuration for the async distributed training pipeline.

    All shared state lives in S3 (DigitalOcean Spaces / Cloudflare R2 / etc).
    Local .cache/ directory holds downloaded files for ONNX Runtime and np.load.
    """

    # --- Async-specific ---
    worker_batch_size: int = 5  # games per flush
    steps_per_cycle: int = 1000  # training steps per cycle
    reload_interval: int = 1000  # reload buffer from disk every N steps for fresh data
    max_train_steps_per_new_data: float = 4.0  # target passes per data point (KataGo-style bucket)
    min_positions_to_start: int = 1_000_000  # bootstrap gate: #15 found ~850k needed to beat heuristic

    # --- Imitation bootstrap ---
    imitation_depth: int = 5  # minimax search depth for imitation targets
    imitation_exploration_plies: int = 30  # plies using softmax sampling for diversity
    imitation_temperature: float = 200.0  # softmax temperature for policy targets and exploration sampling
    imitation_wdl_lambda: float = 0.5  # blend: λ*sigmoid(eval) + (1-λ)*game_outcome
    bootstrap_steps: int = 50_000  # training steps for imitation bootstrap (before self-play)
    bootstrap_learning_rate: float = 0.003  # higher LR for clean supervised signal (3x self-play LR)

    # --- Local cache ---

    @property
    def cache_dir(self) -> Path:
        return _cache_root()

    @property
    def model_cache_dir(self) -> Path:
        return _cache_root() / "models"

    @property
    def data_cache_dir(self) -> Path:
        return _cache_root() / "data"

    def ensure_cache_dirs(self) -> None:
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    cfg = AsyncConfig()
    print("Training pipeline configuration:")
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")
