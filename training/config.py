"""Hyperparameters and paths for the training pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_run_id() -> str:
    """Run identifier for logging/metrics grouping (notes/13 §7.1, §8)."""
    return os.environ.get("RUN_ID", "dev")


def _project_root() -> Path:
    """Return the project root (parent of training/)."""
    return Path(__file__).resolve().parent.parent


def _cache_root() -> Path:
    """Local cache for downloaded S3 objects (models, .npz files)."""
    return _project_root() / ".cache"


@dataclass
class _BaseConfig:
    """Shared hyperparameters for training and model architecture."""

    # --- Run identity ---
    run_id: str = field(default_factory=_default_run_id)

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
    learning_rate: float = 1e-3  # base LR after warmup (plan §4.3)
    momentum: float = 0.9
    l2_regularization: float = 3e-5  # KataGo weight decay (plan §4.3)
    grad_clip_norm: float = 5.0  # plan §4.3
    lr_warmup_steps: int = 2_000  # plan §4.3
    promote_every_new_positions: int = 500_000  # plan §4.5
    runtime_health_check_every_steps: int = 500

    # --- Replay window (sublinear KataGo formula, plan §4.1) ---
    window_c: int = 25_000
    window_alpha: float = 0.75
    window_beta: float = 0.4

    # --- SWA (plan §4.4) ---
    swa_snapshot_every_samples: int = 250_000
    swa_buffer_size: int = 4
    swa_ema_decay: float = 0.75

    # --- Playout Cap Randomization (plan §1.4/§5.4) ---
    pcr_p_full: float = 0.25
    pcr_n_full: int = 800
    pcr_n_fast: int = 160

    # --- Resignation (plan §1.10) ---
    resign_threshold: float = 0.05
    resign_streak: int = 5
    resign_skip_prob: float = 0.10

    # --- Gating (plan §4.6) ---
    gating_enabled_first_n_versions: int = 5
    gating_games: int = 200
    gating_win_threshold: float = 0.50
    gating_max_failures: int = 3

    # --- Network architecture ---
    # Sized to ~5M params: small enough to iterate fast on the cluster + Mac
    # Studio, big enough to learn the ~4200-move policy without bottlenecking.
    num_residual_blocks: int = 8
    num_filters: int = 144
    se_channels: int = 32
    global_pool_channels: int = 32
    global_pool_blocks: tuple[int, ...] = (2, 5)
    policy_channels: int = 4
    aux_policy_channels: int = 2  # narrower opponent-reply head
    value_channels: int = 32
    board_channels: int = 22
    board_height: int = 11
    board_width: int = 11


@dataclass
class AsyncConfig(_BaseConfig):
    """Configuration for the async distributed training pipeline.

    All shared state lives in S3 (DigitalOcean Spaces / Cloudflare R2 / etc).
    Local .cache/ directory holds downloaded files for ONNX Runtime and np.load.
    """

    # --- Async-specific ---
    worker_batch_size: int = 2  # games per flush (small so bootstrap data lands quickly)
    steps_per_cycle: int = 1000  # training steps per cycle
    reload_interval: int = 1000  # reload buffer from disk every N steps for fresh data
    max_train_steps_per_new_data: float = 4.0  # target passes per data point (KataGo-style bucket)
    min_positions_to_start: int = 200_000  # bootstrap gate — workers can generate more after the trainer starts

    # --- Imitation bootstrap ---
    # Depth 5 is prohibitively slow on Glinski (>10 min per game observed).
    # Depth 3 produces still-reasonable targets (captures, obvious tactics)
    # and is ~30x faster, which lets us cycle through bootstrap in hours
    # instead of days.
    imitation_depth: int = 3
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

    def validate(self) -> None:
        """Sanity-check config values; raise ``ValueError`` on bad input."""
        errors: list[str] = []

        def _check(cond: bool, msg: str) -> None:
            if not cond:
                errors.append(msg)

        _check(self.batch_size > 0, "batch_size must be > 0")
        _check(self.learning_rate > 0, "learning_rate must be > 0")
        _check(self.l2_regularization >= 0, "l2_regularization must be >= 0")
        _check(self.grad_clip_norm > 0, "grad_clip_norm must be > 0")
        _check(self.lr_warmup_steps >= 0, "lr_warmup_steps must be >= 0")
        _check(self.runtime_health_check_every_steps > 0,
               "runtime_health_check_every_steps must be > 0")
        _check(self.promote_every_new_positions > 0,
               "promote_every_new_positions must be > 0")
        _check(self.num_simulations > 0, "num_simulations must be > 0")

        _check(self.window_c > 0, "window_c must be > 0")
        _check(self.window_alpha > 0, "window_alpha must be > 0")
        _check(0 < self.window_beta, "window_beta must be > 0")

        _check(0 < self.pcr_p_full <= 1.0, "pcr_p_full must be in (0, 1]")
        _check(self.pcr_n_full > 0, "pcr_n_full must be > 0")
        _check(self.pcr_n_fast > 0, "pcr_n_fast must be > 0")
        _check(self.pcr_n_fast <= self.pcr_n_full,
               "pcr_n_fast must be <= pcr_n_full")

        _check(0.0 < self.resign_threshold < 1.0,
               "resign_threshold must be in (0, 1)")
        _check(self.resign_streak >= 1, "resign_streak must be >= 1")
        _check(0.0 <= self.resign_skip_prob <= 1.0,
               "resign_skip_prob must be in [0, 1]")

        _check(self.gating_games > 0, "gating_games must be > 0")
        _check(0.0 < self.gating_win_threshold <= 1.0,
               "gating_win_threshold must be in (0, 1]")

        _check(self.swa_buffer_size >= 1, "swa_buffer_size must be >= 1")
        _check(0.0 < self.swa_ema_decay <= 1.0,
               "swa_ema_decay must be in (0, 1]")
        _check(self.swa_snapshot_every_samples > 0,
               "swa_snapshot_every_samples must be > 0")

        _check(self.max_train_steps_per_new_data > 0,
               "max_train_steps_per_new_data must be > 0")
        _check(bool(self.run_id), "run_id must be non-empty")

        if errors:
            raise ValueError("Invalid AsyncConfig: " + "; ".join(errors))


if __name__ == "__main__":
    cfg = AsyncConfig()
    print("Training pipeline configuration:")
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")
