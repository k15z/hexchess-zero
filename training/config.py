"""Hyperparameters and paths for the training pipeline."""

from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    """Return the project root (parent of training/)."""
    return Path(__file__).resolve().parent.parent


def _data_root() -> Path:
    return _project_root() / ".data"


@dataclass
class _BaseConfig:
    """Shared hyperparameters for training and model architecture."""

    # --- MCTS ---
    num_simulations: int = 500

    # --- Self-play ---
    temperature_threshold: int = 60  # after this many moves, temperature → near-zero
    temperature_high: float = 1.0
    temperature_low: float = 0.01
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # --- Training ---
    batch_size: int = 256
    learning_rate: float = 0.001
    l2_regularization: float = 1e-4
    replay_buffer_size: int = 500_000

    # --- Network architecture ---
    num_residual_blocks: int = 6
    num_filters: int = 128
    board_channels: int = 19
    board_height: int = 11
    board_width: int = 11

    # --- Arena ---
    arena_games: int = 50
    win_threshold: float = 0.60
    arena_simulations: int = 500


@dataclass
class AsyncConfig(_BaseConfig):
    """Configuration for the async distributed training pipeline.

    Uses a flat directory structure:
      .data/models/       — best.onnx, best.pt, best.meta.json, v{N}.onnx snapshots
      .data/training_data/ — version-tagged .npz files
      .data/logs/          — append-only JSONL logs
    """

    # --- Async-specific ---
    worker_batch_size: int = 5  # games per flush
    steps_per_cycle: int = 2000  # training steps between arena evaluations
    min_positions_to_train: int = 10_000  # wait for this much data before training

    # --- Paths ---

    @property
    def models_dir(self) -> Path:
        return _data_root() / "models"

    @property
    def training_data_dir(self) -> Path:
        return _data_root() / "training_data"

    @property
    def logs_dir(self) -> Path:
        return _data_root() / "logs"

    @property
    def best_model_path(self) -> Path:
        return self.models_dir / "best.onnx"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.models_dir / "best.pt"

    @property
    def best_meta_path(self) -> Path:
        return self.models_dir / "best.meta.json"

    @property
    def candidate_model_path(self) -> Path:
        return self.models_dir / "candidate.onnx"

    @property
    def candidate_checkpoint_path(self) -> Path:
        return self.models_dir / "candidate.pt"

    def ensure_dirs(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    cfg = AsyncConfig()
    print("Training pipeline configuration:")
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")
    print(f"  models_dir: {cfg.models_dir}")
    print(f"  training_data_dir: {cfg.training_data_dir}")
    print(f"  best_model_path: {cfg.best_model_path}")
