"""Hyperparameters and paths for the AlphaZero-style training pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """Return the project root (parent of training/)."""
    return Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # --- MCTS ---
    num_simulations: int = 250

    # --- Self-play ---
    num_self_play_games: int = 500
    temperature_threshold: int = 30  # after this many moves, temperature → near-zero
    temperature_high: float = 1.0
    temperature_low: float = 0.01
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    num_self_play_workers: int = 10

    # --- Training ---
    training_epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 0.001
    l2_regularization: float = 1e-4
    replay_buffer_size: int = 50_000

    # --- Network architecture ---
    num_residual_blocks: int = 4
    num_filters: int = 64
    board_channels: int = 16
    board_height: int = 11
    board_width: int = 11

    # --- Arena ---
    arena_games: int = 25
    win_threshold: float = 0.60
    arena_simulations: int = 250

    # --- Paths (relative to project root) ---
    model_dir: Path = field(default_factory=lambda: _project_root() / "models")
    data_dir: Path = field(default_factory=lambda: _project_root() / "data" / "self_play")
    checkpoint_dir: Path = field(default_factory=lambda: _project_root() / "models" / "checkpoints")
    best_model_path: Path = field(
        default_factory=lambda: _project_root() / "models" / "best.onnx"
    )
    best_checkpoint_path: Path = field(
        default_factory=lambda: _project_root() / "models" / "best.pt"
    )

    def ensure_dirs(self) -> None:
        """Create all necessary directories."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)




if __name__ == "__main__":
    cfg = Config()
    print("Training pipeline configuration:")
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")
