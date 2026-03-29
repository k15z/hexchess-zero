"""Hyperparameters and paths for the AlphaZero-style training pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """Return the project root (parent of training/)."""
    return Path(__file__).resolve().parent.parent


def _data_root() -> Path:
    return _project_root() / ".data"


def latest_generation() -> int:
    """Find the highest existing generation number, or 0 if none exist.

    Only counts generations that contain actual files (not empty dirs
    created by ensure_dirs()).
    """
    root = _data_root()
    if not root.exists():
        return 0
    gens = []
    for d in root.iterdir():
        if d.is_dir() and d.name.startswith("gen"):
            try:
                n = int(d.name[3:])
            except ValueError:
                continue
            if any(d.rglob("*.*")):
                gens.append(n)
    return max(gens) if gens else 0


@dataclass
class Config:
    # --- MCTS ---
    num_simulations: int = 500

    # --- Self-play ---
    num_self_play_games: int = 1000
    temperature_threshold: int = 60  # after this many moves, temperature → near-zero
    temperature_high: float = 1.0
    temperature_low: float = 0.01
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    num_self_play_workers: int = 7

    # --- Training ---
    training_epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 0.001
    l2_regularization: float = 1e-4
    replay_buffer_size: int = 500_000

    # --- Network architecture ---
    num_residual_blocks: int = 6
    num_filters: int = 128
    board_channels: int = 16
    board_height: int = 11
    board_width: int = 11

    # --- Arena ---
    arena_games: int = 50
    win_threshold: float = 0.60
    arena_simulations: int = 500
    num_arena_workers: int = 7

    # --- Generation ---
    generation: int = 1

    # --- Current generation paths ---

    @property
    def generation_dir(self) -> Path:
        return _data_root() / f"gen{self.generation}"

    @property
    def model_dir(self) -> Path:
        return self.generation_dir / "model"

    @property
    def data_dir(self) -> Path:
        return self.generation_dir / "data"

    @property
    def best_model_path(self) -> Path:
        return self.model_dir / "best.onnx"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.model_dir / "best.pt"

    # --- Previous generation paths (for bootstrapping) ---
    # At generation 1 these point to gen0 which won't exist — callers
    # check .exists() and fall back to random/scratch initialization.

    @property
    def prev_generation_dir(self) -> Path:
        return _data_root() / f"gen{self.generation - 1}"

    @property
    def prev_best_model_path(self) -> Path:
        return self.prev_generation_dir / "model" / "best.onnx"

    @property
    def prev_best_checkpoint_path(self) -> Path:
        return self.prev_generation_dir / "model" / "best.pt"

    @property
    def all_data_dirs(self) -> list[Path]:
        """Return data dirs from gen 1 through current generation (most recent last)."""
        dirs = []
        for g in range(1, self.generation + 1):
            d = _data_root() / f"gen{g}" / "data"
            if d.exists():
                dirs.append(d)
        return dirs

    def ensure_dirs(self) -> None:
        """Create all necessary directories."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    cfg = Config()
    print("Training pipeline configuration:")
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")
    print(f"  generation_dir: {cfg.generation_dir}")
    print(f"  model_dir: {cfg.model_dir}")
    print(f"  data_dir: {cfg.data_dir}")
