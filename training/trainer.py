from __future__ import annotations
"""Training loop for the policy + value network."""

import sys
import time
from pathlib import Path

# Ensure progress output is visible immediately
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .model import HexChessNet, build_model


class ReplayBuffer(Dataset):
    """
    Loads training data from .npz files on disk.

    Each .npz contains:
        boards:   (N, 16, 11, 11) float32
        policies: (N, num_move_indices) float32
        outcomes: (N,) float32
    """

    def __init__(self, data_dir: Path, max_positions: int = 50_000):
        self.max_positions = max_positions
        self.boards_arr = np.empty(0)
        self.policies_arr = np.empty(0)
        self.outcomes_arr = np.empty(0)
        self._load_data(data_dir)

    def _load_data(self, data_dir: Path) -> None:
        """Load all .npz files from the data directory."""
        npz_files = sorted(data_dir.glob("*.npz"))
        if not npz_files:
            print(f"Warning: no .npz files found in {data_dir}")
            return

        all_boards = []
        all_policies = []
        all_outcomes = []

        for f in npz_files:
            data = np.load(f)
            all_boards.append(data["boards"])
            all_policies.append(data["policies"])
            all_outcomes.append(data["outcomes"])

        boards = np.concatenate(all_boards, axis=0)
        policies = np.concatenate(all_policies, axis=0)
        outcomes = np.concatenate(all_outcomes, axis=0)

        # Keep only the most recent positions if buffer is too large
        if len(boards) > self.max_positions:
            boards = boards[-self.max_positions:]
            policies = policies[-self.max_positions:]
            outcomes = outcomes[-self.max_positions:]

        self.boards_arr = boards
        self.policies_arr = policies
        self.outcomes_arr = outcomes

        print(f"Loaded {len(self.boards_arr)} positions from {len(npz_files)} files")

    def __len__(self) -> int:
        return len(self.boards_arr)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Data is already float32 from self-play, no copy needed
        board = torch.from_numpy(self.boards_arr[idx])
        policy = torch.from_numpy(self.policies_arr[idx])
        outcome = torch.tensor(self.outcomes_arr[idx], dtype=torch.float32)
        return board, policy, outcome


def train(config: Config | None = None) -> Path:
    """
    Run the training loop.

    Returns the path to the saved checkpoint.
    """
    cfg = config or Config()
    cfg.ensure_dirs()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    # Load data
    dataset = ReplayBuffer(cfg.data_dir, max_positions=cfg.replay_buffer_size)
    if len(dataset) == 0:
        print("No training data available. Run self-play first.")
        return cfg.best_checkpoint_path

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    # Build or load model
    model = build_model(cfg).to(device)
    if cfg.best_checkpoint_path.exists():
        print(f"Loading existing checkpoint: {cfg.best_checkpoint_path}")
        model.load_state_dict(torch.load(cfg.best_checkpoint_path, map_location=device))

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.l2_regularization,
    )

    # Training loop
    model.train()
    total_batches = (len(dataset) + cfg.batch_size - 1) // cfg.batch_size
    for epoch in range(cfg.training_epochs):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for boards, policies, outcomes in dataloader:
            boards = boards.to(device)
            policies = policies.to(device)
            outcomes = outcomes.to(device)

            # Forward pass
            pred_policy, pred_value = model(boards)
            pred_value = pred_value.squeeze(-1)

            # Policy loss: cross-entropy with MCTS policy as soft target
            # Use KL divergence (equivalent to cross-entropy when target is fixed)
            log_probs = torch.log_softmax(pred_policy, dim=1)
            policy_loss = -torch.sum(policies * log_probs, dim=1).mean()

            # Value loss: MSE
            value_loss = nn.functional.mse_loss(pred_value, outcomes)

            # Total loss (L2 regularization handled by weight_decay in optimizer)
            total_loss = policy_loss + value_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_total_loss += total_loss.item()
            num_batches += 1

            if num_batches % 20 == 0 or num_batches == total_batches:
                print(
                    f"  batch {num_batches}/{total_batches} "
                    f"loss={total_loss.item():.4f}",
                    flush=True,
                )

        elapsed = time.time() - t0
        if num_batches > 0:
            print(
                f"Epoch {epoch + 1}/{cfg.training_epochs} "
                f"[{elapsed:.1f}s] "
                f"policy_loss={epoch_policy_loss / num_batches:.4f} "
                f"value_loss={epoch_value_loss / num_batches:.4f} "
                f"total_loss={epoch_total_loss / num_batches:.4f}",
                flush=True,
            )

    # Save checkpoint
    checkpoint_path = cfg.checkpoint_dir / "latest.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    return checkpoint_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the policy + value network")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.training_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr

    train(cfg)
