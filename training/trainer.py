from __future__ import annotations
"""Training loop for the policy + value network."""

import sys
import time
from pathlib import Path

# Ensure progress output is visible immediately
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

from .config import Config
from .model import HexChessNet, build_model


class ReplayBuffer(IterableDataset):
    """
    Streams training data from .npz chunk files on disk (KataGo-style).

    Only the shuffle buffer and one chunk are in memory at a time, so usage
    is bounded regardless of how many files exist on disk.
    """

    SHUFFLE_BUFFER_SIZE = 10_000

    def __init__(self, data_dirs: list[Path], max_positions: int = 500_000):
        self.data_dirs = data_dirs
        self.max_positions = max_positions
        self.files, self.total_positions = self._select_files()

    def _select_files(self) -> tuple[list[Path], int]:
        """Pick the most recent files across all data dirs, up to max_positions.

        Files are sorted by parent directory order (oldest generation first,
        newest last), then by filename within each directory.  We select from
        newest first so recent games are always included.
        """
        all_files: list[Path] = []
        for d in self.data_dirs:
            all_files.extend(sorted(d.glob("*.npz")))
        if not all_files:
            return [], 0

        selected = []
        total = 0
        for f in reversed(all_files):
            with np.load(f) as data:
                n = len(data["outcomes"])
            selected.append(f)
            total += n
            if total >= self.max_positions:
                break

        selected.reverse()
        total = min(total, self.max_positions)
        num_dirs = len({f.parent for f in selected})
        print(f"Replay buffer: {total} positions from {len(selected)} files "
              f"across {num_dirs} generation(s) "
              f"(skipped {len(all_files) - len(selected)} older files)")
        return selected, total

    def __iter__(self):
        # Shuffle file order each epoch
        files = self.files.copy()
        random.shuffle(files)

        buf_b, buf_p, buf_o = [], [], []
        positions_remaining = self.max_positions

        for f in files:
            data = np.load(f)
            boards, policies, outcomes = data["boards"], data["policies"], data["outcomes"]

            # Trim last chunk if we'd exceed max_positions
            if len(boards) > positions_remaining:
                boards = boards[-positions_remaining:]
                policies = policies[-positions_remaining:]
                outcomes = outcomes[-positions_remaining:]
            positions_remaining -= len(boards)

            # Add chunk to shuffle buffer
            buf_b.extend(boards)
            buf_p.extend(policies)
            buf_o.extend(outcomes)

            # When buffer is full, shuffle and drain half
            while len(buf_b) >= self.SHUFFLE_BUFFER_SIZE:
                perm = list(range(len(buf_b)))
                random.shuffle(perm)
                drain = len(buf_b) // 2
                for j in perm[:drain]:
                    yield (torch.from_numpy(buf_b[j]),
                           torch.from_numpy(buf_p[j]),
                           torch.tensor(buf_o[j], dtype=torch.float32))
                keep = perm[drain:]
                buf_b = [buf_b[j] for j in keep]
                buf_p = [buf_p[j] for j in keep]
                buf_o = [buf_o[j] for j in keep]

            if positions_remaining <= 0:
                break

        # Flush remaining buffer
        perm = list(range(len(buf_b)))
        random.shuffle(perm)
        for j in perm:
            yield (torch.from_numpy(buf_b[j]),
                   torch.from_numpy(buf_p[j]),
                   torch.tensor(buf_o[j], dtype=torch.float32))


def train(config: Config | None = None) -> tuple[Path, dict]:
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

    # Load data from current and previous generations
    dataset = ReplayBuffer(cfg.all_data_dirs, max_positions=cfg.replay_buffer_size)
    if dataset.total_positions == 0:
        print("No training data available. Run self-play first.")
        return cfg.best_checkpoint_path

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
    )

    # Build or load model
    model = build_model(cfg).to(device)
    if cfg.prev_best_checkpoint_path.exists():
        print(f"Loading checkpoint from previous generation: {cfg.prev_best_checkpoint_path}")
        model.load_state_dict(torch.load(cfg.prev_best_checkpoint_path, map_location=device))

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.l2_regularization,
    )

    # Training loop
    model.train()
    total_batches = (dataset.total_positions + cfg.batch_size - 1) // cfg.batch_size
    final_losses = {}
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
            avg_policy = epoch_policy_loss / num_batches
            avg_value = epoch_value_loss / num_batches
            avg_total = epoch_total_loss / num_batches
            print(
                f"Epoch {epoch + 1}/{cfg.training_epochs} "
                f"[{elapsed:.1f}s] "
                f"policy_loss={avg_policy:.4f} "
                f"value_loss={avg_value:.4f} "
                f"total_loss={avg_total:.4f}",
                flush=True,
            )
            final_losses = {
                "policy_loss": round(avg_policy, 4),
                "value_loss": round(avg_value, 4),
                "total_loss": round(avg_total, 4),
            }

    # Save checkpoint
    checkpoint_path = cfg.model_dir / "latest.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    stats = {
        "epochs": cfg.training_epochs,
        "positions": dataset.total_positions,
        "final_losses": final_losses,
        "device": str(device),
    }

    return checkpoint_path, stats


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

    train(cfg)  # stats discarded in CLI mode
