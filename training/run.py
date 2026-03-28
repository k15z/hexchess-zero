from __future__ import annotations
"""Main training orchestrator: self-play -> train -> export -> arena -> promote."""

import argparse
import time
from pathlib import Path

from .config import Config


def step_self_play(config: Config) -> Path:
    """Run self-play and return path to the generated data."""
    from .self_play import run_self_play

    print("=" * 60)
    print("STEP: Self-Play")
    print("=" * 60)
    return run_self_play(config)


def step_train(config: Config) -> Path:
    """Train the network and return path to the checkpoint."""
    from .trainer import train

    print("=" * 60)
    print("STEP: Training")
    print("=" * 60)
    return train(config)


def step_export(config: Config) -> Path:
    """Export the latest checkpoint to ONNX."""
    from .export import export_to_onnx

    print("=" * 60)
    print("STEP: Export to ONNX")
    print("=" * 60)

    checkpoint = config.checkpoint_dir / "latest.pt"
    output = config.model_dir / "latest.onnx"

    if not checkpoint.exists():
        print(f"No checkpoint found at {checkpoint}, skipping export.")
        return output

    return export_to_onnx(checkpoint, output, config)


def step_arena(config: Config) -> bool:
    """Run arena and return whether the new model was promoted."""
    from .arena import run_arena, promote_model

    print("=" * 60)
    print("STEP: Arena")
    print("=" * 60)

    results = run_arena(config)
    if results["promoted"]:
        promote_model(config)
    return results["promoted"]


def run_full_loop(config: Config, num_iterations: int = 10) -> None:
    """
    Run the full AlphaZero training loop.

    Each iteration:
      1. Self-play: generate training data
      2. Train: update network weights
      3. Export: convert to ONNX
      4. Arena: evaluate against current best
      5. Promote: if new model wins enough
    """
    print("Starting AlphaZero training loop")
    print(f"  Iterations: {num_iterations}")
    print(f"  Self-play games per iteration: {config.num_self_play_games}")
    print(f"  MCTS simulations: {config.num_simulations}")
    print(f"  Training epochs: {config.training_epochs}")
    print(f"  Arena games: {config.arena_games}")
    print()

    config.ensure_dirs()

    for iteration in range(1, num_iterations + 1):
        print(f"\n{'#' * 60}")
        print(f"# ITERATION {iteration}/{num_iterations}")
        print(f"{'#' * 60}\n")

        t0 = time.time()

        # 1. Self-play
        step_self_play(config)

        # 2. Train
        step_train(config)

        # 3. Export
        step_export(config)

        # 4. Arena + Promote
        promoted = step_arena(config)

        elapsed = time.time() - t0
        status = "PROMOTED" if promoted else "kept current best"
        print(f"\nIteration {iteration} complete in {elapsed:.0f}s ({status})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AlphaZero training pipeline for hexagonal chess"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Full loop
    loop_parser = subparsers.add_parser("loop", help="Run the full training loop")
    loop_parser.add_argument(
        "--iterations", type=int, default=10, help="Number of training iterations"
    )

    # Individual steps
    subparsers.add_parser("self-play", help="Run self-play game generation")
    subparsers.add_parser("train", help="Train the network")
    subparsers.add_parser("export", help="Export model to ONNX")
    subparsers.add_parser("arena", help="Run arena evaluation")

    # Common arguments
    parser.add_argument("--games", type=int, default=None, help="Self-play games")
    parser.add_argument("--simulations", type=int, default=None, help="MCTS simulations")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--workers", type=int, default=None, help="Self-play workers")

    args = parser.parse_args()

    # Build config with overrides
    cfg = Config()
    if args.games is not None:
        cfg.num_self_play_games = args.games
    if args.simulations is not None:
        cfg.num_simulations = args.simulations
    if args.epochs is not None:
        cfg.training_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.workers is not None:
        cfg.num_self_play_workers = args.workers

    if args.command is None:
        parser.print_help()
        return

    if args.command == "loop":
        run_full_loop(cfg, num_iterations=args.iterations)
    elif args.command == "self-play":
        step_self_play(cfg)
    elif args.command == "train":
        step_train(cfg)
    elif args.command == "export":
        step_export(cfg)
    elif args.command == "arena":
        step_arena(cfg)


if __name__ == "__main__":
    main()
