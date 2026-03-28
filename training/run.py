from __future__ import annotations
"""Main training orchestrator: self-play -> train -> export -> arena -> promote."""

import argparse
import time
from pathlib import Path

from .config import Config


def step_self_play(config: Config) -> Path:
    """Run self-play and return path to the generated data."""
    from .self_play import run_self_play

    print("\n--- Self-Play ---", flush=True)
    return run_self_play(config)


def step_train(config: Config) -> Path:
    """Train the network and return path to the checkpoint."""
    from .trainer import train

    print("\n--- Training ---", flush=True)
    return train(config)


def step_export(config: Config) -> Path:
    """Export the latest checkpoint to ONNX."""
    from .export import export_to_onnx

    print("\n--- Export ---", flush=True)

    checkpoint = config.checkpoint_dir / "latest.pt"
    output = config.model_dir / "latest.onnx"

    if not checkpoint.exists():
        print(f"No checkpoint found at {checkpoint}, skipping export.")
        return output

    return export_to_onnx(checkpoint, output, config)


def step_arena(config: Config) -> bool:
    """Run arena and return whether the new model was promoted."""
    from .arena import run_arena, promote_model

    print("\n--- Arena ---", flush=True)

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
    print(f"AlphaZero loop: {num_iterations} iterations, "
          f"{config.num_self_play_games} games/iter, "
          f"{config.num_simulations} sims/move, "
          f"{config.training_epochs} epochs", flush=True)

    config.ensure_dirs()
    loop_t0 = time.time()

    for iteration in range(1, num_iterations + 1):
        print(f"\n====== Iteration {iteration}/{num_iterations} ======", flush=True)
        t0 = time.time()

        step_self_play(config)
        step_train(config)
        step_export(config)

        # Auto-promote on first iteration when no best model exists
        if not config.best_model_path.exists():
            from .arena import promote_model
            print("No best model yet — auto-promoting.", flush=True)
            promote_model(config)
            promoted = True
        else:
            promoted = step_arena(config)

        elapsed = time.time() - t0
        total = time.time() - loop_t0
        status = "PROMOTED" if promoted else "kept"
        print(f"Iteration {iteration} done in {elapsed:.0f}s ({status}) | total {total:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AlphaZero training pipeline for hexagonal chess"
    )
    # Common arguments shared by all subcommands
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--games", type=int, default=None, help="Self-play games")
    common.add_argument("--simulations", type=int, default=None, help="MCTS simulations")
    common.add_argument("--epochs", type=int, default=None, help="Training epochs")
    common.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    common.add_argument("--workers", type=int, default=None, help="Self-play workers")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    loop_parser = subparsers.add_parser("loop", parents=[common], help="Run the full training loop")
    loop_parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations")

    subparsers.add_parser("self-play", parents=[common], help="Run self-play game generation")
    subparsers.add_parser("train", parents=[common], help="Train the network")
    subparsers.add_parser("export", parents=[common], help="Export model to ONNX")
    subparsers.add_parser("arena", parents=[common], help="Run arena evaluation")

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
