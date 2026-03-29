from __future__ import annotations
"""Main training orchestrator: self-play -> train -> export -> arena -> promote."""

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import AsyncConfig, Config, latest_generation


def step_self_play(config: Config) -> tuple[Path, dict]:
    """Run self-play and return (data path, stats)."""
    from .self_play import run_self_play

    print("\n--- Self-Play ---", flush=True)
    return run_self_play(config)


def step_train(config: Config) -> tuple[Path, dict]:
    """Train the network and return (checkpoint path, stats)."""
    from .trainer import train

    print("\n--- Training ---", flush=True)
    return train(config)


def step_export(config: Config) -> Path:
    """Export the latest checkpoint to ONNX."""
    from .export import export_to_onnx

    print("\n--- Export ---", flush=True)

    checkpoint = config.model_dir / "latest.pt"
    output = config.model_dir / "latest.onnx"

    if not checkpoint.exists():
        print(f"No checkpoint found at {checkpoint}, skipping export.")
        return output

    return export_to_onnx(checkpoint, output, config)


def step_arena(config: Config) -> dict:
    """Run arena and return results dict (including 'promoted' bool)."""
    from .arena import run_arena, promote_model

    print("\n--- Arena ---", flush=True)

    results = run_arena(config)
    if results["promoted"]:
        promote_model(config)
    return results


def run_full_loop(config: Config, num_generations: int = 10) -> None:
    """
    Run the full AlphaZero training loop.

    Each generation:
      1. Self-play: generate training data (using prev gen's best model)
      2. Train: update network weights (initializing from prev gen's checkpoint)
      3. Export: convert to ONNX
      4. Arena: evaluate against previous generation's best
      5. Promote: if new model wins enough, save as this gen's best
    """
    start_gen = config.generation
    end_gen = start_gen + num_generations - 1

    print(f"AlphaZero loop: generations {start_gen}-{end_gen}, "
          f"{config.num_self_play_games} games/gen, "
          f"{config.num_simulations} sims/move, "
          f"{config.training_epochs} epochs", flush=True)

    loop_t0 = time.time()

    for gen in range(start_gen, end_gen + 1):
        config.generation = gen
        config.ensure_dirs()

        print(f"\n====== Generation {gen} ======", flush=True)
        t0 = time.time()
        gen_started = datetime.now(timezone.utc).isoformat()

        _, self_play_stats = step_self_play(config)
        _, train_stats = step_train(config)
        step_export(config)

        # Auto-promote on first generation (no previous best to compare against)
        arena_results = None
        if not config.prev_best_model_path.exists():
            from .arena import promote_model
            print("No previous best model — auto-promoting.", flush=True)
            promote_model(config)
            promoted = True
        else:
            arena_results = step_arena(config)
            promoted = arena_results["promoted"]
            if not promoted:
                # Carry forward previous best so the next generation has
                # an unbroken chain to bootstrap from.
                shutil.copy2(config.prev_best_model_path, config.best_model_path)
                if config.prev_best_checkpoint_path.exists():
                    shutil.copy2(config.prev_best_checkpoint_path, config.best_checkpoint_path)
                print(f"Carried forward previous best to {config.best_model_path}", flush=True)

        elapsed = time.time() - t0
        total = time.time() - loop_t0
        status = "PROMOTED" if promoted else "kept"
        print(f"Generation {gen} done in {elapsed:.0f}s ({status}) | total {total:.0f}s", flush=True)

        # Write per-generation metadata
        metadata = {
            "generation": gen,
            "started_at": gen_started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "promoted": promoted,
            "self_play": self_play_stats,
            "training": train_stats,
        }
        if arena_results is not None:
            metadata["arena"] = {
                "new_wins": arena_results["new_wins"],
                "old_wins": arena_results["old_wins"],
                "draws": arena_results["draws"],
                "win_rate": arena_results["win_rate"],
            }
        metadata_path = config.generation_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Wrote {metadata_path}", flush=True)


def cmd_worker(args) -> None:
    """Run the async self-play worker loop."""
    from .worker import run_worker

    cfg = AsyncConfig()
    if args.simulations is not None:
        cfg.num_simulations = args.simulations
    if args.workers is not None:
        cfg.num_self_play_workers = args.workers
    if args.batch_size is not None:
        cfg.worker_batch_size = args.batch_size

    run_worker(cfg)


def cmd_trainer(args) -> None:
    """Run the async continuous trainer loop."""
    from .trainer_loop import run_trainer

    cfg = AsyncConfig()
    if args.simulations is not None:
        cfg.num_simulations = args.simulations
    if args.steps is not None:
        cfg.steps_per_cycle = args.steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.arena_games is not None:
        cfg.arena_games = args.arena_games
    if args.arena_workers is not None:
        cfg.num_arena_workers = args.arena_workers

    run_trainer(cfg)


def cmd_status(args) -> None:
    """Show the status of the async training cluster."""
    cfg = AsyncConfig()

    # Model version
    if cfg.best_meta_path.exists():
        meta = json.loads(cfg.best_meta_path.read_text())
        print(f"Model: v{meta.get('version', '?')} "
              f"(promoted {meta.get('timestamp', '?')})")
        if "win_rate" in meta:
            print(f"  Win rate: {meta['win_rate']:.0%}")
    else:
        print("Model: none (no best model yet)")

    # Training data
    npz_files = list(cfg.training_data_dir.glob("*.npz")) if cfg.training_data_dir.exists() else []
    if npz_files:
        import numpy as np
        total_pos = 0
        for f in npz_files:
            try:
                with np.load(f) as data:
                    total_pos += len(data["outcomes"])
            except (OSError, ValueError, KeyError):
                continue
        print(f"Data: {len(npz_files)} files, {total_pos:,} positions")
    else:
        print("Data: none")

    # Logs
    if cfg.logs_dir.exists():
        for log_file in sorted(cfg.logs_dir.glob("*.jsonl")):
            # Read last line
            lines = log_file.read_text().strip().split("\n")
            if lines:
                last = json.loads(lines[-1])
                ts = last.get("timestamp", "?")
                event = last.get("event", "?")
                print(f"Log {log_file.name}: last event={event} at {ts}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AlphaZero training pipeline for hexagonal chess"
    )
    # Common arguments shared by generational subcommands
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--games", type=int, default=None, help="Self-play games")
    common.add_argument("--simulations", type=int, default=None, help="MCTS simulations")
    common.add_argument("--epochs", type=int, default=None, help="Training epochs")
    common.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    common.add_argument("--workers", type=int, default=None, help="Self-play workers")
    common.add_argument("--generation", type=int, default=None, help="Starting generation number")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- Generational (synchronous) commands ---
    loop_parser = subparsers.add_parser("loop", parents=[common], help="Run the full training loop")
    loop_parser.add_argument("--generations", type=int, default=100, help="Number of generations to run")

    subparsers.add_parser("self-play", parents=[common], help="Run self-play game generation")
    subparsers.add_parser("train", parents=[common], help="Train the network")
    subparsers.add_parser("export", parents=[common], help="Export model to ONNX")
    subparsers.add_parser("arena", parents=[common], help="Run arena evaluation")

    # --- Async distributed commands ---
    worker_parser = subparsers.add_parser("worker", help="Run async self-play worker loop")
    worker_parser.add_argument("--simulations", type=int, default=None, help="MCTS simulations per move")
    worker_parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    worker_parser.add_argument("--batch-size", type=int, default=None, help="Games per batch before flushing")

    trainer_parser = subparsers.add_parser("trainer", help="Run async continuous trainer loop")
    trainer_parser.add_argument("--simulations", type=int, default=None, help="MCTS simulations for arena")
    trainer_parser.add_argument("--steps", type=int, default=None, help="Training steps per cycle")
    trainer_parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    trainer_parser.add_argument("--arena-games", type=int, default=None, help="Arena games per evaluation")
    trainer_parser.add_argument("--arena-workers", type=int, default=None, help="Arena parallel workers")

    subparsers.add_parser("status", help="Show async training cluster status")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # --- Async commands ---
    if args.command == "worker":
        cmd_worker(args)
        return
    elif args.command == "trainer":
        cmd_trainer(args)
        return
    elif args.command == "status":
        cmd_status(args)
        return

    # --- Generational commands ---
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

    # Auto-detect generation: resume after the latest existing one
    if args.generation is not None:
        cfg.generation = args.generation
    else:
        cfg.generation = latest_generation() + 1

    if args.command == "loop":
        run_full_loop(cfg, num_generations=args.generations)
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
