from __future__ import annotations
"""Main training orchestrator for async distributed training."""

import argparse
import json
import sys
import time

from loguru import logger

from .config import AsyncConfig


def _configure_logging() -> None:
    """Set up loguru for container-friendly logging."""
    logger.remove()  # remove default stderr handler
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | {time:HH:mm:ss} | {message}",
        level="DEBUG",
    )


def cmd_worker(args) -> None:
    """Run the async self-play worker loop."""
    _configure_logging()
    from .worker import run_worker

    cfg = AsyncConfig()
    if args.simulations is not None:
        cfg.num_simulations = args.simulations
    if args.batch_size is not None:
        cfg.worker_batch_size = args.batch_size

    run_worker(cfg)


def cmd_trainer(args) -> None:
    """Run the async continuous trainer loop."""
    _configure_logging()
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

    # Versioned model snapshots
    if cfg.models_dir.exists():
        snapshots = sorted(cfg.models_dir.glob("v*.onnx"))
        if snapshots:
            versions = []
            for s in snapshots:
                try:
                    versions.append(int(s.stem[1:]))
                except ValueError:
                    continue
            if versions:
                print(f"Snapshots: {len(versions)} versions (v{min(versions)}–v{max(versions)})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AlphaZero training pipeline for hexagonal chess"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- Worker ---
    worker_parser = subparsers.add_parser("worker", help="Run self-play worker loop")
    worker_parser.add_argument("--simulations", type=int, default=None, help="MCTS simulations per move")
    worker_parser.add_argument("--batch-size", type=int, default=None, help="Games per batch before flushing")

    # --- Trainer ---
    trainer_parser = subparsers.add_parser("trainer", help="Run continuous trainer loop")
    trainer_parser.add_argument("--simulations", type=int, default=None, help="MCTS simulations for arena")
    trainer_parser.add_argument("--steps", type=int, default=None, help="Training steps per cycle")
    trainer_parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    trainer_parser.add_argument("--arena-games", type=int, default=None, help="Arena games per evaluation")

    # --- Imitation ---
    imit_parser = subparsers.add_parser("imitation", help="Generate minimax imitation data")
    imit_parser.add_argument("--depth", type=int, default=None, help="Minimax search depth")
    imit_parser.add_argument("--num-games", type=int, default=None, help="Games to play")
    imit_parser.add_argument("--random-plies", type=int, default=None, help="Random opening plies")

    # --- Status & progress ---
    subparsers.add_parser("status", help="Show training cluster status")
    subparsers.add_parser("progress", help="Show training progress summary")

    # --- Elo ranking (batch, one-shot) ---
    elo_parser = subparsers.add_parser("elo-ranking", help="Run batch Elo ranking vs baselines")
    elo_parser.add_argument("--versions", type=int, default=10, help="Number of recent model versions to rank")
    elo_parser.add_argument("--games-per-side", type=int, default=3, help="Games per side per matchup")
    elo_parser.add_argument("--simulations", type=int, default=500, help="MCTS simulations per move")

    # --- Elo service (continuous) ---
    elo_svc_parser = subparsers.add_parser("elo-service", help="Run continuous Elo rating service")
    elo_svc_parser.add_argument("--simulations", type=int, default=500, help="MCTS simulations per move")
    elo_svc_parser.add_argument("--max-versions", type=int, default=20, help="Max model versions in pool")
    elo_svc_parser.add_argument("--recompute-interval", type=int, default=10, help="Games between Elo recomputation")
    elo_svc_parser.add_argument("--notify-interval", type=int, default=20, help="Games between Slack notifications")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "worker":
        cmd_worker(args)
    elif args.command == "imitation":
        _configure_logging()
        from .imitation import generate_imitation_data
        cfg = AsyncConfig()
        if args.depth is not None:
            cfg.imitation_depth = args.depth
        if args.num_games is not None:
            cfg.imitation_num_games = args.num_games
        if args.random_plies is not None:
            cfg.imitation_random_plies = args.random_plies
        generate_imitation_data(cfg)
    elif args.command == "trainer":
        cmd_trainer(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "progress":
        from .metrics import print_progress
        print_progress()
    elif args.command == "elo-ranking":
        from .elo_ranking import run_elo_ranking
        run_elo_ranking(
            num_versions=args.versions,
            games_per_side=args.games_per_side,
            simulations=args.simulations,
        )
    elif args.command == "elo-service":
        _configure_logging()
        from .elo_service import run_elo_service
        run_elo_service(
            simulations=args.simulations,
            max_versions=args.max_versions,
            recompute_interval=args.recompute_interval,
            notify_interval=args.notify_interval,
        )


if __name__ == "__main__":
    main()
