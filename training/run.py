"""Main training orchestrator for async distributed training."""

from __future__ import annotations

import argparse
import sys
from typing import cast

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from .config import AsyncConfig  # noqa: E402 — must load .env before importing config


class WorkerArgs(argparse.Namespace):
    simulations: int | None
    batch_size: int | None


class TrainerArgs(argparse.Namespace):
    steps: int | None
    batch_size: int | None


class DashboardArgs(argparse.Namespace):
    port: int


class EloServiceArgs(argparse.Namespace):
    simulations: int
    max_versions: int
    notify_interval: int


def _configure_logging() -> None:
    """Set up loguru for container-friendly logging."""
    logger.remove()  # remove default stderr handler
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | {time:HH:mm:ss} | {message}",
        level="DEBUG",
    )


def cmd_worker(args: WorkerArgs) -> None:
    """Run the async self-play worker loop."""
    _configure_logging()
    from .worker import run_worker

    cfg = AsyncConfig()
    if args.simulations is not None:
        cfg.num_simulations = args.simulations
    if args.batch_size is not None:
        cfg.worker_batch_size = args.batch_size

    run_worker(cfg)


def cmd_trainer(args: TrainerArgs) -> None:
    """Run the async continuous trainer loop."""
    _configure_logging()
    from .trainer_loop import run_trainer

    cfg = AsyncConfig()
    if args.steps is not None:
        cfg.steps_per_cycle = args.steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    run_trainer(cfg)


def cmd_status(_args: argparse.Namespace) -> None:
    """Show the status of the training pipeline."""
    from .metrics import print_progress
    print_progress()


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
    trainer_parser.add_argument("--steps", type=int, default=None, help="Training steps per cycle")
    trainer_parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")

    # --- Status & progress ---
    subparsers.add_parser("status", help="Show training cluster status")
    subparsers.add_parser("progress", help="Show training progress summary")

    # --- Dashboard ---
    dash_parser = subparsers.add_parser("dashboard", help="Run training dashboard web UI")
    dash_parser.add_argument("--port", type=int, default=8080, help="HTTP port")

    # --- Elo service (continuous) ---
    elo_svc_parser = subparsers.add_parser("elo-service", help="Run continuous Elo rating service")
    elo_svc_parser.add_argument("--simulations", type=int, default=800, help="MCTS simulations per move")
    elo_svc_parser.add_argument("--max-versions", type=int, default=20, help="Max model versions in pool")
    elo_svc_parser.add_argument("--notify-interval", type=int, default=20, help="Games between Slack notifications")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "worker":
        cmd_worker(cast(WorkerArgs, args))
    elif args.command == "trainer":
        cmd_trainer(cast(TrainerArgs, args))
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "progress":
        from .metrics import print_progress
        print_progress()
    elif args.command == "dashboard":
        from .dashboard import run_dashboard
        dash_args = cast(DashboardArgs, args)
        run_dashboard(AsyncConfig(), port=dash_args.port)
    elif args.command == "elo-service":
        _configure_logging()
        from .elo_service import run_elo_service
        elo_args = cast(EloServiceArgs, args)
        run_elo_service(
            simulations=elo_args.simulations,
            max_versions=elo_args.max_versions,
            notify_interval=elo_args.notify_interval,
        )


if __name__ == "__main__":
    main()
