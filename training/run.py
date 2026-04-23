"""Main training orchestrator for async distributed training."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from .config import AsyncConfig  # noqa: E402 — must load .env before importing config


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
    if args.summary_interval_steps is not None:
        cfg.summary_interval_steps = args.summary_interval_steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    run_trainer(cfg)


def cmd_status(args) -> None:
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
    trainer_parser.add_argument(
        "--summary-interval-steps",
        "--steps",
        dest="summary_interval_steps",
        type=int,
        default=None,
        help="Training steps between trainer summaries",
    )
    trainer_parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")

    # --- Status & progress ---
    subparsers.add_parser("status", help="Show training cluster status")
    subparsers.add_parser("progress", help="Show training progress summary")

    # --- Dashboard ---
    dash_parser = subparsers.add_parser("dashboard", help="Run training dashboard web UI")
    dash_parser.add_argument("--port", type=int, default=8080, help="HTTP port")

    # --- Evaluation service (continuous) ---
    eval_svc_parser = subparsers.add_parser(
        "evaluation-service",
        help="Run continuous candidate evaluation service",
    )
    eval_svc_parser.add_argument(
        "--simulations",
        type=int,
        default=1200,
        help="MCTS simulations per move",
    )
    legacy_eval_parser = subparsers.add_parser(
        "elo-service",
        help="Deprecated alias for evaluation-service",
    )
    legacy_eval_parser.add_argument(
        "--simulations",
        type=int,
        default=1200,
        help="MCTS simulations per move",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "worker":
        cmd_worker(args)
    elif args.command == "trainer":
        cmd_trainer(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "progress":
        from .metrics import print_progress
        print_progress()
    elif args.command == "dashboard":
        from .dashboard import run_dashboard
        run_dashboard(AsyncConfig(), port=args.port)
    elif args.command in {"evaluation-service", "elo-service"}:
        _configure_logging()
        from .evaluation_service import run_evaluation_service

        run_evaluation_service(simulations=args.simulations)


if __name__ == "__main__":
    main()
