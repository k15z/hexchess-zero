from __future__ import annotations
"""Pit two models against each other to decide whether to promote a new model."""

import time
from pathlib import Path

import numpy as np

from .config import Config

try:
    import hexchess
except ImportError:
    hexchess = None


def play_arena_game(
    simulations: int,
    new_goes_first: bool,
) -> str:
    """
    Play one arena game between two random-evaluator MCTS agents.

    In the initial version (no trained models), both sides use the engine's
    built-in RandomEvaluator via hexchess.MctsSearch. Once we have ONNX models,
    this will be extended to load different models for each side.

    Args:
        simulations: Number of MCTS simulations per move.
        new_goes_first: If True, the "new" model plays White.

    Returns:
        "new", "old", or "draw"
    """
    if hexchess is None:
        raise ImportError("hexchess bindings not available")

    game = hexchess.Game()
    search_new = hexchess.MctsSearch(simulations=simulations)
    search_old = hexchess.MctsSearch(simulations=simulations)

    move_count = 0
    max_moves = 300  # safety limit

    while not game.is_game_over() and move_count < max_moves:
        is_white = game.side_to_move() == "white"
        # Decide which search engine to use
        if (is_white and new_goes_first) or (not is_white and not new_goes_first):
            search = search_new
        else:
            search = search_old

        result = search.run(game)
        best_move = result["best_move"]

        game.apply_move(
            best_move["from_q"],
            best_move["from_r"],
            best_move["to_q"],
            best_move["to_r"],
            best_move.get("promotion"),
        )
        move_count += 1

    status = game.status()

    if status == "checkmate_white":
        winner_is_white = True
    elif status == "checkmate_black":
        winner_is_white = False
    else:
        return "draw"

    # Map winner color back to new/old
    if new_goes_first:
        return "new" if winner_is_white else "old"
    else:
        return "new" if not winner_is_white else "old"


def run_arena(config: Config | None = None) -> dict:
    """
    Run an arena match between the new and current-best models.

    Returns a dict with results: {new_wins, old_wins, draws, win_rate, promoted}.
    """
    cfg = config or Config()

    print(f"Arena: playing {cfg.arena_games} games...")

    new_wins = 0
    old_wins = 0
    draws = 0

    t0 = time.time()
    for i in range(cfg.arena_games):
        new_goes_first = (i % 2 == 0)

        result = play_arena_game(
            simulations=cfg.arena_simulations,
            new_goes_first=new_goes_first,
        )

        if result == "new":
            new_wins += 1
        elif result == "old":
            old_wins += 1
        else:
            draws += 1

        total_decided = new_wins + old_wins
        rate = new_wins / total_decided if total_decided > 0 else 0.5
        elapsed = time.time() - t0
        print(
            f"  {i+1}/{cfg.arena_games} games "
            f"(new={new_wins} old={old_wins} draw={draws} "
            f"rate={rate:.0%}) {elapsed:.0f}s",
            end="\r", flush=True,
        )

    print(flush=True)  # newline
    total_decided = new_wins + old_wins
    win_rate = new_wins / total_decided if total_decided > 0 else 0.5
    promoted = win_rate >= cfg.win_threshold

    print(f"Arena result: new={new_wins} old={old_wins} draw={draws} rate={win_rate:.0%} -> {'PROMOTED' if promoted else 'kept current'}", flush=True)

    return {
        "new_wins": new_wins,
        "old_wins": old_wins,
        "draws": draws,
        "win_rate": win_rate,
        "promoted": promoted,
    }


def promote_model(config: Config | None = None) -> None:
    """Copy the latest checkpoint to the best model slot."""
    cfg = config or Config()
    latest_onnx = cfg.model_dir / "latest.onnx"
    latest_pt = cfg.checkpoint_dir / "latest.pt"

    if latest_onnx.exists():
        import shutil
        shutil.copy2(latest_onnx, cfg.best_model_path)
        print(f"Promoted ONNX model: {latest_onnx} -> {cfg.best_model_path}")

    if latest_pt.exists():
        import shutil
        shutil.copy2(latest_pt, cfg.best_checkpoint_path)
        print(f"Promoted checkpoint: {latest_pt} -> {cfg.best_checkpoint_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run arena match between models")
    parser.add_argument("--games", type=int, default=None, help="Number of arena games")
    parser.add_argument("--simulations", type=int, default=None, help="MCTS simulations per move")
    args = parser.parse_args()

    cfg = Config()
    if args.games is not None:
        cfg.arena_games = args.games
    if args.simulations is not None:
        cfg.arena_simulations = args.simulations

    results = run_arena(cfg)
    if results["promoted"]:
        promote_model(cfg)
