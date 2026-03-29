from __future__ import annotations
"""Pit two models against each other to decide whether to promote a new model."""

import time
from pathlib import Path

from .config import Config

try:
    import hexchess
except ImportError:
    hexchess = None


def play_arena_game(
    simulations: int,
    new_goes_first: bool,
    new_model_path: str | None = None,
    old_model_path: str | None = None,
) -> str:
    """
    Play one arena game between two MCTS agents.

    Args:
        simulations: Number of MCTS simulations per move.
        new_goes_first: If True, the "new" model plays White.
        new_model_path: ONNX model for the new agent (None = random).
        old_model_path: ONNX model for the old/best agent (None = random).

    Returns:
        "new", "old", or "draw"
    """
    if hexchess is None:
        raise ImportError("hexchess bindings not available")

    game = hexchess.Game()
    search_new = hexchess.MctsSearch(simulations=simulations, model_path=new_model_path)
    search_old = hexchess.MctsSearch(simulations=simulations, model_path=old_model_path)

    move_count = 0
    max_moves = 500  # safety limit

    while not game.is_game_over() and move_count < max_moves:
        is_white = game.side_to_move() == "white"
        # Decide which search engine to use
        if (is_white and new_goes_first) or (not is_white and not new_goes_first):
            search = search_new
        else:
            search = search_old

        result = search.run(game, temperature=0.1)
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

    new_model = cfg.model_dir / "latest.onnx"
    old_model = cfg.prev_best_model_path
    new_path = str(new_model) if new_model.exists() else None
    old_path = str(old_model) if old_model.exists() else None
    new_label = new_path if new_path else "random"
    old_label = old_path if old_path else "random"
    print(f"Arena: {cfg.arena_games} games, {cfg.arena_simulations} sims/move", flush=True)
    print(f"  new: {new_label}", flush=True)
    print(f"  old: {old_label}", flush=True)

    new_wins = 0
    old_wins = 0
    draws = 0

    t0 = time.time()
    last_log_time = t0
    log_interval = 10  # seconds between progress lines

    for i in range(cfg.arena_games):
        new_goes_first = i % 2 == 0
        result = play_arena_game(
            cfg.arena_simulations, new_goes_first, new_path, old_path,
        )
        game_num = i + 1

        if result == "new":
            new_wins += 1
        elif result == "old":
            old_wins += 1
        else:
            draws += 1

        now = time.time()
        elapsed = now - t0
        total_decided = new_wins + old_wins
        rate = new_wins / total_decided if total_decided > 0 else 0.5
        is_last = game_num == cfg.arena_games
        if game_num == 1 or is_last or (now - last_log_time) >= log_interval:
            last_log_time = now
            print(
                f"  {game_num}/{cfg.arena_games} games "
                f"(new={new_wins} old={old_wins} draw={draws} "
                f"rate={rate:.0%}) {elapsed:.0f}s",
                flush=True,
            )

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
    latest_pt = cfg.model_dir / "latest.pt"

    if latest_onnx.exists():
        import shutil
        shutil.copy2(latest_onnx, cfg.best_model_path)
        print(f"Promoted ONNX model: {latest_onnx} -> {cfg.best_model_path}")

    if latest_pt.exists():
        import shutil
        shutil.copy2(latest_pt, cfg.best_checkpoint_path)
        print(f"Promoted checkpoint: {latest_pt} -> {cfg.best_checkpoint_path}")
