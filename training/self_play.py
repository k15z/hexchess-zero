from __future__ import annotations
"""Self-play game generation using MCTS via the hexchess engine."""

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import Config

try:
    import hexchess
except ImportError:
    hexchess = None


def play_one_game(
    search: "hexchess.MctsSearch",
    config: Config,
) -> tuple[str, list[dict]]:
    """
    Play a single self-play game and return a list of training samples.

    Each sample is a dict with:
        board: np.ndarray of shape (16, 11, 11)
        policy: np.ndarray of shape (num_move_indices,)
        outcome: float  (filled in after game ends)
    """
    game = hexchess.Game()

    samples = []
    move_number = 0

    while not game.is_game_over():
        board_tensor = hexchess.encode_board(game)

        if move_number < config.temperature_threshold:
            temperature = config.temperature_high
        else:
            temperature = config.temperature_low

        result = search.run(
            game,
            temperature=temperature,
            dirichlet_epsilon=config.dirichlet_epsilon,
            dirichlet_alpha=config.dirichlet_alpha,
        )
        policy = result["policy"]

        side = game.side_to_move()
        samples.append(
            {
                "board": board_tensor,
                "policy": policy,
                "side": side,
            }
        )

        best = result["best_move"]
        game.apply_move(
            best["from_q"], best["from_r"],
            best["to_q"], best["to_r"],
            best.get("promotion"),
        )
        move_number += 1

    # Determine game outcome as WDL targets [win, draw, loss]
    status = game.status()
    if status == "checkmate_white":
        wdl_white = [1.0, 0.0, 0.0]  # white won
    elif status == "checkmate_black":
        wdl_white = [0.0, 0.0, 1.0]  # white lost
    else:
        wdl_white = [0.0, 1.0, 0.0]  # draw

    # Fill in WDL outcome from each side's perspective
    for sample in samples:
        if sample["side"] == "white":
            sample["outcome"] = np.array(wdl_white, dtype=np.float32)
        else:
            # Flip W and L for black's perspective
            sample["outcome"] = np.array([wdl_white[2], wdl_white[1], wdl_white[0]], dtype=np.float32)
        del sample["side"]

    return status, samples


def _flush_samples(samples: list[dict], data_dir: Path) -> Path:
    """Write accumulated samples to a timestamped .npz file and return the path."""
    boards = np.stack([s["board"] for s in samples])
    policies = np.stack([s["policy"] for s in samples])
    outcomes = np.stack([s["outcome"] for s in samples])

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = np.random.default_rng().integers(0, 0xFFFF_FFFF)
    filename = f"selfplay_{ts}_{suffix:08x}.npz"
    save_path = data_dir / filename
    np.savez_compressed(save_path, boards=boards, policies=policies, outcomes=outcomes)
    return save_path


def run_self_play(config: Config | None = None) -> tuple[Path, dict]:
    """
    Run a batch of self-play games and save training data.

    Data is flushed to disk every `flush_every` games to bound memory usage.
    Returns the data directory (which may contain multiple .npz files).
    """
    cfg = config or Config()
    cfg.ensure_dirs()

    if hexchess is None:
        raise ImportError(
            "hexchess bindings not available. Run `maturin develop` in bindings/python/"
        )

    print(
        f"Starting self-play: {cfg.num_self_play_games} games, "
        f"{cfg.num_simulations} simulations/move"
    )

    model_path = str(cfg.prev_best_model_path) if cfg.prev_best_model_path.exists() else None
    search = hexchess.MctsSearch(
        simulations=cfg.num_simulations,
        model_path=model_path,
    )

    flush_every = 50
    pending_samples: list[dict] = []
    games_since_flush = 0
    total_positions = 0
    saved_files: list[Path] = []
    outcome_counts: dict[str, int] = {}
    t0 = time.time()
    last_log_time = t0
    log_interval = 10  # seconds between progress lines

    for i in range(cfg.num_self_play_games):
        status, game_samples = play_one_game(search, cfg)
        pending_samples.extend(game_samples)
        total_positions += len(game_samples)
        games_since_flush += 1
        outcome_counts[status] = outcome_counts.get(status, 0) + 1
        game_num = i + 1

        # Flush to disk periodically
        is_last = game_num == cfg.num_self_play_games
        if pending_samples and (games_since_flush >= flush_every or is_last):
            path = _flush_samples(pending_samples, cfg.data_dir)
            saved_files.append(path)
            pending_samples.clear()
            games_since_flush = 0

        now = time.time()
        elapsed = now - t0
        if game_num == 1 or is_last or (now - last_log_time) >= log_interval:
            last_log_time = now
            outcome_str = " ".join(f"{k}={v}" for k, v in sorted(outcome_counts.items()))
            avg_moves = total_positions / game_num
            print(
                f"  {game_num}/{cfg.num_self_play_games} games | "
                f"{total_positions} pos ({avg_moves:.0f} avg moves/game) | "
                f"{elapsed:.0f}s ({elapsed/game_num:.1f}s/game) | {outcome_str}",
                flush=True,
            )

    if total_positions == 0:
        print("Warning: no samples generated.")
    else:
        print(f"Saved {total_positions} positions across {len(saved_files)} files")

    elapsed = time.time() - t0
    stats = {
        "games": cfg.num_self_play_games,
        "total_positions": total_positions,
        "avg_game_length": round(total_positions / max(cfg.num_self_play_games, 1), 1),
        "outcomes": dict(sorted(outcome_counts.items())),
        "elapsed_seconds": round(elapsed, 1),
    }

    return cfg.data_dir, stats
