from __future__ import annotations
"""Self-play game generation using MCTS via the hexchess engine."""

import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .config import Config

try:
    import hexchess
except ImportError:
    hexchess = None



def play_one_game(config: Config) -> tuple[str, list[dict]]:
    """
    Play a single self-play game and return a list of training samples.

    Each sample is a dict with:
        board: np.ndarray of shape (16, 11, 11)
        policy: np.ndarray of shape (num_move_indices,)
        outcome: float  (filled in after game ends)
    """
    if hexchess is None:
        raise ImportError(
            "hexchess bindings not available. Run `maturin develop` in bindings/python/"
        )

    game = hexchess.Game()
    model_path = str(config.prev_best_model_path) if config.prev_best_model_path.exists() else None
    search = hexchess.MctsSearch(
        simulations=config.num_simulations,
        model_path=model_path,
    )
    num_indices = hexchess.num_move_indices()

    samples = []  # list of (board_tensor, policy_vector, side_to_move)
    move_number = 0

    while not game.is_game_over():
        # Encode current board
        board_tensor = hexchess.encode_board(game)  # (16, 11, 11) numpy array

        # Determine temperature
        if move_number < config.temperature_threshold:
            temperature = config.temperature_high
        else:
            temperature = config.temperature_low

        # Run MCTS with Dirichlet noise at the root for exploration
        result = search.run(
            game,
            temperature=temperature,
            dirichlet_epsilon=config.dirichlet_epsilon,
            dirichlet_alpha=config.dirichlet_alpha,
        )
        policy = result["policy"]  # already temperature-scaled by engine

        # Record sample (outcome filled in later)
        side = game.side_to_move()
        samples.append(
            {
                "board": board_tensor,
                "policy": policy,
                "side": side,
            }
        )

        # The engine already selected the best move via temperature sampling
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
        del sample["side"]  # no longer needed

    return status, samples


def _play_game_worker(args: tuple) -> tuple[str, list[dict]]:
    """Worker function for multiprocessing."""
    config, game_idx = args
    return play_one_game(config)


def _flush_samples(samples: list[dict], data_dir: Path) -> Path:
    """Write accumulated samples to a timestamped .npz file and return the path."""
    boards = np.stack([s["board"] for s in samples])
    policies = np.stack([s["policy"] for s in samples])
    outcomes = np.stack([s["outcome"] for s in samples])

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    # Short random suffix to avoid collisions between distributed workers
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

    workers = cfg.num_self_play_workers

    num_indices = hexchess.num_move_indices()
    print(
        f"Starting self-play: {cfg.num_self_play_games} games, "
        f"{cfg.num_simulations} simulations/move"
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

    def _ingest(result: tuple[str, list[dict]], game_num: int) -> None:
        nonlocal last_log_time, games_since_flush, total_positions
        status, game_samples = result
        pending_samples.extend(game_samples)
        total_positions += len(game_samples)
        games_since_flush += 1
        outcome_counts[status] = outcome_counts.get(status, 0) + 1

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

    if workers > 1:
        args = [(cfg, i) for i in range(cfg.num_self_play_games)]
        with Pool(processes=workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_play_game_worker, args)):
                _ingest(result, i + 1)
    else:
        for i in range(cfg.num_self_play_games):
            result = _play_game_worker((cfg, i))
            _ingest(result, i + 1)

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run self-play game generation")
    parser.add_argument("--games", type=int, default=None, help="Number of games")
    parser.add_argument("--simulations", type=int, default=None, help="MCTS simulations per move")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    args = parser.parse_args()

    cfg = Config()
    if args.games is not None:
        cfg.num_self_play_games = args.games
    if args.simulations is not None:
        cfg.num_simulations = args.simulations
    if args.workers is not None:
        cfg.num_self_play_workers = args.workers

    run_self_play(cfg)  # stats discarded in CLI mode
