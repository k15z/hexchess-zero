from __future__ import annotations
"""Self-play game generation using MCTS via the hexchess engine."""

import time
import uuid
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
    model_path = str(config.best_model_path) if config.best_model_path.exists() else None
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

    # Determine game outcome
    status = game.status()
    if status == "checkmate_white":
        outcome_white = 1.0
    elif status == "checkmate_black":
        outcome_white = -1.0
    else:
        outcome_white = 0.0

    # Fill in outcome from each side's perspective
    for sample in samples:
        if sample["side"] == "white":
            sample["outcome"] = outcome_white
        else:
            sample["outcome"] = -outcome_white
        del sample["side"]  # no longer needed

    return status, samples


def _play_game_worker(args: tuple) -> tuple[str, list[dict]]:
    """Worker function for multiprocessing."""
    config, game_idx = args
    return play_one_game(config)


def run_self_play(config: Config | None = None) -> Path:
    """
    Run a batch of self-play games and save training data.

    Returns the path to the saved .npz file.
    """
    cfg = config or Config()
    cfg.ensure_dirs()

    if hexchess is None:
        raise ImportError(
            "hexchess bindings not available. Run `maturin develop` in bindings/python/"
        )

    # When using NN model, run sequentially (each worker loads model separately)
    using_model = cfg.best_model_path.exists()
    workers = 1 if using_model else cfg.num_self_play_workers

    num_indices = hexchess.num_move_indices()
    print(
        f"Starting self-play: {cfg.num_self_play_games} games, "
        f"{cfg.num_simulations} simulations/move"
    )

    all_samples: list[dict] = []
    outcomes: dict[str, int] = {}
    t0 = time.time()
    last_log_time = t0
    log_interval = 10  # seconds between progress lines

    def _ingest(result: tuple[str, list[dict]], game_num: int) -> None:
        nonlocal last_log_time
        status, game_samples = result
        all_samples.extend(game_samples)
        outcomes[status] = outcomes.get(status, 0) + 1

        now = time.time()
        elapsed = now - t0
        # Log on first game, last game, or every log_interval seconds
        is_last = game_num == cfg.num_self_play_games
        if game_num == 1 or is_last or (now - last_log_time) >= log_interval:
            last_log_time = now
            outcome_str = " ".join(f"{k}={v}" for k, v in sorted(outcomes.items()))
            avg_moves = len(all_samples) / game_num
            print(
                f"  {game_num}/{cfg.num_self_play_games} games | "
                f"{len(all_samples)} pos ({avg_moves:.0f} avg moves/game) | "
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

    if not all_samples:
        print("Warning: no samples generated.")
        return cfg.data_dir

    # Stack into arrays
    boards = np.stack([s["board"] for s in all_samples])
    policies = np.stack([s["policy"] for s in all_samples])
    outcomes = np.array([s["outcome"] for s in all_samples], dtype=np.float32)

    # Save to .npz
    filename = f"selfplay_{uuid.uuid4().hex[:8]}.npz"
    save_path = cfg.data_dir / filename
    np.savez_compressed(save_path, boards=boards, policies=policies, outcomes=outcomes)

    print(f"Saved {len(all_samples)} positions to {save_path}")
    return save_path


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

    run_self_play(cfg)
