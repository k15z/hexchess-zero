"""Generate training data by imitating minimax search.

Produces NPZ files in the same format as self-play workers so the trainer
can consume them without modification.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

import hexchess

from .config import AsyncConfig

NUM_MOVES = hexchess.num_move_indices()
WDL_SCALE = 400.0  # centipawn → sigmoid scale for WDL conversion


def _scores_to_policy(move_scores: list[dict], temperature: float) -> np.ndarray:
    """Convert minimax move scores to a softmax policy vector."""
    policy = np.zeros(NUM_MOVES, dtype=np.float32)
    indices = []
    scores = []
    for entry in move_scores:
        mv = entry["move"]
        idx = hexchess.move_to_index(
            mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"], mv.get("promotion"),
        )
        indices.append(idx)
        scores.append(entry["score"])

    scores_arr = np.array(scores, dtype=np.float64)
    # Clamp to avoid overflow from mate scores
    scores_arr = np.clip(scores_arr, -5000, 5000)
    # Softmax with temperature
    scaled = scores_arr / temperature
    scaled -= scaled.max()  # numerical stability
    exp_scores = np.exp(scaled)
    probs = exp_scores / exp_scores.sum()

    for idx, p in zip(indices, probs):
        policy[idx] = p
    return policy


def _score_to_wdl(score: int) -> np.ndarray:
    """Convert a minimax centipawn score to a WDL vector.

    Uses a three-way split: scores near zero produce high draw probability,
    while large scores push toward win or loss.
    """
    score_clamped = max(-5000, min(5000, score))
    s = score_clamped / WDL_SCALE
    # W and L as separate sigmoids offset by a draw margin
    draw_margin = 0.5  # in units of score/scale
    w = 1.0 / (1.0 + np.exp(-(s - draw_margin)))
    l = 1.0 / (1.0 + np.exp(-(-s - draw_margin)))
    d = max(0.0, 1.0 - w - l)
    total = w + d + l
    return np.array([w / total, d / total, l / total], dtype=np.float32)


def _play_imitation_game(cfg: AsyncConfig) -> list[dict]:
    """Play one game collecting imitation samples.

    Starts with random opening moves for diversity, then follows minimax
    best moves. At each non-random position, runs minimax_search_all to
    collect policy and value targets.
    """
    game = hexchess.Game()
    samples = []

    ply = 0
    max_ply = 300  # safety limit

    while not game.is_game_over() and ply < max_ply:
        # Random opening phase
        if ply < cfg.imitation_random_plies:
            moves = game.legal_moves()
            mv = random.choice(moves)
            game.apply_move(mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"], mv.get("promotion"))
            ply += 1
            continue

        # Minimax phase: collect a training sample
        result = hexchess.minimax_search_all(game, cfg.imitation_depth)

        board_tensor = np.array(hexchess.encode_board(game), dtype=np.float32)
        policy = _scores_to_policy(result["moves"], cfg.imitation_temperature)

        # Value: use the best move's score for WDL
        best_score = max(entry["score"] for entry in result["moves"])
        wdl = _score_to_wdl(best_score)

        samples.append({
            "board": board_tensor,
            "policy": policy,
            "outcome": wdl,
        })

        # Play the best move
        best_entry = max(result["moves"], key=lambda e: e["score"])
        mv = best_entry["move"]
        game.apply_move(mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"], mv.get("promotion"))
        ply += 1

    return samples


def _flush_samples(samples: list[dict], data_dir: Path) -> Path:
    """Write imitation samples to a .npz file."""
    boards = np.stack([s["board"] for s in samples])
    policies = np.stack([s["policy"] for s in samples])
    outcomes = np.array([s["outcome"] for s in samples], dtype=np.float32)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = np.random.default_rng().integers(0, 0xFFFF_FFFF)
    basename = f"im_v0_{ts}_{suffix:08x}"

    tmp_path = data_dir / (basename + ".tmp")
    np.savez_compressed(tmp_path, boards=boards, policies=policies, outcomes=outcomes)
    save_path = data_dir / (basename + ".npz")
    (data_dir / (basename + ".tmp.npz")).rename(save_path)

    return save_path


def _play_imitation_game_wrapper(args: tuple) -> list[dict]:
    """Wrapper for multiprocessing (top-level picklable function)."""
    cfg_dict, game_idx = args
    cfg = AsyncConfig(**cfg_dict)
    return _play_imitation_game(cfg)


def generate_imitation_data(cfg: AsyncConfig) -> None:
    """Generate imitation training data from minimax and save as NPZ files."""
    cfg.ensure_dirs()

    num_workers = max(1, os.cpu_count() - 2)
    logger.info(
        "Generating imitation data: {} games, depth {}, {} random plies, {} workers",
        cfg.imitation_num_games, cfg.imitation_depth, cfg.imitation_random_plies,
        num_workers,
    )

    # Serialize config fields for pickling across processes
    cfg_dict = {f.name: getattr(cfg, f.name)
                for f in cfg.__dataclass_fields__.values()}
    tasks = [(cfg_dict, i) for i in range(cfg.imitation_num_games)]

    all_samples: list[dict] = []
    flush_every = 50  # flush every N games
    games_done = 0

    with mp.Pool(num_workers) as pool:
        for samples in pool.imap_unordered(_play_imitation_game_wrapper, tasks):
            all_samples.extend(samples)
            games_done += 1

            if games_done % flush_every == 0 and all_samples:
                path = _flush_samples(all_samples, cfg.training_data_dir)
                logger.info(
                    "Game {}/{}: flushed {} positions to {}",
                    games_done, cfg.imitation_num_games, len(all_samples), path.name,
                )
                all_samples = []

    # Flush remaining
    if all_samples:
        path = _flush_samples(all_samples, cfg.training_data_dir)
        logger.info("Final flush: {} positions to {}", len(all_samples), path.name)

    logger.info("Imitation data generation complete.")
