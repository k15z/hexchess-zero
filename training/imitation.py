"""Generate training data by imitating minimax search.

Produces samples in the same format as self-play workers so the trainer
can consume them without modification.
"""

from __future__ import annotations

import random
import time

import numpy as np
from loguru import logger

import hexchess

from .config import AsyncConfig

NUM_MOVES = hexchess.num_move_indices()
WDL_SCALE = 400.0


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
    scores_arr = np.clip(scores_arr, -5000, 5000)
    scaled = scores_arr / temperature
    scaled -= scaled.max()
    exp_scores = np.exp(scaled)
    probs = exp_scores / exp_scores.sum()

    for idx, p in zip(indices, probs):
        policy[idx] = p
    return policy


def _score_to_wdl(score: int) -> np.ndarray:
    """Convert a minimax centipawn score to a WDL vector."""
    score_clamped = max(-5000, min(5000, score))
    s = score_clamped / WDL_SCALE
    draw_margin = 0.5
    w = 1.0 / (1.0 + np.exp(-(s - draw_margin)))
    l = 1.0 / (1.0 + np.exp(-(-s - draw_margin)))
    d = max(0.0, 1.0 - w - l)
    total = w + d + l
    return np.array([w / total, d / total, l / total], dtype=np.float32)


def play_imitation_game(cfg: AsyncConfig, log_interval: int = 50) -> list[dict]:
    """Play one game collecting imitation samples.

    Starts with random opening moves for diversity, then follows minimax
    best moves. Randomizes between N-1 and N random plies so both sides
    get an equal chance of making the first minimax move.
    """
    game = hexchess.Game()
    samples = []

    ply = 0
    max_ply = 300
    random_plies = cfg.imitation_random_plies + random.randint(-1, 0)
    game_t0 = time.time()

    while not game.is_game_over() and ply < max_ply:
        if ply < random_plies:
            moves = game.legal_moves()
            mv = random.choice(moves)
            game.apply_move(mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"], mv.get("promotion"))
            ply += 1
            continue

        result = hexchess.minimax_search_all(game, cfg.imitation_depth)

        board_tensor = np.array(hexchess.encode_board(game), dtype=np.float32)
        policy = _scores_to_policy(result["moves"], cfg.imitation_temperature)

        best_score = max(entry["score"] for entry in result["moves"])
        wdl = _score_to_wdl(best_score)

        samples.append({
            "board": board_tensor,
            "policy": policy,
            "outcome": wdl,
        })

        best_entry = max(result["moves"], key=lambda e: e["score"])
        mv = best_entry["move"]
        game.apply_move(mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"], mv.get("promotion"))
        ply += 1

        if ply % log_interval == 0:
            elapsed = time.time() - game_t0
            logger.info("    ply {}/{} | {} positions | {:.1f}s | score: {}",
                        ply, max_ply, len(samples), elapsed, best_score)

    elapsed = time.time() - game_t0
    status = game.status() if game.is_game_over() else "max_ply"
    logger.info("  game done: {} plies, {} positions, {:.1f}s | {}",
                ply, len(samples), elapsed, status)

    return samples
