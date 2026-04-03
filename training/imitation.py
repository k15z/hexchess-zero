"""Generate training data by imitating minimax search.

Produces samples in the same format as self-play workers so the trainer
can consume them without modification.

Value targets use the Stockfish NNUE-style lambda blend:
  wdl_target = λ * sigmoid(eval) + (1-λ) * game_outcome
This gives the NN both per-position granularity (from the eval) and
ground truth about who actually won (from the game outcome).
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
    """Convert a minimax centipawn score to a WDL vector via sigmoid.

    Uses the standard logistic model: W = sigmoid(s - margin),
    L = sigmoid(-s - margin), D = 1 - W - L, where s = score/scale.
    The draw margin creates a zone around 0 where draws are most likely.
    """
    score_clamped = max(-5000, min(5000, score))
    s = score_clamped / WDL_SCALE
    draw_margin = 0.5
    w = 1.0 / (1.0 + np.exp(-(s - draw_margin)))
    l = 1.0 / (1.0 + np.exp(-(-s - draw_margin)))
    d = max(0.0, 1.0 - w - l)
    total = w + d + l
    return np.array([w / total, d / total, l / total], dtype=np.float32)


def _outcome_to_wdl(status: str) -> np.ndarray:
    """Convert a game status string to a WDL vector from white's perspective."""
    if status == "checkmate_white":
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif status == "checkmate_black":
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        # All draws (repetition, 50-move, stalemate, insufficient material, max_ply)
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _flip_wdl(wdl: np.ndarray) -> np.ndarray:
    """Flip WDL from white's perspective to black's (swap W and L)."""
    return np.array([wdl[2], wdl[1], wdl[0]], dtype=np.float32)


def play_imitation_game(cfg: AsyncConfig, log_interval: int = 50) -> list[dict]:
    """Play one game collecting imitation samples with blended WDL targets.

    Two-pass approach:
      Pass 1: Play the game following minimax best moves, collecting
              (board_tensor, policy, eval_score, side_to_move) per position.
      Pass 2: After the game ends, determine the actual outcome, then
              blend eval-derived WDL with game outcome for each position:
              wdl = λ * sigmoid(eval) + (1-λ) * game_outcome

    Starts with random opening moves for diversity. Randomizes between
    N and N+1 random plies so both sides make equal random moves and
    alternate who gets the first minimax move.
    """
    game = hexchess.Game()
    pending = []  # (board_tensor, policy, eval_score, side_to_move)

    ply = 0
    max_ply = 300
    random_plies = cfg.imitation_random_plies + random.randint(0, 1)
    game_t0 = time.time()

    # Pass 1: play the game, collect raw data.
    while not game.is_game_over() and ply < max_ply:
        if ply < random_plies:
            moves = game.legal_moves()
            mv = random.choice(moves)
            game.apply_move(mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"], mv.get("promotion"))
            ply += 1
            continue

        result = hexchess.minimax_search_with_policy(game, cfg.imitation_depth)

        board_tensor = np.array(hexchess.encode_board(game), dtype=np.float32)
        policy = _scores_to_policy(result["moves"], cfg.imitation_temperature)
        best_score = result["best_score"]
        side = game.side_to_move()

        pending.append((board_tensor, policy, best_score, side))

        mv = result["best_move"]
        game.apply_move(mv["from_q"], mv["from_r"], mv["to_q"], mv["to_r"], mv.get("promotion"))
        ply += 1

        if ply % log_interval == 0:
            elapsed = time.time() - game_t0
            logger.info("    ply {}/{} | {} positions | {:.1f}s | score: {}",
                        ply, max_ply, len(pending), elapsed, best_score)

    # Pass 2: determine outcome and blend WDL targets.
    status = game.status() if game.is_game_over() else "max_ply"
    outcome_wdl_white = _outcome_to_wdl(status)
    lam = cfg.imitation_wdl_lambda

    samples = []
    for board_tensor, policy, eval_score, side in pending:
        # Eval-derived WDL is from the side-to-move's perspective.
        eval_wdl = _score_to_wdl(eval_score)

        # Game outcome WDL needs to match the side-to-move's perspective.
        if side == "white":
            outcome_wdl = outcome_wdl_white
        else:
            outcome_wdl = _flip_wdl(outcome_wdl_white)

        # Blend: λ * eval_wdl + (1-λ) * outcome_wdl
        blended_wdl = lam * eval_wdl + (1.0 - lam) * outcome_wdl
        # Renormalize (should already sum to ~1, but be safe).
        blended_wdl = blended_wdl / blended_wdl.sum()

        samples.append({
            "board": board_tensor,
            "policy": policy,
            "outcome": blended_wdl,
        })

    elapsed = time.time() - game_t0
    logger.info("  game done: {} plies, {} positions, {:.1f}s | {}",
                ply, len(samples), elapsed, status)

    return samples
