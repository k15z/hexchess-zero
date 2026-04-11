"""Generate training data by imitating minimax search.

Produces samples in the same format as self-play workers so the trainer
can consume them without modification.

Value targets use the Stockfish NNUE-style lambda blend:
  wdl_target = λ * sigmoid(eval) + (1-λ) * game_outcome
This gives the NN both per-position granularity (from the eval) and
ground truth about who actually won (from the game outcome).
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from loguru import logger

try:
    import hexchess
except ImportError:
    hexchess = None  # type: ignore[assignment]

from .config import AsyncConfig
from .types import BoardTensor, ImitationSample, PolicyVector, WdlVector

NUM_MOVES = hexchess.num_move_indices() if hexchess is not None else 0
WDL_SCALE = 400.0


def _softmax_probs(
    move_scores: "Sequence[hexchess.RankedMove]", temperature: float,
) -> np.ndarray:
    """Compute softmax probabilities over minimax move scores."""
    scores = np.array([m.score for m in move_scores], dtype=np.float64)
    scores = np.clip(scores, -5000, 5000)
    scaled = scores / temperature
    scaled -= scaled.max()
    exp_scores = np.exp(scaled)
    return exp_scores / exp_scores.sum()


def _scores_to_policy(
    move_scores: "Sequence[hexchess.RankedMove]", temperature: float,
) -> PolicyVector:
    """Convert minimax move scores to a softmax policy vector."""
    assert hexchess is not None, "hexchess bindings not available"
    probs = _softmax_probs(move_scores, temperature)
    policy = np.zeros(NUM_MOVES, dtype=np.float32)
    for entry, p in zip(move_scores, probs):
        mv = entry.move
        idx = hexchess.move_to_index(
            mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion,
        )
        policy[idx] = p
    return policy


def _score_to_wdl(score: int) -> WdlVector:
    """Convert a minimax centipawn score to a WDL vector via sigmoid.

    Uses the standard logistic model: W = sigmoid(s - margin),
    L = sigmoid(-s - margin), D = 1 - W - L, where s = score/scale.
    The draw margin creates a zone around 0 where draws are most likely.
    """
    score_clamped = max(-5000, min(5000, score))
    s = score_clamped / WDL_SCALE
    # Draw margin calibrated from 22k positions across 200 depth-2 games
    # (scripts/calibrate_wdl_scale.py). Previous value of 0.5 underestimated
    # draw frequency in hex chess (~40-55% near equal positions vs ~25% implied).
    draw_margin = 1.0
    w = 1.0 / (1.0 + np.exp(-(s - draw_margin)))
    loss_prob = 1.0 / (1.0 + np.exp(-(-s - draw_margin)))
    d = max(0.0, 1.0 - w - loss_prob)
    total = w + d + loss_prob
    return np.array([w / total, d / total, loss_prob / total], dtype=np.float32)


def _outcome_to_wdl(status: str) -> WdlVector:
    """Convert a game status string to a WDL vector from white's perspective."""
    if status == "checkmate_white":
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif status == "checkmate_black":
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        # All draws (repetition, 50-move, stalemate, insufficient material, max_ply)
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _flip_wdl(wdl: WdlVector) -> WdlVector:
    """Flip WDL from white's perspective to black's (swap W and L)."""
    return np.array([wdl[2], wdl[1], wdl[0]], dtype=np.float32)


def _sample_move(
    move_scores: "Sequence[hexchess.RankedMove]", temperature: float,
) -> "hexchess.Move":
    """Sample a move from minimax scores using softmax temperature."""
    probs = _softmax_probs(move_scores, temperature)
    idx = np.random.choice(len(move_scores), p=probs)
    return move_scores[idx].move


def play_imitation_game(
    cfg: AsyncConfig, log_interval: int = 50,
) -> list[ImitationSample]:
    """Play one game collecting imitation samples with blended WDL targets.

    Two-pass approach:
      Pass 1: Play the game following minimax best moves, collecting
              (board_tensor, policy, eval_score, side_to_move) per position.
      Pass 2: After the game ends, determine the actual outcome, then
              blend eval-derived WDL with game outcome for each position:
              wdl = λ * sigmoid(eval) + (1-λ) * game_outcome

    Uses softmax temperature-based move selection for the first N plies
    to create diverse openings while keeping positions natural. After the
    exploration phase, always plays the best minimax move.
    """
    assert hexchess is not None, "hexchess bindings not available"
    game = hexchess.Game()
    pending: list[tuple[BoardTensor, PolicyVector, int, str]] = []

    ply = 0
    # Glinski self-play games can run 500+ plies (notes/12 §7); cap at 600
    # so the bootstrap doesn't truncate real games into degenerate samples.
    max_ply = 600
    game_t0 = time.time()

    # Pass 1: play the game, collect raw data.
    while not game.is_game_over() and ply < max_ply:
        result = hexchess.minimax_search_with_policy(game, cfg.imitation_depth)

        board_tensor = np.asarray(hexchess.encode_board(game), dtype=np.float32)
        policy = _scores_to_policy(result.moves, cfg.imitation_temperature)
        best_score = result.best_score
        side = game.side_to_move()

        pending.append((board_tensor, policy, best_score, side))

        if ply < cfg.imitation_exploration_plies and len(result.moves) > 1:
            mv = _sample_move(result.moves, cfg.imitation_temperature)
        else:
            mv = result.best_move
        game.apply(mv)
        ply += 1

        if ply % log_interval == 0:
            elapsed = time.time() - game_t0
            logger.info("    ply {}/{} | {} positions | {:.1f}s | score: {}",
                        ply, max_ply, len(pending), elapsed, best_score)

    # Pass 2: determine outcome and blend WDL targets.
    status = game.status() if game.is_game_over() else "max_ply"
    outcome_wdl_white = _outcome_to_wdl(status)
    lam = cfg.imitation_wdl_lambda

    samples: list[ImitationSample] = []
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
