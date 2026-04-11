"""Tests for training.imitation helpers.

Focused on the STM-frame consistency invariant between the policy target
and the legal mask: both must be indexed via ``game.policy_index`` so the
mask and the policy sit at the same slots as the STM-frame board tensor
produced by ``hexchess.encode_board(game)``.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import hexchess
except ImportError:  # pragma: no cover — bindings are a build-time requirement
    pytest.skip("hexchess binding not available", allow_module_level=True)

from training.imitation import _scores_to_policy_and_mask


def _minimax_moves(game, depth: int = 1):
    """Return ``(moves, _)`` from minimax_search_with_policy."""
    return hexchess.minimax_search_with_policy(game, depth).moves


def _black_to_move_asymmetric_position():
    """Apply a white off-center pawn push so the resulting black-to-move
    position has an asymmetric legal-move set — the mirror-frame and
    absolute-frame masks are guaranteed to differ. Avoids depending on
    engine move ordering for robustness across refactors.
    """
    game = hexchess.Game()
    board = {(p.q, p.r): p for p in game.board_state()}
    chosen = None
    for mv in game.legal_moves():
        src = board.get((mv.from_q, mv.from_r))
        if src is None or src.piece != "pawn" or mv.from_q == 0:
            continue
        chosen = mv
        break
    assert chosen is not None, "no off-center pawn push found at starting position"
    game.apply(chosen)
    assert game.side_to_move() == "black"
    return game


def test_scores_to_policy_and_mask_is_absolute_identity_white() -> None:
    """White-to-move: STM-frame indexing collapses to the identity, so
    the mask must match a hand-built absolute-frame mask exactly. This
    pins the white-half contract independently of the black test, so a
    regression that broke only the white path would still surface here.
    """
    game = hexchess.Game()
    assert game.side_to_move() == "white"
    moves = _minimax_moves(game, depth=1)
    policy, mask = _scores_to_policy_and_mask(game, moves, temperature=100.0)

    expected = np.zeros_like(mask)
    for entry in moves:
        mv = entry.move
        expected[hexchess.move_to_index(
            mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion
        )] = True
    np.testing.assert_array_equal(mask, expected)

    # Mask and policy-support are identical for this helper (imitation
    # always sets both for every legal move).
    np.testing.assert_array_equal(mask, policy > 0.0)


def test_scores_to_policy_and_mask_is_stm_framed_black() -> None:
    """Black-to-move on an asymmetric position: indices must come from
    ``game.policy_index`` (mirrored via MIRROR_INDEX), not
    ``hexchess.move_to_index``. Regression for the bug where the helper
    used absolute indices while the board tensor sat in STM frame.
    """
    game = _black_to_move_asymmetric_position()
    moves = _minimax_moves(game, depth=1)
    policy, mask = _scores_to_policy_and_mask(game, moves, temperature=100.0)

    # Mask must lie exactly where game.policy_index puts each move.
    expected_stm = np.zeros_like(mask)
    for entry in moves:
        mv = entry.move
        expected_stm[game.policy_index(
            mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion
        )] = True
    np.testing.assert_array_equal(mask, expected_stm)

    # And a would-be absolute-frame mask must differ — our asymmetric
    # position guarantees a non-mirror-invariant legal-move set, so the
    # sets-as-bitmaps must not coincide.
    absolute = np.zeros_like(mask)
    for entry in moves:
        mv = entry.move
        absolute[hexchess.move_to_index(
            mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion
        )] = True
    assert not np.array_equal(mask, absolute), (
        "black-to-move mask matches absolute frame — the helper silently "
        "reverted to hexchess.move_to_index"
    )

    # Policy and mask are in the same frame: the policy's support equals
    # the legal mask on a 1:1 basis here.
    np.testing.assert_array_equal(mask, policy > 0.0)
