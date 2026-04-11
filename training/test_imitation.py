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


def test_scores_to_policy_and_mask_indices_match_white() -> None:
    game = hexchess.Game()
    assert game.side_to_move() == "white"
    moves = _minimax_moves(game, depth=1)
    policy, mask = _scores_to_policy_and_mask(game, moves, temperature=100.0)

    # Every move in the minimax result must land at game.policy_index.
    for entry in moves:
        mv = entry.move
        idx = game.policy_index(
            mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion
        )
        assert mask[idx], "legal mask missing a minimax move"
        assert policy[idx] > 0.0, "policy target zero at a minimax move"
    # Mask and policy-support are identical for this helper (imitation
    # always sets both for every legal move).
    np.testing.assert_array_equal(mask, policy > 0.0)


def test_scores_to_policy_and_mask_is_stm_framed_black() -> None:
    """Black-to-move: indices must come from ``game.policy_index``, not
    ``hexchess.move_to_index``. Regression for the bug where the helper
    used absolute indices while the board tensor sat in STM frame.
    """
    game = hexchess.Game()
    game.apply(game.legal_moves()[0])
    assert game.side_to_move() == "black"

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

    # And a would-be absolute-frame mask must differ — otherwise the test
    # is vacuous (no asymmetric moves), which would hide a regression.
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
    # the legal mask on a 1:1 basis here (every minimax move gets both).
    np.testing.assert_array_equal(mask, policy > 0.0)
