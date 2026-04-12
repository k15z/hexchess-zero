"""Rust ↔ Python encoder parity tests.

Verifies that the PyO3 bindings produce byte-identical results to the
underlying Rust implementation for both encode_board and policy_index.
Since both sides call the same Rust code, this primarily tests the PyO3
marshaling layer (numpy shape, dtype, memory layout) and argument parsing.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import hexchess
except ImportError:
    pytest.skip("hexchess binding not available", allow_module_level=True)


def _generate_positions(count: int) -> list[hexchess.Game]:
    """Generate diverse positions by playing random legal moves."""
    import random
    rng = random.Random(42)
    positions = []
    for i in range(count * 2):  # overshoot to handle games that end early
        if len(positions) >= count:
            break
        game = hexchess.Game()
        ply = rng.randint(1, 25)
        for _ in range(ply):
            if game.is_game_over():
                break
            moves = game.legal_moves()
            game.apply(rng.choice(moves))
        if not game.is_game_over():
            positions.append(game)
    return positions


class TestEncodeBoardParity:
    """encode_board via single-game and batch APIs must agree."""

    def test_single_vs_batch_encoding_match(self):
        positions = _generate_positions(50)
        assert len(positions) >= 40, f"need ≥40 positions, got {len(positions)}"

        # Encode individually
        singles = [hexchess.encode_board(g) for g in positions]

        # Encode as batch
        batch = hexchess.encode_batch(positions)

        assert batch.shape == (len(positions), 22, 11, 11)
        assert batch.dtype == np.float32

        for i, single in enumerate(singles):
            assert single.shape == (22, 11, 11)
            assert single.dtype == np.float32
            np.testing.assert_array_equal(
                single, batch[i],
                err_msg=f"position {i}: single vs batch encoding mismatch",
            )

    def test_encoding_shape_and_dtype(self):
        game = hexchess.Game()
        tensor = hexchess.encode_board(game)
        assert tensor.shape == (22, 11, 11)
        assert tensor.dtype == np.float32

    def test_validity_mask_has_91_cells(self):
        """Channel 17 should have exactly 91 ones for every position."""
        positions = _generate_positions(50)
        for i, game in enumerate(positions):
            tensor = hexchess.encode_board(game)
            valid_count = int(tensor[17].sum())
            assert valid_count == 91, (
                f"position {i}: validity mask has {valid_count} cells, expected 91"
            )

    def test_encoding_deterministic(self):
        """Encoding the same position twice must be identical."""
        positions = _generate_positions(20)
        for i, game in enumerate(positions):
            t1 = hexchess.encode_board(game)
            t2 = hexchess.encode_board(game)
            np.testing.assert_array_equal(
                t1, t2, err_msg=f"position {i}: non-deterministic encoding"
            )


class TestPolicyIndexParity:
    """policy_index (STM-frame) vs move_to_index (absolute) parity."""

    def test_white_to_move_identity(self):
        """When white is to move, policy_index == move_to_index."""
        game = hexchess.Game()
        assert game.side_to_move() == "white"

        for mv in game.legal_moves():
            abs_idx = hexchess.move_to_index(
                mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion
            )
            stm_idx = game.policy_index(
                mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion
            )
            assert abs_idx == stm_idx, (
                f"white-to-move: abs={abs_idx} != stm={stm_idx} for {mv}"
            )

    def test_black_to_move_remap(self):
        """When black is to move, at least some moves must be remapped."""
        game = hexchess.Game()
        game.apply(game.legal_moves()[0])
        assert game.side_to_move() == "black"

        remapped = 0
        total = 0
        for mv in game.legal_moves():
            abs_idx = hexchess.move_to_index(
                mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion
            )
            stm_idx = game.policy_index(
                mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion
            )
            if abs_idx != stm_idx:
                remapped += 1
            total += 1

        assert remapped > 0, "no moves remapped for black — mirror inactive"

    def test_policy_index_on_50_positions(self):
        """policy_index returns valid indices for all legal moves across
        50 diverse positions."""
        positions = _generate_positions(50)
        num_indices = hexchess.num_move_indices()
        total_moves = 0

        for i, game in enumerate(positions):
            for mv in game.legal_moves():
                idx = game.policy_index(
                    mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion
                )
                assert 0 <= idx < num_indices, (
                    f"position {i}: policy_index {idx} out of range [0, {num_indices})"
                )
                total_moves += 1

        assert total_moves > 1000, f"only {total_moves} moves tested"

    def test_index_roundtrip(self):
        """move_to_index → index_to_move round-trips correctly."""
        game = hexchess.Game()
        for mv in game.legal_moves():
            idx = hexchess.move_to_index(
                mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion
            )
            rt = hexchess.index_to_move(idx)
            assert rt.from_q == mv.from_q and rt.from_r == mv.from_r, (
                f"from mismatch: {rt} vs {mv}"
            )
            assert rt.to_q == mv.to_q and rt.to_r == mv.to_r, (
                f"to mismatch: {rt} vs {mv}"
            )
            assert rt.promotion == mv.promotion, (
                f"promo mismatch: {rt.promotion} vs {mv.promotion}"
            )
