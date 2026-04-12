"""Tests for the self-play worker training schema.

These tests are hermetic: they bypass the Rust binding by constructing
`GameRecord` objects from fake numpy arrays and exercise the pure
`write_samples_to_npz` / `finalize_game_targets` / `compute_opening_hash`
functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from training.worker import (
    GameRecord,
    PositionSample,
    _git_sha,
    compute_opening_hash,
    finalize_game_targets,
    legal_mask_from_moves,
    write_samples_to_npz,
    write_trace_json,
    _root_q_to_wdl,
)


NUM_MOVES = 4096  # arbitrary for tests; real binding uses hexchess.num_move_indices()


def _fake_position(ply: int, side: str, root_q: float = 0.0) -> PositionSample:
    board = np.zeros((22, 11, 11), dtype=np.float32)
    # Stamp the ply into the first channel so boards aren't all-zero.
    board[0, 0, 0] = float(ply)
    policy = np.zeros(NUM_MOVES, dtype=np.float32)
    policy[ply % NUM_MOVES] = 1.0
    # Legal mask: the visited index plus a handful of unvisited legal
    # moves, so tests exercise the fact that legal_mask is wider than
    # `policy > 0` (the entire point of this schema field).
    legal_mask = np.zeros(NUM_MOVES, dtype=bool)
    legal_mask[ply % NUM_MOVES] = True
    for extra in range(1, 6):
        legal_mask[(ply + extra) % NUM_MOVES] = True
    return PositionSample(
        board=board,
        policy=policy,
        policy_aux_opp=np.full(NUM_MOVES, 1.0 / NUM_MOVES, dtype=np.float32),
        legal_mask=legal_mask,
        root_q=root_q,
        root_n=800,
        root_entropy=1.23,
        nn_value_at_position=root_q,
        legal_count=42,
        ply=ply,
        side=side,
    )


def _fake_record(
    *,
    game_len: int,
    wdl_white: list[float],
    game_id: int = 7,
) -> GameRecord:
    sides = ("white", "black")
    positions = [
        _fake_position(ply=i, side=sides[i % 2], root_q=0.1 * ((-1) ** i))
        for i in range(game_len)
    ]
    return GameRecord(
        positions=positions,
        game_id=game_id,
        model_version=3,
        started_at="2026-04-07T00:00:00+00:00",
        duration_s=12.5,
        result="white_win"
        if wdl_white[0] > 0.5
        else ("draw" if wdl_white[1] > 0.5 else "black_win"),
        termination="checkmate",
        opening_hash="deadbeefcafebabe",
        rng_seed=12345,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.3,
        num_simulations=800,
        worker="test-worker",
        git_sha="abc123def456",
        num_total_positions=game_len,
        wdl_terminal_white=wdl_white,
        game_len_plies=game_len,
    )


# ---------------------------------------------------------------------------
# Schema + sidecar
# ---------------------------------------------------------------------------


def test_npz_schema_dtypes_and_shapes(tmp_path: Path) -> None:
    rec = _fake_record(game_len=5, wdl_white=[1.0, 0.0, 0.0])
    finalize_game_targets(rec)
    npz_path = tmp_path / "g.npz"
    meta = write_samples_to_npz(npz_path, rec, num_move_indices=NUM_MOVES)
    assert npz_path.exists()

    data = np.load(npz_path)
    n = 5
    expected = {
        "boards": ((n, 22, 11, 11), np.float16),
        "policy": ((n, NUM_MOVES), np.float16),
        "policy_aux_opp": ((n, NUM_MOVES), np.float16),
        "legal_mask": ((n, NUM_MOVES), np.bool_),
        "wdl_terminal": ((n, 3), np.float32),
        "wdl_short": ((n, 3), np.float32),
        "mlh": ((n,), np.int16),
        "was_full_search": ((n,), np.bool_),
        "root_q": ((n,), np.float16),
        "root_n": ((n,), np.int32),
        "root_entropy": ((n,), np.float16),
        "nn_value_at_position": ((n,), np.float16),
        "legal_count": ((n,), np.int16),
        "ply": ((n,), np.int16),
        "game_id": ((n,), np.uint64),
    }
    for key, (shape, dtype) in expected.items():
        assert key in data.files, f"missing column {key}"
        arr = data[key]
        assert arr.shape == shape, f"{key}: shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key}: dtype {arr.dtype} != {dtype}"

    # Sidecar
    meta_path = npz_path.with_suffix(".meta.json")
    loaded = json.loads(meta_path.read_text())
    for key in (
        "game_id_range",
        "model_version",
        "worker",
        "started_at",
        "duration_s",
        "num_full_search_positions",
        "num_total_positions",
        "result",
        "termination",
        "openings_hash",
        "git_sha",
        "rng_seed",
        "dirichlet_epsilon",
        "dirichlet_alpha",
        "num_simulations",
    ):
        assert key in loaded, f"meta sidecar missing {key}"
    assert loaded["game_id_range"] == [7, 7]
    assert meta == loaded  # returned dict matches file contents


def test_mlh_counts_down(tmp_path: Path) -> None:
    rec = _fake_record(game_len=20, wdl_white=[0.0, 1.0, 0.0])
    finalize_game_targets(rec)
    npz_path = tmp_path / "mlh.npz"
    write_samples_to_npz(npz_path, rec, num_move_indices=NUM_MOVES)
    mlh = np.load(npz_path)["mlh"]
    # Positions 0..19, game_len=20 → mlh 20, 19, ..., 1.
    assert list(mlh) == list(range(20, 0, -1))


def test_wdl_short_terminal_within_horizon(tmp_path: Path) -> None:
    # Game ends at ply 5; positions 0..4 should see terminal wdl in their short window.
    rec = _fake_record(game_len=5, wdl_white=[1.0, 0.0, 0.0])
    finalize_game_targets(rec)
    npz_path = tmp_path / "short.npz"
    write_samples_to_npz(npz_path, rec, num_move_indices=NUM_MOVES)
    data = np.load(npz_path)
    wdl_short = data["wdl_short"]
    # Terminal is white_win → STM at ply 0 is white → [1,0,0], ply 1 is black → [0,0,1], ...
    for i in range(5):
        expected = [1.0, 0.0, 0.0] if i % 2 == 0 else [0.0, 0.0, 1.0]
        np.testing.assert_allclose(wdl_short[i], expected, atol=1e-6)


def test_wdl_short_nonterminal_uses_root_q(tmp_path: Path) -> None:
    # Long game: positions should use the root_q approximation because target
    # ply (ply+8) is still < game_len.
    rec = _fake_record(game_len=40, wdl_white=[0.0, 1.0, 0.0])
    finalize_game_targets(rec)
    npz_path = tmp_path / "long.npz"
    write_samples_to_npz(npz_path, rec, num_move_indices=NUM_MOVES)
    data = np.load(npz_path)
    wdl_short = data["wdl_short"]
    # Position 0 is white, ply+8=8 is white, root_q at ply 8 = 0.1*(-1)**8 = 0.1.
    # Same side → no flip → WDL = (0.1, 0.9, 0).
    np.testing.assert_allclose(wdl_short[0], _root_q_to_wdl(0.1), atol=1e-6)
    # Position 1 is black, ply+8=9 is black (same parity), root_q = -0.1 (STM).
    np.testing.assert_allclose(wdl_short[1], _root_q_to_wdl(-0.1), atol=1e-6)


def test_opening_hash_deterministic() -> None:
    a = compute_opening_hash(["f5-f6", "f7-f6", "e4-e5", "g7-g6", "c1-d2", "h7-h6"])
    b = compute_opening_hash(["f5-f6", "f7-f6", "e4-e5", "g7-g6", "c1-d2", "h7-h6"])
    c = compute_opening_hash(["f5-f6", "f7-f6", "e4-e5", "g7-g6", "c1-d2", "h7-h7"])
    assert a == b
    assert a != c
    # Truncates at 6 plies — trailing differences don't matter.
    d = compute_opening_hash(
        ["f5-f6", "f7-f6", "e4-e5", "g7-g6", "c1-d2", "h7-h6", "extra"]
    )
    assert a == d


def test_git_sha_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_SHA", "1234567890abcdef")
    assert _git_sha() == "1234567890ab"


def test_write_trace_json(tmp_path: Path) -> None:
    rec = _fake_record(game_len=3, wdl_white=[0.0, 0.0, 1.0])
    finalize_game_targets(rec)
    trace_path = tmp_path / "trace.json"
    write_trace_json(trace_path, rec)
    loaded = json.loads(trace_path.read_text())
    assert loaded["game_id"] == 7
    assert loaded["result"] == "black_win"
    assert loaded["dirichlet_epsilon"] == pytest.approx(0.25)
    assert loaded["dirichlet_alpha"] == pytest.approx(0.3)
    assert loaded["num_simulations"] == 800
    assert len(loaded["entries"]) == 3
    for entry in loaded["entries"]:
        assert "ply" in entry and "side" in entry and "root_q" in entry


def test_empty_record_rejected(tmp_path: Path) -> None:
    rec = _fake_record(game_len=0, wdl_white=[0.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        write_samples_to_npz(tmp_path / "empty.npz", rec, num_move_indices=NUM_MOVES)


# ---------------------------------------------------------------------------
# legal_mask_from_moves — STM-frame consistency with policy targets
# ---------------------------------------------------------------------------


def test_legal_mask_from_moves_is_stm_framed_white() -> None:
    """White-to-move: legal mask indices must equal absolute indices.

    In STM frame, white-to-move policy indexing is the identity
    (``game.policy_index == hexchess.move_to_index``), so the mask built
    by ``legal_mask_from_moves(game, ...)`` must sit at exactly the same
    indices as a hypothetical absolute-frame mask. Catches a regression
    to the previous absolute-frame implementation in the trivial case.
    """
    hexchess = pytest.importorskip("hexchess")

    game = hexchess.Game()
    assert game.side_to_move() == "white"
    legal = game.legal_moves()
    num_moves = hexchess.num_move_indices()
    mask = legal_mask_from_moves(game, legal, num_moves)

    expected = np.zeros(num_moves, dtype=bool)
    for mv in legal:
        expected[
            hexchess.move_to_index(mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion)
        ] = True
    np.testing.assert_array_equal(mask, expected)


def _black_to_move_asymmetric_position():
    """Return a Glinski game in a black-to-move position whose legal-move
    set is deliberately not closed under the mirror involution.

    Apply a white move from an off-center file (``q != 0``) so the
    resulting position is asymmetric under the central inversion that
    defines the STM frame, and so black's legal move set cannot be
    mirror-invariant. This makes the frame regression tests robust to
    engine move-ordering changes — it doesn't rely on ``legal_moves()[0]``
    landing on any particular move.
    """
    hexchess = pytest.importorskip("hexchess")

    game = hexchess.Game()
    board = {(p.q, p.r): p for p in game.board_state()}
    chosen = None
    for mv in game.legal_moves():
        # Off-center pawn push: guarantees q != 0 on both endpoints and
        # therefore a non-self-mirror-symmetric move index, which in turn
        # breaks mirror-invariance of the resulting legal-move set for
        # anything short of a full symmetrically-paired response.
        src = board.get((mv.from_q, mv.from_r))
        if src is None or src.piece != "pawn" or mv.from_q == 0:
            continue
        chosen = mv
        break
    assert chosen is not None, "no off-center pawn push found at starting position"
    game.apply(chosen)
    assert game.side_to_move() == "black"
    return game


def test_legal_mask_from_moves_is_stm_framed_black() -> None:
    """Black-to-move: legal mask indices must match game.policy_index,
    which remaps via MIRROR_INDEX. This is the load-bearing regression
    test: the previous absolute-frame ``hexchess.move_to_index`` impl
    would put the mask at DIFFERENT indices than the policy target
    (built via ``search.run_pcr``'s STM-framed visit distribution).
    """
    hexchess = pytest.importorskip("hexchess")

    game = _black_to_move_asymmetric_position()
    legal = game.legal_moves()
    num_moves = hexchess.num_move_indices()
    mask = legal_mask_from_moves(game, legal, num_moves)

    # Expected: STM-frame indices via game.policy_index.
    expected_stm = np.zeros(num_moves, dtype=bool)
    for mv in legal:
        expected_stm[
            game.policy_index(mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion)
        ] = True
    np.testing.assert_array_equal(mask, expected_stm)

    # And: the would-be absolute-frame mask must *differ* from the STM
    # mask. At an asymmetric black-to-move position, at least one legal
    # move's mirrored index must lie at a different slot than its
    # absolute index — so the sets-as-bitmaps differ.
    absolute = np.zeros(num_moves, dtype=bool)
    for mv in legal:
        absolute[
            hexchess.move_to_index(mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion)
        ] = True
    assert not np.array_equal(mask, absolute), (
        "black-to-move legal mask equals the absolute-frame mask on an "
        "asymmetric position — the STM remap silently reverted to identity"
    )
