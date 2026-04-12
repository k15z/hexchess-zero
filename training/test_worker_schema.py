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
    ResignTracker,
    compute_opening_hash,
    finalize_game_targets,
    legal_mask_from_moves,
    write_samples_to_npz,
    write_trace_json,
    _resign_calibration_correct,
    _resigned_outcome,
    _root_q_to_wdl,
    _wdl_to_expected_score,
    _value_to_p_win,
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
    *, game_len: int, wdl_white: list[float], game_id: int = 7,
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
        result="white_win" if wdl_white[0] > 0.5 else ("draw" if wdl_white[1] > 0.5 else "black_win"),
        termination="checkmate",
        resigned_skipped=False,
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
        "game_id_range", "model_version", "worker", "started_at", "duration_s",
        "num_full_search_positions", "num_total_positions", "result",
        "termination", "resigned_skipped", "openings_hash", "git_sha", "rng_seed",
        "dirichlet_epsilon", "dirichlet_alpha", "num_simulations",
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
    np.testing.assert_allclose(
        wdl_short[0], _root_q_to_wdl(0.1), atol=1e-6
    )
    # Position 1 is black, ply+8=9 is black (same parity), root_q = -0.1 (STM).
    np.testing.assert_allclose(
        wdl_short[1], _root_q_to_wdl(-0.1), atol=1e-6
    )


def test_opening_hash_deterministic() -> None:
    a = compute_opening_hash(["f5-f6", "f7-f6", "e4-e5", "g7-g6", "c1-d2", "h7-h6"])
    b = compute_opening_hash(["f5-f6", "f7-f6", "e4-e5", "g7-g6", "c1-d2", "h7-h6"])
    c = compute_opening_hash(["f5-f6", "f7-f6", "e4-e5", "g7-g6", "c1-d2", "h7-h7"])
    assert a == b
    assert a != c
    # Truncates at 6 plies — trailing differences don't matter.
    d = compute_opening_hash(["f5-f6", "f7-f6", "e4-e5", "g7-g6", "c1-d2", "h7-h6", "extra"])
    assert a == d


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
        expected[hexchess.move_to_index(
            mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion
        )] = True
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
        expected_stm[game.policy_index(
            mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion
        )] = True
    np.testing.assert_array_equal(mask, expected_stm)

    # And: the would-be absolute-frame mask must *differ* from the STM
    # mask. At an asymmetric black-to-move position, at least one legal
    # move's mirrored index must lie at a different slot than its
    # absolute index — so the sets-as-bitmaps differ.
    absolute = np.zeros(num_moves, dtype=bool)
    for mv in legal:
        absolute[hexchess.move_to_index(
            mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion
        )] = True
    assert not np.array_equal(mask, absolute), (
        "black-to-move legal mask equals the absolute-frame mask on an "
        "asymmetric position — the STM remap silently reverted to identity"
    )


# ---------------------------------------------------------------------------
# Resignation tracker + calibration helpers
# ---------------------------------------------------------------------------


def test_value_to_p_win_endpoints_and_mid():
    assert _value_to_p_win(-1.0) == pytest.approx(0.0)
    assert _value_to_p_win(0.0) == pytest.approx(0.5)
    assert _value_to_p_win(1.0) == pytest.approx(1.0)
    # Clamps out-of-range inputs (safety net; MCTS shouldn't emit these).
    assert _value_to_p_win(-2.0) == pytest.approx(0.0)
    assert _value_to_p_win(2.0) == pytest.approx(1.0)


def test_wdl_to_expected_score_handles_draw_heavy_positions():
    assert _wdl_to_expected_score([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert _wdl_to_expected_score([0.0, 1.0, 0.0]) == pytest.approx(0.5)
    assert _wdl_to_expected_score([0.0, 0.0, 1.0]) == pytest.approx(0.0)
    assert _wdl_to_expected_score([0.05, 0.90, 0.05]) == pytest.approx(0.50)


def test_resign_tracker_fires_after_k_consecutive_low():
    t = ResignTracker(v_resign=0.05, k=3)
    assert t.record("white", 0.01) is False
    assert t.record("white", 0.02) is False
    assert t.record("white", 0.01) is True


def test_resign_tracker_resets_on_above_threshold():
    t = ResignTracker(v_resign=0.05, k=3)
    t.record("white", 0.01)
    t.record("white", 0.01)
    # One above-threshold value inside the window prevents firing.
    assert t.record("white", 0.50) is False
    assert t.record("white", 0.01) is False
    assert t.record("white", 0.01) is False
    # Now the sliding window is [0.50, 0.01, 0.01] — still blocked by 0.50.
    # The next low push evicts 0.50 and we have [0.01, 0.01, 0.01].
    assert t.record("white", 0.01) is True


def test_resign_tracker_is_per_side():
    """White's streak must not fire when it's black that has been losing."""
    t = ResignTracker(v_resign=0.05, k=3)
    assert t.record("black", 0.01) is False
    assert t.record("white", 0.99) is False  # white doing fine
    assert t.record("black", 0.01) is False
    assert t.record("white", 0.99) is False
    # Third low-from-black push: black's window is full and all low → fires.
    assert t.record("black", 0.01) is True
    # White has never been below threshold, so its window is empty-ish and
    # pushing a low value doesn't fire (only one below-threshold entry).
    assert t.record("white", 0.01) is False


def test_resign_tracker_disabled_returns_false():
    # Any non-positive k or v_resign disables the tracker entirely.
    t = ResignTracker(v_resign=0.0, k=3)
    assert t.record("white", 0.0) is False
    t2 = ResignTracker(v_resign=0.05, k=0)
    assert t2.record("white", 0.0) is False


def test_expected_score_threshold_does_not_resign_draw_heavy_positions():
    t = ResignTracker(v_resign=0.05, k=3)
    drawish = _wdl_to_expected_score([0.05, 0.90, 0.05])
    assert drawish == pytest.approx(0.50)
    assert t.record("white", drawish) is False
    assert t.record("white", drawish) is False
    assert t.record("white", drawish) is False


def test_resigned_outcome_white_loses():
    result, termination, wdl_white = _resigned_outcome("white")
    assert result == "black_win"
    assert termination == "resignation"
    assert wdl_white == [0.0, 0.0, 1.0]


def test_resigned_outcome_black_loses():
    result, termination, wdl_white = _resigned_outcome("black")
    assert result == "white_win"
    assert termination == "resignation"
    assert wdl_white == [1.0, 0.0, 0.0]


def test_resign_calibration_correct_cases():
    # Never fired — None.
    assert _resign_calibration_correct(None, "white_win") is None
    assert _resign_calibration_correct(None, "draw") is None

    # White would-resign is correct iff white actually lost.
    assert _resign_calibration_correct("white", "black_win") is True
    assert _resign_calibration_correct("white", "white_win") is False
    assert _resign_calibration_correct("white", "draw") is False

    # Black would-resign is correct iff black actually lost.
    assert _resign_calibration_correct("black", "white_win") is True
    assert _resign_calibration_correct("black", "black_win") is False
    assert _resign_calibration_correct("black", "draw") is False


def test_finalize_game_targets_on_resigned_game(tmp_path: Path) -> None:
    """Integration test for the resign break path.

    In ``_play_one_game_pcr`` we break the game loop *before* the final
    ``game.apply(best_move)``, then set ``game_len_plies = total_ply + 1``
    so that ``finalize_game_targets`` treats the resign position as if a
    game-ending move were played from it (MLH = 1, wdl_short = terminal).
    This test drives that exact shape end-to-end.

    Setup: 10 positions (plies 0..9), white-to-move at ply 9 resigns.
    wdl_terminal_white = [0,0,1] (white loses). game_len_plies = 10.
    """
    # Positions 0..9 alternate white/black. White is at even plies.
    sides = ("white", "black")
    positions = [
        _fake_position(ply=i, side=sides[i % 2], root_q=-0.9 if i % 2 == 0 else 0.9)
        for i in range(10)
    ]
    assert positions[9].side == "black"  # sanity: ply 9 is black-to-move
    # But for this test we actually want white (even-ply mover) to be
    # the resigning side for a cleaner assertion pattern. Re-shape:
    # put white at ply 9 as the resigner by using a 9-position game
    # ending at ply 8 (white-to-move) instead.
    positions = [
        _fake_position(ply=i, side=sides[i % 2], root_q=-0.9 if i % 2 == 0 else 0.9)
        for i in range(9)
    ]
    assert positions[8].side == "white"  # resigning side

    rec = GameRecord(
        positions=positions,
        game_id=42,
        model_version=3,
        started_at="2026-04-07T00:00:00+00:00",
        duration_s=5.0,
        result="black_win",
        termination="resignation",
        resigned_skipped=False,
        opening_hash="a" * 16,
        rng_seed=1,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.3,
        num_simulations=800,
        worker="test-worker",
        git_sha="abc123def456",
        num_total_positions=9,           # total_ply at break = 8, +1 = 9
        wdl_terminal_white=[0.0, 0.0, 1.0],  # white loses
        game_len_plies=9,                # == total_ply + 1
        resign_fired=True,
        resign_would_fire_ply=8,
        resign_would_fire_side="white",
        resign_calibration_correct=True,
    )
    finalize_game_targets(rec)

    # MLH at the resign position = 1 (one "virtual" move left).
    resign_pos = rec.positions[8]
    assert resign_pos.trace["mlh"] == 1, (
        "MLH at the resign position should be 1 (game_len - pos.ply = 9 - 8)"
    )
    # MLH at ply 0 = game_len = 9.
    assert rec.positions[0].trace["mlh"] == 9

    # wdl_terminal_stm at the resign position: white is STM, white lost,
    # so STM sees [W,D,L] = [0,0,1].
    assert resign_pos.trace["wdl_terminal_stm"] == [0.0, 0.0, 1.0]
    # Position at ply 0 (also white-to-move): same STM, same terminal.
    assert rec.positions[0].trace["wdl_terminal_stm"] == [0.0, 0.0, 1.0]
    # Position at ply 1 (black-to-move): flipped → black sees white's loss
    # as [1,0,0] from its own perspective.
    assert rec.positions[1].trace["wdl_terminal_stm"] == [1.0, 0.0, 0.0]

    # wdl_short at the resign position: target_ply = 8+8 = 16 >= game_len=9,
    # so it falls into the terminal branch and uses wdl_terminal_stm.
    assert resign_pos.trace["wdl_short_stm"] == [0.0, 0.0, 1.0]
    # wdl_short at an early position (ply 0): target = 8, game_len=9,
    # 8 < 9 → non-terminal branch → uses root_q at ply 8.
    # root_q at ply 8 = -0.9 (STM white, losing). Same-side as ply 0 →
    # no flip → _root_q_to_wdl(-0.9) = [0, 0.1, 0.9].
    expected_short_ply0 = _root_q_to_wdl(-0.9)
    np.testing.assert_allclose(
        rec.positions[0].trace["wdl_short_stm"], expected_short_ply0, atol=1e-6
    )

    # Round-trip through the NPZ writer to ensure the meta carries the
    # resign fields, not just the in-memory record.
    npz_path = tmp_path / "resign_full.npz"
    meta = write_samples_to_npz(npz_path, rec, num_move_indices=NUM_MOVES)
    loaded = json.loads(npz_path.with_suffix(".meta.json").read_text())
    assert loaded["termination"] == "resignation"
    assert loaded["result"] == "black_win"
    assert loaded["resign_fired"] is True
    assert loaded["resign_would_fire_side"] == "white"
    assert loaded["resign_calibration_correct"] is True
    assert meta == loaded

    # And the per-sample WDL target column in the NPZ reflects the
    # resigning-side loss — ply 8 (STM white) should have WDL [0, 0, 1].
    data = np.load(npz_path)
    np.testing.assert_allclose(data["wdl_terminal"][8], [0.0, 0.0, 1.0], atol=1e-6)
    # And ply 0 (STM white) too.
    np.testing.assert_allclose(data["wdl_terminal"][0], [0.0, 0.0, 1.0], atol=1e-6)
    # Ply 1 (STM black) flipped: [1, 0, 0].
    np.testing.assert_allclose(data["wdl_terminal"][1], [1.0, 0.0, 0.0], atol=1e-6)


def test_npz_meta_includes_resignation_fields(tmp_path: Path) -> None:
    """The per-game meta sidecar must carry the resignation audit fields so
    the false-positive rate can be computed offline from S3."""
    rec = _fake_record(game_len=4, wdl_white=[0.0, 0.0, 1.0])
    rec.resign_fired = True
    rec.resign_would_fire_ply = 3
    rec.resign_would_fire_side = "white"
    rec.resign_calibration_correct = True
    rec.termination = "resignation"
    rec.result = "black_win"
    finalize_game_targets(rec)
    npz_path = tmp_path / "resigned.npz"
    meta = write_samples_to_npz(npz_path, rec, num_move_indices=NUM_MOVES)
    loaded = json.loads(npz_path.with_suffix(".meta.json").read_text())
    for key in (
        "resign_fired", "resign_would_fire_ply", "resign_would_fire_side",
        "resign_calibration_correct",
    ):
        assert key in loaded, f"meta sidecar missing {key}"
    assert loaded["resign_fired"] is True
    assert loaded["resign_would_fire_ply"] == 3
    assert loaded["resign_would_fire_side"] == "white"
    assert loaded["resign_calibration_correct"] is True
    assert meta == loaded
