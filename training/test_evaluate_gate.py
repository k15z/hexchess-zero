from training.evaluate_gate import (
    _pair_bucket,
    _pair_sprt_snapshot,
    _sprt_bounds,
)


def test_pair_bucket_formats_half_point_scores() -> None:
    assert _pair_bucket(2.0) == "2.0"
    assert _pair_bucket(1.5) == "1.5"
    assert _pair_bucket(1.0) == "1.0"
    assert _pair_bucket(0.5) == "0.5"
    assert _pair_bucket(0.0) == "0.0"


def test_sprt_bounds_have_expected_order() -> None:
    lower, upper = _sprt_bounds(0.05, 0.05)
    assert lower < 0.0 < upper
    assert round(upper, 3) == 2.944
    assert round(lower, 3) == -2.944


def test_pair_sprt_snapshot_waits_before_min_games() -> None:
    snap = _pair_sprt_snapshot(
        pair_counts={"2.0": 1, "1.5": 1, "1.0": 2, "0.5": 0, "0.0": 0},
        total_games=10,
        alpha=0.05,
        beta=0.05,
        min_games=20,
        max_games=100,
        pass_score=0.55,
    )
    assert snap.status == "pending"


def test_pair_sprt_snapshot_reports_clear_approval() -> None:
    snap = _pair_sprt_snapshot(
        pair_counts={"2.0": 16, "1.5": 0, "1.0": 0, "0.5": 0, "0.0": 0},
        total_games=32,
        alpha=0.05,
        beta=0.05,
        min_games=20,
        max_games=100,
        pass_score=0.55,
    )
    assert snap.status == "approved"


def test_pair_sprt_snapshot_reports_clear_rejection() -> None:
    snap = _pair_sprt_snapshot(
        pair_counts={"2.0": 0, "1.5": 0, "1.0": 0, "0.5": 0, "0.0": 16},
        total_games=32,
        alpha=0.05,
        beta=0.05,
        min_games=20,
        max_games=100,
        pass_score=0.55,
    )
    assert snap.status == "rejected"


def test_pair_sprt_snapshot_stays_pending_on_draw_heavy_edge() -> None:
    snap = _pair_sprt_snapshot(
        pair_counts={"2.0": 3, "1.5": 1, "1.0": 15, "0.5": 1, "0.0": 0},
        total_games=40,
        alpha=0.05,
        beta=0.05,
        min_games=20,
        max_games=100,
        pass_score=0.55,
    )
    assert snap.status == "pending"
