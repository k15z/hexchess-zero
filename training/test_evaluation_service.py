import pytest

import training.evaluation_service as evaluation_service

from training.evaluation_service import (
    BENCHMARK_MAX_GAMES,
    _build_benchmark_summary,
    _build_decision,
    _build_gate_summary,
    _latest_candidate,
    _pair_bucket,
    _series_progress,
)


def _paired_records(candidate: str, opponent: str, outcomes: list[tuple[str, str]]) -> list[dict]:
    records = []
    for idx, (first, second) in enumerate(outcomes):
        pair_id = f"pair-{idx}"
        records.extend(
            [
                {
                    "candidate": candidate,
                    "opponent": opponent,
                    "white": candidate,
                    "black": opponent,
                    "outcome": first,
                    "termination": "checkmate_white" if first == "white" else "draw",
                    "moves": 30,
                    "white_time": 3.0,
                    "black_time": 2.0,
                    "white_moves": 15,
                    "black_moves": 15,
                    "pair_id": pair_id,
                    "pair_game_index": 0,
                },
                {
                    "candidate": candidate,
                    "opponent": opponent,
                    "white": opponent,
                    "black": candidate,
                    "outcome": second,
                    "termination": "checkmate_black" if second == "black" else "draw",
                    "moves": 28,
                    "white_time": 2.5,
                    "black_time": 2.8,
                    "white_moves": 14,
                    "black_moves": 14,
                    "pair_id": pair_id,
                    "pair_game_index": 1,
                },
            ]
        )
    return records


def test_latest_candidate_picks_newest_version_above_approved():
    versions = {
        1: "models/versions/1.onnx",
        2: "models/versions/2.onnx",
        4: "models/versions/4.onnx",
        5: "models/versions/5.onnx",
    }

    assert _latest_candidate(versions, 4) == 5
    assert _latest_candidate(versions, 5) is None


def test_series_progress_counts_completed_pairs_and_color_split():
    records = _paired_records(
        "v5",
        "v4",
        [("white", "black"), ("draw", "draw")],
    )

    progress = _series_progress(records, candidate="v5", opponent="v4")

    assert progress["wins"] == 2
    assert progress["draws"] == 2
    assert progress["losses"] == 0
    assert progress["completed_pairs"] == 2
    assert progress["pair_buckets"]["2.0"] == 1
    assert progress["pair_buckets"]["1.0"] == 1
    assert progress["color_split"]["candidate_white"]["games"] == 2
    assert progress["color_split"]["candidate_black"]["games"] == 2


def test_build_gate_summary_marks_clear_winner_approved():
    records = _paired_records("v6", "v5", [("white", "black")] * 16)

    summary = _build_gate_summary(
        candidate_version=6,
        approved_version=5,
        records=records,
    )

    assert summary["status"] == "approved"
    assert summary["games"] == 32
    assert summary["pair_buckets"]["2.0"] == 16


def test_build_benchmark_summary_applies_reference_tolerance():
    records = _paired_records("v6", "Minimax-2", [("draw", "draw")] * 12)

    summary = _build_benchmark_summary(
        candidate_version=6,
        approved_version=5,
        records=records,
        reference_scores={"Minimax-2": 0.52},
    )

    assert summary["per_opponent"]["Minimax-2"]["target_score"] == 0.47
    assert summary["per_opponent"]["Minimax-2"]["status"] == "approved"


def test_build_benchmark_summary_rejects_regression():
    records = _paired_records("v6", "Minimax-3", [("black", "white")] * 12)

    summary = _build_benchmark_summary(
        candidate_version=6,
        approved_version=5,
        records=records,
        reference_scores={"Minimax-3": 0.55},
    )

    assert summary["per_opponent"]["Minimax-3"]["status"] == "rejected"
    assert summary["status"] == "rejected"


def test_build_benchmark_summary_uses_regression_floor_at_max_games():
    records = _paired_records(
        "v6",
        "Minimax-4",
        [("draw", "draw")] * 11 + [("black", "draw")],
    )

    summary = _build_benchmark_summary(
        candidate_version=6,
        approved_version=5,
        records=records,
        reference_scores={"Minimax-4": 0.52},
    )

    assert summary["per_opponent"]["Minimax-4"]["score"] > 0.47
    assert summary["per_opponent"]["Minimax-4"]["score"] < 0.50
    assert summary["per_opponent"]["Minimax-4"]["status"] == "approved"


def test_build_benchmark_summary_without_reference_completes_after_full_suite():
    records = []
    for opponent in ("Heuristic", "Minimax-2", "Minimax-3", "Minimax-4"):
        records.extend(_paired_records("v5", opponent, [("draw", "draw")] * (BENCHMARK_MAX_GAMES // 2)))

    summary = _build_benchmark_summary(
        candidate_version=5,
        approved_version=5,
        records=records,
        reference_scores=None,
    )

    assert summary["reference_available"] is False
    assert summary["status"] == "complete"


def test_build_decision_requires_gate_and_benchmarks_for_promotion():
    gate_summary = {"status": "approved"}
    benchmark_summary = {"status": "approved", "reference_available": True}

    decision = _build_decision(
        candidate_version=6,
        approved_version=5,
        gate_summary=gate_summary,
        benchmark_summary=benchmark_summary,
    )

    assert decision["status"] == "promote"


def test_pair_bucket_formats_half_scores():
    assert _pair_bucket(2.0) == "2.0"
    assert _pair_bucket(1.5) == "1.5"


def test_run_service_refreshes_promoted_artifacts_with_new_approved_version(
    monkeypatch,
    tmp_path,
):
    class FakeConfig:
        run_id = "run-1"
        model_cache_dir = tmp_path

        def ensure_cache_dirs(self) -> None:
            pass

    class FakeRecords:
        def refresh(self, version: int) -> int:
            return 0

        def records(self, version: int) -> list[dict]:
            return []

    approved_versions = iter([5, 5, 6])
    refresh_calls: list[int] = []
    write_calls: list[dict] = []

    monkeypatch.setattr(evaluation_service, "AsyncConfig", lambda: FakeConfig())
    monkeypatch.setattr(evaluation_service, "setup_json_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluation_service, "log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluation_service, "PlayerProvider", lambda simulations, cache_dir: object())
    monkeypatch.setattr(evaluation_service, "VersionRecordStore", lambda: FakeRecords())
    monkeypatch.setattr(
        evaluation_service,
        "_discover_versions",
        lambda: [(6, "models/versions/6.onnx")],
    )
    monkeypatch.setattr(
        evaluation_service,
        "_read_approved_version",
        lambda: next(approved_versions),
    )
    monkeypatch.setattr(evaluation_service, "_needs_anchor_backfill", lambda approved_version: False)
    monkeypatch.setattr(
        evaluation_service,
        "_acquire_promotion_lease",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(evaluation_service, "_release_promotion_lease", lambda: None)
    monkeypatch.setattr(evaluation_service, "_promote_candidate", lambda version, model_key: None)

    def fake_refresh_artifacts(*, version: int, approved_version: int, records, include_gate: bool):
        refresh_calls.append(approved_version)
        if len(refresh_calls) == 1:
            return (
                {"status": "approved", "games": 20, "approved_version": approved_version},
                {"status": "approved", "reference_available": True},
                {"status": "promote"},
            )
        return (
            {"status": "approved", "games": 20, "approved_version": approved_version},
            {"status": "approved", "reference_available": True},
            {"status": "promote"},
        )

    def fake_write_artifacts(*, version: int, gate_summary: dict | None, benchmark_summary: dict, decision: dict):
        write_calls.append(
            {
                "version": version,
                "approved_version": None if gate_summary is None else gate_summary["approved_version"],
                "status": decision["status"],
            }
        )
        if decision["status"] == "promoted":
            raise SystemExit

    monkeypatch.setattr(evaluation_service, "_refresh_artifacts", fake_refresh_artifacts)
    monkeypatch.setattr(evaluation_service, "_write_artifacts", fake_write_artifacts)

    with pytest.raises(SystemExit):
        evaluation_service.run_evaluation_service(simulations=800)

    assert refresh_calls == [5, 6]
    assert write_calls[-1] == {
        "version": 6,
        "approved_version": 6,
        "status": "promoted",
    }
