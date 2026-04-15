"""Tests for deterministic Elo projection rebuild behavior."""

import json

from training import storage
from training.elo_service import (
    BASELINE_NAMES,
    _assign_colors,
    _build_state,
    _gate_decision,
    _gate_progress_from_records,
    _gate_sprt_decision,
    _desired_active,
    _maybe_resolve_gate,
    _pending_candidate,
    _select_pair,
)


def test_build_state_deterministic_under_record_permutation():
    # Intentionally share timestamps so tie-breakers matter.
    records = [
        {
            "timestamp": "2026-04-11T00:00:00+00:00",
            "white": "v2",
            "black": "Minimax-2",
            "outcome": "white",
            "moves": 10,
            "white_time": 1.1,
            "black_time": 1.0,
            "white_moves": 5,
            "black_moves": 5,
        },
        {
            "timestamp": "2026-04-11T00:00:00+00:00",
            "white": "v1",
            "black": "Minimax-2",
            "outcome": "draw",
            "moves": 12,
            "white_time": 1.2,
            "black_time": 1.3,
            "white_moves": 6,
            "black_moves": 6,
        },
        {
            "timestamp": "2026-04-11T00:00:00+00:00",
            "white": "v1",
            "black": "v2",
            "outcome": "black",
            "moves": 14,
            "white_time": 1.5,
            "black_time": 1.4,
            "white_moves": 7,
            "black_moves": 7,
        },
    ]
    active = ["Heuristic", "Minimax-2", "Minimax-3", "Minimax-4", "v1", "v2"]

    a = _build_state(records, active)
    b = _build_state(list(reversed(records)), active)

    # Strong idempotence check: byte-identical JSON projection.
    assert json.dumps(a, sort_keys=True, separators=(",", ":")) == json.dumps(
        b, sort_keys=True, separators=(",", ":")
    )


def test_desired_active_keeps_approved_candidate_and_recent_versions(monkeypatch):
    monkeypatch.setattr(
        "training.elo_service._discover_versions",
        lambda: [
            (1, "models/versions/1.onnx"),
            (2, "models/versions/2.onnx"),
            (3, "models/versions/3.onnx"),
            (4, "models/versions/4.onnx"),
            (5, "models/versions/5.onnx"),
        ],
    )
    monkeypatch.setattr("training.elo_service._read_approved_version", lambda: 3)
    monkeypatch.setattr(
        "training.elo_service._load_gate_state",
        lambda approved_version: {
            "approved_version": approved_version,
            "decisions": {"v4": {"status": "rejected"}},
        },
    )

    active, version_keys, approved_name, pending, gate_state = _desired_active(5)

    assert active == [*BASELINE_NAMES, "v3", "v5", "v4", "v2", "v1"]
    assert version_keys["v5"] == "models/versions/5.onnx"
    assert approved_name == "v3"
    assert pending == "v5"
    assert gate_state["approved_version"] == 3


def test_desired_active_pins_old_approved_version_inside_window(monkeypatch):
    monkeypatch.setattr(
        "training.elo_service._discover_versions",
        lambda: [
            (1, "models/versions/1.onnx"),
            (2, "models/versions/2.onnx"),
            (3, "models/versions/3.onnx"),
            (4, "models/versions/4.onnx"),
            (5, "models/versions/5.onnx"),
            (6, "models/versions/6.onnx"),
        ],
    )
    monkeypatch.setattr("training.elo_service._read_approved_version", lambda: 1)
    monkeypatch.setattr(
        "training.elo_service._load_gate_state",
        lambda approved_version: {"approved_version": approved_version, "decisions": {}},
    )

    active, _, approved_name, pending, _ = _desired_active(3)

    assert active == [*BASELINE_NAMES, "v1", "v2", "v6"]
    assert approved_name == "v1"
    assert pending == "v2"


def test_select_pair_prefers_gate_matchup():
    state = {
        "active_players": [*BASELINE_NAMES, "v3", "v4"],
        "pair_results": {},
        "ratings": {},
    }

    assert _select_pair(state, approved_name="v3", gate_candidate="v4") == ("v3", "v4")


def test_assign_colors_maps_sorted_pair_counts_back_to_requested_order():
    state = {
        "pair_results": {
            "v1:v4": {
                "a_wins": 0,
                "b_wins": 0,
                "draws": 0,
                "a_as_white": 0,
                "b_as_white": 5,
            },
        },
    }

    assert _assign_colors(state, "v4", "v1") == ("v1", "v4")


def test_gate_decision_approves_early_for_clear_winner():
    decision, games, score, half_width = _gate_decision(18, 2, 0)

    assert decision == "approved"
    assert games == 20
    assert score == 0.9
    assert half_width < (score - 0.55)


def test_gate_decision_rejects_early_for_clear_loser():
    decision, games, score, half_width = _gate_decision(2, 18, 0)

    assert decision == "rejected"
    assert games == 20
    assert score == 0.1
    assert score + half_width < 0.55


def test_gate_decision_waits_when_inconclusive():
    decision, games, score, half_width = _gate_decision(12, 8, 0)

    assert decision is None
    assert games == 20
    assert 0.0 < half_width
    assert score - half_width < 0.55 < score + half_width


def test_gate_decision_forces_result_at_max_games():
    decision, games, score, half_width = _gate_decision(50, 50, 0)

    assert decision == "rejected"
    assert games == 100
    assert score == 0.5
    assert half_width > 0.0


def test_gate_progress_counts_only_completed_pairs():
    records = [
        {
            "white": "v2",
            "black": "v1",
            "outcome": "white",
            "gate_candidate": "v2",
            "gate_incumbent": "v1",
            "gate_pair_id": "pair-a",
            "gate_pair_game_index": 0,
        },
        {
            "white": "v1",
            "black": "v2",
            "outcome": "draw",
            "gate_candidate": "v2",
            "gate_incumbent": "v1",
            "gate_pair_id": "pair-a",
            "gate_pair_game_index": 1,
        },
        {
            "white": "v2",
            "black": "v1",
            "outcome": "black",
            "gate_candidate": "v2",
            "gate_incumbent": "v1",
            "gate_pair_id": "pair-b",
            "gate_pair_game_index": 0,
        },
    ]

    progress = _gate_progress_from_records(records, "v1", "v2")

    assert progress["completed_pairs"] == 1
    assert progress["wins"] == 1
    assert progress["draws"] == 1
    assert progress["losses"] == 0
    assert progress["games"] == 2
    assert progress["pair_buckets"]["1.5"] == 1


def test_gate_sprt_decision_waits_for_more_evidence():
    decision, llr, lower, upper = _gate_sprt_decision(
        {"2.0": 3, "1.5": 1, "1.0": 15, "0.5": 1, "0.0": 0},
        40,
    )

    assert decision is None
    assert lower < llr < upper


def test_gate_sprt_decision_rejects_clear_loser():
    decision, llr, lower, upper = _gate_sprt_decision(
        {"2.0": 0, "1.5": 0, "1.0": 0, "0.5": 0, "0.0": 32},
        64,
    )

    assert decision == "rejected"
    assert llr <= lower
    assert lower < upper


def test_maybe_resolve_gate_waits_on_draw_heavy_positive_result(monkeypatch):
    copied: list[tuple[str, str]] = []
    writes: list[tuple[str, dict]] = []

    monkeypatch.setattr(storage, "copy", lambda src, dst: copied.append((src, dst)))
    monkeypatch.setattr(storage, "put_json", lambda key, obj: writes.append((key, obj)))

    records = []
    pair_specs = (
        [("white", "black")] * 3
        + [("white", "draw")]
        + [("draw", "draw")] * 15
        + [("black", "draw")]
    )
    for i, (first_outcome, second_outcome) in enumerate(pair_specs):
        pair_id = f"pair-{i}"
        records.extend([
            {
                "white": "v2",
                "black": "v1",
                "outcome": first_outcome,
                "gate_candidate": "v2",
                "gate_incumbent": "v1",
                "gate_pair_id": pair_id,
                "gate_pair_game_index": 0,
            },
            {
                "white": "v1",
                "black": "v2",
                "outcome": second_outcome,
                "gate_candidate": "v2",
                "gate_incumbent": "v1",
                "gate_pair_id": pair_id,
                "gate_pair_game_index": 1,
            },
        ])

    gate_state = {"approved_version": 1, "decisions": {}}
    gate_state, changed = _maybe_resolve_gate(
        records,
        {"v2": "models/versions/2.onnx"},
        "v1",
        "v2",
        gate_state,
    )

    assert not changed
    assert gate_state["approved_version"] == 1
    assert gate_state["decisions"] == {}
    assert copied == []
    assert writes == []


def test_maybe_resolve_gate_approves_candidate(monkeypatch):
    copied: list[tuple[str, str]] = []
    writes: list[tuple[str, dict]] = []

    monkeypatch.setattr(storage, "copy", lambda src, dst: copied.append((src, dst)))
    monkeypatch.setattr(storage, "put_json", lambda key, obj: writes.append((key, obj)))

    records = []
    for i in range(32):
        pair_id = f"pair-{i}"
        records.extend([
            {
                "white": "v4",
                "black": "v3",
                "outcome": "white",
                "gate_candidate": "v4",
                "gate_incumbent": "v3",
                "gate_pair_id": pair_id,
                "gate_pair_game_index": 0,
            },
            {
                "white": "v3",
                "black": "v4",
                "outcome": "black",
                "gate_candidate": "v4",
                "gate_incumbent": "v3",
                "gate_pair_id": pair_id,
                "gate_pair_game_index": 1,
            },
        ])
    gate_state = {"approved_version": 3, "decisions": {}}

    gate_state, changed = _maybe_resolve_gate(
        records,
        {"v4": "models/versions/4.onnx"},
        "v3",
        "v4",
        gate_state,
    )

    assert changed
    assert gate_state["approved_version"] == 4
    assert gate_state["decisions"]["v4"]["status"] == "approved"
    assert gate_state["decisions"]["v4"]["games"] == 64
    assert gate_state["decisions"]["v4"]["completed_pairs"] == 32
    assert copied == [("models/versions/4.onnx", storage.APPROVED_ONNX)]
    assert writes[0][0] == storage.APPROVED_META
    assert writes[1][0] == storage.GATE_STATE


def test_maybe_resolve_gate_rejects_candidate(monkeypatch):
    copied: list[tuple[str, str]] = []
    writes: list[tuple[str, dict]] = []

    monkeypatch.setattr(storage, "copy", lambda src, dst: copied.append((src, dst)))
    monkeypatch.setattr(storage, "put_json", lambda key, obj: writes.append((key, obj)))

    records = []
    for i in range(32):
        pair_id = f"pair-{i}"
        records.extend([
            {
                "white": "v4",
                "black": "v3",
                "outcome": "black",
                "gate_candidate": "v4",
                "gate_incumbent": "v3",
                "gate_pair_id": pair_id,
                "gate_pair_game_index": 0,
            },
            {
                "white": "v3",
                "black": "v4",
                "outcome": "white",
                "gate_candidate": "v4",
                "gate_incumbent": "v3",
                "gate_pair_id": pair_id,
                "gate_pair_game_index": 1,
            },
        ])
    gate_state = {"approved_version": 3, "decisions": {}}

    gate_state, changed = _maybe_resolve_gate(
        records,
        {"v4": "models/versions/4.onnx"},
        "v3",
        "v4",
        gate_state,
    )

    assert changed
    assert gate_state["approved_version"] == 3
    assert gate_state["decisions"]["v4"]["status"] == "rejected"
    assert copied == []
    assert writes == [(storage.GATE_STATE, gate_state)]


def test_pending_candidate_returns_none_when_everything_is_rejected():
    version_keys = {
        "v1": "models/versions/1.onnx",
        "v2": "models/versions/2.onnx",
        "v3": "models/versions/3.onnx",
    }
    gate_state = {
        "approved_version": 1,
        "decisions": {
            "v2": {"status": "rejected"},
            "v3": {"status": "rejected"},
        },
    }

    assert _pending_candidate(version_keys, 1, gate_state) is None
