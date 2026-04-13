"""Tests for deterministic Elo projection rebuild behavior."""

import json

from training import storage
from training.elo_service import (
    BASELINE_NAMES,
    _build_state,
    _gate_decision,
    _desired_active,
    _maybe_resolve_gate,
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


def test_desired_active_uses_approved_and_next_candidate(monkeypatch):
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

    assert active == [*BASELINE_NAMES, "v3", "v5"]
    assert version_keys["v5"] == "models/versions/5.onnx"
    assert approved_name == "v3"
    assert pending == "v5"
    assert gate_state["approved_version"] == 3


def test_select_pair_prefers_gate_matchup():
    state = {
        "active_players": [*BASELINE_NAMES, "v3", "v4"],
        "pair_results": {},
        "ratings": {},
    }

    assert _select_pair(state, approved_name="v3", gate_candidate="v4") == ("v3", "v4")


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


def test_maybe_resolve_gate_approves_candidate(monkeypatch):
    copied: list[tuple[str, str]] = []
    writes: list[tuple[str, dict]] = []

    monkeypatch.setattr(storage, "copy", lambda src, dst: copied.append((src, dst)))
    monkeypatch.setattr(storage, "put_json", lambda key, obj: writes.append((key, obj)))

    state = {
        "pair_results": {
            "v3:v4": {
                "a_wins": 2,
                "b_wins": 18,
                "draws": 0,
                "a_as_white": 10,
                "b_as_white": 10,
            }
        }
    }
    gate_state = {"approved_version": 3, "decisions": {}}

    gate_state, changed = _maybe_resolve_gate(
        state,
        {"v4": "models/versions/4.onnx"},
        "v3",
        "v4",
        gate_state,
    )

    assert changed
    assert gate_state["approved_version"] == 4
    assert gate_state["decisions"]["v4"]["status"] == "approved"
    assert gate_state["decisions"]["v4"]["games"] == 20
    assert copied == [("models/versions/4.onnx", storage.APPROVED_ONNX)]
    assert writes[0][0] == storage.APPROVED_META
    assert writes[1][0] == storage.GATE_STATE


def test_maybe_resolve_gate_rejects_candidate(monkeypatch):
    copied: list[tuple[str, str]] = []
    writes: list[tuple[str, dict]] = []

    monkeypatch.setattr(storage, "copy", lambda src, dst: copied.append((src, dst)))
    monkeypatch.setattr(storage, "put_json", lambda key, obj: writes.append((key, obj)))

    state = {
        "pair_results": {
            "v3:v4": {
                "a_wins": 18,
                "b_wins": 2,
                "draws": 0,
                "a_as_white": 10,
                "b_as_white": 10,
            }
        }
    }
    gate_state = {"approved_version": 3, "decisions": {}}

    gate_state, changed = _maybe_resolve_gate(
        state,
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
