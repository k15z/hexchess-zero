"""Tests for deterministic Elo projection rebuild behavior."""

import json

from training.elo_service import _build_state


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

