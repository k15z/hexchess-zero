"""Tests for training.benchmark_runner (plan §6.3)."""

from __future__ import annotations

from training import benchmark_runner


def test_positions_schema_valid():
    positions = benchmark_runner.load_positions()
    assert len(positions) >= 8, f"expected ≥8 positions, got {len(positions)}"
    benchmark_runner.validate_positions(positions)


def test_categories_covered():
    positions = benchmark_runner.load_positions()
    cats = {p["category"] for p in positions}
    # Plan §6.3 lists 5 categories. Endgame and drawn fixtures need
    # hand-curation (TODO) — until then we require the auto-generatable
    # ones (opening, tactical, middlegame).
    assert {"opening", "tactical", "middlegame"} <= cats
