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
    # Plan §6.3 calls for 5 categories eventually; initial stub covers 4.
    assert "opening" in cats
    assert "tactical" in cats
    assert "endgame" in cats
    assert "middlegame" in cats
