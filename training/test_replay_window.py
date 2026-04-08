"""Tests for the sublinear replay-window formula."""

import pytest

from training.replay_window import sublinear_window_size


def test_below_c_returns_n_total():
    assert sublinear_window_size(0) == 0
    assert sublinear_window_size(1_000) == 1_000
    assert sublinear_window_size(10_000) == 10_000


def test_at_c_returns_c():
    assert sublinear_window_size(25_000) == 25_000


def test_sublinear_growth():
    """Window grows with N_total but sublinearly."""
    w1 = sublinear_window_size(100_000)
    w2 = sublinear_window_size(1_000_000)
    w3 = sublinear_window_size(10_000_000)
    assert w1 < w2 < w3
    # Growth is sublinear: 10x N -> less than 10x window.
    assert w2 < 10 * w1
    assert w3 < 10 * w2


@pytest.mark.parametrize(
    "n_total,expected",
    [
        (100_000, 49_379),
        (1_000_000, 223_739),
        (10_000_000, 1_204_236),
    ],
)
def test_worked_examples(n_total, expected):
    """Worked examples from the KataGo formula with c=25k, alpha=0.75, beta=0.4."""
    got = sublinear_window_size(n_total)
    assert got == pytest.approx(expected, rel=0.01)


def test_never_below_c_once_past():
    """Sanity floor: once N_total > c, the window is at least c."""
    assert sublinear_window_size(26_000) >= 25_000
    assert sublinear_window_size(30_000) >= 25_000


def test_custom_c():
    """Custom c still satisfies window(c) == c."""
    assert sublinear_window_size(50_000, c=50_000) == 50_000
