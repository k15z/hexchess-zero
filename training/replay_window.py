"""Sublinear replay window size (KataGo formula).

notes/06 §window, notes/13 §4.1:

    N_window(N_total) = c · (1 + β · ((N_total / c)^α − 1) / α)

with (for our small game) c=25_000, α=0.75, β=0.4. The window grows
sublinearly so old self-play games never dominate the buffer forever and
so the buffer stays well-mixed as N_total → ∞.
"""

from __future__ import annotations


def sublinear_window_size(
    n_total: int,
    c: int = 25_000,
    alpha: float = 0.75,
    beta: float = 0.4,
) -> int:
    """Return the KataGo sublinear replay-window size for N_total samples.

    At N_total == c the formula returns exactly c. Below c we simply
    return ``min(n_total, c)`` as a sanity floor (train on whatever we
    have during early bootstrap). Above c the formula expands
    sublinearly in N_total.
    """
    if n_total <= 0:
        return 0
    if n_total <= c:
        return min(n_total, c)
    ratio = n_total / c
    window = c * (1.0 + beta * ((ratio ** alpha) - 1.0) / alpha)
    window_int = int(round(window))
    # Sanity floor: never shrink below c once we're past it.
    return max(window_int, min(n_total, c))
