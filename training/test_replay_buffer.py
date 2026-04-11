"""Regression tests for the legacy imitation ``ReplayBuffer`` iterator.

These tests pin two behaviors that matter for the legal_mask schema fix:

  1. The iterator yields a 4-tuple ``(boards, policies, legal_mask, outcomes)``
     and the loaded legal mask is the true legality bitmap from the file
     (a strict superset of ``policy > 0``), not a visit-derived mask.

  2. Stale pre-schema-change .npz files without a ``legal_masks`` field
     cause the iterator to raise loudly instead of silently retrying. The
     bootstrap loop blocks on this iterator for its first batch, so a
     silent retry would hang the trainer instead of surfacing the problem.

The buffer's ``__init__`` normally downloads from S3; we bypass it by
constructing via ``__new__`` and filling the instance attributes directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from training.trainer_loop import ReplayBuffer


def _make_buffer(tmp_path: Path, files: list[Path]) -> ReplayBuffer:
    buf = ReplayBuffer.__new__(ReplayBuffer)
    buf.cache_dir = tmp_path
    buf.max_positions = 100
    buf.s3_prefix = "imitation/"
    buf.files = files
    buf.total_positions = sum(
        int(np.load(f)["outcomes"].shape[0]) if f.exists() else 0 for f in files
    )
    # Drain the shuffle buffer as soon as one file is consumed so tests
    # don't wait for 100k samples.
    buf.SHUFFLE_BUFFER_SIZE = 2
    return buf


def _write_imitation_npz(
    path: Path, *, n: int = 4, num_moves: int = 11, include_legal_masks: bool = True,
) -> None:
    rng = np.random.default_rng(0)
    boards = rng.standard_normal((n, 22, 11, 11)).astype(np.float32)
    policies = np.zeros((n, num_moves), dtype=np.float32)
    legal_masks = np.zeros((n, num_moves), dtype=bool)
    for i in range(n):
        # 5 legal, 3 visited — strict superset so tests can distinguish
        # legality from visits.
        legal_idx = rng.choice(num_moves, size=5, replace=False)
        legal_masks[i, legal_idx] = True
        visited = legal_idx[:3]
        pi = rng.random(3).astype(np.float32)
        pi /= pi.sum()
        policies[i, visited] = pi
    outcomes = np.tile(
        np.array([1.0, 0.0, 0.0], dtype=np.float32), (n, 1)
    )
    arrays = {
        "boards": boards,
        "policies": policies,
        "outcomes": outcomes,
    }
    if include_legal_masks:
        arrays["legal_masks"] = legal_masks
    np.savez_compressed(str(path), **arrays)


def test_replay_buffer_yields_true_legal_mask(tmp_path: Path) -> None:
    """The 4-tuple yielded by the iterator contains the written legality mask.

    Regression for the ``_legal_mask_from_policy(policy > 0)`` bug: we
    stamp a strict superset in the file (5 legal, 3 visited) and assert
    the buffer yields a mask wider than the visit set in every row.
    """
    npz = tmp_path / "imit.npz"
    _write_imitation_npz(npz)
    buf = _make_buffer(tmp_path, [npz])

    sample = next(iter(buf))
    assert len(sample) == 4
    boards_t, policies_t, legal_t, outcome_t = sample
    assert legal_t.dtype == torch.bool
    assert legal_t.shape == policies_t.shape
    p_positive = policies_t > 0
    # Every visited move is legal, and there is at least one legal but
    # unvisited move — the whole point of the schema change.
    assert (legal_t | ~p_positive).all()
    assert (legal_t & ~p_positive).any()


def test_replay_buffer_raises_on_missing_legal_masks(tmp_path: Path) -> None:
    """Stale .npz without legal_masks must fail loud, not retry silently."""
    stale = tmp_path / "stale.npz"
    _write_imitation_npz(stale, include_legal_masks=False)
    buf = _make_buffer(tmp_path, [stale])

    with pytest.raises(RuntimeError, match="legal_mask"):
        next(iter(buf))


def test_replay_buffer_error_names_missing_field(tmp_path: Path) -> None:
    """The schema error should name the missing field and the file path."""
    stale = tmp_path / "stale.npz"
    _write_imitation_npz(stale, include_legal_masks=False)
    buf = _make_buffer(tmp_path, [stale])

    with pytest.raises(RuntimeError) as excinfo:
        next(iter(buf))
    msg = str(excinfo.value)
    assert "legal_masks" in msg
    assert "stale.npz" in msg
