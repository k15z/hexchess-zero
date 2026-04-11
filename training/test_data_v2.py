"""Tests for the chunk-6 v2 .npz loader + target-dict builder."""

from pathlib import Path

import numpy as np
import pytest

from training.data_v2 import (
    V2Batch,
    build_targets_dict,
    load_v2_npz,
)


def _write_synthetic_npz(path: Path, n: int = 5, num_moves: int = 17, *,
                         all_full: bool = True) -> None:
    rng = np.random.default_rng(42)
    boards = rng.integers(-2, 3, size=(n, 22, 11, 11), dtype=np.int8)
    # Each position: visit distribution on 3 of 5 legal moves.
    # The legal_mask is wider than the visited set on purpose so tests can
    # verify that load_v2_npz reads it as-is instead of deriving it from
    # `policy > 0` (which would drop the 2 legal-but-unvisited moves).
    policy = np.zeros((n, num_moves), dtype=np.float16)
    legal_mask = np.zeros((n, num_moves), dtype=bool)
    for i in range(n):
        legal_idx = rng.choice(num_moves, size=5, replace=False)
        legal_mask[i, legal_idx] = True
        visited = legal_idx[:3]
        pi = rng.random(3).astype(np.float32)
        pi /= pi.sum()
        policy[i, visited] = pi.astype(np.float16)
    policy_aux_opp = np.zeros((n, num_moves), dtype=np.float16)
    policy_aux_opp[:, 0] = 1.0  # aux collapsed on move 0
    wdl_terminal = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (n, 1))
    wdl_short = np.tile(np.array([0.5, 0.25, 0.25], dtype=np.float32), (n, 1))
    mlh = np.arange(n, dtype=np.int16)
    if all_full:
        was_full_search = np.ones((n,), dtype=bool)
    else:
        was_full_search = np.zeros((n,), dtype=bool)
        was_full_search[::2] = True
    root_q = np.zeros((n,), dtype=np.float16)
    root_n = np.full((n,), 800, dtype=np.int32)
    root_entropy = np.zeros((n,), dtype=np.float16)
    nn_value_at_position = np.zeros((n,), dtype=np.float16)
    legal_count = np.full((n,), 5, dtype=np.int16)
    ply = np.arange(n, dtype=np.int16)
    game_id = np.full((n,), 1234, dtype=np.uint64)
    np.savez_compressed(
        str(path),
        boards=boards, policy=policy, policy_aux_opp=policy_aux_opp,
        legal_mask=legal_mask,
        wdl_terminal=wdl_terminal, wdl_short=wdl_short, mlh=mlh,
        was_full_search=was_full_search, root_q=root_q, root_n=root_n,
        root_entropy=root_entropy, nn_value_at_position=nn_value_at_position,
        legal_count=legal_count, ply=ply, game_id=game_id,
    )


def test_loads_v2_npz(tmp_path: Path):
    path = tmp_path / "sample.npz"
    _write_synthetic_npz(path, n=8)
    batch = load_v2_npz(path)
    assert isinstance(batch, V2Batch)
    assert len(batch) == 8
    assert batch.boards.dtype == np.float32
    assert batch.boards.shape == (8, 22, 11, 11)
    assert batch.policy.dtype == np.float32
    assert batch.aux_policy.dtype == np.float32
    assert batch.wdl_terminal.shape == (8, 3)
    assert batch.wdl_short.shape == (8, 3)
    assert batch.mlh.dtype == np.float32
    assert batch.legal_mask.dtype == bool


def test_filters_non_full_search_rows(tmp_path: Path):
    path = tmp_path / "mixed.npz"
    _write_synthetic_npz(path, n=6, all_full=False)
    batch = load_v2_npz(path)
    # Only indices 0,2,4 are full-search
    assert len(batch) == 3


def test_empty_full_search_raises(tmp_path: Path):
    path = tmp_path / "empty.npz"
    n = 3
    arrays = {
        "boards": np.zeros((n, 22, 11, 11), dtype=np.int8),
        "policy": np.zeros((n, 5), dtype=np.float16),
        "policy_aux_opp": np.zeros((n, 5), dtype=np.float16),
        "legal_mask": np.zeros((n, 5), dtype=bool),
        "wdl_terminal": np.zeros((n, 3), dtype=np.float32),
        "wdl_short": np.zeros((n, 3), dtype=np.float32),
        "mlh": np.zeros((n,), dtype=np.int16),
        "was_full_search": np.zeros((n,), dtype=bool),
        "root_q": np.zeros((n,), dtype=np.float16),
        "root_n": np.zeros((n,), dtype=np.int32),
        "root_entropy": np.zeros((n,), dtype=np.float16),
        "nn_value_at_position": np.zeros((n,), dtype=np.float16),
        "legal_count": np.zeros((n,), dtype=np.int16),
        "ply": np.zeros((n,), dtype=np.int16),
        "game_id": np.zeros((n,), dtype=np.uint64),
    }
    np.savez_compressed(str(path), **arrays)
    with pytest.raises(ValueError):
        load_v2_npz(path)


def test_load_v2_npz_raises_on_missing_legal_mask(tmp_path: Path):
    """Stale v2 .npz without legal_mask must raise a clear schema error.

    This guards the trainer: ``ReplayBufferV2`` lets the KeyError escape
    so stale pre-schema-change cached files crash loudly instead of
    silently hanging the retry loop.
    """
    path = tmp_path / "stale.npz"
    n, nm = 3, 5
    arrays = {
        "boards": np.zeros((n, 22, 11, 11), dtype=np.int8),
        "policy": np.zeros((n, nm), dtype=np.float16),
        "policy_aux_opp": np.zeros((n, nm), dtype=np.float16),
        # Deliberately omit legal_mask.
        "wdl_terminal": np.zeros((n, 3), dtype=np.float32),
        "wdl_short": np.zeros((n, 3), dtype=np.float32),
        "mlh": np.zeros((n,), dtype=np.int16),
        "was_full_search": np.ones((n,), dtype=bool),
        "ply": np.zeros((n,), dtype=np.int16),
    }
    np.savez_compressed(str(path), **arrays)
    with pytest.raises(KeyError, match="legal_mask"):
        load_v2_npz(path)


def test_legal_mask_includes_unvisited_legal_moves(tmp_path: Path):
    """Regression: the loader must read the written legal mask verbatim.

    The previous implementation derived ``legal_mask = policy > 0.0``, which
    silently dropped legal-but-unvisited moves from the softmax denominator
    and left the network unpenalized for placing mass on them. The synthetic
    fixture writes 5 legal moves but only 3 visits, so a strict superset
    relationship proves the mask survived the round trip.
    """
    path = tmp_path / "sample.npz"
    _write_synthetic_npz(path, n=4, num_moves=11)
    batch = load_v2_npz(path)
    assert batch.legal_mask.dtype == bool
    visit_mask = batch.policy > 0.0
    # Every visited move is legal.
    assert (batch.legal_mask | ~visit_mask).all()
    # And the mask is *strictly* wider than the visit set — there are
    # legal-but-unvisited moves in every row.
    assert ((batch.legal_mask & ~visit_mask).sum(axis=1) > 0).all()
    assert batch.legal_mask.sum(axis=1).min() > 0


def test_build_targets_dict_shape_keys(tmp_path: Path):
    path = tmp_path / "s.npz"
    _write_synthetic_npz(path, n=4, num_moves=9)
    batch = load_v2_npz(path)
    targets = build_targets_dict(batch)
    assert set(targets) == {"policy", "wdl", "mlh", "stv", "aux_policy"}
    assert targets["policy"].shape == (4, 9)
    assert targets["wdl"].shape == (4, 3)
    assert targets["mlh"].shape == (4,)
    assert targets["stv"].shape == (4, 3)
    assert targets["aux_policy"].shape == (4, 9)
