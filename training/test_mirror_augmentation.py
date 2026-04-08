"""Mirror-augmentation tests for the v2 data loader."""

import numpy as np
import pytest

from training.data_v2 import MIRROR_INDICES, V2Batch, mirror_batch

if MIRROR_INDICES is None:
    pytest.skip("hexchess binding not built with mirror_indices_array", allow_module_level=True)

NUM_MOVES = MIRROR_INDICES.shape[0]


def _make_batch(n: int = 4) -> V2Batch:
    rng = np.random.default_rng(0)
    return V2Batch(
        boards=rng.standard_normal((n, 22, 11, 11)).astype(np.float32),
        policy=rng.random((n, NUM_MOVES)).astype(np.float32),
        aux_policy=rng.random((n, NUM_MOVES)).astype(np.float32),
        wdl_terminal=rng.random((n, 3)).astype(np.float32),
        wdl_short=rng.random((n, 3)).astype(np.float32),
        mlh=rng.random(n).astype(np.float32),
        legal_mask=(rng.random((n, NUM_MOVES)) > 0.5),
        ply=np.arange(n, dtype=np.int32),
    )


def test_mirror_index_is_involution():
    assert MIRROR_INDICES.shape == (NUM_MOVES,)
    round_trip = MIRROR_INDICES[MIRROR_INDICES]
    assert np.array_equal(round_trip, np.arange(NUM_MOVES))


def test_mirror_one_hot_policy():
    one_hot = np.zeros((1, NUM_MOVES), dtype=np.float32)
    one_hot[0, 17] = 1.0
    batch = V2Batch(
        boards=np.zeros((1, 22, 11, 11), dtype=np.float32),
        policy=one_hot,
        aux_policy=one_hot.copy(),
        wdl_terminal=np.zeros((1, 3), dtype=np.float32),
        wdl_short=np.zeros((1, 3), dtype=np.float32),
        mlh=np.zeros(1, dtype=np.float32),
        legal_mask=(one_hot > 0),
        ply=np.zeros(1, dtype=np.int32),
    )
    mirrored = mirror_batch(batch)
    expected_idx = int(MIRROR_INDICES[17])
    assert mirrored.policy[0, expected_idx] == 1.0
    assert mirrored.policy[0, 17] == 0.0 or expected_idx == 17


def test_mirror_batch_is_involution():
    batch = _make_batch(3)
    twice = mirror_batch(mirror_batch(batch))
    assert np.allclose(twice.policy, batch.policy)
    assert np.allclose(twice.aux_policy, batch.aux_policy)
    assert np.allclose(twice.boards, batch.boards)
    assert np.array_equal(twice.legal_mask, batch.legal_mask)


def test_mirror_preserves_unchanged_targets():
    batch = _make_batch(2)
    mirrored = mirror_batch(batch)
    assert np.array_equal(mirrored.wdl_terminal, batch.wdl_terminal)
    assert np.array_equal(mirrored.wdl_short, batch.wdl_short)
    assert np.array_equal(mirrored.mlh, batch.mlh)
