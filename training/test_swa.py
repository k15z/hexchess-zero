"""Tests for the SwaSnapshotBuffer averaging logic + BN-stat updater."""

import torch
import torch.nn as nn

from training.swa import (
    DEFAULT_PROMOTION_WEIGHTS,
    SwaSnapshotBuffer,
    update_bn_stats,
)


def _sd(v: float) -> dict[str, torch.Tensor]:
    """Make a tiny state_dict with one float tensor = v."""
    return {"w": torch.tensor([v, v, v], dtype=torch.float32)}


def test_empty_buffer_returns_none():
    buf = SwaSnapshotBuffer()
    assert buf.average() is None


def test_single_snapshot_identity():
    buf = SwaSnapshotBuffer()
    buf.append(_sd(5.0))
    avg = buf.average()
    assert torch.allclose(avg["w"], torch.tensor([5.0, 5.0, 5.0]))


def test_two_snapshot_renormalized():
    # weights [0.4, 0.3] renormalized to [4/7, 3/7]
    buf = SwaSnapshotBuffer()
    buf.append(_sd(1.0))   # oldest
    buf.append(_sd(2.0))   # newest (appendleft -> newest first)
    avg = buf.average()
    expected = 2.0 * (4.0 / 7.0) + 1.0 * (3.0 / 7.0)
    assert torch.allclose(avg["w"], torch.tensor([expected] * 3), atol=1e-5)


def test_full_buffer_promotion_weights():
    # newest to oldest: 10, 20, 30, 40
    buf = SwaSnapshotBuffer()
    for v in (40.0, 30.0, 20.0, 10.0):
        buf.append(_sd(v))
    avg = buf.average()
    # appendleft order: newest is 10, so weights hit 10,20,30,40
    expected = 0.4 * 10 + 0.3 * 20 + 0.2 * 30 + 0.1 * 40
    assert torch.allclose(avg["w"], torch.tensor([expected] * 3))


def test_buffer_evicts_oldest():
    buf = SwaSnapshotBuffer(max_snapshots=3, promotion_weights=(0.5, 0.3, 0.2))
    for v in (1.0, 2.0, 3.0, 4.0):  # 1 evicted
        buf.append(_sd(v))
    assert len(buf) == 3
    avg = buf.average()
    # newest first: 4, 3, 2
    expected = 0.5 * 4 + 0.3 * 3 + 0.2 * 2
    assert torch.allclose(avg["w"], torch.tensor([expected] * 3))


def test_nonfloat_tensors_copied_not_averaged():
    buf = SwaSnapshotBuffer()
    for i, v in enumerate((10.0, 20.0)):
        sd = {"w": torch.tensor([v]), "nbt": torch.tensor(i, dtype=torch.long)}
        buf.append(sd)
    avg = buf.average()
    # newest is last appended = i=1
    assert avg["nbt"].item() == 1
    assert avg["nbt"].dtype == torch.long


def test_promotion_weights_sum_to_one():
    assert abs(sum(DEFAULT_PROMOTION_WEIGHTS) - 1.0) < 1e-9


def test_update_bn_stats_runs_through_loader():
    """BN stats should update when forwarding fake batches."""
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4))
    bn = model[1]
    # Poison running stats so we can observe the reset.
    bn.running_mean.fill_(999.0)
    bn.running_var.fill_(0.001)

    batches = [(torch.randn(2, 3, 8, 8),) for _ in range(4)]
    update_bn_stats(model, batches, device="cpu")
    assert (bn.running_mean.abs() < 10).all()
    assert (bn.running_var > 0).all()


def test_update_bn_stats_noop_without_bn():
    model = nn.Sequential(nn.Linear(4, 4))
    update_bn_stats(model, [(torch.randn(1, 4),)], device="cpu")  # should not raise
