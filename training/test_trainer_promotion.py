"""Tests for trainer candidate-export cadence helpers."""

import torch

from training.config import AsyncConfig
from training import trainer_loop
from training.trainer_loop import _promotion_check_ready


def test_promotion_check_ready_when_threshold_met_and_steps_advanced():
    assert _promotion_check_ready(
        new_positions=300_000,
        threshold=300_000,
        total_steps=1_500,
        last_attempt_step=1_000,
    )


def test_promotion_check_not_ready_below_threshold():
    assert not _promotion_check_ready(
        new_positions=299_999,
        threshold=300_000,
        total_steps=2_000,
        last_attempt_step=1_000,
    )


def test_promotion_check_not_ready_without_new_training_progress():
    assert not _promotion_check_ready(
        new_positions=350_000,
        threshold=300_000,
        total_steps=2_000,
        last_attempt_step=2_000,
    )


def test_threshold_export_publishes_without_version_special_case(monkeypatch):
    calls = []
    monkeypatch.setattr(
        trainer_loop,
        "_publish_candidate_model",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    cfg = AsyncConfig()
    cfg.promote_every_new_positions = 300_000

    current_version, watermark, attempt_step, published = trainer_loop._maybe_publish_candidate(
        cfg,
        torch.nn.Linear(1, 1),
        current_version=1,
        positions_at_last_promote=0,
        n_total=1_100_000,
        swa_buf=trainer_loop.SwaSnapshotBuffer(max_snapshots=1, promotion_weights=(1.0,)),
        bn_refresh_batches=[],
        device=torch.device("cpu"),
        total_steps_all_time=10_000,
        last_promotion_attempt_step=-1,
    )

    assert (current_version, watermark, attempt_step, published) == (
        2,
        1_100_000,
        10_000,
        True,
    )
    assert len(calls) == 1
