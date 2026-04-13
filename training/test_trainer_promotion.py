"""Tests for trainer promotion cadence helpers."""

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
