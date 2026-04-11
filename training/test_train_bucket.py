"""Tests for TrainBucket rate limiter."""

import pytest

from training.trainer_loop import TrainBucket

BATCH_SIZE = 256


def test_seed_on_first_update():
    """First update seeds the bucket with total_positions * target_passes."""
    bucket = TrainBucket(target_passes=4.0, batch_size=BATCH_SIZE)
    bucket.update(1000)
    assert bucket.tokens == 4000.0
    assert bucket.has_budget()


def test_seed_capped_by_max_seed():
    """Initial seed is capped by max_seed to prevent runaway budget on restart."""
    bucket = TrainBucket(target_passes=4.0, batch_size=BATCH_SIZE, max_seed=5000)
    bucket.update(1_000_000)  # would be 4M without cap
    assert bucket.tokens == 5000.0


def test_seed_uncapped_when_small():
    """max_seed doesn't inflate a small initial seed."""
    bucket = TrainBucket(target_passes=4.0, batch_size=BATCH_SIZE, max_seed=50000)
    bucket.update(100)  # 400, well under cap
    assert bucket.tokens == 400.0


def test_incremental_updates():
    """Subsequent updates only add tokens for new positions."""
    bucket = TrainBucket(target_passes=4.0, batch_size=BATCH_SIZE)
    bucket.update(1000)  # seed: 4000
    bucket.update(1000)  # no new positions
    assert bucket.tokens == 4000.0

    bucket.update(1200)  # 200 new positions * 4.0 = 800
    assert bucket.tokens == 4800.0


def test_consume_subtracts_batch_size():
    """Each consume() call subtracts batch_size tokens (one step = one batch)."""
    bucket = TrainBucket(target_passes=4.0, batch_size=BATCH_SIZE)
    bucket.update(1000)  # seed: 4000
    bucket.consume()
    assert bucket.tokens == 4000.0 - BATCH_SIZE


def test_budget_exhaustion():
    """Bucket runs out after enough steps."""
    bucket = TrainBucket(target_passes=1.0, batch_size=10)
    bucket.update(100)  # seed: 100
    for _ in range(10):  # 10 steps * 10 batch_size = 100
        bucket.consume()
    assert not bucket.has_budget()
    assert bucket.tokens == 0.0


def test_budget_goes_negative():
    bucket = TrainBucket(target_passes=1.0, batch_size=10)
    bucket.update(5)  # seed: 5
    bucket.consume()  # -10 -> -5
    assert bucket.tokens == -5.0
    assert not bucket.has_budget()


def test_has_budget_requires_full_batch():
    """A partial batch's worth of tokens is not enough budget.

    Regression for a subtle off-by-one: `has_budget` used to return True
    for any positive token count, so a step with < batch_size tokens
    would fire and drive `_tokens` negative. Now the bucket must hold at
    least one full batch to claim budget.
    """
    bucket = TrainBucket(target_passes=1.0, batch_size=10)
    bucket.update(9)  # seed: 9 (one short of a batch)
    assert bucket.tokens == 9.0
    assert not bucket.has_budget()
    bucket.update(10)  # +1 new position -> tokens = 10
    assert bucket.tokens == 10.0
    assert bucket.has_budget()


def test_refill_after_exhaustion():
    """New data refills an exhausted bucket."""
    bucket = TrainBucket(target_passes=2.0, batch_size=10)
    bucket.update(100)   # seed: 200
    for _ in range(20):  # 20 * 10 = 200
        bucket.consume()
    assert not bucket.has_budget()

    bucket.update(200)   # 100 new positions * 2.0 = +200
    assert bucket.tokens == 200.0
    assert bucket.has_budget()


def test_fractional_target_passes():
    """Non-integer target_passes work correctly."""
    bucket = TrainBucket(target_passes=0.5, batch_size=10)
    bucket.update(100)
    assert bucket.tokens == 50.0
    bucket.update(200)  # 100 new -> +50
    assert bucket.tokens == 100.0


def test_no_negative_new_positions():
    """If total_positions decreases (files pruned), no tokens subtracted."""
    bucket = TrainBucket(target_passes=2.0, batch_size=BATCH_SIZE)
    bucket.update(1000)  # seed: 2000
    bucket.update(800)   # positions decreased — should add 0, not subtract
    assert bucket.tokens == 2000.0


def test_zero_target_passes_raises():
    with pytest.raises(ValueError):
        TrainBucket(target_passes=0, batch_size=BATCH_SIZE)


def test_negative_target_passes_raises():
    with pytest.raises(ValueError):
        TrainBucket(target_passes=-1.0, batch_size=BATCH_SIZE)


def test_incremental_not_capped_by_max_seed():
    """max_seed only caps the initial seed, not subsequent refills."""
    bucket = TrainBucket(target_passes=4.0, batch_size=BATCH_SIZE, max_seed=100)
    bucket.update(50)  # seed: min(200, 100) = 100
    assert bucket.tokens == 100.0
    bucket.update(10050)  # 10000 new * 4.0 = 40000 added, no cap
    assert bucket.tokens == 40100.0


def test_max_tokens_caps_accumulation():
    """Tokens cannot exceed max_tokens, preventing unbounded surplus."""
    bucket = TrainBucket(target_passes=4.0, batch_size=BATCH_SIZE, max_tokens=1000.0)
    bucket.update(500)  # would be 2000, capped to 1000
    assert bucket.tokens == 1000.0

    bucket.consume()  # -256
    bucket.update(1000)  # 500 new * 4.0 = 2000 added -> 2744, capped to 1000
    assert bucket.tokens == 1000.0


def test_max_tokens_does_not_inflate():
    """max_tokens is a ceiling, not a floor."""
    bucket = TrainBucket(target_passes=1.0, batch_size=BATCH_SIZE, max_tokens=1000.0)
    bucket.update(50)  # 50 tokens, well under cap
    assert bucket.tokens == 50.0


def test_passes_math():
    """Verify that target_passes=4 gives ~4 passes over data.

    With N new positions and target_passes=4:
      - tokens granted = N * 4
      - steps allowed = N * 4 / batch_size
      - samples drawn = steps * batch_size = N * 4
    So each position is expected to be drawn 4 times (= 4 passes).
    """
    bucket = TrainBucket(target_passes=4.0, batch_size=256)
    bucket.update(0)
    bucket.update(1000)  # 1000 new positions
    tokens = bucket.tokens  # should be 4000
    steps = tokens / 256   # 15.625 steps
    samples = steps * 256  # 4000 samples
    passes = samples / 1000  # 4.0 passes per new position
    assert passes == pytest.approx(4.0)


def test_window_size_caps_initial_seed():
    """Regression: max_seed must not exceed window_size * target_passes.

    notes/13 §4.2 — you cannot claim reuse credit for positions that are
    no longer in the replay window.
    """
    bucket = TrainBucket(
        target_passes=4.0, batch_size=256,
        max_seed=1_000_000, max_tokens=1_000_000,
    )
    # Huge total_positions but tiny window — only 25k positions are
    # actually in the buffer, so the initial seed must be <= 25k * 4.
    bucket.update(10_000_000, window_size=25_000)
    assert bucket.tokens == 25_000 * 4.0


def test_window_size_does_not_inflate_seed():
    """Window cap is a ceiling, not a floor."""
    bucket = TrainBucket(target_passes=4.0, batch_size=256, max_seed=1_000_000)
    bucket.update(1_000, window_size=1_000_000)
    # Without window cap this would already be 4000; cap is 4M, no effect.
    assert bucket.tokens == 4000.0


def test_production_config():
    """Simulate production: target_passes=4, batch_size=256, 2 workers.

    With target_passes=4 and batch_size=256, 3000 new positions grants
    12000 tokens = ~47 training steps. This correctly throttles the
    trainer to prevent overfitting on limited data.
    """
    max_tokens = 5000 * 256  # steps_per_cycle * batch_size
    bucket = TrainBucket(target_passes=4.0, batch_size=256,
                         max_seed=max_tokens, max_tokens=float(max_tokens))

    # Trainer starts with 2M existing positions — seed capped
    bucket.update(2_000_000)
    assert bucket.tokens == max_tokens

    # Train 1000 steps (one reload interval)
    for _ in range(1000):
        bucket.consume()
    assert bucket.tokens == max_tokens - 1000 * 256

    # Workers produce 3000 new positions -> 12000 tokens
    bucket.update(2_003_000)
    assert bucket.tokens == max_tokens - 1000 * 256 + 3000 * 4.0
