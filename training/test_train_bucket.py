"""Tests for TrainBucket rate limiter."""

import pytest

from training.trainer_loop import TrainBucket


def test_seed_on_first_update():
    """First update seeds the bucket with total_positions * ratio."""
    bucket = TrainBucket(ratio=4.0)
    bucket.update(1000)
    assert bucket.tokens == 4000.0
    assert bucket.has_budget()


def test_seed_capped_by_max_seed():
    """Initial seed is capped by max_seed to prevent runaway budget on restart."""
    bucket = TrainBucket(ratio=4.0, max_seed=5000)
    bucket.update(1_000_000)  # would be 4M without cap
    assert bucket.tokens == 5000.0


def test_seed_uncapped_when_small():
    """max_seed doesn't inflate a small initial seed."""
    bucket = TrainBucket(ratio=4.0, max_seed=50000)
    bucket.update(100)  # 400, well under cap
    assert bucket.tokens == 400.0


def test_incremental_updates():
    """Subsequent updates only add tokens for new positions."""
    bucket = TrainBucket(ratio=4.0)
    bucket.update(1000)  # seed: 4000
    bucket.update(1000)  # no new positions
    assert bucket.tokens == 4000.0

    bucket.update(1200)  # 200 new positions
    assert bucket.tokens == 4800.0


def test_consume_reduces_tokens():
    bucket = TrainBucket(ratio=2.0)
    bucket.update(100)  # seed: 200
    bucket.consume(50)
    assert bucket.tokens == 150.0


def test_budget_exhaustion():
    """Bucket runs out after consuming all tokens."""
    bucket = TrainBucket(ratio=1.0)
    bucket.update(10)  # seed: 10
    bucket.consume(10)
    assert not bucket.has_budget()
    assert bucket.tokens == 0.0


def test_budget_goes_negative():
    bucket = TrainBucket(ratio=1.0)
    bucket.update(5)
    bucket.consume(7)
    assert bucket.tokens == -2.0
    assert not bucket.has_budget()


def test_refill_after_exhaustion():
    """New data refills an exhausted bucket."""
    bucket = TrainBucket(ratio=2.0)
    bucket.update(100)   # seed: 200
    bucket.consume(200)  # empty
    assert not bucket.has_budget()

    bucket.update(150)   # 50 new positions -> +100
    assert bucket.tokens == 100.0
    assert bucket.has_budget()


def test_default_consume_is_one():
    bucket = TrainBucket(ratio=1.0)
    bucket.update(5)
    bucket.consume()
    assert bucket.tokens == 4.0


def test_fractional_ratio():
    """Non-integer ratios work correctly."""
    bucket = TrainBucket(ratio=0.5)
    bucket.update(100)
    assert bucket.tokens == 50.0
    bucket.update(200)  # 100 new -> +50
    assert bucket.tokens == 100.0


def test_no_negative_new_positions():
    """If total_positions decreases (files pruned), no tokens subtracted."""
    bucket = TrainBucket(ratio=2.0)
    bucket.update(1000)  # seed: 2000
    bucket.update(800)   # positions decreased — should add 0, not subtract
    assert bucket.tokens == 2000.0


def test_zero_ratio_raises():
    """ratio must be positive."""
    with pytest.raises(ValueError):
        TrainBucket(ratio=0)


def test_negative_ratio_raises():
    with pytest.raises(ValueError):
        TrainBucket(ratio=-1.0)


def test_incremental_not_capped_by_max_seed():
    """max_seed only caps the initial seed, not subsequent refills."""
    bucket = TrainBucket(ratio=4.0, max_seed=100)
    bucket.update(50)  # seed: min(200, 100) = 100
    assert bucket.tokens == 100.0
    bucket.update(10050)  # 10000 new * 4.0 = 40000 added, no cap
    assert bucket.tokens == 40100.0


def test_max_tokens_caps_accumulation():
    """Tokens cannot exceed max_tokens, preventing unbounded surplus."""
    bucket = TrainBucket(ratio=4.0, max_tokens=1000.0)
    bucket.update(500)  # would be 2000, capped to 1000
    assert bucket.tokens == 1000.0

    bucket.consume(500)  # down to 500
    bucket.update(1000)  # 500 new * 4.0 = 2000 added -> 2500, capped to 1000
    assert bucket.tokens == 1000.0


def test_max_tokens_does_not_inflate():
    """max_tokens is a ceiling, not a floor."""
    bucket = TrainBucket(ratio=1.0, max_tokens=1000.0)
    bucket.update(50)  # 50 tokens, well under cap
    assert bucket.tokens == 50.0


def test_max_tokens_after_consume():
    """Cap only applies at update time, not after consume."""
    bucket = TrainBucket(ratio=1.0, max_tokens=100.0)
    bucket.update(200)  # capped to 100
    bucket.consume(80)  # down to 20, no re-capping
    assert bucket.tokens == 20.0


def test_production_config():
    """Simulate the actual production config: ratio=4, max_seed=5000, max_tokens=5000."""
    bucket = TrainBucket(ratio=4.0, max_seed=5000, max_tokens=5000.0)

    # Trainer starts with 2M existing positions — seed capped to 5000
    bucket.update(2_000_000)
    assert bucket.tokens == 5000.0

    # Train a full cycle
    bucket.consume(5000)
    assert bucket.tokens == 0.0
    assert not bucket.has_budget()

    # Workers produce 2000 new positions -> 8000 tokens, capped to 5000
    bucket.update(2_002_000)
    assert bucket.tokens == 5000.0

    # Train half a cycle, then workers add a small batch
    bucket.consume(3000)
    assert bucket.tokens == 2000.0
    bucket.update(2_002_500)  # 500 new -> +2000, total 4000, under cap
    assert bucket.tokens == 4000.0
