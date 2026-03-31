"""Tests for TrainBucket rate limiter."""

from training.trainer_loop import TrainBucket


def test_seed_on_first_update():
    """First update seeds the bucket with total_positions * ratio."""
    bucket = TrainBucket(ratio=4.0)
    bucket.update(1000)
    assert bucket.tokens == 4000.0
    assert bucket.has_budget()


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
