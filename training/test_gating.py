"""Tests for the promotion-gating state machine."""

from training.gating import (
    GATE_ESCAPE_FAILURES,
    GATE_PROMOTION_HORIZON,
    GateState,
    decide_promotion,
)


def _run(score: float, state: GateState) -> "tuple[bool, str, GateState]":
    dec = decide_promotion(
        state, candidate="cand", current="cur",
        play_gauntlet=lambda _c, _cur: score,
    )
    return dec.promote, dec.reason, dec.state


def test_pass_first_gate():
    s = GateState()
    promote, reason, s2 = _run(0.55, s)
    assert promote
    assert reason == "pass"
    assert s2.promotions_since_start == 1
    assert s2.consecutive_gate_failures == 0
    assert s2.gate_enabled


def test_fail_increments_failure_counter():
    s = GateState()
    promote, reason, s2 = _run(0.3, s)
    assert not promote
    assert reason == "fail"
    assert s2.consecutive_gate_failures == 1
    assert s2.promotions_since_start == 0


def test_three_failures_escape_hatch():
    s = GateState()
    for _ in range(GATE_ESCAPE_FAILURES - 1):
        promote, _, s = _run(0.1, s)
        assert not promote
    promote, reason, s = _run(0.1, s)
    assert promote
    assert reason == "escape_hatch"
    # Failure counter resets after escape.
    assert s.consecutive_gate_failures == 0
    # Escape-hatch still counts as a promotion.
    assert s.promotions_since_start == 1


def test_five_passes_disable_gate():
    s = GateState()
    for _ in range(GATE_PROMOTION_HORIZON):
        promote, reason, s = _run(0.8, s)
        assert promote
        assert reason in ("pass", "escape_hatch")
    assert s.promotions_since_start == GATE_PROMOTION_HORIZON
    assert not s.gate_enabled


def test_disabled_gate_always_promotes_without_gauntlet():
    s = GateState(
        promotions_since_start=GATE_PROMOTION_HORIZON,
        consecutive_gate_failures=0,
        gate_enabled=False,
    )
    calls = {"n": 0}

    def gauntlet(_c, _cur):
        calls["n"] += 1
        return 0.0

    dec = decide_promotion(
        s, candidate="c", current="cur", play_gauntlet=gauntlet,
    )
    assert dec.promote
    assert dec.reason == "gate_disabled"
    assert calls["n"] == 0  # gauntlet never invoked


def test_pass_resets_failure_counter():
    s = GateState(consecutive_gate_failures=2)
    _, _, s2 = _run(0.9, s)
    assert s2.consecutive_gate_failures == 0


def test_threshold_exact_50_percent_passes():
    s = GateState()
    promote, reason, _ = _run(0.5, s)
    assert promote
    assert reason == "pass"


def test_gate_state_roundtrip():
    s = GateState(
        promotions_since_start=3, consecutive_gate_failures=1, gate_enabled=True,
    )
    d = s.to_dict()
    assert GateState.from_dict(d) == s
