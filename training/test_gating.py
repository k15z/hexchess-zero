"""Tests for the promotion-gating state machine."""

from pathlib import Path

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


# ---------------------------------------------------------------------------
# _default_gauntlet — regression tests for the elo.play_game adapter.
#
# play_game returns {"outcome": "white" | "black" | "draw", ...}. A prior bug
# read result["winner"], so every game scored as a draw (0.5) regardless of
# the actual outcome. These tests pin the field name so the regression cannot
# recur silently.
# ---------------------------------------------------------------------------


class _StubPlayer:
    def __init__(self, name: str, **_kwargs):
        self.name = name


def _install_gauntlet_stubs(monkeypatch, outcomes):
    """Patch training.elo so _default_gauntlet runs without the hexchess binding.

    ``outcomes`` is a list of "white" | "black" | "draw" returned in order,
    one per play_game call.
    """
    import training.elo as elo_mod

    calls = {"games": 0}

    def fake_play_game(_white, _black, **_kw):
        i = calls["games"]
        calls["games"] += 1
        return {"outcome": outcomes[i], "moves": 0}

    monkeypatch.setattr(elo_mod, "MctsPlayer", _StubPlayer)
    monkeypatch.setattr(elo_mod, "play_game", fake_play_game)
    return calls


def _run_gauntlet(n_games: int) -> float:
    from training.trainer_loop import _default_gauntlet

    return _default_gauntlet(
        Path("cand.onnx"), Path("cur.onnx"),
        simulations=1, n_games=n_games,
    )


def test_default_gauntlet_candidate_sweep(monkeypatch):
    # 4 games, candidate plays white on even indices, black on odd.
    # Outcomes: W (cand wins), B (cand wins), W (cand wins), B (cand wins).
    _install_gauntlet_stubs(monkeypatch, ["white", "black", "white", "black"])
    assert _run_gauntlet(4) == 1.0


def test_default_gauntlet_current_sweep(monkeypatch):
    # Current wins every game regardless of color assignment.
    _install_gauntlet_stubs(monkeypatch, ["black", "white", "black", "white"])
    assert _run_gauntlet(4) == 0.0


def test_default_gauntlet_all_draws(monkeypatch):
    _install_gauntlet_stubs(monkeypatch, ["draw"] * 4)
    assert _run_gauntlet(4) == 0.5


def test_default_gauntlet_mixed(monkeypatch):
    # Game 0 (cand=white): white -> cand wins (1.0)
    # Game 1 (cand=black): white -> cand loses (0.0)
    # Game 2 (cand=white): draw -> 0.5
    # Game 3 (cand=black): black -> cand wins (1.0)
    # Total: 2.5 / 4 = 0.625
    _install_gauntlet_stubs(monkeypatch, ["white", "white", "draw", "black"])
    assert _run_gauntlet(4) == 0.625
