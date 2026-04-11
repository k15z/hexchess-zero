"""Promotion gating for the first 5 self-play promotions.

notes/13 §4.6:

    Gating is **off** in steady state (continuous Elo catches regressions).
    **On** for the first 5 promotions after bootstrap: a candidate must
    score >= 50% over a 200-game gauntlet against the current model.
    Escape hatch: after 3 consecutive failed gates, promote anyway.
    After 5 successful promotions, disable gating permanently.

State persisted in S3 at ``state/gate.json``:

    {
      "promotions_since_start": int,
      "consecutive_gate_failures": int,
      "gate_enabled": bool
    }

The state machine is decoupled from the actual gauntlet runner via a
``play_gauntlet`` callback so tests can inject a mock score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from . import storage

# Number of initial promotions to gate before switching to continuous mode.
GATE_PROMOTION_HORIZON = 5
# Consecutive-failure escape hatch (promote anyway after this many).
GATE_ESCAPE_FAILURES = 3
# Candidate must meet this score fraction to pass the gate.
GATE_PASS_THRESHOLD = 0.5


GauntletFn = Callable[[object, object], float]


@dataclass
class GateState:
    promotions_since_start: int = 0
    consecutive_gate_failures: int = 0
    gate_enabled: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GateState":
        return cls(
            promotions_since_start=int(d.get("promotions_since_start", 0)),
            consecutive_gate_failures=int(d.get("consecutive_gate_failures", 0)),
            gate_enabled=bool(d.get("gate_enabled", True)),
        )


@dataclass
class GateDecision:
    promote: bool
    reason: str     # "gate_disabled" | "pass" | "fail" | "escape_hatch"
    score: float    # candidate score fraction; 0.0 if skipped
    state: GateState


def decide_promotion(
    state: GateState,
    candidate,
    current,
    *,
    play_gauntlet: GauntletFn | None = None,
) -> GateDecision:
    """Advance the gate state machine for one candidate evaluation.

    Args:
        state: the current persisted ``GateState``.
        candidate, current: opaque objects the gauntlet knows how to play.
            Only used when the gate is enabled.
        play_gauntlet: function that plays the 200-game match and returns
            the candidate's score fraction in ``[0, 1]``. If ``None``,
            defaults to returning 0.0 (which would always fail) — tests
            should inject a mock.

    Returns:
        A ``GateDecision`` describing whether to promote and the new
        state that should be persisted afterwards.
    """
    s = GateState(**state.to_dict())  # copy

    if not s.gate_enabled:
        return GateDecision(
            promote=True, reason="gate_disabled", score=0.0, state=s,
        )

    gauntlet = play_gauntlet or (lambda _c, _cur: 0.0)
    score = float(gauntlet(candidate, current))

    if score >= GATE_PASS_THRESHOLD:
        s.promotions_since_start += 1
        s.consecutive_gate_failures = 0
        if s.promotions_since_start >= GATE_PROMOTION_HORIZON:
            s.gate_enabled = False
        return GateDecision(promote=True, reason="pass", score=score, state=s)

    s.consecutive_gate_failures += 1
    if s.consecutive_gate_failures >= GATE_ESCAPE_FAILURES:
        # Escape hatch: promote anyway, reset failure counter.
        s.consecutive_gate_failures = 0
        s.promotions_since_start += 1
        if s.promotions_since_start >= GATE_PROMOTION_HORIZON:
            s.gate_enabled = False
        return GateDecision(
            promote=True, reason="escape_hatch", score=score, state=s,
        )

    return GateDecision(promote=False, reason="fail", score=score, state=s)


# ---------------------------------------------------------------------------
# S3 helpers (thin wrappers; trainer imports these)
# ---------------------------------------------------------------------------

def load_gate_state() -> GateState:
    """Load ``state/gate.json`` from S3, or return defaults on miss."""
    try:
        return GateState.from_dict(storage.get_json(storage.GATE_STATE))
    except KeyError:
        return GateState()


def save_gate_state(state: GateState) -> None:
    storage.put_json(storage.GATE_STATE, state.to_dict())
