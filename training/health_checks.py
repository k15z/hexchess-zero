"""Hard sanity-check invariants for the v2 rebuild (plan §7.5).

Any failure here should crash the trainer (CrashLoopBackoff) on startup.
A cheaper subset (`run_runtime_checks`) is safe to run every N steps.

All 11 hard invariants from the plan are implemented or stubbed:

1.  Move encoding round-trip (hexchess binding).
2.  Mirror table — DEFERRED (raises NotImplementedError on the stub).
3.  Initial loss bounds — delegates to ``losses.assert_healthy_initial_losses``.
4.  No NaN/Inf in loss or grads.
5.  No illegal-move probability mass after softmax.
6.  Validity mask (channel 17 has exactly 91 ones per position).
7.  Side-to-move piece-plane symmetry (population-level).
8.  BN eval mode (``model.training is False``).
9.  Model output shape dict.
10. TT hit-rate range check (pure function).
11. Repetition detection via threefold (short shuffle game).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import hexchess
import torch
import torch.nn.functional as F

from .losses import LossBreakdown, assert_healthy_initial_losses
from .model import NUM_MOVE_INDICES, HexChessNet


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class HealthCheckResult:
    name: str
    passed: bool
    message: str = ""


@dataclass
class HealthCheckReport:
    results: list[HealthCheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def failures(self) -> list[HealthCheckResult]:
        return [r for r in self.results if not r.passed]

    def add(self, result: HealthCheckResult) -> None:
        self.results.append(result)

    def __str__(self) -> str:  # pragma: no cover
        lines = [f"HealthCheckReport ({len(self.results)} checks):"]
        for r in self.results:
            tag = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{tag}] {r.name}: {r.message}")
        return "\n".join(lines)


class HealthCheckError(AssertionError):
    """Raised when an invariant fails in ``strict`` mode."""


# ---------------------------------------------------------------------------
# Individual invariant implementations
# ---------------------------------------------------------------------------


def check_move_encoding_round_trip(num_samples: int = 1000,
                                   seed: int = 0) -> HealthCheckResult:
    """Invariant 1: ``index_to_move(move_to_index(m)) == m`` for 1000 moves."""
    rng = random.Random(seed)
    checked = 0
    game = hexchess.Game()
    while checked < num_samples:
        if game.is_game_over():
            game = hexchess.Game()
            continue
        moves = game.legal_moves()
        if not moves:
            game = hexchess.Game()
            continue
        mv = rng.choice(moves)
        try:
            idx = hexchess.move_to_index(
                mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion
            )
            back = hexchess.index_to_move(idx)
        except Exception as exc:  # noqa: BLE001
            return HealthCheckResult(
                "move_encoding_round_trip",
                False,
                f"binding error at sample {checked}: {exc}",
            )
        if (back.from_q, back.from_r, back.to_q, back.to_r, back.promotion) != (
            mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion
        ):
            return HealthCheckResult(
                "move_encoding_round_trip",
                False,
                f"round-trip mismatch at index {idx}",
            )
        # Advance the game so we sample different positions.
        game.apply(mv)
        checked += 1
    return HealthCheckResult(
        "move_encoding_round_trip", True, f"{num_samples} moves ok"
    )


def check_mirror_table() -> HealthCheckResult:
    """Invariant 2: mirrored board has mirrored policy. **DEFERRED.**

    Needs a mirror symmetry implementation (hex-board reflection + move-index
    remap). Tracked for a later chunk.
    """
    raise NotImplementedError("deferred — needs mirror symmetry impl")


def check_initial_loss_bounds(breakdown: LossBreakdown,
                              num_legal_moves: int = 40) -> HealthCheckResult:
    """Invariant 3: initial losses are within physically sensible bounds."""
    try:
        assert_healthy_initial_losses(breakdown, num_legal_moves=num_legal_moves)
    except AssertionError as exc:
        return HealthCheckResult("initial_loss_bounds", False, str(exc))
    return HealthCheckResult("initial_loss_bounds", True, "within bounds")


def check_no_nan_inf(model: torch.nn.Module,
                    loss: torch.Tensor) -> HealthCheckResult:
    """Invariant 4: loss and every ``.grad`` tensor must be finite.

    Assumes ``loss.backward()`` has already been called (or will be) — we
    inspect ``p.grad`` on every parameter with a non-None grad.
    """
    if not torch.isfinite(loss).all():
        return HealthCheckResult(
            "no_nan_inf", False, f"loss is non-finite: {loss.detach().item()}"
        )
    bad: list[str] = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            bad.append(name)
    if bad:
        return HealthCheckResult(
            "no_nan_inf", False, f"non-finite grads: {bad[:5]} ({len(bad)} total)"
        )
    return HealthCheckResult("no_nan_inf", True, "loss + all grads finite")


def check_no_illegal_prob_mass(model: torch.nn.Module,
                               boards: torch.Tensor,
                               legal_mask: torch.Tensor,
                               tol: float = 1e-3) -> HealthCheckResult:
    """Invariant 5: per-sample sum of softmax over illegal moves < 1e-3."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            preds = model(boards)
        probs = F.softmax(preds["policy"].float(), dim=-1)
        illegal = (~legal_mask).float()
        illegal_mass = (probs * illegal).sum(dim=-1)
        max_mass = float(illegal_mass.max().item())
    finally:
        if was_training:
            model.train()
    if max_mass >= tol:
        return HealthCheckResult(
            "no_illegal_prob_mass",
            False,
            f"max illegal mass {max_mass:.4g} >= {tol}",
        )
    return HealthCheckResult(
        "no_illegal_prob_mass", True, f"max illegal mass {max_mass:.2e}"
    )


def check_validity_mask(boards: torch.Tensor,
                        channel: int = 17,
                        expected: int = 91) -> HealthCheckResult:
    """Invariant 6: channel 17 must contain exactly 91 ones per position."""
    plane = boards[:, channel]
    counts = torch.eq(plane, 1.0).sum(dim=(1, 2))
    bad = (counts != expected).nonzero(as_tuple=False).flatten().tolist()
    if bad:
        bad_counts = counts[bad[:5]].tolist()
        return HealthCheckResult(
            "validity_mask",
            False,
            f"{len(bad)} positions off; sample counts={bad_counts}",
        )
    return HealthCheckResult(
        "validity_mask", True, f"all {counts.numel()} positions have 91 valid cells"
    )


def check_stm_piece_plane_symmetry(boards: torch.Tensor,
                                   tol: float = 0.05) -> HealthCheckResult:
    """Invariant 7: mean(ch 0-5) ≈ mean(ch 6-11) across the batch (±5%)."""
    stm_mean = float(boards[:, 0:6].mean().item())
    opp_mean = float(boards[:, 6:12].mean().item())
    denom = max(abs(stm_mean) + abs(opp_mean), 1e-8) / 2.0
    rel = abs(stm_mean - opp_mean) / denom
    if rel > tol:
        return HealthCheckResult(
            "stm_plane_symmetry",
            False,
            f"stm={stm_mean:.5f} opp={opp_mean:.5f} rel={rel:.3f} > {tol}",
        )
    return HealthCheckResult(
        "stm_plane_symmetry",
        True,
        f"stm={stm_mean:.5f} opp={opp_mean:.5f} rel={rel:.3f}",
    )


def check_bn_eval_mode(model: torch.nn.Module) -> HealthCheckResult:
    """Invariant 8: ``model.training`` must be False (e.g. before ONNX export)."""
    if model.training:
        return HealthCheckResult(
            "bn_eval_mode", False, "model.training is True — call model.eval()"
        )
    return HealthCheckResult("bn_eval_mode", True, "model.training is False")


def check_model_output_shape(model: torch.nn.Module) -> HealthCheckResult:
    """Invariant 9: forward pass on (2, 22, 11, 11) returns the 5-head dict."""
    expected = {
        "policy": (2, NUM_MOVE_INDICES),
        "wdl": (2, 3),
        "mlh": (2, 1),
        "stv": (2, 3),
        "aux_policy": (2, NUM_MOVE_INDICES),
    }
    was_training = model.training
    model.eval()
    try:
        dummy = torch.zeros(2, 22, 11, 11)
        # Match device of the first param, if any.
        try:
            dev = next(model.parameters()).device
            dummy = dummy.to(dev)
        except StopIteration:
            pass
        with torch.no_grad():
            out = model(dummy)
    except Exception as exc:  # noqa: BLE001
        if was_training:
            model.train()
        return HealthCheckResult("model_output_shape", False, f"forward raised: {exc}")
    if was_training:
        model.train()
    if not isinstance(out, dict):
        return HealthCheckResult(
            "model_output_shape", False, f"expected dict, got {type(out).__name__}"
        )
    for key, shape in expected.items():
        if key not in out:
            return HealthCheckResult(
                "model_output_shape", False, f"missing head '{key}'"
            )
        actual = tuple(out[key].shape)
        if actual != shape:
            return HealthCheckResult(
                "model_output_shape",
                False,
                f"{key}: expected {shape}, got {actual}",
            )
    return HealthCheckResult("model_output_shape", True, "all 5 heads OK")


def check_tt_hit_rate(rate: float,
                      lo: float = 0.10,
                      hi: float = 0.90) -> bool:
    """Invariant 10: TT hit rate must be in [lo, hi]. Pure function for workers."""
    return lo <= rate <= hi


def _tt_hit_rate_result(rate: float | None) -> HealthCheckResult:
    if rate is None:
        return HealthCheckResult(
            "tt_hit_rate", True, "no sample provided (skipped)"
        )
    ok = check_tt_hit_rate(rate)
    return HealthCheckResult(
        "tt_hit_rate", ok, f"rate={rate:.3f} {'in' if ok else 'out of'} [0.10, 0.90]"
    )


def check_repetition_detection() -> HealthCheckResult:
    """Invariant 11: threefold repetition is detected after a short shuffle.

    We play an 8-ply king-shuffle from the initial position: white advances a
    knight/king out and shuffles between two cells for 4 reps. Glinski's
    starting position has the king on g1 with two legal king moves, and at
    least one of those targets has a return square; we construct the sequence
    by finding two legal king moves that form a 2-cycle for each side.

    If no such cycle is available without triggering some other rule (50-move
    clock, insufficient material), this check returns a skipped-pass with a
    note — a full explicit repetition test is tracked as a TODO.
    """
    game = hexchess.Game()

    def _find_cycle_move(g: Any) -> Any | None:
        """Find a legal move m such that after applying m, there is a legal
        reply m' whose to-square equals m's from-square (piece returns)."""
        for m in g.legal_moves():
            if m.promotion is not None:
                continue
            # Simulate.
            tmp = g.clone()
            try:
                tmp.apply(m)
            except Exception:  # noqa: BLE001
                continue
            if tmp.is_game_over():
                continue
            for reply in tmp.legal_moves():
                if reply.promotion is not None:
                    continue
                if (reply.from_q, reply.from_r) == (m.to_q, m.to_r) and (
                    reply.to_q, reply.to_r
                ) == (m.from_q, m.from_r):
                    return (m, reply)
        return None

    white_cycle = _find_cycle_move(game)
    if white_cycle is None:
        return HealthCheckResult(
            "repetition_detection",
            True,
            "skipped: no simple 2-cycle from initial position (TODO: build explicit position)",
        )
    w_out, w_back = white_cycle
    # Play one ply to hand the move to black, then find black's cycle.
    game.apply(w_out)
    black_cycle = _find_cycle_move(game)
    game.undo_move()
    if black_cycle is None:
        return HealthCheckResult(
            "repetition_detection",
            True,
            "skipped: no black 2-cycle available (TODO: build explicit position)",
        )
    b_out, b_back = black_cycle

    # Play: w_out, b_out, w_back, b_back, w_out, b_out, w_back, b_back
    # After the 3rd occurrence of the starting position, the game should end.
    sequence = [w_out, b_out, w_back, b_back] * 3
    for mv in sequence:
        if game.is_game_over():
            break
        try:
            game.apply(mv)
        except Exception as exc:  # noqa: BLE001
            return HealthCheckResult(
                "repetition_detection",
                False,
                f"failed to apply shuffle move: {exc}",
            )

    if not game.is_game_over():
        return HealthCheckResult(
            "repetition_detection",
            False,
            f"game not over after 12-ply shuffle (status={game.status()})",
        )
    status = game.status()
    if status != "draw_repetition":
        return HealthCheckResult(
            "repetition_detection",
            False,
            f"game over but status={status}, expected draw_repetition",
        )
    return HealthCheckResult(
        "repetition_detection", True, "threefold detected after shuffle"
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _add(report: HealthCheckReport, result: HealthCheckResult, *, strict: bool) -> None:
    report.add(result)
    if strict and not result.passed:
        raise HealthCheckError(f"[{result.name}] {result.message}")


def run_all_invariants(
    model: HexChessNet,
    batch: dict[str, torch.Tensor] | None = None,
    *,
    strict: bool = False,
    tt_hit_rate: float | None = None,
) -> HealthCheckReport:
    """Run every hard invariant that can be checked here.

    ``batch`` should be a dict with at least ``boards`` (B, 22, 11, 11) and
    ``legal_mask`` (B, num_moves) bool. If not provided, the batch-dependent
    checks (#5, #6, #7) are skipped.

    Invariant 2 (mirror table) is always skipped here — it raises
    ``NotImplementedError`` if called directly. Invariant 3 (initial losses)
    is only checked if the caller passes ``batch['loss_breakdown']``.
    Invariant 4 (NaN/Inf grads) is only checked if the caller passes
    ``batch['loss']``.
    """
    report = HealthCheckReport()

    # 1. Move encoding round-trip.
    _add(report, check_move_encoding_round_trip(), strict=strict)

    # 2. Mirror table (deferred).
    report.add(HealthCheckResult(
        "mirror_table", True, "deferred — see check_mirror_table docstring"
    ))

    # 9. Output shape (also exercises the model).
    _add(report, check_model_output_shape(model), strict=strict)

    # 8. BN eval mode — only meaningful if caller put the model in eval mode
    # before calling us (e.g. export path). We still report current state.
    _add(report, check_bn_eval_mode(model), strict=strict)

    # 11. Repetition detection.
    _add(report, check_repetition_detection(), strict=strict)

    if batch is not None:
        boards = batch.get("boards")
        legal_mask = batch.get("legal_mask")
        if boards is not None:
            _add(report, check_validity_mask(boards), strict=strict)
            _add(report, check_stm_piece_plane_symmetry(boards), strict=strict)
            if legal_mask is not None:
                _add(
                    report,
                    check_no_illegal_prob_mass(model, boards, legal_mask),
                    strict=strict,
                )
        bd = batch.get("loss_breakdown")
        if isinstance(bd, LossBreakdown):
            _add(report, check_initial_loss_bounds(bd), strict=strict)
        loss = batch.get("loss")
        if isinstance(loss, torch.Tensor):
            _add(report, check_no_nan_inf(model, loss), strict=strict)

    # 10. TT hit rate (optional).
    _add(report, _tt_hit_rate_result(tt_hit_rate), strict=strict)

    return report


def run_runtime_checks(
    model: torch.nn.Module,
    loss: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> HealthCheckReport:
    """Fast subset safe to run every N training steps.

    Checks #4 (NaN/Inf), #5 (illegal prob mass), #6 (validity mask),
    and #8 (BN eval mode — here it checks the *current* mode only; the
    trainer expects ``model.training == True`` during training, so this is
    wired as a warn-only report, not a crash).
    """
    report = HealthCheckReport()
    report.add(check_no_nan_inf(model, loss))
    boards = batch.get("boards")
    legal_mask = batch.get("legal_mask")
    if boards is not None:
        report.add(check_validity_mask(boards))
        if legal_mask is not None:
            report.add(check_no_illegal_prob_mass(model, boards, legal_mask))
    # #8 is reported but not evaluated strictly here — training mode is fine
    # mid-training. We still surface it so callers can act on it.
    report.add(HealthCheckResult(
        "bn_eval_mode",
        True,
        f"model.training={model.training} (informational during training)",
    ))
    return report
