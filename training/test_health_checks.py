"""Tests for training.health_checks (chunk 7)."""

from __future__ import annotations


import pytest
import torch
import torch.nn as nn

from training.config import _BaseConfig
from training.health_checks import (
    HealthCheckError,
    HealthCheckReport,
    check_bn_eval_mode,
    check_initial_loss_bounds,
    check_mirror_table,
    check_model_output_shape,
    check_move_encoding_round_trip,
    check_no_illegal_prob_mass,
    check_no_nan_inf,
    check_repetition_detection,
    check_stm_piece_plane_symmetry,
    check_tt_hit_rate,
    check_validity_mask,
    run_all_invariants,
    run_runtime_checks,
)
from training.losses import LossBreakdown
from training.model import NUM_MOVE_INDICES, build_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_model() -> nn.Module:
    cfg = _BaseConfig()
    return build_model(cfg).eval()


def _fake_loss_breakdown(p: float, v: float, a: float) -> LossBreakdown:
    t = lambda x: torch.tensor(float(x))  # noqa: E731
    return LossBreakdown(
        value=t(v), policy=t(p), mlh=t(0.0), stv=t(0.0),
        aux_policy=t(a), total=t(p + v + a),
    )


def _valid_board_batch(batch: int = 2) -> torch.Tensor:
    """Synthesize a (B, 22, 11, 11) tensor with channel 17 set to the 91
    valid hex cells and balanced STM/opponent piece planes."""
    boards = torch.zeros(batch, 22, 11, 11)
    for q in range(-5, 6):
        for r in range(-5, 6):
            if max(abs(q), abs(r), abs(q + r)) <= 5:
                col = q + 5
                row = r + 5
                boards[:, 17, row, col] = 1.0
    # Same tiny amount of "piece" on STM and opponent planes to satisfy
    # the population-symmetry check.
    boards[:, 0, 5, 5] = 1.0
    boards[:, 6, 5, 5] = 1.0
    return boards


# ---------------------------------------------------------------------------
# 1. Move encoding round-trip
# ---------------------------------------------------------------------------


def test_move_encoding_round_trip_passes():
    pytest.importorskip("hexchess")
    r = check_move_encoding_round_trip(num_samples=200)
    assert r.passed, r.message


# ---------------------------------------------------------------------------
# 2. Mirror table — STM-frame policy indexing consistency
# ---------------------------------------------------------------------------


def test_mirror_table_passes_on_live_moves():
    # White-to-move identity + black-to-move mirror remap must both hold.
    pytest.importorskip("hexchess")
    r = check_mirror_table()
    assert r.passed, r.message


def test_mirror_table_black_to_move_actually_remaps():
    # Sanity: when the encoder is STM-framed, at least some moves from a
    # black-to-move position must have game.policy_index != move_to_index.
    # (If someone silently reverts the encoder to absolute frame, this
    # catches it.)
    hexchess = pytest.importorskip("hexchess")

    game = hexchess.Game()
    game.apply(game.legal_moves()[0])
    assert game.side_to_move() == "black"
    any_remap = False
    for mv in game.legal_moves():
        absolute = hexchess.move_to_index(
            mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion
        )
        stm = game.policy_index(
            mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion
        )
        if absolute != stm:
            any_remap = True
            break
    assert any_remap, "black-to-move policy_index is always identity — STM frame inactive"


# ---------------------------------------------------------------------------
# 3. Initial loss bounds
# ---------------------------------------------------------------------------


def test_initial_loss_bounds_passes():
    # log(40) ≈ 3.689, log(3) ≈ 1.099 — but healthy window is [0.95, 1.20].
    bd = _fake_loss_breakdown(p=3.7, v=1.10, a=3.7)
    assert check_initial_loss_bounds(bd).passed


def test_initial_loss_bounds_fails_on_zero_logits():
    # WDL logits all zero => CE = log(3) ≈ 1.099 (OK), policy = log(N_moves) ~ large
    # Use an artificially too-low policy loss to force a failure.
    bd = _fake_loss_breakdown(p=0.1, v=1.10, a=0.1)
    r = check_initial_loss_bounds(bd)
    assert not r.passed


# ---------------------------------------------------------------------------
# 4. No NaN/Inf
# ---------------------------------------------------------------------------


def test_no_nan_inf_passes_for_healthy_grads():
    model = nn.Linear(3, 2)
    x = torch.randn(4, 3)
    y = model(x).sum()
    y.backward()
    r = check_no_nan_inf(model, y)
    assert r.passed


def test_no_nan_inf_fails_on_nan_grad():
    model = nn.Linear(3, 2)
    x = torch.randn(4, 3)
    y = model(x).sum()
    y.backward()
    # Corrupt one grad.
    with torch.no_grad():
        model.weight.grad[0, 0] = float("nan")
    r = check_no_nan_inf(model, y)
    assert not r.passed
    assert "non-finite" in r.message


def test_no_nan_inf_fails_on_nan_loss():
    model = nn.Linear(3, 2)
    loss = torch.tensor(float("inf"))
    r = check_no_nan_inf(model, loss)
    assert not r.passed


# ---------------------------------------------------------------------------
# 5. Illegal prob mass
# ---------------------------------------------------------------------------


class _UniformPolicyModel(nn.Module):
    def forward(self, x):  # noqa: D401
        b = x.shape[0]
        return {
            "policy": torch.zeros(b, NUM_MOVE_INDICES),
            "wdl": torch.zeros(b, 3),
            "mlh": torch.zeros(b, 1),
            "stv": torch.zeros(b, 3),
            "aux_policy": torch.zeros(b, NUM_MOVE_INDICES),
        }


class _MaskedPolicyModel(nn.Module):
    """Outputs -1e9 logits on illegal moves (given a fixed mask at construction)."""

    def __init__(self, legal_mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", legal_mask)

    def forward(self, x):  # noqa: D401
        b = x.shape[0]
        logits = torch.zeros(b, NUM_MOVE_INDICES)
        logits = logits.masked_fill(~self.mask, -1e9)
        return {
            "policy": logits,
            "wdl": torch.zeros(b, 3),
            "mlh": torch.zeros(b, 1),
            "stv": torch.zeros(b, 3),
            "aux_policy": torch.zeros(b, NUM_MOVE_INDICES),
        }


def test_illegal_prob_mass_fails_for_uniform_model():
    boards = _valid_board_batch(batch=2)
    legal = torch.zeros(2, NUM_MOVE_INDICES, dtype=torch.bool)
    legal[:, :40] = True  # 40 legal out of ~4000
    r = check_no_illegal_prob_mass(_UniformPolicyModel(), boards, legal)
    assert not r.passed


def test_illegal_prob_mass_passes_for_masked_model():
    boards = _valid_board_batch(batch=2)
    legal = torch.zeros(2, NUM_MOVE_INDICES, dtype=torch.bool)
    legal[:, :40] = True
    r = check_no_illegal_prob_mass(_MaskedPolicyModel(legal), boards, legal)
    assert r.passed, r.message


# ---------------------------------------------------------------------------
# 6. Validity mask
# ---------------------------------------------------------------------------


def test_validity_mask_passes():
    boards = _valid_board_batch(batch=3)
    r = check_validity_mask(boards)
    assert r.passed, r.message


def test_validity_mask_fails_when_broken():
    boards = _valid_board_batch(batch=2)
    boards[:, 17] = 0.0
    r = check_validity_mask(boards)
    assert not r.passed


# ---------------------------------------------------------------------------
# 7. STM piece plane symmetry
# ---------------------------------------------------------------------------


def test_stm_symmetry_passes_for_balanced_batch():
    boards = _valid_board_batch(batch=4)
    # _valid_board_batch already mirrors one piece on channel 0 and channel 6.
    r = check_stm_piece_plane_symmetry(boards)
    assert r.passed, r.message


def test_stm_symmetry_fails_for_all_white_batch():
    boards = _valid_board_batch(batch=4)
    boards[:, 6:12] = 0.0
    # Add more weight to stm to make the asymmetry strong.
    boards[:, 0, 4, 4] = 1.0
    r = check_stm_piece_plane_symmetry(boards)
    assert not r.passed


# ---------------------------------------------------------------------------
# 8. BN eval mode
# ---------------------------------------------------------------------------


def test_bn_eval_mode_passes_when_eval():
    m = nn.Linear(3, 3).eval()
    assert check_bn_eval_mode(m).passed


def test_bn_eval_mode_fails_when_train():
    m = nn.Linear(3, 3).train()
    assert not check_bn_eval_mode(m).passed


# ---------------------------------------------------------------------------
# 9. Model output shape
# ---------------------------------------------------------------------------


def test_model_output_shape_passes_for_real_model():
    m = _tiny_model()
    r = check_model_output_shape(m)
    assert r.passed, r.message


class _WrongShapeModel(nn.Module):
    def forward(self, x):  # noqa: D401
        b = x.shape[0]
        return {
            "policy": torch.zeros(b, 1),
            "wdl": torch.zeros(b, 3),
            "mlh": torch.zeros(b, 1),
            "stv": torch.zeros(b, 3),
            "aux_policy": torch.zeros(b, NUM_MOVE_INDICES),
        }


def test_model_output_shape_fails_for_wrong_shape():
    r = check_model_output_shape(_WrongShapeModel())
    assert not r.passed


# ---------------------------------------------------------------------------
# 10. TT hit rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rate,expected", [(0.05, False), (0.5, True), (0.95, False)])
def test_tt_hit_rate(rate, expected):
    assert check_tt_hit_rate(rate) is expected


# ---------------------------------------------------------------------------
# 11. Repetition detection
# ---------------------------------------------------------------------------


def test_repetition_detection():
    pytest.importorskip("hexchess")
    r = check_repetition_detection()
    # Either it passes fully (threefold triggered) or it skips with a note
    # because no 2-cycle is available from the initial position.
    assert r.passed, r.message


# ---------------------------------------------------------------------------
# Orchestration: run_all_invariants
# ---------------------------------------------------------------------------


def _good_batch(batch: int = 2) -> dict:
    boards = _valid_board_batch(batch=batch)
    legal = torch.zeros(batch, NUM_MOVE_INDICES, dtype=torch.bool)
    legal[:, :40] = True
    return {"boards": boards, "legal_mask": legal}


def test_run_all_invariants_strict_raises_on_failure():
    m = _tiny_model()
    # Force a validity-mask failure by zeroing channel 17.
    batch = _good_batch()
    batch["boards"][:, 17] = 0.0
    with pytest.raises(HealthCheckError):
        run_all_invariants(m, batch, strict=True)


def test_run_all_invariants_nonstrict_returns_report_with_failures():
    m = _tiny_model()
    batch = _good_batch()
    batch["boards"][:, 17] = 0.0
    report = run_all_invariants(m, batch, strict=False)
    assert not report.all_passed
    names = {f.name for f in report.failures()}
    assert "validity_mask" in names


def test_run_all_invariants_clean_run_passes():
    m = _tiny_model()
    # Don't pass legal_mask — the fresh random model will have high illegal
    # mass, which is expected and not a bug in the invariant. Illegal-mass
    # is tested in isolation above with a masked model.
    batch = {"boards": _valid_board_batch(batch=2)}
    report = run_all_invariants(m, batch, strict=False, tt_hit_rate=0.5)
    # Move-encoding round-trip, repetition detection, and mirror table all
    # require the hexchess binding. CI now installs it via ``uv sync``, but
    # keep tolerating binding-less environments for local/unit-only runs.
    binding_required = {
        "move_encoding_round_trip",
        "repetition_detection",
        "mirror_table",
    }
    failures = [
        r for r in report.failures()
        if r.name not in binding_required
        or "hexchess binding not available" not in r.message
    ]
    assert not failures, "\n".join(f"{r.name}: {r.message}" for r in failures)


# ---------------------------------------------------------------------------
# run_runtime_checks
# ---------------------------------------------------------------------------


def test_run_runtime_checks_basic():
    m = _tiny_model()
    batch = _good_batch(batch=2)
    loss = torch.tensor(1.23)
    report = run_runtime_checks(m, loss, batch)
    assert isinstance(report, HealthCheckReport)
    # Validity mask and no-NaN must pass; illegal prob mass may or may not,
    # but with a fresh random model on 2 samples the mass often exceeds 1e-3.
    names = {r.name for r in report.results}
    assert "no_nan_inf" in names
    assert "validity_mask" in names
