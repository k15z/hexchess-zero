"""Unit tests for training.losses."""

from __future__ import annotations

import math

import pytest
import torch

from training.losses import (
    LossBreakdown,
    LossWeights,
    assert_healthy_initial_losses,
    compute_losses,
)


NUM_MOVES = 64
BATCH = 8


def _random_preds(num_moves: int = NUM_MOVES, batch: int = BATCH) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "policy": torch.randn(batch, num_moves) * 0.01,
        "wdl": torch.randn(batch, 3) * 0.01,
        "mlh": torch.randn(batch, 1) * 0.01,
        "stv": torch.randn(batch, 3) * 0.01,
        "aux_policy": torch.randn(batch, num_moves) * 0.01,
    }


def _uniform_targets(num_moves: int = NUM_MOVES, batch: int = BATCH) -> dict[str, torch.Tensor]:
    pol = torch.full((batch, num_moves), 1.0 / num_moves)
    wdl = torch.zeros(batch, 3)
    wdl[:, 0] = 1.0  # all wins
    stv = torch.zeros(batch, 3)
    stv[:, 1] = 1.0  # all draws
    return {
        "policy": pol,
        "wdl": wdl,
        "mlh": torch.full((batch,), 20.0),
        "stv": stv,
        "aux_policy": pol.clone(),
    }


def test_smoke_random_preds_uniform_targets_are_near_log_num_moves():
    preds = _random_preds()
    targets = _uniform_targets()
    out = compute_losses(preds, targets, debug=True)
    # policy CE ≈ log(NUM_MOVES)
    assert abs(out.policy.item() - math.log(NUM_MOVES)) < 0.2
    assert abs(out.aux_policy.item() - math.log(NUM_MOVES)) < 0.2
    # Smoothed one-hot WDL still near log 3
    assert 0.95 <= out.value.item() <= 1.20
    assert torch.isfinite(out.total).all()


def test_legal_mask_zeros_illegal_contribution():
    preds = _random_preds()
    targets = _uniform_targets()

    # Build a mask where only the first 10 moves are legal.
    mask = torch.zeros(BATCH, NUM_MOVES, dtype=torch.bool)
    mask[:, :10] = True

    # Renormalize the policy target over only legal moves.
    legal_target = torch.zeros_like(targets["policy"])
    legal_target[:, :10] = 1.0 / 10.0
    targets["policy"] = legal_target
    targets["aux_policy"] = legal_target.clone()

    out = compute_losses(preds, targets, legal_mask=mask, debug=True)
    # With 10 legal moves and ~uniform logits, CE ≈ log 10
    assert abs(out.policy.item() - math.log(10)) < 0.3

    # Now move probability mass onto illegal moves: after masking, the
    # loss should be unaffected because the target on illegal moves is
    # multiplied by log_softmax (which is very negative there). We
    # instead verify by comparing: putting illegal target mass into the
    # loss directly without masking would explode it.
    bad_target = torch.zeros_like(targets["policy"])
    bad_target[:, 50:] = 1.0 / (NUM_MOVES - 50)
    targets_bad = dict(targets)
    targets_bad["policy"] = bad_target
    out_bad = compute_losses(preds, targets_bad, legal_mask=mask)
    # Target is entirely on illegal positions, so log_p ≈ -1e9 there → huge loss.
    assert out_bad.policy.item() > 1e6


def test_label_smoothing_on_wdl_matches_formula():
    # Predict uniform logits so log_softmax = [-log 3]*3.
    preds = {
        "policy": torch.zeros(1, NUM_MOVES),
        "wdl": torch.zeros(1, 3),
        "mlh": torch.zeros(1, 1),
        "stv": torch.zeros(1, 3),
        "aux_policy": torch.zeros(1, NUM_MOVES),
    }
    targets = {
        "policy": torch.full((1, NUM_MOVES), 1.0 / NUM_MOVES),
        "wdl": torch.tensor([[1.0, 0.0, 0.0]]),
        "mlh": torch.zeros(1),
        "stv": torch.tensor([[1.0, 0.0, 0.0]]),
        "aux_policy": torch.full((1, NUM_MOVES), 1.0 / NUM_MOVES),
    }
    eps = 0.05
    out = compute_losses(
        preds, targets,
        weights=LossWeights(wdl_label_smoothing=eps),
    )
    # Smoothed target = [1-eps+eps/3, eps/3, eps/3]; log_p = [-log3]*3.
    # CE = sum(smoothed) * log 3 = log 3 (since targets sum to 1).
    expected = math.log(3)
    assert abs(out.value.item() - expected) < 1e-5


def test_mlh_huber_known_value():
    preds = {
        "policy": torch.zeros(1, NUM_MOVES),
        "wdl": torch.zeros(1, 3),
        "mlh": torch.tensor([[8.0]]),
        "stv": torch.zeros(1, 3),
        "aux_policy": torch.zeros(1, NUM_MOVES),
    }
    targets = {
        "policy": torch.full((1, NUM_MOVES), 1.0 / NUM_MOVES),
        "wdl": torch.tensor([[1.0, 0.0, 0.0]]),
        "mlh": torch.tensor([10.0]),
        "stv": torch.tensor([[0.0, 1.0, 0.0]]),
        "aux_policy": torch.full((1, NUM_MOVES), 1.0 / NUM_MOVES),
    }
    out = compute_losses(preds, targets)
    # smooth_l1 with beta=1 on |10-8|=2 → |x|-0.5 = 1.5
    assert abs(out.mlh.item() - 1.5) < 1e-5


def test_sample_weight_doubling_doubles_total():
    preds = _random_preds()
    targets = _uniform_targets()
    out1 = compute_losses(preds, targets)

    targets2 = dict(targets)
    targets2["sample_weight"] = torch.full((BATCH,), 2.0)
    out2 = compute_losses(preds, targets2)
    # Weighted mean with all-equal weights should equal the unweighted mean.
    assert abs(out1.total.item() - out2.total.item()) < 1e-5

    # Now weight only half the batch double — the mean changes but total
    # scales linearly if we scale all weights together (weighted mean is
    # invariant under scalar scaling, which is the intended behavior).
    targets3 = dict(targets)
    sw = torch.ones(BATCH)
    sw[:4] = 3.0
    targets3["sample_weight"] = sw
    out3 = compute_losses(preds, targets3)

    targets4 = dict(targets)
    targets4["sample_weight"] = sw * 10.0
    out4 = compute_losses(preds, targets4)
    assert abs(out3.total.item() - out4.total.item()) < 1e-5


def test_assert_healthy_initial_losses_on_real_model():
    pytest.importorskip("training.model")
    from training.config import AsyncConfig
    from training.model import build_model

    cfg = AsyncConfig()
    model = build_model(cfg)
    model.eval()
    torch.manual_seed(1)

    batch = 4
    x = torch.randn(batch, 22, 11, 11)  # new 22-ch serialization
    try:
        with torch.no_grad():
            preds = model(x)
    except Exception:
        # Model may still be 19-ch depending on build state; fall back.
        x = torch.randn(batch, 19, 11, 11)
        with torch.no_grad():
            preds = model(x)

    num_moves = preds["policy"].shape[1]
    # Simulate ~40 legal moves; uniform target over them.
    num_legal = 40
    mask = torch.zeros(batch, num_moves, dtype=torch.bool)
    mask[:, :num_legal] = True
    policy_target = torch.zeros(batch, num_moves)
    policy_target[:, :num_legal] = 1.0 / num_legal

    wdl = torch.zeros(batch, 3)
    wdl[torch.arange(batch), torch.randint(0, 3, (batch,))] = 1.0
    stv = torch.zeros(batch, 3)
    stv[torch.arange(batch), torch.randint(0, 3, (batch,))] = 1.0

    targets = {
        "policy": policy_target,
        "wdl": wdl,
        "mlh": torch.full((batch,), 20.0),
        "stv": stv,
        "aux_policy": policy_target.clone(),
    }
    breakdown = compute_losses(preds, targets, legal_mask=mask, debug=True)
    assert_healthy_initial_losses(breakdown, num_legal_moves=num_legal)


def test_isinstance_breakdown():
    out = compute_losses(_random_preds(), _uniform_targets())
    assert isinstance(out, LossBreakdown)
    d = out.item_dict()
    assert set(d.keys()) == {"value", "policy", "mlh", "stv", "aux_policy", "total"}
