"""Multi-head loss module for HexChessNet training.

Implements the loss function described in notes/13 §3 and notes/07:

    L_total = λ_v · L_value
            + λ_p · L_policy
            + λ_mlh · L_mlh
            + λ_stv · L_stv
            + λ_aux_pol · L_aux_pol

L2 weight decay (``c_L2 = 3e-5``) is **not** computed here; it is applied
by the optimizer via ``weight_decay``. See notes/07.

All cross-entropy computations run in fp32 (inputs are cast up from
bf16/fp16 if needed) and use ``log_softmax`` rather than ``log(softmax)``
to avoid FP underflow on the ~4000-entry policy vector (notes/10 §13).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# Large-negative masking constant. Must be finite so that softmax is stable
# even when an entire row is illegal-padded (shouldn't happen in practice,
# but ``-inf`` would produce NaNs).
_MASK_FILL = -1e9

# Default label smoothing for the terminal WDL target.
_WDL_LABEL_SMOOTHING = 0.05


@dataclass
class LossWeights:
    """Scalar multipliers for each loss term (notes/13 §3)."""

    value: float = 1.5
    policy: float = 1.0
    mlh: float = 0.1
    stv: float = 0.15
    aux_policy: float = 0.15
    # MLH targets are raw plies (e.g. [0, 250]). Huber loss on raw plies
    # produces gradients ~|pred - target| in the L1 region, which dominates
    # the total loss for an untrained model. Normalize both pred and target
    # by this scale before Huber so the loss lives in [0, ~3]. The MLH
    # head's scalar output learns to predict in normalized units; multiply
    # by mlh_scale at inference time to recover plies.
    mlh_scale: float = 100.0
    # L2 is applied via optimizer weight_decay, not this module.
    l2: float = 3e-5

    # Label smoothing epsilon applied to the terminal WDL target only.
    wdl_label_smoothing: float = _WDL_LABEL_SMOOTHING


@dataclass
class LossBreakdown:
    """All individual loss terms plus the weighted total.

    Each field (except ``total``) is the *unweighted* mean loss so that
    the trainer can log raw values and compare them against the healthy
    initial-loss bounds.
    """

    value: torch.Tensor
    policy: torch.Tensor
    mlh: torch.Tensor
    stv: torch.Tensor
    aux_policy: torch.Tensor
    total: torch.Tensor
    extras: dict = field(default_factory=dict)

    def item_dict(self) -> dict[str, float]:
        """Return a plain-float dict for logging."""
        return {
            "value": float(self.value.detach().item()),
            "policy": float(self.policy.detach().item()),
            "mlh": float(self.mlh.detach().item()),
            "stv": float(self.stv.detach().item()),
            "aux_policy": float(self.aux_policy.detach().item()),
            "total": float(self.total.detach().item()),
        }


def _masked_log_softmax(
    logits: torch.Tensor,
    legal_mask: torch.Tensor | None,
) -> torch.Tensor:
    """fp32 log_softmax with optional illegal masking via ``-1e9``."""
    logits32 = logits.float()
    if legal_mask is not None:
        # legal_mask: True = legal. masked_fill fills where mask is True.
        logits32 = logits32.masked_fill(~legal_mask, _MASK_FILL)
    return F.log_softmax(logits32, dim=-1)


def _weighted_mean(
    per_sample: torch.Tensor,
    sample_weight: torch.Tensor | None,
) -> torch.Tensor:
    if sample_weight is None:
        return per_sample.mean()
    sw = sample_weight.to(per_sample.dtype).to(per_sample.device)
    return (per_sample * sw).sum() / sw.sum().clamp_min(1e-8)


def _ce_against_soft_target(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    legal_mask: torch.Tensor | None,
    sample_weight: torch.Tensor | None,
) -> torch.Tensor:
    """Soft-target cross-entropy ``-Σ π log p`` with fp32 log_softmax."""
    log_p = _masked_log_softmax(logits, legal_mask)
    tgt = target.float()
    per_sample = -(tgt * log_p).sum(dim=-1)
    return _weighted_mean(per_sample, sample_weight)


def _apply_label_smoothing(target: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0.0:
        return target
    num_classes = target.shape[-1]
    return target * (1.0 - eps) + (eps / num_classes)


def compute_losses(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    legal_mask: torch.Tensor | None = None,
    weights: LossWeights | None = None,
    debug: bool = False,
) -> LossBreakdown:
    """Compute the multi-head loss.

    Args:
        preds: Output dict from ``HexChessNet.forward``. Expected keys:
            ``policy``, ``wdl``, ``mlh``, ``stv``, ``aux_policy``.
        targets: Dict with keys ``policy``, ``wdl``, ``mlh``, ``stv``,
            ``aux_policy``, and optionally ``sample_weight``.
            - ``policy``      (B, num_moves) float, normalized (PTP-pruned).
            - ``wdl``         (B, 3) float one-hot (or soft) terminal.
            - ``mlh``         (B,) float plies-to-end.
            - ``stv``         (B, 3) float horizon-8 WDL target.
            - ``aux_policy``  (B, num_moves) float, normalized.
            - ``sample_weight`` (B,) float, optional, default 1.0 each.
        legal_mask: Optional ``(B, num_moves)`` bool tensor; ``True`` =
            legal. Applied identically to policy and aux_policy heads.
        weights: Loss weight overrides; see :class:`LossWeights`.
        debug: If True, assert that the total is finite (raises).

    Returns:
        A :class:`LossBreakdown` with every term plus the weighted total.

    Notes:
        - L2 weight decay is **not** included here; apply it via the
          optimizer's ``weight_decay`` parameter.
        - Cross-entropy is computed as ``-(target * log_softmax(logits)).sum``
          with fp32 softmax regardless of input dtype.
        - Illegal moves are masked with ``-1e9`` (not ``-inf``) before
          softmax.
    """
    w = weights or LossWeights()
    sample_weight = targets.get("sample_weight")

    # ---- Value (terminal WDL) with label smoothing ------------------------
    wdl_target = _apply_label_smoothing(targets["wdl"].float(), w.wdl_label_smoothing)
    log_p_wdl = F.log_softmax(preds["wdl"].float(), dim=-1)
    value_per_sample = -(wdl_target * log_p_wdl).sum(dim=-1)
    value_loss = _weighted_mean(value_per_sample, sample_weight)

    # ---- Main policy (masked soft-target CE) ------------------------------
    policy_loss = _ce_against_soft_target(
        preds["policy"],
        targets["policy"],
        legal_mask=legal_mask,
        sample_weight=sample_weight,
    )

    # ---- Moves-left (Huber / smooth L1, normalized to [0, ~3]) -----------
    mlh_pred = preds["mlh"].float().squeeze(-1) / w.mlh_scale
    mlh_target = targets["mlh"].float() / w.mlh_scale
    mlh_per_sample = F.smooth_l1_loss(mlh_pred, mlh_target, reduction="none", beta=1.0)
    mlh_loss = _weighted_mean(mlh_per_sample, sample_weight)

    # ---- Short-term value (no label smoothing) ----------------------------
    stv_target = targets["stv"].float()
    log_p_stv = F.log_softmax(preds["stv"].float(), dim=-1)
    stv_per_sample = -(stv_target * log_p_stv).sum(dim=-1)
    stv_loss = _weighted_mean(stv_per_sample, sample_weight)

    # ---- Auxiliary policy (UNMASKED soft-target CE) -----------------------
    # The aux target is the *opponent's* visit distribution at the next ply,
    # which has support over a different legal-move set than the current STM.
    # Applying the STM legal_mask here puts -1e9 on opponent-legal moves the
    # target has nonzero mass on, blowing up CE. Don't mask aux at all —
    # KataGo's opponent-reply head is unmasked too.
    aux_policy_loss = _ce_against_soft_target(
        preds["aux_policy"],
        targets["aux_policy"],
        legal_mask=None,
        sample_weight=sample_weight,
    )

    total = (
        w.value * value_loss
        + w.policy * policy_loss
        + w.mlh * mlh_loss
        + w.stv * stv_loss
        + w.aux_policy * aux_policy_loss
    )

    if debug:
        assert torch.isfinite(total).all(), (
            f"Non-finite loss: value={value_loss.item()} policy={policy_loss.item()} "
            f"mlh={mlh_loss.item()} stv={stv_loss.item()} aux={aux_policy_loss.item()}"
        )

    return LossBreakdown(
        value=value_loss,
        policy=policy_loss,
        mlh=mlh_loss,
        stv=stv_loss,
        aux_policy=aux_policy_loss,
        total=total,
    )


def assert_healthy_initial_losses(
    breakdown: LossBreakdown,
    num_legal_moves: int = 40,
) -> None:
    """Assert that a *fresh random init* produces physically sensible losses.

    Bounds (notes/13 §4, notes/07):
        - policy CE  ∈ [3.0, 5.0]   (≈ log 40)
        - wdl CE     ∈ [0.95, 1.20] (≈ log 3)
        - aux_policy ∈ [3.0, 5.0]

    Intended to be called exactly once on the first batch of a fresh
    training run. If this fails, something is wrong with encoding,
    masking, label-smoothing, or logit scale.
    """
    policy = float(breakdown.policy.detach().item())
    wdl = float(breakdown.value.detach().item())
    aux = float(breakdown.aux_policy.detach().item())
    mlh = float(breakdown.mlh.detach().item())
    # Normalized Huber: pred ≈ 0, target ≈ plies/100 ∈ [0, 3]. Untrained
    # loss is dominated by L1 region: |target| ≈ 0.5–2.0. Allow [0.0, 3.0].
    assert 0.0 <= mlh <= 3.0, (
        f"Initial MLH loss {mlh:.3f} outside [0.0, 3.0]. "
        "Check that mlh targets are scaled by losses.LossWeights.mlh_scale "
        "(default 100) and that the MLH head outputs are not saturated."
    )

    log_legal = math.log(max(num_legal_moves, 2))
    lower_p, upper_p = max(3.0, log_legal - 1.0), 5.0
    assert lower_p <= policy <= upper_p, (
        f"Initial policy loss {policy:.3f} outside [{lower_p:.2f}, {upper_p:.2f}] "
        f"(expected ≈ log({num_legal_moves}) = {log_legal:.3f}). "
        "Check logit masking / target normalization."
    )
    assert 0.95 <= wdl <= 1.20, (
        f"Initial WDL loss {wdl:.3f} outside [0.95, 1.20] (expected ≈ log 3 = 1.099). "
        "Check label smoothing and one-hot encoding."
    )
    # Aux policy is unmasked → CE is over the full move table (~4206 moves),
    # so the expected initial value is log(4206) ≈ 8.34, not log(num_legal).
    # The target distribution still concentrates on a small subset (the
    # opponent's legal replies), so a fresh-init network's loss is closer
    # to log(num_aux_target_support). Allow a wide window: [3.0, 9.5].
    assert 3.0 <= aux <= 9.5, (
        f"Initial aux-policy loss {aux:.3f} outside [3.0, 9.5]. "
        "Check aux-policy target normalization (sum to 1) and that the "
        "head outputs over the full move-index space."
    )
