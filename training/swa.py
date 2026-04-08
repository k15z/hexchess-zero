"""Stochastic Weight Averaging (SWA) for model promotion.

notes/06 §SWA, notes/13 §4.4.

KataGo recipe:
  - Snapshot the trainer's weights every ~250k samples (~1000 batches at bs=256).
  - Keep a rolling buffer of the four most recent snapshots.
  - Promotion weights are a fixed [0.4, 0.3, 0.2, 0.1] linear mixture over
    the newest-to-oldest snapshots.
  - BN running stats need to be re-estimated on a held-out batch after
    averaging, because the per-snapshot running means/vars are stale for
    the averaged weights.

We keep the buffer as a list of state_dicts on CPU to avoid pinning
GPU memory for up to four copies of the model.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import torch
import torch.nn as nn

# Most-recent-first mixture weights. Length == max buffer size.
DEFAULT_PROMOTION_WEIGHTS: tuple[float, ...] = (0.4, 0.3, 0.2, 0.1)
DEFAULT_MAX_SNAPSHOTS = len(DEFAULT_PROMOTION_WEIGHTS)


class SwaSnapshotBuffer:
    """Rolling buffer of recent trainer state_dicts, newest-first."""

    def __init__(
        self,
        max_snapshots: int = DEFAULT_MAX_SNAPSHOTS,
        promotion_weights: tuple[float, ...] = DEFAULT_PROMOTION_WEIGHTS,
    ):
        if max_snapshots < 1:
            raise ValueError("max_snapshots must be >= 1")
        if len(promotion_weights) != max_snapshots:
            raise ValueError(
                f"promotion_weights length {len(promotion_weights)} "
                f"!= max_snapshots {max_snapshots}"
            )
        self.max_snapshots = max_snapshots
        self.promotion_weights = promotion_weights
        self._snaps: deque[dict[str, torch.Tensor]] = deque(maxlen=max_snapshots)

    def __len__(self) -> int:
        return len(self._snaps)

    def append(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Store a deep CPU copy of the given state_dict as the newest snapshot."""
        cpu_copy = {k: v.detach().to("cpu").clone() for k, v in state_dict.items()}
        self._snaps.appendleft(cpu_copy)

    def average(self) -> dict[str, torch.Tensor] | None:
        """Return the promotion-weighted average of the buffered snapshots.

        If the buffer holds fewer than ``max_snapshots`` snapshots, the
        weights are renormalized over the available prefix so a single
        snapshot still produces a valid average (= identity).

        Returns None if the buffer is empty.
        """
        if not self._snaps:
            return None
        k = len(self._snaps)
        raw = self.promotion_weights[:k]
        total = sum(raw)
        norm = [w / total for w in raw]
        out: dict[str, torch.Tensor] = {}
        # Use the first snapshot's structure as the template.
        template = self._snaps[0]
        for key, ref in template.items():
            if not torch.is_floating_point(ref):
                # Non-float tensors (e.g. BN num_batches_tracked int64)
                # should not be averaged — copy the newest snapshot.
                out[key] = ref.clone()
                continue
            acc = torch.zeros_like(ref, dtype=torch.float32)
            for w, snap in zip(norm, self._snaps):
                acc.add_(snap[key].to(torch.float32), alpha=w)
            out[key] = acc.to(ref.dtype)
        return out


@torch.no_grad()
def update_bn_stats(
    model: nn.Module,
    loader: Iterable,
    device: torch.device | str = "cpu",
    max_batches: int | None = None,
) -> None:
    """Re-estimate BN running stats for ``model`` by forwarding ``loader``.

    Standard torch SWA recipe: put the model in ``train()`` mode (so BN
    updates its running stats) but guard the call with ``no_grad`` and do
    not invoke the optimizer. One pass over a small held-out loader is
    enough to reset running means/vars on the averaged weights.

    Resets each BN layer's running stats and momentum to ``None`` (which
    switches to cumulative averaging) before the pass, then restores
    the original momentum.
    """
    momenta: dict[nn.Module, float | None] = {}
    has_bn = False
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                          nn.SyncBatchNorm)):
            has_bn = True
            momenta[m] = m.momentum
            m.running_mean = torch.zeros_like(m.running_mean)
            m.running_var = torch.ones_like(m.running_var)
            m.momentum = None  # cumulative moving average
            m.num_batches_tracked.zero_()
    if not has_bn:
        return

    was_training = model.training
    model.train()
    try:
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            elif isinstance(batch, dict):
                x = batch["boards"]
            else:
                x = batch
            x = x.to(device)
            model(x)
    finally:
        for m, mom in momenta.items():
            m.momentum = mom
        if not was_training:
            model.eval()
