"""Continuous trainer for async distributed training.

Two training regimes:

1. **Bootstrap** (`_run_bootstrap`): Runs when no model exists yet. Waits for
   workers to generate enough imitation data, then trains for multiple epochs.

2. **Self-play loop** (`run_trainer` main loop): Runs continuously after
   bootstrap. Samples from a recency-weighted replay buffer, trains across fixed
   summary intervals, and checks promotion whenever fresh-data polls indicate
   the latest candidate is eligible.

All data exchange happens via S3 (DigitalOcean Spaces / R2 / etc).
"""

from __future__ import annotations

import random
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, IterableDataset

from . import storage
from .config import AsyncConfig
from .data import TrainingBatch, load_imitation_npz, load_selfplay_npz
from .logging_setup import log_event, setup_json_logging
from .export import export_to_onnx
from .health_checks import (
    HealthCheckError,
    run_all_invariants,
    run_runtime_checks,
)
from .losses import (
    LossBreakdown,
    LossWeights,
    assert_healthy_initial_losses,
    compute_losses,
)
from .model import build_model
from .replay_window import sublinear_window_size
from .slack import notify_training_summary
from .swa import SwaSnapshotBuffer, update_bn_stats


# ---------------------------------------------------------------------------
# Trainer tunables that aren't in AsyncConfig yet. notes/13 §4.3–§4.6.
# Chunk 13 will promote these into the config dataclass.
# ---------------------------------------------------------------------------
# Module-level constants are gone — read everything off the active config
# instance via cfg.foo. See training/config.py for the field definitions
# and validate() ranges.


def _batch_to_torch(
    batch_np: dict[str, np.ndarray], device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Collate a dict of numpy arrays into torch tensors."""
    boards = torch.from_numpy(batch_np["boards"]).to(device)
    targets = {
        "policy": torch.from_numpy(batch_np["policy"]).to(device),
        "wdl": torch.from_numpy(batch_np["wdl_terminal"]).to(device),
        "mlh": torch.from_numpy(batch_np["mlh"]).to(device),
        "stv": torch.from_numpy(batch_np["wdl_short"]).to(device),
        "aux_policy": torch.from_numpy(batch_np["aux_policy"]).to(device),
    }
    legal_mask = torch.from_numpy(batch_np["legal_mask"]).to(device)
    return boards, targets, legal_mask


def _read_model_version() -> int:
    """Read the current model version from S3."""
    try:
        return storage.get_json(storage.LATEST_META).get("version", 0)
    except KeyError:
        return 0


def _read_positions_at_last_promote() -> int:
    """Read the positions watermark from S3.

    Returns 0 if no metadata exists (bootstrap case). If metadata exists, the
    promotion watermark must be present explicitly.
    """
    try:
        meta = storage.get_json(storage.LATEST_META)
    except KeyError:
        # No metadata at all — bootstrap case
        return 0
    return meta["positions_at_promote"]


# ---------------------------------------------------------------------------
# Train bucket (rate limiter)
# ---------------------------------------------------------------------------

class TrainBucket:
    """Token-bucket rate limiter that throttles training to match data inflow.

    KataGo-style: each new data position adds ``target_passes`` tokens.
    Each training step consumes ``batch_size`` tokens (since one step
    processes a full batch of samples).  This ensures each data point is
    seen ~target_passes times on average over its buffer lifetime.

    On first call to :meth:`update`, the bucket is seeded with enough
    tokens for one summary interval so the trainer can start immediately.
    """

    def __init__(self, target_passes: float, batch_size: int,
                 max_seed: float | None = None,
                 max_tokens: float | None = None):
        if target_passes <= 0:
            raise ValueError(f"target_passes must be positive, got {target_passes}")
        self.target_passes = target_passes
        self.batch_size = batch_size
        self._max_seed = max_seed
        self._max_tokens = max_tokens
        self._tokens: float = 0.0
        self._prev_positions: int | None = None
        self._cumulative_positions: int = 0
        self._last_new: int = 0
        self._last_added: float = 0.0

    def update(self, total_positions: int, window_size: int | None = None) -> None:
        """Advance the bucket with the current total position count.

        ``window_size`` (optional) is the current replay-window size. It
        caps the initial seed: ``max_seed`` can never exceed
        ``window_size * target_passes`` because you cannot legitimately
        claim reuse credit for samples that are no longer in the window.
        (notes/13 §4.2 — fix for the "max_seed should not exceed window
        size" bug.)
        """
        if self._prev_positions is None:
            tokens = total_positions * self.target_passes
            if self._max_seed is not None:
                tokens = min(tokens, self._max_seed)
            if window_size is not None:
                window_cap = float(window_size) * self.target_passes
                tokens = min(tokens, window_cap)
            self._tokens = tokens
            self._last_new = total_positions
            self._last_added = tokens
        else:
            new = max(0, total_positions - self._prev_positions)
            added = new * self.target_passes
            self._tokens += added
            self._last_new = new
            self._last_added = added
        if self._max_tokens is not None:
            self._tokens = min(self._tokens, self._max_tokens)
        self._prev_positions = total_positions
        self._cumulative_positions = total_positions

    def consume(self) -> None:
        self._tokens -= self.batch_size

    @property
    def tokens(self) -> float:
        return self._tokens

    def has_budget(self) -> bool:
        # Must have enough tokens to cover a full batch; otherwise
        # `consume()` drives `_tokens` negative on a step we should have
        # waited out.
        return self._tokens >= self.batch_size


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

def _yield_samples_from_batch(
    batch: TrainingBatch, sample_per_file: int,
):
    n = len(batch)
    k = min(n, sample_per_file)
    idx = np.random.choice(n, size=k, replace=False)
    for i in idx:
        yield {
            "boards": batch.boards[i],
            "policy": batch.policy[i],
            "aux_policy": batch.aux_policy[i],
            "wdl_terminal": batch.wdl_terminal[i],
            "wdl_short": batch.wdl_short[i],
            "mlh": batch.mlh[i],
            "legal_mask": batch.legal_mask[i],
        }


def _download_selected_files(
    cache_dir: Path, selected: list[dict],
) -> tuple[list[Path], list[int], int]:
    if not selected:
        return [], [], 0
    cache_dir.mkdir(parents=True, exist_ok=True)
    needed: set[str] = set()
    local_files: list[Path] = []
    weights: list[int] = []
    total = 0
    for entry in selected:
        if entry["positions"] <= 0:
            continue
        safe_name = entry["key"].replace("/", "_")
        local_path = cache_dir / safe_name
        needed.add(safe_name)
        if not local_path.exists():
            storage.get_file(entry["key"], local_path)
        local_files.append(local_path)
        weights.append(entry["positions"])
        total += entry["positions"]
    for f in cache_dir.iterdir():
        if f.name not in needed and f.suffix == ".npz":
            f.unlink()
    return local_files, weights, total


class ImitationBuffer(IterableDataset):
    """Streams imitation data from S3 using the current sample schema.

    Selects the most recent files up to max_positions (by parsing
    timestamps from S3 keys), downloads them to a local cache,
    and samples from them with a shuffle buffer.
    """

    SHUFFLE_BUFFER_SIZE = 20_000
    SAMPLE_PER_FILE = 2048

    def __init__(self, cache_dir: Path, max_positions: int):
        self.cache_dir = cache_dir
        self.max_positions = max_positions
        selected = storage.select_recent_files(
            storage.IMITATION_PREFIX, self.max_positions
        )
        self.files, self.file_weights, self.total_positions = _download_selected_files(
            self.cache_dir, selected
        )

    def stats(self) -> dict:
        if not self.files:
            return {}
        return {
            "files": len(self.files),
            "positions": self.total_positions,
        }

    def __iter__(self):
        if not self.files:
            return

        shuffle_buf: list[dict] = []

        while True:
            [chosen] = random.choices(self.files, weights=self.file_weights, k=1)
            try:
                batch = load_imitation_npz(chosen)
            except (OSError, ValueError):
                continue
            for sample in _yield_samples_from_batch(batch, self.SAMPLE_PER_FILE):
                shuffle_buf.append(sample)
            if len(shuffle_buf) >= self.SHUFFLE_BUFFER_SIZE:
                random.shuffle(shuffle_buf)
                drain = len(shuffle_buf) // 2
                for sample in shuffle_buf[:drain]:
                    yield sample
                shuffle_buf = shuffle_buf[drain:]


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer(IterableDataset):
    """Streams self-play samples from S3 with a sublinear KataGo window.

    Yields a dict per sample (converted to torch tensors at collate time)
    including the per-sample legal mask. Optionally mixes imitation files
    during early versions to stabilize training.
    """

    SHUFFLE_BUFFER_SIZE = 20_000
    SAMPLE_PER_FILE = 2048

    def __init__(
        self,
        cache_dir: Path,
        window_size: int,
        s3_prefix: str = storage.SELFPLAY_PREFIX,
        *,
        imitation_mix: float = 0.0,
    ):
        self.cache_dir = cache_dir
        self.window_size = window_size
        self.s3_prefix = s3_prefix
        self.imitation_mix = imitation_mix
        selected = storage.select_recent_files(self.s3_prefix, self.window_size)
        self.files, self.file_weights, self.total_positions = _download_selected_files(
            self.cache_dir, selected
        )
        # Also load imitation files if mixing is enabled.
        self.imitation_files: list[Path] = []
        self.imitation_weights: list[int] = []
        if imitation_mix > 0:
            self.imitation_files, self.imitation_weights = self._download_imitation()

    def _download_imitation(self) -> tuple[list[Path], list[int]]:
        """Download imitation files to a separate cache."""
        imit_cache = self.cache_dir.parent / "imitation"
        selected = storage.select_recent_files(
            storage.IMITATION_PREFIX, 500_000  # all imitation data
        )
        local_files, weights, _total = _download_selected_files(imit_cache, selected)
        return local_files, weights

    def stats(self) -> dict:
        return {"files": len(self.files), "positions": self.total_positions,
                "window_size": self.window_size,
                "imitation_files": len(self.imitation_files)}

    def __iter__(self):
        if not self.files:
            return

        shuffle_buf: list[dict] = []
        while True:
            # Mix imitation data: with probability imitation_mix, load from
            # imitation files instead of self-play. This anchors the policy
            # to the minimax teacher signal during early training when
            # self-play data is noisy.
            use_imitation = (
                self.imitation_files
                and random.random() < self.imitation_mix
            )
            # NB: we deliberately do NOT catch KeyError here. The loaders
            # raise KeyError when a .npz is missing the legal_mask field,
            # which means stale pre-schema-change data is still in the
            # cache — we want the trainer to fail loudly instead of
            # silently retrying forever.
            if use_imitation:
                [chosen] = random.choices(
                    self.imitation_files, weights=self.imitation_weights, k=1
                )
                try:
                    b = load_imitation_npz(chosen)
                except (OSError, ValueError):
                    continue
            else:
                # Position-weighted sampling: files with more positions are
                # selected proportionally more often, so each position has
                # equal probability of being sampled (position-uniform).
                [chosen] = random.choices(self.files, weights=self.file_weights, k=1)
                try:
                    b = load_selfplay_npz(chosen)
                except (OSError, ValueError):
                    continue
            # NOTE: no mirror augmentation. The encoder now produces STM-frame
            # tensors (see engine::serialization::encode_board), which bakes
            # the color symmetry into the representation by construction —
            # mirror aug would be a no-op and the old absolute-frame
            # implementation was silently corrupting pawn targets.
            for sample in _yield_samples_from_batch(b, self.SAMPLE_PER_FILE):
                shuffle_buf.append(sample)
            if len(shuffle_buf) >= self.SHUFFLE_BUFFER_SIZE:
                random.shuffle(shuffle_buf)
                drain = len(shuffle_buf) // 2
                for s in shuffle_buf[:drain]:
                    yield s
                shuffle_buf = shuffle_buf[drain:]


def _collate_samples(samples: list[dict]) -> dict[str, np.ndarray]:
    """Stack a list of per-sample dicts into a batch dict of numpy arrays."""
    out: dict[str, np.ndarray] = {}
    for k in samples[0].keys():
        out[k] = np.stack([s[k] for s in samples])
    return out


# ---------------------------------------------------------------------------
# Bootstrap training
# ---------------------------------------------------------------------------

def _run_bootstrap(cfg: AsyncConfig, model: torch.nn.Module,
                   optimizer: optim.Optimizer, device: torch.device) -> int:
    """Train on imitation data. Returns new model version."""
    logger.info("Bootstrap mode: waiting for {:,} imitation positions...",
                cfg.min_positions_to_start)

    while True:
        available = storage.count_positions(storage.IMITATION_PREFIX)
        if available >= cfg.min_positions_to_start:
            break
        logger.info("Waiting for data: {:,}/{:,} positions",
                    available, cfg.min_positions_to_start)
        time.sleep(30)

    total_steps_target = cfg.bootstrap_steps

    logger.info("")
    logger.info("=" * 60)
    logger.info("Bootstrap training: {:,} steps over {:,} positions (lr={:.4f})",
                total_steps_target, available, cfg.bootstrap_learning_rate)
    logger.info("=" * 60)

    def reload_buffer():
        ds = ImitationBuffer(
            cfg.data_cache_dir / "imitation",
            max_positions=cfg.min_positions_to_start,
        )
        dl = DataLoader(
            ds, batch_size=cfg.batch_size, num_workers=0,
            collate_fn=_collate_samples,
        )
        return ds, dl

    dataset, dataloader = reload_buffer()

    for pg in optimizer.param_groups:
        pg['lr'] = cfg.bootstrap_learning_rate

    model.train()
    total_steps = 0
    cumulative_policy_loss = 0.0
    cumulative_value_loss = 0.0
    cumulative_mlh_loss = 0.0
    cumulative_stv_loss = 0.0
    cumulative_aux_loss = 0.0
    loss_weights = LossWeights()
    t0 = time.time()

    while total_steps < total_steps_target:
        for batch_np in dataloader:
            if total_steps >= total_steps_target:
                break

            boards, targets, legal_mask = _batch_to_torch(batch_np, device)
            legal_mask = legal_mask.bool()

            preds = model(boards)
            breakdown: LossBreakdown = compute_losses(
                preds, targets, legal_mask=legal_mask,
                weights=loss_weights, debug=(total_steps == 0),
            )
            if total_steps == 0:
                try:
                    assert_healthy_initial_losses(breakdown, num_legal_moves=40)
                except AssertionError as exc:
                    logger.warning("Healthy-initial-loss check failed: {}", exc)

            optimizer.zero_grad()
            breakdown.total.backward()
            optimizer.step()

            cumulative_policy_loss += breakdown.policy.item()
            cumulative_value_loss += breakdown.value.item()
            cumulative_mlh_loss += breakdown.mlh.item()
            cumulative_stv_loss += breakdown.stv.item()
            cumulative_aux_loss += breakdown.aux_policy.item()
            total_steps += 1

            if total_steps % 100 == 0:
                elapsed = time.time() - t0
                avg_p = cumulative_policy_loss / total_steps
                avg_v = cumulative_value_loss / total_steps
                avg_mlh = cumulative_mlh_loss / total_steps
                avg_stv = cumulative_stv_loss / total_steps
                avg_aux = cumulative_aux_loss / total_steps
                logger.info(
                    "  step {:>5}/{:,} | policy={:.4f} value={:.4f} "
                    "mlh={:.4f} stv={:.4f} aux={:.4f} | {:.1f} steps/s | {:.0f}s",
                    total_steps, total_steps_target,
                    avg_p, avg_v, avg_mlh, avg_stv, avg_aux,
                    total_steps / elapsed, elapsed,
                )

            if total_steps % cfg.reload_interval == 0 and total_steps < total_steps_target:
                dataset, dataloader = reload_buffer()
                logger.info("  Reloaded buffer: {} files, {:,} positions",
                            len(dataset.files), dataset.total_positions)
                break

    train_elapsed = time.time() - t0
    logger.info("Bootstrap training complete: {:,} steps in {:.0f}s", total_steps, train_elapsed)

    # Export and promote v1. Self-play hasn't started yet, so positions = 0.
    new_version = 1
    bootstrap_selfplay_positions = storage.count_positions(storage.SELFPLAY_PREFIX)
    _promote_model(
        cfg,
        model,
        new_version,
        positions_at_promote=bootstrap_selfplay_positions,
        publish_approved=True,
    )

    logger.info("Promoted bootstrap model to v{} ({:,} steps, {:.0f}s)",
                new_version, total_steps, train_elapsed)
    return new_version


# ---------------------------------------------------------------------------
# Model promotion
# ---------------------------------------------------------------------------

def _promote_model(
    cfg: AsyncConfig,
    model: torch.nn.Module,
    version: int,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
    positions_at_promote: int | None = None,
    publish_approved: bool = False,
) -> None:
    """Export ``model`` (or an override ``state_dict``) and publish as latest.

    If ``state_dict`` is given (e.g. SWA-averaged weights), it is written
    to disk and exported to ONNX instead of ``model.state_dict()``. The
    live trainer weights are not mutated.

    ``positions_at_promote`` is persisted in the meta.json so the trainer
    can resume the promotion cadence watermark after a restart.
    """
    cfg.ensure_cache_dirs()

    local_pt = cfg.model_cache_dir / "checkpoint.pt"
    local_onnx = cfg.model_cache_dir / "latest.onnx"

    sd = state_dict if state_dict is not None else model.state_dict()
    torch.save(sd, local_pt)
    export_to_onnx(local_pt, local_onnx, cfg)

    storage.put_file(f"{storage.VERSIONS_PREFIX}{version}.onnx", local_onnx)
    storage.put_file(storage.CHECKPOINT_PT, local_pt)
    storage.copy(f"{storage.VERSIONS_PREFIX}{version}.onnx", storage.LATEST_ONNX)
    # Meta is the commit point — workers poll this, so write it last
    meta: dict = {
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if positions_at_promote is not None:
        meta["positions_at_promote"] = positions_at_promote
    storage.put_json(storage.LATEST_META, meta)
    if publish_approved:
        storage.copy(f"{storage.VERSIONS_PREFIX}{version}.onnx", storage.APPROVED_ONNX)
        storage.put_json(
            storage.APPROVED_META,
            {
                "version": version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


# ---------------------------------------------------------------------------
# Optimizer + LR + autocast helpers (chunk 6)
# ---------------------------------------------------------------------------

def _make_optimizer(model: torch.nn.Module, cfg: AsyncConfig) -> optim.Optimizer:
    """SGD momentum 0.9, wd from cfg (notes/13 §4.3)."""
    return optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.l2_regularization,
    )


def _lr_for_step(step: int, *, base_lr: float, warmup: int) -> float:
    """Linear warmup 0 -> base_lr over the first ``warmup`` fresh-run steps."""
    if warmup <= 0 or step >= warmup:
        return base_lr
    return base_lr * (step + 1) / warmup


def _set_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _autocast_dtype(device: torch.device) -> torch.dtype | None:
    """bf16 on CUDA, else None (disable autocast)."""
    if device.type == "cuda":
        return torch.bfloat16
    return None


def _promotion_check_ready(
    *,
    new_positions: int,
    threshold: int,
    total_steps: int,
    last_attempt_step: int,
) -> bool:
    """Return whether the trainer should evaluate a promotion candidate now.

    Promotion cadence is driven by fresh self-play positions, not by the end
    of a trainer summary interval. ``last_attempt_step`` suppresses repeated gate/export
    attempts when nothing has changed since the previous check.
    """
    return new_positions >= threshold and total_steps > last_attempt_step


def _maybe_promote(
    cfg: AsyncConfig,
    model: torch.nn.Module,
    *,
    current_version: int,
    positions_at_last_promote: int,
    n_total: int,
    swa_buf: SwaSnapshotBuffer,
    bn_refresh_batches: deque[torch.Tensor],
    device: torch.device,
    total_steps_all_time: int,
    last_promotion_attempt_step: int,
    log_skip: bool = False,
) -> tuple[int, int, int, bool]:
    """Export/promote if the current trainer state is eligible.

    Returns ``(current_version, positions_at_last_promote,
    last_promotion_attempt_step, promoted)`` with updated values.
    """
    new_positions = n_total - positions_at_last_promote
    if new_positions < cfg.promote_every_new_positions:
        if log_skip:
            logger.info(
                "Promotion not yet eligible — only {:,} new positions since v{} "
                "(need {:,})",
                new_positions, current_version, cfg.promote_every_new_positions,
            )
        return (
            current_version,
            positions_at_last_promote,
            last_promotion_attempt_step,
            False,
        )

    if not _promotion_check_ready(
        new_positions=new_positions,
        threshold=cfg.promote_every_new_positions,
        total_steps=total_steps_all_time,
        last_attempt_step=last_promotion_attempt_step,
    ):
        return (
            current_version,
            positions_at_last_promote,
            last_promotion_attempt_step,
            False,
        )

    last_promotion_attempt_step = total_steps_all_time

    # Build SWA candidate weights (fall back to raw if buffer empty).
    averaged_sd = swa_buf.average()
    if averaged_sd is None:
        logger.info("SWA buffer empty, using raw trainer weights for promotion")
        averaged_sd = {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()}
    else:
        # Re-estimate BN stats on averaged weights using a scratch model copy.
        scratch = build_model(cfg).to(device)
        scratch.load_state_dict({k: v.to(device) for k, v in averaged_sd.items()})
        if bn_refresh_batches:
            update_bn_stats(
                scratch,
                ((b,) for b in bn_refresh_batches),
                device=device,
            )
            averaged_sd = {k: v.detach().cpu().clone()
                           for k, v in scratch.state_dict().items()}
            logger.info(
                "  Re-estimated BN stats on {} batches", len(bn_refresh_batches)
            )
        del scratch

    new_version = current_version + 1
    _promote_model(
        cfg, model, new_version,
        state_dict=averaged_sd,
        positions_at_promote=n_total,
    )
    logger.info(
        "Promoted to v{} | total steps: {:,}",
        new_version, total_steps_all_time,
    )
    return new_version, n_total, last_promotion_attempt_step, True


# ---------------------------------------------------------------------------
# Trainer metrics publishing (plan §7.6 page 3)
# ---------------------------------------------------------------------------

def _publish_trainer_metrics(
    *,
    summary: int,
    version: int,
    steps: int,
    total_steps: int,
    n_total: int,
    avg_policy: float,
    avg_value: float,
    avg_mlh: float,
    avg_stv: float,
    avg_aux: float,
) -> None:
    """Append a trainer-summary record to ``state/trainer_metrics.json``.

    This is a read-modify-write operation. It is safe because only one trainer
    runs at a time — concurrent writes are not possible in normal operation.
    A crash between GET and PUT silently drops the in-flight summary from the
    history, which is acceptable (the loss value is recoverable from logs).
    """
    try:
        existing = storage.get_json(storage.TRAINER_METRICS)
        history: list[dict] = [
            {
                **{k: v for k, v in entry.items() if k != "cycle"},
                "summary": entry.get("summary", entry.get("cycle")),
            }
            for entry in existing.get("summaries", existing.get("cycles", []))
        ]
    except KeyError:
        history = []
    history.append({
        "summary": summary,
        "version": version,
        "steps": steps,
        "total_steps": total_steps,
        "n_total": n_total,
        "ts": datetime.now(timezone.utc).isoformat(),
        "loss_policy": round(avg_policy, 6),
        "loss_value": round(avg_value, 6),
        "loss_mlh": round(avg_mlh, 6),
        "loss_stv": round(avg_stv, 6),
        "loss_aux": round(avg_aux, 6),
    })
    if len(history) > 200:
        history = history[-200:]
    storage.put_json(storage.TRAINER_METRICS, {"summaries": history})


# ---------------------------------------------------------------------------
# Main trainer loop
# ---------------------------------------------------------------------------

def run_trainer(cfg: AsyncConfig) -> None:
    """Run the continuous trainer loop."""
    cfg.ensure_cache_dirs()
    cfg.validate()
    setup_json_logging("trainer", run_id=cfg.run_id)
    log_event("trainer.start", run_id=cfg.run_id,
              summary_interval_steps=cfg.summary_interval_steps,
              batch_size=cfg.batch_size)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    current_n_total = storage.count_positions(storage.SELFPLAY_PREFIX)
    initial_window = sublinear_window_size(
        current_n_total, c=cfg.window_c, alpha=cfg.window_alpha, beta=cfg.window_beta,
    )
    logger.info("Trainer starting on device: {} | N_total={:,} window={:,} "
                "steps/summary={}",
                device, current_n_total, initial_window, cfg.summary_interval_steps)

    current_version = _read_model_version()

    model = build_model(cfg).to(device)
    if current_version > 0:
        local_pt = cfg.model_cache_dir / "checkpoint.pt"
        local_onnx = cfg.model_cache_dir / "latest.onnx"
        storage.get_file(storage.CHECKPOINT_PT, local_pt)
        storage.get_file(storage.LATEST_ONNX, local_onnx)
        logger.info("Loading checkpoint v{}", current_version)
        model.load_state_dict(torch.load(local_pt, map_location=device, weights_only=True))

    optimizer = _make_optimizer(model, cfg)

    # Hard invariants (plan §7.5). Crash the trainer if any of these fail —
    # this is intentional CrashLoopBackoff behavior so encoding/shape bugs
    # don't silently corrupt a long training run.
    model.eval()
    try:
        report = run_all_invariants(model, batch=None, strict=True)
        logger.info("Startup health checks passed ({} invariants)",
                    len(report.results))
    except HealthCheckError as exc:
        logger.error("STARTUP HEALTH CHECK FAILED: {}", exc)
        raise
    finally:
        model.train()

    # Bootstrap if no model exists
    if current_version == 0:
        current_version = _run_bootstrap(cfg, model, optimizer, device)
        optimizer = _make_optimizer(model, cfg)

    summary = 0
    total_steps_all_time = 0
    fresh_run_steps = 0  # for LR warmup — resets on promotion is NOT needed, it's "fresh run"
    bucket = TrainBucket(cfg.max_train_steps_per_new_data,
                         batch_size=cfg.batch_size,
                         max_seed=cfg.summary_interval_steps * cfg.batch_size,
                         max_tokens=float(cfg.summary_interval_steps * cfg.batch_size))

    # SWA snapshot buffer + sample counter.
    # Build EMA-derived promotion weights from config: w_i = decay^i, then normalize.
    _raw_w = tuple(cfg.swa_ema_decay ** i for i in range(cfg.swa_buffer_size))
    _w_sum = sum(_raw_w)
    _swa_weights = tuple(w / _w_sum for w in _raw_w)
    swa_buf = SwaSnapshotBuffer(
        max_snapshots=cfg.swa_buffer_size,
        promotion_weights=_swa_weights,
    )
    samples_since_last_snapshot = 0
    # Promotion is gated on *new* positions since the last promotion.
    # Read the watermark from S3 so promotion cadence survives restarts.
    # Falls back to 0 for bootstrap (no meta.json yet).
    positions_at_last_promote = _read_positions_at_last_promote()
    logger.info("Loaded positions watermark: {:,}", positions_at_last_promote)
    last_promotion_attempt_step = -1
    # Rolling buffer of recent training batches for re-estimating BN
    # running stats after SWA averaging. update_bn_stats uses cumulative
    # averaging (momentum=None), so forwarding a single batch leaves the
    # averaged model with running mean/var equal to that batch's stats —
    # a noisy, stale estimate that silently degrades every promoted model.
    # We keep the buffer on CPU to avoid pinning GPU memory across summaries.
    bn_refresh_batches: deque[torch.Tensor] = deque(
        maxlen=cfg.swa_bn_refresh_batches
    )

    ac_dtype = _autocast_dtype(device)

    def reload_buffer() -> tuple[ReplayBuffer, DataLoader, int]:
        n_total = storage.count_positions(storage.SELFPLAY_PREFIX)
        window = sublinear_window_size(
            n_total, c=cfg.window_c, alpha=cfg.window_alpha, beta=cfg.window_beta,
        )
        # Imitation mix is decayed against the *current* model version
        # (captured from the enclosing scope, which advances on promotion)
        # so the teacher signal fades as self-play matures — see
        # AsyncConfig.imitation_mix_for_version.
        mix = cfg.imitation_mix_for_version(current_version)
        ds = ReplayBuffer(
            cfg.data_cache_dir / "selfplay",
            window_size=window,
            s3_prefix=storage.SELFPLAY_PREFIX,
            imitation_mix=mix,
        )
        dl = DataLoader(
            ds, batch_size=cfg.batch_size, num_workers=0,
            collate_fn=_collate_samples,
        )
        return ds, dl, n_total

    while True:
        dataset, dataloader, n_total = reload_buffer()
        if dataset.total_positions == 0:
            logger.info("No valid training data found, waiting...")
            time.sleep(30)
            continue

        summary += 1
        summary_t0 = time.time()

        logger.info("")
        logger.info("=" * 60)
        logger.info("Trainer summary {} | model v{} | N_total={:,} window={:,}",
                    summary, current_version, n_total, dataset.window_size)
        logger.info("=" * 60)

        bucket.update(n_total, window_size=dataset.window_size)

        if not bucket.has_budget():
            logger.info("Train bucket empty ({:.0f} tokens), waiting for new data...",
                        bucket.tokens)
            while not bucket.has_budget():
                time.sleep(30)
                n_total = storage.count_positions(storage.SELFPLAY_PREFIX)
                bucket.update(n_total, window_size=dataset.window_size)
                if not bucket.has_budget():
                    logger.info("  Still waiting... bucket={:.0f} tokens, {:,} cumulative",
                                bucket.tokens, bucket._cumulative_positions)

        logger.info("Replay buffer: {} files, {:,} positions (window={:,}) "
                    "| imitation_mix={:.2f} ({} imitation files)",
                    len(dataset.files), dataset.total_positions, dataset.window_size,
                    dataset.imitation_mix, len(dataset.imitation_files))
        logger.info("Train bucket: {:.0f} tokens available ({:,} new, +{:.0f} tokens)",
                    bucket.tokens, bucket._last_new, bucket._last_added)

        # Training
        model.train()
        step = 0
        summary_policy_loss = 0.0
        summary_value_loss = 0.0
        summary_mlh_loss = 0.0
        summary_stv_loss = 0.0
        summary_aux_loss = 0.0
        summary_total_loss = 0.0
        loss_weights = LossWeights()
        train_t0 = time.time()

        logger.info("Training for {} steps (batch_size={})...",
                    cfg.summary_interval_steps, cfg.batch_size)

        while step < cfg.summary_interval_steps:
            if not bucket.has_budget():
                logger.info("  Bucket empty mid-summary at step {}, waiting...", step)
                while not bucket.has_budget():
                    time.sleep(30)
                    n_total = storage.count_positions(storage.SELFPLAY_PREFIX)
                    bucket.update(n_total, window_size=dataset.window_size)
                    (
                        current_version,
                        positions_at_last_promote,
                        last_promotion_attempt_step,
                        promoted,
                    ) = _maybe_promote(
                        cfg,
                        model,
                        current_version=current_version,
                        positions_at_last_promote=positions_at_last_promote,
                        n_total=n_total,
                        swa_buf=swa_buf,
                        bn_refresh_batches=bn_refresh_batches,
                        device=device,
                        total_steps_all_time=total_steps_all_time,
                        last_promotion_attempt_step=last_promotion_attempt_step,
                    )
                    if promoted:
                        dataset, dataloader, n_total = reload_buffer()
                        bucket.update(n_total, window_size=dataset.window_size)

            for batch_np in dataloader:
                if step >= cfg.summary_interval_steps or not bucket.has_budget():
                    break

                # Snapshot the CPU numpy boards into the BN refresh buffer
                # *before* the device transfer — sourcing from the GPU copy
                # would add a D2H sync every training step. .clone()
                # decouples from numpy memory that the dataloader may reuse.
                bn_refresh_batches.append(
                    torch.from_numpy(batch_np["boards"]).clone()
                )

                boards, targets, legal_mask = _batch_to_torch(batch_np, device)

                # LR warmup.
                lr = _lr_for_step(
                    fresh_run_steps,
                    base_lr=cfg.learning_rate,
                    warmup=cfg.lr_warmup_steps,
                )
                _set_lr(optimizer, lr)

                optimizer.zero_grad()
                if ac_dtype is not None:
                    with torch.amp.autocast(device_type=device.type, dtype=ac_dtype):
                        preds = model(boards)
                        breakdown: LossBreakdown = compute_losses(
                            preds, targets, legal_mask=legal_mask,
                            weights=loss_weights,
                            debug=(total_steps_all_time == 0 and step == 0),
                        )
                else:
                    preds = model(boards)
                    breakdown = compute_losses(
                        preds, targets, legal_mask=legal_mask,
                        weights=loss_weights,
                        debug=(total_steps_all_time == 0 and step == 0),
                    )

                if total_steps_all_time == 0 and step == 0:
                    try:
                        assert_healthy_initial_losses(breakdown, num_legal_moves=40)
                    except AssertionError as exc:
                        logger.warning("Healthy-initial-loss check failed: {}", exc)

                breakdown.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

                summary_policy_loss += breakdown.policy.item()
                summary_value_loss += breakdown.value.item()
                summary_mlh_loss += breakdown.mlh.item()
                summary_stv_loss += breakdown.stv.item()
                summary_aux_loss += breakdown.aux_policy.item()
                summary_total_loss += breakdown.total.item()
                step += 1
                total_steps_all_time += 1
                fresh_run_steps += 1
                bucket.consume()
                samples_since_last_snapshot += cfg.batch_size

                # SWA snapshot.
                if samples_since_last_snapshot >= cfg.swa_snapshot_every_samples:
                    swa_buf.append(model.state_dict())
                    samples_since_last_snapshot = 0
                    logger.info("  SWA snapshot taken ({} in buffer)", len(swa_buf))

                # Tier-1 per-step structured event (plan §7.2).
                log_event(
                    "train.step",
                    step_id=total_steps_all_time,
                    summary=summary,
                    version=current_version,
                    wall_ms=int((time.time() - train_t0) * 1000),
                    loss_policy=float(breakdown.policy.item()),
                    loss_value=float(breakdown.value.item()),
                    loss_mlh=float(breakdown.mlh.item()),
                    loss_stv=float(breakdown.stv.item()),
                    loss_aux_policy=float(breakdown.aux_policy.item()),
                    loss_total=float(breakdown.total.item()),
                    optim_lr=float(lr),
                    data_window_size=int(dataset.window_size),
                    data_cumulative_positions=int(n_total),
                )

                if step % 100 == 0 or step == cfg.summary_interval_steps:
                    elapsed = time.time() - train_t0
                    sps = step / elapsed if elapsed > 0 else 0
                    logger.info(
                        "  step {:>5}/{} lr={:.5f} | policy={:.4f} value={:.4f} "
                        "mlh={:.4f} stv={:.4f} aux={:.4f} total={:.4f} "
                        "| {:.1f} steps/s | {:.0f}s",
                        step, cfg.summary_interval_steps, lr,
                        summary_policy_loss / step, summary_value_loss / step,
                        summary_mlh_loss / step, summary_stv_loss / step,
                        summary_aux_loss / step, summary_total_loss / step,
                        sps, elapsed,
                    )

                if step % cfg.runtime_health_check_every_steps == 0:
                    rt_report = run_runtime_checks(
                        model,
                        breakdown.total.detach(),
                        {"boards": boards, "legal_mask": legal_mask},
                    )
                    for f in rt_report.failures():
                        logger.warning("runtime health check: {}: {}",
                                       f.name, f.message)

                if step % cfg.reload_interval == 0 and step < cfg.summary_interval_steps:
                    dataset, dataloader, n_total = reload_buffer()
                    bucket.update(n_total, window_size=dataset.window_size)
                    logger.info(
                        "  Reloaded: {} files, {:,} pos (window={:,}) | bucket: {:.0f}",
                        len(dataset.files), dataset.total_positions,
                        dataset.window_size, bucket.tokens,
                    )
                    (
                        current_version,
                        positions_at_last_promote,
                        last_promotion_attempt_step,
                        promoted,
                    ) = _maybe_promote(
                        cfg,
                        model,
                        current_version=current_version,
                        positions_at_last_promote=positions_at_last_promote,
                        n_total=n_total,
                        swa_buf=swa_buf,
                        bn_refresh_batches=bn_refresh_batches,
                        device=device,
                        total_steps_all_time=total_steps_all_time,
                        last_promotion_attempt_step=last_promotion_attempt_step,
                    )
                    if promoted:
                        dataset, dataloader, n_total = reload_buffer()
                        bucket.update(n_total, window_size=dataset.window_size)
                        logger.info(
                            "  Reloaded after promotion: {} files, {:,} pos "
                            "(window={:,}) | bucket: {:.0f}",
                            len(dataset.files), dataset.total_positions,
                            dataset.window_size, bucket.tokens,
                        )
                    break

        latest_n_total = storage.count_positions(storage.SELFPLAY_PREFIX)
        if latest_n_total != n_total:
            n_total = latest_n_total
            bucket.update(n_total, window_size=dataset.window_size)

        train_elapsed = time.time() - train_t0
        avg_policy = summary_policy_loss / max(step, 1)
        avg_value = summary_value_loss / max(step, 1)
        avg_mlh = summary_mlh_loss / max(step, 1)
        avg_stv = summary_stv_loss / max(step, 1)
        avg_aux = summary_aux_loss / max(step, 1)
        logger.info(
            "Summary done: {} steps in {:.0f}s | policy={:.4f} value={:.4f} "
            "mlh={:.4f} stv={:.4f} aux={:.4f}",
            step, train_elapsed, avg_policy, avg_value,
            avg_mlh, avg_stv, avg_aux,
        )

        (
            current_version,
            positions_at_last_promote,
            last_promotion_attempt_step,
            _promoted,
        ) = _maybe_promote(
            cfg,
            model,
            current_version=current_version,
            positions_at_last_promote=positions_at_last_promote,
            n_total=n_total,
            swa_buf=swa_buf,
            bn_refresh_batches=bn_refresh_batches,
            device=device,
            total_steps_all_time=total_steps_all_time,
            last_promotion_attempt_step=last_promotion_attempt_step,
            log_skip=True,
        )

        summary_elapsed = time.time() - summary_t0
        notify_training_summary(
            summary=summary, version=current_version, steps=step,
            total_steps=total_steps_all_time, positions=dataset.total_positions,
            policy_loss=avg_policy, value_loss=avg_value,
            elapsed_seconds=summary_elapsed,
        )
        try:
            _publish_trainer_metrics(
                summary=summary,
                version=current_version,
                steps=step,
                total_steps=total_steps_all_time,
                n_total=n_total,
                avg_policy=avg_policy,
                avg_value=avg_value,
                avg_mlh=avg_mlh,
                avg_stv=avg_stv,
                avg_aux=avg_aux,
            )
        except Exception as _exc:  # noqa: BLE001
            logger.warning("Failed to publish trainer metrics to S3: {}", _exc)
