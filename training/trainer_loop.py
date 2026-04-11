"""Continuous trainer for async distributed training.

Two training regimes:

1. **Bootstrap** (`_run_bootstrap`): Runs when no model exists yet. Waits for
   workers to generate enough imitation data, then trains for multiple epochs.

2. **Self-play loop** (`run_trainer` main loop): Runs continuously after
   bootstrap. Samples from a recency-weighted replay buffer, trains for N
   steps per cycle, exports and promotes unconditionally, repeat.

All data exchange happens via S3 (DigitalOcean Spaces / R2 / etc).
"""

from __future__ import annotations

from collections.abc import Iterator
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from loguru import logger
from numpy.lib.npyio import NpzFile
from torch.utils.data import DataLoader, IterableDataset

from . import storage
from .config import AsyncConfig
from .logging_setup import log_event, setup_json_logging
from .data_v2 import MIRROR_INDICES, V2Batch, load_imitation_npz, load_v2_npz, mirror_batch
from .export import export_to_onnx
from .health_checks import (
    HealthCheckError,
    run_all_invariants,
    run_runtime_checks,
)
from .gating import (
    GateState,
    decide_promotion,
    load_gate_state,
    save_gate_state,
)
from .losses import (
    LossBreakdown,
    LossWeights,
    assert_healthy_initial_losses,
    compute_losses,
)
from .model import build_model
from .replay_window import sublinear_window_size
from .slack import notify_training_cycle
from .swa import SwaSnapshotBuffer, update_bn_stats
from .types import V2BatchDict, V2Sample, parse_latest_model_meta


# ---------------------------------------------------------------------------
# Trainer tunables that aren't in AsyncConfig yet. notes/13 §4.3–§4.6.
# Chunk 13 will promote these into the config dataclass.
# ---------------------------------------------------------------------------
# Module-level constants are gone — read everything off the active config
# instance via cfg.foo. See training/config.py for the field definitions
# and validate() ranges.


def _build_imitation_targets(
    policies: torch.Tensor,
    outcomes: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    """Bootstrap targets dict from the old imitation .npz schema.

    Imitation data only has ``boards``, ``policies``, ``outcomes`` — we
    supply neutral placeholders for the MLH/STV/aux heads so they still
    produce gradients (weighted small per :class:`LossWeights`) without
    biasing the trunk. Legal mask is derived from the policy target.
    """
    batch = policies.shape[0]
    num_moves = policies.shape[1]
    device = policies.device
    mlh_placeholder = torch.zeros(batch, device=device)
    stv_placeholder = outcomes.detach().clone()  # terminal ≈ horizon-8 fallback
    aux_policy_placeholder = torch.full(
        (batch, num_moves), 1.0 / num_moves, device=device
    )
    legal_mask = policies > 0.0
    return {
        "policy": policies,
        "wdl": outcomes,
        "mlh": mlh_placeholder,
        "stv": stv_placeholder,
        "aux_policy": aux_policy_placeholder,
    }, legal_mask


def _v2_batch_to_torch(
    batch_np: V2BatchDict, device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Collate a dict of numpy arrays from the v2 loader into torch tensors."""
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
        return parse_latest_model_meta(storage.get_json(storage.LATEST_META))["version"]
    except KeyError:
        return 0


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
    tokens for one cycle so the trainer can start immediately.
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
        return self._tokens > 0


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer(IterableDataset):
    """Streams training data from S3 with uniform sampling.

    Selects the most recent files up to max_positions (by parsing
    timestamps from S3 keys), downloads them to a local cache,
    and samples from them with a shuffle buffer.
    """

    SHUFFLE_BUFFER_SIZE = 100_000
    SAMPLE_PER_FILE = 2048

    def __init__(self, cache_dir: Path, max_positions: int = 5_000_000,
                 s3_prefix: str = storage.SELFPLAY_PREFIX):
        self.cache_dir = cache_dir
        self.max_positions = max_positions
        self.s3_prefix = s3_prefix
        self.files, self.total_positions = self._select_and_download()

    def _select_and_download(self) -> tuple[list[Path], int]:
        """Select recent files from S3 and download to local cache."""
        selected = storage.select_recent_files(self.s3_prefix, self.max_positions)
        if not selected:
            return [], 0

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        needed = set()
        local_files = []
        total = 0
        for entry in selected:
            safe_name = entry["key"].replace("/", "_")
            local_path = self.cache_dir / safe_name
            needed.add(safe_name)
            if not local_path.exists():
                storage.get_file(entry["key"], local_path)
            local_files.append(local_path)
            total += entry["positions"]

        # Prune cached files no longer in the selection
        for f in self.cache_dir.iterdir():
            if f.name not in needed and f.suffix == ".npz":
                f.unlink()

        return local_files, total

    def stats(self) -> dict[str, int]:
        if not self.files:
            return {}
        return {
            "files": len(self.files),
            "positions": self.total_positions,
        }

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if not self.files:
            return

        buf_b, buf_p, buf_o = [], [], []

        while True:
            [chosen] = random.choices(self.files, k=1)
            try:
                data: NpzFile = np.load(chosen, mmap_mode="r")
                boards, policies, outcomes = data["boards"], data["policies"], data["outcomes"]
            except (OSError, ValueError, KeyError):
                continue

            n = len(outcomes)
            k = min(n, self.SAMPLE_PER_FILE)
            idx = np.random.choice(n, size=k, replace=False)
            idx.sort()
            buf_b.append(np.array(boards[idx]))
            buf_p.append(np.array(policies[idx]))
            buf_o.append(np.array(outcomes[idx]))

            total = sum(len(b) for b in buf_b)

            if total >= self.SHUFFLE_BUFFER_SIZE:
                merged_b = np.concatenate(buf_b)
                merged_p = np.concatenate(buf_p)
                merged_o = np.concatenate(buf_o)
                perm = np.random.permutation(len(merged_b))
                drain = len(perm) // 2
                for j in perm[:drain]:
                    yield (torch.from_numpy(merged_b[j].copy()),
                           torch.from_numpy(merged_p[j].copy()),
                           torch.tensor(merged_o[j], dtype=torch.float32))
                keep = perm[drain:]
                buf_b = [merged_b[keep]]
                buf_p = [merged_p[keep]]
                buf_o = [merged_o[keep]]


# ---------------------------------------------------------------------------
# v2 self-play replay buffer (chunk 6)
# ---------------------------------------------------------------------------

class ReplayBufferV2(IterableDataset):
    """Streams v2 self-play samples from S3 with a sublinear KataGo window.

    Differences from the legacy ``ReplayBuffer``:
      - loads the v2 schema via :func:`training.data_v2.load_v2_npz`,
      - window size is computed dynamically from the KataGo sublinear
        formula rather than a fixed ``replay_buffer_size``,
      - yields a dict per sample (converted to torch tensors at collate
        time) including the per-sample legal mask.
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
        self.files, self.total_positions = self._select_and_download()
        # Also load imitation files if mixing is enabled.
        self.imitation_files: list[Path] = []
        if imitation_mix > 0:
            self.imitation_files = self._download_imitation()

    def _select_and_download(self) -> tuple[list[Path], int]:
        selected = storage.select_recent_files(self.s3_prefix, self.window_size)
        if not selected:
            return [], 0
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        needed = set()
        local_files: list[Path] = []
        total = 0
        for entry in selected:
            safe_name = entry["key"].replace("/", "_")
            local_path = self.cache_dir / safe_name
            needed.add(safe_name)
            if not local_path.exists():
                storage.get_file(entry["key"], local_path)
            local_files.append(local_path)
            total += entry["positions"]
        # Prune stale cache entries
        for f in self.cache_dir.iterdir():
            if f.name not in needed and f.suffix == ".npz":
                f.unlink()
        return local_files, total

    def _download_imitation(self) -> list[Path]:
        """Download imitation files to a separate cache."""
        imit_cache = self.cache_dir.parent / "imitation"
        imit_cache.mkdir(parents=True, exist_ok=True)
        selected = storage.select_recent_files(
            storage.IMITATION_PREFIX, 500_000  # all imitation data
        )
        local_files: list[Path] = []
        for entry in selected:
            safe = entry["key"].replace("/", "_")
            lp = imit_cache / safe
            if not lp.exists():
                storage.get_file(entry["key"], lp)
            local_files.append(lp)
        return local_files

    def stats(self) -> dict[str, int]:
        return {"files": len(self.files), "positions": self.total_positions,
                "window_size": self.window_size,
                "imitation_files": len(self.imitation_files)}

    def __iter__(self) -> Iterator[V2Sample]:
        if not self.files:
            return

        def _yield_from_batch(b: V2Batch) -> Iterator[V2Sample]:
            n = len(b)
            k = min(n, self.SAMPLE_PER_FILE)
            idx = np.random.choice(n, size=k, replace=False)
            for i in idx:
                yield {
                    "boards": b.boards[i],
                    "policy": b.policy[i],
                    "aux_policy": b.aux_policy[i],
                    "wdl_terminal": b.wdl_terminal[i],
                    "wdl_short": b.wdl_short[i],
                    "mlh": b.mlh[i],
                    "legal_mask": b.legal_mask[i],
                }

        shuffle_buf: list[V2Sample] = []
        while True:
            # Mix imitation data: with probability imitation_mix, load from
            # imitation files instead of self-play. This anchors the policy
            # to the minimax teacher signal during early training when
            # self-play data is noisy.
            use_imitation = (
                self.imitation_files
                and random.random() < self.imitation_mix
            )
            if use_imitation:
                [chosen] = random.choices(self.imitation_files, k=1)
                try:
                    b = load_imitation_npz(chosen)
                except (OSError, ValueError, KeyError):
                    continue
            else:
                [chosen] = random.choices(self.files, k=1)
                try:
                    b = load_v2_npz(chosen)
                except (OSError, ValueError, KeyError):
                    continue
            # Mirror augmentation: 50% of files get horizontally mirrored
            # via the Rust mirror table. Cheap 2x effective data multiplier.
            if MIRROR_INDICES is not None and random.random() < 0.5:
                b = mirror_batch(b)
            for sample in _yield_from_batch(b):
                shuffle_buf.append(sample)
            if len(shuffle_buf) >= self.SHUFFLE_BUFFER_SIZE:
                random.shuffle(shuffle_buf)
                drain = len(shuffle_buf) // 2
                for s in shuffle_buf[:drain]:
                    yield s
                shuffle_buf = shuffle_buf[drain:]


def _v2_collate(samples: list[V2Sample]) -> V2BatchDict:
    """Stack a list of per-sample dicts into a batch dict of numpy arrays."""
    out: V2BatchDict = {
        "boards": np.stack([s["boards"] for s in samples]),
        "policy": np.stack([s["policy"] for s in samples]),
        "aux_policy": np.stack([s["aux_policy"] for s in samples]),
        "wdl_terminal": np.stack([s["wdl_terminal"] for s in samples]),
        "wdl_short": np.stack([s["wdl_short"] for s in samples]),
        "mlh": np.stack([s["mlh"] for s in samples]),
        "legal_mask": np.stack([s["legal_mask"] for s in samples]),
    }
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
        ds = ReplayBuffer(cfg.data_cache_dir / "imitation",
                          max_positions=cfg.min_positions_to_start,
                          s3_prefix=storage.IMITATION_PREFIX)
        dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=0)
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
        for boards, policies, outcomes in dataloader:
            if total_steps >= total_steps_target:
                break

            boards = boards.to(device)
            policies = policies.to(device)
            outcomes = outcomes.to(device)

            preds = model(boards)
            targets, legal_mask = _build_imitation_targets(policies, outcomes)
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

    # Export and promote v1
    new_version = 1
    _promote_model(cfg, model, new_version)

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
) -> None:
    """Export ``model`` (or an override ``state_dict``) and publish as latest.

    If ``state_dict`` is given (e.g. SWA-averaged weights), it is written
    to disk and exported to ONNX instead of ``model.state_dict()``. The
    live trainer weights are not mutated.
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
    storage.put_json(storage.LATEST_META, {
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


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


# ---------------------------------------------------------------------------
# Gauntlet runner for first-5 gating (chunk 6)
# ---------------------------------------------------------------------------

def _default_gauntlet(candidate_onnx: Path, current_onnx: Path,
                      *, simulations: int, n_games: int) -> float:
    """Play ``n_games`` between candidate and current ONNX models.

    Returns candidate's score fraction in [0, 1]. Lazy-imports from
    :mod:`training.elo` so unit tests can run without the hexchess
    binding installed.
    """
    from .elo import MctsPlayer, play_game

    cand = MctsPlayer(name="candidate", simulations=simulations,
                      model_path=str(candidate_onnx))
    cur = MctsPlayer(name="current", simulations=simulations,
                     model_path=str(current_onnx))
    wins = 0.0
    for i in range(n_games):
        if i % 2 == 0:
            result = play_game(cand, cur)
            outcome = result.get("outcome")
            if outcome == "white":
                wins += 1.0
            elif outcome == "draw":
                wins += 0.5
        else:
            result = play_game(cur, cand)
            outcome = result.get("outcome")
            if outcome == "black":
                wins += 1.0
            elif outcome == "draw":
                wins += 0.5
    return wins / n_games


def _try_gate_promotion(
    cfg: AsyncConfig,
    averaged_sd: dict[str, torch.Tensor],
    candidate_version: int,
    current_version: int,
    play_gauntlet=None,
) -> tuple[bool, GateState, float, str]:
    """Run the gate against a candidate SWA state_dict.

    Writes the candidate ONNX to a scratch path before invoking the
    gauntlet. Returns (promote, new_state, score, reason).
    """
    state = load_gate_state()
    if not state.gate_enabled:
        decision = decide_promotion(state, candidate=None, current=None)
        return decision.promote, decision.state, decision.score, decision.reason

    # Materialize candidate ONNX to a scratch path so the gauntlet can load it.
    scratch_pt = cfg.model_cache_dir / "candidate.pt"
    scratch_onnx = cfg.model_cache_dir / "candidate.onnx"
    torch.save(averaged_sd, scratch_pt)
    export_to_onnx(scratch_pt, scratch_onnx, cfg)

    # The current model is already on disk from the previous promotion.
    current_onnx = cfg.model_cache_dir / "latest.onnx"

    gauntlet = play_gauntlet
    if gauntlet is None:
        def gauntlet(_c, _cur):
            return _default_gauntlet(
                scratch_onnx, current_onnx,
                simulations=cfg.num_simulations,
                n_games=cfg.gating_games,
            )

    decision = decide_promotion(
        state, candidate=scratch_onnx, current=current_onnx,
        play_gauntlet=gauntlet,
    )
    return decision.promote, decision.state, decision.score, decision.reason


# ---------------------------------------------------------------------------
# Main trainer loop
# ---------------------------------------------------------------------------

def run_trainer(cfg: AsyncConfig) -> None:
    """Run the continuous trainer loop."""
    cfg.ensure_cache_dirs()
    cfg.validate()
    setup_json_logging("trainer", run_id=cfg.run_id)
    log_event("trainer.start", run_id=cfg.run_id,
              steps_per_cycle=cfg.steps_per_cycle,
              batch_size=cfg.batch_size)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    current_n_total = storage.count_positions(storage.SELFPLAY_PREFIX)
    initial_window = sublinear_window_size(current_n_total)
    logger.info("Trainer starting on device: {} | N_total={:,} window={:,} "
                "steps/cycle={}",
                device, current_n_total, initial_window, cfg.steps_per_cycle)

    current_version = _read_model_version()

    model = build_model(cfg).to(device)
    if current_version > 0:
        local_pt = cfg.model_cache_dir / "checkpoint.pt"
        storage.get_file(storage.CHECKPOINT_PT, local_pt)
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

    cycle = 0
    total_steps_all_time = 0
    fresh_run_steps = 0  # for LR warmup — resets on promotion is NOT needed, it's "fresh run"
    bucket = TrainBucket(cfg.max_train_steps_per_new_data,
                         batch_size=cfg.batch_size,
                         max_seed=cfg.steps_per_cycle * cfg.batch_size,
                         max_tokens=float(cfg.steps_per_cycle * cfg.batch_size))

    # SWA snapshot buffer + sample counter.
    swa_buf = SwaSnapshotBuffer()
    samples_since_last_snapshot = 0
    # Promotion is gated on *new* positions since the last promotion.
    # On a fresh trainer start we set this to 0 so that all existing
    # self-play data counts as "new since v{current}" — otherwise a
    # restart after the trainer has been idle (or crashed before
    # promoting) would treat the accumulated backlog as already-promoted
    # and wait another full cycle to promote anything.
    positions_at_last_promote = 0
    # Gauntlet held-out batch for BN-stat update after SWA average.
    held_out_batch: torch.Tensor | None = None

    ac_dtype = _autocast_dtype(device)

    def reload_buffer() -> tuple[ReplayBufferV2, DataLoader, int]:
        n_total = storage.count_positions(storage.SELFPLAY_PREFIX)
        window = sublinear_window_size(n_total)
        ds = ReplayBufferV2(
            cfg.data_cache_dir / "selfplay",
            window_size=window,
            s3_prefix=storage.SELFPLAY_PREFIX,
            imitation_mix=cfg.imitation_mix,
        )
        dl = DataLoader(
            ds, batch_size=cfg.batch_size, num_workers=0,
            collate_fn=_v2_collate,
        )
        return ds, dl, n_total

    while True:
        dataset, dataloader, n_total = reload_buffer()
        if dataset.total_positions == 0:
            logger.info("No valid training data found, waiting...")
            time.sleep(30)
            continue

        cycle += 1
        cycle_t0 = time.time()

        logger.info("")
        logger.info("=" * 60)
        logger.info("Training cycle {} | model v{} | N_total={:,} window={:,}",
                    cycle, current_version, n_total, dataset.window_size)
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

        logger.info("Replay buffer: {} files, {:,} positions (window={:,})",
                    len(dataset.files), dataset.total_positions, dataset.window_size)
        logger.info("Train bucket: {:.0f} tokens available ({:,} new, +{:.0f} tokens)",
                    bucket.tokens, bucket._last_new, bucket._last_added)

        # Training
        model.train()
        step = 0
        cycle_policy_loss = 0.0
        cycle_value_loss = 0.0
        cycle_mlh_loss = 0.0
        cycle_stv_loss = 0.0
        cycle_aux_loss = 0.0
        cycle_total_loss = 0.0
        loss_weights = LossWeights()
        train_t0 = time.time()

        logger.info("Training for {} steps (batch_size={})...",
                    cfg.steps_per_cycle, cfg.batch_size)

        while step < cfg.steps_per_cycle:
            if not bucket.has_budget():
                logger.info("  Bucket empty mid-cycle at step {}, waiting...", step)
                while not bucket.has_budget():
                    time.sleep(30)
                    n_total = storage.count_positions(storage.SELFPLAY_PREFIX)
                    bucket.update(n_total, window_size=dataset.window_size)

            for batch_np in dataloader:
                if step >= cfg.steps_per_cycle or not bucket.has_budget():
                    break

                boards, targets, legal_mask = _v2_batch_to_torch(batch_np, device)

                # Keep one batch aside for post-SWA BN-stat update.
                if held_out_batch is None:
                    held_out_batch = boards.detach().clone()

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

                cycle_policy_loss += breakdown.policy.item()
                cycle_value_loss += breakdown.value.item()
                cycle_mlh_loss += breakdown.mlh.item()
                cycle_stv_loss += breakdown.stv.item()
                cycle_aux_loss += breakdown.aux_policy.item()
                cycle_total_loss += breakdown.total.item()
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
                    cycle=cycle,
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

                if step % 100 == 0 or step == cfg.steps_per_cycle:
                    elapsed = time.time() - train_t0
                    sps = step / elapsed if elapsed > 0 else 0
                    logger.info(
                        "  step {:>5}/{} lr={:.5f} | policy={:.4f} value={:.4f} "
                        "mlh={:.4f} stv={:.4f} aux={:.4f} total={:.4f} "
                        "| {:.1f} steps/s | {:.0f}s",
                        step, cfg.steps_per_cycle, lr,
                        cycle_policy_loss / step, cycle_value_loss / step,
                        cycle_mlh_loss / step, cycle_stv_loss / step,
                        cycle_aux_loss / step, cycle_total_loss / step,
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

                if step % cfg.reload_interval == 0 and step < cfg.steps_per_cycle:
                    dataset, dataloader, n_total = reload_buffer()
                    bucket.update(n_total, window_size=dataset.window_size)
                    logger.info(
                        "  Reloaded: {} files, {:,} pos (window={:,}) | bucket: {:.0f}",
                        len(dataset.files), dataset.total_positions,
                        dataset.window_size, bucket.tokens,
                    )
                    break

        train_elapsed = time.time() - train_t0
        avg_policy = cycle_policy_loss / max(step, 1)
        avg_value = cycle_value_loss / max(step, 1)
        logger.info(
            "Cycle done: {} steps in {:.0f}s | policy={:.4f} value={:.4f} "
            "mlh={:.4f} stv={:.4f} aux={:.4f}",
            step, train_elapsed, avg_policy, avg_value,
            cycle_mlh_loss / max(step, 1), cycle_stv_loss / max(step, 1),
            cycle_aux_loss / max(step, 1),
        )

        # Promotion gate: only promote every cfg.promote_every_new_positions new positions.
        new_positions = n_total - positions_at_last_promote
        if new_positions < cfg.promote_every_new_positions:
            logger.info(
                "Skipping promotion — only {:,} new positions since v{} "
                "(need {:,})",
                new_positions, current_version, cfg.promote_every_new_positions,
            )
            continue

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
            if held_out_batch is not None:
                update_bn_stats(scratch, [(held_out_batch,)], device=device)
                averaged_sd = {k: v.detach().cpu().clone()
                               for k, v in scratch.state_dict().items()}
            del scratch

        new_version = current_version + 1
        promote, gate_state, gate_score, gate_reason = _try_gate_promotion(
            cfg, averaged_sd, new_version, current_version,
        )
        save_gate_state(gate_state)

        if promote:
            _promote_model(cfg, model, new_version, state_dict=averaged_sd)
            current_version = new_version
            positions_at_last_promote = n_total
            logger.info(
                "Promoted to v{} ({}, score={:.3f}) | total steps: {:,}",
                new_version, gate_reason, gate_score, total_steps_all_time,
            )
        else:
            logger.warning(
                "Promotion gated out ({}, score={:.3f}) — staying on v{}",
                gate_reason, gate_score, current_version,
            )

        cycle_elapsed = time.time() - cycle_t0
        notify_training_cycle(
            cycle=cycle, version=current_version, steps=step,
            total_steps=total_steps_all_time, positions=dataset.total_positions,
            policy_loss=avg_policy, value_loss=avg_value,
            elapsed_seconds=cycle_elapsed,
        )
