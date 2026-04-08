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

import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, IterableDataset

from . import storage
from .config import AsyncConfig
from .export import export_to_onnx
from .model import build_model
from .slack import notify_training_cycle


def _read_model_version() -> int:
    """Read the current model version from S3."""
    try:
        return storage.get_json(storage.LATEST_META).get("version", 0)
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

    def update(self, total_positions: int) -> None:
        if self._prev_positions is None:
            tokens = total_positions * self.target_passes
            if self._max_seed is not None:
                tokens = min(tokens, self._max_seed)
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

        buf_b, buf_p, buf_o = [], [], []

        while True:
            [chosen] = random.choices(self.files, k=1)
            try:
                data = np.load(chosen, mmap_mode="r")
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
                          max_positions=cfg.replay_buffer_size,
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
    t0 = time.time()

    while total_steps < total_steps_target:
        for boards, policies, outcomes in dataloader:
            if total_steps >= total_steps_target:
                break

            boards = boards.to(device)
            policies = policies.to(device)
            outcomes = outcomes.to(device)

            preds = model(boards)
            pred_policy = preds["policy"]
            pred_wdl = preds["wdl"]
            # preds["mlh"], preds["stv"], preds["aux_policy"] computed but
            # unused in loss until chunk 4 wires up the real loss module.

            log_probs = torch.log_softmax(pred_policy, dim=1)
            policy_loss = -torch.sum(policies * log_probs, dim=1).mean()

            log_probs_v = torch.log_softmax(pred_wdl, dim=1)
            value_loss = -torch.sum(outcomes * log_probs_v, dim=1).mean()

            total_loss = policy_loss + value_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            cumulative_policy_loss += policy_loss.item()
            cumulative_value_loss += value_loss.item()
            total_steps += 1

            if total_steps % 100 == 0:
                elapsed = time.time() - t0
                avg_p = cumulative_policy_loss / total_steps
                avg_v = cumulative_value_loss / total_steps
                logger.info(
                    "  step {:>5}/{:,} | loss: policy={:.4f} value={:.4f} "
                    "total={:.4f} | {:.1f} steps/s | {:.0f}s elapsed",
                    total_steps, total_steps_target,
                    avg_p, avg_v, avg_p + avg_v,
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

def _promote_model(cfg: AsyncConfig, model: torch.nn.Module, version: int) -> None:
    """Export model and upload to S3 as the latest version."""
    cfg.ensure_cache_dirs()

    local_pt = cfg.model_cache_dir / "checkpoint.pt"
    local_onnx = cfg.model_cache_dir / "latest.onnx"

    torch.save(model.state_dict(), local_pt)
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
# Main trainer loop
# ---------------------------------------------------------------------------

def run_trainer(cfg: AsyncConfig) -> None:
    """Run the continuous trainer loop."""
    cfg.ensure_cache_dirs()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Trainer starting on device: {} | buffer={:,} steps/cycle={}",
                device, cfg.replay_buffer_size, cfg.steps_per_cycle)

    current_version = _read_model_version()

    model = build_model(cfg).to(device)
    if current_version > 0:
        local_pt = cfg.model_cache_dir / "checkpoint.pt"
        storage.get_file(storage.CHECKPOINT_PT, local_pt)
        logger.info("Loading checkpoint v{}", current_version)
        model.load_state_dict(torch.load(local_pt, map_location=device, weights_only=True))

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.l2_regularization,
    )

    # Bootstrap if no model exists
    if current_version == 0:
        current_version = _run_bootstrap(cfg, model, optimizer, device)
        optimizer = optim.SGD(
            model.parameters(), lr=cfg.learning_rate,
            momentum=cfg.momentum, weight_decay=cfg.l2_regularization,
        )

    cycle = 0
    total_steps_all_time = 0
    bucket = TrainBucket(cfg.max_train_steps_per_new_data,
                         batch_size=cfg.batch_size,
                         max_seed=cfg.steps_per_cycle * cfg.batch_size,
                         max_tokens=float(cfg.steps_per_cycle * cfg.batch_size))

    def reload_buffer():
        ds = ReplayBuffer(cfg.data_cache_dir / "selfplay",
                          max_positions=cfg.replay_buffer_size,
                          s3_prefix=storage.SELFPLAY_PREFIX)
        dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=0)
        return ds, dl

    while True:
        # Wait for selfplay data before starting a cycle
        dataset, dataloader = reload_buffer()
        if dataset.total_positions == 0:
            logger.info("No valid training data found, waiting...")
            time.sleep(30)
            continue

        cycle += 1
        cycle_t0 = time.time()

        logger.info("")
        logger.info("=" * 60)
        logger.info("Training cycle {} | model v{}", cycle, current_version)
        logger.info("=" * 60)

        bucket.update(storage.count_positions(storage.SELFPLAY_PREFIX))

        # Rate limit: wait for workers to produce enough data
        if not bucket.has_budget():
            logger.info("Train bucket empty ({:.0f} tokens), waiting for new data...",
                        bucket.tokens)
            while not bucket.has_budget():
                time.sleep(30)
                bucket.update(storage.count_positions(storage.SELFPLAY_PREFIX))
                if not bucket.has_budget():
                    logger.info("  Still waiting... bucket={:.0f} tokens, {:,} cumulative positions",
                                bucket.tokens, bucket._cumulative_positions)

        logger.info("Replay buffer: {} files, {:,} positions",
                    len(dataset.files), dataset.total_positions)
        logger.info("Train bucket: {:.0f} tokens available ({:,} new positions, +{:.0f} tokens)",
                    bucket.tokens, bucket._last_new, bucket._last_added)

        # Training
        model.train()
        step = 0
        cycle_policy_loss = 0.0
        cycle_value_loss = 0.0
        train_t0 = time.time()

        logger.info("Training for {} steps (batch_size={})...", cfg.steps_per_cycle, cfg.batch_size)

        while step < cfg.steps_per_cycle:
            if not bucket.has_budget():
                logger.info("  Bucket empty mid-cycle at step {}, waiting for new data...", step)
                while not bucket.has_budget():
                    time.sleep(30)
                    bucket.update(storage.count_positions(storage.SELFPLAY_PREFIX))
                    if not bucket.has_budget():
                        logger.info("    Still waiting... bucket={:.0f} tokens", bucket.tokens)

            for boards, policies, outcomes in dataloader:
                if step >= cfg.steps_per_cycle or not bucket.has_budget():
                    break

                boards = boards.to(device)
                policies = policies.to(device)
                outcomes = outcomes.to(device)

                preds = model(boards)
                pred_policy = preds["policy"]
                pred_wdl = preds["wdl"]

                log_probs = torch.log_softmax(pred_policy, dim=1)
                policy_loss = -torch.sum(policies * log_probs, dim=1).mean()

                log_probs_v = torch.log_softmax(pred_wdl, dim=1)
                value_loss = -torch.sum(outcomes * log_probs_v, dim=1).mean()

                total_loss = policy_loss + value_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                cycle_policy_loss += policy_loss.item()
                cycle_value_loss += value_loss.item()
                step += 1
                total_steps_all_time += 1
                bucket.consume()

                if step % 100 == 0 or step == cfg.steps_per_cycle:
                    elapsed = time.time() - train_t0
                    steps_per_sec = step / elapsed if elapsed > 0 else 0
                    avg_p = cycle_policy_loss / step
                    avg_v = cycle_value_loss / step
                    logger.info(
                        "  step {:>5}/{} | loss: policy={:.4f} value={:.4f} total={:.4f} | "
                        "{:.1f} steps/s | {:.0f}s elapsed",
                        step, cfg.steps_per_cycle, avg_p, avg_v, avg_p + avg_v,
                        steps_per_sec, elapsed,
                    )

                if step % cfg.reload_interval == 0 and step < cfg.steps_per_cycle:
                    dataset, dataloader = reload_buffer()
                    bucket.update(storage.count_positions(storage.SELFPLAY_PREFIX))
                    logger.info("  Reloaded buffer: {} files, {:,} positions | bucket: {:.0f} tokens",
                                len(dataset.files), dataset.total_positions, bucket.tokens)
                    break

        train_elapsed = time.time() - train_t0
        avg_policy = cycle_policy_loss / max(step, 1)
        avg_value = cycle_value_loss / max(step, 1)
        logger.info("Training complete: {} steps in {:.0f}s | policy={:.4f} value={:.4f}",
                    step, train_elapsed, avg_policy, avg_value)

        # Promote
        new_version = current_version + 1
        _promote_model(cfg, model, new_version)
        current_version = new_version

        cycle_elapsed = time.time() - cycle_t0
        logger.info("Promoted to v{} (cycle {} done in {:.0f}s | total steps: {:,})",
                     new_version, cycle, cycle_elapsed, total_steps_all_time)

        notify_training_cycle(
            cycle=cycle, version=new_version, steps=step,
            total_steps=total_steps_all_time, positions=dataset.total_positions,
            policy_loss=avg_policy, value_loss=avg_value,
            elapsed_seconds=cycle_elapsed,
        )
