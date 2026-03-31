from __future__ import annotations
"""Continuous trainer for async distributed training.

Two training regimes:

1. **Bootstrap** (`_run_bootstrap`): Runs when no model exists yet. Waits for
   workers to generate enough imitation data, then trains for multiple epochs
   over the full static dataset. Promotes v1 once.

2. **Self-play loop** (`run_trainer` main loop): Runs continuously after
   bootstrap. Samples from a recency-weighted replay buffer, trains for N
   steps per cycle, exports and promotes unconditionally, repeat. Workers
   pick up new models and generate progressively stronger data.
"""

import json
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, IterableDataset

from .config import AsyncConfig
from .export import export_to_onnx
from .model import build_model
from .slack import notify_training_cycle

_shutdown_requested = False


def _handle_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.warning("Shutdown requested (signal {}), finishing current cycle...", signum)


def _read_model_version(cfg: AsyncConfig) -> int:
    if cfg.best_meta_path.exists():
        try:
            return json.loads(cfg.best_meta_path.read_text()).get("version", 0)
        except (json.JSONDecodeError, OSError):
            return 0
    return 0


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.rename(path)


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy file atomically via write-to-temp-then-rename."""
    import shutil
    tmp = dst.with_name(dst.name + '.tmp')
    shutil.copy2(src, tmp)
    tmp.rename(dst)


# ---------------------------------------------------------------------------
# Train bucket (rate limiter)
# ---------------------------------------------------------------------------

class TrainBucket:
    """Token-bucket rate limiter that throttles training to match data inflow.

    Each new data position adds ``ratio`` tokens to the bucket. Each training
    step consumes 1 token.  When the bucket is empty the trainer should pause
    and wait for workers to produce more data.

    On first call to :meth:`update`, the bucket is seeded with the full
    position count so the trainer can start immediately.
    """

    def __init__(self, ratio: float, max_seed: int | None = None,
                 max_tokens: float | None = None):
        if ratio <= 0:
            raise ValueError(f"ratio must be positive, got {ratio}")
        self.ratio = ratio
        self._max_seed = max_seed
        self._max_tokens = max_tokens
        self._tokens: float = 0.0
        self._prev_positions: int | None = None
        self._cumulative_positions: int = 0
        self._last_new: int = 0
        self._last_added: float = 0.0

    def update(self, total_positions: int) -> None:
        """Refresh the bucket with newly observed positions.

        ``total_positions`` should be a cumulative count (all data ever
        produced), **not** the capped replay-buffer size. Otherwise the
        bucket starves once the buffer saturates.
        """
        if self._prev_positions is None:
            # First observation — seed so training can begin right away,
            # but cap so a restart doesn't grant unlimited budget.
            tokens = total_positions * self.ratio
            if self._max_seed is not None:
                tokens = min(tokens, self._max_seed)
            self._tokens = tokens
            self._last_new = total_positions
            self._last_added = tokens
        else:
            new = max(0, total_positions - self._prev_positions)
            added = new * self.ratio
            self._tokens += added
            self._last_new = new
            self._last_added = added
        if self._max_tokens is not None:
            self._tokens = min(self._tokens, self._max_tokens)
        self._prev_positions = total_positions
        self._cumulative_positions = total_positions

    def consume(self, n: int = 1) -> None:
        """Consume tokens for completed training steps."""
        self._tokens -= n

    @property
    def tokens(self) -> float:
        return self._tokens

    def has_budget(self) -> bool:
        return self._tokens > 0


# ---------------------------------------------------------------------------
# Replay buffer (self-play regime)
# ---------------------------------------------------------------------------

class ReplayBuffer(IterableDataset):
    """Streams training data with uniform sampling from a sliding window.

    Selects the most recent .npz files up to max_positions, then samples
    files uniformly at random. Positions within each file pass through a
    shuffle buffer to decorrelate batches.

    The iterator is infinite — the training loop controls how many steps
    to consume before reloading.
    """

    SHUFFLE_BUFFER_SIZE = 100_000

    def __init__(self, data_dir: Path, max_positions: int = 5_000_000):
        self.data_dir = data_dir
        self.max_positions = max_positions
        self.files, self.total_positions = self._select_files()

    def _select_files(self) -> tuple[list[Path], int]:
        all_files = sorted(
            (f for f in self.data_dir.glob("*.npz") if ".tmp" not in f.name),
            key=lambda f: f.stat().st_mtime,
        )
        if not all_files:
            return [], 0

        selected = []
        total = 0
        for f in reversed(all_files):
            try:
                with np.load(f) as data:
                    n = len(data["outcomes"])
            except (OSError, ValueError, KeyError):
                continue
            selected.append(f)
            total += n
            if total >= self.max_positions:
                break

        selected.reverse()
        total = min(total, self.max_positions)
        return selected, total

    def stats(self) -> dict:
        """Return buffer diagnostics for logging."""
        if not self.files:
            return {}
        now = time.time()
        mtimes = [f.stat().st_mtime for f in self.files]
        ages = sorted(now - t for t in mtimes)
        # Parse version from filenames like sp_v5_... or im_v0_...
        version_counts: dict[str, int] = {}
        for f in self.files:
            parts = f.name.split("_")
            v = parts[1] if len(parts) >= 2 else "unknown"
            version_counts[v] = version_counts.get(v, 0) + 1
        return {
            "files": len(self.files),
            "positions": self.total_positions,
            "oldest_age_min": round(ages[-1] / 60, 1),
            "newest_age_min": round(ages[0] / 60, 1),
            "median_age_min": round(ages[len(ages) // 2] / 60, 1),
            "versions": version_counts,
        }

    def __iter__(self):
        if not self.files:
            return

        buf_b, buf_p, buf_o = [], [], []

        while True:
            # Pick a file uniformly at random
            [chosen] = random.choices(self.files, k=1)
            try:
                data = np.load(chosen)
                boards, policies, outcomes = data["boards"], data["policies"], data["outcomes"]
            except (OSError, ValueError, KeyError):
                continue

            buf_b.extend(boards)
            buf_p.extend(policies)
            buf_o.extend(outcomes)

            # Drain shuffle buffer when large enough
            while len(buf_b) >= self.SHUFFLE_BUFFER_SIZE:
                perm = list(range(len(buf_b)))
                random.shuffle(perm)
                drain = len(buf_b) // 2
                for j in perm[:drain]:
                    yield (torch.from_numpy(buf_b[j]),
                           torch.from_numpy(buf_p[j]),
                           torch.tensor(buf_o[j], dtype=torch.float32))
                keep = perm[drain:]
                buf_b = [buf_b[j] for j in keep]
                buf_p = [buf_p[j] for j in keep]
                buf_o = [buf_o[j] for j in keep]


# ---------------------------------------------------------------------------
# Bootstrap data loading
# ---------------------------------------------------------------------------

def _count_positions(data_dir: Path) -> int:
    """Count total positions across all .npz files."""
    total = 0
    for f in data_dir.glob("*.npz"):
        if ".tmp" in f.name:
            continue
        try:
            with np.load(f) as data:
                total += len(data["outcomes"])
        except (OSError, ValueError, KeyError):
            continue
    return total


# ---------------------------------------------------------------------------
# Bootstrap training (imitation regime)
# ---------------------------------------------------------------------------

def _run_bootstrap(cfg: AsyncConfig, model: torch.nn.Module,
                   optimizer: optim.Optimizer, device: torch.device) -> int:
    """Train on imitation data for many steps. Returns new model version.

    Runs bootstrap_steps steps with uniform sampling (no recency bias)
    over all available imitation data.
    """
    logger.info("Bootstrap mode: waiting for {:,} imitation positions...",
                cfg.min_positions_to_start)

    # Wait for enough data
    while not _shutdown_requested:
        available = _count_positions(cfg.training_data_dir)
        if available >= cfg.min_positions_to_start:
            break
        logger.info("Waiting for data: {:,}/{:,} positions",
                    available, cfg.min_positions_to_start)
        time.sleep(30)

    if _shutdown_requested:
        return 0

    total_steps_target = cfg.bootstrap_steps

    logger.info("")
    logger.info("=" * 60)
    logger.info("Bootstrap training: {:,} steps over {:,} positions",
                total_steps_target, available)
    logger.info("=" * 60)

    dataset = ReplayBuffer(cfg.training_data_dir,
                           max_positions=cfg.replay_buffer_size)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0)

    model.train()
    total_steps = 0
    cumulative_policy_loss = 0.0
    cumulative_value_loss = 0.0
    t0 = time.time()

    for boards, policies, outcomes in dataloader:
        if _shutdown_requested or total_steps >= total_steps_target:
            break

        boards = boards.to(device)
        policies = policies.to(device)
        outcomes = outcomes.to(device)

        pred_policy, pred_wdl = model(boards)

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

    if _shutdown_requested:
        torch.save(model.state_dict(), cfg.candidate_checkpoint_path)
        logger.info("Saved checkpoint before shutdown.")
        return 0

    train_elapsed = time.time() - t0
    logger.info("Bootstrap training complete: {:,} steps in {:.0f}s", total_steps, train_elapsed)

    # Export and promote v1
    logger.info("Exporting bootstrap model...")
    torch.save(model.state_dict(), cfg.candidate_checkpoint_path)
    export_to_onnx(cfg.candidate_checkpoint_path, cfg.candidate_model_path, cfg)

    new_version = 1
    _atomic_copy(cfg.candidate_model_path, cfg.best_model_path)
    _atomic_copy(cfg.candidate_checkpoint_path, cfg.best_checkpoint_path)
    _atomic_copy(cfg.candidate_model_path, cfg.models_dir / f"v{new_version}.onnx")
    _atomic_write_json(cfg.best_meta_path, {
        "version": new_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    logger.info("Promoted bootstrap model to v{} ({:,} steps, {:.0f}s)",
                new_version, total_steps, train_elapsed)

    _log_event(cfg, {
        "event": "bootstrap_complete",
        "version": new_version,
        "steps": total_steps,
        "positions": available,
        "elapsed_seconds": round(train_elapsed, 1),
    })

    return new_version


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_buffer_stats(prefix: str, s: dict) -> None:
    logger.info(
        "{}: {:,} pos, {} files | age: {:.0f}–{:.0f}min (median {:.0f}min) | versions: {}",
        prefix, s.get("positions", 0), s.get("files", 0),
        s.get("newest_age_min", 0), s.get("oldest_age_min", 0),
        s.get("median_age_min", 0), s.get("versions", {}),
    )


def _log_event(cfg: AsyncConfig, event: dict) -> None:
    log_path = cfg.logs_dir / "trainer.jsonl"
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a") as f:
        f.write(json.dumps(event) + "\n")


# ---------------------------------------------------------------------------
# Main trainer loop (self-play regime)
# ---------------------------------------------------------------------------

def run_trainer(cfg: AsyncConfig) -> None:
    """Run the continuous trainer loop."""
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    cfg.ensure_dirs()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Trainer starting on device: {} | buffer={:,} steps/cycle={}",
                device, cfg.replay_buffer_size, cfg.steps_per_cycle)

    current_version = _read_model_version(cfg)

    # Persistent model and optimizer across cycles
    model = build_model(cfg).to(device)
    if cfg.best_checkpoint_path.exists():
        logger.info("Loading checkpoint: {}", cfg.best_checkpoint_path)
        model.load_state_dict(torch.load(cfg.best_checkpoint_path, map_location=device, weights_only=True))

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.l2_regularization,
    )

    # Bootstrap: train on imitation data if no model exists yet
    if not cfg.best_model_path.exists():
        current_version = _run_bootstrap(cfg, model, optimizer, device)
        if _shutdown_requested:
            return

    cycle = 0
    total_steps_all_time = 0
    bucket = TrainBucket(cfg.max_train_steps_per_new_data,
                         max_seed=cfg.steps_per_cycle,
                         max_tokens=float(cfg.steps_per_cycle))

    while not _shutdown_requested:
        cycle += 1
        cycle_t0 = time.time()

        logger.info("")
        logger.info("=" * 60)
        logger.info("Training cycle {} | model v{}", cycle, current_version)
        logger.info("=" * 60)

        # Check if model was promoted externally (shouldn't happen with single
        # trainer, but defensive)
        disk_version = _read_model_version(cfg)
        if disk_version > current_version and cfg.best_checkpoint_path.exists():
            logger.info("Model updated externally: v{} -> v{}, reloading", current_version, disk_version)
            model.load_state_dict(torch.load(cfg.best_checkpoint_path, map_location=device, weights_only=True))
            optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.l2_regularization)
            current_version = disk_version

        # Load data
        def reload_buffer():
            ds = ReplayBuffer(cfg.training_data_dir,
                              max_positions=cfg.replay_buffer_size)
            dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=0)
            return ds, dl

        dataset, dataloader = reload_buffer()
        if dataset.total_positions == 0:
            logger.info("No valid training data found, waiting...")
            time.sleep(30)
            continue

        bucket.update(_count_positions(cfg.training_data_dir))

        # Rate limit: wait for workers to produce enough data
        if not bucket.has_budget():
            logger.info("Train bucket empty ({:.0f} tokens), waiting for new data...",
                        bucket.tokens)
            while not bucket.has_budget() and not _shutdown_requested:
                time.sleep(30)
                dataset, dataloader = reload_buffer()
                bucket.update(_count_positions(cfg.training_data_dir))
                if not bucket.has_budget():
                    logger.info("  Still waiting... bucket={:.0f} tokens, {:,} cumulative positions",
                                bucket.tokens, bucket._cumulative_positions)
            if _shutdown_requested:
                break

        buf_stats = dataset.stats()
        _log_buffer_stats("Replay buffer", buf_stats)
        logger.info("Train bucket: {:.0f} tokens available ({:,} new positions, +{:.0f} tokens, {:,} cumulative)",
                    bucket.tokens, bucket._last_new, bucket._last_added, bucket._cumulative_positions)

        # Step-based training
        model.train()
        step = 0
        cycle_policy_loss = 0.0
        cycle_value_loss = 0.0
        train_t0 = time.time()
        log_interval = 100  # log every N steps

        logger.info("Training for {} steps (batch_size={})...", cfg.steps_per_cycle, cfg.batch_size)

        while step < cfg.steps_per_cycle and not _shutdown_requested:
            # Wait for budget if bucket is exhausted mid-cycle
            if not bucket.has_budget():
                logger.info("  Bucket empty mid-cycle at step {}, waiting for new data...", step)
                while not bucket.has_budget() and not _shutdown_requested:
                    time.sleep(30)
                    dataset, dataloader = reload_buffer()
                    bucket.update(_count_positions(cfg.training_data_dir))
                    if not bucket.has_budget():
                        logger.info("    Still waiting... bucket={:.0f} tokens, {:,} positions",
                                    bucket.tokens, dataset.total_positions)
                if _shutdown_requested:
                    break

            for boards, policies, outcomes in dataloader:
                if _shutdown_requested or step >= cfg.steps_per_cycle or not bucket.has_budget():
                    break

                boards = boards.to(device)
                policies = policies.to(device)
                outcomes = outcomes.to(device)

                pred_policy, pred_wdl = model(boards)

                # Policy loss: cross-entropy with MCTS policy as soft target
                log_probs = torch.log_softmax(pred_policy, dim=1)
                policy_loss = -torch.sum(policies * log_probs, dim=1).mean()

                # Value loss: cross-entropy on WDL targets
                log_probs_v = torch.log_softmax(pred_wdl, dim=1)
                value_loss = -torch.sum(outcomes * log_probs_v, dim=1).mean()

                total_loss = policy_loss + value_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                pl = policy_loss.item()
                vl = value_loss.item()
                cycle_policy_loss += pl
                cycle_value_loss += vl
                step += 1
                total_steps_all_time += 1
                bucket.consume()

                if step % log_interval == 0 or step == cfg.steps_per_cycle:
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

                # Periodic reload: pick up fresh worker data and refill bucket
                if step % cfg.reload_interval == 0 and step < cfg.steps_per_cycle:
                    dataset, dataloader = reload_buffer()
                    bucket.update(_count_positions(cfg.training_data_dir))
                    _log_buffer_stats("  Reloaded buffer", dataset.stats())
                    logger.info("  Train bucket: {:.0f} tokens remaining ({:,} new positions, +{:.0f} tokens)",
                                bucket.tokens, bucket._last_new, bucket._last_added)
                    break  # restart with new dataloader

        train_elapsed = time.time() - train_t0
        avg_policy = cycle_policy_loss / max(step, 1)
        avg_value = cycle_value_loss / max(step, 1)
        logger.info(
            "Training complete: {} steps in {:.0f}s | policy={:.4f} value={:.4f}",
            step, train_elapsed, avg_policy, avg_value,
        )

        if _shutdown_requested:
            torch.save(model.state_dict(), cfg.candidate_checkpoint_path)
            logger.info("Saved checkpoint before shutdown.")
            break

        # Export candidate
        logger.info("Exporting candidate ONNX model...")
        candidate_pt = cfg.candidate_checkpoint_path
        candidate_onnx = cfg.candidate_model_path
        torch.save(model.state_dict(), candidate_pt)
        export_to_onnx(candidate_pt, candidate_onnx, cfg)

        # Always promote — Elo service handles strength tracking
        new_version = current_version + 1
        _atomic_copy(candidate_onnx, cfg.best_model_path)
        _atomic_copy(candidate_pt, cfg.best_checkpoint_path)
        _atomic_copy(candidate_onnx, cfg.models_dir / f"v{new_version}.onnx")
        _atomic_write_json(cfg.best_meta_path, {
            "version": new_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        current_version = new_version

        cycle_elapsed = time.time() - cycle_t0
        logger.info("Promoted to v{} (cycle {} done in {:.0f}s | total steps: {:,})",
                     new_version, cycle, cycle_elapsed, total_steps_all_time)

        _log_event(cfg, {
            "event": "cycle_complete",
            "cycle": cycle,
            "version": new_version,
            "steps": step,
            "total_steps": total_steps_all_time,
            "positions": dataset.total_positions,
            "policy_loss": round(avg_policy, 4),
            "value_loss": round(avg_value, 4),
            "elapsed_seconds": round(cycle_elapsed, 1),
            "bucket_tokens_remaining": round(bucket.tokens, 1),
            "cumulative_positions": bucket._cumulative_positions,
            "initial_buffer_stats": buf_stats,
        })
        notify_training_cycle(
            cycle=cycle, version=new_version, steps=step,
            total_steps=total_steps_all_time, positions=dataset.total_positions,
            policy_loss=avg_policy, value_loss=avg_value,
            elapsed_seconds=cycle_elapsed,
        )

    logger.info("Trainer shutdown.")
    sys.exit(0)
