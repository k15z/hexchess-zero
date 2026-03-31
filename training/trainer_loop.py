from __future__ import annotations
"""Continuous trainer for async distributed training.

Runs continuously: sample from recency-weighted replay buffer, train for N
steps, export and promote unconditionally, repeat. No generation gating —
fresh worker data is picked up via periodic buffer reloads. Strength tracking
is handled by the Elo service.
"""

import json
import math
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import random

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


class ReplayBuffer(IterableDataset):
    """Streams training data with recency-weighted sampling.

    Selects the most recent .npz files up to max_positions, then samples
    files with probability proportional to exp(-age / half_life). Positions
    within each file pass through a shuffle buffer to decorrelate batches.

    The iterator is infinite — the training loop controls how many steps
    to consume before reloading.
    """

    SHUFFLE_BUFFER_SIZE = 100_000

    def __init__(self, data_dir: Path, max_positions: int = 5_000_000,
                 half_life: float = 10800.0):
        self.data_dir = data_dir
        self.max_positions = max_positions
        self.half_life = half_life
        self.files, self.total_positions = self._select_files()
        self.weights = self._compute_weights()

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

    def _compute_weights(self) -> list[float]:
        if not self.files:
            return []
        self._mtimes = [f.stat().st_mtime for f in self.files]
        newest = max(self._mtimes)
        return [math.exp(-(newest - t) / self.half_life) for t in self._mtimes]

    def stats(self) -> dict:
        """Return buffer diagnostics for logging."""
        if not self.files:
            return {}
        now = time.time()
        ages = sorted(now - t for t in self._mtimes)
        # Parse version from filenames like sp_v5_... or im_v0_...
        file_versions = []
        for f in self.files:
            parts = f.name.split("_")
            file_versions.append(parts[1] if len(parts) >= 2 else "unknown")
        version_counts: dict[str, int] = {}
        for v in file_versions:
            version_counts[v] = version_counts.get(v, 0) + 1
        # Effective sampling weight per version
        total_weight = sum(self.weights)
        version_weight: dict[str, float] = {}
        for v, w in zip(file_versions, self.weights):
            version_weight[v] = version_weight.get(v, 0.0) + w
        version_weight_pct = {
            v: round(w / total_weight * 100, 1)
            for v, w in sorted(version_weight.items())
        }
        return {
            "files": len(self.files),
            "positions": self.total_positions,
            "oldest_age_min": round(ages[-1] / 60, 1),
            "newest_age_min": round(ages[0] / 60, 1),
            "median_age_min": round(ages[len(ages) // 2] / 60, 1),
            "versions": version_counts,
            "weight_pct_by_version": version_weight_pct,
        }

    def __iter__(self):
        if not self.files:
            return

        buf_b, buf_p, buf_o = [], [], []

        while True:
            # Pick a file weighted by recency
            [chosen] = random.choices(self.files, weights=self.weights, k=1)
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



def _estimate_positions(data_dir: Path) -> int:
    """Estimate total positions by sampling a few files."""
    npz_files = [f for f in data_dir.glob("*.npz") if ".tmp" not in f.name]
    if not npz_files:
        return 0
    sample = npz_files[-min(3, len(npz_files)):]
    total_sampled = 0
    for f in sample:
        try:
            with np.load(f) as data:
                total_sampled += len(data["outcomes"])
        except (OSError, ValueError, KeyError):
            continue
    if not total_sampled:
        return 0
    return int(total_sampled / len(sample) * len(npz_files))


def _prune_old_data(cfg: AsyncConfig) -> int:
    """Remove old .npz files beyond the replay buffer limit and stale .tmp files."""
    all_files = sorted(
        (f for f in cfg.training_data_dir.glob("*.npz") if ".tmp" not in f.name),
        key=lambda f: f.stat().st_mtime,
    )

    # Clean up orphaned .tmp.npz files older than 1 hour
    now = time.time()
    removed = 0
    for f in cfg.training_data_dir.glob("*.tmp.npz"):
        try:
            if now - f.stat().st_mtime > 3600:
                f.unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue
    if removed:
        logger.info("Cleaned up {} orphaned .tmp.npz files", removed)

    if not all_files:
        return removed

    total = 0
    keep_from = len(all_files)
    for i in range(len(all_files) - 1, -1, -1):
        try:
            with np.load(all_files[i]) as data:
                total += len(data["outcomes"])
        except (OSError, ValueError, KeyError):
            continue
        if total >= cfg.replay_buffer_size:
            keep_from = i
            break

    to_remove = all_files[:keep_from]
    for f in to_remove:
        f.unlink(missing_ok=True)

    if to_remove:
        logger.info("Pruned {} old data files", len(to_remove))
    return len(to_remove) + removed


def _log_buffer_stats(prefix: str, s: dict) -> None:
    logger.info(
        "{}: {:,} pos, {} files | age: {:.0f}–{:.0f}min (median {:.0f}min) | weight: {}",
        prefix, s.get("positions", 0), s.get("files", 0),
        s.get("newest_age_min", 0), s.get("oldest_age_min", 0),
        s.get("median_age_min", 0), s.get("weight_pct_by_version", {}),
    )


def _log_event(cfg: AsyncConfig, event: dict) -> None:
    log_path = cfg.logs_dir / "trainer.jsonl"
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a") as f:
        f.write(json.dumps(event) + "\n")


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
    logger.info("Trainer starting on device: {} | half_life={:.0f}s buffer={:,} steps/cycle={}",
                device, cfg.sample_half_life, cfg.replay_buffer_size, cfg.steps_per_cycle)

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

    cycle = 0
    total_steps_all_time = 0

    while not _shutdown_requested:
        # Bootstrap: generate imitation data if no model exists yet
        if not cfg.best_model_path.exists():
            has_any_data = any(
                f for f in cfg.training_data_dir.glob("*.npz") if ".tmp" not in f.name
            )
            if not has_any_data:
                from .imitation import generate_imitation_data
                logger.info("No model found. Generating minimax imitation data...")
                generate_imitation_data(cfg)
                continue

        # Bootstrap gate: wait for minimum data before first cycle
        available = _estimate_positions(cfg.training_data_dir)
        if available < cfg.min_positions_to_start:
            logger.info("Waiting for data: ~{:,}/{:,} positions",
                        available, cfg.min_positions_to_start)
            time.sleep(30)
            continue

        cycle += 1
        cycle_t0 = time.time()
        logger.info("")
        logger.info("=" * 60)
        logger.info("Training cycle {} | model v{} | ~{:,} positions available",
                     cycle, current_version, available)
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
                              max_positions=cfg.replay_buffer_size,
                              half_life=cfg.sample_half_life)
            dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=0)
            return ds, dl

        dataset, dataloader = reload_buffer()
        if dataset.total_positions == 0:
            logger.info("No valid training data found, waiting...")
            time.sleep(30)
            continue

        buf_stats = dataset.stats()
        _log_buffer_stats("Replay buffer", buf_stats)

        # Step-based training
        model.train()
        step = 0
        cycle_policy_loss = 0.0
        cycle_value_loss = 0.0
        train_t0 = time.time()
        log_interval = 100  # log every N steps

        logger.info("Training for {} steps (batch_size={})...", cfg.steps_per_cycle, cfg.batch_size)

        while step < cfg.steps_per_cycle and not _shutdown_requested:
            for boards, policies, outcomes in dataloader:
                if _shutdown_requested or step >= cfg.steps_per_cycle:
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

                # Periodic reload: pick up fresh worker data and recompute weights
                if step % cfg.reload_interval == 0 and step < cfg.steps_per_cycle:
                    dataset, dataloader = reload_buffer()
                    _log_buffer_stats("  Reloaded buffer", dataset.stats())
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
            "initial_buffer_stats": buf_stats,
        })
        notify_training_cycle(
            cycle=cycle, version=new_version, steps=step,
            total_steps=total_steps_all_time, positions=dataset.total_positions,
            policy_loss=avg_policy, value_loss=avg_value,
            elapsed_seconds=cycle_elapsed,
        )

        _prune_old_data(cfg)

    logger.info("Trainer shutdown.")
    sys.exit(0)
