from __future__ import annotations
"""Continuous trainer for async distributed training.

Runs an infinite loop: collect recent training data from the sliding window,
train for N steps, export candidate, run arena inline, promote or discard,
prune old data, repeat.
"""

import json
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

from .arena import play_arena_game
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


class SlidingWindowBuffer(IterableDataset):
    """Streams training data from the flat training_data/ directory.

    Selects the most recent .npz files up to max_positions.
    """

    SHUFFLE_BUFFER_SIZE = 10_000

    def __init__(self, data_dir: Path, max_positions: int = 500_000):
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

    def __iter__(self):
        files = self.files.copy()
        random.shuffle(files)

        buf_b, buf_p, buf_o = [], [], []
        positions_remaining = self.max_positions

        for f in files:
            try:
                data = np.load(f)
                boards, policies, outcomes = data["boards"], data["policies"], data["outcomes"]
            except (OSError, ValueError, KeyError):
                continue

            if len(boards) > positions_remaining:
                boards = boards[-positions_remaining:]
                policies = policies[-positions_remaining:]
                outcomes = outcomes[-positions_remaining:]
            positions_remaining -= len(boards)

            buf_b.extend(boards)
            buf_p.extend(policies)
            buf_o.extend(outcomes)

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

            if positions_remaining <= 0:
                break

        # Flush remaining
        perm = list(range(len(buf_b)))
        random.shuffle(perm)
        for j in perm:
            yield (torch.from_numpy(buf_b[j]),
                   torch.from_numpy(buf_p[j]),
                   torch.tensor(buf_o[j], dtype=torch.float32))


def _count_available_positions(data_dir: Path) -> int:
    """Estimate positions available by counting .npz files and sampling size."""
    npz_files = [f for f in data_dir.glob("*.npz") if ".tmp" not in f.name]
    if not npz_files:
        return 0
    # Sample up to 3 files to estimate average positions per file
    sample = npz_files[-min(3, len(npz_files)):]
    total_sampled = 0
    for f in sample:
        try:
            with np.load(f) as data:
                total_sampled += len(data["outcomes"])
        except (OSError, ValueError, KeyError):
            continue
    if not sample:
        return 0
    avg_per_file = total_sampled / len(sample)
    return int(avg_per_file * len(npz_files))


def _run_arena_inline(cfg: AsyncConfig) -> dict:
    """Run arena: candidate vs best, return results dict."""
    new_path = str(cfg.candidate_model_path) if cfg.candidate_model_path.exists() else None
    old_path = str(cfg.best_model_path) if cfg.best_model_path.exists() else None

    logger.info("Arena: {} games, {} sims/move",
                cfg.arena_games, cfg.arena_simulations)

    new_wins = old_wins = draws = 0
    t0 = time.time()
    last_log_time = t0

    for i in range(cfg.arena_games):
        new_goes_first = i % 2 == 0
        result = play_arena_game(
            cfg.arena_simulations, new_goes_first, new_path, old_path,
        )
        game_num = i + 1

        if result == "new":
            new_wins += 1
        elif result == "old":
            old_wins += 1
        else:
            draws += 1

        now = time.time()
        total_decided = new_wins + old_wins
        rate = new_wins / total_decided if total_decided > 0 else 0.5
        is_last = game_num == cfg.arena_games
        if game_num == 1 or is_last or (now - last_log_time) >= 15:
            last_log_time = now
            elapsed = now - t0
            logger.info("  arena {}/{} (new={} old={} draw={} rate={:.0%}) {:.0f}s",
                        game_num, cfg.arena_games, new_wins, old_wins, draws, rate, elapsed)

    total_decided = new_wins + old_wins
    win_rate = new_wins / total_decided if total_decided > 0 else 0.5
    promoted = win_rate >= cfg.win_threshold

    elapsed = time.time() - t0
    verdict = "PROMOTED" if promoted else "kept current"
    logger.info("Arena done: new={} old={} draw={} rate={:.0%} ({:.0f}s) -> {}",
                new_wins, old_wins, draws, win_rate, elapsed, verdict)

    return {
        "new_wins": new_wins,
        "old_wins": old_wins,
        "draws": draws,
        "win_rate": win_rate,
        "promoted": promoted,
    }


def _prune_old_data(cfg: AsyncConfig) -> int:
    """Remove old .npz files beyond the sliding window and stale .tmp files."""
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
    logger.info("Trainer starting on device: {}", device)

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
        # Wait for enough training data
        available = _count_available_positions(cfg.training_data_dir)
        if available < cfg.min_positions_to_train:
            logger.info("Waiting for data: ~{}/{} positions", available, cfg.min_positions_to_train)
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
        dataset = SlidingWindowBuffer(cfg.training_data_dir, max_positions=cfg.replay_buffer_size)
        if dataset.total_positions == 0:
            logger.info("No valid training data found, waiting...")
            time.sleep(30)
            continue

        logger.info("Sliding window: {:,} positions from {} files",
                     dataset.total_positions, len(dataset.files))

        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0)

        # Step-based training
        model.train()
        step = 0
        cycle_policy_loss = 0.0
        cycle_value_loss = 0.0
        train_t0 = time.time()
        log_interval = 100  # log every N steps

        logger.info("Training for {} steps (batch_size={})...", cfg.steps_per_cycle, cfg.batch_size)

        training_done = False
        data_passes = 0
        while not training_done and not _shutdown_requested:
            data_passes += 1
            for boards, policies, outcomes in dataloader:
                if _shutdown_requested or step >= cfg.steps_per_cycle:
                    training_done = True
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

            if not training_done:
                # Dataset exhausted before reaching steps_per_cycle — reload
                # with latest data (workers may have produced more)
                dataset = SlidingWindowBuffer(cfg.training_data_dir, max_positions=cfg.replay_buffer_size)
                dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0)

        train_elapsed = time.time() - train_t0
        avg_policy = cycle_policy_loss / max(step, 1)
        avg_value = cycle_value_loss / max(step, 1)
        logger.info(
            "Training complete: {} steps in {:.0f}s ({} data pass{}) | policy={:.4f} value={:.4f}",
            step, train_elapsed, data_passes, "es" if data_passes > 1 else "",
            avg_policy, avg_value,
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

        # First model ever — auto-promote
        if not cfg.best_model_path.exists():
            new_version = current_version + 1
            _atomic_copy(candidate_onnx, cfg.best_model_path)
            _atomic_copy(candidate_pt, cfg.best_checkpoint_path)
            _atomic_write_json(cfg.best_meta_path, {
                "version": new_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "auto_promoted": True,
            })
            current_version = new_version
            cycle_elapsed = time.time() - cycle_t0
            logger.info("Auto-promoted first model as v{} (cycle took {:.0f}s)",
                        new_version, cycle_elapsed)

            _log_event(cfg, {
                "event": "promotion",
                "version": new_version,
                "auto": True,
                "steps": step,
                "total_steps": total_steps_all_time,
                "policy_loss": round(avg_policy, 4),
                "value_loss": round(avg_value, 4),
                "elapsed_seconds": round(cycle_elapsed, 1),
            })
            notify_training_cycle(
                cycle=cycle, version=new_version, steps=step,
                total_steps=total_steps_all_time, positions=dataset.total_positions,
                policy_loss=avg_policy, value_loss=avg_value,
                promoted=True, elapsed_seconds=cycle_elapsed,
            )
            _prune_old_data(cfg)
            continue

        # Arena evaluation
        arena_results = _run_arena_inline(cfg)

        if arena_results["promoted"]:
            new_version = current_version + 1
            _atomic_copy(candidate_onnx, cfg.best_model_path)
            _atomic_copy(candidate_pt, cfg.best_checkpoint_path)
            _atomic_write_json(cfg.best_meta_path, {
                "version": new_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "win_rate": arena_results["win_rate"],
            })
            current_version = new_version
            logger.info("Promoted to v{} (win rate: {:.0%})", new_version, arena_results["win_rate"])
        else:
            # Reload the best checkpoint since our candidate didn't win
            logger.warning("Not promoted (win rate: {:.0%}). Reverting to v{}.",
                           arena_results["win_rate"], current_version)
            model.load_state_dict(torch.load(cfg.best_checkpoint_path, map_location=device, weights_only=True))
            optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.l2_regularization)

        cycle_elapsed = time.time() - cycle_t0
        logger.info("Cycle {} done in {:.0f}s | total steps: {:,}",
                     cycle, cycle_elapsed, total_steps_all_time)

        _log_event(cfg, {
            "event": "cycle_complete",
            "cycle": cycle,
            "version": current_version,
            "promoted": arena_results["promoted"],
            "win_rate": arena_results["win_rate"],
            "steps": step,
            "total_steps": total_steps_all_time,
            "positions": dataset.total_positions,
            "policy_loss": round(avg_policy, 4),
            "value_loss": round(avg_value, 4),
            "elapsed_seconds": round(cycle_elapsed, 1),
        })
        notify_training_cycle(
            cycle=cycle, version=current_version, steps=step,
            total_steps=total_steps_all_time, positions=dataset.total_positions,
            policy_loss=avg_policy, value_loss=avg_value,
            promoted=arena_results["promoted"],
            win_rate=arena_results["win_rate"],
            elapsed_seconds=cycle_elapsed,
        )

        # Prune old data
        _prune_old_data(cfg)

    logger.info("Trainer shutdown.")
    sys.exit(0)
