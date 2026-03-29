from __future__ import annotations
"""Continuous trainer for async distributed training.

Runs an infinite loop: collect recent training data from the sliding window,
train for N steps, export candidate, run arena inline, promote or discard,
prune old data, repeat.
"""

import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

from .arena import play_arena_game, _arena_game_worker
from .config import AsyncConfig
from .export import export_to_onnx
from .model import build_model

try:
    import hexchess
except ImportError:
    hexchess = None

_shutdown_requested = False


def _handle_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print(f"Shutdown requested (signal {signum}), finishing current cycle...", flush=True)


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
    from multiprocessing import Pool

    new_path = str(cfg.candidate_model_path) if cfg.candidate_model_path.exists() else None
    old_path = str(cfg.best_model_path) if cfg.best_model_path.exists() else None

    print(f"Arena: {cfg.arena_games} games, {cfg.arena_simulations} sims/move, "
          f"{cfg.num_arena_workers} workers", flush=True)

    new_wins = old_wins = draws = 0
    t0 = time.time()
    last_log_time = t0

    args = [
        (cfg.arena_simulations, i % 2 == 0, new_path, old_path)
        for i in range(cfg.arena_games)
    ]

    def _tally(result: str, game_num: int) -> None:
        nonlocal new_wins, old_wins, draws, last_log_time
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
            print(
                f"  arena {game_num}/{cfg.arena_games} "
                f"(new={new_wins} old={old_wins} draw={draws} "
                f"rate={rate:.0%}) {elapsed:.0f}s",
                flush=True,
            )

    workers = cfg.num_arena_workers
    if workers > 1:
        with Pool(processes=workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_arena_game_worker, args)):
                _tally(result, i + 1)
    else:
        for i, a in enumerate(args):
            result = _arena_game_worker(a)
            _tally(result, i + 1)

    total_decided = new_wins + old_wins
    win_rate = new_wins / total_decided if total_decided > 0 else 0.5
    promoted = win_rate >= cfg.win_threshold

    elapsed = time.time() - t0
    verdict = "PROMOTED" if promoted else "kept current"
    print(
        f"Arena done: new={new_wins} old={old_wins} draw={draws} "
        f"rate={win_rate:.0%} ({elapsed:.0f}s) -> {verdict}",
        flush=True,
    )

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
        print(f"Cleaned up {removed} orphaned .tmp.npz files", flush=True)

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
        print(f"Pruned {len(to_remove)} old data files", flush=True)
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
    print(f"Trainer starting on device: {device}", flush=True)

    current_version = _read_model_version(cfg)

    # Persistent model and optimizer across cycles
    model = build_model(cfg).to(device)
    if cfg.best_checkpoint_path.exists():
        print(f"Loading checkpoint: {cfg.best_checkpoint_path}", flush=True)
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
            print(
                f"Waiting for data: ~{available}/{cfg.min_positions_to_train} positions",
                flush=True,
            )
            time.sleep(30)
            continue

        cycle += 1
        cycle_t0 = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f"Training cycle {cycle} | model v{current_version} | "
              f"~{available:,} positions available", flush=True)
        print(f"{'='*60}", flush=True)

        # Check if model was promoted externally (shouldn't happen with single
        # trainer, but defensive)
        disk_version = _read_model_version(cfg)
        if disk_version > current_version and cfg.best_checkpoint_path.exists():
            print(f"Model updated externally: v{current_version} -> v{disk_version}, reloading", flush=True)
            model.load_state_dict(torch.load(cfg.best_checkpoint_path, map_location=device, weights_only=True))
            optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.l2_regularization)
            current_version = disk_version

        # Load data
        dataset = SlidingWindowBuffer(cfg.training_data_dir, max_positions=cfg.replay_buffer_size)
        if dataset.total_positions == 0:
            print("No valid training data found, waiting...", flush=True)
            time.sleep(30)
            continue

        print(f"Sliding window: {dataset.total_positions:,} positions from "
              f"{len(dataset.files)} files", flush=True)

        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0)

        # Step-based training
        model.train()
        step = 0
        cycle_policy_loss = 0.0
        cycle_value_loss = 0.0
        train_t0 = time.time()
        log_interval = 100  # log every N steps

        print(f"Training for {cfg.steps_per_cycle} steps (batch_size={cfg.batch_size})...", flush=True)

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
                    print(
                        f"  step {step:>5}/{cfg.steps_per_cycle} | "
                        f"loss: policy={avg_p:.4f} value={avg_v:.4f} total={avg_p+avg_v:.4f} | "
                        f"{steps_per_sec:.1f} steps/s | "
                        f"{elapsed:.0f}s elapsed",
                        flush=True,
                    )

            if not training_done:
                # Dataset exhausted before reaching steps_per_cycle — reload
                # with latest data (workers may have produced more)
                dataset = SlidingWindowBuffer(cfg.training_data_dir, max_positions=cfg.replay_buffer_size)
                dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0)

        train_elapsed = time.time() - train_t0
        avg_policy = cycle_policy_loss / max(step, 1)
        avg_value = cycle_value_loss / max(step, 1)
        print(
            f"Training complete: {step} steps in {train_elapsed:.0f}s "
            f"({data_passes} data pass{'es' if data_passes > 1 else ''}) | "
            f"policy={avg_policy:.4f} value={avg_value:.4f}",
            flush=True,
        )

        if _shutdown_requested:
            torch.save(model.state_dict(), cfg.candidate_checkpoint_path)
            print("Saved checkpoint before shutdown.", flush=True)
            break

        # Export candidate
        print("Exporting candidate ONNX model...", flush=True)
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
            print(f"Auto-promoted first model as v{new_version} "
                  f"(cycle took {cycle_elapsed:.0f}s)", flush=True)

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
            print(f"Promoted to v{new_version} "
                  f"(win rate: {arena_results['win_rate']:.0%})", flush=True)
        else:
            # Reload the best checkpoint since our candidate didn't win
            print(f"Not promoted (win rate: {arena_results['win_rate']:.0%}). "
                  f"Reverting to v{current_version}.", flush=True)
            model.load_state_dict(torch.load(cfg.best_checkpoint_path, map_location=device, weights_only=True))
            optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.l2_regularization)

        cycle_elapsed = time.time() - cycle_t0
        print(f"Cycle {cycle} done in {cycle_elapsed:.0f}s | "
              f"total steps: {total_steps_all_time:,}", flush=True)

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

        # Prune old data
        _prune_old_data(cfg)

    print("Trainer shutdown.", flush=True)
    sys.exit(0)
