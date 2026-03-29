from __future__ import annotations
"""Continuous self-play worker for async distributed training.

Runs an infinite loop: fetch the latest model, play a batch of games,
write training data, repeat. Polls best.meta.json between batches to
pick up newly promoted models.
"""

import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .config import AsyncConfig

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

try:
    import hexchess
except ImportError:
    hexchess = None

def _handle_signal(signum, frame):
    print(f"Shutdown requested (signal {signum}), exiting immediately.", flush=True)
    os._exit(0)


def _read_model_version(cfg: AsyncConfig) -> tuple[int, str | None]:
    """Read the current best model version and path.

    Returns (version, model_path). version=0 and model_path=None if no model exists yet.
    """
    if cfg.best_meta_path.exists():
        try:
            meta = json.loads(cfg.best_meta_path.read_text())
            version = meta.get("version", 0)
        except (json.JSONDecodeError, OSError):
            version = 0
    else:
        version = 0

    model_path = str(cfg.best_model_path) if cfg.best_model_path.exists() else None
    return version, model_path


def _play_one_game(args: tuple) -> tuple[str, list[dict]]:
    """Worker function: play a single self-play game."""
    cfg, model_path = args
    if hexchess is None:
        raise ImportError("hexchess bindings not available")

    game = hexchess.Game()
    search = hexchess.MctsSearch(
        simulations=cfg.num_simulations,
        model_path=model_path,
    )

    samples = []
    move_number = 0

    while not game.is_game_over():
        board_tensor = hexchess.encode_board(game)

        if move_number < cfg.temperature_threshold:
            temperature = cfg.temperature_high
        else:
            temperature = cfg.temperature_low

        result = search.run(
            game,
            temperature=temperature,
            dirichlet_epsilon=cfg.dirichlet_epsilon,
            dirichlet_alpha=cfg.dirichlet_alpha,
        )

        side = game.side_to_move()
        samples.append({
            "board": board_tensor,
            "policy": result["policy"],
            "side": side,
        })

        best = result["best_move"]
        game.apply_move(
            best["from_q"], best["from_r"],
            best["to_q"], best["to_r"],
            best.get("promotion"),
        )
        move_number += 1

    # Determine game outcome as WDL targets [win, draw, loss]
    status = game.status()
    if status == "checkmate_white":
        wdl_white = [1.0, 0.0, 0.0]  # white won
    elif status == "checkmate_black":
        wdl_white = [0.0, 0.0, 1.0]  # white lost
    else:
        wdl_white = [0.0, 1.0, 0.0]  # draw

    # Fill in WDL outcome from each side's perspective
    for sample in samples:
        if sample["side"] == "white":
            sample["outcome"] = np.array(wdl_white, dtype=np.float32)
        else:
            # Flip W and L for black's perspective
            sample["outcome"] = np.array([wdl_white[2], wdl_white[1], wdl_white[0]], dtype=np.float32)
        del sample["side"]

    return status, samples


def _flush_samples(samples: list[dict], data_dir: Path, model_version: int) -> Path:
    """Write samples to a version-tagged .npz file."""
    boards = np.stack([s["board"] for s in samples])
    policies = np.stack([s["policy"] for s in samples])
    outcomes = np.array([s["outcome"] for s in samples], dtype=np.float32)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = np.random.default_rng().integers(0, 0xFFFF_FFFF)
    basename = f"sp_v{model_version}_{ts}_{suffix:08x}"

    # np.savez_compressed auto-appends .npz to the path.
    # Write to a temp name then rename for atomicity on NFS.
    tmp_path = data_dir / (basename + ".tmp")
    np.savez_compressed(tmp_path, boards=boards, policies=policies, outcomes=outcomes)
    # Actual file on disk is basename.tmp.npz — rename to final
    save_path = data_dir / (basename + ".npz")
    (data_dir / (basename + ".tmp.npz")).rename(save_path)

    return save_path


def _log_event(cfg: AsyncConfig, event: dict) -> None:
    """Append a JSON event to the worker log file."""
    hostname = os.environ.get("HOSTNAME", os.environ.get("POD_NAME", "local"))
    log_path = cfg.logs_dir / f"worker-{hostname}.jsonl"
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a") as f:
        f.write(json.dumps(event) + "\n")


def run_worker(cfg: AsyncConfig) -> None:
    """Run the continuous self-play worker loop."""
    if hexchess is None:
        raise ImportError("hexchess bindings not available. Run `maturin develop` in bindings/python/")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    cfg.ensure_dirs()

    current_version, model_path = _read_model_version(cfg)
    workers = cfg.num_self_play_workers
    batch_size = cfg.worker_batch_size
    total_games = 0
    total_positions = 0

    print(f"Worker starting: {workers} processes, {batch_size} games/batch, "
          f"{cfg.num_simulations} sims/move", flush=True)
    print(f"Model version: v{current_version} ({model_path or 'random'})", flush=True)

    while True:
        batch_t0 = time.time()
        pending_samples: list[dict] = []
        outcome_counts: dict[str, int] = {}
        batch_games = 0

        args = [(cfg, model_path) for _ in range(batch_size)]

        if workers > 1:
            with Pool(processes=workers) as pool:
                for result in pool.imap_unordered(_play_one_game, args):
                    status, game_samples = result
                    pending_samples.extend(game_samples)
                    outcome_counts[status] = outcome_counts.get(status, 0) + 1
                    batch_games += 1
        else:
            for a in args:
                status, game_samples = _play_one_game(a)
                pending_samples.extend(game_samples)
                outcome_counts[status] = outcome_counts.get(status, 0) + 1
                batch_games += 1

        if pending_samples:
            path = _flush_samples(pending_samples, cfg.training_data_dir, current_version)
            total_games += batch_games
            total_positions += len(pending_samples)
            elapsed = time.time() - batch_t0

            print(
                f"Batch: {batch_games} games, {len(pending_samples)} pos, "
                f"{elapsed:.0f}s ({elapsed/max(batch_games,1):.1f}s/game) | "
                f"total: {total_games} games, {total_positions} pos | "
                f"v{current_version} | {path.name}",
                flush=True,
            )

            _log_event(cfg, {
                "event": "batch_complete",
                "model_version": current_version,
                "games": batch_games,
                "positions": len(pending_samples),
                "elapsed_seconds": round(elapsed, 1),
                "outcomes": outcome_counts,
                "file": path.name,
            })

        # Check for model update
        new_version, new_model_path = _read_model_version(cfg)
        if new_version > current_version:
            print(f"Model updated: v{current_version} -> v{new_version}", flush=True)
            current_version = new_version
            model_path = new_model_path

