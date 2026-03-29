from __future__ import annotations
"""Continuous self-play worker for async distributed training.

Runs an infinite loop: fetch the latest model, play a game, accumulate
samples, flush to disk periodically. Polls best.meta.json between
batches to pick up newly promoted models.

Concurrency is handled at the k8s level (multiple replicas), not via
multiprocessing. Each replica runs a single process and lets ONNX
Runtime use all available cores for inference.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

from .config import AsyncConfig

try:
    import hexchess
except ImportError:
    hexchess = None


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


def _play_one_game(
    search: "hexchess.MctsSearch",
    cfg: AsyncConfig,
) -> tuple[str, list[dict]]:
    """Play a single self-play game and return training samples."""
    game = hexchess.Game()
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

    cfg.ensure_dirs()

    current_version, model_path = _read_model_version(cfg)
    batch_size = cfg.worker_batch_size
    total_games = 0
    total_positions = 0

    logger.info("Worker starting: {} games/batch, {} sims/move",
                batch_size, cfg.num_simulations)
    logger.info("Model version: v{} ({})", current_version, model_path or "random")

    search = hexchess.MctsSearch(
        simulations=cfg.num_simulations,
        model_path=model_path,
    )

    while True:
        batch_t0 = time.time()
        pending_samples: list[dict] = []
        outcome_counts: dict[str, int] = {}
        batch_games = 0

        for gi in range(batch_size):
            game_t0 = time.time()
            status, game_samples = _play_one_game(search, cfg)
            game_elapsed = time.time() - game_t0
            pending_samples.extend(game_samples)
            outcome_counts[status] = outcome_counts.get(status, 0) + 1
            batch_games += 1
            logger.info(
                "  game {}/{}: {} moves, {:.1f}s ({:.2f}s/move) | {}",
                gi + 1, batch_size, len(game_samples),
                game_elapsed, game_elapsed / max(len(game_samples), 1),
                status,
            )

        if pending_samples:
            path = _flush_samples(pending_samples, cfg.training_data_dir, current_version)
            total_games += batch_games
            total_positions += len(pending_samples)
            elapsed = time.time() - batch_t0

            logger.info(
                "Batch: {} games, {} pos, {:.0f}s ({:.1f}s/game) | "
                "total: {} games, {} pos | v{} | {}",
                batch_games, len(pending_samples),
                elapsed, elapsed / max(batch_games, 1),
                total_games, total_positions,
                current_version, path.name,
            )

            tt = search.tt_stats()
            hit_rate = tt["hits"] / max(tt["hits"] + tt["misses"], 1) * 100
            logger.info(
                "TT: {} entries, {:.0f}% hit rate ({} hits, {} misses, {} clears)",
                tt["current_size"], hit_rate,
                tt["hits"], tt["misses"], tt["clears"],
            )

            _log_event(cfg, {
                "event": "batch_complete",
                "model_version": current_version,
                "games": batch_games,
                "positions": len(pending_samples),
                "elapsed_seconds": round(elapsed, 1),
                "outcomes": outcome_counts,
                "file": path.name,
                "tt_hits": tt["hits"],
                "tt_misses": tt["misses"],
                "tt_clears": tt["clears"],
                "tt_size": tt["current_size"],
            })

        # Check for model update
        new_version, new_model_path = _read_model_version(cfg)
        if new_version > current_version:
            logger.info("Model updated: v{} -> v{}", current_version, new_version)
            current_version = new_version
            model_path = new_model_path
            search = hexchess.MctsSearch(
                simulations=cfg.num_simulations,
                model_path=model_path,
            )
