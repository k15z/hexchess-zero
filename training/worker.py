"""Continuous self-play worker for async distributed training.

Runs an infinite loop: fetch the latest model from S3, play games,
flush training data to S3, repeat. Polls for newly promoted models
between batches.

Concurrency: each worker process lets ONNX Runtime use all cores.
For imitation (minimax), parallelizes across cores via ProcessPoolExecutor.
"""

from __future__ import annotations

import json
import os
import platform
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone

import numpy as np
from loguru import logger

from . import storage
from .config import AsyncConfig
from .imitation import play_imitation_game

try:
    import hexchess
except ImportError:
    hexchess = None


def _worker_name() -> str:
    return os.environ.get("WORKER_NAME", platform.node())


def _read_model_version(cfg: AsyncConfig) -> tuple[int, str | None]:
    """Read the current model version from S3.

    Returns (version, local_model_path). version=0 and path=None if no model
    exists. Only downloads the ONNX model if the version has changed.
    """
    try:
        meta = storage.get_json(storage.LATEST_META)
        version = meta.get("version", 0)
    except KeyError:
        return 0, None

    local_path = cfg.model_cache_dir / "latest.onnx"
    local_meta = cfg.model_cache_dir / "latest.meta.json"

    if local_meta.exists() and local_path.exists():
        cached = json.loads(local_meta.read_text())
        if cached.get("version") == version:
            return version, str(local_path)

    storage.get_file(storage.LATEST_ONNX, local_path)
    local_meta.write_text(json.dumps(meta))
    return version, str(local_path)


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

        result = search.run(game, temperature=temperature)

        side = game.side_to_move()
        samples.append({
            "board": board_tensor,
            "policy": result.policy,
            "side": side,
        })

        game.apply(result.best_move)
        move_number += 1

    status = game.status()
    if status == "checkmate_white":
        wdl_white = [1.0, 0.0, 0.0]
    elif status == "checkmate_black":
        wdl_white = [0.0, 0.0, 1.0]
    else:
        wdl_white = [0.0, 1.0, 0.0]

    for sample in samples:
        if sample["side"] == "white":
            sample["outcome"] = np.array(wdl_white, dtype=np.float32)
        else:
            sample["outcome"] = np.array([wdl_white[2], wdl_white[1], wdl_white[0]], dtype=np.float32)
        del sample["side"]

    return status, samples


def _write_heartbeat(cfg: AsyncConfig, version: int, total_games: int,
                     total_positions: int) -> None:
    storage.put_json(f"{storage.HEARTBEATS_PREFIX}{_worker_name()}.json", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": version,
        "total_games": total_games,
        "total_positions": total_positions,
    })


def run_worker(cfg: AsyncConfig) -> None:
    """Run the continuous self-play worker loop."""
    if hexchess is None:
        raise ImportError("hexchess bindings not available")

    # Ensure child processes are killed when the worker is terminated.
    def _cleanup(signum, frame):
        logger.info("Received signal {}, terminating child processes...", signum)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        os.killpg(0, signal.SIGTERM)
        sys.exit(1)

    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)
    # Put this process in its own process group so killpg only affects us + children.
    try:
        os.setpgrp()
    except OSError:
        pass

    cfg.ensure_cache_dirs()

    current_version, model_path = _read_model_version(cfg)

    # Bootstrap: generate imitation data from minimax until a model appears.
    # Minimax is single-threaded, so we parallelize across cores. Games are
    # flushed every worker_batch_size completions to keep file sizes consistent
    # with self-play and avoid holding data in memory waiting for stragglers.
    if model_path is None:
        num_workers = max(1, os.cpu_count() or 1)
        logger.info("No model found — generating minimax imitation data "
                     "(depth {}, {} parallel workers)", cfg.imitation_depth, num_workers)
        imitation_games = 0
        imitation_positions = 0
        batch_size = cfg.worker_batch_size
        pending_samples: list[dict] = []
        pending_games = 0
        batch_t0 = time.time()

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            # Keep num_workers * 2 games in flight so the pool stays busy
            futures = {pool.submit(play_imitation_game, cfg) for _ in range(num_workers * 2)}

            while model_path is None:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    samples = future.result()
                    pending_samples.extend(samples)
                    pending_games += 1
                    imitation_games += 1
                    imitation_positions += len(samples)
                    logger.info("  game {} complete: {} positions",
                                imitation_games, len(samples))

                    futures.add(pool.submit(play_imitation_game, cfg))

                    if pending_games >= batch_size:
                        batch_elapsed = time.time() - batch_t0
                        key = storage.flush_samples(
                            pending_samples, storage.IMITATION_PREFIX)
                        logger.info(
                            "Imitation batch: {} games, {} pos, {:.0f}s ({:.1f}s/game) | "
                            "total: {} games, {} pos | {}",
                            pending_games, len(pending_samples), batch_elapsed,
                            batch_elapsed / pending_games, imitation_games,
                            imitation_positions, key,
                        )
                        _write_heartbeat(cfg, 0, imitation_games, imitation_positions)
                        pending_samples = []
                        pending_games = 0
                        batch_t0 = time.time()

                        current_version, model_path = _read_model_version(cfg)

            for f in futures:
                f.cancel()

        if pending_samples:
            storage.flush_samples(pending_samples, storage.IMITATION_PREFIX)

        logger.info("Model appeared (v{}), switching to self-play", current_version)

    batch_size = cfg.worker_batch_size
    total_games = 0
    total_positions = 0

    logger.info("Worker starting self-play: {} games/batch, {} sims/move",
                batch_size, cfg.num_simulations)
    logger.info("Model version: v{} ({})", current_version, model_path)

    search = hexchess.MctsSearch(
        simulations=cfg.num_simulations,
        model_path=model_path,
        dirichlet_epsilon=cfg.dirichlet_epsilon,
        dirichlet_alpha=cfg.dirichlet_alpha,
    )

    while True:
        batch_t0 = time.time()
        pending_samples: list[dict] = []
        batch_games = 0

        for gi in range(batch_size):
            game_t0 = time.time()
            status, game_samples = _play_one_game(search, cfg)
            game_elapsed = time.time() - game_t0
            pending_samples.extend(game_samples)
            batch_games += 1
            logger.info(
                "  game {}/{}: {} moves, {:.1f}s ({:.2f}s/move) | {}",
                gi + 1, batch_size, len(game_samples),
                game_elapsed, game_elapsed / max(len(game_samples), 1),
                status,
            )

        if pending_samples:
            key = storage.flush_samples(
                pending_samples,
                f"{storage.SELFPLAY_PREFIX}v{current_version}/",
            )
            total_games += batch_games
            total_positions += len(pending_samples)
            elapsed = time.time() - batch_t0

            logger.info(
                "Batch: {} games, {} pos, {:.0f}s ({:.1f}s/game) | "
                "total: {} games, {} pos | v{} | {}",
                batch_games, len(pending_samples),
                elapsed, elapsed / max(batch_games, 1),
                total_games, total_positions,
                current_version, key,
            )

            tt = search.tt_stats()
            hit_rate = tt.hits / max(tt.hits + tt.misses, 1) * 100
            logger.info(
                "TT: {} entries, {:.0f}% hit rate ({} hits, {} misses, {} clears)",
                tt.current_size, hit_rate,
                tt.hits, tt.misses, tt.clears,
            )

            _write_heartbeat(cfg, current_version, total_games, total_positions)

        new_version, new_model_path = _read_model_version(cfg)
        if new_version > current_version:
            logger.info("Model updated: v{} -> v{}", current_version, new_version)
            current_version = new_version
            model_path = new_model_path
            search = hexchess.MctsSearch(
                simulations=cfg.num_simulations,
                model_path=model_path,
                dirichlet_epsilon=cfg.dirichlet_epsilon,
                dirichlet_alpha=cfg.dirichlet_alpha,
            )
