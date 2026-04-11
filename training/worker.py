"""Continuous self-play worker for async distributed training (v2 schema).

Runs an infinite loop: fetch the latest model from S3, play self-play games
using the Rust MCTS engine's Playout-Cap-Randomization (PCR) API, buffer the
full-search positions of each game, fill per-game targets (MLH, short-horizon
WDL), and flush a rich `.npz` + metadata sidecar + per-game trace to S3.

Schema (see notes/13-implementation-plan.md §5.3):
    boards                (N, 22, 11, 11)        int8
    policy                (N, num_move_indices)  float16
    policy_aux_opp        (N, num_move_indices)  float16
    wdl_terminal          (N, 3)                 float32
    wdl_short             (N, 3)                 float32
    mlh                   (N,)                   int16
    was_full_search       (N,)                   bool
    root_q                (N,)                   float16
    root_n                (N,)                   int32
    root_entropy          (N,)                   float16
    nn_value_at_position  (N,)                   float16
    legal_count           (N,)                   int16
    ply                   (N,)                   int16
    game_id               (N,)                   uint64

Only full-search PCR positions (was_full_search=True) are kept as samples.
The `boards` int8 cast is lossless for the binary piece planes; the four
normalized meta planes (fullmove, halfmove clock, repetition, validity/other
scalar channels) are rounded to the nearest integer in int8 range. This is a
small precision loss the trainer will compensate for by rescaling on load.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from . import storage
from .config import AsyncConfig
from .imitation import play_imitation_game
from .logging_setup import log_event, setup_json_logging

try:
    import hexchess
except ImportError:
    hexchess = None


# ---------------------------------------------------------------------------
# In-memory game record types
# ---------------------------------------------------------------------------

@dataclass
class PositionSample:
    """One full-search position to be emitted as a training sample."""
    board: np.ndarray            # (22, 11, 11) float32 from encode_board
    policy: np.ndarray           # (num_moves,) float32 (PTP-pruned)
    policy_aux_opp: np.ndarray   # (num_moves,) float32 opponent-reply visit dist
    root_q: float                # [-1, 1] from STM perspective
    root_n: int                  # total visits at root
    root_entropy: float
    nn_value_at_position: float  # raw network value prior (currently = root_q fallback)
    legal_count: int
    ply: int                     # half-move number when this position was reached
    side: str                    # "white" | "black"
    was_full_search: bool = True

    # Per-ply trace entry (recorded at emit time; opaque dict).
    trace: dict = field(default_factory=dict)


@dataclass
class GameRecord:
    """A finished self-play game."""
    positions: list[PositionSample]
    game_id: int
    model_version: int
    started_at: str
    duration_s: float
    result: str                  # "white_win" | "draw" | "black_win"
    termination: str             # "checkmate" | "stalemate" | "threefold" | "50move" | ...
    resigned_skipped: bool
    opening_hash: str
    rng_seed: int
    dirichlet_epsilon: float
    dirichlet_alpha: float
    num_simulations: int
    worker: str
    git_sha: str
    num_total_positions: int     # full + fast combined
    wdl_terminal_white: list[float]  # [W, D, L] from WHITE perspective
    game_len_plies: int


# ---------------------------------------------------------------------------
# Pure helpers (testable without the Rust binding)
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()[:12]
    except Exception:
        return "unknown"


def _worker_name() -> str:
    return os.environ.get("WORKER_NAME", platform.node())


def _status_to_result_termination(status: str) -> tuple[str, str, list[float]]:
    """Map a Rust `Game.status()` string to (result, termination, wdl_white)."""
    if status == "checkmate_white":
        return "white_win", "checkmate", [1.0, 0.0, 0.0]
    if status == "checkmate_black":
        return "black_win", "checkmate", [0.0, 0.0, 1.0]
    if status == "stalemate":
        return "draw", "stalemate", [0.0, 1.0, 0.0]
    if status == "draw_repetition":
        return "draw", "threefold", [0.0, 1.0, 0.0]
    if status == "draw_fifty":
        return "draw", "50move", [0.0, 1.0, 0.0]
    if status == "draw_material":
        return "draw", "insufficient_material", [0.0, 1.0, 0.0]
    return "draw", status or "movelimit", [0.0, 1.0, 0.0]


def _stm_wdl(wdl_white: list[float], side: str) -> list[float]:
    if side == "white":
        return list(wdl_white)
    return [wdl_white[2], wdl_white[1], wdl_white[0]]


def _root_q_to_wdl(q: float) -> list[float]:
    """Approximate a scalar root Q in [-1,1] as a WDL distribution.

    W = max(q, 0), L = max(-q, 0), D = 1 - |q|.

    Documented approximation (plan §5.3 `wdl_short` fallback). The network
    outputs proper WDL logits during chunk 3+, but `run_pcr`'s SearchResult
    only gives us the scalar root Q right now — this three-point mass is the
    closest faithful re-projection. Chunk 13 can thread the actual WDL through
    the binding if a sharper signal is wanted.
    """
    q = float(max(-1.0, min(1.0, q)))
    win = max(q, 0.0)
    loss = max(-q, 0.0)
    draw = max(0.0, 1.0 - abs(q))
    s = win + draw + loss
    if s > 0:
        win, draw, loss = win / s, draw / s, loss / s
    return [win, draw, loss]


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    mask = p > 0
    if not mask.any():
        return 0.0
    return float(-(p[mask] * np.log(p[mask])).sum())


def compute_opening_hash(move_strings: list[str]) -> str:
    """SHA1 of the concatenated first-6-ply move strings."""
    first_six = move_strings[:6]
    joined = "|".join(first_six)
    return hashlib.sha1(joined.encode()).hexdigest()[:16]


def finalize_game_targets(record: GameRecord) -> None:
    """In-place fill per-position `mlh` and `wdl_short` on a completed game.

    Also stamps each position with `wdl_terminal_stm`. Call exactly once at
    game end, when `record.game_len_plies` and `record.wdl_terminal_white`
    are known.
    """
    game_len = record.game_len_plies
    # Index positions by their ply for horizon lookup.
    by_ply: dict[int, PositionSample] = {p.ply: p for p in record.positions}
    sorted_plies = sorted(by_ply.keys())

    for pos in record.positions:
        # --- mlh ---
        mlh = max(0, game_len - pos.ply)
        pos.trace["mlh"] = mlh
        pos.trace["wdl_terminal_stm"] = _stm_wdl(record.wdl_terminal_white, pos.side)

        # --- wdl_short: horizon 8 plies ---
        target_ply = pos.ply + 8
        if target_ply >= game_len:
            # Terminal within horizon — use terminal WDL from STM POV.
            pos.trace["wdl_short_stm"] = _stm_wdl(record.wdl_terminal_white, pos.side)
        else:
            # Non-terminal: approximate from root Q at the nearest future
            # full-search position (fallback to this position's own root_q).
            future: PositionSample | None = None
            for p in sorted_plies:
                if p >= target_ply:
                    future = by_ply[p]
                    break
            q = future.root_q if future is not None else pos.root_q
            # `q` is STM-relative at the future position; flip if STMs differ.
            if future is not None and future.side != pos.side:
                q = -q
            pos.trace["wdl_short_stm"] = _root_q_to_wdl(q)
        pos.trace["mlh_ply"] = pos.ply
        pos.trace["game_len"] = game_len


# ---------------------------------------------------------------------------
# NPZ + sidecar writers
# ---------------------------------------------------------------------------

def write_samples_to_npz(
    npz_path: str | Path,
    record: GameRecord,
    *,
    num_move_indices: int | None = None,
) -> dict[str, Any]:
    """Materialize a GameRecord to disk as .npz + .meta.json sidecar.

    Writes two files:
      - `{npz_path}`                — compressed NPZ with the v2 schema.
      - `{npz_path with .meta.json}` — per-game metadata sidecar.

    Returns the metadata dict.

    Pure I/O; no S3, no binding calls. Safe to unit-test by passing a
    hand-constructed GameRecord.
    """
    npz_path = Path(npz_path)
    if num_move_indices is None:
        if not record.positions:
            raise ValueError("cannot infer num_move_indices from empty record")
        num_move_indices = int(record.positions[0].policy.shape[0])

    n = len(record.positions)
    if n == 0:
        raise ValueError("refusing to write empty npz")

    boards = np.zeros((n, 22, 11, 11), dtype=np.int8)
    policy = np.zeros((n, num_move_indices), dtype=np.float16)
    policy_aux_opp = np.zeros((n, num_move_indices), dtype=np.float16)
    wdl_terminal = np.zeros((n, 3), dtype=np.float32)
    wdl_short = np.zeros((n, 3), dtype=np.float32)
    mlh = np.zeros((n,), dtype=np.int16)
    was_full_search = np.ones((n,), dtype=bool)
    root_q = np.zeros((n,), dtype=np.float16)
    root_n = np.zeros((n,), dtype=np.int32)
    root_entropy = np.zeros((n,), dtype=np.float16)
    nn_value_at_position = np.zeros((n,), dtype=np.float16)
    legal_count = np.zeros((n,), dtype=np.int16)
    ply = np.zeros((n,), dtype=np.int16)
    game_id = np.full((n,), record.game_id, dtype=np.uint64)

    for i, pos in enumerate(record.positions):
        # Cast float board to int8. Piece planes are {0,1}; scalar meta
        # planes in [0, ~127] round safely into int8 (worker.py module
        # docstring documents this).
        boards[i] = np.clip(np.rint(pos.board), -128, 127).astype(np.int8)
        policy[i] = pos.policy.astype(np.float16)
        policy_aux_opp[i] = pos.policy_aux_opp.astype(np.float16)
        wdl_terminal[i] = pos.trace.get("wdl_terminal_stm", [0.0, 1.0, 0.0])
        wdl_short[i] = pos.trace.get("wdl_short_stm", [0.0, 1.0, 0.0])
        mlh[i] = int(pos.trace.get("mlh", 0))
        was_full_search[i] = bool(pos.was_full_search)
        root_q[i] = np.float16(pos.root_q)
        root_n[i] = int(pos.root_n)
        root_entropy[i] = np.float16(pos.root_entropy)
        nn_value_at_position[i] = np.float16(pos.nn_value_at_position)
        legal_count[i] = int(pos.legal_count)
        ply[i] = int(pos.ply)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(npz_path),
        boards=boards,
        policy=policy,
        policy_aux_opp=policy_aux_opp,
        wdl_terminal=wdl_terminal,
        wdl_short=wdl_short,
        mlh=mlh,
        was_full_search=was_full_search,
        root_q=root_q,
        root_n=root_n,
        root_entropy=root_entropy,
        nn_value_at_position=nn_value_at_position,
        legal_count=legal_count,
        ply=ply,
        game_id=game_id,
    )

    meta = {
        "game_id_range": [int(record.game_id), int(record.game_id)],
        "model_version": int(record.model_version),
        "worker": record.worker,
        "started_at": record.started_at,
        "duration_s": float(record.duration_s),
        "num_full_search_positions": int(n),
        "num_total_positions": int(record.num_total_positions),
        "result": record.result,
        "termination": record.termination,
        "resigned_skipped": bool(record.resigned_skipped),
        "openings_hash": record.opening_hash,
        "git_sha": record.git_sha,
        "rng_seed": int(record.rng_seed),
        "dirichlet_epsilon": float(record.dirichlet_epsilon),
        "dirichlet_alpha": float(record.dirichlet_alpha),
        "num_simulations": int(record.num_simulations),
    }
    meta_path = npz_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def write_trace_json(path: str | Path, record: GameRecord) -> None:
    """Write a per-game trace sidecar (one entry per full-search ply)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for pos in record.positions:
        entry = {
            "ply": int(pos.ply),
            "side": pos.side,
            "root_q": float(pos.root_q),
            "root_n": int(pos.root_n),
            "root_entropy": float(pos.root_entropy),
            "nn_value_at_position": float(pos.nn_value_at_position),
            "legal_count": int(pos.legal_count),
        }
        entry.update(pos.trace)
        entries.append(entry)
    path.write_text(json.dumps({
        "game_id": int(record.game_id),
        "model_version": int(record.model_version),
        "result": record.result,
        "termination": record.termination,
        "rng_seed": int(record.rng_seed),
        "dirichlet_epsilon": float(record.dirichlet_epsilon),
        "dirichlet_alpha": float(record.dirichlet_alpha),
        "num_simulations": int(record.num_simulations),
        "entries": entries,
    }, indent=2))


# ---------------------------------------------------------------------------
# Self-play (uses hexchess binding)
# ---------------------------------------------------------------------------

def _move_to_str(mv) -> str:
    try:
        return mv.notation
    except Exception:
        return f"{mv.from_q},{mv.from_r}->{mv.to_q},{mv.to_r}"


def _play_one_game_pcr(
    search: "hexchess.MctsSearch",
    cfg: AsyncConfig,
    *,
    game_id: int,
    model_version: int,
    py_rng: random.Random,
) -> GameRecord:
    """Play one self-play game using PCR. Returns a finalized GameRecord."""
    assert hexchess is not None
    num_moves = hexchess.num_move_indices()

    # Per-game seed + resign-skip coin.
    seed = py_rng.getrandbits(64)
    search.set_rng_seed(seed)
    resigned_skipped = py_rng.random() < 0.1
    search.set_resign_enabled(not resigned_skipped)

    game = hexchess.Game()
    positions: list[PositionSample] = []
    opening_moves: list[str] = []
    total_ply = 0
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    while not game.is_game_over():
        side = game.side_to_move()
        board_tensor = hexchess.encode_board(game)  # (22, 11, 11) float32
        legal = game.legal_moves()
        legal_count = len(legal)

        outcome = search.run_pcr(game, total_ply)
        best_move = outcome["best_move"]
        was_full = bool(outcome["was_full_search"])
        value = float(outcome["value"])
        nodes = int(outcome["nodes"])
        policy_target = outcome["policy_target"]  # np.ndarray or None

        if was_full and policy_target is not None:
            policy_np = np.asarray(policy_target, dtype=np.float32)

            # Opponent-reply visit distribution from the MCTS tree. Falls
            # back to uniform-over-legal if the best child is unexpanded
            # (e.g. terminal position or very small search).
            aux_opp_raw = search.aux_opponent_policy()
            if aux_opp_raw is not None:
                aux_opp = np.asarray(aux_opp_raw, dtype=np.float32)
            else:
                aux_opp = np.zeros(num_moves, dtype=np.float32)
                if legal_count > 0:
                    inv = 1.0 / legal_count
                    for mv in legal:
                        idx = hexchess.move_to_index(
                            mv.from_q, mv.from_r, mv.to_q, mv.to_r, mv.promotion
                        )
                        aux_opp[idx] = inv

            pos = PositionSample(
                board=board_tensor.astype(np.float32),
                policy=policy_np,
                policy_aux_opp=aux_opp,
                root_q=value,
                root_n=nodes,
                root_entropy=_entropy(policy_np),
                nn_value_at_position=value,  # TODO(chunk 13): expose raw NN prior separately
                legal_count=legal_count,
                ply=total_ply,
                side=side,
                was_full_search=True,
                trace={
                    "selected_move": _move_to_str(best_move),
                    "selection_reason": "max_visits",
                    "noise_used": True,
                    "search_ms": None,
                },
            )
            positions.append(pos)

        if total_ply < 6:
            opening_moves.append(_move_to_str(best_move))

        game.apply(best_move)
        total_ply += 1

    duration = time.time() - t0
    status = game.status()
    result, termination, wdl_white = _status_to_result_termination(status)

    record = GameRecord(
        positions=positions,
        game_id=game_id,
        model_version=model_version,
        started_at=started_at,
        duration_s=duration,
        result=result,
        termination=termination,
        resigned_skipped=resigned_skipped,
        opening_hash=compute_opening_hash(opening_moves),
        rng_seed=seed,
        dirichlet_epsilon=cfg.dirichlet_epsilon,
        dirichlet_alpha=cfg.dirichlet_alpha,
        num_simulations=cfg.num_simulations,
        worker=_worker_name(),
        git_sha=_git_sha(),
        num_total_positions=total_ply,
        wdl_terminal_white=wdl_white,
        game_len_plies=total_ply,
    )
    finalize_game_targets(record)
    return record


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

def flush_game_record(record: GameRecord, model_version: int) -> str | None:
    """Write and upload a single GameRecord's .npz + sidecars to S3.

    Returns the .npz S3 key, or None if the record had zero full-search
    positions (nothing to flush).
    """
    if not record.positions:
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    rand = np.random.default_rng().integers(0, 0xFFFF_FFFF)
    n = len(record.positions)
    basename = f"{ts}_{int(rand):08x}_n{n}"
    npz_key = f"{storage.SELFPLAY_PREFIX}v{model_version}/{basename}.npz"
    meta_key = f"{storage.SELFPLAY_PREFIX}v{model_version}/{basename}.meta.json"
    trace_key = f"{storage.SELFPLAY_TRACES_PREFIX}v{model_version}/{record.game_id}.json"

    with tempfile.TemporaryDirectory() as td:
        npz_path = Path(td) / f"{basename}.npz"
        trace_path = Path(td) / f"{record.game_id}.json"
        write_samples_to_npz(npz_path, record)
        write_trace_json(trace_path, record)
        storage.put_file(npz_key, npz_path)
        storage.put_file(meta_key, npz_path.with_suffix(".meta.json"))
        storage.put_file(trace_key, trace_path)
    return npz_key


# ---------------------------------------------------------------------------
# Model refresh + heartbeat (unchanged)
# ---------------------------------------------------------------------------

def _read_model_version(cfg: AsyncConfig) -> tuple[int, str | None]:
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


def _write_heartbeat(cfg: AsyncConfig, version: int, total_games: int,
                     total_positions: int,
                     search_stats: dict | None = None) -> None:
    d: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": version,
        "total_games": total_games,
        "total_positions": total_positions,
    }
    if search_stats:
        d.update(search_stats)
    storage.put_json(f"{storage.HEARTBEATS_PREFIX}{_worker_name()}.json", d)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_worker(cfg: AsyncConfig) -> None:
    """Run the continuous self-play worker loop (v2 schema)."""
    if hexchess is None:
        raise ImportError("hexchess bindings not available")

    cfg.validate()
    setup_json_logging("worker", run_id=cfg.run_id)
    log_event("worker.start", run_id=cfg.run_id)

    def _cleanup(signum, frame):
        logger.info("Received signal {}, terminating child processes...", signum)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        os.killpg(0, signal.SIGTERM)
        sys.exit(1)

    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)
    try:
        os.setpgrp()
    except OSError:
        pass

    cfg.ensure_cache_dirs()
    current_version, model_path = _read_model_version(cfg)
    _write_heartbeat(cfg, current_version, 0, 0)

    # --- Bootstrap: imitation data until the first model appears ---
    if model_path is None:
        num_workers = max(1, os.cpu_count() or 1)
        logger.info("No model found — generating minimax imitation data "
                    "(depth {}, {} parallel workers)", cfg.imitation_depth, num_workers)
        imitation_games = 0
        imitation_positions = 0
        batch_size = cfg.worker_batch_size
        pending_samples: list[dict] = []
        pending_games = 0

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(play_imitation_game, cfg) for _ in range(num_workers * 2)}
            while model_path is None:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        samples = future.result()
                    except Exception as exc:
                        logger.exception("Imitation worker task failed: {}", exc)
                        futures.add(pool.submit(play_imitation_game, cfg))
                        _write_heartbeat(cfg, current_version, imitation_games, imitation_positions)
                        continue
                    pending_samples.extend(samples)
                    pending_games += 1
                    imitation_games += 1
                    imitation_positions += len(samples)
                    logger.info("  game {} complete: {} positions",
                                imitation_games, len(samples))
                    futures.add(pool.submit(play_imitation_game, cfg))
                    if pending_games >= batch_size:
                        key = storage.flush_samples(
                            pending_samples, storage.IMITATION_PREFIX)
                        logger.info(
                            "Imitation batch flushed: {} games, {} pos | {}",
                            pending_games, len(pending_samples), key,
                        )
                        _write_heartbeat(cfg, current_version, imitation_games, imitation_positions)
                        pending_samples = []
                        pending_games = 0
                        current_version, model_path = _read_model_version(cfg)
            for f in futures:
                f.cancel()

        if pending_samples:
            storage.flush_samples(pending_samples, storage.IMITATION_PREFIX)
        logger.info("Model appeared (v{}), switching to self-play", current_version)

    # --- Self-play ---
    logger.info("Worker starting self-play: v{} ({})", current_version, model_path)
    search = hexchess.MctsSearch(
        simulations=cfg.num_simulations,
        model_path=model_path,
        dirichlet_epsilon=cfg.dirichlet_epsilon,
        dirichlet_alpha=cfg.dirichlet_alpha,
    )
    py_rng = random.Random()
    total_games = 0
    total_positions = 0
    # Game-id generator: (unix-seconds << 16) | game-index, unique per worker.
    game_id_base = int(time.time()) << 20
    # Rolling window for search stats (plan §7.6 page 4).
    _SEARCH_STAT_WINDOW = 100
    recent_game_qs: list[float] = []
    recent_game_entropies: list[float] = []
    recent_game_lengths: list[int] = []

    while True:
        game_t0 = time.time()
        game_id = game_id_base + total_games
        try:
            record = _play_one_game_pcr(
                search, cfg,
                game_id=game_id,
                model_version=current_version,
                py_rng=py_rng,
            )
        except Exception as exc:
            logger.exception("self-play game crashed (game_id={}): {}", game_id, exc)
            _write_heartbeat(cfg, current_version, total_games, total_positions)
            time.sleep(1)
            continue
        elapsed = time.time() - game_t0
        total_games += 1
        total_positions += len(record.positions)

        key = flush_game_record(record, current_version)
        # Tier-1 per-game structured event (plan §7.3).
        mcts_qs = [p.root_q for p in record.positions] if record.positions else []
        mcts_entropies = [p.root_entropy for p in record.positions] if record.positions else []
        log_event(
            "selfplay.game",
            game_id=int(game_id),
            model_version=int(current_version),
            result=record.result,
            termination=getattr(record, "termination", None),
            num_full_positions=len(record.positions),
            num_total_positions=int(record.num_total_positions),
            duration_s=float(elapsed),
            mcts_mean_root_q=float(np.mean(mcts_qs)) if mcts_qs else 0.0,
            mcts_mean_root_entropy=float(np.mean(mcts_entropies)) if mcts_entropies else 0.0,
            pcr_full_position_count=len(record.positions),
            pcr_fast_position_count=int(record.num_total_positions) - len(record.positions),
        )
        logger.info(
            "game {} ({}): {} full-search / {} total plies, {:.1f}s | v{} | {}",
            total_games, record.result,
            len(record.positions), record.num_total_positions,
            elapsed, current_version, key,
        )

        # Update rolling search stats for heartbeat (plan §7.6 page 4).
        if mcts_qs:
            recent_game_qs.append(float(np.mean(mcts_qs)))
            recent_game_entropies.append(float(np.mean(mcts_entropies)))
        recent_game_lengths.append(record.game_len_plies)
        if len(recent_game_qs) > _SEARCH_STAT_WINDOW:
            recent_game_qs = recent_game_qs[-_SEARCH_STAT_WINDOW:]
            recent_game_entropies = recent_game_entropies[-_SEARCH_STAT_WINDOW:]
        if len(recent_game_lengths) > _SEARCH_STAT_WINDOW:
            recent_game_lengths = recent_game_lengths[-_SEARCH_STAT_WINDOW:]
        search_stats: dict | None = None
        if recent_game_qs:
            search_stats = {
                "recent_mean_root_q": round(float(np.mean(recent_game_qs)), 4),
                "recent_mean_root_entropy": round(float(np.mean(recent_game_entropies)), 4),
                "recent_mean_game_length": round(float(np.mean(recent_game_lengths)), 1),
                # Use Q window size (full-search games only) since Q/entropy stats
                # are only appended when mcts_qs is non-empty.
                "recent_games_window": len(recent_game_qs),
            }
        _write_heartbeat(cfg, current_version, total_games, total_positions, search_stats)

        # Per-game model refresh (plan §5.2).
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
