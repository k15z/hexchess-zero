"""Continuous Elo rating service with uncertainty-based matchmaking.

Horizontally scalable: multiple replicas can run concurrently. The source of
truth for game results is ``state/elo_games/{ts}_{rand}.json`` — one immutable
object per game. Writes are race-free (unique keys). The legacy single-blob
``state/elo.json`` and append-only ``state/elo_games.jsonl`` are now *derived
projections* rebuilt from the per-game objects. Any replica can overwrite them
idempotently; last-writer-wins is fine because they are pure functions of the
game log.

Each replica plays one game at a time (no in-process parallelism). K8s
replicas provide parallelism; matchmaking collisions across replicas are
harmless — two pods occasionally picking the same high-uncertainty pair just
means more samples on a matchup we already care about.
"""

from __future__ import annotations

import os
import random
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from . import storage
from .config import AsyncConfig
from .elo import (
    MctsPlayer,
    MinimaxPlayer,
    Player,
    baselines,
    format_elo_table,
    new_rating,
    play_game,
    update_ratings,
)

try:
    import hexchess
except ImportError:
    hexchess = None


BASELINE_NAMES = frozenset({"Minimax-2", "Minimax-3", "Minimax-4", "Heuristic"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pair_key(a: str, b: str) -> str:
    return ":".join(sorted([a, b]))


# ---------------------------------------------------------------------------
# Player cache (LRU to bound memory)
# ---------------------------------------------------------------------------


class PlayerCache:
    """LRU cache for Player objects. Evicts least-recently-used when full."""

    def __init__(self, simulations: int, cache_dir: Path, max_size: int = 6):
        self.simulations = simulations
        self.cache_dir = cache_dir
        self.max_size = max(max_size, 4)
        self._cache: OrderedDict[str, Player] = OrderedDict()
        self._baselines: dict[str, Player] = {}

    def _init_baselines(self) -> None:
        if self._baselines:
            return
        for p in baselines(self.simulations):
            self._baselines[p.name] = p
            self._cache[p.name] = p

    def get(self, name: str, s3_model_key: str | None = None) -> Player:
        self._init_baselines()

        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]

        if name in self._baselines:
            player = self._baselines[name]
        elif name.startswith("Minimax"):
            depth = int(name.split("-")[1])
            player = MinimaxPlayer(name=name, depth=depth)
        else:
            local_path = self.cache_dir / f"{name}.onnx"
            if not local_path.exists() and s3_model_key:
                storage.get_file(s3_model_key, local_path)
            player = MctsPlayer(
                name=name,
                simulations=self.simulations,
                model_path=str(local_path),
            )

        while len(self._cache) >= self.max_size:
            oldest_name, _ = next(iter(self._cache.items()))
            if oldest_name in self._baselines:
                self._cache.move_to_end(oldest_name)
                continue
            self._cache.pop(oldest_name)
            logger.debug("Evicted player {} from cache", oldest_name)

        self._cache[name] = player
        return player


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


def _discover_versions() -> list[tuple[int, str]]:
    """Find all model versions in S3. Returns [(version, s3_key), ...] sorted."""
    versions = []
    for key in storage.ls(storage.VERSIONS_PREFIX):
        name = key.split("/")[-1]
        if not name.endswith(".onnx"):
            continue
        try:
            v = int(name.replace(".onnx", ""))
            versions.append((v, key))
        except ValueError:
            continue
    versions.sort(key=lambda x: x[0])
    return versions


def _desired_active(max_versions: int) -> tuple[list[str], dict[str, str]]:
    """Return (active_player_list, version_key_map) for the current S3 state.

    Active set = baselines + latest N model versions (baselines first).
    """
    versions = _discover_versions()
    version_keys = {f"v{v}": key for v, key in versions}
    sorted_names = sorted(version_keys.keys(), key=lambda n: int(n[1:]))
    desired_models = sorted_names[-max_versions:] if max_versions > 0 else []
    active = list(BASELINE_NAMES) + desired_models
    return active, version_keys


# ---------------------------------------------------------------------------
# State projection: rebuild ratings / pair_results / player_stats from the
# per-game record log. This is a pure function of (records, active_players).
# ---------------------------------------------------------------------------


def _empty_state() -> dict:
    return {
        "active_players": [],
        "retired_players": [],
        "ratings": {},
        "pair_results": {},
        "player_stats": {},
        "total_games": 0,
    }


def _apply_record(state: dict, rec: dict) -> bool:
    """Fold a single game record into ``state`` in place. Returns True on success.

    Callable both from the full replay path (``_build_state``) and incrementally
    after a local game, so the rating math lives in exactly one place.
    """
    white = rec.get("white")
    black = rec.get("black")
    outcome = rec.get("outcome")
    if not white or not black or outcome not in ("white", "black", "draw"):
        return False

    ratings = state["ratings"]
    for name in (white, black):
        if name not in ratings:
            ratings[name] = new_rating()

    a, b = sorted([white, black])
    key = _pair_key(a, b)
    result = state["pair_results"].setdefault(
        key,
        {"a_wins": 0, "b_wins": 0, "draws": 0, "a_as_white": 0, "b_as_white": 0},
    )
    result["a_as_white" if white == a else "b_as_white"] += 1

    if outcome == "draw":
        os_outcome = "draw"
        result["draws"] += 1
    else:
        winner_is_a = (outcome == "white") == (white == a)
        os_outcome = "a_wins" if winner_is_a else "b_wins"
        result[os_outcome] += 1

    ratings[a], ratings[b] = update_ratings(ratings[a], ratings[b], os_outcome)

    player_stats = state["player_stats"]
    for name, t_key, m_key in (
        (white, "white_time", "white_moves"),
        (black, "black_time", "black_moves"),
    ):
        ps = player_stats.setdefault(name, {"total_time": 0.0, "total_moves": 0})
        ps["total_time"] = round(ps["total_time"] + float(rec.get(t_key, 0.0)), 2)
        ps["total_moves"] += int(rec.get(m_key, 0))

    state["total_games"] += 1
    return True


def _finalize_active(state: dict, active_players: list[str]) -> None:
    """Set active/retired player lists and ensure every active player has a rating."""
    ratings = state["ratings"]
    for name in active_players:
        if name not in ratings:
            ratings[name] = new_rating()
    state["active_players"] = list(active_players)
    state["retired_players"] = sorted(
        set(ratings.keys()) - set(active_players) - BASELINE_NAMES,
    )


def _build_state(records: list[dict], active_players: list[str]) -> dict:
    """Replay per-game records to produce the full Elo state projection."""
    state = _empty_state()
    for rec in sorted(records, key=lambda r: r.get("timestamp", "") or ""):
        _apply_record(state, rec)
    _finalize_active(state, active_players)
    return state


# ---------------------------------------------------------------------------
# Record store: incremental cache of per-game objects from S3
# ---------------------------------------------------------------------------


class GameRecordStore:
    """Caches per-game records. Refreshes by LIST + fetch-new."""

    def __init__(self) -> None:
        self._records: dict[str, dict] = {}  # key -> record

    def __len__(self) -> int:
        return len(self._records)

    def all_records(self) -> list[dict]:
        return list(self._records.values())

    def add_local(self, key: str, record: dict) -> None:
        """Register a record that this replica just wrote, so we don't refetch."""
        self._records[key] = record

    def refresh(self) -> int:
        """Fetch any new per-game objects from S3. Returns number added."""
        remote = set(storage.list_game_record_keys())
        new_keys = sorted(remote - self._records.keys())
        added = _fetch_records_parallel(new_keys, self._records)
        for k in list(self._records.keys()):
            if k not in remote:
                self._records.pop(k, None)
        return added


def _fetch_records_parallel(
    keys: list[str], into: dict[str, dict], max_workers: int = 16,
) -> int:
    """Parallel GET of JSON objects. S3 GETs are I/O-bound; serial fetches of
    thousands of records on cold start are the service's startup bottleneck.
    """
    if not keys:
        return 0
    from concurrent.futures import ThreadPoolExecutor

    added = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for k, rec in zip(keys, ex.map(_safe_get_json, keys)):
            if rec is not None:
                into[k] = rec
                added += 1
    return added


def _safe_get_json(key: str) -> dict | None:
    try:
        return storage.get_json(key)
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# Matchmaking
# ---------------------------------------------------------------------------


def _is_lopsided(result: dict, min_games: int = 30, threshold: float = 0.85) -> bool:
    games = result["a_wins"] + result["b_wins"] + result["draws"]
    if games < min_games:
        return False
    winner_wins = max(result["a_wins"], result["b_wins"])
    return winner_wins / games >= threshold


def _sentinel_pair(state: dict) -> tuple[str, str] | None:
    versions = sorted(
        int(n[1:]) for n in state["active_players"]
        if n.startswith("v") and n[1:].isdigit()
    )
    if len(versions) < SENTINEL_LOOKBACK + 1:
        return None
    latest = versions[-1]
    ancestor = latest - SENTINEL_LOOKBACK
    a, b = f"v{latest}", f"v{ancestor}"
    if a in state["active_players"] and b in state["active_players"]:
        return a, b
    return None


def _select_pair(state: dict) -> tuple[str, str] | None:
    """Select the pair that would benefit most from another game.

    No in-flight exclusion: cross-replica collisions just add more samples on
    a high-uncertainty matchup, which is fine.
    """
    players = state["active_players"]
    if len(players) < 2:
        return None

    if random.random() < SENTINEL_PROBABILITY:
        sp = _sentinel_pair(state)
        if sp is not None:
            return sp

    ratings = state["ratings"]
    candidates = []

    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            a, b = players[i], players[j]
            key = _pair_key(a, b)
            result = state["pair_results"].get(key)

            if result is not None and _is_lopsided(result):
                continue

            ra = ratings.get(a, new_rating())
            rb = ratings.get(b, new_rating())

            uncertainty = ra["sigma"] + rb["sigma"]
            mu_diff = abs(ra["mu"] - rb["mu"])
            closeness = 1.0 / (1.0 + mu_diff)

            score = uncertainty * (1.0 + closeness)
            candidates.append((score, key))

    if not candidates:
        return None

    max_score = max(s for s, _ in candidates)
    top_pairs = [key for s, key in candidates if s >= max_score * 0.95]
    chosen = random.choice(top_pairs)
    a, b = chosen.split(":")
    return a, b


def _assign_colors(state: dict, pair_key: str, a: str, b: str) -> tuple[str, str]:
    result = state["pair_results"].get(pair_key)
    if result is None:
        return (a, b) if random.random() < 0.5 else (b, a)

    a_as_white = result.get("a_as_white", 0)
    b_as_white = result.get("b_as_white", 0)

    if a_as_white <= b_as_white:
        return a, b
    return b, a


def _pair_game_count(state: dict, pair_key: str) -> int:
    r = state["pair_results"].get(pair_key)
    if r is None:
        return 0
    return r["a_wins"] + r["b_wins"] + r["draws"]


# ---------------------------------------------------------------------------
# Migration: one-shot backfill from the legacy jsonl log to per-game objects.
# Idempotent — safe to run from multiple replicas simultaneously because the
# per-game key is derived from the record timestamp plus a content hash, so
# duplicate writes collide to the same key and overwrite identically.
# ---------------------------------------------------------------------------


def migrate_legacy_jsonl() -> int:
    """Backfill legacy state/elo_games.jsonl into per-game objects. Idempotent."""
    import hashlib
    import json

    records = storage.get_jsonl(storage.ELO_GAMES_LOG)
    if not records:
        logger.info("No legacy jsonl to migrate.")
        return 0

    existing = set(storage.list_game_record_keys())
    written = 0
    for record in records:
        # Content-hashed key makes the migration idempotent: replays (even
        # concurrent ones from multiple replicas) collide to the same key.
        ts = record.get("timestamp", "19700101T000000").replace(":", "").replace("-", "")[:15]
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
        h = hashlib.sha1(payload.encode()).hexdigest()[:12]
        key = f"{storage.ELO_GAMES_PREFIX}{ts}_{h}.json"
        if key in existing:
            continue
        storage.put_json(key, record)
        written += 1

    logger.info("Migrated {} legacy records to per-game objects.", written)
    return written


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


PLAYER_CACHE_SIZE = 6

# How often to re-LIST per-game objects + re-discover model versions.
SYNC_INTERVAL_SECONDS = 300

# How often a replica writes the derived state/elo.json projection for the
# dashboard. Races across replicas are fine because the projection is a pure
# function of the game log.
PROJECTION_INTERVAL_SECONDS = 120

SENTINEL_LOOKBACK = 4
SENTINEL_PROBABILITY = 0.33

# Slack notifications should fire from at most one replica. Gate behind an
# env var the operator sets on exactly one pod (or zero — they're optional).
SLACK_LEADER_ENV = "ELO_SLACK_LEADER"


def run_elo_service(
    simulations: int = 500,
    max_versions: int = 5,
    notify_interval: int = 20,
) -> None:
    """Run the continuous Elo rating service. One game at a time per replica."""
    cfg = AsyncConfig()
    cfg.ensure_cache_dirs()

    cache_dir = cfg.model_cache_dir / "elo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    players_cache = PlayerCache(simulations, cache_dir, max_size=PLAYER_CACHE_SIZE)
    records = GameRecordStore()

    slack_leader = os.environ.get(SLACK_LEADER_ENV) == "1"
    last_slack_game = 0

    logger.info(
        "Elo service starting (simulations={}, max_versions={}, slack_leader={})",
        simulations, max_versions, slack_leader,
    )

    # Initial full sync: pull all records + discover models + build state.
    records.refresh()
    active, version_keys = _desired_active(max_versions)
    state = _build_state(records.all_records(), active)
    logger.info(
        "Initial state: {} games, {} active players",
        state["total_games"], len(state["active_players"]),
    )

    last_sync_ts = time.monotonic()
    last_projection_ts = 0.0

    while True:
        now = time.monotonic()

        # Periodic full rebuild from S3: picks up games written by peer replicas
        # and newly-promoted models.
        if now - last_sync_ts >= SYNC_INTERVAL_SECONDS:
            try:
                added = records.refresh()
                active, version_keys = _desired_active(max_versions)
                state = _build_state(records.all_records(), active)
                last_sync_ts = now
                if added:
                    logger.info(
                        "Sync: merged {} peer records ({} total, {} active)",
                        added, state["total_games"], len(state["active_players"]),
                    )
            except Exception as e:
                logger.warning("Sync failed: {}", e)

        if len(state["active_players"]) < 2:
            logger.info(
                "Waiting for models... ({} players)",
                len(state["active_players"]),
            )
            time.sleep(60)
            continue

        pair = _select_pair(state)
        if pair is None:
            logger.info("No dispatchable pairs, waiting...")
            time.sleep(60)
            continue

        a, b = pair
        pair_key = _pair_key(a, b)
        white_name, black_name = _assign_colors(state, pair_key, a, b)
        pair_games = _pair_game_count(state, pair_key)

        logger.info(
            "Dispatch game {}: {} (W) vs {} (B) [pair has {} prior games]",
            state["total_games"] + 1, white_name, black_name, pair_games,
        )

        try:
            white = players_cache.get(white_name, version_keys.get(white_name))
            black = players_cache.get(black_name, version_keys.get(black_name))
            result = play_game(white, black)
        except Exception as e:
            logger.error("Game {} vs {} failed: {}", white_name, black_name, e)
            time.sleep(5)
            continue

        outcome = result["outcome"]
        w_avg = result["white_time"] / max(result["white_moves"], 1)
        b_avg = result["black_time"] / max(result["black_moves"], 1)
        logger.info(
            "  Result: {} ({} moves) | {} {:.2f}s/move | {} {:.2f}s/move",
            outcome, result["moves"],
            white_name, w_avg, black_name, b_avg,
        )

        record = {
            "white": white_name,
            "black": black_name,
            "outcome": outcome,
            "moves": result["moves"],
            "white_time": result["white_time"],
            "black_time": result["black_time"],
            "white_moves": result["white_moves"],
            "black_moves": result["black_moves"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            key = storage.put_game_record(record)
            records.add_local(key, record)
        except Exception as e:
            logger.error("Failed to persist game record: {}", e)
            continue

        # Incremental fold — avoids an O(N) OpenSkill replay after every game.
        # The periodic sync above rebuilds from scratch to absorb peer writes.
        _apply_record(state, record)

        if state["total_games"] % 10 == 0:
            logger.info(
                "Ratings (game {}):\n{}",
                state["total_games"],
                format_elo_table(state["ratings"]),
            )

        # Periodically write the derived projection for the dashboard/metrics.
        if time.monotonic() - last_projection_ts >= PROJECTION_INTERVAL_SECONDS:
            try:
                storage.put_json(storage.ELO_STATE, state)
                last_projection_ts = time.monotonic()
            except Exception as e:
                logger.warning("Projection write failed: {}", e)

        if slack_leader:
            games_since_slack = state["total_games"] - last_slack_game
            if games_since_slack >= notify_interval:
                try:
                    _notify_slack(state)
                    last_slack_game = state["total_games"]
                except Exception as e:
                    logger.warning("Slack notify failed: {}", e)


def _notify_slack(state: dict) -> None:
    from .slack import notify_elo_update
    if state["ratings"]:
        notify_elo_update(state["ratings"], state["total_games"])
