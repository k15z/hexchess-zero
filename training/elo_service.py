"""Continuous Elo rating service with uncertainty-based matchmaking."""

from __future__ import annotations

import random
import time
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
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
# State management (S3-backed)
# ---------------------------------------------------------------------------


def _pair_key(a: str, b: str) -> str:
    return ":".join(sorted([a, b]))


def _default_state() -> dict:
    return {
        "version": 2,
        "active_players": [],
        "retired_players": [],
        "ratings": {},  # {name: {"mu": float, "sigma": float}}
        "pair_results": {},  # {pair_key: {"a_wins": int, "b_wins": int, "draws": int, "a_as_white": int, "b_as_white": int}}
        "player_stats": {},  # {name: {"total_time": float, "total_moves": int}}
        "total_games": 0,
        "last_slack_game": 0,
    }


def _load_state() -> dict:
    """Load state from S3, or return default if not found."""
    try:
        state = storage.get_json(storage.ELO_STATE)
        if state.get("version") == 2:
            return state
        logger.warning("Unknown state version {}, starting fresh", state.get("version"))
    except KeyError as e:
        logger.info("No existing elo state: {}", e)
    return _default_state()


def _save_state(state: dict) -> None:
    """Write state to S3."""
    storage.put_json(storage.ELO_STATE, state)


# ---------------------------------------------------------------------------
# Player cache (LRU to bound memory)
# ---------------------------------------------------------------------------


class PlayerCache:
    """LRU cache for Player objects. Evicts least-recently-used when full."""

    def __init__(self, simulations: int, cache_dir, max_size: int = 6):
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
        """Get or create a player. Moves it to the front of the LRU."""
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
# Model discovery from S3
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


def _sync_player_pool(
    state: dict,
    max_versions: int,
) -> tuple[dict[str, str | None], list[str]]:
    """Update active players with new models from S3."""
    versions = _discover_versions()
    all_version_names = {f"v{v}" for v, _ in versions}
    version_keys = {f"v{v}": key for v, key in versions}

    for name in BASELINE_NAMES:
        if name not in state["active_players"]:
            state["active_players"].append(name)
        if name not in state["ratings"]:
            state["ratings"][name] = new_rating()

    new_models = []
    for v, key in versions:
        name = f"v{v}"
        if name not in state["active_players"] and name not in state["retired_players"]:
            state["active_players"].append(name)
            state["ratings"][name] = new_rating()
            new_models.append(name)
            logger.info("New model discovered: {}", name)

    # Retire old versions
    model_players = [
        n for n in state["active_players"]
        if n not in BASELINE_NAMES and n in all_version_names
    ]
    if len(model_players) > max_versions:
        model_players.sort(key=lambda n: int(n[1:]))
        to_retire = model_players[:-max_versions]
        for name in to_retire:
            state["active_players"].remove(name)
            if name not in state["retired_players"]:
                state["retired_players"].append(name)
            logger.info("Retired old model: {}", name)

    for name in list(state["active_players"]):
        if name not in BASELINE_NAMES and name not in all_version_names:
            state["active_players"].remove(name)
            if name not in state["retired_players"]:
                state["retired_players"].append(name)
            logger.info("Retired missing model: {}", name)

    path_map: dict[str, str | None] = {}
    for name in state["active_players"]:
        path_map[name] = version_keys.get(name)

    return path_map, new_models


# ---------------------------------------------------------------------------
# Matchmaking (uncertainty-based via OpenSkill sigma)
# ---------------------------------------------------------------------------


def _is_lopsided(result: dict, min_games: int = 30, threshold: float = 0.85) -> bool:
    games = result["a_wins"] + result["b_wins"] + result["draws"]
    if games < min_games:
        return False
    winner_wins = max(result["a_wins"], result["b_wins"])
    return winner_wins / games >= threshold


def _select_pair(state: dict, exclude: set[str] | None = None) -> tuple[str, str] | None:
    """Select the pair that would benefit most from another game.

    Uses OpenSkill sigma (uncertainty) of both players plus closeness of
    ratings to prioritize informative matchups.
    """
    players = state["active_players"]
    if len(players) < 2:
        raise ValueError("Need at least 2 active players")

    ratings = state["ratings"]
    candidates = []

    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            a, b = players[i], players[j]
            key = _pair_key(a, b)
            result = state["pair_results"].get(key)

            if exclude and key in exclude:
                continue

            if result is not None and _is_lopsided(result):
                logger.debug("Skipping lopsided pair: {}", key)
                continue

            ra = ratings.get(a, new_rating())
            rb = ratings.get(b, new_rating())

            # Combined uncertainty: higher sigma = more to learn
            uncertainty = ra["sigma"] + rb["sigma"]

            # Closeness bonus: similar-rated pairs give more information
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
    else:
        return b, a


# ---------------------------------------------------------------------------
# Result recording
# ---------------------------------------------------------------------------


def _record_result(
    state: dict, pair_key: str, white_name: str, outcome: str,
    move_count: int,
) -> None:
    """Record game result: update pair counts, OpenSkill ratings, and game log."""
    a, b = pair_key.split(":")

    if pair_key not in state["pair_results"]:
        state["pair_results"][pair_key] = {
            "a_wins": 0, "b_wins": 0, "draws": 0,
            "a_as_white": 0, "b_as_white": 0,
        }

    result = state["pair_results"][pair_key]

    if white_name == a:
        result["a_as_white"] += 1
    else:
        result["b_as_white"] += 1

    # Determine outcome in terms of a/b
    if outcome == "draw":
        result["draws"] += 1
        os_outcome = "draw"
    elif outcome == "white":
        if white_name == a:
            result["a_wins"] += 1
            os_outcome = "a_wins"
        else:
            result["b_wins"] += 1
            os_outcome = "b_wins"
    elif outcome == "black":
        if white_name == a:
            result["b_wins"] += 1
            os_outcome = "b_wins"
        else:
            result["a_wins"] += 1
            os_outcome = "a_wins"

    # Update OpenSkill ratings incrementally
    ra = state["ratings"].get(a, new_rating())
    rb = state["ratings"].get(b, new_rating())
    state["ratings"][a], state["ratings"][b] = update_ratings(ra, rb, os_outcome)

    # Append to game log
    black_name = b if white_name == a else a
    game_record = {
        "game": state["total_games"] + 1,
        "white": white_name,
        "black": black_name,
        "outcome": outcome,
        "moves": move_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        storage.append_jsonl(storage.ELO_GAMES_LOG, game_record)
    except Exception as e:
        logger.warning("Failed to append game log: {}", e)


def _pair_game_count(state: dict, pair_key: str) -> int:
    r = state["pair_results"].get(pair_key)
    if r is None:
        return 0
    return r["a_wins"] + r["b_wins"] + r["draws"]


# ---------------------------------------------------------------------------
# Worker process: plays games in a subprocess with its own PlayerCache
# ---------------------------------------------------------------------------


_worker_cache: PlayerCache | None = None


def _worker_init(simulations: int, cache_dir_str: str, cache_size: int) -> None:
    """Initializer for ProcessPoolExecutor workers."""
    global _worker_cache
    _worker_cache = PlayerCache(simulations, Path(cache_dir_str), max_size=cache_size)


def _worker_play(
    white_name: str,
    black_name: str,
    path_map: dict[str, str | None],
) -> dict:
    assert _worker_cache is not None, "Worker not initialized"
    white = _worker_cache.get(white_name, path_map.get(white_name))
    black = _worker_cache.get(black_name, path_map.get(black_name))
    return play_game(white, black)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


# Peak ONNX session memory: num_workers * WORKER_CACHE_SIZE * ~800MB.
# With defaults (2 workers, 6 slots) that's ~10 GB, fitting a 16Gi pod.
WORKER_CACHE_SIZE = 6

# Throttle S3 LIST of model versions. New models appear on the scale of
# hours; polling every game is wasteful.
SYNC_INTERVAL_SECONDS = 300

# Throttle S3 PUT of state/elo.json. elo_games.jsonl is the durable record,
# so state only needs to be consistent on restart.
SAVE_EVERY_N_GAMES = 5


def run_elo_service(
    simulations: int = 500,
    max_versions: int = 20,
    notify_interval: int = 20,
    num_workers: int = 2,
) -> None:
    """Run the continuous Elo rating service with parallel game workers."""
    cfg = AsyncConfig()
    cfg.ensure_cache_dirs()

    state = _load_state()
    cache_dir = cfg.model_cache_dir / "elo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Elo service starting ({} games in state, {} workers)",
        state["total_games"], num_workers,
    )

    executor = ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(simulations, str(cache_dir), WORKER_CACHE_SIZE),
    )

    in_flight: dict[Future, tuple[str, str, str]] = {}
    in_flight_keys: set[str] = set()
    pending_notifications: list[str] = []
    last_sync_ts = 0.0
    path_map: dict[str, str | None] = {}
    games_since_save = 0

    try:
        while True:
            now = time.monotonic()
            if now - last_sync_ts >= SYNC_INTERVAL_SECONDS or not path_map:
                path_map, new_models = _sync_player_pool(state, max_versions)
                pending_notifications.extend(new_models)
                last_sync_ts = now

            if len(state["active_players"]) < 2:
                logger.info(
                    "Waiting for models... ({} players)",
                    len(state["active_players"]),
                )
                time.sleep(60)
                continue

            while len(in_flight) < num_workers:
                pair = _select_pair(state, exclude=in_flight_keys)
                if pair is None:
                    break
                a, b = pair
                pair_key = _pair_key(a, b)
                white_name, black_name = _assign_colors(state, pair_key, a, b)

                pair_games = _pair_game_count(state, pair_key)
                logger.info(
                    "Dispatch game {}: {} (W) vs {} (B) [pair has {} prior games]",
                    state["total_games"] + len(in_flight) + 1,
                    white_name, black_name, pair_games,
                )
                fut = executor.submit(
                    _worker_play, white_name, black_name, path_map,
                )
                in_flight[fut] = (pair_key, white_name, black_name)
                in_flight_keys.add(pair_key)

            if not in_flight:
                logger.info("No dispatchable pairs, waiting for new models...")
                time.sleep(60)
                continue

            done, _pending = wait(
                list(in_flight.keys()), return_when=FIRST_COMPLETED,
            )

            for fut in done:
                pair_key, white_name, black_name = in_flight.pop(fut)
                in_flight_keys.discard(pair_key)
                try:
                    result = fut.result()
                except Exception as e:
                    logger.error(
                        "Game {} vs {} failed: {}", white_name, black_name, e,
                    )
                    continue

                outcome = result["outcome"]
                w_avg = result["white_time"] / max(result["white_moves"], 1)
                b_avg = result["black_time"] / max(result["black_moves"], 1)
                logger.info(
                    "  Result: {} ({} moves) | {} {:.2f}s/move | {} {:.2f}s/move",
                    outcome, result["moves"],
                    white_name, w_avg, black_name, b_avg,
                )

                # OpenSkill updates must stay serial for reproducibility.
                _record_result(
                    state, pair_key, white_name, outcome, result["moves"],
                )
                state["total_games"] += 1
                games_since_save += 1

                player_stats = state.setdefault("player_stats", {})
                for name, t, m in [
                    (white_name, result["white_time"], result["white_moves"]),
                    (black_name, result["black_time"], result["black_moves"]),
                ]:
                    ps = player_stats.setdefault(
                        name, {"total_time": 0.0, "total_moves": 0},
                    )
                    ps["total_time"] = round(ps["total_time"] + t, 2)
                    ps["total_moves"] += m

                if state["total_games"] % 10 == 0:
                    logger.info(
                        "Ratings (game {}):\n{}",
                        state["total_games"],
                        format_elo_table(state["ratings"]),
                    )

            still_pending = []
            for model_name in pending_notifications:
                if model_name in state["ratings"]:
                    _notify_new_model(state, model_name)
                else:
                    still_pending.append(model_name)
            pending_notifications = still_pending

            games_since_slack = state["total_games"] - state["last_slack_game"]
            if games_since_slack >= notify_interval:
                _notify_slack(state)

            if games_since_save >= SAVE_EVERY_N_GAMES:
                _save_state(state)
                games_since_save = 0
    finally:
        if games_since_save > 0:
            try:
                _save_state(state)
            except Exception as e:
                logger.warning("Final state save failed: {}", e)
        executor.shutdown(wait=False, cancel_futures=True)


def _notify_slack(state: dict) -> None:
    from .slack import notify_elo_update
    if state["ratings"]:
        notify_elo_update(state["ratings"], state["total_games"])
        state["last_slack_game"] = state["total_games"]


def _notify_new_model(state: dict, model_name: str) -> None:
    from .slack import notify_elo_update
    notify_elo_update(state["ratings"], state["total_games"], new_model=model_name)
    state["last_slack_game"] = state["total_games"]
