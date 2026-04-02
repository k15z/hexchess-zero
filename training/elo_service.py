"""Continuous Elo rating service with uncertainty-based matchmaking.

Each iteration:
1. Polls for new model versions (models/versions/*.onnx in S3)
2. Picks the most uncertain pair (fewest games played)
3. Plays one game, records the result
4. Periodically recomputes Elo and notifies Slack

Usage:
    python -m training elo-service
"""

from __future__ import annotations

import math
import random
import time
from collections import OrderedDict
from datetime import datetime, timezone

from loguru import logger

from . import storage
from .config import AsyncConfig
from .elo import (
    MctsPlayer,
    MinimaxPlayer,
    Player,
    baselines,
    compute_elo,
    format_elo_table,
    play_game,
)

try:
    import hexchess
except ImportError:
    hexchess = None


# ---------------------------------------------------------------------------
# State management (S3-backed)
# ---------------------------------------------------------------------------


def _pair_key(a: str, b: str) -> str:
    return ":".join(sorted([a, b]))


def _default_state() -> dict:
    return {
        "version": 1,
        "active_players": [],
        "retired_players": [],
        "pair_results": {},
        "ratings": {},
        "player_stats": {},
        "total_games": 0,
        "last_recompute_game": 0,
        "last_slack_game": 0,
    }


def _load_state() -> dict:
    """Load state from S3, or return default if not found."""
    try:
        state = storage.get_json(storage.ELO_STATE)
        if state.get("version") == 1:
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

    def __init__(self, simulations: int, cache_dir, max_size: int = 10):
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
            # Download model from S3 to local cache
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
    baseline_names = {"Minimax-2", "Minimax-3", "Minimax-4", "Heuristic"}
    versions = _discover_versions()
    all_version_names = {f"v{v}" for v, _ in versions}
    version_keys = {f"v{v}": key for v, key in versions}

    for name in baseline_names:
        if name not in state["active_players"]:
            state["active_players"].append(name)

    new_models = []
    for v, key in versions:
        name = f"v{v}"
        if name not in state["active_players"] and name not in state["retired_players"]:
            state["active_players"].append(name)
            new_models.append(name)
            logger.info("New model discovered: {}", name)

    # Retire old versions
    model_players = [
        n for n in state["active_players"]
        if n not in baseline_names and n in all_version_names
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
        if name not in baseline_names and name not in all_version_names:
            state["active_players"].remove(name)
            if name not in state["retired_players"]:
                state["retired_players"].append(name)
            logger.info("Retired missing model: {}", name)

    path_map: dict[str, str | None] = {}
    for name in state["active_players"]:
        path_map[name] = version_keys.get(name)

    return path_map, new_models


# ---------------------------------------------------------------------------
# Matchmaking
# ---------------------------------------------------------------------------


def _is_lopsided(result: dict, min_games: int = 30, threshold: float = 0.85) -> bool:
    games = result["a_wins"] + result["b_wins"] + result["draws"]
    if games < min_games:
        return False
    winner_wins = max(result["a_wins"], result["b_wins"])
    return winner_wins / games >= threshold


def _select_pair(state: dict) -> tuple[str, str] | None:
    players = state["active_players"]
    if len(players) < 2:
        raise ValueError("Need at least 2 active players")

    candidates = []
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            a, b = players[i], players[j]
            key = _pair_key(a, b)
            result = state["pair_results"].get(key)
            if result is None:
                games = 0
            else:
                games = result["a_wins"] + result["b_wins"] + result["draws"]
                if _is_lopsided(result):
                    logger.debug("Skipping lopsided pair: {}", key)
                    continue

            uncertainty = float("inf") if games == 0 else 1.0 / math.sqrt(games)
            candidates.append((uncertainty, key))

    if not candidates:
        return None

    max_uncertainty = max(u for u, _ in candidates)
    top_pairs = [key for u, key in candidates if u == max_uncertainty]
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
    state: dict, pair_key: str, white_name: str, outcome: str
) -> None:
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

    if outcome == "draw":
        result["draws"] += 1
    elif outcome == "white":
        if white_name == a:
            result["a_wins"] += 1
        else:
            result["b_wins"] += 1
    elif outcome == "black":
        if white_name == a:
            result["b_wins"] += 1
        else:
            result["a_wins"] += 1


def _recompute_elo(state: dict) -> None:
    active = state["active_players"]
    if len(active) < 2:
        return

    results = []
    for key, r in state["pair_results"].items():
        a, b = key.split(":")
        if a in active and b in active:
            results.append({
                "a": a, "b": b,
                "a_wins": r["a_wins"], "b_wins": r["b_wins"], "draws": r["draws"],
            })

    if not results:
        return

    state["ratings"] = compute_elo(active, results)
    state["last_recompute_game"] = state["total_games"]


def _pair_game_count(state: dict, pair_key: str) -> int:
    r = state["pair_results"].get(pair_key)
    if r is None:
        return 0
    return r["a_wins"] + r["b_wins"] + r["draws"]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_elo_service(
    simulations: int = 500,
    max_versions: int = 20,
    recompute_interval: int = 10,
    notify_interval: int = 20,
) -> None:
    """Run the continuous Elo rating service."""
    cfg = AsyncConfig()
    cfg.ensure_cache_dirs()

    state = _load_state()
    cache = PlayerCache(simulations, cfg.model_cache_dir / "elo")

    logger.info("Elo service starting ({} games in state)", state["total_games"])

    while True:
        # 1. Sync player pool from S3
        path_map, new_models = _sync_player_pool(state, max_versions)

        if len(state["active_players"]) < 2:
            logger.info("Waiting for models... ({} players)", len(state["active_players"]))
            time.sleep(60)
            continue

        pending_notifications = new_models

        # 2. Select most uncertain pair
        pair = _select_pair(state)
        if pair is None:
            logger.info("All pairs resolved, waiting for new models...")
            time.sleep(60)
            continue
        a, b = pair
        pair_key = _pair_key(a, b)

        # 3. Assign colors
        white_name, black_name = _assign_colors(state, pair_key, a, b)

        # 4. Get player objects
        white_player = cache.get(white_name, path_map.get(white_name))
        black_player = cache.get(black_name, path_map.get(black_name))

        # 5. Play game
        pair_games = _pair_game_count(state, pair_key)
        logger.info(
            "Game {}: {} (W) vs {} (B) [pair has {} prior games]",
            state["total_games"] + 1, white_name, black_name, pair_games,
        )

        result = play_game(white_player, black_player)
        outcome = result["outcome"]

        w_avg = result["white_time"] / max(result["white_moves"], 1)
        b_avg = result["black_time"] / max(result["black_moves"], 1)
        logger.info(
            "  Result: {} ({} moves) | {} {:.2f}s/move | {} {:.2f}s/move",
            outcome, result["moves"],
            white_name, w_avg, black_name, b_avg,
        )

        # 6. Record result and timing
        _record_result(state, pair_key, white_name, outcome)
        state["total_games"] += 1

        player_stats = state.setdefault("player_stats", {})
        for name, t, m in [(white_name, result["white_time"], result["white_moves"]),
                           (black_name, result["black_time"], result["black_moves"])]:
            ps = player_stats.setdefault(name, {"total_time": 0.0, "total_moves": 0})
            ps["total_time"] = round(ps["total_time"] + t, 2)
            ps["total_moves"] += m

        # 7. Maybe recompute Elo
        games_since_recompute = state["total_games"] - state["last_recompute_game"]
        if games_since_recompute >= recompute_interval:
            _recompute_elo(state)
            logger.info("Elo updated (game {}):\n{}", state["total_games"],
                        format_elo_table(state["ratings"]))

            for model_name in pending_notifications:
                if model_name in state["ratings"]:
                    _notify_new_model(state, model_name)
            pending_notifications = []

        # 8. Maybe notify Slack
        games_since_slack = state["total_games"] - state["last_slack_game"]
        if games_since_slack >= notify_interval:
            _notify_slack(state)

        # 9. Save state to S3
        _save_state(state)


def _notify_slack(state: dict) -> None:
    from .slack import notify_elo_update
    if state["ratings"]:
        notify_elo_update(state["ratings"], state["total_games"])
        state["last_slack_game"] = state["total_games"]


def _notify_new_model(state: dict, model_name: str) -> None:
    from .slack import notify_elo_update
    notify_elo_update(state["ratings"], state["total_games"], new_model=model_name)
    state["last_slack_game"] = state["total_games"]
