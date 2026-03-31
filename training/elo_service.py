"""Continuous Elo rating service with uncertainty-based matchmaking.

Runs as a k8s Deployment. Each iteration:
1. Polls for new model versions (models/v*.onnx)
2. Picks the most uncertain pair (fewest games played)
3. Plays one game, records the result
4. Periodically recomputes Elo and notifies Slack

Usage:
    python -m training elo-service
    python -m training elo-service --max-versions 20 --simulations 500
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from .config import _data_root
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
# State management
# ---------------------------------------------------------------------------


def _pair_key(a: str, b: str) -> str:
    """Canonical pair key: sorted names joined by ':'."""
    return ":".join(sorted([a, b]))


def _default_state() -> dict:
    return {
        "version": 1,
        "active_players": [],
        "retired_players": [],
        "pair_results": {},
        "ratings": {},
        "total_games": 0,
        "last_recompute_game": 0,
        "last_slack_game": 0,
    }


def _load_state(path: Path) -> dict:
    """Load state from disk, or return default if not found."""
    if path.exists():
        try:
            state = json.loads(path.read_text())
            if state.get("version") == 1:
                return state
            logger.warning("Unknown state version {}, starting fresh", state.get("version"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load state: {}, starting fresh", e)

    # Try bootstrapping from elo_rankings.jsonl
    rankings_path = _data_root() / "elo_rankings.jsonl"
    if rankings_path.exists():
        try:
            with open(rankings_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                latest = json.loads(lines[-1])
                state = _default_state()
                # Import matchup results
                for m in latest.get("matchups", []):
                    key = _pair_key(m["a"], m["b"])
                    a, b = key.split(":")
                    a_wins = m["a_wins"] if m["a"] == a else m["b_wins"]
                    b_wins = m["b_wins"] if m["a"] == a else m["a_wins"]
                    state["pair_results"][key] = {
                        "a_wins": a_wins,
                        "b_wins": b_wins,
                        "draws": m["draws"],
                        "a_as_white": (a_wins + m["draws"]) // 2,
                        "b_as_white": (b_wins + m["draws"]) // 2,
                    }
                state["ratings"] = latest.get("ratings", {})
                state["active_players"] = list(state["ratings"].keys())
                total = sum(
                    r["a_wins"] + r["b_wins"] + r["draws"]
                    for r in state["pair_results"].values()
                )
                state["total_games"] = total
                state["last_recompute_game"] = total
                state["last_slack_game"] = total
                logger.info("Bootstrapped state from elo_rankings.jsonl ({} games)", total)
                return state
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.warning("Failed to bootstrap from rankings: {}", e)

    return _default_state()


def _save_state(state: dict, path: Path) -> None:
    """Atomically write state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.rename(path)


# ---------------------------------------------------------------------------
# Player cache (LRU to bound memory)
# ---------------------------------------------------------------------------


class PlayerCache:
    """LRU cache for Player objects. Evicts least-recently-used when full."""

    def __init__(self, simulations: int, max_size: int = 10):
        self.simulations = simulations
        self.max_size = max(max_size, 4)  # must exceed baseline count (3)
        self._cache: OrderedDict[str, Player] = OrderedDict()
        self._baselines: dict[str, Player] = {}

    def _init_baselines(self) -> None:
        if self._baselines:
            return
        for p in baselines(self.simulations):
            self._baselines[p.name] = p
            self._cache[p.name] = p

    def get(self, name: str, model_path: str | None = None) -> Player:
        """Get or create a player. Moves it to the front of the LRU."""
        self._init_baselines()

        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]

        # Create new player
        if name in self._baselines:
            player = self._baselines[name]
        elif name.startswith("Minimax"):
            depth = int(name.split("-")[1])
            player = MinimaxPlayer(name=name, depth=depth)
        else:
            player = MctsPlayer(
                name=name,
                simulations=self.simulations,
                model_path=model_path,
            )

        # Evict if over capacity (don't evict baselines)
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
# Model discovery and player pool sync
# ---------------------------------------------------------------------------


def _discover_versions(models_dir: Path) -> list[tuple[int, Path]]:
    """Find all v*.onnx files, sorted by version number."""
    if not models_dir.exists():
        return []
    versions = []
    for f in models_dir.glob("v*.onnx"):
        try:
            v = int(f.stem[1:])
            versions.append((v, f))
        except ValueError:
            continue
    versions.sort(key=lambda x: x[0])
    return versions


def _sync_player_pool(
    state: dict,
    models_dir: Path,
    max_versions: int,
) -> tuple[dict[str, str | None], list[str]]:
    """Update active players with new models.

    Retires old versions beyond max_versions. Baselines are always active.
    Returns (path_map, new_models) where path_map maps player name to model
    path (None for non-model players) and new_models lists newly discovered versions.
    """
    baseline_names = {"Minimax-2", "Minimax-3", "Minimax-4", "Heuristic"}
    versions = _discover_versions(models_dir)
    all_version_names = {f"v{v}" for v, _ in versions}
    version_paths = {f"v{v}": str(p) for v, p in versions}

    # Add baselines if not present
    for name in baseline_names:
        if name not in state["active_players"]:
            state["active_players"].append(name)

    # Add new model versions
    new_models = []
    for v, path in versions:
        name = f"v{v}"
        if name not in state["active_players"] and name not in state["retired_players"]:
            state["active_players"].append(name)
            new_models.append(name)
            logger.info("New model discovered: {}", name)

    # Retire old versions (keep only most recent max_versions)
    model_players = [
        n for n in state["active_players"]
        if n not in baseline_names and n in all_version_names
    ]
    if len(model_players) > max_versions:
        # Sort by version number
        model_players.sort(key=lambda n: int(n[1:]))
        to_retire = model_players[:-max_versions]
        for name in to_retire:
            state["active_players"].remove(name)
            if name not in state["retired_players"]:
                state["retired_players"].append(name)
            logger.info("Retired old model: {}", name)

    # Also retire players whose model files no longer exist
    for name in list(state["active_players"]):
        if name not in baseline_names and name not in all_version_names:
            state["active_players"].remove(name)
            if name not in state["retired_players"]:
                state["retired_players"].append(name)
            logger.info("Retired missing model: {}", name)

    # Build path map
    path_map: dict[str, str | None] = {}
    for name in state["active_players"]:
        path_map[name] = version_paths.get(name)

    return path_map, new_models


# ---------------------------------------------------------------------------
# Uncertainty-based matchmaking
# ---------------------------------------------------------------------------


def _is_lopsided(result: dict, min_games: int = 30, threshold: float = 0.85) -> bool:
    """Return True if one side dominates decisively."""
    games = result["a_wins"] + result["b_wins"] + result["draws"]
    if games < min_games:
        return False
    winner_wins = max(result["a_wins"], result["b_wins"])
    return winner_wins / games >= threshold


def _select_pair(state: dict) -> tuple[str, str] | None:
    """Pick the pair with highest uncertainty (fewest games played).

    Skips lopsided pairs where the outcome is already clear.
    Returns (a, b) in canonical order, or None if all pairs are resolved.
    """
    players = state["active_players"]
    if len(players) < 2:
        raise ValueError("Need at least 2 active players")

    # Build list of (uncertainty, pair) tuples
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

            # Uncertainty: inf for unplayed, 1/sqrt(n) otherwise
            uncertainty = float("inf") if games == 0 else 1.0 / math.sqrt(games)
            candidates.append((uncertainty, key))

    # If all pairs are lopsided, signal caller to wait
    if not candidates:
        return None

    # Pick highest uncertainty, break ties randomly
    max_uncertainty = max(u for u, _ in candidates)
    top_pairs = [key for u, key in candidates if u == max_uncertainty]
    chosen = random.choice(top_pairs)
    a, b = chosen.split(":")
    return a, b


def _assign_colors(state: dict, pair_key: str, a: str, b: str) -> tuple[str, str]:
    """Return (white, black) based on color balance for this pair."""
    result = state["pair_results"].get(pair_key)
    if result is None:
        # First game — random
        return (a, b) if random.random() < 0.5 else (b, a)

    a_as_white = result.get("a_as_white", 0)
    b_as_white = result.get("b_as_white", 0)

    if a_as_white <= b_as_white:
        return a, b  # a plays white
    else:
        return b, a  # b plays white


# ---------------------------------------------------------------------------
# Result recording
# ---------------------------------------------------------------------------


def _record_result(
    state: dict, pair_key: str, white_name: str, outcome: str
) -> None:
    """Update pair_results with the game outcome."""
    a, b = pair_key.split(":")

    if pair_key not in state["pair_results"]:
        state["pair_results"][pair_key] = {
            "a_wins": 0, "b_wins": 0, "draws": 0,
            "a_as_white": 0, "b_as_white": 0,
        }

    result = state["pair_results"][pair_key]

    # Track who played white
    if white_name == a:
        result["a_as_white"] += 1
    else:
        result["b_as_white"] += 1

    # Record outcome
    if outcome == "draw":
        result["draws"] += 1
    elif outcome == "white":
        # White won — which player is that?
        if white_name == a:
            result["a_wins"] += 1
        else:
            result["b_wins"] += 1
    elif outcome == "black":
        # Black won
        if white_name == a:
            result["b_wins"] += 1
        else:
            result["a_wins"] += 1


# ---------------------------------------------------------------------------
# Elo recomputation
# ---------------------------------------------------------------------------


def _recompute_elo(state: dict) -> None:
    """Recompute Elo from full result matrix for active players."""
    active = state["active_players"]
    if len(active) < 2:
        return

    # Convert pair_results to the format compute_elo expects
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


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log_event(logs_dir: Path, event: dict) -> None:
    """Append JSON event to elo-service.jsonl."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    log_path = logs_dir / "elo-service.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(event) + "\n")


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
    state_path = _data_root() / "elo_state.json"
    logs_dir = _data_root() / "logs"
    models_dir = _data_root() / "models"

    state = _load_state(state_path)
    cache = PlayerCache(simulations)

    logger.info("Elo service starting ({} games in state)", state["total_games"])

    while True:
        # 1. Sync player pool
        path_map, new_models = _sync_player_pool(state, models_dir, max_versions)

        # 2. Need at least 2 active players
        if len(state["active_players"]) < 2:
            logger.info("Waiting for models... ({} players)", len(state["active_players"]))
            time.sleep(60)
            continue

        # 3. Track newly discovered models (notify after first Elo recompute)
        pending_notifications = new_models

        # 4. Select most uncertain pair (None if all resolved)
        pair = _select_pair(state)
        if pair is None:
            logger.info("All pairs resolved, waiting for new models...")
            time.sleep(60)
            continue
        a, b = pair
        pair_key = _pair_key(a, b)

        # 5. Assign colors
        white_name, black_name = _assign_colors(state, pair_key, a, b)

        # 6. Get player objects from cache
        white_player = cache.get(white_name, path_map.get(white_name))
        black_player = cache.get(black_name, path_map.get(black_name))

        # 7. Play game
        pair_games = _pair_game_count(state, pair_key)
        logger.info(
            "Game {}: {} (W) vs {} (B) [pair has {} prior games]",
            state["total_games"] + 1, white_name, black_name, pair_games,
        )

        t0 = time.time()
        outcome = play_game(white_player, black_player)
        elapsed = time.time() - t0

        logger.info(
            "  Result: {} ({:.0f}s)", outcome, elapsed,
        )

        # 8. Record result
        _record_result(state, pair_key, white_name, outcome)
        state["total_games"] += 1

        # 9. Log event
        _log_event(logs_dir, {
            "event": "game",
            "white": white_name,
            "black": black_name,
            "outcome": outcome,
            "elapsed_seconds": round(elapsed, 1),
            "total_games": state["total_games"],
        })

        # 10. Maybe recompute Elo
        games_since_recompute = state["total_games"] - state["last_recompute_game"]
        if games_since_recompute >= recompute_interval:
            _recompute_elo(state)
            logger.info("Elo updated (game {}):\n{}", state["total_games"],
                        format_elo_table(state["ratings"]))

            # Append to elo_rankings.jsonl for compatibility
            _append_rankings(state, simulations)

            # Notify about newly ranked models (deferred from discovery)
            for model_name in pending_notifications:
                if model_name in state["ratings"]:
                    _notify_new_model(state, model_name)
            pending_notifications = []

        # 11. Maybe notify Slack
        games_since_slack = state["total_games"] - state["last_slack_game"]
        if games_since_slack >= notify_interval:
            _notify_slack(state)

        # 12. Save state
        _save_state(state, state_path)


def _pair_game_count(state: dict, pair_key: str) -> int:
    r = state["pair_results"].get(pair_key)
    if r is None:
        return 0
    return r["a_wins"] + r["b_wins"] + r["draws"]


def _notify_slack(state: dict) -> None:
    from .slack import notify_elo_update
    if state["ratings"]:
        notify_elo_update(state["ratings"], state["total_games"])
        state["last_slack_game"] = state["total_games"]


def _notify_new_model(state: dict, model_name: str) -> None:
    from .slack import notify_elo_update
    notify_elo_update(state["ratings"], state["total_games"], new_model=model_name)
    state["last_slack_game"] = state["total_games"]


def _append_rankings(state: dict, simulations: int) -> None:
    """Append current ratings to elo_rankings.jsonl for backward compatibility."""
    if not state["ratings"]:
        return
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ratings": state["ratings"],
        "total_games": state["total_games"],
        "simulations": simulations,
        "source": "elo-service",
    }
    path = _data_root() / "elo_rankings.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
