"""Continuous Elo rating service with uncertainty-based matchmaking.

Horizontally scalable: multiple replicas can run concurrently. The source of
truth for game results is ``state/elo_games/{ts}_{rand}.json`` — one immutable
object per game. Writes are race-free (unique keys). ``state/elo.json`` is a
*derived projection* rebuilt from the per-game objects. Any replica can
overwrite it idempotently; last-writer-wins is fine because it is a pure
function of the game log.

Each replica usually plays one game at a time (no in-process parallelism).
When a promotion gate is active, a replica that picks the gate matchup plays a
two-game paired-color mini-match back-to-back. K8s replicas provide overall
parallelism; matchmaking collisions across replicas are harmless — two pods
occasionally picking the same high-uncertainty pair just means more samples on
a matchup we already care about.
"""

from __future__ import annotations

import math
import os
import random
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from . import storage
from .config import AsyncConfig
from .logging_setup import log_event, setup_json_logging
from .elo import (
    MctsPlayer,
    MinimaxPlayer,
    Player,
    baselines,
    format_elo_table,
    new_rating,
    play_game,
    predict_draw,
    update_ratings,
)

# Tuple, not set: ordering matters for `active != state["active_players"]`
# equality checks (frozenset iteration order is stable within a process but
# a hidden invariant we'd rather make explicit).
BASELINE_NAMES: tuple[str, ...] = ("Heuristic", "Minimax-2", "Minimax-3", "Minimax-4")
GATE_MIN_GAMES = 20
GATE_MAX_GAMES = 100
GATE_PASS_SCORE = 0.55
GATE_Z_VALUE = 1.96
GATE_PAIR_GAME_COUNT = 2
GATE_SPRT_P0 = 0.50
GATE_SPRT_P1 = 0.55
GATE_SPRT_ALPHA = 0.05
GATE_SPRT_BETA = 0.05
GATE_PAIR_BUCKETS: tuple[str, ...] = ("2.0", "1.5", "1.0", "0.5", "0.0")
GATE_PAIR_PSEUDOCOUNT = 0.25
GATE_PAIR_BUCKET_SCORES: dict[str, float] = {
    "2.0": 1.0,
    "1.5": 0.75,
    "1.0": 0.5,
    "0.5": 0.25,
    "0.0": 0.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pair_key(a: str, b: str) -> str:
    return ":".join(sorted([a, b]))


def _gate_pair_id() -> str:
    return f"{random.getrandbits(64):016x}"


def _gate_pair_bucket(pair_score: float) -> str:
    return f"{pair_score:.1f}"


def _candidate_points(outcome: str, *, candidate_is_white: bool) -> float:
    if outcome == "draw":
        return 0.5
    if outcome == "white":
        return 1.0 if candidate_is_white else 0.0
    return 0.0 if candidate_is_white else 1.0


def _sprt_bounds(
    alpha: float = GATE_SPRT_ALPHA,
    beta: float = GATE_SPRT_BETA,
) -> tuple[float, float]:
    return (
        math.log(beta / (1.0 - alpha)),
        math.log((1.0 - beta) / alpha),
    )


def _pair_bucket_mean(pair_buckets: dict[str, int]) -> tuple[float, int]:
    total_pairs = sum(int(pair_buckets.get(bucket, 0)) for bucket in GATE_PAIR_BUCKETS)
    if total_pairs <= 0:
        return 0.0, 0
    total_score = sum(
        GATE_PAIR_BUCKET_SCORES[bucket] * int(pair_buckets.get(bucket, 0))
        for bucket in GATE_PAIR_BUCKETS
    )
    return total_score / total_pairs, total_pairs


def _pair_bucket_support(pair_buckets: dict[str, int]) -> list[tuple[float, int]]:
    return [
        (GATE_PAIR_BUCKET_SCORES[bucket], int(pair_buckets.get(bucket, 0)))
        for bucket in GATE_PAIR_BUCKETS
        if int(pair_buckets.get(bucket, 0)) > 0
    ]


def _smoothed_pair_support(
    pair_buckets: dict[str, int],
    pseudocount: float = GATE_PAIR_PSEUDOCOUNT,
) -> list[tuple[float, float]]:
    return [
        (GATE_PAIR_BUCKET_SCORES[bucket], float(pair_buckets.get(bucket, 0)) + pseudocount)
        for bucket in GATE_PAIR_BUCKETS
    ]


def _empirical_likelihood_lambda(
    pair_buckets: dict[str, int],
    target_score: float,
    pseudocount: float = GATE_PAIR_PSEUDOCOUNT,
) -> float | None:
    support = _smoothed_pair_support(pair_buckets, pseudocount)
    if not support:
        return 0.0

    scores = [score for score, _count in support]
    min_score = min(scores)
    max_score = max(scores)
    tol = 1e-12
    if min_score == max_score:
        return 0.0 if abs(target_score - min_score) <= tol else None
    if target_score <= min_score + tol or target_score >= max_score - tol:
        return None

    total_pairs = sum(weight for _score, weight in support)
    empirical_mean = sum(score * count for score, count in support) / total_pairs
    if abs(empirical_mean - target_score) <= tol:
        return 0.0

    def g(lam: float) -> float:
        total = 0.0
        for score, count in support:
            delta = score - target_score
            total += count * delta / (1.0 + lam * delta)
        return total

    if target_score < empirical_mean:
        lower = 0.0
        upper = min(-1.0 / (score - target_score) for score, _count in support if score < target_score)
    else:
        lower = max(-1.0 / (score - target_score) for score, _count in support if score > target_score)
        upper = 0.0

    eps = 1e-12
    lo = lower + eps
    hi = upper - eps
    for _ in range(100):
        mid = (lo + hi) / 2.0
        value = g(mid)
        if abs(value) <= 1e-12:
            return mid
        if value > 0.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _empirical_likelihood_log_prob(
    pair_buckets: dict[str, int],
    target_score: float,
    pseudocount: float = GATE_PAIR_PSEUDOCOUNT,
) -> float:
    lam = _empirical_likelihood_lambda(pair_buckets, target_score, pseudocount)
    if lam is None:
        return float("-inf")
    support = _smoothed_pair_support(pair_buckets, pseudocount)
    total = 0.0
    for score, count in support:
        denom = 1.0 + lam * (score - target_score)
        if denom <= 0.0:
            return float("-inf")
        total -= count * math.log(denom)
    return total


def _gate_pentanomial_llr(
    pair_buckets: dict[str, int],
    p0: float = GATE_SPRT_P0,
    p1: float = GATE_SPRT_P1,
    pseudocount: float = GATE_PAIR_PSEUDOCOUNT,
) -> float:
    mean_score, total_pairs = _pair_bucket_mean(pair_buckets)
    if total_pairs <= 0:
        return 0.0
    log_p0 = _empirical_likelihood_log_prob(pair_buckets, p0, pseudocount)
    log_p1 = _empirical_likelihood_log_prob(pair_buckets, p1, pseudocount)
    return log_p1 - log_p0


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
        self._hits = 0
        self._misses = 0

    def _init_baselines(self) -> None:
        if self._baselines:
            return
        for p in baselines(self.simulations):
            self._baselines[p.name] = p
            self._cache[p.name] = p

    def get(self, name: str, s3_model_key: str | None = None) -> Player:
        self._init_baselines()

        if name in self._cache:
            self._hits += 1
            self._cache.move_to_end(name)
            return self._cache[name]
        self._misses += 1

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

    def stats(self) -> dict[str, float]:
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total else 0.0
        return {
            "size": float(len(self._cache)),
            "max_size": float(self.max_size),
            "hits": float(self._hits),
            "misses": float(self._misses),
            "hit_rate": hit_rate,
        }


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


def _read_approved_version() -> int:
    """Return the model version approved for self-play.

    Falls back to ``latest`` for compatibility with pre-gating runs.
    """
    for key in (storage.APPROVED_META, storage.LATEST_META):
        try:
            return int(storage.get_json(key).get("version", 0))
        except KeyError:
            continue
    return 0


def _load_gate_state(approved_version: int) -> dict:
    """Load gate state from S3, or return an empty default."""
    try:
        state = storage.get_json(storage.GATE_STATE)
    except KeyError:
        state = {}
    decisions = state.get("decisions")
    if not isinstance(decisions, dict):
        decisions = {}
    return {
        "approved_version": max(
            int(state.get("approved_version", 0)),
            approved_version,
        ),
        "decisions": decisions,
    }


def _pending_candidate(
    version_keys: dict[str, str],
    approved_version: int,
    gate_state: dict,
) -> str | None:
    """Return the next ungated candidate after the approved version."""
    for name in sorted(version_keys.keys(), key=lambda n: int(n[1:])):
        if int(name[1:]) <= approved_version:
            continue
        if gate_state["decisions"].get(name, {}).get("status") == "rejected":
            continue
        return name
    return None


def _desired_active(
    max_versions: int,
) -> tuple[list[str], dict[str, str], str | None, str | None, dict]:
    """Return active players, version key map, approved player, pending candidate, and gate state."""
    versions = _discover_versions()
    version_keys = {f"v{v}": key for v, key in versions}
    approved_version = _read_approved_version()
    gate_state = _load_gate_state(approved_version)
    approved_key = f"v{approved_version}"
    approved_name = (
        approved_key if approved_version > 0 and approved_key in version_keys else None
    )
    pending = _pending_candidate(version_keys, approved_version, gate_state)
    discovered_names = sorted(version_keys.keys(), key=lambda n: int(n[1:]), reverse=True)
    desired_models: list[str] = []

    # Always keep the incumbent and any pending gate candidate visible, then
    # retain a rolling window of the newest snapshots so recent trends remain
    # comparable in the active Elo pool.
    pinned = [name for name in (approved_name, pending) if name is not None]
    for name in pinned + discovered_names:
        if name in desired_models:
            continue
        desired_models.append(name)
        if max_versions > 0 and len(desired_models) >= max_versions:
            break
    active = list(BASELINE_NAMES) + desired_models
    return active, version_keys, approved_name, pending, gate_state


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


def _apply_record(state: dict, rec: dict) -> None:
    """Fold a single game record into ``state`` in place.

    Called both from the full replay path (``_build_state``) and incrementally
    after a local game, so the rating math lives in exactly one place.
    """
    white = rec.get("white")
    black = rec.get("black")
    outcome = rec.get("outcome")
    if not white or not black or outcome not in ("white", "black", "draw"):
        return

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


def _finalize_active(state: dict, active_players: list[str]) -> None:
    """Set active/retired player lists and ensure every active player has a rating."""
    ratings = state["ratings"]
    for name in active_players:
        if name not in ratings:
            ratings[name] = new_rating()
    state["active_players"] = list(active_players)
    state["retired_players"] = sorted(
        set(ratings.keys()) - set(active_players) - set(BASELINE_NAMES),
    )


def _build_state(records: list[dict], active_players: list[str]) -> dict:
    """Replay per-game records to produce the full Elo state projection."""
    state = _empty_state()
    # Stable total ordering: timestamp first, then payload fields as
    # tie-breakers so projection rebuilds are deterministic even when many
    # records share the same timestamp granularity.
    def _record_sort_key(r: dict) -> tuple:
        return (
            r.get("timestamp", "") or "",
            r.get("white", "") or "",
            r.get("black", "") or "",
            r.get("outcome", "") or "",
            int(r.get("moves", 0) or 0),
            int(r.get("white_moves", 0) or 0),
            int(r.get("black_moves", 0) or 0),
            float(r.get("white_time", 0.0) or 0.0),
            float(r.get("black_time", 0.0) or 0.0),
        )

    for rec in sorted(records, key=_record_sort_key):
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
        remote = storage.list_game_record_keys()
        new_keys = [k for k in remote if k not in self._records]
        if not new_keys:
            return 0
        # S3 GETs are I/O-bound; serial fetches of thousands of records on
        # cold start were the service's startup bottleneck.
        from concurrent.futures import ThreadPoolExecutor

        def _fetch(k: str) -> tuple[str, dict | None]:
            try:
                return k, storage.get_json(k)
            except KeyError:
                return k, None

        added = 0
        with ThreadPoolExecutor(max_workers=16) as ex:
            for k, rec in ex.map(_fetch, new_keys):
                if rec is not None:
                    self._records[k] = rec
                    added += 1
        return added


# ---------------------------------------------------------------------------
# Matchmaking
# ---------------------------------------------------------------------------


# Stop dispatching pairs that already have this many games — diminishing
# returns on additional samples, the OpenSkill posterior is tight by then.
SATURATION_THRESHOLD = 30


def _pair_game_count(state: dict, a: str, b: str) -> int:
    r = state["pair_results"].get(_pair_key(a, b))
    if r is None:
        return 0
    return r["a_wins"] + r["b_wins"] + r["draws"]


def _placement_pair(state: dict) -> tuple[str, str] | None:
    """Force a brand-new model to play one game vs each baseline first.

    Returns (model, baseline) if any active model has zero prior games against
    one or more baselines; this anchors the new model's rating against the
    fixed reference points before it enters general matchmaking. Without this
    the predict_draw heuristic actively avoids new-model-vs-baseline pairings
    (large |Δμ| → low draw probability) and the new model only ever plays
    other near-mu peers, leaving its rating poorly calibrated.
    """
    candidates: list[tuple[str, str]] = []
    for name in state["active_players"]:
        if name in BASELINE_NAMES:
            continue
        for baseline in BASELINE_NAMES:
            if baseline not in state["active_players"]:
                continue
            if _pair_game_count(state, name, baseline) == 0:
                candidates.append((name, baseline))
    if not candidates:
        return None
    return random.choice(candidates)


def _gate_progress_from_records(
    records: list[dict],
    approved_name: str,
    candidate_name: str,
) -> dict:
    """Summarize completed paired-color gate mini-matches from immutable records."""
    pending_pairs: dict[str, dict[int, dict]] = {}
    for rec in records:
        if rec.get("gate_candidate") != candidate_name:
            continue
        if rec.get("gate_incumbent") != approved_name:
            continue
        pair_id = rec.get("gate_pair_id")
        pair_game_index = rec.get("gate_pair_game_index")
        if not pair_id or pair_game_index not in (0, 1):
            continue
        pair_bucket = pending_pairs.setdefault(str(pair_id), {})
        pair_bucket[int(pair_game_index)] = rec

    wins = 0
    losses = 0
    draws = 0
    total_score = 0.0
    total_games = 0
    completed_pairs = 0
    pair_buckets = {bucket: 0 for bucket in GATE_PAIR_BUCKETS}

    for pair_id, pair_games in pending_pairs.items():
        del pair_id
        if len(pair_games) != GATE_PAIR_GAME_COUNT:
            continue
        completed_pairs += 1
        pair_score = 0.0
        for idx in (0, 1):
            rec = pair_games[idx]
            candidate_is_white = rec.get("white") == candidate_name
            outcome = rec.get("outcome")
            pts = _candidate_points(outcome, candidate_is_white=candidate_is_white)
            pair_score += pts
            total_score += pts
            total_games += 1
            if pts == 1.0:
                wins += 1
            elif pts == 0.5:
                draws += 1
            else:
                losses += 1
        pair_buckets[_gate_pair_bucket(pair_score)] += 1

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": total_games,
        "score": (total_score / total_games) if total_games else 0.0,
        "total_score": total_score,
        "completed_pairs": completed_pairs,
        "pair_buckets": pair_buckets,
    }


def _gate_score_stats(wins: int, losses: int, draws: int) -> tuple[int, float, float]:
    """Return ``(games, score, ci_half_width)`` for candidate gate results."""
    total = wins + losses + draws
    if total <= 0:
        return 0, 0.0, 1.0

    score = (wins + 0.5 * draws) / total
    if total == 1:
        return total, score, 1.0

    # Treat outcomes as scalar scores in {1.0, 0.5, 0.0} and compute the
    # unbiased sample variance explicitly. This is clearer than the
    # sum-of-squares shortcut and keeps the loss term visible instead of
    # relying on the fact that losses contribute 0^2 to Σx².
    sum_sq_dev = (
        wins * (1.0 - score) ** 2
        + draws * (0.5 - score) ** 2
        + losses * score**2
    )
    sample_var = max(0.0, sum_sq_dev / (total - 1))
    half_width = GATE_Z_VALUE * math.sqrt(sample_var / total)
    return total, score, half_width


def _gate_decision(wins: int, losses: int, draws: int) -> tuple[str | None, int, float, float]:
    """Return ``(decision, games, score, ci_half_width)`` for the current gate.

    ``decision`` is ``approved``, ``rejected``, or ``None`` if the sequential
    test is still inconclusive.
    """
    total, score, half_width = _gate_score_stats(wins, losses, draws)
    if total < GATE_MIN_GAMES:
        return None, total, score, half_width

    lower = score - half_width
    upper = score + half_width
    if lower >= GATE_PASS_SCORE:
        return "approved", total, score, half_width
    if upper < GATE_PASS_SCORE:
        return "rejected", total, score, half_width
    if total >= GATE_MAX_GAMES:
        return ("approved" if score >= GATE_PASS_SCORE else "rejected"), total, score, half_width
    return None, total, score, half_width


def _gate_sprt_decision(
    pair_buckets: dict[str, int],
    total_games: int,
) -> tuple[str | None, float, float, float]:
    """Return ``(decision, llr, lower_bound, upper_bound)`` for the paired GSPRT.

    This treats each paired-color mini-match as one pentanomial observation and
    evaluates the null/alternative score hypotheses via a generalized
    likelihood ratio over the five pair buckets.
    """
    lower, upper = _sprt_bounds()
    llr = _gate_pentanomial_llr(pair_buckets)
    if total_games < GATE_MIN_GAMES:
        return None, llr, lower, upper
    if llr >= upper:
        return "approved", llr, lower, upper
    if llr <= lower:
        return "rejected", llr, lower, upper
    if total_games >= GATE_MAX_GAMES:
        mean_score, _total_pairs = _pair_bucket_mean(pair_buckets)
        return ("approved" if mean_score >= GATE_PASS_SCORE else "rejected"), llr, lower, upper
    return None, llr, lower, upper


def _maybe_resolve_gate(
    records: list[dict],
    version_keys: dict[str, str],
    approved_name: str | None,
    candidate_name: str | None,
    gate_state: dict,
) -> tuple[dict, bool]:
    """Approve or reject the current candidate once the direct gate finishes."""
    if approved_name is None or candidate_name is None:
        return gate_state, False
    if candidate_name in gate_state["decisions"]:
        return gate_state, False

    progress = _gate_progress_from_records(records, approved_name, candidate_name)
    status, llr, sprt_lower, sprt_upper = _gate_sprt_decision(
        progress["pair_buckets"],
        progress["games"],
    )
    total, score, half_width = _gate_score_stats(
        progress["wins"],
        progress["losses"],
        progress["draws"],
    )
    if status is None:
        return gate_state, False

    resolved_at = datetime.now(timezone.utc).isoformat()
    gate_state["decisions"][candidate_name] = {
        "status": status,
        "approved_against": approved_name,
        "wins": progress["wins"],
        "losses": progress["losses"],
        "draws": progress["draws"],
        "games": progress["games"],
        "score": progress["score"],
        "ci_half_width": half_width,
        "completed_pairs": progress["completed_pairs"],
        "pair_buckets": progress["pair_buckets"],
        "sprt_llr": llr,
        "sprt_lower_bound": sprt_lower,
        "sprt_upper_bound": sprt_upper,
        "gate_mode": "paired_pentanomial_gsprt",
        "resolved_at": resolved_at,
    }
    if status == "approved":
        version = int(candidate_name[1:])
        storage.copy(version_keys[candidate_name], storage.APPROVED_ONNX)
        storage.put_json(
            storage.APPROVED_META,
            {
                "version": version,
                "timestamp": resolved_at,
            },
        )
        gate_state["approved_version"] = version

    storage.put_json(storage.GATE_STATE, gate_state)
    logger.info(
        "Gate resolved: {} {} vs {} (wins={} losses={} draws={} score={:.3f} +/- {:.3f}, pairs={}, llr={:.3f})",
        candidate_name,
        status,
        approved_name,
        progress["wins"],
        progress["losses"],
        progress["draws"],
        progress["score"],
        half_width,
        progress["completed_pairs"],
        llr,
    )
    return gate_state, True


def _select_pair(
    state: dict,
    *,
    approved_name: str | None = None,
    gate_candidate: str | None = None,
) -> tuple[str, str] | None:
    """Placement → uncertainty exploration → predict_draw matchmaking.

    Priority order:
    0. Gating: if an ungated candidate exists, play it directly against the
       approved self-play model until the gate resolves.
    1. Placement: brand-new models play 1 game vs each baseline first.
    2. Uncertainty: if any player has σ > 4, pair them with a well-measured
       opponent (lowest σ) to shrink the wide CI fastest. Without this,
       predict_draw actively avoids new models because large |Δμ| → low
       draw probability → never selected. Observed: v6/v7 got stuck at
       σ=6+ for hours while v4/v5 kept getting re-matched.
    3. predict_draw: sample uniformly from the top-5% most-likely-draw pairs
       among unsaturated matchups.
    """
    players = state["active_players"]
    if len(players) < 2:
        return None

    if (
        approved_name is not None
        and gate_candidate is not None
        and approved_name in players
        and gate_candidate in players
        and _pair_game_count(state, approved_name, gate_candidate) < GATE_MAX_GAMES
    ):
        return (approved_name, gate_candidate)

    pair = _placement_pair(state)
    if pair is not None:
        return pair

    # Uncertainty exploration: prioritize highest-σ player.
    ratings = state["ratings"]
    uncertain = [
        (ratings[p].get("sigma", 0), p)
        for p in players
        if p in ratings and ratings[p].get("sigma", 0) > 4
    ]
    if uncertain:
        uncertain.sort(reverse=True)
        target = uncertain[0][1]
        # Pair with the lowest-σ opponent (most informative matchup).
        opponents = [
            (ratings[p].get("sigma", 99), p)
            for p in players
            if p != target and p in ratings
            and _pair_game_count(state, target, p) < SATURATION_THRESHOLD
        ]
        if opponents:
            opponents.sort()
            return (target, opponents[0][1])

    candidates: list[tuple[float, str, str]] = []
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            a, b = players[i], players[j]
            if _pair_game_count(state, a, b) >= SATURATION_THRESHOLD:
                continue
            score = predict_draw(ratings[a], ratings[b])
            candidates.append((score, a, b))

    if not candidates:
        return None

    max_score = max(s for s, _, _ in candidates)
    top = [(a, b) for s, a, b in candidates if s >= max_score * 0.95]
    return random.choice(top)


def _assign_colors(state: dict, a: str, b: str) -> tuple[str, str]:
    result = state["pair_results"].get(_pair_key(a, b))
    if result is None:
        return (a, b) if random.random() < 0.5 else (b, a)

    a_as_white = result.get("a_as_white", 0)
    b_as_white = result.get("b_as_white", 0)

    if a_as_white <= b_as_white:
        return a, b
    return b, a


def _make_game_record(
    white_name: str,
    black_name: str,
    result: dict,
    *,
    gate_candidate: str | None = None,
    gate_incumbent: str | None = None,
    gate_pair_id: str | None = None,
    gate_pair_game_index: int | None = None,
) -> dict:
    record = {
        "white": white_name,
        "black": black_name,
        "outcome": result["outcome"],
        "moves": result["moves"],
        "white_time": result["white_time"],
        "black_time": result["black_time"],
        "white_moves": result["white_moves"],
        "black_moves": result["black_moves"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if gate_candidate is not None:
        record["gate_candidate"] = gate_candidate
    if gate_incumbent is not None:
        record["gate_incumbent"] = gate_incumbent
    if gate_pair_id is not None:
        record["gate_pair_id"] = gate_pair_id
    if gate_pair_game_index is not None:
        record["gate_pair_game_index"] = gate_pair_game_index
    return record


def _persist_record(records: GameRecordStore, state: dict, record: dict) -> bool:
    try:
        key = storage.put_game_record(record)
        records.add_local(key, record)
    except Exception as e:
        logger.error("Failed to persist game record: {}", e)
        return False
    _apply_record(state, record)
    return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


PLAYER_CACHE_SIZE = 6

# How often to re-LIST per-game objects from S3 to absorb peer replica writes.
# Kept short so the saturation cap and predict_draw matchmaking see roughly
# global game counts: with N replicas and a 5-minute lag, a near-saturated
# pair could be oversampled ~N× before any replica notices. 60s keeps that
# slack to a single extra game per replica per pair.
PEER_SYNC_INTERVAL_SECONDS = 60

# How often a replica writes the derived state/elo.json projection for the
# dashboard. Each write is preceded by a peer-record refresh so the projection
# reflects the freshest possible game log; races across replicas are then
# benign because every writer is computing from (essentially) the same input.
PROJECTION_INTERVAL_SECONDS = 120


def run_elo_service(
    simulations: int = 800,
    max_versions: int = 5,
    notify_interval: int = 20,
) -> None:
    """Run the continuous Elo rating service.

    Ordinary matchmaking plays one game per loop. An active promotion gate is
    evaluated as a paired-color two-game mini-match on the same replica.
    """
    cfg = AsyncConfig()
    cfg.ensure_cache_dirs()
    setup_json_logging("elo", run_id=cfg.run_id)
    log_event("elo.start", run_id=cfg.run_id, simulations=simulations)

    cache_dir = cfg.model_cache_dir / "elo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    players_cache = PlayerCache(simulations, cache_dir, max_size=PLAYER_CACHE_SIZE)
    records = GameRecordStore()

    # Slack notifications should fire from at most one replica; gate on an
    # env var the operator sets on exactly one pod.
    slack_leader = os.environ.get("ELO_SLACK_LEADER") == "1"
    last_slack_game = 0

    logger.info(
        "Elo service starting (simulations={}, max_versions={}, slack_leader={})",
        simulations, max_versions, slack_leader,
    )

    records.refresh()
    (
        active,
        version_keys,
        approved_name,
        gate_candidate,
        gate_state,
    ) = _desired_active(max_versions)
    state = _build_state(records.all_records(), active)
    logger.info(
        "Initial state: {} games, {} active players, approved={}, candidate={}",
        state["total_games"],
        len(state["active_players"]),
        approved_name,
        gate_candidate,
    )

    last_peer_sync_ts = time.monotonic()
    last_projection_ts = 0.0

    def _sync_peer_records() -> None:
        """LIST + fetch new peer records, replay full state if any arrived."""
        nonlocal state, last_peer_sync_ts
        try:
            added = records.refresh()
            if added:
                state = _build_state(records.all_records(), active)
                logger.info(
                    "Sync: merged {} peer records ({} total, {} active)",
                    added, state["total_games"],
                    len(state["active_players"]),
                )
            last_peer_sync_ts = time.monotonic()
        except Exception as e:
            logger.warning("Peer sync failed: {}", e)

    def _refresh_gate_view_if_ready() -> None:
        """Refresh peer results before resolving a completed gate."""
        nonlocal state, approved_name, gate_candidate, gate_state, active, version_keys
        if approved_name is None or gate_candidate is None:
            return
        progress = _gate_progress_from_records(
            records.all_records(),
            approved_name,
            gate_candidate,
        )
        if progress["games"] < GATE_MIN_GAMES:
            return
        _sync_peer_records()
        (
            active,
            version_keys,
            approved_name,
            gate_candidate,
            gate_state,
        ) = _desired_active(max_versions)
        state = _build_state(records.all_records(), active)

    while True:
        # Active player set is cheap to recompute (one S3 LIST) and we want
        # to pick up newly-promoted models without waiting for the peer sync.
        (
            active,
            version_keys,
            approved_name,
            gate_candidate,
            gate_state,
        ) = _desired_active(max_versions)
        if active != state["active_players"]:
            _finalize_active(state, active)

        _refresh_gate_view_if_ready()
        gate_state, gate_changed = _maybe_resolve_gate(
            records.all_records(),
            version_keys,
            approved_name,
            gate_candidate,
            gate_state,
        )
        if gate_changed:
            (
                active,
                version_keys,
                approved_name,
                gate_candidate,
                gate_state,
            ) = _desired_active(max_versions)
            state = _build_state(records.all_records(), active)
            continue

        if time.monotonic() - last_peer_sync_ts >= PEER_SYNC_INTERVAL_SECONDS:
            _sync_peer_records()

        if len(state["active_players"]) < 2:
            logger.info(
                "Waiting for models... ({} players)",
                len(state["active_players"]),
            )
            time.sleep(60)
            continue

        pair = _select_pair(
            state,
            approved_name=approved_name,
            gate_candidate=gate_candidate,
        )
        if pair is None:
            logger.info("No dispatchable pairs, waiting...")
            time.sleep(60)
            continue

        a, b = pair
        is_gate_pair = (
            approved_name is not None
            and gate_candidate is not None
            and {a, b} == {approved_name, gate_candidate}
        )

        if is_gate_pair:
            gate_pair_id = _gate_pair_id()
            logger.info(
                "Dispatch gate pair {}: {} vs {} (2 games, {} prior games)",
                gate_pair_id,
                gate_candidate,
                approved_name,
                _pair_game_count(state, a, b),
            )
            gate_games = [
                (gate_candidate, approved_name, 0),
                (approved_name, gate_candidate, 1),
            ]
            gate_failed = False
            for white_name, black_name, gate_game_index in gate_games:
                try:
                    white = players_cache.get(white_name, version_keys.get(white_name))
                    black = players_cache.get(black_name, version_keys.get(black_name))
                    result = play_game(white, black)
                except Exception as e:
                    logger.error("Gate game {} vs {} failed: {}", white_name, black_name, e)
                    gate_failed = True
                    break

                outcome = result["outcome"]
                w_avg = result["white_time"] / max(result["white_moves"], 1)
                b_avg = result["black_time"] / max(result["black_moves"], 1)
                logger.info(
                    "  Gate game {} result: {} ({} moves) | {} {:.2f}s/move | {} {:.2f}s/move",
                    gate_game_index + 1,
                    outcome,
                    result["moves"],
                    white_name,
                    w_avg,
                    black_name,
                    b_avg,
                )
                record = _make_game_record(
                    white_name,
                    black_name,
                    result,
                    gate_candidate=gate_candidate,
                    gate_incumbent=approved_name,
                    gate_pair_id=gate_pair_id,
                    gate_pair_game_index=gate_game_index,
                )
                if not _persist_record(records, state, record):
                    gate_failed = True
                    break

            if gate_failed:
                time.sleep(5)
                continue
        else:
            white_name, black_name = _assign_colors(state, a, b)
            logger.info(
                "Dispatch game {}: {} (W) vs {} (B) [pair has {} prior games]",
                state["total_games"] + 1, white_name, black_name,
                _pair_game_count(state, a, b),
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
            record = _make_game_record(white_name, black_name, result)
            if not _persist_record(records, state, record):
                continue

        _refresh_gate_view_if_ready()
        gate_state, gate_changed = _maybe_resolve_gate(
            records.all_records(),
            version_keys,
            approved_name,
            gate_candidate,
            gate_state,
        )
        if gate_changed:
            (
                active,
                version_keys,
                approved_name,
                gate_candidate,
                gate_state,
            ) = _desired_active(max_versions)
            state = _build_state(records.all_records(), active)

        if state["total_games"] % 10 == 0:
            logger.info(
                "Ratings (game {}):\n{}",
                state["total_games"],
                format_elo_table(state["ratings"]),
            )
            cs = players_cache.stats()
            logger.info(
                "Player cache: size={:.0f}/{:.0f} hit_rate={:.1%} hits={:.0f} misses={:.0f}",
                cs["size"], cs["max_size"], cs["hit_rate"], cs["hits"], cs["misses"],
            )

        # Periodically write the derived projection for the dashboard/metrics.
        # Refresh peer records first so the projection reflects everyone's
        # games — otherwise concurrent replicas overwrite elo.json with
        # strict subsets of the global game log and the dashboard flickers.
        if time.monotonic() - last_projection_ts >= PROJECTION_INTERVAL_SECONDS:
            _sync_peer_records()
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
