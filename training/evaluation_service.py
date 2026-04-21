"""Continuous candidate evaluation service.

The service evaluates the newest candidate model against the currently
approved model plus a fixed anchor suite. All outputs are version-scoped:

    state/evals/v{N}/gate_summary.json
    state/evals/v{N}/benchmark_summary.json
    state/evals/v{N}/decision.json
    state/evals/v{N}/games/{ts}_{rand}.json

There is no standing rating pool, no cross-version league, and no global Elo
projection. Promotion is driven only by direct candidate-vs-approved gate
results plus benchmark non-regression checks against anchors.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

from loguru import logger

from . import storage
from .config import AsyncConfig
from .elo import MctsPlayer, Player, baselines, play_game
from .logging_setup import log_event, setup_json_logging

ANCHOR_NAMES: tuple[str, ...] = ("Heuristic", "Minimax-2", "Minimax-3", "Minimax-4")
PAIR_GAME_COUNT = 2
PAIR_BUCKETS: tuple[str, ...] = ("2.0", "1.5", "1.0", "0.5", "0.0")
PAIR_BUCKET_SCORES: dict[str, float] = {
    "2.0": 1.0,
    "1.5": 0.75,
    "1.0": 0.5,
    "0.5": 0.25,
    "0.0": 0.0,
}
PAIR_PSEUDOCOUNT = 0.25

GATE_MIN_GAMES = 20
GATE_MAX_GAMES = 100
GATE_PASS_SCORE = 0.55
GATE_SPRT_P0 = 0.50
GATE_SPRT_P1 = 0.55
GATE_SPRT_ALPHA = 0.05
GATE_SPRT_BETA = 0.05
GATE_Z_VALUE = NormalDist().inv_cdf(1.0 - GATE_SPRT_ALPHA / 2.0)

BENCHMARK_MIN_GAMES = 12
BENCHMARK_MAX_GAMES = 24
BENCHMARK_REGRESSION_TOLERANCE = 0.05
BENCHMARK_SPRT_MARGIN = 0.03

OPENING_RANDOM_PLIES = 2
MODEL_CACHE_LIMIT = 2
SYNC_INTERVAL_SECONDS = 60
IDLE_SECONDS = 60
PROMOTION_LEASE_STALE_SECONDS = 15 * 60


def _pair_bucket(pair_score: float) -> str:
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
    total_pairs = sum(int(pair_buckets.get(bucket, 0)) for bucket in PAIR_BUCKETS)
    if total_pairs <= 0:
        return 0.0, 0
    total_score = sum(
        PAIR_BUCKET_SCORES[bucket] * int(pair_buckets.get(bucket, 0))
        for bucket in PAIR_BUCKETS
    )
    return total_score / total_pairs, total_pairs


def _smoothed_pair_support(
    pair_buckets: dict[str, int],
    pseudocount: float = PAIR_PSEUDOCOUNT,
) -> list[tuple[float, float]]:
    return [
        (PAIR_BUCKET_SCORES[bucket], float(pair_buckets.get(bucket, 0)) + pseudocount)
        for bucket in PAIR_BUCKETS
    ]


def _empirical_likelihood_lambda(
    pair_buckets: dict[str, int],
    target_score: float,
    pseudocount: float = PAIR_PSEUDOCOUNT,
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
        upper = min(
            -1.0 / (score - target_score)
            for score, _count in support
            if score < target_score
        )
    else:
        lower = max(
            -1.0 / (score - target_score)
            for score, _count in support
            if score > target_score
        )
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
    pseudocount: float = PAIR_PSEUDOCOUNT,
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


def _finite(x: float, fallback: float = 0.0) -> float:
    return x if math.isfinite(x) else fallback


def _gate_pentanomial_llr(
    pair_buckets: dict[str, int],
    p0: float = GATE_SPRT_P0,
    p1: float = GATE_SPRT_P1,
    pseudocount: float = PAIR_PSEUDOCOUNT,
) -> float:
    mean_score, total_pairs = _pair_bucket_mean(pair_buckets)
    if total_pairs <= 0:
        return 0.0
    log_p0 = _empirical_likelihood_log_prob(pair_buckets, p0, pseudocount)
    log_p1 = _empirical_likelihood_log_prob(pair_buckets, p1, pseudocount)
    return log_p1 - log_p0


def _discover_versions() -> list[tuple[int, str]]:
    versions = []
    for key in storage.ls(storage.VERSIONS_PREFIX):
        name = key.rsplit("/", 1)[-1]
        if not name.endswith(".onnx"):
            continue
        try:
            versions.append((int(name[:-5]), key))
        except ValueError:
            continue
    versions.sort(key=lambda item: item[0])
    return versions


def _read_approved_version() -> int:
    for key in (storage.APPROVED_META, storage.LATEST_META):
        try:
            return int(storage.get_json(key).get("version", 0))
        except KeyError:
            continue
    return 0


def _latest_candidate(versions: dict[int, str], approved_version: int) -> int | None:
    candidates = [version for version in versions if version > approved_version]
    return max(candidates) if candidates else None


def _score_stats(wins: int, losses: int, draws: int) -> tuple[int, float, float]:
    total = wins + losses + draws
    if total <= 0:
        return 0, 0.0, 1.0
    score = (wins + 0.5 * draws) / total
    if total == 1:
        return total, score, 1.0
    sum_sq_dev = (
        wins * (1.0 - score) ** 2
        + draws * (0.5 - score) ** 2
        + losses * score**2
    )
    sample_var = max(0.0, sum_sq_dev / (total - 1))
    half_width = GATE_Z_VALUE * math.sqrt(sample_var / total)
    return total, score, half_width


def _gate_decision(wins: int, losses: int, draws: int) -> tuple[str | None, int, float, float]:
    total, score, half_width = _score_stats(wins, losses, draws)
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


def _paired_sprt_decision(
    pair_buckets: dict[str, int],
    *,
    total_games: int,
    min_games: int,
    max_games: int,
    p0: float,
    p1: float,
    final_threshold: float | None = None,
    alpha: float = GATE_SPRT_ALPHA,
    beta: float = GATE_SPRT_BETA,
) -> tuple[str | None, float, float, float]:
    lower, upper = _sprt_bounds(alpha, beta)
    llr = _gate_pentanomial_llr(pair_buckets, p0=p0, p1=p1)
    if total_games < min_games:
        return None, llr, lower, upper
    if llr >= upper:
        return "approved", llr, lower, upper
    if llr <= lower:
        return "rejected", llr, lower, upper
    if total_games >= max_games:
        mean_score, _ = _pair_bucket_mean(pair_buckets)
        threshold = p1 if final_threshold is None else final_threshold
        return ("approved" if mean_score >= threshold else "rejected"), llr, lower, upper
    return None, llr, lower, upper


def _gate_sprt_decision(
    pair_buckets: dict[str, int],
    total_games: int,
) -> tuple[str | None, float, float, float]:
    return _paired_sprt_decision(
        pair_buckets,
        total_games=total_games,
        min_games=GATE_MIN_GAMES,
        max_games=GATE_MAX_GAMES,
        p0=GATE_SPRT_P0,
        p1=GATE_SPRT_P1,
    )


def _threshold_decision(
    *,
    wins: int,
    losses: int,
    draws: int,
    target_score: float,
    min_games: int,
    max_games: int,
) -> tuple[str | None, int, float, float]:
    total, score, half_width = _score_stats(wins, losses, draws)
    if total < min_games:
        return None, total, score, half_width
    lower = score - half_width
    upper = score + half_width
    if lower >= target_score:
        return "approved", total, score, half_width
    if upper < target_score:
        return "rejected", total, score, half_width
    if total >= max_games:
        return ("approved" if score >= target_score else "rejected"), total, score, half_width
    return None, total, score, half_width


def _series_progress(records: list[dict], *, candidate: str, opponent: str) -> dict:
    pair_games: dict[str, dict[int, dict]] = {}
    wins = losses = draws = 0
    candidate_white = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    candidate_black = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    termination_mix: dict[str, int] = defaultdict(int)
    candidate_total_time = 0.0
    opponent_total_time = 0.0
    candidate_total_moves = 0
    opponent_total_moves = 0
    total_moves = 0

    for rec in records:
        if rec.get("candidate") != candidate or rec.get("opponent") != opponent:
            continue
        outcome = rec.get("outcome")
        if outcome not in ("white", "black", "draw"):
            continue
        candidate_is_white = rec.get("white") == candidate
        pts = _candidate_points(outcome, candidate_is_white=candidate_is_white)
        if pts == 1.0:
            wins += 1
            bucket = candidate_white if candidate_is_white else candidate_black
            bucket["wins"] += 1
        elif pts == 0.5:
            draws += 1
            bucket = candidate_white if candidate_is_white else candidate_black
            bucket["draws"] += 1
        else:
            losses += 1
            bucket = candidate_white if candidate_is_white else candidate_black
            bucket["losses"] += 1

        bucket["games"] += 1
        termination_mix[str(rec.get("termination") or "unknown")] += 1
        total_moves += int(rec.get("moves", 0) or 0)

        if candidate_is_white:
            candidate_total_time += float(rec.get("white_time", 0.0) or 0.0)
            opponent_total_time += float(rec.get("black_time", 0.0) or 0.0)
            candidate_total_moves += int(rec.get("white_moves", 0) or 0)
            opponent_total_moves += int(rec.get("black_moves", 0) or 0)
        else:
            candidate_total_time += float(rec.get("black_time", 0.0) or 0.0)
            opponent_total_time += float(rec.get("white_time", 0.0) or 0.0)
            candidate_total_moves += int(rec.get("black_moves", 0) or 0)
            opponent_total_moves += int(rec.get("white_moves", 0) or 0)

        pair_id = rec.get("pair_id")
        pair_game_index = rec.get("pair_game_index")
        if pair_id is not None and pair_game_index in (0, 1):
            pair_games.setdefault(str(pair_id), {})[int(pair_game_index)] = rec

    pair_buckets = {bucket: 0 for bucket in PAIR_BUCKETS}
    completed_pairs = 0
    pending_pairs = 0
    for games in pair_games.values():
        if len(games) != PAIR_GAME_COUNT:
            pending_pairs += 1
            continue
        completed_pairs += 1
        pair_score = 0.0
        for index in (0, 1):
            rec = games[index]
            candidate_is_white = rec.get("white") == candidate
            pair_score += _candidate_points(
                rec.get("outcome"),
                candidate_is_white=candidate_is_white,
            )
        pair_buckets[_pair_bucket(pair_score)] += 1

    games = wins + losses + draws
    score = (wins + 0.5 * draws) / games if games else 0.0
    confidence_games, confidence_score, half_width = _score_stats(wins, losses, draws)
    del confidence_games, confidence_score
    avg_game_length = total_moves / games if games else 0.0
    candidate_spm = candidate_total_time / candidate_total_moves if candidate_total_moves else 0.0
    opponent_spm = opponent_total_time / opponent_total_moves if opponent_total_moves else 0.0

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": games,
        "score": score,
        "completed_pairs": completed_pairs,
        "pending_pairs": pending_pairs,
        "pair_buckets": pair_buckets,
        "color_split": {
            "candidate_white": candidate_white,
            "candidate_black": candidate_black,
        },
        "confidence": {
            "score": score,
            "ci_half_width": half_width,
            "lower": score - half_width,
            "upper": score + half_width,
            "games": games,
        },
        "move_time_stats": {
            "candidate_total_seconds": round(candidate_total_time, 2),
            "opponent_total_seconds": round(opponent_total_time, 2),
            "candidate_total_moves": candidate_total_moves,
            "opponent_total_moves": opponent_total_moves,
            "candidate_seconds_per_move": round(candidate_spm, 3),
            "opponent_seconds_per_move": round(opponent_spm, 3),
            "average_game_length_moves": round(avg_game_length, 2),
        },
        "termination_mix": dict(sorted(termination_mix.items())),
    }


def _empty_progress() -> dict:
    return {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "games": 0,
        "score": 0.0,
        "completed_pairs": 0,
        "pending_pairs": 0,
        "pair_buckets": {bucket: 0 for bucket in PAIR_BUCKETS},
        "color_split": {
            "candidate_white": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
            "candidate_black": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
        },
        "confidence": {
            "score": 0.0,
            "ci_half_width": 1.0,
            "lower": -1.0,
            "upper": 1.0,
            "games": 0,
        },
        "move_time_stats": {
            "candidate_total_seconds": 0.0,
            "opponent_total_seconds": 0.0,
            "candidate_total_moves": 0,
            "opponent_total_moves": 0,
            "candidate_seconds_per_move": 0.0,
            "opponent_seconds_per_move": 0.0,
            "average_game_length_moves": 0.0,
        },
        "termination_mix": {},
    }


def _merge_progress(progresses: list[dict]) -> dict:
    if not progresses:
        return _empty_progress()

    wins = sum(p["wins"] for p in progresses)
    losses = sum(p["losses"] for p in progresses)
    draws = sum(p["draws"] for p in progresses)
    pair_buckets = {
        bucket: sum(p["pair_buckets"].get(bucket, 0) for p in progresses)
        for bucket in PAIR_BUCKETS
    }

    termination_mix: dict[str, int] = defaultdict(int)
    candidate_white = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    candidate_black = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    move_time_stats = {
        "candidate_total_seconds": 0.0,
        "opponent_total_seconds": 0.0,
        "candidate_total_moves": 0,
        "opponent_total_moves": 0,
        "average_game_length_moves_weighted": 0.0,
    }
    total_games = 0

    for progress in progresses:
        total_games += progress["games"]
        for key in ("candidate_white", "candidate_black"):
            src = progress["color_split"][key]
            dst = candidate_white if key == "candidate_white" else candidate_black
            for field in ("games", "wins", "losses", "draws"):
                dst[field] += src[field]
        for key, value in progress["termination_mix"].items():
            termination_mix[key] += value
        mts = progress["move_time_stats"]
        move_time_stats["candidate_total_seconds"] += float(mts["candidate_total_seconds"])
        move_time_stats["opponent_total_seconds"] += float(mts["opponent_total_seconds"])
        move_time_stats["candidate_total_moves"] += int(mts["candidate_total_moves"])
        move_time_stats["opponent_total_moves"] += int(mts["opponent_total_moves"])
        move_time_stats["average_game_length_moves_weighted"] += (
            float(mts["average_game_length_moves"]) * progress["games"]
        )

    games, score, half_width = _score_stats(wins, losses, draws)
    avg_game_length = (
        move_time_stats["average_game_length_moves_weighted"] / total_games
        if total_games
        else 0.0
    )
    candidate_spm = (
        move_time_stats["candidate_total_seconds"] / move_time_stats["candidate_total_moves"]
        if move_time_stats["candidate_total_moves"]
        else 0.0
    )
    opponent_spm = (
        move_time_stats["opponent_total_seconds"] / move_time_stats["opponent_total_moves"]
        if move_time_stats["opponent_total_moves"]
        else 0.0
    )

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": games,
        "score": score,
        "completed_pairs": sum(p["completed_pairs"] for p in progresses),
        "pending_pairs": sum(p["pending_pairs"] for p in progresses),
        "pair_buckets": pair_buckets,
        "color_split": {
            "candidate_white": candidate_white,
            "candidate_black": candidate_black,
        },
        "confidence": {
            "score": score,
            "ci_half_width": half_width,
            "lower": score - half_width,
            "upper": score + half_width,
            "games": games,
        },
        "move_time_stats": {
            "candidate_total_seconds": round(move_time_stats["candidate_total_seconds"], 2),
            "opponent_total_seconds": round(move_time_stats["opponent_total_seconds"], 2),
            "candidate_total_moves": move_time_stats["candidate_total_moves"],
            "opponent_total_moves": move_time_stats["opponent_total_moves"],
            "candidate_seconds_per_move": round(candidate_spm, 3),
            "opponent_seconds_per_move": round(opponent_spm, 3),
            "average_game_length_moves": round(avg_game_length, 2),
        },
        "termination_mix": dict(sorted(termination_mix.items())),
    }


def _build_gate_summary(
    *,
    candidate_version: int,
    approved_version: int,
    records: list[dict],
) -> dict:
    candidate = f"v{candidate_version}"
    approved = f"v{approved_version}"
    progress = _series_progress(records, candidate=candidate, opponent=approved)
    status, llr, lower_bound, upper_bound = _gate_sprt_decision(
        progress["pair_buckets"],
        progress["games"],
    )
    fallback_status, _games, _score, _half_width = _gate_decision(
        progress["wins"],
        progress["losses"],
        progress["draws"],
    )
    decision = status or fallback_status or "pending"
    return {
        "eval_type": "gate",
        "candidate": candidate,
        "candidate_version": candidate_version,
        "approved": approved,
        "approved_version": approved_version,
        "opponent": approved,
        "status": decision,
        **progress,
        "sprt": {
            "status": "approved" if status == "approved" else "rejected" if status == "rejected" else "pending",
            "llr": _finite(llr),
            "lower_bound": _finite(lower_bound),
            "upper_bound": _finite(upper_bound),
            "null_score": GATE_SPRT_P0,
            "alt_score": GATE_SPRT_P1,
        },
        "promotion_target_score": GATE_PASS_SCORE,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_reference_scores(approved_version: int) -> dict[str, float] | None:
    try:
        summary = storage.get_json(storage.eval_benchmark_summary_key(approved_version))
    except KeyError:
        return None
    per_opponent = summary.get("per_opponent", {})
    out = {}
    for opponent, info in per_opponent.items():
        try:
            out[opponent] = float(info["score"])
        except (KeyError, TypeError, ValueError):
            continue
    return out if out else None


def _build_benchmark_opponent_summary(
    *,
    candidate_version: int,
    approved_version: int,
    opponent: str,
    records: list[dict],
    reference_scores: dict[str, float] | None,
) -> dict:
    candidate = f"v{candidate_version}"
    progress = _series_progress(records, candidate=candidate, opponent=opponent)
    reference_score = None if reference_scores is None else reference_scores.get(opponent)

    if reference_score is None:
        target_score = None
        decision = "complete" if progress["games"] >= BENCHMARK_MAX_GAMES else "pending"
        sprt = {
            "status": "not_applicable",
            "llr": 0.0,
            "lower_bound": None,
            "upper_bound": None,
            "null_score": None,
            "alt_score": None,
        }
    else:
        target_score = round(
            max(0.0, reference_score - BENCHMARK_REGRESSION_TOLERANCE),
            6,
        )
        ci_status, _games, _score, _half_width = _threshold_decision(
            wins=progress["wins"],
            losses=progress["losses"],
            draws=progress["draws"],
            target_score=target_score,
            min_games=BENCHMARK_MIN_GAMES,
            max_games=BENCHMARK_MAX_GAMES,
        )
        alt_score = min(0.99, target_score + BENCHMARK_SPRT_MARGIN)
        sprt_status, llr, lower_bound, upper_bound = _paired_sprt_decision(
            progress["pair_buckets"],
            total_games=progress["games"],
            min_games=BENCHMARK_MIN_GAMES,
            max_games=BENCHMARK_MAX_GAMES,
            p0=target_score,
            p1=alt_score,
            final_threshold=target_score,
        )
        decision = sprt_status or ci_status or "pending"
        sprt = {
            "status": "approved" if sprt_status == "approved" else "rejected" if sprt_status == "rejected" else "pending",
            "llr": _finite(llr),
            "lower_bound": _finite(lower_bound),
            "upper_bound": _finite(upper_bound),
            "null_score": target_score,
            "alt_score": alt_score,
        }

    return {
        "opponent": opponent,
        "reference_version": approved_version if reference_score is not None else None,
        "reference_score": reference_score,
        "regression_tolerance": BENCHMARK_REGRESSION_TOLERANCE if reference_score is not None else None,
        "target_score": target_score,
        "status": decision,
        **progress,
        "sprt": sprt,
    }


def _build_benchmark_summary(
    *,
    candidate_version: int,
    approved_version: int,
    records: list[dict],
    reference_scores: dict[str, float] | None,
) -> dict:
    candidate = f"v{candidate_version}"
    per_opponent = {
        opponent: _build_benchmark_opponent_summary(
            candidate_version=candidate_version,
            approved_version=approved_version,
            opponent=opponent,
            records=records,
            reference_scores=reference_scores,
        )
        for opponent in ANCHOR_NAMES
    }
    aggregate = _merge_progress(list(per_opponent.values()))

    statuses = [summary["status"] for summary in per_opponent.values()]
    if reference_scores is None:
        overall_status = (
            "complete"
            if all(status == "complete" for status in statuses)
            else "pending"
        )
    elif any(status == "rejected" for status in statuses):
        overall_status = "rejected"
    elif all(status == "approved" for status in statuses):
        overall_status = "approved"
    else:
        overall_status = "pending"

    return {
        "eval_type": "benchmark",
        "candidate": candidate,
        "candidate_version": candidate_version,
        "approved_version": approved_version,
        "status": overall_status,
        "reference_available": reference_scores is not None,
        "per_opponent": per_opponent,
        **aggregate,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _benchmark_collection_complete(benchmark_summary: dict) -> bool:
    if not benchmark_summary.get("reference_available", False):
        return benchmark_summary["status"] == "complete"
    return all(
        summary["status"] != "pending"
        for summary in benchmark_summary.get("per_opponent", {}).values()
    )


def _build_decision(
    *,
    candidate_version: int,
    approved_version: int,
    gate_summary: dict | None,
    benchmark_summary: dict,
) -> dict:
    candidate = f"v{candidate_version}"
    decision_status = "pending"
    reasons: list[str] = []
    promotion_eligible = True
    collection_complete = False

    if gate_summary is None:
        if benchmark_summary["status"] == "complete":
            decision_status = "baseline_ready"
            collection_complete = True
        else:
            decision_status = "pending"
            reasons.append("collecting anchor baseline for current approved model")
    else:
        gate_failed = gate_summary["status"] == "rejected"
        benchmark_failed = (
            benchmark_summary["reference_available"]
            and benchmark_summary["status"] == "rejected"
        )
        promotion_eligible = not gate_failed and not benchmark_failed
        gate_complete = gate_summary["status"] != "pending"
        benchmark_complete = _benchmark_collection_complete(benchmark_summary)
        collection_complete = gate_complete and benchmark_complete

        if gate_failed:
            reasons.append("candidate failed direct gate against approved model")
        if benchmark_failed:
            reasons.append("candidate regressed against fixed-anchor benchmark tolerance")

        if gate_summary["status"] == "approved" and benchmark_summary["status"] == "approved":
            decision_status = "promote"
            reasons.append("candidate passed gate and anchor regression checks")
        elif not promotion_eligible and collection_complete:
            decision_status = "rejected"
        elif not promotion_eligible:
            decision_status = "rejected_pending"
            reasons.append("continuing to collect remaining evaluation evidence")
        else:
            decision_status = "pending"
            if gate_summary["status"] != "approved":
                reasons.append("gate still collecting evidence")
            if benchmark_summary["status"] != "approved":
                reasons.append("benchmark suite still collecting evidence")

    return {
        "candidate": candidate,
        "candidate_version": candidate_version,
        "approved_version": approved_version,
        "gate_status": None if gate_summary is None else gate_summary["status"],
        "benchmark_status": benchmark_summary["status"],
        "promotion_eligible": promotion_eligible,
        "collection_complete": collection_complete,
        "status": decision_status,
        "reasons": reasons,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


class VersionRecordStore:
    """Caches immutable per-game records per evaluated version."""

    def __init__(self) -> None:
        self._records_by_version: dict[int, dict[str, dict]] = {}

    def add_local(self, version: int, key: str, record: dict) -> None:
        self._records_by_version.setdefault(version, {})[key] = record

    def refresh(self, version: int) -> int:
        current = self._records_by_version.setdefault(version, {})
        remote_keys = storage.list_eval_game_record_keys(version)
        new_keys = [key for key in remote_keys if key not in current]
        added = 0
        for key in new_keys:
            try:
                current[key] = storage.get_json(key)
                added += 1
            except KeyError:
                continue
        return added

    def records(self, version: int) -> list[dict]:
        return list(self._records_by_version.get(version, {}).values())


class PlayerProvider:
    """Very small model cache: only the active candidate/incumbent sessions."""

    def __init__(self, simulations: int, cache_dir: Path) -> None:
        self.simulations = simulations
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._anchors = {player.name: player for player in baselines(simulations)}
        self._models: dict[str, Player] = {}

    def get(self, name: str, model_key: str | None = None) -> Player:
        if name in self._anchors:
            return self._anchors[name]
        if name in self._models:
            return self._models[name]
        if model_key is None:
            raise KeyError(f"missing model key for {name}")

        while len(self._models) >= MODEL_CACHE_LIMIT:
            stale_name = next(iter(self._models))
            self._models.pop(stale_name, None)

        local_path = self.cache_dir / f"{name}.onnx"
        if not local_path.exists():
            storage.get_file(model_key, local_path)
        player = MctsPlayer(name=name, simulations=self.simulations, model_path=str(local_path))
        self._models[name] = player
        return player


def _pair_id() -> str:
    return f"{random.getrandbits(64):016x}"


def _make_record(
    *,
    eval_type: str,
    candidate_version: int,
    approved_version: int,
    opponent: str,
    white_name: str,
    black_name: str,
    result: dict,
    pair_id: str,
    pair_game_index: int,
    opening_seed: int,
) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_type": eval_type,
        "candidate": f"v{candidate_version}",
        "candidate_version": candidate_version,
        "approved_version": approved_version,
        "opponent": opponent,
        "white": white_name,
        "black": black_name,
        "outcome": result["outcome"],
        "termination": result.get("termination"),
        "moves": result["moves"],
        "white_time": result["white_time"],
        "black_time": result["black_time"],
        "white_moves": result["white_moves"],
        "black_moves": result["black_moves"],
        "pair_id": pair_id,
        "pair_game_index": pair_game_index,
        "opening_seed": opening_seed,
        "opening_plies": result.get("opening_plies", OPENING_RANDOM_PLIES),
    }


def _persist_record(records: VersionRecordStore, version: int, record: dict) -> str | None:
    try:
        key = storage.put_eval_game_record(version, record)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to persist evaluation record: {}", exc)
        return None
    records.add_local(version, key, record)
    return key


def _write_artifacts(
    *,
    version: int,
    gate_summary: dict | None,
    benchmark_summary: dict,
    decision: dict,
) -> None:
    if gate_summary is not None:
        storage.put_json(storage.eval_gate_summary_key(version), gate_summary)
    storage.put_json(storage.eval_benchmark_summary_key(version), benchmark_summary)
    storage.put_json(storage.eval_decision_key(version), decision)


def _refresh_artifacts(
    *,
    version: int,
    approved_version: int,
    records: VersionRecordStore,
    include_gate: bool,
) -> tuple[dict | None, dict, dict]:
    version_records = records.records(version)
    gate_summary = (
        _build_gate_summary(
            candidate_version=version,
            approved_version=approved_version,
            records=version_records,
        )
        if include_gate
        else None
    )
    reference_scores = None if version == approved_version else _load_reference_scores(approved_version)
    benchmark_summary = _build_benchmark_summary(
        candidate_version=version,
        approved_version=approved_version,
        records=version_records,
        reference_scores=reference_scores,
    )
    decision = _build_decision(
        candidate_version=version,
        approved_version=approved_version,
        gate_summary=gate_summary,
        benchmark_summary=benchmark_summary,
    )
    _write_artifacts(
        version=version,
        gate_summary=gate_summary,
        benchmark_summary=benchmark_summary,
        decision=decision,
    )
    return gate_summary, benchmark_summary, decision


def _needs_anchor_backfill(approved_version: int) -> bool:
    if approved_version <= 0:
        return False
    try:
        summary = storage.get_json(storage.eval_benchmark_summary_key(approved_version))
    except KeyError:
        return True
    return summary.get("status") != "complete"


def _next_benchmark_opponent(summary: dict) -> str | None:
    pending = [
        entry
        for entry in summary["per_opponent"].values()
        if entry["status"] == "pending"
    ]
    if not pending:
        return None
    pending.sort(key=lambda item: (item["games"], item["opponent"]))
    return str(pending[0]["opponent"])


def _promote_candidate(version: int, model_key: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    storage.copy(model_key, storage.APPROVED_ONNX)
    storage.put_json(
        storage.APPROVED_META,
        {
            "version": version,
            "timestamp": now,
        },
    )


def _acquire_promotion_lease(
    *,
    run_id: str,
    candidate_version: int,
    approved_version: int,
) -> bool:
    lease_meta = storage.head(storage.EVAL_PROMOTION_LOCK)
    if lease_meta is not None:
        last_modified = lease_meta.get("last_modified")
        if hasattr(last_modified, "tzinfo"):
            age = (datetime.now(timezone.utc) - last_modified).total_seconds()
            if age > PROMOTION_LEASE_STALE_SECONDS:
                try:
                    storage.delete(storage.EVAL_PROMOTION_LOCK)
                except Exception:
                    pass
    try:
        storage.put_json(
            storage.EVAL_PROMOTION_LOCK,
            {
                "run_id": run_id,
                "candidate_version": candidate_version,
                "approved_version": approved_version,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            if_none_match="*",
        )
    except storage.ConditionalWriteFailed:
        return False
    return True


def _release_promotion_lease() -> None:
    try:
        storage.delete(storage.EVAL_PROMOTION_LOCK)
    except Exception:
        pass


def _run_pair(
    *,
    players: PlayerProvider,
    candidate_version: int,
    approved_version: int,
    version_keys: dict[int, str],
    opponent: str,
    eval_type: str,
    records: VersionRecordStore,
) -> bool:
    candidate_name = f"v{candidate_version}"
    pair_id = _pair_id()
    opening_seed = random.getrandbits(32)
    matchups = [
        (candidate_name, opponent, 0),
        (opponent, candidate_name, 1),
    ]

    for white_name, black_name, pair_game_index in matchups:
        try:
            white = players.get(white_name, version_keys.get(int(white_name[1:])) if white_name.startswith("v") else None)
            black = players.get(black_name, version_keys.get(int(black_name[1:])) if black_name.startswith("v") else None)
            result = play_game(
                white,
                black,
                random_opening_plies=OPENING_RANDOM_PLIES,
                opening_seed=opening_seed,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "{} pair game failed for candidate v{}: {} vs {} ({})",
                eval_type,
                candidate_version,
                white_name,
                black_name,
                exc,
            )
            return False

        record = _make_record(
            eval_type=eval_type,
            candidate_version=candidate_version,
            approved_version=approved_version,
            opponent=opponent,
            white_name=white_name,
            black_name=black_name,
            result=result,
            pair_id=pair_id,
            pair_game_index=pair_game_index,
            opening_seed=opening_seed,
        )
        if _persist_record(records, candidate_version, record) is None:
            return False

    return True


def run_evaluation_service(simulations: int = 800) -> None:
    cfg = AsyncConfig()
    cfg.ensure_cache_dirs()
    setup_json_logging("evaluation", run_id=cfg.run_id)
    log_event("evaluation.start", run_id=cfg.run_id, simulations=simulations)

    cache_dir = cfg.model_cache_dir / "evaluation"
    players = PlayerProvider(simulations, cache_dir)
    records = VersionRecordStore()

    last_sync_ts = 0.0
    logger.info("Evaluation service starting (simulations={})", simulations)

    while True:
        versions = dict(_discover_versions())
        approved_version = _read_approved_version()
        candidate_version = _latest_candidate(versions, approved_version)

        target_version = candidate_version
        include_gate = candidate_version is not None
        if target_version is None and _needs_anchor_backfill(approved_version):
            target_version = approved_version
            include_gate = False
        elif target_version is not None and _needs_anchor_backfill(approved_version):
            target_version = approved_version
            include_gate = False

        if target_version is None or target_version <= 0:
            logger.info("No candidate evaluation work available")
            time.sleep(IDLE_SECONDS)
            continue

        if target_version not in versions:
            logger.info("Waiting for immutable snapshot for v{}", target_version)
            time.sleep(5)
            continue

        if time.monotonic() - last_sync_ts >= SYNC_INTERVAL_SECONDS:
            added = records.refresh(target_version)
            if added:
                logger.info("Merged {} peer evaluation records for v{}", added, target_version)
            last_sync_ts = time.monotonic()

        gate_summary, benchmark_summary, decision = _refresh_artifacts(
            version=target_version,
            approved_version=approved_version,
            records=records,
            include_gate=include_gate,
        )

        if decision["status"] == "promote" and candidate_version == target_version:
            if not _acquire_promotion_lease(
                run_id=cfg.run_id,
                candidate_version=candidate_version,
                approved_version=approved_version,
            ):
                logger.info(
                    "Promotion lease busy for v{} over v{}; waiting for peer",
                    candidate_version,
                    approved_version,
                )
                time.sleep(5)
                continue
            try:
                current_approved_version = _read_approved_version()
                if current_approved_version != approved_version:
                    logger.info(
                        "Approved pointer moved from v{} to v{} before promoting v{}; retrying",
                        approved_version,
                        current_approved_version,
                        candidate_version,
                    )
                    continue
                _promote_candidate(candidate_version, versions[candidate_version])
                approved_version = _read_approved_version()
                gate_summary, benchmark_summary, decision = _refresh_artifacts(
                    version=target_version,
                    approved_version=approved_version,
                    records=records,
                    include_gate=include_gate,
                )
                decision = dict(decision)
                decision["status"] = "promoted"
                decision["promoted_at"] = datetime.now(timezone.utc).isoformat()
                _write_artifacts(
                    version=target_version,
                    gate_summary=gate_summary,
                    benchmark_summary=benchmark_summary,
                    decision=decision,
                )
                logger.info("Promoted v{} over v{}", candidate_version, current_approved_version)
            finally:
                _release_promotion_lease()
            continue

        if decision["status"] in {"rejected", "promoted", "baseline_ready"}:
            logger.info(
                "Evaluation idle for v{}: {}",
                target_version,
                decision["status"],
            )
            time.sleep(IDLE_SECONDS)
            continue

        if include_gate and gate_summary is not None and gate_summary["status"] == "pending":
            gate_progress = gate_summary["games"] / GATE_MAX_GAMES
        else:
            gate_progress = 1.0

        benchmark_opponent = _next_benchmark_opponent(benchmark_summary)
        benchmark_progress = 1.0
        if benchmark_opponent is not None:
            benchmark_progress = (
                benchmark_summary["per_opponent"][benchmark_opponent]["games"] / BENCHMARK_MAX_GAMES
            )

        if include_gate and gate_summary is not None and gate_summary["status"] == "pending" and (
            benchmark_opponent is None or gate_progress <= benchmark_progress
        ):
            logger.info(
                "Dispatch gate pair: v{} vs v{} ({} games so far)",
                candidate_version,
                approved_version,
                gate_summary["games"],
            )
            _run_pair(
                players=players,
                candidate_version=candidate_version,
                approved_version=approved_version,
                version_keys=versions,
                opponent=f"v{approved_version}",
                eval_type="gate",
                records=records,
            )
        elif benchmark_opponent is not None:
            logger.info(
                "Dispatch benchmark pair: v{} vs {} ({} games so far)",
                target_version,
                benchmark_opponent,
                benchmark_summary["per_opponent"][benchmark_opponent]["games"],
            )
            _run_pair(
                players=players,
                candidate_version=target_version,
                approved_version=approved_version,
                version_keys=versions,
                opponent=benchmark_opponent,
                eval_type="benchmark",
                records=records,
            )
        else:
            logger.info("Waiting for more evaluation work on v{}", target_version)
            time.sleep(IDLE_SECONDS)


def run_elo_service(simulations: int = 800, **_: int) -> None:
    """Backward-compatible alias for old entrypoints."""
    run_evaluation_service(simulations=simulations)
