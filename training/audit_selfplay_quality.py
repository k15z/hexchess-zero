"""Self-play data quality audit for recent S3 slices.

Usage::

    uv run python -m training.audit_selfplay_quality --limit-files 100
    uv run python -m training.audit_selfplay_quality --limit-positions 50000
    uv run python -m training.audit_selfplay_quality --version v1 --since 20260412T201700

Audits recent ``data/selfplay/v*/`` objects together with their ``.meta.json``
sidecars and per-game trace JSONs. The output is meant to answer two questions:

1. Is the data structurally valid for training?
2. Are the operational regressions we care about currently absent?

In particular, it highlights:
- missing sidecars / traces
- unknown ``git_sha`` provenance
- any resignation terminations
- NaN / Inf / illegal policy mass / legality mismatches
- WDL normalization issues
- termination mix, PCR keep ratio, and opening diversity
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from . import storage


@dataclass
class AuditSummary:
    selected_files: int = 0
    selected_games: int = 0
    selected_versions: list[str] = field(default_factory=list)
    selected_workers: list[str] = field(default_factory=list)
    total_full_positions: int = 0
    total_positions: int = 0
    mean_game_length: float = 0.0
    mean_keep_ratio: float = 0.0
    root_n_min: int = 0
    root_n_max: int = 0
    root_q_mean: float = 0.0
    root_entropy_mean: float = 0.0
    unknown_git_sha_games: int = 0
    resignation_games: int = 0
    missing_meta_files: int = 0
    missing_trace_games: int = 0
    trace_meta_termination_mismatches: int = 0
    duplicate_game_id_count: int = 0
    duplicate_game_id_games: int = 0
    nan_or_inf_rows: int = 0
    negative_policy_rows: int = 0
    illegal_policy_rows: int = 0
    legal_count_mismatch_rows: int = 0
    wdl_terminal_bad_rows: int = 0
    wdl_short_bad_rows: int = 0
    non_full_search_rows: int = 0
    meta_position_mismatches: int = 0
    top_opening_share: float = 0.0
    top_worker_share: float = 0.0
    termination_counts: dict[str, int] = field(default_factory=dict)
    result_counts: dict[str, int] = field(default_factory=dict)
    git_sha_counts: dict[str, int] = field(default_factory=dict)
    top_openings: list[tuple[str, int]] = field(default_factory=list)
    top_workers: list[tuple[str, int]] = field(default_factory=list)
    selected_keys: list[str] = field(default_factory=list)


def _load_npz_arrays(key: str) -> dict[str, np.ndarray]:
    data = storage.get(key)
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _load_json(key: str) -> dict[str, Any]:
    return json.loads(storage.get(key))


def _meta_key_for_npz(key: str) -> str:
    return key[:-4] + ".meta.json" if key.endswith(".npz") else key + ".meta.json"


def _trace_key_for_game(version: str, game_id: int) -> str:
    return f"{storage.SELFPLAY_TRACES_PREFIX}{version}/{int(game_id)}.json"


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _select_files(
    *,
    prefix: str,
    version: str | None,
    since: str | None,
    until: str | None,
    limit_files: int | None,
    limit_positions: int | None,
) -> list[dict]:
    files = storage.list_data_files(prefix)
    if version:
        files = [f for f in files if f["version"] == version]
    if since:
        files = [f for f in files if f["timestamp"] >= since]
    if until:
        files = [f for f in files if f["timestamp"] <= until]

    selected: list[dict] = []
    seen_positions = 0
    for f in files:
        selected.append(f)
        seen_positions += int(f["positions"])
        if limit_files is not None and len(selected) >= limit_files:
            break
        if limit_positions is not None and seen_positions >= limit_positions:
            break
    return selected


def audit(
    *,
    prefix: str = storage.SELFPLAY_PREFIX,
    version: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit_files: int | None = 100,
    limit_positions: int | None = None,
    out=sys.stdout,
) -> AuditSummary:
    selected = _select_files(
        prefix=prefix,
        version=version,
        since=since,
        until=until,
        limit_files=limit_files,
        limit_positions=limit_positions,
    )

    terminations: Counter[str] = Counter()
    results: Counter[str] = Counter()
    git_shas: Counter[str] = Counter()
    openings: Counter[str] = Counter()
    workers: Counter[str] = Counter()
    game_ids: Counter[int] = Counter()

    game_lengths: list[float] = []
    keep_ratios: list[float] = []
    root_qs: list[float] = []
    root_entropies: list[float] = []
    root_ns: list[int] = []

    summary = AuditSummary(
        selected_files=len(selected),
        selected_keys=[f["key"] for f in selected],
    )

    for f in selected:
        key = f["key"]
        version_name = f["version"]
        try:
            arrays = _load_npz_arrays(key)
        except Exception as exc:  # pragma: no cover
            print(f"warn: failed to load {key}: {exc}", file=out)
            continue

        n_rows = int(arrays["boards"].shape[0])
        summary.total_full_positions += n_rows
        root_qs.extend(np.asarray(arrays["root_q"], dtype=np.float64).tolist())
        root_entropies.extend(
            np.asarray(arrays["root_entropy"], dtype=np.float64).tolist()
        )
        root_ns.extend(np.asarray(arrays["root_n"], dtype=np.int64).tolist())

        if "was_full_search" in arrays:
            was_full = np.asarray(arrays["was_full_search"]).astype(bool)
            summary.non_full_search_rows += int((~was_full).sum())

        legal_mask = np.asarray(arrays["legal_mask"]).astype(bool)
        policy = np.asarray(arrays["policy"], dtype=np.float64)
        legal_count = np.asarray(arrays["legal_count"], dtype=np.int64)
        wdl_terminal = np.asarray(arrays["wdl_terminal"], dtype=np.float64)
        wdl_short = np.asarray(arrays["wdl_short"], dtype=np.float64)

        finite_ok = True
        for arr_name in (
            "boards",
            "policy",
            "policy_aux_opp",
            "wdl_terminal",
            "wdl_short",
            "mlh",
            "root_q",
            "root_entropy",
            "nn_value_at_position",
        ):
            arr = np.asarray(arrays[arr_name])
            row_finite = np.isfinite(arr).all(axis=tuple(range(1, arr.ndim))) if arr.ndim > 1 else np.isfinite(arr)
            bad = int((~row_finite).sum())
            summary.nan_or_inf_rows += bad
            finite_ok = finite_ok and bad == 0

        summary.negative_policy_rows += int((policy < -1e-6).any(axis=1).sum())
        illegal_mass = np.where(legal_mask, 0.0, policy).sum(axis=1)
        summary.illegal_policy_rows += int((illegal_mass > 1e-5).sum())
        summary.legal_count_mismatch_rows += int(
            (legal_mask.sum(axis=1).astype(np.int64) != legal_count).sum()
        )
        summary.wdl_terminal_bad_rows += int(
            (~np.isclose(wdl_terminal.sum(axis=1), 1.0, atol=1e-4)).sum()
        )
        summary.wdl_short_bad_rows += int(
            (~np.isclose(wdl_short.sum(axis=1), 1.0, atol=1e-4)).sum()
        )

        meta_key = _meta_key_for_npz(key)
        try:
            meta = _load_json(meta_key)
        except KeyError:
            summary.missing_meta_files += 1
            continue

        summary.selected_games += 1
        summary.total_positions += int(meta.get("num_total_positions", n_rows))
        summary.selected_versions.append(version_name)

        game_len = int(meta.get("num_total_positions", n_rows))
        keep_ratio = float(n_rows) / float(max(game_len, 1))
        game_lengths.append(float(game_len))
        keep_ratios.append(keep_ratio)

        if int(meta.get("num_full_search_positions", n_rows)) != n_rows:
            summary.meta_position_mismatches += 1

        termination = str(meta.get("termination", "unknown"))
        result = str(meta.get("result", "unknown"))
        git_sha = str(meta.get("git_sha", "unknown") or "unknown")
        opening = str(meta.get("openings_hash", "unknown"))
        worker = str(meta.get("worker", "unknown"))

        terminations[termination] += 1
        results[result] += 1
        git_shas[git_sha] += 1
        openings[opening] += 1
        workers[worker] += 1

        if git_sha == "unknown":
            summary.unknown_git_sha_games += 1
        if termination == "resignation":
            summary.resignation_games += 1

        game_id_range = meta.get("game_id_range") or []
        if game_id_range:
            game_id = int(game_id_range[0])
            game_ids[game_id] += 1
            trace_key = _trace_key_for_game(version_name, game_id)
            try:
                trace = _load_json(trace_key)
                if str(trace.get("termination", "unknown")) != termination:
                    summary.trace_meta_termination_mismatches += 1
            except KeyError:
                summary.missing_trace_games += 1
        else:
            summary.missing_trace_games += 1

    summary.selected_versions = sorted(set(summary.selected_versions))
    summary.selected_workers = sorted(workers.keys())
    summary.mean_game_length = _safe_mean(game_lengths)
    summary.mean_keep_ratio = _safe_mean(keep_ratios)
    summary.root_q_mean = _safe_mean(root_qs)
    summary.root_entropy_mean = _safe_mean(root_entropies)
    summary.root_n_min = min(root_ns) if root_ns else 0
    summary.root_n_max = max(root_ns) if root_ns else 0
    summary.termination_counts = dict(terminations)
    summary.result_counts = dict(results)
    summary.git_sha_counts = dict(git_shas)
    summary.top_openings = openings.most_common(10)
    summary.top_workers = workers.most_common(10)
    summary.duplicate_game_id_count = sum(1 for c in game_ids.values() if c > 1)
    summary.duplicate_game_id_games = sum(c for c in game_ids.values() if c > 1)
    if summary.selected_games:
        summary.top_opening_share = (
            summary.top_openings[0][1] / summary.selected_games if summary.top_openings else 0.0
        )
        summary.top_worker_share = (
            summary.top_workers[0][1] / summary.selected_games if summary.top_workers else 0.0
        )

    print(f"files={summary.selected_files} games={summary.selected_games}", file=out)
    print(
        f"full_positions={summary.total_full_positions:,} total_positions={summary.total_positions:,} "
        f"mean_game_len={summary.mean_game_length:.1f} keep_ratio={summary.mean_keep_ratio:.3f}",
        file=out,
    )
    print(
        f"root_n=[{summary.root_n_min}, {summary.root_n_max}] "
        f"mean_root_q={summary.root_q_mean:.4f} mean_root_entropy={summary.root_entropy_mean:.4f}",
        file=out,
    )
    print(
        f"unknown_git_sha_games={summary.unknown_git_sha_games} "
        f"resignation_games={summary.resignation_games} "
        f"missing_meta={summary.missing_meta_files} missing_trace={summary.missing_trace_games} "
        f"trace_mismatches={summary.trace_meta_termination_mismatches}",
        file=out,
    )
    print(
        f"duplicate_game_id_count={summary.duplicate_game_id_count} "
        f"duplicate_game_id_games={summary.duplicate_game_id_games}",
        file=out,
    )
    print(
        f"nan_or_inf_rows={summary.nan_or_inf_rows} "
        f"negative_policy_rows={summary.negative_policy_rows} "
        f"illegal_policy_rows={summary.illegal_policy_rows} "
        f"legal_count_mismatch_rows={summary.legal_count_mismatch_rows}",
        file=out,
    )
    print(
        f"wdl_terminal_bad_rows={summary.wdl_terminal_bad_rows} "
        f"wdl_short_bad_rows={summary.wdl_short_bad_rows} "
        f"non_full_search_rows={summary.non_full_search_rows} "
        f"meta_position_mismatches={summary.meta_position_mismatches}",
        file=out,
    )
    print(f"versions={summary.selected_versions}", file=out)
    print(f"top_workers={summary.top_workers[:5]}", file=out)
    print(f"termination_counts={summary.termination_counts}", file=out)
    print(f"result_counts={summary.result_counts}", file=out)
    print(f"top_openings={summary.top_openings[:5]}", file=out)

    return summary


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=storage.SELFPLAY_PREFIX)
    parser.add_argument("--version")
    parser.add_argument("--since", help="Lower timestamp bound (YYYYMMDDTHHMMSS)")
    parser.add_argument("--until", help="Upper timestamp bound (YYYYMMDDTHHMMSS)")
    parser.add_argument("--limit-files", type=int, default=100)
    parser.add_argument("--limit-positions", type=int)
    parser.add_argument("--json", action="store_true", dest="json_out")
    args = parser.parse_args(argv)

    summary = audit(
        prefix=args.prefix,
        version=args.version,
        since=args.since,
        until=args.until,
        limit_files=args.limit_files,
        limit_positions=args.limit_positions,
        out=sys.stdout if not args.json_out else io.StringIO(),
    )
    if args.json_out:
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
