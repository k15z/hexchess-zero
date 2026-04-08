"""Replay buffer audit tool (plan §7.9).

Usage::

    uv run python -m training.audit_buffer [--window auto|N]

Lists ``data/selfplay/v*/*.npz`` in S3, selects either the sublinear window
(``auto``) or the N most recent positions, aggregates per-version stats from
.npz metadata, and prints terminal-friendly histograms of key diagnostics.
"""

from __future__ import annotations

import argparse
import io
import sys
from collections import defaultdict
from typing import Iterable

import numpy as np

from . import storage
from .replay_window import sublinear_window_size


_NUMERIC_COLS = ("root_q", "root_entropy", "legal_count", "mlh", "ply")


def _load_npz_arrays(key: str) -> dict[str, np.ndarray]:
    """Download a .npz and return its arrays as a dict."""
    data = storage.get(key)
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _safe_mean(arr: np.ndarray | None) -> float:
    if arr is None or arr.size == 0:
        return float("nan")
    return float(np.asarray(arr).mean())


def _ascii_histogram(values: Iterable[float], bins: int = 20, width: int = 40) -> str:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "  (no data)"
    hist, edges = np.histogram(arr, bins=bins)
    peak = hist.max() or 1
    lines = []
    for i, c in enumerate(hist):
        bar = "#" * int(width * c / peak)
        lines.append(f"  [{edges[i]:8.3f}, {edges[i + 1]:8.3f})  {c:>6}  {bar}")
    return "\n".join(lines)


def audit(
    window: str | int = "auto",
    *,
    prefix: str = storage.SELFPLAY_PREFIX,
    max_files_for_histograms: int = 50,
    out=sys.stdout,
) -> dict:
    """Run the audit. Returns the aggregated per-version stats dict (for testing)."""
    n_total = storage.count_positions(prefix)
    if window == "auto":
        target = sublinear_window_size(n_total)
    else:
        target = int(window)
    files = storage.select_recent_files(prefix, target)

    print(f"N_total = {n_total:,}", file=out)
    print(f"window  = {target:,} ({'auto' if window == 'auto' else 'manual'})", file=out)
    print(f"files   = {len(files)}", file=out)
    print(file=out)

    # Per-version aggregation from filenames alone.
    per_version: dict[str, dict] = defaultdict(
        lambda: {"file_count": 0, "position_count": 0}
    )
    for f in files:
        v = f["version"]
        per_version[v]["file_count"] += 1
        per_version[v]["position_count"] += f["positions"]

    # Deep-load a sample of files for histogram data + per-version means.
    sample_files = files[-max_files_for_histograms:]
    hist_accum: dict[str, list[float]] = defaultdict(list)
    per_version_samples: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for f in sample_files:
        try:
            arrs = _load_npz_arrays(f["key"])
        except Exception as exc:  # pragma: no cover
            print(f"  warn: failed to load {f['key']}: {exc}", file=out)
            continue
        v = f["version"]
        for col in _NUMERIC_COLS:
            if col in arrs:
                vals = np.asarray(arrs[col]).astype(np.float64).ravel()
                hist_accum[col].extend(vals.tolist())
                per_version_samples[v][col].extend(vals.tolist())

    # Per-version summary table.
    print("Per-version summary:", file=out)
    header = f"{'version':<8} {'files':>6} {'positions':>10} {'mean_q':>9} {'mean_H':>9} {'mean_ply':>9} {'mean_mlh':>9}"
    print(header, file=out)
    print("-" * len(header), file=out)
    for v in sorted(per_version):
        row = per_version[v]
        s = per_version_samples.get(v, {})
        print(
            f"{v:<8} {row['file_count']:>6} {row['position_count']:>10,} "
            f"{_safe_mean(np.array(s.get('root_q', []))):>9.4f} "
            f"{_safe_mean(np.array(s.get('root_entropy', []))):>9.4f} "
            f"{_safe_mean(np.array(s.get('ply', []))):>9.2f} "
            f"{_safe_mean(np.array(s.get('mlh', []))):>9.2f}",
            file=out,
        )
    print(file=out)

    # Histograms.
    for col in ("root_q", "root_entropy", "legal_count", "mlh"):
        print(f"Histogram — {col}:", file=out)
        print(_ascii_histogram(hist_accum.get(col, [])), file=out)
        print(file=out)

    return {"per_version": dict(per_version), "hist": dict(hist_accum)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="auto",
                        help="'auto' (sublinear formula) or integer position count")
    args = parser.parse_args(argv)

    window: str | int = "auto"
    if args.window != "auto":
        window = int(args.window)
    audit(window=window)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
