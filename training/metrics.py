"""Progress tracking for the training pipeline.

Reads trainer logs from .data/logs/trainer.jsonl and displays a summary
of training progress over time.
"""

from __future__ import annotations

import json

from .config import _data_root


def print_progress() -> None:
    """Print a summary table from the trainer log."""
    log_path = _data_root() / "logs" / "trainer.jsonl"
    if not log_path.exists():
        print("No trainer log found. Start the trainer first.")
        return

    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("event") == "cycle_complete":
                entries.append(entry)

    if not entries:
        print("No completed training cycles found in log.")
        return

    # Header
    header = (
        f"{'Cycle':>6}  {'Version':>8}  {'Policy':>8}  {'Value':>8}  "
        f"{'Total':>8}  {'Positions':>10}  {'Time':>6}"
    )
    print(header)
    print("-" * len(header))

    for e in entries:
        cycle = e.get("cycle", "")
        version = f"v{e.get('version', '?')}"
        pl = e.get("policy_loss", "")
        vl = e.get("value_loss", "")
        tl = round(pl + vl, 4) if isinstance(pl, (int, float)) and isinstance(vl, (int, float)) else ""
        positions = e.get("positions", "")
        elapsed = e.get("elapsed_seconds", 0)
        minutes = f"{elapsed / 60:.1f}m"

        print(
            f"{cycle:>6}  {version:>8}  {pl:>8}  {vl:>8}  "
            f"{tl:>8}  {positions:>10}  {minutes:>6}"
        )

    print(f"\n{len(entries)} training cycle(s) logged to {log_path}")

    # Also show Elo rankings if available
    elo_path = _data_root() / "elo_rankings.jsonl"
    if elo_path.exists():
        from collections import deque
        with open(elo_path) as f:
            last_line = deque(f, maxlen=1)
        if last_line:
            latest = json.loads(last_line[0])
            ratings = latest.get("ratings", {})
            ts = latest.get("timestamp", "?")
            if ratings:
                from .elo import format_elo_table
                print(f"\nLatest Elo ranking ({ts}):")
                print(format_elo_table(ratings))
