"""Progress tracking for the training pipeline.

Reads Elo state from S3 and displays a summary.
"""

from __future__ import annotations

from . import storage
from .elo import format_elo_table


def print_progress() -> None:
    """Print current training status from S3."""
    # Model version
    try:
        meta = storage.get_json(storage.LATEST_META)
        print(f"Model: v{meta.get('version', '?')} "
              f"(promoted {meta.get('timestamp', '?')})")
    except KeyError:
        print("Model: none (no model yet)")

    # Data stats
    sp_count = storage.count_positions(storage.SELFPLAY_PREFIX)
    im_count = storage.count_positions(storage.IMITATION_PREFIX)
    print(f"Self-play: {sp_count:,} positions")
    print(f"Imitation: {im_count:,} positions")

    # Elo rankings
    try:
        elo_state = storage.get_json(storage.ELO_STATE)
        ratings = elo_state.get("ratings", {})
        total_games = elo_state.get("total_games", 0)
        if ratings:
            print(f"\nElo rankings ({total_games} games):")
            print(format_elo_table(ratings))
    except KeyError:
        print("\nNo Elo data yet.")

    # Worker heartbeats
    heartbeats = storage.ls(storage.HEARTBEATS_PREFIX)
    if heartbeats:
        print(f"\nWorkers ({len(heartbeats)} reporting):")
        for key in heartbeats:
            hb = storage.get_json(key)
            name = key.split("/")[-1].replace(".json", "")
            print(f"  {name}: v{hb.get('model_version', '?')}, "
                  f"{hb.get('total_games', 0)} games, "
                  f"last seen {hb.get('timestamp', '?')}")
