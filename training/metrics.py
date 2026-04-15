"""Progress tracking for the training pipeline."""

from __future__ import annotations

from . import storage
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

    approved_version = 0
    try:
        approved = storage.get_json(storage.APPROVED_META)
        approved_version = int(approved.get("version", 0))
    except KeyError:
        approved = {}

    eval_versions = storage.list_eval_versions()
    candidate_versions = [v for v in eval_versions if v > approved_version]
    candidate_version = max(candidate_versions) if candidate_versions else None

    if candidate_version is not None:
        print(f"\nCurrent evaluation: v{candidate_version} vs approved v{approved_version}")
        try:
            gate = storage.get_json(storage.eval_gate_summary_key(candidate_version))
            print(
                "  gate: "
                f"{gate.get('status', 'pending')} "
                f"({gate.get('wins', 0)}-{gate.get('losses', 0)}-{gate.get('draws', 0)}, "
                f"score={gate.get('score', 0.0):.3f})"
            )
        except KeyError:
            print("  gate: no data yet")
        try:
            bench = storage.get_json(storage.eval_benchmark_summary_key(candidate_version))
            print(
                "  benchmark: "
                f"{bench.get('status', 'pending')} "
                f"({bench.get('games', 0)} games across anchors)"
            )
        except KeyError:
            print("  benchmark: no data yet")
    elif approved_version:
        print(f"\nNo pending candidate. Approved model is v{approved_version}.")

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
