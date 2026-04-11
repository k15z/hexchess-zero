"""Slack webhook notifications for training progress."""

from __future__ import annotations

import json
import os
import urllib.request
from loguru import logger


SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")


def notify_training_cycle(
    *,
    cycle: int,
    version: int,
    steps: int,
    total_steps: int,
    positions: int,
    policy_loss: float,
    value_loss: float,
    elapsed_seconds: float,
) -> None:
    """Send a Slack message summarizing a completed training cycle."""
    if not SLACK_WEBHOOK_URL:
        return

    total_loss = policy_loss + value_loss

    text = (
        f"*Hexchess Training - Cycle {cycle}*\n"
        f"Model: v{version} | Steps: {steps:,} ({total_steps:,} total)\n"
        f"Loss: policy={policy_loss:.4f} value={value_loss:.4f} total={total_loss:.4f}\n"
        f"Positions: {positions:,} | Time: {elapsed_seconds:.0f}s"
    )

    _post(text)


def notify_elo_update(
    ratings: dict[str, dict],
    total_games: int,
    new_model: str | None = None,
) -> None:
    """Send periodic Elo update or new-model-ranked notification.

    ratings: {name: {"mu": float, "sigma": float}}
    """
    if not SLACK_WEBHOOK_URL:
        return

    from .elo import conservative_rating, is_evaluated

    sorted_players = sorted(
        ratings.items(),
        key=lambda x: conservative_rating(x[1]["mu"], x[1]["sigma"]),
        reverse=True,
    )
    lines = []
    for rank, (name, r) in enumerate(sorted_players, 1):
        cr = conservative_rating(r["mu"], r["sigma"])
        marker = "" if is_evaluated(r["sigma"]) else " [prov]"
        lines.append(
            f"{rank}. {name} {cr:+.2f} (μ={r['mu']:.2f} ±{r['sigma']:.2f}){marker}"
        )
    table = "\n".join(lines)

    header = "*Hexchess Elo Update*"
    if new_model:
        header = f"*Hexchess Elo: {new_model} ranked!*"

    _post(f"{header}\n```\n{table}\n```\n({total_games} games played)")


def _post(text: str) -> None:
    """Post a message to the configured Slack webhook."""
    payload = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        SLACK_WEBHOOK_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        logger.warning("Slack notification failed: {}", e)
