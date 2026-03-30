from __future__ import annotations
"""Slack webhook notifications for training progress."""

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
    promoted: bool,
    win_rate: float | None = None,
    elapsed_seconds: float,
) -> None:
    """Send a Slack message summarizing a completed training cycle."""
    if not SLACK_WEBHOOK_URL:
        return

    total_loss = policy_loss + value_loss
    status = f"v{version}" if promoted else f"v{version} (kept)"

    text = (
        f"*Hexchess Training - Cycle {cycle}*\n"
        f"Model: {status} | Steps: {steps:,} ({total_steps:,} total)\n"
        f"Loss: policy={policy_loss:.4f} value={value_loss:.4f} total={total_loss:.4f}\n"
        f"Positions: {positions:,} | Time: {elapsed_seconds:.0f}s"
    )
    if win_rate is not None:
        text += f" | Win rate: {win_rate:.0%}"

    _post(text)


def notify_elo_ranking(ratings: dict[str, int | float], elapsed_seconds: float) -> None:
    """Send a Slack message with Elo ranking results."""
    if not SLACK_WEBHOOK_URL:
        return

    sorted_players = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    lines = [f"{rank}. {name} ({elo:+d})" for rank, (name, elo) in enumerate(sorted_players, 1)]
    table = "\n".join(lines)

    text = (
        f"*Hexchess Elo Ranking*\n"
        f"```\n{table}\n```\n"
        f"({elapsed_seconds:.0f}s elapsed)"
    )
    _post(text)


def notify_elo_update(
    ratings: dict[str, int | float],
    total_games: int,
    new_model: str | None = None,
) -> None:
    """Send periodic Elo update or new-model-ranked notification."""
    if not SLACK_WEBHOOK_URL:
        return

    sorted_players = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    lines = [f"{rank}. {name} ({elo:+d})" for rank, (name, elo) in enumerate(sorted_players, 1)]
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
