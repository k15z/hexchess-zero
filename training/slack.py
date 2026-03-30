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
