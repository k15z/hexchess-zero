from __future__ import annotations
"""ResNet-style policy + value network for hexagonal chess."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config

try:
    import hexchess

    NUM_MOVE_INDICES = hexchess.num_move_indices()
except ImportError:
    # Fallback when hexchess bindings are not installed yet.
    # ~91*91 possible (from, to) pairs plus promotion variants; actual value set by engine.
    NUM_MOVE_INDICES = 4000


class ResidualBlock(nn.Module):
    """A single residual block: conv-bn-relu-conv-bn + skip connection."""

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class HexChessNet(nn.Module):
    """
    Policy + value network for hexagonal chess.

    Input:  (batch, 16, 11, 11)
    Output: (policy_logits: (batch, num_move_indices),
             value: (batch, 1))
    """

    def __init__(self, config: Config | None = None):
        super().__init__()
        cfg = config or Config()
        self.num_filters = cfg.num_filters
        self.board_h = cfg.board_height
        self.board_w = cfg.board_width

        # --- Input block ---
        self.input_conv = nn.Conv2d(
            cfg.board_channels, cfg.num_filters, 3, padding=1, bias=False
        )
        self.input_bn = nn.BatchNorm2d(cfg.num_filters)

        # --- Residual tower ---
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(cfg.num_filters) for _ in range(cfg.num_residual_blocks)]
        )

        # --- Policy head ---
        self.policy_conv = nn.Conv2d(cfg.num_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.board_h * self.board_w, NUM_MOVE_INDICES)

        # --- Value head ---
        self.value_conv = nn.Conv2d(cfg.num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * self.board_h * self.board_w, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Input block
        x = F.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        x = self.residual_blocks(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


def build_model(config: Config | None = None) -> HexChessNet:
    """Convenience constructor."""
    return HexChessNet(config)


if __name__ == "__main__":
    cfg = Config()
    model = build_model(cfg)

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Smoke test with random input
    dummy = torch.randn(1, cfg.board_channels, cfg.board_height, cfg.board_width)
    policy, value = model(dummy)
    print(f"Policy shape: {policy.shape}")  # (1, NUM_MOVE_INDICES)
    print(f"Value shape:  {value.shape}")   # (1, 1)
    print(f"Value range:  [{value.min().item():.4f}, {value.max().item():.4f}]")
