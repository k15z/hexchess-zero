from __future__ import annotations
"""ResNet policy + value network with SE and global pooling for hexagonal chess.

Architecture: global_pool_se — validated as best cost/quality tradeoff in
architecture experiments (see training/experiments/RESULTS.md).

Every residual block has Squeeze-and-Excitation (Leela-style scale+bias).
Blocks at configured positions also get KataGo-style global pooling, which
injects board-wide context (material balance, king safety) into local
convolutions via a pooled bias vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import _BaseConfig

try:
    import hexchess

    NUM_MOVE_INDICES = hexchess.num_move_indices()
except ImportError:
    # Fallback when hexchess bindings are not installed yet.
    NUM_MOVE_INDICES = 4206


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation with per-channel scale + bias (Leela variant)."""

    def __init__(self, num_filters: int, se_channels: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(num_filters, se_channels)
        self.fc2 = nn.Linear(se_channels, 2 * num_filters)
        self.num_filters = num_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        squeezed = x.mean(dim=(2, 3))                          # (B, C)
        excited = F.relu(self.fc1(squeezed))                    # (B, se_ch)
        excited = self.fc2(excited)                             # (B, 2*C)
        scale = torch.sigmoid(excited[:, :c]).view(b, c, 1, 1)
        bias = excited[:, c:].view(b, c, 1, 1)
        return scale * x + bias


class SEResidualBlock(nn.Module):
    """Residual block with SE: conv-bn-relu-conv-bn-SE + skip."""

    def __init__(self, num_filters: int, se_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.se = SEBlock(num_filters, se_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + x)


class SEGlobalPoolBlock(nn.Module):
    """Residual block with SE and KataGo-style global pooling bias.

    A parallel conv branch computes pool_channels features, pools them
    three ways (mean, scaled mean, max), and projects to a per-channel
    bias added to the main path before the second conv.
    """

    def __init__(self, num_filters: int, se_channels: int = 32,
                 pool_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        # Global pool branch
        self.pool_conv = nn.Conv2d(num_filters, pool_channels, 3, padding=1, bias=False)
        self.pool_bn = nn.BatchNorm2d(pool_channels)
        self.pool_fc = nn.Linear(3 * pool_channels, num_filters)
        # Second conv + SE
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.se = SEBlock(num_filters, se_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        # Global pool: mean, scaled mean, max
        p = F.relu(self.pool_bn(self.pool_conv(x)))
        p_mean = p.mean(dim=(2, 3))
        p_scaled = p_mean * (11.0 / 9.54)  # scale by board dimension ratio
        p_max = p.amax(dim=(2, 3))
        pooled = torch.cat([p_mean, p_scaled, p_max], dim=1)
        bias = self.pool_fc(pooled).unsqueeze(-1).unsqueeze(-1)
        out = out + bias
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + x)


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------

class HexChessNet(nn.Module):
    """
    Policy + WDL value network for hexagonal chess.

    Architecture: SE residual tower with KataGo global pooling in select blocks.

    Input:  (batch, 19, 11, 11)
    Output: (policy_logits: (batch, num_move_indices),
             wdl_logits: (batch, 3))
    """

    def __init__(self, config: _BaseConfig | None = None):
        super().__init__()
        cfg = config or _BaseConfig()
        nf = cfg.num_filters
        se_ch = cfg.se_channels
        pool_ch = cfg.global_pool_channels
        pool_blocks = set(cfg.global_pool_blocks)

        self.board_h = cfg.board_height
        self.board_w = cfg.board_width

        # --- Input block ---
        self.input_conv = nn.Conv2d(
            cfg.board_channels, nf, 3, padding=1, bias=False
        )
        self.input_bn = nn.BatchNorm2d(nf)

        # --- Residual tower ---
        blocks = []
        for i in range(cfg.num_residual_blocks):
            if i in pool_blocks:
                blocks.append(SEGlobalPoolBlock(nf, se_ch, pool_ch))
            else:
                blocks.append(SEResidualBlock(nf, se_ch))
        self.residual_blocks = nn.Sequential(*blocks)

        # --- Policy head ---
        self.policy_conv = nn.Conv2d(nf, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.board_h * self.board_w, NUM_MOVE_INDICES)

        # --- Value head ---
        self.value_conv = nn.Conv2d(nf, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * self.board_h * self.board_w, 256)
        self.value_fc2 = nn.Linear(256, 3)

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

        # Value head (WDL logits)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)

        return p, v


def build_model(config: _BaseConfig | None = None) -> HexChessNet:
    """Convenience constructor."""
    return HexChessNet(config)


if __name__ == "__main__":
    cfg = _BaseConfig()
    model = build_model(cfg)

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Smoke test with random input
    dummy = torch.randn(1, cfg.board_channels, cfg.board_height, cfg.board_width)
    policy, value = model(dummy)
    print(f"Policy shape: {policy.shape}")  # (1, NUM_MOVE_INDICES)
    print(f"WDL shape:    {value.shape}")   # (1, 3)
    wdl_probs = torch.softmax(value, dim=1)
    print(f"WDL probs:    {wdl_probs[0].tolist()}")
