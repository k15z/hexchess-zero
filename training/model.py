"""ResNet policy + value network with SE and global pooling for hexagonal chess.

Architecture: global_pool_se — validated as best cost/quality tradeoff in
architecture experiments (see training/experiments/RESULTS.md).

Every residual block has Squeeze-and-Excitation (Leela-style scale+bias).
Blocks at configured positions also get KataGo-style global pooling, which
injects board-wide context (material balance, king safety) into local
convolutions via a pooled bias vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import _BaseConfig
from .hexchess_binding import load_hexchess

_hexchess = load_hexchess(required=False)
NUM_MOVE_INDICES = _hexchess.num_move_indices() if _hexchess is not None else 4206


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

    Input:  (batch, 22, 11, 11)
    Output: dict with keys:
        policy:     (batch, num_move_indices)   main policy logits
        wdl:        (batch, 3)                  Win/Draw/Loss logits
        mlh:        (batch, 1)                  moves-left head (plies, regression)
        stv:        (batch, 3)                  short-term (h=8) WDL logits
        aux_policy: (batch, num_move_indices)   opponent-reply policy logits
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

        # --- Policy head (8-channel conv for richer spatial features) ---
        policy_ch = cfg.policy_channels
        self.policy_conv = nn.Conv2d(nf, policy_ch, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_ch)
        self.policy_fc = nn.Linear(policy_ch * self.board_h * self.board_w, NUM_MOVE_INDICES)

        # --- Value head (global avg pool for translation-invariant value) ---
        value_ch = cfg.value_channels
        self.value_conv = nn.Conv2d(nf, value_ch, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_ch)
        self.value_fc1 = nn.Linear(value_ch, 256)
        self.value_fc2 = nn.Linear(256, 3)

        # --- Moves-left head (MLH): predicts plies-to-end (scalar regression) ---
        mlh_ch = 16
        self.mlh_conv = nn.Conv2d(nf, mlh_ch, 1, bias=False)
        self.mlh_bn = nn.BatchNorm2d(mlh_ch)
        self.mlh_fc1 = nn.Linear(mlh_ch, 64)
        self.mlh_fc2 = nn.Linear(64, 1)

        # --- Short-term value head (STV): WDL at horizon h=8 plies ---
        stv_ch = 32
        self.stv_conv = nn.Conv2d(nf, stv_ch, 1, bias=False)
        self.stv_bn = nn.BatchNorm2d(stv_ch)
        self.stv_fc1 = nn.Linear(stv_ch, 128)
        self.stv_fc2 = nn.Linear(128, 3)

        # --- Auxiliary opponent-reply policy head (narrower than main) ---
        aux_ch = cfg.aux_policy_channels
        self.aux_policy_conv = nn.Conv2d(nf, aux_ch, 1, bias=False)
        self.aux_policy_bn = nn.BatchNorm2d(aux_ch)
        self.aux_policy_fc = nn.Linear(
            aux_ch * self.board_h * self.board_w, NUM_MOVE_INDICES
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Input block
        x = F.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        x = self.residual_blocks(x)

        # Main policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value head (WDL logits — global avg pool)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.mean(dim=(2, 3))
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)

        # Moves-left head (scalar plies)
        m = F.relu(self.mlh_bn(self.mlh_conv(x)))
        m = m.mean(dim=(2, 3))
        m = F.relu(self.mlh_fc1(m))
        mlh = self.mlh_fc2(m)

        # Short-term value head (WDL at h=8 plies)
        s = F.relu(self.stv_bn(self.stv_conv(x)))
        s = s.mean(dim=(2, 3))
        s = F.relu(self.stv_fc1(s))
        stv = self.stv_fc2(s)

        # Auxiliary opponent-reply policy head
        ap = F.relu(self.aux_policy_bn(self.aux_policy_conv(x)))
        ap = ap.view(ap.size(0), -1)
        aux_policy = self.aux_policy_fc(ap)

        return {
            "policy": p,
            "wdl": v,
            "mlh": mlh,
            "stv": stv,
            "aux_policy": aux_policy,
        }


def build_model(config: _BaseConfig | None = None) -> HexChessNet:
    """Convenience constructor."""
    return HexChessNet(config)


if __name__ == "__main__":
    cfg = _BaseConfig()
    model = build_model(cfg)

    # Parameter count. Plan §2 estimated ~3.3M but that assumed a smaller
    # trunk; the real 10x192 trunk + two ~4M policy heads puts us near 15M.
    # Loose ceiling: <20M. TODO(chunk 13): revisit model sizing.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    assert total_params < 20_000_000, f"param count {total_params} exceeds 20M ceiling"

    # Shape smoke test
    model.eval()
    dummy = torch.randn(2, cfg.board_channels, cfg.board_height, cfg.board_width)
    with torch.no_grad():
        out = model(dummy)
    expected_shapes = {
        "policy": (2, NUM_MOVE_INDICES),
        "wdl": (2, 3),
        "mlh": (2, 1),
        "stv": (2, 3),
        "aux_policy": (2, NUM_MOVE_INDICES),
    }
    for k, expected in expected_shapes.items():
        actual = tuple(out[k].shape)
        assert actual == expected, f"{k}: expected {expected}, got {actual}"
        print(f"  {k}: {actual}")

    # BN eval-mode determinism: two forwards with the same input must match exactly.
    with torch.no_grad():
        out2 = model(dummy)
    for k in out:
        assert torch.equal(out[k], out2[k]), f"{k} not deterministic in eval mode"
    print("eval-mode determinism: OK")
