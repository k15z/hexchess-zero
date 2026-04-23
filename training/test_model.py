"""Smoke tests for HexChessNet sizing and forward shapes."""

import torch

from training.config import _BaseConfig
from training.model import build_model


def test_param_count_in_target_range():
    """Guard against accidental size blow-up for the current production net.

    The current default is a 10x192 trunk, which should land around 10M params.
    """
    model = build_model(_BaseConfig())
    n = sum(p.numel() for p in model.parameters())
    assert 9_000_000 <= n <= 12_000_000, (
        f"param count {n:,} outside [9M, 12M] — did the trunk or head sizes change?"
    )


def test_forward_shapes_all_heads():
    cfg = _BaseConfig()
    model = build_model(cfg).eval()
    out = model(torch.zeros(2, cfg.board_channels, cfg.board_height, cfg.board_width))
    assert set(out.keys()) == {"policy", "wdl", "mlh", "stv", "aux_policy"}
    assert out["policy"].shape[0] == 2
    assert out["wdl"].shape == (2, 3)
    assert out["mlh"].shape == (2, 1)
    assert out["stv"].shape == (2, 3)
    assert out["aux_policy"].shape == out["policy"].shape


def test_eval_mode_is_deterministic():
    cfg = _BaseConfig()
    model = build_model(cfg).eval()
    x = torch.randn(2, cfg.board_channels, cfg.board_height, cfg.board_width)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    for k in a:
        assert torch.equal(a[k], b[k]), f"{k} not deterministic in eval mode"
