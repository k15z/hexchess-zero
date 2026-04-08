"""Tests for training.config (plan §8)."""

from __future__ import annotations

import pytest

from training.config import AsyncConfig


def test_defaults_instantiate_and_validate():
    cfg = AsyncConfig()
    cfg.validate()  # should not raise
    assert cfg.run_id  # non-empty
    assert cfg.batch_size > 0
    assert 0 < cfg.pcr_p_full <= 1.0
    assert cfg.window_c > 0


def test_run_id_env_override(monkeypatch):
    monkeypatch.setenv("RUN_ID", "mytestrun")
    cfg = AsyncConfig()
    assert cfg.run_id == "mytestrun"


def test_validate_rejects_bad_pcr():
    cfg = AsyncConfig()
    cfg.pcr_p_full = 0.0
    with pytest.raises(ValueError, match="pcr_p_full"):
        cfg.validate()


def test_validate_rejects_bad_window():
    cfg = AsyncConfig()
    cfg.window_alpha = -0.5
    with pytest.raises(ValueError, match="window_alpha"):
        cfg.validate()


def test_validate_rejects_bad_resign():
    cfg = AsyncConfig()
    cfg.resign_threshold = 1.5
    with pytest.raises(ValueError, match="resign_threshold"):
        cfg.validate()


def test_validate_rejects_empty_run_id():
    cfg = AsyncConfig()
    cfg.run_id = ""
    with pytest.raises(ValueError, match="run_id"):
        cfg.validate()


def test_validate_accumulates_errors():
    cfg = AsyncConfig()
    cfg.pcr_p_full = -1
    cfg.window_alpha = -1
    with pytest.raises(ValueError) as exc_info:
        cfg.validate()
    msg = str(exc_info.value)
    assert "pcr_p_full" in msg
    assert "window_alpha" in msg
