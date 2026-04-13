"""Tests for worker model-pointer selection."""

from __future__ import annotations

from pathlib import Path

from training import storage
from training.config import AsyncConfig
from training.worker import _read_model_version


class _TestConfig(AsyncConfig):
    def __init__(self, cache_root: Path):
        super().__init__()
        self._cache_root = cache_root

    @property
    def model_cache_dir(self) -> Path:
        return self._cache_root


def test_worker_prefers_approved_model_pointer(tmp_path, monkeypatch):
    downloaded: list[str] = []

    def fake_get_json(key: str) -> dict:
        if key == storage.APPROVED_META:
            return {"version": 4}
        if key == storage.LATEST_META:
            return {"version": 5}
        raise KeyError(key)

    def fake_get_file(key: str, local_path: str | Path) -> Path:
        downloaded.append(key)
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text("stub")
        return local_path

    monkeypatch.setattr(storage, "get_json", fake_get_json)
    monkeypatch.setattr(storage, "get_file", fake_get_file)

    version, path = _read_model_version(_TestConfig(tmp_path))

    assert version == 4
    assert Path(path).name == "approved.onnx"
    assert downloaded == [storage.APPROVED_ONNX]


def test_worker_falls_back_to_latest_when_no_approved_pointer(tmp_path, monkeypatch):
    downloaded: list[str] = []

    def fake_get_json(key: str) -> dict:
        if key == storage.APPROVED_META:
            raise KeyError(key)
        if key == storage.LATEST_META:
            return {"version": 5}
        raise KeyError(key)

    def fake_get_file(key: str, local_path: str | Path) -> Path:
        downloaded.append(key)
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text("stub")
        return local_path

    monkeypatch.setattr(storage, "get_json", fake_get_json)
    monkeypatch.setattr(storage, "get_file", fake_get_file)

    version, path = _read_model_version(_TestConfig(tmp_path))

    assert version == 5
    assert Path(path).name == "latest.onnx"
    assert downloaded == [storage.LATEST_ONNX]
