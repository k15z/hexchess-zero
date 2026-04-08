"""Tests for training.audit_buffer (plan §7.9)."""

from __future__ import annotations

import io

import numpy as np

from training import audit_buffer


def _make_npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def test_audit_aggregates_per_version(monkeypatch):
    v1_bytes = _make_npz_bytes(
        root_q=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        root_entropy=np.array([1.5, 1.4, 1.6], dtype=np.float32),
        ply=np.array([10, 20, 30], dtype=np.int16),
        mlh=np.array([5, 4, 3], dtype=np.int16),
        legal_count=np.array([40, 42, 38], dtype=np.int16),
    )
    v2_bytes = _make_npz_bytes(
        root_q=np.array([0.5, 0.6], dtype=np.float32),
        root_entropy=np.array([1.0, 0.9], dtype=np.float32),
        ply=np.array([15, 25], dtype=np.int16),
        mlh=np.array([2, 1], dtype=np.int16),
        legal_count=np.array([30, 35], dtype=np.int16),
    )
    files = [
        {"key": "data/selfplay/v1/20260101T000000_a_n3.npz",
         "positions": 3, "version": "v1", "timestamp": "20260101T000000"},
        {"key": "data/selfplay/v2/20260102T000000_b_n2.npz",
         "positions": 2, "version": "v2", "timestamp": "20260102T000000"},
    ]
    blobs = {
        "data/selfplay/v1/20260101T000000_a_n3.npz": v1_bytes,
        "data/selfplay/v2/20260102T000000_b_n2.npz": v2_bytes,
    }

    monkeypatch.setattr(audit_buffer.storage, "count_positions", lambda prefix=None: 5)
    monkeypatch.setattr(audit_buffer.storage, "select_recent_files",
                        lambda prefix, max_positions: files)
    monkeypatch.setattr(audit_buffer.storage, "get", lambda key: blobs[key])

    out = io.StringIO()
    result = audit_buffer.audit(window="auto", out=out)

    pv = result["per_version"]
    assert pv["v1"]["file_count"] == 1
    assert pv["v1"]["position_count"] == 3
    assert pv["v2"]["file_count"] == 1
    assert pv["v2"]["position_count"] == 2

    text = out.getvalue()
    assert "v1" in text and "v2" in text
    assert "Histogram" in text
