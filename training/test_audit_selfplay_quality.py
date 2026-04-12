from __future__ import annotations

import io
import json

import numpy as np

from training import audit_selfplay_quality


def _npz_bytes(*, legal_count: int = 3, illegal_mass: float = 0.0) -> bytes:
    policy = np.zeros((2, 8), dtype=np.float16)
    policy[:, 0] = np.float16(1.0 - illegal_mass)
    if illegal_mass:
        policy[:, 7] = np.float16(illegal_mass)
    legal_mask = np.zeros((2, 8), dtype=bool)
    legal_mask[:, :legal_count] = True
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        boards=np.zeros((2, 22, 11, 11), dtype=np.float16),
        policy=policy,
        policy_aux_opp=np.full((2, 8), 0.125, dtype=np.float16),
        legal_mask=legal_mask,
        wdl_terminal=np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (2, 1)),
        wdl_short=np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (2, 1)),
        mlh=np.array([2, 1], dtype=np.int16),
        was_full_search=np.ones(2, dtype=bool),
        root_q=np.array([0.1, -0.1], dtype=np.float16),
        root_n=np.array([800, 800], dtype=np.int32),
        root_entropy=np.array([2.8, 2.9], dtype=np.float16),
        nn_value_at_position=np.array([0.1, -0.1], dtype=np.float16),
        legal_count=np.array([legal_count, legal_count], dtype=np.int16),
        ply=np.array([0, 1], dtype=np.int16),
        game_id=np.array([7, 7], dtype=np.uint64),
    )
    return buf.getvalue()


def test_audit_detects_provenance_and_policy_issues(monkeypatch) -> None:
    files = [
        {
            "key": "data/selfplay/v1/20260412T201700_abcd_n2.npz",
            "positions": 2,
            "timestamp": "20260412T201700",
            "version": "v1",
        }
    ]
    meta = {
        "game_id_range": [7, 7],
        "model_version": 1,
        "worker": "worker-a",
        "started_at": "2026-04-12T20:17:00+00:00",
        "duration_s": 10.0,
        "num_full_search_positions": 2,
        "num_total_positions": 4,
        "result": "draw",
        "termination": "resignation",
        "openings_hash": "open-a",
        "git_sha": "unknown",
        "rng_seed": 1,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.3,
        "num_simulations": 800,
    }
    blobs = {
        files[0]["key"]: _npz_bytes(legal_count=3, illegal_mass=0.2),
        "data/selfplay/v1/20260412T201700_abcd_n2.meta.json": json.dumps(meta).encode(),
    }

    monkeypatch.setattr(audit_selfplay_quality.storage, "list_data_files", lambda prefix: files)
    monkeypatch.setattr(audit_selfplay_quality.storage, "get", lambda key: blobs[key])

    summary = audit_selfplay_quality.audit(limit_files=10, out=io.StringIO())

    assert summary.selected_games == 1
    assert summary.unknown_git_sha_games == 1
    assert summary.resignation_games == 1
    assert summary.missing_trace_games == 1
    assert summary.illegal_policy_rows == 2
    assert summary.legal_count_mismatch_rows == 0
    assert summary.wdl_terminal_bad_rows == 0
