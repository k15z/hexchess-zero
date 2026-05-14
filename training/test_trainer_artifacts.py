from __future__ import annotations

import io
from collections import deque

import numpy as np
import pytest
import torch

import training.config as config_module
from training import storage, trainer_loop
from training.config import AsyncConfig


def _npz_bytes(*, root_entropy: list[float]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        root_entropy=np.asarray(root_entropy, dtype=np.float16),
    )
    return buf.getvalue()


def test_build_recent_selfplay_summary_reports_sampled_metrics(monkeypatch):
    files = [
        {
            "key": "data/selfplay/v1/20260420T000000_aaaa_n10.npz",
            "positions": 10,
            "timestamp": "20260420T000000",
            "version": "v1",
        },
        {
            "key": "data/selfplay/v1/20260421T000000_bbbb_n20.npz",
            "positions": 20,
            "timestamp": "20260421T000000",
            "version": "v1",
        },
        {
            "key": "data/selfplay/v2/20260422T000000_cccc_n30.npz",
            "positions": 30,
            "timestamp": "20260422T000000",
            "version": "v2",
        },
    ]
    metas = {
        "data/selfplay/v1/20260420T000000_aaaa_n10.meta.json": {
            "num_total_positions": 12,
            "result": "draw",
            "termination": "stalemate",
        },
        "data/selfplay/v1/20260421T000000_bbbb_n20.meta.json": {
            "num_total_positions": 25,
            "result": "white",
            "termination": "checkmate_white",
        },
        "data/selfplay/v2/20260422T000000_cccc_n30.meta.json": {
            "num_total_positions": 40,
            "result": "black",
            "termination": "checkmate_black",
        },
    }
    blobs = {
        "data/selfplay/v1/20260420T000000_aaaa_n10.npz": _npz_bytes(root_entropy=[1.0, 2.0]),
        "data/selfplay/v1/20260421T000000_bbbb_n20.npz": _npz_bytes(root_entropy=[3.0]),
        "data/selfplay/v2/20260422T000000_cccc_n30.npz": _npz_bytes(root_entropy=[4.0, 5.0, 6.0]),
    }

    monkeypatch.setattr(storage, "select_recent_files", lambda prefix, max_positions: files)
    monkeypatch.setattr(storage, "get_json", lambda key: metas[key])
    monkeypatch.setattr(storage, "get", lambda key: blobs[key])

    summary = trainer_loop._build_recent_selfplay_summary(60)

    assert summary["selected_files"] == 3
    assert summary["selected_full_positions"] == 60
    assert summary["selected_versions"] == {"v1": 2, "v2": 1}
    assert summary["positions_per_day"] == 30.0
    assert summary["sampled_games_with_meta"] == 3
    assert summary["sampled_mean_game_length"] == 25.67
    assert summary["sampled_draw_rate"] == 0.3333
    assert summary["sampled_keep_ratio"] == pytest.approx(60 / 77, rel=1e-4)
    assert summary["sampled_result_counts"] == {"black": 1, "draw": 1, "white": 1}
    assert summary["sampled_termination_counts"] == {
        "checkmate_black": 1,
        "checkmate_white": 1,
        "stalemate": 1,
    }
    assert summary["root_entropy"]["mean_root_entropy"] == 3.5
    assert summary["root_entropy"]["sampled_positions"] == 6


def test_publish_candidate_model_writes_versioned_metadata_and_summary(monkeypatch, tmp_path):
    cfg = AsyncConfig(run_id="test-run")
    monkeypatch.setattr(config_module, "_cache_root", lambda: tmp_path)
    monkeypatch.setattr(
        trainer_loop,
        "_build_promotion_summary",
        lambda **kwargs: {"version": kwargs["version"], "summary": "ok"},
    )
    monkeypatch.setattr(
        trainer_loop,
        "export_to_onnx",
        lambda local_pt, local_onnx, cfg: local_onnx.write_bytes(b"onnx"),
    )

    file_uploads: list[str] = []
    json_writes: dict[str, dict] = {}
    copies: list[tuple[str, str]] = []

    monkeypatch.setattr(
        storage,
        "put_file",
        lambda key, local_path: file_uploads.append(key),
    )
    monkeypatch.setattr(
        storage,
        "put_json",
        lambda key, obj: json_writes.setdefault(key, obj),
    )
    monkeypatch.setattr(
        storage,
        "copy",
        lambda src, dst: copies.append((src, dst)),
    )

    model = torch.nn.Linear(1, 1)
    trainer_loop._publish_candidate_model(
        cfg,
        model,
        7,
        positions_at_promote=123,
        replay_window_size=456,
        imitation_mix=0.2,
        selfplay_tranche_positions=100,
        source_version=6,
        training_phase="selfplay",
    )

    assert storage.version_onnx_key(7) in file_uploads
    assert storage.version_checkpoint_key(7) in file_uploads
    assert storage.CHECKPOINT_PT in file_uploads
    assert copies == [
        (storage.version_onnx_key(7), storage.LATEST_ONNX),
    ]
    assert json_writes[storage.version_meta_key(7)]["version"] == 7
    assert json_writes[storage.version_meta_key(7)]["positions_at_promote"] == 123
    assert json_writes[storage.version_meta_key(7)]["replay_window_size"] == 456
    assert json_writes[storage.version_meta_key(7)]["imitation_mix"] == 0.2
    assert json_writes[storage.version_meta_key(7)]["source_version"] == 6
    assert json_writes[storage.version_meta_key(7)]["config_snapshot"]["run_id"] == "test-run"
    assert json_writes[storage.promotion_summary_key(7)] == {"version": 7, "summary": "ok"}
    assert json_writes[storage.LATEST_META]["version_metadata_key"] == storage.version_meta_key(7)
    assert json_writes[storage.LATEST_META]["promotion_summary_key"] == storage.promotion_summary_key(7)


def test_publish_live_weights_writes_separate_inspection_artifact(monkeypatch, tmp_path):
    cfg = AsyncConfig(run_id="test-run")
    monkeypatch.setattr(config_module, "_cache_root", lambda: tmp_path)

    file_uploads: list[str] = []
    json_writes: dict[str, dict] = {}

    monkeypatch.setattr(
        storage,
        "put_file",
        lambda key, local_path: file_uploads.append(key),
    )
    monkeypatch.setattr(
        storage,
        "put_json",
        lambda key, obj: json_writes.setdefault(key, obj),
    )

    model = torch.nn.Linear(1, 1)
    trainer_loop._publish_live_weights(
        cfg,
        model,
        published_version=3,
        summary=12,
        total_steps=12_345,
        n_total=987_654,
        positions_at_last_promote=500_000,
    )

    assert file_uploads == [storage.TRAINER_LIVE_WEIGHTS]
    assert json_writes[storage.TRAINER_LIVE_WEIGHTS_META]["published_version"] == 3
    assert json_writes[storage.TRAINER_LIVE_WEIGHTS_META]["summary"] == 12
    assert json_writes[storage.TRAINER_LIVE_WEIGHTS_META]["total_steps"] == 12_345
    assert json_writes[storage.TRAINER_LIVE_WEIGHTS_META]["n_total"] == 987_654
    assert (
        json_writes[storage.TRAINER_LIVE_WEIGHTS_META]["positions_at_last_promote"]
        == 500_000
    )
    assert "timestamp" in json_writes[storage.TRAINER_LIVE_WEIGHTS_META]


def test_publish_resume_checkpoint_writes_resumable_state(monkeypatch, tmp_path):
    cfg = AsyncConfig(run_id="test-run")
    monkeypatch.setattr(config_module, "_cache_root", lambda: tmp_path)

    file_uploads: list[str] = []
    json_writes: dict[str, dict] = {}

    monkeypatch.setattr(
        storage,
        "put_file",
        lambda key, local_path: file_uploads.append(key),
    )
    monkeypatch.setattr(
        storage,
        "put_json",
        lambda key, obj: json_writes.setdefault(key, obj),
    )

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    bucket = trainer_loop.TrainBucket(4.0, batch_size=256)
    bucket.update(1_000)
    swa_buf = trainer_loop.SwaSnapshotBuffer(max_snapshots=1, promotion_weights=(1.0,))

    trainer_loop._publish_trainer_resume_checkpoint(
        cfg,
        model,
        optimizer,
        current_version=1,
        report_index=4,
        total_steps_all_time=1234,
        fresh_run_steps=1234,
        positions_at_last_promote=0,
        last_promotion_attempt_step=-1,
        bucket=bucket,
        swa_buf=swa_buf,
        samples_since_last_snapshot=512,
        bn_refresh_batches=deque([torch.zeros(2, 22, 11, 11)]),
        n_total=1_100_000,
    )

    assert file_uploads == [storage.TRAINER_RESUME_CHECKPOINT]
    meta = json_writes[storage.TRAINER_RESUME_CHECKPOINT_META]
    assert meta["current_version"] == 1
    assert meta["total_steps_all_time"] == 1234
    assert meta["n_total"] == 1_100_000


def test_optimizer_state_dict_to_cpu_does_not_mutate_live_optimizer():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    loss = model(torch.ones(1, 1)).sum()
    loss.backward()
    optimizer.step()

    live_slot = next(iter(optimizer.state.values()))
    live_momentum = live_slot["momentum_buffer"]

    checkpoint_state = trainer_loop._optimizer_state_dict_to_cpu(optimizer)
    checkpoint_slot = next(iter(checkpoint_state["state"].values()))

    assert live_slot["momentum_buffer"] is live_momentum
    assert checkpoint_slot["momentum_buffer"] is not live_momentum
    assert checkpoint_slot["momentum_buffer"].device.type == "cpu"
