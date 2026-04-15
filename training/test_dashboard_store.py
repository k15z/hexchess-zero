"""Tests for DashboardStore incremental sync."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from training import storage
from training.dashboard_store import DashboardStore


class FakeStorage:
    def __init__(self) -> None:
        self._objs: dict[str, dict] = {}
        self._etag_counter = 0

    def put(self, key: str, body: bytes | str) -> None:
        if isinstance(body, str):
            body = body.encode()
        self._etag_counter += 1
        self._objs[key] = {
            "body": body,
            "etag": f"etag-{self._etag_counter}",
            "last_modified": datetime.now(timezone.utc),
        }

    def put_json(self, key: str, obj: dict) -> None:
        self.put(key, json.dumps(obj))

    def delete(self, key: str) -> None:
        self._objs.pop(key, None)

    def head(self, key: str) -> dict | None:
        obj = self._objs.get(key)
        if obj is None:
            return None
        return {
            "etag": obj["etag"],
            "size": len(obj["body"]),
            "last_modified": obj["last_modified"],
        }

    def get(self, key: str) -> bytes:
        obj = self._objs.get(key)
        if obj is None:
            raise KeyError(key)
        return obj["body"]

    def get_json(self, key: str) -> dict:
        return json.loads(self.get(key))

    def list_data_files(self, prefix: str) -> list[dict]:
        import re

        pat = re.compile(r"_n(\d+)\.npz$")
        out = []
        for key in self._objs:
            if not key.startswith(prefix):
                continue
            match = pat.search(key)
            if not match:
                continue
            parts = key.split("/")
            version = parts[2] if len(parts) >= 3 else "unknown"
            out.append(
                {
                    "key": key,
                    "positions": int(match.group(1)),
                    "timestamp": key.rsplit("/", 1)[-1].split("_")[0],
                    "version": version,
                }
            )
        return out

    def list_with_meta(self, prefix: str) -> list[dict]:
        return [
            {
                "key": key,
                "size": len(obj["body"]),
                "last_modified": obj["last_modified"],
                "etag": obj["etag"],
            }
            for key, obj in self._objs.items()
            if key.startswith(prefix)
        ]

    def ls(self, prefix: str) -> list[str]:
        return [key for key in self._objs if key.startswith(prefix)]

    def list_eval_versions(self) -> list[int]:
        versions = set()
        for key in self._objs:
            if not key.startswith(storage.EVALS_PREFIX):
                continue
            parts = key.split("/")
            if len(parts) < 3:
                continue
            name = parts[2]
            if name.startswith("v"):
                versions.add(int(name[1:]))
        return sorted(versions)

    def list_eval_game_record_keys(self, version: int | str) -> list[str]:
        return sorted(self.ls(storage.eval_games_prefix(version)))


@pytest.fixture
def fake() -> FakeStorage:
    return FakeStorage()


@pytest.fixture
def store(fake: FakeStorage) -> DashboardStore:
    return DashboardStore(storage_mod=fake, interval=3600.0)


def test_empty_snapshot_has_expected_shape(store: DashboardStore) -> None:
    store.refresh_once()
    snap = store.snapshot()

    assert snap["model"] == {"version": 0, "promoted_at": None, "positions_at_promote": None}
    assert snap["approved_model"] == {"version": 0, "promoted_at": None}
    assert snap["evaluations"]["versions"] == []
    assert snap["evaluations"]["focus"] is None
    assert snap["recent_games"] == []
    assert snap["workers"] == {}
    assert snap["initialised"] is True


def test_eval_summaries_follow_etags(fake: FakeStorage, store: DashboardStore) -> None:
    fake.put_json(storage.LATEST_META, {"version": 5, "timestamp": "2026-04-15T00:00:00Z"})
    fake.put_json(storage.APPROVED_META, {"version": 4, "timestamp": "2026-04-14T00:00:00Z"})
    fake.put_json(storage.eval_gate_summary_key(5), {"status": "pending", "games": 10})
    fake.put_json(storage.eval_benchmark_summary_key(5), {"status": "pending", "games": 8})
    fake.put_json(storage.eval_decision_key(5), {"status": "pending", "updated_at": "2026-04-15T01:00:00Z"})

    store.refresh_once()
    snap = store.snapshot()
    assert snap["evaluations"]["focus_version"] == 5
    assert snap["evaluations"]["focus"]["gate"]["games"] == 10

    calls: list[str] = []
    real = fake.get_json

    def tracking(key: str) -> dict:
        calls.append(key)
        return real(key)

    fake.get_json = tracking  # type: ignore[method-assign]
    store.refresh_once()
    assert calls == []

    fake.put_json(storage.eval_gate_summary_key(5), {"status": "approved", "games": 20})
    store.refresh_once()
    assert calls == [storage.eval_gate_summary_key(5)]
    assert store.snapshot()["evaluations"]["focus"]["gate"]["status"] == "approved"


def test_recent_games_follow_focus_version(fake: FakeStorage, store: DashboardStore) -> None:
    fake.put_json(storage.LATEST_META, {"version": 6, "timestamp": "2026-04-15T00:00:00Z"})
    fake.put_json(storage.APPROVED_META, {"version": 5, "timestamp": "2026-04-14T00:00:00Z"})
    fake.put_json(storage.eval_decision_key(6), {"status": "pending", "updated_at": "2026-04-15T01:00:00Z"})
    fake.put_json(storage.eval_benchmark_summary_key(6), {"status": "pending", "games": 0})
    fake.put_json(storage.eval_gate_summary_key(6), {"status": "pending", "games": 0})
    fake.put_json(
        storage.eval_games_prefix(6) + "20260415T010000_aaaa.json",
        {"candidate": "v6", "opponent": "v5", "outcome": "white"},
    )

    store.refresh_once()
    assert len(store.snapshot()["recent_games"]) == 1

    fake.put_json(
        storage.eval_games_prefix(6) + "20260415T010100_bbbb.json",
        {"candidate": "v6", "opponent": "Minimax-2", "outcome": "draw"},
    )

    calls: list[str] = []
    real = fake.get_json

    def tracking(key: str) -> dict:
        calls.append(key)
        return real(key)

    fake.get_json = tracking  # type: ignore[method-assign]
    store.refresh_once()
    assert calls == [storage.eval_games_prefix(6) + "20260415T010100_bbbb.json"]
    assert len(store.snapshot()["recent_games"]) == 2


def test_data_and_worker_sync_still_work(fake: FakeStorage, store: DashboardStore) -> None:
    fake.put("data/selfplay/v4/20260415T000000_aaaa_n100.npz", b"x")
    fake.put("data/imitation/20260415T000000_bbbb_n50.npz", b"x")
    fake.put_json("heartbeats/worker-a.json", {"model_version": 4, "total_games": 10, "timestamp": "2026-04-15T00:00:00Z"})

    store.refresh_once()
    snap = store.snapshot()
    assert snap["data"]["selfplay"]["total_positions"] == 100
    assert snap["data"]["imitation"]["total_positions"] == 50
    assert snap["workers"]["worker-a"]["total_games"] == 10


def test_trainer_metrics_sync(fake: FakeStorage, store: DashboardStore) -> None:
    fake.put_json(
        storage.TRAINER_METRICS,
        {"summaries": [{"summary": 9, "loss_policy": 0.5, "version": 6}]},
    )

    store.refresh_once()
    assert store.snapshot()["trainer_metrics"]["summaries"][0]["summary"] == 9
