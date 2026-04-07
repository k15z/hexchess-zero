"""Tests for DashboardStore incremental sync.

These exercise the store against a fake storage module so we can drive
state changes (file appends, ETag rotations, heartbeat updates) and verify
the snapshot reflects them without any real S3 traffic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from training import storage
from training.dashboard_store import DashboardStore


class FakeStorage:
    """Minimal in-memory stand-in for the ``training.storage`` module."""

    def __init__(self) -> None:
        # key -> {"body": bytes, "etag": str, "last_modified": datetime}
        self._objs: dict[str, dict] = {}
        self._etag_counter = 0

    # ------------------------------------------------------------ test helpers

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

    def append(self, key: str, body: bytes | str) -> None:
        if isinstance(body, str):
            body = body.encode()
        cur = self._objs.get(key)
        new_body = (cur["body"] if cur else b"") + body
        self.put(key, new_body)

    def delete(self, key: str) -> None:
        self._objs.pop(key, None)

    # --------------------------------------------- storage-module-like surface

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

    def get_range(self, key: str, start: int, end: int | None = None) -> bytes:
        obj = self._objs.get(key)
        if obj is None:
            raise KeyError(key)
        body = obj["body"]
        if start >= len(body):
            return b""
        if end is None:
            return body[start:]
        return body[start : end + 1]

    def list_data_files(self, prefix: str) -> list[dict]:
        import re

        pat = re.compile(r"_n(\d+)\.npz$")
        out = []
        for k in self._objs:
            if not k.startswith(prefix):
                continue
            m = pat.search(k)
            if not m:
                continue
            parts = k.split("/")
            version = parts[2] if len(parts) >= 3 else "unknown"
            out.append({
                "key": k,
                "positions": int(m.group(1)),
                "timestamp": k.rsplit("/", 1)[-1].split("_")[0],
                "version": version,
            })
        return out

    def list_with_meta(self, prefix: str) -> list[dict]:
        return [
            {
                "key": k,
                "size": len(v["body"]),
                "last_modified": v["last_modified"],
                "etag": v["etag"],
            }
            for k, v in self._objs.items()
            if k.startswith(prefix)
        ]

    def ls(self, prefix: str) -> list[str]:
        return [k for k in self._objs if k.startswith(prefix)]


@pytest.fixture
def fake() -> FakeStorage:
    return FakeStorage()


@pytest.fixture
def store(fake: FakeStorage) -> DashboardStore:
    return DashboardStore(storage_mod=fake, interval=3600.0)


def test_empty_snapshot_has_expected_shape(store: DashboardStore) -> None:
    store.refresh_once()
    snap = store.snapshot()
    assert snap["model"] == {"version": 0, "promoted_at": None}
    assert snap["workers"] == {}
    assert snap["recent_games"] == []
    assert snap["data"]["selfplay"]["total_positions"] == 0
    assert snap["data"]["selfplay"]["total_files"] == 0
    assert snap["data"]["imitation"]["total_files"] == 0
    assert snap["initialised"] is True


def test_model_and_elo_use_etag_skip(
    fake: FakeStorage, store: DashboardStore
) -> None:
    fake.put_json(storage.LATEST_META, {"version": 3, "timestamp": "2026-04-06T00:00:00Z"})
    fake.put_json(
        storage.ELO_STATE,
        {"ratings": {"alice": {"mu": 25, "sigma": 8}}, "total_games": 10},
    )

    store.refresh_once()
    snap = store.snapshot()
    assert snap["model"]["version"] == 3
    assert snap["elo"]["total_games"] == 10

    # If nothing changed, ETag match should cause get_json NOT to be called.
    calls = {"n": 0}
    real_get_json = fake.get_json

    def counting_get_json(key: str) -> dict:
        calls["n"] += 1
        return real_get_json(key)

    fake.get_json = counting_get_json  # type: ignore[method-assign]
    store.refresh_once()
    # Neither meta nor elo should have been re-downloaded (heartbeats empty).
    assert calls["n"] == 0

    # But bumping the elo state (new etag) triggers exactly one GET.
    fake.put_json(
        storage.ELO_STATE,
        {"ratings": {"alice": {"mu": 26, "sigma": 8}}, "total_games": 11},
    )
    store.refresh_once()
    assert calls["n"] == 1
    assert store.snapshot()["elo"]["total_games"] == 11


def test_games_per_object_incremental_fetch(
    fake: FakeStorage, store: DashboardStore
) -> None:
    k1 = storage.ELO_GAMES_PREFIX + "20260101T000000_aaaa.json"
    k2 = storage.ELO_GAMES_PREFIX + "20260101T000100_bbbb.json"
    fake.put_json(k1, {"game": 1, "white": "a", "black": "b", "outcome": "white"})
    store.refresh_once()
    assert [g["game"] for g in store.snapshot()["recent_games"]] == [1]

    # Add a second game — only the new key should be GET'd.
    fake.put_json(k2, {"game": 2, "white": "a", "black": "b", "outcome": "draw"})

    gets: list[str] = []
    real_get_json = fake.get_json

    def tracking_get_json(key: str) -> dict:
        gets.append(key)
        return real_get_json(key)

    fake.get_json = tracking_get_json  # type: ignore[method-assign]
    store.refresh_once()

    assert gets == [k2]
    assert [g["game"] for g in store.snapshot()["recent_games"]] == [1, 2]


def test_data_files_incremental_aggregation(
    fake: FakeStorage, store: DashboardStore
) -> None:
    fake.put("data/selfplay/v1/20260101T000000_aaaa_n100.npz", b"x")
    fake.put("data/selfplay/v1/20260102T000000_bbbb_n250.npz", b"x")
    fake.put("data/selfplay/v2/20260103T000000_cccc_n50.npz", b"x")
    fake.put("data/imitation/20260101T000000_dddd_n1000.npz", b"x")
    store.refresh_once()

    sp = store.snapshot()["data"]["selfplay"]
    assert sp["total_files"] == 3
    assert sp["total_positions"] == 400
    assert sp["by_version"]["v1"] == {"count": 2, "positions": 350}
    assert sp["by_version"]["v2"] == {"count": 1, "positions": 50}
    assert store.snapshot()["data"]["imitation"]["total_positions"] == 1000

    # Add a new file — only the new key should be parsed; old aggregates persist.
    fake.put("data/selfplay/v2/20260104T000000_eeee_n75.npz", b"x")
    store.refresh_once()
    sp = store.snapshot()["data"]["selfplay"]
    assert sp["total_files"] == 4
    assert sp["total_positions"] == 475
    assert sp["by_version"]["v2"] == {"count": 2, "positions": 125}

    # Remove a file — it should drop out of the aggregate.
    fake.delete("data/selfplay/v1/20260101T000000_aaaa_n100.npz")
    store.refresh_once()
    sp = store.snapshot()["data"]["selfplay"]
    assert sp["total_files"] == 3
    assert sp["total_positions"] == 375
    assert sp["by_version"]["v1"] == {"count": 1, "positions": 250}


def test_heartbeats_only_refetch_changed(
    fake: FakeStorage, store: DashboardStore
) -> None:
    fake.put_json("heartbeats/worker-a.json", {"total_games": 10, "timestamp": "t"})
    fake.put_json("heartbeats/worker-b.json", {"total_games": 20, "timestamp": "t"})
    store.refresh_once()
    assert set(store.snapshot()["workers"].keys()) == {"worker-a", "worker-b"}

    calls: list[str] = []
    real_get_json = fake.get_json

    def tracking_get_json(key: str) -> dict:
        calls.append(key)
        return real_get_json(key)

    fake.get_json = tracking_get_json  # type: ignore[method-assign]
    store.refresh_once()
    # Nothing changed → no GETs for heartbeats or the small JSONs.
    assert calls == []

    # Mutate worker-a only.
    fake.put_json("heartbeats/worker-a.json", {"total_games": 11, "timestamp": "t"})
    store.refresh_once()
    assert calls == ["heartbeats/worker-a.json"]
    assert store.snapshot()["workers"]["worker-a"]["total_games"] == 11

    # Remove worker-b → should drop from snapshot.
    fake.delete("heartbeats/worker-b.json")
    store.refresh_once()
    assert "worker-b" not in store.snapshot()["workers"]


def test_snapshots_listing(fake: FakeStorage, store: DashboardStore) -> None:
    fake.put("models/versions/1.onnx", b"x")
    fake.put("models/versions/2.onnx", b"x")
    store.refresh_once()
    names = [s["name"] for s in store.snapshot()["data"]["models"]]
    assert sorted(names) == ["1.onnx", "2.onnx"]


