"""In-memory, incrementally-synced store powering the training dashboard.

The dashboard used to call into S3 on every request, re-listing data prefixes
and re-downloading the full elo state + games log each time. That made first
paint slow and limited refresh cadence.

This module runs a single background thread that periodically reconciles a
snapshot dict with S3. Each data source uses the cheapest change-detection
primitive available:

    - ``models/latest.meta.json`` and ``state/elo.json``: HEAD + ETag, re-GET
      only when the ETag changes.
    - ``state/elo_games.jsonl``: HEAD + size, ranged GET for the appended tail
      only. Parsed lines are kept in a bounded deque.
    - ``data/selfplay/`` and ``data/imitation/``: full LIST each cycle, but
      aggregates are maintained incrementally — new keys only are parsed, and
      the aggregate dict is rebuilt only when the file set actually changes.
    - ``heartbeats/``: LIST with LastModified, GET only workers whose modified
      time advanced.
    - ``models/versions/``: plain LIST (names only).

The HTTP handler just reads ``store.snapshot()`` — no S3 calls on the hot path.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable

from . import storage

_GAMES_TAIL = 50


class DashboardStore:
    """Background-synced, thread-safe dashboard state.

    Call ``start()`` once to launch the refresh loop. ``snapshot()`` returns the
    latest status dict (safe to serialise without the lock).
    """

    def __init__(self, storage_mod=storage, interval: float = 60.0) -> None:
        self._s = storage_mod
        self._interval = interval
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._etag_meta: str | None = None
        self._etag_elo: str | None = None
        self._model: dict = {"version": 0, "promoted_at": None}
        self._elo: dict = {}

        self._sp_files: dict[str, tuple[int, str]] = {}
        self._im_files: dict[str, tuple[int, str]] = {}
        self._sp_agg: dict[str, Any] = _empty_agg()
        self._im_agg: dict[str, Any] = _empty_agg()

        self._games_offset: int = 0
        self._games_buf: bytes = b""
        self._games: deque[dict] = deque(maxlen=_GAMES_TAIL)

        # name -> {"lm": iso, "data": {...}}
        self._heartbeats: dict[str, dict] = {}

        self._snapshots: list[dict] = []

        self._last_sync: str | None = None
        self._last_error: str | None = None
        self._initialised = False

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self.refresh_once()
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._loop, name="dashboard-store", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.refresh_once()
            except Exception as e:  # noqa: BLE001 — background loop must not die
                with self._lock:
                    self._last_error = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------- sync

    def refresh_once(self) -> None:
        self._model = self._sync_etagged_json(
            storage.LATEST_META,
            "_etag_meta",
            lambda d: {
                "version": d.get("version", 0),
                "promoted_at": d.get("timestamp"),
            },
            fallback=self._model,
        )
        self._elo = self._sync_etagged_json(
            storage.ELO_STATE, "_etag_elo", lambda d: d, fallback=self._elo
        )
        self._sync_games()
        self._sync_data(storage.SELFPLAY_PREFIX, self._sp_files, "_sp_agg")
        self._sync_data(storage.IMITATION_PREFIX, self._im_files, "_im_agg")
        self._sync_heartbeats()
        self._sync_snapshots()

        with self._lock:
            self._last_sync = datetime.now(timezone.utc).isoformat()
            self._last_error = None
            self._initialised = True

    def _sync_etagged_json(
        self,
        key: str,
        etag_attr: str,
        transform: Callable[[dict], dict],
        *,
        fallback: dict,
    ) -> dict:
        meta = self._s.head(key)
        if meta is None:
            return fallback
        if meta["etag"] == getattr(self, etag_attr):
            return fallback
        try:
            data = self._s.get_json(key)
        except KeyError:
            return fallback
        with self._lock:
            setattr(self, etag_attr, meta["etag"])
        return transform(data)

    def _sync_games(self) -> None:
        meta = self._s.head(storage.ELO_GAMES_LOG)
        if meta is None:
            return

        size = meta["size"]
        # Shrunk file = truncation/rotation. Restart from zero.
        if size < self._games_offset:
            self._games_offset = 0
            self._games_buf = b""
            self._games.clear()

        if size <= self._games_offset:
            return

        new_bytes = self._s.get_range(
            storage.ELO_GAMES_LOG, self._games_offset, size - 1
        )
        if not new_bytes:
            return

        buf = self._games_buf + new_bytes
        lines = buf.split(b"\n")
        # Trailing element is either empty (clean newline) or a partial line
        # to carry over.
        partial = lines[-1]

        parsed: list[dict] = []
        for line in lines[:-1]:
            if not line:
                continue
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        with self._lock:
            for rec in parsed:
                self._games.append(rec)
            self._games_offset = size
            self._games_buf = partial

    def _sync_data(
        self,
        prefix: str,
        cache: dict[str, tuple[int, str]],
        agg_attr: str,
    ) -> None:
        listed = self._s.list_data_files(prefix)
        seen = {f["key"] for f in listed}

        to_add = [
            (f["key"], f["positions"], f["version"])
            for f in listed
            if f["key"] not in cache
        ]
        stale = [k for k in cache if k not in seen]
        if not to_add and not stale:
            return

        with self._lock:
            for key, pos, ver in to_add:
                cache[key] = (pos, ver)
            for k in stale:
                cache.pop(k, None)
            setattr(self, agg_attr, _aggregate(cache))

    def _sync_heartbeats(self) -> None:
        listed = self._s.list_with_meta(storage.HEARTBEATS_PREFIX)
        seen_names: set[str] = set()
        updates: dict[str, dict] = {}
        for obj in listed:
            name = obj["key"].rsplit("/", 1)[-1].removesuffix(".json")
            if not name:
                continue
            seen_names.add(name)
            lm = obj["last_modified"]
            lm_iso = lm.isoformat() if hasattr(lm, "isoformat") else str(lm)
            cached = self._heartbeats.get(name)
            if cached is not None and cached.get("lm") == lm_iso:
                continue
            try:
                data = self._s.get_json(obj["key"])
            except KeyError:
                continue
            updates[name] = {"lm": lm_iso, "data": data}

        with self._lock:
            for name, entry in updates.items():
                self._heartbeats[name] = entry
            for name in list(self._heartbeats):
                if name not in seen_names:
                    self._heartbeats.pop(name, None)

    def _sync_snapshots(self) -> None:
        keys = self._s.ls(storage.VERSIONS_PREFIX)
        snaps = [{"name": k.rsplit("/", 1)[-1]} for k in keys]
        with self._lock:
            self._snapshots = snaps

    # -------------------------------------------------------------- read-side

    def snapshot(self) -> dict:
        """Return the current status dict. Shape matches the old collect_status."""
        with self._lock:
            im_agg = {k: v for k, v in self._im_agg.items() if k != "by_version"}
            return {
                "model": dict(self._model),
                "workers": {n: e["data"] for n, e in self._heartbeats.items()},
                "elo": {
                    "ratings": self._elo.get("ratings", {}),
                    "total_games": self._elo.get("total_games", 0),
                    "active_players": self._elo.get("active_players", []),
                    "retired_players": self._elo.get("retired_players", []),
                    "player_stats": self._elo.get("player_stats", {}),
                    "pair_results": self._elo.get("pair_results", {}),
                },
                "recent_games": list(self._games),
                "data": {
                    "selfplay": dict(self._sp_agg),
                    "imitation": im_agg,
                    "models": list(self._snapshots),
                },
                "timestamp": self._last_sync
                or datetime.now(timezone.utc).isoformat(),
                "sync_error": self._last_error,
                "initialised": self._initialised,
            }


def _empty_agg() -> dict[str, Any]:
    return {"total_files": 0, "total_positions": 0, "by_version": {}}


def _aggregate(cache: dict[str, tuple[int, str]]) -> dict[str, Any]:
    by_version: dict[str, dict[str, int]] = {}
    total_positions = 0
    for pos, ver in cache.values():
        total_positions += pos
        bucket = by_version.setdefault(ver, {"count": 0, "positions": 0})
        bucket["count"] += 1
        bucket["positions"] += pos
    return {
        "total_files": len(cache),
        "total_positions": total_positions,
        "by_version": by_version,
    }
