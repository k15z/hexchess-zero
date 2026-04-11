"""In-memory, incrementally-synced store powering the training dashboard.

The dashboard used to call into S3 on every request, re-listing data prefixes
and re-downloading the full elo state + games log each time. That made first
paint slow and limited refresh cadence.

This module runs a single background thread that periodically reconciles a
snapshot dict with S3. Each data source uses the cheapest change-detection
primitive available:

    - ``models/latest.meta.json`` and ``state/elo.json``: HEAD + ETag, re-GET
      only when the ETag changes.
    - ``state/elo_games/``: LIST the prefix, fetch only newly-seen keys. The
      per-game object layout replaced the old append-only jsonl so multiple
      elo-service replicas can write concurrently.
    - ``data/selfplay/`` and ``data/imitation/``: full LIST each cycle, but
      aggregates are maintained incrementally — new keys only are parsed, and
      the aggregate dict is rebuilt only when the file set actually changes.
    - ``heartbeats/``: LIST with LastModified, GET only workers whose modified
      time advanced.
    - ``models/versions/``: plain LIST (names only).

The HTTP handler just reads ``store.snapshot()`` — no S3 calls on the hot path.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Callable

from . import storage

_GAMES_TAIL = 50

# Heartbeat keys older than this are deleted from S3 by the dashboard sync.
# Workers refresh their heartbeat at startup and after every batch, so any key
# older than this belongs to a retired pod (k8s rolling restart, OOM, scale-in)
# whose name will never recur.
_HEARTBEAT_STALE_SECONDS = 2 * 60 * 60


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
        self._etag_trainer: str | None = None
        self._model: dict = {"version": 0, "promoted_at": None}
        self._elo: dict = {}
        self._trainer_metrics: dict = {}

        self._sp_files: dict[str, tuple[int, str]] = {}
        self._im_files: dict[str, tuple[int, str]] = {}
        self._sp_agg: dict[str, Any] = _empty_agg()
        self._im_agg: dict[str, Any] = _empty_agg()

        self._game_records: dict[str, dict] = {}  # key -> record (tail window)

        # name -> {"lm": iso, "data": {...}}
        self._heartbeats: dict[str, dict] = {}

        # benchmark: version_str -> result dict (immutable, permanent cache)
        self._benchmark_results: dict[str, dict] = {}
        self._benchmark_versions: list[dict] = []

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
        self._trainer_metrics = self._sync_etagged_json(
            storage.TRAINER_METRICS,
            "_etag_trainer",
            lambda d: d,
            fallback=self._trainer_metrics,
        )
        self._sync_games()
        self._sync_data(storage.SELFPLAY_PREFIX, self._sp_files, "_sp_agg")
        self._sync_data(storage.IMITATION_PREFIX, self._im_files, "_im_agg")
        self._sync_heartbeats()
        self._sync_snapshots()
        self._sync_benchmarks()

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
        # Only the most recent _GAMES_TAIL records matter for the dashboard;
        # LIST is cheap, but GET-per-object isn't — fetch only keys we haven't
        # already cached and early-out when the tail window is unchanged.
        keys = sorted(self._s.ls(storage.ELO_GAMES_PREFIX), reverse=True)
        tail = set(keys[:_GAMES_TAIL])
        if tail == self._game_records.keys():
            return

        new_records = {}
        for k in tail:
            if k in self._game_records:
                new_records[k] = self._game_records[k]
                continue
            try:
                new_records[k] = self._s.get_json(k)
            except KeyError:
                continue

        with self._lock:
            self._game_records = new_records

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
        now = datetime.now(timezone.utc)
        seen_names: set[str] = set()
        updates: dict[str, dict] = {}
        for obj in listed:
            name = obj["key"].rsplit("/", 1)[-1].removesuffix(".json")
            if not name:
                continue
            lm = obj["last_modified"]
            # Garbage-collect heartbeats from retired pods. Pod names contain
            # a replicaset hash that changes on every redeploy, so stale keys
            # accumulate forever otherwise.
            if hasattr(lm, "tzinfo"):
                age = (now - lm).total_seconds()
                if age > _HEARTBEAT_STALE_SECONDS:
                    try:
                        self._s.delete(obj["key"])
                    except Exception:
                        pass
                    continue
            seen_names.add(name)
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

    def _sync_benchmarks(self) -> None:
        # LIST available result versions; benchmark files are immutable so we
        # never re-fetch a version we've already downloaded.
        _MAX_VERSIONS = 10
        keys = self._s.ls(storage.BENCHMARK_RESULTS_PREFIX)
        available = []
        for k in keys:
            name = k.rsplit("/", 1)[-1]
            if name.startswith("v") and name.endswith(".json"):
                try:
                    ver = int(name[1:-5])
                    available.append({"key": k, "name": name, "version": ver})
                except ValueError:
                    continue
        available.sort(key=lambda x: x["version"])
        to_use = available[-_MAX_VERSIONS:]

        new_results: dict[str, dict] = {}
        for entry in to_use:
            ver_str = entry["name"][:-5]  # "v42"
            if ver_str in self._benchmark_results:
                new_results[ver_str] = self._benchmark_results[ver_str]
                continue
            try:
                new_results[ver_str] = self._s.get_json(entry["key"])
            except KeyError:
                continue

        with self._lock:
            self._benchmark_versions = [
                {"name": e["name"], "version": e["version"]} for e in to_use
            ]
            self._benchmark_results = new_results

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
                "recent_games": [
                    self._game_records[k] for k in sorted(self._game_records)
                ],
                "data": {
                    "selfplay": dict(self._sp_agg),
                    "imitation": im_agg,
                    "models": list(self._snapshots),
                },
                "trainer_metrics": dict(self._trainer_metrics),
                "benchmark_versions": list(self._benchmark_versions),
                "benchmark_results": {
                    v: dict(r) for v, r in self._benchmark_results.items()
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
