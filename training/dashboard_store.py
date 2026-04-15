"""In-memory, incrementally-synced store powering the training dashboard."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Callable

from . import storage

_RECENT_GAMES_TAIL = 40
_EVAL_MAX_VERSIONS = 8
_BENCHMARK_MAX_VERSIONS = 10
_HEARTBEAT_STALE_SECONDS = 2 * 60 * 60


class DashboardStore:
    """Background-synced, thread-safe dashboard state."""

    def __init__(self, storage_mod=storage, interval: float = 60.0) -> None:
        self._s = storage_mod
        self._interval = interval
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._etag_meta: str | None = None
        self._etag_approved_meta: str | None = None
        self._etag_trainer: str | None = None
        self._model: dict = {"version": 0, "promoted_at": None, "positions_at_promote": None}
        self._approved_model: dict = {"version": 0, "promoted_at": None}
        self._trainer_metrics: dict = {}

        self._sp_files: dict[str, tuple[int, str]] = {}
        self._im_files: dict[str, tuple[int, str]] = {}
        self._sp_agg: dict[str, Any] = _empty_agg()
        self._im_agg: dict[str, Any] = _empty_agg()

        self._eval_file_cache: dict[str, tuple[str, dict]] = {}
        self._eval_versions: list[int] = []
        self._eval_summaries: dict[str, dict] = {}
        self._focus_version: int | None = None
        self._recent_games: dict[str, dict] = {}

        self._heartbeats: dict[str, dict] = {}
        self._benchmark_results: dict[str, dict] = {}
        self._benchmark_versions: list[dict] = []
        self._snapshots: list[dict] = []

        self._last_sync: str | None = None
        self._last_error: str | None = None
        self._initialised = False

    def start(self) -> None:
        self.refresh_once()
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._loop,
                name="dashboard-store",
                daemon=True,
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
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"

    def refresh_once(self) -> None:
        self._model = self._sync_etagged_json(
            storage.LATEST_META,
            "_etag_meta",
            lambda data: {
                "version": data.get("version", 0),
                "promoted_at": data.get("timestamp"),
                "positions_at_promote": data.get("positions_at_promote"),
            },
            fallback=self._model,
        )
        self._approved_model = self._sync_etagged_json(
            storage.APPROVED_META,
            "_etag_approved_meta",
            lambda data: {
                "version": data.get("version", 0),
                "promoted_at": data.get("timestamp"),
            },
            fallback=self._approved_model,
        )
        self._trainer_metrics = self._sync_etagged_json(
            storage.TRAINER_METRICS,
            "_etag_trainer",
            lambda data: {
                **{k: v for k, v in data.items() if k not in {"cycles", "summaries"}},
                "summaries": [
                    {
                        **{kk: vv for kk, vv in entry.items() if kk != "cycle"},
                        "summary": entry.get("summary", entry.get("cycle")),
                    }
                    for entry in data.get("summaries", data.get("cycles", []))
                ],
            },
            fallback=self._trainer_metrics,
        )
        self._sync_evaluations()
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

    def _sync_cached_json(self, key: str) -> dict | None:
        meta = self._s.head(key)
        cached = self._eval_file_cache.get(key)
        if meta is None:
            self._eval_file_cache.pop(key, None)
            return None
        if cached is not None and cached[0] == meta["etag"]:
            return cached[1]
        try:
            data = self._s.get_json(key)
        except KeyError:
            return None
        self._eval_file_cache[key] = (meta["etag"], data)
        return data

    def _sync_evaluations(self) -> None:
        versions = self._s.list_eval_versions()
        latest_versions = versions[-_EVAL_MAX_VERSIONS:]
        summaries: dict[str, dict] = {}
        for version in latest_versions:
            key = f"v{version}"
            gate = self._sync_cached_json(storage.eval_gate_summary_key(version))
            benchmark = self._sync_cached_json(storage.eval_benchmark_summary_key(version))
            decision = self._sync_cached_json(storage.eval_decision_key(version))
            summaries[key] = {
                "version": version,
                "gate": gate,
                "benchmark": benchmark,
                "decision": decision,
            }

        approved_version = int(self._approved_model.get("version") or 0)
        focus_version = None
        candidate_versions = [version for version in latest_versions if version > approved_version]
        if candidate_versions:
            focus_version = max(candidate_versions)
        elif approved_version in latest_versions:
            focus_version = approved_version

        self._sync_recent_games(focus_version)

        with self._lock:
            self._eval_versions = latest_versions
            self._eval_summaries = summaries
            self._focus_version = focus_version

    def _sync_recent_games(self, version: int | None) -> None:
        if version is None:
            with self._lock:
                self._recent_games = {}
            return
        keys = sorted(self._s.list_eval_game_record_keys(version), reverse=True)
        tail = set(keys[:_RECENT_GAMES_TAIL])
        if tail == self._recent_games.keys():
            return

        new_records = {}
        for key in tail:
            if key in self._recent_games:
                new_records[key] = self._recent_games[key]
                continue
            try:
                new_records[key] = self._s.get_json(key)
            except KeyError:
                continue

        with self._lock:
            self._recent_games = new_records

    def _sync_data(
        self,
        prefix: str,
        cache: dict[str, tuple[int, str]],
        agg_attr: str,
    ) -> None:
        listed = self._s.list_data_files(prefix)
        seen = {entry["key"] for entry in listed}

        to_add = [
            (entry["key"], entry["positions"], entry["version"])
            for entry in listed
            if entry["key"] not in cache
        ]
        stale = [key for key in cache if key not in seen]
        if not to_add and not stale:
            return

        with self._lock:
            for key, pos, version in to_add:
                cache[key] = (pos, version)
            for key in stale:
                cache.pop(key, None)
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
        snaps = [{"name": key.rsplit("/", 1)[-1]} for key in keys]
        with self._lock:
            self._snapshots = snaps

    def _sync_benchmarks(self) -> None:
        keys = self._s.ls(storage.BENCHMARK_RESULTS_PREFIX)
        available = []
        for key in keys:
            name = key.rsplit("/", 1)[-1]
            if not name.startswith("v") or not name.endswith(".json"):
                continue
            try:
                version = int(name[1:-5])
            except ValueError:
                continue
            available.append({"key": key, "name": name, "version": version})
        available.sort(key=lambda item: item["version"])
        to_use = available[-_BENCHMARK_MAX_VERSIONS:]

        new_results: dict[str, dict] = {}
        for entry in to_use:
            version_key = entry["name"][:-5]
            if version_key in self._benchmark_results:
                new_results[version_key] = self._benchmark_results[version_key]
                continue
            try:
                new_results[version_key] = self._s.get_json(entry["key"])
            except KeyError:
                continue

        with self._lock:
            self._benchmark_versions = [
                {"name": entry["name"], "version": entry["version"]}
                for entry in to_use
            ]
            self._benchmark_results = new_results

    def snapshot(self) -> dict:
        with self._lock:
            im_agg = {k: v for k, v in self._im_agg.items() if k != "by_version"}
            focus_key = None if self._focus_version is None else f"v{self._focus_version}"
            return {
                "model": dict(self._model),
                "approved_model": dict(self._approved_model),
                "evaluations": {
                    "focus_version": self._focus_version,
                    "versions": list(self._eval_versions),
                    "summaries": {key: dict(value) for key, value in self._eval_summaries.items()},
                    "focus": None if focus_key is None else dict(self._eval_summaries.get(focus_key, {})),
                },
                "recent_games": [
                    self._recent_games[key]
                    for key in sorted(self._recent_games)
                ],
                "workers": {name: entry["data"] for name, entry in self._heartbeats.items()},
                "data": {
                    "selfplay": dict(self._sp_agg),
                    "imitation": im_agg,
                    "models": list(self._snapshots),
                },
                "trainer_metrics": dict(self._trainer_metrics),
                "benchmark_versions": list(self._benchmark_versions),
                "benchmark_results": {key: dict(value) for key, value in self._benchmark_results.items()},
                "timestamp": self._last_sync or datetime.now(timezone.utc).isoformat(),
                "sync_error": self._last_error,
                "initialised": self._initialised,
            }


def _empty_agg() -> dict[str, Any]:
    return {"total_files": 0, "total_positions": 0, "by_version": {}}


def _aggregate(cache: dict[str, tuple[int, str]]) -> dict[str, Any]:
    by_version: dict[str, dict[str, int]] = {}
    total_positions = 0
    for positions, version in cache.values():
        total_positions += positions
        bucket = by_version.setdefault(version, {"count": 0, "positions": 0})
        bucket["count"] += 1
        bucket["positions"] += positions
    return {
        "total_files": len(cache),
        "total_positions": total_positions,
        "by_version": by_version,
    }
