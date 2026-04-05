"""Lightweight training dashboard — serves status from S3."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from . import storage
from .config import AsyncConfig

_DASHBOARD_HTML: str | None = None
_status_cache: dict = {}
_status_cache_time: float = 0
_STATUS_CACHE_TTL = 10  # seconds


def collect_status(cfg: AsyncConfig) -> dict:
    """Collect all dashboard data from S3."""
    global _status_cache, _status_cache_time
    now = time.time()
    if now - _status_cache_time < _STATUS_CACHE_TTL:
        return _status_cache

    # Model info
    try:
        meta = storage.get_json(storage.LATEST_META)
    except KeyError:
        meta = {}
    model = {
        "version": meta.get("version", 0),
        "promoted_at": meta.get("timestamp"),
    }

    # Elo state (OpenSkill ratings + head-to-head)
    try:
        elo_state = storage.get_json(storage.ELO_STATE)
    except KeyError:
        elo_state = {}

    # Training data stats (from S3 key listing, no file opens)
    sp_files = storage.list_data_files(storage.SELFPLAY_PREFIX)
    im_files = storage.list_data_files(storage.IMITATION_PREFIX)

    sp_by_version: dict[str, dict] = {}
    for f in sp_files:
        parts = f["key"].split("/")
        v = parts[2] if len(parts) >= 3 else "unknown"
        if v not in sp_by_version:
            sp_by_version[v] = {"count": 0, "positions": 0}
        sp_by_version[v]["count"] += 1
        sp_by_version[v]["positions"] += f["positions"]

    # Model snapshots
    versions = storage.ls(storage.VERSIONS_PREFIX)
    snapshots = []
    for key in versions:
        name = key.split("/")[-1]
        snapshots.append({"name": name})

    # Heartbeats
    heartbeats = {}
    for key in storage.ls(storage.HEARTBEATS_PREFIX):
        try:
            hb = storage.get_json(key)
            name = key.split("/")[-1].replace(".json", "")
            heartbeats[name] = hb
        except KeyError:
            continue

    # Recent game log (last 50 games)
    try:
        all_games = storage.get_jsonl(storage.ELO_GAMES_LOG)
        recent_games = all_games[-50:]
    except Exception:
        recent_games = []

    status = {
        "model": model,
        "workers": heartbeats,
        "elo": {
            "ratings": elo_state.get("ratings", {}),
            "total_games": elo_state.get("total_games", 0),
            "active_players": elo_state.get("active_players", []),
            "retired_players": elo_state.get("retired_players", []),
            "player_stats": elo_state.get("player_stats", {}),
            "pair_results": elo_state.get("pair_results", {}),
        },
        "recent_games": recent_games,
        "data": {
            "selfplay": {
                "total_files": len(sp_files),
                "total_positions": sum(f["positions"] for f in sp_files),
                "by_version": sp_by_version,
            },
            "imitation": {
                "total_files": len(im_files),
                "total_positions": sum(f["positions"] for f in im_files),
            },
            "models": snapshots,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _status_cache = status
    _status_cache_time = now
    return status


def _load_html() -> str:
    global _DASHBOARD_HTML
    if _DASHBOARD_HTML is None:
        html_path = Path(__file__).parent / "dashboard.html"
        _DASHBOARD_HTML = html_path.read_text()
    return _DASHBOARD_HTML


class DashboardHandler(BaseHTTPRequestHandler):
    cfg: AsyncConfig

    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path == "/api/status":
            self._serve_json(collect_status(self.cfg))
        else:
            self.send_error(404)

    def _serve_html(self):
        body = _load_html().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def run_dashboard(cfg: AsyncConfig, port: int = 8080) -> None:
    """Start the dashboard HTTP server."""
    DashboardHandler.cfg = cfg
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"Dashboard running on http://0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
