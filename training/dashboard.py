"""Lightweight training dashboard — serves status from an in-memory store."""

from __future__ import annotations

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from .config import AsyncConfig
from .dashboard_store import DashboardStore

_DASHBOARD_HTML: str | None = None


def _load_html() -> str:
    global _DASHBOARD_HTML
    if _DASHBOARD_HTML is None:
        html_path = Path(__file__).parent / "dashboard.html"
        _DASHBOARD_HTML = html_path.read_text()
    return _DASHBOARD_HTML


class DashboardHandler(BaseHTTPRequestHandler):
    cfg: AsyncConfig
    store: DashboardStore

    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path == "/api/status":
            self._serve_json(self.store.snapshot())
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
    store = DashboardStore(interval=60.0)
    print("Dashboard: priming store from S3 ...")
    store.start()
    DashboardHandler.cfg = cfg
    DashboardHandler.store = store
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"Dashboard running on http://0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
    store.stop()
