"""Lightweight training dashboard — serves status, Elo, and pod logs."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from .config import AsyncConfig

_DASHBOARD_HTML: str | None = None
_status_cache: dict = {}
_status_cache_time: float = 0
_STATUS_CACHE_TTL = 10  # seconds

# Lazy-loaded k8s client
_k8s_api: object | None = None
_NAMESPACE = "hexchess"


def _get_k8s_api():
    """Get or create the k8s CoreV1Api client."""
    global _k8s_api
    if _k8s_api is None:
        from kubernetes import client, config
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        _k8s_api = client.CoreV1Api()
    return _k8s_api


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl_tail(path: Path, n: int = 20) -> list[dict]:
    """Read last N lines of a JSONL file."""
    if not path.exists():
        return []
    try:
        with open(path) as f:
            lines = deque(f, maxlen=n)
        results = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return results
    except OSError:
        return []


def _collect_training_data_stats(data_dir: Path) -> dict:
    """Aggregate training data file stats without opening .npz files."""
    by_version: dict[str, dict] = {}
    total_count = 0
    total_bytes = 0

    if not data_dir.exists():
        return {"total_count": 0, "total_size_mb": 0, "by_version": {}}

    try:
        for entry in os.scandir(data_dir):
            if not entry.name.endswith(".npz") or ".tmp" in entry.name:
                continue
            try:
                size = entry.stat().st_size
            except OSError:
                continue

            total_count += 1
            total_bytes += size

            # Parse version from filename: sp_v5_... or im_v0_...
            parts = entry.name.split("_")
            if len(parts) >= 2:
                version = parts[1]  # "v0", "v5", etc.
                prefix = parts[0]   # "sp" or "im"
                key = f"{prefix}_{version}"
            else:
                key = "unknown"

            if key not in by_version:
                by_version[key] = {"count": 0, "size_mb": 0}
            by_version[key]["count"] += 1
            by_version[key]["size_mb"] += size / (1024 * 1024)
    except OSError:
        pass

    # Round sizes
    for v in by_version.values():
        v["size_mb"] = round(v["size_mb"], 1)

    return {
        "total_count": total_count,
        "total_size_mb": round(total_bytes / (1024 * 1024), 1),
        "by_version": dict(sorted(by_version.items())),
    }


def _collect_model_snapshots(models_dir: Path) -> list[dict]:
    """List versioned model snapshots."""
    snapshots = []
    if not models_dir.exists():
        return snapshots
    for f in sorted(models_dir.glob("v*.onnx")):
        try:
            snapshots.append({
                "name": f.name,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
            })
        except OSError:
            continue
    return snapshots


def _get_pod_status() -> dict:
    """Get pod status via k8s API."""
    try:
        api = _get_k8s_api()
        pod_list = api.list_namespaced_pod(namespace=_NAMESPACE)
        pods = {}
        for pod in pod_list.items:
            name = pod.metadata.name
            # Map pod name to role
            if "trainer" in name:
                role = "trainer"
            elif "worker" in name:
                existing = [k for k in pods if k.startswith("worker")]
                role = f"worker-{len(existing)}"
            elif "elo-service" in name:
                role = "elo-service"
            elif "dashboard" in name:
                continue  # skip ourselves
            else:
                role = name

            # Determine status
            if pod.status.container_statuses:
                cs = pod.status.container_statuses[0]
                if cs.ready:
                    status = "Running"
                elif cs.state.waiting:
                    status = cs.state.waiting.reason or "Waiting"
                elif cs.state.terminated:
                    status = cs.state.terminated.reason or "Terminated"
                else:
                    status = pod.status.phase or "Unknown"
            else:
                status = pod.status.phase or "Unknown"

            pods[role] = status
        return pods
    except Exception:
        return {}


def collect_status(cfg: AsyncConfig) -> dict:
    """Collect all dashboard data."""
    global _status_cache, _status_cache_time
    now = time.time()
    if now - _status_cache_time < _STATUS_CACHE_TTL:
        return _status_cache

    # Model info
    meta = _read_json(cfg.best_meta_path)
    model = {
        "version": meta.get("version", 0) if meta else 0,
        "promoted_at": meta.get("timestamp") if meta else None,
    }

    # Training history from JSONL
    trainer_events = _read_jsonl_tail(cfg.logs_dir / "trainer.jsonl", 20)
    cycles = [e for e in trainer_events if e.get("event") == "cycle_complete"]
    bootstrap = [e for e in trainer_events if e.get("event") == "bootstrap_complete"]

    latest_cycle = cycles[-1] if cycles else None
    loss_history = [round(e.get("policy_loss", 0) + e.get("value_loss", 0), 4) for e in cycles]

    # Elo
    elo_state = _read_json(cfg.models_dir.parent / "elo_state.json")
    elo_history = _read_jsonl_tail(cfg.models_dir.parent / "elo_rankings.jsonl", 10)

    # Data stats
    data_stats = _collect_training_data_stats(cfg.training_data_dir)
    model_snapshots = _collect_model_snapshots(cfg.models_dir)

    # Pod status
    pods = _get_pod_status()

    status = {
        "model": model,
        "pods": pods,
        "training": {
            "latest_cycle": latest_cycle,
            "bootstrap": bootstrap[-1] if bootstrap else None,
            "cycles": cycles[-10:],  # last 10 for table
            "loss_history": loss_history,
        },
        "elo": {
            "ratings": elo_state.get("ratings", {}) if elo_state else {},
            "total_games": elo_state.get("total_games", 0) if elo_state else 0,
            "active_players": elo_state.get("active_players", []) if elo_state else [],
            "player_stats": elo_state.get("player_stats", {}) if elo_state else {},
            "history": elo_history,
        },
        "data": {
            "training_files": data_stats,
            "models": model_snapshots,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _status_cache = status
    _status_cache_time = now
    return status


def get_pod_logs(role: str, tail: int = 50) -> list[str]:
    """Fetch logs for a pod by role."""
    try:
        api = _get_k8s_api()
        pod_list = api.list_namespaced_pod(namespace=_NAMESPACE)

        # Find pods matching the role
        matching = []
        for pod in pod_list.items:
            name = pod.metadata.name
            if role == "trainer" and "trainer" in name:
                matching.append(name)
            elif role.startswith("worker") and "worker" in name and "dashboard" not in name:
                matching.append(name)
            elif role == "elo-service" and "elo-service" in name:
                matching.append(name)

        matching.sort()

        if not matching:
            return [f"No pods found for role: {role}"]

        # For worker-N, pick the Nth worker pod
        if role.startswith("worker-"):
            idx = int(role.split("-")[1])
            if idx >= len(matching):
                return [f"{role} not found (only {len(matching)} workers)"]
            pod_name = matching[idx]
        else:
            pod_name = matching[0]

        logs = api.read_namespaced_pod_log(
            name=pod_name, namespace=_NAMESPACE, tail_lines=tail,
        )
        return logs.strip().split("\n") if logs else ["(no logs)"]
    except Exception as e:
        return [f"Error: {e}"]


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
        elif self.path.startswith("/api/logs/"):
            role = self.path.split("/api/logs/")[-1]
            self._serve_json({"lines": get_pod_logs(role)})
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
        pass  # suppress default access logs


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
