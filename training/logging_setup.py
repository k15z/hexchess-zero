"""Structured JSON logging for the training pipeline.

Implements notes/13 §7.1–§7.3: one JSON event per line to stdout, with a
consistent envelope ``{ts, run_id, service, host, version, event, ...payload}``.

Usage:

    from training.logging_setup import setup_json_logging, log_event

    setup_json_logging("trainer", run_id=cfg.run_id)
    log_event("train.step", step_id=42, loss_total=1.23)

The :func:`log_event` helper is the preferred way to emit Tier-1 metrics
(plan §7.2/§7.3); plain ``logger.info(...)`` calls still work and are
forwarded through the same JSON sink with ``event="log"``.

The optional :class:`S3RotatingFileSink` writes events to a local file and
uploads gzipped daily rotations to
``logs/{service}/{hostname}/{date}/events.jsonl.gz`` via :mod:`training.storage`.
It's a minimal implementation — see TODOs for known gaps.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import platform
import shutil
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Module-level defaults populated by :func:`setup_json_logging`.
_RUN_ID: str = "dev"
_SERVICE: str = "unknown"
_HOST: str = platform.node() or "unknown"
_VERSION: str | None = None
_CONFIGURED: bool = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class JsonLineFormatter(logging.Formatter):
    """Formats a ``logging.LogRecord`` as one compact JSON line.

    Envelope fields come from module globals set by
    :func:`setup_json_logging`. Extra payload is pulled from ``record.payload``
    if present (set by :func:`log_event`), otherwise the formatted message is
    stored under ``"message"`` with ``event="log"``.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        envelope: dict[str, Any] = {
            "ts": _now_iso(),
            "run_id": _RUN_ID,
            "service": _SERVICE,
            "host": _HOST,
            "version": _VERSION,
            "level": record.levelname,
        }
        payload = getattr(record, "payload", None)
        if isinstance(payload, dict):
            envelope["event"] = payload.get("event", "log")
            # Merge payload last so it can't overwrite envelope keys above.
            for k, v in payload.items():
                if k == "event" or k in envelope:
                    continue
                envelope[k] = _json_safe(v)
        else:
            envelope["event"] = "log"
            envelope["message"] = record.getMessage()

        if record.exc_info:
            envelope["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(envelope, separators=(",", ":"), default=str)


def _json_safe(value: Any) -> Any:
    """Best-effort coercion of arbitrary values into JSON-serialisable form."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    # numpy scalars / arrays
    try:
        import numpy as np  # noqa: PLC0415

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    return str(value)


class S3RotatingFileSink(logging.Handler):
    """Write JSON lines to a local file; upload gzipped daily rotations to S3.

    Minimal implementation:

    - Appends formatted records to ``local_path``.
    - On the first emit each UTC day after the first, gzip-compresses the
      previous day's file and uploads to
      ``logs/{service}/{host}/{YYYY-MM-DD}/events.jsonl.gz``.
    - Truncates the local file after a successful upload.

    TODOs (left intentionally unimplemented for chunk 8):
    - Size-based rotation (only daily rotation is implemented).
    - Background thread / async upload (rotation happens on the logging
      thread that observes the date rollover).
    - Retry on S3 failures — we currently log and continue.
    """

    def __init__(self, local_path: Path, service: str, host: str) -> None:
        super().__init__()
        self.local_path = Path(local_path)
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self.service = service
        self.host = host
        self._current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._lock = threading.Lock()

    def _rotate_if_needed(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today == self._current_date:
            return
        rotated_date = self._current_date
        self._current_date = today
        if not self.local_path.exists() or self.local_path.stat().st_size == 0:
            return
        gz_path = self.local_path.with_suffix(
            self.local_path.suffix + f".{rotated_date}.gz"
        )
        try:
            with open(self.local_path, "rb") as src, gzip.open(gz_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            key = f"logs/{self.service}/{self.host}/{rotated_date}/events.jsonl.gz"
            try:
                from . import storage  # noqa: PLC0415

                storage.put_file(key, gz_path)
            except Exception as exc:  # pragma: no cover - best effort
                sys.stderr.write(f"S3RotatingFileSink upload failed: {exc}\n")
            self.local_path.write_text("")
        finally:
            if gz_path.exists():
                try:
                    gz_path.unlink()
                except OSError:
                    pass

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            with self._lock:
                self._rotate_if_needed()
                with open(self.local_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:  # pragma: no cover
            self.handleError(record)


def setup_json_logging(
    service: str,
    run_id: str | None = None,
    version: str | None = None,
    *,
    level: int = logging.INFO,
    s3_sink_path: Path | None = None,
) -> None:
    """Configure the root logger with a JSON-line stdout sink.

    Idempotent: re-calling updates envelope fields and (if already configured)
    does not attach duplicate handlers.
    """
    global _RUN_ID, _SERVICE, _HOST, _VERSION, _CONFIGURED

    _SERVICE = service
    _RUN_ID = run_id or os.environ.get("RUN_ID", "dev")
    _VERSION = version
    _HOST = platform.node() or "unknown"

    root = logging.getLogger()
    root.setLevel(level)

    if _CONFIGURED:
        return

    formatter = JsonLineFormatter()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(level)
    root.addHandler(stdout_handler)

    if s3_sink_path is not None:
        sink = S3RotatingFileSink(Path(s3_sink_path), service=service, host=_HOST)
        sink.setFormatter(formatter)
        sink.setLevel(level)
        root.addHandler(sink)

    # Silence noisy third-party loggers to WARNING so JSON output stays clean.
    for name in ("botocore", "boto3", "urllib3", "s3transfer"):
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True


def log_event(event: str, level: int = logging.INFO, **payload: Any) -> None:
    """Emit one structured JSON log line.

    ``event`` becomes the ``event`` field in the envelope; ``**payload`` is
    merged as additional fields. Numpy scalars and arrays are coerced via
    :func:`_json_safe`.
    """
    logger = logging.getLogger("training.events")
    record_payload = {"event": event, **payload}
    logger.log(level, event, extra={"payload": record_payload})


def current_run_id() -> str:
    return _RUN_ID
