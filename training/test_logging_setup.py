"""Tests for training.logging_setup (plan §7.1)."""

from __future__ import annotations

import io
import json
import logging

import pytest

from training import logging_setup, storage
from training.logging_setup import (
    JsonLineFormatter,
    S3RotatingFileSink,
    log_event,
    setup_json_logging,
)


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset the global logging state between tests."""
    root = logging.getLogger()
    handlers = list(root.handlers)
    for h in handlers:
        root.removeHandler(h)
    logging_setup._CONFIGURED = False
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
    logging_setup._CONFIGURED = False


def _capture_to_buffer() -> io.StringIO:
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonLineFormatter())
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    return buf


def test_setup_json_logging_is_idempotent():
    setup_json_logging("trainer", run_id="r1")
    n1 = len(logging.getLogger().handlers)
    setup_json_logging("trainer", run_id="r1")
    assert len(logging.getLogger().handlers) == n1


def test_log_event_emits_json_with_envelope_fields():
    setup_json_logging("trainer", run_id="testrun", version="v7")
    buf = _capture_to_buffer()

    log_event("train.step", step_id=42, loss_total=1.25)

    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert lines, "no log output captured"
    payload = json.loads(lines[-1])

    for field in ("ts", "run_id", "service", "host", "version", "event"):
        assert field in payload, f"missing envelope field: {field}"

    assert payload["run_id"] == "testrun"
    assert payload["service"] == "trainer"
    assert payload["version"] == "v7"
    assert payload["event"] == "train.step"
    assert payload["step_id"] == 42
    assert payload["loss_total"] == 1.25


def test_json_formatter_handles_plain_log_messages():
    setup_json_logging("trainer", run_id="testrun")
    buf = _capture_to_buffer()

    logging.getLogger().info("hello world")

    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    payload = json.loads(lines[-1])
    assert payload["event"] == "log"
    assert payload["message"] == "hello world"


def test_json_formatter_coerces_numpy_values():
    np = pytest.importorskip("numpy")
    setup_json_logging("trainer", run_id="testrun")
    buf = _capture_to_buffer()

    log_event("metric", mean=np.float32(0.5), counts=np.array([1, 2, 3]))

    payload = json.loads(buf.getvalue().splitlines()[-1])
    assert payload["mean"] == pytest.approx(0.5)
    assert payload["counts"] == [1, 2, 3]


def test_setup_json_logging_attaches_s3_sink_when_path_provided(tmp_path):
    sink_path = tmp_path / "logs" / "trainer" / "testrun.events.jsonl"

    setup_json_logging("trainer", run_id="testrun", s3_sink_path=sink_path)

    s3_handlers = [
        h for h in logging.getLogger().handlers if isinstance(h, S3RotatingFileSink)
    ]
    assert len(s3_handlers) == 1
    assert s3_handlers[0].local_path == sink_path


def test_s3_sink_uploads_pending_file_on_close(tmp_path, monkeypatch):
    sink_path = tmp_path / "logs" / "trainer" / "testrun.events.jsonl"
    sink = S3RotatingFileSink(sink_path, service="trainer", host="test-host")
    sink_path.write_text('{"event":"train.step"}\n', encoding="utf-8")

    uploads: list[tuple[str, str]] = []

    def fake_put_file(key: str, local_path: str) -> None:
        uploads.append((key, str(local_path)))

    monkeypatch.setattr(storage, "put_file", fake_put_file)

    sink.close()

    assert uploads == [
        (
            f"logs/trainer/test-host/{sink._current_date}/events.jsonl.gz",
            str(sink_path.with_suffix(".jsonl." + sink._current_date + ".gz")),
        )
    ]
    assert sink_path.read_text(encoding="utf-8") == ""
