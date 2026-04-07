"""S3 storage layer for the training pipeline.

All shared state goes through this module. Uses S3-compatible object storage
(DigitalOcean Spaces, Cloudflare R2, AWS S3, MinIO, etc.).

Bucket layout:
    models/latest.onnx              Current model for inference
    models/latest.meta.json         {"version": N, "timestamp": "..."}
    models/checkpoint.pt            PyTorch training checkpoint
    models/versions/{N}.onnx        Immutable version snapshots

    data/selfplay/v{N}/{ts}_{rand}_n{count}.npz
    data/imitation/{ts}_{rand}_n{count}.npz

    state/elo.json                  Elo ratings + pair results

    heartbeats/{hostname}.json      Worker liveness + stats
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# S3 key constants
# ---------------------------------------------------------------------------

LATEST_ONNX = "models/latest.onnx"
LATEST_META = "models/latest.meta.json"
CHECKPOINT_PT = "models/checkpoint.pt"
VERSIONS_PREFIX = "models/versions/"
SELFPLAY_PREFIX = "data/selfplay/"
IMITATION_PREFIX = "data/imitation/"
ELO_STATE = "state/elo.json"
ELO_GAMES_PREFIX = "state/elo_games/"  # one object per game, race-free writes
HEARTBEATS_PREFIX = "heartbeats/"

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_cached_client = None


def _client():
    global _cached_client
    if _cached_client is None:
        _cached_client = boto3.client(
            "s3",
            endpoint_url=os.environ["ENDPOINT"],
            aws_access_key_id=os.environ["ACCESS_KEY"],
            aws_secret_access_key=os.environ["SECRET_KEY"],
        )
    return _cached_client


def _bucket() -> str:
    return os.environ["BUCKET_NAME"]


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def put(key: str, data: bytes | str) -> None:
    """Upload bytes or string to S3."""
    if isinstance(data, str):
        data = data.encode()
    _client().put_object(Bucket=_bucket(), Key=key, Body=data)


def put_file(key: str, local_path: str | Path) -> None:
    """Upload a local file to S3."""
    _client().upload_file(str(local_path), _bucket(), key)


def get(key: str) -> bytes:
    """Download an object as bytes. Raises KeyError if not found."""
    try:
        resp = _client().get_object(Bucket=_bucket(), Key=key)
        return resp["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            raise KeyError(key) from e
        raise


def get_json(key: str) -> dict:
    """Download and parse a JSON object."""
    return json.loads(get(key))


def get_file(key: str, local_path: str | Path) -> Path:
    """Download an object to a local file. Returns the local path."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(_bucket(), key, str(local_path))
    return local_path


def put_json(key: str, obj: dict) -> None:
    """Upload a dict as JSON."""
    put(key, json.dumps(obj, indent=2))


def copy(src_key: str, dst_key: str) -> None:
    """Server-side copy within the same bucket."""
    _client().copy_object(
        Bucket=_bucket(), Key=dst_key,
        CopySource={"Bucket": _bucket(), "Key": src_key},
    )


def head(key: str) -> dict | None:
    """Return {'etag', 'size', 'last_modified'} for a key, or None if missing.

    Cheap change-detection primitive — a single HEAD request.
    """
    try:
        resp = _client().head_object(Bucket=_bucket(), Key=key)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404", "NotFound"):
            return None
        raise
    return {
        "etag": resp.get("ETag", "").strip('"'),
        "size": int(resp.get("ContentLength", 0)),
        "last_modified": resp.get("LastModified"),
    }


def get_range(key: str, start: int, end: int | None = None) -> bytes:
    """Ranged GET starting at byte `start` (inclusive). Used for tailing logs."""
    rng = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
    try:
        resp = _client().get_object(Bucket=_bucket(), Key=key, Range=rng)
        return resp["Body"].read()
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("NoSuchKey", "404"):
            raise KeyError(key) from e
        if code == "InvalidRange":
            return b""
        raise


def list_with_meta(prefix: str) -> list[dict]:
    """List keys under a prefix with size + last_modified metadata."""
    client = _client()
    bucket = _bucket()
    out = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            out.append({
                "key": obj["Key"],
                "size": int(obj.get("Size", 0)),
                "last_modified": obj.get("LastModified"),
                "etag": obj.get("ETag", "").strip('"'),
            })
        if not resp.get("IsTruncated"):
            break
        token = resp["NextContinuationToken"]
    return out


def ls(prefix: str) -> list[str]:
    """List all keys under a prefix."""
    client = _client()
    bucket = _bucket()
    keys = []
    continuation_token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            keys.append(obj["Key"])
        if not resp.get("IsTruncated"):
            break
        continuation_token = resp["NextContinuationToken"]
    return keys


# ---------------------------------------------------------------------------
# Training data helpers
# ---------------------------------------------------------------------------

_NPZ_PATTERN = re.compile(r"_n(\d+)\.npz$")


def list_data_files(prefix: str = SELFPLAY_PREFIX) -> list[dict]:
    """List training data files with parsed metadata.

    Returns list of {"key", "positions", "timestamp", "version"} sorted by
    timestamp descending (most recent first). ``version`` is the ``v{N}``
    path segment for selfplay data, or "unknown" for other prefixes.
    """
    files = []
    for key in ls(prefix):
        m = _NPZ_PATTERN.search(key)
        if not m:
            continue
        basename = key.rsplit("/", 1)[-1]
        ts = basename.split("_")[0]
        parts = key.split("/")
        version = parts[2] if len(parts) >= 3 else "unknown"
        files.append({
            "key": key,
            "positions": int(m.group(1)),
            "timestamp": ts,
            "version": version,
        })
    files.sort(key=lambda f: f["timestamp"], reverse=True)
    return files


def count_positions(prefix: str = SELFPLAY_PREFIX) -> int:
    """Count total positions by parsing _n{count} from filenames."""
    return sum(f["positions"] for f in list_data_files(prefix))


def select_recent_files(prefix: str, max_positions: int) -> list[dict]:
    """Select most recent files up to max_positions.

    Returns in chronological order (oldest first).
    """
    files = list_data_files(prefix)
    selected = []
    total = 0
    for f in files:
        selected.append(f)
        total += f["positions"]
        if total >= max_positions:
            break
    selected.reverse()
    return selected


# ---------------------------------------------------------------------------
# Flush helpers
# ---------------------------------------------------------------------------

def flush_samples(samples: list[dict], key_prefix: str) -> str:
    """Stack sample dicts, compress, and upload as .npz. Returns the S3 key.

    Each sample must have 'board', 'policy', and 'outcome' numpy arrays.
    """
    boards = np.stack([s["board"] for s in samples])
    policies = np.stack([s["policy"] for s in samples])
    outcomes = np.array([s["outcome"] for s in samples], dtype=np.float32)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    rand = np.random.default_rng().integers(0, 0xFFFF_FFFF)
    n = len(outcomes)
    key = f"{key_prefix}{ts}_{rand:08x}_n{n}.npz"

    upload_npz(key, boards=boards, policies=policies, outcomes=outcomes)
    return key


def upload_npz(key: str, *, boards: np.ndarray, policies: np.ndarray,
               outcomes: np.ndarray) -> None:
    """Compress and upload training data arrays as a .npz file."""
    # suffix=".npz" ensures numpy doesn't double-append
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        np.savez_compressed(tmp_path, boards=boards, policies=policies,
                            outcomes=outcomes)
        put_file(key, tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def put_game_record(record: dict) -> str:
    """Write one Elo game result as its own S3 object. Returns the key.

    Per-game objects replace the old append-to-jsonl pattern so multiple
    elo-service replicas can write concurrently without clobbering each other.
    Key format: ``state/elo_games/{ts}_{rand:08x}.json``.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    rand = np.random.default_rng().integers(0, 0xFFFF_FFFF)
    key = f"{ELO_GAMES_PREFIX}{ts}_{int(rand):08x}.json"
    put_json(key, record)
    return key


def list_game_record_keys() -> list[str]:
    """List all per-game Elo record keys, sorted (timestamp-prefixed)."""
    return sorted(ls(ELO_GAMES_PREFIX))


def key_basename(key: str) -> str:
    """Extract the filename without extension from an S3 key."""
    return key.rsplit("/", 1)[-1].split(".")[0]
