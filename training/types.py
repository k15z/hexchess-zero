"""Shared structural types for the training pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

BoardTensor: TypeAlias = NDArray[np.float32]
PolicyVector: TypeAlias = NDArray[np.float32]
WdlVector: TypeAlias = NDArray[np.float32]
FloatBatch: TypeAlias = NDArray[np.float32]
BoolBatch: TypeAlias = NDArray[np.bool_]
BoolVector: TypeAlias = NDArray[np.bool_]


class LatestModelMeta(TypedDict):
    version: int
    timestamp: str


class HeartbeatRecord(TypedDict):
    timestamp: str
    model_version: int
    total_games: int
    total_positions: int


class HeadObjectRecord(TypedDict):
    etag: str
    size: int
    last_modified: datetime | None


class ListedObjectRecord(TypedDict):
    key: str
    size: int
    last_modified: datetime | None
    etag: str


class DataFileRecord(TypedDict):
    key: str
    positions: int
    timestamp: str
    version: str


class AggregateBucket(TypedDict):
    count: int
    positions: int


class DataAggregate(TypedDict):
    total_files: int
    total_positions: int
    by_version: dict[str, AggregateBucket]


class DashboardModelState(TypedDict):
    version: int
    promoted_at: str | None


class CachedHeartbeat(TypedDict):
    lm: str
    data: HeartbeatRecord


class ModelSnapshotEntry(TypedDict):
    name: str


class V2Sample(TypedDict):
    boards: BoardTensor
    policy: PolicyVector
    aux_policy: PolicyVector
    wdl_terminal: WdlVector
    wdl_short: WdlVector
    mlh: np.float32
    legal_mask: BoolVector


class V2BatchDict(TypedDict):
    boards: FloatBatch
    policy: FloatBatch
    aux_policy: FloatBatch
    wdl_terminal: FloatBatch
    wdl_short: FloatBatch
    mlh: FloatBatch
    legal_mask: BoolBatch


class ImitationSample(TypedDict):
    board: BoardTensor
    policy: PolicyVector
    outcome: WdlVector


class StorageLike(Protocol):
    def delete(self, key: str) -> None: ...

    def get_json(self, key: str) -> JsonObject: ...

    def head(self, key: str) -> HeadObjectRecord | None: ...

    def list_data_files(self, prefix: str) -> list[DataFileRecord]: ...

    def list_with_meta(self, prefix: str) -> list[ListedObjectRecord]: ...

    def ls(self, prefix: str) -> list[str]: ...


def parse_latest_model_meta(raw: Mapping[str, object]) -> LatestModelMeta:
    """Coerce a loose JSON mapping into the latest-model metadata shape."""
    timestamp = raw.get("timestamp")
    return {
        "version": coerce_int(raw.get("version")),
        "timestamp": "" if timestamp is None else str(timestamp),
    }


def parse_heartbeat_record(raw: Mapping[str, object]) -> HeartbeatRecord:
    """Coerce a loose JSON mapping into the worker-heartbeat shape."""
    timestamp = raw.get("timestamp")
    return {
        "timestamp": "" if timestamp is None else str(timestamp),
        "model_version": coerce_int(raw.get("model_version")),
        "total_games": coerce_int(raw.get("total_games")),
        "total_positions": coerce_int(raw.get("total_positions")),
    }


def coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0
