"""Stable contracts shared by every identifiability-paper stage."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ObservationRecord:
    observation_id: str
    burst_id: int
    original_roi_id: str
    observation_site_id: str
    canonical_neuron_id: str | None
    x_px: float
    y_px: float
    geometry_kind: str
    geometry_hash: str
    original_start_frame_ui: int
    original_end_frame_ui: int
    event_onset_ui: int | None
    event_peak_ui: int | None
    event_end_ui: int | None
    include_confirmed: bool
    include_inclusive: bool
    review_status: str
    disposition: str

    def __post_init__(self) -> None:
        if not self.observation_id or not self.observation_site_id:
            raise ValueError("observation and site identifiers are required")
        if self.burst_id < 1:
            raise ValueError("burst_id must be positive")
        if self.original_start_frame_ui < 1 or self.original_end_frame_ui < self.original_start_frame_ui:
            raise ValueError("invalid one-based inclusive frame interval")
        timing = (self.event_onset_ui, self.event_peak_ui, self.event_end_ui)
        if any(value is not None for value in timing):
            if any(value is None for value in timing) or not (timing[0] <= timing[1] <= timing[2]):
                raise ValueError("adjudicated timing must be complete and ordered")

    @property
    def start_zero(self) -> int:
        return self.original_start_frame_ui - 1

    @property
    def stop_zero_exclusive(self) -> int:
        return self.original_end_frame_ui

    def trace_key(self, video_hash: str, channel: str, parameters_hash: str) -> tuple[str, str, str, str, str]:
        return (self.observation_site_id, self.geometry_hash, video_hash, channel, parameters_hash)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ObservationRecord":
        return cls(**value)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def disk_geometry_hash(x_px: float, y_px: float, radius_px: int) -> str:
    return stable_hash({"kind": "disk", "x_px": round(float(x_px), 6), "y_px": round(float(y_px), 6), "radius_px": int(radius_px)})


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate_unique(records: Iterable[ObservationRecord]) -> list[ObservationRecord]:
    values = list(records)
    ids = [record.observation_id for record in values]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate observation_id")
    return values
