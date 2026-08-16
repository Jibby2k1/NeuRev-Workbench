"""Validated data contracts for bounded exhaustive truth sets."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from neurobench.validation.schemas import validate_dict


OBJECT_DISPOSITIONS = {"neuron", "artifact", "background", "unresolved"}
EVENT_DISPOSITIONS = {"event", "artifact", "background", "unresolved"}


def ui_to_array_interval(interval: Sequence[int]) -> tuple[int, int]:
    """Convert one-based inclusive UI frames to zero-based half-open indices."""
    if len(interval) != 2 or any(isinstance(value, bool) or not isinstance(value, int) for value in interval):
        raise ValueError("UI interval must contain two integer frames")
    start, end = interval
    if start < 1 or end < start:
        raise ValueError("UI interval must be one-based inclusive with end >= start")
    return start - 1, end


def array_to_ui_interval(interval: Sequence[int]) -> tuple[int, int]:
    if len(interval) != 2 or any(isinstance(value, bool) or not isinstance(value, int) for value in interval):
        raise ValueError("array interval must contain two integer indices")
    start, end = interval
    if start < 0 or end <= start:
        raise ValueError("array interval must be zero-based half-open with end > start")
    return start + 1, end


def _copy(payload: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(payload))


@dataclass(frozen=True)
class TruthSetRegion:
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TruthSetRegion":
        result = cls(_copy(payload))
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        return _copy(self.payload)

    def validate(self) -> None:
        validate_dict(self.payload, "truth_set_region")
        bounds = self.payload["spatial_bounds_px"]
        if bounds["x_max_exclusive"] <= bounds["x_min"] or bounds["y_max_exclusive"] <= bounds["y_min"]:
            raise ValueError("region spatial bounds must have positive area")
        if tuple(self.payload["array_frame_interval"]) != ui_to_array_interval(self.payload["ui_frame_interval"]):
            raise ValueError("UI and array frame intervals do not describe the same frames")
        lock = self.payload["protected_lock"]
        if self.payload["role"] != "protected" and (lock["locked"] or lock["unsealed"]):
            raise ValueError("only protected regions may carry protected lock state")
        if lock["unsealed"] and (not lock["locked"] or not lock["locked_at"] or not lock["unsealed_at"]):
            raise ValueError("unsealed protected regions require a prior lock and both timestamps")
        if self.payload["raw_first_complete"] and self.payload["review_status"] in {"planned", "raw_first_in_progress"}:
            raise ValueError("raw-first completion requires a locked or later review status")
        union_count = int(self.payload.get("candidate_union_count", 0))
        if self.payload["candidate_union_disposition_count"] > union_count:
            raise ValueError("candidate disposition count cannot exceed candidate union count")


def regions_overlap(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    a = first["spatial_bounds_px"]
    b = second["spatial_bounds_px"]
    spatial = not (
        a["x_max_exclusive"] <= b["x_min"] or b["x_max_exclusive"] <= a["x_min"]
        or a["y_max_exclusive"] <= b["y_min"] or b["y_max_exclusive"] <= a["y_min"]
    )
    af, bf = first["array_frame_interval"], second["array_frame_interval"]
    temporal = not (af[1] <= bf[0] or bf[1] <= af[0])
    return spatial and temporal


@dataclass(frozen=True)
class TruthSetManifest:
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TruthSetManifest":
        result = cls(_copy(payload))
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        return _copy(self.payload)

    def validate(self) -> None:
        validate_dict(self.payload, "truth_set_manifest")
        regions = [TruthSetRegion.from_dict(item).to_dict() for item in self.payload["regions"]]
        revision_refs = self.payload["annotation_revisions"]
        published_ids = [item["revision_id"] for item in revision_refs["published_revisions"]]
        if published_ids != revision_refs["published_revision_ids"]:
            raise ValueError("published annotation revision IDs and references must agree in order")
        region_ids = [item["region_id"] for item in regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("truth-set region IDs must be unique")
        for field, role in (("calibration_region_ids", "calibration"), ("protected_region_ids", "protected")):
            for region_id in self.payload[field]:
                if region_id not in region_ids:
                    raise ValueError(f"{field} references unknown region {region_id}")
                if next(item for item in regions if item["region_id"] == region_id)["role"] != role:
                    raise ValueError(f"{region_id} does not have role {role}")
        calibration = [item for item in regions if item["role"] == "calibration"]
        protected = [item for item in regions if item["role"] == "protected"]
        for first in calibration:
            for second in protected:
                if regions_overlap(first, second):
                    raise ValueError(f"calibration and protected regions overlap: {first['region_id']} / {second['region_id']}")
        width, height, frames = (
            self.payload["source_video"]["width_px"],
            self.payload["source_video"]["height_px"],
            self.payload["source_video"]["frame_count"],
        )
        for region in regions:
            bounds = region["spatial_bounds_px"]
            if bounds["x_max_exclusive"] > width or bounds["y_max_exclusive"] > height:
                raise ValueError(f"region {region['region_id']} exceeds source-pixel dimensions")
            if region["array_frame_interval"][1] > frames:
                raise ValueError(f"region {region['region_id']} exceeds source frame count")
        for candidate in self.payload["candidates"]:
            if candidate["region_id"] not in region_ids:
                raise ValueError("candidate references an unknown region")
            if not all(math.isfinite(float(candidate[key])) for key in ("x_px", "y_px", "score")):
                raise ValueError("candidate coordinates and scores must be finite")


def validate_object_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {"object_id", "region_id", "geometry", "support_ui_frames", "disposition", "morphology", "context", "visibility_confidence", "quality_flags", "reviewer_id", "source_provenance", "annotation_revision_id"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"object record missing fields: {', '.join(missing)}")
    if payload["disposition"] not in OBJECT_DISPOSITIONS:
        raise ValueError("invalid object disposition")
    if payload["morphology"] not in {"center", "membrane", "other", "uncertain"}:
        raise ValueError("invalid object morphology")
    if payload["context"] not in {"isolated", "crowded", "boundary", "uncertain"}:
        raise ValueError("invalid object context")
    ui_to_array_interval(payload["support_ui_frames"])
    return _copy(payload)


def validate_event_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {"event_id", "object_id", "region_id", "onset_ui", "peak_ui", "end_ui", "array_interval", "disposition", "visibility_confidence", "timing_confidence", "quality_flags", "reviewer_id"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"event record missing fields: {', '.join(missing)}")
    onset, peak, end = int(payload["onset_ui"]), int(payload["peak_ui"]), int(payload["end_ui"])
    if not (1 <= onset <= peak <= end):
        raise ValueError("event onset, peak, and end must be ordered one-based UI frames")
    if tuple(payload["array_interval"]) != ui_to_array_interval((onset, end)):
        raise ValueError("event array interval does not match UI frames")
    if payload["disposition"] not in EVENT_DISPOSITIONS:
        raise ValueError("invalid event disposition")
    return _copy(payload)
