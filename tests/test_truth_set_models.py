from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from neurobench.models.truth_set import TruthSetManifest, TruthSetRegion, array_to_ui_interval, regions_overlap, ui_to_array_interval, validate_event_record, validate_object_record


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "spon_ca_burst_truth_set_v1.example.json"


def manifest() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def test_truth_set_and_region_schemas_validate_synthetic_example() -> None:
    model = TruthSetManifest.from_dict(manifest())
    assert model.payload["truth_set_id"].endswith("_v1")
    assert all(TruthSetRegion.from_dict(item) for item in model.payload["regions"])


def test_frame_conventions_round_trip_exactly() -> None:
    assert ui_to_array_interval([1, 6]) == (0, 6)
    assert array_to_ui_interval([0, 6]) == (1, 6)
    with pytest.raises(ValueError, match="one-based"):
        ui_to_array_interval([0, 2])


def test_source_pixel_bounds_are_enforced() -> None:
    payload = manifest()
    payload["regions"][0]["spatial_bounds_px"]["x_max_exclusive"] = 17
    with pytest.raises(ValueError, match="source-pixel"):
        TruthSetManifest.from_dict(payload)


def test_calibration_and_protected_regions_must_not_overlap() -> None:
    payload = manifest()
    payload["regions"][1]["spatial_bounds_px"]["x_min"] = 7
    payload["regions"][1]["spatial_bounds_px"]["y_min"] = 5
    payload["regions"][1]["ui_frame_interval"] = [6, 12]
    payload["regions"][1]["array_frame_interval"] = [5, 12]
    with pytest.raises(ValueError, match="overlap"):
        TruthSetManifest.from_dict(payload)


def test_object_and_event_records_are_separate_and_linkable() -> None:
    obj = validate_object_record({"object_id": "obj_1", "region_id": "r", "geometry": {"kind": "center", "x": 1, "y": 2}, "support_ui_frames": [2, 5], "disposition": "unresolved", "morphology": "uncertain", "context": "crowded", "visibility_confidence": "low", "quality_flags": {"motion": False}, "reviewer_id": "a", "source_provenance": "raw", "annotation_revision_id": "ann_pub"})
    event = validate_event_record({"event_id": "ev_1", "object_id": "obj_1", "region_id": "r", "onset_ui": 2, "peak_ui": 3, "end_ui": 5, "array_interval": [1, 5], "disposition": "unresolved", "visibility_confidence": "low", "timing_confidence": "low", "quality_flags": {"motion": False}, "reviewer_id": "a"})
    assert obj["object_id"] == event["object_id"]
    assert obj["disposition"] == event["disposition"] == "unresolved"


def test_region_lock_requires_prior_lock_timestamp() -> None:
    region = deepcopy(manifest()["regions"][1])
    region["protected_lock"].update({"unsealed": True, "unsealed_at": "2026-08-16T00:00:00Z", "unsealed_by": "a", "unseal_reason": "review"})
    region["protected_lock"]["locked_at"] = None
    with pytest.raises(ValueError, match="prior lock"):
        TruthSetRegion.from_dict(region)
