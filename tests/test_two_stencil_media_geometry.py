"""Independent post-hoc probe geometry and complete video-inventory checks."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from neurobench.experiments.gamma_ls_difference.two_stencil_media_geometry import (
    STAGES,
    validate_audit_media_geometry,
    validate_state_geometry,
    validate_video_geometry,
)


def contract():
    return {"stage_sequence": STAGES, "source_frames_ui": list(range(1, 13)),
            "source_fps": 50, "fullfield_step": 5, "fullfield_fps": 10,
            "fullfield_panel_image_area_size_wh": [180, 136],
            "closeup_panel_image_area_size_wh": [144, 112]}


def video(path, frames, *, kind="closeup", section="expert"):
    full = kind == "fullfield"
    fps = 10 if full else 50
    return {"path": str(path), "source_frames_ui": frames,
            "source_frames_sha256": hashlib.sha256(json.dumps(frames, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "frame_count": len(frames), "annotation_section": section, "fps": fps,
            "probe": {"nb_frames": str(len(frames)), "r_frame_rate": f"{fps}/1",
                      "duration": f"{len(frames)/fps:.6f}", "width": 720 if full else 576,
                      "height": 332 if full else 284},
            "panel_image_geometry": {"panel_image_area_size_wh": [180, 136] if full else [144, 112]},
            "encoding_revision": "encoding_v2", "all_frames_rgb_byte_exact": True,
            "all_frames_marker_expectations_pass": True, "decoded_frame_count": len(frames)}


def test_rational_fps_and_duration_follow_played_samples_not_source_span():
    row = video("closeup.mp4", [2, 10, 11])
    row["probe"]["r_frame_rate"] = "100/2"
    result = validate_video_geometry(row, contract=contract(), expected_frames_ui=[2, 10, 11],
                                     kind="closeup", annotation_section="expert")
    assert result["fps_fraction"] == "50"
    assert result["expected_duration_fraction"] == "3/50"
    assert result["canvas_width"] == 576 and result["canvas_height"] == 284
    row["probe"]["duration"] = "0.200000"  # source span is not displayed duration
    with pytest.raises(ValueError, match="duration"):
        validate_video_geometry(row, contract=contract(), expected_frames_ui=[2, 10, 11],
                                 kind="closeup", annotation_section="expert")


@pytest.mark.parametrize("field,value,error", [
    ("r_frame_rate", "25/1", "FPS"),
    ("r_frame_rate", "0/0", "fraction"),
    ("duration", "0.080000", "duration"),
    ("width", 575, "dimensions"),
    ("height", 285, "dimensions"),
    ("nb_frames", "2", "frame count"),
])
def test_tampered_probe_fields_fail(field, value, error):
    row = video("closeup.mp4", [2, 10, 11])
    row["probe"][field] = value
    with pytest.raises(ValueError, match=error):
        validate_video_geometry(row, contract=contract(), expected_frames_ui=[2, 10, 11],
                                 kind="closeup", annotation_section="expert")


def _state(root):
    plan = {"fullfield_source_frames_ui": [1, 6, 11, 12], "expected_video_count": 4,
            "expert_roi_count": 1, "model_roi_count": 1,
            "expert_closeup_source_frames_ui": {"roi001": [2, 3, 4]},
            "model_closeup_source_frames_ui": {"site001": [7, 8, 9]}}
    rows = [video(root / "1_Expert_Annotations/videos/expert_sequential_full_field.mp4", [1, 6, 11, 12], kind="fullfield"),
            video(root / "2_Model_Annotations/videos/model_sequential_full_field.mp4", [1, 6, 11, 12], kind="fullfield", section="model"),
            video(root / "1_Expert_Annotations/videos/closeups/roi001.mp4", [2, 3, 4]),
            video(root / "2_Model_Annotations/videos/closeups/site001.mp4", [7, 8, 9], section="model")]
    return plan, rows


def test_state_requires_both_fullfields_and_every_closeup_exactly_once(tmp_path):
    plan, rows = _state(tmp_path)
    result = validate_state_geometry(rows, output_root=tmp_path, contract=contract(), plan=plan)
    assert result["validated_video_count"] == 4
    assert result["validated_displayed_frame_count"] == 14
    with pytest.raises(ValueError, match="omits or adds"):
        validate_state_geometry(rows[:-1], output_root=tmp_path, contract=contract(), plan=plan)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_state_geometry([*rows, rows[-1]], output_root=tmp_path, contract=contract(), plan=plan)


def test_inclusive_last_source_frame_and_exact_occurrence_map_are_required(tmp_path):
    plan, rows = _state(tmp_path)
    bad_plan = deepcopy(plan)
    bad_plan["fullfield_source_frames_ui"] = [1, 6, 11]
    with pytest.raises(ValueError, match="inclusive final"):
        validate_state_geometry(rows, output_root=tmp_path, contract=contract(), plan=bad_plan)
    changed = deepcopy(rows)
    changed[-1]["source_frames_ui"] = [7, 9]
    with pytest.raises(ValueError, match="source-frame map"):
        validate_state_geometry(changed, output_root=tmp_path, contract=contract(), plan=plan)


def test_incomplete_campaign_replaces_stale_pass_with_failure(tmp_path: Path):
    for filename, payload in [("protocol.json", {"cells": []}),
                              ("audit_completion.json", {"audit_revision": "encoding_v2", "q1_scientific_audit_complete": False}),
                              ("audit_inventory_plan_encoding_v2.json", {})]:
        (tmp_path / filename).write_text(json.dumps(payload))
    (tmp_path / "results.tsv").write_text("small numerical fixture\n")
    output = tmp_path / "audit_media_geometry_validation.json"
    output.write_text(json.dumps({"status": "PASS", "media_geometry_complete": True}))
    with pytest.raises(ValueError, match="must complete"):
        validate_audit_media_geometry(tmp_path)
    failure = json.loads(output.read_text())
    assert failure["status"] == "FAIL" and not failure["media_geometry_complete"]
    assert failure["expected_state_count"] == 25 and failure["expected_video_count"] == 8176
