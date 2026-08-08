from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from neurobench.workbench.model_proposals import _occurrence_rows, build_model_proposal_package


def _source_app(tmp_path: Path) -> Path:
    app = tmp_path / "source_app"
    for view in ("raw", "processed"):
        root = app / "frames" / view
        root.mkdir(parents=True, exist_ok=True)
        for frame in (10, 11):
            (root / f"frame_{frame:04d}.png").write_bytes(b"fixture")
    review = {
        "dataset": {"dataset_id": "pending_fish", "name": "Pending fish"},
        "video": {"name": "pending.tif", "width": 8, "height": 6, "frames": 2, "fps": 50.0, "framePattern": "frames/raw/frame_%04d.png"},
        "parameters": {"frozen_lane": "raw_msica_msln_v1"},
        "rois": [],
        "annotationCorrection": {
            "schema_version": 1,
            "source_video_id": "pending_fish",
            "read_only": False,
            "revision": {
                "frozenRunId": "raw_msica_msln_v1",
            },
            "view_contracts": [
                {
                    "schema_version": 1,
                    "view_id": "raw",
                    "source_video_id": "pending_fish",
                    "shape_tyx": [2, 6, 8],
                    "source_to_view": {"kind": "identity"},
                    "frame_mapping": {"kind": "identity", "offset": 9},
                    "intensity_semantics": "raw_amplitude",
                    "frame_pattern": "frames/raw/frame_%04d.png",
                },
                {
                    "schema_version": 1,
                    "view_id": "processed",
                    "source_video_id": "pending_fish",
                    "shape_tyx": [2, 6, 8],
                    "source_to_view": {"kind": "identity"},
                    "frame_mapping": {"kind": "identity", "offset": 9},
                    "intensity_semantics": "normalized_signed_visualization",
                    "frame_pattern": "frames/processed/frame_%04d.png",
                },
            ],
            "expert_rois": [{"id": "expert_leak", "source_xy": [1, 1], "ui_frame": 10}],
            "matches": [{"expert_id": "expert_leak", "model_id": "model_1"}],
            "model_rois": [
                {
                    "id": "model_1",
                    "source_xy": [2.0, 3.0],
                    "ui_frame": 10,
                    "events": [10, 11],
                    "eventIntervals": [[10, 10], [11, 11]],
                    "geometry": {"kind": "center"},
                    "status": "matched",
                    "linked_expert_id": "expert_leak",
                    "traces": {
                        "raw": {"pixel": [1.0, 2.0], "roi_mean": [1.2, 2.1]},
                        "processed": {"pixel": [0.1, 0.8], "roi_mean": [0.2, 0.7]},
                    },
                    "members": [
                        {
                            "burst": 1,
                            "x": 2,
                            "y": 3,
                            "start_ui": 10,
                            "end_ui": 10,
                            "rank": 1,
                            "score": 9.0,
                            "expert_supported": True,
                            "matched_expert_roi": "expert_leak",
                            "match_distance_px": 0.2,
                        },
                        {
                            "burst": 2,
                            "x": 2.2,
                            "y": 3.1,
                            "start_ui": 11,
                            "end_ui": 11,
                            "rank": 2,
                            "score": 7.0,
                        },
                    ],
                }
            ],
        },
    }
    (app / "review_data.json").write_text(json.dumps(review), encoding="utf-8")
    (app / "architecture_runs.json").write_text('{"schema_version":1,"runs":[]}\n', encoding="utf-8")
    return app


def test_model_proposal_package_removes_expert_evidence_and_writes_review_artifacts(tmp_path: Path) -> None:
    source = _source_app(tmp_path)
    output = tmp_path / "package"

    result = build_model_proposal_package(
        source_app_dir=source,
        output_root=output,
        event_source="supplied",
    )

    review = json.loads((output / "app" / "review_data.json").read_text(encoding="utf-8"))
    correction = review["annotationCorrection"]
    annotations = json.loads((output / "app" / "annotations.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "proposal_manifest.json").read_text(encoding="utf-8"))
    long_text = (output / "proposal_exports" / "model_proposals_long.tsv").read_text(encoding="utf-8")

    assert result["model_identity_count"] == 1
    assert result["model_occurrence_count"] == 2
    assert correction["mode"] == "model_only"
    assert correction["expert_annotation_state"] == "not_applicable_pending_labels"
    assert correction["expert_rois"] == []
    assert correction["matches"] == []
    assert correction["model_rois"][0]["status"] == "unknown"
    assert correction["model_rois"][0]["linked_expert_id"] == ""
    assert "expert_supported" not in correction["model_rois"][0]["members"][0]
    assert "matched_expert_roi" not in correction["model_rois"][0]["members"][0]
    assert annotations["rois"] == {}
    assert manifest["event_windows"] == [
        {"burst_id": 1, "start_frame_ui": 10, "end_frame_ui": 10},
        {"burst_id": 2, "start_frame_ui": 11, "end_frame_ui": 11},
    ]
    assert "model_1__burst_001__occurrence_001" in long_text
    assert "model_1__burst_002__occurrence_002" in long_text
    assert "expert_leak" not in long_text
    assert (output / "audit" / "1_Expert_Annotations" / "status.json").is_file()
    assert (output / "audit" / "3_Comparison" / "status.json").is_file()
    assert (output / "app" / "frames" / "raw" / "frame_0010.png").is_file()


def test_model_proposal_workbooks_are_valid_ooxml_and_separate_blinded_coordinates(tmp_path: Path) -> None:
    output = tmp_path / "package"
    build_model_proposal_package(source_app_dir=_source_app(tmp_path), output_root=output)

    proposal = output / "proposal_exports" / "MODEL_PROPOSALS_FOR_REVIEW.xlsx"
    blinded = output / "proposal_exports" / "BLINDED_EXPERT_TEMPLATE.xlsx"
    with zipfile.ZipFile(proposal) as archive:
        assert archive.testzip() is None
        workbook = archive.read("xl/workbook.xml").decode("utf-8")
        layout = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
        details = archive.read("xl/worksheets/sheet2.xml").decode("utf-8")
        provenance = archive.read("xl/worksheets/sheet4.xml").decode("utf-8")
    with zipfile.ZipFile(blinded) as archive:
        assert archive.testzip() is None
        blind_layout = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")

    assert "Expert-compatible layout" in workbook
    assert "Model proposal details" in workbook
    assert "UNREVIEWED" in layout
    assert "<v>2.0</v>" in layout and "<v>3.0</v>" in layout
    assert "model_1" in details
    assert "unknown" in details
    assert "NO_EXPERT" not in details
    assert "candidate_interpretation" in provenance
    assert "BLINDED EXPERT ANNOTATION TEMPLATE" in blind_layout
    assert "model_1" not in blind_layout




def test_occurrence_ids_remain_unique_when_one_model_repeats_within_a_burst() -> None:
    rows = _occurrence_rows(
        [
            {
                "id": "model_repeat",
                "source_xy": [1, 2],
                "members": [
                    {"burst": 1, "start_ui": 10, "end_ui": 12, "rank": 1, "score": 4, "x": 1, "y": 2},
                    {"burst": 1, "start_ui": 10, "end_ui": 12, "rank": 2, "score": 3, "x": 2, "y": 3},
                ],
            }
        ],
        event_source="supplied",
    )

    identifiers = [row["occurrence_id"] for row in rows]
    assert len(identifiers) == len(set(identifiers)) == 2


def test_model_proposal_package_refuses_output_collision(tmp_path: Path) -> None:
    source = _source_app(tmp_path)
    output = tmp_path / "package"
    output.mkdir()

    with pytest.raises(FileExistsError, match="collision"):
        build_model_proposal_package(source_app_dir=source, output_root=output)
