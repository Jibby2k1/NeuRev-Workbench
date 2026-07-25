from __future__ import annotations

import json
from pathlib import Path

from neurobench.data.catalog import (
    bounded_named_files,
    dataset_id_from_review,
    dataset_record_for_app,
    discover_dataset_catalog,
    llm_catalog_context,
    query_dataset_catalog,
    raw_video_from_review,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_catalog_joins_manifest_dashboard_and_review_app(tmp_path: Path) -> None:
    workspace = tmp_path
    dataset_root = workspace / "Outputs" / "GammaCFAR" / "burst"
    app = dataset_root / "app"
    raw = workspace / "Inputs" / "Spon Ca Burst" / "burst.tif"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"video")
    _write_json(
        dataset_root / "dataset_manifest.json",
        {
            "schema_version": 1,
            "dataset_id": "burst",
            "name": "Burst movie",
            "frame_rate_hz": 50,
            "paths": {
                "raw_video": "Inputs/Spon Ca Burst/burst.tif",
                "app_dir": "Outputs/GammaCFAR/burst/app",
                "review_data": "Outputs/GammaCFAR/burst/app/review_data.json",
                "annotations": "Outputs/GammaCFAR/burst/app/annotations.json",
            },
        },
    )
    _write_json(
        dataset_root / "dashboard_manifest.json",
        {
            "schema_version": 1,
            "dashboard_id": "burst_dashboard",
            "dashboard_type": "neuron_workbench",
            "dataset_id": "burst",
            "entrypoint": "Outputs/GammaCFAR/burst/app/index.html",
            "serve_command": "neurobench workbench serve --dataset-id burst",
        },
    )
    _write_json(
        app / "review_data.json",
        {
            "dataset": {"dataset_id": "burst", "raw_video": "Inputs/Spon Ca Burst/burst.tif"},
            "video": {
                "name": "burst.tif",
                "frames": 20,
                "height": 10,
                "width": 12,
                "framePattern": "frames/frame_%03d.png",
                "views": [
                    {"view_id": "left", "label": "Left view", "bounds": {"x": 0, "y": 0, "width": 6, "height": 10}},
                    {"view_id": "right", "label": "Right view", "bounds": {"x": 6, "y": 0, "width": 6, "height": 10}},
                ],
            },
            "rois": [{"id": 1}],
            "discovery": {"suggestions": [{"id": "s1"}]},
        },
    )
    _write_json(app / "annotations.json", {"schema_version": 3})
    (app / "index.html").write_text('<div id="manualRoiMode"></div><div id="cfarMaskAnnotationPanel"></div>', encoding="utf-8")

    records = discover_dataset_catalog(workspace)

    assert len(records) == 1
    record = records[0]
    assert record["dataset_id"] == "burst"
    assert record["paths"]["raw_video"] == "Inputs/Spon Ca Burst/burst.tif"
    assert record["video"]["frames"] == 20
    assert record["roi_count"] == 1
    assert record["suggestion_count"] == 1
    assert record["capabilities"]["cfar_annotation"] is True
    assert record["capabilities"]["logical_views"] is True
    assert record["ready"] is True
    assert record["readiness"] == {"review_ready": True, "video_ready": True}
    assert query_dataset_catalog(records, "spon burst")[0]["dataset_id"] == "burst"
    assert query_dataset_catalog(records, "right view")[0]["dataset_id"] == "burst"
    assert llm_catalog_context(records)["datasets"][0]["raw_video"] == "Inputs/Spon Ca Burst/burst.tif"
    assert llm_catalog_context(records)["datasets"][0]["views"][1]["view_id"] == "right"


def test_catalog_for_one_app_prefers_explicit_review_identity(tmp_path: Path) -> None:
    app = tmp_path / "Outputs" / "NeuronReview" / "folder_name" / "app"
    _write_json(
        app / "review_data.json",
        {
            "dataset": {"dataset_id": "declared", "raw_video": "Inputs/declared.tif"},
            "video": {"name": "declared.tif", "frames": 3, "height": 4, "width": 5, "framePattern": "frames/frame_%03d.png"},
        },
    )
    (app / "index.html").write_text("workbench", encoding="utf-8")

    record = dataset_record_for_app(app, workspace_root=tmp_path)

    assert dataset_id_from_review(json.loads((app / "review_data.json").read_text())) == "declared"
    assert raw_video_from_review(json.loads((app / "review_data.json").read_text())) == "Inputs/declared.tif"
    assert record["dataset_id"] == "declared"
    assert record["paths"]["raw_video"] == "Inputs/declared.tif"


def test_raw_video_from_review_accepts_nested_dataset_paths() -> None:
    payload = {"dataset": {"dataset_id": "nested", "paths": {"raw_video": "Inputs/nested.tif"}}}

    assert raw_video_from_review(payload) == "Inputs/nested.tif"


def test_catalog_discovers_labeled_video_collections(tmp_path: Path) -> None:
    raw = tmp_path / "Inputs" / "fish" / "2 left.tif"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"video")
    _write_json(
        tmp_path / "Outputs" / "GridModel" / "fish" / "manifest" / "video_manifest.json",
        {
            "schema_version": 1,
            "dataset_id": "fish_intent",
            "root": "Inputs/fish",
            "frame_rate_hz": 50.0,
            "split_policy": "by_video",
            "label_set": ["left"],
            "label_counts": {"left": 1},
            "videos": [
                {
                    "video_id": "2 left",
                    "fish_id": "2 left",
                    "path": "Inputs/fish/2 left.tif",
                    "label": "left",
                    "condition": "left",
                    "frame_count": 11,
                    "width": 12,
                    "height": 10,
                    "views": [
                        {
                            "view_id": "stimulus_side",
                            "label": "Stimulus side",
                            "bounds": {"x": 0, "y": 0, "width": 6, "height": 10},
                        }
                    ],
                }
            ],
        },
    )

    records = discover_dataset_catalog(tmp_path)
    record = query_dataset_catalog(records, "left")[0]
    llm_record = llm_catalog_context([record])["datasets"][0]

    assert record["dataset_id"] == "fish_intent"
    assert record["video"]["frames_total"] == 11
    assert record["video"]["frame_rate_hz"] == 50.0
    assert record["capabilities"]["video_collection"] is True
    assert record["exists"]["raw_videos"] is True
    assert record["readiness"] == {"review_ready": False, "video_ready": True}
    assert record["ready"] is True
    assert query_dataset_catalog(records, "stimulus side")[0]["dataset_id"] == "fish_intent"
    assert llm_record["videos"][0]["path"] == "Inputs/fish/2 left.tif"
    assert llm_record["videos"][0]["frame_rate_hz"] == 50.0
    assert llm_record["readiness"]["video_ready"] is True


def test_duplicate_video_manifest_retains_common_frame_rate(tmp_path: Path) -> None:
    raw = tmp_path / "Inputs" / "fish" / "left.tif"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"video")
    common = {
        "schema_version": 1,
        "dataset_id": "fish_intent",
        "videos": [{"video_id": "left", "path": "Inputs/fish/left.tif", "label": "left"}],
    }
    _write_json(
        tmp_path / "Outputs" / "a_declared" / "manifest" / "video_manifest.json",
        {**common, "frame_rate_hz": 50.0},
    )
    _write_json(
        tmp_path / "Outputs" / "b_legacy" / "manifest" / "video_manifest.json",
        common,
    )

    record = discover_dataset_catalog(tmp_path)[0]
    llm_record = llm_catalog_context([record])["datasets"][0]

    assert record["video"]["frame_rate_hz"] == 50.0
    assert record["videos"][0]["frame_rate_hz"] == 50.0
    assert llm_record["videos"][0]["frame_rate_hz"] == 50.0


def test_bounded_discovery_prunes_deeper_files(tmp_path: Path) -> None:
    shallow = tmp_path / "one" / "dataset_manifest.json"
    deep = tmp_path / "one" / "two" / "three" / "dataset_manifest.json"
    _write_json(shallow, {})
    _write_json(deep, {})

    assert bounded_named_files(tmp_path, {"dataset_manifest.json"}, max_depth=1) == [shallow]
