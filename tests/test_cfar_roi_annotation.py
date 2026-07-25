from __future__ import annotations

import csv
from pathlib import Path
import json
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "neurobench" / "workbench" / "assets"


def test_workbench_exposes_cfar_semantic_mask_controls():
    html = (ASSETS / "workbench.html").read_text(encoding="utf-8")
    source = (ASSETS / "src" / "24_cfar_mask_annotation.js").read_text(
        encoding="utf-8"
    )
    core = (ASSETS / "src" / "20_review_core.js").read_text(encoding="utf-8")

    for control_id in (
        "cfarMaskTarget",
        "cfarMaskTool",
        "cfarMaskBrushRadius",
        "cfarFloodTolerance",
        "cfarFloodBound",
        "cfarFloodPadding",
        "cfarFloodRadius",
        "cfarMaskUndoBtn",
        "cfarMaskClearBtn",
        "cfarMaskDoneBtn",
        "cfarMaskStatus",
    ):
        assert f'id="{control_id}"' in html

    assert "Brush select" in html
    assert "Brush deselect" in html
    assert "Flood select" in html
    assert "Flood deselect" in html
    assert "ROI box + padding" in html
    assert "Radius from click" in html
    assert "Full frame (capped)" in html
    assert "CFAR_FLOOD_PIXEL_LIMIT = 50000" in source
    assert "connectedCfarFlood" in source
    assert "pushCfarMaskHistory" in source
    assert "const cfarMaskUndoHistory = new Map()" in source
    assert "foreground_bits: cfarMaskBitset" in source
    assert "delete regions.edit_history" in source
    assert "regions.edit_history =" not in source
    assert "edit_history:" not in source
    assert "regions[otherKey]" in source
    assert "drawSelectedCfarMasks" in core
    assert "drawCfarMasksOnCrop" in core


def test_generated_bundle_contains_cfar_module_in_order():
    bundle = (ASSETS / "workbench.js").read_text(encoding="utf-8")
    assert "// --- 24_cfar_mask_annotation.js ---" in bundle
    assert bundle.index("// --- 20_review_core.js ---") < bundle.index(
        "// --- 24_cfar_mask_annotation.js ---"
    )
    assert bundle.index("// --- 24_cfar_mask_annotation.js ---") < bundle.index(
        "// --- 25_review_controls.js ---"
    )


def test_annotation_tools_are_visible_in_guided_mode_and_share_one_canvas():
    html = (ASSETS / "workbench.html").read_text(encoding="utf-8")
    state = (ASSETS / "src" / "10_state_persistence.js").read_text(encoding="utf-8")

    assert 'class="annotationWorkspace"' in html
    assert 'class="toolbarDisclosure annotationToolPanel" id="roiAnnotationPanel"' in html
    assert 'class="toolbarDisclosure annotationToolPanel" id="cfarMaskAnnotationPanel"' in html
    assert 'expertOnly" id="cfarMaskAnnotationPanel"' not in html
    assert 'id="overlay" tabindex="0" role="application"' in html
    assert 'id="videoViewSelect"' in html
    assert "reviewLayoutVersion: 2" in state
    assert "reviewSideBySide: false" in state
    assert "activeVideoViewId: ''" in state
    assert ".replace(/%0?(\\d*)d/g" in state
    assert "function activeLogicalVideoView()" in (ASSETS / "src" / "20_review_core.js").read_text(encoding="utf-8")
    assert "view_id:activeView?.view_id" in (ASSETS / "src" / "24_cfar_mask_annotation.js").read_text(encoding="utf-8")
    controls = (ASSETS / "src" / "25_review_controls.js").read_text(encoding="utf-8")
    assert "setSetting('activeVideoViewId', event.target.value)" in controls
    assert "focusLogicalVideoView();" in controls
    start_manual = controls.split("document.getElementById('startManualNeuronBtn').onclick", 1)[1].split("};", 1)[0]
    assert "ensureSingleAnnotationView();" in start_manual


def test_review_and_qc_polygon_helpers_do_not_collide():
    core = (ASSETS / "src" / "20_review_core.js").read_text(encoding="utf-8")
    qc = (ASSETS / "src" / "70_dataset_qc.js").read_text(encoding="utf-8")
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((ASSETS / "src").glob("*.js"))
    )

    assert "function reviewPointInPolygon(x, y, polygon)" in core
    assert "reviewPointInPolygon(x + 0.5, y + 0.5, path)" in core
    assert "function qcPointInPolygon(point, polygon)" in qc
    assert "qcPointInPolygon(point, polygon)" in qc
    assert "function pointInPolygon(" not in combined
    assert "function cleanTsv(" not in combined


def test_review_lasso_polygon_is_orientation_independent():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the browser geometry behavior check")
    core = (ASSETS / "src" / "20_review_core.js").read_text(encoding="utf-8")
    start = core.index("function reviewPointInPolygon")
    end = core.index("\n\nfunction lassoPoints", start)
    function_source = core[start:end]
    script = "\n".join(
        [
            function_source,
            "const clockwise = [{x:0,y:0},{x:0,y:10},{x:10,y:0}];",
            "const counterclockwise = [...clockwise].reverse();",
            "const result = [",
            "  reviewPointInPolygon(2, 2, clockwise),",
            "  reviewPointInPolygon(2, 2, counterclockwise),",
            "  reviewPointInPolygon(9, 9, clockwise),",
            "  reviewPointInPolygon(9, 9, counterclockwise),",
            "];",
            "process.stdout.write(JSON.stringify(result));",
        ]
    )
    result = subprocess.run([node, "-e", script], text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [True, True, False, False]


def test_browser_frame_pattern_supports_general_padding_widths():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the browser frame-pattern behavior check")
    state = (ASSETS / "src" / "10_state_persistence.js").read_text(encoding="utf-8")
    start = state.index("function framePatternPath")
    end = state.index("\nfunction rebaseRelativeAsset", start)
    function_source = state[start:end]
    script = "\n".join(
        [
            "const artifactUrl = value => value;",
            function_source,
            "process.stdout.write(JSON.stringify([",
            "  framePatternPath('frame_%04d.png', 7),",
            "  framePatternPath('frame_{frame:05d}.png', 7),",
            "  framePatternPath('frame_{frame03}.png', 7),",
            "  framePatternPath('frame_{frame}.png', 7),",
            "]));",
        ]
    )
    result = subprocess.run([node, "-e", script], text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["frame_0007.png", "frame_00007.png", "frame_007.png", "frame_7.png"]


def test_annotation_export_preserves_masks_and_reports_counts(tmp_path):
    from neurobench.exports.annotations import export_annotation_profile

    review_data = {
        "dataset": {"dataset_id": "cfar-mask-fixture"},
        "video": {
            "width": 16,
            "height": 16,
            "frames": 5,
            "framePattern": "frame_%04d.png",
        },
        "rois": [{"id": 1, "events": []}],
    }
    annotations = {
        "schema_version": 3,
        "rois": {
            "1": {
                "cell_state": "accepted",
                "cfar_regions": {
                    "schema_version": 1,
                    "foreground_points": [[4, 4], [4, 5]],
                    "background_points": [[1, 1], [1, 2], [2, 1]],
                    "reference_frames": [2, 4],
                    "provenance": "manual_cfar_feature_annotation",
                },
            },
            "MR_1": {
                "cell_state": "accepted",
                "cfar_regions": {
                    "schema_version": 1,
                    "foreground_points": [[8, 8]],
                    "background_points": [[6, 6], [6, 7]],
                    "reference_frames": [3],
                    "provenance": "manual_cfar_feature_annotation",
                },
            },
        },
        "events": {},
        "virtualRois": {
            "MR_1": {
                "id": "MR_1",
                "roi_kind": "manual_lasso",
                "source_roi_ids": [],
                "cell_state": "accepted",
                "points": [[8, 8]],
            }
        },
    }

    export_annotation_profile(
        review_data,
        annotations,
        tmp_path,
        profile="accepted_only",
        created_at="2026-07-21T00:00:00Z",
    )

    with (tmp_path / "accepted_rois.tsv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = {row["roi_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    exported = json.loads((tmp_path / "annotations_v3.json").read_text(encoding="utf-8"))

    assert rows["1"]["cfar_foreground_px"] == "2"
    assert rows["1"]["cfar_background_px"] == "3"
    assert rows["1"]["cfar_reference_frames"] == "2,4"
    assert rows["MR_1"]["cfar_foreground_px"] == "1"
    assert rows["MR_1"]["cfar_background_px"] == "2"
    assert exported["rois"]["1"]["cfar_regions"]["foreground_points"] == [
        [4, 4],
        [4, 5],
    ]
