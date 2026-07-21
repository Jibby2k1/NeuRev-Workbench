from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from neurobench.dynamics.manual_annotations import evaluate_manual_roi_spikes_on_dataset, import_manual_roi_spikes


def _write_minimal_xlsx(path: Path) -> None:
    sheet = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="inlineStr"><is><t>15_right_crop512x512</t></is></c></row>
    <row r="2"><c r="A2" t="inlineStr"><is><t>Node #</t></is></c><c r="B2" t="inlineStr"><is><t>Coordinates (x,y)</t></is></c><c r="E2" t="inlineStr"><is><t>Spike Frames</t></is></c></row>
    <row r="3"><c r="A3"><v>1</v></c><c r="B3"><v>256</v></c><c r="C3"><v>128</v></c><c r="D3"><v>12</v></c><c r="E3" t="inlineStr"><is><t>3-4</t></is></c><c r="F3" t="inlineStr"><is><t>9-7</t></is></c></row>
  </sheetData>
</worksheet>
"""
    workbook = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>
</workbook>
"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>
"""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet)


def test_import_manual_roi_spikes_projects_crop512_to_grid128(tmp_path):
    xlsx = tmp_path / "ROIs_15_right_crop512x512.xlsx"
    _write_minimal_xlsx(xlsx)

    manifest = import_manual_roi_spikes(inputs=[xlsx], out_dir=tmp_path / "out")

    assert manifest["annotation_count"] == 1
    roi = manifest["annotations"][0]
    assert roi["video_id"] == "15 right"
    assert roi["grid_col"] == 64
    assert roi["grid_row"] == 32
    assert roi["spike_interval_count"] == 1
    assert {warning["kind"] for warning in manifest["warnings"]} == {"reversed_frame_range"}
    assert Path(manifest["roi_tsv_path"]).exists()


def test_evaluate_manual_roi_spikes_on_dataset_scores_event_windows(tmp_path):
    arrays_path = tmp_path / "arrays.npz"
    windows = np.zeros((5, 2, 1, 128, 128), dtype=np.float32)
    targets = np.zeros((5, 1, 128, 128), dtype=np.float32)
    windows[:, -1, 0, 32, 64] = 0.1
    targets[2:4, 0, 32, 64] = 0.9
    np.savez(
        arrays_path,
        windows=windows,
        targets=targets,
        window_video_ids=np.asarray(["15 right"] * 5, dtype="U64"),
        target_frame_indices=np.asarray([1, 2, 3, 4, 5], dtype=np.int64),
    )
    dataset = {
        "dataset_id": "unit",
        "array_path": str(arrays_path),
        "splits": {"train_video_ids": ["15 right"], "val_video_ids": [], "test_video_ids": []},
    }
    annotations = {
        "annotations": [
            {
                "annotation_id": "15 right:roi_1",
                "video_id": "15 right",
                "roi_id": "1",
                "grid_row": 32,
                "grid_col": 64,
                "spike_interval_count": 1,
                "spike_intervals": [{"start_frame": 3, "end_frame": 4}],
            }
        ]
    }

    report = evaluate_manual_roi_spikes_on_dataset(dataset=dataset, annotations=annotations, out_dir=tmp_path / "eval")

    row = report["rows"][0]
    assert row["matched_event_window_count"] == 2
    assert row["split"] == "train"
    assert row["event_target_mean"] == pytest.approx(0.9)
    assert row["event_persistence_mse"] > 0
    assert Path(report["markdown_path"]).exists()


def test_manual_roi_temporal_cnn_followup_manifest_is_memory_guarded():
    manifest_path = Path("Outputs/GridModel/060126_crop512_grid128_max_v1/plans/manual_roi_spike_temporal_cnn_followup_v1/next_sweep_manifest.json")
    if not manifest_path.exists():
        pytest.skip("manual ROI temporal-CNN follow-up manifest has not been generated in this checkout")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["manifest_kind"] == "manual_roi_temporal_cnn_followup"
    assert manifest["planned_experiment_count"] == len(manifest["experiments"])
    assert 1 <= len(manifest["experiments"]) <= 16
    assert {spec["kind"] for spec in manifest["experiments"]} == {"temporal_cnn_pixel"}
    assert {spec["dataset_key"] for spec in manifest["experiments"]} == {"w8_s1_h2", "w8_s1_h5"}
    for spec in manifest["experiments"]:
        params = spec["params"]
        assert params["batch_size"] == 2
        assert params["recommended_batch_size"] == 2
        assert params["loss_mode"] == "residual_mse"
        assert params["manual_roi_gate"] == "required_score_after_completion"
        assert "manual_roi_annotation_manifest" in params
