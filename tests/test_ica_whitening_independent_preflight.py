from __future__ import annotations

import json
from pathlib import Path

from neurobench.experiments.ica_whitening_evaluation.independent_preflight import (
    run_independent_preflight,
)


def test_independent_preflight_excludes_identity_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "root"
    dataset = root / "dataset"
    (dataset / "manifest").mkdir(parents=True)
    labels_dir = dataset / "annotations" / "manual_roi_spikes_v1"
    labels_dir.mkdir(parents=True)
    video_dir = root / "videos"
    video_dir.mkdir()
    for video_id in ("good", "bad"):
        (video_dir / f"{video_id}.tif").write_bytes(video_id.encode())
    videos = [{
        "video_id": video_id, "path": f"videos/{video_id}.tif",
        "frame_count": 10, "height": 2, "width": 2,
    } for video_id in ("good", "bad")]
    (dataset / "manifest" / "video_manifest.json").write_text(json.dumps({
        "frame_rate_hz": 50.0, "videos": videos,
    }))
    (dataset / "crop_manifest.json").write_text(json.dumps({"videos": [{
        "output_path": f"videos/{video_id}.tif", "output_shape": [10, 2, 2],
    } for video_id in ("good", "bad")]}))
    annotations = [{
        "video_id": video_id, "source_file": f"{video_id}.xlsx",
        "source_sheet_title": video_id, "spike_intervals": [{"start_frame": 2, "end_frame": 3}],
    } for video_id in ("good", "bad")]
    (labels_dir / "manual_roi_spike_annotations.json").write_text(json.dumps({
        "frame_rate_hz": 50.0, "video_roi_counts": {"good": 1, "bad": 1},
        "annotations": annotations,
        "warnings": [{"file": "bad.xlsx", "kind": "title_mismatch"}],
    }))
    result = run_independent_preflight(
        dataset_root=root, dataset_dir=dataset, destination=tmp_path / "out",
    )
    assert result["eligible_recordings"] == ["good"]
    by_id = {row["video_id"]: row for row in result["recordings"]}
    assert by_id["good"]["spike_interval_count"] == 1
    assert by_id["bad"]["exclusion_reasons"] == ["workbook_video_identity_mismatch"]
    assert result["confirmation_guard"]["unmatched_candidates"] == "unknown_not_negative"
