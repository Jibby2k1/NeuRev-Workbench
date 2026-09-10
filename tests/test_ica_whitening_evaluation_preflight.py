import json
from pathlib import Path

import numpy as np

from neurobench.experiments.ica_whitening_evaluation.config import ICAWhiteningConfig
from neurobench.experiments.ica_whitening_evaluation.preflight import (
    matching_preflight,
    preflight,
)

from test_ica_whitening_evaluation_design import _manifest


def test_preflight_freezes_design_inputs_and_projection(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    video = np.random.default_rng(3).normal(size=(40, 12, 13)).astype(np.float32)
    np.save(tmp_path / "video.npy", video)
    (tmp_path / "labels.tsv").write_text(
        "burst_id\tstart_frame_ui\tend_frame_ui\tstart_frame_zero\t"
        "stop_frame_zero_exclusive\tpoint_index\troi_identity\tx_px\ty_px\t"
        "recurrence_count\n"
        "1\t12\t18\t11\t18\t1\troi_001\t4\t5\t1\n",
        encoding="utf-8",
    )
    (tmp_path / "labels.json").write_text(
        json.dumps({"total_point_window_labels": 1, "unique_roi_coordinates": 1}),
        encoding="utf-8",
    )
    config = ICAWhiteningConfig.from_json(manifest)
    result = preflight(config, artifact_dir=tmp_path / "preflight")
    assert result["ready"]
    assert result["design"]["fit_count"] > 0
    assert (tmp_path / "preflight/frozen_design.tsv").is_file()
    assert (tmp_path / "preflight/label_projection_overlay.png").is_file()
    assert matching_preflight(config, tmp_path / "preflight")["ready"]
