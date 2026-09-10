import json
from pathlib import Path

import numpy as np

from neurobench.experiments.ica_whitening_evaluation.config import ICAWhiteningConfig
from neurobench.experiments.ica_whitening_evaluation.preflight import preflight
from neurobench.experiments.ica_whitening_evaluation.synthetic_runner import (
    evaluate_specification,
    run_synthetic_screen,
    select_design_shard,
)

from test_ica_whitening_evaluation_design import _manifest


def _ready_config(tmp_path: Path) -> ICAWhiteningConfig:
    manifest = _manifest(tmp_path)
    video = np.random.default_rng(3).normal(size=(40, 12, 13)).astype(np.float32)
    np.save(tmp_path / "video.npy", video)
    header = (
        "burst_id\tstart_frame_ui\tend_frame_ui\tstart_frame_zero\t"
        "stop_frame_zero_exclusive\tpoint_index\troi_identity\tx_px\ty_px\t"
        "recurrence_count\n"
    )
    (tmp_path / "labels.tsv").write_text(
        header + "1\t12\t18\t11\t18\t1\troi_001\t4\t5\t1\n",
        encoding="utf-8",
    )
    (tmp_path / "labels.json").write_text(
        json.dumps({"total_point_window_labels": 1, "unique_roi_coordinates": 1}),
        encoding="utf-8",
    )
    return ICAWhiteningConfig.from_json(manifest)


def test_one_truth_known_specification_has_required_metrics(tmp_path: Path) -> None:
    config = _ready_config(tmp_path)
    from neurobench.experiments.ica_whitening_evaluation.design import build_design
    result = evaluate_specification(build_design(config)[0], config, cases=("isolated", "pure_noise"))
    assert result["case_count"] == 2
    assert result["mean_truth_source_correlation"] is not None
    assert result["unresolved_accuracy"] in {0.0, 1.0}


def test_smoke_is_resumable_and_validated(tmp_path: Path) -> None:
    config = _ready_config(tmp_path)
    preflight(config, artifact_dir=tmp_path / "preflight")
    first = run_synthetic_screen(config, preflight_dir=tmp_path / "preflight", limit=2)
    second = run_synthetic_screen(config, preflight_dir=tmp_path / "preflight", limit=2)
    assert first["design_is_complete"] and second["design_is_complete"]
    validation = json.loads(
        (config.output_dir / "stages/S1_SMOKE_2/validation.json").read_text()
    )
    assert validation["status"] == "pass"
    assert validation["observed_fit_count"] == 2


def test_shards_are_disjoint_balanced_and_complete(tmp_path: Path) -> None:
    config = _ready_config(tmp_path)
    from neurobench.experiments.ica_whitening_evaluation.design import build_design
    design = build_design(config)
    shards = [select_design_shard(design, index, 4) for index in range(4)]
    identifiers = [{row["fit_id"] for row in shard} for shard in shards]
    assert set.union(*identifiers) == {row["fit_id"] for row in design}
    assert sum(len(values) for values in identifiers) == len(design)
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
