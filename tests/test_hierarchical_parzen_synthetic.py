import json

import numpy as np
import pytest

from neurobench.experiments.hierarchical_parzen_ica.evaluation import (
    run_stage1_synthetic_matrix,
)
from neurobench.experiments.hierarchical_parzen_ica.synthetic import (
    STAGE1_SYNTHETIC_CASES,
    generate_stage1_synthetic_case,
    stage1_synthetic_suite,
)


def test_stage1_suite_is_deterministic_complete_and_exactly_closed() -> None:
    first = stage1_synthetic_suite((7,))
    second = stage1_synthetic_suite((7,))
    assert len(first) == len(STAGE1_SYNTHETIC_CASES) == 12
    assert tuple(case.case_id for case in first) == STAGE1_SYNTHETIC_CASES
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.observation, right.observation)
        closure = (
            left.observation
            - left.background
            - left.signal
            - left.artifact
            - left.noise
        )
        assert np.max(np.abs(closure)) < 1e-12


def test_saturation_and_translation_remain_explicit_artifacts() -> None:
    saturation = generate_stage1_synthetic_case(
        "saturation_clipping", 13
    )
    translation = generate_stage1_synthetic_case("translation_edge", 13)
    assert np.any(saturation.signal)
    assert np.any(saturation.artifact)
    assert np.any(translation.artifact)
    assert not np.any(translation.signal)


def test_tiny_matrix_writes_collision_safe_explicit_artifacts(tmp_path) -> None:
    destination = tmp_path / "matrix"
    result = run_stage1_synthetic_matrix(
        destination,
        seeds=(7,),
        case_ids=("slow_ramp_plateau", "translation_edge", "pure_noise"),
    )
    assert result["combination_count"] == 12
    assert result["summary"]["completed_count"] == 12
    assert (
        result["summary"]["gates"]["stage1_numerical_stability"]
        == "pass"
    )
    for name in (
        "manifest.json",
        "progress.jsonl",
        "results.json",
        "results.tsv",
        "summary.json",
        "REPORT.md",
    ):
        assert (destination / name).is_file()
    manifest = json.loads(
        (destination / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["real_data_used"] is False
    assert manifest["labels_used"] is False
    rows = json.loads(
        (destination / "results.json").read_text(encoding="utf-8")
    )
    pure_noise = [row for row in rows if row["case_id"] == "pure_noise"]
    assert all(row["background_nmse"] is None for row in pure_noise)
    assert all(row["applied_feedback_safe"] for row in rows)
    with pytest.raises(FileExistsError):
        run_stage1_synthetic_matrix(
            destination,
            seeds=(7,),
            case_ids=("slow_ramp_plateau",),
        )
