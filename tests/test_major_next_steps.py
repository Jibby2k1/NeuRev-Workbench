import json
from pathlib import Path

from neurobench.experiments.neuron_identifiability.major_next_steps import run_detector_level_benchmark, run_generator_family_holdout, run_label_free_false_alarm_benchmark, run_realistic_movie_benchmark


def test_realistic_movie_benchmark_is_complete(tmp_path: Path) -> None:
    result = run_realistic_movie_benchmark(tmp_path / "run", seeds=1)
    validation = json.loads((tmp_path / "run" / "validation.json").read_text())
    assert result["design"]["factorial_cells"] == 36
    assert validation["status"] == "passed"
    assert validation["rows"] == 72
    assert 0 <= result["models"]["spatial_context"]["mean_pixel_auc"] <= 1


def test_detector_level_benchmark_has_identity_metrics_and_disjoint_calibration(tmp_path: Path) -> None:
    result = run_detector_level_benchmark(tmp_path / "detector", seeds=2)
    validation = json.loads((tmp_path / "detector" / "validation.json").read_text())
    assert validation["status"] == "passed"
    assert validation["operating_rows"] == 2 * 36 * 4 * 3
    assert result["design"]["one_to_one_matching"] is True
    assert set(result["operating_points"]) == {"carrier", "spatial_context", "kinetic", "combined"}
    assert validation["calibration_seed_disjoint"] is True


def test_generator_family_holdout_covers_shift_families(tmp_path: Path) -> None:
    result=run_generator_family_holdout(tmp_path/"families",empirical_profile={"simulator_read_sigma":.3},seeds=1)
    validation=json.loads((tmp_path/"families"/"validation.json").read_text())
    assert validation["status"] == "passed"
    assert validation["movies"] == 18
    assert validation["holdout_rows"] == 24
    assert "combined_shift" in result["design"]["families"]


def test_label_free_thresholds_use_disjoint_families(tmp_path: Path) -> None:
    result=run_label_free_false_alarm_benchmark(tmp_path/"false_alarm",empirical_profile={"simulator_read_sigma":.3},seeds=1)
    validation=json.loads((tmp_path/"false_alarm"/"validation.json").read_text())
    assert validation["status"] == "passed"
    assert validation["thresholds_use_no_source_truth"] is True
    assert result["split"]["family_overlap"] is False
    assert set(result["untouched_evaluation"]) == {"spatial_context","gated_fusion"}
