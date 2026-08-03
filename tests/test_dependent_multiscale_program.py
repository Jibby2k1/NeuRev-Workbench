import json

import pytest

from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_program import (
    generated_smoke,
    report,
    synthetic,
)


def test_generated_smoke_passes_closure_without_qualifying_noise():
    metrics = generated_smoke(fixture_ids=("compact_isolated_center", "motion_edge_without_neural_activity"))
    assert metrics["fixture_count"] == 2
    assert metrics["gates"]["C1_numerical_reconstruction"] == "pass"
    assert metrics["gates"]["C4_residual_qualification"] == "not_qualified"
    assert all(row["residual_name"] == "noise_candidate" for row in metrics["fixtures"])


def test_synthetic_is_atomic_collision_safe_and_report_is_read_only(tmp_path):
    output = tmp_path / "generated"
    metrics = synthetic(output)
    assert (output / "REPORT.md").is_file()
    assert (output / "progress.jsonl").is_file()
    before = (output / "metrics.json").stat().st_mtime_ns
    loaded = report(output)
    assert loaded["fixture_count"] == metrics["fixture_count"] == 15
    assert (output / "metrics.json").stat().st_mtime_ns == before
    with pytest.raises(FileExistsError):
        synthetic(output)
    assert json.loads((output / "run_state.json").read_text())["status"] == "completed_generated_only"
