from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neurobench.experiments.hierarchical_parzen_ica.innovation_grid import (
    _aggregate_observations,
    _grid_counts,
    _score,
)
from neurobench.experiments.hierarchical_parzen_ica.innovation_grid_config import (
    InnovationGridConfig,
    InnovationGridConfigError,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/spon_ca_burst_stochastic_architecture_grid.example.json"


def test_large_grid_is_unique_and_canonicalizes_zero_correction() -> None:
    config = InnovationGridConfig.load(EXAMPLE)
    counts = _grid_counts(config)
    assert counts == {
        "innovation_unique": 130,
        "fixed_point": 55,
        "screen_total": 185,
        "always_evaluated_controls": 4,
    }
    ids = [str(row["lane_id"]) for row in config.innovation_specs]
    assert len(ids) == len(set(ids))
    assert sum(float(row["correction_fraction"]) == 0 for row in config.innovation_specs) == 5
    assert "innovation_h10_e0.1_c4" in ids


def test_fixed_grid_is_stable_and_steady_state_parameterized() -> None:
    config = InnovationGridConfig.load(EXAMPLE)
    assert sum(float(row["steady_state_observation_fraction"]) == 0 for row in config.fixed_specs) == 1
    assert any(row["lane_id"] == "fixed_static_quiet_reference" for row in config.fixed_specs)
    for row in config.fixed_specs:
        memory = float(row["memory_coefficient"])
        current = float(row["current_coefficient"])
        steady = float(row["steady_state_observation_fraction"])
        assert 0 < memory < 1
        assert current == pytest.approx((1 - memory) * steady)
        assert 0 <= steady <= 1


def test_unknown_manifest_field_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["grid"]["blind_sweep"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(InnovationGridConfigError, match="fields differ"):
        InnovationGridConfig.load(path)


def test_observation_aggregation_excludes_heldout_burst() -> None:
    observations = [
        {
            "eligible": True, "burst_id": 1, "peak_retention": 0.1,
            "area_retention": 0.2, "late_retention": 0.3,
            "waveform_correlation": 0.4,
        },
        {
            "eligible": True, "burst_id": 2, "peak_retention": 0.9,
            "area_retention": 0.8, "late_retention": 0.7,
            "waveform_correlation": 0.6,
        },
        {
            "eligible": False, "burst_id": 2, "peak_retention": 99,
            "area_retention": 99, "late_retention": 99,
            "waveform_correlation": 1,
        },
    ]
    result = _aggregate_observations(observations, excluded_burst=1)
    assert result["eligible_observations"] == 1
    assert result["median_peak_retention"] == pytest.approx(0.9)
    assert result["median_area_retention"] == pytest.approx(0.8)
    assert result["median_late_retention"] == pytest.approx(0.7)


def test_screen_hard_gate_requires_retention_and_noise_controls() -> None:
    metrics = {
        "median_peak_retention": 0.9,
        "median_area_retention": 0.85,
        "median_late_retention": 0.8,
        "median_waveform_correlation": 0.95,
        "quiet_rms_ratio": 1.1,
        "artifact_dynamics_ratio": 0.7,
        "active_unlabeled_dynamics_ratio": 1.0,
    }
    gates = InnovationGridConfig.load(EXAMPLE).screening["hard_gates"]
    score, passed = _score(metrics, gates)
    assert np.isfinite(score)
    assert passed
    metrics["artifact_dynamics_ratio"] = 1.5
    _, passed = _score(metrics, gates)
    assert not passed
