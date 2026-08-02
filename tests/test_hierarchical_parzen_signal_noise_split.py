from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neurobench.experiments.hierarchical_parzen_ica.signal_noise_config import (
    SignalNoiseConfig,
    SignalNoiseConfigError,
)
from neurobench.experiments.hierarchical_parzen_ica.signal_noise_split import (
    SCIENTIFIC_STATUS,
    _apply_lookup,
    _lookup,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/spon_ca_burst_noisy_parzen_signal_split.example.json"


def test_signal_noise_example_loads_bounded_grid() -> None:
    config = SignalNoiseConfig.load(EXAMPLE)
    assert config.input_lane["id"] == "reference_parzen_innovation"
    assert len(config.posterior["bandwidths"]) * len(
        config.posterior["noise_variance_multipliers"]
    ) == 16
    assert "not_complete_patchwise_noisy_ica" in SCIENTIFIC_STATUS


def test_unknown_signal_noise_field_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["posterior"]["classify_noise_as_truth"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SignalNoiseConfigError, match="fields differ"):
        SignalNoiseConfig.load(path)


def test_lookup_split_is_finite_monotone_and_exactly_closes() -> None:
    config = SignalNoiseConfig.load(EXAMPLE)
    centers = np.asarray([0, 0, 0, -4, -2, 2, 4], dtype=np.float64)
    grid, posterior = _lookup(centers, 0.5, 1.0, config)
    assert np.isfinite(posterior).all()
    assert np.all(np.diff(posterior) >= -1e-10)
    observed = np.linspace(-8, 8, 1001, dtype=np.float32)
    signal = _apply_lookup(observed, grid, posterior)
    noise = observed - signal
    np.testing.assert_allclose(signal + noise, observed, rtol=0, atol=1e-6)
    assert np.mean(np.abs(signal)) < np.mean(np.abs(observed))
