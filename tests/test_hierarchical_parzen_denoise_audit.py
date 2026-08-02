from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neurobench.experiments.hierarchical_parzen_ica.denoise_audit import VARIANT_IDS
from neurobench.experiments.hierarchical_parzen_ica.denoise_audit_config import (
    DenoiseAuditConfig,
    DenoiseAuditConfigError,
)
from neurobench.experiments.hierarchical_parzen_ica.denoise_methods import (
    causal_kalman,
    frame_gamma,
    local_low_rank,
    robust_gamma,
    savgol_signal,
    spatial_evidence_gate,
    temporal_evidence_gate,
    undecimated_haar_like,
    quiet_wiener,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/spon_ca_burst_sequential_denoise_audit.example.json"


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    values = rng.normal(0, 0.2, (24, 8, 8)).astype(np.float32)
    values[8:16, 3:5, 3:5] += np.linspace(0, 2, 8)[:, None, None]
    scale = np.full((8, 8), 0.2, dtype=np.float32)
    return values, scale


def test_example_declares_all_sequential_variants() -> None:
    config = DenoiseAuditConfig.load(EXAMPLE)
    assert config.variant_count == len(VARIANT_IDS) == 11
    assert config.input_lane["id"] == "reference_parzen_innovation"
    assert set(config.methods) == {
        "pointwise",
        "spatial_gate",
        "temporal_gate",
        "temporal_filters",
        "local_pca",
        "noise_normalized_pca",
        "component_parzen",
    }


def test_unknown_top_level_field_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["overwrite"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DenoiseAuditConfigError, match="top-level fields differ"):
        DenoiseAuditConfig.load(path)


def test_dense_methods_are_finite_and_preserve_shape() -> None:
    values, scale = _fixture()
    outputs = [
        frame_gamma(values, 2),
        robust_gamma(values, 2, 1, 99.5),
        quiet_wiener(values, scale, 1),
        spatial_evidence_gate(
            values, scale, sigma_px=1, lambda_z=1, structural_floor=0.25
        ),
        temporal_evidence_gate(
            values,
            scale,
            sigma_px=1,
            lambda_z=1,
            structural_floor=0.25,
            half_life_ms=80,
            frame_period_ms=20,
        ),
        savgol_signal(values, 7, 2),
        undecimated_haar_like(values, scale, levels=3, threshold_z=1),
        causal_kalman(
            values,
            scale,
            frame_period_ms=20,
            decay_ms=320,
            process_variance=0.08,
            observation_variance=1,
        ),
    ]
    for output in outputs:
        assert output.shape == values.shape
        assert output.dtype == np.float32
        assert np.isfinite(output).all()
        remainder = values - output
        np.testing.assert_allclose(output + remainder, values, rtol=0, atol=1e-6)


def test_causal_filters_do_not_respond_before_impulse() -> None:
    values = np.zeros((24, 8, 8), dtype=np.float32)
    values[12, 4, 4] = 2
    scale = np.ones((8, 8), dtype=np.float32)
    gated = temporal_evidence_gate(
        values,
        scale,
        sigma_px=1,
        lambda_z=1,
        structural_floor=0.25,
        half_life_ms=80,
        frame_period_ms=20,
    )
    kalman = causal_kalman(
        values,
        scale,
        frame_period_ms=20,
        decay_ms=320,
        process_variance=0.08,
        observation_variance=1,
    )
    assert np.count_nonzero(gated[:12]) == 0
    assert np.count_nonzero(kalman[:12]) == 0


@pytest.mark.parametrize("noise_normalized", [False, True])
def test_local_low_rank_is_bounded_and_finite(noise_normalized: bool) -> None:
    values, scale = _fixture()
    output, diagnostics = local_low_rank(
        values,
        scale,
        patch_size=8,
        stride=4,
        rank=3,
        oversample=1,
        batch_size=2,
        device="cpu",
        noise_normalized=noise_normalized,
        quiet_count=8,
    )
    assert output.shape == values.shape
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert diagnostics["patch_count"] == 1
    assert diagnostics["noise_normalized"] is noise_normalized


def test_component_parzen_lane_is_finite() -> None:
    values, scale = _fixture()
    output, diagnostics = local_low_rank(
        values,
        scale,
        patch_size=8,
        stride=4,
        rank=3,
        oversample=1,
        batch_size=2,
        device="cpu",
        noise_normalized=True,
        quiet_count=8,
        component_parzen={
            "ica_iterations": 3,
            "dictionary_centers": 8,
            "dictionary_zero_fraction": 0.5,
            "bandwidth": 0.5,
            "noise_variance": 1,
            "lookup_points": 128,
            "lookup_abs_z": 10,
        },
    )
    assert output.shape == values.shape
    assert np.isfinite(output).all()
    assert diagnostics["component_parzen"] is True
