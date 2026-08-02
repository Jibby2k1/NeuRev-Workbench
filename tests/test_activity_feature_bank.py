from __future__ import annotations

import numpy as np

from neurobench.algorithms.activity_feature_bank import (
    bounded_square,
    cross_scale_consensus_score,
    derivative_feature_iterator,
    morphology_feature_iterator,
    persistence_features,
)


def _fixture() -> np.ndarray:
    rng = np.random.default_rng(17)
    video = rng.normal(0, 0.05, size=(24, 16, 16)).astype(np.float32)
    video[12:16, 7:10, 7:10] += np.asarray([0.5, 1.0, 0.7, 0.2])[:, None, None]
    return video


def test_bounded_square_is_even_and_bounded() -> None:
    values = np.asarray([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    result = bounded_square(values, 1.0)
    assert np.allclose(result, result[::-1])
    assert np.all((0 <= result) & (result < 1))


def test_derivative_bank_has_frozen_ids_and_unit_ranges() -> None:
    rows = list(
        derivative_feature_iterator(
            _fixture(),
            quiet_count=10,
            spatial_sigma_px=1.0,
            lags=(1, 2, 4),
            clip_z=6.0,
            power=1.5,
            energy_tau_z=1.0,
            huber_delta_z=1.0,
        )
    )
    assert [row[0] for row in rows] == [
        "derivative_positive_lag1",
        "derivative_negative_lag1",
        "derivative_absolute_lag1",
        "derivative_power1p5_lag1",
        "derivative_log_square_lag1",
        "derivative_huber_lag1",
        "derivative_square_lag1",
        "derivative_square_lag2",
        "derivative_square_lag4",
    ]
    for _, values, _ in rows:
        assert values.shape == (24, 16, 16)
        assert np.isfinite(values).all()
        assert np.all((0 <= values) & (values <= 1))


def test_persistence_features_are_causal() -> None:
    video = _fixture()
    prefix = persistence_features(
        video[:18],
        frame_period_ms=20,
        persistence_half_life_seconds=2,
        dynamic_half_life_seconds=0.5,
        energy_tau_z=1,
    )
    complete = persistence_features(
        video,
        frame_period_ms=20,
        persistence_half_life_seconds=2,
        dynamic_half_life_seconds=0.5,
        energy_tau_z=1,
    )
    assert set(complete) == {
        "persistence_activity_gate",
        "persistent_artifact_score",
    }
    for key in complete:
        assert np.allclose(prefix[key], complete[key][:18])
        assert np.all((0 <= complete[key]) & (complete[key] <= 1))


def test_spatial_feature_families_are_finite_and_bounded() -> None:
    video = _fixture()
    consensus = cross_scale_consensus_score(
        video,
        spatial_scales_px=(1, 2, 4),
        agreement_power=4,
        evidence_threshold_z=1.5,
    )
    assert consensus.shape == video.shape
    assert np.all((0 <= consensus) & (consensus <= 1))
    morphology = list(
        morphology_feature_iterator(
            video,
            quiet_count=10,
            center_sigma_px=1,
            ring_sigma_px=3.5,
            crowd_sigma_px=7,
            clip_z=6,
        )
    )
    assert {row[0] for row in morphology} == {
        "morphology_center_isolated",
        "morphology_membrane_isolated",
        "morphology_center_crowded",
        "morphology_membrane_crowded",
    }
    for _, values, _ in morphology:
        assert np.isfinite(values).all()
        assert np.all((0 <= values) & (values <= 1))
