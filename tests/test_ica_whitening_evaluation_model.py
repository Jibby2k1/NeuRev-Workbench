import numpy as np
import pytest

from neurobench.experiments.ica_whitening_evaluation.model import (
    activity_priority_order,
    extract_patch_observations,
    fit_ica,
)
from neurobench.experiments.ica_whitening_evaluation.responses import (
    component_response_summary,
)


def test_all_model_geometries_fit_and_export_responses() -> None:
    movie = np.random.default_rng(12).standard_t(4, size=(24, 9, 10)).astype(np.float32)
    for family in ("temporal", "spatial", "joint_spatiotemporal"):
        patches = extract_patch_observations(
            movie, family=family, spatial_width=3, temporal_width=3,
            causality="causal", maximum_samples=256, seed=5,
        )
        result = fit_ica(
            patches.values,
            {"objective": "fastica_logcosh", "rank": 2, "seed": 5, "objective_scale": 1.0},
            maximum_fit_samples=256,
        )
        assert result.sources.shape == (2, 256)
        assert np.isfinite(result.sources).all()
        response = component_response_summary(
            result.demixing, family=family, spatial_width=3,
            temporal_width=3, frame_period_ms=20.0,
        )
        assert len(response["components"]) == 2


def test_hsic_reference_is_bounded_and_finite() -> None:
    observations = np.random.default_rng(4).standard_t(5, size=(3, 80))
    result = fit_ica(
        observations,
        {"objective": "hsic_pairwise", "rank": 2, "seed": 7, "objective_scale": 1.0},
        maximum_fit_samples=80,
    )
    assert np.isfinite(result.sources).all()


def test_shared_activity_order_preserves_exact_patch_selection() -> None:
    movie = np.random.default_rng(17).normal(size=(18, 8, 9)).astype(np.float32)
    kwargs = dict(
        family="joint_spatiotemporal", spatial_width=3, temporal_width=3,
        causality="centered", maximum_samples=128, seed=9,
        quiet_frames=6, activity_fraction=.5,
    )
    direct = extract_patch_observations(movie, **kwargs)
    cached = extract_patch_observations(
        movie, **kwargs, activity_order=activity_priority_order(movie, 6)
    )
    np.testing.assert_array_equal(cached.times, direct.times)
    np.testing.assert_array_equal(cached.rows, direct.rows)
    np.testing.assert_array_equal(cached.columns, direct.columns)
    np.testing.assert_array_equal(cached.values, direct.values)


def test_causal_temporal_observations_allow_even_support() -> None:
    movie = np.arange(7 * 5 * 5, dtype=np.float32).reshape(7, 5, 5)
    observations = extract_patch_observations(
        movie, family="temporal", spatial_width=None, temporal_width=2,
        causality="causal", maximum_samples=32, seed=3,
    )
    assert observations.values.shape == (2, 32)


def test_centered_temporal_observations_reject_even_support() -> None:
    movie = np.arange(7 * 5 * 5, dtype=np.float32).reshape(7, 5, 5)
    with pytest.raises(ValueError, match="centered"):
        extract_patch_observations(
            movie, family="temporal", spatial_width=None, temporal_width=2,
            causality="centered", maximum_samples=32, seed=3,
        )
