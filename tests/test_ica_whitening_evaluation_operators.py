import numpy as np

from neurobench.experiments.ica_whitening_evaluation.operators import (
    apply_whitening,
    deterministic_joint_samples,
)


def _specification(geometry: str, scope: str = "global_quiet") -> dict:
    return {
        "whitening_geometry": geometry,
        "covariance_scope": scope,
        "causality": "causal",
        "spatial_width_px": 3,
        "temporal_width_frames": 3,
        "spatial_exponent": 0.7,
        "spatial_shrinkage": 0.1,
        "spatial_eigen_floor_ratio": 1e-4,
        "temporal_exponent": 0.6,
        "temporal_shrinkage": 0.1,
        "temporal_eigen_floor_ratio": 1e-4,
        "joint_exponent": 0.8,
        "joint_shrinkage": 0.1,
        "joint_eigen_floor_ratio": 1e-4,
        "raw_preserving_blend": 1.0,
        "seed": 7,
    }


def test_joint_samples_and_all_whitening_geometries_are_finite() -> None:
    rng = np.random.default_rng(9)
    movie = rng.normal(size=(18, 10, 12)).astype(np.float32)
    samples = deterministic_joint_samples(
        movie[:8], 3, 3, causality="causal", maximum_samples=100, seed=2
    )
    assert samples.shape == (100, 27)
    for geometry in (
        "none", "spatial", "temporal", "spatial_then_temporal",
        "temporal_then_spatial", "joint_spatiotemporal",
    ):
        result = apply_whitening(
            movie, 8, _specification(geometry), maximum_samples=200
        )
        assert result.output.shape == movie.shape
        assert np.isfinite(result.output).all()
        assert result.diagnostics["resolved"]


def test_regional_scope_fits_four_distinct_regions() -> None:
    movie = np.random.default_rng(4).normal(size=(16, 12, 14)).astype(np.float32)
    result = apply_whitening(
        movie, 8, _specification("spatial", "regional_quiet"),
        maximum_samples=128,
    )
    assert len(result.diagnostics["region_fits"]) == 4


def test_zero_raw_blend_is_exact_identity() -> None:
    movie = np.random.default_rng(5).normal(size=(16, 10, 10)).astype(np.float32)
    specification = _specification("joint_spatiotemporal")
    specification["raw_preserving_blend"] = 0.0
    result = apply_whitening(movie, 8, specification, maximum_samples=128)
    assert np.array_equal(result.output, movie)
