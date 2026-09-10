import numpy as np
import pytest

from neurobench.experiments.ica_whitening_evaluation.operators import apply_whitening
from neurobench.experiments.ica_whitening_evaluation.sparse_whitening import (
    apply_sparse, apply_sparse_sequential, fit_sparse_sequential,
    fit_sparse_single_stage, whitened_patch_observations,
)
from neurobench.experiments.ica_whitening_evaluation.real_runner import (
    _observations_at_coordinates,
)


@pytest.mark.parametrize("geometry", ["spatial", "temporal", "joint_spatiotemporal"])
@pytest.mark.parametrize("scope", ["global_quiet", "regional_quiet"])
def test_sparse_single_stage_matches_dense_operator(geometry, scope):
    rng = np.random.default_rng(9)
    movie = rng.normal(size=(16, 10, 12)).astype(np.float32)
    specification = {
        "whitening_geometry": geometry, "covariance_scope": scope,
        "causality": "centered", "spatial_width_px": 3,
        "temporal_width_frames": 3, "seed": 7,
        "raw_preserving_blend": .65,
        f"{'joint' if geometry == 'joint_spatiotemporal' else geometry}_exponent": .7,
        f"{'joint' if geometry == 'joint_spatiotemporal' else geometry}_shrinkage": .05,
        f"{'joint' if geometry == 'joint_spatiotemporal' else geometry}_eigen_floor_ratio": 1e-4,
    }
    dense = apply_whitening(movie, 8, specification, maximum_samples=128).output
    operator = fit_sparse_single_stage(movie, 8, specification, maximum_samples=128)
    t = np.asarray([0, 5, 15, 7]); y = np.asarray([0, 4, 9, 5]); x = np.asarray([0, 6, 11, 5])
    sparse = apply_sparse(movie, t, y, x, operator)
    np.testing.assert_allclose(sparse, dense[t, y, x], rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("geometry", ["spatial_then_temporal", "temporal_then_spatial"])
@pytest.mark.parametrize("scope", ["global_quiet", "regional_quiet"])
def test_sparse_sequential_matches_dense_operator(geometry, scope):
    rng = np.random.default_rng(11)
    movie = rng.normal(size=(16, 10, 12)).astype(np.float32)
    specification = {
        "whitening_geometry": geometry, "covariance_scope": scope,
        "causality": "centered", "spatial_width_px": 3,
        "temporal_width_frames": 3, "seed": 5,
        "raw_preserving_blend": .6,
        "spatial_exponent": .55, "spatial_shrinkage": .08,
        "spatial_eigen_floor_ratio": 1e-4,
        "temporal_exponent": .75, "temporal_shrinkage": .03,
        "temporal_eigen_floor_ratio": 1e-4,
    }
    dense = apply_whitening(movie, 8, specification, maximum_samples=128).output
    operator = fit_sparse_sequential(movie, 8, specification, maximum_samples=128)
    t = np.asarray([0, 5, 15, 7]); y = np.asarray([0, 4, 9, 5]); x = np.asarray([0, 6, 11, 5])
    sparse = apply_sparse_sequential(movie, t, y, x, operator)
    np.testing.assert_allclose(sparse, dense[t, y, x], rtol=3e-5, atol=3e-5)


def test_deduplicated_whitened_patches_match_dense_patch_extraction():
    rng = np.random.default_rng(13)
    movie = rng.normal(size=(12, 9, 10)).astype(np.float32)
    specification = {
        "family": "joint_spatiotemporal", "whitening_geometry": "spatial",
        "covariance_scope": "regional_quiet", "causality": "centered",
        "spatial_width_px": 3, "temporal_width_frames": 3, "seed": 3,
        "raw_preserving_blend": .7, "spatial_exponent": .6,
        "spatial_shrinkage": .04, "spatial_eigen_floor_ratio": 1e-4,
    }
    dense = apply_whitening(movie, 6, specification, maximum_samples=96).output
    operator = fit_sparse_single_stage(movie, 6, specification, maximum_samples=96)
    times = np.asarray([0, 4, 4, 11]); rows = np.asarray([0, 4, 4, 8]); columns = np.asarray([0, 5, 5, 9])
    expected = _observations_at_coordinates(dense, times, rows, columns, specification)
    observed = whitened_patch_observations(
        movie, times, rows, columns, specification, operator, center_chunk_size=4
    )
    np.testing.assert_allclose(observed, expected, rtol=3e-5, atol=3e-5)
