import numpy as np

from neurobench.algorithms.representation_benchmark import (
    matched_component_stability,
    orient_components,
    reconstruction,
    symmetric_fastica,
    truncated_pca,
    whiten_spatial_scores,
)


def test_pca_reconstructs_low_rank_matrix() -> None:
    rng = np.random.default_rng(4)
    spatial = rng.normal(size=(80, 3))
    temporal = rng.normal(size=(3, 24))
    values = spatial @ temporal
    fit = truncated_pca(values, 3)
    restored = reconstruction(fit.spatial_scores, fit.temporal_basis)
    assert np.linalg.norm(values - restored) / np.linalg.norm(values) < 1e-6
    assert np.isclose(fit.explained_energy_ratio.sum(), 1.0)


def test_fastica_returns_compatible_factors() -> None:
    rng = np.random.default_rng(5)
    independent = np.column_stack((
        rng.laplace(size=3000),
        rng.uniform(-2, 2, size=3000),
        rng.standard_t(4, size=3000),
    ))
    mixing = rng.normal(size=(3, 3))
    values = independent @ mixing
    pca = truncated_pca(values, 3)
    whitened, scale = whiten_spatial_scores(pca.spatial_scores)
    fit = symmetric_fastica(
        whitened, pca.temporal_basis, scale, seed=7,
        max_iterations=500, tolerance=1e-5,
    )
    assert fit.spatial_sources.shape == (3000, 3)
    assert fit.temporal_traces.shape == (3, 3)
    assert fit.converged
    assert np.isfinite(fit.spatial_sources).all()


def test_orientation_and_stability_are_sign_invariant() -> None:
    spatial = np.array([[1, -2], [2, -1], [3, 0]], dtype=np.float32)
    temporal = np.array([
        [0, 0, 1, 2, 2],
        [0, 0, -1, -3, -2],
    ], dtype=np.float32)
    oriented_spatial, oriented_temporal, signs = orient_components(
        spatial, temporal, 2, [(2, 5)]
    )
    assert signs.tolist() == [1.0, -1.0]
    assert oriented_temporal[:, 2:].mean(axis=1).min() > 0
    stability = matched_component_stability(
        oriented_spatial, oriented_spatial[:, ::-1] * np.array([-1, 1])
    )
    assert stability["mean_absolute_correlation"] > 0.999
