import numpy as np

from neurobench.algorithms.multiscale_subspace import (
    apply_per_context_fit,
    fit_cross_context,
    fit_per_context_ica,
    predeclared_group_energy,
)


def _mixture(seed: int = 8, samples: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    persistence = rng.laplace(size=samples)
    innovation = rng.standard_t(4, size=samples)
    return np.column_stack(
        [persistence - 0.45 * innovation, persistence + 0.45 * innovation]
    )


def test_per_context_fit_is_deterministic_and_canonically_ordered() -> None:
    pairs = _mixture()
    first = fit_per_context_ica(
        "spatial_5_meanstd", pairs[:256], pairs[256:], kernel_block_rows=64
    )
    second = fit_per_context_ica(
        "spatial_5_meanstd", pairs[:256], pairs[256:], kernel_block_rows=64
    )
    np.testing.assert_allclose(first.demixing, second.demixing)
    output = apply_per_context_fit(pairs, first)
    assert output.shape == pairs.shape
    assert first.derivative_angle_distance_degrees <= 45.0


def test_cross_context_identity_and_full_rank_pca_reconstruct() -> None:
    values = np.random.default_rng(2).normal(size=(100, 4))
    for mode in ("identity", "pca"):
        fit = fit_cross_context(values, mode=mode, max_components=4)
        reconstructed = fit.inverse_values(fit.transform_values(values))
        np.testing.assert_allclose(reconstructed, values, atol=1e-5)


def test_predeclared_group_energy_uses_fixed_indices() -> None:
    values = np.asarray([[1.0, 2.0, 3.0], [2.0, 0.0, 4.0]])
    result = predeclared_group_energy(values, {"compact": [0, 1]})
    np.testing.assert_allclose(result["compact"], [5.0, 4.0])
