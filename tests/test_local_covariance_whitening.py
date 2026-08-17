from dataclasses import replace

import numpy as np
import pytest

from neurobench.algorithms.local_covariance_whitening import (
    CovarianceEstimatorConfig,
    SpatialTileConfig,
    WhiteningFeatureBank,
    apply_tiled_feature_whiteners,
    contiguous_quiet_partitions,
    enumerate_spatial_tiles,
    fit_covariance,
    fit_empirical_quiet_calibration,
    fit_tiled_feature_whiteners,
    gather_covariance_samples,
)
from neurobench.metrics.whiteness import covariance_identity_error


def _bank(values: np.ndarray) -> WhiteningFeatureBank:
    dimension = values.shape[-1]
    return WhiteningFeatureBank(
        tuple(f"f{index}" for index in range(dimension)),
        values.astype(np.float32),
        np.ones(len(values), dtype=bool),
        tuple({"scientific_array": "signed_msln"} for _ in range(dimension)),
        {"feature_order_frozen": True},
    )


def test_known_spd_zca_and_mahalanobis_invariance() -> None:
    rng = np.random.default_rng(4)
    covariance = np.asarray([[3.0, 1.2, 0.4], [1.2, 2.0, 0.3], [0.4, 0.3, 1.0]])
    samples = rng.multivariate_normal(np.zeros(3), covariance, size=200_000)
    fit = fit_covariance(samples, CovarianceEstimatorConfig(method="fixed_ridge", ridge_ratio=0))
    assert fit.resolved
    assert covariance_identity_error(fit.whitening @ fit.covariance @ fit.whitening.T) <= 1e-8
    transformed = np.asarray([[2.0, 0.3, 0.1], [0.2, 1.5, 0.4], [0.1, 0.2, 1.2]])
    other = samples @ transformed.T
    other_fit = fit_covariance(other, CovarianceEstimatorConfig(method="fixed_ridge", ridge_ratio=0))
    query = samples[:100] - fit.mean
    other_query = other[:100] - other_fit.mean
    q1 = np.einsum("ni,ij,nj->n", query, np.linalg.inv(fit.covariance), query)
    q2 = np.einsum("ni,ij,nj->n", other_query, np.linalg.inv(other_fit.covariance), other_query)
    np.testing.assert_allclose(q1, q2, rtol=2e-5, atol=2e-5)


def test_diagonal_and_full_controls_remain_distinct() -> None:
    rng = np.random.default_rng(8)
    samples = rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], size=10_000)
    diagonal = fit_covariance(samples, CovarianceEstimatorConfig(method="diagonal"))
    full = fit_covariance(samples, CovarianceEstimatorConfig(method="oas"))
    heldout = rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], size=20_000)
    diagonal_cov = np.cov(((heldout - diagonal.mean) @ diagonal.whitening.T).T, bias=True)
    full_cov = np.cov(((heldout - full.mean) @ full.whitening.T).T, bias=True)
    assert covariance_identity_error(full_cov) < covariance_identity_error(diagonal_cov)


def test_unresolved_fit_is_explicit() -> None:
    fit = fit_covariance(np.zeros((8, 3)), CovarianceEstimatorConfig())
    assert not fit.resolved
    assert fit.unresolved_reason == "degenerate_covariance"
    assert fit.diagnostics["identity_is_placeholder_not_success"] is True


def test_centered_residual_is_a_valid_signed_scientific_feature() -> None:
    values = np.ones((3, 2, 2, 1), dtype=np.float32)
    bank = WhiteningFeatureBank(
        ("residual",), values, np.ones(3, dtype=bool),
        ({"scientific_array": "centered_residual_numerator"},),
        {"feature_order_frozen": True},
    )
    assert bank.source_contexts[0]["scientific_array"] == "centered_residual_numerator"


def test_nonlinear_or_display_feature_is_rejected() -> None:
    values = np.ones((3, 2, 2, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="signed MSLN or centered-residual"):
        WhiteningFeatureBank(
            ("energy",), values, np.ones(3, dtype=bool),
            ({"scientific_array": "mahalanobis_energy"},),
            {"feature_order_frozen": True},
        )


def test_deterministic_sample_gathering_and_contiguous_partitions() -> None:
    values = np.arange(12 * 6 * 6 * 2, dtype=np.float32).reshape(12, 6, 6, 2)
    bank = _bank(values)
    partitions = contiguous_quiet_partitions(np.ones(12, dtype=bool))
    left, diagnostics = gather_covariance_samples(bank, partitions["fit"], maximum_samples=11)
    right, _ = gather_covariance_samples(bank, partitions["fit"], maximum_samples=11)
    np.testing.assert_array_equal(left, right)
    assert diagnostics["selected_sample_count"] == 11
    assert not np.any(partitions["fit"] & partitions["calibration"])
    assert not np.any(partitions["calibration"] & partitions["holdout"])


def test_tile_blending_has_coverage_and_matches_identical_global_transform() -> None:
    rng = np.random.default_rng(12)
    values = rng.normal(size=(12, 8, 8, 2)).astype(np.float32)
    bank = _bank(values)
    quiet = np.zeros(12, dtype=bool); quiet[:6] = True
    config = SpatialTileConfig(6, 6, 3, 3, "hann", "crop", 8, 1000, 1, 1)
    fits = fit_tiled_feature_whiteners(bank, quiet, config, CovarianceEstimatorConfig(method="oas"))
    global_fit = fits.global_full_fit
    identical = replace(
        fits,
        primary_fits=tuple(replace(global_fit, fit_id=f"same_{i}", tile_bounds_yx=bounds) for i, bounds in enumerate(fits.tile_bounds_yx)),
    )
    calibration = np.zeros(12, dtype=bool); calibration[6:9] = True
    result = apply_tiled_feature_whiteners(bank, identical, config, calibration, frame_chunk=3)
    expected = np.einsum("...d,ed->...e", values.astype(np.float64) - global_fit.mean, global_fit.whitening).astype(np.float32)
    np.testing.assert_allclose(result.zca_features, expected, atol=2e-5)
    assert float(np.min(result.blend_weight)) > 0
    assert not result.unresolved_tile_mask.any()


def test_empirical_survival_is_monotone_with_deterministic_ties() -> None:
    energy = np.asarray([0, 1, 1, 2, 3], dtype=np.float32)[:, None, None]
    calibration = fit_empirical_quiet_calibration(energy, np.ones(5, dtype=bool), probability_floor=1e-4)
    probability = calibration.survival_probability(np.asarray([0, 1, 1, 2, 4]))
    assert probability[1] == probability[2]
    assert np.all(np.diff(probability) <= 0)
    assert np.all(np.diff(calibration.surprise(np.asarray([0, 1, 2, 4]))) >= 0)


def test_tile_enumeration_forces_exact_crop_coverage() -> None:
    config = SpatialTileConfig(6, 6, 4, 4, "uniform", "crop", 8, 100, 1, 1)
    tiles = enumerate_spatial_tiles(11, 13, config)
    coverage = np.zeros((11, 13), dtype=int)
    for y0, y1, x0, x1 in tiles:
        coverage[y0:y1, x0:x1] += 1
    assert np.all(coverage > 0)
    assert max(item[1] for item in tiles) == 11
    assert max(item[3] for item in tiles) == 13
