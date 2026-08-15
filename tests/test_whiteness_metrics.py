import numpy as np
import pytest

from neurobench.metrics.whiteness import (
    covariance_identity_error,
    effective_rank,
    fit_holdout_covariance_error,
    maximum_absolute_correlation,
    normalized_off_diagonal_energy,
    temporal_autocorrelation_summary,
    tile_boundary_discontinuity,
)


def test_analytical_covariance_metrics() -> None:
    identity = np.eye(3)
    assert covariance_identity_error(identity) == pytest.approx(0.0)
    assert normalized_off_diagonal_energy(identity) == pytest.approx(0.0)
    assert maximum_absolute_correlation(identity) == pytest.approx(0.0)
    covariance = np.asarray([[4.0, 1.0], [1.0, 1.0]])
    assert maximum_absolute_correlation(covariance) == pytest.approx(0.5)
    assert normalized_off_diagonal_energy(covariance) == pytest.approx(np.sqrt(2 / 19))
    assert effective_rank(np.ones(4)) == pytest.approx(4.0)


def test_fit_holdout_error_is_zero_for_matching_covariance() -> None:
    covariance = np.asarray([[2.0, 0.6], [0.6, 1.0]])
    assert fit_holdout_covariance_error(covariance, covariance) <= 1e-12


def test_temporal_autocorrelation_detects_recurrence() -> None:
    values = np.arange(20, dtype=np.float64)[:, None]
    summary = temporal_autocorrelation_summary(values, 2)
    assert summary["lags"][0]["mean_correlation"] == pytest.approx(1.0)
    assert summary["coordinate_count"] == 1


def test_tile_boundary_metric_detects_added_seam() -> None:
    smooth = np.broadcast_to(np.arange(8, dtype=np.float64), (8, 8)).copy()
    seamed = smooth.copy(); seamed[:, 4:] += 20
    geometry = [(0, 8, 0, 4), (0, 8, 4, 8)]
    left = tile_boundary_discontinuity(smooth, geometry)
    right = tile_boundary_discontinuity(seamed, geometry)
    assert right["boundary_mean_absolute_jump"] > left["boundary_mean_absolute_jump"]
    assert right["boundary_to_all_ratio"] > 1
