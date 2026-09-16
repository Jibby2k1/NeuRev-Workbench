"""Meaningful geometry contracts; no movie processing or detector outcomes."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neurobench.algorithms.gamma_spatiotemporal import GammaSTSpec, build_kernels
from neurobench.experiments.gamma_ls_difference.reference_geometry import (
    MINIMUM_RETAINED_MASS, mean_matched_reference, reference_grid,
)


@pytest.fixture(scope="module")
def grid():
    return reference_grid()


def test_central_anchor_is_exact_without_mutating_supplied_array():
    original = build_kernels(GammaSTSpec()).reference[0].copy()
    before = original.tobytes()
    result = reference_grid(original)
    central = next(k for k in result if k.spec.preserved_anchor)
    assert central.spec.spec_id == "mean1_n9"
    assert central.weights.tobytes() == before == original.tobytes()
    assert original.flags.writeable
    assert not np.shares_memory(central.weights, original)
    with pytest.raises(ValueError):
        central.weights[0, 0] = 1
    with pytest.raises(FrozenInstanceError):
        central.spec.reference_n = 3
    original[0, 0] += 1e-4
    assert central.weights.tobytes() == before
    with pytest.raises(ValueError, match="Anchor bytes differ"):
        reference_grid(original)


def test_grid_mass_support_symmetry_and_no_explicit_mask(grid):
    assert len(grid) == 9 and len({k.spec.spec_id for k in grid}) == 9
    for kernel in grid:
        w = kernel.weights
        half = kernel.spec.spatial_half_width_px
        assert w.dtype == np.float64 and w.shape == (2 * half + 1,) * 2
        assert np.isfinite(w).all() and (w >= 0).all()
        assert w.sum() == pytest.approx(1, abs=2e-14)
        assert np.count_nonzero(w == 0) == 1 and w[half, half] == 0
        np.testing.assert_array_equal(w, w[::-1])
        np.testing.assert_array_equal(w, w[:, ::-1])
        assert half + 4 <= 49
        assert kernel.metadata["retention"]["discrete_retained_mass_estimate"] >= MINIMUM_RETAINED_MASS
        assert kernel.metadata["retention"]["relative_convergence_mass_gap"] <= 1e-7
        assert not w.flags.writeable


def test_mean_location_is_matched_while_concentration_changes(grid):
    anchor = next(k for k in grid if k.spec.preserved_anchor)
    mean = anchor.metadata["realized_mean_radius_px"]
    for factor in (2 / 3, 1, 4 / 3):
        row = sorted((k for k in grid if k.spec.mean_factor == factor), key=lambda k: k.spec.reference_n)
        assert [k.spec.reference_n for k in row] == [3, 9, 15]
        for k in row:
            assert k.metadata["realized_mean_radius_px"] == pytest.approx(factor * mean, abs=2e-11)
            assert k.spec.profile_mode_radius_px == (k.spec.reference_n - 1) / k.spec.reference_mu_per_px
        assert row[0].metadata["realized_radius_sd_px"] > row[1].metadata["realized_radius_sd_px"] > row[2].metadata["realized_radius_sd_px"]


def test_fixed_support_expansion_resolves_mean_again_instead_of_cropping():
    result = mean_matched_reference(9.0, 3, initial_half_width_px=8)
    history = result.metadata["fixed_support_solve_history"]
    assert len(history) > 1 and history[-1]["status"] == "accepted"
    assert all(a["half_width_px"] < b["half_width_px"] for a, b in zip(history, history[1:]))
    assert result.metadata["realized_mean_radius_px"] == pytest.approx(9, abs=2e-11)
    mus = [h["reference_mu_per_px"] for h in history if "reference_mu_per_px" in h]
    assert len(set(mus)) > 1
    with pytest.raises(ValueError, match="exceeds the fixed"):
        mean_matched_reference(9, 3, initial_half_width_px=8, max_half_width_px=8)


def test_exact_radial_quantiles_and_overlap_measures_match_brute_grid(grid):
    k = next(k for k in grid if k.spec.preserved_anchor)
    w, h = k.weights, k.spec.spatial_half_width_px
    yy, xx = np.meshgrid(np.arange(-h, h + 1), np.arange(-h, h + 1), indexing="ij")
    r = np.hypot(xx, yy)
    for key, value in k.metadata["radial_mass_quantiles_px"].items():
        q = int(key[1:]) / 100
        assert w[r <= value].sum() >= q - 1e-14
        assert w[r < value].sum() < q + 1e-14
    for entry in k.metadata["source_overlap"]:
        mask = (xx - entry["offset_x_px"]) ** 2 + yy ** 2 <= 4
        assert entry["reference_mass_in_disk"] == pytest.approx(w[mask].sum(), abs=1e-15)
        assert 0 <= entry["conditioned_source_reference_mean_gain"] <= 1
        assert entry["conditioned_source_reference_variance_gain"] >= 0
    assert k.metadata["effective_sample_count"] == pytest.approx(1 / (w * w).sum())


@pytest.mark.parametrize("mean,n", [(1, 3), (0, 9), (np.nan, 9), (9, 1), (9, np.inf)])
def test_invalid_mean_or_order_is_rejected(mean, n):
    with pytest.raises(ValueError):
        mean_matched_reference(mean, n)


def test_no_silent_reconfiguration_or_shared_array_mutation(grid):
    with pytest.raises(ValueError, match="Maximum halfwidth"):
        mean_matched_reference(9, 3, max_half_width_px=46)
    with pytest.raises(ValueError, match="Initial halfwidth"):
        mean_matched_reference(9, 3, initial_half_width_px=1.5)
    first = grid[0]
    fresh = mean_matched_reference(first.spec.requested_mean_radius_px, first.spec.reference_n)
    assert fresh.weights.tobytes() == first.weights.tobytes()
    assert not np.shares_memory(fresh.weights, first.weights)
    assert fresh.metadata["source_overlap"] is not first.metadata["source_overlap"]
