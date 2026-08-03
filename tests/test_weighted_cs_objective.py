import numpy as np
import pytest

from neurobench.algorithms.pairwise_separation import (
    cs_parzen_objective,
    fit_cs_parzen_ica,
)


def test_weighted_objective_uniform_scale_and_block_parity():
    rng = np.random.default_rng(17)
    values = rng.normal(size=(31, 2))
    unit = cs_parzen_objective(values, 0.35, block_rows=7)
    ones = cs_parzen_objective(
        values, 0.35, weights=np.ones(31), block_rows=11
    )
    scaled = cs_parzen_objective(
        values, 0.35, weights=np.linspace(1, 2, 31) * 19, block_rows=5
    )
    reference = cs_parzen_objective(
        values, 0.35, weights=np.linspace(1, 2, 31), block_rows=64
    )
    assert unit.objective == pytest.approx(ones.objective, abs=1e-12)
    assert scaled.objective == pytest.approx(reference.objective, abs=1e-12)


def test_weighted_objective_matches_integer_repetition_and_zero_exclusion():
    rng = np.random.default_rng(23)
    values = rng.normal(size=(9, 2))
    weights = np.asarray([1, 2, 3, 1, 2, 1, 4, 1, 2], dtype=float)
    weighted = cs_parzen_objective(
        values, 0.35, weights=weights, block_rows=3
    )
    repeated = cs_parzen_objective(
        np.repeat(values, weights.astype(int), axis=0),
        0.35,
        block_rows=4,
    )
    padded = cs_parzen_objective(
        np.vstack((values, [[1000, -1000]])),
        0.35,
        weights=np.r_[weights, 0],
        block_rows=2,
    )
    assert weighted.objective == pytest.approx(repeated.objective, abs=1e-12)
    assert padded.objective == pytest.approx(weighted.objective, abs=1e-12)


@pytest.mark.parametrize(
    "weights",
    [
        np.asarray([1.0, -1.0]),
        np.asarray([1.0, np.nan]),
        np.asarray([1.0, np.inf]),
        np.asarray([0.0, 0.0]),
    ],
)
def test_weighted_objective_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        cs_parzen_objective(np.asarray([[0.0, 1.0], [1.0, 0.0]]), 0.35, weights=weights)


def test_weighted_rotation_search_is_periodic_and_deterministic():
    rng = np.random.default_rng(31)
    samples = rng.laplace(size=(2, 80))
    weights = np.linspace(0.5, 1.5, 80)
    first = fit_cs_parzen_ica(
        samples[:, :40],
        samples,
        screen_weights=weights[:40],
        confirm_weights=weights,
        screen_step_degrees=15,
        refine_half_width_degrees=1,
        refine_step_degrees=0.5,
        block_rows=16,
    )
    second = fit_cs_parzen_ica(
        samples[:, :40],
        samples,
        screen_weights=weights[:40] * 7,
        confirm_weights=weights * 7,
        screen_step_degrees=15,
        refine_half_width_degrees=1,
        refine_step_degrees=0.5,
        block_rows=16,
    )
    assert first.diagnostics["selected_angle_degrees"] == second.diagnostics["selected_angle_degrees"]
