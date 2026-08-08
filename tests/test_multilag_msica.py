import numpy as np
import pytest

from neurobench.algorithms.multilag_msica import (
    fit_delay_embedding,
    fit_multilag_2d,
    gather_delay_embedding,
    gather_multilag_pairs,
    lag_weights,
    matrix_renyi_mutual_information,
    project_temporal_fit,
    sample_anchor_indices,
)


def _movie(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(90, 7, 9)).astype(np.float32)


def test_lag_weights_and_gathering_contracts() -> None:
    movie = np.arange(12 * 2 * 3, dtype=np.float32).reshape(12, 2, 3)
    anchors = np.asarray([[8, 1, 2], [10, 0, 1]], dtype=np.int32)
    lags = (0, 1, 2, 4)
    weights = lag_weights(lags, 0.8)
    pairs = gather_multilag_pairs(movie, anchors, lags)
    embedding = gather_delay_embedding(movie, anchors, lags)
    assert pairs.shape == (4, 2, 2)
    assert embedding.shape == (4, 2)
    assert weights.sum() == pytest.approx(1.0)
    assert pairs[0, 0].tolist() == [movie[7, 1, 2], movie[8, 1, 2]]
    assert embedding[:, 0].tolist() == [movie[8, 1, 2], movie[7, 1, 2], movie[6, 1, 2], movie[4, 1, 2]]


def test_anchor_sampling_is_bounded_and_deterministic() -> None:
    first = sample_anchor_indices((30, 5, 6), history=8, count=100, seed=11)
    second = sample_anchor_indices((30, 5, 6), history=8, count=100, seed=11)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (100, 3)
    assert first[:, 0].min() >= 8
    assert first[:, 0].max() < 30


def test_matrix_renyi_detects_stronger_dependence() -> None:
    rng = np.random.default_rng(19)
    left = rng.normal(size=96)
    related = left + 0.05 * rng.normal(size=96)
    unrelated = rng.normal(size=96)
    dependent = matrix_renyi_mutual_information(left, related, alpha=1.5, bandwidth_scale=1.0)
    independent = matrix_renyi_mutual_information(left, unrelated, alpha=1.5, bandwidth_scale=1.0)
    assert dependent > independent
    assert independent >= 0.0


def test_multilag_fit_and_projection_preserve_pair_order() -> None:
    movie = _movie()
    anchors = sample_anchor_indices(movie.shape, history=5, count=160, seed=3)
    pairs = gather_multilag_pairs(movie, anchors, (0, 1, 2, 4))
    fit = fit_multilag_2d(
        pairs[:, :80],
        pairs[:, 80:],
        lags=(0, 1, 2, 4),
        weights=lag_weights((0, 1, 2, 4), 0.85),
        objective_family="normalized_hsic",
        objective_parameter={"bandwidth_scale": 1.0},
        coarse_step_degrees=15.0,
        refine_half_width_degrees=5.0,
        refine_step_degrees=2.5,
    )
    outputs = project_temporal_fit(movie, fit)
    direct = fit.demixing @ (
        np.stack((movie[:-1], movie[1:]), axis=0).reshape(2, -1)
        - fit.center[:, None]
    )
    np.testing.assert_allclose(
        outputs["persistence"].ravel(),
        direct[fit.persistence_index],
        rtol=2e-5,
        atol=2e-5,
    )
    assert outputs["innovation"].shape == (89, 7, 9)
    assert np.isfinite(fit.objective)


def test_delay_embedding_uses_held_out_objective_and_projects_residual() -> None:
    movie = _movie(13)
    anchors = sample_anchor_indices(movie.shape, history=8, count=160, seed=5)
    embedded = gather_delay_embedding(movie, anchors, (0, 1, 2, 4, 8))
    fit = fit_delay_embedding(
        embedded[:, :80],
        embedded[:, 80:],
        lags=(0, 1, 2, 4, 8),
        objective_family="ksg_mi",
        objective_parameter={"neighbors": 3},
        angle_step_degrees=30.0,
        max_sweeps=2,
    )
    outputs = project_temporal_fit(movie, fit)
    assert outputs["persistence"].shape == (82, 7, 9)
    assert outputs["innovation"].shape == (82, 7, 9)
    assert outputs["residual_group"].shape == (82, 7, 9)
    assert fit.diagnostics["confirmation_samples"] == 80
    assert len(fit.residual_indices) == 3
    assert np.isfinite(fit.objective)
