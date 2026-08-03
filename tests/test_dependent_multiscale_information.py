import numpy as np

from neurobench.algorithms.dependent_multiscale import PatchDecomposition
from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_information import (
    build_frame_nuisance,
    matrix_renyi_entropy,
    matrix_renyi_group_dependence,
    normalized_gaussian_gram,
    refine_group_dependence,
)


def test_matrix_renyi_dependence_distinguishes_dependent_groups():
    rng = np.random.default_rng(20)
    left = rng.normal(size=(300, 3))
    dependent = left @ np.array([[1.0, 0.2], [0.0, 0.8], [0.3, 0.1]]) + 0.05 * rng.normal(size=(300, 2))
    independent = rng.normal(size=(300, 2))
    related = matrix_renyi_group_dependence(left, dependent, maximum_samples=300)
    unrelated = matrix_renyi_group_dependence(left, independent, maximum_samples=300)
    assert related["dependence"] > unrelated["dependence"]
    gram, _ = normalized_gaussian_gram(left)
    assert np.isfinite(matrix_renyi_entropy(gram))
    np.testing.assert_allclose(np.trace(gram), 1.0, atol=1e-12)


def test_nuisance_residualization_removes_explained_dependence():
    rng = np.random.default_rng(21)
    nuisance = rng.normal(size=(240, 1))
    first = nuisance @ np.array([[2.0, -1.0]]) + 0.2 * rng.normal(size=(240, 2))
    second = nuisance @ np.array([[1.5, 0.5]]) + 0.2 * rng.normal(size=(240, 2))
    unconditional = matrix_renyi_group_dependence(first, second)
    conditional = matrix_renyi_group_dependence(first, second, nuisance=nuisance)
    assert conditional["dependence"] < unconditional["dependence"]


def test_refinement_closes_and_never_penalizes_within_neural_dependence():
    rng = np.random.default_rng(22)
    observation = rng.normal(size=(18, 9, 9))
    background = np.broadcast_to(np.linspace(-1, 1, 18)[:, None, None], observation.shape) * 0.2
    artifact = np.zeros_like(observation)
    signal = observation - background
    noise = np.zeros_like(observation)
    baseline = PatchDecomposition(
        "p", background, signal, artifact, noise, np.zeros_like(observation), None, {}
    )
    nuisance, names = build_frame_nuisance(observation)
    result = refine_group_dependence(
        baseline, observation=observation, nuisance=nuisance, authority=0.5,
    )
    restored = (
        result.decomposition.background + result.decomposition.structured_signal
        + result.decomposition.structured_artifact + result.decomposition.noise_candidate
        + result.decomposition.closure_residual
    )
    np.testing.assert_allclose(restored, observation, atol=2e-6)
    assert names == ("global_intensity", "slow_drift")
    assert not result.diagnostics["within_neural_dependence_penalized"]
