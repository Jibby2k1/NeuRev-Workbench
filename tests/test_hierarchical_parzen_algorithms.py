import numpy as np

from neurobench.algorithms.hierarchical_parzen_ica import (
    ParzenDictionaryConfig,
    center_and_whiten_2d,
    component_derivative_energy,
    decomposition_closure,
    fit_stochastic_parzen_ica,
    gaussian_parzen_log_density,
    gaussian_parzen_score,
    initialize_parzen_dictionary,
    noisy_parzen_log_density,
    noisy_parzen_posterior_mean,
    noisy_parzen_score,
    parzen_responsibilities,
    projected_noise_variance,
    symmetric_decorrelate,
    track_demixing_components,
    update_parzen_dictionary,
)


def _dictionary_config(seed: int = 7) -> ParzenDictionaryConfig:
    return ParzenDictionaryConfig(
        maximum_centers=8,
        minimum_center_separation=0.05,
        bandwidth=0.35,
        bandwidth_min=0.1,
        bandwidth_max=1.0,
        update_rate=0.01,
        replacement_policy="farthest_center",
        warmup_samples=32,
        seed=seed,
    )


def test_clean_parzen_density_normalizes_and_score_matches_finite_difference() -> None:
    grid = np.linspace(-8, 8, 20001)
    centers = np.array([-1.0, 0.25, 1.5])
    bandwidth = 0.45
    density = np.exp(gaussian_parzen_log_density(grid, centers, bandwidth))
    assert abs(np.trapezoid(density, grid) - 1.0) < 1e-7

    points = np.array([-0.8, 0.0, 1.1])
    step = 1e-6
    derivative = (
        gaussian_parzen_log_density(points + step, centers, bandwidth)
        - gaussian_parzen_log_density(points - step, centers, bandwidth)
    ) / (2 * step)
    np.testing.assert_allclose(
        gaussian_parzen_score(points, centers, bandwidth),
        -derivative,
        rtol=2e-6,
        atol=2e-7,
    )


def test_noisy_density_score_and_posterior_match_single_gaussian_conditioning() -> None:
    values = np.array([-1.0, 0.0, 2.0])
    centers = np.array([0.0])
    bandwidth = 0.5
    noise_variance = 1.0
    total = bandwidth**2 + noise_variance
    expected_log = -0.5 * (values**2 / total + np.log(2 * np.pi * total))
    np.testing.assert_allclose(
        noisy_parzen_log_density(values, centers, bandwidth, noise_variance),
        expected_log,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        noisy_parzen_score(values, centers, bandwidth, noise_variance),
        values / total,
        atol=1e-12,
    )
    mean, variance = noisy_parzen_posterior_mean(
        values, centers, bandwidth, noise_variance, return_variance=True
    )
    np.testing.assert_allclose(mean, bandwidth**2 * values / total, atol=1e-12)
    np.testing.assert_allclose(variance, bandwidth**2 * noise_variance / total, atol=1e-12)


def test_responsibilities_are_log_stable_and_leave_one_out() -> None:
    values = np.array([1e6, -1e6])
    centers = np.array([-1.0, 0.0, 1.0])
    responsibility = parzen_responsibilities(values, centers, 0.1)
    assert np.isfinite(responsibility).all()
    np.testing.assert_allclose(responsibility.sum(axis=-1), 1.0, atol=1e-12)

    own_values = centers.copy()
    leave_one_out = parzen_responsibilities(
        own_values,
        centers,
        0.5,
        exclude_center_indices=np.arange(len(centers)),
    )
    assert np.all(leave_one_out[np.arange(len(centers)), np.arange(len(centers))] == 0)
    np.testing.assert_allclose(leave_one_out.sum(axis=-1), 1.0, atol=1e-12)


def test_whitening_is_stable_and_near_singular_case_is_unresolved() -> None:
    rng = np.random.default_rng(4)
    samples = rng.normal(size=(2, 4000))
    samples[1] += 0.6 * samples[0]
    whitened, result = center_and_whiten_2d(samples)
    np.testing.assert_allclose(
        whitened @ whitened.T / whitened.shape[1], np.eye(2), atol=2e-6
    )
    assert result.identifiable and result.effective_rank == 2
    _, singular = center_and_whiten_2d(np.vstack((samples[0], samples[0])))
    assert not singular.identifiable and singular.effective_rank == 1


def test_dictionary_initialization_and_updates_are_deterministic_and_bounded() -> None:
    samples = np.random.default_rng(8).normal(size=200)
    config = _dictionary_config()
    first = initialize_parzen_dictionary(samples, config)
    second = initialize_parzen_dictionary(samples, config)
    np.testing.assert_array_equal(first.centers, second.centers)
    update_values = np.linspace(-4, 4, 101)
    updated_a = update_parzen_dictionary(first, update_values, config)
    updated_b = update_parzen_dictionary(second, update_values, config)
    np.testing.assert_array_equal(updated_a.centers, updated_b.centers)
    np.testing.assert_array_equal(updated_a.usage, updated_b.usage)
    assert len(updated_a.centers) <= config.maximum_centers
    assert updated_a.diagnostics["updates"] == len(update_values)


def test_decorrelation_and_stochastic_fit_are_finite_and_deterministic() -> None:
    rng = np.random.default_rng(10)
    sources = np.vstack((rng.laplace(size=384), rng.uniform(-2, 2, size=384)))
    mixed = np.array([[1.0, 0.4], [0.2, 1.0]]) @ sources
    whitened, _ = center_and_whiten_2d(mixed)
    matrix = symmetric_decorrelate(np.array([[1.0, 0.3], [-0.2, 0.8]]))
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(2), atol=1e-10)
    kwargs = dict(
        learning_rate=0.001,
        batch_size=64,
        maximum_iterations=4,
        tolerance=1e-12,
    )
    fit_a, dictionaries_a = fit_stochastic_parzen_ica(
        whitened, _dictionary_config(), **kwargs
    )
    fit_b, dictionaries_b = fit_stochastic_parzen_ica(
        whitened, _dictionary_config(), **kwargs
    )
    np.testing.assert_allclose(fit_a.demixing, fit_b.demixing, atol=1e-12)
    np.testing.assert_allclose(fit_a.demixing @ fit_a.demixing.T, np.eye(2), atol=1e-8)
    assert np.isfinite(fit_a.objective) and fit_a.update_count > 0
    for left, right in zip(dictionaries_a, dictionaries_b):
        np.testing.assert_array_equal(left.centers, right.centers)


def test_component_tracking_derivative_energy_and_noise_projection() -> None:
    previous = np.eye(2)
    current = np.array([[0.0, -1.0], [1.0, 0.0]])
    aligned, assignment, signs, diagnostics = track_demixing_components(previous, current)
    np.testing.assert_allclose(aligned, previous)
    assert assignment == (1, 0) and signs == (1, -1)
    assert diagnostics["permutation_changed"] and diagnostics["sign_flip_count"] == 1

    components = np.array([[0, 0, 0, 0], [0, 1, 0, 1]], dtype=float)
    energy = component_derivative_energy(components)
    assert energy["first_difference_energy"][0] == 0
    assert energy["first_difference_energy"][1] > 0
    variance = projected_noise_variance(
        np.eye(2), np.diag([0.2, 0.4]), variance_floor=0.1, variance_ceiling=0.3
    )
    np.testing.assert_allclose(variance, [0.2, 0.3])


def test_decomposition_closure_is_exact_and_keeps_artifact_separate() -> None:
    rng = np.random.default_rng(12)
    channels = [rng.normal(size=(6, 5)) for _ in range(4)]
    observation = sum(channels)
    residual, metrics = decomposition_closure(observation, *channels)
    np.testing.assert_allclose(residual, 0, atol=1e-12)
    assert metrics["normalized_squared_error"] < 1e-30
    assert metrics["maximum_absolute_error"] < 1e-12
