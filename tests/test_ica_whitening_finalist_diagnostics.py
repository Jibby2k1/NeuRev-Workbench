import numpy as np

from neurobench.experiments.ica_whitening_evaluation.finalist_diagnostics import (
    approximate_source_snr, ordered_trace_coherence, reconstruction_integrity,
    seed_stability,
)


def test_reconstruction_integrity_is_exact_for_identity_model():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(3, 80))
    result = reconstruction_integrity(values, values, np.eye(3), np.zeros(3))
    assert result["reconstruction_nmse_centered"] == 0.0
    assert np.isclose(result["raw_reconstruction_correlation"], 1.0)
    assert np.isclose(result["absolute_area_retention"], 1.0)


def test_approximate_snr_is_explicit_and_detects_event_scale():
    rng = np.random.default_rng(2)
    quiet = rng.normal(size=(2, 100))
    event = 4 * rng.normal(size=(2, 100))
    result = approximate_source_snr(
        np.concatenate([quiet, event], axis=1), np.arange(200), 100
    )
    assert result["semantics"].endswith("not_true_snr")
    assert result["median_ratio"] > 2.5


def test_ordered_trace_coherence_identical_trace_is_unity():
    trace = np.sin(np.linspace(0, 8 * np.pi, 128))
    result = ordered_trace_coherence(trace, trace, sample_period_seconds=.1)
    assert np.isclose(result["zero_lag_correlation"], 1.0)
    assert np.isclose(result["mean_magnitude_squared_coherence"], 1.0)
    assert result["lag_at_maximum_frames"] == 0


def test_seed_stability_aligns_permutations_and_signs():
    first = np.asarray([[1., 0., 0.], [0., 1., 0.]])
    second = np.asarray([[0., -1., 0.], [1., 0., 0.]])
    result = seed_stability([first, second])
    assert np.isclose(result["mean_aligned_component_cosine"], 1.0)
    assert result["individual_component_interpretation_allowed"] is True
