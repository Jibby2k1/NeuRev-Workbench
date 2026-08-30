from __future__ import annotations

import numpy as np

from neurobench.experiments.neuron_identifiability.full_trace_feature_panel import (
    causal_mean,
    exponential_matched_filter,
    make_event_and_quiet_masks,
    occurrence_score,
    quiet_standardize,
    quiet_window_maxima,
)


def test_masks_use_one_based_inclusive_ui_frames_and_guard() -> None:
    event, quiet = make_event_and_quiet_masks(100, [(1810, 1819)])
    assert event.sum() == 10
    assert np.flatnonzero(event).tolist() == list(range(10, 20))
    assert not quiet[:35].any()
    assert quiet[35:].all()


def test_quiet_standardization_uses_only_quiet_samples() -> None:
    trace = np.array([0.0, 1.0, 2.0, 100.0])
    quiet = np.array([True, True, True, False])
    standardized = quiet_standardize(trace, quiet)
    assert standardized[1] == 0.0
    assert standardized[3] > 50


def test_causal_mean_and_matched_filter_do_not_use_future_values() -> None:
    values = np.zeros(12)
    values[7] = 10.0
    assert np.allclose(causal_mean(values, 3)[:7], 0.0)
    assert np.allclose(exponential_matched_filter(values)[:7], 0.0)


def test_event_score_is_maximal_for_isolated_event() -> None:
    feature = np.zeros(100)
    feature[50:55] = np.arange(1, 6)
    quiet = np.ones(100, dtype=bool)
    quiet[45:60] = False
    union = np.zeros(100, dtype=bool)
    union[50:55] = True
    result = occurrence_score(feature, 1850, 1854, quiet, union)
    assert result["event_localization_percentile"] == 1.0
    assert result["event_energy_fraction"] == 1.0
    assert len(quiet_window_maxima(feature, 5, quiet)) > 0
