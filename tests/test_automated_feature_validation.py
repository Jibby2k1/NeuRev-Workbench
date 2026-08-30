from __future__ import annotations

import numpy as np

from neurobench.experiments.neuron_identifiability.automated_feature_validation import (
    choose_control,
    frame_auc_only,
    frame_retrieval,
    nonparametric_site_fraction,
)


def test_frame_retrieval_rewards_aligned_event() -> None:
    trace = np.zeros(100)
    trace[50:60] = np.arange(1, 11)
    quiet = np.ones(100, dtype=bool)
    quiet[35:75] = False
    metrics = frame_retrieval(trace, 1850, 1859, quiet)
    assert metrics["frame_auc"] == 1.0
    assert np.isclose(metrics["reference_average_precision"], 1.0)
    assert metrics["top_10pct_event_recall"] > 0.5
    assert np.isclose(frame_auc_only(trace, 1850, 1859, quiet), metrics["frame_auc"])


def test_nonparametric_site_fraction_detects_stable_site_offsets() -> None:
    stable = np.array([[0.1, 0.11, 0.09], [0.8, 0.79, 0.81], [0.4, 0.39, 0.41]])
    unstable = np.array([[0.1, 0.9, 0.3], [0.8, 0.1, 0.6], [0.4, 0.7, 0.1]])
    assert nonparametric_site_fraction(stable) > nonparametric_site_fraction(unstable)
    assert 0 <= nonparametric_site_fraction(stable) <= 1


def test_control_is_same_field_and_away_from_labels() -> None:
    raw = np.zeros((20, 100, 500), dtype=float)
    quiet = np.ones(20, dtype=bool)
    labels = [(100, 50), (130, 50)]
    x, y = choose_control(100, 50, raw, quiet, labels)
    assert x < 286
    assert min(np.hypot(x - lx, y - ly) for lx, ly in labels) >= 12
