from __future__ import annotations

import numpy as np

from neurobench.experiments.neuron_identifiability.feature_atlas_v1 import (
    _extract_patch,
    competition_features,
    envelope_features,
    heldout_map_trace_correlation,
    split_half_map_cosine,
    trailing_maximum,
)


def test_trailing_maximum_is_causal_and_includes_current_sample() -> None:
    values = np.asarray([0.0, 2.0, 1.0, 3.0, 0.5])
    assert np.allclose(trailing_maximum(values, 3), [0.0, 2.0, 2.0, 3.0, 3.0])


def test_envelope_peak_matches_instantaneous_peak_but_area_changes() -> None:
    trace = np.asarray([0.0, 1.0, 4.0, 3.0, 2.0, 0.0])
    features = envelope_features(trace, 1, 5)
    assert features["envelope_h3_peak"] == 4.0
    assert features["envelope_h5_peak"] == 4.0
    assert 0.0 < features["envelope_h5_area_ratio"] < 1.0
    assert 0.0 < features["envelope_h5_core_fraction"] <= 1.0


def test_split_half_map_cosine_recognizes_repeated_map() -> None:
    spatial = np.asarray([[0.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 0.0]])
    patch = np.stack([spatial * value for value in (1, 2, 3, 4, 5, 6)])
    assert split_half_map_cosine(patch) > 0.999


def test_heldout_map_trace_correlation_uses_surround_not_center_tautology() -> None:
    rng = np.random.default_rng(12)
    time = rng.normal(0, 1.0, 80)
    coherent = rng.normal(0, 0.01, (80, 5, 5)) + time[:, None, None]
    coherent[:, 2, 2] = time
    incoherent = rng.normal(0, 1.0, (80, 5, 5))
    incoherent[:, 2, 2] = time
    assert heldout_map_trace_correlation(coherent) > 0.95
    assert heldout_map_trace_correlation(coherent) > heldout_map_trace_correlation(incoherent) + 0.5


def test_competition_features_are_partition_local() -> None:
    coordinates = np.asarray([[0, 0], [3, 0], [100, 100], [0, 0]], dtype=float)
    scores = np.asarray([5.0, 4.0, 1.0, 20.0])
    partitions = np.asarray([1, 1, 1, 2])
    isolation, margin = competition_features(coordinates, scores, partitions, radius=5.0)
    assert np.allclose(isolation, [-1.0, -1.0, 0.0, 0.0])
    assert np.allclose(margin, [1.0, -1.0, 0.0, 0.0])


def test_extract_patch_copies_read_only_float32_source() -> None:
    source = np.arange(8 * 9 * 9, dtype=np.float32).reshape(8, 9, 9)
    source.flags.writeable = False
    patch = _extract_patch(source, slice(1, 4), 4, 4, radius=2)
    assert patch.flags.writeable
    patch -= 1.0
