from __future__ import annotations

from pathlib import Path

import numpy as np

from neurobench.experiments.hierarchical_parzen_ica.feature_utility_config import (
    FEATURE_IDS,
    FeatureUtilityConfig,
)
from neurobench.experiments.hierarchical_parzen_ica.feature_utility_program import (
    _combine_maps,
    _fit_weights,
)


MANIFEST = Path("examples/spon_ca_burst_feature_utility_v1.example.json")


def test_frozen_manifest_counts_and_output_contract() -> None:
    config = FeatureUtilityConfig.load(MANIFEST)
    assert config.feature_count == 25
    assert config.fixed_lane_count == 176
    assert config.learned_scalar_fit_count == 100
    assert config.multifeature_fit_count == 4
    assert len(config.feature_bank["tiff_feature_ids"]) == 10
    assert set(config.feature_bank["tiff_feature_ids"]) <= set(FEATURE_IDS)


def test_map_fusions_preserve_the_carrier_and_bounds() -> None:
    carrier = {
        "quiet": [np.ones((2, 2), dtype=np.float32)],
        "events": {1: np.ones((2, 2), dtype=np.float32)},
    }
    feature = {
        "quiet": [np.asarray([[0, 1], [2, -1]], dtype=np.float32)],
        "events": {1: np.full((2, 2), 0.5, dtype=np.float32)},
    }
    boosted = _combine_maps(carrier, feature, kind="boost", value=0.5)
    gated = _combine_maps(carrier, feature, kind="gate", value=0.75)
    assert np.allclose(boosted["quiet"][0], [[1, 1.5], [1.5, 1]])
    assert np.allclose(gated["quiet"][0], [[0.75, 1], [1, 0.75]])
    assert np.all(boosted["events"][1] >= carrier["events"][1])


def test_multifeature_fit_is_nonnegative_and_bounded() -> None:
    raw_positive = np.asarray([1.0, 0.8, 1.2])
    raw_negative = np.asarray([0.4, 0.3, 0.5])
    feature_positive = np.asarray([[0.8, 0.2], [0.9, 0.1], [0.7, 0.3]])
    feature_negative = np.asarray([[0.1, 0.2], [0.2, 0.1], [0.1, 0.2]])
    weights, history = _fit_weights(
        raw_positive,
        feature_positive,
        raw_negative,
        feature_negative,
        learning_rate=0.02,
        epochs=100,
        l2=0.1,
        maximum_total=0.5,
    )
    assert np.all(weights >= 0)
    assert weights.sum() <= 0.5 + 1e-12
    assert history[-1] < history[0]
