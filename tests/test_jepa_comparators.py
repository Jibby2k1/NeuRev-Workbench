from __future__ import annotations

import numpy as np

from neurobench.experiments.neuron_identifiability.jepa_comparators import (
    frozen_handcrafted_comparator,
)
from neurobench.experiments.neuron_identifiability.major_next_steps import (
    _score_maps,
)


def test_source_off_score_is_exact_legacy_combined_regression() -> None:
    movie = np.random.default_rng(7001).normal(size=(32, 24, 25)).astype(np.float32)
    observed = frozen_handcrafted_comparator(movie)
    expected = _score_maps(movie)["combined"]
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_pair_score_reuses_source_off_component_calibration() -> None:
    rng = np.random.default_rng(7002)
    source_off = rng.normal(size=(32, 24, 25)).astype(np.float32)
    source_on = source_off.copy()
    source_on[8:20, 10:14, 11:15] += 7.0
    fitted = frozen_handcrafted_comparator.fit_source_off(source_off)
    off_map, on_map = fitted(source_off), fitted(source_on)
    np.testing.assert_allclose(off_map, frozen_handcrafted_comparator(source_off))
    assert not np.allclose(on_map, frozen_handcrafted_comparator(source_on))
    assert frozen_handcrafted_comparator.contract()["source_on_self_calibration"] is False
