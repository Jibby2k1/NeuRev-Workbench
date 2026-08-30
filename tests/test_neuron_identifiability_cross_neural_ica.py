import numpy as np

from neurobench.experiments.neuron_identifiability.cross_neural_ica import _bh_adjust, _global_adjust, analyze_pairs


def test_global_adjust_removes_shared_trace_without_changing_shape():
    time = np.linspace(-1, 1, 12)
    cube = np.stack([np.stack([time + site for _ in range(4)]) for site in range(5)])
    adjusted = _global_adjust(cube)
    assert adjusted.shape == cube.shape
    assert np.max(np.abs(adjusted)) < 1e-10


def test_bh_adjust_is_monotone_in_sorted_p_values():
    adjusted = _bh_adjust([0.001, 0.02, 0.03, 0.8])
    assert adjusted[0] <= adjusted[1] <= adjusted[2] <= adjusted[3]
    assert all(0 <= value <= 1 for value in adjusted)


def test_pair_analysis_includes_frozen_ica_and_matched_pair_counts():
    rng = np.random.default_rng(4)
    base = rng.normal(size=(4, 16))
    cubes = {}
    for representation in ("raw", "residual", "global_adjusted_raw", "frozen_two_frame_ica"):
        cubes[representation] = np.stack((base, base + rng.normal(scale=.01, size=base.shape), rng.normal(size=base.shape)))
    rows, summary = analyze_pairs(["a", "b", "c"], cubes, {"a": (0, 0), "b": (3, 4), "c": (8, 0)}, permutations=20, seed=2)
    assert len(rows) == 12
    assert set(summary) == {"raw", "residual", "global_adjusted_raw", "frozen_two_frame_ica"}
    assert all(value["pairs"] == 3 for value in summary.values())
    assert all("circular_shift_q_bh" in row for row in rows)
