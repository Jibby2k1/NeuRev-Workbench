from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from neurobench.experiments.causal_proposal_program import (
    _load_maps,
    _cache_maps,
    _perturb,
    build_method_specs,
    planned_counts,
)


def _design() -> SimpleNamespace:
    return SimpleNamespace(
        spatial_sigmas=(0.0, 0.5, 1.0, 1.5, 2.0),
        temporal_ema_spans=(1.0, 2.0, 4.0, 8.0, 16.0),
        artifact_attenuations=(0.0, 0.35, 0.7, 0.9),
        intensity_transforms=("linear", "asinh2", "asinh5", "asinh10"),
        baseline_modes=("frozen_median", "slow_clipped_ema", "robust_adaptive"),
        pool_modes=("lme0.1", "lme0.25", "lme0.5", "mean", "max"),
        fractional_count=48,
        fractional_seed=20260727,
        fusion_finalists=8,
        fusion_variants_per_finalist=12,
        robustness_finalists=12,
        calibration_conditions=8,
        perturbation_types=(
            "gain", "offset", "noise", "translation", "stripe", "saturation_bloom",
            "frame_drop", "quiet_contamination", "photobleach",
        ),
        perturbation_severities=(0.1, 0.25),
        horizon_frames=(12, 24, 28, 47, 64),
    )


def test_preregistered_design_has_exact_counts_and_unique_methods() -> None:
    config = _design()
    methods = build_method_specs(config)
    counts = planned_counts(config)

    assert len(methods) == len({row["method_id"] for row in methods}) == 72
    assert sum(row["method_id"].startswith("ofat_") for row in methods) == 20
    assert sum(row["method_id"].startswith("fractional_") for row in methods) == 48
    assert counts == {
        "breadth_methods": 72,
        "policies": 9,
        "breadth_evaluations": 648,
        "maximum_fusion_methods": 96,
        "maximum_fusion_evaluations": 864,
        "robustness_conditions": 31,
        "maximum_robustness_evaluations": 372,
        "maximum_logical_evaluations": 1884,
        "maximum_fold_condition_scores": 7536,
    }


def test_all_raw_perturbations_are_finite_shape_preserving_and_bounded() -> None:
    raw = np.linspace(0, 4095, 60 * 24 * 28, dtype=np.float32).reshape(60, 24, 28)
    for index, kind in enumerate(_design().perturbation_types):
        result = _perturb(raw, kind, 0.1, 100 + index)
        assert result.shape == raw.shape
        assert result.dtype == np.float32
        assert np.isfinite(result).all()
        assert 0 <= result.min() <= result.max() <= 4095


def test_score_map_cache_is_atomic_and_round_trips(tmp_path) -> None:
    maps = {
        "quiet": [np.full((5, 7), i, dtype=np.float32) for i in range(4)],
        "events": {i: np.full((5, 7), i + 4, dtype=np.float32) for i in range(1, 5)},
    }
    path = tmp_path / "maps.npz"

    _cache_maps(path, maps)
    loaded = _load_maps(path)

    assert path.is_file()
    assert not list(tmp_path.glob("*.partial"))
    for expected, actual in zip(maps["quiet"], loaded["quiet"]):
        np.testing.assert_array_equal(expected, actual)
    for burst in range(1, 5):
        np.testing.assert_array_equal(maps["events"][burst], loaded["events"][burst])
