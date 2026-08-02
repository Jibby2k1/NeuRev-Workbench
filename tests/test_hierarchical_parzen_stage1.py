import numpy as np

from neurobench.algorithms.hierarchical_parzen_ica import (
    ParzenDictionaryConfig,
    component_staticness_score,
)
from neurobench.experiments.hierarchical_parzen_ica.stage1 import (
    build_aligned_observations,
    fit_stage1_lane,
)
from neurobench.metrics.hierarchical_separation import stage1_leakage_metrics


def _quiet_then_ramp(seed: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    frame_count, height, width = 15, 10, 11
    base = np.linspace(0.2, 1.0, height * width).reshape(height, width)
    noise = 0.002 * rng.normal(size=(frame_count, height, width))
    signal = np.zeros_like(noise)
    signal[7:, 2:5, 4:8] = np.linspace(0.1, 0.8, frame_count - 7)[:, None, None]
    return base + noise + signal, base + noise, signal


def test_aligned_observations_freeze_axes_and_frame_indices() -> None:
    frames = np.arange(8 * 4 * 5, dtype=float).reshape(8, 4, 5)
    pairs = build_aligned_observations(frames, lag_frames=2)
    assert pairs.previous.shape == (6, 4, 5)
    assert pairs.current.shape == (6, 4, 5)
    assert pairs.samples.shape == (2, 120)
    np.testing.assert_array_equal(pairs.previous, frames[:-2])
    np.testing.assert_array_equal(pairs.current, frames[2:])
    np.testing.assert_array_equal(pairs.output_frame_indices_zero, np.arange(2, 8))
    assert pairs.diagnostics["axes"] == "TYX"


def test_staticness_tie_is_explicitly_unresolved() -> None:
    trace = np.arange(60, dtype=float).reshape(5, 3, 4)
    classification = component_staticness_score(
        np.stack((trace, trace)),
        np.eye(2),
        minimum_confidence_margin=0.1,
    )
    assert classification["background_component"] is None
    assert classification["classification_status"] == "unresolved"
    assert classification["background_confidence"] == 0.0
    assert classification["labels_used"] is False


def test_fixed_reference_preserves_sustained_current_frame_event_amplitude() -> None:
    frames, _, _ = _quiet_then_ramp()
    fitted = fit_stage1_lane(
        frames,
        "fixed_common_difference_reference",
        calibration_frame_count=7,
        fit_sample_pixels=600,
        covariance_mode="ordinary",
        staticness={"minimum_confidence_margin": 0.01},
    )
    assert fitted.result.classification_status == "resolved"
    assert fitted.result.background_component == 0
    expected_amplitude = (frames[-1] - frames[0])[2:5, 4:8].mean()
    recovered_amplitude = fitted.result.dynamic_residual[-1, 2:5, 4:8].mean()
    assert recovered_amplitude > 0.75
    assert abs(recovered_amplitude - expected_amplitude) < 2e-3
    assert np.max(np.abs(fitted.result.closure_residual)) == 0
    assert fitted.diagnostics["calibration_source_frames"] == 7
    assert fitted.diagnostics["labels_used"] is False


def test_adaptive_gain_removes_global_illumination_drift_better_than_fixed() -> None:
    rng = np.random.default_rng(5)
    frame_count, height, width = 12, 14, 13
    alpha = 1.035
    base = np.linspace(0.2, 1.0, height * width).reshape(height, width)
    frames = [base]
    for _ in range(frame_count - 1):
        frames.append(alpha * frames[-1] + 0.001 * rng.normal(size=(height, width)))
    movie = np.stack(frames)
    shared = {
        "calibration_frame_count": 9,
        "fit_sample_pixels": 1200,
        "covariance_mode": "ordinary",
        "staticness": {"minimum_confidence_margin": 0.01},
    }
    adaptive = fit_stage1_lane(
        movie, "adaptive_gain_common_difference", **shared
    )
    fixed = fit_stage1_lane(
        movie, "fixed_common_difference_reference", **shared
    )
    adaptive_rms = float(np.sqrt(np.mean(adaptive.result.dynamic_residual**2)))
    fixed_rms = float(np.sqrt(np.mean(fixed.result.dynamic_residual**2)))
    assert abs(adaptive.alpha_gain - alpha) < 2e-4
    assert adaptive.result.background_component == 0
    assert adaptive_rms < 0.15 * fixed_rms


def test_rank_degenerate_input_falls_back_without_subtraction() -> None:
    frame = np.arange(30, dtype=float).reshape(5, 6)
    frames = np.broadcast_to(frame, (9, 5, 6)).copy()
    fitted = fit_stage1_lane(
        frames,
        "fixed_common_difference_reference",
        calibration_frame_count=6,
        covariance_mode="ordinary",
        staticness={"minimum_confidence_margin": 0.01},
    )
    assert not fitted.whitening.identifiable
    assert fitted.result.classification_status == "degenerate"
    assert fitted.result.background_component is None
    np.testing.assert_array_equal(fitted.result.background, 0)
    np.testing.assert_array_equal(fitted.result.dynamic_residual, frames[1:])
    np.testing.assert_array_equal(fitted.result.closure_residual, 0)


def test_batch_and_stochastic_lanes_are_finite_and_bounded() -> None:
    frames, _, _ = _quiet_then_ramp(seed=11)
    common = {
        "calibration_frame_count": 7,
        "fit_sample_pixels": 256,
        "covariance_mode": "ordinary",
        "staticness": {"minimum_confidence_margin": 0.0},
    }
    batch = fit_stage1_lane(
        frames,
        "batch_cs_parzen_pairwise",
        batch_cs_parzen={
            "block_rows": 32,
            "screen_step_degrees": 30.0,
            "refine_half_width_degrees": 2.0,
            "refine_step_degrees": 2.0,
        },
        **common,
    )
    stochastic = fit_stage1_lane(
        frames,
        "stochastic_parzen_score_pairwise",
        stochastic_dictionary=ParzenDictionaryConfig(
            maximum_centers=12,
            minimum_center_separation=0.05,
            bandwidth=0.35,
            bandwidth_min=0.1,
            bandwidth_max=1.0,
            update_rate=0.01,
            replacement_policy="farthest_center",
            warmup_samples=64,
            seed=17,
        ),
        stochastic_fit={
            "batch_size": 64,
            "maximum_iterations": 3,
            "tolerance": 1e-12,
        },
        **common,
    )
    for fitted in (batch, stochastic):
        assert np.isfinite(fitted.demixing_fit.demixing).all()
        assert np.isfinite(fitted.result.background).all()
        assert np.isfinite(fitted.result.dynamic_residual).all()
        assert np.max(np.abs(fitted.result.closure_residual)) < 1e-6
    assert batch.demixing_fit.method_id == "batch_cs_parzen_pairwise"
    assert stochastic.demixing_fit.method_id == "stochastic_parzen_score_pairwise"
    assert len(stochastic.dictionary_states) == 2
    assert all(len(state.centers) <= 12 for state in stochastic.dictionary_states)


def test_recursive_inference_preserves_slow_ramp_signal_amplitude() -> None:
    frames, true_background, true_signal = _quiet_then_ramp(seed=9)
    fitted = fit_stage1_lane(
        frames,
        "fixed_common_difference_reference",
        calibration_frame_count=7,
        fit_sample_pixels=600,
        covariance_mode="ordinary",
        staticness={"minimum_confidence_margin": 0.01},
    )
    metrics = stage1_leakage_metrics(
        true_background[1:],
        true_signal[1:],
        fitted.result.background,
        fitted.result.dynamic_residual,
    )
    assert metrics["closure_max_absolute"] < 1e-6
    assert metrics["signal_residual_nmse"] < 0.01
    assert metrics["signal_leakage_into_background"] < 0.01


def test_motion_edges_remain_a_known_stage1_counterexample() -> None:
    rng = np.random.default_rng(23)
    base = rng.uniform(0.1, 1.0, size=(18, 20))
    frames = np.stack(
        [base + 0.001 * rng.normal(size=base.shape) for _ in range(10)]
    )
    frames[7:] = np.roll(frames[7:], shift=1, axis=2)
    fitted = fit_stage1_lane(
        frames,
        "fixed_common_difference_reference",
        calibration_frame_count=7,
        fit_sample_pixels=1000,
        covariance_mode="ordinary",
        staticness={"minimum_confidence_margin": 0.01},
    )
    quiet_rms = float(np.sqrt(np.mean(fitted.result.dynamic_residual[:5] ** 2)))
    motion_rms = float(np.sqrt(np.mean(fitted.result.dynamic_residual[6] ** 2)))
    assert motion_rms > 20 * quiet_rms
