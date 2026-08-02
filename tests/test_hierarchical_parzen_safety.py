import numpy as np

from neurobench.algorithms.hierarchical_parzen_ica import ParzenDictionaryConfig
from neurobench.experiments.hierarchical_parzen_ica.safety import (
    stage1_feedback_diagnostics,
)
from neurobench.experiments.hierarchical_parzen_ica.stage1 import fit_stage1_lane
from neurobench.metrics.hierarchical_separation import stage1_leakage_metrics
from tests.test_hierarchical_parzen_stage1 import _quiet_then_ramp


def _common() -> dict:
    return {
        "calibration_frame_count": 7,
        "fit_sample_pixels": 256,
        "covariance_mode": "ordinary",
        "staticness": {"minimum_confidence_margin": 0.0},
    }


def _batch_options() -> dict:
    return {
        "block_rows": 32,
        "screen_step_degrees": 30.0,
        "refine_half_width_degrees": 2.0,
        "refine_step_degrees": 2.0,
    }


def _dictionary() -> ParzenDictionaryConfig:
    return ParzenDictionaryConfig(
        maximum_centers=12,
        minimum_center_separation=0.05,
        bandwidth=0.35,
        bandwidth_min=0.1,
        bandwidth_max=1.0,
        update_rate=0.01,
        replacement_policy="farthest_center",
        warmup_samples=64,
        seed=17,
    )


def test_batch_raw_feedback_is_rejected_and_safe_fraction_is_applied() -> None:
    frames, background, signal = _quiet_then_ramp(seed=11)
    fitted = fit_stage1_lane(
        frames,
        "batch_cs_parzen_pairwise",
        batch_cs_parzen=_batch_options(),
        **_common(),
    )
    anchoring = fitted.diagnostics["safety"]["reference_anchoring"]
    assert not anchoring["raw_feedback"]["safe"]
    assert "previous_background_coefficient" in (
        anchoring["raw_feedback"]["rejection_reasons"]
    )
    assert anchoring["accepted_learned_fraction"] == 0.003125
    assert anchoring["accepted_feedback"]["safe"]
    assert fitted.diagnostics["safety"]["status"] == "accepted"

    metrics = stage1_leakage_metrics(
        background[1:],
        signal[1:],
        fitted.result.background,
        fitted.result.dynamic_residual,
    )
    assert metrics["signal_residual_nmse"] < 0.1
    assert metrics["closure_max_absolute"] < 1e-6


def test_nonconverged_stochastic_fit_uses_recorded_reference_fallback() -> None:
    frames, background, signal = _quiet_then_ramp(seed=11)
    fitted = fit_stage1_lane(
        frames,
        "stochastic_parzen_score_pairwise",
        stochastic_dictionary=_dictionary(),
        stochastic_fit={
            "batch_size": 64,
            "maximum_iterations": 3,
            "tolerance": 1e-12,
        },
        **_common(),
    )
    safety = fitted.diagnostics["safety"]
    assert not fitted.demixing_fit.converged
    assert safety["status"] == "reference_fallback"
    assert safety["fallback_reasons"] == ["optimizer_not_converged"]
    assert (
        fitted.demixing_fit.diagnostics["applied_model"]
        == "adaptive_gain_reference_fallback"
    )
    assert safety["feedback"]["safe"]

    metrics = stage1_leakage_metrics(
        background[1:],
        signal[1:],
        fitted.result.background,
        fitted.result.dynamic_residual,
    )
    assert metrics["signal_residual_nmse"] < 0.01


def test_feedback_diagnostics_reject_direct_observation_leakage() -> None:
    frames, _, _ = _quiet_then_ramp(seed=17)
    reference = fit_stage1_lane(
        frames,
        "adaptive_gain_common_difference",
        **_common(),
    )
    unsafe = np.eye(2)
    feedback = stage1_feedback_diagnostics(
        reference.whitening,
        unsafe,
        background_component=0,
        maximum_current_observation_coefficient=0.1,
    )
    assert not feedback["safe"]
    assert "current_observation_coefficient" in feedback["rejection_reasons"]


def test_reference_initialized_stochastic_fit_is_deterministic() -> None:
    frames, _, _ = _quiet_then_ramp(seed=19)
    kwargs = {
        **_common(),
        "stochastic_dictionary": _dictionary(),
        "stochastic_fit": {
            "batch_size": 64,
            "maximum_iterations": 4,
            "tolerance": 1e-12,
        },
    }
    first = fit_stage1_lane(
        frames, "stochastic_parzen_score_pairwise", **kwargs
    )
    second = fit_stage1_lane(
        frames, "stochastic_parzen_score_pairwise", **kwargs
    )
    np.testing.assert_allclose(
        first.demixing_fit.demixing,
        second.demixing_fit.demixing,
        atol=1e-12,
    )
    assert (
        first.demixing_fit.diagnostics["initialization"]
        == "provided_symmetric_decorrelation"
    )
    assert first.diagnostics["safety"] == second.diagnostics["safety"]
