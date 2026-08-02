"""Pure-array orchestration for Stage-1 background reconstruction lanes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neurobench.algorithms.hierarchical_parzen_ica import (
    DemixingFit,
    ParzenDictionaryConfig,
    ParzenDictionaryState,
    Stage1Result,
    WhiteningResult,
    center_and_whiten_2d,
    component_staticness_score,
    fit_batch_cs_parzen_2d,
    fit_stochastic_parzen_ica,
    initialize_parzen_dictionary,
    stage1_recursive_background_residual,
    symmetric_decorrelate,
)
from neurobench.algorithms.pairwise_separation import estimate_quiet_gain
from neurobench.experiments.hierarchical_parzen_ica.safety import (
    anchor_demixing_to_reference,
    stage1_feedback_diagnostics,
)


STAGE1_METHODS = {
    "fixed_common_difference_reference",
    "adaptive_gain_common_difference",
    "batch_cs_parzen_pairwise",
    "stochastic_parzen_score_pairwise",
}


@dataclass(frozen=True)
class AlignedFramePairs:
    previous: np.ndarray
    current: np.ndarray
    samples: np.ndarray
    lag_frames: int
    source_frame_count: int
    output_frame_indices_zero: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class Stage1LaneFit:
    method_id: str
    alpha_gain: float
    whitening: WhiteningResult
    demixing_fit: DemixingFit
    dictionary_states: tuple[ParzenDictionaryState, ...]
    staticness: dict[str, Any]
    result: Stage1Result
    diagnostics: dict[str, Any]


def build_aligned_observations(
    frames: np.ndarray,
    lag_frames: int = 1,
) -> AlignedFramePairs:
    """Construct aligned ``[I(t-k), I(t)]`` observations from a TYX movie."""
    values = np.asarray(frames, dtype=np.float64)
    lag = int(lag_frames)
    if values.ndim != 3 or not values.size or not np.isfinite(values).all():
        raise ValueError("frames must be a non-empty finite TYX array")
    if not 1 <= lag < len(values) - 2:
        raise ValueError("lag_frames must leave at least three aligned frames")
    previous = values[:-lag]
    current = values[lag:]
    samples = np.stack((previous.reshape(-1), current.reshape(-1)), axis=0)
    output_indices = np.arange(lag, len(values), dtype=np.int64)
    return AlignedFramePairs(
        previous=previous,
        current=current,
        samples=samples,
        lag_frames=lag,
        source_frame_count=len(values),
        output_frame_indices_zero=output_indices,
        diagnostics={
            "axes": "TYX",
            "sample_axes": "CN",
            "lag_frames": lag,
            "aligned_frames": len(current),
            "pixels_per_frame": int(np.prod(values.shape[1:])),
            "frame_contract": "zero_based_half_open",
        },
    )


def _bounded_sample_indices(total: int, count: int, seed: int) -> np.ndarray:
    requested = int(count)
    if total < 3 or requested < 3:
        raise ValueError("Stage-1 fit sampling requires at least three samples")
    selected = min(total, requested)
    if selected == total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(total, size=selected, replace=False)).astype(np.int64)


def _reference_demixing(whitening: WhiteningResult, alpha_gain: float) -> np.ndarray:
    alpha = float(alpha_gain)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha_gain must be finite and positive")
    # Source 0 is the previous-frame background contribution. Source 1 is the
    # full current-frame innovation, so reconstruction does not split a
    # current-only event equally between common and differential directions.
    observation_mixing = np.asarray(
        [[1.0, 0.0], [alpha, 1.0]], dtype=np.float64
    )
    demixing_observation = np.linalg.inv(observation_mixing)
    demixing_whitened = demixing_observation @ whitening.dewhitening
    if not np.isfinite(demixing_whitened).all():
        raise FloatingPointError("reference demixing is non-finite")
    return demixing_whitened


def _fit_reference(method_id: str, demixing: np.ndarray) -> DemixingFit:
    return DemixingFit(
        method_id=method_id,
        demixing=demixing,
        converged=True,
        iterations=0,
        objective=None,
        gradient_norm=None,
        update_count=0,
        diagnostics={
            "reference": True,
            "observation_directions": "gain_adjusted_common_difference",
        },
    )


def _reshape_result(result: Stage1Result, shape: tuple[int, int, int]) -> Stage1Result:
    return Stage1Result(
        background=result.background.reshape(shape),
        dynamic_residual=result.dynamic_residual.reshape(shape),
        differential_component=result.differential_component.reshape(shape),
        closure_residual=result.closure_residual.reshape(shape),
        background_component=result.background_component,
        confidence=result.confidence,
        classification_status=result.classification_status,
        diagnostics={**result.diagnostics, "axes": "TYX", "shape": list(shape)},
    )


def fit_stage1_lane(
    frames: np.ndarray,
    method_id: str,
    *,
    lag_frames: int = 1,
    calibration_frame_count: int,
    fit_sample_pixels: int = 4096,
    sample_seed: int = 20260729,
    covariance_mode: str = "robust",
    eigenvalue_floor_ratio: float = 1e-6,
    condition_number_max: float = 1e8,
    alpha_min: float = 0.8,
    alpha_max: float = 1.2,
    subtraction_mode: str = "exact",
    batch_cs_parzen: dict[str, Any] | None = None,
    stochastic_dictionary: ParzenDictionaryConfig | None = None,
    stochastic_fit: dict[str, Any] | None = None,
    staticness: dict[str, Any] | None = None,
    safety: dict[str, Any] | None = None,
) -> Stage1LaneFit:
    """Fit one bounded Stage-1 lane and reconstruct its signed residual movie."""
    if method_id not in STAGE1_METHODS:
        raise ValueError(f"unsupported Stage-1 method: {method_id}")
    pairs = build_aligned_observations(frames, lag_frames)
    calibration_count = int(calibration_frame_count)
    if not lag_frames + 3 <= calibration_count <= pairs.source_frame_count:
        raise ValueError("calibration_frame_count must contain at least three aligned pairs")
    calibration_pairs = build_aligned_observations(
        np.asarray(frames)[:calibration_count], lag_frames
    )
    indices = _bounded_sample_indices(
        calibration_pairs.samples.shape[1], fit_sample_pixels, sample_seed
    )
    fit_samples = calibration_pairs.samples[:, indices]
    fit_whitened, whitening = center_and_whiten_2d(
        fit_samples,
        eigenvalue_floor_ratio=eigenvalue_floor_ratio,
        condition_number_max=condition_number_max,
        covariance_mode=covariance_mode,
    )
    alpha_gain = 1.0
    gain_diagnostics: dict[str, Any] = {"mode": "fixed", "alpha_gain": 1.0}
    if method_id != "fixed_common_difference_reference":
        alpha_gain, gain_diagnostics = estimate_quiet_gain(
            calibration_pairs.previous,
            calibration_pairs.current,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        gain_diagnostics = {"mode": "robust_pairwise", **gain_diagnostics}

    safety_options = {
        "maximum_previous_background_coefficient": 1.2,
        "maximum_current_observation_coefficient": 0.1,
        "maximum_reconstruction_operator_norm": 2.0,
        "maximum_learned_fraction": 0.1,
        "minimum_learned_fraction": 0.0015625,
        "require_convergence_for_learned": True,
        "unsafe_policy": "reference_fallback",
        **(safety or {}),
    }
    if safety_options["unsafe_policy"] != "reference_fallback":
        raise ValueError("Stage-1 unsafe_policy must be reference_fallback")
    feedback_keys = {
        "maximum_previous_background_coefficient",
        "maximum_current_observation_coefficient",
        "maximum_reconstruction_operator_norm",
    }
    feedback_options = {
        key: float(safety_options[key]) for key in feedback_keys
    }
    reference_demixing = (
        _reference_demixing(whitening, alpha_gain)
        if whitening.identifiable else np.eye(2, dtype=np.float64)
    )

    dictionary_states: tuple[ParzenDictionaryState, ...] = ()
    if not whitening.identifiable:
        demixing = np.eye(2, dtype=np.float64)
        demixing_fit = DemixingFit(
            method_id=method_id,
            demixing=demixing,
            converged=False,
            iterations=0,
            objective=None,
            gradient_norm=None,
            update_count=0,
            diagnostics={"status": "degenerate_whitening"},
        )
    elif method_id in {
        "fixed_common_difference_reference", "adaptive_gain_common_difference"
    }:
        demixing = reference_demixing.copy()
        demixing_fit = _fit_reference(method_id, demixing)
    elif method_id == "batch_cs_parzen_pairwise":
        options = {
            "bandwidth": 0.35,
            "block_rows": 64,
            "screen_step_degrees": 5.0,
            "refine_half_width_degrees": 3.0,
            "refine_step_degrees": 0.5,
            **(batch_cs_parzen or {}),
        }
        split = max(3, len(indices) // 2)
        demixing_fit = fit_batch_cs_parzen_2d(
            fit_whitened[:, :split], fit_whitened, **options
        )
        demixing = demixing_fit.demixing
    else:
        dictionary = stochastic_dictionary or ParzenDictionaryConfig(
            maximum_centers=64,
            minimum_center_separation=0.05,
            bandwidth=0.35,
            bandwidth_min=0.1,
            bandwidth_max=1.0,
            update_rate=0.01,
            replacement_policy="farthest_center",
            warmup_samples=min(256, fit_whitened.shape[1]),
            seed=sample_seed,
        )
        options = {
            "initial_demixing": symmetric_decorrelate(reference_demixing),
            "learning_rate": 0.001,
            "gradient_clip": 5.0,
            "maximum_angle_update_degrees": 1.0,
            "batch_size": min(128, fit_whitened.shape[1]),
            "maximum_iterations": 100,
            "tolerance": 1e-6,
            **(stochastic_fit or {}),
        }
        core_fit, dictionary_states = fit_stochastic_parzen_ica(
            fit_whitened, dictionary, **options
        )
        demixing_fit = DemixingFit(
            method_id=method_id,
            demixing=core_fit.demixing,
            converged=core_fit.converged,
            iterations=core_fit.iterations,
            objective=core_fit.objective,
            gradient_norm=core_fit.gradient_norm,
            update_count=core_fit.update_count,
            diagnostics={
                **core_fit.diagnostics,
                "core_method_id": core_fit.method_id,
            },
        )
        demixing = demixing_fit.demixing

    anchoring: dict[str, Any] | None = None
    if whitening.identifiable and method_id in {
        "batch_cs_parzen_pairwise", "stochastic_parzen_score_pairwise"
    }:
        unconstrained_demixing = demixing.copy()
        demixing, anchoring = anchor_demixing_to_reference(
            reference_demixing,
            unconstrained_demixing,
            whitening,
            maximum_learned_fraction=float(
                safety_options["maximum_learned_fraction"]
            ),
            minimum_learned_fraction=float(
                safety_options["minimum_learned_fraction"]
            ),
            **feedback_options,
        )
        demixing_fit = DemixingFit(
            method_id=demixing_fit.method_id,
            demixing=demixing,
            converged=demixing_fit.converged,
            iterations=demixing_fit.iterations,
            objective=demixing_fit.objective,
            gradient_norm=demixing_fit.gradient_norm,
            update_count=demixing_fit.update_count,
            diagnostics={
                **demixing_fit.diagnostics,
                "unconstrained_demixing": unconstrained_demixing.tolist(),
                "reference_anchoring": anchoring,
            },
        )
        if method_id == "stochastic_parzen_score_pairwise":
            dictionary_states = tuple(
                initialize_parzen_dictionary(
                    (demixing @ fit_whitened)[component], dictionary
                )
                for component in range(2)
            )

    full_whitened = whitening.whitening @ (
        pairs.samples - whitening.mean[:, None]
    )
    components = (demixing @ full_whitened).reshape(
        2, *pairs.current.shape
    )
    calibration_aligned_frames = len(calibration_pairs.current)
    calibration_components = components[:, :calibration_aligned_frames]
    observation_mixing = whitening.dewhitening @ np.linalg.pinv(demixing)
    static_options = {
        "first_difference_weight": 1.0,
        "second_difference_weight": 0.5,
        "common_direction_weight": 1.0,
        "spatial_high_frequency_weight": 0.25,
        "global_intensity_weight": 0.25,
        "minimum_confidence_margin": 0.1,
        **(staticness or {}),
    }
    classification = component_staticness_score(
        calibration_components,
        observation_mixing,
        alpha_gain=alpha_gain,
        global_intensity=calibration_pairs.current.reshape(
            calibration_aligned_frames, -1
        ).mean(axis=1),
        **static_options,
    )
    if not whitening.identifiable:
        classification = {
            **classification,
            "background_component": None,
            "classification_status": "degenerate",
        }
    selected = classification["background_component"]
    feedback: dict[str, Any] | None = None
    fallback_reasons: list[str] = []
    model_safety_status = "not_evaluated"
    if selected is not None and whitening.identifiable:
        feedback = stage1_feedback_diagnostics(
            whitening, demixing, int(selected), **feedback_options
        )
        fallback_reasons.extend(feedback["rejection_reasons"])
        if (
            method_id in {
                "batch_cs_parzen_pairwise",
                "stochastic_parzen_score_pairwise",
            }
            and bool(safety_options["require_convergence_for_learned"])
            and not demixing_fit.converged
        ):
            fallback_reasons.append("optimizer_not_converged")
        if fallback_reasons and method_id in {
            "batch_cs_parzen_pairwise",
            "stochastic_parzen_score_pairwise",
        }:
            pre_safety_component = int(selected)
            demixing = reference_demixing.copy()
            selected = 0
            feedback = stage1_feedback_diagnostics(
                whitening, demixing, selected, **feedback_options
            )
            classification = {
                **classification,
                "background_component": selected,
                "classification_status": "resolved",
                "pre_safety_background_component": pre_safety_component,
                "safety_override": "adaptive_reference_fallback",
            }
            demixing_fit = DemixingFit(
                method_id=demixing_fit.method_id,
                demixing=demixing,
                converged=demixing_fit.converged,
                iterations=demixing_fit.iterations,
                objective=demixing_fit.objective,
                gradient_norm=demixing_fit.gradient_norm,
                update_count=demixing_fit.update_count,
                diagnostics={
                    **demixing_fit.diagnostics,
                    "applied_model": "adaptive_gain_reference_fallback",
                    "fallback_reasons": list(fallback_reasons),
                },
            )
            if method_id == "stochastic_parzen_score_pairwise":
                dictionary_states = tuple(
                    initialize_parzen_dictionary(
                        (demixing @ fit_whitened)[component], dictionary
                    )
                    for component in range(2)
                )
            model_safety_status = "reference_fallback"
        elif fallback_reasons:
            selected = None
            classification = {
                **classification,
                "background_component": None,
                "classification_status": "unresolved",
                "safety_override": "no_subtraction",
            }
            model_safety_status = "unsafe_no_subtraction"
        else:
            model_safety_status = (
                "reference_fallback"
                if anchoring is not None
                and anchoring["reference_fallback"]
                else "accepted"
            )

    raw_result = stage1_recursive_background_residual(
        np.asarray(frames, dtype=np.float64),
        int(lag_frames),
        whitening,
        demixing,
        selected,
        confidence=float(classification["background_confidence"]),
        subtraction_mode=(
            "no_subtraction" if selected is None else subtraction_mode
        ),
        method_id=method_id,
    )
    if classification["classification_status"] == "degenerate":
        raw_result = Stage1Result(
            background=raw_result.background,
            dynamic_residual=raw_result.dynamic_residual,
            differential_component=raw_result.differential_component,
            closure_residual=raw_result.closure_residual,
            background_component=None,
            confidence=raw_result.confidence,
            classification_status="degenerate",
            diagnostics=raw_result.diagnostics,
        )
    result = _reshape_result(raw_result, pairs.current.shape)
    return Stage1LaneFit(
        method_id=method_id,
        alpha_gain=float(alpha_gain),
        whitening=whitening,
        demixing_fit=demixing_fit,
        dictionary_states=dictionary_states,
        staticness=classification,
        result=result,
        diagnostics={
            "lag_frames": int(lag_frames),
            "calibration_source_frames": calibration_count,
            "calibration_aligned_frames": calibration_aligned_frames,
            "fit_sample_count": int(len(indices)),
            "fit_sample_indices_checksum": int(np.sum(indices, dtype=np.int64)),
            "gain": gain_diagnostics,
            "safety": {
                "status": model_safety_status,
                "feedback": feedback,
                "fallback_reasons": list(fallback_reasons),
                "reference_anchoring": anchoring,
                "policy": safety_options["unsafe_policy"],
            },
            "output_frame_indices_zero": pairs.output_frame_indices_zero.tolist(),
            "labels_used": False,
            "causal_status": "causal_frozen_recursive_background",
        },
    )
