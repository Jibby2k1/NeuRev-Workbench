"""Fit and apply the five declared pairwise source-separation lanes."""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import numpy as np

from neurobench.algorithms.pairwise_separation import (
    SeparationFit,
    adaptive_difference,
    apply_linear_separation,
    center_and_whiten_2d,
    estimate_quiet_gain,
    fit_cs_parzen_ica,
    fit_infomax_tanh_ica,
    fit_shared_background_nmf,
    fixed_difference,
    orient_and_select_activity_component,
    quiet_difference_stats,
    standardized_positive_mask,
)

from .config import PairwiseSeparationConfig
from .sampling import sample_pair_observations, uniform_anatomy_mask


def _stats_payload(stats) -> dict[str, Any]:
    return {"scale_floor": stats.scale_floor, "zero_scale_fraction": float(np.mean(stats.scale <= stats.scale_floor)),
            "scale_median": float(np.median(stats.scale)), "scale_p95": float(np.percentile(stats.scale, 95)),
            "finite_pixels": int(np.isfinite(stats.scale).sum()), "invalid_pixels": int((~np.isfinite(stats.scale)).sum())}


def _calibrate(activity: np.ndarray, quiet_count: int, config: PairwiseSeparationConfig) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    lag = config.preprocessing.lag_frames
    stats = quiet_difference_stats(activity[lag:quiet_count], floor_percentile=config.thresholding.quiet_mad_floor_percentile)
    z, mask = standardized_positive_mask(activity, stats, config.thresholding.primary_z_threshold,
                                         undefined_leading_frames=lag)
    return z, mask, _stats_payload(stats)


def _apply_ica(
    filtered: np.ndarray,
    whitening,
    fit: SeparationFit,
    component: int,
    signs: tuple[int, int],
    chunk: int,
    lag: int,
) -> np.ndarray:
    output = np.zeros_like(filtered, dtype=np.float32)
    pixels = filtered.shape[1] * filtered.shape[2]
    for start in range(lag, len(filtered), chunk):
        stop = min(len(filtered), start + chunk)
        samples = np.vstack((filtered[start-lag:stop-lag].reshape(-1), filtered[start:stop].reshape(-1)))
        separated = apply_linear_separation(samples, whitening.mean, whitening.whitening, fit.demixing)
        output[start:stop] = (separated[component] * signs[component]).reshape(stop-start, *filtered.shape[1:])
    return output


def fit_lanes(filtered: np.ndarray, quiet_count: int, config: PairwiseSeparationConfig) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return aligned continuous/z/mask arrays plus fit metadata for enabled lanes."""
    p, s, methods = config.preprocessing, config.sampling, config.methods
    anatomy, anatomy_summary = uniform_anatomy_mask(filtered[:quiet_count])
    observations, identities, sample_manifest = sample_pair_observations(filtered, anatomy, p.lag_frames, s.confirm_samples, s.seed)
    screen = observations[:, :s.screen_samples]
    quiet_previous = filtered[:quiet_count-p.lag_frames]
    quiet_current = filtered[p.lag_frames:quiet_count]
    adaptive_cfg = methods.adaptive_binary_difference
    alpha, alpha_diagnostics = estimate_quiet_gain(
        quiet_previous, quiet_current, alpha_min=float(adaptive_cfg["alpha_min"]),
        alpha_max=float(adaptive_cfg["alpha_max"]), trim_fraction=float(adaptive_cfg["trim_fraction"]),
        refinement_iterations=int(adaptive_cfg["refinement_iterations"]),
    )
    lanes: dict[str, dict[str, Any]] = {}

    def add(method_id: str, activity: np.ndarray, fit: dict[str, Any], started: float) -> None:
        z, mask, calibration = _calibrate(activity, quiet_count, config)
        lanes[method_id] = {"continuous": activity.astype(np.float32), "positive_z": np.maximum(z, 0),
                            "binary_mask": mask, "fit": fit, "diagnostics": {"calibration": calibration,
                            "motion_correction": False, "nonzero_fraction": float(mask.mean())},
                            "timing": {"fit_and_apply_seconds": time.perf_counter() - started,
                                       "mean_per_frame_ms": (time.perf_counter()-started) * 1000 / len(filtered)}}

    if methods.fixed_binary_difference["enabled"]:
        started = time.perf_counter(); activity = fixed_difference(filtered, p.lag_frames)
        add("fixed_binary_difference", activity, {"status":"resolved", "lag_frames":p.lag_frames,
            "assumption":"equal shared background", "motion_correction":False}, started)
    if methods.adaptive_binary_difference["enabled"]:
        started = time.perf_counter(); activity = adaptive_difference(filtered, alpha, p.lag_frames)
        add("adaptive_binary_difference", activity, {"status":"resolved", "alpha_quiet":alpha,
            "gain_diagnostics":alpha_diagnostics, "lag_frames":p.lag_frames, "motion_correction":False}, started)

    whitened_confirm, whitening = center_and_whiten_2d(observations)
    whitened_screen = whitening.whitening @ (screen - whitening.mean[:, None])
    derivative = observations[1] - observations[0]; common = observations[1] + observations[0]
    ica_specs = []
    if methods.infomax_tanh_ica["enabled"]:
        cfg = methods.infomax_tanh_ica; started = time.perf_counter()
        fit = fit_infomax_tanh_ica(whitened_confirm, initial_angles_degrees=cfg["initial_angles_degrees"],
                                  max_iterations=int(cfg["max_iterations"]), learning_rate=float(cfg["learning_rate"]),
                                  tolerance=float(cfg["tolerance"])) if whitening.identifiable else None
        ica_specs.append(("infomax_tanh_ica", fit, started))
    if methods.cs_parzen_ica["enabled"]:
        cfg = methods.cs_parzen_ica; started = time.perf_counter()
        fit = fit_cs_parzen_ica(whitened_screen, whitened_confirm, bandwidth=float(cfg["bandwidth"]),
            block_rows=int(cfg["kernel_block_rows"]), screen_step_degrees=s.screen_angle_step_degrees,
            refine_half_width_degrees=s.refine_half_width_degrees, refine_step_degrees=s.refine_angle_step_degrees) if whitening.identifiable else None
        ica_specs.append(("cs_parzen_ica", fit, started))
    for method_id, fit, started in ica_specs:
        if fit is None:
            lanes[method_id] = {"continuous_components": None, "binary_mask": None,
                "fit":{"status":"unidentifiable", "condition_number":whitening.condition_number,
                       "artifact_omissions":["continuous_activity.npy","binary_mask.npy","binary_mask.tif"]},
                "diagnostics":{"component_selection":"unresolved", "motion_correction":False},
                "timing":{"fit_and_apply_seconds":time.perf_counter()-started}}
            continue
        sample_components = fit.demixing @ whitened_confirm
        _, component, signs, selection = orient_and_select_activity_component(sample_components, derivative, common)
        fit_payload = {"status":"resolved" if component is not None else "unresolved_component",
            "mean":whitening.mean.tolist(), "covariance":whitening.covariance.tolist(),
            "whitening":whitening.whitening.tolist(), "demixing":fit.demixing.tolist(),
            "mixing":None if fit.mixing is None else fit.mixing.tolist(), "objective_value":fit.objective,
            "converged":fit.converged, "iterations":fit.iterations, "activity_component":component,
            "activity_sign":None if component is None else signs[component], "component_selection":selection,
            "condition_number":whitening.condition_number, "diagnostics":fit.diagnostics}
        if component is None:
            fit_payload["artifact_omissions"] = ["continuous_activity.npy", "binary_mask.npy", "binary_mask.tif"]
            lanes[method_id] = {"continuous_components":None, "binary_mask":None, "fit":fit_payload,
                "diagnostics":{"component_selection":"unresolved", "motion_correction":False},
                "timing":{"fit_and_apply_seconds":time.perf_counter()-started}}
        else:
            activity = _apply_ica(
                filtered,
                whitening,
                fit,
                component,
                signs,
                config.resources.frame_chunk,
                p.lag_frames,
            )
            add(method_id, activity, fit_payload, started)

    if methods.shared_background_nmf["enabled"]:
        cfg = methods.shared_background_nmf; started = time.perf_counter()
        quiet_scale = max(float(np.percentile(filtered[:quiet_count], 99.5)), 1e-6)
        activity = np.zeros_like(filtered, dtype=np.float32); objectives = []; violations = 0
        for index in range(p.lag_frames, len(filtered)):
            nmf = fit_shared_background_nmf(np.maximum(filtered[index-p.lag_frames]/quiet_scale,0),
                np.maximum(filtered[index]/quiet_scale,0), alpha, activity_l1=float(cfg["activity_l1"]),
                max_iterations=int(cfg["max_iterations"]), tolerance=float(cfg["tolerance"]))
            activity[index] = nmf.activity; objectives.append(nmf.objectives[-1]); violations += nmf.diagnostics["monotonicity_violations"]
        positive_adaptive = np.maximum(adaptive_difference(filtered/quiet_scale, alpha, p.lag_frames), 0)
        correlation = float(np.corrcoef(activity.ravel(), positive_adaptive.ravel())[0,1]) if np.std(activity) and np.std(positive_adaptive) else 1.0
        nmad = float(np.mean(np.abs(activity-positive_adaptive))/max(np.mean(np.abs(positive_adaptive)),1e-12))
        add("shared_background_nmf", activity, {"status":"resolved", "alpha_quiet":alpha,
            "activity_l1":float(cfg["activity_l1"]), "objective_last_mean":float(np.mean(objectives)),
            "monotonicity_violations":violations, "positive_adaptive_correlation":correlation,
            "normalized_mean_absolute_difference":nmad,
            "equivalent_to_adaptive_residual":bool(correlation>0.995 and nmad<0.05)}, started)
    sample_manifest.update({"identities":identities, "anatomy":anatomy_summary, "alpha_quiet":alpha,
                            "alpha_diagnostics":alpha_diagnostics})
    return lanes, sample_manifest
