"""Generated W5 comparison and scientific gates for dependent multiscale."""
from __future__ import annotations

from typing import Any

import numpy as np

from neurobench.algorithms.dependent_multiscale import ScaleViewSpec, build_scale_views, decompose_patch_baseline
from neurobench.metrics.multiscale_decomposition import attribution_metrics, closure_metrics

from .dependent_multiscale_information import build_frame_nuisance, refine_group_dependence
from .dependent_multiscale_synthetic import FIXTURE_IDS, make_fixture


def _specs() -> tuple[ScaleViewSpec, ...]:
    return tuple(
        ScaleViewSpec(
            f"scale_{support}", support, "normalized_box_support", "none",
            {"nested": True, "padding": "reflect"},
        )
        for support in (5, 7, 15)
    )


def _channels(decomposition) -> dict[str, np.ndarray]:
    return {
        "background": decomposition.background,
        "structured_signal": decomposition.structured_signal,
        "structured_artifact": decomposition.structured_artifact,
        "noise_candidate": decomposition.noise_candidate,
    }


def _truth(fixture) -> dict[str, np.ndarray]:
    return {
        "background": fixture.background,
        "structured_signal": fixture.structured_signal,
        "structured_artifact": fixture.structured_artifact,
        "noise_candidate": fixture.noise,
    }


def _preservation(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float] | None:
    true = np.asarray(truth, dtype=np.float64)
    predicted = np.asarray(estimate, dtype=np.float64)
    if not np.any(np.abs(true) > 1e-8):
        return None
    true_trace = np.sum(np.maximum(true, 0), axis=(1, 2))
    estimate_trace = np.sum(np.maximum(predicted, 0), axis=(1, 2))
    peak_ratio = float(np.max(estimate_trace) / max(np.max(true_trace), 1e-12))
    area_ratio = float(np.sum(estimate_trace) / max(np.sum(true_trace), 1e-12))
    peak_error = abs(int(np.argmax(estimate_trace)) - int(np.argmax(true_trace)))
    return {
        "peak_amplitude_ratio": peak_ratio,
        "temporal_area_ratio": area_ratio,
        "peak_time_error_frames": float(peak_error),
    }


def evaluate_generated_matrix(
    *, seeds: tuple[int, ...] = (7, 13, 19), reference_authority: float = 0.5,
) -> dict[str, Any]:
    """Compare independent structure, orthogonal baseline, and W4 refinement."""
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for fixture_id in FIXTURE_IDS:
            fixture = make_fixture(fixture_id, seed=seed)
            views = build_scale_views(fixture.observation, _specs(), quiet_count=8)
            baseline = decompose_patch_baseline(
                fixture.observation, views, patch_id=f"{fixture_id}__seed{seed}"
            )
            nuisance, nuisance_names = build_frame_nuisance(fixture.observation)
            reference = refine_group_dependence(
                baseline, observation=fixture.observation, nuisance=nuisance,
                authority=reference_authority,
            )
            nearby = refine_group_dependence(
                baseline, observation=fixture.observation, nuisance=nuisance,
                authority=min(1.0, 1.5 * reference_authority),
            )
            truth = _truth(fixture)
            lane_values = {
                "orthogonal_shared_private": baseline,
                "dependent_groups_only": reference.decomposition,
                "dependent_groups_joint_quiet": nearby.decomposition,
            }
            lanes = {}
            for lane_id, decomposition in lane_values.items():
                lanes[lane_id] = {
                    "attribution": attribution_metrics(truth, _channels(decomposition)),
                    "closure": closure_metrics(fixture.observation, _channels(decomposition)),
                    "signal_preservation": _preservation(
                        fixture.structured_signal, decomposition.structured_signal
                    ),
                    "noise_name": "noise_candidate",
                }
            lanes["dependent_groups_only"]["objective_terms"] = reference.objective_terms
            lanes["dependent_groups_joint_quiet"]["objective_terms"] = nearby.objective_terms
            rows.append({
                "fixture_id": fixture_id,
                "seed": seed,
                "nuisance_variables": list(nuisance_names),
                "lanes": lanes,
            })
    baseline_leakage = np.asarray([
        row["lanes"]["orthogonal_shared_private"]["attribution"]["primary_signal_leakage"]
        for row in rows
    ])
    full_leakage = np.asarray([
        row["lanes"]["dependent_groups_joint_quiet"]["attribution"]["primary_signal_leakage"]
        for row in rows
    ])
    baseline_diagonality = np.asarray([
        row["lanes"]["orthogonal_shared_private"]["attribution"]["diagonality_margin"]
        for row in rows
    ])
    full_diagonality = np.asarray([
        row["lanes"]["dependent_groups_joint_quiet"]["attribution"]["diagonality_margin"]
        for row in rows
    ])
    closure_max = max(
        lane["closure"]["normalized_maximum"]
        for row in rows for lane in row["lanes"].values()
    )
    preservation = [
        row["lanes"]["dependent_groups_joint_quiet"]["signal_preservation"]
        for row in rows
        if row["lanes"]["dependent_groups_joint_quiet"]["signal_preservation"] is not None
    ]
    relative_improvement = float(
        (np.median(baseline_leakage) - np.median(full_leakage))
        / max(np.median(baseline_leakage), 1e-12)
    )
    c2 = bool(
        relative_improvement >= 0.05
        and np.median(full_diagonality) > np.median(baseline_diagonality)
    )
    median_peak = float(np.median([item["peak_amplitude_ratio"] for item in preservation]))
    median_area = float(np.median([item["temporal_area_ratio"] for item in preservation]))
    median_timing = float(np.median([item["peak_time_error_frames"] for item in preservation]))
    c3 = bool(0.90 <= median_peak <= 1.10 and 0.85 <= median_area <= 1.15 and median_timing <= 1)
    return {
        "schema_version": 1,
        "status": "completed_generated_w5",
        "fixture_count": len(FIXTURE_IDS),
        "seed_count": len(seeds),
        "evaluation_count": len(rows) * 3,
        "reference_authority": reference_authority,
        "nearby_authority": min(1.0, 1.5 * reference_authority),
        "rows": rows,
        "summary": {
            "maximum_normalized_closure": closure_max,
            "baseline_median_signal_leakage": float(np.median(baseline_leakage)),
            "full_median_signal_leakage": float(np.median(full_leakage)),
            "relative_signal_leakage_improvement": relative_improvement,
            "baseline_median_diagonality": float(np.median(baseline_diagonality)),
            "full_median_diagonality": float(np.median(full_diagonality)),
            "median_peak_amplitude_ratio": median_peak,
            "median_temporal_area_ratio": median_area,
            "median_peak_time_error_frames": median_timing,
        },
        "gates": {
            "C1_numerical_reconstruction": "pass" if closure_max <= 1e-4 else "fail",
            "C2_generated_attribution": "pass" if c2 else "fail",
            "C3_signal_preservation": "pass" if c3 else "fail",
            "C4_residual_qualification": "not_qualified",
            "C5_stability": "diagnostic_only",
        },
        "advance_to_real_scientific_run": bool(c2 and c3 and closure_max <= 1e-4),
    }
