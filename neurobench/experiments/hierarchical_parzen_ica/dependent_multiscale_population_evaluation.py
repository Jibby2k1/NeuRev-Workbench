"""Frozen W5b evaluation for population-preserving patchwise attribution."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from neurobench.algorithms.dependent_multiscale import ScaleViewSpec, build_scale_views
from neurobench.metrics.multiscale_decomposition import attribution_metrics, closure_metrics

from .dependent_multiscale_evaluation import _channels, _preservation, _truth
from .dependent_multiscale_population import population_preserving_movie
from .dependent_multiscale_synthetic import FIXTURE_IDS, make_fixture


REQUIRED_C2_CASES = (
    "compact_isolated_center",
    "broad_legitimate_neural_source",
    "two_correlated_neurons",
    "motion_edge_crossing_a_neuron",
    "heteroscedastic_shot_like_noise",
)


def _specs() -> tuple[ScaleViewSpec, ...]:
    return tuple(
        ScaleViewSpec(f"scale_{support}", support, "normalized_box_support", "none", {"nested": True})
        for support in (5, 7, 15)
    )


def evaluate_population_generated_matrix(*, seeds: tuple[int, ...] = (7, 13, 19)) -> dict[str, Any]:
    """Compare geometry-matched patchwise baseline and frozen population lane."""
    rows = []
    for seed in seeds:
        for fixture_id in FIXTURE_IDS:
            fixture = make_fixture(fixture_id, seed=seed)
            views = build_scale_views(fixture.observation, _specs(), quiet_count=8)
            baseline = population_preserving_movie(
                fixture.observation, views, patch_px=15, stride_px=10,
                population_gain=0.0, residual_recapture_authority=0.0,
            ).decomposition
            revised_result = population_preserving_movie(
                fixture.observation, views, patch_px=15, stride_px=10,
            )
            revised = revised_result.decomposition
            truth = _truth(fixture)
            rows.append({
                "fixture_id": fixture_id,
                "seed": int(seed),
                "baseline": {
                    "attribution": attribution_metrics(truth, _channels(baseline)),
                    "signal_preservation": _preservation(fixture.structured_signal, baseline.structured_signal),
                },
                "population_preserving": {
                    "attribution": attribution_metrics(truth, _channels(revised)),
                    "signal_preservation": _preservation(fixture.structured_signal, revised.structured_signal),
                    "closure": closure_metrics(fixture.observation, _channels(revised)),
                    "patch_count": revised_result.diagnostics["patch_count"],
                },
            })
    baseline_leakage = np.asarray([row["baseline"]["attribution"]["primary_signal_leakage"] for row in rows])
    revised_leakage = np.asarray([row["population_preserving"]["attribution"]["primary_signal_leakage"] for row in rows])
    baseline_diagonality = np.asarray([row["baseline"]["attribution"]["diagonality_margin"] for row in rows])
    revised_diagonality = np.asarray([row["population_preserving"]["attribution"]["diagonality_margin"] for row in rows])
    relative_improvement = float(
        (np.median(baseline_leakage) - np.median(revised_leakage))
        / max(float(np.median(baseline_leakage)), 1e-12)
    )
    case_improvements = {}
    for fixture_id in REQUIRED_C2_CASES:
        selected = [row for row in rows if row["fixture_id"] == fixture_id]
        before = np.median([row["baseline"]["attribution"]["primary_signal_leakage"] for row in selected])
        after = np.median([row["population_preserving"]["attribution"]["primary_signal_leakage"] for row in selected])
        case_improvements[fixture_id] = float((before - after) / max(float(before), 1e-12))
    preservation = [
        row["population_preserving"]["signal_preservation"] for row in rows
        if row["population_preserving"]["signal_preservation"] is not None
    ]
    subgroup = {}
    for fixture_id in FIXTURE_IDS:
        values = [
            row["population_preserving"]["signal_preservation"] for row in rows
            if row["fixture_id"] == fixture_id
            and row["population_preserving"]["signal_preservation"] is not None
        ]
        if values:
            subgroup[fixture_id] = {
                "median_peak_amplitude_ratio": float(np.median([item["peak_amplitude_ratio"] for item in values])),
                "median_temporal_area_ratio": float(np.median([item["temporal_area_ratio"] for item in values])),
                "median_peak_time_error_frames": float(np.median([item["peak_time_error_frames"] for item in values])),
            }
    median_peak = float(np.median([item["peak_amplitude_ratio"] for item in preservation]))
    median_area = float(np.median([item["temporal_area_ratio"] for item in preservation]))
    median_time = float(np.median([item["peak_time_error_frames"] for item in preservation]))
    p95_time = float(np.quantile([item["peak_time_error_frames"] for item in preservation], 0.95))
    subgroup_pass = all(
        0.80 <= item["median_peak_amplitude_ratio"] <= 1.20
        and 0.75 <= item["median_temporal_area_ratio"] <= 1.25
        for item in subgroup.values()
    )
    closure_max = max(row["population_preserving"]["closure"]["normalized_maximum"] for row in rows)
    c1 = closure_max <= 1e-4
    c2 = bool(
        relative_improvement >= 0.05
        and np.median(revised_diagonality) > np.median(baseline_diagonality)
        and all(value >= 0.05 for value in case_improvements.values())
    )
    c3_aggregate = bool(
        0.90 <= median_peak <= 1.10 and 0.85 <= median_area <= 1.15
        and median_time <= 1 and p95_time <= 3
    )
    c3 = bool(c3_aggregate and subgroup_pass)
    return {
        "schema_version": 1,
        "status": "completed_generated_w5b",
        "fixture_count": len(FIXTURE_IDS),
        "seed_count": len(seeds),
        "evaluation_count": len(rows) * 2,
        "frozen_parameters": {
            "patch_px": 15,
            "stride_px": 10,
            "population_window_frames": 23,
            "population_gain": 1.25,
            "residual_recapture_authority": 0.25,
            "maximum_positive_trace_gain": 2.25,
        },
        "summary": {
            "maximum_normalized_closure": float(closure_max),
            "baseline_median_signal_leakage": float(np.median(baseline_leakage)),
            "population_median_signal_leakage": float(np.median(revised_leakage)),
            "relative_signal_leakage_improvement": relative_improvement,
            "baseline_median_diagonality": float(np.median(baseline_diagonality)),
            "population_median_diagonality": float(np.median(revised_diagonality)),
            "median_peak_amplitude_ratio": median_peak,
            "median_temporal_area_ratio": median_area,
            "median_peak_time_error_frames": median_time,
            "p95_peak_time_error_frames": p95_time,
            "C3_aggregate_pass": c3_aggregate,
            "C3_subgroup_pass": subgroup_pass,
        },
        "required_case_leakage_improvement": case_improvements,
        "preservation_by_fixture": subgroup,
        "gates": {
            "C1_numerical_reconstruction": "pass" if c1 else "fail",
            "C2_generated_attribution": "pass" if c2 else "fail",
            "C3_signal_preservation": "pass" if c3 else "fail",
            "C4_residual_qualification": "not_qualified",
            "C5_stability": "diagnostic_only",
        },
        "advance_to_W6": bool(c1 and c2 and c3),
        "rows": rows,
    }


def write_population_generated_evaluation(output_dir: str | Path) -> dict[str, Any]:
    """Write a collision-safe W5b gate artifact."""
    target = Path(output_dir).resolve()
    partial = Path(str(target) + ".partial")
    if target.exists() or partial.exists():
        raise FileExistsError("W5b output or partial root exists")
    partial.mkdir(parents=True)
    started = time.time()
    try:
        result = evaluate_population_generated_matrix()
        result["elapsed_seconds"] = time.time() - started
        (partial / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = result["summary"]
        lines = [
            "# Population-preserving W5b generated gate", "",
            f"Decision: **{'advance to W6' if result['advance_to_W6'] else 'do not advance'}**.", "",
            f"- C1: {result['gates']['C1_numerical_reconstruction']}",
            f"- C2: {result['gates']['C2_generated_attribution']}",
            f"- C3: {result['gates']['C3_signal_preservation']}",
            f"- Leakage improvement: {summary['relative_signal_leakage_improvement']:.4f}",
            f"- Peak / area medians: {summary['median_peak_amplitude_ratio']:.4f} / {summary['median_temporal_area_ratio']:.4f}",
            f"- Aggregate preservation pass: {summary['C3_aggregate_pass']}",
            f"- Subgroup preservation pass: {summary['C3_subgroup_pass']}", "",
            "The subgroup gate is authoritative; aggregate medians cannot conceal morphology-specific attenuation or amplification.",
        ]
        (partial / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        partial.replace(target)
        return result
    except Exception:
        raise
