"""Frozen generated W5c confirmation-authority comparison."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from neurobench.algorithms.dependent_multiscale import ScaleViewSpec, build_scale_views
from neurobench.metrics.multiscale_decomposition import attribution_metrics, closure_metrics

from .dependent_multiscale_confirmation import apply_confirmation_authority, build_confirmation_maps
from .dependent_multiscale_evaluation import _channels, _preservation, _truth
from .dependent_multiscale_population import population_preserving_movie
from .dependent_multiscale_population_evaluation import REQUIRED_C2_CASES
from .dependent_multiscale_synthetic import FIXTURE_IDS, make_fixture


LANES = (
    "orthogonal_patchwise",
    "population_w5b",
    "coherence_confirmed",
    "carrier_constrained",
    "coherence_carrier_combined",
)
PRIMARY_LANE = "coherence_carrier_combined"


def _specs() -> tuple[ScaleViewSpec, ...]:
    return tuple(
        ScaleViewSpec(f"scale_{support}", support, "normalized_box_support", "none", {"nested": True})
        for support in (5, 7, 15)
    )


def evaluate_confirmation_generated_matrix(*, seeds: tuple[int, ...] = (7, 13, 19)) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        for fixture_id in FIXTURE_IDS:
            fixture = make_fixture(fixture_id, seed=seed)
            views = build_scale_views(fixture.observation, _specs(), quiet_count=8)
            baseline = population_preserving_movie(
                fixture.observation, views, patch_px=15, stride_px=10,
                population_gain=0, residual_recapture_authority=0,
            ).decomposition
            population = population_preserving_movie(
                fixture.observation, views, patch_px=15, stride_px=10,
            ).decomposition
            confirmation = build_confirmation_maps(fixture.observation, views, quiet_count=8)
            decompositions = {
                "orthogonal_patchwise": baseline,
                "population_w5b": population,
                **{
                    lane: apply_confirmation_authority(baseline, population, confirmation, lane_id=lane)
                    for lane in LANES[2:]
                },
            }
            truth = _truth(fixture)
            lane_values = {}
            for lane, decomposition in decompositions.items():
                lane_values[lane] = {
                    "attribution": attribution_metrics(truth, _channels(decomposition)),
                    "preservation": _preservation(fixture.structured_signal, decomposition.structured_signal),
                    "closure": closure_metrics(fixture.observation, _channels(decomposition)),
                    "diagnostics": decomposition.diagnostics,
                }
            rows.append({"fixture_id": fixture_id, "seed": int(seed), "lanes": lane_values})
    baseline_leakage = np.asarray([
        row["lanes"]["orthogonal_patchwise"]["attribution"]["primary_signal_leakage"]
        for row in rows
    ])
    baseline_diagonality = np.asarray([
        row["lanes"]["orthogonal_patchwise"]["attribution"]["diagonality_margin"]
        for row in rows
    ])
    lane_summaries = {}
    for lane in LANES[1:]:
        leakage = np.asarray([row["lanes"][lane]["attribution"]["primary_signal_leakage"] for row in rows])
        diagonality = np.asarray([row["lanes"][lane]["attribution"]["diagonality_margin"] for row in rows])
        preservation = [
            row["lanes"][lane]["preservation"] for row in rows
            if row["lanes"][lane]["preservation"] is not None
        ]
        subgroup = {}
        for fixture_id in FIXTURE_IDS:
            values = [
                row["lanes"][lane]["preservation"] for row in rows
                if row["fixture_id"] == fixture_id and row["lanes"][lane]["preservation"] is not None
            ]
            if values:
                subgroup[fixture_id] = {
                    "peak": float(np.median([value["peak_amplitude_ratio"] for value in values])),
                    "area": float(np.median([value["temporal_area_ratio"] for value in values])),
                    "peak_time_error": float(np.median([value["peak_time_error_frames"] for value in values])),
                }
        improvement = float(
            (np.median(baseline_leakage) - np.median(leakage))
            / max(float(np.median(baseline_leakage)), 1e-12)
        )
        required_cases = {}
        for fixture_id in REQUIRED_C2_CASES:
            selected = [row for row in rows if row["fixture_id"] == fixture_id]
            before = np.median([
                row["lanes"]["orthogonal_patchwise"]["attribution"]["primary_signal_leakage"]
                for row in selected
            ])
            after = np.median([row["lanes"][lane]["attribution"]["primary_signal_leakage"] for row in selected])
            required_cases[fixture_id] = float((before - after) / max(float(before), 1e-12))
        peak = float(np.median([value["peak_amplitude_ratio"] for value in preservation]))
        area = float(np.median([value["temporal_area_ratio"] for value in preservation]))
        timing = float(np.median([value["peak_time_error_frames"] for value in preservation]))
        timing_p95 = float(np.quantile([value["peak_time_error_frames"] for value in preservation], 0.95))
        subgroup_pass = all(0.80 <= value["peak"] <= 1.20 and 0.75 <= value["area"] <= 1.25 for value in subgroup.values())
        aggregate_pass = 0.90 <= peak <= 1.10 and 0.85 <= area <= 1.15 and timing <= 1 and timing_p95 <= 3
        c2 = bool(
            improvement >= 0.05
            and np.median(diagonality) > np.median(baseline_diagonality)
            and all(value >= 0.05 for value in required_cases.values())
        )
        lane_summaries[lane] = {
            "median_signal_leakage": float(np.median(leakage)),
            "relative_leakage_improvement": improvement,
            "median_diagonality": float(np.median(diagonality)),
            "median_peak_ratio": peak,
            "median_area_ratio": area,
            "median_peak_time_error": timing,
            "p95_peak_time_error": timing_p95,
            "required_case_improvement": required_cases,
            "preservation_by_fixture": subgroup,
            "C2_pass": c2,
            "C3_aggregate_pass": bool(aggregate_pass),
            "C3_subgroup_pass": bool(subgroup_pass),
            "C3_pass": bool(aggregate_pass and subgroup_pass),
        }
    closure_max = max(
        row["lanes"][lane]["closure"]["normalized_maximum"]
        for row in rows for lane in LANES
    )
    primary = lane_summaries[PRIMARY_LANE]
    c1 = closure_max <= 1e-4
    return {
        "schema_version": 1,
        "status": "completed_generated_w5c",
        "primary_lane": PRIMARY_LANE,
        "lanes": list(LANES),
        "fixture_count": len(FIXTURE_IDS),
        "seed_count": len(seeds),
        "evaluation_count": len(rows) * len(LANES),
        "quiet_calibration_only": True,
        "labels_used_for_fit": False,
        "baseline": {
            "median_signal_leakage": float(np.median(baseline_leakage)),
            "median_diagonality": float(np.median(baseline_diagonality)),
        },
        "lane_summaries": lane_summaries,
        "maximum_normalized_closure": float(closure_max),
        "gates": {
            "C1_numerical_reconstruction": "pass" if c1 else "fail",
            "C2_generated_attribution": "pass" if primary["C2_pass"] else "fail",
            "C3_signal_preservation": "pass" if primary["C3_pass"] else "fail",
            "C4_residual_qualification": "not_qualified",
            "C5_stability": "diagnostic_only",
        },
        "advance_to_W6": bool(c1 and primary["C2_pass"] and primary["C3_pass"]),
        "rows": rows,
    }


def write_confirmation_generated_evaluation(output_dir: str | Path) -> dict[str, Any]:
    target = Path(output_dir).resolve()
    partial = Path(str(target) + ".partial")
    if target.exists() or partial.exists():
        raise FileExistsError("W5c output or partial root exists")
    partial.mkdir(parents=True)
    started = time.time()
    result = evaluate_confirmation_generated_matrix()
    result["elapsed_seconds"] = time.time() - started
    (partial / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    primary = result["lane_summaries"][PRIMARY_LANE]
    lines = [
        "# Confirmation-authority W5c generated gate", "",
        f"Decision: **{'advance to W6' if result['advance_to_W6'] else 'do not advance'}**.", "",
        f"Primary lane: `{PRIMARY_LANE}`.",
        f"C1/C2/C3: {result['gates']['C1_numerical_reconstruction']} / {result['gates']['C2_generated_attribution']} / {result['gates']['C3_signal_preservation']}.",
        f"Primary leakage improvement: {primary['relative_leakage_improvement']:.4f}.",
        f"Primary peak/area medians: {primary['median_peak_ratio']:.4f} / {primary['median_area_ratio']:.4f}.",
        f"Primary aggregate/subgroup C3: {primary['C3_aggregate_pass']} / {primary['C3_subgroup_pass']}.", "",
        "All authority maps use quiet calibration only. Sparse-positive labels were not used for fitting or threshold selection.",
    ]
    (partial / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    partial.replace(target)
    return result
