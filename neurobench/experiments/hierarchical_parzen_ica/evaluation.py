"""Bounded Stage-1 synthetic evaluation and explicit checkpoint artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from neurobench.algorithms.hierarchical_parzen_ica import ParzenDictionaryConfig
from neurobench.experiments.hierarchical_parzen_ica.stage1 import (
    STAGE1_METHODS,
    fit_stage1_lane,
)
from neurobench.experiments.hierarchical_parzen_ica.synthetic import (
    STAGE1_SYNTHETIC_CASES,
    Stage1SyntheticCase,
    stage1_synthetic_suite,
)


def _energy(reference: np.ndarray) -> float:
    return max(float(np.sum(np.asarray(reference, dtype=np.float64) ** 2)), 1e-15)


def _nmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sum((reference - estimate) ** 2) / _energy(reference))


def _projection_gain(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sum(reference * estimate) / _energy(reference))


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _method_kwargs(method_id: str, seed: int) -> dict[str, Any]:
    if method_id == "batch_cs_parzen_pairwise":
        return {
            "batch_cs_parzen": {
                "bandwidth": 0.35,
                "block_rows": 64,
                "screen_step_degrees": 15.0,
                "refine_half_width_degrees": 2.0,
                "refine_step_degrees": 1.0,
            }
        }
    if method_id == "stochastic_parzen_score_pairwise":
        return {
            "stochastic_dictionary": ParzenDictionaryConfig(
                maximum_centers=32,
                minimum_center_separation=0.05,
                bandwidth=0.35,
                bandwidth_min=0.1,
                bandwidth_max=1.0,
                update_rate=0.01,
                replacement_policy="farthest_center",
                warmup_samples=128,
                seed=int(seed),
            ),
            "stochastic_fit": {
                "learning_rate": 0.0002,
                "gradient_clip": 5.0,
                "maximum_angle_update_degrees": 0.25,
                "batch_size": 128,
                "maximum_iterations": 25,
                "tolerance": 1e-5,
            },
        }
    return {}


def evaluate_stage1_synthetic_case(
    case: Stage1SyntheticCase,
    method_id: str,
) -> dict[str, Any]:
    """Fit one method/case pair and retain explicit B/S/A/N measurements."""
    if method_id not in STAGE1_METHODS:
        raise ValueError(f"unsupported Stage-1 method: {method_id}")
    started = time.perf_counter()
    fitted = fit_stage1_lane(
        case.observation,
        method_id,
        calibration_frame_count=case.calibration_frame_count,
        fit_sample_pixels=512,
        sample_seed=case.seed,
        covariance_mode="ordinary",
        staticness={"minimum_confidence_margin": 0.1},
        **_method_kwargs(method_id, case.seed),
    )
    elapsed = time.perf_counter() - started
    lag = int(fitted.diagnostics["lag_frames"])
    background = case.background[lag:]
    signal = case.signal[lag:]
    artifact = case.artifact[lag:]
    noise = case.noise[lag:]
    dynamic = signal + artifact + noise
    estimated_background = fitted.result.background.astype(np.float64)
    residual = fitted.result.dynamic_residual.astype(np.float64)
    saved_closure = (
        case.observation[lag:].astype(np.float64)
        - estimated_background
        - residual
    )
    signal_present = _energy(signal) > 1e-12
    artifact_present = _energy(artifact) > 1e-12
    background_present = _energy(background) > 1e-12
    cleaned_signal_estimate = residual - artifact - noise
    cleaned_artifact_estimate = residual - signal - noise
    safety = fitted.diagnostics["safety"]
    anchoring = safety["reference_anchoring"]
    raw_feedback = None if anchoring is None else anchoring["raw_feedback"]
    learned_fraction = (
        None if anchoring is None else anchoring["accepted_learned_fraction"]
    )
    feedback = safety["feedback"]
    observation_rms = max(
        float(np.sqrt(np.mean(case.observation[lag:] ** 2))), 1e-15
    )
    return {
        "case_id": case.case_id,
        "seed": case.seed,
        "method_id": method_id,
        "run_status": "completed",
        "elapsed_seconds": elapsed,
        "classification_status": fitted.result.classification_status,
        "expected_unresolved": bool(
            case.metadata.get("expected_unresolved", False)
        ),
        "unresolved_expectation_met": (
            fitted.result.classification_status == "unresolved"
            if case.metadata.get("expected_unresolved", False) else None
        ),
        "background_component": fitted.result.background_component,
        "staticness_margin": fitted.result.confidence,
        "whitening_identifiable": fitted.whitening.identifiable,
        "whitening_condition_number": fitted.whitening.condition_number,
        "optimizer_converged": fitted.demixing_fit.converged,
        "optimizer_iterations": fitted.demixing_fit.iterations,
        "optimizer_updates": fitted.demixing_fit.update_count,
        "safety_status": safety["status"],
        "fallback_reasons": safety["fallback_reasons"],
        "accepted_learned_fraction": learned_fraction,
        "raw_feedback_safe": None if raw_feedback is None else raw_feedback["safe"],
        "raw_previous_coefficient": (
            None if raw_feedback is None
            else raw_feedback["previous_background_coefficient"]
        ),
        "raw_current_coefficient": (
            None if raw_feedback is None
            else raw_feedback["current_observation_coefficient"]
        ),
        "applied_feedback_safe": (
            None if feedback is None else feedback["safe"]
        ),
        "applied_previous_coefficient": (
            None if feedback is None
            else feedback["previous_background_coefficient"]
        ),
        "applied_current_coefficient": (
            None if feedback is None
            else feedback["current_observation_coefficient"]
        ),
        "background_present": background_present,
        "background_nmse": (
            _nmse(background, estimated_background)
            if background_present else None
        ),
        "dynamic_residual_nmse": _nmse(dynamic, residual),
        "signal_present": signal_present,
        "signal_nmse_after_known_artifact_noise_removal": (
            _nmse(signal, cleaned_signal_estimate)
            if signal_present else None
        ),
        "signal_amplitude_gain": (
            _projection_gain(signal, cleaned_signal_estimate)
            if signal_present else None
        ),
        "signal_leakage_gain_into_background": (
            _projection_gain(signal, estimated_background - background)
            if signal_present else None
        ),
        "artifact_present": artifact_present,
        "artifact_amplitude_gain": (
            _projection_gain(artifact, cleaned_artifact_estimate)
            if artifact_present else None
        ),
        "output_background_rms_ratio": float(
            np.sqrt(np.mean(estimated_background**2)) / observation_rms
        ),
        "output_residual_rms_ratio": float(
            np.sqrt(np.mean(residual**2)) / observation_rms
        ),
        "closure_max_absolute": float(np.max(np.abs(saved_closure))),
        "closure_normalized_squared_error": float(
            np.sum(saved_closure**2)
            / _energy(case.observation[lag:])
        ),
        "labels_used_for_fit": False,
    }


def _numeric(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = [
        float(row[key])
        for row in rows
        if row.get("run_status") == "completed" and row.get(key) is not None
    ]
    return np.asarray(values, dtype=np.float64)


def summarize_stage1_synthetic(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize method and case distributions without hiding failed runs."""
    by_method: dict[str, Any] = {}
    for method_id in sorted({str(row["method_id"]) for row in rows}):
        selected = [row for row in rows if row["method_id"] == method_id]
        completed = [row for row in selected if row["run_status"] == "completed"]
        signal_nmse = _numeric(
            completed, "signal_nmse_after_known_artifact_noise_removal"
        )
        background_nmse = _numeric(completed, "background_nmse")
        by_method[method_id] = {
            "attempts": len(selected),
            "completed": len(completed),
            "errors": len(selected) - len(completed),
            "resolved": sum(
                row["classification_status"] == "resolved"
                for row in completed
            ),
            "optimizer_converged": sum(
                bool(row["optimizer_converged"]) for row in completed
            ),
            "reference_fallbacks": sum(
                row["safety_status"] == "reference_fallback"
                or row.get("accepted_learned_fraction") == 0.0
                for row in completed
            ),
            "raw_feedback_rejections": sum(
                row["raw_feedback_safe"] is False for row in completed
            ),
            "expected_unresolved_runs": sum(
                bool(row["expected_unresolved"]) for row in completed
            ),
            "unresolved_expectations_met": sum(
                row["unresolved_expectation_met"] is True
                for row in completed
            ),
            "median_signal_nmse": (
                float(np.median(signal_nmse)) if signal_nmse.size else None
            ),
            "worst_signal_nmse": (
                float(np.max(signal_nmse)) if signal_nmse.size else None
            ),
            "median_background_nmse": (
                float(np.median(background_nmse))
                if background_nmse.size else None
            ),
            "worst_background_nmse": (
                float(np.max(background_nmse))
                if background_nmse.size else None
            ),
            "maximum_closure_absolute": (
                float(np.max(_numeric(completed, "closure_max_absolute")))
                if completed else None
            ),
        }
    by_case_method: dict[str, Any] = {}
    for case_id in sorted({str(row["case_id"]) for row in rows}):
        by_case_method[case_id] = {}
        for method_id in sorted({str(row["method_id"]) for row in rows}):
            selected = [
                row for row in rows
                if row["case_id"] == case_id
                and row["method_id"] == method_id
                and row["run_status"] == "completed"
            ]
            by_case_method[case_id][method_id] = {
                "runs": len(selected),
                "fallbacks": sum(
                    row["safety_status"] == "reference_fallback"
                    or row.get("accepted_learned_fraction") == 0.0
                    for row in selected
                ),
                "median_background_nmse": (
                    float(np.median(_numeric(selected, "background_nmse")))
                    if _numeric(selected, "background_nmse").size else None
                ),
                "median_signal_nmse": (
                    float(np.median(_numeric(
                        selected,
                        "signal_nmse_after_known_artifact_noise_removal",
                    )))
                    if _numeric(
                        selected,
                        "signal_nmse_after_known_artifact_noise_removal",
                    ).size else None
                ),
            }
    completed = [row for row in rows if row["run_status"] == "completed"]
    numerical_pass = bool(
        len(completed) == len(rows)
        and all(
            row["applied_feedback_safe"] in {True, None}
            for row in completed
        )
        and max(
            (float(row["closure_max_absolute"]) for row in completed),
            default=np.inf,
        ) < 1e-5
        and max(
            (
                max(
                    float(row["output_background_rms_ratio"]),
                    float(row["output_residual_rms_ratio"]),
                )
                for row in completed
            ),
            default=np.inf,
        ) < 10
    )
    learned_ids = {
        "batch_cs_parzen_pairwise",
        "stochastic_parzen_score_pairwise",
    }
    learned_nonfallback = [
        row for row in completed
        if row["method_id"] in learned_ids
        and row["safety_status"] == "accepted"
    ]
    scientific_pass = bool(
        numerical_pass
        and learned_nonfallback
        and all(
            not row["expected_unresolved"]
            or row["unresolved_expectation_met"] is True
            for row in completed
        )
        and all(
            row["signal_nmse_after_known_artifact_noise_removal"] is None
            or float(
                row["signal_nmse_after_known_artifact_noise_removal"]
            ) <= 0.1
            for row in learned_nonfallback
        )
    )
    return {
        "combination_count": len(rows),
        "completed_count": len(completed),
        "method_summary": by_method,
        "case_method_summary": by_case_method,
        "gates": {
            "stage1_numerical_stability": (
                "pass" if numerical_pass else "fail"
            ),
            "stage1_scientific_validity": (
                "pass" if scientific_pass else "fail"
            ),
            "scientific_gate_note": (
                "A numerical fallback is not learned-method improvement."
            ),
        },
    }


def _report_markdown(
    summary: dict[str, Any],
    *,
    seeds: tuple[int, ...],
    case_ids: tuple[str, ...],
    methods: tuple[str, ...],
) -> str:
    lines = [
        "# Stage-1 guarded synthetic matrix",
        "",
        "This report uses generated arrays only. It does not use Spon data or labels.",
        "",
        f"- Seeds: `{list(seeds)}`",
        f"- Cases: `{len(case_ids)}`",
        f"- Methods: `{len(methods)}`",
        f"- Exact combinations: `{summary['combination_count']}`",
        f"- Completed: `{summary['completed_count']}`",
        "",
        "## Gate status",
        "",
        (
            "- Stage-1 numerical stability: "
            f"**{summary['gates']['stage1_numerical_stability']}**"
        ),
        (
            "- Stage-1 scientific validity: "
            f"**{summary['gates']['stage1_scientific_validity']}**"
        ),
        "- A reference fallback is safe behavior, not learned-method success.",
        "",
        "## Method summary",
        "",
        (
            "| Method | Runs | Errors | Converged | Reference fallbacks | "
            "Raw feedback rejected | Median signal NMSE | Worst signal NMSE |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method_id in methods:
        item = summary["method_summary"][method_id]
        median = item["median_signal_nmse"]
        worst = item["worst_signal_nmse"]
        lines.append(
            f"| {method_id} | {item['completed']} | {item['errors']} | "
            f"{item['optimizer_converged']} | {item['reference_fallbacks']} | "
            f"{item['raw_feedback_rejections']} | "
            f"{'n/a' if median is None else f'{median:.6g}'} | "
            f"{'n/a' if worst is None else f'{worst:.6g}'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Read `results.tsv` for every case/seed/method row and `summary.json` "
            "for the full per-case aggregation. Signal NMSE is calculated only "
            "when a known synthetic neural signal exists. Motion and clipping "
            "remain artifacts, not neural positives.",
            "",
        ]
    )
    return "\n".join(lines)


def run_stage1_synthetic_matrix(
    output_dir: str | Path,
    *,
    seeds: Iterable[int] = (7, 13, 19, 29, 37),
    case_ids: Iterable[str] = STAGE1_SYNTHETIC_CASES,
    methods: Iterable[str] = (
        "fixed_common_difference_reference",
        "adaptive_gain_common_difference",
        "batch_cs_parzen_pairwise",
        "stochastic_parzen_score_pairwise",
    ),
) -> dict[str, Any]:
    """Run a new collision-safe CPU synthetic matrix and write concise evidence."""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    normalized_seeds = tuple(int(seed) for seed in seeds)
    normalized_cases = tuple(str(case_id) for case_id in case_ids)
    normalized_methods = tuple(str(method_id) for method_id in methods)
    suite = stage1_synthetic_suite(
        normalized_seeds, case_ids=normalized_cases
    )
    expected = len(suite) * len(normalized_methods)
    manifest = {
        "schema_version": 1,
        "experiment_id": destination.name,
        "kind": "generated_stage1_synthetic_matrix",
        "axes": "TYX",
        "seeds": list(normalized_seeds),
        "case_ids": list(normalized_cases),
        "methods": list(normalized_methods),
        "combination_count": expected,
        "labels_used": False,
        "real_data_used": False,
        "device": "cpu",
        "fit_sample_pixels": 512,
        "safety_policy": "reference_anchored_feedback_gate",
    }
    _atomic_text(
        destination / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    rows: list[dict[str, Any]] = []
    progress_path = destination / "progress.jsonl"
    for index, case in enumerate(suite):
        for method_id in normalized_methods:
            try:
                row = evaluate_stage1_synthetic_case(case, method_id)
            except Exception as exc:
                row = {
                    "case_id": case.case_id,
                    "seed": case.seed,
                    "method_id": method_id,
                    "run_status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "labels_used_for_fit": False,
                }
            rows.append(row)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(
                    {
                        "completed": len(rows),
                        "total": expected,
                        "case_id": case.case_id,
                        "seed": case.seed,
                        "method_id": method_id,
                        "run_status": row["run_status"],
                    },
                    sort_keys=True,
                ) + "\n")
                handle.flush()
    summary = summarize_stage1_synthetic(rows)
    _atomic_text(
        destination / "results.json",
        json.dumps(rows, indent=2, sort_keys=True, default=_json_default) + "\n",
    )
    columns = sorted({key for row in rows for key in row})
    tsv_temporary = destination / "results.tsv.tmp"
    with tsv_temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    json.dumps(value, sort_keys=True, default=_json_default)
                    if isinstance(value, (dict, list, tuple)) else value
                )
                for key, value in row.items()
            })
    tsv_temporary.replace(destination / "results.tsv")
    _atomic_text(
        destination / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
    )
    _atomic_text(
        destination / "REPORT.md",
        _report_markdown(
            summary,
            seeds=normalized_seeds,
            case_ids=normalized_cases,
            methods=normalized_methods,
        ),
    )
    return {
        "output_dir": str(destination),
        "combination_count": expected,
        "summary": summary,
    }
