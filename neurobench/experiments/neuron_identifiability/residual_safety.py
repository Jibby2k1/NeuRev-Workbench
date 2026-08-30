"""Source-off-only safety policy for conditional-background residuals.

The policy never uses injected-source recovery, expert labels, or candidate
identity to decide whether a residual may replace the raw movie.  It collapses
repeated source-off diagnostics to one immutable decision per
method/background-window pair and falls back to the frozen raw endpoint when
any registered nuisance guard is violated.

This is an engineering guardrail.  Passing it does not establish denoising,
neuron identity, precision, or biological specificity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import stable_hash


RAW_METHOD = "raw_frozen_handcrafted_stack"
JEPA_RESIDUAL_METHOD = "jepa_conditional_pixel_residual_frozen_handcrafted_stack"
RANDOM_RESIDUAL_METHOD = "random_encoder_conditional_pixel_residual_frozen_handcrafted_stack"


class ResidualSafetyError(ValueError):
    """Raised when a source-off safety decision would be ambiguous."""


@dataclass(frozen=True)
class ResidualSafetyThresholds:
    """Conservative no-amplification guards frozen for future residuals.

    The seam tolerance allows small numerical/lattice imbalance while blocking
    the large boundary artifacts observed in EXP-0029 Run B.  These values were
    created after Run B and therefore must not be described as preregistered
    Run-B gates.
    """

    maximum_background_rms_ratio: float = 1.0
    maximum_dynamic_mad_ratio: float = 1.0
    maximum_seam_to_interior_jump_ratio: float = 1.25
    repeated_metric_absolute_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        values = (
            self.maximum_background_rms_ratio,
            self.maximum_dynamic_mad_ratio,
            self.maximum_seam_to_interior_jump_ratio,
        )
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ResidualSafetyError("safety thresholds must be finite and positive")
        if (
            not np.isfinite(self.repeated_metric_absolute_tolerance)
            or self.repeated_metric_absolute_tolerance < 0
        ):
            raise ResidualSafetyError("repeated-metric tolerance must be finite and nonnegative")

    def to_manifest(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "decision_inputs": "source_off_diagnostics_only",
            "unsafe_action": "use_frozen_raw_endpoint",
            "scientific_claim_consequence": "none",
            "historical_note": "defined_after_EXP_0029_Run_B_for_future_guardrail_use",
        }


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ResidualSafetyError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise ResidualSafetyError(f"{name} must be finite")
    return result


def _source_off_values(row: Mapping[str, Any]) -> tuple[str, str, str, tuple[float, float, float]]:
    try:
        method = str(row["method"])
        recording_id = str(row["background_recording_id"])
        window_id = str(row["background_window_id"])
        background = row["background"]
        seams = row["seams"]
    except (KeyError, TypeError) as exc:
        raise ResidualSafetyError("malformed residual diagnostic row") from exc
    if not method or not recording_id or not window_id:
        raise ResidualSafetyError("method, recording, and window identifiers are required")
    # EXP-0029's top-level seam value describes the source-on prediction and
    # can vary with injected morphology.  The decoder-reference record carries
    # the explicit source-off residual seam.  New callers may provide that
    # value directly as ``source_off_seams``; the legacy top-level fallback is
    # retained only for compact synthetic fixtures and future tables whose
    # schema explicitly declares source-off rows.
    source_off_seams = row.get("source_off_seams")
    if source_off_seams is None:
        try:
            source_off_seams = row["decoder_reference"]["seams"]["signed_residual_off"]
        except (KeyError, TypeError):
            source_off_seams = seams
    seam_value = (
        source_off_seams.get("seam_to_interior_jump_ratio")
        if "seam_to_interior_jump_ratio" in source_off_seams
        else source_off_seams.get("boundary_to_within_jump_ratio")
    )
    values = (
        _finite_float(background.get("background_rms_ratio"), name="background_rms_ratio"),
        _finite_float(background.get("dynamic_mad_ratio"), name="dynamic_mad_ratio"),
        _finite_float(
            seam_value,
            name="seam_to_interior_jump_ratio",
        ),
    )
    return method, recording_id, window_id, values


def collapse_source_off_diagnostics(
    rows: Iterable[Mapping[str, Any]],
    *,
    thresholds: ResidualSafetyThresholds = ResidualSafetyThresholds(),
) -> list[dict[str, Any]]:
    """Collapse fixture repetition to one verified row per method/window."""

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_row in rows:
        method, recording_id, window_id, values = _source_off_values(raw_row)
        key = (method, window_id)
        if key not in grouped:
            grouped[key] = {
                "method": method,
                "background_recording_id": recording_id,
                "background_window_id": window_id,
                "background_rms_ratio": values[0],
                "dynamic_mad_ratio": values[1],
                "seam_to_interior_jump_ratio": values[2],
                "repeated_fixture_rows": 1,
            }
            continue
        prior = grouped[key]
        if prior["background_recording_id"] != recording_id:
            raise ResidualSafetyError("one window maps to multiple recording identifiers")
        prior_values = (
            prior["background_rms_ratio"],
            prior["dynamic_mad_ratio"],
            prior["seam_to_interior_jump_ratio"],
        )
        if any(
            abs(float(observed) - float(expected))
            > thresholds.repeated_metric_absolute_tolerance
            for observed, expected in zip(values, prior_values, strict=True)
        ):
            raise ResidualSafetyError(
                f"source-off diagnostics drift across repeated fixtures for {method}/{window_id}"
            )
        prior["repeated_fixture_rows"] += 1
    if not grouped:
        raise ResidualSafetyError("no residual diagnostic rows were supplied")
    return [grouped[key] for key in sorted(grouped)]


def evaluate_residual_safety(
    rows: Iterable[Mapping[str, Any]],
    *,
    thresholds: ResidualSafetyThresholds = ResidualSafetyThresholds(),
) -> dict[str, Any]:
    """Return a label-free, one-decision-per-window residual policy."""

    collapsed = collapse_source_off_diagnostics(rows, thresholds=thresholds)
    decisions: list[dict[str, Any]] = []
    for row in collapsed:
        checks = {
            "background_rms_not_amplified": bool(
                row["background_rms_ratio"] <= thresholds.maximum_background_rms_ratio
            ),
            "dynamic_mad_not_amplified": bool(
                row["dynamic_mad_ratio"] <= thresholds.maximum_dynamic_mad_ratio
            ),
            "spatial_seam_within_tolerance": bool(
                row["seam_to_interior_jump_ratio"]
                <= thresholds.maximum_seam_to_interior_jump_ratio
            ),
        }
        safe = all(checks.values())
        decisions.append(
            {
                **row,
                "checks": checks,
                "safe_for_residual_use": safe,
                "selected_endpoint": row["method"] if safe else RAW_METHOD,
                "decision_uses_source_truth_or_labels": False,
            }
        )
    methods: dict[str, dict[str, Any]] = {}
    for method in sorted({row["method"] for row in decisions}):
        subset = [row for row in decisions if row["method"] == method]
        safe_count = sum(bool(row["safe_for_residual_use"]) for row in subset)
        methods[method] = {
            "window_count": len(subset),
            "safe_window_count": safe_count,
            "safe_window_fraction": safe_count / len(subset),
            "all_windows_safe": safe_count == len(subset),
        }
    result: dict[str, Any] = {
        "schema_version": "neurobench.residual_safety.v1",
        "thresholds": thresholds.to_manifest(),
        "decisions": decisions,
        "methods": methods,
        "decision_grain": "residual_method_by_background_window",
        "decision_uses_source_truth_or_labels": False,
        "interpretation": (
            "Engineering admissibility only; unsafe windows use the frozen raw endpoint. "
            "Native candidates remain biologically unknown."
        ),
    }
    result["result_sha256"] = stable_hash(result)
    return result


def apply_residual_safety_policy(
    evaluation_rows: Sequence[Mapping[str, Any]],
    safety: Mapping[str, Any],
    *,
    residual_method: str,
    raw_method: str = RAW_METHOD,
) -> dict[str, Any]:
    """Apply frozen window decisions to paired exact-truth rows.

    Recovery is evaluated only after decisions have been fixed from source-off
    diagnostics.  It never enters the selection rule.
    """

    decision_by_window = {
        str(row["background_window_id"]): str(row["selected_endpoint"])
        for row in safety["decisions"]
        if row["method"] == residual_method
    }
    if not decision_by_window:
        raise ResidualSafetyError(f"no safety decisions found for {residual_method}")
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in evaluation_rows:
        method = str(row.get("method", ""))
        if method not in {raw_method, residual_method}:
            continue
        fixture_id = str(row.get("fixture_id", ""))
        if not fixture_id:
            raise ResidualSafetyError("evaluation row lacks fixture_id")
        if method in grouped.setdefault(fixture_id, {}):
            raise ResidualSafetyError(f"duplicate {method} evaluation for {fixture_id}")
        grouped[fixture_id][method] = row
    selected_rows: list[dict[str, Any]] = []
    for fixture_id in sorted(grouped):
        arms = grouped[fixture_id]
        if set(arms) != {raw_method, residual_method}:
            raise ResidualSafetyError(f"incomplete raw/residual pair for {fixture_id}")
        window_id = str(arms[raw_method].get("background_window_id", ""))
        if window_id != str(arms[residual_method].get("background_window_id", "")):
            raise ResidualSafetyError(f"window mismatch for {fixture_id}")
        try:
            selected_method = decision_by_window[window_id]
        except KeyError as exc:
            raise ResidualSafetyError(f"missing safety decision for {window_id}") from exc
        selected = arms[selected_method]
        recovery = selected.get("source_on_recovery")
        if not isinstance(recovery, Mapping):
            raise ResidualSafetyError("evaluation row lacks source_on_recovery")
        selected_rows.append(
            {
                "fixture_id": fixture_id,
                "background_recording_id": str(selected["background_recording_id"]),
                "background_window_id": window_id,
                "selected_method": selected_method,
                "residual_admitted": selected_method == residual_method,
                "source_on_recall": _finite_float(recovery.get("recall"), name="source_on_recall"),
                "recovered_sources": int(recovery["recovered_sources"]),
                "injected_sources": int(recovery["injected_sources"]),
            }
        )
    if not selected_rows:
        raise ResidualSafetyError("no complete paired evaluations were supplied")
    result = {
        "schema_version": "neurobench.residual_safety_policy_application.v1",
        "residual_method": residual_method,
        "raw_method": raw_method,
        "fixture_count": len(selected_rows),
        "residual_admitted_fixture_count": sum(row["residual_admitted"] for row in selected_rows),
        "macro_source_on_recall": float(np.mean([row["source_on_recall"] for row in selected_rows])),
        "micro_recovered_sources": sum(row["recovered_sources"] for row in selected_rows),
        "micro_injected_sources": sum(row["injected_sources"] for row in selected_rows),
        "selection_uses_source_truth_or_labels": False,
        "rows": selected_rows,
    }
    result["result_sha256"] = stable_hash(result)
    return result


__all__ = [
    "JEPA_RESIDUAL_METHOD",
    "RANDOM_RESIDUAL_METHOD",
    "RAW_METHOD",
    "ResidualSafetyError",
    "ResidualSafetyThresholds",
    "apply_residual_safety_policy",
    "collapse_source_off_diagnostics",
    "evaluate_residual_safety",
]
