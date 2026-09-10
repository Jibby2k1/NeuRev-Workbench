"""Fail-closed requirement audit for the complete real-data ICA program."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neurobench.experiments.frame_difference import _atomic_json


def _json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def audit_ica_whitening_completion(
    experiment_root: str | Path, *, expected_factorial_fits: int = 30_891,
) -> dict[str, Any]:
    root = Path(experiment_root).expanduser().resolve()
    real = root / "real_data_v1"; stages = real / "stages"
    amendment = _json(root / "decision_amendment.json")
    preflight = _json(root.parent / "preflight_spon_ca_burst_ica_whitening_real_data_v1_v2" / "preflight.json")
    s3 = _json(stages / "S3_COMPLETE_FACTORIAL_ANALYSIS" / "summary.json")
    s3_validation = _json(stages / "S3_COMPLETE_FACTORIAL_ANALYSIS" / "validation.json")
    s4 = _json(stages / "S4_FINALIST_CONFIRMATION" / "summary.json")
    s4_validation = _json(stages / "S4_FINALIST_CONFIRMATION" / "validation.json")
    s5 = _json(stages / "S5_SCIENTIFIC_AUDIT" / "summary.json")
    s5_validation = _json(stages / "S5_SCIENTIFIC_AUDIT" / "validation.json")
    s6 = _json(stages / "S6_INDEPENDENT_CONFIRMATION" / "summary.json")
    s6_validation = _json(stages / "S6_INDEPENDENT_CONFIRMATION" / "validation.json")
    s4_refits = sorted((stages / "S4_FINALIST_CONFIRMATION" / "refits").glob("*.json"))
    diagnostic_keys = {
        "reconstruction_integrity", "approximate_snr", "candidate_trace_coherence",
        "component_response", "external_whitening", "rank_matched_controls",
    }
    refit_diagnostics_complete = bool(s4_refits) and all(
        diagnostic_keys <= set((_json(path) or {})) for path in s4_refits
    )
    learned_figures = stages / "S3_COMPLETE_FACTORIAL_ANALYSIS" / "figures" / "learned_parameters"
    expected_figures = {
        "frequency_response_vs_rank.png", "frequency_response_vs_support.png",
        "whitening_diagnostics_by_geometry.png",
    }
    gates = {
        "synthetic_failure_amended_non_blocking": bool(
            amendment
            and amendment.get("amended_downstream_decision")
            == "continue_real_data_with_synthetic_warning"
        ),
        "real_data_contract_ready": bool(
            preflight and preflight.get("ready")
            and preflight.get("labels", {}).get("unmatched_candidates") == "unknown_not_negative"
        ),
        "complete_factorial_exact_coverage": bool(
            s3 and s3_validation and s3_validation.get("status") == "pass"
            and int(s3_validation.get("expected_fit_count", -1)) == expected_factorial_fits
            and int(s3_validation.get("observed_fit_count", -2)) == expected_factorial_fits
            and int(s3_validation.get("unique_fit_count", -3)) == expected_factorial_fits
            and s3.get("exact_coverage") is True
        ),
        "conditional_statistics_and_interpretability": bool(
            s3 and s3.get("conditional_factor_summaries")
            and s3.get("learned_parameter_summary")
            and s3.get("formal_sobol_indices_identified") is False
            and learned_figures.is_dir()
            and expected_figures <= {path.name for path in learned_figures.glob("*.png")}
        ),
        "fold_specific_protected_confirmation": bool(
            s4 and s4_validation and s4_validation.get("status") == "pass"
            and s4.get("claim_scope")
            == "fold_specific_strict_held_out_burst_within_recording_confirmation"
            and set(map(int, s4.get("selected_fit_ids_by_held_out_burst", {}))) == {1, 2, 3, 4}
            and s4.get("protected_held_out_performance", {}).get("grouping_unit")
            == "held_out_burst"
            and refit_diagnostics_complete
        ),
        "all_finalist_scientific_visual_audit": bool(
            s4 and s5 and s5_validation and s5_validation.get("status") == "pass"
            and set(map(str, s5.get("selected_fit_ids", [])))
            == set(map(str, s4.get("selected_fit_ids", [])))
            and set(map(str, s5_validation.get("validated_finalists", [])))
            == set(map(str, s4.get("selected_fit_ids", [])))
            and s5_validation.get("promotion_allowed_by_audit") is True
        ),
        "independent_confirmation_exact_coverage": bool(
            s6 and s6_validation and s6_validation.get("status") == "pass"
            and s6.get("recording_id") == "15 right"
            and s6.get("exact_artifact_coverage") is True
            and s6.get("labels_loaded_after_all_score_files") is True
            and s6.get("unmatched_candidates") == "unknown_not_negative"
        ),
        "superseded_mixed_spatial_excluded": bool(
            (_json(real / "superseded" / "mixed_implementation_spatial_20260831"
                   / "supersession.json") or {}).get("scientific_use") == "prohibited"
        ),
    }
    incomplete = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": 1,
        "status": "complete" if not incomplete else "incomplete",
        "gates": gates, "incomplete_gates": incomplete,
        "completion_claim_allowed": not incomplete,
        "expected_factorial_fit_count": expected_factorial_fits,
    }


def write_completion_audit(
    experiment_root: str | Path, destination: str | Path,
    *, expected_factorial_fits: int = 30_891,
) -> dict[str, Any]:
    result = audit_ica_whitening_completion(
        experiment_root, expected_factorial_fits=expected_factorial_fits,
    )
    _atomic_json(Path(destination).expanduser().resolve(), result)
    return result
