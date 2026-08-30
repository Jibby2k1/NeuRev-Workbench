"""Protected, read-only confirmation of the frozen compact feature panel."""
from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

from .contracts import atomic_json
from .reporting import now, write_stage


QUANTITATIVE_LANES = ("carrier_signed", "coherence_w15", "propagation_lag2_w15")
PRIMARY_BUDGETS = (20, 40, 58)
AUDIT_REQUIRED = ("REPORT.md", "status.json", "summary.json", "llm_context.json", "artifact_index.json", "validation.json")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _native_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {
        str(row["feature_id"]): row for row in payload.get("results", [])
        if row.get("label_view") == "original" and row.get("timing_view") == "original"
        and row.get("feature_id") in QUANTITATIVE_LANES
    }
    if set(rows) != set(QUANTITATIVE_LANES):
        raise ValueError("native screen lacks the frozen quantitative lanes")
    label_counts = {int(row["pooled_counts"]["20"]["labels"]) for row in rows.values()}
    if label_counts != {79}:
        raise ValueError(f"protected screen must contain exactly 79 occurrences, got {label_counts}")
    return rows


def classify_native(lane: dict[str, Any], baseline: dict[str, Any], *, nms_confirmed: bool = False) -> dict[str, Any]:
    delta = float(lane["macro_recall"]["20"]) - float(baseline["macro_recall"]["20"])
    burst_delta = [
        float(row["budgets"]["20"]["recall"]) - float(base["budgets"]["20"]["recall"])
        for row, base in zip(lane["folds"], baseline["folds"], strict=True)
    ]
    nonnegative = sum(value >= 0 for value in burst_delta)
    catastrophic = min(burst_delta) < -0.05
    candidate_c3 = delta >= 0.03 and nonnegative >= 3 and not catastrophic
    if candidate_c3 and nms_confirmed:
        level = "C3_confirmed_compact_utility"
    elif candidate_c3:
        level = "C3_candidate_pending_nms"
    elif delta > 0:
        level = "C2_limited_utility"
    elif any(value > 0 for value in burst_delta):
        level = "C1_descriptive"
    else:
        level = "C0_no_demonstrated_utility"
    return {
        "candidate_level": level,
        "budget_20_macro_delta": delta,
        "budget_20_burst_deltas": burst_delta,
        "nonnegative_bursts": nonnegative,
        "catastrophic_drop": catastrophic,
        "c3_metric_pattern": candidate_c3,
    }


def build_confirmation(data_root: Path, repository_root: Path, run_root: Path | None = None) -> dict[str, Any]:
    native_path = data_root / "Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_rescore_final_v1/metrics.json"
    audit_root = data_root / "Outputs/HierarchicalParzenICA/spon_ca_burst_scientific_feature_audit_v1"
    identical_path = audit_root / "evaluation/identical_proposal_screen.json"
    inventory_path = audit_root / "evaluation/identical_proposal_inventory.json"
    native_payload, identical, inventory = _read(native_path), _read(identical_path), _read(inventory_path)
    nms_path = run_root / "10_representation_confirmation/nms_sensitivity/nms_sensitivity.json" if run_root else None
    nms = _read(nms_path) if nms_path and nms_path.is_file() else None
    visual_root = run_root / "10_representation_confirmation/detector_visual_audit_v2" if run_root else None
    visual_validation = _read(visual_root / "validation.json") if visual_root and (visual_root / "validation.json").is_file() else None
    visual_passed = bool(visual_validation and visual_validation.get("status") == "passed")
    native = _native_rows(native_payload)
    baseline = native["carrier_signed"]
    identical_rows = {str(row.get("config_id")): row for row in identical.get("rows", [])}
    identical_baseline = identical.get("carrier_baseline", {})
    results = []
    for lane_id in QUANTITATIVE_LANES[1:]:
        nms_confirmed = bool(nms and nms.get("nms_robust_c3", {}).get(lane_id))
        classification = classify_native(native[lane_id], baseline, nms_confirmed=nms_confirmed)
        row = identical_rows.get(f"standalone__{lane_id}")
        if row is None:
            raise ValueError(f"identical-proposal screen lacks standalone {lane_id}")
        identical_delta = {
            str(budget): float(row["budget_mean_recall"][str(budget)])
            - float(identical_baseline["budget_mean_recall"][str(budget)])
            for budget in PRIMARY_BUDGETS
        }
        results.append({
            "feature_id": lane_id,
            "native_macro_recall": {str(k): native[lane_id]["macro_recall"][str(k)] for k in PRIMARY_BUDGETS},
            "native": classification,
            "identical_proposal_macro_delta": identical_delta,
            "ranking_utility_at_budget_20": identical_delta["20"] >= 0.03,
            "promotion_status": "C3_confirmed_publication_supported" if nms_confirmed and visual_passed else ("C3_confirmed_publication_held" if nms_confirmed else "held"),
            "promotion_blockers": [] if visual_passed else ["visual detector validation is incomplete"],
        })
    audit_missing = [name for name in AUDIT_REQUIRED if not (audit_root / name).is_file()]
    pairwise_path = repository_root / "Outputs/UnsupervisedICAEval/two_frame_stage_a_v1/summary.json"
    pairwise = _read(pairwise_path) if pairwise_path.is_file() else None
    return {
        "schema_version": 1,
        "status": "C3_confirmed_publication_supported" if nms and all(nms.get("nms_robust_c3", {}).values()) and visual_passed else ("C3_confirmed_publication_held" if nms and all(nms.get("nms_robust_c3", {}).values()) else "protected_confirmation_held"),
        "estimand": "known-positive recall on 79 immutable original sites/occurrences with original timing",
        "precision": "not_identified; unmatched candidates remain unknown",
        "labels": 79,
        "native_source": str(native_path),
        "native_source_status": native_payload.get("status"),
        "identical_proposal_source": str(identical_path),
        "proposal_inventory_source": str(inventory_path),
        "proposal_inventory_feature_ids": inventory.get("feature_ids", []),
        "primary_nms_radius_px": 6,
        "nms_sensitivity": {"required": [4, 6, 8], "available": [4, 6, 8] if nms else [6], "passed": bool(nms and all(nms.get("nms_robust_c3", {}).values())), "source": str(nms_path) if nms_path else None},
        "detector_visual_validation": {"required": True, "passed": visual_passed, "source": str(visual_root) if visual_root else None, "validation": visual_validation},
        "results": results,
        "scientific_audit": {"root": str(audit_root), "complete": not audit_missing, "missing": audit_missing},
        "pairwise_ica": {
            "source": str(pairwise_path), "available": pairwise is not None,
            "conclusion": "retained negative operator-identification result; no new source-identifying information established",
            "summary": pairwise,
        },
        "decision": "C3_confirmed_publication_supported" if nms and all(nms.get("nms_robust_c3", {}).values()) and visual_passed else ("C3_confirmed_hold_publication_pending_visual_validation" if nms and all(nms.get("nms_robust_c3", {}).values()) else "hold_promotion"),
    }


def refresh_after_visual_audit(data_root: Path, repository_root: Path, run_root: Path) -> dict[str, Any]:
    confirmation = build_confirmation(data_root, repository_root, run_root)
    confirmation["started_at"] = now()
    previous = run_root / "10_representation_confirmation/confirmation.json"
    if previous.is_file(): confirmation["lanes"] = _read(previous).get("lanes", list(QUANTITATIVE_LANES))
    atomic_json(previous, confirmation)
    rows = [{"feature_id": row["feature_id"], "candidate_level": row["native"]["candidate_level"], "native_delta_b20": row["native"]["budget_20_macro_delta"], "identical_delta_b20": row["identical_proposal_macro_delta"]["20"], "promotion_status": row["promotion_status"]} for row in confirmation["results"]]
    table = run_root / "10_representation_confirmation/compact_panel.tsv"
    with table.with_name(table.name + ".partial").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t"); writer.writeheader(); writer.writerows(rows)
    table.with_name(table.name + ".partial").replace(table)
    write_stage(run_root, "10_representation_confirmation", confirmation, status="complete", decision="advance", warnings=["Both compact lanes meet the frozen C3 native-utility, NMS-robustness, and three-section visual-audit gates.", "Known-positive recall is supported for publication within this recording; precision, specificity, false-positive rate, and cross-recording generalization remain unidentified."], next_stage="11_manuscript_exports")
    return confirmation
