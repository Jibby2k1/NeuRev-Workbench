"""Exact merge and matched-control analysis for real-data ICA stages."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from neurobench.experiments.frame_difference import _atomic_json
from neurobench.experiments.learned_operator_selection.data import (
    canonical_event_intervals, load_and_validate_labels,
)
from neurobench.metrics.sparse_detection import temporal_pool

from .config import ICAWhiteningConfig
from .real_config import RealDataConfig
from .real_preflight import _lane_stacks
from .real_runner import (
    _label_metrics, _load_proposals, _load_specs, _signed_review,
)


def _controls(
    real: RealDataConfig, parent: ICAWhiteningConfig, preflight_root: Path,
) -> dict[str, Any]:
    source = np.load(parent.source_video, mmap_mode="r", allow_pickle=False)
    movie = _signed_review(source, parent)
    labels = load_and_validate_labels(parent.labels_tsv, parent.label_summary, tuple(source.shape[1:]))
    proposals = _load_proposals(preflight_root / "common_proposal_universe.tsv")
    intervals = canonical_event_intervals(labels, parent.frames.review_start_ui)
    results = {}
    for lane, stack in _lane_stacks(movie).items():
        maps = {
            burst: temporal_pool(stack[start:stop], real.proposals.temporal_pool)
            for burst, (start, stop) in intervals.items()
        }
        scores = np.asarray([
            maps[row["burst_id"]][row["y_px"], row["x_px"]] for row in proposals
        ])
        results[lane] = _label_metrics(
            proposals, scores, labels, real.proposals.evaluation_budgets,
            real.proposals.match_radius_px,
        )
    return results


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row["label_metrics"]
    return {
        "fit_id": row["fit_id"], "family": row["family"],
        "objective": row["objective"], "rank": row["rank"],
        "causality": row["causality"],
        "spatial_width_px": row.get("spatial_width_px"),
        "temporal_width_frames": row.get("temporal_width_frames"),
        "seed": row["seed"], "real_data_fit_converged": row["real_data_fit_converged"],
        "macro_known_positive_recall_at_58": metrics["macro_known_positive_recall"],
        "pooled_known_positive_recall_at_58": metrics["pooled_known_positive_recall"],
        "mean_reciprocal_rank": metrics["mean_reciprocal_rank"],
        "candidate_score_sha256": row["candidate_score_sha256_before_label_metrics"],
    }


def merge_no_whitening(
    real: RealDataConfig, *, preflight_dir: str | Path,
) -> dict[str, Any]:
    preflight_root = Path(preflight_dir).expanduser().resolve()
    parent = ICAWhiteningConfig.from_json(real.parent_config)
    expected_specs = _load_specs(real.synthetic_registry.parent.parent, whitening_geometry="none")
    expected = {row["fit_id"] for row in expected_specs}
    rows = []; observed = set(); shards = []
    for index in range(real.fitting.shard_count):
        root = real.output_dir / "stages" / f"S2A_NO_WHITENING_SHARD_{index:02d}_OF_{real.fitting.shard_count:02d}"
        validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
        if validation.get("status") != "pass":
            raise RuntimeError(f"invalid real-data shard {index}")
        count = 0
        with (root / "fit_results.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip(): continue
                row = json.loads(line); fit_id = row["fit_id"]
                if fit_id in observed: raise RuntimeError(f"duplicate fit: {fit_id}")
                observed.add(fit_id); rows.append(row); count += 1
        shards.append({"shard_index": index, "fit_count": count, "path": str(root.relative_to(real.output_dir))})
    if observed != expected:
        raise RuntimeError(f"coverage mismatch: missing={len(expected-observed)} extra={len(observed-expected)}")
    destination = real.output_dir / "stages" / "S2A_NO_WHITENING_MERGED"
    destination.mkdir(parents=True, exist_ok=False)
    flat = [_flatten(row) for row in sorted(rows, key=lambda row: row["fit_id"])]
    with (destination / "fit_registry.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(flat)
    families = []
    for family in parent.design.families:
        selected = [row for row in rows if row["family"] == family]
        best = max(selected, key=lambda row: (
            row["label_metrics"]["macro_known_positive_recall"],
            row["label_metrics"]["mean_reciprocal_rank"], row["fit_id"],
        ))
        families.append({
            "family": family, "fit_count": len(selected),
            "converged_fraction": float(np.mean([row["real_data_fit_converged"] for row in selected])),
            "mean_macro_known_positive_recall_at_58": float(np.mean([
                row["label_metrics"]["macro_known_positive_recall"] for row in selected
            ])),
            "best_fit_id": best["fit_id"],
            "best_macro_known_positive_recall_at_58": best["label_metrics"]["macro_known_positive_recall"],
            "best_mean_reciprocal_rank": best["label_metrics"]["mean_reciprocal_rank"],
        })
    controls = _controls(real, parent, preflight_root)
    summary = {
        "schema_version": 1, "status": "complete", "stage": "S2A_NO_WHITENING_MERGED",
        "fit_count": len(rows), "exact_coverage": True, "families": families,
        "matched_controls": controls,
        "interpretation": "within-recording common-universe sparse-positive utility; unmatched candidates unknown",
        "synthetic_warning": "non_blocking_but_source_identity_claims_prohibited",
    }
    _atomic_json(destination / "summary.json", summary)
    _atomic_json(destination / "validation.json", {
        "status": "pass", "expected_fit_count": len(expected),
        "observed_fit_count": len(observed), "unique_fit_count": len(observed),
        "missing_fit_count": 0, "extra_fit_count": 0,
    })
    _atomic_json(destination / "shard_manifest.json", {"shards": shards})
    _atomic_json(destination / "artifact_index.json", {"artifacts": [
        {"path": "summary.json", "role": "stage_summary"},
        {"path": "validation.json", "role": "exact_coverage_validation"},
        {"path": "fit_registry.tsv", "role": "complete_fit_registry"},
        {"path": "shard_manifest.json", "role": "checkpoint_provenance"},
    ]})
    return summary


def merge_event_standardized(
    real: RealDataConfig, *, preflight_dir: str | Path,
) -> dict[str, Any]:
    preflight_root = Path(preflight_dir).expanduser().resolve()
    parent = ICAWhiteningConfig.from_json(real.parent_config)
    expected_specs = _load_specs(real.synthetic_registry.parent.parent, whitening_geometry="none")
    expected = {row["fit_id"] for row in expected_specs}
    rows = []; observed = set(); shards = []
    for index in range(real.fitting.shard_count):
        root = real.output_dir / "stages" / f"S2B_EVENT_STANDARDIZED_SHARD_{index:02d}_OF_{real.fitting.shard_count:02d}"
        validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
        if validation.get("status") != "pass": raise RuntimeError(f"invalid standardized shard {index}")
        count = 0
        with (root / "fit_results.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip(): continue
                row = json.loads(line); fit_id = row["fit_id"]
                if fit_id in observed: raise RuntimeError(f"duplicate fit: {fit_id}")
                observed.add(fit_id); rows.append(row); count += 1
        shards.append({"shard_index": index, "fit_count": count, "path": str(root.relative_to(real.output_dir))})
    if observed != expected:
        raise RuntimeError(f"coverage mismatch: missing={len(expected-observed)} extra={len(observed-expected)}")
    destination = real.output_dir / "stages" / "S2B_EVENT_STANDARDIZED_MERGED"
    destination.mkdir(parents=True, exist_ok=False)
    flat = [_flatten(row) for row in sorted(rows, key=lambda row: row["fit_id"])]
    with (destination / "fit_registry.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(flat)
    families = []
    for family in parent.design.families:
        selected = [row for row in rows if row["family"] == family]
        best = max(selected, key=lambda row: (
            row["label_metrics"]["macro_known_positive_recall"],
            row["label_metrics"]["mean_reciprocal_rank"], row["fit_id"],
        ))
        families.append({
            "family": family, "fit_count": len(selected),
            "converged_fraction": float(np.mean([row["real_data_fit_converged"] for row in selected])),
            "mean_macro_known_positive_recall_at_58": float(np.mean([
                row["label_metrics"]["macro_known_positive_recall"] for row in selected
            ])),
            "best_fit_id": best["fit_id"],
            "best_macro_known_positive_recall_at_58": best["label_metrics"]["macro_known_positive_recall"],
            "best_pooled_known_positive_recall_at_58": best["label_metrics"]["pooled_known_positive_recall"],
            "best_mean_reciprocal_rank": best["label_metrics"]["mean_reciprocal_rank"],
        })
    summary = {
        "schema_version": 1, "status": "complete",
        "stage": "S2B_EVENT_STANDARDIZED_MERGED", "fit_count": len(rows),
        "exact_coverage": True, "families": families,
        "external_whitening_application_counts": {
            mode: sum(1 for row in rows if row.get(
                "external_whitening_application", "sparse_equivalent_legacy"
            ) == mode)
            for mode in sorted({row.get(
                "external_whitening_application", "sparse_equivalent_legacy"
            ) for row in rows})
        },
        "mixed_application_equivalence": "dense_and_sparse_paths_guarded_by_exact_coordinate_tests",
        "matched_controls": _controls(real, parent, preflight_root),
        "score_semantics": "quiet_mad_standardized_max_abs_component_lme0.25_over_event",
        "interpretation": "within-recording common-universe sparse-positive utility; unmatched candidates unknown",
        "synthetic_warning": "non_blocking_but_source_identity_claims_prohibited",
    }
    _atomic_json(destination / "summary.json", summary)
    _atomic_json(destination / "validation.json", {
        "status": "pass", "expected_fit_count": len(expected),
        "observed_fit_count": len(observed), "unique_fit_count": len(observed),
        "missing_fit_count": 0, "extra_fit_count": 0,
    })
    _atomic_json(destination / "shard_manifest.json", {"shards": shards})
    _atomic_json(destination / "artifact_index.json", {"artifacts": [
        {"path": "summary.json", "role": "stage_summary"},
        {"path": "validation.json", "role": "exact_coverage_validation"},
        {"path": "fit_registry.tsv", "role": "complete_fit_registry"},
        {"path": "shard_manifest.json", "role": "checkpoint_provenance"},
    ]})
    return summary


def merge_single_stage_whitening(
    real: RealDataConfig, *, whitening_geometry: str,
) -> dict[str, Any]:
    """Merge one completed sparse-whitening geometry with exact ID coverage."""
    if whitening_geometry not in {"spatial", "temporal", "joint_spatiotemporal",
                                  "spatial_then_temporal", "temporal_then_spatial"}:
        raise ValueError("invalid whitening geometry")
    expected_specs = _load_specs(real.synthetic_registry.parent.parent,
                                 whitening_geometry=whitening_geometry)
    expected = {row["fit_id"] for row in expected_specs}
    rows = []; observed = set(); shards = []
    tag = whitening_geometry.upper()
    for index in range(real.fitting.shard_count):
        root = real.output_dir / "stages" / f"S2C_{tag}_WHITENING_SHARD_{index:02d}_OF_{real.fitting.shard_count:02d}"
        validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
        if validation.get("status") != "pass":
            raise RuntimeError(f"invalid {whitening_geometry} whitening shard {index}")
        count = 0
        with (root / "fit_results.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip(): continue
                row = json.loads(line); fit_id = row["fit_id"]
                if fit_id in observed: raise RuntimeError(f"duplicate fit: {fit_id}")
                observed.add(fit_id); rows.append(row); count += 1
        shards.append({"shard_index": index, "fit_count": count,
                       "path": str(root.relative_to(real.output_dir))})
    if observed != expected:
        raise RuntimeError(f"coverage mismatch: missing={len(expected-observed)} extra={len(observed-expected)}")
    destination = real.output_dir / "stages" / f"S2C_{tag}_WHITENING_MERGED"
    destination.mkdir(parents=True, exist_ok=False)
    registry_fields = list(_flatten(rows[0])) + [
        "whitening_geometry", "covariance_scope", "raw_preserving_blend",
        "external_whitening_resolved", "external_whitening_condition_number",
        "external_whitening_effective_rank_fraction",
    ]
    flat = []
    for row in sorted(rows, key=lambda item: item["fit_id"]):
        record = _flatten(row)
        diagnostics = row["external_whitening_diagnostics"]
        record.update({
            "whitening_geometry": row["whitening_geometry"],
            "covariance_scope": row["covariance_scope"],
            "raw_preserving_blend": row["raw_preserving_blend"],
            "external_whitening_resolved": diagnostics["resolved"],
            "external_whitening_condition_number": diagnostics["maximum_condition_number"],
            "external_whitening_effective_rank_fraction": diagnostics["minimum_effective_rank_fraction"],
        }); flat.append(record)
    with (destination / "fit_registry.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=registry_fields, delimiter="\t")
        writer.writeheader(); writer.writerows(flat)
    families = []
    for family in sorted({row["family"] for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        best = max(selected, key=lambda row: (
            row["label_metrics"]["macro_known_positive_recall"],
            row["label_metrics"]["mean_reciprocal_rank"], row["fit_id"],
        ))
        families.append({
            "family": family, "fit_count": len(selected), "best_fit_id": best["fit_id"],
            "best_macro_known_positive_recall_at_58": best["label_metrics"]["macro_known_positive_recall"],
            "best_pooled_known_positive_recall_at_58": best["label_metrics"]["pooled_known_positive_recall"],
            "best_mean_reciprocal_rank": best["label_metrics"]["mean_reciprocal_rank"],
            "converged_fraction": float(np.mean([row["real_data_fit_converged"] for row in selected])),
            "external_whitening_resolved_fraction": float(np.mean([
                row["external_whitening_diagnostics"]["resolved"] for row in selected
            ])),
        })
    summary = {
        "schema_version": 1, "status": "complete", "stage": f"S2C_{tag}_WHITENING_MERGED",
        "whitening_geometry": whitening_geometry, "fit_count": len(rows),
        "exact_coverage": True, "families": families,
        "score_semantics": "quiet_mad_standardized_max_abs_component_lme0.25_over_event",
        "interpretation": "within-recording common-universe sparse-positive utility; unmatched candidates unknown",
        "synthetic_warning": "non_blocking_but_source_identity_claims_prohibited",
    }
    _atomic_json(destination / "summary.json", summary)
    _atomic_json(destination / "validation.json", {
        "status": "pass", "expected_fit_count": len(expected),
        "observed_fit_count": len(observed), "unique_fit_count": len(observed),
        "missing_fit_count": 0, "extra_fit_count": 0,
    })
    _atomic_json(destination / "shard_manifest.json", {"shards": shards})
    return summary
