"""Resumable numerical and truth-known factorial screen."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from typing import Any, Iterable

import numpy as np
from scipy.stats import kurtosis, spearmanr

from neurobench.experiments.frame_difference import _atomic_json
from neurobench.experiments.information_source_separation.synthetic import (
    make_spatiotemporal_fixture,
)
from neurobench.metrics.source_separation import aligned_source_metrics

from .config import ICAWhiteningConfig
from .design import build_design, design_digest
from .model import extract_patch_observations, fit_ica
from .operators import apply_whitening
from .preflight import matching_preflight
from .responses import component_response_summary


DEFAULT_CASES = ("isolated", "overlap", "synchronous", "pure_noise")


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def select_design_shard(
    design: list[dict[str, Any]], shard_index: int, shard_count: int
) -> list[dict[str, Any]]:
    if shard_count < 2 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    return [row for index, row in enumerate(design) if index % shard_count == shard_index]


def _signed_global(movie: np.ndarray, quiet_frames: int) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(movie, dtype=np.float32)
    quiet = values[:quiet_frames]
    center = float(np.median(quiet))
    low, high = np.percentile(quiet, [1, 99.9])
    scale = max(float(high - low), np.finfo(float).eps)
    return ((values - center) / scale).astype(np.float32), {
        "quiet_global_median": center,
        "quiet_percentile_1": float(low),
        "quiet_percentile_99p9": float(high),
        "global_scale": scale,
    }


def _true_sources(fixture: Any, patches: Any) -> np.ndarray:
    return (
        fixture.traces[:, patches.times]
        * fixture.footprints[:, patches.rows, patches.columns]
    ).astype(np.float64)


def _trace_preservation(
    reference: np.ndarray, transformed: np.ndarray, footprints: np.ndarray,
    *, quiet_frames: int,
) -> float:
    rows = []
    for footprint in footprints:
        y, x = np.unravel_index(int(np.argmax(footprint)), footprint.shape)
        left = np.asarray(reference[:, y, x], dtype=np.float64)
        right = np.asarray(transformed[:, y, x], dtype=np.float64)
        if np.std(left[quiet_frames:]) == 0 or np.std(right[quiet_frames:]) == 0:
            continue
        value = spearmanr(left[quiet_frames:], right[quiet_frames:]).statistic
        if np.isfinite(value):
            rows.append(float(value))
    return float(np.median(rows)) if rows else float("nan")


def _integrity_metrics(
    signed: np.ndarray, transformed: np.ndarray, fixture: Any,
    *, quiet_frames: int,
) -> dict[str, float]:
    left = np.asarray(signed, dtype=np.float64).ravel()
    right = np.asarray(transformed, dtype=np.float64).ravel()
    raw_output_correlation = (
        float(np.corrcoef(left, right)[0, 1])
        if np.std(left) > 0 and np.std(right) > 0 else 0.0
    )
    neural = np.asarray(fixture.neural_signal, dtype=np.float64).ravel()
    output_neural_correlation = (
        float(np.corrcoef(right, neural)[0, 1])
        if np.std(right) > 0 and np.std(neural) > 0 else 0.0
    )
    quiet = np.asarray(transformed[:quiet_frames], dtype=np.float64)
    event = np.asarray(transformed[quiet_frames:], dtype=np.float64)
    quiet_center = float(np.median(quiet))
    quiet_scale = max(
        float(1.4826 * np.median(np.abs(quiet - quiet_center))),
        np.finfo(float).eps,
    )
    event_scale = float(np.sqrt(np.mean((event - quiet_center) ** 2)))
    residual = np.asarray(signed - transformed, dtype=np.float64)
    return {
        "raw_output_correlation": raw_output_correlation,
        "output_neural_correlation": output_neural_correlation,
        "approximate_event_to_quiet_snr": event_scale / quiet_scale,
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "trace_preservation": _trace_preservation(
            signed, transformed, fixture.footprints, quiet_frames=quiet_frames
        ),
    }


def _compact_whitening(payload: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: payload[key] for key in (
            "geometry", "covariance_scope", "causality", "resolved",
            "maximum_condition_number", "minimum_effective_rank_fraction",
            "finite_output_fraction", "raw_preserving_blend",
        ) if key in payload
    }
    if "region_fits" in payload:
        result["region_fits"] = [
            {key: row[key] for key in (
                "region", "condition_number", "effective_rank_fraction",
                "sample_count", "resolved", "stop_reason", "shrinkage",
                "exponent", "eigen_floor_ratio",
            )}
            for row in payload["region_fits"]
        ]
    if "stages" in payload:
        result["stages"] = [_compact_whitening(stage) for stage in payload["stages"]]
    return result


def _compact_model(model: Any) -> dict[str, Any]:
    return {
        "method_id": model.method_id,
        "converged": model.converged,
        "iterations": model.iterations,
        "objective": model.objective,
        "retained_rank": model.whitening.retained_rank,
        "explained_fraction": model.whitening.explained_fraction,
        "condition_number": model.whitening.condition_number,
        "relative_subspace_closure_error": model.diagnostics.get(
            "relative_subspace_closure_error"
        ),
        "final_delta": model.diagnostics.get("final_delta"),
    }


def _compact_recovery(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        key: payload[key] for key in (
            "true_source_count", "estimated_source_count", "matched_source_count",
            "mean_absolute_correlation", "worst_absolute_correlation",
            "mean_aligned_nmse", "worst_aligned_nmse",
            "mean_absolute_crosstalk", "worst_absolute_crosstalk",
        )
    }


def evaluate_specification(
    specification: dict[str, Any],
    config: ICAWhiteningConfig,
    *,
    cases: Iterable[str] = DEFAULT_CASES,
) -> dict[str, Any]:
    quiet_frames = 32
    case_rows = []
    for case_index, case_id in enumerate(cases):
        fixture = make_spatiotemporal_fixture(
            case_id, seed=int(specification["seed"]) + case_index * 10007,
            frame_count=96, shape=(12, 12), snr=8.0,
            frame_period_ms=config.frames.frame_period_ms,
        )
        signed, normalization = _signed_global(fixture.observation, quiet_frames)
        whitening = apply_whitening(
            signed, quiet_frames, specification,
            maximum_samples=min(config.resources.maximum_fit_samples, 1024),
        )
        patches = extract_patch_observations(
            whitening.output,
            family=specification["family"],
            spatial_width=specification.get("spatial_width_px"),
            temporal_width=specification.get("temporal_width_frames"),
            causality=specification["causality"],
            maximum_samples=min(config.resources.maximum_fit_samples, 1024),
            seed=int(specification["seed"]) + case_index * 7919,
            quiet_frames=quiet_frames,
            activity_fraction=0.5,
        )
        model = fit_ica(
            patches.values, specification,
            maximum_fit_samples=config.resources.maximum_fit_samples,
        )
        source_statistics = {
            "maximum_absolute_excess_kurtosis": float(np.max(np.abs(
                kurtosis(model.sources, axis=1, fisher=True, bias=False)
            ))),
            "maximum_component_rms": float(np.max(np.sqrt(np.mean(model.sources**2, axis=1)))),
        }
        if fixture.identifiable:
            recovery = aligned_source_metrics(_true_sources(fixture, patches), model.sources)
        else:
            recovery = None
        response = component_response_summary(
            model.demixing,
            family=specification["family"],
            spatial_width=specification.get("spatial_width_px"),
            temporal_width=specification.get("temporal_width_frames"),
            frame_period_ms=config.frames.frame_period_ms,
        )
        integrity = _integrity_metrics(
            signed, whitening.output, fixture, quiet_frames=quiet_frames
        )
        label_free_unresolved = bool(
            not fixture.identifiable
            and source_statistics["maximum_absolute_excess_kurtosis"] < 0.75
            and integrity["approximate_event_to_quiet_snr"] < 1.5
        )
        case_rows.append({
            "case_id": case_id,
            "identifiable": fixture.identifiable,
            "normalization": normalization,
            "whitening": _compact_whitening(whitening.diagnostics),
            "model": _compact_model(model),
            "source_statistics": source_statistics,
            "recovery": _compact_recovery(recovery),
            "integrity": integrity,
            "component_response": response["permutation_invariant_mean"],
            "label_free_unresolved": label_free_unresolved,
        })
    identifiable = [row for row in case_rows if row["identifiable"]]
    unidentifiable = [row for row in case_rows if not row["identifiable"]]
    return {
        **specification,
        "case_count": len(case_rows),
        "all_numerically_resolved": all(
            row["whitening"]["resolved"] and np.isfinite(row["model"]["objective"])
            for row in case_rows
        ),
        "converged_fraction": float(np.mean([
            row["model"]["converged"] for row in case_rows
        ])),
        "mean_truth_source_correlation": float(np.mean([
            row["recovery"]["mean_absolute_correlation"] for row in identifiable
        ])) if identifiable else None,
        "worst_truth_source_correlation": float(np.min([
            row["recovery"]["worst_absolute_correlation"] for row in identifiable
        ])) if identifiable else None,
        "mean_truth_crosstalk": float(np.mean([
            row["recovery"]["mean_absolute_crosstalk"] for row in identifiable
        ])) if identifiable else None,
        "mean_trace_preservation": float(np.nanmean([
            row["integrity"]["trace_preservation"] for row in identifiable
        ])) if identifiable else None,
        "unresolved_accuracy": float(np.mean([
            row["label_free_unresolved"] for row in unidentifiable
        ])) if unidentifiable else None,
        "cases": case_rows,
    }


def _load_completed(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.is_file():
        return [], set()
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid checkpoint line {line_number}") from error
    return rows, {row["fit_id"] for row in rows}


def _append_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_registry(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "fit_id", "cell_id", "family", "objective", "whitening_geometry",
        "covariance_scope", "causality", "rank", "spatial_width_px",
        "temporal_width_frames", "point_id", "point_kind", "seed",
        "all_numerically_resolved", "converged_fraction",
        "mean_truth_source_correlation", "worst_truth_source_correlation",
        "mean_truth_crosstalk", "mean_trace_preservation", "unresolved_accuracy",
    ]
    temporary = path.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    temporary.replace(path)


def _summarize(rows: list[dict[str, Any]], config: ICAWhiteningConfig) -> dict[str, Any]:
    families = []
    for family in config.design.families:
        selected = [row for row in rows if row["family"] == family]
        passing = [
            row for row in selected
            if row["all_numerically_resolved"]
            and row["converged_fraction"] >= config.gates.minimum_converged_fraction
            and row["mean_truth_source_correlation"] >= config.gates.minimum_truth_source_correlation
            and row["mean_truth_crosstalk"] <= config.gates.maximum_truth_crosstalk
            and row["mean_trace_preservation"] >= config.gates.minimum_trace_preservation
            and row["unresolved_accuracy"] == 1.0
        ]
        ordered = sorted(
            selected,
            key=lambda row: (
                -float(row["mean_truth_source_correlation"] or -1),
                float(row["mean_truth_crosstalk"] or 1e9), row["fit_id"],
            ),
        )
        families.append({
            "family": family,
            "fit_count": len(selected),
            "resolved_fraction": float(np.mean([
                row["all_numerically_resolved"] for row in selected
            ])) if selected else 0.0,
            "passing_fit_count": len(passing),
            "best_fit_id": ordered[0]["fit_id"] if ordered else None,
            "best_mean_truth_source_correlation": (
                ordered[0]["mean_truth_source_correlation"] if ordered else None
            ),
        })
    advance = bool(families and all(row["passing_fit_count"] > 0 for row in families))
    return {
        "stage": "S1_NUMERICAL_TRUTH_KNOWN",
        "fit_count": len(rows),
        "families": families,
        "decision": "advance_to_frozen_real_data" if advance else "stop_before_real_data",
        "advance": advance,
        "gate_contract": {
            "minimum_converged_fraction": config.gates.minimum_converged_fraction,
            "minimum_truth_source_correlation": config.gates.minimum_truth_source_correlation,
            "maximum_truth_crosstalk": config.gates.maximum_truth_crosstalk,
            "minimum_trace_preservation": config.gates.minimum_trace_preservation,
            "required_unresolved_accuracy": 1.0,
            "all_families_require_at_least_one_passing_fit": True,
        },
    }


def run_synthetic_screen(
    config: ICAWhiteningConfig,
    *,
    preflight_dir: str | Path,
    limit: int | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> dict[str, Any]:
    matching_preflight(config, preflight_dir)
    design = build_design(config)
    if (shard_index is None) != (shard_count is None):
        raise ValueError("shard_index and shard_count must be supplied together")
    if shard_count is not None:
        if limit is not None:
            raise ValueError("limit and sharding are mutually exclusive")
        assert shard_index is not None
        design = select_design_shard(design, shard_index, shard_count)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in design:
            key = (
                row["family"], row["objective"], row["whitening_geometry"],
                row["covariance_scope"], row["causality"], row["rank"],
            )
            groups.setdefault(key, []).append(row)
        keys = sorted(groups)
        rng = np.random.default_rng(config.design.master_seed + limit)
        keys = [keys[index] for index in rng.permutation(len(keys))]
        selected: list[dict[str, Any]] = []
        depth = 0
        while len(selected) < min(limit, len(design)):
            added = False
            for key in keys:
                if depth < len(groups[key]):
                    selected.append(groups[key][depth])
                    added = True
                    if len(selected) == min(limit, len(design)):
                        break
            if not added:
                break
            depth += 1
        design = selected
    if shard_count is not None:
        stage_name = f"S1_NUMERICAL_TRUTH_KNOWN_SHARD_{shard_index:02d}_OF_{shard_count:02d}"
    else:
        stage_name = "S1_NUMERICAL_TRUTH_KNOWN" if limit is None else f"S1_SMOKE_{limit}"
    root = config.output_dir / "stages" / stage_name
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "run_manifest.json"
    run_manifest = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "stage": stage_name,
        "full_design_digest": design_digest(build_design(config)),
        "implementation_sha256": _implementation_digest(),
        "selected_fit_ids": [row["fit_id"] for row in design],
        "shard_index": shard_index,
        "shard_count": shard_count,
        "case_ids": list(DEFAULT_CASES),
        "current_video_labels_used": False,
        "scientific_audit": {
            "enabled": config.scientific_audit.enabled,
            "applicability": "unlabeled candidate-surrogate panel after frozen selection",
        },
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != run_manifest:
            raise RuntimeError("existing stage manifest differs")
    else:
        _atomic_json(manifest_path, run_manifest)
    checkpoint = root / "fit_results.jsonl"
    rows, completed = _load_completed(checkpoint)
    started = time.monotonic()
    for index, specification in enumerate(design):
        if specification["fit_id"] in completed:
            continue
        result = evaluate_specification(specification, config)
        _append_checkpoint(checkpoint, result)
        rows.append(result)
        completed.add(specification["fit_id"])
        _atomic_json(root / "progress.json", {
            "status": "running", "completed": len(rows), "expected": len(design),
            "last_fit_id": specification["fit_id"],
            "elapsed_seconds_this_invocation": time.monotonic() - started,
        })
    _write_registry(root / "fit_registry.tsv", rows)
    summary = _summarize(rows, config)
    summary.update({
        "status": "complete", "stage": stage_name,
        "design_is_complete": len(rows) == len(design),
        "runtime_seconds_this_invocation": time.monotonic() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "interpretation": (
            "truth-known computational evidence only; no biological labels, "
            "precision, source identity, or cross-recording generalization"
        ),
    })
    _atomic_json(root / "summary.json", summary)
    _atomic_json(root / "validation.json", {
        "status": "pass" if summary["design_is_complete"] else "fail",
        "expected_fit_count": len(design), "observed_fit_count": len(rows),
        "unique_fit_count": len({row["fit_id"] for row in rows}),
        "all_fit_ids_expected": {row["fit_id"] for row in rows} == {
            row["fit_id"] for row in design
        },
    })
    _atomic_json(root / "progress.json", {
        "status": "complete", "completed": len(rows), "expected": len(design)
    })
    return summary


def merge_synthetic_shards(
    config: ICAWhiteningConfig,
    *,
    preflight_dir: str | Path,
    shard_count: int,
) -> dict[str, Any]:
    """Validate and merge exact non-overlapping shard coverage."""
    matching_preflight(config, preflight_dir)
    if shard_count < 2:
        raise ValueError("shard_count must be at least two")
    expected_design = build_design(config)
    expected = {row["fit_id"] for row in expected_design}
    root = config.output_dir / "stages" / "S1_NUMERICAL_TRUTH_KNOWN_SHARDED"
    if root.exists():
        raise FileExistsError(f"merged stage root exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    observed: set[str] = set()
    shard_records = []
    registry_fields = {
        "fit_id", "cell_id", "family", "objective", "whitening_geometry",
        "covariance_scope", "causality", "rank", "spatial_width_px",
        "temporal_width_frames", "point_id", "point_kind", "seed",
        "all_numerically_resolved", "converged_fraction",
        "mean_truth_source_correlation", "worst_truth_source_correlation",
        "mean_truth_crosstalk", "mean_trace_preservation", "unresolved_accuracy",
    }
    implementation_hashes = set()
    for shard_index in range(shard_count):
        shard = config.output_dir / "stages" / (
            f"S1_NUMERICAL_TRUTH_KNOWN_SHARD_{shard_index:02d}_OF_{shard_count:02d}"
        )
        manifest = json.loads((shard / "run_manifest.json").read_text(encoding="utf-8"))
        validation = json.loads((shard / "validation.json").read_text(encoding="utf-8"))
        if manifest.get("shard_index") != shard_index or manifest.get("shard_count") != shard_count:
            raise RuntimeError(f"shard identity mismatch: {shard}")
        if validation.get("status") != "pass" or not validation.get("all_fit_ids_expected"):
            raise RuntimeError(f"shard validation failed: {shard}")
        implementation_hashes.add(manifest["implementation_sha256"])
        count = 0
        with (shard / "fit_results.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                payload = json.loads(line)
                fit_id = payload["fit_id"]
                if fit_id in observed:
                    raise RuntimeError(f"duplicate fit across shards: {fit_id}")
                observed.add(fit_id)
                rows.append({key: payload.get(key) for key in registry_fields})
                count += 1
        shard_records.append({
            "shard_index": shard_index,
            "path": str(shard.relative_to(config.output_dir)),
            "fit_count": count,
            "implementation_sha256": manifest["implementation_sha256"],
        })
    if len(implementation_hashes) != 1:
        raise RuntimeError("shards used different implementation hashes")
    missing, extra = expected - observed, observed - expected
    if missing or extra:
        raise RuntimeError(
            f"shard coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    rows.sort(key=lambda row: row["fit_id"])
    _write_registry(root / "fit_registry.tsv", rows)
    summary = _summarize(rows, config)
    summary.update({
        "status": "complete",
        "stage": "S1_NUMERICAL_TRUTH_KNOWN_SHARDED",
        "design_is_complete": True,
        "design_digest": design_digest(expected_design),
        "implementation_sha256": next(iter(implementation_hashes)),
        "shard_count": shard_count,
        "interpretation": (
            "truth-known computational evidence only; no biological labels, "
            "precision, source identity, or cross-recording generalization"
        ),
    })
    validation = {
        "status": "pass",
        "expected_fit_count": len(expected),
        "observed_fit_count": len(observed),
        "unique_fit_count": len(observed),
        "missing_fit_count": 0,
        "extra_fit_count": 0,
        "single_implementation_hash": True,
        "all_shards_validated": True,
    }
    _atomic_json(root / "summary.json", summary)
    _atomic_json(root / "validation.json", validation)
    _atomic_json(root / "shard_manifest.json", {"shards": shard_records})
    _atomic_json(root / "llm_context.json", {
        "experiment_id": config.experiment_id,
        "entrypoint": "summary.json",
        "stage": "truth_known_factorial_screen",
        "labels_used": False,
        "fit_count": len(observed),
        "primary_table": "fit_registry.tsv",
        "shard_manifest": "shard_manifest.json",
        "decision": summary["decision"],
        "limitations": [
            "computational truth-known evidence only",
            "no biological precision or source identity",
            "no independent-recording confirmation",
        ],
    })
    _atomic_json(root / "artifact_index.json", {
        "artifacts": [
            {"path": "summary.json", "role": "stage_summary"},
            {"path": "validation.json", "role": "coverage_validation"},
            {"path": "fit_registry.tsv", "role": "complete_fit_registry"},
            {"path": "shard_manifest.json", "role": "detailed_checkpoint_map"},
            {"path": "llm_context.json", "role": "llm_entrypoint"},
        ]
    })
    return summary
