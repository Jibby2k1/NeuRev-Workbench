"""Collision-safe deterministic tiny runner for numerical contract validation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from neurobench.algorithms.information_source_separation import (
    fit_kernel_hsic_pairwise_rotation,
    fit_knn_mi_pairwise_rotation,
    fit_multilag_sobi,
    pca_whiten,
)
from neurobench.metrics.source_separation import aligned_source_metrics

from .config import InformationSeparationConfig
from .preflight import audit
from .synthetic import make_spatiotemporal_fixture


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _metric_row(
    *,
    case_id: str,
    seed: int,
    snr: float,
    method_id: str,
    configuration: dict[str, Any],
    sources: np.ndarray,
    truth: np.ndarray,
    runtime_seconds: float,
    converged: bool,
    iterations: int,
    closure_error: float,
) -> dict[str, Any]:
    metrics = aligned_source_metrics(truth, sources)
    return {
        "case_id": case_id,
        "seed": int(seed),
        "snr": float(snr),
        "method_id": method_id,
        "configuration_json": json.dumps(configuration, sort_keys=True),
        "converged": bool(converged),
        "iterations": int(iterations),
        "runtime_seconds": float(runtime_seconds),
        "relative_subspace_closure_error": float(closure_error),
        "mean_absolute_correlation": metrics["mean_absolute_correlation"],
        "worst_absolute_correlation": metrics["worst_absolute_correlation"],
        "mean_aligned_nmse": metrics["mean_aligned_nmse"],
        "worst_aligned_nmse": metrics["worst_aligned_nmse"],
        "mean_absolute_crosstalk": metrics["mean_absolute_crosstalk"],
        "worst_absolute_crosstalk": metrics["worst_absolute_crosstalk"],
        "identifiability_metric": "permutation_sign_scale_aligned_temporal_sources",
        "scientific_reconstruction_available": False,
        "amplitude_timing_gate_applied": False,
        "unresolved_expected": case_id in {"pure_noise", "unresolved"},
        "reported_unresolved": False,
    }


def run_tiny_smoke(
    config: InformationSeparationConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Run two fixtures and one configuration per implemented method."""
    output = output_dir.resolve()
    partial = Path(str(output) + ".partial")
    preflight = audit(config, output_dir=output)
    if not preflight["ready_for_tiny_cpu_smoke"]:
        raise RuntimeError(f"tiny smoke preflight failed: {preflight['gates']}")
    partial.mkdir(parents=True, exist_ok=False)
    _atomic_json(partial / "preflight.json", preflight)
    _atomic_json(partial / "config.resolved.json", config.to_dict())
    cases = ["isolated", "unresolved"]
    seed = int(config.generated["seeds"][0])
    snr = float(config.generated["snr_levels"][len(config.generated["snr_levels"]) // 2])
    rows: list[dict[str, Any]] = []
    progress = partial / "progress.jsonl"
    for case_id in cases:
        fixture = make_spatiotemporal_fixture(
            case_id,
            seed=seed,
            frame_count=int(config.generated["frame_count"]),
            shape=(int(config.generated["height_px"]), int(config.generated["width_px"])),
            snr=snr,
            frame_period_ms=float(config.generated["frame_period_ms"]),
        )
        observations = fixture.observation.reshape(len(fixture.observation), -1).T
        methods: list[tuple[str, dict[str, Any]]] = []
        if config.methods["pca_reference"]["enabled"]:
            methods.append(("pca_reference", {"rank": int(config.methods["pca_reference"]["ranks"][0])}))
        if config.methods["multilag_sobi"]["enabled"]:
            methods.append(("multilag_sobi", {
                "rank": int(config.methods["multilag_sobi"]["ranks"][0]),
                "lags": list(config.methods["multilag_sobi"]["lag_sets"][1]),
                "covariance_shrinkage": float(config.methods["multilag_sobi"]["covariance_shrinkages"][1]),
            }))
        if config.methods["kernel_hsic_pairwise_rotation"]["enabled"]:
            methods.append(("kernel_hsic_pairwise_rotation", {
                "rank": int(config.methods["kernel_hsic_pairwise_rotation"]["ranks"][0]),
                "bandwidth_scale": 1.0,
            }))
        if config.methods["knn_mi_pairwise_rotation"]["enabled"]:
            methods.append(("knn_mi_pairwise_rotation", {
                "rank": int(config.methods["knn_mi_pairwise_rotation"]["ranks"][0]),
                "neighbors": 5,
            }))
        for method_id, specification in methods:
            started = perf_counter()
            if method_id == "pca_reference":
                sources, model = pca_whiten(observations, rank=specification["rank"])
                converged, iterations = True, 0
                closure = float(
                    np.linalg.norm(
                        observations - (model.dewhitening @ sources + model.mean[:, None])
                    ) / max(np.linalg.norm(observations - model.mean[:, None]), np.finfo(float).eps)
                )
            elif method_id == "multilag_sobi":
                result = fit_multilag_sobi(observations, **specification)
                sources, converged, iterations = result.sources, result.converged, result.iterations
                closure = result.diagnostics["relative_subspace_closure_error"]
            elif method_id == "kernel_hsic_pairwise_rotation":
                method = config.methods[method_id]
                result = fit_kernel_hsic_pairwise_rotation(
                    observations, **specification,
                    angle_step_degrees=float(method["angle_step_degrees"]),
                    max_sweeps=min(2, int(method["max_sweeps"])),
                    max_fit_samples=min(96, int(method["max_fit_samples"])),
                    seed=seed,
                )
                sources, converged, iterations = result.sources, result.converged, result.iterations
                closure = result.diagnostics["relative_subspace_closure_error"]
            else:
                method = config.methods[method_id]
                result = fit_knn_mi_pairwise_rotation(
                    observations, **specification,
                    angle_step_degrees=15.0,
                    max_sweeps=min(2, int(method["max_sweeps"])),
                    max_fit_samples=min(160, int(method["max_fit_samples"])),
                    seed=seed,
                )
                sources, converged, iterations = result.sources, result.converged, result.iterations
                closure = result.diagnostics["relative_subspace_closure_error"]
            row = _metric_row(
                case_id=case_id, seed=seed, snr=snr, method_id=method_id,
                configuration=specification, sources=sources, truth=fixture.traces,
                runtime_seconds=perf_counter() - started,
                converged=converged, iterations=iterations, closure_error=closure,
            )
            rows.append(row)
            with progress.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "completed": len(rows), "total": preflight["tiny_smoke_fit_count"],
                    "case_id": case_id, "method_id": method_id,
                }, sort_keys=True) + "\n")
    payload = {
        "schema_version": 1,
        "status": "tiny_smoke_complete_not_scientific_selection",
        "experiment_id": config.experiment_id,
        "fit_count": len(rows),
        "full_generated_matrix_run": False,
        "semi_synthetic_run": False,
        "spon_benchmark_run": False,
        "gpu_used": False,
        "model_selection_performed": False,
        "cnmf_backend": preflight["cnmf_backend"],
        "gated_methods": ["group_energy_isa", "spatial_noisy_parzen_infomax"],
        "rows": rows,
        "interpretation": (
            "This smoke validates interfaces, finite execution, artifact contracts, "
            "and source-alignment metrics only. Scale-aligned temporal recovery does "
            "not establish amplitude-faithful movie reconstruction or benchmark utility."
        ),
    }
    _atomic_json(partial / "metrics.json", payload)
    _atomic_tsv(partial / "comparison.tsv", rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(output)
    return payload
