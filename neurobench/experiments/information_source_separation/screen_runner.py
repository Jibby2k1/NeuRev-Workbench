"""Resumable generated-fixture screen; implementation does not authorize execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from neurobench.algorithms.information_source_separation import (
    fit_kernel_hsic_pairwise_rotation,
    fit_knn_mi_pairwise_rotation,
    fit_multilag_sobi,
)
from neurobench.metrics.source_separation import aligned_source_metrics

from .config import InformationSeparationConfig
from .design import method_screen, screen_fixtures, select_finalists
from .qualification import qualify_temporal_components
from .references import fit_amplitude_pca_reference
from .synthetic import make_spatiotemporal_fixture


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _fit_id(fixture: dict[str, Any], method_key: str) -> str:
    payload = json.dumps(
        {"fixture": fixture, "method_key": method_key},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _execute_method(
    movie: np.ndarray,
    method_id: str,
    parameters: dict[str, Any],
    config: InformationSeparationConfig,
    seed: int,
) -> dict[str, Any]:
    observations = movie.reshape(len(movie), -1).T
    if method_id == "pca_reference":
        result = fit_amplitude_pca_reference(movie, rank=int(parameters["rank"]))
        centered = observations - observations.mean(axis=1, keepdims=True)
        residual = float(
            np.linalg.norm(movie - result.reconstruction)
            / max(np.linalg.norm(centered), np.finfo(float).eps)
        )
        return {
            "reported_method_id": result.method_id,
            "sources": result.temporal_sources,
            "spatial_maps": result.spatial_maps,
            "converged": result.converged,
            "iterations": result.iterations,
            "relative_observation_residual": residual,
            "diagnostics": result.diagnostics,
        }
    if method_id == "multilag_sobi":
        result = fit_multilag_sobi(observations, **parameters)
    elif method_id == "kernel_hsic_pairwise_rotation":
        settings = config.methods[method_id]
        result = fit_kernel_hsic_pairwise_rotation(
            observations,
            **parameters,
            angle_step_degrees=float(settings["angle_step_degrees"]),
            max_sweeps=int(settings["max_sweeps"]),
            max_fit_samples=int(settings["max_fit_samples"]),
            seed=int(seed),
        )
    elif method_id == "knn_mi_pairwise_rotation":
        settings = config.methods[method_id]
        result = fit_knn_mi_pairwise_rotation(
            observations,
            **parameters,
            angle_step_degrees=float(settings["angle_step_degrees"]),
            max_sweeps=int(settings["max_sweeps"]),
            max_fit_samples=int(settings["max_fit_samples"]),
            seed=int(seed),
        )
    else:
        raise ValueError(f"screen method is not implemented: {method_id}")
    return {
        "reported_method_id": result.method_id,
        "sources": result.sources,
        "spatial_maps": result.mixing,
        "converged": result.converged,
        "iterations": result.iterations,
        "relative_observation_residual": float(
            result.diagnostics["relative_subspace_closure_error"]
        ),
        "diagnostics": result.diagnostics,
    }


def run_generated_screen(
    config: InformationSeparationConfig,
    *,
    output_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute the preregistered generated screen with per-fit resumability.

    Calling this function is a full 672-fit generated run and requires explicit
    user selection after a resource preflight. Merely importing it has no side
    effects.
    """
    output = output_dir.resolve()
    partial = Path(str(output) + ".partial")
    if output.exists():
        raise FileExistsError(f"completed screen exists: {output}")
    if partial.exists() and not resume:
        raise FileExistsError("partial screen exists; pass resume=True after audit")
    if not partial.exists():
        partial.mkdir(parents=True, exist_ok=False)
        (partial / "fits").mkdir()
        _atomic_json(partial / "config.resolved.json", config.to_dict())
    else:
        stored = json.loads(
            (partial / "config.resolved.json").read_text(encoding="utf-8")
        )
        if stored != config.to_dict():
            raise RuntimeError("resume manifest differs from partial screen")
    fixtures = screen_fixtures(config)
    methods = method_screen(config)
    expected = len(fixtures) * len(methods)
    progress_path = partial / "progress.jsonl"
    rows: list[dict[str, Any]] = []
    for fixture_specification in fixtures:
        fixture_dict = {
            "case_id": fixture_specification.case_id,
            "seed": fixture_specification.seed,
            "snr": fixture_specification.snr,
        }
        fixture = make_spatiotemporal_fixture(
            fixture_specification.case_id,
            seed=fixture_specification.seed,
            frame_count=int(config.generated["frame_count"]),
            shape=(
                int(config.generated["height_px"]),
                int(config.generated["width_px"]),
            ),
            snr=fixture_specification.snr,
            frame_period_ms=float(config.generated["frame_period_ms"]),
        )
        for method in methods:
            fit_id = _fit_id(fixture_dict, method.key)
            fit_path = partial / "fits" / f"{fit_id}.json"
            if fit_path.exists():
                rows.append(json.loads(fit_path.read_text(encoding="utf-8")))
                continue
            started = perf_counter()
            execution = _execute_method(
                fixture.observation,
                method.method_id,
                method.parameters,
                config,
                fixture_specification.seed,
            )
            recovery = aligned_source_metrics(fixture.traces, execution["sources"])
            qualification = qualify_temporal_components(
                execution["spatial_maps"],
                execution["sources"],
                spatial_shape=(
                    int(config.generated["height_px"]),
                    int(config.generated["width_px"]),
                ),
            )
            unresolved_expected = fixture_specification.case_id == "unresolved"
            row = {
                **fixture_dict,
                "fit_id": fit_id,
                "method_id": method.method_id,
                "reported_method_id": execution["reported_method_id"],
                "configuration_json": json.dumps(method.parameters, sort_keys=True),
                "converged": bool(execution["converged"]),
                "iterations": int(execution["iterations"]),
                "runtime_seconds": float(perf_counter() - started),
                "relative_observation_residual": execution["relative_observation_residual"],
                "mean_absolute_correlation": recovery["mean_absolute_correlation"],
                "worst_absolute_correlation": recovery["worst_absolute_correlation"],
                "mean_aligned_nmse": recovery["mean_aligned_nmse"],
                "worst_aligned_nmse": recovery["worst_aligned_nmse"],
                "mean_absolute_crosstalk": recovery["mean_absolute_crosstalk"],
                "worst_absolute_crosstalk": recovery["worst_absolute_crosstalk"],
                "unresolved_expected": unresolved_expected,
                "reported_unresolved": qualification["status"] == "unresolved",
                "unresolved_correct": (
                    unresolved_expected == (qualification["status"] == "unresolved")
                ),
                "qualification_status": qualification["status"],
                "qualification_top_score": qualification["top_score"],
                "qualification_margin": qualification["score_margin"],
                "scientific_reconstruction_available": False,
                "amplitude_timing_gate_applied": False,
                "fit_diagnostics": execution["diagnostics"],
                "qualification": qualification,
            }
            _atomic_json(fit_path, row)
            rows.append(row)
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "completed": len(rows), "total": expected,
                    "fit_id": fit_id, "case_id": fixture_specification.case_id,
                    "method_id": method.method_id,
                }, sort_keys=True) + "\n")
    if len(rows) != expected:
        raise RuntimeError(f"screen row count differs: {len(rows)} != {expected}")
    selection = select_finalists(
        rows,
        expected_fixture_count=len(fixtures),
        finalists_per_new_method=2,
        require_unresolved_accuracy=bool(
            config.selection["require_correct_unresolved"]
        ),
    )
    payload = {
        "schema_version": 1,
        "status": "generated_screen_complete",
        "experiment_id": config.experiment_id,
        "fit_count": len(rows),
        "fixture_count": len(fixtures),
        "configuration_count": len(methods),
        "gpu_used": False,
        "real_spon_benchmark_run": False,
        "selection": selection,
        "interpretation": (
            "Generated truth selects candidates for confirmation only. Scale-aligned "
            "source recovery is not an amplitude-faithful scientific reconstruction."
        ),
    }
    _atomic_json(partial / "metrics.json", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(output)
    return payload
