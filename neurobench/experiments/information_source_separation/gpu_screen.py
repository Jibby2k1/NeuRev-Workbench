"""Parity-gated CUDA execution for the preregistered generated screen."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

import numpy as np

from neurobench.algorithms.information_source_separation import normalized_hsic
from neurobench.algorithms.information_source_separation_cuda import (
    fit_kernel_hsic_pairwise_rotation_cuda,
    normalized_hsic_cuda,
)
from neurobench.metrics.source_separation import aligned_source_metrics

from .config import InformationSeparationConfig
from .design import method_screen, screen_fixtures, select_finalists, staged_fit_counts
from .qualification import qualify_temporal_components
from .screen_runner import _atomic_json, _execute_method, _fit_id
from .synthetic import make_spatiotemporal_fixture


def audit_cuda_screen(
    config: InformationSeparationConfig,
    *,
    output_dir: Path,
    device: str = "cuda:0",
    parity_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Perform a read-only CUDA/resource/parity audit without authorizing work."""
    output = output_dir.resolve()
    partial = Path(str(output) + ".partial")
    counts = staged_fit_counts(config)
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            index = torch.device(device).index or 0
            properties = torch.cuda.get_device_properties(index)
            total_mib = properties.total_memory / 2**20
            free_bytes, _ = torch.cuda.mem_get_info(index)
            free_mib = free_bytes / 2**20
            gpu_name = properties.name
        else:
            total_mib = free_mib = 0.0
            gpu_name = None
    except (ImportError, RuntimeError, ValueError):
        cuda_available = False
        total_mib = free_mib = 0.0
        gpu_name = None
    parity_error = None
    if cuda_available:
        rng = np.random.default_rng(20260801)
        left = rng.normal(size=96)
        right = 0.4 * left**2 + rng.normal(size=96)
        cpu_score = normalized_hsic(left, right, bandwidth_scale=1.0)
        gpu_score = normalized_hsic_cuda(left, right, bandwidth_scale=1.0, device=device)
        parity_error = abs(cpu_score - gpu_score)
    probe = output.parent
    while not probe.exists():
        probe = probe.parent
    free_disk_mib = shutil.disk_usage(probe).free / 2**20
    gates = {
        "output_absent": not output.exists(),
        "partial_output_absent": not partial.exists(),
        "cuda_available": cuda_available,
        "gpu_free_memory_at_least_4096_mib": free_mib >= 4096,
        "cpu_gpu_hsic_parity": parity_error is not None and parity_error <= parity_tolerance,
        "disk_headroom_sufficient": free_disk_mib >= int(config.resources["min_free_disk_mib"]),
        "cpu_threads_bounded": 1 <= int(config.resources["cpu_threads"]) <= 8,
    }
    return {
        "schema_version": 1,
        "kind": "information_source_separation_cuda_generated_screen_read_only_preflight",
        "experiment_id": config.experiment_id,
        "ready_for_explicit_user_selection": bool(all(gates.values())),
        "run_authorized": False,
        "gates": gates,
        "output_dir": str(output),
        "counts": counts,
        "execution_contract": {
            "device": device,
            "gpu_method": "kernel_hsic_pairwise_rotation",
            "gpu_scope": "normalized_hsic_objective_only",
            "cpu_methods": ["pca_reference", "multilag_sobi", "knn_mi_pairwise_rotation"],
            "dtype": "float64",
            "parity_tolerance": parity_tolerance,
        },
        "gpu": {
            "name": gpu_name,
            "total_memory_mib": total_mib,
            "free_memory_mib": free_mib,
            "hsic_absolute_parity_error": parity_error,
        },
        "resources": {
            "free_disk_mib": free_disk_mib,
            **config.resources,
        },
        "interpretation": (
            "CUDA accelerates only the HSIC dependence objective. CPU reference methods "
            "remain unchanged, and the later confirmation/Spon stages remain unauthorized."
        ),
    }


def _execute_cuda_method(
    movie: np.ndarray,
    method_id: str,
    parameters: dict[str, Any],
    config: InformationSeparationConfig,
    seed: int,
    device: str,
) -> dict[str, Any]:
    if method_id != "kernel_hsic_pairwise_rotation":
        result = _execute_method(movie, method_id, parameters, config, seed)
        result["execution_backend"] = "numpy_scipy_cpu"
        return result
    observations = movie.reshape(len(movie), -1).T
    settings = config.methods[method_id]
    fitted = fit_kernel_hsic_pairwise_rotation_cuda(
        observations,
        **parameters,
        angle_step_degrees=float(settings["angle_step_degrees"]),
        max_sweeps=int(settings["max_sweeps"]),
        max_fit_samples=int(settings["max_fit_samples"]),
        seed=int(seed),
        device=device,
    )
    return {
        "reported_method_id": fitted.method_id,
        "sources": fitted.sources,
        "spatial_maps": fitted.mixing,
        "converged": fitted.converged,
        "iterations": fitted.iterations,
        "relative_observation_residual": float(fitted.diagnostics["relative_subspace_closure_error"]),
        "diagnostics": fitted.diagnostics,
        "execution_backend": "torch_cuda",
    }


def run_cuda_generated_screen(
    config: InformationSeparationConfig,
    *,
    output_dir: Path,
    device: str = "cuda:0",
    resume: bool = False,
) -> dict[str, Any]:
    """Execute the explicitly selected 672-fit mixed CUDA/CPU screen."""
    output = output_dir.resolve()
    partial = Path(str(output) + ".partial")
    if output.exists():
        raise FileExistsError(f"completed screen exists: {output}")
    if partial.exists() and not resume:
        raise FileExistsError("partial screen exists; pass resume=True after audit")
    execution_contract = {
        "hsic_backend": "torch_cuda", "device": device, "dtype": "float64",
        "cpu_reference_backend": "numpy_scipy_cpu",
    }
    if not partial.exists():
        partial.mkdir(parents=True, exist_ok=False)
        (partial / "fits").mkdir()
        _atomic_json(partial / "config.resolved.json", {
            "scientific_config": config.to_dict(), "execution": execution_contract,
        })
    else:
        stored = json.loads((partial / "config.resolved.json").read_text(encoding="utf-8"))
        expected = {"scientific_config": config.to_dict(), "execution": execution_contract}
        if stored != expected:
            raise RuntimeError("resume manifest differs from partial CUDA screen")
    fixtures = screen_fixtures(config)
    methods = method_screen(config)
    expected_count = len(fixtures) * len(methods)
    rows: list[dict[str, Any]] = []
    progress_path = partial / "progress.jsonl"
    for specification in fixtures:
        fixture_dict = {"case_id": specification.case_id, "seed": specification.seed, "snr": specification.snr}
        fixture = make_spatiotemporal_fixture(
            specification.case_id, seed=specification.seed,
            frame_count=int(config.generated["frame_count"]),
            shape=(int(config.generated["height_px"]), int(config.generated["width_px"])),
            snr=specification.snr, frame_period_ms=float(config.generated["frame_period_ms"]),
        )
        for method in methods:
            fit_id = _fit_id(fixture_dict, method.key)
            fit_path = partial / "fits" / f"{fit_id}.json"
            if fit_path.exists():
                rows.append(json.loads(fit_path.read_text(encoding="utf-8")))
                continue
            started = perf_counter()
            execution = _execute_cuda_method(
                fixture.observation, method.method_id, method.parameters,
                config, specification.seed, device,
            )
            recovery = aligned_source_metrics(fixture.traces, execution["sources"])
            qualification = qualify_temporal_components(
                execution["spatial_maps"], execution["sources"],
                spatial_shape=(int(config.generated["height_px"]), int(config.generated["width_px"])),
            )
            unresolved_expected = specification.case_id == "unresolved"
            row = {
                **fixture_dict, "fit_id": fit_id, "method_id": method.method_id,
                "reported_method_id": execution["reported_method_id"],
                "configuration_json": json.dumps(method.parameters, sort_keys=True),
                "execution_backend": execution["execution_backend"],
                "converged": bool(execution["converged"]), "iterations": int(execution["iterations"]),
                "runtime_seconds": float(perf_counter() - started),
                "relative_observation_residual": execution["relative_observation_residual"],
                **recovery,
                "unresolved_expected": unresolved_expected,
                "reported_unresolved": qualification["status"] == "unresolved",
                "unresolved_correct": unresolved_expected == (qualification["status"] == "unresolved"),
                "qualification_status": qualification["status"],
                "qualification_top_score": qualification["top_score"],
                "qualification_margin": qualification["score_margin"],
                "scientific_reconstruction_available": False,
                "amplitude_timing_gate_applied": False,
                "fit_diagnostics": execution["diagnostics"], "qualification": qualification,
            }
            _atomic_json(fit_path, row)
            rows.append(row)
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "completed": len(rows), "total": expected_count, "fit_id": fit_id,
                    "case_id": specification.case_id, "method_id": method.method_id,
                    "execution_backend": execution["execution_backend"],
                }, sort_keys=True) + "\n")
    if len(rows) != expected_count:
        raise RuntimeError(f"screen row count differs: {len(rows)} != {expected_count}")
    selection = select_finalists(
        rows, expected_fixture_count=len(fixtures), finalists_per_new_method=2,
        require_unresolved_accuracy=bool(config.selection["require_correct_unresolved"]),
    )
    payload = {
        "schema_version": 1, "status": "generated_screen_complete",
        "experiment_id": config.experiment_id, "fit_count": len(rows),
        "fixture_count": len(fixtures), "configuration_count": len(methods),
        "gpu_used": True, "gpu_scope": "normalized_hsic_objective_only",
        "execution_contract": execution_contract, "real_spon_benchmark_run": False,
        "selection": selection,
        "interpretation": (
            "Generated truth selects candidates for confirmation only. Scale-aligned source "
            "recovery is not an amplitude-faithful scientific reconstruction."
        ),
    }
    _atomic_json(partial / "metrics.json", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(output)
    return payload
