"""Bounded native-best development and held semi-synthetic evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter, time
from typing import Any

import numpy as np
from scipy import sparse

from neurobench.metrics.component_reconstruction import component_product_metrics
from neurobench.metrics.signal_movie import signal_movie_metrics

from .caiman_native import fit_caiman_movie
from .config import InformationSeparationConfig
from .consensus import fit_multistart_consensus
from .conclusive_config import ConclusiveBatchConfig
from .parzen_native import fit_spatial_stochastic_parzen_noisy_posterior
from .references import fit_dense_patch_fastica_wiener_reference
from .screen_runner import _atomic_json
from .semi_synthetic import make_real_background_fixture


MORPHOLOGIES = ("isolated", "overlap", "synchronous", "correlated", "similar_persistence")
SEEDS = (20260911, 20260921, 20260931)


def _configure_gpu_allocation_limit(
    config: ConclusiveBatchConfig,
) -> dict[str, Any]:
    """Apply the declared per-process CUDA allocator cap before any GPU fit."""
    device = str(config.resources["gpu_device"])
    if not device.startswith("cuda"):
        return {"device": device, "enabled": False, "reason": "non_cuda_device"}
    try:
        import torch

        if not torch.cuda.is_available():
            return {"device": device, "enabled": False, "reason": "cuda_unavailable"}
        index = torch.device(device).index or 0
        total_mib = torch.cuda.get_device_properties(index).total_memory / 2**20
        cap_mib = float(config.resources["gpu_allocation_cap_mib"])
        fraction = min(cap_mib / total_mib, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, index)
        return {
            "device": device,
            "enabled": True,
            "allocation_cap_mib": cap_mib,
            "device_total_mib": total_mib,
            "allocator_fraction": fraction,
        }
    except (ImportError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"could not enforce GPU allocation cap for {device}: {exc}"
        ) from exc


def _configurations(config: ConclusiveBatchConfig) -> list[dict[str, Any]]:
    allowed = {"dense_patch_fastica_wiener_reference", "spatial_noisy_parzen_infomax",
               "multistart_consensus", "caiman_cnmf", "caiman_cnmfe"}
    rows = []
    for method in config.methods:
        if method["enabled"] and method["method_id"] in allowed:
            rows.extend({"method_id": method["method_id"], "track": method["track"],
                         "parameters": dict(parameters)} for parameters in method["configurations"])
    return rows


def _specifications(scientific: InformationSeparationConfig, split: str):
    seeds = SEEDS[:1] if split == "development" else SEEDS[1:]
    for crop_index, origin in enumerate(scientific.semi_synthetic["crop_origins_xy"]):
        for morphology in MORPHOLOGIES:
            for amplitude in scientific.semi_synthetic["injection_amplitudes"]:
                for seed in seeds:
                    yield {"split": split, "crop_index": crop_index,
                           "crop_origin_xy": list(map(int, origin)),
                           "morphology": morphology, "amplitude": float(amplitude),
                           "seed": int(seed)}


def _key(specification: dict[str, Any], method: dict[str, Any]) -> str:
    encoded = json.dumps({"fixture": specification, "method": method}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:20]


def _run_method(fixture, method: dict[str, Any], scientific: InformationSeparationConfig,
                config: ConclusiveBatchConfig, artifact_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    method_id = method["method_id"]; parameters = method["parameters"]
    if method_id == "dense_patch_fastica_wiener_reference":
        result = fit_dense_patch_fastica_wiener_reference(
            fixture.observation, quiet_frames=32,
            patch_size=int(parameters["patch_size"]), rank=int(parameters["rank"]),
            sample_count=512, seed=fixture.metadata["source_frames_ui_inclusive"][0]+int(parameters["rank"]),
            wiener_lambda_z=float(parameters["wiener_lambda_z"]),
        )
        metrics = signal_movie_metrics(fixture.traces, fixture.footprints, result["signal"],
                                       native_background=fixture.native_background)
        return metrics, {"converged": bool(result["model_diagnostics"]["fastica_converged"]),
                         "reported_method_id": result["method_id"],
                         "diagnostics": {"model": result["model_diagnostics"], "application": result["application_diagnostics"]}}
    if method_id == "spatial_noisy_parzen_infomax":
        result = fit_spatial_stochastic_parzen_noisy_posterior(
            fixture.observation, quiet_frames=32, patch_size=int(parameters["patch_size"]),
            rank=int(parameters["rank"]), noise_scale=float(parameters["noise_scale"]),
            seed=int(fixture.metadata["source_frames_ui_inclusive"][0])+int(parameters["rank"]),
            device="cuda", sample_count=1024)
        metrics = signal_movie_metrics(fixture.traces, fixture.footprints, result["signal"],
                                       native_background=fixture.native_background)
        return metrics, {"converged": result["converged"],
                         "reported_method_id": result["method_id"],
                         "diagnostics": {"model": result["model_diagnostics"], "application": result["application_diagnostics"]}}
    if method_id == "multistart_consensus":
        result = fit_multistart_consensus(
            fixture.observation, base_method=str(parameters["base_method"]),
            rank=int(parameters["rank"]), starts=int(parameters["starts"]),
            scientific_config=scientific, seed=int(fixture.metadata["source_frames_ui_inclusive"][0]),
            device=str(config.resources["gpu_device"]))
        metrics = component_product_metrics(fixture.traces, fixture.footprints,
            result["sources"], result["spatial_maps"], background=fixture.native_background)
        return metrics, {"converged": result["converged"],
                         "reported_method_id": result["method_id"], "diagnostics": result["diagnostics"]}
    fit_caiman_movie(fixture.observation, method_id=method_id,
        parameters={**parameters, "n_processes": min(4, int(config.resources["maximum_caiman_processes"]))},
        output_dir=artifact_dir, python_executable=config.caiman_python)
    spatial_components = sparse.load_npz(artifact_dir/"result"/"spatial_components.npz").toarray()
    traces = np.load(artifact_dir/"result"/"temporal_components.npz")["C"]
    if traces.shape[0] == 0:
        return {}, {"converged": True, "reported_method_id": method_id,
                    "unresolved": True, "diagnostics": {"component_count": 0}}
    metrics = component_product_metrics(fixture.traces, fixture.footprints,
        traces, spatial_components, background=fixture.native_background)
    fit = json.loads((artifact_dir/"fit.json").read_text())
    return metrics, {"converged": True, "reported_method_id": method_id,
                     "unresolved": False, "diagnostics": fit}


def _valid(metrics: dict[str, Any], config: ConclusiveBatchConfig) -> bool:
    return bool(metrics
        and metrics["mean_peak_retention"] >= float(config.gates["minimum_peak_retention"])
        and metrics["mean_peak_retention"] <= 1.25
        and metrics["mean_area_retention"] >= float(config.gates["minimum_area_retention"])
        and metrics["mean_area_retention"] <= 1.25
        and metrics["mean_waveform_correlation"] >= float(config.gates["minimum_waveform_correlation"])
        and metrics["maximum_absolute_peak_error_frames"] <= int(config.gates["maximum_timing_error_frames"])
        and metrics["mean_footprint_iou"] >= 0.25)


def _select(rows: list[dict[str, Any]], methods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for method_id in sorted({row["method_id"] for row in rows}):
        candidates = []
        for method in [item for item in methods if item["method_id"] == method_id]:
            group = [row for row in rows if row["method_id"] == method_id and row["parameters"] == method["parameters"]]
            candidates.append((
                -sum(row["scientific_valid"] for row in group)/len(group),
                float(np.median([row["metrics"].get("neural_reconstruction_nmse", np.inf) for row in group])),
                json.dumps(method["parameters"], sort_keys=True), method,
            ))
        candidates.sort(key=lambda item: item[:3])
        selected.append(candidates[0][3])
    return selected


def run(config: ConclusiveBatchConfig, *, maximum_fits: int | None = None) -> dict[str, Any]:
    partial = Path(str(config.output_root)+".partial")
    scientific = InformationSeparationConfig.load(config.scientific_config_path)
    methods = _configurations(config)
    stage = partial/"stages"/"04_native_semi_synthetic"
    rows_root = stage/"rows"; artifacts = stage/"native_artifacts"
    rows_root.mkdir(parents=True, exist_ok=True); artifacts.mkdir(exist_ok=True)
    gpu_limit = _configure_gpu_allocation_limit(config)
    _atomic_json(stage/"recovery_resource_limits.json", {
        "schema_version": 1,
        "status": "conservative_recovery_limits_applied",
        "general_cpu_workers": int(config.resources["general_cpu_workers"]),
        "maximum_caiman_processes": int(config.resources["maximum_caiman_processes"]),
        "worker_threads": int(config.resources["worker_threads"]),
        "rss_soft_cap_mib": int(config.resources["rss_soft_cap_mib"]),
        "rss_hard_stop_mib": int(config.resources["rss_hard_stop_mib"]),
        "gpu": gpu_limit,
    })
    created = 0
    def execute(specifications, active_methods):
        nonlocal created
        for specification in specifications:
            fixture = make_real_background_fixture(
                config.source_video, quiet_start_ui=int(scientific.semi_synthetic["quiet_start_ui"]),
                quiet_end_ui=int(scientific.semi_synthetic["quiet_end_ui"]),
                crop_origin_xy=tuple(specification["crop_origin_xy"]),
                crop_size_px=int(scientific.semi_synthetic["crop_size_px"]),
                amplitude=float(specification["amplitude"]), seed=int(specification["seed"]),
                morphology_case=str(specification["morphology"]))
            for method in active_methods:
                key = _key(specification, method); destination = rows_root/f"{key}.json"
                if destination.exists(): continue
                if maximum_fits is not None and created >= maximum_fits:
                    return False
                started = perf_counter()
                metrics, execution = _run_method(fixture, method, scientific, config, artifacts/key)
                row = {"fit_id": key, "fixture": specification, "method_id": method["method_id"],
                       "track": method["track"], "parameters": method["parameters"],
                       "reported_method_id": execution["reported_method_id"],
                       "converged": execution["converged"], "scientific_valid": _valid(metrics, config),
                       "metrics": metrics, "diagnostics": execution["diagnostics"],
                       "runtime_seconds": perf_counter()-started}
                _atomic_json(destination, row); created += 1
                count = len(list(rows_root.glob("*.json")))
                _atomic_json(stage/"heartbeat.json", {"completed": count, "phase": specification["split"], "updated_unix": time()})
        return True
    development = list(_specifications(scientific, "development"))
    if not execute(development, methods):
        return {"status": "stage4_partial", "new_fits": created}
    dev_rows = [json.loads(path.read_text()) for path in rows_root.glob("*.json") if json.loads(path.read_text())["fixture"]["split"] == "development"]
    selected = _select(dev_rows, methods)
    _atomic_json(stage/"frozen_native_methods.json", {"selected": selected})
    if not execute(list(_specifications(scientific, "evaluation")), selected):
        return {"status": "stage4_partial", "new_fits": created, "selected": selected}
    rows = [json.loads(path.read_text()) for path in sorted(rows_root.glob("*.json"))]
    results = []
    for method in selected:
        group = [row for row in rows if row["fixture"]["split"] == "evaluation" and row["method_id"] == method["method_id"] and row["parameters"] == method["parameters"]]
        valid_fraction = sum(row["scientific_valid"] for row in group)/len(group)
        morphology = {name: sum(row["scientific_valid"] for row in group if row["fixture"]["morphology"] == name)/len([row for row in group if row["fixture"]["morphology"] == name]) for name in MORPHOLOGIES}
        passed = bool(valid_fraction >= 0.8 and min(morphology.values()) >= 0.5 and np.mean([row["converged"] for row in group]) >= 0.95)
        results.append({"method_id": method["method_id"], "parameters": method["parameters"],
                        "fit_count": len(group), "scientific_valid_fraction": valid_fraction,
                        "morphology_valid_fraction": morphology,
                        "converged_fraction": float(np.mean([row["converged"] for row in group])),
                        "median_neural_reconstruction_nmse": float(np.median([row["metrics"].get("neural_reconstruction_nmse", np.inf) for row in group])),
                        "gate_passed": passed})
    payload = {"schema_version": 1, "status": "native_semi_synthetic_complete",
               "fit_count": len(rows), "development_fit_count": len(dev_rows),
               "frozen_methods": selected, "method_results": results,
               "passing_methods": [row["method_id"] for row in results if row["gate_passed"]],
               "interpretation": "Native methods were tuned only on seed-20260911 fixtures and evaluated on disjoint seeds. Common-input methods that failed selective risk were not revived by this stage."}
    _atomic_json(stage/"metrics.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--maximum-fits", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(run(ConclusiveBatchConfig.load(args.config), maximum_fits=args.maximum_fits), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
