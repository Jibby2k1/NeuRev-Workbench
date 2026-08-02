"""Guarded C1-C3 spatial ICA screen on the accepted Parzen innovation."""
from __future__ import annotations

from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any, Callable

import numpy as np
import tifffile

from neurobench.algorithms.spatial_patch_ica import (
    fit_parzen_shrinkage,
    fit_spatial_patch_fastica,
    sample_spatial_patches,
)
from neurobench.algorithms.spatial_patch_ica_reconstruction import (
    dense_convolutional_reconstruction,
    patch_lattice_reconstruction,
)
from neurobench.experiments.learnable_contrast import core as label_core

from .denoise_audit import (
    _display_scale,
    _roi_matrix,
    _synthetic_fixture,
    _variant_metrics,
    _write_tiffs,
)
from .innovation_grid import (
    _atomic_json,
    _available_ram_mib,
    _progress,
    _sha256,
    _snapshots,
)
from .signal_noise_split import (
    _coefficients,
    _innovation_residual,
    _quiet_standardization,
)
from .spatial_ica_config import SpatialICAConfig


VARIANT_IDS = (
    "c1_patch_fastica",
    "c2_dense_convolutional_fastica",
    "c3_dense_convolutional_parzen",
)


def preflight(config: SpatialICAConfig, *, write_artifacts: bool = True) -> dict[str, Any]:
    inputs = (config.source_video, config.labels_tsv, config.architecture_manifest)
    missing = [str(path) for path in inputs if not path.is_file()]
    shape = None
    dtype = None
    bounds = labels_valid = fit_valid = finite = False
    labels: list[dict[str, Any]] = []
    gpu: dict[str, Any] = {"available": False}
    if not missing:
        video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
        shape, dtype = list(video.shape), str(video.dtype)
        start = int(config.frames["review_start_ui"]) - 1
        stop = int(config.frames["review_end_ui"])
        bounds = (
            video.ndim == 3
            and 0 <= start < int(config.frames["quiet_end_ui"]) <= stop <= len(video)
        )
        finite = bounds and bool(np.isfinite(video[start:stop:20, ::16, ::16]).all())
        labels = label_core.load_labels(config.labels_tsv)
        labels_valid = bool(
            len(labels) == 79
            and len({row["roi_identity"] for row in labels}) == 27
            and all(
                0 <= row["x_px"] < video.shape[2]
                and 0 <= row["y_px"] < video.shape[1]
                for row in labels
            )
        )
        architecture = json.loads(config.architecture_manifest.read_text(encoding="utf-8"))
        fit = architecture.get("raw_stochastic_fit", {})
        fit_valid = bool(
            architecture.get("source_video") == str(config.source_video)
            and fit.get("classification_status") == "resolved"
            and fit.get("optimizer_converged") is True
            and fit.get("safety", {}).get("status") == "accepted"
        )
    try:
        import torch

        gpu["available"] = torch.cuda.is_available()
        if gpu["available"]:
            free, total = torch.cuda.mem_get_info()
            gpu.update(
                name=torch.cuda.get_device_name(0),
                free_mib=free / 2**20,
                total_mib=total / 2**20,
            )
    except ImportError:
        pass
    frame_count = (
        int(config.frames["review_end_ui"])
        - int(config.frames["review_start_ui"])
        + 1
    )
    pixels = 0 if shape is None else int(shape[1]) * int(shape[2])
    dense_mib = frame_count * pixels * 4 / 2**20
    patch = int(config.model["patch_size"])
    rank = int(config.model["rank"])
    convolution_workspace = (
        int(config.resources["frame_batch_size"])
        * pixels * (patch * patch + 2 * rank) * 4 / 2**20
    )
    estimated_ram = 4.5 * dense_mib + 1024
    estimated_gpu = 2.5 * convolution_workspace + 512
    uncompressed_output = config.variant_count * 2 * frame_count * pixels * 2 / 2**20
    probe = config.output_dir.parent
    while not probe.exists():
        probe = probe.parent
    free_disk = shutil.disk_usage(probe).free / 2**20
    cuda_requested = config.resources["device"] == "cuda"
    gates = {
        "inputs_exist": not missing,
        "source_is_npy": config.source_video.suffix == ".npy",
        "frame_bounds_valid": bounds,
        "finite_sample": finite,
        "labels_valid": labels_valid,
        "accepted_fit_matches_source": fit_valid,
        "output_absent": not config.output_dir.exists(),
        "partial_output_absent": not Path(str(config.output_dir) + ".partial").exists(),
        "preflight_separate_from_output": config.preflight_dir != config.output_dir,
        "ram_cap_sufficient": estimated_ram <= int(config.resources["max_ram_mib"]),
        "available_ram_sufficient": estimated_ram <= _available_ram_mib(),
        "disk_headroom_sufficient": free_disk >= int(config.resources["min_free_disk_mib"]),
        "output_cap_sufficient": uncompressed_output <= int(config.resources["max_output_mib"]),
        "requested_device_available": (not cuda_requested) or gpu["available"],
        "gpu_memory_cap_sufficient": (
            (not cuda_requested)
            or estimated_gpu <= int(config.resources["max_gpu_memory_mib"])
        ),
        "live_gpu_memory_sufficient": (
            (not cuda_requested) or estimated_gpu <= gpu.get("free_mib", 0)
        ),
    }
    payload = {
        "schema_version": 1,
        "kind": "read_only_parzen_innovation_spatial_ica_preflight",
        "experiment_id": config.experiment_id,
        "ready": all(gates.values()),
        "gates": gates,
        "variant_ids": list(VARIANT_IDS),
        "variant_count": len(VARIANT_IDS),
        "source_shape": shape,
        "source_dtype": dtype,
        "label_rows": len(labels),
        "roi_identities": len({row["roi_identity"] for row in labels}),
        "resources": {
            "estimated_peak_ram_mib": estimated_ram,
            "available_ram_mib": _available_ram_mib(),
            "estimated_peak_gpu_memory_mib": estimated_gpu,
            "uncompressed_output_mib": uncompressed_output,
            "free_disk_mib": free_disk,
            "gpu": gpu,
            **config.resources,
        },
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in inputs if path.is_file()
        ],
        "system_snapshot": _snapshots(),
        "scientific_contract": (
            "The unsupervised exploratory fit is transductive across the review "
            "window. Sparse labels are used only for evaluation; unmatched "
            "candidates remain unknown rather than negative."
        ),
    }
    if write_artifacts:
        config.preflight_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(config.preflight_dir / "preflight.json", payload)
        _atomic_json(config.preflight_dir / "config.resolved.json", config.to_dict())
        if not missing and bounds:
            label_core._write_overlay(
                np.load(config.source_video, mmap_mode="r", allow_pickle=False),
                labels,
                config.preflight_dir / "label_projection_overlay.png",
            )
    if not payload["ready"]:
        raise RuntimeError(f"spatial ICA preflight failed: {payload}")
    return payload


def _matching_preflight(config: SpatialICAConfig) -> dict[str, Any]:
    audit = json.loads((config.preflight_dir / "preflight.json").read_text(encoding="utf-8"))
    resolved = json.loads(
        (config.preflight_dir / "config.resolved.json").read_text(encoding="utf-8")
    )
    if not audit.get("ready") or resolved != config.to_dict():
        raise RuntimeError("run requires a matching ready preflight")
    if config.output_dir.exists() or Path(str(config.output_dir) + ".partial").exists():
        raise FileExistsError("completed or partial output already exists")
    return audit


def _write_model_artifacts(destination: Path, model: Any) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "spatial_patch_ica_model.npz"
    temporary = destination / "spatial_patch_ica_model.partial.npz"
    np.savez_compressed(
        temporary,
        patch_mean=model.patch_mean,
        analysis_filters=model.analysis_filters,
        synthesis_atoms=model.synthesis_atoms,
        component_scale=model.component_scale,
        explained_variance_ratio=model.explained_variance_ratio,
    )
    temporary.replace(model_path)
    filters = model.analysis_filters.reshape(
        model.rank, model.patch_size, model.patch_size
    )
    magnitude = max(float(np.percentile(np.abs(filters), 99.5)), 1e-8)
    display = np.rint(
        (np.clip(filters / magnitude, -1, 1) + 1) * 32767.5
    ).astype(np.uint16)
    filter_path = destination / "analysis_filters_signed.tif"
    tifffile.imwrite(
        filter_path,
        display,
        photometric="minisblack",
        compression="zlib",
        metadata=None,
        description=json.dumps(
            {"source_limits": [-magnitude, magnitude], "zero_display_value": 32768},
            sort_keys=True,
        ),
    )
    return {
        "model_npz": str(model_path.relative_to(destination.parent)),
        "analysis_filters_tiff": str(filter_path.relative_to(destination.parent)),
    }


def _report(metrics: dict[str, Any]) -> str:
    rows = metrics["variants"]
    lines = [
        f"# {metrics['experiment_id']}",
        "",
        "## Result",
        "",
        metrics["conclusion"],
        "",
        "| Lane | Recall | Fixed-budget recall | Candidates | Peak | Area | Peak error | Quiet RMS | Synthetic r | Compute s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['variant_id']}` | {row['mean_recall']:.3f} | "
            f"{row['fixed_budget_mean_recall']:.3f} | {row['event_candidates']} | "
            f"{row['median_peak_retention']:.3f} | "
            f"{row['median_area_retention']:.3f} | "
            f"{row['median_peak_frame_error']:.1f} | "
            f"{row['quiet_signal_rms_ratio']:.3f} | "
            f"{row['synthetic_correlation']:.3f} | "
            f"{row['runtime_seconds_compute']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "This is an exploratory transductive architecture screen, not a held-burst "
            "generalization result. The patch and dense-Wiener lanes share the same "
            "learned filters, so their difference isolates dense translation-shared "
            "application. The Parzen lane then changes only component shrinkage.",
            "",
            "Sparse-label candidate counts are burden, not false-positive counts. "
            "Review each signal and remainder TIFF before advancing a lane.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config: SpatialICAConfig) -> dict[str, Any]:
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(config.resources["cpu_threads"])
    audit = _matching_preflight(config)
    partial = Path(str(config.output_dir) + ".partial")
    partial.mkdir(parents=True)
    _atomic_json(partial / "config.resolved.json", config.to_dict())
    _atomic_json(partial / "preflight.json", audit)
    progress = partial / "progress.jsonl"
    started = time.time()
    source = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    start = int(config.frames["review_start_ui"]) - 1
    stop = int(config.frames["review_end_ui"])
    raw = np.asarray(source[start:stop], dtype=np.float32)
    labels = label_core.load_labels(config.labels_tsv)
    quiet_count = (
        int(config.frames["quiet_end_ui"])
        - int(config.frames["quiet_start_ui"])
        + 1
    )
    _progress(progress, "parzen_innovation_start")
    residual, innovation = _innovation_residual(
        raw, quiet_count, _coefficients(config), config
    )
    del raw
    center, scale, standardization = _quiet_standardization(
        residual, quiet_count, 10.0
    )
    centered = residual - center[None]
    standardized = centered / scale[None]
    _progress(progress, "spatial_patch_fit_start")
    settings = config.model
    patches = sample_spatial_patches(
        standardized,
        patch_size=int(settings["patch_size"]),
        sample_count=int(settings["sample_count"]),
        seed=int(settings["seed"]),
    )
    model = fit_spatial_patch_fastica(
        patches,
        rank=int(settings["rank"]),
        seed=int(settings["seed"]),
        max_iterations=int(settings["fastica_max_iterations"]),
        tolerance=float(settings["fastica_tolerance"]),
    )
    quiet_patches = sample_spatial_patches(
        standardized,
        patch_size=int(settings["patch_size"]),
        sample_count=min(int(settings["sample_count"]), 12000),
        seed=int(settings["seed"]) + 1,
        frame_indices=np.arange(quiet_count),
    )
    quiet_components = (
        quiet_patches - model.patch_mean[None]
    ) @ model.analysis_filters.T
    quiet_scale = np.maximum(
        np.std(quiet_components, axis=0, ddof=1), 1e-6
    ).astype(np.float32)
    model = replace(model, component_scale=quiet_scale)
    training_components = (
        patches - model.patch_mean[None]
    ) @ model.analysis_filters.T
    standardized_components = training_components / model.component_scale[None]
    parzen = fit_parzen_shrinkage(
        standardized_components,
        **config.parzen,
    )
    model_artifacts = _write_model_artifacts(partial / "model", model)
    _atomic_json(partial / "model" / "diagnostics.json", {
        **model.diagnostics(),
        "sample_count": len(patches),
        "quiet_component_scales": quiet_scale.tolist(),
        "parzen": parzen.diagnostics(),
        "fit_scope": "all review frames, labels excluded",
    })
    del quiet_patches, quiet_components, training_components
    roi_selected, roi_matrix = _roi_matrix(
        residual.shape[1:], labels, int(config.evaluation["roi_radius_px"])
    )
    synthetic_observed, synthetic_truth, synthetic_scale = _synthetic_fixture(
        centered, scale, config
    )
    synthetic_standardized = synthetic_observed / synthetic_scale[None]
    common_max = _display_scale(
        np.maximum(centered, 0),
        float(config.visualization["positive_upper_percentile"]),
        config,
    )

    def patch_lane(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        result = patch_lattice_reconstruction(
            values,
            model,
            stride=int(settings["patch_lattice_stride"]),
            shrinkage="wiener",
            lambda_z=float(settings["wiener_lambda_z"]),
        )
        return result, {
            "architecture": "overlapping_patch_lattice",
            "application_stride": int(settings["patch_lattice_stride"]),
            "shrinkage": "quiet_wiener",
        }

    def dense_lane(
        values: np.ndarray, shrinkage: str
    ) -> tuple[np.ndarray, dict[str, Any]]:
        result, diagnostics = dense_convolutional_reconstruction(
            values,
            model,
            shrinkage=shrinkage,
            lambda_z=float(settings["wiener_lambda_z"]),
            parzen=parzen if shrinkage == "parzen" else None,
            device=str(config.resources["device"]),
            frame_batch_size=int(config.resources["frame_batch_size"]),
        )
        diagnostics.update(
            architecture="dense_translation_shared_analysis_synthesis",
            shrinkage=shrinkage,
        )
        return result, diagnostics

    lanes: list[
        tuple[str, Callable[[np.ndarray], tuple[np.ndarray, dict[str, Any]]]]
    ] = [
        ("c1_patch_fastica", patch_lane),
        ("c2_dense_convolutional_fastica", lambda value: dense_lane(value, "wiener")),
        ("c3_dense_convolutional_parzen", lambda value: dense_lane(value, "parzen")),
    ]
    rows = []
    for index, (variant_id, function) in enumerate(lanes, start=1):
        _progress(
            progress,
            "lane_start",
            variant_id=variant_id,
            variant_index=index,
            variant_total=len(lanes),
        )
        lane_started = time.perf_counter()
        standardized_signal, diagnostics = function(standardized)
        runtime = time.perf_counter() - lane_started
        signal = standardized_signal * scale[None]
        synthetic_signal_z, _ = function(synthetic_standardized)
        synthetic_signal = synthetic_signal_z * synthetic_scale[None]
        row = _variant_metrics(
            variant_id,
            signal,
            residual,
            labels,
            roi_selected,
            roi_matrix,
            quiet_count,
            config,
            runtime,
            synthetic_signal,
            synthetic_truth,
        )
        parameters = {
            "model": settings,
            "parzen": config.parzen if variant_id.endswith("parzen") else None,
            **diagnostics,
        }
        row["parameters"] = parameters
        row["method_diagnostics"] = diagnostics
        row["videos"] = _write_tiffs(
            partial / "methods" / variant_id,
            signal,
            residual,
            common_max,
            variant_id,
            parameters,
            config,
        )
        rows.append(row)
        _atomic_json(partial / "checkpoint.json", {
            "phase": "spatial_ica_screen",
            "completed": index,
            "total": len(lanes),
            "last_variant": variant_id,
        })
        del standardized_signal, signal, synthetic_signal_z, synthetic_signal
        gc.collect()
    best = max(
        rows,
        key=lambda row: (
            row["fixed_budget_mean_recall"],
            row["mean_recall"],
            -row["event_candidates"],
        ),
    )
    conclusion = (
        f"The strongest exploratory lane by fixed-budget recall was "
        f"{best['variant_id']}. This does not authorize the noisy-Parzen Infomax "
        "optimization stage until timing, remainder leakage, and held-burst "
        "generalization are reviewed."
    )
    metrics = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": "completed",
        "variant_count": len(rows),
        "variants": rows,
        "best_exploratory_variant": best["variant_id"],
        "conclusion": conclusion,
        "elapsed_seconds": time.time() - started,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "input_lane": config.input_lane,
        "innovation_calibration": innovation,
        "quiet_standardization": standardization,
        "model_artifacts": model_artifacts,
        "scientific_status": "exploratory_transductive_architecture_screen",
    }
    _atomic_json(partial / "metrics.json", metrics)
    (partial / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    _atomic_json(partial / "run_state.json", {
        "status": "completed",
        "completed_unix": time.time(),
        "elapsed_seconds": metrics["elapsed_seconds"],
        "max_rss_mib": metrics["max_rss_mib"],
        "variant_count": len(rows),
        "tiff_count": 2 * len(rows) + 1,
    })
    partial.replace(config.output_dir)
    return metrics
