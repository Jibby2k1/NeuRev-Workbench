"""Successive-halving program for ten carrier-preserving denoising families."""
from __future__ import annotations

from dataclasses import replace
import csv
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any

import numpy as np

from neurobench.algorithms.advanced_denoising import (
    bounded_noise_subtraction,
    carrier_blend,
    dense_ica_denoise,
    fit_component_parzen_shrinkages,
    multiscale_group_shrinkage,
    noise_psd_wiener,
    nonlocal_means_spatial,
    undecimated_spatial_group_shrinkage,
    windowed_nonnegative_factorization,
    windowed_robust_low_rank_sparse,
)
from neurobench.algorithms.spatial_patch_ica import (
    fit_parzen_shrinkage,
    fit_spatial_patch_fastica,
    sample_spatial_patches,
)
from neurobench.experiments.learnable_contrast import core as label_core
from neurobench.metrics.synthetic_denoising import (
    localized_synthetic_denoising_metrics,
)
from neurobench.experiments.pairwise_separation.evaluation import event_intervals

from .denoise_audit import (
    _display_scale,
    _roi_matrix,
    _synthetic_fixture,
    _variant_metrics,
    _write_tiffs,
)
from .denoising_program_config import (
    DenoisingProgramConfig,
    FAMILY_IDS,
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


def _variant_id(row: dict[str, Any]) -> str:
    return f"{row['family_id']}__{int(row['variant_index']):02d}"


def _atomic_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def preflight(
    config: DenoisingProgramConfig, *, write_artifacts: bool = True
) -> dict[str, Any]:
    inputs = (config.source_video, config.labels_tsv, config.architecture_manifest)
    missing = [str(path) for path in inputs if not path.is_file()]
    shape = None
    dtype = None
    bounds = finite = labels_valid = fit_valid = False
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
    frames = (
        int(config.frames["review_end_ui"])
        - int(config.frames["review_start_ui"])
        + 1
    )
    pixels = 0 if shape is None else int(shape[1]) * int(shape[2])
    dense_mib = frames * pixels * 4 / 2**20
    largest_patch = max(int(value) for value in config.shared_ica["patch_sizes"])
    batch = int(config.resources["frame_batch_size"])
    gpu_workspace = (
        batch * pixels * (largest_patch**2 + 2 * int(config.shared_ica["rank"]))
        * 4 / 2**20
    )
    estimated_ram = 8.5 * dense_mib + 1536
    estimated_gpu = 3.0 * gpu_workspace + 768
    finalist_tiff_mib = (
        config.finalist_count * 2 * frames * pixels * 2 / 2**20
    )
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
        "output_cap_sufficient": finalist_tiff_mib <= int(config.resources["max_output_mib"]),
        "requested_device_available": (not cuda_requested) or gpu["available"],
        "gpu_memory_cap_sufficient": (
            (not cuda_requested)
            or estimated_gpu <= int(config.resources["max_gpu_memory_mib"])
        ),
        "live_gpu_memory_sufficient": (
            (not cuda_requested) or estimated_gpu <= gpu.get("free_mib", 0)
        ),
    }
    design_counts = {
        family: len(rows) for family, rows in config.designs().items()
    }
    payload = {
        "schema_version": 1,
        "kind": "read_only_advanced_denoising_program_preflight",
        "experiment_id": config.experiment_id,
        "ready": all(gates.values()),
        "gates": gates,
        "source_shape": shape,
        "source_dtype": dtype,
        "label_rows": len(labels),
        "roi_identities": len({row["roi_identity"] for row in labels}),
        "design": {
            "family_count": len(FAMILY_IDS),
            "family_ids": list(FAMILY_IDS),
            "breadth_combination_count": config.breadth_combination_count,
            "breadth_counts_by_family": design_counts,
            "full_field_semifinal_count": config.full_field_combination_count,
            "finalist_count": config.finalist_count,
            "confirmation_record_count": config.confirmation_evaluation_count,
            "total_declared_evaluation_records": (
                config.breadth_combination_count
                + config.full_field_combination_count
                + config.confirmation_evaluation_count
            ),
        },
        "resources": {
            "estimated_peak_ram_mib": estimated_ram,
            "available_ram_mib": _available_ram_mib(),
            "estimated_peak_gpu_memory_mib": estimated_gpu,
            "uncompressed_finalist_tiff_mib": finalist_tiff_mib,
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
            "Stage A tunes all ten families on the complete event timeline inside "
            "a label-enclosing crop plus exact semi-synthetic truth. Stage B uses "
            "full-field detection. Sparse unlabeled candidates remain unknown. "
            "Only family finalists receive TIFFs."
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
        raise RuntimeError(f"advanced denoising preflight failed: {payload}")
    return payload


def _matching_preflight(config: DenoisingProgramConfig) -> dict[str, Any]:
    audit = json.loads((config.preflight_dir / "preflight.json").read_text(encoding="utf-8"))
    resolved = json.loads(
        (config.preflight_dir / "config.resolved.json").read_text(encoding="utf-8")
    )
    if not audit.get("ready") or resolved != config.to_dict():
        raise RuntimeError("run requires a matching ready preflight")
    if config.output_dir.exists() or Path(str(config.output_dir) + ".partial").exists():
        raise FileExistsError("completed or partial output already exists")
    return audit


def _fit_ica_context(
    standardized: np.ndarray,
    quiet_count: int,
    config: DenoisingProgramConfig,
    *,
    seed: int | None = None,
    frame_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    settings = config.shared_ica
    resolved_seed = int(settings["seed"] if seed is None else seed)
    models: dict[int, Any] = {}
    training_components: dict[int, np.ndarray] = {}
    for patch in settings["patch_sizes"]:
        patch = int(patch)
        patches = sample_spatial_patches(
            standardized,
            patch_size=patch,
            sample_count=int(settings["sample_count"]),
            seed=resolved_seed + patch,
            frame_indices=frame_indices,
        )
        model = fit_spatial_patch_fastica(
            patches,
            rank=min(int(settings["rank"]), patch * patch),
            seed=resolved_seed + 10 * patch,
            max_iterations=int(settings["fastica_max_iterations"]),
            tolerance=float(settings["fastica_tolerance"]),
        )
        quiet_patches = sample_spatial_patches(
            standardized,
            patch_size=patch,
            sample_count=min(int(settings["sample_count"]), 8000),
            seed=resolved_seed + 100 * patch,
            frame_indices=np.arange(quiet_count),
        )
        quiet_components = (
            quiet_patches - model.patch_mean[None]
        ) @ model.analysis_filters.T
        component_scale = np.maximum(
            np.std(quiet_components, axis=0, ddof=1), 1e-6
        ).astype(np.float32)
        model = replace(model, component_scale=component_scale)
        models[patch] = model
        training_components[patch] = (
            (patches - model.patch_mean[None]) @ model.analysis_filters.T
        ) / model.component_scale[None]
    base_patch = 11
    parzen_settings = dict(settings["parzen"])
    shared_parzen = fit_parzen_shrinkage(
        training_components[base_patch].ravel(), **parzen_settings
    )
    return {
        "models": models,
        "training_components": training_components,
        "shared_parzen": shared_parzen,
        "component_parzen": {},
        "cache": {},
        "seed": resolved_seed,
    }


def _component_posteriors(
    context: dict[str, Any],
    row: dict[str, Any],
    config: DenoisingProgramConfig,
):
    key = (
        float(row["zero_fraction"]),
        float(row["bandwidth"]),
        float(row["noise_variance"]),
    )
    if key not in context["component_parzen"]:
        settings = dict(config.shared_ica["parzen"])
        settings.update(
            zero_fraction=key[0], bandwidth=key[1], noise_variance=key[2]
        )
        context["component_parzen"][key] = fit_component_parzen_shrinkages(
            context["training_components"][11], **settings
        )
    return context["component_parzen"][key]


def _cached_dense(
    values: np.ndarray,
    scope: str,
    context: dict[str, Any],
    config: DenoisingProgramConfig,
    *,
    patch: int,
    mode: str,
    cache_suffix: tuple[Any, ...] = (),
    component_parzen=None,
    kalman_half_life_frames: float | None = None,
    kalman_process_variance: float = 0.08,
) -> np.ndarray:
    key = (scope, patch, mode, *cache_suffix)
    if key not in context["cache"]:
        result, _ = dense_ica_denoise(
            values,
            context["models"][patch],
            mode=mode,
            shared_parzen=(
                context["shared_parzen"] if mode == "shared_parzen" else None
            ),
            component_parzen=component_parzen,
            wiener_lambda_z=float(config.shared_ica["wiener_lambda_z"]),
            device=str(config.resources["device"]),
            frame_batch_size=int(config.resources["frame_batch_size"]),
            kalman_half_life_frames=kalman_half_life_frames,
            kalman_process_variance=kalman_process_variance,
        )
        context["cache"][key] = result
    return context["cache"][key]


def _apply_variant(
    values: np.ndarray,
    row: dict[str, Any],
    context: dict[str, Any],
    config: DenoisingProgramConfig,
    *,
    quiet_count: int,
    scope: str,
    seed: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    family = row["family_id"]
    device = str(config.resources["device"])
    if family == "skip_connected_parzen_ica":
        estimate = _cached_dense(
            values, scope, context, config, patch=11, mode="shared_parzen"
        )
        output = carrier_blend(values, estimate, float(row["alpha"]))
    elif family == "per_component_parzen_ica":
        posteriors = _component_posteriors(context, row, config)
        suffix = (
            float(row["zero_fraction"]),
            float(row["bandwidth"]),
            float(row["noise_variance"]),
        )
        estimate = _cached_dense(
            values,
            scope,
            context,
            config,
            patch=11,
            mode="component_parzen",
            cache_suffix=suffix,
            component_parzen=posteriors,
        )
        output = carrier_blend(values, estimate, float(row["alpha"]))
    elif family == "multiscale_convolutional_ica":
        estimates = [
            _cached_dense(
                values, scope, context, config, patch=int(patch), mode="wiener"
            )
            for patch in row["patch_sizes"]
        ]
        output = multiscale_group_shrinkage(
            values,
            estimates,
            lambda_z=float(row["lambda_z"]),
            alpha=float(row["alpha"]),
        )
    elif family == "bounded_ica_noise_subtraction":
        estimate = _cached_dense(
            values, scope, context, config, patch=11, mode="shared_parzen"
        )
        output = bounded_noise_subtraction(
            values,
            estimate,
            alpha=float(row["alpha"]),
            correction_limit_z=float(row["correction_limit_z"]),
        )
    elif family == "noise_psd_wiener":
        output = noise_psd_wiener(
            values,
            quiet_count=quiet_count,
            noise_multiplier=float(row["noise_multiplier"]),
            frequency_smoothing_sigma=float(row["frequency_smoothing_sigma"]),
        )
    elif family == "robust_low_rank_sparse":
        output = windowed_robust_low_rank_sparse(
            values,
            window_frames=int(row["window_frames"]),
            rank=int(row["rank"]),
            sparse_lambda_z=float(row["sparse_lambda_z"]),
            alpha=float(row["alpha"]),
            device=device,
        )
    elif family == "nonnegative_factorization":
        output = windowed_nonnegative_factorization(
            values,
            window_frames=int(row["window_frames"]),
            rank=int(row["rank"]),
            iterations=int(row["iterations"]),
            alpha=float(row["alpha"]),
            seed=int(context["seed"] if seed is None else seed),
            device=device,
        )
    elif family == "nonlocal_patch_denoising":
        output = nonlocal_means_spatial(
            values,
            search_radius=int(row["search_radius"]),
            patch_size=int(row["patch_size"]),
            bandwidth_z=float(row["bandwidth_z"]),
            alpha=float(row["alpha"]),
            device=device,
            frame_batch_size=int(config.resources["frame_batch_size"]),
        )
    elif family == "component_kalman":
        half_life = float(row["kalman_half_life_frames"])
        process = float(row["kalman_process_variance"])
        estimate = _cached_dense(
            values,
            scope,
            context,
            config,
            patch=11,
            mode="kalman",
            cache_suffix=(half_life, process),
            kalman_half_life_frames=half_life,
            kalman_process_variance=process,
        )
        output = carrier_blend(values, estimate, 0.5)
    elif family == "undecimated_wavelet":
        output = undecimated_spatial_group_shrinkage(
            values,
            levels=int(row["levels"]),
            threshold_z=float(row["threshold_z"]),
            group_sigma_px=float(row["group_sigma_px"]),
            coarse_keep=float(row["coarse_keep"]),
        )
    else:
        raise ValueError(f"unknown denoising family: {family}")
    if output.shape != values.shape or not np.isfinite(output).all():
        raise RuntimeError(f"invalid denoising output for {_variant_id(row)}")
    return output.astype(np.float32), {
        "family_id": family,
        "parameters": {
            key: value
            for key, value in row.items()
            if key not in {"family_id", "variant_index"}
        },
    }


def _crop_contract(
    labels: list[dict[str, Any]],
    shape: tuple[int, int],
    margin: int,
) -> tuple[tuple[slice, slice], list[dict[str, Any]], dict[str, int]]:
    xs = [float(row["x_px"]) for row in labels]
    ys = [float(row["y_px"]) for row in labels]
    x0 = max(0, int(np.floor(min(xs))) - margin)
    x1 = min(shape[1], int(np.ceil(max(xs))) + margin + 1)
    y0 = max(0, int(np.floor(min(ys))) - margin)
    y1 = min(shape[0], int(np.ceil(max(ys))) + margin + 1)
    shifted = [
        {**row, "x_px": float(row["x_px"]) - x0, "y_px": float(row["y_px"]) - y0}
        for row in labels
    ]
    return (slice(y0, y1), slice(x0, x1)), shifted, {
        "x0": x0, "x1": x1, "y0": y0, "y1": y1,
        "height": y1 - y0, "width": x1 - x0,
    }


def _selection_score(metrics: dict[str, Any]) -> float:
    candidate_penalty = 0.002 * float(metrics["event_candidates"])
    timing_penalty = 0.05 * float(metrics["median_peak_frame_error"])
    return float(
        metrics["fixed_budget_mean_recall"]
        + 0.5 * metrics["mean_recall"]
        + 0.2 * np.clip(metrics["median_peak_retention"], 0, 1)
        + 0.2 * np.clip(metrics["median_area_retention"], 0, 1)
        + 0.2 * np.clip(metrics["median_waveform_correlation"], 0, 1)
        + 0.2 * np.clip(metrics["synthetic_correlation"], -1, 1)
        - candidate_penalty
        - timing_penalty
    )


def _depth_gate(metrics: dict[str, Any], config: DenoisingProgramConfig) -> bool:
    evaluation = config.evaluation
    return bool(
        metrics["median_peak_retention"]
        >= float(evaluation["minimum_peak_retention"])
        and metrics["median_area_retention"]
        >= float(evaluation["minimum_area_retention"])
        and metrics["median_peak_frame_error"]
        <= float(evaluation["maximum_peak_frame_error"])
        and metrics["synthetic_correlation"]
        >= float(evaluation["minimum_synthetic_correlation"])
    )


def _evaluate(
    row: dict[str, Any],
    standardized: np.ndarray,
    residual: np.ndarray,
    scale: np.ndarray,
    labels: list[dict[str, Any]],
    quiet_count: int,
    synthetic_z: np.ndarray,
    synthetic_scale: np.ndarray,
    synthetic_truth: np.ndarray,
    context: dict[str, Any],
    config: DenoisingProgramConfig,
    *,
    scope: str,
) -> tuple[dict[str, Any], np.ndarray]:
    started = time.perf_counter()
    estimate_z, diagnostics = _apply_variant(
        standardized,
        row,
        context,
        config,
        quiet_count=quiet_count,
        scope=scope,
    )
    runtime = time.perf_counter() - started
    signal = estimate_z * scale[None]
    synthetic_estimate_z, _ = _apply_variant(
        synthetic_z,
        row,
        context,
        config,
        quiet_count=min(quiet_count, max(8, len(synthetic_z) // 4)),
        scope=scope + "_synthetic",
    )
    synthetic_signal = synthetic_estimate_z * synthetic_scale[None]
    roi_selected, roi_matrix = _roi_matrix(
        residual.shape[1:], labels, int(config.evaluation["roi_radius_px"])
    )
    metrics = _variant_metrics(
        _variant_id(row),
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
    metrics.update(
        localized_synthetic_denoising_metrics(
            synthetic_signal,
            synthetic_z * synthetic_scale[None],
            synthetic_truth,
        )
    )
    metrics.update(
        family_id=row["family_id"],
        variant_index=int(row["variant_index"]),
        parameters=diagnostics["parameters"],
    )
    metrics["selection_score"] = _selection_score(metrics)
    metrics["depth_gate_pass"] = _depth_gate(metrics, config)
    return metrics, signal


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['experiment_id']}",
        "",
        "## Outcome",
        "",
        payload["conclusion"],
        "",
        "## Family finalists",
        "",
        "| Family | Variant | Recall | Fixed recall | Candidates | Peak | Area | Synthetic r | Gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["finalists"]:
        lines.append(
            f"| `{row['family_id']}` | `{row['variant_id']}` | "
            f"{row['mean_recall']:.3f} | {row['fixed_budget_mean_recall']:.3f} | "
            f"{row['event_candidates']} | {row['median_peak_retention']:.3f} | "
            f"{row['median_area_retention']:.3f} | "
            f"{row['synthetic_correlation']:.3f} | "
            f"{'pass' if row['depth_gate_pass'] else 'flag'} |"
        )
    lines.extend([
        "",
        "Sparse unmatched candidates are unknown, not false positives. Stage A "
        "used a label-enclosing crop; finalist numbers shown here are full-field. "
        "Visual signal/remainder review and confirmation records remain required.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config: DenoisingProgramConfig) -> dict[str, Any]:
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(config.resources["cpu_threads"])
    audit = _matching_preflight(config)
    partial = Path(str(config.output_dir) + ".partial")
    partial.mkdir(parents=True)
    for directory in ("stage_a", "stage_b", "confirmation", "finalists"):
        (partial / directory).mkdir()
    _atomic_json(partial / "config.resolved.json", config.to_dict())
    _atomic_json(partial / "preflight.json", audit)
    progress = partial / "progress.jsonl"
    started = time.time()
    source = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    frame_start = int(config.frames["review_start_ui"]) - 1
    frame_stop = int(config.frames["review_end_ui"])
    raw = np.asarray(source[frame_start:frame_stop], dtype=np.float32)
    labels = label_core.load_labels(config.labels_tsv)
    quiet_count = (
        int(config.frames["quiet_end_ui"])
        - int(config.frames["quiet_start_ui"])
        + 1
    )
    _progress(progress, "input_parzen_innovation_start")
    residual, innovation = _innovation_residual(
        raw, quiet_count, _coefficients(config), config
    )
    del raw
    center, scale, standardization = _quiet_standardization(
        residual, quiet_count, 10.0
    )
    centered = residual - center[None]
    standardized = centered / scale[None]
    crop_slices, crop_labels, crop = _crop_contract(
        labels,
        residual.shape[1:],
        int(config.stages["stage_a_crop_margin_px"]),
    )
    y_slice, x_slice = crop_slices
    crop_residual = residual[:, y_slice, x_slice]
    crop_scale = scale[y_slice, x_slice]
    crop_standardized = standardized[:, y_slice, x_slice]
    synthetic_observed, synthetic_truth, synthetic_scale = _synthetic_fixture(
        centered, scale, config
    )
    synthetic_z = synthetic_observed / synthetic_scale[None]
    _atomic_json(partial / "common_contract.json", {
        "input_lane": config.input_lane,
        "innovation_calibration": innovation,
        "quiet_standardization": standardization,
        "stage_a_crop": crop,
        "breadth_combination_count": config.breadth_combination_count,
        "full_field_semifinal_count": config.full_field_combination_count,
        "finalist_count": config.finalist_count,
        "confirmation_record_count": config.confirmation_evaluation_count,
    })
    _progress(progress, "shared_ica_fit_start")
    context = _fit_ica_context(standardized, quiet_count, config)

    stage_a_rows: list[dict[str, Any]] = []
    designs = config.designs()
    total_a = config.breadth_combination_count
    counter = 0
    for family in FAMILY_IDS:
        for row in designs[family]:
            counter += 1
            _progress(
                progress,
                "stage_a_start",
                variant_id=_variant_id(row),
                combination_index=counter,
                combination_total=total_a,
            )
            metrics, signal = _evaluate(
                row,
                crop_standardized,
                crop_residual,
                crop_scale,
                crop_labels,
                quiet_count,
                synthetic_z,
                synthetic_scale,
                synthetic_truth,
                context,
                config,
                scope="stage_a_crop",
            )
            stage_a_rows.append(metrics)
            _atomic_json(
                partial / "stage_a" / f"{_variant_id(row)}.json", metrics
            )
            del signal
            gc.collect()
    _atomic_json(partial / "stage_a" / "metrics.json", {
        "combination_count": len(stage_a_rows), "rows": stage_a_rows
    })
    selected_a: list[dict[str, Any]] = []
    for family in FAMILY_IDS:
        ranked = sorted(
            (row for row in stage_a_rows if row["family_id"] == family),
            key=lambda row: (
                row["depth_gate_pass"],
                row["selection_score"],
            ),
            reverse=True,
        )
        selected_a.extend(ranked[: int(config.stages["stage_a_top_per_family"])])
    _atomic_json(partial / "stage_a" / "selection.json", {
        "selected_count": len(selected_a),
        "selected_variant_ids": [row["variant_id"] for row in selected_a],
    })

    design_lookup = {
        _variant_id(row): row
        for rows in designs.values()
        for row in rows
    }
    context["cache"].clear()
    gc.collect()
    stage_b_rows: list[dict[str, Any]] = []
    for index, selected in enumerate(selected_a, start=1):
        row = design_lookup[selected["variant_id"]]
        _progress(
            progress,
            "stage_b_start",
            variant_id=_variant_id(row),
            combination_index=index,
            combination_total=len(selected_a),
        )
        metrics, signal = _evaluate(
            row,
            standardized,
            residual,
            scale,
            labels,
            quiet_count,
            synthetic_z,
            synthetic_scale,
            synthetic_truth,
            context,
            config,
            scope="stage_b_full",
        )
        stage_b_rows.append(metrics)
        _atomic_json(
            partial / "stage_b" / f"{_variant_id(row)}.json", metrics
        )
        del signal
        gc.collect()
    _atomic_json(partial / "stage_b" / "metrics.json", {
        "combination_count": len(stage_b_rows), "rows": stage_b_rows
    })
    finalists: list[dict[str, Any]] = []
    for family in FAMILY_IDS:
        ranked = sorted(
            (row for row in stage_b_rows if row["family_id"] == family),
            key=lambda row: (
                row["depth_gate_pass"],
                row["selection_score"],
            ),
            reverse=True,
        )
        finalists.extend(ranked[: int(config.stages["stage_b_top_per_family"])])
    _atomic_json(partial / "stage_b" / "selection.json", {
        "selected_count": len(finalists),
        "selected_variant_ids": [row["variant_id"] for row in finalists],
    })

    common_max = _display_scale(
        np.maximum(centered, 0),
        float(config.visualization["positive_upper_percentile"]),
        config,
    )
    context["cache"].clear()
    gc.collect()
    finalist_payload = []
    for index, selected in enumerate(finalists, start=1):
        row = design_lookup[selected["variant_id"]]
        _progress(
            progress,
            "finalist_tiff_start",
            variant_id=_variant_id(row),
            finalist_index=index,
            finalist_total=len(finalists),
        )
        estimate_z, diagnostics = _apply_variant(
            standardized,
            row,
            context,
            config,
            quiet_count=quiet_count,
            scope=f"finalist_{row['family_id']}",
        )
        signal = estimate_z * scale[None]
        videos = _write_tiffs(
            partial / "finalists" / row["family_id"],
            signal,
            residual,
            common_max,
            _variant_id(row),
            diagnostics["parameters"],
            config,
        )
        selected = {**selected, "videos": videos}
        finalist_payload.append(selected)
        del estimate_z, signal
        context["cache"].clear()
        gc.collect()

    confirmation_rows = []
    intervals = event_intervals(labels, int(config.frames["review_start_ui"]))
    for finalist in finalist_payload:
        for seed in config.stages["confirmation_seeds"]:
            for burst in config.stages["confirmation_holdout_bursts"]:
                fold = next(
                    row
                    for row in finalist["detection_folds"]
                    if int(row["burst_id"]) == int(burst)
                )
                confirmation_rows.append({
                    "variant_id": finalist["variant_id"],
                    "family_id": finalist["family_id"],
                    "seed": int(seed),
                    "holdout_burst": int(burst),
                    "seed_mode": (
                        "fit_seed_pending_refit"
                        if finalist["family_id"] in {
                            "skip_connected_parzen_ica",
                            "per_component_parzen_ica",
                            "multiscale_convolutional_ica",
                            "bounded_ica_noise_subtraction",
                            "nonnegative_factorization",
                            "component_kalman",
                        }
                        else "deterministic_replicate"
                    ),
                    "event_start": int(intervals[int(burst)][0]),
                    "event_stop": int(intervals[int(burst)][1]),
                    **fold,
                })
    _atomic_json(partial / "confirmation" / "records.json", {
        "status": "screening_records_written_confirmation_refits_not_executed",
        "record_count": len(confirmation_rows),
        "rows": confirmation_rows,
    })
    _atomic_tsv(partial / "confirmation" / "records.tsv", confirmation_rows)

    passed = [row for row in finalist_payload if row["depth_gate_pass"]]
    best = max(
        finalist_payload,
        key=lambda row: (
            row["depth_gate_pass"],
            row["fixed_budget_mean_recall"],
            row["mean_recall"],
            -row["event_candidates"],
        ),
    )
    conclusion = (
        f"{len(passed)} of {len(finalist_payload)} family finalists passed the "
        f"frozen preservation gate. The strongest full-field finalist by the "
        f"declared ordering was {best['variant_id']}. Seed-by-held-burst records "
        "are written, but stochastic refits remain an explicit final checkpoint."
    )
    metrics = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": "completed_screening_confirmation_refits_not_executed",
        "stage_a_combination_count": len(stage_a_rows),
        "stage_b_combination_count": len(stage_b_rows),
        "finalist_count": len(finalist_payload),
        "confirmation_record_count": len(confirmation_rows),
        "finalists": finalist_payload,
        "passed_finalist_ids": [row["variant_id"] for row in passed],
        "best_full_field_finalist": best["variant_id"],
        "conclusion": conclusion,
        "elapsed_seconds": time.time() - started,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "scientific_status": "screening_complete_confirmation_refits_not_executed",
    }
    _atomic_json(partial / "metrics.json", metrics)
    flat = [
        {
            key: value
            for key, value in row.items()
            if not isinstance(value, (dict, list))
        }
        for row in finalist_payload
    ]
    _atomic_tsv(partial / "finalist_comparison.tsv", flat)
    _write_report(partial / "REPORT.md", metrics)
    _atomic_json(partial / "run_state.json", {
        "status": metrics["status"],
        "completed_unix": time.time(),
        "elapsed_seconds": metrics["elapsed_seconds"],
        "max_rss_mib": metrics["max_rss_mib"],
        "stage_a_combination_count": len(stage_a_rows),
        "stage_b_combination_count": len(stage_b_rows),
        "finalist_count": len(finalist_payload),
        "tiff_count": 2 * len(finalist_payload),
    })
    partial.replace(config.output_dir)
    return metrics
