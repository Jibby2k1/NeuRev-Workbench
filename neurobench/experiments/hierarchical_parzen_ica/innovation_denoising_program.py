"""Staged v3 program for bounded innovation denoisers and Pareto mixtures."""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any, Sequence

import numpy as np

from neurobench.algorithms.advanced_denoising import (
    carrier_blend,
    dense_ica_denoise,
)
from neurobench.algorithms.innovative_denoising import (
    apply_blindspot_linear_model,
    bounded_mixture,
    cross_scale_consensus_shrinkage,
    fit_blindspot_linear_model,
    graph_edge_aware_diffusion,
    local_noise_psd_wiener,
    morphology_conditioned_shrinkage,
    selected_component_nmf,
    select_diverse_pareto_rows,
    tempered_residual_posterior,
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

from .advanced_denoising_program import (
    _atomic_tsv,
    _crop_contract,
)
from .denoise_audit import (
    _display_scale,
    _roi_matrix,
    _synthetic_fixture,
    _variant_metrics,
    _write_tiffs,
)
from .innovation_denoising_config import FAMILY_IDS, InnovationDenoisingConfig
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


def _resource_checkpoint(
    config: InnovationDenoisingConfig, progress: Path, stage: str
) -> float:
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    _progress(progress, "resource_checkpoint", checkpoint=stage, max_rss_mib=rss_mib)
    if rss_mib > float(config.resources["max_ram_mib"]):
        raise MemoryError(
            f"{stage} exceeded RAM cap: {rss_mib:.1f} MiB > "
            f"{config.resources['max_ram_mib']} MiB"
        )
    return rss_mib


def _parameters(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"family_id", "variant_index"}
    }


def preflight(
    config: InnovationDenoisingConfig, *, write_artifacts: bool = True
) -> dict[str, Any]:
    """Validate inputs, frozen design, collision safety, and resource headroom."""
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
        finite = bounds and bool(
            np.isfinite(video[start:stop:20, ::16, ::16]).all()
        )
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
        architecture = json.loads(
            config.architecture_manifest.read_text(encoding="utf-8")
        )
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
    patch = int(config.shared_ica["patch_size"])
    batch = int(config.resources["frame_batch_size"])
    gpu_workspace = (
        batch
        * pixels
        * (patch**2 + 2 * int(config.shared_ica["rank"]))
        * 4
        / 2**20
    )
    estimated_ram = 26.5 * dense_mib + 1024
    estimated_gpu = 3.0 * gpu_workspace + 768
    tiff_mib = (
        config.tiff_finalist_count * 2 * frames * pixels * 2 / 2**20
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
        "partial_output_absent": not Path(
            str(config.output_dir) + ".partial"
        ).exists(),
        "preflight_separate_from_output": config.preflight_dir != config.output_dir,
        "ram_cap_sufficient": estimated_ram
        <= int(config.resources["max_ram_mib"]),
        "available_ram_sufficient": estimated_ram <= _available_ram_mib(),
        "disk_headroom_sufficient": free_disk
        >= int(config.resources["min_free_disk_mib"]),
        "output_cap_sufficient": tiff_mib
        <= int(config.resources["max_output_mib"]),
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
        "kind": "read_only_innovation_denoising_v3_preflight",
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
            "breadth_counts_by_family": {
                family: len(rows)
                for family, rows in config.designs().items()
            },
            "full_field_semifinal_count": config.full_field_combination_count,
            "family_finalist_count": config.family_finalist_count,
            "pareto_mixture_count": config.mixture_combination_count,
            "maximum_confirmation_refit_count": (
                config.maximum_confirmation_refit_count
            ),
            "tiff_finalist_count": config.tiff_finalist_count,
            "total_screen_evaluations": (
                config.breadth_combination_count
                + config.full_field_combination_count
                + config.mixture_combination_count
            ),
        },
        "resources": {
            "estimated_peak_ram_mib": estimated_ram,
            "available_ram_mib": _available_ram_mib(),
            "estimated_peak_gpu_memory_mib": estimated_gpu,
            "uncompressed_finalist_tiff_mib": tiff_mib,
            "free_disk_mib": free_disk,
            "gpu": gpu,
            **config.resources,
        },
        "inputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in inputs
            if path.is_file()
        ],
        "system_snapshot": _snapshots(),
        "scientific_contract": (
            "The accepted Parzen Innovation residual is the immutable carrier. "
            "All eight family lanes and all mixtures are bounded corrections. "
            "Sparse unmatched real candidates remain unknown, so candidate "
            "burden is a precision proxy rather than measured false positives. "
            "Exact synthetic truth evaluates four morphology cases."
        ),
    }
    if write_artifacts:
        config.preflight_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(config.preflight_dir / "preflight.json", payload)
        _atomic_json(
            config.preflight_dir / "config.resolved.json", config.to_dict()
        )
        if not missing and bounds:
            label_core._write_overlay(
                np.load(config.source_video, mmap_mode="r", allow_pickle=False),
                labels,
                config.preflight_dir / "label_projection_overlay.png",
            )
    if not payload["ready"]:
        raise RuntimeError(f"innovation denoising v3 preflight failed: {payload}")
    return payload


def _matching_preflight(config: InnovationDenoisingConfig) -> dict[str, Any]:
    audit = json.loads(
        (config.preflight_dir / "preflight.json").read_text(encoding="utf-8")
    )
    resolved = json.loads(
        (config.preflight_dir / "config.resolved.json").read_text(
            encoding="utf-8"
        )
    )
    if not audit.get("ready") or resolved != config.to_dict():
        raise RuntimeError("run requires a matching ready preflight")
    if config.output_dir.exists() or Path(
        str(config.output_dir) + ".partial"
    ).exists():
        raise FileExistsError("completed or partial output already exists")
    return audit


def _fit_context(
    standardized: np.ndarray,
    quiet_count: int,
    config: InnovationDenoisingConfig,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    settings = config.shared_ica
    resolved_seed = int(settings["seed"] if seed is None else seed)
    patch = int(settings["patch_size"])
    samples = sample_spatial_patches(
        standardized,
        patch_size=patch,
        sample_count=int(settings["sample_count"]),
        seed=resolved_seed + patch,
    )
    model = fit_spatial_patch_fastica(
        samples,
        rank=min(int(settings["rank"]), patch * patch),
        seed=resolved_seed + 10 * patch,
        max_iterations=int(settings["fastica_max_iterations"]),
        tolerance=float(settings["fastica_tolerance"]),
    )
    quiet_samples = sample_spatial_patches(
        standardized,
        patch_size=patch,
        sample_count=min(int(settings["sample_count"]), 8000),
        seed=resolved_seed + 100 * patch,
        frame_indices=np.arange(quiet_count),
    )
    quiet_components = (
        quiet_samples - model.patch_mean[None]
    ) @ model.analysis_filters.T
    component_scale = np.maximum(
        np.std(quiet_components, axis=0, ddof=1), 1e-6
    ).astype(np.float32)
    model = replace(model, component_scale=component_scale)
    training_components = (
        (samples - model.patch_mean[None]) @ model.analysis_filters.T
    ) / model.component_scale[None]
    parzen = fit_parzen_shrinkage(
        training_components.ravel(), **dict(settings["parzen"])
    )
    return {
        "model": model,
        "shared_parzen": parzen,
        "cache": {},
        "blindspot_models": {},
        "blindspot_training_video": standardized,
        "quiet_count": quiet_count,
        "seed": resolved_seed,
    }


def _cached_dense(
    values: np.ndarray,
    scope: str,
    context: dict[str, Any],
    config: InnovationDenoisingConfig,
    *,
    mode: str,
    cache_suffix: tuple[Any, ...] = (),
    **settings: Any,
) -> np.ndarray:
    key = (scope, mode, *cache_suffix)
    if key not in context["cache"]:
        result, _ = dense_ica_denoise(
            values,
            context["model"],
            mode=mode,
            shared_parzen=(
                context["shared_parzen"] if mode == "shared_parzen" else None
            ),
            wiener_lambda_z=float(config.shared_ica["wiener_lambda_z"]),
            device=str(config.resources["device"]),
            frame_batch_size=int(config.resources["frame_batch_size"]),
            **settings,
        )
        context["cache"][key] = result
    return context["cache"][key]


def _blindspot_model(
    row: dict[str, Any],
    context: dict[str, Any],
) -> Any:
    key = (
        int(row["radius"]),
        int(row["sample_count"]),
        float(row["ridge"]),
        str(row["fit_scope"]),
        int(context["seed"]),
    )
    if key not in context["blindspot_models"]:
        fit_count = (
            int(context["quiet_count"])
            if row["fit_scope"] == "quiet"
            else None
        )
        context["blindspot_models"][key] = fit_blindspot_linear_model(
            context["blindspot_training_video"],
            radius=int(row["radius"]),
            sample_count=int(row["sample_count"]),
            ridge=float(row["ridge"]),
            seed=int(context["seed"]) + 1000 * int(row["variant_index"]),
            fit_frame_count=fit_count,
        )
    return context["blindspot_models"][key]


def _apply_variant(
    values: np.ndarray,
    row: dict[str, Any],
    context: dict[str, Any],
    config: InnovationDenoisingConfig,
    *,
    quiet_count: int,
    scope: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    family = row["family_id"]
    device = str(config.resources["device"])
    diagnostics: dict[str, Any] = {}
    if family == "local_psd_wiener":
        output = local_noise_psd_wiener(
            values,
            quiet_count=quiet_count,
            tile_size=int(row["tile_size"]),
            overlap_fraction=float(row["overlap_fraction"]),
            noise_multiplier=float(row["noise_multiplier"]),
            frequency_smoothing_sigma=float(row["frequency_smoothing_sigma"]),
            transfer_floor=float(row["transfer_floor"]),
            alpha=float(row["alpha"]),
        )
    elif family == "morphology_conditioned":
        output = morphology_conditioned_shrinkage(
            values,
            center_sigma_px=float(row["center_sigma_px"]),
            ring_sigma_px=float(row["ring_sigma_px"]),
            crowd_sigma_px=float(row["crowd_sigma_px"]),
            isolated_threshold_z=float(row["isolated_threshold_z"]),
            crowded_threshold_z=float(row["crowded_threshold_z"]),
            gate_temperature_z=float(row["gate_temperature_z"]),
            gain_floor=float(row["gain_floor"]),
            alpha=float(row["alpha"]),
        )
    elif family == "selected_nmf":
        output, diagnostics = selected_component_nmf(
            values,
            window_frames=int(row["window_frames"]),
            rank=int(row["rank"]),
            iterations=int(row["iterations"]),
            minimum_spatial_concentration=float(
                row["minimum_spatial_concentration"]
            ),
            minimum_temporal_dynamics=float(row["minimum_temporal_dynamics"]),
            selection_temperature=float(row["selection_temperature"]),
            alpha=float(row["alpha"]),
            seed=int(context["seed"]) + int(row["variant_index"]),
            device=device,
        )
    elif family == "asymmetric_component_dynamics":
        estimate = _cached_dense(
            values,
            scope,
            context,
            config,
            mode="asymmetric",
            cache_suffix=(
                float(row["rise_gain"]),
                float(row["decay_gain"]),
                float(row["innovation_threshold_z"]),
                float(row["innovation_temperature_z"]),
            ),
            asymmetric_rise_gain=float(row["rise_gain"]),
            asymmetric_decay_gain=float(row["decay_gain"]),
            asymmetric_innovation_threshold_z=float(
                row["innovation_threshold_z"]
            ),
            asymmetric_innovation_temperature_z=float(
                row["innovation_temperature_z"]
            ),
        )
        output = carrier_blend(values, estimate, float(row["alpha"]))
    elif family == "tempered_parzen_posterior":
        posterior = _cached_dense(
            values,
            scope,
            context,
            config,
            mode="shared_parzen",
        )
        output = tempered_residual_posterior(
            values,
            posterior,
            activity_threshold_z=float(row["activity_threshold_z"]),
            temperature_z=float(row["temperature_z"]),
            posterior_authority=float(row["posterior_authority"]),
            correction_limit_z=float(row["correction_limit_z"]),
        )
    elif family == "graph_spatial_diffusion":
        output = graph_edge_aware_diffusion(
            values,
            quiet_count=quiet_count,
            radius=int(row["radius"]),
            signal_bandwidth_z=float(row["signal_bandwidth_z"]),
            guide_bandwidth_z=float(row["guide_bandwidth_z"]),
            iterations=int(row["iterations"]),
            alpha=float(row["alpha"]),
            device=device,
            frame_batch_size=int(config.resources["frame_batch_size"]),
        )
    elif family == "cross_scale_consensus":
        output = cross_scale_consensus_shrinkage(
            values,
            spatial_scales_px=row["spatial_scales_px"],
            agreement_power=float(row["agreement_power"]),
            evidence_threshold_z=float(row["evidence_threshold_z"]),
            gain_floor=float(row["gain_floor"]),
            alpha=float(row["alpha"]),
        )
    elif family == "blindspot_self_supervised":
        model = _blindspot_model(row, context)
        output = apply_blindspot_linear_model(
            values,
            model,
            alpha=float(row["alpha"]),
            correction_limit_z=float(row["correction_limit_z"]),
            device=device,
            frame_batch_size=int(config.resources["frame_batch_size"]),
        )
        diagnostics = {
            "blindspot_fit_mse": float(model.fit_mse),
            "blindspot_weight_l1": float(np.sum(np.abs(model.weights))),
            "blindspot_intercept": float(model.intercept),
        }
    else:
        raise ValueError(f"unknown innovation denoising family: {family}")
    if output.shape != values.shape or not np.isfinite(output).all():
        raise RuntimeError(f"invalid output for {_variant_id(row)}")
    return output.astype(np.float32), {
        "family_id": family,
        "parameters": _parameters(row),
        "algorithm_diagnostics": diagnostics,
    }


def _selection_score(
    metrics: dict[str, Any], baseline: dict[str, Any]
) -> float:
    candidate_ratio = float(metrics["event_candidates"]) / max(
        float(baseline["event_candidates"]), 1
    )
    return float(
        metrics["fixed_budget_mean_recall"]
        + 0.5 * metrics["mean_recall"]
        + 0.15 * np.clip(metrics["median_peak_retention"], 0, 1)
        + 0.15 * np.clip(metrics["median_area_retention"], 0, 1)
        + 0.15 * np.clip(metrics["median_waveform_correlation"], 0, 1)
        + 0.2 * np.clip(metrics["synthetic_correlation"], -1, 1)
        - 0.1 * candidate_ratio
        - 0.05 * float(metrics["median_peak_frame_error"])
    )


def _advancement_gate(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    config: InnovationDenoisingConfig,
) -> bool:
    evaluation = config.evaluation
    candidate_ratio = float(metrics["event_candidates"]) / max(
        float(baseline["event_candidates"]), 1
    )
    fixed_gain = (
        float(metrics["fixed_budget_mean_recall"])
        - float(baseline["fixed_budget_mean_recall"])
    )
    return bool(
        metrics["median_peak_retention"]
        >= float(evaluation["minimum_peak_retention"])
        and metrics["median_area_retention"]
        >= float(evaluation["minimum_area_retention"])
        and metrics["median_peak_frame_error"]
        <= float(evaluation["maximum_peak_frame_error"])
        and metrics["synthetic_correlation"]
        >= float(evaluation["minimum_synthetic_correlation"])
        and fixed_gain
        >= float(evaluation["minimum_fixed_budget_recall_gain"])
        and candidate_ratio <= float(evaluation["maximum_candidate_multiplier"])
    )


def _metrics_for_outputs(
    *,
    variant_id: str,
    family_id: str,
    parameters: dict[str, Any],
    algorithm_diagnostics: dict[str, Any],
    estimate_z: np.ndarray,
    synthetic_estimate_z: np.ndarray,
    residual: np.ndarray,
    scale: np.ndarray,
    labels: list[dict[str, Any]],
    quiet_count: int,
    synthetic_z: np.ndarray,
    synthetic_scale: np.ndarray,
    synthetic_truth: np.ndarray,
    config: InnovationDenoisingConfig,
    runtime_seconds: float,
    baseline: dict[str, Any] | None,
) -> tuple[dict[str, Any], np.ndarray]:
    signal = estimate_z * scale[None]
    synthetic_signal = synthetic_estimate_z * synthetic_scale[None]
    roi_selected, roi_matrix = _roi_matrix(
        residual.shape[1:],
        labels,
        int(config.evaluation["roi_radius_px"]),
    )
    metrics = _variant_metrics(
        variant_id,
        signal,
        residual,
        labels,
        roi_selected,
        roi_matrix,
        quiet_count,
        config,
        runtime_seconds,
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
        family_id=family_id,
        parameters=parameters,
        algorithm_diagnostics=algorithm_diagnostics,
    )
    if baseline is None:
        metrics.update(
            selection_score=0.0,
            advancement_gate_pass=False,
            fixed_budget_recall_gain=0.0,
            candidate_multiplier=1.0,
        )
    else:
        metrics["selection_score"] = _selection_score(metrics, baseline)
        metrics["advancement_gate_pass"] = _advancement_gate(
            metrics, baseline, config
        )
        metrics["fixed_budget_recall_gain"] = float(
            metrics["fixed_budget_mean_recall"]
            - baseline["fixed_budget_mean_recall"]
        )
        metrics["candidate_multiplier"] = float(
            metrics["event_candidates"] / max(baseline["event_candidates"], 1)
        )
    return metrics, signal


def _evaluate_variant(
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
    config: InnovationDenoisingConfig,
    *,
    scope: str,
    baseline: dict[str, Any],
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
    synthetic_estimate_z, synthetic_diagnostics = _apply_variant(
        synthetic_z,
        row,
        context,
        config,
        quiet_count=min(quiet_count, max(8, len(synthetic_z) // 4)),
        scope=scope + "_synthetic",
    )
    metrics, signal = _metrics_for_outputs(
        variant_id=_variant_id(row),
        family_id=row["family_id"],
        parameters=diagnostics["parameters"],
        algorithm_diagnostics={
            "real": diagnostics["algorithm_diagnostics"],
            "synthetic": synthetic_diagnostics["algorithm_diagnostics"],
        },
        estimate_z=estimate_z,
        synthetic_estimate_z=synthetic_estimate_z,
        residual=residual,
        scale=scale,
        labels=labels,
        quiet_count=quiet_count,
        synthetic_z=synthetic_z,
        synthetic_scale=synthetic_scale,
        synthetic_truth=synthetic_truth,
        config=config,
        runtime_seconds=time.perf_counter() - started,
        baseline=baseline,
    )
    metrics["variant_index"] = int(row["variant_index"])
    return metrics, signal


def _baseline_metrics(
    standardized: np.ndarray,
    residual: np.ndarray,
    scale: np.ndarray,
    labels: list[dict[str, Any]],
    quiet_count: int,
    synthetic_z: np.ndarray,
    synthetic_scale: np.ndarray,
    synthetic_truth: np.ndarray,
    config: InnovationDenoisingConfig,
) -> dict[str, Any]:
    metrics, _ = _metrics_for_outputs(
        variant_id="reference_parzen_innovation",
        family_id="identity_carrier",
        parameters={},
        algorithm_diagnostics={},
        estimate_z=standardized,
        synthetic_estimate_z=synthetic_z,
        residual=residual,
        scale=scale,
        labels=labels,
        quiet_count=quiet_count,
        synthetic_z=synthetic_z,
        synthetic_scale=synthetic_scale,
        synthetic_truth=synthetic_truth,
        config=config,
        runtime_seconds=0.0,
        baseline=None,
    )
    return metrics


def _mixture_specs(
    sources: Sequence[dict[str, Any]], config: InnovationDenoisingConfig
) -> list[dict[str, Any]]:
    pair_weight = float(config.mixture["pair_weight"])
    specs: list[dict[str, Any]] = []
    for left, right in combinations(range(len(sources)), 2):
        specs.append(
            {
                "source_indices": [left, right],
                "weights": [pair_weight, pair_weight],
            }
        )
    for weight in config.mixture["all_source_weights"]:
        specs.append(
            {
                "source_indices": list(range(len(sources))),
                "weights": [float(weight)] * len(sources),
            }
        )
    return [
        {
            "family_id": "bounded_pareto_mixture",
            "variant_index": index,
            **spec,
        }
        for index, spec in enumerate(specs, start=1)
    ]


def _report(path: Path, payload: dict[str, Any]) -> None:
    baseline = payload["baseline"]
    lines = [
        f"# {payload['experiment_id']}",
        "",
        "## Outcome",
        "",
        payload["conclusion"],
        "",
        "The identity carrier scored "
        f"{baseline['mean_recall']:.3f} threshold recall, "
        f"{baseline['fixed_budget_mean_recall']:.3f} fixed-budget recall, and "
        f"{baseline['event_candidates']} candidates.",
        "",
        "## Family finalists",
        "",
        "| Family | Variant | Recall | Fixed | Candidates | Peak | Area | Synthetic r | Gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["family_finalists"]:
        lines.append(
            f"| `{row['family_id']}` | `{row['variant_id']}` | "
            f"{row['mean_recall']:.3f} | "
            f"{row['fixed_budget_mean_recall']:.3f} | "
            f"{row['event_candidates']} | "
            f"{row['median_peak_retention']:.3f} | "
            f"{row['median_area_retention']:.3f} | "
            f"{row['synthetic_correlation']:.3f} | "
            f"{'pass' if row['advancement_gate_pass'] else 'stop'} |"
        )
    lines.extend(
        [
            "",
            "## Pareto mixtures",
            "",
            "| Variant | Sources | Recall | Fixed | Candidates | Synthetic r | Gate |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["mixtures"]:
        lines.append(
            f"| `{row['variant_id']}` | "
            f"{', '.join(row['source_variant_ids'])} | "
            f"{row['mean_recall']:.3f} | "
            f"{row['fixed_budget_mean_recall']:.3f} | "
            f"{row['event_candidates']} | "
            f"{row['synthetic_correlation']:.3f} | "
            f"{'pass' if row['advancement_gate_pass'] else 'stop'} |"
        )
    lines.extend(
        [
            "",
            "Unmatched candidates are unknown because real labels are sparse. "
            "Candidate count is therefore a precision-pressure proxy, not a "
            "false-positive count. Exact synthetic metrics remain the only "
            "complete signal/noise ground truth in this run.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config: InnovationDenoisingConfig) -> dict[str, Any]:
    """Execute breadth, full-field, Pareto mixture, and gated seed stages."""
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(config.resources["cpu_threads"])
    audit = _matching_preflight(config)
    partial = Path(str(config.output_dir) + ".partial")
    partial.mkdir(parents=True)
    for directory in (
        "stage_a",
        "stage_b",
        "mixtures",
        "confirmation",
        "finalists",
    ):
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
    crop_baseline = _baseline_metrics(
        crop_standardized,
        crop_residual,
        crop_scale,
        crop_labels,
        quiet_count,
        synthetic_z,
        synthetic_scale,
        synthetic_truth,
        config,
    )
    full_baseline = _baseline_metrics(
        standardized,
        residual,
        scale,
        labels,
        quiet_count,
        synthetic_z,
        synthetic_scale,
        synthetic_truth,
        config,
    )
    _atomic_json(partial / "baseline.json", full_baseline)
    _atomic_json(
        partial / "common_contract.json",
        {
            "input_lane": config.input_lane,
            "innovation_calibration": innovation,
            "quiet_standardization": standardization,
            "stage_a_crop": crop,
            "breadth_combination_count": config.breadth_combination_count,
            "full_field_semifinal_count": (
                config.full_field_combination_count
            ),
            "family_finalist_count": config.family_finalist_count,
            "pareto_mixture_count": config.mixture_combination_count,
            "maximum_confirmation_refit_count": (
                config.maximum_confirmation_refit_count
            ),
        },
    )
    _progress(progress, "shared_ica_fit_start")
    context = _fit_context(standardized, quiet_count, config)
    designs = config.designs()
    design_lookup = {
        _variant_id(row): row
        for rows in designs.values()
        for row in rows
    }

    stage_a_rows: list[dict[str, Any]] = []
    counter = 0
    for family in FAMILY_IDS:
        for row in designs[family]:
            counter += 1
            _progress(
                progress,
                "stage_a_start",
                variant_id=_variant_id(row),
                combination_index=counter,
                combination_total=config.breadth_combination_count,
            )
            metrics, signal = _evaluate_variant(
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
                baseline=crop_baseline,
            )
            stage_a_rows.append(metrics)
            _atomic_json(
                partial / "stage_a" / f"{_variant_id(row)}.json", metrics
            )
            del signal
            gc.collect()
    _atomic_json(
        partial / "stage_a" / "metrics.json",
        {"combination_count": len(stage_a_rows), "rows": stage_a_rows},
    )
    selected_a: list[dict[str, Any]] = []
    for family in FAMILY_IDS:
        ranked = sorted(
            (row for row in stage_a_rows if row["family_id"] == family),
            key=lambda row: (
                row["advancement_gate_pass"],
                row["selection_score"],
            ),
            reverse=True,
        )
        selected_a.extend(
            ranked[: int(config.stages["stage_a_top_per_family"])]
        )
    _atomic_json(
        partial / "stage_a" / "selection.json",
        {
            "selected_count": len(selected_a),
            "selected_variant_ids": [row["variant_id"] for row in selected_a],
        },
    )
    _resource_checkpoint(config, progress, "stage_a_complete")

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
        metrics, signal = _evaluate_variant(
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
            baseline=full_baseline,
        )
        stage_b_rows.append(metrics)
        _atomic_json(
            partial / "stage_b" / f"{_variant_id(row)}.json", metrics
        )
        del signal
        gc.collect()
    _atomic_json(
        partial / "stage_b" / "metrics.json",
        {"combination_count": len(stage_b_rows), "rows": stage_b_rows},
    )
    family_finalists: list[dict[str, Any]] = []
    for family in FAMILY_IDS:
        ranked = sorted(
            (row for row in stage_b_rows if row["family_id"] == family),
            key=lambda row: (
                row["advancement_gate_pass"],
                row["selection_score"],
            ),
            reverse=True,
        )
        family_finalists.extend(
            ranked[: int(config.stages["stage_b_top_per_family"])]
        )
    _atomic_json(
        partial / "stage_b" / "selection.json",
        {
            "selected_count": len(family_finalists),
            "selected_variant_ids": [
                row["variant_id"] for row in family_finalists
            ],
        },
    )
    _resource_checkpoint(config, progress, "stage_b_complete")

    pareto_sources = select_diverse_pareto_rows(
        family_finalists,
        maximize=config.mixture["maximize_objectives"],
        minimize=config.mixture["minimize_objectives"],
        count=int(config.mixture["pareto_source_count"]),
    )
    _atomic_json(
        partial / "mixtures" / "pareto_sources.json",
        {
            "source_count": len(pareto_sources),
            "source_variant_ids": [
                row["variant_id"] for row in pareto_sources
            ],
            "maximize": config.mixture["maximize_objectives"],
            "minimize": config.mixture["minimize_objectives"],
        },
    )
    context["cache"].clear()
    source_real: list[np.ndarray] = []
    source_synthetic: list[np.ndarray] = []
    for selected in pareto_sources:
        row = design_lookup[selected["variant_id"]]
        real_z, _ = _apply_variant(
            standardized,
            row,
            context,
            config,
            quiet_count=quiet_count,
            scope="mixture_source_full",
        )
        synthetic_output_z, _ = _apply_variant(
            synthetic_z,
            row,
            context,
            config,
            quiet_count=min(quiet_count, max(8, len(synthetic_z) // 4)),
            scope="mixture_source_synthetic",
        )
        source_real.append(real_z)
        source_synthetic.append(synthetic_output_z)
    mixture_rows: list[dict[str, Any]] = []
    mixture_specs = _mixture_specs(pareto_sources, config)
    for spec in mixture_specs:
        mixture_id = f"bounded_pareto_mixture__{spec['variant_index']:02d}"
        _progress(
            progress,
            "mixture_start",
            variant_id=mixture_id,
            combination_index=int(spec["variant_index"]),
            combination_total=len(mixture_specs),
        )
        started_mix = time.perf_counter()
        indices = spec["source_indices"]
        real_z = bounded_mixture(
            standardized,
            [source_real[index] for index in indices],
            spec["weights"],
            correction_limit_z=float(config.mixture["correction_limit_z"]),
        )
        synthetic_output_z = bounded_mixture(
            synthetic_z,
            [source_synthetic[index] for index in indices],
            spec["weights"],
            correction_limit_z=float(config.mixture["correction_limit_z"]),
        )
        source_ids = [pareto_sources[index]["variant_id"] for index in indices]
        metrics, signal = _metrics_for_outputs(
            variant_id=mixture_id,
            family_id="bounded_pareto_mixture",
            parameters={
                "source_variant_ids": source_ids,
                "weights": spec["weights"],
                "correction_limit_z": config.mixture["correction_limit_z"],
            },
            algorithm_diagnostics={},
            estimate_z=real_z,
            synthetic_estimate_z=synthetic_output_z,
            residual=residual,
            scale=scale,
            labels=labels,
            quiet_count=quiet_count,
            synthetic_z=synthetic_z,
            synthetic_scale=synthetic_scale,
            synthetic_truth=synthetic_truth,
            config=config,
            runtime_seconds=time.perf_counter() - started_mix,
            baseline=full_baseline,
        )
        metrics.update(
            variant_index=int(spec["variant_index"]),
            source_variant_ids=source_ids,
            weights=spec["weights"],
        )
        mixture_rows.append(metrics)
        _atomic_json(
            partial / "mixtures" / f"{mixture_id}.json", metrics
        )
        del real_z, synthetic_output_z, signal
    _atomic_json(
        partial / "mixtures" / "metrics.json",
        {"combination_count": len(mixture_rows), "rows": mixture_rows},
    )
    _resource_checkpoint(config, progress, "mixtures_complete")
    ranked_mixtures = sorted(
        mixture_rows,
        key=lambda row: (
            row["advancement_gate_pass"],
            row["selection_score"],
        ),
        reverse=True,
    )
    top_mixtures = ranked_mixtures[
        : int(config.stages["write_top_mixture_tiffs"])
    ]

    common_max = _display_scale(
        np.maximum(centered, 0),
        float(config.visualization["positive_upper_percentile"]),
        config,
    )
    context["cache"].clear()
    gc.collect()
    family_payload: list[dict[str, Any]] = []
    for selected in family_finalists:
        row = design_lookup[selected["variant_id"]]
        _progress(
            progress,
            "family_finalist_tiff_start",
            variant_id=row["family_id"],
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
        family_payload.append({**selected, "videos": videos})
        del estimate_z, signal
        context["cache"].clear()
        gc.collect()
    mixture_payload: list[dict[str, Any]] = []
    for rank, selected in enumerate(top_mixtures, start=1):
        indices = [
            next(
                index
                for index, row in enumerate(pareto_sources)
                if row["variant_id"] == source_id
            )
            for source_id in selected["source_variant_ids"]
        ]
        estimate_z = bounded_mixture(
            standardized,
            [source_real[index] for index in indices],
            selected["weights"],
            correction_limit_z=float(config.mixture["correction_limit_z"]),
        )
        signal = estimate_z * scale[None]
        videos = _write_tiffs(
            partial / "finalists" / f"mixture_{rank:02d}",
            signal,
            residual,
            common_max,
            selected["variant_id"],
            selected["parameters"],
            config,
        )
        mixture_payload.append({**selected, "videos": videos})
        del estimate_z, signal

    promoted = sorted(
        [
            row
            for row in [*family_finalists, *mixture_rows]
            if row["advancement_gate_pass"]
        ],
        key=lambda row: row["selection_score"],
        reverse=True,
    )[: int(config.stages["confirmation_top_count"])]
    confirmation_rows: list[dict[str, Any]] = []
    for candidate in promoted:
        for seed in config.stages["confirmation_seeds"]:
            _progress(
                progress,
                "confirmation_refit_start",
                variant_id=candidate["variant_id"],
                seed=int(seed),
            )
            seed_context = _fit_context(
                standardized, quiet_count, config, seed=int(seed)
            )
            if candidate["family_id"] == "bounded_pareto_mixture":
                real_outputs = []
                synthetic_outputs = []
                for source_id in candidate["source_variant_ids"]:
                    source_row = design_lookup[source_id]
                    real_z, _ = _apply_variant(
                        standardized,
                        source_row,
                        seed_context,
                        config,
                        quiet_count=quiet_count,
                        scope=f"confirm_{seed}_real",
                    )
                    synthetic_output_z, _ = _apply_variant(
                        synthetic_z,
                        source_row,
                        seed_context,
                        config,
                        quiet_count=min(
                            quiet_count, max(8, len(synthetic_z) // 4)
                        ),
                        scope=f"confirm_{seed}_synthetic",
                    )
                    real_outputs.append(real_z)
                    synthetic_outputs.append(synthetic_output_z)
                real_z = bounded_mixture(
                    standardized,
                    real_outputs,
                    candidate["weights"],
                    correction_limit_z=float(
                        config.mixture["correction_limit_z"]
                    ),
                )
                synthetic_output_z = bounded_mixture(
                    synthetic_z,
                    synthetic_outputs,
                    candidate["weights"],
                    correction_limit_z=float(
                        config.mixture["correction_limit_z"]
                    ),
                )
            else:
                candidate_row = design_lookup[candidate["variant_id"]]
                real_z, _ = _apply_variant(
                    standardized,
                    candidate_row,
                    seed_context,
                    config,
                    quiet_count=quiet_count,
                    scope=f"confirm_{seed}_real",
                )
                synthetic_output_z, _ = _apply_variant(
                    synthetic_z,
                    candidate_row,
                    seed_context,
                    config,
                    quiet_count=min(
                        quiet_count, max(8, len(synthetic_z) // 4)
                    ),
                    scope=f"confirm_{seed}_synthetic",
                )
            confirmation, signal = _metrics_for_outputs(
                variant_id=candidate["variant_id"],
                family_id=candidate["family_id"],
                parameters=candidate["parameters"],
                algorithm_diagnostics={},
                estimate_z=real_z,
                synthetic_estimate_z=synthetic_output_z,
                residual=residual,
                scale=scale,
                labels=labels,
                quiet_count=quiet_count,
                synthetic_z=synthetic_z,
                synthetic_scale=synthetic_scale,
                synthetic_truth=synthetic_truth,
                config=config,
                runtime_seconds=0.0,
                baseline=full_baseline,
            )
            for fold in confirmation["detection_folds"]:
                confirmation_rows.append(
                    {
                        "variant_id": candidate["variant_id"],
                        "family_id": candidate["family_id"],
                        "seed": int(seed),
                        **fold,
                        "synthetic_correlation": confirmation[
                            "synthetic_correlation"
                        ],
                        "peak_retention": confirmation[
                            "median_peak_retention"
                        ],
                        "area_retention": confirmation[
                            "median_area_retention"
                        ],
                        "gate_pass": confirmation[
                            "advancement_gate_pass"
                        ],
                    }
                )
            del real_z, synthetic_output_z, signal, seed_context
            gc.collect()
    _resource_checkpoint(config, progress, "tiffs_and_confirmation_complete")
    confirmation_status = (
        "actual_seed_refits_completed"
        if confirmation_rows
        else "not_run_no_candidate_passed_advancement_gate"
    )
    _atomic_json(
        partial / "confirmation" / "records.json",
        {
            "status": confirmation_status,
            "promoted_variant_ids": [row["variant_id"] for row in promoted],
            "record_count": len(confirmation_rows),
            "rows": confirmation_rows,
        },
    )
    _atomic_tsv(
        partial / "confirmation" / "records.tsv",
        confirmation_rows
        or [
            {
                "status": confirmation_status,
                "record_count": 0,
            }
        ],
    )

    passed = [
        row
        for row in [*family_finalists, *mixture_rows]
        if row["advancement_gate_pass"]
    ]
    all_candidates = [*family_finalists, *mixture_rows]
    best = max(
        all_candidates,
        key=lambda row: (
            row["advancement_gate_pass"],
            row["selection_score"],
        ),
    )
    conclusion = (
        f"{len(passed)} of {len(all_candidates)} family-finalist and mixture "
        f"candidates passed the joint carrier-relative gate. The strongest "
        f"candidate by the preregistered ordering was {best['variant_id']}. "
        f"Confirmation status: {confirmation_status}."
    )
    metrics = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": "completed",
        "stage_a_combination_count": len(stage_a_rows),
        "stage_b_combination_count": len(stage_b_rows),
        "family_finalist_count": len(family_payload),
        "mixture_combination_count": len(mixture_rows),
        "mixture_tiff_count": len(mixture_payload),
        "confirmation_status": confirmation_status,
        "confirmation_record_count": len(confirmation_rows),
        "baseline": full_baseline,
        "family_finalists": family_payload,
        "pareto_source_variant_ids": [
            row["variant_id"] for row in pareto_sources
        ],
        "mixtures": mixture_rows,
        "top_mixtures": mixture_payload,
        "passed_candidate_ids": [row["variant_id"] for row in passed],
        "best_candidate": best["variant_id"],
        "conclusion": conclusion,
        "elapsed_seconds": time.time() - started,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "scientific_status": (
            "screening_complete_sparse_real_labels_candidate_burden_is_proxy"
        ),
    }
    _atomic_json(partial / "metrics.json", metrics)
    flat = [
        {
            key: value
            for key, value in row.items()
            if not isinstance(value, (dict, list))
        }
        for row in all_candidates
    ]
    _atomic_tsv(partial / "candidate_comparison.tsv", flat)
    _report(partial / "REPORT.md", metrics)
    _atomic_json(
        partial / "run_state.json",
        {
            "status": "completed",
            "completed_unix": time.time(),
            "elapsed_seconds": metrics["elapsed_seconds"],
            "max_rss_mib": metrics["max_rss_mib"],
            "stage_a_combination_count": len(stage_a_rows),
            "stage_b_combination_count": len(stage_b_rows),
            "mixture_combination_count": len(mixture_rows),
            "confirmation_record_count": len(confirmation_rows),
            "tiff_count": 2 * (len(family_payload) + len(mixture_payload)),
        },
    )
    partial.replace(config.output_dir)
    return metrics
