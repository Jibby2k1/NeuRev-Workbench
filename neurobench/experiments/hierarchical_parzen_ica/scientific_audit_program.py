"""Guarded acquisition, morphology, and structured-feature audit."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any, Mapping

import numpy as np

from neurobench.algorithms.activity_feature_bank import quiet_robust_z
from neurobench.algorithms.patch_information import cauchy_schwarz_divergence_tensor
from neurobench.algorithms.scientific_feature_audit import (
    causal_local_correlation_feature,
    fit_poisson_gaussian_noise,
    fit_zcut_templates_at_points,
    generalized_anscombe,
    radial_zone_histograms_tensor,
    zcut_response_maps,
    zcut_template_bank,
)
from neurobench.experiments.learnable_contrast import core as label_core
from neurobench.experiments.pairwise_separation.evaluation import event_intervals

from .feature_utility_config import FeatureUtilityConfig
from .innovation_grid import (
    _atomic_json, _available_ram_mib, _progress, _sha256, _snapshots,
)
from .innovation_ranker_config import InnovationRankerConfig
from .innovation_ranker_program import (
    _generate_maps as _generate_v5_maps,
    _write_score_tiff,
)
from .multiscale_information_program import (
    _crossfit, _identical_proposal_rows, _lane_rows,
)
from .patch_information_program import BURSTS, _pool_values
from .patch_information_video import display_limits, _write_display_tiff
from .scientific_audit_config import ScientificAuditConfig


RADIAL_FEATURES = (
    "radial_cs_center", "radial_cs_shell", "radial_cs_outer",
    "radial_cs_center_contrast", "radial_cs_shell_contrast",
    "radial_cs_morph_max",
)
MORPHOLOGY_FEATURES = (
    "zcut_cytosol_center", "zcut_membrane_ring", "zcut_membrane_cap",
    "zcut_crowd_context",
)


def _feature_inventory(config: ScientificAuditConfig) -> tuple[tuple[str, ...], dict[str, str]]:
    coherence = tuple(
        f"coherence_w{int(window)}"
        for window in config.propagation["coherence_windows_frames"]
    )
    propagation = tuple(
        f"propagation_lag{int(lag)}_w{int(window)}"
        for lag, window in config.propagation["lag_window_pairs"]
    )
    feature_ids = (*RADIAL_FEATURES, *MORPHOLOGY_FEATURES, *coherence, *propagation)
    families = {
        **{feature_id: "radial_information" for feature_id in RADIAL_FEATURES},
        **{feature_id: "generative_zcut" for feature_id in MORPHOLOGY_FEATURES},
        **{feature_id: "local_coherence" for feature_id in coherence},
        **{feature_id: "lagged_propagation" for feature_id in propagation},
    }
    if len(feature_ids) != len(set(feature_ids)) or len(feature_ids) != config.feature_count:
        raise RuntimeError("scientific feature inventory mismatch")
    return tuple(feature_ids), families


def preflight(config: ScientificAuditConfig, *, write_artifacts: bool = True) -> dict[str, Any]:
    inputs = (
        config.source_ranker_config,
        config.source_multiscale_root / "metrics.json",
        config.source_video,
        config.carrier_path,
        config.labels_tsv,
    )
    missing = [str(path) for path in inputs if not path.is_file()]
    video_shape = carrier_shape = None
    labels: list[dict[str, Any]] = []
    source_valid = labels_valid = split_valid = False
    if not missing:
        source = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
        carrier = np.load(config.carrier_path, mmap_mode="r", allow_pickle=False)
        video_shape, carrier_shape = list(source.shape), list(carrier.shape)
        expected = config.frames["review_end_ui"] - config.frames["review_start_ui"] + 1
        source_valid = bool(
            source.shape == (2359, 340, 573)
            and carrier.shape == (expected, 340, 573)
            and source.dtype == np.uint16
        )
        labels = label_core.load_labels(config.labels_tsv)
        labels_valid = len(labels) == 79 and len({row["roi_identity"] for row in labels}) == 27
        split = int(config.acquisition["split_x_px"])
        split_valid = bool(all(float(row["x_px"]) >= split for row in labels))
    gpu: dict[str, Any] = {"available": False}
    try:
        import torch
        gpu["available"] = bool(torch.cuda.is_available())
        if gpu["available"]:
            free, total = torch.cuda.mem_get_info()
            gpu.update(name=torch.cuda.get_device_name(0), free_mib=free/2**20, total_mib=total/2**20)
    except ImportError:
        pass
    feature_ids, _ = _feature_inventory(config)
    estimated_ram = 7168.0
    estimated_gpu = 3072.0
    estimated_output = 1536.0
    probe = config.output_dir.parent
    while not probe.exists():
        probe = probe.parent
    free_disk = shutil.disk_usage(probe).free / 2**20
    partial = Path(str(config.output_dir) + ".partial")
    gates = {
        "inputs_exist": not missing,
        "source_geometry_valid": source_valid,
        "labels_valid": labels_valid,
        "all_labels_in_provisional_right_field": split_valid,
        "feature_inventory_valid": len(feature_ids) == 16,
        "evaluated_lane_count_valid": config.evaluated_lane_count == 192,
        "output_absent": not config.output_dir.exists(),
        "partial_output_absent": not partial.exists(),
        "preflight_separate_from_output": config.preflight_dir != config.output_dir,
        "ram_cap_sufficient": estimated_ram <= int(config.resources["max_ram_mib"]),
        "available_ram_sufficient": estimated_ram <= _available_ram_mib(),
        "gpu_available": gpu["available"],
        "gpu_cap_sufficient": estimated_gpu <= int(config.resources["max_gpu_memory_mib"]),
        "live_gpu_sufficient": estimated_gpu <= gpu.get("free_mib", 0),
        "output_cap_sufficient": estimated_output <= int(config.resources["max_output_mib"]),
        "disk_headroom_sufficient": free_disk >= estimated_output + int(config.resources["min_free_disk_mib"]),
    }
    payload = {
        "schema_version": 1,
        "kind": "read_only_spon_scientific_feature_audit_preflight",
        "experiment_id": config.experiment_id,
        "ready": all(gates.values()), "gates": gates,
        "video_shape": video_shape, "carrier_shape": carrier_shape,
        "label_rows": len(labels), "roi_identities": len({row["roi_identity"] for row in labels}),
        "design": {
            "feature_ids": list(feature_ids), "feature_count": len(feature_ids),
            "full_native_lanes": config.lane_count_per_field,
            "right_native_lanes": config.lane_count_per_field,
            "identical_proposal_lanes": config.lane_count_per_field,
            "evaluated_lane_count": config.evaluated_lane_count,
        },
        "acquisition_assumptions": {
            "provisional_split_x_px": config.acquisition["split_x_px"],
            "right_field_is_annotated": split_valid,
            "halves_semantics": "unknown; audited separately; no correspondence assumed",
            "frame_period": config.acquisition["frame_period_ms"],
            "frame_period_provenance": config.acquisition["frame_period_provenance"],
            "pixel_size_um": None, "microscope_type": None, "indicator": None,
            "objective_na": None, "z_plane_thickness_um": None,
        },
        "resources": {
            "estimated_peak_ram_mib": estimated_ram,
            "available_ram_mib": _available_ram_mib(),
            "estimated_peak_gpu_memory_mib": estimated_gpu,
            "estimated_output_mib": estimated_output,
            "free_disk_mib": free_disk, "gpu": gpu, **config.resources,
        },
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in inputs if path.is_file()
        ],
        "system_snapshot": _snapshots(),
        "scientific_contract": (
            "Noise fits are descriptive, z-cut phenotypes are hypotheses rather than "
            "ground truth, propagation is causal correlation rather than causality, and "
            "unmatched candidates remain unknown."
        ),
    }
    if write_artifacts:
        config.preflight_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(config.preflight_dir / "preflight.json", payload)
        _atomic_json(config.preflight_dir / "config.resolved.json", config.to_dict())
        if source_valid and labels_valid:
            label_core._write_overlay(
                np.load(config.source_video, mmap_mode="r", allow_pickle=False),
                labels, config.preflight_dir / "label_projection_overlay.png",
            )
    if not payload["ready"]:
        raise RuntimeError(f"scientific audit preflight failed: {payload}")
    return payload


def _matching_preflight(config: ScientificAuditConfig) -> dict[str, Any]:
    audit = json.loads((config.preflight_dir / "preflight.json").read_text(encoding="utf-8"))
    resolved = json.loads((config.preflight_dir / "config.resolved.json").read_text(encoding="utf-8"))
    if not audit.get("ready") or resolved != config.to_dict():
        raise RuntimeError("run requires a matching ready preflight")
    if config.output_dir.exists() or Path(str(config.output_dir) + ".partial").exists():
        raise FileExistsError("completed or partial output already exists")
    return audit


def _write_video(
    values: np.ndarray, path: Path, config: ScientificAuditConfig,
    *, feature_id: str, signed: bool = False,
) -> dict[str, Any]:
    quiet = int(config.frames["quiet_count"])
    if signed:
        sampled = np.asarray(values[::4, ::4, ::4], dtype=np.float32)
        limit = max(float(np.percentile(np.abs(sampled), 99.8)), 1e-8)
        shifted = np.clip(np.asarray(values, dtype=np.float32) / (2*limit) + 0.5, 0, 1)
        black, white = 0.0, 1.0
        source = shifted
    else:
        black, white = display_limits(
            values, quiet_count=quiet,
            upper_percentile=float(config.visualization["upper_percentile"]),
        )
        source = values
    _write_display_tiff(
        source, path, black=black, white=white,
        compression=str(config.visualization["compression"]),
        description={
            "feature_id": feature_id, "frame_count": len(values),
            "source_ui_frames_inclusive": [config.frames["review_start_ui"], config.frames["review_end_ui"]],
            "global_display": {"black": black, "white": white, "signed_zero": 0.5 if signed else None},
        },
    )
    return {"path": str(path.name), "bytes": path.stat().st_size, "frames": len(values)}


def _clean_noise_model(model: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in model.items() if not isinstance(value, np.ndarray)}


def _noise_stage(
    config: ScientificAuditConfig, raw: np.ndarray, diagnostics: Path,
    videos: Path, progress: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from scipy.ndimage import gaussian_filter

    quiet_count = int(config.frames["quiet_count"])
    split = int(config.acquisition["split_x_px"])
    saturation = float(config.acquisition["saturation_value"])
    slices = {"full": slice(None), "left": slice(0, split), "right": slice(split, None)}
    models = {
        name: fit_poisson_gaussian_noise(
            raw[:quiet_count, :, xs],
            intensity_bins=int(config.noise["intensity_bins"]),
            saturation_value=saturation,
        )
        for name, xs in slices.items()
    }
    trace = raw.mean(axis=(1, 2), dtype=np.float64)
    left_trace = raw[:, :, :split].mean(axis=(1, 2), dtype=np.float64)
    right_trace = raw[:, :, split:].mean(axis=(1, 2), dtype=np.float64)
    time_axis = np.arange(len(raw), dtype=np.float64)
    full_model = models["full"]
    mean_map = full_model["quiet_mean_map"]
    variance_map = full_model["pair_difference_variance_map"]
    smooth_mean = gaussian_filter(
        mean_map, sigma=float(config.noise["spatial_diagnostic_sigma_px"]), mode="reflect"
    )
    fixed_pattern = mean_map - smooth_mean
    saturation_occupancy = np.mean(raw >= saturation, axis=0, dtype=np.float64).astype(np.float32)
    diagnostic = _write_score_tiff(
        diagnostics / "noise_physics_maps.tif",
        [mean_map, variance_map, np.abs(fixed_pattern), saturation_occupancy],
        compression=str(config.visualization["compression"]),
        description={
            "page_order": [
                "quiet_mean",
                "pair_difference_variance",
                "fixed_pattern_absolute_residual",
                "saturation_occupancy",
            ],
            "split_x_px": split,
        },
    )
    transformed = generalized_anscombe(
        raw,
        variance_intercept=full_model["variance_intercept_raw2"],
        variance_slope=full_model["variance_slope_raw"],
    )
    vst_residual = quiet_robust_z(transformed, quiet_count)
    video = _write_video(
        vst_residual, videos / "noise_vst_residual.tif", config,
        feature_id="noise_vst_residual", signed=True,
    )
    result = {
        "field_models": {name: _clean_noise_model(model) for name, model in models.items()},
        "saturation_fraction": {
            name: float(np.mean(raw[:, :, xs] >= saturation)) for name, xs in slices.items()
        },
        "global_intensity_drift_raw_per_frame": float(np.polyfit(time_axis, trace, 1)[0]),
        "left_intensity_drift_raw_per_frame": float(np.polyfit(time_axis, left_trace, 1)[0]),
        "right_intensity_drift_raw_per_frame": float(np.polyfit(time_axis, right_trace, 1)[0]),
        "left_right_global_trace_correlation": float(np.corrcoef(left_trace, right_trace)[0, 1]),
        "boundary_mean_absolute_jump_raw": float(np.mean(np.abs(raw[:, :, split-1] - raw[:, :, split]))),
        "internal_adjacent_x_mean_absolute_jump_raw": float(np.median(np.mean(np.abs(np.diff(raw, axis=2)), axis=(0,1)))),
        "diagnostic_tiff": diagnostic,
        "vst_residual_video": video,
        "caveat": "Pair differences include any fast biology; the Poisson-Gaussian fit is descriptive, not a calibrated sensor model.",
    }
    _atomic_json(diagnostics / "noise_audit.json", result)
    _progress(progress, "noise_audit_complete")
    return result, {"mean": mean_map, "variance": variance_map, "saturation": saturation_occupancy}


def _pool_video(values: np.ndarray, labels: list[dict[str, Any]], config: ScientificAuditConfig) -> dict[str, Any]:
    intervals = event_intervals(labels, int(config.frames["review_start_ui"]))
    events = {burst: values[start:stop] for burst, (start, stop) in intervals.items()}
    return _pool_values(
        values[: int(config.frames["quiet_count"])], events, 0.25
    )


def _quiet_calibrate(values: np.ndarray, quiet_count: int) -> np.ndarray:
    quiet = np.asarray(values[:quiet_count], dtype=np.float32)
    baseline = np.median(quiet, axis=0).astype(np.float32)
    sampled = np.maximum(quiet[:, ::4, ::4] - baseline[None, ::4, ::4], 0)
    scale = max(float(np.percentile(sampled, 99.5)), 1e-6)
    return np.maximum(
        (np.asarray(values, dtype=np.float32) - baseline[None]) / scale, 0
    ).astype(np.float16)


def _radial_stage(
    config: ScientificAuditConfig, carrier: np.ndarray,
    labels: list[dict[str, Any]], videos: Path, progress: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    settings = config.radial_information
    centers = tuple(float(value) for value in settings["bin_centers_z"])
    device = torch.device(str(config.resources["device"]))
    batch = int(settings["frame_batch_size"])
    quiet_count = int(config.frames["quiet_count"])
    shape = carrier.shape
    quiet_histograms = {
        name: torch.zeros((len(centers), *shape[1:]), dtype=torch.float32, device=device)
        for name in ("center", "shell", "outer")
    }
    with torch.inference_mode():
        for start in range(0, quiet_count, batch):
            stop = min(quiet_count, start + batch)
            frames = torch.as_tensor(carrier[start:stop], device=device)
            zones = radial_zone_histograms_tensor(
                frames, centers=centers,
                center_radius_px=settings["center_radius_px"],
                shell_radius_px=settings["shell_radius_px"],
                outer_radius_px=settings["outer_radius_px"],
            )
            for name in quiet_histograms:
                quiet_histograms[name] += zones[name].sum(0)
        for name in quiet_histograms:
            quiet_histograms[name] /= float(quiet_count)
    raw = {name: np.empty(shape, dtype=np.float16) for name in quiet_histograms}
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(carrier), batch):
            stop = min(len(carrier), start + batch)
            frames = torch.as_tensor(carrier[start:stop], device=device)
            zones = radial_zone_histograms_tensor(
                frames, centers=centers,
                center_radius_px=settings["center_radius_px"],
                shell_radius_px=settings["shell_radius_px"],
                outer_radius_px=settings["outer_radius_px"],
            )
            for name in raw:
                reference = quiet_histograms[name][None].expand_as(zones[name])
                values = cauchy_schwarz_divergence_tensor(
                    zones[name], reference, centers=centers,
                    bandwidth=float(settings["kernel_bandwidth_z"]),
                )
                raw[name][start:stop] = values.cpu().numpy().astype(np.float16)
    compute_seconds = time.perf_counter() - started
    calibrated = {name: _quiet_calibrate(values, quiet_count) for name, values in raw.items()}
    authority = float(settings["context_authority"])
    center = calibrated["center"]
    shell = calibrated["shell"]
    outer = calibrated["outer"]
    feature_videos = {
        "radial_cs_center": center,
        "radial_cs_shell": shell,
        "radial_cs_outer": outer,
        "radial_cs_center_contrast": np.maximum(
            center.astype(np.float32) - authority * outer.astype(np.float32), 0
        ).astype(np.float16),
        "radial_cs_shell_contrast": np.maximum(
            shell.astype(np.float32) - authority * (
                center.astype(np.float32) + outer.astype(np.float32)
            ), 0
        ).astype(np.float16),
        "radial_cs_morph_max": np.maximum(center, shell),
    }
    maps = {feature_id: _pool_video(values, labels, config) for feature_id, values in feature_videos.items()}
    artifacts = {}
    for feature_id in config.visualization["feature_video_ids"]:
        if feature_id in feature_videos:
            artifacts[feature_id] = _write_video(
                feature_videos[feature_id], videos / f"{feature_id}.tif", config,
                feature_id=feature_id,
            )
    metrics = {
        "feature_ids": list(feature_videos), "compute_seconds": compute_seconds,
        "batched_frames_per_second": len(carrier) / compute_seconds,
        "frame_batch_size": batch, "videos": artifacts,
    }
    _progress(progress, "radial_information_complete", **metrics)
    return maps, metrics


def _propagation_stage(
    config: ScientificAuditConfig, carrier: np.ndarray,
    labels: list[dict[str, Any]], videos: Path, progress: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    maps = {}
    artifacts = {}
    timings = {}
    sigma = float(config.propagation["spatial_sigma_px"])
    specifications = [
        (f"coherence_w{int(window)}", int(window), 0)
        for window in config.propagation["coherence_windows_frames"]
    ] + [
        (f"propagation_lag{int(lag)}_w{int(window)}", int(window), int(lag))
        for lag, window in config.propagation["lag_window_pairs"]
    ]
    for feature_id, window, lag in specifications:
        started = time.perf_counter()
        values = causal_local_correlation_feature(
            carrier, window_frames=window, lag_frames=lag,
            spatial_sigma_px=sigma, activity_qualified=True,
        )
        values = _quiet_calibrate(values, int(config.frames["quiet_count"]))
        timings[feature_id] = time.perf_counter() - started
        maps[feature_id] = _pool_video(values, labels, config)
        if feature_id in config.visualization["feature_video_ids"]:
            artifacts[feature_id] = _write_video(
                values, videos / f"{feature_id}.tif", config, feature_id=feature_id
            )
        del values
        _progress(progress, "propagation_feature_complete", feature_id=feature_id)
    return maps, {"compute_seconds": timings, "videos": artifacts}


def _morphology_stage(
    config: ScientificAuditConfig, carrier_maps: Mapping[str, Any],
    labels: list[dict[str, Any]], diagnostics: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from scipy.ndimage import gaussian_filter

    bank = zcut_template_bank(
        size_px=int(config.morphology["template_size_px"]),
        radii_px=config.morphology["radii_px"],
        z_offsets_fraction=config.morphology["z_offsets_fraction"],
        membrane_thickness_px=float(config.morphology["membrane_thickness_px"]),
        psf_sigmas_px=config.morphology["psf_sigmas_px"],
    )
    phenotype_to_id = {
        "cytosol_center": "zcut_cytosol_center",
        "membrane_ring": "zcut_membrane_ring",
        "membrane_cap": "zcut_membrane_cap",
    }
    maps = {
        feature_id: {"quiet": [], "events": {}}
        for feature_id in MORPHOLOGY_FEATURES
    }
    for image in carrier_maps["quiet"]:
        responses = zcut_response_maps(image, bank)
        for phenotype, feature_id in phenotype_to_id.items():
            maps[feature_id]["quiet"].append(responses[phenotype])
        maps["zcut_crowd_context"]["quiet"].append(
            gaussian_filter(np.maximum(image, 0), float(config.morphology["crowd_sigma_px"]), mode="reflect")
        )
    for burst, image in carrier_maps["events"].items():
        responses = zcut_response_maps(image, bank)
        for phenotype, feature_id in phenotype_to_id.items():
            maps[feature_id]["events"][burst] = responses[phenotype]
        maps["zcut_crowd_context"]["events"][burst] = gaussian_filter(
            np.maximum(image, 0), float(config.morphology["crowd_sigma_px"]), mode="reflect"
        )
    audit_rows = []
    for burst in BURSTS:
        selected = [row for row in labels if int(row["burst_id"]) == burst]
        fits = fit_zcut_templates_at_points(
            carrier_maps["events"][burst],
            [(row["x_px"], row["y_px"]) for row in selected], bank,
        )
        crowd_map = maps["zcut_crowd_context"]["events"][burst]
        for label, fit in zip(selected, fits):
            x, y = int(round(label["x_px"])), int(round(label["y_px"]))
            audit_rows.append({
                "burst_id": burst, "roi_identity": label["roi_identity"],
                "point_index": int(label["point_index"]), **fit,
                "crowd_context": float(crowd_map[y, x]),
            })
    with (diagnostics / "per_label_zcut_audit.tsv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "burst_id", "roi_identity", "point_index", "x_px", "y_px",
            "best_score", "best_phenotype", "best_radius_px",
            "best_z_offset_fraction", "best_psf_sigma_px", "crowd_context",
            "phenotype_scores",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in audit_rows:
            writer.writerow({**row, "phenotype_scores": json.dumps(row["phenotype_scores"], sort_keys=True)})
    pages = [maps[feature_id]["events"][burst] for feature_id in MORPHOLOGY_FEATURES for burst in BURSTS]
    diagnostic = _write_score_tiff(
        diagnostics / "generative_zcut_maps.tif", pages,
        compression=str(config.visualization["compression"]),
        description={"feature_ids": list(MORPHOLOGY_FEATURES), "page_order": "feature-major; bursts 1-4"},
    )
    counts = {
        phenotype: sum(row["best_phenotype"] == phenotype for row in audit_rows)
        for phenotype in sorted({row["best_phenotype"] for row in audit_rows})
    }
    return maps, {
        "template_count": len(bank), "per_label_rows": len(audit_rows),
        "best_phenotype_counts": counts, "diagnostic_tiff": diagnostic,
        "interpretation": "Template classes are fitted hypotheses; phenotype ground truth has not been annotated.",
    }


def _crop_maps(values: Mapping[str, Any], split: int) -> dict[str, Any]:
    return {
        "quiet": [np.asarray(image)[:, split:] for image in values["quiet"]],
        "events": {burst: np.asarray(image)[:, split:] for burst, image in values["events"].items()},
    }


class _EvaluationAdapter:
    def __init__(self, config: ScientificAuditConfig):
        self.evaluation = config.evaluation
        self.fusions = {"carrier_boosts": config.evaluation["carrier_boosts"]}


def _evaluate(
    config: ScientificAuditConfig, maps: Mapping[str, Any], families: Mapping[str, str],
    v5_maps: Mapping[str, Any], carrier: Mapping[str, Any],
    labels: list[dict[str, Any]], ranker: InnovationRankerConfig,
    evaluation_dir: Path,
) -> dict[str, Any]:
    adapter = _EvaluationAdapter(config)
    full_baseline, full_rows = _lane_rows(adapter, maps, carrier, labels, ranker, families)
    full_crossfit = _crossfit(full_rows, adapter)
    split = int(config.acquisition["split_x_px"])
    right_maps = {feature_id: _crop_maps(values, split) for feature_id, values in maps.items()}
    right_carrier = _crop_maps(carrier, split)
    right_labels = [{**row, "x_px": float(row["x_px"]) - split} for row in labels]
    right_baseline, right_rows = _lane_rows(
        adapter, right_maps, right_carrier, right_labels, ranker, families
    )
    right_crossfit = _crossfit(right_rows, adapter)
    all_maps = {**v5_maps, **maps}
    feature_ids = tuple(maps)
    identical_baseline, identical_rows, inventory = _identical_proposal_rows(
        adapter, all_maps, feature_ids, families, labels, ranker
    )
    identical_crossfit = _crossfit(identical_rows, adapter)
    family_results = {
        family: _crossfit(full_rows, adapter, family=family)
        for family in sorted(set(families.values()))
    }
    _atomic_json(evaluation_dir / "full_native_screen.json", {
        "carrier_baseline": full_baseline, "rows": full_rows,
        "crossfitted_all": full_crossfit, "crossfitted_families": family_results,
    })
    _atomic_json(evaluation_dir / "right_native_screen.json", {
        "carrier_baseline": right_baseline, "rows": right_rows,
        "crossfitted_all": right_crossfit,
    })
    _atomic_json(evaluation_dir / "identical_proposal_screen.json", {
        "carrier_baseline": identical_baseline, "rows": identical_rows,
        "crossfitted_all": identical_crossfit,
    })
    _atomic_json(evaluation_dir / "identical_proposal_inventory.json", inventory)
    return {
        "full_native_carrier": full_baseline,
        "full_native_crossfit": full_crossfit,
        "full_native_family_crossfits": family_results,
        "right_native_carrier": right_baseline,
        "right_native_crossfit": right_crossfit,
        "identical_proposal_carrier": identical_baseline,
        "identical_proposal_crossfit": identical_crossfit,
    }


def _report(path: Path, metrics: Mapping[str, Any]) -> None:
    evaluation = metrics["evaluation"]
    full = evaluation["full_native_crossfit"]["budget_mean_recall"]
    baseline = evaluation["full_native_carrier"]["budget_mean_recall"]
    identical = evaluation["identical_proposal_crossfit"]["budget_mean_recall"]
    identical_base = evaluation["identical_proposal_carrier"]["budget_mean_recall"]
    noise = metrics["noise_audit"]
    lines = [
        f"# {metrics['experiment_id']}", "", "## Executive result", "",
        metrics["conclusion"], "", "## Held-out known-positive recall", "",
        "| Estimand | B20 | B40 | B58 |", "| --- | ---: | ---: | ---: |",
        f"| Full carrier | {baseline['20']:.3f} | {baseline['40']:.3f} | {baseline['58']:.3f} |",
        f"| Selected scientific feature lane | {full['20']:.3f} | {full['40']:.3f} | {full['58']:.3f} |",
        f"| Same-proposal carrier | {identical_base['20']:.3f} | {identical_base['40']:.3f} | {identical_base['58']:.3f} |",
        f"| Same-proposal selected feature | {identical['20']:.3f} | {identical['40']:.3f} | {identical['58']:.3f} |",
        "", "## Acquisition deductions", "",
        f"- Left/right global trace correlation: {noise['left_right_global_trace_correlation']:.3f}.",
        f"- Saturated sample fraction: {noise['saturation_fraction']['full']:.5f}.",
        f"- Provisional split: x={metrics['split_x_px']}; every known label is in the right field.",
        "- Frame period is inferred from the filename, not embedded TIFF metadata.",
        "", "## Interpretation guards", "",
        "Noise fits are descriptive. Z-cut classes are hypotheses. Lagged correlation is not causal propagation. Sparse unmatched candidates remain unknown rather than false positives.",
        "", "## Artifacts", "", "See `diagnostics/`, `videos/`, `evaluation/`, and `metrics.json`.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config: ScientificAuditConfig) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(config.resources["cpu_threads"])
    audit = _matching_preflight(config)
    ranker = InnovationRankerConfig.load(config.source_ranker_config)
    partial = Path(str(config.output_dir) + ".partial")
    partial.mkdir(parents=True)
    diagnostics = partial / "diagnostics"
    videos = partial / "videos"
    evaluation_dir = partial / "evaluation"
    for directory in (diagnostics, videos, evaluation_dir):
        directory.mkdir()
    _atomic_json(partial / "preflight.json", audit)
    _atomic_json(partial / "config.resolved.json", config.to_dict())
    progress = partial / "progress.jsonl"
    started = time.time()
    labels = label_core.load_labels(config.labels_tsv)
    source = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    start = int(config.frames["review_start_ui"]) - 1
    stop = int(config.frames["review_end_ui"])
    raw = np.asarray(source[start:stop], dtype=np.float32)
    noise, _ = _noise_stage(config, raw, diagnostics, videos, progress)
    del raw
    carrier = np.asarray(
        np.load(config.carrier_path, mmap_mode="r", allow_pickle=False),
        dtype=np.float32,
    )
    radial_maps, radial_metrics = _radial_stage(
        config, carrier, labels, videos, progress
    )
    propagation_maps, propagation_metrics = _propagation_stage(
        config, carrier, labels, videos, progress
    )
    base_config = FeatureUtilityConfig.load(ranker.feature_root / "config.resolved.json")
    v5_maps, _ = _generate_v5_maps(ranker, labels, base_config, progress)
    carrier_maps = v5_maps["carrier_signed"]
    morphology_maps, morphology_metrics = _morphology_stage(
        config, carrier_maps, labels, diagnostics
    )
    maps = {**radial_maps, **morphology_maps, **propagation_maps}
    feature_ids, families = _feature_inventory(config)
    if tuple(maps) != feature_ids:
        raise RuntimeError("computed feature order differs from frozen inventory")
    evaluation = _evaluate(
        config, maps, families, v5_maps, carrier_maps, labels, ranker,
        evaluation_dir,
    )
    full = evaluation["full_native_crossfit"]["budget_mean_recall"]
    baseline = evaluation["full_native_carrier"]["budget_mean_recall"]
    conclusion = (
        f"Completed acquisition/noise, generative z-cut, radial-information, and "
        f"causal propagation audits across {config.feature_count} features and "
        f"{config.evaluated_lane_count} scored lanes. Leakage-safe full-field "
        f"selection changed recall from {baseline['20']:.3f}/{baseline['40']:.3f}/"
        f"{baseline['58']:.3f} to {full['20']:.3f}/{full['40']:.3f}/{full['58']:.3f} "
        f"at budgets 20/40/58."
    )
    metrics = {
        "schema_version": 1, "experiment_id": config.experiment_id,
        "status": "completed", "feature_count": config.feature_count,
        "evaluated_lane_count": config.evaluated_lane_count,
        "split_x_px": int(config.acquisition["split_x_px"]),
        "noise_audit": noise, "morphology_audit": morphology_metrics,
        "radial_information_audit": radial_metrics,
        "propagation_audit": propagation_metrics,
        "evaluation": evaluation, "conclusion": conclusion,
        "precision_contract": "Sparse unmatched candidates are unknown; candidate burden is not precision.",
        "elapsed_seconds": time.time() - started,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    _atomic_json(partial / "metrics.json", metrics)
    _report(partial / "REPORT.md", metrics)
    output_bytes = sum(path.stat().st_size for path in partial.rglob("*") if path.is_file())
    if output_bytes > int(config.resources["max_output_mib"]) * 2**20:
        raise RuntimeError("completed audit exceeds output cap")
    _atomic_json(partial / "run_state.json", {
        "status": "completed", "elapsed_seconds": metrics["elapsed_seconds"],
        "max_rss_mib": metrics["max_rss_mib"], "output_bytes": output_bytes,
    })
    partial.replace(config.output_dir)
    return metrics
