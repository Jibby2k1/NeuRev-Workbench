"""Guarded real-data visuals for four stochastic-Parzen state architectures."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import time
from typing import Any

import numpy as np
import tifffile

from neurobench.algorithms.hierarchical_parzen_ica import ParzenDictionaryConfig
from neurobench.experiments.hierarchical_parzen_ica.architecture_config import (
    ArchitectureVisualConfig,
)
from neurobench.experiments.hierarchical_parzen_ica.architecture_lanes import (
    ARCHITECTURE_IDS,
    AffineICAReconstruction,
    InnovationCalibration,
    calibrate_reference_parzen_innovation,
    iter_architecture_frames,
    quiet_median_background,
)
from neurobench.experiments.hierarchical_parzen_ica.stage1 import fit_stage1_lane


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _memory_available_mib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def _process_snapshot() -> list[str]:
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid,comm,%cpu,%mem,rss", "--sort=-rss"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return completed.stdout.splitlines()[:11]


def _gpu_snapshot() -> list[str]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return completed.stdout.splitlines()


def preflight(config: ArchitectureVisualConfig) -> dict[str, Any]:
    """Read-only validation of exact data, output, and resource bounds."""
    source = config.source_video
    source_exists = source.is_file()
    shape = None
    dtype = None
    finite_sample = False
    resolved_frames = 0
    frame_bounds_valid = False
    if source_exists:
        video = np.load(source, mmap_mode="r")
        shape = list(video.shape)
        dtype = str(video.dtype)
        frames = config.frames
        start = int(frames["review_start_ui"]) - 1
        end = int(frames["review_end_ui"])
        quiet_end = int(frames["quiet_end_ui"])
        frame_bounds_valid = (
            video.ndim == 3
            and 0 <= start < quiet_end <= end <= len(video)
        )
        if frame_bounds_valid:
            resolved_frames = end - start
            stride = max(1, resolved_frames // 20)
            finite_sample = bool(
                np.isfinite(video[start:end:stride, ::16, ::16]).all()
            )
    stat = os.statvfs(config.output_dir.parent)
    free_disk_mib = stat.f_bavail * stat.f_frsize / (1024.0**2)
    height = 0 if not shape else int(shape[1])
    width = 0 if not shape else int(shape[2])
    output_frames = max(0, resolved_frames - 1)
    tiff_count = 1 + 2 * len(ARCHITECTURE_IDS)
    uncompressed_output_mib = (
        tiff_count * output_frames * height * width * 2 / (1024.0**2)
    )
    estimated_peak_ram_mib = (
        3584.0
        + resolved_frames * height * width * 2 / (1024.0**2)
        + 512.0
    )
    gates = {
        "source_exists": source_exists,
        "source_is_npy": source.suffix.lower() == ".npy",
        "frame_bounds_valid": frame_bounds_valid,
        "finite_sample": finite_sample,
        "output_absent": not config.output_dir.exists(),
        "partial_output_absent": not Path(
            str(config.output_dir) + ".partial"
        ).exists(),
        "ram_cap_sufficient": estimated_peak_ram_mib
        <= float(config.resources["max_ram_mib"]),
        "available_ram_sufficient": (
            _memory_available_mib() is None
            or estimated_peak_ram_mib <= float(_memory_available_mib())
        ),
        "disk_headroom_sufficient": free_disk_mib
        >= float(config.resources["min_free_disk_mib"]),
        "output_cap_sufficient": uncompressed_output_mib
        <= float(config.resources["max_output_mib"]),
        "cpu_only": config.resources["device"] == "cpu",
    }
    return {
        "schema_version": 1,
        "kind": "read_only_stage1_architecture_visual_preflight",
        "source_video": str(source),
        "source_shape": shape,
        "source_dtype": dtype,
        "review_interval_ui_inclusive": [
            int(config.frames["review_start_ui"]),
            int(config.frames["review_end_ui"]),
        ],
        "quiet_interval_ui_inclusive": [
            int(config.frames["quiet_start_ui"]),
            int(config.frames["quiet_end_ui"]),
        ],
        "resolved_source_frames": resolved_frames,
        "resolved_output_frames": output_frames,
        "architecture_count": len(ARCHITECTURE_IDS),
        "tiff_count": tiff_count,
        "estimated_peak_ram_mib": estimated_peak_ram_mib,
        "available_ram_mib": _memory_available_mib(),
        "max_ram_mib": int(config.resources["max_ram_mib"]),
        "uncompressed_output_mib": uncompressed_output_mib,
        "max_output_mib": int(config.resources["max_output_mib"]),
        "free_disk_mib": free_disk_mib,
        "gpu_snapshot_informational_cpu_run": _gpu_snapshot(),
        "active_process_snapshot": _process_snapshot(),
        "gates": gates,
        "ready": all(gates.values()),
        "labels_used": False,
        "stage2_run": False,
    }


def _max_rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _output_mib(root: Path) -> float:
    return sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    ) / (1024.0**2)


def _fit_raw_stochastic(
    calibration: np.ndarray,
    config: ArchitectureVisualConfig,
) -> tuple[AffineICAReconstruction, dict[str, Any]]:
    settings = config.stochastic
    dictionary = ParzenDictionaryConfig(
        **{
            key: (
                int(value)
                if key
                in {"maximum_centers", "warmup_samples", "seed"}
                else float(value)
                if key
                not in {"replacement_policy"}
                else str(value)
            )
            for key, value in settings["dictionary"].items()
        }
    )
    optimizer = {
        "learning_rate": float(settings["optimizer"]["learning_rate"]),
        "gradient_clip": float(settings["optimizer"]["gradient_clip"]),
        "maximum_angle_update_degrees": float(
            settings["optimizer"]["maximum_angle_update_degrees"]
        ),
        "batch_size": int(settings["optimizer"]["batch_size"]),
        "maximum_iterations": int(
            settings["optimizer"]["maximum_iterations"]
        ),
        "tolerance": float(settings["optimizer"]["tolerance"]),
    }
    safety = {
        key: (
            bool(value)
            if key == "require_convergence_for_learned"
            else str(value)
            if key == "unsafe_policy"
            else float(value)
        )
        for key, value in settings["safety"].items()
    }
    started = time.perf_counter()
    lane = fit_stage1_lane(
        calibration,
        "stochastic_parzen_score_pairwise",
        calibration_frame_count=len(calibration),
        fit_sample_pixels=int(settings["fit_sample_pixels"]),
        sample_seed=int(settings["sample_seed"]),
        covariance_mode=str(settings["covariance_mode"]),
        eigenvalue_floor_ratio=float(settings["eigenvalue_floor_ratio"]),
        condition_number_max=float(settings["condition_number_max"]),
        alpha_min=float(settings["alpha_min"]),
        alpha_max=float(settings["alpha_max"]),
        subtraction_mode="exact",
        stochastic_dictionary=dictionary,
        stochastic_fit=optimizer,
        staticness={
            "minimum_confidence_margin": float(
                settings["minimum_confidence_margin"]
            )
        },
        safety=safety,
    )
    elapsed = time.perf_counter() - started
    anchoring = lane.diagnostics["safety"]["reference_anchoring"]
    learned_fraction = (
        None if anchoring is None else anchoring["accepted_learned_fraction"]
    )
    if (
        lane.result.background_component is None
        or lane.result.classification_status != "resolved"
        or lane.diagnostics["safety"]["status"] != "accepted"
        or learned_fraction != 1.0
    ):
        raise RuntimeError(
            "raw stochastic fit did not pass resolved, accepted, "
            "full-learned-fraction gates"
        )
    feedback = lane.diagnostics["safety"]["feedback"]
    coefficients = AffineICAReconstruction.from_feedback(feedback)
    record = {
        "fit_seconds": elapsed,
        "alpha_gain": lane.alpha_gain,
        "classification_status": lane.result.classification_status,
        "background_component": lane.result.background_component,
        "staticness_margin": lane.result.confidence,
        "whitening_identifiable": lane.whitening.identifiable,
        "whitening_condition_number": lane.whitening.condition_number,
        "optimizer_converged": lane.demixing_fit.converged,
        "optimizer_iterations": lane.demixing_fit.iterations,
        "optimizer_updates": lane.demixing_fit.update_count,
        "demixing": lane.demixing_fit.demixing,
        "safety": lane.diagnostics["safety"],
        "affine_reconstruction": {
            "previous_coefficient": coefficients.previous_coefficient,
            "current_coefficient": coefficients.current_coefficient,
            "offset": coefficients.offset,
        },
        "labels_used": False,
    }
    del lane
    gc.collect()
    return coefficients, record


def _sampled_innovation(
    calibration: InnovationCalibration,
    row_stride: int,
    column_stride: int,
) -> InnovationCalibration:
    return InnovationCalibration(
        quiet_background=calibration.quiet_background[
            ::row_stride, ::column_stride
        ],
        correction_bias=calibration.correction_bias[
            ::row_stride, ::column_stride
        ],
        correction_limit=calibration.correction_limit,
        reference_refresh=calibration.reference_refresh,
        reference_half_life_seconds=(
            calibration.reference_half_life_seconds
        ),
        correction_fraction=calibration.correction_fraction,
        correction_clip_mad=calibration.correction_clip_mad,
        quiet_correction_mad=calibration.quiet_correction_mad,
    )


def _display_scales(
    frames: np.ndarray,
    coefficients: AffineICAReconstruction,
    quiet_background: np.ndarray,
    innovation: InnovationCalibration,
    config: ArchitectureVisualConfig,
) -> dict[str, Any]:
    visual = config.visualization
    frame_stride = int(visual["sample_frame_stride"])
    row_stride = int(visual["sample_row_stride"])
    column_stride = int(visual["sample_column_stride"])
    sampled_frames = frames[:, ::row_stride, ::column_stride]
    sampled_quiet = quiet_background[::row_stride, ::column_stride]
    sampled_calibration = _sampled_innovation(
        innovation,
        row_stride,
        column_stride,
    )
    positive_parts = [
        np.asarray(
            sampled_frames[1::frame_stride],
            dtype=np.float32,
        ).reshape(-1)
    ]
    dynamics_parts: list[np.ndarray] = []
    for architecture_id in ARCHITECTURE_IDS:
        background_parts = []
        dynamic_parts = []
        for item in iter_architecture_frames(
            sampled_frames,
            architecture_id,
            coefficients,
            quiet_background=sampled_quiet,
            innovation=sampled_calibration,
        ):
            if (item.output_index_zero - 1) % frame_stride == 0:
                background_parts.append(item.background.reshape(-1))
                dynamic_parts.append(item.dynamics_noise.reshape(-1))
        positive_parts.append(np.concatenate(background_parts))
        dynamics_parts.append(np.concatenate(dynamic_parts))
    positive = np.concatenate(positive_parts)
    dynamics = np.concatenate(dynamics_parts)
    lower = float(
        np.percentile(
            positive,
            float(visual["positive_lower_percentile"]),
        )
    )
    upper = float(
        np.percentile(
            positive,
            float(visual["positive_upper_percentile"]),
        )
    )
    magnitude = max(
        float(
            np.percentile(
                np.abs(dynamics),
                float(visual["dynamics_absolute_percentile"]),
            )
        ),
        1e-6,
    )
    if upper <= lower:
        raise RuntimeError("invalid shared background display scale")
    return {
        "background": {
            "mode": "shared_linear_percentile",
            "source_limits": [lower, upper],
            "display_limits": [0, 65535],
        },
        "dynamics_noise": {
            "mode": "shared_symmetric_absolute_percentile",
            "source_limits": [-magnitude, magnitude],
            "display_limits": [0, 65535],
            "display_zero": 32768,
            "interpretation": (
                "negative=below-mid-gray; zero=mid-gray; "
                "positive=above-mid-gray"
            ),
        },
        "sampling": {
            "frame_stride": frame_stride,
            "row_stride": row_stride,
            "column_stride": column_stride,
        },
        "fixed_across_frames": True,
        "shared_across_architectures": True,
        "display_only": True,
    }


def _encode_linear(
    frame: np.ndarray,
    limits: list[float],
) -> np.ndarray:
    lower, upper = limits
    normalized = (np.asarray(frame, dtype=np.float64) - lower) / (
        upper - lower
    )
    return np.rint(np.clip(normalized, 0.0, 1.0) * 65535.0).astype(
        np.uint16
    )


def _encode_signed(frame: np.ndarray, magnitude: float) -> np.ndarray:
    normalized = np.clip(
        np.asarray(frame, dtype=np.float64) / magnitude,
        -1.0,
        1.0,
    )
    return np.rint((normalized + 1.0) * 32767.5).astype(np.uint16)


def _verify_tiff(
    path: Path,
    frame_count: int,
    spatial_shape: tuple[int, int],
) -> None:
    with tifffile.TiffFile(path) as tiff:
        if (
            len(tiff.pages) != frame_count
            or tiff.pages[0].shape != spatial_shape
            or tiff.pages[-1].shape != spatial_shape
            or tiff.pages[0].dtype != np.dtype(np.uint16)
        ):
            raise RuntimeError(f"TIFF verification failed: {path}")
        json.loads(tiff.pages[0].description)


def _write_input_tiff(
    path: Path,
    frames: np.ndarray,
    description: dict[str, Any],
    limits: list[float],
    compression: str,
) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with tifffile.TiffWriter(temporary, bigtiff=True) as writer:
        for index, frame in enumerate(frames[1:]):
            writer.write(
                _encode_linear(frame, limits),
                photometric="minisblack",
                compression=compression,
                metadata=None,
                description=(
                    json.dumps(description, sort_keys=True)
                    if index == 0
                    else None
                ),
            )
    temporary.replace(path)
    _verify_tiff(path, len(frames) - 1, tuple(frames.shape[1:]))


def _write_architecture_tiffs(
    destination: Path,
    architecture_id: str,
    frames: np.ndarray,
    coefficients: AffineICAReconstruction,
    quiet_background: np.ndarray,
    innovation: InnovationCalibration,
    descriptions: dict[str, dict[str, Any]],
    scales: dict[str, Any],
    compression: str,
    quiet_frame_count: int,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    background_path = destination / "background.tif"
    dynamics_path = destination / "dynamics_noise.tif"
    background_temporary = background_path.with_suffix(".tif.partial")
    dynamics_temporary = dynamics_path.with_suffix(".tif.partial")
    background_limits = scales["background"]["source_limits"]
    dynamics_magnitude = scales["dynamics_noise"]["source_limits"][1]
    total_pixels = 0
    negative_background = 0
    closure_max = 0.0
    quiet_squared = 0.0
    quiet_count = 0
    event_squared = 0.0
    event_count = 0
    first_mean = None
    first_std = None
    last_mean = None
    last_std = None
    background_clipped = 0
    dynamics_clipped = 0
    with tifffile.TiffWriter(
        background_temporary,
        bigtiff=True,
    ) as background_writer, tifffile.TiffWriter(
        dynamics_temporary,
        bigtiff=True,
    ) as dynamics_writer:
        for page_index, item in enumerate(
            iter_architecture_frames(
                frames,
                architecture_id,
                coefficients,
                quiet_background=quiet_background,
                innovation=innovation,
            )
        ):
            current = np.asarray(
                frames[item.output_index_zero],
                dtype=np.float32,
            )
            closure_max = max(
                closure_max,
                float(
                    np.max(
                        np.abs(
                            current
                            - item.background
                            - item.dynamics_noise
                        )
                    )
                ),
            )
            count = item.background.size
            total_pixels += count
            negative_background += int(np.count_nonzero(item.background < 0))
            background_clipped += int(
                np.count_nonzero(
                    (item.background < background_limits[0])
                    | (item.background > background_limits[1])
                )
            )
            dynamics_clipped += int(
                np.count_nonzero(
                    np.abs(item.dynamics_noise) > dynamics_magnitude
                )
            )
            if first_mean is None:
                first_mean = float(np.mean(item.background))
                first_std = float(np.std(item.background))
            last_mean = float(np.mean(item.background))
            last_std = float(np.std(item.background))
            squared = float(
                np.sum(
                    np.asarray(item.dynamics_noise, dtype=np.float64) ** 2
                )
            )
            if item.output_index_zero < quiet_frame_count:
                quiet_squared += squared
                quiet_count += count
            else:
                event_squared += squared
                event_count += count
            background_writer.write(
                _encode_linear(item.background, background_limits),
                photometric="minisblack",
                compression=compression,
                metadata=None,
                description=(
                    json.dumps(
                        descriptions["background"],
                        sort_keys=True,
                    )
                    if page_index == 0
                    else None
                ),
            )
            dynamics_writer.write(
                _encode_signed(item.dynamics_noise, dynamics_magnitude),
                photometric="minisblack",
                compression=compression,
                metadata=None,
                description=(
                    json.dumps(
                        descriptions["dynamics_noise"],
                        sort_keys=True,
                    )
                    if page_index == 0
                    else None
                ),
            )
    background_temporary.replace(background_path)
    dynamics_temporary.replace(dynamics_path)
    _verify_tiff(
        background_path,
        len(frames) - 1,
        tuple(frames.shape[1:]),
    )
    _verify_tiff(
        dynamics_path,
        len(frames) - 1,
        tuple(frames.shape[1:]),
    )
    return {
        "architecture_id": architecture_id,
        "background_tiff": str(background_path.name),
        "dynamics_noise_tiff": str(dynamics_path.name),
        "closure_max_absolute": closure_max,
        "background_negative_fraction": negative_background / total_pixels,
        "background_display_clip_fraction": background_clipped / total_pixels,
        "dynamics_display_clip_fraction": dynamics_clipped / total_pixels,
        "first_background_mean": first_mean,
        "last_background_mean": last_mean,
        "first_background_spatial_std": first_std,
        "last_background_spatial_std": last_std,
        "background_spatial_std_ratio_last_to_first": (
            None
            if first_std is None or first_std <= 0
            else last_std / first_std
        ),
        "quiet_dynamics_rms": (
            None if quiet_count == 0 else (quiet_squared / quiet_count) ** 0.5
        ),
        "post_quiet_dynamics_rms": (
            None if event_count == 0 else (event_squared / event_count) ** 0.5
        ),
    }


def _report(
    manifest: dict[str, Any],
    records: dict[str, Any],
) -> str:
    coefficients = manifest["raw_stochastic_fit"]["affine_reconstruction"]
    lines = [
        "# Spon Ca Stage-1 stochastic architecture visual comparison",
        "",
        "This run fits one raw stochastic-Parzen ICA demixer on the quiet prefix "
        "and changes only the inference architecture. It is a visual and "
        "rollout diagnostic, not a neuron-detection benchmark.",
        "",
        "## Shared fitted ICA reconstruction",
        "",
        "```text",
        (
            "P(t) = "
            f"{coefficients['previous_coefficient']:.9g} * previous + "
            f"{coefficients['current_coefficient']:.9g} * current + "
            f"{coefficients['offset']:.9g}"
        ),
        "```",
        "",
        "## Architectures",
        "",
        "1. `teacher_forced_stochastic`: previous is the real prior frame.",
        "2. `raw_stochastic_recurrence`: previous is the prior estimated background.",
        "3. `quiet_fixed_point_recurrence`: recurrence is centered on the frozen "
        "per-pixel quiet median, eliminating free offset accumulation.",
        "4. `reference_parzen_innovation`: a slow fixed-point EMA background plus "
        "a quiet-zeroed, clipped, fractional Parzen correction.",
        "",
        "## Rollout summary",
        "",
        (
            "| Architecture | Last/first background spatial SD | "
            "Negative background | Quiet dynamics RMS | Post-quiet dynamics RMS |"
        ),
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for architecture_id in ARCHITECTURE_IDS:
        row = records[architecture_id]
        lines.append(
            f"| {architecture_id} | "
            f"{row['background_spatial_std_ratio_last_to_first']:.6g} | "
            f"{row['background_negative_fraction']:.6g} | "
            f"{row['quiet_dynamics_rms']:.6g} | "
            f"{row['post_quiet_dynamics_rms']:.6g} |"
        )
    lines.extend(
        [
            "",
            "Every architecture directory contains `background.tif` and "
            "`dynamics_noise.tif`. TIFF scales are fixed across time and shared "
            "across architectures. Mid-gray is zero in dynamics/noise.",
            "",
            "Stage 2 and neuron detection were not run. Unmatched real activity "
            "therefore remains visually interpretable but quantitatively unknown.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config: ArchitectureVisualConfig) -> dict[str, Any]:
    """Run the four collision-safe CPU lanes and write comparable TIFFs."""
    check = preflight(config)
    if not check["ready"]:
        failed = [key for key, value in check["gates"].items() if not value]
        raise RuntimeError(f"preflight failed: {', '.join(failed)}")
    final_root = config.output_dir
    partial_root = Path(str(final_root) + ".partial")
    partial_root.mkdir(parents=True, exist_ok=False)
    started = time.time()

    def enforce(phase: str) -> dict[str, Any]:
        rss = _max_rss_mib()
        output = _output_mib(partial_root)
        if rss > float(config.resources["max_ram_mib"]):
            raise MemoryError(f"RAM cap exceeded during {phase}: {rss:.1f} MiB")
        if output > float(config.resources["max_output_mib"]):
            raise OSError(
                f"output cap exceeded during {phase}: {output:.1f} MiB"
            )
        return {
            "phase": phase,
            "max_rss_mib": rss,
            "output_mib": output,
        }

    def heartbeat(payload: dict[str, Any]) -> None:
        with (partial_root / "progress.jsonl").open(
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(
                json.dumps(
                    payload,
                    sort_keys=True,
                    default=_json_default,
                )
                + "\n"
            )
            stream.flush()
        print(json.dumps(payload, sort_keys=True), flush=True)

    try:
        _atomic_json(partial_root / "preflight.json", check)
        _atomic_json(
            partial_root / "resolved_config.json",
            config.to_dict(),
        )
        _atomic_json(
            partial_root / "run_state.json",
            {
                "status": "running",
                "started_unix": started,
                "completed_architectures": 0,
                "total_architectures": len(ARCHITECTURE_IDS),
            },
        )
        heartbeat(enforce("initialized"))
        video = np.load(config.source_video, mmap_mode="r")
        start = int(config.frames["review_start_ui"]) - 1
        end = int(config.frames["review_end_ui"])
        quiet_count = (
            int(config.frames["quiet_end_ui"])
            - int(config.frames["quiet_start_ui"])
            + 1
        )
        frames = video[start:end]
        coefficients, fit_record = _fit_raw_stochastic(
            frames[:quiet_count],
            config,
        )
        heartbeat(
            {
                **enforce("stochastic_fit_completed"),
                "optimizer_converged": fit_record["optimizer_converged"],
                "accepted_learned_fraction": fit_record["safety"][
                    "reference_anchoring"
                ]["accepted_learned_fraction"],
            }
        )
        quiet_background = quiet_median_background(frames, quiet_count)
        architecture = config.architectures
        innovation = calibrate_reference_parzen_innovation(
            frames,
            quiet_count,
            coefficients,
            frame_period_ms=float(config.frames["frame_period_ms"]),
            reference_half_life_seconds=float(
                architecture["reference_half_life_seconds"]
            ),
            correction_fraction=float(
                architecture["correction_fraction"]
            ),
            correction_clip_mad=float(
                architecture["correction_clip_mad"]
            ),
        )
        scales = _display_scales(
            frames,
            coefficients,
            quiet_background,
            innovation,
            config,
        )
        manifest = {
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "kind": "real_spon_stage1_stochastic_architecture_visuals",
            "source_video": str(config.source_video),
            "source_shape": list(video.shape),
            "source_dtype": str(video.dtype),
            "review_interval_ui_inclusive": [
                int(config.frames["review_start_ui"]),
                int(config.frames["review_end_ui"]),
            ],
            "quiet_interval_ui_inclusive": [
                int(config.frames["quiet_start_ui"]),
                int(config.frames["quiet_end_ui"]),
            ],
            "output_interval_ui_inclusive": [
                int(config.frames["review_start_ui"]) + 1,
                int(config.frames["review_end_ui"]),
            ],
            "output_shape": [len(frames) - 1, *frames.shape[1:]],
            "frame_period_ms": float(config.frames["frame_period_ms"]),
            "architectures": list(ARCHITECTURE_IDS),
            "raw_stochastic_fit": fit_record,
            "innovation_regularization": {
                "quiet_background": "per_pixel_quiet_median",
                "reference_half_life_seconds": (
                    innovation.reference_half_life_seconds
                ),
                "reference_refresh": innovation.reference_refresh,
                "correction_fraction": innovation.correction_fraction,
                "correction_clip_mad": innovation.correction_clip_mad,
                "quiet_correction_mad": innovation.quiet_correction_mad,
                "correction_limit": innovation.correction_limit,
                "correction_bias_source": "quiet_prefix_only",
            },
            "display_normalization": scales,
            "labels_used": False,
            "stage2_run": False,
            "scientific_status": (
                "architecture_visual_diagnostic_not_detection_evidence"
            ),
        }
        _atomic_json(partial_root / "manifest.json", manifest)
        base_description = {
            "schema_version": 1,
            "display_only": True,
            "axes": "TYX",
            "source_ui_frames_inclusive": (
                manifest["output_interval_ui_inclusive"]
            ),
            "frame_period_ms": manifest["frame_period_ms"],
            "viewer_dtype": "uint16",
            "normalization_fixed_across_frames": True,
            "normalization_shared_across_architectures": True,
        }
        _write_input_tiff(
            partial_root / "input.tif",
            frames,
            {
                **base_description,
                "role": "aligned_observation",
                "normalization": scales["background"],
            },
            scales["background"]["source_limits"],
            str(config.visualization["compression"]),
        )
        heartbeat(enforce("input_written"))
        records: dict[str, Any] = {}
        for index, architecture_id in enumerate(ARCHITECTURE_IDS, 1):
            heartbeat(
                {
                    **enforce("architecture_started"),
                    "architecture_id": architecture_id,
                    "completed_architectures": index - 1,
                    "total_architectures": len(ARCHITECTURE_IDS),
                }
            )
            lane_started = time.perf_counter()
            descriptions = {
                "background": {
                    **base_description,
                    "architecture_id": architecture_id,
                    "role": "background_estimation",
                    "normalization": scales["background"],
                },
                "dynamics_noise": {
                    **base_description,
                    "architecture_id": architecture_id,
                    "role": "signed_dynamics_plus_noise",
                    "normalization": scales["dynamics_noise"],
                },
            }
            record = _write_architecture_tiffs(
                partial_root / architecture_id,
                architecture_id,
                frames,
                coefficients,
                quiet_background,
                innovation,
                descriptions,
                scales,
                str(config.visualization["compression"]),
                quiet_count,
            )
            record["elapsed_seconds"] = time.perf_counter() - lane_started
            records[architecture_id] = record
            _atomic_json(
                partial_root / "architecture_records.json",
                records,
            )
            _atomic_json(
                partial_root / "run_state.json",
                {
                    "status": "running",
                    "started_unix": started,
                    "completed_architectures": index,
                    "total_architectures": len(ARCHITECTURE_IDS),
                    "last_architecture": architecture_id,
                    "max_rss_mib": _max_rss_mib(),
                    "output_mib": _output_mib(partial_root),
                },
            )
            heartbeat(
                {
                    **enforce("architecture_completed"),
                    "architecture_id": architecture_id,
                    "completed_architectures": index,
                    "total_architectures": len(ARCHITECTURE_IDS),
                    "elapsed_seconds": record["elapsed_seconds"],
                }
            )
        (partial_root / "REPORT.md").write_text(
            _report(manifest, records),
            encoding="utf-8",
        )
        completed = time.time()
        _atomic_json(
            partial_root / "run_state.json",
            {
                "status": "completed",
                "started_unix": started,
                "completed_unix": completed,
                "elapsed_seconds": completed - started,
                "completed_architectures": len(ARCHITECTURE_IDS),
                "total_architectures": len(ARCHITECTURE_IDS),
                "max_rss_mib": _max_rss_mib(),
                "output_mib": _output_mib(partial_root),
            },
        )
        partial_root.replace(final_root)
        return {
            "status": "completed",
            "output_dir": str(final_root),
            "architecture_count": len(ARCHITECTURE_IDS),
            "tiff_count": 1 + 2 * len(ARCHITECTURE_IDS),
            "elapsed_seconds": completed - started,
            "max_rss_mib": _max_rss_mib(),
            "output_mib": _output_mib(final_root),
            "records": records,
        }
    except Exception as exc:
        _atomic_json(
            partial_root / "run_state.json",
            {
                "status": "failed",
                "started_unix": started,
                "failed_unix": time.time(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "max_rss_mib": _max_rss_mib(),
                "output_mib": _output_mib(partial_root),
            },
        )
        raise
