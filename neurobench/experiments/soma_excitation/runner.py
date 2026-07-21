"""Bounded orchestration for the dark-soma excitation experiment."""

from __future__ import annotations

import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

import numpy as np

from neurobench.data.video import iter_video_chunks

from .config import SomaExcitationConfig
from .preflight import MIB, ResourceBudgetError, build_soma_excitation_preflight


SCHEMA_VERSION = 1
_PROC_SELF_STATUS = Path("/proc/self/status")
_ARM_SPECS: tuple[tuple[str, str], ...] = (
    ("adaptive_full_fov_128", "adaptive max pooling over the complete field of view to 128x128"),
    ("fixed_native_pool4", "fixed 4x4 max pooling with native aspect ratio and bottom/right trim"),
)


class _PooledProvider:
    """Normalize and pool one memory-mapped source frame on demand."""

    def __init__(self, source: np.ndarray, bounds: Any, arm_name: str) -> None:
        self.source = source
        self.bounds = bounds
        self.arm_name = arm_name

    def __len__(self) -> int:
        return int(self.source.shape[0])

    def __getitem__(self, index: int) -> np.ndarray:
        from .transfer import adaptive_max_pool_frame, fixed_max_pool_frame, normalize_frame

        frame = normalize_frame(self.source[index], self.bounds)
        if self.arm_name == "adaptive_full_fov_128":
            return adaptive_max_pool_frame(frame, (128, 128))
        if self.arm_name == "fixed_native_pool4":
            return fixed_max_pool_frame(frame, pool_size=4)
        raise KeyError(f"Unknown spatial arm: {self.arm_name}")


def run_soma_excitation_experiment(
    config_or_path: SomaExcitationConfig | Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Run the CPU-only detector and optional frozen TemporalCNN transfer.

    The output directory must not already exist.  Preflight is read-only and
    completes before the directory is created, so an output collision or an
    unsafe resource plan cannot partially overwrite a prior experiment.
    """

    config = _coerce_config(config_or_path)
    preflight = build_soma_excitation_preflight(config, allow_existing_output=False)
    if not preflight.get("ready"):
        raise RuntimeError("Soma-excitation preflight did not return a ready plan.")
    _configure_cpu_limits(config.resources.cpu_threads)

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started_at = _utc_now()
    max_ram_bytes = config.resources.max_ram_mib * MIB
    memory_observations: list[dict[str, Any]] = []
    _record_memory_observation(
        memory_observations,
        stage="start",
        max_ram_bytes=max_ram_bytes,
        enforce=False,
    )
    started_state = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "experiment_id": config.experiment_id,
        "device": "cpu",
        "started_at": started_at,
        "resources": _memory_resource_fields(memory_observations, max_ram_bytes),
    }
    _atomic_json(output / "resolved_config.json", config.to_dict())
    _atomic_json(output / "preflight.json", preflight)
    _atomic_json(output / "run_state.json", started_state)

    try:
        from .detector import run_streamed_detector

        bounds = preflight["frame_bounds"]
        detector = run_streamed_detector(
            config.source_video,
            control_start=int(bounds["control_start_frame_zero"]),
            control_stop=int(bounds["control_stop_frame_zero_exclusive"]),
            score_start=int(bounds["score_start_frame_zero"]),
            score_stop=int(bounds["score_stop_frame_zero_exclusive"]),
            chunk_frames=int(preflight["resources"]["resolved_chunk_frames"]),
            cfar=config.cfar,
            zone_config=config.dark_zones.to_zone_config(),
        )
        _record_memory_observation(
            memory_observations,
            stage="after_detector",
            max_ram_bytes=max_ram_bytes,
            enforce=True,
        )
        _atomic_npz(output / "detector_arrays.npz", detector.array_payload())
        _atomic_json(output / "detector_summary.json", detector.summary)
        _atomic_json(
            output / "dark_zones.json",
            {
                "schema_version": SCHEMA_VERSION,
                "semantics": "provisional dark-core anatomy; positive excitation is measured in the surrounding annulus",
                "zone_count": int(detector.summary["dark_zones"]["count"]),
                "zones": detector.summary["dark_zones"]["zones"],
                "union_core_pixel_count": int(np.count_nonzero(detector.core_mask)),
                "union_ring_pixel_count": int(np.count_nonzero(detector.ring_mask)),
            },
        )

        review_indices = _select_review_indices(detector)
        review_frames = _read_selected_frames(config.source_video, review_indices)
        _atomic_npz(
            output / "review_frames.npz",
            {
                "raw_frames": review_frames,
                "source_indices": np.asarray(review_indices, dtype=np.int64),
                "ui_frames": np.asarray(review_indices, dtype=np.int64) + 1,
            },
        )

        transfer = _run_transfer(config, preflight, detector)
        _record_memory_observation(
            memory_observations,
            stage="after_transfer",
            max_ram_bytes=max_ram_bytes,
            enforce=True,
        )
        _atomic_json(output / "transfer_results.json", transfer)
        _record_memory_observation(
            memory_observations,
            stage="before_completion",
            max_ram_bytes=max_ram_bytes,
            enforce=True,
        )
        summary = _compact_summary(
            config,
            preflight,
            detector,
            transfer,
            review_indices,
            memory_observations,
        )
        summary.update({"started_at": started_at, "completed_at": _utc_now()})
        actual_bytes = _finalize_success(output, summary, config.resources.max_output_mib * MIB)
        summary["resources"]["actual_output_bytes"] = actual_bytes
        return summary
    except BaseException as exc:
        failed = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "experiment_id": config.experiment_id,
            "device": "cpu",
            "started_at": started_at,
            "failed_at": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "output_bytes": _directory_size(output),
            "resources": _memory_resource_fields(memory_observations, max_ram_bytes),
        }
        try:
            _atomic_json(output / "run_state.json", failed)
        except Exception:
            pass
        raise


def _run_transfer(config: SomaExcitationConfig, preflight: Mapping[str, Any], detector: Any) -> dict[str, Any]:
    interpretation = {
        "status": "exploratory_out_of_distribution",
        "frozen_models": True,
        "manual_ground_truth_available": False,
        "claim_limit": "These scores measure frozen transfer versus persistence; they do not establish soma-event accuracy.",
    }
    if not config.dynamics_checkpoints:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "not_run",
            "reason": "no dynamics checkpoints configured",
            "interpretation": interpretation,
            "spatial_arms": {},
        }
    source_path = Path(config.source_video)
    if source_path.suffix.lower() != ".npy":
        raise ValueError("Dynamics transfer requires a .npy source so every frame remains memory-mapped.")
    for checkpoint in config.dynamics_checkpoints:
        if checkpoint.horizon_frames is None:
            raise ValueError(
                f"Checkpoint {checkpoint.path} requires explicit horizon_frames in the experiment config."
            )

    from .transfer import (
        adaptive_max_pool_frame,
        evaluate_temporal_cnn_transfer,
        fit_robust_normalization_bounds,
        fixed_max_pool_frame,
        load_temporal_cnn_checkpoint,
    )

    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if source.ndim != 3:
        raise ValueError(f"Dynamics transfer requires a grayscale (frames, height, width) NPY; got {source.shape}.")
    frame_bounds = preflight["frame_bounds"]
    control_indices = range(
        int(frame_bounds["control_start_frame_zero"]),
        int(frame_bounds["control_stop_frame_zero_exclusive"]),
    )
    normalization = fit_robust_normalization_bounds(source, control_indices)
    targets = tuple(
        range(
            int(frame_bounds["score_start_frame_zero"]),
            int(frame_bounds["score_stop_frame_zero_exclusive"]),
        )
    )
    arms: dict[str, Any] = {
        name: {"description": description, "models": []} for name, description in _ARM_SPECS
    }
    mask_poolers: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "adaptive_full_fov_128": lambda mask: adaptive_max_pool_frame(mask.astype(np.float32), (128, 128)) > 0,
        "fixed_native_pool4": lambda mask: fixed_max_pool_frame(mask.astype(np.float32), pool_size=4) > 0,
    }
    for checkpoint in config.dynamics_checkpoints:
        loaded = load_temporal_cnn_checkpoint(
            checkpoint.path,
            model_id=checkpoint.model_id,
            horizon_frames=checkpoint.horizon_frames,
            cpu_threads=config.resources.cpu_threads,
        )
        for arm_name, _description in _ARM_SPECS:
            provider = _PooledProvider(source, normalization, arm_name)
            core_mask = mask_poolers[arm_name](detector.core_mask)
            ring_mask = mask_poolers[arm_name](detector.ring_mask)
            result = evaluate_temporal_cnn_transfer(
                loaded,
                provider,
                targets,
                horizon_frames=checkpoint.horizon_frames,
                frame_rate_hz=config.frame_rate_hz,
                core_mask=core_mask,
                ring_mask=ring_mask,
            )
            result["spatial_arm"] = arm_name
            result["spatial_shape"] = [int(value) for value in provider[targets[0]].shape]
            result["control_frames_scored"] = False
            arms[arm_name]["models"].append(result)
            del provider, core_mask, ring_mask, result
        del loaded
        gc.collect()
    del source
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "interpretation": interpretation,
        "normalization": {
            **normalization.to_dict(),
            "fit_indices_semantics": "control-only, zero-based source frames",
            "event_frames_used_for_fit": False,
        },
        "target_range": {
            "first_source_index": targets[0],
            "last_source_index": targets[-1],
            "target_count": len(targets),
            "control_frames_scored": False,
        },
        "spatial_arms": arms,
    }


def _compact_summary(
    config: SomaExcitationConfig,
    preflight: Mapping[str, Any],
    detector: Any,
    transfer: Mapping[str, Any],
    review_indices: list[int],
    memory_observations: list[dict[str, Any]],
) -> dict[str, Any]:
    arm_metrics: dict[str, list[dict[str, Any]]] = {}
    for name, arm in transfer.get("spatial_arms", {}).items():
        arm_metrics[name] = [_compact_model_metrics(result) for result in arm["models"]]
    residual = detector.summary["metrics"]["residual"]
    positive_signal = detector.summary["metrics"]["positive_residual_signal"]
    cfar_activation = detector.summary["zone_activation"]["residual"]
    signal_activation = detector.summary["zone_activation"]["positive_residual_signal"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "experiment_id": config.experiment_id,
        "interpretation": {
            "primary": "Dark soma cores are treated as background anchors; positive excitation is tested in perisomatic rings.",
            "transfer": "Frozen current-model transfer is exploratory/out-of-distribution.",
            "ground_truth": "No manual ground truth is available; results are signals for review, not accuracy estimates.",
        },
        "frame_bounds": preflight["frame_bounds"],
        "detector": {
            "dark_zone_count": int(detector.summary["dark_zones"]["count"]),
            # Preserve the v1 field as an explicit residual-CFAR alias.
            "activated_zone_count": int(cfar_activation["activated_zone_count"]),
            "cfar_activated_zone_count": int(cfar_activation["activated_zone_count"]),
            "signal_activated_zone_count": int(signal_activation["activated_zone_count"]),
            "positive_residual_signal_global": positive_signal["global"],
            "positive_residual_signal_ring": positive_signal["ring"],
            "positive_residual_signal_ring_enrichment": positive_signal["ring_enrichment"],
            "residual_global": residual["global"],
            "residual_ring": residual["ring"],
            "residual_ring_enrichment": residual["ring_enrichment"],
        },
        "transfer": {
            "status": transfer["status"],
            "checkpoint_count": len(config.dynamics_checkpoints),
            "spatial_arm_metrics": arm_metrics,
        },
        "review": {"selected_source_indices": review_indices, "selected_ui_frames": [v + 1 for v in review_indices]},
        "resources": {
            "device": "cpu",
            "worker_count": 1,
            "cpu_threads": config.resources.cpu_threads,
            "resolved_chunk_frames": preflight["resources"]["resolved_chunk_frames"],
            "max_output_bytes": config.resources.max_output_mib * MIB,
            "actual_output_bytes": 0,
            **_memory_resource_fields(
                memory_observations,
                config.resources.max_ram_mib * MIB,
            ),
        },
        "artifacts": {
            "resolved_config": "resolved_config.json",
            "preflight": "preflight.json",
            "detector_arrays": "detector_arrays.npz",
            "detector_summary": "detector_summary.json",
            "dark_zones": "dark_zones.json",
            "review_frames": "review_frames.npz",
            "transfer_results": "transfer_results.json",
            "experiment_summary": "experiment_summary.json",
            "report": "report.md",
            "run_state": "run_state.json",
        },
    }


def _compact_model_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    masked = metrics["masked"]
    return {
        "model_id": result["checkpoint"].get("model_id"),
        "prediction_mse": metrics.get("prediction_mse"),
        "persistence_mse": metrics.get("persistence_mse"),
        "improvement_over_persistence_mse": metrics.get("improvement_over_persistence_mse"),
        "high_change_improvement": metrics["high_change"].get("improvement_over_persistence_mse"),
        "core_improvement": masked["core"].get("improvement_over_persistence_mse"),
        "ring_improvement": masked["ring"].get("improvement_over_persistence_mse"),
        "positive_change_correlation": metrics.get("positive_change_correlation"),
    }


def _select_review_indices(detector: Any, limit: int = 12) -> list[int]:
    # Direct amplitude is primary because local CFAR can normalize away a
    # spatially broad excitation shared with its background estimate.
    evidence = np.asarray(
        detector.traces["positive_residual_ring_mean"], dtype=np.float64
    )
    score_positions = np.flatnonzero(np.asarray(detector.is_score_frame, dtype=bool))
    ranked = sorted(
        score_positions.tolist(),
        key=lambda position: (-float(evidence[position]), int(detector.frame_indices[position])),
    )[:limit]
    return sorted(int(detector.frame_indices[position]) for position in ranked)


def _read_selected_frames(source: str | Path, indices: list[int]) -> np.ndarray:
    path = Path(source)
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return np.stack([np.asarray(array[index]) for index in indices], axis=0)
    frames: list[np.ndarray] = []
    for index in indices:
        chunks = list(iter_video_chunks(path, chunk_size=1, start_frame=index, end_frame=index + 1))
        if len(chunks) != 1 or chunks[0].frame_count != 1:
            raise RuntimeError(f"Could not read review frame {index} with bounded access.")
        frames.append(np.asarray(chunks[0].data[0]))
    return np.stack(frames, axis=0)


def _finalize_success(output: Path, summary: dict[str, Any], max_output_bytes: int) -> int:
    candidate = _directory_size(output)
    for _ in range(8):
        summary["resources"]["actual_output_bytes"] = candidate
        _atomic_json(output / "experiment_summary.json", summary)
        _atomic_text(output / "report.md", _render_report(summary))
        _atomic_json(
            output / "run_state.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "experiment_id": summary["experiment_id"],
                "device": "cpu",
                "started_at": summary["started_at"],
                "completed_at": summary["completed_at"],
                "output_bytes": candidate,
                "max_output_bytes": max_output_bytes,
                "resources": summary["resources"],
            },
        )
        measured = _directory_size(output)
        if measured > max_output_bytes:
            raise ResourceBudgetError(
                f"Actual output {measured} bytes exceeds the configured cap {max_output_bytes} bytes."
            )
        if measured == candidate:
            return measured
        candidate = measured
    raise RuntimeError("Could not stabilize the self-reported output byte count.")


def _render_report(summary: Mapping[str, Any]) -> str:
    detector = summary["detector"]
    lines = [
        f"# Soma excitation experiment: {summary['experiment_id']}", "", "## Interpretation", "",
        f"- {summary['interpretation']['primary']}",
        f"- {summary['interpretation']['transfer']}",
        f"- {summary['interpretation']['ground_truth']}",
        "- Gamma-CFAR remains one-sided positive-excursion evidence; it does not label the dark soma core as an event.",
        "", "## Detector result", "",
        f"Detected {detector['dark_zone_count']} provisional dark zones; {detector['signal_activated_zone_count']} crossed the direct positive-residual rule and {detector['cfar_activated_zone_count']} crossed the local residual-CFAR rule.",
        "", "The direct baseline-change lane is primary for broad excitation; local CFAR is retained as complementary spatially-local evidence.",
        "", "| Direct positive-residual lane | Control mean | Event mean | Difference | Ratio |",
        "|---|---:|---:|---:|---:|",
        _detector_metric_row("Global", detector["positive_residual_signal_global"], "mean"),
        _detector_metric_row("Perisomatic ring", detector["positive_residual_signal_ring"], "mean"),
        "", "| Local residual-CFAR lane | Control fraction | Event fraction | Difference | Ratio |",
        "|---|---:|---:|---:|---:|",
        _detector_metric_row("Global", detector["residual_global"], "fraction"),
        _detector_metric_row("Perisomatic ring", detector["residual_ring"], "fraction"),
        "", "## Frozen transfer", "",
        f"Status: {summary['transfer']['status']}. Checkpoints: {summary['transfer']['checkpoint_count']}. Both full-FOV 128x128 and fixed 4x4 native-aspect arms are reported separately.",
        "", "| Spatial arm | Model | Model MSE | Persistence MSE | Global delta | High-change delta | Core delta | Ring delta | Positive-change corr. |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *_transfer_metric_rows(summary["transfer"]["spatial_arm_metrics"]),
        "", "Positive deltas mean lower MSE than persistence; negative deltas mean worse than persistence.",
        "", "## Resource contract", "",
        f"CPU only, one worker, {summary['resources']['cpu_threads']} CPU thread(s), chunk size {summary['resources']['resolved_chunk_frames']} frame(s).",
        f"Peak RSS: {_format_mib(summary['resources']['observed_peak_rss_bytes'])} / {_format_mib(summary['resources']['max_ram_bytes'])}; guard status: {summary['resources']['memory_guard_status']}.",
        f"Output: {summary['resources']['actual_output_bytes']} / {summary['resources']['max_output_bytes']} bytes.", "",
    ]
    return "\n".join(lines)


def _detector_metric_row(
    label: str, metric: Mapping[str, Any], measure: str
) -> str:
    values = [
        _format_metric(metric.get(key))
        for key in (f"pre_{measure}", f"post_{measure}", "difference", "ratio")
    ]
    return f"| {label} | {' | '.join(values)} |"


def _format_mib(value: Any) -> str:
    return "unavailable" if value is None else f"{float(value) / MIB:.1f} MiB"


def _transfer_metric_rows(arms: Mapping[str, Any]) -> list[str]:
    rows: list[str] = []
    for arm_name, models in arms.items():
        for model in models:
            values = [_format_metric(model.get(key)) for key in (
                "prediction_mse", "persistence_mse", "improvement_over_persistence_mse",
                "high_change_improvement", "core_improvement", "ring_improvement",
                "positive_change_correlation",
            )]
            rows.append(f"| {arm_name} | {model.get('model_id') or 'unlabeled'} | {' | '.join(values)} |")
    return rows or ["| - | No checkpoint evaluated | - | - | - | - | - | - | - |"]


def _format_metric(value: Any) -> str:
    return "-" if value is None else f"{float(value):+.6g}"


def _configure_cpu_limits(cpu_threads: int) -> None:
    value = str(int(cpu_threads))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = value
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _read_process_memory_status(
    status_path: str | Path = _PROC_SELF_STATUS,
) -> dict[str, Any]:
    """Read Linux process RSS counters without importing psutil."""

    path = Path(status_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {
            "current_rss_bytes": None,
            "peak_rss_bytes": None,
            "source": str(path),
            "peak_guard_available": False,
            "warning": f"Process RSS guard unavailable: {type(exc).__name__} reading {path}.",
        }

    values: dict[str, int] = {}
    for line in lines:
        key, separator, raw_value = line.partition(":")
        if not separator or key not in {"VmRSS", "VmHWM"}:
            continue
        parts = raw_value.split()
        if len(parts) != 2 or parts[1] != "kB":
            continue
        try:
            kib = int(parts[0])
        except ValueError:
            continue
        if kib >= 0:
            values[key] = kib * 1024

    current = values.get("VmRSS")
    peak = values.get("VmHWM")
    missing = [name for name, value in (("VmRSS", current), ("VmHWM", peak)) if value is None]
    warning = None
    if missing:
        warning = f"Process RSS field(s) unavailable in {path}: {', '.join(missing)}."
    return {
        "current_rss_bytes": current,
        "peak_rss_bytes": peak,
        "source": str(path),
        "peak_guard_available": peak is not None,
        "warning": warning,
    }


def _record_memory_observation(
    observations: list[dict[str, Any]],
    *,
    stage: str,
    max_ram_bytes: int,
    enforce: bool,
) -> None:
    snapshot = _read_process_memory_status()
    peak = snapshot["peak_rss_bytes"]
    if peak is None:
        guard_status = "unavailable"
    elif not enforce:
        guard_status = "observed"
    elif int(peak) <= max_ram_bytes:
        guard_status = "pass"
    else:
        guard_status = "failed"
    observation = {
        "stage": stage,
        **snapshot,
        "max_ram_bytes": int(max_ram_bytes),
        "guard_status": guard_status,
    }
    observations.append(observation)
    if guard_status == "failed":
        raise ResourceBudgetError(
            f"Process peak RSS {int(peak)} bytes exceeds max_ram_mib="
            f"{max_ram_bytes / MIB:.0f} at {stage.replace('_', ' ')}."
        )


def _memory_resource_fields(
    observations: list[dict[str, Any]],
    max_ram_bytes: int,
) -> dict[str, Any]:
    last = observations[-1] if observations else {}
    peaks = [
        int(item["peak_rss_bytes"])
        for item in observations
        if item.get("peak_rss_bytes") is not None
    ]
    warnings = list(
        dict.fromkeys(
            str(item["warning"])
            for item in observations
            if item.get("warning")
        )
    )
    status = str(last.get("guard_status") or "unavailable")
    return {
        "max_ram_bytes": int(max_ram_bytes),
        "observed_current_rss_bytes": last.get("current_rss_bytes"),
        "observed_peak_rss_bytes": max(peaks) if peaks else None,
        "process_memory_source": last.get("source", str(_PROC_SELF_STATUS)),
        "memory_guard_enforced": status in {"pass", "failed"},
        "memory_guard_status": status,
        "memory_guard_warning": "; ".join(warnings) if warnings else None,
        "memory_observations": [dict(item) for item in observations],
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_config(value: SomaExcitationConfig | Mapping[str, Any] | str | Path) -> SomaExcitationConfig:
    if isinstance(value, SomaExcitationConfig):
        value.validate()
        return value
    if isinstance(value, Mapping):
        return SomaExcitationConfig.from_dict(value)
    return SomaExcitationConfig.load_json(value)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _atomic_text(path: Path, text: str) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
