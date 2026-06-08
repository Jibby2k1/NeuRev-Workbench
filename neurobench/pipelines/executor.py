"""Dry-run pipeline executor and execution planning helpers."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from neurobench.data.checksums import checksum_file
from neurobench.data.video import load_video_array
from neurobench.models.pipeline import PipelineRun
from neurobench.pipeline_catalog import normalize_pipeline
from neurobench.pipelines.artifacts import ArtifactStore
from neurobench.pipelines.devices import resolve_device_from_spec
from neurobench.pipelines.specs import pipeline_spec_parameter_hash
from neurobench.pipelines.stages import StageRegistry, default_stage_registry
from neurobench.logging import RunLogger


@dataclass(frozen=True)
class DryRunStep:
    """One validated stage in a dry-run execution plan."""

    step_id: str
    stage_id: str
    input_artifact: str
    output_artifact: str
    params: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "stage_id": self.stage_id,
            "input_artifact": self.input_artifact,
            "output_artifact": self.output_artifact,
            "params": dict(self.params),
            "metadata": dict(self.metadata),
        }


def dry_run_pipeline(
    spec_or_pipeline: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    registry: StageRegistry | None = None,
    require_executable: bool = True,
    validate_artifacts: bool = False,
    initial_artifacts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate a structured pipeline and return an inspectable execution plan.

    The dry run does not execute image processing code. It validates stage IDs,
    parameter defaults/ranges, stage availability, and optionally artifact flow.
    """
    registry = registry or default_stage_registry(runner_stage_ids=tuple(_STAGE_RUNNERS))
    spec = _as_spec(spec_or_pipeline)
    pipeline = normalize_pipeline(spec.get("pipeline"), require_structured=True)
    steps = registry.validate_steps(pipeline, require_executable=require_executable)
    available_artifacts = set(initial_artifacts or ())
    planned_steps: list[DryRunStep] = []
    for step in steps:
        stage = registry.get(str(step["stage_id"]))
        input_artifact = stage.input_artifact
        output_artifact = stage.output_artifact
        if validate_artifacts and input_artifact and input_artifact not in available_artifacts:
            raise ValueError(
                f"Pipeline step '{step['id']}' requires missing artifact '{input_artifact}'."
            )
        if output_artifact:
            available_artifacts.add(output_artifact)
        planned_steps.append(
            DryRunStep(
                step_id=str(step["id"]),
                stage_id=str(step["stage_id"]),
                input_artifact=input_artifact,
                output_artifact=output_artifact,
                params=dict(step.get("params") or {}),
                metadata={
                    **dict(step.get("metadata") or {}),
                    "expected_qc_outputs": list(stage.expected_qc_outputs),
                },
            )
        )

    return {
        "status": "dry_run_ok",
        "dataset_id": spec.get("dataset_id", ""),
        "run_id": spec.get("run_id", ""),
        "parameter_hash": pipeline_spec_parameter_hash(spec),
        "require_executable": require_executable,
        "validate_artifacts": validate_artifacts,
        "steps": [step.as_dict() for step in planned_steps],
        "available_artifacts": sorted(available_artifacts),
    }


def execute_pipeline(
    spec_or_pipeline: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    run_root: str | Path,
    registry: StageRegistry | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Execute the currently wired CPU-safe pipeline subset.

    This first executor intentionally supports only small, deterministic Python
    stages. Unsupported catalog stages fail clearly so planned UI workflows do
    not appear silently executable.
    """
    spec = _as_spec(spec_or_pipeline)
    device_spec = resolve_device_from_spec(spec, override=device)
    plan = dry_run_pipeline(spec, registry=registry, require_executable=True, validate_artifacts=False)
    created_at = datetime.now(timezone.utc).isoformat()
    pipeline_run = PipelineRun(
        schema_version=1,
        run_id=str(spec.get("run_id") or f"run_{plan['parameter_hash'][:12]}"),
        dataset_id=str(spec.get("dataset_id") or "unknown_dataset"),
        pipeline_spec_id=str(spec.get("pipeline_spec_id") or spec.get("run_id") or "inline_pipeline_spec"),
        status="running",
        created_at=created_at,
        parameter_hash=str(plan["parameter_hash"]),
        artifacts=[],
        environment={
            "runner": "neurobench.pipelines.executor",
            "device": device_spec.resolved,
            "device_requested": device_spec.requested,
            "device_backend": device_spec.backend,
            "device_available": device_spec.available,
            "device_reason": device_spec.reason,
        },
        extras={"input_checksums": []},
    )
    store = ArtifactStore(run_root, pipeline_run)
    logger = RunLogger(run_root, pipeline_run)
    artifacts: dict[str, Path] = {}
    try:
        for step in plan["steps"]:
            stage_id = str(step["stage_id"])
            logger.stage_started(stage_id, step_id=step["step_id"])
            runner = _STAGE_RUNNERS.get(stage_id)
            if runner is None:
                raise NotImplementedError(f"Pipeline stage '{stage_id}' is not wired for local execution yet.")
            output_key, output_path = runner(step, artifacts, store)
            if output_key:
                artifacts[output_key] = output_path
                artifacts[f"{step['step_id']}:{output_key}"] = output_path
                artifacts[str(step["step_id"])] = output_path
            logger.stage_completed(stage_id, step_id=step["step_id"], output_artifact=output_key)
        pipeline_run.status = "completed"
        pipeline_run.completed_at = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        pipeline_run.status = "failed"
        pipeline_run.completed_at = datetime.now(timezone.utc).isoformat()
        logger.error(str(exc), event_type="pipeline_failed")
        store.write_manifest()
        raise
    store.write_manifest()
    return {"status": pipeline_run.status, "run_root": str(Path(run_root)), "plan": plan, "pipeline_run": pipeline_run.to_dict()}


def _as_spec(spec_or_pipeline: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if isinstance(spec_or_pipeline, Mapping):
        if "pipeline" not in spec_or_pipeline:
            raise ValueError("Pipeline spec is missing required 'pipeline' field.")
        return dict(spec_or_pipeline)
    if isinstance(spec_or_pipeline, Sequence) and not isinstance(spec_or_pipeline, (str, bytes, bytearray)):
        return {"pipeline": list(spec_or_pipeline)}
    raise TypeError("dry_run_pipeline expects a pipeline spec mapping or pipeline step sequence.")


def _load_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required to execute local pipeline stages.") from exc
    return np


def _load_npy(path: Path):
    np = _load_numpy()
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        return load_video_array(path)
    raise ValueError(f"Local executor supports .npy/.tif/.tiff video artifacts, got: {path}")


def _run_source_video_import(step: Mapping[str, Any], artifacts: Mapping[str, Path], store: ArtifactStore) -> tuple[str, Path]:
    source = Path(str(step["params"]["source"])).expanduser()
    if not source.is_absolute():
        source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source video does not exist: {source}")
    summary: dict[str, Any] = {}
    if source.suffix.lower() in {".npy", ".tif", ".tiff"}:
        video = _load_npy(source)
        summary.update({"shape": [int(value) for value in video.shape], "dtype": str(video.dtype)})
    _record_input_checksum(store, checksum_file(source, path_id="raw_video"))
    store.register_file(
        source,
        artifact_id="raw_video.v1",
        kind="raw_video",
        producer_stage=str(step["stage_id"]),
        summary=summary,
    )
    return "raw_video", source


def _run_temporal_highpass_gaussian(
    step: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    store: ArtifactStore,
) -> tuple[str, Path]:
    np = _load_numpy()
    source = _require_artifact(artifacts, "raw_video", step)
    video = _load_npy(source).astype(np.float32, copy=False)
    sigma = float(step["params"].get("sigma_frames", 6.0))
    try:
        from scipy.ndimage import gaussian_filter1d

        baseline = gaussian_filter1d(video, sigma=sigma, axis=0, mode="nearest") if sigma > 0 else video * 0
    except ModuleNotFoundError:
        baseline = np.mean(video, axis=0, keepdims=True)
    highpass = (video - baseline).astype(np.float32, copy=False)
    out = store.artifact_path("preprocessing", "highpass_video.npy")
    np.save(out, highpass)
    store.register_file(
        out,
        artifact_id="highpass_video.v1",
        kind="highpass_video",
        producer_stage=str(step["stage_id"]),
        summary={"shape": [int(value) for value in highpass.shape], "sigma_frames": sigma},
    )
    return "highpass_video", out


def _run_robust_positive_local_z(
    step: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    store: ArtifactStore,
) -> tuple[str, Path]:
    np = _load_numpy()
    source = artifacts.get("denoised_video") or artifacts.get("highpass_video") or artifacts.get("raw_video")
    if source is None:
        raise ValueError(f"Pipeline step '{step['step_id']}' requires missing artifact 'highpass_video'.")
    video = _load_npy(source).astype(np.float32, copy=False)
    epsilon = float(step["params"].get("epsilon", 1.0))
    frame_median = np.median(video, axis=(1, 2), keepdims=True)
    mad = np.median(np.abs(video - frame_median), axis=(1, 2), keepdims=True)
    z_stack = np.maximum((video - frame_median) / (1.4826 * mad + epsilon), 0.0).astype(np.float32, copy=False)
    out = store.artifact_path("preprocessing", "z_stack.npy")
    np.save(out, z_stack)
    store.register_file(
        out,
        artifact_id="z_stack.v1",
        kind="z_stack",
        producer_stage=str(step["stage_id"]),
        summary={"shape": [int(value) for value in z_stack.shape], "max_z": float(np.max(z_stack))},
    )
    return "z_stack", out


def _run_event_preserving_noise_suppression(
    step: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    store: ArtifactStore,
) -> tuple[str, Path]:
    np = _load_numpy()
    source = artifacts.get("highpass_video") or artifacts.get("raw_video")
    if source is None:
        raise ValueError(f"Pipeline step '{step['step_id']}' requires missing artifact 'highpass_video'.")
    video = _load_npy(source).astype(np.float32, copy=False)
    params = step["params"]
    sigma_px = float(params.get("spatial_sigma_px", 1.0))
    temporal_window = max(1, int(params.get("temporal_window_frames", 3)))
    threshold_z = float(params.get("threshold_z", 6.0))
    denoised = video.astype(np.float32, copy=True)
    try:
        from scipy.ndimage import gaussian_filter, median_filter

        if sigma_px > 0:
            denoised = gaussian_filter(denoised, sigma=(0.0, sigma_px, sigma_px), mode="nearest").astype(np.float32, copy=False)
        if temporal_window > 1:
            if temporal_window % 2 == 0:
                temporal_window += 1
            temporal_median = median_filter(video, size=(temporal_window, 1, 1), mode="nearest")
            residual = video - temporal_median
            frame_scale = np.median(np.abs(residual), axis=(1, 2), keepdims=True) * 1.4826 + 1e-6
            impulse = np.abs(residual) > threshold_z * frame_scale
            denoised[impulse] = temporal_median[impulse]
    except ModuleNotFoundError:
        pass
    out = store.artifact_path("preprocessing", "denoised_video.npy")
    np.save(out, denoised.astype(np.float32, copy=False))
    store.register_file(
        out,
        artifact_id="denoised_video.v1",
        kind="denoised_video",
        producer_stage=str(step["stage_id"]),
        summary={"shape": [int(value) for value in denoised.shape], "spatial_sigma_px": sigma_px, "temporal_window_frames": temporal_window},
    )
    return "denoised_video", out


def _run_adaptive_ewma_z(
    step: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    store: ArtifactStore,
) -> tuple[str, Path]:
    np = _load_numpy()
    source = artifacts.get("denoised_video") or artifacts.get("highpass_video") or _require_artifact(artifacts, "raw_video", step)
    video = _load_npy(source).astype(np.float32, copy=False)
    params = step["params"]
    alpha = float(params.get("alpha", 0.02))
    threshold_z = float(params.get("threshold_z", 3.0))
    epsilon = float(params.get("epsilon", 1.0))
    if not 0 < alpha <= 1:
        raise ValueError("adaptive_ewma_z alpha must be in (0, 1].")
    mean = video[0].astype(np.float32, copy=True)
    var = np.ones_like(mean, dtype=np.float32) * float(np.var(video[0]) + epsilon)
    z_stack = np.zeros_like(video, dtype=np.float32)
    for index, frame in enumerate(video):
        std = np.sqrt(np.maximum(var, 0.0) + epsilon)
        z_stack[index] = np.maximum((frame - mean) / std, 0.0)
        delta = frame - mean
        mean = mean + alpha * delta
        var = (1.0 - alpha) * (var + alpha * delta * delta)
    out = store.artifact_path("preprocessing", "adaptive_ewma_z_stack.npy")
    np.save(out, z_stack.astype(np.float32, copy=False))
    store.register_file(
        out,
        artifact_id="adaptive_ewma_z_stack.v1",
        kind="z_stack",
        producer_stage=str(step["stage_id"]),
        summary={
            "shape": [int(value) for value in z_stack.shape],
            "alpha": alpha,
            "threshold_z": threshold_z,
            "active_fraction": float(np.mean(z_stack >= threshold_z)),
        },
    )
    return "z_stack", out


def _run_candidate_event_pipeline(
    step: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    store: ArtifactStore,
) -> tuple[str, Path]:
    np = _load_numpy()
    source = _require_artifact(artifacts, "z_stack", step)
    z_stack = _load_npy(source).astype(np.float32, copy=False)
    params = step["params"]
    threshold = float(params.get("event_threshold_z", params.get("threshold_z", 2.4)))
    min_area = int(params.get("min_area_px", 4))
    mask = z_stack >= threshold
    events: list[dict[str, Any]] = []
    try:
        from scipy import ndimage

        for frame_index, frame_mask in enumerate(mask):
            labels, count = ndimage.label(frame_mask)
            objects = ndimage.find_objects(labels)
            for label_index, slices in enumerate(objects, start=1):
                if slices is None:
                    continue
                component = labels[slices] == label_index
                area = int(np.count_nonzero(component))
                if area < min_area:
                    continue
                ys, xs = np.nonzero(component)
                y0, x0 = slices[0].start, slices[1].start
                abs_xs = xs + x0
                abs_ys = ys + y0
                peak = float(np.max(z_stack[frame_index][slices][component]))
                events.append(
                    {
                        "event_id": f"event_{len(events) + 1:04d}",
                        "frame": int(frame_index),
                        "x": float(np.mean(abs_xs)),
                        "y": float(np.mean(abs_ys)),
                        "area_px": area,
                        "peak_z": peak,
                    }
                )
    except ModuleNotFoundError:
        frame_indices, ys, xs = np.nonzero(mask)
        for frame_index, y, x in zip(frame_indices, ys, xs):
            events.append(
                {
                    "event_id": f"event_{len(events) + 1:04d}",
                    "frame": int(frame_index),
                    "x": float(x),
                    "y": float(y),
                    "area_px": 1,
                    "peak_z": float(z_stack[frame_index, y, x]),
                }
            )
    out = store.artifact_path("events", "candidate_event_components.json")
    out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_threshold_z": threshold,
                "min_area_px": min_area,
                "events": events,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    store.register_file(
        out,
        artifact_id="candidate_event_components.v1",
        kind="candidate_events",
        producer_stage=str(step["stage_id"]),
        summary={"event_count": len(events), "event_threshold_z": threshold, "min_area_px": min_area},
    )
    return "candidate_events", out


def _run_spatial_gaussian(
    step: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    store: ArtifactStore,
) -> tuple[str, Path]:
    np = _load_numpy()
    source = artifacts.get("denoised_video") or _require_artifact(artifacts, "highpass_video", step)
    video = _load_npy(source).astype(np.float32, copy=False)
    sigma = float(step["params"].get("sigma_px", 0.8))
    if sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
        except ModuleNotFoundError as exc:
            raise RuntimeError("SciPy is required for spatial_gaussian execution.") from exc
        smoothed = gaussian_filter(video, sigma=(0.0, sigma, sigma), mode="nearest").astype(np.float32, copy=False)
    else:
        smoothed = video.astype(np.float32, copy=True)
    out = store.artifact_path("preprocessing", "smoothed_video.npy")
    np.save(out, smoothed)
    store.register_file(
        out,
        artifact_id="smoothed_video.v1",
        kind="smoothed_video",
        producer_stage=str(step["stage_id"]),
        summary={"shape": [int(value) for value in smoothed.shape], "sigma_px": sigma},
    )
    return "smoothed_video", out


def _run_rigid_shift_estimate(
    step: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    store: ArtifactStore,
) -> tuple[str, Path]:
    np = _load_numpy()
    from neurobench.algorithms.motion import estimate_rigid_shifts

    source = _require_artifact(artifacts, "raw_video", step)
    video = _load_npy(source).astype(np.float32, copy=False)
    params = step["params"]
    max_shift_px = int(params.get("max_shift_px", 4))
    reference = str(params.get("reference", "first"))
    result = estimate_rigid_shifts(video, max_shift_px=max_shift_px, reference=reference, device=_resolved_device(store))

    shift_trace_out = store.artifact_path("motion", "rigid_shift_trace.json")
    shift_trace_out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage_id": str(step["stage_id"]),
                "summary": result["summary"],
                "shifts": result["shifts"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    store.register_file(
        shift_trace_out,
        artifact_id="rigid_shift_trace.v1",
        kind="rigid_shift_trace",
        producer_stage=str(step["stage_id"]),
        summary=dict(result["summary"]),
    )

    out = store.artifact_path("motion", "registered_video.npy")
    np.save(out, result["registered_video"].astype(np.float32, copy=False))
    store.register_file(
        out,
        artifact_id="registered_video.v1",
        kind="registered_video",
        producer_stage=str(step["stage_id"]),
        summary=dict(result["summary"]),
    )
    return "registered_video", out


def _run_gamma_cfar(
    step: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    store: ArtifactStore,
) -> tuple[str, Path]:
    np = _load_numpy()
    from neurobench.algorithms.cfar import robust_local_cfar

    source = _require_artifact(artifacts, "smoothed_video", step)
    video = _load_npy(source).astype(np.float32, copy=False)
    params = step["params"]
    pfa = float(params.get("pfa", 0.001))
    guard_px = int(params.get("guard_px", 2))
    training_radius_px = int(params.get("training_radius_px", max(guard_px + 1, 11)))
    epsilon = float(params.get("epsilon", 1e-6))
    result = robust_local_cfar(
        video,
        pfa=pfa,
        guard_px=guard_px,
        training_radius_px=training_radius_px,
        epsilon=epsilon,
        device=_resolved_device(store),
    )
    mask = result["mask"].astype(np.uint8, copy=False)
    metadata = dict(step.get("metadata") or {})
    previous_step = metadata.get("previous_mask_step") or params.get("previous_mask_step")
    combine_mode = str(metadata.get("combine_mode") or params.get("combine_mode") or "replace")
    if previous_step:
        previous_path = artifacts.get(f"{previous_step}:candidate_mask") or artifacts.get(str(previous_step))
        if previous_path is None:
            raise ValueError(f"Pipeline step '{step['step_id']}' references missing previous mask step '{previous_step}'.")
        previous_mask = _load_npy(previous_path).astype(bool, copy=False)
        if previous_mask.shape != mask.shape:
            raise ValueError(
                f"Pipeline step '{step['step_id']}' previous mask shape {previous_mask.shape} does not match {mask.shape}."
            )
        if combine_mode == "intersection":
            mask = (previous_mask & mask.astype(bool, copy=False)).astype(np.uint8, copy=False)
        elif combine_mode == "union":
            mask = (previous_mask | mask.astype(bool, copy=False)).astype(np.uint8, copy=False)
        elif combine_mode == "replace":
            pass
        else:
            raise ValueError(f"Unsupported CFAR combine_mode '{combine_mode}'. Use replace, intersection, or union.")
    out = store.artifact_path("candidates", f"{_safe_step_name(str(step['step_id']))}_candidate_mask.npy")
    np.save(out, mask)
    summary: dict[str, Any] = {
        "shape": [int(value) for value in mask.shape],
        "pfa": pfa,
        "guard_px": guard_px,
        "training_radius_px": training_radius_px,
        "threshold_z": float(result["threshold_z"]),
        "active_fraction": float(np.mean(mask)),
        "combine_mode": combine_mode,
    }
    if previous_step:
        summary["previous_mask_step"] = str(previous_step)
    if "update_alpha" in params:
        summary["update_alpha"] = float(params["update_alpha"])
    latest_out = store.artifact_path("candidates", "candidate_mask.npy")
    np.save(latest_out, mask)
    store.register_file(
        out,
        artifact_id=f"{_safe_step_name(str(step['step_id']))}_candidate_mask.v1",
        kind="candidate_mask",
        producer_stage=str(step["stage_id"]),
        summary=summary,
    )
    return "candidate_mask", out


def _run_component_filter(step: Mapping[str, Any], artifacts: Mapping[str, Path], store: ArtifactStore) -> tuple[str, Path]:
    np = _load_numpy()
    params = step["params"]
    seed_z = float(params.get("seed_z", 2.0))
    min_area = int(params.get("min_area_px", 4))
    max_area = int(params.get("max_area_px", 260))
    support_min_frames = max(1, int(params.get("support_min_frames", 1)))
    if "candidate_mask" in artifacts:
        source = _require_artifact(artifacts, "candidate_mask", step)
        candidate_mask = _load_npy(source).astype(bool, copy=False)
        projection = np.sum(candidate_mask.astype(np.float32, copy=False), axis=0)
        z_stack = _load_npy(artifacts["z_stack"]).astype(np.float32, copy=False) if "z_stack" in artifacts else None
        if z_stack is not None:
            evidence_projection = np.max(z_stack, axis=0)
            mask = (projection >= float(support_min_frames)) & (evidence_projection >= seed_z)
        else:
            mask = projection >= float(support_min_frames)
        evidence_source = "candidate_mask"
    else:
        source = _require_artifact(artifacts, "z_stack", step)
        z_stack = _load_npy(source).astype(np.float32, copy=False)
        projection = np.max(z_stack, axis=0)
        mask = projection >= seed_z
        evidence_source = "z_stack"
    try:
        from scipy import ndimage
    except ModuleNotFoundError as exc:
        raise RuntimeError("SciPy is required for component_filter execution.") from exc
    labels, count = ndimage.label(mask)
    objects = ndimage.find_objects(labels)
    candidates: list[dict[str, Any]] = []
    for label_index, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        component = labels[slices] == label_index
        area = int(np.count_nonzero(component))
        if area < min_area or area > max_area:
            continue
        ys, xs = np.nonzero(component)
        y0, x0 = slices[0].start, slices[1].start
        abs_xs = xs + x0
        abs_ys = ys + y0
        if z_stack is not None:
            peak = float(np.max(np.max(z_stack, axis=0)[slices][component]))
        else:
            peak = float(np.max(projection[slices][component]))
        candidates.append(
            {
                "id": f"roi_{len(candidates) + 1:03d}",
                "x": float(np.mean(abs_xs)),
                "y": float(np.mean(abs_ys)),
                "area_px": area,
                "peak_z": peak,
                "bbox": [int(np.min(abs_xs)), int(np.min(abs_ys)), int(np.max(abs_xs)), int(np.max(abs_ys))],
            }
        )
    out = store.artifact_path("candidates", "roi_candidates.json")
    out.write_text(json.dumps({"schema_version": 1, "candidates": candidates}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    store.register_file(
        out,
        artifact_id="roi_candidates.v1",
        kind="roi_candidates",
        producer_stage=str(step["stage_id"]),
        summary={
            "count": len(candidates),
            "seed_z": seed_z,
            "min_area_px": min_area,
            "max_area_px": max_area,
            "support_min_frames": support_min_frames,
            "evidence_source": evidence_source,
        },
    )
    return "roi_candidates", out


def _run_local_background_ring(
    step: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    store: ArtifactStore,
) -> tuple[str, Path]:
    np = _load_numpy()
    source = artifacts.get("denoised_video") or artifacts.get("highpass_video") or artifacts.get("raw_video")
    if source is None:
        raise ValueError(f"Pipeline step '{step['step_id']}' requires a video artifact for trace extraction.")
    video = _load_npy(source).astype(np.float32, copy=False)
    candidate_path = _require_artifact(artifacts, "roi_candidates", step)
    candidates = json.loads(candidate_path.read_text(encoding="utf-8")).get("candidates", [])
    params = step["params"]
    outer_radius = max(1, int(params.get("outer_radius_px", 15)))
    neuropil_weight = float(params.get("neuropil_weight", 0.7))
    y_grid, x_grid = np.mgrid[0 : video.shape[1], 0 : video.shape[2]]
    traces = []
    for candidate in candidates:
        cx = float(candidate.get("x", 0.0))
        cy = float(candidate.get("y", 0.0))
        area = max(1.0, float(candidate.get("area_px", 9.0)))
        inner_radius = max(1.0, float(np.sqrt(area / np.pi)))
        distances = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        roi_mask = distances <= inner_radius
        ring_mask = (distances > inner_radius + 1.0) & (distances <= max(inner_radius + 2.0, float(outer_radius)))
        if not np.any(roi_mask):
            continue
        raw_trace = video[:, roi_mask].mean(axis=1)
        if np.any(ring_mask):
            background_trace = video[:, ring_mask].mean(axis=1)
        else:
            background_trace = np.zeros_like(raw_trace)
        corrected = raw_trace - neuropil_weight * background_trace
        traces.append(
            {
                "roi_id": str(candidate.get("id") or candidate.get("candidate_id") or f"roi_{len(traces) + 1:03d}"),
                "x": cx,
                "y": cy,
                "area_px": area,
                "inner_radius_px": inner_radius,
                "outer_radius_px": outer_radius,
                "neuropil_weight": neuropil_weight,
                "raw_trace": [float(value) for value in raw_trace],
                "background_trace": [float(value) for value in background_trace],
                "corrected_trace": [float(value) for value in corrected],
            }
        )
    out = store.artifact_path("traces", "roi_traces.json")
    out.write_text(json.dumps({"schema_version": 1, "traces": traces}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    store.register_file(
        out,
        artifact_id="roi_traces.v1",
        kind="roi_traces",
        producer_stage=str(step["stage_id"]),
        summary={"count": len(traces), "outer_radius_px": outer_radius, "neuropil_weight": neuropil_weight},
    )
    return "roi_traces", out


def _run_trace_event_scoring(
    step: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    store: ArtifactStore,
) -> tuple[str, Path]:
    payload = _trace_events_payload(step, artifacts, mode="robust_z")
    out = store.artifact_path("events", "candidate_events.json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    store.register_file(
        out,
        artifact_id="candidate_events.v1",
        kind="candidate_events",
        producer_stage=str(step["stage_id"]),
        summary={"event_count": len(payload["events"]), "roi_count": len(payload["roi_event_counts"]), "mode": "robust_z"},
    )
    return "candidate_events", out


def _run_robust_kalman_positive_innovation(
    step: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    store: ArtifactStore,
) -> tuple[str, Path]:
    payload = _trace_events_payload(step, artifacts, mode="robust_kalman")
    out = store.artifact_path("events", "kalman_candidate_events.json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    store.register_file(
        out,
        artifact_id="kalman_candidate_events.v1",
        kind="candidate_events",
        producer_stage=str(step["stage_id"]),
        summary={"event_count": len(payload["events"]), "roi_count": len(payload["roi_event_counts"]), "mode": "robust_kalman"},
    )
    return "candidate_events", out


def _trace_events_payload(step: Mapping[str, Any], artifacts: Mapping[str, Path], *, mode: str) -> dict[str, Any]:
    np = _load_numpy()
    traces_path = _require_artifact(artifacts, "roi_traces", step)
    traces = json.loads(traces_path.read_text(encoding="utf-8")).get("traces", [])
    params = step["params"]
    threshold = float(params.get("event_threshold_z", params.get("threshold_z", 2.4)))
    events: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for trace_row in traces:
        roi_id = str(trace_row.get("roi_id") or "")
        trace = np.asarray(trace_row.get("corrected_trace") or trace_row.get("raw_trace") or [], dtype=np.float32)
        if not roi_id or trace.size == 0:
            continue
        if mode == "robust_kalman":
            score = _kalman_positive_innovation_score(
                trace,
                kalman_gain=float(params.get("kalman_gain", 0.06)),
                spike_gain=float(params.get("spike_gain", 0.008)),
                negative_gain=float(params.get("negative_gain", 0.11)),
            )
        else:
            center = float(np.median(trace))
            scale = float(1.4826 * np.median(np.abs(trace - center)) + 1e-6)
            score = np.maximum((trace - center) / scale, 0.0)
        frames = _local_maxima(score, threshold)
        counts[roi_id] = len(frames)
        for frame in frames:
            events.append(
                {
                    "roi_id": roi_id,
                    "frame": int(frame),
                    "score": float(score[frame]),
                    "amplitude": float(trace[frame]),
                    "mode": mode,
                }
            )
    return {
        "schema_version": 1,
        "event_threshold_z": threshold,
        "mode": mode,
        "events": events,
        "roi_event_counts": counts,
    }


def _kalman_positive_innovation_score(trace, *, kalman_gain: float, spike_gain: float, negative_gain: float):
    np = _load_numpy()
    baseline = float(trace[0])
    innovations = np.zeros_like(trace, dtype=np.float32)
    scale_values: list[float] = []
    for index, value in enumerate(trace):
        innovation = float(value - baseline)
        innovations[index] = max(0.0, innovation)
        scale_values.append(abs(innovation))
        if innovation > 0:
            baseline += spike_gain * innovation
        else:
            baseline += negative_gain * innovation
        baseline += kalman_gain * (float(value) - baseline)
    scale = float(1.4826 * np.median(np.asarray(scale_values, dtype=np.float32)) + 1e-6)
    return innovations / scale


def _local_maxima(score, threshold: float) -> list[int]:
    frames: list[int] = []
    for index, value in enumerate(score):
        if float(value) < threshold:
            continue
        left = float(score[index - 1]) if index > 0 else float("-inf")
        right = float(score[index + 1]) if index + 1 < len(score) else float("-inf")
        if float(value) >= left and float(value) >= right:
            frames.append(index)
    return frames


def _safe_step_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value).strip("._-")
    return cleaned or "step"


def _run_heuristic_priority_v1(
    step: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    store: ArtifactStore,
) -> tuple[str, Path]:
    from neurobench.discovery.ranking import rank_candidates

    source = _require_artifact(artifacts, "roi_candidates", step)
    payload = json.loads(source.read_text(encoding="utf-8"))
    candidates = list(payload.get("candidates", []) or [])
    weights = {
        "local_correlation_weight": step["params"].get("local_correlation_weight", 0.2),
        "event_support_weight": step["params"].get("event_support_weight", 0.2),
        "artifact_weight": step["params"].get("artifact_weight", -0.15),
    }
    video_shape = None
    if "raw_video" in artifacts:
        try:
            video_shape = _load_npy(artifacts["raw_video"]).shape
        except Exception:
            video_shape = None
    ranked = rank_candidates(candidates, video_shape=video_shape, weights=weights)
    out = store.artifact_path("candidates", "ranked_candidates.json")
    out.write_text(json.dumps(ranked, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    top_score = ranked["ranked_candidates"][0]["priority_score"] if ranked["ranked_candidates"] else 0.0
    store.register_file(
        out,
        artifact_id="ranked_candidates.v1",
        kind="ranked_candidates",
        producer_stage=str(step["stage_id"]),
        summary={"count": len(ranked["ranked_candidates"]), "top_priority_score": float(top_score), "weights": weights},
    )
    return "ranked_candidates", out


def _run_generate_neuron_review_app(
    step: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    store: ArtifactStore,
) -> tuple[str, Path]:
    upstream = (
        artifacts.get("ranked_candidates")
        or artifacts.get("candidate_events")
        or artifacts.get("roi_candidates")
        or artifacts.get("z_stack")
    )
    if upstream is None:
        raise ValueError(f"Pipeline step '{step['step_id']}' requires candidate artifacts for review app generation.")
    payload = {
        "schema_version": 1,
        "stage_id": str(step["stage_id"]),
        "step_id": str(step["step_id"]),
        "status": "manifest_only",
        "message": "The local executor recorded review-app intent. Full dashboard generation is handled by the workbench Generate View job.",
        "source_artifact": str(upstream),
    }
    out = store.artifact_path("review_app", "review_app_manifest.json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    store.register_file(
        out,
        artifact_id="review_app_manifest.v1",
        kind="review_app",
        producer_stage=str(step["stage_id"]),
        summary={"status": "manifest_only", "source_artifact": upstream.name},
    )
    return "review_app", out


def _run_video_manifest_build(step: Mapping[str, Any], artifacts: Mapping[str, Path], store: ArtifactStore) -> tuple[str, Path]:
    from neurobench.data.video_manifest import build_video_manifest

    params = step["params"]
    out = store.artifact_path("manifest", "video_manifest.json")
    manifest = build_video_manifest(
        input_dir=params.get("input_dir"),
        dataset_id=str(params.get("dataset_id") or store.pipeline_run.dataset_id),
        filename_regex=str(params.get("filename_regex")),
        strict=bool(params.get("strict", False)),
    )
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    label_counts = store.artifact_path("manifest", "label_counts.json")
    label_counts.write_text(json.dumps(manifest.get("label_counts") or {}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    store.register_file(out, artifact_id="video_manifest.v1", kind="video_manifest", producer_stage=str(step["stage_id"]), schema="video_manifest", summary={"video_count": len(manifest.get("videos") or []), "label_counts": manifest.get("label_counts") or {}})
    return "video_manifest", out


def _run_template_build_from_video(step: Mapping[str, Any], artifacts: Mapping[str, Path], store: ArtifactStore) -> tuple[str, Path]:
    from neurobench.algorithms.template_matching import write_template_artifacts
    from neurobench.data.video_manifest import video_by_id
    from neurobench.manifests import load_json

    manifest_path = artifacts.get("video_manifest") or Path(str(step["params"].get("manifest")))
    manifest = load_json(manifest_path)
    params = step["params"]
    ref = str(params.get("reference_video_id") or "1_neutral")
    video = video_by_id(manifest, ref)
    out_dir = store.artifact_path("template", "template_spec.json").parent
    write_template_artifacts(video_path=video["path"], source_video_id=ref, out_dir=out_dir, outlier_rejection=bool(params.get("outlier_rejection", True)), max_outlier_fraction=float(params.get("max_outlier_fraction", 0.05)), z_threshold=float(params.get("z_threshold", 3.5)), chunk_size_frames=int(params.get("chunk_size_frames", 64)))
    out = out_dir / "template_spec.json"
    spec = json.loads(out.read_text(encoding="utf-8"))
    store.register_file(out, artifact_id="template_spec.v1", kind="template_spec", producer_stage=str(step["stage_id"]), schema="template_spec", summary={"template_id": spec.get("template_id"), "removed_frames": len(spec.get("outlier_rejection", {}).get("removed_frame_indices") or [])})
    return "template_spec", out


def _run_template_register_video(step: Mapping[str, Any], artifacts: Mapping[str, Path], store: ArtifactStore) -> tuple[str, Path]:
    from neurobench.algorithms.template_matching import write_registration_artifacts
    from neurobench.manifests import load_json

    manifest = load_json(_require_artifact(artifacts, "video_manifest", step))
    template = load_json(_require_artifact(artifacts, "template_spec", step))
    params = step["params"]
    out_dir = store.artifact_path("registration", "registration_summary.json").parent
    results = []
    rot = params.get("rotation_range_deg") or [-10.0, 10.0]
    for video in manifest.get("videos", []) or []:
        results.append(write_registration_artifacts(video_path=video["path"], video_id=str(video["video_id"]), template_spec=template, out_dir=out_dir, transform_model=str(params.get("transform_model", "rigid")), rotation_range_deg=(float(rot[0]), float(rot[1])), rotation_step_deg=float(params.get("rotation_step_deg", 0.5)), allow_uniform_scale=bool(params.get("allow_uniform_scale", False)), chunk_size_frames=int(params.get("chunk_size_frames", 64))))
    summary = {"schema_version": 1, "registration_dir": str(out_dir), "video_count": len(results), "warnings": sum(len(r.get("qc", {}).get("warnings") or []) for r in results), "results": [{"video_id": r["video_id"], "result": str(out_dir / r["video_id"] / "registration_result.json"), "score": r.get("score"), "qc": r.get("qc")} for r in results]}
    out = out_dir / "registration_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    store.register_file(out, artifact_id="registration_results.v1", kind="registration_results", producer_stage=str(step["stage_id"]), schema="registration_result", summary={"video_count": len(results), "warnings": summary["warnings"]})
    return "registration_results", out


def _run_apply_video_registration(step: Mapping[str, Any], artifacts: Mapping[str, Path], store: ArtifactStore) -> tuple[str, Path]:
    from neurobench.algorithms.template_matching import write_registered_video_artifacts
    from neurobench.manifests import load_json

    manifest = load_json(_require_artifact(artifacts, "video_manifest", step))
    template = load_json(_require_artifact(artifacts, "template_spec", step))
    registration_dir = _require_artifact(artifacts, "registration_results", step).parent
    out_dir = store.artifact_path("registered", "registered_summary.json").parent
    summaries = []
    for video in manifest.get("videos", []) or []:
        result = load_json(registration_dir / str(video["video_id"]) / "registration_result.json")
        summaries.append(write_registered_video_artifacts(video_path=video["path"], registration_result=result, template_spec=template, out_dir=out_dir, output_dtype=str(step["params"].get("output_dtype", "float32")), chunk_size_frames=int(step["params"].get("chunk_size_frames", 64))))
    summary = {"schema_version": 1, "registered_dir": str(out_dir), "video_count": len(summaries), "videos": summaries}
    out = out_dir / "registered_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    store.register_file(out, artifact_id="registered_videos.v1", kind="registered_videos", producer_stage=str(step["stage_id"]), summary={"video_count": len(summaries)})
    return "registered_videos", out


def _run_grid_32x32_generate(step: Mapping[str, Any], artifacts: Mapping[str, Path], store: ArtifactStore) -> tuple[str, Path]:
    from neurobench.algorithms.grid_regions import write_grid_spec_artifacts
    from neurobench.manifests import load_json

    out = store.artifact_path("grid", "grid_spec_32x32.json")
    spec = write_grid_spec_artifacts(template_spec=load_json(_require_artifact(artifacts, "template_spec", step)), out_path=out, rows=int(step["params"].get("rows", 32)), cols=int(step["params"].get("cols", 32)))
    store.register_file(out, artifact_id="grid_spec.v1", kind="grid_spec", producer_stage=str(step["stage_id"]), schema="grid_spec", summary={"rows": spec["rows"], "cols": spec["cols"], "region_count": spec["region_count"]})
    return "grid_spec", out


def _run_grid_state_extract(step: Mapping[str, Any], artifacts: Mapping[str, Path], store: ArtifactStore) -> tuple[str, Path]:
    from neurobench.algorithms.grid_regions import write_grid_state_artifacts
    from neurobench.manifests import load_json

    manifest = load_json(_require_artifact(artifacts, "video_manifest", step))
    grid = load_json(_require_artifact(artifacts, "grid_spec", step))
    registered_dir = _require_artifact(artifacts, "registered_videos", step).parent
    out_dir = store.artifact_path("grid_states", "grid_states_summary.json").parent
    summaries = []
    for video in manifest.get("videos", []) or []:
        vid = str(video["video_id"])
        summaries.append(write_grid_state_artifacts(registered_video_path=registered_dir / vid / "registered_video.npy", grid_spec=grid, out_dir=out_dir, video_id=vid, label=str(video.get("label") or ""), features=step["params"].get("features") or ["mean_intensity"], normalization=str(step["params"].get("normalization", "per_video_robust_percentile")), frame_rate_hz=video.get("frame_rate_hz"), chunk_size_frames=int(step["params"].get("chunk_size_frames", 64)), max_grid_state_bytes=step["params"].get("max_grid_state_bytes", 1_000_000_000)))
    summary = {"schema_version": 1, "grid_states_dir": str(out_dir), "video_count": len(summaries), "videos": summaries}
    out = out_dir / "grid_states_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    store.register_file(out, artifact_id="grid_states.v1", kind="grid_states", producer_stage=str(step["stage_id"]), summary={"video_count": len(summaries), "shape": summaries[0].get("shape") if summaries else []})
    return "grid_states", out


def _run_grid_dynamics_dataset_build(step: Mapping[str, Any], artifacts: Mapping[str, Path], store: ArtifactStore) -> tuple[str, Path]:
    from neurobench.dynamics.datasets import build_dynamics_dataset
    from neurobench.manifests import load_json

    out_dir = store.artifact_path("dynamics", "dynamics_dataset.json").parent
    dataset = build_dynamics_dataset(manifest=load_json(_require_artifact(artifacts, "video_manifest", step)), grid_states_dir=_require_artifact(artifacts, "grid_states", step).parent, out_dir=out_dir, window_frames=int(step["params"].get("window_frames", 8)), prediction_horizon_frames=int(step["params"].get("prediction_horizon_frames", 1)), split_method=str(step["params"].get("split_method", "stratified_by_label")))
    out = out_dir / "dynamics_dataset.json"
    store.register_file(out, artifact_id="dynamics_dataset.v1", kind="dynamics_dataset", producer_stage=str(step["stage_id"]), schema="dynamics_dataset", summary={"window_count": dataset.get("extras", {}).get("window_count", 0), "split_unit": "video"})
    return "dynamics_dataset", out


def _run_grid_autoencoder_train(step: Mapping[str, Any], artifacts: Mapping[str, Path], store: ArtifactStore) -> tuple[str, Path]:
    from neurobench.dynamics.train import train_autoencoder
    from neurobench.manifests import load_json

    params = step["params"]
    out_dir = store.artifact_path("models", "autoencoder", "autoencoder_run.json").parent
    train_autoencoder(dataset=load_json(_require_artifact(artifacts, "dynamics_dataset", step)), out_dir=out_dir, latent_dim=int(params.get("latent_dim", 32)), epochs=int(params.get("epochs", 10)), batch_size=int(params.get("batch_size", 32)), learning_rate=float(params.get("learning_rate", 0.001)), seed=int(params.get("seed", 7)), device=str(params.get("device", "cpu")))
    out = out_dir / "autoencoder_run.json"
    store.register_file(out, artifact_id="autoencoder_run.v1", kind="autoencoder_run", producer_stage=str(step["stage_id"]), schema="autoencoder_run", summary={"latent_dim": int(params.get("latent_dim", 32)), "epochs": int(params.get("epochs", 10))})
    return "autoencoder_run", out


def _run_latent_rnn_train(step: Mapping[str, Any], artifacts: Mapping[str, Path], store: ArtifactStore) -> tuple[str, Path]:
    from neurobench.dynamics.train import train_latent_rnn
    from neurobench.manifests import load_json

    params = step["params"]
    out_dir = store.artifact_path("models", "latent_rnn", "latent_rnn_run.json").parent
    train_latent_rnn(dataset=load_json(_require_artifact(artifacts, "dynamics_dataset", step)), autoencoder_run=load_json(_require_artifact(artifacts, "autoencoder_run", step)), out_dir=out_dir, window_frames=int(params.get("window_frames", 8)), hidden_dim=int(params.get("hidden_dim", 64)), epochs=int(params.get("epochs", 10)), batch_size=int(params.get("batch_size", 32)), learning_rate=float(params.get("learning_rate", 0.001)), seed=int(params.get("seed", 7)), device=str(params.get("device", "cpu")))
    out = out_dir / "latent_rnn_run.json"
    store.register_file(out, artifact_id="latent_rnn_run.v1", kind="latent_rnn_run", producer_stage=str(step["stage_id"]), schema="latent_rnn_run", summary={"hidden_dim": int(params.get("hidden_dim", 64)), "epochs": int(params.get("epochs", 10))})
    return "latent_rnn_run", out


def _run_latent_classifier_train(step: Mapping[str, Any], artifacts: Mapping[str, Path], store: ArtifactStore) -> tuple[str, Path]:
    from neurobench.dynamics.classifier import train_latent_classifier
    from neurobench.manifests import load_json

    out_dir = store.artifact_path("classifier", "latent_classifier_run.json").parent
    train_latent_classifier(dataset=load_json(_require_artifact(artifacts, "dynamics_dataset", step)), autoencoder_run=load_json(_require_artifact(artifacts, "autoencoder_run", step)), out_dir=out_dir, classifier=str(step["params"].get("classifier", "logistic_regression")), split_method=str(step["params"].get("evaluation", "stratified_kfold")))
    out = out_dir / "latent_classifier_run.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    store.register_file(out, artifact_id="latent_classifier_run.v1", kind="latent_classifier_run", producer_stage=str(step["stage_id"]), schema="latent_classifier_run", summary={"accuracy": payload.get("metrics", {}).get("accuracy")})
    return "latent_classifier_run", out


def _require_artifact(artifacts: Mapping[str, Path], key: str, step: Mapping[str, Any]) -> Path:
    if key not in artifacts:
        raise ValueError(f"Pipeline step '{step['step_id']}' requires missing artifact '{key}'.")
    return artifacts[key]


def _resolved_device(store: ArtifactStore) -> str:
    return str(store.pipeline_run.environment.get("device") or "cpu")


def _record_input_checksum(store: ArtifactStore, record: Mapping[str, Any]) -> None:
    records = list(store.pipeline_run.extras.get("input_checksums") or [])
    path_id = record.get("path_id")
    if path_id is not None:
        records = [item for item in records if item.get("path_id") != path_id]
    records.append(dict(record))
    store.pipeline_run.extras["input_checksums"] = records


_STAGE_RUNNERS = {

    "video_manifest_build": _run_video_manifest_build,
    "template_build_from_video": _run_template_build_from_video,
    "template_register_video": _run_template_register_video,
    "apply_video_registration": _run_apply_video_registration,
    "grid_32x32_generate": _run_grid_32x32_generate,
    "grid_state_extract": _run_grid_state_extract,
    "grid_dynamics_dataset_build": _run_grid_dynamics_dataset_build,
    "grid_autoencoder_train": _run_grid_autoencoder_train,
    "latent_rnn_train": _run_latent_rnn_train,
    "latent_classifier_train": _run_latent_classifier_train,
    "source_video_import": _run_source_video_import,
    "temporal_highpass_gaussian": _run_temporal_highpass_gaussian,
    "event_preserving_noise_suppression": _run_event_preserving_noise_suppression,
    "robust_positive_local_z": _run_robust_positive_local_z,
    "adaptive_ewma_z": _run_adaptive_ewma_z,
    "spatial_gaussian": _run_spatial_gaussian,
    "rigid_shift_estimate": _run_rigid_shift_estimate,
    "gamma_cfar": _run_gamma_cfar,
    "adaptive_gamma_cfar": _run_gamma_cfar,
    "candidate_event_pipeline": _run_candidate_event_pipeline,
    "component_filter": _run_component_filter,
    "local_background_ring": _run_local_background_ring,
    "trace_event_scoring": _run_trace_event_scoring,
    "robust_kalman_positive_innovation": _run_robust_kalman_positive_innovation,
    "heuristic_priority_v1": _run_heuristic_priority_v1,
    "generate_neuron_review_app": _run_generate_neuron_review_app,
}
