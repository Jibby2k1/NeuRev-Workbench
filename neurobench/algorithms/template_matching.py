"""Template construction and rigid registration helpers for grid dynamics."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from neurobench.data.checksums import sha256_path
from neurobench.data.video import iter_video_chunks, load_video_array, video_metadata
from neurobench.workbench.intermediates import normalize_array_frame, write_png_gray8


def robust_frame_projection(
    video: Any,
    *,
    projection_kind: str = "mean",
    sample_stride: int = 1,
    normalize: bool = True,
) -> np.ndarray:
    """Return a finite 2-D projection from a frame-first video."""
    arr = np.asarray(video, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected frame-first [T,H,W] video, got shape {arr.shape}.")
    sampled = arr[:: max(1, int(sample_stride))]
    if sampled.size == 0:
        raise ValueError("Cannot project an empty video.")
    kind = projection_kind.replace("mean_after_outlier_rejection", "mean")
    if kind == "mean":
        projection = np.nanmean(sampled, axis=0)
    elif kind == "median":
        projection = np.nanmedian(sampled, axis=0)
    elif kind == "max":
        projection = np.nanmax(sampled, axis=0)
    else:
        raise ValueError(f"Unsupported projection_kind '{projection_kind}'.")
    projection = np.asarray(projection, dtype=np.float32)
    projection[~np.isfinite(projection)] = 0.0
    if normalize:
        projection = normalize_projection(projection)
    return projection.astype(np.float32, copy=False)


def normalize_projection(image: Any) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.float32)
    values = arr[finite]
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    if hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    out = np.clip(np.where(finite, arr, lo), lo, hi)
    return ((out - lo) / (hi - lo)).astype(np.float32, copy=False)


def score_frame_outliers(
    video: Any,
    *,
    method: str = "projection_residual_zscore",
    sample_stride: int = 1,
) -> dict[str, Any]:
    """Score frames by residual distance from a robust median projection."""
    arr = np.asarray(video, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected frame-first [T,H,W] video, got shape {arr.shape}.")
    if method != "projection_residual_zscore":
        raise ValueError(f"Unsupported outlier method '{method}'.")
    stride = max(1, int(sample_stride))
    sampled_indices = np.arange(0, arr.shape[0], stride, dtype=int)
    sampled = arr[sampled_indices]
    reference = robust_frame_projection(sampled, projection_kind="median", normalize=True)
    frame_means = arr.reshape(arr.shape[0], -1).mean(axis=1)
    mean_center = float(np.median(frame_means))
    mean_scale = float(1.4826 * np.median(np.abs(frame_means - mean_center)) + 1e-6)
    scores = []
    for index, frame in enumerate(arr):
        frame_norm = normalize_projection(frame)
        residual_score = float(np.mean(np.abs(frame_norm - reference)))
        intensity_score = float(abs(frame_means[index] - mean_center) / mean_scale)
        scores.append(residual_score + intensity_score)
    raw = np.asarray(scores, dtype=np.float32)
    center = float(np.median(raw))
    scale = float(1.4826 * np.median(np.abs(raw - center)) + 1e-6)
    z = (raw - center) / scale
    return {
        "method": method,
        "scores": [float(v) for v in raw],
        "z_scores": [float(v) for v in z],
        "center": center,
        "scale": scale,
        "sampled_frame_indices": [int(v) for v in sampled_indices],
    }


def select_template_frames(
    video: Any,
    *,
    max_outlier_fraction: float = 0.05,
    z_threshold: float = 3.5,
    method: str = "projection_residual_zscore",
) -> dict[str, Any]:
    scores = score_frame_outliers(video, method=method)
    z = np.asarray(scores["z_scores"], dtype=np.float32)
    frame_count = int(z.size)
    cap = int(np.floor(max(0.0, float(max_outlier_fraction)) * frame_count))
    candidates = np.where(z > float(z_threshold))[0]
    if cap >= 0 and candidates.size > cap:
        order = np.argsort(z[candidates])[::-1]
        candidates = candidates[order[:cap]]
    removed = sorted(int(v) for v in candidates)
    keep = np.ones(frame_count, dtype=bool)
    keep[removed] = False
    scores.update(
        {
            "z_threshold": float(z_threshold),
            "max_outlier_fraction": float(max_outlier_fraction),
            "removed_frame_indices": removed,
            "removed_fraction": float(len(removed) / max(frame_count, 1)),
            "kept_frame_indices": [int(v) for v in np.where(keep)[0]],
        }
    )
    return scores


def build_template_from_reference_video(
    video: Any,
    *,
    outlier_method: str = "projection_residual_zscore",
    max_outlier_fraction: float = 0.05,
    projection_kind: str = "mean_after_outlier_rejection",
    outlier_rejection: bool = True,
    z_threshold: float = 3.5,
) -> dict[str, Any]:
    arr = np.asarray(video, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected frame-first [T,H,W] video, got shape {arr.shape}.")
    if outlier_rejection:
        outliers = select_template_frames(
            arr,
            max_outlier_fraction=max_outlier_fraction,
            z_threshold=z_threshold,
            method=outlier_method,
        )
        kept = outliers["kept_frame_indices"] or list(range(arr.shape[0]))
        projection = robust_frame_projection(arr[kept], projection_kind="mean", normalize=False)
    else:
        outliers = {
            "method": outlier_method,
            "scores": [],
            "z_scores": [],
            "z_threshold": float(z_threshold),
            "max_outlier_fraction": float(max_outlier_fraction),
            "removed_frame_indices": [],
            "removed_fraction": 0.0,
            "kept_frame_indices": [int(v) for v in range(arr.shape[0])],
        }
        projection = robust_frame_projection(arr, projection_kind="mean", normalize=False)
    projection[~np.isfinite(projection)] = 0.0
    return {
        "projection": projection.astype(np.float32, copy=False),
        "outlier_rejection": outliers,
        "projection_kind": "mean_after_outlier_rejection" if outlier_rejection else projection_kind,
    }



def projection_from_video_path(
    video_path: str | Path,
    *,
    projection_kind: str = "mean",
    sample_stride: int = 1,
    normalize: bool = True,
    chunk_size_frames: int = 64,
    keep_frame_indices: list[int] | None = None,
) -> np.ndarray:
    """Compute a 2-D projection by streaming frame chunks from disk."""
    kind = projection_kind.replace("mean_after_outlier_rejection", "mean")
    if kind not in {"mean", "max"}:
        raise ValueError(f"Streaming projection supports mean or max, got '{projection_kind}'.")
    stride = max(1, int(sample_stride))
    keep_set = set(int(v) for v in keep_frame_indices) if keep_frame_indices is not None else None
    sum_image: np.ndarray | None = None
    count_image: np.ndarray | None = None
    max_image: np.ndarray | None = None
    selected = 0
    for chunk in iter_video_chunks(video_path, chunk_size=int(chunk_size_frames)):
        arr = np.asarray(chunk.data, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected frame-first [T,H,W] video chunk, got shape {arr.shape}.")
        indices = np.arange(chunk.start_frame, chunk.end_frame, dtype=int)
        mask = (indices % stride) == 0
        if keep_set is not None:
            mask &= np.asarray([int(index) in keep_set for index in indices], dtype=bool)
        if not np.any(mask):
            continue
        arr = arr[mask]
        selected += int(arr.shape[0])
        finite = np.isfinite(arr)
        if kind == "mean":
            if sum_image is None:
                sum_image = np.zeros(arr.shape[1:], dtype=np.float64)
                count_image = np.zeros(arr.shape[1:], dtype=np.int64)
            sum_image += np.where(finite, arr, 0.0).sum(axis=0, dtype=np.float64)
            count_image += finite.sum(axis=0, dtype=np.int64)
        else:
            chunk_max = np.nanmax(np.where(finite, arr, -np.inf), axis=0).astype(np.float32)
            max_image = chunk_max if max_image is None else np.maximum(max_image, chunk_max)
    if selected == 0:
        raise ValueError("Cannot project a video with no selected frames.")
    if kind == "mean":
        assert sum_image is not None and count_image is not None
        projection = (sum_image / np.maximum(count_image, 1)).astype(np.float32)
    else:
        assert max_image is not None
        projection = max_image.astype(np.float32)
        projection[~np.isfinite(projection)] = 0.0
    projection[~np.isfinite(projection)] = 0.0
    if normalize:
        projection = normalize_projection(projection)
    return projection.astype(np.float32, copy=False)


def score_frame_outliers_from_path(
    video_path: str | Path,
    *,
    method: str = "projection_residual_zscore",
    sample_stride: int = 1,
    chunk_size_frames: int = 64,
) -> dict[str, Any]:
    """Score frame outliers without holding the full video stack in memory."""
    if method != "projection_residual_zscore":
        raise ValueError(f"Unsupported outlier method '{method}'.")
    reference = projection_from_video_path(
        video_path,
        projection_kind="mean",
        sample_stride=sample_stride,
        normalize=True,
        chunk_size_frames=chunk_size_frames,
    )
    residual_scores: list[float] = []
    frame_means: list[float] = []
    frame_indices: list[int] = []
    for chunk in iter_video_chunks(video_path, chunk_size=int(chunk_size_frames)):
        arr = np.asarray(chunk.data, dtype=np.float32)
        for offset, frame in enumerate(arr):
            frame_indices.append(int(chunk.start_frame + offset))
            finite = np.isfinite(frame)
            frame_means.append(float(np.mean(frame[finite])) if np.any(finite) else 0.0)
            frame_norm = normalize_projection(frame)
            residual_scores.append(float(np.mean(np.abs(frame_norm - reference))))
    if not residual_scores:
        raise ValueError("Cannot score outliers for an empty video.")
    means = np.asarray(frame_means, dtype=np.float32)
    residual = np.asarray(residual_scores, dtype=np.float32)
    mean_center = float(np.median(means))
    mean_scale = float(1.4826 * np.median(np.abs(means - mean_center)) + 1e-6)
    raw = residual + np.abs(means - mean_center) / mean_scale
    center = float(np.median(raw))
    scale = float(1.4826 * np.median(np.abs(raw - center)) + 1e-6)
    z = (raw - center) / scale
    return {
        "method": method,
        "scores": [float(v) for v in raw],
        "z_scores": [float(v) for v in z],
        "center": center,
        "scale": scale,
        "sampled_frame_indices": [int(v) for v in range(0, len(raw), max(1, int(sample_stride)))],
        "frame_indices": frame_indices,
    }


def select_template_frames_from_path(
    video_path: str | Path,
    *,
    max_outlier_fraction: float = 0.05,
    z_threshold: float = 3.5,
    method: str = "projection_residual_zscore",
    chunk_size_frames: int = 64,
) -> dict[str, Any]:
    scores = score_frame_outliers_from_path(video_path, method=method, chunk_size_frames=chunk_size_frames)
    z = np.asarray(scores["z_scores"], dtype=np.float32)
    frame_indices = [int(v) for v in scores.get("frame_indices") or range(int(z.size))]
    frame_count = int(z.size)
    cap = int(np.floor(max(0.0, float(max_outlier_fraction)) * frame_count))
    candidates = np.where(z > float(z_threshold))[0]
    if cap >= 0 and candidates.size > cap:
        order = np.argsort(z[candidates])[::-1]
        candidates = candidates[order[:cap]]
    removed = sorted(int(frame_indices[int(v)]) for v in candidates)
    removed_set = set(removed)
    kept = [int(v) for v in frame_indices if int(v) not in removed_set]
    scores.update(
        {
            "z_threshold": float(z_threshold),
            "max_outlier_fraction": float(max_outlier_fraction),
            "removed_frame_indices": removed,
            "removed_fraction": float(len(removed) / max(frame_count, 1)),
            "kept_frame_indices": kept,
        }
    )
    return scores


def build_template_from_video_path(
    video_path: str | Path,
    *,
    outlier_method: str = "projection_residual_zscore",
    max_outlier_fraction: float = 0.05,
    projection_kind: str = "mean_after_outlier_rejection",
    outlier_rejection: bool = True,
    z_threshold: float = 3.5,
    chunk_size_frames: int = 64,
) -> dict[str, Any]:
    if outlier_rejection:
        outliers = select_template_frames_from_path(
            video_path,
            max_outlier_fraction=max_outlier_fraction,
            z_threshold=z_threshold,
            method=outlier_method,
            chunk_size_frames=chunk_size_frames,
        )
        kept = outliers["kept_frame_indices"]
        projection = projection_from_video_path(
            video_path,
            projection_kind="mean",
            normalize=False,
            chunk_size_frames=chunk_size_frames,
            keep_frame_indices=kept,
        )
    else:
        meta = video_metadata(video_path)
        outliers = {
            "method": outlier_method,
            "scores": [],
            "z_scores": [],
            "z_threshold": float(z_threshold),
            "max_outlier_fraction": float(max_outlier_fraction),
            "removed_frame_indices": [],
            "removed_fraction": 0.0,
            "kept_frame_indices": [int(v) for v in range(int(meta["frames"]))],
        }
        projection = projection_from_video_path(video_path, projection_kind="mean", normalize=False, chunk_size_frames=chunk_size_frames)
    projection[~np.isfinite(projection)] = 0.0
    return {
        "projection": projection.astype(np.float32, copy=False),
        "outlier_rejection": outliers,
        "projection_kind": "mean_after_outlier_rejection" if outlier_rejection else projection_kind,
    }

def write_template_artifacts(
    *,
    video_path: str | Path,
    source_video_id: str,
    out_dir: str | Path,
    template_id: str | None = None,
    outlier_rejection: bool = True,
    max_outlier_fraction: float = 0.05,
    z_threshold: float = 3.5,
    chunk_size_frames: int = 64,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = build_template_from_video_path(
        video_path,
        outlier_rejection=outlier_rejection,
        max_outlier_fraction=max_outlier_fraction,
        z_threshold=z_threshold,
        chunk_size_frames=chunk_size_frames,
    )
    projection = result["projection"]
    projection_path = out / "template_projection.npy"
    preview_path = out / "template_projection.png"
    score_tsv = out / "outlier_frame_scores.tsv"
    score_png = out / "outlier_frame_scores.png"
    np.save(projection_path, projection)
    write_gray_preview(preview_path, projection)
    write_outlier_scores(score_tsv, score_png, result["outlier_rejection"])
    template_id = template_id or f"template_from_{source_video_id}_v1"
    spec = {
        "schema_version": 1,
        "template_id": template_id,
        "source_video_id": source_video_id,
        "source_video_path": str(video_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "coordinate_system": {"height": int(projection.shape[0]), "width": int(projection.shape[1]), "origin": "top_left", "units": "px"},
        "projection": {
            "kind": result["projection_kind"],
            "path": str(projection_path),
            "preview_png": str(preview_path),
            "dtype": str(projection.dtype),
            "sha256": sha256_path(projection_path),
        },
        "outlier_rejection": {
            "enabled": bool(outlier_rejection),
            "method": "projection_residual_zscore",
            "max_outlier_fraction": float(max_outlier_fraction),
            "z_threshold": float(z_threshold),
            "removed_frame_indices": result["outlier_rejection"].get("removed_frame_indices", []),
            "removed_fraction": result["outlier_rejection"].get("removed_fraction", 0.0),
            "score_tsv": str(score_tsv),
            "score_png": str(score_png),
        },
        "notes": "Initial anatomical template from one reference video mean projection.",
        "extras": {"chunk_size_frames": int(chunk_size_frames)},
    }
    spec_path = out / "template_spec.json"
    write_json(spec_path, spec)
    return spec

def estimate_rigid_registration(
    source_projection: Any,
    template_projection: Any,
    *,
    transform_model: str = "rigid",
    rotation_range_deg: tuple[float, float] = (-10.0, 10.0),
    rotation_step_deg: float = 0.5,
    allow_uniform_scale: bool = False,
    scale_range: tuple[float, float] = (1.0, 1.0),
    scale_step: float = 0.01,
) -> dict[str, Any]:
    """Estimate a source-to-template translation/rotation transform."""
    if transform_model not in {"translation", "rigid", "similarity"}:
        raise ValueError("transform_model must be translation, rigid, or similarity")
    template = normalize_projection(template_projection)
    source = normalize_projection(source_projection)
    if template.shape != source.shape:
        source = _resize_to_shape(source, template.shape)
    angles = [0.0] if transform_model == "translation" else _float_range(rotation_range_deg[0], rotation_range_deg[1], rotation_step_deg)
    scales = [1.0]
    if allow_uniform_scale or transform_model == "similarity":
        scales = _float_range(scale_range[0], scale_range[1], scale_step)
    best: dict[str, Any] | None = None
    for scale in scales:
        for angle in angles:
            rotated = _warp_similarity(source, template.shape, rotation_deg=float(angle), scale=float(scale))
            dy, dx = _phase_shift(template, rotated)
            aligned = _shift_image(rotated, dy=dy, dx=dx)
            score = _ncc(template, aligned)
            if best is None or score > best["score"]:
                best = {"rotation_deg": float(angle), "scale": float(scale), "dy": float(dy), "dx": float(dx), "score": float(score), "registered_projection": aligned}
    assert best is not None
    min_angle, max_angle = float(rotation_range_deg[0]), float(rotation_range_deg[1])
    boundary = abs(best["rotation_deg"] - min_angle) < 1e-9 or abs(best["rotation_deg"] - max_angle) < 1e-9
    warnings: list[str] = []
    confidence = "ok"
    if best["score"] < 0.35:
        confidence = "low"
        warnings.append("low registration correlation score")
    if boundary and transform_model != "translation":
        warnings.append("best rotation is at search boundary")
    blank_fraction = float(np.mean(best["registered_projection"] <= 1e-6))
    if blank_fraction > 0.35:
        warnings.append("large blank fraction after warp")
    matrix = transform_matrix(best["rotation_deg"], best["dx"], best["dy"], best["scale"])
    return {
        "transform": {
            "model": transform_model,
            "matrix_3x3": matrix,
            "rotation_deg": best["rotation_deg"],
            "translation_px": [best["dx"], best["dy"]],
            "scale": best["scale"],
        },
        "score": {"normalized_cross_correlation": best["score"], "confidence": confidence},
        "qc": {"warnings": warnings, "blank_fraction_after_warp": blank_fraction, "best_angle_at_boundary": boundary},
        "registered_projection": best["registered_projection"].astype(np.float32),
    }


def apply_registration_transform(video: Any, transform: Mapping[str, Any], *, output_shape: tuple[int, int], output_dtype: str = "float32") -> np.ndarray:
    arr = np.asarray(video, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected frame-first [T,H,W] video, got shape {arr.shape}.")
    rotation = float(transform.get("rotation_deg", 0.0))
    scale = float(transform.get("scale", 1.0))
    translation = transform.get("translation_px") or [0.0, 0.0]
    dx, dy = float(translation[0]), float(translation[1])
    out = np.empty((arr.shape[0], int(output_shape[0]), int(output_shape[1])), dtype=np.float32)
    for i, frame in enumerate(arr):
        work = _resize_to_shape(frame, output_shape) if frame.shape != output_shape else np.asarray(frame, dtype=np.float32)
        out[i] = _warp_similarity(work, output_shape, rotation_deg=rotation, scale=scale, dx=dx, dy=dy)
    return out.astype(output_dtype, copy=False)


def write_registration_artifacts(
    *,
    video_path: str | Path,
    video_id: str,
    template_spec: Mapping[str, Any],
    out_dir: str | Path,
    transform_model: str = "rigid",
    rotation_range_deg: tuple[float, float] = (-10.0, 10.0),
    rotation_step_deg: float = 0.5,
    allow_uniform_scale: bool = False,
    chunk_size_frames: int = 64,
) -> dict[str, Any]:
    out = Path(out_dir) / video_id
    out.mkdir(parents=True, exist_ok=True)
    source_projection = projection_from_video_path(
        video_path,
        projection_kind="mean",
        normalize=False,
        chunk_size_frames=chunk_size_frames,
    )
    template_path = Path(str(template_spec["projection"]["path"]))
    template_projection = np.load(template_path)
    result = estimate_rigid_registration(
        source_projection,
        template_projection,
        transform_model=transform_model,
        rotation_range_deg=rotation_range_deg,
        rotation_step_deg=rotation_step_deg,
        allow_uniform_scale=allow_uniform_scale,
    )
    source_path = out / "source_projection.npy"
    source_png = out / "source_projection.png"
    registered_png = out / "registered_projection.png"
    overlay_png = out / "overlay_before_after.png"
    residual_png = out / "residual.png"
    np.save(source_path, source_projection.astype(np.float32))
    write_gray_preview(source_png, source_projection)
    write_gray_preview(registered_png, result["registered_projection"])
    write_overlay_preview(overlay_png, template_projection, source_projection, result["registered_projection"])
    write_gray_preview(residual_png, np.abs(normalize_projection(template_projection) - normalize_projection(result["registered_projection"])))
    payload = {
        "schema_version": 1,
        "video_id": video_id,
        "template_id": str(template_spec.get("template_id")),
        "registration_scope": "per_video",
        "source_projection": {"kind": "mean_after_outlier_rejection", "path": str(source_path), "preview_png": str(source_png)},
        "transform": result["transform"],
        "score": result["score"],
        "qc": result["qc"],
        "artifacts": {"registered_projection_png": str(registered_png), "overlay_before_after_png": str(overlay_png), "residual_png": str(residual_png)},
        "extras": {"chunk_size_frames": int(chunk_size_frames)},
    }
    write_json(out / "registration_result.json", payload)
    return payload

def write_registered_video_artifacts(
    *,
    video_path: str | Path,
    registration_result: Mapping[str, Any],
    template_spec: Mapping[str, Any],
    out_dir: str | Path,
    output_dtype: str = "float32",
    chunk_size_frames: int = 64,
) -> dict[str, Any]:
    video_id = str(registration_result["video_id"])
    out = Path(out_dir) / video_id
    out.mkdir(parents=True, exist_ok=True)
    coord = template_spec["coordinate_system"]
    output_shape = (int(coord["height"]), int(coord["width"]))
    meta = video_metadata(video_path)
    frame_count = int(meta["frames"])
    if frame_count <= 0:
        raise ValueError(f"Cannot register empty video: {video_path}")
    video_path_out = out / "registered_video.npy"
    preview = out / "registered_projection.png"
    summary_path = out / "registered_video_summary.json"
    dtype = np.dtype(output_dtype)
    registered_map = np.lib.format.open_memmap(video_path_out, mode="w+", dtype=dtype, shape=(frame_count, output_shape[0], output_shape[1]))
    projection_sum = np.zeros(output_shape, dtype=np.float64)
    finite_count = 0
    nonzero_count = 0
    total_values = frame_count * output_shape[0] * output_shape[1]
    for chunk in iter_video_chunks(video_path, chunk_size=int(chunk_size_frames)):
        registered = apply_registration_transform(
            chunk.data,
            registration_result["transform"],
            output_shape=output_shape,
            output_dtype=str(dtype),
        )
        registered_map[chunk.start_frame : chunk.end_frame] = registered
        finite = np.isfinite(registered)
        projection_sum += np.where(finite, registered, 0.0).sum(axis=0, dtype=np.float64)
        finite_count += int(finite.sum())
        nonzero_count += int(np.count_nonzero(registered))
    registered_map.flush()
    del registered_map
    projection = (projection_sum / max(frame_count, 1)).astype(np.float32)
    write_gray_preview(preview, projection)
    summary = {
        "schema_version": 1,
        "video_id": video_id,
        "path": str(video_path_out),
        "preview_png": str(preview),
        "shape": [frame_count, int(output_shape[0]), int(output_shape[1])],
        "dtype": str(dtype),
        "finite_fraction": float(finite_count / max(total_values, 1)),
        "nonzero_fraction": float(nonzero_count / max(total_values, 1)),
        "source_video": str(video_path),
        "extras": {"chunk_size_frames": int(chunk_size_frames)},
    }
    write_json(summary_path, summary)
    return summary

def write_gray_preview(path: str | Path, image: Any) -> None:
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D preview image, got {arr.shape}.")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_png_gray8(out, int(arr.shape[1]), int(arr.shape[0]), normalize_array_frame(arr))


def write_overlay_preview(path: str | Path, template: Any, source: Any, registered: Any) -> None:
    t = normalize_projection(template)
    s = normalize_projection(_resize_to_shape(source, t.shape))
    r = normalize_projection(_resize_to_shape(registered, t.shape))
    canvas = np.concatenate([s, r, np.abs(t - r)], axis=1)
    write_gray_preview(path, canvas)


def write_outlier_scores(tsv_path: Path, png_path: Path, report: Mapping[str, Any]) -> None:
    scores = list(report.get("scores") or [])
    z_scores = list(report.get("z_scores") or [])
    removed = set(int(v) for v in report.get("removed_frame_indices") or [])
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["frame\tscore\tz_score\tremoved\n"]
    for i, score in enumerate(scores):
        z = z_scores[i] if i < len(z_scores) else 0.0
        lines.append(f"{i}\t{float(score):.8g}\t{float(z):.8g}\t{1 if i in removed else 0}\n")
    tsv_path.write_text("".join(lines), encoding="utf-8")
    if scores:
        arr = np.asarray(z_scores or scores, dtype=np.float32).reshape(1, -1)
        arr = np.repeat(arr, 48, axis=0)
    else:
        arr = np.zeros((16, 16), dtype=np.float32)
    write_gray_preview(png_path, arr)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    import json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def transform_matrix(rotation_deg: float, dx: float, dy: float, scale: float) -> list[list[float]]:
    theta = np.deg2rad(rotation_deg)
    c = float(np.cos(theta) * scale)
    s = float(np.sin(theta) * scale)
    return [[c, -s, float(dx)], [s, c, float(dy)], [0.0, 0.0, 1.0]]


def _phase_shift(template: np.ndarray, source: np.ndarray) -> tuple[float, float]:
    f0 = np.fft.fftn(np.asarray(template, dtype=np.float32))
    f1 = np.fft.fftn(np.asarray(source, dtype=np.float32))
    cross = f0 * f1.conjugate()
    cross /= np.maximum(np.abs(cross), 1e-9)
    corr = np.fft.ifftn(cross)
    maxima = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    shifts = np.array(maxima, dtype=float)
    for dim, size in enumerate(template.shape):
        if shifts[dim] > size // 2:
            shifts[dim] -= size
    return float(shifts[0]), float(shifts[1])


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32) - float(np.mean(a))
    bb = np.asarray(b, dtype=np.float32) - float(np.mean(b))
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)) + 1e-9)
    return float(np.sum(aa * bb) / denom)


def _resize_to_shape(arr: Any, shape: tuple[int, int]) -> np.ndarray:
    src = np.asarray(arr, dtype=np.float32)
    if src.shape == tuple(shape):
        return src
    out_h, out_w = int(shape[0]), int(shape[1])
    if out_h <= 0 or out_w <= 0:
        raise ValueError("output shape must be positive")
    if src.size == 0:
        return np.zeros((out_h, out_w), dtype=np.float32)
    y = np.linspace(0.0, max(src.shape[0] - 1, 0), out_h, dtype=np.float32)
    x = np.linspace(0.0, max(src.shape[1] - 1, 0), out_w, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return _sample_bilinear(src, yy, xx)


def _scale_about_center(arr: Any, scale: float, output_shape: tuple[int, int]) -> np.ndarray:
    return _warp_similarity(arr, output_shape, scale=float(scale))


def _shift_image(arr: Any, *, dy: float, dx: float) -> np.ndarray:
    src = np.asarray(arr, dtype=np.float32)
    yy, xx = np.indices(src.shape, dtype=np.float32)
    return _sample_bilinear(src, yy - float(dy), xx - float(dx))


def _warp_similarity(
    arr: Any,
    output_shape: tuple[int, int],
    *,
    rotation_deg: float = 0.0,
    scale: float = 1.0,
    dx: float = 0.0,
    dy: float = 0.0,
) -> np.ndarray:
    src = np.asarray(arr, dtype=np.float32)
    out_h, out_w = int(output_shape[0]), int(output_shape[1])
    if src.ndim != 2:
        raise ValueError(f"Expected 2-D image for warp, got shape {src.shape}.")
    if out_h <= 0 or out_w <= 0:
        raise ValueError("output shape must be positive")
    scale = float(scale)
    if abs(scale) < 1e-9:
        raise ValueError("scale must be nonzero")
    yy, xx = np.indices((out_h, out_w), dtype=np.float32)
    src_cy = (src.shape[0] - 1) / 2.0
    src_cx = (src.shape[1] - 1) / 2.0
    out_cy = (out_h - 1) / 2.0
    out_cx = (out_w - 1) / 2.0
    theta = np.deg2rad(float(rotation_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    x_rel = xx - out_cx - float(dx)
    y_rel = yy - out_cy - float(dy)
    src_x = (c * x_rel + s * y_rel) / scale + src_cx
    src_y = (-s * x_rel + c * y_rel) / scale + src_cy
    return _sample_bilinear(src, src_y, src_x)


def _sample_bilinear(src: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float32)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1
    valid = (y0 >= 0) & (x0 >= 0) & (y1 < src.shape[0]) & (x1 < src.shape[1])
    y0c = np.clip(y0, 0, max(src.shape[0] - 1, 0))
    y1c = np.clip(y1, 0, max(src.shape[0] - 1, 0))
    x0c = np.clip(x0, 0, max(src.shape[1] - 1, 0))
    x1c = np.clip(x1, 0, max(src.shape[1] - 1, 0))
    wy = y - y0
    wx = x - x0
    out = (
        src[y0c, x0c] * (1.0 - wy) * (1.0 - wx)
        + src[y0c, x1c] * (1.0 - wy) * wx
        + src[y1c, x0c] * wy * (1.0 - wx)
        + src[y1c, x1c] * wy * wx
    )
    return np.where(valid, out, 0.0).astype(np.float32, copy=False)


def _float_range(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    count = int(np.floor((float(stop) - float(start)) / float(step))) + 1
    values = [float(start) + i * float(step) for i in range(max(1, count))]
    if values[-1] < float(stop) - 1e-9:
        values.append(float(stop))
    return [round(v, 10) for v in values]
