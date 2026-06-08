"""Rectangular template-grid generation and state extraction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from neurobench.algorithms.template_matching import apply_registration_transform, write_gray_preview
from neurobench.data.video import iter_video_chunks, video_metadata


def generate_grid_spec(
    *,
    template_id: str,
    height: int,
    width: int,
    rows: int = 32,
    cols: int = 32,
    grid_id: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic rectangular grid spec covering the full image."""
    rows = int(rows)
    cols = int(cols)
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    y_edges = np.linspace(0, int(height), rows + 1)
    x_edges = np.linspace(0, int(width), cols + 1)
    y_edges_i = np.round(y_edges).astype(int)
    x_edges_i = np.round(x_edges).astype(int)
    regions: list[dict[str, Any]] = []
    for row in range(rows):
        for col in range(cols):
            x0, x1 = int(x_edges_i[col]), int(x_edges_i[col + 1])
            y0, y1 = int(y_edges_i[row]), int(y_edges_i[row + 1])
            regions.append(
                {
                    "region_id": region_id(row, col),
                    "row": row,
                    "col": col,
                    "bbox": [x0, y0, x1, y1],
                    "center": [float((x0 + x1) / 2.0), float((y0 + y1) / 2.0)],
                    "pixel_count": int(max(0, x1 - x0) * max(0, y1 - y0)),
                    "anatomy_fraction": None,
                    "anatomy_status": "unknown",
                }
            )
    return {
        "schema_version": 1,
        "grid_id": grid_id or f"grid_{rows}x{cols}_{template_id}",
        "template_id": template_id,
        "rows": rows,
        "cols": cols,
        "region_count": rows * cols,
        "coordinate_system": {"height": int(height), "width": int(width), "origin": "top_left", "units": "px"},
        "bounds": "full_template_image",
        "cell_policy": "rectangular_image_coordinates",
        "regions": regions,
        "extras": {},
    }


def region_id(row: int, col: int) -> str:
    return f"R{int(row):02d}C{int(col):02d}"


def write_grid_spec_artifacts(
    *,
    template_spec: Mapping[str, Any],
    out_path: str | Path,
    rows: int = 32,
    cols: int = 32,
) -> dict[str, Any]:
    coord = template_spec["coordinate_system"]
    spec = generate_grid_spec(
        template_id=str(template_spec["template_id"]),
        height=int(coord["height"]),
        width=int(coord["width"]),
        rows=rows,
        cols=cols,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    preview = out.with_name("grid_overlay.png")
    projection_path = Path(str(template_spec.get("projection", {}).get("path", "")))
    if projection_path.is_file():
        projection = np.load(projection_path)
    else:
        projection = np.zeros((int(coord["height"]), int(coord["width"])), dtype=np.float32)
    write_grid_overlay(preview, projection, spec)
    spec["extras"]["overlay_png"] = str(preview)
    out.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec


def extract_grid_states(
    video: Any,
    grid_spec: Mapping[str, Any],
    *,
    features: Sequence[str] = ("mean_intensity",),
    normalization: str = "per_video_robust_percentile",
) -> dict[str, Any]:
    arr = np.asarray(video, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected frame-first [T,H,W] video, got shape {arr.shape}.")
    rows = int(grid_spec["rows"])
    cols = int(grid_spec["cols"])
    if tuple(arr.shape[1:]) != (int(grid_spec["coordinate_system"]["height"]), int(grid_spec["coordinate_system"]["width"])):
        raise ValueError("registered video shape does not match grid coordinate system")
    work = normalize_video(arr, normalization)
    feature_names = list(features)
    if feature_names != ["mean_intensity"]:
        raise ValueError("MVP grid extraction currently supports only mean_intensity")
    grid = np.zeros((arr.shape[0], rows, cols, 1), dtype=np.float32)
    region_ids: list[str] = []
    for region in grid_spec["regions"]:
        row = int(region["row"])
        col = int(region["col"])
        x0, y0, x1, y1 = [int(v) for v in region["bbox"]]
        if x1 <= x0 or y1 <= y0:
            values = np.zeros(arr.shape[0], dtype=np.float32)
        else:
            values = work[:, y0:y1, x0:x1].mean(axis=(1, 2))
        grid[:, row, col, 0] = values.astype(np.float32)
        region_ids.append(str(region["region_id"]))
    flat = grid.reshape((arr.shape[0], rows * cols, 1)).astype(np.float32, copy=False)
    return {"grid_state": grid, "flat_state": flat, "region_ids": np.asarray(region_ids), "feature_names": np.asarray(feature_names), "normalization": normalization}


def normalize_video(video: np.ndarray, normalization: str) -> np.ndarray:
    if normalization in {"none", "raw"}:
        return np.asarray(video, dtype=np.float32)
    if normalization != "per_video_robust_percentile":
        raise ValueError(f"Unsupported normalization '{normalization}'.")
    arr = np.asarray(video, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.float32)
    values = arr[finite]
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return ((np.clip(np.where(finite, arr, lo), lo, hi) - lo) / (hi - lo)).astype(np.float32)



def estimate_video_normalization_bounds(
    video_path: str | Path,
    normalization: str,
    *,
    chunk_size_frames: int = 64,
    max_sample_values: int = 1_000_000,
) -> tuple[float, float] | None:
    """Estimate robust normalization bounds from streamed video samples."""
    if normalization in {"none", "raw"}:
        return None
    if normalization != "per_video_robust_percentile":
        raise ValueError(f"Unsupported normalization '{normalization}'.")
    samples: list[np.ndarray] = []
    remaining = max(1, int(max_sample_values))
    for chunk in iter_video_chunks(video_path, chunk_size=int(chunk_size_frames)):
        values = np.asarray(chunk.data, dtype=np.float32).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        take = min(values.size, max(1, min(remaining, max_sample_values // 32 or 1)))
        stride = max(1, values.size // take)
        sample = values[::stride][:take].astype(np.float32, copy=False)
        samples.append(sample)
        remaining -= int(sample.size)
        if remaining <= 0:
            break
    if not samples:
        return (0.0, 0.0)
    values = np.concatenate(samples)
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    if hi <= lo:
        return (0.0, 0.0)
    return (lo, hi)


def normalize_video_chunk(video: np.ndarray, normalization: str, bounds: tuple[float, float] | None) -> np.ndarray:
    if normalization in {"none", "raw"}:
        return np.asarray(video, dtype=np.float32)
    if normalization != "per_video_robust_percentile":
        raise ValueError(f"Unsupported normalization '{normalization}'.")
    arr = np.asarray(video, dtype=np.float32)
    if bounds is None:
        return normalize_video(arr, normalization)
    lo, hi = float(bounds[0]), float(bounds[1])
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    finite_fill = np.where(np.isfinite(arr), arr, lo)
    return ((np.clip(finite_fill, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def pool_grid_chunk(
    video_chunk: Any,
    grid_spec: Mapping[str, Any],
    *,
    features: Sequence[str] = ("mean_intensity",),
) -> dict[str, Any]:
    return extract_grid_states(video_chunk, grid_spec, features=features, normalization="none")

def write_grid_state_artifacts(
    *,
    registered_video_path: str | Path,
    grid_spec: Mapping[str, Any],
    out_dir: str | Path,
    video_id: str,
    label: str,
    features: Sequence[str] = ("mean_intensity",),
    normalization: str = "per_video_robust_percentile",
    frame_rate_hz: float | None = None,
    chunk_size_frames: int = 64,
    max_grid_state_bytes: int | None = 1_000_000_000,
) -> dict[str, Any]:
    out = Path(out_dir) / video_id
    out.mkdir(parents=True, exist_ok=True)
    meta = video_metadata(registered_video_path)
    frame_count = int(meta["frames"])
    rows = int(grid_spec["rows"])
    cols = int(grid_spec["cols"])
    feature_names = list(features)
    if feature_names != ["mean_intensity"]:
        raise ValueError("MVP grid extraction currently supports only mean_intensity")
    if tuple(int(v) for v in meta["shape"][1:]) != (int(grid_spec["coordinate_system"]["height"]), int(grid_spec["coordinate_system"]["width"])):
        raise ValueError("registered video shape does not match grid coordinate system")
    grid_bytes = frame_count * rows * cols * len(feature_names) * np.dtype(np.float32).itemsize
    if max_grid_state_bytes is not None and grid_bytes * 2 > int(max_grid_state_bytes):
        raise RuntimeError(
            f"Refusing to materialize grid states for {video_id}: estimated grid_state+flat_state payload "
            f"is {grid_bytes * 2} bytes, above the safety limit {int(max_grid_state_bytes)}. "
            "Increase --max-grid-state-bytes only after confirming disk/RAM capacity."
        )
    bounds = estimate_video_normalization_bounds(
        registered_video_path,
        normalization,
        chunk_size_frames=chunk_size_frames,
    )
    grid_state = np.empty((frame_count, rows, cols, len(feature_names)), dtype=np.float32)
    region_ids: np.ndarray | None = None
    for chunk in iter_video_chunks(registered_video_path, chunk_size=int(chunk_size_frames)):
        chunk_arr = np.asarray(chunk.data, dtype=np.float32)
        chunk_norm = normalize_video_chunk(chunk_arr, normalization, bounds)
        states = pool_grid_chunk(chunk_norm, grid_spec, features=feature_names)
        grid_state[chunk.start_frame : chunk.end_frame] = states["grid_state"].astype(np.float32, copy=False)
        if region_ids is None:
            region_ids = states["region_ids"]
    if region_ids is None:
        region_ids = np.asarray([str(region["region_id"]) for region in grid_spec["regions"]])
    flat_state = grid_state.reshape((frame_count, rows * cols, len(feature_names))).astype(np.float32, copy=False)
    npz_path = out / "grid_states.npz"
    arrays = {
        "grid_state": grid_state.astype(np.float32, copy=False),
        "flat_state": flat_state,
        "region_ids": region_ids.astype("U8"),
        "feature_names": np.asarray(feature_names).astype("U32"),
        "grid_id": np.asarray(str(grid_spec.get("grid_id") or f"grid_{rows}x{cols}")),
        "video_id": np.asarray(video_id),
        "label": np.asarray(label),
        "normalization": np.asarray(normalization),
        "source_registered_video": np.asarray(str(registered_video_path)),
    }
    if frame_rate_hz:
        arrays["time_sec"] = (np.arange(frame_count, dtype=np.float32) / float(frame_rate_hz)).astype(np.float32)
    np.savez(npz_path, **arrays)
    region_features_path = out / "region_features.tsv"
    write_region_features_tsv(region_features_path, flat_state, region_ids)
    preview_path = out / "grid_preview.png"
    trace_path = out / "grid_trace_summary.png"
    write_gray_preview(preview_path, grid_state.mean(axis=0)[:, :, 0] if frame_count else np.zeros((rows, cols), dtype=np.float32))
    write_trace_summary(trace_path, flat_state[:, :, 0])
    summary = {
        "schema_version": 1,
        "video_id": video_id,
        "label": label,
        "grid_states_npz": str(npz_path),
        "region_features_tsv": str(region_features_path),
        "grid_preview_png": str(preview_path),
        "grid_trace_summary_png": str(trace_path),
        "grid_id": str(grid_spec.get("grid_id") or f"grid_{rows}x{cols}"),
        "shape": [int(v) for v in grid_state.shape],
        "flat_shape": [int(v) for v in flat_state.shape],
        "region_count": int(flat_state.shape[1]),
        "feature_names": [str(v) for v in feature_names],
        "normalization": normalization,
        "nonfinite_fraction": float(np.mean(~np.isfinite(grid_state))) if grid_state.size else 0.0,
        "source_registered_video": str(registered_video_path),
        "extras": {"chunk_size_frames": int(chunk_size_frames), "normalization_bounds": list(bounds) if bounds is not None else None},
    }
    summary_path = out / "region_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def write_registered_grid_state_artifacts(
    *,
    video_path: str | Path,
    registration_result: Mapping[str, Any],
    grid_spec: Mapping[str, Any],
    out_dir: str | Path,
    video_id: str,
    label: str,
    features: Sequence[str] = ("mean_intensity",),
    normalization: str = "per_video_robust_percentile",
    frame_rate_hz: float | None = None,
    chunk_size_frames: int = 64,
    max_grid_state_bytes: int | None = 1_000_000_000,
    device: str = "auto",
) -> dict[str, Any]:
    """Apply registration in memory and write grid states without registered video files."""
    out = Path(out_dir) / video_id
    out.mkdir(parents=True, exist_ok=True)
    meta = video_metadata(video_path)
    frame_count = int(meta["frames"])
    rows = int(grid_spec["rows"])
    cols = int(grid_spec["cols"])
    feature_names = list(features)
    if feature_names != ["mean_intensity"]:
        raise ValueError("MVP grid extraction currently supports only mean_intensity")
    coord = grid_spec["coordinate_system"]
    output_shape = (int(coord["height"]), int(coord["width"]))
    grid_bytes = frame_count * rows * cols * len(feature_names) * np.dtype(np.float32).itemsize
    if max_grid_state_bytes is not None and grid_bytes * 2 > int(max_grid_state_bytes):
        raise RuntimeError(
            f"Refusing to materialize grid states for {video_id}: estimated grid_state+flat_state payload "
            f"is {grid_bytes * 2} bytes, above the safety limit {int(max_grid_state_bytes)}. "
            "Increase --max-grid-state-bytes only after confirming disk/RAM capacity."
        )
    transform = registration_result.get("transform") or {}
    resolved_device = _resolve_registration_device(device)
    bounds = estimate_registered_video_normalization_bounds(
        video_path,
        transform,
        output_shape=output_shape,
        normalization=normalization,
        chunk_size_frames=chunk_size_frames,
        device=device,
    )
    grid_state = np.empty((frame_count, rows, cols, len(feature_names)), dtype=np.float32)
    region_ids: np.ndarray | None = None
    finite_count = 0
    total_values = frame_count * output_shape[0] * output_shape[1]
    for chunk in iter_video_chunks(video_path, chunk_size=int(chunk_size_frames)):
        registered = apply_registration_transform_chunk(
            chunk.data,
            transform,
            output_shape=output_shape,
            device=device,
        )
        finite_count += int(np.isfinite(registered).sum())
        chunk_norm = normalize_video_chunk(registered, normalization, bounds)
        states = pool_grid_chunk(chunk_norm, grid_spec, features=feature_names)
        grid_state[chunk.start_frame : chunk.end_frame] = states["grid_state"].astype(np.float32, copy=False)
        if region_ids is None:
            region_ids = states["region_ids"]
    if region_ids is None:
        region_ids = np.asarray([str(region["region_id"]) for region in grid_spec["regions"]])
    flat_state = grid_state.reshape((frame_count, rows * cols, len(feature_names))).astype(np.float32, copy=False)
    npz_path = out / "grid_states.npz"
    arrays = {
        "grid_state": grid_state.astype(np.float32, copy=False),
        "flat_state": flat_state,
        "region_ids": region_ids.astype("U8"),
        "feature_names": np.asarray(feature_names).astype("U32"),
        "grid_id": np.asarray(str(grid_spec.get("grid_id") or f"grid_{rows}x{cols}")),
        "video_id": np.asarray(video_id),
        "label": np.asarray(label),
        "normalization": np.asarray(normalization),
        "source_video": np.asarray(str(video_path)),
        "registration_result": np.asarray(str(registration_result.get("video_id") or video_id)),
    }
    if frame_rate_hz:
        arrays["time_sec"] = (np.arange(frame_count, dtype=np.float32) / float(frame_rate_hz)).astype(np.float32)
    np.savez(npz_path, **arrays)
    region_features_path = out / "region_features.tsv"
    write_region_features_tsv(region_features_path, flat_state, region_ids)
    preview_path = out / "grid_preview.png"
    trace_path = out / "grid_trace_summary.png"
    write_gray_preview(preview_path, grid_state.mean(axis=0)[:, :, 0] if frame_count else np.zeros((rows, cols), dtype=np.float32))
    write_trace_summary(trace_path, flat_state[:, :, 0])
    summary = {
        "schema_version": 1,
        "video_id": video_id,
        "label": label,
        "grid_states_npz": str(npz_path),
        "region_features_tsv": str(region_features_path),
        "grid_preview_png": str(preview_path),
        "grid_trace_summary_png": str(trace_path),
        "grid_id": str(grid_spec.get("grid_id") or f"grid_{rows}x{cols}"),
        "shape": [int(v) for v in grid_state.shape],
        "flat_shape": [int(v) for v in flat_state.shape],
        "region_count": int(flat_state.shape[1]),
        "feature_names": [str(v) for v in feature_names],
        "normalization": normalization,
        "nonfinite_fraction": float(np.mean(~np.isfinite(grid_state))) if grid_state.size else 0.0,
        "source_video": str(video_path),
        "registration_video_id": str(registration_result.get("video_id") or video_id),
        "registered_video_materialized": False,
        "registered_finite_fraction": float(finite_count / max(total_values, 1)),
        "extras": {
            "chunk_size_frames": int(chunk_size_frames),
            "normalization_bounds": list(bounds) if bounds is not None else None,
            "device": resolved_device,
            "device_requested": str(device),
        },
    }
    summary_path = out / "region_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def estimate_registered_video_normalization_bounds(
    video_path: str | Path,
    transform: Mapping[str, Any],
    *,
    output_shape: tuple[int, int],
    normalization: str,
    chunk_size_frames: int = 64,
    max_sample_values: int = 1_000_000,
    device: str = "cpu",
) -> tuple[float, float] | None:
    """Estimate robust bounds from registered chunks without materializing a stack."""
    if normalization in {"none", "raw"}:
        return None
    if normalization != "per_video_robust_percentile":
        raise ValueError(f"Unsupported normalization '{normalization}'.")
    samples: list[np.ndarray] = []
    remaining = max(1, int(max_sample_values))
    per_chunk_target = max(1, max_sample_values // 32 or 1)
    for chunk in iter_video_chunks(video_path, chunk_size=int(chunk_size_frames)):
        registered = apply_registration_transform_chunk(
            chunk.data,
            transform,
            output_shape=output_shape,
            device=device,
        )
        values = np.asarray(registered, dtype=np.float32).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        take = min(values.size, max(1, min(remaining, per_chunk_target)))
        stride = max(1, values.size // take)
        samples.append(values[::stride][:take].astype(np.float32, copy=False))
        remaining -= int(samples[-1].size)
        if remaining <= 0:
            break
    if not samples:
        return (0.0, 0.0)
    values = np.concatenate(samples)
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    if hi <= lo:
        return (0.0, 0.0)
    return (lo, hi)


def apply_registration_transform_chunk(
    video_chunk: Any,
    transform: Mapping[str, Any],
    *,
    output_shape: tuple[int, int],
    device: str = "auto",
) -> np.ndarray:
    """Apply a per-video registration transform to one frame chunk."""
    resolved = _resolve_registration_device(device)
    if resolved == "cuda":
        try:
            return _apply_registration_transform_torch(video_chunk, transform, output_shape=output_shape, device=resolved)
        except Exception:
            if str(device).lower() == "cuda":
                raise
    return apply_registration_transform(video_chunk, transform, output_shape=output_shape, output_dtype="float32")


def _resolve_registration_device(device: str) -> str:
    requested = str(device or "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cpu":
        return "cpu"
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        if requested == "cuda":
            raise RuntimeError("PyTorch is required for --device cuda.")
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return "cpu"


def _apply_registration_transform_torch(
    video_chunk: Any,
    transform: Mapping[str, Any],
    *,
    output_shape: tuple[int, int],
    device: str,
) -> np.ndarray:
    import torch  # type: ignore
    import torch.nn.functional as F

    arr = np.asarray(video_chunk, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected frame-first [T,H,W] video chunk, got shape {arr.shape}.")
    out_h, out_w = int(output_shape[0]), int(output_shape[1])
    if out_h <= 0 or out_w <= 0:
        raise ValueError("output shape must be positive")
    scale = float(transform.get("scale", 1.0))
    if abs(scale) < 1e-9:
        raise ValueError("scale must be nonzero")
    rotation = float(transform.get("rotation_deg", 0.0))
    translation = transform.get("translation_px") or [0.0, 0.0]
    dx, dy = float(translation[0]), float(translation[1])
    src_h, src_w = int(arr.shape[1]), int(arr.shape[2])
    with torch.no_grad():
        tensor = torch.as_tensor(arr, dtype=torch.float32, device=device).unsqueeze(1)
        yy, xx = torch.meshgrid(
            torch.arange(out_h, dtype=torch.float32, device=device),
            torch.arange(out_w, dtype=torch.float32, device=device),
            indexing="ij",
        )
        src_cy = (src_h - 1) / 2.0
        src_cx = (src_w - 1) / 2.0
        out_cy = (out_h - 1) / 2.0
        out_cx = (out_w - 1) / 2.0
        theta = np.deg2rad(rotation)
        c = float(np.cos(theta))
        ss = float(np.sin(theta))
        x_rel = xx - out_cx - dx
        y_rel = yy - out_cy - dy
        src_x = (c * x_rel + ss * y_rel) / scale + src_cx
        src_y = (-ss * x_rel + c * y_rel) / scale + src_cy
        grid_x = (src_x / float(src_w - 1)) * 2.0 - 1.0 if src_w > 1 else torch.zeros_like(src_x)
        grid_y = (src_y / float(src_h - 1)) * 2.0 - 1.0 if src_h > 1 else torch.zeros_like(src_y)
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(arr.shape[0], -1, -1, -1)
        sampled = F.grid_sample(tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return sampled.squeeze(1).detach().cpu().numpy().astype(np.float32, copy=False)

def write_region_features_tsv(path: Path, flat_state: np.ndarray, region_ids: Sequence[str]) -> None:
    mean = flat_state[:, :, 0].mean(axis=0)
    std = flat_state[:, :, 0].std(axis=0)
    lines = ["region_id\tmean_intensity_mean\tmean_intensity_std\n"]
    for rid, m, s in zip(region_ids, mean, std):
        lines.append(f"{rid}\t{float(m):.8g}\t{float(s):.8g}\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_grid_overlay(path: str | Path, projection: Any, grid_spec: Mapping[str, Any]) -> None:
    arr = np.asarray(projection, dtype=np.float32).copy()
    if arr.ndim != 2:
        raise ValueError("grid overlay projection must be 2-D")
    if arr.size:
        arr = arr - float(np.nanmin(arr))
        arr = arr / max(float(np.nanmax(arr)), 1e-6)
    for region in grid_spec["regions"]:
        x0, y0, x1, y1 = [int(v) for v in region["bbox"]]
        if x0 < arr.shape[1]:
            arr[:, x0] = 1.0
        if y0 < arr.shape[0]:
            arr[y0, :] = 1.0
        if x1 - 1 < arr.shape[1]:
            arr[:, max(0, x1 - 1)] = 0.0
        if y1 - 1 < arr.shape[0]:
            arr[max(0, y1 - 1), :] = 0.0
    write_gray_preview(path, arr)


def write_trace_summary(path: str | Path, flat: np.ndarray) -> None:
    trace = np.asarray(flat, dtype=np.float32).mean(axis=1)
    if trace.size == 0:
        image = np.zeros((32, 32), dtype=np.float32)
    else:
        image = np.repeat(trace.reshape(1, -1), 48, axis=0)
    write_gray_preview(path, image)
