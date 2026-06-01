#!/usr/bin/env python3
"""Prepare local video datasets and attach Gamma CFAR sweep outputs to the workbench."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurobench.architecture_runs import build_planned_manifest
from neurobench.manifests import write_json
from neurobench.workbench.intermediates import normalize_array_frame, write_png_gray8
from neurobench.workbench.roi_payloads import (
    events_by_roi_from_payload,
    stencil_points_from_annotations,
    write_review_roi_sidecars,
)


INTERMEDIATE_STEPS = {
    "highpass": {
        "artifact_id": "highpass_video.v1",
        "label": "Temporal high-pass",
        "description": "Slow temporal baseline removed before adaptive detection.",
    },
    "smooth": {
        "artifact_id": "smoothed_video.v1",
        "label": "Spatial Gaussian",
        "description": "Spatially smoothed high-pass video used by Gamma CFAR.",
    },
    "score": {
        "artifact_id": "z_stack.v1",
        "label": "Robust local-z evidence",
        "description": "Positive robust-z evidence used for candidate peak scoring.",
    },
    "green_input": {
        "artifact_id": "green_excess_input.v1",
        "label": "Green-excess input",
        "description": "Positive green-minus-red/blue excess used as the ROI-location input.",
    },
    "cfar_small_ref": {
        "artifact_id": "cfar_small_ref_candidate_mask.v1",
        "label": "Small-reference Gamma CFAR mask",
        "description": "Permissive local Gamma CFAR candidate mask.",
    },
    "cfar_large_ref": {
        "artifact_id": "cfar_large_ref_candidate_mask.v1",
        "label": "Large-reference Gamma CFAR fused mask",
        "description": "Fused small/large-reference Gamma CFAR candidate mask.",
    },
    "green_single_cfar_mask": {
        "artifact_id": "green_single_cfar_candidate_mask.v1",
        "label": "Green-excess single Gamma CFAR mask",
        "description": "Thresholded candidate mask from the green-excess single Gamma CFAR pass.",
    },
    "green_projection_blob_map": {
        "artifact_id": "green_projection_blob_mask.v1",
        "label": "Green-excess projection blob mask",
        "description": "Persistent or high-percentile green blobs unioned into the ROI candidate footprint.",
    },
    "green_projection_score": {
        "artifact_id": "green_projection_score.v1",
        "label": "Green-excess projection evidence",
        "description": "Continuous persistent green evidence used to recover long-lived active neurons.",
    },
}


GREEN_MULTISCALE_SWEEP_ID = "green_excess_multiscale_cfar_v3"
GREEN_MULTISCALE_RUN_PREFIX = "green_roi_mscfar_v3"
GREEN_MULTISCALE_OUTPUT = PROJECT_ROOT / "Outputs/GammaCFAR/external_test/green_excess_multiscale_cfar_v3"
MIN_OVERNIGHT_FREE_DISK_GB = 150.0
MIN_OVERNIGHT_AVAILABLE_RAM_GB = 16.0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def rel_to(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def frame_path(out_dir: Path, index: int) -> Path:
    return out_dir / f"frame_{index:03d}.png"


def write_starter_review_dataset(
    *,
    dataset_id: str,
    source_path: Path,
    source_name: str,
    app_dir: Path,
    manifest_path: Path,
    frame_rate_hz: float,
    pixel_size_microns: float | None,
    width: int,
    height: int,
    frame_count: int,
    means: list[float],
    maxes: list[float],
    source_metadata: Mapping[str, Any] | None = None,
) -> None:
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "name": source_name,
        "frame_rate_hz": frame_rate_hz,
        "pixel_size_microns": pixel_size_microns,
        "paths": {
            "raw_video": str(source_path),
            "app_dir": str(app_dir),
            "review_data": str(app_dir / "review_data.json"),
            "annotations": str(app_dir / "annotations.json"),
            "architecture_runs": str(app_dir / "architecture_runs.json"),
        },
    }
    if source_metadata:
        manifest["source"] = dict(source_metadata)
    write_json(manifest_path, manifest)

    review_data = {
        "schema_version": 1,
        "dataset": {
            "dataset_id": dataset_id,
            "frame_rate_hz": frame_rate_hz,
            "pixel_size_microns": pixel_size_microns,
            "raw_video": str(source_path),
        },
        "video": {
            "name": source_name,
            "width": int(width),
            "height": int(height),
            "frames": int(frame_count),
            "frameRateHz": float(frame_rate_hz),
            "framePattern": "frames/frame_%03d.png",
        },
        "qc": {
            "frameMeanTrace": [round(value, 6) for value in means],
            "frameMaxTrace": [round(value, 6) for value in maxes],
        },
        "parameters": {
            "datasetId": dataset_id,
            "frameRateHz": frame_rate_hz,
            "pixelSizeMicrons": pixel_size_microns,
        },
        "rois": [],
        "discovery": {"suggestions": [], "evidenceMaps": []},
    }
    if source_metadata:
        review_data["source"] = dict(source_metadata)
    write_json_atomic(app_dir / "review_data.json", review_data)
    if not (app_dir / "annotations.json").exists():
        write_json_atomic(app_dir / "annotations.json", {"schema_version": 3, "version": 3, "rois": {}, "events": {}, "suggestions": {}})


def prepare_dataset(args: argparse.Namespace) -> None:
    import numpy as np
    import tifffile

    input_tif = args.input_tif
    dataset_id = args.dataset_id
    app_dir = args.app_dir
    output_npy = args.output_npy
    frames_dir = app_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_npy.parent.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(input_tif) as tif:
        pages = list(tif.pages)
        if not pages:
            raise SystemExit(f"No TIFF pages found in {input_tif}")
        first = pages[0].asarray()
        height, width = first.shape[:2]
        frame_count = len(pages)
        stack = np.lib.format.open_memmap(output_npy, mode="w+", dtype=first.dtype, shape=(frame_count, height, width))
        means: list[float] = []
        maxes: list[float] = []
        for index, page in enumerate(pages, start=1):
            frame = first if index == 1 else page.asarray()
            if frame.ndim != 2:
                frame = np.asarray(frame).squeeze()
            if frame.shape != (height, width):
                raise SystemExit(f"Frame {index} has shape {frame.shape}, expected {(height, width)}")
            stack[index - 1] = frame
            means.append(float(np.mean(frame)))
            maxes.append(float(np.max(frame)))
            write_png_gray8(frame_path(frames_dir, index), int(width), int(height), normalize_array_frame(frame))
        stack.flush()

    write_starter_review_dataset(
        dataset_id=dataset_id,
        source_path=input_tif,
        source_name=input_tif.name,
        app_dir=app_dir,
        manifest_path=args.manifest,
        frame_rate_hz=args.frame_rate_hz,
        pixel_size_microns=args.pixel_size_microns,
        width=int(width),
        height=int(height),
        frame_count=int(frame_count),
        means=means,
        maxes=maxes,
        source_metadata={"template": "local_tiff", "converted_npy": str(output_npy)},
    )

    print(json.dumps({"dataset_id": dataset_id, "shape": [frame_count, height, width], "npy": str(output_npy)}, indent=2))


def mp4_channel_decode_spec(channel: str) -> dict[str, Any]:
    if channel == "luma":
        return {"pix_fmt": "gray", "frame_size_multiplier": 1, "dtype": "uint8"}
    if channel == "green_excess":
        return {"pix_fmt": "rgb24", "frame_size_multiplier": 3, "dtype": "float32"}
    raise SystemExit(f"Unsupported MP4 channel conversion: {channel}")


def convert_rgb_frame_to_channel(rgb: Any, channel: str):
    import numpy as np

    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected an RGB frame with shape HxWx3, got {arr.shape}.")
    if channel == "green_excess":
        float_rgb = arr.astype(np.float32, copy=False)
        return np.maximum(float_rgb[:, :, 1] - 0.5 * (float_rgb[:, :, 0] + float_rgb[:, :, 2]), 0.0).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported RGB channel conversion: {channel}")


def prepare_mp4_dataset(args: argparse.Namespace) -> None:
    import numpy as np

    input_mp4 = args.input_mp4
    if not input_mp4.exists():
        raise SystemExit(f"MP4 not found: {input_mp4}")

    decode_spec = mp4_channel_decode_spec(args.channel)
    metadata = mp4_video_metadata(input_mp4)
    width = int(metadata["width"])
    height = int(metadata["height"])
    frame_count = int(metadata["frame_count"])
    frame_rate_hz = float(args.frame_rate_hz or metadata["frame_rate_hz"])
    app_dir = args.app_dir
    output_npy = args.output_npy
    frames_dir = app_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_npy.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(input_mp4),
        "-map",
        "0:v:0",
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        str(decode_spec["pix_fmt"]),
        "-",
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise SystemExit("ffmpeg is required for MP4 conversion.") from exc
    if process.stdout is None or process.stderr is None:
        raise SystemExit("Could not open ffmpeg pipes for MP4 conversion.")

    frame_size = width * height * int(decode_spec["frame_size_multiplier"])
    stack = np.lib.format.open_memmap(output_npy, mode="w+", dtype=np.dtype(str(decode_spec["dtype"])), shape=(frame_count, height, width))
    means: list[float] = []
    maxes: list[float] = []
    decoded = 0
    while decoded < frame_count:
        payload = process.stdout.read(frame_size)
        if not payload:
            break
        if len(payload) != frame_size:
            process.kill()
            raise SystemExit(f"Decoded partial frame {decoded + 1}: got {len(payload)} bytes, expected {frame_size}.")
        if args.channel == "luma":
            frame = np.frombuffer(payload, dtype=np.uint8).reshape((height, width))
        else:
            rgb = np.frombuffer(payload, dtype=np.uint8).reshape((height, width, 3))
            frame = convert_rgb_frame_to_channel(rgb, args.channel)
        stack[decoded] = frame
        means.append(float(np.mean(frame)))
        maxes.append(float(np.max(frame)))
        write_png_gray8(frame_path(frames_dir, decoded + 1), width, height, normalize_array_frame(frame))
        decoded += 1
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    stack.flush()
    if return_code != 0:
        raise SystemExit(f"ffmpeg failed while decoding {input_mp4}:\n{stderr.strip()}")
    if decoded != frame_count:
        raise SystemExit(f"Decoded {decoded} frame(s), expected {frame_count} from ffprobe.")

    source_metadata = {
        "template": "local_mp4",
        "input_mp4": str(input_mp4),
        "converted_npy": str(output_npy),
        "conversion_channel": args.channel,
        "codec_name": metadata.get("codec_name", ""),
        "encoded_frame_rate_hz": metadata.get("frame_rate_hz"),
    }
    write_starter_review_dataset(
        dataset_id=args.dataset_id,
        source_path=input_mp4,
        source_name=input_mp4.name,
        app_dir=app_dir,
        manifest_path=args.manifest,
        frame_rate_hz=frame_rate_hz,
        pixel_size_microns=args.pixel_size_microns,
        width=width,
        height=height,
        frame_count=frame_count,
        means=means,
        maxes=maxes,
        source_metadata=source_metadata,
    )
    print(json.dumps({"dataset_id": args.dataset_id, "shape": [frame_count, height, width], "npy": str(output_npy), "channel": args.channel}, indent=2))


def mp4_video_metadata(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,codec_name,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise SystemExit("ffprobe is required for MP4 metadata inspection.") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ffprobe failed for {path}:\n{exc.stderr}") from exc
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise SystemExit(f"No video stream found in {path}")
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise SystemExit(f"Could not determine MP4 dimensions for {path}")
    frame_rate_hz = parse_frame_rate(stream.get("avg_frame_rate")) or parse_frame_rate(stream.get("r_frame_rate"))
    frame_count = parse_positive_int(stream.get("nb_read_frames")) or parse_positive_int(stream.get("nb_frames"))
    duration = parse_positive_float((payload.get("format") or {}).get("duration"))
    if frame_count is None and frame_rate_hz and duration:
        frame_count = max(1, int(round(frame_rate_hz * duration)))
    if frame_count is None or frame_count <= 0:
        raise SystemExit(f"Could not determine MP4 frame count for {path}")
    if not frame_rate_hz:
        raise SystemExit(f"Could not determine MP4 frame rate for {path}")
    return {
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "frame_rate_hz": frame_rate_hz,
        "duration_seconds": duration,
        "codec_name": stream.get("codec_name", ""),
    }


def parse_frame_rate(value: Any) -> float | None:
    if value in {None, "", "0/0"}:
        return None
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        den = parse_positive_float(denominator)
        num = parse_positive_float(numerator)
        if not num or not den:
            return None
        return num / den
    return parse_positive_float(text)


def parse_positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def parse_positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def write_sweep_spec(args: argparse.Namespace) -> None:
    spec = {
        "schema_version": 1,
        "dataset_id": args.dataset_id,
        "run_id": args.run_id,
        "label": "Cascaded Gamma CFAR 36-run grid",
        "pipeline": [
            {"id": "source", "stage_id": "source_video_import", "params": {"source": str(args.source_npy)}},
            {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 6.0}},
            {"id": "smooth", "stage_id": "spatial_gaussian", "params": {"sigma_px": 0.6}},
            {"id": "score", "stage_id": "robust_positive_local_z", "params": {"local_radius_px": 11, "epsilon": 1.0}},
            {"id": "cfar_small_ref", "stage_id": "gamma_cfar", "params": {"pfa": 0.06, "guard_px": 2, "training_radius_px": 6}},
            {
                "id": "cfar_large_ref",
                "stage_id": "gamma_cfar",
                "params": {"pfa": 0.06, "guard_px": 3, "training_radius_px": 12},
                "metadata": {"previous_mask_step": "cfar_small_ref", "combine_mode": "intersection"},
            },
            {"id": "components", "stage_id": "component_filter", "params": {"seed_z": 3.2, "min_area_px": 6, "max_area_px": 260, "support_min_frames": 12}},
            {"id": "traces", "stage_id": "local_background_ring", "params": {"outer_radius_px": 15, "neuropil_weight": 0.7}},
            {"id": "events", "stage_id": "robust_kalman_positive_innovation", "params": {"event_threshold_z": 2.4, "kalman_gain": 0.06, "spike_gain": 0.008}},
            {"id": "rank", "stage_id": "heuristic_priority_v1", "params": {"local_correlation_weight": 0.25, "event_support_weight": 0.35, "artifact_weight": -0.3}},
        ],
        "sweep": {
            "id": "gamma_cfar_cascade_grid_v2",
            "label": "High-recall cascaded Gamma CFAR grid",
            "parameters": [
                {"stage": "cfar_small_ref", "param": "pfa", "values": [0.04, 0.06, 0.10, 0.14]},
                {"stage": "cfar_large_ref", "param": "training_radius_px", "values": [10, 14, 18]},
                {"stage": "components", "param": "support_min_frames", "values": [8, 12, 18]},
            ],
        },
        "artifacts": {},
    }
    write_json_atomic(args.out, spec)
    print(f"Wrote {args.out}")


def write_green_excess_cfar_spec(args: argparse.Namespace) -> None:
    spec = {
        "schema_version": 1,
        "dataset_id": args.dataset_id,
        "run_id": args.run_id,
        "label": "Green-excess single Gamma CFAR ROI grid",
        "pipeline": [
            {"id": "source", "stage_id": "source_video_import", "params": {"source": str(args.source_npy), "channel": "green_excess"}},
            {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 0.0}},
            {"id": "smooth", "stage_id": "spatial_gaussian", "params": {"sigma_px": 0.8}},
            {"id": "score", "stage_id": "robust_positive_local_z", "params": {"local_radius_px": 11, "epsilon": 1.0}},
            {"id": "green_single_cfar", "stage_id": "gamma_cfar", "params": {"pfa": 0.02, "guard_px": 2, "training_radius_px": 12}},
            {"id": "components", "stage_id": "component_filter", "params": {"seed_z": 2.5, "min_area_px": 8, "max_area_px": 320, "support_min_frames": 1}},
            {"id": "traces", "stage_id": "local_background_ring", "params": {"outer_radius_px": 15, "neuropil_weight": 0.7}},
            {"id": "events", "stage_id": "robust_kalman_positive_innovation", "params": {"event_threshold_z": 2.4, "kalman_gain": 0.06, "spike_gain": 0.008}},
            {"id": "rank", "stage_id": "heuristic_priority_v1", "params": {"local_correlation_weight": 0.25, "event_support_weight": 0.35, "artifact_weight": -0.3}},
        ],
        "sweep": {
            "id": "green_excess_single_cfar_v1",
            "label": "Green-excess single Gamma CFAR six-run ROI grid",
            "parameters": [
                {"stage": "green_single_cfar", "param": "pfa", "values": [0.01, 0.02, 0.04]},
                {"stage": "components", "param": "support_min_frames", "values": [1, 6]},
            ],
        },
        "artifacts": {},
    }
    write_json_atomic(args.out, spec)
    print(f"Wrote {args.out}")


def write_green_excess_roi_state_spec(args: argparse.Namespace) -> None:
    spec = {
        "schema_version": 1,
        "dataset_id": args.dataset_id,
        "run_id": args.run_id,
        "label": "Green-excess ROI-state detection grid",
        "pipeline": [
            {"id": "source", "stage_id": "source_video_import", "params": {"source": str(args.source_npy), "channel": "green_excess"}},
            {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 0.0}},
            {"id": "smooth", "stage_id": "spatial_gaussian", "params": {"sigma_px": 0.8}},
            {"id": "score", "stage_id": "robust_positive_local_z", "params": {"local_radius_px": 11, "epsilon": 1.0}},
            {"id": "green_single_cfar", "stage_id": "gamma_cfar", "params": {"pfa": 0.04, "guard_px": 2, "training_radius_px": 12}},
            {"id": "components", "stage_id": "component_filter", "params": {"seed_z": 1.8, "min_area_px": 6, "max_area_px": 450, "support_min_frames": 1, "projection_blob_z": 1.5}},
            {"id": "traces", "stage_id": "local_background_ring", "params": {"outer_radius_px": 15, "neuropil_weight": 0.7}},
            {"id": "events", "stage_id": "robust_kalman_positive_innovation", "params": {"event_threshold_z": 2.4, "kalman_gain": 0.06, "spike_gain": 0.008}},
            {"id": "activity_states", "stage_id": "trace_event_scoring", "params": {"event_threshold_z": 2.4, "sustained_z": 1.2, "tonic_z": 2.0, "peak_window_frames": 1}},
            {"id": "rank", "stage_id": "heuristic_priority_v1", "params": {"local_correlation_weight": 0.25, "event_support_weight": 0.35, "artifact_weight": -0.3}},
        ],
        "sweep": {
            "id": "green_excess_roi_state_v2",
            "label": "Green-excess ROI-state recall grid",
            "description": "Single CFAR plus projection-blob ROI candidates with peak/sustained/inactive activity states.",
            "parameters": [
                {"stage": "green_single_cfar", "param": "pfa", "values": [0.02, 0.04, 0.08]},
                {"stage": "components", "param": "projection_blob_z", "values": [1.5, 2.0]},
                {"stage": "components", "param": "support_min_frames", "values": [1, 5]},
            ],
        },
        "artifacts": {},
    }
    write_json_atomic(args.out, spec)
    print(f"Wrote {args.out}")


def _scaled_tag(value: float, *, scale: float, width: int) -> str:
    return f"{int(round(float(value) * scale)):0{width}d}"


def green_multiscale_run_identity(run: Mapping[str, Any]) -> dict[str, Any]:
    small = pipeline_stage_params(run, "cfar_small_ref", "gamma_cfar")
    large = pipeline_stage_params(run, "cfar_large_ref", "gamma_cfar")
    components = pipeline_stage_params(run, "components", "component_filter")
    pfa = float(small.get("pfa", large.get("pfa", 0.04)))
    small_radius = int(small.get("training_radius_px", 6))
    large_radius = int(large.get("training_radius_px", 18))
    fusion_mode = str(components.get("fusion_mode", "intersection")).strip().lower()
    support = int(components.get("support_min_frames", 15))
    projection_blob_z = float(components.get("projection_blob_z", 1.5))
    if fusion_mode not in {"intersection", "union"}:
        raise ValueError("green multiscale CFAR fusion_mode must be 'intersection' or 'union'.")
    run_id = (
        f"{GREEN_MULTISCALE_RUN_PREFIX}_"
        f"pfa{_scaled_tag(pfa, scale=100.0, width=3)}_"
        f"sR{small_radius:02d}_lR{large_radius:02d}_"
        f"{fusion_mode}_sup{support:03d}_pz{_scaled_tag(projection_blob_z, scale=10.0, width=2)}"
    )
    label = (
        "Green ROI MSCFAR v3 | "
        f"pfa={pfa:g} | smallR={small_radius} | largeR={large_radius} | "
        f"{fusion_mode} | support={support} | projection_z={projection_blob_z:g}"
    )
    return {
        "run_id": run_id,
        "label": label,
        "pfa": pfa,
        "small_training_radius_px": small_radius,
        "large_training_radius_px": large_radius,
        "fusion_mode": fusion_mode,
        "support_min_frames": support,
        "projection_blob_z": projection_blob_z,
    }


def apply_green_multiscale_run_metadata(planned: dict[str, Any]) -> dict[str, Any]:
    if (planned.get("sweep") or {}).get("id") != GREEN_MULTISCALE_SWEEP_ID:
        return planned
    for run in planned.get("runs", []) or []:
        identity = green_multiscale_run_identity(run)
        run["run_id"] = identity["run_id"]
        run["label"] = identity["label"]
        run_summary = dict(run.get("summary") or {})
        run_summary.update({key: identity[key] for key in ("pfa", "small_training_radius_px", "large_training_radius_px", "fusion_mode", "support_min_frames", "projection_blob_z")})
        run["summary"] = run_summary
        for step in run.get("pipeline", []) or []:
            if step.get("id") == "cfar_large_ref":
                params = dict(step.get("params") or {})
                params["pfa"] = identity["pfa"]
                step["params"] = params
                metadata = dict(step.get("metadata") or {})
                metadata["previous_mask_step"] = "cfar_small_ref"
                metadata["combine_mode"] = identity["fusion_mode"]
                step["metadata"] = metadata
            if step.get("id") == "components":
                params = dict(step.get("params") or {})
                params["fusion_mode"] = identity["fusion_mode"]
                step["params"] = params
    return planned


def planned_manifest_for_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    return apply_green_multiscale_run_metadata(build_planned_manifest(spec))


def write_green_excess_multiscale_cfar_spec(args: argparse.Namespace) -> None:
    spec = {
        "schema_version": 1,
        "dataset_id": args.dataset_id,
        "run_id": args.run_id,
        "label": "GPU green-excess multiscale Gamma CFAR ROI-state grid",
        "execution": {"device": "cuda", "backend": "cupy_cuda", "cpu_fallback": False},
        "pipeline": [
            {"id": "source", "stage_id": "source_video_import", "params": {"source": str(args.source_npy), "channel": "green_excess"}},
            {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 0.0}},
            {"id": "smooth", "stage_id": "spatial_gaussian", "params": {"sigma_px": 0.8}},
            {"id": "score", "stage_id": "robust_positive_local_z", "params": {"local_radius_px": 11, "epsilon": 1.0}},
            {"id": "cfar_small_ref", "stage_id": "gamma_cfar", "params": {"pfa": 0.04, "guard_px": 2, "training_radius_px": 6}},
            {
                "id": "cfar_large_ref",
                "stage_id": "gamma_cfar",
                "params": {"pfa": 0.04, "guard_px": 2, "training_radius_px": 18},
                "metadata": {"previous_mask_step": "cfar_small_ref", "combine_mode": "intersection"},
            },
            {
                "id": "components",
                "stage_id": "component_filter",
                "params": {
                    "seed_z": 1.8,
                    "grow_z": 1.1,
                    "min_area_px": 6,
                    "max_area_px": 450,
                    "split_large_components": True,
                    "split_min_distance_px": 6,
                    "split_area_px": 80,
                    "split_max_peaks": 40,
                    "support_min_frames": 15,
                    "projection_blob_z": 1.5,
                    "fusion_mode": "intersection",
                },
            },
            {"id": "traces", "stage_id": "local_background_ring", "params": {"outer_radius_px": 15, "neuropil_weight": 0.7}},
            {"id": "events", "stage_id": "robust_kalman_positive_innovation", "params": {"event_threshold_z": 2.4, "kalman_gain": 0.06, "spike_gain": 0.008}},
            {"id": "activity_states", "stage_id": "trace_event_scoring", "params": {"event_threshold_z": 2.4, "sustained_z": 1.2, "tonic_z": 2.0, "peak_window_frames": 1}},
            {"id": "rank", "stage_id": "heuristic_priority_v1", "params": {"local_correlation_weight": 0.25, "event_support_weight": 0.35, "artifact_weight": -0.3}},
        ],
        "sweep": {
            "id": GREEN_MULTISCALE_SWEEP_ID,
            "label": "GPU green-excess multiscale Gamma CFAR overnight grid",
            "description": "Small/large-reference Gamma CFAR fusion plus persistent green projection evidence for tonic active neurons.",
            "parameters": [
                {"stage": "cfar_small_ref", "param": "pfa", "values": [0.02, 0.04, 0.08]},
                {"stage": "cfar_small_ref", "param": "training_radius_px", "values": [6, 8]},
                {"stage": "cfar_large_ref", "param": "training_radius_px", "values": [18, 24]},
                {"stage": "components", "param": "fusion_mode", "values": ["intersection", "union"]},
                {"stage": "components", "param": "support_min_frames", "values": [1, 15, 30]},
                {"stage": "components", "param": "projection_blob_z", "values": [1.5, 2.0]},
            ],
        },
        "artifacts": {},
    }
    planned = planned_manifest_for_spec(spec)
    if len(planned.get("runs", [])) != 144:
        raise RuntimeError(f"Green multiscale CFAR spec should expand to 144 runs, got {len(planned.get('runs', []))}.")
    write_json_atomic(args.out, spec)
    print(f"Wrote {args.out}")


def attach_sweep(args: argparse.Namespace) -> None:
    spec = load_json(args.spec)
    planned = planned_manifest_for_spec(spec)
    sweep_summary = load_json(args.sweep_root / "sweep_summary.json")
    run_records = {run["run_id"]: run for run in sweep_summary.get("runs", [])}
    recommended_run_id = str(sweep_summary.get("recommended_run_id") or "")
    pixel_size_um = review_data_pixel_size(args.app_dir / "review_data.json")

    baseline = {
        "schema_version": 1,
        "run_id": "current_review_pipeline",
        "dataset_id": args.dataset_id,
        "label": "Raw review baseline",
        "execution": {"status": "completed"},
        "pipeline": [{"name": "review_data_import"}],
        "summary": {"frame_count": args.frame_count, "roi_count": 0, "event_count": 0},
        "artifacts": {
            "review_data": str(args.app_dir / "review_data.json"),
            "frames": str(args.app_dir / "frames"),
            "intermediates": [],
        },
    }
    new_runs = [baseline]
    for run in planned.get("runs", []):
        run = dict(run)
        run_id = str(run["run_id"])
        record = dict(run_records.get(run_id) or {})
        run_root = args.sweep_root / str(record.get("run_root") or "")
        if not run_root.exists():
            run_root = next(args.sweep_root.glob(f"*_{run_id}"), run_root)
        pipeline_run_path = run_root / "pipeline_run.json"
        run["execution"] = {
            "status": record.get("status", "missing"),
            "run_root": str(run_root),
            "pipeline_run": str(pipeline_run_path),
        }
        run["artifacts"] = {
            "pipeline_run": str(pipeline_run_path),
            "sweep_summary": str(args.sweep_root / "sweep_summary.json"),
            "sweep_brief": str(args.sweep_root / "gamma_cfar_grid_brief.md"),
            "intermediates": [],
        }
        if pipeline_run_path.exists():
            attach_run_artifacts(run, run_root, args.app_dir, args.frame_count, pixel_size_um=pixel_size_um)
        if recommended_run_id and run_id == recommended_run_id:
            run.setdefault("summary", {})["recommended_default"] = True
            run.setdefault("summary", {})["default_selection_reason"] = sweep_summary.get("selection_criteria", "recommended by sweep summary")
            run.setdefault("artifacts", {})["recommended_default"] = True
        new_runs.append(run)

    new_experiment = {
        "id": spec["run_id"],
        "label": spec.get("label", spec["run_id"]),
        "source": "local_sweep",
        "run_ids": [run["run_id"] for run in new_runs[1:]],
        "summary": str(args.sweep_root / "gamma_cfar_grid_brief.md"),
    }
    runs = new_runs
    experiments = [new_experiment]
    extra_artifacts: dict[str, Any] = {}
    architecture_runs_path = args.app_dir / "architecture_runs.json"
    if getattr(args, "merge_existing", False) and architecture_runs_path.exists():
        existing = load_json(architecture_runs_path)
        new_ids = {str(run.get("run_id") or "") for run in new_runs}
        preserved_runs = [
            run
            for run in existing.get("runs", []) or []
            if str(run.get("run_id") or "") not in new_ids and str(run.get("run_id") or "") != "current_review_pipeline"
        ]
        runs = [baseline] + preserved_runs + new_runs[1:]
        experiments = [
            item
            for item in existing.get("experiments", []) or []
            if str(item.get("id") or "") != str(spec["run_id"])
        ] + [new_experiment]
        extra_artifacts = dict(existing.get("artifacts") or {})

    manifest = {
        "schema_version": 1,
        "dataset_id": args.dataset_id,
        "sweep": planned.get("sweep", {}),
        "experiments": experiments,
        "runs": runs,
    }
    if extra_artifacts:
        manifest["artifacts"] = extra_artifacts
    write_json_atomic(architecture_runs_path, manifest)
    print(json.dumps({"architecture_runs": str(architecture_runs_path), "runs": len(runs), "merged": bool(getattr(args, "merge_existing", False))}, indent=2))


def run_fast_grid(args: argparse.Namespace) -> None:
    """Run the fixed 36-run grid while sharing expensive preprocessing outputs."""
    import numpy as np
    from scipy import ndimage

    from neurobench.discovery.ranking import rank_candidates

    spec = load_json(args.spec)
    planned = planned_manifest_for_spec(spec)
    planned_runs = planned.get("runs", []) or []
    if not planned_runs:
        raise SystemExit("Sweep spec did not expand to any runs.")
    first_run = args.sweep_root / f"001_{safe_name(str(planned_runs[0]['run_id']))}"
    preprocessing = first_run / "artifacts" / "preprocessing"
    highpass_path = preprocessing / "highpass_video.npy"
    smoothed_path = preprocessing / "smoothed_video.npy"
    z_path = preprocessing / "z_stack.npy"
    ensure_shared_preprocessing(
        args=args,
        spec=spec,
        first_run=first_run,
        highpass_path=highpass_path,
        smoothed_path=smoothed_path,
        z_path=z_path,
    )

    args.sweep_root.mkdir(parents=True, exist_ok=True)
    shared_mask_root = args.sweep_root / "shared_masks"
    shared_mask_root.mkdir(parents=True, exist_ok=True)
    smoothed = np.load(smoothed_path, mmap_mode="r").astype(np.float32, copy=False)
    z_stack = np.load(z_path, mmap_mode="r").astype(np.float32, copy=False)
    highpass = np.load(highpass_path, mmap_mode="r").astype(np.float32, copy=False)
    z_projection = np.max(z_stack, axis=0)

    small_specs = sorted(
        {
            (
                float(pipeline_stage_params(run, "cfar_small_ref", "gamma_cfar").get("pfa", 0.06)),
                int(pipeline_stage_params(run, "cfar_small_ref", "gamma_cfar").get("guard_px", 2)),
                int(pipeline_stage_params(run, "cfar_small_ref", "gamma_cfar").get("training_radius_px", 6)),
            )
            for run in planned_runs
        }
    )
    small_masks: dict[tuple[float, int, int], tuple[Path, dict[str, Any]]] = {}
    for pfa, guard_px, training_radius_px in small_specs:
        key = f"small_pfa_{str(pfa).replace('.', 'p')}_guard_{guard_px}_radius_{training_radius_px}"
        path = shared_mask_root / f"{key}.npy"
        if path.exists():
            mask = np.load(path, mmap_mode="r")
            summary = {"active_fraction": float(np.mean(mask)), "pfa": pfa, "guard_px": guard_px, "training_radius_px": training_radius_px, "threshold_z": cfar_threshold(pfa), "combine_mode": "replace", "shape": list(mask.shape)}
        else:
            summary = write_chunked_cfar_mask(
                smoothed,
                path,
                pfa=pfa,
                guard_px=guard_px,
                training_radius_px=training_radius_px,
                chunk_frames=args.cfar_chunk_frames,
            )
        small_masks[(pfa, guard_px, training_radius_px)] = (path, summary)

    large_specs = sorted(
        {
            (
                float(pipeline_stage_params(run, "cfar_large_ref", "gamma_cfar").get("pfa", 0.06)),
                int(pipeline_stage_params(run, "cfar_large_ref", "gamma_cfar").get("guard_px", 3)),
                int(pipeline_stage_params(run, "cfar_large_ref", "gamma_cfar").get("training_radius_px", 12)),
            )
            for run in planned_runs
        }
    )
    large_masks: dict[tuple[float, int, int], tuple[Path, dict[str, Any]]] = {}
    for pfa, guard_px, training_radius_px in large_specs:
        key = f"large_pfa_{str(pfa).replace('.', 'p')}_guard_{guard_px}_radius_{training_radius_px}"
        path = shared_mask_root / f"{key}.npy"
        if path.exists():
            mask = np.load(path, mmap_mode="r")
            summary = {"active_fraction": float(np.mean(mask)), "pfa": pfa, "guard_px": guard_px, "training_radius_px": training_radius_px, "threshold_z": cfar_threshold(pfa), "combine_mode": "replace", "shape": list(mask.shape)}
        else:
            summary = write_chunked_cfar_mask(
                smoothed,
                path,
                pfa=pfa,
                guard_px=guard_px,
                training_radius_px=training_radius_px,
                chunk_frames=args.cfar_chunk_frames,
            )
        large_masks[(pfa, guard_px, training_radius_px)] = (path, summary)

    summary_runs = []
    raw_dtype = str(np.load(args.source_npy, mmap_mode="r").dtype)
    for index, planned_run in enumerate(planned_runs, start=1):
        run_id = planned_run["run_id"]
        small_params = pipeline_stage_params(planned_run, "cfar_small_ref", "gamma_cfar")
        large_params = pipeline_stage_params(planned_run, "cfar_large_ref", "gamma_cfar")
        component_params = pipeline_stage_params(planned_run, "components", "component_filter")
        pfa = float(small_params.get("pfa", 0.06))
        small_guard = int(small_params.get("guard_px", 2))
        small_radius = int(small_params.get("training_radius_px", 6))
        large_pfa = float(large_params.get("pfa", 0.06))
        large_guard = int(large_params.get("guard_px", 3))
        radius = int(large_params.get("training_radius_px", 12))
        support = max(1, int(component_params.get("support_min_frames", 12)))
        seed_z = float(component_params.get("seed_z", 3.2))
        min_area = int(component_params.get("min_area_px", 6))
        max_area = int(component_params.get("max_area_px", 260))
        run_root = args.sweep_root / f"{index:03d}_{safe_name(run_id)}"
        run_root.mkdir(parents=True, exist_ok=True)
        candidates_dir = run_root / "artifacts" / "candidates"
        traces_dir = run_root / "artifacts" / "traces"
        events_dir = run_root / "artifacts" / "events"
        for directory in (candidates_dir, traces_dir, events_dir):
            directory.mkdir(parents=True, exist_ok=True)

        small_path, small_summary = small_masks[(pfa, small_guard, small_radius)]
        large_path, large_summary = large_masks[(large_pfa, large_guard, radius)]
        small_mask = np.load(small_path, mmap_mode="r").astype(bool, copy=False)
        large_mask = np.load(large_path, mmap_mode="r").astype(bool, copy=False)
        final_mask = small_mask & large_mask
        candidate_path = candidates_dir / "cfar_large_ref_candidate_mask.npy"
        np.save(candidate_path, final_mask.astype(np.uint8, copy=False))
        latest_mask_path = candidates_dir / "candidate_mask.npy"
        np.save(latest_mask_path, final_mask.astype(np.uint8, copy=False))
        final_summary = dict(large_summary)
        final_summary.update({"active_fraction": float(np.mean(final_mask)), "combine_mode": "intersection", "previous_mask_step": "cfar_small_ref"})

        candidates = component_candidates(final_mask, z_projection, support_min_frames=support, seed_z=seed_z, min_area=min_area, max_area=max_area, ndimage=ndimage)
        candidate_json = candidates_dir / "roi_candidates.json"
        write_json_atomic(candidate_json, {"schema_version": 1, "candidates": candidates})
        traces = extract_traces(highpass, candidates)
        trace_json = traces_dir / "roi_traces.json"
        write_json_atomic(trace_json, {"schema_version": 1, "traces": traces})
        events = score_events(traces)
        event_json = events_dir / "kalman_candidate_events.json"
        write_json_atomic(event_json, events)
        ranked = rank_candidates(candidates, video_shape=highpass.shape, weights={"local_correlation_weight": 0.25, "event_support_weight": 0.35, "artifact_weight": -0.3})
        ranked_json = candidates_dir / "ranked_candidates.json"
        write_json_atomic(ranked_json, ranked)

        pipeline_run = fast_pipeline_run_manifest(
            planned_run=planned_run,
            run_root=run_root,
            raw_path=args.source_npy,
            highpass_path=highpass_path,
            smoothed_path=smoothed_path,
            z_path=z_path,
            small_path=small_path,
            small_summary=small_summary,
            final_path=candidate_path,
            final_summary=final_summary,
            candidate_json=candidate_json,
            candidate_count=len(candidates),
            seed_z=seed_z,
            min_area=min_area,
            max_area=max_area,
            support_min_frames=support,
            trace_json=trace_json,
            event_json=event_json,
            event_count=len(events["events"]),
            ranked_json=ranked_json,
            shape=list(highpass.shape),
            raw_dtype=raw_dtype,
        )
        write_json_atomic(run_root / "pipeline_run.json", pipeline_run)
        summary_runs.append(
            {
                "run_id": run_id,
                "run_root": run_root.name,
                "status": "completed",
                "sweep_index": index - 1,
                "sweep_total": len(planned_runs),
                "sweep_parameters": planned_run.get("sweep", {}).get("parameters", []),
                "artifact_count": len(pipeline_run["artifacts"]),
            }
        )
        print(f"completed {index:03d}/{len(planned_runs)}: {run_id} candidates={len(candidates)} events={len(events['events'])}", flush=True)

    sweep_summary = {
        "schema_version": 1,
        "dataset_id": planned.get("dataset_id", spec.get("dataset_id", "")),
        "sweep": planned.get("sweep", {}),
        "status": "completed",
        "total": len(summary_runs),
        "succeeded": len(summary_runs),
        "failed": 0,
        "runs": summary_runs,
    }
    write_json_atomic(args.sweep_root / "sweep_summary.json", sweep_summary)
    print(json.dumps({"sweep_summary": str(args.sweep_root / "sweep_summary.json"), "runs": len(summary_runs)}, indent=2))


def run_green_excess_grid(args: argparse.Namespace) -> None:
    """Run the green-excess six-run grid with a single Gamma CFAR pass."""
    import numpy as np
    from scipy import ndimage

    from neurobench.discovery.ranking import rank_candidates

    spec = load_json(args.spec)
    planned = planned_manifest_for_spec(spec)
    planned_runs = planned.get("runs", []) or []
    if not planned_runs:
        raise SystemExit("Sweep spec did not expand to any runs.")
    first_run = args.sweep_root / f"001_{safe_name(str(planned_runs[0]['run_id']))}"
    preprocessing = first_run / "artifacts" / "preprocessing"
    highpass_path = preprocessing / "highpass_video.npy"
    smoothed_path = preprocessing / "smoothed_video.npy"
    z_path = preprocessing / "z_stack.npy"
    ensure_shared_preprocessing(
        args=args,
        spec=spec,
        first_run=first_run,
        highpass_path=highpass_path,
        smoothed_path=smoothed_path,
        z_path=z_path,
    )

    args.sweep_root.mkdir(parents=True, exist_ok=True)
    shared_mask_root = args.sweep_root / "shared_masks"
    shared_mask_root.mkdir(parents=True, exist_ok=True)
    smoothed = np.load(smoothed_path, mmap_mode="r").astype(np.float32, copy=False)
    z_stack = np.load(z_path, mmap_mode="r").astype(np.float32, copy=False)
    highpass = np.load(highpass_path, mmap_mode="r").astype(np.float32, copy=False)
    z_projection = np.max(z_stack, axis=0)

    cfar_specs = sorted(
        {
            (
                float(pipeline_stage_params(run, "green_single_cfar", "gamma_cfar").get("pfa", 0.02)),
                int(pipeline_stage_params(run, "green_single_cfar", "gamma_cfar").get("guard_px", 2)),
                int(pipeline_stage_params(run, "green_single_cfar", "gamma_cfar").get("training_radius_px", 12)),
            )
            for run in planned_runs
        }
    )
    cfar_masks: dict[tuple[float, int, int], tuple[Path, dict[str, Any]]] = {}
    for pfa, guard_px, training_radius_px in cfar_specs:
        key = f"green_single_pfa_{str(pfa).replace('.', 'p')}_guard_{guard_px}_radius_{training_radius_px}"
        path = shared_mask_root / f"{key}.npy"
        if path.exists():
            mask = np.load(path, mmap_mode="r")
            summary = {"active_fraction": float(np.mean(mask)), "pfa": pfa, "guard_px": guard_px, "training_radius_px": training_radius_px, "threshold_z": cfar_threshold(pfa), "combine_mode": "replace", "shape": list(mask.shape)}
        else:
            summary = write_chunked_cfar_mask(
                smoothed,
                path,
                pfa=pfa,
                guard_px=guard_px,
                training_radius_px=training_radius_px,
                chunk_frames=args.cfar_chunk_frames,
            )
        cfar_masks[(pfa, guard_px, training_radius_px)] = (path, summary)

    summary_runs = []
    raw_dtype = str(np.load(args.source_npy, mmap_mode="r").dtype)
    highpass_sigma = float(pipeline_stage_params(spec, "highpass", "temporal_highpass_gaussian").get("sigma_frames", 0.0))
    smooth_sigma = float(pipeline_stage_params(spec, "smooth", "spatial_gaussian").get("sigma_px", 0.8))
    for index, planned_run in enumerate(planned_runs, start=1):
        run_id = planned_run["run_id"]
        cfar_params = pipeline_stage_params(planned_run, "green_single_cfar", "gamma_cfar")
        component_params = pipeline_stage_params(planned_run, "components", "component_filter")
        pfa = float(cfar_params.get("pfa", 0.02))
        guard = int(cfar_params.get("guard_px", 2))
        radius = int(cfar_params.get("training_radius_px", 12))
        support = max(1, int(component_params.get("support_min_frames", 1)))
        seed_z = float(component_params.get("seed_z", 2.5))
        min_area = int(component_params.get("min_area_px", 8))
        max_area = int(component_params.get("max_area_px", 320))
        run_root = args.sweep_root / f"{index:03d}_{safe_name(run_id)}"
        run_root.mkdir(parents=True, exist_ok=True)
        candidates_dir = run_root / "artifacts" / "candidates"
        traces_dir = run_root / "artifacts" / "traces"
        events_dir = run_root / "artifacts" / "events"
        for directory in (candidates_dir, traces_dir, events_dir):
            directory.mkdir(parents=True, exist_ok=True)

        mask_path, mask_summary = cfar_masks[(pfa, guard, radius)]
        final_mask = np.load(mask_path, mmap_mode="r")
        candidate_path = candidates_dir / "green_single_cfar_candidate_mask.npy"
        np.save(candidate_path, np.asarray(final_mask, dtype=np.uint8))
        latest_mask_path = candidates_dir / "candidate_mask.npy"
        np.save(latest_mask_path, np.asarray(final_mask, dtype=np.uint8))
        final_summary = dict(mask_summary)
        final_summary.update({"active_fraction": float(np.mean(final_mask)), "combine_mode": "replace"})

        candidates = component_candidates(final_mask, z_projection, support_min_frames=support, seed_z=seed_z, min_area=min_area, max_area=max_area, ndimage=ndimage)
        candidate_json = candidates_dir / "roi_candidates.json"
        write_json_atomic(candidate_json, {"schema_version": 1, "candidates": candidates})
        traces = extract_traces(highpass, candidates)
        trace_json = traces_dir / "roi_traces.json"
        write_json_atomic(trace_json, {"schema_version": 1, "traces": traces})
        events = score_events(traces)
        event_json = events_dir / "kalman_candidate_events.json"
        write_json_atomic(event_json, events)
        ranked = rank_candidates(candidates, video_shape=highpass.shape, weights={"local_correlation_weight": 0.25, "event_support_weight": 0.35, "artifact_weight": -0.3})
        ranked_json = candidates_dir / "ranked_candidates.json"
        write_json_atomic(ranked_json, ranked)

        pipeline_run = green_single_pipeline_run_manifest(
            planned_run=planned_run,
            run_root=run_root,
            raw_path=args.source_npy,
            highpass_path=highpass_path,
            smoothed_path=smoothed_path,
            z_path=z_path,
            final_path=candidate_path,
            final_summary=final_summary,
            candidate_json=candidate_json,
            candidate_count=len(candidates),
            seed_z=seed_z,
            min_area=min_area,
            max_area=max_area,
            support_min_frames=support,
            trace_json=trace_json,
            event_json=event_json,
            event_count=len(events["events"]),
            ranked_json=ranked_json,
            shape=list(highpass.shape),
            raw_dtype=raw_dtype,
            highpass_sigma=highpass_sigma,
            smooth_sigma=smooth_sigma,
        )
        write_json_atomic(run_root / "pipeline_run.json", pipeline_run)
        summary_runs.append(
            {
                "run_id": run_id,
                "run_root": run_root.name,
                "status": "completed",
                "sweep_index": index - 1,
                "sweep_total": len(planned_runs),
                "sweep_parameters": planned_run.get("sweep", {}).get("parameters", []),
                "artifact_count": len(pipeline_run["artifacts"]),
                "candidate_count": len(candidates),
                "event_count": len(events["events"]),
                "active_fraction": final_summary.get("active_fraction"),
            }
        )
        print(f"completed {index:03d}/{len(planned_runs)}: {run_id} candidates={len(candidates)} events={len(events['events'])}", flush=True)

    sweep_summary = {
        "schema_version": 1,
        "dataset_id": planned.get("dataset_id", spec.get("dataset_id", "")),
        "sweep": planned.get("sweep", {}),
        "status": "completed",
        "total": len(summary_runs),
        "succeeded": len(summary_runs),
        "failed": 0,
        "runs": summary_runs,
    }
    write_json_atomic(args.sweep_root / "sweep_summary.json", sweep_summary)
    write_sweep_brief(
        args.sweep_root / "gamma_cfar_grid_brief.md",
        title=str(planned.get("sweep", {}).get("label") or "Green-excess single Gamma CFAR grid"),
        summary_runs=summary_runs,
    )
    print(json.dumps({"sweep_summary": str(args.sweep_root / "sweep_summary.json"), "runs": len(summary_runs)}, indent=2))


def run_green_excess_roi_state_grid(args: argparse.Namespace) -> None:
    """Run the green-excess ROI-state grid with CFAR plus projection-blob candidates."""
    import numpy as np
    from scipy import ndimage

    from neurobench.discovery.ranking import rank_candidates

    spec = load_json(args.spec)
    planned = planned_manifest_for_spec(spec)
    planned_runs = planned.get("runs", []) or []
    if not planned_runs:
        raise SystemExit("Sweep spec did not expand to any runs.")
    first_run = args.sweep_root / f"001_{safe_name(str(planned_runs[0]['run_id']))}"
    preprocessing = first_run / "artifacts" / "preprocessing"
    highpass_path = preprocessing / "highpass_video.npy"
    smoothed_path = preprocessing / "smoothed_video.npy"
    z_path = preprocessing / "z_stack.npy"
    ensure_shared_preprocessing(
        args=args,
        spec=spec,
        first_run=first_run,
        highpass_path=highpass_path,
        smoothed_path=smoothed_path,
        z_path=z_path,
    )

    args.sweep_root.mkdir(parents=True, exist_ok=True)
    shared_mask_root = args.sweep_root / "shared_masks"
    shared_mask_root.mkdir(parents=True, exist_ok=True)
    smoothed = np.load(smoothed_path, mmap_mode="r").astype(np.float32, copy=False)
    z_stack = np.load(z_path, mmap_mode="r").astype(np.float32, copy=False)
    highpass = np.load(highpass_path, mmap_mode="r").astype(np.float32, copy=False)
    source_video = np.load(args.source_npy, mmap_mode="r").astype(np.float32, copy=False)
    z_projection = np.max(z_stack, axis=0)

    projection_score_path = shared_mask_root / "green_projection_score.npy"
    if projection_score_path.exists():
        projection_score = np.load(projection_score_path, mmap_mode="r").astype(np.float32, copy=False)
    else:
        projection_score = projection_blob_evidence(smoothed)
        np.save(projection_score_path, projection_score)
        print(f"wrote shared projection score: {projection_score_path}", flush=True)

    cfar_specs = sorted(
        {
            (
                float(pipeline_stage_params(run, "green_single_cfar", "gamma_cfar").get("pfa", 0.04)),
                int(pipeline_stage_params(run, "green_single_cfar", "gamma_cfar").get("guard_px", 2)),
                int(pipeline_stage_params(run, "green_single_cfar", "gamma_cfar").get("training_radius_px", 12)),
            )
            for run in planned_runs
        }
    )
    cfar_masks: dict[tuple[float, int, int], tuple[Path, dict[str, Any]]] = {}
    for pfa, guard_px, training_radius_px in cfar_specs:
        key = f"green_state_pfa_{str(pfa).replace('.', 'p')}_guard_{guard_px}_radius_{training_radius_px}"
        path = shared_mask_root / f"{key}.npy"
        if path.exists():
            mask = np.load(path, mmap_mode="r")
            summary = {"active_fraction": float(np.mean(mask)), "pfa": pfa, "guard_px": guard_px, "training_radius_px": training_radius_px, "threshold_z": cfar_threshold(pfa), "combine_mode": "replace", "shape": list(mask.shape)}
        else:
            summary = write_chunked_cfar_mask(
                smoothed,
                path,
                pfa=pfa,
                guard_px=guard_px,
                training_radius_px=training_radius_px,
                chunk_frames=args.cfar_chunk_frames,
            )
        cfar_masks[(pfa, guard_px, training_radius_px)] = (path, summary)

    summary_runs = []
    raw_dtype = str(source_video.dtype)
    highpass_sigma = float(pipeline_stage_params(spec, "highpass", "temporal_highpass_gaussian").get("sigma_frames", 0.0))
    smooth_sigma = float(pipeline_stage_params(spec, "smooth", "spatial_gaussian").get("sigma_px", 0.8))
    activity_params = pipeline_stage_params(spec, "activity_states", "trace_event_scoring")
    sustained_z = float(activity_params.get("sustained_z", 1.2))
    tonic_z = float(activity_params.get("tonic_z", 2.0))
    peak_window_frames = int(activity_params.get("peak_window_frames", 1))
    for index, planned_run in enumerate(planned_runs, start=1):
        run_id = planned_run["run_id"]
        cfar_params = pipeline_stage_params(planned_run, "green_single_cfar", "gamma_cfar")
        component_params = pipeline_stage_params(planned_run, "components", "component_filter")
        pfa = float(cfar_params.get("pfa", 0.04))
        guard = int(cfar_params.get("guard_px", 2))
        radius = int(cfar_params.get("training_radius_px", 12))
        support = max(1, int(component_params.get("support_min_frames", 1)))
        seed_z = float(component_params.get("seed_z", 1.8))
        projection_blob_z = float(component_params.get("projection_blob_z", 1.5))
        min_area = int(component_params.get("min_area_px", 6))
        max_area = int(component_params.get("max_area_px", 450))
        run_root = args.sweep_root / f"{index:03d}_{safe_name(run_id)}"
        run_root.mkdir(parents=True, exist_ok=True)
        candidates_dir = run_root / "artifacts" / "candidates"
        traces_dir = run_root / "artifacts" / "traces"
        events_dir = run_root / "artifacts" / "events"
        for directory in (candidates_dir, traces_dir, events_dir):
            directory.mkdir(parents=True, exist_ok=True)

        mask_path, mask_summary = cfar_masks[(pfa, guard, radius)]
        final_mask = np.load(mask_path, mmap_mode="r")
        candidate_path = candidates_dir / "green_single_cfar_candidate_mask.npy"
        np.save(candidate_path, np.asarray(final_mask, dtype=np.uint8))
        latest_mask_path = candidates_dir / "candidate_mask.npy"
        np.save(latest_mask_path, np.asarray(final_mask, dtype=np.uint8))
        final_summary = dict(mask_summary)
        final_summary.update({"active_fraction": float(np.mean(final_mask)), "combine_mode": "replace"})

        projection_support = projection_blob_mask(projection_score, projection_blob_z=projection_blob_z)
        projection_path = candidates_dir / "green_projection_blob_mask.npy"
        np.save(projection_path, projection_support.astype(np.uint8, copy=False))
        projection_summary = {
            "active_fraction": float(np.mean(projection_support)),
            "projection_blob_z": projection_blob_z,
            "projection_mode": "max_of_p95_and_max_robust_z",
            "shape": list(projection_support.shape),
        }

        candidates = union_component_candidates(
            final_mask,
            z_projection,
            projection_score,
            projection_support,
            support_min_frames=support,
            seed_z=seed_z,
            min_area=min_area,
            max_area=max_area,
            ndimage=ndimage,
        )
        source_counts: dict[str, int] = {}
        for candidate in candidates:
            source_counts[str(candidate.get("candidate_source") or "unknown")] = source_counts.get(str(candidate.get("candidate_source") or "unknown"), 0) + 1
        candidate_json = candidates_dir / "roi_candidates.json"
        write_json_atomic(candidate_json, {"schema_version": 1, "candidates": candidates})
        traces = extract_traces(highpass, candidates)
        events = score_events(traces)
        traces = attach_activity_states(traces, events, sustained_z=sustained_z, tonic_z=tonic_z, peak_window_frames=peak_window_frames)
        trace_json = traces_dir / "roi_traces.json"
        write_json_atomic(trace_json, {"schema_version": 1, "traces": traces})
        event_json = events_dir / "kalman_candidate_events.json"
        write_json_atomic(event_json, events)
        ranked = rank_candidates(candidates, video_shape=highpass.shape, weights={"local_correlation_weight": 0.25, "event_support_weight": 0.35, "artifact_weight": -0.3})
        ranked_json = candidates_dir / "ranked_candidates.json"
        write_json_atomic(ranked_json, ranked)
        coverage = diagnostic_bright_blob_coverage(source_video, candidates, ndimage=ndimage)

        pipeline_run = green_roi_state_pipeline_run_manifest(
            planned_run=planned_run,
            run_root=run_root,
            raw_path=args.source_npy,
            highpass_path=highpass_path,
            smoothed_path=smoothed_path,
            z_path=z_path,
            final_path=candidate_path,
            final_summary=final_summary,
            projection_score_path=projection_score_path,
            projection_path=projection_path,
            projection_summary=projection_summary,
            candidate_json=candidate_json,
            candidate_count=len(candidates),
            seed_z=seed_z,
            projection_blob_z=projection_blob_z,
            min_area=min_area,
            max_area=max_area,
            support_min_frames=support,
            source_counts=source_counts,
            trace_json=trace_json,
            event_json=event_json,
            event_count=len(events["events"]),
            ranked_json=ranked_json,
            shape=list(highpass.shape),
            raw_dtype=raw_dtype,
            highpass_sigma=highpass_sigma,
            smooth_sigma=smooth_sigma,
            activity_params={"sustained_z": sustained_z, "tonic_z": tonic_z, "peak_window_frames": peak_window_frames},
        )
        write_json_atomic(run_root / "pipeline_run.json", pipeline_run)
        summary_runs.append(
            {
                "run_id": run_id,
                "run_root": run_root.name,
                "status": "completed",
                "sweep_index": index - 1,
                "sweep_total": len(planned_runs),
                "sweep_parameters": planned_run.get("sweep", {}).get("parameters", []),
                "artifact_count": len(pipeline_run["artifacts"]),
                "candidate_count": len(candidates),
                "candidate_source_counts": source_counts,
                "event_count": len(events["events"]),
                "active_fraction": final_summary.get("active_fraction"),
                "diagnostic_bright_blob_coverage": coverage,
            }
        )
        print(
            f"completed {index:03d}/{len(planned_runs)}: {run_id} candidates={len(candidates)} events={len(events['events'])} coverage={coverage['coverage']:.3f}",
            flush=True,
        )

    recommended = select_green_roi_state_default_run(summary_runs)
    for row in summary_runs:
        row["recommended_default"] = bool(recommended and row.get("run_id") == recommended.get("run_id"))
    sweep_summary = {
        "schema_version": 1,
        "dataset_id": planned.get("dataset_id", spec.get("dataset_id", "")),
        "sweep": planned.get("sweep", {}),
        "status": "completed",
        "total": len(summary_runs),
        "succeeded": len(summary_runs),
        "failed": 0,
        "recommended_run_id": recommended.get("run_id") if recommended else "",
        "selection_criteria": "max diagnostic bright-blob coverage with roi_count <= 300, then lower ROI count, then lower pfa",
        "runs": summary_runs,
    }
    write_json_atomic(args.sweep_root / "sweep_summary.json", sweep_summary)
    write_sweep_brief(
        args.sweep_root / "gamma_cfar_grid_brief.md",
        title=str(planned.get("sweep", {}).get("label") or "Green-excess ROI-state grid"),
        summary_runs=summary_runs,
    )
    print(json.dumps({"sweep_summary": str(args.sweep_root / "sweep_summary.json"), "runs": len(summary_runs), "recommended_run_id": sweep_summary["recommended_run_id"]}, indent=2))


def available_ram_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                kb = float(line.split()[1])
                return kb / (1024.0 * 1024.0)
    except OSError:
        return 0.0
    return 0.0


def validate_overnight_resources(path: Path) -> None:
    import shutil

    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    free_disk_gb = usage.free / (1024.0**3)
    if free_disk_gb < MIN_OVERNIGHT_FREE_DISK_GB:
        raise RuntimeError(
            f"Refusing overnight run: {free_disk_gb:.1f} GiB free at {path}, "
            f"requires at least {MIN_OVERNIGHT_FREE_DISK_GB:.1f} GiB."
        )
    free_ram_gb = available_ram_gb()
    if free_ram_gb and free_ram_gb < MIN_OVERNIGHT_AVAILABLE_RAM_GB:
        raise RuntimeError(
            f"Refusing overnight run: {free_ram_gb:.1f} GiB available RAM, "
            f"requires at least {MIN_OVERNIGHT_AVAILABLE_RAM_GB:.1f} GiB."
        )


def require_cupy_cuda() -> dict[str, Any]:
    try:
        import cupy as cp  # type: ignore
        from cupyx.scipy.ndimage import gaussian_filter, uniform_filter  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("CUDA execution requires CuPy and cupyx.scipy.ndimage; CPU fallback is disabled for this runner.") from exc
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError(
            "CUDA execution requested, but CuPy cannot access a CUDA device. "
            "Check nvidia-smi, driver modules, and /dev/nvidia*. CPU fallback is disabled."
        ) from exc
    if device_count < 1:
        raise RuntimeError("CUDA execution requested, but CuPy reports zero CUDA devices. CPU fallback is disabled.")
    try:
        sample = cp.ones((1, 4, 4), dtype=cp.float32)
        filtered = uniform_filter(sample, size=(1, 3, 3), mode="nearest")
        value = float(filtered.sum().get())
        if value <= 0:
            raise RuntimeError("CuPy uniform_filter smoke test produced an invalid result.")
    except Exception as exc:
        raise RuntimeError("CUDA execution requested, but the CuPy uniform_filter smoke test failed. CPU fallback is disabled.") from exc
    device = cp.cuda.Device()
    free_vram, total_vram = device.mem_info
    return {
        "cp": cp,
        "uniform_filter": uniform_filter,
        "gaussian_filter": gaussian_filter,
        "report": {
            "backend": "cupy_cuda",
            "device_id": int(device.id),
            "device_count": device_count,
            "free_vram_gb": round(float(free_vram) / (1024.0**3), 6),
            "total_vram_gb": round(float(total_vram) / (1024.0**3), 6),
            "cupy_version": getattr(cp, "__version__", ""),
        },
    }


def free_cupy_blocks(gpu: Mapping[str, Any]) -> None:
    cp = gpu["cp"]
    cp.get_default_memory_pool().free_all_blocks()
    try:
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def ensure_shared_preprocessing_gpu(
    *,
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    first_run: Path,
    highpass_path: Path,
    smoothed_path: Path,
    z_path: Path,
    gpu: Mapping[str, Any],
) -> None:
    """Create shared preprocessing arrays with CUDA/CuPy in bounded chunks."""
    if highpass_path.exists() and smoothed_path.exists() and z_path.exists():
        return
    import numpy as np

    cp = gpu["cp"]
    gaussian_filter = gpu["gaussian_filter"]
    first_run.mkdir(parents=True, exist_ok=True)
    highpass_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.source_npy.exists():
        raise SystemExit(f"Source .npy not found: {args.source_npy}")

    source_cpu = np.load(args.source_npy, mmap_mode="r")
    shape = tuple(int(value) for value in source_cpu.shape)
    chunk_frames = max(1, int(getattr(args, "gpu_preprocess_chunk_frames", getattr(args, "gpu_cfar_chunk_frames", 32))))
    highpass_params = pipeline_stage_params(spec, "highpass", "temporal_highpass_gaussian")
    sigma_frames = float(highpass_params.get("sigma_frames", 0.0))
    if sigma_frames != 0.0:
        raise RuntimeError("GPU overnight preprocessing currently requires highpass sigma_frames=0.0 to avoid full-volume temporal filtering.")

    if not highpass_path.exists():
        highpass_out = np.lib.format.open_memmap(highpass_path, mode="w+", dtype=np.float32, shape=shape)
        for start in range(0, shape[0], chunk_frames):
            stop = min(shape[0], start + chunk_frames)
            block_gpu = cp.asarray(np.asarray(source_cpu[start:stop], dtype=np.float32), dtype=cp.float32)
            highpass_out[start:stop] = cp.asnumpy(block_gpu)
            highpass_out.flush()
            del block_gpu
            free_cupy_blocks(gpu)
            print(f"  GPU highpass frames {start + 1}-{stop}/{shape[0]}", flush=True)
        highpass_out.flush()
        print(f"wrote shared GPU highpass: {highpass_path}", flush=True)

    highpass_cpu = np.load(highpass_path, mmap_mode="r")
    smooth_params = pipeline_stage_params(spec, "smooth", "spatial_gaussian")
    sigma_px = float(smooth_params.get("sigma_px", 0.8))
    if not smoothed_path.exists():
        smoothed_out = np.lib.format.open_memmap(smoothed_path, mode="w+", dtype=np.float32, shape=shape)
        for start in range(0, shape[0], chunk_frames):
            stop = min(shape[0], start + chunk_frames)
            block_gpu = cp.asarray(np.asarray(highpass_cpu[start:stop], dtype=np.float32), dtype=cp.float32)
            if sigma_px > 0:
                smoothed_gpu = gaussian_filter(block_gpu, sigma=(0.0, sigma_px, sigma_px), mode="nearest").astype(cp.float32, copy=False)
            else:
                smoothed_gpu = block_gpu.astype(cp.float32, copy=True)
            smoothed_out[start:stop] = cp.asnumpy(smoothed_gpu)
            smoothed_out.flush()
            del block_gpu, smoothed_gpu
            free_cupy_blocks(gpu)
            print(f"  GPU smooth frames {start + 1}-{stop}/{shape[0]}", flush=True)
        smoothed_out.flush()
        print(f"wrote shared GPU smoothed video: {smoothed_path}", flush=True)

    if not z_path.exists():
        score_params = pipeline_stage_params(spec, "score", "robust_positive_local_z")
        epsilon = float(score_params.get("epsilon", 1.0))
        z_out = np.lib.format.open_memmap(z_path, mode="w+", dtype=np.float32, shape=shape)
        for start in range(0, shape[0], chunk_frames):
            stop = min(shape[0], start + chunk_frames)
            block_gpu = cp.asarray(np.asarray(highpass_cpu[start:stop], dtype=np.float32), dtype=cp.float32)
            frame_median = cp.median(block_gpu, axis=(1, 2), keepdims=True)
            mad = cp.median(cp.abs(block_gpu - frame_median), axis=(1, 2), keepdims=True)
            z_gpu = cp.maximum((block_gpu - frame_median) / (1.4826 * mad + epsilon), 0.0).astype(cp.float32, copy=False)
            z_out[start:stop] = cp.asnumpy(z_gpu)
            z_out.flush()
            del block_gpu, frame_median, mad, z_gpu
            free_cupy_blocks(gpu)
            print(f"  GPU z-stack frames {start + 1}-{stop}/{shape[0]}", flush=True)
        z_out.flush()
        print(f"wrote shared GPU z-stack: {z_path}", flush=True)
    free_cupy_blocks(gpu)

def write_chunked_cfar_mask_gpu(
    video,
    path: Path,
    *,
    pfa: float,
    guard_px: int,
    training_radius_px: int,
    chunk_frames: int,
    gpu: Mapping[str, Any],
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    import numpy as np
    import torch  # type: ignore
    import torch.nn.functional as functional  # type: ignore

    if training_radius_px <= guard_px:
        raise ValueError("training_radius_px must be larger than guard_px.")
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("Torch CUDA is required for GPU CFAR mask generation.")

    def box_mean(tensor, radius: int):
        if radius == 0:
            return tensor.clone()
        window = 2 * radius + 1
        frames = tensor.unsqueeze(1)
        padded = functional.pad(frames, (radius, radius, radius, radius), mode="replicate")
        return functional.avg_pool2d(padded, kernel_size=window, stride=1).squeeze(1)

    threshold_z = cfar_threshold(pfa)
    chunk_frames = max(1, int(chunk_frames))
    shape = tuple(int(value) for value in video.shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
    outer_area = float((2 * training_radius_px + 1) ** 2)
    guard_area = float((2 * guard_px + 1) ** 2)
    training_area = outer_area - guard_area
    active = 0
    total = int(np.prod(shape))
    for start in range(0, shape[0], chunk_frames):
        stop = min(shape[0], start + chunk_frames)
        block_np = np.asarray(video[start:stop], dtype=np.float32).copy()
        with torch.no_grad():
            evidence = torch.as_tensor(block_np, dtype=torch.float32, device="cuda").clamp_min(0.0)
            outer_mean = box_mean(evidence, training_radius_px)
            outer_sq_mean = box_mean(evidence * evidence, training_radius_px)
            guard_mean = box_mean(evidence, guard_px)
            guard_sq_mean = box_mean(evidence * evidence, guard_px)
            local_mean = ((outer_mean * outer_area) - (guard_mean * guard_area)) / training_area
            local_sq_mean = ((outer_sq_mean * outer_area) - (guard_sq_mean * guard_area)) / training_area
            local_var = (local_sq_mean - (local_mean * local_mean)).clamp_min(0.0)
            local_std = torch.sqrt(local_var + float(epsilon))
            score = ((evidence - local_mean) / (local_std + float(epsilon))).clamp_min(0.0)
            mask_gpu = score >= threshold_z
            active += int(mask_gpu.sum().item())
            output[start:stop] = mask_gpu.to(dtype=torch.uint8).detach().cpu().numpy()
            del evidence, outer_mean, outer_sq_mean, guard_mean, guard_sq_mean, local_mean, local_sq_mean, local_var, local_std, score, mask_gpu
        output.flush()
        torch.cuda.empty_cache()
        free_cupy_blocks(gpu)
        print(f"  GPU CFAR pfa={pfa} radius={training_radius_px}: frames {start + 1}-{stop}/{shape[0]}", flush=True)
    output.flush()
    return {
        "active_fraction": float(active / total) if total else 0.0,
        "pfa": pfa,
        "guard_px": guard_px,
        "training_radius_px": training_radius_px,
        "threshold_z": threshold_z,
        "combine_mode": "replace",
        "shape": list(shape),
        "backend": "torch_cuda",
        "chunk_frames": chunk_frames,
    }


def write_chunked_cfar_mask_gpu_with_retries(video, path: Path, *, pfa: float, guard_px: int, training_radius_px: int, chunk_frames: int, gpu: Mapping[str, Any]) -> dict[str, Any]:
    attempts = []
    for value in (int(chunk_frames), 16, 8):
        if value >= 1 and value not in attempts:
            attempts.append(value)
    last_error: Exception | None = None
    cp = gpu["cp"]
    oom_error = getattr(cp.cuda.memory, "OutOfMemoryError", RuntimeError)
    for attempt in attempts:
        try:
            return write_chunked_cfar_mask_gpu(video, path, pfa=pfa, guard_px=guard_px, training_radius_px=training_radius_px, chunk_frames=attempt, gpu=gpu)
        except oom_error as exc:
            last_error = exc
            if path.exists():
                path.unlink()
            free_cupy_blocks(gpu)
            print(f"  GPU OOM at chunk_frames={attempt}; retrying smaller chunk if available", flush=True)
    raise RuntimeError(f"GPU CFAR failed after chunk-size retries for {path}.") from last_error


def write_fused_mask_gpu(small_path: Path, large_path: Path, final_path: Path, *, fusion_mode: str, chunk_frames: int, gpu: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    cp = gpu["cp"]
    small = np.load(small_path, mmap_mode="r")
    large = np.load(large_path, mmap_mode="r")
    if tuple(small.shape) != tuple(large.shape):
        raise ValueError(f"Cannot fuse CFAR masks with different shapes: {small.shape} vs {large.shape}")
    mode = str(fusion_mode).strip().lower()
    if mode not in {"intersection", "union"}:
        raise ValueError("fusion_mode must be intersection or union.")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(final_path, mode="w+", dtype=np.uint8, shape=tuple(small.shape))
    active = 0
    total = int(np.prod(small.shape))
    for start in range(0, int(small.shape[0]), max(1, int(chunk_frames))):
        stop = min(int(small.shape[0]), start + max(1, int(chunk_frames)))
        small_gpu = cp.asarray(np.asarray(small[start:stop], dtype=np.uint8), dtype=cp.bool_)
        large_gpu = cp.asarray(np.asarray(large[start:stop], dtype=np.uint8), dtype=cp.bool_)
        fused = small_gpu & large_gpu if mode == "intersection" else small_gpu | large_gpu
        active += int(cp.count_nonzero(fused).get())
        output[start:stop] = cp.asnumpy(fused.astype(cp.uint8, copy=False))
        output.flush()
        del small_gpu, large_gpu, fused
        free_cupy_blocks(gpu)
    output.flush()
    return {"active_fraction": float(active / total) if total else 0.0, "combine_mode": mode, "shape": list(small.shape), "backend": "cupy_cuda"}


def median_float(values) -> float:
    sorted_values = sorted(float(value) for value in values)
    count = len(sorted_values)
    if count == 0:
        return 0.0
    midpoint = count // 2
    if count % 2:
        return sorted_values[midpoint]
    return 0.5 * (sorted_values[midpoint - 1] + sorted_values[midpoint])


def robust_positive_z_map_gpu(frame_gpu, *, gpu: Mapping[str, Any]):
    import numpy as np

    cp = gpu["cp"]
    # CuPy 14.1.1 + CUDA 13 can segfault while NVRTC-compiling median on this
    # projection-sized array. The projection is small, so compute the robust
    # scale on host and return a CuPy array for the surrounding GPU pipeline.
    values_cpu = cp.asnumpy(cp.asarray(frame_gpu, dtype=cp.float32)).astype(np.float32, copy=False)
    center = median_float(values_cpu.ravel())
    mad = median_float(np.abs(values_cpu - center).ravel())
    scale = 1.4826 * mad
    if scale < 1e-6:
        scale = float(np.std(values_cpu))
    if scale < 1e-6:
        scale = 1.0
    score = np.maximum((values_cpu - center) / scale, 0.0).astype(np.float32, copy=False)
    return cp.asarray(score, dtype=cp.float32)


def projection_blob_evidence_gpu(video, path: Path, *, gpu: Mapping[str, Any]):
    import numpy as np

    if path.exists():
        return np.load(path, mmap_mode="r").astype(np.float32, copy=False)
    cp = gpu["cp"]
    positive = cp.maximum(cp.asarray(video, dtype=cp.float32), 0.0)
    p95_projection = cp.percentile(positive, 95, axis=0).astype(cp.float32, copy=False)
    max_projection = cp.max(positive, axis=0).astype(cp.float32, copy=False)
    evidence = cp.maximum(
        robust_positive_z_map_gpu(p95_projection, gpu=gpu),
        robust_positive_z_map_gpu(max_projection, gpu=gpu),
    ).astype(cp.float32, copy=False)
    result = cp.asnumpy(evidence)
    np.save(path, result)
    print(f"wrote shared GPU projection score: {path}", flush=True)
    free_cupy_blocks(gpu)
    return np.load(path, mmap_mode="r").astype(np.float32, copy=False)


def support_map_from_mask_gpu(mask, *, gpu: Mapping[str, Any]):
    import numpy as np

    cp = gpu["cp"]
    shape = tuple(int(value) for value in mask.shape)
    chunk_frames = 32
    support = cp.zeros(shape[1:], dtype=cp.float32)
    for start in range(0, shape[0], chunk_frames):
        stop = min(shape[0], start + chunk_frames)
        block_gpu = cp.asarray(np.asarray(mask[start:stop], dtype=np.uint8), dtype=cp.uint8)
        support += cp.sum(block_gpu, axis=0, dtype=cp.float32)
        del block_gpu
        free_cupy_blocks(gpu)
    result = cp.asnumpy(support).astype(np.float32, copy=False)
    del support
    free_cupy_blocks(gpu)
    return result


def union_component_candidates_from_support(
    support,
    z_projection,
    projection_score,
    projection_support,
    *,
    support_min_frames: int,
    seed_z: float,
    min_area: int,
    max_area: int,
    ndimage: Any,
    split_large_components: bool = False,
    split_min_distance_px: int = 6,
    split_area_px: int = 80,
    split_max_peaks: int = 40,
) -> list[dict[str, Any]]:
    import numpy as np

    support = np.asarray(support, dtype=np.float32)
    z_map = np.asarray(z_projection, dtype=np.float32)
    projection_map = np.asarray(projection_score, dtype=np.float32)
    projection_footprint = np.asarray(projection_support, dtype=bool)
    cfar_footprint = (support >= float(support_min_frames)) & (z_map >= float(seed_z))
    footprint = cfar_footprint | projection_footprint
    labels, _ = ndimage.label(footprint)
    objects = ndimage.find_objects(labels)
    candidates: list[dict[str, Any]] = []
    score_map = np.maximum(z_map, projection_map).astype(np.float32, copy=False)
    split_distance = max(1, int(split_min_distance_px))
    split_area = max(int(min_area), int(split_area_px))
    split_limit = max(1, int(split_max_peaks))

    def source_for(cfar_pixels: int, projection_pixels: int) -> str:
        if cfar_pixels and projection_pixels:
            return "union"
        if projection_pixels:
            return "projection_blob"
        return "cfar_support"

    def append_candidate(
        *,
        x: float,
        y: float,
        area_px: int,
        bbox: list[int],
        source: str,
        source_pixels: dict[str, int],
        component_z,
        component_projection,
        component_support,
        parent_area_px: int | None = None,
        split_index: int | None = None,
    ) -> None:
        row = {
            "id": f"roi_{len(candidates) + 1:03d}",
            "x": float(x),
            "y": float(y),
            "area_px": int(area_px),
            "peak_z": float(np.max(component_z)) if component_z.size else 0.0,
            "projection_z": float(np.max(component_projection)) if component_projection.size else 0.0,
            "support_frames": int(np.max(component_support)) if component_support.size else 0,
            "candidate_source": source,
            "source_pixels": source_pixels,
            "bbox": bbox,
        }
        if parent_area_px is not None:
            row["parent_area_px"] = int(parent_area_px)
            row["split_from_large_component"] = True
        if split_index is not None:
            row["split_index"] = int(split_index)
        candidates.append(row)

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
        cfar_pixels = int(np.count_nonzero(cfar_footprint[slices] & component))
        projection_pixels = int(np.count_nonzero(projection_footprint[slices] & component))
        source = source_for(cfar_pixels, projection_pixels)
        component_z = z_map[slices][component]
        component_projection = projection_map[slices][component]
        component_support = support[slices][component]
        bbox = [int(np.min(abs_xs)), int(np.min(abs_ys)), int(np.max(abs_xs)), int(np.max(abs_ys))]
        if area <= max_area:
            append_candidate(
                x=float(np.mean(abs_xs)),
                y=float(np.mean(abs_ys)),
                area_px=area,
                bbox=bbox,
                source=source,
                source_pixels={"cfar": cfar_pixels, "projection": projection_pixels},
                component_z=component_z,
                component_projection=component_projection,
                component_support=component_support,
            )
            continue
        if not split_large_components:
            continue

        subscore = np.where(component, score_map[slices], -np.inf)
        maxima = subscore == ndimage.maximum_filter(subscore, size=max(3, split_distance), mode="nearest")
        maxima &= component
        peak_ys, peak_xs = np.nonzero(maxima)
        peaks: list[tuple[float, int, int]] = []
        for peak_y, peak_x in zip(peak_ys, peak_xs):
            value = float(subscore[peak_y, peak_x])
            if value <= 0.0:
                continue
            peaks.append((value, int(peak_y), int(peak_x)))
        peaks.sort(reverse=True)
        accepted: list[tuple[int, int]] = []
        for _, peak_y, peak_x in peaks:
            global_x = peak_x + x0
            global_y = peak_y + y0
            if all((global_x - other_x) ** 2 + (global_y - other_y) ** 2 >= float(split_distance**2) for other_x, other_y in accepted):
                accepted.append((global_x, global_y))
            if len(accepted) >= min(split_limit, max(1, int(math.ceil(area / float(split_area))))):
                break
        if not accepted:
            accepted = [(int(round(float(np.mean(abs_xs)))), int(round(float(np.mean(abs_ys)))))]
        radius = max(1.0, math.sqrt(float(split_area) / math.pi))
        for split_index, (global_x, global_y) in enumerate(accepted, start=1):
            x_min = max(0, int(math.floor(global_x - radius)))
            y_min = max(0, int(math.floor(global_y - radius)))
            x_max = min(score_map.shape[1] - 1, int(math.ceil(global_x + radius)))
            y_max = min(score_map.shape[0] - 1, int(math.ceil(global_y + radius)))
            local = (slice(y_min, y_max + 1), slice(x_min, x_max + 1))
            local_mask = footprint[local]
            if np.any(local_mask):
                local_z = z_map[local][local_mask]
                local_projection = projection_map[local][local_mask]
                local_support = support[local][local_mask]
                local_cfar_pixels = int(np.count_nonzero(cfar_footprint[local] & local_mask))
                local_projection_pixels = int(np.count_nonzero(projection_footprint[local] & local_mask))
            else:
                local_z = np.asarray([z_map[global_y, global_x]], dtype=np.float32)
                local_projection = np.asarray([projection_map[global_y, global_x]], dtype=np.float32)
                local_support = np.asarray([support[global_y, global_x]], dtype=np.float32)
                local_cfar_pixels = int(bool(cfar_footprint[global_y, global_x]))
                local_projection_pixels = int(bool(projection_footprint[global_y, global_x]))
            append_candidate(
                x=float(global_x),
                y=float(global_y),
                area_px=split_area,
                bbox=[x_min, y_min, x_max, y_max],
                source=f"{source}_split",
                source_pixels={"cfar": local_cfar_pixels, "projection": local_projection_pixels, "parent_cfar": cfar_pixels, "parent_projection": projection_pixels},
                component_z=local_z,
                component_projection=local_projection,
                component_support=local_support,
                parent_area_px=area,
                split_index=split_index,
            )
    return candidates


def extract_traces_gpu(gpu: Mapping[str, Any], video, candidates: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cp = gpu["cp"]
    video_gpu = cp.asarray(video, dtype=cp.float32)
    height, width = video_gpu.shape[1:]
    y_grid, x_grid = cp.mgrid[0:height, 0:width]
    traces = []
    for candidate in candidates:
        cx = float(candidate.get("x", 0.0))
        cy = float(candidate.get("y", 0.0))
        area = max(1.0, float(candidate.get("area_px", 1.0)))
        inner_radius = max(1.0, math.sqrt(area / math.pi))
        distances = cp.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        roi_mask = distances <= inner_radius
        ring_mask = (distances > inner_radius + 1.0) & (distances <= max(inner_radius + 2.0, 15.0))
        raw_trace = cp.asnumpy(video_gpu[:, roi_mask].mean(axis=1)).astype("float32")
        if bool(cp.any(ring_mask).get()):
            background_trace = cp.asnumpy(video_gpu[:, ring_mask].mean(axis=1)).astype("float32")
        else:
            import numpy as np

            background_trace = np.zeros_like(raw_trace)
        corrected = raw_trace - 0.7 * background_trace
        traces.append(
            {
                "roi_id": str(candidate.get("id")),
                "x": cx,
                "y": cy,
                "area_px": area,
                "inner_radius_px": inner_radius,
                "outer_radius_px": 15,
                "neuropil_weight": 0.7,
                "raw_trace": [float(value) for value in raw_trace],
                "background_trace": [float(value) for value in background_trace],
                "corrected_trace": [float(value) for value in corrected],
            }
        )
    del video_gpu, y_grid, x_grid
    free_cupy_blocks(gpu)
    return traces


def max_projection_from_stack_gpu(stack, *, gpu: Mapping[str, Any], chunk_frames: int):
    import numpy as np

    cp = gpu["cp"]
    shape = tuple(int(value) for value in stack.shape)
    projection = None
    chunk_frames = max(1, int(chunk_frames))
    for start in range(0, shape[0], chunk_frames):
        stop = min(shape[0], start + chunk_frames)
        block_gpu = cp.asarray(np.asarray(stack[start:stop], dtype=np.float32), dtype=cp.float32)
        block_projection = cp.max(block_gpu, axis=0).astype(cp.float32, copy=False)
        projection = block_projection if projection is None else cp.maximum(projection, block_projection).astype(cp.float32, copy=False)
        del block_gpu, block_projection
        free_cupy_blocks(gpu)
    if projection is None:
        projection = cp.zeros(shape[1:], dtype=cp.float32)
    result = cp.asnumpy(projection).astype(np.float32, copy=False)
    del projection
    free_cupy_blocks(gpu)
    return result


def green_multiscale_summary_row_from_pipeline_run(
    *,
    planned_run: Mapping[str, Any],
    run_root: Path,
    pipeline_run: Mapping[str, Any],
    index: int,
    total: int,
    existing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts = list(pipeline_run.get("artifacts") or [])

    def artifact_summary(*, kind: str | None = None, artifact_id: str | None = None) -> dict[str, Any]:
        matches = []
        for artifact in artifacts:
            if kind is not None and artifact.get("kind") != kind:
                continue
            if artifact_id is not None and artifact.get("artifact_id") != artifact_id:
                continue
            matches.append(artifact)
        if not matches:
            return {}
        return dict(matches[-1].get("summary") or {})

    candidate_summary = artifact_summary(kind="roi_candidates")
    event_summary = artifact_summary(kind="candidate_events")
    mask_summary = artifact_summary(artifact_id="cfar_large_ref_candidate_mask.v1")
    row = {
        "run_id": planned_run["run_id"],
        "run_root": run_root.name,
        "status": str(pipeline_run.get("status") or "completed"),
        "sweep_index": index - 1,
        "sweep_total": total,
        "sweep_parameters": planned_run.get("sweep", {}).get("parameters", []),
        "artifact_count": len(artifacts),
        "candidate_count": int(candidate_summary.get("count") or 0),
        "candidate_source_counts": dict(candidate_summary.get("candidate_source_counts") or {}),
        "event_count": int(event_summary.get("event_count") or 0),
        "active_fraction": mask_summary.get("active_fraction"),
    }
    if existing and existing.get("diagnostic_bright_blob_coverage") is not None:
        row["diagnostic_bright_blob_coverage"] = existing.get("diagnostic_bright_blob_coverage")
    return row


def select_green_multiscale_default_run(summary_runs: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not summary_runs:
        return None
    eligible = [row for row in summary_runs if int(row.get("candidate_count") or 0) <= 300]
    if not eligible:
        eligible = list(summary_runs)

    def pfa_value(row: Mapping[str, Any]) -> float:
        for item in row.get("sweep_parameters", []) or []:
            if item.get("stage") == "cfar_small_ref" and item.get("param") == "pfa":
                try:
                    return float(item.get("value"))
                except (TypeError, ValueError):
                    return 999.0
        return 999.0

    def fusion_value(row: Mapping[str, Any]) -> int:
        for item in row.get("sweep_parameters", []) or []:
            if item.get("stage") == "components" and item.get("param") == "fusion_mode":
                return 0 if str(item.get("value") or "") == "intersection" else 1
        return 1

    def coverage_value(row: Mapping[str, Any]) -> float:
        payload = row.get("diagnostic_bright_blob_coverage") or {}
        try:
            return float(payload.get("coverage", 0.0))
        except (AttributeError, TypeError, ValueError):
            return 0.0

    return sorted(
        eligible,
        key=lambda row: (
            -coverage_value(row),
            int(row.get("candidate_count") or 0),
            pfa_value(row),
            fusion_value(row),
            str(row.get("run_id") or ""),
        ),
    )[0]


def write_green_multiscale_sweep_summary(
    path: Path,
    *,
    planned: Mapping[str, Any],
    spec: Mapping[str, Any],
    summary_runs: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    recommended = select_green_multiscale_default_run(summary_runs)
    for row in summary_runs:
        row["recommended_default"] = bool(recommended and row.get("run_id") == recommended.get("run_id"))
    sweep_summary = {
        "schema_version": 1,
        "dataset_id": planned.get("dataset_id", spec.get("dataset_id", "")),
        "sweep": planned.get("sweep", {}),
        "status": status,
        "total": len(planned.get("runs", []) or []),
        "succeeded": sum(1 for row in summary_runs if row.get("status") == "completed"),
        "failed": sum(1 for row in summary_runs if row.get("status") == "failed"),
        "recommended_run_id": recommended.get("run_id") if recommended else "",
        "selection_criteria": "max diagnostic bright-blob coverage with roi_count <= 300, then lower ROI count, lower pfa, and intersection before union",
        "runs": summary_runs,
    }
    write_json_atomic(path, sweep_summary)
    return sweep_summary


def hardlink_or_copy_npy(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        import shutil

        shutil.copy2(source, target)


def run_green_excess_multiscale_cfar_grid(args: argparse.Namespace) -> None:
    """Run the GPU-only green-excess multiscale Gamma-CFAR overnight grid."""
    import numpy as np
    from scipy import ndimage

    from neurobench.discovery.ranking import rank_candidates

    validate_overnight_resources(args.sweep_root)
    gpu = require_cupy_cuda()
    spec = load_json(args.spec)
    planned = planned_manifest_for_spec(spec)
    planned_runs = planned.get("runs", []) or []
    if len(planned_runs) != 144:
        raise SystemExit(f"Expected 144 planned multiscale runs, got {len(planned_runs)}.")
    if not planned_runs:
        raise SystemExit("Sweep spec did not expand to any runs.")
    args.sweep_root.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"cuda": gpu["report"], "planned_runs": len(planned_runs)}, indent=2), flush=True)

    first_run = args.sweep_root / f"001_{safe_name(str(planned_runs[0]['run_id']))}"
    preprocessing = first_run / "artifacts" / "preprocessing"
    highpass_path = preprocessing / "highpass_video.npy"
    smoothed_path = preprocessing / "smoothed_video.npy"
    z_path = preprocessing / "z_stack.npy"
    ensure_shared_preprocessing_gpu(
        args=args,
        spec=spec,
        first_run=first_run,
        highpass_path=highpass_path,
        smoothed_path=smoothed_path,
        z_path=z_path,
        gpu=gpu,
    )

    shared_mask_root = args.sweep_root / "shared_masks"
    shared_mask_root.mkdir(parents=True, exist_ok=True)
    smoothed = np.load(smoothed_path, mmap_mode="r")
    z_stack = np.load(z_path, mmap_mode="r")
    highpass = np.load(highpass_path, mmap_mode="r")
    source_video = np.load(args.source_npy, mmap_mode="r")
    raw_dtype = str(source_video.dtype)
    chunk_frames = max(1, int(args.gpu_cfar_chunk_frames))
    z_projection = max_projection_from_stack_gpu(z_stack, gpu=gpu, chunk_frames=chunk_frames)

    projection_score_path = shared_mask_root / "green_projection_score.npy"
    projection_score = projection_blob_evidence_gpu(smoothed, projection_score_path, gpu=gpu)

    small_specs = sorted(
        {
            (
                float(pipeline_stage_params(run, "cfar_small_ref", "gamma_cfar").get("pfa", 0.04)),
                int(pipeline_stage_params(run, "cfar_small_ref", "gamma_cfar").get("guard_px", 2)),
                int(pipeline_stage_params(run, "cfar_small_ref", "gamma_cfar").get("training_radius_px", 6)),
            )
            for run in planned_runs
        }
    )
    small_masks: dict[tuple[float, int, int], tuple[Path, dict[str, Any]]] = {}
    for pfa, guard_px, training_radius_px in small_specs:
        key = f"green_mscfar_small_pfa_{str(pfa).replace('.', 'p')}_guard_{guard_px}_radius_{training_radius_px}"
        path = shared_mask_root / f"{key}.npy"
        if path.exists():
            mask = np.load(path, mmap_mode="r")
            summary = {
                "active_fraction": float(np.mean(mask)),
                "pfa": pfa,
                "guard_px": guard_px,
                "training_radius_px": training_radius_px,
                "threshold_z": cfar_threshold(pfa),
                "combine_mode": "replace",
                "shape": list(mask.shape),
                "backend": "cupy_cuda",
            }
        else:
            summary = write_chunked_cfar_mask_gpu_with_retries(
                smoothed,
                path,
                pfa=pfa,
                guard_px=guard_px,
                training_radius_px=training_radius_px,
                chunk_frames=chunk_frames,
                gpu=gpu,
            )
        small_masks[(pfa, guard_px, training_radius_px)] = (path, summary)

    large_specs = sorted(
        {
            (
                float(pipeline_stage_params(run, "cfar_large_ref", "gamma_cfar").get("pfa", 0.04)),
                int(pipeline_stage_params(run, "cfar_large_ref", "gamma_cfar").get("guard_px", 2)),
                int(pipeline_stage_params(run, "cfar_large_ref", "gamma_cfar").get("training_radius_px", 18)),
            )
            for run in planned_runs
        }
    )
    large_masks: dict[tuple[float, int, int], tuple[Path, dict[str, Any]]] = {}
    for pfa, guard_px, training_radius_px in large_specs:
        key = f"green_mscfar_large_pfa_{str(pfa).replace('.', 'p')}_guard_{guard_px}_radius_{training_radius_px}"
        path = shared_mask_root / f"{key}.npy"
        if path.exists():
            mask = np.load(path, mmap_mode="r")
            summary = {
                "active_fraction": float(np.mean(mask)),
                "pfa": pfa,
                "guard_px": guard_px,
                "training_radius_px": training_radius_px,
                "threshold_z": cfar_threshold(pfa),
                "combine_mode": "replace",
                "shape": list(mask.shape),
                "backend": "cupy_cuda",
            }
        else:
            summary = write_chunked_cfar_mask_gpu_with_retries(
                smoothed,
                path,
                pfa=pfa,
                guard_px=guard_px,
                training_radius_px=training_radius_px,
                chunk_frames=chunk_frames,
                gpu=gpu,
            )
        large_masks[(pfa, guard_px, training_radius_px)] = (path, summary)

    summary_path = args.sweep_root / "sweep_summary.json"
    existing_rows_by_id: dict[str, Mapping[str, Any]] = {}
    if summary_path.exists():
        existing_summary = load_json(summary_path)
        existing_rows_by_id = {str(row.get("run_id") or ""): row for row in existing_summary.get("runs", []) or []}

    summary_runs: list[dict[str, Any]] = []
    highpass_sigma = float(pipeline_stage_params(spec, "highpass", "temporal_highpass_gaussian").get("sigma_frames", 0.0))
    smooth_sigma = float(pipeline_stage_params(spec, "smooth", "spatial_gaussian").get("sigma_px", 0.8))
    activity_params = pipeline_stage_params(spec, "activity_states", "trace_event_scoring")
    sustained_z = float(activity_params.get("sustained_z", 1.2))
    tonic_z = float(activity_params.get("tonic_z", 2.0))
    peak_window_frames = int(activity_params.get("peak_window_frames", 1))

    for index, planned_run in enumerate(planned_runs, start=1):
        identity = green_multiscale_run_identity(planned_run)
        run_id = str(planned_run["run_id"])
        run_root = args.sweep_root / f"{index:03d}_{safe_name(run_id)}"
        pipeline_run_path = run_root / "pipeline_run.json"
        existing_row = existing_rows_by_id.get(run_id)
        if pipeline_run_path.exists():
            pipeline_run = load_json(pipeline_run_path)
            if pipeline_run.get("status") == "completed":
                row = green_multiscale_summary_row_from_pipeline_run(
                    planned_run=planned_run,
                    run_root=run_root,
                    pipeline_run=pipeline_run,
                    index=index,
                    total=len(planned_runs),
                    existing=existing_row,
                )
                summary_runs.append(row)
                write_green_multiscale_sweep_summary(summary_path, planned=planned, spec=spec, summary_runs=summary_runs, status="running")
                print(f"skipped completed {index:03d}/{len(planned_runs)}: {run_id}", flush=True)
                continue

        small_params = pipeline_stage_params(planned_run, "cfar_small_ref", "gamma_cfar")
        large_params = pipeline_stage_params(planned_run, "cfar_large_ref", "gamma_cfar")
        component_params = pipeline_stage_params(planned_run, "components", "component_filter")
        pfa = float(small_params.get("pfa", 0.04))
        small_guard = int(small_params.get("guard_px", 2))
        small_radius = int(small_params.get("training_radius_px", 6))
        large_pfa = float(large_params.get("pfa", pfa))
        large_guard = int(large_params.get("guard_px", 2))
        large_radius = int(large_params.get("training_radius_px", 18))
        fusion_mode = str(component_params.get("fusion_mode", identity["fusion_mode"])).strip().lower()
        support = max(1, int(component_params.get("support_min_frames", 15)))
        seed_z = float(component_params.get("seed_z", 1.8))
        projection_blob_z = float(component_params.get("projection_blob_z", 1.5))
        min_area = int(component_params.get("min_area_px", 6))
        max_area = int(component_params.get("max_area_px", 450))
        split_large_components = bool(component_params.get("split_large_components", False))
        split_min_distance_px = int(component_params.get("split_min_distance_px", 6))
        split_area_px = int(component_params.get("split_area_px", 80))
        split_max_peaks = int(component_params.get("split_max_peaks", 40))

        run_root.mkdir(parents=True, exist_ok=True)
        candidates_dir = run_root / "artifacts" / "candidates"
        traces_dir = run_root / "artifacts" / "traces"
        events_dir = run_root / "artifacts" / "events"
        for directory in (candidates_dir, traces_dir, events_dir):
            directory.mkdir(parents=True, exist_ok=True)

        small_path, small_summary = small_masks[(pfa, small_guard, small_radius)]
        large_path, large_summary = large_masks[(large_pfa, large_guard, large_radius)]
        candidate_path = candidates_dir / "cfar_large_ref_candidate_mask.npy"
        if candidate_path.exists():
            final_mask = np.load(candidate_path, mmap_mode="r")
            final_summary = dict(large_summary)
            final_summary.update({"active_fraction": float(np.mean(final_mask)), "combine_mode": fusion_mode, "backend": "cupy_cuda"})
        else:
            final_summary = write_fused_mask_gpu(
                small_path,
                large_path,
                candidate_path,
                fusion_mode=fusion_mode,
                chunk_frames=chunk_frames,
                gpu=gpu,
            )
        final_summary.update(
            {
                "pfa": large_pfa,
                "guard_px": large_guard,
                "training_radius_px": large_radius,
                "small_training_radius_px": small_radius,
                "large_training_radius_px": large_radius,
                "small_mask_active_fraction": small_summary.get("active_fraction"),
                "large_mask_active_fraction": large_summary.get("active_fraction"),
                "previous_mask_step": "cfar_small_ref",
            }
        )
        latest_mask_path = candidates_dir / "candidate_mask.npy"
        hardlink_or_copy_npy(candidate_path, latest_mask_path)
        final_mask = np.load(candidate_path, mmap_mode="r")

        projection_support = projection_blob_mask(projection_score, projection_blob_z=projection_blob_z)
        projection_path = candidates_dir / "green_projection_blob_mask.npy"
        np.save(projection_path, projection_support.astype(np.uint8, copy=False))
        projection_summary = {
            "active_fraction": float(np.mean(projection_support)),
            "projection_blob_z": projection_blob_z,
            "projection_mode": "max_of_p95_and_max_robust_z",
            "shape": list(projection_support.shape),
            "backend": "cupy_cuda",
        }

        support_map = support_map_from_mask_gpu(final_mask, gpu=gpu)
        candidates = union_component_candidates_from_support(
            support_map,
            z_projection,
            projection_score,
            projection_support,
            support_min_frames=support,
            seed_z=seed_z,
            min_area=min_area,
            max_area=max_area,
            ndimage=ndimage,
            split_large_components=split_large_components,
            split_min_distance_px=split_min_distance_px,
            split_area_px=split_area_px,
            split_max_peaks=split_max_peaks,
        )
        source_counts: dict[str, int] = {}
        for candidate in candidates:
            source = str(candidate.get("candidate_source") or "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        candidate_json = candidates_dir / "roi_candidates.json"
        write_json_atomic(candidate_json, {"schema_version": 1, "candidates": candidates})

        traces = extract_traces_gpu(gpu, highpass, candidates)
        events = score_events(traces)
        traces = attach_activity_states(traces, events, sustained_z=sustained_z, tonic_z=tonic_z, peak_window_frames=peak_window_frames)
        trace_json = traces_dir / "roi_traces.json"
        write_json_atomic(trace_json, {"schema_version": 1, "traces": traces})
        event_json = events_dir / "kalman_candidate_events.json"
        write_json_atomic(event_json, events)
        ranked = rank_candidates(candidates, video_shape=highpass.shape, weights={"local_correlation_weight": 0.25, "event_support_weight": 0.35, "artifact_weight": -0.3})
        ranked_json = candidates_dir / "ranked_candidates.json"
        write_json_atomic(ranked_json, ranked)
        coverage = diagnostic_bright_blob_coverage(source_video, candidates, ndimage=ndimage)

        pipeline_run = green_multiscale_pipeline_run_manifest(
            planned_run=planned_run,
            run_root=run_root,
            raw_path=args.source_npy,
            highpass_path=highpass_path,
            smoothed_path=smoothed_path,
            z_path=z_path,
            small_path=small_path,
            small_summary=small_summary,
            large_path=large_path,
            large_summary=large_summary,
            final_path=candidate_path,
            final_summary=final_summary,
            projection_score_path=projection_score_path,
            projection_path=projection_path,
            projection_summary=projection_summary,
            candidate_json=candidate_json,
            candidate_count=len(candidates),
            seed_z=seed_z,
            projection_blob_z=projection_blob_z,
            min_area=min_area,
            max_area=max_area,
            support_min_frames=support,
            source_counts=source_counts,
            trace_json=trace_json,
            event_json=event_json,
            event_count=len(events["events"]),
            ranked_json=ranked_json,
            shape=list(highpass.shape),
            raw_dtype=raw_dtype,
            highpass_sigma=highpass_sigma,
            smooth_sigma=smooth_sigma,
            activity_params={"sustained_z": sustained_z, "tonic_z": tonic_z, "peak_window_frames": peak_window_frames},
            gpu_report=gpu["report"],
        )
        write_json_atomic(pipeline_run_path, pipeline_run)
        summary_runs.append(
            {
                "run_id": run_id,
                "run_root": run_root.name,
                "status": "completed",
                "sweep_index": index - 1,
                "sweep_total": len(planned_runs),
                "sweep_parameters": planned_run.get("sweep", {}).get("parameters", []),
                "artifact_count": len(pipeline_run["artifacts"]),
                "candidate_count": len(candidates),
                "candidate_source_counts": source_counts,
                "event_count": len(events["events"]),
                "active_fraction": final_summary.get("active_fraction"),
                "diagnostic_bright_blob_coverage": coverage,
            }
        )
        write_green_multiscale_sweep_summary(summary_path, planned=planned, spec=spec, summary_runs=summary_runs, status="running")
        print(
            f"completed {index:03d}/{len(planned_runs)}: {run_id} candidates={len(candidates)} events={len(events['events'])} coverage={coverage['coverage']:.3f}",
            flush=True,
        )

    sweep_summary = write_green_multiscale_sweep_summary(summary_path, planned=planned, spec=spec, summary_runs=summary_runs, status="completed")
    write_sweep_brief(
        args.sweep_root / "gamma_cfar_grid_brief.md",
        title=str(planned.get("sweep", {}).get("label") or "GPU green-excess multiscale Gamma CFAR grid"),
        summary_runs=summary_runs,
    )
    print(json.dumps({"sweep_summary": str(summary_path), "runs": len(summary_runs), "recommended_run_id": sweep_summary["recommended_run_id"]}, indent=2))


def diagnostic_bright_blob_coverage(
    source_video,
    candidates: list[Mapping[str, Any]],
    *,
    ndimage: Any,
    frames: tuple[int, ...] = (132, 320, 402),
    match_radius_px: float = 8.0,
    min_area: int = 6,
    max_area: int = 500,
) -> dict[str, Any]:
    import numpy as np

    candidate_centers = [
        (float(candidate.get("x", candidate.get("centroidX", 0.0))), float(candidate.get("y", candidate.get("centroidY", 0.0))))
        for candidate in candidates
    ]
    frame_rows = []
    total_blobs = 0
    matched_blobs = 0
    for frame_number in frames:
        index = int(frame_number) - 1
        if index < 0 or index >= int(source_video.shape[0]):
            continue
        frame = np.maximum(np.asarray(source_video[index], dtype=np.float32), 0.0)
        center = float(np.median(frame))
        mad = float(np.median(np.abs(frame - center)))
        scale = max(1e-6, 1.4826 * mad)
        threshold = max(float(np.percentile(frame, 99.3)), center + 2.0 * scale)
        labels, _ = ndimage.label(frame >= threshold)
        objects = ndimage.find_objects(labels)
        frame_total = 0
        frame_matched = 0
        for label_index, slices in enumerate(objects, start=1):
            if slices is None:
                continue
            component = labels[slices] == label_index
            area = int(np.count_nonzero(component))
            if area < min_area or area > max_area:
                continue
            ys, xs = np.nonzero(component)
            cy = float(np.mean(ys + slices[0].start))
            cx = float(np.mean(xs + slices[1].start))
            frame_total += 1
            if any((cx - rx) ** 2 + (cy - ry) ** 2 <= match_radius_px ** 2 for rx, ry in candidate_centers):
                frame_matched += 1
        total_blobs += frame_total
        matched_blobs += frame_matched
        frame_rows.append({"frame": int(frame_number), "bright_blobs": frame_total, "matched_blobs": frame_matched})
    return {
        "frames": frame_rows,
        "bright_blobs": int(total_blobs),
        "matched_blobs": int(matched_blobs),
        "coverage": round(float(matched_blobs / total_blobs), 6) if total_blobs else 0.0,
        "match_radius_px": match_radius_px,
    }


def select_green_roi_state_default_run(summary_runs: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not summary_runs:
        return None
    eligible = [row for row in summary_runs if int(row.get("candidate_count") or 0) <= 300]
    if not eligible:
        eligible = list(summary_runs)

    def pfa_value(row: Mapping[str, Any]) -> float:
        for item in row.get("sweep_parameters", []) or []:
            if item.get("stage") == "green_single_cfar" and item.get("param") == "pfa":
                try:
                    return float(item.get("value"))
                except (TypeError, ValueError):
                    return 999.0
        return 999.0

    def coverage_value(row: Mapping[str, Any]) -> float:
        payload = row.get("diagnostic_bright_blob_coverage") or {}
        try:
            return float(payload.get("coverage", 0.0))
        except (AttributeError, TypeError, ValueError):
            return 0.0

    return sorted(eligible, key=lambda row: (-coverage_value(row), int(row.get("candidate_count") or 0), pfa_value(row), str(row.get("run_id") or "")))[0]


def write_sweep_brief(path: Path, *, title: str, summary_runs: list[Mapping[str, Any]]) -> None:
    lines = [f"# {title}", "", "| Run | ROI candidates | Events | Active fraction |", "| --- | ---: | ---: | ---: |"]
    for row in summary_runs:
        active = row.get("active_fraction")
        active_text = "n/a" if active is None else f"{float(active):.4f}"
        lines.append(f"| `{row.get('run_id')}` | {int(row.get('candidate_count') or 0)} | {int(row.get('event_count') or 0)} | {active_text} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_shared_preprocessing(
    *,
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    first_run: Path,
    highpass_path: Path,
    smoothed_path: Path,
    z_path: Path,
) -> None:
    """Create shared preprocessing arrays when the fast grid has not been bootstrapped yet."""
    if highpass_path.exists() and smoothed_path.exists() and z_path.exists():
        return

    import numpy as np
    from scipy.ndimage import gaussian_filter, gaussian_filter1d

    first_run.mkdir(parents=True, exist_ok=True)
    highpass_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.source_npy.exists():
        raise SystemExit(f"Source .npy not found: {args.source_npy}")

    source = np.load(args.source_npy, mmap_mode="r").astype(np.float32, copy=False)
    if not highpass_path.exists():
        highpass_params = pipeline_stage_params(spec, "highpass", "temporal_highpass_gaussian")
        sigma_frames = float(highpass_params.get("sigma_frames", 6.0))
        baseline = gaussian_filter1d(source, sigma=sigma_frames, axis=0, mode="nearest") if sigma_frames > 0 else source * 0
        highpass = (source - baseline).astype(np.float32, copy=False)
        np.save(highpass_path, highpass)
        print(f"wrote shared highpass: {highpass_path}", flush=True)

    highpass = np.load(highpass_path, mmap_mode="r").astype(np.float32, copy=False)
    if not smoothed_path.exists():
        smooth_params = pipeline_stage_params(spec, "smooth", "spatial_gaussian")
        sigma_px = float(smooth_params.get("sigma_px", 0.6))
        if sigma_px > 0:
            smoothed = gaussian_filter(highpass, sigma=(0.0, sigma_px, sigma_px), mode="nearest").astype(np.float32, copy=False)
        else:
            smoothed = highpass.astype(np.float32, copy=True)
        np.save(smoothed_path, smoothed)
        print(f"wrote shared smoothed video: {smoothed_path}", flush=True)

    if not z_path.exists():
        score_params = pipeline_stage_params(spec, "score", "robust_positive_local_z")
        epsilon = float(score_params.get("epsilon", 1.0))
        frame_median = np.median(highpass, axis=(1, 2), keepdims=True)
        mad = np.median(np.abs(highpass - frame_median), axis=(1, 2), keepdims=True)
        z_stack = np.maximum((highpass - frame_median) / (1.4826 * mad + epsilon), 0.0).astype(np.float32, copy=False)
        np.save(z_path, z_stack)
        print(f"wrote shared z-stack: {z_path}", flush=True)


def pipeline_stage_params(spec: Mapping[str, Any], step_id: str, stage_id: str) -> dict[str, Any]:
    for step in spec.get("pipeline", []) or []:
        if step.get("id") == step_id:
            return dict(step.get("params") or {})
    for step in spec.get("pipeline", []) or []:
        if step.get("stage_id") == stage_id:
            return dict(step.get("params") or {})
    return {}


def cfar_threshold(pfa: float) -> float:
    import numpy as np

    return float(np.sqrt(max(0.0, -2.0 * np.log(float(pfa)))))


def write_chunked_cfar_mask(
    video,
    path: Path,
    *,
    pfa: float,
    guard_px: int,
    training_radius_px: int,
    chunk_frames: int,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Write a Gamma-CFAR mask without materializing the full 3-D score stack."""
    import numpy as np
    from scipy.ndimage import uniform_filter

    if training_radius_px <= guard_px:
        raise ValueError("training_radius_px must be larger than guard_px.")
    threshold_z = cfar_threshold(pfa)
    chunk_frames = max(1, int(chunk_frames))
    shape = tuple(int(value) for value in video.shape)
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
    outer_area = float((2 * training_radius_px + 1) ** 2)
    guard_area = float((2 * guard_px + 1) ** 2)
    training_area = outer_area - guard_area
    active = 0
    total = int(np.prod(shape))

    for start in range(0, shape[0], chunk_frames):
        stop = min(shape[0], start + chunk_frames)
        block = np.asarray(video[start:stop], dtype=np.float32)
        evidence = np.maximum(block, 0.0).astype(np.float32, copy=False)
        outer_mean = uniform_filter(evidence, size=(1, 2 * training_radius_px + 1, 2 * training_radius_px + 1), mode="nearest")
        outer_sq_mean = uniform_filter(evidence * evidence, size=(1, 2 * training_radius_px + 1, 2 * training_radius_px + 1), mode="nearest")
        guard_mean = uniform_filter(evidence, size=(1, 2 * guard_px + 1, 2 * guard_px + 1), mode="nearest")
        guard_sq_mean = uniform_filter(evidence * evidence, size=(1, 2 * guard_px + 1, 2 * guard_px + 1), mode="nearest")
        local_mean = ((outer_mean * outer_area) - (guard_mean * guard_area)) / training_area
        local_sq_mean = ((outer_sq_mean * outer_area) - (guard_sq_mean * guard_area)) / training_area
        local_var = np.maximum(local_sq_mean - (local_mean * local_mean), 0.0)
        local_std = np.sqrt(local_var + float(epsilon)).astype(np.float32, copy=False)
        score = np.maximum((evidence - local_mean) / (local_std + float(epsilon)), 0.0).astype(np.float32, copy=False)
        mask = score >= threshold_z
        active += int(np.count_nonzero(mask))
        output[start:stop] = mask.astype(np.uint8, copy=False)
        output.flush()
        print(f"  CFAR pfa={pfa} radius={training_radius_px}: frames {start + 1}-{stop}/{shape[0]}", flush=True)

    output.flush()
    return {
        "active_fraction": float(active / total) if total else 0.0,
        "pfa": pfa,
        "guard_px": guard_px,
        "training_radius_px": training_radius_px,
        "threshold_z": threshold_z,
        "combine_mode": "replace",
        "shape": list(shape),
    }


def robust_positive_z_map(frame) -> Any:
    import numpy as np

    values = np.asarray(frame, dtype=np.float32)
    center = median_float(values)
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if scale < 1e-6:
        scale = float(np.std(values))
    if scale < 1e-6:
        scale = 1.0
    return np.maximum((values - center) / scale, 0.0).astype(np.float32, copy=False)


def projection_blob_evidence(video) -> Any:
    import numpy as np

    positive = np.maximum(np.asarray(video, dtype=np.float32), 0.0)
    p95_projection = np.percentile(positive, 95, axis=0).astype(np.float32, copy=False)
    max_projection = np.max(positive, axis=0).astype(np.float32, copy=False)
    return np.maximum(robust_positive_z_map(p95_projection), robust_positive_z_map(max_projection)).astype(np.float32, copy=False)


def projection_blob_mask(projection_score, *, projection_blob_z: float) -> Any:
    import numpy as np

    return np.asarray(projection_score >= float(projection_blob_z), dtype=bool)


def component_candidates(mask, z_projection, *, support_min_frames: int, seed_z: float, min_area: int, max_area: int, ndimage: Any) -> list[dict[str, Any]]:
    import numpy as np

    projection_score = np.zeros_like(np.asarray(z_projection, dtype=np.float32), dtype=np.float32)
    projection_support = np.zeros_like(projection_score, dtype=bool)
    return union_component_candidates(
        mask,
        z_projection,
        projection_score,
        projection_support,
        support_min_frames=support_min_frames,
        seed_z=seed_z,
        min_area=min_area,
        max_area=max_area,
        ndimage=ndimage,
    )


def union_component_candidates(
    mask,
    z_projection,
    projection_score,
    projection_support,
    *,
    support_min_frames: int,
    seed_z: float,
    min_area: int,
    max_area: int,
    ndimage: Any,
) -> list[dict[str, Any]]:
    import numpy as np

    mask_array = np.asarray(mask, dtype=bool)
    support = np.sum(mask_array, axis=0)
    z_map = np.asarray(z_projection, dtype=np.float32)
    projection_map = np.asarray(projection_score, dtype=np.float32)
    projection_footprint = np.asarray(projection_support, dtype=bool)
    cfar_footprint = (support >= float(support_min_frames)) & (z_map >= float(seed_z))
    footprint = cfar_footprint | projection_footprint
    labels, _ = ndimage.label(footprint)
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
        cfar_pixels = int(np.count_nonzero(cfar_footprint[slices] & component))
        projection_pixels = int(np.count_nonzero(projection_footprint[slices] & component))
        if cfar_pixels and projection_pixels:
            source = "union"
        elif projection_pixels:
            source = "projection_blob"
        else:
            source = "cfar_support"
        component_z = z_map[slices][component]
        component_projection = projection_map[slices][component]
        component_support = support[slices][component]
        candidates.append(
            {
                "id": f"roi_{len(candidates) + 1:03d}",
                "x": float(np.mean(abs_xs)),
                "y": float(np.mean(abs_ys)),
                "area_px": area,
                "peak_z": float(np.max(component_z)) if component_z.size else 0.0,
                "projection_z": float(np.max(component_projection)) if component_projection.size else 0.0,
                "support_frames": int(np.max(component_support)) if component_support.size else 0,
                "candidate_source": source,
                "source_pixels": {"cfar": cfar_pixels, "projection": projection_pixels},
                "bbox": [int(np.min(abs_xs)), int(np.min(abs_ys)), int(np.max(abs_xs)), int(np.max(abs_ys))],
            }
        )
    return candidates

def extract_traces(video, candidates: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    import numpy as np

    height, width = video.shape[1:]
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    traces = []
    for candidate in candidates:
        cx = float(candidate.get("x", 0.0))
        cy = float(candidate.get("y", 0.0))
        area = max(1.0, float(candidate.get("area_px", 1.0)))
        inner_radius = max(1.0, math.sqrt(area / math.pi))
        distances = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        roi_mask = distances <= inner_radius
        ring_mask = (distances > inner_radius + 1.0) & (distances <= max(inner_radius + 2.0, 15.0))
        raw_trace = np.asarray(video[:, roi_mask].mean(axis=1), dtype=np.float32)
        background_trace = np.asarray(video[:, ring_mask].mean(axis=1), dtype=np.float32) if np.any(ring_mask) else np.zeros_like(raw_trace)
        corrected = raw_trace - 0.7 * background_trace
        traces.append(
            {
                "roi_id": str(candidate.get("id")),
                "x": cx,
                "y": cy,
                "area_px": area,
                "inner_radius_px": inner_radius,
                "outer_radius_px": 15,
                "neuropil_weight": 0.7,
                "raw_trace": [float(value) for value in raw_trace],
                "background_trace": [float(value) for value in background_trace],
                "corrected_trace": [float(value) for value in corrected],
            }
        )
    return traces


def score_events(traces: list[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    events = []
    counts: dict[str, int] = {}
    for row in traces:
        roi_id = str(row.get("roi_id") or "")
        trace = np.asarray(row.get("corrected_trace") or [], dtype=np.float32)
        if not roi_id or trace.size < 3:
            counts[roi_id] = 0
            continue
        score = kalman_positive_innovation(trace)
        frames = []
        for index in range(1, len(score) - 1):
            if score[index] >= 2.4 and score[index] >= score[index - 1] and score[index] >= score[index + 1]:
                frames.append(index)
                events.append({"roi_id": roi_id, "frame": int(index), "score": float(score[index]), "amplitude": float(trace[index]), "mode": "robust_kalman"})
        counts[roi_id] = len(frames)
    return {"schema_version": 1, "event_threshold_z": 2.4, "mode": "robust_kalman", "events": events, "roi_event_counts": counts}


def robust_trace_scale(trace) -> float:
    import numpy as np

    values = np.asarray(trace, dtype=np.float32)
    if values.size == 0:
        return 1e-6
    center = median_float(values)
    scale = float(1.4826 * median_float(np.abs(values - center)))
    if scale < 1e-6 and values.size > 1:
        diffs = np.diff(values)
        diff_center = median_float(diffs)
        scale = float(1.4826 * median_float(np.abs(diffs - diff_center)))
    if scale < 1e-6:
        scale = float(np.std(values))
    return max(scale, 1e-6)


def trace_activity_state_payload(
    trace,
    peak_events: list[Mapping[str, Any]] | None = None,
    *,
    sustained_z: float = 1.2,
    tonic_z: float = 2.0,
    peak_window_frames: int = 1,
) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(trace, dtype=np.float32)
    if values.size == 0:
        return {
            "activity_intervals": [],
            "activity_summary": {"peak_frame_count": 0, "sustained_frame_count": 0, "sustained_fraction": 0.0, "tonic_score": 0.0},
        }
    scale = robust_trace_scale(values)
    center = float(np.percentile(values, 20))
    relative_z = (values - center) / scale
    tonic_score = float(median_float(np.maximum(values, 0.0)) / scale)
    peak_mask = np.zeros(values.shape, dtype=bool)
    peak_window = max(0, int(peak_window_frames))
    for event in peak_events or []:
        try:
            frame = int(event.get("frame"))
        except (AttributeError, TypeError, ValueError):
            continue
        candidate_indices = {frame, frame - 1}
        for center_index in candidate_indices:
            for index in range(center_index - peak_window, center_index + peak_window + 1):
                if 0 <= index < values.size:
                    peak_mask[index] = True
    sustained_mask = (values > 0.0) & ((relative_z >= float(sustained_z)) | (tonic_score >= float(tonic_z)))
    sustained_mask &= ~peak_mask
    intervals = boolean_mask_intervals(sustained_mask, state="sustained")
    summary = {
        "peak_frame_count": int(np.count_nonzero(peak_mask)),
        "sustained_frame_count": int(np.count_nonzero(sustained_mask)),
        "sustained_fraction": round(float(np.count_nonzero(sustained_mask) / values.size), 6),
        "tonic_score": round(tonic_score, 6),
    }
    return {"activity_intervals": intervals, "activity_summary": summary}


def boolean_mask_intervals(mask, *, state: str) -> list[dict[str, Any]]:
    import numpy as np

    values = np.asarray(mask, dtype=bool)
    intervals: list[dict[str, Any]] = []
    start: int | None = None
    for index, value in enumerate(values, start=1):
        if value and start is None:
            start = index
        elif not value and start is not None:
            intervals.append({"start": start, "end": index - 1, "state": state})
            start = None
    if start is not None:
        intervals.append({"start": start, "end": int(values.size), "state": state})
    return intervals


def attach_activity_states(
    traces: list[dict[str, Any]],
    events_payload: Mapping[str, Any],
    *,
    sustained_z: float = 1.2,
    tonic_z: float = 2.0,
    peak_window_frames: int = 1,
) -> list[dict[str, Any]]:
    grouped = events_by_roi_from_payload(events_payload)
    for row in traces:
        roi_id = str(row.get("roi_id") or "")
        payload = trace_activity_state_payload(
            row.get("corrected_trace") or [],
            grouped.get(roi_id, []),
            sustained_z=sustained_z,
            tonic_z=tonic_z,
            peak_window_frames=peak_window_frames,
        )
        row["activity_intervals"] = payload["activity_intervals"]
        row["activity_summary"] = payload["activity_summary"]
    return traces


def kalman_positive_innovation(trace):
    import numpy as np

    baseline = float(trace[0])
    innovations = np.zeros_like(trace, dtype=np.float32)
    scale_values = []
    for index, value in enumerate(trace):
        innovation = float(value - baseline)
        innovations[index] = max(0.0, innovation)
        scale_values.append(abs(innovation))
        if innovation > 0:
            baseline += 0.008 * innovation
        else:
            baseline += 0.11 * innovation
        baseline += 0.06 * (float(value) - baseline)
    scale = float(1.4826 * median_float(scale_values) + 1e-6)
    return innovations / scale


def fast_pipeline_run_manifest(
    *,
    planned_run: Mapping[str, Any],
    run_root: Path,
    raw_path: Path,
    highpass_path: Path,
    smoothed_path: Path,
    z_path: Path,
    small_path: Path,
    small_summary: Mapping[str, Any],
    final_path: Path,
    final_summary: Mapping[str, Any],
    candidate_json: Path,
    candidate_count: int,
    seed_z: float,
    min_area: int,
    max_area: int,
    support_min_frames: int,
    trace_json: Path,
    event_json: Path,
    event_count: int,
    ranked_json: Path,
    shape: list[int],
    raw_dtype: str,
) -> dict[str, Any]:
    def artifact(artifact_id: str, kind: str, path: Path, producer_stage: str, summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_id": artifact_id,
            "kind": kind,
            "path": rel_to(path, run_root),
            "producer_stage": producer_stage,
            "sha256": "",
            "summary": dict(summary or {}),
        }

    return {
        "schema_version": 1,
        "run_id": planned_run["run_id"],
        "dataset_id": planned_run["dataset_id"],
        "pipeline_spec_id": planned_run["run_id"],
        "status": "completed",
        "parameter_hash": "",
        "environment": {"runner": "tools.prepare_gamma_cfar_workbench_run.fast_grid", "device": "cpu"},
        "artifacts": [
            artifact("raw_video.v1", "raw_video", raw_path, "source_video_import", {"shape": shape, "dtype": raw_dtype}),
            artifact("highpass_video.v1", "highpass_video", highpass_path, "temporal_highpass_gaussian", {"shape": shape, "sigma_frames": 6.0}),
            artifact("smoothed_video.v1", "smoothed_video", smoothed_path, "spatial_gaussian", {"shape": shape, "sigma_px": 0.6}),
            artifact("z_stack.v1", "z_stack", z_path, "robust_positive_local_z", {"shape": shape}),
            artifact("cfar_small_ref_candidate_mask.v1", "candidate_mask", small_path, "gamma_cfar", small_summary),
            artifact("cfar_large_ref_candidate_mask.v1", "candidate_mask", final_path, "gamma_cfar", final_summary),
            artifact("roi_candidates.v1", "roi_candidates", candidate_json, "component_filter", {"count": candidate_count, "seed_z": seed_z, "min_area_px": min_area, "max_area_px": max_area, "support_min_frames": support_min_frames}),
            artifact("roi_traces.v1", "roi_traces", trace_json, "local_background_ring", {"count": candidate_count, "outer_radius_px": 15, "neuropil_weight": 0.7}),
            artifact("kalman_candidate_events.v1", "candidate_events", event_json, "robust_kalman_positive_innovation", {"event_count": event_count, "roi_count": candidate_count, "mode": "robust_kalman"}),
            artifact("ranked_candidates.v1", "ranked_candidates", ranked_json, "heuristic_priority_v1", {"count": candidate_count}),
        ],
    }


def green_single_pipeline_run_manifest(
    *,
    planned_run: Mapping[str, Any],
    run_root: Path,
    raw_path: Path,
    highpass_path: Path,
    smoothed_path: Path,
    z_path: Path,
    final_path: Path,
    final_summary: Mapping[str, Any],
    candidate_json: Path,
    candidate_count: int,
    seed_z: float,
    min_area: int,
    max_area: int,
    support_min_frames: int,
    trace_json: Path,
    event_json: Path,
    event_count: int,
    ranked_json: Path,
    shape: list[int],
    raw_dtype: str,
    highpass_sigma: float,
    smooth_sigma: float,
) -> dict[str, Any]:
    def artifact(artifact_id: str, kind: str, path: Path, producer_stage: str, summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_id": artifact_id,
            "kind": kind,
            "path": rel_to(path, run_root),
            "producer_stage": producer_stage,
            "sha256": "",
            "summary": dict(summary or {}),
        }

    input_summary = {"shape": shape, "dtype": raw_dtype, "channel": "green_excess"}
    return {
        "schema_version": 1,
        "run_id": planned_run["run_id"],
        "dataset_id": planned_run["dataset_id"],
        "pipeline_spec_id": planned_run["run_id"],
        "status": "completed",
        "parameter_hash": "",
        "environment": {"runner": "tools.prepare_gamma_cfar_workbench_run.green_excess_grid", "device": "cpu"},
        "artifacts": [
            artifact("raw_video.v1", "raw_video", raw_path, "source_video_import", input_summary),
            artifact("green_excess_input.v1", "analysis_input", raw_path, "source_video_import", input_summary),
            artifact("highpass_video.v1", "highpass_video", highpass_path, "temporal_highpass_gaussian", {"shape": shape, "sigma_frames": highpass_sigma}),
            artifact("smoothed_video.v1", "smoothed_video", smoothed_path, "spatial_gaussian", {"shape": shape, "sigma_px": smooth_sigma}),
            artifact("z_stack.v1", "z_stack", z_path, "robust_positive_local_z", {"shape": shape}),
            artifact("green_single_cfar_candidate_mask.v1", "candidate_mask", final_path, "gamma_cfar", final_summary),
            artifact("roi_candidates.v1", "roi_candidates", candidate_json, "component_filter", {"count": candidate_count, "seed_z": seed_z, "min_area_px": min_area, "max_area_px": max_area, "support_min_frames": support_min_frames}),
            artifact("roi_traces.v1", "roi_traces", trace_json, "local_background_ring", {"count": candidate_count, "outer_radius_px": 15, "neuropil_weight": 0.7}),
            artifact("kalman_candidate_events.v1", "candidate_events", event_json, "robust_kalman_positive_innovation", {"event_count": event_count, "roi_count": candidate_count, "mode": "robust_kalman"}),
            artifact("ranked_candidates.v1", "ranked_candidates", ranked_json, "heuristic_priority_v1", {"count": candidate_count}),
        ],
    }


def green_roi_state_pipeline_run_manifest(
    *,
    planned_run: Mapping[str, Any],
    run_root: Path,
    raw_path: Path,
    highpass_path: Path,
    smoothed_path: Path,
    z_path: Path,
    final_path: Path,
    final_summary: Mapping[str, Any],
    projection_score_path: Path,
    projection_path: Path,
    projection_summary: Mapping[str, Any],
    candidate_json: Path,
    candidate_count: int,
    seed_z: float,
    projection_blob_z: float,
    min_area: int,
    max_area: int,
    support_min_frames: int,
    source_counts: Mapping[str, int],
    trace_json: Path,
    event_json: Path,
    event_count: int,
    ranked_json: Path,
    shape: list[int],
    raw_dtype: str,
    highpass_sigma: float,
    smooth_sigma: float,
    activity_params: Mapping[str, Any],
) -> dict[str, Any]:
    def artifact(artifact_id: str, kind: str, path: Path, producer_stage: str, summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_id": artifact_id,
            "kind": kind,
            "path": rel_to(path, run_root),
            "producer_stage": producer_stage,
            "sha256": "",
            "summary": dict(summary or {}),
        }

    input_summary = {"shape": shape, "dtype": raw_dtype, "channel": "green_excess"}
    candidate_summary = {
        "count": candidate_count,
        "seed_z": seed_z,
        "projection_blob_z": projection_blob_z,
        "min_area_px": min_area,
        "max_area_px": max_area,
        "support_min_frames": support_min_frames,
        "candidate_source_counts": dict(source_counts),
    }
    return {
        "schema_version": 1,
        "run_id": planned_run["run_id"],
        "dataset_id": planned_run["dataset_id"],
        "pipeline_spec_id": planned_run["run_id"],
        "status": "completed",
        "parameter_hash": "",
        "environment": {"runner": "tools.prepare_gamma_cfar_workbench_run.green_excess_roi_state_grid", "device": "cpu"},
        "artifacts": [
            artifact("raw_video.v1", "raw_video", raw_path, "source_video_import", input_summary),
            artifact("green_excess_input.v1", "analysis_input", raw_path, "source_video_import", input_summary),
            artifact("highpass_video.v1", "highpass_video", highpass_path, "temporal_highpass_gaussian", {"shape": shape, "sigma_frames": highpass_sigma}),
            artifact("smoothed_video.v1", "smoothed_video", smoothed_path, "spatial_gaussian", {"shape": shape, "sigma_px": smooth_sigma}),
            artifact("z_stack.v1", "z_stack", z_path, "robust_positive_local_z", {"shape": shape}),
            artifact("green_single_cfar_candidate_mask.v1", "candidate_mask", final_path, "gamma_cfar", final_summary),
            artifact(
                "green_projection_score.v1",
                "candidate_projection_score",
                projection_score_path,
                "component_filter",
                {
                    "projection_mode": "max_of_p95_and_max_robust_z",
                    "shape": [shape[1], shape[2]] if len(shape) >= 3 else shape,
                },
            ),
            artifact("green_projection_blob_mask.v1", "candidate_projection_mask", projection_path, "component_filter", projection_summary),
            artifact("roi_candidates.v1", "roi_candidates", candidate_json, "component_filter", candidate_summary),
            artifact("roi_traces.v1", "roi_traces", trace_json, "local_background_ring", {"count": candidate_count, "outer_radius_px": 15, "neuropil_weight": 0.7, "activity_states": dict(activity_params)}),
            artifact("kalman_candidate_events.v1", "candidate_events", event_json, "robust_kalman_positive_innovation", {"event_count": event_count, "roi_count": candidate_count, "mode": "robust_kalman"}),
            artifact("ranked_candidates.v1", "ranked_candidates", ranked_json, "heuristic_priority_v1", {"count": candidate_count}),
        ],
    }


def green_multiscale_pipeline_run_manifest(
    *,
    planned_run: Mapping[str, Any],
    run_root: Path,
    raw_path: Path,
    highpass_path: Path,
    smoothed_path: Path,
    z_path: Path,
    small_path: Path,
    small_summary: Mapping[str, Any],
    large_path: Path,
    large_summary: Mapping[str, Any],
    final_path: Path,
    final_summary: Mapping[str, Any],
    projection_score_path: Path,
    projection_path: Path,
    projection_summary: Mapping[str, Any],
    candidate_json: Path,
    candidate_count: int,
    seed_z: float,
    projection_blob_z: float,
    min_area: int,
    max_area: int,
    support_min_frames: int,
    source_counts: Mapping[str, int],
    trace_json: Path,
    event_json: Path,
    event_count: int,
    ranked_json: Path,
    shape: list[int],
    raw_dtype: str,
    highpass_sigma: float,
    smooth_sigma: float,
    activity_params: Mapping[str, Any],
    gpu_report: Mapping[str, Any],
) -> dict[str, Any]:
    def artifact(artifact_id: str, kind: str, path: Path, producer_stage: str, summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_id": artifact_id,
            "kind": kind,
            "path": rel_to(path, run_root),
            "producer_stage": producer_stage,
            "sha256": "",
            "summary": dict(summary or {}),
        }

    input_summary = {"shape": shape, "dtype": raw_dtype, "channel": "green_excess"}
    small_artifact_summary = dict(small_summary)
    small_artifact_summary.setdefault("mask_role", "small_reference")
    large_reference_summary = dict(large_summary)
    large_reference_summary.setdefault("mask_role", "large_reference_unfused")
    final_artifact_summary = dict(final_summary)
    final_artifact_summary.setdefault("mask_role", "fused_small_large_reference")
    final_artifact_summary.setdefault("large_reference_mask", rel_to(large_path, run_root))
    candidate_summary = {
        "count": candidate_count,
        "seed_z": seed_z,
        "projection_blob_z": projection_blob_z,
        "min_area_px": min_area,
        "max_area_px": max_area,
        "support_min_frames": support_min_frames,
        "candidate_source_counts": dict(source_counts),
    }
    return {
        "schema_version": 1,
        "run_id": planned_run["run_id"],
        "dataset_id": planned_run["dataset_id"],
        "pipeline_spec_id": planned_run["run_id"],
        "status": "completed",
        "parameter_hash": "",
        "environment": {
            "runner": "tools.prepare_gamma_cfar_workbench_run.green_excess_multiscale_cfar_grid",
            "device": "cuda",
            "backend": "cupy_cuda",
            "cpu_fallback": False,
            "cuda": dict(gpu_report),
        },
        "artifacts": [
            artifact("raw_video.v1", "raw_video", raw_path, "source_video_import", input_summary),
            artifact("green_excess_input.v1", "analysis_input", raw_path, "source_video_import", input_summary),
            artifact("highpass_video.v1", "highpass_video", highpass_path, "temporal_highpass_gaussian", {"shape": shape, "sigma_frames": highpass_sigma, "backend": "cupy_cuda"}),
            artifact("smoothed_video.v1", "smoothed_video", smoothed_path, "spatial_gaussian", {"shape": shape, "sigma_px": smooth_sigma, "backend": "cupy_cuda"}),
            artifact("z_stack.v1", "z_stack", z_path, "robust_positive_local_z", {"shape": shape, "backend": "cupy_cuda"}),
            artifact("cfar_small_ref_candidate_mask.v1", "candidate_mask", small_path, "gamma_cfar", small_artifact_summary),
            artifact("cfar_large_ref_candidate_mask.v1", "candidate_mask", final_path, "gamma_cfar", final_artifact_summary),
            artifact(
                "green_projection_score.v1",
                "candidate_projection_score",
                projection_score_path,
                "component_filter",
                {
                    "projection_mode": "max_of_p95_and_max_robust_z",
                    "shape": [shape[1], shape[2]] if len(shape) >= 3 else shape,
                    "backend": "cupy_cuda",
                },
            ),
            artifact("green_projection_blob_mask.v1", "candidate_projection_mask", projection_path, "component_filter", projection_summary),
            artifact("roi_candidates.v1", "roi_candidates", candidate_json, "component_filter", candidate_summary),
            artifact("roi_traces.v1", "roi_traces", trace_json, "local_background_ring", {"count": candidate_count, "outer_radius_px": 15, "neuropil_weight": 0.7, "activity_states": dict(activity_params), "backend": "cupy_cuda"}),
            artifact("kalman_candidate_events.v1", "candidate_events", event_json, "robust_kalman_positive_innovation", {"event_count": event_count, "roi_count": candidate_count, "mode": "robust_kalman"}),
            artifact("ranked_candidates.v1", "ranked_candidates", ranked_json, "heuristic_priority_v1", {"count": candidate_count}),
        ],
    }


def attach_run_artifacts(run: dict[str, Any], run_root: Path, app_dir: Path, frame_count: int, *, pixel_size_um: float | None = 0.5) -> None:
    pipeline_run = load_json(run_root / "pipeline_run.json")
    artifacts = list(pipeline_run.get("artifacts") or [])
    run["summary"] = build_run_summary(run, artifacts, run_root, pixel_size_um=pixel_size_um)
    run["artifacts"]["intermediates"] = export_known_intermediates(run, artifacts, run_root, app_dir)
    review_rois = build_review_rois(artifacts, run_root)
    review_path = app_dir / "generated_runs" / safe_name(run["run_id"]) / "review_rois.json"
    write_json_atomic(
        review_path,
        {
            "schema_version": 1,
            "run_id": run["run_id"],
            "frame_count": frame_count,
            "review_rois": review_rois,
        },
    )
    run["artifacts"]["review_rois_file"] = rel_to(review_path, app_dir)
    summary_path = review_path.with_name("review_rois.summary.json")
    shard_dir = review_path.parent / "roi_trace_shards"
    gap_path = review_path.parent / "stencil_gap_report.json"
    stencil_points = []
    annotations_path = app_dir / "annotations.json"
    if annotations_path.exists():
        stencil_points = stencil_points_from_annotations(load_json(annotations_path))
    event_threshold = 2.4
    event_payload = load_artifact_json(run_root, last_artifact(artifacts, "candidate_events"))
    if event_payload.get("event_threshold_z") is not None:
        event_threshold = float(event_payload["event_threshold_z"])
    sidecars = write_review_roi_sidecars(
        review_rois,
        summary_path=summary_path,
        shard_dir=shard_dir,
        run_id=str(run["run_id"]),
        frame_count=frame_count,
        event_threshold_z=event_threshold,
        events_by_roi=events_by_roi_from_payload(event_payload),
        stencil_points=stencil_points,
        gap_report_path=gap_path,
    )
    run["artifacts"]["review_rois_summary_file"] = rel_to(summary_path, app_dir)
    run["artifacts"]["review_trace_shards_dir"] = rel_to(shard_dir, app_dir)
    run["artifacts"]["stencil_gap_report_file"] = rel_to(gap_path, app_dir)
    run["artifacts"]["review_roi_payload_version"] = "summary_shards_v1"
    run["artifacts"]["review_roi_summary"] = {
        "roi_count": sidecars["roi_count"],
        "trace_shard_count": sidecars["trace_shard_count"],
        "gap_count": sidecars["gap_count"],
    }
    run["artifacts"]["review_note"] = "Review overlays are shown on raw video from Gamma CFAR candidate locations."


def build_run_summary(run: Mapping[str, Any], artifacts: list[Mapping[str, Any]], run_root: Path, *, pixel_size_um: float | None = 0.5) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for item in run.get("sweep", {}).get("parameters", []) or []:
        summary[f"{item.get('stage')}.{item.get('param')}"] = item.get("value")
    for artifact in artifacts:
        kind = artifact.get("kind")
        a_summary = artifact.get("summary") or {}
        if kind == "candidate_mask":
            summary["final_active_fraction"] = a_summary.get("active_fraction")
        elif kind == "roi_candidates":
            summary["roi_count"] = a_summary.get("count", 0)
            candidates = load_artifact_json(run_root, artifact).get("candidates", [])
            summary.update(candidate_size_summary(candidates, pixel_size_um=pixel_size_um))
        elif kind == "candidate_events":
            summary["event_count"] = a_summary.get("event_count", 0)
    return summary


def candidate_size_summary(candidates: list[Mapping[str, Any]], *, pixel_size_um: float | None = 0.5) -> dict[str, Any]:
    areas = [float(item.get("area_px") or 0) for item in candidates if float(item.get("area_px") or 0) > 0]
    if not areas:
        return {
            "median_area_px": None,
            "median_equivalent_diameter_px": None,
            "median_equivalent_diameter_um": None,
            "plausible_size_fraction": None if pixel_size_um is None else 0.0,
        }
    areas.sort()
    diameters_px = [math.sqrt(4.0 * area / math.pi) for area in areas]
    diameters_px.sort()
    result: dict[str, Any] = {
        "median_area_px": round(median_value(areas), 6),
        "median_equivalent_diameter_px": round(median_value(diameters_px), 6),
        "median_equivalent_diameter_um": None,
        "plausible_size_fraction": None,
    }
    if pixel_size_um is None:
        return result
    diameters_um = [diameter * pixel_size_um for diameter in diameters_px]
    median_um = median_value(diameters_um)
    plausible = sum(1 for value in diameters_um if 5.0 <= value <= 10.0) / len(diameters_um)
    result["median_equivalent_diameter_um"] = round(median_um, 6)
    result["plausible_size_fraction"] = round(plausible, 6)
    return result


def median_value(values: list[float]) -> float:
    return values[len(values) // 2] if len(values) % 2 else 0.5 * (values[len(values) // 2 - 1] + values[len(values) // 2])


def review_data_pixel_size(review_data_path: Path) -> float | None:
    if not review_data_path.exists():
        return 0.5
    payload = load_json(review_data_path)
    value = (payload.get("dataset") or {}).get("pixel_size_microns")
    if value is None:
        value = (payload.get("parameters") or {}).get("pixelSizeMicrons")
    parsed = parse_positive_float(value)
    return parsed


def export_known_intermediates(run: Mapping[str, Any], artifacts: list[Mapping[str, Any]], run_root: Path, app_dir: Path) -> list[dict[str, Any]]:
    records = []
    by_id = {artifact.get("artifact_id"): artifact for artifact in artifacts}
    for step_id, config in INTERMEDIATE_STEPS.items():
        artifact = by_id.get(config["artifact_id"])
        if not artifact:
            continue
        source = artifact_path(run_root, artifact)
        if not source.exists():
            continue
        out_dir = app_dir / "generated_runs" / "_shared_intermediates" / shared_intermediate_key(run, step_id, artifact)
        frame_count = export_npy_frames(source, out_dir)
        records.append(
            {
                "id": step_id,
                "label": config["label"],
                "description": config["description"],
                "stage_id": artifact.get("producer_stage") or step_id,
                "step_id": step_id,
                "media_type": "frame_sequence",
                "frame_count": frame_count,
                "frame_pattern": rel_to(out_dir / "frame_%03d.png", app_dir),
                "source": str(source),
                "summary": artifact.get("summary") or {},
            }
        )
    return records


def shared_intermediate_key(run: Mapping[str, Any], step_id: str, artifact: Mapping[str, Any]) -> str:
    params = {f"{item.get('stage')}.{item.get('param')}": item.get("value") for item in run.get("sweep", {}).get("parameters", []) or []}
    if step_id in {"highpass", "smooth", "score"}:
        run_id = str(run.get("run_id") or "")
        if "green_excess" in run_id:
            return safe_name(f"green_excess_{step_id}")
        return step_id
    if step_id == "cfar_small_ref":
        summary = dict(artifact.get("summary") or {})
        pfa = params.get("cfar_small_ref.pfa", summary.get("pfa", "x"))
        radius = params.get("cfar_small_ref.training_radius_px", summary.get("training_radius_px", "x"))
        return safe_name(f"{step_id}_pfa_{pfa}_radius_{radius}")
    if step_id == "cfar_large_ref":
        summary = dict(artifact.get("summary") or {})
        pfa = params.get("cfar_small_ref.pfa", summary.get("pfa", "x"))
        small_radius = params.get("cfar_small_ref.training_radius_px", summary.get("small_training_radius_px", "x"))
        large_radius = params.get("cfar_large_ref.training_radius_px", summary.get("large_training_radius_px", summary.get("training_radius_px", "x")))
        fusion = params.get("components.fusion_mode", summary.get("combine_mode", "intersection"))
        return safe_name(f"{step_id}_pfa_{pfa}_small_radius_{small_radius}_large_radius_{large_radius}_{fusion}")
    if step_id == "green_input":
        return safe_name(f"{step_id}_{Path(str(artifact.get('path') or '')).stem}")
    if step_id == "green_single_cfar_mask":
        pfa = params.get("green_single_cfar.pfa", dict(artifact.get("summary") or {}).get("pfa", "x"))
        support = params.get("components.support_min_frames", "x")
        return safe_name(f"{step_id}_pfa_{pfa}_support_{support}")
    if step_id == "green_projection_blob_map":
        blob_z = params.get("components.projection_blob_z", dict(artifact.get("summary") or {}).get("projection_blob_z", "x"))
        return safe_name(f"{step_id}_z_{blob_z}")
    if step_id == "green_projection_score":
        return safe_name("green_excess_projection_score")
    return safe_name(step_id)


def export_npy_frames(source: Path, out_dir: Path) -> int:
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    stack = np.load(source, mmap_mode="r")
    if stack.ndim == 2:
        if frame_path(out_dir, 1).exists():
            return 1
        write_png_gray8(frame_path(out_dir, 1), int(stack.shape[1]), int(stack.shape[0]), normalize_array_frame(stack))
        return 1
    expected_count = int(stack.shape[0])
    if expected_count and frame_path(out_dir, 1).exists() and frame_path(out_dir, expected_count).exists():
        return expected_count
    for index, frame in enumerate(stack, start=1):
        write_png_gray8(frame_path(out_dir, index), int(frame.shape[1]), int(frame.shape[0]), normalize_array_frame(frame))
    return expected_count


def build_review_rois(artifacts: list[Mapping[str, Any]], run_root: Path) -> list[dict[str, Any]]:
    candidates = load_artifact_json(run_root, last_artifact(artifacts, "roi_candidates")).get("candidates", [])
    traces = {
        row.get("roi_id"): row
        for row in load_artifact_json(run_root, last_artifact(artifacts, "roi_traces")).get("traces", [])
    }
    ranked = {
        item.get("candidate_id"): item
        for item in load_artifact_json(run_root, last_artifact(artifacts, "ranked_candidates")).get("ranked_candidates", [])
    }
    events_by_roi = events_by_roi_from_payload(load_artifact_json(run_root, last_artifact(artifacts, "candidate_events")))
    review_rois = []
    for index, candidate in enumerate(candidates, start=1):
        roi_id = str(candidate.get("id") or f"roi_{index:03d}")
        trace = traces.get(roi_id, {})
        rank = ranked.get(roi_id, {})
        area = int(round(float(candidate.get("area_px") or candidate.get("area") or 1)))
        cx = float(candidate.get("x") or candidate.get("centroidX") or 0.0)
        cy = float(candidate.get("y") or candidate.get("centroidY") or 0.0)
        points, bbox = circle_points(cx, cy, area)
        corrected = [float(value) for value in trace.get("corrected_trace") or trace.get("raw_trace") or []]
        activity_intervals = trace.get("activity_intervals") or trace.get("activityIntervals") or []
        activity_summary = trace.get("activity_summary") or trace.get("activitySummary") or {}
        review_rois.append(
            {
                "id": roi_id,
                "roi_kind": "gamma_cfar_candidate",
                "centroidX": round(cx, 3),
                "centroidY": round(cy, 3),
                "area": area,
                "bbox": candidate.get("bbox") or bbox,
                "points": points,
                "peakScore": candidate.get("peak_z") or candidate.get("peakScore") or 0,
                "rawTrace": [float(value) for value in trace.get("raw_trace") or []],
                "backgroundTrace": [float(value) for value in trace.get("background_trace") or []],
                "dffTrace": corrected,
                "events": events_by_roi.get(roi_id, []),
                "activityIntervals": activity_intervals,
                "activitySummary": activity_summary,
                "candidateSource": candidate.get("candidate_source"),
                "projectionScore": candidate.get("projection_z", 0),
                "supportFrames": candidate.get("support_frames", 0),
                "priorityScore": rank.get("priority_score", 0),
                "rank": rank.get("rank", index),
                "artifactScore": rank.get("artifact_score", 0),
                "eventSupport": rank.get("event_support", 0),
                "localCorrelationMean": rank.get("local_correlation", 0),
                "traceSnr": rank.get("trace_snr", 0),
            }
        )
    return review_rois


def circle_points(cx: float, cy: float, area: int) -> tuple[list[list[int]], list[int]]:
    radius = max(1.5, math.sqrt(max(1.0, float(area)) / math.pi))
    x0, x1 = math.floor(cx - radius), math.ceil(cx + radius)
    y0, y1 = math.floor(cy - radius), math.ceil(cy + radius)
    points: list[list[int]] = []
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                points.append([int(x), int(y)])
    return points, [int(x0), int(y0), int(x1), int(y1)]


def load_artifact_json(run_root: Path, artifact: Mapping[str, Any] | None) -> dict[str, Any]:
    if not artifact:
        return {}
    path = artifact_path(run_root, artifact)
    if not path.exists() or path.suffix.lower() != ".json":
        return {}
    return load_json(path)


def artifact_path(run_root: Path, artifact: Mapping[str, Any]) -> Path:
    path = Path(str(artifact.get("path") or ""))
    return path if path.is_absolute() else run_root / path


def last_artifact(artifacts: list[Mapping[str, Any]], kind: str) -> Mapping[str, Any] | None:
    matches = [artifact for artifact in artifacts if artifact.get("kind") == kind]
    return matches[-1] if matches else None


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value).strip("._-")
    return cleaned or "run"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare-dataset")
    prep.add_argument("--input-tif", type=Path, required=True)
    prep.add_argument("--dataset-id", required=True)
    prep.add_argument("--output-npy", type=Path, required=True)
    prep.add_argument("--app-dir", type=Path, required=True)
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--frame-rate-hz", type=float, default=5.0)
    prep.add_argument("--pixel-size-microns", type=float, default=0.5)
    prep.set_defaults(func=prepare_dataset)

    mp4 = sub.add_parser("prepare-mp4-dataset")
    mp4.add_argument("--input-mp4", type=Path, required=True)
    mp4.add_argument("--dataset-id", required=True)
    mp4.add_argument("--output-npy", type=Path, required=True)
    mp4.add_argument("--app-dir", type=Path, required=True)
    mp4.add_argument("--manifest", type=Path, required=True)
    mp4.add_argument("--frame-rate-hz", type=float, default=None, help="Override encoded MP4 frame rate for dashboard playback.")
    mp4.add_argument("--pixel-size-microns", type=float, default=None, help="Physical pixel size if known; omitted for pixel-unit reporting.")
    mp4.add_argument("--channel", choices=["luma", "green_excess"], default="luma", help="MP4 channel conversion used for analysis.")
    mp4.set_defaults(func=prepare_mp4_dataset)

    spec = sub.add_parser("write-sweep-spec")
    spec.add_argument("--dataset-id", required=True)
    spec.add_argument("--run-id", default="gamma_cfar_cascade_grid_v2")
    spec.add_argument("--source-npy", type=Path, required=True)
    spec.add_argument("--out", type=Path, required=True)
    spec.set_defaults(func=write_sweep_spec)

    green_spec = sub.add_parser("write-green-excess-cfar-spec")
    green_spec.add_argument("--dataset-id", required=True)
    green_spec.add_argument("--run-id", default="green_excess_single_cfar_v1")
    green_spec.add_argument("--source-npy", type=Path, required=True)
    green_spec.add_argument("--out", type=Path, required=True)
    green_spec.set_defaults(func=write_green_excess_cfar_spec)

    green_state_spec = sub.add_parser("write-green-excess-roi-state-spec")
    green_state_spec.add_argument("--dataset-id", required=True)
    green_state_spec.add_argument("--run-id", default="green_excess_roi_state_v2")
    green_state_spec.add_argument("--source-npy", type=Path, required=True)
    green_state_spec.add_argument("--out", type=Path, required=True)
    green_state_spec.set_defaults(func=write_green_excess_roi_state_spec)

    green_multiscale_spec = sub.add_parser("write-green-excess-multiscale-cfar-spec")
    green_multiscale_spec.add_argument("--dataset-id", required=True)
    green_multiscale_spec.add_argument("--run-id", default=GREEN_MULTISCALE_SWEEP_ID)
    green_multiscale_spec.add_argument("--source-npy", type=Path, required=True)
    green_multiscale_spec.add_argument("--out", type=Path, required=True)
    green_multiscale_spec.set_defaults(func=write_green_excess_multiscale_cfar_spec)

    attach = sub.add_parser("attach-sweep")
    attach.add_argument("--dataset-id", required=True)
    attach.add_argument("--spec", type=Path, required=True)
    attach.add_argument("--sweep-root", type=Path, required=True)
    attach.add_argument("--app-dir", type=Path, required=True)
    attach.add_argument("--frame-count", type=int, required=True)
    attach.add_argument("--merge-existing", action="store_true", help="Preserve existing experiments and runs while replacing this sweep's run ids.")
    attach.set_defaults(func=attach_sweep)

    fast = sub.add_parser("run-fast-grid")
    fast.add_argument("--spec", type=Path, required=True)
    fast.add_argument("--sweep-root", type=Path, required=True)
    fast.add_argument("--source-npy", type=Path, required=True)
    fast.add_argument("--cfar-chunk-frames", type=int, default=24, help="Frames per CFAR chunk for conservative memory use.")
    fast.set_defaults(func=run_fast_grid)

    green_fast = sub.add_parser("run-green-excess-grid")
    green_fast.add_argument("--spec", type=Path, required=True)
    green_fast.add_argument("--sweep-root", type=Path, required=True)
    green_fast.add_argument("--source-npy", type=Path, required=True)
    green_fast.add_argument("--cfar-chunk-frames", type=int, default=24, help="Frames per CFAR chunk for conservative memory use.")
    green_fast.set_defaults(func=run_green_excess_grid)

    green_state_fast = sub.add_parser("run-green-excess-roi-state-grid")
    green_state_fast.add_argument("--spec", type=Path, required=True)
    green_state_fast.add_argument("--sweep-root", type=Path, required=True)
    green_state_fast.add_argument("--source-npy", type=Path, required=True)
    green_state_fast.add_argument("--cfar-chunk-frames", type=int, default=24, help="Frames per CFAR chunk for conservative memory use.")
    green_state_fast.set_defaults(func=run_green_excess_roi_state_grid)

    green_multiscale_fast = sub.add_parser("run-green-excess-multiscale-cfar-grid")
    green_multiscale_fast.add_argument("--spec", type=Path, required=True)
    green_multiscale_fast.add_argument("--sweep-root", type=Path, required=True)
    green_multiscale_fast.add_argument("--source-npy", type=Path, required=True)
    green_multiscale_fast.add_argument("--gpu-cfar-chunk-frames", type=int, default=32, help="Frames per CUDA CFAR/fusion chunk; retries smaller chunks on GPU OOM.")
    green_multiscale_fast.add_argument("--gpu-preprocess-chunk-frames", type=int, default=32, help="Frames per CUDA preprocessing chunk.")
    green_multiscale_fast.set_defaults(func=run_green_excess_multiscale_cfar_grid)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
