"""Complete, resumable three-section media audit of one sealed two-stencil state.

The CLI consumes a JSON configuration with the keyword arguments accepted by
``run_two_stencil_audit``. The inventory planner uses only the standard library;
array/image libraries are imported only by the explicitly scheduled renderer.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence


GREEN = (70, 220, 125)
ORANGE = (255, 145, 35)
COMPARISON_PANELS = ["Raw matched comparison", "Two-stencil LS matched comparison"]
SOURCE_TIME_SEMANTICS = "source UI one-based inclusive; no per-neuron onset truth"
PANEL_FIT_SEMANTICS = (
    "contain the complete source rectangle with a common aspect-ratio scale; "
    "round fitted width/height to nearest integer pixel; center with black "
    "letterboxing (odd extra padding on right/bottom); bilinear resampling; "
    "transform markers by the actual fitted dimensions using pixel-center coordinates"
)
VIDEO_ENCODING = {
    "revision": "encoding_v2",
    "codec": "libx264rgb",
    "crf": 0,
    "input_pixel_format": "rgb24",
    "encoded_pixel_format": "gbrp",
    "profile": "High 4:4:4 Predictive",
    "validation": "every decoded RGB frame must be byte-identical to its source composite",
    "playback_limit": "High 4:4:4 Predictive may require a software-capable video player; PNG previews are retained",
    "lossy_proxy_is_validation_authority": False,
}
_TRACE_CANVASES: dict[tuple[str, ...], Any] = {}


class _Progress:
    def __init__(self, output: Path, callback: Any = None) -> None:
        self.output = output
        self.callback = callback
        self.last = 0.0

    def emit(self, stage: str, *, force: bool = False, **fields: Any) -> None:
        now = time.monotonic()
        if not force and now - self.last < 30.0:
            return
        self.last = now
        payload = {"stage": stage, **fields}
        _json(self.output / "heartbeat.json", payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        if self.callback is not None:
            self.callback(payload)


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _canonical_digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, (str, Path)):
        path = Path(value)
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream, delimiter="\t" if path.suffix == ".tsv" else ","))
    return [dict(row) for row in value]


def _table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["no_rows"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe(value: str) -> str:
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
        raise ValueError(f"Identifier is not a safe filename: {value}")
    return value


def _int(value: Any, field: str) -> int:
    number = float(value)
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"{field} must be an integer")
    return int(number)


def spatial_review_sites(candidate_rows: Sequence[Mapping[str, Any]], radius: int = 6) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group all emitted rows into review locations without inventing identities.

    Spatial bins accelerate the search but preserve the first accepted site
    within the radius. Grouping is deliberately nontransitive.
    """
    if radius != 6:
        raise ValueError("The frozen audit consolidation radius is six pixels")
    normalized, identifiers = [], set()
    for source in candidate_rows:
        row = dict(source)
        identifier = str(row["proposal_id"])
        if not identifier or identifier in identifiers:
            raise ValueError("Every proposal must have a unique nonempty identifier")
        identifiers.add(identifier)
        for key in ("x_px", "y_px", "source_frame_ui"):
            row[key] = _int(row[key], key)
        row["score"] = float(row["score"])
        if min(row["x_px"], row["y_px"]) < 0 or row["source_frame_ui"] < 1 or not math.isfinite(row["score"]):
            raise ValueError("Invalid proposal coordinate, frame, or score")
        row["proposal_id"] = identifier
        normalized.append(row)
    normalized.sort(key=lambda row: (-row["score"], row["y_px"], row["x_px"], row["source_frame_ui"], row["proposal_id"]))
    sites: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    bins: dict[tuple[int, int], list[int]] = {}
    for row in normalized:
        bx, by = row["x_px"] // radius, row["y_px"] // radius
        neighbors = [index for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                     for index in bins.get((bx + dx, by + dy), [])
                     if (row["x_px"] - sites[index]["x_px"]) ** 2 + (row["y_px"] - sites[index]["y_px"]) ** 2 <= radius ** 2]
        if neighbors:
            index = min(neighbors)
        else:
            index = len(sites)
            sites.append({**row, "model_roi_id": f"review_site_{index + 1:05d}",
                          "representative_proposal_id": row["proposal_id"],
                          "interpretation": "spatial_review_location_not_biological_identity_or_unique_event",
                          "member_source_frames_ui": [], "member_proposal_ids": []})
            bins.setdefault((bx, by), []).append(index)
        site = sites[index]
        site["member_source_frames_ui"].append(row["source_frame_ui"])
        site["member_proposal_ids"].append(row["proposal_id"])
        assignments.append({**row, "model_roi_id": site["model_roi_id"],
                            "distance_to_representative_px": math.hypot(row["x_px"] - site["x_px"], row["y_px"] - site["y_px"]),
                            "is_representative": row["proposal_id"] == site["representative_proposal_id"]})
    for site in sites:
        site["member_source_frames_ui"] = sorted(set(site["member_source_frames_ui"]))
        site["member_proposal_count"] = len(site["member_proposal_ids"])
    return sites, assignments


def context_source_frames(intervals: Sequence[Sequence[int]], available: Sequence[int], context: int = 5) -> list[int]:
    if context < 0:
        raise ValueError("Context cannot be negative")
    return [frame for frame in available if any(int(start) - context <= frame <= int(stop) + context for start, stop in intervals)]


def audit_inventory_plan(candidate_rows: Any, expert_occurrences: Any, source_frames_ui: Sequence[int], *, context_frames: int = 5, fullfield_step: int = 5) -> dict[str, Any]:
    """Inspect exact cardinalities before loading a source array or rendering."""
    candidates, experts = _rows(candidate_rows), _rows(expert_occurrences)
    frames = [_int(x, "source_frame_ui") for x in source_frames_ui]
    if not frames or frames[0] < 1 or any(b != a + 1 for a, b in zip(frames, frames[1:])):
        raise ValueError("The full-duration source frame list must be consecutive")
    if fullfield_step < 1:
        raise ValueError("fullfield_step must be positive")
    source_set = set(frames)
    for row in candidates:
        if _int(row["source_frame_ui"], "source_frame_ui") not in source_set:
            raise ValueError("Candidate frame is outside the audited source interval")
    sites, assignments = spatial_review_sites(candidates)
    unique_experts: dict[str, dict[str, Any]] = {}
    observations = set()
    for row in sorted(experts, key=lambda item: str(item["observation_id"])):
        observation = _safe(str(row["observation_id"]))
        if observation in observations:
            raise ValueError("Duplicate expert occurrence")
        observations.add(observation)
        identifier = _safe(str(row["canonical_roi_id"]))
        x, y = float(row["x_px"]), float(row["y_px"])
        if not all(math.isfinite(v) and v >= 0 for v in (x, y)):
            raise ValueError("Expert coordinates must be finite and nonnegative")
        start, stop = int(row["source_start_ui"]), int(row["source_stop_ui"])
        if start > stop or start not in source_set or stop not in source_set:
            raise ValueError("Expert occurrence is not fully covered by audited source frames")
        if identifier not in unique_experts:
            unique_experts[identifier] = {"canonical_roi_id": identifier, "x_px": x, "y_px": y, "intervals_ui": [],
                                          "trace_anchor_observation_id": observation,
                                          "trace_anchor_rule": "first observation_id actual coordinate; no coordinate averaging",
                                          "coordinate_variants": [], "occurrences": []}
        unique_experts[identifier]["intervals_ui"].append([start, stop])
        unique_experts[identifier]["coordinate_variants"].append({"observation_id": observation, "x_px": x, "y_px": y})
        unique_experts[identifier]["occurrences"].append(dict(row))
    if not experts:
        raise ValueError("This renderer requires a labeled experiment")
    sampled = frames[::fullfield_step]
    if sampled[-1] != frames[-1]:
        sampled.append(frames[-1])
    expert_clips = {key: context_source_frames(row["intervals_ui"], frames, context_frames)
                    for key, row in unique_experts.items()}
    model_clips = {site["model_roi_id"]: context_source_frames([(f, f) for f in site["member_source_frames_ui"]], frames, context_frames)
                   for site in sites}
    return {
        "expert_roi_count": len(unique_experts), "expert_occurrence_count": len(experts),
        "model_roi_count": len(sites), "all_model_proposal_count": len(candidates),
        "source_frame_count": len(frames), "fullfield_source_frames_ui": sampled,
        "fullfield_video_frame_count_each": len(sampled),
        "expected_video_count": 2 + len(unique_experts) + len(sites),
        "expected_trace_figure_count": len(unique_experts) + len(sites) + len(experts),
        "closeup_video_frame_count_total": sum(map(len, expert_clips.values())) + sum(map(len, model_clips.values())),
        "context_frames": context_frames, "fullfield_step": fullfield_step,
        "unique_experts": unique_experts, "model_sites": sites, "model_assignments": assignments,
        "expert_closeup_source_frames_ui": expert_clips, "model_closeup_source_frames_ui": model_clips,
        "no_model_anchor_sampling": True, "all_occurrences_covered": True,
    }


@lru_cache(maxsize=1)
def _load_deps():
    import numpy as np
    from PIL import Image, ImageDraw
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return np, Image, ImageDraw, plt


def panel_image_geometry(source_width: int, source_height: int, panel_width: int, panel_height: int) -> dict[str, Any]:
    """Describe a complete source rectangle fitted into a fixed image panel.

    Integer raster dimensions approximate the common aspect-ratio scale to at
    most half a destination pixel per dimension. No source region is cropped.
    This helper does not import array, plotting, or image libraries.
    """
    sizes = (source_width, source_height, panel_width, panel_height)
    if any(isinstance(size, bool) or int(size) != size or size < 1 for size in sizes):
        raise ValueError("Source and panel dimensions must be positive integers")
    scale = min(panel_width / source_width, panel_height / source_height)
    fitted_width = min(panel_width, max(1, int(math.floor(source_width * scale + .5))))
    fitted_height = min(panel_height, max(1, int(math.floor(source_height * scale + .5))))
    left, top = (panel_width - fitted_width) // 2, (panel_height - fitted_height) // 2
    return {
        "fit_semantics": PANEL_FIT_SEMANTICS,
        "source_size_wh": [source_width, source_height],
        "panel_image_area_size_wh": [panel_width, panel_height],
        "fitted_size_wh": [fitted_width, fitted_height],
        "image_rectangle_xyxy_half_open": [left, top, left + fitted_width, top + fitted_height],
        "padding_left_top_right_bottom": [left, top, panel_width - fitted_width - left,
                                          panel_height - fitted_height - top],
        "nominal_uniform_scale": scale,
        "actual_raster_scale_xy": [fitted_width / source_width, fitted_height / source_height],
        "source_region_cropped_during_fit": False,
        "marker_pixel_center_mapping": "padding + (source_local_coordinate + 0.5) * actual_raster_scale - 0.5",
    }


def panel_marker_position(x: float, y: float, crop: Sequence[int], geometry: Mapping[str, Any]) -> tuple[float, float]:
    """Map a source-pixel center to its local image-panel position."""
    left, top = geometry["image_rectangle_xyxy_half_open"][:2]
    sx, sy = geometry["actual_raster_scale_xy"]
    return left + (x - crop[0] + .5) * sx - .5, top + (y - crop[1] + .5) * sy - .5


def _panel(stage_frames: Mapping[str, Any], limits: Mapping[str, Sequence[float]], *, source_ui: int, source_fps: float, markers: Sequence[Mapping[str, Any]], color: tuple[int, int, int], crop: Sequence[int] | None = None, panel_width: int = 180, panel_height: int = 136):
    np, Image, ImageDraw, _ = _load_deps()
    columns, header = 4, 30
    rows = math.ceil(len(stage_frames) / columns)
    canvas = Image.new("RGB", (columns * panel_width, rows * (panel_height + header)), "black")
    draw = ImageDraw.Draw(canvas)
    source_height, source_width = next(iter(stage_frames.values())).shape
    x0, y0, x1, y1 = crop or (0, 0, source_width, source_height)
    geometry = panel_image_geometry(x1 - x0, y1 - y0, panel_width, panel_height)
    image_left, image_top = geometry["image_rectangle_xyxy_half_open"][:2]
    for index, (name, frame) in enumerate(stage_frames.items()):
        left, top = (index % columns) * panel_width, (index // columns) * (panel_height + header)
        low, high = map(float, limits[name])
        pixels = (np.clip((np.asarray(frame[y0:y1, x0:x1], dtype=np.float32) - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        image = Image.fromarray(pixels, "L").convert("RGB").resize(tuple(geometry["fitted_size_wh"]), Image.Resampling.BILINEAR)
        canvas.paste(image, (left + image_left, top + header + image_top))
        draw.text((left + 3, top + 2), f"{name}  UI {source_ui}  {(source_ui-1)/source_fps:.2f}s", fill="white")
        draw.text((left + 3, top + 15), f"scale [{low:.3g}, {high:.3g}]", fill="white")
        for marker in markers:
            x, y = float(marker["x_px"]), float(marker["y_px"])
            if not x0 <= x < x1 or not y0 <= y < y1:
                continue
            local_x, local_y = panel_marker_position(x, y, (x0, y0, x1, y1), geometry)
            px, py = left + local_x, top + header + local_y
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), outline=color, width=2)
    return np.asarray(canvas)


def _crop(x: float, y: float, width: int, height: int, radius: int = 24) -> tuple[int, int, int, int]:
    cx, cy = int(math.floor(x + 0.5)), int(math.floor(y + 0.5))
    return max(0, cx-radius), max(0, cy-radius), min(width, cx+radius+1), min(height, cy+radius+1)


def identity_crop(location: Mapping[str, Any], width: int, height: int, radius: int = 24) -> tuple[int, int, int, int]:
    """One fixed crop covers every actual coordinate of an expert identity."""
    variants = location.get("coordinate_variants") or [location]
    crops = [_crop(float(row["x_px"]), float(row["y_px"]), width, height, radius) for row in variants]
    return min(c[0] for c in crops), min(c[1] for c in crops), max(c[2] for c in crops), max(c[3] for c in crops)


def _verify_files(records: Sequence[Mapping[str, Any]], *, root: Path | None = None) -> None:
    for row in records:
        path = Path(row["path"]) if root is None else root / str(row["path"])
        if not path.is_file() or _digest(path) != row["sha256"]:
            raise ValueError(f"Frozen source/artifact SHA-256 mismatch: {path}")


def verify_completed_audit(output: Path) -> None:
    """A completed status alone never authorizes reuse of changed artifacts."""
    index = json.loads((output / "artifact_index.json").read_text())
    _verify_files(index["artifacts"], root=output)
    sources = json.loads((output / "source_manifest.json").read_text())
    _verify_files(sources["sources"])
    _verify_files(sources.get("numeric_state_sources", []))


def _ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,profile,pix_fmt,width,height,nb_frames,r_frame_rate,duration", "-of", "json", str(path)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-1000:])
    return json.loads(result.stdout)["streams"][0]


def _encoded_markers(path: Path, index: int, width: int, height: int) -> dict[str, int]:
    np, _, _, _ = _load_deps()
    command = ["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path), "-vf", f"select=eq(n\\,{index})", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    result = subprocess.run(command, capture_output=True)
    if result.returncode or len(result.stdout) != width * height * 3:
        raise RuntimeError("Encoded marker verification failed")
    rgb = np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3).astype(np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return {"green_pixels": int(((green > 170) & (green > red+50) & (green > blue+30)).sum()),
            "orange_pixels": int(((red > 150) & (red > green+35) & (green > blue+20) & (blue < 150)).sum())}


def _source_palette(frame: Any, annotation_section: str) -> dict[str, Any]:
    """Require exact grayscale plus the section's one declared marker color."""
    np, _, _, _ = _load_deps()
    if annotation_section not in {"expert", "model"}:
        raise ValueError("Unknown annotation section")
    pixels = np.asarray(frame)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
        raise ValueError("Video source must be an RGB uint8 composite")
    color = GREEN if annotation_section == "expert" else ORANGE
    gray = (pixels[..., 0] == pixels[..., 1]) & (pixels[..., 1] == pixels[..., 2])
    own = np.all(pixels == color, axis=2)
    if np.any(~gray & ~own):
        raise ValueError(f"Source palette violation in {annotation_section} composite")
    return {"source_palette_pure": True, "own_marker_pixels": int(own.sum()),
            "own_marker_expected": bool(own.any())}


def _validate_lossless_video(path: Path, *, width: int, height: int,
                             source_frames: Sequence[Mapping[str, Any]],
                             annotation_section: str, marker_sample_index: int = 0) -> dict[str, Any]:
    """Decode the entire clip and validate each frame against its source hash."""
    np, _, _, _ = _load_deps()
    command = ["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path),
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    stream_digest = hashlib.sha256()
    decoded_count = 0
    sample_counts = None
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        assert process.stdout is not None and process.stderr is not None
        try:
            for index, source in enumerate(source_frames):
                raw = process.stdout.read(width * height * 3)
                if len(raw) != width * height * 3:
                    raise RuntimeError("Full lossless decode ended before its source-frame inventory")
                if hashlib.sha256(raw).hexdigest() != source["rgb_sha256"]:
                    raise RuntimeError(f"Decoded/source RGB mismatch at source UI {source['source_frame_ui']}")
                pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                palette = _source_palette(pixels, annotation_section)
                if palette["own_marker_pixels"] != source["own_marker_pixels"] or palette["own_marker_expected"] != source["own_marker_expected"]:
                    raise RuntimeError("Decoded marker presence disagrees with its source frame")
                if index == marker_sample_index:
                    sample_counts = {"green_pixels": palette["own_marker_pixels"] if annotation_section == "expert" else 0,
                                     "orange_pixels": palette["own_marker_pixels"] if annotation_section == "model" else 0}
                stream_digest.update(raw)
                decoded_count += 1
            if process.stdout.read(1):
                raise RuntimeError("Decoded video has frames outside its source-frame inventory")
            error = process.stderr.read().decode(errors="replace")
            if process.wait() != 0:
                raise RuntimeError("Full lossless video decode failed: " + error[-1000:])
        except BaseException:
            process.kill()
            process.wait()
            raise
    if sample_counts is None:
        raise ValueError("Marker sample index is outside the decoded inventory")
    return {"decoded_frame_count": decoded_count, "all_frames_rgb_byte_exact": True,
            "all_frames_source_palette_pure": True, "all_frames_decoded_palette_pure": True,
            "all_frames_marker_expectations_pass": True,
            "decoded_rgb_stream_sha256": stream_digest.hexdigest(), "decode_command": command,
            "encoded_sample_marker_counts": sample_counts}


def _write_video(path: Path, frames: Sequence[int], frame_factory: Any, *, fps: float, marker_sample_index: int | None, annotation_section: str, checkpoint: dict[str, Any], checkpoint_path: Path, progress: _Progress | None = None) -> dict[str, Any]:
    np, Image, _, _ = _load_deps()
    key = str(path)
    previous = checkpoint.get(key)
    frame_digest = _canonical_digest(list(frames))
    if previous and previous.get("encoding_revision") == VIDEO_ENCODING["revision"] and previous.get("all_frames_rgb_byte_exact") and previous.get("source_frames_sha256") == frame_digest and path.exists() and _digest(path) == previous["sha256"]:
        thumbnail = path.with_suffix(".png")
        if not thumbnail.is_file() or _digest(thumbnail) != previous.get("thumbnail_sha256"):
            sampled_index = marker_sample_index if marker_sample_index is not None else 0
            Image.fromarray(frame_factory(frames[sampled_index])).save(thumbnail)
            previous["thumbnail"] = str(thumbnail)
            previous["thumbnail_sha256"] = _digest(thumbnail)
            _json(checkpoint_path, checkpoint)
        return previous
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.mp4")
    first = frame_factory(frames[0])
    height, width = first.shape[:2]
    source_records = []
    source_stream_digest = hashlib.sha256()
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264rgb", "-threads", "1", "-preset", "veryfast", "-crf", "0", "-pix_fmt", "rgb24", "-movflags", "+faststart", str(temporary)]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        assert process.stdin is not None and process.stderr is not None
        try:
            for index, source in enumerate(frames):
                frame = first if index == 0 else frame_factory(source)
                if frame.shape != first.shape:
                    raise ValueError("Video panel dimensions changed")
                palette = _source_palette(frame, annotation_section)
                raw = np.ascontiguousarray(frame).tobytes()
                source_records.append({"source_frame_ui": int(source), "rgb_sha256": hashlib.sha256(raw).hexdigest(), **palette})
                source_stream_digest.update(raw)
                process.stdin.write(raw)
                if progress is not None:
                    progress.emit("video_frames", video=str(path), frames_written=index+1, frames_expected=len(frames))
            process.stdin.close()
            error = process.stderr.read().decode(errors="replace")
            if process.wait() != 0:
                raise RuntimeError(error[-1500:])
        except BaseException:
            process.kill()
            process.wait()
            raise
    temporary.replace(path)
    metadata = _ffprobe(path)
    if int(metadata.get("nb_frames", -1)) != len(frames):
        raise RuntimeError("Encoded video frame count disagrees with exact source frame map")
    if metadata.get("codec_name") != "h264" or metadata.get("profile") != VIDEO_ENCODING["profile"] or metadata.get("pix_fmt") != VIDEO_ENCODING["encoded_pixel_format"]:
        raise RuntimeError("Encoded video profile or pixel format disagrees with the lossless RGB contract")
    sampled_index = marker_sample_index if marker_sample_index is not None else 0
    decoded = _validate_lossless_video(path, width=width, height=height,
                                       source_frames=source_records, annotation_section=annotation_section,
                                       marker_sample_index=sampled_index)
    if decoded["decoded_rgb_stream_sha256"] != source_stream_digest.hexdigest():
        raise RuntimeError("Decoded/source RGB stream digest mismatch")
    counts = decoded["encoded_sample_marker_counts"]
    own = "green_pixels" if annotation_section == "expert" else "orange_pixels"
    forbidden = "orange_pixels" if annotation_section == "expert" else "green_pixels"
    if counts[forbidden] or (marker_sample_index is not None and not counts[own]):
        raise RuntimeError(f"Encoded annotation separation failed: {path}: {counts}")
    thumbnail = path.with_suffix(".png")
    Image.fromarray(frame_factory(frames[sampled_index])).save(thumbnail)
    record = {"path": str(path), "sha256": _digest(path), "source_frames_ui": list(frames),
              "source_frames_sha256": frame_digest, "frame_count": len(frames), "fps": fps,
              "annotation_section": annotation_section, "full_decode_pass": True,
              "encoded_marker_counts": counts, "encoded_marker_source_frame_ui": frames[sampled_index],
              "own_marker_expected_on_sample": marker_sample_index is not None,
              "encoded_marker_separation_pass": True, "probe": metadata,
              "encoding_revision": VIDEO_ENCODING["revision"], "encoding": dict(VIDEO_ENCODING),
              "encoding_command": command,
              "source_palette_summary": {"checked_frame_count": len(source_records),
                  "all_frames_gray_or_declared_marker_color": True,
                  "frames_with_own_marker": sum(row["own_marker_expected"] for row in source_records),
                  "own_marker_pixel_count_total": sum(row["own_marker_pixels"] for row in source_records),
                  "minimum_own_marker_pixels_per_frame": min(row["own_marker_pixels"] for row in source_records),
                  "maximum_own_marker_pixels_per_frame": max(row["own_marker_pixels"] for row in source_records)},
              "source_rgb_stream_sha256": source_stream_digest.hexdigest(), **decoded,
              "thumbnail": str(thumbnail), "thumbnail_sha256": _digest(thumbnail)}
    checkpoint[key] = record
    _json(checkpoint_path, checkpoint)
    return record


def _trace_plot(path: Path, stage_traces: Mapping[str, Any], frames: Sequence[int], *, title: str, spans: Sequence[Sequence[int]], color: str, second: Mapping[str, Any] | None = None, assigned: Mapping[str, Any] | None = None, threshold: float | None = None) -> None:
    """Reuse fixed axes/lines; all samples and the original dimensions/DPI remain."""
    _, _, _, plt = _load_deps()
    keys = tuple(stage_traces)
    if keys not in _TRACE_CANVASES:
        fig, axes = plt.subplots(len(keys), 1, figsize=(9, 1.45 * len(keys)), sharex=True, squeeze=False)
        fig.subplots_adjust(left=.095, right=.985, bottom=.058, top=.94, hspace=.19)
        lines = []
        for name, axis in zip(keys, axes[:, 0]):
            primary, = axis.plot([], [], linewidth=.7)
            neighbor, = axis.plot([], [], color="#ff9123", linewidth=.7)
            assignment, = axis.plot([], [], color="#8b7d43", linewidth=.7, linestyle=":")
            cutoff, = axis.plot([], [], color="black", linewidth=.7, linestyle="--")
            lines.append((primary, neighbor, assignment, cutoff))
            axis.set_ylabel(name)
            axis.grid(alpha=.2)
            axis.tick_params(labelsize=8)
        axes[-1, 0].set_xlabel("source frame UI (one-based); all samples, no onset truth", fontsize=9)
        title_artist = fig.suptitle("", fontsize=9, y=.98)
        _TRACE_CANVASES[keys] = (fig, axes[:, 0], lines, title_artist)
    fig, axes, lines, title_artist = _TRACE_CANVASES[keys]
    for (name, values), axis, (primary, neighbor, assignment, cutoff) in zip(stage_traces.items(), axes, lines):
        primary.set_data(frames, values)
        primary.set_color(color)
        primary.set_label("expert" if second is not None else "exact pixel")
        for line, data, label in ((neighbor, second, "nearest site"), (assignment, assigned, "assigned site")):
            line.set_visible(data is not None)
            line.set_data(frames if data is not None else [], data[name] if data is not None else [])
            line.set_label(label)
        show_cutoff = name == "Z" and threshold is not None
        cutoff.set_visible(show_cutoff)
        cutoff.set_data([frames[0], frames[-1]] if show_cutoff else [], [threshold, threshold] if show_cutoff else [])
        cutoff.set_label("frozen threshold")
        for patch in list(axis.patches):
            patch.remove()
        for start, stop in spans:
            axis.axvspan(start, stop, color="#888888", alpha=.1)
        # relim(visible_only=True) prevents a previous card's hidden comparison
        # trace or threshold from determining this card's axes.
        axis.relim(visible_only=True)
        axis.autoscale_view(scalex=False, scaley=True)
        axis.set_xlim(frames[0], frames[-1] if len(frames) > 1 else frames[0]+1)
    previous_legend = axes[0].get_legend()
    if previous_legend is not None:
        previous_legend.remove()
    legend_handles = [line for line in lines[0][:3] if line.get_visible()]
    axes[0].legend(handles=legend_handles, fontsize=7, loc="upper right")
    title_artist.set_text(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.png")
    fig.savefig(temporary, dpi=105)
    temporary.replace(path)


def _artifact_cached(path: Path, checkpoint: Mapping[str, Any]) -> bool:
    previous = checkpoint.get(str(path))
    return bool(previous and path.is_file() and path.stat().st_size == previous["size_bytes"] and _digest(path) == previous["sha256"])


def _checkpoint_artifact(path: Path, checkpoint: dict[str, Any], checkpoint_path: Path) -> None:
    checkpoint[str(path)] = {"sha256": _digest(path), "size_bytes": path.stat().st_size}
    _json(checkpoint_path, checkpoint)


def _correlation(a: Any, b: Any) -> float | None:
    np, _, _, _ = _load_deps()
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    result = float(np.corrcoef(a, b)[0, 1])
    return result if math.isfinite(result) else None


def _lag(a: Any, b: Any, max_lag: int = 5) -> tuple[int | None, float | None]:
    choices = []
    for lag in range(-max_lag, max_lag + 1):
        aa, bb = (a[-lag:], b[:lag]) if lag < 0 else ((a[:-lag], b[lag:]) if lag else (a, b))
        value = _correlation(aa, bb)
        if value is not None:
            choices.append((value, -abs(lag), -lag, lag))
    if not choices:
        return None, None
    best = max(choices)
    return best[-1], best[0]


def _comparison_figure(path: Path, raw: Any, z: Any, experts: Sequence[Mapping[str, Any]], sites: Sequence[Mapping[str, Any]], occurrence_rows: Sequence[Mapping[str, Any]], limits: Mapping[str, Sequence[float]], title: str) -> None:
    _, _, _, plt = _load_deps()
    site_by_id = {row["site_id"]: row for row in sites}
    expert_by_id = {row["observation_id"]: row for row in experts}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    for axis, data, name, key in zip(axes, (raw, z), COMPARISON_PANELS, ("Raw", "Z")):
        axis.imshow(data, cmap="gray", vmin=limits[key][0], vmax=limits[key][1])
        axis.scatter([float(r["x_px"]) for r in experts], [float(r["y_px"]) for r in experts], s=35, facecolors="none", edgecolors="#46dc7d", linewidths=.7)
        axis.scatter([float(r["x_px"]) for r in sites], [float(r["y_px"]) for r in sites], s=25, facecolors="none", edgecolors="#ff9123", marker="s", linewidths=.7)
        for row in occurrence_rows:
            if row["matched_site_id"]:
                site, expert = site_by_id[row["matched_site_id"]], expert_by_id[row["observation_id"]]
                axis.plot([float(expert["x_px"]), site["x_px"]], [float(expert["y_px"]), site["y_px"]], color="#f6e7a1", linewidth=.8)
        axis.set_title(name, fontsize=10)
        axis.set_xlabel("x = column")
        axis.set_ylabel("y = row")
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=115)
    plt.close(fig)


def run_two_stencil_audit(output_root: str | Path, *, stage_paths: Mapping[str, str | Path], signed_stage_keys: Sequence[str], display_limits: Mapping[str, Sequence[float]], source_frames_ui: Sequence[int], source_binding: Mapping[str, Any], operating_point: Mapping[str, Any], fullnumeric_q_paths: Mapping[str, str | Path], candidate_rows: Any = None, candidates_path: str | Path | None = None, expert_occurrences: Any, context_frames: int = 5, fullfield_step: int = 5, fps: float = 50.0, progress_callback: Any = None) -> dict[str, Any]:
    """Render one q1 state completely; never select, fit, or alter a detector.

    All stage arrays must have identical TYX shape, matching source_frames_ui.
    Every consolidated review location receives a closeup and exact-pixel trace.
    q!=1 files are linked as numeric-only states, never represented as audited.
    """
    if candidate_rows is not None and candidates_path is not None:
        raise ValueError("Pass candidate_rows or candidates_path, not both")
    candidate_input = candidate_rows if candidate_rows is not None else candidates_path
    if candidate_input is None:
        raise ValueError("A sealed candidate table is required")
    candidates, experts = _rows(candidate_input), _rows(expert_occurrences)
    for row in candidates:
        if row.get("target_proposals_per_frame") is not None and float(row["target_proposals_per_frame"]) != 1.0:
            raise ValueError("This full-media audit is frozen at q1, not another numeric curve state")
    if operating_point.get("target_proposals_per_frame") is not None and float(operating_point["target_proposals_per_frame"]) != 1.0:
        raise ValueError("The declared audit operating point must be q1")
    plan = audit_inventory_plan(candidates, experts, source_frames_ui, context_frames=context_frames, fullfield_step=fullfield_step)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Source fps must be finite and positive")
    if set(stage_paths) != {"Raw", "X", "A", "M", "sigma", "contrast", "Z"}:
        raise ValueError("Seven exact stages Raw/X/A/M/sigma/contrast/Z are required")
    if set(source_binding.get("stage_sha256", {})) != set(stage_paths):
        raise ValueError("source_binding.stage_sha256 must seal all seven input stages before rendering")
    if set(display_limits) != set(stage_paths) or not set(signed_stage_keys) <= set(stage_paths):
        raise ValueError("Every stage requires a declared fixed display scale")
    for name, bounds in display_limits.items():
        low, high = map(float, bounds)
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError("Display limits must be finite and ordered")
        if name in signed_stage_keys and not math.isclose(low, -high, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("Signed stage scales must be symmetric around zero")
    output = Path(output_root).resolve()
    paths = {name: Path(stage_paths[name]).resolve() for name in ("Raw", "X", "A", "M", "sigma", "contrast", "Z")}
    contract = {"schema_version": 1, "renderer_sha256": _digest(Path(__file__)),
                "evaluation_helper_sha256": _digest(Path(__file__).with_name("two_stencil_evaluation.py")),
                "stage_paths": {k: str(v) for k, v in paths.items()},
                "stage_sequence": list(paths), "signed_stage_keys": list(signed_stage_keys),
                "display_limits": dict(display_limits), "source_frames_ui": list(source_frames_ui),
                "source_binding": dict(source_binding), "operating_point": dict(operating_point),
                "candidate_rows_sha256": _canonical_digest(candidates), "experts_sha256": _canonical_digest(experts),
                "fullnumeric_q_paths": {str(k): str(Path(v).resolve()) for k, v in fullnumeric_q_paths.items()},
                "context_frames": context_frames, "fullfield_step": fullfield_step, "source_fps": fps,
                "fullfield_fps": fps / fullfield_step,
                "video_encoding": dict(VIDEO_ENCODING),
                "panel_fit_semantics": PANEL_FIT_SEMANTICS,
                "fullfield_panel_image_area_size_wh": [180, 136],
                "closeup_panel_image_area_size_wh": [144, 112],
                "fullfield_sampling": "every declared step plus final source frame; omitted source frames remain in closeups and full traces",
                "closeup_sampling": "all source samples in union of every occurrence/proposal frame plus fixed context",
                "no_roi_or_occurrence_sampling": True,
                "model_roi_interpretation": "spatial review location, not unique neuron or biological event",
                "numeric_curve_media_scope": "this q1 state only; other q numeric/candidates only"}
    contract_digest = _canonical_digest(contract)
    if output.exists() and (output / "run_contract.json").exists():
        existing = json.loads((output / "run_contract.json").read_text())
        if _canonical_digest(existing) != contract_digest:
            raise ValueError("Cannot resume an audit under a changed source/configuration contract")
        if (output / "status.json").exists() and json.loads((output / "status.json").read_text()).get("status") == "complete":
            verify_completed_audit(output)
            return json.loads((output / "summary.json").read_text())
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an unrelated output root")
    output.mkdir(parents=True, exist_ok=True)
    progress = _Progress(output, progress_callback)
    _json(output / "run_contract.json", contract)
    _json(output / "status.json", {"status": "rendering", "scientific_audit_complete": False})
    _json(output / "inventory_plan.json", plan)
    progress.emit("audit_started", force=True, expert_rois=plan["expert_roi_count"], model_review_locations=plan["model_roi_count"], occurrences=plan["expert_occurrence_count"], expected_videos=plan["expected_video_count"])
    source_records = []
    expected_stage_hashes = dict(source_binding.get("stage_sha256", {}))
    for name, path in paths.items():
        digest = _digest(path)
        if name in expected_stage_hashes and digest != expected_stage_hashes[name]:
            raise ValueError(f"Stage {name} disagrees with its supplied sealed SHA-256")
        source_records.append({"stage": name, "path": str(path), "sha256": digest, "size_bytes": path.stat().st_size})
        progress.emit("source_hash_verified", force=True, stage_name=name, sha256=digest, source_path=str(path))
    for label, source in (("candidates", candidate_input), ("expert_occurrences", expert_occurrences)):
        expected_hash = source_binding.get(f"{label}_sha256")
        if isinstance(source, (str, Path)):
            path = Path(source).resolve()
            digest = _digest(path)
            if expected_hash and digest != expected_hash:
                raise ValueError(f"{label} differs from its sealed SHA-256")
            source_records.append({"stage": label, "path": str(path), "sha256": digest, "size_bytes": path.stat().st_size})
        elif expected_hash and _canonical_digest(_rows(source)) != expected_hash:
            raise ValueError(f"{label} rows differ from their sealed canonical SHA-256")
    numeric_sources = []
    for q, source in fullnumeric_q_paths.items():
        path = Path(source).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Declared numeric q state is missing: {path}")
        if path.is_dir() and (output.is_relative_to(path) or path.is_relative_to(output)):
            raise ValueError("Numeric source directories must not overlap the audit output")
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in {".csv", ".tsv", ".json"})
        numeric_sources.extend({"q": str(q), "path": str(p), "sha256": _digest(p), "size_bytes": p.stat().st_size} for p in files)
    source_manifest = {"sources": source_records, "numeric_state_sources": numeric_sources, "source_binding": dict(source_binding), "renderer_sha256": _digest(Path(__file__))}
    if (output / "source_manifest.json").exists():
        prior_sources = json.loads((output / "source_manifest.json").read_text())
        for key in ("sources", "numeric_state_sources"):
            if _canonical_digest(prior_sources.get(key, [])) != _canonical_digest(source_manifest[key]):
                raise ValueError("Cannot resume media under changed source-file bytes")
    _json(output / "source_manifest.json", source_manifest)
    np, Image, _, plt = _load_deps()
    arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in paths.items()}
    shape = arrays["Raw"].shape
    if len(shape) != 3 or shape[0] != len(source_frames_ui) or any(a.shape != shape for a in arrays.values()):
        raise ValueError("Every sealed array must align to the same full-duration TYX source grid")
    if any(a.dtype.kind not in "fui" for a in arrays.values()):
        raise ValueError("Stage arrays must be numeric")
    frames = list(map(int, source_frames_ui))
    frame_index = {frame: i for i, frame in enumerate(frames)}
    height, width = shape[1:]
    for row in [*experts, *candidates]:
        if not 0 <= float(row["x_px"]) < width or not 0 <= float(row["y_px"]) < height:
            raise ValueError("Annotation/proposal coordinates are outside the source image")
    from .two_stencil_evaluation import evaluate_occurrence_windows
    intervals = {int(r["burst_id"]): (int(r["source_start_ui"]), int(r["source_stop_ui"])) for r in experts}
    evaluation = evaluate_occurrence_windows(candidates, experts, burst_intervals_ui=intervals)
    expert_root, model_root, comparison = (output / name for name in ("1_Expert_Annotations", "2_Model_Annotations", "3_Comparison"))
    for root in (expert_root, model_root):
        for sub in ("videos/closeups", "figures/traces", "metadata", "exact_pixel_traces"):
            (root / sub).mkdir(parents=True, exist_ok=True)
    (comparison / "trace_comparisons").mkdir(parents=True, exist_ok=True)
    _table(expert_root / "expert_occurrences.csv", experts)
    _table(model_root / "model_occurrences.csv", plan["model_assignments"])
    _table(model_root / "review_sites.csv", plan["model_sites"])
    _json(model_root / "review_sites.json", plan["model_sites"])
    _table(comparison / "expert_model_matches.csv", evaluation["occurrence_rows"])
    _table(comparison / "burst_sites.csv", evaluation["site_rows"])
    _table(comparison / "site_memberships.csv", evaluation["membership_rows"])
    checkpoint_path = output / "render_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    figure_checkpoint_path = output / "figure_checkpoint.json"
    figure_checkpoint = json.loads(figure_checkpoint_path.read_text()) if figure_checkpoint_path.exists() else {}
    videos = []
    candidates_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in candidates:
        candidates_by_frame.setdefault(int(row["source_frame_ui"]), []).append(row)
    def stages(frame: int):
        index = frame_index[frame]
        return {name: a[index] for name, a in arrays.items()}
    def expert_markers(frame: int):
        return [row for row in experts if int(row["source_start_ui"]) <= frame <= int(row["source_stop_ui"])]
    for section, root, color, marker_function in (("expert", expert_root, GREEN, expert_markers), ("model", model_root, ORANGE, lambda f: candidates_by_frame.get(f, []))):
        shown = plan["fullfield_source_frames_ui"]
        marker_index = next((i for i, frame in enumerate(shown) if marker_function(frame)), None)
        videos.append(_write_video(root / "videos" / f"{section}_sequential_full_field.mp4", shown,
            lambda frame, c=color, mf=marker_function: _panel(stages(frame), display_limits, source_ui=frame, source_fps=fps, markers=mf(frame), color=c),
            fps=fps/fullfield_step, marker_sample_index=marker_index, annotation_section=section,
            checkpoint=checkpoint, checkpoint_path=checkpoint_path, progress=progress))
        videos[-1]["panel_image_geometry"] = panel_image_geometry(width, height, 180, 136)
        videos[-1]["source_crop_xyxy_half_open"] = [0, 0, width, height]
        progress.emit(f"{section}_fullfield_complete", force=True, completed_videos=len(videos), expected_videos=plan["expected_video_count"])
    trace_cache: dict[tuple[int, int], dict[str, Any]] = {}
    def exact_traces(x: float, y: float):
        # Explicit nearest pixel convention, avoiding NumPy's ties-to-even ambiguity.
        px, py = min(width-1, int(math.floor(x+.5))), min(height-1, int(math.floor(y+.5)))
        key = (px, py)
        if key not in trace_cache:
            values = {name: np.asarray(a[:, py, px], dtype=np.float32).copy() for name, a in arrays.items()}
            if any(not np.isfinite(v).all() for v in values.values()):
                raise ValueError("Nonfinite exact-pixel trace")
            trace_cache[key] = values
        return trace_cache[key], px, py
    threshold_values = {float(row["threshold_z"]) for row in candidates if row.get("threshold_z") is not None}
    threshold = next(iter(threshold_values)) if len(threshold_values) == 1 else operating_point.get("threshold_z")
    completed_rois = 0
    expected_rois = plan["expert_roi_count"] + plan["model_roi_count"]
    for section, root, locations, color in (("expert", expert_root, list(plan["unique_experts"].values()), GREEN), ("model", model_root, plan["model_sites"], ORANGE)):
        for location in locations:
            identifier = location["canonical_roi_id"] if section == "expert" else location["model_roi_id"]
            shown = plan[f"{section}_closeup_source_frames_ui"][identifier]
            relevant = location["intervals_ui"] if section == "expert" else [(f, f) for f in location["member_source_frames_ui"]]
            x, y = float(location["x_px"]), float(location["y_px"])
            bounds = identity_crop(location, width, height)
            def active(frame: int, intervals=relevant):
                return any(int(a) <= frame <= int(b) for a, b in intervals)
            # In model clips the member proposal coordinates, not a relocated marker, are shown.
            model_member_ids = set(location.get("member_proposal_ids", []))
            def close_markers(frame: int, loc=location, sec=section, member_ids=model_member_ids, af=active):
                if sec == "expert":
                    return [r for r in loc["occurrences"] if int(r["source_start_ui"]) <= frame <= int(r["source_stop_ui"])]
                return [r for r in candidates_by_frame.get(frame, []) if r["proposal_id"] in member_ids]
            marker_index = next(i for i, frame in enumerate(shown) if close_markers(frame))
            videos.append(_write_video(root / "videos/closeups" / f"{identifier}.mp4", shown,
                lambda frame, c=color, mf=close_markers, crop=bounds: _panel(stages(frame), display_limits, source_ui=frame, source_fps=fps, markers=mf(frame), color=c, crop=crop, panel_width=144, panel_height=112),
                fps=fps, marker_sample_index=marker_index, annotation_section=section,
                checkpoint=checkpoint, checkpoint_path=checkpoint_path, progress=progress))
            closeup_geometry = panel_image_geometry(bounds[2] - bounds[0], bounds[3] - bounds[1], 144, 112)
            videos[-1]["panel_image_geometry"] = closeup_geometry
            videos[-1]["source_crop_xyxy_half_open"] = list(bounds)
            traces, px, py = exact_traces(x, y)
            trace_path = root / "exact_pixel_traces" / f"{identifier}.csv"
            if not _artifact_cached(trace_path, figure_checkpoint):
                _table(trace_path, [{"source_frame_ui": frame, **{name: float(values[i]) for name, values in traces.items()}} for i, frame in enumerate(frames)])
                _checkpoint_artifact(trace_path, figure_checkpoint, figure_checkpoint_path)
            figure_path = root / "figures/traces" / f"{identifier}.png"
            if not _artifact_cached(figure_path, figure_checkpoint):
                _trace_plot(figure_path, traces, frames, title=f"{identifier} | exact pixel ({px},{py}); all {len(frames)} source samples", spans=relevant, color="#46dc7d" if section == "expert" else "#ff9123", threshold=threshold)
                _checkpoint_artifact(figure_path, figure_checkpoint, figure_checkpoint_path)
            metadata_path = root / "metadata" / f"{identifier}.json"
            if not _artifact_cached(metadata_path, figure_checkpoint):
                _json(metadata_path, {**location, "exact_trace_pixel_xy": [px, py], "rounding": "floor(coordinate+0.5), clipped at last image pixel", "closeup_source_frames_ui": shown, "crop_xyxy_half_open": bounds, "panel_image_geometry": closeup_geometry, "all_source_samples_in_trace": True, "source_time_semantics": SOURCE_TIME_SEMANTICS})
                _checkpoint_artifact(metadata_path, figure_checkpoint, figure_checkpoint_path)
            completed_rois += 1
            progress.emit(f"{section}_roi", force=completed_rois % 25 == 0 or completed_rois == expected_rois,
                          roi=identifier, completed_rois=completed_rois, expected_rois=expected_rois,
                          completed_videos=len(videos), expected_videos=plan["expected_video_count"])
    sites_by_id = {row["site_id"]: row for row in evaluation["site_rows"]}
    nearest_metrics = []
    for comparison_index, row in enumerate(evaluation["occurrence_rows"], start=1):
        observation = row["observation_id"]
        start, stop = int(row["window_start_frame_ui"]), int(row["window_stop_frame_ui"])
        expert_values, px, py = exact_traces(float(row["x_px"]), float(row["y_px"]))
        near = sites_by_id.get(row["nearest_site_id"])
        assigned = sites_by_id.get(row["matched_site_id"])
        near_values = exact_traces(float(near["x_px"]), float(near["y_px"]))[0] if near else None
        assigned_values = exact_traces(float(assigned["x_px"]), float(assigned["y_px"]))[0] if assigned and (not near or assigned["site_id"] != near["site_id"]) else None
        indices = slice(frame_index[start], frame_index[stop]+1)
        comparison_frames = context_source_frames([(start, stop)], frames, context_frames)
        comparison_slice = slice(frame_index[comparison_frames[0]], frame_index[comparison_frames[-1]]+1)
        correlation = _correlation(expert_values["Z"][indices], near_values["Z"][indices]) if near_values else None
        lag, lag_correlation = _lag(expert_values["Z"][indices], near_values["Z"][indices]) if near_values else (None, None)
        metric = {**row, "nearest_candidate_rank": near["site_rank"] if near else None,
                  "nearest_candidate_score": near["score"] if near else None,
                  "event_gamma_correlation": correlation, "best_lag_frames": lag,
                  "best_lag_correlation": lag_correlation, "nearest_assignment_computed_separately": True,
                  "nearest_equals_assigned": bool(near and assigned and near["site_id"] == assigned["site_id"]),
                  "event_window_semantics": "configured_burst_window_not_per_roi_onset",
                  "comparison_plot_scope": "declared_burst_plus_fixed_context; full_duration_ROI_traces_remain_separate",
                  "comparison_plot_context_frames": context_frames,
                  "comparison_plot_source_frames_ui": comparison_frames,
                  "comparison_plot_start_frame_ui": comparison_frames[0],
                  "comparison_plot_stop_frame_ui": comparison_frames[-1],
                  "correlation_and_lag_window_ui_inclusive": [start, stop],
                  "unmatched_candidate_interpretation": "unknown_not_negative"}
        nearest_metrics.append(metric)
        comparison_path = comparison / "trace_comparisons" / f"{observation}.png"
        if not _artifact_cached(comparison_path, figure_checkpoint):
            _trace_plot(comparison_path, {name: values[comparison_slice] for name, values in expert_values.items()}, comparison_frames,
                        title=f"{observation} | matched={row['matched']} | source UI {comparison_frames[0]}–{comparison_frames[-1]} (burst {start}–{stop}, context ±{context_frames})\nnearest {row['nearest_site_id']} | assigned {row['matched_site_id']}",
                        spans=[(start, stop)], color="#46dc7d",
                        second={name: values[comparison_slice] for name, values in near_values.items()} if near_values is not None else None,
                        assigned={name: values[comparison_slice] for name, values in assigned_values.items()} if assigned_values is not None else None,
                        threshold=threshold)
            _checkpoint_artifact(comparison_path, figure_checkpoint, figure_checkpoint_path)
        comparison_metadata = comparison / "trace_comparisons" / f"{observation}.json"
        if not _artifact_cached(comparison_metadata, figure_checkpoint):
            _json(comparison_metadata, metric)
            _checkpoint_artifact(comparison_metadata, figure_checkpoint, figure_checkpoint_path)
        progress.emit("occurrence_comparison", force=comparison_index % 25 == 0 or comparison_index == len(experts),
                      completed_occurrences=comparison_index, expected_occurrences=len(experts))
    _table(comparison / "nearest_roi_trace_metrics.csv", nearest_metrics)
    # Exact grayscale temporal means are accumulated without loading a whole movie.
    raw_overview = np.zeros((height, width), dtype=np.float64)
    z_overview = np.zeros((height, width), dtype=np.float64)
    all_burst_indices = []
    for burst, (start, stop) in sorted(intervals.items()):
        indices = list(range(frame_index[start], frame_index[stop]+1))
        all_burst_indices.extend(indices)
        raw_map = np.zeros((height, width), dtype=np.float64)
        z_map = np.zeros((height, width), dtype=np.float64)
        for i in indices:
            raw_map += arrays["Raw"][i]
            z_map += arrays["Z"][i]
        raw_overview += raw_map
        z_overview += z_map
        _comparison_figure(comparison / f"burst_{burst}_comparison.png", raw_map/len(indices), z_map/len(indices),
            [r for r in experts if int(r["burst_id"]) == burst], [r for r in evaluation["site_rows"] if r["burst_id"] == burst],
            [r for r in evaluation["occurrence_rows"] if r["burst_id"] == burst], display_limits,
            f"Burst {burst}; grayscale temporal mean, annotations from frozen framewise-site readout")
    _comparison_figure(comparison / "spatial_overview.png", raw_overview/len(all_burst_indices), z_overview/len(all_burst_indices), experts,
                       evaluation["site_rows"], evaluation["occurrence_rows"], display_limits,
                       "Declared-burst temporal means; site markers are review representatives, not a pooled detector")
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    finite = [r for r in nearest_metrics if r["nearest_site_distance_px"] is not None]
    paired = [r for r in finite if r["event_gamma_correlation"] is not None]
    axes[0, 0].scatter([r["nearest_site_distance_px"] for r in paired], [r["event_gamma_correlation"] for r in paired], s=10, color="black")
    axes[0, 0].set(xlabel="Nearest site distance (px)", ylabel="Burst Z correlation")
    axes[0, 1].scatter([r["nearest_site_distance_px"] for r in finite], [r["nearest_candidate_rank"] for r in finite], s=10, color="black")
    axes[0, 1].set(xlabel="Nearest site distance (px)", ylabel="Site rank")
    axes[1, 0].hist([r["nearest_site_distance_px"] for r in finite], bins=12, color="gray")
    axes[1, 0].set(xlabel="Nearest site distance (px)", ylabel="Known occurrences")
    axes[1, 1].hist([len(r["member_source_frames_ui"]) for r in plan["model_sites"]], bins=12, color="gray")
    axes[1, 1].set(xlabel="Member source frames per review site", ylabel="Spatial review sites")
    figure.savefig(comparison / "aggregate_diagnostics.png", dpi=115)
    plt.close(figure)
    _json(output / "video_manifest.json", {"videos": videos})
    _json(output / "display_limits.json", {"limits": dict(display_limits), "signed_stage_keys": list(signed_stage_keys), "across_cell_source": source_binding.get("display_limits_source", "caller_supplied_common_frozen_limits")})
    coverage = {"audited_operating_point": dict(operating_point), "media_state": "q1",
                "q_coverage": {str(q): {"numeric_path": str(Path(path).resolve()), "full_media_for_this_q": float(q) == 1.0} for q, path in fullnumeric_q_paths.items()},
                "expert_roi_expected": plan["expert_roi_count"], "expert_occurrences_expected": plan["expert_occurrence_count"],
                "model_review_roi_expected": plan["model_roi_count"], "all_model_rois_rendered": True,
                "all_expert_occurrences_compared": True, "model_sites_are_biological_identities": False,
                "fullfield_step": fullfield_step, "fullfield_fps": fps/fullfield_step,
                "fullfield_omitted_source_frames": len(frames)-len(plan["fullfield_source_frames_ui"]),
                "closeups_full_rate_all_relevant_source_frames": True,
                "full_duration_exact_pixel_trace_sample_count": len(frames)}
    coverage["occurrence_comparison_plot_extent"] = f"exact declared burst plus {context_frames} source frames of context per side, clipped to source extent"
    _json(output / "coverage_manifest.json", coverage)
    llm = {"schema_version": 1, "entrypoint": "summary.json", "annotation_separation": "strict",
           "comparison_spatial_panels": COMPARISON_PANELS, "model_stage_sequence": list(arrays),
           "coordinate_convention": "x=column,y=row", "frame_convention": SOURCE_TIME_SEMANTICS,
           "coverage": coverage, "marker_semantics": {"green": "expert", "orange": "model proposals", "pale_yellow": "one-to-one assignment links"},
           "primary_tables": ["1_Expert_Annotations/expert_occurrences.csv", "2_Model_Annotations/model_occurrences.csv", "3_Comparison/expert_model_matches.csv", "3_Comparison/nearest_roi_trace_metrics.csv"],
           "limitations": ["sparse positive; unmatched unknown", "q!=1 numerical only", "no biological onset truth", "review sites not neuron or event identities", "fullfield video temporally decimated; all relevant closeup frames and trace samples retained"]}
    _json(output / "llm_context.json", llm)
    for root, description in ((expert_root, "Expert-only markers, every canonical ROI and occurrence."), (model_root, "Model-only markers, every consolidated review location; no biological identities."), (comparison, "Figures/tables only. Nearest-candidate similarity and one-to-one assignment are separate.")):
        (root / "README.md").write_text(description + "\n\nScientific video masters use lossless RGB H.264 (High 4:4:4 Predictive). Every decoded frame is byte-identical to its source composite, with exact source/decoded palette and marker-presence checks. This profile may require a software-capable video player; PNG previews are retained. No unchecked lossy proxy is a validated scientific master.\n")
    (output / "REPORT.md").write_text(
        "# Two-stencil scientific audit\n\n"
        f"Full media audit of the declared q1 state: {plan['expert_roi_count']} expert ROIs, {plan['expert_occurrence_count']} labeled occurrences, "
        f"and every one of {plan['model_roi_count']} consolidated spatial review locations from {len(candidates)} emitted rows. "
        "No location or occurrence was omitted. Source arrays and input tables are hash-bound; scores and detector membership were not changed.\n\n"
        f"Full-field movies display every {fullfield_step}th source frame plus the last at {fps/fullfield_step:g} fps, with explicit source-frame maps. "
        "These videos omit intermediate frames and do not measure event timing accuracy. Closeups include all occurrence/proposal source frames plus fixed context; "
        f"all per-ROI exact-pixel traces contain {len(frames)} samples. Per-occurrence comparison plots show the exact declared burst plus {context_frames} source frames "
        "of context on each side, with their rendered source-frame lists recorded in metadata; correlation and lag still use only the declared burst. "
        "Movie panels preserve the source aspect ratio with black letterboxing and integer raster rounding; the complete source/crop extent is fitted, "
        "and marker coordinates follow the fitted image rectangle. Exact geometry and padding are recorded for every video. "
        "Scientific video masters use lossless RGB H.264 (High 4:4:4 Predictive): every decoded frame is byte-checked against its source composite, "
        "and exact annotation palette/presence is checked on every frame. This profile may require a software-capable video player; PNG previews are retained. "
        "No unchecked lossy proxy is a validated scientific master. "
        "Comparison maps are grayscale temporal means, not the detector's readout.\n\n"
        "Other calibration-curve q values have complete numerical/candidate files but are outside this q1 media audit. "
        "Unmatched proposals are unknown. Review sites are not unique events or neurons. No exact neural-onset truth is available.\n"
    )
    summary = {"status": "validation_pending", "scientific_audit_complete": False, "audited_media_state": "q1",
               "expert_roi_count": plan["expert_roi_count"], "expert_occurrence_count": plan["expert_occurrence_count"],
               "model_roi_count": plan["model_roi_count"], "model_proposal_count": len(candidates),
               "video_count": len(videos), "comparison_trace_count": len(nearest_metrics),
               "all_trace_sample_count": len(frames), "metric_summary": evaluation["summary"],
               "claim_boundary": "q1 scientific artifact fidelity; no biological identity/precision/onset or all-q-media claim"}
    _json(output / "summary.json", summary)
    failures = []
    for section, expected in ((expert_root, plan["expert_roi_count"]), (model_root, plan["model_roi_count"])):
        for folder, extension in (("videos/closeups", "mp4"), ("figures/traces", "png"), ("metadata", "json"), ("exact_pixel_traces", "csv")):
            if len(list((section / folder).glob(f"*.{extension}"))) != expected:
                failures.append(f"{section.name}/{folder} cardinality mismatch")
    if len(videos) != plan["expected_video_count"] or len(nearest_metrics) != len(experts):
        failures.append("video or occurrence cardinality mismatch")
    for video in videos:
        thumbnail = Path(video["thumbnail"])
        if not thumbnail.is_file() or _digest(thumbnail) != video.get("thumbnail_sha256"):
            failures.append(f"video thumbnail missing or changed: {thumbnail}")
    pngs = list(output.rglob("*.png"))
    for path in pngs:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            failures.append(f"PNG decode failure: {path}: {error}")
    if list(comparison.rglob("*.mp4")):
        failures.append("comparison contains a video")
    validation = {"status": "passed" if not failures else "failed", "scientific_audit_complete": not failures,
                  "failures": failures, "inventory_complete": not failures, "all_videos_full_decode": all(r["full_decode_pass"] for r in videos),
                  "encoded_marker_separation_pass": all(r["encoded_marker_separation_pass"] for r in videos),
                  "png_count": len(pngs), "source_hashes_computed": True, "no_new_detector_score": True}
    _json(output / "validation.json", validation)
    def index():
        _json(output / "artifact_index.json", {"artifacts": [{"path": str(p.relative_to(output)), "sha256": _digest(p), "size_bytes": p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_index.json"]})
    index()
    if plan["model_roi_count"]:
        from neurobench.reports.scientific_audit import require_three_section_scientific_audit
        inventory = require_three_section_scientific_audit(output, expected_expert_roi_count=plan["expert_roi_count"], expected_model_roi_count=plan["model_roi_count"], expected_expert_occurrence_count=len(experts), expected_comparison_panels=COMPARISON_PANELS)
        _json(output / "inventory.json", inventory.to_dict())
    else:
        _json(output / "inventory.json", {"complete": not failures, "model_roi_count": 0, "zero_model_roi_case": "no invented ROI; empty-model fullfield and tables retained"})
    if failures:
        _json(output / "status.json", {"status": "failed_validation", "scientific_audit_complete": False})
        raise RuntimeError("; ".join(failures))
    summary.update(status="complete", scientific_audit_complete=True)
    _json(output / "summary.json", summary)
    _json(output / "status.json", {"status": "complete", "scientific_audit_complete": True})
    index()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.preflight:
        plan = audit_inventory_plan(config.get("candidate_rows", config.get("candidates_path")), config["expert_occurrences"], config["source_frames_ui"], context_frames=config.get("context_frames", 5), fullfield_step=config.get("fullfield_step", 5))
        print(json.dumps({k: v for k, v in plan.items() if k not in {"unique_experts", "model_sites", "model_assignments", "expert_closeup_source_frames_ui", "model_closeup_source_frames_ui", "fullfield_source_frames_ui"}}, sort_keys=True))
    else:
        print(json.dumps(run_two_stencil_audit(**config), sort_keys=True))


if __name__ == "__main__":
    main()
