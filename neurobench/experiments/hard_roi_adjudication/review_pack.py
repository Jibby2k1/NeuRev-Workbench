"""Collision-safe detector-blinded review-pack generation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from neurobench.experiments.hierarchical_parzen_ica.missed_neuron_video import (
    _Mp4Writer,
    _probe_video,
    _zoom_box,
)
from neurobench.experiments.learnable_contrast import core as label_core

from .adjudication import ADJUDICATION_FIELDS, draft_rows, write_tsv
from .config import HardRoiAdjudicationConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def preflight(
    config: HardRoiAdjudicationConfig, *, write_artifacts: bool = True
) -> dict[str, Any]:
    panel = config.panel_paths()
    inputs = [
        config.source_video,
        config.original_labels_tsv,
        config.source_ranker_root / "metrics.json",
        config.source_scientific_audit_root / "metrics.json",
        *panel.values(),
    ]
    missing = [str(path) for path in inputs if not path.is_file()]
    shape = dtype = None
    labels: list[dict[str, Any]] = []
    source_valid = labels_valid = targets_valid = panel_valid = False
    if not missing:
        video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
        shape, dtype = list(video.shape), str(video.dtype)
        source_valid = bool(video.ndim == 3 and np.issubdtype(video.dtype, np.integer))
        labels = label_core.load_labels(config.original_labels_tsv)
        identities = {str(row["roi_identity"]) for row in labels}
        targets = set(map(str, config.review["target_roi_ids"]))
        labels_valid = bool(
            labels
            and len({(int(row["burst_id"]), str(row["roi_identity"])) for row in labels})
            == len(labels)
            and all(
                0 <= float(row["x_px"]) < video.shape[2]
                and 0 <= float(row["y_px"]) < video.shape[1]
                and 1 <= int(row["start_frame_ui"]) <= int(row["end_frame_ui"]) <= len(video)
                for row in labels
            )
        )
        targets_valid = targets.issubset(identities)
        expected_frames = 560
        panel_valid = True
        for row in config.frozen_panel:
            path = Path(str(row["path"])).resolve()
            if row["storage"] == "npy_float":
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                panel_valid &= array.shape == (expected_frames, *video.shape[1:])
            else:
                import tifffile
                with tifffile.TiffFile(path) as tif:
                    panel_valid &= tif.series[0].shape == (expected_frames, *video.shape[1:])
    estimated_output_mib = 256.0
    probe = config.output_dir.parent
    while not probe.exists():
        probe = probe.parent
    free_disk_mib = shutil.disk_usage(probe).free / 2**20
    partial = Path(str(config.output_dir) + ".partial")
    gates = {
        "inputs_exist": not missing,
        "source_valid": source_valid,
        "labels_valid": labels_valid,
        "target_roi_ids_exist": targets_valid,
        "frozen_panel_geometry_valid": panel_valid,
        "output_absent": not config.output_dir.exists(),
        "partial_output_absent": not partial.exists(),
        "preflight_separate_from_output": config.preflight_dir != config.output_dir,
        "output_cap_sufficient": estimated_output_mib <= int(config.resources["max_output_mib"]),
        "disk_headroom_sufficient": free_disk_mib >= estimated_output_mib + int(config.resources["min_free_disk_mib"]),
    }
    payload = {
        "schema_version": 1,
        "kind": "spon_ca_burst_hard_roi_adjudication_preflight",
        "experiment_id": config.experiment_id,
        "ready": all(gates.values()),
        "gates": gates,
        "source_shape": shape,
        "source_dtype": dtype,
        "original_label_rows": len(labels),
        "original_roi_identities": len({row["roi_identity"] for row in labels}),
        "target_roi_ids": list(config.review["target_roi_ids"]),
        "frozen_panel_ids": sorted(panel),
        "resources": {**config.resources, "estimated_output_mib": estimated_output_mib, "free_disk_mib": free_disk_mib},
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in inputs if path.is_file()
        ],
        "scientific_contract": (
            "Original labels remain immutable; review clips contain no detector outcome; "
            "provisional expert notes are not final adjudication; unmatched candidates "
            "remain unknown rather than negative."
        ),
    }
    if write_artifacts:
        if config.preflight_dir.exists():
            raise FileExistsError(f"preflight directory already exists: {config.preflight_dir}")
        config.preflight_dir.mkdir(parents=True, exist_ok=False)
        _atomic_json(config.preflight_dir / "preflight.json", payload)
        _atomic_json(config.preflight_dir / "config.resolved.json", config.to_dict())
        if source_valid and labels_valid:
            label_core._write_overlay(
                np.load(config.source_video, mmap_mode="r", allow_pickle=False),
                labels,
                config.preflight_dir / "label_projection_overlay.png",
            )
    if not payload["ready"]:
        raise RuntimeError(f"hard-ROI adjudication preflight failed: {payload}")
    return payload


def _matching_preflight(config: HardRoiAdjudicationConfig) -> dict[str, Any]:
    preflight_path = config.preflight_dir / "preflight.json"
    resolved_path = config.preflight_dir / "config.resolved.json"
    if not preflight_path.is_file() or not resolved_path.is_file():
        raise RuntimeError("review-pack creation requires a reviewed preflight")
    audit = json.loads(preflight_path.read_text(encoding="utf-8"))
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not audit.get("ready") or resolved != config.to_dict():
        raise RuntimeError("review-pack configuration differs from preflight")
    if config.output_dir.exists() or Path(str(config.output_dir) + ".partial").exists():
        raise FileExistsError("completed or partial adjudication output already exists")
    recorded = {item["path"]: item["sha256"] for item in audit.get("inputs", [])}
    if recorded.get(str(config.original_labels_tsv)) != _sha256(config.original_labels_tsv):
        raise RuntimeError("original label file changed after preflight")
    return audit


def _color_lut() -> np.ndarray:
    anchors = np.asarray(
        [[0, 0, 0], [45, 5, 90], [150, 20, 90], [235, 85, 35], [255, 245, 120]],
        dtype=np.float32,
    )
    axis = np.linspace(0, len(anchors) - 1, 256)
    low = np.floor(axis).astype(int)
    high = np.minimum(low + 1, len(anchors) - 1)
    weight = (axis - low)[:, None]
    return np.rint(anchors[low] * (1 - weight) + anchors[high] * weight).astype(np.uint8)


_LUT = _color_lut()


def _normalize(frame: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((np.asarray(frame, dtype=np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


def _render_review_frame(
    frame: np.ndarray,
    baseline: np.ndarray,
    *,
    frame_ui: int,
    crop: tuple[int, int, int, int],
    coordinates: Mapping[str, tuple[int, int]],
    display_lo: float,
    display_hi: float,
    difference_hi: float,
    box_half_size_px: int,
) -> np.ndarray:
    x0, y0, x1, y1 = crop
    raw = np.asarray(frame[y0:y1, x0:x1], dtype=np.float32)
    base = np.asarray(baseline[y0:y1, x0:x1], dtype=np.float32)
    unit = _normalize(raw, display_lo, display_hi)
    gray = np.repeat(np.rint(unit[..., None] * 255), 3, axis=2).astype(np.uint8)
    pseudo = _LUT[np.rint(unit * 255).astype(np.uint8)]
    positive = np.clip((raw - base) / max(difference_hi, 1e-6), 0, 1)
    difference = _LUT[np.rint(positive * 255).astype(np.uint8)]
    gap, header = 4, 58
    panel_h, panel_w = gray.shape[:2]
    canvas = Image.new("RGB", (panel_w * 3 + gap * 2, panel_h + header), "black")
    for index, panel in enumerate((gray, pseudo, difference)):
        canvas.paste(Image.fromarray(panel), (index * (panel_w + gap), header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((6, 5), f"Detector-blinded hard-ROI review | UI frame {frame_ui}", fill="white", font=font)
    draw.text((6, 23), "RAW (fixed scale)                PSEUDO-COLOR                 POSITIVE CHANGE", fill=(205, 205, 205), font=font)
    draw.text((6, 40), "Neutral boxes identify review targets; no detector result is shown.", fill=(150, 220, 230), font=font)
    for panel_index in range(3):
        offset = panel_index * (panel_w + gap)
        for roi, (x, y) in coordinates.items():
            px, py = x - x0 + offset, y - y0 + header
            half, color = int(box_half_size_px), (80, 220, 235)
            draw.rectangle((px-half, py-half, px+half, py+half), outline=color, width=2)
            draw.text((px-half, py-half-10), roi.replace("roi_", ""), fill=color, font=font)
    width, height = canvas.size
    if width % 2 or height % 2:
        padded = Image.new("RGB", (width + width % 2, height + height % 2), "black")
        padded.paste(canvas, (0, 0))
        canvas = padded
    return np.asarray(canvas)


def _projection_overlay(
    frames: np.ndarray,
    crop: tuple[int, int, int, int],
    coordinates: Mapping[str, tuple[int, int]],
    path: Path,
) -> None:
    x0, y0, x1, y1 = crop
    projection = np.percentile(np.asarray(frames[:, y0:y1, x0:x1]), 95, axis=0)
    lo, hi = np.percentile(projection, [1, 99.8])
    unit = _normalize(projection, float(lo), float(hi))
    rgb = np.repeat(np.rint(unit[..., None] * 255), 3, axis=2).astype(np.uint8)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for roi, (x, y) in coordinates.items():
        px, py = x - x0, y - y0
        draw.ellipse((px-5, py-5, px+5, py+5), outline=(80, 220, 235), width=2)
        draw.text((px+6, py-6), roi, fill=(80, 220, 235), font=font)
    image.save(path)


def _write_review_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "burst_id", "clip_start_frame_ui", "clip_end_frame_ui", "target_roi_ids",
        "video_path", "projection_overlay_path", "detector_outcomes_visible",
    )
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def create_review_pack(config: HardRoiAdjudicationConfig) -> dict[str, Any]:
    audit = _matching_preflight(config)
    source = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    labels = label_core.load_labels(config.original_labels_tsv)
    targets = set(map(str, config.review["target_roi_ids"]))
    partial = Path(str(config.output_dir) + ".partial")
    clips_dir, overlays_dir = partial / "clips", partial / "overlays"
    clips_dir.mkdir(parents=True, exist_ok=False)
    overlays_dir.mkdir(parents=True, exist_ok=False)
    review_rows: list[dict[str, Any]] = []
    videos: dict[str, Any] = {}
    for burst in sorted({int(row["burst_id"]) for row in labels}):
        burst_rows = [row for row in labels if int(row["burst_id"]) == burst and str(row["roi_identity"]) in targets]
        if not burst_rows:
            continue
        start_ui = max(1, min(int(row["start_frame_ui"]) for row in burst_rows) - int(config.review["pad_before_frames"]))
        end_ui = min(len(source), max(int(row["end_frame_ui"]) for row in burst_rows) + int(config.review["pad_after_frames"]))
        start_zero, stop_zero = start_ui - 1, end_ui
        coordinates = {
            str(row["roi_identity"]): (int(round(float(row["x_px"]))), int(round(float(row["y_px"]))))
            for row in burst_rows
        }
        crop = _zoom_box(coordinates, (int(source.shape[1]), int(source.shape[2])), int(config.review["crop_padding_px"]))
        raw_clip = np.asarray(source[start_zero:stop_zero])
        original_start = min(int(row["start_frame_ui"]) for row in burst_rows)
        baseline_stop = max(start_zero + 1, original_start - 1)
        baseline = np.median(source[start_zero:baseline_stop], axis=0).astype(np.float32)
        sample = raw_clip[:, crop[1]:crop[3], crop[0]:crop[2]]
        display_lo, display_hi = np.percentile(sample, [float(config.review["display_lower_percentile"]), float(config.review["display_upper_percentile"])])
        difference_hi = max(float(np.percentile(np.maximum(sample - baseline[crop[1]:crop[3], crop[0]:crop[2]], 0), 99.5)), 1.0)
        video_path = clips_dir / f"burst_{burst:02d}_detector_blinded.mp4"
        first = _render_review_frame(raw_clip[0], baseline, frame_ui=start_ui, crop=crop, coordinates=coordinates, display_lo=float(display_lo), display_hi=float(display_hi), difference_hi=difference_hi, box_half_size_px=int(config.review["box_half_size_px"]))
        writer = _Mp4Writer(video_path, first.shape[:2], float(config.review["playback_fps"]))
        try:
            writer.write(first)
            for offset, frame in enumerate(raw_clip[1:], start=1):
                writer.write(_render_review_frame(frame, baseline, frame_ui=start_ui + offset, crop=crop, coordinates=coordinates, display_lo=float(display_lo), display_hi=float(display_hi), difference_hi=difference_hi, box_half_size_px=int(config.review["box_half_size_px"])))
            writer.close()
        except Exception:
            if writer.process.poll() is None:
                writer.abort()
            raise
        overlay_path = overlays_dir / f"burst_{burst:02d}_projection.png"
        _projection_overlay(raw_clip, crop, coordinates, overlay_path)
        probe = _probe_video(video_path)
        if probe["frame_count"] != stop_zero - start_zero:
            raise RuntimeError(f"burst {burst} clip frame count mismatch")
        videos[str(burst)] = {
            "path": str(video_path.relative_to(partial)), "probe": probe,
            "source_frames_ui_inclusive": [start_ui, end_ui], "crop_xyxy": list(crop),
            "target_roi_ids": sorted(coordinates),
            "display_limits_raw": [float(display_lo), float(display_hi)],
            "positive_difference_white": difference_hi,
        }
        review_rows.append({
            "burst_id": burst, "clip_start_frame_ui": start_ui, "clip_end_frame_ui": end_ui,
            "target_roi_ids": ",".join(sorted(coordinates)),
            "video_path": str(video_path.relative_to(partial)),
            "projection_overlay_path": str(overlay_path.relative_to(partial)),
            "detector_outcomes_visible": "false",
        })
    adjudication_rows = draft_rows(labels, targets)
    write_tsv(partial / "adjudication_draft.tsv", adjudication_rows)
    _write_review_manifest(partial / "review_manifest.tsv", review_rows)
    _atomic_text(
        partial / "README.md",
        "# Hard-ROI adjudication review pack\n\n"
        "The clips show raw intensity, fixed pseudo-color, and positive change with neutral target boxes. They do not show detector outcomes. UI frame numbers are one-based and inclusive.\n\n"
        "Edit a copy of `adjudication_draft.tsv`; never overwrite the original label TSV. Provisional expert-note rows are not final until `review_status=adjudicated`, reviewer identity and review time are filled, and event onset/peak/end are supplied where observation-specific timing is claimed. ROI 010/015 must be confirmed as one anatomical identity before using the canonical merge in primary metrics. Unmatched candidates remain unknown rather than negative.\n",
    )
    payload = {
        "schema_version": 1, "status": "review_pack_ready",
        "experiment_id": config.experiment_id, "source_video": str(config.source_video),
        "original_labels_tsv": str(config.original_labels_tsv),
        "original_labels_sha256": _sha256(config.original_labels_tsv),
        "preflight": audit, "detector_blinded": True,
        "target_roi_ids": sorted(targets), "adjudication_columns": list(ADJUDICATION_FIELDS),
        "observation_rows": len(adjudication_rows),
        "provisional_rows": sum(row["review_status"] == "provisional_expert_note" for row in adjudication_rows),
        "pending_target_rows": sum(row["original_roi_id"] in targets and row["review_status"] == "pending" for row in adjudication_rows),
        "videos": videos,
        "scientific_contract": "This pack supports label adjudication only. It does not tune a model, establish precision, or convert unmatched candidates into negatives.",
    }
    _atomic_json(partial / "manifest.json", payload)
    partial.replace(config.output_dir)
    return payload
