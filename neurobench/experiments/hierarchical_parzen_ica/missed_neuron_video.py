"""Raw-video overlays for burst observations missed by a ranked detector."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from neurobench.experiments.learnable_contrast import core as label_core

from .innovation_ranker_config import InnovationRankerConfig


COLORS = {
    "inactive": (50, 175, 255),
    "active_missed": (255, 55, 55),
    "active_recovered": (70, 245, 105),
}


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_audit(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty per-neuron audit: {path}")
    return rows


def missed_identity_records(
    audit_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    *,
    recovery_field: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Join exact burst recovery with label intervals for missed identities."""
    if not audit_rows or recovery_field not in audit_rows[0]:
        raise ValueError(f"Recovery field is unavailable: {recovery_field}")
    audit_by_key = {
        (int(row["burst_id"]), str(row["roi_identity"])): row
        for row in audit_rows
    }
    missed = sorted(
        {
            str(row["roi_identity"])
            for row in audit_rows
            if not _truth(row[recovery_field])
        }
    )
    if not missed:
        raise ValueError("The selected recovery field has no missed identities")
    records = []
    for label in labels:
        identity = str(label["roi_identity"])
        if identity not in missed:
            continue
        key = (int(label["burst_id"]), identity)
        audit = audit_by_key.get(key)
        if audit is None:
            raise ValueError(f"Audit row is missing for burst/ROI {key}")
        records.append(
            {
                "burst_id": key[0],
                "roi_identity": identity,
                "x_px": float(label["x_px"]),
                "y_px": float(label["y_px"]),
                "start_frame_ui": int(label["start_frame_zero"]) + 1,
                "end_frame_ui": int(label["stop_frame_zero_exclusive"]),
                "start_frame_zero": int(label["start_frame_zero"]),
                "stop_frame_zero_exclusive": int(
                    label["stop_frame_zero_exclusive"]
                ),
                "recovered": _truth(audit[recovery_field]),
                "oracle_union_recoverable": _truth(
                    audit["oracle_union_recoverable"]
                ),
            }
        )
    return missed, records


def frame_identity_status(
    records: Sequence[Mapping[str, Any]],
    identity: str,
    frame_zero: int,
) -> tuple[str, int | None]:
    for row in records:
        if (
            row["roi_identity"] == identity
            and int(row["start_frame_zero"]) <= int(frame_zero)
            < int(row["stop_frame_zero_exclusive"])
        ):
            return (
                "active_recovered" if bool(row["recovered"])
                else "active_missed",
                int(row["burst_id"]),
            )
    return "inactive", None


class _Mp4Writer:
    def __init__(self, path: Path, shape: tuple[int, int], fps: float) -> None:
        height, width = shape
        self.path = path
        self.temporary = path.with_name(path.stem + ".partial.mp4")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(float(fps)),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-threads",
            "4",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.temporary),
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin was not created")
        self.shape = (height, width, 3)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != self.shape or frame.dtype != np.uint8:
            raise ValueError(
                f"Rendered frame {frame.shape}/{frame.dtype} != {self.shape}/uint8"
            )
        assert self.process.stdin is not None
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        stderr = (
            self.process.stderr.read().decode("utf-8", "replace")
            if self.process.stderr is not None
            else ""
        )
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed ({return_code}): {stderr[-2000:]}")
        self.temporary.replace(self.path)

    def abort(self) -> None:
        self.process.kill()
        self.process.wait()


def _identity_coordinates(
    identities: Sequence[str], records: Sequence[Mapping[str, Any]]
) -> dict[str, tuple[int, int]]:
    coordinates = {}
    for identity in identities:
        rows = [row for row in records if row["roi_identity"] == identity]
        coordinates[identity] = (
            int(round(float(np.median([row["x_px"] for row in rows])))),
            int(round(float(np.median([row["y_px"] for row in rows])))),
        )
    return coordinates


def _zoom_box(
    coordinates: Mapping[str, tuple[int, int]],
    shape: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    height, width = shape
    xs = [value[0] for value in coordinates.values()]
    ys = [value[1] for value in coordinates.values()]
    x0, x1 = max(0, min(xs) - padding), min(width, max(xs) + padding + 1)
    y0, y1 = max(0, min(ys) - padding), min(height, max(ys) + padding + 1)
    if (x1 - x0) % 2:
        if x1 < width:
            x1 += 1
        elif x0 > 0:
            x0 -= 1
    if (y1 - y0) % 2:
        if y1 < height:
            y1 += 1
        elif y0 > 0:
            y0 -= 1
    return x0, y0, x1, y1


def _render(
    raw: np.ndarray,
    *,
    frame_zero: int,
    identities: Sequence[str],
    coordinates: Mapping[str, tuple[int, int]],
    records: Sequence[Mapping[str, Any]],
    display_lo: float,
    display_hi: float,
    crop: tuple[int, int, int, int],
    box_half_size_px: int,
    title: str,
    magnification: int = 1,
    minimum_canvas_width: int = 0,
) -> np.ndarray:
    x0, y0, x1, y1 = crop
    gray = np.clip(
        (np.asarray(raw[y0:y1, x0:x1], dtype=np.float32) - display_lo)
        / max(display_hi - display_lo, 1e-6),
        0,
        1,
    )
    rgb = np.repeat(np.rint(gray[..., None] * 255), 3, axis=2).astype(np.uint8)
    if magnification < 1 or magnification > 4:
        raise ValueError("magnification must be in [1, 4]")
    if magnification > 1:
        rgb = np.asarray(
            Image.fromarray(rgb).resize(
                (rgb.shape[1] * magnification, rgb.shape[0] * magnification),
                resample=Image.Resampling.BILINEAR,
            )
        )
    header_height = 66
    width = max(rgb.shape[1], int(minimum_canvas_width))
    width += width % 2
    height = rgb.shape[0] + header_height
    height += height % 2
    canvas = Image.new("RGB", (width, height), "black")
    image_offset_x = (width - rgb.shape[1]) // 2
    canvas.paste(Image.fromarray(rgb), (image_offset_x, header_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    active_bursts = set()
    for identity in identities:
        status, burst_id = frame_identity_status(records, identity, frame_zero)
        if burst_id is not None:
            active_bursts.add(burst_id)
        color = COLORS[status]
        x, y = coordinates[identity]
        x = (x - x0) * magnification + image_offset_x
        y = (y - y0) * magnification + header_height
        half = int(box_half_size_px) * magnification
        line_width = (3 if status != "inactive" else 2) * magnification
        draw.rectangle(
            (x - half, y - half, x + half, y + half),
            outline=color,
            width=line_width,
        )
        label = identity.replace("roi_", "")
        text_box = draw.textbbox((x - half, y - half - 10), label, font=font)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((x - half, y - half - 10), label, fill=color, font=font)
    burst_text = (
        "none" if not active_bursts else ",".join(map(str, sorted(active_bursts)))
    )
    draw.text(
        (7, 5),
        f"{title} | UI frame {frame_zero + 1} | active burst {burst_text}",
        fill="white",
        font=font,
    )
    draw.text(
        (7, 24),
        "BLUE inactive missed identity   RED active + missed   GREEN active + recovered",
        fill=(215, 215, 215),
        font=font,
    )
    draw.text(
        (7, 43),
        "nested linear ranker, budget 58 | fixed raw display scale | labels are burst intervals",
        fill=(170, 170, 170),
        font=font,
    )
    return np.asarray(canvas)


def _probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    stream = json.loads(completed.stdout)["streams"][0]
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": stream.get("r_frame_rate"),
        "frame_count": int(stream["nb_read_frames"]),
        "duration_seconds": float(stream.get("duration", 0)),
        "bytes": path.stat().st_size,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def generate(
    config: InnovationRankerConfig,
    *,
    ranker_root: Path,
    output_dir: Path,
    recovery_field: str = "linear_recovered",
    fps: float = 10.0,
    box_half_size_px: int = 9,
    zoom_padding_px: int = 36,
) -> dict[str, Any]:
    """Write full-field and zoomed raw overlays for missed identities."""
    if fps <= 0 or fps > 60:
        raise ValueError("fps must be in (0, 60]")
    if box_half_size_px < 3 or box_half_size_px > 30:
        raise ValueError("box_half_size_px must be in [3, 30]")
    if output_dir.exists() or Path(str(output_dir) + ".partial").exists():
        raise FileExistsError(f"Diagnostic output already exists: {output_dir}")
    audit_path = ranker_root / "evaluation" / "per_neuron_audit.tsv"
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    source = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    if source.ndim != 3:
        raise ValueError("Source video must have shape [frames, rows, columns]")
    labels = label_core.load_labels(config.labels_tsv)
    audit_rows = _load_audit(audit_path)
    identities, records = missed_identity_records(
        audit_rows, labels, recovery_field=recovery_field
    )
    coordinates = _identity_coordinates(identities, records)
    start = int(config.frames["review_start_ui"]) - 1
    stop = int(config.frames["review_end_ui"])
    if not (0 <= start < stop <= len(source)):
        raise ValueError("Review interval is outside the source video")
    sample = np.asarray(source[start:stop:8, ::4, ::4], dtype=np.float32)
    display_lo, display_hi = np.percentile(sample, [0.5, 99.9])
    full_crop = (0, 0, int(source.shape[2]), int(source.shape[1]))
    zoom_crop = _zoom_box(
        coordinates, (int(source.shape[1]), int(source.shape[2])), zoom_padding_px
    )
    partial = Path(str(output_dir) + ".partial")
    partial.mkdir(parents=True, exist_ok=False)
    full_path = partial / "missed_neurons_raw_full.mp4"
    zoom_path = partial / "missed_neurons_raw_zoom.mp4"
    first_full = _render(
        source[start],
        frame_zero=start,
        identities=identities,
        coordinates=coordinates,
        records=records,
        display_lo=float(display_lo),
        display_hi=float(display_hi),
        crop=full_crop,
        box_half_size_px=box_half_size_px,
        title="Spon Ca Burst missed-neuron audit | full field",
    )
    first_zoom = _render(
        source[start],
        frame_zero=start,
        identities=identities,
        coordinates=coordinates,
        records=records,
        display_lo=float(display_lo),
        display_hi=float(display_hi),
        crop=zoom_crop,
        box_half_size_px=box_half_size_px,
        title="Spon Ca Burst missed-neuron audit | zoom",
        magnification=2,
        minimum_canvas_width=640,
    )
    writers = [
        _Mp4Writer(full_path, first_full.shape[:2], fps),
        _Mp4Writer(zoom_path, first_zoom.shape[:2], fps),
    ]
    try:
        writers[0].write(first_full)
        writers[1].write(first_zoom)
        for frame_zero in range(start + 1, stop):
            frame = source[frame_zero]
            writers[0].write(
                _render(
                    frame,
                    frame_zero=frame_zero,
                    identities=identities,
                    coordinates=coordinates,
                    records=records,
                    display_lo=float(display_lo),
                    display_hi=float(display_hi),
                    crop=full_crop,
                    box_half_size_px=box_half_size_px,
                    title="Spon Ca Burst missed-neuron audit | full field",
                )
            )
            writers[1].write(
                _render(
                    frame,
                    frame_zero=frame_zero,
                    identities=identities,
                    coordinates=coordinates,
                    records=records,
                    display_lo=float(display_lo),
                    display_hi=float(display_hi),
                    crop=zoom_crop,
                    box_half_size_px=box_half_size_px,
                    title="Spon Ca Burst missed-neuron audit | zoom",
                    magnification=2,
                    minimum_canvas_width=640,
                )
            )
        for writer in writers:
            writer.close()
    except Exception:
        for writer in writers:
            if writer.process.poll() is None:
                writer.abort()
        raise
    probes = {
        "full": _probe_video(full_path),
        "zoom": _probe_video(zoom_path),
    }
    expected_frames = stop - start
    if any(probe["frame_count"] != expected_frames for probe in probes.values()):
        raise RuntimeError("Encoded video frame count differs from review interval")
    missed_observations = [row for row in records if not row["recovered"]]
    payload = {
        "schema_version": 1,
        "status": "completed",
        "source_video": str(config.source_video),
        "source_ranker_root": str(ranker_root),
        "source_per_neuron_audit": str(audit_path),
        "recovery_field": recovery_field,
        "review_frames_ui_inclusive": [start + 1, stop],
        "source_frame_rate_hz": 50.0,
        "playback_fps": float(fps),
        "playback_speed_relative_to_source": float(fps / 50.0),
        "frame_count": expected_frames,
        "missed_identity_count": len(identities),
        "missed_observation_count": len(missed_observations),
        "missed_identities": identities,
        "display_source_limits": [float(display_lo), float(display_hi)],
        "display_contract": "one fixed raw-intensity scale; no per-frame normalization",
        "box_half_size_px": int(box_half_size_px),
        "zoom_crop_xyxy": list(zoom_crop),
        "zoom_magnification": 2,
        "legend": {
            "blue": "identity missed in at least one burst; inactive in this frame",
            "red": "label interval active and preferred linear ranker missed it",
            "green": "label interval active and preferred linear ranker recovered it",
        },
        "videos": {
            "full": {"path": "missed_neurons_raw_full.mp4", **probes["full"]},
            "zoom": {"path": "missed_neurons_raw_zoom.mp4", **probes["zoom"]},
        },
        "observations": records,
    }
    _atomic_json(partial / "manifest.json", payload)
    with (partial / "missed_observations.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = list(missed_observations[0])
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(missed_observations)
    partial.replace(output_dir)
    return payload
