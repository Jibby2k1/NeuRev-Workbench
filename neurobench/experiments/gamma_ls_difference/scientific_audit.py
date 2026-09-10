"""Scientific-audit renderer for the frozen Gamma-LS paper pipeline.

The metric runners intentionally seal compact numeric artifacts before opening
positive annotations.  This module consumes those sealed artifacts after the
fact.  It never selects a detector, changes a score, or alters a metric table.

Two entry points are provided:

``run_independent_scientific_audit``
    Replays the already selected signed-difference/Gamma-LS stages on CPU and
    renders the three strictly separated audit sections for ``15 right``.

``write_protected_h15_audit_readiness``
    Writes a fail-closed readiness packet for the protected Spon artifact.  The
    protected artifact does not contain a four-burst candidate stream for the
    exact deployment h15/mode-0.5 context, so manufacturing a passing visual
    audit from it would silently introduce a new label-adjacent evaluation.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import fftconvolve

from neurobench.algorithms.gamma_local_standardization import (
    GammaReferenceSpec,
    gamma_local_standardization,
    gamma_reference_kernel,
)
from neurobench.experiments.gamma_ls_difference.independent_validation import (
    _CausalRepresentationProcessor,
)
from neurobench.reports.scientific_audit import (
    require_three_section_scientific_audit,
)


GREEN = (70, 220, 125)
ORANGE = (255, 145, 35)
PALE_YELLOW = "#f6e7a1"
COMPARISON_PANELS = [
    "Raw matched comparison",
    "Signed difference + Gamma-LS matched comparison",
]
DEFAULT_MODEL_ANCHOR_LIMIT = 24
MODEL_CONSOLIDATION_RADIUS_PX = 3.0


class GammaScientificAuditError(RuntimeError):
    """Raised when a frozen audit input or rendered artifact is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GammaScientificAuditError(f"expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["id"]
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _artifact_index(root: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifacts": [
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.name != "artifact_index.json"
            and not path.name.endswith(".partial")
        ],
    }


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _verify_indexed_inputs(root: Path, relative_paths: Sequence[str]) -> None:
    index = _read_json(root / "artifact_index.json")
    by_path = {str(row["path"]): row for row in index.get("artifacts", [])}
    for relative in relative_paths:
        path = root / relative
        if relative not in by_path or not path.is_file():
            raise GammaScientificAuditError(f"indexed audit input is missing: {relative}")
        if _sha256(path) != str(by_path[relative]["sha256"]):
            raise GammaScientificAuditError(f"indexed audit input hash changed: {relative}")


def _fft_gamma_local_standardization(
    values: np.ndarray,
    reference: GammaReferenceSpec,
    *,
    kernel: np.ndarray | None = None,
    reference_mass: np.ndarray | None = None,
) -> np.ndarray:
    """CPU FFT replay of the frozen valid-renormalized Gamma-LS operator.

    The production metric path uses Torch convolution.  This replay is used
    only to make the already scored sequential stage visible.  The audit also
    writes a bounded numerical comparison against the Torch reference.
    """

    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 3 or source.size == 0:
        raise ValueError("values must be a non-empty TYX array")
    if reference.boundary_mode != "valid_renormalized_zero":
        raise ValueError("audit FFT replay supports valid_renormalized_zero only")
    if kernel is None:
        kernel = (
            gamma_reference_kernel(reference, device="cpu")
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
    kernel3 = np.asarray(kernel, dtype=np.float32)[None]
    if reference_mass is None:
        reference_mass = fftconvolve(
            np.ones((1, source.shape[1], source.shape[2]), dtype=np.float32),
            kernel3,
            mode="same",
            axes=(-2, -1),
        )
    mean = fftconvolve(source, kernel3, mode="same", axes=(-2, -1)) / reference_mass
    second = (
        fftconvolve(source * source, kernel3, mode="same", axes=(-2, -1))
        / reference_mass
    )
    standard_deviation = np.sqrt(np.maximum(second - mean * mean, 0.0))
    denominator = np.maximum(standard_deviation, float(reference.scale_floor))
    denominator += float(reference.epsilon)
    return ((source - mean) / denominator).astype(np.float32, copy=False)


def _best_lag_correlation(
    first: np.ndarray, second: np.ndarray, *, maximum_lag: int = 5
) -> tuple[float | None, int | None]:
    one = np.asarray(first, dtype=np.float64)
    two = np.asarray(second, dtype=np.float64)
    best: tuple[float, int] | None = None
    for lag in range(-maximum_lag, maximum_lag + 1):
        if abs(lag) >= min(len(one), len(two)):
            continue
        if lag < 0:
            left, right = one[-lag:], two[: len(two) + lag]
        elif lag > 0:
            left, right = one[: len(one) - lag], two[lag:]
        else:
            left, right = one, two
        finite = np.isfinite(left) & np.isfinite(right)
        left, right = left[finite], right[finite]
        if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
            continue
        correlation = float(np.corrcoef(left, right)[0, 1])
        candidate = (correlation, lag)
        if best is None or (candidate[0], -abs(candidate[1]), -candidate[1]) > (
            best[0],
            -abs(best[1]),
            -best[1],
        ):
            best = candidate
    return (None, None) if best is None else best


def _limits_unsigned(values: np.ndarray, low: float = 1.0, high: float = 99.8) -> tuple[float, float]:
    lower, upper = np.percentile(np.asarray(values, dtype=np.float32), [low, high])
    return float(lower), max(float(upper), float(lower) + np.finfo(np.float32).eps)


def _gray_unsigned(
    frame: np.ndarray, limits: tuple[float, float], size: tuple[int, int]
) -> Image.Image:
    lower, upper = limits
    scaled = np.clip((np.asarray(frame, dtype=np.float32) - lower) / (upper - lower), 0, 1)
    return Image.fromarray((scaled * 255).astype(np.uint8), mode="L").resize(
        size, Image.Resampling.BILINEAR
    ).convert("RGB")


def _gray_signed(frame: np.ndarray, absolute_limit: float, size: tuple[int, int]) -> Image.Image:
    scaled = np.clip(
        0.5 + np.asarray(frame, dtype=np.float32) / (2.0 * absolute_limit), 0, 1
    )
    return Image.fromarray((scaled * 255).astype(np.uint8), mode="L").resize(
        size, Image.Resampling.BILINEAR
    ).convert("RGB")


@dataclass
class _VideoWriter:
    path: Path
    width: int
    height: int
    fps: float
    process: subprocess.Popen[bytes] | None = None
    frames: int = 0

    def open(self) -> None:
        if self.process is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_suffix(".partial.mp4")
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
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-preset",
            "veryfast",
            "-crf",
            "27",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(partial),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.open()
        assert self.process is not None and self.process.stdin is not None
        values = np.asarray(frame, dtype=np.uint8)
        if values.shape != (self.height, self.width, 3):
            raise ValueError(f"video frame shape changed for {self.path}: {values.shape}")
        self.process.stdin.write(np.ascontiguousarray(values).tobytes())
        self.frames += 1

    def close(self) -> None:
        if self.process is None or self.process.stdin is None or self.process.stderr is None:
            raise GammaScientificAuditError(f"video received no frames: {self.path}")
        self.process.stdin.close()
        error = self.process.stderr.read().decode("utf-8", errors="replace")
        return_code = self.process.wait()
        if return_code:
            raise GammaScientificAuditError(
                f"ffmpeg failed for {self.path}: {error[-2000:]}"
            )
        self.path.with_suffix(".partial.mp4").replace(self.path)


def _draw_markers(
    canvas: Image.Image,
    rows: Iterable[Mapping[str, Any]],
    *,
    color: tuple[int, int, int],
    panels: int,
    panel_size: tuple[int, int],
    header: int,
    source_shape: tuple[int, int],
    square: bool = False,
) -> None:
    draw = ImageDraw.Draw(canvas)
    panel_width, panel_height = panel_size
    source_height, source_width = source_shape
    for row in rows:
        x, y = float(row["x_px"]), float(row["y_px"])
        for panel in range(panels):
            center_x = panel * panel_width + x * panel_width / source_width
            center_y = header + y * panel_height / source_height
            box = (center_x - 4, center_y - 4, center_x + 4, center_y + 4)
            if square:
                draw.rectangle(box, outline=color, width=2)
            else:
                draw.ellipse(box, outline=color, width=2)


def _full_panel(
    stages: Sequence[np.ndarray],
    *,
    titles: Sequence[str],
    raw_limits: tuple[float, float],
    difference_limit: float,
    gamma_limit: float,
    lme_limits: tuple[float, float],
    source_frame: int,
) -> Image.Image:
    panel_size, header = (220, 220), 44
    canvas = Image.new("RGB", (panel_size[0] * 4, panel_size[1] + header), "black")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((6, 5), f"source frame UI {source_frame + 1}", fill="white", font=font)
    images = [
        _gray_unsigned(stages[0], raw_limits, panel_size),
        _gray_signed(stages[1], difference_limit, panel_size),
        _gray_signed(stages[2], gamma_limit, panel_size),
        _gray_unsigned(stages[3], lme_limits, panel_size),
    ]
    for index, (image, title) in enumerate(zip(images, titles, strict=True)):
        canvas.paste(image, (index * panel_size[0], header))
        draw.text((index * panel_size[0] + 5, 24), title, fill="white", font=font)
    return canvas


def _crop_bounds(x: float, y: float, width: int, height: int, radius: int = 24) -> tuple[int, int, int, int]:
    center_x, center_y = int(round(x)), int(round(y))
    x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
    y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
    return x0, y0, x1, y1


def _close_panel(
    stages: Sequence[np.ndarray],
    *,
    titles: Sequence[str],
    raw_limits: tuple[float, float],
    difference_limit: float,
    gamma_limit: float,
    lme_limits: tuple[float, float],
    source_frame: int,
    x_px: float,
    y_px: float,
    marker_color: tuple[int, int, int],
) -> Image.Image:
    height, width = stages[0].shape
    x0, y0, x1, y1 = _crop_bounds(x_px, y_px, width, height)
    panel, header = 96, 34
    canvas = Image.new("RGB", (panel * 4, panel + header), "black")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((4, 3), f"source UI {source_frame + 1}", fill="white", font=font)
    crops = [stage[y0:y1, x0:x1] for stage in stages]
    images = [
        _gray_unsigned(crops[0], raw_limits, (panel, panel)),
        _gray_signed(crops[1], difference_limit, (panel, panel)),
        _gray_signed(crops[2], gamma_limit, (panel, panel)),
        _gray_unsigned(crops[3], lme_limits, (panel, panel)),
    ]
    for index, (image, title) in enumerate(zip(images, titles, strict=True)):
        canvas.paste(image, (index * panel, header))
        draw.text((index * panel + 3, 18), title, fill="white", font=font)
        center_x = index * panel + (x_px - x0) * panel / max(x1 - x0, 1)
        center_y = header + (y_px - y0) * panel / max(y1 - y0, 1)
        draw.ellipse(
            (center_x - 5, center_y - 5, center_x + 5, center_y + 5),
            outline=marker_color,
            width=2,
        )
    return canvas


def _fixed_deployment_full_panel(
    stages: Sequence[np.ndarray],
    *,
    titles: Sequence[str],
    raw_limits: tuple[float, float],
    conditioned_limits: tuple[float, float],
    difference_limit: float,
    gamma_limit: float,
    occupancy_limit: float,
    source_frame_ui: int,
) -> Image.Image:
    """Five-stage full-field panel for the fixed protected replay."""

    panel_size, header = (200, 200), 44
    canvas = Image.new(
        "RGB", (panel_size[0] * 5, panel_size[1] + header), "black"
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((6, 5), f"source frame UI {source_frame_ui}", fill="white", font=font)
    images = [
        _gray_unsigned(stages[0], raw_limits, panel_size),
        _gray_unsigned(stages[1], conditioned_limits, panel_size),
        _gray_signed(stages[2], difference_limit, panel_size),
        _gray_signed(stages[3], gamma_limit, panel_size),
        _gray_unsigned(stages[4], (0.0, occupancy_limit), panel_size),
    ]
    for index, (stage_image, title) in enumerate(zip(images, titles, strict=True)):
        canvas.paste(stage_image, (index * panel_size[0], header))
        draw.text((index * panel_size[0] + 4, 24), title, fill="white", font=font)
    return canvas


def _fixed_deployment_close_panel(
    stages: Sequence[np.ndarray],
    *,
    titles: Sequence[str],
    raw_limits: tuple[float, float],
    conditioned_limits: tuple[float, float],
    difference_limit: float,
    gamma_limit: float,
    occupancy_limit: float,
    source_frame_ui: int,
    x_px: float,
    y_px: float,
    marker_color: tuple[int, int, int],
) -> Image.Image:
    height, width = stages[0].shape
    x0, y0, x1, y1 = _crop_bounds(x_px, y_px, width, height)
    panel, header = 96, 34
    canvas = Image.new("RGB", (panel * 5, panel + header), "black")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((4, 3), f"source UI {source_frame_ui}", fill="white", font=font)
    crops = [stage[y0:y1, x0:x1] for stage in stages]
    images = [
        _gray_unsigned(crops[0], raw_limits, (panel, panel)),
        _gray_unsigned(crops[1], conditioned_limits, (panel, panel)),
        _gray_signed(crops[2], difference_limit, (panel, panel)),
        _gray_signed(crops[3], gamma_limit, (panel, panel)),
        _gray_unsigned(crops[4], (0.0, occupancy_limit), (panel, panel)),
    ]
    for index, (stage_image, title) in enumerate(zip(images, titles, strict=True)):
        canvas.paste(stage_image, (index * panel, header))
        draw.text((index * panel + 3, 18), title, fill="white", font=font)
        center_x = index * panel + (x_px - x0) * panel / max(x1 - x0, 1)
        center_y = header + (y_px - y0) * panel / max(y1 - y0, 1)
        draw.ellipse(
            (center_x - 5, center_y - 5, center_x + 5, center_y + 5),
            outline=marker_color,
            width=2,
        )
    return canvas


def _fixed_deployment_trace_figure(
    path: Path,
    traces: Mapping[str, np.ndarray],
    *,
    title: str,
    color: str,
    spans: Sequence[tuple[int, int]],
    review_start_ui: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ("raw", "conditioned", "difference", "gamma", "occupancy")
    names = (
        "Acquired raw",
        "Causal conditioned",
        "Signed difference",
        "Signed Gamma-LS",
        "Burst occupancy",
    )
    figure, axes = plt.subplots(
        5, 1, figsize=(9, 8.3), sharex=True, constrained_layout=True
    )
    frames = np.arange(len(traces["raw"])) + int(review_start_ui)
    for axis, key, label in zip(axes, keys, names, strict=True):
        axis.plot(frames, traces[key], color=color, linewidth=0.7)
        for start, stop in spans:
            axis.axvspan(start, stop, color=color, alpha=0.08)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("source frame (UI one-based)")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def _write_fixed_stage_crop_npz(
    path: Path,
    metadata_path: Path,
    preview_path: Path,
    *,
    raw_movie: np.ndarray,
    conditioned: np.ndarray,
    difference: np.ndarray,
    gamma: np.ndarray,
    occupancy: np.ndarray,
    candidate: Mapping[str, Any],
    review_start_ui: int,
    source_movie_sha256: str,
    threshold_z: float,
    nms_distance_px: int,
    crop_radius_px: int = 40,
) -> dict[str, Any]:
    """Persist the deterministic label-free same-crop manuscript stage packet.

    The protected replay's primary proposal head thresholds each frame, forms a
    burst occupancy map, and applies NMS to that aggregate.  For architecture
    illustration only, this packet additionally applies the *same frozen q=1
    threshold* and maintained NMS6 implementation to the selected Gamma frame.
    The two products are stored under distinct names and explicit semantics.
    """

    from neurobench.experiments.gamma_ls_difference.evaluation import (
        strict_separated_nms,
    )

    source_ui = int(candidate["peak_source_frame_ui"])
    review_index = source_ui - int(review_start_ui)
    if not 1 <= review_index < len(conditioned):
        raise GammaScientificAuditError(
            "Fig3 candidate must have a persisted preceding conditioned frame"
        )
    x_px, y_px = int(candidate["x_px"]), int(candidate["y_px"])
    height, width = conditioned.shape[1:]
    x0 = max(0, x_px - int(crop_radius_px))
    x1 = min(width, x_px + int(crop_radius_px) + 1)
    y0 = max(0, y_px - int(crop_radius_px))
    y1 = min(height, y_px + int(crop_radius_px) + 1)
    threshold = float(threshold_z)
    nms_distance = int(nms_distance_px)
    if not math.isfinite(threshold) or nms_distance != 6:
        raise GammaScientificAuditError(
            "Fig3 requires the frozen finite q=1 threshold and primary NMS6"
        )
    gamma_frame = np.asarray(gamma[review_index], dtype=np.float32)
    framewise_peaks = strict_separated_nms(
        gamma_frame,
        distance_px=nms_distance,
        threshold=threshold,
        limit=10_000,
    )
    threshold_exceedance = np.asarray(gamma_frame > threshold, dtype=np.uint8)
    framewise_nms_mask = np.zeros_like(threshold_exceedance, dtype=np.uint8)
    for _score, peak_x, peak_y in framewise_peaks:
        framewise_nms_mask[int(peak_y), int(peak_x)] = 1
    arrays = {
        "raw": np.asarray(
            raw_movie[source_ui - 1, y0:y1, x0:x1], dtype=np.uint16
        ),
        "conditioned_previous": np.asarray(
            conditioned[review_index - 1, y0:y1, x0:x1], dtype=np.float32
        ),
        "conditioned_current": np.asarray(
            conditioned[review_index, y0:y1, x0:x1], dtype=np.float32
        ),
        "difference": np.asarray(
            difference[review_index, y0:y1, x0:x1], dtype=np.float32
        ),
        "gamma_ls": np.asarray(
            gamma[review_index, y0:y1, x0:x1], dtype=np.float32
        ),
        "threshold_exceedance": np.asarray(
            threshold_exceedance[y0:y1, x0:x1], dtype=np.uint8
        ),
        "proposals": np.asarray(
            framewise_nms_mask[y0:y1, x0:x1], dtype=np.uint8
        ),
        "burst_occupancy": np.asarray(
            occupancy[y0:y1, x0:x1], dtype=np.float32
        ),
    }
    residual = arrays["conditioned_current"] - arrays["conditioned_previous"]
    maximum_error = float(np.max(np.abs(residual - arrays["difference"])))
    if maximum_error > 1e-5:
        raise GammaScientificAuditError(
            f"Fig3 conditioned-difference alignment failed: {maximum_error}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    raw_limits = tuple(map(float, np.percentile(arrays["raw"], [1.0, 99.0])))
    conditioned_limits = tuple(
        map(float, np.percentile(arrays["conditioned_current"], [1.0, 99.0]))
    )
    difference_limit = max(
        float(np.percentile(np.abs(arrays["difference"]), 99.0)), 1e-6
    )
    gamma_limit = max(
        float(np.percentile(np.abs(arrays["gamma_ls"]), 99.0)), 1e-6
    )
    occupancy_limit = max(float(np.max(arrays["burst_occupancy"])), 1e-6)
    panel_size, header = (180, 180), 48
    canvas = Image.new("RGB", (panel_size[0] * 6, panel_size[1] + header), "black")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    panels = [
        _gray_unsigned(arrays["raw"], raw_limits, panel_size),
        _gray_unsigned(arrays["conditioned_current"], conditioned_limits, panel_size),
        _gray_signed(arrays["difference"], difference_limit, panel_size),
        _gray_signed(arrays["gamma_ls"], gamma_limit, panel_size),
        _gray_signed(arrays["gamma_ls"], gamma_limit, panel_size),
        _gray_unsigned(
            arrays["burst_occupancy"], (0.0, occupancy_limit), panel_size
        ),
    ]
    exceedance_image = Image.fromarray(
        arrays["threshold_exceedance"] * np.uint8(255), mode="L"
    ).resize(panel_size, resample=Image.Resampling.NEAREST)
    overlay_pixels = np.asarray(panels[4], dtype=np.uint8).copy()
    exceedance_pixels = np.asarray(exceedance_image, dtype=np.uint8) > 0
    overlay_pixels[exceedance_pixels] = (
        overlay_pixels[exceedance_pixels].astype(np.uint16) // 3
        + np.asarray((170, 145, 30), dtype=np.uint16)
    ).astype(np.uint8)
    panels[4] = Image.fromarray(overlay_pixels, mode="RGB")
    titles = (
        "Acquired raw",
        "Causal conditioned",
        "Signed difference",
        "Gamma-LS z",
        "q1 threshold + NMS6",
        "Protected burst occupancy",
    )
    for index, (panel_image, title) in enumerate(zip(panels, titles, strict=True)):
        x_offset = index * panel_size[0]
        canvas.paste(panel_image, (x_offset, header))
        draw.text((x_offset + 4, 27), title, fill="white", font=font)
    draw.text(
        (5, 5),
        f"source UI {source_ui}; same 81x81 crop; q1 threshold={threshold:.4g}",
        fill="white",
        font=font,
    )
    for peak_score, peak_x, peak_y in framewise_peaks:
        if not (x0 <= int(peak_x) < x1 and y0 <= int(peak_y) < y1):
            continue
        center_x = 4 * panel_size[0] + (int(peak_x) - x0 + 0.5) * (
            panel_size[0] / (x1 - x0)
        )
        center_y = header + (int(peak_y) - y0 + 0.5) * (
            panel_size[1] / (y1 - y0)
        )
        draw.ellipse(
            (center_x - 7, center_y - 7, center_x + 7, center_y + 7),
            outline=ORANGE,
            width=3,
        )
        draw.text(
            (center_x + 8, center_y - 8),
            f"{float(peak_score):.2f}",
            fill=ORANGE,
            font=font,
        )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(preview_path)
    framewise_peaks_in_crop = [
        {
            "score_z": float(score),
            "x_px_full_frame": int(peak_x),
            "y_px_full_frame": int(peak_y),
            "x_px_crop": int(peak_x) - x0,
            "y_px_crop": int(peak_y) - y0,
        }
        for score, peak_x, peak_y in framewise_peaks
        if x0 <= int(peak_x) < x1 and y0 <= int(peak_y) < y1
    ]
    metadata = {
        "schema_version": 1,
        "selection_rule": "highest_ranked_candidate_at_predeclared_label_free_audit_state",
        "candidate_id": candidate["candidate_id"],
        "burst_id": int(candidate["burst_id"]),
        "source_frame_ui": source_ui,
        "review_frame_zero": review_index,
        "center_x_px": x_px,
        "center_y_px": y_px,
        "crop_xyxy_half_open": [x0, y0, x1, y1],
        "coordinate_convention": "x=column,y=row",
        "source_movie_sha256": source_movie_sha256,
        "raw_provenance": (
            "exact uint16 acquired frame extracted from the hash-bound source movie"
        ),
        "conditioned_previous_provenance": (
            "persisted preceding aligned conditioned_current frame; not reconstructed"
        ),
        "threshold_exceedance_semantics": (
            "binary exact same-frame Gamma-LS > frozen protected q1 threshold"
        ),
        "proposals_semantics": (
            "binary exact deterministic NMS6 peaks on the selected full Gamma frame "
            "after the frozen protected q1 threshold; cropped only after NMS"
        ),
        "burst_occupancy_semantics": (
            "sealed fraction of burst frames Gamma-LS > frozen protected q1 threshold; "
            "this is the protected replay primary aggregation head"
        ),
        "framewise_diagnostic": {
            "threshold_z": threshold,
            "threshold_rule": "strictly_greater_than",
            "threshold_source": "threshold_calibration.tsv:a_train_b_test:q1p0",
            "threshold_calibration_unit": "nms_peaks_per_duration_matched_pseudo_burst",
            "nms_distance_px": nms_distance,
            "nms_algorithm": "maintained_deterministic_greedy_euclidean_separated",
            "nms_applied_on": "full_340x573_selected_gamma_frame_before_crop",
            "full_frame_threshold_exceedance_pixel_count": int(
                np.count_nonzero(threshold_exceedance)
            ),
            "full_frame_nms_peak_count": len(framewise_peaks),
            "crop_nms_peak_count": len(framewise_peaks_in_crop),
            "peaks_in_crop": framewise_peaks_in_crop,
            "candidate_center_is_framewise_nms_peak": bool(
                framewise_nms_mask[y_px, x_px]
            ),
            "claim_boundary": (
                "label-free same-frame diagnostic under the protected pseudo-burst-"
                "calibrated threshold; not the protected primary burst-occupancy head "
                "and not the separately calibrated full-record operational head"
            ),
        },
        "preview": preview_path.name,
        "difference_alignment_max_abs_error": maximum_error,
        "keys": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
    }
    _atomic_json(metadata_path, metadata)
    return metadata


def _consolidate_model_anchors(
    candidates: Sequence[Mapping[str, Any]], *, limit: int
) -> tuple[list[dict[str, Any]], dict[str, str], int]:
    if limit < 1:
        raise ValueError("model anchor limit must be positive")
    ordered = sorted(
        (dict(row) for row in candidates),
        key=lambda row: (
            int(row["block_id"]),
            int(row["block_rank"]),
            str(row["candidate_id"]),
        ),
    )
    clusters: list[dict[str, Any]] = []
    assignment: dict[str, str] = {}
    squared_radius = MODEL_CONSOLIDATION_RADIUS_PX**2
    for candidate in ordered:
        x, y = float(candidate["x_px"]), float(candidate["y_px"])
        cluster = next(
            (
                item
                for item in clusters
                if (float(item["x_px"]) - x) ** 2 + (float(item["y_px"]) - y) ** 2
                <= squared_radius
            ),
            None,
        )
        if cluster is None:
            cluster = {
                "model_roi_id": f"model_roi_{len(clusters) + 1:04d}",
                "x_px": x,
                "y_px": y,
                "candidate_ids": [],
                "block_ids": [],
                "best_block_rank": int(candidate["block_rank"]),
                "best_pooled_lme_score": float(candidate["pooled_lme_score"]),
            }
            clusters.append(cluster)
        cluster["candidate_ids"].append(str(candidate["candidate_id"]))
        cluster["block_ids"].append(int(candidate["block_id"]))
        cluster["best_block_rank"] = min(
            int(cluster["best_block_rank"]), int(candidate["block_rank"])
        )
        cluster["best_pooled_lme_score"] = max(
            float(cluster["best_pooled_lme_score"]),
            float(candidate["pooled_lme_score"]),
        )
        assignment[str(candidate["candidate_id"])] = str(cluster["model_roi_id"])
    ranked = sorted(
        clusters,
        key=lambda item: (
            -len(set(item["block_ids"])),
            int(item["best_block_rank"]),
            -float(item["best_pooled_lme_score"]),
            str(item["model_roi_id"]),
        ),
    )
    selected = ranked[:limit]
    return selected, assignment, len(clusters)


def _trace_figure(
    path: Path,
    traces: Mapping[str, np.ndarray],
    *,
    title: str,
    color: str,
    spans: Sequence[tuple[int, int]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["Acquired raw", "Signed difference", "Signed Gamma-LS", "Block LME"]
    figure, axes = plt.subplots(4, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    frames = np.arange(len(traces["raw"])) + 1
    for axis, key, label in zip(axes, ["raw", "difference", "gamma", "lme"], names, strict=True):
        axis.plot(frames, traces[key], color=color, linewidth=0.7)
        for start, stop in spans:
            axis.axvspan(start + 1, stop + 1, color=color, alpha=0.08)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("source frame (UI one-based)")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def _candidate_blocks(candidate_rows: Sequence[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        grouped.setdefault(int(row["block_id"]), []).append(dict(row))
    return grouped


def _expected_marker_frame_for_expert(row: Mapping[str, Any]) -> int:
    return int(row["start_frame_zero_inclusive"])


def _source_hash_matches_contract(contract: Mapping[str, Any]) -> bool:
    source = contract["source"]
    path = Path(str(source["movie_path"]))
    return path.is_file() and _sha256(path) == str(source["movie_sha256"])


def run_independent_scientific_audit(
    metric_root: str | Path,
    audit_root: str | Path,
    *,
    model_anchor_limit: int = DEFAULT_MODEL_ANCHOR_LIMIT,
    cpu_threads: int = 4,
) -> dict[str, Any]:
    """Render the independent three-section packet without changing candidates."""

    import tifffile
    import torch

    metric = Path(metric_root).expanduser().resolve()
    destination = Path(audit_root).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not metric.is_dir() or not destination.parent.is_dir():
        raise FileNotFoundError("metric root and audit output parent must exist")
    metric_validation = _read_json(metric / "validation.json")
    if not metric_validation.get("all_checks_pass"):
        raise GammaScientificAuditError("independent metric artifact is not validated")
    required_inputs = [
        "run_contract.json",
        "calibration.json",
        "candidate_score_seal.json",
        "block_lme_score_maps.npy",
        "raw_max_projection.npy",
        "scientific_audit_inputs/expert_occurrences.tsv",
        "scientific_audit_inputs/primary_model_candidates.tsv",
        "scientific_audit_inputs/nearest_time_eligible_candidate_pairs.tsv",
        "scientific_audit_inputs/primary_one_to_one_matches.tsv",
    ]
    _verify_indexed_inputs(metric, required_inputs)
    contract = _read_json(metric / "run_contract.json")
    if contract.get("representation_arm") != "difference_signed":
        raise GammaScientificAuditError("this renderer is frozen to difference_signed")
    if not _source_hash_matches_contract(contract):
        raise GammaScientificAuditError("source movie is missing or its SHA-256 changed")

    expert_occurrences = _read_tsv(
        metric / "scientific_audit_inputs/expert_occurrences.tsv"
    )
    candidates = _read_tsv(
        metric / "scientific_audit_inputs/primary_model_candidates.tsv"
    )
    nearest_pairs = _read_tsv(
        metric / "scientific_audit_inputs/nearest_time_eligible_candidate_pairs.tsv"
    )
    primary_matches = _read_tsv(
        metric / "scientific_audit_inputs/primary_one_to_one_matches.tsv"
    )
    if len(expert_occurrences) != 51 or len(candidates) != 32 * 58:
        raise GammaScientificAuditError("frozen expert or B58 candidate count changed")
    if len(nearest_pairs) != len(expert_occurrences):
        raise GammaScientificAuditError("nearest-pair coverage is incomplete")
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise GammaScientificAuditError("candidate IDs are not unique")

    unique_experts: dict[str, dict[str, Any]] = {}
    for row in expert_occurrences:
        unique_experts.setdefault(str(row["annotation_id"]), dict(row))
    if len(unique_experts) != 10:
        raise GammaScientificAuditError("expert ROI identity count changed")
    selected_anchors, cluster_assignment, prelimit_cluster_count = _consolidate_model_anchors(
        candidates, limit=model_anchor_limit
    )
    selected_cluster_ids = {str(row["model_roi_id"]) for row in selected_anchors}
    candidate_by_id = {str(row["candidate_id"]): dict(row) for row in candidates}
    occurrence_by_id = {str(row["positive_id"]): dict(row) for row in expert_occurrences}
    nearest_by_positive = {
        str(row["positive_id"]): dict(row) for row in nearest_pairs
    }
    matched_by_positive = {
        str(row["positive_id"]): dict(row) for row in primary_matches
    }
    if set(nearest_by_positive) != set(occurrence_by_id):
        raise GammaScientificAuditError("nearest pairs do not cover the exact positives")

    work = destination.parent / f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    work.mkdir()
    os.environ.setdefault("MPLCONFIGDIR", str(work / ".matplotlib"))
    expert_root = work / "1_Expert_Annotations"
    model_root = work / "2_Model_Annotations"
    comparison_root = work / "3_Comparison"
    for path in (
        expert_root / "videos/closeups",
        expert_root / "figures/traces",
        expert_root / "metadata",
        model_root / "videos/closeups",
        model_root / "figures/traces",
        model_root / "metadata",
        comparison_root / "trace_comparisons",
        work / "projection_checks",
    ):
        path.mkdir(parents=True, exist_ok=True)

    try:
        for name in (
            "expert_projection_coordinate_check.png",
            "model_projection_coordinate_check.png",
            "comparison_projection_coordinate_check.png",
        ):
            shutil.copy2(metric / "scientific_audit_inputs" / name, work / "projection_checks" / name)

        movie = tifffile.memmap(contract["source"]["movie_path"], mode="r")
        if tuple(map(int, movie.shape)) != (1608, 512, 512) or str(movie.dtype) != "uint16":
            raise GammaScientificAuditError("source movie shape/dtype changed")
        block_maps = np.load(metric / "block_lme_score_maps.npy", mmap_mode="r")
        if tuple(map(int, block_maps.shape)) != (32, 512, 512):
            raise GammaScientificAuditError("sealed block-LME map shape changed")
        calibration = _read_json(metric / "calibration.json")
        gamma = contract["gamma_context"]
        reference = GammaReferenceSpec.from_mode(
            str(gamma["context_id"]),
            support_width_px=int(gamma["support_width_px"]),
            shape_n=float(gamma["shape_n"]),
            mode_radius_px=float(gamma["mode_radius_px"]),
            guard_radius_px=float(gamma["guard_radius_px"]),
            support_geometry=str(gamma["support_geometry"]),
            boundary_mode=str(gamma["boundary_mode"]),
            epsilon=float(gamma["epsilon"]),
            scale_floor=float(calibration["scale_floor"]),
        )
        torch.set_num_threads(int(cpu_threads))
        kernel = (
            gamma_reference_kernel(reference, device="cpu")
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        reference_mass = fftconvolve(
            np.ones((1, 512, 512), dtype=np.float32),
            kernel[None],
            mode="same",
            axes=(-2, -1),
        )

        # Fixed, label-free display scales come from the original calibration
        # interval and sealed pooled maps.  They never affect ranking or metrics.
        scale_processor = _CausalRepresentationProcessor(
            arm="difference_signed", model=None, device=torch.device("cpu")
        )
        scale_representation, _ = scale_processor.process(
            np.asarray(movie[:100]), source_start_frame=0
        )
        scale_values = scale_representation.detach().cpu().numpy().astype(np.float32, copy=False)
        scale_gamma = _fft_gamma_local_standardization(
            scale_values, reference, kernel=kernel, reference_mass=reference_mass
        )
        raw_sample = np.asarray(movie[::101, ::4, ::4], dtype=np.float32)
        raw_limits = _limits_unsigned(raw_sample, 0.5, 99.8)
        difference_limit = max(float(np.percentile(np.abs(scale_values), 99.8)), 1e-6)
        gamma_limit = max(float(np.percentile(np.abs(scale_gamma), 99.8)), 1e-6)
        lme_limits = (0.0, max(float(np.percentile(block_maps, 99.8)), 1e-6))
        display_scales = {
            "raw": {"type": "fixed_linear", "vmin": raw_limits[0], "vmax": raw_limits[1]},
            "difference_signed": {
                "type": "fixed_symmetric_zero_midgray",
                "absolute_limit": difference_limit,
            },
            "signed_radial_gamma_ls": {
                "type": "fixed_symmetric_zero_midgray",
                "absolute_limit": gamma_limit,
            },
            "block_lme_evidence": {
                "type": "fixed_zero_black",
                "vmin": lme_limits[0],
                "vmax": lme_limits[1],
            },
            "scale_population": "first_100_source_frame_calibration_and_sealed_block_maps",
            "labels_used": False,
        }
        _atomic_json(work / "display_scales.json", display_scales)

        # Bounded numerical parity check between audit FFT replay and the exact
        # Torch production operator.
        reference_values = gamma_local_standardization(
            scale_representation[:8], reference, device="cpu", chunk_frames=4,
            return_statistics=False,
        ).values.detach().cpu().numpy()
        replay_values = scale_gamma[:8]
        absolute_difference = np.abs(reference_values - replay_values)
        replay_validation = {
            "schema_version": 1,
            "production_reference": "Torch conv2d Gamma-LS on CPU",
            "audit_replay": "SciPy FFT convolution with identical kernel and valid-reference mass",
            "frames_compared": 8,
            "max_absolute_error": float(np.max(absolute_difference)),
            "p999_absolute_error": float(np.quantile(absolute_difference, 0.999)),
            "mean_absolute_error": float(np.mean(absolute_difference)),
            "tolerance": 0.0001,
            "pass": bool(float(np.max(absolute_difference)) <= 0.0001),
        }
        if not replay_validation["pass"]:
            raise GammaScientificAuditError("CPU FFT Gamma replay failed numerical parity")
        _atomic_json(work / "replay_numerical_validation.json", replay_validation)

        point_rows: dict[str, dict[str, Any]] = {}
        for expert_id, row in unique_experts.items():
            point_rows[f"expert::{expert_id}"] = {
                "x_px": float(row["x_px"]), "y_px": float(row["y_px"])
            }
        for pair in nearest_pairs:
            candidate = candidate_by_id[str(pair["nearest_time_eligible_candidate_id"])]
            point_rows[f"candidate::{candidate['candidate_id']}"] = {
                "x_px": float(candidate["x_px"]), "y_px": float(candidate["y_px"])
            }
        for anchor in selected_anchors:
            point_rows[f"anchor::{anchor['model_roi_id']}"] = {
                "x_px": float(anchor["x_px"]), "y_px": float(anchor["y_px"])
            }
        point_ids = sorted(point_rows)
        point_index = {point_id: index for index, point_id in enumerate(point_ids)}
        point_x = np.asarray([int(round(point_rows[key]["x_px"])) for key in point_ids])
        point_y = np.asarray([int(round(point_rows[key]["y_px"])) for key in point_ids])
        traces = {
            key: np.full((len(point_ids), 1600), np.nan, dtype=np.float32)
            for key in ("raw", "difference", "gamma", "lme")
        }

        expert_intervals: dict[str, list[tuple[int, int]]] = {}
        for row in expert_occurrences:
            expert_intervals.setdefault(str(row["annotation_id"]), []).append(
                (
                    int(row["start_frame_zero_inclusive"]),
                    int(row["stop_frame_zero_inclusive"]),
                )
            )
        anchor_by_block: dict[int, list[dict[str, Any]]] = {}
        for anchor in selected_anchors:
            for block_id in sorted(set(map(int, anchor["block_ids"]))):
                anchor_by_block.setdefault(block_id, []).append(anchor)
        candidates_by_block = _candidate_blocks(candidates)
        expert_by_frame: dict[int, list[dict[str, Any]]] = {}
        for row in expert_occurrences:
            for frame in range(
                int(row["start_frame_zero_inclusive"]),
                int(row["stop_frame_zero_inclusive"]) + 1,
            ):
                if 0 <= frame < 1600:
                    expert_by_frame.setdefault(frame, []).append(dict(row))

        stage_titles = ["Acquired raw", "Signed difference", "Signed Gamma-LS", "Block LME"]
        expert_full = _VideoWriter(
            expert_root / "videos/expert_annotations_full_field.mp4", 880, 264, 50.0
        )
        model_full = _VideoWriter(
            model_root / "videos/model_annotations_sequential_full_field.mp4", 880, 264, 50.0
        )
        expert_writers = {
            expert_id: _VideoWriter(
                expert_root / f"videos/closeups/{_safe_id(expert_id)}.mp4", 384, 130, 50.0
            )
            for expert_id in unique_experts
        }
        model_writers = {
            str(anchor["model_roi_id"]): _VideoWriter(
                model_root / f"videos/closeups/{anchor['model_roi_id']}.mp4", 384, 130, 50.0
            )
            for anchor in selected_anchors
        }

        processor = _CausalRepresentationProcessor(
            arm="difference_signed", model=None, device=torch.device("cpu")
        )
        for block_index in range(32):
            start, stop = block_index * 50, (block_index + 1) * 50
            source = np.asarray(movie[start:stop])
            representation, source_indices_tensor = processor.process(
                source, source_start_frame=start
            )
            source_indices = source_indices_tensor.detach().cpu().numpy().astype(np.int64)
            difference = representation.detach().cpu().numpy().astype(np.float32, copy=False)
            gamma_scores = _fft_gamma_local_standardization(
                difference, reference, kernel=kernel, reference_mass=reference_mass
            )
            difference_by_source = np.zeros((50, 512, 512), dtype=np.float32)
            gamma_by_source = np.zeros((50, 512, 512), dtype=np.float32)
            relative = source_indices - start
            difference_by_source[relative] = difference
            gamma_by_source[relative] = gamma_scores
            lme = np.asarray(block_maps[block_index], dtype=np.float32)

            traces["raw"][:, start:stop] = source[:, point_y, point_x].T
            traces["difference"][:, source_indices] = difference[:, point_y, point_x].T
            traces["gamma"][:, source_indices] = gamma_scores[:, point_y, point_x].T
            traces["lme"][:, start:stop] = lme[point_y, point_x, None]

            block_candidates = candidates_by_block[block_index + 1]
            active_anchors = anchor_by_block.get(block_index + 1, [])
            for relative_frame, source_frame in enumerate(range(start, stop)):
                stages = [
                    source[relative_frame],
                    difference_by_source[relative_frame],
                    gamma_by_source[relative_frame],
                    lme,
                ]
                base = _full_panel(
                    stages,
                    titles=stage_titles,
                    raw_limits=raw_limits,
                    difference_limit=difference_limit,
                    gamma_limit=gamma_limit,
                    lme_limits=lme_limits,
                    source_frame=source_frame,
                )
                expert_canvas = base.copy()
                _draw_markers(
                    expert_canvas,
                    expert_by_frame.get(source_frame, []),
                    color=GREEN,
                    panels=4,
                    panel_size=(220, 220),
                    header=44,
                    source_shape=(512, 512),
                )
                expert_full.write(np.asarray(expert_canvas, dtype=np.uint8))
                model_canvas = base.copy()
                _draw_markers(
                    model_canvas,
                    block_candidates,
                    color=ORANGE,
                    panels=4,
                    panel_size=(220, 220),
                    header=44,
                    source_shape=(512, 512),
                    square=True,
                )
                model_full.write(np.asarray(model_canvas, dtype=np.uint8))

                for expert_id, intervals in expert_intervals.items():
                    if any(left <= source_frame <= right for left, right in intervals):
                        row = unique_experts[expert_id]
                        canvas = _close_panel(
                            stages,
                            titles=stage_titles,
                            raw_limits=raw_limits,
                            difference_limit=difference_limit,
                            gamma_limit=gamma_limit,
                            lme_limits=lme_limits,
                            source_frame=source_frame,
                            x_px=float(row["x_px"]),
                            y_px=float(row["y_px"]),
                            marker_color=GREEN,
                        )
                        expert_writers[expert_id].write(np.asarray(canvas, dtype=np.uint8))
                for anchor in active_anchors:
                    model_id = str(anchor["model_roi_id"])
                    canvas = _close_panel(
                        stages,
                        titles=stage_titles,
                        raw_limits=raw_limits,
                        difference_limit=difference_limit,
                        gamma_limit=gamma_limit,
                        lme_limits=lme_limits,
                        source_frame=source_frame,
                        x_px=float(anchor["x_px"]),
                        y_px=float(anchor["y_px"]),
                        marker_color=ORANGE,
                    )
                    model_writers[model_id].write(np.asarray(canvas, dtype=np.uint8))

        expert_full.close()
        model_full.close()
        for writer in expert_writers.values():
            writer.close()
        for writer in model_writers.values():
            writer.close()

        video_manifest_rows: list[dict[str, Any]] = [
            {
                "path": "1_Expert_Annotations/videos/expert_annotations_full_field.mp4",
                "section": "expert",
                "expected_marker": "green",
                "sample_encoded_frame": min(_expected_marker_frame_for_expert(row) for row in expert_occurrences),
                "expected_frames": expert_full.frames,
            },
            {
                "path": "2_Model_Annotations/videos/model_annotations_sequential_full_field.mp4",
                "section": "model",
                "expected_marker": "orange",
                "sample_encoded_frame": 0,
                "expected_frames": model_full.frames,
            },
        ]
        for expert_id, writer in expert_writers.items():
            video_manifest_rows.append(
                {
                    "path": writer.path.relative_to(work).as_posix(),
                    "section": "expert",
                    "expected_marker": "green",
                    "sample_encoded_frame": 0,
                    "expected_frames": writer.frames,
                    "source_segments_zero_inclusive": expert_intervals[expert_id],
                }
            )
        for anchor in selected_anchors:
            writer = model_writers[str(anchor["model_roi_id"])]
            video_manifest_rows.append(
                {
                    "path": writer.path.relative_to(work).as_posix(),
                    "section": "model",
                    "expected_marker": "orange",
                    "sample_encoded_frame": 0,
                    "expected_frames": writer.frames,
                    "source_blocks": sorted(set(map(int, anchor["block_ids"]))),
                }
            )
        _atomic_json(work / "video_manifest.json", {"schema_version": 1, "videos": video_manifest_rows})

        # Full-duration exact-pixel traces and per-ROI metadata.
        for expert_id, row in unique_experts.items():
            index = point_index[f"expert::{expert_id}"]
            spans = expert_intervals[expert_id]
            _trace_figure(
                expert_root / f"figures/traces/{_safe_id(expert_id)}.png",
                {key: value[index] for key, value in traces.items()},
                title=f"{expert_id} | exact pixel x={row['x_px']}, y={row['y_px']}",
                color="#46dc7d",
                spans=spans,
            )
            _atomic_json(
                expert_root / f"metadata/roi_{_safe_id(expert_id)}.json",
                {
                    "annotation_id": expert_id,
                    "roi_id": row["roi_id"],
                    "x_px": float(row["x_px"]),
                    "y_px": float(row["y_px"]),
                    "occurrence_intervals_zero_inclusive": spans,
                    "coordinate_convention": "x=column,y=row",
                },
            )
        for anchor in selected_anchors:
            model_id = str(anchor["model_roi_id"])
            index = point_index[f"anchor::{model_id}"]
            spans = [((block - 1) * 50, block * 50 - 1) for block in sorted(set(anchor["block_ids"]))]
            _trace_figure(
                model_root / f"figures/traces/{model_id}.png",
                {key: value[index] for key, value in traces.items()},
                title=f"{model_id} | exact pixel x={anchor['x_px']}, y={anchor['y_px']}",
                color="#ff9123",
                spans=spans,
            )
            _atomic_json(
                model_root / f"metadata/roi_{model_id}.json",
                {
                    **anchor,
                    "block_ids": sorted(set(map(int, anchor["block_ids"]))),
                    "coordinate_convention": "x=column,y=row",
                    "identity_interpretation": "deterministic audit review anchor, not a biological identity",
                },
            )

        _write_csv(expert_root / "expert_occurrences.csv", expert_occurrences)
        model_occurrence_rows = [
            {
                **row,
                "consolidated_model_roi_id": cluster_assignment[str(row["candidate_id"])],
                "selected_for_per_roi_media": cluster_assignment[str(row["candidate_id"])]
                in selected_cluster_ids,
            }
            for row in candidates
        ]
        _write_csv(model_root / "model_occurrences.csv", model_occurrence_rows)

        # One trace comparison and one metrics row per expert occurrence.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        comparison_rows: list[dict[str, Any]] = []
        for positive_id, occurrence in occurrence_by_id.items():
            pair = nearest_by_positive[positive_id]
            candidate = candidate_by_id[str(pair["nearest_time_eligible_candidate_id"])]
            expert_index = point_index[f"expert::{occurrence['annotation_id']}"]
            candidate_index = point_index[f"candidate::{candidate['candidate_id']}"]
            start = int(occurrence["start_frame_zero_inclusive"])
            stop = int(occurrence["stop_frame_zero_inclusive"])
            event_slice = slice(start, stop + 1)
            correlation, lag = _best_lag_correlation(
                traces["gamma"][expert_index, event_slice],
                traces["gamma"][candidate_index, event_slice],
            )
            matched = matched_by_positive.get(positive_id)
            row = {
                "occurrence_id": positive_id,
                "annotation_id": occurrence["annotation_id"],
                "nearest_candidate_id": candidate["candidate_id"],
                "distance_px": float(pair["nearest_spatial_distance_px"]),
                "candidate_rank": int(pair["nearest_candidate_block_rank"]),
                "candidate_score": float(pair["nearest_candidate_score"]),
                "event_gamma_correlation": "" if correlation is None else correlation,
                "best_lag_frames": "" if lag is None else lag,
                "one_to_one_match": matched is not None,
                "one_to_one_candidate_id": "" if matched is None else matched["candidate_id"],
                "nearest_identity_is_distinct_from_one_to_one_assignment": pair[
                    "nearest_identity_distinct_from_one_to_one_assignment"
                ],
                "unmatched_candidate_interpretation": "unknown_not_negative",
            }
            comparison_rows.append(row)
            frames = np.arange(start, stop + 1) + 1
            figure, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
            for axis, key, label in zip(
                axes,
                ["raw", "difference", "gamma"],
                ["Acquired raw", "Signed difference", "Signed Gamma-LS"],
                strict=True,
            ):
                axis.plot(frames, traces[key][expert_index, event_slice], color="#46dc7d", label="expert pixel")
                axis.plot(frames, traces[key][candidate_index, event_slice], color="#ff9123", linestyle="--", label="nearest frozen candidate")
                axis.set_ylabel(label)
                axis.grid(alpha=0.2)
                axis.legend(fontsize=7)
            axes[-1].set_xlabel("source frame (UI one-based)")
            figure.suptitle(
                f"{positive_id} | nearest distance {float(pair['nearest_spatial_distance_px']):.2f} px"
            )
            figure.savefig(
                comparison_root / f"trace_comparisons/{_safe_id(positive_id)}.png", dpi=110
            )
            plt.close(figure)
        _write_csv(comparison_root / "nearest_roi_trace_metrics.csv", comparison_rows)
        _write_csv(comparison_root / "expert_model_matches.csv", comparison_rows)

        raw_projection = np.load(metric / "raw_max_projection.npy", allow_pickle=False)
        lme_projection = np.max(block_maps, axis=0)

        def _spatial_figure(
            path: Path,
            raw_background: np.ndarray,
            evidence_background: np.ndarray,
            displayed_experts: Sequence[Mapping[str, Any]],
            displayed_candidates: Sequence[Mapping[str, Any]],
            displayed_matches: Sequence[Mapping[str, Any]],
            title: str,
        ) -> None:
            figure, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
            axes[0].imshow(raw_background, cmap="gray", vmin=raw_limits[0], vmax=raw_limits[1])
            axes[0].set_title(COMPARISON_PANELS[0])
            axes[1].imshow(evidence_background, cmap="gray", vmin=lme_limits[0], vmax=lme_limits[1])
            axes[1].set_title(COMPARISON_PANELS[1])
            for axis in axes:
                for expert_row in displayed_experts:
                    axis.add_patch(Circle((float(expert_row["x_px"]), float(expert_row["y_px"])), 6, fill=False, edgecolor="#46dc7d", linewidth=1.0))
            for candidate_row in displayed_candidates:
                axes[1].add_patch(Circle((float(candidate_row["x_px"]), float(candidate_row["y_px"])), 2.2, fill=False, edgecolor="#ff9123", linewidth=0.35, alpha=0.5))
            for match in displayed_matches:
                expert_row = occurrence_by_id[str(match["positive_id"])]
                candidate_row = candidate_by_id[str(match["candidate_id"])]
                axes[1].plot(
                    [float(expert_row["x_px"]), float(candidate_row["x_px"])],
                    [float(expert_row["y_px"]), float(candidate_row["y_px"])],
                    color=PALE_YELLOW,
                    linewidth=0.8,
                )
            for axis in axes:
                axis.set_axis_off()
            figure.suptitle(title)
            figure.savefig(path, dpi=130)
            plt.close(figure)

        _spatial_figure(
            comparison_root / "spatial_overview.png",
            raw_projection,
            lme_projection,
            list(unique_experts.values()),
            candidates,
            primary_matches,
            "15 right: green sparse positives, orange frozen B58 candidates, pale-yellow one-to-one links",
        )
        blocks_with_experts = sorted(
            {
                int(frame // 50) + 1
                for row in expert_occurrences
                for frame in range(
                    int(row["start_frame_zero_inclusive"]),
                    min(int(row["stop_frame_zero_inclusive"]), 1599) + 1,
                )
            }
        )
        for block_id in blocks_with_experts:
            start, stop = (block_id - 1) * 50, block_id * 50
            block_experts = [
                row for row in expert_occurrences
                if int(row["start_frame_zero_inclusive"]) < stop
                and int(row["stop_frame_zero_inclusive"]) >= start
            ]
            block_matches = [
                row for row in primary_matches
                if int(row["candidate_block_id"]) == block_id
            ]
            _spatial_figure(
                comparison_root / f"block_{block_id:02d}_comparison.png",
                np.max(np.asarray(movie[start:stop]), axis=0),
                np.asarray(block_maps[block_id - 1]),
                block_experts,
                candidates_by_block[block_id],
                block_matches,
                f"Block {block_id}: exact frozen comparison",
            )

        numeric = [row for row in comparison_rows if row["event_gamma_correlation"] != ""]
        recurrence = [len(set(map(int, row["block_ids"]))) for row in selected_anchors]
        figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        axes[0, 0].hist([float(row["distance_px"]) for row in comparison_rows], bins=12, color="#ff9123", edgecolor="#333333")
        axes[0, 0].axvline(6.0, color="#333333", linestyle="--")
        axes[0, 0].set(title="Nearest time-eligible distance", xlabel="distance (px)", ylabel="expert occurrences")
        axes[0, 1].scatter([float(row["distance_px"]) for row in numeric], [float(row["event_gamma_correlation"]) for row in numeric], color="#ff9123", s=18)
        axes[0, 1].set(title="Distance versus event similarity", xlabel="distance (px)", ylabel="best-lag Gamma correlation")
        axes[1, 0].scatter([int(row["candidate_rank"]) for row in comparison_rows], [float(row["distance_px"]) for row in comparison_rows], color="#ff9123", s=18)
        axes[1, 0].set(title="Rank versus distance", xlabel="candidate block rank", ylabel="distance (px)")
        axes[1, 1].hist(recurrence, bins=np.arange(0.5, max(recurrence) + 1.5), color="#ff9123", edgecolor="#333333")
        axes[1, 1].set(title="Selected audit-anchor recurrence", xlabel="blocks represented", ylabel="anchors")
        figure.savefig(comparison_root / "aggregate_diagnostics.png", dpi=130)
        plt.close(figure)

        _atomic_text(
            expert_root / "README.md",
            "# Expert Annotations\n\nGreen sparse-positive expert markers only. The video synchronizes acquired Raw, signed adjacent difference, signed radial Gamma-LS, and the exact sealed block-LME map. No model marker appears in this section.\n",
        )
        _atomic_text(
            model_root / "README.md",
            "# Model Annotations\n\nOrange frozen B58 model markers only. The full-field video shows all 1,856 primary candidate occurrences and the exact sequential stages. Per-ROI media use the repository's deterministic 24-anchor review-panel convention: label-free 3-pixel spatial consolidation, then recurrence, rank, score, and ID ordering. These anchors are review surrogates, not biological identities.\n",
        )
        _atomic_text(
            comparison_root / "README.md",
            "# Comparison\n\nFigures and tables only. Spatial figures contain exactly the two declared grayscale panels. Green is expert, orange is the frozen model, and pale yellow is reserved for one-to-one links. Nearest-candidate identity remains distinct from the one-to-one assignment. Unmatched candidates are unknown, not negative.\n",
        )

        summary = {
            "schema_version": 1,
            "status": "rendered_pending_visual_inspection",
            "source_metric_root": str(metric),
            "source_metric_artifact_index_sha256": _sha256(metric / "artifact_index.json"),
            "source_movie_sha256": contract["source"]["movie_sha256"],
            "recording_id": "15 right",
            "claim_scope": "one_recording_sparse_positive_confirmation_no_precision",
            "operating_point": {
                "representation": "difference_signed",
                "gamma_context_id": gamma["context_id"],
                "budget_per_block": 58,
                "nms_distance_px": 6,
                "blocks": 32,
                "scored_output_frames": 1599,
            },
            "expert_roi_count": len(unique_experts),
            "expert_occurrence_count": len(expert_occurrences),
            "model_occurrence_count": len(candidates),
            "model_identity_count_before_anchor_sampling": prelimit_cluster_count,
            "model_roi_count": len(selected_anchors),
            "model_anchor_limit": model_anchor_limit,
            "model_anchor_rule": "label_free_greedy_3px_spatial_consolidation_then_recurrence_rank_score_id_top24",
            "one_to_one_match_count": len(primary_matches),
            "known_positive_recall_at_primary_budget": len(primary_matches) / len(expert_occurrences),
            "precision_specificity_false_positive_rate": "not_identified",
            "unmatched_candidates": "unknown_not_negative",
            "stage_replay": replay_validation,
        }
        _atomic_json(work / "summary.json", summary)
        _atomic_json(
            work / "llm_context.json",
            {
                "schema_version": 1,
                "entrypoint": "summary.json",
                "annotation_separation": "strict",
                "comparison_spatial_panels": COMPARISON_PANELS,
                "model_stage_sequence": [
                    "Raw acquired frame",
                    "conditioned signed adjacent temporal difference",
                    "signed radial Gamma-LS",
                    "sealed per-block lme0.25 evidence",
                    "frozen B58 deterministic spatial NMS candidates",
                ],
                "coordinate_convention": "x=column,y=row",
                "frame_convention": "source zero-based in computation and UI one-based in media",
                "marker_semantics": {
                    "green": "expert sparse-positive ROI",
                    "orange": "frozen model candidate or audit anchor",
                    "pale_yellow": "primary one-to-one match link",
                },
                "display_scales": "display_scales.json",
                "expected_and_observed_counts": summary,
                "primary_tables": [
                    "1_Expert_Annotations/expert_occurrences.csv",
                    "2_Model_Annotations/model_occurrences.csv",
                    "3_Comparison/nearest_roi_trace_metrics.csv",
                    "3_Comparison/expert_model_matches.csv",
                ],
                "representative_artifacts": [
                    "1_Expert_Annotations/videos/expert_annotations_full_field.mp4",
                    "2_Model_Annotations/videos/model_annotations_sequential_full_field.mp4",
                    "3_Comparison/spatial_overview.png",
                    "3_Comparison/aggregate_diagnostics.png",
                    "projection_checks/comparison_projection_coordinate_check.png",
                ],
                "limitations": [
                    "single-recording sparse-positive sensitivity only",
                    "unmatched candidates are unknown and do not identify precision",
                    "the 24 model close-up identities are a deterministic audit review panel, not biological identities",
                    "Gamma stage media use numerically validated CPU FFT replay; sealed block-LME maps and candidates are reused unchanged",
                ],
            },
        )
        _atomic_text(
            work / "REPORT.md",
            "# Independent signed-difference Gamma-LS scientific audit\n\n"
            "This three-section packet binds the already frozen `15 right` B58 result to synchronized acquired-Raw, signed-difference, signed radial Gamma-LS, sealed block-LME, and deterministic-NMS views. It contains 10 expert ROIs, 51 expert occurrences, all 1,856 primary candidate occurrences in the model-only full field, and a deterministic 24-anchor label-free model review panel for close-ups and exact-pixel traces. Six of 51 sparse-positive occurrences have primary one-to-one matches. The packet does not identify precision, specificity, false-positive rate, biological identity, or population generalization.\n",
        )
        _atomic_json(work / "validation.json", {"status": "pending_visual_inspection"})
        _atomic_json(work / "artifact_index.json", {"status": "building"})
        inventory = require_three_section_scientific_audit(
            work,
            expected_expert_roi_count=len(unique_experts),
            expected_model_roi_count=len(selected_anchors),
            expected_expert_occurrence_count=len(expert_occurrences),
            expected_comparison_panels=COMPARISON_PANELS,
        )
        strict_checks = {
            "exact_10_expert_rois": len(unique_experts) == 10,
            "exact_51_expert_occurrences": len(expert_occurrences) == 51,
            "exact_1856_primary_candidate_occurrences": len(candidates) == 1856,
            "exact_one_nearest_trace_per_expert_occurrence": len(comparison_rows) == 51,
            "nearest_identity_distinct_from_one_to_one_assignment": all(
                "one_to_one_match" in row for row in comparison_rows
            ),
            "all_candidate_markers_in_model_full_field": model_full.frames == 1600,
            "no_comparison_videos": not list(comparison_root.rglob("*.mp4")),
            "projection_checks_copied": len(list((work / "projection_checks").glob("*.png"))) == 3,
            "cpu_stage_replay_parity": bool(replay_validation["pass"]),
            "source_movie_hash_verified": True,
            "metric_tables_unchanged": True,
        }
        if not all(strict_checks.values()):
            raise GammaScientificAuditError(f"strict inventory failed: {strict_checks}")
        _atomic_json(
            work / "inventory.json",
            {
                **inventory.to_dict(),
                "strict_checks": strict_checks,
                "model_identity_count_before_anchor_sampling": prelimit_cluster_count,
                "model_anchor_sampling_disclosed": True,
            },
        )
        _atomic_json(
            work / "status.json",
            {
                "status": "rendered_pending_visual_inspection",
                "scientific_audit_complete": False,
                "inventory_complete": inventory.complete,
            },
        )
        _atomic_json(work / "artifact_index.json", _artifact_index(work))
        work.replace(destination)
        return {
            "output": str(destination),
            "summary": summary,
            "inventory": inventory.to_dict(),
        }
    except BaseException:
        # Preserve the non-colliding partial output for diagnosis and resume.
        raise


def _probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,nb_frames",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise GammaScientificAuditError(result.stderr[-2000:])
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise GammaScientificAuditError(f"video stream count invalid: {path}")
    return streams[0]


def _decode_marker_frame(path: Path, frame_index: int, width: int, height: int) -> np.ndarray:
    expression = f"select=eq(n\\,{int(frame_index)})"
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-vf", expression,
            "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        capture_output=True,
    )
    if result.returncode:
        raise GammaScientificAuditError(result.stderr.decode(errors="replace")[-2000:])
    expected = width * height * 3
    if len(result.stdout) != expected:
        raise GammaScientificAuditError(f"encoded marker frame missing: {path}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3)


def _marker_counts(frame: np.ndarray) -> dict[str, int]:
    values = np.asarray(frame, dtype=np.int16)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    # yuv420 chroma subsampling substantially desaturates the thin green
    # full-field rings.  Requiring both green-vs-red and green-vs-blue margins
    # separates those rings from the complementary fringes around orange
    # markers, which can otherwise look weakly green after H.264 encoding.
    green_mask = (green > 170) & (green > red + 50) & (green > blue + 30)
    orange_mask = (red > 150) & (red > green + 35) & (green > blue + 20) & (blue < 150)
    return {"green_pixels": int(green_mask.sum()), "orange_pixels": int(orange_mask.sum())}


def validate_independent_scientific_audit(
    audit_root: str | Path,
    *,
    visual_passed: bool,
    inspection_note: str,
) -> dict[str, Any]:
    """Fully decode media, check encoded marker separation, and record visual QA."""

    root = Path(audit_root).expanduser().resolve()
    summary = _read_json(root / "summary.json")
    inventory = require_three_section_scientific_audit(
        root,
        expected_expert_roi_count=int(summary["expert_roi_count"]),
        expected_model_roi_count=int(summary["model_roi_count"]),
        expected_expert_occurrence_count=int(summary["expert_occurrence_count"]),
        expected_comparison_panels=COMPARISON_PANELS,
    )
    manifest = _read_json(root / "video_manifest.json")["videos"]
    video_results = []
    video_failures = []
    marker_failures = []
    for row in manifest:
        path = root / str(row["path"])
        probe = _probe_video(path)
        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
            capture_output=True,
        )
        if decoded.returncode:
            video_failures.append(str(row["path"]))
        observed_frames = int(probe.get("nb_frames") or 0)
        if observed_frames != int(row["expected_frames"]):
            video_failures.append(str(row["path"]))
        frame = _decode_marker_frame(
            path,
            int(row["sample_encoded_frame"]),
            int(probe["width"]),
            int(probe["height"]),
        )
        counts = _marker_counts(frame)
        if row["expected_marker"] == "green":
            marker_pass = counts["green_pixels"] > 0 and counts["orange_pixels"] == 0
        else:
            marker_pass = counts["orange_pixels"] > 0 and counts["green_pixels"] == 0
        if not marker_pass:
            marker_failures.append(str(row["path"]))
        video_results.append(
            {
                "path": row["path"],
                "codec": probe.get("codec_name"),
                "width": int(probe["width"]),
                "height": int(probe["height"]),
                "frames": observed_frames,
                "full_decode_pass": decoded.returncode == 0,
                "encoded_marker_counts": counts,
                "encoded_marker_separation_pass": marker_pass,
            }
        )
    image_failures = []
    image_count = 0
    for path in sorted(root.rglob("*.png")):
        image_count += 1
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            image_failures.append(path.relative_to(root).as_posix())
    passed = bool(
        visual_passed
        and inventory.complete
        and not video_failures
        and not marker_failures
        and not image_failures
    )
    validation = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "inventory_complete": inventory.complete,
        "video_count": len(manifest),
        "video_failures": sorted(set(video_failures)),
        "encoded_marker_separation_failures": marker_failures,
        "video_results": video_results,
        "png_count": image_count,
        "png_decode_failures": image_failures,
        "visual_inspection": {
            "performed": True,
            "passed": bool(visual_passed),
            "note": inspection_note,
            "checks": [
                "expert/model marker separation",
                "grayscale scientific backgrounds",
                "synchronized sequential stages",
                "fixed display scales and zero semantics",
                "projection and trace legibility",
            ],
        },
        "scientific_audit_complete": passed,
        "claim_boundary": summary.get(
            "claim_boundary_text",
            "one-recording sparse-positive sensitivity; no precision or population generalization",
        ),
    }
    finalized_status = "complete" if passed else "failed_media_validation"
    summary["status"] = finalized_status
    summary["scientific_audit_complete"] = passed
    summary["visual_inspection"] = {
        "performed": True,
        "passed": bool(visual_passed),
        "note": inspection_note,
    }
    llm_context = _read_json(root / "llm_context.json")
    llm_context["expected_and_observed_counts"] = summary
    llm_context["audit_finalization"] = {
        "status": finalized_status,
        "scientific_audit_complete": passed,
        "validation": "validation.json",
    }
    _atomic_json(root / "summary.json", summary)
    _atomic_json(root / "llm_context.json", llm_context)
    _atomic_json(root / "validation.json", validation)
    _atomic_json(
        root / "status.json",
        {
            "status": finalized_status,
            "scientific_audit_complete": passed,
            "inventory_complete": inventory.complete,
        },
    )
    _atomic_json(root / "artifact_index.json", _artifact_index(root))
    return validation


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def run_fixed_deployment_scientific_audit(
    metric_root: str | Path,
    config_path: str | Path,
    audit_root: str | Path,
    *,
    model_anchor_limit: int = DEFAULT_MODEL_ANCHOR_LIMIT,
) -> dict[str, Any]:
    """Render the frozen protected global-h15 post-selection characterization."""

    from neurobench.experiments.gamma_ls_difference.config import (
        GammaLSDifferenceConfig,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    metric = Path(metric_root).expanduser().resolve()
    destination = Path(audit_root).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not metric.is_dir() or not destination.parent.is_dir():
        raise FileNotFoundError("metric root and audit output parent must exist")
    metric_validation = _read_json(metric / "validation.json")
    if not metric_validation.get("all_checks_pass"):
        raise GammaScientificAuditError("fixed-deployment metric artifact is not validated")
    required_inputs = [
        "summary.json",
        "work_contract.json",
        "candidate_score_seal.json",
        "stage_array_manifest.json",
        "stage_arrays/conditioned_current_frame.npy",
        "stage_arrays/difference_signed.npy",
        "stage_arrays/gamma_a_train_b_test.npy",
        "stage_arrays/occupancy_maps.npy",
        "threshold_calibration.tsv",
        "occupancy_map_index.tsv",
        "audit_inputs/stage_sources.json",
        "audit_inputs/expert_occurrences.tsv",
        "audit_inputs/model_occurrences.tsv",
        "audit_inputs/one_to_one_matches.tsv",
    ]
    _verify_indexed_inputs(metric, required_inputs)
    summary_source = _read_json(metric / "summary.json")
    if summary_source.get("status") != (
        "complete_metric_and_audit_inputs_scientific_audit_pending"
    ):
        raise GammaScientificAuditError("fixed-deployment metric status changed")
    science = summary_source["science_contract"]
    if (
        science.get("representation") != "difference_signed"
        or science.get("context", {}).get("context_id")
        != "support_support_a_h15_g7_n9_m0p5"
        or science.get("evidence_role")
        != "post_selection_within_recording_characterization"
        or science.get("protected_selection_estimate") is not False
        or science.get("independent_confirmation") is not False
    ):
        raise GammaScientificAuditError("fixed-deployment claim contract changed")
    audit_state = science["audit_representative_state"]
    if (
        audit_state.get("quiet_swap") != "a_train_b_test"
        or float(audit_state.get("target_nms_peaks_per_pseudo_burst")) != 1.0
        or int(audit_state.get("candidate_budget_per_burst")) != 58
    ):
        raise GammaScientificAuditError("representative audit state changed")
    config = GammaLSDifferenceConfig.load(config_path)
    movie_path = config.source_paths["movie"]
    stage_sources = _read_json(metric / "audit_inputs/stage_sources.json")
    if not movie_path.is_file() or _sha256(movie_path) != stage_sources[
        "source_movie_sha256"
    ]:
        raise GammaScientificAuditError("source movie is missing or changed")
    movie = np.load(movie_path, mmap_mode="r", allow_pickle=False)
    if tuple(movie.shape) != (2359, 340, 573) or str(movie.dtype) != "uint16":
        raise GammaScientificAuditError("source movie shape/dtype changed")
    review_start_ui, review_stop_ui = map(int, science["review_interval_ui"])
    if (review_start_ui, review_stop_ui) != (1800, 2359):
        raise GammaScientificAuditError("review interval changed")
    frame_count = review_stop_ui - review_start_ui + 1
    raw_review = movie[review_start_ui - 1 : review_stop_ui]
    conditioned = np.load(
        metric / stage_sources["conditioned_current_frame"],
        mmap_mode="r",
        allow_pickle=False,
    )
    difference = np.load(
        metric / stage_sources["difference_signed"],
        mmap_mode="r",
        allow_pickle=False,
    )
    gamma = np.load(
        metric / stage_sources["gamma_score"], mmap_mode="r", allow_pickle=False
    )
    occupancy_stack = np.load(
        metric / stage_sources["occupancy_maps"],
        mmap_mode="r",
        allow_pickle=False,
    )
    for name, values in (
        ("conditioned", conditioned),
        ("difference", difference),
        ("gamma", gamma),
    ):
        if tuple(values.shape) != (frame_count, 340, 573) or str(values.dtype) != "float32":
            raise GammaScientificAuditError(f"{name} stage shape/dtype changed")
    if tuple(occupancy_stack.shape) != (2, 5, 4, 340, 573):
        raise GammaScientificAuditError("occupancy-map stack shape changed")

    occupancy_index = _read_tsv(metric / "occupancy_map_index.tsv")
    selected_index = [
        row
        for row in occupancy_index
        if row["quiet_swap"] == audit_state["quiet_swap"]
        and float(row["target_nms_peaks_per_pseudo_burst"]) == 1.0
    ]
    if len(selected_index) != 4:
        raise GammaScientificAuditError("representative occupancy maps are incomplete")
    occupancy_by_burst = {
        int(row["burst_id"]): np.asarray(
            occupancy_stack[
                int(row["occupancy_swap_index"]),
                int(row["occupancy_burden_index"]),
                int(row["occupancy_burst_index"]),
            ],
            dtype=np.float32,
        )
        for row in selected_index
    }
    calibration_rows = [
        row
        for row in _read_tsv(metric / "threshold_calibration.tsv")
        if row["quiet_swap"] == audit_state["quiet_swap"]
        and float(row["target_nms_peaks_per_pseudo_burst"]) == 1.0
        and int(row["nms_distance_px"]) == 6
        and row["nms_role"] == "primary"
    ]
    if len(calibration_rows) != 1:
        raise GammaScientificAuditError(
            "frozen q1 primary threshold calibration row is not unique"
        )
    representative_threshold_z = float(calibration_rows[0]["threshold_z"])
    if not math.isfinite(representative_threshold_z):
        raise GammaScientificAuditError("frozen q1 threshold is not finite")
    experts = _read_tsv(metric / "audit_inputs/expert_occurrences.tsv")
    candidates = _read_tsv(metric / "audit_inputs/model_occurrences.tsv")
    match_rows = _read_tsv(metric / "audit_inputs/one_to_one_matches.tsv")
    if len(experts) != 79 or len(candidates) != 9 or len(match_rows) != 79:
        raise GammaScientificAuditError("fixed expert/candidate/audit-match counts changed")
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise GammaScientificAuditError("candidate IDs are not unique")
    unique_experts: dict[str, dict[str, Any]] = {}
    for row in experts:
        unique_experts.setdefault(str(row["canonical_roi_id"]), dict(row))
    if len(unique_experts) != 26:
        raise GammaScientificAuditError("protected-v1 expert identity count changed")
    transformed_candidates = [
        {
            **row,
            "block_id": int(row["burst_id"]),
            "block_rank": int(row["candidate_rank"]),
            "pooled_lme_score": float(row["occupancy_score"]),
        }
        for row in candidates
    ]
    selected_anchors, cluster_assignment, prelimit_cluster_count = (
        _consolidate_model_anchors(
            transformed_candidates, limit=model_anchor_limit
        )
    )
    if not selected_anchors:
        raise GammaScientificAuditError("no model anchors were available")
    selected_cluster_ids = {
        str(row["model_roi_id"]) for row in selected_anchors
    }

    work = destination.parent / (
        f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    )
    work.mkdir()
    os.environ.setdefault("MPLCONFIGDIR", str(work / ".matplotlib"))
    expert_root = work / "1_Expert_Annotations"
    model_root = work / "2_Model_Annotations"
    comparison_root = work / "3_Comparison"
    for path in (
        expert_root / "videos/closeups",
        expert_root / "figures/traces",
        expert_root / "metadata",
        model_root / "videos/closeups",
        model_root / "figures/traces",
        model_root / "metadata",
        comparison_root / "trace_comparisons",
        work / "projection_checks",
        work / "paper_stage_crop",
    ):
        path.mkdir(parents=True, exist_ok=True)

    try:
        raw_sample = np.asarray(raw_review[::7, ::4, ::4], dtype=np.float32)
        conditioned_sample = np.asarray(conditioned[::7, ::4, ::4], dtype=np.float32)
        difference_sample = np.asarray(difference[::7, ::4, ::4], dtype=np.float32)
        gamma_sample = np.asarray(gamma[::7, ::4, ::4], dtype=np.float32)
        raw_limits = _limits_unsigned(raw_sample, 0.5, 99.8)
        conditioned_limits = _limits_unsigned(conditioned_sample, 0.5, 99.8)
        difference_limit = max(
            float(np.percentile(np.abs(difference_sample), 99.8)), 1e-6
        )
        gamma_limit = max(float(np.percentile(np.abs(gamma_sample), 99.8)), 1e-6)
        occupancy_limit = max(
            float(max(np.max(value) for value in occupancy_by_burst.values())),
            1e-6,
        )
        display_scales = {
            "raw": {"type": "fixed_linear", "vmin": raw_limits[0], "vmax": raw_limits[1]},
            "causal_conditioned": {
                "type": "fixed_linear",
                "vmin": conditioned_limits[0],
                "vmax": conditioned_limits[1],
            },
            "difference_signed": {
                "type": "fixed_symmetric_zero_midgray",
                "absolute_limit": difference_limit,
            },
            "signed_radial_gamma_ls": {
                "type": "fixed_symmetric_zero_midgray",
                "absolute_limit": gamma_limit,
            },
            "burst_threshold_occupancy": {
                "type": "fixed_zero_black",
                "vmin": 0.0,
                "vmax": occupancy_limit,
            },
            "scale_population": "label_free_review_subsample_and_sealed_occupancy_maps",
            "labels_used": False,
        }
        _atomic_json(work / "display_scales.json", display_scales)

        bursts = {
            int(key): tuple(map(int, value))
            for key, value in config.payload["frames"]["burst_intervals_ui"].items()
        }
        zero_occupancy = np.zeros((340, 573), dtype=np.float32)
        evidence_by_frame: list[np.ndarray] = [zero_occupancy] * frame_count
        expert_by_frame: dict[int, list[dict[str, Any]]] = {}
        candidate_by_frame: dict[int, list[dict[str, Any]]] = {}
        for burst_id, (start_ui, stop_ui) in bursts.items():
            for source_ui in range(start_ui, stop_ui + 1):
                review_index = source_ui - review_start_ui
                evidence_by_frame[review_index] = occupancy_by_burst[burst_id]
        for row in experts:
            for source_ui in range(
                int(row["source_start_ui"]), int(row["source_stop_ui"]) + 1
            ):
                expert_by_frame.setdefault(source_ui - review_start_ui, []).append(
                    dict(row)
                )
        for row in candidates:
            start_ui, stop_ui = bursts[int(row["burst_id"])]
            for source_ui in range(start_ui, stop_ui + 1):
                candidate_by_frame.setdefault(
                    source_ui - review_start_ui, []
                ).append(dict(row))

        expert_spans: dict[str, list[tuple[int, int]]] = {}
        for row in experts:
            expert_spans.setdefault(str(row["canonical_roi_id"]), []).append(
                (int(row["source_start_ui"]), int(row["source_stop_ui"]))
            )
        anchor_spans: dict[str, list[tuple[int, int]]] = {}
        for anchor in selected_anchors:
            anchor_spans[str(anchor["model_roi_id"])] = [
                bursts[burst_id]
                for burst_id in sorted(set(map(int, anchor["block_ids"])))
            ]

        point_rows: dict[str, dict[str, Any]] = {}
        for expert_id, row in unique_experts.items():
            point_rows[f"expert::{expert_id}"] = {
                "x_px": float(row["x_px"]), "y_px": float(row["y_px"])
            }
        for row in candidates:
            point_rows[f"candidate::{row['candidate_id']}"] = {
                "x_px": float(row["x_px"]), "y_px": float(row["y_px"])
            }
        for anchor in selected_anchors:
            point_rows[f"anchor::{anchor['model_roi_id']}"] = {
                "x_px": float(anchor["x_px"]), "y_px": float(anchor["y_px"])
            }
        point_ids = sorted(point_rows)
        point_index = {point_id: index for index, point_id in enumerate(point_ids)}
        point_x = np.asarray(
            [int(round(point_rows[key]["x_px"])) for key in point_ids], dtype=np.int64
        )
        point_y = np.asarray(
            [int(round(point_rows[key]["y_px"])) for key in point_ids], dtype=np.int64
        )
        traces = {
            key: np.zeros((len(point_ids), frame_count), dtype=np.float32)
            for key in ("raw", "conditioned", "difference", "gamma", "occupancy")
        }
        traces["raw"][:] = np.asarray(raw_review[:, point_y, point_x], dtype=np.float32).T
        traces["conditioned"][:] = np.asarray(
            conditioned[:, point_y, point_x], dtype=np.float32
        ).T
        traces["difference"][:] = np.asarray(
            difference[:, point_y, point_x], dtype=np.float32
        ).T
        traces["gamma"][:] = np.asarray(gamma[:, point_y, point_x], dtype=np.float32).T
        for review_index, evidence in enumerate(evidence_by_frame):
            traces["occupancy"][:, review_index] = evidence[point_y, point_x]

        stage_titles = [
            "Acquired raw",
            "Causal conditioned",
            "Signed difference",
            "Signed Gamma-LS",
            "Burst occupancy",
        ]
        expert_full = _VideoWriter(
            expert_root / "videos/expert_annotations_full_field.mp4",
            1000,
            244,
            50.0,
        )
        model_full = _VideoWriter(
            model_root / "videos/model_annotations_sequential_full_field.mp4",
            1000,
            244,
            50.0,
        )
        expert_writers = {
            expert_id: _VideoWriter(
                expert_root / f"videos/closeups/{_safe_id(expert_id)}.mp4",
                480,
                130,
                50.0,
            )
            for expert_id in unique_experts
        }
        model_writers = {
            str(anchor["model_roi_id"]): _VideoWriter(
                model_root / f"videos/closeups/{anchor['model_roi_id']}.mp4",
                480,
                130,
                50.0,
            )
            for anchor in selected_anchors
        }
        anchors_by_frame: dict[int, list[dict[str, Any]]] = {}
        for anchor in selected_anchors:
            for start_ui, stop_ui in anchor_spans[str(anchor["model_roi_id"])]:
                for source_ui in range(start_ui, stop_ui + 1):
                    anchors_by_frame.setdefault(
                        source_ui - review_start_ui, []
                    ).append(anchor)

        for review_index in range(frame_count):
            source_ui = review_start_ui + review_index
            stages = [
                np.asarray(raw_review[review_index]),
                np.asarray(conditioned[review_index]),
                np.asarray(difference[review_index]),
                np.asarray(gamma[review_index]),
                evidence_by_frame[review_index],
            ]
            base = _fixed_deployment_full_panel(
                stages,
                titles=stage_titles,
                raw_limits=raw_limits,
                conditioned_limits=conditioned_limits,
                difference_limit=difference_limit,
                gamma_limit=gamma_limit,
                occupancy_limit=occupancy_limit,
                source_frame_ui=source_ui,
            )
            expert_canvas = base.copy()
            _draw_markers(
                expert_canvas,
                expert_by_frame.get(review_index, []),
                color=GREEN,
                panels=5,
                panel_size=(200, 200),
                header=44,
                source_shape=(340, 573),
            )
            expert_full.write(np.asarray(expert_canvas, dtype=np.uint8))
            model_canvas = base.copy()
            _draw_markers(
                model_canvas,
                candidate_by_frame.get(review_index, []),
                color=ORANGE,
                panels=5,
                panel_size=(200, 200),
                header=44,
                source_shape=(340, 573),
                square=True,
            )
            model_full.write(np.asarray(model_canvas, dtype=np.uint8))
            for expert_id, spans in expert_spans.items():
                if any(start <= source_ui <= stop for start, stop in spans):
                    row = unique_experts[expert_id]
                    canvas = _fixed_deployment_close_panel(
                        stages,
                        titles=stage_titles,
                        raw_limits=raw_limits,
                        conditioned_limits=conditioned_limits,
                        difference_limit=difference_limit,
                        gamma_limit=gamma_limit,
                        occupancy_limit=occupancy_limit,
                        source_frame_ui=source_ui,
                        x_px=float(row["x_px"]),
                        y_px=float(row["y_px"]),
                        marker_color=GREEN,
                    )
                    expert_writers[expert_id].write(
                        np.asarray(canvas, dtype=np.uint8)
                    )
            for anchor in anchors_by_frame.get(review_index, []):
                model_id = str(anchor["model_roi_id"])
                canvas = _fixed_deployment_close_panel(
                    stages,
                    titles=stage_titles,
                    raw_limits=raw_limits,
                    conditioned_limits=conditioned_limits,
                    difference_limit=difference_limit,
                    gamma_limit=gamma_limit,
                    occupancy_limit=occupancy_limit,
                    source_frame_ui=source_ui,
                    x_px=float(anchor["x_px"]),
                    y_px=float(anchor["y_px"]),
                    marker_color=ORANGE,
                )
                model_writers[model_id].write(np.asarray(canvas, dtype=np.uint8))
        expert_full.close()
        model_full.close()
        for writer in expert_writers.values():
            writer.close()
        for writer in model_writers.values():
            writer.close()

        expert_sample = min(
            int(row["source_start_ui"]) - review_start_ui for row in experts
        )
        model_sample = min(
            bursts[int(row["burst_id"])][0] - review_start_ui for row in candidates
        )
        video_manifest_rows: list[dict[str, Any]] = [
            {
                "path": "1_Expert_Annotations/videos/expert_annotations_full_field.mp4",
                "section": "expert",
                "expected_marker": "green",
                "sample_encoded_frame": expert_sample,
                "expected_frames": expert_full.frames,
            },
            {
                "path": "2_Model_Annotations/videos/model_annotations_sequential_full_field.mp4",
                "section": "model",
                "expected_marker": "orange",
                "sample_encoded_frame": model_sample,
                "expected_frames": model_full.frames,
            },
        ]
        for expert_id, writer in expert_writers.items():
            video_manifest_rows.append(
                {
                    "path": writer.path.relative_to(work).as_posix(),
                    "section": "expert",
                    "expected_marker": "green",
                    "sample_encoded_frame": 0,
                    "expected_frames": writer.frames,
                    "source_segments_ui_inclusive": expert_spans[expert_id],
                }
            )
        for anchor in selected_anchors:
            model_id = str(anchor["model_roi_id"])
            writer = model_writers[model_id]
            video_manifest_rows.append(
                {
                    "path": writer.path.relative_to(work).as_posix(),
                    "section": "model",
                    "expected_marker": "orange",
                    "sample_encoded_frame": 0,
                    "expected_frames": writer.frames,
                    "source_segments_ui_inclusive": anchor_spans[model_id],
                }
            )
        _atomic_json(
            work / "video_manifest.json",
            {"schema_version": 1, "videos": video_manifest_rows},
        )

        for expert_id, row in unique_experts.items():
            index = point_index[f"expert::{expert_id}"]
            _fixed_deployment_trace_figure(
                expert_root / f"figures/traces/{_safe_id(expert_id)}.png",
                {key: values[index] for key, values in traces.items()},
                title=(
                    f"{expert_id} | exact pixel x={row['x_px']}, y={row['y_px']}"
                ),
                color="#46dc7d",
                spans=expert_spans[expert_id],
                review_start_ui=review_start_ui,
            )
            _atomic_json(
                expert_root / f"metadata/roi_{_safe_id(expert_id)}.json",
                {
                    "canonical_roi_id": expert_id,
                    "x_px": float(row["x_px"]),
                    "y_px": float(row["y_px"]),
                    "occurrence_intervals_ui_inclusive": expert_spans[expert_id],
                    "coordinate_convention": "x=column,y=row",
                    "temporal_extent_semantics": (
                        "configured burst window, not per-ROI onset"
                    ),
                },
            )
        for anchor in selected_anchors:
            model_id = str(anchor["model_roi_id"])
            index = point_index[f"anchor::{model_id}"]
            _fixed_deployment_trace_figure(
                model_root / f"figures/traces/{model_id}.png",
                {key: values[index] for key, values in traces.items()},
                title=(
                    f"{model_id} | exact pixel x={anchor['x_px']}, y={anchor['y_px']}"
                ),
                color="#ff9123",
                spans=anchor_spans[model_id],
                review_start_ui=review_start_ui,
            )
            _atomic_json(
                model_root / f"metadata/roi_{model_id}.json",
                {
                    **anchor,
                    "block_ids": sorted(set(map(int, anchor["block_ids"]))),
                    "coordinate_convention": "x=column,y=row",
                    "identity_interpretation": (
                        "deterministic audit review anchor, not biological identity"
                    ),
                },
            )
        _write_csv(expert_root / "expert_occurrences.csv", experts)
        model_occurrence_rows = [
            {
                **row,
                "consolidated_model_roi_id": cluster_assignment[row["candidate_id"]],
                "selected_for_per_roi_media": cluster_assignment[row["candidate_id"]]
                in selected_cluster_ids,
            }
            for row in candidates
        ]
        _write_csv(model_root / "model_occurrences.csv", model_occurrence_rows)

        candidate_by_id = {row["candidate_id"]: row for row in candidates}
        candidates_by_burst = {
            burst: [row for row in candidates if int(row["burst_id"]) == burst]
            for burst in (1, 2, 3, 4)
        }
        match_by_occurrence = {row["observation_id"]: row for row in match_rows}
        comparison_rows: list[dict[str, Any]] = []
        matched_links: list[dict[str, Any]] = []
        for occurrence in experts:
            occurrence_id = occurrence["observation_id"]
            burst_id = int(occurrence["burst_id"])
            eligible = candidates_by_burst[burst_id]
            nearest = min(
                eligible,
                key=lambda row: (
                    (float(row["x_px"]) - float(occurrence["x_px"])) ** 2
                    + (float(row["y_px"]) - float(occurrence["y_px"])) ** 2,
                    int(row["candidate_rank"]),
                    row["candidate_id"],
                ),
            )
            distance = math.hypot(
                float(nearest["x_px"]) - float(occurrence["x_px"]),
                float(nearest["y_px"]) - float(occurrence["y_px"]),
            )
            expert_index = point_index[
                f"expert::{occurrence['canonical_roi_id']}"
            ]
            candidate_index = point_index[f"candidate::{nearest['candidate_id']}"]
            start_ui = int(occurrence["source_start_ui"])
            stop_ui = int(occurrence["source_stop_ui"])
            event_slice = slice(
                start_ui - review_start_ui,
                stop_ui - review_start_ui + 1,
            )
            correlation, lag = _best_lag_correlation(
                traces["gamma"][expert_index, event_slice],
                traces["gamma"][candidate_index, event_slice],
            )
            assignment = match_by_occurrence[occurrence_id]
            one_to_one = _truthy(assignment["matched"])
            one_to_one_id = ""
            if one_to_one:
                matched_rank = int(assignment["matched_candidate_rank"])
                assigned = next(
                    row
                    for row in eligible
                    if int(row["candidate_rank"]) == matched_rank
                )
                one_to_one_id = assigned["candidate_id"]
                matched_links.append(
                    {
                        "occurrence_id": occurrence_id,
                        "candidate_id": one_to_one_id,
                    }
                )
            row = {
                "occurrence_id": occurrence_id,
                "canonical_roi_id": occurrence["canonical_roi_id"],
                "burst_id": burst_id,
                "nearest_candidate_id": nearest["candidate_id"],
                "distance_px": distance,
                "candidate_rank": int(nearest["candidate_rank"]),
                "candidate_score": float(nearest["occupancy_score"]),
                "event_gamma_correlation": "" if correlation is None else correlation,
                "best_lag_frames": "" if lag is None else lag,
                "one_to_one_match": one_to_one,
                "one_to_one_candidate_id": one_to_one_id,
                "nearest_identity_computed_separately_from_one_to_one_assignment": True,
                "nearest_equals_one_to_one_candidate": bool(
                    one_to_one and nearest["candidate_id"] == one_to_one_id
                ),
                "event_window_semantics": "configured_burst_window_not_per_roi_onset",
                "unmatched_candidate_interpretation": "unknown_not_negative",
            }
            comparison_rows.append(row)
            frames = np.arange(start_ui, stop_ui + 1)
            figure, axes = plt.subplots(
                3, 1, figsize=(8, 6), sharex=True, constrained_layout=True
            )
            for axis, key, label in zip(
                axes,
                ("raw", "difference", "gamma"),
                ("Acquired raw", "Signed difference", "Signed Gamma-LS"),
                strict=True,
            ):
                axis.plot(
                    frames,
                    traces[key][expert_index, event_slice],
                    color="#46dc7d",
                    label="expert pixel",
                )
                axis.plot(
                    frames,
                    traces[key][candidate_index, event_slice],
                    color="#ff9123",
                    linestyle="--",
                    label="nearest frozen candidate",
                )
                axis.set_ylabel(label)
                axis.grid(alpha=0.2)
                axis.legend(fontsize=7)
            axes[-1].set_xlabel("source frame (UI one-based)")
            figure.suptitle(
                f"{occurrence_id} | nearest distance {distance:.2f} px"
            )
            figure.savefig(
                comparison_root
                / f"trace_comparisons/{_safe_id(occurrence_id)}.png",
                dpi=110,
            )
            plt.close(figure)
        _write_csv(
            comparison_root / "nearest_roi_trace_metrics.csv", comparison_rows
        )
        _write_csv(
            comparison_root / "expert_model_matches.csv", comparison_rows
        )

        expert_by_occurrence = {row["observation_id"]: row for row in experts}

        def spatial_figure(
            path: Path,
            raw_background: np.ndarray,
            gamma_background: np.ndarray,
            displayed_experts: Sequence[Mapping[str, Any]],
            displayed_candidates: Sequence[Mapping[str, Any]],
            displayed_links: Sequence[Mapping[str, Any]],
            title: str,
        ) -> None:
            figure, axes = plt.subplots(
                1, 2, figsize=(12, 6), constrained_layout=True
            )
            axes[0].imshow(
                raw_background, cmap="gray", vmin=raw_limits[0], vmax=raw_limits[1]
            )
            axes[0].set_title(COMPARISON_PANELS[0])
            axes[1].imshow(
                gamma_background, cmap="gray", vmin=0.0, vmax=gamma_limit
            )
            axes[1].set_title(COMPARISON_PANELS[1])
            for axis in axes:
                for expert_row in displayed_experts:
                    axis.add_patch(
                        Circle(
                            (float(expert_row["x_px"]), float(expert_row["y_px"])),
                            6,
                            fill=False,
                            edgecolor="#46dc7d",
                            linewidth=1.0,
                        )
                    )
            for candidate_row in displayed_candidates:
                axes[1].add_patch(
                    Circle(
                        (float(candidate_row["x_px"]), float(candidate_row["y_px"])),
                        2.2,
                        fill=False,
                        edgecolor="#ff9123",
                        linewidth=0.8,
                    )
                )
            for link in displayed_links:
                expert_row = expert_by_occurrence[link["occurrence_id"]]
                candidate_row = candidate_by_id[link["candidate_id"]]
                axes[1].plot(
                    [float(expert_row["x_px"]), float(candidate_row["x_px"])],
                    [float(expert_row["y_px"]), float(candidate_row["y_px"])],
                    color=PALE_YELLOW,
                    linewidth=0.8,
                )
            for axis in axes:
                axis.set_axis_off()
            figure.suptitle(title)
            figure.savefig(path, dpi=130)
            plt.close(figure)

        raw_projection = np.max(np.asarray(raw_review), axis=0)
        gamma_projection = np.max(np.asarray(gamma), axis=0)
        spatial_figure(
            comparison_root / "spatial_overview.png",
            raw_projection,
            gamma_projection,
            experts,
            candidates,
            matched_links,
            (
                "Protected within-recording q1/B58: green expert, orange frozen "
                "candidate, pale-yellow one-to-one link"
            ),
        )
        for burst_id, (start_ui, stop_ui) in bursts.items():
            event_slice = slice(
                start_ui - review_start_ui, stop_ui - review_start_ui + 1
            )
            spatial_figure(
                comparison_root / f"burst_{burst_id}_comparison.png",
                np.max(np.asarray(raw_review[event_slice]), axis=0),
                np.max(np.asarray(gamma[event_slice]), axis=0),
                [row for row in experts if int(row["burst_id"]) == burst_id],
                candidates_by_burst[burst_id],
                [
                    row
                    for row in matched_links
                    if int(expert_by_occurrence[row["occurrence_id"]]["burst_id"])
                    == burst_id
                ],
                f"Burst {burst_id}: exact frozen q1/B58 comparison",
            )

        def projection_check(
            path: Path,
            *,
            show_experts: bool,
            show_candidates: bool,
            show_links: bool,
            title: str,
        ) -> None:
            figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
            axis.imshow(
                raw_projection, cmap="gray", vmin=raw_limits[0], vmax=raw_limits[1]
            )
            if show_experts:
                for row in experts:
                    axis.add_patch(
                        Circle(
                            (float(row["x_px"]), float(row["y_px"])),
                            5,
                            fill=False,
                            edgecolor="#46dc7d",
                            linewidth=0.8,
                        )
                    )
            if show_candidates:
                for row in candidates:
                    axis.add_patch(
                        Circle(
                            (float(row["x_px"]), float(row["y_px"])),
                            3,
                            fill=False,
                            edgecolor="#ff9123",
                            linewidth=0.8,
                        )
                    )
            if show_links:
                for link in matched_links:
                    expert_row = expert_by_occurrence[link["occurrence_id"]]
                    candidate_row = candidate_by_id[link["candidate_id"]]
                    axis.plot(
                        [float(expert_row["x_px"]), float(candidate_row["x_px"])],
                        [float(expert_row["y_px"]), float(candidate_row["y_px"])],
                        color=PALE_YELLOW,
                        linewidth=0.8,
                    )
            axis.set_title(title)
            axis.set_axis_off()
            figure.savefig(path, dpi=130)
            plt.close(figure)

        projection_check(
            work / "projection_checks/expert_projection_coordinate_check.png",
            show_experts=True,
            show_candidates=False,
            show_links=False,
            title="Expert-only coordinate projection (green)",
        )
        projection_check(
            work / "projection_checks/model_projection_coordinate_check.png",
            show_experts=False,
            show_candidates=True,
            show_links=False,
            title="Model-only coordinate projection (orange)",
        )
        projection_check(
            work / "projection_checks/comparison_projection_coordinate_check.png",
            show_experts=True,
            show_candidates=True,
            show_links=True,
            title="Comparison projection with one-to-one links",
        )

        numeric = [
            row for row in comparison_rows if row["event_gamma_correlation"] != ""
        ]
        recurrence = [
            len(set(map(int, row["block_ids"]))) for row in selected_anchors
        ]
        figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        axes[0, 0].hist(
            [float(row["distance_px"]) for row in comparison_rows],
            bins=12,
            color="#ff9123",
            edgecolor="#333333",
        )
        axes[0, 0].axvline(6.0, color="#333333", linestyle="--")
        axes[0, 0].set(
            title="Nearest same-burst distance",
            xlabel="distance (px)",
            ylabel="expert occurrences",
        )
        axes[0, 1].scatter(
            [float(row["distance_px"]) for row in numeric],
            [float(row["event_gamma_correlation"]) for row in numeric],
            color="#ff9123",
            s=18,
        )
        axes[0, 1].set(
            title="Distance versus burst-window similarity",
            xlabel="distance (px)",
            ylabel="best-lag Gamma correlation",
        )
        axes[1, 0].scatter(
            [int(row["candidate_rank"]) for row in comparison_rows],
            [float(row["distance_px"]) for row in comparison_rows],
            color="#ff9123",
            s=18,
        )
        axes[1, 0].set(
            title="Rank versus distance",
            xlabel="candidate burst rank",
            ylabel="distance (px)",
        )
        axes[1, 1].hist(
            recurrence,
            bins=np.arange(0.5, max(recurrence) + 1.5),
            color="#ff9123",
            edgecolor="#333333",
        )
        axes[1, 1].set(
            title="Model-anchor burst recurrence",
            xlabel="bursts represented",
            ylabel="anchors",
        )
        figure.savefig(comparison_root / "aggregate_diagnostics.png", dpi=130)
        plt.close(figure)

        fig3_candidate = min(
            candidates,
            key=lambda row: (
                int(row["burst_id"]),
                int(row["candidate_rank"]),
                row["candidate_id"],
            ),
        )
        fig3_metadata = _write_fixed_stage_crop_npz(
            work / "paper_stage_crop/fig3_stage_crop.npz",
            work / "paper_stage_crop/fig3_stage_crop.json",
            work / "paper_stage_crop/fig3_stage_crop_preview.png",
            raw_movie=movie,
            conditioned=conditioned,
            difference=difference,
            gamma=gamma,
            occupancy=occupancy_by_burst[int(fig3_candidate["burst_id"])],
            candidate=fig3_candidate,
            review_start_ui=review_start_ui,
            source_movie_sha256=stage_sources["source_movie_sha256"],
            threshold_z=representative_threshold_z,
            nms_distance_px=6,
        )

        _atomic_text(
            expert_root / "README.md",
            "# Expert Annotations\n\nGreen protected-v1 sparse-positive markers only. "
            "Each ROI close-up uses configured burst windows because the sparse-positive "
            "table does not declare per-ROI onset/offset. No model marker appears here.\n",
        )
        _atomic_text(
            model_root / "README.md",
            "# Model Annotations\n\nOrange candidates from the predeclared q1, "
            "a-train/b-test, B58 display state only. All nine candidate occurrences "
            "are shown. Three-pixel label-free consolidation defines the per-ROI review "
            "surrogates; these are not biological identities.\n",
        )
        _atomic_text(
            comparison_root / "README.md",
            "# Comparison\n\nFigures and tables only. Spatial figures contain exactly "
            "the two declared grayscale panels. Each protected-v1 occurrence is compared "
            "with the nearest candidate from the same configured burst; this identity is "
            "computed separately from one-to-one assignment. Unmatched candidates are unknown.\n",
        )
        one_to_one_count = sum(_truthy(row["matched"]) for row in match_rows)
        summary = {
            "schema_version": 1,
            "status": "rendered_pending_visual_inspection",
            "source_metric_root": str(metric),
            "source_metric_artifact_index_sha256": _sha256(
                metric / "artifact_index.json"
            ),
            "source_movie_sha256": stage_sources["source_movie_sha256"],
            "claim_scope": "post_selection_within_recording_characterization",
            "claim_boundary_text": (
                "single-recording post-selection within-recording characterization; "
                "not protected selection, independent confirmation, precision, or population generalization"
            ),
            "operating_point": audit_state,
            "expert_roi_count": len(unique_experts),
            "expert_occurrence_count": len(experts),
            "model_occurrence_count": len(candidates),
            "model_identity_count_before_anchor_sampling": prelimit_cluster_count,
            "model_roi_count": len(selected_anchors),
            "model_anchor_limit": model_anchor_limit,
            "model_anchor_rule": (
                "label_free_greedy_3px_spatial_consolidation_then_recurrence_rank_score_id"
            ),
            "one_to_one_match_count": one_to_one_count,
            "known_positive_recall_at_representative_state": one_to_one_count
            / len(experts),
            "precision_specificity_false_positive_rate": "not_identified",
            "unmatched_candidates": "unknown_not_negative",
            "stage_arrays": "exact_hash_sealed_production_replay",
            "fig3_stage_crop": fig3_metadata,
            "fig3_proposal_plane_scope": (
                "derived_same_frame_nms6_using_frozen_protected_q1_pseudo_burst_threshold"
            ),
            "fig3_primary_protected_output_plane": "burst_occupancy",
            "fig3_full_record_operational_head_reproduced": False,
            "full_record_q1_371_over_2259_model_media_covered": False,
            "renderer_implementation": {
                "path": "repo://neurobench/experiments/gamma_ls_difference/scientific_audit.py",
                "sha256": _sha256(Path(__file__).resolve()),
                "source_metric_preflight_is_older_than_renderer": True,
                "metric_scores_or_selection_changed": False,
            },
        }
        _atomic_json(work / "summary.json", summary)
        _atomic_json(
            work / "llm_context.json",
            {
                "schema_version": 1,
                "entrypoint": "summary.json",
                "annotation_separation": "strict",
                "comparison_spatial_panels": COMPARISON_PANELS,
                "model_stage_sequence": stage_titles,
                "coordinate_convention": "x=column,y=row",
                "frame_convention": "source UI one-based inclusive; review arrays zero-based",
                "marker_semantics": {
                    "green": "protected-v1 sparse-positive ROI",
                    "orange": "frozen q1/B58 model candidate or audit anchor",
                    "pale_yellow": "primary one-to-one match link",
                },
                "display_scales": "display_scales.json",
                "expected_and_observed_counts": summary,
                "primary_tables": [
                    "1_Expert_Annotations/expert_occurrences.csv",
                    "2_Model_Annotations/model_occurrences.csv",
                    "3_Comparison/nearest_roi_trace_metrics.csv",
                    "3_Comparison/expert_model_matches.csv",
                ],
                "representative_artifacts": [
                    "1_Expert_Annotations/videos/expert_annotations_full_field.mp4",
                    "2_Model_Annotations/videos/model_annotations_sequential_full_field.mp4",
                    "3_Comparison/spatial_overview.png",
                    "3_Comparison/aggregate_diagnostics.png",
                    "paper_stage_crop/fig3_stage_crop.npz",
                    "paper_stage_crop/fig3_stage_crop_preview.png",
                ],
                "limitations": [
                    "post-selection within-recording characterization only",
                    "configured burst windows are used as occurrence temporal extents",
                    "unmatched candidates are unknown and do not identify precision",
                    "model anchors are deterministic review surrogates, not biological identities",
                    "Fig3 framewise proposals are a derived same-frame diagnostic under the protected pseudo-burst-calibrated threshold, not the full-record operational head",
                    "full-record 371/2259 operational model-only media remain a separate gap",
                ],
            },
        )
        _atomic_text(
            work / "REPORT.md",
            "# Fixed global-h15 protected within-recording scientific audit\n\n"
            f"This three-section packet renders 26 expert ROIs, 79 sparse-positive "
            f"occurrences, all {len(candidates)} candidates at the predeclared q1/B58 "
            f"display state, and {len(selected_anchors)} deterministic model review "
            f"anchors. {one_to_one_count} of 79 occurrences have one-to-one matches. "
            "The synchronized five-stage views use exact hash-sealed replay arrays. "
            "The Fig. 3 packet keeps protected burst occupancy separate from a derived "
            "same-frame NMS6 diagnostic using the frozen protected q1 threshold; the "
            "latter is not the separately calibrated full-record operational head. "
            "This is post-selection within-recording characterization, not a protected "
            "selection estimate or independent confirmation, and it does not identify "
            "precision. The full-record 371/2259 operational proposal stream remains "
            "outside this media audit.\n",
        )
        _atomic_json(work / "validation.json", {"status": "pending_visual_inspection"})
        _atomic_json(work / "artifact_index.json", {"status": "building"})
        inventory = require_three_section_scientific_audit(
            work,
            expected_expert_roi_count=len(unique_experts),
            expected_model_roi_count=len(selected_anchors),
            expected_expert_occurrence_count=len(experts),
            expected_comparison_panels=COMPARISON_PANELS,
        )
        strict_checks = {
            "exact_26_expert_rois": len(unique_experts) == 26,
            "exact_79_expert_occurrences": len(experts) == 79,
            "exact_9_representative_model_occurrences": len(candidates) == 9,
            "exact_6_one_to_one_matches": one_to_one_count == 6,
            "exact_one_nearest_trace_per_expert_occurrence": len(comparison_rows)
            == 79,
            "nearest_identity_distinct_from_one_to_one_assignment": all(
                row[
                    "nearest_identity_computed_separately_from_one_to_one_assignment"
                ]
                is True
                for row in comparison_rows
            ),
            "all_candidate_markers_in_model_full_field": model_full.frames == frame_count,
            "expert_full_field_complete": expert_full.frames == frame_count,
            "no_comparison_videos": not list(comparison_root.rglob("*.mp4")),
            "projection_checks_present": len(
                list((work / "projection_checks").glob("*.png"))
            )
            == 3,
            "fig3_npz_keys_exact": set(
                np.load(work / "paper_stage_crop/fig3_stage_crop.npz").files
            )
            == {
                "raw",
                "conditioned_previous",
                "conditioned_current",
                "difference",
                "gamma_ls",
                "threshold_exceedance",
                "proposals",
                "burst_occupancy",
            },
            "fig3_preview_present": (
                work / "paper_stage_crop/fig3_stage_crop_preview.png"
            ).is_file(),
            "fig3_framewise_and_burst_heads_separated": (
                fig3_metadata["framewise_diagnostic"]["nms_distance_px"] == 6
                and fig3_metadata["framewise_diagnostic"][
                    "threshold_calibration_unit"
                ]
                == "nms_peaks_per_duration_matched_pseudo_burst"
                and fig3_metadata["burst_occupancy_semantics"].startswith(
                    "sealed fraction"
                )
            ),
            "source_movie_hash_verified": True,
            "metric_tables_unchanged": True,
            "full_record_gap_disclosed": True,
        }
        if not all(strict_checks.values()):
            raise GammaScientificAuditError(
                f"fixed-deployment strict inventory failed: {strict_checks}"
            )
        _atomic_json(
            work / "inventory.json",
            {**inventory.to_dict(), "strict_checks": strict_checks},
        )
        _atomic_json(
            work / "status.json",
            {
                "status": "rendered_pending_visual_inspection",
                "scientific_audit_complete": False,
                "inventory_complete": inventory.complete,
            },
        )
        _atomic_json(work / "artifact_index.json", _artifact_index(work))
        work.replace(destination)
        return {
            "output": str(destination),
            "summary": summary,
            "inventory": inventory.to_dict(),
        }
    except BaseException:
        raise


def write_protected_h15_audit_readiness(
    protected_metric_root: str | Path,
    preflight_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Write a fail-closed audit gap packet for the protected h15 request."""

    metric = Path(protected_metric_root).expanduser().resolve()
    preflight = Path(preflight_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    validation = _read_json(metric / "validation.json")
    if not validation.get("all_checks_pass"):
        raise GammaScientificAuditError("protected metric artifact is not validated")
    candidates = _read_tsv(metric / "candidates_label_sealed.tsv")
    observations = _read_tsv(metric / "protected_v1_observation_matches.tsv")
    exact_context = "support_support_a_h15_g7_n9_m0p5"
    exact_rows = [
        row for row in candidates
        if row["representation"] == "difference_signed"
        and row["context_id"] == exact_context
        and int(row["nms_distance_px"]) == 6
    ]
    covered_bursts = sorted({int(row["burst_id"]) for row in exact_rows})
    protected_observations = {
        str(row["observation_id"]): row
        for row in observations
        if row["cohort"] == "protected_v1"
    }
    expert_identities = {str(row["canonical_roi_id"]) for row in protected_observations.values()}
    expected_stage_arrays = [
        "difference_signed.npy",
        "signed_radial_gamma_ls.npy",
        "pooled_detection_maps.npy",
    ]
    present_stage_arrays = [name for name in expected_stage_arrays if (metric / name).is_file()]
    blockers = [
        {
            "id": "missing_exact_h15_burst_2_candidate_stream",
            "severity": "critical",
            "evidence": {
                "exact_context": exact_context,
                "covered_heldout_bursts": covered_bursts,
                "missing_heldout_bursts": sorted(set([1, 2, 3, 4]) - set(covered_bursts)),
            },
            "impact": "A four-burst protected model-only and matched-comparison audit cannot be bound to the deployment h15/mode-0.5 pipeline.",
            "remediation": "Run one new label-isolated protected candidate execution for the already frozen global context; do not select or retune it on protected labels.",
        },
        {
            "id": "per_frame_sequential_stage_arrays_not_persisted",
            "severity": "high",
            "evidence": {
                "required": expected_stage_arrays,
                "present": present_stage_arrays,
            },
            "impact": "The exact signed-difference, signed Gamma-LS, and pooled-map stages cannot be rendered or traced from the metric artifact alone.",
            "remediation": "Replay the frozen pipeline and verify score/candidate parity before rendering; preserve the arrays or deterministic per-coordinate traces.",
        },
        {
            "id": "no_single_frozen_quiet_burden_for_visual_operating_point",
            "severity": "high",
            "evidence": {
                "available_quiet_burdens": sorted({float(row["target_nms_peaks_per_pseudo_burst"]) for row in exact_rows}),
                "quiet_swaps": sorted({row["quiet_swap"] for row in exact_rows}),
            },
            "impact": "Choosing one candidate overlay now would be a post-result operating-point decision; overlaying their union would not represent one detector state.",
            "remediation": "Freeze one display/evaluation operating point from the existing curve without consulting coordinates, or explicitly audit every curve point as distinct model states.",
        },
        {
            "id": "candidate_peak_frame_and_gamma_trace_absent",
            "severity": "high",
            "evidence": {
                "candidate_columns": list(candidates[0]) if candidates else [],
            },
            "impact": "Per-occurrence event correlation, best lag, peak-frame close-ups, and exact nearest-candidate Gamma traces are not recoverable from the sealed rows.",
            "remediation": "During frozen replay, persist source-frame indices and exact candidate-coordinate stage traces without changing candidate membership.",
        },
    ]
    work = destination.parent / f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    work.mkdir()
    try:
        for section, text in (
            ("1_Expert_Annotations", "Expert media are not emitted in isolation because the requested three-section packet cannot yet bind a complete exact-h15 model stream."),
            ("2_Model_Annotations", "Not renderable from the protected metric artifact: exact h15/mode-0.5 burst-2 candidates and sequential stage arrays are absent."),
            ("3_Comparison", "Not renderable without inventing a post-result operating point or recomputing the missing exact-h15 candidate stream."),
        ):
            _atomic_text(work / section / "README.md", f"# {section.replace('_', ' ')}\n\n{text}\n")
        projection = preflight / "label_projection_overlay.png"
        if not projection.is_file():
            raise GammaScientificAuditError("preflight label-projection overlay is missing")
        shutil.copy2(projection, work / "label_projection_overlay.png")
        summary = {
            "schema_version": 1,
            "status": "blocked_fail_closed",
            "scientific_audit_complete": False,
            "requested_pipeline": {
                "representation": "difference_signed",
                "gamma_context_id": exact_context,
                "support_width_px": 31,
                "guard_radius_px": 7,
                "shape_n": 9,
                "mode_radius_px": 7.5,
                "nms_distance_px": 6,
            },
            "protected_population": {
                "expert_roi_count": len(expert_identities),
                "expert_occurrence_count": len(protected_observations),
                "bursts": [1, 2, 3, 4],
            },
            "exact_h15_candidate_rows": len(exact_rows),
            "exact_h15_covered_bursts": covered_bursts,
            "blockers": blockers,
            "claim_boundary": "This is an audit-readiness finding, not scientific-audit completion and not a new metric result.",
        }
        _atomic_json(work / "summary.json", summary)
        _atomic_json(
            work / "llm_context.json",
            {
                "schema_version": 1,
                "entrypoint": "summary.json",
                "requested_stage_sequence": [
                    "Raw acquired frame",
                    "conditioned signed adjacent temporal difference",
                    "signed radial Gamma-LS h15/g7/n9/mode7.5",
                    "frozen CFAR operating point",
                    "deterministic NMS6 candidates",
                ],
                "annotation_separation": "planned_strict_but_not_renderable",
                "projection_overlay": "label_projection_overlay.png",
                "blockers": blockers,
            },
        )
        _atomic_text(
            work / "REPORT.md",
            "# Protected signed-difference h15 scientific-audit readiness\n\n"
            "The requested exact deployment-pipeline audit cannot be completed from the current protected artifact without creating new candidate evidence. The sealed table contains h15/mode-0.5 signed-difference rows for held-out bursts 1, 3, and 4, but not burst 2; burst 2 used the separate h15/mode-0.75 fold context. The artifact also omits per-frame signed-difference and Gamma-LS arrays, pooled maps, candidate peak frames, and a single frozen quiet-burden display operating point. The existing projection overlay is preserved here. This packet fails closed and does not reinterpret the 79 sparse-positive occurrences or select a new detector.\n",
        )
        _atomic_json(
            work / "validation.json",
            {
                "schema_version": 1,
                "status": "failed_expected_missing_inputs",
                "all_checks_pass": False,
                "scientific_audit_complete": False,
                "projection_overlay_present": True,
                "blocker_ids": [row["id"] for row in blockers],
            },
        )
        _atomic_json(
            work / "status.json",
            {
                "status": "blocked_fail_closed",
                "scientific_audit_complete": False,
                "new_metric_or_model_selection_performed": False,
            },
        )
        _atomic_json(work / "artifact_index.json", _artifact_index(work))
        work.replace(destination)
        return {"output": str(destination), "summary": summary}
    except BaseException:
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    independent = subparsers.add_parser("independent", help="render independent three-section audit")
    independent.add_argument("--metric-root", required=True)
    independent.add_argument("--output-root", required=True)
    independent.add_argument("--model-anchor-limit", type=int, default=DEFAULT_MODEL_ANCHOR_LIMIT)
    independent.add_argument("--cpu-threads", type=int, default=4)
    validate = subparsers.add_parser("validate-independent", help="validate rendered independent media")
    validate.add_argument("--audit-root", required=True)
    validate.add_argument("--visual-passed", action="store_true")
    validate.add_argument("--inspection-note", required=True)
    fixed = subparsers.add_parser(
        "fixed-deployment", help="render fixed protected three-section audit"
    )
    fixed.add_argument("--metric-root", required=True)
    fixed.add_argument("--config", required=True)
    fixed.add_argument("--output-root", required=True)
    fixed.add_argument("--model-anchor-limit", type=int, default=DEFAULT_MODEL_ANCHOR_LIMIT)
    validate_fixed = subparsers.add_parser(
        "validate-fixed-deployment", help="validate rendered fixed protected media"
    )
    validate_fixed.add_argument("--audit-root", required=True)
    validate_fixed.add_argument("--visual-passed", action="store_true")
    validate_fixed.add_argument("--inspection-note", required=True)
    protected = subparsers.add_parser("protected-readiness", help="write fail-closed protected h15 gap packet")
    protected.add_argument("--metric-root", required=True)
    protected.add_argument("--preflight-root", required=True)
    protected.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "independent":
        result = run_independent_scientific_audit(
            arguments.metric_root,
            arguments.output_root,
            model_anchor_limit=arguments.model_anchor_limit,
            cpu_threads=arguments.cpu_threads,
        )
    elif arguments.command == "validate-independent":
        result = validate_independent_scientific_audit(
            arguments.audit_root,
            visual_passed=arguments.visual_passed,
            inspection_note=arguments.inspection_note,
        )
    elif arguments.command == "fixed-deployment":
        result = run_fixed_deployment_scientific_audit(
            arguments.metric_root,
            arguments.config,
            arguments.output_root,
            model_anchor_limit=arguments.model_anchor_limit,
        )
    elif arguments.command == "validate-fixed-deployment":
        result = validate_independent_scientific_audit(
            arguments.audit_root,
            visual_passed=arguments.visual_passed,
            inspection_note=arguments.inspection_note,
        )
    else:
        result = write_protected_h15_audit_readiness(
            arguments.metric_root, arguments.preflight_root, arguments.output_root
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
