"""Selected-alpha-only fixed-scale diagnostic videos."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_figures import (
    _AtomicMp4Writer,
)

from .artifacts import atomic_json


def _gray(values: np.ndarray, bounds: tuple[float, float], *, signed: bool) -> Image.Image:
    low, high = bounds
    normalized = np.clip((np.asarray(values, dtype=np.float32) - low) / (high - low), 0, 1)
    if signed:
        normalized = np.clip(normalized, 0, 1)
    pixels = np.rint(normalized * 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="L").convert("RGB").resize(
        (300, 178), Image.Resampling.BILINEAR
    )


def _bounds(values: np.ndarray, *, signed: bool) -> tuple[float, float]:
    sample = np.asarray(values[::4, ::4, ::4], dtype=np.float32)
    if signed:
        limit = max(float(np.quantile(np.abs(sample), 0.995)), 1e-6)
        return (-limit, limit)
    low, high = np.quantile(sample, [0.005, 0.995])
    return (float(low), max(float(high), float(low) + 1e-6))


def _auto_alpha(rows: Sequence[dict[str, Any]]) -> float | None:
    candidates = [
        row
        for row in rows
        if row["lane"] == "roi_balanced"
        and row["whitening_mode"] == "natural_fixed"
        and row["alpha"] > 0
        and not row["weight_concentration_warning"]
    ]
    if not candidates:
        return None
    alphas = sorted({float(row["alpha"]) for row in candidates})
    return max(
        alphas,
        key=lambda alpha: float(
            np.median(
                [
                    abs(row["angle_shift_from_alpha0_degrees"])
                    for row in candidates
                    if row["alpha"] == alpha
                ]
            )
        ),
    )


def render_selected_videos(
    root: Path,
    filtered: np.ndarray,
    labels: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    requested: Sequence[float | str],
    *,
    review_start_ui: int,
    fps: float = 10.0,
) -> dict[str, Any]:
    selected: list[float] = []
    auto = _auto_alpha(rows)
    for value in requested:
        if isinstance(value, str):
            if value != "auto_best_stable":
                raise ValueError(f"unknown selected video alpha {value}")
            if auto is not None:
                selected.append(auto)
        else:
            selected.append(float(value))
    selected = list(dict.fromkeys(selected))
    target = root / "videos"
    target.mkdir(exist_ok=True)
    filtered_bounds = _bounds(filtered, signed=False)
    derivative = np.zeros_like(filtered, dtype=np.float32)
    derivative[1:] = filtered[1:] - filtered[:-1]
    derivative_bounds = _bounds(derivative, signed=True)
    rendered = []
    fold = min(int(row["fold_id"]) for row in rows)
    for alpha in selected:
        matches = [
            row
            for row in rows
            if int(row["fold_id"]) == fold
            and row["lane"] in {"roi_balanced", "natural"}
            and row["whitening_mode"] == "natural_fixed"
            and float(row["alpha"]) == alpha
        ]
        if not matches:
            continue
        row = sorted(matches, key=lambda item: item["lane"] != "roi_balanced")[0]
        direction = np.asarray(row["effective_innovation_direction"], dtype=np.float32)
        mean = np.asarray(row["whitening"]["mean"], dtype=np.float32)
        weighted = np.zeros_like(filtered, dtype=np.float32)
        weighted[1:] = (
            direction[0] * (filtered[:-1] - mean[0])
            + direction[1] * (filtered[1:] - mean[1])
        )
        weighted_bounds = _bounds(weighted, signed=True)
        path = target / f"fold_{fold}_roi_balanced_alpha_{alpha:.3f}.mp4"
        writer = _AtomicMp4Writer(path, (300, 960), fps)
        try:
            for index in range(len(filtered)):
                canvas = Image.new("RGB", (960, 300), (14, 14, 14))
                draw = ImageDraw.Draw(canvas)
                ui_frame = review_start_ui + index
                draw.text(
                    (12, 8),
                    f"Event-balanced CS-Parzen | fold {fold} | alpha={alpha:.3f} | UI frame {ui_frame}",
                    fill="white",
                    font=ImageFont.load_default(),
                )
                panels = (
                    ("Causal input P_t", filtered[index], filtered_bounds, False),
                    ("Fixed derivative", derivative[index], derivative_bounds, True),
                    ("Weighted innovation", weighted[index], weighted_bounds, True),
                )
                for panel_index, (title, values, bounds, signed) in enumerate(panels):
                    left = 10 + 320 * panel_index
                    canvas.paste(_gray(values, bounds, signed=signed), (left, 55))
                    draw.text((left, 38), title, fill=(220, 220, 220), font=ImageFont.load_default())
                    for label in labels:
                        if int(label["start_frame_ui"]) <= ui_frame <= int(label["end_frame_ui"]):
                            x = left + int(round(float(label["x_px"]) / filtered.shape[2] * 300))
                            y = 55 + int(round(float(label["y_px"]) / filtered.shape[1] * 178))
                            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline="white", width=1)
                draw.text(
                    (12, 248),
                    "Signed panels use fixed bounds; mid-gray is zero. Rings are sparse known positives; unmarked pixels are unknown.",
                    fill=(190, 190, 190),
                    font=ImageFont.load_default(),
                )
                draw.text(
                    (12, 267),
                    f"ESS fraction={row['weight_ess_fraction']:.3f}; angle shift={row['angle_shift_from_alpha0_degrees']:+.3f} deg",
                    fill=(190, 190, 190),
                    font=ImageFont.load_default(),
                )
                writer.write(np.asarray(canvas, dtype=np.uint8))
            writer.close()
        except Exception:
            writer.abort()
            raise
        rendered.append(
            {
                "path": str(path.relative_to(root)),
                "fold_id": fold,
                "lane": row["lane"],
                "alpha": alpha,
                "fps": fps,
                "frame_count": len(filtered),
                "fixed_display_bounds": {
                    "filtered": list(filtered_bounds),
                    "derivative": list(derivative_bounds),
                    "weighted_innovation": list(weighted_bounds),
                },
                "weight_ess_fraction": row["weight_ess_fraction"],
                "angle_degrees": row["angle_degrees"],
            }
        )
    payload = {
        "policy": "selected_only",
        "requested": list(requested),
        "auto_best_stable": auto,
        "rendered": rendered,
    }
    atomic_json(root / "selected_video_manifest.json", payload)
    return payload
