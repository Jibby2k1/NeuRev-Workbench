#!/usr/bin/env python3
"""Render raw-intensity histograms for the reduced 15-right recording.

The script is intentionally read-only with respect to the frozen source movie and
annotation authority. It streams the 512x512 movie in chunks and writes a new,
non-colliding diagnostic artifact directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


DATA_ROOT = Path(os.environ.get("NEUROBENCH_DATA_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_MOVIE = DATA_ROOT / (
    "Outputs/GridModel/060126_crop512_grid32_v1/cropped_videos/15 right.tif"
)
DEFAULT_ANNOTATIONS = DATA_ROOT / (
    "Outputs/GridModel/060126_crop512_grid128_max_v1/annotations/"
    "manual_roi_spikes_v1/manual_roi_spike_annotations.json"
)
DEFAULT_OUTPUT = Path(
    "Outputs/ReducedDatasetHistograms/15_right_crop512_raw_intensity_v1"
)

INK = "#17202A"
BLUE = "#2C6EAA"
BLUE_LIGHT = "#D9EAF5"
ORANGE = "#D9772A"
PINK = "#B83B73"
GRID = "#DDE3E8"
MIN_FRAME_BIN_PIXELS_FOR_SHARE = 64
MIN_VIDEO_BIN_PIXELS_FOR_SHARE = 1000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rois(path: Path, video_id: str) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload["annotations"] if row["video_id"] == video_id]
    rows.sort(key=lambda row: int(row["roi_id"]))
    if not rows:
        raise ValueError(f"no annotations found for {video_id!r}")
    return rows


def neighborhood_indices(
    y: int, x: int, *, radius: int, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = (xx * xx + yy * yy <= radius * radius) & ~((xx == 0) & (yy == 0))
    rows, columns = np.nonzero(mask)
    rows = rows + y - radius
    columns = columns + x - radius
    inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
    return rows[inside], columns[inside]


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#AAB4BC")
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)


def save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie", type=Path, default=DEFAULT_MOVIE)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--video-id", default="15 right")
    parser.add_argument("--neighborhood-radius", type=int, default=6)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--chunk-frames", type=int, default=16)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)

    movie = tifffile.memmap(args.movie, mode="r")
    if movie.ndim != 3 or movie.dtype != np.uint16:
        raise ValueError(f"expected uint16 TYX movie, received {movie.shape} {movie.dtype}")
    frames, height, width = movie.shape
    rois = load_rois(args.annotations, args.video_id)

    centers: list[tuple[int, int]] = []
    neighborhoods: list[tuple[np.ndarray, np.ndarray]] = []
    for roi in rois:
        x, y = int(round(float(roi["crop_x"]))), int(round(float(roi["crop_y"])))
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"ROI {roi['roi_id']} center is outside the movie")
        centers.append((y, x))
        neighborhoods.append(
            neighborhood_indices(
                y, x, radius=args.neighborhood_radius, height=height, width=width
            )
        )

    # Union masks make the contribution decomposition mutually exclusive even if
    # future ROI neighborhoods overlap. Centers take precedence over neighborhoods.
    center_mask = np.zeros((height, width), dtype=bool)
    neighborhood_mask = np.zeros((height, width), dtype=bool)
    for (y, x), (rows, columns) in zip(centers, neighborhoods, strict=True):
        center_mask[y, x] = True
        neighborhood_mask[rows, columns] = True
    neighborhood_mask[center_mask] = False
    center_rows, center_columns = np.nonzero(center_mask)
    neighborhood_rows, neighborhood_columns = np.nonzero(neighborhood_mask)

    # One exact pass establishes a shared raw-intensity domain.
    value_min, value_max = np.iinfo(movie.dtype).max, np.iinfo(movie.dtype).min
    for start in range(0, frames, args.chunk_frames):
        block = np.asarray(movie[start : start + args.chunk_frames])
        value_min = min(value_min, int(block.min()))
        value_max = max(value_max, int(block.max()))
    edges = np.linspace(value_min, value_max + 1, args.bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2.0

    center_counts = np.zeros((len(rois), args.bins), dtype=np.int64)
    neighborhood_counts = np.zeros_like(center_counts)
    frame_counts = np.zeros((frames, args.bins), dtype=np.int64)
    frame_center_counts = np.zeros_like(frame_counts)
    frame_neighborhood_counts = np.zeros_like(frame_counts)
    whole_counts = np.zeros(args.bins, dtype=np.int64)

    for start in range(0, frames, args.chunk_frames):
        block = np.asarray(movie[start : start + args.chunk_frames])
        stop = start + len(block)
        whole_counts += np.histogram(block, bins=edges)[0]
        for local_index, frame in enumerate(block):
            frame_counts[start + local_index] = np.histogram(frame, bins=edges)[0]
            frame_center_counts[start + local_index] = np.histogram(
                frame[center_rows, center_columns], bins=edges
            )[0]
            frame_neighborhood_counts[start + local_index] = np.histogram(
                frame[neighborhood_rows, neighborhood_columns], bins=edges
            )[0]
        for roi_index, ((y, x), (rows, columns)) in enumerate(
            zip(centers, neighborhoods, strict=True)
        ):
            center_counts[roi_index] += np.histogram(block[:, y, x], bins=edges)[0]
            neighborhood_counts[roi_index] += np.histogram(
                block[:, rows, columns], bins=edges
            )[0]

    # Detailed per-ROI center histograms.
    figure, axes = plt.subplots(5, 2, figsize=(12, 15), sharex=True)
    for axis, roi, counts in zip(axes.flat, rois, center_counts, strict=True):
        axis.fill_between(mids, counts, step="mid", color=BLUE_LIGHT)
        axis.plot(mids, counts, color=BLUE, linewidth=1.2)
        axis.set_title(f"ROI {roi['roi_id']}  center ({roi['crop_x']:.0f}, {roi['crop_y']:.0f})", loc="left")
        axis.set_ylabel("Frames")
        style_axis(axis)
    axes[-1, 0].set_xlabel("Raw intensity (uint16)")
    axes[-1, 1].set_xlabel("Raw intensity (uint16)")
    figure.suptitle("Per-ROI center-pixel intensity histograms — 15 right (reduced 512×512)", color=INK, fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    save_figure(figure, args.output / "01_per_roi_center_histograms")

    # Detailed per-ROI neighborhood histograms.
    figure, axes = plt.subplots(5, 2, figsize=(12, 15), sharex=True)
    for axis, roi, counts, (rows, _) in zip(
        axes.flat, rois, neighborhood_counts, neighborhoods, strict=True
    ):
        axis.fill_between(mids, counts, step="mid", color=BLUE_LIGHT)
        axis.plot(mids, counts, color=BLUE, linewidth=1.2)
        axis.set_title(
            f"ROI {roi['roi_id']}  radius {args.neighborhood_radius}px, center excluded (n={len(rows)} pixels)",
            loc="left",
        )
        axis.set_ylabel("Pixel-frames")
        style_axis(axis)
    axes[-1, 0].set_xlabel("Raw intensity (uint16)")
    axes[-1, 1].set_xlabel("Raw intensity (uint16)")
    figure.suptitle("Per-ROI neighborhood intensity histograms — 15 right (reduced 512×512)", color=INK, fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    save_figure(figure, args.output / "02_per_roi_neighborhood_histograms")

    # A per-frame histogram is represented as a distribution heatmap to retain all frames.
    frame_percent = frame_counts / frame_counts.sum(axis=1, keepdims=True) * 100.0
    positive = frame_percent[frame_percent > 0]
    vmax = float(np.percentile(positive, 99.5))
    figure, axis = plt.subplots(figsize=(14, 6.5))
    image = axis.imshow(
        frame_percent.T,
        origin="lower",
        aspect="auto",
        extent=(0, frames - 1, edges[0], edges[-1]),
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )
    axis.set_title("Per-frame raw-intensity histograms — each frame normalized to 100%", loc="left")
    axis.set_xlabel("Frame index (zero-based)")
    axis.set_ylabel("Raw intensity (uint16)")
    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("Pixels in intensity bin (%)")
    figure.tight_layout()
    save_figure(figure, args.output / "03_per_frame_histogram_heatmap")

    # Entire-video histogram.
    figure, axis = plt.subplots(figsize=(12, 6.5))
    axis.fill_between(mids, whole_counts, step="mid", color=BLUE_LIGHT)
    axis.plot(mids, whole_counts, color=BLUE, linewidth=1.4)
    axis.set_yscale("log")
    axis.set_title("Entire-video raw-intensity histogram — 15 right (reduced 512×512)", loc="left")
    axis.set_xlabel("Raw intensity (uint16)")
    axis.set_ylabel("Pixel-frames (log scale)")
    style_axis(axis)
    figure.tight_layout()
    save_figure(figure, args.output / "04_entire_video_histogram")

    # Contribution-aware per-frame view. The numerator is the number of pixels
    # from the named ROI source in a frame/bin; the denominator is all pixels in
    # that same frame/bin. Empty bins remain zero rather than becoming NaN.
    frame_center_share = np.divide(
        frame_center_counts,
        frame_counts,
        out=np.zeros_like(frame_counts, dtype=np.float64),
        where=frame_counts > 0,
    ) * 100.0
    frame_neighborhood_share = np.divide(
        frame_neighborhood_counts,
        frame_counts,
        out=np.zeros_like(frame_counts, dtype=np.float64),
        where=frame_counts > 0,
    ) * 100.0
    frame_center_share[frame_counts < MIN_FRAME_BIN_PIXELS_FOR_SHARE] = np.nan
    frame_neighborhood_share[frame_counts < MIN_FRAME_BIN_PIXELS_FOR_SHARE] = np.nan
    figure, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True, sharey=True)
    panels = (
        (frame_percent, "All pixels: per-frame intensity distribution", "Blues", vmax, "Pixels in frame (%)"),
        (
            frame_neighborhood_share,
            f"ROI neighborhoods: contribution to supported frame/intensity bins (union radius {args.neighborhood_radius}px)",
            "Oranges",
            float(np.nanpercentile(frame_neighborhood_share[frame_neighborhood_share > 0], 99.5)),
            "Bin supplied by neighborhoods (%)",
        ),
        (
            frame_center_share,
            "Exact ROI centers: contribution to supported frame/intensity bins",
            "RdPu",
            float(np.nanpercentile(frame_center_share[frame_center_share > 0], 99.5)),
            "Bin supplied by centers (%)",
        ),
    )
    for axis, (values, title, cmap, panel_vmax, label) in zip(axes, panels, strict=True):
        image = axis.imshow(
            values.T,
            origin="lower",
            aspect="auto",
            extent=(0, frames - 1, edges[0], edges[-1]),
            cmap=cmap,
            vmin=0,
            vmax=panel_vmax,
            interpolation="nearest",
        )
        axis.set_title(title, loc="left", fontsize=11)
        axis.set_ylabel("Raw intensity")
        colorbar = figure.colorbar(image, ax=axis, pad=0.012)
        colorbar.set_label(label)
    axes[-1].set_xlabel("Frame index (zero-based)")
    axes[-1].text(
        0,
        -0.28,
        f"Contribution panels mask frame/intensity bins containing fewer than {MIN_FRAME_BIN_PIXELS_FOR_SHARE} total pixels.",
        transform=axes[-1].transAxes,
        color="#5F6B75",
        fontsize=9,
    )
    figure.suptitle("Per-frame histogram contribution from known-ROI pixels", color=INK, fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    save_figure(figure, args.output / "05_per_frame_roi_contribution")

    # Full-video contribution view: actual counts remain on a shared log axis;
    # the lower panel exposes each source's percentage of the whole-video bin.
    whole_center_counts = frame_center_counts.sum(axis=0)
    whole_neighborhood_counts = frame_neighborhood_counts.sum(axis=0)
    whole_center_share = np.divide(
        whole_center_counts,
        whole_counts,
        out=np.zeros_like(whole_counts, dtype=np.float64),
        where=whole_counts > 0,
    ) * 100.0
    whole_neighborhood_share = np.divide(
        whole_neighborhood_counts,
        whole_counts,
        out=np.zeros_like(whole_counts, dtype=np.float64),
        where=whole_counts > 0,
    ) * 100.0
    whole_center_share[whole_counts < MIN_VIDEO_BIN_PIXELS_FOR_SHARE] = np.nan
    whole_neighborhood_share[whole_counts < MIN_VIDEO_BIN_PIXELS_FOR_SHARE] = np.nan
    figure, (axis_count, axis_share) = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    axis_count.plot(mids, whole_counts, color="#6B7680", linewidth=1.4, label="All pixels")
    axis_count.plot(mids, whole_neighborhood_counts, color=ORANGE, linewidth=1.5, label="ROI neighborhoods")
    axis_count.plot(mids, whole_center_counts, color=PINK, linewidth=1.5, label="Exact ROI centers")
    axis_count.set_yscale("log")
    axis_count.set_ylabel("Pixel-frames (log scale)")
    axis_count.set_title("Actual contribution to the entire-video histogram", loc="left")
    axis_count.legend(frameon=False)
    style_axis(axis_count)
    axis_share.plot(mids, whole_neighborhood_share, color=ORANGE, linewidth=1.5, label="ROI neighborhoods")
    axis_share.plot(mids, whole_center_share, color=PINK, linewidth=1.5, label="Exact ROI centers")
    axis_share.axhline(
        neighborhood_mask.mean() * 100.0,
        color=ORANGE,
        linewidth=1.0,
        linestyle="--",
        alpha=0.65,
        label="Neighborhood pixel share",
    )
    axis_share.axhline(
        center_mask.mean() * 100.0,
        color=PINK,
        linewidth=1.0,
        linestyle="--",
        alpha=0.65,
        label="Center pixel share",
    )
    axis_share.set_xlabel("Raw intensity (uint16)")
    axis_share.set_ylabel("Contribution to bin (%)")
    axis_share.set_title("Fraction of each intensity bin supplied by known-ROI pixels", loc="left")
    axis_share.legend(frameon=False, ncol=2, fontsize=9)
    style_axis(axis_share)
    axis_share.text(
        0,
        -0.32,
        f"Percentage curves mask intensity bins containing fewer than {MIN_VIDEO_BIN_PIXELS_FOR_SHARE:,} total pixel-frames; raw counts above remain complete.",
        transform=axis_share.transAxes,
        color="#5F6B75",
        fontsize=9,
    )
    figure.tight_layout()
    save_figure(figure, args.output / "06_entire_video_roi_contribution")

    with (args.output / "histogram_counts.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scope", "roi_id", "bin_left", "bin_right", "count"])
        for index in range(args.bins):
            writer.writerow(["entire_video", "", edges[index], edges[index + 1], whole_counts[index]])
            writer.writerow(["all_roi_centers", "", edges[index], edges[index + 1], whole_center_counts[index]])
            writer.writerow(["all_roi_neighborhoods", "", edges[index], edges[index + 1], whole_neighborhood_counts[index]])
        for roi_index, roi in enumerate(rois):
            for scope, values in (
                ("roi_center", center_counts[roi_index]),
                ("roi_neighborhood", neighborhood_counts[roi_index]),
            ):
                for index in range(args.bins):
                    writer.writerow([scope, roi["roi_id"], edges[index], edges[index + 1], values[index]])

    metadata = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "descriptive raw-intensity histogram audit; no model evaluation",
        "video_id": args.video_id,
        "movie": {"path": str(args.movie), "sha256": sha256(args.movie)},
        "annotations": {"path": str(args.annotations), "sha256": sha256(args.annotations)},
        "shape_tyx": list(movie.shape),
        "dtype": str(movie.dtype),
        "frame_rate_hz": 50.0,
        "roi_count": len(rois),
        "roi_centers_xy": [
            {"roi_id": roi["roi_id"], "x": roi["crop_x"], "y": roi["crop_y"]}
            for roi in rois
        ],
        "neighborhood": {
            "geometry": "closed Euclidean disk with exact center pixel excluded",
            "radius_px": args.neighborhood_radius,
            "union_pixel_count": int(neighborhood_mask.sum()),
            "union_fraction_of_frame": float(neighborhood_mask.mean()),
        },
        "center_union_pixel_count": int(center_mask.sum()),
        "center_union_fraction_of_frame": float(center_mask.mean()),
        "histogram": {
            "bin_count": args.bins,
            "shared_raw_intensity_min": value_min,
            "shared_raw_intensity_max": value_max,
            "shared_edges": edges.tolist(),
            "per_frame_normalization": "each frame sums to 100 percent",
            "whole_video_y_scale": "log10 display; counts remain untransformed in CSV",
            "contribution_definition": "ROI source count divided by all-pixel count within the same intensity bin",
            "contribution_partition": "exact centers, union of radius-6 neighborhoods excluding every center, and remainder are mutually exclusive",
            "minimum_frame_bin_pixels_for_contribution_display": MIN_FRAME_BIN_PIXELS_FOR_SHARE,
            "minimum_video_bin_pixels_for_contribution_display": MIN_VIDEO_BIN_PIXELS_FOR_SHARE,
        },
        "coordinate_convention": "x=column, y=row; frame indices are zero-based",
        "claim_limit": "descriptive distributions only",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
