#!/usr/bin/env python3
"""Build RGB detection-overlay TIFFs and missed-neuron reports.

The experiment has sparse positive labels.  Therefore the four rendered states
are known true positive, known false negative, unmatched candidate (unknown),
and unscored background.  The latter two must not be interpreted as false
positive and true negative without additional exhaustive annotations.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import shutil
import time
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile

from neurobench.experiments.frame_difference import _atomic_json
from neurobench.experiments.learnable_contrast import core as v1
from neurobench.experiments.representation_benchmark.config import (
    RepresentationBenchmarkConfig,
)
from neurobench.metrics.sparse_detection import match_peaks_one_to_one


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "Outputs/RepresentationBenchmark/spon_ca_burst_representation_benchmark_v1"
BEST_VISUALS = (
    ROOT
    / "Outputs/RepresentationBenchmark/spon_ca_burst_representation_benchmark_v1_best_visuals"
)
DEFAULT_CONFIG = ROOT / "examples/spon_ca_burst_representation_benchmark.example.json"

TP_COLOR = (48, 235, 104)
FN_COLOR = (255, 64, 216)
UNKNOWN_COLOR = (255, 166, 48)
BACKGROUND_TINT = (0.78, 0.86, 1.0)


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    title: str
    lane: str
    source_kind: str
    source_path: Path


METHODS = (
    MethodSpec(
        "raw_direct", "Raw Direct", "raw_direct", "raw",
        Path(""),
    ),
    MethodSpec(
        "pca_amplitude_rank8", "PCA amplitude rank 8 component evidence",
        "pca_amplitude_rank8_components", "tiff",
        BEST_VISUALS / "representative_tiffs/pca_amplitude_rank8_component_evidence.tif",
    ),
    MethodSpec(
        "pca_residual_rank64", "PCA quiet-residual rank 64 reconstruction",
        "pca_quiet_residual_rank64_reconstruction", "tiff",
        PRIMARY / "representative_tiffs/pca_residual_reconstruction.tif",
    ),
    MethodSpec(
        "ica_residual_rank64", "Spatial FastICA quiet-residual rank 64 seed 7",
        "spatial_fastica_quiet_residual_rank64_seed7", "tiff",
        PRIMARY / "representative_tiffs/ica_residual_component_evidence.tif",
    ),
    MethodSpec(
        "linear_ae_residual_rank64", "Linear autoencoder residual rank 64 seed 7",
        "linear_autoencoder_quiet_residual_rank64_seed7_reconstruction", "tiff",
        PRIMARY / "representative_tiffs/linear_autoencoder_residual_reconstruction.tif",
    ),
    MethodSpec(
        "nonlinear_ae_residual_rank64", "Nonlinear autoencoder residual rank 64 seed 7",
        "nonlinear_autoencoder_quiet_residual_rank64_seed7_reconstruction", "tiff",
        PRIMARY / "representative_tiffs/nonlinear_autoencoder_residual_reconstruction.tif",
    ),
)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _observation_id(row: dict[str, Any]) -> str:
    return f"burst_{int(row['burst_id'])}:{row['roi_identity']}"


def classify_burst(
    peaks: list[tuple[float, int, int]],
    labels: list[dict[str, Any]],
    radius: float = 6.0,
) -> dict[str, Any]:
    """Return explicit sparse-positive classification for one burst."""
    ranked = sorted(peaks, key=lambda peak: peak[0], reverse=True)
    matches, matched_peak_indices = match_peaks_one_to_one(ranked, labels, radius)
    matched_label_indices = {int(item[0]) for item in matches}
    match_by_label = {
        int(label_index): {
            "score": float(score), "candidate_x_px": int(x), "candidate_y_px": int(y),
            "distance_px": float(distance),
        }
        for label_index, score, x, y, distance in matches
    }
    return {
        "true_positive_labels": [
            {**labels[index], **match_by_label[index]}
            for index in sorted(matched_label_indices)
        ],
        "false_negative_labels": [
            labels[index] for index in range(len(labels)) if index not in matched_label_indices
        ],
        "unmatched_candidates_unknown": [
            {"score": float(score), "x_px": int(x), "y_px": int(y)}
            for index, (score, x, y) in enumerate(ranked)
            if index not in matched_peak_indices
        ],
        "matched": len(matches),
        "labels": len(labels),
        "candidates": len(ranked),
    }


def _build_classifications(
    labels: list[dict[str, Any]],
    candidates: list[dict[str, str]],
) -> tuple[dict[str, dict[int, dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    labels_by_burst = {
        burst: [row for row in labels if int(row["burst_id"]) == burst]
        for burst in sorted({int(row["burst_id"]) for row in labels})
    }
    candidate_lookup: dict[tuple[str, int], list[tuple[float, int, int]]] = defaultdict(list)
    for row in candidates:
        candidate_lookup[(row["lane"], int(row["burst_id"]))].append(
            (float(row["score"]), int(row["x_px"]), int(row["y_px"]))
        )
    by_method: dict[str, dict[int, dict[str, Any]]] = {}
    missed_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    for spec in METHODS:
        by_method[spec.method_id] = {}
        for burst, burst_labels in labels_by_burst.items():
            classified = classify_burst(
                candidate_lookup[(spec.lane, burst)], burst_labels, radius=6.0
            )
            by_method[spec.method_id][burst] = classified
            method_rows.append({
                "method_id": spec.method_id, "title": spec.title, "lane": spec.lane,
                "burst_id": burst, "known_tp": classified["matched"],
                "known_fn": len(classified["false_negative_labels"]),
                "unknown_candidates_not_fp": len(classified["unmatched_candidates_unknown"]),
                "candidates": classified["candidates"],
                "known_recall": classified["matched"] / max(1, classified["labels"]),
                "tn_available": False, "fp_available": False,
            })
            for row in classified["false_negative_labels"]:
                missed_rows.append({
                    "method_id": spec.method_id, "title": spec.title, "lane": spec.lane,
                    "observation_id": _observation_id(row), "burst_id": burst,
                    "roi_identity": row["roi_identity"], "x_px": row["x_px"],
                    "y_px": row["y_px"], "start_frame_ui": row["start_frame_ui"],
                    "end_frame_ui": row["end_frame_ui"],
                    "recurrence_count": row["recurrence_count"],
                    "classification": "known_false_negative",
                })
    return by_method, missed_rows, method_rows


def _aggregate_missed(
    labels: list[dict[str, Any]],
    missed_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    method_ids = [spec.method_id for spec in METHODS]
    missed_by_method = {
        method_id: {
            row["observation_id"] for row in missed_rows if row["method_id"] == method_id
        }
        for method_id in method_ids
    }
    observation_rows = []
    for label in labels:
        observation = _observation_id(label)
        missed = [method for method in method_ids if observation in missed_by_method[method]]
        observation_rows.append({
            "observation_id": observation, "burst_id": int(label["burst_id"]),
            "roi_identity": label["roi_identity"], "x_px": label["x_px"],
            "y_px": label["y_px"], "missed_method_count": len(missed),
            "method_count": len(method_ids), "missed_by_all_methods": len(missed) == len(method_ids),
            "missed_methods": ",".join(missed),
        })
    identity_rows = []
    identities = sorted({str(row["roi_identity"]) for row in labels})
    for identity in identities:
        observations = {_observation_id(row) for row in labels if row["roi_identity"] == identity}
        method_status = {
            method: len(observations & missed_by_method[method]) for method in method_ids
        }
        always_missed = [
            method for method, count in method_status.items() if count == len(observations)
        ]
        identity_rows.append({
            "roi_identity": identity, "annotated_observations": len(observations),
            "total_missed_method_observations": sum(method_status.values()),
            "always_missed_by_method_count": len(always_missed),
            "always_missed_by_all_methods": len(always_missed) == len(method_ids),
            "always_missed_methods": ",".join(always_missed),
            **{f"missed_observations__{method}": count for method, count in method_status.items()},
        })
    overlap_rows = []
    for left in method_ids:
        for right in method_ids:
            union = missed_by_method[left] | missed_by_method[right]
            intersection = missed_by_method[left] & missed_by_method[right]
            overlap_rows.append({
                "method_a": left, "method_b": right,
                "intersection": len(intersection), "union": len(union),
                "jaccard": len(intersection) / len(union) if union else 1.0,
            })
    return observation_rows, identity_rows, overlap_rows


def _base_rgb(gray: np.ndarray) -> np.ndarray:
    values = np.asarray(gray, dtype=np.uint8)
    channels = [
        np.clip(values.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        for factor in BACKGROUND_TINT
    ]
    return np.stack(channels, axis=-1)


def _draw_circle(draw: ImageDraw.ImageDraw, x: float, y: float, radius: int, color, width: int = 2) -> None:
    draw.ellipse(
        (int(round(x)) - radius, int(round(y)) - radius,
         int(round(x)) + radius, int(round(y)) + radius),
        outline=color, width=width,
    )


def _draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, radius: int, color, width: int = 2) -> None:
    x0, y0 = int(round(x)), int(round(y))
    draw.line((x0 - radius, y0 - radius, x0 + radius, y0 + radius), fill=color, width=width)
    draw.line((x0 - radius, y0 + radius, x0 + radius, y0 - radius), fill=color, width=width)


def _short_roi(identity: str) -> str:
    return identity.removeprefix("roi_").lstrip("0") or "0"


def render_overlay(
    gray: np.ndarray,
    *,
    frame_ui: int,
    title: str,
    classification: dict[str, Any] | None,
) -> np.ndarray:
    """Render one RGB frame with a compact, explicit sparse-label legend."""
    height, width = gray.shape
    canvas = np.zeros((height + 34, width, 3), dtype=np.uint8)
    canvas[:height] = _base_rgb(gray)
    canvas[height:] = (10, 16, 27)
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    tp = fn = unknown = 0
    if classification is not None:
        for row in classification["true_positive_labels"]:
            draw.line(
                (
                    int(row["x_px"]), int(row["y_px"]),
                    int(row["candidate_x_px"]), int(row["candidate_y_px"]),
                ),
                fill=TP_COLOR, width=1,
            )
            _draw_circle(draw, row["candidate_x_px"], row["candidate_y_px"], 6, TP_COLOR, 2)
            _draw_cross(draw, row["x_px"], row["y_px"], 3, TP_COLOR, 1)
            tp += 1
        for row in classification["false_negative_labels"]:
            _draw_circle(draw, row["x_px"], row["y_px"], 7, FN_COLOR, 2)
            _draw_cross(draw, row["x_px"], row["y_px"], 4, FN_COLOR, 2)
            draw.text(
                (int(row["x_px"]) + 8, int(row["y_px"]) - 8),
                _short_roi(str(row["roi_identity"])), fill=FN_COLOR, font=font,
            )
            fn += 1
        for row in classification["unmatched_candidates_unknown"]:
            _draw_circle(draw, row["x_px"], row["y_px"], 4, UNKNOWN_COLOR, 2)
            unknown += 1
    legend = (
        f"{title} | UI {frame_ui} | "
        f"GREEN known TP {tp}  MAGENTA known FN {fn}  "
        f"ORANGE unmatched/unknown {unknown}"
    )
    caveat = "Blue-gray = unscored background; FP and TN unavailable with sparse-positive labels"
    draw.text((5, height + 3), legend, fill=(235, 241, 250), font=font)
    draw.text((5, height + 18), caveat, fill=(150, 190, 235), font=font)
    return np.asarray(image)


def _raw_scaler(video: np.ndarray, quiet_frames: int) -> tuple[float, float]:
    low, high = np.percentile(np.asarray(video[:quiet_frames, ::4, ::4], dtype=np.float32), [1, 99.5])
    return float(low), max(float(high), float(low) + 1.0)


def _gray_from_raw(frame: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((np.asarray(frame, dtype=np.float32) - low) / (high - low) * 255, 0, 255).astype(np.uint8)


def _gray_from_display(frame: np.ndarray) -> np.ndarray:
    values = np.asarray(frame)
    if values.dtype == np.uint16:
        return (values // 257).astype(np.uint8)
    low, high = np.percentile(values, [0.5, 99.5])
    return np.clip((values - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)


def _frame_burst(labels: list[dict[str, Any]], frame_ui: int) -> int | None:
    for burst in sorted({int(row["burst_id"]) for row in labels}):
        row = next(item for item in labels if int(item["burst_id"]) == burst)
        if int(row["start_frame_ui"]) <= frame_ui <= int(row["end_frame_ui"]):
            return burst
    return None


def _write_diagnostic_tiff(
    path: Path,
    spec: MethodSpec,
    config: RepresentationBenchmarkConfig,
    labels: list[dict[str, Any]],
    classifications: dict[int, dict[str, Any]],
    progress: Path,
) -> dict[str, Any]:
    frame_count = config.frames.review_end_ui - config.frames.review_start_ui + 1
    source_video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    review = source_video[config.frames.review_start_ui - 1:config.frames.review_end_ui]
    quiet_frames = config.frames.quiet_end_ui - config.frames.quiet_start_ui + 1
    raw_low, raw_high = _raw_scaler(review, quiet_frames)
    display_tiff = tifffile.TiffFile(spec.source_path) if spec.source_kind == "tiff" else None
    description = json.dumps({
        "schema_version": 1, "method_id": spec.method_id, "lane": spec.lane,
        "source_kind": spec.source_kind, "axes": "TYXS", "channels": "RGB",
        "review_ui_inclusive": [config.frames.review_start_ui, config.frames.review_end_ui],
        "classification_contract": {
            "green": "known_true_positive",
            "magenta": "known_false_negative",
            "orange": "unmatched_candidate_unknown_not_false_positive",
            "blue_gray": "unscored_background_not_true_negative",
        },
    }, sort_keys=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        if display_tiff is not None and len(display_tiff.pages) != frame_count:
            raise ValueError(f"{spec.source_path} has {len(display_tiff.pages)} pages, expected {frame_count}")
        with tifffile.TiffWriter(temporary, bigtiff=True) as writer:
            for index in range(frame_count):
                frame_ui = config.frames.review_start_ui + index
                if display_tiff is None:
                    gray = _gray_from_raw(review[index], raw_low, raw_high)
                else:
                    gray = _gray_from_display(display_tiff.pages[index].asarray())
                burst = _frame_burst(labels, frame_ui)
                classification = classifications.get(burst) if burst is not None else None
                rgb = render_overlay(
                    gray, frame_ui=frame_ui, title=spec.title,
                    classification=classification,
                )
                writer.write(
                    rgb, photometric="rgb", compression="zlib",
                    description=description if index == 0 else None,
                    metadata=None,
                )
                if index % 50 == 0 or index + 1 == frame_count:
                    with progress.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({
                            "time_unix": time.time(), "method_id": spec.method_id,
                            "frame_index": index, "frame_ui": frame_ui,
                        }, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        if display_tiff is not None:
            display_tiff.close()
        if temporary.exists():
            temporary.unlink()
    return {
        "method_id": spec.method_id, "lane": spec.lane, "path": str(path),
        "bytes": path.stat().st_size, "frames": frame_count,
        "shape": [frame_count, review.shape[1] + 34, review.shape[2], 3],
    }


def _write_report(
    path: Path,
    method_rows: list[dict[str, Any]],
    missed_rows: list[dict[str, Any]],
    observation_rows: list[dict[str, Any]],
    identity_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Representation detection diagnostics", "",
        "## Classification contract", "",
        "- Green: detection matched one-to-one to a known sparse-positive label (known TP).",
        "- Magenta: known sparse-positive label not matched by the method (known FN).",
        "- Orange: unmatched candidate. Its truth is unknown; it is **not a confirmed FP**.",
        "- Blue-gray: unscored background. It is **not a confirmed TN**.", "",
        "Ordinary FP, TN, specificity, and precision require exhaustive negatives or reviewed "
        "background/candidate annotations and are not identified by this dataset.", "",
        "## Per-method totals", "",
        "| Method | Known TP | Known FN | Unknown candidates | Known recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for spec in METHODS:
        rows = [row for row in method_rows if row["method_id"] == spec.method_id]
        tp = sum(int(row["known_tp"]) for row in rows)
        fn = sum(int(row["known_fn"]) for row in rows)
        unknown = sum(int(row["unknown_candidates_not_fp"]) for row in rows)
        recall = tp / max(1, tp + fn)
        lines.append(f"| {spec.title} | {tp} | {fn} | {unknown} | {recall:.4f} |")
    all_methods = len(METHODS)
    all_missed_observations = [
        row for row in observation_rows if bool(row["missed_by_all_methods"])
    ]
    all_missed_identities = [
        row for row in identity_rows if bool(row["always_missed_by_all_methods"])
    ]
    lines.extend([
        "", "## Shared misses", "",
        f"`{len(all_missed_observations)}` of the 79 burst-specific label observations were "
        f"missed by all {all_methods} methods.",
        f"`{len(all_missed_identities)}` ROI identities were missed in every annotated occurrence "
        f"by all {all_methods} methods.", "",
    ])
    if all_missed_observations:
        lines.extend([
            "Burst-specific observations missed by every method:", "",
            ", ".join(row["observation_id"] for row in all_missed_observations), "",
        ])
    if all_missed_identities:
        lines.extend([
            "ROI identities always missed by every method:", "",
            ", ".join(row["roi_identity"] for row in all_missed_identities), "",
        ])
    lines.extend([
        "## Method-specific missed neurons", "",
        "The detailed table is `missed_observations.tsv`. Coordinates use `x=column`, "
        "`y=row`; UI frames are one-based and inclusive.", "",
    ])
    for spec in METHODS:
        rows = [row for row in missed_rows if row["method_id"] == spec.method_id]
        lines.extend([
            f"### {spec.title}", "",
            f"Missed `{len(rows)}` of 79 burst-specific observations.", "",
            ", ".join(row["observation_id"] for row in rows) if rows else "None.", "",
        ])
    lines.extend([
        "## Comparison files", "",
        "- `missed_observations.tsv`: every method × missed label observation.",
        "- `observation_miss_frequency.tsv`: how many methods missed each burst-specific observation.",
        "- `roi_identity_miss_frequency.tsv`: recurrence-aware misses for each ROI identity.",
        "- `missed_set_jaccard.tsv`: pairwise overlap of method miss sets.",
        "- `method_burst_summary.tsv`: per-method, per-burst counts.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def build(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = RepresentationBenchmarkConfig.load(config_path)
    if output_dir.exists():
        raise FileExistsError(f"Output exists: {output_dir}")
    inputs = [
        config.source_video, config.labels_tsv,
        PRIMARY / "metrics/candidate_peaks.tsv",
        *(spec.source_path for spec in METHODS if spec.source_kind == "tiff"),
    ]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    frame_count = config.frames.review_end_ui - config.frames.review_start_ui + 1
    height, width = np.load(config.source_video, mmap_mode="r", allow_pickle=False).shape[1:]
    estimated_bytes = len(METHODS) * frame_count * (height + 34) * width * 3
    free = shutil.disk_usage(output_dir.parent).free
    if free < max(estimated_bytes * 2, 4 * 2**30):
        raise RuntimeError("Insufficient disk headroom for diagnostic TIFFs")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "diagnostic_tiffs").mkdir()
    _atomic_json(output_dir / "run_state.json", {
        "status": "running", "phase": "classification",
    })
    progress = output_dir / "progress.jsonl"
    started = time.monotonic()
    try:
        labels = v1.load_labels(config.labels_tsv)
        candidate_rows = _read_tsv(PRIMARY / "metrics/candidate_peaks.tsv")
        by_method, missed_rows, method_rows = _build_classifications(labels, candidate_rows)
        observation_rows, identity_rows, overlap_rows = _aggregate_missed(labels, missed_rows)
        _write_tsv(
            output_dir / "missed_observations.tsv", missed_rows,
            [
                "method_id", "title", "lane", "observation_id", "burst_id",
                "roi_identity", "x_px", "y_px", "start_frame_ui", "end_frame_ui",
                "recurrence_count", "classification",
            ],
        )
        _write_tsv(
            output_dir / "method_burst_summary.tsv", method_rows,
            [
                "method_id", "title", "lane", "burst_id", "known_tp", "known_fn",
                "unknown_candidates_not_fp", "candidates", "known_recall",
                "tn_available", "fp_available",
            ],
        )
        _write_tsv(
            output_dir / "observation_miss_frequency.tsv", observation_rows,
            [
                "observation_id", "burst_id", "roi_identity", "x_px", "y_px",
                "missed_method_count", "method_count", "missed_by_all_methods",
                "missed_methods",
            ],
        )
        identity_fields = [
            "roi_identity", "annotated_observations", "total_missed_method_observations",
            "always_missed_by_method_count", "always_missed_by_all_methods",
            "always_missed_methods",
            *[f"missed_observations__{spec.method_id}" for spec in METHODS],
        ]
        _write_tsv(output_dir / "roi_identity_miss_frequency.tsv", identity_rows, identity_fields)
        _write_tsv(
            output_dir / "missed_set_jaccard.tsv", overlap_rows,
            ["method_a", "method_b", "intersection", "union", "jaccard"],
        )
        tiffs = []
        _atomic_json(output_dir / "run_state.json", {
            "status": "running", "phase": "tiff_generation",
        })
        for spec in METHODS:
            tiffs.append(_write_diagnostic_tiff(
                output_dir / "diagnostic_tiffs" / f"{spec.method_id}_sparse_label_diagnostic.tif",
                spec, config, labels, by_method[spec.method_id], progress,
            ))
        _write_report(
            output_dir / "report.md", method_rows, missed_rows,
            observation_rows, identity_rows,
        )
        shared_observations = [
            row["observation_id"] for row in observation_rows if row["missed_by_all_methods"]
        ]
        shared_identities = [
            row["roi_identity"] for row in identity_rows if row["always_missed_by_all_methods"]
        ]
        manifest = {
            "schema_version": 1, "status": "complete",
            "source_experiment": str(PRIMARY),
            "evaluation_policy": "existing_quiet_calibrated_candidate_peaks",
            "match_radius_px": 6, "methods": [spec.method_id for spec in METHODS],
            "tiffs": tiffs, "known_label_observations": len(labels),
            "shared_missed_observations": shared_observations,
            "shared_always_missed_roi_identities": shared_identities,
            "classification_contract": {
                "tp": "known one-to-one label match",
                "fn": "known label without a match",
                "unmatched_candidate": "unknown; not a confirmed false positive",
                "background": "unscored; not a confirmed true negative",
                "ordinary_fp_tn_available": False,
            },
            "estimated_uncompressed_bytes": estimated_bytes,
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(output_dir / "manifest.json", manifest)
        _atomic_json(output_dir / "run_state.json", {
            "status": "complete", "phase": "complete",
            "elapsed_seconds": manifest["elapsed_seconds"],
        })
        return manifest
    except Exception as exc:
        _atomic_json(output_dir / "run_state.json", {
            "status": "failed", "phase": "failed", "error": repr(exc),
            "elapsed_seconds": time.monotonic() - started,
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
