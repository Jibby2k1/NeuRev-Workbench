"""Amplitude-preserving, site-keyed trace extraction and metrics."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .contracts import ObservationRecord, stable_hash


@dataclass(frozen=True)
class Geometry:
    center_radius: int = 2
    annulus_inner: int = 3
    annulus_outer: int = 6
    outer_inner: int = 7
    outer_outer: int = 10

    @property
    def parameters_hash(self) -> str:
        return stable_hash(self.__dict__)


def _disk_trace(video: np.ndarray, x: float, y: float, radius: int) -> np.ndarray:
    xi, yi = int(round(x)), int(round(y))
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = xx * xx + yy * yy <= radius * radius
    y0, y1, x0, x1 = yi - radius, yi + radius + 1, xi - radius, xi + radius + 1
    if y0 < 0 or x0 < 0 or y1 > video.shape[1] or x1 > video.shape[2]:
        raise ValueError("center geometry exceeds video bounds")
    return np.asarray(video[:, y0:y1, x0:x1], dtype=np.float64)[:, mask].mean(axis=1)


def _ring_trace(video: np.ndarray, x: float, y: float, inner: int, outer: int) -> np.ndarray:
    xi, yi = int(round(x)), int(round(y))
    yy, xx = np.ogrid[-outer:outer + 1, -outer:outer + 1]
    rr = xx * xx + yy * yy
    mask = (rr >= inner * inner) & (rr <= outer * outer)
    y0, y1, x0, x1 = yi - outer, yi + outer + 1, xi - outer, xi + outer + 1
    if y0 < 0 or x0 < 0 or y1 > video.shape[1] or x1 > video.shape[2]:
        raise ValueError("ring geometry exceeds video bounds")
    return np.asarray(video[:, y0:y1, x0:x1], dtype=np.float64)[:, mask].mean(axis=1)


def extract_site_traces(video: np.ndarray, records: Iterable[ObservationRecord], geometry: Geometry) -> tuple[list[str], dict[str, np.ndarray]]:
    sites: dict[str, ObservationRecord] = {}
    for record in records:
        previous = sites.setdefault(record.observation_site_id, record)
        if previous.geometry_hash != record.geometry_hash:
            raise ValueError(f"site geometry changed across occurrences: {record.observation_site_id}")
    order = sorted(sites)
    channels = {name: [] for name in ("raw", "annulus", "outer_ring", "residual", "spatial_control", "signed_difference")}
    for site in order:
        record = sites[site]
        raw = _disk_trace(video, record.x_px, record.y_px, geometry.center_radius)
        annulus = _ring_trace(video, record.x_px, record.y_px, geometry.annulus_inner, geometry.annulus_outer)
        outer = _ring_trace(video, record.x_px, record.y_px, geometry.outer_inner, geometry.outer_outer)
        channels["raw"].append(raw)
        channels["annulus"].append(annulus)
        channels["outer_ring"].append(outer)
        channels["residual"].append(raw - annulus)
        channels["spatial_control"].append(annulus - outer)
        channels["signed_difference"].append(np.r_[np.nan, np.diff(raw)])
    return order, {name: np.stack(values) for name, values in channels.items()}


def frozen_two_frame_ica_traces(raw_traces: np.ndarray, frozen_path: Path) -> tuple[np.ndarray, str]:
    """Apply a hash-verified, label-free two-frame operator to site traces."""
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    canonical = json.dumps(
        {"fit": frozen["fit"], "selection": frozen["selection"]},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    fingerprint = hashlib.sha256(canonical).hexdigest()
    if fingerprint != frozen["sha256"]:
        raise ValueError("frozen two-frame ICA fingerprint mismatch")
    selected = int(frozen["selection"]["selected_component"])
    direction = np.asarray(frozen["fit"]["effective_directions"], dtype=float)[selected].copy()
    derivative = np.asarray([-1.0, 1.0]) / np.sqrt(2.0)
    direction *= np.sign(direction @ derivative) or 1.0
    mean = np.asarray(frozen["fit"]["mean"], dtype=float)
    pairs = np.stack((raw_traces[:, :-1], raw_traces[:, 1:]), axis=-1)
    values = (pairs - mean) @ direction
    return np.concatenate((np.full((len(raw_traces), 1), np.nan), values), axis=1), fingerprint


def extract_disk_sites(values: np.ndarray, records: Iterable[ObservationRecord], site_order: list[str], radius: int) -> np.ndarray:
    """Extract one trace per immutable site from a quantitative feature tensor."""
    by_site = {record.observation_site_id: record for record in records}
    return np.stack([_disk_trace(values, by_site[site].x_px, by_site[site].y_px, radius) for site in site_order])


def occurrence_metrics(record: ObservationRecord, traces: dict[str, np.ndarray], site_index: int, pre_frames: int = 20) -> dict[str, object]:
    start, stop = record.start_zero, record.stop_zero_exclusive
    baseline_start = max(0, start - pre_frames)
    raw = traces["raw"][site_index]
    annulus = traces["annulus"][site_index]
    residual = traces["residual"][site_index]
    base = raw[baseline_start:start]
    event = raw[start:stop]
    center = float(np.median(base)); mad = float(np.median(np.abs(base - center)) * 1.4826)
    delta = event - center
    ann_base = float(np.median(annulus[baseline_start:start]))
    ann_delta = annulus[start:stop] - ann_base
    residual_base = float(np.median(residual[baseline_start:start]))
    residual_delta = residual[start:stop] - residual_base
    peak_offset = int(np.argmax(delta))
    denom = abs(float(np.max(delta))) + abs(float(np.max(ann_delta))) + np.finfo(float).eps
    paired_raw = raw[baseline_start:stop]; paired_annulus = annulus[baseline_start:stop]
    corr = float(np.corrcoef(paired_raw, paired_annulus)[0, 1]) if len(paired_raw) > 2 and np.std(paired_raw) > 0 and np.std(paired_annulus) > 0 else 0.0
    slope = float(np.polyfit(np.arange(len(base)), base, 1)[0]) if len(base) > 1 else float("nan")
    result: dict[str, object] = {
        "observation_id": record.observation_id, "burst_id": record.burst_id,
        "original_roi_id": record.original_roi_id, "observation_site_id": record.observation_site_id,
        "canonical_neuron_id": record.canonical_neuron_id or "", "geometry_hash": record.geometry_hash,
        "analysis_view": "original_site_original_timing", "units": "native_uint16_intensity",
        "baseline_median": center, "baseline_mean": float(np.mean(base)), "baseline_mad": mad,
        "baseline_slope_per_frame": slope, "signed_peak_amplitude": float(np.max(delta)),
        "absolute_peak_amplitude": float(np.max(np.abs(delta))),
        "robust_peak_snr": float(np.max(delta) / max(mad, np.finfo(float).eps)),
        "event_area": float(np.sum(delta)), "positive_area": float(np.sum(np.maximum(delta, 0))),
        "time_to_peak_frames": peak_offset, "peak_frame_ui": start + peak_offset + 1,
        "residual_peak_amplitude": float(np.max(residual_delta)),
        "annulus_peak_amplitude": float(np.max(ann_delta)), "annulus_correlation": corr,
        "normalized_spatial_specificity": float((np.max(delta) - np.max(ann_delta)) / denom),
        "saturation_fraction": float(np.mean(event >= np.iinfo(np.uint16).max)) if event.size else 0.0,
        "boundary_clipped": False, "undefined_reason": "",
    }
    for channel in ("frozen_two_frame_ica", "carrier_signed", "coherence_w15", "propagation_lag2_w15"):
        if channel not in traces:
            result[f"{channel}_available"] = False
            continue
        values = traces[channel][site_index, start:stop]
        finite = np.isfinite(values)
        result[f"{channel}_available"] = bool(np.any(finite))
        result[f"{channel}_signed_peak"] = float(np.max(values[finite])) if np.any(finite) else ""
        if channel == "frozen_two_frame_ica" and np.sum(finite) > 2:
            paired_raw = delta[finite]
            paired_values = values[finite]
            result["raw_frozen_ica_event_correlation"] = float(np.corrcoef(paired_raw, paired_values)[0, 1]) if np.std(paired_raw) > 0 and np.std(paired_values) > 0 else 0.0
            result["raw_frozen_ica_peak_offset_frames"] = int(np.argmax(paired_values) - np.argmax(paired_raw))
    return result
