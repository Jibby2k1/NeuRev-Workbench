"""Helpers for lightweight review ROI payloads and trace shards."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


TRACE_KEYS = (
    "rawTrace",
    "backgroundTrace",
    "dffTrace",
    "baselineTrace",
    "eventTrace",
    "zTrace",
)


def safe_payload_name(value: Any) -> str:
    """Return a filesystem-safe identifier for sidecar shard names."""
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(value))
    return cleaned.strip("._-") or "roi"


def event_calls_for_trace(
    trace: Sequence[Any],
    *,
    threshold_z: float = 2.4,
    kalman_gain: float = 0.06,
    spike_gain: float = 0.008,
) -> list[dict[str, float | int]]:
    """Mirror the browser's simple positive-innovation event model."""
    values = [float(value) for value in trace]
    if len(values) < 3:
        return []
    center = _median(values)
    sigma = max(1e-6, 1.4826 * _median([abs(value - center) for value in values]))
    baseline = center
    event_trace: list[float] = []
    z_trace: list[float] = []
    for value in values:
        residual = value - baseline
        gain = kalman_gain
        if residual > 2.5 * sigma:
            gain = spike_gain
        if residual < -1.0 * sigma:
            gain = min(0.18, kalman_gain * 1.8)
        baseline += gain * residual
        event_value = max(0.0, value - baseline)
        event_trace.append(event_value)
        z_trace.append(event_value / sigma)
    events: list[dict[str, float | int]] = []
    for index in range(1, len(z_trace) - 1):
        value = z_trace[index]
        if value >= threshold_z and value >= z_trace[index - 1] and value >= z_trace[index + 1]:
            events.append(
                {
                    "frame": index + 1,
                    "z": round(float(value), 6),
                    "amplitude": round(float(event_trace[index]), 6),
                }
            )
    return events


def normalized_events(events: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize candidate event rows for compact browser use."""
    rows: list[dict[str, Any]] = []
    for event in events or []:
        try:
            frame = int(event.get("frame"))
        except (TypeError, ValueError):
            continue
        z_value = _first_number(event, ("z", "score", "peak_z", "event_z"))
        amplitude = _first_number(event, ("amplitude", "event_amplitude"))
        row: dict[str, Any] = {"frame": frame}
        if z_value is not None:
            row["z"] = round(float(z_value), 6)
        if amplitude is not None:
            row["amplitude"] = round(float(amplitude), 6)
        if event.get("mode"):
            row["mode"] = str(event["mode"])
        rows.append(row)
    rows.sort(key=lambda item: int(item["frame"]))
    return rows


def events_by_roi_from_payload(payload: Mapping[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    """Group candidate event artifact rows by ROI ID."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in (payload or {}).get("events", []) or []:
        if not isinstance(event, Mapping):
            continue
        roi_id = event.get("roi_id") or event.get("candidate_id")
        if roi_id is None:
            continue
        grouped.setdefault(str(roi_id), []).append(dict(event))
    return {roi_id: normalized_events(events) for roi_id, events in grouped.items()}


def write_review_roi_sidecars(
    review_rois: Sequence[Mapping[str, Any]],
    *,
    summary_path: str | Path,
    shard_dir: str | Path,
    run_id: str,
    frame_count: int | None = None,
    event_threshold_z: float = 2.4,
    kalman_gain: float = 0.06,
    spike_gain: float = 0.008,
    events_by_roi: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    stencil_points: Sequence[Sequence[Any]] | None = None,
    gap_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write compact ROI summaries plus per-ROI trace shards."""
    summary = Path(summary_path)
    shards = Path(shard_dir)
    summary.parent.mkdir(parents=True, exist_ok=True)
    shards.mkdir(parents=True, exist_ok=True)
    trace_dir_rel = _rel_to(shards, summary.parent)
    compact_rois: list[dict[str, Any]] = []
    shard_count = 0
    for index, roi in enumerate(review_rois, start=1):
        roi_id = str(roi.get("id") or f"roi_{index:03d}")
        trace_payload = _trace_payload(roi, run_id=run_id, roi_id=roi_id)
        explicit_events = list((events_by_roi or {}).get(roi_id) or [])
        events = normalized_events(explicit_events or roi.get("events") or [])
        if not events and isinstance(roi.get("dffTrace"), Sequence):
            events = event_calls_for_trace(
                roi.get("dffTrace") or [],
                threshold_z=event_threshold_z,
                kalman_gain=kalman_gain,
                spike_gain=spike_gain,
            )
        if events:
            trace_payload["events"] = events
        trace_file = ""
        if any(key in trace_payload for key in TRACE_KEYS):
            trace_file_path = shards / f"{safe_payload_name(roi_id)}.json"
            _write_json_atomic(trace_file_path, trace_payload)
            trace_file = f"{trace_dir_rel}/{trace_file_path.name}" if trace_dir_rel else trace_file_path.name
            shard_count += 1
        compact_rois.append(_compact_roi(roi, roi_id=roi_id, trace_file=trace_file, events=events))

    payload = {
        "schema_version": 1,
        "payload_kind": "review_rois_summary",
        "run_id": run_id,
        "frame_count": frame_count,
        "roi_count": len(compact_rois),
        "event_threshold_z": event_threshold_z,
        "trace_shards_dir": trace_dir_rel,
        "trace_shard_count": shard_count,
        "review_rois": compact_rois,
    }
    _write_json_atomic(summary, payload)
    gap_payload = None
    if gap_report_path is not None:
        gap_payload = build_stencil_gap_report(compact_rois, stencil_points or [], run_id=run_id)
        _write_json_atomic(Path(gap_report_path), gap_payload)
    return {
        "summary_path": summary,
        "shard_dir": shards,
        "gap_report_path": Path(gap_report_path) if gap_report_path is not None else None,
        "roi_count": len(compact_rois),
        "trace_shard_count": shard_count,
        "gap_count": len((gap_payload or {}).get("gaps", [])),
    }


def build_stencil_gap_report(
    rois: Sequence[Mapping[str, Any]],
    stencil_points: Sequence[Sequence[Any]],
    *,
    run_id: str,
    grid_size: int = 4,
) -> dict[str, Any]:
    """Return rough low-coverage boxes inside the saved anatomy stencil."""
    points = [_point(point) for point in stencil_points]
    points = [point for point in points if point is not None]
    if len(points) < 3:
        return {"schema_version": 1, "run_id": run_id, "stencil_available": False, "gaps": []}
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    grid_size = max(2, int(grid_size))
    width = max(1.0, (x1 - x0) / grid_size)
    height = max(1.0, (y1 - y0) / grid_size)
    centers = [_roi_center(roi) for roi in rois]
    centers = [center for center in centers if center is not None]
    gaps: list[dict[str, Any]] = []
    for row in range(grid_size):
        for col in range(grid_size):
            bx0 = x0 + col * width
            by0 = y0 + row * height
            bx1 = x0 + (col + 1) * width
            by1 = y0 + (row + 1) * height
            cx = 0.5 * (bx0 + bx1)
            cy = 0.5 * (by0 + by1)
            if not _point_in_polygon((cx, cy), points):
                continue
            count = sum(1 for rx, ry in centers if bx0 <= rx <= bx1 and by0 <= ry <= by1)
            gaps.append(
                {
                    "id": f"gap_r{row + 1}_c{col + 1}",
                    "bbox": [round(bx0, 3), round(by0, 3), round(bx1, 3), round(by1, 3)],
                    "center": [round(cx, 3), round(cy, 3)],
                    "roi_count": int(count),
                    "priority": round(float((width * height) / max(1, count + 1)), 6),
                }
            )
    gaps.sort(key=lambda item: (int(item["roi_count"]), -float(item["priority"]), str(item["id"])))
    return {
        "schema_version": 1,
        "run_id": run_id,
        "stencil_available": True,
        "grid_size": grid_size,
        "stencil_bounds": [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)],
        "gaps": gaps[:12],
    }


def stencil_points_from_annotations(annotations: Mapping[str, Any] | None) -> list[list[float]]:
    """Extract the saved anatomy stencil polygon from annotations.json."""
    polygon = ((annotations or {}).get("settings") or {}).get("anatomyStencil", {}).get("polygon", [])
    points: list[list[float]] = []
    for point in polygon or []:
        parsed = _point(point)
        if parsed is not None:
            points.append([parsed[0], parsed[1]])
    return points


def _compact_roi(
    roi: Mapping[str, Any],
    *,
    roi_id: str,
    trace_file: str,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    compact = {key: value for key, value in roi.items() if key not in TRACE_KEYS}
    compact["id"] = roi_id
    compact["events"] = list(events)
    compact["event_count"] = len(events)
    if trace_file:
        compact["trace_file"] = trace_file
        compact["trace_status"] = "lazy"
    return compact


def _trace_payload(roi: Mapping[str, Any], *, run_id: str, roi_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"schema_version": 1, "run_id": run_id, "roi_id": roi_id}
    for key in TRACE_KEYS:
        if key in roi:
            payload[key] = roi[key]
    for key in ("noiseSigma", "traceSnr", "backgroundCorrelation", "eventSupport", "trace_materialization"):
        if key in roi:
            payload[key] = roi[key]
    return payload


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return 0.5 * (float(ordered[mid - 1]) + float(ordered[mid]))


def _first_number(row: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
    return None


def _roi_center(roi: Mapping[str, Any]) -> tuple[float, float] | None:
    x_value = roi.get("centroidX", roi.get("x", roi.get("center_x", roi.get("centerX"))))
    y_value = roi.get("centroidY", roi.get("y", roi.get("center_y", roi.get("centerY"))))
    try:
        x = float(x_value)
        y = float(y_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _point(value: Sequence[Any]) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or len(value) < 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _point_in_polygon(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _rel_to(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return os.path.relpath(path, base).replace(os.sep, "/")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
