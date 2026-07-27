"""Pure external-label reconciliation for workbench imports.

This module deliberately performs no locking, job-state mutation, or file
publication.  Callers own source-identity checks and durable writes.
"""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from neurobench.data.imports import (
    MAX_LABEL_ARTIFACT_BYTES,
    infer_label_mapping,
    iter_label_rows,
)


@dataclass(frozen=True)
class LabelReconciliationResult:
    """Complete in-memory result ready for guarded publication."""

    artifact: dict[str, Any]
    artifact_bytes: bytes
    overlay_svg: bytes
    summary: dict[str, int]


def _canonical_label_mapping(mapping: Mapping[str, Any]) -> dict[str, str | None]:
    return {
        "roi_identity": str(mapping.get("roi_identity") or mapping.get("roi_id") or "") or None,
        "x": str(mapping.get("x") or "") or None,
        "y": str(mapping.get("y") or "") or None,
        "start_frame_ui": str(mapping.get("start_frame_ui") or mapping.get("start_frame") or "") or None,
        "end_frame_ui": str(mapping.get("end_frame_ui") or mapping.get("end_frame") or "") or None,
        "label": str(mapping.get("label") or "") or None,
        "confidence": str(mapping.get("confidence") or "") or None,
    }


def _json_safe_label_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return str(value)


def _normalize_roi_identity(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _finite_float(value: Any, field: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _ui_frame(value: Any, field: str, frame_count: int) -> int | None:
    numeric = _finite_float(value, field)
    if numeric is None:
        return None
    if not numeric.is_integer():
        raise ValueError(f"{field} must be an integer")
    frame = int(numeric)
    if frame < 1 or frame > frame_count:
        raise ValueError(f"{field} must be within 1..{frame_count} using one-based inclusive UI frames")
    return frame


def _label_overlay_svg(width: int, height: int, points: list[dict[str, Any]], *, truncated: bool) -> bytes:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="none"/>',
        f'<metadata>external-label projection; points={len(points)}; truncated={str(truncated).lower()}</metadata>',
    ]
    colors = {"matched": "#22c55e", "unmatched": "#f59e0b", "rejected": "#ef4444"}
    for point in points:
        status = str(point.get("status") or "unmatched")
        row_key = escape(str(point.get("row_key") or ""), quote=True)
        lines.append(
            f'<circle cx="{float(point["x"]):.3f}" cy="{float(point["y"]):.3f}" r="2.5" fill="{colors.get(status, colors["unmatched"])}" fill-opacity="0.72" data-row="{row_key}"/>'
        )
    lines.append("</svg>")
    return ("\n".join(lines) + "\n").encode("utf-8")


def reconcile_label_table(
    *,
    source: str | Path,
    review: Mapping[str, Any],
    import_record: Mapping[str, Any],
    mapping: Mapping[str, Any] | None = None,
    artifact_budget_bytes: int = MAX_LABEL_ARTIFACT_BYTES,
    max_overlay_points: int = 100_000,
    progress: Callable[[int, int], None] | None = None,
) -> LabelReconciliationResult:
    """Reconcile every source row without mutating jobs or the filesystem."""

    video = review.get("video") if isinstance(review.get("video"), Mapping) else {}
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    frame_count = int(video.get("frames") or 0)
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise ValueError("Review video dimensions and frame count must be positive.")

    native_lookup: dict[str, Any] = {}
    for roi in review.get("rois") or []:
        if isinstance(roi, Mapping) and roi.get("id") not in (None, ""):
            native_lookup[_normalize_roi_identity(roi.get("id"))] = roi.get("id")

    metadata = import_record.get("metadata") if isinstance(import_record.get("metadata"), Mapping) else {}
    inferred = metadata.get("label_mapping") or infer_label_mapping(metadata.get("columns", []))
    active_mapping = _canonical_label_mapping(dict(mapping or inferred))
    columns = {str(value) for value in metadata.get("columns", [])}
    if not active_mapping["roi_identity"]:
        raise ValueError("A roi_identity column mapping is required.")
    unknown_columns = sorted({column for column in active_mapping.values() if column and column not in columns})
    if unknown_columns:
        raise ValueError("Label mapping references unknown columns: " + ", ".join(unknown_columns))

    artifact_budget = min(MAX_LABEL_ARTIFACT_BYTES, max(1, int(artifact_budget_bytes)))
    overlay_limit = max(0, int(max_overlay_points))
    row_entries: dict[str, Any] = {}
    overlay_points: list[dict[str, Any]] = []
    projection_candidates = 0
    seen_identities: set[str] = set()
    summary = {
        "total_rows": 0,
        "matched_rows": 0,
        "unmatched_rows": 0,
        "duplicate_rows": 0,
        "rejected_rows": 0,
    }
    estimated_bytes = 4096
    expected_rows = max(1, int(metadata.get("row_count") or 1))
    for row_number, row in enumerate(iter_label_rows(source), start=2):
        row_key = f"row_{row_number:08d}"
        source_row = {str(key): _json_safe_label_value(value) for key, value in row.items()}
        errors: list[str] = []
        identity = _normalize_roi_identity(row.get(str(active_mapping["roi_identity"])))
        if not identity:
            errors.append("roi_identity is required")
        duplicate = bool(identity and identity in seen_identities)
        if identity:
            seen_identities.add(identity)
        normalized: dict[str, Any] = {"roi_identity": identity or None}
        x: float | None = None
        y: float | None = None
        try:
            x = _finite_float(row.get(str(active_mapping["x"])), "x") if active_mapping["x"] else None
            y = _finite_float(row.get(str(active_mapping["y"])), "y") if active_mapping["y"] else None
            if (x is None) != (y is None):
                raise ValueError("x and y must either both be present or both be empty")
            if x is not None and not 0 <= x < width:
                raise ValueError(f"x must be within 0..{width - 1}")
            if y is not None and not 0 <= y < height:
                raise ValueError(f"y must be within 0..{height - 1}")
            if x is not None:
                normalized.update({"x": x, "y": y})
        except ValueError as exc:
            errors.append(str(exc))
        try:
            start_ui = _ui_frame(row.get(str(active_mapping["start_frame_ui"])), "start_frame_ui", frame_count) if active_mapping["start_frame_ui"] else None
            end_ui = _ui_frame(row.get(str(active_mapping["end_frame_ui"])), "end_frame_ui", frame_count) if active_mapping["end_frame_ui"] else None
            if (start_ui is None) != (end_ui is None):
                raise ValueError("start_frame_ui and end_frame_ui must either both be present or both be empty")
            if start_ui is not None and end_ui is not None and end_ui < start_ui:
                raise ValueError("end_frame_ui must be greater than or equal to start_frame_ui")
            if start_ui is not None:
                normalized.update({"start_frame_ui": start_ui, "end_frame_ui": end_ui})
        except ValueError as exc:
            errors.append(str(exc))
        if active_mapping["label"]:
            normalized["label"] = _json_safe_label_value(row.get(str(active_mapping["label"])))
        if active_mapping["confidence"]:
            try:
                confidence = _finite_float(row.get(str(active_mapping["confidence"])), "confidence")
                if confidence is not None and not 0 <= confidence <= 1:
                    raise ValueError("confidence must be within 0..1")
                normalized["confidence"] = confidence
            except ValueError as exc:
                errors.append(str(exc))
        if errors:
            status = "rejected"
            summary["rejected_rows"] += 1
        elif identity in native_lookup:
            status = "matched"
            summary["matched_rows"] += 1
        else:
            status = "unmatched"
            summary["unmatched_rows"] += 1
        classifications = [status]
        if duplicate:
            classifications.append("duplicate")
            summary["duplicate_rows"] += 1
        entry = {
            "source_row_number": row_number,
            "source_row": source_row,
            "normalized": normalized,
            "reconciliation": {
                "status": status,
                "classifications": classifications,
                "matched_native_roi_id": _json_safe_label_value(native_lookup.get(identity)) if status == "matched" else None,
                "errors": errors,
            },
        }
        estimated_bytes += len(json.dumps(entry, separators=(",", ":"), ensure_ascii=False).encode("utf-8")) + len(row_key) + 8
        if estimated_bytes > artifact_budget:
            raise ValueError(f"External label artifact exceeds the bounded {artifact_budget:,}-byte memory/artifact budget.")
        row_entries[row_key] = entry
        summary["total_rows"] += 1
        if x is not None and y is not None:
            projection_candidates += 1
            if len(overlay_points) < overlay_limit:
                overlay_points.append({"row_key": row_key, "x": x, "y": y, "status": status})
        if progress is not None and summary["total_rows"] % 1000 == 0:
            progress(summary["total_rows"], expected_rows)

    overlay_truncated = projection_candidates > len(overlay_points)
    artifact = {
        "schema_version": 1,
        "kind": "neurobench_external_label_artifact",
        "dataset_id": str(import_record.get("dataset_id") or ""),
        "import_id": import_record.get("import_id"),
        "source": {
            "path": import_record.get("destination_path") or import_record.get("source_path"),
            "checksum": dict(import_record.get("checksum") or {}),
        },
        "mapping": active_mapping,
        "coordinate_contract": {
            "xy": "native full-frame pixels; x=column, y=row",
            "frames": "one-based inclusive UI frames",
        },
        "review_snapshot": {
            "width": width,
            "height": height,
            "frames": frame_count,
            "native_roi_count": len(native_lookup),
        },
        "summary": summary,
        "projection": {"point_count": len(overlay_points), "truncated": overlay_truncated},
        "rows": row_entries,
    }
    artifact_bytes = (json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    if len(artifact_bytes) > artifact_budget or len(artifact_bytes) > MAX_LABEL_ARTIFACT_BYTES:
        raise ValueError("Serialized external label artifact exceeds its bounded size limit.")
    overlay_svg = _label_overlay_svg(width, height, overlay_points, truncated=overlay_truncated)
    return LabelReconciliationResult(
        artifact=artifact,
        artifact_bytes=artifact_bytes,
        overlay_svg=overlay_svg,
        summary=summary,
    )
