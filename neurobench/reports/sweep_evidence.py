"""Automated evidence reports for workbench sweep runs."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence


DEFAULT_STABILITY_RADIUS_PX = 10.0
DEFAULT_STABILITY_MIN_SUPPORT_RUNS = 2
DEFAULT_STENCIL_EDGE_MARGIN_PX = 12.0
DEFAULT_TARGET_ROI_RANGE = (40, 180)


def build_sweep_evidence_report(
    app_dir: str | Path,
    *,
    architecture_runs_path: str | Path | None = None,
    annotations_path: str | Path | None = None,
    stability_radius_px: float = DEFAULT_STABILITY_RADIUS_PX,
    stability_min_support_runs: int = DEFAULT_STABILITY_MIN_SUPPORT_RUNS,
    stencil_edge_margin_px: float = DEFAULT_STENCIL_EDGE_MARGIN_PX,
    target_roi_range: tuple[int, int] = DEFAULT_TARGET_ROI_RANGE,
    top_n: int = 8,
) -> dict[str, Any]:
    """Build a reproducible metric report for all generated sweep runs in an app."""
    app_path = Path(app_dir).expanduser().resolve()
    architecture_path = Path(architecture_runs_path or app_path / "architecture_runs.json").expanduser().resolve()
    annotations_file = Path(annotations_path or app_path / "annotations.json").expanduser().resolve()
    if not architecture_path.exists():
        raise FileNotFoundError(f"Missing architecture run manifest: {architecture_path}")

    project_root = _find_project_root(app_path)
    manifest = _load_json(architecture_path)
    annotations = _load_json(annotations_file) if annotations_file.exists() else {}
    stencil_points = _stencil_points_from_annotations(annotations)

    rows = [
        _summarize_run(
            run,
            app_dir=app_path,
            project_root=project_root,
            stencil_points=stencil_points,
            stencil_edge_margin_px=stencil_edge_margin_px,
        )
        for run in manifest.get("runs", []) or []
        if isinstance(run, Mapping) and _should_report_run(run)
    ]
    _attach_candidate_stability(rows, radius_px=stability_radius_px, min_support_runs=stability_min_support_runs)
    for row in rows:
        row["diagnostics"] = _diagnostics_for_row(row)
        row["evidence_score"] = _evidence_score(row, target_roi_range=target_roi_range)

    analyzable = [row for row in rows if int(row.get("roi_count") or 0) > 0]
    ranked = sorted(analyzable, key=lambda row: (-float(row.get("evidence_score") or 0), str(row.get("run_id") or "")))
    diagnosis_counts = Counter(diag["code"] for row in rows for diag in row.get("diagnostics", []))
    return {
        "schema_version": 1,
        "payload_kind": "sweep_evidence_report",
        "dataset_id": str(manifest.get("dataset_id") or ""),
        "app_dir": str(app_path),
        "architecture_runs_file": str(architecture_path),
        "settings": {
            "stability_radius_px": float(stability_radius_px),
            "stability_min_support_runs": int(stability_min_support_runs),
            "stencil_edge_margin_px": float(stencil_edge_margin_px),
            "target_roi_range": list(target_roi_range),
            "ranking": "higher evidence_score means stronger automated sweep evidence, not ground truth accuracy",
        },
        "summary": _report_summary(rows, diagnosis_counts),
        "recommended_runs": [_recommendation_row(row) for row in ranked[:top_n]],
        "diagnosis_counts": dict(sorted(diagnosis_counts.items())),
        "runs": rows,
    }


def render_sweep_evidence_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact professor-facing Markdown summary."""
    summary = dict(report.get("summary") or {})
    lines = [
        f"# Sweep Evidence Report: {report.get('dataset_id', '')}",
        "",
        "## Summary",
        "",
        f"- Runs analyzed: {summary.get('analyzed_run_count', 0)}",
        f"- Runs with ROI sidecars: {summary.get('runs_with_roi_sidecars', 0)}",
        f"- Runs with saved-stencil metrics: {summary.get('runs_with_stencil_metrics', 0)}",
        f"- Runs with CFAR contrast maps: {summary.get('runs_with_contrast_maps', 0)}",
        f"- Median stencil coverage: {_fmt(summary.get('median_stencil_coverage_fraction'), 3)}",
        f"- Median candidate stability: {_fmt(summary.get('median_stable_roi_fraction'), 3)}",
        "",
        "Stencil coverage and stability are automated triage signals. They are not ground-truth accuracy.",
        "",
        "## Recommended Inspection Order",
        "",
        "| Rank | Run | Score | ROIs | Events | Stencil | Stable | Zero-Gap Boxes | Parameters |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    recommendations = list(report.get("recommended_runs") or [])
    if recommendations:
        for index, row in enumerate(recommendations, start=1):
            params = row.get("parameters") or {}
            param_text = ", ".join(
                part
                for part in [
                    f"pfa={_fmt(params.get('cfar_small_ref.pfa'), 4)}" if params.get("cfar_small_ref.pfa") is not None else "",
                    f"ref={_fmt(params.get('cfar_large_ref.training_radius_px'), 0)}" if params.get("cfar_large_ref.training_radius_px") is not None else "",
                    f"support={_fmt(params.get('components.support_min_frames'), 0)}" if params.get("components.support_min_frames") is not None else "",
                ]
                if part
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(index),
                        f"`{row.get('run_id', '')}`",
                        _fmt(row.get("evidence_score"), 1),
                        str(row.get("roi_count", 0)),
                        str(row.get("event_count", 0)),
                        _fmt(row.get("stencil_coverage_fraction"), 3),
                        _fmt(row.get("stable_roi_fraction"), 3),
                        str(row.get("zero_roi_gap_count", 0)),
                        param_text or "n/a",
                    ]
                )
                + " |"
            )
    else:
        lines.append("|  | No candidate-bearing runs were available. |  |  |  |  |  |  |  |")

    lines.extend(["", "## Diagnostic Counts", ""])
    counts = dict(report.get("diagnosis_counts") or {})
    if counts:
        for code, count in counts.items():
            lines.append(f"- `{code}`: {count}")
    else:
        lines.append("- No automated diagnostics were triggered.")

    lines.extend(
        [
            "",
            "## Stage Evidence Notes",
            "",
            "- Use Data > Compare to inspect raw frames against CFAR contrast maps for the ranked sweeps.",
            "- Use Review > Overlap to inspect spatial consistency and sweep-specific ROI positions.",
            "- Use Review > Triage to inspect stencil gap boxes for the active sweep.",
            "- If a run has strong contrast maps but weak stencil coverage, inspect the threshold/component-filter stage before changing the whole architecture.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_sweep_evidence_report(
    app_dir: str | Path,
    *,
    output: str | Path | None = None,
    markdown_output: str | Path | None = None,
    architecture_runs_path: str | Path | None = None,
    annotations_path: str | Path | None = None,
    attach: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write JSON and Markdown sweep evidence reports, optionally attaching them to the manifest."""
    app_path = Path(app_dir).expanduser().resolve()
    architecture_path = Path(architecture_runs_path or app_path / "architecture_runs.json").expanduser().resolve()
    output_path = Path(output or app_path / "sweep_evidence_report.json").expanduser().resolve()
    markdown_path = Path(markdown_output or output_path.with_suffix(".md")).expanduser().resolve()
    report = build_sweep_evidence_report(
        app_path,
        architecture_runs_path=architecture_path,
        annotations_path=annotations_path,
        **kwargs,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, report)
    markdown_path.write_text(render_sweep_evidence_markdown(report), encoding="utf-8")
    if attach:
        _attach_report_to_manifest(architecture_path, app_path, output_path, markdown_path)
    return report


def _summarize_run(
    run: Mapping[str, Any],
    *,
    app_dir: Path,
    project_root: Path,
    stencil_points: Sequence[tuple[float, float]],
    stencil_edge_margin_px: float,
) -> dict[str, Any]:
    artifacts = dict(run.get("artifacts") or {})
    summary = dict(run.get("summary") or {})
    roi_payload, roi_payload_path = _load_optional_json(
        _resolve_artifact_path(app_dir, project_root, artifacts.get("review_rois_summary_file") or artifacts.get("review_rois_file"))
    )
    gap_payload, gap_payload_path = _load_optional_json(_resolve_artifact_path(app_dir, project_root, artifacts.get("stencil_gap_report_file")))
    rois = _review_rois(roi_payload)
    centers = [_roi_center_row(roi) for roi in rois]
    centers = [center for center in centers if center is not None]
    roi_count = _as_int((roi_payload or {}).get("roi_count"), default=len(rois) or _as_int(summary.get("roi_count"), 0))
    event_count = _event_count(rois, default=_as_int(summary.get("event_count"), 0))
    stencil = _stencil_coverage(centers, stencil_points, edge_margin_px=stencil_edge_margin_px)
    gaps = list((gap_payload or {}).get("gaps") or [])
    zero_gap_count = sum(1 for gap in gaps if _as_int(gap.get("roi_count"), 0) == 0)
    contrast_maps = _contrast_maps(artifacts)
    parameters = _run_parameters(run, summary)
    return {
        "run_id": str(run.get("run_id") or ""),
        "label": str(run.get("label") or ""),
        "status": str((run.get("execution") or {}).get("status") or run.get("status") or ""),
        "parameters": parameters,
        "roi_count": int(roi_count),
        "event_count": int(event_count),
        "event_density_per_roi": round(float(event_count) / float(max(1, roi_count)), 6),
        "median_equivalent_diameter_um": _as_float(summary.get("median_equivalent_diameter_um")),
        "plausible_size_fraction": _as_float(summary.get("plausible_size_fraction"), 0.0),
        "final_active_fraction": _as_float(summary.get("final_active_fraction")),
        "roi_sidecar": _sidecar_summary(roi_payload, roi_payload_path, rois),
        "stencil": stencil,
        "stencil_coverage_fraction": stencil.get("coverage_fraction"),
        "stencil_gap_report": {
            "available": bool(gap_payload),
            "path": str(gap_payload_path) if gap_payload_path else "",
            "gap_count": len(gaps),
            "zero_roi_gap_count": int(zero_gap_count),
            "largest_gap_priority": max((_as_float(gap.get("priority"), 0.0) or 0.0 for gap in gaps), default=0.0),
        },
        "contrast_maps": contrast_maps,
        "_centers": centers,
    }


def _should_report_run(run: Mapping[str, Any]) -> bool:
    artifacts = dict(run.get("artifacts") or {})
    return bool(
        artifacts.get("review_rois_summary_file")
        or artifacts.get("review_rois_file")
        or artifacts.get("intermediates")
        or str(run.get("run_id") or "").startswith("gamma_cfar")
    )


def _attach_candidate_stability(rows: list[dict[str, Any]], *, radius_px: float, min_support_runs: int) -> None:
    indexes = {row["run_id"]: _spatial_index(row.get("_centers") or [], radius_px) for row in rows}
    for row in rows:
        centers = list(row.get("_centers") or [])
        supports: list[int] = []
        for center in centers:
            support = 0
            for other in rows:
                if other is row:
                    continue
                if _has_nearby_center(center, indexes.get(other["run_id"], {}), radius_px):
                    support += 1
            supports.append(support)
        stable = sum(1 for value in supports if value >= min_support_runs)
        row["candidate_stability"] = {
            "radius_px": float(radius_px),
            "min_support_runs": int(min_support_runs),
            "roi_count": len(centers),
            "stable_roi_count": int(stable),
            "stable_roi_fraction": round(stable / float(len(centers)), 6) if centers else None,
            "median_support_runs": round(float(median(supports)), 6) if supports else None,
        }
        row["stable_roi_fraction"] = row["candidate_stability"]["stable_roi_fraction"]
        row.pop("_centers", None)


def _evidence_score(row: Mapping[str, Any], *, target_roi_range: tuple[int, int]) -> float:
    roi_count = int(row.get("roi_count") or 0)
    if roi_count <= 0:
        return 0.0
    coverage = _as_float(row.get("stencil_coverage_fraction"), 0.0) or 0.0
    stability = _as_float(row.get("stable_roi_fraction"), 0.0) or 0.0
    plausible = _as_float(row.get("plausible_size_fraction"), 0.0) or 0.0
    event_density = min(1.0, (_as_float(row.get("event_density_per_roi"), 0.0) or 0.0) / 8.0)
    low, high = target_roi_range
    if roi_count < low:
        roi_balance = roi_count / float(max(1, low))
    elif roi_count > high:
        roi_balance = max(0.0, 1.0 - (roi_count - high) / float(max(1, high)))
    else:
        roi_balance = 1.0
    gap_report = dict(row.get("stencil_gap_report") or {})
    gap_count = int(gap_report.get("gap_count") or 0)
    zero_gap_fraction = int(gap_report.get("zero_roi_gap_count") or 0) / float(max(1, gap_count))
    score = 100.0 * (
        0.34 * coverage
        + 0.26 * stability
        + 0.18 * roi_balance
        + 0.12 * event_density
        + 0.10 * plausible
        - 0.20 * zero_gap_fraction
    )
    return round(max(0.0, min(100.0, score)), 3)


def _diagnostics_for_row(row: Mapping[str, Any]) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    roi_count = int(row.get("roi_count") or 0)
    coverage = _as_float(row.get("stencil_coverage_fraction"))
    stability = _as_float(row.get("stable_roi_fraction"))
    gap_report = dict(row.get("stencil_gap_report") or {})
    if roi_count == 0:
        diagnostics.append({"code": "no_candidate_rois", "message": "No candidate ROIs were available for this run."})
    elif roi_count < 20:
        diagnostics.append({"code": "sparse_candidates", "message": "The run has a small candidate set and may be over-filtered."})
    elif roi_count > 300:
        diagnostics.append({"code": "high_candidate_burden", "message": "The run may create too much review burden."})
    if not dict(row.get("stencil") or {}).get("available"):
        diagnostics.append({"code": "no_saved_stencil", "message": "No saved anatomy stencil was available for coverage scoring."})
    elif coverage is not None and coverage < 0.5:
        diagnostics.append({"code": "low_stencil_coverage", "message": "Few candidates fall inside or near the stencil."})
    if int(gap_report.get("zero_roi_gap_count") or 0) >= 4:
        diagnostics.append({"code": "hindbrain_gap_boxes", "message": "Several stencil boxes contain no detected ROI centers."})
    if stability is not None and roi_count >= 20 and stability < 0.35:
        diagnostics.append({"code": "low_candidate_stability", "message": "Many candidates are not spatially repeated across neighboring sweeps."})
    if not row.get("contrast_maps"):
        diagnostics.append({"code": "no_cfar_contrast_maps", "message": "No CFAR contrast-map artifact is attached to this run."})
    diameter = _as_float(row.get("median_equivalent_diameter_um"))
    if diameter is not None and diameter < 5.0:
        diagnostics.append({"code": "undersized_components", "message": "Median equivalent diameter is below the expected 5-10 micron soma range."})
    return diagnostics


def _report_summary(rows: Sequence[Mapping[str, Any]], diagnosis_counts: Counter[str]) -> dict[str, Any]:
    coverages = [_as_float(row.get("stencil_coverage_fraction")) for row in rows]
    stabilities = [_as_float(row.get("stable_roi_fraction")) for row in rows]
    scores = [_as_float(row.get("evidence_score")) for row in rows]
    coverages = [value for value in coverages if value is not None]
    stabilities = [value for value in stabilities if value is not None]
    scores = [value for value in scores if value is not None]
    return {
        "analyzed_run_count": len(rows),
        "candidate_bearing_run_count": sum(1 for row in rows if int(row.get("roi_count") or 0) > 0),
        "runs_with_roi_sidecars": sum(1 for row in rows if dict(row.get("roi_sidecar") or {}).get("available")),
        "runs_with_stencil_metrics": sum(1 for row in rows if dict(row.get("stencil") or {}).get("available")),
        "runs_with_contrast_maps": sum(1 for row in rows if row.get("contrast_maps")),
        "median_stencil_coverage_fraction": round(float(median(coverages)), 6) if coverages else None,
        "median_stable_roi_fraction": round(float(median(stabilities)), 6) if stabilities else None,
        "median_evidence_score": round(float(median(scores)), 6) if scores else None,
        "top_problem_classes": [
            {"code": code, "count": count} for code, count in diagnosis_counts.most_common(8)
        ],
    }


def _recommendation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    gap_report = dict(row.get("stencil_gap_report") or {})
    return {
        "run_id": row.get("run_id", ""),
        "label": row.get("label", ""),
        "evidence_score": row.get("evidence_score"),
        "roi_count": row.get("roi_count", 0),
        "event_count": row.get("event_count", 0),
        "stencil_coverage_fraction": row.get("stencil_coverage_fraction"),
        "stable_roi_fraction": row.get("stable_roi_fraction"),
        "zero_roi_gap_count": gap_report.get("zero_roi_gap_count", 0),
        "parameters": row.get("parameters", {}),
        "diagnostic_codes": [diag["code"] for diag in row.get("diagnostics", [])],
    }


def _sidecar_summary(payload: Mapping[str, Any] | None, path: Path | None, rois: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "available": bool(payload),
        "path": str(path) if path else "",
        "payload_kind": str((payload or {}).get("payload_kind") or ""),
        "roi_count": _as_int((payload or {}).get("roi_count"), len(rois)),
        "trace_shard_count": _as_int((payload or {}).get("trace_shard_count"), 0),
    }


def _contrast_maps(artifacts: Mapping[str, Any]) -> list[dict[str, Any]]:
    maps = []
    for item in artifacts.get("intermediates") or []:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("artifact_kind") or item.get("kind") or "")
        label = str(item.get("label") or item.get("id") or "")
        if kind != "cfar_contrast_map" and "contrast" not in label.lower():
            continue
        summary = dict(item.get("summary") or {})
        norm = dict(summary.get("normalization") or {})
        maps.append(
            {
                "id": str(item.get("id") or item.get("step_id") or label),
                "label": label,
                "frame_count": _as_int(item.get("frame_count") or summary.get("frame_count"), 0),
                "training_radius_px": _as_float(summary.get("training_radius_px")),
                "guard_px": _as_float(summary.get("guard_px")),
                "pfa": _as_float(summary.get("pfa")),
                "normalization_hi": _as_float(norm.get("hi")),
                "sample_max": _as_float(norm.get("sample_max")),
            }
        )
    return maps


def _run_parameters(run: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for key, value in summary.items():
        if "." in str(key):
            parameters[str(key)] = value
    for item in (run.get("sweep") or {}).get("parameters") or []:
        if isinstance(item, Mapping) and item.get("stage") and item.get("param"):
            parameters[f"{item['stage']}.{item['param']}"] = item.get("value")
    return dict(sorted(parameters.items()))


def _review_rois(payload: Mapping[str, Any] | list[Any] | None) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    rows = payload.get("review_rois") or payload.get("rois") or payload.get("candidates") or []
    return [row for row in rows if isinstance(row, Mapping)]


def _roi_center_row(roi: Mapping[str, Any]) -> dict[str, Any] | None:
    x = _as_float(roi.get("centroidX", roi.get("x", roi.get("center_x", roi.get("centerX")))))
    y = _as_float(roi.get("centroidY", roi.get("y", roi.get("center_y", roi.get("centerY")))))
    if x is None or y is None:
        return None
    return {
        "id": str(roi.get("id") or ""),
        "x": float(x),
        "y": float(y),
        "area": _as_float(roi.get("area"), 0.0) or 0.0,
        "event_count": _as_int(roi.get("event_count"), len(roi.get("events") or [])),
    }


def _event_count(rois: Sequence[Mapping[str, Any]], *, default: int) -> int:
    if not rois:
        return int(default)
    total = 0
    for roi in rois:
        if "event_count" in roi:
            total += _as_int(roi.get("event_count"), 0)
        elif isinstance(roi.get("events"), Sequence):
            total += len(roi.get("events") or [])
        else:
            total += _as_int(roi.get("eventSupport"), 0)
    return int(total)


def _stencil_coverage(
    centers: Sequence[Mapping[str, Any]],
    stencil_points: Sequence[tuple[float, float]],
    *,
    edge_margin_px: float,
) -> dict[str, Any]:
    if len(stencil_points) < 3:
        return {"available": False, "point_count": len(stencil_points), "coverage_fraction": None}
    inside = edge = outside = 0
    for center in centers:
        point = (float(center["x"]), float(center["y"]))
        is_inside = _point_in_polygon(point, stencil_points)
        is_edge = _distance_to_polygon(point, stencil_points) <= edge_margin_px
        if is_edge:
            edge += 1
        elif is_inside:
            inside += 1
        else:
            outside += 1
    total = len(centers)
    in_near = inside + edge
    return {
        "available": True,
        "point_count": len(stencil_points),
        "inside_count": int(inside),
        "edge_count": int(edge),
        "outside_count": int(outside),
        "in_near_count": int(in_near),
        "total_roi_centers": int(total),
        "coverage_fraction": round(in_near / float(total), 6) if total else 0.0,
    }


def _spatial_index(centers: Sequence[Mapping[str, Any]], radius_px: float) -> dict[tuple[int, int], list[Mapping[str, Any]]]:
    cell = max(1.0, float(radius_px))
    index: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for center in centers:
        key = (math.floor(float(center["x"]) / cell), math.floor(float(center["y"]) / cell))
        index.setdefault(key, []).append(center)
    return index


def _has_nearby_center(center: Mapping[str, Any], index: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]], radius_px: float) -> bool:
    cell = max(1.0, float(radius_px))
    cx = math.floor(float(center["x"]) / cell)
    cy = math.floor(float(center["y"]) / cell)
    radius_sq = radius_px * radius_px
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for other in index.get((cx + dx, cy + dy), []):
                ox = float(other["x"]) - float(center["x"])
                oy = float(other["y"]) - float(center["y"])
                if ox * ox + oy * oy <= radius_sq:
                    return True
    return False


def _stencil_points_from_annotations(annotations: Mapping[str, Any]) -> list[tuple[float, float]]:
    polygon = ((annotations.get("settings") or {}).get("anatomyStencil") or {}).get("polygon") or []
    points: list[tuple[float, float]] = []
    for point in polygon:
        if isinstance(point, Mapping):
            x, y = _as_float(point.get("x")), _as_float(point.get("y"))
        elif isinstance(point, Sequence) and len(point) >= 2:
            x, y = _as_float(point[0]), _as_float(point[1])
        else:
            x = y = None
        if x is not None and y is not None:
            points.append((float(x), float(y)))
    return points


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


def _distance_to_polygon(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> float:
    best = math.inf
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        best = min(best, _point_segment_distance(point, start, end))
    return best


def _point_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    t = 0.0 if denom <= 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    x, y = ax + t * dx, ay + t * dy
    return math.sqrt((px - x) * (px - x) + (py - y) * (py - y))


def _resolve_artifact_path(app_dir: Path, project_root: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    app_relative = app_dir / path
    if app_relative.exists():
        return app_relative
    root_relative = project_root / path
    if root_relative.exists():
        return root_relative
    return app_relative


def _load_optional_json(path: Path | None) -> tuple[dict[str, Any] | list[Any] | None, Path | None]:
    if path is None or not path.exists():
        return None, path
    return json.loads(path.read_text(encoding="utf-8")), path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _attach_report_to_manifest(architecture_path: Path, app_dir: Path, json_path: Path, markdown_path: Path) -> None:
    manifest = _load_json(architecture_path)
    artifacts = manifest.setdefault("artifacts", {})
    artifacts["sweep_evidence_report"] = _rel_to(json_path, app_dir)
    artifacts["sweep_evidence_markdown"] = _rel_to(markdown_path, app_dir)
    _write_json(architecture_path, manifest)


def _find_project_root(path: Path) -> Path:
    for candidate in [path, *path.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd().resolve()


def _rel_to(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _fmt(value: Any, digits: int = 2) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    if digits <= 0:
        return str(int(round(number)))
    return f"{number:.{digits}f}"
