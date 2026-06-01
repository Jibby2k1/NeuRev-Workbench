"""Readable summaries for small Gamma CFAR parameter sweeps."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


EXPECTED_DIAMETER_UM = (5.0, 10.0)


def summarize_gamma_cfar_sweep(
    sweep_root: str | Path,
    *,
    pixel_size_um: float | None = 0.5,
    top_n: int = 5,
    size_mode: str = "microns",
) -> dict[str, Any]:
    """Return an interpretable summary for a locally executed Gamma CFAR sweep."""
    if size_mode not in {"microns", "pixels"}:
        raise ValueError("size_mode must be 'microns' or 'pixels'.")
    if size_mode == "pixels":
        pixel_size_um = None
    root = Path(sweep_root).expanduser().resolve()
    sweep_summary = _load_json(root / "sweep_summary.json")
    rows = [_summarize_run(root, run, pixel_size_um=pixel_size_um, top_n=top_n) for run in sweep_summary.get("runs", [])]
    completed = [row for row in rows if row["status"] == "completed"]
    return {
        "schema_version": 1,
        "sweep_root": str(root),
        "dataset_id": sweep_summary.get("dataset_id", ""),
        "sweep": sweep_summary.get("sweep", {}),
        "status": sweep_summary.get("status", ""),
        "run_count": len(rows),
        "completed_count": len(completed),
        "failed_count": len(rows) - len(completed),
        "size_mode": size_mode,
        "pixel_size_um": float(pixel_size_um) if pixel_size_um is not None else None,
        "expected_diameter_um": list(EXPECTED_DIAMETER_UM) if pixel_size_um is not None else None,
        "recommended_runs": _recommended_runs(completed),
        "runs": rows,
    }


def render_gamma_cfar_sweep_markdown(summary: Mapping[str, Any]) -> str:
    """Render a concise lab-readable Markdown brief."""
    pixel_size_um = summary.get("pixel_size_um")
    pixel_units = pixel_size_um is None or summary.get("size_mode") == "pixels"
    lines = [
        "# Gamma CFAR Grid Brief",
        "",
        f"- Dataset: `{summary.get('dataset_id', '')}`",
        f"- Sweep: `{dict(summary.get('sweep') or {}).get('id', '')}`",
        f"- Status: `{summary.get('status', '')}`",
        f"- Runs: {summary.get('completed_count', 0)} completed, {summary.get('failed_count', 0)} failed",
    ]
    if pixel_units:
        lines.append("- Pixel size: unknown; ROI size metrics are reported in pixels")
    else:
        lines.extend([
            f"- Pixel size: {float(pixel_size_um):.3g} um/px",
            "- Expected hindbrain neuron diameter: 5-10 um",
        ])
    lines.extend([
        "",
        "## What This Sweep Tests",
        "",
        "This is a cascaded Gamma CFAR detector screen. The first CFAR stage is permissive and local; the second CFAR stage uses a larger reference region and intersects with the first mask to suppress broad clustered background. Component Extraction then removes very small impulse components and oversized merged regions. Heuristic Priority Ranking orders the remaining candidates for review but does not delete them.",
        "",
        "## Recommended Inspection Order",
        "",
    ])
    recommendations = list(summary.get("recommended_runs") or [])
    if recommendations:
        for item in recommendations:
            size_text = (
                f"median_diameter_px={_fmt_float(item.get('median_equivalent_diameter_px'))}"
                if pixel_units
                else f"median_diameter={_fmt_float(item.get('median_equivalent_diameter_um'))} um, plausible_size_fraction={_fmt_float(item.get('plausible_size_fraction'))}"
            )
            lines.append(
                "- "
                + f"`{item['run_id']}`: candidates={item['candidate_count']}, "
                + f"final_active_fraction={_fmt_float(item['final_active_fraction'])}, "
                + f"support_min_frames={_fmt_float(item.get('component_support_min_frames'))}, "
                + size_text
            )
    else:
        lines.append("- No completed run had candidates to inspect.")
    lines.extend(["", "## Run Table", ""])
    if pixel_units:
        lines.extend([
            "| Run | CFAR pfa | Large ref | Min area | Support frames | Candidates | Final active frac | Median area px | Median diameter px | Events |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
    else:
        lines.extend([
            "| Run | CFAR pfa | Large ref | Min area | Support frames | Candidates | Final active frac | Median diameter | Plausible size | Events |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
    for row in summary.get("runs", []) or []:
        params = dict(row.get("parameters") or {})
        size_columns = (
            [_fmt_float(row.get("median_area_px")), _fmt_float(row.get("median_equivalent_diameter_px"))]
            if pixel_units
            else [f"{_fmt_float(row.get('median_equivalent_diameter_um'))} um", _fmt_float(row.get("plausible_size_fraction"))]
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('run_id', '')}`",
                    _fmt_float(params.get("cfar_small_ref.pfa")),
                    _fmt_float(params.get("cfar_large_ref.training_radius_px")),
                    _fmt_float(params.get("components.min_area_px", row.get("component_min_area_px"))),
                    _fmt_float(params.get("components.support_min_frames", row.get("component_support_min_frames"))),
                    str(row.get("candidate_count", 0)),
                    _fmt_float(row.get("final_active_fraction")),
                    *size_columns,
                    str(row.get("event_count", 0)),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Top Candidates From Recommended Runs", ""])
    for item in recommendations[:3]:
        top = list(item.get("top_ranked_candidates") or [])
        lines.append(f"### `{item['run_id']}`")
        if not top:
            lines.append("")
            lines.append("No ranked candidates were generated.")
            lines.append("")
            continue
        for candidate in top:
            reasons = ", ".join(candidate.get("reasons") or []) or "no reasons recorded"
            lines.append(
                f"- rank {candidate.get('rank')}: `{candidate.get('candidate_id')}` "
                f"score={_fmt_float(candidate.get('priority_score'))}; {reasons}"
            )
        lines.append("")
    lines.extend([
        "## Interpretation Notes",
        "",
        "- These are not accuracy numbers because no ground-truth labels are being used here.",
        "- A higher `pfa` is more permissive and should usually increase candidate burden.",
        "- Larger CFAR reference regions should better estimate broad background, but can suppress candidates in locally clustered regions.",
        "- `support_min_frames` is the main impulse-noise control in this CFAR-first architecture.",
        "- `min_area_px` is a secondary morphology gate that removes tiny spatial components after temporal support is applied.",
    ])
    if pixel_units:
        lines.append("- Physical cell-size interpretation is pending pixel-size metadata; median equivalent diameter is reported in pixels only.")
    else:
        lines.append("- Median equivalent diameter is computed from component area and compared to the 5-10 um expected cell-diameter range.")
    return "\n".join(lines).rstrip() + "\n"


def write_gamma_cfar_sweep_brief(
    sweep_root: str | Path,
    output: str | Path,
    *,
    pixel_size_um: float | None = 0.5,
    top_n: int = 5,
    size_mode: str = "microns",
) -> dict[str, Any]:
    """Write JSON and Markdown summaries for a Gamma CFAR sweep."""
    summary = summarize_gamma_cfar_sweep(sweep_root, pixel_size_um=pixel_size_um, top_n=top_n, size_mode=size_mode)
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_gamma_cfar_sweep_markdown(summary), encoding="utf-8")
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _summarize_run(root: Path, run: Mapping[str, Any], *, pixel_size_um: float | None, top_n: int) -> dict[str, Any]:
    run_root = _run_root(root, run)
    parameters = {f"{item.get('stage')}.{item.get('param')}": item.get("value") for item in run.get("sweep_parameters", [])}
    row: dict[str, Any] = {
        "run_id": str(run.get("run_id", "")),
        "status": str(run.get("status", "")),
        "run_root": str(run_root),
        "parameters": parameters,
        "candidate_count": 0,
        "event_count": 0,
        "small_active_fraction": None,
        "final_active_fraction": None,
        "median_area_px": None,
        "median_equivalent_diameter_px": None,
        "median_equivalent_diameter_um": None,
        "plausible_size_fraction": None,
        "component_min_area_px": None,
        "component_support_min_frames": None,
        "top_ranked_candidates": [],
    }
    if row["status"] != "completed":
        row["error"] = run.get("error", "")
        return row
    manifest_path = run_root / "pipeline_run.json"
    if not manifest_path.exists():
        row["status"] = "missing_manifest"
        return row
    manifest = _load_json(manifest_path)
    artifacts = list(manifest.get("artifacts") or [])
    mask_artifacts = [artifact for artifact in artifacts if artifact.get("kind") == "candidate_mask"]
    if mask_artifacts:
        row["small_active_fraction"] = _as_float(mask_artifacts[0].get("summary", {}).get("active_fraction"))
        row["final_active_fraction"] = _as_float(mask_artifacts[-1].get("summary", {}).get("active_fraction"))
    candidate_artifact = _last_artifact(artifacts, "roi_candidates")
    if candidate_artifact:
        candidate_summary = dict(candidate_artifact.get("summary") or {})
        row["candidate_count"] = int(_as_float(candidate_summary.get("count"), 0.0))
        row["component_min_area_px"] = _as_float(candidate_summary.get("min_area_px"))
        row["component_support_min_frames"] = _as_float(candidate_summary.get("support_min_frames"))
        candidates = _artifact_payload(run_root, candidate_artifact).get("candidates", [])
        row.update(_candidate_size_summary(candidates, pixel_size_um=pixel_size_um))
    event_artifact = _last_artifact(artifacts, "candidate_events")
    if event_artifact:
        row["event_count"] = int(_as_float(event_artifact.get("summary", {}).get("event_count"), 0.0))
    ranked_artifact = _last_artifact(artifacts, "ranked_candidates")
    if ranked_artifact:
        ranked = _artifact_payload(run_root, ranked_artifact).get("ranked_candidates", [])
        row["top_ranked_candidates"] = [_ranked_summary(item) for item in ranked[:top_n]]
    return row


def _candidate_size_summary(candidates: Sequence[Mapping[str, Any]], *, pixel_size_um: float | None) -> dict[str, Any]:
    areas = [_as_float(candidate.get("area_px")) for candidate in candidates if _as_float(candidate.get("area_px")) > 0]
    if not areas:
        return {
            "median_area_px": None,
            "median_equivalent_diameter_px": None,
            "median_equivalent_diameter_um": None,
            "plausible_size_fraction": 0.0,
        }
    diameters_px = [math.sqrt(4.0 * area / math.pi) for area in areas]
    result: dict[str, Any] = {
        "median_area_px": round(float(median(areas)), 6),
        "median_equivalent_diameter_px": round(float(median(diameters_px)), 6),
        "median_equivalent_diameter_um": None,
        "plausible_size_fraction": None,
    }
    if pixel_size_um is None:
        return result
    diameters_um = [diameter * pixel_size_um for diameter in diameters_px]
    low, high = EXPECTED_DIAMETER_UM
    plausible = sum(1 for diameter in diameters_um if low <= diameter <= high)
    result["median_equivalent_diameter_um"] = round(float(median(diameters_um)), 6)
    result["plausible_size_fraction"] = round(plausible / float(len(diameters_um)), 6)
    return result


def _recommended_runs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates = [row for row in rows if int(row.get("candidate_count") or 0) > 0]
    ranked = sorted(candidates, key=_recommendation_key)
    return [dict(row) for row in ranked[:3]]


def _recommendation_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
    candidate_count = float(row.get("candidate_count") or 0.0)
    plausible = float(row.get("plausible_size_fraction") or 0.0)
    active_fraction = float(row.get("final_active_fraction") or 0.0)
    burden_penalty = abs(candidate_count - 80.0) / 80.0
    mask_penalty = max(0.0, active_fraction - 0.08) * 8.0
    return (burden_penalty + mask_penalty - plausible, mask_penalty, burden_penalty)


def _ranked_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(item.get("candidate_id", "")),
        "rank": int(_as_float(item.get("rank"), 0.0)),
        "priority_score": _as_float(item.get("priority_score")),
        "reasons": list(item.get("reasons") or []),
    }


def _run_root(root: Path, run: Mapping[str, Any]) -> Path:
    value = Path(str(run.get("run_root") or ""))
    if value.is_absolute():
        return value
    return (root / value).resolve()


def _last_artifact(artifacts: Sequence[Mapping[str, Any]], kind: str) -> Mapping[str, Any] | None:
    matches = [artifact for artifact in artifacts if artifact.get("kind") == kind]
    return matches[-1] if matches else None


def _artifact_payload(run_root: Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(artifact.get("path", "")))
    if not path.is_absolute():
        path = run_root / path
    if not path.exists() or path.suffix.lower() != ".json":
        return {}
    return _load_json(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _fmt_float(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    if abs(number) < 0.001 and number != 0:
        return f"{number:.2e}"
    return f"{number:.4g}"
