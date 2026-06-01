#!/usr/bin/env python3
"""Build lightweight review ROI summaries and trace shards for workbench runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurobench.workbench.roi_payloads import stencil_points_from_annotations, write_review_roi_sidecars


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rel_to(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_app_path(app_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return app_dir / path


def review_rois_from_payload(payload: Mapping[str, Any] | list[Any]) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    rows = payload.get("review_rois") or payload.get("rois") or payload.get("candidates") or []
    return [row for row in rows if isinstance(row, Mapping)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, required=True, help="Workbench app directory containing architecture_runs.json.")
    parser.add_argument("--architecture-runs", type=Path, help="Architecture run manifest. Defaults to app-dir/architecture_runs.json.")
    parser.add_argument("--annotations", type=Path, help="Annotations file with saved anatomy stencil. Defaults to app-dir/annotations.json.")
    parser.add_argument("--event-threshold-z", type=float, help="Fallback event threshold for trace-derived summary events.")
    args = parser.parse_args(argv)

    app_dir = args.app_dir.resolve()
    architecture_runs_path = (args.architecture_runs or app_dir / "architecture_runs.json").resolve()
    annotations_path = (args.annotations or app_dir / "annotations.json").resolve()
    review_data_path = app_dir / "review_data.json"
    if not architecture_runs_path.exists():
        raise SystemExit(f"Missing architecture run manifest: {architecture_runs_path}")
    manifest = load_json(architecture_runs_path)
    review_data = load_json(review_data_path) if review_data_path.exists() else {}
    annotations = load_json(annotations_path) if annotations_path.exists() else {}
    stencil_points = stencil_points_from_annotations(annotations)
    event_threshold_z = float(
        args.event_threshold_z
        if args.event_threshold_z is not None
        else (review_data.get("parameters") or {}).get("eventZThreshold", 2.4)
    )
    frame_count = int((review_data.get("video") or {}).get("frames") or 0) or None
    changed = 0
    results: list[dict[str, Any]] = []
    for run in manifest.get("runs", []) or []:
        if not isinstance(run, dict):
            continue
        artifacts = run.setdefault("artifacts", {})
        source = resolve_app_path(app_dir, artifacts.get("review_rois_file"))
        if source is None or not source.exists():
            continue
        payload = load_json(source)
        rois = review_rois_from_payload(payload)
        if not rois:
            continue
        run_dir = source.parent
        summary_path = run_dir / "review_rois.summary.json"
        shard_dir = run_dir / "roi_trace_shards"
        gap_path = run_dir / "stencil_gap_report.json"
        sidecars = write_review_roi_sidecars(
            rois,
            summary_path=summary_path,
            shard_dir=shard_dir,
            run_id=str(run.get("run_id") or source.parent.name),
            frame_count=int(payload.get("frame_count") or frame_count or 0) or None,
            event_threshold_z=event_threshold_z,
            stencil_points=stencil_points,
            gap_report_path=gap_path,
        )
        artifacts["review_rois_summary_file"] = rel_to(summary_path, app_dir)
        artifacts["review_trace_shards_dir"] = rel_to(shard_dir, app_dir)
        artifacts["stencil_gap_report_file"] = rel_to(gap_path, app_dir)
        artifacts["review_roi_payload_version"] = "summary_shards_v1"
        changed += 1
        results.append(
            {
                "run_id": run.get("run_id"),
                "roi_count": sidecars["roi_count"],
                "trace_shard_count": sidecars["trace_shard_count"],
                "gap_count": sidecars["gap_count"],
            }
        )
    if changed:
        write_json(architecture_runs_path, manifest)
    print(json.dumps({"updated_runs": changed, "runs": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
