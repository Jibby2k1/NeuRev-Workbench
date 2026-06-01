"""Execution helpers for LLM-proposed Neurobench architecture experiments."""
from __future__ import annotations

import csv
from pathlib import Path
import time
from typing import Any, Mapping

from neurobench.llm_planning import proposal_set_to_architecture_manifest
from neurobench.manifests import load_json, write_json
from neurobench.metrics.detection import object_matching_metrics
from neurobench.pipelines.executor import execute_pipeline
from neurobench.pipelines.sweeps import render_sweep_summary_markdown


def execute_llm_proposal_experiments(
    proposal_set: Mapping[str, Any],
    *,
    run_root: str | Path,
    max_combinations: int | None = None,
    ground_truth_csv: str | Path | None = None,
    centroid_tolerance_px: float = 4.0,
    event_tolerance_frames: int = 2,
    stop_on_error: bool = False,
) -> dict[str, Any]:
    """Execute locally runnable LLM proposal runs and write summary artifacts."""

    manifest = proposal_set_to_architecture_manifest(proposal_set, max_combinations=max_combinations)
    ground_truth = _load_ground_truth(Path(ground_truth_csv) if ground_truth_csv else None)
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    runs = []
    for index, run in enumerate(manifest.get("runs", []) or [], start=1):
        run_id = str(run.get("run_id") or f"llm_run_{index:03d}")
        run_dir = root / f"{index:03d}_{_safe_name(run_id)}"
        record = {
            "run_id": run_id,
            "run_root": _display_path(run_dir, root),
            "status": "planned",
            "proposal_id": (run.get("artifacts") or {}).get("proposal_id", ""),
            "sweep_parameters": list((run.get("sweep") or {}).get("parameters") or []),
        }
        try:
            started = time.perf_counter()
            result = execute_pipeline(run, run_root=run_dir)
            runtime_sec = time.perf_counter() - started
            pipeline_run = result["pipeline_run"]
            record.update(
                {
                    "status": "completed",
                    "artifact_count": len(pipeline_run.get("artifacts", [])),
                    "parameter_hash": pipeline_run.get("parameter_hash", ""),
                    "runtime_sec": round(runtime_sec, 6),
                    "metrics": _run_metrics(
                        pipeline_run,
                        run_dir=run_dir,
                        ground_truth=ground_truth,
                        centroid_tolerance_px=centroid_tolerance_px,
                        event_tolerance_frames=event_tolerance_frames,
                    ),
                }
            )
        except Exception as exc:
            record.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
            if stop_on_error:
                runs.append(record)
                break
        runs.append(record)
    failed = sum(1 for run in runs if run["status"] == "failed")
    summary = {
        "schema_version": 1,
        "dataset_id": manifest.get("dataset_id", ""),
        "proposal_set_id": proposal_set.get("proposal_set_id", ""),
        "status": "completed" if failed == 0 else "completed_with_failures",
        "total": len(runs),
        "succeeded": len(runs) - failed,
        "failed": failed,
        "metric_overview": _metric_overview(runs),
        "runs": runs,
    }
    write_json(root / "llm_experiment_summary.json", summary)
    (root / "llm_experiment_report.md").write_text(render_sweep_summary_markdown(summary), encoding="utf-8")
    return summary


def _run_metrics(
    pipeline_run: Mapping[str, Any],
    *,
    run_dir: Path,
    ground_truth: dict[str, Any],
    centroid_tolerance_px: float,
    event_tolerance_frames: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    candidate_events: list[dict[str, Any]] = []
    for artifact in pipeline_run.get("artifacts", []) or []:
        kind = artifact.get("kind")
        summary = artifact.get("summary") or {}
        if kind == "candidate_mask":
            metrics["active_fraction"] = summary.get("active_fraction")
        if kind == "roi_candidates":
            metrics["candidate_count"] = summary.get("count")
            candidates = _load_candidates(run_dir, artifact)
        if kind == "candidate_events":
            metrics["event_count"] = summary.get("event_count")
            candidate_events = _load_candidate_events(run_dir, artifact)
    metrics.setdefault("candidate_count", len(candidates))
    metrics.setdefault("event_count", len(candidate_events))
    if ground_truth.get("objects"):
        object_metrics = object_matching_metrics(
            ground_truth["objects"],
            candidates,
            iou_threshold=1.0,
            centroid_tolerance_px=centroid_tolerance_px,
        )
        metrics.update(
            {
                "object_count_gt": object_metrics["object_count_gt"],
                "object_precision": object_metrics["object_precision"],
                "object_recall": object_metrics["object_recall"],
                "object_tp": object_metrics["TP"],
                "object_fp": object_metrics["FP"],
                "object_fn": object_metrics["FN"],
                "mean_matched_centroid_distance_px": object_metrics["mean_matched_centroid_distance_px"],
            }
        )
    if ground_truth.get("events"):
        metrics.update(_event_onset_metrics(ground_truth["events"], candidate_events, tolerance_frames=event_tolerance_frames))
    return metrics


def _load_candidates(run_dir: Path, artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = _load_artifact_json(run_dir, artifact)
    rows = payload.get("ranked_candidates") or payload.get("candidates") or []
    candidates = []
    for index, row in enumerate(rows):
        candidate = dict(row)
        candidate.setdefault("id", candidate.get("candidate_id") or candidate.get("roi_id") or f"candidate_{index + 1:03d}")
        if "centroid" not in candidate and "x" in candidate and "y" in candidate:
            candidate["centroid"] = [candidate["x"], candidate["y"]]
        candidates.append(candidate)
    return candidates


def _load_candidate_events(run_dir: Path, artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = _load_artifact_json(run_dir, artifact)
    return [dict(item) for item in payload.get("events", []) or []]


def _load_artifact_json(run_dir: Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    raw_path = Path(str(artifact.get("path") or ""))
    path = raw_path if raw_path.is_absolute() else run_dir / raw_path
    if not path.is_file():
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def _load_ground_truth(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"objects": [], "events": []}
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth CSV does not exist: {path}")
    objects_by_id: dict[str, dict[str, Any]] = {}
    events = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, raw_row in enumerate(reader):
            row = {str(key).strip(): value for key, value in raw_row.items()}
            object_id = str(row.get("ID") or row.get("id") or row.get("object_id") or f"gt_{index + 1:03d}")
            x = float(row.get("X") or row.get("x") or row.get("centroid_x") or 0.0)
            y = float(row.get("Y") or row.get("y") or row.get("centroid_y") or 0.0)
            start = int(float(row.get("Start Frame") or row.get("start_frame") or row.get("frame") or 0))
            end = int(float(row.get("End Frame") or row.get("end_frame") or start))
            objects_by_id.setdefault(object_id, {"id": object_id, "centroid": [x, y], "x": x, "y": y})
            events.append({"id": object_id, "start_frame": start, "end_frame": end, "x": x, "y": y})
    return {"objects": list(objects_by_id.values()), "events": events}


def _event_onset_metrics(ground_truth_events: list[Mapping[str, Any]], candidate_events: list[Mapping[str, Any]], *, tolerance_frames: int) -> dict[str, Any]:
    candidate_frames = [int(event.get("frame", -10**9)) for event in candidate_events if event.get("frame") is not None]
    matched = 0
    for event in ground_truth_events:
        start = int(event.get("start_frame", 0))
        end = int(event.get("end_frame", start))
        if any(start - tolerance_frames <= frame <= end + tolerance_frames for frame in candidate_frames):
            matched += 1
    gt_count = len(ground_truth_events)
    event_count = len(candidate_frames)
    return {
        "event_count_gt": gt_count,
        "event_onset_tp": matched,
        "event_onset_fn": max(0, gt_count - matched),
        "event_onset_recall": matched / gt_count if gt_count else 0.0,
        "event_burden_ratio": event_count / gt_count if gt_count else None,
    }


def _metric_overview(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [run for run in runs if run.get("status") == "completed"]
    if not completed:
        return {}

    def metric_value(run: Mapping[str, Any], key: str, default: float = 0.0) -> float:
        value = (run.get("metrics") or {}).get(key, default)
        return float(value) if value is not None else default

    best_by_recall = max(
        completed,
        key=lambda run: (
            metric_value(run, "object_recall"),
            metric_value(run, "event_onset_recall"),
            -metric_value(run, "candidate_count", 10**9),
        ),
    )
    lowest_burden = min(completed, key=lambda run: (metric_value(run, "candidate_count", 10**9), metric_value(run, "event_count", 10**9)))
    return {
        "best_recall_run_id": best_by_recall.get("run_id"),
        "lowest_burden_run_id": lowest_burden.get("run_id"),
        "completed_run_count": len(completed),
        "mean_candidate_count": sum(metric_value(run, "candidate_count") for run in completed) / len(completed),
        "mean_event_count": sum(metric_value(run, "event_count") for run in completed) / len(completed),
    }


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value).strip("._-") or "run"


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())
