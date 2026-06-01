"""Workbench build helpers shared by CLI tools and future package entrypoints."""
from __future__ import annotations

import json
import hashlib
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from neurobench.annotations import migrate_annotations_v3
from neurobench.manifests import load_dataset_manifest, load_json, manifest_path
from neurobench.pipeline_catalog import catalog_as_dict


def architecture_runs_from_review(data: Mapping[str, Any], review_data_path: Path, dataset_id: str) -> dict[str, Any]:
    """Create a baseline architecture-run manifest from a review_data payload."""
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "runs": [
            {
                "schema_version": 1,
                "run_id": "current_review_pipeline",
                "dataset_id": dataset_id,
                "label": "Current review pipeline",
                "pipeline": [
                    {"name": "generate_neuron_review_app"},
                    {"name": "trace_event_scoring", "params": {"event_threshold_z": data.get("parameters", {}).get("eventZThreshold")}},
                ],
                "parameters": data.get("parameters", {}),
                "execution": {"status": "completed"},
                "summary": {
                    "roi_count": len(data.get("rois", [])),
                    "event_count": sum(len(roi.get("events", [])) for roi in data.get("rois", [])),
                    "suggestion_count": len(data.get("discovery", {}).get("suggestions", [])),
                    "frame_count": data.get("video", {}).get("frames"),
                },
                "artifacts": {
                    "review_data": str(review_data_path),
                    "app_url": "index.html",
                    "frames": str(review_data_path.parent / "frames"),
                    "intermediates": [],
                    "evidence_maps": data.get("discovery", {}).get("evidenceMaps", []),
                    "roi_summary_tsv": str(review_data_path.parent / "roi_summary.tsv"),
                    "discovery_suggestions_tsv": str(review_data_path.parent / "discovery_suggestions.tsv"),
                },
            }
        ],
    }


def load_workbench_asset(name: str, fallback: str = "") -> str:
    """Load packaged workbench assets, using a provided fallback during migration."""
    try:
        asset = resources.files("neurobench.workbench").joinpath("assets", name)
        return asset.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, ModuleNotFoundError):
        return fallback.strip()


def resolve_build_inputs(
    *,
    app_dir: str | Path | None = None,
    review_data: str | Path | None = None,
    dataset_manifest: str | Path | None = None,
    architecture_runs: str | Path | None = None,
    default_app_dir: str | Path,
    default_review_data: str | Path,
    default_dataset_id: str,
) -> dict[str, Any]:
    """Resolve builder paths from direct args, a dataset manifest, and defaults."""
    manifest = load_dataset_manifest(dataset_manifest) if dataset_manifest else None
    resolved_app_dir = Path(app_dir) if app_dir is not None else None
    resolved_review_data = Path(review_data) if review_data is not None else None
    resolved_architecture_runs = Path(architecture_runs) if architecture_runs is not None else None
    if manifest:
        resolved_app_dir = resolved_app_dir or manifest_path(manifest, "app_dir")
        resolved_review_data = resolved_review_data or manifest_path(manifest, "review_data")
        resolved_architecture_runs = resolved_architecture_runs or manifest_path(manifest, "architecture_runs")
    review_data_path = (resolved_review_data or Path(default_review_data)).resolve()
    review_dataset_id = None
    if review_data_path.exists():
        review_payload = load_json(review_data_path)
        if isinstance(review_payload.get("dataset"), Mapping):
            review_dataset_id = review_payload["dataset"].get("dataset_id")
        review_dataset_id = review_dataset_id or review_payload.get("dataset_id")
    return {
        "dataset_manifest": manifest,
        "app_dir": (resolved_app_dir or Path(default_app_dir)).resolve(),
        "review_data_path": review_data_path,
        "architecture_runs_path": resolved_architecture_runs.resolve() if resolved_architecture_runs else None,
        "dataset_id": (manifest or {}).get("dataset_id") or review_dataset_id or default_dataset_id,
    }


def build_workbench(
    *,
    app_dir: str | Path,
    review_data_path: str | Path,
    dataset_id: str,
    html_template: str,
    dataset_manifest: Mapping[str, Any] | None = None,
    architecture_runs_path: str | Path | None = None,
    css_fallback: str = "",
    js_fallback: str = "",
) -> dict[str, Path]:
    """Build the browser workbench and return generated paths."""
    app_path = Path(app_dir)
    review_path = Path(review_data_path)
    manifest_payload = dict(dataset_manifest or {})
    data = load_json(review_path)
    review_dataset = data.get("dataset") if isinstance(data.get("dataset"), Mapping) else {}
    data["dataset"] = {
        **review_dataset,
        **{key: value for key, value in manifest_payload.items() if not str(key).startswith("_")},
    }
    data["dataset"].setdefault("dataset_id", dataset_id)
    data["pipelineCatalog"] = catalog_as_dict()
    if architecture_runs_path and Path(architecture_runs_path).exists():
        data["architectureRuns"] = load_json(architecture_runs_path)
    else:
        data["architectureRuns"] = architecture_runs_from_review(data, review_path, dataset_id)

    app_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "index": app_path / "index.html",
        "css": app_path / "workbench.css",
        "js": app_path / "workbench.js",
        "annotations": app_path / "annotations.json",
        "architecture_runs": app_path / "architecture_runs.json",
    }
    paths["architecture_runs"].write_text(json.dumps(data["architectureRuns"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    css_text = load_workbench_asset("workbench.css", css_fallback)
    js_text = load_workbench_asset("workbench.js", js_fallback)
    asset_version = hashlib.sha256((css_text + js_text).encode("utf-8")).hexdigest()[:12]
    paths["css"].write_text(css_text + "\n", encoding="utf-8")
    paths["js"].write_text(js_text + "\n", encoding="utf-8")
    html = html_template.format(
        dataset_id=dataset_id,
        frames=data["video"]["frames"],
        asset_version=asset_version,
        data_json=json.dumps(data, separators=(",", ":")).replace("</script>", "<\\/script>"),
    )
    paths["index"].write_text(html, encoding="utf-8")
    if not paths["annotations"].exists():
        annotations = migrate_annotations_v3(None)
    else:
        annotations = migrate_annotations_v3(load_json(paths["annotations"]))
    paths["annotations"].write_text(json.dumps(annotations, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths
