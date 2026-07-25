"""Bounded dataset/video discovery shared by CLI, workbench, and LLM tools.

The catalog joins the three durable contracts already used by Neurobench:
``dataset_manifest.json``, ``dashboard_manifest.json``, and a workbench
``review_data.json``.  It intentionally does not crawl arbitrary experiment
artifacts or inspect video pixels.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import os
from pathlib import Path
from typing import Any


CATALOG_SCHEMA_VERSION = 1
CATALOG_FILE_NAMES = {
    "dataset_manifest.json",
    "dataset_manifest.generated.json",
    "video_manifest.json",
    "dashboard_manifest.json",
}
DEFAULT_SEARCH_ROOTS = ("Outputs",)
DEFAULT_MAX_DEPTH = 4


def dataset_id_from_review(review_data: Mapping[str, Any], *, fallback: str = "") -> str:
    """Return the canonical dataset identifier from review data."""

    dataset = review_data.get("dataset") if isinstance(review_data.get("dataset"), Mapping) else {}
    parameters = review_data.get("parameters") if isinstance(review_data.get("parameters"), Mapping) else {}
    return str(dataset.get("dataset_id") or review_data.get("dataset_id") or parameters.get("datasetId") or fallback)


def raw_video_from_review(review_data: Mapping[str, Any]) -> str:
    """Return an explicitly declared raw-video path without filesystem guessing."""

    dataset = review_data.get("dataset") if isinstance(review_data.get("dataset"), Mapping) else {}
    dataset_paths = dataset.get("paths") if isinstance(dataset.get("paths"), Mapping) else {}
    source = review_data.get("source") if isinstance(review_data.get("source"), Mapping) else {}
    video = review_data.get("video") if isinstance(review_data.get("video"), Mapping) else {}
    return str(
        dataset.get("raw_video")
        or dataset_paths.get("raw_video")
        or source.get("raw_video")
        or source.get("path")
        or video.get("source_path")
        or ""
    )


def bounded_named_files(root: str | Path, names: Iterable[str], *, max_depth: int = DEFAULT_MAX_DEPTH) -> list[Path]:
    """Find selected filenames while pruning traversal at ``max_depth``."""

    base = Path(root).expanduser()
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if not base.exists() or not base.is_dir():
        return []
    wanted = {str(name) for name in names}
    found: list[Path] = []
    for directory, child_dirs, filenames in os.walk(base):
        current = Path(directory)
        try:
            depth = len(current.relative_to(base).parts)
        except ValueError:
            continue
        child_dirs[:] = sorted(name for name in child_dirs if not name.startswith(".") and name != "__pycache__")
        if depth >= max_depth:
            child_dirs[:] = []
        for name in sorted(wanted.intersection(filenames)):
            found.append(current / name)
    return sorted(found)


def discover_dataset_catalog(
    workspace_root: str | Path,
    *,
    search_roots: Iterable[str | Path] | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> list[dict[str, Any]]:
    """Discover and join canonical dataset records under bounded roots."""

    workspace = Path(workspace_root).expanduser().resolve()
    roots = list(search_roots or DEFAULT_SEARCH_ROOTS)
    files: list[Path] = []
    for root in roots:
        path = Path(root).expanduser()
        if not path.is_absolute():
            path = workspace / path
        files.extend(bounded_named_files(path, CATALOG_FILE_NAMES, max_depth=max_depth))
        files.extend(bounded_named_files(path, {"review_data.json"}, max_depth=max_depth))

    dataset_manifests: list[tuple[Path, dict[str, Any]]] = []
    video_manifests: list[tuple[Path, dict[str, Any]]] = []
    dashboard_manifests: list[tuple[Path, dict[str, Any]]] = []
    review_payloads: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(set(files)):
        payload = _load_json_object(path)
        if payload is None:
            continue
        if path.name.startswith("dataset_manifest") and payload.get("dataset_id"):
            dataset_manifests.append((path, payload))
        elif path.name == "video_manifest.json" and payload.get("dataset_id"):
            video_manifests.append((path, payload))
        elif path.name == "dashboard_manifest.json" and payload.get("dataset_id"):
            dashboard_manifests.append((path, payload))
        elif path.name == "review_data.json" and path.parent.name == "app":
            review_payloads.append((path, payload))

    records: dict[str, dict[str, Any]] = {}
    for path, payload in dataset_manifests:
        dataset_id = str(payload["dataset_id"])
        record = records.setdefault(dataset_id, _empty_record(dataset_id))
        _merge_dataset_manifest(record, path, payload, workspace)
    for path, payload in video_manifests:
        dataset_id = str(payload["dataset_id"])
        record = records.setdefault(dataset_id, _empty_record(dataset_id))
        _merge_video_manifest(record, path, payload, workspace)
    for path, payload in review_payloads:
        dataset_id = dataset_id_from_review(payload, fallback=path.parent.parent.name)
        if not dataset_id:
            continue
        record = records.setdefault(dataset_id, _empty_record(dataset_id))
        _merge_review_data(record, path, payload, workspace)
    for path, payload in dashboard_manifests:
        dataset_id = str(payload["dataset_id"])
        record = records.setdefault(dataset_id, _empty_record(dataset_id))
        _merge_dashboard_manifest(record, path, payload, workspace)

    output = [_finalize_record(record, workspace) for record in records.values()]
    return sorted(output, key=lambda item: (str(item.get("name") or "").lower(), str(item["dataset_id"])))


def query_dataset_catalog(records: Iterable[Mapping[str, Any]], query: str) -> list[dict[str, Any]]:
    """Rank catalog records by exact, prefix, then token containment matches."""

    needle = " ".join(str(query or "").lower().split())
    if not needle:
        return [dict(record) for record in records]
    tokens = needle.split()
    ranked: list[tuple[int, str, dict[str, Any]]] = []
    for source in records:
        record = dict(source)
        dataset_id = str(record.get("dataset_id") or "").lower()
        name = str(record.get("name") or "").lower()
        paths = record.get("paths") if isinstance(record.get("paths"), Mapping) else {}
        videos = record.get("videos") if isinstance(record.get("videos"), list) else []
        video_terms = [
            str(value).lower()
            for video in videos
            if isinstance(video, Mapping)
            for value in (video.get("video_id"), video.get("path"), video.get("label"), video.get("condition"))
            if value is not None
        ]
        logical_views = []
        record_video = record.get("video") if isinstance(record.get("video"), Mapping) else {}
        logical_views.extend(record_video.get("views") or [])
        for video in videos:
            if isinstance(video, Mapping):
                logical_views.extend(video.get("views") or [])
        view_terms = [
            str(value).lower()
            for view in logical_views
            if isinstance(view, Mapping)
            for value in (view.get("view_id"), view.get("label"), view.get("role"))
            if value is not None
        ]
        haystack = " ".join([dataset_id, name, *(str(value).lower() for value in paths.values()), *video_terms, *view_terms])
        if not all(token in haystack for token in tokens):
            continue
        score = 0 if needle == dataset_id else 1 if dataset_id.startswith(needle) else 2 if needle == name else 3 if needle in name else 4
        ranked.append((score, dataset_id, record))
    return [record for _, _, record in sorted(ranked, key=lambda row: (row[0], row[1]))]


def dataset_record_for_app(app_dir: str | Path, *, workspace_root: str | Path | None = None) -> dict[str, Any]:
    """Build one catalog record for a served app without scanning the workspace."""

    app = Path(app_dir).expanduser().resolve()
    workspace = Path(workspace_root).expanduser().resolve() if workspace_root else _infer_workspace_root(app)
    review_path = app / "review_data.json"
    review = _load_json_object(review_path) or {}
    dataset_id = dataset_id_from_review(review, fallback=app.parent.name)
    record = _empty_record(dataset_id)
    for candidate in (app / "dataset_manifest.generated.json", app.parent / "dataset_manifest.json"):
        payload = _load_json_object(candidate)
        if payload and str(payload.get("dataset_id") or dataset_id) == dataset_id:
            _merge_dataset_manifest(record, candidate, payload, workspace)
    if review_path.exists():
        _merge_review_data(record, review_path, review, workspace)
    dashboard_path = app.parent / "dashboard_manifest.json"
    dashboard = _load_json_object(dashboard_path)
    if dashboard and str(dashboard.get("dataset_id") or dataset_id) == dataset_id:
        _merge_dashboard_manifest(record, dashboard_path, dashboard, workspace)
    return _finalize_record(record, workspace)


def llm_catalog_context(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a compact, path-grounded catalog suitable for an LLM handoff."""

    datasets: list[dict[str, Any]] = []
    for record in records:
        video = record.get("video") if isinstance(record.get("video"), Mapping) else {}
        paths = record.get("paths") if isinstance(record.get("paths"), Mapping) else {}
        datasets.append(
            {
                "dataset_id": record.get("dataset_id"),
                "name": record.get("name"),
                "raw_video": paths.get("raw_video", ""),
                "app_dir": paths.get("app_dir", ""),
                "review_data": paths.get("review_data", ""),
                "frames": video.get("frames"),
                "height": video.get("height"),
                "width": video.get("width"),
                "frame_rate_hz": video.get("frame_rate_hz"),
                "views": video.get("views", []),
                "videos": [
                    {
                        key: item.get(key)
                        for key in ("video_id", "path", "label", "condition", "fish_id", "frame_count", "frame_rate_hz", "width", "height", "views")
                        if item.get(key) is not None
                    }
                    for item in record.get("videos", [])
                    if isinstance(item, Mapping)
                ],
                "capabilities": record.get("capabilities", {}),
                "readiness": record.get("readiness", {}),
                "ready": record.get("ready", False),
            }
        )
    return {"schema_version": CATALOG_SCHEMA_VERSION, "kind": "neurobench_dataset_catalog", "datasets": datasets}


def _empty_record(dataset_id: str) -> dict[str, Any]:
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "name": dataset_id,
        "video": {},
        "paths": {},
        "manifests": {},
        "dashboard": {},
        "capabilities": {},
        "videos": [],
    }


def _merge_dataset_manifest(record: dict[str, Any], path: Path, payload: Mapping[str, Any], workspace: Path) -> None:
    record["name"] = str(payload.get("name") or record.get("name") or record["dataset_id"])
    record["modality"] = payload.get("modality", record.get("modality", ""))
    record["indicator"] = payload.get("indicator", record.get("indicator", ""))
    record["pixel_size_microns"] = payload.get("pixel_size_microns", record.get("pixel_size_microns"))
    video = record.setdefault("video", {})
    if payload.get("frame_rate_hz") is not None:
        video["frame_rate_hz"] = payload.get("frame_rate_hz")
    for key, value in dict(payload.get("paths") or {}).items():
        resolved = _resolve_declared_path(value, declaration=path, workspace=workspace)
        record.setdefault("paths", {})[str(key)] = _display_path(resolved, workspace)
    record.setdefault("manifests", {})["dataset"] = _display_path(path, workspace)


def _merge_review_data(record: dict[str, Any], path: Path, payload: Mapping[str, Any], workspace: Path) -> None:
    dataset = payload.get("dataset") if isinstance(payload.get("dataset"), Mapping) else {}
    video_payload = payload.get("video") if isinstance(payload.get("video"), Mapping) else {}
    record["name"] = str(video_payload.get("name") or dataset.get("name") or record.get("name") or record["dataset_id"])
    video = record.setdefault("video", {})
    for source_key, target_key in (
        ("frames", "frames"),
        ("height", "height"),
        ("width", "width"),
        ("frameRateHz", "frame_rate_hz"),
        ("framePattern", "frame_pattern"),
    ):
        if video_payload.get(source_key) is not None:
            video[target_key] = video_payload[source_key]
    if dataset.get("frame_rate_hz") is not None:
        video["frame_rate_hz"] = dataset["frame_rate_hz"]
    if dataset.get("pixel_size_microns") is not None:
        record["pixel_size_microns"] = dataset["pixel_size_microns"]
    if isinstance(video_payload.get("views"), list):
        video["views"] = [dict(view) for view in video_payload["views"] if isinstance(view, Mapping)]
    app = path.parent.resolve()
    paths = record.setdefault("paths", {})
    paths.setdefault("app_dir", _display_path(app, workspace))
    paths["review_data"] = _display_path(path, workspace)
    for name in ("annotations.json", "architecture_runs.json", "index.html"):
        candidate = app / name
        key = {"annotations.json": "annotations", "architecture_runs.json": "architecture_runs", "index.html": "entrypoint"}[name]
        if candidate.exists() or key not in paths:
            paths[key] = _display_path(candidate, workspace)
    explicit_raw = raw_video_from_review(payload)
    if explicit_raw and not paths.get("raw_video"):
        paths["raw_video"] = _display_path(_resolve_declared_path(explicit_raw, declaration=path, workspace=workspace), workspace)
    record["roi_count"] = len(payload.get("rois") or [])
    discovery = payload.get("discovery") if isinstance(payload.get("discovery"), Mapping) else {}
    record["suggestion_count"] = len(discovery.get("suggestions") or [])
    qc = payload.get("qc") if isinstance(payload.get("qc"), Mapping) else {}
    area_stats = qc.get("roiAreaStats") if isinstance(qc.get("roiAreaStats"), Mapping) else {}
    if area_stats.get("median") is not None:
        record["median_roi_area"] = area_stats["median"]


def _merge_video_manifest(record: dict[str, Any], path: Path, payload: Mapping[str, Any], workspace: Path) -> None:
    videos: list[dict[str, Any]] = []
    record_video = record.setdefault("video", {})
    default_frame_rate = payload.get("frame_rate_hz")
    if default_frame_rate is None:
        default_frame_rate = record_video.get("frame_rate_hz")
    for source in payload.get("videos") or []:
        if not isinstance(source, Mapping):
            continue
        item = dict(source)
        if item.get("path"):
            resolved = _resolve_declared_path(item["path"], declaration=path, workspace=workspace)
            item["path"] = _display_path(resolved, workspace)
        if item.get("frame_rate_hz") is None and default_frame_rate is not None:
            item["frame_rate_hz"] = float(default_frame_rate)
        videos.append(item)
    record["videos"] = videos
    record["labels"] = list(payload.get("label_set") or [])
    record["label_counts"] = dict(payload.get("label_counts") or {})
    record["split_policy"] = payload.get("split_policy", "")
    record.setdefault("manifests", {})["video"] = _display_path(path, workspace)
    video = record_video
    video["count"] = len(videos)
    frame_counts = [int(item["frame_count"]) for item in videos if isinstance(item.get("frame_count"), int)]
    if frame_counts:
        video["frames_total"] = sum(frame_counts)
    for key in ("width", "height", "frame_rate_hz"):
        values = {item.get(key) for item in videos if item.get(key) is not None}
        if len(values) == 1:
            video[key] = values.pop()


def _merge_dashboard_manifest(record: dict[str, Any], path: Path, payload: Mapping[str, Any], workspace: Path) -> None:
    dashboard = record.setdefault("dashboard", {})
    for key in ("dashboard_id", "dashboard_type", "serve_command"):
        if payload.get(key) is not None:
            dashboard[key] = payload[key]
    dashboard["manifest"] = _display_path(path, workspace)
    if payload.get("entrypoint"):
        resolved = _resolve_declared_path(payload["entrypoint"], declaration=path, workspace=workspace)
        record.setdefault("paths", {})["entrypoint"] = _display_path(resolved, workspace)
    source = payload.get("source") if isinstance(payload.get("source"), Mapping) else {}
    if source.get("raw_video") and not record.setdefault("paths", {}).get("raw_video"):
        resolved = _resolve_declared_path(source["raw_video"], declaration=path, workspace=workspace)
        record["paths"]["raw_video"] = _display_path(resolved, workspace)


def _finalize_record(record: dict[str, Any], workspace: Path) -> dict[str, Any]:
    paths = record.setdefault("paths", {})
    app_dir = _absolute_from_display(paths.get("app_dir"), workspace)
    entrypoint = _absolute_from_display(paths.get("entrypoint"), workspace)
    if entrypoint is None and app_dir is not None:
        entrypoint = app_dir / "index.html"
        paths["entrypoint"] = _display_path(entrypoint, workspace)
    raw_video = _absolute_from_display(paths.get("raw_video"), workspace)
    video_paths = [
        _absolute_from_display(item.get("path"), workspace)
        for item in record.get("videos", [])
        if isinstance(item, Mapping) and item.get("path")
    ]
    review_data = _absolute_from_display(paths.get("review_data"), workspace)
    annotations = _absolute_from_display(paths.get("annotations"), workspace)
    html_has_cfar = bool(entrypoint and _file_contains(entrypoint, b"cfarMaskAnnotationPanel"))
    record["capabilities"] = {
        "review_app": bool(entrypoint and entrypoint.is_file()),
        "annotations": bool(annotations and annotations.is_file()),
        "cfar_annotation": html_has_cfar,
        "manual_roi_annotation": bool(entrypoint and _file_contains(entrypoint, b'manualRoiMode')),
        "logical_views": bool((record.get("video") or {}).get("views"))
        or any(bool(item.get("views")) for item in record.get("videos", []) if isinstance(item, Mapping)),
        "video_collection": bool(record.get("videos")),
    }
    record["exists"] = {
        "raw_video": bool(raw_video and raw_video.is_file()),
        "review_data": bool(review_data and review_data.is_file()),
        "app_dir": bool(app_dir and app_dir.is_dir()),
        "raw_videos": bool(video_paths) and all(path is not None and path.is_file() for path in video_paths),
    }
    review_ready = bool(record["exists"]["review_data"] and record["capabilities"]["review_app"])
    video_ready = bool(record["exists"]["raw_video"] or record["exists"]["raw_videos"])
    record["readiness"] = {
        "review_ready": review_ready,
        "video_ready": video_ready,
    }
    record["ready"] = review_ready or video_ready
    return record


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_declared_path(value: Any, *, declaration: Path, workspace: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    workspace_candidate = (workspace / path).resolve()
    declaration_candidate = (declaration.parent / path).resolve()
    if workspace_candidate.exists() or not declaration_candidate.exists():
        return workspace_candidate
    return declaration_candidate


def _display_path(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _absolute_from_display(value: Any, workspace: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def _file_contains(path: Path, marker: bytes) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            remainder = b""
            while chunk := handle.read(65536):
                block = remainder + chunk
                if marker in block:
                    return True
                remainder = block[-max(0, len(marker) - 1) :]
    except OSError:
        return False
    return False


def _infer_workspace_root(app_dir: Path) -> Path:
    for candidate in (app_dir, *app_dir.parents):
        if (candidate / "neurobench").is_dir() and (candidate / "Outputs").is_dir():
            return candidate
    return Path.cwd().resolve()
