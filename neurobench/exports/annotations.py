"""Profiled annotation exports for downstream analysis."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

from neurobench.annotations import migrate_annotations_v3
from neurobench.models.exports import ExportBundle


EXPORT_PROFILES = {
    "accepted_only": {
        "description": "Only accepted ROIs and accepted events are exported for downstream modeling.",
        "include_rois": "accepted",
        "include_events": "accepted",
        "review_state_required": ["accepted"],
        "roi_filename": "accepted_rois.tsv",
        "event_filename": "accepted_events.tsv",
    },
    "all_reviewed": {
        "description": "All reviewed ROI and event decisions are exported for audit and adjudication.",
        "include_rois": "reviewed",
        "include_events": "reviewed",
        "review_state_required": ["accepted", "rejected", "unsure"],
        "roi_filename": "reviewed_rois.tsv",
        "event_filename": "reviewed_events.tsv",
    },
    "all_candidates": {
        "description": "Every candidate ROI and event is exported, including unlabeled items.",
        "include_rois": "all",
        "include_events": "all",
        "review_state_required": [],
        "roi_filename": "candidate_rois.tsv",
        "event_filename": "candidate_events.tsv",
    },
}

ROI_COLUMNS = [
    "roi_id",
    "roi_kind",
    "source_roi_ids",
    "cell_state",
    "trace_quality",
    "control_ready",
    "artifact_class",
    "identity_group",
    "needs_action",
    "confidence",
    "reason_tags",
    "notes",
    "reviewer_id",
    "updatedAt",
    "cfar_foreground_px",
    "cfar_background_px",
    "cfar_reference_frames",
]
EVENT_COLUMNS = [
    "roi_id",
    "frame",
    "event_state",
    "event_type",
    "timing_quality",
    "confidence",
    "reason_tags",
    "notes",
    "reviewer_id",
    "updatedAt",
]
SPLIT_MERGE_COLUMNS = [
    "decision_id",
    "decision_type",
    "decision_state",
    "source_roi_ids",
    "target_roi_ids",
    "virtual_roi_id",
    "identity_group",
    "needs_action",
    "confidence",
    "reason_tags",
    "notes",
    "reviewer_id",
    "updatedAt",
]


def export_annotation_profile(
    review_data: Mapping[str, Any],
    annotations: Mapping[str, Any] | None,
    out_dir: str | Path,
    *,
    profile: str = "accepted_only",
    dataset_id: str | None = None,
    run_ids: list[str] | None = None,
    created_at: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> ExportBundle:
    """Write annotation TSVs and an ExportBundle manifest for one profile."""
    if profile not in EXPORT_PROFILES:
        raise ValueError(f"Unknown annotation export profile '{profile}'.")
    policy = EXPORT_PROFILES[profile]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ann = migrate_annotations_v3(annotations)
    _write_text(out / "annotations_v3.json", _json_dumps(ann))

    roi_rows = _roi_rows(review_data, ann, profile)
    event_rows = _event_rows(review_data, ann, profile, accepted_roi_ids={row["roi_id"] for row in roi_rows if row["cell_state"] == "accepted"})
    roi_path = out / str(policy["roi_filename"])
    event_path = out / str(policy["event_filename"])
    metadata_path = out / "neuron_metadata.tsv"
    split_merge_path = out / "split_merge_decisions.tsv"
    split_merge_rows = _split_merge_rows(ann, profile)
    _write_tsv(roi_path, ROI_COLUMNS, roi_rows)
    _write_tsv(event_path, EVENT_COLUMNS, event_rows)
    _write_tsv(metadata_path, ROI_COLUMNS, roi_rows)
    _write_tsv(split_merge_path, SPLIT_MERGE_COLUMNS, split_merge_rows)

    files = [
        _file_record("annotations_v3", out / "annotations_v3.json", "json", "Migrated annotation payload."),
        _file_record("roi_annotations", roi_path, "tsv", str(policy["description"]), rows=len(roi_rows)),
        _file_record("event_annotations", event_path, "tsv", str(policy["description"]), rows=len(event_rows)),
        _file_record("neuron_metadata", metadata_path, "tsv", "ROI metadata table matching the selected export profile.", rows=len(roi_rows)),
        _file_record("split_merge_decisions", split_merge_path, "tsv", "Explicit split/merge review decisions matching the selected export profile.", rows=len(split_merge_rows)),
    ]
    checksums = {item["path"]: item["sha256"] for item in files if item.get("sha256")}
    bundle = ExportBundle(
        schema_version=1,
        export_bundle_id=f"annotation_export_{profile}",
        dataset_id=dataset_id or _dataset_id(review_data),
        run_ids=list(run_ids or _run_ids(review_data)),
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        profile=profile,
        selection_policy={
            "name": profile,
            "description": str(policy["description"]),
            "include_rois": str(policy["include_rois"]),
            "include_events": str(policy["include_events"]),
            "review_state_required": list(policy["review_state_required"]),
        },
        alignment_status="not_provided",
        alignment={"status": "not_provided"},
        files=files,
        checksums=checksums,
        warnings=[] if profile == "accepted_only" else [f"Profile '{profile}' is not limited to accepted downstream modeling targets."],
        provenance=dict(provenance or {}),
    )
    bundle.validate()
    bundle.write_json(out / "export_bundle.json")
    return bundle


def _roi_rows(review_data: Mapping[str, Any], ann: Mapping[str, Any], profile: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for roi in review_data.get("rois", []) or []:
        roi_id = str(roi.get("id"))
        item = dict((ann.get("rois") or {}).get(roi_id, {}))
        state = str(item.get("cell_state") or "").strip().lower()
        if not _include_state(state, profile):
            continue
        rows.append(_roi_row(roi_id, "source", "", item))
    for virtual in (ann.get("virtualRois") or {}).values():
        item = dict(virtual)
        item.update(dict((ann.get("rois") or {}).get(str(item.get("id", "")), {})))
        state = str(item.get("cell_state") or item.get("state") or "").strip().lower()
        if not _include_state(state, profile):
            continue
        rows.append(
            _roi_row(
                str(item.get("id", "")),
                str(item.get("roi_kind", "virtual")),
                ",".join(str(value) for value in item.get("source_roi_ids", []) or []),
                item,
                needs_action=str(item.get("needs_action") or ("merge_needed" if item.get("roi_kind") == "virtual_merge" else "")),
            )
        )
    rows.sort(key=lambda row: row["roi_id"])
    return rows


def _event_rows(review_data: Mapping[str, Any], ann: Mapping[str, Any], profile: str, *, accepted_roi_ids: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for roi in review_data.get("rois", []) or []:
        roi_id = str(roi.get("id"))
        for event in roi.get("events", []) or []:
            frame = str(event.get("frame"))
            key = f"{roi_id}:{frame}"
            item = dict((ann.get("events") or {}).get(key, {}))
            state = str(item.get("event_state") or "").strip().lower()
            if not _include_state(state, profile):
                continue
            if profile == "accepted_only" and roi_id not in accepted_roi_ids:
                continue
            rows.append(
                {
                    "roi_id": _clean(roi_id),
                    "frame": _clean(frame),
                    "event_state": _clean(item.get("event_state", "")),
                    "event_type": _clean(item.get("event_type", "")),
                    "timing_quality": _clean(item.get("timing_quality", "")),
                    "confidence": _clean(item.get("confidence", "")),
                    "reason_tags": _clean(",".join(str(tag) for tag in item.get("reason_tags", []) or [])),
                    "notes": _clean(item.get("notes", "")),
                    "reviewer_id": _clean(item.get("reviewer_id", "")),
                    "updatedAt": _clean(item.get("updatedAt", "")),
                }
            )
    rows.sort(key=lambda row: (row["roi_id"], _as_int(row["frame"])))
    return rows


def _split_merge_rows(ann: Mapping[str, Any], profile: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for decision_id, decision in (ann.get("splitMergeDecisions") or {}).items():
        item = dict(decision)
        state = str(item.get("decision_state") or "").strip().lower()
        if not _include_decision_state(state, profile):
            continue
        rows.append(
            {
                "decision_id": _clean(item.get("id") or decision_id),
                "decision_type": _clean(item.get("decision_type", "")),
                "decision_state": _clean(item.get("decision_state", "")),
                "source_roi_ids": _clean(",".join(str(value) for value in item.get("source_roi_ids", []) or [])),
                "target_roi_ids": _clean(",".join(str(value) for value in item.get("target_roi_ids", []) or [])),
                "virtual_roi_id": _clean(item.get("virtual_roi_id", "")),
                "identity_group": _clean(item.get("identity_group", "")),
                "needs_action": _clean(item.get("needs_action", "")),
                "confidence": _clean(item.get("confidence", "")),
                "reason_tags": _clean(",".join(str(tag) for tag in item.get("reason_tags", []) or [])),
                "notes": _clean(item.get("notes", "")),
                "reviewer_id": _clean(item.get("reviewer_id", "")),
                "updatedAt": _clean(item.get("updatedAt", "")),
            }
        )
    rows.sort(key=lambda row: row["decision_id"])
    return rows


def _roi_row(roi_id: str, roi_kind: str, source_roi_ids: str, item: Mapping[str, Any], *, needs_action: str | None = None) -> dict[str, str]:
    cfar_regions = dict(item.get("cfar_regions") or {})
    foreground_points = cfar_regions.get("foreground_points") or []
    background_points = cfar_regions.get("background_points") or []
    reference_frames = cfar_regions.get("reference_frames") or []
    return {
        "roi_id": _clean(roi_id),
        "roi_kind": _clean(roi_kind),
        "source_roi_ids": _clean(source_roi_ids),
        "cell_state": _clean(item.get("cell_state", item.get("state", ""))),
        "trace_quality": _clean(item.get("trace_quality", "")),
        "control_ready": _clean(item.get("control_ready", "")),
        "artifact_class": _clean(item.get("artifact_class", "")),
        "identity_group": _clean(item.get("identity_group", "")),
        "needs_action": _clean(needs_action if needs_action is not None else item.get("needs_action", "")),
        "confidence": _clean(item.get("confidence", "")),
        "reason_tags": _clean(",".join(str(tag) for tag in item.get("reason_tags", []) or [])),
        "notes": _clean(item.get("notes", "")),
        "reviewer_id": _clean(item.get("reviewer_id", "")),
        "updatedAt": _clean(item.get("updatedAt", "")),
        "cfar_foreground_px": _clean(len(foreground_points)),
        "cfar_background_px": _clean(len(background_points)),
        "cfar_reference_frames": _clean(",".join(str(frame) for frame in reference_frames)),
    }


def _include_state(state: str, profile: str) -> bool:
    if profile == "accepted_only":
        return state == "accepted"
    if profile == "all_reviewed":
        return state in {"accepted", "rejected", "unsure"}
    return True


def _include_decision_state(state: str, profile: str) -> bool:
    if profile == "accepted_only":
        return state == "accepted"
    if profile == "all_reviewed":
        return state in {"accepted", "rejected", "unsure"}
    return True


def _write_tsv(path: Path, columns: list[str], rows: list[Mapping[str, Any]]) -> None:
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(_clean(row.get(column, "")) for column in columns))
    _write_text(path, "\n".join(lines) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _file_record(kind: str, path: Path, file_format: str, description: str, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": kind,
        "path": path.name,
        "format": file_format,
        "sha256": _sha256_file(path),
        "description": description,
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_id(review_data: Mapping[str, Any]) -> str:
    dataset = review_data.get("dataset") or {}
    if isinstance(dataset, Mapping) and dataset.get("dataset_id"):
        return str(dataset["dataset_id"])
    video = review_data.get("video") or {}
    return str(video.get("name") or "unknown_dataset")


def _run_ids(review_data: Mapping[str, Any]) -> list[str]:
    runs = review_data.get("runs")
    if isinstance(runs, list):
        ids = [str(item.get("run_id")) for item in runs if isinstance(item, Mapping) and item.get("run_id")]
        if ids:
            return ids
    run_id = review_data.get("run_id")
    return [str(run_id)] if run_id else []


def _clean(value: Any) -> str:
    return str(value if value is not None else "").replace("\t", " ").replace("\n", " ")


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _json_dumps(payload: Mapping[str, Any]) -> str:
    import json

    return json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
