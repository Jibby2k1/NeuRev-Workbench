"""Annotation schema migration helpers."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


ROI_STATE_MAP = {"accept": "accepted", "reject": "rejected", "unsure": "unsure", "": ""}
EVENT_STATE_MAP = {"accept": "accepted", "reject": "rejected", "unsure": "unsure", "": ""}
CONFIDENCE_VALUES = {"", "low", "medium", "high"}
ANNOTATION_TASKS = {"neuron_validation", "missed_neuron_search", "event_validation", "artifact_resolution", "exhaustive_tile", "signal_background"}
LEGACY_TASK_MAP = {
    "fast_triage": "neuron_validation",
    "validate_neurons": "neuron_validation",
    "event_validation": "event_validation",
    "missed_neuron_search": "missed_neuron_search",
    "find_missed_neurons": "missed_neuron_search",
    "artifact_cleanup": "artifact_resolution",
    "clean_artifacts": "artifact_resolution",
    "mask_editing": "signal_background",
}


def default_annotations_v3() -> dict[str, Any]:
    return {
        "schema_version": 3,
        "updatedAt": None,
        "rois": {},
        "events": {},
        "suggestions": {},
        "promotedRois": {},
        "virtualRois": {},
        "splitMergeDecisions": {},
        "reviewStats": {"sessionStartedAt": None, "lastActionAt": None, "actions": {}},
        "settings": {},
    }


def migrate_annotations_v3(incoming: Mapping[str, Any] | None) -> dict[str, Any]:
    src = deepcopy(dict(incoming or {}))
    migrated = default_annotations_v3()
    migrated.update({k: v for k, v in src.items() if k not in {"version", "schema_version", "rois", "events", "suggestions"}})
    migrated["schema_version"] = 3
    migrated.pop("version", None)

    migrated["rois"] = {}
    for roi_id, ann in dict(src.get("rois", {})).items():
        item = dict(ann or {})
        state = item.get("cell_state", ROI_STATE_MAP.get(item.get("state", ""), item.get("state", "")))
        item.setdefault("state", item.get("state", ""))
        item.setdefault("cell_state", state)
        item.setdefault("trace_quality", "")
        item.setdefault("control_ready", "")
        item.setdefault("artifact_class", "")
        item.setdefault("identity_group", "")
        item.setdefault("needs_action", "")
        item["reason_tags"] = _normalize_string_list(item.get("reason_tags", item.get("reason_codes", [])))
        item["confidence"] = _normalize_confidence(item.get("confidence", ""))
        _strip_cfar_edit_history(item)
        migrated["rois"][str(roi_id)] = item

    migrated["events"] = {}
    for event_id, ann in dict(src.get("events", {})).items():
        item = dict(ann or {})
        state = item.get("event_state", EVENT_STATE_MAP.get(item.get("state", ""), item.get("state", "")))
        item.setdefault("state", item.get("state", ""))
        item.setdefault("event_state", state)
        item.setdefault("event_type", "")
        item.setdefault("timing_quality", "")
        item["reason_tags"] = _normalize_string_list(item.get("reason_tags", item.get("reason_codes", [])))
        item["confidence"] = _normalize_confidence(item.get("confidence", ""))
        migrated["events"][str(event_id)] = item

    migrated["suggestions"] = {}
    for suggestion_id, ann in dict(src.get("suggestions", {})).items():
        item = dict(ann or {})
        item["reason_tags"] = _normalize_string_list(item.get("reason_tags", item.get("reason_codes", [])))
        item["confidence"] = _normalize_confidence(item.get("confidence", ""))
        migrated["suggestions"][str(suggestion_id)] = item
    migrated["promotedRois"] = dict(src.get("promotedRois", {}))
    migrated["virtualRois"] = {}
    for roi_id, ann in dict(src.get("virtualRois", {})).items():
        item = dict(ann or {})
        item["reason_tags"] = _normalize_string_list(item.get("reason_tags", item.get("reason_codes", [])))
        item["confidence"] = _normalize_confidence(item.get("confidence", ""))
        migrated["virtualRois"][str(roi_id)] = item
    migrated["splitMergeDecisions"] = {}
    for decision_id, ann in dict(src.get("splitMergeDecisions", {})).items():
        item = _normalize_split_merge_decision(ann)
        migrated["splitMergeDecisions"][str(decision_id)] = item
    migrated["reviewStats"] = dict(src.get("reviewStats", migrated["reviewStats"]))
    migrated["reviewStats"]["actions"] = dict(migrated["reviewStats"].get("actions", {}))
    migrated["settings"] = dict(src.get("settings", {}))
    previous_task = str(migrated["settings"].get("annotationTask") or "")
    legacy_preset = str(migrated["settings"].get("reviewWorkflowPreset") or "")
    task = previous_task if previous_task in ANNOTATION_TASKS else LEGACY_TASK_MAP.get(legacy_preset, "neuron_validation")
    migrated["settings"]["annotationTask"] = task
    migrated["settings"]["researchToolsEnabled"] = bool(migrated["settings"].get("researchToolsEnabled", False))
    migrated["ui_migration"] = {
        "schema_version": 1,
        "annotation_task": task,
        "source_review_workflow_preset": legacy_preset,
        "research_tools_enabled": migrated["settings"]["researchToolsEnabled"],
    }
    # Layout v1 could silently duplicate the same source frame as two panes.
    # Move old files to the annotation-safe single canvas while preserving an
    # explicit comparison choice made after this migration.
    if int(migrated["settings"].get("reviewLayoutVersion", 0) or 0) < 2:
        migrated["settings"]["reviewSideBySide"] = False
    migrated["settings"]["reviewLayoutVersion"] = 2
    for bucket in dict(migrated.get("runs", {})).values():
        if not isinstance(bucket, Mapping):
            continue
        for item in dict(bucket.get("rois", {})).values():
            if isinstance(item, dict):
                _strip_cfar_edit_history(item)
    return migrated


def _strip_cfar_edit_history(annotation: dict[str, Any]) -> None:
    regions = annotation.get("cfar_regions")
    if not isinstance(regions, Mapping):
        return
    cleaned = dict(regions)
    cleaned.pop("edit_history", None)
    annotation["cfar_regions"] = cleaned


def _normalize_string_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",")]
        return [part for part in parts if part]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_confidence(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in CONFIDENCE_VALUES else ""


def _normalize_split_merge_decision(value: Any) -> dict[str, Any]:
    item = dict(value or {})
    decision_type = str(item.get("decision_type") or item.get("type") or "").strip().lower()
    item["decision_type"] = decision_type if decision_type in {"split", "merge"} else ""
    state = str(item.get("decision_state") or item.get("state") or "").strip().lower()
    item["decision_state"] = state if state in {"", "proposed", "accepted", "rejected", "unsure"} else ""
    item["source_roi_ids"] = _normalize_string_list(item.get("source_roi_ids", item.get("source_rois", [])))
    item["target_roi_ids"] = _normalize_string_list(item.get("target_roi_ids", item.get("target_rois", [])))
    item.setdefault("virtual_roi_id", "")
    item.setdefault("identity_group", "")
    item.setdefault("needs_action", "")
    item["reason_tags"] = _normalize_string_list(item.get("reason_tags", item.get("reason_codes", [])))
    item["confidence"] = _normalize_confidence(item.get("confidence", ""))
    item.setdefault("notes", "")
    return item
