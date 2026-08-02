"""Adjudication-table contract and label-view construction."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


ADJUDICATION_FIELDS = (
    "schema_version", "observation_id", "burst_id", "original_roi_id",
    "canonical_roi_id", "x_px", "y_px", "original_start_frame_ui",
    "original_end_frame_ui", "event_onset_ui", "event_peak_ui",
    "event_end_ui", "neuron_confidence", "activity_confidence",
    "morphology", "context", "disposition", "include_confirmed",
    "include_inclusive", "review_status", "reviewer_id", "reviewed_at",
    "source_note", "merge_reason",
)

DISPOSITIONS = {
    "confirmed_neuron", "activity_visible_identity_uncertain", "artifact",
    "background", "unresolved", "pending_review",
}
REVIEW_STATUSES = {
    "legacy_source", "pending", "provisional_expert_note", "adjudicated",
}
NEURON_CONFIDENCE = {
    "legacy_label", "confirmed", "probable", "uncertain", "not_neuron",
    "unreviewed",
}
ACTIVITY_CONFIDENCE = {
    "legacy_label", "visible", "probable", "uncertain", "not_visible",
    "unreviewed",
}


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _truth(value: Any) -> bool:
    text = str(value).strip().lower()
    if text not in {"true", "false"}:
        raise ValueError(f"expected true/false, got {value!r}")
    return text == "true"


def observation_id(row: Mapping[str, Any]) -> str:
    return f"b{int(row['burst_id']):02d}__{row['roi_identity']}"


def _seed_override(row: Mapping[str, Any]) -> dict[str, str]:
    """Encode Yinong's targeted notes as provisional, never final, decisions."""
    burst = int(row["burst_id"])
    roi = str(row["roi_identity"])
    common = {
        "review_status": "provisional_expert_note",
        "reviewer_id": "Yinong",
    }
    if burst == 1 and roi == "roi_007":
        return {**common, "neuron_confidence": "uncertain",
                "activity_confidence": "visible", "morphology": "no_boundary",
                "disposition": "unresolved", "include_confirmed": "false",
                "include_inclusive": "true",
                "source_note": "Lights up, but cellular structure is not visible; reviewer recommends cautious exclusion."}
    if burst == 1 and roi in {"roi_008", "roi_010"}:
        return {**common, "neuron_confidence": "confirmed",
                "activity_confidence": "visible", "morphology": "unreviewed",
                "disposition": "confirmed_neuron", "include_confirmed": "true",
                "include_inclusive": "true",
                "source_note": "Reviewer reports successful detection and visible activation."}
    if burst == 1 and roi == "roi_014":
        return {**common, "neuron_confidence": "confirmed",
                "activity_confidence": "visible", "morphology": "round_center",
                "disposition": "confirmed_neuron", "include_confirmed": "true",
                "include_inclusive": "true",
                "source_note": "Reviewer sees activation and a round neuron-like structure."}
    if burst == 1 and roi == "roi_015":
        return {**common, "canonical_roi_id": "roi_010",
                "neuron_confidence": "probable", "activity_confidence": "visible",
                "morphology": "overlap", "context": "overlapping",
                "disposition": "confirmed_neuron", "include_confirmed": "true",
                "include_inclusive": "true",
                "source_note": "Reviewer reports substantial overlap with ROI 010 and suggests merging.",
                "merge_reason": "ROI centers are 3.171 px apart, inside the frozen 6 px matching/NMS radius."}
    if burst == 2 and roi in {"roi_010", "roi_014", "roi_019"}:
        morphology = "round_center" if roi == "roi_014" else "unreviewed"
        note = "Reviewer sees activation as an individual neuron."
        if roi == "roi_019":
            note += " Intensity change is subtle."
        return {**common, "neuron_confidence": "confirmed",
                "activity_confidence": "visible", "morphology": morphology,
                "disposition": "confirmed_neuron", "include_confirmed": "true",
                "include_inclusive": "true", "source_note": note}
    if burst == 2 and roi in {"roi_008", "roi_017", "roi_020"}:
        note = (
            "Reviewer sees a flash but cannot resolve a cell boundary."
            if roi != "roi_020" else
            "Reviewer sees only a localized dot activate; biological identity is unresolved."
        )
        return {**common, "neuron_confidence": "uncertain",
                "activity_confidence": "visible",
                "morphology": "dot" if roi == "roi_020" else "no_boundary",
                "disposition": "activity_visible_identity_uncertain",
                "include_confirmed": "false", "include_inclusive": "true",
                "source_note": note}
    return {}


def draft_rows(labels: Iterable[Mapping[str, Any]], targets: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label in labels:
        roi = str(label["roi_identity"])
        base = {
            "schema_version": "1",
            "observation_id": observation_id(label),
            "burst_id": str(int(label["burst_id"])),
            "original_roi_id": roi,
            "canonical_roi_id": roi,
            "x_px": f"{float(label['x_px']):.6f}",
            "y_px": f"{float(label['y_px']):.6f}",
            "original_start_frame_ui": str(int(label["start_frame_ui"])),
            "original_end_frame_ui": str(int(label["end_frame_ui"])),
            "event_onset_ui": "",
            "event_peak_ui": "",
            "event_end_ui": "",
            "neuron_confidence": "legacy_label",
            "activity_confidence": "legacy_label",
            "morphology": "unreviewed",
            "context": "unreviewed",
            "disposition": "confirmed_neuron",
            "include_confirmed": "true",
            "include_inclusive": "true",
            "review_status": "legacy_source" if roi not in targets else "pending",
            "reviewer_id": "original_workbook" if roi not in targets else "",
            "reviewed_at": "",
            "source_note": "Original sparse-positive workbook label." if roi not in targets else "Targeted hard-ROI review pending.",
            "merge_reason": "",
        }
        base.update(_seed_override(label))
        rows.append(base)
    return rows


def write_tsv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ADJUDICATION_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def load_tsv(path: Path, *, require_adjudicated_targets: set[str] | None = None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != ADJUDICATION_FIELDS:
            raise ValueError("adjudication columns differ from the version-1 contract")
        raw = list(reader)
    if not raw:
        raise ValueError("adjudication table is empty")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for item in raw:
        if item["schema_version"] != "1":
            raise ValueError("unsupported adjudication row schema")
        oid = item["observation_id"]
        if not oid or oid in seen:
            raise ValueError(f"duplicate or empty observation_id: {oid!r}")
        seen.add(oid)
        if item["disposition"] not in DISPOSITIONS:
            raise ValueError(f"invalid disposition for {oid}")
        if item["review_status"] not in REVIEW_STATUSES:
            raise ValueError(f"invalid review_status for {oid}")
        if item["neuron_confidence"] not in NEURON_CONFIDENCE:
            raise ValueError(f"invalid neuron_confidence for {oid}")
        if item["activity_confidence"] not in ACTIVITY_CONFIDENCE:
            raise ValueError(f"invalid activity_confidence for {oid}")
        start = int(item["original_start_frame_ui"])
        end = int(item["original_end_frame_ui"])
        timing = []
        for key in ("event_onset_ui", "event_peak_ui", "event_end_ui"):
            timing.append(None if not item[key].strip() else int(item[key]))
        if any(value is not None for value in timing):
            if any(value is None for value in timing):
                raise ValueError(f"partial event timing for {oid}")
            assert all(value is not None for value in timing)
            if not (1 <= timing[0] <= timing[1] <= timing[2]):
                raise ValueError(f"invalid event timing order for {oid}")
        row: dict[str, Any] = dict(item)
        row.update(
            burst_id=int(item["burst_id"]), x_px=float(item["x_px"]),
            y_px=float(item["y_px"]), original_start_frame_ui=start,
            original_end_frame_ui=end, event_onset_ui=timing[0],
            event_peak_ui=timing[1], event_end_ui=timing[2],
            include_confirmed=_truth(item["include_confirmed"]),
            include_inclusive=_truth(item["include_inclusive"]),
        )
        rows.append(row)
    if require_adjudicated_targets:
        incomplete = sorted({
            row["original_roi_id"] for row in rows
            if row["original_roi_id"] in require_adjudicated_targets
            and row["review_status"] != "adjudicated"
        })
        if incomplete:
            raise ValueError(f"target ROIs still require adjudication: {incomplete}")
    return rows


def label_view(rows: Iterable[Mapping[str, Any]], view: str, timing: str) -> list[dict[str, Any]]:
    if view not in {"original", "confirmed", "inclusive"}:
        raise ValueError(f"unknown label view: {view}")
    if timing not in {"original", "adjudicated"}:
        raise ValueError(f"unknown timing view: {timing}")
    selected: list[dict[str, Any]] = []
    for row in rows:
        if view == "confirmed" and not bool(row["include_confirmed"]):
            continue
        if view == "inclusive" and not bool(row["include_inclusive"]):
            continue
        canonical = str(row["original_roi_id"] if view == "original" else row["canonical_roi_id"])
        onset = int(row["original_start_frame_ui"])
        end = int(row["original_end_frame_ui"])
        peak = None
        if timing == "adjudicated" and row.get("event_onset_ui") is not None:
            onset = int(row["event_onset_ui"])
            peak = int(row["event_peak_ui"])
            end = int(row["event_end_ui"])
        selected.append({
            "observation_id": row["observation_id"], "burst_id": int(row["burst_id"]),
            "roi_identity": canonical, "original_roi_id": row["original_roi_id"],
            "x_px": float(row["x_px"]), "y_px": float(row["y_px"]),
            "start_frame_ui": onset, "peak_frame_ui": peak, "end_frame_ui": end,
            "disposition": row["disposition"],
        })
    if view != "original":
        collapsed: dict[tuple[int, str], dict[str, Any]] = {}
        for row in selected:
            key = (row["burst_id"], row["roi_identity"])
            if key not in collapsed or row["original_roi_id"] == row["roi_identity"]:
                collapsed[key] = row
            else:
                current = collapsed[key]
                current["start_frame_ui"] = min(current["start_frame_ui"], row["start_frame_ui"])
                current["end_frame_ui"] = max(current["end_frame_ui"], row["end_frame_ui"])
        selected = list(collapsed.values())
    return sorted(selected, key=lambda row: (row["burst_id"], row["roi_identity"]))
