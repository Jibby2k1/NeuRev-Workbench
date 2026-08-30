"""Adjudication adapters that never conflate spatial sites with neurons."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from .contracts import ObservationRecord, disk_geometry_hash, validate_unique


def _truth(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def load_adjudication(path: Path, radius_px: int = 2) -> list[ObservationRecord]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    records = []
    for row in rows:
        original = str(row["original_roi_id"])
        x_px, y_px = float(row["x_px"]), float(row["y_px"])
        canonical = str(row.get("canonical_roi_id") or "").strip() or None
        records.append(ObservationRecord(
            observation_id=str(row["observation_id"]), burst_id=int(row["burst_id"]),
            original_roi_id=original, observation_site_id=original,
            canonical_neuron_id=canonical, x_px=x_px, y_px=y_px,
            geometry_kind=f"disk_r{radius_px}", geometry_hash=disk_geometry_hash(x_px, y_px, radius_px),
            original_start_frame_ui=int(row["original_start_frame_ui"]),
            original_end_frame_ui=int(row["original_end_frame_ui"]),
            event_onset_ui=_optional_int(row.get("event_onset_ui")),
            event_peak_ui=_optional_int(row.get("event_peak_ui")),
            event_end_ui=_optional_int(row.get("event_end_ui")),
            include_confirmed=_truth(row.get("include_confirmed", True)),
            include_inclusive=_truth(row.get("include_inclusive", True)),
            review_status=str(row.get("review_status", "legacy_source")),
            disposition=str(row.get("disposition", "unresolved")),
        ))
    return validate_unique(records)


def _optional_int(value: Any) -> int | None:
    text = "" if value is None else str(value).strip()
    return None if not text else int(text)


def analysis_view(records: Iterable[ObservationRecord], view: str) -> list[ObservationRecord]:
    allowed = {
        "original_site_original_timing", "original_site_adjudicated_timing",
        "canonical_proposed_original_timing", "canonical_confirmed_adjudicated_timing",
        "canonical_inclusive_adjudicated_timing",
    }
    if view not in allowed:
        raise ValueError(f"unknown analysis view: {view}")
    values = list(records)
    if view.startswith("canonical_confirmed"):
        values = [record for record in values if record.include_confirmed]
    elif view.startswith("canonical_inclusive"):
        values = [record for record in values if record.include_inclusive]
    return values


def identity_counts(records: Iterable[ObservationRecord]) -> dict[str, int]:
    values = list(records)
    return {
        "occurrences": len(values),
        "original_sites": len({r.observation_site_id for r in values}),
        "proposed_canonical_identities": len({r.canonical_neuron_id for r in values if r.canonical_neuron_id}),
    }
