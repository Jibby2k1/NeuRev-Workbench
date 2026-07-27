"""Shared truthful capability-state vocabulary for the normal workbench."""
from __future__ import annotations

from typing import Any, Mapping


CAPABILITY_STATES = ("ready", "import_only", "planned", "blocked", "unavailable")


def capability_state(*, ready: bool = False, import_only: bool = False, planned: bool = False, blocked: bool = False) -> str:
    if blocked:
        return "blocked"
    if ready:
        return "ready"
    if import_only:
        return "import_only"
    if planned:
        return "planned"
    return "unavailable"


def capability_states(record: Mapping[str, Any]) -> dict[str, str]:
    """Map a catalog record to UI states without hiding missing evidence."""

    capabilities = record.get("capabilities") if isinstance(record.get("capabilities"), Mapping) else {}
    exists = record.get("exists") if isinstance(record.get("exists"), Mapping) else {}
    readiness = record.get("readiness") if isinstance(record.get("readiness"), Mapping) else {}
    review_ready = bool(readiness.get("review_ready"))
    results_ready = bool(readiness.get("scientific_results_ready") or capabilities.get("scientific_results"))
    video_present = bool(exists.get("raw_video") or exists.get("raw_videos"))
    return {
        "annotate": capability_state(ready=review_ready, import_only=video_present),
        "results": capability_state(ready=results_ready, planned=review_ready or video_present),
        "raw_video": capability_state(ready=video_present),
        "manual_roi": capability_state(ready=bool(capabilities.get("manual_roi_annotation"))),
        "cfar_annotation": capability_state(ready=bool(capabilities.get("cfar_annotation")), planned=review_ready),
        "research_tools": capability_state(ready=True),
    }


__all__ = ["CAPABILITY_STATES", "capability_state", "capability_states"]
