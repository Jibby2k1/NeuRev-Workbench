"""Common, atomic stage artifact contract."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import atomic_json, atomic_text


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_stage(root: Path, stage: str, metrics: dict[str, Any], *, status: str = "complete", decision: str = "advance", warnings: list[str] | None = None, next_stage: str | None = None, review: str = "No stage-specific human review is required.\n", decisions: list[dict[str, Any]] | None = None) -> None:
    target = root / stage
    target.mkdir(parents=True, exist_ok=True)
    warnings = warnings or []
    atomic_json(target / "METRICS.json", metrics)
    atomic_json(target / "status.json", {"schema_version": 1, "stage": stage, "status": status, "scientific_decision": decision, "started_at": metrics.get("started_at"), "completed_at": now(), "input_hashes": metrics.get("input_hashes", {}), "output_hashes": {}, "warnings": warnings, "blocked_dependencies": [], "next_stage": next_stage})
    atomic_json(target / "artifact_index.json", {"schema_version": 1, "stage": stage, "artifacts": sorted(p.name for p in target.iterdir() if p.is_file())})
    atomic_json(target / "llm_context.json", {"schema_version": 1, "stage": stage, "status": status, "scientific_decision": decision, "primary_metrics": "METRICS.json", "warnings": warnings})
    atomic_text(target / "STAGE_SUMMARY.md", f"# {stage}\n\nStatus: `{status}`. Scientific decision: `{decision}`.\n")
    atomic_text(target / "VALIDATION_REPORT.md", f"# Validation\n\nStage contract written atomically. Status: `{status}`.\n")
    atomic_text(target / "FIGURE_INDEX.md", "# Figure index\n\nSee `artifact_index.json` for the machine-readable inventory.\n")
    atomic_text(target / "REVIEW_REQUIRED.md", "# Review required\n\n" + review)
    atomic_text(target / "DECISIONS_REQUESTED.yaml", "schema_version: 1\nitems:\n" + ("\n".join(f"  - id: {item['id']}\n    status: pending" for item in decisions) if decisions else "  []") + "\n")
