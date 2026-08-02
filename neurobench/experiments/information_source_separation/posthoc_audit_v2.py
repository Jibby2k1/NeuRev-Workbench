"""Non-destructive margin-aware audit for a completed generated screen."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .screen_runner import _atomic_json
from .selection_v3 import select_finalists_with_equivalence_margin


def audit_completed_screen(source_root: Path, output_root: Path) -> dict[str, Any]:
    source = source_root.resolve()
    output = output_root.resolve()
    if output.exists():
        raise FileExistsError(f"posthoc audit root exists: {output}")
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    resolved_config = json.loads((source / "config.resolved.json").read_text(encoding="utf-8"))
    scientific = resolved_config["scientific_config"]
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((source / "fits").glob("*.json"))]
    if len(rows) != int(metrics["fit_count"]):
        raise RuntimeError("completed fit count differs from source metrics")
    selection = select_finalists_with_equivalence_margin(
        rows, expected_fixture_count=int(metrics["fixture_count"]),
        equivalence_margin=float(scientific["selection"]["equivalence_margin"]),
        finalists_per_new_method=2, require_unresolved_accuracy=True,
    )
    payload = {
        "schema_version": 2,
        "kind": "information_source_separation_posthoc_selection_audit",
        "source_root": str(source), "source_root_unchanged": True,
        "fit_count": len(rows), "selection": selection,
        "interpretation": (
            "Posthoc correction: unresolved controls affect only abstention, and lower "
            "rank is preferred within the frozen equivalence margin. This does not "
            "authorize confirmation or alter the completed screen."
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "metrics.json", payload)
    return payload
