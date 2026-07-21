"""Small helpers for dataset-level dashboard manifests.

Dashboard-producing workflows should write a `dashboard_manifest.json` at the
dataset or output root. The manifest is intentionally lightweight: it points to
servable apps, reports, source data, and recommended inspection runs without
requiring callers to know workflow-specific directory names.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


DASHBOARD_MANIFEST_NAME = "dashboard_manifest.json"
REQUIRED_FIELDS = ("schema_version", "dashboard_id", "dashboard_type", "dataset_id", "entrypoint")


class DashboardManifestError(ValueError):
    """Raised when a dashboard manifest is missing required structure."""


def dashboard_manifest_path(root: str | Path) -> Path:
    """Return the standard dashboard manifest path for an output root."""

    return Path(root) / DASHBOARD_MANIFEST_NAME


def validate_dashboard_manifest(payload: Mapping[str, Any]) -> None:
    """Validate the minimal dashboard manifest contract.

    This intentionally avoids a heavyweight schema so dashboard-producing tools
    can use it without importing the full validation stack.
    """

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise DashboardManifestError(f"Dashboard manifest missing required field(s): {', '.join(missing)}")
    if int(payload.get("schema_version", 0)) < 1:
        raise DashboardManifestError("Dashboard manifest schema_version must be >= 1")
    for field in ("dashboard_id", "dashboard_type", "dataset_id", "entrypoint"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise DashboardManifestError(f"Dashboard manifest field '{field}' must be a non-empty string")


def write_dashboard_manifest(root: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write `dashboard_manifest.json` under `root`."""

    validate_dashboard_manifest(payload)
    path = dashboard_manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_dashboard_manifest(root_or_manifest: str | Path) -> dict[str, Any]:
    """Load and validate a dashboard manifest from a root or explicit file path."""

    path = Path(root_or_manifest)
    if path.is_dir():
        path = dashboard_manifest_path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_dashboard_manifest(payload)
    return payload


def discover_dashboard_manifests(root: str | Path, *, max_depth: int = 4) -> list[Path]:
    """Find dashboard manifests under `root`, bounded by directory depth."""

    base = Path(root)
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    results: list[Path] = []
    for path in base.rglob(DASHBOARD_MANIFEST_NAME):
        try:
            depth = len(path.relative_to(base).parts) - 1
        except ValueError:
            continue
        if depth <= max_depth:
            results.append(path)
    return sorted(results)


def summarize_dashboard_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact, display-oriented summary for an inspected manifest."""

    validate_dashboard_manifest(payload)
    gamma = payload.get("gamma_cfar") if isinstance(payload.get("gamma_cfar"), Mapping) else {}
    review_app = payload.get("review_app") if isinstance(payload.get("review_app"), Mapping) else {}
    return {
        "dashboard_id": payload.get("dashboard_id"),
        "dashboard_type": payload.get("dashboard_type"),
        "dataset_id": payload.get("dataset_id"),
        "entrypoint": payload.get("entrypoint"),
        "serve_command": payload.get("serve_command", ""),
        "recommended_runs": list(gamma.get("recommended_runs") or []),
        "app_dir": review_app.get("app_dir", ""),
    }
