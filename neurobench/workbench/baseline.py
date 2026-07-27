"""Wave 0 preservation locks for local NeuRev workbench migrations.

The baseline is intentionally small and content-addressed.  It records only
stable app contracts (identity, review data, annotations, architecture-run
catalog, and served assets), never raw video pixels or generated experiment
trees.  A later migration can compare this report before and after a build
without rewriting any historical artifact.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from neurobench.data.catalog import dataset_record_for_app, discover_dataset_catalog


BASELINE_SCHEMA_VERSION = 2
LOCKED_APP_FILES = (
    "review_data.json",
    "annotations.json",
    "architecture_runs.json",
    "dataset_manifest.generated.json",
    "index.html",
    "workbench.css",
    "workbench.js",
)
REQUIRED_APP_FILES = (
    "review_data.json",
    "annotations.json",
    "architecture_runs.json",
    "index.html",
    "workbench.css",
    "workbench.js",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_under_workspace(path: str | Path, workspace: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    if resolved != workspace and workspace not in resolved.parents:
        raise ValueError(f"Baseline path is outside workspace root {workspace}: {resolved}")
    return resolved


def _app_record(app_dir: Path, workspace: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_APP_FILES if not (app_dir / name).is_file()]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(f"Protected workbench files missing under {app_dir}: {names}")
    files: dict[str, dict[str, Any]] = {}
    for name in LOCKED_APP_FILES:
        path = app_dir / name
        if not path.is_file():
            continue
        files[name] = {
            "path": _relative(path, workspace),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    record = dataset_record_for_app(app_dir, workspace_root=workspace)
    return {
        "dataset_id": record.get("dataset_id"),
        "app_dir": _relative(app_dir, workspace),
        "files": files,
        "identity": {
            "name": record.get("name"),
            "paths": dict(record.get("paths") or {}),
            "video": dict(record.get("video") or {}),
        },
        "capability_states": dict(record.get("capability_states") or {}),
    }


def baseline_stable_payload(baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Return only preservation identity fields, excluding capture metadata."""

    return {
        "schema_version": baseline.get("schema_version"),
        "kind": baseline.get("kind"),
        "locked_contracts": dict(baseline.get("locked_contracts") or {}),
        "catalog": list(baseline.get("catalog") or []),
        "apps": list(baseline.get("apps") or []),
    }


def _stable_digest(baseline: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        baseline_stable_payload(baseline),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _catalog_app_candidates(catalog: Iterable[Mapping[str, Any]], workspace: Path) -> list[Path]:
    candidates: list[Path] = []
    for record in catalog:
        paths = record.get("paths") if isinstance(record.get("paths"), Mapping) else {}
        app_value = paths.get("app_dir")
        if not app_value:
            continue
        app = _resolve_under_workspace(str(app_value), workspace)
        if app not in candidates:
            candidates.append(app)
    return candidates


def capture_baseline(
    workspace_root: str | Path,
    *,
    app_dirs: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Capture a deterministic Wave 0 baseline without mutating the workspace."""

    workspace = Path(workspace_root).expanduser().resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(f"Workspace root does not exist: {workspace}")
    catalog = discover_dataset_catalog(workspace)
    if app_dirs is None:
        candidates = _catalog_app_candidates(catalog, workspace)
        for candidate in sorted((workspace / "Outputs" / "NeuronReview").glob("*/app")):
            resolved = _resolve_under_workspace(candidate, workspace)
            if resolved not in candidates:
                candidates.append(resolved)
    else:
        candidates = [_resolve_under_workspace(value, workspace) for value in app_dirs]
    resolved_apps: list[Path] = []
    for candidate in candidates:
        app = _resolve_under_workspace(candidate, workspace)
        if not app.is_dir():
            raise FileNotFoundError(f"Workbench app directory does not exist: {app}")
        if app not in resolved_apps:
            resolved_apps.append(app)
    apps = [_app_record(app, workspace) for app in sorted(resolved_apps, key=lambda item: item.as_posix())]
    catalog_identity = [
        {
            "dataset_id": item.get("dataset_id"),
            "name": item.get("name"),
            "paths": dict(item.get("paths") or {}),
            "capability_states": dict(item.get("capability_states") or {}),
            "exists": dict(item.get("exists") or {}),
            "lifecycle": dict(item.get("lifecycle") or {}),
        }
        for item in catalog
    ]
    baseline = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "kind": "neurobench_wave0_baseline",
        "captured_at": _utc_now(),
        "workspace_root": str(workspace),
        "locked_contracts": {
            "annotations_preserved": True,
            "architecture_run_catalog_preserved": True,
            "dataset_identity_preserved": True,
            "historical_outputs_untouched": True,
            "unknown_metadata_remains_unknown": True,
        },
        "catalog": catalog_identity,
        "apps": apps,
    }
    baseline["stable_identity"] = {
        "algorithm": "sha256",
        "sha256": _stable_digest(baseline),
    }
    return baseline


def write_baseline(
    path: str | Path,
    baseline: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a baseline, refusing replacement by default."""

    target = Path(path).expanduser().resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"Baseline already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(baseline), indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists() and not overwrite:
            raise FileExistsError(f"Baseline already exists: {target}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_baseline(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate one preservation baseline."""

    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != "neurobench_wave0_baseline":
        raise ValueError(f"Not a NeuRev Wave 0 baseline: {source}")
    if not isinstance(payload.get("apps"), list) or not isinstance(payload.get("catalog"), list):
        raise ValueError(f"Baseline is missing apps or catalog records: {source}")
    return payload


def _diff_values(expected: Any, actual: Any, *, path: str, changes: list[dict[str, Any]]) -> None:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in sorted(set(expected) | set(actual), key=str):
            child = f"{path}.{key}"
            if key not in expected:
                changes.append({"path": child, "expected": "<missing>", "actual": actual[key]})
            elif key not in actual:
                changes.append({"path": child, "expected": expected[key], "actual": "<missing>"})
            else:
                _diff_values(expected[key], actual[key], path=child, changes=changes)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        length = max(len(expected), len(actual))
        for index in range(length):
            child = f"{path}[{index}]"
            if index >= len(expected):
                changes.append({"path": child, "expected": "<missing>", "actual": actual[index]})
            elif index >= len(actual):
                changes.append({"path": child, "expected": expected[index], "actual": "<missing>"})
            else:
                _diff_values(expected[index], actual[index], path=child, changes=changes)
        return
    if expected != actual:
        changes.append({"path": path, "expected": expected, "actual": actual})


def diff_baselines(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> dict[str, Any]:
    """Compare stable preservation identity while ignoring capture timestamps."""

    expected_payload = baseline_stable_payload(expected)
    actual_payload = baseline_stable_payload(actual)
    changes: list[dict[str, Any]] = []
    _diff_values(expected_payload, actual_payload, path="$", changes=changes)
    expected_digest = _stable_digest(expected)
    actual_digest = _stable_digest(actual)
    expected_identity = expected.get("stable_identity") if isinstance(expected.get("stable_identity"), Mapping) else {}
    actual_identity = actual.get("stable_identity") if isinstance(actual.get("stable_identity"), Mapping) else {}
    recorded_expected = str(expected_identity.get("sha256") or "")
    recorded_actual = str(actual_identity.get("sha256") or "")
    expected_identity_valid = recorded_expected == expected_digest
    actual_identity_valid = recorded_actual == actual_digest
    return {
        "match": not changes and expected_identity_valid and actual_identity_valid,
        "expected_identity": expected_digest,
        "actual_identity": actual_digest,
        "expected_identity_valid": expected_identity_valid,
        "actual_identity_valid": actual_identity_valid,
        "changes": changes,
    }


def verify_baseline(
    workspace_root: str | Path,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """Recapture protected apps and compare them with a stored baseline."""

    app_dirs = [
        str(app.get("app_dir"))
        for app in baseline.get("apps") or []
        if isinstance(app, Mapping) and app.get("app_dir")
    ]
    actual = capture_baseline(workspace_root, app_dirs=app_dirs)
    report = diff_baselines(baseline, actual)
    report["actual_baseline"] = actual
    return report


__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "LOCKED_APP_FILES",
    "REQUIRED_APP_FILES",
    "baseline_stable_payload",
    "capture_baseline",
    "diff_baselines",
    "load_baseline",
    "verify_baseline",
    "write_baseline",
]
