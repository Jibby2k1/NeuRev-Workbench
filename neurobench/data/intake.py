"""Dataset intake helpers for local and public neuroimaging sources."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any, Mapping


PUBLIC_DATASET_TEMPLATES: dict[str, dict[str, Any]] = {
    "local": {
        "label": "Local file",
        "description": "A local TIFF, NPY, or future converted movie already present on disk.",
        "required_optional_dependencies": (),
    },
    "dandi-nwb": {
        "label": "DANDI / NWB",
        "description": "DANDI-hosted NWB calcium-imaging datasets. Intake records metadata only; it does not download data.",
        "required_optional_dependencies": ("pynwb", "dandi"),
    },
    "janelia-figshare": {
        "label": "Janelia / Figshare",
        "description": "Published zebrafish light-sheet datasets that may need a local conversion step before review.",
        "required_optional_dependencies": (),
    },
}

SUPPORTED_LOCAL_SUFFIXES = {".npy", ".tif", ".tiff"}
PLANNED_SUFFIXES = {".nwb", ".h5", ".hdf5"}


@dataclass(frozen=True)
class IntakeCheck:
    name: str
    status: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def build_dataset_intake_manifest(
    *,
    dataset_id: str,
    raw_video: str | Path,
    app_dir: str | Path | None = None,
    frame_rate_hz: float | None = None,
    pixel_size_microns: float | None = None,
    source_template: str = "local",
    name: str | None = None,
    modality: str = "light_sheet_calcium",
    indicator: str = "GCaMP",
) -> dict[str, Any]:
    """Create a manifest stub for a dataset before heavy processing runs."""
    if not dataset_id:
        raise ValueError("dataset_id is required")
    template = PUBLIC_DATASET_TEMPLATES.get(source_template)
    if template is None:
        raise ValueError(f"Unknown source template: {source_template}")
    raw_path = Path(raw_video)
    app = Path(app_dir) if app_dir is not None else Path("Outputs/NeuronReview") / dataset_id / "app"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "name": name or raw_path.name or dataset_id,
        "modality": modality,
        "indicator": indicator,
        "source": {
            "template": source_template,
            "label": template["label"],
            "description": template["description"],
        },
        "paths": {
            "raw_video": str(raw_video),
            "app_dir": str(app),
            "review_data": str(app / "review_data.json"),
            "annotations": str(app / "annotations.json"),
            "architecture_runs": str(app / "architecture_runs.json"),
        },
    }
    if frame_rate_hz is not None:
        manifest["frame_rate_hz"] = float(frame_rate_hz)
    if pixel_size_microns is not None:
        manifest["pixel_size_microns"] = float(pixel_size_microns)
    return manifest


def intake_checks(manifest: Mapping[str, Any], *, base_dir: str | Path | None = None) -> list[IntakeCheck]:
    """Return conservative readiness checks for a dataset manifest stub."""
    checks: list[IntakeCheck] = []
    paths = manifest.get("paths") or {}
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    raw_value = str(paths.get("raw_video") or "")
    raw_path = Path(raw_value).expanduser()
    raw_resolved = raw_path if raw_path.is_absolute() else base / raw_path
    suffix = raw_path.suffix.lower()
    if raw_value and raw_resolved.exists():
        checks.append(IntakeCheck("raw_video", "ok", f"found {raw_value}"))
    elif raw_value:
        checks.append(IntakeCheck("raw_video", "warn", f"not found locally yet: {raw_value}"))
    else:
        checks.append(IntakeCheck("raw_video", "error", "paths.raw_video is required"))
    if suffix in SUPPORTED_LOCAL_SUFFIXES:
        checks.append(IntakeCheck("format", "ok", f"{suffix} is supported by current local intake"))
    elif suffix in PLANNED_SUFFIXES:
        checks.append(IntakeCheck("format", "warn", f"{suffix} needs a conversion/import bridge before review generation"))
    else:
        checks.append(IntakeCheck("format", "warn", "unknown format; expected .tif/.tiff/.npy or a planned public-data container"))
    frame_rate = manifest.get("frame_rate_hz")
    if isinstance(frame_rate, (int, float)) and frame_rate > 0:
        checks.append(IntakeCheck("frame_rate_hz", "ok", f"{frame_rate:g} Hz"))
    else:
        checks.append(IntakeCheck("frame_rate_hz", "error", "set frame_rate_hz before planning event/support windows"))
    pixel_size = manifest.get("pixel_size_microns")
    if isinstance(pixel_size, (int, float)) and pixel_size > 0:
        checks.append(IntakeCheck("pixel_size_microns", "ok", f"{pixel_size:g} um/px"))
    else:
        checks.append(IntakeCheck("pixel_size_microns", "warn", "missing physical pixel size disables soma-size QC"))
    source_template = str((manifest.get("source") or {}).get("template") or "local")
    deps = PUBLIC_DATASET_TEMPLATES.get(source_template, {}).get("required_optional_dependencies", ())
    for dep in deps:
        status = "ok" if importlib.util.find_spec(dep) else "warn"
        detail = f"optional dependency {dep} {'available' if status == 'ok' else 'not installed'}"
        checks.append(IntakeCheck(f"dependency:{dep}", status, detail))
    return checks


def dataset_intake_report(manifest: Mapping[str, Any], *, base_dir: str | Path | None = None) -> dict[str, Any]:
    checks = [check.as_dict() for check in intake_checks(manifest, base_dir=base_dir)]
    return {
        "schema_version": 1,
        "kind": "neurobench_dataset_intake_report",
        "dataset_id": manifest.get("dataset_id", ""),
        "source": manifest.get("source") or {"template": "local"},
        "checks": checks,
        "ready": all(check["status"] != "error" for check in checks),
    }
