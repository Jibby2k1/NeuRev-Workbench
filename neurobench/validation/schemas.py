"""JSON Schema loading and validation helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import jsonschema


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = PROJECT_ROOT / "schemas"
SCHEMA_ALIASES = {
    "dataset": "dataset_manifest.schema.json",
    "dataset_manifest": "dataset_manifest.schema.json",
    "dataset_manifest.schema": "dataset_manifest.schema.json",
    "architecture_run": "architecture_run.schema.json",
    "architecture_runs": "architecture_run.schema.json",
    "run": "architecture_run.schema.json",
    "pipeline_run": "pipeline_run.schema.json",
    "pipeline_spec": "pipeline_spec.schema.json",
    "dataset_import": "dataset_import.schema.json",
    "import": "dataset_import.schema.json",

    "video_manifest": "video_manifest.schema.json",
    "video": "video_manifest.schema.json",
    "template_spec": "template_spec.schema.json",
    "template": "template_spec.schema.json",
    "registration_result": "registration_result.schema.json",
    "registration": "registration_result.schema.json",
    "grid_spec": "grid_spec.schema.json",
    "grid": "grid_spec.schema.json",
    "dynamics_dataset": "dynamics_dataset.schema.json",
    "autoencoder_run": "autoencoder_run.schema.json",
    "latent_rnn_run": "latent_rnn_run.schema.json",
    "latent_classifier_run": "latent_classifier_run.schema.json",
    "llm_architecture_proposal": "llm_architecture_proposal.schema.json",
    "llm_proposal": "llm_architecture_proposal.schema.json",
    "fish_control_program": "fish_control_program.schema.json",
    "fish_program": "fish_control_program.schema.json",
    "artifact_record": "artifact_record.schema.json",
    "artifact": "artifact_record.schema.json",
    "annotations": "annotations.schema.json",
    "annotation": "annotations.schema.json",
    "review_data": "review_data.schema.json",
    "metrics_report": "metrics_report.schema.json",
    "metrics": "metrics_report.schema.json",
    "export_bundle": "export_bundle.schema.json",
    "export": "export_bundle.schema.json",
}


def schema_path(schema_name: str) -> Path:
    """Return the repository path for a known schema name or filename."""
    raw = str(schema_name).strip()
    if not raw:
        raise FileNotFoundError("Schema name is empty.")
    key = raw.removesuffix(".json")
    filename = SCHEMA_ALIASES.get(key, raw if raw.endswith(".json") else f"{raw}.schema.json")
    path = SCHEMA_DIR / filename
    if not path.exists():
        known = ", ".join(sorted(SCHEMA_ALIASES))
        raise FileNotFoundError(f"Unknown schema '{schema_name}'. Known schema names: {known}")
    return path


def load_schema(schema_name: str) -> dict[str, Any]:
    """Load and check a repository JSON Schema."""
    path = schema_path(schema_name)
    with path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    jsonschema.Draft202012Validator.check_schema(schema)
    return schema


def validate_dict(payload: Mapping[str, Any], schema_name: str) -> None:
    """Validate a JSON-like mapping against a repository schema."""
    schema = load_schema(schema_name)
    jsonschema.Draft202012Validator(schema).validate(dict(payload))
    if schema_path(schema_name).name == "annotations.schema.json":
        _validate_annotation_cfar_invariants(payload)
    if schema_path(schema_name).name == "dataset_import.schema.json":
        _validate_dataset_import_invariants(payload)


def _validate_dataset_import_invariants(payload: Mapping[str, Any]) -> None:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    if metadata.get("kind") not in {"video", "label_table", "neurev_json"}:
        raise jsonschema.ValidationError("metadata.kind must be video, label_table, or neurev_json")
    checksum = payload.get("checksum") if isinstance(payload.get("checksum"), Mapping) else {}
    sha = str(checksum.get("sha256") or "")
    if len(sha) != 64 or any(character not in "0123456789abcdef" for character in sha.lower()):
        raise jsonschema.ValidationError("checksum.sha256 must be a lowercase SHA-256 digest")
    if int(payload.get("revision") or 1) < 1:
        raise jsonschema.ValidationError("revision must be positive")
    role = str(
        payload.get("source_role")
        or (
            "primary_video_candidate"
            if metadata.get("kind") == "video"
            else "neurev_json_attachment"
            if metadata.get("kind") == "neurev_json"
            else "label_attachment"
        )
    )
    if role not in {"primary_video_candidate", "primary_video", "label_attachment", "neurev_json_attachment"}:
        raise jsonschema.ValidationError("source_role is invalid")


def _validate_annotation_cfar_invariants(payload: Mapping[str, Any]) -> None:
    containers: list[tuple[str, Mapping[str, Any]]] = []
    for key in ("rois", "virtualRois"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            containers.append((key, value))
    runs = payload.get("runs")
    if isinstance(runs, Mapping):
        for run_id, bucket in runs.items():
            if isinstance(bucket, Mapping) and isinstance(bucket.get("rois"), Mapping):
                containers.append((f"runs.{run_id}.rois", bucket["rois"]))
    for prefix, annotations in containers:
        for roi_id, annotation in annotations.items():
            if not isinstance(annotation, Mapping):
                continue
            region = annotation.get("cfar_regions")
            if not isinstance(region, Mapping):
                continue
            foreground = {tuple(point) for point in region.get("foreground_points", [])}
            background = {tuple(point) for point in region.get("background_points", [])}
            overlap = foreground & background
            if overlap:
                point = sorted(overlap)[0]
                raise jsonschema.ValidationError(
                    f"{prefix}.{roi_id}.cfar_regions foreground/background points must be mutually exclusive; overlap at {list(point)}"
                )


def validate_json(path: str | Path, schema_name: str) -> dict[str, Any]:
    """Load and validate a JSON file, returning the parsed payload."""
    json_path = Path(path).expanduser()
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_dict(payload, schema_name)
    return payload


def validation_error_summary(exc: Exception) -> str:
    """Return a concise, field-oriented validation error message."""
    if isinstance(exc, jsonschema.ValidationError):
        field = ".".join(str(part) for part in exc.absolute_path) or "(root)"
        schema_field = ".".join(str(part) for part in exc.absolute_schema_path) or "(schema root)"
        return f"field: {field}; problem: {exc.message}; schema: {schema_field}"
    return str(exc)
