"""Collision-safe persistence for complete annotation revision roots."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

from neurobench.models.annotation_revision import AnnotationOperation, AnnotationRevision
from neurobench.models.annotations import AnnotationSet


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"


def initialize_revision_root(
    revisions_root: str | Path,
    *,
    revision: AnnotationRevision | Mapping[str, Any],
    annotations: AnnotationSet | Mapping[str, Any],
    operations: Iterable[AnnotationOperation | Mapping[str, Any]] = (),
) -> Path:
    """Create one complete revision directory without overwriting any target."""

    revision_model = revision if isinstance(revision, AnnotationRevision) else AnnotationRevision.from_dict(revision)
    annotation_model = annotations if isinstance(annotations, AnnotationSet) else AnnotationSet.from_dict(annotations)
    revision_model.validate()
    annotation_model.validate()

    operation_models = [
        item if isinstance(item, AnnotationOperation) else AnnotationOperation.from_dict(item)
        for item in operations
    ]
    operation_ids = [item.payload["operationId"] for item in operation_models]
    if len(operation_ids) != len(set(operation_ids)):
        raise ValueError("operationId values must be unique")
    if revision_model.payload["operationCount"] != len(operation_models):
        raise ValueError("operationCount must equal the number of stored operations")
    for expected_token, operation in enumerate(operation_models):
        if operation.payload["expectedRevisionToken"] != expected_token:
            raise ValueError("operation expectedRevisionToken values must be contiguous from zero")
        if operation.payload["reviewerId"] != revision_model.payload["reviewerId"]:
            raise ValueError("operation reviewerId must match revision reviewerId")

    root = Path(revisions_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    target = root / revision_model.payload["revisionId"]
    if target.exists():
        raise FileExistsError(f"Refusing annotation revision collision: {target}")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=root))
    try:
        (staging / "revision.json").write_text(_json_text(revision_model.to_dict()), encoding="utf-8")
        (staging / "annotations.json").write_text(_json_text(annotation_model.to_dict()), encoding="utf-8")
        operation_text = "".join(json.dumps(item.to_dict(), sort_keys=True) + "\n" for item in operation_models)
        (staging / "operations.jsonl").write_text(operation_text, encoding="utf-8")
        (staging / "exports").mkdir()
        try:
            os.rename(staging, target)
        except FileExistsError as exc:
            raise FileExistsError(f"Refusing annotation revision collision: {target}") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def load_revision_root(path: str | Path) -> tuple[AnnotationRevision, AnnotationSet, list[AnnotationOperation]]:
    """Load and cross-check one complete revision root."""

    root = Path(path).expanduser()
    revision = AnnotationRevision.from_dict(json.loads((root / "revision.json").read_text(encoding="utf-8")))
    annotations = AnnotationSet.from_dict(json.loads((root / "annotations.json").read_text(encoding="utf-8")))
    annotations.validate()
    operations = [
        AnnotationOperation.from_dict(json.loads(line))
        for line in (root / "operations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if root.name != revision.payload["revisionId"]:
        raise ValueError("revision directory name must equal revisionId")
    if len(operations) != revision.payload["operationCount"]:
        raise ValueError("operationCount does not match operations.jsonl")
    return revision, annotations, operations

class RevisionConflictError(ValueError):
    """Raised when an optimistic draft write targets a stale revision token."""


def revision_snapshot(path: str | Path) -> dict[str, Any]:
    """Return one validated revision root as a JSON-ready API payload."""

    revision, annotations, operations = load_revision_root(path)
    return {
        "revision": revision.to_dict(),
        "annotations": annotations.to_dict(),
        "operations": [item.to_dict() for item in operations],
    }


def list_revision_roots(revisions_root: str | Path) -> list[dict[str, Any]]:
    """List valid revision metadata without creating or modifying the root."""

    root = Path(revisions_root).expanduser()
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in root.iterdir() if item.is_dir() and not item.name.startswith(".")):
        try:
            revision, _, _ = load_revision_root(path)
        except (OSError, ValueError):
            continue
        records.append(revision.to_dict())
    return sorted(records, key=lambda item: (str(item.get("updatedAt") or ""), str(item["revisionId"])), reverse=True)


def fork_revision_root(
    source_root: str | Path,
    revisions_root: str | Path,
    *,
    revision_id: str,
    reviewer_id: str,
    timestamp: str,
) -> Path:
    """Fork the current projection into a new, independently editable draft.

    The server constructs ancestry and provenance from the validated source;
    clients cannot replace annotations or scientific identity fields while
    forking. The new draft starts a fresh append-only operation history.
    """

    source_revision, annotations, _ = load_revision_root(source_root)
    identifier = str(revision_id).strip()
    reviewer = str(reviewer_id).strip()
    if not reviewer:
        raise ValueError("reviewerId is required")
    resolve_revision_root(revisions_root, identifier)
    source = source_revision.to_dict()
    revision = {
        **source,
        "revisionId": identifier,
        "parentRevisionId": source["revisionId"],
        "state": "draft",
        "reviewerId": reviewer,
        "createdAt": timestamp,
        "updatedAt": timestamp,
        "revisionToken": 0,
        "operationCount": 0,
    }
    return initialize_revision_root(
        revisions_root,
        revision=revision,
        annotations=annotations,
        operations=(),
    )


def publish_revision_root(
    draft_root: str | Path,
    revisions_root: str | Path,
    *,
    revision_id: str,
    expected_revision_token: int,
    timestamp: str,
) -> Path:
    """Publish a validated draft as a separate immutable child revision."""

    draft_revision, annotations, operations = load_revision_root(draft_root)
    draft = draft_revision.to_dict()
    if draft["state"] != "draft":
        raise ValueError("only draft annotation revisions can be published")
    if expected_revision_token != draft["revisionToken"]:
        raise RevisionConflictError(
            "revision token conflict: expected "
            f"{draft['revisionToken']}, received {expected_revision_token}"
        )
    identifier = str(revision_id).strip()
    resolve_revision_root(revisions_root, identifier)
    revision = {
        **draft,
        "revisionId": identifier,
        "parentRevisionId": draft["revisionId"],
        "state": "published",
        "createdAt": timestamp,
        "updatedAt": timestamp,
    }
    return initialize_revision_root(
        revisions_root,
        revision=revision,
        annotations=annotations,
        operations=operations,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.partial-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _operation_projection(current: Mapping[str, Any] | None, operation: AnnotationOperation) -> dict[str, Any]:
    payload = operation.payload
    operation_type = payload["operationType"]
    before = payload["before"]
    after = payload["after"]
    if operation_type in {"create", "promote"}:
        if current is not None:
            raise RevisionConflictError(f"target already exists: {payload['targetId']}")
        return deepcopy(after)
    if current is None:
        raise RevisionConflictError(f"target does not exist: {payload['targetId']}")
    if dict(current) != before:
        raise RevisionConflictError(f"target projection changed: {payload['targetId']}")
    if operation_type == "tombstone":
        result = deepcopy(dict(current))
        result["deleted"] = True
        result["tombstonedByOperationId"] = payload["operationId"]
        return result
    return deepcopy(after)


def append_revision_operation(
    revision_root: str | Path,
    operation: AnnotationOperation | Mapping[str, Any],
) -> dict[str, Any]:
    """Apply one append-only draft operation with optimistic token checking.

    Projection, operation log, and revision metadata use same-directory atomic
    replacements. The revision file is replaced last and acts as the commit
    marker; ordinary write failures restore the prior validated bytes.
    """

    root = Path(revision_root).expanduser()
    operation_model = operation if isinstance(operation, AnnotationOperation) else AnnotationOperation.from_dict(operation)
    revision, annotations, operations = load_revision_root(root)
    revision_payload = revision.to_dict()
    operation_payload = operation_model.to_dict()
    if revision_payload["state"] != "draft":
        raise ValueError("published annotation revisions are immutable")
    expected = revision_payload["revisionToken"]
    if operation_payload["expectedRevisionToken"] != expected:
        raise RevisionConflictError(
            f"revision token conflict: expected {expected}, received {operation_payload['expectedRevisionToken']}"
        )
    if operation_payload["reviewerId"] != revision_payload["reviewerId"]:
        raise ValueError("operation reviewerId must match revision reviewerId")
    if any(item.payload["operationId"] == operation_payload["operationId"] for item in operations):
        raise RevisionConflictError(f"operationId already exists: {operation_payload['operationId']}")

    annotation_payload = annotations.to_dict()
    rois = annotation_payload.setdefault("rois", {})
    target_id = operation_payload["targetId"]
    current = deepcopy(rois.get(target_id))
    rois[target_id] = _operation_projection(current, operation_model)
    annotation_payload["updatedAt"] = operation_payload["timestamp"]
    annotation_model = AnnotationSet.from_dict(annotation_payload)
    annotation_model.validate()

    updated_operations = [*operations, operation_model]
    revision_payload["updatedAt"] = operation_payload["timestamp"]
    revision_payload["revisionToken"] = expected + 1
    revision_payload["operationCount"] = expected + 1
    updated_revision = AnnotationRevision.from_dict(revision_payload)

    paths = {
        "annotations": root / "annotations.json",
        "operations": root / "operations.jsonl",
        "revision": root / "revision.json",
    }
    before_bytes = {key: path.read_bytes() for key, path in paths.items()}
    after_bytes = {
        "annotations": _json_text(annotation_model.to_dict()).encode("utf-8"),
        "operations": "".join(
            json.dumps(item.to_dict(), sort_keys=True) + "\n" for item in updated_operations
        ).encode("utf-8"),
        "revision": _json_text(updated_revision.to_dict()).encode("utf-8"),
    }
    try:
        _atomic_write(paths["annotations"], after_bytes["annotations"])
        _atomic_write(paths["operations"], after_bytes["operations"])
        _atomic_write(paths["revision"], after_bytes["revision"])
    except Exception:
        for key in ("annotations", "operations", "revision"):
            _atomic_write(paths[key], before_bytes[key])
        raise
    return revision_snapshot(root)



def resolve_revision_root(revisions_root: str | Path, revision_id: str) -> Path:
    """Resolve one schema-safe revision directory without permitting traversal."""

    identifier = str(revision_id)
    if not identifier or not identifier[0].isalnum() or any(
        not (character.isalnum() or character in "._-") for character in identifier
    ):
        raise ValueError("invalid annotation revision ID")
    root = Path(revisions_root).expanduser().resolve()
    target = (root / identifier).resolve()
    if target.parent != root:
        raise ValueError("invalid annotation revision ID")
    return target
