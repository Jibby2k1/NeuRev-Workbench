"""Shared server-side helpers for the Neurobench workbench."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import mimetypes
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import parse_qs, unquote, urlparse

from neurobench.architecture_runs import as_run_manifest
from neurobench.data.catalog import (
    dataset_id_from_review,
    dataset_record_for_app,
    discover_dataset_catalog,
    raw_video_from_review,
)
from neurobench.data.intake import build_dataset_intake_manifest
from neurobench.data.qc import compute_video_qc
from neurobench.data.video import iter_video_chunks
from neurobench.data.imports import (
    MAX_IMPORT_BYTES,
    MAX_LABEL_ARTIFACT_BYTES,
    MAX_NEUREV_JSON_BYTES,
    SUPPORTED_LABEL_SUFFIXES,
    SUPPORTED_NEUREV_SUFFIXES,
    SUPPORTED_VIDEO_SUFFIXES,
    atomic_write_import_record,
    dataset_manifest_from_import,
    import_id,
    inspect_source,
    inspect_neurev_json,
    iter_label_rows,
    infer_label_mapping,
    make_import_record,
    normalize_dataset_id,
    read_import_record,
    relative_workspace_path,
    resolve_allowed_local_path,
    source_kind,
    transition_import_record,
    update_import_record,
    verify_source_identity,
)
from neurobench.llm_planning import proposal_set_to_architecture_manifest, validate_proposal_set
from neurobench.pipeline_catalog import normalize_pipeline
from neurobench.workbench.annotation_revisions import (
    RevisionConflictError,
    append_revision_operation,
    fork_revision_root,
    initialize_revision_root,
    list_revision_roots,
    publish_revision_root,
    resolve_revision_root,
    revision_snapshot,
)
from neurobench.workbench.materialize import materialize_virtual_roi_traces
from neurobench.workbench.label_reconciliation import reconcile_label_table
from neurobench.validation.schemas import validate_dict
from neurobench.workbench.jobs import JobStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APP_DIR = PROJECT_ROOT / "Outputs/NeuronReview/calcium_video_2/app"
DEFAULT_FIJI = Path("/home/jibby2k1/.local/bin/fiji")
MAX_LOG_LINES = 300
ALLOWED_BACKENDS = {"auto", "fiji_groovy", "python_gpu"}
GENERATION_STAGES = "all"
IMPORT_RECORD_DIRNAME = "imports"
MAX_POST_JSON_BYTES = 2_000_000
MAX_QC_SAMPLE_FRAMES = 64
MAX_RENDER_FRAMES = 50_000
MAX_RENDER_ESTIMATED_BYTES = 2_000_000_000
MAX_RENDER_WORKING_BYTES = 1_500_000_000
MAX_FRAME_PIXELS = 100_000_000
MAX_LABEL_OVERLAY_POINTS = 100_000
_DATASET_LOCKS_GUARD = threading.Lock()
_DATASET_LOCKS: dict[str, threading.RLock] = {}
_JOB_STORES_GUARD = threading.Lock()
_JOB_STORES: dict[str, JobStore] = {}


def dataset_lock(dataset_id: str) -> threading.RLock:
    key = normalize_dataset_id(dataset_id)
    with _DATASET_LOCKS_GUARD:
        return _DATASET_LOCKS.setdefault(key, threading.RLock())


def job_store_for_app(app_dir: Path) -> JobStore:
    root = (app_dir.resolve() / ".neurobench" / "jobs").resolve()
    key = str(root)
    with _JOB_STORES_GUARD:
        store = _JOB_STORES.get(key)
        if store is None:
            store = JobStore(root)
            store.recover_incomplete()
            _JOB_STORES[key] = store
        return store


def durable_job_records_for_app(app_dir: Path) -> list[dict[str, Any]]:
    """Read durable jobs without creating directories or recovering state."""

    root = app_dir.resolve() / ".neurobench" / "jobs"
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = load_json(path)
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return sorted(records, key=lambda item: (str(item.get("created_at") or ""), str(item.get("job_id") or "")))


def durable_job_record_for_app(app_dir: Path, job_id: str) -> dict[str, Any] | None:
    identifier = str(job_id)
    if not identifier or safe_run_id(identifier) != identifier:
        return None
    path = app_dir.resolve() / ".neurobench" / "jobs" / f"{identifier}.json"
    if not path.is_file():
        return None
    try:
        payload = load_json(path)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def import_dataset_root(dataset_id: str) -> Path:
    return PROJECT_ROOT / "Outputs" / "NeuronReview" / normalize_dataset_id(dataset_id)


def import_app_dir(dataset_id: str) -> Path:
    return import_dataset_root(dataset_id) / "app"


def dataset_id_for_app(app_dir: Path) -> str:
    resolved = app_dir.resolve()
    if resolved.name != "app":
        raise ValueError("Dataset app directory must be named app.")
    review_path = resolved / "review_data.json"
    if review_path.is_file():
        try:
            review = load_json(review_path)
            declared = dataset_id_from_review(review, fallback=resolved.parent.name)
            if declared:
                return str(declared)
        except (OSError, ValueError):
            pass
    for name in ("dataset_manifest.generated.json", "dataset_manifest.json"):
        manifest_path = resolved / name
        if manifest_path.is_file():
            try:
                declared = load_json(manifest_path).get("dataset_id")
                if declared:
                    return str(declared)
            except (OSError, ValueError):
                pass
    return resolved.parent.name


def app_dataset_id(app_dir: Path) -> str:
    return dataset_id_for_app(app_dir)


def import_app_for_record(record: Mapping[str, Any]) -> Path:
    raw = str(record.get("app_dir") or "")
    if not raw:
        return import_app_dir(str(record.get("dataset_id") or "dataset")).resolve()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    outputs = (PROJECT_ROOT / "Outputs").resolve()
    if candidate.name != "app" or outputs not in candidate.parents:
        raise ValueError("Import app_dir must resolve to an app directory under Outputs.")
    return candidate


def build_dataset_app_registry(
    *,
    configured_app: Path | None = None,
    root_dir: Path | None = None,
    conflicts_out: dict[str, tuple[str, ...]] | None = None,
) -> Mapping[str, Path]:
    """Build a bounded immutable dataset-ID to app mapping for one server."""

    workspace = PROJECT_ROOT.resolve()
    outputs = (workspace / "Outputs").resolve()
    candidates: dict[str, set[Path]] = {}

    def add_candidate(dataset_id: Any, path: Path, *, catalog_owned: bool = True) -> None:
        identifier = str(dataset_id or "")
        candidate = path.expanduser().resolve()
        if not identifier or candidate.name != "app" or not candidate.is_dir():
            return
        if catalog_owned and outputs not in candidate.parents:
            return
        candidates.setdefault(identifier, set()).add(candidate)

    if outputs.is_dir():
        for candidate in sorted(outputs.glob("*/*/app")):
            try:
                add_candidate(dataset_id_for_app(candidate), candidate)
            except ValueError:
                continue
    if root_dir is not None and root_dir.is_dir():
        for candidate in sorted(root_dir.glob("*/app")):
            try:
                add_candidate(dataset_id_for_app(candidate), candidate, catalog_owned=False)
            except ValueError:
                continue
    for record in discover_dataset_catalog(workspace):
        paths = record.get("paths") if isinstance(record.get("paths"), Mapping) else {}
        raw_app = paths.get("app_dir")
        if not raw_app:
            continue
        candidate = Path(str(raw_app)).expanduser()
        if not candidate.is_absolute():
            candidate = workspace / candidate
        add_candidate(record.get("dataset_id"), candidate)

    conflicts = {
        dataset_id: tuple(sorted(str(path) for path in paths))
        for dataset_id, paths in candidates.items()
        if len(paths) > 1
    }
    if conflicts_out is not None:
        conflicts_out.update(conflicts)
    registry = {dataset_id: next(iter(paths)) for dataset_id, paths in candidates.items() if len(paths) == 1}
    if configured_app is not None:
        configured = configured_app.resolve()
        try:
            registry[dataset_id_for_app(configured)] = configured
        except ValueError:
            pass
    return MappingProxyType(dict(sorted(registry.items())))


def import_record_path(dataset_id: str, import_id_value: str, *, app_dir: Path | None = None) -> Path:
    app = app_dir.resolve() if app_dir is not None else import_app_dir(dataset_id).resolve()
    return app / IMPORT_RECORD_DIRNAME / f"{safe_run_id(import_id_value)}.json"


def import_records(*, dataset_id: str | None = None, app_dir: Path | None = None) -> list[dict[str, Any]]:
    if app_dir is not None:
        app_contexts = [(app_dir.resolve(), str(dataset_id or app_dataset_id(app_dir)))]
    elif dataset_id:
        app_contexts = [(import_app_dir(dataset_id).resolve(), normalize_dataset_id(dataset_id))]
    else:
        root = PROJECT_ROOT / "Outputs" / "NeuronReview"
        app_contexts = []
        for candidate in sorted(root.glob("*/app")):
            if not candidate.is_dir():
                continue
            resolved = candidate.resolve()
            try:
                expected_dataset_id = dataset_id_for_app(resolved)
            except ValueError:
                continue
            app_contexts.append((resolved, expected_dataset_id))
    records: list[dict[str, Any]] = []
    for record_app, expected_dataset_id in app_contexts:
        directory = record_app / IMPORT_RECORD_DIRNAME
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                payload = read_import_record(
                    path,
                    expected_dataset_id=expected_dataset_id,
                    expected_app_dir=record_app,
                    workspace_root=PROJECT_ROOT,
                )
            except (OSError, ValueError):
                continue
            records.append(payload)
    return sorted(records, key=lambda item: (str(item.get("updated_at") or ""), str(item.get("import_id") or "")))


def import_record_for(dataset_id: str, import_id_value: str, *, app_dir: Path | None = None) -> dict[str, Any] | None:
    record_app = app_dir.resolve() if app_dir is not None else import_app_dir(dataset_id).resolve()
    path = import_record_path(dataset_id, import_id_value, app_dir=record_app)
    if not path.is_file():
        return None
    try:
        return read_import_record(
            path,
            expected_dataset_id=dataset_id,
            expected_app_dir=record_app,
            workspace_root=PROJECT_ROOT,
        )
    except (OSError, ValueError):
        return None


def _metadata_overrides(metadata: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(metadata)
    for key in ("frame_rate_hz", "pixel_size_microns", "modality", "indicator"):
        if key not in payload:
            continue
        value = payload.get(key)
        if value in (None, ""):
            merged[key] = None
        elif key in {"frame_rate_hz", "pixel_size_microns"}:
            numeric = float(value)
            if numeric <= 0:
                raise ValueError(f"{key} must be positive when supplied")
            merged[key] = numeric
        else:
            merged[key] = str(value).strip() or None
    return merged


def persist_import_record(record: Mapping[str, Any]) -> dict[str, Any]:
    candidate = dict(record)
    dataset_id = str(candidate.get("dataset_id") or "")
    if not dataset_id or normalize_dataset_id(dataset_id) != dataset_id:
        raise ValueError("Import record dataset_id must be normalized.")
    import_value = str(candidate.get("import_id") or "")
    if not import_value or safe_run_id(import_value) != import_value:
        raise ValueError("Import record import_id is not filesystem-safe.")
    validate_dict(candidate, "dataset_import")
    app_dir = import_app_for_record(candidate)
    target = import_record_path(dataset_id, import_value, app_dir=app_dir)
    with dataset_lock(dataset_id):
        current = import_record_for(dataset_id, import_value, app_dir=app_dir)
        if current == candidate:
            return candidate
        revision = int(candidate.get("revision") or 1)
        if current is None and revision != 1:
            raise ValueError("A new import record must begin at revision 1.")
        if current is not None and revision != int(current.get("revision") or 1) + 1:
            raise ValueError("Import record revision conflict.")
        atomic_write_import_record(target, candidate)
    return candidate


def promote_import_primary_video(record: Mapping[str, Any], *, replace: bool = False) -> dict[str, Any]:
    """Explicitly promote one verified video import to the canonical dataset manifest."""

    dataset_id = str(record.get("dataset_id") or "")
    if (record.get("metadata") or {}).get("kind") != "video":
        raise ValueError("Only a video import can be promoted as the primary dataset video.")
    with dataset_lock(dataset_id):
        app_dir = import_app_for_record(record)
        current = import_record_for(dataset_id, str(record.get("import_id") or ""), app_dir=app_dir)
        if current is None:
            raise ValueError("Import must be persisted before primary-video promotion.")
        source = source_path_for_import(current)
        verify_source_identity(source, current)
        manifest_path = app_dir / "dataset_manifest.generated.json"
        previous_import_id = ""
        if manifest_path.is_file():
            existing = load_json(manifest_path)
            validate_dict(existing, "dataset")
            previous_import_id = str((existing.get("source") or {}).get("import_id") or "")
            if previous_import_id and previous_import_id != current["import_id"] and not replace:
                raise FileExistsError("A different primary video is already configured; explicit replacement is required.")
        manifest = dataset_manifest_from_import(current, app_dir=app_dir)
        validate_dict(manifest, "dataset")
        atomic_write_json(manifest_path, manifest)
        promoted = update_import_record(current, source_role="primary_video", is_primary_video=True)
        promoted = persist_import_record(promoted)
        if replace and previous_import_id and previous_import_id != promoted["import_id"]:
            previous = import_record_for(dataset_id, previous_import_id, app_dir=app_dir)
            if previous is not None:
                persist_import_record(update_import_record(previous, source_role="primary_video_candidate", is_primary_video=False))
        return promoted


def register_import_source(
    *,
    dataset_id: str,
    source: Path,
    source_mode: str,
    destination_path: str,
    metadata_payload: Mapping[str, Any] | None = None,
    promote_primary_video: bool = False,
    replace_primary_video: bool = False,
    app_dir: Path | None = None,
) -> dict[str, Any]:
    inspected = inspect_source(source, workspace_root=PROJECT_ROOT)
    metadata = _metadata_overrides(inspected["metadata"], metadata_payload or {})
    normalized_dataset_id = normalize_dataset_id(dataset_id or source.stem)
    record = make_import_record(
        dataset_id=normalized_dataset_id,
        import_id_value=import_id(),
        source_mode=source_mode,
        original_name=source.name,
        source_path=relative_workspace_path(source, workspace_root=PROJECT_ROOT),
        destination_path=destination_path,
        metadata=metadata,
        warnings=inspected["warnings"],
    )
    record_app = (app_dir or import_app_dir(normalized_dataset_id)).resolve()
    outputs = (PROJECT_ROOT / "Outputs").resolve()
    if record_app.name != "app" or outputs not in record_app.parents:
        raise ValueError("Import app must be an app directory under Outputs.")
    record["app_dir"] = relative_workspace_path(record_app, workspace_root=PROJECT_ROOT)
    persisted = persist_import_record(record)
    if promote_primary_video:
        persisted = promote_import_primary_video(persisted, replace=replace_primary_video)
    return persisted


def persist_failed_upload_record(
    *,
    dataset_id: str,
    app_dir: Path,
    filename: str,
    destination_path: str,
    received_size: int,
    sha256: str,
    error: Exception | str,
) -> dict[str, Any]:
    kind = source_kind(Path(filename))
    if kind not in {"video", "label_table", "neurev_json"}:
        raise ValueError("Failed upload record requires a supported source suffix.")
    metadata = {
        "format": Path(filename).suffix.lower().lstrip("."),
        "kind": kind,
        "original_name": filename,
        "size_bytes": int(received_size),
        "sha256": str(sha256),
        "frame_rate_hz": None,
        "pixel_size_microns": None,
        "modality": None,
        "indicator": None,
    }
    record = make_import_record(
        dataset_id=dataset_id,
        import_id_value=import_id(),
        source_mode="upload",
        original_name=filename,
        source_path=destination_path,
        destination_path=destination_path,
        metadata=metadata,
        warnings=("Upload failed; staged bytes were removed and the intended source is unavailable.",),
        state="failed",
    )
    record["app_dir"] = relative_workspace_path(app_dir, workspace_root=PROJECT_ROOT)
    record["error"] = str(error)
    record["upload_failure"] = {"received_size_bytes": int(received_size), "source_available": False}
    return persist_import_record(record)


def sampled_video_qc(source: Path, *, dataset_id: str, max_frames: int = MAX_QC_SAMPLE_FRAMES) -> dict[str, Any]:
    """Compute QC from a bounded, uniformly spaced frame sample."""

    metadata = inspect_source(source)["metadata"]
    total = int(metadata.get("frames") or 0)
    if total <= 0:
        raise ValueError("Video has no frames to inspect.")
    import numpy as np

    height = int(metadata.get("height") or 0)
    width = int(metadata.get("width") or 0)
    pixels = height * width
    if pixels <= 0 or pixels > MAX_FRAME_PIXELS:
        raise ValueError("Video frame dimensions exceed the bounded QC safety limit.")
    frame_bytes = pixels * max(1, np.dtype(str(metadata.get("dtype") or "float32")).itemsize)
    memory_budget = min(MAX_RENDER_WORKING_BYTES, max(1, int(_available_memory_bytes() * 0.2)))
    memory_limited_frames = max(1, memory_budget // max(1, frame_bytes * 8))
    sample_count = min(max(1, int(max_frames)), total, memory_limited_frames)
    if sample_count * frame_bytes * 8 > memory_budget:
        raise ValueError("Video frames exceed the bounded QC memory budget.")
    indices = np.linspace(0, total - 1, sample_count, dtype=int).tolist()
    wanted = set(indices)
    frames: list[Any] = []
    for chunk in iter_video_chunks(source, chunk_size=64, mmap=True):
        local = [index - chunk.start_frame for index in indices if chunk.start_frame <= index < chunk.end_frame and index in wanted]
        for offset in local:
            frames.append(np.asarray(chunk.data[offset]))
        if len(frames) >= sample_count:
            break
    if not frames:
        raise ValueError("Could not read any sample frames.")
    sampled = np.stack(frames[:sample_count], axis=0)
    qc = compute_video_qc(sampled, dataset_id=dataset_id, source_path=str(source))
    qc["coverage"] = {"mode": "uniform_sample", "sampled_frames": len(frames[:sample_count]), "total_frames": total}
    return qc


def _render_import_frames(source: Path, frames_dir: Path, *, max_frames: int = MAX_RENDER_FRAMES, progress: Any = None) -> int:
    """Render grayscale frame PNGs in bounded chunks for the normal annotator."""

    import numpy as np
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Processing imported videos requires Pillow for frame rendering.") from exc
    metadata = inspect_source(source)["metadata"]
    total = int(metadata.get("frames") or 0)
    if total <= 0:
        raise ValueError("Video has no frames to render.")
    if total > max_frames:
        raise ValueError(f"Video has {total:,} frames; safe browser rendering is limited to {max_frames:,} frames. Use Research Tools for tiled processing.")
    height = int(metadata.get("height") or 0)
    width = int(metadata.get("width") or 0)
    pixels = height * width
    if pixels <= 0 or pixels > MAX_FRAME_PIXELS:
        raise ValueError("Video frame dimensions exceed the bounded renderer safety limit.")
    source_itemsize = max(1, np.dtype(str(metadata.get("dtype") or "float32")).itemsize)
    working_budget = min(MAX_RENDER_WORKING_BYTES, max(1, int(_available_memory_bytes() * 0.2)))
    per_frame_working = pixels * (source_itemsize + 9)
    chunk_size = min(32, total, max(1, working_budget // max(1, per_frame_working)))
    if per_frame_working > working_budget:
        raise ValueError("One video frame exceeds the bounded renderer memory budget.")
    frames_dir.parent.mkdir(parents=True, exist_ok=True)
    estimated_bytes = total * pixels
    available_bytes = shutil.disk_usage(frames_dir.parent).free
    if estimated_bytes > MAX_RENDER_ESTIMATED_BYTES or estimated_bytes > int(available_bytes * 0.65):
        raise ValueError("Safe browser rendering would exceed the local disk guard; use Research Tools for tiled processing.")
    frames_dir.mkdir(parents=True, exist_ok=True)
    sample_values: list[Any] = []
    sample_count = min(chunk_size, total)
    for chunk in iter_video_chunks(source, chunk_size=sample_count, mmap=True, start_frame=0, end_frame=sample_count):
        sample_values.append(np.asarray(chunk.data))
    sample = np.concatenate(sample_values, axis=0) if sample_values else np.zeros((1, height, width), dtype=np.float32)
    low, high = np.percentile(sample.astype(np.float32), [1, 99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(sample)), float(np.max(sample) or np.min(sample) + 1.0)
    written = 0
    for chunk in iter_video_chunks(source, chunk_size=chunk_size, mmap=True):
        for offset in range(chunk.frame_count):
            frame = np.asarray(chunk.data[offset]).astype(np.float32, copy=False)
            normalized = np.clip((frame - low) * (255.0 / max(high - low, 1e-9)), 0, 255).astype(np.uint8)
            Image.fromarray(normalized, mode="L").save(frames_dir / f"frame_{chunk.start_frame + offset + 1:06d}.png")
            written += 1
            if progress is not None and total:
                progress(written / total)
    return written


def _commit_upload_without_overwrite(partial: Path, destination: Path) -> None:
    """Atomically publish an upload while refusing duplicate destinations."""

    try:
        os.link(partial, destination)
    except FileExistsError:
        raise ValueError(f"Upload destination already exists: {destination.name}") from None
    except OSError:
        # Some filesystems do not support hard links. Exclusive creation keeps
        # the no-overwrite contract while copying in bounded chunks.
        try:
            with partial.open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        except FileExistsError:
            raise ValueError(f"Upload destination already exists: {destination.name}") from None
        except Exception:
            destination.unlink(missing_ok=True)
            raise
    finally:
        partial.unlink(missing_ok=True)


def _available_memory_bytes() -> int:
    """Return a conservative available-memory estimate without optional dependencies."""

    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        return 512 * 1024 * 1024


class GenerationJob:
    """Mutable status record for a generated workbench run."""

    def __init__(self, *, app_dir: Path, payload: dict[str, Any]) -> None:
        self.job_id = uuid.uuid4().hex[:12]
        self.app_dir = app_dir.resolve()
        self.output_app_dir = self.app_dir / "generated_runs" / safe_run_id(
            str(payload.get("run_id") or "current_review_pipeline")
        )
        self.output_root = self.output_app_dir / "pipeline_outputs"
        self.payload = dict(payload)
        self.run_id = str(payload.get("run_id") or "current_review_pipeline")
        self.dataset_id = str(payload.get("dataset_id") or self.app_dir.parent.name)
        self.backend = str(payload.get("backend") or "auto")
        self.preview = bool(payload.get("preview"))
        self.status = "queued"
        self.stage = "queued"
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.return_code: int | None = None
        self.outputs: dict[str, str] = {}
        self.error = ""
        self.log_lines: list[str] = []

    def append_log(self, line: str) -> None:
        self.log_lines.append(line.rstrip())
        if len(self.log_lines) > MAX_LOG_LINES:
            self.log_lines = self.log_lines[-MAX_LOG_LINES:]

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "backend": self.backend,
            "preview": self.preview,
            "output_app_dir": str(self.output_app_dir),
            "status": self.status,
            "stage": self.stage,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "return_code": self.return_code,
            "outputs": self.outputs,
            "error": self.error,
            "log_tail": self.log_lines[-80:],
        }


class JobRegistry:
    """Thread-safe in-memory registry for local generation jobs."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jobs: dict[str, GenerationJob] = {}

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            return [job.as_dict() for job in self.jobs.values()]

    def get(self, job_id: str) -> GenerationJob | None:
        with self.lock:
            return self.jobs.get(job_id)

    def active_for(self, app_dir: Path, run_id: str) -> GenerationJob | None:
        with self.lock:
            for job in self.jobs.values():
                if job.app_dir == app_dir.resolve() and job.run_id == run_id and job.status in {"queued", "running"}:
                    return job
        return None

    def add(self, job: GenerationJob) -> None:
        with self.lock:
            self.jobs[job.job_id] = job


JOBS = JobRegistry()


def safe_run_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)
    return cleaned.strip("._") or "run"


def rel_to_app(app_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(app_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def owner_token_required() -> bool:
    return bool(os.environ.get("NEUROBENCH_OWNER_TOKEN"))


def owner_token_matches(value: str | None) -> bool:
    expected = os.environ.get("NEUROBENCH_OWNER_TOKEN")
    if not expected:
        return True
    return value == expected


def threshold_tag(value: float) -> str:
    return f"{round(value * 10):03d}"


def generation_labels(payload: dict[str, Any]) -> tuple[str, str]:
    sigma = payload.get("sigma_label")
    seed = payload.get("component_seed_z")
    grow = payload.get("component_grow_z")
    min_area = payload.get("component_min_area_px")
    if sigma is None:
        sigma = "06"
    sigma_label = f"sigma{sigma}" if not str(sigma).startswith("sigma") else str(sigma)
    if seed is not None or grow is not None or min_area is not None:
        seed_v = float(seed if seed is not None else 2.0)
        grow_v = float(grow if grow is not None else 1.1)
        min_v = int(float(min_area if min_area is not None else 4))
        preset_tag = f"run_seed{threshold_tag(seed_v)}_grow{threshold_tag(grow_v)}_min{min_v}"
    else:
        preset_tag = "balanced_seed017_grow009_min3"
    return sigma_label, preset_tag


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def validated_architecture_runs_update(path: Path, payload: Any) -> dict[str, Any]:
    """Validate new runs strictly while retaining unchanged legacy pipelines."""

    if not isinstance(payload, dict):
        raise ValueError("Architecture-run manifest must be an object.")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Architecture-run manifest runs must be an array.")

    # Validate the manifest envelope and its auxiliary collections separately.
    # Some retained historical runs predate required run-level fields, so a
    # whole-manifest validation would prevent an unchanged file from being
    # saved. New or modified rows are still validated strictly below.
    envelope = dict(payload)
    envelope["runs"] = []
    validate_dict(envelope, "architecture_runs")
    existing_payload = load_json(path) if path.is_file() else {"runs": []}
    existing_runs = {
        str(run.get("run_id")): run
        for run in existing_payload.get("runs", [])
        if isinstance(run, dict) and run.get("run_id")
    }
    dataset_id = str(payload.get("dataset_id") or "")
    normalized_runs: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"Architecture run at index {index} must be an object.")
        run_id = str(run.get("run_id") or "")
        if not run_id:
            raise ValueError(f"Architecture run at index {index} is missing run_id.")
        if str(run.get("dataset_id") or "") != dataset_id:
            raise ValueError(
                f"Run '{run_id}' dataset_id must match manifest dataset_id '{dataset_id}'."
            )
        if run_id in seen_run_ids:
            raise ValueError(f"Duplicate architecture run_id '{run_id}'.")
        seen_run_ids.add(run_id)
        existing = existing_runs.get(run_id)
        if existing == run:
            normalized_runs.append(dict(run))
            continue
        candidate_manifest = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "runs": [run],
        }
        try:
            validate_dict(candidate_manifest, "architecture_runs")
            normalized = as_run_manifest(candidate_manifest)["runs"][0]
        except Exception as exc:
            if existing is None or existing.get("pipeline") != run.get("pipeline"):
                raise ValueError(
                    f"Run '{run_id or index}' has invalid new or modified pipeline metadata: {exc}"
                ) from exc
            # A metadata-only edit may retain an older pipeline that no longer
            # normalizes, but the edited run must satisfy today's JSON schema.
            try:
                validate_dict(candidate_manifest, "architecture_runs")
            except Exception as schema_exc:
                raise ValueError(
                    f"Run '{run_id or index}' has invalid new or modified pipeline metadata: {schema_exc}"
                ) from schema_exc
            normalized = dict(run)
        normalized_runs.append(normalized)
    result = dict(payload)
    result["runs"] = normalized_runs
    return result


def resolve_materialization_raw_video(app_dir: Path, payload: dict[str, Any], review_data: dict[str, Any]) -> Path:
    raw_video = payload.get("raw_video")
    if raw_video:
        raw_path = Path(str(raw_video)).expanduser()
        if not raw_path.is_absolute():
            raw_path = (PROJECT_ROOT / raw_path).resolve()
        return raw_path
    manifest_path = app_dir / "dataset_manifest.generated.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        raw = (manifest.get("paths") or {}).get("raw_video")
        if raw:
            return Path(str(raw)).expanduser().resolve()
    dataset_id = infer_dataset_id(app_dir, review_data)
    inferred = find_raw_video(review_data, dataset_id)
    if inferred is None:
        raise RuntimeError("Could not infer raw video path. Add raw_video to the materialization request.")
    return inferred.resolve()


def load_run(app_dir: Path, run_id: str) -> dict[str, Any] | None:
    path = app_dir / "architecture_runs.json"
    if not path.exists():
        return None
    manifest = as_run_manifest(load_json(path))
    return next((dict(run) for run in manifest.get("runs", []) if run.get("run_id") == run_id), None)


def run_generation_params(run: dict[str, Any] | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if not run:
        return params
    for stage in normalize_pipeline(run.get("pipeline") or []):
        stage_id = stage.get("stage_id") or stage.get("op") or stage.get("name")
        stage_params = dict(stage.get("params") or {})
        if stage_id == "temporal_highpass_gaussian" and "sigma_frames" in stage_params:
            params["sigma_frames"] = stage_params["sigma_frames"]
            sigma_frames = float(stage_params["sigma_frames"])
            params["sigma_label"] = f"{int(sigma_frames):02d}" if sigma_frames.is_integer() else f"{round(sigma_frames * 10):03d}"
        if stage_id == "component_filter":
            for key in ("seed_z", "grow_z", "min_area_px", "max_area_px"):
                if key in stage_params:
                    params[f"component_{key}"] = stage_params[key]
        if stage_id in {"robust_kalman_positive_innovation", "trace_event_scoring", "candidate_event_pipeline"} and "event_threshold_z" in stage_params:
            params["event_threshold_z"] = stage_params["event_threshold_z"]
    return params


def environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.executable,
        "fiji": "",
        "fiji_available": False,
        "owner_token_required": owner_token_required(),
        "modules": {},
        "gpu": {"torch": False, "cuda": False, "cupy": False},
    }
    fiji = Path(shutil.which("fiji") or DEFAULT_FIJI)
    report["fiji"] = str(fiji)
    report["fiji_available"] = bool(fiji.exists())
    modules = {}
    for name in ["PIL", "numpy", "scipy", "tifffile", "torch", "cupy"]:
        modules[name] = importlib.util.find_spec(name) is not None
    report["modules"] = modules
    if modules.get("torch"):
        try:
            import torch  # type: ignore

            report["gpu"]["torch"] = True
            report["gpu"]["cuda"] = bool(torch.cuda.is_available())
            report["gpu"]["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        except Exception as exc:
            report["gpu"]["torch_error"] = str(exc)
    report["gpu"]["cupy"] = bool(modules.get("cupy"))
    return report


def infer_dataset_id(app_dir: Path, review_data: dict[str, Any] | None = None) -> str:
    return dataset_id_from_review(review_data or {}, fallback=app_dir.parent.name)


def find_raw_video(review_data: dict[str, Any], dataset_id: str) -> Path | None:
    explicit = raw_video_from_review(review_data)
    if explicit:
        declared = Path(explicit).expanduser()
        declared = declared.resolve() if declared.is_absolute() else (PROJECT_ROOT / declared).resolve()
        if declared.is_file():
            return declared
    video_name = review_data.get("video", {}).get("name")
    candidates: list[Path] = []
    if video_name:
        candidates.extend(PROJECT_ROOT.glob(f"Inputs/**/{video_name}"))
    candidates.extend(PROJECT_ROOT.glob(f"Inputs/**/*{dataset_id}*.tif"))
    candidates.extend(PROJECT_ROOT.glob(f"Inputs/**/*{dataset_id}*.tiff"))
    for path in candidates:
        if path.is_file():
            return path
    return None


def generated_dataset_manifest(app_dir: Path, payload: dict[str, Any], *, output_app_dir: Path | None = None) -> Path:
    output_app_dir = (output_app_dir or app_dir).resolve()
    review_data_path = app_dir / "review_data.json"
    review_data = load_json(review_data_path) if review_data_path.exists() else {}
    dataset_id = str(payload.get("dataset_id") or infer_dataset_id(app_dir, review_data))
    raw_video = payload.get("raw_video")
    raw_path = Path(raw_video).expanduser() if raw_video else find_raw_video(review_data, dataset_id)
    if raw_path is None:
        raise RuntimeError("Could not infer raw video path. Add raw_video to the generation request or dataset manifest.")
    if not raw_path.is_absolute():
        raw_path = (PROJECT_ROOT / raw_path).resolve()
    dataset = review_data.get("dataset") if isinstance(review_data.get("dataset"), dict) else {}
    video = review_data.get("video") if isinstance(review_data.get("video"), dict) else {}
    manifest = build_dataset_intake_manifest(
        dataset_id=dataset_id,
        raw_video=str(raw_path),
        app_dir=output_app_dir,
        frame_rate_hz=payload.get("frame_rate_hz") or dataset.get("frame_rate_hz") or video.get("frameRateHz"),
        pixel_size_microns=payload.get("pixel_size_microns") or dataset.get("pixel_size_microns"),
        name=video.get("name") or raw_path.name,
        modality=dataset.get("modality"),
        indicator=dataset.get("indicator"),
    )
    manifest["paths"]["architecture_runs"] = str(app_dir / "architecture_runs.json")
    out = output_app_dir / "dataset_manifest.generated.json"
    atomic_write_json(out, manifest)
    return out


def ensure_run_record(
    app_dir: Path,
    run_id: str,
    dataset_id: str,
    status: str,
    *,
    output_app_dir: Path | None = None,
    output_root: Path | None = None,
) -> None:
    path = app_dir / "architecture_runs.json"
    manifest = as_run_manifest(load_json(path)) if path.exists() else {"schema_version": 1, "dataset_id": dataset_id, "runs": []}
    runs = list(manifest.get("runs") or [])
    run = next((item for item in runs if item.get("run_id") == run_id), None)
    if run is None:
        run = {"schema_version": 1, "run_id": run_id, "dataset_id": dataset_id, "label": run_id.replace("_", " "), "pipeline": []}
        runs.append(run)
    run["execution"] = dict(run.get("execution") or {}, status=status)
    if output_root is not None:
        run["execution"]["output_root"] = str(output_root)
    artifacts = dict(run.get("artifacts") or {})
    artifact_app_dir = (output_app_dir or app_dir).resolve()
    artifacts.update(
        {
            "review_data": rel_to_app(app_dir, artifact_app_dir / "review_data.json"),
            "app_url": rel_to_app(app_dir, artifact_app_dir / "index.html"),
            "frames": rel_to_app(app_dir, artifact_app_dir / "frames"),
        }
    )
    run["artifacts"] = artifacts
    manifest["runs"] = runs
    manifest["dataset_id"] = manifest.get("dataset_id") or dataset_id
    atomic_write_json(path, manifest)


def import_llm_proposals_into_app(app_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and merge an LLM proposal set into a workbench architecture manifest."""

    proposal = payload.get("proposal") or payload.get("proposal_set") or payload
    if not isinstance(proposal, dict):
        raise ValueError("LLM proposal import payload must be an object or contain a proposal object.")
    max_combinations = payload.get("max_combinations")
    max_combinations_int = int(max_combinations) if max_combinations is not None else None
    try:
        validated = validate_proposal_set(proposal, max_combinations=max_combinations_int)
    except Exception as exc:
        raise ValueError(f"Invalid LLM proposal set: {exc}") from exc
    path = app_dir / "architecture_runs.json"
    base = as_run_manifest(load_json(path)) if path.exists() else None
    manifest = proposal_set_to_architecture_manifest(
        validated,
        base_manifest=base,
        max_combinations=max_combinations_int,
    )
    atomic_write_json(path, manifest)
    proposal_set_id = str(validated.get("proposal_set_id") or "")
    proposal_run_ids = [
        str(run.get("run_id"))
        for run in manifest.get("runs", [])
        if (run.get("artifacts") or {}).get("proposal_set_id") == proposal_set_id
    ]
    template_ids = [
        str(template.get("id"))
        for template in manifest.get("saved_pipelines", [])
        if template.get("proposal_set_id") == proposal_set_id
    ]
    return {
        "ok": True,
        "architecture_runs": str(path),
        "proposal_set_id": proposal_set_id,
        "run_ids": proposal_run_ids,
        "saved_pipeline_ids": template_ids,
        "validation_report": validated.get("validation_report", {}),
    }


def run_process(job: GenerationJob, command: list[str], *, stage: str, env: dict[str, str] | None = None) -> int:
    job.stage = stage
    job.append_log("+ " + " ".join(shlex.quote(str(part)) for part in command))
    proc = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        job.append_log(line)
    proc.wait()
    return int(proc.returncode)


def export_known_intermediates(job: GenerationJob) -> None:
    app_dir = job.app_dir
    dataset_id = job.dataset_id
    run_id = job.run_id
    sigma_label, preset_tag = generation_labels(job.payload)
    output_root = job.output_root
    specs = [
        ("temporal_highpass_gaussian", "Temporal high-pass", output_root / "HighPass" / dataset_id / f"{dataset_id}_hp_gaussian_{sigma_label}f_float32.tif"),
        ("event_preserving_noise_suppression", "Event-preserving denoise/z", output_root / "EventPreservingNoiseSuppression" / dataset_id / f"{dataset_id}_{sigma_label}_positive_local_z_float32.tif"),
        ("robust_positive_local_z", "Robust positive local-z", output_root / "CandidateEventPipeline" / dataset_id / f"{dataset_id}_{sigma_label}_robust_positive_z_float32.tif"),
        ("component_filter", "Candidate mask", output_root / "CandidateEventPipeline" / dataset_id / f"{dataset_id}_{sigma_label}_{preset_tag}_mask.tif"),
        ("trace_event_scoring", "Temporal candidate mask", output_root / "TemporalCandidateScoring" / dataset_id / f"{dataset_id}_{sigma_label}_{preset_tag}_score_ge_050_mask.tif"),
    ]
    for stage_id, label, tif_path in specs:
        if not tif_path.exists():
            job.append_log(f"skip intermediate {stage_id}: missing {tif_path}")
            continue
        out_dir = app_dir / "generated_runs" / safe_run_id(run_id) / "intermediates" / stage_id
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "tools/export_intermediate_frames.py"),
            "--input-tif",
            str(tif_path),
            "--out-dir",
            str(out_dir),
            "--architecture-runs",
            str(app_dir / "architecture_runs.json"),
            "--run-id",
            run_id,
            "--stage-id",
            stage_id,
            "--label",
            label,
        ]
        code = run_process(job, cmd, stage=f"export {stage_id}")
        if code != 0:
            job.append_log(f"intermediate export failed for {stage_id} with exit code {code}")


def execute_generation_job(job: GenerationJob) -> None:
    job.status = "running"
    job.started_at = time.time()
    try:
        env = environment_report()
        if job.backend not in ALLOWED_BACKENDS:
            raise RuntimeError(f"Unsupported backend: {job.backend}")
        if job.backend == "python_gpu" and not env.get("gpu", {}).get("cuda"):
            job.status = "blocked"
            job.error = "Python GPU backend requested, but Torch CUDA is not available."
            job.append_log(job.error)
            return
        if job.backend == "python_gpu":
            job.append_log("Python GPU generation is not yet a full Review builder; using whitelisted Fiji/Groovy review generation after GPU readiness check.")
        run = load_run(job.app_dir, job.run_id)
        job.payload.update(run_generation_params(run))
        job.output_app_dir.mkdir(parents=True, exist_ok=True)
        manifest = generated_dataset_manifest(job.app_dir, job.payload, output_app_dir=job.output_app_dir)
        job.dataset_id = load_json(manifest)["dataset_id"]
        ensure_run_record(job.app_dir, job.run_id, job.dataset_id, "running", output_app_dir=job.output_app_dir, output_root=job.output_root)
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "tools/run_neuron_review_pipeline.py"),
            "--dataset-manifest",
            str(manifest),
            "--output-root",
            str(job.output_root),
            "--architecture-runs",
            str(job.app_dir / "architecture_runs.json"),
            "--run-id",
            job.run_id,
            "--stages",
            str(job.payload.get("stages") or GENERATION_STAGES),
        ]
        code = run_process(job, cmd, stage="review pipeline")
        job.return_code = code
        if code != 0:
            job.status = "failed"
            job.error = f"review pipeline exited with code {code}"
            ensure_run_record(job.app_dir, job.run_id, job.dataset_id, "failed", output_app_dir=job.output_app_dir, output_root=job.output_root)
            return
        if job.payload.get("generate_intermediates", True):
            export_known_intermediates(job)
        ensure_run_record(job.app_dir, job.run_id, job.dataset_id, "completed", output_app_dir=job.output_app_dir, output_root=job.output_root)
        job.outputs = {
            "review_data": str(job.output_app_dir / "review_data.json"),
            "architecture_runs": str(job.app_dir / "architecture_runs.json"),
            "app_url": str(job.output_app_dir / "index.html"),
        }
        job.status = "completed"
        job.stage = "completed"
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        job.append_log(f"ERROR: {exc}")
        try:
            ensure_run_record(job.app_dir, job.run_id, job.dataset_id, "failed", output_app_dir=job.output_app_dir, output_root=job.output_root)
        except Exception:
            pass
    finally:
        job.finished_at = time.time()


class WorkbenchHandler(BaseHTTPRequestHandler):
    """HTTP handler for static workbench files, autosave, and generation jobs."""

    app_dir: Path
    root_dir: Path | None = None
    asset_mode = "current"
    dataset_apps: Mapping[str, Path] = MappingProxyType({})
    dataset_registry_conflicts: Mapping[str, tuple[str, ...]] = MappingProxyType({})
    created_dataset_apps: dict[str, Path]
    dataset_registry_lock: threading.RLock
    POST_HANDLERS = {
        ("jobs", "generate-view"): "_handle_generation_post",
        ("jobs", "generate-preview"): "_handle_generation_post",
        ("materialize-traces",): "_handle_materialize_traces_post",
        ("llm-proposals", "import"): "_handle_llm_proposal_import_post",
        ("imports", "register"): "_handle_import_register_post",
        ("imports", "upload"): "_handle_import_upload_post",
        ("imports", "promote"): "_handle_import_promote_post",
        ("imports", "metadata"): "_handle_import_metadata_post",
        ("imports", "qc"): "_handle_import_qc_post",
        ("imports", "process"): "_handle_import_process_post",
        ("labels", "preview"): "_handle_label_preview_post",
        ("labels", "import"): "_handle_label_import_post",
        ("neurev", "preview"): "_handle_neurev_preview_post",
        ("neurev", "import"): "_handle_neurev_import_post",
    }

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str, *, include_body: bool = True) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any] | list[Any], *, include_body: bool = True) -> None:
        self._send(status, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n", "application/json", include_body=include_body)

    def _known_dataset_app(self, dataset_id: str) -> Path | None:
        with self.dataset_registry_lock:
            return self.created_dataset_apps.get(dataset_id) or self.dataset_apps.get(dataset_id)

    def _new_dataset_app(self, dataset_id: str) -> Path | None:
        if not dataset_id or normalize_dataset_id(dataset_id) != dataset_id:
            return None
        output_root = (PROJECT_ROOT / "Outputs" / "NeuronReview").resolve()
        candidate = (output_root / dataset_id / "app").resolve()
        if output_root not in candidate.parents:
            return None
        return candidate

    def _remember_created_dataset(self, dataset_id: str, app_dir: Path) -> None:
        expected = self._new_dataset_app(dataset_id)
        resolved = app_dir.resolve()
        if expected is None or resolved != expected:
            return
        with self.dataset_registry_lock:
            existing = self.dataset_apps.get(dataset_id) or self.created_dataset_apps.get(dataset_id)
            if existing is not None and existing != resolved:
                raise ValueError("Dataset ID is already bound to a different app in this server.")
            self.created_dataset_apps[dataset_id] = resolved

    def _virtual_static_target(self, parsed_path: str) -> tuple[Path, Path] | None:
        parts = Path(unquote(parsed_path).lstrip("/")).parts
        if len(parts) < 2 or parts[0] != "_datasets":
            return None
        app_dir = self._known_dataset_app(parts[1])
        if app_dir is None:
            return None
        relative = parts[2:] or ("index.html",)
        if any(part.startswith(".") for part in relative):
            return None
        candidate = app_dir.joinpath(*relative).resolve()
        if app_dir != candidate and app_dir not in candidate.parents:
            return None
        return candidate, app_dir

    def _safe_path(self) -> Path | None:
        parsed = urlparse(self.path)
        virtual = self._virtual_static_target(parsed.path)
        if virtual is not None:
            return virtual[0]
        rel = unquote(parsed.path).lstrip("/") or "index.html"
        if any(part.startswith(".") for part in Path(rel).parts):
            return None
        root = (self.root_dir or self.app_dir).resolve()
        candidate = (root / rel).resolve()
        if candidate == root or root not in candidate.parents:
            return None
        return candidate

    def _safe_put_path(self, parsed_path: str) -> Path | None:
        rel = unquote(parsed_path).lstrip("/")
        parts = Path(rel).parts
        if len(parts) == 3 and parts[0] == "_datasets" and parts[2] in {"annotations.json", "architecture_runs.json"}:
            app_dir = self._known_dataset_app(parts[1])
            if app_dir is None:
                return None
            return (app_dir / parts[2]).resolve()
        if len(parts) == 4 and parts[:2] == ("api", "datasets") and parts[3] in {"annotations", "architecture-runs"}:
            app_dir = self._known_dataset_app(parts[2])
            if app_dir is None:
                return None
            name = "annotations.json" if parts[3] == "annotations" else "architecture_runs.json"
            return (app_dir / name).resolve()
        if self.root_dir is None:
            if rel not in {"annotations.json", "architecture_runs.json"}:
                return None
            return (self.app_dir / rel).resolve()
        if len(parts) != 3 or parts[1] != "app" or parts[2] not in {"annotations.json", "architecture_runs.json"}:
            return None
        root = self.root_dir.resolve()
        candidate = (root / rel).resolve()
        if root not in candidate.parents:
            return None
        try:
            registered = self._known_dataset_app(dataset_id_for_app(candidate.parent))
        except ValueError:
            registered = None
        if registered != candidate.parent:
            return None
        return candidate

    def _api_route(self, parsed_path: str) -> tuple[Path, tuple[str, ...]] | None:
        rel = unquote(parsed_path).lstrip("/")
        parts = Path(rel).parts
        if parts == ("api", "datasets"):
            return self.app_dir.resolve(), ("datasets",)
        if len(parts) >= 3 and parts[:2] == ("api", "datasets"):
            tail = tuple(parts[3:])
            app_dir = self._known_dataset_app(parts[2])
            if app_dir is None and self.command == "POST" and tail in {("imports", "register"), ("imports", "upload")}:
                app_dir = self._new_dataset_app(parts[2])
            if app_dir is None:
                return None
            return app_dir, tail
        if len(parts) >= 3 and parts[0] == "_datasets" and parts[2] == "api":
            tail = tuple(parts[3:])
            app_dir = self._known_dataset_app(parts[1])
            if app_dir is None and self.command == "POST" and tail in {("imports", "register"), ("imports", "upload")}:
                app_dir = self._new_dataset_app(parts[1])
            if app_dir is None:
                return None
            return app_dir, tail
        if self.root_dir is None:
            if not (rel == "api" or rel.startswith("api/")):
                return None
            return self.app_dir.resolve(), tuple(parts[1:])
        if len(parts) >= 3 and parts[1] == "app" and parts[2] == "api":
            root = self.root_dir.resolve()
            candidate = (root / parts[0] / "app").resolve()
            if root in candidate.parents and candidate.exists():
                try:
                    registered = self._known_dataset_app(dataset_id_for_app(candidate))
                except ValueError:
                    registered = None
                if registered == candidate:
                    return candidate, tuple(parts[3:])
        return None

    def _api_app_dir(self, parsed_path: str) -> Path | None:
        route = self._api_route(parsed_path)
        return route[0] if route is not None else None

    def _api_tail(self, parsed_path: str) -> tuple[str, ...] | None:
        route = self._api_route(parsed_path)
        return route[1] if route is not None else None

    def _read_post_payload(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            self._send_json(400, {"error": "invalid Content-Length"})
            return None
        if length <= 0 or length > MAX_POST_JSON_BYTES:
            self._send_json(413, {"error": "invalid request size"})
            return None
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            self._send_json(400, {"error": f"invalid json: {exc}"})
            return None
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "payload must be an object"})
            return None
        return payload

    def do_OPTIONS(self) -> None:
        self._send(204, b"", "text/plain")

    def do_GET(self) -> None:
        if self._serve_api_get():
            return
        self._serve_file(include_body=True)

    def do_HEAD(self) -> None:
        if self._serve_api_get(include_body=False):
            return
        self._serve_file(include_body=False)

    def _serve_api_get(self, *, include_body: bool = True) -> bool:
        parsed = urlparse(self.path)
        route = self._api_route(parsed.path)
        if route is None:
            return False
        app_dir, route_tail = route
        tail = list(route_tail)
        if not tail:
            self._send_json(200, dataset_record_for_app(app_dir, workspace_root=PROJECT_ROOT), include_body=include_body)
            return True
        if tail == ["environment"]:
            report = environment_report()
            report["dataset_registry_conflicts"] = dict(self.dataset_registry_conflicts)
            self._send_json(200, report, include_body=include_body)
            return True
        if tail == ["dataset"]:
            self._send_json(
                200,
                dataset_record_for_app(app_dir, workspace_root=PROJECT_ROOT),
                include_body=include_body,
            )
            return True
        if tail == ["datasets"]:
            self._send_json(
                200,
                {
                    "schema_version": 1,
                    "kind": "neurobench_dataset_catalog",
                    "datasets": discover_dataset_catalog(PROJECT_ROOT),
                    "registry_conflicts": dict(self.dataset_registry_conflicts),
                },
                include_body=include_body,
            )
            return True
        if tail == ["annotation-revisions"]:
            self._send_json(
                200,
                {"schema_version": 1, "revisions": list_revision_roots(app_dir / "annotation_revisions")},
                include_body=include_body,
            )
            return True
        if len(tail) == 2 and tail[0] == "annotation-revisions":
            try:
                root = resolve_revision_root(app_dir / "annotation_revisions", tail[1])
                if not root.is_dir():
                    self._send_json(404, {"error": "annotation revision not found"}, include_body=include_body)
                else:
                    self._send_json(200, revision_snapshot(root), include_body=include_body)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)}, include_body=include_body)
            return True
        if tail == ["jobs"]:
            self._send_json(200, {"jobs": JOBS.list(), "durable_jobs": durable_job_records_for_app(app_dir)}, include_body=include_body)
            return True
        if tail == ["imports"]:
            self._send_json(200, {"schema_version": 1, "imports": import_records(dataset_id=app_dataset_id(app_dir), app_dir=app_dir)}, include_body=include_body)
            return True
        if len(tail) == 2 and tail[0] == "imports":
            record = import_record_for(app_dataset_id(app_dir), tail[1], app_dir=app_dir)
            if record is None:
                self._send_json(404, {"error": "import not found"}, include_body=include_body)
            else:
                self._send_json(200, record, include_body=include_body)
            return True
        if len(tail) == 2 and tail[0] == "durable-jobs":
            record = durable_job_record_for_app(app_dir, tail[1])
            if record is None:
                self._send_json(404, {"error": "job not found"}, include_body=include_body)
            else:
                self._send_json(200, record, include_body=include_body)
            return True
        if len(tail) == 2 and tail[0] == "jobs":
            job = JOBS.get(tail[1])
            if job is None:
                self._send_json(404, {"error": "job not found"}, include_body=include_body)
                return True
            self._send_json(200, job.as_dict(), include_body=include_body)
            return True
        self._send_json(404, {"error": "unknown api endpoint"}, include_body=include_body)
        return True

    def _materialize_traces(self, app_dir: Path, payload: dict[str, Any]) -> None:
        review_path = app_dir / "review_data.json"
        if not review_path.exists():
            self._send_json(404, {"error": "review_data.json not found"})
            return
        review_data = load_json(review_path)
        annotations_path = app_dir / "annotations.json"
        if isinstance(payload.get("annotations"), dict):
            annotations = dict(payload["annotations"])
        elif annotations_path.exists():
            annotations = load_json(annotations_path)
        else:
            annotations = {"version": 3, "schema_version": 3, "settings": {}, "virtualRois": {}, "runs": {}}
        try:
            raw_path = resolve_materialization_raw_video(app_dir, payload, review_data)
            if not raw_path.exists():
                self._send_json(404, {"error": f"raw video not found: {raw_path}"})
                return
            result = materialize_virtual_roi_traces(
                review_data=review_data,
                annotations=annotations,
                raw_video_path=raw_path,
                run_id=payload.get("run_id") or (annotations.get("settings") or {}).get("activeRunId"),
                roi_ids=payload.get("roi_ids") or None,
                outer_radius_px=int(payload.get("outer_radius_px") or 15),
                neuropil_weight=float(payload.get("neuropil_weight") or 0.7),
                event_threshold_z=float(payload.get("event_threshold_z") or 2.4),
                kalman_gain=float(payload.get("kalman_gain") or 0.06),
                spike_gain=float(payload.get("spike_gain") or 0.008),
                negative_gain=float(payload.get("negative_gain") or 0.11),
            )
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})
            return
        atomic_write_json(annotations_path, result["annotations"])
        self._send_json(
            200,
            {
                "ok": True,
                "raw_video": str(raw_path),
                "materialized_ids": result["materialized_ids"],
                "annotations": result["annotations"],
            },
        )

    def _serve_file(self, *, include_body: bool) -> None:
        path = self._safe_path()
        if path is None:
            self._send(403, b"Forbidden\n", "text/plain", include_body=include_body)
            return
        virtual = self._virtual_static_target(urlparse(self.path).path)
        asset_app: Path | None = virtual[1] if virtual is not None else None
        if asset_app is None and self.root_dir is None:
            asset_app = self.app_dir.resolve()
        if asset_app is None and path.parent.name == "app":
            asset_app = path.parent.resolve()
        if self.asset_mode == "current" and asset_app is not None and path.name in {"index.html", "workbench.css", "workbench.js"}:
            review_path = asset_app / "review_data.json"
            if review_path.is_file():
                try:
                    from neurobench.workbench.builder import render_workbench_assets

                    manifest_path = asset_app / "dataset_manifest.generated.json"
                    manifest = load_json(manifest_path) if manifest_path.is_file() else None
                    assets = render_workbench_assets(
                        review_data_path=review_path,
                        dataset_id=app_dataset_id(asset_app),
                        dataset_manifest=manifest,
                        architecture_runs_path=(asset_app / "architecture_runs.json") if (asset_app / "architecture_runs.json").is_file() else None,
                        app_dir=asset_app,
                    )
                    body = assets[path.name]
                    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    self._send(200, body, ctype, include_body=include_body)
                    return
                except Exception as exc:
                    self._send_json(500, {"error": f"could not render current workbench assets: {exc}"}, include_body=include_body)
                    return
        if not path.exists() or not path.is_file():
            self._send(404, b"Not found\n", "text/plain", include_body=include_body)
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), ctype, include_body=include_body)

    def do_PUT(self) -> None:
        if not owner_token_matches(self.headers.get("X-Neurobench-Owner-Token")):
            self._send_json(401, {"error": "owner token required"})
            return
        parsed = urlparse(self.path)
        out = self._safe_put_path(parsed.path)
        if out is None:
            self._send(404, b"Only per-dataset annotations.json and architecture_runs.json can be updated\n", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            self._send(400, b"Invalid Content-Length\n", "text/plain")
            return
        if length <= 0 or length > 20_000_000:
            self._send(413, b"Invalid request size\n", "text/plain")
            return
        raw = self.rfile.read(length)
        try:
            parsed_json = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._send(400, f"Invalid JSON: {exc}\n".encode(), "text/plain")
            return
        if not isinstance(parsed_json, dict):
            self._send(400, b"JSON document must be an object\n", "text/plain")
            return
        if out.name == "architecture_runs.json":
            try:
                parsed_json = validated_architecture_runs_update(out, parsed_json)
            except Exception as exc:
                self._send(400, f"Invalid architecture run manifest: {exc}\n".encode(), "text/plain")
                return
        atomic_write_json(out, parsed_json)
        self._send(200, b'{"ok":true}\n', "application/json")

    def do_POST(self) -> None:
        if not owner_token_matches(self.headers.get("X-Neurobench-Owner-Token")):
            self._send_json(401, {"error": "owner token required"})
            return
        parsed = urlparse(self.path)
        app_dir = self._api_app_dir(parsed.path)
        if app_dir is None:
            self._send_json(404, {"error": "unknown api endpoint"})
            return
        tail = self._api_tail(parsed.path)
        revision_action = (
            tail[2]
            if tail and len(tail) == 3 and tail[0] == "annotation-revisions"
            else None
        )
        if tail == ("annotation-revisions",) or revision_action in {"operations", "fork", "publish"}:
            payload = self._read_post_payload()
            if payload is None:
                return
            handler = {
                "operations": self._handle_revision_append_post,
                "fork": self._handle_revision_fork_post,
                "publish": self._handle_revision_publish_post,
            }.get(revision_action, self._handle_revision_create_post)
            handler(app_dir, payload, tail or ())
            return
        if tail and tail[0] == "imports":
            if tail == ("imports", "upload"):
                self._handle_import_upload_post(app_dir, None, tail)
                return
            dynamic_action = tail[2] if len(tail) == 3 else None
            if tail in self.POST_HANDLERS or dynamic_action in {"metadata", "qc", "process", "promote"}:
                payload = self._read_post_payload()
                if payload is None:
                    return
                if dynamic_action:
                    payload["import_id"] = tail[1]
                if tail == ("imports", "register"):
                    self._handle_import_register_post(app_dir, payload, tail)
                elif tail == ("imports", "metadata") or dynamic_action == "metadata":
                    self._handle_import_metadata_post(app_dir, payload, tail)
                elif tail == ("imports", "qc") or dynamic_action == "qc":
                    self._handle_import_qc_post(app_dir, payload, tail)
                elif tail == ("imports", "process") or dynamic_action == "process":
                    self._handle_import_process_post(app_dir, payload, tail)
                elif tail == ("imports", "promote") or dynamic_action == "promote":
                    self._handle_import_promote_post(app_dir, payload, tail)
                else:
                    self._send_json(404, {"error": "unknown import endpoint"})
                return
        handler_name = self.POST_HANDLERS.get(tail or ())
        if handler_name is None:
            self._send_json(404, {"error": "unknown api endpoint"})
            return
        payload = self._read_post_payload()
        if payload is None:
            return
        getattr(self, handler_name)(app_dir, payload, tail or ())


    def _handle_revision_create_post(self, app_dir: Path, payload: dict[str, Any], tail: tuple[str, ...]) -> None:
        revision = payload.get("revision")
        annotations = payload.get("annotations")
        operations = payload.get("operations") or []
        if not isinstance(revision, dict) or not isinstance(annotations, dict) or not isinstance(operations, list):
            self._send_json(400, {"error": "revision, annotations, and operations are required draft objects"})
            return
        try:
            with dataset_lock(app_dataset_id(app_dir)):
                root = initialize_revision_root(
                    app_dir / "annotation_revisions",
                    revision=revision,
                    annotations=annotations,
                    operations=operations,
                )
            self._send_json(201, revision_snapshot(root))
        except FileExistsError as exc:
            self._send_json(409, {"error": str(exc)})
        except (OSError, ValueError) as exc:
            self._send_json(400, {"error": str(exc)})

    def _handle_revision_append_post(self, app_dir: Path, payload: dict[str, Any], tail: tuple[str, ...]) -> None:
        operation = payload.get("operation") if isinstance(payload.get("operation"), dict) else payload
        try:
            root = resolve_revision_root(app_dir / "annotation_revisions", tail[1])
            if not root.is_dir():
                self._send_json(404, {"error": "annotation revision not found"})
                return
            with dataset_lock(app_dataset_id(app_dir)):
                snapshot = append_revision_operation(root, operation)
            self._send_json(200, snapshot)
        except RevisionConflictError as exc:
            current = None
            try:
                current = revision_snapshot(root) if root.is_dir() else None
            except (OSError, ValueError):
                pass
            self._send_json(409, {"error": str(exc), "current": current})
        except (OSError, ValueError) as exc:
            self._send_json(400, {"error": str(exc)})

    def _handle_revision_fork_post(self, app_dir: Path, payload: dict[str, Any], tail: tuple[str, ...]) -> None:
        revision_id = payload.get("revisionId")
        reviewer_id = payload.get("reviewerId")
        if not isinstance(revision_id, str) or not isinstance(reviewer_id, str):
            self._send_json(400, {"error": "revisionId and reviewerId are required strings"})
            return
        try:
            revisions_root = app_dir / "annotation_revisions"
            source_root = resolve_revision_root(revisions_root, tail[1])
            if not source_root.is_dir():
                self._send_json(404, {"error": "annotation revision not found"})
                return
            timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            with dataset_lock(app_dataset_id(app_dir)):
                root = fork_revision_root(
                    source_root,
                    revisions_root,
                    revision_id=revision_id,
                    reviewer_id=reviewer_id,
                    timestamp=timestamp,
                )
            self._send_json(201, revision_snapshot(root))
        except FileExistsError as exc:
            self._send_json(409, {"error": str(exc)})
        except (OSError, ValueError) as exc:
            self._send_json(400, {"error": str(exc)})

    def _handle_revision_publish_post(self, app_dir: Path, payload: dict[str, Any], tail: tuple[str, ...]) -> None:
        revision_id = payload.get("revisionId")
        expected_token = payload.get("expectedRevisionToken")
        if (
            not isinstance(revision_id, str)
            or not isinstance(expected_token, int)
            or isinstance(expected_token, bool)
        ):
            self._send_json(400, {"error": "revisionId and integer expectedRevisionToken are required"})
            return
        draft_root: Path | None = None
        try:
            revisions_root = app_dir / "annotation_revisions"
            draft_root = resolve_revision_root(revisions_root, tail[1])
            if not draft_root.is_dir():
                self._send_json(404, {"error": "annotation revision not found"})
                return
            timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            with dataset_lock(app_dataset_id(app_dir)):
                root = publish_revision_root(
                    draft_root,
                    revisions_root,
                    revision_id=revision_id,
                    expected_revision_token=expected_token,
                    timestamp=timestamp,
                )
            self._send_json(201, revision_snapshot(root))
        except RevisionConflictError as exc:
            current = None
            try:
                current = revision_snapshot(draft_root) if draft_root and draft_root.is_dir() else None
            except (OSError, ValueError):
                pass
            self._send_json(409, {"error": str(exc), "current": current})
        except FileExistsError as exc:
            self._send_json(409, {"error": str(exc)})
        except (OSError, ValueError) as exc:
            self._send_json(400, {"error": str(exc)})

    def _handle_materialize_traces_post(self, app_dir: Path, payload: dict[str, Any], tail: tuple[str, ...]) -> None:
        self._materialize_traces(app_dir, payload)

    def _handle_llm_proposal_import_post(self, app_dir: Path, payload: dict[str, Any], tail: tuple[str, ...]) -> None:
        try:
            self._send_json(200, import_llm_proposals_into_app(app_dir, payload))
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})

    def _handle_import_register_post(self, app_dir: Path, payload: dict[str, Any], tail: tuple[str, ...]) -> None:
        try:
            raw_source = payload.get("source_path") or payload.get("path")
            if not raw_source:
                raise ValueError("source_path is required")
            source = resolve_allowed_local_path(raw_source, workspace_root=PROJECT_ROOT)
            dataset = app_dataset_id(app_dir)
            requested_dataset = payload.get("dataset_id")
            if requested_dataset and str(requested_dataset) != dataset:
                raise ValueError("Payload dataset_id does not match the dataset-qualified route.")
            record = register_import_source(
                dataset_id=dataset,
                source=source,
                source_mode="local_registration",
                destination_path=relative_workspace_path(source, workspace_root=PROJECT_ROOT),
                metadata_payload=payload,
                app_dir=app_dir,
            )
            self._remember_created_dataset(dataset, app_dir)
            if bool(payload.get("promote_primary_video")):
                try:
                    record = promote_import_primary_video(record, replace=bool(payload.get("replace_primary_video")))
                except FileExistsError as exc:
                    self._send_json(409, {"error": str(exc), "import": record})
                    return
            self._send_json(201, {"ok": True, "import": record})
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})

    def _handle_import_upload_post(self, app_dir: Path, payload: dict[str, Any] | None, tail: tuple[str, ...]) -> None:
        query = parse_qs(urlparse(self.path).query)
        filename = str((query.get("filename") or [self.headers.get("X-Neurobench-Filename", "")])[0] or "")
        dataset = app_dataset_id(app_dir)
        requested_dataset = str((query.get("dataset_id") or [self.headers.get("X-Neurobench-Dataset-Id", "")])[0] or "")
        if requested_dataset and requested_dataset != dataset:
            self._send_json(400, {"error": "Requested dataset does not match the dataset-qualified route."})
            return
        if not filename or Path(filename).name != filename or any(part in {"", ".", ".."} for part in Path(filename).parts):
            self._send_json(400, {"error": "filename must be a simple supported file name"})
            return
        if Path(filename).suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES | SUPPORTED_LABEL_SUFFIXES | SUPPORTED_NEUREV_SUFFIXES:
            self._send_json(400, {"error": "unsupported upload suffix"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            self._send_json(400, {"error": "invalid Content-Length"})
            return
        if length <= 0 or length > MAX_IMPORT_BYTES:
            self._send_json(413, {"error": "invalid upload size"})
            return
        if Path(filename).suffix.lower() in SUPPORTED_NEUREV_SUFFIXES and length > MAX_NEUREV_JSON_BYTES:
            self._send_json(413, {"error": f"NeuRev JSON exceeds the {MAX_NEUREV_JSON_BYTES:,}-byte safety limit."})
            return
        destination = PROJECT_ROOT / "Inputs" / dataset / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            self._send_json(409, {"error": f"upload destination already exists: {destination.name}"})
            return
        if length > int(shutil.disk_usage(destination.parent).free * 0.65):
            self._send_json(413, {"error": "insufficient disk headroom for upload"})
            return
        upload_token = uuid.uuid4().hex
        partial = destination.with_name(f".{upload_token}.{destination.stem}.partial{destination.suffix}")
        published = False
        registered = False
        staged = False
        received = 0
        digest = hashlib.sha256()
        try:
            remaining = length
            with partial.open("xb") as handle:
                staged = True
                while remaining:
                    chunk = self.rfile.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("upload ended before Content-Length bytes were received")
                    handle.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    remaining -= len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            inspect_source(partial, workspace_root=PROJECT_ROOT)
            _commit_upload_without_overwrite(partial, destination)
            published = True
            record = register_import_source(
                dataset_id=dataset,
                source=destination,
                source_mode="upload",
                destination_path=relative_workspace_path(destination, workspace_root=PROJECT_ROOT),
                metadata_payload={},
                app_dir=app_dir,
            )
            registered = True
            self._remember_created_dataset(dataset, app_dir)
            promote = str((query.get("promote_primary_video") or [""])[0]).lower() in {"1", "true", "yes"}
            replace = str((query.get("replace_primary_video") or [""])[0]).lower() in {"1", "true", "yes"}
            if promote:
                try:
                    record = promote_import_primary_video(record, replace=replace)
                except FileExistsError as exc:
                    self._send_json(409, {"error": str(exc), "import": record})
                    return
            self._send_json(201, {"ok": True, "import": record})
        except Exception as exc:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass
            if published and not registered:
                destination.unlink(missing_ok=True)
            failed_record = None
            if staged and not registered:
                try:
                    failed_record = persist_failed_upload_record(
                        dataset_id=dataset,
                        app_dir=app_dir,
                        filename=filename,
                        destination_path=relative_workspace_path(destination, workspace_root=PROJECT_ROOT),
                        received_size=received,
                        sha256=digest.hexdigest(),
                        error=exc,
                    )
                    self._remember_created_dataset(dataset, app_dir)
                except Exception:
                    failed_record = None
            response: dict[str, Any] = {"error": str(exc)}
            if failed_record is not None:
                response["import"] = failed_record
            self._send_json(400, response)

    def _record_for_payload(self, app_dir: Path, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        import_value = str(payload.get("import_id") or "")
        if not import_value:
            return None
        return import_record_for(app_dataset_id(app_dir), import_value, app_dir=app_dir)

    def _handle_import_metadata_post(self, app_dir: Path, payload: dict[str, Any], tail: tuple[str, ...]) -> None:
        record = self._record_for_payload(app_dir, payload)
        if record is None:
            self._send_json(404, {"error": "import not found"})
            return
        try:
            dataset = str(record["dataset_id"])
            with dataset_lock(dataset):
                current = import_record_for(dataset, str(record["import_id"]), app_dir=app_dir)
                if current is None:
                    raise ValueError("Import disappeared before metadata update.")
                metadata = _metadata_overrides(current.get("metadata") or {}, payload)
                warnings = [warning for warning in current.get("warnings", []) if "unknown until supplied" not in str(warning)]
                if current.get("state") == "failed":
                    has_qc = isinstance(current.get("qc"), Mapping) and bool((current.get("generated_artifacts") or {}).get("qc"))
                    retry_state = "qc_ready" if (metadata.get("kind") in {"label_table", "neurev_json"} or has_qc) else "metadata_needed"
                    updated = transition_import_record(current, retry_state, metadata=metadata, warnings=warnings, error="")
                else:
                    updated = update_import_record(current, metadata=metadata, warnings=warnings)
                self._send_json(200, {"ok": True, "import": persist_import_record(updated)})
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})

    def _handle_import_promote_post(self, app_dir: Path, payload: dict[str, Any], tail: tuple[str, ...]) -> None:
        record = self._record_for_payload(app_dir, payload)
        if record is None:
            self._send_json(404, {"error": "import not found"})
            return
        try:
            promoted = promote_import_primary_video(record, replace=bool(payload.get("replace_primary_video")))
            self._send_json(200, {"ok": True, "import": promoted})
        except FileExistsError as exc:
            self._send_json(409, {"error": str(exc), "import": record})
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})

    def _handle_import_qc_post(self, app_dir: Path, payload: dict[str, Any], tail: tuple[str, ...]) -> None:
        record = self._record_for_payload(app_dir, payload)
        if record is None:
            self._send_json(404, {"error": "import not found"})
            return
        store = job_store_for_app(app_dir)
        key = f"{record['dataset_id']}:{record['import_id']}:qc"
        job, created = store.create_or_get_active("dataset_qc", {"dataset_id": record["dataset_id"], "import_id": record["import_id"]}, dedupe_key=key)
        if created:
            threading.Thread(target=execute_import_qc_job, args=(store, job["job_id"], record), daemon=True).start()
        self._send_json(202, {"ok": True, "deduplicated": not created, "job": job})

    def _handle_import_process_post(self, app_dir: Path, payload: dict[str, Any], tail: tuple[str, ...]) -> None:
        record = self._record_for_payload(app_dir, payload)
        if record is None:
            self._send_json(404, {"error": "import not found"})
            return
        store = job_store_for_app(app_dir)
        key = f"{record['dataset_id']}:{record['import_id']}:process"
        job, created = store.create_or_get_active("dataset_process", {"dataset_id": record["dataset_id"], "import_id": record["import_id"]}, dedupe_key=key)
        if created:
            threading.Thread(target=execute_import_process_job, args=(store, job["job_id"], record), daemon=True).start()
        self._send_json(202, {"ok": True, "deduplicated": not created, "job": job})

    def _handle_label_preview_post(self, app_dir: Path, payload: dict[str, Any], tail: tuple[str, ...]) -> None:
        record = self._record_for_payload(app_dir, payload)
        if record is None:
            self._send_json(404, {"error": "import not found"})
            return
        try:
            if (record.get("metadata") or {}).get("kind") != "label_table":
                raise ValueError("The selected import is not a label table.")
            source = source_path_for_import(record)
            verify_source_identity(source, record)
            metadata = inspect_source(source)["metadata"]
            rows = list(iter_label_rows(source, limit=10))
            self._send_json(200, {"import_id": record["import_id"], "columns": metadata.get("columns", []), "row_count": metadata.get("row_count", 0), "label_mapping": metadata.get("label_mapping") or infer_label_mapping(metadata.get("columns", [])), "sample_rows": rows})
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})

    def _handle_label_import_post(self, app_dir: Path, payload: dict[str, Any], tail: tuple[str, ...]) -> None:
        if payload.get("confirmed") is not True:
            self._send_json(400, {"error": "confirmed must be true after reviewing the label preview"})
            return
        record = self._record_for_payload(app_dir, payload)
        if record is None:
            self._send_json(404, {"error": "import not found"})
            return
        mapping = payload.get("label_mapping") if isinstance(payload.get("label_mapping"), Mapping) else None
        store = job_store_for_app(app_dir)
        key = f"{record['dataset_id']}:{record['import_id']}:labels"
        job, created = store.create_or_get_active("label_import", {"dataset_id": record["dataset_id"], "import_id": record["import_id"], "label_mapping": dict(mapping or {})}, dedupe_key=key)
        if created:
            threading.Thread(target=execute_label_import_job, args=(store, job["job_id"], record, mapping), daemon=True).start()
        self._send_json(202, {"ok": True, "deduplicated": not created, "job": job})

    def _handle_neurev_preview_post(self, app_dir: Path, payload: dict[str, Any], tail: tuple[str, ...]) -> None:
        record = self._record_for_payload(app_dir, payload)
        if record is None:
            self._send_json(404, {"error": "import not found"})
            return
        try:
            if (record.get("metadata") or {}).get("kind") != "neurev_json":
                raise ValueError("The selected import is not NeuRev JSON.")
            source = source_path_for_import(record)
            verify_source_identity(source, record)
            summary = inspect_neurev_json(source)
            _require_matching_neurev_dataset(record, summary)
            self._send_json(
                200,
                {
                    "import_id": record["import_id"],
                    "payload_kind": summary["payload_kind"],
                    "payload_schema_version": summary.get("payload_schema_version"),
                    "declared_dataset_id": summary.get("declared_dataset_id"),
                    "counts": dict(summary.get("counts") or {}),
                    "source": {
                        "original_name": record.get("original_name"),
                        "checksum": dict(record.get("checksum") or {}),
                    },
                    "confirmation_required": True,
                    "publication": "external_neurev_lossless_copy",
                },
            )
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})

    def _handle_neurev_import_post(self, app_dir: Path, payload: dict[str, Any], tail: tuple[str, ...]) -> None:
        if payload.get("confirmed") is not True:
            self._send_json(400, {"error": "confirmed must be true after reviewing the NeuRev JSON preview"})
            return
        record = self._record_for_payload(app_dir, payload)
        if record is None:
            self._send_json(404, {"error": "import not found"})
            return
        if (record.get("metadata") or {}).get("kind") != "neurev_json":
            self._send_json(400, {"error": "The selected import is not NeuRev JSON."})
            return
        store = job_store_for_app(app_dir)
        key = f"{record['dataset_id']}:{record['import_id']}:neurev-json"
        job, created = store.create_or_get_active(
            "neurev_json_import",
            {"dataset_id": record["dataset_id"], "import_id": record["import_id"]},
            dedupe_key=key,
        )
        if created:
            threading.Thread(
                target=execute_neurev_json_import_job,
                args=(store, job["job_id"], record),
                daemon=True,
            ).start()
        self._send_json(202, {"ok": True, "deduplicated": not created, "job": job})

    def _handle_generation_post(self, app_dir: Path, payload: dict[str, Any], tail: tuple[str, ...]) -> None:
        if tail == ("jobs", "generate-preview"):
            payload["preview"] = True
            payload.setdefault("stages", "high-pass,event-denoise,candidates,temporal-scoring,review-data,workbench")
        backend = str(payload.get("backend") or "auto")
        if backend not in ALLOWED_BACKENDS:
            self._send_json(400, {"error": f"unsupported backend: {backend}"})
            return
        run_id = str(payload.get("run_id") or "current_review_pipeline")
        if not payload.get("force"):
            active = JOBS.active_for(app_dir, run_id)
            if active is not None:
                self._send_json(409, {"error": "generation already running for this run", "job": active.as_dict()})
                return
        job = GenerationJob(app_dir=app_dir, payload=payload)
        JOBS.add(job)
        thread = threading.Thread(target=execute_generation_job, args=(job,), daemon=True)
        thread.start()
        self._send_json(202, job.as_dict())


def source_path_for_import(record: Mapping[str, Any]) -> Path:
    raw = str(record.get("destination_path") or record.get("source_path") or "")
    if not raw:
        raise ValueError("Import does not declare a source path.")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return resolve_allowed_local_path(candidate, workspace_root=PROJECT_ROOT)


def _publish_file_exclusive(staged: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(staged, destination)
    except FileExistsError:
        raise FileExistsError(f"Artifact already exists: {destination}") from None
    except OSError:
        created = False
        try:
            with staged.open("rb") as source, destination.open("xb") as target:
                created = True
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
        except FileExistsError:
            raise FileExistsError(f"Artifact already exists: {destination}") from None
        except Exception:
            if created:
                destination.unlink(missing_ok=True)
            raise
    staged.unlink(missing_ok=True)


def _publish_directory_exclusive(staged: Path, destination: Path) -> None:
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"Artifact directory already exists: {destination}") from None
    try:
        for source in sorted(staged.rglob("*")):
            relative = source.relative_to(staged)
            target = destination / relative
            if source.is_dir():
                target.mkdir(exist_ok=False)
            elif source.is_file():
                _publish_file_exclusive(source, target)
        shutil.rmtree(staged)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _fail_import_record(record: Mapping[str, Any], error: Exception | str) -> None:
    dataset_id = str(record.get("dataset_id") or "")
    app_dir = import_app_for_record(record)
    current = import_record_for(dataset_id, str(record.get("import_id") or ""), app_dir=app_dir)
    if current is None or current.get("state") in {"complete", "ready"}:
        return
    try:
        failed = transition_import_record(current, "failed", error=str(error))
    except ValueError:
        return
    persist_import_record(failed)


def _require_matching_neurev_dataset(record: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    declared = str(summary.get("declared_dataset_id") or "").strip()
    if declared and normalize_dataset_id(declared) != str(record.get("dataset_id") or ""):
        raise ValueError(
            f"NeuRev JSON declares dataset_id '{declared}', which does not match "
            f"the target dataset '{record.get('dataset_id')}'."
        )


def _stage_lossless_neurev_json(source: Path, destination: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy bounded source bytes exactly while rechecking the recorded identity."""

    expected = record.get("checksum") if isinstance(record.get("checksum"), Mapping) else {}
    expected_size = int(expected.get("size_bytes") or -1)
    expected_sha = str(expected.get("sha256") or "")
    observed_size = source.stat().st_size
    if observed_size > MAX_NEUREV_JSON_BYTES:
        raise ValueError(f"NeuRev JSON exceeds the {MAX_NEUREV_JSON_BYTES:,}-byte safety limit.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    copied = 0
    try:
        with source.open("rb") as source_handle, destination.open("xb") as target_handle:
            while True:
                chunk = source_handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > MAX_NEUREV_JSON_BYTES:
                    raise ValueError(f"NeuRev JSON exceeds the {MAX_NEUREV_JSON_BYTES:,}-byte safety limit.")
                digest.update(chunk)
                target_handle.write(chunk)
            target_handle.flush()
            os.fsync(target_handle.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    observed_sha = digest.hexdigest()
    if copied != expected_size or observed_sha != expected_sha:
        destination.unlink(missing_ok=True)
        raise ValueError("NeuRev JSON source changed while its lossless external copy was being published.")
    return {"sha256": observed_sha, "size_bytes": copied}


def execute_neurev_json_import_job(store: JobStore, job_id: str, record: Mapping[str, Any]) -> None:
    """Publish confirmed NeuRev JSON separately without merging native app state."""

    dataset_id = str(record.get("dataset_id") or "")
    app_dir = import_app_for_record(record)
    stage_root = app_dir / ".neurobench" / "staging" / job_id
    artifact_path = app_dir / "external_neurev" / f"{record.get('import_id')}.json"
    published = False
    try:
        with dataset_lock(dataset_id):
            current = import_record_for(dataset_id, str(record.get("import_id") or ""), app_dir=app_dir)
            if current is None:
                raise ValueError("Import record no longer exists.")
            if (current.get("metadata") or {}).get("kind") != "neurev_json":
                raise ValueError("The selected import is not NeuRev JSON.")
            if current.get("state") == "complete" and artifact_path.is_file():
                verify_source_identity(artifact_path, current)
                store.update(
                    job_id,
                    status="completed",
                    stage="already_complete",
                    progress=1.0,
                    outputs={"external_neurev_json": str(artifact_path)},
                )
                return
            if current.get("state") != "qc_ready":
                raise ValueError(f"NeuRev JSON import requires qc_ready state; found {current.get('state')}.")
            if artifact_path.exists():
                raise FileExistsError("Protected external NeuRev JSON artifact already exists for this import.")
            if stage_root.exists():
                raise FileExistsError(f"Job staging path already exists: {stage_root}")
            source = source_path_for_import(current)
            verify_source_identity(source, current)
            summary = inspect_neurev_json(source)
            _require_matching_neurev_dataset(current, summary)
            recorded_kind = str((current.get("metadata") or {}).get("payload_kind") or "")
            if recorded_kind and recorded_kind != summary["payload_kind"]:
                raise ValueError("NeuRev JSON payload kind changed after registration.")

            store.update(job_id, status="running", stage="validating", progress=0.1)
            current = transition_import_record(current, "processing", error="")
            current = persist_import_record(current)
            store.update(job_id, stage="copying_losslessly", progress=0.35)
            copied_identity = _stage_lossless_neurev_json(source, stage_root / "payload.json", current)
            store.update(job_id, stage="publishing", progress=0.85)
            _publish_file_exclusive(stage_root / "payload.json", artifact_path)
            published = True
            artifacts = dict(current.get("generated_artifacts") or {})
            artifacts["external_neurev_json"] = rel_to_app(app_dir, artifact_path)
            completed = transition_import_record(
                current,
                "complete",
                generated_artifacts=artifacts,
                neurev_json_summary=dict(summary),
                error="",
            )
            persist_import_record(completed)
            store.update(
                job_id,
                status="completed",
                stage="complete",
                progress=1.0,
                outputs={
                    "external_neurev_json": str(artifact_path),
                    "payload_kind": summary["payload_kind"],
                    "counts": dict(summary.get("counts") or {}),
                    "checksum": copied_identity,
                },
            )
    except Exception as exc:
        try:
            with dataset_lock(dataset_id):
                if published:
                    artifact_path.unlink(missing_ok=True)
                _fail_import_record(record, exc)
        except Exception:
            pass
        store.update(
            job_id,
            status="failed",
            stage="collision" if isinstance(exc, FileExistsError) else "failed",
            error=str(exc),
        )
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def execute_import_qc_job(store: JobStore, job_id: str, record: Mapping[str, Any]) -> None:
    dataset_id = str(record.get("dataset_id") or "")
    app_dir = import_app_for_record(record)
    try:
        with dataset_lock(dataset_id):
            current = import_record_for(dataset_id, str(record.get("import_id") or ""), app_dir=app_dir)
            if current is None:
                raise ValueError("Import record no longer exists.")
            if (current.get("metadata") or {}).get("kind") != "video":
                raise ValueError("QC is only available for video imports.")
            qc_path = app_dir / "qc" / f"{current['import_id']}.json"
            if current.get("state") == "qc_ready" and qc_path.is_file() and isinstance(current.get("qc"), Mapping):
                store.update(job_id, status="completed", stage="already_complete", progress=1.0, outputs={"qc": str(qc_path)})
                return
            if current.get("state") != "metadata_needed":
                raise ValueError(f"QC requires metadata_needed state; found {current.get('state')}.")
            source = source_path_for_import(current)
            verify_source_identity(source, current)
            store.update(job_id, status="running", stage="sampling", progress=0.05)
            qc = sampled_video_qc(source, dataset_id=dataset_id)
            stage_path = app_dir / ".neurobench" / "staging" / job_id / "qc.json"
            if stage_path.parent.exists():
                raise FileExistsError(f"Job staging path already exists: {stage_path.parent}")
            atomic_write_json(stage_path, qc)
            try:
                _publish_file_exclusive(stage_path, qc_path)
            finally:
                shutil.rmtree(stage_path.parents[0], ignore_errors=True)
            artifacts = dict(current.get("generated_artifacts") or {})
            artifacts["qc"] = rel_to_app(app_dir, qc_path)
            updated = transition_import_record(current, "qc_ready", qc=qc, generated_artifacts=artifacts, error="")
            persist_import_record(updated)
            store.update(job_id, status="completed", stage="complete", progress=1.0, outputs={"qc": str(qc_path)})
    except Exception as exc:
        try:
            with dataset_lock(dataset_id):
                _fail_import_record(record, exc)
        except Exception:
            pass
        store.update(job_id, status="failed", stage="failed", error=str(exc))


def execute_import_process_job(store: JobStore, job_id: str, record: Mapping[str, Any]) -> None:
    dataset_id = str(record.get("dataset_id") or "")
    app_dir = import_app_for_record(record)
    stage_root = app_dir / ".neurobench" / "staging" / job_id
    stage_app = stage_root / "app"
    published: list[Path] = []
    try:
        from neurobench.annotations import migrate_annotations_v3
        from neurobench.workbench.builder import render_workbench_assets

        with dataset_lock(dataset_id):
            current = import_record_for(dataset_id, str(record.get("import_id") or ""), app_dir=app_dir)
            if current is None:
                raise ValueError("Import record no longer exists.")
            metadata = dict(current.get("metadata") or {})
            if metadata.get("kind") != "video":
                raise ValueError("Only video imports can be opened in the normal annotator.")
            if current.get("state") == "ready":
                store.update(job_id, status="completed", stage="already_complete", progress=1.0, outputs=dict(current.get("generated_artifacts") or {}))
                return
            if current.get("state") != "qc_ready" or not isinstance(current.get("qc"), Mapping):
                raise ValueError("Processing requires a completed QC record in qc_ready state.")
            if not current.get("is_primary_video") or current.get("source_role") != "primary_video":
                raise ValueError("Processing requires explicit promotion of this import as the primary video.")
            source = source_path_for_import(current)
            verify_source_identity(source, current)
            manifest_path = app_dir / "dataset_manifest.generated.json"
            if not manifest_path.is_file():
                raise ValueError("Canonical dataset manifest is missing; promote the primary video first.")
            manifest = load_json(manifest_path)
            validate_dict(manifest, "dataset")
            if str((manifest.get("source") or {}).get("import_id") or "") != str(current["import_id"]):
                raise ValueError("Canonical dataset manifest points to a different primary video.")
            targets = {
                "frames": app_dir / "frames",
                "review_data": app_dir / "review_data.json",
                "annotations": app_dir / "annotations.json",
                "architecture_runs": app_dir / "architecture_runs.json",
                "index": app_dir / "index.html",
                "css": app_dir / "workbench.css",
                "js": app_dir / "workbench.js",
            }
            collisions = [str(path) for path in targets.values() if path.exists()]
            if collisions:
                raise FileExistsError("Protected workbench artifacts already exist: " + ", ".join(collisions))
            if stage_root.exists():
                raise FileExistsError(f"Job staging path already exists: {stage_root}")
            store.update(job_id, status="running", stage="preflight", progress=0.02)
            current = transition_import_record(current, "processing", error="")
            current = persist_import_record(current)

            last_progress = [-1.0]

            def progress(value: float) -> None:
                if value - last_progress[0] >= 0.02 or value >= 1.0:
                    last_progress[0] = value
                    store.update(job_id, stage="rendering_frames", progress=0.05 + 0.7 * value)

            frame_count = _render_import_frames(source, stage_app / "frames", progress=progress)
            review_data: dict[str, Any] = {
                "schema_version": 1,
                "dataset": {
                    "dataset_id": dataset_id,
                    "name": current.get("original_name") or dataset_id,
                },
                "video": {
                    "name": current.get("original_name") or source.name,
                    "width": int(metadata.get("width") or 1),
                    "height": int(metadata.get("height") or 1),
                    "frames": frame_count,
                    "framePattern": "frames/frame_%06d.png",
                },
                "parameters": {"source_mode": current.get("source_mode"), "import_id": current.get("import_id")},
                "qc": dict(current["qc"]),
                "rois": [],
                "discovery": {"suggestions": []},
            }
            if metadata.get("modality") is not None:
                review_data["dataset"]["modality"] = str(metadata["modality"])
            validate_dict(review_data, "review_data")
            architecture_runs = {"schema_version": 1, "dataset_id": dataset_id, "runs": []}
            validate_dict(architecture_runs, "architecture_runs")
            annotations = migrate_annotations_v3(None)
            validate_dict(annotations, "annotations")
            atomic_write_json(stage_app / "review_data.json", review_data)
            atomic_write_json(stage_app / "architecture_runs.json", architecture_runs)
            atomic_write_json(stage_app / "annotations.json", annotations)
            store.update(job_id, stage="rendering_current_assets", progress=0.82)
            assets = render_workbench_assets(
                review_data_path=stage_app / "review_data.json",
                dataset_id=dataset_id,
                dataset_manifest=manifest,
                architecture_runs_path=stage_app / "architecture_runs.json",
                app_dir=stage_app,
            )
            for name, body in assets.items():
                atomic_write_bytes(stage_app / name, body)

            store.update(job_id, stage="publishing", progress=0.9)
            staged_targets = [
                (stage_app / "frames", targets["frames"], True),
                (stage_app / "review_data.json", targets["review_data"], False),
                (stage_app / "annotations.json", targets["annotations"], False),
                (stage_app / "architecture_runs.json", targets["architecture_runs"], False),
                (stage_app / "index.html", targets["index"], False),
                (stage_app / "workbench.css", targets["css"], False),
                (stage_app / "workbench.js", targets["js"], False),
            ]
            for staged, destination, is_directory in staged_targets:
                if is_directory:
                    _publish_directory_exclusive(staged, destination)
                else:
                    _publish_file_exclusive(staged, destination)
                published.append(destination)
            artifacts = dict(current.get("generated_artifacts") or {})
            artifacts.update({key: rel_to_app(app_dir, value) for key, value in targets.items()})
            ready = transition_import_record(current, "ready", generated_artifacts=artifacts, error="")
            persist_import_record(ready)
            store.update(job_id, status="completed", stage="complete", progress=1.0, outputs={key: str(value) for key, value in targets.items()})
    except Exception as exc:
        try:
            with dataset_lock(dataset_id):
                for path in reversed(published):
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)
                _fail_import_record(record, exc)
        except Exception:
            pass
        store.update(job_id, status="failed", stage="collision" if isinstance(exc, FileExistsError) else "failed", error=str(exc))
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def execute_label_import_job(store: JobStore, job_id: str, record: Mapping[str, Any], mapping: Mapping[str, Any] | None = None) -> None:
    dataset_id = str(record.get("dataset_id") or "")
    app_dir = import_app_for_record(record)
    stage_root = app_dir / ".neurobench" / "staging" / job_id
    published: list[Path] = []
    try:
        with dataset_lock(dataset_id):
            current = import_record_for(dataset_id, str(record.get("import_id") or ""), app_dir=app_dir)
            if current is None:
                raise ValueError("Import record no longer exists.")
            if (current.get("metadata") or {}).get("kind") != "label_table":
                raise ValueError("The selected import is not a label table.")
            artifact_path = app_dir / "external_labels" / f"{current['import_id']}.json"
            overlay_path = app_dir / "external_labels" / f"{current['import_id']}.overlay.svg"
            if current.get("state") == "complete" and artifact_path.is_file() and overlay_path.is_file():
                store.update(job_id, status="completed", stage="already_complete", progress=1.0, outputs={"external_labels": str(artifact_path), "label_overlay": str(overlay_path)})
                return
            if current.get("state") != "qc_ready":
                raise ValueError(f"Label reconciliation requires qc_ready state; found {current.get('state')}.")
            if artifact_path.exists() or overlay_path.exists():
                raise FileExistsError("Protected external-label artifacts already exist for this import.")
            source = source_path_for_import(current)
            verify_source_identity(source, current)
            review_path = app_dir / "review_data.json"
            if not review_path.is_file():
                raise ValueError("review_data.json is required to reconcile labels against native ROI identities.")
            review = load_json(review_path)
            if stage_root.exists():
                raise FileExistsError(f"Job staging path already exists: {stage_root}")
            store.update(job_id, status="running", stage="reading_labels", progress=0.05)
            current = transition_import_record(current, "processing", error="")
            current = persist_import_record(current)

            artifact_budget = min(MAX_LABEL_ARTIFACT_BYTES, max(1_000_000, int(_available_memory_bytes() * 0.15)))

            def progress(row_count: int, expected_rows: int) -> None:
                store.update(
                    job_id,
                    stage="reading_labels",
                    progress=min(0.86, 0.05 + row_count / expected_rows * 0.8),
                )

            result = reconcile_label_table(
                source=source,
                review=review,
                import_record=current,
                mapping=mapping,
                artifact_budget_bytes=artifact_budget,
                max_overlay_points=MAX_LABEL_OVERLAY_POINTS,
                progress=progress,
            )
            summary = result.summary
            atomic_write_bytes(stage_root / "labels.json", result.artifact_bytes)
            atomic_write_bytes(stage_root / "overlay.svg", result.overlay_svg)
            store.update(job_id, stage="publishing", progress=0.92)
            _publish_file_exclusive(stage_root / "labels.json", artifact_path)
            published.append(artifact_path)
            _publish_file_exclusive(stage_root / "overlay.svg", overlay_path)
            published.append(overlay_path)
            artifacts = dict(current.get("generated_artifacts") or {})
            artifacts["external_labels"] = rel_to_app(app_dir, artifact_path)
            artifacts["label_overlay"] = rel_to_app(app_dir, overlay_path)
            completed = transition_import_record(current, "complete", generated_artifacts=artifacts, label_reconciliation=summary, error="")
            persist_import_record(completed)
            store.update(job_id, status="completed", stage="complete", progress=1.0, outputs={"external_labels": str(artifact_path), "label_overlay": str(overlay_path), "summary": summary})
    except Exception as exc:
        try:
            with dataset_lock(dataset_id):
                for path in reversed(published):
                    path.unlink(missing_ok=True)
                _fail_import_record(record, exc)
        except Exception:
            pass
        store.update(job_id, status="failed", stage="collision" if isinstance(exc, FileExistsError) else "failed", error=str(exc))
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def configure_workbench_handler(
    *,
    app_dir: Path = DEFAULT_APP_DIR,
    root_dir: Path | None = None,
    asset_mode: str = "current",
) -> tuple[type[WorkbenchHandler], Path]:
    """Validate serving roots and return an isolated configured handler class."""

    if asset_mode not in {"current", "installed"}:
        raise ValueError("asset_mode must be current or installed")

    if root_dir:
        root_dir = root_dir.resolve()
        if not (root_dir / "index.html").exists():
            raise SystemExit(f"index.html not found in {root_dir}")
        configured_app = root_dir
        configured_root: Path | None = root_dir
        served = root_dir
    else:
        app_dir = app_dir.resolve()
        if not (app_dir / "index.html").exists() and not (asset_mode == "current" and (app_dir / "review_data.json").is_file()):
            raise SystemExit(f"Neither index.html nor renderable review_data.json found in {app_dir}")
        configured_app = app_dir
        configured_root = None
        served = app_dir
    registry_conflicts: dict[str, tuple[str, ...]] = {}
    registry = build_dataset_app_registry(
        configured_app=configured_app if configured_root is None else None,
        root_dir=configured_root,
        conflicts_out=registry_conflicts,
    )
    handler = type(
        f"ConfiguredWorkbenchHandler_{uuid.uuid4().hex}",
        (WorkbenchHandler,),
        {
            "app_dir": configured_app,
            "root_dir": configured_root,
            "asset_mode": asset_mode,
            "dataset_apps": registry,
            "dataset_registry_conflicts": MappingProxyType(registry_conflicts),
            "created_dataset_apps": {},
            "dataset_registry_lock": threading.RLock(),
        },
    )
    return handler, served


def create_workbench_server(
    *,
    app_dir: Path = DEFAULT_APP_DIR,
    root_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    asset_mode: str = "current",
) -> tuple[ThreadingHTTPServer, Path]:
    handler, served = configure_workbench_handler(app_dir=app_dir, root_dir=root_dir, asset_mode=asset_mode)
    return ThreadingHTTPServer((host, port), handler), served


def serve_workbench(
    *,
    app_dir: Path = DEFAULT_APP_DIR,
    root_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    asset_mode: str = "current",
) -> None:
    server, served = create_workbench_server(app_dir=app_dir, root_dir=root_dir, host=host, port=port, asset_mode=asset_mode)
    print(f"Serving {served}")
    print(f"Open http://{host}:{port}/")
    server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the neuron workbench with local autosave.")
    parser.add_argument("--app-dir", type=Path, default=DEFAULT_APP_DIR)
    parser.add_argument("--root-dir", type=Path, default=None, help="Serve a multi-dataset Outputs/NeuronReview root.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--asset-mode", choices=("current", "installed"), default="current")
    args = parser.parse_args(argv)
    serve_workbench(app_dir=args.app_dir, root_dir=args.root_dir, host=args.host, port=args.port, asset_mode=args.asset_mode)
