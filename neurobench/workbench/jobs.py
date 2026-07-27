"""Small restart-safe job store for local workbench workflows.

The workbench is intentionally local-first.  A job record therefore lives
beside the app, is written atomically, and contains enough state to explain a
restart without pretending that an interrupted process completed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import errno
import fcntl
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterable, Mapping
import uuid


JOB_SCHEMA_VERSION = 1
JOB_STATES = ("queued", "running", "failed", "stopped", "completed")
_ROOT_LOCKS_GUARD = threading.Lock()
_ROOT_LOCKS: dict[str, threading.RLock] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _root_lock(root: Path) -> threading.RLock:
    key = str(root)
    with _ROOT_LOCKS_GUARD:
        return _ROOT_LOCKS.setdefault(key, threading.RLock())


def _safe_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))
    return cleaned.strip("._")[:120] or "job"


class JobStore:
    """Persist bounded job records and progress files with atomic replacement."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = _root_lock(self.root)

    def _record_path(self, job_id: str) -> Path:
        return self.root / f"{_safe_id(job_id)}.json"

    @contextmanager
    def _exclusive(self):
        """Serialize mutations across threads and local server processes."""

        with self._lock:
            lock_path = self.root / ".jobs.lock"
            with lock_path.open("a+b") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read(self, path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return dict(payload) if isinstance(payload, Mapping) else None

    def _write(self, path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return dict(payload)

    def _create_unlocked(
        self,
        kind: str,
        payload: Mapping[str, Any] | None,
        *,
        job_id: str | None = None,
        dedupe_key: str | None = None,
    ) -> dict[str, Any]:
        identifier = _safe_id(job_id or f"job_{uuid.uuid4().hex[:16]}")
        if self._record_path(identifier).exists():
            raise FileExistsError(f"job already exists: {identifier}")
        now = _now()
        record = {
            "schema_version": JOB_SCHEMA_VERSION,
            "kind": str(kind),
            "job_id": identifier,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "payload": dict(payload or {}),
            "outputs": {},
            "error": "",
            "log_tail": [],
            "owner_pid": os.getpid(),
            "created_at": now,
            "updated_at": now,
        }
        if dedupe_key is not None:
            record["dedupe_key"] = str(dedupe_key)
        return self._write(self._record_path(identifier), record)

    def create(self, kind: str, payload: Mapping[str, Any] | None = None, *, job_id: str | None = None) -> dict[str, Any]:
        """Create a queued job under a cross-process mutation lock."""

        with self._exclusive():
            return self._create_unlocked(kind, payload, job_id=job_id)

    def create_or_get_active(
        self,
        kind: str,
        payload: Mapping[str, Any] | None,
        *,
        dedupe_key: str,
    ) -> tuple[dict[str, Any], bool]:
        """Return an active equivalent job or create one under a shared root lock."""

        with self._exclusive():
            for path in sorted(self.root.glob("*.json")):
                record = self._read(path)
                if record is None:
                    continue
                if record.get("dedupe_key") == dedupe_key and record.get("status") in {"queued", "running"}:
                    return record, False
            return self._create_unlocked(kind, payload, dedupe_key=str(dedupe_key)), True

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._read(self._record_path(job_id))

    def list(self, *, kinds: Iterable[str] | None = None) -> list[dict[str, Any]]:
        wanted = {str(kind) for kind in kinds} if kinds is not None else None
        with self._lock:
            records = []
            for path in sorted(self.root.glob("*.json")):
                record = self._read(path)
                if record is None or (wanted is not None and record.get("kind") not in wanted):
                    continue
                records.append(record)
            return sorted(records, key=lambda item: (str(item.get("created_at") or ""), str(item.get("job_id") or "")))

    def update(self, job_id: str, **updates: Any) -> dict[str, Any]:
        with self._exclusive():
            current = self._read(self._record_path(job_id))
            if current is None:
                raise KeyError(f"job not found: {job_id}")
            if "status" in updates and updates["status"] not in JOB_STATES:
                raise ValueError(f"unknown job status: {updates['status']}")
            if "progress" in updates:
                progress = float(updates["progress"])
                if not 0 <= progress <= 1:
                    raise ValueError("job progress must be between 0 and 1")
                updates["progress"] = progress
            current.update(updates)
            current["updated_at"] = _now()
            return self._write(self._record_path(job_id), current)

    def append_log(self, job_id: str, line: str, *, limit: int = 120) -> dict[str, Any]:
        with self._exclusive():
            current = self._read(self._record_path(job_id))
            if current is None:
                raise KeyError(f"job not found: {job_id}")
            lines = list(current.get("log_tail") or [])
            lines.append(str(line).rstrip())
            current["log_tail"] = lines[-max(1, int(limit)) :]
            current["updated_at"] = _now()
            return self._write(self._record_path(job_id), current)

    def recover_incomplete(self) -> list[dict[str, Any]]:
        """Mark jobs that were active at process exit as stopped and explain why."""

        recovered: list[dict[str, Any]] = []
        with self._exclusive():
            for path in sorted(self.root.glob("*.json")):
                record = self._read(path)
                if record is None:
                    continue
                if record.get("status") not in {"queued", "running"}:
                    continue
                owner_pid = int(record.get("owner_pid") or 0)
                if owner_pid > 0 and _pid_alive(owner_pid):
                    continue
                recovered.append(
                    self._write(
                        self._record_path(str(record["job_id"])),
                        dict(
                            record,
                            status="stopped",
                            stage="recovered_after_restart",
                            error="Job was active when the local workbench restarted.",
                            updated_at=_now(),
                        ),
                    )
                )
        return recovered


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True
