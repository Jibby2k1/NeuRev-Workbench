"""Full follow-up audit verification with one-pass, stat-guarded SHA reuse.

Every artifact and source checked by two_stencil_audit.verify_completed_audit
is verified. Only repeated reads of the same resolved file are deduplicated;
the cache is never persisted or trusted across validation passes. No media is
decoded again: the verified per-cell validation records retain that evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Any, Mapping


DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "Outputs/GammaLSFollowup/followup_20260914_r1"
METADATA = ("summary.json", "status.json", "validation.json", "inventory.json",
            "artifact_index.json", "source_manifest.json", "run_contract.json", "llm_context.json")
CANDIDATE_FIELDS = ("proposal_id", "source_frame_ui", "x_px", "y_px", "score",
                    "threshold_z", "candidate_rank_within_frame")


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _signature(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class FileVerifier:
    """Hash each resolved path once and fail on any observed subsequent change."""

    def __init__(self) -> None:
        self._files: dict[Path, dict[str, Any]] = {}
        self._aliases: dict[Path, Path] = {}
        self.bytes_hashed = 0
        self.cache_hits = 0

    def _resolve(self, path: str | Path) -> Path:
        alias = Path(path).absolute()
        resolved = alias.resolve(strict=True)
        if alias in self._aliases:
            require(self._aliases[alias] == resolved, f"Source/artifact link target changed: {alias}")
        self._aliases[alias] = resolved
        return resolved

    @staticmethod
    def _stat(path: Path) -> os.stat_result:
        value = path.stat()
        require(stat.S_ISREG(value.st_mode), f"Source/artifact is not a regular file: {path}")
        return value

    def _stable(self, path: Path) -> None:
        require(_signature(self._stat(path)) == self._files[path]["signature"],
                f"Source/artifact changed during validation: {path}")

    def binding(self, path: str | Path) -> dict[str, Any]:
        resolved = self._resolve(path)
        if resolved in self._files:
            self._stable(resolved)
            self.cache_hits += 1
            return dict(self._files[resolved]["binding"])
        before = self._stat(resolved)
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            require(_signature(os.fstat(stream.fileno())) == _signature(before),
                    f"Source/artifact replaced before hashing: {resolved}")
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
            require(_signature(os.fstat(stream.fileno())) == _signature(before),
                    f"Source/artifact changed while hashing: {resolved}")
        require(_signature(self._stat(resolved)) == _signature(before),
                f"Source/artifact replaced while hashing: {resolved}")
        require(self._resolve(path) == resolved, f"Source/artifact link changed while hashing: {path}")
        record = dict(path=str(resolved), sha256=digest.hexdigest(), size_bytes=before.st_size)
        self._files[resolved] = dict(binding=record, signature=_signature(before))
        self.bytes_hashed += before.st_size
        return dict(record)

    def verify(self, record: Mapping[str, Any], *, root: Path | None = None) -> dict[str, Any]:
        path = Path(record["path"]) if root is None else root / str(record["path"])
        expected = record["sha256"]
        require(isinstance(expected, str) and len(expected) == 64
                and all(c in "0123456789abcdef" for c in expected), f"Invalid SHA-256 declaration: {path}")
        size = record.get("size_bytes")
        require(size is None or (isinstance(size, int) and not isinstance(size, bool) and size >= 0),
                f"Invalid file-size declaration: {path}")
        resolved = self._resolve(path)
        if resolved in self._files:
            known = self._files[resolved]["binding"]
            require(expected == known["sha256"] and (size is None or size == known["size_bytes"]),
                    f"Conflicting frozen source/artifact binding: {resolved}")
        actual = self.binding(path)
        require(actual["sha256"] == expected and (size is None or actual["size_bytes"] == size),
                f"Frozen source/artifact SHA-256 or size mismatch: {path}")
        return actual

    def read_json(self, path: str | Path) -> Any:
        record = self.binding(path)
        resolved = Path(record["path"])
        self._stable(resolved)
        value = json.loads(resolved.read_text())
        self._stable(resolved)
        require(self._resolve(path) == resolved, f"JSON link target changed while reading: {path}")
        return value

    def assert_unchanged(self) -> None:
        for alias, resolved in self._aliases.items():
            require(alias.resolve(strict=True) == resolved, f"Source/artifact link target changed: {alias}")
        for path in self._files:
            self._stable(path)

    @property
    def unique_file_count(self) -> int:
        return len(self._files)


def _workers(root: Path, verifier: FileVerifier, state_count: int) -> list[dict[str, Any]]:
    paths = sorted(root.glob("media_worker_*.json"))
    require(bool(paths), "No completed media-worker receipts")
    records = [verifier.read_json(path) for path in paths]
    count = records[0].get("workers")
    require(type(count) is int and 1 <= count <= 3, "Invalid media-worker count")
    require(len(records) == count, "Not all media workers have receipts")
    expected = set(range(count))
    require({r.get("worker") for r in records} == expected, "Missing or duplicate media-worker IDs")
    result = []
    for path, row in zip(paths, records):
        worker = row.get("worker")
        require(type(worker) is int and path.name == f"media_worker_{worker}.json", "Worker receipt path/ID mismatch")
        require(row.get("status") == "PASS" and row.get("workers") == count,
                f"Media worker {worker} is not complete")
        require(row.get("cells") == len(range(worker, state_count, count)),
                f"Media worker {worker} did not complete its full partition")
        result.append(dict(worker=worker, workers=count, cells=row["cells"],
                           receipt=verifier.binding(path), log=verifier.binding(root / f"media_worker_{worker}.log")))
    return result


def verify_audit_files(audit: Path, verifier: FileVerifier) -> dict[str, Any]:
    """The exact three file lists from the original completed-audit verifier."""
    index = verifier.read_json(audit / "artifact_index.json")
    require(isinstance(index.get("artifacts"), list) and bool(index["artifacts"]), "Empty audit artifact index")
    for row in index["artifacts"]:
        verifier.verify(row, root=audit)
    sources = verifier.read_json(audit / "source_manifest.json")
    require(isinstance(sources.get("sources"), list) and bool(sources["sources"]), "Empty audit source manifest")
    for row in sources["sources"]:
        verifier.verify(row)
    for row in sources.get("numeric_state_sources", []):
        verifier.verify(row)
    indexed = {(audit / str(r["path"])).resolve() for r in index["artifacts"]}
    require(all((audit / name).resolve() in indexed for name in METADATA if name != "artifact_index.json"),
            "Required audit metadata missing from sealed artifact index")
    return sources


def _cell_seal(cell: Path, verifier: FileVerifier) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:
    seal = verifier.read_json(cell / "sealed.json")
    require(seal.get("status") == "SEALED_BEFORE_ACTIVITY_TRUTH_JOIN", f"Cell not sealed: {cell}")
    for key in ("prefix", "audit_candidates", "threshold_plan", "calibration"):
        verifier.verify(seal[key])
    if "setup_prefix" in seal:
        verifier.verify(seal["setup_prefix"])
    for record in seal.get("dataset_bindings", []):
        verifier.verify(record)
    evaluated = verifier.read_json(cell / "evaluated.json")
    require(evaluated.get("status") == "PASS", f"Cell not evaluated: {cell}")
    require(Path(evaluated["seal"]["path"]).resolve() == (cell / "sealed.json").resolve(), "Wrong evaluation seal path")
    verifier.verify(evaluated["seal"])
    for record in evaluated["outputs"]:
        verifier.verify(record)
    op = verifier.read_json(cell / "calibration.json")
    require(op.get("threshold_id") == "q1" and op.get("threshold_frozen_from_calibration_only") is True,
            f"Audit operating point must be the setup-frozen q1: {cell}")
    candidates = verifier.read_json(cell / "audit_candidates.json")
    require(isinstance(candidates, list), "Candidate table must contain rows")
    return seal, op, candidates


def _validate(root: Path, verifier: FileVerifier) -> dict[str, Any]:
    # Worker completion is checked before reading or hashing any large source.
    worker_records = _workers(root, verifier, 242)
    preflight = verifier.read_json(root / "preflight.json")
    verifier.verify(dict(path=str(root / "protocol.json"), sha256=preflight["protocol_sha256"]))
    protocol = verifier.read_json(root / "protocol.json")
    require(preflight.get("status") == "PASS" and protocol.get("expected_cells") == 242,
            "Follow-up preflight or matrix size differs")
    for record in protocol["code_bindings"]:
        verifier.verify(record)
    for key in ("baseline_completion", "baseline_protocol", "real_annotation_acceptance"):
        verifier.verify(protocol[key])
    if "baseline_dataset_completion" in protocol:
        verifier.verify(protocol["baseline_dataset_completion"])
    baseline = Path(protocol["baseline_protocol"]["path"]).resolve().parent
    require(verifier.read_json(protocol["baseline_completion"]["path"]).get("status") == "PASS",
            "Baseline completion is not PASS")
    require(verifier.read_json(protocol["real_annotation_acceptance"]["path"]).get("accepted") is False,
            "Real truth acceptance changed; this frozen sparse-positive study cannot absorb it")
    # Reuse the protocol-bound matrix constructor, never infer states from files.
    from .followup_study import cells
    states = cells(protocol)
    keys = [(c["case_id"], c["arm_id"]) for c in states]
    require(len(keys) == len(set(keys)) == 242, "Expected exactly 242 distinct audit states")
    is_reused = lambda c: c["study"] == "regional" and c["calibration_method"] == "global"
    reused_keys = {(c["case_id"], c["arm_id"]) for c in states if is_reused(c)}
    require(len(reused_keys) == 85, "Expected exactly 85 reused and 157 new states")
    evaluation = verifier.read_json(root / "evaluation_complete.json")
    require(evaluation == dict(status="PASS", cells=242, curve_rows=3140), "Numerical evaluation is incomplete")
    forecast = verifier.read_json(root / "audit_forecast.json")
    forecast_keys = [(r["case_id"], r["arm_id"]) for r in forecast["cells"]]
    require(forecast.get("status") == "PASS" and len(forecast_keys) == len(set(forecast_keys)) == 242
            and set(forecast_keys) == set(keys), "Audit forecast does not cover the exact matrix")
    expected = dict(zip(forecast_keys, forecast["cells"]))
    replication = verifier.read_json(root / "baseline_replication.json")
    replication_keys = [(r["case_id"], r["arm_id"]) for r in replication["cells"]]
    require(replication.get("status") == "PASS" and len(replication_keys) == len(set(replication_keys)) == 85
            and set(replication_keys) == reused_keys
            and all(r.get("all_thresholds_equal") is True and r.get("metrics_equal") is True for r in replication["cells"]),
            "Global control replication is incomplete")
    display = verifier.binding(root / "display_contract.json")
    result = []
    for cell in states:
        case, arm = cell["case_id"], cell["arm_id"]
        audit = root / "audits" / case / arm
        current_cell = root / "cells" / case / arm
        new_seal, new_op, new_candidates = _cell_seal(current_cell, verifier)
        reused = is_reused(cell)
        if reused:
            old_cell = baseline / "cells" / case / cell["base_arm"]
            require(audit.resolve(strict=True) == (baseline / "audits" / case / cell["base_arm"]).resolve(strict=True),
                    "Reused audit does not resolve to the original baseline audit")
            old_seal, old_op, old_candidates = _cell_seal(old_cell, verifier)
            for name in ("Raw", "Input", "Score"):
                require(new_seal["stages"][name]["sha256"] == old_seal["stages"][name]["sha256"], "Reused audit stage differs")
            require(len(new_candidates) == len(old_candidates)
                    and all(all(a.get(k) == b.get(k) for k in CANDIDATE_FIELDS) for a, b in zip(new_candidates, old_candidates)),
                    "Reused audit candidates differ")
            for name in ("threshold", "threshold_z", "setup_source_frames_ui", "scale_floor",
                         "target_proposals_per_frame", "application_source_start_ui", "application_source_stop_ui"):
                require(new_op[name] == old_op[name], f"Reused q1 operating point differs: {name}")
            owner_cell, owner_seal, owner_op = old_cell, old_seal, old_op
        else:
            require(audit.resolve(strict=True) == audit.absolute(), "New audit unexpectedly aliases another root")
            owner_cell, owner_seal, owner_op = current_cell, new_seal, new_op
        sources = verify_audit_files(audit, verifier)
        summary = verifier.read_json(audit / "summary.json")
        status = verifier.read_json(audit / "status.json")
        validation = verifier.read_json(audit / "validation.json")
        inventory = verifier.read_json(audit / "inventory.json")
        require(status.get("status") == "complete" and status.get("scientific_audit_complete") is True,
                f"Audit status is incomplete: {case}/{arm}")
        require(summary.get("scientific_audit_complete") is True and validation.get("status") == "passed"
                and validation.get("failures") == [] and validation.get("scientific_audit_complete") is True
                and inventory.get("complete") is True, f"Audit validation is incomplete: {case}/{arm}")
        wanted = expected[(case, arm)]
        for actual, target in (("video_count", "expected_videos"), ("expert_roi_count", "expert_rois"),
                               ("model_roi_count", "model_rois"), ("expert_occurrence_count", "expert_occurrences")):
            require(summary[actual] == wanted[target], f"Audit inventory forecast differs: {case}/{arm}/{actual}")
        require(summary["model_proposal_count"] == len(new_candidates)
                and summary["comparison_trace_count"] == summary["expert_occurrence_count"], "Audit q1 proposal/comparison counts differ")
        contract = verifier.read_json(audit / "run_contract.json")
        require(contract["operating_point"] == owner_op and summary["operating_point"] == owner_op,
                "Audit operating point differs from its frozen calibration")
        source_binding = contract["source_binding"]
        require(sources["source_binding"] == source_binding, "Audit source-manifest contract differs")
        require(source_binding["candidates_sha256"] == owner_seal["audit_candidates"]["sha256"]
                and source_binding["candidate_seal_sha256"] == verifier.binding(owner_cell / "sealed.json")["sha256"],
                "Audit is not bound to its scored candidate seal")
        if not reused:
            require(source_binding["display_contract_sha256"] == display["sha256"], "New audit display contract changed")
        for name, path in contract["stage_paths"].items():
            record = owner_seal["stages"][name]
            require(Path(path).resolve() == Path(record["path"]).resolve()
                    and source_binding["stage_sha256"][name] == record["sha256"], "Audit stage differs from sealed score state")
            verifier.verify(record)
        result.append(dict(cell, reused=reused, summary=summary,
                           metadata_bindings={name: verifier.binding(audit / name) for name in METADATA}))
    return dict(status="PASS", cells=242, new_cells=157, reused_cells=85, audits=result,
                worker_receipts=worker_records)


def _write(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def validate(root: str | Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Write aggregate PASS only after the complete, single-pass file verification."""
    root = Path(root).resolve(strict=True)
    started = time.monotonic()
    verifier = FileVerifier()
    validator = verifier.binding(Path(__file__))
    reference = verifier.binding(Path(__file__).with_name("two_stencil_audit.py"))
    try:
        result = _validate(root, verifier)
        verifier.assert_unchanged()
        result.update(validated_utc=datetime.now(timezone.utc).isoformat(), validator=validator,
                      equivalent_file_verifier=reference, unique_hash_count=verifier.unique_file_count,
                      bytes_hashed=verifier.bytes_hashed, cached_binding_checks=verifier.cache_hits,
                      runtime_seconds=time.monotonic() - started,
                      verification_scope="Every artifact_index.artifacts, source_manifest.sources and numeric_state_sources file was SHA-256 checked. Repeated resolved paths reused a hash only while file identity, size, mtime and ctime and observed symlink targets remained unchanged. All were stat-checked again at closure; the cache is private to this pass. Full video decoding and RGB/marker checks are inherited from the verified completed per-cell audit records, not rerun here.")
        _write(root / "audit_complete.json", result)
        return result
    except Exception as error:
        failure = dict(status="FAIL", cells=0, scientific_audit_complete=False,
                       validator=validator, error_type=type(error).__name__, error=str(error),
                       unique_hash_count=verifier.unique_file_count, bytes_hashed=verifier.bytes_hashed,
                       runtime_seconds=time.monotonic() - started)
        _write(root / "audit_validation_failure.json", failure)
        # A failed revalidation must never leave an older aggregate PASS usable.
        if (root / "audit_complete.json").exists():
            _write(root / "audit_complete.json", failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    value = validate(args.root)
    print(json.dumps({k: value[k] for k in ("status", "cells", "new_cells", "reused_cells", "unique_hash_count", "bytes_hashed", "runtime_seconds")}), flush=True)
