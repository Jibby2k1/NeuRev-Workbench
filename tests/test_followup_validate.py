"""Isolated file-verification tests; no production arrays, rendering or mocks."""
import hashlib
import json
import os

import pytest

from neurobench.experiments.gamma_ls_difference.followup_validate import (
    FileVerifier, METADATA, _workers, validate, verify_audit_files,
)


def record(path, *, relative=None):
    return dict(path=str(path.relative_to(relative) if relative else path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def test_shared_file_and_symlink_are_hashed_once(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"shared source bytes")
    alias = tmp_path / "alias.bin"
    alias.symlink_to(source)
    verifier = FileVerifier()
    verifier.verify(record(source))
    verifier.verify(record(alias))
    verifier.verify(record(source, relative=tmp_path), root=tmp_path)
    verifier.assert_unchanged()
    assert verifier.unique_file_count == 1
    assert verifier.bytes_hashed == len(b"shared source bytes")
    assert verifier.cache_hits == 2


def test_changed_same_size_file_rejected_even_after_mtime_restoration(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"old")
    original_stat = path.stat()
    expected = record(path)
    verifier = FileVerifier()
    verifier.verify(expected)
    path.write_bytes(b"new")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    with pytest.raises(ValueError, match="changed during validation"):
        verifier.verify(expected)


def test_same_content_replacement_is_not_reused(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"unchanged bytes")
    verifier = FileVerifier()
    verifier.verify(record(path))
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    with pytest.raises(ValueError, match="changed during validation"):
        verifier.binding(path)


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("size_bytes", 999)])
def test_conflicting_bindings_fail(tmp_path, field, value):
    path = tmp_path / "source.bin"
    path.write_bytes(b"fixed")
    expected = record(path)
    verifier = FileVerifier()
    verifier.verify(expected)
    with pytest.raises(ValueError, match="Conflicting frozen"):
        verifier.verify(dict(expected, **{field: value}))


def test_existence_is_not_a_sha_check(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"wrong but present")
    verifier = FileVerifier()
    with pytest.raises(ValueError, match="SHA-256 or size mismatch"):
        verifier.verify(dict(record(path), sha256="0" * 64))


def test_final_sweep_rejects_later_change(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"first")
    verifier = FileVerifier()
    verifier.verify(record(path))
    path.write_bytes(b"different length")
    with pytest.raises(ValueError, match="changed during validation"):
        verifier.assert_unchanged()


def test_alias_target_change_rejected_even_for_equal_bytes(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    alias = tmp_path / "alias"
    alias.symlink_to(first)
    verifier = FileVerifier()
    expected = record(alias)
    verifier.verify(expected)
    alias.unlink()
    alias.symlink_to(second)
    with pytest.raises(ValueError, match="link target changed"):
        verifier.verify(expected)


def audit_fixture(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    shared = tmp_path / "shared.bin"
    shared.write_bytes(b"source and numerical authority")
    artifact = audit / "movie.bin"
    artifact.write_bytes(b"encoded artifact stand-in")
    for name in METADATA:
        if name not in ("artifact_index.json", "source_manifest.json"):
            write_json(audit / name, {"name": name})
    write_json(audit / "source_manifest.json", dict(
        sources=[record(shared)], numeric_state_sources=[record(shared)]))
    write_json(audit / "artifact_index.json", dict(
        artifacts=[record(p, relative=audit) for p in sorted(audit.iterdir())]))
    return audit, shared, artifact


def test_all_original_file_lists_checked_with_cross_list_deduplication(tmp_path):
    audit, shared, _ = audit_fixture(tmp_path)
    verifier = FileVerifier()
    verify_audit_files(audit, verifier)
    verifier.assert_unchanged()
    assert verifier.unique_file_count == len(list(audit.iterdir())) + 1
    assert verifier.bytes_hashed == sum(p.stat().st_size for p in audit.iterdir()) + shared.stat().st_size


def test_changed_numeric_source_and_missing_artifact_fail(tmp_path):
    audit, shared, artifact = audit_fixture(tmp_path)
    shared.write_bytes(b"changed shared authority")
    with pytest.raises(ValueError, match="SHA-256 or size mismatch"):
        verify_audit_files(audit, FileVerifier())
    artifact.unlink()
    with pytest.raises(FileNotFoundError):
        verify_audit_files(audit, FileVerifier())


def write_workers(root, workers=3):
    for worker in range(workers):
        write_json(root / f"media_worker_{worker}.json", dict(
            status="PASS", worker=worker, workers=workers, cells=len(range(worker, 242, workers))))
        (root / f"media_worker_{worker}.log").write_text(f"worker {worker} complete\n")


def test_worker_partitions_and_logs_are_bound(tmp_path):
    write_workers(tmp_path)
    verifier = FileVerifier()
    rows = _workers(tmp_path, verifier, 242)
    assert [r["cells"] for r in rows] == [81, 81, 80]
    assert len(rows) == 3 and verifier.unique_file_count == 6
    assert all(r["receipt"]["sha256"] and r["log"]["sha256"] for r in rows)


def test_missing_or_partial_worker_blocks_validation(tmp_path):
    write_workers(tmp_path)
    (tmp_path / "media_worker_2.json").unlink()
    with pytest.raises(ValueError, match="Not all media workers"):
        _workers(tmp_path, FileVerifier(), 242)
    write_json(tmp_path / "media_worker_2.json", dict(status="PASS", worker=2, workers=3, cells=79))
    with pytest.raises(ValueError, match="full partition"):
        _workers(tmp_path, FileVerifier(), 242)


def test_failure_cannot_leave_stale_aggregate_pass(tmp_path):
    write_json(tmp_path / "audit_complete.json", dict(status="PASS", cells=242))
    with pytest.raises(ValueError, match="No completed media-worker"):
        validate(tmp_path)
    aggregate = json.loads((tmp_path / "audit_complete.json").read_text())
    assert aggregate["status"] == "FAIL" and aggregate["scientific_audit_complete"] is False
    assert (tmp_path / "audit_validation_failure.json").is_file()
