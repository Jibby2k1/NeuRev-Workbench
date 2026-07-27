from __future__ import annotations

from pathlib import Path
import tempfile


def test_job_store_is_atomic_and_restart_aware():
    from neurobench.workbench.jobs import JobStore

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "jobs"
        store = JobStore(root)
        created = store.create("import", {"dataset_id": "fish"}, job_id="job one")
        assert created["status"] == "queued"
        store.update(created["job_id"], status="running", stage="copy", progress=0.25, owner_pid=99999999)
        store.append_log(created["job_id"], "copying")
        restored = JobStore(root)
        recovered = restored.recover_incomplete()
        assert len(recovered) == 1
        current = restored.get(created["job_id"])
        assert current is not None
        assert current["status"] == "stopped"
        assert current["stage"] == "recovered_after_restart"
        assert current["log_tail"] == ["copying"]


def test_job_store_rejects_invalid_state_and_progress():
    import pytest
    from neurobench.workbench.jobs import JobStore

    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp))
        job = store.create("test")
        with pytest.raises(ValueError):
            store.update(job["job_id"], status="running-ish")
        with pytest.raises(ValueError):
            store.update(job["job_id"], progress=2)


def test_job_store_deduplicates_active_actions_only():
    from neurobench.workbench.jobs import JobStore

    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp))
        first, first_created = store.create_or_get_active("dataset_qc", {"import_id": "imp_a"}, dedupe_key="demo:imp_a:qc")
        duplicate, duplicate_created = store.create_or_get_active("dataset_qc", {"import_id": "imp_a"}, dedupe_key="demo:imp_a:qc")
        store.update(first["job_id"], status="completed", stage="complete", progress=1.0)
        retry, retry_created = store.create_or_get_active("dataset_qc", {"import_id": "imp_a"}, dedupe_key="demo:imp_a:qc")

    assert first_created is True
    assert duplicate_created is False
    assert duplicate["job_id"] == first["job_id"]
    assert retry_created is True
    assert retry["job_id"] != first["job_id"]


def test_recovery_does_not_stop_a_live_owner_process():
    from neurobench.workbench.jobs import JobStore

    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp))
        job = store.create("dataset_qc")
        store.update(job["job_id"], status="running", stage="sampling")
        recovered = JobStore(Path(tmp)).recover_incomplete()
        current = store.get(job["job_id"])

    assert recovered == []
    assert current["status"] == "running"
