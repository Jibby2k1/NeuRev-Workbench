import json
from pathlib import Path

from neurobench.dynamics.supervisor import build_sweep_health_report, build_sweep_live_status, classify_failure, create_resume_script


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_sweep_health_report_summarizes_current_and_archived_failures(tmp_path):
    sweep = tmp_path / "sweep"
    _write_json(
        sweep / "sweep_manifest.json",
        {
            "profile": "grid128_sequence_1day",
            "experiment_count": 3,
            "batch_size": 4,
            "epochs": 50,
            "seeds": [7, 13],
            "device": "cuda",
            "experiments": [
                {"experiment_id": "g128_a", "kind": "latent_gru", "dataset_key": "w8_s1_h2", "params": {"hidden_dim": 64}},
                {"experiment_id": "g128_b", "kind": "latent_gru", "dataset_key": "w8_s1_h2", "params": {"hidden_dim": 64}},
                {"experiment_id": "g128_c", "kind": "convgru_pixel", "dataset_key": "w8_s1_h5", "params": {"architecture": "convgru_pixel", "hidden_channels": 64}},
            ],
        },
    )
    _append_jsonl(
        sweep / "sweep_progress.jsonl",
        [
            {"index": 1, "experiment_count": 3, "experiment_id": "g128_a", "kind": "latent_gru", "dataset_key": "w8_s1_h2", "status": "completed"},
            {"index": 2, "experiment_count": 3, "experiment_id": "g128_b", "kind": "latent_gru", "dataset_key": "w8_s1_h2", "status": "completed"},
        ],
    )
    _append_jsonl(
        sweep / "sweep_progress_batch64_oom.jsonl",
        [
            {
                "index": 3,
                "experiment_count": 3,
                "experiment_id": "g128_c",
                "kind": "convgru_pixel",
                "dataset_key": "w8_s1_h5",
                "status": "failed",
                "error": "OutOfMemoryError('CUDA out of memory')",
            }
        ],
    )

    summary = build_sweep_health_report(sweep_dir=sweep)
    report = Path(summary["report_path"]).read_text(encoding="utf-8")

    assert summary["status_counts"] == {"completed": 2}
    assert summary["archive_failure_summary"]["sweep_progress_batch64_oom.jsonl"]["failure_classes"] == {"cuda_oom": 1}
    assert "Archived Failure Evidence" in report
    assert "Archived OOMs found" in report
    assert "g128_c" in report


def test_sweep_health_report_monitors_oom_cluster_followed_by_success(tmp_path):
    sweep = tmp_path / "sweep"
    _write_json(
        sweep / "sweep_manifest.json",
        {
            "profile": "grid128_sequence_1day",
            "experiment_count": 4,
            "batch_size": 4,
            "experiments": [
                {"experiment_id": "g128_a", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2"},
                {"experiment_id": "g128_b", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2"},
                {"experiment_id": "g128_c", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2"},
                {"experiment_id": "g128_d", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2"},
            ],
        },
    )
    _append_jsonl(
        sweep / "sweep_progress.jsonl",
        [
            {"index": 1, "experiment_count": 4, "experiment_id": "g128_a", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "failed", "error": "CUDA out of memory"},
            {"index": 2, "experiment_count": 4, "experiment_id": "g128_b", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "failed", "error": "CUDA out of memory"},
            {"index": 3, "experiment_count": 4, "experiment_id": "g128_c", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "completed"},
            {"index": 4, "experiment_count": 4, "experiment_id": "g128_d", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "completed"},
        ],
    )

    summary = build_sweep_health_report(sweep_dir=sweep, include_archives=False)
    report = Path(summary["report_path"]).read_text(encoding="utf-8")

    assert summary["current_failure_summary"]["successful_records_after_last_failure"] == 2
    assert summary["current_failure_summary"]["trailing_failure_count"] == 0
    assert "later progress-file records completed or skipped" in report
    assert "keep batch_size=4 for now" in report


def test_sweep_health_report_infers_next_spec_without_active_status(tmp_path):
    sweep = tmp_path / "sweep"
    _write_json(
        sweep / "sweep_manifest.json",
        {
            "profile": "grid128_sequence_1day",
            "experiment_count": 2,
            "batch_size": 2,
            "experiments": [
                {"experiment_id": "g128_a", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "seed": 7, "params": {"hyperparameter_summary": "hc=32"}},
                {"experiment_id": "g128_b", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "seed": 7, "params": {"hyperparameter_summary": "hc=64"}},
            ],
        },
    )
    _append_jsonl(
        sweep / "sweep_progress.jsonl",
        [
            {"index": 1, "experiment_count": 2, "experiment_id": "g128_a", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "skipped"},
        ],
    )

    summary = build_sweep_health_report(sweep_dir=sweep, include_archives=False)
    report = Path(summary["report_path"]).read_text(encoding="utf-8")

    assert summary["inferred_next_spec"]["index"] == 2
    assert summary["inferred_next_spec"]["experiment_id"] == "g128_b"
    assert summary["inferred_next_spec"]["hyperparameter_summary"] == "hc=64"
    assert "## Inferred Next Spec" in report
    assert "g128_b" in report
    assert "No `sweep_active.json` was found" in report


def test_sweep_health_report_includes_active_status_when_present(tmp_path):
    sweep = tmp_path / "sweep"
    _write_json(
        sweep / "sweep_manifest.json",
        {"profile": "grid128_sequence_1day", "experiment_count": 2, "batch_size": 2},
    )
    _append_jsonl(
        sweep / "sweep_progress.jsonl",
        [
            {"index": 1, "experiment_count": 2, "experiment_id": "g128_a", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "skipped"},
        ],
    )
    _write_json(
        sweep / "sweep_active.json",
        {
            "index": 2,
            "experiment_count": 2,
            "experiment_id": "g128_b",
            "kind": "convgru_pixel",
            "dataset_key": "w8_s1_h2",
            "status": "running",
            "updated_at": "2026-06-10T14:24:00+00:00",
        },
    )

    summary = build_sweep_health_report(sweep_dir=sweep, include_archives=False)
    report = Path(summary["report_path"]).read_text(encoding="utf-8")

    assert summary["active_status"]["status"] == "running"
    assert summary["active_status"]["experiment_id"] == "g128_b"
    assert "## Active Spec" in report
    assert "g128_b" in report
    assert "running" in report




def test_sweep_health_report_does_not_call_completed_active_marker_stale(tmp_path):
    sweep = tmp_path / "sweep"
    _write_json(
        sweep / "sweep_manifest.json",
        {"profile": "grid128_sequence_1day", "experiment_count": 2, "batch_size": 2},
    )
    progress_path = sweep / "sweep_progress.jsonl"
    _append_jsonl(
        progress_path,
        [
            {"index": 1, "experiment_count": 2, "experiment_id": "g128_a", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "completed"},
            {"index": 2, "experiment_count": 2, "experiment_id": "g128_b", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "completed"},
        ],
    )
    _write_json(
        sweep / "sweep_active.json",
        {
            "index": 2,
            "experiment_count": 2,
            "experiment_id": "g128_b",
            "kind": "convgru_pixel",
            "dataset_key": "w8_s1_h2",
            "status": "completed",
            "updated_at": "2026-06-10T14:24:00+00:00",
        },
    )
    old_time = 1_700_000_000
    progress_path.touch()
    import os

    os.utime(progress_path, (old_time, old_time))

    summary = build_sweep_health_report(sweep_dir=sweep, include_archives=False, stale_minutes=0.01)
    report = Path(summary["report_path"]).read_text(encoding="utf-8")

    assert summary["active_status"]["status"] == "completed"
    assert not any("Progress file is stale" in flag for flag in summary["health_flags"])
    assert "Progress appears stale" not in report
    assert "progress age reflects a stopped run" in report

def test_sweep_health_report_treats_resumed_lower_index_skips_as_recovery(tmp_path):
    sweep = tmp_path / "sweep"
    _write_json(
        sweep / "sweep_manifest.json",
        {"profile": "grid128_sequence_1day", "experiment_count": 4, "batch_size": 2},
    )
    _append_jsonl(
        sweep / "sweep_progress.jsonl",
        [
            {"index": 3, "experiment_count": 4, "experiment_id": "g128_c", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "failed", "error": "CUDA out of memory"},
            {"index": 4, "experiment_count": 4, "experiment_id": "g128_d", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "failed", "error": "CUDA out of memory"},
            {"index": 1, "experiment_count": 4, "experiment_id": "g128_a", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "skipped"},
            {"index": 2, "experiment_count": 4, "experiment_id": "g128_b", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "skipped"},
        ],
    )

    summary = build_sweep_health_report(sweep_dir=sweep, include_archives=False)
    report = Path(summary["report_path"]).read_text(encoding="utf-8")

    assert summary["current_failure_summary"]["successful_records_after_last_failure"] == 2
    assert summary["current_failure_summary"]["trailing_failure_count"] == 0
    assert "later progress-file records completed or skipped" in report
    assert "keep batch_size=2 for now" in report
    assert "CUDA OOMs are present in current progress; archive progress" not in report
    assert "latest 2 record(s) failed with CUDA OOM" not in report


def test_sweep_health_report_recommends_resume_for_trailing_ooms(tmp_path):
    sweep = tmp_path / "sweep"
    _write_json(
        sweep / "sweep_manifest.json",
        {"profile": "grid128_sequence_1day", "experiment_count": 3, "batch_size": 4},
    )
    _append_jsonl(
        sweep / "sweep_progress.jsonl",
        [
            {"index": 1, "experiment_count": 3, "experiment_id": "g128_a", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "completed"},
            {"index": 2, "experiment_count": 3, "experiment_id": "g128_b", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "failed", "error": "CUDA out of memory"},
            {"index": 3, "experiment_count": 3, "experiment_id": "g128_c", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "failed", "error": "CUDA out of memory"},
        ],
    )

    summary = build_sweep_health_report(sweep_dir=sweep, include_archives=False)
    report = Path(summary["report_path"]).read_text(encoding="utf-8")

    assert summary["current_failure_summary"]["successful_records_after_last_failure"] == 0
    assert summary["current_failure_summary"]["trailing_failure_count"] == 2
    assert "latest 2 record(s) failed with CUDA OOM" in report
    assert "resume with batch_size=2" in report


def test_create_resume_script_uses_manifest_defaults_and_requested_batch_size(tmp_path):
    sweep = tmp_path / "sweep"
    _write_json(
        sweep / "sweep_manifest.json",
        {"profile": "grid128_sequence_1day", "batch_size": 64, "epochs": 50, "seeds": [7, 13], "device": "cuda", "time_limit_hours": 48},
    )
    script = create_resume_script(sweep_dir=sweep, batch_size=4, script_path=tmp_path / "resume.sh")
    text = script.read_text(encoding="utf-8")

    assert "--profile' 'grid128_sequence_1day" in text
    assert "--batch-size' '4" in text
    assert "--seeds' '7,13" in text
    assert "setsid" in text


def test_classify_failure_groups_common_failure_modes():
    assert classify_failure("OutOfMemoryError('CUDA out of memory')") == "cuda_oom"
    assert classify_failure("FileNotFoundError: missing file") == "missing_artifact"
    assert classify_failure("RuntimeError: size mismatch for layer") == "shape_mismatch"
    assert classify_failure("loss is NaN") == "numeric_instability"


def test_sweep_live_status_reports_active_training_with_gpu_evidence(tmp_path, monkeypatch):
    sweep = tmp_path / "sweep"
    _write_json(
        sweep / "sweep_manifest.json",
        {
            "profile": "grid128_sequence_1day",
            "experiment_count": 2,
            "experiments": [
                {"experiment_id": "g128_a", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2"},
                {"experiment_id": "g128_b", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2"},
            ],
        },
    )
    _append_jsonl(
        sweep / "sweep_progress.jsonl",
        [{"index": 1, "experiment_count": 2, "experiment_id": "g128_a", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "skipped"}],
    )
    _write_json(
        sweep / "sweep_active.json",
        {"index": 2, "experiment_count": 2, "experiment_id": "g128_b", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "running"},
    )
    (sweep / "g128_b").mkdir(parents=True)
    (sweep / "g128_b" / "experiment_config.json").write_text("{}\n", encoding="utf-8")

    class Result:
        def __init__(self, stdout: str, returncode: int = 0):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = returncode

    def fake_run(command, check=False, capture_output=True, text=True):
        cmd = " ".join(command)
        if command[:2] == ["ps", "-p"]:
            return Result(f"2235445 2040 Rsl 00:12 99.9 13.7 11307108 46775100 python -m neurobench.dynamics.overnight_sweep --out-dir {sweep}\n")
        if "--query-gpu" in cmd:
            return Result("2026/06/10 11:00:00.000, 99, 60, 11620, 12282\n")
        if "--query-compute-apps" in cmd:
            return Result("2235445, python, 10664\n")
        raise AssertionError(command)

    monkeypatch.setattr("neurobench.dynamics.supervisor.subprocess.run", fake_run)

    status = build_sweep_live_status(sweep_dir=sweep, pid=2235445)
    report = Path(status["report_path"]).read_text(encoding="utf-8")

    assert status["live_state"] == "active_training"
    assert status["process"]["pid"] == 2235445
    assert status["gpu"]["process_used_memory_mib"] == 10664.0
    assert status["active_artifacts"]["metrics_present"] is False
    assert "Sweep Live Status" in report
    assert "active_training" in report


def test_sweep_live_status_reports_cpu_phase_when_gpu_util_is_low(tmp_path, monkeypatch):
    sweep = tmp_path / "sweep"
    _write_json(
        sweep / "sweep_manifest.json",
        {
            "profile": "grid128_sequence_1day",
            "experiment_count": 2,
            "experiments": [
                {"experiment_id": "g128_a", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2"},
                {"experiment_id": "g128_b", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2"},
            ],
        },
    )
    _append_jsonl(
        sweep / "sweep_progress.jsonl",
        [{"index": 1, "experiment_count": 2, "experiment_id": "g128_a", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "skipped"}],
    )
    _write_json(
        sweep / "sweep_active.json",
        {"index": 2, "experiment_count": 2, "experiment_id": "g128_b", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "running"},
    )
    (sweep / "g128_b").mkdir(parents=True)
    (sweep / "g128_b" / "experiment_config.json").write_text("{}\n", encoding="utf-8")

    class Result:
        def __init__(self, stdout: str, returncode: int = 0):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = returncode

    def fake_run(command, check=False, capture_output=True, text=True):
        cmd = " ".join(command)
        if command[:2] == ["ps", "-p"]:
            return Result(f"2235445 2040 Rsl 58:11 100.0 31.5 11307108 46775100 python -m neurobench.dynamics.overnight_sweep --out-dir {sweep}\n")
        if "--query-gpu" in cmd:
            return Result("2026/06/10 11:00:00.000, 5, 4, 11620, 12282\n")
        if "--query-compute-apps" in cmd:
            return Result("2235445, python, 10664\n")
        raise AssertionError(command)

    monkeypatch.setattr("neurobench.dynamics.supervisor.subprocess.run", fake_run)

    status = build_sweep_live_status(sweep_dir=sweep, pid=2235445)
    report = Path(status["report_path"]).read_text(encoding="utf-8")

    assert status["live_state"] == "active_cpu_phase"
    assert status["process"]["cpu_percent"] == 100.0
    assert status["gpu"]["utilization_gpu_percent"] == 5.0
    assert status["gpu"]["process_used_memory_mib"] == 10664.0
    assert "active_cpu_phase" in report


def test_sweep_live_status_detects_metrics_artifact(tmp_path):
    sweep = tmp_path / "sweep"
    _write_json(sweep / "sweep_manifest.json", {"profile": "grid128_sequence_1day", "experiment_count": 1})
    _write_json(
        sweep / "sweep_active.json",
        {"index": 1, "experiment_count": 1, "experiment_id": "g128_done", "kind": "convgru_pixel", "dataset_key": "w8_s1_h2", "status": "running"},
    )
    metrics = sweep / "g128_done" / "convgru_pixel_residual_mse" / "concept_metrics.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("{}\n", encoding="utf-8")

    status = build_sweep_live_status(sweep_dir=sweep, include_gpu=False)

    assert status["active_artifacts"]["metrics_present"] is True
    assert status["live_state"] == "metrics_written"
