import json

from neurobench.cli.main import main
from neurobench.dynamics.artifact_audit import ArtifactSpec, build_grid128_artifact_audit
from neurobench.dynamics.launch_readiness import build_grid128_stage_b_launch_readiness


def _write_readiness_inputs(root):
    sweep_dir = root / "sweeps" / "grid128_sequence_1day_v1"
    plan_dir = root / "plans" / "grid128_sequence_stage_b_v1"
    stop_dir = root / "plans" / "grid128_stage_a_stop_review_v1"
    report_dir = root / "reports" / "grid128_sequence_1day_partial_report_v1"
    audit_dir = root / "plans" / "grid128_artifact_audit_v1"
    for directory in (sweep_dir, plan_dir / "stage_b_sweep", stop_dir, report_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    active = {
        "index": 7,
        "experiment_count": 10,
        "status": "completed",
        "experiment_id": "model_done",
        "finished_at": "2026-06-13T00:00:00+00:00",
    }
    (sweep_dir / "sweep_active.json").write_text(json.dumps(active), encoding="utf-8")
    (sweep_dir / "sweep_progress.jsonl").write_text(
        json.dumps({"index": 6, "experiment_count": 10, "status": "failed", "experiment_id": "model_failed"})
        + "\n"
        + json.dumps(active)
        + "\n",
        encoding="utf-8",
    )
    plan = {
        "created_at": "2026-06-13T01:00:00+00:00",
        "planned_experiment_count": 3,
        "selection_counts": {"latent_gru": 2, "linear_latent": 1},
    }
    manifest_experiments = [{"experiment_id": "a"}, {"experiment_id": "b"}, {"experiment_id": "c"}]
    (plan_dir / "next_sweep_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (plan_dir / "next_sweep_manifest.json").write_text(
        json.dumps({"planned_experiment_count": 3, "experiments": manifest_experiments}),
        encoding="utf-8",
    )
    (plan_dir / "stage_b_sweep" / "sweep_manifest.json").write_text(
        json.dumps({"experiment_count": 3, "experiments": manifest_experiments}),
        encoding="utf-8",
    )
    (stop_dir / "stage_a_stop_review.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-06-13T02:00:00+00:00",
                "former_pid": 123,
                "recommendation": "Use the refreshed Stage B manifest as the default next GPU job unless the user explicitly asks to continue the original Stage A sweep from index 478.",
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "dynamics_experiment_report.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-06-13T02:30:00+00:00",
                "artifact_audit_summary": {"artifact_count": 5, "status_counts": {"ok": 5}},
            }
        ),
        encoding="utf-8",
    )
    (audit_dir / "grid128_artifact_audit.json").write_text(
        json.dumps({"artifact_count": 5, "status_counts": {"ok": 5}, "ok": True}),
        encoding="utf-8",
    )


def test_stage_b_launch_readiness_builder_writes_current_handoff(tmp_path):
    root = tmp_path / "root"
    _write_readiness_inputs(root)

    readiness = build_grid128_stage_b_launch_readiness(
        root=root,
        out_dir=root / "plans" / "grid128_stage_b_launch_readiness_v1",
    )

    assert readiness["status"] == "ready_pending_user_approval"
    assert readiness["decision"]["default_next_gpu_job"] == "stage_b_manifest"
    assert readiness["decision"]["requires_user_approval"] is True
    assert readiness["stage_a"]["progress_text"] == "7 / 10"
    assert readiness["stage_a"]["status_counts"] == {"failed": 1, "completed": 1}
    assert readiness["stage_b"]["planned_experiment_count"] == 3
    assert readiness["stage_b"]["manifest_list_count"] == 3
    assert readiness["stage_b"]["dry_run_validated"] is True
    assert readiness["evidence"]["audit_artifact_count"] == 5
    assert readiness["evidence"]["report_embedded_audit_status_counts"] == {"ok": 5}
    assert {item["id"] for item in readiness["pre_launch_checklist"]} == {
        "confirm_user_gpu_choice",
        "rerun_process_check",
        "verify_artifact_audit_ok",
        "launch_stage_b_manifest_or_explicit_stage_a_resume",
    }

    markdown = (root / "plans" / "grid128_stage_b_launch_readiness_v1" / "stage_b_launch_readiness.md").read_text(encoding="utf-8")
    assert "## Pre-Launch Checklist" in markdown
    assert "--manifest" in markdown
    assert "stage_b_sweep" in markdown


def test_stage_b_launch_readiness_output_passes_artifact_audit(tmp_path):
    root = tmp_path / "root"
    _write_readiness_inputs(root)
    build_grid128_stage_b_launch_readiness(
        root=root,
        out_dir=root / "plans" / "grid128_stage_b_launch_readiness_v1",
    )

    audit = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[
            ArtifactSpec(
                "Stage B launch readiness JSON",
                "plans/grid128_stage_b_launch_readiness_v1/stage_b_launch_readiness.json",
                "plan",
                True,
            )
        ],
    )

    assert audit["ok"] is True
    assert audit["artifacts"][0]["summary"]["stage_b_launch_readiness_matches_current_state"] is True


def test_stage_b_launch_readiness_cli_writes_artifacts(tmp_path, capsys):
    root = tmp_path / "root"
    out_dir = root / "plans" / "grid128_stage_b_launch_readiness_v1"
    _write_readiness_inputs(root)

    exit_code = main(
        [
            "dynamics",
            "stage-b-launch-readiness",
            "--root",
            str(root),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Stage B launch readiness:" in captured.out
    assert (out_dir / "stage_b_launch_readiness.json").is_file()
