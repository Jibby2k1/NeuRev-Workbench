import json
from pathlib import Path
from types import SimpleNamespace

from neurobench.dynamics.artifact_audit import ArtifactSpec, DEFAULT_GRID128_ARTIFACTS, build_grid128_artifact_audit


def test_grid128_artifact_audit_reports_valid_missing_and_invalid_json(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.json").write_text(json.dumps({"models": [{"id": "a"}], "missing_visual_count": 0}), encoding="utf-8")
    (root / "bad.json").write_text("{not-json", encoding="utf-8")

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[
            ArtifactSpec("Valid review", "ok.json", "review", True),
            ArtifactSpec("Invalid JSON", "bad.json", "review", True),
            ArtifactSpec("Missing plan", "missing.md", "plan"),
        ],
    )

    assert report["ok"] is False
    assert report["status_counts"] == {"ok": 1, "invalid_json": 1, "missing": 1}
    rows = {row["label"]: row for row in report["artifacts"]}
    assert rows["Valid review"]["summary"]["model_count"] == 1
    assert rows["Valid review"]["summary"]["missing_visual_count"] == 0
    assert rows["Invalid JSON"]["error"]
    assert Path(report["json_path"]).is_file()
    markdown = Path(report["markdown_path"]).read_text(encoding="utf-8")
    assert "Valid review" in markdown
    assert "missing.md" in markdown


def test_grid128_artifact_audit_is_ok_when_all_artifacts_exist(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "report.md").write_text("# Report\n", encoding="utf-8")
    (root / "plan.json").write_text(json.dumps({"recommended_candidates": [{"config_id": "next"}]}), encoding="utf-8")

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[
            ArtifactSpec("Report", "report.md", "report"),
            ArtifactSpec("Plan", "plan.json", "plan", True),
        ],
    )

    assert report["ok"] is True
    assert report["status_counts"] == {"ok": 2}
    plan_row = next(row for row in report["artifacts"] if row["label"] == "Plan")
    assert plan_row["summary"]["next_candidate"] == "next"


def test_grid128_default_artifacts_include_current_backfill_preflights():
    labels = {spec.label for spec in DEFAULT_GRID128_ARTIFACTS}
    paths = {spec.relative_path for spec in DEFAULT_GRID128_ARTIFACTS}

    assert "Current active-cell backfill preflight" in labels
    assert "Current learned-leader backfill preflight" in labels
    assert "Current active-cell challenger backfill preflight" in labels
    assert "plans/grid128_backfill_preflight_v1/current_active_cell_leader_metric_backfill_preflight.json" in paths
    assert "plans/grid128_backfill_preflight_v1/current_learned_leader_metric_backfill_preflight.json" in paths
    assert "plans/grid128_backfill_preflight_v1/current_active_cell_challenger_metric_backfill_preflight.json" in paths
    assert "Stage A stop review" in labels
    assert "Stage A stop review JSON" in labels
    assert "Stage B launch readiness" in labels
    assert "Stage B launch readiness JSON" in labels
    assert "plans/grid128_stage_a_stop_review_v1/stage_a_stop_review.md" in paths
    assert "plans/grid128_stage_a_stop_review_v1/stage_a_stop_review.json" in paths
    assert "plans/grid128_stage_b_launch_readiness_v1/stage_b_launch_readiness.md" in paths
    assert "plans/grid128_stage_b_launch_readiness_v1/stage_b_launch_readiness.json" in paths


def _write_stage_a_stop_review_fixture(root: Path, *, progress_index: int = 7, stage_b_count: int = 3) -> None:
    sweep_dir = root / "sweeps" / "grid128_sequence_1day_v1"
    stage_b_dir = root / "plans" / "grid128_sequence_stage_b_v1"
    review_dir = root / "plans" / "grid128_stage_a_stop_review_v1"
    report_dir = root / "reports" / "grid128_sequence_1day_partial_report_v1"
    audit_dir = root / "plans" / "grid128_artifact_audit_v1"
    sweep_dir.mkdir(parents=True)
    stage_b_dir.mkdir(parents=True)
    review_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)
    active = {
        "index": progress_index,
        "experiment_count": 10,
        "status": "completed",
        "experiment_id": "model_done",
        "finished_at": "2026-06-13T00:00:00+00:00",
    }
    (sweep_dir / "sweep_active.json").write_text(json.dumps(active), encoding="utf-8")
    (sweep_dir / "sweep_progress.jsonl").write_text(
        json.dumps({"index": progress_index - 1, "experiment_count": 10, "status": "failed", "experiment_id": "model_failed"})
        + "\n"
        + json.dumps(active)
        + "\n",
        encoding="utf-8",
    )
    stage_b_plan = {
        "created_at": "2026-06-13T01:00:00+00:00",
        "planned_experiment_count": stage_b_count,
        "selection_counts": {"latent_gru": stage_b_count},
    }
    (stage_b_dir / "next_sweep_plan.json").write_text(json.dumps(stage_b_plan), encoding="utf-8")
    (stage_b_dir / "stage_b_sweep").mkdir()
    (stage_b_dir / "stage_b_sweep" / "sweep_manifest.json").write_text(
        json.dumps({"experiment_count": stage_b_count}),
        encoding="utf-8",
    )
    review = {
        "status": "stopped",
        "progress": {
            "current_index": progress_index,
            "experiment_count": 10,
            "progress_text": f"{progress_index} / 10",
            "record_count": 2,
            "status_counts": {"failed": 1, "completed": 1},
        },
        "latest_active_status": active,
        "stage_b": {
            "created_at": "2026-06-13T01:00:00+00:00",
            "planned_experiment_count": stage_b_count,
            "dry_run_experiment_count": stage_b_count,
            "selection_counts": {"latent_gru": stage_b_count},
        },
    }
    (review_dir / "stage_a_stop_review.json").write_text(json.dumps(review), encoding="utf-8")
    (report_dir / "dynamics_experiment_report.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-06-13T01:30:00+00:00",
                "artifact_audit_summary": {"artifact_count": 5, "status_counts": {"ok": 5}},
            }
        ),
        encoding="utf-8",
    )
    (audit_dir / "grid128_artifact_audit.json").write_text(
        json.dumps({"artifact_count": 5, "status_counts": {"ok": 5}, "ok": True}),
        encoding="utf-8",
    )


def test_grid128_artifact_audit_validates_stage_a_stop_review_matches_current_state(tmp_path):
    root = tmp_path / "root"
    _write_stage_a_stop_review_fixture(root)

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[
            ArtifactSpec(
                "Stage A stop review JSON",
                "plans/grid128_stage_a_stop_review_v1/stage_a_stop_review.json",
                "plan",
                True,
            )
        ],
    )

    row = report["artifacts"][0]
    assert report["ok"] is True
    assert row["summary"]["stage_a_stop_review_matches_current_state"] is True
    assert row["summary"]["stage_a_stop_review_progress"] == "7 / 10"
    assert row["summary"]["stage_a_stop_review_stage_b_count"] == 3


def test_grid128_artifact_audit_flags_stale_stage_a_stop_review(tmp_path):
    root = tmp_path / "root"
    _write_stage_a_stop_review_fixture(root)
    review_path = root / "plans" / "grid128_stage_a_stop_review_v1" / "stage_a_stop_review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["progress"]["current_index"] = 6
    review["progress"]["progress_text"] = "6 / 10"
    review["stage_b"]["planned_experiment_count"] = 2
    review_path.write_text(json.dumps(review), encoding="utf-8")

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[
            ArtifactSpec(
                "Stage A stop review JSON",
                "plans/grid128_stage_a_stop_review_v1/stage_a_stop_review.json",
                "plan",
                True,
            )
        ],
    )

    row = report["artifacts"][0]
    assert report["ok"] is False
    assert report["status_counts"] == {"stale_stage_a_stop_review": 1}
    assert row["status"] == "stale_stage_a_stop_review"
    assert row["summary"]["stage_a_stop_review_matches_current_state"] is False
    assert set(row["summary"]["stage_a_stop_review_mismatches"]) == {
        "progress.current_index",
        "progress.progress_text",
        "stage_b.planned_experiment_count",
    }




def _write_stage_b_launch_readiness_fixture(root: Path, *, progress_index: int = 7, stage_b_count: int = 3) -> None:
    _write_stage_a_stop_review_fixture(root, progress_index=progress_index, stage_b_count=stage_b_count)
    readiness_dir = root / "plans" / "grid128_stage_b_launch_readiness_v1"
    readiness_dir.mkdir(parents=True)
    readiness = {
        "schema_version": 1,
        "generated_at": "2026-06-13T02:00:00+00:00",
        "status": "ready_pending_user_approval",
        "decision": {
            "default_next_gpu_job": "stage_b_manifest",
            "requires_user_approval": True,
            "recommendation": "Use the refreshed Stage B manifest as the default next GPU job unless the user explicitly asks to continue the original Stage A sweep from index 478.",
        },
        "stage_a": {
            "current_index": progress_index,
            "experiment_count": 10,
            "progress_text": f"{progress_index} / 10",
            "record_count": 2,
            "status_counts": {"failed": 1, "completed": 1},
            "active_status": "completed",
            "active_experiment_id": "model_done",
        },
        "stage_b": {
            "plan_created_at": "2026-06-13T01:00:00+00:00",
            "planned_experiment_count": stage_b_count,
            "manifest_experiment_count": stage_b_count,
            "manifest_list_count": 0,
            "dry_run_experiment_count": stage_b_count,
            "selection_counts": {"latent_gru": stage_b_count},
        },
        "evidence": {
            "report_generated_at": "2026-06-13T01:30:00+00:00",
            "report_embedded_audit_artifact_count": 5,
            "report_embedded_audit_status_counts": {"ok": 5},
            "audit_artifact_count": 5,
            "audit_status_counts": {"ok": 5},
            "audit_ok": True,
        },
        "pre_launch_checklist": [
            {"id": "confirm_user_gpu_choice", "required": True},
            {"id": "rerun_process_check", "required": True},
            {"id": "verify_artifact_audit_ok", "required": True},
            {"id": "launch_stage_b_manifest_or_explicit_stage_a_resume", "required": True},
        ],
    }
    (readiness_dir / "stage_b_launch_readiness.json").write_text(json.dumps(readiness), encoding="utf-8")


def test_grid128_artifact_audit_validates_stage_b_launch_readiness_matches_current_state(tmp_path):
    root = tmp_path / "root"
    _write_stage_b_launch_readiness_fixture(root)

    report = build_grid128_artifact_audit(
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

    row = report["artifacts"][0]
    assert report["ok"] is True
    assert row["summary"]["stage_b_launch_readiness_matches_current_state"] is True
    assert row["summary"]["stage_b_launch_readiness_progress"] == "7 / 10"
    assert row["summary"]["stage_b_launch_readiness_plan_count"] == 3
    assert row["summary"]["stage_b_launch_readiness_audit_artifact_count"] == 5
    assert row["summary"]["expected_audit_artifact_count"] == 5
    assert row["summary"]["stage_b_launch_readiness_checklist_count"] == 4
    assert row["summary"]["stage_b_launch_readiness_missing_checklist_ids"] == []


def test_grid128_artifact_audit_flags_stale_stage_b_launch_readiness(tmp_path):
    root = tmp_path / "root"
    _write_stage_b_launch_readiness_fixture(root)
    readiness_path = root / "plans" / "grid128_stage_b_launch_readiness_v1" / "stage_b_launch_readiness.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["stage_a"]["current_index"] = 6
    readiness["stage_a"]["progress_text"] = "6 / 10"
    readiness["stage_b"]["planned_experiment_count"] = 2
    readiness["decision"]["requires_user_approval"] = False
    readiness["evidence"]["audit_artifact_count"] = 4
    readiness["evidence"]["report_embedded_audit_status_counts"] = {"ok": 4}
    readiness["pre_launch_checklist"] = [
        item for item in readiness["pre_launch_checklist"] if item["id"] != "rerun_process_check"
    ]
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")

    report = build_grid128_artifact_audit(
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

    row = report["artifacts"][0]
    assert report["ok"] is False
    assert report["status_counts"] == {"stale_stage_b_launch_readiness": 1}
    assert row["status"] == "stale_stage_b_launch_readiness"
    assert row["summary"]["stage_b_launch_readiness_matches_current_state"] is False
    assert set(row["summary"]["stage_b_launch_readiness_mismatches"]) == {
        "stage_a.current_index",
        "stage_a.progress_text",
        "stage_b.planned_experiment_count",
        "decision.requires_user_approval",
        "evidence.audit_artifact_count",
        "evidence.report_embedded_audit_status_counts",
        "pre_launch_checklist",
    }


def test_grid128_artifact_audit_summarizes_backfill_preflight_fields(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    preflight = {
        "schema_version": 1,
        "created_at": "2026-06-11T00:00:00+00:00",
        "dry_run": True,
        "architecture": "convgru_pixel",
        "dataset_window_count": 17565,
        "estimated_metric_batches": 1098,
        "estimated_uncompressed_gib": 10.737,
        "would_update_metrics": True,
        "would_backfill_metrics": True,
        "example_count": 3,
        "would_write_files": ["prediction_examples.json", "prediction_examples_backfill.json", "concept_metrics.json"],
    }
    (root / "preflight.json").write_text(json.dumps(preflight), encoding="utf-8")

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[ArtifactSpec("Backfill preflight", "preflight.json", "plan", True)],
    )

    row = report["artifacts"][0]
    assert report["ok"] is True
    assert row["summary"]["dry_run"] is True
    assert row["summary"]["architecture"] == "convgru_pixel"
    assert row["summary"]["dataset_window_count"] == 17565
    assert row["summary"]["estimated_metric_batches"] == 1098
    assert row["summary"]["estimated_uncompressed_gib"] == 10.737
    assert row["summary"]["would_update_metrics"] is True
    assert row["summary"]["would_backfill_metrics"] is True
    assert row["summary"]["would_write_file_count"] == 3


def test_grid128_artifact_audit_validates_backfill_preflight_input_references(tmp_path):
    root = tmp_path / "root"
    run_dir = root / "runs" / "model"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "concept_checkpoint.pt"
    metrics = run_dir / "concept_metrics.json"
    arrays = root / "datasets" / "dynamics_arrays.npz"
    arrays.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    metrics.write_text("{}", encoding="utf-8")
    arrays.write_bytes(b"arrays")
    preflight = {
        "schema_version": 1,
        "dry_run": True,
        "would_backfill_metrics": True,
        "run_dir": "runs/model",
        "checkpoint_path": "runs/model/concept_checkpoint.pt",
        "metrics_path": "runs/model/concept_metrics.json",
        "dataset_array_path": "datasets/dynamics_arrays.npz",
    }
    (root / "preflight.json").write_text(json.dumps(preflight), encoding="utf-8")

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[ArtifactSpec("Backfill preflight", "preflight.json", "plan", True)],
    )

    row = report["artifacts"][0]
    assert report["ok"] is True
    assert row["summary"]["preflight_input_reference_count"] == 4
    assert row["summary"]["missing_preflight_input_reference_count"] == 0


def test_grid128_artifact_audit_flags_missing_backfill_preflight_input_references(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    preflight = {
        "schema_version": 1,
        "dry_run": True,
        "would_backfill_metrics": True,
        "run_dir": "missing/run",
        "checkpoint_path": "missing/run/concept_checkpoint.pt",
        "metrics_path": "missing/run/concept_metrics.json",
        "dataset_array_path": "missing/dynamics_arrays.npz",
    }
    (root / "preflight.json").write_text(json.dumps(preflight), encoding="utf-8")

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[ArtifactSpec("Backfill preflight", "preflight.json", "plan", True)],
    )

    row = report["artifacts"][0]
    assert report["ok"] is False
    assert report["status_counts"] == {"missing_reference": 1}
    assert row["status"] == "missing_reference"
    assert row["summary"]["preflight_input_reference_count"] == 4
    assert row["summary"]["missing_preflight_input_reference_count"] == 4
    assert row["preflight_reference_summary"]["missing_references"][0]["field"] == "run_dir"


def test_grid128_artifact_audit_validates_review_referenced_files(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "review.html").write_text("<html></html>\n", encoding="utf-8")
    (root / "panel.png").write_bytes(b"png")
    review = {
        "schema_version": 1,
        "html_path": "review.html",
        "models": [
            {
                "experiment_id": "model_a",
                "panel_png": "panel.png",
            }
        ],
    }
    (root / "review.json").write_text(json.dumps(review), encoding="utf-8")

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[ArtifactSpec("Review", "review.json", "review", True)],
    )

    row = report["artifacts"][0]
    assert report["ok"] is True
    assert row["summary"]["referenced_file_count"] == 2
    assert row["summary"]["missing_referenced_file_count"] == 0


def test_grid128_artifact_audit_reports_missing_review_references(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    review = {
        "schema_version": 1,
        "models": [
            {
                "experiment_id": "model_a",
                "panel_png": "missing_panel.png",
                "clip_panel_png": "missing_clip.png",
            }
        ],
    }
    (root / "review.json").write_text(json.dumps(review), encoding="utf-8")

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[ArtifactSpec("Review", "review.json", "review", True)],
    )

    row = report["artifacts"][0]
    assert report["ok"] is False
    assert report["status_counts"] == {"missing_reference": 1}
    assert row["status"] == "missing_reference"
    assert row["summary"]["referenced_file_count"] == 2
    assert row["summary"]["missing_referenced_file_count"] == 2
    assert row["reference_summary"]["missing_references"][0]["experiment_id"] == "model_a"


def test_audit_grid128_cli_fail_on_issues_returns_nonzero(monkeypatch, tmp_path):
    from neurobench.cli import dynamics as dynamics_cli

    def fake_audit(**kwargs):
        return {
            "markdown_path": str(tmp_path / "audit.md"),
            "json_path": str(tmp_path / "audit.json"),
            "status_counts": {"missing": 1},
            "ok": False,
        }

    monkeypatch.setattr(dynamics_cli, "build_grid128_artifact_audit", fake_audit)
    args = SimpleNamespace(root=tmp_path, out_dir=tmp_path / "audit", title="Audit", fail_on_issues=True)

    assert dynamics_cli.dynamics_audit_grid128_artifacts_command(args) == 2


def test_audit_grid128_cli_allows_issues_without_fail_flag(monkeypatch, tmp_path):
    from neurobench.cli import dynamics as dynamics_cli

    def fake_audit(**kwargs):
        return {
            "markdown_path": str(tmp_path / "audit.md"),
            "json_path": str(tmp_path / "audit.json"),
            "status_counts": {"missing_reference": 1},
            "ok": False,
        }

    monkeypatch.setattr(dynamics_cli, "build_grid128_artifact_audit", fake_audit)
    args = SimpleNamespace(root=tmp_path, out_dir=tmp_path / "audit", title="Audit", fail_on_issues=False)

    assert dynamics_cli.dynamics_audit_grid128_artifacts_command(args) == 0

def test_grid128_artifact_audit_validates_partial_report_artifact_links(tmp_path):
    root = tmp_path / "root"
    report_dir = root / "reports" / "grid128_sequence_1day_partial_report_v1"
    comparison_dir = root / "comparison"
    report_dir.mkdir(parents=True)
    comparison_dir.mkdir(parents=True)
    report_md = report_dir / "dynamics_experiment_report.md"
    dashboard = comparison_dir / "comparison_dashboard.html"
    manifest = comparison_dir / "comparison_manifest.json"
    intelligence = comparison_dir / "results_intelligence.json"
    report_json = report_dir / "dynamics_experiment_report.json"
    report_md.write_text("# Report", encoding="utf-8")
    dashboard.write_text("<html></html>", encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")
    intelligence.write_text("{}", encoding="utf-8")
    report_json.write_text(
        json.dumps(
            {
                "report_path": str(report_json),
                "markdown_path": str(report_md),
                "comparison_manifest_path": str(manifest),
                "results_intelligence_path": str(intelligence),
                "artifacts": {
                    "comparison_dashboard": str(dashboard),
                    "comparison_manifest": str(manifest),
                    "results_intelligence": str(intelligence),
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[
            ArtifactSpec(
                "Partial experiment report JSON",
                "reports/grid128_sequence_1day_partial_report_v1/dynamics_experiment_report.json",
                "report",
                True,
            )
        ],
    )

    row = report["artifacts"][0]
    assert report["ok"] is True
    assert row["summary"]["report_artifact_reference_count"] == 7
    assert row["summary"]["missing_report_artifact_reference_count"] == 0


def test_grid128_artifact_audit_flags_missing_partial_report_artifact_links(tmp_path):
    root = tmp_path / "root"
    report_dir = root / "reports" / "grid128_sequence_1day_partial_report_v1"
    report_dir.mkdir(parents=True)
    report_json = report_dir / "dynamics_experiment_report.json"
    report_json.write_text(
        json.dumps(
            {
                "report_path": str(report_json),
                "markdown_path": "missing/report.md",
                "comparison_manifest_path": "missing/comparison_manifest.json",
                "results_intelligence_path": "missing/results_intelligence.json",
                "artifacts": {"comparison_dashboard": "missing/comparison_dashboard.html"},
            }
        ),
        encoding="utf-8",
    )

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[
            ArtifactSpec(
                "Partial experiment report JSON",
                "reports/grid128_sequence_1day_partial_report_v1/dynamics_experiment_report.json",
                "report",
                True,
            )
        ],
    )

    row = report["artifacts"][0]
    assert report["ok"] is False
    assert report["status_counts"] == {"missing_reference": 1}
    assert row["status"] == "missing_reference"
    assert row["summary"]["report_artifact_reference_count"] == 5
    assert row["summary"]["missing_report_artifact_reference_count"] == 4
    assert row["report_reference_summary"]["missing_references"][0]["field"] == "markdown_path"


def test_grid128_artifact_audit_validates_report_active_summary(tmp_path):
    root = tmp_path / "root"
    report_dir = root / "reports" / "grid128_sequence_1day_partial_report_v1"
    sweep_dir = root / "sweeps" / "grid128_sequence_1day_v1"
    report_dir.mkdir(parents=True)
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "sweep_active.json").write_text(
        json.dumps({"status": "running", "index": 3, "experiment_id": "active_a", "dataset_key": "w8", "kind": "convgru_pixel"}),
        encoding="utf-8",
    )
    (sweep_dir / "sweep_progress.jsonl").write_text(
        json.dumps({"index": 2, "experiment_count": 5, "status": "completed", "experiment_id": "done"}) + "\n",
        encoding="utf-8",
    )
    (report_dir / "dynamics_experiment_report.json").write_text(
        json.dumps(
            {
                "active_sweep_summary": {
                    "status": "running",
                    "index": 3,
                    "experiment_id": "active_a",
                    "dataset_key": "w8",
                    "kind": "convgru_pixel",
                    "progress": "2 / 5",
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[
            ArtifactSpec(
                "Partial experiment report JSON",
                "reports/grid128_sequence_1day_partial_report_v1/dynamics_experiment_report.json",
                "report",
                True,
            )
        ],
    )

    row = report["artifacts"][0]
    assert report["ok"] is True
    assert row["summary"]["active_summary_matches_sweep"] is True
    assert row["summary"]["expected_active_progress"] == "2 / 5"


def test_grid128_artifact_audit_flags_stale_report_active_summary(tmp_path):
    root = tmp_path / "root"
    report_dir = root / "reports" / "grid128_sequence_1day_partial_report_v1"
    sweep_dir = root / "sweeps" / "grid128_sequence_1day_v1"
    report_dir.mkdir(parents=True)
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "sweep_active.json").write_text(
        json.dumps({"status": "running", "index": 4, "experiment_id": "active_new", "dataset_key": "w8", "kind": "convgru_pixel"}),
        encoding="utf-8",
    )
    (sweep_dir / "sweep_progress.jsonl").write_text(
        json.dumps({"index": 3, "experiment_count": 5, "status": "completed", "experiment_id": "done"}) + "\n",
        encoding="utf-8",
    )
    (report_dir / "dynamics_experiment_report.json").write_text(
        json.dumps(
            {
                "active_sweep_summary": {
                    "status": "running",
                    "index": 3,
                    "experiment_id": "active_old",
                    "dataset_key": "w8",
                    "kind": "convgru_pixel",
                    "progress": "2 / 5",
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[
            ArtifactSpec(
                "Partial experiment report JSON",
                "reports/grid128_sequence_1day_partial_report_v1/dynamics_experiment_report.json",
                "report",
                True,
            )
        ],
    )

    row = report["artifacts"][0]
    assert report["ok"] is False
    assert report["status_counts"] == {"stale_active_summary": 1}
    assert row["status"] == "stale_active_summary"
    assert row["summary"]["active_summary_matches_sweep"] is False
    assert set(row["summary"]["active_summary_mismatches"]) == {"index", "experiment_id", "progress"}


def test_grid128_artifact_audit_validates_report_embedded_audit_summary(tmp_path):
    root = tmp_path / "root"
    report_dir = root / "reports" / "grid128_sequence_1day_partial_report_v1"
    sweep_dir = root / "sweeps" / "grid128_sequence_1day_v1"
    comparison_dir = root / "comparison_grid128_sequence_1day_v1"
    report_dir.mkdir(parents=True)
    sweep_dir.mkdir(parents=True)
    comparison_dir.mkdir(parents=True)
    metrics = root / "metrics.json"
    examples = root / "examples.json"
    metrics.write_text("{}", encoding="utf-8")
    examples.write_text("{}", encoding="utf-8")
    (sweep_dir / "sweep_active.json").write_text(
        json.dumps({"status": "running", "index": 4, "experiment_id": "active_now"}),
        encoding="utf-8",
    )
    (sweep_dir / "sweep_progress.jsonl").write_text(
        json.dumps({"index": 3, "experiment_count": 5, "status": "completed", "experiment_id": "done"}) + "\n",
        encoding="utf-8",
    )
    (comparison_dir / "comparison_manifest.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "experiment_id": "done",
                        "metrics_path": str(metrics),
                        "prediction_examples_path": str(examples),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    audit_dir = root / "plans" / "grid128_artifact_audit_v1"
    audit_dir.mkdir(parents=True)
    (audit_dir / "grid128_artifact_audit.json").write_text(
        json.dumps({"artifact_count": 5, "status_counts": {"ok": 5}}),
        encoding="utf-8",
    )
    (report_dir / "dynamics_experiment_report.json").write_text(
        json.dumps(
            {
                "active_sweep_summary": {"status": "running", "index": 4, "experiment_id": "active_now", "progress": "3 / 5"},
                "artifact_audit_summary": {
                    "created_at": "2026-06-11T00:00:00+00:00",
                    "artifact_count": 5,
                    "status_counts": {"ok": 5},
                    "consistency_checks": [
                        {"check": "active_sweep_summary", "detail": "report=3 / 5 expected=3 / 5"},
                        {"check": "sweep_status_markdown", "detail": "report=3 / 5 expected=3 / 5"},
                    ],
                    "review_reference_counts": [
                        {
                            "label": "Comparison manifest",
                            "referenced_file_count": 2,
                            "missing_referenced_file_count": 0,
                            "referenced_metric_file_count": 1,
                            "referenced_prediction_file_count": 1,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[
            ArtifactSpec(
                "Partial experiment report JSON",
                "reports/grid128_sequence_1day_partial_report_v1/dynamics_experiment_report.json",
                "report",
                True,
            )
        ],
    )

    row = report["artifacts"][0]
    assert report["ok"] is True
    assert row["summary"]["embedded_audit_summary_matches_current_state"] is True
    assert row["summary"]["embedded_audit_artifact_count"] == 5
    assert row["summary"]["expected_audit_artifact_count"] == 5
    assert row["summary"]["expected_comparison_reference_count"] == 2


def test_grid128_artifact_audit_flags_stale_report_embedded_audit_summary(tmp_path):
    root = tmp_path / "root"
    report_dir = root / "reports" / "grid128_sequence_1day_partial_report_v1"
    sweep_dir = root / "sweeps" / "grid128_sequence_1day_v1"
    comparison_dir = root / "comparison_grid128_sequence_1day_v1"
    report_dir.mkdir(parents=True)
    sweep_dir.mkdir(parents=True)
    comparison_dir.mkdir(parents=True)
    metrics = root / "metrics.json"
    examples = root / "examples.json"
    metrics.write_text("{}", encoding="utf-8")
    examples.write_text("{}", encoding="utf-8")
    (sweep_dir / "sweep_active.json").write_text(
        json.dumps({"status": "running", "index": 4, "experiment_id": "active_now"}),
        encoding="utf-8",
    )
    (sweep_dir / "sweep_progress.jsonl").write_text(
        json.dumps({"index": 3, "experiment_count": 5, "status": "completed", "experiment_id": "done"}) + "\n",
        encoding="utf-8",
    )
    (comparison_dir / "comparison_manifest.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "experiment_id": "done",
                        "metrics_path": str(metrics),
                        "prediction_examples_path": str(examples),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    audit_dir = root / "plans" / "grid128_artifact_audit_v1"
    audit_dir.mkdir(parents=True)
    (audit_dir / "grid128_artifact_audit.json").write_text(
        json.dumps({"artifact_count": 5, "status_counts": {"ok": 5}}),
        encoding="utf-8",
    )
    (report_dir / "dynamics_experiment_report.json").write_text(
        json.dumps(
            {
                "active_sweep_summary": {"status": "running", "index": 4, "experiment_id": "active_now", "progress": "3 / 5"},
                "artifact_audit_summary": {
                    "created_at": "2026-06-11T00:00:00+00:00",
                    "artifact_count": 4,
                    "status_counts": {"ok": 4},
                    "consistency_checks": [
                        {"check": "active_sweep_summary", "detail": "report=2 / 5 expected=2 / 5"},
                        {"check": "sweep_status_markdown", "detail": "report=2 / 5 expected=2 / 5"},
                    ],
                    "review_reference_counts": [
                        {
                            "label": "Comparison manifest",
                            "referenced_file_count": 1,
                            "missing_referenced_file_count": 0,
                            "referenced_metric_file_count": 1,
                            "referenced_prediction_file_count": 0,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[
            ArtifactSpec(
                "Partial experiment report JSON",
                "reports/grid128_sequence_1day_partial_report_v1/dynamics_experiment_report.json",
                "report",
                True,
            )
        ],
    )

    row = report["artifacts"][0]
    assert report["ok"] is False
    assert report["status_counts"] == {"stale_embedded_audit_summary": 1}
    assert row["status"] == "stale_embedded_audit_summary"
    assert row["summary"]["embedded_audit_summary_matches_current_state"] is False
    assert set(row["summary"]["embedded_audit_summary_mismatches"]) == {
        "artifact_count",
        "status_counts",
        "active_sweep_summary",
        "comparison_manifest.referenced_file_count",
        "comparison_manifest.referenced_prediction_file_count",
    }


def test_grid128_artifact_audit_validates_stage_b_manifest_matches_plan(tmp_path):
    root = tmp_path / "root"
    plan_dir = root / "plans" / "grid128_sequence_stage_b_v1"
    plan_dir.mkdir(parents=True)
    progress = {"current_index": 8, "experiment_count": 10, "last_experiment_id": "done", "last_status": "completed"}
    (plan_dir / "next_sweep_manifest.json").write_text(
        json.dumps({"planned_experiment_count": 2, "progress_summary": progress, "experiments": [{"id": "a"}, {"id": "b"}]}),
        encoding="utf-8",
    )
    (plan_dir / "next_sweep_plan.json").write_text(
        json.dumps({"planned_experiment_count": 2, "progress_summary": progress, "manifest_path": str(plan_dir / "next_sweep_manifest.json")}),
        encoding="utf-8",
    )

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[
            ArtifactSpec("Stage B plan JSON", "plans/grid128_sequence_stage_b_v1/next_sweep_plan.json", "plan", True)
        ],
    )

    row = report["artifacts"][0]
    assert report["ok"] is True
    assert row["summary"]["stage_b_manifest_matches_plan"] is True
    assert row["summary"]["stage_b_manifest_experiment_count"] == 2


def test_grid128_artifact_audit_flags_stale_stage_b_manifest(tmp_path):
    root = tmp_path / "root"
    plan_dir = root / "plans" / "grid128_sequence_stage_b_v1"
    plan_dir.mkdir(parents=True)
    (plan_dir / "next_sweep_manifest.json").write_text(
        json.dumps(
            {
                "planned_experiment_count": 1,
                "progress_summary": {"current_index": 7, "experiment_count": 10, "last_experiment_id": "old", "last_status": "completed"},
                "experiments": [{"id": "a"}],
            }
        ),
        encoding="utf-8",
    )
    (plan_dir / "next_sweep_plan.json").write_text(
        json.dumps(
            {
                "planned_experiment_count": 2,
                "progress_summary": {"current_index": 8, "experiment_count": 10, "last_experiment_id": "new", "last_status": "completed"},
                "manifest_path": str(plan_dir / "next_sweep_manifest.json"),
            }
        ),
        encoding="utf-8",
    )

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[
            ArtifactSpec("Stage B plan JSON", "plans/grid128_sequence_stage_b_v1/next_sweep_plan.json", "plan", True)
        ],
    )

    row = report["artifacts"][0]
    assert report["ok"] is False
    assert report["status_counts"] == {"stale_stage_b_manifest": 1}
    assert row["status"] == "stale_stage_b_manifest"
    assert row["summary"]["stage_b_manifest_matches_plan"] is False
    assert set(row["summary"]["stage_b_manifest_mismatches"]) == {
        "planned_experiment_count",
        "manifest_experiment_count",
        "progress_summary.current_index",
        "progress_summary.last_experiment_id",
    }


def test_grid128_artifact_audit_validates_stage_b_source_progress_matches_sweep(tmp_path):
    root = tmp_path / "root"
    plan_dir = root / "plans" / "grid128_sequence_stage_b_v1"
    sweep_dir = root / "sweeps" / "grid128_sequence_1day_v1"
    plan_dir.mkdir(parents=True)
    sweep_dir.mkdir(parents=True)
    progress_rows = [
        {"index": 0, "experiment_count": 10, "status": "completed", "experiment_id": "first"},
        {"index": 1, "experiment_count": 10, "status": "failed", "experiment_id": "failed"},
        {"index": 2, "experiment_count": 10, "status": "completed", "experiment_id": "latest"},
    ]
    (sweep_dir / "sweep_progress.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in progress_rows),
        encoding="utf-8",
    )
    (plan_dir / "next_sweep_manifest.json").write_text(
        json.dumps({"planned_experiment_count": 1, "experiments": [{"id": "a"}]}),
        encoding="utf-8",
    )
    (plan_dir / "next_sweep_plan.json").write_text(
        json.dumps(
            {
                "planned_experiment_count": 1,
                "source_sweep_dir": str(sweep_dir),
                "progress_summary": {
                    "current_index": 2,
                    "experiment_count": 10,
                    "last_experiment_id": "latest",
                    "last_status": "completed",
                    "current_records": 3,
                    "current_status_counts": {"completed": 2, "failed": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[
            ArtifactSpec("Stage B plan JSON", "plans/grid128_sequence_stage_b_v1/next_sweep_plan.json", "plan", True)
        ],
    )

    row = report["artifacts"][0]
    assert report["ok"] is True
    assert row["summary"]["stage_b_source_progress_matches_sweep"] is True
    assert row["summary"]["stage_b_source_progress_index"] == 2
    assert row["summary"]["stage_b_source_progress_records"] == 3


def test_grid128_artifact_audit_flags_stale_stage_b_source_progress(tmp_path):
    root = tmp_path / "root"
    plan_dir = root / "plans" / "grid128_sequence_stage_b_v1"
    sweep_dir = root / "sweeps" / "grid128_sequence_1day_v1"
    plan_dir.mkdir(parents=True)
    sweep_dir.mkdir(parents=True)
    progress_rows = [
        {"index": 2, "experiment_count": 10, "status": "completed", "experiment_id": "old"},
        {"index": 3, "experiment_count": 10, "status": "completed", "experiment_id": "new"},
    ]
    (sweep_dir / "sweep_progress.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in progress_rows),
        encoding="utf-8",
    )
    (plan_dir / "next_sweep_manifest.json").write_text(
        json.dumps({"planned_experiment_count": 1, "experiments": [{"id": "a"}]}),
        encoding="utf-8",
    )
    (plan_dir / "next_sweep_plan.json").write_text(
        json.dumps(
            {
                "planned_experiment_count": 1,
                "source_sweep_dir": str(sweep_dir),
                "progress_summary": {
                    "current_index": 2,
                    "experiment_count": 10,
                    "last_experiment_id": "old",
                    "last_status": "completed",
                    "current_records": 1,
                    "current_status_counts": {"completed": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[
            ArtifactSpec("Stage B plan JSON", "plans/grid128_sequence_stage_b_v1/next_sweep_plan.json", "plan", True)
        ],
    )

    row = report["artifacts"][0]
    assert report["ok"] is False
    assert report["status_counts"] == {"stale_stage_b_source_progress": 1}
    assert row["status"] == "stale_stage_b_source_progress"
    assert row["summary"]["stage_b_source_progress_matches_sweep"] is False
    assert set(row["summary"]["stage_b_source_progress_mismatches"]) == {
        "current_index",
        "last_experiment_id",
        "current_records",
        "current_status_counts",
    }


def test_grid128_artifact_audit_validates_stage_b_dry_run_manifest_matches_source(tmp_path):
    root = tmp_path / "root"
    plan_dir = root / "plans" / "grid128_sequence_stage_b_v1"
    dry_run_dir = plan_dir / "stage_b_sweep"
    dry_run_dir.mkdir(parents=True)
    source_experiments = [{"experiment_id": "e0"}, {"experiment_id": "e1"}]
    (plan_dir / "next_sweep_manifest.json").write_text(
        json.dumps({"planned_experiment_count": 2, "experiments": source_experiments}),
        encoding="utf-8",
    )
    (dry_run_dir / "sweep_manifest.json").write_text(
        json.dumps({"experiment_count": 2, "experiments": source_experiments}),
        encoding="utf-8",
    )

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[
            ArtifactSpec(
                "Stage B dry-run manifest",
                "plans/grid128_sequence_stage_b_v1/stage_b_sweep/sweep_manifest.json",
                "plan",
                True,
            )
        ],
    )

    row = report["artifacts"][0]
    assert report["ok"] is True
    assert row["summary"]["stage_b_dry_run_matches_manifest"] is True
    assert row["summary"]["stage_b_dry_run_experiment_count"] == 2


def test_grid128_artifact_audit_flags_stale_stage_b_dry_run_manifest(tmp_path):
    root = tmp_path / "root"
    plan_dir = root / "plans" / "grid128_sequence_stage_b_v1"
    dry_run_dir = plan_dir / "stage_b_sweep"
    dry_run_dir.mkdir(parents=True)
    (plan_dir / "next_sweep_manifest.json").write_text(
        json.dumps({"planned_experiment_count": 2, "experiments": [{"experiment_id": "e0"}, {"experiment_id": "e1"}]}),
        encoding="utf-8",
    )
    (dry_run_dir / "sweep_manifest.json").write_text(
        json.dumps({"experiment_count": 1, "experiments": [{"experiment_id": "e1"}]}),
        encoding="utf-8",
    )

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[
            ArtifactSpec(
                "Stage B dry-run manifest",
                "plans/grid128_sequence_stage_b_v1/stage_b_sweep/sweep_manifest.json",
                "plan",
                True,
            )
        ],
    )

    row = report["artifacts"][0]
    assert report["ok"] is False
    assert report["status_counts"] == {"stale_stage_b_dry_run": 1}
    assert row["status"] == "stale_stage_b_dry_run"
    assert row["summary"]["stage_b_dry_run_matches_manifest"] is False
    assert set(row["summary"]["stage_b_dry_run_mismatches"]) == {
        "experiment_count",
        "experiment_list_length",
        "experiment_id_order",
    }


def test_grid128_artifact_audit_validates_sweep_status_markdown_matches_current_state(tmp_path):
    root = tmp_path / "root"
    sweep_dir = root / "sweeps" / "grid128_sequence_1day_v1"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "sweep_active.json").write_text(
        json.dumps({"status": "running", "index": 4, "experiment_id": "active_now"}),
        encoding="utf-8",
    )
    (sweep_dir / "sweep_progress.jsonl").write_text(
        json.dumps({"index": 3, "experiment_count": 5, "status": "completed", "experiment_id": "done"}) + "\n",
        encoding="utf-8",
    )
    (sweep_dir / "sweep_live_status.md").write_text(
        """# Sweep Live Status

Progress: `3` / `5`

## Active Spec

| Field | Value |
| --- | --- |
| status | running |
| index | 4 |
| experiment | active_now |
""",
        encoding="utf-8",
    )

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[ArtifactSpec("Sweep live status", "sweeps/grid128_sequence_1day_v1/sweep_live_status.md", "sweep_status")],
    )

    row = report["artifacts"][0]
    assert report["ok"] is True
    assert row["summary"]["sweep_status_matches_sweep"] is True
    assert row["summary"]["expected_progress"] == "3 / 5"


def test_grid128_artifact_audit_flags_stale_sweep_status_markdown(tmp_path):
    root = tmp_path / "root"
    sweep_dir = root / "sweeps" / "grid128_sequence_1day_v1"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "sweep_active.json").write_text(
        json.dumps({"status": "running", "index": 5, "experiment_id": "active_new"}),
        encoding="utf-8",
    )
    (sweep_dir / "sweep_progress.jsonl").write_text(
        json.dumps({"index": 4, "experiment_count": 6, "status": "completed", "experiment_id": "done"}) + "\n",
        encoding="utf-8",
    )
    (sweep_dir / "sweep_health_report.md").write_text(
        """# Sweep Health Report

Progress: `3` / `6`

## Active Spec

| Field | Value |
| --- | --- |
| status | running |
| index | 4 |
| experiment | active_old |
""",
        encoding="utf-8",
    )

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[ArtifactSpec("Sweep health report", "sweeps/grid128_sequence_1day_v1/sweep_health_report.md", "sweep_status")],
    )

    row = report["artifacts"][0]
    assert report["ok"] is False
    assert report["status_counts"] == {"stale_sweep_status": 1}
    assert row["status"] == "stale_sweep_status"
    assert row["summary"]["sweep_status_matches_sweep"] is False
    assert set(row["summary"]["sweep_status_mismatches"]) == {"progress", "active_index", "active_experiment_id"}




def test_grid128_artifact_audit_flags_stopped_sweep_health_stale_warning(tmp_path):
    root = tmp_path / "root"
    sweep_dir = root / "sweeps" / "grid128_sequence_1day_v1"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "sweep_active.json").write_text(
        json.dumps({"status": "completed", "index": 4, "experiment_id": "active_done"}),
        encoding="utf-8",
    )
    (sweep_dir / "sweep_progress.jsonl").write_text(
        json.dumps({"index": 4, "experiment_count": 6, "status": "completed", "experiment_id": "done"}) + "\n",
        encoding="utf-8",
    )
    (sweep_dir / "sweep_health_report.md").write_text(
        """# Sweep Health Report

Progress: `4` / `6`

## Active Spec

| Field | Value |
| --- | --- |
| status | completed |
| index | 4 |
| experiment | active_done |

## Health Flags

- Progress file is stale: last update was 500.0 minutes ago.
""",
        encoding="utf-8",
    )

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[ArtifactSpec("Sweep health report", "sweeps/grid128_sequence_1day_v1/sweep_health_report.md", "sweep_status")],
    )

    row = report["artifacts"][0]
    assert report["ok"] is False
    assert report["status_counts"] == {"stale_sweep_status": 1}
    assert row["status"] == "stale_sweep_status"
    assert set(row["summary"]["sweep_status_mismatches"]) == {"stopped_run_stale_warning", "stopped_run_marker"}

def test_grid128_artifact_audit_validates_comparison_manifest_references(tmp_path):
    root = tmp_path / "root"
    metrics = root / "runs" / "model" / "metrics.json"
    examples = root / "runs" / "model" / "prediction_examples.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("{}", encoding="utf-8")
    examples.write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "rows": [
            {
                "experiment_id": "model_a",
                "metrics_path": "runs/model/metrics.json",
                "prediction_examples_path": "runs/model/prediction_examples.json",
            }
        ],
    }
    (root / "comparison_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[ArtifactSpec("Comparison manifest", "comparison_manifest.json", "comparison", True)],
    )

    row = report["artifacts"][0]
    assert report["ok"] is True
    assert row["summary"]["referenced_file_count"] == 2
    assert row["summary"]["referenced_metric_file_count"] == 1
    assert row["summary"]["referenced_prediction_file_count"] == 1
    assert row["summary"]["missing_referenced_file_count"] == 0


def test_grid128_artifact_audit_flags_missing_comparison_manifest_references(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    manifest = {
        "schema_version": 1,
        "rows": [
            {
                "experiment_id": "model_a",
                "metrics_path": "missing/metrics.json",
                "prediction_examples_path": "missing/prediction_examples.json",
            }
        ],
    }
    (root / "comparison_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[ArtifactSpec("Comparison manifest", "comparison_manifest.json", "comparison", True)],
    )

    row = report["artifacts"][0]
    assert report["ok"] is False
    assert report["status_counts"] == {"missing_reference": 1}
    assert row["status"] == "missing_reference"
    assert row["summary"]["missing_referenced_file_count"] == 2
    assert row["manifest_reference_summary"]["missing_references"][0]["experiment_id"] == "model_a"



def test_grid128_artifact_audit_summarizes_shared_horizon_baseline_comparison(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    comparison = {
        "schema_version": 1,
        "title": "Shared-Horizon Baseline Comparison",
        "created_at": "2026-06-11T00:00:00+00:00",
        "runs": [{"label": "shared_linear"}, {"label": "shared_gru"}],
        "horizon_summary": [{"dataset_key": "w8_s1_h2"}, {"dataset_key": "w8_s1_h5"}],
        "active_cell_warnings": ["shared_linear is negative on active cells"],
        "best_overall": {
            "label": "shared_linear",
            "improvement_over_persistence_mse": 0.0005,
        },
    }
    (root / "shared_horizon_baseline_comparison.json").write_text(json.dumps(comparison), encoding="utf-8")

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[
            ArtifactSpec(
                "Shared-horizon baseline comparison JSON",
                "shared_horizon_baseline_comparison.json",
                "report",
                True,
            )
        ],
    )

    row = report["artifacts"][0]
    assert report["ok"] is True
    assert row["summary"]["run_count"] == 2
    assert row["summary"]["horizon_summary_count"] == 2
    assert row["summary"]["active_cell_warning_count"] == 1
    assert row["summary"]["best_overall_label"] == "shared_linear"
    assert row["summary"]["best_overall_improvement_over_persistence_mse"] == 0.0005

def test_grid128_artifact_audit_validates_review_count_fields(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "review.html").write_text("<html></html>", encoding="utf-8")
    (root / "panel_a.png").write_bytes(b"png")
    (root / "clip_a.png").write_bytes(b"png")
    review = {
        "schema_version": 1,
        "html_path": "review.html",
        "selected_model_count": 1,
        "missing_visual_example_count": 0,
        "temporal_clip_model_count": 1,
        "models": [{"experiment_id": "model_a", "panel_png": "panel_a.png", "clip_panel_png": "clip_a.png"}],
        "missing_visual_example_rows": [],
    }
    (root / "video_error_review.json").write_text(json.dumps(review), encoding="utf-8")

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[ArtifactSpec("Review", "video_error_review.json", "review", True)],
    )

    row = report["artifacts"][0]
    assert report["ok"] is True
    assert row["summary"]["review_counts_match"] is True
    assert row["summary"]["actual_model_count"] == 1
    assert row["summary"]["actual_temporal_clip_model_count"] == 1


def test_grid128_artifact_audit_flags_stale_review_count_fields(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "panel_a.png").write_bytes(b"png")
    review = {
        "schema_version": 1,
        "selected_model_count": 2,
        "missing_visual_example_count": 0,
        "temporal_clip_model_count": 1,
        "models": [{"experiment_id": "model_a", "panel_png": "panel_a.png"}],
        "missing_visual_example_rows": [{"experiment_id": "missing_model"}],
    }
    (root / "video_error_review.json").write_text(json.dumps(review), encoding="utf-8")

    report = build_grid128_artifact_audit(
        root=root,
        out_dir=tmp_path / "audit",
        artifact_specs=[ArtifactSpec("Review", "video_error_review.json", "review", True)],
    )

    row = report["artifacts"][0]
    assert report["ok"] is False
    assert report["status_counts"] == {"stale_review_counts": 1}
    assert row["status"] == "stale_review_counts"
    assert set(row["summary"]["review_count_mismatches"]) == {
        "selected_model_count",
        "missing_visual_example_count",
        "temporal_clip_model_count",
    }
