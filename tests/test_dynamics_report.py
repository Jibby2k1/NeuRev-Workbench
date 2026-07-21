import json
from pathlib import Path

from neurobench.dynamics.report import build_dynamics_experiment_report


def _write_experiment(sweep: Path, *, experiment_id: str, kind: str, family: str, dataset_key: str, improve: float, baseline_name: str | None = None, prediction_target: str | None = None, params: dict | None = None) -> None:
    exp = sweep / experiment_id
    exp.mkdir(parents=True)
    payload = {
        "experiment_id": experiment_id,
        "kind": kind,
        "dataset_key": dataset_key,
        "seed": 0,
        "params": params or {},
    }
    if baseline_name:
        payload["params"]["baseline_name"] = baseline_name
    if prediction_target:
        payload["params"]["prediction_target"] = prediction_target
    (exp / "experiment_config.json").write_text(json.dumps(payload), encoding="utf-8")
    metrics = {
        "objective": f"{experiment_id}_objective",
        "model_kind": kind,
        "model_family": family,
        "baseline_name": baseline_name,
        "prediction_target": prediction_target,
        "decoded_prediction_mse": 1.0 - improve,
        "persistence_mse": 1.0,
        "improvement_over_persistence_mse": improve,
        "val_decoded_prediction_mse": 1.0 - improve,
        "val_persistence_mse": 1.0,
        "val_improvement_over_persistence_mse": improve,
        "test_decoded_prediction_mse": 1.0 - improve,
        "test_persistence_mse": 1.0,
        "test_improvement_over_persistence_mse": improve,
        "test_active_cell_improvement_over_persistence_mse": improve * 0.5,
        "test_top_activity_improvement_over_persistence_mse": improve * 0.4,
        "test_high_change_improvement_over_persistence_mse": improve * 0.3,
        "test_active_cell_mse": 0.5,
        "test_active_cell_persistence_mse": 0.5 + improve * 0.5,
        "split_metrics": {
            "test": {
                "per_video": {
                    f"{experiment_id}_left_good_video": {
                        "window_count": 2,
                        "decoded_prediction_mse": 1.0 - improve,
                        "persistence_mse": 1.0,
                        "improvement_over_persistence_mse": improve,
                    },
                    f"{experiment_id}_right_bad_video": {
                        "window_count": 2,
                        "decoded_prediction_mse": 1.0,
                        "persistence_mse": 1.0 - improve,
                        "improvement_over_persistence_mse": -improve,
                    },
                }
            }
        },
    }
    if kind == "array_baseline":
        (exp / "array_baseline_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    elif kind == "latent_gru":
        (exp / "latent_rnn_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    else:
        raise AssertionError(kind)


def test_build_dynamics_experiment_report_combines_health_intelligence_and_kinetics(tmp_path):
    sweep = tmp_path / "active"
    sweep.mkdir()
    manifest = {
        "profile": "unit_profile",
        "experiment_count": 3,
        "datasets": {
            "w8_s1_h2": {
                "windowing": {"window_frames": 8, "prediction_horizon_frames": 2, "prediction_horizon_sec": 0.04, "effective_frame_rate_hz": 50.0},
                "splits": {"train_video_ids": ["a"], "val_video_ids": ["b"], "test_video_ids": ["c"]},
            }
        },
        "experiments": [
            {"experiment_id": "learned", "kind": "latent_gru", "dataset_key": "w8_s1_h2", "seed": 7, "params": {"prediction_target": "delta", "hyperparameter_summary": "target=delta"}},
            {"experiment_id": "failed", "kind": "latent_gru", "dataset_key": "w8_s1_h2", "seed": 7, "params": {"prediction_target": "delta"}},
        ],
    }
    (sweep / "sweep_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (sweep / "sweep_progress.jsonl").write_text(
        json.dumps({"index": 1, "experiment_count": 3, "experiment_id": "learned", "kind": "latent_gru", "dataset_key": "w8_s1_h2", "status": "completed", "elapsed_seconds": 42.0})
        + "\n"
        + json.dumps({"index": 2, "experiment_count": 3, "experiment_id": "failed", "kind": "latent_gru", "dataset_key": "w8_s1_h2", "status": "failed", "error": "CUDA out of memory"})
        + "\n",
        encoding="utf-8",
    )
    (sweep / "sweep_active.json").write_text(
        json.dumps(
            {
                "status": "running",
                "index": 3,
                "experiment_id": "active_unit",
                "dataset_key": "w8_s1_h2",
                "kind": "latent_gru",
                "updated_at": "2026-06-10T00:03:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (sweep / "sweep_live_status.md").write_text("# Sweep Live Status\n", encoding="utf-8")
    _write_experiment(sweep, experiment_id="learned", kind="latent_gru", family="latent_gru", dataset_key="w8_s1_h2", improve=0.05, prediction_target="delta", params={"hyperparameter_summary": "target=delta, lr=1e-3, hd=32", "learning_rate": 0.001, "hidden_dim": 32, "hyperparameter_group": "latent_recurrent"})

    kinetics = tmp_path / "kinetics"
    kinetics.mkdir()
    (kinetics / "sweep_manifest.json").write_text(json.dumps({"profile": "kinetics_baselines", "datasets": manifest["datasets"], "experiments": []}), encoding="utf-8")
    _write_experiment(kinetics, experiment_id="kinetics_decay", kind="array_baseline", family="kinetics_baseline", dataset_key="w8_s1_h2", improve=0.08, baseline_name="exponential_decay_10hz", params={"hyperparameter_summary": "reaction=10hz"})
    _write_experiment(kinetics, experiment_id="moving", kind="array_baseline", family="array_baseline", dataset_key="w8_s1_h2", improve=0.03, baseline_name="moving_average")
    review_dir = tmp_path / "reviews" / "best_test_review"
    review_dir.mkdir(parents=True)
    (review_dir / "video_error_review.html").write_text("<html></html>", encoding="utf-8")
    (review_dir / "video_error_review.json").write_text(
        json.dumps(
            {
                "title": "Unit Best-Test Review",
                "selection_mode": "best_test",
                "split": "test",
                "selected_model_count": 2,
                "temporal_clip_model_count": 1,
                "missing_visual_example_count": 1,
                "html_path": str(review_dir / "video_error_review.html"),
                "models": [{"experiment_id": "learned"}, {"experiment_id": "kinetics_decay"}],
                "missing_visual_example_rows": [{"experiment_id": "missing_visual"}],
                "limitations": ["unit limitation"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stage_b_dir = tmp_path / "plans" / "stage_b"
    stage_b_dir.mkdir(parents=True)
    (stage_b_dir / "next_sweep_plan.json").write_text(
        json.dumps(
            {
                "created_at": "2026-06-10T00:00:00+00:00",
                "planned_experiment_count": 5,
                "selection_counts": {"latent_gru": 2, "latent_transformer": 3},
                "dataset_counts": {"w8_s1_h2": 3, "w8_s1_h5": 2},
                "target_counts": {"delta": 5},
                "progress_summary": {"current_index": 2, "experiment_count": 3, "last_experiment_id": "learned"},
                "suggested_command": "python -m neurobench.dynamics.overnight_sweep --manifest next.json",
                "summary_path": str(stage_b_dir / "next_sweep_plan.json"),
                "markdown_path": str(stage_b_dir / "next_sweep_plan.md"),
                "manifest_path": str(stage_b_dir / "next_sweep_manifest.json"),
                "source_sweep_dir": str(sweep),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rescue_dir = tmp_path / "plans" / "active_cell_rescue"
    rescue_dir.mkdir(parents=True)
    (rescue_dir / "active_cell_rescue_plan.json").write_text(
        json.dumps(
            {
                "title": "Unit Active-Cell Rescue Plan",
                "created_at": "2026-06-10T00:01:00+00:00",
                "best_overall_label": "shared_linear",
                "best_overall_improvement_over_persistence_mse": 0.0123,
                "active_cell_warning_count": 4,
                "completed_shared_neural_count": 1,
                "pending_shared_neural_count": 7,
                "recommended_candidates": [
                    {
                        "config_id": "unit_transformer",
                        "model_family": "multi_horizon_latent_transformer",
                        "priority": "transformer_candidate",
                        "rationale": "unit rationale",
                        "command": "python -m neurobench.cli.main dynamics train-shared-transformer-horizons",
                        "out_dir": str(rescue_dir / "unit_transformer"),
                    }
                ],
                "recommendations": ["run unit_transformer next"],
                "plan_path": str(rescue_dir / "active_cell_rescue_plan.json"),
                "markdown_path": str(rescue_dir / "active_cell_rescue_plan.md"),
                "grid_status": str(rescue_dir / "grid_status.json"),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audit_dir = tmp_path / "plans" / "artifact_audit"
    audit_dir.mkdir(parents=True)
    (audit_dir / "grid128_artifact_audit.json").write_text(
        json.dumps(
            {
                "created_at": "2026-06-10T00:02:00+00:00",
                "ok": True,
                "artifact_count": 2,
                "status_counts": {"ok": 2},
                "markdown_path": str(audit_dir / "grid128_artifact_audit.md"),
                "json_path": str(audit_dir / "grid128_artifact_audit.json"),
                "artifacts": [
                    {
                        "label": "sweep_live_status",
                        "status": "ok",
                        "relative_path": "sweeps/unit/sweep_live_status.md",
                        "summary": {
                            "sweep_status_check": "compared",
                            "sweep_status_matches_sweep": True,
                            "report_progress": "2 / 3",
                            "expected_progress": "2 / 3",
                        },
                    },
                    {
                        "label": "partial_report",
                        "status": "ok",
                        "relative_path": "reports/unit/dynamics_experiment_report.md",
                        "summary": {
                            "active_summary_check": "compared",
                            "active_summary_matches_sweep": True,
                            "report_active_progress": "2 / 3",
                            "expected_active_progress": "2 / 3",
                            "embedded_audit_check": "compared",
                            "embedded_audit_summary_matches_current_state": True,
                            "embedded_audit_expected_progress": "2 / 3",
                            "embedded_audit_comparison_reference_count": 7,
                            "expected_comparison_reference_count": 7,
                        },
                    },
                    {
                        "label": "stage_b_plan",
                        "status": "ok",
                        "relative_path": "plans/stage_b/next_sweep_plan.json",
                        "summary": {
                            "stage_b_manifest_check": "compared",
                            "stage_b_manifest_matches_plan": True,
                            "stage_b_plan_count": 5,
                            "stage_b_manifest_count": 5,
                            "stage_b_manifest_experiment_count": 5,
                            "stage_b_source_progress_check": "compared",
                            "stage_b_source_progress_matches_sweep": True,
                            "stage_b_plan_progress_index": 2,
                            "stage_b_source_progress_index": 2,
                            "stage_b_source_progress_records": 3,
                        },
                    },
                    {
                        "label": "stage_b_dry_run",
                        "status": "ok",
                        "relative_path": "plans/stage_b/stage_b_sweep/sweep_manifest.json",
                        "summary": {
                            "stage_b_dry_run_check": "compared",
                            "stage_b_dry_run_matches_manifest": True,
                            "stage_b_source_manifest_experiment_count": 5,
                            "stage_b_dry_run_experiment_count": 5,
                        },
                    },
                    {
                        "label": "comparison_manifest",
                        "status": "ok",
                        "relative_path": "comparison/comparison_manifest.json",
                        "summary": {
                            "referenced_file_count": 7,
                            "missing_referenced_file_count": 0,
                            "referenced_metric_file_count": 5,
                            "referenced_prediction_file_count": 2,
                        },
                    },
                    {
                        "label": "best_test_review",
                        "status": "ok",
                        "relative_path": "reviews/best_test_review/video_error_review.json",
                        "summary": {"referenced_file_count": 3, "missing_referenced_file_count": 0},
                    },
                    {
                        "label": "learned_leader_preflight",
                        "status": "ok",
                        "relative_path": "plans/backfill/learned_preflight.json",
                        "summary": {
                            "referenced_file_count": 4,
                            "missing_referenced_file_count": 0,
                            "preflight_input_reference_count": 4,
                            "missing_preflight_input_reference_count": 0,
                        },
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_dynamics_experiment_report(
        sweep_dirs=[sweep, kinetics],
        comparison_dir=tmp_path / "comparison",
        out_dir=tmp_path / "report",
        refresh_dashboard=True,
    )
    markdown = Path(report["markdown_path"]).read_text(encoding="utf-8")

    assert report["run_summary"]["completed_metric_rows"] == 3
    assert report["run_summary"]["failure_count"] == 1
    assert report["dataset_summary"][0]["prediction_horizon_frames"] == 2
    assert report["baseline_comparison"]["best_kinetics_baseline"]["experiment_id"] == "kinetics_decay"
    assert report["baseline_comparison"]["best_learned_model"]["experiment_id"] == "learned"
    assert report["best_models"]["test_learned"]["experiment_id"] == "learned"
    assert report["baseline_comparison"]["learned_minus_kinetics_improvement"] < 0
    assert report["runtime_summary"]["available"] is True
    assert report["runtime_summary"]["by_family"]["latent_gru"]["median_seconds"] == 42.0
    assert report["active_sweep_summary"]["available"] is True
    assert report["active_sweep_summary"]["experiment_id"] == "active_unit"
    assert report["active_sweep_summary"]["status"] == "running"
    assert report["active_sweep_summary"]["progress"] == "2 / 3"
    assert report["active_sweep_summary"]["live_status_path"].endswith("sweep_live_status.md")
    assert report["hyperparameter_findings"]["available"] is True
    assert report["hyperparameter_findings"]["dimensions"]["model_family"]["groups"][0]["value"] == "kinetics_baseline"
    assert report["hyperparameter_findings"]["dimensions"]["learning_rate"]["groups"][0]["value"] == "0.001"
    assert report["active_error_summary"]["available"] is True
    assert report["video_error_summary"]["available"] is True
    assert report["visual_review_summary"]["available"] is True
    assert report["visual_review_summary"]["reviews"][0]["title"] == "Unit Best-Test Review"
    assert report["visual_review_summary"]["reviews"][0]["selection_mode"] == "best_test"
    assert report["visual_review_summary"]["reviews"][0]["model_ids"] == ["learned", "kinetics_decay"]
    assert report["visual_review_summary"]["reviews"][0]["temporal_clip_model_count"] == 1
    assert report["visual_review_summary"]["reviews"][0]["missing_model_ids"] == ["missing_visual"]
    assert report["next_sweep_recommendation"]["available"] is True
    assert report["next_sweep_recommendation"]["stage_b_plan"]["planned_experiment_count"] == 5
    assert report["next_sweep_recommendation"]["stage_b_plan"]["selection_counts"]["latent_transformer"] == 3
    assert report["next_sweep_recommendation"]["active_cell_rescue"]["next_candidate"]["config_id"] == "unit_transformer"
    assert report["artifact_audit_summary"]["available"] is True
    assert report["artifact_audit_summary"]["ok"] is True
    assert report["artifact_audit_summary"]["artifact_count"] == 2
    reference_rows = {row["label"]: row for row in report["artifact_audit_summary"]["review_reference_counts"]}
    assert reference_rows["comparison_manifest"]["referenced_file_count"] == 7
    assert reference_rows["comparison_manifest"]["referenced_metric_file_count"] == 5
    assert reference_rows["comparison_manifest"]["referenced_prediction_file_count"] == 2
    assert reference_rows["best_test_review"]["referenced_file_count"] == 3
    assert reference_rows["learned_leader_preflight"]["referenced_input_file_count"] == 4
    checks = {row["check"]: row for row in report["artifact_audit_summary"]["consistency_checks"]}
    assert checks["sweep_status_markdown"]["detail"] == "report=2 / 3 expected=2 / 3"
    assert checks["active_sweep_summary"]["ok"] is True
    assert checks["embedded_artifact_audit_summary"]["detail"] == "progress=2 / 3 references=7/7"
    assert checks["stage_b_plan_manifest"]["detail"] == "plan=5 manifest=5 experiments=5"
    assert checks["stage_b_source_progress"]["detail"] == "plan=2 source=2 records=3"
    assert checks["stage_b_dry_run_manifest"]["detail"] == "source=5 dry_run=5"
    assert report["video_error_summary"]["rows"][0]["best_videos"][0]["video_id"] == "kinetics_decay_left_good_video"
    assert report["video_error_summary"]["rows"][0]["worst_videos"][0]["video_id"] == "kinetics_decay_right_bad_video"
    assert report["video_error_summary"]["rows"][0]["label_summary"][0]["label"] == "left"
    assert report["video_error_summary"]["rows"][0]["label_summary"][1]["label"] == "right"
    assert report["active_error_summary"]["best_kinetics"]["experiment_id"] == "kinetics_decay"
    assert report["active_error_summary"]["top_active_rows"][0]["experiment_id"] == "kinetics_decay"
    assert report["active_error_summary"]["active_global_tradeoff"]["same_experiment"] is False
    assert report["active_error_summary"]["active_global_tradeoff"]["best_active_experiment_id"] == "kinetics_decay"
    assert report["active_error_summary"]["active_global_tradeoff"]["best_global_experiment_id"] == "learned"
    assert report["active_error_summary"]["learned_minus_kinetics_active_improvement"] < 0
    assert "Do not claim learned dynamics beat kinetics-aware baselines yet" in " ".join(report["recommendations"])
    assert "# Grid Dynamics Experiment Report" in markdown
    assert "Persistence And Kinetics Baseline Comparison" in markdown
    assert "Active Sweep Liveness" in markdown
    assert "active_unit" in markdown
    assert "sweep_live_status.md" in markdown
    assert "Runtime Summary" in markdown
    assert "42s" in markdown
    assert "Hyperparameter Findings" in markdown
    assert "Learning rate" in markdown
    assert "0.001" in markdown
    assert "Per-Video Evidence" in markdown
    assert "kinetics_decay_left_good_video" in markdown
    assert "left 0.08" in markdown
    assert "Visual Examples" in markdown
    assert "Unit Best-Test Review" in markdown
    assert "best_test" in markdown
    assert "`learned`, `kinetics_decay`" in markdown
    assert "missing_visual" in markdown
    assert "Artifact Integrity Audit" in markdown
    assert "Embedded audit snapshot" in markdown
    assert "Snapshot generated" in markdown
    assert "sweep_status_markdown" in markdown
    assert "embedded_artifact_audit_summary" in markdown
    assert "stage_b_dry_run_manifest" in markdown
    assert "| comparison_manifest | ok | 7 | 5 | 2 | 0 | 0 |" in markdown
    assert "| learned_leader_preflight | ok | 4 | 0 | 0 | 4 | 0 |" in markdown
    assert "Audit OK: `True`" in markdown
    assert "best_test_review" in markdown
    assert "Recommended Next Sweep" in markdown
    assert "Stage B Sweep Plan" in markdown
    assert "latent_transformer=3" in markdown
    assert "unit_transformer" in markdown
    assert "train-shared-transformer-horizons" in markdown
    assert "Active-Cell Error Check" in markdown
    assert "Top active-cell rows" in markdown
    assert "Global improve" in markdown
    assert "Failure Analysis" in markdown
