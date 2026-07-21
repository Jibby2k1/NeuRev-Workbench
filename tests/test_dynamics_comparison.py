import json
from pathlib import Path

from neurobench.dynamics.comparison import build_comparison_dashboard


def test_build_comparison_dashboard_writes_manifest_and_html(tmp_path):
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    (sweep / "sweep_manifest.json").write_text(json.dumps({"profile": "upgrade", "datasets": {"demo_h50": {"window_frames": 8}}, "experiments": [{"experiment_id": "failed_gru", "kind": "latent_gru", "dataset_key": "demo_h50", "seed": 7, "params": {"prediction_target": "delta", "hidden_dim": 256, "batch_size": 64, "hyperparameter_summary": "target=delta, hd=256, batch=64"}}]}), encoding="utf-8")
    exp = sweep / "convgru_demo"
    metrics_dir = exp / "convgru_pixel_mse"
    metrics_dir.mkdir(parents=True)
    (exp / "experiment_config.json").write_text(
        json.dumps(
            {
                "experiment_id": "convgru_demo",
                "kind": "convgru_pixel",
                "dataset_key": "demo_h50",
                "seed": 7,
                "params": {"variant": "convgru_pixel_mse", "loss_mode": "frame_mse", "hidden_channels": 16, "grid_size": 128, "grid_pooling": "max_intensity"},
            }
        ),
        encoding="utf-8",
    )
    (metrics_dir / "concept_metrics.json").write_text(
        json.dumps(
            {
                "objective": "convgru_pixel_frame_mse",
                "model_kind": "pixel_convgru_residual",
                "model_family": "pixel_convgru",
                "loss_mode": "frame_mse",
                "val_decoded_prediction_mse": 0.09,
                "val_persistence_mse": 0.10,
                "val_improvement_over_persistence_mse": 0.01,
                "test_decoded_prediction_mse": 0.08,
                "test_persistence_mse": 0.10,
                "test_improvement_over_persistence_mse": 0.02,
                "prediction_examples_path": str(metrics_dir / "review_artifacts" / "prediction_examples.json"),
                "split_metrics": {
                    "test": {
                        "per_video": {
                            "video_good_left": {
                                "window_count": 2,
                                "decoded_prediction_mse": 0.05,
                                "persistence_mse": 0.10,
                                "improvement_over_persistence_mse": 0.05,
                            },
                            "video_bad_right": {
                                "window_count": 2,
                                "decoded_prediction_mse": 0.12,
                                "persistence_mse": 0.10,
                                "improvement_over_persistence_mse": -0.02,
                            },
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    baseline = sweep / "baseline_demo"
    baseline.mkdir()
    (baseline / "experiment_config.json").write_text(
        json.dumps({"experiment_id": "baseline_demo", "kind": "array_baseline", "dataset_key": "demo_h50", "seed": 0, "params": {"baseline_name": "linear_extrapolation"}}),
        encoding="utf-8",
    )
    (baseline / "array_baseline_metrics.json").write_text(
        json.dumps(
            {
                "objective": "array_linear_extrapolation_baseline",
                "model_kind": "array_baseline",
                "model_family": "array_baseline",
                "baseline_name": "linear_extrapolation",
                "val_decoded_prediction_mse": 0.11,
                "val_persistence_mse": 0.10,
                "val_improvement_over_persistence_mse": -0.01,
                "test_decoded_prediction_mse": 0.12,
                "test_persistence_mse": 0.10,
                "test_improvement_over_persistence_mse": -0.02,
            }
        ),
        encoding="utf-8",
    )

    charts = sweep / "visuals" / "charts"
    charts.mkdir(parents=True)
    for name in ["demo_best_8_left_intensity.mp4", "demo_best_8_left_motion.mp4"]:
        (charts / name).write_bytes(b"fake-mp4")
    (charts / "original_vs_reconstruction_selector.json").write_text(
        json.dumps(
            {
                "panel_order": ["target_frame", "model_prediction_shifted_by_horizon", "persistence_prediction_shifted_by_horizon", "lag_compensated_absolute_error"],
                "segment_selection": "highest motion segment",
                "models": [{"tag": "best", "label": "Best", "experiment_id": "convgru_demo", "dataset_key": "demo_h50"}],
                "options": [
                    {
                        "model_tag": "best",
                        "model_label": "Best",
                        "experiment_id": "convgru_demo",
                        "dataset_key": "demo_h50",
                        "video_id": "8 left",
                        "split": "test",
                        "target_offset_raw_frames": 50,
                        "improvement_over_persistence_mse": 0.02,
                        "intensity_file": "demo_best_8_left_intensity.mp4",
                        "motion_file": "demo_best_8_left_motion.mp4",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (sweep / "sweep_progress_batch64_oom.jsonl").write_text(
        json.dumps(
            {
                "index": 3,
                "experiment_count": 4,
                "experiment_id": "failed_gru",
                "kind": "latent_gru",
                "dataset_key": "demo_h50",
                "seed": 7,
                "status": "failed",
                "error": "torch.cuda.OutOfMemoryError: CUDA out of memory",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (sweep / "sweep_progress.jsonl").write_text(
        json.dumps(
            {
                "index": 1,
                "experiment_count": 4,
                "experiment_id": "convgru_demo",
                "kind": "convgru_pixel",
                "dataset_key": "demo_h50",
                "seed": 7,
                "status": "completed",
                "elapsed_seconds": 123.5,
                "started_at": "2026-01-01T00:00:00+00:00",
                "finished_at": "2026-01-01T00:02:03+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_comparison_dashboard(sweep_dirs=[sweep], out_dir=tmp_path / "comparison")
    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    html = Path(summary["html_path"]).read_text(encoding="utf-8")
    intelligence = json.loads(Path(summary["intelligence_path"]).read_text(encoding="utf-8"))
    intelligence_md = Path(summary["intelligence_md_path"]).read_text(encoding="utf-8")

    assert summary["row_count"] == 2
    assert summary["video_collection_count"] == 1
    assert summary["failure_count"] == 1
    assert summary["positive_test_count"] == 1
    assert manifest["selected_models"][0]["experiment_id"] == "convgru_demo"
    assert manifest["selected_models"][0]["prediction_examples_path"].endswith("review_artifacts/prediction_examples.json")
    assert manifest["selected_models"][0]["elapsed_seconds"] == 123.5
    assert manifest["intelligence"]["runtime_summary"]["available"] is True
    assert manifest["intelligence"]["runtime_summary"]["by_family"]["pixel_convgru"]["median_seconds"] == 123.5
    video_summary = manifest["selected_models"][0]["video_error_summary"]["test"]
    assert video_summary["video_count"] == 2
    assert video_summary["best_videos"][0]["video_id"] == "video_good_left"
    assert video_summary["worst_videos"][0]["video_id"] == "video_bad_right"
    assert video_summary["best_videos"][0]["video_label"] == "left"
    assert video_summary["worst_videos"][0]["video_label"] == "right"
    assert video_summary["label_summary"][0]["label"] == "left"
    assert video_summary["label_summary"][0]["improvement_over_persistence_mse"] == 0.05
    assert video_summary["label_summary"][1]["label"] == "right"
    assert video_summary["label_summary"][1]["improvement_over_persistence_mse"] == -0.02
    assert manifest["intelligence"]["leaderboards"]["test"][0]["video_error_summary"]["test"]["best_videos"][0]["video_id"] == "video_good_left"
    assert manifest["intelligence"]["best_by_family"]["test"]["pixel_convgru"]["experiment_id"] == "convgru_demo"
    assert manifest["intelligence"]["target_comparison"]["test"]["unspecified"]["count"] == 1
    assert intelligence["failure_summary"]["by_class"] == {"cuda_oom": 1}
    assert intelligence["failures"][0]["hyperparameter_summary"] == "target=delta, hd=256, batch=64"
    assert "Results Intelligence" in intelligence_md
    assert "hc=16" in manifest["selected_models"][0]["hyperparameter_summary"]
    assert "grid=128" in manifest["selected_models"][0]["hyperparameter_summary"]
    assert manifest["video_collections"][0]["options"][0]["intensity_src"].endswith("demo_best_8_left_intensity.mp4")
    assert "inputVideoFilter" in html
    assert "HParams" in html
    assert "hyperparameter_summary" in html
    assert "JSON.stringify(row.params" in html
    assert "Video Comparison" in html
    assert "videoGrid" in html
    assert "clipSet" in html
    assert "Held-out-first" in html
    assert "Results Intelligence" in html
    assert "Family Winners" in html
    assert "Failure Heatmap" in html
    assert "Runtime" in html
    assert "renderIntelRuntime" in html
    assert "payload.intelligence" in html
    assert "renderVideoEvidence" in html
    assert "label_summary" in html
