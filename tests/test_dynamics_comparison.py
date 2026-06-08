import json
from pathlib import Path

from neurobench.dynamics.comparison import build_comparison_dashboard


def test_build_comparison_dashboard_writes_manifest_and_html(tmp_path):
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    (sweep / "sweep_manifest.json").write_text(json.dumps({"profile": "upgrade", "datasets": {"demo_h50": {"window_frames": 8}}}), encoding="utf-8")
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
                "params": {"variant": "convgru_pixel_mse", "loss_mode": "frame_mse", "hidden_channels": 16},
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

    summary = build_comparison_dashboard(sweep_dirs=[sweep], out_dir=tmp_path / "comparison")
    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    html = Path(summary["html_path"]).read_text(encoding="utf-8")

    assert summary["row_count"] == 2
    assert summary["video_collection_count"] == 1
    assert manifest["selected_models"][0]["experiment_id"] == "convgru_demo"
    assert manifest["video_collections"][0]["options"][0]["intensity_src"].endswith("demo_best_8_left_intensity.mp4")
    assert "inputVideoFilter" in html
    assert "Video Comparison" in html
    assert "videoGrid" in html
    assert "clipSet" in html
    assert "Held-out-first" in html
