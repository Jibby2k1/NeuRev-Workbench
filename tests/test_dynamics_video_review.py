import json
from pathlib import Path

from neurobench.dynamics.video_review import build_video_error_review


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _example(scale: float) -> dict:
    return {
        "index": 0,
        "video_id": "vid_test",
        "split": "test",
        "input_last": [[0.0, 0.2], [0.3, 0.4]],
        "target_next": [[0.1, 0.2], [0.5, 0.4]],
        "predicted_next": [[0.1, 0.2], [0.5 * scale, 0.4]],
        "abs_error_mean": 0.01,
    }


def _clip(scale: float) -> dict:
    frames = []
    for offset in range(3):
        frame = _example(scale)
        frame["index"] = offset
        frame["target_frame_index"] = 100 + offset
        frame["window_start_index"] = 90 + offset
        frame["window_end_index"] = 99 + offset
        frame["persistence_next"] = frame["input_last"]
        frames.append(frame)
    return {
        "clip_index": 0,
        "video_id": "vid_test",
        "split": "test",
        "start_target_frame_index": 100,
        "end_target_frame_index": 102,
        "frame_count": len(frames),
        "frames": frames,
    }


def test_build_video_error_review_renders_selected_prediction_panels(tmp_path):
    comparison = tmp_path / "comparison"
    sweep = tmp_path / "sweep"
    rows = []
    for exp_id, family, improve, scale in [("gru_best", "latent_gru", 0.2, 1.0), ("xfmr_best", "latent_transformer", 0.1, 0.9)]:
        run = sweep / exp_id
        metrics_path = run / "latent_rnn_metrics.json"
        _write_json(metrics_path, {"ok": True})
        _write_json(run / "prediction_examples.json", {"schema_version": 1, "examples": [_example(scale)]})
        rows.append(
            {
                "experiment_id": exp_id,
                "row_id": f"unit:{exp_id}",
                "kind": family,
                "model_family": family,
                "dataset_key": "w8_s1_h5",
                "prediction_target": "delta",
                "hyperparameter_summary": f"model={family}, target=delta",
                "metrics_path": str(metrics_path),
                "test_improvement_over_persistence_mse": improve,
                "test_decoded_prediction_mse": 1.0 - improve,
                "test_persistence_mse": 1.0,
                "all_improvement_over_persistence_mse": improve / 2.0,
                "all_decoded_prediction_mse": 1.0 - improve / 2.0,
                "all_persistence_mse": 1.0,
            }
        )
    _write_json(
        comparison / "comparison_manifest.json",
        {
            "schema_version": 1,
            "rows": rows,
            "datasets": {
                "w8_s1_h5": {
                    "windowing": {
                        "window_frames": 8,
                        "prediction_horizon_frames": 5,
                        "effective_frame_rate_hz": 50.0,
                    }
                }
            },
        },
    )

    summary = build_video_error_review(comparison_dir=comparison, out_dir=tmp_path / "review", selection_mode="best_by_family", split="test", max_models=5)
    html = Path(summary["html_path"]).read_text(encoding="utf-8")

    assert summary["selected_model_count"] == 2
    assert summary["models"][0]["experiment_id"] == "gru_best"
    assert summary["models"][0]["example_video_id"] == "vid_test"
    assert summary["models"][0]["example_split"] == "test"
    assert Path(summary["models"][0]["panel_png"]).is_file()
    assert Path(summary["models"][0]["panel_png"]).read_bytes().startswith(bytes([137]) + b"PNG")
    assert "Grid Dynamics Video Error Review" in html
    assert "Panels show target" in html
    assert "gru_best" in html
    assert "vid_test" in html

    all_summary = build_video_error_review(comparison_dir=comparison, out_dir=tmp_path / "review_all", selection_mode="best_test", split="all", max_models=1)
    assert all_summary["models"][0]["split_improvement_over_persistence_mse"] == 0.1

def test_build_video_error_review_reports_top_rows_missing_visual_examples(tmp_path):
    comparison = tmp_path / "comparison"
    sweep = tmp_path / "sweep"
    missing_run = sweep / "convgru_top"
    missing_metrics = missing_run / "concept_metrics.json"
    _write_json(missing_metrics, {"ok": True})
    visible_run = sweep / "gru_visible"
    visible_metrics = visible_run / "latent_rnn_metrics.json"
    _write_json(visible_metrics, {"ok": True})
    _write_json(visible_run / "prediction_examples.json", {"schema_version": 1, "examples": [_example(1.0)]})
    _write_json(
        comparison / "comparison_manifest.json",
        {
            "schema_version": 1,
            "rows": [
                {
                    "experiment_id": "convgru_top",
                    "row_id": "unit:convgru_top",
                    "kind": "convgru_pixel",
                    "model_family": "pixel_convgru",
                    "dataset_key": "w8_s1_h2",
                    "prediction_target": None,
                    "hyperparameter_summary": "model=convgru pixel",
                    "metrics_path": str(missing_metrics),
                    "test_improvement_over_persistence_mse": 0.5,
                    "test_decoded_prediction_mse": 0.5,
                    "test_persistence_mse": 1.0,
                },
                {
                    "experiment_id": "gru_visible",
                    "row_id": "unit:gru_visible",
                    "kind": "latent_gru",
                    "model_family": "latent_gru",
                    "dataset_key": "w8_s1_h2",
                    "prediction_target": "delta",
                    "hyperparameter_summary": "model=latent GRU, target=delta",
                    "metrics_path": str(visible_metrics),
                    "test_improvement_over_persistence_mse": 0.2,
                    "test_decoded_prediction_mse": 0.8,
                    "test_persistence_mse": 1.0,
                },
            ],
            "datasets": {"w8_s1_h2": {"windowing": {"window_frames": 8, "prediction_horizon_frames": 2, "effective_frame_rate_hz": 50.0}}},
        },
    )

    summary = build_video_error_review(comparison_dir=comparison, out_dir=tmp_path / "review", selection_mode="best_test", split="test", max_models=2)
    html = Path(summary["html_path"]).read_text(encoding="utf-8")

    assert summary["selected_model_count"] == 1
    assert summary["models"][0]["experiment_id"] == "gru_visible"
    assert summary["missing_visual_example_count"] == 1
    assert summary["missing_visual_example_rows"][0]["experiment_id"] == "convgru_top"
    assert summary["missing_visual_example_rows"][0]["split_improvement_over_persistence_mse"] == 0.5
    assert "Top-ranked rows without visual examples" in html
    assert "convgru_top" in html
    assert "missing prediction_examples.json" in html


def test_build_video_error_review_renders_temporal_clip_artifacts(tmp_path):
    comparison = tmp_path / "comparison"
    sweep = tmp_path / "sweep"
    run = sweep / "shared_gru_clip"
    metrics_path = run / "multi_horizon_gru_metrics.json"
    _write_json(metrics_path, {"ok": True})
    _write_json(run / "prediction_clip_examples.json", {"schema_version": 1, "clip_count": 1, "clips": [_clip(0.95)]})
    _write_json(
        comparison / "comparison_manifest.json",
        {
            "schema_version": 1,
            "rows": [
                {
                    "experiment_id": "shared_gru_clip",
                    "row_id": "unit:shared_gru_clip",
                    "kind": "latent_gru",
                    "model_family": "latent_gru",
                    "dataset_key": "w8_s1_h5",
                    "prediction_target": "delta",
                    "hyperparameter_summary": "model=shared_gru, target=delta",
                    "metrics_path": str(metrics_path),
                    "test_improvement_over_persistence_mse": 0.3,
                    "test_decoded_prediction_mse": 0.7,
                    "test_persistence_mse": 1.0,
                }
            ],
            "datasets": {
                "w8_s1_h5": {
                    "windowing": {
                        "window_frames": 8,
                        "prediction_horizon_frames": 5,
                        "effective_frame_rate_hz": 50.0,
                    }
                }
            },
        },
    )

    summary = build_video_error_review(comparison_dir=comparison, out_dir=tmp_path / "clip_review", selection_mode="best_test", split="test", max_models=1)
    html = Path(summary["html_path"]).read_text(encoding="utf-8")
    model = summary["models"][0]

    assert summary["temporal_clip_model_count"] == 1
    assert model["artifact_mode"] == "temporal_clip"
    assert model["clip_frame_count"] == 3
    assert model["clip_start_target_frame_index"] == 100
    assert model["clip_end_target_frame_index"] == 102
    assert model["target_frame_index"] == 100
    assert Path(model["clip_panel_png"]).is_file()
    assert Path(model["clip_panel_png"]).read_bytes().startswith(bytes([137]) + b"PNG")
    assert "Temporal clips" in html
    assert "temporal_clip" in html
    assert "100 to 102" in html

def test_build_video_error_review_uses_explicit_artifact_paths(tmp_path):
    comparison = tmp_path / "comparison"
    sweep = tmp_path / "sweep"
    run = sweep / "convgru_top"
    metrics_path = run / "concept_metrics.json"
    artifact_path = tmp_path / "artifacts" / "convgru_prediction_examples.json"
    _write_json(metrics_path, {"ok": True})
    _write_json(artifact_path, {"schema_version": 1, "examples": [_example(1.0)]})
    _write_json(
        comparison / "comparison_manifest.json",
        {
            "schema_version": 1,
            "rows": [
                {
                    "experiment_id": "convgru_top",
                    "row_id": "unit:convgru_top",
                    "kind": "convgru_pixel",
                    "model_family": "pixel_convgru",
                    "dataset_key": "w8_s1_h2",
                    "prediction_target": None,
                    "hyperparameter_summary": "model=convgru pixel",
                    "metrics_path": str(metrics_path),
                    "prediction_examples_path": str(artifact_path),
                    "test_improvement_over_persistence_mse": 0.5,
                    "test_decoded_prediction_mse": 0.5,
                    "test_persistence_mse": 1.0,
                }
            ],
            "datasets": {"w8_s1_h2": {"windowing": {"window_frames": 8, "prediction_horizon_frames": 2, "effective_frame_rate_hz": 50.0}}},
        },
    )

    summary = build_video_error_review(comparison_dir=comparison, out_dir=tmp_path / "review", selection_mode="best_test", split="test", max_models=1)

    assert summary["selected_model_count"] == 1
    assert summary["missing_visual_example_count"] == 0
    assert summary["models"][0]["experiment_id"] == "convgru_top"
    assert Path(summary["models"][0]["panel_png"]).is_file()


def test_build_video_error_review_removes_stale_generated_panels(tmp_path):
    comparison = tmp_path / "comparison"
    sweep = tmp_path / "sweep"
    run = sweep / "fresh_model"
    metrics_path = run / "latent_rnn_metrics.json"
    _write_json(metrics_path, {"ok": True})
    _write_json(run / "prediction_examples.json", {"schema_version": 1, "examples": [_example(1.0)]})
    _write_json(
        comparison / "comparison_manifest.json",
        {
            "schema_version": 1,
            "rows": [
                {
                    "experiment_id": "fresh_model",
                    "row_id": "unit:fresh_model",
                    "kind": "latent_gru",
                    "model_family": "latent_gru",
                    "dataset_key": "w8_s1_h2",
                    "metrics_path": str(metrics_path),
                    "test_improvement_over_persistence_mse": 0.3,
                    "test_decoded_prediction_mse": 0.7,
                    "test_persistence_mse": 1.0,
                }
            ],
            "datasets": {"w8_s1_h2": {"windowing": {"window_frames": 8, "prediction_horizon_frames": 2, "effective_frame_rate_hz": 50.0}}},
        },
    )
    out_dir = tmp_path / "review"
    out_dir.mkdir()
    stale_example = out_dir / "model_99_old_example_0.png"
    stale_clip = out_dir / "model_99_old_clip_0.png"
    unrelated = out_dir / "keep.png"
    stale_example.write_bytes(b"old")
    stale_clip.write_bytes(b"old")
    unrelated.write_bytes(b"keep")

    summary = build_video_error_review(comparison_dir=comparison, out_dir=out_dir, selection_mode="best_test", split="test", max_models=1)

    assert not stale_example.exists()
    assert not stale_clip.exists()
    assert unrelated.read_bytes() == b"keep"
    assert Path(summary["models"][0]["panel_png"]).is_file()


def test_build_video_error_review_selects_first_heldout_artifact(tmp_path):
    comparison = tmp_path / "comparison"
    sweep = tmp_path / "sweep"
    run = sweep / "heldout_model"
    metrics_path = run / "latent_rnn_metrics.json"
    _write_json(metrics_path, {"ok": True})
    train_example = _example(1.0)
    train_example["index"] = 0
    train_example["video_id"] = "train_video"
    train_example["split"] = "train"
    test_example = _example(0.9)
    test_example["index"] = 7
    test_example["video_id"] = "heldout_test_video"
    test_example["split"] = "test"
    later_test_example = _example(0.8)
    later_test_example["index"] = 8
    later_test_example["video_id"] = "later_test_video"
    later_test_example["split"] = "test"
    _write_json(run / "prediction_examples.json", {"schema_version": 1, "examples": [train_example, test_example, later_test_example]})
    _write_json(
        comparison / "comparison_manifest.json",
        {
            "schema_version": 1,
            "rows": [
                {
                    "experiment_id": "heldout_model",
                    "row_id": "unit:heldout_model",
                    "kind": "latent_gru",
                    "model_family": "latent_gru",
                    "dataset_key": "w8_s1_h2",
                    "prediction_target": "delta",
                    "metrics_path": str(metrics_path),
                    "test_improvement_over_persistence_mse": 0.2,
                    "test_decoded_prediction_mse": 0.8,
                    "test_persistence_mse": 1.0,
                }
            ],
            "datasets": {"w8_s1_h2": {"windowing": {"window_frames": 8, "prediction_horizon_frames": 2, "effective_frame_rate_hz": 50.0}}},
        },
    )

    summary = build_video_error_review(
        comparison_dir=comparison,
        out_dir=tmp_path / "heldout_review",
        selection_mode="heldout_first",
        split="test",
        max_models=1,
        example_index=0,
    )
    html = Path(summary["html_path"]).read_text(encoding="utf-8")

    assert summary["models"][0]["experiment_id"] == "heldout_model"
    assert summary["models"][0]["example_index"] == 7
    assert summary["models"][0]["example_video_id"] == "heldout_test_video"
    assert summary["models"][0]["example_split"] == "test"
    assert summary["models"][0]["selection_metric_name"] == "improvement_over_persistence_mse"
    assert summary["models"][0]["selection_metric_value"] == 0.2
    assert "heldout_test_video" in html


def test_build_video_error_review_selects_per_video_extremes(tmp_path):
    comparison = tmp_path / "comparison"
    sweep = tmp_path / "sweep"
    rows = []
    for exp_id, global_improve, best_video_improve, worst_video_improve, scale in [
        ("global_winner", 0.5, 0.03, -0.01, 1.0),
        ("video_winner", 0.1, 0.09, -0.02, 0.9),
        ("video_loser", 0.2, 0.04, -0.08, 0.8),
    ]:
        run = sweep / exp_id
        metrics_path = run / "latent_rnn_metrics.json"
        _write_json(metrics_path, {"ok": True})
        first_example = _example(scale)
        first_example["video_id"] = f"{exp_id}_other"
        first_example["index"] = 0
        good_example = _example(scale)
        good_example["video_id"] = f"{exp_id}_good"
        good_example["index"] = 11
        bad_example = _example(scale)
        bad_example["video_id"] = f"{exp_id}_bad"
        bad_example["index"] = 22
        _write_json(run / "prediction_examples.json", {"schema_version": 1, "examples": [first_example, good_example, bad_example]})
        rows.append(
            {
                "experiment_id": exp_id,
                "row_id": f"unit:{exp_id}",
                "kind": "latent_gru",
                "model_family": "latent_gru",
                "dataset_key": "w8_s1_h2",
                "prediction_target": "delta",
                "metrics_path": str(metrics_path),
                "test_improvement_over_persistence_mse": global_improve,
                "test_decoded_prediction_mse": 1.0 - global_improve,
                "test_persistence_mse": 1.0,
                "video_error_summary": {
                    "test": {
                        "video_count": 2,
                        "best_videos": [
                            {
                                "video_id": f"{exp_id}_good",
                                "window_count": 3,
                                "decoded_prediction_mse": 0.1 - best_video_improve,
                                "persistence_mse": 0.1,
                                "improvement_over_persistence_mse": best_video_improve,
                            }
                        ],
                        "worst_videos": [
                            {
                                "video_id": f"{exp_id}_bad",
                                "window_count": 4,
                                "decoded_prediction_mse": 0.1 - worst_video_improve,
                                "persistence_mse": 0.1,
                                "improvement_over_persistence_mse": worst_video_improve,
                            }
                        ],
                    }
                },
            }
        )
    _write_json(
        comparison / "comparison_manifest.json",
        {
            "schema_version": 1,
            "rows": rows,
            "datasets": {"w8_s1_h2": {"windowing": {"window_frames": 8, "prediction_horizon_frames": 2, "effective_frame_rate_hz": 50.0}}},
        },
    )

    best_summary = build_video_error_review(comparison_dir=comparison, out_dir=tmp_path / "best_video", selection_mode="most_improved_video", split="test", max_models=1)
    best_html = Path(best_summary["html_path"]).read_text(encoding="utf-8")
    worst_summary = build_video_error_review(comparison_dir=comparison, out_dir=tmp_path / "worst_video", selection_mode="least_improved_video", split="test", max_models=1)
    worst_html = Path(worst_summary["html_path"]).read_text(encoding="utf-8")

    assert best_summary["models"][0]["experiment_id"] == "video_winner"
    assert best_summary["models"][0]["selection_metric_name"] == "best_video_improvement_over_persistence_mse"
    assert best_summary["models"][0]["selection_metric_value"] == 0.09
    assert best_summary["models"][0]["selection_video_id"] == "video_winner_good"
    assert best_summary["models"][0]["selection_video_window_count"] == 3
    assert best_summary["models"][0]["example_video_id"] == "video_winner_good"
    assert best_summary["models"][0]["example_index"] == 11
    assert "best-video improve 0.09" in best_html
    assert "video_winner_good improve 0.09 (3 windows)" in best_html

    assert worst_summary["models"][0]["experiment_id"] == "video_loser"
    assert worst_summary["models"][0]["selection_metric_name"] == "worst_video_improvement_over_persistence_mse"
    assert worst_summary["models"][0]["selection_metric_value"] == -0.08
    assert worst_summary["models"][0]["selection_video_id"] == "video_loser_bad"
    assert worst_summary["models"][0]["selection_video_window_count"] == 4
    assert worst_summary["models"][0]["example_video_id"] == "video_loser_bad"
    assert worst_summary["models"][0]["example_index"] == 22
    assert "worst-video improve -0.08" in worst_html
    assert "video_loser_bad improve -0.08 (4 windows)" in worst_html


def test_build_video_error_review_selects_best_active_cell_rows(tmp_path):
    comparison = tmp_path / "comparison"
    sweep = tmp_path / "sweep"
    rows = []
    for exp_id, test_improve, active_improve, scale in [
        ("global_best", 0.5, 0.05, 1.0),
        ("active_best", 0.2, 0.4, 0.9),
    ]:
        run = sweep / exp_id
        metrics_path = run / "concept_metrics.json"
        _write_json(metrics_path, {"ok": True})
        _write_json(run / "prediction_examples.json", {"schema_version": 1, "examples": [_example(scale)]})
        rows.append(
            {
                "experiment_id": exp_id,
                "row_id": f"unit:{exp_id}",
                "kind": "convgru_pixel",
                "model_family": "pixel_convgru",
                "dataset_key": "w8_s1_h2",
                "hyperparameter_summary": f"model={exp_id}",
                "metrics_path": str(metrics_path),
                "test_improvement_over_persistence_mse": test_improve,
                "test_active_cell_improvement_over_persistence_mse": active_improve,
                "test_decoded_prediction_mse": 1.0 - test_improve,
                "test_persistence_mse": 1.0,
            }
        )
    _write_json(
        comparison / "comparison_manifest.json",
        {
            "schema_version": 1,
            "rows": rows,
            "datasets": {"w8_s1_h2": {"windowing": {"window_frames": 8, "prediction_horizon_frames": 2, "effective_frame_rate_hz": 50.0}}},
        },
    )

    summary = build_video_error_review(comparison_dir=comparison, out_dir=tmp_path / "active_review", selection_mode="best_active_cell", split="test", max_models=1)
    html = Path(summary["html_path"]).read_text(encoding="utf-8")

    assert summary["selected_model_count"] == 1
    assert summary["selection_mode"] == "best_active_cell"
    assert summary["models"][0]["experiment_id"] == "active_best"
    assert summary["models"][0]["split_improvement_over_persistence_mse"] == 0.2
    assert summary["models"][0]["selection_metric_name"] == "active_cell_improvement_over_persistence_mse"
    assert summary["models"][0]["selection_metric_value"] == 0.4
    assert "active-cell improve 0.4" in html
