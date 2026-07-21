import json
from pathlib import Path

import numpy as np

from neurobench.dynamics.multi_horizon import build_active_cell_rescue_plan, build_multi_horizon_report, build_shared_horizon_baseline_comparison, build_shared_horizon_neural_grid_plan, build_shared_horizon_neural_grid_status, build_shared_horizon_review_manifest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _row(exp_id: str, dataset_key: str, improve: float, *, hidden_dim: int = 64):
    return {
        "experiment_id": exp_id,
        "row_id": f"unit:{exp_id}",
        "kind": "latent_gru",
        "model_family": "latent_gru",
        "dataset_key": dataset_key,
        "prediction_target": "delta",
        "hyperparameter_summary": f"model=latent GRU, target=delta, hd={hidden_dim}, lr=1e-04, batch=4",
        "test_decoded_prediction_mse": 1.0 - improve,
        "test_persistence_mse": 1.0,
        "test_improvement_over_persistence_mse": improve,
        "params": {
            "prediction_target": "delta",
            "hidden_dim": hidden_dim,
            "learning_rate": 1e-4,
            "batch_size": 4,
            "grid_size": 128,
            "grid_pooling": "max_intensity",
            "model_label": "latent GRU",
        },
    }


def test_build_multi_horizon_report_pairs_matching_hyperparameters_and_plans_shared_configs(tmp_path):
    comparison = tmp_path / "comparison"
    _write_json(
        comparison / "comparison_manifest.json",
        {
            "schema_version": 1,
            "datasets": {
                "w8_s1_h2": {"windowing": {"prediction_horizon_frames": 2, "prediction_horizon_sec": 0.04, "effective_frame_rate_hz": 50.0}},
                "w8_s1_h5": {"windowing": {"prediction_horizon_frames": 5, "prediction_horizon_sec": 0.10, "effective_frame_rate_hz": 50.0}},
            },
            "rows": [
                _row("gru_h2", "w8_s1_h2", 0.20),
                _row("gru_h5", "w8_s1_h5", 0.12),
                _row("gru_unpaired", "w8_s1_h2", 0.30, hidden_dim=128),
            ],
        },
    )

    report = build_multi_horizon_report(comparison_dir=comparison, out_dir=tmp_path / "mh", split="test", max_candidates=5)
    markdown = Path(report["markdown_path"]).read_text(encoding="utf-8")
    plan = json.loads(Path(report["plan_manifest_path"]).read_text(encoding="utf-8"))

    assert report["paired_group_count"] == 1
    assert report["top_candidates"][0]["min_improvement_over_persistence_mse"] == 0.12
    assert report["top_candidates"][0]["long_minus_short_improvement"] == -0.08000000000000002
    assert report["family_summary"][0]["positive_all_horizon_count"] == 1
    assert report["planned_shared_horizon_configs"]
    assert report["planned_shared_horizon_configs"][0]["shared_horizons_frames"] == [2, 5]
    assert plan["planned_configs"][0]["source_experiment_ids"] == ["gru_h2", "gru_h5"]
    assert "Multi-Horizon Forecasting Report" in markdown
    assert "Top Shared-Horizon Candidates" in markdown





def test_build_shared_horizon_baseline_comparison_ranks_runs_and_warns_on_active_cells(tmp_path):
    linear = tmp_path / "linear" / "multi_horizon_linear_metrics.json"
    gru = tmp_path / "gru" / "multi_horizon_gru_metrics.json"
    _write_json(
        linear,
        {
            "model_family": "multi_horizon_linear_latent",
            "model_kind": "shared_multi_horizon_linear_latent",
            "shared_horizons_frames": [2, 5],
            "decoded_prediction_mse": 0.6,
            "persistence_mse": 1.0,
            "improvement_over_persistence_mse": 0.4,
            "selection_latent_code_mse": 0.3,
            "per_horizon_metrics": {
                "w8_s1_h2": {
                    "prediction_horizon_frames": 2,
                    "test_decoded_prediction_mse": 0.8,
                    "test_persistence_mse": 1.0,
                    "test_improvement_over_persistence_mse": 0.2,
                    "test_active_cell_improvement_over_persistence_mse": 0.1,
                    "test_high_change_improvement_over_persistence_mse": 0.3,
                },
                "w8_s1_h5": {
                    "prediction_horizon_frames": 5,
                    "test_decoded_prediction_mse": 0.9,
                    "test_persistence_mse": 1.0,
                    "test_improvement_over_persistence_mse": 0.1,
                    "test_active_cell_improvement_over_persistence_mse": 0.05,
                    "test_high_change_improvement_over_persistence_mse": 0.25,
                },
            },
        },
    )
    _write_json(
        gru,
        {
            "model_family": "multi_horizon_latent_gru",
            "model_kind": "shared_multi_horizon_latent_gru",
            "shared_horizons_frames": [2, 5],
            "decoded_prediction_mse": 0.5,
            "persistence_mse": 1.0,
            "improvement_over_persistence_mse": 0.5,
            "selection_latent_code_mse": 0.2,
            "decoded_evaluation_mode": "chunked",
            "evaluation_batch_size": 16,
            "per_horizon_metrics": {
                "w8_s1_h2": {
                    "prediction_horizon_frames": 2,
                    "test_decoded_prediction_mse": 0.7,
                    "test_persistence_mse": 1.0,
                    "test_improvement_over_persistence_mse": 0.3,
                    "test_active_cell_improvement_over_persistence_mse": -0.2,
                    "test_high_change_improvement_over_persistence_mse": 0.4,
                    "prediction_clip_examples_path": "clip_h2.json",
                },
                "w8_s1_h5": {
                    "prediction_horizon_frames": 5,
                    "test_decoded_prediction_mse": 0.75,
                    "test_persistence_mse": 1.0,
                    "test_improvement_over_persistence_mse": 0.25,
                    "test_active_cell_improvement_over_persistence_mse": -0.1,
                    "test_high_change_improvement_over_persistence_mse": 0.5,
                    "prediction_clip_examples_path": "clip_h5.json",
                },
            },
        },
    )

    report = build_shared_horizon_baseline_comparison(
        runs=[f"linear={linear}", f"planned_gru={gru}"],
        out_dir=tmp_path / "comparison",
    )
    markdown = Path(report["markdown_path"]).read_text(encoding="utf-8")

    assert report["best_overall"]["label"] == "planned_gru"
    assert report["horizon_summary"][0]["best_label"] == "planned_gru"
    assert len(report["active_cell_warnings"]) == 2
    assert report["runs"][0]["params"]["decoded_evaluation_mode"] == "chunked"
    assert "Active-Cell Warnings" in markdown
    assert "clip_h2.json" in markdown


def test_build_shared_horizon_review_manifest_writes_review_rows_and_dataset_metadata(tmp_path):
    comparison = tmp_path / "comparison"
    _write_json(
        comparison / "comparison_manifest.json",
        {
            "schema_version": 1,
            "datasets": {
                "w8_s1_h2": {"windowing": {"prediction_horizon_frames": 2}},
                "w8_s1_h5": {"windowing": {"prediction_horizon_frames": 5}},
            },
            "rows": [],
        },
    )
    run_dir = tmp_path / "shared_gru"
    (run_dir / "w8_s1_h2").mkdir(parents=True)
    (run_dir / "w8_s1_h5").mkdir(parents=True)
    h2_review = run_dir / "w8_s1_h2" / "per_horizon_metrics_for_review.json"
    h2_examples = run_dir / "w8_s1_h2" / "prediction_examples.json"
    h2_clips = run_dir / "w8_s1_h2" / "prediction_clip_examples.json"
    h5_clips = run_dir / "w8_s1_h5" / "prediction_clip_examples.json"
    h2_review.write_text("{}\n", encoding="utf-8")
    h2_examples.write_text("{}\n", encoding="utf-8")
    h2_clips.write_text("{}\n", encoding="utf-8")
    h5_clips.write_text("{}\n", encoding="utf-8")
    metrics_path = run_dir / "multi_horizon_gru_metrics.json"
    _write_json(
        metrics_path,
        {
            "model_family": "multi_horizon_latent_gru",
            "model_kind": "shared_multi_horizon_latent_gru",
            "prediction_target": "delta",
            "hidden_dim": 32,
            "num_layers": 1,
            "learning_rate": 0.0003,
            "evaluation_batch_size": 16,
            "shared_horizons_frames": [2, 5],
            "improvement_over_persistence_mse": 0.4,
            "per_horizon_metrics": {
                "w8_s1_h2": {
                    "prediction_horizon_frames": 2,
                    "test_decoded_prediction_mse": 0.7,
                    "test_persistence_mse": 1.0,
                    "test_improvement_over_persistence_mse": 0.3,
                    "test_active_cell_improvement_over_persistence_mse": -0.2,
                },
                "w8_s1_h5": {
                    "prediction_horizon_frames": 5,
                    "test_decoded_prediction_mse": 0.8,
                    "test_persistence_mse": 1.0,
                    "test_improvement_over_persistence_mse": 0.2,
                    "prediction_clip_examples_path": str(h5_clips),
                },
            },
        },
    )

    manifest = build_shared_horizon_review_manifest(
        runs=[f"planned_gru={metrics_path}"],
        out_dir=tmp_path / "review_input",
        comparison_dir=comparison,
    )
    saved = json.loads(Path(manifest["manifest_path"]).read_text(encoding="utf-8"))

    assert saved["manifest_kind"] == "shared_horizon_review_input"
    assert saved["run_count"] == 1
    assert saved["row_count"] == 2
    assert saved["datasets"]["w8_s1_h2"]["windowing"]["prediction_horizon_frames"] == 2
    h2_row = next(row for row in saved["rows"] if row["dataset_key"] == "w8_s1_h2")
    h5_row = next(row for row in saved["rows"] if row["dataset_key"] == "w8_s1_h5")
    assert h2_row["experiment_id"] == "planned_gru_w8_s1_h2"
    assert h2_row["model_family"] == "latent_gru"
    assert h2_row["metrics_path"] == str(h2_review)
    assert h2_row["prediction_examples_path"] == str(h2_examples)
    assert h2_row["prediction_clip_examples_path"] == str(h2_clips)
    assert h2_row["test_active_cell_improvement_over_persistence_mse"] == -0.2
    assert "hidden_dim=32" in h2_row["hyperparameter_summary"]
    assert h5_row["prediction_clip_examples_path"] == str(h5_clips)


def test_build_active_cell_rescue_plan_prioritizes_transformer_after_active_warnings(tmp_path):
    comparison = tmp_path / "shared_horizon_baseline_comparison.json"
    status = tmp_path / "shared_horizon_neural_grid_status.json"
    _write_json(
        comparison,
        {
            "best_overall": {"label": "shared_linear", "improvement_over_persistence_mse": 0.5},
            "active_cell_warnings": ["shared_linear w8_s1_h2 has negative test active-cell improvement"],
        },
    )
    _write_json(
        status,
        {
            "rows": [
                {
                    "config_id": "done_gru",
                    "model_family": "multi_horizon_latent_gru",
                    "status": "completed",
                    "params": {"hidden_dim": 32, "num_layers": 1},
                },
                {
                    "config_id": "pending_gru",
                    "model_family": "multi_horizon_latent_gru",
                    "status": "pending",
                    "priority": "core",
                    "params": {"hidden_dim": 32, "num_layers": 1, "learning_rate": 0.0003},
                    "command": "train-shared-gru-horizons",
                },
                {
                    "config_id": "small_transformer",
                    "model_family": "multi_horizon_latent_transformer",
                    "status": "pending",
                    "priority": "transformer_candidate",
                    "params": {"model_dim": 64, "num_heads": 2, "num_layers": 1},
                    "command": "train-shared-transformer-horizons",
                },
            ]
        },
    )

    plan = build_active_cell_rescue_plan(comparison_report=comparison, grid_status=status, out_dir=tmp_path / "rescue")
    markdown = Path(plan["markdown_path"]).read_text(encoding="utf-8")

    assert plan["active_cell_warning_count"] == 1
    assert plan["recommended_candidates"][0]["config_id"] == "small_transformer"
    assert plan["recommended_candidates"][0]["command"] == "train-shared-transformer-horizons"
    assert "## Next Candidate Command" in markdown
    assert "train-shared-transformer-horizons" in markdown
    assert "CPU/RAM headroom" in markdown
    assert "active-cell" in markdown
    assert "Do not run another same-objective GRU" in markdown


def test_build_shared_horizon_neural_grid_status_reads_completed_metrics(tmp_path):
    run_root = tmp_path / "runs"
    done = run_root / "done_gru"
    pending = run_root / "pending_xfmr"
    done.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "manifest_kind": "shared_horizon_neural_followup_grid",
        "planned_configs": [
            {
                "config_id": "done_gru",
                "model_family": "multi_horizon_latent_gru",
                "model_kind": "shared_multi_horizon_latent_gru",
                "status": "ready",
                "out_dir": str(done),
                "command": "train-shared-gru-horizons",
                "params": {"hidden_dim": 32},
            },
            {
                "config_id": "pending_xfmr",
                "model_family": "multi_horizon_latent_transformer",
                "model_kind": "shared_multi_horizon_latent_transformer",
                "status": "ready",
                "out_dir": str(pending),
                "command": "train-shared-transformer-horizons",
                "params": {"model_dim": 64},
            },
        ],
    }
    manifest_path = tmp_path / "shared_horizon_neural_grid_manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    (done / "multi_horizon_gru_metrics.json").write_text(
        json.dumps(
            {
                "decoded_prediction_mse": 0.4,
                "persistence_mse": 1.0,
                "improvement_over_persistence_mse": 0.6,
                "selection_latent_code_mse": 0.2,
                "evaluation_window_count": 10,
                "training_window_count": 5,
                "decoded_evaluation_mode": "chunked",
                "evaluation_batch_size": 2,
                "shared_horizons_frames": [1, 2],
                "per_horizon_metrics": {
                    "w8_s1_h1": {
                        "dataset_key": "w8_s1_h1",
                        "prediction_horizon_frames": 1,
                        "test_improvement_over_persistence_mse": 0.05,
                        "test_active_cell_improvement_over_persistence_mse": -0.01,
                        "test_high_change_improvement_over_persistence_mse": 0.2,
                        "test_top_activity_improvement_over_persistence_mse": -0.02,
                    },
                    "w8_s1_h2": {
                        "dataset_key": "w8_s1_h2",
                        "prediction_horizon_frames": 2,
                        "test_improvement_over_persistence_mse": 0.04,
                        "test_active_cell_improvement_over_persistence_mse": 0.03,
                        "test_high_change_improvement_over_persistence_mse": 0.1,
                        "test_top_activity_improvement_over_persistence_mse": 0.01,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    status = build_shared_horizon_neural_grid_status(manifest_path=manifest_path, out_dir=tmp_path / "status")

    assert status["status_counts"] == {"completed": 1, "pending": 1}
    assert status["completed_count"] == 1
    assert status["best_completed"][0]["config_id"] == "done_gru"
    assert status["best_completed"][0]["improvement_over_persistence_mse"] == 0.6
    assert status["best_completed"][0]["min_test_active_cell_improvement_over_persistence_mse"] == -0.01
    assert status["best_completed"][0]["min_test_high_change_improvement_over_persistence_mse"] == 0.1
    assert status["best_completed"][0]["test_active_cell_positive_horizon_count"] == 1
    assert status["best_completed"][0]["test_active_cell_horizon_count"] == 2
    assert status["best_completed"][0]["all_test_active_cell_positive"] is False
    assert Path(status["status_path"]).is_file()
    assert Path(status["markdown_path"]).is_file()
    markdown = Path(status["markdown_path"]).read_text(encoding="utf-8")
    assert "done_gru" in markdown
    assert "pending_xfmr" in markdown
    assert "Test active min" in markdown
    assert "Every completed shared-neural entry" in markdown


def test_build_shared_horizon_neural_grid_plan_writes_executable_gru_commands(tmp_path):
    h2 = tmp_path / "w8_s1_h2" / "dynamics_dataset.json"
    h5 = tmp_path / "w8_s1_h5" / "dynamics_dataset.json"
    h2.parent.mkdir(parents=True)
    h5.parent.mkdir(parents=True)
    h2.write_text("{}\n", encoding="utf-8")
    h5.write_text("{}\n", encoding="utf-8")
    ae = tmp_path / "ae" / "autoencoder_run.json"
    ae.parent.mkdir(parents=True)
    ae.write_text("{}\n", encoding="utf-8")

    plan = build_shared_horizon_neural_grid_plan(
        datasets=[h2, h5],
        autoencoder_run=ae,
        out_dir=tmp_path / "plan",
        run_root=tmp_path / "runs",
        epochs=3,
        batch_size=8,
        evaluation_batch_size=2,
        seeds=[7],
        max_gru_configs=3,
        include_transformer_placeholders=True,
    )

    manifest_path = Path(plan["manifest_path"])
    markdown_path = Path(plan["markdown_path"])
    script_path = Path(plan["script_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_kind"] == "shared_horizon_neural_followup_grid"
    assert manifest["directly_executable_count"] == 7
    assert manifest["placeholder_count"] == 0
    ready = [spec for spec in manifest["planned_configs"] if spec["status"] == "ready"]
    gru_ready = [spec for spec in ready if spec["model_family"] == "multi_horizon_latent_gru"]
    transformer_ready = [spec for spec in ready if spec["model_family"] == "multi_horizon_latent_transformer"]
    assert len(gru_ready) == 3
    assert len(transformer_ready) == 4
    assert all("--evaluation-batch-size 2" in spec["command"] for spec in ready)
    assert all("train-shared-gru-horizons" in spec["command"] for spec in gru_ready)
    assert all("train-shared-transformer-horizons" in spec["command"] for spec in transformer_ready)
    assert all(spec["params"]["decoded_evaluation_mode"] == "chunked" for spec in ready)
    assert markdown_path.is_file()
    script = script_path.read_text(encoding="utf-8")
    assert script.startswith("#!/usr/bin/env bash")
    assert script.count("train-shared-gru-horizons") == 3
    assert script.count("train-shared-transformer-horizons") == 4


def test_shared_multi_horizon_linear_latent_reports_each_horizon(tmp_path):
    from neurobench.dynamics.multi_horizon_linear import evaluate_shared_multi_horizon_linear_latent
    from neurobench.dynamics.train import train_autoencoder

    rng = np.random.default_rng(42)
    video_ids = ["train_v", "val_v", "test_v"]
    frames_by_video = {vid: rng.random((7, 1, 32, 32), dtype=np.float32) for vid in video_ids}
    frames = np.concatenate([frames_by_video[vid] for vid in video_ids], axis=0)
    frame_video_ids = np.asarray([vid for vid in video_ids for _ in range(7)])

    def make_dataset(name: str, horizon_frames: int) -> dict:
        windows = []
        targets = []
        window_video_ids = []
        for vid in video_ids:
            video_frames = frames_by_video[vid]
            for i in range(3):
                windows.append(video_frames[i : i + 3])
                targets.append(video_frames[i + 3 + horizon_frames - 1])
                window_video_ids.append(vid)
        arrays = tmp_path / name / "arrays.npz"
        arrays.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            arrays,
            frames=frames,
            frame_video_ids=frame_video_ids,
            frame_labels=np.asarray(["left"] * len(frame_video_ids)),
            windows=np.stack(windows).astype(np.float32),
            targets=np.stack(targets).astype(np.float32),
            window_video_ids=np.asarray(window_video_ids),
            window_labels=np.asarray(["left"] * len(window_video_ids)),
        )
        return {
            "array_path": str(arrays),
            "windowing": {"window_frames": 3, "prediction_horizon_frames": horizon_frames, "prediction_horizon_sec": horizon_frames / 50.0},
            "splits": {
                "split_method": "stratified_by_label",
                "train_video_ids": ["train_v"],
                "val_video_ids": ["val_v"],
                "test_video_ids": ["test_v"],
            },
        }

    h1 = make_dataset("w3_s1_h1", 1)
    h2 = make_dataset("w3_s1_h2", 2)
    ae = train_autoencoder(dataset=h1, out_dir=tmp_path / "ae", latent_dim=4, base_channels=4, epochs=1, batch_size=4)
    run = evaluate_shared_multi_horizon_linear_latent(
        datasets={"w3_s1_h1": h1, "w3_s1_h2": h2},
        autoencoder_run=ae,
        out_dir=tmp_path / "shared_linear",
        prediction_target="delta",
        alphas=[0.0, 0.1],
        batch_size=3,
        device="cpu",
    )
    metrics = json.loads(Path(run["metrics_path"]).read_text(encoding="utf-8"))

    assert run["model_kind"] == "shared_multi_horizon_linear_latent"
    assert run["shared_horizons_frames"] == [1, 2]
    assert metrics["prediction_target"] == "delta"
    assert set(metrics["per_horizon_metrics"]) == {"w3_s1_h1", "w3_s1_h2"}
    assert metrics["per_horizon_metrics"]["w3_s1_h1"]["test_window_count"] == 3
    assert metrics["per_horizon_metrics"]["w3_s1_h2"]["test_window_count"] == 3
    assert (tmp_path / "shared_linear" / "w3_s1_h1" / "prediction_examples.json").is_file()
    assert (tmp_path / "shared_linear" / "w3_s1_h2" / "prediction_examples.png").is_file()




def test_shared_multi_horizon_latent_transformer_reports_each_horizon(tmp_path):
    from neurobench.dynamics.multi_horizon_neural import train_shared_multi_horizon_latent_transformer
    from neurobench.dynamics.train import train_autoencoder

    rng = np.random.default_rng(44)
    video_ids = ["train_v", "val_v", "test_v"]
    frames_by_video = {vid: rng.random((7, 1, 32, 32), dtype=np.float32) for vid in video_ids}
    frames = np.concatenate([frames_by_video[vid] for vid in video_ids], axis=0)
    frame_video_ids = np.asarray([vid for vid in video_ids for _ in range(7)])

    def make_dataset(name: str, horizon_frames: int) -> dict:
        windows = []
        targets = []
        window_video_ids = []
        for vid in video_ids:
            video_frames = frames_by_video[vid]
            for i in range(3):
                windows.append(video_frames[i : i + 3])
                targets.append(video_frames[i + 3 + horizon_frames - 1])
                window_video_ids.append(vid)
        arrays = tmp_path / name / "arrays.npz"
        arrays.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            arrays,
            frames=frames,
            frame_video_ids=frame_video_ids,
            frame_labels=np.asarray(["left"] * len(frame_video_ids)),
            windows=np.stack(windows).astype(np.float32),
            targets=np.stack(targets).astype(np.float32),
            window_video_ids=np.asarray(window_video_ids),
            window_labels=np.asarray(["left"] * len(window_video_ids)),
        )
        return {
            "array_path": str(arrays),
            "windowing": {"window_frames": 3, "prediction_horizon_frames": horizon_frames, "prediction_horizon_sec": horizon_frames / 50.0},
            "splits": {
                "split_method": "stratified_by_label",
                "train_video_ids": ["train_v"],
                "val_video_ids": ["val_v"],
                "test_video_ids": ["test_v"],
            },
        }

    h1 = make_dataset("w3_s1_h1", 1)
    h2 = make_dataset("w3_s1_h2", 2)
    ae = train_autoencoder(dataset=h1, out_dir=tmp_path / "ae_xfmr", latent_dim=4, base_channels=4, epochs=1, batch_size=4)
    progress_messages = []
    run = train_shared_multi_horizon_latent_transformer(
        datasets={"w3_s1_h1": h1, "w3_s1_h2": h2},
        autoencoder_run=ae,
        out_dir=tmp_path / "shared_transformer",
        model_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        epochs=1,
        batch_size=3,
        learning_rate=0.001,
        prediction_target="delta",
        seed=7,
        device="cpu",
        evaluation_batch_size=2,
        progress=progress_messages.append,
        progress_interval_epochs=1,
    )
    metrics = json.loads(Path(run["metrics_path"]).read_text(encoding="utf-8"))

    assert run["model_kind"] == "shared_multi_horizon_latent_transformer"
    assert run["shared_horizons_frames"] == [1, 2]
    assert metrics["model_family"] == "multi_horizon_latent_transformer"
    assert metrics["decoded_evaluation_mode"] == "chunked"
    assert metrics["evaluation_batch_size"] == 2
    assert set(metrics["per_horizon_metrics"]) == {"w3_s1_h1", "w3_s1_h2"}
    assert metrics["per_horizon_metrics"]["w3_s1_h1"]["test_window_count"] == 3
    assert Path(run["checkpoint_path"]).is_file()
    assert (tmp_path / "shared_transformer" / "multi_horizon_transformer_progress.jsonl").is_file()
    latest = json.loads((tmp_path / "shared_transformer" / "multi_horizon_transformer_progress_latest.json").read_text(encoding="utf-8"))
    assert latest["phase"] == "complete"
    assert any("shared-transformer" in message for message in progress_messages)
    examples = json.loads((tmp_path / "shared_transformer" / "w3_s1_h1" / "prediction_examples.json").read_text(encoding="utf-8"))
    assert examples["examples"][0]["window_start_index"] == 0
    assert examples["examples"][0]["target_frame_index"] == 3
    clips = json.loads((tmp_path / "shared_transformer" / "w3_s1_h1" / "prediction_clip_examples.json").read_text(encoding="utf-8"))
    assert clips["clip_count"] >= 1
    assert clips["clips"][0]["frame_count"] >= 2
    assert "predicted_next" in clips["clips"][0]["frames"][0]
    review_metrics = json.loads((tmp_path / "shared_transformer" / "w3_s1_h1" / "per_horizon_metrics_for_review.json").read_text(encoding="utf-8"))
    assert review_metrics["prediction_examples_path"].endswith("prediction_examples.json")
    assert review_metrics["prediction_clip_examples_path"].endswith("prediction_clip_examples.json")
    assert metrics["per_horizon_metrics"]["w3_s1_h1"]["per_horizon_metrics_for_review_path"].endswith("per_horizon_metrics_for_review.json")


def test_shared_multi_horizon_latent_gru_reports_each_horizon(tmp_path):
    from neurobench.dynamics.multi_horizon_neural import train_shared_multi_horizon_latent_gru
    from neurobench.dynamics.train import train_autoencoder

    rng = np.random.default_rng(43)
    video_ids = ["train_v", "val_v", "test_v"]
    frames_by_video = {vid: rng.random((7, 1, 32, 32), dtype=np.float32) for vid in video_ids}
    frames = np.concatenate([frames_by_video[vid] for vid in video_ids], axis=0)
    frame_video_ids = np.asarray([vid for vid in video_ids for _ in range(7)])

    def make_dataset(name: str, horizon_frames: int) -> dict:
        windows = []
        targets = []
        window_video_ids = []
        for vid in video_ids:
            video_frames = frames_by_video[vid]
            for i in range(3):
                windows.append(video_frames[i : i + 3])
                targets.append(video_frames[i + 3 + horizon_frames - 1])
                window_video_ids.append(vid)
        arrays = tmp_path / name / "arrays.npz"
        arrays.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            arrays,
            frames=frames,
            frame_video_ids=frame_video_ids,
            frame_labels=np.asarray(["left"] * len(frame_video_ids)),
            windows=np.stack(windows).astype(np.float32),
            targets=np.stack(targets).astype(np.float32),
            window_video_ids=np.asarray(window_video_ids),
            window_labels=np.asarray(["left"] * len(window_video_ids)),
        )
        return {
            "array_path": str(arrays),
            "windowing": {"window_frames": 3, "prediction_horizon_frames": horizon_frames, "prediction_horizon_sec": horizon_frames / 50.0},
            "splits": {
                "split_method": "stratified_by_label",
                "train_video_ids": ["train_v"],
                "val_video_ids": ["val_v"],
                "test_video_ids": ["test_v"],
            },
        }

    h1 = make_dataset("w3_s1_h1", 1)
    h2 = make_dataset("w3_s1_h2", 2)
    ae = train_autoencoder(dataset=h1, out_dir=tmp_path / "ae_gru", latent_dim=4, base_channels=4, epochs=1, batch_size=4)
    progress_messages = []
    run = train_shared_multi_horizon_latent_gru(
        datasets={"w3_s1_h1": h1, "w3_s1_h2": h2},
        autoencoder_run=ae,
        out_dir=tmp_path / "shared_gru",
        hidden_dim=5,
        num_layers=1,
        epochs=1,
        batch_size=3,
        learning_rate=0.001,
        prediction_target="delta",
        seed=7,
        device="cpu",
        evaluation_batch_size=2,
        progress=progress_messages.append,
        progress_interval_epochs=1,
    )
    metrics = json.loads(Path(run["metrics_path"]).read_text(encoding="utf-8"))

    assert run["model_kind"] == "shared_multi_horizon_latent_gru"
    assert run["shared_horizons_frames"] == [1, 2]
    assert metrics["model_family"] == "multi_horizon_latent_gru"
    assert metrics["prediction_target"] == "delta"
    assert len(metrics["training_loss"]) == 1
    assert metrics["decoded_evaluation_mode"] == "chunked"
    assert metrics["evaluation_batch_size"] == 2
    assert set(metrics["per_horizon_metrics"]) == {"w3_s1_h1", "w3_s1_h2"}
    assert metrics["per_horizon_metrics"]["w3_s1_h1"]["test_window_count"] == 3
    assert metrics["per_horizon_metrics"]["w3_s1_h2"]["test_window_count"] == 3
    assert metrics["per_horizon_metrics"]["w3_s1_h1"]["decoded_evaluation_mode"] == "chunked"
    assert metrics["per_horizon_metrics"]["w3_s1_h1"]["evaluation_batch_size"] == 2
    assert Path(run["checkpoint_path"]).is_file()
    assert metrics["progress_log_path"].endswith("multi_horizon_gru_progress.jsonl")
    assert metrics["progress_latest_path"].endswith("multi_horizon_gru_progress_latest.json")
    progress_path = tmp_path / "shared_gru" / "multi_horizon_gru_progress.jsonl"
    latest_path = tmp_path / "shared_gru" / "multi_horizon_gru_progress_latest.json"
    assert progress_path.is_file()
    assert latest_path.is_file()
    progress_events = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    progress_phases = [event["phase"] for event in progress_events]
    assert progress_phases[0] == "start"
    assert "encode" in progress_phases
    assert "train_epoch" in progress_phases
    assert "evaluate_done" in progress_phases
    assert progress_phases[-1] == "complete"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    assert latest["phase"] == "complete"
    assert latest["metrics_path"] == run["metrics_path"]
    assert any("train_epoch" in message for message in progress_messages)
    examples = json.loads((tmp_path / "shared_gru" / "w3_s1_h1" / "prediction_examples.json").read_text(encoding="utf-8"))
    assert examples["examples"][0]["window_start_index"] == 0
    assert examples["examples"][0]["target_frame_index"] == 3
    clips = json.loads((tmp_path / "shared_gru" / "w3_s1_h1" / "prediction_clip_examples.json").read_text(encoding="utf-8"))
    assert clips["clip_count"] >= 1
    assert clips["clips"][0]["frame_count"] >= 2
    review_metrics = json.loads((tmp_path / "shared_gru" / "w3_s1_h1" / "per_horizon_metrics_for_review.json").read_text(encoding="utf-8"))
    assert review_metrics["prediction_examples_path"].endswith("prediction_examples.json")
    assert review_metrics["prediction_clip_examples_path"].endswith("prediction_clip_examples.json")
    assert metrics["per_horizon_metrics"]["w3_s1_h1"]["per_horizon_metrics_for_review_path"].endswith("per_horizon_metrics_for_review.json")
    assert (tmp_path / "shared_gru" / "w3_s1_h2" / "prediction_examples.png").is_file()
