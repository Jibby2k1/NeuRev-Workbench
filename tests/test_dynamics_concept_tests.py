import argparse
import json
from pathlib import Path

import numpy as np

from neurobench.cli import dynamics as dynamics_cli
from neurobench.dynamics import concept_tests


def test_concept_tests_imports_structured_error_helpers():
    assert callable(concept_tests.structured_prediction_error_metrics)
    assert callable(concept_tests.promote_structured_error_metrics)


def test_concept_prediction_examples_include_review_metadata(tmp_path):
    windows = np.array(
        [
            [[[[0.0, 0.1], [0.2, 0.3]]], [[[0.4, 0.5], [0.6, 0.7]]]],
            [[[[0.1, 0.2], [0.3, 0.4]]], [[[0.5, 0.6], [0.7, 0.8]]]],
        ],
        dtype=np.float32,
    )
    targets = np.array([[[[0.45, 0.55], [0.65, 0.75]]], [[[0.55, 0.65], [0.75, 0.85]]]], dtype=np.float32)
    pred = targets - 0.05
    path = concept_tests._write_prediction_examples(
        tmp_path / "prediction_examples.json",
        windows=windows,
        targets=targets,
        pred=pred,
        video_ids=np.array(["vid_train", "vid_test"]),
        splits={"train_video_ids": ["vid_train"], "test_video_ids": ["vid_test"]},
        windowing={"window_frames": 2, "prediction_horizon_frames": 1, "effective_frame_rate_hz": 50.0},
    )

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    first = payload["examples"][0]
    second = payload["examples"][1]

    assert payload["schema_version"] == 1
    assert first["input_last"] == [[0.4, 0.5], [0.6, 0.7]]
    assert first["target_next"] == [[0.45, 0.55], [0.65, 0.75]]
    assert first["predicted_next"] == [[0.4, 0.5], [0.6, 0.7]]
    assert first["split"] == "train"
    assert second["video_id"] == "vid_test"
    assert second["split"] == "test"
    assert second["prediction_horizon_frames"] == 1


def test_concept_prediction_metrics_can_reference_prediction_examples(tmp_path):
    windows = np.array([[[[[0.0, 0.0], [0.0, 0.0]]], [[[0.2, 0.2], [0.2, 0.2]]]]], dtype=np.float32)
    targets = np.array([[[[0.3, 0.3], [0.3, 0.3]]]], dtype=np.float32)
    pred = np.array([[[[0.25, 0.25], [0.25, 0.25]]]], dtype=np.float32)
    metrics = concept_tests._prediction_metrics(
        pred=pred,
        targets=targets,
        windows=windows,
        video_ids=np.array(["vid_test"]),
        splits={"test_video_ids": ["vid_test"]},
        objective="unit",
        training_loss=[0.1],
        train_count=0,
        active_threshold=0.02,
        active_weight=0.0,
    )
    examples_path = concept_tests._write_prediction_examples(
        tmp_path / "prediction_examples.json",
        windows=windows,
        targets=targets,
        pred=pred,
        video_ids=np.array(["vid_test"]),
        splits={"test_video_ids": ["vid_test"]},
        windowing={},
    )
    metrics["prediction_examples_path"] = str(examples_path)

    assert metrics["test_improvement_over_persistence_mse"] > 0
    assert Path(metrics["prediction_examples_path"]).is_file()

def test_backfill_spatial_prediction_examples_updates_metrics(tmp_path):
    torch = concept_tests._torch()
    arrays_path = tmp_path / "arrays.npz"
    windows = np.linspace(0.0, 0.6, num=2 * 2 * 1 * 4 * 4, dtype=np.float32).reshape(2, 2, 1, 4, 4)
    targets = np.clip(windows[:, -1] + 0.05, 0.0, 1.0).astype(np.float32)
    np.savez(arrays_path, windows=windows, targets=targets, window_video_ids=np.array(["vid_train", "vid_test"]))
    dataset = {
        "array_path": str(arrays_path),
        "splits": {"train_video_ids": ["vid_train"], "test_video_ids": ["vid_test"]},
        "windowing": {"window_frames": 2, "prediction_horizon_frames": 1, "effective_frame_rate_hz": 50.0},
    }
    run_dir = tmp_path / "run" / "convgru_pixel_residual_mse"
    run_dir.mkdir(parents=True)
    model = concept_tests._build_spatial_pixel_model(
        architecture="convgru_pixel",
        input_channels=1,
        window_frames=2,
        hidden_channels=2,
        num_layers=1,
        residual_scale=0.1,
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": "convgru_pixel",
            "input_channels": 1,
            "window_frames": 2,
            "hidden_channels": 2,
            "num_layers": 1,
            "residual_scale": 0.1,
        },
        run_dir / "concept_checkpoint.pt",
    )
    (run_dir / "concept_metrics.json").write_text(json.dumps({"schema_version": 1, "objective": "unit"}), encoding="utf-8")

    summary = concept_tests.backfill_spatial_prediction_examples(
        dataset=dataset,
        run_dir=run_dir,
        batch_size=1,
        max_examples=2,
        device="cpu",
    )

    examples_path = Path(summary["prediction_examples_path"])
    examples = json.loads(examples_path.read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "concept_metrics.json").read_text(encoding="utf-8"))
    backfill = json.loads((run_dir / "prediction_examples_backfill.json").read_text(encoding="utf-8"))

    assert summary["metrics_updated"] is True
    assert summary["architecture"] == "convgru_pixel"
    assert examples["schema_version"] == 1
    assert len(examples["examples"]) == 2
    assert examples["examples"][0]["video_id"] == "vid_test"
    assert examples["examples"][0]["split"] == "test"
    assert metrics["prediction_examples_path"] == str(examples_path)
    assert "prediction_examples_backfilled_at" in metrics
    assert backfill["dataset_window_count"] == 2




def test_backfill_spatial_prediction_examples_can_backfill_per_video_metrics(tmp_path):
    torch = concept_tests._torch()
    arrays_path = tmp_path / "arrays.npz"
    windows = np.linspace(0.0, 0.9, num=3 * 2 * 1 * 4 * 4, dtype=np.float32).reshape(3, 2, 1, 4, 4)
    targets = np.clip(windows[:, -1] + 0.03, 0.0, 1.0).astype(np.float32)
    np.savez(
        arrays_path,
        windows=windows,
        targets=targets,
        window_video_ids=np.array(["vid_train", "vid_test_left", "vid_test_right"]),
    )
    dataset = {
        "array_path": str(arrays_path),
        "splits": {"train_video_ids": ["vid_train"], "test_video_ids": ["vid_test_left", "vid_test_right"]},
        "windowing": {"window_frames": 2, "prediction_horizon_frames": 1, "effective_frame_rate_hz": 50.0},
    }
    run_dir = tmp_path / "run" / "convgru_pixel_residual_mse"
    run_dir.mkdir(parents=True)
    model = concept_tests._build_spatial_pixel_model(
        architecture="convgru_pixel",
        input_channels=1,
        window_frames=2,
        hidden_channels=2,
        num_layers=1,
        residual_scale=0.1,
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": "convgru_pixel",
            "input_channels": 1,
            "window_frames": 2,
            "hidden_channels": 2,
            "num_layers": 1,
            "residual_scale": 0.1,
            "objective": "unit_backfill",
        },
        run_dir / "concept_checkpoint.pt",
    )
    (run_dir / "concept_metrics.json").write_text(
        json.dumps({"schema_version": 1, "objective": "unit_backfill", "training_loss": [0.5], "model_family": "pixel_convgru"}),
        encoding="utf-8",
    )

    summary = concept_tests.backfill_spatial_prediction_examples(
        dataset=dataset,
        run_dir=run_dir,
        batch_size=1,
        max_examples=2,
        device="cpu",
        backfill_metrics=True,
    )

    metrics = json.loads((run_dir / "concept_metrics.json").read_text(encoding="utf-8"))

    assert summary["prediction_metrics_backfilled"] is True
    assert summary["metrics_evaluation_window_count"] == 3
    assert summary["metrics_split_window_counts"]["test"] == 2
    assert summary["metrics_per_video_counts"]["test"] == 2
    assert metrics["evaluation_window_count"] == 3
    assert metrics["training_loss"] == [0.5]
    assert metrics["model_family"] == "pixel_convgru"
    assert "structured_error_metrics" in metrics
    assert set(metrics["split_metrics"]["test"]["per_video"]) == {"vid_test_left", "vid_test_right"}
    assert metrics["prediction_metrics_backfilled_at"] == summary["created_at"]



def test_backfill_spatial_prediction_examples_dry_run_estimates_without_writing(tmp_path):
    torch = concept_tests._torch()
    arrays_path = tmp_path / "arrays.npz"
    windows = np.linspace(0.0, 0.6, num=2 * 2 * 1 * 4 * 4, dtype=np.float32).reshape(2, 2, 1, 4, 4)
    targets = np.clip(windows[:, -1] + 0.05, 0.0, 1.0).astype(np.float32)
    np.savez(
        arrays_path,
        windows=windows,
        targets=targets,
        window_video_ids=np.array(["vid_train", "vid_test"]),
        window_labels=np.array(["left", "right"]),
    )
    dataset = {
        "array_path": str(arrays_path),
        "splits": {"train_video_ids": ["vid_train"], "test_video_ids": ["vid_test"]},
        "windowing": {"window_frames": 2, "prediction_horizon_frames": 1, "effective_frame_rate_hz": 50.0},
    }
    run_dir = tmp_path / "run" / "convgru_pixel_residual_mse"
    run_dir.mkdir(parents=True)
    model = concept_tests._build_spatial_pixel_model(
        architecture="convgru_pixel",
        input_channels=1,
        window_frames=2,
        hidden_channels=2,
        num_layers=1,
        residual_scale=0.1,
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": "convgru_pixel",
            "input_channels": 1,
            "window_frames": 2,
            "hidden_channels": 2,
            "num_layers": 1,
            "residual_scale": 0.1,
        },
        run_dir / "concept_checkpoint.pt",
    )
    metrics_path = run_dir / "concept_metrics.json"
    metrics_path.write_text(json.dumps({"schema_version": 1, "objective": "unit"}), encoding="utf-8")

    summary = concept_tests.backfill_spatial_prediction_examples(
        dataset=dataset,
        run_dir=run_dir,
        batch_size=1,
        max_examples=2,
        device="cuda",
        backfill_metrics=True,
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert summary["dataset_window_count"] == 2
    assert summary["dataset_arrays"]["windows"]["shape"] == [2, 2, 1, 4, 4]
    assert summary["example_count"] == 2
    assert summary["source_indices"] == [1, 0]
    assert summary["example_preview"] == [
        {"index": 1, "video_id": "vid_test", "split": "test"},
        {"index": 0, "video_id": "vid_train", "split": "train"},
    ]
    assert summary["split_window_counts"] == {"test": 1, "train": 1}
    assert summary["split_video_counts"] == {"test": 1, "train": 1}
    assert summary["split_label_counts"] == {"test": {"right": 1}, "train": {"left": 1}}
    assert summary["split_top_videos"] == {
        "test": [{"video_id": "vid_test", "window_count": 1}],
        "train": [{"video_id": "vid_train", "window_count": 1}],
    }
    assert summary["estimated_uncompressed_bytes"] > 0
    assert summary["estimated_uncompressed_gib"] == round(summary["estimated_uncompressed_bytes"] / 1024**3, 3)
    assert summary["estimated_compressed_bytes"] > 0
    assert summary["estimated_compressed_gib"] == round(summary["estimated_compressed_bytes"] / 1024**3, 3)
    assert summary["requested_batch_size"] == 1
    assert summary["estimated_example_batches"] == 2
    assert summary["estimated_metric_batches"] == 2
    assert summary["would_update_metrics"] is True
    assert summary["would_backfill_metrics"] is True
    assert summary["would_write_files"] == [
        str(run_dir / "prediction_examples.json"),
        str(run_dir / "prediction_examples_backfill.json"),
        str(metrics_path),
    ]
    assert summary["metrics_updated"] is False
    assert summary["prediction_metrics_backfilled"] is False
    assert not (run_dir / "prediction_examples.json").exists()
    assert not (run_dir / "prediction_examples_backfill.json").exists()
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == {"schema_version": 1, "objective": "unit"}



def test_backfill_cli_json_output_is_machine_readable(monkeypatch, tmp_path, capsys):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"

    def fake_load_json(path):
        assert Path(path) == dataset_path
        return {"loaded": True}

    def fake_backfill(**kwargs):
        assert kwargs["dataset"] == {"loaded": True}
        assert kwargs["run_dir"] == run_dir
        assert kwargs["dry_run"] is True
        return {
            "schema_version": 1,
            "dry_run": True,
            "prediction_examples_path": str(run_dir / "prediction_examples.json"),
            "split_window_counts": {"test": 2},
            "metrics_updated": False,
            "prediction_metrics_backfilled": False,
        }

    monkeypatch.setattr(dynamics_cli, "load_json", fake_load_json)
    monkeypatch.setattr(dynamics_cli, "backfill_spatial_prediction_examples", fake_backfill)

    rc = dynamics_cli.dynamics_backfill_concept_examples_command(
        argparse.Namespace(
            dataset=dataset_path,
            run_dir=run_dir,
            checkpoint=None,
            metrics=None,
            out=None,
            batch_size=16,
            max_examples=3,
            device="cuda",
            no_update_metrics=False,
            backfill_metrics=True,
            dry_run=True,
            json=True,
            markdown_out=None,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["split_window_counts"] == {"test": 2}


def test_backfill_cli_writes_markdown_preflight(monkeypatch, tmp_path, capsys):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    markdown_out = tmp_path / "preflight" / "summary.md"

    monkeypatch.setattr(dynamics_cli, "load_json", lambda path: {"loaded": Path(path) == dataset_path})

    def fake_backfill(**kwargs):
        assert kwargs["dataset"] == {"loaded": True}
        assert kwargs["run_dir"] == run_dir
        assert kwargs["backfill_metrics"] is True
        assert kwargs["dry_run"] is True
        return {
            "schema_version": 1,
            "created_at": "2026-06-10T21:40:21+00:00",
            "dry_run": True,
            "run_dir": str(run_dir),
            "checkpoint_path": str(run_dir / "concept_checkpoint.pt"),
            "metrics_path": str(run_dir / "concept_metrics.json"),
            "prediction_examples_path": str(run_dir / "prediction_examples.json"),
            "architecture": "convgru_pixel",
            "dataset_window_count": 3,
            "estimated_uncompressed_gib": 0.001,
            "estimated_compressed_gib": 0.001,
            "requested_batch_size": 2,
            "estimated_example_batches": 1,
            "estimated_metric_batches": 2,
            "would_update_metrics": True,
            "would_backfill_metrics": True,
            "would_write_files": [str(run_dir / "prediction_examples.json"), str(run_dir / "concept_metrics.json")],
            "split_window_counts": {"test": 2, "train": 1},
            "split_video_counts": {"test": 1, "train": 1},
            "split_label_counts": {"test": {"left": 2}, "train": {"right": 1}},
            "split_top_videos": {"test": [{"video_id": "vid_test", "window_count": 2}]},
            "example_preview": [{"index": 2, "video_id": "vid_test", "split": "test"}],
            "metrics_updated": False,
            "prediction_metrics_backfilled": False,
        }

    monkeypatch.setattr(dynamics_cli, "backfill_spatial_prediction_examples", fake_backfill)

    rc = dynamics_cli.dynamics_backfill_concept_examples_command(
        argparse.Namespace(
            dataset=dataset_path,
            run_dir=run_dir,
            checkpoint=None,
            metrics=None,
            out=None,
            batch_size=2,
            max_examples=1,
            device="cuda",
            no_update_metrics=False,
            backfill_metrics=True,
            dry_run=True,
            json=True,
            markdown_out=markdown_out,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    markdown = markdown_out.read_text(encoding="utf-8")
    assert "# Concept Prediction Backfill Preflight" in markdown
    assert "- Dataset windows: `3`" in markdown
    assert "- Split windows: test=2, train=1" in markdown
    assert "Selected examples: 2:test:vid_test" in markdown
    assert str(run_dir / "concept_metrics.json") in markdown
