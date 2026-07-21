from __future__ import annotations

import unittest
import numpy as np


class DynamicsBaselineTests(unittest.TestCase):
    def test_persistence_metrics_match_hand_checked_values(self):
        from neurobench.dynamics.baselines import evaluate_baselines_from_arrays

        windows = np.asarray([[[[[1.0]]], [[[2.0]]]], [[[[2.0]]], [[[4.0]]]]], dtype=np.float32)
        targets = np.asarray([[[[3.0]]], [[[5.0]]]], dtype=np.float32)
        metrics = evaluate_baselines_from_arrays({"windows": windows, "targets": targets, "window_video_ids": np.asarray(["a", "b"]), "window_labels": np.asarray(["left", "right"])})
        self.assertAlmostEqual(metrics["persistence"]["mse"], 1.0)
        self.assertAlmostEqual(metrics["persistence"]["mae"], 1.0)
        self.assertIn("linear_extrapolation", metrics)
        self.assertIn("mean_delta", metrics)

    def test_linear_and_mean_delta_predictions_are_clipped(self):
        from neurobench.dynamics.baselines import baseline_prediction

        windows = np.asarray([[[[[0.2]]], [[[0.4]]], [[[0.6]]]], [[[[0.6]]], [[[0.9]]], [[[0.95]]]]], dtype=np.float32)
        linear = baseline_prediction(windows, "linear_extrapolation")
        mean_delta = baseline_prediction(windows, "mean_delta")

        self.assertAlmostEqual(float(linear[0, 0, 0, 0]), 0.8, places=6)
        self.assertAlmostEqual(float(mean_delta[0, 0, 0, 0]), 0.8, places=6)
        self.assertAlmostEqual(float(linear[1, 0, 0, 0]), 1.0)
        self.assertLessEqual(float(mean_delta[1, 0, 0, 0]), 1.0)

    def test_kinetics_baselines_decay_with_horizon_and_rate(self):
        from neurobench.dynamics.baselines import baseline_prediction

        windows = np.asarray([[[[[0.1]]], [[[0.1]]], [[[0.9]]]]], dtype=np.float32)
        slow = baseline_prediction(windows, "exponential_decay_10hz", prediction_horizon_frames=2, frame_rate_hz=50.0)
        fast = baseline_prediction(windows, "exponential_decay_30hz", prediction_horizon_frames=2, frame_rate_hz=50.0)
        ar1 = baseline_prediction(windows, "ar1_per_cell", prediction_horizon_frames=2, frame_rate_hz=50.0)

        self.assertEqual(slow.shape, windows[:, -1].shape)
        self.assertLess(float(fast[0, 0, 0, 0]), float(slow[0, 0, 0, 0]))
        self.assertGreaterEqual(float(ar1[0, 0, 0, 0]), 0.0)
        self.assertLessEqual(float(ar1[0, 0, 0, 0]), 1.0)


def test_evaluate_kinetics_baselines_writes_sweep_compatible_metrics(tmp_path):
    from neurobench.dynamics.kinetics_baselines import evaluate_kinetics_baselines
    from neurobench.dynamics.overnight_sweep import collect_metric_rows

    array_path = tmp_path / "arrays.npz"
    windows = np.asarray(
        [
            [[[[0.1]]], [[[0.2]]], [[[0.4]]]],
            [[[[0.3]]], [[[0.4]]], [[[0.5]]]],
            [[[[0.2]]], [[[0.2]]], [[[0.8]]]],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([[[[0.45]]], [[[0.55]]], [[[0.5]]]], dtype=np.float32)
    np.savez(
        array_path,
        windows=windows,
        targets=targets,
        window_video_ids=np.asarray(["train_a", "val_a", "test_a"]),
        window_labels=np.asarray(["left", "neutral", "right"]),
    )
    dataset = {
        "array_path": str(array_path),
        "windowing": {"effective_frame_rate_hz": 50.0, "prediction_horizon_frames": 2, "prediction_horizon_sec": 0.04},
        "splits": {"train_video_ids": ["train_a"], "val_video_ids": ["val_a"], "test_video_ids": ["test_a"]},
    }

    summary = evaluate_kinetics_baselines(
        datasets={"demo_h2": dataset},
        out_dir=tmp_path / "kinetics",
        baseline_names=("exponential_decay_10hz", "ar1_per_cell"),
    )
    rows = collect_metric_rows(tmp_path / "kinetics")
    metrics = sorted(rows, key=lambda row: row["experiment_id"])[0]

    assert summary["experiment_count"] == 2
    assert (tmp_path / "kinetics" / "sweep_manifest.json").is_file()
    assert (tmp_path / "kinetics" / "kinetics_baseline_summary.md").is_file()
    assert len(rows) == 2
    assert {row["model_family"] for row in rows} == {"kinetics_baseline"}
    assert metrics["hyperparameter_summary"]
    assert metrics["test_improvement_over_persistence_mse"] is not None
