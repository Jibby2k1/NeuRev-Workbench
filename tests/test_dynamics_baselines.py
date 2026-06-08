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
