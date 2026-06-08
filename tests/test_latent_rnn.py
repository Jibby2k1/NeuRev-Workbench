from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


class LatentRnnTests(unittest.TestCase):
    def test_tiny_train_exports_persistence_comparison(self):
        from neurobench.dynamics.train import train_autoencoder, train_latent_rnn

        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            frames=np.random.default_rng(2).random((10,1,32,32), dtype=np.float32)
            windows=np.stack([frames[i:i+3] for i in range(6)]).astype(np.float32)
            targets=np.stack([frames[i+3] for i in range(6)]).astype(np.float32)
            arrays=root/"arrays.npz"
            np.savez(arrays, frames=frames, frame_video_ids=np.asarray(["v"]*10), frame_labels=np.asarray(["left"]*10), windows=windows, targets=targets, window_video_ids=np.asarray(["v"]*6), window_labels=np.asarray(["left"]*6))
            dataset={"array_path":str(arrays),"windowing":{"prediction_horizon_frames":1}}
            ae=train_autoencoder(dataset=dataset, out_dir=root/"ae", latent_dim=4, epochs=1, batch_size=4)
            run=train_latent_rnn(dataset=dataset, autoencoder_run=ae, out_dir=root/"rnn", window_frames=3, hidden_dim=8, epochs=1, batch_size=3)
            baseline_exists = Path(run["baseline_metrics_path"]).is_file()
            examples_exist = Path(run["prediction_examples_path"]).is_file()
            metrics = __import__("json").loads(Path(run["metrics_path"]).read_text())

        self.assertTrue(baseline_exists)
        self.assertTrue(examples_exist)
        self.assertEqual(run["rnn_objective"], "predict_next_latent_code")
        self.assertEqual(metrics["objective"], "next_code_mse")
        self.assertEqual(metrics["latent_code_normalization"], "standard_score_per_dimension")
        self.assertEqual(metrics["decoded_output_normalization"], "sigmoid_unit_interval")
        self.assertIn("latent_code_mse", metrics)
        self.assertIn("latent_code_raw_mse", metrics)
        self.assertIn("decoded_prediction_mse", metrics)



    def test_delta_latent_rnn_reports_delta_objective(self):
        import json
        from neurobench.dynamics.train import train_autoencoder, train_latent_rnn

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = np.random.default_rng(4).random((10, 1, 32, 32), dtype=np.float32)
            windows = np.stack([frames[i : i + 3] for i in range(6)]).astype(np.float32)
            targets = np.stack([frames[i + 3] for i in range(6)]).astype(np.float32)
            arrays = root / "arrays.npz"
            np.savez(
                arrays,
                frames=frames,
                frame_video_ids=np.asarray(["v"] * 10),
                frame_labels=np.asarray(["left"] * 10),
                windows=windows,
                targets=targets,
                window_video_ids=np.asarray(["v"] * 6),
                window_labels=np.asarray(["left"] * 6),
            )
            dataset = {"array_path": str(arrays), "windowing": {"prediction_horizon_frames": 1}}
            ae = train_autoencoder(dataset=dataset, out_dir=root / "ae", latent_dim=4, base_channels=8, epochs=1, batch_size=4)
            run = train_latent_rnn(
                dataset=dataset,
                autoencoder_run=ae,
                out_dir=root / "rnn",
                window_frames=3,
                hidden_dim=8,
                epochs=1,
                batch_size=3,
                prediction_target="delta",
            )
            metrics = json.loads(Path(run["metrics_path"]).read_text())

        self.assertEqual(run["rnn_objective"], "predict_next_delta_latent_code")
        self.assertEqual(run["prediction_target"], "delta")
        self.assertEqual(metrics["objective"], "next_delta_code_mse")
        self.assertEqual(metrics["prediction_target"], "delta")
        self.assertEqual(run["extras"]["predicted_code_space"], "standardized_latent_delta")

    def test_linear_latent_baseline_exports_split_metrics(self):
        import json
        from neurobench.dynamics.linear import evaluate_linear_latent_baseline
        from neurobench.dynamics.train import train_autoencoder

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(5)
            frames_by_video = {vid: rng.random((6, 1, 32, 32), dtype=np.float32) for vid in ["train_v", "val_v", "test_v"]}
            frames = np.concatenate([frames_by_video[vid] for vid in ["train_v", "val_v", "test_v"]], axis=0)
            frame_video_ids = np.asarray([vid for vid in ["train_v", "val_v", "test_v"] for _ in range(6)])
            windows = []
            targets = []
            window_video_ids = []
            for vid, video_frames in frames_by_video.items():
                for i in range(3):
                    windows.append(video_frames[i : i + 3])
                    targets.append(video_frames[i + 3])
                    window_video_ids.append(vid)
            arrays = root / "arrays.npz"
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
            dataset = {
                "array_path": str(arrays),
                "windowing": {"window_frames": 3, "prediction_horizon_frames": 1},
                "splits": {
                    "split_method": "stratified_by_label",
                    "train_video_ids": ["train_v"],
                    "val_video_ids": ["val_v"],
                    "test_video_ids": ["test_v"],
                },
            }
            ae = train_autoencoder(dataset=dataset, out_dir=root / "ae", latent_dim=4, epochs=1, batch_size=4)
            run = evaluate_linear_latent_baseline(
                dataset=dataset,
                autoencoder_run=ae,
                out_dir=root / "linear",
                prediction_target="delta",
                alphas=[0.0, 0.1],
                batch_size=3,
                device="cpu",
            )
            metrics = json.loads(Path(run["metrics_path"]).read_text())

        self.assertEqual(run["model_kind"], "linear_latent_baseline")
        self.assertEqual(run["prediction_target"], "delta")
        self.assertEqual(metrics["prediction_target"], "delta")
        self.assertIn("best_alpha", metrics)
        self.assertEqual(metrics["val_window_count"], 3)
        self.assertEqual(metrics["test_window_count"], 3)

    def test_video_split_limits_training_and_reports_eval_splits(self):
        import json
        from neurobench.dynamics.train import train_autoencoder, train_latent_rnn

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(12)
            frames_by_video = {vid: rng.random((6, 1, 32, 32), dtype=np.float32) for vid in ["train_v", "val_v", "test_v"]}
            frames = np.concatenate([frames_by_video[vid] for vid in ["train_v", "val_v", "test_v"]], axis=0)
            frame_video_ids = np.asarray([vid for vid in ["train_v", "val_v", "test_v"] for _ in range(6)])
            windows = []
            targets = []
            window_video_ids = []
            for vid, video_frames in frames_by_video.items():
                for i in range(3):
                    windows.append(video_frames[i : i + 3])
                    targets.append(video_frames[i + 3])
                    window_video_ids.append(vid)
            arrays = root / "arrays.npz"
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
            dataset = {
                "array_path": str(arrays),
                "windowing": {"prediction_horizon_frames": 1},
                "splits": {
                    "split_method": "stratified_by_label",
                    "train_video_ids": ["train_v"],
                    "val_video_ids": ["val_v"],
                    "test_video_ids": ["test_v"],
                },
            }
            ae = train_autoencoder(dataset=dataset, out_dir=root / "ae", latent_dim=4, epochs=1, batch_size=4)
            run = train_latent_rnn(dataset=dataset, autoencoder_run=ae, out_dir=root / "rnn", window_frames=3, hidden_dim=8, epochs=1, batch_size=3)
            ae_metrics = json.loads(Path(ae["metrics_path"]).read_text())
            rnn_metrics = json.loads(Path(run["metrics_path"]).read_text())

        self.assertEqual(ae["extras"]["train_frame_count"], 6)
        self.assertEqual(ae_metrics["val_frame_count"], 6)
        self.assertEqual(run["extras"]["train_window_count"], 3)
        self.assertEqual(run["extras"]["evaluation_window_count"], 9)
        self.assertEqual(rnn_metrics["val_window_count"], 3)
        self.assertEqual(rnn_metrics["test_window_count"], 3)
        self.assertIsNotNone(rnn_metrics["val_latent_code_mse"])
        self.assertIsNotNone(rnn_metrics["test_decoded_prediction_mse"])

    def test_latent_rnn_does_not_decode_under_grad(self):
        from neurobench.dynamics import train as train_module
        from neurobench.dynamics.train import train_autoencoder, train_latent_rnn

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = np.random.default_rng(3).random((10, 1, 32, 32), dtype=np.float32)
            windows = np.stack([frames[i : i + 3] for i in range(6)]).astype(np.float32)
            targets = np.stack([frames[i + 3] for i in range(6)]).astype(np.float32)
            arrays = root / "arrays.npz"
            np.savez(
                arrays,
                frames=frames,
                frame_video_ids=np.asarray(["v"] * 10),
                frame_labels=np.asarray(["left"] * 10),
                windows=windows,
                targets=targets,
                window_video_ids=np.asarray(["v"] * 6),
                window_labels=np.asarray(["left"] * 6),
            )
            dataset = {"array_path": str(arrays), "windowing": {"prediction_horizon_frames": 1}}
            ae = train_autoencoder(dataset=dataset, out_dir=root / "ae", latent_dim=4, epochs=1, batch_size=4)
            original_decode = train_module.GridAutoencoder.decode

            def decode_guard(self, z):
                import torch

                if torch.is_grad_enabled():
                    raise AssertionError("decoder should not be in the latent RNN training loss")
                return original_decode(self, z)

            with mock.patch.object(train_module.GridAutoencoder, "decode", decode_guard):
                run = train_latent_rnn(
                    dataset=dataset,
                    autoencoder_run=ae,
                    out_dir=root / "rnn",
                    window_frames=3,
                    hidden_dim=8,
                    epochs=1,
                    batch_size=3,
                )

        self.assertFalse(run["extras"]["decoded_prediction_used_for_training"])
        self.assertEqual(run["extras"]["predicted_code_space"], "standardized_latent")
        self.assertEqual(run["extras"]["decoded_prediction_space"], "unit_interval_grid")

