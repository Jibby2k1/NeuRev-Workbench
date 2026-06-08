from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class DynamicsSweepTests(unittest.TestCase):
    def test_tiny_sweep_ranks_latent_rnn_candidates(self):
        from neurobench.dynamics.sweep import run_latent_dynamics_sweep

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = np.random.default_rng(9).random((10, 1, 32, 32), dtype=np.float32)
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
            dataset = {
                "dataset_id": "tiny",
                "array_path": str(arrays),
                "input_shape": [1, 32, 32],
                "windowing": {"window_frames": 3, "prediction_horizon_frames": 1},
                "splits": {"split_method": "train_all_smoke"},
                "warnings": ["smoke"],
            }
            summary = run_latent_dynamics_sweep(
                dataset=dataset,
                out_dir=root / "sweep",
                latent_dims=[4],
                autoencoder_epochs=[1],
                autoencoder_learning_rates=[0.001],
                autoencoder_batch_size=4,
                rnn_hidden_dims=[8],
                rnn_epochs=[1],
                rnn_learning_rates=[0.001],
                rnn_batch_size=3,
                max_autoencoders=1,
                max_rnn_runs=1,
                device="cpu",
                seed=11,
            )

        self.assertEqual(summary["counts"]["autoencoder_completed"], 1)
        self.assertEqual(summary["counts"]["latent_rnn_completed"], 1)
        self.assertEqual(summary["search_config"]["ranking_primary_metric"], "selection_latent_code_mse")
        self.assertIsNotNone(summary["best"]["latent_rnn_by_selection_latent_code_mse"])
        self.assertTrue(Path(summary["best"]["latent_rnn_by_selection_latent_code_mse"]["run_path"]).name.endswith("latent_rnn_run.json"))
