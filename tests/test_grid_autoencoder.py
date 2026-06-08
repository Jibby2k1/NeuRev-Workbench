from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class GridAutoencoderTests(unittest.TestCase):
    def test_forward_shape_and_tiny_train_checkpoint(self):
        import torch
        from neurobench.dynamics.models import GridAutoencoder
        from neurobench.dynamics.train import train_autoencoder

        model = GridAutoencoder(input_channels=1, latent_dim=4)
        x = torch.zeros((2, 1, 32, 32), dtype=torch.float32)
        recon, z = model(x)
        self.assertEqual(tuple(recon.shape), (2, 1, 32, 32))
        self.assertEqual(tuple(z.shape), (2, 4))
        self.assertGreaterEqual(float(recon.detach().min()), 0.0)
        self.assertLessEqual(float(recon.detach().max()), 1.0)

        model64 = GridAutoencoder(input_channels=1, latent_dim=4, input_shape=(1, 64, 64))
        x64 = torch.zeros((2, 1, 64, 64), dtype=torch.float32)
        recon64, z64 = model64(x64)
        self.assertEqual(tuple(recon64.shape), (2, 1, 64, 64))
        self.assertEqual(tuple(z64.shape), (2, 4))

        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            arrays=root/"arrays.npz"
            frames=np.random.default_rng(1).random((6,1,32,32), dtype=np.float32)
            np.savez(arrays, frames=frames, frame_video_ids=np.asarray(["v"]*6), frame_labels=np.asarray(["left"]*6))
            dataset={"array_path":str(arrays)}
            run=train_autoencoder(dataset=dataset, out_dir=root/"ae", latent_dim=4, epochs=1, batch_size=3)
            ckpt=torch.load(run["checkpoint_path"], map_location="cpu")
            examples_exist = Path(run["reconstruction_examples_path"]).is_file()
            with np.load(run["latent_codes_path"], allow_pickle=False) as data:
                codes = data["latent_codes"].copy()
                raw_codes = data["latent_codes_raw"].copy()

        self.assertIn("model_state", ckpt)
        self.assertEqual(ckpt["output_normalization"], "sigmoid_unit_interval")
        self.assertEqual(ckpt["latent_code_normalization"], "standard_score_per_dimension")
        self.assertEqual(ckpt["base_channels"], 16)
        self.assertTrue(examples_exist)
        self.assertTrue(np.all(np.isfinite(codes)))
        self.assertTrue(np.all(np.isfinite(raw_codes)))
        self.assertLess(float(abs(codes.mean())), 1e-5)
