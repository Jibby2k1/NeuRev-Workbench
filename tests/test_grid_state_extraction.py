from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class GridStateExtractionTests(unittest.TestCase):
    def test_grid_has_1024_regions_and_covers_image(self):
        from neurobench.algorithms.grid_regions import generate_grid_spec

        grid = generate_grid_spec(template_id="t", height=65, width=67)
        self.assertEqual(grid["region_count"], 1024)
        self.assertEqual(grid["regions"][0]["region_id"], "R00C00")
        self.assertEqual(grid["regions"][-1]["bbox"][2:], [67, 65])

    def test_extracts_constant_and_known_activation(self):
        from neurobench.algorithms.grid_regions import extract_grid_states, generate_grid_spec

        grid = generate_grid_spec(template_id="t", height=64, width=64)
        constant = np.ones((3, 64, 64), dtype=np.float32) * 5
        states = extract_grid_states(constant, grid, normalization="none")
        self.assertEqual(states["grid_state"].shape, (3, 32, 32, 1))
        self.assertTrue(np.allclose(states["grid_state"], 5.0))

        video = np.zeros((1, 64, 64), dtype=np.float32)
        video[:, 10:12, 14:16] = 9.0
        states = extract_grid_states(video, grid, normalization="none")
        row, col = np.unravel_index(np.argmax(states["grid_state"][0, :, :, 0]), (32, 32))
        self.assertEqual((row, col), (5, 7))
