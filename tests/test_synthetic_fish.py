from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class SyntheticFishTests(unittest.TestCase):
    def test_writes_filename_compatible_videos(self):
        from neurobench.data.synthetic_fish import generate_synthetic_grid_fish_videos

        with tempfile.TemporaryDirectory() as tmp:
            bundle = generate_synthetic_grid_fish_videos(video_count_per_label=1, frames=5, height=24, width=32)
            paths = bundle.write(Path(tmp), suffix=".tif")

        self.assertIn("1_left", paths["videos"])
        self.assertIn("1_neutral", bundle.labels)
        self.assertEqual(bundle.videos["1_right"].shape, (5, 24, 32))
