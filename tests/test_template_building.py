from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile


class TemplateBuildingTests(unittest.TestCase):
    def test_outlier_rejection_records_removed_frame(self):
        from neurobench.algorithms.template_matching import build_template_from_reference_video

        video = np.zeros((20, 16, 18), dtype=np.float32)
        video += np.linspace(0, 1, 18, dtype=np.float32)[None, None, :]
        video[7] += 50.0
        result = build_template_from_reference_video(video, max_outlier_fraction=0.1, z_threshold=3.0)

        self.assertIn(7, result["outlier_rejection"]["removed_frame_indices"])
        self.assertEqual(result["projection"].shape, (16, 18))
        self.assertTrue(np.isfinite(result["projection"]).all())

    def test_writes_template_artifacts_from_tif(self):
        from neurobench.algorithms.template_matching import write_template_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "1_neutral.tif"
            tifffile.imwrite(path, np.ones((4, 12, 14), dtype=np.float32))
            spec = write_template_artifacts(video_path=path, source_video_id="1_neutral", out_dir=root / "template")

            self.assertTrue((root / "template" / "template_spec.json").is_file())
            self.assertTrue((root / "template" / "template_projection.png").is_file())
            self.assertEqual(spec["coordinate_system"]["height"], 12)
