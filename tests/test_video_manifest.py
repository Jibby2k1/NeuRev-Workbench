from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile


class VideoManifestTests(unittest.TestCase):
    def test_parses_labels_and_warns_on_bad_names(self):
        from neurobench.data.video_manifest import build_video_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tifffile.imwrite(root / "1_left.tif", np.zeros((2, 4, 5), dtype=np.float32))
            tifffile.imwrite(root / "2_right.tiff", np.zeros((3, 4, 5), dtype=np.float32))
            tifffile.imwrite(root / "foo.tif", np.zeros((3, 4, 5), dtype=np.float32))
            manifest = build_video_manifest(input_dir=root)

        self.assertEqual([v["video_id"] for v in manifest["videos"]], ["1_left", "2_right"])
        self.assertEqual(manifest["label_counts"]["left"], 1)
        self.assertEqual(manifest["label_counts"]["right"], 1)
        self.assertEqual(manifest["split_policy"], "by_video")
        self.assertIn("foo.tif", manifest["warnings"][0])


    def test_label_alias_maps_rest_to_neutral(self):
        from neurobench.data.video_manifest import build_video_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tifffile.imwrite(root / "1 rest.tif", np.zeros((2, 4, 5), dtype=np.float32))
            manifest = build_video_manifest(
                input_dir=root,
                filename_regex=r"^(?P<index>[0-9]+) (?P<label>rest|left|right)\.tiff?$",
                label_aliases={"rest": "neutral"},
            )

        self.assertEqual(manifest["videos"][0]["label"], "neutral")
        self.assertEqual(manifest["videos"][0]["condition"], "rest")
        self.assertEqual(manifest["label_counts"]["neutral"], 1)
