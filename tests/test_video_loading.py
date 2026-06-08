from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


class VideoLoadingTests(unittest.TestCase):
    def test_load_npy_and_tif_frame_first(self):
        from neurobench.data.video import load_video_array
        import tifffile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arr = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
            np.save(root / "video.npy", arr)
            tifffile.imwrite(root / "video.tif", arr)

            self.assertEqual(load_video_array(root / "video.npy").shape, (3, 4, 5))
            self.assertEqual(load_video_array(root / "video.tif").shape, (3, 4, 5))


    def test_large_eager_load_guard_can_be_overridden(self):
        from neurobench.data.video import ALLOW_LARGE_EAGER_LOADS_ENV, load_video_array

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arr = np.zeros((2, 4, 5), dtype=np.float32)
            np.save(root / "video.npy", arr)

            with self.assertRaisesRegex(RuntimeError, "Refusing to eagerly load"):
                load_video_array(root / "video.npy", max_eager_bytes=1)

            previous = os.environ.get(ALLOW_LARGE_EAGER_LOADS_ENV)
            os.environ[ALLOW_LARGE_EAGER_LOADS_ENV] = "1"
            try:
                loaded = load_video_array(root / "video.npy", max_eager_bytes=1)
            finally:
                if previous is None:
                    os.environ.pop(ALLOW_LARGE_EAGER_LOADS_ENV, None)
                else:
                    os.environ[ALLOW_LARGE_EAGER_LOADS_ENV] = previous

        self.assertEqual(loaded.shape, (2, 4, 5))

    def test_rejects_ambiguous_shape(self):
        from neurobench.data.video import coerce_frame_first_video

        with self.assertRaisesRegex(ValueError, "Expected video array shape"):
            coerce_frame_first_video(np.zeros((2, 3, 4, 5), dtype=np.float32))
