from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class TemplateGridSafetyTests(unittest.TestCase):
    def test_iter_video_chunks_for_npy_and_tiff(self):
        from neurobench.data.video import iter_video_chunks
        import tifffile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arr = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
            np.save(root / "video.npy", arr)
            tifffile.imwrite(root / "video.tif", arr)

            npy_chunks = list(iter_video_chunks(root / "video.npy", chunk_size=2))
            tif_chunks = list(iter_video_chunks(root / "video.tif", chunk_size=2))

        self.assertEqual([(c.start_frame, c.end_frame) for c in npy_chunks], [(0, 2), (2, 4), (4, 5)])
        self.assertEqual(np.concatenate([c.data for c in npy_chunks], axis=0).shape, arr.shape)
        self.assertEqual(np.concatenate([c.data for c in tif_chunks], axis=0).shape, arr.shape)

    def test_chunked_registration_and_grid_artifacts(self):
        from neurobench.algorithms.grid_regions import generate_grid_spec, write_grid_state_artifacts, write_registered_grid_state_artifacts
        from neurobench.algorithms.template_matching import write_registered_video_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = np.zeros((5, 16, 16), dtype=np.float32)
            video[:, 6:10, 6:10] = 1.0
            source = root / "1_neutral.npy"
            np.save(source, video)
            template = {
                "template_id": "template_from_1_neutral_v1",
                "coordinate_system": {"height": 16, "width": 16},
            }
            registration = {
                "video_id": "1_neutral",
                "transform": {"rotation_deg": 0.0, "translation_px": [0.0, 0.0], "scale": 1.0},
            }
            summary = write_registered_video_artifacts(
                video_path=source,
                registration_result=registration,
                template_spec=template,
                out_dir=root / "registered",
                chunk_size_frames=2,
            )
            grid = generate_grid_spec(template_id="template_from_1_neutral_v1", height=16, width=16, rows=4, cols=4)
            grid_summary = write_grid_state_artifacts(
                registered_video_path=summary["path"],
                grid_spec=grid,
                out_dir=root / "grid",
                video_id="1_neutral",
                label="neutral",
                normalization="none",
                chunk_size_frames=2,
            )
            streaming_summary = write_registered_grid_state_artifacts(
                video_path=source,
                registration_result=registration,
                grid_spec=grid,
                out_dir=root / "stream_grid",
                video_id="1_neutral",
                label="neutral",
                normalization="none",
                chunk_size_frames=2,
                device="cpu",
            )
            with np.load(root / "grid" / "1_neutral" / "grid_states.npz", allow_pickle=False) as data:
                shape = data["grid_state"].shape
                materialized_grid = data["grid_state"].copy()
            with np.load(root / "stream_grid" / "1_neutral" / "grid_states.npz", allow_pickle=False) as data:
                streamed_grid = data["grid_state"].copy()

        self.assertEqual(summary["shape"], [5, 16, 16])
        self.assertEqual(grid_summary["shape"], [5, 4, 4, 1])
        self.assertEqual(streaming_summary["shape"], [5, 4, 4, 1])
        self.assertFalse(streaming_summary["registered_video_materialized"])
        self.assertEqual(shape, (5, 4, 4, 1))
        self.assertTrue(np.allclose(streamed_grid, materialized_grid))

    def test_preflight_estimates_without_pixel_load(self):
        from neurobench.data.preflight import build_template_grid_preflight

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "1_left.tif"
            path.write_bytes(b"placeholder")
            manifest = {
                "dataset_id": "demo",
                "videos": [
                    {
                        "video_id": "1_left",
                        "path": str(path),
                        "label": "left",
                        "frame_count": 10,
                        "height": 8,
                        "width": 8,
                    }
                ],
            }
            preflight = build_template_grid_preflight(manifest=manifest, output_root=root / "out", rows=4, cols=4, expected_video_count=1)

        self.assertEqual(preflight["video_count"], 1)
        self.assertGreater(preflight["estimated_output_total_bytes"], 0)
        self.assertEqual(preflight["label_counts"]["left"], 1)
