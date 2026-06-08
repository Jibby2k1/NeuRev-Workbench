from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


def _write_grid_state(root: Path, video_id: str, label: str):
    out = root / video_id
    out.mkdir(parents=True, exist_ok=True)
    grid = np.random.default_rng(abs(hash(video_id)) % 1000).random((6, 32, 32, 1), dtype=np.float32)
    np.savez(out / "grid_states.npz", grid_state=grid, flat_state=grid.reshape(6, 1024, 1), region_ids=np.asarray([f"R{i:04d}" for i in range(1024)]), feature_names=np.asarray(["mean_intensity"]), video_id=np.asarray(video_id), label=np.asarray(label), normalization=np.asarray("none"), source_registered_video=np.asarray("registered.npy"))


class GridDynamicsDatasetTests(unittest.TestCase):
    def test_windows_do_not_cross_video_boundaries(self):
        from neurobench.dynamics.datasets import build_dynamics_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos=[]
            for label in ["left", "right", "neutral"]:
                for idx in [1, 2, 3]:
                    vid=f"{idx}_{label}"
                    _write_grid_state(root / "grid", vid, label)
                    videos.append({"video_id": vid, "label": label})
            manifest={"schema_version":1,"dataset_id":"d","videos":videos,"label_set":["left","right","neutral"]}
            ds=build_dynamics_dataset(manifest=manifest, grid_states_dir=root/"grid", out_dir=root/"dyn", window_frames=3)
            arrays=np.load(ds["array_path"])

        self.assertEqual(ds["splits"]["split_unit"], "video")
        for ids in [ds["splits"]["train_video_ids"], ds["splits"]["val_video_ids"], ds["splits"]["test_video_ids"]]:
            self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(arrays["window_video_ids"] == arrays["window_video_ids"]))
        self.assertEqual(arrays["windows"].shape[1], 3)


    def test_temporal_stride_downsamples_before_windowing(self):
        from neurobench.dynamics.datasets import build_dynamics_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_grid_state(root / "grid", "v_left", "left")
            manifest = {"schema_version": 1, "dataset_id": "d", "videos": [{"video_id": "v_left", "label": "left"}], "label_set": ["left"]}
            ds = build_dynamics_dataset(
                manifest=manifest,
                grid_states_dir=root / "grid",
                out_dir=root / "dyn",
                window_frames=2,
                prediction_horizon_frames=1,
                temporal_stride_frames=2,
                split_method="train_all_smoke",
            )
            arrays = np.load(ds["array_path"])

        self.assertEqual(ds["windowing"]["temporal_stride_frames"], 2)
        self.assertEqual(arrays["frames"].shape[0], 3)
        self.assertEqual(arrays["windows"].shape[0], 1)

    def test_train_all_smoke_split_records_warning(self):
        from neurobench.dynamics.datasets import build_dynamics_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = []
            for label in ["left", "right", "neutral"]:
                vid = f"1_{label}"
                _write_grid_state(root / "grid", vid, label)
                videos.append({"video_id": vid, "label": label})
            manifest = {"schema_version": 1, "dataset_id": "d", "videos": videos, "label_set": ["left", "right", "neutral"]}
            ds = build_dynamics_dataset(
                manifest=manifest,
                grid_states_dir=root / "grid",
                out_dir=root / "dyn",
                window_frames=3,
                split_method="train_all_smoke",
            )

        self.assertEqual(sorted(ds["splits"]["train_video_ids"]), ["1_left", "1_neutral", "1_right"])
        self.assertEqual(ds["splits"]["val_video_ids"], [])
        self.assertEqual(ds["splits"]["test_video_ids"], [])
        self.assertTrue(any("train-all smoke" in warning for warning in ds["warnings"]))

