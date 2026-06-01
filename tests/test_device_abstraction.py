from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def require_numpy():
    if np is None:
        raise unittest.SkipTest("numpy is not installed in this Python environment")


class DeviceAbstractionTests(unittest.TestCase):
    def test_auto_device_falls_back_to_cpu_when_cuda_unavailable(self):
        from neurobench.pipelines.devices import resolve_device

        with patch("neurobench.pipelines.devices._cuda_backend", return_value=""):
            spec = resolve_device("auto")

        self.assertEqual(spec.requested, "auto")
        self.assertEqual(spec.resolved, "cpu")
        self.assertEqual(spec.backend, "numpy")
        self.assertTrue(spec.available)
        self.assertIn("fallback", spec.reason)

    def test_explicit_cuda_request_fails_when_cuda_unavailable(self):
        from neurobench.pipelines.devices import resolve_device

        with patch("neurobench.pipelines.devices._cuda_backend", return_value=""):
            with self.assertRaisesRegex(RuntimeError, "CUDA device requested"):
                resolve_device("cuda")

    def test_auto_device_metadata_reaches_algorithm_output(self):
        require_numpy()
        from neurobench.algorithms.cfar import robust_local_cfar

        video = np.zeros((2, 12, 12), dtype=np.float32)
        video[0, 6, 6] = 5.0
        with patch("neurobench.pipelines.devices._cuda_backend", return_value=""):
            result = robust_local_cfar(video, pfa=0.1, guard_px=1, training_radius_px=4, device="auto")

        self.assertEqual(result["device"]["requested"], "auto")
        self.assertEqual(result["device"]["resolved"], "cpu")
        self.assertEqual(result["device"]["backend"], "numpy")

    def test_execute_pipeline_records_auto_cpu_fallback(self):
        require_numpy()
        from neurobench.data.synthetic import generate_synthetic_calcium_dataset
        from neurobench.pipelines.executor import execute_pipeline

        dataset = generate_synthetic_calcium_dataset(frames=6, height=12, width=12, include_impulse_artifact=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = dataset.write(root / "fixture", dataset_id="device_auto")
            spec = {
                "schema_version": 1,
                "dataset_id": "device_auto",
                "run_id": "device_auto_pipeline",
                "execution": {"device": "auto"},
                "pipeline": [
                    {"id": "source", "stage_id": "source_video_import", "params": {"source": paths["video"]}},
                    {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 2.0}},
                    {"id": "smooth", "stage_id": "spatial_gaussian", "params": {"sigma_px": 0.0}},
                    {"id": "cfar", "stage_id": "gamma_cfar", "params": {"pfa": 0.2, "guard_px": 1, "training_radius_px": 4}},
                ],
            }
            with patch("neurobench.pipelines.devices._cuda_backend", return_value=""):
                result = execute_pipeline(spec, run_root=root / "run")
            manifest = json.loads((root / "run" / "pipeline_run.json").read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "completed")
        self.assertEqual(manifest["environment"]["device_requested"], "auto")
        self.assertEqual(manifest["environment"]["device"], "cpu")
        self.assertEqual(manifest["environment"]["device_backend"], "numpy")
        self.assertIn("fallback", manifest["environment"]["device_reason"])

    def test_execute_pipeline_supports_multi_stage_cfar_cascade(self):
        require_numpy()
        from neurobench.pipelines.executor import execute_pipeline

        video = np.zeros((4, 24, 24), dtype=np.float32)
        video[:, 8, 8] = 8.0
        video[:, 16, 16] = 4.0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw.npy"
            np.save(source, video)
            spec = {
                "schema_version": 1,
                "dataset_id": "cfar_cascade",
                "run_id": "cfar_cascade_pipeline",
                "pipeline": [
                    {"id": "source", "stage_id": "source_video_import", "params": {"source": source}},
                    {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 0.1}},
                    {"id": "smooth", "stage_id": "spatial_gaussian", "params": {"sigma_px": 0.0}},
                    {"id": "score", "stage_id": "robust_positive_local_z", "params": {"epsilon": 0.05}},
                    {"id": "cfar_small_ref", "stage_id": "gamma_cfar", "params": {"pfa": 0.2, "guard_px": 1, "training_radius_px": 4}},
                    {
                        "id": "cfar_large_ref",
                        "stage_id": "gamma_cfar",
                        "params": {"pfa": 0.2, "guard_px": 2, "training_radius_px": 7},
                        "metadata": {"previous_mask_step": "cfar_small_ref", "combine_mode": "intersection"},
                    },
                    {
                        "id": "components",
                        "stage_id": "component_filter",
                        "params": {"seed_z": 0.5, "min_area_px": 1, "max_area_px": 50, "support_min_frames": 1},
                    },
                ],
            }
            result = execute_pipeline(spec, run_root=root / "run")
            manifest = json.loads((root / "run" / "pipeline_run.json").read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "completed")
        cfar_artifacts = [item for item in manifest["artifacts"] if item["kind"] == "candidate_mask"]
        self.assertEqual(len(cfar_artifacts), 2)
        cascade_artifact = next(item for item in cfar_artifacts if item["summary"].get("previous_mask_step"))
        self.assertEqual(cascade_artifact["summary"]["previous_mask_step"], "cfar_small_ref")
        roi_artifact = next(item for item in manifest["artifacts"] if item["kind"] == "roi_candidates")
        self.assertEqual(roi_artifact["summary"]["evidence_source"], "candidate_mask")


if __name__ == "__main__":
    unittest.main()
