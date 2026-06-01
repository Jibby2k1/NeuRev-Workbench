from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import scipy  # noqa: F401
except ModuleNotFoundError:
    scipy = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "export_cfar_contrast_maps.py"


def require_numpy_scipy():
    if np is None:
        raise unittest.SkipTest("numpy is not installed in this Python environment")
    if scipy is None:
        raise unittest.SkipTest("scipy is not installed in this Python environment")


def pipeline_stage(step_id: str, stage_id: str, params: dict) -> dict:
    return {"id": step_id, "stage_id": stage_id, "params": params}


class CfarContrastMapTests(unittest.TestCase):
    def test_contrast_threshold_matches_existing_cfar_mask_logic(self):
        require_numpy_scipy()
        from neurobench.workbench.cfar_contrast_maps import cfar_threshold, compute_cfar_contrast_block
        from tools.prepare_gamma_cfar_workbench_run import write_chunked_cfar_mask

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            y, x = np.mgrid[:12, :12]
            stack = np.zeros((3, 12, 12), dtype=np.float32)
            stack[:, 5, 5] = [8.0, 10.0, 12.0]
            stack += (x + y).astype(np.float32) * 0.03
            mask_path = root / "mask.npy"

            summary = write_chunked_cfar_mask(
                stack,
                mask_path,
                pfa=0.04,
                guard_px=1,
                training_radius_px=3,
                chunk_frames=2,
                epsilon=1e-6,
            )
            contrast = compute_cfar_contrast_block(stack, guard_px=1, training_radius_px=3, epsilon=1e-6)
            expected = contrast >= cfar_threshold(0.04)
            mask = np.load(mask_path).astype(bool)

        self.assertEqual(summary["threshold_z"], cfar_threshold(0.04))
        np.testing.assert_array_equal(mask, expected)

    def test_cli_attaches_shared_contrast_artifacts_to_runs(self):
        require_numpy_scipy()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_dir = root / "app"
            sweep_root = root / "sweep"
            source = root / "smoothed.npy"
            app_dir.mkdir()
            sweep_root.mkdir()
            stack = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)
            np.save(source, stack)
            runs = []
            for index in [1, 2]:
                run_id = f"gamma_cfar_test__sweep_{index:03d}"
                run_dir = sweep_root / f"{index:03d}_{run_id}"
                run_dir.mkdir()
                pipeline_run = {
                    "schema_version": 1,
                    "run_id": run_id,
                    "artifacts": [
                        {
                            "artifact_id": "smoothed_video.v1",
                            "kind": "smoothed_video",
                            "path": str(source),
                            "producer_stage": "spatial_gaussian",
                            "summary": {"shape": list(stack.shape)},
                        },
                        {
                            "artifact_id": "cfar_small_ref_candidate_mask.v1",
                            "kind": "candidate_mask",
                            "path": "small.npy",
                            "producer_stage": "gamma_cfar",
                            "summary": {"guard_px": 1, "training_radius_px": 3, "pfa": 0.02},
                        },
                        {
                            "artifact_id": "cfar_large_ref_candidate_mask.v1",
                            "kind": "candidate_mask",
                            "path": "large.npy",
                            "producer_stage": "gamma_cfar",
                            "summary": {"guard_px": 1, "training_radius_px": 4, "pfa": 0.06},
                        },
                    ],
                }
                (run_dir / "pipeline_run.json").write_text(json.dumps(pipeline_run), encoding="utf-8")
                runs.append(
                    {
                        "run_id": run_id,
                        "pipeline": [
                            pipeline_stage("cfar_small_ref", "gamma_cfar", {"guard_px": 1, "training_radius_px": 3}),
                            pipeline_stage("cfar_large_ref", "gamma_cfar", {"guard_px": 1, "training_radius_px": 4}),
                        ],
                        "artifacts": {"intermediates": []},
                    }
                )
            manifest_path = app_dir / "architecture_runs.json"
            manifest_path.write_text(json.dumps({"schema_version": 1, "runs": runs}), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--app-dir",
                    str(app_dir),
                    "--sweep-root",
                    str(sweep_root),
                    "--all-runs",
                    "--chunk-frames",
                    "2",
                    "--normalization-sample-stride",
                    "1",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(result.stdout)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            first_frame_exists = (app_dir / manifest["runs"][0]["artifacts"]["intermediates"][0]["frame_pattern"].replace("%03d", "001")).is_file()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["attached_runs"], 2)
        self.assertEqual(payload["generated_sequences"], 2)
        first_records = manifest["runs"][0]["artifacts"]["intermediates"]
        second_records = manifest["runs"][1]["artifacts"]["intermediates"]
        self.assertEqual([item["id"] for item in first_records], ["cfar_small_ref", "cfar_large_ref"])
        self.assertEqual(first_records[0]["artifact_kind"], "cfar_contrast_map")
        self.assertEqual(first_records[0]["frame_pattern"], second_records[0]["frame_pattern"])
        self.assertTrue(first_frame_exists)


if __name__ == "__main__":
    unittest.main()
