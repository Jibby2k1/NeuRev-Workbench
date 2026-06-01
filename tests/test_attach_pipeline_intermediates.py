from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "attach_pipeline_intermediates.py"


def require_numpy():
    if np is None:
        raise unittest.SkipTest("numpy is not installed in this Python environment")


def png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) < 24 or not header.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError(f"{path} is not a PNG file")
    return struct.unpack(">II", header[16:24])


class AttachPipelineIntermediatesTests(unittest.TestCase):
    def test_attach_pipeline_intermediates_exports_selected_npy_artifacts(self):
        require_numpy()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "pipeline_run"
            artifacts_dir = run_root / "artifacts" / "preprocessing"
            artifacts_dir.mkdir(parents=True)
            highpass = artifacts_dir / "highpass_video.npy"
            z_stack = artifacts_dir / "z_stack.npy"
            np.save(highpass, np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5))
            np.save(z_stack, np.ones((2, 4, 5), dtype=np.float32))
            pipeline_run_path = run_root / "pipeline_run.json"
            pipeline_run_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "run_a",
                        "artifacts": [
                            {
                                "artifact_id": "highpass_video.v1",
                                "kind": "highpass_video",
                                "path": "artifacts/preprocessing/highpass_video.npy",
                                "producer_stage": "temporal_highpass_gaussian",
                                "summary": {"sigma_frames": 2.0},
                            },
                            {
                                "artifact_id": "z_stack.v1",
                                "kind": "z_stack",
                                "path": "artifacts/preprocessing/z_stack.npy",
                                "producer_stage": "robust_positive_local_z",
                            },
                            {
                                "artifact_id": "roi_candidates.v1",
                                "kind": "roi_candidates",
                                "path": "artifacts/candidates/roi_candidates.json",
                                "producer_stage": "component_filter",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            app_dir = root / "app"
            app_dir.mkdir()
            architecture_runs_path = app_dir / "architecture_runs.json"
            architecture_runs_path.write_text(
                json.dumps({"schema_version": 1, "runs": [{"run_id": "run_a", "artifacts": {"intermediates": []}}]}),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--pipeline-run",
                    str(pipeline_run_path),
                    "--architecture-runs",
                    str(architecture_runs_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(result.stdout)
            manifest = json.loads(architecture_runs_path.read_text(encoding="utf-8"))
            highpass_png = app_dir / "generated_runs" / "run_a" / "intermediates" / "temporal_highpass_gaussian" / "frame_001.png"
            z_png = app_dir / "generated_runs" / "run_a" / "intermediates" / "robust_positive_local_z" / "frame_002.png"
            highpass_size = png_size(highpass_png)
            z_size = png_size(z_png)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["exported_count"], 2)
        self.assertEqual(highpass_size, (5, 4))
        self.assertEqual(z_size, (5, 4))
        attached = manifest["runs"][0]["artifacts"]["intermediates"]
        self.assertEqual([item["stage_id"] for item in attached], ["temporal_highpass_gaussian", "robust_positive_local_z"])
        self.assertEqual(attached[0]["summary"], {"sigma_frames": 2.0})
        self.assertEqual(
            attached[0]["frame_pattern"],
            "generated_runs/run_a/intermediates/temporal_highpass_gaussian/frame_%03d.png",
        )

    def test_attach_pipeline_intermediates_can_limit_kinds_and_override_run_id(self):
        require_numpy()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "pipeline_run"
            artifacts_dir = run_root / "artifacts" / "candidates"
            artifacts_dir.mkdir(parents=True)
            mask = artifacts_dir / "candidate_mask.npy"
            np.save(mask, np.ones((1, 3, 4), dtype=np.uint8))
            pipeline_run_path = run_root / "pipeline_run.json"
            pipeline_run_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "executor_run",
                        "artifacts": [
                            {
                                "artifact_id": "candidate_mask.v1",
                                "kind": "candidate_mask",
                                "path": "artifacts/candidates/candidate_mask.npy",
                                "producer_stage": "gamma_cfar",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            app_dir = root / "app"
            app_dir.mkdir()
            architecture_runs_path = app_dir / "architecture_runs.json"
            architecture_runs_path.write_text(
                json.dumps({"schema_version": 1, "runs": [{"run_id": "workbench_run", "artifacts": {}}]}),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--pipeline-run",
                    str(pipeline_run_path),
                    "--architecture-runs",
                    str(architecture_runs_path),
                    "--run-id",
                    "workbench_run",
                    "--include-kind",
                    "candidate_mask",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(result.stdout)
            manifest = json.loads(architecture_runs_path.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["run_id"], "workbench_run")
        self.assertEqual(payload["exported_count"], 1)
        self.assertEqual(manifest["runs"][0]["artifacts"]["intermediates"][0]["stage_id"], "gamma_cfar")


if __name__ == "__main__":
    unittest.main()
