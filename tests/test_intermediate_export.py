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
SCRIPT = ROOT / "tools" / "export_intermediate_frames.py"


def require_numpy():
    if np is None:
        raise unittest.SkipTest("numpy is not installed in this Python environment")


def png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) < 24 or not header.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError(f"{path} is not a PNG file")
    return struct.unpack(">II", header[16:24])


class IntermediateExportTests(unittest.TestCase):
    def test_export_npy_stack_writes_png_frames_and_manifest_entry(self):
        require_numpy()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stack = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)
            input_path = root / "highpass_video.npy"
            np.save(input_path, stack)
            app_dir = root / "app"
            out_dir = app_dir / "generated_runs" / "run_a" / "intermediates" / "temporal_highpass_gaussian"
            manifest_path = app_dir / "architecture_runs.json"
            app_dir.mkdir()
            manifest_path.write_text(
                json.dumps({"schema_version": 1, "runs": [{"run_id": "run_a", "artifacts": {"intermediates": []}}]}),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input-npy",
                    str(input_path),
                    "--out-dir",
                    str(out_dir),
                    "--architecture-runs",
                    str(manifest_path),
                    "--run-id",
                    "run_a",
                    "--stage-id",
                    "temporal_highpass_gaussian",
                    "--step-id",
                    "highpass",
                    "--label",
                    "High-pass video",
                    "--description",
                    "Synthetic high-pass frames.",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            artifact = json.loads(result.stdout)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            first = out_dir / "frame_001.png"
            third = out_dir / "frame_003.png"
            first_size = png_size(first)
            third_size = png_size(third)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(artifact["frame_count"], 3)
        self.assertEqual(artifact["media_type"], "frame_sequence")
        self.assertEqual(first_size, (7, 5))
        self.assertEqual(third_size, (7, 5))
        attached = manifest["runs"][0]["artifacts"]["intermediates"][0]
        self.assertEqual(attached["id"], "highpass")
        self.assertEqual(attached["stage_id"], "temporal_highpass_gaussian")
        self.assertEqual(attached["frame_count"], 3)
        self.assertEqual(attached["frame_pattern"], "generated_runs/run_a/intermediates/temporal_highpass_gaussian/frame_%03d.png")
        self.assertEqual(attached["description"], "Synthetic high-pass frames.")

    def test_export_npy_2d_image_writes_single_frame(self):
        require_numpy()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.ones((4, 6), dtype=np.float32)
            input_path = root / "projection.npy"
            np.save(input_path, image)
            out_dir = root / "frames"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input-npy",
                    str(input_path),
                    "--out-dir",
                    str(out_dir),
                    "--stage-id",
                    "robust_positive_local_z",
                    "--frame-pattern",
                    "projection_%03d.png",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            artifact = json.loads(result.stdout)
            frame = out_dir / "projection_001.png"
            frame_size = png_size(frame)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(artifact["frame_count"], 1)
        self.assertEqual(frame_size, (6, 4))


if __name__ == "__main__":
    unittest.main()
