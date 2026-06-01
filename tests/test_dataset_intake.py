from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DatasetIntakeTests(unittest.TestCase):
    def test_build_public_dataset_intake_report_flags_planned_nwb_bridge(self):
        from neurobench.data.intake import build_dataset_intake_manifest, dataset_intake_report

        manifest = build_dataset_intake_manifest(
            dataset_id="public_fish",
            raw_video="Inputs/Public/fish.nwb",
            frame_rate_hz=50.0,
            pixel_size_microns=0.45,
            source_template="dandi-nwb",
        )
        report = dataset_intake_report(manifest, base_dir=ROOT)
        self.assertEqual(manifest["source"]["template"], "dandi-nwb")
        self.assertEqual(manifest["paths"]["review_data"], "Outputs/NeuronReview/public_fish/app/review_data.json")
        checks = {item["name"]: item for item in report["checks"]}
        self.assertEqual(checks["frame_rate_hz"]["status"], "ok")
        self.assertEqual(checks["format"]["status"], "warn")
        self.assertIn("conversion/import bridge", checks["format"]["detail"])

    def test_dataset_intake_cli_writes_manifest_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "movie.npy"
            raw.write_bytes(b"fixture")
            manifest_path = tmp_path / "dataset.json"
            report_path = tmp_path / "intake_report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurobench.cli.main",
                    "dataset",
                    "intake",
                    "--dataset-id",
                    "local_fish",
                    "--raw-video",
                    str(raw),
                    "--frame-rate-hz",
                    "50",
                    "--pixel-size-microns",
                    "0.5",
                    "--out",
                    str(manifest_path),
                    "--report-out",
                    str(report_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["dataset_id"], "local_fish")
        self.assertTrue(report["ready"])
        self.assertIn("ready: yes", result.stdout)


if __name__ == "__main__":
    unittest.main()
