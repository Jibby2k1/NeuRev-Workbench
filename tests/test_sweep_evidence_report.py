from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_sweep_evidence_report.py"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class SweepEvidenceReportTests(unittest.TestCase):
    def test_report_scores_stencil_coverage_stability_and_diagnostics(self):
        from neurobench.reports.sweep_evidence import build_sweep_evidence_report, render_sweep_evidence_markdown

        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp) / "app"
            app.mkdir()
            self._write_fixture(app)

            report = build_sweep_evidence_report(app, stability_radius_px=6, stability_min_support_runs=1)
            markdown = render_sweep_evidence_markdown(report)

        self.assertEqual(report["payload_kind"], "sweep_evidence_report")
        self.assertEqual(report["summary"]["analyzed_run_count"], 2)
        self.assertEqual(report["summary"]["runs_with_roi_sidecars"], 2)
        self.assertEqual(report["recommended_runs"][0]["run_id"], "gamma_cfar_sweep_a")
        run_a = next(row for row in report["runs"] if row["run_id"] == "gamma_cfar_sweep_a")
        run_b = next(row for row in report["runs"] if row["run_id"] == "gamma_cfar_sweep_b")
        self.assertGreater(run_a["stencil_coverage_fraction"], run_b["stencil_coverage_fraction"])
        self.assertGreater(run_a["stable_roi_fraction"], 0)
        self.assertIn("low_stencil_coverage", [item["code"] for item in run_b["diagnostics"]])
        self.assertIn("Sweep Evidence Report", markdown)
        self.assertIn("gamma_cfar_sweep_a", markdown)

    def test_cli_writes_and_attaches_report_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp) / "app"
            app.mkdir()
            self._write_fixture(app)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--app-dir",
                    str(app),
                    "--stability-radius-px",
                    "6",
                    "--stability-min-support-runs",
                    "1",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            manifest = json.loads((app / "architecture_runs.json").read_text(encoding="utf-8"))
            report = json.loads((app / "sweep_evidence_report.json").read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(manifest["artifacts"]["sweep_evidence_report"], "sweep_evidence_report.json")
        self.assertEqual(manifest["artifacts"]["sweep_evidence_markdown"], "sweep_evidence_report.md")
        self.assertEqual(report["summary"]["candidate_bearing_run_count"], 2)

    def _write_fixture(self, app: Path) -> None:
        write_json(
            app / "annotations.json",
            {
                "settings": {
                    "anatomyStencil": {
                        "polygon": [[0, 0], [60, 0], [60, 60], [0, 60]],
                    }
                }
            },
        )
        write_json(
            app / "architecture_runs.json",
            {
                "schema_version": 1,
                "dataset_id": "fish_fixture",
                "runs": [
                    self._run("gamma_cfar_sweep_a", roi_count=3, event_count=8, pfa=0.03),
                    self._run("gamma_cfar_sweep_b", roi_count=3, event_count=3, pfa=0.01),
                ],
            },
        )
        write_json(
            app / "generated_runs" / "gamma_cfar_sweep_a" / "review_rois.summary.json",
            {
                "payload_kind": "review_rois_summary",
                "roi_count": 3,
                "trace_shard_count": 3,
                "review_rois": [
                    {"id": "a1", "centroidX": 10, "centroidY": 10, "area": 80, "events": [{"frame": 1}]},
                    {"id": "a2", "centroidX": 20, "centroidY": 20, "area": 90, "events": [{"frame": 2}, {"frame": 4}]},
                    {"id": "a3", "centroidX": 80, "centroidY": 80, "area": 40, "events": [{"frame": 3}]},
                ],
            },
        )
        write_json(
            app / "generated_runs" / "gamma_cfar_sweep_b" / "review_rois.summary.json",
            {
                "payload_kind": "review_rois_summary",
                "roi_count": 3,
                "trace_shard_count": 3,
                "review_rois": [
                    {"id": "b1", "centroidX": 11, "centroidY": 11, "area": 80, "events": [{"frame": 1}]},
                    {"id": "b2", "centroidX": 85, "centroidY": 85, "area": 60, "events": []},
                    {"id": "b3", "centroidX": 90, "centroidY": 90, "area": 65, "events": []},
                ],
            },
        )
        for run_id in ["gamma_cfar_sweep_a", "gamma_cfar_sweep_b"]:
            write_json(
                app / "generated_runs" / run_id / "stencil_gap_report.json",
                {
                    "stencil_available": True,
                    "gaps": [
                        {"id": "gap_1", "roi_count": 0, "priority": 10},
                        {"id": "gap_2", "roi_count": 1, "priority": 5},
                    ],
                },
            )

    def _run(self, run_id: str, *, roi_count: int, event_count: int, pfa: float) -> dict[str, object]:
        return {
            "run_id": run_id,
            "label": run_id,
            "execution": {"status": "completed"},
            "summary": {
                "roi_count": roi_count,
                "event_count": event_count,
                "median_equivalent_diameter_um": 6.0,
                "plausible_size_fraction": 1.0,
                "cfar_small_ref.pfa": pfa,
            },
            "sweep": {
                "parameters": [
                    {"stage": "cfar_small_ref", "param": "pfa", "value": pfa},
                    {"stage": "cfar_large_ref", "param": "training_radius_px", "value": 12},
                    {"stage": "components", "param": "support_min_frames", "value": 20},
                ]
            },
            "artifacts": {
                "review_rois_summary_file": f"generated_runs/{run_id}/review_rois.summary.json",
                "stencil_gap_report_file": f"generated_runs/{run_id}/stencil_gap_report.json",
                "intermediates": [
                    {
                        "artifact_kind": "cfar_contrast_map",
                        "id": "cfar_large_ref",
                        "label": "Large-reference Gamma CFAR contrast",
                        "frame_count": 20,
                        "summary": {"training_radius_px": 12, "guard_px": 3, "pfa": pfa},
                    }
                ],
            },
        }


if __name__ == "__main__":
    unittest.main()
