from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def require_numpy():
    if np is None:
        raise unittest.SkipTest("numpy is not installed in this Python environment")


def _executable_sweep_spec(source: str) -> dict:
    return {
        "schema_version": 1,
        "dataset_id": "sweep_synthetic",
        "run_id": "sweep_localz",
        "pipeline": [
            {"id": "source", "stage_id": "source_video_import", "params": {"source": source}},
            {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 2.0}},
            {"id": "score", "stage_id": "robust_positive_local_z", "params": {"epsilon": 0.05}},
            {
                "id": "components",
                "stage_id": "component_filter",
                "params": {"seed_z": 1.5, "min_area_px": 3, "max_area_px": 100},
            },
        ],
        "sweep": {
            "id": "component_thresholds",
            "parameters": [
                {"stage": "components", "param": "seed_z", "values": [1.4, 1.8]},
                {"stage": "components", "param": "min_area_px", "values": [3, 5]},
            ],
        },
    }


class PipelineSweepExecutionTests(unittest.TestCase):
    def test_execute_parameter_sweep_runs_all_expanded_specs_and_writes_report(self):
        require_numpy()
        from neurobench.data.synthetic import generate_synthetic_calcium_dataset
        from neurobench.pipelines.sweeps import execute_parameter_sweep

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = generate_synthetic_calcium_dataset(include_impulse_artifact=False).write(root / "fixture")
            summary = execute_parameter_sweep(_executable_sweep_spec(paths["video"]), run_root=root / "sweep")
            summary_from_disk = json.loads((root / "sweep" / "sweep_summary.json").read_text(encoding="utf-8"))
            report = (root / "sweep" / "sweep_report.md").read_text(encoding="utf-8")
            first_manifest = json.loads((root / "sweep" / "001_sweep_localz__sweep_001" / "pipeline_run.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["succeeded"], 4)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary_from_disk, summary)
        self.assertEqual(summary["runs"][0]["sweep_parameters"][0]["value"], 1.4)
        self.assertEqual(summary["runs"][1]["sweep_parameters"][1]["value"], 5)
        self.assertEqual(summary["runs"][0]["artifact_count"], 4)
        self.assertEqual(first_manifest["status"], "completed")
        self.assertIn("# Neurobench Sweep Summary", report)
        self.assertIn("component_thresholds", report)
        self.assertIn("components.seed_z=1.4", report)

    def test_execute_parameter_sweep_records_failures_and_continues(self):
        require_numpy()
        from neurobench.pipelines.sweeps import execute_parameter_sweep

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = _executable_sweep_spec(str(root / "missing.npy"))
            summary = execute_parameter_sweep(spec, run_root=root / "sweep")

        self.assertEqual(summary["status"], "completed_with_failures")
        self.assertEqual(summary["succeeded"], 0)
        self.assertEqual(summary["failed"], 4)
        self.assertIn("missing.npy", summary["runs"][0]["error"])

    def test_render_sweep_summary_markdown_includes_failures(self):
        from neurobench.pipelines.sweeps import render_sweep_summary_markdown

        markdown = render_sweep_summary_markdown(
            {
                "dataset_id": "d",
                "sweep": {"id": "s"},
                "status": "completed_with_failures",
                "succeeded": 0,
                "failed": 1,
                "runs": [
                    {
                        "run_id": "r",
                        "status": "failed",
                        "artifact_count": 0,
                        "metrics": {"candidate_count": 3, "object_recall": 0.5},
                        "run_root": "001_r",
                        "sweep_parameters": [{"stage": "score", "param": "epsilon", "value": 0.1}],
                        "error_type": "RuntimeError",
                        "error": "boom",
                    }
                ],
            }
        )

        self.assertIn("## Failures", markdown)
        self.assertIn("RuntimeError: boom", markdown)
        self.assertIn("candidates=3", markdown)
        self.assertIn("obj recall=0.5", markdown)


if __name__ == "__main__":
    unittest.main()
