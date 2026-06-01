from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DocsWorkflowTests(unittest.TestCase):
    def test_raw_video_to_report_workflow_uses_relative_paths_and_stable_commands(self):
        path = ROOT / "docs" / "workflows" / "raw_video_to_report.md"
        text = path.read_text(encoding="utf-8")

        self.assertIn("python -m neurobench.cli.main dataset qc", text)
        self.assertIn("python -m neurobench.cli.main run execute", text)
        self.assertIn("python -m neurobench.cli.main report generate", text)
        self.assertIn("python -m neurobench.cli.main run sweep", text)
        self.assertIn("tools/export_annotations.py", text)
        self.assertIn("tools/export_inverse_dynamics.py", text)
        self.assertNotRegex(text, r"/home/|/Users/|[A-Za-z]:\\\\")

    def test_raw_video_to_report_embedded_pipeline_spec_is_valid_json(self):
        text = (ROOT / "docs" / "workflows" / "raw_video_to_report.md").read_text(encoding="utf-8")
        match = re.search(r"```json\n(\{\n  \"schema_version\".*?\n\})\n```", text, re.DOTALL)

        self.assertIsNotNone(match)
        payload = json.loads(match.group(1))
        self.assertEqual(payload["dataset_id"], "tutorial_synthetic")
        self.assertEqual(payload["pipeline"][0]["stage_id"], "source_video_import")

    def test_readme_links_raw_video_to_report_workflow(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("docs/README.md", readme)
        self.assertIn("docs/workflows/raw_video_to_report.md", readme)

    def test_docs_index_points_to_user_facing_workbench_pages(self):
        text = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

        for expected in [
            "NEURON_WORKBENCH.md",
            "RESTING_VIDEO_ALGORITHM_BRIEF.md",
            "ARCHITECTURE_LAB.md",
            "DATASET_QC.md",
            "METRICS_AUDIT.md",
            "ANNOTATION_SCHEMA.md",
            "developer/adding_pipeline_stage.md",
        ]:
            self.assertIn(expected, text)

    def test_workbench_docs_explain_current_dashboard_pages(self):
        text = (ROOT / "docs" / "NEURON_WORKBENCH.md").read_text(encoding="utf-8")

        for expected in [
            "Quick Start",
            "What To Use When",
            "Suggested Review Workflow",
            "Experiment Lab",
            "Data",
            "Progress",
            "Review Session",
        ]:
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
