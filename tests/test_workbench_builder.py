from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _review_payload() -> dict:
    return {
        "video": {"name": "synthetic.npy", "width": 8, "height": 8, "frames": 4, "framePattern": "frames/frame_%03d.png"},
        "parameters": {"eventZThreshold": 2.4},
        "rois": [{"id": 1, "area": 12, "events": [{"frame": 2, "z": 3.1}], "dffTrace": [0, 0.1, 1.0, 0.2]}],
        "discovery": {"evidenceMaps": [], "suggestions": []},
    }


class WorkbenchBuilderTests(unittest.TestCase):
    def test_workbench_assets_packaged(self):
        from neurobench.workbench.builder import load_workbench_asset

        css = load_workbench_asset("workbench.css")
        js = load_workbench_asset("workbench.js")

        self.assertIn(".app", css)
        self.assertIn("traceEventCache", js)

    def test_build_workbench_outputs_html_assets_and_manifests(self):
        from neurobench.workbench.builder import build_workbench
        from tools import build_neuron_workbench_v2 as legacy_builder

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_path = root / "review_data.json"
            review_path.write_text(json.dumps(_review_payload()), encoding="utf-8")

            paths = build_workbench(
                app_dir=root / "app",
                review_data_path=review_path,
                dataset_id="synthetic_app",
                html_template=legacy_builder.HTML_TEMPLATE,
                dataset_manifest={"dataset_id": "synthetic_app", "paths": {"review_data": str(review_path)}},
                css_fallback=legacy_builder.CSS,
                js_fallback=legacy_builder.JS,
            )
            index = paths["index"].read_text(encoding="utf-8")
            embedded = index.split('<script id="review-data" type="application/json">', 1)[1].split("</script>", 1)[0]
            data = json.loads(embedded)
            annotations = json.loads(paths["annotations"].read_text(encoding="utf-8"))
            architecture_runs = json.loads(paths["architecture_runs"].read_text(encoding="utf-8"))

        self.assertIn("Neuron Annotation Workbench - synthetic_app", index)
        self.assertIn("workbench.css?v=", index)
        self.assertIn("workbench.js?v=", index)
        self.assertEqual(data["dataset"]["dataset_id"], "synthetic_app")
        self.assertIn("pipelineCatalog", data)
        self.assertEqual(annotations["schema_version"], 3)
        self.assertEqual(architecture_runs["runs"][0]["run_id"], "current_review_pipeline")
        self.assertTrue(paths["css"].name.endswith(".css"))
        self.assertTrue(paths["js"].name.endswith(".js"))

    def test_resolve_build_inputs_uses_review_data_dataset_id_without_manifest(self):
        from neurobench.workbench.builder import resolve_build_inputs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_path = root / "review_data.json"
            payload = _review_payload()
            payload["dataset"] = {"dataset_id": "external_test"}
            review_path.write_text(json.dumps(payload), encoding="utf-8")

            inputs = resolve_build_inputs(
                app_dir=root / "app",
                review_data=review_path,
                default_app_dir=root / "default_app",
                default_review_data=root / "missing_review_data.json",
                default_dataset_id="calcium_video_2",
            )

        self.assertEqual(inputs["dataset_id"], "external_test")
        self.assertEqual(inputs["review_data_path"], review_path.resolve())

    def test_build_workbench_preserves_review_dataset_without_manifest(self):
        from neurobench.workbench.builder import build_workbench
        from tools import build_neuron_workbench_v2 as legacy_builder

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_path = root / "review_data.json"
            payload = _review_payload()
            payload["dataset"] = {
                "dataset_id": "external_test",
                "raw_video": "Inputs/external_test/zebrafish_test.mp4",
            }
            review_path.write_text(json.dumps(payload), encoding="utf-8")

            paths = build_workbench(
                app_dir=root / "app",
                review_data_path=review_path,
                dataset_id="external_test",
                html_template=legacy_builder.HTML_TEMPLATE,
                css_fallback=legacy_builder.CSS,
                js_fallback=legacy_builder.JS,
            )
            index = paths["index"].read_text(encoding="utf-8")
            embedded = index.split('<script id="review-data" type="application/json">', 1)[1].split("</script>", 1)[0]
            data = json.loads(embedded)

        self.assertEqual(data["dataset"]["dataset_id"], "external_test")
        self.assertEqual(data["dataset"]["raw_video"], "Inputs/external_test/zebrafish_test.mp4")

    def test_legacy_build_script_uses_package_builder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_path = root / "review_data.json"
            app_dir = root / "app"
            review_path.write_text(json.dumps(_review_payload()), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "tools/build_neuron_workbench_v2.py",
                    "--review-data",
                    str(review_path),
                    "--app-dir",
                    str(app_dir),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            index_exists = (app_dir / "index.html").is_file()
            js_exists = (app_dir / "workbench.js").is_file()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Wrote workbench", result.stdout)
        self.assertTrue(index_exists)
        self.assertTrue(js_exists)


if __name__ == "__main__":
    unittest.main()
