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


def _embedded_review_data(index: str) -> dict:
    embedded = index.split('<script id="review-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    return json.loads(embedded)


class WorkbenchBuilderTests(unittest.TestCase):
    def test_workbench_assets_packaged(self):
        from neurobench.workbench.builder import load_workbench_asset, workbench_asset_version

        css = load_workbench_asset("workbench.css")
        js = load_workbench_asset("workbench.js")

        self.assertIn(".app", css)
        self.assertIn("traceEventCache", js)
        self.assertEqual(len(workbench_asset_version()), 12)
        self.assertNotEqual(workbench_asset_version(html_text="one"), workbench_asset_version(html_text="two"))

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

        self.assertIn("<title>NeuRev - synthetic_app</title>", index)
        self.assertIn("workbench.css?v=", index)
        self.assertIn("workbench.js?v=", index)
        self.assertEqual(data["dataset"]["dataset_id"], "synthetic_app")
        self.assertIn("pipelineCatalog", data)
        self.assertEqual(annotations["schema_version"], 3)
        self.assertEqual(architecture_runs, {"schema_version": 1, "dataset_id": "synthetic_app", "runs": []})
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

    def test_resolve_build_inputs_uses_legacy_parameter_dataset_id(self):
        from neurobench.workbench.builder import resolve_build_inputs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_path = root / "Outputs" / "NeuronReview" / "legacy_folder" / "app" / "review_data.json"
            review_path.parent.mkdir(parents=True)
            payload = _review_payload()
            payload["parameters"]["datasetId"] = "legacy_declared"
            review_path.write_text(json.dumps(payload), encoding="utf-8")
            inputs = resolve_build_inputs(
                review_data=review_path,
                default_app_dir=review_path.parent,
                default_review_data=review_path,
                default_dataset_id="dataset",
            )

        self.assertEqual(inputs["dataset_id"], "legacy_declared")

    def test_manifest_placeholder_catalog_allows_new_app_baseline(self):
        from neurobench.workbench.builder import build_workbench, resolve_build_inputs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_dir = root / "new_dataset" / "app"
            review_path = app_dir / "review_data.json"
            review_path.parent.mkdir(parents=True)
            review_path.write_text(json.dumps(_review_payload()), encoding="utf-8")
            manifest_path = root / "dataset_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset_id": "new_dataset",
                        "paths": {
                            "app_dir": str(app_dir),
                            "review_data": str(review_path),
                            "architecture_runs": str(app_dir / "architecture_runs.json"),
                        },
                    }
                ),
                encoding="utf-8",
            )
            inputs = resolve_build_inputs(
                dataset_manifest=manifest_path,
                default_app_dir=app_dir,
                default_review_data=review_path,
                default_dataset_id="dataset",
            )
            paths = build_workbench(
                app_dir=inputs["app_dir"],
                review_data_path=inputs["review_data_path"],
                dataset_id=inputs["dataset_id"],
                dataset_manifest=inputs["dataset_manifest"],
                architecture_runs_path=inputs["architecture_runs_path"],
            )
            runs = json.loads(paths["architecture_runs"].read_text(encoding="utf-8"))

        self.assertIsNone(inputs["architecture_runs_path"])
        self.assertEqual(runs, {"schema_version": 1, "dataset_id": "new_dataset", "runs": []})

    def test_in_memory_renderer_preserves_review_and_attached_run_bytes(self):
        from neurobench.workbench.builder import render_workbench_assets

        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp) / "historical" / "app"
            app.mkdir(parents=True)
            review_path = app / "review_data.json"
            review_bytes = json.dumps(_review_payload(), indent=1).encode("utf-8") + b"\n"
            annotation_bytes = b'{ "schema_version" : 2, "reviewer" : "historical" }\n'
            run_bytes = b'{"schema_version":1,"dataset_id":"historical","runs":[{"run_id":"attached_scientific_run"}]}\n'
            review_path.write_bytes(review_bytes)
            (app / "annotations.json").write_bytes(annotation_bytes)
            (app / "architecture_runs.json").write_bytes(run_bytes)
            before = {path.name: path.read_bytes() for path in app.iterdir()}

            rendered = render_workbench_assets(
                app_dir=app,
                review_data_path=review_path,
                dataset_id="historical",
            )

            after = {path.name: path.read_bytes() for path in app.iterdir()}
            embedded = _embedded_review_data(rendered["index.html"].decode("utf-8"))

        self.assertEqual(set(rendered), {"index.html", "workbench.css", "workbench.js"})
        self.assertTrue(all(isinstance(value, bytes) for value in rendered.values()))
        self.assertEqual(before, after)
        self.assertEqual(embedded["architectureRuns"]["runs"][0]["run_id"], "attached_scientific_run")

    def test_rebuild_byte_preserves_nonempty_historical_artifacts_by_default(self):
        from neurobench.workbench.builder import build_workbench

        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp) / "historical" / "app"
            app.mkdir(parents=True)
            review_path = app / "review_data.json"
            review_bytes = json.dumps(_review_payload(), indent=3).encode("utf-8") + b"\n"
            annotation_bytes = b'{ "schema_version" : 2, "reviewer" : "historical", "decisions" : [1] }\n'
            run_bytes = b'{ "schema_version" : 1, "dataset_id" : "historical", "runs" : [{"run_id":"valuable"}] }\n'
            review_path.write_bytes(review_bytes)
            (app / "annotations.json").write_bytes(annotation_bytes)
            (app / "architecture_runs.json").write_bytes(run_bytes)
            protected = {
                name: (app / name).read_bytes()
                for name in ("review_data.json", "annotations.json", "architecture_runs.json")
            }

            paths = build_workbench(
                app_dir=app,
                review_data_path=review_path,
                dataset_id="historical",
            )

            after = {name: (app / name).read_bytes() for name in protected}

        self.assertEqual(after, protected)
        self.assertTrue(paths["index"].name.endswith(".html"))

    def test_explicit_annotation_migration_is_the_only_rewrite_path(self):
        from neurobench.workbench.builder import build_workbench

        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp) / "app"
            app.mkdir()
            review_path = app / "review_data.json"
            review_path.write_text(json.dumps(_review_payload()), encoding="utf-8")
            original = b'{"schema_version":2,"settings":{"reviewWorkflowPreset":"cfar_mask"}}\n'
            (app / "annotations.json").write_bytes(original)
            (app / "architecture_runs.json").write_text(
                json.dumps({"schema_version": 1, "dataset_id": "demo", "runs": []}),
                encoding="utf-8",
            )

            build_workbench(
                app_dir=app,
                review_data_path=review_path,
                dataset_id="demo",
                migrate_annotations=True,
            )
            migrated_bytes = (app / "annotations.json").read_bytes()
            migrated = json.loads(migrated_bytes)

        self.assertNotEqual(migrated_bytes, original)
        self.assertEqual(migrated["schema_version"], 3)

    def test_asset_status_rejects_tampered_js_even_with_current_html_marker(self):
        from neurobench.workbench.builder import build_workbench, workbench_asset_status

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_path = root / "review_data.json"
            review_path.write_text(json.dumps(_review_payload()), encoding="utf-8")
            paths = build_workbench(
                app_dir=root / "app",
                review_data_path=review_path,
                dataset_id="demo",
            )
            before = workbench_asset_status(root / "app")
            paths["js"].write_bytes(paths["js"].read_bytes() + b"// tampered\n")
            after = workbench_asset_status(root / "app")

        self.assertTrue(before["current"])
        self.assertTrue(after["marker_current"])
        self.assertTrue(after["css_current"])
        self.assertFalse(after["js_current"])
        self.assertFalse(after["current"])

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

    def test_rebuild_preserves_existing_architecture_run_catalog(self):
        from neurobench.workbench.builder import build_workbench
        from tools import build_neuron_workbench_v2 as legacy_builder

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_dir = root / "app"
            app_dir.mkdir()
            review_path = root / "review_data.json"
            review_path.write_text(json.dumps(_review_payload()), encoding="utf-8")
            catalog_text = '{"schema_version":1,"dataset_id":"synthetic_app","runs":[{"run_id":"attached_scientific_run"}]}\n'
            catalog_path = app_dir / "architecture_runs.json"
            catalog_path.write_text(catalog_text, encoding="utf-8")

            paths = build_workbench(
                app_dir=app_dir,
                review_data_path=review_path,
                dataset_id="synthetic_app",
                html_template=legacy_builder.HTML_TEMPLATE,
                css_fallback=legacy_builder.CSS,
                js_fallback=legacy_builder.JS,
            )
            index = paths["index"].read_text(encoding="utf-8")
            embedded = index.split('<script id="review-data" type="application/json">', 1)[1].split("</script>", 1)[0]
            data = json.loads(embedded)
            preserved_text = paths["architecture_runs"].read_text(encoding="utf-8")

        self.assertEqual(preserved_text, catalog_text)
        self.assertEqual(data["architectureRuns"]["runs"][0]["run_id"], "attached_scientific_run")

    def test_missing_explicit_architecture_catalog_fails_without_clobbering_existing_catalog(self):
        from neurobench.workbench.builder import build_workbench

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_dir = root / "app"
            app_dir.mkdir()
            review_path = root / "review_data.json"
            review_path.write_text(json.dumps(_review_payload()), encoding="utf-8")
            catalog_path = app_dir / "architecture_runs.json"
            catalog_text = '{"schema_version":1,"dataset_id":"synthetic_app","runs":[{"run_id":"valuable"}]}\n'
            catalog_path.write_text(catalog_text, encoding="utf-8")
            before = {path.name: path.read_bytes() for path in app_dir.iterdir()}

            with self.assertRaisesRegex(FileNotFoundError, "Architecture-run catalog does not exist"):
                build_workbench(
                    app_dir=app_dir,
                    review_data_path=review_path,
                    dataset_id="synthetic_app",
                    architecture_runs_path=root / "missing_architecture_runs.json",
                )

            preserved_text = catalog_path.read_text(encoding="utf-8")
            after = {path.name: path.read_bytes() for path in app_dir.iterdir()}

        self.assertEqual(preserved_text, catalog_text)
        self.assertEqual(after, before)

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
