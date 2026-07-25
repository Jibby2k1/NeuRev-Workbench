from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import urlopen
from urllib.request import Request


def _llm_proposal_set() -> dict:
    return {
        "schema_version": 1,
        "proposal_set_id": "server_import_v1",
        "dataset_id": "demo",
        "objective": "review_efficiency",
        "max_combinations_per_architecture": 16,
        "proposals": [
            {
                "id": "small_cfar",
                "label": "Small CFAR",
                "rationale": "Validate server-side import of a bounded proposal.",
                "hypothesis": "A compact sweep can be tested locally.",
                "priority": 1,
                "expected_tradeoffs": "Small synthetic test only.",
                "pipeline": [
                    {"id": "source", "stage_id": "source_video_import", "params": {"source": "raw.npy"}},
                    {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 2.0}},
                    {"id": "smooth", "stage_id": "spatial_gaussian", "params": {"sigma_px": 0.4}},
                    {"id": "cfar", "stage_id": "gamma_cfar", "params": {"pfa": 0.01, "guard_px": 1, "training_radius_px": 5}},
                ],
                "sweep": {"parameters": [{"stage": "cfar", "param": "pfa", "values": [0.01, 0.02]}]},
            }
        ],
    }


class WorkbenchServerTests(unittest.TestCase):
    def test_environment_report_has_generation_keys(self):
        from neurobench.workbench.server import environment_report

        report = environment_report()

        self.assertIn("fiji_available", report)
        self.assertIn("modules", report)
        self.assertIn("gpu", report)
        self.assertIn("cuda", report["gpu"])

    def test_generated_manifest_uses_whitelisted_app_paths(self):
        from neurobench.workbench.server import generated_dataset_manifest, load_json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_dir = root / "Outputs" / "NeuronReview" / "demo" / "app"
            app_dir.mkdir(parents=True)
            raw = root / "Inputs" / "demo.tif"
            raw.parent.mkdir()
            raw.write_bytes(b"fake")
            (app_dir / "review_data.json").write_text(
                json.dumps({"video": {"name": "demo.tif"}, "parameters": {"datasetId": "demo"}}),
                encoding="utf-8",
            )

            manifest_path = generated_dataset_manifest(app_dir, {"raw_video": str(raw)})
            manifest = load_json(manifest_path)

            self.assertEqual(manifest["dataset_id"], "demo")
            self.assertEqual(manifest["paths"]["app_dir"], str(app_dir))
            self.assertEqual(manifest["paths"]["raw_video"], str(raw))
            self.assertEqual(manifest["paths"]["architecture_runs"], str(app_dir / "architecture_runs.json"))

            run_app = app_dir / "generated_runs" / "planned_a"
            manifest_path = generated_dataset_manifest(app_dir, {"raw_video": str(raw)}, output_app_dir=run_app)
            manifest = load_json(manifest_path)
            self.assertEqual(manifest["paths"]["app_dir"], str(run_app))
            self.assertEqual(manifest["paths"]["review_data"], str(run_app / "review_data.json"))
            self.assertEqual(manifest["paths"]["architecture_runs"], str(app_dir / "architecture_runs.json"))

    def test_job_registry_rejects_duplicate_active_run(self):
        from neurobench.workbench.server import GenerationJob, JobRegistry

        with tempfile.TemporaryDirectory() as tmp:
            app_dir = Path(tmp) / "app"
            app_dir.mkdir()
            registry = JobRegistry()
            job = GenerationJob(app_dir=app_dir, payload={"run_id": "planned_a"})
            job.status = "running"
            registry.add(job)

            self.assertIs(registry.active_for(app_dir, "planned_a"), job)
            self.assertIsNone(registry.active_for(app_dir, "planned_b"))

    def test_run_generation_params_extracts_executable_knobs(self):
        from neurobench.workbench.server import run_generation_params

        run = {
            "run_id": "planned",
            "dataset_id": "demo",
            "pipeline": [
                {"id": "hp", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 8}},
                {"id": "components", "stage_id": "component_filter", "params": {"seed_z": 1.6, "grow_z": 0.8, "min_area_px": 3}},
                {"id": "events", "stage_id": "robust_kalman_positive_innovation", "params": {"event_threshold_z": 2.1}},
            ],
        }

        params = run_generation_params(run)

        self.assertEqual(params["sigma_label"], "08")
        self.assertEqual(params["component_seed_z"], 1.6)
        self.assertEqual(params["component_grow_z"], 0.8)
        self.assertEqual(params["component_min_area_px"], 3)
        self.assertEqual(params["event_threshold_z"], 2.1)

    def test_owner_token_matching_is_optional_and_exact(self):
        from neurobench.workbench.server import owner_token_matches, owner_token_required

        old = os.environ.pop("NEUROBENCH_OWNER_TOKEN", None)
        try:
            self.assertFalse(owner_token_required())
            self.assertTrue(owner_token_matches(None))
            os.environ["NEUROBENCH_OWNER_TOKEN"] = "secret"
            self.assertTrue(owner_token_required())
            self.assertTrue(owner_token_matches("secret"))
            self.assertFalse(owner_token_matches("wrong"))
            self.assertFalse(owner_token_matches(None))
        finally:
            if old is not None:
                os.environ["NEUROBENCH_OWNER_TOKEN"] = old
            else:
                os.environ.pop("NEUROBENCH_OWNER_TOKEN", None)

    def test_post_handlers_are_explicitly_whitelisted(self):
        from neurobench.workbench.server import WorkbenchHandler

        self.assertEqual(
            set(WorkbenchHandler.POST_HANDLERS),
            {
                ("jobs", "generate-view"),
                ("jobs", "generate-preview"),
                ("materialize-traces",),
                ("llm-proposals", "import"),
            },
        )
        self.assertEqual(WorkbenchHandler.POST_HANDLERS[("llm-proposals", "import")], "_handle_llm_proposal_import_post")

    def test_llm_proposal_import_validates_and_updates_architecture_runs(self):
        from neurobench.workbench.server import import_llm_proposals_into_app, load_json

        with tempfile.TemporaryDirectory() as tmp:
            app_dir = Path(tmp) / "app"
            app_dir.mkdir()
            (app_dir / "architecture_runs.json").write_text(
                json.dumps({"schema_version": 1, "dataset_id": "demo", "runs": []}),
                encoding="utf-8",
            )

            result = import_llm_proposals_into_app(app_dir, {"proposal": _llm_proposal_set()})
            manifest = load_json(app_dir / "architecture_runs.json")

            self.assertTrue(result["ok"])
            self.assertEqual(result["proposal_set_id"], "server_import_v1")
            self.assertEqual(result["validation_report"]["status"], "valid")
            self.assertEqual(len(result["run_ids"]), 2)
            self.assertEqual(len(result["saved_pipeline_ids"]), 1)
            self.assertEqual(len(manifest["runs"]), 2)
            self.assertEqual(manifest["llm_proposal_sets"][0]["proposal_set_id"], "server_import_v1")

    def test_llm_proposal_import_rejects_invalid_payload(self):
        from neurobench.workbench.server import import_llm_proposals_into_app

        with tempfile.TemporaryDirectory() as tmp:
            app_dir = Path(tmp) / "app"
            app_dir.mkdir()

            with self.assertRaisesRegex(ValueError, "required"):
                import_llm_proposals_into_app(app_dir, {"proposal": {"schema_version": 1}})

    def test_legacy_server_script_reexports_package_helpers(self):
        from neurobench.workbench import GenerationJob as PackageGenerationJob
        from neurobench.workbench import WorkbenchHandler as PackageWorkbenchHandler
        from tools.serve_neuron_workbench import GenerationJob as ToolGenerationJob
        from tools.serve_neuron_workbench import WorkbenchHandler as ToolWorkbenchHandler

        self.assertIs(ToolGenerationJob, PackageGenerationJob)
        self.assertIs(ToolWorkbenchHandler, PackageWorkbenchHandler)

    def test_server_factory_configures_single_dataset_app(self):
        from neurobench.workbench.server import WorkbenchHandler, create_workbench_server

        with tempfile.TemporaryDirectory() as tmp:
            app_dir = Path(tmp) / "app"
            app_dir.mkdir()
            (app_dir / "index.html").write_text("<!doctype html><title>demo</title>", encoding="utf-8")

            server, served = create_workbench_server(app_dir=app_dir, host="127.0.0.1", port=0)
            try:
                self.assertEqual(served, app_dir.resolve())
                self.assertEqual(WorkbenchHandler.app_dir, app_dir.resolve())
                self.assertIsNone(WorkbenchHandler.root_dir)
            finally:
                server.server_close()

    def test_server_exposes_canonical_dataset_record(self):
        from neurobench.workbench.server import create_workbench_server

        with tempfile.TemporaryDirectory() as tmp:
            app_dir = Path(tmp) / "Outputs" / "NeuronReview" / "declared" / "app"
            app_dir.mkdir(parents=True)
            (app_dir / "index.html").write_text(
                '<div id="manualRoiMode"></div><div id="cfarMaskAnnotationPanel"></div>',
                encoding="utf-8",
            )
            (app_dir / "review_data.json").write_text(
                json.dumps(
                    {
                        "dataset": {"dataset_id": "declared", "paths": {"raw_video": "Inputs/declared.tif"}},
                        "video": {"name": "declared.tif", "frames": 3, "width": 5, "height": 4, "framePattern": "frames/frame_%03d.png"},
                    }
                ),
                encoding="utf-8",
            )
            server, _ = create_workbench_server(app_dir=app_dir, host="127.0.0.1", port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = server.server_address[:2]
                with urlopen(f"http://{host}:{port}/api/dataset", timeout=5) as response:
                    payload = json.loads(response.read())
                with urlopen(f"http://{host}:{port}/api/datasets", timeout=5) as response:
                    catalog = json.loads(response.read())
                options_request = Request(
                    f"http://{host}:{port}/annotations.json",
                    method="OPTIONS",
                    headers={"Origin": "https://untrusted.example"},
                )
                with urlopen(options_request, timeout=5) as response:
                    cors_origin = response.headers.get("Access-Control-Allow-Origin")
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

        self.assertEqual(payload["dataset_id"], "declared")
        self.assertTrue(payload["capabilities"]["manual_roi_annotation"])
        self.assertTrue(payload["capabilities"]["cfar_annotation"])
        self.assertEqual(catalog["kind"], "neurobench_dataset_catalog")
        self.assertIsNone(cors_origin)

    def test_architecture_update_retains_unchanged_legacy_pipeline_only(self):
        from neurobench.workbench.server import validated_architecture_runs_update

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "architecture_runs.json"
            legacy = {
                "run_id": "legacy_soma",
                "dataset_id": "demo",
                "pipeline": [{"id": "source", "stage_id": "source_video_import", "params": {}}],
                "execution": {"status": "completed"},
            }
            path.write_text(
                json.dumps({"schema_version": 1, "dataset_id": "demo", "runs": [legacy]}),
                encoding="utf-8",
            )
            valid = {
                "schema_version": 1,
                "run_id": "new_valid",
                "dataset_id": "demo",
                "pipeline": [
                    {"id": "source", "stage_id": "source_video_import", "params": {"source": "raw.npy"}}
                ],
                "execution": {"status": "planned"},
                "artifacts": {},
            }
            updated = validated_architecture_runs_update(
                path,
                {"schema_version": 1, "dataset_id": "demo", "runs": [legacy, valid]},
            )
            modified = dict(legacy)
            modified["pipeline"] = [
                {"id": "source", "stage_id": "source_video_import", "params": {"different": True}}
            ]

            with self.assertRaisesRegex(ValueError, "invalid new or modified pipeline metadata"):
                validated_architecture_runs_update(
                    path,
                    {"schema_version": 1, "dataset_id": "demo", "runs": [modified]},
                )

        self.assertEqual([run["run_id"] for run in updated["runs"]], ["legacy_soma", "new_valid"])

    def test_architecture_update_rejects_nested_manifest_as_run(self):
        from neurobench.workbench.server import validated_architecture_runs_update

        nested_manifest = {
            "schema_version": 1,
            "dataset_id": "demo",
            "runs": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "architecture_runs.json"
            path.write_text(
                json.dumps({"schema_version": 1, "dataset_id": "demo", "runs": []}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing run_id"):
                validated_architecture_runs_update(
                    path,
                    {"schema_version": 1, "dataset_id": "demo", "runs": [nested_manifest]},
                )

    def test_server_factory_configures_root_index(self):
        from neurobench.workbench.server import WorkbenchHandler, configure_workbench_handler

        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp) / "NeuronReview"
            root_dir.mkdir()
            (root_dir / "index.html").write_text("<!doctype html><title>index</title>", encoding="utf-8")

            handler, served = configure_workbench_handler(root_dir=root_dir)

            self.assertIs(handler, WorkbenchHandler)
            self.assertEqual(served, root_dir.resolve())
            self.assertEqual(WorkbenchHandler.root_dir, root_dir.resolve())
            self.assertEqual(WorkbenchHandler.app_dir, root_dir.resolve())


if __name__ == "__main__":
    unittest.main()
