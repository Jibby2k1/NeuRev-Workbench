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
                ("imports", "register"),
                ("imports", "upload"),
                ("imports", "promote"),
                ("imports", "metadata"),
                ("imports", "qc"),
                ("imports", "process"),
                ("labels", "preview"),
                ("labels", "import"),
                ("neurev", "preview"),
                ("neurev", "import"),
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
                handler = server.RequestHandlerClass
                self.assertIsNot(handler, WorkbenchHandler)
                self.assertTrue(issubclass(handler, WorkbenchHandler))
                self.assertEqual(handler.app_dir, app_dir.resolve())
                self.assertIsNone(handler.root_dir)
                self.assertEqual(handler.asset_mode, "current")
            finally:
                server.server_close()

    def test_current_dataset_route_serves_packaged_assets_without_writes(self):
        import neurobench.workbench.server as server_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_root = root / "Outputs" / "NeuronReview"
            app_dir = review_root / "demo" / "app"
            app_dir.mkdir(parents=True)
            (review_root / "index.html").write_text("catalog", encoding="utf-8")
            (app_dir / "index.html").write_text("stale installed html", encoding="utf-8")
            (app_dir / "workbench.css").write_text("stale installed css", encoding="utf-8")
            (app_dir / "workbench.js").write_text("stale installed js", encoding="utf-8")
            (app_dir / "review_data.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": {"dataset_id": "demo"},
                        "video": {"name": "demo.npy", "frames": 2, "width": 4, "height": 3, "framePattern": "frames/frame_%06d.png"},
                        "parameters": {},
                        "rois": [],
                    }
                ),
                encoding="utf-8",
            )

            def snapshot() -> tuple[tuple[str, ...], dict[str, bytes]]:
                directories = tuple(sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()))
                files = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
                return directories, files

            before = snapshot()
            previous_root = server_module.PROJECT_ROOT
            server_module.PROJECT_ROOT = root
            http_server, _ = server_module.create_workbench_server(root_dir=review_root, host="127.0.0.1", port=0, asset_mode="current")
            thread = threading.Thread(target=http_server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = http_server.server_address[:2]
                with urlopen(f"http://{host}:{port}/_datasets/demo/", timeout=5) as response:
                    html = response.read().decode("utf-8")
                with urlopen(f"http://{host}:{port}/_datasets/demo/workbench.css", timeout=5) as response:
                    css = response.read().decode("utf-8")
                with urlopen(f"http://{host}:{port}/_datasets/demo/workbench.js", timeout=5) as response:
                    javascript = response.read().decode("utf-8")
                with urlopen(f"http://{host}:{port}/api/datasets/demo/jobs", timeout=5) as response:
                    jobs = json.loads(response.read())
            finally:
                http_server.shutdown()
                thread.join(timeout=5)
                http_server.server_close()
                server_module.PROJECT_ROOT = previous_root
            after = snapshot()

        self.assertNotIn("stale installed html", html)
        self.assertNotIn("stale installed css", css)
        self.assertNotIn("stale installed js", javascript)
        self.assertEqual(jobs["durable_jobs"], [])
        self.assertEqual(before, after)

    def test_dataset_qualified_mutations_require_auth_and_cannot_cross_datasets(self):
        from urllib.error import HTTPError
        import neurobench.workbench.server as server_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_dir = root / "Outputs" / "NeuronReview" / "demo" / "app"
            app_dir.mkdir(parents=True)
            (app_dir / "index.html").write_text("demo", encoding="utf-8")
            source = root / "Inputs" / "demo" / "movie.npy"
            source.parent.mkdir(parents=True)
            import numpy as np

            np.save(source, np.zeros((2, 3, 4), dtype=np.uint16))
            previous_root = server_module.PROJECT_ROOT
            previous_token = os.environ.get("NEUROBENCH_OWNER_TOKEN")
            server_module.PROJECT_ROOT = root
            os.environ["NEUROBENCH_OWNER_TOKEN"] = "owner-secret"
            http_server, _ = server_module.create_workbench_server(app_dir=app_dir, host="127.0.0.1", port=0, asset_mode="installed")
            thread = threading.Thread(target=http_server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = http_server.server_address[:2]
                base = f"http://{host}:{port}"
                body = json.dumps({"dataset_id": "demo", "source_path": "Inputs/demo/movie.npy"}).encode("utf-8")
                with self.assertRaises(HTTPError) as unauthorized:
                    urlopen(Request(f"{base}/api/datasets/demo/imports/register", data=body, headers={"Content-Type": "application/json"}, method="POST"), timeout=5)
                self.assertEqual(unauthorized.exception.code, 401)
                self.assertFalse((app_dir / "imports").exists())
                request = Request(
                    f"{base}/api/datasets/demo/imports/register",
                    data=body,
                    headers={"Content-Type": "application/json", "X-Neurobench-Owner-Token": "owner-secret"},
                    method="POST",
                )
                with urlopen(request, timeout=5) as response:
                    registered = json.loads(response.read())
                cross_body = json.dumps({"dataset_id": "other", "source_path": "Inputs/demo/movie.npy"}).encode("utf-8")
                with self.assertRaises(HTTPError) as cross_error:
                    urlopen(Request(f"{base}/api/datasets/demo/imports/register", data=cross_body, headers={"Content-Type": "application/json", "X-Neurobench-Owner-Token": "owner-secret"}, method="POST"), timeout=5)
                self.assertEqual(cross_error.exception.code, 400)
                with self.assertRaises(HTTPError) as unknown_get:
                    urlopen(f"{base}/api/datasets/other/imports", timeout=5)
                self.assertEqual(unknown_get.exception.code, 404)
                put_body = b"{}"
                with self.assertRaises(HTTPError) as put_error:
                    urlopen(Request(f"{base}/api/datasets/demo/annotations", data=put_body, method="PUT"), timeout=5)
                self.assertEqual(put_error.exception.code, 401)
            finally:
                http_server.shutdown()
                thread.join(timeout=5)
                http_server.server_close()
                server_module.PROJECT_ROOT = previous_root
                if previous_token is None:
                    os.environ.pop("NEUROBENCH_OWNER_TOKEN", None)
                else:
                    os.environ["NEUROBENCH_OWNER_TOKEN"] = previous_token

        self.assertEqual(registered["import"]["dataset_id"], "demo")

    def test_configured_non_neuronreview_app_wins_same_id_registry_binding(self):
        import numpy as np
        import neurobench.workbench.server as server_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gamma_app = root / "Outputs" / "GammaCFAR" / "shared" / "app"
            neuron_app = root / "Outputs" / "NeuronReview" / "shared" / "app"
            for app, marker in ((gamma_app, "gamma"), (neuron_app, "neuron")):
                app.mkdir(parents=True)
                (app / "index.html").write_text(marker, encoding="utf-8")
                (app / "review_data.json").write_text(
                    json.dumps({"dataset": {"dataset_id": "shared", "name": marker}, "video": {"name": marker, "frames": 1, "width": 2, "height": 2, "framePattern": "frames/frame_%06d.png"}, "parameters": {}, "rois": []}),
                    encoding="utf-8",
                )
            source = root / "Inputs" / "shared" / "movie.npy"
            source.parent.mkdir(parents=True)
            np.save(source, np.zeros((1, 2, 2), dtype=np.uint16))
            neuron_before = {path.relative_to(neuron_app).as_posix(): path.read_bytes() for path in neuron_app.rglob("*") if path.is_file()}
            previous_root = server_module.PROJECT_ROOT
            server_module.PROJECT_ROOT = root
            http_server, _ = server_module.create_workbench_server(app_dir=gamma_app, host="127.0.0.1", port=0, asset_mode="installed")
            registry_binding = http_server.RequestHandlerClass.dataset_apps["shared"]
            registry_conflicts = dict(http_server.RequestHandlerClass.dataset_registry_conflicts)
            thread = threading.Thread(target=http_server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = http_server.server_address[:2]
                base = f"http://{host}:{port}"
                with urlopen(f"{base}/_datasets/shared/review_data.json", timeout=5) as response:
                    served_review = json.loads(response.read())
                body = json.dumps({"dataset_id": "shared", "source_path": "Inputs/shared/movie.npy"}).encode("utf-8")
                with urlopen(Request(f"{base}/api/datasets/shared/imports/register", data=body, headers={"Content-Type": "application/json"}, method="POST"), timeout=5) as response:
                    registered = json.loads(response.read())
                put_body = b"{\"schema_version\":3}"
                with urlopen(Request(f"{base}/api/datasets/shared/annotations", data=put_body, headers={"Content-Type": "application/json"}, method="PUT"), timeout=5) as response:
                    self.assertEqual(response.status, 200)
                gamma_record_exists = (gamma_app / "imports" / f"{registered['import']['import_id']}.json").is_file()
                gamma_annotations_exists = (gamma_app / "annotations.json").is_file()
                neuron_after = {path.relative_to(neuron_app).as_posix(): path.read_bytes() for path in neuron_app.rglob("*") if path.is_file()}
            finally:
                http_server.shutdown()
                thread.join(timeout=5)
                http_server.server_close()
                server_module.PROJECT_ROOT = previous_root

        self.assertEqual(served_review["video"]["name"], "gamma")
        self.assertEqual(registry_binding, gamma_app.resolve())
        self.assertEqual(len(registry_conflicts["shared"]), 2)
        self.assertTrue(gamma_record_exists)
        self.assertEqual(neuron_before, neuron_after)
        self.assertTrue(gamma_annotations_exists)

    def test_new_normalized_dataset_registers_into_catalog_then_promotes_and_qcs(self):
        from urllib.error import HTTPError
        import time as clock
        import numpy as np
        import neurobench.workbench.server as server_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_root = root / "Outputs" / "NeuronReview"
            review_root.mkdir(parents=True)
            (review_root / "index.html").write_text("catalog", encoding="utf-8")
            source = root / "Inputs" / "new-fish" / "movie.npy"
            source.parent.mkdir(parents=True)
            np.save(source, np.arange(24, dtype=np.uint16).reshape(2, 3, 4))
            previous_root = server_module.PROJECT_ROOT
            server_module.PROJECT_ROOT = root
            http_server, _ = server_module.create_workbench_server(root_dir=review_root, host="127.0.0.1", port=0, asset_mode="installed")
            thread = threading.Thread(target=http_server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = http_server.server_address[:2]
                base = f"http://{host}:{port}"
                with self.assertRaises(HTTPError) as unknown_process:
                    urlopen(Request(f"{base}/api/datasets/new-fish/imports/imp_missing/process", data=b"{}", headers={"Content-Type": "application/json"}, method="POST"), timeout=5)
                self.assertEqual(unknown_process.exception.code, 404)
                register_body = json.dumps({"dataset_id": "new-fish", "source_path": "Inputs/new-fish/movie.npy"}).encode("utf-8")
                with self.assertRaises(HTTPError) as alias_error:
                    urlopen(Request(f"{base}/api/datasets/New-Fish/imports/register", data=register_body, headers={"Content-Type": "application/json"}, method="POST"), timeout=5)
                self.assertEqual(alias_error.exception.code, 404)
                with urlopen(Request(f"{base}/api/datasets/new-fish/imports/register", data=register_body, headers={"Content-Type": "application/json"}, method="POST"), timeout=5) as response:
                    registered = json.loads(response.read())["import"]
                with urlopen(f"{base}/api/datasets", timeout=5) as response:
                    initial_catalog = json.loads(response.read())["datasets"]
                initial = next(item for item in initial_catalog if item["dataset_id"] == "new-fish")
                promote_body = b"{}"
                with urlopen(Request(f"{base}/api/datasets/new-fish/imports/{registered['import_id']}/promote", data=promote_body, headers={"Content-Type": "application/json"}, method="POST"), timeout=5) as response:
                    promoted = json.loads(response.read())["import"]
                with urlopen(Request(f"{base}/api/datasets/new-fish/imports/{registered['import_id']}/qc", data=b"{}", headers={"Content-Type": "application/json"}, method="POST"), timeout=5) as response:
                    qc_job = json.loads(response.read())["job"]
                durable = None
                for _ in range(100):
                    with urlopen(f"{base}/api/datasets/new-fish/durable-jobs/{qc_job['job_id']}", timeout=5) as response:
                        durable = json.loads(response.read())
                    if durable["status"] in {"completed", "failed", "stopped"}:
                        break
                    clock.sleep(0.02)
                with urlopen(f"{base}/api/datasets", timeout=5) as response:
                    final_catalog = json.loads(response.read())["datasets"]
                final = next(item for item in final_catalog if item["dataset_id"] == "new-fish")
            finally:
                http_server.shutdown()
                thread.join(timeout=5)
                http_server.server_close()
                server_module.PROJECT_ROOT = previous_root

        self.assertEqual(initial["lifecycle"]["state"], "import_only")
        self.assertFalse(initial["readiness"]["video_ready"])
        self.assertEqual(initial["imports"][0]["source_role"], "primary_video_candidate")
        self.assertTrue(initial["imports"][0]["source_available"])
        self.assertTrue(promoted["is_primary_video"])
        self.assertEqual(durable["status"], "completed")
        self.assertTrue(final["imports"][0]["is_primary_video"])
        self.assertTrue(final["imports"][0]["has_qc"])

    def test_failed_upload_is_durable_and_cleans_partial_files(self):
        from urllib.error import HTTPError
        import neurobench.workbench.server as server_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_root = root / "Outputs" / "NeuronReview"
            review_root.mkdir(parents=True)
            (review_root / "index.html").write_text("catalog", encoding="utf-8")
            previous_root = server_module.PROJECT_ROOT
            server_module.PROJECT_ROOT = root
            http_server, _ = server_module.create_workbench_server(root_dir=review_root, host="127.0.0.1", port=0, asset_mode="installed")
            thread = threading.Thread(target=http_server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = http_server.server_address[:2]
                request = Request(
                    f"http://{host}:{port}/api/datasets/bad-upload/imports/upload?filename=broken.npy",
                    data=b"not-a-valid-numpy-file",
                    headers={"Content-Type": "application/octet-stream"},
                    method="POST",
                )
                with self.assertRaises(HTTPError) as upload_error:
                    urlopen(request, timeout=5)
                self.assertEqual(upload_error.exception.code, 400)
                failure = json.loads(upload_error.exception.read())["import"]
                with urlopen(f"http://{host}:{port}/api/datasets/bad-upload/imports", timeout=5) as response:
                    listed = json.loads(response.read())["imports"]
                input_dir = root / "Inputs" / "bad-upload"
                leftovers = [path.name for path in input_dir.iterdir()] if input_dir.is_dir() else []
                destination_exists = (input_dir / "broken.npy").exists()
                sidecar_exists = (review_root / "bad-upload" / "app" / "imports" / f"{failure['import_id']}.json").is_file()
            finally:
                http_server.shutdown()
                thread.join(timeout=5)
                http_server.server_close()
                server_module.PROJECT_ROOT = previous_root

        self.assertEqual(failure["state"], "failed")
        self.assertFalse(failure["upload_failure"]["source_available"])
        self.assertEqual([item["import_id"] for item in listed], [failure["import_id"]])
        self.assertTrue(sidecar_exists)
        self.assertFalse(destination_exists)
        self.assertEqual(leftovers, [])

    def test_json_upload_rejects_arbitrary_and_oversized_payloads(self):
        from urllib.error import HTTPError
        import neurobench.workbench.server as server_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_root = root / "Outputs" / "NeuronReview"
            review_root.mkdir(parents=True)
            (review_root / "index.html").write_text("catalog", encoding="utf-8")
            previous_root = server_module.PROJECT_ROOT
            server_module.PROJECT_ROOT = root
            http_server, _ = server_module.create_workbench_server(root_dir=review_root, host="127.0.0.1", port=0, asset_mode="installed")
            thread = threading.Thread(target=http_server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = http_server.server_address[:2]
                base = f"http://{host}:{port}/api/datasets/json-reject/imports/upload"
                arbitrary = b'{"hello":"world"}'
                with self.assertRaises(HTTPError) as arbitrary_error:
                    urlopen(
                        Request(
                            f"{base}?filename=arbitrary.json",
                            data=arbitrary,
                            headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(arbitrary))},
                            method="POST",
                        ),
                        timeout=5,
                    )
                arbitrary_failure = json.loads(arbitrary_error.exception.read())["import"]
                with self.assertRaises(HTTPError) as oversize_error:
                    urlopen(
                        Request(
                            f"{base}?filename=oversize.json",
                            data=b"x",
                            headers={
                                "Content-Type": "application/octet-stream",
                                "Content-Length": str(server_module.MAX_NEUREV_JSON_BYTES + 1),
                            },
                            method="POST",
                        ),
                        timeout=5,
                    )
                oversize_body = json.loads(oversize_error.exception.read())
                input_dir = root / "Inputs" / "json-reject"
                leftovers = [path.name for path in input_dir.iterdir()] if input_dir.is_dir() else []
            finally:
                http_server.shutdown()
                thread.join(timeout=5)
                http_server.server_close()
                server_module.PROJECT_ROOT = previous_root

        self.assertEqual(arbitrary_error.exception.code, 400)
        self.assertEqual(arbitrary_failure["metadata"]["kind"], "neurev_json")
        self.assertEqual(arbitrary_failure["state"], "failed")
        self.assertEqual(oversize_error.exception.code, 413)
        self.assertIn("safety limit", oversize_body["error"])
        self.assertEqual(leftovers, [])

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

            self.assertIsNot(handler, WorkbenchHandler)
            self.assertTrue(issubclass(handler, WorkbenchHandler))
            self.assertEqual(served, root_dir.resolve())
            self.assertEqual(handler.root_dir, root_dir.resolve())
            self.assertEqual(handler.app_dir, root_dir.resolve())


if __name__ == "__main__":
    unittest.main()
