from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np


class WorkbenchImportFlowTests(unittest.TestCase):
    @staticmethod
    def _annotations_payload(state: str = "accepted") -> dict:
        return {
            "schema_version": 3,
            "rois": {"r1": {"cell_state": state}},
            "events": {},
            "suggestions": {},
            "settings": {},
        }

    def test_video_import_qc_and_process_are_bounded_and_resume_safe(self):
        import neurobench.workbench.server as server
        from neurobench.workbench.jobs import JobStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "Inputs" / "demo" / "movie.npy"
            source.parent.mkdir(parents=True)
            np.save(source, np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5))
            previous_root = server.PROJECT_ROOT
            server.PROJECT_ROOT = root
            try:
                record = server.register_import_source(dataset_id="demo", source=source, source_mode="local_registration", destination_path="Inputs/demo/movie.npy")
                app = root / "Outputs" / "NeuronReview" / "demo" / "app"
                self.assertFalse((app / "dataset_manifest.generated.json").exists())
                record = server.promote_import_primary_video(record)
                store = JobStore(root / "jobs")
                qc_job = store.create("dataset_qc")
                server.execute_import_qc_job(store, qc_job["job_id"], record)
                qc_record = server.import_record_for("demo", record["import_id"])
                process_job = store.create("dataset_process")
                server.execute_import_process_job(store, process_job["job_id"], qc_record)
                process_result = store.get(process_job["job_id"])
                final_record = server.import_record_for("demo", record["import_id"])
                qc_status = store.get(qc_job["job_id"])["status"]
                app_exists = (app / "index.html").is_file()
                frame_count = len(list((app / "frames").glob("*.png")))
                architecture_runs = json.loads((app / "architecture_runs.json").read_text(encoding="utf-8"))
            finally:
                server.PROJECT_ROOT = previous_root

        self.assertEqual(qc_status, "completed")
        self.assertEqual(process_result["status"], "completed")
        self.assertEqual(final_record["state"], "ready")
        self.assertTrue(app_exists)
        self.assertEqual(frame_count, 3)
        self.assertEqual(architecture_runs["runs"], [])

    def test_process_refuses_existing_workbench_artifacts(self):
        import neurobench.workbench.server as server
        from neurobench.workbench.jobs import JobStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "Outputs" / "NeuronReview" / "demo" / "app"
            app.mkdir(parents=True)
            review = app / "review_data.json"
            review.write_text("{\"keep\": true}\n", encoding="utf-8")
            source = root / "Inputs" / "demo" / "movie.npy"
            source.parent.mkdir(parents=True)
            np.save(source, np.zeros((2, 3, 4), dtype=np.uint16))
            previous_root = server.PROJECT_ROOT
            server.PROJECT_ROOT = root
            try:
                record = server.register_import_source(dataset_id="demo", source=source, source_mode="local_registration", destination_path="Inputs/demo/movie.npy")
                record = server.promote_import_primary_video(record)
                store = JobStore(root / "jobs")
                qc_job = store.create("dataset_qc")
                server.execute_import_qc_job(store, qc_job["job_id"], record)
                record = server.import_record_for("demo", record["import_id"])
                job = store.create("dataset_process")
                server.execute_import_process_job(store, job["job_id"], record)
                result = store.get(job["job_id"])
                review_content = review.read_text(encoding="utf-8")
            finally:
                server.PROJECT_ROOT = previous_root

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stage"], "collision")
        self.assertEqual(review_content, "{\"keep\": true}\n")

    def test_http_register_upload_label_preview_and_durable_job(self):
        import neurobench.workbench.server as server

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "Outputs" / "NeuronReview" / "demo" / "app"
            app.mkdir(parents=True)
            (app / "index.html").write_text("<!doctype html><title>demo</title>\n", encoding="utf-8")
            (app / "review_data.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": {"dataset_id": "demo"},
                        "video": {"name": "movie.npy", "frames": 3, "width": 5, "height": 4, "framePattern": "frames/frame_%06d.png"},
                        "parameters": {},
                        "rois": [{"id": "r1"}],
                    }
                ),
                encoding="utf-8",
            )
            annotations_path = app / "annotations.json"
            annotations_path.write_text("{\"sentinel\":true}\n", encoding="utf-8")
            movie = root / "Inputs" / "demo" / "movie.npy"
            movie.parent.mkdir(parents=True)
            np.save(movie, np.zeros((3, 4, 5), dtype=np.uint16))
            labels = root / "Inputs" / "demo" / "labels.tsv"
            labels.write_text("roi_id\tlabel\nr1\tneuron\n", encoding="utf-8")
            previous_root = server.PROJECT_ROOT
            server.PROJECT_ROOT = root
            http_server, _ = server.create_workbench_server(app_dir=app, host="127.0.0.1", port=0)
            thread = __import__("threading").Thread(target=http_server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = http_server.server_address[:2]
                base = f"http://{host}:{port}"
                body = json.dumps({"dataset_id": "demo", "source_path": "Inputs/demo/movie.npy", "promote_primary_video": True}).encode("utf-8")
                with urlopen(Request(f"{base}/api/imports/register", data=body, headers={"Content-Type": "application/json"}, method="POST"), timeout=5) as response:
                    registered = json.loads(response.read())
                manifest_path = app / "dataset_manifest.generated.json"
                manifest_before_labels = manifest_path.read_bytes()
                upload_bytes = movie.read_bytes()
                with urlopen(Request(f"{base}/api/imports/upload?dataset_id=demo&filename=uploaded.npy", data=upload_bytes, headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(upload_bytes))}, method="POST"), timeout=5) as response:
                    uploaded = json.loads(response.read())
                with self.assertRaises(HTTPError) as duplicate_error:
                    urlopen(Request(f"{base}/api/imports/upload?dataset_id=demo&filename=uploaded.npy", data=upload_bytes, headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(upload_bytes))}, method="POST"), timeout=5)
                self.assertEqual(duplicate_error.exception.code, 409)
                label_body = json.dumps({"dataset_id": "demo", "source_path": "Inputs/demo/labels.tsv"}).encode("utf-8")
                with urlopen(Request(f"{base}/api/imports/register", data=label_body, headers={"Content-Type": "application/json"}, method="POST"), timeout=5) as response:
                    label_record = json.loads(response.read())["import"]
                manifest_after_labels = manifest_path.read_bytes()
                preview_body = json.dumps({"import_id": label_record["import_id"]}).encode("utf-8")
                with urlopen(Request(f"{base}/api/labels/preview", data=preview_body, headers={"Content-Type": "application/json"}, method="POST"), timeout=5) as response:
                    preview = json.loads(response.read())
                with self.assertRaises(HTTPError) as unconfirmed_error:
                    urlopen(Request(f"{base}/api/labels/import", data=preview_body, headers={"Content-Type": "application/json"}, method="POST"), timeout=5)
                self.assertEqual(unconfirmed_error.exception.code, 400)
                confirmed_body = json.dumps({"import_id": label_record["import_id"], "confirmed": True}).encode("utf-8")
                with urlopen(Request(f"{base}/api/labels/import", data=confirmed_body, headers={"Content-Type": "application/json"}, method="POST"), timeout=5) as response:
                    label_job = json.loads(response.read())["job"]
                durable = None
                for _ in range(50):
                    with urlopen(f"{base}/api/durable-jobs/{label_job['job_id']}", timeout=5) as response:
                        durable = json.loads(response.read())
                    if durable["status"] in {"completed", "failed", "stopped"}:
                        break
                    time.sleep(0.02)
                annotations_after = annotations_path.read_text(encoding="utf-8")
                external_labels = json.loads(Path(durable["outputs"]["external_labels"]).read_text(encoding="utf-8")) if durable and durable["status"] == "completed" else None
            finally:
                http_server.shutdown()
                thread.join(timeout=5)
                http_server.server_close()
                server.PROJECT_ROOT = previous_root

        self.assertEqual(registered["import"]["metadata"]["kind"], "video")
        self.assertEqual(uploaded["import"]["source_mode"], "upload")
        self.assertEqual(preview["sample_rows"][0]["roi_id"], "r1")
        self.assertEqual(durable["status"], "completed")
        self.assertEqual(manifest_before_labels, manifest_after_labels)
        self.assertEqual(annotations_after, "{\"sentinel\":true}\n")
        self.assertEqual(external_labels["summary"]["matched_rows"], 1)

    def test_label_import_is_streamed_and_preserves_external_rows(self):
        import neurobench.workbench.server as server
        from neurobench.workbench.jobs import JobStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "Inputs" / "demo" / "labels.tsv"
            source.parent.mkdir(parents=True)
            source.write_text(
                "roi_id\tx\ty\tstart_frame\tend_frame\tlabel\tconfidence\n"
                "r1\t1\t1\t1\t2\tneuron\t0.9\n"
                "r1\t2\t2\t2\t3\tduplicate\t0.8\n"
                "r2\t3\t2\t1\t1\tunsure\t0.4\n"
                "\t1\t1\t0\t2\tbad\t0.5\n",
                encoding="utf-8",
            )
            app = root / "Outputs" / "NeuronReview" / "demo" / "app"
            app.mkdir(parents=True)
            (app / "review_data.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": {"dataset_id": "demo"},
                        "video": {"name": "movie.npy", "frames": 3, "width": 5, "height": 4, "framePattern": "frames/frame_%06d.png"},
                        "parameters": {},
                        "rois": [{"id": "r1"}],
                    }
                ),
                encoding="utf-8",
            )
            annotations_path = app / "annotations.json"
            annotations_path.write_text("{\"keep\":true}\n", encoding="utf-8")
            annotations_before = annotations_path.read_bytes()
            previous_root = server.PROJECT_ROOT
            server.PROJECT_ROOT = root
            try:
                record = server.register_import_source(
                    dataset_id="demo",
                    source=source,
                    source_mode="local_registration",
                    destination_path="Inputs/demo/labels.tsv",
                    app_dir=app,
                )
                store = JobStore(app / ".neurobench" / "jobs")
                job = store.create("label_import", {"dataset_id": "demo", "import_id": record["import_id"]})
                server.execute_label_import_job(store, job["job_id"], record)
                result = store.get(job["job_id"])
                artifact_path = app / "external_labels" / f"{record['import_id']}.json"
                overlay_path = app / "external_labels" / f"{record['import_id']}.overlay.svg"
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                overlay_exists = overlay_path.is_file()
                annotations_after = annotations_path.read_bytes()
                final_record = server.import_record_for("demo", record["import_id"], app_dir=app)
            finally:
                server.PROJECT_ROOT = previous_root

        self.assertEqual(result["status"], "completed")
        self.assertEqual(annotations_before, annotations_after)
        self.assertTrue(overlay_exists)
        self.assertEqual(list(artifact["rows"]), ["row_00000002", "row_00000003", "row_00000004", "row_00000005"])
        self.assertEqual(artifact["rows"]["row_00000002"]["normalized"]["label"], "neuron")
        self.assertEqual(artifact["rows"]["row_00000003"]["reconciliation"]["classifications"], ["matched", "duplicate"])
        self.assertEqual(artifact["rows"]["row_00000004"]["reconciliation"]["status"], "unmatched")
        self.assertEqual(artifact["summary"], {"total_rows": 4, "matched_rows": 2, "unmatched_rows": 1, "duplicate_rows": 1, "rejected_rows": 1})
        self.assertEqual(final_record["state"], "complete")

    def test_neurev_json_register_upload_preview_confirm_and_preserve_native_state(self):
        import neurobench.workbench.server as server

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "Outputs" / "NeuronReview" / "demo" / "app"
            app.mkdir(parents=True)
            (app / "index.html").write_text("<!doctype html><title>demo</title>\n", encoding="utf-8")
            (app / "review_data.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": {"dataset_id": "demo"},
                        "video": {"name": "movie", "frames": 2, "width": 4, "height": 3, "framePattern": "frames/frame_%06d.png"},
                        "parameters": {},
                        "rois": [{"id": "r1"}],
                    }
                ),
                encoding="utf-8",
            )
            native_annotations = b'{"native_annotations":"keep exactly"}\n'
            native_runs = b'{"native_runs":"keep exactly"}\n'
            native_manifest = b'{"native_manifest":"keep exactly"}\n'
            (app / "annotations.json").write_bytes(native_annotations)
            (app / "architecture_runs.json").write_bytes(native_runs)
            (app / "dataset_manifest.generated.json").write_bytes(native_manifest)
            input_dir = root / "Inputs" / "demo"
            input_dir.mkdir(parents=True)
            registered_source = input_dir / "external_annotations.json"
            registered_source.write_text(json.dumps(self._annotations_payload()), encoding="utf-8")
            uploaded_payload = json.dumps(
                {"schema_version": 1, "dataset_id": "demo", "runs": []},
                indent=1,
            ).encode("utf-8") + b"\n"
            previous_root = server.PROJECT_ROOT
            server.PROJECT_ROOT = root
            http_server, _ = server.create_workbench_server(app_dir=app, host="127.0.0.1", port=0)
            thread = __import__("threading").Thread(target=http_server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = http_server.server_address[:2]
                base = f"http://{host}:{port}/api/datasets/demo"
                register_body = json.dumps({"source_path": "Inputs/demo/external_annotations.json"}).encode("utf-8")
                with urlopen(Request(f"{base}/imports/register", data=register_body, headers={"Content-Type": "application/json"}, method="POST"), timeout=5) as response:
                    registered = json.loads(response.read())["import"]
                preview_body = json.dumps({"import_id": registered["import_id"]}).encode("utf-8")
                with urlopen(Request(f"{base}/neurev/preview", data=preview_body, headers={"Content-Type": "application/json"}, method="POST"), timeout=5) as response:
                    registered_preview = json.loads(response.read())

                upload_url = f"{base}/imports/upload?filename=external_runs.json"
                with urlopen(Request(upload_url, data=uploaded_payload, headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(uploaded_payload))}, method="POST"), timeout=5) as response:
                    uploaded = json.loads(response.read())["import"]
                upload_preview_body = json.dumps({"import_id": uploaded["import_id"]}).encode("utf-8")
                with urlopen(Request(f"{base}/neurev/preview", data=upload_preview_body, headers={"Content-Type": "application/json"}, method="POST"), timeout=5) as response:
                    uploaded_preview = json.loads(response.read())
                with self.assertRaises(HTTPError) as unconfirmed_error:
                    urlopen(Request(f"{base}/neurev/import", data=upload_preview_body, headers={"Content-Type": "application/json"}, method="POST"), timeout=5)
                self.assertEqual(unconfirmed_error.exception.code, 400)
                confirmed = json.dumps({"import_id": uploaded["import_id"], "confirmed": True}).encode("utf-8")
                with urlopen(Request(f"{base}/neurev/import", data=confirmed, headers={"Content-Type": "application/json"}, method="POST"), timeout=5) as response:
                    job = json.loads(response.read())["job"]
                durable = None
                for _ in range(50):
                    with urlopen(f"{base}/durable-jobs/{job['job_id']}", timeout=5) as response:
                        durable = json.loads(response.read())
                    if durable["status"] in {"completed", "failed", "stopped"}:
                        break
                    time.sleep(0.02)
                external_copy = Path(durable["outputs"]["external_neurev_json"]).read_bytes() if durable and durable["status"] == "completed" else b""
                final_record = server.import_record_for("demo", uploaded["import_id"], app_dir=app)
                native_after = (
                    (app / "annotations.json").read_bytes(),
                    (app / "architecture_runs.json").read_bytes(),
                    (app / "dataset_manifest.generated.json").read_bytes(),
                )
            finally:
                http_server.shutdown()
                thread.join(timeout=5)
                http_server.server_close()
                server.PROJECT_ROOT = previous_root

        self.assertEqual(registered["metadata"]["kind"], "neurev_json")
        self.assertEqual(registered_preview["payload_kind"], "annotations")
        self.assertEqual(registered_preview["counts"]["roi_annotation_count"], 1)
        self.assertNotIn("payload", registered["metadata"])
        self.assertEqual(uploaded_preview["payload_kind"], "architecture_runs")
        self.assertEqual(external_copy, uploaded_payload)
        self.assertEqual(final_record["state"], "complete")
        self.assertEqual(native_after, (native_annotations, native_runs, native_manifest))

    def test_neurev_json_confirmation_rejects_source_mutation(self):
        import neurobench.workbench.server as server
        from neurobench.workbench.jobs import JobStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "Outputs" / "NeuronReview" / "demo" / "app"
            app.mkdir(parents=True)
            native_annotations = b'{"keep":true}\n'
            native_runs = b'{"keep_runs":true}\n'
            native_manifest = b'{"keep_manifest":true}\n'
            (app / "annotations.json").write_bytes(native_annotations)
            (app / "architecture_runs.json").write_bytes(native_runs)
            (app / "dataset_manifest.generated.json").write_bytes(native_manifest)
            source = root / "Inputs" / "demo" / "external.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(self._annotations_payload("accepted")), encoding="utf-8")
            previous_root = server.PROJECT_ROOT
            server.PROJECT_ROOT = root
            try:
                record = server.register_import_source(
                    dataset_id="demo",
                    source=source,
                    source_mode="local_registration",
                    destination_path="Inputs/demo/external.json",
                    app_dir=app,
                )
                source.write_text(json.dumps(self._annotations_payload("rejected")), encoding="utf-8")
                store = JobStore(app / ".neurobench" / "jobs")
                job = store.create("neurev_json_import")
                server.execute_neurev_json_import_job(store, job["job_id"], record)
                result = store.get(job["job_id"])
                final_record = server.import_record_for("demo", record["import_id"], app_dir=app)
                artifact_exists = (app / "external_neurev" / f"{record['import_id']}.json").exists()
                native_after = (
                    (app / "annotations.json").read_bytes(),
                    (app / "architecture_runs.json").read_bytes(),
                    (app / "dataset_manifest.generated.json").read_bytes(),
                )
            finally:
                server.PROJECT_ROOT = previous_root

        self.assertEqual(result["status"], "failed")
        self.assertIn("checksum changed", result["error"])
        self.assertEqual(final_record["state"], "failed")
        self.assertFalse(artifact_exists)
        self.assertEqual(native_after, (native_annotations, native_runs, native_manifest))

    def test_label_reconciliation_rejects_source_mutation(self):
        import neurobench.workbench.server as server
        from neurobench.workbench.jobs import JobStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "Outputs" / "NeuronReview" / "demo" / "app"
            app.mkdir(parents=True)
            (app / "review_data.json").write_text(
                json.dumps({"video": {"frames": 2, "width": 4, "height": 3, "framePattern": "frames/frame_%06d.png"}, "parameters": {}, "rois": [{"id": "r1"}]}),
                encoding="utf-8",
            )
            source = root / "Inputs" / "demo" / "labels.tsv"
            source.parent.mkdir(parents=True)
            source.write_text("roi_id\nr1\n", encoding="utf-8")
            previous_root = server.PROJECT_ROOT
            server.PROJECT_ROOT = root
            try:
                record = server.register_import_source(dataset_id="demo", source=source, source_mode="local_registration", destination_path="Inputs/demo/labels.tsv", app_dir=app)
                source.write_text("roi_id\nr2\n", encoding="utf-8")
                store = JobStore(app / ".neurobench" / "jobs")
                job = store.create("label_import")
                server.execute_label_import_job(store, job["job_id"], record)
                result = store.get(job["job_id"])
                final_record = server.import_record_for("demo", record["import_id"], app_dir=app)
                artifact_exists = (app / "external_labels" / f"{record['import_id']}.json").exists()
            finally:
                server.PROJECT_ROOT = previous_root

        self.assertEqual(result["status"], "failed")
        self.assertIn("checksum changed", result["error"])
        self.assertEqual(final_record["state"], "failed")
        self.assertFalse(artifact_exists)

    def test_server_import_readers_fail_closed_on_misbound_sidecar(self):
        import neurobench.workbench.server as server

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "Outputs" / "NeuronReview" / "demo" / "app"
            source = root / "Inputs" / "demo" / "labels.tsv"
            source.parent.mkdir(parents=True)
            source.write_text("roi_id\ncell-1\n", encoding="utf-8")
            previous_root = server.PROJECT_ROOT
            server.PROJECT_ROOT = root
            try:
                record = server.register_import_source(
                    dataset_id="demo",
                    source=source,
                    source_mode="local_registration",
                    destination_path="Inputs/demo/labels.tsv",
                    app_dir=app,
                )
                sidecar = app / "imports" / f"{record['import_id']}.json"
                corrupted = json.loads(sidecar.read_text(encoding="utf-8"))
                corrupted["app_dir"] = "Outputs/NeuronReview/other/app"
                sidecar.write_text(json.dumps(corrupted), encoding="utf-8")

                selected = server.import_record_for("demo", record["import_id"], app_dir=app)
                listed = server.import_records(dataset_id="demo", app_dir=app)
            finally:
                server.PROJECT_ROOT = previous_root

        self.assertIsNone(selected)
        self.assertEqual(listed, [])

    def test_global_import_reader_uses_declared_dataset_identity(self):
        import neurobench.workbench.server as server

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "Outputs" / "NeuronReview" / "folder_name" / "app"
            source = root / "Inputs" / "declared" / "labels.tsv"
            source.parent.mkdir(parents=True)
            source.write_text("roi_id\ncell-1\n", encoding="utf-8")
            review_path = app / "review_data.json"
            review_path.parent.mkdir(parents=True)
            review_path.write_text(
                json.dumps(
                    {
                        "dataset": {"dataset_id": "declared"},
                        "video": {"name": "movie", "frames": 1, "height": 2, "width": 2},
                        "parameters": {},
                        "rois": [],
                    }
                ),
                encoding="utf-8",
            )
            previous_root = server.PROJECT_ROOT
            server.PROJECT_ROOT = root
            try:
                record = server.register_import_source(
                    dataset_id="declared",
                    source=source,
                    source_mode="local_registration",
                    destination_path="Inputs/declared/labels.tsv",
                    app_dir=app,
                )
                listed = server.import_records()
            finally:
                server.PROJECT_ROOT = previous_root

        self.assertEqual([item["import_id"] for item in listed], [record["import_id"]])

    def test_job_restart_marks_active_records_stopped(self):
        from neurobench.workbench.jobs import JobStore

        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "jobs")
            job = store.create("demo")
            store.update(job["job_id"], status="running", stage="work", owner_pid=99999999)
            recovered = JobStore(Path(tmp) / "jobs").recover_incomplete()
            current = store.get(job["job_id"])

        self.assertEqual(len(recovered), 1)
        self.assertEqual(current["status"], "stopped")
        self.assertEqual(current["stage"], "recovered_after_restart")


if __name__ == "__main__":
    unittest.main()
