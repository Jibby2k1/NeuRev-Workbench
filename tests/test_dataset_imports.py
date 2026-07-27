from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


class DatasetImportContractTests(unittest.TestCase):
    def test_video_import_preserves_unknown_scientific_metadata(self):
        from neurobench.data.imports import dataset_manifest_from_import, inspect_source, make_import_record

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "Inputs" / "Fish 01"
            inputs.mkdir(parents=True)
            source = inputs / "movie.npy"
            np.save(source, np.zeros((4, 6, 8), dtype=np.uint16))
            inspected = inspect_source(source, workspace_root=root)
            record = make_import_record(
                dataset_id="Fish 01",
                import_id_value="imp_test",
                source_mode="local_registration",
                original_name=source.name,
                source_path=inspected["metadata"]["source_path"],
                destination_path=inspected["metadata"]["source_path"],
                metadata=inspected["metadata"],
                warnings=inspected["warnings"],
            )
            manifest = dataset_manifest_from_import(record, app_dir=root / "Outputs" / "NeuronReview" / "fish-01" / "app")

        self.assertEqual(record["state"], "metadata_needed")
        self.assertEqual(record["metadata"]["frames"], 4)
        self.assertIsNone(record["metadata"]["frame_rate_hz"])
        self.assertIsNone(record["metadata"]["pixel_size_microns"])
        self.assertNotIn("modality", manifest)
        self.assertNotIn("indicator", manifest)
        self.assertNotIn("frame_rate_hz", manifest)

    def test_local_registration_rejects_paths_outside_inputs_and_outputs(self):
        from neurobench.data.imports import resolve_allowed_local_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowed = root / "Inputs" / "movie.npy"
            allowed.parent.mkdir(parents=True)
            allowed.write_bytes(b"ok")
            outside = root / "outside.npy"
            outside.write_bytes(b"no")
            self.assertEqual(resolve_allowed_local_path(allowed, workspace_root=root), allowed.resolve())
            with self.assertRaises(ValueError):
                resolve_allowed_local_path(outside, workspace_root=root)

    def test_label_mapping_is_non_destructive_and_schema_validates(self):
        import json
        from neurobench.data.imports import inspect_source, make_import_record, update_import_record
        from neurobench.validation.schemas import validate_dict

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "labels.tsv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(["ROI ID", "centroid_x", "centroid_y", "label"])
                writer.writerow(["r1", 3, 4, "neuron"])
            inspected = inspect_source(source)
            record = make_import_record(
                dataset_id="labels",
                import_id_value="imp_labels",
                source_mode="upload",
                original_name=source.name,
                source_path=source.name,
                destination_path=source.name,
                metadata=inspected["metadata"],
                warnings=inspected["warnings"],
            )
            validate_dict(record, "dataset_import")
            changed = update_import_record(record, state="ready", warnings=["reviewed"])
            with self.assertRaises(ValueError):
                update_import_record(record, dataset_id="other")

        self.assertEqual(inspected["metadata"]["row_count"], 1)
        self.assertEqual(inspected["metadata"]["label_mapping"]["roi_id"], "ROI ID")
        self.assertEqual(changed["state"], "ready")
        self.assertEqual(changed["checksum"], record["checksum"])

    def test_schema_alias_is_public(self):
        from neurobench.validation.schemas import schema_path

        self.assertEqual(schema_path("dataset_import").name, "dataset_import.schema.json")
        self.assertEqual(schema_path("import").name, "dataset_import.schema.json")

    def test_non_video_import_cannot_become_dataset_manifest(self):
        from neurobench.data.imports import dataset_manifest_from_import, inspect_source, make_import_record

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "labels.tsv"
            source.write_text("roi_id\tlabel\nr1\tneuron\n", encoding="utf-8")
            inspected = inspect_source(source)
            record = make_import_record(
                dataset_id="labels",
                import_id_value="imp_source_only",
                source_mode="upload",
                original_name=source.name,
                source_path=source.name,
                destination_path=source.name,
                metadata=inspected["metadata"],
            )
            with self.assertRaisesRegex(ValueError, "primary video"):
                dataset_manifest_from_import(record, app_dir=root / "Outputs" / "NeuronReview" / "labels" / "app")

    def test_neurev_json_recognizes_only_native_schema_valid_documents(self):
        from neurobench.data.imports import inspect_source, source_kind

        documents = {
            "review_data": {
                "schema_version": 1,
                "dataset": {"dataset_id": "demo"},
                "video": {"width": 4, "height": 3, "frames": 2, "framePattern": "frames/frame_%06d.png"},
                "parameters": {},
                "rois": [{"id": "r1", "events": [{"frame": 0}]}],
                "discovery": {"suggestions": [{"id": "s1"}]},
            },
            "annotations": {
                "schema_version": 3,
                "rois": {"r1": {"cell_state": "accepted"}},
                "events": {"r1:1": {"event_state": "accepted"}},
                "suggestions": {},
                "settings": {},
            },
            "architecture_runs": {"schema_version": 1, "dataset_id": "demo", "runs": []},
            "export_bundle": {
                "schema_version": 1,
                "export_bundle_id": "bundle-1",
                "dataset_id": "demo",
                "run_ids": [],
                "created_at": "2026-07-25T00:00:00Z",
                "selection_policy": {"name": "all reviewed"},
                "alignment_status": "not_provided",
                "files": [],
                "provenance": {},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            observed = {}
            for payload_kind, payload in documents.items():
                source = root / f"{payload_kind}.json"
                source.write_text(json.dumps(payload), encoding="utf-8")
                inspected = inspect_source(source)
                observed[payload_kind] = inspected["metadata"]
                self.assertEqual(source_kind(source), "neurev_json")
                self.assertEqual(inspected["metadata"]["payload_kind"], payload_kind)
                self.assertEqual(inspected["metadata"]["kind"], "neurev_json")
                self.assertNotIn("payload", inspected["metadata"])

        self.assertEqual(observed["review_data"]["counts"], {"roi_count": 1, "event_count": 1, "suggestion_count": 1})
        self.assertEqual(observed["annotations"]["counts"]["roi_annotation_count"], 1)
        self.assertEqual(observed["architecture_runs"]["counts"], {"run_count": 0})
        self.assertEqual(observed["export_bundle"]["counts"], {"run_id_count": 0, "file_count": 0})

    def test_neurev_json_rejects_arbitrary_mixed_invalid_and_duplicate_key_json(self):
        from neurobench.data.imports import inspect_source

        review = {
            "schema_version": 1,
            "video": {"width": 4, "height": 3, "frames": 2, "framePattern": "frames/frame_%06d.png"},
            "parameters": {},
            "rois": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arbitrary = root / "arbitrary.json"
            arbitrary.write_text('{"hello":"world"}', encoding="utf-8")
            mixed = root / "mixed.json"
            mixed.write_text(json.dumps({**review, "events": {}, "suggestions": {}, "settings": {}}), encoding="utf-8")
            invalid_utf8 = root / "invalid_utf8.json"
            invalid_utf8.write_bytes(b'{"video":"\xff"}')
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"schema_version":1,"dataset_id":"demo","dataset_id":"other","runs":[]}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not a recognized NeuRev"):
                inspect_source(arbitrary)
            with self.assertRaisesRegex(ValueError, "mixes multiple native document shapes"):
                inspect_source(mixed)
            with self.assertRaisesRegex(ValueError, "valid UTF-8"):
                inspect_source(invalid_utf8)
            with self.assertRaisesRegex(ValueError, "duplicate object key"):
                inspect_source(duplicate)

    def test_neurev_json_hard_cap_is_checked_before_payload_read(self):
        from neurobench.data.imports import MAX_NEUREV_JSON_BYTES, inspect_source

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "oversize.json"
            with source.open("wb") as handle:
                handle.truncate(MAX_NEUREV_JSON_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "safety limit"):
                inspect_source(source)

    def test_import_sidecar_reader_enforces_schema_and_storage_identity(self):
        from neurobench.data.imports import inspect_source, make_import_record, read_import_record

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "Outputs" / "NeuronReview" / "demo" / "app"
            imports = app / "imports"
            imports.mkdir(parents=True)
            source = root / "Inputs" / "demo" / "labels.tsv"
            source.parent.mkdir(parents=True)
            source.write_text("roi_id\tlabel\nr1\tneuron\n", encoding="utf-8")
            inspected = inspect_source(source, workspace_root=root)
            base = make_import_record(
                dataset_id="demo",
                import_id_value="imp_valid",
                source_mode="local_registration",
                original_name=source.name,
                source_path="Inputs/demo/labels.tsv",
                destination_path="Inputs/demo/labels.tsv",
                metadata=inspected["metadata"],
                warnings=inspected["warnings"],
            )
            base["app_dir"] = "Outputs/NeuronReview/demo/app"
            valid_path = imports / "imp_valid.json"
            valid_path.write_text(json.dumps(base), encoding="utf-8")

            self.assertEqual(
                read_import_record(
                    valid_path,
                    expected_dataset_id="demo",
                    expected_app_dir=app,
                    workspace_root=root,
                )["import_id"],
                "imp_valid",
            )

            cases = []
            schema = dict(base, import_id="imp_schema", state="not-a-state")
            cases.append((imports / "imp_schema.json", schema, "schema validation"))
            filename = dict(base, import_id="imp_declared")
            cases.append((imports / "imp_filename.json", filename, "filename/import_id mismatch"))
            dataset = dict(base, import_id="imp_dataset", dataset_id="other")
            cases.append((imports / "imp_dataset.json", dataset, "dataset_id mismatch"))
            wrong_app = dict(base, import_id="imp_app", app_dir="Outputs/NeuronReview/other/app")
            cases.append((imports / "imp_app.json", wrong_app, "app_dir mismatch"))
            for path, payload, message in cases:
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    read_import_record(
                        path,
                        expected_dataset_id="demo",
                        expected_app_dir=app,
                        workspace_root=root,
                    )

            duplicate_path = imports / "imp_duplicate.json"
            duplicate = dict(base, import_id="imp_duplicate")
            duplicate_text = json.dumps(duplicate)[:-1] + ', "import_id": "imp_duplicate"}'
            duplicate_path.write_text(duplicate_text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate object key"):
                read_import_record(
                    duplicate_path,
                    expected_dataset_id="demo",
                    expected_app_dir=app,
                    workspace_root=root,
                )

    def test_import_sidecar_reader_checks_size_before_reading_payload(self):
        from neurobench.data.imports import MAX_IMPORT_RECORD_BYTES, read_import_record

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "Outputs" / "NeuronReview" / "demo" / "app"
            path = app / "imports" / "imp_oversize.json"
            path.parent.mkdir(parents=True)
            with path.open("wb") as handle:
                handle.truncate(MAX_IMPORT_RECORD_BYTES + 1)

            with self.assertRaisesRegex(ValueError, "safety limit"):
                read_import_record(
                    path,
                    expected_dataset_id="demo",
                    expected_app_dir=app,
                    workspace_root=root,
                )


if __name__ == "__main__":
    unittest.main()
