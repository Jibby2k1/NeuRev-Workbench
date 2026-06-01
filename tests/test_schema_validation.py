from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[1]


class SchemaValidationTests(unittest.TestCase):
    def test_schema_path_supports_public_aliases(self):
        from neurobench.validation.schemas import schema_path

        self.assertEqual(schema_path("dataset").name, "dataset_manifest.schema.json")
        self.assertEqual(schema_path("dataset_manifest").name, "dataset_manifest.schema.json")
        self.assertEqual(schema_path("architecture_run").name, "architecture_run.schema.json")
        self.assertEqual(schema_path("pipeline_spec").name, "pipeline_spec.schema.json")
        self.assertEqual(schema_path("pipeline_run").name, "pipeline_run.schema.json")
        self.assertEqual(schema_path("llm_architecture_proposal").name, "llm_architecture_proposal.schema.json")
        self.assertEqual(schema_path("artifact_record").name, "artifact_record.schema.json")
        self.assertEqual(schema_path("review_data").name, "review_data.schema.json")
        self.assertEqual(schema_path("export_bundle").name, "export_bundle.schema.json")
        self.assertEqual(schema_path("export").name, "export_bundle.schema.json")

    def test_load_schema_checks_schema_document(self):
        from neurobench.validation.schemas import load_schema

        schema = load_schema("dataset")

        self.assertEqual(schema["title"], "Neurobench Dataset Manifest")

    def test_validate_existing_examples(self):
        from neurobench.validation.schemas import validate_json

        examples = [
            ("examples/dataset_manifest.example.json", "dataset"),
            ("examples/architecture_runs.example.json", "architecture_run"),
            ("examples/pipeline_spec.example.json", "pipeline_spec"),
        ]

        for path, schema_name in examples:
            with self.subTest(path=path, schema=schema_name):
                payload = validate_json(ROOT / path, schema_name)
                self.assertEqual(payload["schema_version"], 1)

    def test_validate_json_missing_file(self):
        from neurobench.validation.schemas import validate_json

        with self.assertRaises(FileNotFoundError):
            validate_json(ROOT / "examples" / "missing.json", "dataset")

    def test_validate_json_bad_schema_name(self):
        from neurobench.validation.schemas import load_schema

        with self.assertRaisesRegex(FileNotFoundError, "Unknown schema"):
            load_schema("not_a_schema")

    def test_validation_error_summary_contains_field_path(self):
        from neurobench.validation.schemas import validate_dict, validation_error_summary

        bad_manifest = {
            "schema_version": 1,
            "dataset_id": "bad",
            "paths": {
                "app_dir": "Outputs/app",
                "review_data": "Outputs/app/review_data.json",
            },
        }

        with self.assertRaises(jsonschema.ValidationError) as ctx:
            validate_dict(bad_manifest, "dataset")

        summary = validation_error_summary(ctx.exception)
        self.assertIn("field: paths", summary)
        self.assertIn("raw_video", summary)

    def test_validate_json_returns_payload(self):
        from neurobench.validation.schemas import validate_json

        payload = {
            "schema_version": 1,
            "dataset_id": "tmp",
            "paths": {
                "raw_video": "raw.tif",
                "app_dir": "app",
                "review_data": "app/review_data.json",
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertEqual(validate_json(path, "dataset"), payload)


if __name__ == "__main__":
    unittest.main()
