from __future__ import annotations

import json
from pathlib import Path
import unittest


def _sweep_spec() -> dict:
    return {
        "schema_version": 1,
        "dataset_id": "d",
        "run_id": "planned_candidate_review_v1",
        "pipeline": [
            {"id": "source", "stage_id": "source_video_import", "params": {"source": "raw.tif"}},
            {"id": "candidates", "stage_id": "candidate_event_pipeline"},
            {"id": "review_app", "stage_id": "generate_neuron_review_app"},
        ],
        "sweep": {
            "id": "candidate_grid",
            "parameters": [
                {"stage": "candidates", "param": "event_threshold_z", "values": [2.0, 2.4]},
                {"stage": "candidates", "param": "min_area_px", "values": [4, 8]},
            ],
        },
        "artifacts": {"source_video": "raw.tif"},
    }


class PipelineSweepTests(unittest.TestCase):
    def test_manifest_normalization_preserves_dashboard_root_metadata(self):
        from neurobench.architecture_runs import as_run_manifest

        manifest = as_run_manifest(
            {
                "schema_version": 1,
                "dataset_id": "d",
                "saved_pipelines": [{"id": "template_a", "label": "Template A", "pipeline": []}],
                "experiments": [{"id": "experiment_a", "run_ids": ["run_a"]}],
                "optimization_studies": [{"id": "study_a", "base_pipeline_id": "template_a"}],
                "llm_proposal_sets": [{"proposal_set_id": "proposal_a"}],
                "runs": [
                    {
                        "schema_version": 1,
                        "run_id": "run_a",
                        "dataset_id": "d",
                        "artifacts": {},
                    }
                ],
            }
        )

        self.assertEqual(manifest["saved_pipelines"][0]["id"], "template_a")
        self.assertEqual(manifest["experiments"][0]["id"], "experiment_a")
        self.assertEqual(manifest["optimization_studies"][0]["id"], "study_a")
        self.assertEqual(manifest["llm_proposal_sets"][0]["proposal_set_id"], "proposal_a")

    def test_planned_manifest_expands_sweep_deterministically(self):
        from neurobench.architecture_runs import build_planned_manifest

        first = build_planned_manifest(_sweep_spec())
        second = build_planned_manifest(_sweep_spec())

        self.assertEqual(first, second)
        self.assertEqual(first["sweep"]["total_runs"], 4)
        self.assertEqual(
            [run["run_id"] for run in first["runs"]],
            [
                "planned_candidate_review_v1__sweep_001",
                "planned_candidate_review_v1__sweep_002",
                "planned_candidate_review_v1__sweep_003",
                "planned_candidate_review_v1__sweep_004",
            ],
        )
        self.assertEqual(
            [
                (
                    run["pipeline"][1]["params"]["event_threshold_z"],
                    run["pipeline"][1]["params"]["min_area_px"],
                )
                for run in first["runs"]
            ],
            [(2.0, 4), (2.0, 8), (2.4, 4), (2.4, 8)],
        )
        self.assertEqual(first["runs"][2]["sweep"]["index"], 2)
        self.assertEqual(
            first["runs"][2]["sweep"]["parameters"],
            [
                {
                    "stage": "candidates",
                    "stage_id": "candidate_event_pipeline",
                    "param": "event_threshold_z",
                    "value": 2.4,
                },
                {
                    "stage": "candidates",
                    "stage_id": "candidate_event_pipeline",
                    "param": "min_area_px",
                    "value": 4,
                },
            ],
        )

    def test_sweep_rejects_invalid_stage(self):
        from neurobench.architecture_runs import build_planned_manifest

        spec = _sweep_spec()
        spec["sweep"]["parameters"][0]["stage"] = "missing"

        with self.assertRaisesRegex(ValueError, "Unknown sweep stage 'missing'"):
            build_planned_manifest(spec)

    def test_sweep_rejects_invalid_param(self):
        from neurobench.architecture_runs import build_planned_manifest

        spec = _sweep_spec()
        spec["sweep"]["parameters"][0]["param"] = "not_a_param"

        with self.assertRaisesRegex(ValueError, "not valid for pipeline stage 'candidate_event_pipeline'"):
            build_planned_manifest(spec)

    def test_sweep_rejects_invalid_value(self):
        from neurobench.architecture_runs import build_planned_manifest

        spec = _sweep_spec()
        spec["sweep"]["parameters"][0]["values"] = [-1.0]

        with self.assertRaisesRegex(ValueError, "below minimum"):
            build_planned_manifest(spec)

    def test_schema_accepts_sweep_manifest(self):
        import jsonschema

        from neurobench.architecture_runs import build_planned_manifest

        root = Path(__file__).resolve().parents[1]
        schema = json.loads((root / "schemas" / "architecture_run.schema.json").read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(build_planned_manifest(_sweep_spec()))


if __name__ == "__main__":
    unittest.main()
