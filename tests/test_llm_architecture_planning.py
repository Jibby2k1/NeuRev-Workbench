from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


def require_numpy():
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError as exc:
        raise unittest.SkipTest("numpy is not installed in this Python environment") from exc


def _proposal_set() -> dict:
    return {
        "schema_version": 1,
        "proposal_set_id": "cfar_lab_notes_v1",
        "dataset_id": "d",
        "objective": "review_efficiency",
        "max_combinations_per_architecture": 4096,
        "proposals": [
            {
                "id": "multi_stage_cfar",
                "label": "Small then large reference CFAR",
                "rationale": "Use a permissive local reference first, then a broader reference to reject clustered background.",
                "hypothesis": "Compact events survive both masks while local background clutter is reduced.",
                "priority": 1,
                "expected_tradeoffs": "May lose very diffuse events.",
                "pipeline": [
                    {"id": "source", "stage_id": "source_video_import", "params": {"source": "raw.npy"}},
                    {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 6.0}},
                    {"id": "smooth", "stage_id": "spatial_gaussian", "params": {"sigma_px": 0.6}},
                    {"id": "cfar_small_ref", "stage_id": "gamma_cfar", "params": {"pfa": 0.01, "guard_px": 1, "training_radius_px": 5}},
                    {
                        "id": "cfar_large_ref",
                        "stage_id": "gamma_cfar",
                        "params": {"pfa": 0.001, "guard_px": 2, "training_radius_px": 17},
                        "metadata": {
                            "previous_mask_step": "cfar_small_ref",
                            "combine_mode": "intersection",
                            "cfar_role": "large_reference",
                        },
                    },
                    {"id": "components", "stage_id": "component_filter", "params": {"seed_z": 1.0, "min_area_px": 2, "max_area_px": 80}},
                ],
                "sweep": {
                    "parameters": [
                        {"stage": "cfar_small_ref", "param": "pfa", "values": [0.01, 0.03]},
                        {"stage": "cfar_large_ref", "param": "training_radius_px", "values": [13, 17]},
                    ]
                },
            }
        ],
    }


class LlmArchitecturePlanningTests(unittest.TestCase):
    def test_context_builder_includes_catalog_and_constraints(self):
        from neurobench.llm_planning import build_llm_context, render_llm_prompt

        context = build_llm_context(
            dataset_manifest={"schema_version": 1, "dataset_id": "d"},
            architecture_runs={"schema_version": 1, "dataset_id": "d", "runs": []},
            lab_notes="Try multi-stage CFAR.",
        )

        self.assertEqual(context["dataset_id"], "d")
        self.assertIn("gamma_cfar", context["stage_catalog"])
        self.assertIn("Represent multi-stage CFAR", " ".join(context["constraints"]))
        self.assertIn("Return JSON", render_llm_prompt(context))

    def test_valid_proposal_imports_saved_pipeline_and_planned_runs(self):
        from neurobench.llm_planning import proposal_set_to_architecture_manifest

        manifest = proposal_set_to_architecture_manifest(_proposal_set())

        self.assertEqual(len(manifest["saved_pipelines"]), 1)
        self.assertEqual(manifest["saved_pipelines"][0]["id"], "llm_cfar_lab_notes_v1_multi_stage_cfar")
        self.assertEqual(len(manifest["runs"]), 4)
        self.assertEqual(manifest["runs"][0]["pipeline"][3]["id"], "cfar_small_ref")
        self.assertEqual(manifest["runs"][0]["pipeline"][4]["metadata"]["combine_mode"], "intersection")
        self.assertEqual(manifest["experiments"][0]["source"], "llm_architecture_proposal")
        self.assertEqual(manifest["llm_proposal_sets"][0]["validation_report"]["status"], "valid")

    def test_reimport_replaces_same_proposal_runs(self):
        from neurobench.llm_planning import proposal_set_to_architecture_manifest

        first = proposal_set_to_architecture_manifest(_proposal_set())
        second = proposal_set_to_architecture_manifest(_proposal_set(), base_manifest=first)

        self.assertEqual(len(first["runs"]), 4)
        self.assertEqual(len(second["runs"]), 4)
        self.assertEqual(len(second["saved_pipelines"]), 1)

    def test_import_rejects_ambiguous_stage_id_sweep_reference(self):
        from neurobench.llm_planning import validate_proposal_set

        proposal = _proposal_set()
        proposal["proposals"][0]["sweep"]["parameters"][0]["stage"] = "gamma_cfar"

        with self.assertRaisesRegex(ValueError, "unknown pipeline step 'gamma_cfar'"):
            validate_proposal_set(proposal)

    def test_import_rejects_large_sweep_budget(self):
        from neurobench.llm_planning import validate_proposal_set

        proposal = _proposal_set()
        proposal["proposals"][0]["sweep"]["parameters"][0]["values"] = list(range(65))
        proposal["proposals"][0]["sweep"]["parameters"][1]["values"] = list(range(65))

        with self.assertRaisesRegex(ValueError, "above limit 4096"):
            validate_proposal_set(proposal)

    def test_import_cli_writes_manifest_and_report(self):
        from tools.import_llm_architecture_proposals import main
        import sys
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposal_path = root / "proposal.json"
            out = root / "architecture_runs.json"
            report = root / "report.json"
            proposal_path.write_text(json.dumps(_proposal_set()), encoding="utf-8")
            argv = [
                "import_llm_architecture_proposals.py",
                "--proposal",
                str(proposal_path),
                "--out",
                str(out),
                "--validation-report",
                str(report),
            ]
            with patch.object(sys, "argv", argv):
                main()

            self.assertTrue(out.exists())
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["proposal_count"], 1)

    def test_llm_experiment_runner_writes_evaluation_metrics(self):
        require_numpy()
        from neurobench.data.synthetic import generate_synthetic_calcium_dataset
        from tools.run_llm_architecture_experiments import main
        import sys
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = generate_synthetic_calcium_dataset(include_impulse_artifact=False).write(root / "fixture", dataset_id="synthetic_llm")
            proposal = _proposal_set()
            proposal["dataset_id"] = "synthetic_llm"
            proposal["proposal_set_id"] = "synthetic_eval"
            proposal["proposals"][0]["pipeline"] = [
                {"id": "source", "stage_id": "source_video_import", "params": {"source": paths["video"]}},
                {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 2.0}},
                {"id": "score", "stage_id": "robust_positive_local_z", "params": {"epsilon": 0.05}},
                {
                    "id": "components",
                    "stage_id": "component_filter",
                    "params": {"seed_z": 1.4, "min_area_px": 3, "max_area_px": 120},
                },
                {"id": "traces", "stage_id": "local_background_ring", "params": {"outer_radius_px": 8, "neuropil_weight": 0.2}},
                {"id": "events", "stage_id": "robust_kalman_positive_innovation", "params": {"event_threshold_z": 1.0}},
            ]
            proposal["proposals"][0].pop("sweep", None)
            proposal_path = root / "proposal.json"
            proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
            run_root = root / "llm_runs"
            argv = [
                "run_llm_architecture_experiments.py",
                "--proposal",
                str(proposal_path),
                "--run-root",
                str(run_root),
                "--ground-truth-csv",
                paths["ground_truth"],
            ]

            with patch.object(sys, "argv", argv):
                main()

            summary = json.loads((run_root / "llm_experiment_summary.json").read_text(encoding="utf-8"))
            report = (run_root / "llm_experiment_report.md").read_text(encoding="utf-8")

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["total"], 1)
        self.assertIn("metric_overview", summary)
        self.assertIn("runtime_sec", summary["runs"][0])
        self.assertIn("candidate_count", summary["runs"][0]["metrics"])
        self.assertIn("object_recall", summary["runs"][0]["metrics"])
        self.assertIn("event_onset_recall", summary["runs"][0]["metrics"])
        self.assertIn("## Metric Overview", report)
        self.assertIn("obj recall", report)


if __name__ == "__main__":
    unittest.main()
