from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


ROOT = Path(__file__).resolve().parents[1]


def require_numpy():
    if np is None:
        raise unittest.SkipTest("numpy is not installed in this Python environment")


def _llm_proposal(source: str = "raw.npy") -> dict:
    return {
        "schema_version": 1,
        "proposal_set_id": "cli_llm_v1",
        "dataset_id": "cli_llm",
        "objective": "review_efficiency",
        "max_combinations_per_architecture": 8,
        "proposals": [
            {
                "id": "local_z",
                "label": "Local z baseline",
                "rationale": "Small CLI smoke proposal.",
                "hypothesis": "The local z baseline should produce candidates on synthetic data.",
                "priority": 1,
                "expected_tradeoffs": "Smoke test only.",
                "pipeline": [
                    {"id": "source", "stage_id": "source_video_import", "params": {"source": source}},
                    {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 2.0}},
                    {"id": "score", "stage_id": "robust_positive_local_z", "params": {"epsilon": 0.05}},
                    {"id": "components", "stage_id": "component_filter", "params": {"seed_z": 1.4, "min_area_px": 3, "max_area_px": 120}},
                ],
            }
        ],
    }


class CliMainTests(unittest.TestCase):
    def test_pyproject_declares_neurobench_console_script(self):
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(pyproject["project"]["scripts"]["neurobench"], "neurobench.cli.main:main")
        self.assertIn("neurobench*", pyproject["tool"]["setuptools"]["packages"]["find"]["include"])

    def test_cli_help_entrypoint_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "neurobench.cli.main", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("usage: neurobench", result.stdout)
        self.assertIn("dataset", result.stdout)
        self.assertIn("workbench", result.stdout)
        self.assertIn("run", result.stdout)
        self.assertIn("llm", result.stdout)

    def test_cli_workbench_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "neurobench.cli.main", "workbench", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("attach-intermediates", result.stdout)
        self.assertIn("export-intermediate", result.stdout)

    def test_cli_no_args_prints_help_and_returns_zero(self):
        from neurobench.cli.main import main

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main([])

        self.assertEqual(code, 0)
        self.assertIn("usage: neurobench", buffer.getvalue())

    def test_cli_placeholder_group_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "neurobench.cli.main", "dataset", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Create, validate, and inspect dataset manifests.", result.stdout)
        self.assertIn("validate", result.stdout)

    def test_cli_dataset_validate_example(self):
        result = subprocess.run(
            [sys.executable, "-m", "neurobench.cli.main", "dataset", "validate", "examples/dataset_manifest.example.json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Validated dataset manifest", result.stdout)
        self.assertIn("calcium_video_2", result.stdout)

    def test_cli_validate_dataset_alias_example(self):
        result = subprocess.run(
            [sys.executable, "-m", "neurobench.cli.main", "validate", "dataset", "examples/dataset_manifest.example.json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Validated dataset manifest", result.stdout)

    def test_cli_run_validate_example(self):
        result = subprocess.run(
            [sys.executable, "-m", "neurobench.cli.main", "run", "validate", "examples/pipeline_spec.example.json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Validated pipeline spec", result.stdout)
        self.assertIn("planned_candidate_review_v1", result.stdout)

    def test_cli_run_dry_run_json_example(self):
        result = subprocess.run(
            [sys.executable, "-m", "neurobench.cli.main", "run", "dry-run", "--json", "examples/pipeline_spec.example.json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        plan = json.loads(result.stdout)

        self.assertEqual(result.returncode, 0)
        self.assertEqual(plan["status"], "dry_run_ok")
        self.assertEqual(plan["run_id"], "planned_candidate_review_v1")
        self.assertGreaterEqual(len(plan["steps"]), 1)

    def test_cli_run_execute_synthetic_pipeline(self):
        require_numpy()
        from neurobench.data.synthetic import generate_synthetic_calcium_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_paths = generate_synthetic_calcium_dataset(include_impulse_artifact=False).write(root / "fixture")
            spec_path = root / "pipeline_spec.json"
            run_root = root / "run"
            spec_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset_id": "cli_synthetic",
                        "run_id": "cli_synthetic_e2e",
                        "pipeline": [
                            {"id": "source", "stage_id": "source_video_import", "params": {"source": fixture_paths["video"]}},
                            {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 2.0}},
                            {"id": "score", "stage_id": "robust_positive_local_z", "params": {"epsilon": 0.05}},
                            {
                                "id": "components",
                                "stage_id": "component_filter",
                                "params": {"seed_z": 1.5, "min_area_px": 3, "max_area_px": 100},
                            },
                        ],
                        "artifacts": {},
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, "-m", "neurobench.cli.main", "run", "execute", str(spec_path), "--run-root", str(run_root)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            manifest = json.loads((run_root / "pipeline_run.json").read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Pipeline execution completed", result.stdout)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["run_id"], "cli_synthetic_e2e")
        self.assertEqual(len(manifest["artifacts"]), 4)
        self.assertEqual(manifest["extras"]["input_checksums"][0]["path_id"], "raw_video")
        self.assertEqual(manifest["extras"]["input_checksums"][0]["sha256"], manifest["artifacts"][0]["sha256"])

    def test_cli_workbench_attach_intermediates(self):
        require_numpy()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "run"
            artifact_dir = run_root / "artifacts" / "preprocessing"
            artifact_dir.mkdir(parents=True)
            np.save(artifact_dir / "highpass_video.npy", np.ones((2, 4, 5), dtype=np.float32))
            pipeline_run = run_root / "pipeline_run.json"
            pipeline_run.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "cli_workbench_run",
                        "artifacts": [
                            {
                                "artifact_id": "highpass_video.v1",
                                "kind": "highpass_video",
                                "path": "artifacts/preprocessing/highpass_video.npy",
                                "producer_stage": "temporal_highpass_gaussian",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            app_dir = root / "app"
            app_dir.mkdir()
            architecture_runs = app_dir / "architecture_runs.json"
            architecture_runs.write_text(
                json.dumps({"schema_version": 1, "runs": [{"run_id": "cli_workbench_run", "artifacts": {}}]}),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurobench.cli.main",
                    "workbench",
                    "attach-intermediates",
                    "--pipeline-run",
                    str(pipeline_run),
                    "--architecture-runs",
                    str(architecture_runs),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(result.stdout)
            manifest = json.loads(architecture_runs.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["exported_count"], 1)
        self.assertEqual(manifest["runs"][0]["artifacts"]["intermediates"][0]["stage_id"], "temporal_highpass_gaussian")

    def test_cli_run_batch_records_success_and_failure(self):
        require_numpy()
        from neurobench.data.synthetic import generate_synthetic_calcium_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_paths = generate_synthetic_calcium_dataset(include_impulse_artifact=False).write(root / "fixture")
            good_spec = root / "good.json"
            bad_spec = root / "bad.json"
            base_spec = {
                "schema_version": 1,
                "dataset_id": "cli_batch",
                "pipeline": [
                    {"id": "source", "stage_id": "source_video_import", "params": {"source": fixture_paths["video"]}},
                    {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 2.0}},
                    {"id": "score", "stage_id": "robust_positive_local_z", "params": {"epsilon": 0.05}},
                    {
                        "id": "components",
                        "stage_id": "component_filter",
                        "params": {"seed_z": 1.5, "min_area_px": 3, "max_area_px": 100},
                    },
                ],
                "artifacts": {},
            }
            good = dict(base_spec)
            good["run_id"] = "cli_batch_good"
            bad = json.loads(json.dumps(base_spec))
            bad["run_id"] = "cli_batch_bad"
            bad["pipeline"][0]["params"]["source"] = str(root / "missing.npy")
            good_spec.write_text(json.dumps(good), encoding="utf-8")
            bad_spec.write_text(json.dumps(bad), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurobench.cli.main",
                    "run",
                    "batch",
                    str(good_spec),
                    str(bad_spec),
                    "--run-root",
                    str(root / "batch"),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            summary = json.loads((root / "batch" / "batch_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 1)
        self.assertIn("completed_with_failures", result.stdout)
        self.assertEqual(summary["succeeded"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["runs"][0]["status"], "completed")
        self.assertEqual(summary["runs"][1]["status"], "failed")

    def test_cli_run_sweep_executes_parameter_grid(self):
        require_numpy()
        from neurobench.data.synthetic import generate_synthetic_calcium_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_paths = generate_synthetic_calcium_dataset(include_impulse_artifact=False).write(root / "fixture")
            spec_path = root / "sweep.json"
            run_root = root / "sweep_out"
            spec_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset_id": "cli_sweep",
                        "run_id": "cli_sweep",
                        "pipeline": [
                            {"id": "source", "stage_id": "source_video_import", "params": {"source": fixture_paths["video"]}},
                            {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 2.0}},
                            {"id": "score", "stage_id": "robust_positive_local_z", "params": {"epsilon": 0.05}},
                            {
                                "id": "components",
                                "stage_id": "component_filter",
                                "params": {"seed_z": 1.5, "min_area_px": 3, "max_area_px": 100},
                            },
                        ],
                        "sweep": {
                            "id": "cli_component_sweep",
                            "parameters": [{"stage": "components", "param": "seed_z", "values": [1.4, 1.8]}],
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurobench.cli.main",
                    "run",
                    "sweep",
                    str(spec_path),
                    "--run-root",
                    str(run_root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            summary = json.loads((run_root / "sweep_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Sweep execution summary", result.stdout)
        self.assertEqual(summary["succeeded"], 2)
        self.assertEqual(summary["failed"], 0)

    def test_cli_llm_context_writes_prompt_and_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context_path = root / "context.json"
            prompt_path = root / "prompt.md"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurobench.cli.main",
                    "llm",
                    "context",
                    "--dataset-manifest",
                    "examples/dataset_manifest.example.json",
                    "--objective",
                    "artifact_control",
                    "--context-out",
                    str(context_path),
                    "--prompt-out",
                    str(prompt_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            context = json.loads(context_path.read_text(encoding="utf-8"))
            prompt = prompt_path.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Neurobench Architecture Proposal Request", result.stdout)
        self.assertEqual(context["objective"], "artifact_control")
        self.assertIn("Return JSON", prompt)

    def test_cli_llm_import_proposals_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposal_path = root / "proposal.json"
            out = root / "architecture_runs.json"
            report = root / "validation_report.json"
            proposal_path.write_text(json.dumps(_llm_proposal()), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurobench.cli.main",
                    "llm",
                    "import-proposals",
                    str(proposal_path),
                    "--out",
                    str(out),
                    "--validation-report",
                    str(report),
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            summary = json.loads(result.stdout)
            manifest = json.loads(out.read_text(encoding="utf-8"))
            report_exists = report.exists()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(summary["proposal_count"], 1)
        self.assertEqual(summary["run_count"], 1)
        self.assertEqual(manifest["saved_pipelines"][0]["source"], "llm_architecture_proposal")
        self.assertTrue(report_exists)

    def test_cli_llm_run_proposals_executes_synthetic(self):
        require_numpy()
        from neurobench.data.synthetic import generate_synthetic_calcium_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_paths = generate_synthetic_calcium_dataset(include_impulse_artifact=False).write(root / "fixture")
            proposal_path = root / "proposal.json"
            run_root = root / "llm_runs"
            proposal_path.write_text(json.dumps(_llm_proposal(fixture_paths["video"])), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurobench.cli.main",
                    "llm",
                    "run-proposals",
                    str(proposal_path),
                    "--run-root",
                    str(run_root),
                    "--ground-truth-csv",
                    fixture_paths["ground_truth"],
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            summary = json.loads(result.stdout)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["succeeded"], 1)
        self.assertIn("object_recall", summary["runs"][0]["metrics"])

    def test_cli_dataset_validate_malformed_manifest_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_manifest.json"
            path.write_text(json.dumps({"schema_version": 1, "dataset_id": "bad", "paths": {}}), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "neurobench.cli.main", "dataset", "validate", str(path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("Dataset manifest validation failed", result.stderr)
        self.assertIn("raw_video", result.stderr)

    def test_legacy_create_dataset_manifest_script_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dataset.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "tools/create_dataset_manifest.py",
                    "--out",
                    str(out),
                    "--dataset-id",
                    "tmp_dataset",
                    "--raw-video",
                    "Inputs/tmp.tif",
                    "--app-dir",
                    "Outputs/NeuronReview/tmp_dataset/app",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0)
        self.assertEqual(payload["dataset_id"], "tmp_dataset")
        self.assertEqual(payload["paths"]["raw_video"], "Inputs/tmp.tif")

    def test_cli_invalid_command_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "neurobench.cli.main", "not-a-command"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr)


if __name__ == "__main__":
    unittest.main()
