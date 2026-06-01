from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def require_numpy():
    if np is None:
        raise unittest.SkipTest("numpy is not installed in this Python environment")


def _dry_run_spec():
    return {
        "schema_version": 1,
        "dataset_id": "synthetic",
        "run_id": "dry_run_001",
        "pipeline": [
            {"id": "source", "stage_id": "source_video_import", "params": {"source": "raw.tif"}},
            {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 4.0}},
            {"id": "score", "stage_id": "robust_positive_local_z"},
            {"id": "components", "stage_id": "component_filter"},
        ],
    }


class PipelineExecutorTests(unittest.TestCase):
    def test_stage_registry_lists_executable_stages(self):
        from neurobench.pipelines.stages import default_stage_registry

        registry = default_stage_registry()
        executable_ids = registry.executable_stage_ids()

        self.assertIn("source_video_import", executable_ids)
        self.assertIn("gamma_cfar", executable_ids)
        self.assertIn("event_preserving_noise_suppression", executable_ids)
        self.assertNotIn("flat_field_background", executable_ids)
        self.assertNotIn("review_data_import", executable_ids)
        self.assertTrue(registry.get("source_video_import").executable)
        self.assertTrue(registry.get("event_preserving_noise_suppression").runner_available)
        self.assertFalse(registry.get("review_data_import").runner_available)
        self.assertFalse(registry.get("flat_field_background").executable)

    def test_pipeline_executor_dry_run_returns_normalized_plan(self):
        from neurobench.pipelines.executor import dry_run_pipeline

        plan = dry_run_pipeline(_dry_run_spec(), validate_artifacts=True)

        self.assertEqual(plan["status"], "dry_run_ok")
        self.assertEqual(plan["dataset_id"], "synthetic")
        self.assertEqual(plan["run_id"], "dry_run_001")
        self.assertEqual(len(plan["parameter_hash"]), 64)
        self.assertEqual(
            [step["stage_id"] for step in plan["steps"]],
            [
                "source_video_import",
                "temporal_highpass_gaussian",
                "robust_positive_local_z",
                "component_filter",
            ],
        )
        self.assertEqual(plan["steps"][1]["params"]["sigma_frames"], 4.0)
        self.assertEqual(plan["steps"][2]["params"]["local_radius_px"], 11)
        self.assertIn("highpass_frame", plan["steps"][1]["metadata"]["expected_qc_outputs"])
        self.assertIn("positive_z_frame", plan["steps"][2]["metadata"]["expected_qc_outputs"])
        self.assertIn("roi_candidate_overlay", plan["steps"][3]["metadata"]["expected_qc_outputs"])
        self.assertIn("roi_candidates", plan["available_artifacts"])

    def test_pipeline_executor_missing_artifact_error(self):
        from neurobench.pipelines.executor import dry_run_pipeline

        with self.assertRaisesRegex(ValueError, "requires missing artifact 'z_stack'"):
            dry_run_pipeline(
                {
                    "pipeline": [
                        {"id": "components", "stage_id": "component_filter"},
                    ]
                },
                validate_artifacts=True,
            )

    def test_pipeline_executor_rejects_planned_stage_by_default(self):
        from neurobench.pipelines.executor import dry_run_pipeline

        with self.assertRaisesRegex(ValueError, "not executable"):
            dry_run_pipeline(
                {
                    "pipeline": [
                        {"id": "source", "stage_id": "source_video_import", "params": {"source": "raw.tif"}},
                        {"id": "flat", "stage_id": "flat_field_background"},
                    ]
                }
            )

    def test_pipeline_executor_can_dry_run_planned_stage_when_allowed(self):
        from neurobench.pipelines.executor import dry_run_pipeline

        plan = dry_run_pipeline(
            {
                "pipeline": [
                    {"id": "source", "stage_id": "source_video_import", "params": {"source": "raw.tif"}},
                    {"id": "flat", "stage_id": "flat_field_background"},
                ]
            },
            require_executable=False,
        )

        self.assertEqual(plan["steps"][1]["stage_id"], "flat_field_background")
        self.assertIn("corrected_frame", plan["steps"][1]["metadata"]["expected_qc_outputs"])
        self.assertFalse(plan["require_executable"])

    def test_synthetic_pipeline_e2e_writes_manifest_and_artifacts(self):
        require_numpy()
        from neurobench.data.synthetic import generate_synthetic_calcium_dataset
        from neurobench.pipelines.executor import execute_pipeline

        dataset = generate_synthetic_calcium_dataset(include_impulse_artifact=False)
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixture"
            paths = dataset.write(fixture_dir, dataset_id="synthetic_e2e")
            result = execute_pipeline(
                {
                    "schema_version": 1,
                    "dataset_id": "synthetic_e2e",
                    "run_id": "synthetic_pipeline_e2e",
                    "pipeline": [
                        {"id": "source", "stage_id": "source_video_import", "params": {"source": paths["video"]}},
                        {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 2.0}},
                        {"id": "score", "stage_id": "robust_positive_local_z", "params": {"epsilon": 0.05}},
                        {
                            "id": "components",
                            "stage_id": "component_filter",
                            "params": {"seed_z": 1.5, "min_area_px": 3, "max_area_px": 100},
                        },
                    ],
                },
                run_root=Path(tmp) / "run",
            )
            run_root = Path(result["run_root"])
            manifest = json.loads((run_root / "pipeline_run.json").read_text(encoding="utf-8"))
            candidates = json.loads((run_root / "artifacts" / "candidates" / "roi_candidates.json").read_text(encoding="utf-8"))
            events_log_exists = (run_root / "logs" / "events.jsonl").is_file()

            self.assertEqual(result["status"], "completed")
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["run_id"], "synthetic_pipeline_e2e")
            self.assertEqual([item["kind"] for item in manifest["artifacts"]], ["raw_video", "highpass_video", "z_stack", "roi_candidates"])
            self.assertEqual(manifest["extras"]["input_checksums"][0]["path_id"], "raw_video")
            self.assertEqual(manifest["extras"]["input_checksums"][0]["sha256"], manifest["artifacts"][0]["sha256"])
            self.assertEqual(manifest["extras"]["input_checksums"][0]["size_bytes"], Path(paths["video"]).stat().st_size)
            self.assertGreaterEqual(candidates["candidates"][0]["peak_z"], 1.5)
            self.assertTrue(events_log_exists)

    def test_synthetic_pipeline_executes_heuristic_priority_ranking(self):
        require_numpy()
        from neurobench.data.synthetic import generate_synthetic_calcium_dataset
        from neurobench.pipelines.executor import dry_run_pipeline, execute_pipeline

        dataset = generate_synthetic_calcium_dataset(include_impulse_artifact=False)
        with tempfile.TemporaryDirectory() as tmp:
            paths = dataset.write(Path(tmp) / "fixture", dataset_id="synthetic_rank")
            spec = {
                "schema_version": 1,
                "dataset_id": "synthetic_rank",
                "run_id": "synthetic_pipeline_rank",
                "pipeline": [
                    {"id": "source", "stage_id": "source_video_import", "params": {"source": paths["video"]}},
                    {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 2.0}},
                    {"id": "score", "stage_id": "robust_positive_local_z", "params": {"epsilon": 0.05}},
                    {
                        "id": "components",
                        "stage_id": "component_filter",
                        "params": {"seed_z": 1.5, "min_area_px": 3, "max_area_px": 100},
                    },
                    {"id": "rank", "stage_id": "heuristic_priority_v1"},
                ],
            }

            plan = dry_run_pipeline(spec, validate_artifacts=True)
            result = execute_pipeline(spec, run_root=Path(tmp) / "run")
            run_root = Path(result["run_root"])
            manifest = json.loads((run_root / "pipeline_run.json").read_text(encoding="utf-8"))
            ranked = json.loads((run_root / "artifacts" / "candidates" / "ranked_candidates.json").read_text(encoding="utf-8"))

            self.assertIn("ranked_candidates", plan["available_artifacts"])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(manifest["artifacts"][-1]["kind"], "ranked_candidates")
            self.assertGreater(ranked["candidate_count"], 0)
            self.assertEqual(ranked["ranked_candidates"][0]["rank"], 1)
            self.assertTrue(ranked["ranked_candidates"][0]["explanation"]["contributions"])

    def test_synthetic_pipeline_executes_spatial_gaussian_and_gamma_cfar(self):
        require_numpy()
        from neurobench.data.synthetic import generate_synthetic_calcium_dataset
        from neurobench.pipelines.executor import dry_run_pipeline, execute_pipeline

        dataset = generate_synthetic_calcium_dataset(include_impulse_artifact=False)
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixture"
            paths = dataset.write(fixture_dir, dataset_id="synthetic_cfar")
            spec = {
                "schema_version": 1,
                "dataset_id": "synthetic_cfar",
                "run_id": "synthetic_pipeline_cfar",
                "pipeline": [
                    {"id": "source", "stage_id": "source_video_import", "params": {"source": paths["video"]}},
                    {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 2.0}},
                    {"id": "smooth", "stage_id": "spatial_gaussian", "params": {"sigma_px": 0.4}},
                    {
                        "id": "cfar",
                        "stage_id": "gamma_cfar",
                        "params": {"pfa": 0.2, "guard_px": 1, "training_radius_px": 5},
                    },
                ],
            }

            plan = dry_run_pipeline(spec, validate_artifacts=True)
            result = execute_pipeline(spec, run_root=Path(tmp) / "run")
            run_root = Path(result["run_root"])
            manifest = json.loads((run_root / "pipeline_run.json").read_text(encoding="utf-8"))
            candidate_mask = np.load(run_root / "artifacts" / "candidates" / "candidate_mask.npy")

            self.assertEqual(plan["available_artifacts"][-1], "smoothed_video")
            self.assertIn("candidate_mask", plan["available_artifacts"])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(
                [item["kind"] for item in manifest["artifacts"]],
                ["raw_video", "highpass_video", "smoothed_video", "candidate_mask"],
            )
            self.assertEqual(candidate_mask.shape, dataset.video.shape)
            self.assertGreater(int(np.count_nonzero(candidate_mask)), 0)
            self.assertLess(manifest["artifacts"][-1]["summary"]["active_fraction"], 0.25)

    def test_component_filter_uses_z_seed_when_reducing_candidate_mask(self):
        require_numpy()
        from neurobench.models.pipeline import PipelineRun
        from neurobench.pipelines.artifacts import ArtifactStore
        from neurobench.pipelines.executor import _run_component_filter

        candidate_mask = np.zeros((6, 20, 24), dtype=np.uint8)
        candidate_mask[:, 1:5, 1:5] = 1
        candidate_mask[:, 12:16, 14:18] = 1
        z_stack = np.zeros((6, 20, 24), dtype=np.float32)
        z_stack[:, 1:5, 1:5] = 2.0
        z_stack[:, 12:16, 14:18] = 5.0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask_path = root / "candidate_mask.npy"
            z_path = root / "z_stack.npy"
            np.save(mask_path, candidate_mask)
            np.save(z_path, z_stack)
            store = ArtifactStore(
                root / "run",
                PipelineRun(
                    schema_version=1,
                    run_id="component_seed",
                    dataset_id="synthetic",
                    pipeline_spec_id="component_seed",
                    status="running",
                    created_at="2026-01-01T00:00:00+00:00",
                    parameter_hash="0" * 64,
                    artifacts=[],
                ),
            )
            _, out = _run_component_filter(
                {
                    "step_id": "components",
                    "stage_id": "component_filter",
                    "params": {"seed_z": 4.0, "min_area_px": 3, "max_area_px": 100, "support_min_frames": 1},
                },
                {"candidate_mask": mask_path, "z_stack": z_path},
                store,
            )
            candidates = json.loads(out.read_text())

            self.assertEqual(len(candidates["candidates"]), 1)
            self.assertGreater(candidates["candidates"][0]["y"], 10)

    def test_synthetic_pipeline_executes_rigid_shift_estimate(self):
        require_numpy()
        from neurobench.algorithms.motion import shift_frame_integer
        from neurobench.pipelines.executor import dry_run_pipeline, execute_pipeline

        y_grid, x_grid = np.mgrid[0:24, 0:28]
        reference = np.exp(-(((x_grid - 14.0) ** 2) + ((y_grid - 11.0) ** 2)) / (2 * 2.0**2)).astype(np.float32)
        video = np.stack([reference, shift_frame_integer(reference, 2, -1), shift_frame_integer(reference, -1, 2)])
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "shifted_video.npy"
            np.save(video_path, video)
            spec = {
                "schema_version": 1,
                "dataset_id": "synthetic_motion",
                "run_id": "synthetic_pipeline_motion",
                "pipeline": [
                    {"id": "source", "stage_id": "source_video_import", "params": {"source": str(video_path)}},
                    {"id": "motion", "stage_id": "rigid_shift_estimate", "params": {"max_shift_px": 4}},
                ],
            }

            plan = dry_run_pipeline(spec, validate_artifacts=True)
            result = execute_pipeline(spec, run_root=Path(tmp) / "run")
            run_root = Path(result["run_root"])
            manifest = json.loads((run_root / "pipeline_run.json").read_text(encoding="utf-8"))
            registered = np.load(run_root / "artifacts" / "motion" / "registered_video.npy")
            shift_trace = json.loads((run_root / "artifacts" / "motion" / "rigid_shift_trace.json").read_text(encoding="utf-8"))

            self.assertIn("registered_video", plan["available_artifacts"])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(
                [item["kind"] for item in manifest["artifacts"]],
                ["raw_video", "rigid_shift_trace", "registered_video"],
            )
            self.assertEqual([(item["dy"], item["dx"]) for item in shift_trace["shifts"]], [(0, 0), (-2, 1), (1, -2)])
            self.assertLess(float(np.mean(np.abs(registered[0] - registered[1]))), 0.03)
            self.assertEqual(manifest["artifacts"][-1]["summary"]["max_abs_l1_shift_px"], 3)

    def test_execute_pipeline_rejects_implemented_stage_without_local_runner(self):
        require_numpy()
        from neurobench.data.synthetic import generate_synthetic_calcium_dataset
        from neurobench.pipelines.executor import execute_pipeline

        dataset = generate_synthetic_calcium_dataset(frames=4, height=8, width=8)
        with tempfile.TemporaryDirectory() as tmp:
            paths = dataset.write(Path(tmp) / "fixture")
            with self.assertRaisesRegex(ValueError, "no local runner"):
                execute_pipeline(
                    {
                        "dataset_id": "synthetic",
                        "run_id": "unwired",
                        "pipeline": [
                            {"id": "source", "stage_id": "source_video_import", "params": {"source": paths["video"]}},
                            {"id": "import_review", "stage_id": "review_data_import", "params": {"review_data": "review_data.json"}},
                        ],
                    },
                    run_root=Path(tmp) / "run",
                )

    def test_synthetic_pipeline_executes_trace_event_stages(self):
        require_numpy()
        from neurobench.data.synthetic import generate_synthetic_calcium_dataset
        from neurobench.pipelines.executor import dry_run_pipeline, execute_pipeline

        dataset = generate_synthetic_calcium_dataset(include_impulse_artifact=False)
        with tempfile.TemporaryDirectory() as tmp:
            paths = dataset.write(Path(tmp) / "fixture", dataset_id="synthetic_trace_events")
            spec = {
                "schema_version": 1,
                "dataset_id": "synthetic_trace_events",
                "run_id": "synthetic_trace_events",
                "pipeline": [
                    {"id": "source", "stage_id": "source_video_import", "params": {"source": paths["video"]}},
                    {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 2.0}},
                    {"id": "denoise", "stage_id": "event_preserving_noise_suppression", "params": {"spatial_sigma_px": 0.2}},
                    {"id": "score", "stage_id": "adaptive_ewma_z", "params": {"alpha": 0.15, "threshold_z": 1.5, "epsilon": 0.05}},
                    {
                        "id": "components",
                        "stage_id": "component_filter",
                        "params": {"seed_z": 1.2, "min_area_px": 3, "max_area_px": 120},
                    },
                    {"id": "traces", "stage_id": "local_background_ring", "params": {"outer_radius_px": 8, "neuropil_weight": 0.2}},
                    {"id": "events", "stage_id": "robust_kalman_positive_innovation", "params": {"event_threshold_z": 1.2}},
                ],
            }

            plan = dry_run_pipeline(spec, validate_artifacts=True)
            result = execute_pipeline(spec, run_root=Path(tmp) / "run")
            run_root = Path(result["run_root"])
            manifest = json.loads((run_root / "pipeline_run.json").read_text(encoding="utf-8"))
            traces = json.loads((run_root / "artifacts" / "traces" / "roi_traces.json").read_text(encoding="utf-8"))
            events = json.loads((run_root / "artifacts" / "events" / "kalman_candidate_events.json").read_text(encoding="utf-8"))

            self.assertIn("roi_traces", plan["available_artifacts"])
            self.assertIn("candidate_events", plan["available_artifacts"])
            self.assertEqual(result["status"], "completed")
            self.assertIn("roi_traces", [item["kind"] for item in manifest["artifacts"]])
            self.assertIn("candidate_events", [item["kind"] for item in manifest["artifacts"]])
            self.assertGreater(len(traces["traces"]), 0)
            self.assertGreater(len(events["events"]), 0)


if __name__ == "__main__":
    unittest.main()
