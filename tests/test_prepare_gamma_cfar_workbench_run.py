from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "prepare_gamma_cfar_workbench_run.py"


def load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_gamma_cfar_workbench_run", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PrepareGammaCfarWorkbenchRunTests(unittest.TestCase):
    def test_fast_grid_bootstrap_writes_shared_preprocessing(self):
        try:
            import numpy as np
            import scipy  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("NumPy and SciPy are required for preprocessing bootstrap")
        module = load_prepare_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.npy"
            np.save(source, np.arange(5 * 6 * 7, dtype=np.uint8).reshape(5, 6, 7))
            spec = {
                "pipeline": [
                    {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 1.0}},
                    {"id": "smooth", "stage_id": "spatial_gaussian", "params": {"sigma_px": 0.4}},
                    {"id": "score", "stage_id": "robust_positive_local_z", "params": {"epsilon": 1.0}},
                ]
            }
            first_run = root / "sweep" / "001_run"
            preprocessing = first_run / "artifacts" / "preprocessing"

            module.ensure_shared_preprocessing(
                args=SimpleNamespace(source_npy=source),
                spec=spec,
                first_run=first_run,
                highpass_path=preprocessing / "highpass_video.npy",
                smoothed_path=preprocessing / "smoothed_video.npy",
                z_path=preprocessing / "z_stack.npy",
            )

            for name in ["highpass_video.npy", "smoothed_video.npy", "z_stack.npy"]:
                path = preprocessing / name
                self.assertTrue(path.exists(), name)
                self.assertEqual(tuple(np.load(path).shape), (5, 6, 7))


    def test_write_sweep_spec_defaults_to_high_recall_components(self):
        module = load_prepare_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "sweep.json"
            module.write_sweep_spec(
                SimpleNamespace(
                    dataset_id="tiny",
                    run_id="grid",
                    source_npy=root / "source.npy",
                    out=out,
                )
            )
            spec = json.loads(out.read_text(encoding="utf-8"))

        components = next(step for step in spec["pipeline"] if step["id"] == "components")
        support_axis = next(item for item in spec["sweep"]["parameters"] if item["stage"] == "components")
        pfa_axis = next(item for item in spec["sweep"]["parameters"] if item["stage"] == "cfar_small_ref")
        self.assertEqual(components["params"]["min_area_px"], 6)
        self.assertLess(components["params"]["seed_z"], 4.0)
        self.assertIn(8, support_axis["values"])
        self.assertIn(0.14, pfa_axis["values"])


    def test_green_excess_channel_conversion_uses_positive_green_minus_red_blue(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("NumPy is required for channel conversion")
        module = load_prepare_module()
        rgb = np.array([[[10, 40, 4], [20, 12, 18], [0, 5, 0]]], dtype=np.uint8)
        converted = module.convert_rgb_frame_to_channel(rgb, "green_excess")
        np.testing.assert_allclose(converted, np.array([[33.0, 0.0, 5.0]], dtype=np.float32))
        self.assertEqual(converted.dtype, np.float32)

    def test_pipeline_stage_params_prefers_exact_step_id_before_stage_id_fallback(self):
        module = load_prepare_module()
        spec = {
            "pipeline": [
                {"id": "cfar_small_ref", "stage_id": "gamma_cfar", "params": {"training_radius_px": 6}},
                {"id": "cfar_large_ref", "stage_id": "gamma_cfar", "params": {"training_radius_px": 24}},
            ]
        }
        self.assertEqual(module.pipeline_stage_params(spec, "cfar_large_ref", "gamma_cfar")["training_radius_px"], 24)
        self.assertEqual(module.pipeline_stage_params(spec, "missing", "gamma_cfar")["training_radius_px"], 6)

    def test_write_green_excess_cfar_spec_defines_single_cfar_mini_grid(self):
        module = load_prepare_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "green.json"
            module.write_green_excess_cfar_spec(
                SimpleNamespace(
                    dataset_id="external_test",
                    run_id="green_excess_single_cfar_v1",
                    source_npy=root / "green.npy",
                    out=out,
                )
            )
            spec = json.loads(out.read_text(encoding="utf-8"))

        stages = {step["id"]: step for step in spec["pipeline"]}
        self.assertEqual(stages["highpass"]["params"]["sigma_frames"], 0.0)
        self.assertEqual(stages["smooth"]["params"]["sigma_px"], 0.8)
        self.assertIn("green_single_cfar", stages)
        self.assertEqual(stages["components"]["params"]["seed_z"], 2.5)
        axes = {(item["stage"], item["param"]): item["values"] for item in spec["sweep"]["parameters"]}
        self.assertEqual(axes[("green_single_cfar", "pfa")], [0.01, 0.02, 0.04])
        self.assertEqual(axes[("components", "support_min_frames")], [1, 6])

    def test_write_green_excess_roi_state_spec_defines_high_recall_grid(self):
        module = load_prepare_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "green_state.json"
            module.write_green_excess_roi_state_spec(
                SimpleNamespace(
                    dataset_id="external_test",
                    run_id="green_excess_roi_state_v2",
                    source_npy=root / "green.npy",
                    out=out,
                )
            )
            spec = json.loads(out.read_text(encoding="utf-8"))

        stages = {step["id"]: step for step in spec["pipeline"]}
        self.assertEqual(stages["highpass"]["params"]["sigma_frames"], 0.0)
        self.assertEqual(stages["components"]["params"]["projection_blob_z"], 1.5)
        self.assertEqual(stages["activity_states"]["params"]["sustained_z"], 1.2)
        axes = {(item["stage"], item["param"]): item["values"] for item in spec["sweep"]["parameters"]}
        self.assertEqual(axes[("green_single_cfar", "pfa")], [0.02, 0.04, 0.08])
        self.assertEqual(axes[("components", "projection_blob_z")], [1.5, 2.0])
        self.assertEqual(axes[("components", "support_min_frames")], [1, 5])

        from neurobench.architecture_runs import build_planned_manifest

        planned = build_planned_manifest(spec)
        self.assertEqual(len(planned["runs"]), 12)

    def test_write_green_excess_multiscale_cfar_spec_expands_to_144_descriptive_runs(self):
        module = load_prepare_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "green_multiscale.json"
            module.write_green_excess_multiscale_cfar_spec(
                SimpleNamespace(
                    dataset_id="external_test",
                    run_id="green_excess_multiscale_cfar_v3",
                    source_npy=root / "green.npy",
                    out=out,
                )
            )
            spec = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(spec["execution"], {"device": "cuda", "backend": "cupy_cuda", "cpu_fallback": False})
        planned = module.planned_manifest_for_spec(spec)
        self.assertEqual(len(planned["runs"]), 144)
        first = planned["runs"][0]
        self.assertTrue(first["run_id"].startswith("green_roi_mscfar_v3_pfa"))
        self.assertIn("_sR", first["run_id"])
        self.assertIn("_lR", first["run_id"])
        self.assertIn("Green ROI MSCFAR v3", first["label"])
        small = module.pipeline_stage_params(first, "cfar_small_ref", "gamma_cfar")
        large = module.pipeline_stage_params(first, "cfar_large_ref", "gamma_cfar")
        components = module.pipeline_stage_params(first, "components", "component_filter")
        self.assertEqual(large["pfa"], small["pfa"])
        self.assertIn(components["fusion_mode"], {"intersection", "union"})
        self.assertEqual(first["summary"]["small_training_radius_px"], small["training_radius_px"])
        self.assertEqual(first["summary"]["large_training_radius_px"], large["training_radius_px"])
        self.assertTrue(components["split_large_components"])
        self.assertEqual(components["split_min_distance_px"], 6)

    def test_green_multiscale_large_components_split_into_peak_candidates(self):
        try:
            import numpy as np
            from scipy import ndimage
        except ModuleNotFoundError:
            self.skipTest("NumPy and SciPy are required for component splitting")
        module = load_prepare_module()
        support = np.zeros((30, 30), dtype=np.float32)
        z_projection = np.zeros((30, 30), dtype=np.float32)
        projection_score = np.zeros((30, 30), dtype=np.float32)
        projection_support = np.zeros((30, 30), dtype=bool)
        projection_support[5:22, 5:24] = True
        for y, x, value in [(8, 8, 9.0), (8, 18, 8.0), (17, 13, 7.0)]:
            z_projection[y, x] = value
            projection_score[y, x] = value
            support[y, x] = 20

        without_split = module.union_component_candidates_from_support(
            support,
            z_projection,
            projection_score,
            projection_support,
            support_min_frames=1,
            seed_z=1.0,
            min_area=6,
            max_area=50,
            ndimage=ndimage,
            split_large_components=False,
        )
        with_split = module.union_component_candidates_from_support(
            support,
            z_projection,
            projection_score,
            projection_support,
            support_min_frames=1,
            seed_z=1.0,
            min_area=6,
            max_area=50,
            ndimage=ndimage,
            split_large_components=True,
            split_min_distance_px=6,
            split_area_px=80,
            split_max_peaks=10,
        )

        self.assertEqual(without_split, [])
        self.assertEqual(len(with_split), 3)
        self.assertTrue(all(row.get("split_from_large_component") for row in with_split))
        centers = {(round(row["x"]), round(row["y"])) for row in with_split}
        self.assertEqual(centers, {(8, 8), (18, 8), (13, 17)})

    def test_green_multiscale_commands_are_registered(self):
        module = load_prepare_module()
        parser = module.build_arg_parser()
        args = parser.parse_args(
            [
                "write-green-excess-multiscale-cfar-spec",
                "--dataset-id",
                "external_test",
                "--source-npy",
                "green.npy",
                "--out",
                "spec.json",
            ]
        )
        self.assertIs(args.func, module.write_green_excess_multiscale_cfar_spec)

        args = parser.parse_args(
            [
                "run-green-excess-multiscale-cfar-grid",
                "--spec",
                "spec.json",
                "--sweep-root",
                "runs",
                "--source-npy",
                "green.npy",
            ]
        )
        self.assertIs(args.func, module.run_green_excess_multiscale_cfar_grid)
        self.assertEqual(args.gpu_cfar_chunk_frames, 32)
        self.assertEqual(args.gpu_preprocess_chunk_frames, 32)

    def test_green_multiscale_shared_intermediate_keys_include_radii_and_fusion(self):
        module = load_prepare_module()
        run = {
            "run_id": "green_roi_mscfar_v3_pfa004_sR06_lR18_union_sup015_pz15",
            "sweep": {
                "parameters": [
                    {"stage": "cfar_small_ref", "param": "pfa", "value": 0.04},
                    {"stage": "cfar_small_ref", "param": "training_radius_px", "value": 6},
                    {"stage": "cfar_large_ref", "param": "training_radius_px", "value": 18},
                    {"stage": "components", "param": "fusion_mode", "value": "union"},
                ]
            },
        }
        small_artifact = {"summary": {"pfa": 0.04, "training_radius_px": 6}, "path": "small.npy"}
        large_artifact = {
            "summary": {
                "pfa": 0.04,
                "training_radius_px": 18,
                "small_training_radius_px": 6,
                "large_training_radius_px": 18,
                "combine_mode": "union",
            },
            "path": "large.npy",
        }

        self.assertEqual(
            module.shared_intermediate_key(run, "cfar_small_ref", small_artifact),
            "cfar_small_ref_pfa_0.04_radius_6",
        )
        self.assertEqual(
            module.shared_intermediate_key(run, "cfar_large_ref", large_artifact),
            "cfar_large_ref_pfa_0.04_small_radius_6_large_radius_18_union",
        )

    def test_require_cupy_cuda_fails_without_cpu_fallback_message_when_cuda_unavailable(self):
        module = load_prepare_module()
        try:
            module.require_cupy_cuda()
        except RuntimeError as exc:
            self.assertIn("CPU fallback", str(exc))
        else:
            self.skipTest("CUDA/CuPy is available in this environment")

    def test_green_roi_state_commands_are_registered(self):
        module = load_prepare_module()
        parser = module.build_arg_parser()
        args = parser.parse_args(
            [
                "write-green-excess-roi-state-spec",
                "--dataset-id",
                "external_test",
                "--source-npy",
                "green.npy",
                "--out",
                "spec.json",
            ]
        )
        self.assertIs(args.func, module.write_green_excess_roi_state_spec)

        args = parser.parse_args(
            [
                "run-green-excess-roi-state-grid",
                "--spec",
                "spec.json",
                "--sweep-root",
                "runs",
                "--source-npy",
                "green.npy",
            ]
        )
        self.assertIs(args.func, module.run_green_excess_roi_state_grid)

    def test_union_component_candidates_preserves_candidate_sources(self):
        try:
            import numpy as np
            from scipy import ndimage
        except ModuleNotFoundError:
            self.skipTest("NumPy and SciPy are required for component source test")
        module = load_prepare_module()
        mask = np.zeros((2, 6, 6), dtype=bool)
        mask[:, 1, 1] = True
        z_projection = np.zeros((6, 6), dtype=np.float32)
        z_projection[1, 1] = 4.0
        projection_score = np.zeros((6, 6), dtype=np.float32)
        projection_score[1, 2] = 2.0
        projection_score[4, 4] = 2.5
        projection_support = projection_score > 0

        candidates = module.union_component_candidates(
            mask,
            z_projection,
            projection_score,
            projection_support,
            support_min_frames=2,
            seed_z=2.0,
            min_area=1,
            max_area=20,
            ndimage=ndimage,
        )

        sources = {candidate["candidate_source"] for candidate in candidates}
        self.assertIn("union", sources)
        self.assertIn("projection_blob", sources)
        union = next(candidate for candidate in candidates if candidate["candidate_source"] == "union")
        self.assertEqual(union["source_pixels"], {"cfar": 1, "projection": 1})

    def test_green_roi_state_shared_intermediate_keys_are_distinct(self):
        module = load_prepare_module()
        run = {
            "run_id": "green_excess_roi_state_v2__sweep_001",
            "sweep": {
                "parameters": [
                    {"stage": "components", "param": "projection_blob_z", "value": 2.0},
                    {"stage": "components", "param": "support_min_frames", "value": 5},
                ]
            },
        }
        artifact = {"summary": {"projection_blob_z": 1.5}, "path": "projection.npy"}

        self.assertEqual(module.shared_intermediate_key(run, "highpass", artifact), "green_excess_highpass")
        self.assertEqual(
            module.shared_intermediate_key(run, "green_projection_blob_map", artifact),
            "green_projection_blob_map_z_2.0",
        )
        self.assertEqual(
            module.shared_intermediate_key(run, "green_projection_score", artifact),
            "green_excess_projection_score",
        )

    def test_trace_activity_state_separates_peak_and_sustained_frames(self):
        try:
            import numpy as np  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("NumPy is required for activity-state scoring")
        module = load_prepare_module()
        payload = module.trace_activity_state_payload(
            [0.0, 0.0, 5.0, 0.0, 2.0, 2.0, 2.0, 2.0],
            [{"frame": 2}],
            sustained_z=0.5,
            tonic_z=99.0,
            peak_window_frames=0,
        )

        self.assertEqual(payload["activity_summary"]["peak_frame_count"], 2)
        self.assertGreater(payload["activity_summary"]["sustained_frame_count"], 0)
        self.assertIn({"start": 5, "end": 8, "state": "sustained"}, payload["activity_intervals"])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe are required")
    def test_prepare_mp4_dataset_decodes_luma_frames_and_manifest(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("NumPy is required for MP4 conversion")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mp4 = root / "tiny.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=8x6:rate=2:duration=1",
                    "-pix_fmt",
                    "yuv420p",
                    str(mp4),
                ],
                check=True,
            )
            out_npy = root / "out" / "tiny.npy"
            app_dir = root / "app"
            manifest = root / "manifest.json"

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "prepare-mp4-dataset",
                    "--input-mp4",
                    str(mp4),
                    "--dataset-id",
                    "tiny",
                    "--output-npy",
                    str(out_npy),
                    "--app-dir",
                    str(app_dir),
                    "--manifest",
                    str(manifest),
                    "--channel",
                    "luma",
                ],
                cwd=ROOT,
                check=True,
            )

            stack = np.load(out_npy)
            self.assertEqual(stack.ndim, 3)
            self.assertEqual(stack.shape[1:], (6, 8))
            self.assertEqual(stack.dtype, np.uint8)
            self.assertTrue((app_dir / "frames" / "frame_001.png").exists())
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset_id"], "tiny")
            self.assertIsNone(payload["pixel_size_microns"])
            self.assertEqual(payload["source"]["template"], "local_mp4")


if __name__ == "__main__":
    unittest.main()
