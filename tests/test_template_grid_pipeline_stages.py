from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class TemplateGridPipelineStageTests(unittest.TestCase):
    def test_synthetic_template_grid_pipeline_executes(self):
        from neurobench.data.synthetic_fish import generate_synthetic_grid_fish_videos
        from neurobench.pipelines.executor import dry_run_pipeline, execute_pipeline

        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            generate_synthetic_grid_fish_videos(video_count_per_label=1, frames=8, height=32, width=32, rotation_deg_range=(-1,1), translation_px_range=(-1,1), include_outlier_frames=False).write(root/"videos", suffix=".tif")
            spec={"schema_version":1,"dataset_id":"synthetic_grid","run_id":"template_grid_pipeline","pipeline":[
                {"id":"manifest","stage_id":"video_manifest_build","params":{"input_dir":str(root/"videos")}},
                {"id":"template","stage_id":"template_build_from_video","params":{"reference_video_id":"1_neutral"}},
                {"id":"registration","stage_id":"template_register_video","params":{"rotation_range_deg":[-2,2],"rotation_step_deg":1.0}},
                {"id":"registered","stage_id":"apply_video_registration"},
                {"id":"grid","stage_id":"grid_32x32_generate"},
                {"id":"grid_states","stage_id":"grid_state_extract"},
            ]}
            plan=dry_run_pipeline(spec, validate_artifacts=True)
            result=execute_pipeline(spec, run_root=root/"run")
            manifest=json.loads((root/"run"/"pipeline_run.json").read_text())

        self.assertIn("grid_states", plan["available_artifacts"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(manifest["artifacts"][-1]["kind"], "grid_states")
