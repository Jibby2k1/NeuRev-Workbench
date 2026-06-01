from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class GammaCfarSweepReportTests(unittest.TestCase):
    def test_summarizer_reports_burden_size_and_ranked_candidates(self):
        from neurobench.reports.gamma_cfar_sweep import render_gamma_cfar_sweep_markdown, summarize_gamma_cfar_sweep

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "001_gamma"
            artifacts = run_root / "artifacts" / "candidates"
            artifacts.mkdir(parents=True)
            (root / "sweep_summary.json").write_text(
                json.dumps(
                    {
                        "dataset_id": "d",
                        "sweep": {"id": "s"},
                        "status": "completed",
                        "runs": [
                            {
                                "run_id": "gamma",
                                "run_root": "001_gamma",
                                "status": "completed",
                                "sweep_parameters": [
                                    {"stage": "cfar_small_ref", "param": "pfa", "value": 0.06},
                                    {"stage": "components", "param": "min_area_px", "value": 6},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (artifacts / "roi_candidates.json").write_text(
                json.dumps({"candidates": [{"id": "roi_001", "area_px": 100}, {"id": "roi_002", "area_px": 25}]}),
                encoding="utf-8",
            )
            (artifacts / "ranked_candidates.json").write_text(
                json.dumps(
                    {
                        "ranked_candidates": [
                            {"candidate_id": "roi_001", "rank": 1, "priority_score": 2.0, "reasons": ["usable trace SNR"]}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (run_root / "pipeline_run.json").write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "kind": "candidate_mask",
                                "path": "artifacts/candidates/small.npy",
                                "summary": {"active_fraction": 0.04},
                            },
                            {
                                "kind": "candidate_mask",
                                "path": "artifacts/candidates/large.npy",
                                "summary": {"active_fraction": 0.02, "previous_mask_step": "cfar_small_ref"},
                            },
                            {
                                "kind": "roi_candidates",
                                "path": "artifacts/candidates/roi_candidates.json",
                                "summary": {"count": 2, "min_area_px": 20, "support_min_frames": 25},
                            },
                            {
                                "kind": "candidate_events",
                                "path": "artifacts/events/kalman_candidate_events.json",
                                "summary": {"event_count": 3},
                            },
                            {
                                "kind": "ranked_candidates",
                                "path": "artifacts/candidates/ranked_candidates.json",
                                "summary": {"count": 2},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize_gamma_cfar_sweep(root)
            markdown = render_gamma_cfar_sweep_markdown(summary)

        self.assertEqual(summary["completed_count"], 1)
        self.assertEqual(summary["runs"][0]["candidate_count"], 2)
        self.assertEqual(summary["runs"][0]["event_count"], 3)
        self.assertEqual(summary["runs"][0]["final_active_fraction"], 0.02)
        self.assertEqual(summary["runs"][0]["component_support_min_frames"], 25)
        self.assertGreater(summary["runs"][0]["median_equivalent_diameter_um"], 0)
        self.assertIn("Gamma CFAR Grid Brief", markdown)
        self.assertIn("roi_001", markdown)

    def test_summarizer_supports_pixel_unit_briefs_when_pixel_size_unknown(self):
        from neurobench.reports.gamma_cfar_sweep import render_gamma_cfar_sweep_markdown, summarize_gamma_cfar_sweep

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "001_gamma"
            artifacts = run_root / "artifacts" / "candidates"
            artifacts.mkdir(parents=True)
            (root / "sweep_summary.json").write_text(
                json.dumps(
                    {
                        "dataset_id": "external_test",
                        "sweep": {"id": "s"},
                        "status": "completed",
                        "runs": [{"run_id": "gamma", "run_root": "001_gamma", "status": "completed"}],
                    }
                ),
                encoding="utf-8",
            )
            (artifacts / "roi_candidates.json").write_text(
                json.dumps({"candidates": [{"id": "roi_001", "area_px": 100}]}),
                encoding="utf-8",
            )
            (run_root / "pipeline_run.json").write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {"kind": "candidate_mask", "path": "mask.npy", "summary": {"active_fraction": 0.02}},
                            {
                                "kind": "roi_candidates",
                                "path": "artifacts/candidates/roi_candidates.json",
                                "summary": {"count": 1, "min_area_px": 20, "support_min_frames": 20},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize_gamma_cfar_sweep(root, size_mode="pixels")
            markdown = render_gamma_cfar_sweep_markdown(summary)

        self.assertIsNone(summary["pixel_size_um"])
        self.assertIsNone(summary["runs"][0]["median_equivalent_diameter_um"])
        self.assertIn("Pixel size: unknown", markdown)
        self.assertIn("Median diameter px", markdown)
        self.assertNotIn("um/px", markdown)
        self.assertNotIn("Expected hindbrain neuron diameter", markdown)
        self.assertNotIn("Plausible size", markdown)


if __name__ == "__main__":
    unittest.main()
