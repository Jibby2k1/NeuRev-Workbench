from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class ReviewRoiSidecarTests(unittest.TestCase):
    def test_sidecar_writer_splits_trace_heavy_rois_from_summary(self):
        from neurobench.workbench.roi_payloads import build_stencil_gap_report, write_review_roi_sidecars

        rois = [
            {
                "id": "roi_001",
                "centroidX": 10,
                "centroidY": 12,
                "area": 9,
                "points": [[10, 12]],
                "dffTrace": [0.0, 0.1, 1.0, 0.2, 1.2, 0.1],
                "rawTrace": [1, 2, 5, 2, 6, 2],
            },
            {
                "id": "roi_002",
                "centroidX": 90,
                "centroidY": 90,
                "area": 12,
                "events": [{"frame": 3, "score": 4.5}],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = write_review_roi_sidecars(
                rois,
                summary_path=root / "review_rois.summary.json",
                shard_dir=root / "roi_trace_shards",
                run_id="run_a",
                frame_count=6,
                event_threshold_z=1.0,
                stencil_points=[[0, 0], [50, 0], [50, 50], [0, 50]],
                gap_report_path=root / "stencil_gap_report.json",
            )
            summary = json.loads((root / "review_rois.summary.json").read_text(encoding="utf-8"))
            shard = json.loads((root / "roi_trace_shards" / "roi_001.json").read_text(encoding="utf-8"))
            gap_report = json.loads((root / "stencil_gap_report.json").read_text(encoding="utf-8"))

        self.assertEqual(result["roi_count"], 2)
        self.assertEqual(result["trace_shard_count"], 1)
        self.assertEqual(summary["payload_kind"], "review_rois_summary")
        self.assertNotIn("dffTrace", summary["review_rois"][0])
        self.assertEqual(summary["review_rois"][0]["trace_file"], "roi_trace_shards/roi_001.json")
        self.assertGreater(summary["review_rois"][0]["event_count"], 0)
        self.assertEqual(shard["roi_id"], "roi_001")
        self.assertIn("dffTrace", shard)
        self.assertTrue(gap_report["stencil_available"])
        self.assertGreaterEqual(len(build_stencil_gap_report(summary["review_rois"], [[0, 0], [50, 0], [50, 50], [0, 50]], run_id="run_a")["gaps"]), 1)


if __name__ == "__main__":
    unittest.main()
