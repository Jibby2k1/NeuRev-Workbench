from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[1]


class AnnotationModelTests(unittest.TestCase):
    def test_migration_strips_persisted_cfar_undo_snapshots_but_keeps_masks(self):
        from neurobench.annotations import migrate_annotations_v3

        region = {
            "foreground_points": [[2, 3]],
            "background_points": [[4, 5]],
            "edit_history": [{"foreground_points": [[1, 1]]}],
        }
        migrated = migrate_annotations_v3(
            {
                "schema_version": 3,
                "rois": {"1": {"cfar_regions": region}},
                "runs": {"run_a": {"rois": {"1": {"cfar_regions": region}}}},
            }
        )

        self.assertNotIn("edit_history", migrated["rois"]["1"]["cfar_regions"])
        self.assertNotIn("edit_history", migrated["runs"]["run_a"]["rois"]["1"]["cfar_regions"])
        self.assertEqual(migrated["rois"]["1"]["cfar_regions"]["foreground_points"], [[2, 3]])
    def test_layout_migration_uses_single_canvas_without_changing_labels(self):
        from neurobench.annotations import migrate_annotations_v3

        migrated = migrate_annotations_v3(
            {
                "rois": {"1": {"cell_state": "accepted"}},
                "settings": {"reviewSideBySide": True, "uiMode": "guided"},
            }
        )
        current = migrate_annotations_v3(
            {"settings": {"reviewLayoutVersion": 2, "reviewSideBySide": True}}
        )

        self.assertFalse(migrated["settings"]["reviewSideBySide"])
        self.assertEqual(migrated["settings"]["reviewLayoutVersion"], 2)
        self.assertEqual(migrated["rois"]["1"]["cell_state"], "accepted")
        self.assertTrue(current["settings"]["reviewSideBySide"])

    def test_migration_adds_reason_tags_and_confidence_defaults(self):
        from neurobench.annotations import migrate_annotations_v3

        migrated = migrate_annotations_v3(
            {
                "rois": {"1": {"state": "accept", "reason_codes": "compact,event_supported", "confidence": "HIGH", "reviewer_id": "RV"}},
                "events": {"1:4": {"state": "unsure", "updatedAt": "2026-05-14T00:00:00Z"}},
                "suggestions": {"S1": {"state": "missed", "reason_tags": ["uncovered"]}},
                "virtualRois": {"V1": {"roi_kind": "virtual_merge", "confidence": "invalid"}},
                "splitMergeDecisions": {
                    "D1": {
                        "type": "MERGE",
                        "state": "ACCEPTED",
                        "source_rois": "1, 2",
                        "confidence": "MEDIUM",
                        "reason_codes": "overlap,same_trace",
                    }
                },
            }
        )

        self.assertEqual(migrated["rois"]["1"]["cell_state"], "accepted")
        self.assertEqual(migrated["rois"]["1"]["reason_tags"], ["compact", "event_supported"])
        self.assertEqual(migrated["rois"]["1"]["confidence"], "high")
        self.assertEqual(migrated["rois"]["1"]["reviewer_id"], "RV")
        self.assertEqual(migrated["events"]["1:4"]["reason_tags"], [])
        self.assertEqual(migrated["events"]["1:4"]["updatedAt"], "2026-05-14T00:00:00Z")
        self.assertEqual(migrated["events"]["1:4"]["confidence"], "")
        self.assertEqual(migrated["suggestions"]["S1"]["reason_tags"], ["uncovered"])
        self.assertEqual(migrated["virtualRois"]["V1"]["confidence"], "")
        self.assertEqual(migrated["splitMergeDecisions"]["D1"]["decision_type"], "merge")
        self.assertEqual(migrated["splitMergeDecisions"]["D1"]["decision_state"], "accepted")
        self.assertEqual(migrated["splitMergeDecisions"]["D1"]["source_roi_ids"], ["1", "2"])
        self.assertEqual(migrated["splitMergeDecisions"]["D1"]["reason_tags"], ["overlap", "same_trace"])
        self.assertEqual(migrated["splitMergeDecisions"]["D1"]["confidence"], "medium")

    def test_annotation_set_roundtrip_validate_and_summary(self):
        from neurobench.models.annotations import AnnotationSet

        annotations = AnnotationSet.from_dict(
            {
                "rois": {"1": {"cell_state": "accepted", "confidence": "high", "reason_tags": ["compact"]}},
                "events": {"1:4": {"event_state": "accepted", "confidence": "medium", "reason_tags": ["clear_onset"]}},
                "suggestions": {"S1": {"state": "missed", "confidence": "low", "reason_tags": ["uncovered"]}},
                "splitMergeDecisions": {"D1": {"decision_type": "split", "source_roi_ids": ["1"], "decision_state": "accepted"}},
            }
        )

        annotations.validate()
        summary = annotations.summary()

        self.assertEqual(summary["roi_annotations"], 1)
        self.assertEqual(summary["event_annotations"], 1)
        self.assertEqual(summary["suggestion_annotations"], 1)
        self.assertEqual(summary["split_merge_decisions"], 1)
        self.assertEqual(summary["roi_confidence_counts"], {"high": 1})
        self.assertEqual(summary["event_confidence_counts"], {"medium": 1})
        self.assertEqual(summary["reason_tag_counts"], {"compact": 1, "clear_onset": 1, "uncovered": 1})

    def test_manual_virtual_roi_geometry_roundtrip(self):
        from neurobench.models.annotations import AnnotationSet

        annotations = AnnotationSet.from_dict(
            {
                "virtualRois": {
                    "MR_1": {
                        "id": "MR_1",
                        "roi_kind": "manual_circle",
                        "source_roi_ids": [],
                        "cell_state": "accepted",
                        "confidence": "medium",
                        "reason_tags": ["manual"],
                        "centroidX": 4.0,
                        "centroidY": 5.0,
                        "bbox": [2, 3, 6, 7],
                        "area": 13,
                        "points": [[4, 5], [4, 6]],
                    }
                }
            }
        )

        annotations.validate()
        payload = annotations.to_dict()

        self.assertEqual(payload["virtualRois"]["MR_1"]["roi_kind"], "manual_circle")
        self.assertEqual(payload["virtualRois"]["MR_1"]["reason_tags"], ["manual"])
        self.assertEqual(payload["virtualRois"]["MR_1"]["points"], [[4, 5], [4, 6]])

    def test_annotation_schema_rejects_invalid_confidence(self):
        from neurobench.validation.schemas import validate_dict

        payload = {
            "schema_version": 3,
            "rois": {"1": {"cell_state": "accepted", "confidence": "certain"}},
            "events": {},
            "suggestions": {},
            "settings": {},
        }

        with self.assertRaises(jsonschema.ValidationError) as ctx:
            validate_dict(payload, "annotations")

        self.assertEqual(list(ctx.exception.path), ["rois", "1", "confidence"])

    def test_annotation_set_load_write_json_roundtrip(self):
        from neurobench.models.annotations import AnnotationSet

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.json"
            AnnotationSet.from_dict({"rois": {"1": {"confidence": "high"}}}).write_json(path)
            loaded = AnnotationSet.load_json(path)

        self.assertEqual(loaded.to_dict()["rois"]["1"]["confidence"], "high")

    def test_export_annotations_includes_confidence_and_reason_tags(self):
        review = {
            "video": {"width": 4, "height": 4, "frames": 2, "framePattern": "frame_%04d.png"},
            "parameters": {},
            "rois": [{"id": 1, "events": [{"frame": 4}]}],
        }
        annotations = {
            "schema_version": 3,
            "rois": {"1": {"cell_state": "accepted", "confidence": "high", "reason_tags": ["compact"], "reviewer_id": "RV"}},
            "events": {"1:4": {"event_state": "accepted", "confidence": "medium", "reason_tags": ["clear_onset"], "reviewer_id": "RV"}},
            "suggestions": {},
            "settings": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_path = root / "review_data.json"
            ann_path = root / "annotations.json"
            out_dir = root / "exports"
            review_path.write_text(json.dumps(review), encoding="utf-8")
            ann_path.write_text(json.dumps(annotations), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "tools/export_annotations.py",
                    "--review-data",
                    str(review_path),
                    "--annotations",
                    str(ann_path),
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            roi_tsv = (out_dir / "accepted_rois.tsv").read_text(encoding="utf-8")
            event_tsv = (out_dir / "accepted_events.tsv").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("confidence\treason_tags", roi_tsv.splitlines()[0])
        self.assertIn("reviewer_id\tupdatedAt", roi_tsv.splitlines()[0])
        self.assertIn("high\tcompact", roi_tsv)
        self.assertIn("medium\tclear_onset", event_tsv)
        self.assertIn("RV", event_tsv)


if __name__ == "__main__":
    unittest.main()
