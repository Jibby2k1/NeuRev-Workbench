from __future__ import annotations

import unittest


class PipelineCatalogTests(unittest.TestCase):
    def test_valid_structured_pipeline_merges_defaults(self):
        from neurobench.pipeline_catalog import normalize_pipeline

        pipeline = normalize_pipeline(
            [
                {"id": "source", "stage_id": "source_video_import", "params": {"source": "raw.tif"}},
                {"id": "denoise", "stage_id": "event_preserving_noise_suppression"},
                {"id": "score", "stage_id": "trace_event_scoring", "params": {"event_threshold_z": 2.4}},
            ],
            require_structured=True,
        )

        self.assertEqual(pipeline[1]["params"]["spatial_sigma_px"], 1.0)
        self.assertEqual(pipeline[1]["params"]["temporal_window_frames"], 3)
        self.assertEqual(pipeline[2]["params"]["event_threshold_z"], 2.4)

    def test_structured_pipeline_rejects_invalid_order(self):
        from neurobench.pipeline_catalog import normalize_pipeline

        with self.assertRaisesRegex(ValueError, "out of order"):
            normalize_pipeline(
                [
                    {"id": "score", "stage_id": "trace_event_scoring", "params": {"event_threshold_z": 2.4}},
                    {"id": "source", "stage_id": "source_video_import", "params": {"source": "raw.tif"}},
                ],
                require_structured=True,
            )

    def test_structured_pipeline_rejects_missing_required_params(self):
        from neurobench.pipeline_catalog import normalize_pipeline

        with self.assertRaisesRegex(ValueError, "missing required param 'event_threshold_z'"):
            normalize_pipeline(
                [
                    {"id": "source", "stage_id": "source_video_import", "params": {"source": "raw.tif"}},
                    {"id": "score", "stage_id": "trace_event_scoring"},
                ],
                require_structured=True,
            )

    def test_structured_pipeline_rejects_duplicate_ids(self):
        from neurobench.pipeline_catalog import normalize_pipeline

        with self.assertRaisesRegex(ValueError, "Duplicate pipeline step id 'source'"):
            normalize_pipeline(
                [
                    {"id": "source", "stage_id": "source_video_import", "params": {"source": "raw.tif"}},
                    {"id": "source", "stage_id": "temporal_highpass_gaussian"},
                ],
                require_structured=True,
            )

    def test_legacy_pipeline_is_preserved_without_structured_validation(self):
        from neurobench.pipeline_catalog import normalize_pipeline

        legacy = [
            {"name": "generate_neuron_review_app"},
            {"name": "trace_event_scoring", "params": {"event_threshold_z": 2.4}},
        ]

        self.assertEqual(normalize_pipeline(legacy), legacy)

    def test_catalog_exposes_parameter_docs_and_realtime_metadata(self):
        from neurobench.pipeline_catalog import LOCAL_RUNNER_STAGE_IDS, catalog_as_dict

        catalog = catalog_as_dict()

        self.assertIn("adaptive_gamma_cfar", catalog)
        self.assertIn("artifact_classifier_v1", catalog)
        self.assertEqual(catalog["artifact_classifier_v1"]["availability"], "planned")
        self.assertEqual(catalog["adaptive_gamma_cfar"]["ui_group"], "detection")
        self.assertTrue(catalog["adaptive_gamma_cfar"]["runner_available"])
        self.assertTrue(catalog["adaptive_gamma_cfar"]["locally_runnable"])
        self.assertFalse(catalog["review_data_import"]["runner_available"])
        self.assertFalse(catalog["review_data_import"]["locally_runnable"])
        self.assertIn("adaptive_threshold_trace", catalog["adaptive_gamma_cfar"]["expected_qc_outputs"])
        self.assertIn("projection_blob_z", catalog["component_filter"]["parameter_docs"])
        self.assertIn("sustained_z", catalog["trace_event_scoring"]["parameter_docs"])
        self.assertIn("tonic_z", catalog["trace_event_scoring"]["parameter_docs"])
        self.assertIn("peak_window_frames", catalog["trace_event_scoring"]["parameter_docs"])
        for stage_id, stage in catalog.items():
            with self.subTest(stage_id=stage_id):
                self.assertTrue(stage["description"])
                self.assertTrue(stage["why_use_it"])
                self.assertIn(stage["availability"], {"implemented", "planned", "external_import"})
                self.assertIsInstance(stage["runner_available"], bool)
                self.assertIsInstance(stage["locally_runnable"], bool)
                self.assertTrue(stage["ui_group"])
                self.assertIsInstance(stage["expected_qc_outputs"], tuple)
                self.assertIn(stage["real_time_profile"]["mode"], {"streaming", "batch", "offline", "unknown"})
                self.assertIn("closed_loop_candidate", stage["real_time_profile"])
                for name, doc in stage["parameter_docs"].items():
                    self.assertTrue(doc["meaning"], name)
                    self.assertIn("why", doc)

        for stage_id in LOCAL_RUNNER_STAGE_IDS:
            with self.subTest(local_runner_stage=stage_id):
                self.assertTrue(
                    catalog[stage_id]["expected_qc_outputs"],
                    f"{stage_id} should declare Process Lab outputs",
                )


if __name__ == "__main__":
    unittest.main()
