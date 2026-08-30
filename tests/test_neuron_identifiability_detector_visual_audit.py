from neurobench.experiments.neuron_identifiability.detector_visual_audit import _consolidate


def test_detector_audit_consolidates_within_lane_only():
    rows = [
        {"lane": "coherence_w15", "burst_id": 1, "x_px": 10, "y_px": 10, "score": 2.0, "start_ui": 1, "end_ui": 2},
        {"lane": "coherence_w15", "burst_id": 2, "x_px": 13, "y_px": 14, "score": 1.0, "start_ui": 3, "end_ui": 4},
        {"lane": "propagation_lag2_w15", "burst_id": 1, "x_px": 10, "y_px": 10, "score": 3.0, "start_ui": 1, "end_ui": 2},
    ]
    models = _consolidate(rows)
    assert len(models) == 2
    assert {model["lane"] for model in models} == {"coherence_w15", "propagation_lag2_w15"}
    assert rows[0]["model_roi"] == rows[1]["model_roi"]
    assert rows[0]["model_roi"] != rows[2]["model_roi"]
