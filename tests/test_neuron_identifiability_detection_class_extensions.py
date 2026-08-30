import numpy as np

from neurobench.experiments.neuron_identifiability.detection_class_extensions import cramers_v, greedy_spatial_match


def test_greedy_spatial_match_is_one_to_one_and_within_burst():
    detections = [
        {"detection_occurrence_id": "d1", "burst_id": 1, "x_px": 0, "y_px": 0},
        {"detection_occurrence_id": "d2", "burst_id": 1, "x_px": 1, "y_px": 0},
        {"detection_occurrence_id": "d3", "burst_id": 2, "x_px": 0, "y_px": 0},
    ]
    labels = [
        {"observation_id": "a", "burst_id": 1, "x_px": 0, "y_px": 0},
        {"observation_id": "b", "burst_id": 2, "x_px": 0, "y_px": 0},
    ]
    matches = greedy_spatial_match(detections, labels, 2)
    assert [(row[0]["detection_occurrence_id"], row[1]["observation_id"]) for row in matches] == [("d1", "a"), ("d3", "b")]


def test_cramers_v_has_expected_extremes():
    assert cramers_v(np.asarray([[5, 0], [0, 5]])) == 1.0
    assert cramers_v(np.asarray([[5, 5], [5, 5]])) == 0.0
