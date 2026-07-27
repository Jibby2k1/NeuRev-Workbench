import numpy as np

from neurobench.metrics.sparse_detection import (
    extract_local_maxima,
    match_peaks_one_to_one,
    quiet_calibrated_threshold,
)


def test_peak_ties_use_declared_secondary_then_y_x_order():
    score = np.zeros((9, 9), np.float32)
    tie = np.zeros_like(score)
    score[2, 2] = score[2, 6] = score[6, 2] = 1
    tie[2, 6] = 4
    peaks = extract_local_maxima(score, 1, threshold=0.5, tie_breaker=tie)
    assert [(x, y) for _, x, y in peaks] == [(6, 2), (2, 2), (2, 6)]


def test_one_to_one_matching_does_not_reuse_labels():
    peaks = [(5.0, 5, 5), (4.0, 6, 5)]
    labels = [{"x_px": 5, "y_px": 5}]
    matches, peak_indices = match_peaks_one_to_one(peaks, labels, 2)
    assert len(matches) == 1
    assert peak_indices == {0}


def test_quiet_threshold_is_strictly_above_allowed_peak():
    maps = []
    for value in (1, 2, 3, 4):
        score = np.zeros((9, 9), np.float32); score[4, 4] = value; maps.append(score)
    threshold = quiet_calibrated_threshold(maps, 1, 0.5)
    assert threshold > 2
    assert threshold <= 3
