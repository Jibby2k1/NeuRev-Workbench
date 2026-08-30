import numpy as np

from neurobench.metrics.sparse_detection import extract_local_maxima, extract_separated_local_maxima


def test_separated_local_maxima_collapses_flat_plateau():
    score=np.zeros((30,30),float); score[10:14,10:14]=5
    assert len(extract_local_maxima(score,3,threshold=1)) == 16
    peaks=extract_separated_local_maxima(score,3,threshold=1)
    assert peaks == [(5.0,10,10),(5.0,13,11)]
    assert (peaks[0][1]-peaks[1][1])**2+(peaks[0][2]-peaks[1][2])**2 > 9


def test_separated_local_maxima_uses_label_free_tie_breaker():
    score=np.zeros((30,30),float); score[10:13,10:13]=5
    secondary=np.zeros_like(score); secondary[12,12]=9
    assert extract_separated_local_maxima(score,4,threshold=1,tie_breaker=secondary) == [(5.0,12,12)]
