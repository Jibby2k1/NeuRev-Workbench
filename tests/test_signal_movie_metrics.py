import numpy as np
from neurobench.metrics.signal_movie import signal_movie_metrics


def test_signal_movie_metrics_are_exact_for_truth():
    traces=np.asarray([[0,1,2,1,0,0,0,0]],float)
    footprints=np.zeros((1,4,4)); footprints[0,1:3,1:3]=1
    truth=np.einsum('st,shw->thw',traces,footprints)
    result=signal_movie_metrics(traces,footprints,truth)
    assert result['neural_reconstruction_nmse'] == 0
    assert result['mean_peak_retention'] == 1
    assert result['mean_footprint_iou'] == 1
