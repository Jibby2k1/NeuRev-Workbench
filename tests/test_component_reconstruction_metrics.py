import numpy as np

from neurobench.metrics.component_reconstruction import component_product_metrics


def test_component_product_metrics_are_scale_and_permutation_invariant():
    traces = np.asarray([[0,1,2,1,0,0,0,0], [0,0,0,0,1,2,1,0]], dtype=float)
    footprints = np.zeros((2,4,4), dtype=float)
    footprints[0,1,1] = 1
    footprints[1,2,2] = 1
    sources = np.stack([traces[1]*3, traces[0]*-2])
    maps = np.stack([(footprints[1]/3).ravel(), (footprints[0]/-2).ravel()], axis=1)
    result = component_product_metrics(traces, footprints, sources, maps)
    assert result["neural_reconstruction_nmse"] < 1e-12
    assert abs(result["mean_peak_retention"]-1) < 1e-12
    assert result["mean_footprint_iou"] == 1
