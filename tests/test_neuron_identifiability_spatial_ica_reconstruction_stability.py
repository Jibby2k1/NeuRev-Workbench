import numpy as np
from neurobench.experiments.neuron_identifiability.spatial_ica_reconstruction_stability import _corr,compare_reconstructions

def test_corr_handles_scaled_identity():
    x=np.arange(24,dtype=float).reshape(2,3,4)
    assert _corr(x,3*x+2)>.999999

def test_reconstruction_comparison_identical_fixture():
    rng=np.random.default_rng(2);video=np.maximum(rng.normal(size=(24,24,24)),0)
    labels=[{"burst_id":1,"x_px":x,"y_px":y} for x,y in ((8,8),(12,12),(16,16))]
    metrics,rows=compare_reconstructions(video,video,labels,{1:(4,20)},[1],.25)
    assert metrics["pixel_correlation"]>.999999
    assert metrics["median_event_map_correlation"]>.999999
    assert metrics["median_morphology_spearman"]>.999999
    assert metrics["median_peak_error_frames"]==0
    assert len(rows)==1
