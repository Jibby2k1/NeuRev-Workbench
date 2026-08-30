import numpy as np
from neurobench.experiments.neuron_identifiability.spatial_ica_objective_morphology import _bh,_component_metrics


def test_component_metrics_prefers_centered_compact_signal():
    yy,xx=np.indices((41,41)); centered=np.exp(-((xx-20)**2+(yy-20)**2)/8)
    shifted=np.exp(-((xx-29)**2+(yy-20)**2)/8)
    good=_component_metrics(centered,20,20);bad=_component_metrics(shifted,20,20)
    assert good["center_annulus_contrast"] > bad["center_annulus_contrast"]
    assert good["peak_offset_px"] < bad["peak_offset_px"]
    assert np.isfinite(good["effective_radius_px"])


def test_bh_is_bounded_and_monotone_by_sorted_p():
    p=[.04,.001,.2,.02];q=_bh(p)
    assert all(0<=x<=1 for x in q)
    order=np.argsort(p)
    assert all(q[order[i]]<=q[order[i+1]] for i in range(len(order)-1))
