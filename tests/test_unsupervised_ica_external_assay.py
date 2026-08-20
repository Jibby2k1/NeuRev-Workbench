import numpy as np
from neurobench.experiments.unsupervised_ica_eval.external_assay import _event_score, _operator


def test_frozen_operator_orientation_and_event_score():
    raw=np.asarray([0.,1.,3.,2.,2.]); fit={"mean":[0.,0.],"effective_directions":[[1.,1.],[-1.,1.]]}
    out=_operator(raw,fit,1)
    assert np.allclose(out,np.diff(raw))
    assert _event_score(out,0,4)==2
