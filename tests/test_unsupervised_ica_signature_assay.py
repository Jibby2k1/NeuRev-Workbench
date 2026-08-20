import numpy as np
from neurobench.experiments.unsupervised_ica_eval.signature_assay import _corr, _fold, _window


def test_window_uses_baseline_scale_without_peak_alignment():
    trace=np.asarray([1,1,1,1,2,4,2,1],float); result=_window(trace,4,6,3)
    assert len(result)==6 and np.argmax(result)==4


def test_fold_and_correlation_are_deterministic():
    assert _fold("roi_010",5)==_fold("roi_010",5)
    assert np.isclose(_corr(np.arange(5),np.arange(5)*2),1)
