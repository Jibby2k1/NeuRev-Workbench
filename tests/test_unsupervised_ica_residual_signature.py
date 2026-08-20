import numpy as np
from neurobench.experiments.unsupervised_ica_eval.residual_signature_assay import _peak_amplitude


def test_peak_amplitude_preserves_intensity_units():
    trace=np.asarray([10,10,10,10,13,18,12,10],float)
    assert _peak_amplitude(trace,4,3,3)==8
