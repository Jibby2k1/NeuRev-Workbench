import numpy as np
import cupy as cp
from neurobench.experiments.msln_msica.quantized_coactivity import _causal_support

def test_causal_support_uses_no_future_frames():
    x=np.zeros((5,1,1),dtype=np.float32); x[3]=1
    result=cp.asnumpy(_causal_support(cp.asarray(x),3))
    np.testing.assert_allclose(result[:,0,0],[0,0,0,1/3,1/3])
