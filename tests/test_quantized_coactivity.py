import numpy as np
import pytest

cp = pytest.importorskip("cupy", reason="quantized coactivity is an optional CUDA integration")
from neurobench.experiments.msln_msica.quantized_coactivity import _causal_support

def test_causal_support_uses_no_future_frames():
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy is installed but no CUDA device is available")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CuPy is installed but the CUDA runtime has no usable device")
    x=np.zeros((5,1,1),dtype=np.float32); x[3]=1
    result=cp.asnumpy(_causal_support(cp.asarray(x),3))
    np.testing.assert_allclose(result[:,0,0],[0,0,0,1/3,1/3])
