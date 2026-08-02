import numpy as np
import pytest

from neurobench.algorithms.information_source_separation import normalized_hsic
from neurobench.algorithms.information_source_separation_cuda import normalized_hsic_cuda


def test_cuda_hsic_matches_cpu_definition() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(17)
    left = rng.normal(size=96)
    right = 0.3 * left**2 + rng.normal(size=96)
    assert normalized_hsic_cuda(left, right) == pytest.approx(
        normalized_hsic(left, right), abs=1e-10
    )
