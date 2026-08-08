import numpy as np
import pytest

from neurobench.reports.zero_anchored_display import (
    global_positive_max,
    zero_anchored_exp2,
    zero_anchored_signed,
)


def test_zero_anchored_signed_uses_black_zero_and_fixed_global_max() -> None:
    values = np.asarray([-3.0, 0.0, 1.0, 4.0, 8.0], dtype=np.float32)
    result = zero_anchored_signed(values, positive_max=4.0)
    np.testing.assert_allclose(result, [0.0, 0.0, 0.25, 1.0, 1.0])
    assert result.dtype == np.float32


def test_zero_anchored_exp2_preserves_positive_order_for_each_alpha() -> None:
    values = np.asarray([-2.0, 0.0, 0.25, 1.0, 2.0], dtype=np.float32)
    for alpha in (0.05, 0.1, 0.25, 0.5, 1.0):
        result = zero_anchored_exp2(values, alpha=alpha, positive_max=2.0)
        assert result[0] == result[1] == 0.0
        assert np.all(np.diff(result[1:]) > 0.0)
        assert result[-1] == pytest.approx(1.0)
        assert result.dtype == np.float32


def test_global_positive_max_and_invalid_contract() -> None:
    assert global_positive_max(np.asarray([-1.0, 0.0, 3.0])) == 3.0
    with pytest.raises(ValueError):
        global_positive_max(np.asarray([-1.0, 0.0]))
    with pytest.raises(ValueError):
        zero_anchored_exp2(np.ones(2), alpha=0.0, positive_max=1.0)
