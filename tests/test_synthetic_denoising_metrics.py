import numpy as np

from neurobench.metrics.synthetic_denoising import (
    localized_synthetic_denoising_metrics,
)


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    truth = np.zeros((32, 16, 16), dtype=np.float32)
    for index, (y, x) in enumerate(((4, 4), (4, 12), (12, 4), (12, 12))):
        trace = np.zeros(32, dtype=np.float32)
        trace[4 + 4 * index : 12 + 4 * index] = np.hanning(8)
        truth[:, y, x] = trace
    rng = np.random.default_rng(8)
    observed = truth + 0.1 * rng.normal(size=truth.shape)
    return observed.astype(np.float32), truth


def test_truth_is_perfect_and_noisy_input_is_finite() -> None:
    observed, truth = _fixture()
    perfect = localized_synthetic_denoising_metrics(truth, observed, truth)
    noisy = localized_synthetic_denoising_metrics(observed, observed, truth)
    assert perfect["synthetic_correlation"] > 0.999
    assert perfect["synthetic_peak_frame_error"] == 0
    assert perfect["synthetic_nmse"] == 0
    assert np.isfinite(noisy["synthetic_correlation"])
    assert noisy["synthetic_input_nmse"] == noisy["synthetic_nmse"]


def test_local_metric_does_not_report_false_shift_from_remote_noise() -> None:
    observed, truth = _fixture()
    estimate = truth.copy()
    estimate[31, 7, 7] = 100
    result = localized_synthetic_denoising_metrics(estimate, observed, truth)
    assert result["synthetic_peak_frame_error"] == 0
