import numpy as np
import pytest

from neurobench.algorithms.local_covariance_whitening import (
    CovarianceEstimatorConfig,
    SpatialTileConfig,
    WhiteningFeatureBank,
    apply_tiled_feature_whiteners,
    fit_tiled_feature_whiteners,
)
from neurobench.algorithms.local_covariance_whitening_cuda import (
    apply_tiled_feature_whiteners_cuda,
    estimate_cuda_apply_peak_bytes,
)


def _cuda_available() -> bool:
    try:
        import cupy as cp
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def _fixture() -> tuple[WhiteningFeatureBank, np.ndarray, SpatialTileConfig]:
    rng = np.random.default_rng(20260817)
    values = rng.normal(size=(14, 10, 12, 3)).astype(np.float32)
    values[..., 1] += .6 * values[..., 0]
    bank = WhiteningFeatureBank(
        ("a", "b", "c"), values, np.ones(14, dtype=bool),
        tuple({"scientific_array": "signed_msln"} for _ in range(3)),
        {"feature_order_frozen": True},
    )
    quiet = np.zeros(14, dtype=bool); quiet[:8] = True
    tile = SpatialTileConfig(8, 8, 4, 4, "hann", "crop", 32, 1000, 1, 1)
    return bank, quiet, tile


def test_cuda_peak_estimate_is_positive_and_chunk_bounded() -> None:
    tile = SpatialTileConfig(8, 8, 4, 4)
    small = estimate_cuda_apply_peak_bytes((20, 16, 16, 3), tile, frame_chunk=2)
    large = estimate_cuda_apply_peak_bytes((20, 16, 16, 3), tile, frame_chunk=4)
    assert 0 < small < large


@pytest.mark.skipif(not _cuda_available(), reason="CUDA/CuPy unavailable")
def test_cuda_application_matches_cpu_reference() -> None:
    bank, quiet, tile = _fixture()
    fits = fit_tiled_feature_whiteners(bank, quiet, tile, CovarianceEstimatorConfig(method="oas"))
    calibration = np.zeros(14, dtype=bool); calibration[8:11] = True
    cpu = apply_tiled_feature_whiteners(bank, fits, tile, calibration, frame_chunk=3)
    cuda = apply_tiled_feature_whiteners_cuda(
        bank, fits, tile, calibration, frame_chunk=3, max_vram_bytes=512 * 2**20,
    )
    np.testing.assert_allclose(cuda.zca_features, cpu.zca_features, atol=2e-5)
    np.testing.assert_allclose(cuda.mahalanobis_energy, cpu.mahalanobis_energy, atol=1e-4)
    np.testing.assert_allclose(cuda.quiet_surprise, cpu.quiet_surprise, atol=1e-4)
    np.testing.assert_array_equal(cuda.unresolved_tile_mask, cpu.unresolved_tile_mask)
    assert cuda.diagnostics["application_backend"] == "cupy_cuda"
    assert cuda.diagnostics["observed_peak_vram_bytes"] <= 512 * 2**20


@pytest.mark.skipif(not _cuda_available(), reason="CUDA/CuPy unavailable")
def test_cuda_refuses_vram_cap_before_allocation() -> None:
    bank, quiet, tile = _fixture()
    fits = fit_tiled_feature_whiteners(bank, quiet, tile, CovarianceEstimatorConfig(method="oas"))
    with pytest.raises(MemoryError, match="estimate"):
        apply_tiled_feature_whiteners_cuda(
            bank, fits, tile, quiet, frame_chunk=3, max_vram_bytes=1,
        )
