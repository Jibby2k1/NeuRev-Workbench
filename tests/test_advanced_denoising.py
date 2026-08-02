import numpy as np

from neurobench.algorithms.advanced_denoising import (
    bounded_noise_subtraction,
    carrier_blend,
    multiscale_group_shrinkage,
    noise_psd_wiener,
    nonlocal_means_spatial,
    undecimated_spatial_group_shrinkage,
    windowed_nonnegative_factorization,
    windowed_robust_low_rank_sparse,
)


def _movie() -> np.ndarray:
    rng = np.random.default_rng(19)
    values = 0.25 * rng.normal(size=(24, 16, 18))
    yy, xx = np.ogrid[:16, :18]
    blob = np.exp(-((xx - 9) ** 2 + (yy - 8) ** 2) / 5)
    values[12:18] += np.linspace(0.2, 2, 6)[:, None, None] * blob
    return values.astype(np.float32)


def test_carrier_blend_endpoints_and_bounded_correction() -> None:
    source = _movie()
    estimate = source * 0.1
    np.testing.assert_allclose(carrier_blend(source, estimate, 0), source)
    np.testing.assert_allclose(
        carrier_blend(source, estimate, 1), estimate, atol=1e-7
    )
    bounded = bounded_noise_subtraction(
        source, estimate, alpha=1, correction_limit_z=0.2
    )
    assert np.max(np.abs(bounded - source)) <= 0.200001


def test_multiscale_group_shrinkage_is_finite_and_carrier_preserving() -> None:
    source = _movie()
    output = multiscale_group_shrinkage(
        source,
        [source * 0.8, source * 0.6],
        lambda_z=1,
        alpha=0.5,
        gain_floor=0.25,
    )
    assert output.shape == source.shape
    assert np.isfinite(output).all()
    assert np.linalg.norm(output - source) < np.linalg.norm(source)


def test_psd_wiener_and_wavelet_outputs_align() -> None:
    source = _movie()
    wiener = noise_psd_wiener(
        source,
        quiet_count=8,
        noise_multiplier=1,
        frequency_smoothing_sigma=1,
        alpha=0.5,
    )
    wavelet = undecimated_spatial_group_shrinkage(
        source,
        levels=2,
        threshold_z=1,
        group_sigma_px=1,
        coarse_keep=0.5,
    )
    for result in (wiener, wavelet):
        assert result.shape == source.shape
        assert np.isfinite(result).all()


def test_windowed_factorizations_are_bounded_and_reproducible() -> None:
    source = _movie()
    robust = windowed_robust_low_rank_sparse(
        source,
        window_frames=8,
        rank=2,
        sparse_lambda_z=0.5,
        alpha=0.5,
        device="cpu",
    )
    first = windowed_nonnegative_factorization(
        source,
        window_frames=8,
        rank=2,
        iterations=4,
        alpha=0.5,
        seed=3,
        device="cpu",
    )
    second = windowed_nonnegative_factorization(
        source,
        window_frames=8,
        rank=2,
        iterations=4,
        alpha=0.5,
        seed=3,
        device="cpu",
    )
    assert np.isfinite(robust).all()
    np.testing.assert_allclose(first, second)


def test_nonlocal_means_suppresses_quiet_energy_without_shape_change() -> None:
    source = _movie()
    output = nonlocal_means_spatial(
        source,
        search_radius=1,
        patch_size=3,
        bandwidth_z=1,
        alpha=1,
        device="cpu",
        frame_batch_size=4,
    )
    assert output.shape == source.shape
    assert np.isfinite(output).all()
    assert np.mean(output[:8] ** 2) <= np.mean(source[:8] ** 2)
