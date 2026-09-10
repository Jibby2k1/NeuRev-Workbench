from __future__ import annotations

import numpy as np
import pytest
import torch

from neurobench.algorithms.gamma_local_standardization import (
    GammaReferenceSpec,
    gamma_cfar,
    gamma_local_standardization,
    gamma_reference_kernel,
)


def test_legacy_exact_matches_archived_kernel_and_cfar() -> None:
    from core.detection import CFAR
    from core.filters import generate_gamma_kernel

    rng = np.random.default_rng(17)
    values = torch.from_numpy(
        (900.0 + rng.normal(0.0, 30.0, size=(5, 25, 27))).astype(np.float32)
    )
    values[2, 12, 13] += 280.0
    spec = GammaReferenceSpec.legacy_exact()

    expected_kernel = generate_gamma_kernel(2, 9.0, 8.0 / 35.0, (23, 23))
    actual_kernel = gamma_reference_kernel(spec)
    torch.testing.assert_close(actual_kernel, expected_kernel, rtol=0.0, atol=0.0)

    expected_z, expected_mask, expected_std, expected_mean = CFAR(
        {
            "T": 3.0,
            "gamma_radial_params": (9.0, 35.0),
            "kernel_size": (23, 23),
            "eps": 64.0,
        }
    )(values.unsqueeze(1))
    actual = gamma_cfar(values, spec, threshold_z=3.0)

    torch.testing.assert_close(actual.score, expected_z.squeeze(1), rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.local_mean, expected_mean.squeeze(1), rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.local_std, expected_std.squeeze(1), rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.mask, expected_mask.squeeze(1).bool(), rtol=0.0, atol=0.0)
    tied_threshold = float(actual.score[0, 0, 0])
    tied = gamma_cfar(values, spec, threshold_z=tied_threshold)
    assert not bool(tied.mask[0, 0, 0])
    assert spec.support_geometry == "square"
    assert spec.guard_radius_px == 0.0
    assert spec.nominal_mode_radius_px == pytest.approx(35.0)


def test_guarded_disk_kernel_has_radial_gamma_weights_not_a_square_annulus() -> None:
    nine = 9
    spec = GammaReferenceSpec.from_mode(
        "guarded_gamma_test",
        support_width_px=nine,
        shape_n=5.0,
        mode_radius_px=3.0,
        guard_radius_px=1.0,
        support_geometry="disk",
    )
    kernel = gamma_reference_kernel(spec, dtype=torch.float64)
    half = nine // 2
    yy, xx = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float64),
        torch.arange(-half, half + 1, dtype=torch.float64),
        indexing="ij",
    )
    radius = torch.sqrt(xx.square() + yy.square())

    assert torch.count_nonzero(kernel[radius <= 1.0]) == 0
    assert torch.count_nonzero(kernel[radius > half]) == 0
    assert float(kernel.sum()) == pytest.approx(1.0, abs=1e-12)
    positive = kernel[kernel > 0]
    assert torch.unique(positive).numel() > 3
    assert spec.boundary_mode == "valid_renormalized_zero"


def test_modern_border_uses_valid_weight_renormalization_without_reflection() -> None:
    spec = GammaReferenceSpec.from_mode(
        "border_gamma_test",
        support_width_px=7,
        shape_n=3.0,
        mode_radius_px=2.5,
        guard_radius_px=1.0,
        support_geometry="disk",
        epsilon=0.25,
    )
    constant = torch.full((1, 9, 11), 7.0, dtype=torch.float32)
    constant_result = gamma_local_standardization(constant, spec)
    torch.testing.assert_close(
        constant_result.local_mean,
        constant,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        constant_result.values,
        torch.zeros_like(constant),
        rtol=0.0,
        atol=2e-5,
    )

    corner_impulse = torch.zeros_like(constant)
    corner_impulse[0, 0, 0] = 10.0
    impulse_result = gamma_local_standardization(corner_impulse, spec)
    assert impulse_result.local_mean[0, 0, 0] == 0.0
    assert impulse_result.values[0, 0, 0] > 0.0
    assert impulse_result.diagnostics["edge_reference_renormalized"] is True


def test_signed_scores_are_preserved_and_cfar_is_one_sided() -> None:
    spec = GammaReferenceSpec.from_mode(
        "signed_gamma_test",
        support_width_px=7,
        shape_n=3.0,
        mode_radius_px=2.5,
        guard_radius_px=1.0,
        support_geometry="disk",
        epsilon=0.5,
    )
    values = torch.zeros((2, 13, 15), dtype=torch.float32)
    values[0, 6, 7] = 5.0
    values[1, 6, 7] = -5.0

    standardized = gamma_local_standardization(values, spec)
    detected = gamma_cfar(values, spec, threshold_z=3.0)

    assert standardized.values[0, 6, 7] > 0
    assert standardized.values[1, 6, 7] < 0
    assert bool(detected.mask[0, 6, 7])
    assert not bool(detected.mask[1, 6, 7])
    assert standardized.diagnostics["signed_output"] is True
    assert detected.diagnostics["probability_of_false_alarm_claimed"] is False


def test_constant_field_is_finite_zero() -> None:
    spec = GammaReferenceSpec.from_mode(
        "constant_gamma_test",
        support_width_px=9,
        shape_n=4.0,
        mode_radius_px=3.0,
        guard_radius_px=1.0,
        scale_floor=0.25,
    )
    result = gamma_local_standardization(
        torch.full((4, 17, 19), -7.0, dtype=torch.float32), spec
    )
    assert bool(torch.isfinite(result.values).all())
    torch.testing.assert_close(
        result.values, torch.zeros_like(result.values), atol=2e-5, rtol=0
    )


def test_chunked_and_full_results_match_and_remain_on_input_device() -> None:
    generator = torch.Generator().manual_seed(23)
    values = torch.randn((11, 19, 21), generator=generator, dtype=torch.float32)
    spec = GammaReferenceSpec.from_mode(
        "chunk_gamma_test",
        support_width_px=7,
        shape_n=3.0,
        mode_radius_px=2.0,
        guard_radius_px=1.0,
        support_geometry="disk",
        epsilon=1e-4,
        scale_floor=1e-3,
    )
    full = gamma_local_standardization(values, spec)
    chunked = gamma_local_standardization(values, spec, chunk_frames=3)

    torch.testing.assert_close(chunked.values, full.values, rtol=0.0, atol=0.0)
    torch.testing.assert_close(chunked.local_mean, full.local_mean, rtol=0.0, atol=0.0)
    torch.testing.assert_close(chunked.local_std, full.local_std, rtol=0.0, atol=0.0)
    assert chunked.values.device == values.device
    assert chunked.reference_kernel.device == values.device
    assert chunked.local_mean.device == values.device
    assert chunked.local_std.device == values.device


def test_statistics_can_be_omitted_without_changing_scores() -> None:
    values = torch.arange(5 * 13 * 13, dtype=torch.float32).reshape(5, 13, 13)
    spec = GammaReferenceSpec.from_mode(
        "compact_gamma_test",
        support_width_px=5,
        shape_n=3.0,
        mode_radius_px=1.5,
        guard_radius_px=0.0,
    )
    full = gamma_local_standardization(values, spec)
    compact = gamma_local_standardization(values, spec, return_statistics=False)
    torch.testing.assert_close(compact.values, full.values, rtol=0.0, atol=0.0)
    assert compact.local_mean is None
    assert compact.local_std is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu_and_retains_device_outputs() -> None:
    generator = torch.Generator().manual_seed(29)
    values = torch.randn((7, 31, 33), generator=generator, dtype=torch.float32)
    spec = GammaReferenceSpec.from_mode(
        "cuda_gamma_test",
        support_width_px=11,
        shape_n=5.0,
        mode_radius_px=4.0,
        guard_radius_px=2.0,
        support_geometry="disk",
        epsilon=1e-5,
        scale_floor=1e-4,
    )
    cpu = gamma_local_standardization(values, spec, chunk_frames=3)
    gpu = gamma_local_standardization(values.cuda(), spec, chunk_frames=3)

    assert gpu.values.is_cuda
    assert gpu.reference_kernel.is_cuda
    assert gpu.local_mean.is_cuda
    assert gpu.local_std.is_cuda
    torch.testing.assert_close(gpu.values.cpu(), cpu.values, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(gpu.local_mean.cpu(), cpu.local_mean, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(gpu.local_std.cpu(), cpu.local_std, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "builder, message",
    [
        (
            lambda: GammaReferenceSpec(
                "bad_width", 4, 3.0, 1.0, guard_radius_px=0.0
            ),
            "support_width_px",
        ),
        (
            lambda: GammaReferenceSpec.from_mode(
                "bad_mode", support_width_px=7, shape_n=1.0, mode_radius_px=2.0
            ),
            "shape_n",
        ),
        (
            lambda: GammaReferenceSpec.from_mode(
                "bad_guard",
                support_width_px=7,
                shape_n=3.0,
                mode_radius_px=2.0,
                guard_radius_px=3.0,
            ),
            "guard_radius_px",
        ),
        (
            lambda: GammaReferenceSpec.from_mode(
                "bad_reflect",
                support_width_px=7,
                shape_n=3.0,
                mode_radius_px=2.0,
                guard_radius_px=1.0,
                boundary_mode="reflect",
            ),
            "reflect is reserved",
        ),
    ],
)
def test_invalid_reference_specs_fail_closed(builder, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        builder()
