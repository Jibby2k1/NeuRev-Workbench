"""Independent numerical checks for the new two-stencil scientific operator."""
from __future__ import annotations

from dataclasses import replace
import itertools

import numpy as np
import pytest
import torch

from neurobench.algorithms.gamma_two_stencil import (
    TwoStencilSpec,
    build_two_stencil_factorial,
    build_two_stencil_kernels,
    deployed_anchor_cell,
    match_centered_gamma_mu,
    two_stencil_local_standardization,
)


def _tiny_spec(**overrides) -> TwoStencilSpec:
    return TwoStencilSpec(**{
        "target_width_px": 5,
        "target_mu": 1.7,
        "reference_width_px": 5,
        "reference_shape_n": 3.0,
        "reference_mode_radius_px": 1.5,
        "scale_floor": 0.2,
        **overrides,
    })


def _numpy_kernels(spec: TwoStencilSpec) -> tuple[np.ndarray, np.ndarray]:
    def radii(width):
        axis = np.arange(-(width // 2), width // 2 + 1, dtype=np.float64)
        y, x = np.meshgrid(axis, axis, indexing="ij")
        return np.hypot(x, y)

    if spec.design == "point":
        target = np.ones((1, 1))
    else:
        target = np.exp(-spec.target_mu * radii(spec.target_width_px))
        target /= target.sum()
    r = radii(spec.reference_width_px)
    reference = np.ones_like(r)
    if spec.reference_family == "gamma":
        reference = r ** (spec.reference_shape_n - 1) * np.exp(-spec.reference_mu * r)
    if spec.guard_radius_px > 0:
        reference[r <= spec.guard_radius_px] = 0
    if spec.support_geometry == "disk":
        reference[r > spec.reference_width_px // 2] = 0
    reference /= reference.sum()
    return target, reference


def _brute_valid_average(video: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Explicit finite-domain weighted sums, without convolution/padding APIs."""
    output = np.empty_like(video, dtype=np.float64)
    half = kernel.shape[0] // 2
    for t, y, x in np.ndindex(video.shape):
        value, mass = 0.0, 0.0
        for ky, kx in np.ndindex(kernel.shape):
            sy, sx = y + ky - half, x + kx - half
            if 0 <= sy < video.shape[1] and 0 <= sx < video.shape[2]:
                mass += kernel[ky, kx]
                value += kernel[ky, kx] * video[t, sy, sx]
        assert mass > 0
        output[t, y, x] = value / mass
    return output


@pytest.mark.parametrize("design,family,guard,geometry", itertools.product(
    ("point", "direct", "serial"), ("uniform", "gamma"), (0.0, 1.0), ("square", "disk")
))
def test_operator_matches_independent_pixel_sum_reference(design, family, guard, geometry):
    spec = _tiny_spec(design=design, reference_family=family, guard_radius_px=guard, support_geometry=geometry)
    x = np.random.default_rng(41).normal(size=(2, 4, 6))
    kt, kr = _numpy_kernels(spec)
    a = _brute_valid_average(x, kt)
    domain = a if design == "serial" else x
    mean = _brute_valid_average(domain, kr)
    std = np.sqrt(np.maximum(0, _brute_valid_average(domain ** 2, kr) - mean ** 2))
    expected = (a - mean) / (np.maximum(std, spec.scale_floor) + spec.epsilon)
    actual = two_stencil_local_standardization(x, spec, chunk_frames=1)
    for got, want in ((actual.target_response, a), (actual.reference_mean, mean),
                      (actual.reference_std, std), (actual.contrast, a - mean), (actual.values, expected)):
        np.testing.assert_allclose(got.numpy(), want, rtol=2e-11, atol=2e-11)


def test_center_guard_and_reference_geometry_are_separate():
    for family in ("uniform", "gamma"):
        unguarded = _tiny_spec(reference_family=family)
        target, reference = build_two_stencil_kernels(unguarded, dtype=torch.float64)
        assert float(target[2, 2]) == float(target.max()) > 0
        assert bool(reference[2, 2] > 0) == (family == "uniform")
        assert reference[0, 0] > 0  # no hidden disk cutoff
        _, guarded = build_two_stencil_kernels(replace(unguarded, guard_radius_px=1), dtype=torch.float64)
        assert guarded[2, 2] == guarded[1, 2] == guarded[2, 1] == 0
        assert guarded[1, 1] > 0
        _, disk = build_two_stencil_kernels(replace(unguarded, support_geometry="disk"), dtype=torch.float64)
        assert disk[0, 0] == 0
        torch.testing.assert_close(reference.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_matched_target_second_moment_and_factorial_identity():
    mu = match_centered_gamma_mu()
    axis = np.arange(-4, 5, dtype=np.float64)
    gaussian = np.exp(-axis ** 2 / 2)
    expected_moment = 2 * np.sum(axis ** 2 * gaussian) / gaussian.sum()
    target, _ = build_two_stencil_kernels(TwoStencilSpec(), dtype=torch.float64)
    axis31 = torch.arange(-15, 16, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis31, axis31, indexing="ij")
    assert float((target * (xx ** 2 + yy ** 2)).sum()) == pytest.approx(expected_moment, abs=1e-12)
    assert mu == match_centered_gamma_mu()
    assert float(target[15, 15]) > 0
    cells = build_two_stencil_factorial()
    assert len(cells) == len({cell.cell_id for cell in cells}) == 24
    assert len({(cell.input_representation, cell.spec) for cell in cells}) == 24
    assert all(cell.spec.target_mu == mu for cell in cells)
    assert all(cell.spec.support_geometry == "square" and not cell.is_deployed_anchor for cell in cells)
    with_anchor = build_two_stencil_factorial(include_deployed_anchor=True)
    assert len(with_anchor) == 25
    assert with_anchor[-1] == deployed_anchor_cell()


@pytest.mark.parametrize("design", ("point", "direct", "serial"))
def test_constant_field_stays_zero_including_boundaries(design):
    x = torch.full((2, 5, 6), -4.0, dtype=torch.float64)
    actual = two_stencil_local_standardization(x, _tiny_spec(design=design, guard_radius_px=1))
    torch.testing.assert_close(actual.target_response, x, rtol=0, atol=2e-14)
    torch.testing.assert_close(actual.reference_mean, x, rtol=0, atol=2e-14)
    torch.testing.assert_close(actual.values, torch.zeros_like(x), rtol=0, atol=1e-12)


def test_direct_and_serial_are_equal_for_delta_target_but_not_smoothed_target():
    x = torch.tensor(np.random.default_rng(5).normal(size=(2, 7, 8)), dtype=torch.float64)
    delta = _tiny_spec(target_width_px=1)
    results = [two_stencil_local_standardization(x, replace(delta, design=design)) for design in ("point", "direct", "serial")]
    for other in results[1:]:
        for field in ("values", "target_response", "reference_mean", "reference_std", "contrast"):
            torch.testing.assert_close(getattr(results[0], field), getattr(other, field), rtol=0, atol=0)
    direct = two_stencil_local_standardization(x, _tiny_spec(design="direct"))
    serial = two_stencil_local_standardization(x, _tiny_spec(design="serial"))
    assert not torch.allclose(direct.reference_mean, serial.reference_mean)
    assert not torch.allclose(direct.reference_std, serial.reference_std)


def test_exact_deployed_anchor_matches_frozen_module():
    from neurobench.algorithms.gamma_local_standardization import GammaReferenceSpec, gamma_local_standardization

    x = torch.tensor(np.random.default_rng(51).normal(size=(2, 19, 21)), dtype=torch.float32)
    anchor = replace(deployed_anchor_cell().spec, scale_floor=0.37)
    old_spec = GammaReferenceSpec.from_mode(
        "anchor_parity", support_width_px=31, shape_n=9,
        mode_radius_px=7.5, guard_radius_px=7, scale_floor=0.37,
    )
    expected = gamma_local_standardization(x, old_spec, chunk_frames=1)
    actual = two_stencil_local_standardization(x, anchor, chunk_frames=1)
    for got, want in ((actual.values, expected.values), (actual.reference_kernel, expected.reference_kernel),
                      (actual.reference_mean, expected.local_mean), (actual.reference_std, expected.local_std)):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_temporal_chunking_and_score_only_output():
    x = torch.tensor(np.random.default_rng(2).normal(size=(5, 8, 9)), dtype=torch.float64)
    spec = _tiny_spec(design="serial")
    full = two_stencil_local_standardization(x, spec)
    chunked = two_stencil_local_standardization(x, spec, chunk_frames=2)
    score_only = two_stencil_local_standardization(x, spec, chunk_frames=2, return_stages=False)
    for field in ("values", "target_response", "reference_mean", "reference_std", "contrast"):
        torch.testing.assert_close(getattr(chunked, field), getattr(full, field), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(score_only.values, full.values)
    assert score_only.target_response is score_only.reference_mean is score_only.reference_std is score_only.contrast is None


@pytest.mark.parametrize("design", ("direct", "serial"))
def test_spatial_tile_matches_full_frame_only_with_complete_halo(design):
    x = torch.tensor(np.random.default_rng(6).normal(size=(1, 23, 25)), dtype=torch.float64)
    spec = _tiny_spec(design=design, reference_width_px=7)
    full = two_stencil_local_standardization(x, spec)
    halo = full.diagnostics["required_input_halo_px"]
    y0, y1, x0, x1 = 8, 13, 9, 15
    tile = x[:, y0 - halo:y1 + halo, x0 - halo:x1 + halo]
    result = two_stencil_local_standardization(tile, spec)
    torch.testing.assert_close(result.values[:, halo:-halo, halo:-halo], full.values[:, y0:y1, x0:x1], rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("design,scale", itertools.product(("point", "direct", "serial"), (3.0, -3.0)))
def test_signed_affine_behavior_when_floor_and_epsilon_scale_with_units(design, scale):
    x = torch.tensor(np.random.default_rng(9).normal(size=(1, 7, 8)), dtype=torch.float64)
    spec = _tiny_spec(design=design, epsilon=0.03)
    baseline = two_stencil_local_standardization(x, spec)
    converted = two_stencil_local_standardization(
        scale * x + 2.0,
        replace(spec, epsilon=abs(scale) * spec.epsilon, scale_floor=abs(scale) * spec.scale_floor),
    )
    torch.testing.assert_close(converted.values, np.sign(scale) * baseline.values, rtol=2e-11, atol=2e-11)
    # A fixed additive epsilon is deliberately NOT claimed scale invariant.


@pytest.mark.parametrize("kwargs", (
    {"design": "unknown"}, {"reference_family": "box"}, {"support_geometry": "triangle"},
    {"target_width_px": 4}, {"reference_width_px": 2}, {"reference_width_px": True},
    {"guard_radius_px": -1}, {"guard_radius_px": 22}, {"reference_shape_n": 1},
    {"reference_mode_radius_px": 0}, {"target_mu": float("nan")},
    {"epsilon": 0}, {"epsilon": float("inf")}, {"scale_floor": -1},
))
def test_invalid_specs_fail_before_computation(kwargs):
    with pytest.raises(ValueError):
        TwoStencilSpec(**kwargs)


@pytest.mark.parametrize("bad", (np.zeros((3, 4)), np.zeros((0, 3, 4)),
                                  np.full((1, 3, 4), np.nan), np.ones((1, 3, 4), dtype=np.complex128)))
def test_invalid_arrays_are_rejected(bad):
    with pytest.raises(ValueError):
        two_stencil_local_standardization(bad, _tiny_spec())


@pytest.mark.parametrize("chunk", (0, -1, 1.5, True))
def test_invalid_chunk_size_is_rejected(chunk):
    with pytest.raises(ValueError, match="chunk_frames"):
        two_stencil_local_standardization(np.ones((1, 5, 6)), _tiny_spec(), chunk_frames=chunk)


def test_no_valid_reference_mass_fails_instead_of_inventing_background():
    with pytest.raises(ValueError, match="zero valid mass"):
        two_stencil_local_standardization(np.ones((1, 2, 2)), TwoStencilSpec(guard_radius_px=7))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_stages_match_cpu():
    x = torch.tensor(np.random.default_rng(17).normal(size=(2, 9, 11)), dtype=torch.float32)
    spec = _tiny_spec(design="serial", guard_radius_px=1)
    cpu = two_stencil_local_standardization(x, spec, chunk_frames=1)
    gpu = two_stencil_local_standardization(x, spec, device="cuda", chunk_frames=1)
    assert gpu.values.device.type == "cuda"
    for field in ("values", "target_response", "reference_mean", "reference_std", "contrast"):
        torch.testing.assert_close(getattr(gpu, field).cpu(), getattr(cpu, field), rtol=3e-5, atol=3e-5)
