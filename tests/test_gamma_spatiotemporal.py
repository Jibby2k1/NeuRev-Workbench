"""Small causal/convolution and profile checks; no real recording or GPU run."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from neurobench.algorithms.gamma_spatiotemporal import (
    GammaSTKernels, GammaSTSpec, apply_chunks, build_kernels, describe,
    fft_working_memory_estimate, iter_chunks, spec_grid, support_convergence,
)


def test_exact_21_configuration_matrix():
    specs = spec_grid()
    assert len(specs) == len({spec.spec_id for spec in specs}) == 21
    assert {(s.radius_px, s.time_scale_ms) for s in specs[:12]} == {
        (r, t) for r in (5, 7.5, 10) for t in (0, 20, 60, 200)}
    assert all(s.target_sigma_px == 1 and s.reference_n == 9 for s in specs[:12])
    assert specs[12] == GammaSTSpec(time_scale_ms=60, construction="separable")
    assert {(s.target_sigma_px, s.reference_n, s.time_scale_ms) for s in specs[13:]} == {
        (sigma, n, time) for sigma, n in ((2, 9), (4, 9), (1, 3), (1, 15)) for time in (0, 60)}


@pytest.mark.parametrize("parameters", [dict(radius_px=0), dict(time_scale_ms=-1),
    dict(target_sigma_px=float("nan")), dict(reference_n=1), dict(dt_ms=0),
    dict(construction="separable"), dict(construction="serial")])
def test_invalid_operator_spec_rejected(parameters):
    with pytest.raises(ValueError):
        GammaSTSpec(**parameters)


@pytest.fixture(scope="module")
def small_kernels():
    return build_kernels(GammaSTSpec(radius_px=2, time_scale_ms=20, target_sigma_px=.5, reference_n=3))


def test_profiles_use_approved_distance_and_recover_2d_conventions():
    from neurobench.algorithms.gamma_two_stencil import match_centered_gamma_mu
    spatial = build_kernels(GammaSTSpec())
    meta = describe(spatial)
    assert spatial.target.shape[0] == spatial.reference.shape[0] == 1
    assert meta["maximum_history_ms"] == 0
    assert meta["target_mu_per_px"] == match_centered_gamma_mu(31)
    half = spatial.target.shape[1]//2
    mu = meta["target_mu_per_px"]
    assert spatial.target[0, half, half+1]/spatial.target[0, half, half] == pytest.approx(math.exp(-mu))
    assert spatial.reference[0, half, half] == 0  # Profile zero, no explicit guard.
    expected = (2**8 * math.exp(-8*2/7.5))/(math.exp(-8/7.5))
    assert spatial.reference[0, half, half+2]/spatial.reference[0, half, half+1] == pytest.approx(expected)
    joint = build_kernels(GammaSTSpec(time_scale_ms=60))
    h = joint.target.shape[1]//2
    q = math.sqrt(1 + (7.5*20/60)**2)
    assert joint.target[1, h, h+1]/joint.target[0, h, h] == pytest.approx(math.exp(-mu*q))
    assert joint.reference[1, h, h] > 0  # Own past location is included.
    assert meta["ideal_continuous_reference_shell_shape"] == 10
    assert joint.metadata["ideal_continuous_reference_shell_shape"] == 11
    assert joint.metadata["per_voxel_reference_exponent"] == 8


@pytest.mark.parametrize("spec", [GammaSTSpec(), GammaSTSpec(time_scale_ms=60),
    GammaSTSpec(radius_px=10, time_scale_ms=200), GammaSTSpec(target_sigma_px=4, time_scale_ms=60),
    GammaSTSpec(reference_n=3, time_scale_ms=60), GammaSTSpec(reference_n=15, time_scale_ms=60)])
def test_support_retention_normalization_actual_marginals_and_metadata(spec):
    kernels = build_kernels(spec); metadata = describe(kernels)
    assert min(metadata["ideal_continuous_mass_lower_bound"].values()) >= .99
    assert min(metadata["discrete_retained_mass_estimate"].values()) >= .99
    assert max(metadata["relative_convergence_mass_gap"].values()) <= 1e-7
    for name in ("target", "reference"):
        kernel = getattr(kernels, name); stats = metadata[name]
        assert kernel.sum() == pytest.approx(1, abs=2e-14)
        assert not kernel.flags.writeable and np.all(kernel >= 0)
        np.testing.assert_allclose(kernel.sum(axis=(1, 2)), stats["lag_marginal"], atol=2e-15)
        np.testing.assert_allclose(kernel.sum(axis=0), stats["spatial_marginal"], atol=2e-15)
        np.testing.assert_allclose(kernel, kernel[:, ::-1, :], atol=2e-15)
        np.testing.assert_allclose(kernel, kernel[:, :, ::-1], atol=2e-15)
    json.dumps(metadata, allow_nan=False)
    metadata["spec"]["radius_px"] = -10
    assert kernels.metadata["spec"]["radius_px"] == spec.radius_px


def test_separable_control_preserves_each_joint_marginal():
    joint = build_kernels(GammaSTSpec(time_scale_ms=60))
    control = build_kernels(GammaSTSpec(time_scale_ms=60, construction="separable"))
    assert joint.target.shape == control.target.shape
    for name in ("target", "reference"):
        a, b = getattr(joint, name), getattr(control, name)
        np.testing.assert_allclose(a.sum(axis=0), b.sum(axis=0), rtol=0, atol=2e-15)
        np.testing.assert_allclose(a.sum(axis=(1, 2)), b.sum(axis=(1, 2)), rtol=0, atol=2e-15)
        assert not np.allclose(a, b, rtol=1e-4, atol=1e-12)


def _direct_oracle(values, kernels):
    result = {name: np.empty_like(values, dtype=np.float64) for name in ("A", "M", "variance")}
    nt, height, width = values.shape
    for t in range(nt):
        for y in range(height):
            for x in range(width):
                answers = []
                for kernel in (kernels.target, kernels.reference):
                    total = mean = second = 0.0
                    hy, hx = kernel.shape[1]//2, kernel.shape[2]//2
                    for lag in range(min(t+1, len(kernel))):
                        for ky in range(kernel.shape[1]):
                            for kx in range(kernel.shape[2]):
                                yy, xx = y+hy-ky, x+hx-kx
                                if 0 <= yy < height and 0 <= xx < width:
                                    w = kernel[lag, ky, kx]; value = values[t-lag, yy, xx]
                                    total += w; mean += w*value; second += w*value*value
                    answers.append((mean/total, second/total))
                result["A"][t, y, x] = answers[0][0]
                result["M"][t, y, x] = answers[1][0]
                result["variance"][t, y, x] = max(0, answers[1][1]-answers[1][0]**2)
    return result


def test_fft_matches_direct_causal_moments_at_all_boundaries():
    rng = np.random.default_rng(405)
    target = np.arange(1, 28, dtype=float).reshape(3, 3, 3); target /= target.sum()
    reference = target[::-1].copy(); reference /= reference.sum()
    kernels = GammaSTKernels(GammaSTSpec(time_scale_ms=20), target, reference, {})
    values = rng.normal(size=(4, 3, 4))
    expected = _direct_oracle(values, kernels)
    actual = apply_chunks(values, kernels, chunk_frames=2)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=3e-12, atol=3e-12)


def test_constant_field_at_spatial_and_missing_past_boundaries(small_kernels):
    result = apply_chunks(np.full((6, 9, 11), 4.25), small_kernels, chunk_frames=2)
    np.testing.assert_allclose(result["A"], 4.25, rtol=0, atol=3e-11)
    np.testing.assert_allclose(result["M"], 4.25, rtol=0, atol=3e-11)
    np.testing.assert_allclose(result["variance"], 0, rtol=0, atol=3e-10)


def test_full_versus_chunked_and_future_perturbation(small_kernels):
    values = np.random.default_rng(803).normal(size=(9, 9, 11))
    full = apply_chunks(values, small_kernels, chunk_frames=9)
    for count in (1, 2, 4):
        chunks = apply_chunks(values, small_kernels, chunk_frames=count)
        for name in full:
            np.testing.assert_allclose(full[name], chunks[name], rtol=2e-11, atol=2e-11)
    changed = values.copy(); changed[5:] += 20
    perturbed = apply_chunks(changed, small_kernels, chunk_frames=9)
    for name in full:
        np.testing.assert_allclose(full[name][:5], perturbed[name][:5], rtol=2e-10, atol=2e-10)


def test_impulse_uses_current_and_past_lag_with_correct_spatial_orientation():
    target = np.zeros((3, 3, 3)); target[0, 1, 1] = .5; target[2, 1, 2] = .5
    reference = np.ones((3, 3, 3))/27
    kernels = GammaSTKernels(GammaSTSpec(time_scale_ms=20), target, reference, {})
    values = np.zeros((7, 9, 9)); values[2, 4, 4] = 1
    result = apply_chunks(values, kernels, chunk_frames=2)["A"]
    assert result[2, 4, 4] == pytest.approx(.5, abs=1e-14)
    assert result[4, 4, 5] == pytest.approx(.5, abs=1e-14)
    np.testing.assert_allclose(result[:2], 0, atol=1e-14)
    assert result.sum() == pytest.approx(1, abs=2e-14)


def test_memmap_outputs_and_input_alias_rejection(tmp_path, small_kernels):
    values = np.random.default_rng(8).normal(size=(4, 7, 8))
    output = {name: np.lib.format.open_memmap(tmp_path/f"{name}.npy", mode="w+", shape=values.shape, dtype="float64")
              for name in ("A", "M", "variance")}
    result = apply_chunks(values, small_kernels, chunk_frames=2, output=output)
    assert all(result[name] is output[name] for name in output)
    assert list((start, stop) for start, stop, _ in iter_chunks(values, small_kernels, chunk_frames=3)) == [(0, 3), (3, 4)]
    with pytest.raises(ValueError, match="overwrite source"):
        apply_chunks(values, small_kernels, output={**output, "A": values})
    with pytest.raises(ValueError, match="must not overlap"):
        apply_chunks(values, small_kernels, output={**output, "A": output["M"]})


def test_support_convergence_reports_changes_in_realized_width_and_lag():
    rows = support_convergence(GammaSTSpec(radius_px=2, time_scale_ms=20, target_sigma_px=.5, reference_n=3))
    assert [row["retained_mass"] for row in rows] == [.99, .999, .9999]
    assert rows[-1]["support_shape"][1] >= rows[0]["support_shape"][1]
    assert abs(rows[-1]["reference_sigma_x_px"]-rows[-2]["reference_sigma_x_px"]) < .02
    assert abs(rows[-1]["reference_mean_lag_ms"]-rows[-2]["reference_mean_lag_ms"]) < .2


def test_fft_memory_budget_fails_before_allocating_transforms(small_kernels):
    values = np.zeros((4, 7, 8))
    estimate = fft_working_memory_estimate(values.shape, small_kernels.target.shape, chunk_frames=2)
    assert estimate["conservative_fft_bytes"] > 0
    with pytest.raises(MemoryError, match="byte budget"):
        apply_chunks(values, small_kernels, chunk_frames=2, max_fft_bytes=1)


def test_tiny_cuda_matches_cpu_direct_moments_and_causality(small_kernels):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable in this process namespace")
    values = np.random.default_rng(918).normal(size=(7, 9, 11))
    cpu = apply_chunks(values, small_kernels, chunk_frames=2)
    gpu = apply_chunks(values, small_kernels, chunk_frames=3, device="cuda")
    for name in cpu:
        np.testing.assert_allclose(gpu[name], cpu[name], rtol=2e-10, atol=2e-10)
    changed = values.copy(); changed[4:] += 20
    future = apply_chunks(changed, small_kernels, chunk_frames=7, device="cuda")
    for name in cpu:
        np.testing.assert_allclose(future[name][:4], cpu[name][:4], rtol=2e-10, atol=2e-10)
