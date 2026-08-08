import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from neurobench.algorithms.msln_msica_cuda import (
    apply_per_context_fit_cuda,
    bounded_residual_gate_cuda,
    causal_joint_msln_cuda,
)
from neurobench.algorithms.multiscale_local_normalization import (
    JointSTContext,
    causal_joint_msln,
)
from neurobench.algorithms.multiscale_subspace import fit_per_context_ica
from neurobench.algorithms.pairwise_separation import cs_parzen_objective
from neurobench.experiments.msln_msica.inference import apply_innovation
from neurobench.experiments.msln_msica.routing import bounded_residual_gate


def _cuda_ready() -> bool:
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _cuda_ready(), reason="CUDA unavailable")


def test_cuda_joint_and_gate_match_cpu() -> None:
    rng = np.random.default_rng(4)
    values = (500 + rng.normal(size=(24, 15, 17))).astype(np.float32)
    quiet = np.arange(24) < 12
    context = JointSTContext("joint_cuda_test", 7, 3, 5, 1)
    cpu = causal_joint_msln(values, context, quiet_mask=quiet)
    gpu = causal_joint_msln_cuda(
        values,
        context,
        quiet_mask=quiet,
        review_crop_frames=5,
        max_vram_bytes=512 * 2**20,
    )
    actual = cp.asnumpy(gpu.values)
    np.testing.assert_allclose(actual, cpu.values[5:], rtol=2e-5, atol=2e-5)
    cpu_gate = bounded_residual_gate(cpu.values[5:], beta=0.25, kappa=2)
    gpu_gate = cp.asnumpy(
        bounded_residual_gate_cuda(gpu.values, beta=0.25, kappa=2)
    )
    np.testing.assert_allclose(gpu_gate, cpu_gate, rtol=1e-6, atol=1e-6)


def test_cuda_parzen_and_ica_projection_match_cpu() -> None:
    rng = np.random.default_rng(8)
    pairs = rng.normal(size=(256, 2))
    cpu_objective = cs_parzen_objective(
        pairs, 0.35, block_rows=64, kernel_dtype=np.float32
    )
    gpu_objective = cs_parzen_objective(
        pairs,
        0.35,
        block_rows=64,
        kernel_dtype=np.float32,
        backend="cuda",
    )
    assert gpu_objective.objective == pytest.approx(
        cpu_objective.objective, abs=1e-6
    )
    video = rng.normal(size=(12, 9, 11)).astype(np.float32)
    flat = video.reshape(12, -1)
    fit_pairs = np.column_stack((flat[:-1].ravel(), flat[1:].ravel()))
    fit = fit_per_context_ica(
        "cuda_projection",
        fit_pairs[:512],
        fit_pairs[:1024],
        coarse_step_degrees=15,
        refine_half_width_degrees=1,
        refine_step_degrees=0.5,
        kernel_block_rows=64,
    )
    valid = np.ones(len(video), dtype=bool)
    cpu_p, cpu_i = apply_innovation(video, fit, valid)
    gpu_p, gpu_i = apply_per_context_fit_cuda(cp.asarray(video), fit)
    np.testing.assert_allclose(cp.asnumpy(gpu_p), cpu_p, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(cp.asnumpy(gpu_i), cpu_i, rtol=2e-5, atol=2e-5)


def test_cuda_joint_refuses_vram_oversubscription() -> None:
    values = np.ones((12, 9, 9), dtype=np.float32)
    context = JointSTContext("joint_cuda_cap", 5, 1, 3, 1)
    with pytest.raises(MemoryError):
        causal_joint_msln_cuda(
            values,
            context,
            quiet_mask=np.arange(12) < 6,
            review_crop_frames=3,
            max_vram_bytes=1,
        )


def test_cuda_joint_accepts_valid_fixed_scale_floor() -> None:
    rng = np.random.default_rng(19)
    values = rng.normal(size=(14, 11, 13)).astype(np.float32)
    quiet = np.ones(14, dtype=bool)
    context = JointSTContext("joint_cuda_fixed_floor", 5, 1, 5, 1)
    calibrated = causal_joint_msln_cuda(
        values, context, quiet_mask=quiet, review_crop_frames=5,
        max_vram_bytes=512 * 2**20,
    )
    fixed = causal_joint_msln_cuda(
        values, context, quiet_mask=np.zeros(14, dtype=bool),
        review_crop_frames=5, max_vram_bytes=512 * 2**20,
        scale_floor_override=calibrated.scale_floor,
    )
    assert fixed.scale_floor == pytest.approx(calibrated.scale_floor)
    assert fixed.diagnostics["scale_floor_source"] == "override"
    assert np.isfinite(cp.asnumpy(fixed.values)).all()
