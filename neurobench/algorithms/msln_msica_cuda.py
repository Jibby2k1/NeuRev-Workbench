"""Resource-bounded CuPy primitives for the MSLN/MS-ICA pipeline."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neurobench.algorithms.multiscale_local_normalization import JointSTContext


@dataclass(frozen=True)
class CUDAJointResult:
    values: Any
    scale_floor: float
    diagnostics: dict[str, Any]


def _cupy() -> Any:
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("CuPy is required for the CUDA backend") from exc
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA device is available")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        raise RuntimeError("CUDA runtime is unavailable") from exc
    return cp


def cuda_device_summary() -> dict[str, Any]:
    cp = _cupy()
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    free, total = cp.cuda.runtime.memGetInfo()
    return {
        "device_id": int(device.id),
        "name": properties["name"].decode(),
        "free_bytes": int(free),
        "total_bytes": int(total),
        "cupy_version": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
    }


def _box_sum(values: Any, width: int) -> Any:
    from cupyx.scipy.ndimage import uniform_filter

    area = int(width) ** 2
    return uniform_filter(
        values,
        size=(1, int(width), int(width)),
        mode="constant",
        cval=0.0,
    ) * area


def _reference_sum(values: Any, window: int, guard: int, crop: int) -> Any:
    cp = _cupy()
    cumulative = cp.concatenate(
        (
            cp.zeros((1, *values.shape[1:]), dtype=cp.float64),
            cp.cumsum(values, axis=0, dtype=cp.float64),
        ),
        axis=0,
    )
    current = cp.arange(int(crop), len(values), dtype=cp.int32)
    result = cumulative[current - int(guard)] - cumulative[current - int(window)]
    del cumulative
    return result


def causal_joint_msln_cuda(
    values: np.ndarray,
    context: JointSTContext,
    *,
    quiet_mask: np.ndarray,
    review_crop_frames: int,
    max_vram_bytes: int,
    scale_floor_override: float | None = None,
) -> CUDAJointResult:
    """Compute a causal joint MSLN review map while retaining it on the GPU."""
    cp = _cupy()
    started = time.monotonic()
    source = np.asarray(values)
    if source.ndim != 3 or not source.size or context.estimator != "mean_std":
        raise ValueError("CUDA joint MSLN requires a non-empty TYX mean/std input")
    crop = int(review_crop_frames)
    window = int(context.temporal_window_frames)
    guard = int(context.temporal_guard_frames)
    if crop < window or crop >= len(source):
        raise ValueError("review crop must include the full causal pre-roll")
    quiet = np.asarray(quiet_mask, dtype=bool)
    if quiet.shape != (len(source),):
        raise ValueError("quiet mask must align with the source frames")
    free_before, _ = cp.cuda.runtime.memGetInfo()
    cap = min(int(max_vram_bytes), int(free_before))
    # Measured implementation peak is below 42 bytes/source element plus
    # 20 bytes/review element. Refuse rather than oversubscribe or spill.
    source_elements = int(np.prod(source.shape))
    review_elements = int(np.prod((len(source) - crop, *source.shape[1:])))
    estimated_peak = source_elements * 42 + review_elements * 20
    if estimated_peak > cap:
        raise MemoryError(
            f"CUDA joint MSLN estimate {estimated_peak} exceeds cap {cap}"
        )

    peak_used = 0

    def track() -> None:
        nonlocal peak_used
        free_now, _ = cp.cuda.runtime.memGetInfo()
        peak_used = max(peak_used, int(free_before - free_now))

    device = cp.asarray(source, dtype=cp.float32)
    if not bool(cp.isfinite(device).all()):
        raise ValueError("source contains non-finite values")
    sample_step = max(1, len(device) // 16)
    offset = float(cp.asnumpy(cp.median(device[::sample_step, ::4, ::4])))
    centered = device - cp.float32(offset)
    del device
    track()

    ones = cp.ones((1, *source.shape[1:]), dtype=cp.float32)
    spatial_count = (
        _box_sum(ones, context.spatial_outer_width_px)
        - _box_sum(ones, context.spatial_guard_width_px)
    )[0]
    del ones
    annulus_sum = (
        _box_sum(centered, context.spatial_outer_width_px)
        - _box_sum(centered, context.spatial_guard_width_px)
    )
    track()
    reference_sum = _reference_sum(annulus_sum, window, guard, crop)
    del annulus_sum
    count = spatial_count * cp.float32(window - guard)
    reference_mean = (reference_sum / count).astype(cp.float32)
    del reference_sum
    cp.get_default_memory_pool().free_all_blocks()
    track()

    centered_square = cp.square(centered, dtype=cp.float32)
    annulus_square_sum = (
        _box_sum(centered_square, context.spatial_outer_width_px)
        - _box_sum(centered_square, context.spatial_guard_width_px)
    )
    del centered_square
    track()
    reference_square_sum = _reference_sum(
        annulus_square_sum, window, guard, crop
    )
    del annulus_square_sum
    second_moment = reference_square_sum / count
    del reference_square_sum, count, spatial_count
    variance = cp.maximum(
        second_moment - cp.square(reference_mean, dtype=cp.float32),
        0.0,
    )
    del second_moment
    scale = cp.sqrt(variance).astype(cp.float32)
    del variance
    numerator = centered[crop:] - reference_mean
    del centered, reference_mean
    track()

    quiet_review = cp.asarray(quiet[crop:])
    quiet_scales = scale[quiet_review]
    positive = quiet_scales[quiet_scales > 0]
    if scale_floor_override is not None:
        fitted_floor = float(scale_floor_override)
        if not np.isfinite(fitted_floor) or fitted_floor <= 0:
            raise ValueError("scale_floor_override must be finite and positive")
    else:
        fitted_floor = (
            float(cp.asnumpy(cp.percentile(positive, context.scale_floor_percentile)))
            if int(positive.size)
            else 1.0
        )
    fitted_floor = max(fitted_floor, float(np.finfo(np.float32).eps))
    del quiet_scales, positive, quiet_review
    result = numerator / cp.maximum(scale, cp.float32(fitted_floor))
    result = result.astype(cp.float32)
    del numerator, scale
    cp.get_default_memory_pool().free_all_blocks()
    track()
    if peak_used > int(max_vram_bytes):
        raise MemoryError("observed CUDA allocation exceeded the configured cap")
    if not bool(cp.isfinite(result).all()):
        raise ValueError("CUDA joint MSLN produced non-finite values")
    return CUDAJointResult(
        values=result,
        scale_floor=fitted_floor,
        diagnostics={
            "context_id": context.context_id,
            "kind": "causal_joint_spatiotemporal",
            "backend": "cupy_cuda",
            "causal": True,
            "current_frame_excluded": True,
            "boundary_corrected": True,
            "spatial_outer_width_px": context.spatial_outer_width_px,
            "spatial_guard_width_px": context.spatial_guard_width_px,
            "temporal_window_frames": window,
            "temporal_guard_frames": guard,
            "reference_frame_count": window - guard,
            "scale_floor": fitted_floor,
            "scale_floor_source": "override" if scale_floor_override is not None else "quiet_calibration",
            "centering_offset": offset,
            "estimated_peak_vram_bytes": estimated_peak,
            "observed_peak_vram_bytes": peak_used,
            "runtime_seconds": time.monotonic() - started,
        },
    )


def bounded_residual_gate_cuda(values: Any, *, beta: float, kappa: float) -> Any:
    cp = _cupy()
    floor = float(beta)
    threshold = float(kappa)
    if not 0 <= floor <= 1 or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("beta must be in [0,1] and kappa positive")
    square = cp.square(values, dtype=cp.float32)
    return (
        floor
        + (1.0 - floor) * square / (threshold * threshold + square)
    ).astype(cp.float32)


def cs_parzen_objective_cuda(
    y: np.ndarray,
    bandwidth: float,
    *,
    weights: np.ndarray | None = None,
    block_rows: int = 256,
    kernel_dtype: np.dtype = np.float32,
) -> Any:
    """Evaluate the blockwise two-output CS objective with CuPy kernels."""
    from neurobench.algorithms.pairwise_separation import CSObjectiveResult

    cp = _cupy()
    values = np.asarray(y, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not values.size:
        raise ValueError("y must be a non-empty array with shape [N,2]")
    if not np.isfinite(values).all():
        raise ValueError("y must be finite")
    if not np.isfinite(bandwidth) or bandwidth <= 0 or block_rows < 1:
        raise ValueError("bandwidth and block_rows must be positive")
    kernel = np.dtype(kernel_dtype)
    if kernel not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("CUDA kernel dtype must be float32 or float64")
    if weights is None:
        sample_weights = np.ones(len(values), dtype=np.float64)
    else:
        sample_weights = np.asarray(weights, dtype=np.float64)
        if (
            sample_weights.shape != (len(values),)
            or not np.isfinite(sample_weights).all()
            or np.any(sample_weights < 0)
        ):
            raise ValueError("weights must be finite nonnegative with shape [N]")
    weight_sum = float(sample_weights.sum(dtype=np.float64))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("weights must have a positive finite sum")

    device_values = cp.asarray(values, dtype=cp.float64)
    device_weights_64 = cp.asarray(sample_weights, dtype=cp.float64)
    normalized = device_weights_64 / weight_sum
    mean = normalized @ device_values
    centered = device_values - mean
    variance = normalized @ cp.square(centered)
    standardized = centered / cp.sqrt(cp.maximum(variance, 1e-24))
    u, v = standardized.T
    device_weights = device_weights_64.astype(
        cp.float32 if kernel == np.dtype(np.float32) else cp.float64
    )
    joint_num = cp.asarray(0.0, dtype=cp.float64)
    marginal_1_num = cp.asarray(0.0, dtype=cp.float64)
    marginal_2_num = cp.asarray(0.0, dtype=cp.float64)
    cross_num = cp.asarray(0.0, dtype=cp.float64)
    cuda_kernel_dtype = cp.float32 if kernel == np.dtype(np.float32) else cp.float64
    for start in range(0, len(values), int(block_rows)):
        stop = min(len(values), start + int(block_rows))
        block_weights = device_weights[start:stop]
        delta_1 = (u[start:stop, None] - u[None, :]) / float(bandwidth)
        delta_2 = (v[start:stop, None] - v[None, :]) / float(bandwidth)
        kernel_1 = cp.exp(-0.5 * cp.square(delta_1)).astype(cuda_kernel_dtype)
        kernel_2 = cp.exp(-0.5 * cp.square(delta_2)).astype(cuda_kernel_dtype)
        kernel_1_weights = kernel_1 @ device_weights
        kernel_2_weights = kernel_2 @ device_weights
        joint_num += cp.asarray(
            cp.dot(block_weights, (kernel_1 * kernel_2) @ device_weights),
            dtype=cp.float64,
        )
        marginal_1_num += cp.asarray(
            cp.dot(block_weights, kernel_1_weights), dtype=cp.float64
        )
        marginal_2_num += cp.asarray(
            cp.dot(block_weights, kernel_2_weights), dtype=cp.float64
        )
        cross_num += cp.asarray(
            cp.dot(block_weights, kernel_1_weights * kernel_2_weights),
            dtype=cp.float64,
        )
    joint_value, marginal_1_value, marginal_2_value, cross_value = map(
        float,
        cp.asnumpy(
            cp.stack(
                (joint_num, marginal_1_num, marginal_2_num, cross_num)
            )
        ),
    )
    squared_sum = weight_sum * weight_sum
    v_joint = joint_value / squared_sum
    v_marginal_1 = marginal_1_value / squared_sum
    v_marginal_2 = marginal_2_value / squared_sum
    v_product = v_marginal_1 * v_marginal_2
    v_cross = cross_value / (squared_sum * weight_sum)
    terms = {
        "v_joint": v_joint,
        "v_product": v_product,
        "v_cross": v_cross,
    }
    epsilon = np.finfo(np.float64).tiny
    clamped = tuple(name for name, value in terms.items() if value <= epsilon)
    safe = {name: max(value, epsilon) for name, value in terms.items()}
    objective = -np.log(
        safe["v_cross"] / np.sqrt(safe["v_joint"] * safe["v_product"])
    )
    return CSObjectiveResult(
        objective=float(objective),
        v_joint=float(v_joint),
        v_marginal_1=float(v_marginal_1),
        v_marginal_2=float(v_marginal_2),
        v_product=float(v_product),
        v_cross=float(v_cross),
        numerical_clamps=len(clamped),
        clamped_terms=clamped,
        sample_count=len(values),
        positive_weight_count=int(np.count_nonzero(sample_weights)),
        weight_sum=weight_sum,
        block_rows=int(block_rows),
    )


def apply_per_context_fit_cuda(values: Any, fit: Any) -> tuple[Any, Any]:
    """Apply a fitted adjacent-frame transform without materializing pairs."""
    cp = _cupy()
    source = cp.asarray(values, dtype=cp.float32)
    effective = np.asarray(fit.demixing) @ np.asarray(fit.whitening)
    first = source[:-1] - np.float32(fit.center[0])
    second = source[1:] - np.float32(fit.center[1])
    persistence = cp.zeros_like(source)
    innovation = cp.zeros_like(source)
    persistence[1:] = effective[0, 0] * first + effective[0, 1] * second
    innovation[1:] = effective[1, 0] * first + effective[1, 1] * second
    return persistence, innovation


def gather_adjacent_pairs_cuda(values: Any, indices: np.ndarray) -> np.ndarray:
    cp = _cupy()
    selected = np.asarray(indices, dtype=np.int32)
    t, y, x = (cp.asarray(selected[:, item]) for item in range(3))
    result = cp.stack((values[t - 1, y, x], values[t, y, x]), axis=1)
    return cp.asnumpy(result).astype(np.float64)


def atomic_npy_from_cuda(path: Path, values: Any, *, frame_chunk: int = 8) -> None:
    cp = _cupy()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.npy")
    mapped = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.dtype(values.dtype),
        shape=tuple(map(int, values.shape)),
    )
    for start in range(0, len(values), int(frame_chunk)):
        stop = min(start + int(frame_chunk), len(values))
        mapped[start:stop] = cp.asnumpy(values[start:stop])
    mapped.flush()
    del mapped
    temporary.replace(destination)
