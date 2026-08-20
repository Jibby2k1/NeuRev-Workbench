"""Bounded CuPy application backend for quiet-fitted covariance whitening."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from neurobench.algorithms.local_covariance_whitening import (
    EmpiricalQuietCalibration,
    LocalWhiteningResult,
    SpatialTileConfig,
    TiledWhiteningFits,
    WhiteningFeatureBank,
    _selected_fit,
    fit_empirical_quiet_calibration,
    tile_blend_weight,
)
from neurobench.algorithms.msln_msica_cuda import _cupy, cuda_device_summary


def estimate_cuda_apply_peak_bytes(
    feature_shape: tuple[int, int, int, int],
    tile_config: SpatialTileConfig,
    *,
    frame_chunk: int,
) -> int:
    """Conservative device-allocation estimate for one application chunk."""
    _, height, width, dimension = map(int, feature_shape)
    chunk = int(frame_chunk)
    tile_elements = chunk * tile_config.tile_height * tile_config.tile_width * dimension
    field_elements = chunk * height * width * dimension
    # Input, centered/transformed tile temporaries, accumulated field, energy,
    # and a 25% allocator/workspace margin. Host outputs are not counted here.
    return int((field_elements * 2 + tile_elements * 3 + chunk * height * width) * 4 * 1.25)


def apply_tiled_feature_whiteners_cuda(
    feature_bank: WhiteningFeatureBank,
    fits: TiledWhiteningFits,
    tile_config: SpatialTileConfig,
    quiet_calibration: np.ndarray | EmpiricalQuietCalibration,
    *,
    frame_chunk: int = 8,
    probability_floor: float = 1e-6,
    max_vram_bytes: int = 4 * 2**30,
) -> LocalWhiteningResult:
    """Apply CPU-fitted transforms on CUDA in bounded frame chunks.

    Covariance fitting remains the deterministic float64 CPU reference. CUDA
    receives only frozen means/transforms and performs float32 application,
    overlap blending, and Mahalanobis-energy calculation.
    """
    if tile_config != fits.tile_config or feature_bank.feature_ids != fits.feature_ids:
        raise ValueError("feature order or tile configuration differs from fitted state")
    if frame_chunk < 1 or max_vram_bytes < 1:
        raise ValueError("frame_chunk and max_vram_bytes must be positive")
    cp = _cupy()
    free_before, _ = cp.cuda.runtime.memGetInfo()
    estimated_peak = estimate_cuda_apply_peak_bytes(
        tuple(map(int, feature_bank.values.shape)), tile_config, frame_chunk=frame_chunk
    )
    cap = min(int(max_vram_bytes), int(free_before))
    if estimated_peak > cap:
        raise MemoryError(f"CUDA whitening estimate {estimated_peak} exceeds cap {cap}")
    started = time.monotonic()
    frames, height, width, dimension = feature_bank.values.shape
    zca = np.empty((frames, height, width, dimension), dtype=np.float32)
    energy = np.empty((frames, height, width), dtype=np.float32)
    blend_sum = np.zeros((height, width), dtype=np.float32)
    unresolved = np.zeros((height, width), dtype=bool)
    fallback_counts: dict[str, int] = {}
    host_weight = tile_blend_weight(
        tile_config.tile_height, tile_config.tile_width, tile_config.blend
    )
    for bounds, primary, diagonal in zip(
        fits.tile_bounds_yx, fits.primary_fits, fits.local_diagonal_fits
    ):
        y0, y1, x0, x1 = bounds
        _, provenance = _selected_fit(primary, fits.global_full_fit, diagonal)
        fallback_counts[provenance] = fallback_counts.get(provenance, 0) + 1
        blend_sum[y0:y1, x0:x1] += host_weight
        if not primary.resolved:
            unresolved[y0:y1, x0:x1] = True
    if np.any(blend_sum <= 0):
        raise RuntimeError("tile blending left uncovered pixels")

    device_weight = cp.asarray(host_weight, dtype=cp.float32)
    device_blend_sum = cp.asarray(blend_sum, dtype=cp.float32)
    peak_used = 0

    def track() -> None:
        nonlocal peak_used
        free_now, _ = cp.cuda.runtime.memGetInfo()
        peak_used = max(peak_used, int(free_before - free_now))

    for start in range(0, frames, int(frame_chunk)):
        stop = min(frames, start + int(frame_chunk))
        accumulated = cp.zeros((stop - start, height, width, dimension), dtype=cp.float32)
        for bounds, primary, diagonal in zip(
            fits.tile_bounds_yx, fits.primary_fits, fits.local_diagonal_fits
        ):
            y0, y1, x0, x1 = bounds
            selected, _ = _selected_fit(primary, fits.global_full_fit, diagonal)
            block = cp.asarray(
                feature_bank.values[start:stop, y0:y1, x0:x1], dtype=cp.float32
            )
            if selected is None:
                transformed = block
            else:
                mean = cp.asarray(selected.mean, dtype=cp.float32)
                whitening = cp.asarray(selected.whitening, dtype=cp.float32)
                transformed = cp.einsum("...d,ed->...e", block - mean, whitening)
            accumulated[:, y0:y1, x0:x1] += transformed * device_weight[None, :, :, None]
            del block, transformed
        chunk_zca = accumulated / device_blend_sum[None, :, :, None]
        valid = cp.asarray(feature_bank.valid_frames[start:stop])
        chunk_zca[~valid] = 0
        chunk_energy = cp.sum(cp.square(chunk_zca, dtype=cp.float32), axis=-1, dtype=cp.float32)
        zca[start:stop] = cp.asnumpy(chunk_zca)
        energy[start:stop] = cp.asnumpy(chunk_energy)
        del accumulated, chunk_zca, chunk_energy, valid
        cp.get_default_memory_pool().free_all_blocks()
        track()
    del device_weight, device_blend_sum
    cp.get_default_memory_pool().free_all_blocks()
    track()
    if peak_used > int(max_vram_bytes):
        raise MemoryError("observed CUDA allocation exceeded the configured cap")
    calibration = (
        quiet_calibration
        if isinstance(quiet_calibration, EmpiricalQuietCalibration)
        else fit_empirical_quiet_calibration(
            energy, quiet_calibration, probability_floor=probability_floor
        )
    )
    surprise = calibration.surprise(energy)
    surprise[~feature_bank.valid_frames] = 0
    device = cuda_device_summary()
    return LocalWhiteningResult(
        zca_features=zca,
        mahalanobis_energy=energy,
        quiet_surprise=surprise,
        blend_weight=blend_sum,
        unresolved_tile_mask=unresolved,
        valid_frames=feature_bank.valid_frames.copy(),
        diagnostics={
            "feature_ids": list(feature_bank.feature_ids),
            "energy_derivation": "sum_of_squares_after_feature_blending",
            "fallback_policy": list(fits.fallback_policy),
            "fallback_counts": fallback_counts,
            "unresolved_pixel_fraction": float(np.mean(unresolved)),
            "calibration": calibration.diagnostics,
            "float64_fit_float32_apply": True,
            "application_backend": "cupy_cuda",
            "frame_chunk": int(frame_chunk),
            "estimated_peak_vram_bytes": estimated_peak,
            "observed_peak_vram_bytes": peak_used,
            "max_vram_bytes": int(max_vram_bytes),
            "runtime_seconds": time.monotonic() - started,
            "cuda_device": device,
        },
    )
