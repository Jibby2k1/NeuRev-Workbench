"""Causal preprocessing and label-free bounded sample selection."""
from __future__ import annotations

from typing import Any

import numpy as np


def causal_preprocess(frames: np.ndarray, spatial_sigma_px: float, ema_span_frames: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    values = np.asarray(frames, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all() or spatial_sigma_px < 0 or ema_span_frames < 1:
        raise ValueError("invalid causal preprocessing inputs")
    spatial = (gaussian_filter(values, sigma=(0, spatial_sigma_px, spatial_sigma_px), mode="reflect", truncate=4)
               if spatial_sigma_px else values.copy()).astype(np.float32)
    if ema_span_frames == 1:
        return spatial
    alpha = 2 / (ema_span_frames + 1)
    result = np.empty_like(spatial); result[0] = spatial[0]
    for index in range(1, len(spatial)):
        result[index] = alpha * spatial[index] + (1 - alpha) * result[index - 1]
    return result


def uniform_anatomy_mask(quiet: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(quiet, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("quiet must be finite T,Y,X")
    median = np.median(values, axis=0)
    low, high = np.percentile(median, [5, 99.5])
    saturation = np.iinfo(np.uint16).max if median.max() > 4095 else 4095
    mask = np.isfinite(median) & (median >= low) & (median <= high) & (median < saturation)
    return mask, {"quiet_intensity_bounds": [float(low), float(high)], "eligible_pixels": int(mask.sum()),
                  "saturation_value": float(saturation)}


def sample_pair_observations(
    frames: np.ndarray, anatomy_mask: np.ndarray, lag_frames: int, sample_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = np.asarray(frames, dtype=np.float32)
    mask = np.asarray(anatomy_mask, dtype=bool)
    if values.ndim != 3 or mask.shape != values.shape[1:] or not 1 <= lag_frames < len(values):
        raise ValueError("invalid frames/mask/lag")
    pixels = np.flatnonzero(mask.ravel())
    pair_count = len(values) - lag_frames
    population = pair_count * len(pixels)
    if not 1 <= sample_count <= population:
        raise ValueError("sample_count exceeds valid pair/pixel population")
    rng = np.random.default_rng(seed)
    identities = np.sort(rng.choice(population, size=sample_count, replace=False))
    pair_indices = identities // len(pixels) + lag_frames
    local_pixels = pixels[identities % len(pixels)]
    y, x = np.unravel_index(local_pixels, mask.shape)
    observations = np.vstack((values[pair_indices - lag_frames, y, x], values[pair_indices, y, x])).astype(np.float64)
    manifest = {"seed": int(seed), "sample_count": int(sample_count), "population": int(population),
                "frame_indices_zero": pair_indices.astype(np.int32), "x_px": x.astype(np.int32), "y_px": y.astype(np.int32)}
    return observations, identities.astype(np.int64), manifest
