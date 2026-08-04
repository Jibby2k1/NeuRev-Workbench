"""Leakage-safe deterministic sample selection and per-context fitting."""
from __future__ import annotations

import numpy as np

from neurobench.algorithms.multiscale_subspace import PerContextICAFit, fit_per_context_ica
from .config import MSLNMSICAConfig


def adjacent_sample_indices(shape: tuple[int, int, int], valid_frames: np.ndarray, *, count: int, seed: int) -> np.ndarray:
    frames, height, width = shape
    valid_t = np.flatnonzero(np.asarray(valid_frames, dtype=bool) & np.r_[False, np.asarray(valid_frames[:-1], dtype=bool)])
    total = len(valid_t) * height * width
    if total < 2:
        raise ValueError("too few valid adjacent samples")
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(total, size=min(int(count), total), replace=False))
    t_pos, pixel = np.divmod(selected, height * width)
    return np.column_stack([valid_t[t_pos], pixel // width, pixel % width]).astype(np.int32)


def pairs_at(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    t, y, x = np.asarray(indices).T
    return np.column_stack([values[t - 1, y, x], values[t, y, x]]).astype(np.float64)


def fit_context(context_id: str, values: np.ndarray, valid_frames: np.ndarray, config: MSLNMSICAConfig) -> tuple[PerContextICAFit, np.ndarray, np.ndarray]:
    confirmation = adjacent_sample_indices(values.shape, valid_frames, count=config.sampling.per_context_confirmation_samples, seed=config.sampling.seed)
    screen_count = min(config.sampling.per_context_screen_samples, len(confirmation))
    rng = np.random.default_rng(config.sampling.seed + 1)
    screen = confirmation[
        np.sort(rng.choice(len(confirmation), size=screen_count, replace=False))
    ]
    fit = fit_per_context_ica(context_id, pairs_at(values, screen), pairs_at(values, confirmation), objective=config.per_context_ica.primary_objective, parzen_bandwidth=config.per_context_ica.parzen_bandwidth, eigenvalue_floor_ratio=config.per_context_ica.eigenvalue_floor_ratio, coarse_step_degrees=config.per_context_ica.coarse_step_degrees, refine_half_width_degrees=config.per_context_ica.refine_half_width_degrees, refine_step_degrees=config.per_context_ica.refine_step_degrees, kernel_block_rows=config.per_context_ica.kernel_block_rows, kernel_dtype=np.dtype(config.compute.kernel_dtype))
    return fit, screen, confirmation
