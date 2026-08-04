"""Chunked application of aligned two-frame context fits."""
from __future__ import annotations

import numpy as np

from neurobench.algorithms.multiscale_subspace import PerContextICAFit, apply_per_context_fit


def apply_innovation(values: np.ndarray, fit: PerContextICAFit, valid_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    persistence = np.zeros_like(values, dtype=np.float32)
    innovation = np.zeros_like(values, dtype=np.float32)
    for frame in np.flatnonzero(np.asarray(valid_frames, dtype=bool) & np.r_[False, np.asarray(valid_frames[:-1], dtype=bool)]):
        pairs = np.column_stack([values[frame - 1].ravel(), values[frame].ravel()])
        output = apply_per_context_fit(pairs, fit)
        persistence[frame] = output[:, 0].reshape(values.shape[1:])
        innovation[frame] = output[:, 1].reshape(values.shape[1:])
    return persistence, innovation
