# utils.py
"""Legacy grid-search utilities preserved for reproducibility.

New maintained work belongs under :mod:`neurobench`; see
``docs/CODEBASE_NAVIGATION.md`` before extending this compatibility surface.
"""
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def extract_pixel_detections(mask_np: np.ndarray) -> Dict[int, List[Dict[str, float]]]:
    """Extracts coordinates of detected pixels from a boolean mask."""
    detections = defaultdict(list)
    for i, frame_mask in enumerate(mask_np):
        if frame_mask.max() > 0:
            coords = np.argwhere(frame_mask > 0)
            detections[i].extend([{"y": float(y), "x": float(x)} for y, x in coords])
    return dict(detections)


def extract_component_detections(mask_np: np.ndarray, min_area: int = 1) -> Dict[int, List[Dict[str, float]]]:
    """Extracts one centroid detection per connected component in each frame.

    Components are 8-connected. The returned coordinates use the same ``x``/``y``
    keys as ``extract_pixel_detections`` so existing metric functions can consume
    them directly. ``area`` is included for object-level filtering and audits.
    """
    if mask_np.ndim != 3:
        raise ValueError(f"mask_np must have shape (T, H, W), got {mask_np.shape}")
    if min_area < 1:
        raise ValueError("min_area must be >= 1")

    detections = defaultdict(list)
    neighbors = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    for frame_idx, frame_mask in enumerate(mask_np > 0):
        if not frame_mask.any():
            continue
        height, width = frame_mask.shape
        visited = np.zeros_like(frame_mask, dtype=bool)
        ys, xs = np.nonzero(frame_mask)
        for start_y, start_x in zip(ys, xs):
            if visited[start_y, start_x]:
                continue
            stack = [(int(start_y), int(start_x))]
            visited[start_y, start_x] = True
            pixels = []
            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                for dx, dy in neighbors:
                    nx = x + dx
                    ny = y + dy
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    if visited[ny, nx] or not frame_mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
            if len(pixels) < min_area:
                continue
            py = np.array([p[0] for p in pixels], dtype=float)
            px = np.array([p[1] for p in pixels], dtype=float)
            detections[frame_idx].append(
                {"y": float(py.mean()), "x": float(px.mean()), "area": int(len(pixels))}
            )
    return dict(detections)
