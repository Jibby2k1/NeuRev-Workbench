# utils.py
"""Shared utility functions and classes."""
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
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