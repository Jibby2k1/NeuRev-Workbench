"""Synthetic fish-like videos for template-grid workflow tests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from neurobench.algorithms.template_matching import _shift_image, _warp_similarity


@dataclass(frozen=True)
class SyntheticGridFishBundle:
    videos: dict[str, np.ndarray]
    labels: dict[str, str]
    transforms: dict[str, dict[str, float]]
    template: np.ndarray
    activated_regions: dict[str, tuple[int, int]]

    def write(self, root: str | Path, *, suffix: str = ".tif") -> dict[str, Any]:
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        if suffix.lower() in {".tif", ".tiff"}:
            import tifffile
        for video_id, video in self.videos.items():
            path = root_path / f"{video_id}{suffix}"
            if suffix.lower() == ".npy":
                np.save(path, video.astype(np.float32))
            elif suffix.lower() in {".tif", ".tiff"}:
                tifffile.imwrite(path, video.astype(np.float32))
            else:
                raise ValueError("suffix must be .npy, .tif, or .tiff")
            paths[video_id] = str(path)
        return {"root": str(root_path), "videos": paths, "transforms": self.transforms}


def generate_synthetic_grid_fish_videos(
    *,
    video_count_per_label: int = 3,
    labels: Iterable[str] = ("left", "right", "neutral"),
    frames: int = 64,
    height: int = 96,
    width: int = 128,
    grid_rows: int = 32,
    grid_cols: int = 32,
    rotation_deg_range: tuple[float, float] = (-5.0, 5.0),
    translation_px_range: tuple[float, float] = (-4.0, 4.0),
    noise_sigma: float = 0.05,
    seed: int = 7,
    include_outlier_frames: bool = True,
) -> SyntheticGridFishBundle:
    rng = np.random.default_rng(seed)
    base = _fish_projection(height, width)
    videos: dict[str, np.ndarray] = {}
    label_map: dict[str, str] = {}
    transforms: dict[str, dict[str, float]] = {}
    activated: dict[str, tuple[int, int]] = {}
    label_list = list(labels)
    for label in label_list:
        for index in range(1, video_count_per_label + 1):
            video_index = index
            video_id = f"{video_index}_{label}"
            rotation = float(rng.uniform(*rotation_deg_range))
            dy = float(rng.uniform(*translation_px_range))
            dx = float(rng.uniform(*translation_px_range))
            source_base = _warp_similarity(base, base.shape, rotation_deg=rotation)
            source_base = _shift_image(source_base, dy=dy, dx=dx)
            stack = np.repeat(source_base[None, :, :], frames, axis=0)
            rr, cc = _label_region(label, grid_rows, grid_cols)
            y0 = int(rr * height / grid_rows)
            y1 = int((rr + 1) * height / grid_rows)
            x0 = int(cc * width / grid_cols)
            x1 = int((cc + 1) * width / grid_cols)
            pulse = (np.sin(np.linspace(0, np.pi * 4, frames)) > 0.72).astype(np.float32)
            if label == "neutral":
                pulse *= 0.25
            for frame_index, value in enumerate(pulse):
                stack[frame_index, y0:y1, x0:x1] += float(value) * 0.65
            stack += rng.normal(0.0, noise_sigma, size=stack.shape).astype(np.float32)
            if include_outlier_frames and frames >= 8:
                stack[frames // 3] += 3.0
            videos[video_id] = stack.astype(np.float32)
            label_map[video_id] = label
            transforms[video_id] = {"source_rotation_deg": rotation, "source_translation_y_px": dy, "source_translation_x_px": dx}
            activated[video_id] = (rr, cc)
    return SyntheticGridFishBundle(videos=videos, labels=label_map, transforms=transforms, template=base, activated_regions=activated)


def _fish_projection(height: int, width: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    cy, cx = height * 0.52, width * 0.50
    body = np.exp(-(((x - cx) / (width * 0.30)) ** 2 + ((y - cy) / (height * 0.22)) ** 2))
    head = np.exp(-(((x - width * 0.68) / (width * 0.13)) ** 2 + ((y - height * 0.45) / (height * 0.16)) ** 2))
    tail = np.exp(-(((x - width * 0.25) / (width * 0.18)) ** 2 + ((y - height * 0.57) / (height * 0.10)) ** 2))
    texture = 0.08 * np.sin(x / 5.0) + 0.06 * np.cos(y / 7.0)
    arr = body + 0.75 * head + 0.45 * tail + texture
    arr -= arr.min()
    arr /= max(float(arr.max()), 1e-6)
    return arr.astype(np.float32)


def _label_region(label: str, rows: int, cols: int) -> tuple[int, int]:
    if label == "left":
        return rows // 2, cols // 4
    if label == "right":
        return rows // 2, (3 * cols) // 4
    return rows // 2, cols // 2
