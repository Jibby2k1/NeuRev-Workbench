"""Application operators for fitted spatial patch ICA models."""
from __future__ import annotations

from typing import Any

import numpy as np

from neurobench.algorithms.spatial_patch_ica import (
    ParzenShrinkage,
    SpatialPatchICAModel,
    shrink_components,
)


def _finite_video(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or not array.size or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty finite TYX array")
    return array


def patch_lattice_reconstruction(
    standardized_video: np.ndarray,
    model: SpatialPatchICAModel,
    *,
    stride: int,
    shrinkage: str,
    lambda_z: float = 1.0,
    parzen: ParzenShrinkage | None = None,
) -> np.ndarray:
    """Apply one shared patch transform on a bounded overlap-add lattice."""
    video = _finite_video(standardized_video)
    patch = model.patch_size
    step = int(stride)
    if not 1 <= step <= patch:
        raise ValueError("stride must be between one and patch_size")

    def positions(length: int) -> list[int]:
        selected = list(range(0, length - patch + 1, step))
        if not selected or selected[-1] != length - patch:
            selected.append(length - patch)
        return selected

    ys = positions(video.shape[1])
    xs = positions(video.shape[2])
    edge = np.maximum(np.hanning(patch), 0.1).astype(np.float32)
    window = np.outer(edge, edge).astype(np.float32)
    output = np.zeros_like(video)
    weight = np.zeros(video.shape[1:], dtype=np.float32)
    for y in ys:
        for x in xs:
            blocks = video[:, y : y + patch, x : x + patch].reshape(
                len(video), -1
            )
            components = (
                blocks - model.patch_mean[None]
            ) @ model.analysis_filters.T
            clean = shrink_components(
                components,
                model.component_scale,
                method=shrinkage,
                lambda_z=lambda_z,
                parzen=parzen,
            )
            reconstructed = (clean @ model.synthesis_atoms.T).reshape(
                len(video), patch, patch
            )
            output[:, y : y + patch, x : x + patch] += (
                reconstructed * window[None]
            )
            weight[y : y + patch, x : x + patch] += window
    output /= np.maximum(weight[None], 1e-6)
    return output


def dense_convolutional_reconstruction(
    standardized_video: np.ndarray,
    model: SpatialPatchICAModel,
    *,
    shrinkage: str,
    lambda_z: float = 1.0,
    parzen: ParzenShrinkage | None = None,
    device: str = "cpu",
    frame_batch_size: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply patch ICA densely using unfold/fold convolutional weight sharing."""
    import torch
    import torch.nn.functional as functional

    video = _finite_video(standardized_video)
    target = torch.device(
        "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    patch = model.patch_size
    radius = patch // 2
    edge = np.maximum(np.hanning(patch), 0.1).astype(np.float32)
    window = np.outer(edge, edge).reshape(-1).astype(np.float32)
    analysis = torch.as_tensor(model.analysis_filters, device=target)
    synthesis = torch.as_tensor(
        model.synthesis_atoms * window[:, None], device=target
    )
    patch_mean = torch.as_tensor(model.patch_mean, device=target)
    component_scale = torch.as_tensor(model.component_scale, device=target)
    output = np.empty_like(video)
    resolved_batch = max(1, int(frame_batch_size))
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    with torch.inference_mode():
        for start in range(0, len(video), resolved_batch):
            stop = min(len(video), start + resolved_batch)
            frames = torch.as_tensor(video[start:stop, None], device=target)
            padded = functional.pad(
                frames, (radius, radius, radius, radius), mode="reflect"
            )
            patches = functional.unfold(padded, kernel_size=patch)
            components = torch.einsum(
                "kd,bdn->bkn", analysis, patches - patch_mean[None, :, None]
            )
            standardized = components / torch.clamp(
                component_scale[None, :, None], min=1e-6
            )
            if shrinkage == "wiener":
                clean = standardized * (
                    standardized.square()
                    / (standardized.square() + float(lambda_z) ** 2)
                )
            elif shrinkage == "parzen":
                if parzen is None:
                    raise ValueError("parzen shrinkage requires a fitted posterior")
                standardized_np = standardized.cpu().numpy()
                clean_np = np.interp(
                    np.clip(
                        standardized_np, parzen.grid[0], parzen.grid[-1]
                    ),
                    parzen.grid,
                    parzen.posterior_mean,
                ).astype(np.float32)
                clean = torch.as_tensor(clean_np, device=target)
            else:
                raise ValueError("shrinkage must be wiener or parzen")
            clean = clean * component_scale[None, :, None]
            reconstructed_patches = torch.einsum(
                "dk,bkn->bdn", synthesis, clean
            )
            restored = functional.fold(
                reconstructed_patches,
                output_size=padded.shape[-2:],
                kernel_size=patch,
            )
            count = patches.shape[-1]
            weight_patches = torch.as_tensor(
                window, device=target
            )[None, :, None].expand(len(frames), -1, count)
            weights = functional.fold(
                weight_patches,
                output_size=padded.shape[-2:],
                kernel_size=patch,
            )
            restored = restored / torch.clamp(weights, min=1e-6)
            output[start:stop] = (
                restored[:, 0, radius:-radius, radius:-radius].cpu().numpy()
            )
            del frames, padded, patches, components, standardized, clean
            del reconstructed_patches, restored, weight_patches, weights
    if target.type == "cuda":
        peak_mib = torch.cuda.max_memory_allocated(target) / 2**20
        torch.cuda.empty_cache()
    else:
        peak_mib = 0.0
    return output, {
        "device": str(target),
        "frame_batch_size": resolved_batch,
        "application_stride": 1,
        "translation_shared": True,
        "peak_gpu_memory_mib": float(peak_mib),
    }
