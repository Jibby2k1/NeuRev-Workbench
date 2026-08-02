"""Local information-theoretic patch statistics for calcium-imaging data."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def gaussian_information_kernel(
    centers: Sequence[float], bandwidth: float
) -> np.ndarray:
    """Return the Gaussian pair-interaction matrix for quadratic Renyi IP.

    Two Gaussian Parzen kernels of standard deviation ``bandwidth`` integrate
    to a kernel proportional to ``exp(-(a-b)^2 / (4 bandwidth^2))``.  The
    omitted common normalization cancels in Cauchy--Schwarz divergence and is
    irrelevant to feature ordering.
    """
    values = np.asarray(centers, dtype=np.float64)
    scale = float(bandwidth)
    if values.ndim != 1 or len(values) < 3 or not np.all(np.diff(values) > 0):
        raise ValueError("centers must be a strictly increasing vector")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("bandwidth must be finite and positive")
    delta = values[:, None] - values[None, :]
    return np.exp(-(delta * delta) / (4.0 * scale * scale)).astype(np.float32)


def quantization_boundaries(centers: Sequence[float]) -> np.ndarray:
    """Return midpoint boundaries for nearest-center scalar quantization."""
    values = np.asarray(centers, dtype=np.float64)
    if values.ndim != 1 or len(values) < 3 or not np.all(np.diff(values) > 0):
        raise ValueError("centers must be a strictly increasing vector")
    return ((values[:-1] + values[1:]) * 0.5).astype(np.float32)


def local_histogram_tensor(
    frames: Any,
    *,
    centers: Sequence[float],
    patch_size_px: int,
):
    """Quantize ``NYX`` frames and return local probabilities as ``NBYX``.

    The return value is a torch tensor on the same device as ``frames``.
    Boundary windows use the number of valid pixels rather than zero padding.
    """
    return local_histogram_pyramid_tensor(
        frames, centers=centers, patch_sizes_px=(int(patch_size_px),)
    )[int(patch_size_px)]


def local_histogram_pyramid_tensor(
    frames: Any,
    *,
    centers: Sequence[float],
    patch_sizes_px: Sequence[int],
) -> dict[int, Any]:
    """Quantize once and return local probability tensors at several scales."""
    import torch
    import torch.nn.functional as functional

    values = frames if torch.is_tensor(frames) else torch.as_tensor(frames)
    if values.ndim != 3 or not values.is_floating_point():
        raise ValueError("frames must be a floating NYX tensor")
    patches = tuple(int(value) for value in patch_sizes_px)
    if (
        not patches
        or len(set(patches)) != len(patches)
        or any(value < 3 or value % 2 != 1 or value > 31 for value in patches)
    ):
        raise ValueError("patch sizes must be unique odd integers in [3, 31]")
    boundaries = torch.as_tensor(
        quantization_boundaries(centers), device=values.device, dtype=values.dtype
    )
    indices = torch.bucketize(values.contiguous(), boundaries)
    bins = len(tuple(centers))
    one_hot = functional.one_hot(indices, num_classes=bins).permute(0, 3, 1, 2)
    one_hot = one_hot.to(dtype=values.dtype)
    return {
        patch: functional.avg_pool2d(
            one_hot,
            kernel_size=patch,
            stride=1,
            padding=patch // 2,
            count_include_pad=False,
        )
        for patch in patches
    }


def local_center_annulus_histograms_tensor(
    frames: Any,
    *,
    centers: Sequence[float],
    center_patch_px: int,
    outer_patch_px: int,
) -> tuple[Any, Any]:
    """Return exact center and square-annulus local distributions.

    Counts are corrected at image boundaries. The annulus excludes every
    sample in the centered inner square from the larger outer square.
    """
    import torch
    import torch.nn.functional as functional

    values = frames if torch.is_tensor(frames) else torch.as_tensor(frames)
    inner = int(center_patch_px)
    outer = int(outer_patch_px)
    if (
        values.ndim != 3
        or not values.is_floating_point()
        or inner < 3
        or inner % 2 != 1
        or outer <= inner
        or outer % 2 != 1
        or outer > 31
    ):
        raise ValueError("invalid frames or center/outer patch pair")
    boundaries = torch.as_tensor(
        quantization_boundaries(centers), device=values.device, dtype=values.dtype
    )
    indices = torch.bucketize(values.contiguous(), boundaries)
    one_hot = functional.one_hot(indices, num_classes=len(tuple(centers)))
    one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=values.dtype)

    def sums(tensor, patch: int):
        return functional.avg_pool2d(
            tensor,
            kernel_size=patch,
            stride=1,
            padding=patch // 2,
            count_include_pad=False,
            divisor_override=1,
        )

    inner_sum = sums(one_hot, inner)
    outer_sum = sums(one_hot, outer)
    ones = torch.ones(
        (values.shape[0], 1, values.shape[1], values.shape[2]),
        dtype=values.dtype,
        device=values.device,
    )
    inner_count = sums(ones, inner)
    outer_count = sums(ones, outer)
    center = inner_sum / inner_count.clamp_min(1.0)
    annulus = (outer_sum - inner_sum).clamp_min(0.0)
    annulus = annulus / (outer_count - inner_count).clamp_min(1.0)
    return center, annulus


def cauchy_schwarz_divergence_tensor(
    first,
    second,
    *,
    centers: Sequence[float],
    bandwidth: float,
    epsilon: float = 1e-8,
):
    """Return Gaussian-Parzen Cauchy–Schwarz divergence for aligned PDFs."""
    import torch

    p, q = first, second
    if p.shape != q.shape or p.ndim not in {3, 4}:
        raise ValueError("aligned distributions must have shape BYX or NBYX")
    kernel = torch.as_tensor(
        gaussian_information_kernel(centers, bandwidth),
        dtype=p.dtype,
        device=p.device,
    )
    if p.ndim == 4:
        kp = torch.einsum("bc,ncyx->nbyx", kernel, p)
        kq = torch.einsum("bc,ncyx->nbyx", kernel, q)
        dimension = 1
    else:
        kp = torch.einsum("bc,cyx->byx", kernel, p)
        kq = torch.einsum("bc,cyx->byx", kernel, q)
        dimension = 0
    pp = torch.sum(p * kp, dim=dimension).clamp_min(float(epsilon))
    qq = torch.sum(q * kq, dim=dimension).clamp_min(float(epsilon))
    pq = torch.sum(p * kq, dim=dimension).clamp_min(float(epsilon))
    cosine_sq = (pq * pq / (pp * qq)).clamp(
        min=float(epsilon), max=1.0
    )
    return -torch.log(cosine_sq)


def information_fields_tensor(
    histogram,
    center_values,
    quiet_histogram,
    *,
    centers: Sequence[float],
    bandwidth: float,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Compute quadratic IP, quiet CS divergence, and local correntropy.

    ``histogram`` is ``NBYX`` and ``quiet_histogram`` is ``BYX``.  The
    Cauchy--Schwarz statistic compares the current local Parzen density to the
    frozen quiet local density at the same spatial location.
    """
    import torch

    p = histogram
    q = quiet_histogram
    if p.ndim != 4 or q.ndim != 3 or p.shape[1:] != q.shape:
        raise ValueError("histogram and quiet_histogram shapes do not align")
    if center_values.shape != (p.shape[0], p.shape[2], p.shape[3]):
        raise ValueError("center_values must have shape NYX")
    kernel = torch.as_tensor(
        gaussian_information_kernel(centers, bandwidth),
        dtype=p.dtype,
        device=p.device,
    )
    kp = torch.einsum("bc,ncyx->nbyx", kernel, p)
    pp = torch.sum(p * kp, dim=1).clamp_min(float(epsilon))
    cs_divergence = cauchy_schwarz_divergence_tensor(
        p,
        q[None].expand_as(p),
        centers=centers,
        bandwidth=bandwidth,
        epsilon=epsilon,
    )
    center_grid = torch.as_tensor(
        tuple(float(value) for value in centers), dtype=p.dtype, device=p.device
    )
    correntropy_kernel = torch.exp(
        -(
            center_values[:, None] - center_grid[None, :, None, None]
        ).square()
        / (2.0 * float(bandwidth) ** 2)
    )
    correntropy = torch.sum(p * correntropy_kernel, dim=1)
    return {
        "renyi2_information_potential": pp,
        "cs_quiet_divergence": cs_divergence,
        "local_correntropy": correntropy,
    }
