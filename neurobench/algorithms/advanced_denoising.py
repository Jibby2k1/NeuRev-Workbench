"""Carrier-preserving denoisers for Parzen-innovation calcium movies.

Every operator accepts a signed, quiet-standardized ``TYX`` residual and
returns a signal estimate with the same shape.  Filesystem, labels, experiment
selection, and visualization deliberately live outside this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from neurobench.algorithms.spatial_patch_ica import (
    ParzenShrinkage,
    SpatialPatchICAModel,
    fit_parzen_shrinkage,
)


def _video(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 3 or not result.size or not np.isfinite(result).all():
        raise ValueError("values must be a finite non-empty TYX array")
    return result


def carrier_blend(
    carrier: np.ndarray,
    estimate: np.ndarray,
    alpha: float,
    *,
    correction_limit_z: float | None = None,
) -> np.ndarray:
    """Apply a bounded residual correction while retaining the carrier."""
    source = _video(carrier)
    target = _video(estimate)
    fraction = float(alpha)
    if source.shape != target.shape or not 0 <= fraction <= 1:
        raise ValueError("carrier/estimate must align and alpha must be in [0,1]")
    correction = target - source
    if correction_limit_z is not None:
        limit = float(correction_limit_z)
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError("correction_limit_z must be positive")
        correction = np.clip(correction, -limit, limit)
    return (source + fraction * correction).astype(np.float32)


def fit_component_parzen_shrinkages(
    standardized_component_samples: np.ndarray,
    **settings: Any,
) -> tuple[ParzenShrinkage, ...]:
    """Fit one bounded noisy-Parzen posterior per ICA component."""
    samples = np.asarray(standardized_component_samples, dtype=np.float64)
    if samples.ndim != 2 or not np.isfinite(samples).all():
        raise ValueError("component samples must be finite samples-by-components")
    return tuple(
        fit_parzen_shrinkage(samples[:, index], **settings)
        for index in range(samples.shape[1])
    )


def _interp_torch(values, posterior: ParzenShrinkage):
    """Piecewise-linear lookup without transferring dense maps off CUDA."""
    import torch

    grid = torch.as_tensor(posterior.grid, device=values.device, dtype=values.dtype)
    lookup = torch.as_tensor(
        posterior.posterior_mean, device=values.device, dtype=values.dtype
    )
    clipped = torch.clamp(values, float(posterior.grid[0]), float(posterior.grid[-1]))
    indices = torch.searchsorted(grid, clipped.contiguous())
    indices = torch.clamp(indices, 1, len(grid) - 1)
    left = indices - 1
    x0 = grid[left]
    x1 = grid[indices]
    y0 = lookup[left]
    y1 = lookup[indices]
    fraction = (clipped - x0) / torch.clamp(x1 - x0, min=1e-12)
    return y0 + fraction * (y1 - y0)


def dense_ica_denoise(
    standardized_video: np.ndarray,
    model: SpatialPatchICAModel,
    *,
    mode: str,
    shared_parzen: ParzenShrinkage | None = None,
    component_parzen: Sequence[ParzenShrinkage] | None = None,
    wiener_lambda_z: float = 1.0,
    device: str = "cpu",
    frame_batch_size: int = 1,
    kalman_half_life_frames: float | None = None,
    kalman_process_variance: float = 0.08,
    kalman_observation_variance: float = 1.0,
    asymmetric_rise_gain: float = 0.85,
    asymmetric_decay_gain: float = 0.12,
    asymmetric_innovation_threshold_z: float = 0.75,
    asymmetric_innovation_temperature_z: float = 0.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Dense ICA analysis, component denoising, and overlap-add synthesis."""
    import torch
    import torch.nn.functional as functional

    video = _video(standardized_video)
    if mode not in {
        "wiener",
        "shared_parzen",
        "component_parzen",
        "kalman",
        "asymmetric",
    }:
        raise ValueError("unsupported dense ICA denoising mode")
    if mode == "shared_parzen" and shared_parzen is None:
        raise ValueError("shared_parzen mode requires a posterior")
    if mode == "component_parzen" and (
        component_parzen is None or len(component_parzen) != model.rank
    ):
        raise ValueError("component_parzen mode requires one posterior per component")
    target = torch.device(
        "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    batch_size = (
        1 if mode in {"kalman", "asymmetric"} else max(1, int(frame_batch_size))
    )
    patch = model.patch_size
    radius = patch // 2
    edge = np.maximum(np.hanning(patch), 0.1).astype(np.float32)
    window = np.outer(edge, edge).reshape(-1).astype(np.float32)
    analysis = torch.as_tensor(model.analysis_filters, device=target)
    synthesis = torch.as_tensor(
        model.synthesis_atoms * window[:, None], device=target
    )
    mean = torch.as_tensor(model.patch_mean, device=target)
    scale = torch.as_tensor(model.component_scale, device=target)
    window_tensor = torch.as_tensor(window, device=target)
    output = np.empty_like(video)
    state = None
    variance = float(kalman_observation_variance)
    if mode == "kalman":
        if kalman_half_life_frames is None or float(kalman_half_life_frames) <= 0:
            raise ValueError("kalman mode requires a positive half life")
        gamma = 0.5 ** (1.0 / float(kalman_half_life_frames))
    else:
        gamma = 0.0
    if mode == "asymmetric" and not (
        0 < float(asymmetric_decay_gain) <= float(asymmetric_rise_gain) <= 1
        and float(asymmetric_innovation_temperature_z) > 0
    ):
        raise ValueError("invalid asymmetric component-dynamics gains")
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    with torch.inference_mode():
        for start in range(0, len(video), batch_size):
            stop = min(len(video), start + batch_size)
            frames = torch.as_tensor(video[start:stop, None], device=target)
            padded = functional.pad(
                frames, (radius, radius, radius, radius), mode="reflect"
            )
            patches = functional.unfold(padded, kernel_size=patch)
            components = torch.einsum(
                "kd,bdn->bkn", analysis, patches - mean[None, :, None]
            )
            standardized = components / torch.clamp(scale[None, :, None], min=1e-6)
            if mode == "wiener":
                clean = standardized * (
                    standardized.square()
                    / (
                        standardized.square()
                        + float(wiener_lambda_z) ** 2
                    )
                )
            elif mode == "shared_parzen":
                clean = _interp_torch(standardized, shared_parzen)
            elif mode == "component_parzen":
                clean = torch.empty_like(standardized)
                for component, posterior in enumerate(component_parzen or ()):
                    clean[:, component] = _interp_torch(
                        standardized[:, component], posterior
                    )
            elif mode == "kalman":
                observation = standardized[0]
                predicted = (
                    torch.zeros_like(observation)
                    if state is None
                    else float(gamma) * state
                )
                predicted_variance = (
                    variance
                    if state is None
                    else gamma * gamma * variance
                    + float(kalman_process_variance)
                )
                gain = predicted_variance / (
                    predicted_variance + float(kalman_observation_variance)
                )
                state = predicted + gain * (observation - predicted)
                variance = (1.0 - gain) * predicted_variance
                clean = state[None]
            else:
                observation = standardized[0]
                if state is None:
                    state = observation
                else:
                    innovation = observation - state
                    rising = torch.abs(observation) > torch.abs(state)
                    base_gain = torch.where(
                        rising,
                        torch.full_like(observation, float(asymmetric_rise_gain)),
                        torch.full_like(observation, float(asymmetric_decay_gain)),
                    )
                    gate = torch.sigmoid(
                        (
                            torch.abs(innovation)
                            - float(asymmetric_innovation_threshold_z)
                        )
                        / float(asymmetric_innovation_temperature_z)
                    )
                    gain = (
                        float(asymmetric_decay_gain)
                        + (base_gain - float(asymmetric_decay_gain)) * gate
                    )
                    state = state + gain * innovation
                clean = state[None]
            clean = clean * scale[None, :, None]
            reconstructed_patches = torch.einsum(
                "dk,bkn->bdn", synthesis, clean
            )
            restored = functional.fold(
                reconstructed_patches,
                output_size=padded.shape[-2:],
                kernel_size=patch,
            )
            count = patches.shape[-1]
            weights = functional.fold(
                window_tensor[None, :, None].expand(len(frames), -1, count),
                output_size=padded.shape[-2:],
                kernel_size=patch,
            )
            restored /= torch.clamp(weights, min=1e-6)
            output[start:stop] = (
                restored[:, 0, radius:-radius, radius:-radius].cpu().numpy()
            )
            del frames, padded, patches, components, standardized, clean
            del reconstructed_patches, restored, weights
    peak_mib = (
        float(torch.cuda.max_memory_allocated(target) / 2**20)
        if target.type == "cuda"
        else 0.0
    )
    if target.type == "cuda":
        torch.cuda.empty_cache()
    return output, {
        "mode": mode,
        "device": str(target),
        "frame_batch_size": batch_size,
        "peak_gpu_memory_mib": peak_mib,
        "kalman_half_life_frames": kalman_half_life_frames,
        "asymmetric_rise_gain": (
            float(asymmetric_rise_gain) if mode == "asymmetric" else None
        ),
        "asymmetric_decay_gain": (
            float(asymmetric_decay_gain) if mode == "asymmetric" else None
        ),
    }


def multiscale_group_shrinkage(
    carrier: np.ndarray,
    scale_estimates: Sequence[np.ndarray],
    *,
    lambda_z: float,
    alpha: float,
    gain_floor: float = 0.25,
) -> np.ndarray:
    """Fuse multiple ICA denoisers using cross-scale energy evidence."""
    source = _video(carrier)
    estimates = [_video(item) for item in scale_estimates]
    if not estimates or any(item.shape != source.shape for item in estimates):
        raise ValueError("all scale estimates must align with the carrier")
    energy = np.mean(
        np.stack([np.square(item, dtype=np.float32) for item in estimates]),
        axis=0,
    )
    gain = energy / (energy + float(lambda_z) ** 2)
    gain = float(gain_floor) + (1.0 - float(gain_floor)) * gain
    target = np.mean(np.stack(estimates), axis=0) * gain
    return carrier_blend(source, target, float(alpha))


def bounded_noise_subtraction(
    carrier: np.ndarray,
    denoised_estimate: np.ndarray,
    *,
    alpha: float,
    correction_limit_z: float,
) -> np.ndarray:
    """Subtract only a bounded fraction of the ICA-estimated noise."""
    return carrier_blend(
        carrier,
        denoised_estimate,
        alpha,
        correction_limit_z=correction_limit_z,
    )


def noise_psd_wiener(
    values: np.ndarray,
    *,
    quiet_count: int,
    noise_multiplier: float,
    frequency_smoothing_sigma: float,
    alpha: float = 1.0,
) -> np.ndarray:
    """Global spatial Wiener filter calibrated from quiet-frame noise PSD."""
    from scipy.ndimage import gaussian_filter

    video = _video(values)
    if not 2 <= int(quiet_count) < len(video):
        raise ValueError("quiet_count must define a strict leading subset")
    quiet_fft = np.fft.rfft2(video[: int(quiet_count)], axes=(-2, -1))
    total_fft = np.fft.rfft2(video, axes=(-2, -1))
    noise_psd = np.mean(np.abs(quiet_fft) ** 2, axis=0)
    total_psd = np.mean(np.abs(total_fft) ** 2, axis=0)
    signal_psd = np.maximum(
        total_psd - float(noise_multiplier) * noise_psd, 0
    )
    transfer = signal_psd / np.maximum(signal_psd + noise_psd, 1e-12)
    sigma = float(frequency_smoothing_sigma)
    if sigma > 0:
        transfer = gaussian_filter(transfer, sigma=sigma, mode="nearest")
    filtered = np.fft.irfft2(
        total_fft * transfer[None], s=video.shape[-2:], axes=(-2, -1)
    ).real.astype(np.float32)
    return carrier_blend(video, filtered, float(alpha))


def windowed_robust_low_rank_sparse(
    values: np.ndarray,
    *,
    window_frames: int,
    rank: int,
    sparse_lambda_z: float,
    alpha: float,
    device: str = "cpu",
) -> np.ndarray:
    """Short-window low-rank plus soft-sparse decomposition."""
    import torch

    video = _video(values)
    window = int(window_frames)
    selected_rank = int(rank)
    if not 4 <= window <= len(video) or not 1 <= selected_rank < window:
        raise ValueError("invalid robust low-rank window or rank")
    target = torch.device(
        "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    step = max(1, window // 2)
    starts = list(range(0, len(video) - window + 1, step))
    if starts[-1] != len(video) - window:
        starts.append(len(video) - window)
    output = np.zeros_like(video)
    weight = np.zeros(len(video), dtype=np.float32)
    taper = np.maximum(np.hanning(window), 0.1).astype(np.float32)
    with torch.inference_mode():
        for start in starts:
            block = torch.as_tensor(
                video[start : start + window].reshape(window, -1),
                device=target,
            )
            center = torch.median(block, dim=0, keepdim=True).values
            centered = block - center
            covariance = centered @ centered.T / centered.shape[1]
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            basis = eigenvectors[:, -selected_rank:]
            low_rank = basis @ (basis.T @ centered)
            remainder = centered - low_rank
            sparse = torch.sign(remainder) * torch.relu(
                torch.abs(remainder) - float(sparse_lambda_z)
            )
            estimate = (low_rank + sparse).reshape(
                window, video.shape[1], video.shape[2]
            ).cpu().numpy()
            output[start : start + window] += estimate * taper[:, None, None]
            weight[start : start + window] += taper
    output /= np.maximum(weight[:, None, None], 1e-6)
    if target.type == "cuda":
        torch.cuda.empty_cache()
    return carrier_blend(video, output, float(alpha))


def windowed_nonnegative_factorization(
    values: np.ndarray,
    *,
    window_frames: int,
    rank: int,
    iterations: int,
    alpha: float,
    seed: int,
    device: str = "cpu",
) -> np.ndarray:
    """Windowed nonnegative low-rank factorization with signed carrier skip."""
    import torch

    video = _video(values)
    window = int(window_frames)
    selected_rank = int(rank)
    if not 4 <= window <= len(video) or not 1 <= selected_rank < window:
        raise ValueError("invalid NMF window or rank")
    target = torch.device(
        "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    step = max(1, window // 2)
    starts = list(range(0, len(video) - window + 1, step))
    if starts[-1] != len(video) - window:
        starts.append(len(video) - window)
    output = np.zeros_like(video)
    weight = np.zeros(len(video), dtype=np.float32)
    taper = np.maximum(np.hanning(window), 0.1).astype(np.float32)
    generator = torch.Generator(device=target)
    generator.manual_seed(int(seed))
    epsilon = 1e-6
    with torch.inference_mode():
        for start in starts:
            x = torch.as_tensor(
                np.maximum(
                    video[start : start + window].reshape(window, -1), 0
                ),
                device=target,
            )
            temporal = torch.rand(
                (window, selected_rank), generator=generator, device=target
            ) + 0.1
            spatial = torch.rand(
                (selected_rank, x.shape[1]), generator=generator, device=target
            ) + 0.1
            for _ in range(int(iterations)):
                spatial *= (temporal.T @ x) / torch.clamp(
                    temporal.T @ temporal @ spatial, min=epsilon
                )
                temporal *= (x @ spatial.T) / torch.clamp(
                    temporal @ spatial @ spatial.T, min=epsilon
                )
            estimate = (temporal @ spatial).reshape(
                window, video.shape[1], video.shape[2]
            ).cpu().numpy()
            output[start : start + window] += estimate * taper[:, None, None]
            weight[start : start + window] += taper
    output /= np.maximum(weight[:, None, None], 1e-6)
    if target.type == "cuda":
        torch.cuda.empty_cache()
    return carrier_blend(video, output, float(alpha))


def nonlocal_means_spatial(
    values: np.ndarray,
    *,
    search_radius: int,
    patch_size: int,
    bandwidth_z: float,
    alpha: float,
    device: str = "cpu",
    frame_batch_size: int = 4,
) -> np.ndarray:
    """GPU-bounded spatial nonlocal means using local patch distances."""
    import torch
    import torch.nn.functional as functional

    video = _video(values)
    radius = int(search_radius)
    patch = int(patch_size)
    if not 1 <= radius <= 4 or patch < 1 or patch % 2 == 0:
        raise ValueError("invalid nonlocal search radius or patch size")
    target = torch.device(
        "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    output = np.empty_like(video)
    batch = max(1, int(frame_batch_size))
    denominator_scale = max(float(bandwidth_z) ** 2, 1e-6)
    with torch.inference_mode():
        for start in range(0, len(video), batch):
            stop = min(len(video), start + batch)
            source = torch.as_tensor(video[start:stop, None], device=target)
            padded = functional.pad(
                source, (radius, radius, radius, radius), mode="reflect"
            )
            numerator = torch.zeros_like(source)
            denominator = torch.zeros_like(source)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    shifted = padded[
                        :,
                        :,
                        radius + dy : radius + dy + source.shape[-2],
                        radius + dx : radius + dx + source.shape[-1],
                    ]
                    distance = functional.avg_pool2d(
                        (source - shifted).square(),
                        kernel_size=patch,
                        stride=1,
                        padding=patch // 2,
                    )
                    weight = torch.exp(-distance / denominator_scale)
                    numerator += weight * shifted
                    denominator += weight
            estimate = numerator / torch.clamp(denominator, min=1e-6)
            output[start:stop] = estimate[:, 0].cpu().numpy()
    if target.type == "cuda":
        torch.cuda.empty_cache()
    return carrier_blend(video, output, float(alpha))


def undecimated_spatial_group_shrinkage(
    values: np.ndarray,
    *,
    levels: int,
    threshold_z: float,
    group_sigma_px: float,
    coarse_keep: float,
) -> np.ndarray:
    """Shift-invariant spatial multiscale decomposition with energy shrinkage."""
    from scipy.ndimage import gaussian_filter

    video = _video(values)
    level_count = int(levels)
    if not 1 <= level_count <= 5 or not 0 <= float(coarse_keep) <= 1:
        raise ValueError("invalid wavelet levels or coarse_keep")
    approximation = video.copy()
    retained = np.zeros_like(video)
    for level in range(level_count):
        sigma = float(2**level)
        smooth = gaussian_filter(
            approximation, sigma=(0, sigma, sigma), mode="reflect"
        )
        detail = approximation - smooth
        energy = gaussian_filter(
            np.square(detail, dtype=np.float32),
            sigma=(0, float(group_sigma_px), float(group_sigma_px)),
            mode="reflect",
        )
        gain = energy / (energy + float(threshold_z) ** 2)
        retained += detail * gain
        approximation = smooth
    return (
        retained + float(coarse_keep) * approximation
    ).astype(np.float32)
