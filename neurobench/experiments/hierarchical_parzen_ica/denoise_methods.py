"""Pure denoising operators used by the sequential Parzen-Innovation audit."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from neurobench.algorithms.hierarchical_parzen_ica import noisy_parzen_posterior_mean


def frame_gamma(values: np.ndarray, gamma: float) -> np.ndarray:
    """Literal per-frame positive min/max shrinkage with large values preserved."""
    x = np.asarray(values, dtype=np.float32)
    positive = np.maximum(x, 0)
    high = np.max(positive, axis=(-2, -1), keepdims=True)
    unit = np.clip(positive / np.maximum(high, 1e-6), 0, 1)
    return (positive * np.power(unit, float(gamma) - 1.0)).astype(np.float32)


def robust_gamma(
    values: np.ndarray, gamma: float, lower_percentile: float, upper_percentile: float
) -> np.ndarray:
    """Per-frame robust-quantile shrinkage; values beyond the knee pass unchanged."""
    x = np.asarray(values, dtype=np.float32)
    positive = np.maximum(x, 0)
    flat = positive.reshape(len(positive), -1)
    low = np.percentile(flat, float(lower_percentile), axis=1).reshape(-1, 1, 1)
    high = np.percentile(flat, float(upper_percentile), axis=1).reshape(-1, 1, 1)
    shifted = np.maximum(positive - low, 0)
    width = np.maximum(high - low, 1e-6)
    unit = np.clip(shifted / width, 0, 1)
    return (shifted * np.power(unit, float(gamma) - 1.0)).astype(np.float32)


def quiet_wiener(
    values: np.ndarray, scale: np.ndarray, lambda_z: float
) -> np.ndarray:
    """Signed per-pixel Wiener-like shrinkage in quiet-standardized units."""
    x = np.asarray(values, dtype=np.float32)
    variance = (float(lambda_z) * np.asarray(scale, dtype=np.float32)) ** 2
    return (x * (x * x / (x * x + variance[None] + 1e-12))).astype(np.float32)


def spatial_evidence_gate(
    values: np.ndarray,
    scale: np.ndarray,
    *,
    sigma_px: float,
    lambda_z: float,
    structural_floor: float,
) -> np.ndarray:
    """Use smoothed evidence to gate the untouched signed residual carrier."""
    from scipy.ndimage import gaussian_filter

    x = np.asarray(values, dtype=np.float32)
    z = x / np.asarray(scale, dtype=np.float32)[None]
    output = np.empty_like(x)
    for index, frame in enumerate(z):
        evidence = gaussian_filter(np.abs(frame), sigma=float(sigma_px), mode="reflect")
        gate = evidence * evidence / (evidence * evidence + float(lambda_z) ** 2)
        gain = float(structural_floor) + (1.0 - float(structural_floor)) * gate
        output[index] = x[index] * gain
    return output


def temporal_evidence_gate(
    values: np.ndarray,
    scale: np.ndarray,
    *,
    sigma_px: float,
    lambda_z: float,
    structural_floor: float,
    half_life_ms: float,
    frame_period_ms: float,
) -> np.ndarray:
    """Causal spatial-energy gate with a declared short temporal half-life."""
    from scipy.ndimage import gaussian_filter

    x = np.asarray(values, dtype=np.float32)
    z = x / np.asarray(scale, dtype=np.float32)[None]
    refresh = 1.0 - 0.5 ** (float(frame_period_ms) / float(half_life_ms))
    state = np.zeros_like(scale, dtype=np.float32)
    output = np.empty_like(x)
    for index, frame in enumerate(z):
        evidence = gaussian_filter(np.abs(frame), sigma=float(sigma_px), mode="reflect")
        state = (1.0 - refresh) * state + refresh * evidence * evidence
        gate = state / (state + float(lambda_z) ** 2)
        gain = float(structural_floor) + (1.0 - float(structural_floor)) * gate
        output[index] = x[index] * gain
    return output


def savgol_signal(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    from scipy.signal import savgol_filter

    return savgol_filter(
        np.asarray(values, dtype=np.float32),
        window_length=int(window),
        polyorder=int(polyorder),
        axis=0,
        mode="interp",
    ).astype(np.float32)


def undecimated_haar_like(
    values: np.ndarray, scale: np.ndarray, *, levels: int, threshold_z: float
) -> np.ndarray:
    """Translation-preserving multiscale Haar-like detail shrinkage."""
    x = np.asarray(values, dtype=np.float32)
    z = x / np.asarray(scale, dtype=np.float32)[None]
    flat = z.reshape(len(z), -1)
    result = np.empty_like(flat)
    chunk = 16384
    for start in range(0, flat.shape[1], chunk):
        stop = min(flat.shape[1], start + chunk)
        approximation = flat[:, start:stop].copy()
        retained = np.zeros_like(approximation)
        for level in range(int(levels)):
            step = 2**level
            shifted = np.empty_like(approximation)
            shifted[:step] = approximation[:step][::-1]
            shifted[step:] = approximation[:-step]
            smooth = 0.5 * (approximation + shifted)
            detail = approximation - smooth
            retained += np.sign(detail) * np.maximum(
                np.abs(detail) - float(threshold_z), 0
            )
            approximation = smooth
        result[:, start:stop] = approximation + retained
    return (result.reshape(x.shape) * scale[None]).astype(np.float32)


def causal_kalman(
    values: np.ndarray,
    scale: np.ndarray,
    *,
    frame_period_ms: float,
    decay_ms: float,
    process_variance: float,
    observation_variance: float,
) -> np.ndarray:
    """Memory-bounded vectorized scalar AR(1) Kalman filter."""
    x = np.asarray(values, dtype=np.float32)
    z = x / np.asarray(scale, dtype=np.float32)[None]
    gamma = min(math.exp(-float(frame_period_ms) / float(decay_ms)), 0.999)
    state = np.zeros_like(scale, dtype=np.float32)
    variance = float(observation_variance)
    output = np.empty_like(z)
    for index, frame in enumerate(z):
        predicted = gamma * state
        predicted_variance = (
            gamma * gamma * variance + float(process_variance)
            if index else variance
        )
        gain = predicted_variance / (
            predicted_variance + float(observation_variance)
        )
        state = predicted + gain * (frame - predicted)
        variance = (1.0 - gain) * predicted_variance
        output[index] = state
    return (output * scale[None]).astype(np.float32)


def patch_positions(length: int, patch: int, stride: int) -> list[int]:
    result = list(range(0, length - patch + 1, stride))
    if not result or result[-1] != length - patch:
        result.append(length - patch)
    return result


def _component_parzen_scores(
    scores,
    *,
    quiet_count: int,
    settings: dict[str, Any],
):
    """Batched local FastICA followed by one batchwise noisy-Parzen posterior."""
    import torch

    batch, frames, rank = scores.shape
    mean = scores.mean(dim=1, keepdim=True)
    std = torch.clamp(scores.std(dim=1, unbiased=True, keepdim=True), min=1e-6)
    z = (scores - mean) / std
    generator = torch.Generator(device=scores.device)
    generator.manual_seed(20260729)
    initial = torch.randn((rank, rank), generator=generator, device=scores.device)
    q, _ = torch.linalg.qr(initial)
    unmixing = q.T[None].repeat(batch, 1, 1)
    for _ in range(int(settings["ica_iterations"])):
        projected = z @ unmixing.transpose(1, 2)
        nonlinear = torch.tanh(projected)
        derivative = (1.0 - nonlinear.square()).mean(dim=1)
        update = nonlinear.transpose(1, 2) @ z / frames - derivative[:, :, None] * unmixing
        eigenvalues, eigenvectors = torch.linalg.eigh(update @ update.transpose(1, 2))
        inverse = eigenvectors @ torch.diag_embed(
            torch.rsqrt(torch.clamp(eigenvalues, min=1e-8))
        ) @ eigenvectors.transpose(1, 2)
        unmixing = inverse @ update
    sources = z @ unmixing.transpose(1, 2)
    source_np = sources.detach().cpu().numpy()
    active = source_np[:, quiet_count:].ravel()
    active = active[np.abs(active) >= 1.5]
    maximum = int(settings["dictionary_centers"])
    zero_count = int(round(maximum * float(settings["dictionary_zero_fraction"])))
    slab_count = maximum - zero_count
    if len(active) < slab_count:
        return scores
    centers = np.concatenate([
        np.zeros(zero_count),
        np.quantile(active, np.linspace(0, 1, slab_count)),
    ])
    grid = np.linspace(
        -float(settings["lookup_abs_z"]), float(settings["lookup_abs_z"]),
        int(settings["lookup_points"]),
    )
    posterior = noisy_parzen_posterior_mean(
        grid, centers, float(settings["bandwidth"]),
        float(settings["noise_variance"]),
    )
    clean_np = np.interp(
        np.clip(source_np, grid[0], grid[-1]), grid, posterior
    ).astype(np.float32)
    clean = torch.as_tensor(clean_np, device=scores.device)
    restored_z = clean @ unmixing
    return restored_z * std + mean


def local_low_rank(
    values: np.ndarray,
    scale: np.ndarray,
    *,
    patch_size: int,
    stride: int,
    rank: int,
    oversample: int,
    batch_size: int,
    device: str,
    noise_normalized: bool,
    component_parzen: dict[str, Any] | None = None,
    quiet_count: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Randomized overlapping local PCA, optionally noise-normalized/Parzen."""
    import torch

    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("local low-rank input must be TYX")
    target = torch.device(
        "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    patch = int(patch_size)
    ys = patch_positions(x.shape[1], patch, int(stride))
    xs = patch_positions(x.shape[2], patch, int(stride))
    positions = [(y, column) for y in ys for column in xs]
    window_1d = np.maximum(np.hanning(patch), 0.1).astype(np.float32)
    window = np.outer(window_1d, window_1d).astype(np.float32)
    output = np.zeros_like(x)
    weight = np.zeros(x.shape[1:], dtype=np.float32)
    generator = torch.Generator(device=target)
    generator.manual_seed(20260729)
    dimensions = patch * patch
    sketch_rank = min(dimensions, int(rank) + int(oversample))
    omega = torch.randn(
        (dimensions, sketch_rank), generator=generator, device=target
    )
    resolved_batch = int(batch_size)
    for batch_start in range(0, len(positions), resolved_batch):
        batch_positions = positions[batch_start:batch_start + resolved_batch]
        blocks = np.stack([
            x[:, y:y + patch, column:column + patch].reshape(len(x), -1)
            for y, column in batch_positions
        ])
        if noise_normalized:
            block_scale = np.stack([
                scale[y:y + patch, column:column + patch].reshape(-1)
                for y, column in batch_positions
            ])
            blocks = blocks / np.maximum(block_scale[:, None], 1e-6)
        tensor = torch.as_tensor(blocks, device=target)
        projection = tensor @ omega
        q, _ = torch.linalg.qr(projection, mode="reduced")
        small = q.transpose(1, 2) @ tensor
        u, singular, vh = torch.linalg.svd(small, full_matrices=False)
        temporal = (q @ u[:, :, : int(rank)]) * singular[:, None, : int(rank)]
        if component_parzen is not None:
            temporal = _component_parzen_scores(
                temporal, quiet_count=quiet_count, settings=component_parzen
            )
        reconstructed = temporal @ vh[:, : int(rank)]
        reconstructed_np = reconstructed.detach().cpu().numpy()
        if noise_normalized:
            reconstructed_np *= block_scale[:, None]
        reconstructed_np = reconstructed_np.reshape(
            len(batch_positions), len(x), patch, patch
        )
        for index, (y, column) in enumerate(batch_positions):
            output[:, y:y + patch, column:column + patch] += (
                reconstructed_np[index] * window[None]
            )
            weight[y:y + patch, column:column + patch] += window
        del tensor, projection, q, small, u, singular, vh, temporal, reconstructed
    output /= np.maximum(weight[None], 1e-6)
    if target.type == "cuda":
        torch.cuda.empty_cache()
    return output.astype(np.float32), {
        "patch_count": len(positions), "patch_size": patch, "stride": int(stride),
        "rank": int(rank), "oversample": int(oversample),
        "noise_normalized": bool(noise_normalized),
        "component_parzen": component_parzen is not None,
        "device": str(target), "batch_size": resolved_batch,
    }
