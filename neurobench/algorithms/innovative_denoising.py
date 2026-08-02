"""Bounded, carrier-preserving denoisers for the Spon Ca Burst v3 program.

The accepted Parzen Innovation residual is always the carrier.  Every method
in this module either returns that carrier unchanged or applies a finite,
auditable correction to it.  This makes zero authority a common identity
baseline and keeps combinations from silently replacing the measured signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Sequence

import numpy as np

from neurobench.algorithms.advanced_denoising import carrier_blend


def _video(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 3 or not result.size or not np.isfinite(result).all():
        raise ValueError("values must be a finite non-empty TYX array")
    return result


def _starts(length: int, tile: int, step: int) -> list[int]:
    if tile >= length:
        return [0]
    values = list(range(0, length - tile + 1, step))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def local_noise_psd_wiener(
    values: np.ndarray,
    *,
    quiet_count: int,
    tile_size: int,
    overlap_fraction: float,
    noise_multiplier: float,
    frequency_smoothing_sigma: float,
    transfer_floor: float,
    alpha: float,
) -> np.ndarray:
    """Overlap-add local anisotropic Wiener filtering from quiet-frame PSDs."""
    from scipy.ndimage import gaussian_filter

    video = _video(values)
    quiet = int(quiet_count)
    tile = int(tile_size)
    overlap = float(overlap_fraction)
    floor = float(transfer_floor)
    if not 2 <= quiet < len(video):
        raise ValueError("quiet_count must define a strict leading subset")
    if tile < 16 or not 0 <= overlap < 0.9 or not 0 <= floor <= 1:
        raise ValueError("invalid local PSD geometry or transfer floor")
    tile_y = min(tile, video.shape[1])
    tile_x = min(tile, video.shape[2])
    step_y = max(1, int(round(tile_y * (1.0 - overlap))))
    step_x = max(1, int(round(tile_x * (1.0 - overlap))))
    ys = _starts(video.shape[1], tile_y, step_y)
    xs = _starts(video.shape[2], tile_x, step_x)
    wy = np.maximum(np.hanning(tile_y), 0.1).astype(np.float32)
    wx = np.maximum(np.hanning(tile_x), 0.1).astype(np.float32)
    window = np.outer(wy, wx).astype(np.float32)
    output = np.zeros_like(video)
    weight = np.zeros(video.shape[1:], dtype=np.float32)
    sigma = float(frequency_smoothing_sigma)
    for y0, x0 in product(ys, xs):
        block = video[:, y0 : y0 + tile_y, x0 : x0 + tile_x]
        quiet_fft = np.fft.rfft2(block[:quiet], axes=(-2, -1))
        total_fft = np.fft.rfft2(block, axes=(-2, -1))
        noise_psd = np.mean(np.abs(quiet_fft) ** 2, axis=0)
        total_psd = np.mean(np.abs(total_fft) ** 2, axis=0)
        signal_psd = np.maximum(
            total_psd - float(noise_multiplier) * noise_psd, 0
        )
        transfer = signal_psd / np.maximum(signal_psd + noise_psd, 1e-12)
        if sigma > 0:
            transfer = gaussian_filter(transfer, sigma=sigma, mode="nearest")
        transfer = floor + (1.0 - floor) * np.clip(transfer, 0, 1)
        filtered = np.fft.irfft2(
            total_fft * transfer[None],
            s=(tile_y, tile_x),
            axes=(-2, -1),
        ).real.astype(np.float32)
        output[:, y0 : y0 + tile_y, x0 : x0 + tile_x] += (
            filtered * window[None]
        )
        weight[y0 : y0 + tile_y, x0 : x0 + tile_x] += window
    output /= np.maximum(weight[None], 1e-6)
    return carrier_blend(video, output, float(alpha))


def morphology_conditioned_shrinkage(
    values: np.ndarray,
    *,
    center_sigma_px: float,
    ring_sigma_px: float,
    crowd_sigma_px: float,
    isolated_threshold_z: float,
    crowded_threshold_z: float,
    gate_temperature_z: float,
    gain_floor: float,
    alpha: float,
) -> np.ndarray:
    """Soft four-case gate for isolated/crowded center and membrane signals."""
    from scipy.ndimage import gaussian_filter
    from scipy.special import expit

    video = _video(values)
    center_sigma = float(center_sigma_px)
    ring_sigma = float(ring_sigma_px)
    crowd_sigma = float(crowd_sigma_px)
    temperature = float(gate_temperature_z)
    floor = float(gain_floor)
    if not 0 < center_sigma < ring_sigma < crowd_sigma:
        raise ValueError("morphology scales must be strictly increasing")
    if temperature <= 0 or not 0 <= floor <= 1:
        raise ValueError("invalid morphology gate temperature or gain floor")
    positive = np.maximum(video, 0)
    center = gaussian_filter(
        positive, sigma=(0, center_sigma, center_sigma), mode="reflect"
    )
    ring_blur = gaussian_filter(
        positive, sigma=(0, ring_sigma, ring_sigma), mode="reflect"
    )
    crowd_blur = gaussian_filter(
        positive, sigma=(0, crowd_sigma, crowd_sigma), mode="reflect"
    )
    membrane = np.maximum(ring_blur - 0.55 * center, 0)
    crowd = expit(
        (crowd_blur - float(crowded_threshold_z)) / temperature
    ).astype(np.float32)
    isolated = 1.0 - crowd
    centered_isolated = isolated * center
    membrane_isolated = isolated * membrane
    centered_crowded = crowd * center
    membrane_crowded = crowd * membrane
    evidence = np.maximum.reduce(
        (
            centered_isolated,
            membrane_isolated,
            centered_crowded,
            membrane_crowded,
        )
    )
    threshold = (
        isolated * float(isolated_threshold_z)
        + crowd * float(crowded_threshold_z)
    )
    gate = expit((evidence - threshold) / temperature).astype(np.float32)
    gate = floor + (1.0 - floor) * gate
    return carrier_blend(video, video * gate, float(alpha))


def selected_component_nmf(
    values: np.ndarray,
    *,
    window_frames: int,
    rank: int,
    iterations: int,
    minimum_spatial_concentration: float,
    minimum_temporal_dynamics: float,
    selection_temperature: float,
    alpha: float,
    seed: int,
    device: str = "cpu",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Windowed NMF with explicit soft neural/background component selection."""
    import torch

    video = _video(values)
    window = int(window_frames)
    selected_rank = int(rank)
    temperature = float(selection_temperature)
    if not 4 <= window <= len(video) or not 1 <= selected_rank < window:
        raise ValueError("invalid selected-NMF window or rank")
    if temperature <= 0:
        raise ValueError("selection_temperature must be positive")
    target = torch.device(
        "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    step = max(1, window // 2)
    starts = _starts(len(video), window, step)
    output = np.zeros_like(video)
    weight = np.zeros(len(video), dtype=np.float32)
    taper = np.maximum(np.hanning(window), 0.1).astype(np.float32)
    generator = torch.Generator(device=target)
    generator.manual_seed(int(seed))
    epsilon = 1e-6
    keep_values: list[float] = []
    concentration_values: list[float] = []
    dynamics_values: list[float] = []
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
            top_count = max(1, int(round(0.01 * spatial.shape[1])))
            concentration = torch.topk(
                spatial, top_count, dim=1
            ).values.sum(dim=1) / torch.clamp(spatial.sum(dim=1), min=epsilon)
            temporal_centered = temporal - temporal.mean(dim=0, keepdim=True)
            dynamics = torch.std(
                temporal[1:] - temporal[:-1], dim=0
            ) / torch.clamp(torch.std(temporal_centered, dim=0), min=epsilon)
            keep = torch.sigmoid(
                (concentration - float(minimum_spatial_concentration))
                / temperature
            ) * torch.sigmoid(
                (dynamics - float(minimum_temporal_dynamics)) / temperature
            )
            estimate = (
                (temporal * keep[None]) @ spatial
            ).reshape(window, video.shape[1], video.shape[2]).cpu().numpy()
            output[start : start + window] += estimate * taper[:, None, None]
            weight[start : start + window] += taper
            keep_values.extend(keep.cpu().tolist())
            concentration_values.extend(concentration.cpu().tolist())
            dynamics_values.extend(dynamics.cpu().tolist())
    output /= np.maximum(weight[:, None, None], 1e-6)
    target_signal = np.minimum(video, 0) + output
    if target.type == "cuda":
        torch.cuda.empty_cache()
    return carrier_blend(video, target_signal, float(alpha)), {
        "component_keep_mean": float(np.mean(keep_values)),
        "component_keep_fraction_ge_half": float(
            np.mean(np.asarray(keep_values) >= 0.5)
        ),
        "spatial_concentration_mean": float(np.mean(concentration_values)),
        "temporal_dynamics_mean": float(np.mean(dynamics_values)),
    }


def tempered_residual_posterior(
    carrier: np.ndarray,
    posterior_estimate: np.ndarray,
    *,
    activity_threshold_z: float,
    temperature_z: float,
    posterior_authority: float,
    correction_limit_z: float,
) -> np.ndarray:
    """Temper a Parzen correction as activity confidence increases."""
    from scipy.special import expit

    source = _video(carrier)
    posterior = _video(posterior_estimate)
    if source.shape != posterior.shape or float(temperature_z) <= 0:
        raise ValueError("posterior must align and temperature must be positive")
    if not 0 <= float(posterior_authority) <= 1:
        raise ValueError("posterior_authority must be in [0,1]")
    activity_confidence = expit(
        (np.abs(source) - float(activity_threshold_z)) / float(temperature_z)
    ).astype(np.float32)
    quiet_authority = float(posterior_authority) * (1.0 - activity_confidence)
    correction = np.clip(
        posterior - source,
        -float(correction_limit_z),
        float(correction_limit_z),
    )
    return (source + quiet_authority * correction).astype(np.float32)


def graph_edge_aware_diffusion(
    values: np.ndarray,
    *,
    quiet_count: int,
    radius: int,
    signal_bandwidth_z: float,
    guide_bandwidth_z: float,
    iterations: int,
    alpha: float,
    device: str = "cpu",
    frame_batch_size: int = 4,
) -> np.ndarray:
    """Edge-aware graph diffusion guided by quiet anatomy and current signal."""
    import torch
    import torch.nn.functional as functional

    video = _video(values)
    graph_radius = int(radius)
    if not 1 <= graph_radius <= 2 or not 1 <= int(iterations) <= 3:
        raise ValueError("invalid graph radius or iteration count")
    signal_scale = max(float(signal_bandwidth_z) ** 2, 1e-6)
    guide_scale = max(float(guide_bandwidth_z) ** 2, 1e-6)
    target = torch.device(
        "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    guide = torch.as_tensor(
        np.median(video[: int(quiet_count)], axis=0)[None, None],
        device=target,
    )
    output = np.empty_like(video)
    batch = max(1, int(frame_batch_size))
    with torch.inference_mode():
        for start in range(0, len(video), batch):
            stop = min(len(video), start + batch)
            current = torch.as_tensor(video[start:stop, None], device=target)
            for _ in range(int(iterations)):
                padded = functional.pad(
                    current,
                    (
                        graph_radius,
                        graph_radius,
                        graph_radius,
                        graph_radius,
                    ),
                    mode="reflect",
                )
                guide_padded = functional.pad(
                    guide,
                    (
                        graph_radius,
                        graph_radius,
                        graph_radius,
                        graph_radius,
                    ),
                    mode="reflect",
                )
                numerator = current.clone()
                denominator = torch.ones_like(current)
                for dy in range(-graph_radius, graph_radius + 1):
                    for dx in range(-graph_radius, graph_radius + 1):
                        if dy == 0 and dx == 0:
                            continue
                        shifted = padded[
                            :,
                            :,
                            graph_radius + dy : graph_radius + dy + current.shape[-2],
                            graph_radius + dx : graph_radius + dx + current.shape[-1],
                        ]
                        shifted_guide = guide_padded[
                            :,
                            :,
                            graph_radius + dy : graph_radius + dy + current.shape[-2],
                            graph_radius + dx : graph_radius + dx + current.shape[-1],
                        ]
                        edge_weight = torch.exp(
                            -(current - shifted).square() / signal_scale
                            - (guide - shifted_guide).square() / guide_scale
                        )
                        numerator += edge_weight * shifted
                        denominator += edge_weight
                current = numerator / torch.clamp(denominator, min=1e-6)
            output[start:stop] = current[:, 0].cpu().numpy()
    if target.type == "cuda":
        torch.cuda.empty_cache()
    return carrier_blend(video, output, float(alpha))


def cross_scale_consensus_shrinkage(
    values: np.ndarray,
    *,
    spatial_scales_px: Sequence[float],
    agreement_power: float,
    evidence_threshold_z: float,
    gain_floor: float,
    alpha: float,
) -> np.ndarray:
    """Retain pixels supported with consistent sign across spatial scales."""
    from scipy.ndimage import gaussian_filter
    from scipy.special import expit

    video = _video(values)
    scales = tuple(float(value) for value in spatial_scales_px)
    if len(scales) < 2 or any(value <= 0 for value in scales):
        raise ValueError("at least two positive spatial scales are required")
    floor = float(gain_floor)
    if not 0 <= floor <= 1 or float(agreement_power) <= 0:
        raise ValueError("invalid consensus gain parameters")
    estimates = np.stack(
        [
            gaussian_filter(video, sigma=(0, scale, scale), mode="reflect")
            for scale in scales
        ]
    )
    sign_mean = np.abs(np.mean(np.sign(estimates), axis=0))
    agreement = np.power(sign_mean, float(agreement_power))
    evidence = np.median(np.abs(estimates), axis=0)
    strength = expit(
        (evidence - float(evidence_threshold_z))
        / max(0.25 * float(evidence_threshold_z), 0.1)
    )
    gain = floor + (1.0 - floor) * agreement * strength
    return carrier_blend(video, video * gain.astype(np.float32), float(alpha))


@dataclass(frozen=True)
class BlindSpotLinearModel:
    offsets: tuple[tuple[int, int], ...]
    weights: np.ndarray
    intercept: float
    radius: int
    fit_mse: float


def fit_blindspot_linear_model(
    values: np.ndarray,
    *,
    radius: int,
    sample_count: int,
    ridge: float,
    seed: int,
    fit_frame_count: int | None = None,
) -> BlindSpotLinearModel:
    """Noise2Self-style linear center prediction without a center-pixel input."""
    video = _video(values)
    selected_radius = int(radius)
    if not 1 <= selected_radius <= 2 or int(sample_count) < 256:
        raise ValueError("invalid blind-spot radius or sample count")
    offsets = tuple(
        (dy, dx)
        for dy in range(-selected_radius, selected_radius + 1)
        for dx in range(-selected_radius, selected_radius + 1)
        if (dy, dx) != (0, 0)
    )
    frames = len(video) if fit_frame_count is None else int(fit_frame_count)
    if not 2 <= frames <= len(video):
        raise ValueError("fit_frame_count must be within the video")
    rng = np.random.default_rng(int(seed))
    count = int(sample_count)
    ts = rng.integers(0, frames, size=count)
    ys = rng.integers(
        selected_radius, video.shape[1] - selected_radius, size=count
    )
    xs = rng.integers(
        selected_radius, video.shape[2] - selected_radius, size=count
    )
    design = np.stack(
        [video[ts, ys + dy, xs + dx] for dy, dx in offsets], axis=1
    ).astype(np.float64)
    target = video[ts, ys, xs].astype(np.float64)
    mean = design.mean(axis=0)
    scale = np.maximum(design.std(axis=0), 1e-6)
    normalized = (design - mean[None]) / scale[None]
    target_mean = float(target.mean())
    centered_target = target - target_mean
    gram = normalized.T @ normalized
    gram.flat[:: len(offsets) + 1] += float(ridge) * count
    normalized_weights = np.linalg.solve(gram, normalized.T @ centered_target)
    weights = normalized_weights / scale
    l1 = float(np.sum(np.abs(weights)))
    if l1 > 2:
        weights *= 2 / l1
    intercept = target_mean - float(mean @ weights)
    prediction = design @ weights + intercept
    return BlindSpotLinearModel(
        offsets=offsets,
        weights=weights.astype(np.float32),
        intercept=float(intercept),
        radius=selected_radius,
        fit_mse=float(np.mean((prediction - target) ** 2)),
    )


def apply_blindspot_linear_model(
    values: np.ndarray,
    model: BlindSpotLinearModel,
    *,
    alpha: float,
    correction_limit_z: float,
    device: str = "cpu",
    frame_batch_size: int = 4,
) -> np.ndarray:
    """Apply a fitted blind-spot predictor as a bounded carrier correction."""
    import torch
    import torch.nn.functional as functional

    video = _video(values)
    target = torch.device(
        "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    size = 2 * model.radius + 1
    kernel = torch.zeros((1, 1, size, size), device=target)
    for (dy, dx), weight in zip(model.offsets, model.weights):
        kernel[0, 0, dy + model.radius, dx + model.radius] = float(weight)
    output = np.empty_like(video)
    batch = max(1, int(frame_batch_size))
    with torch.inference_mode():
        for start in range(0, len(video), batch):
            stop = min(len(video), start + batch)
            source = torch.as_tensor(video[start:stop, None], device=target)
            padded = functional.pad(
                source,
                (model.radius, model.radius, model.radius, model.radius),
                mode="reflect",
            )
            prediction = functional.conv2d(padded, kernel) + float(model.intercept)
            output[start:stop] = prediction[:, 0].cpu().numpy()
    if target.type == "cuda":
        torch.cuda.empty_cache()
    return carrier_blend(
        video,
        output,
        float(alpha),
        correction_limit_z=float(correction_limit_z),
    )


def bounded_mixture(
    carrier: np.ndarray,
    estimates: Sequence[np.ndarray],
    weights: Sequence[float],
    *,
    correction_limit_z: float,
) -> np.ndarray:
    """Combine denoiser corrections with a nonnegative bounded authority budget."""
    source = _video(carrier)
    candidates = [_video(value) for value in estimates]
    coefficients = np.asarray(weights, dtype=np.float64)
    if (
        not candidates
        or len(candidates) != len(coefficients)
        or any(value.shape != source.shape for value in candidates)
        or np.any(coefficients < 0)
        or float(coefficients.sum()) > 1.000001
    ):
        raise ValueError("mixture estimates/weights are invalid or exceed authority one")
    correction = np.zeros_like(source)
    for estimate, coefficient in zip(candidates, coefficients):
        correction += float(coefficient) * (estimate - source)
    correction = np.clip(
        correction,
        -float(correction_limit_z),
        float(correction_limit_z),
    )
    return (source + correction).astype(np.float32)


def pareto_front_indices(
    rows: Sequence[dict[str, Any]],
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
) -> list[int]:
    """Return stable indices not dominated across the declared objectives."""
    if not rows:
        return []
    keys = tuple(maximize) + tuple(minimize)
    if not keys:
        raise ValueError("at least one Pareto objective is required")
    values = np.asarray(
        [
            [
                float(row[key]) if key in maximize else -float(row[key])
                for key in keys
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("Pareto objectives must be finite")
    keep: list[int] = []
    for index in range(len(rows)):
        dominated = False
        for challenger in range(len(rows)):
            if challenger == index:
                continue
            if np.all(values[challenger] >= values[index]) and np.any(
                values[challenger] > values[index]
            ):
                dominated = True
                break
        if not dominated:
            keep.append(index)
    return keep


def select_diverse_pareto_rows(
    rows: Sequence[dict[str, Any]],
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
    count: int,
) -> list[dict[str, Any]]:
    """Select a deterministic, normalized-objective diverse Pareto subset."""
    front = [rows[index] for index in pareto_front_indices(
        rows, maximize=maximize, minimize=minimize
    )]
    if len(front) <= int(count):
        remainder = [row for row in rows if row not in front]
        remainder = sorted(
            remainder,
            key=lambda row: float(row.get("selection_score", 0)),
            reverse=True,
        )
        return list(front) + remainder[: max(0, int(count) - len(front))]
    keys = tuple(maximize) + tuple(minimize)
    matrix = np.asarray(
        [
            [
                float(row[key]) if key in maximize else -float(row[key])
                for key in keys
            ]
            for row in front
        ],
        dtype=np.float64,
    )
    span = np.maximum(matrix.max(axis=0) - matrix.min(axis=0), 1e-12)
    normalized = (matrix - matrix.min(axis=0)) / span
    chosen = [
        int(np.argmax(np.mean(normalized, axis=1)))
    ]
    while len(chosen) < int(count):
        distances = np.min(
            np.linalg.norm(
                normalized[:, None, :] - normalized[chosen][None, :, :],
                axis=2,
            ),
            axis=1,
        )
        distances[chosen] = -1
        chosen.append(int(np.argmax(distances)))
    return [front[index] for index in chosen]
