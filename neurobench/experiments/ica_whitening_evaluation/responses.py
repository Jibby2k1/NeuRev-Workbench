"""Permutation-invariant response and separability summaries."""
from __future__ import annotations

from typing import Any

import numpy as np


def _weighted_moments(energy: np.ndarray, coordinates: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(energy, dtype=np.float64).ravel()
    points = np.asarray(coordinates, dtype=np.float64).ravel()
    total = max(float(weights.sum()), np.finfo(float).eps)
    centroid = float(np.sum(weights * points) / total)
    spread = float(np.sqrt(np.sum(weights * (points - centroid) ** 2) / total))
    return centroid, spread


def temporal_response(kernel: np.ndarray, frame_period_ms: float) -> dict[str, float]:
    values = np.asarray(kernel, dtype=np.float64).ravel()
    response = np.fft.rfft(values, n=max(64, 8 * len(values)))
    frequencies = np.fft.rfftfreq(len(response) * 2 - 2, d=frame_period_ms / 1000.0)
    energy = np.abs(response) ** 2
    centroid, bandwidth = _weighted_moments(energy, frequencies)
    total = max(float(energy.sum()), np.finfo(float).eps)
    nyquist = float(frequencies[-1])
    phase = np.unwrap(np.angle(response))
    angular = 2 * np.pi * frequencies
    group_delay = -np.gradient(phase, angular, edge_order=1)
    valid = frequencies > 0
    delay_seconds = float(np.sum(energy[valid] * group_delay[valid])
                          / max(float(energy[valid].sum()), np.finfo(float).eps))
    normalized = values / max(float(np.linalg.norm(values)), np.finfo(float).eps)
    common = np.ones(len(values)); common /= np.linalg.norm(common)
    difference = np.zeros(len(values)); difference[0], difference[-1] = -1, 1
    difference /= np.linalg.norm(difference)
    def fraction(low: float, high: float) -> float:
        selected = (frequencies >= low * nyquist) & (frequencies < high * nyquist)
        return float(energy[selected].sum() / total)
    return {
        "dc_gain": float(abs(response[0])),
        "peak_frequency_hz": float(frequencies[int(np.argmax(energy))]),
        "frequency_centroid_hz": centroid,
        "frequency_bandwidth_hz": bandwidth,
        "kernel_l2": float(np.linalg.norm(values)),
        "kernel_sum": float(values.sum()),
        "phase_at_peak_radians": float(phase[int(np.argmax(energy))]),
        "energy_weighted_group_delay_seconds": delay_seconds,
        "low_band_energy_fraction": fraction(0.0, 1 / 3),
        "mid_band_energy_fraction": fraction(1 / 3, 2 / 3),
        "high_band_energy_fraction": fraction(2 / 3, 1.000001),
        "absolute_common_mode_cosine": float(abs(normalized @ common)),
        "absolute_signed_difference_cosine": float(abs(normalized @ difference)),
    }


def spatial_response(kernel: np.ndarray, width: int) -> dict[str, float]:
    values = np.asarray(kernel, dtype=np.float64).reshape(width, width)
    response = np.fft.fftshift(np.fft.fft2(values, s=(64, 64)))
    energy = np.abs(response) ** 2
    fy, fx = np.meshgrid(
        np.fft.fftshift(np.fft.fftfreq(64)),
        np.fft.fftshift(np.fft.fftfreq(64)), indexing="ij",
    )
    radius = np.sqrt(fx**2 + fy**2)
    centroid, bandwidth = _weighted_moments(energy, radius)
    covariance_xx = float(np.sum(energy * fx * fx) / max(float(energy.sum()), np.finfo(float).eps))
    covariance_yy = float(np.sum(energy * fy * fy) / max(float(energy.sum()), np.finfo(float).eps))
    covariance_xy = float(np.sum(energy * fx * fy) / max(float(energy.sum()), np.finfo(float).eps))
    anisotropy = abs(covariance_xx - covariance_yy) / max(covariance_xx + covariance_yy, np.finfo(float).eps)
    center = width // 2
    center_value = float(values[center, center])
    surround = float((values.sum() - center_value) / max(width * width - 1, 1))
    yy, xx = np.meshgrid(np.arange(width) - center, np.arange(width) - center, indexing="ij")
    kernel_energy = values**2; kernel_total = max(float(kernel_energy.sum()), np.finfo(float).eps)
    center_x = float(np.sum(kernel_energy * xx) / kernel_total)
    center_y = float(np.sum(kernel_energy * yy) / kernel_total)
    orientation = .5 * np.arctan2(2 * covariance_xy, covariance_xx - covariance_yy)
    return {
        "dc_gain": float(abs(np.fft.fft2(values)[0, 0])),
        "radial_frequency_centroid_cycles_per_px": centroid,
        "radial_frequency_bandwidth_cycles_per_px": bandwidth,
        "frequency_anisotropy": float(anisotropy),
        "center_surround_difference": center_value - surround,
        "kernel_l2": float(np.linalg.norm(values)),
        "kernel_sum": float(values.sum()),
        "orientation_radians_mod_pi": float(orientation),
        "energy_center_offset_px": float(np.hypot(center_x, center_y)),
        "central_energy_fraction": float(kernel_energy[np.hypot(xx, yy) <= max(1, width // 4)].sum() / kernel_total),
        "center_surround_balance": float(center_value / max(abs(surround), np.finfo(float).eps)),
    }


def joint_response(
    kernel: np.ndarray, spatial_width: int, temporal_width: int,
    frame_period_ms: float,
) -> dict[str, float]:
    values = np.asarray(kernel, dtype=np.float64).reshape(
        temporal_width, spatial_width, spatial_width
    )
    matrix = values.reshape(temporal_width, -1)
    singular = np.linalg.svd(matrix, compute_uv=False)
    separable_fraction = float(singular[0] ** 2 / max(float(np.sum(singular**2)), np.finfo(float).eps))
    temporal_kernel = matrix.sum(axis=1)
    spatial_kernel = values.sum(axis=0)
    temporal = temporal_response(temporal_kernel, frame_period_ms)
    spatial = spatial_response(spatial_kernel, spatial_width)
    spectrum = np.abs(np.fft.fftn(
        values, s=(max(32, 4 * temporal_width), 32, 32), axes=(0, 1, 2)
    )) ** 2
    ft = np.fft.fftfreq(spectrum.shape[0], d=frame_period_ms / 1000.0)
    fy = np.fft.fftfreq(spectrum.shape[1]); fx = np.fft.fftfreq(spectrum.shape[2])
    tt, yy, xx = np.meshgrid(ft, fy, fx, indexing="ij")
    radial = np.hypot(xx, yy); valid = radial > 1 / 32
    weights = spectrum[valid]; apparent_speed = np.abs(tt[valid]) / radial[valid]
    speed = float(np.sum(weights * apparent_speed)
                  / max(float(weights.sum()), np.finfo(float).eps))
    return {
        "best_rank1_separable_energy_fraction": separable_fraction,
        "separability_residual_fraction": 1.0 - separable_fraction,
        "temporal_frequency_centroid_hz": temporal["frequency_centroid_hz"],
        "temporal_frequency_bandwidth_hz": temporal["frequency_bandwidth_hz"],
        "spatial_frequency_centroid_cycles_per_px": spatial[
            "radial_frequency_centroid_cycles_per_px"
        ],
        "spatial_frequency_bandwidth_cycles_per_px": spatial[
            "radial_frequency_bandwidth_cycles_per_px"
        ],
        "kernel_l2": float(np.linalg.norm(values)),
        "kernel_sum": float(values.sum()),
        "energy_weighted_apparent_speed_px_per_second": speed,
        "temporal_dc_gain": temporal["dc_gain"],
        "spatial_dc_gain": spatial["dc_gain"],
    }


def component_response_summary(
    demixing: np.ndarray,
    *,
    family: str,
    spatial_width: int | None,
    temporal_width: int | None,
    frame_period_ms: float,
) -> dict[str, Any]:
    values = np.asarray(demixing, dtype=np.float64)
    rows = []
    for index, kernel in enumerate(values):
        if family == "temporal":
            summary = temporal_response(kernel, frame_period_ms)
        elif family == "spatial":
            assert spatial_width is not None
            summary = spatial_response(kernel, spatial_width)
        else:
            assert spatial_width is not None and temporal_width is not None
            summary = joint_response(
                kernel, spatial_width, temporal_width, frame_period_ms
            )
        rows.append({"component": index, **summary})
    numeric_keys = sorted(set.intersection(*(
        {key for key, value in row.items() if key != "component" and isinstance(value, (int, float))}
        for row in rows
    ))) if rows else []
    return {
        "components": rows,
        "permutation_invariant_mean": {
            key: float(np.mean([row[key] for row in rows])) for key in numeric_keys
        },
    }
