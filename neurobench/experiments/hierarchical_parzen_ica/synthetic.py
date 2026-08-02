"""Deterministic Stage-1 B/S/A/N fixtures for falsification testing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


STAGE1_SYNTHETIC_CASES = (
    "static_white_noise",
    "gain_scaled_static",
    "linear_background_drift",
    "nonlinear_slow_drift",
    "fast_event",
    "slow_ramp_plateau",
    "similar_persistence",
    "pure_noise",
    "translation_edge",
    "saturation_clipping",
    "heteroscedastic_noise",
    "unresolved_equal_staticness",
)


@dataclass(frozen=True)
class Stage1SyntheticCase:
    case_id: str
    seed: int
    observation: np.ndarray
    background: np.ndarray
    signal: np.ndarray
    artifact: np.ndarray
    noise: np.ndarray
    calibration_frame_count: int
    metadata: dict[str, Any]

    def validate(self) -> None:
        arrays = (
            self.observation,
            self.background,
            self.signal,
            self.artifact,
            self.noise,
        )
        if any(
            value.ndim != 3
            or value.shape != arrays[0].shape
            or not np.isfinite(value).all()
            for value in arrays
        ):
            raise ValueError("synthetic B/S/A/N arrays must be aligned finite TYX")
        closure = self.observation - sum(arrays[1:])
        if np.max(np.abs(closure)) > 1e-12:
            raise ValueError("synthetic fixture violates exact B/S/A/N closure")
        if not 4 <= self.calibration_frame_count < len(self.observation):
            raise ValueError("invalid synthetic calibration prefix")


def _spatial_fields(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:height, 0:width]
    base = (
        0.25
        + 0.35 * x / max(width - 1, 1)
        + 0.20 * y / max(height - 1, 1)
        + 0.12 * np.sin(2 * np.pi * x / max(width, 1))
    )
    center = np.exp(
        -0.5
        * (
            ((y - 0.48 * height) / (0.11 * height)) ** 2
            + ((x - 0.56 * width) / (0.10 * width)) ** 2
        )
    )
    neighbor = np.exp(
        -0.5
        * (
            ((y - 0.35 * height) / (0.13 * height)) ** 2
            + ((x - 0.34 * width) / (0.12 * width)) ** 2
        )
    )
    return base.astype(np.float64), center, neighbor


def _event_trace(case_id: str, frame_count: int, calibration: int) -> np.ndarray:
    trace = np.zeros(frame_count, dtype=np.float64)
    if case_id == "fast_event":
        trace[calibration + 2:calibration + 4] = (0.8, 0.45)
    elif case_id == "slow_ramp_plateau":
        ramp_length = min(5, frame_count - calibration)
        trace[calibration:calibration + ramp_length] = np.linspace(
            0.12, 0.72, ramp_length
        )
        trace[calibration + ramp_length:] = 0.72
    elif case_id == "similar_persistence":
        trace[calibration:] = np.linspace(
            0.08, 0.55, frame_count - calibration
        )
    elif case_id == "saturation_clipping":
        trace[calibration:] = np.linspace(
            0.2, 0.9, frame_count - calibration
        )
    return trace


def generate_stage1_synthetic_case(
    case_id: str,
    seed: int,
    *,
    frame_count: int = 20,
    height: int = 24,
    width: int = 24,
    calibration_frame_count: int = 8,
) -> Stage1SyntheticCase:
    """Generate one exact-closure Stage-1 falsification case."""
    if case_id not in STAGE1_SYNTHETIC_CASES:
        raise ValueError(f"unknown Stage-1 synthetic case: {case_id}")
    if frame_count < calibration_frame_count + 6 or min(height, width) < 8:
        raise ValueError("synthetic fixture dimensions are too small")
    rng = np.random.default_rng(int(seed))
    base, center, neighbor = _spatial_fields(height, width)
    background = np.broadcast_to(
        base, (frame_count, height, width)
    ).copy()
    signal = np.zeros_like(background)
    artifact = np.zeros_like(background)
    noise_sigma = 0.008

    if case_id == "gain_scaled_static":
        gains = np.power(1.012, np.arange(frame_count))
        background *= gains[:, None, None]
    elif case_id == "linear_background_drift":
        drift = np.linspace(0.0, 0.16, frame_count)
        background += drift[:, None, None] * (
            0.4 + 0.6 * np.linspace(0, 1, width)[None, :]
        )
    elif case_id == "nonlinear_slow_drift":
        phase = np.linspace(0, 1.25 * np.pi, frame_count)
        background *= (
            1.0 + 0.08 * np.sin(phase)
        )[:, None, None]
    elif case_id in {
        "fast_event",
        "slow_ramp_plateau",
        "similar_persistence",
        "saturation_clipping",
    }:
        trace = _event_trace(case_id, frame_count, calibration_frame_count)
        signal = trace[:, None, None] * center[None, :, :]
        if case_id == "similar_persistence":
            background += np.linspace(
                0.0, 0.22, frame_count
            )[:, None, None] * neighbor[None, :, :]
    elif case_id == "pure_noise":
        background.fill(0.0)
        noise_sigma = 0.04
    elif case_id == "translation_edge":
        shifted = np.roll(base, shift=1, axis=1)
        artifact[calibration_frame_count:] = shifted - base
    elif case_id == "heteroscedastic_noise":
        noise_sigma = 0.0
    elif case_id == "unresolved_equal_staticness":
        phase = np.linspace(0, 4 * np.pi, frame_count)
        background += (
            0.06 * np.sin(phase)[:, None, None] * center[None, :, :]
        )
        artifact += (
            0.06 * np.cos(phase)[:, None, None] * neighbor[None, :, :]
        )

    if case_id == "heteroscedastic_noise":
        sigma = 0.004 + 0.025 * np.sqrt(np.maximum(background, 0))
        noise = rng.normal(size=background.shape) * sigma
    else:
        noise = noise_sigma * rng.normal(size=background.shape)

    if case_id == "saturation_clipping":
        unclipped = background + signal + noise
        clipped = np.clip(unclipped, 0.0, 1.0)
        artifact = clipped - unclipped

    observation = background + signal + artifact + noise
    result = Stage1SyntheticCase(
        case_id=case_id,
        seed=int(seed),
        observation=observation,
        background=background,
        signal=signal,
        artifact=artifact,
        noise=noise,
        calibration_frame_count=int(calibration_frame_count),
        metadata={
            "axes": "TYX",
            "units": "synthetic_intensity",
            "frame_count": int(frame_count),
            "height": int(height),
            "width": int(width),
            "signal_present": bool(np.any(signal)),
            "artifact_present": bool(np.any(artifact)),
            "expected_unresolved": case_id == "unresolved_equal_staticness",
            "noise_kind": (
                "heteroscedastic_gaussian"
                if case_id == "heteroscedastic_noise"
                else "white_gaussian"
            ),
            "labels_used_for_fit": False,
        },
    )
    result.validate()
    return result


def stage1_synthetic_suite(
    seeds: Iterable[int],
    *,
    case_ids: Iterable[str] = STAGE1_SYNTHETIC_CASES,
) -> tuple[Stage1SyntheticCase, ...]:
    """Return a deterministic bounded cross-product of cases and seeds."""
    normalized_seeds = tuple(int(seed) for seed in seeds)
    normalized_cases = tuple(str(case_id) for case_id in case_ids)
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("synthetic seeds must be non-empty and unique")
    if not normalized_cases or len(set(normalized_cases)) != len(normalized_cases):
        raise ValueError("synthetic cases must be non-empty and unique")
    return tuple(
        generate_stage1_synthetic_case(case_id, seed)
        for case_id in normalized_cases
        for seed in normalized_seeds
    )
