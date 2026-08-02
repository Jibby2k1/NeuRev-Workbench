"""Alternative Stage-1 state architectures around one fitted ICA demixer.

ICA determines an affine current-coordinate reconstruction from an aligned
pair. These functions keep that fitted geometry separate from the state
architecture used during inference.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterator

import numpy as np


ARCHITECTURE_IDS = (
    "teacher_forced_stochastic",
    "raw_stochastic_recurrence",
    "quiet_fixed_point_recurrence",
    "reference_parzen_innovation",
)


@dataclass(frozen=True)
class AffineICAReconstruction:
    """Current-coordinate reconstruction induced by a selected ICA source."""

    previous_coefficient: float
    current_coefficient: float
    offset: float

    @classmethod
    def from_feedback(
        cls,
        feedback: dict[str, Any],
    ) -> "AffineICAReconstruction":
        return cls(
            previous_coefficient=float(
                feedback["previous_background_coefficient"]
            ),
            current_coefficient=float(
                feedback["current_observation_coefficient"]
            ),
            offset=float(feedback["offset"]),
        ).validated()

    def validated(self) -> "AffineICAReconstruction":
        values = np.asarray(
            (
                self.previous_coefficient,
                self.current_coefficient,
                self.offset,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("ICA affine coefficients must be finite")
        return self

    def teacher_forced(
        self,
        previous: np.ndarray,
        current: np.ndarray,
    ) -> np.ndarray:
        return (
            self.previous_coefficient * np.asarray(previous, dtype=np.float64)
            + self.current_coefficient * np.asarray(current, dtype=np.float64)
            + self.offset
        )


@dataclass(frozen=True)
class InnovationCalibration:
    """Quiet-only controls for the regularized Parzen innovation lane."""

    quiet_background: np.ndarray
    correction_bias: np.ndarray
    correction_limit: float
    reference_refresh: float
    reference_half_life_seconds: float
    correction_fraction: float
    correction_clip_mad: float
    quiet_correction_mad: float


@dataclass(frozen=True)
class ArchitectureFrame:
    """One aligned scientific output frame."""

    output_index_zero: int
    background: np.ndarray
    dynamics_noise: np.ndarray


def refresh_from_half_life(
    half_life_seconds: float,
    frame_period_ms: float,
) -> float:
    """Convert an EMA half-life to its per-frame refresh coefficient."""
    half_life = float(half_life_seconds)
    period = float(frame_period_ms)
    if not np.isfinite([half_life, period]).all() or min(
        half_life, period
    ) <= 0:
        raise ValueError("half-life and frame period must be finite and positive")
    frames = half_life * 1000.0 / period
    return float(1.0 - math.pow(0.5, 1.0 / frames))


def quiet_median_background(
    frames: np.ndarray,
    quiet_frame_count: int,
) -> np.ndarray:
    """Return the frozen per-pixel median of the quiet source prefix."""
    values = np.asarray(frames)
    count = int(quiet_frame_count)
    if values.ndim != 3 or not 3 <= count <= len(values):
        raise ValueError("quiet background requires TYX frames and count >= 3")
    quiet = np.asarray(values[:count], dtype=np.float64)
    if not np.isfinite(quiet).all():
        raise ValueError("quiet frames must be finite")
    return np.median(quiet, axis=0)


def calibrate_reference_parzen_innovation(
    frames: np.ndarray,
    quiet_frame_count: int,
    coefficients: AffineICAReconstruction,
    *,
    frame_period_ms: float,
    reference_half_life_seconds: float = 10.0,
    correction_fraction: float = 0.1,
    correction_clip_mad: float = 4.0,
    minimum_correction_limit: float = 1e-6,
) -> InnovationCalibration:
    """Freeze a quiet fixed point and bounded zero-quiet Parzen correction."""
    values = np.asarray(frames)
    count = int(quiet_frame_count)
    fraction = float(correction_fraction)
    clip_mad = float(correction_clip_mad)
    minimum_limit = float(minimum_correction_limit)
    if values.ndim != 3 or not 4 <= count <= len(values):
        raise ValueError("innovation calibration requires quiet TYX frames")
    if not (
        0 <= fraction <= 1
        and np.isfinite([clip_mad, minimum_limit]).all()
        and clip_mad > 0
        and minimum_limit > 0
    ):
        raise ValueError("invalid innovation regularization controls")
    coefficients.validated()
    quiet_background = quiet_median_background(values, count)
    refresh = refresh_from_half_life(
        reference_half_life_seconds,
        frame_period_ms,
    )
    reference_state = quiet_background.copy()
    corrections = np.empty(
        (count - 1, *quiet_background.shape),
        dtype=np.float32,
    )
    for output_index, frame_index in enumerate(range(1, count)):
        previous = np.asarray(values[frame_index - 1], dtype=np.float64)
        current = np.asarray(values[frame_index], dtype=np.float64)
        reference_state = (
            (1.0 - refresh) * reference_state + refresh * current
        )
        parzen_background = coefficients.teacher_forced(previous, current)
        corrections[output_index] = (
            parzen_background - reference_state
        ).astype(np.float32)
    correction_bias = np.median(corrections, axis=0).astype(np.float64)
    centered = corrections - correction_bias[None]
    correction_mad = float(
        1.4826 * np.median(np.abs(centered), overwrite_input=True)
    )
    correction_limit = max(
        clip_mad * correction_mad,
        minimum_limit,
    )
    return InnovationCalibration(
        quiet_background=quiet_background,
        correction_bias=correction_bias,
        correction_limit=correction_limit,
        reference_refresh=refresh,
        reference_half_life_seconds=float(reference_half_life_seconds),
        correction_fraction=fraction,
        correction_clip_mad=clip_mad,
        quiet_correction_mad=correction_mad,
    )


def iter_architecture_frames(
    frames: np.ndarray,
    architecture_id: str,
    coefficients: AffineICAReconstruction,
    *,
    quiet_background: np.ndarray,
    innovation: InnovationCalibration | None = None,
) -> Iterator[ArchitectureFrame]:
    """Yield aligned background and signed dynamics/noise without dense copies."""
    values = np.asarray(frames)
    if values.ndim != 3 or len(values) < 4:
        raise ValueError("architecture inference requires TYX frames")
    if architecture_id not in ARCHITECTURE_IDS:
        raise ValueError(f"unknown Stage-1 architecture: {architecture_id}")
    coefficients.validated()
    baseline = np.asarray(quiet_background, dtype=np.float64)
    if baseline.shape != values.shape[1:] or not np.isfinite(baseline).all():
        raise ValueError("quiet background must match the spatial frame shape")
    if (
        architecture_id == "reference_parzen_innovation"
        and innovation is None
    ):
        raise ValueError("reference Parzen innovation calibration is required")

    state = (
        np.asarray(values[0], dtype=np.float64).copy()
        if architecture_id == "raw_stochastic_recurrence"
        else baseline.copy()
    )
    for frame_index in range(1, len(values)):
        previous = np.asarray(values[frame_index - 1], dtype=np.float64)
        current = np.asarray(values[frame_index], dtype=np.float64)
        if architecture_id == "teacher_forced_stochastic":
            background = coefficients.teacher_forced(previous, current)
        elif architecture_id == "raw_stochastic_recurrence":
            background = coefficients.teacher_forced(state, current)
            state = background
        elif architecture_id == "quiet_fixed_point_recurrence":
            background = (
                baseline
                + coefficients.previous_coefficient * (state - baseline)
                + coefficients.current_coefficient * (current - baseline)
            )
            state = background
        else:
            assert innovation is not None
            state = (
                (1.0 - innovation.reference_refresh) * state
                + innovation.reference_refresh * current
            )
            parzen_background = coefficients.teacher_forced(previous, current)
            correction = (
                parzen_background
                - state
                - innovation.correction_bias
            )
            correction = np.clip(
                correction,
                -innovation.correction_limit,
                innovation.correction_limit,
            )
            background = (
                state + innovation.correction_fraction * correction
            )
        dynamics_noise = current - background
        yield ArchitectureFrame(
            output_index_zero=frame_index,
            background=np.asarray(background, dtype=np.float32),
            dynamics_noise=np.asarray(dynamics_noise, dtype=np.float32),
        )
