"""Deterministic truth-known fixtures for source-separation development."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


FixtureCase = Literal[
    "isolated",
    "overlap",
    "synchronous",
    "correlated",
    "fast_onset",
    "slow_plateau",
    "similar_persistence",
    "illumination_drift",
    "motion_edge",
    "saturation",
    "heteroscedastic_noise",
    "pure_noise",
    "unresolved",
]


@dataclass(frozen=True)
class SpatiotemporalFixture:
    case_id: str
    observation: np.ndarray
    neural_signal: np.ndarray
    background: np.ndarray
    structured_artifact: np.ndarray
    measurement_noise: np.ndarray
    footprints: np.ndarray
    traces: np.ndarray
    frame_period_ms: float
    identifiable: bool
    metadata: dict[str, object]


def _gaussian(shape: tuple[int, int], x: float, y: float, sigma: float) -> np.ndarray:
    rows, columns = np.indices(shape)
    values = np.exp(-0.5 * (((columns - x) / sigma) ** 2 + ((rows - y) / sigma) ** 2))
    return values / max(float(values.max()), np.finfo(float).eps)


def _ring(shape: tuple[int, int], x: float, y: float, radius: float, width: float) -> np.ndarray:
    rows, columns = np.indices(shape)
    distance = np.sqrt((columns - x) ** 2 + (rows - y) ** 2)
    values = np.exp(-0.5 * ((distance - radius) / width) ** 2)
    return values / max(float(values.max()), np.finfo(float).eps)


def _calcium_trace(
    count: int,
    events: list[tuple[int, float]],
    *,
    decay: float,
    rise: float = 1.0,
) -> np.ndarray:
    impulses = np.zeros(count, dtype=np.float64)
    for frame, amplitude in events:
        if 0 <= frame < count:
            impulses[frame] += float(amplitude)
    trace = np.zeros(count, dtype=np.float64)
    state = 0.0
    for frame in range(count):
        state = decay * state + rise * impulses[frame]
        trace[frame] = state
    return trace


def make_spatiotemporal_fixture(
    case_id: FixtureCase,
    *,
    seed: int,
    frame_count: int = 256,
    shape: tuple[int, int] = (16, 16),
    snr: float = 8.0,
    frame_period_ms: float = 20.0,
) -> SpatiotemporalFixture:
    """Create a B/S/A/N fixture with exact additive closure."""
    if case_id not in FixtureCase.__args__:
        raise ValueError(f"unknown fixture case: {case_id}")
    if frame_count < 96 or min(shape) < 8 or snr <= 0 or frame_period_ms <= 0:
        raise ValueError("fixture dimensions, SNR, and frame period are invalid")
    rng = np.random.default_rng(int(seed))
    height, width = shape
    if case_id == "overlap":
        centers = ((0.48 * width, 0.48 * height), (0.52 * width, 0.54 * height), (0.72 * width, 0.30 * height))
    else:
        centers = ((0.28 * width, 0.32 * height), (0.68 * width, 0.38 * height), (0.52 * width, 0.72 * height))
    footprints = np.stack([
        _gaussian(shape, *centers[0], sigma=1.25),
        _ring(shape, *centers[1], radius=1.7, width=0.65),
        _gaussian(shape, *centers[2], sigma=1.5),
    ])
    base_events = [
        [(64, 1.0), (154, 0.75)],
        [(91, 0.85), (184, 1.0)],
        [(121, 1.1), (210, 0.7)],
    ]
    if case_id == "synchronous":
        base_events = [[(80, 1.0), (176, 0.8)], [(80, 0.9), (176, 1.0)], [(80, 1.1), (176, 0.7)]]
    elif case_id == "fast_onset":
        base_events[0] = [(38, 1.2), (42, 0.5), (160, 0.8)]
    traces = np.stack([
        _calcium_trace(frame_count, events, decay=decay)
        for events, decay in zip(base_events, (0.90, 0.94, 0.97))
    ])
    if case_id == "correlated":
        traces[1] = 0.65 * traces[0] + 0.35 * traces[1]
    elif case_id == "slow_plateau":
        traces[2, 72:168] += np.linspace(0.0, 0.8, 96)
        traces[2, 168:220] += 0.8
    elif case_id in {"similar_persistence", "unresolved"}:
        traces[0] = _calcium_trace(frame_count, [(58, 1.0), (164, 0.8)], decay=0.992)
    elif case_id == "pure_noise":
        traces[:] = 0.0
    neural_signal = np.einsum("st,shw->thw", traces, footprints)

    rows, columns = np.indices(shape)
    broad = 0.8 + 0.15 * columns / max(width - 1, 1) + 0.1 * rows / max(height - 1, 1)
    anatomy = 0.35 * _gaussian(shape, 0.5 * width, 0.5 * height, sigma=0.32 * min(shape))
    time = np.arange(frame_count, dtype=np.float64)
    drift = 1.0 + 0.04 * np.sin(2 * np.pi * time / frame_count)
    if case_id == "illumination_drift":
        drift += 0.20 * time / max(frame_count - 1, 1)
    if case_id in {"similar_persistence", "unresolved"}:
        drift += 0.35 * _calcium_trace(frame_count, [(55, 1.0), (160, 0.8)], decay=0.992)
    background = drift[:, None, None] * (broad + anatomy)[None, :, :]

    artifact = np.zeros_like(background)
    if case_id == "motion_edge":
        shifted = np.roll(broad + anatomy, 1, axis=1)
        artifact[104:116] = shifted - (broad + anatomy)
    if case_id == "saturation":
        artifact[150:166, 2:6, 2:7] += 1.8

    signal_scale = max(float(np.std(neural_signal)), 0.05)
    noise_scale = signal_scale / float(snr)
    if case_id == "heteroscedastic_noise":
        local_scale = noise_scale * np.sqrt(np.maximum(background + neural_signal, 0.05))
    else:
        local_scale = np.full_like(background, noise_scale)
    measurement_noise = rng.normal(size=background.shape) * local_scale
    observation = background + neural_signal + artifact + measurement_noise
    clipped_fraction = 0.0
    if case_id == "saturation":
        ceiling = float(np.quantile(observation, 0.985))
        clipped = np.minimum(observation, ceiling)
        clipping_error = clipped - observation
        artifact = artifact + clipping_error
        observation = clipped
        clipped_fraction = float(np.mean(clipping_error != 0))
    identifiable = case_id not in {"pure_noise", "unresolved"}
    closure = observation - background - neural_signal - artifact - measurement_noise
    return SpatiotemporalFixture(
        case_id=case_id,
        observation=observation.astype(np.float32),
        neural_signal=neural_signal.astype(np.float32),
        background=background.astype(np.float32),
        structured_artifact=artifact.astype(np.float32),
        measurement_noise=measurement_noise.astype(np.float32),
        footprints=footprints.astype(np.float32),
        traces=traces.astype(np.float32),
        frame_period_ms=float(frame_period_ms),
        identifiable=identifiable,
        metadata={
            "seed": int(seed),
            "snr": float(snr),
            "shape": list(shape),
            "frame_count": int(frame_count),
            "source_count": int(len(traces)),
            "clipped_fraction": clipped_fraction,
            "maximum_closure_absolute": float(np.max(np.abs(closure))),
        },
    )
