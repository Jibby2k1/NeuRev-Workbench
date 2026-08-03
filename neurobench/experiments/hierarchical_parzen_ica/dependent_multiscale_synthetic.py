"""Deterministic exact-truth fixtures for dependent multiscale demixing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import shift


FIXTURE_IDS = (
    "compact_isolated_center", "compact_isolated_annulus",
    "broad_legitimate_neural_source", "compact_source_on_broad_drift",
    "two_correlated_neurons", "population_burst_plus_private_activity",
    "overlapping_center_and_membrane_sources", "motion_edge_without_neural_activity",
    "motion_edge_crossing_a_neuron", "multiplicative_illumination_drift",
    "heteroscedastic_shot_like_noise", "correlated_multiscale_noise",
    "pure_quiet_noise", "clipping_or_saturation", "patch_boundary_source",
)


@dataclass(frozen=True)
class SyntheticDecompositionFixture:
    fixture_id: str
    background: np.ndarray
    structured_signal: np.ndarray
    structured_artifact: np.ndarray
    noise: np.ndarray
    observation: np.ndarray
    diagnostics: dict[str, object]


def _footprint(y: np.ndarray, x: np.ndarray, cy: float, cx: float, sigma: float) -> np.ndarray:
    return np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma**2))


def make_fixture(fixture_id: str, *, seed: int = 7) -> SyntheticDecompositionFixture:
    """Return one small exact B/S/A/N fixture without labels in its construction."""
    if fixture_id not in FIXTURE_IDS:
        raise ValueError(f"unknown fixture_id: {fixture_id}")
    rng = np.random.default_rng(int(seed))
    t_count, y_size, x_size = 24, 25, 25
    time = np.arange(t_count, dtype=np.float64)
    y, x = np.mgrid[:y_size, :x_size]
    background = np.broadcast_to(
        0.15 + 0.002 * time[:, None, None] + 0.0015 * x[None],
        (t_count, y_size, x_size),
    ).copy()
    signal = np.zeros_like(background)
    artifact = np.zeros_like(background)
    noise_scale = 0.015
    event = np.exp(-0.5 * ((time - 14) / 2.0) ** 2)
    center = _footprint(y, x, 12, 12, 1.6)
    annulus = np.exp(-0.5 * ((np.hypot(y - 12, x - 12) - 3.0) / 0.7) ** 2)

    if fixture_id == "compact_isolated_center":
        signal = 0.8 * event[:, None, None] * center
    elif fixture_id == "compact_isolated_annulus":
        signal = 0.7 * event[:, None, None] * annulus
    elif fixture_id == "broad_legitimate_neural_source":
        signal = 0.75 * event[:, None, None] * _footprint(y, x, 12, 12, 5.0)
    elif fixture_id == "compact_source_on_broad_drift":
        background += 0.2 * np.sin(time[:, None, None] / 8) * _footprint(y, x, 12, 12, 8)
        signal = 0.8 * event[:, None, None] * center
    elif fixture_id == "two_correlated_neurons":
        shared = event + 0.4 * np.exp(-0.5 * ((time - 18) / 1.5) ** 2)
        signal = shared[:, None, None] * (
            0.7 * _footprint(y, x, 9, 9, 1.5) + 0.6 * _footprint(y, x, 16, 16, 1.7)
        )
    elif fixture_id == "population_burst_plus_private_activity":
        private = np.exp(-0.5 * ((time - 9) / 1.2) ** 2)
        signal = event[:, None, None] * (
            _footprint(y, x, 8, 8, 1.4) + _footprint(y, x, 17, 16, 1.5)
        ) + 0.5 * private[:, None, None] * _footprint(y, x, 8, 8, 1.4)
    elif fixture_id == "overlapping_center_and_membrane_sources":
        signal = event[:, None, None] * (0.6 * center + 0.5 * annulus)
    elif fixture_id in {"motion_edge_without_neural_activity", "motion_edge_crossing_a_neuron"}:
        edge = (x > 12).astype(float)
        for index in range(t_count):
            artifact[index] = 0.15 * shift(edge, (0, 0.2 * np.sin(index / 2)), order=1, mode="nearest")
        if fixture_id.endswith("crossing_a_neuron"):
            signal = 0.7 * event[:, None, None] * center
    elif fixture_id == "multiplicative_illumination_drift":
        background *= (1 + 0.25 * np.sin(time / 7))[:, None, None]
    elif fixture_id == "heteroscedastic_shot_like_noise":
        noise_scale = 0.025
        signal = 0.6 * event[:, None, None] * center
    elif fixture_id == "correlated_multiscale_noise":
        common = rng.normal(0, 0.012, size=background.shape)
        noise = common + 0.5 * np.roll(common, 1, axis=2)
        observation = background + signal + artifact + noise
        return SyntheticDecompositionFixture(
            fixture_id, background.astype(np.float32), signal.astype(np.float32),
            artifact.astype(np.float32), noise.astype(np.float32), observation.astype(np.float32),
            {"seed": seed, "exact_truth": True},
        )
    elif fixture_id == "clipping_or_saturation":
        signal = 2.0 * event[:, None, None] * center
        artifact = np.maximum(background + signal - 1.0, 0)
        signal = np.minimum(signal, np.maximum(1.0 - background, 0))
    elif fixture_id == "patch_boundary_source":
        signal = 0.8 * event[:, None, None] * _footprint(y, x, 8, 8, 1.7)
    # pure_quiet_noise intentionally retains zero S/A.
    scale = noise_scale * np.sqrt(np.maximum(background, 0.01) / np.mean(background))
    noise = rng.normal(size=background.shape) * scale
    observation = background + signal + artifact + noise
    return SyntheticDecompositionFixture(
        fixture_id=fixture_id,
        background=background.astype(np.float32),
        structured_signal=signal.astype(np.float32),
        structured_artifact=artifact.astype(np.float32),
        noise=noise.astype(np.float32),
        observation=observation.astype(np.float32),
        diagnostics={"seed": int(seed), "exact_truth": True, "axes": "T,Y,X"},
    )
