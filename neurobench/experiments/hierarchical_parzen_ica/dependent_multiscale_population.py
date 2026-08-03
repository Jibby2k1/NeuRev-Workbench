"""Patchwise population-preserving attribution for dependent multiscale W5b."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import uniform_filter1d

from neurobench.algorithms.dependent_multiscale import (
    PatchDecomposition,
    decompose_patch_baseline,
    overlap_add,
)

from .dependent_multiscale_information import _project_group


@dataclass(frozen=True)
class PopulationPreservingResult:
    decomposition: PatchDecomposition
    diagnostics: dict[str, Any]


def _positive_trace(values: np.ndarray) -> np.ndarray:
    return np.sum(np.maximum(np.asarray(values, dtype=np.float64), 0), axis=(1, 2))


def population_preserving_patch(
    observation: np.ndarray,
    views: Mapping[str, np.ndarray],
    *,
    patch_id: str,
    population_window_frames: int = 23,
    population_gain: float = 1.25,
    residual_recapture_authority: float = 0.25,
    maximum_positive_trace_gain: float = 2.25,
) -> PopulationPreservingResult:
    """Preserve transient broad neural drive and recapture synchronous residual.

    The method is label-free. It conserves original-space mass by moving a
    bounded transient component from background to signal and a bounded
    population-synchronous component from noise candidate to signal. It does
    not force individual neural traces to be independent.
    """
    window = int(population_window_frames)
    if window < 3 or window % 2 != 1:
        raise ValueError("population_window_frames must be an odd integer >= 3")
    gain = float(population_gain)
    recapture = float(residual_recapture_authority)
    trace_cap = float(maximum_positive_trace_gain)
    if gain < 0 or not 0 <= recapture <= 1 or trace_cap < 1:
        raise ValueError("population parameters are outside their bounded ranges")
    baseline = decompose_patch_baseline(observation, views, patch_id=patch_id)
    background = baseline.background.copy()
    signal = baseline.structured_signal.copy()
    artifact = baseline.structured_artifact.copy()
    noise = baseline.noise_candidate.copy()
    baseline_trace = _positive_trace(signal)

    population = gain * (
        background
        - uniform_filter1d(background, size=min(window, len(background) | 1), axis=0, mode="nearest")
    )
    signal += population
    background -= population
    synchronous_residual = recapture * _project_group(noise, signal, nuisance=None)
    proposed_signal = signal + synchronous_residual
    proposed_trace = _positive_trace(proposed_signal)
    peak_gain = float(np.max(proposed_trace) / max(np.max(baseline_trace), 1e-12))
    area_gain = float(np.sum(proposed_trace) / max(np.sum(baseline_trace), 1e-12))
    correction_scale = min(
        1.0,
        trace_cap / max(peak_gain, 1e-12),
        trace_cap / max(area_gain, 1e-12),
    )
    if correction_scale < 1.0:
        # Scale both transfers together so conservation and semantic direction
        # remain explicit instead of clipping the reconstructed signal.
        background += (1.0 - correction_scale) * population
        signal -= (1.0 - correction_scale) * population
        synchronous_residual *= correction_scale
    signal += synchronous_residual
    noise -= synchronous_residual
    closure = np.asarray(observation, dtype=np.float32) - background - signal - artifact - noise
    decomposition = PatchDecomposition(
        patch_id=patch_id,
        background=background,
        structured_signal=signal,
        structured_artifact=artifact,
        noise_candidate=noise,
        closure_residual=closure,
        posterior_uncertainty=None,
        diagnostics={
            **baseline.diagnostics,
            "method_id": "population_preserving_conditional_attribution",
            "population_drive_preserved": True,
            "individual_neural_independence_forced": False,
            "residual_recapture_authority": recapture,
            "maximum_positive_trace_gain": trace_cap,
            "noise_status": "noise_candidate",
        },
    )
    diagnostics = {
        "population_window_frames": window,
        "population_gain": gain,
        "residual_recapture_authority": recapture,
        "unbounded_peak_trace_gain": peak_gain,
        "unbounded_area_trace_gain": area_gain,
        "applied_correction_scale": correction_scale,
        "population_energy": float(np.mean(population**2)),
        "synchronous_residual_energy": float(np.mean(synchronous_residual**2)),
    }
    return PopulationPreservingResult(decomposition, diagnostics)


def _origins(size: int, patch: int, stride: int) -> tuple[int, ...]:
    if patch > size or patch < 3 or stride < 1:
        raise ValueError("patch/stride are incompatible with the movie geometry")
    values = list(range(0, size - patch + 1, stride))
    if values[-1] != size - patch:
        values.append(size - patch)
    return tuple(values)


def population_preserving_movie(
    observation: np.ndarray,
    views: Mapping[str, np.ndarray],
    *,
    patch_px: int,
    stride_px: int,
    population_window_frames: int = 23,
    population_gain: float = 1.25,
    residual_recapture_authority: float = 0.25,
) -> PopulationPreservingResult:
    """Apply the frozen attribution patchwise and blend with floored Hann OLA."""
    movie = np.asarray(observation)
    if movie.ndim != 3 or any(np.asarray(value).shape != movie.shape for value in views.values()):
        raise ValueError("observation and views must be aligned [T,Y,X]")
    origins = tuple(
        (y, x)
        for y in _origins(movie.shape[1], int(patch_px), int(stride_px))
        for x in _origins(movie.shape[2], int(patch_px), int(stride_px))
    )
    patch_channels: dict[str, list[tuple[tuple[int, int], np.ndarray]]] = {
        name: [] for name in ("background", "structured_signal", "structured_artifact", "noise_candidate")
    }
    patch_diagnostics = []
    for index, (y, x) in enumerate(origins):
        slices = (slice(None), slice(y, y + patch_px), slice(x, x + patch_px))
        result = population_preserving_patch(
            movie[slices], {name: value[slices] for name, value in views.items()},
            patch_id=f"patch_{index:05d}_y{y}_x{x}",
            population_window_frames=population_window_frames,
            population_gain=population_gain,
            residual_recapture_authority=residual_recapture_authority,
        )
        decomposition = result.decomposition
        for name in patch_channels:
            patch_channels[name].append(((y, x), np.asarray(getattr(decomposition, name))))
        patch_diagnostics.append(result.diagnostics)
    blended: dict[str, np.ndarray] = {}
    overlap_diagnostics = {}
    for name, patches in patch_channels.items():
        blended[name], overlap_diagnostics[name] = overlap_add(patches, movie.shape)
    closure = np.asarray(movie, dtype=np.float32) - sum(blended.values())
    decomposition = PatchDecomposition(
        patch_id="overlap_add_movie",
        background=blended["background"],
        structured_signal=blended["structured_signal"],
        structured_artifact=blended["structured_artifact"],
        noise_candidate=blended["noise_candidate"],
        closure_residual=closure,
        posterior_uncertainty=None,
        diagnostics={
            "method_id": "patchwise_population_preserving_conditional_attribution",
            "patch_count": len(origins),
            "noise_status": "noise_candidate",
            "labels_used": False,
        },
    )
    return PopulationPreservingResult(decomposition, {
        "patch_count": len(origins),
        "origins_yx": [list(value) for value in origins],
        "overlap_add": overlap_diagnostics,
        "patches": patch_diagnostics,
    })
