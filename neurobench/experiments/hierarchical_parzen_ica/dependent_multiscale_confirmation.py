"""Quiet-calibrated confirmation authority for dependent multiscale W5c."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from neurobench.algorithms.dependent_multiscale import PatchDecomposition
from neurobench.algorithms.scientific_feature_audit import causal_local_correlation_feature


@dataclass(frozen=True)
class ConfirmationMaps:
    coherence: np.ndarray
    carrier: np.ndarray
    motion: np.ndarray
    positive_innovation: np.ndarray
    diagnostics: dict[str, Any]


def _quadratic_trend(values: np.ndarray) -> np.ndarray:
    movie = np.asarray(values, dtype=np.float32)
    time = np.linspace(-1.0, 1.0, len(movie))
    design = np.vander(time, 3, increasing=True)
    basis, _ = np.linalg.qr(design)
    matrix = movie.reshape(len(movie), -1)
    return (basis @ (basis.T @ matrix)).reshape(movie.shape).astype(np.float32)


def _unit_quiet_excess(values: np.ndarray, quiet_count: int) -> tuple[np.ndarray, float]:
    array = np.asarray(values, dtype=np.float32)
    quiet = array[: int(quiet_count)]
    baseline = np.median(quiet, axis=0)
    scale = max(float(np.percentile(np.maximum(quiet - baseline, 0), 99.5)), 1e-6)
    excess = np.maximum(array - baseline, 0) / scale
    return (excess / (1.0 + excess)).astype(np.float32), scale


def build_confirmation_maps(
    observation: np.ndarray,
    views: Mapping[str, np.ndarray],
    *,
    quiet_count: int,
) -> ConfirmationMaps:
    """Build label-free coherence, carrier, and motion confirmation maps."""
    movie = np.asarray(observation, dtype=np.float32)
    quiet = movie[: int(quiet_count)]
    center = np.median(quiet, axis=0)
    mad = 1.4826 * np.median(np.abs(quiet - center), axis=0)
    positive = mad[mad > 1e-8]
    floor = max(float(np.median(positive)) if positive.size else 1.0, 1e-4)
    standardized = (movie - center) / np.maximum(mad, floor)
    coherence_raw = causal_local_correlation_feature(
        standardized, window_frames=15, lag_frames=0,
        spatial_sigma_px=1.5, activity_qualified=True,
    )
    coherence, coherence_scale = _unit_quiet_excess(coherence_raw, quiet_count)
    compact = np.asarray(views["scale_5"], dtype=np.float32)
    innovation = compact - _quadratic_trend(compact)
    positive_innovation = np.maximum(innovation, 0)
    carrier, carrier_scale = _unit_quiet_excess(positive_innovation, quiet_count)
    disagreement = np.abs(
        compact - np.asarray(views["scale_7"], dtype=np.float32)
    )
    motion, motion_scale = _unit_quiet_excess(disagreement, quiet_count)
    return ConfirmationMaps(
        coherence=coherence,
        carrier=carrier,
        motion=motion,
        positive_innovation=positive_innovation,
        diagnostics={
            "quiet_count": int(quiet_count),
            "coherence_window_frames": 15,
            "coherence_spatial_sigma_px": 1.5,
            "coherence_quiet_scale": coherence_scale,
            "carrier_quiet_scale": carrier_scale,
            "motion_quiet_scale": motion_scale,
            "labels_used": False,
        },
    )


def apply_confirmation_authority(
    baseline: PatchDecomposition,
    population: PatchDecomposition,
    maps: ConfirmationMaps,
    *,
    lane_id: str,
) -> PatchDecomposition:
    """Apply one frozen confirmation lane with conservative mass transfer."""
    if lane_id == "coherence_confirmed":
        authority = maps.coherence / (1.0 + 2.0 * maps.motion)
        target_gain = 3.0
    elif lane_id == "carrier_constrained":
        authority = maps.carrier / (1.0 + 2.0 * maps.motion)
        target_gain = 2.5
    elif lane_id == "coherence_carrier_combined":
        authority = np.sqrt(maps.coherence * maps.carrier) / (1.0 + 2.0 * maps.motion)
        target_gain = 3.0
    else:
        raise ValueError(f"unknown confirmation lane: {lane_id}")
    authority = np.clip(authority, 0, 1).astype(np.float32)
    background = baseline.background + authority * (population.background - baseline.background)
    signal = baseline.structured_signal + authority * (
        population.structured_signal - baseline.structured_signal
    )
    artifact = baseline.structured_artifact.copy()
    noise = baseline.noise_candidate + authority * (
        population.noise_candidate - baseline.noise_candidate
    )
    missing = np.maximum(target_gain * maps.positive_innovation - signal, 0)
    correction = authority * missing
    signal += correction
    noise -= correction
    observation = (
        baseline.background + baseline.structured_signal
        + baseline.structured_artifact + baseline.noise_candidate
        + baseline.closure_residual
    )
    closure = observation - background - signal - artifact - noise
    confirmed = authority >= 0.5
    nuisance = authority <= 0.1
    unresolved = ~(confirmed | nuisance)
    return PatchDecomposition(
        patch_id=f"{population.patch_id}__{lane_id}",
        background=background.astype(np.float32),
        structured_signal=signal.astype(np.float32),
        structured_artifact=artifact.astype(np.float32),
        noise_candidate=noise.astype(np.float32),
        closure_residual=closure.astype(np.float32),
        posterior_uncertainty=(1.0 - np.abs(2.0 * authority - 1.0)).astype(np.float32),
        diagnostics={
            "method_id": lane_id,
            "target_gain": target_gain,
            "authority_minimum": float(authority.min()),
            "authority_maximum": float(authority.max()),
            "authority_mean": float(authority.mean()),
            "confirmed_fraction": float(confirmed.mean()),
            "nuisance_fraction": float(nuisance.mean()),
            "unresolved_fraction": float(unresolved.mean()),
            "labels_used": False,
            "noise_status": "noise_candidate",
            "carrier_role": "bounded_confirmation_constraint_not_output_target",
        },
    )
