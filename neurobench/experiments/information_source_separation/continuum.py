"""Continuous truth-known identifiability fixtures for the conclusive batch."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.stats import qmc

from .synthetic import SpatiotemporalFixture, make_spatiotemporal_fixture


@dataclass(frozen=True)
class ContinuumSpecification:
    fixture_id: str
    seed: int
    overlap: float
    temporal_collinearity: float
    amplitude_ratio: float
    snr: float
    background_alias: float
    artifact_family: str
    identifiable: bool
    degeneracy: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id, "seed": self.seed,
            "overlap": self.overlap,
            "temporal_collinearity": self.temporal_collinearity,
            "amplitude_ratio": self.amplitude_ratio, "snr": self.snr,
            "background_alias": self.background_alias,
            "artifact_family": self.artifact_family,
            "identifiable": self.identifiable, "degeneracy": self.degeneracy,
        }


def space_filling_continuum(
    count: int, *, seed: int, split: str
) -> tuple[ContinuumSpecification, ...]:
    """Create a deterministic Latin-hypercube continuum with exact controls."""
    if count < 12 or count % 4:
        raise ValueError("continuum count must be >=12 and divisible by four")
    if not split:
        raise ValueError("split must be non-empty")
    unit = qmc.LatinHypercube(d=5, seed=int(seed)).random(count)
    artifacts = ("none", "illumination", "motion", "clipping", "heteroscedastic")
    degeneracies = ("spatial_rank_deficient", "temporal_rank_deficient", "exact_duplicate_sources", "pure_noise")
    rows = []
    negative_start = count * 3 // 4
    for index, values in enumerate(unit):
        identifiable = index < negative_start
        degeneracy = None if identifiable else degeneracies[(index-negative_start) % len(degeneracies)]
        rows.append(ContinuumSpecification(
            fixture_id=f"{split}_{index:04d}", seed=int(seed)+index*1009,
            overlap=float(0.02 + 0.96*values[0]),
            temporal_collinearity=float(0.02 + 0.975*values[1]),
            amplitude_ratio=float(2.0 ** (-1.0 + 2.0*values[2])),
            snr=float(2.0 ** (2.0 + 2.0*values[3])),
            background_alias=float(0.6*values[4]),
            artifact_family=artifacts[index % len(artifacts)],
            identifiable=identifiable, degeneracy=degeneracy,
        ))
    return tuple(rows)


def make_continuum_fixture(
    specification: ContinuumSpecification,
    *, frame_count: int = 256,
    shape: tuple[int, int] = (16, 16),
) -> SpatiotemporalFixture:
    """Materialize a continuous B/S/A/N mixture with exact closure."""
    spec = specification
    base = make_spatiotemporal_fixture(
        "isolated", seed=spec.seed, frame_count=frame_count, shape=shape,
        snr=max(spec.snr, 1.0),
    )
    footprints = np.asarray(base.footprints, dtype=np.float64).copy()
    traces = np.asarray(base.traces, dtype=np.float64).copy()
    footprints[1] = (
        (1.0-spec.overlap)*footprints[1] + spec.overlap*footprints[0]
    )
    footprints[1] /= max(float(footprints[1].max()), np.finfo(float).eps)
    traces[1] = (
        spec.temporal_collinearity*traces[0]
        + (1.0-spec.temporal_collinearity)*traces[1]
    ) * spec.amplitude_ratio
    if spec.degeneracy in {"spatial_rank_deficient", "exact_duplicate_sources"}:
        footprints[1] = footprints[0]
    if spec.degeneracy in {"temporal_rank_deficient", "exact_duplicate_sources"}:
        traces[1] = traces[0]
    if spec.degeneracy == "pure_noise":
        traces[:] = 0.0
    neural = np.einsum("st,shw->thw", traces, footprints)
    background = np.asarray(base.background, dtype=np.float64).copy()
    alias_trace = traces[0] if np.any(traces[0]) else np.sin(np.linspace(0, 4*np.pi, frame_count))
    background += spec.background_alias * alias_trace[:, None, None] * footprints[0][None]
    artifact = np.zeros_like(background)
    if spec.artifact_family == "illumination":
        artifact += np.linspace(0, 0.35, frame_count)[:, None, None] * np.mean(background, axis=0)[None]
    elif spec.artifact_family == "motion":
        anatomy = np.mean(background[:8], axis=0)
        artifact[frame_count//3:frame_count//3+12] = np.roll(anatomy, 1, axis=1)-anatomy
    elif spec.artifact_family == "clipping":
        artifact[frame_count//2:frame_count//2+14, 2:7, 2:7] += 1.5
    signal_scale = max(float(np.std(neural)), 0.05)
    noise_scale = signal_scale/spec.snr
    rng = np.random.default_rng(spec.seed+77)
    if spec.artifact_family == "heteroscedastic":
        local = noise_scale*np.sqrt(np.maximum(background+neural, 0.05))
    else:
        local = np.full_like(background, noise_scale)
    noise = rng.normal(size=background.shape)*local
    observation = background+neural+artifact+noise
    if spec.artifact_family == "clipping":
        ceiling = float(np.quantile(observation, 0.99))
        clipped = np.minimum(observation, ceiling)
        artifact += clipped-observation
        observation = clipped
    spatial_matrix = footprints.reshape(len(footprints), -1).T
    closure = observation-background-neural-artifact-noise
    return replace(
        base, case_id=spec.fixture_id,
        observation=observation.astype(np.float32),
        neural_signal=neural.astype(np.float32),
        background=background.astype(np.float32),
        structured_artifact=artifact.astype(np.float32),
        measurement_noise=noise.astype(np.float32),
        footprints=footprints.astype(np.float32), traces=traces.astype(np.float32),
        identifiable=bool(spec.identifiable),
        metadata={
            **spec.to_dict(),
            "spatial_condition_number": float(np.linalg.cond(spatial_matrix)),
            "temporal_condition_number": float(np.linalg.cond(traces)),
            "spatial_rank": int(np.linalg.matrix_rank(spatial_matrix)),
            "temporal_rank": int(np.linalg.matrix_rank(traces)),
            "maximum_closure_absolute": float(np.max(np.abs(closure))),
        },
    )
