"""Truth-known, numerically distinct identifiability calibration fixtures."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .synthetic import SpatiotemporalFixture, make_spatiotemporal_fixture


IDENTIFIABLE_CASES = (
    "isolated", "overlap", "synchronous", "motion_edge", "saturation",
    "similar_persistence",
)
UNIDENTIFIABLE_CASES = (
    "spatial_rank_deficient", "temporal_rank_deficient",
    "exact_duplicate_sources", "pure_noise",
)
ALL_IDENTIFIABILITY_CASES = IDENTIFIABLE_CASES + UNIDENTIFIABLE_CASES


def make_identifiability_fixture(
    case_id: str,
    *,
    seed: int,
    frame_count: int = 256,
    shape: tuple[int, int] = (16, 16),
    snr: float = 8.0,
) -> SpatiotemporalFixture:
    """Create a fixture whose identifiability label follows its actual arrays."""
    if case_id in IDENTIFIABLE_CASES:
        return make_spatiotemporal_fixture(
            case_id, seed=seed, frame_count=frame_count, shape=shape, snr=snr
        )
    if case_id == "pure_noise":
        return make_spatiotemporal_fixture(
            "pure_noise", seed=seed, frame_count=frame_count, shape=shape, snr=snr
        )
    if case_id not in UNIDENTIFIABLE_CASES:
        raise ValueError(f"unknown identifiability case: {case_id}")
    base = make_spatiotemporal_fixture(
        "isolated", seed=seed, frame_count=frame_count, shape=shape, snr=snr
    )
    footprints = np.asarray(base.footprints, dtype=np.float64).copy()
    traces = np.asarray(base.traces, dtype=np.float64).copy()
    if case_id in {"spatial_rank_deficient", "exact_duplicate_sources"}:
        footprints[1] = footprints[0]
    if case_id in {"temporal_rank_deficient", "exact_duplicate_sources"}:
        traces[1] = traces[0]
    neural = np.einsum("st,shw->thw", traces, footprints)
    observation = (
        np.asarray(base.background, dtype=np.float64)
        + neural
        + np.asarray(base.structured_artifact, dtype=np.float64)
        + np.asarray(base.measurement_noise, dtype=np.float64)
    )
    closure = (
        observation - np.asarray(base.background) - neural
        - np.asarray(base.structured_artifact) - np.asarray(base.measurement_noise)
    )
    return replace(
        base,
        case_id=case_id,
        observation=observation.astype(np.float32),
        neural_signal=neural.astype(np.float32),
        footprints=footprints.astype(np.float32),
        traces=traces.astype(np.float32),
        identifiable=False,
        metadata={
            **base.metadata,
            "identifiability_mechanism": case_id,
            "mixing_rank": int(np.linalg.matrix_rank(footprints.reshape(3, -1).T)),
            "temporal_rank": int(np.linalg.matrix_rank(traces)),
            "maximum_closure_absolute": float(np.max(np.abs(closure))),
        },
    )


def assert_distinct_case_contract(*, seed: int = 101) -> dict[str, object]:
    """Fail if a positive and negative identifiability fixture are identical."""
    fixtures = {
        case: make_identifiability_fixture(case, seed=seed)
        for case in ALL_IDENTIFIABILITY_CASES
    }
    collisions = []
    for positive in IDENTIFIABLE_CASES:
        for negative in UNIDENTIFIABLE_CASES:
            if np.array_equal(fixtures[positive].observation, fixtures[negative].observation):
                collisions.append([positive, negative])
    if collisions:
        raise RuntimeError(f"identifiability fixture collisions: {collisions}")
    return {
        "seed": int(seed), "case_count": len(fixtures), "collisions": collisions,
        "identifiable_count": len(IDENTIFIABLE_CASES),
        "unidentifiable_count": len(UNIDENTIFIABLE_CASES),
    }
