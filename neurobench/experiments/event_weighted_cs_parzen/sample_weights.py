"""Auditable sample identities and event-mixture weights for CS-Parzen ICA."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, order=True)
class PairSampleIndex:
    frame_ui: int
    y: int
    x: int
    event_id: int | None = None
    stratum: str = "natural"
    phase: str | None = None

    @property
    def identity(self) -> tuple[int, int, int]:
        return (self.frame_ui, self.y, self.x)


@dataclass(frozen=True)
class PairSampleIndexTable:
    frame_ui: np.ndarray
    y: np.ndarray
    x: np.ndarray
    event_id: np.ndarray
    stratum: tuple[str, ...]
    phase: tuple[str | None, ...]

    @classmethod
    def from_indices(
        cls, indices: Sequence[PairSampleIndex]
    ) -> "PairSampleIndexTable":
        return cls(
            frame_ui=np.asarray([row.frame_ui for row in indices], dtype=np.int32),
            y=np.asarray([row.y for row in indices], dtype=np.int32),
            x=np.asarray([row.x for row in indices], dtype=np.int32),
            event_id=np.asarray(
                [-1 if row.event_id is None else row.event_id for row in indices],
                dtype=np.int32,
            ),
            stratum=tuple(row.stratum for row in indices),
            phase=tuple(row.phase for row in indices),
        )

    def __len__(self) -> int:
        return len(self.frame_ui)

    def identities(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (int(frame), int(y), int(x))
            for frame, y, x in zip(self.frame_ui, self.y, self.x)
        )


@dataclass(frozen=True)
class WeightedPairBatch:
    samples: np.ndarray
    weights: np.ndarray
    indices: PairSampleIndexTable
    weight_sum: float
    weight_ess: float
    per_event_mass: dict[int, float]

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples)
        weights = np.asarray(self.weights)
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError("samples must have shape [N,2]")
        if weights.shape != (len(samples),) or len(self.indices) != len(samples):
            raise ValueError("weights and indices must align with samples")
        if not np.isfinite(samples).all() or not np.isfinite(weights).all():
            raise ValueError("samples and weights must be finite")
        if np.any(weights < 0) or float(weights.sum()) <= 0:
            raise ValueError("weights must be nonnegative with positive sum")
        if not np.isclose(float(weights.sum()), self.weight_sum):
            raise ValueError("weight_sum does not match weights")
        if not np.isclose(weight_effective_sample_size(weights), self.weight_ess):
            raise ValueError("weight_ess does not match weights")


def weight_effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("weights must be a finite non-empty vector")
    if np.any(values < 0) or float(values.sum()) <= 0:
        raise ValueError("weights must be nonnegative with positive sum")
    return float(values.sum() ** 2 / np.dot(values, values))


def repeat_equivalent(alpha: float, natural_count: int, event_count: int) -> float:
    if not 0 <= alpha < 1 or natural_count < 1 or event_count < 1:
        raise ValueError("alpha must be in [0,1) and counts must be positive")
    return float(alpha * natural_count / ((1 - alpha) * event_count))


def equal_event_weights(indices: Sequence[PairSampleIndex]) -> np.ndarray:
    if not indices:
        raise ValueError("event indices must not be empty")
    event_ids = sorted(
        {row.event_id for row in indices if row.event_id is not None}
    )
    if not event_ids or any(row.event_id is None for row in indices):
        raise ValueError("every event index must declare event_id")
    result = np.zeros(len(indices), dtype=np.float64)
    for event_id in event_ids:
        positions = [i for i, row in enumerate(indices) if row.event_id == event_id]
        result[positions] = 1.0 / (len(event_ids) * len(positions))
    return result


def _check_aligned(
    samples: np.ndarray, indices: Sequence[PairSampleIndex], name: str
) -> np.ndarray:
    values = np.asarray(samples)
    if values.ndim != 2 or values.shape != (len(indices), 2):
        raise ValueError(f"{name} samples/indices must align as [N,2]")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} samples must be finite")
    identities = [row.identity for row in indices]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{name} sample identities must be unique")
    return values


def build_weighted_pair_batch(
    natural_samples: np.ndarray,
    natural_indices: Sequence[PairSampleIndex],
    event_samples: np.ndarray,
    event_indices: Sequence[PairSampleIndex],
    *,
    alpha: float,
) -> WeightedPairBatch:
    """Merge identity overlap and represent (1-alpha)P_nat + alpha P_event."""
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0,1)")
    natural = _check_aligned(natural_samples, natural_indices, "natural")
    events = _check_aligned(event_samples, event_indices, "event")
    if not natural_indices or not event_indices:
        raise ValueError("natural and event pools must both be non-empty")

    natural_mass = np.full(len(natural_indices), 1.0 / len(natural_indices))
    event_mass = equal_event_weights(event_indices)
    rows: dict[tuple[int, int, int], dict[str, object]] = {}
    for sample, index, mass in zip(natural, natural_indices, natural_mass):
        rows[index.identity] = {
            "sample": np.asarray(sample),
            "index": index,
            "weight": (1 - alpha) * float(mass),
        }
    for sample, index, mass in zip(events, event_indices, event_mass):
        existing = rows.get(index.identity)
        if existing is None:
            rows[index.identity] = {
                "sample": np.asarray(sample),
                "index": index,
                "weight": alpha * float(mass),
            }
        else:
            if not np.allclose(existing["sample"], sample, rtol=0, atol=1e-7):
                raise ValueError("overlapping identities have inconsistent samples")
            existing["weight"] = float(existing["weight"]) + alpha * float(mass)
            existing["index"] = index

    ordered = [rows[key] for key in sorted(rows)]
    samples = np.asarray([row["sample"] for row in ordered], dtype=np.float64)
    weights = np.asarray([row["weight"] for row in ordered], dtype=np.float64)
    indices = PairSampleIndexTable.from_indices(
        [row["index"] for row in ordered]  # type: ignore[list-item]
    )
    event_ids = sorted({int(row.event_id) for row in event_indices if row.event_id is not None})
    per_event_mass = {
        event_id: float(alpha / len(event_ids)) for event_id in event_ids
    }
    return WeightedPairBatch(
        samples=samples,
        weights=weights,
        indices=indices,
        weight_sum=float(weights.sum()),
        weight_ess=weight_effective_sample_size(weights),
        per_event_mass=per_event_mass,
    )


def validate_split_integrity(
    training_indices: Iterable[PairSampleIndex],
    *,
    heldout_interval_ui: tuple[int, int],
    guard_frames: int,
    heldout_event_id: int,
) -> None:
    if guard_frames < 0:
        raise ValueError("guard_frames must be nonnegative")
    excluded_start = heldout_interval_ui[0] - guard_frames
    excluded_stop = heldout_interval_ui[1] + guard_frames
    for row in training_indices:
        if excluded_start <= row.frame_ui <= excluded_stop:
            raise ValueError(
                f"training sample at UI frame {row.frame_ui} violates held-out guard"
            )
        if row.event_id == heldout_event_id:
            raise ValueError("held-out event appears in training metadata")
