"""Split-first, bounded pixel-time sampling for event-weighted pairwise ICA."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .sample_weights import PairSampleIndex, validate_split_integrity


@dataclass(frozen=True)
class FoldSamplePools:
    natural_screen_indices: tuple[PairSampleIndex, ...]
    natural_confirm_indices: tuple[PairSampleIndex, ...]
    event_screen_indices: tuple[PairSampleIndex, ...]
    event_confirm_indices: tuple[PairSampleIndex, ...]
    excluded_interval_ui: tuple[int, int]
    heldout_event_id: int
    train_event_ids: tuple[int, ...]


def _draw_positions(population: int, count: int, seed: int) -> np.ndarray:
    if population < 1 or count < 1:
        raise ValueError("population and count must be positive")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(population, size=min(count, population), replace=False))


def sample_natural_indices(
    eligible_frames_ui: Sequence[int],
    anatomy_mask: np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> tuple[PairSampleIndex, ...]:
    frames = np.asarray(sorted(set(int(x) for x in eligible_frames_ui)), dtype=np.int32)
    mask = np.asarray(anatomy_mask, dtype=bool)
    if frames.ndim != 1 or not len(frames) or mask.ndim != 2:
        raise ValueError("eligible frames and 2-D anatomy mask are required")
    pixels = np.flatnonzero(mask.ravel())
    if not len(pixels):
        raise ValueError("anatomy mask has no eligible pixels")
    positions = _draw_positions(len(frames) * len(pixels), sample_count, seed)
    frame_positions = positions // len(pixels)
    local_pixels = pixels[positions % len(pixels)]
    y, x = np.unravel_index(local_pixels, mask.shape)
    return tuple(
        PairSampleIndex(
            frame_ui=int(frames[frame_position]),
            y=int(row),
            x=int(column),
            stratum="natural",
        )
        for frame_position, row, column in zip(frame_positions, y, x)
    )


def _event_frame_candidates(
    interval_ui: tuple[int, int],
    anatomy_mask: np.ndarray,
    event_id: int,
    count: int,
    seed: int,
) -> tuple[PairSampleIndex, ...]:
    frames = np.arange(interval_ui[0], interval_ui[1] + 1, dtype=np.int32)
    pixels = np.flatnonzero(np.asarray(anatomy_mask, dtype=bool).ravel())
    positions = _draw_positions(len(frames) * len(pixels), count, seed)
    frame_positions = positions // len(pixels)
    local_pixels = pixels[positions % len(pixels)]
    y, x = np.unravel_index(local_pixels, anatomy_mask.shape)
    return tuple(
        PairSampleIndex(
            frame_ui=int(frames[frame_position]),
            y=int(row),
            x=int(column),
            event_id=event_id,
            stratum="event_frame",
        )
        for frame_position, row, column in zip(frame_positions, y, x)
    )


def _event_roi_candidates(
    interval_ui: tuple[int, int],
    labels: Sequence[Mapping[str, Any]],
    anatomy_mask: np.ndarray,
    event_id: int,
    radius_px: int,
    count: int,
    seed: int,
) -> tuple[PairSampleIndex, ...]:
    if radius_px < 0:
        raise ValueError("ROI radius must be nonnegative")
    height, width = anatomy_mask.shape
    spatial: set[tuple[int, int]] = set()
    for row in labels:
        if int(row["burst_id"]) != event_id:
            continue
        center_x, center_y = float(row["x_px"]), float(row["y_px"])
        x0, y0 = int(round(center_x)), int(round(center_y))
        for y in range(max(0, y0 - radius_px), min(height, y0 + radius_px + 1)):
            for x in range(max(0, x0 - radius_px), min(width, x0 + radius_px + 1)):
                if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius_px**2:
                    if anatomy_mask[y, x]:
                        spatial.add((y, x))
    if not spatial:
        raise ValueError(f"event {event_id} has no eligible ROI pixels")
    candidates = [
        (frame_ui, y, x)
        for frame_ui in range(interval_ui[0], interval_ui[1] + 1)
        for y, x in sorted(spatial)
    ]
    positions = _draw_positions(len(candidates), count, seed)
    return tuple(
        PairSampleIndex(
            frame_ui=int(candidates[position][0]),
            y=int(candidates[position][1]),
            x=int(candidates[position][2]),
            event_id=event_id,
            stratum="event_roi",
        )
        for position in positions
    )


def sample_event_indices(
    *,
    mode: str,
    event_ids: Sequence[int],
    event_intervals_ui: Mapping[int, tuple[int, int]],
    labels: Sequence[Mapping[str, Any]],
    anatomy_mask: np.ndarray,
    max_samples_per_event: int,
    event_roi_radius_px: int,
    seed: int,
) -> tuple[PairSampleIndex, ...]:
    if mode not in {"frame_balanced", "roi_balanced"}:
        raise ValueError("mode must be frame_balanced or roi_balanced")
    rows: list[PairSampleIndex] = []
    for event_id in sorted(set(int(x) for x in event_ids)):
        if event_id not in event_intervals_ui:
            raise ValueError(f"missing interval for event {event_id}")
        event_seed = int(seed + 1009 * event_id + (0 if mode == "frame_balanced" else 1))
        if mode == "frame_balanced":
            selected = _event_frame_candidates(
                event_intervals_ui[event_id],
                anatomy_mask,
                event_id,
                max_samples_per_event,
                event_seed,
            )
        else:
            selected = _event_roi_candidates(
                event_intervals_ui[event_id],
                labels,
                anatomy_mask,
                event_id,
                event_roi_radius_px,
                max_samples_per_event,
                event_seed,
            )
        rows.extend(selected)
    return tuple(rows)


def build_fold_sample_pools(
    *,
    heldout_event_id: int,
    event_intervals_ui: Mapping[int, tuple[int, int]],
    review_interval_ui: tuple[int, int],
    anatomy_mask: np.ndarray,
    labels: Sequence[Mapping[str, Any]],
    mode: str,
    heldout_guard_frames: int,
    screen_samples: int,
    confirmation_samples: int,
    event_screen_max_samples_per_event: int,
    event_confirmation_max_samples_per_event: int,
    event_roi_radius_px: int,
    seed: int,
    bad_frames_ui: Sequence[int] = (),
) -> FoldSamplePools:
    if heldout_event_id not in event_intervals_ui:
        raise ValueError("held-out event has no declared interval")
    heldout = event_intervals_ui[heldout_event_id]
    excluded = (
        heldout[0] - heldout_guard_frames,
        heldout[1] + heldout_guard_frames,
    )
    bad = set(int(x) for x in bad_frames_ui)
    eligible = [
        frame
        for frame in range(review_interval_ui[0] + 1, review_interval_ui[1] + 1)
        if not excluded[0] <= frame <= excluded[1] and frame not in bad
    ]
    natural_confirm = sample_natural_indices(
        eligible, anatomy_mask, sample_count=confirmation_samples, seed=seed
    )
    natural_screen = natural_confirm[: min(screen_samples, len(natural_confirm))]
    train_events = tuple(sorted(set(event_intervals_ui) - {heldout_event_id}))
    event_confirm = sample_event_indices(
        mode=mode,
        event_ids=train_events,
        event_intervals_ui=event_intervals_ui,
        labels=labels,
        anatomy_mask=anatomy_mask,
        max_samples_per_event=event_confirmation_max_samples_per_event,
        event_roi_radius_px=event_roi_radius_px,
        seed=seed,
    )
    event_screen = sample_event_indices(
        mode=mode,
        event_ids=train_events,
        event_intervals_ui=event_intervals_ui,
        labels=labels,
        anatomy_mask=anatomy_mask,
        max_samples_per_event=event_screen_max_samples_per_event,
        event_roi_radius_px=event_roi_radius_px,
        seed=seed + 7919,
    )
    validate_split_integrity(
        (*natural_confirm, *event_confirm),
        heldout_interval_ui=heldout,
        guard_frames=heldout_guard_frames,
        heldout_event_id=heldout_event_id,
    )
    return FoldSamplePools(
        natural_screen_indices=natural_screen,
        natural_confirm_indices=natural_confirm,
        event_screen_indices=event_screen,
        event_confirm_indices=event_confirm,
        excluded_interval_ui=excluded,
        heldout_event_id=heldout_event_id,
        train_event_ids=train_events,
    )


def extract_pair_samples(
    frames: np.ndarray,
    indices: Sequence[PairSampleIndex],
    *,
    review_start_ui: int,
) -> np.ndarray:
    values = np.asarray(frames)
    if values.ndim != 3:
        raise ValueError("frames must have shape [T,Y,X]")
    result = np.empty((len(indices), 2), dtype=np.float64)
    for position, row in enumerate(indices):
        current_zero = row.frame_ui - review_start_ui
        if not 1 <= current_zero < len(values):
            raise ValueError(f"UI frame {row.frame_ui} has no valid adjacent pair")
        result[position] = (
            values[current_zero - 1, row.y, row.x],
            values[current_zero, row.y, row.x],
        )
    if not np.isfinite(result).all():
        raise ValueError("sample extraction produced non-finite values")
    return result
