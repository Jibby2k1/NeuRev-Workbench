"""Leakage-safe evaluation primitives for the Gamma-LS difference ablation.

Calibration functions in this module accept score arrays and training-quiet
masks only.  Sparse-positive coordinates first enter the explicitly named
``evaluate_sparse_positive_recall`` function.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from neurobench.metrics.sparse_detection import (
    Peak,
    extract_separated_local_maxima,
    match_peaks_one_to_one,
)


QUIET_NMS_PEAK_BURDENS = (0.25, 0.5, 1.0, 2.0, 5.0)
CANDIDATE_BUDGETS_PER_BURST = (20, 40, 58, 80, 100)
NMS_DISTANCE_PX = 6
NMS_SENSITIVITY_DISTANCES_PX = (4, 8)
ALLOWED_NMS_DISTANCES_PX = (4, 6, 8)
MATCH_RADIUS_PX = 6.0


@dataclass(frozen=True)
class LocalStdScaleFloorFit:
    """Positive local-standard-deviation floor fitted on training quiet only."""

    scale_floor: float
    percentile: float
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale_floor) or self.scale_floor <= 0:
            raise ValueError("scale_floor must be finite and positive")
        if not math.isfinite(self.percentile) or not 0 <= self.percentile <= 100:
            raise ValueError("percentile must be in [0,100]")


@dataclass(frozen=True)
class BurstCandidateResult:
    """Label-free threshold-occupancy maps and strict separated peaks."""

    occupancy_maps: Mapping[Any, np.ndarray]
    peaks: Mapping[Any, tuple[Peak, ...]]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class QuietThresholdCalibration:
    """Frozen score thresholds calibrated to empirical quiet peak burdens."""

    operating_points: tuple[Mapping[str, Any], ...]
    pseudo_burst_windows: Mapping[Any, tuple[int, int]]
    diagnostics: Mapping[str, Any]

    def threshold_for(self, target_burden: float) -> float:
        """Return the frozen score threshold for an exact declared target."""

        target = float(target_burden)
        for row in self.operating_points:
            if float(row["target_nms_peaks_per_pseudo_burst"]) == target:
                return float(row["threshold_z"])
        raise KeyError(f"undeclared target burden: {target_burden}")


def _numpy(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def _video(values: Any, name: str) -> np.ndarray:
    result = _numpy(values).astype(np.float32, copy=False)
    if result.ndim != 3 or min(result.shape) < 1:
        raise ValueError(f"{name} must be a non-empty [T,Y,X] array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _training_quiet_mask(mask: Any, frame_count: int) -> np.ndarray:
    result = _numpy(mask)
    if result.dtype != np.bool_ or result.shape != (frame_count,):
        raise ValueError(
            "training_quiet_mask must be a boolean vector with one entry per frame"
        )
    if not np.any(result):
        raise ValueError("training_quiet_mask must select at least one frame")
    return result


def fit_training_quiet_scale_floor(
    local_std: Any,
    training_quiet_mask: Any,
    *,
    percentile: float,
) -> LocalStdScaleFloorFit:
    """Fit a positive local-std floor using training quiet frames and nothing else."""

    values = _video(local_std, "local_std")
    if np.any(values < 0):
        raise ValueError("local_std cannot contain negative values")
    quiet = _training_quiet_mask(training_quiet_mask, values.shape[0])
    requested = float(percentile)
    if not math.isfinite(requested) or not 0 <= requested <= 100:
        raise ValueError("percentile must be finite and in [0,100]")
    selected = values[quiet].astype(np.float64, copy=False)
    positive = selected[selected > 0]
    if positive.size == 0:
        raise ValueError("training quiet local_std contains no positive values")
    floor = float(np.percentile(positive, requested))
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("training quiet percentile did not produce a positive floor")
    diagnostics = {
        "fit_scope": "training_quiet_frames_only",
        "training_quiet_frame_count": int(np.count_nonzero(quiet)),
        "training_quiet_sample_count": int(selected.size),
        "positive_training_quiet_sample_count": int(positive.size),
        "excluded_nonpositive_training_quiet_fraction": float(
            1.0 - positive.size / selected.size
        ),
        "percentile_method": "numpy_linear",
        "quiet_interval_used": True,
        "sparse_positive_coordinates_used": False,
        "sparse_positive_identities_used": False,
    }
    return LocalStdScaleFloorFit(floor, requested, diagnostics)


def _duration_items(
    target_durations: Mapping[Any, int] | Sequence[int],
) -> list[tuple[Any, int]]:
    if isinstance(target_durations, Mapping):
        items = list(target_durations.items())
    else:
        items = list(enumerate(target_durations, start=1))
    if not items:
        raise ValueError("at least one target duration is required")
    result = []
    for key, duration_value in items:
        duration = int(duration_value)
        if duration != duration_value or duration < 1:
            raise ValueError("target durations must be positive integers")
        result.append((key, duration))
    return result


def _quiet_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    before = np.concatenate((np.asarray([False]), mask[:-1]))
    after = np.concatenate((mask[1:], np.asarray([False])))
    starts = np.flatnonzero(mask & ~before)
    stops = np.flatnonzero(mask & ~after) + 1
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def duration_matched_quiet_windows(
    training_quiet_mask: Any,
    target_durations: Mapping[Any, int] | Sequence[int],
    *,
    starts: Mapping[Any, int] | None = None,
) -> dict[Any, tuple[int, int]]:
    """Construct deterministic quiet windows with the declared burst durations.

    Automatic construction uses non-overlapping windows from left to right when
    possible.  If the total requested duration exceeds the quiet interval, a
    later window is right-aligned within a sufficiently long quiet run.  Such
    deterministic overlap is reported by the calibration diagnostics.
    """

    mask = _numpy(training_quiet_mask)
    if mask.ndim != 1 or mask.dtype != np.bool_ or not np.any(mask):
        raise ValueError("training_quiet_mask must be a non-empty boolean vector")
    items = _duration_items(target_durations)
    runs = _quiet_runs(mask)
    if starts is not None and set(starts) != {key for key, _ in items}:
        raise ValueError("starts must contain exactly the target-duration identifiers")

    windows: dict[Any, tuple[int, int]] = {}
    if starts is not None:
        for key, duration in items:
            start = int(starts[key])
            stop = start + duration
            if start < 0 or stop > mask.size or not np.all(mask[start:stop]):
                raise ValueError(f"declared window {key!r} is not entirely training quiet")
            windows[key] = (start, stop)
        return windows

    cursors = [start for start, _ in runs]
    for key, duration in items:
        fresh = [
            (cursors[index], index)
            for index, (_, stop) in enumerate(runs)
            if cursors[index] + duration <= stop
        ]
        if fresh:
            start, run_index = min(fresh)
            cursors[run_index] = start + duration
        else:
            reusable = [
                (run_start, index)
                for index, (run_start, run_stop) in enumerate(runs)
                if run_stop - run_start >= duration
            ]
            if not reusable:
                raise ValueError(
                    f"no contiguous training-quiet run can fit duration {duration}"
                )
            _, run_index = min(reusable)
            start = runs[run_index][1] - duration
        stop = start + duration
        if not np.all(mask[start:stop]):  # defensive: run construction should ensure this
            raise AssertionError("constructed pseudo-burst escaped training quiet")
        windows[key] = (int(start), int(stop))
    return windows


def temporal_threshold_occupancy(scores: Any, *, threshold_z: float) -> np.ndarray:
    """Aggregate a burst as the fraction of frames strictly above threshold."""

    values = _video(scores, "scores")
    threshold = float(threshold_z)
    if not math.isfinite(threshold):
        raise ValueError("threshold_z must be finite")
    return np.mean(values > threshold, axis=0, dtype=np.float64).astype(np.float32)


def strict_separated_nms(
    score: Any,
    *,
    distance_px: int = NMS_DISTANCE_PX,
    threshold: float = 0.0,
    limit: int = 10_000,
) -> list[Peak]:
    """Apply maintained deterministic NMS with a strict score cutoff."""

    values = _numpy(score).astype(np.float64, copy=False)
    cutoff = float(threshold)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("score must be a finite two-dimensional array")
    if not math.isfinite(cutoff):
        raise ValueError("threshold must be finite")
    strict_cutoff = float(np.nextafter(np.float64(cutoff), np.float64(np.inf)))
    return extract_separated_local_maxima(
        values,
        int(distance_px),
        threshold=strict_cutoff,
        limit=int(limit),
    )


def _windows(
    frame_count: int,
    windows: Mapping[Any, tuple[int, int]],
) -> dict[Any, tuple[int, int]]:
    if not windows:
        raise ValueError("at least one burst window is required")
    result = {}
    for key, bounds in windows.items():
        if len(bounds) != 2:
            raise ValueError("each burst window must be (start, stop_exclusive)")
        start, stop = map(int, bounds)
        if not 0 <= start < stop <= frame_count:
            raise ValueError(f"invalid burst window {key!r}: {(start, stop)}")
        result[key] = (start, stop)
    return result


def extract_burst_candidates(
    scores: Any,
    burst_windows: Mapping[Any, tuple[int, int]],
    *,
    threshold_z: float,
    nms_distance_px: int = NMS_DISTANCE_PX,
    limit_per_burst: int = 10_000,
) -> BurstCandidateResult:
    """Threshold frames, aggregate occupancy, then run strict separated NMS."""

    values = _video(scores, "scores")
    windows = _windows(values.shape[0], burst_windows)
    if int(nms_distance_px) not in ALLOWED_NMS_DISTANCES_PX:
        raise ValueError(
            f"nms_distance_px must be one of {ALLOWED_NMS_DISTANCES_PX}; "
            f"{NMS_DISTANCE_PX} px remains primary"
        )
    if int(limit_per_burst) < 1:
        raise ValueError("limit_per_burst must be positive")
    occupancy_maps: dict[Any, np.ndarray] = {}
    peaks: dict[Any, tuple[Peak, ...]] = {}
    for key, (start, stop) in windows.items():
        occupancy = temporal_threshold_occupancy(
            values[start:stop],
            threshold_z=threshold_z,
        )
        occupancy_maps[key] = occupancy
        peaks[key] = tuple(
            strict_separated_nms(
                occupancy,
                distance_px=nms_distance_px,
                threshold=0.0,
                limit=limit_per_burst,
            )
        )
    diagnostics = {
        "burst_aggregation": "temporal_threshold_occupancy",
        "temporal_max_pooling_used": False,
        "frame_decision": "score_strictly_greater_than_threshold_z",
        "spatial_decision": "occupancy_strictly_greater_than_zero",
        "threshold_z": float(threshold_z),
        "nms": "deterministic_greedy_euclidean_separated",
        "nms_distance_px": int(nms_distance_px),
        "nms_role": (
            "primary"
            if int(nms_distance_px) == NMS_DISTANCE_PX
            else "descriptive_sensitivity"
        ),
        "burst_windows_used": True,
        "sparse_positive_coordinates_used": False,
        "sparse_positive_identities_used": False,
    }
    return BurstCandidateResult(occupancy_maps, peaks, diagnostics)


def _threshold_grid(
    training_quiet_scores: np.ndarray,
    *,
    max_candidates: int,
) -> np.ndarray:
    values = np.asarray(training_quiet_scores, dtype=np.float64).ravel()
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("training quiet scores must contain finite samples")
    if max_candidates < 16:
        raise ValueError("max_threshold_candidates must be at least 16")
    if values.size <= 1_000_000:
        candidates = np.unique(values)
        if candidates.size > max_candidates - 1:
            quantiles = np.linspace(0.0, 1.0, max_candidates - 1)
            candidates = np.quantile(values, quantiles, method="nearest")
    else:
        broad_count = max(8, max_candidates // 4)
        tail_count = max_candidates - broad_count - 1
        broad = np.linspace(0.0, 0.9, broad_count, endpoint=False)
        minimum_tail = max(1.0 / values.size, np.finfo(np.float64).eps)
        upper_tail = 1.0 - np.geomspace(minimum_tail, 0.1, tail_count)
        quantiles = np.unique(np.concatenate((broad, upper_tail, [1.0])))
        candidates = np.quantile(values, quantiles, method="nearest")
    minimum = float(np.min(values))
    below_minimum = float(np.nextafter(minimum, -np.inf))
    if not math.isfinite(below_minimum):
        below_minimum = minimum
    return np.unique(np.concatenate(([below_minimum], candidates))).astype(np.float64)


def calibrate_training_quiet_thresholds(
    scores: Any,
    training_quiet_mask: Any,
    target_durations: Mapping[Any, int] | Sequence[int],
    *,
    target_peak_burdens: Sequence[float] = QUIET_NMS_PEAK_BURDENS,
    nms_distance_px: int = NMS_DISTANCE_PX,
    pseudo_burst_starts: Mapping[Any, int] | None = None,
    max_threshold_candidates: int = 129,
) -> QuietThresholdCalibration:
    """Freeze score cutoffs from duration-matched training-quiet pseudo-bursts.

    For each declared burden, the selected empirical threshold maximizes the
    achieved burden without exceeding the target; ties select the lower score
    threshold.  These are empirical operating points, not PFA estimates.
    """

    values = _video(scores, "scores")
    quiet = _training_quiet_mask(training_quiet_mask, values.shape[0])
    targets = tuple(float(value) for value in target_peak_burdens)
    if targets != QUIET_NMS_PEAK_BURDENS:
        raise ValueError(
            f"target_peak_burdens must be exactly {QUIET_NMS_PEAK_BURDENS}"
        )
    if int(nms_distance_px) not in ALLOWED_NMS_DISTANCES_PX:
        raise ValueError(
            f"nms_distance_px must be one of {ALLOWED_NMS_DISTANCES_PX}; "
            f"{NMS_DISTANCE_PX} px remains primary"
        )
    windows = duration_matched_quiet_windows(
        quiet,
        target_durations,
        starts=pseudo_burst_starts,
    )
    threshold_candidates = _threshold_grid(
        values[quiet],
        max_candidates=int(max_threshold_candidates),
    )
    pseudo_burst_count = len(windows)
    maximum_allowed_total = int(math.floor(max(targets) * pseudo_burst_count))
    count_limit = maximum_allowed_total + 1
    evaluated: list[dict[str, Any]] = []
    for threshold in threshold_candidates:
        result = extract_burst_candidates(
            values,
            windows,
            threshold_z=float(threshold),
            nms_distance_px=nms_distance_px,
            limit_per_burst=count_limit,
        )
        counts = {key: len(peaks) for key, peaks in result.peaks.items()}
        total = int(sum(counts.values()))
        evaluated.append(
            {
                "threshold_z": float(threshold),
                "peak_counts": counts,
                "total_peaks": total,
                "peaks_per_pseudo_burst": float(total / pseudo_burst_count),
                "count_is_lower_bound_above_largest_target": bool(
                    any(count >= count_limit for count in counts.values())
                ),
            }
        )

    operating_points = []
    for target in targets:
        allowed_total = target * pseudo_burst_count
        feasible = [row for row in evaluated if row["total_peaks"] <= allowed_total]
        if not feasible:  # The maximum observed score always yields zero peaks.
            raise RuntimeError(f"no quiet threshold satisfies target burden {target}")
        selected = min(
            feasible,
            key=lambda row: (
                allowed_total - row["total_peaks"],
                row["threshold_z"],
            ),
        )
        operating_points.append(
            {
                "target_nms_peaks_per_pseudo_burst": float(target),
                "threshold_z": float(selected["threshold_z"]),
                "achieved_nms_peaks_per_pseudo_burst": float(
                    selected["peaks_per_pseudo_burst"]
                ),
                "total_nms_peaks": int(selected["total_peaks"]),
                "peak_counts_by_pseudo_burst": dict(selected["peak_counts"]),
                "pseudo_burst_count": pseudo_burst_count,
                "probability_of_false_alarm_claimed": False,
            }
        )

    overlap_frames = 0
    coverage = np.zeros(values.shape[0], dtype=np.int32)
    for start, stop in windows.values():
        coverage[start:stop] += 1
    overlap_frames = int(np.count_nonzero(coverage > 1))
    diagnostics = {
        "calibration_scope": "training_quiet_frames_only",
        "duration_matching": "one_pseudo_burst_per_declared_target_duration",
        "pseudo_burst_windows_overlap_frames": overlap_frames,
        "threshold_candidate_count": int(threshold_candidates.size),
        "threshold_candidates_derived_from_training_quiet_only": True,
        "selection_rule": (
            "largest empirical NMS burden not exceeding target; lower threshold tie break"
        ),
        "frame_aggregation": "temporal_threshold_occupancy",
        "temporal_max_pooling_used": False,
        "nms_distance_px": int(nms_distance_px),
        "nms_role": (
            "primary"
            if int(nms_distance_px) == NMS_DISTANCE_PX
            else "descriptive_sensitivity"
        ),
        "probability_of_false_alarm_claimed": False,
        "unmatched_candidates_interpreted": False,
        "quiet_interval_used": True,
        "sparse_positive_coordinates_used": False,
        "sparse_positive_identities_used": False,
    }
    return QuietThresholdCalibration(tuple(operating_points), windows, diagnostics)


def _validated_peaks(peaks: Sequence[Peak], burst_id: Any) -> list[Peak]:
    result = []
    previous_score = float("inf")
    for peak in peaks:
        if len(peak) != 3:
            raise ValueError(f"burst {burst_id!r} has an invalid peak")
        score, x_value, y_value = peak
        score = float(score)
        x, y = int(x_value), int(y_value)
        if not math.isfinite(score) or x != x_value or y != y_value:
            raise ValueError(f"burst {burst_id!r} peaks must be finite score/int x/int y")
        if score > previous_score:
            raise ValueError(f"burst {burst_id!r} peaks must be score-ranked")
        previous_score = score
        result.append((score, x, y))
    return result


def evaluate_sparse_positive_recall(
    burst_peaks: Mapping[Any, Sequence[Peak]],
    sparse_positives: Sequence[Mapping[str, Any]],
    *,
    budgets: Sequence[int] = CANDIDATE_BUDGETS_PER_BURST,
    match_radius_px: float = MATCH_RADIUS_PX,
) -> dict[str, Any]:
    """Evaluate frozen candidates against sparse positives at fixed budgets.

    This is intentionally the only public function in this module that accepts
    known-positive coordinates.  Candidates not matched to those coordinates
    remain unknown; no precision estimate is produced.
    """

    frozen_budgets = tuple(int(value) for value in budgets)
    if frozen_budgets != CANDIDATE_BUDGETS_PER_BURST:
        raise ValueError(
            f"budgets must be exactly {CANDIDATE_BUDGETS_PER_BURST}"
        )
    if float(match_radius_px) != MATCH_RADIUS_PX:
        raise ValueError(f"this experiment freezes match radius at {MATCH_RADIUS_PX} px")
    if not sparse_positives:
        raise ValueError("sparse_positives cannot be empty")

    peaks_by_burst: dict[int, list[Peak]] = {}
    for raw_burst, peaks in burst_peaks.items():
        burst = int(raw_burst)
        if burst in peaks_by_burst:
            raise ValueError("burst_peaks contains duplicate normalized burst IDs")
        peaks_by_burst[burst] = _validated_peaks(peaks, raw_burst)

    positives_by_burst: dict[int, list[Mapping[str, Any]]] = {}
    observed_ids: set[str] = set()
    for index, row in enumerate(sparse_positives):
        if not {"burst_id", "x_px", "y_px"}.issubset(row):
            raise ValueError("each sparse positive requires burst_id, x_px, and y_px")
        x, y = float(row["x_px"]), float(row["y_px"])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("sparse-positive coordinates must be finite")
        observation_id = str(row.get("observation_id", f"row_{index}"))
        if observation_id in observed_ids:
            raise ValueError("sparse-positive observation_id values must be unique")
        observed_ids.add(observation_id)
        positives_by_burst.setdefault(int(row["burst_id"]), []).append(row)
    if set(peaks_by_burst) != set(positives_by_burst):
        raise ValueError(
            "burst_peaks and sparse_positives must cover the same burst identifiers"
        )

    rows = []
    matched_totals = {budget: 0 for budget in frozen_budgets}
    label_total = sum(len(rows_for_burst) for rows_for_burst in positives_by_burst.values())
    for burst in sorted(peaks_by_burst):
        peaks = peaks_by_burst[burst]
        positives = positives_by_burst[burst]
        for budget in frozen_budgets:
            selected = peaks[:budget]
            matches, matched_peak_indices = match_peaks_one_to_one(
                selected,
                positives,
                MATCH_RADIUS_PX,
            )
            matched = len(matches)
            matched_totals[budget] += matched
            rows.append(
                {
                    "burst_id": burst,
                    "candidate_budget": budget,
                    "effective_candidate_count": len(selected),
                    "known_positive_count": len(positives),
                    "matched_known_positive_count": matched,
                    "known_positive_recall": matched / len(positives),
                    "unmatched_candidate_count": len(selected) - len(matched_peak_indices),
                    "unmatched_candidates": "unknown_not_negative",
                }
            )
    recall_by_budget = {
        str(budget): matched_totals[budget] / label_total
        for budget in frozen_budgets
    }
    macro_recall_by_budget = {
        str(budget): float(
            np.mean(
                [
                    row["known_positive_recall"]
                    for row in rows
                    if row["candidate_budget"] == budget
                ]
            )
        )
        for budget in frozen_budgets
    }
    return {
        "rows": rows,
        "candidate_budgets_per_burst": list(frozen_budgets),
        "match_radius_px": MATCH_RADIUS_PX,
        "one_to_one_matching": True,
        "total_known_positive_occurrences": label_total,
        "matched_known_positives_by_budget": {
            str(key): value for key, value in matched_totals.items()
        },
        "pooled_known_positive_recall_by_budget": recall_by_budget,
        "macro_known_positive_recall_by_budget": macro_recall_by_budget,
        "unmatched_candidates": "unknown_not_negative",
        "precision_identified": False,
    }


def paired_representation_equivalence(
    reference: Any,
    candidate: Any,
    *,
    top_fraction: float = 0.01,
) -> dict[str, Any]:
    """Return correlation, scale-adjusted nRMS, and absolute-tail Jaccard."""

    first = _numpy(reference).astype(np.float64, copy=False)
    second = _numpy(candidate).astype(np.float64, copy=False)
    if first.shape != second.shape or first.size < 2:
        raise ValueError("paired representations must share a shape with at least two values")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("paired representations must be finite")
    fraction = float(top_fraction)
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("top_fraction must be in (0,1]")
    x, y = first.ravel(), second.ravel()
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    centered_norm = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    x_energy = float(np.dot(x, x))
    y_rms = float(np.sqrt(np.mean(y * y)))
    epsilon = np.finfo(np.float64).eps
    if centered_norm <= epsilon or x_energy <= epsilon or y_rms <= epsilon:
        raise ValueError("paired equivalence requires non-constant, non-zero representations")
    correlation = float(np.dot(x_centered, y_centered) / centered_norm)
    from scipy.stats import spearmanr

    spearman = float(spearmanr(x, y).statistic)
    if not math.isfinite(spearman):
        raise ValueError("Spearman correlation is undefined for these representations")
    scale = float(np.dot(y, x) / x_energy)
    residual = y - scale * x
    nrms = float(np.sqrt(np.mean(residual * residual)) / y_rms)
    top_count = max(1, int(round(x.size * fraction)))
    indices = np.arange(x.size)
    reference_top = set(
        np.lexsort((indices, -np.abs(x)))[:top_count].tolist()
    )
    candidate_top = set(
        np.lexsort((indices, -np.abs(y)))[:top_count].tolist()
    )
    intersection = len(reference_top & candidate_top)
    union = len(reference_top | candidate_top)
    return {
        "paired_value_count": int(x.size),
        "correlation": correlation,
        "pearson_correlation": correlation,
        "spearman_correlation": spearman,
        "nrms": nrms,
        "nrms_definition": (
            "RMS(candidate-beta*reference)/RMS(candidate), beta least-squares fitted"
        ),
        "least_squares_reference_to_candidate_scale": scale,
        "top_fraction": fraction,
        "top_count": top_count,
        "top_1pct_jaccard": float(intersection / union),
        "top_set_definition": "largest absolute values with flat-index tie break",
        "sparse_positive_coordinates_used": False,
        "sparse_positive_identities_used": False,
    }


__all__ = [
    "BurstCandidateResult",
    "ALLOWED_NMS_DISTANCES_PX",
    "CANDIDATE_BUDGETS_PER_BURST",
    "LocalStdScaleFloorFit",
    "MATCH_RADIUS_PX",
    "NMS_DISTANCE_PX",
    "NMS_SENSITIVITY_DISTANCES_PX",
    "QUIET_NMS_PEAK_BURDENS",
    "QuietThresholdCalibration",
    "calibrate_training_quiet_thresholds",
    "duration_matched_quiet_windows",
    "evaluate_sparse_positive_recall",
    "extract_burst_candidates",
    "fit_training_quiet_scale_floor",
    "paired_representation_equivalence",
    "strict_separated_nms",
    "temporal_threshold_occupancy",
]
