"""Truth-bounded evaluation utilities for frozen spatiotemporal representations.

The primary endpoint in this module is recovery of sources that were injected
into an otherwise untouched real movie crop.  The native crop is paired with
the source-on crop, so a score-map difference measures the *incremental*
effect of the known injection.  It does not turn native biological structure
into negative truth and it is not a precision estimate.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
import hashlib
import math

import numpy as np
from scipy.ndimage import label, maximum_filter
from scipy.optimize import linear_sum_assignment

from neurobench.experiments.information_source_separation.semi_synthetic import (
    SemiSyntheticFixture,
)


ScoreMapFunction = Callable[[np.ndarray], np.ndarray]


class InjectionEvaluationError(ValueError):
    """Raised when a paired-injection evaluation contract is invalid."""


def additive_closure_metrics(
    observation: np.ndarray,
    native_background: np.ndarray,
    injected_signal: np.ndarray,
) -> dict[str, float]:
    """Measure additive closure in absolute units and float32 ULPs."""

    observed = np.asarray(observation, dtype=np.float32)
    expected = np.asarray(native_background, dtype=np.float64) + np.asarray(
        injected_signal, dtype=np.float64
    )
    closure = observed.astype(np.float64) - expected
    spacing = np.abs(np.spacing(observed)).astype(np.float64)
    ratio = np.divide(
        np.abs(closure),
        spacing,
        out=np.full_like(closure, np.inf),
        where=spacing > 0,
    )
    return {
        "maximum_absolute": float(np.max(np.abs(closure))),
        "maximum_float32_ulp": float(np.max(ratio)),
    }


@dataclass(frozen=True)
class RecoveryResult:
    """One-to-one recovery of exact injected identities at a fixed budget."""

    recovered_sources: int
    injected_sources: int
    candidate_count: int
    unmatched_candidate_count: int
    localization_errors_px: tuple[float, ...]
    matched_candidate_indices: tuple[int, ...]
    matched_source_indices: tuple[int, ...]

    @property
    def recall(self) -> float:
        return self.recovered_sources / self.injected_sources

    def to_dict(self) -> dict[str, object]:
        return {
            "recovered_sources": self.recovered_sources,
            "injected_sources": self.injected_sources,
            "candidate_count": self.candidate_count,
            "unmatched_candidate_count_unknown": self.unmatched_candidate_count,
            "recall": self.recall,
            "localization_errors_px": list(self.localization_errors_px),
            "matched_candidate_indices": list(self.matched_candidate_indices),
            "matched_source_indices": list(self.matched_source_indices),
        }


def _injection_centers_yx(
    shape_yx: tuple[int, int],
    *,
    source_count: int,
    seed: int,
    border_px: int,
    minimum_separation_px: float,
    crowding_case: str,
) -> tuple[tuple[float, float], ...]:
    height, width = shape_yx
    if source_count < 1:
        raise InjectionEvaluationError("source_count must be positive")
    if min(height, width) <= 2 * border_px:
        raise InjectionEvaluationError("injection border excludes the complete crop")
    rng = np.random.default_rng(seed)
    centers: list[tuple[float, float]] = []
    if crowding_case not in {"single", "separated", "near_neighbor", "overlap"}:
        raise InjectionEvaluationError(f"unsupported crowding_case: {crowding_case}")
    if source_count == 1 and crowding_case != "single":
        raise InjectionEvaluationError("one-source fixtures require crowding_case='single'")
    if source_count >= 2 and crowding_case == "single":
        raise InjectionEvaluationError("multi-source fixtures require a crowding case")

    if crowding_case in {"near_neighbor", "overlap"}:
        forced_distance = 6.0 if crowding_case == "near_neighbor" else 3.75
        margin = border_px + forced_distance
        if height <= 2 * margin or width <= 2 * margin:
            raise InjectionEvaluationError("crop is too small for the forced crowded pair")
        first_row = float(rng.uniform(margin, height - margin))
        first_column = float(rng.uniform(margin, width - margin))
        angle = float(rng.uniform(0, 2 * np.pi))
        centers.extend(
            [
                (first_row, first_column),
                (
                    first_row + forced_distance * math.sin(angle),
                    first_column + forced_distance * math.cos(angle),
                ),
            ]
        )
        if len(centers) == source_count:
            return tuple(centers)
    maximum_attempts = max(2_000, 500 * source_count)
    for _ in range(maximum_attempts):
        row = float(rng.uniform(border_px, height - border_px))
        column = float(rng.uniform(border_px, width - border_px))
        if all(
            math.hypot(row - previous_row, column - previous_column)
            >= minimum_separation_px
            for previous_row, previous_column in centers
        ):
            centers.append((row, column))
            if len(centers) == source_count:
                return tuple(centers)
    raise InjectionEvaluationError(
        "could not place separated injected sources in the requested crop"
    )


def make_native_background_injection(
    native_background: np.ndarray,
    *,
    fixture_id: str,
    source_count: int,
    seed: int,
    amplitude_multiplier: float = 1.0,
    border_px: int = 8,
    minimum_separation_px: float = 10.0,
    crowding_case: str = "mixed",
) -> SemiSyntheticFixture:
    """Add exact calcium-like ellipse/crescent sources to a native raw clip.

    Each source is scaled independently to the robust frame-difference noise
    of the paired native clip.  The native channel is retained intact and is
    never decomposed into biological truth.
    """

    native = np.asarray(native_background, dtype=np.float32)
    if native.ndim != 3 or min(native.shape) < 4 or not np.isfinite(native).all():
        raise InjectionEvaluationError("native background must be a finite TYX clip")
    if not fixture_id.strip():
        raise InjectionEvaluationError("fixture_id is required")
    if amplitude_multiplier <= 0 or not math.isfinite(amplitude_multiplier):
        raise InjectionEvaluationError("amplitude_multiplier must be finite and positive")
    resolved_crowding = crowding_case
    if crowding_case == "mixed":
        resolved_crowding = (
            "single"
            if source_count == 1
            else ("overlap", "near_neighbor", "separated")[seed % 3]
        )
    pattern_seed = int.from_bytes(
        hashlib.sha256(f"{fixture_id}|{int(seed)}".encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    ) % (2**63)
    centers = _injection_centers_yx(
        tuple(int(value) for value in native.shape[1:]),
        source_count=source_count,
        seed=pattern_seed,
        border_px=border_px,
        minimum_separation_px=minimum_separation_px,
        crowding_case=resolved_crowding,
    )
    rng = np.random.default_rng((pattern_seed + 1_000_003) % (2**63))
    rows, columns = np.mgrid[: native.shape[1], : native.shape[2]]
    footprints: list[np.ndarray] = []
    morphology: list[str] = []
    for source_index, (center_row, center_column) in enumerate(centers):
        angle = float(rng.uniform(0, np.pi))
        cosine, sine = math.cos(angle), math.sin(angle)
        relative_column = columns - center_column
        relative_row = rows - center_row
        rotated_column = cosine * relative_column + sine * relative_row
        rotated_row = -sine * relative_column + cosine * relative_row
        broad = np.exp(-0.5 * ((rotated_column / 2.5) ** 2 + (rotated_row / 1.65) ** 2))
        if (source_index + pattern_seed) % 2:
            cut = np.exp(
                -0.5
                * (
                    ((rotated_column - 1.15) / 1.85) ** 2
                    + ((rotated_row + 0.15) / 1.35) ** 2
                )
            )
            footprint = np.clip(broad - 0.72 * cut, 0, None)
            morphology.append("crescent")
        else:
            footprint = broad
            morphology.append("ellipse")
        footprint = footprint / max(float(np.max(footprint)), np.finfo(float).eps)
        footprints.append(footprint.astype(np.float32))

    frame_axis = np.arange(native.shape[0], dtype=np.float32)
    kernel_axis = np.arange(native.shape[0], dtype=np.float32)
    kernel = (1 - np.exp(-kernel_axis / 2.0)) * np.exp(-kernel_axis / 8.0)
    kernel /= max(float(np.max(kernel)), np.finfo(float).eps)
    traces: list[np.ndarray] = []
    earliest = 2
    latest = max(earliest + 1, native.shape[0] - 10)
    for source_index in range(source_count):
        onset = int(rng.integers(earliest, latest))
        impulse = np.zeros_like(frame_axis)
        impulse[onset] = float(rng.uniform(0.85, 1.15))
        if native.shape[0] >= 24 and rng.random() < 0.5:
            second = min(native.shape[0] - 2, onset + int(rng.integers(7, 13)))
            impulse[second] += float(rng.uniform(0.35, 0.7))
        trace = np.convolve(impulse, kernel, mode="full")[: native.shape[0]]
        trace /= max(float(np.max(trace)), np.finfo(float).eps)
        traces.append(trace.astype(np.float32))

    differences = np.diff(native.astype(np.float64), axis=0)
    difference_center = np.median(differences, axis=0, keepdims=True)
    temporal_noise = float(1.4826 * np.median(np.abs(differences - difference_center)))
    if not math.isfinite(temporal_noise) or temporal_noise <= 0:
        raise InjectionEvaluationError(
            "native frame-difference MAD must be finite and positive; "
            "the injection contract does not permit an implicit raw-unit floor"
        )
    source_peak = temporal_noise * float(amplitude_multiplier)
    footprint_array = np.stack(footprints)
    footprint_peak_centers = footprint_centers_yx(footprint_array)
    trace_array = np.stack(traces) * np.float32(source_peak)
    injected = np.einsum("st,shw->thw", trace_array, footprint_array).astype(np.float32)
    observation = (native + injected).astype(np.float32)
    closure_metrics = additive_closure_metrics(observation, native, injected)
    return SemiSyntheticFixture(
        fixture_id=fixture_id,
        observation=observation,
        native_background=native,
        injected_neural_signal=injected,
        footprints=footprint_array,
        traces=trace_array,
        metadata={
            "fixture_contract": "native_background_calcium_injection_v1",
            "source_count": int(source_count),
            "seed": int(seed),
            "top_level_injection_seed": int(seed),
            "fixture_pattern_seed": int(pattern_seed),
            "fixture_pattern_seed_derivation": "sha256(fixture_id|top_level_injection_seed)_first_u64",
            "amplitude_multiplier": float(amplitude_multiplier),
            "native_difference_mad": temporal_noise,
            "source_peak_raw_units": source_peak,
            "amplitude_floor_raw_units": None,
            "primary_localization_target": "footprint_peak_yx",
            "placement_centers_yx": [list(center) for center in centers],
            "footprint_peak_centers_yx": [
                list(center) for center in footprint_peak_centers
            ],
            "placement_to_peak_offsets_px": [
                float(math.hypot(peak[0] - placed[0], peak[1] - placed[1]))
                for placed, peak in zip(centers, footprint_peak_centers, strict=True)
            ],
            "crowding_case": resolved_crowding,
            "morphology": morphology,
            "kinetics": {
                "family": "fixed_calcium_like_double_exponential",
                "rise_frames": 2.0,
                "decay_frames": 8.0,
                "peak_normalized": True,
            },
            "native_background_is_not_decomposed_truth": True,
            "maximum_closure_absolute": closure_metrics["maximum_absolute"],
            "maximum_closure_float32_ulp": closure_metrics["maximum_float32_ulp"],
        },
    )


def footprint_centers_yx(footprints: np.ndarray) -> tuple[tuple[float, float], ...]:
    """Return the peak row/column of each exact injected footprint."""

    values = np.asarray(footprints)
    if values.ndim != 3 or values.shape[0] < 1:
        raise InjectionEvaluationError("footprints must have shape source,row,column")
    if not np.isfinite(values).all() or np.any(np.max(values, axis=(1, 2)) <= 0):
        raise InjectionEvaluationError("every footprint must be finite and nonzero")
    centers: list[tuple[float, float]] = []
    for footprint in values:
        row, column = np.unravel_index(int(np.argmax(footprint)), footprint.shape)
        centers.append((float(row), float(column)))
    return tuple(centers)


def local_maxima_yx(
    score_map: np.ndarray,
    *,
    budget: int,
    minimum_distance_px: int = 2,
    border_px: int = 2,
) -> tuple[tuple[int, int, float], ...]:
    """Select deterministic local maxima without assigning background labels."""

    score = np.asarray(score_map, dtype=np.float64)
    if score.ndim != 2 or not np.isfinite(score).all():
        raise InjectionEvaluationError("score map must be a finite row-by-column array")
    if budget < 1 or minimum_distance_px < 0 or border_px < 0:
        raise InjectionEvaluationError("budget must be positive and distances nonnegative")
    if 2 * border_px >= min(score.shape):
        raise InjectionEvaluationError("border excludes the complete score map")
    window = 2 * minimum_distance_px + 1
    maxima = score == maximum_filter(score, size=window, mode="nearest")
    if border_px:
        maxima[:border_px] = False
        maxima[-border_px:] = False
        maxima[:, :border_px] = False
        maxima[:, -border_px:] = False
    # Collapse connected equal-valued plateaus to one deterministic proposal;
    # otherwise an upsampled block score can consume the full budget at one
    # physical maximum.  Greedy Euclidean NMS then enforces the declared
    # minimum proposal separation even across distinct plateau components.
    components, component_count = label(maxima, structure=np.ones((3, 3), dtype=np.uint8))
    representatives: list[tuple[int, int, float]] = []
    for component_id in range(1, component_count + 1):
        rows, columns = np.nonzero(components == component_id)
        values = score[rows, columns]
        maximum = float(np.max(values))
        tied = np.flatnonzero(values == maximum)
        tied_rows = rows[tied]
        tied_columns = columns[tied]
        centroid_row = float(np.mean(tied_rows))
        centroid_column = float(np.mean(tied_columns))
        squared_distance = (tied_rows - centroid_row) ** 2 + (
            tied_columns - centroid_column
        ) ** 2
        # Pick the pixel nearest the plateau centroid so block-upsampled token
        # maps do not acquire a systematic top-left localization bias.  The
        # row/column keys make equidistant ties deterministic.
        order = np.lexsort((tied_columns, tied_rows, squared_distance))
        index = int(tied[int(order[0])])
        representatives.append((int(rows[index]), int(columns[index]), maximum))
    representatives.sort(key=lambda item: (-item[2], item[0], item[1]))
    selected: list[tuple[int, int, float]] = []
    for candidate in representatives:
        if all(
            math.hypot(candidate[0] - row, candidate[1] - column)
            >= minimum_distance_px
            for row, column, _ in selected
        ):
            selected.append(candidate)
            if len(selected) == budget:
                break
    return tuple(selected)


def match_injected_sources(
    candidates_yx: Sequence[tuple[int, int, float]],
    centers_yx: Sequence[tuple[float, float]],
    *,
    radius_px: float,
) -> RecoveryResult:
    """Match candidates to injected identities one-to-one within ``radius_px``."""

    if radius_px <= 0 or not math.isfinite(radius_px):
        raise InjectionEvaluationError("match radius must be finite and positive")
    if not centers_yx:
        raise InjectionEvaluationError("at least one injected source is required")
    if not candidates_yx:
        return RecoveryResult(0, len(centers_yx), 0, 0, (), (), ())
    distances = np.asarray(
        [
            [math.hypot(row - source_row, column - source_column) for source_row, source_column in centers_yx]
            for row, column, _ in candidates_yx
        ],
        dtype=np.float64,
    )
    # Augment the assignment with explicit unmatched rows/columns.  A valid
    # edge is always cheaper than leaving both endpoints unmatched, while an
    # out-of-radius edge is always more expensive.  This maximizes cardinality
    # within the radius first and minimizes localization distance second.
    candidate_count, source_count = distances.shape
    unmatched_penalty = float(radius_px + 1.0)
    invalid_penalty = 10.0 * unmatched_penalty * (candidate_count + source_count + 1)
    augmented = np.full(
        (candidate_count + source_count, source_count + candidate_count),
        invalid_penalty,
        dtype=np.float64,
    )
    augmented[:candidate_count, :source_count] = np.where(
        distances <= radius_px, distances, invalid_penalty
    )
    augmented[:candidate_count, source_count:] = unmatched_penalty
    augmented[candidate_count:, :source_count] = unmatched_penalty
    augmented[candidate_count:, source_count:] = 0.0
    assigned_rows, assigned_columns = linear_sum_assignment(augmented)
    accepted = [
        (int(candidate), int(source))
        for candidate, source in zip(assigned_rows, assigned_columns, strict=True)
        if candidate < candidate_count
        and source < source_count
        and distances[candidate, source] <= radius_px
    ]
    errors = tuple(float(distances[candidate, source]) for candidate, source in accepted)
    return RecoveryResult(
        recovered_sources=len(accepted),
        injected_sources=len(centers_yx),
        candidate_count=len(candidates_yx),
        unmatched_candidate_count=len(candidates_yx) - len(accepted),
        localization_errors_px=errors,
        matched_candidate_indices=tuple(candidate for candidate, _ in accepted),
        matched_source_indices=tuple(source for _, source in accepted),
    )


def _recovery_by_source_group(
    recovery: RecoveryResult, source_groups: Sequence[str]
) -> dict[str, dict[str, object]]:
    if len(source_groups) != recovery.injected_sources:
        raise InjectionEvaluationError("source-group labels must match injected sources")
    error_by_source = {
        source: error
        for source, error in zip(
            recovery.matched_source_indices,
            recovery.localization_errors_px,
            strict=True,
        )
    }
    result: dict[str, dict[str, object]] = {}
    for group in sorted(set(source_groups)):
        indices = [index for index, value in enumerate(source_groups) if value == group]
        matched = [index for index in indices if index in error_by_source]
        result[group] = {
            "injected_sources": len(indices),
            "recovered_sources": len(matched),
            "recall": len(matched) / len(indices),
            "localization_errors_px": [error_by_source[index] for index in matched],
        }
    return result


def _validate_fixture(fixture: SemiSyntheticFixture) -> None:
    arrays = {
        "observation": fixture.observation,
        "native_background": fixture.native_background,
        "injected_neural_signal": fixture.injected_neural_signal,
    }
    shapes = {name: np.asarray(value).shape for name, value in arrays.items()}
    if len(set(shapes.values())) != 1 or len(next(iter(shapes.values()))) != 3:
        raise InjectionEvaluationError(f"paired movie shapes differ: {shapes}")
    if not all(np.isfinite(np.asarray(value)).all() for value in arrays.values()):
        raise InjectionEvaluationError("paired movie arrays must be finite")
    closure = additive_closure_metrics(
        fixture.observation,
        fixture.native_background,
        fixture.injected_neural_signal,
    )
    if closure["maximum_float32_ulp"] > 0.51:
        raise InjectionEvaluationError(
            "source-on movie exceeds correctly rounded float32 additive closure"
        )
    footprints = np.asarray(fixture.footprints)
    if footprints.shape[1:] != shapes["observation"][1:]:
        raise InjectionEvaluationError("footprint geometry differs from movie geometry")


def _validated_score_map(
    scorer: ScoreMapFunction,
    movie: np.ndarray,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    value = np.asarray(scorer(movie), dtype=np.float64)
    if value.shape != expected_shape or not np.isfinite(value).all():
        raise InjectionEvaluationError(
            f"scorer must return finite map with shape {expected_shape}; got {value.shape}"
        )
    return value


def _background_scale(score_map: np.ndarray) -> tuple[float, float]:
    center = float(np.median(score_map))
    scale = float(1.4826 * np.median(np.abs(score_map - center)))
    if not math.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        scale = float(np.std(score_map))
    return center, max(scale, 1e-8)


def evaluate_paired_fixture(
    fixture: SemiSyntheticFixture,
    scorer: ScoreMapFunction,
    *,
    method: str,
    candidate_budget: int,
    match_radius_px: float = 3.0,
    minimum_distance_px: int = 2,
    border_px: int = 2,
) -> dict[str, object]:
    """Evaluate one frozen scorer on one exact source-off/source-on pair.

    ``source_on_recovery`` is the deployment-like fixed-budget endpoint.  The
    paired ``intervention_recovery`` is a mechanistic sensitivity endpoint
    based on source-on minus source-off scores.  Unmatched candidates remain
    unknown because the native background has no exhaustive biological truth.
    """

    _validate_fixture(fixture)
    if not method.strip():
        raise InjectionEvaluationError("method name is required")
    spatial_shape = tuple(int(value) for value in fixture.observation.shape[1:])
    fit_source_off = getattr(scorer, "fit_source_off", None)
    if callable(fit_source_off):
        # Fitting receives source-off only and returns one frozen callable used
        # unchanged on both arms.  The interface therefore cannot silently fit
        # calibration parameters on the injected/source-on distribution.
        frozen_scorer = fit_source_off(fixture.native_background)
        if not callable(frozen_scorer):
            raise InjectionEvaluationError(
                "fit_source_off must return a frozen callable scorer"
            )
        background_map = _validated_score_map(
            frozen_scorer, fixture.native_background, spatial_shape
        )
        source_on_map = _validated_score_map(
            frozen_scorer, fixture.observation, spatial_shape
        )
        score_pair_policy = "fit_source_off_then_frozen_apply_to_both_arms"
    else:
        background_map = _validated_score_map(scorer, fixture.native_background, spatial_shape)
        source_on_map = _validated_score_map(scorer, fixture.observation, spatial_shape)
        score_pair_policy = "stateless_frozen_scorer"
    background_center, background_scale = _background_scale(background_map)
    background_z = (background_map - background_center) / background_scale
    source_on_z = (source_on_map - background_center) / background_scale
    intervention = source_on_z - background_z
    centers = footprint_centers_yx(fixture.footprints)

    on_candidates = local_maxima_yx(
        source_on_z,
        budget=candidate_budget,
        minimum_distance_px=minimum_distance_px,
        border_px=border_px,
    )
    intervention_candidates = local_maxima_yx(
        intervention,
        budget=candidate_budget,
        minimum_distance_px=minimum_distance_px,
        border_px=border_px,
    )
    source_on_recovery = match_injected_sources(
        on_candidates, centers, radius_px=match_radius_px
    )
    intervention_recovery = match_injected_sources(
        intervention_candidates, centers, radius_px=match_radius_px
    )
    morphology_values = fixture.metadata.get("morphology")
    morphology = (
        tuple(str(value) for value in morphology_values)
        if isinstance(morphology_values, (tuple, list))
        and len(morphology_values) == len(centers)
        else None
    )
    morphology_sensitivity = (
        {
            "status": "reported",
            "source_on": _recovery_by_source_group(source_on_recovery, morphology),
            "intervention": _recovery_by_source_group(
                intervention_recovery, morphology
            ),
        }
        if morphology is not None
        else {"status": "unavailable_fixture_has_no_source_morphology"}
    )
    placement_sensitivity: dict[str, object]
    placement_values = fixture.metadata.get("placement_centers_yx")
    if isinstance(placement_values, (tuple, list)) and len(placement_values) == len(centers):
        placement_centers = tuple(
            (float(value[0]), float(value[1]))
            for value in placement_values
            if isinstance(value, (tuple, list)) and len(value) == 2
        )
        if len(placement_centers) != len(centers) or not all(
            math.isfinite(row) and math.isfinite(column)
            for row, column in placement_centers
        ):
            raise InjectionEvaluationError("placement centers must be finite y/x pairs")
        placement_source_on = match_injected_sources(
            on_candidates, placement_centers, radius_px=match_radius_px
        )
        placement_intervention = match_injected_sources(
            intervention_candidates, placement_centers, radius_px=match_radius_px
        )
        placement_sensitivity = {
            "status": "reported_secondary_not_primary_matching_truth",
            "target": "continuous_generative_placement_center_yx",
            "source_on_recovery": placement_source_on.to_dict(),
            "intervention_recovery": placement_intervention.to_dict(),
            "morphology_stratified": (
                {
                    "source_on": _recovery_by_source_group(
                        placement_source_on, morphology
                    ),
                    "intervention": _recovery_by_source_group(
                        placement_intervention, morphology
                    ),
                }
                if morphology is not None
                else None
            ),
        }
    else:
        placement_sensitivity = {
            "status": "unavailable_fixture_has_no_placement_centers",
            "target": "continuous_generative_placement_center_yx",
        }
    center_rows = np.asarray([round(row) for row, _ in centers], dtype=int)
    center_columns = np.asarray([round(column) for _, column in centers], dtype=int)
    center_deltas = intervention[center_rows, center_columns]
    closure = additive_closure_metrics(
        fixture.observation,
        fixture.native_background,
        fixture.injected_neural_signal,
    )
    return {
        "fixture_id": fixture.fixture_id,
        "method": method,
        "cluster_unit": "background_crop_seed",
        "candidate_budget": int(candidate_budget),
        "match_radius_px": float(match_radius_px),
        "score_pair_policy": score_pair_policy,
        "primary_localization_target": "footprint_peak_yx",
        "footprint_peak_centers_yx": [list(center) for center in centers],
        "footprint_peak_morphology_sensitivity": morphology_sensitivity,
        "placement_center_sensitivity": placement_sensitivity,
        "injected_source_count": len(centers),
        "source_on_recovery": source_on_recovery.to_dict(),
        "intervention_recovery": intervention_recovery.to_dict(),
        "median_injected_center_score_delta_background_mad": float(np.median(center_deltas)),
        "minimum_injected_center_score_delta_background_mad": float(np.min(center_deltas)),
        "maximum_pair_closure_absolute": closure["maximum_absolute"],
        "maximum_pair_closure_float32_ulp": closure["maximum_float32_ulp"],
        "interpretation": {
            "source_on": "fixed maximum-proposal-budget sensitivity to exact injected sources in a real background",
            "intervention": "incremental injected-source effect; mechanism-only, not deployment",
            "unmatched_candidates": "unknown native structure, never biological negatives",
            "precision": "unavailable without exhaustive review of the native background",
        },
    }


def evaluate_paired_suite(
    fixtures: Iterable[SemiSyntheticFixture],
    scorers: Mapping[str, ScoreMapFunction],
    *,
    candidate_budget: int,
    match_radius_px: float = 3.0,
    minimum_distance_px: int = 2,
    border_px: int = 2,
) -> list[dict[str, object]]:
    """Evaluate every method on every paired fixture in deterministic order."""

    fixture_list = sorted(fixtures, key=lambda item: item.fixture_id)
    if not fixture_list:
        raise InjectionEvaluationError("at least one paired fixture is required")
    if not scorers:
        raise InjectionEvaluationError("at least one scorer is required")
    rows: list[dict[str, object]] = []
    for fixture in fixture_list:
        for method in sorted(scorers):
            rows.append(
                evaluate_paired_fixture(
                    fixture,
                    scorers[method],
                    method=method,
                    candidate_budget=candidate_budget,
                    match_radius_px=match_radius_px,
                    minimum_distance_px=minimum_distance_px,
                    border_px=border_px,
                )
            )
    return rows


def paired_sign_flip_test(
    differences: Sequence[float],
    *,
    seed: int = 0,
    monte_carlo_draws: int = 100_000,
) -> dict[str, object]:
    """Two-sided cluster-level sign-flip test of a paired mean difference."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise InjectionEvaluationError("paired differences must contain at least two finite values")
    observed = float(np.mean(values))
    exact = values.size <= 20
    if exact:
        statistics = np.fromiter(
            (
                float(np.mean(values * np.asarray(signs, dtype=np.float64)))
                for signs in product((-1.0, 1.0), repeat=values.size)
            ),
            dtype=np.float64,
            count=2 ** values.size,
        )
    else:
        if monte_carlo_draws < 1000:
            raise InjectionEvaluationError("Monte Carlo sign-flip test needs at least 1000 draws")
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(monte_carlo_draws, values.size))
        statistics = np.mean(signs * values[None, :], axis=1)
    exceedances = int(np.count_nonzero(np.abs(statistics) >= abs(observed) - 1e-15))
    p_value = exceedances / len(statistics) if exact else (exceedances + 1) / (len(statistics) + 1)
    return {
        "cluster_count": int(values.size),
        "mean_difference": observed,
        "median_difference": float(np.median(values)),
        "two_sided_p_value": float(p_value),
        "exact": exact,
        "permutations": int(len(statistics)),
        "cluster_unit": "background_crop_seed",
    }


def hierarchical_grouped_bootstrap(
    cluster_rows: Sequence[Mapping[str, object]],
    *,
    value_key: str,
    recording_key: str = "background_recording_id",
    window_key: str = "background_window_id",
    seed_key: str = "injection_seed",
    draws: int = 5_000,
    seed: int = 0,
) -> dict[str, object]:
    """Bootstrap a paired effect through recording/window/injection hierarchy.

    Rows must already aggregate nested source-count conditions so those source
    counts cannot masquerade as independent replicates.
    """

    if draws < 1_000:
        raise InjectionEvaluationError("grouped bootstrap requires at least 1000 draws")
    required = (recording_key, window_key, seed_key, value_key)
    parsed: dict[str, dict[str, dict[str, float]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for row in cluster_rows:
        if any(key not in row for key in required):
            raise InjectionEvaluationError(f"bootstrap row misses one of {required}")
        recording = str(row[recording_key])
        window = str(row[window_key])
        injection_seed = str(row[seed_key])
        key = (recording, window, injection_seed)
        if key in seen:
            raise InjectionEvaluationError(f"duplicate grouped-bootstrap cluster: {key}")
        seen.add(key)
        value = float(row[value_key])
        if not math.isfinite(value):
            raise InjectionEvaluationError("bootstrap values must be finite")
        parsed.setdefault(recording, {}).setdefault(window, {})[injection_seed] = value
    if len(parsed) < 2:
        raise InjectionEvaluationError("grouped bootstrap requires at least two recordings")
    observed_values = [
        value
        for windows in parsed.values()
        for seeds in windows.values()
        for value in seeds.values()
    ]
    rng = np.random.default_rng(seed)
    recording_ids = sorted(parsed)
    samples = np.empty(draws, dtype=np.float64)
    for draw_index in range(draws):
        selected_recordings = rng.choice(recording_ids, size=len(recording_ids), replace=True)
        selected_values: list[float] = []
        for recording in selected_recordings:
            windows = parsed[str(recording)]
            window_ids = sorted(windows)
            selected_windows = rng.choice(window_ids, size=len(window_ids), replace=True)
            for window in selected_windows:
                injection_seeds = windows[str(window)]
                injection_seed_ids = sorted(injection_seeds)
                selected_seeds = rng.choice(
                    injection_seed_ids, size=len(injection_seed_ids), replace=True
                )
                selected_values.extend(injection_seeds[str(item)] for item in selected_seeds)
        samples[draw_index] = float(np.mean(selected_values))
    return {
        "observed_mean": float(np.mean(observed_values)),
        "confidence_interval_95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "draws": int(draws),
        "bootstrap_seed": int(seed),
        "hierarchy": [recording_key, window_key, seed_key],
        "nested_not_resampled_as_independent": ["source_count"],
        "recording_count": len(recording_ids),
        "cluster_count": len(observed_values),
    }


def hierarchical_strongest_comparator_bootstrap(
    cluster_rows: Sequence[Mapping[str, object]],
    *,
    primary_method: str,
    comparator_methods: Sequence[str],
    method_values_key: str = "method_values",
    recording_key: str = "background_recording_id",
    window_key: str = "background_window_id",
    seed_key: str = "injection_seed",
    draws: int = 5_000,
    seed: int = 0,
) -> dict[str, object]:
    """Bootstrap primary minus max comparator, reselecting max per draw."""

    comparators = tuple(str(value) for value in comparator_methods)
    methods = (str(primary_method), *comparators)
    if not primary_method or not comparators or len(set(methods)) != len(methods):
        raise InjectionEvaluationError("primary and comparator methods must be unique")
    if draws < 1_000:
        raise InjectionEvaluationError("grouped bootstrap requires at least 1000 draws")
    parsed: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for row in cluster_rows:
        required = (recording_key, window_key, seed_key, method_values_key)
        if any(key not in row for key in required):
            raise InjectionEvaluationError("strongest-comparator bootstrap row is incomplete")
        recording = str(row[recording_key])
        window = str(row[window_key])
        injection_seed = str(row[seed_key])
        cluster = (recording, window, injection_seed)
        if cluster in seen:
            raise InjectionEvaluationError(
                f"duplicate strongest-comparator bootstrap cluster: {cluster}"
            )
        seen.add(cluster)
        raw_values = row[method_values_key]
        if not isinstance(raw_values, Mapping) or set(map(str, raw_values)) != set(methods):
            raise InjectionEvaluationError(
                "every cluster must contain exactly the primary and comparator methods"
            )
        values = {method: float(raw_values[method]) for method in methods}
        if not all(math.isfinite(value) for value in values.values()):
            raise InjectionEvaluationError("method values must be finite")
        parsed.setdefault(recording, {}).setdefault(window, {})[injection_seed] = values
    if len(parsed) < 2:
        raise InjectionEvaluationError("grouped bootstrap requires at least two recordings")

    observed_rows = [
        values
        for windows in parsed.values()
        for seeds in windows.values()
        for values in seeds.values()
    ]
    observed_means = {
        method: float(np.mean([row[method] for row in observed_rows]))
        for method in methods
    }
    observed_strongest = max(
        comparators, key=lambda method: (observed_means[method], method)
    )
    observed_effect = observed_means[primary_method] - observed_means[observed_strongest]

    rng = np.random.default_rng(seed)
    recording_ids = sorted(parsed)
    samples = np.empty(draws, dtype=np.float64)
    selection_counts = {method: 0 for method in comparators}
    for draw_index in range(draws):
        selected_recordings = rng.choice(
            recording_ids, size=len(recording_ids), replace=True
        )
        selected: list[dict[str, float]] = []
        for recording in selected_recordings:
            windows = parsed[str(recording)]
            window_ids = sorted(windows)
            selected_windows = rng.choice(
                window_ids, size=len(window_ids), replace=True
            )
            for window in selected_windows:
                injection_seeds = windows[str(window)]
                injection_seed_ids = sorted(injection_seeds)
                selected_seeds = rng.choice(
                    injection_seed_ids, size=len(injection_seed_ids), replace=True
                )
                selected.extend(injection_seeds[str(item)] for item in selected_seeds)
        draw_means = {
            method: float(np.mean([row[method] for row in selected]))
            for method in methods
        }
        strongest = max(comparators, key=lambda method: (draw_means[method], method))
        selection_counts[strongest] += 1
        samples[draw_index] = draw_means[primary_method] - draw_means[strongest]
    return {
        "observed_mean": float(observed_effect),
        "observed_method_macro_recall": observed_means,
        "observed_strongest_comparator": observed_strongest,
        "confidence_interval_95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "draws": int(draws),
        "bootstrap_seed": int(seed),
        "strongest_comparator_reselected_inside_each_draw": True,
        "bootstrap_strongest_selection_fraction": {
            method: count / draws for method, count in selection_counts.items()
        },
        "hierarchy": [recording_key, window_key, seed_key],
        "nested_not_resampled_as_independent": ["source_count"],
        "recording_count": len(recording_ids),
        "cluster_count": len(observed_rows),
    }
