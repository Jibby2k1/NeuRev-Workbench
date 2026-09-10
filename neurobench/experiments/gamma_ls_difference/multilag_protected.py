"""Protected matched-six-lag Gamma-LS representation ablation.

This experiment is deliberately separate from the adjacent-frame protected
comparison.  It refits a full-rank six-coordinate whitening and a CS-Parzen
orthogonal rotation inside each leave-one-burst-out fold, using acquisition-raw
samples from only the three training burst windows.  The archived whole-review
v5 fit is inspected only to freeze the bounded bandwidth grid; its matrices are
never loaded into a protected projection.

Candidates and thresholds are written and hashed before either sparse-positive
table is parsed.  Unmatched candidates remain unknown, so this module reports
known-positive recovery and candidate burden but never precision.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "4")

import numpy as np

from neurobench.algorithms.information_source_separation import (
    WhiteningModel,
    pca_whiten,
)
from neurobench.algorithms.multilag_msica import TemporalMSICAFit
from neurobench.algorithms.pairwise_separation import cs_parzen_objective
from neurobench.experiments.pairwise_separation.sampling import uniform_anatomy_mask

from .config import GammaLSDifferenceConfig
from .cuda_runtime import CudaRuntimeUnavailable, require_cuda_device
from .evaluation import (
    CANDIDATE_BUDGETS_PER_BURST,
    NMS_DISTANCE_PX,
    paired_representation_equivalence,
)
from .gpu_representations import (
    DeviceRepresentationMap,
    V5_LAGS,
)
from .preflight import verify_matching_preflight
from .protected import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    ContextLane,
    _fit_selection_metrics,
    _mask_interval,
    _read_sparse_positives,
    _score_candidates_for_arm,
    _summary_by_arm,
    aggregate_match_rows,
    clustered_bootstrap_contrasts,
    load_fold_context_plan,
    observation_match_rows,
)
from .screen import ENERGY_EPSILON, FoldContract, build_fold_contracts


MULTILAG_ARMS = (
    "difference_multilag_energy_normalized",
    "pca_whitened_delay_total_energy",
    "cs_parzen_delay_residual",
)
CS_PARZEN_BANDWIDTHS = (0.25, 0.35)
SAMPLE_SEEDS = (20260805, 20260806, 20260807)
SCREEN_SAMPLES = 384
CONFIRMATION_SAMPLES = 384
ANGLE_STEP_DEGREES = 5.0
MAX_SWEEPS = 6
IMPROVEMENT_TOLERANCE = 1e-4
EIGENVALUE_FLOOR_RATIO = 1e-6
KERNEL_BLOCK_ROWS = 256
KERNEL_DTYPE = np.float32
DEFAULT_CONTEXT_ROLE = "size_sufficient_candidate"
NMS_DISTANCES_PX = (4, 6, 8)


class MultilagProtectedUnavailable(RuntimeError):
    """Raised when the matched-six-lag run cannot satisfy a hard gate."""


@dataclass(frozen=True)
class FoldDelaySamples:
    """Two disjoint balanced delay-embedding samples for one outer fold."""

    screen: np.ndarray
    confirmation: np.ndarray
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        expected_screen = (len(V5_LAGS), SCREEN_SAMPLES)
        expected_confirmation = (len(V5_LAGS), CONFIRMATION_SAMPLES)
        if self.screen.shape != expected_screen:
            raise ValueError(f"screen samples must have shape {expected_screen}")
        if self.confirmation.shape != expected_confirmation:
            raise ValueError(
                f"confirmation samples must have shape {expected_confirmation}"
            )
        if not np.isfinite(self.screen).all() or not np.isfinite(
            self.confirmation
        ).all():
            raise ValueError("delay samples must be finite")


@dataclass(frozen=True)
class FoldDelayWhitening:
    """One full-rank whitening shared by both CS-Parzen bandwidths."""

    screen: np.ndarray
    confirmation: np.ndarray
    model: WhiteningModel
    fit_seconds: float
    screen_observations_sha256: str
    confirmation_observations_sha256: str

    def __post_init__(self) -> None:
        dimension = len(V5_LAGS)
        if self.screen.shape != (dimension, SCREEN_SAMPLES):
            raise ValueError("whitened screen sample has the wrong shape")
        if self.confirmation.shape != (dimension, CONFIRMATION_SAMPLES):
            raise ValueError("whitened confirmation sample has the wrong shape")
        if not np.isfinite(self.screen).all() or not np.isfinite(
            self.confirmation
        ).all():
            raise ValueError("whitened delay samples must be finite")
        if (
            self.model.retained_rank != dimension
            or self.model.whitening.shape != (dimension, dimension)
            or np.linalg.matrix_rank(self.model.whitening) != dimension
        ):
            raise ValueError("delay whitening must retain all six coordinates")
        if not math.isfinite(self.fit_seconds) or self.fit_seconds < 0.0:
            raise ValueError("whitening fit time must be finite and non-negative")


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"table {path.name} has inconsistent fields")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text.rstrip() + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _artifact_index(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and path.name != "artifact_index.json"
            and not path.name.endswith(".partial")
        ):
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return {"schema_version": 1, "artifacts": rows}


def multilag_design() -> dict[str, Any]:
    """Return the predeclared bounded design and exact work counts."""

    pair_count = len(V5_LAGS) * (len(V5_LAGS) - 1) // 2
    angles_per_pair = int(round(90.0 / ANGLE_STEP_DEGREES)) + 1
    objective_calls_per_fit_upper_bound = (
        pair_count
        + MAX_SWEEPS * (pair_count * angles_per_pair + pair_count)
        + 2 * pair_count
    )
    fit_count = 4 * len(SAMPLE_SEEDS) * len(CS_PARZEN_BANDWIDTHS)
    return {
        "lags": list(V5_LAGS),
        "input_domain": "acquisition_raw",
        "arms": list(MULTILAG_ARMS),
        "cs_parzen_bandwidths": list(CS_PARZEN_BANDWIDTHS),
        "sample_seeds": list(SAMPLE_SEEDS),
        "outer_folds": 4,
        "screen_samples_per_fit": SCREEN_SAMPLES,
        "confirmation_samples_per_fit": CONFIRMATION_SAMPLES,
        "balanced_samples_per_training_burst_per_split": SCREEN_SAMPLES // 3,
        "unique_whitening_models": 4 * len(SAMPLE_SEEDS),
        "whitening_computations": 4 * len(SAMPLE_SEEDS),
        "cs_parzen_rotation_fits": fit_count,
        "pairwise_cs_objective_calls_per_fit_upper_bound": (
            objective_calls_per_fit_upper_bound
        ),
        "pairwise_cs_objective_calls_total_upper_bound": (
            fit_count * objective_calls_per_fit_upper_bound
        ),
        "gamma_selection_maps": 4
        * len(SAMPLE_SEEDS)
        * (1 + len(CS_PARZEN_BANDWIDTHS)),
        "protected_context_arm_cells": 4 * len(MULTILAG_ARMS),
        "protected_operating_point_cells": (
            4 * len(MULTILAG_ARMS) * 2 * len(NMS_DISTANCES_PX) * 5
        ),
        "candidate_budgets": list(CANDIDATE_BUDGETS_PER_BURST),
        "primary_nms_distance_px": NMS_DISTANCE_PX,
        "nms_sensitivity_distances_px": [4, 8],
        "context_role": DEFAULT_CONTEXT_ROLE,
        "archived_v5_fit_role": "grid_basis_only_never_projected",
    }


def audit_historical_v5_surface(
    surface_path: str | Path, *, selected_config_id: str
) -> dict[str, Any]:
    """Validate the prior label-free grid basis without exposing fit matrices."""

    path = Path(surface_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise ValueError("historical v5 surface is not complete")
    if payload.get("selection_labels_used") is not False:
        raise ValueError("historical v5 surface used labels for selection")
    rows = []
    selected_seen = False
    for row in payload.get("expansion_rows", []):
        if (
            row.get("formulation") != "delay_embedding"
            or row.get("objective_family") != "cs_parzen"
            or row.get("profile") != "long"
        ):
            continue
        fit = row.get("fit", {})
        if tuple(int(value) for value in fit.get("lags", ())) != V5_LAGS:
            raise ValueError("historical long CS-Parzen fit changed its lag support")
        bandwidth = float(row.get("parameter", {}).get("bandwidth"))
        if bandwidth not in CS_PARZEN_BANDWIDTHS:
            continue
        config_id = str(row.get("config_id"))
        selected_seen = selected_seen or config_id == selected_config_id
        baseline = float(fit["baseline_objective"])
        objective = float(fit["objective"])
        rows.append(
            {
                "config_id": config_id,
                "bandwidth": bandwidth,
                "held_out_gain_fraction": float(row["held_out_gain_fraction"]),
                "confirmation_baseline_objective": baseline,
                "confirmation_objective": objective,
                "converged": bool(fit["converged"]),
                "accepted_updates": int(fit["diagnostics"]["accepted_updates"]),
                "sweeps": int(fit["diagnostics"]["sweeps"]),
            }
        )
    if {row["bandwidth"] for row in rows} != set(CS_PARZEN_BANDWIDTHS):
        raise ValueError("historical v5 surface lacks the two frozen bandwidths")
    if not selected_seen:
        raise ValueError("configured historical v5 selection is absent from the surface")
    return {
        "surface_path": str(path),
        "surface_sha256": _sha256(path),
        "selected_config_id": selected_config_id,
        "bounded_grid_basis": sorted(rows, key=lambda row: row["bandwidth"]),
        "fit_matrices_loaded_for_protected_projection": False,
        "historical_role": "transductive_grid_basis_and_parity_only",
    }


def select_fold_contexts(
    plan: Mapping[int, Sequence[ContextLane]], *, role: str
) -> dict[int, ContextLane]:
    """Select exactly one already-frozen context role in every outer fold."""

    if not role:
        raise ValueError("context role cannot be empty")
    selected: dict[int, ContextLane] = {}
    for fold, lanes in sorted(plan.items()):
        matches = [lane for lane in lanes if lane.role == role]
        if len(matches) != 1:
            raise ValueError(
                f"outer fold {fold} must contain exactly one context role {role!r}"
            )
        lane = matches[0]
        if not lane.selection_source:
            raise ValueError("selected context lacks pre-label selection provenance")
        selected[int(fold)] = lane
    if set(selected) != {1, 2, 3, 4}:
        raise ValueError("matched-six-lag comparison requires exactly four folds")
    return selected


def eligible_delay_current_frames(
    fold: FoldContract,
    bursts: Mapping[str, Sequence[int]],
    *,
    lags: Sequence[int] = V5_LAGS,
) -> dict[str, np.ndarray]:
    """Return training-burst current frames whose complete history clears guard."""

    lag_values = tuple(int(value) for value in lags)
    if lag_values != V5_LAGS:
        raise ValueError(f"lags must be exactly {V5_LAGS}")
    guard_start, guard_stop = map(int, fold.heldout_guard_ui)
    output: dict[str, np.ndarray] = {}
    for burst_id in fold.training_bursts:
        start, stop = map(int, bursts[str(burst_id)])
        current = np.arange(start, stop + 1, dtype=np.int64)
        eligible = np.ones(len(current), dtype=bool)
        for lag in lag_values:
            source_ui = current - lag
            eligible &= (source_ui < guard_start) | (source_ui > guard_stop)
        retained = current[eligible]
        if retained.size == 0:
            raise ValueError(f"training burst {burst_id} has no guard-safe delay anchors")
        output[str(burst_id)] = retained
    if set(output) != set(fold.training_bursts):
        raise AssertionError("delay fit escaped the three outer-training bursts")
    return output


def guard_safe_selection_bursts(
    fold: FoldContract, bursts: Mapping[str, Sequence[int]]
) -> dict[str, list[int]]:
    """Trim training event windows so every selected map has safe lag history."""

    eligible = eligible_delay_current_frames(fold, bursts)
    result = {str(key): [int(value[0]), int(value[1])] for key, value in bursts.items()}
    for burst_id, frames in eligible.items():
        if frames.size < 1 or (frames.size > 1 and not np.all(np.diff(frames) == 1)):
            raise ValueError(
                "guard-safe selection frames must remain one contiguous interval"
            )
        result[burst_id] = [int(frames[0]), int(frames[-1])]
    return result


def _gather_delay_columns(
    movie: Any,
    current_ui: np.ndarray,
    flat_pixels: np.ndarray,
    *,
    shape_yx: tuple[int, int],
) -> np.ndarray:
    y, x = np.unravel_index(flat_pixels, shape_yx)
    current_zero = np.asarray(current_ui, dtype=np.int64) - 1
    rows = [
        np.asarray(movie[current_zero - lag, y, x], dtype=np.float64)
        for lag in V5_LAGS
    ]
    result = np.stack(rows, axis=0)
    if result.shape[0] != len(V5_LAGS) or not np.isfinite(result).all():
        raise RuntimeError("sampled delay embedding is invalid")
    return result


def sample_fold_delay_observations(
    movie: Any,
    *,
    fold: FoldContract,
    bursts: Mapping[str, Sequence[int]],
    anatomy_mask: np.ndarray,
    seed: int,
) -> FoldDelaySamples:
    """Sample balanced, disjoint screen/confirmation columns from training bursts."""

    if SCREEN_SAMPLES % 3 or CONFIRMATION_SAMPLES % 3:
        raise AssertionError("sample counts must divide evenly over three bursts")
    if getattr(movie, "ndim", None) != 3:
        raise ValueError("movie must be TYX")
    mask = np.asarray(anatomy_mask, dtype=bool)
    if mask.shape != tuple(movie.shape[1:]) or not mask.any():
        raise ValueError("anatomy mask must be non-empty and match movie geometry")
    pixels = np.flatnonzero(mask.ravel())
    frames_by_burst = eligible_delay_current_frames(fold, bursts)
    rng = np.random.default_rng(int(seed))
    screen_columns = []
    confirmation_columns = []
    identity_payload = []
    counts: dict[str, dict[str, int]] = {}
    for burst_id in fold.training_bursts:
        current_frames = frames_by_burst[str(burst_id)]
        population = int(len(current_frames) * len(pixels))
        screen_count = SCREEN_SAMPLES // 3
        confirmation_count = CONFIRMATION_SAMPLES // 3
        total = screen_count + confirmation_count
        if total > population:
            raise ValueError("requested delay samples exceed a training-burst population")
        identities = rng.choice(population, size=total, replace=False)
        frame_local = identities // len(pixels)
        pixel_local = identities % len(pixels)
        selected_frames = current_frames[frame_local]
        selected_pixels = pixels[pixel_local]
        screen_columns.append(
            _gather_delay_columns(
                movie,
                selected_frames[:screen_count],
                selected_pixels[:screen_count],
                shape_yx=mask.shape,
            )
        )
        confirmation_columns.append(
            _gather_delay_columns(
                movie,
                selected_frames[screen_count:],
                selected_pixels[screen_count:],
                shape_yx=mask.shape,
            )
        )
        identity_payload.append(
            np.column_stack(
                (
                    np.full(total, int(burst_id), dtype=np.int64),
                    selected_frames,
                    selected_pixels,
                )
            )
        )
        counts[str(burst_id)] = {
            "eligible_current_frames": int(len(current_frames)),
            "screen_samples": screen_count,
            "confirmation_samples": confirmation_count,
            "population": population,
        }
    screen = np.concatenate(screen_columns, axis=1)
    confirmation = np.concatenate(confirmation_columns, axis=1)
    identities = np.concatenate(identity_payload, axis=0).astype("<i8", copy=False)
    manifest = {
        "seed": int(seed),
        "lags": list(V5_LAGS),
        "input_domain": "acquisition_raw",
        "training_bursts": list(fold.training_bursts),
        "heldout_burst": fold.heldout_burst,
        "heldout_guard_ui": list(fold.heldout_guard_ui),
        "guard_rule": "every_t_minus_lag_outside_inclusive_heldout_guard",
        "sampling_balance": "equal_columns_per_outer_training_burst",
        "anatomy_pixel_count": int(len(pixels)),
        "screen_sample_count": int(screen.shape[1]),
        "confirmation_sample_count": int(confirmation.shape[1]),
        "screen_confirmation_disjoint": True,
        "counts_by_training_burst": counts,
        "sample_identity_sha256": hashlib.sha256(identities.tobytes()).hexdigest(),
        "positive_coordinates_used": False,
        "positive_identities_used": False,
    }
    return FoldDelaySamples(screen=screen, confirmation=confirmation, manifest=manifest)


def _total_pairwise_cs(
    values: np.ndarray,
    *,
    bandwidth: float,
    backend: str,
    counter: dict[str, int],
) -> float:
    total = 0.0
    for left in range(values.shape[0] - 1):
        for right in range(left + 1, values.shape[0]):
            result = cs_parzen_objective(
                np.column_stack((values[left], values[right])),
                float(bandwidth),
                block_rows=KERNEL_BLOCK_ROWS,
                kernel_dtype=KERNEL_DTYPE,
                accumulator_dtype=np.float64,
                backend=backend,
            )
            counter["calls"] += 1
            counter["numerical_clamps"] += int(result.numerical_clamps)
            total += float(result.objective)
    return float(total)


def fit_full_rank_delay_whitening(
    observations: np.ndarray,
    confirmation_observations: np.ndarray,
    *,
    eigenvalue_floor_ratio: float = EIGENVALUE_FLOOR_RATIO,
) -> FoldDelayWhitening:
    """Fit one six-coordinate PCA whitening before any rotation search."""

    values = np.asarray(observations, dtype=np.float64)
    confirmation = np.asarray(confirmation_observations, dtype=np.float64)
    dimension = len(V5_LAGS)
    if (
        values.shape != (dimension, SCREEN_SAMPLES)
        or confirmation.shape != (dimension, CONFIRMATION_SAMPLES)
        or not np.isfinite(values).all()
        or not np.isfinite(confirmation).all()
    ):
        raise ValueError("protected delay samples have the wrong shape or are non-finite")
    started = time.perf_counter()
    whitened, model = pca_whiten(
        values,
        rank=dimension,
        eigenvalue_floor_ratio=float(eigenvalue_floor_ratio),
    )
    confirmation_whitened = model.whitening @ (
        confirmation - model.mean[:, None]
    )
    elapsed = time.perf_counter() - started
    result = FoldDelayWhitening(
        screen=whitened,
        confirmation=confirmation_whitened,
        model=model,
        fit_seconds=float(elapsed),
        screen_observations_sha256=_array_sha256(values),
        confirmation_observations_sha256=_array_sha256(confirmation),
    )
    if result.model.explained_fraction < 1.0 - 1e-12:
        raise RuntimeError("full-rank delay whitening discarded variance")
    return result


def fit_cs_parzen_delay_embedding(
    observations: np.ndarray,
    confirmation_observations: np.ndarray,
    *,
    bandwidth: float,
    whitening_fit: FoldDelayWhitening | None = None,
    backend: str = "cuda",
    angle_step_degrees: float = ANGLE_STEP_DEGREES,
    max_sweeps: int = MAX_SWEEPS,
    improvement_tolerance: float = IMPROVEMENT_TOLERANCE,
    eigenvalue_floor_ratio: float = EIGENVALUE_FLOOR_RATIO,
) -> TemporalMSICAFit:
    """Fit the protected full-rank CS-Parzen delay rotation.

    The post-fit component labels are analytic: persistence is the effective
    demixing row closest to the all-lag common axis; innovation is the remaining
    row closest to the t-minus-t-1 axis; the other four rows form the residual
    energy subspace.  No sparse-positive fields enter this rule.
    """

    values = np.asarray(observations, dtype=np.float64)
    confirmation = np.asarray(confirmation_observations, dtype=np.float64)
    dimension = len(V5_LAGS)
    if (
        values.shape != (dimension, SCREEN_SAMPLES)
        or confirmation.shape != (dimension, CONFIRMATION_SAMPLES)
        or not np.isfinite(values).all()
        or not np.isfinite(confirmation).all()
    ):
        raise ValueError("protected delay samples have the wrong shape or are non-finite")
    if backend not in {"cpu", "cuda"}:
        raise ValueError("backend must be cpu or cuda")
    if float(bandwidth) not in CS_PARZEN_BANDWIDTHS:
        raise ValueError("bandwidth is outside the frozen protected grid")
    if not 0 < float(angle_step_degrees) <= 45 or int(max_sweeps) < 1:
        raise ValueError("invalid Jacobi rotation search")

    owns_whitening = whitening_fit is None
    if whitening_fit is None:
        whitening_fit = fit_full_rank_delay_whitening(
            values,
            confirmation,
            eigenvalue_floor_ratio=float(eigenvalue_floor_ratio),
        )
    elif (
        whitening_fit.screen_observations_sha256 != _array_sha256(values)
        or whitening_fit.confirmation_observations_sha256
        != _array_sha256(confirmation)
    ):
        raise ValueError("shared whitening was not fitted from these exact samples")
    whitened = whitening_fit.screen
    confirmation_whitened = whitening_fit.confirmation
    whitening_model = whitening_fit.model
    fit_started = time.perf_counter()
    rotation_started = time.perf_counter()
    rotation = np.eye(dimension, dtype=np.float64)
    current = whitened.copy()
    angles = np.deg2rad(
        np.arange(
            -45.0,
            45.0 + float(angle_step_degrees) / 2.0,
            float(angle_step_degrees),
        )
    )
    zero_index = int(np.argmin(np.abs(angles)))
    counter = {"calls": 0, "numerical_clamps": 0}
    history = [
        _total_pairwise_cs(
            current, bandwidth=bandwidth, backend=backend, counter=counter
        )
    ]
    accepted = 0
    converged = False
    completed_sweeps = 0
    for sweep in range(1, int(max_sweeps) + 1):
        completed_sweeps = sweep
        start_objective = history[-1]
        for left in range(dimension - 1):
            for right in range(left + 1, dimension):
                pair = current[[left, right]]
                candidates = []
                for angle in angles:
                    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
                    candidate = np.column_stack(
                        (
                            cosine * pair[0] + sine * pair[1],
                            -sine * pair[0] + cosine * pair[1],
                        )
                    )
                    result = cs_parzen_objective(
                        candidate,
                        float(bandwidth),
                        block_rows=KERNEL_BLOCK_ROWS,
                        kernel_dtype=KERNEL_DTYPE,
                        accumulator_dtype=np.float64,
                        backend=backend,
                    )
                    counter["calls"] += 1
                    counter["numerical_clamps"] += int(result.numerical_clamps)
                    candidates.append(float(result.objective))
                best = int(np.argmin(candidates))
                if candidates[zero_index] - candidates[best] <= float(
                    improvement_tolerance
                ):
                    continue
                cosine, sine = float(np.cos(angles[best])), float(np.sin(angles[best]))
                jacobi = np.eye(dimension, dtype=np.float64)
                jacobi[left, left] = cosine
                jacobi[left, right] = sine
                jacobi[right, left] = -sine
                jacobi[right, right] = cosine
                current = jacobi @ current
                rotation = jacobi @ rotation
                accepted += 1
        history.append(
            _total_pairwise_cs(
                current, bandwidth=bandwidth, backend=backend, counter=counter
            )
        )
        if start_objective - history[-1] <= float(improvement_tolerance):
            converged = True
            break

    confirmation_baseline = _total_pairwise_cs(
        confirmation_whitened,
        bandwidth=bandwidth,
        backend=backend,
        counter=counter,
    )
    confirmation_rotated = rotation @ confirmation_whitened
    confirmation_objective = _total_pairwise_cs(
        confirmation_rotated,
        bandwidth=bandwidth,
        backend=backend,
        counter=counter,
    )
    rotation_seconds = time.perf_counter() - rotation_started

    effective = rotation @ whitening_model.whitening
    norms = np.maximum(np.linalg.norm(effective, axis=1, keepdims=True), 1e-12)
    normalized = effective / norms
    common_axis = np.ones(dimension, dtype=np.float64)
    common_axis /= np.linalg.norm(common_axis)
    difference_axis = np.zeros(dimension, dtype=np.float64)
    difference_axis[0], difference_axis[1] = 1.0, -1.0
    difference_axis /= np.linalg.norm(difference_axis)
    persistence = int(np.argmax(np.abs(normalized @ common_axis)))
    innovation_candidates = [index for index in range(dimension) if index != persistence]
    innovation = max(
        innovation_candidates,
        key=lambda index: abs(float(normalized[index] @ difference_axis)),
    )
    signs = np.ones(dimension, dtype=np.int32)
    signs[persistence] = (
        1 if float(normalized[persistence] @ common_axis) >= 0.0 else -1
    )
    signs[innovation] = (
        1 if float(normalized[innovation] @ difference_axis) >= 0.0 else -1
    )
    signed_rotation = np.diag(signs) @ rotation
    residual = tuple(
        index for index in range(dimension) if index not in (persistence, innovation)
    )
    screen_energy = np.sum(np.square(whitened), axis=0)
    rotated_energy = np.sum(np.square(rotation @ whitened), axis=0)
    energy_error = float(
        np.max(np.abs(screen_energy - rotated_energy))
        / max(float(np.max(np.abs(screen_energy))), np.finfo(float).eps)
    )
    return TemporalMSICAFit(
        formulation="delay_embedding",
        objective_family="cs_parzen",
        objective_parameter={"bandwidth": float(bandwidth)},
        lags=V5_LAGS,
        lag_weights=(),
        center=whitening_model.mean,
        whitening=whitening_model.whitening,
        rotation=signed_rotation,
        demixing=signed_rotation @ whitening_model.whitening,
        objective=float(confirmation_objective),
        baseline_objective=float(confirmation_baseline),
        persistence_index=persistence,
        innovation_index=innovation,
        residual_indices=residual,
        component_signs=tuple(int(value) for value in signs),
        converged=converged,
        diagnostics={
            "objective_history": history,
            "accepted_updates": accepted,
            "sweeps": completed_sweeps,
            "screen_objective": float(history[-1]),
            "screen_baseline_objective": float(history[0]),
            "confirmation_samples": int(confirmation.shape[1]),
            "condition_number": float(whitening_model.condition_number),
            "explained_fraction": float(whitening_model.explained_fraction),
            "persistence_cosines": (normalized @ common_axis).tolist(),
            "innovation_cosines": (normalized @ difference_axis).tolist(),
            "component_rule": (
                "persistence=max_abs_common_axis; innovation=max_abs_t_minus_t1_"
                "axis_excluding_persistence; residual=all_other_four_components"
            ),
            "full_rotation_total_energy_relative_max_error": energy_error,
            "pairwise_cs_objective_calls": int(counter["calls"]),
            "numerical_clamps": int(counter["numerical_clamps"]),
            "backend": backend,
            "kernel_dtype": str(np.dtype(KERNEL_DTYPE)),
            "accumulator_dtype": str(np.dtype(np.float64)),
            "kernel_block_rows": KERNEL_BLOCK_ROWS,
            "whitening_fit_seconds": float(whitening_fit.fit_seconds),
            "whitening_reused_across_bandwidths": not owns_whitening,
            "cs_parzen_rotation_fit_seconds": float(rotation_seconds),
            "fit_seconds": float(time.perf_counter() - fit_started),
            "positive_coordinates_used": False,
            "positive_identities_used": False,
        },
    )


def fit_from_dict(payload: Mapping[str, Any]) -> TemporalMSICAFit:
    """Reconstruct one checkpointed temporal fit with strict dimensions."""

    fit = TemporalMSICAFit(
        formulation=str(payload["formulation"]),
        objective_family=str(payload["objective_family"]),
        objective_parameter=dict(payload["objective_parameter"]),
        lags=tuple(int(value) for value in payload["lags"]),
        lag_weights=tuple(float(value) for value in payload["lag_weights"]),
        center=np.asarray(payload["center"], dtype=np.float64),
        whitening=np.asarray(payload["whitening"], dtype=np.float64),
        rotation=np.asarray(payload["rotation"], dtype=np.float64),
        demixing=np.asarray(payload["demixing"], dtype=np.float64),
        objective=float(payload["objective"]),
        baseline_objective=float(payload["baseline_objective"]),
        persistence_index=int(payload["persistence_index"]),
        innovation_index=int(payload["innovation_index"]),
        residual_indices=tuple(int(value) for value in payload["residual_indices"]),
        component_signs=tuple(int(value) for value in payload["component_signs"]),
        converged=bool(payload["converged"]),
        diagnostics=dict(payload["diagnostics"]),
    )
    dimension = len(V5_LAGS)
    matrices = (fit.center, fit.whitening, fit.rotation, fit.demixing)
    if (
        fit.formulation != "delay_embedding"
        or fit.objective_family != "cs_parzen"
        or fit.lags != V5_LAGS
        or float(fit.objective_parameter.get("bandwidth", -1.0))
        not in CS_PARZEN_BANDWIDTHS
        or fit.center.shape != (dimension,)
        or fit.whitening.shape != (dimension, dimension)
        or fit.rotation.shape != (dimension, dimension)
        or fit.demixing.shape != (dimension, dimension)
        or not all(np.isfinite(value).all() for value in matrices)
        or not math.isfinite(fit.objective)
        or not math.isfinite(fit.baseline_objective)
        or fit.persistence_index not in range(dimension)
        or fit.innovation_index not in range(dimension)
        or fit.persistence_index == fit.innovation_index
        or len(fit.residual_indices) != 4
        or set(fit.residual_indices)
        != set(range(dimension)) - {fit.persistence_index, fit.innovation_index}
        or len(fit.component_signs) != dimension
        or set(fit.component_signs) - {-1, 1}
        or np.linalg.matrix_rank(fit.whitening) != dimension
        or not np.allclose(fit.rotation @ fit.rotation.T, np.eye(dimension), atol=1e-8)
        or not np.allclose(fit.demixing, fit.rotation @ fit.whitening, atol=1e-10)
        or fit.diagnostics.get("positive_coordinates_used") is not False
        or fit.diagnostics.get("positive_identities_used") is not False
    ):
        raise ValueError("checkpoint does not satisfy the protected six-lag fit contract")
    return fit


def select_fold_models(
    rows: Sequence[Mapping[str, Any]],
    *,
    fold: int,
    bandwidths: Sequence[float] = CS_PARZEN_BANDWIDTHS,
    seeds: Sequence[int] = SAMPLE_SEEDS,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Select PCA and ICA independently from the exact label-free fold grid."""

    selected = [row for row in rows if int(row["training_fold"]) == int(fold)]
    expected = {
        (float(bandwidth), int(seed))
        for bandwidth in bandwidths
        for seed in seeds
    }
    observed = {
        (float(row["bandwidth"]), int(row["sample_seed"])) for row in selected
    }
    if len(selected) != len(expected) or observed != expected:
        raise ValueError("fold selection requires the exact frozen bandwidth-by-seed grid")
    for row in selected:
        if row.get("positive_coordinates_used") is not False:
            raise ValueError("fit selection row used sparse-positive coordinates")
        if row.get("positive_identities_used") is not False:
            raise ValueError("fit selection row used sparse-positive identities")
    unique_pca = []
    for seed in seeds:
        seed_rows = sorted(
            (row for row in selected if int(row["sample_seed"]) == int(seed)),
            key=lambda row: float(row["bandwidth"]),
        )
        reference = seed_rows[0]
        for row in seed_rows[1:]:
            for field in (
                "sample_identity_sha256",
                "pca_mean_positive_tail_contrast",
                "pca_minimum_quiet_swap_positive_tail_contrast",
                "whitening_sha256",
            ):
                if row[field] != reference[field]:
                    raise ValueError(
                        f"bandwidth changed bandwidth-independent PCA field {field}"
                    )
        unique_pca.append(reference)
    pca = sorted(
        unique_pca,
        key=lambda row: (
            -float(row["pca_mean_positive_tail_contrast"]),
            -float(row["pca_minimum_quiet_swap_positive_tail_contrast"]),
            int(row["sample_seed"]),
        ),
    )[0]
    eligible_ica = [
        row
        for row in selected
        if bool(row["converged"]) and int(row["numerical_clamps"]) == 0
    ]
    if not eligible_ica:
        raise ValueError(
            "outer fold has no converged, unclamped CS-Parzen rotation candidate"
        )
    ica = sorted(
        eligible_ica,
        key=lambda row: (
            -float(row["ica_mean_positive_tail_contrast"]),
            -float(row["ica_minimum_quiet_swap_positive_tail_contrast"]),
            float(row["bandwidth"]),
            int(row["sample_seed"]),
        ),
    )[0]
    return pca, ica


def _stream_raw_history_to_device(
    movie: Any,
    *,
    review_start_ui: int,
    review_stop_ui: int,
    device: Any,
    chunk_frames: int,
    heartbeat: Callable[..., None],
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    history = max(V5_LAGS)
    source_start_zero = int(review_start_ui) - 1 - history
    source_stop_zero = int(review_stop_ui)
    if source_start_zero < 0 or source_stop_zero > len(movie):
        raise ValueError("review interval lacks complete six-lag history")
    frame_count = source_stop_zero - source_start_zero
    expected = int(review_stop_ui) - int(review_start_ui) + 1 + history
    if frame_count != expected:
        raise AssertionError("raw six-lag source alignment changed")
    output = torch.empty(
        (frame_count, int(movie.shape[1]), int(movie.shape[2])),
        dtype=torch.float32,
        device=device,
    )
    h2d_seconds = 0.0
    for local_start in range(0, frame_count, int(chunk_frames)):
        count = min(int(chunk_frames), frame_count - local_start)
        host = np.asarray(
            movie[
                source_start_zero + local_start : source_start_zero + local_start + count
            ],
            dtype=np.float32,
        )
        started = time.perf_counter()
        output[local_start : local_start + count].copy_(torch.from_numpy(host))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        h2d_seconds += time.perf_counter() - started
        heartbeat(
            "raw_h2d",
            copied_frames=local_start + count,
            total_frames=frame_count,
        )
    source_frame_ui = torch.arange(
        int(review_start_ui) - history,
        int(review_stop_ui) + 1,
        dtype=torch.int64,
        device=device,
    )
    return output, source_frame_ui, {
        "input_domain": "acquisition_raw_uint16_to_float32",
        "history_frames": history,
        "input_frame_count": frame_count,
        "output_frame_count": expected - history,
        "source_ui_first": int(review_start_ui) - history,
        "source_ui_last": int(review_stop_ui),
        "chunk_frames": int(chunk_frames),
        "h2d_seconds": float(h2d_seconds),
        "h2d_ms_per_loaded_frame": float(1000.0 * h2d_seconds / frame_count),
    }


def _timed_representation(
    operation: Callable[[], Any], *, device: Any, arm: str
) -> tuple[Any, dict[str, Any]]:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    frame_count = int(result.values.shape[0])
    return result, {
        "representation": arm,
        "representation_inference_seconds": float(seconds),
        "representation_inference_ms_per_frame": float(
            1000.0 * seconds / frame_count
        ),
        "output_frames": frame_count,
        "peak_allocated_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "input_transfer_included": False,
        "fit_time_included": False,
    }


def _representation_for_arm(
    raw: Any,
    source_frame_ui: Any,
    *,
    arm: str,
    fit: TemporalMSICAFit | None,
    chunk_frames: int = 64,
) -> Any:
    """Project one exact six-lag arm with bounded device working memory."""

    import torch

    if (
        not torch.is_tensor(raw)
        or raw.dtype != torch.float32
        or raw.ndim != 3
        or raw.shape[0] <= max(V5_LAGS)
    ):
        raise ValueError("raw must be a float32 TYX tensor with six-lag history")
    if (
        not torch.is_tensor(source_frame_ui)
        or source_frame_ui.dtype != torch.int64
        or source_frame_ui.ndim != 1
        or source_frame_ui.shape[0] != raw.shape[0]
        or source_frame_ui.device != raw.device
        or not bool(torch.all(source_frame_ui[1:] - source_frame_ui[:-1] == 1))
    ):
        raise ValueError("source_frame_ui must be contiguous int64 on the raw device")
    chunk = int(chunk_frames)
    if chunk < 1:
        raise ValueError("chunk_frames must be positive")
    history = max(V5_LAGS)
    length = int(raw.shape[0]) - history
    output = torch.empty(
        (length, int(raw.shape[1]), int(raw.shape[2])),
        dtype=torch.float32,
        device=raw.device,
    )
    output_indices = source_frame_ui[history:].clone()

    if arm == "difference_multilag_energy_normalized":
        if fit is not None:
            raise ValueError("fixed multi-lag difference must not receive a learned fit")
        for start in range(0, length, chunk):
            count = min(chunk, length - start)
            absolute = history + start
            current = raw[absolute : absolute + count]
            energy_squared = torch.zeros_like(current)
            for lag in V5_LAGS[1:]:
                previous = raw[absolute - lag : absolute - lag + count]
                normalized = (current - previous) / torch.sqrt(
                    current.square() + previous.square() + ENERGY_EPSILON
                )
                energy_squared.add_(normalized.square())
            output[start : start + count] = torch.sqrt(energy_squared)
        diagnostics = {
            "representation": "multilag_energy_normalized_difference",
            "formula": (
                "sqrt(sum_lag(((I[t]-I[t-lag]) / "
                "sqrt(I[t]^2+I[t-lag]^2+epsilon))^2))"
            ),
            "input_domain": "acquisition_raw",
            "lags": list(V5_LAGS[1:]),
            "epsilon": ENERGY_EPSILON,
            "lag_reduction": "sqrt(sum(normalized_difference**2))",
            "history_frames": history,
            "chunk_frames": chunk,
            "bounded_projection": True,
            "labels_used": False,
        }
        return DeviceRepresentationMap(output, output_indices, diagnostics)
    if fit is None:
        raise ValueError(f"learned arm {arm} requires a fold-local fit")
    fit_from_dict(fit.to_dict())
    center = torch.as_tensor(
        fit.center, dtype=torch.float32, device=raw.device
    ).reshape(len(V5_LAGS), 1)
    if arm == "pca_whitened_delay_total_energy":
        transform = torch.as_tensor(
            fit.whitening, dtype=torch.float32, device=raw.device
        )
        diagnostic = {
            "representation": "frozen_v5_pca_whitened_delay_total_energy",
            "operator": "fold_fitted_full_rank_whitening_Q_before_rotation",
            "coordinate_count": len(V5_LAGS),
            "full_rank_whitening": True,
            "ica_rotation_applied": False,
            "reduction": "sqrt(sum(all_whitened_coordinates**2))",
        }
    elif arm == "cs_parzen_delay_residual":
        indices = torch.as_tensor(
            fit.residual_indices, dtype=torch.int64, device=raw.device
        )
        demixing = torch.as_tensor(
            fit.demixing, dtype=torch.float32, device=raw.device
        )
        transform = torch.index_select(demixing, 0, indices)
        diagnostic = {
            "representation": "fold_fitted_cs_parzen_delay_residual_group",
            "residual_indices": list(fit.residual_indices),
            "residual_reduction": "sqrt(sum(component**2))",
            "component_rule": fit.diagnostics["component_rule"],
        }
    else:
        raise ValueError(f"unknown matched-six-lag arm {arm!r}")
    for start in range(0, length, chunk):
        count = min(chunk, length - start)
        absolute = history + start
        stack = torch.stack(
            [
                raw[absolute - lag : absolute - lag + count]
                for lag in V5_LAGS
            ],
            dim=0,
        )
        flat = stack.reshape(len(V5_LAGS), -1) - center
        transformed = transform @ flat
        energy = torch.sqrt(torch.sum(transformed.square(), dim=0)).reshape(
            count, int(raw.shape[1]), int(raw.shape[2])
        )
        output[start : start + count] = energy
    diagnostics = {
        **diagnostic,
        "fit_reused_without_refitting": True,
        "fit_scope": "outer_training_burst_windows_excluding_heldout_guard_history",
        "lags": list(V5_LAGS),
        "input_domain": "acquisition_raw",
        "history_frames": history,
        "chunk_frames": chunk,
        "bounded_projection": True,
        "labels_used": False,
    }
    return DeviceRepresentationMap(output, output_indices, diagnostics)


def _assert_acquisition_raw_output(
    representation: Any,
    *,
    raw: Any,
    source_frame_ui: Any,
    review_frame_ui: Any,
) -> None:
    """Fail unless an arm declares and preserves the shared raw-domain timeline."""

    import torch

    if raw.dtype != torch.float32 or raw.ndim != 3:
        raise AssertionError("shared acquisition-raw tensor must be float32 TYX")
    if representation.diagnostics.get("input_domain") != "acquisition_raw":
        raise AssertionError("six-lag arm did not attest acquisition_raw input")
    expected = source_frame_ui[max(V5_LAGS) :]
    if not torch.equal(expected, review_frame_ui):
        raise AssertionError("loaded acquisition-raw history does not align to review UI")
    if not torch.equal(representation.source_frame_indices, expected):
        raise AssertionError("six-lag representation escaped the raw-domain timeline")
    if representation.values.shape[0] != review_frame_ui.numel():
        raise AssertionError("six-lag representation output frame count changed")


def _fit_key(fold: int, seed: int, bandwidth: float) -> str:
    token = str(float(bandwidth)).replace(".", "p")
    return f"fold_{fold}__seed_{seed}__bandwidth_{token}"


def _arm_model_sha256(
    arm: str, fit: TemporalMSICAFit | None
) -> str:
    """Hash exactly the fixed or learned operator used by one candidate cell."""

    if arm == "difference_multilag_energy_normalized":
        if fit is not None:
            raise ValueError("fixed difference arm cannot have a fitted model")
        return _canonical_sha256(
            {
                "arm": arm,
                "lags": list(V5_LAGS),
                "epsilon": ENERGY_EPSILON,
            }
        )
    if fit is None:
        raise ValueError(f"learned arm {arm} requires a fit identity")
    if arm == "pca_whitened_delay_total_energy":
        payload = {
            "arm": arm,
            "lags": list(fit.lags),
            "center": fit.center.tolist(),
            "whitening": fit.whitening.tolist(),
        }
    elif arm == "cs_parzen_delay_residual":
        payload = {
            "arm": arm,
            "lags": list(fit.lags),
            "center": fit.center.tolist(),
            "demixing": fit.demixing.tolist(),
            "residual_indices": list(fit.residual_indices),
            "component_rule": fit.diagnostics["component_rule"],
        }
    else:
        raise ValueError(f"unknown matched-six-lag arm {arm!r}")
    return _canonical_sha256(payload)


def _read_selection_checkpoint(
    path: Path, expected_contract: Mapping[str, Any]
) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract") != dict(expected_contract):
        raise RuntimeError(f"fit-selection checkpoint contract changed: {path.name}")
    metric = payload.get("metric")
    if not isinstance(metric, dict):
        raise RuntimeError(f"fit-selection checkpoint lacks metrics: {path.name}")
    finite_fields = (
        "mean_positive_tail_contrast",
        "minimum_quiet_swap_positive_tail_contrast",
        "gamma_runtime_ms_per_frame",
    )
    if (
        any(
            not math.isfinite(float(metric.get(field, float("nan"))))
            for field in finite_fields
        )
        or len(metric.get("quiet_swap_positive_tail_contrasts", ())) != 2
        or not all(
            math.isfinite(float(value))
            for value in metric.get("quiet_swap_positive_tail_contrasts", ())
        )
        or metric.get("selection_uses_training_burst_windows") is not True
        or metric.get("positive_coordinates_used") is not False
        or metric.get("positive_identities_used") is not False
    ):
        raise RuntimeError(f"fit-selection checkpoint is invalid: {path.name}")
    return metric


def _fit_packet_row(
    *,
    fold: FoldContract,
    seed: int,
    bandwidth: float,
    sample_manifest: Mapping[str, Any],
    fit: TemporalMSICAFit,
    pca_metric: Mapping[str, Any],
    ica_metric: Mapping[str, Any],
) -> dict[str, Any]:
    whitening_sha = _canonical_sha256(
        {
            "center": fit.center.tolist(),
            "whitening": fit.whitening.tolist(),
        }
    )
    return {
        "training_fold": fold.training_fold,
        "heldout_burst": int(fold.heldout_burst),
        "bandwidth": float(bandwidth),
        "sample_seed": int(seed),
        "sample_identity_sha256": sample_manifest["sample_identity_sha256"],
        "whitening_sha256": whitening_sha,
        "whitening_fit_seconds": float(
            fit.diagnostics["whitening_fit_seconds"]
        ),
        "cs_parzen_rotation_fit_seconds": float(
            fit.diagnostics["cs_parzen_rotation_fit_seconds"]
        ),
        "fit_seconds": float(
            fit.diagnostics["whitening_fit_seconds"]
            + fit.diagnostics["fit_seconds"]
        ),
        "fit_seconds_excluding_shared_whitening": float(
            fit.diagnostics["fit_seconds"]
        ),
        "whitening_reused_across_bandwidths": bool(
            fit.diagnostics["whitening_reused_across_bandwidths"]
        ),
        "confirmation_objective": float(fit.objective),
        "confirmation_baseline_objective": float(fit.baseline_objective),
        "confirmation_gain_fraction": float(
            (fit.baseline_objective - fit.objective)
            / max(abs(fit.baseline_objective), np.finfo(float).eps)
        ),
        "converged": bool(fit.converged),
        "sweeps": int(fit.diagnostics["sweeps"]),
        "accepted_updates": int(fit.diagnostics["accepted_updates"]),
        "pairwise_cs_objective_calls": int(
            fit.diagnostics["pairwise_cs_objective_calls"]
        ),
        "numerical_clamps": int(fit.diagnostics["numerical_clamps"]),
        "whitening_condition_number": float(fit.diagnostics["condition_number"]),
        "full_rotation_total_energy_relative_max_error": float(
            fit.diagnostics["full_rotation_total_energy_relative_max_error"]
        ),
        "persistence_index": int(fit.persistence_index),
        "innovation_index": int(fit.innovation_index),
        "residual_indices": json.dumps(list(fit.residual_indices)),
        "pca_mean_positive_tail_contrast": float(
            pca_metric["mean_positive_tail_contrast"]
        ),
        "pca_minimum_quiet_swap_positive_tail_contrast": float(
            pca_metric["minimum_quiet_swap_positive_tail_contrast"]
        ),
        "ica_mean_positive_tail_contrast": float(
            ica_metric["mean_positive_tail_contrast"]
        ),
        "ica_minimum_quiet_swap_positive_tail_contrast": float(
            ica_metric["minimum_quiet_swap_positive_tail_contrast"]
        ),
        "fit_uses_only_outer_training_burst_windows": True,
        "selection_event_windows_are_outer_training_bursts_only": True,
        "selection_quiet_reference_is_predeclared_crossfit_quiet_halves": True,
        "positive_coordinates_used": False,
        "positive_identities_used": False,
        "archived_fit_loaded": False,
    }


def run_multilag_protected_experiment(
    config: GammaLSDifferenceConfig,
    *,
    preflight_dir: str | Path,
    context_selection_dir: str | Path,
    output_dir: str | Path,
    context_role: str = DEFAULT_CONTEXT_ROLE,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run, seal, and evaluate the protected matched-six-lag comparison."""

    if not isinstance(config, GammaLSDifferenceConfig):
        raise TypeError("config must be a validated GammaLSDifferenceConfig")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"matched-six-lag output exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"matched-six-lag output parent does not exist: {destination.parent}"
        )

    # Complete all read-only gates before creating or updating resumable output.
    preflight = verify_matching_preflight(config, preflight_dir, require_gpu_ready=True)
    folds = build_fold_contracts(config)
    context_plan, context_provenance = load_fold_context_plan(
        context_selection_dir, folds
    )
    selected_contexts = select_fold_contexts(context_plan, role=context_role)
    historical = audit_historical_v5_surface(
        config.source_paths["multilag_v5_surface"],
        selected_config_id=str(config.payload["sources"]["multilag_config_id"]),
    )
    try:
        runtime = require_cuda_device(device)
    except CudaRuntimeUnavailable as error:
        raise MultilagProtectedUnavailable(str(error)) from error
    try:
        import cupy as cp
        import torch

        resolved_device = torch.device(str(runtime["resolved_device"]))
        if resolved_device.type != "cuda":
            raise RuntimeError("matched-six-lag protected run requires CUDA")
        cp.cuda.Device(resolved_device.index or 0).use()
    except Exception as error:
        raise MultilagProtectedUnavailable(
            f"Torch/CuPy cannot share the requested CUDA device: {error}"
        ) from error
    movie = np.load(config.source_paths["movie"], mmap_mode="r", allow_pickle=False)
    if movie.ndim != 3 or str(movie.dtype) != "uint16":
        raise ValueError("matched-six-lag source must be the frozen uint16 TYX movie")

    contract = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "run_type": "protected_outer_fold_matched_six_lag_representation",
        "portable_config_sha256": _canonical_sha256(config.portable_dict()),
        "executor_sha256": _sha256(Path(__file__).resolve()),
        "protected_helpers_sha256": _sha256(
            Path(__file__).with_name("protected.py").resolve()
        ),
        "gpu_representations_sha256": _sha256(
            Path(__file__).with_name("gpu_representations.py").resolve()
        ),
        "multilag_algorithm_sha256": _sha256(
            config.repository / "neurobench/algorithms/multilag_msica.py"
        ),
        "preflight_sha256": _sha256(
            Path(preflight_dir).expanduser().resolve() / "preflight.json"
        ),
        "movie_sha256": preflight["source"]["movie"]["sha256"],
        "context_selection_source_sha256": context_provenance["source_sha256"],
        "selected_context_role": context_role,
        "selected_contexts": {
            str(fold): lane.as_dict()
            for fold, lane in sorted(selected_contexts.items())
        },
        "design": multilag_design(),
        "historical_grid_basis": historical,
        "bootstrap": {
            "cluster_field": "canonical_roi_id",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
        },
        "sparse_positive_fields_available_to_fit_or_selection": False,
        "archived_fit_loaded_for_projection": False,
    }
    contract_sha = _canonical_sha256(contract)
    work = destination.parent / f".{destination.name}.multilag-protected-work"
    if work.exists():
        contract_path = work / "run_contract.json"
        if not contract_path.is_file():
            raise RuntimeError("resumable work directory lacks run_contract.json")
        frozen = json.loads(contract_path.read_text(encoding="utf-8"))
        if _canonical_sha256(frozen) != contract_sha:
            raise RuntimeError("existing resumable work directory has a different contract")
    else:
        work.mkdir()
        _atomic_json(work / "run_contract.json", contract)
    fit_dir = work / "fit_models"
    selection_dir = work / "fit_selection_cells"
    cell_dir = work / "candidate_cells"
    fit_dir.mkdir(exist_ok=True)
    selection_dir.mkdir(exist_ok=True)
    cell_dir.mkdir(exist_ok=True)

    def heartbeat(stage: str, **details: Any) -> None:
        _atomic_json(
            work / "heartbeat.json",
            {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                "status": "running",
                **details,
            },
        )

    try:
        # The fit objective uses CuPy while representation inference uses Torch;
        # both were pinned to the same physical device before output mutation.
        quiet_start, quiet_stop = map(
            int, config.payload["frames"]["quiet_interval_ui"]
        )
        quiet_values = np.asarray(
            movie[quiet_start - 1 : quiet_stop], dtype=np.float32
        )
        anatomy_mask, anatomy_summary = uniform_anatomy_mask(quiet_values)
        del quiet_values
        bursts = config.payload["frames"]["burst_intervals_ui"]

        # Fit packets are checkpointed before any sparse-positive table is parsed.
        packets: dict[tuple[int, int, float], TemporalMSICAFit] = {}
        sample_manifests: dict[tuple[int, int], Mapping[str, Any]] = {}
        total_fits = multilag_design()["cs_parzen_rotation_fits"]
        completed_fits = 0
        for fold in folds:
            for seed in SAMPLE_SEEDS:
                samples = sample_fold_delay_observations(
                    movie,
                    fold=fold,
                    bursts=bursts,
                    anatomy_mask=anatomy_mask,
                    seed=seed,
                )
                sample_manifests[(fold.training_fold, seed)] = samples.manifest
                whitening_fit = fit_full_rank_delay_whitening(
                    samples.screen,
                    samples.confirmation,
                )
                for bandwidth in CS_PARZEN_BANDWIDTHS:
                    key = _fit_key(fold.training_fold, seed, bandwidth)
                    checkpoint = fit_dir / f"{key}.json"
                    if checkpoint.is_file():
                        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                        if saved.get("sample_manifest") != samples.manifest:
                            raise RuntimeError("fit checkpoint sample manifest changed")
                        fit = fit_from_dict(saved["fit"])
                        if (
                            not np.allclose(
                                fit.center,
                                whitening_fit.model.mean,
                                atol=1e-12,
                                rtol=1e-12,
                            )
                            or not np.allclose(
                                fit.whitening,
                                whitening_fit.model.whitening,
                                atol=1e-12,
                                rtol=1e-12,
                            )
                        ):
                            raise RuntimeError(
                                "fit checkpoint changed its shared whitening model"
                            )
                    else:
                        heartbeat(
                            "multilag_fit",
                            completed_fits=completed_fits,
                            total_fits=total_fits,
                            training_fold=fold.training_fold,
                            sample_seed=seed,
                            bandwidth=bandwidth,
                        )
                        fit = fit_cs_parzen_delay_embedding(
                            samples.screen,
                            samples.confirmation,
                            bandwidth=bandwidth,
                            whitening_fit=whitening_fit,
                            backend="cuda",
                        )
                        _atomic_json(
                            checkpoint,
                            {
                                "training_fold": fold.training_fold,
                                "heldout_burst": fold.heldout_burst,
                                "sample_seed": seed,
                                "bandwidth": bandwidth,
                                "sample_manifest": samples.manifest,
                                "fit": fit.to_dict(),
                            },
                        )
                    packets[(fold.training_fold, seed, bandwidth)] = fit
                    completed_fits += 1
                    heartbeat(
                        "multilag_fit",
                        completed_fits=completed_fits,
                        total_fits=total_fits,
                    )
                    print(
                        f"MULTILAG FIT {completed_fits}/{total_fits} "
                        f"fold={fold.training_fold} seed={seed} bandwidth={bandwidth}",
                        flush=True,
                    )
        if completed_fits != total_fits or len(packets) != total_fits:
            raise AssertionError("matched-six-lag fit grid is incomplete")

        # The fit objective uses small CuPy allocations.  Release its cache
        # before allocating the full review movie in Torch so the 8-GiB
        # experiment cap measures useful tensors rather than allocator residue.
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()

        start_ui, stop_ui = map(
            int, config.payload["frames"]["review_interval_ui"]
        )
        chunk_frames = max(
            int(value) for value in config.payload["efficiency"]["frame_chunks"]
        )
        raw, source_frame_ui, raw_load_timing = _stream_raw_history_to_device(
            movie,
            review_start_ui=start_ui,
            review_stop_ui=stop_ui,
            device=resolved_device,
            chunk_frames=chunk_frames,
            heartbeat=heartbeat,
        )
        frame_ui = np.arange(start_ui, stop_ui + 1, dtype=np.int64)
        frame_ui_device = torch.as_tensor(frame_ui, device=resolved_device)
        raw_source_contract_sha256 = _canonical_sha256(
            {
                "movie_sha256": preflight["source"]["movie"]["sha256"],
                "input_domain": "acquisition_raw",
                "preprocessing": "none",
                "loaded_source_frame_ui": [
                    start_ui - max(V5_LAGS),
                    stop_ui,
                ],
                "review_output_frame_ui": [start_ui, stop_ui],
                "lags": list(V5_LAGS),
            }
        )
        scale_percentile = float(
            config.payload["gamma_ls_grid"]["finalist_scale_floor_percentiles"][0]
        )
        tail_quantile = float(config.payload["screen"]["positive_tail_quantile"])

        fit_rows: list[dict[str, Any]] = []
        pca_metrics: dict[tuple[int, int], Mapping[str, Any]] = {}
        completed_selection = 0
        total_selection = multilag_design()["gamma_selection_maps"]
        for fold in folds:
            lane = selected_contexts[fold.training_fold]
            selection_bursts = guard_safe_selection_bursts(fold, bursts)
            for seed in SAMPLE_SEEDS:
                reference_fit = packets[
                    (fold.training_fold, seed, CS_PARZEN_BANDWIDTHS[0])
                ]
                selection_intervals = {
                    burst_id: selection_bursts[burst_id]
                    for burst_id in fold.training_bursts
                }
                pca_contract = {
                    "training_fold": fold.training_fold,
                    "heldout_burst": fold.heldout_burst,
                    "context_role": lane.role,
                    "context_id": lane.context_id,
                    "representation": "pca_whitened_delay_total_energy",
                    "sample_seed": seed,
                    "arm_model_sha256": _arm_model_sha256(
                        "pca_whitened_delay_total_energy", reference_fit
                    ),
                    "raw_source_contract_sha256": raw_source_contract_sha256,
                    "guard_safe_training_burst_intervals_ui": selection_intervals,
                    "input_domain": "acquisition_raw",
                }
                pca_path = selection_dir / (
                    f"fold_{fold.training_fold}__seed_{seed}__pca.json"
                )
                if pca_path.is_file():
                    pca_metric = _read_selection_checkpoint(
                        pca_path, pca_contract
                    )
                else:
                    pca_map, _ = _timed_representation(
                        lambda fit=reference_fit: _representation_for_arm(
                            raw,
                            source_frame_ui,
                            arm="pca_whitened_delay_total_energy",
                            fit=fit,
                            chunk_frames=chunk_frames,
                        ),
                        device=resolved_device,
                        arm="pca_whitened_delay_total_energy",
                    )
                    _assert_acquisition_raw_output(
                        pca_map,
                        raw=raw,
                        source_frame_ui=source_frame_ui,
                        review_frame_ui=frame_ui_device,
                    )
                    pca_metric = _fit_selection_metrics(
                        pca_map.values,
                        representation_name="pca_whitened_delay_total_energy",
                        lane=lane,
                        fold=fold,
                        frame_ui_device=frame_ui_device,
                        bursts=selection_bursts,
                        scale_floor_percentile=scale_percentile,
                        tail_quantile=tail_quantile,
                        chunk_frames=chunk_frames,
                    )
                    _atomic_json(
                        pca_path,
                        {"contract": pca_contract, "metric": pca_metric},
                    )
                    del pca_map
                    torch.cuda.empty_cache()
                pca_metrics[(fold.training_fold, seed)] = pca_metric
                completed_selection += 1
                heartbeat(
                    "fit_selection_maps",
                    completed_maps=completed_selection,
                    total_maps=total_selection,
                )
                for bandwidth in CS_PARZEN_BANDWIDTHS:
                    fit = packets[(fold.training_fold, seed, bandwidth)]
                    bandwidth_token = str(float(bandwidth)).replace(".", "p")
                    ica_contract = {
                        "training_fold": fold.training_fold,
                        "heldout_burst": fold.heldout_burst,
                        "context_role": lane.role,
                        "context_id": lane.context_id,
                        "representation": "cs_parzen_delay_residual",
                        "sample_seed": seed,
                        "bandwidth": bandwidth,
                        "arm_model_sha256": _arm_model_sha256(
                            "cs_parzen_delay_residual", fit
                        ),
                        "raw_source_contract_sha256": raw_source_contract_sha256,
                        "guard_safe_training_burst_intervals_ui": (
                            selection_intervals
                        ),
                        "input_domain": "acquisition_raw",
                    }
                    ica_path = selection_dir / (
                        f"fold_{fold.training_fold}__seed_{seed}__"
                        f"bandwidth_{bandwidth_token}__ica.json"
                    )
                    if ica_path.is_file():
                        ica_metric = _read_selection_checkpoint(
                            ica_path, ica_contract
                        )
                    else:
                        ica_map, _ = _timed_representation(
                            lambda fit=fit: _representation_for_arm(
                                raw,
                                source_frame_ui,
                                arm="cs_parzen_delay_residual",
                                fit=fit,
                                chunk_frames=chunk_frames,
                            ),
                            device=resolved_device,
                            arm="cs_parzen_delay_residual",
                        )
                        _assert_acquisition_raw_output(
                            ica_map,
                            raw=raw,
                            source_frame_ui=source_frame_ui,
                            review_frame_ui=frame_ui_device,
                        )
                        ica_metric = _fit_selection_metrics(
                            ica_map.values,
                            representation_name="cs_parzen_delay_residual",
                            lane=lane,
                            fold=fold,
                            frame_ui_device=frame_ui_device,
                            bursts=selection_bursts,
                            scale_floor_percentile=scale_percentile,
                            tail_quantile=tail_quantile,
                            chunk_frames=chunk_frames,
                        )
                        _atomic_json(
                            ica_path,
                            {"contract": ica_contract, "metric": ica_metric},
                        )
                        del ica_map
                        torch.cuda.empty_cache()
                    completed_selection += 1
                    row = _fit_packet_row(
                        fold=fold,
                        seed=seed,
                        bandwidth=bandwidth,
                        sample_manifest=sample_manifests[(fold.training_fold, seed)],
                        fit=fit,
                        pca_metric=pca_metrics[(fold.training_fold, seed)],
                        ica_metric=ica_metric,
                    )
                    row["context_role"] = lane.role
                    row["context_id"] = lane.context_id
                    row["guard_safe_training_burst_intervals_ui"] = json.dumps(
                        {
                            burst_id: selection_bursts[burst_id]
                            for burst_id in fold.training_bursts
                        },
                        sort_keys=True,
                    )
                    fit_rows.append(row)
                    heartbeat(
                        "fit_selection_maps",
                        completed_maps=completed_selection,
                        total_maps=total_selection,
                    )
        if completed_selection != total_selection:
            raise AssertionError("fit-selection map count changed")

        selections: dict[int, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
        selected_rows: list[dict[str, Any]] = []
        for fold in folds:
            pca_row, ica_row = select_fold_models(
                fit_rows, fold=fold.training_fold
            )
            selections[fold.training_fold] = (pca_row, ica_row)
            for arm, row, metric_prefix in (
                ("pca_whitened_delay_total_energy", pca_row, "pca"),
                ("cs_parzen_delay_residual", ica_row, "ica"),
            ):
                selected_rows.append(
                    {
                        "training_fold": fold.training_fold,
                        "heldout_burst": int(fold.heldout_burst),
                        "context_role": row["context_role"],
                        "context_id": row["context_id"],
                        "selected_for_representation": arm,
                        "sample_seed": int(row["sample_seed"]),
                        "bandwidth": (
                            float(row["bandwidth"])
                            if arm == "cs_parzen_delay_residual"
                            else "not_applicable_to_whitening"
                        ),
                        "selection_metric": f"{metric_prefix}_mean_positive_tail_contrast",
                        "mean_positive_tail_contrast": float(
                            row[f"{metric_prefix}_mean_positive_tail_contrast"]
                        ),
                        "minimum_quiet_swap_positive_tail_contrast": float(
                            row[
                                f"{metric_prefix}_minimum_quiet_swap_positive_tail_contrast"
                            ]
                        ),
                        "sample_identity_sha256": row["sample_identity_sha256"],
                        "positive_coordinates_used": False,
                        "positive_identities_used": False,
                    }
                )
        _atomic_tsv(work / "fit_grid.tsv", fit_rows)
        _atomic_tsv(work / "selected_fit_rows.tsv", selected_rows)

        candidate_rows: list[dict[str, Any]] = []
        calibration_rows: list[dict[str, Any]] = []
        timing_rows: list[dict[str, Any]] = []
        equivalence_rows: list[dict[str, Any]] = []
        total_cells = multilag_design()["protected_context_arm_cells"]
        completed_cells = 0
        for fold in folds:
            lane = selected_contexts[fold.training_fold]
            pca_row, ica_row = selections[fold.training_fold]
            pca_fit = packets[
                (
                    fold.training_fold,
                    int(pca_row["sample_seed"]),
                    CS_PARZEN_BANDWIDTHS[0],
                )
            ]
            ica_fit = packets[
                (
                    fold.training_fold,
                    int(ica_row["sample_seed"]),
                    float(ica_row["bandwidth"]),
                )
            ]
            fits_by_arm: dict[str, TemporalMSICAFit | None] = {
                "difference_multilag_energy_normalized": None,
                "pca_whitened_delay_total_energy": pca_fit,
                "cs_parzen_delay_residual": ica_fit,
            }
            heldout_mask = torch.as_tensor(
                _mask_interval(frame_ui, bursts[fold.heldout_burst]),
                device=resolved_device,
            )
            heldout_maps: dict[str, Any] = {}
            for arm in MULTILAG_ARMS:
                arm_model_sha256 = _arm_model_sha256(arm, fits_by_arm[arm])
                cell_path = cell_dir / f"fold_{fold.training_fold}__{arm}.json"
                if cell_path.is_file():
                    cell = json.loads(cell_path.read_text(encoding="utf-8"))
                    expected_cell = {
                        "training_fold": fold.training_fold,
                        "heldout_burst": fold.heldout_burst,
                        "context_role": lane.role,
                        "context_id": lane.context_id,
                        "representation": arm,
                        "arm_model_sha256": arm_model_sha256,
                    }
                    if any(cell.get(key) != value for key, value in expected_cell.items()):
                        raise RuntimeError(
                            f"candidate checkpoint contract changed: {cell_path.name}"
                        )
                    if cell.get("timing", {}).get(
                        "raw_source_contract_sha256"
                    ) != raw_source_contract_sha256:
                        raise RuntimeError(
                            f"candidate checkpoint raw source changed: {cell_path.name}"
                        )
                    candidate_rows.extend(cell["candidates"])
                    calibration_rows.extend(cell["calibrations"])
                    timing_rows.append(cell["timing"])
                    for row in cell.get("equivalence_rows", []):
                        equivalence_rows.append(row)
                    completed_cells += 1
                    heartbeat(
                        "candidate_freeze",
                        completed_context_arm_cells=completed_cells,
                        total_context_arm_cells=total_cells,
                    )
                    continue
                representation, representation_timing = _timed_representation(
                    lambda arm=arm: _representation_for_arm(
                        raw,
                        source_frame_ui,
                        arm=arm,
                        fit=fits_by_arm[arm],
                        chunk_frames=chunk_frames,
                    ),
                    device=resolved_device,
                    arm=arm,
                )
                _assert_acquisition_raw_output(
                    representation,
                    raw=raw,
                    source_frame_ui=source_frame_ui,
                    review_frame_ui=frame_ui_device,
                )
                heldout_maps[arm] = representation.values[heldout_mask].detach().cpu()
                score_started = time.perf_counter()
                candidates, calibrations, score_timing = _score_candidates_for_arm(
                    representation.values,
                    arm=arm,
                    lane=lane,
                    fold=fold,
                    frame_ui=frame_ui,
                    bursts=bursts,
                    scale_floor_percentile=scale_percentile,
                    chunk_frames=chunk_frames,
                    nms_distances_px=NMS_DISTANCES_PX,
                )
                score_seconds = time.perf_counter() - score_started
                peak_end_to_end = int(torch.cuda.max_memory_allocated(resolved_device))
                timing = {
                    **score_timing,
                    **representation_timing,
                    "candidate_scoring_end_to_end_seconds": float(score_seconds),
                    "candidate_scoring_end_to_end_ms_per_frame": float(
                        1000.0 * score_seconds / len(frame_ui)
                    ),
                    "fit_time_included": False,
                    "input_transfer_included": False,
                    "peak_allocated_vram_bytes_end_to_end": peak_end_to_end,
                    "source_domain_assertion_passed": True,
                    "source_domain": "acquisition_raw",
                    "raw_source_contract_sha256": raw_source_contract_sha256,
                    "arm_model_sha256": arm_model_sha256,
                    "selected_fit_seconds": (
                        0.0
                        if fits_by_arm[arm] is None
                        else float(
                            fits_by_arm[arm].diagnostics[
                                "whitening_fit_seconds"
                            ]
                            + (
                                0.0
                                if arm == "pca_whitened_delay_total_energy"
                                else fits_by_arm[arm].diagnostics["fit_seconds"]
                            )
                        )
                    ),
                }
                _atomic_json(
                    cell_path,
                    {
                        "training_fold": fold.training_fold,
                        "heldout_burst": fold.heldout_burst,
                        "context_role": lane.role,
                        "context_id": lane.context_id,
                        "representation": arm,
                        "arm_model_sha256": arm_model_sha256,
                        "candidates": candidates,
                        "calibrations": calibrations,
                        "timing": timing,
                        "equivalence_rows": [],
                    },
                )
                candidate_rows.extend(candidates)
                calibration_rows.extend(calibrations)
                timing_rows.append(timing)
                del representation
                torch.cuda.empty_cache()
                completed_cells += 1
                heartbeat(
                    "candidate_freeze",
                    completed_context_arm_cells=completed_cells,
                    total_context_arm_cells=total_cells,
                )
                print(
                    f"MULTILAG CANDIDATE {completed_cells}/{total_cells} "
                    f"fold={fold.training_fold} arm={arm}",
                    flush=True,
                )

            # If cells were resumed, regenerate only the three bounded held-out
            # maps needed for equivalence; they are never retained as artifacts.
            if set(heldout_maps) != set(MULTILAG_ARMS):
                heldout_maps = {}
                for arm in MULTILAG_ARMS:
                    representation, _ = _timed_representation(
                        lambda arm=arm: _representation_for_arm(
                            raw,
                            source_frame_ui,
                            arm=arm,
                            fit=fits_by_arm[arm],
                            chunk_frames=chunk_frames,
                        ),
                        device=resolved_device,
                        arm=arm,
                    )
                    _assert_acquisition_raw_output(
                        representation,
                        raw=raw,
                        source_frame_ui=source_frame_ui,
                        review_frame_ui=frame_ui_device,
                    )
                    heldout_maps[arm] = (
                        representation.values[heldout_mask].detach().cpu()
                    )
                    del representation
                    torch.cuda.empty_cache()
            comparisons = (
                (
                    "difference_multilag_energy_normalized",
                    "pca_whitened_delay_total_energy",
                    False,
                ),
                (
                    "difference_multilag_energy_normalized",
                    "cs_parzen_delay_residual",
                    False,
                ),
                (
                    "pca_whitened_delay_total_energy",
                    "cs_parzen_delay_residual",
                    False,
                ),
            )
            for reference, candidate, matched_packet in comparisons:
                equivalence_rows.append(
                    {
                        "training_fold": fold.training_fold,
                        "heldout_burst": int(fold.heldout_burst),
                        "context_role": lane.role,
                        "context_id": lane.context_id,
                        "reference_representation": reference,
                        "candidate_representation": candidate,
                        "matched_packet_rotation_subspace_diagnostic": matched_packet,
                        **paired_representation_equivalence(
                            heldout_maps[reference], heldout_maps[candidate]
                        ),
                    }
                )
            if int(pca_row["sample_seed"]) != int(ica_row["sample_seed"]):
                matched_pca, _ = _timed_representation(
                    lambda: _representation_for_arm(
                        raw,
                        source_frame_ui,
                        arm="pca_whitened_delay_total_energy",
                        fit=ica_fit,
                        chunk_frames=chunk_frames,
                    ),
                    device=resolved_device,
                    arm="pca_matched_to_selected_ica",
                )
                _assert_acquisition_raw_output(
                    matched_pca,
                    raw=raw,
                    source_frame_ui=source_frame_ui,
                    review_frame_ui=frame_ui_device,
                )
                equivalence_rows.append(
                    {
                        "training_fold": fold.training_fold,
                        "heldout_burst": int(fold.heldout_burst),
                        "context_role": lane.role,
                        "context_id": lane.context_id,
                        "reference_representation": "pca_matched_to_selected_ica",
                        "candidate_representation": "cs_parzen_delay_residual",
                        "matched_packet_rotation_subspace_diagnostic": True,
                        **paired_representation_equivalence(
                            matched_pca.values[heldout_mask].detach().cpu(),
                            heldout_maps["cs_parzen_delay_residual"],
                        ),
                    }
                )
                del matched_pca
                torch.cuda.empty_cache()
            else:
                equivalence_rows[-1][
                    "matched_packet_rotation_subspace_diagnostic"
                ] = True
            del heldout_maps

        if completed_cells != total_cells:
            raise AssertionError("protected context-arm cell count changed")
        _atomic_tsv(work / "candidates_label_sealed.tsv", candidate_rows)
        _atomic_tsv(work / "threshold_calibration.tsv", calibration_rows)
        _atomic_tsv(work / "timings.tsv", timing_rows)
        _atomic_tsv(work / "representation_equivalence.tsv", equivalence_rows)
        current_sealed_hashes = {
            "candidate_table_sha256": _sha256(
                work / "candidates_label_sealed.tsv"
            ),
            "threshold_table_sha256": _sha256(work / "threshold_calibration.tsv"),
            "fit_grid_table_sha256": _sha256(work / "fit_grid.tsv"),
            "selected_fit_table_sha256": _sha256(work / "selected_fit_rows.tsv"),
            "equivalence_table_sha256": _sha256(
                work / "representation_equivalence.tsv"
            ),
        }
        seal_path = work / "candidate_seal.json"
        if seal_path.is_file():
            # Preserve the original pre-label timestamp across a recovery after
            # label joining.  A changed upstream byte fails closed instead of
            # silently creating a post-label replacement seal.
            candidate_seal = json.loads(seal_path.read_text(encoding="utf-8"))
            frozen_hashes = {
                "candidate_table_sha256": candidate_seal["candidate_table"][
                    "sha256"
                ],
                "threshold_table_sha256": candidate_seal[
                    "threshold_table_sha256"
                ],
                "fit_grid_table_sha256": candidate_seal["fit_grid_table_sha256"],
                "selected_fit_table_sha256": candidate_seal[
                    "selected_fit_table_sha256"
                ],
                "equivalence_table_sha256": candidate_seal[
                    "equivalence_table_sha256"
                ],
            }
            if frozen_hashes != current_sealed_hashes:
                raise RuntimeError(
                    "resumed upstream artifacts differ from the original pre-label seal"
                )
        else:
            candidate_seal = {
                "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
                "sparse_positive_fields_parsed_before_seal": False,
                "positive_coordinates_used": False,
                "positive_identities_used": False,
                "candidate_table": {
                    "path": "candidates_label_sealed.tsv",
                    "sha256": current_sealed_hashes["candidate_table_sha256"],
                    "rows": len(candidate_rows),
                },
                "threshold_table_sha256": current_sealed_hashes[
                    "threshold_table_sha256"
                ],
                "fit_grid_table_sha256": current_sealed_hashes[
                    "fit_grid_table_sha256"
                ],
                "selected_fit_table_sha256": current_sealed_hashes[
                    "selected_fit_table_sha256"
                ],
                "equivalence_table_sha256": current_sealed_hashes[
                    "equivalence_table_sha256"
                ],
                "unmatched_candidates": "unknown_not_negative",
            }
            _atomic_json(seal_path, candidate_seal)

        # This is the first point at which sparse-positive rows are parsed.
        heartbeat(
            "protected_label_join",
            candidate_seal_sha256=_sha256(work / "candidate_seal.json"),
        )
        v1 = _read_sparse_positives(
            config.source_paths["protected_labels_v1"],
            selector="include_inclusive",
            expected_rows=79,
            movie_shape_yx=tuple(movie.shape[1:]),
        )
        v7 = _read_sparse_positives(
            config.source_paths["latest_labels_v7"],
            selector="include_confirmed",
            expected_rows=106,
            movie_shape_yx=tuple(movie.shape[1:]),
        )
        v1_matches = observation_match_rows(
            candidate_rows,
            v1,
            cohort="protected_v1",
            operating_rows=calibration_rows,
        )
        v7_matches = observation_match_rows(
            candidate_rows,
            v7,
            cohort="latest_v7_sensitivity",
            operating_rows=calibration_rows,
        )
        v1_aggregate = aggregate_match_rows(v1_matches)
        v7_aggregate = aggregate_match_rows(v7_matches)
        contrasts = clustered_bootstrap_contrasts(
            v1_matches,
            learned_arm="cs_parzen_delay_residual",
            controls=(
                "difference_multilag_energy_normalized",
                "pca_whitened_delay_total_energy",
            ),
        )
        v1_summary = _summary_by_arm(v1_matches)
        v7_summary = _summary_by_arm(v7_matches)
        _atomic_tsv(work / "protected_v1_observation_matches.tsv", v1_matches)
        _atomic_tsv(work / "protected_v1_recall.tsv", v1_aggregate)
        _atomic_tsv(
            work / "protected_v1_clustered_bootstrap_contrasts.tsv", contrasts
        )
        _atomic_tsv(work / "protected_v1_arm_summary.tsv", v1_summary)
        _atomic_tsv(work / "latest_v7_observation_matches.tsv", v7_matches)
        _atomic_tsv(work / "latest_v7_sensitivity.tsv", v7_aggregate)
        _atomic_tsv(work / "latest_v7_arm_summary.tsv", v7_summary)

        rotation_fit_seconds = sum(
            float(fit.diagnostics["fit_seconds"]) for fit in packets.values()
        )
        whitening_fit_seconds = sum(
            float(
                packets[(fold.training_fold, seed, CS_PARZEN_BANDWIDTHS[0])]
                .diagnostics["whitening_fit_seconds"]
            )
            for fold in folds
            for seed in SAMPLE_SEEDS
        )
        fit_seconds = whitening_fit_seconds + rotation_fit_seconds
        peak_vram = max(
            int(row["peak_allocated_vram_bytes_end_to_end"])
            for row in timing_rows
        )
        vram_cap = int(float(config.payload["resources"]["max_peak_vram_gib"]) * 2**30)
        claim_boundary = {
            "protected_v1_population": {
                "selector": "include_inclusive",
                "occurrences": len(v1),
                "canonical_identity_count": len(
                    {row["canonical_roi_id"] for row in v1}
                ),
            },
            "latest_v7_population": {
                "selector": "include_confirmed",
                "occurrences": len(v7),
                "canonical_identity_count": len(
                    {row["canonical_roi_id"] for row in v7}
                ),
                "role": "descriptive_sensitivity_not_independent_confirmation",
            },
            "unmatched_candidates": "unknown_not_negative",
            "precision_identified": False,
            "archived_v5_fit_used_for_projection": False,
            "fold_local_models": True,
            "fit_uses_only_outer_training_burst_windows": True,
            "selection_event_windows_are_outer_training_bursts_only": True,
            "selection_quiet_reference_is_predeclared_crossfit_quiet_halves": True,
            "heldout_burst_and_guard_excluded_from_fit": True,
            "all_six_lag_fit_and_inference_inputs_are_acquisition_raw": True,
            "candidate_artifacts_sealed_before_label_join": True,
            "context_selection_used_sparse_positive_labels": False,
            "scientific_audit_complete": False,
            "paper_promotion_ready": False,
        }
        _atomic_json(work / "claim_boundary.json", claim_boundary)
        summary = {
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "run_type": "protected_outer_fold_matched_six_lag_representation",
            "status": "complete_protected_metrics_scientific_audit_pending",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime,
            "design": multilag_design(),
            "historical_grid_basis": historical,
            "selected_context_role": context_role,
            "selected_contexts": {
                str(fold): lane.as_dict()
                for fold, lane in sorted(selected_contexts.items())
            },
            "anatomy_sampling": anatomy_summary,
            "raw_load_timing": raw_load_timing,
            "raw_source_contract_sha256": raw_source_contract_sha256,
            "fit_timing_seconds": float(fit_seconds),
            "whitening_fit_timing_seconds": float(whitening_fit_seconds),
            "rotation_fit_timing_seconds": float(rotation_fit_seconds),
            "fit_timing_aggregation": (
                "12_unique_whitenings_plus_24_bandwidth_specific_rotations"
            ),
            "fit_count": len(packets),
            "selected_fit_row_count": len(selected_rows),
            "candidate_row_count": len(candidate_rows),
            "candidate_seal": candidate_seal,
            "peak_allocated_vram_bytes": peak_vram,
            "protected_v1": {
                "occurrences": len(v1),
                "canonical_identities": len(
                    {row["canonical_roi_id"] for row in v1}
                ),
                "arm_summary": v1_summary,
                "clustered_bootstrap_contrasts": contrasts,
            },
            "latest_v7_sensitivity": {
                "occurrences": len(v7),
                "canonical_identities": len(
                    {row["canonical_roi_id"] for row in v7}
                ),
                "arm_summary": v7_summary,
                "inferential_claim": False,
            },
            "claim_boundary": claim_boundary,
        }
        _atomic_json(work / "summary.json", summary)
        checks = {
            "exact_24_cs_parzen_rotation_fits": len(packets) == 24,
            "exact_12_unique_full_rank_whitening_models": len(
                {
                    (
                        int(row["training_fold"]),
                        int(row["sample_seed"]),
                        str(row["whitening_sha256"]),
                    )
                    for row in fit_rows
                }
            )
            == 12,
            "four_outer_folds": len(folds) == 4,
            "one_common_context_per_fold": len(selected_contexts) == 4,
            "fit_uses_training_bursts_only": all(
                row["fit_uses_only_outer_training_burst_windows"] for row in fit_rows
            ),
            "all_fit_samples_are_acquisition_raw": len(sample_manifests) == 12
            and all(
                manifest["input_domain"] == "acquisition_raw"
                and manifest["guard_rule"]
                == "every_t_minus_lag_outside_inclusive_heldout_guard"
                and manifest["positive_coordinates_used"] is False
                and manifest["positive_identities_used"] is False
                for manifest in sample_manifests.values()
            ),
            "all_six_lag_arms_assert_acquisition_raw_source": all(
                row["source_domain_assertion_passed"]
                and row["source_domain"] == "acquisition_raw"
                for row in timing_rows
            ),
            "archived_fit_not_loaded": True,
            "candidate_seal_precedes_label_join": True,
            "nms_4_6_8_present": {
                int(row["nms_distance_px"]) for row in calibration_rows
            }
            == {4, 6, 8},
            "candidate_budgets_20_40_58_80_100_present": {
                int(row["candidate_budget"]) for row in v1_matches
            }
            == set(CANDIDATE_BUDGETS_PER_BURST),
            "protected_v1_uses_79_inclusive": len(v1) == 79,
            "protected_v1_has_26_identity_clusters": len(
                {row["canonical_roi_id"] for row in v1}
            )
            == 26,
            "v7_uses_106_confirmed": len(v7) == 106,
            "full_rotation_energy_invariant": all(
                float(row["full_rotation_total_energy_relative_max_error"]) <= 1e-10
                for row in fit_rows
            ),
            "peak_vram_within_manifest_cap": peak_vram <= vram_cap,
            "all_candidates_unknown_before_join": all(
                row["interpretation_before_label_join"] == "unknown_candidate"
                for row in candidate_rows
            ),
            "precision_not_claimed": True,
            "scientific_audit_pending": True,
        }
        _atomic_json(
            work / "validation.json",
            {
                "status": (
                    "passed_protected_metric_artifact_contract"
                    if all(checks.values())
                    else "failed_protected_metric_artifact_contract"
                ),
                "checks": checks,
                "all_checks_pass": all(checks.values()),
            },
        )
        if not all(checks.values()):
            raise RuntimeError("matched-six-lag validation failed closed")
        _atomic_json(
            work / "llm_context.json",
            {
                "entrypoint": "summary.json",
                "grain": (
                    "representation x quiet swap x quiet burden x heldout burst x "
                    "candidate budget"
                ),
                "candidate_seal": "candidate_seal.json",
                "protected_primary": "protected_v1_arm_summary.tsv",
                "paired_inference": (
                    "protected_v1_clustered_bootstrap_contrasts.tsv"
                ),
                "per_observation_recomputability": (
                    "protected_v1_observation_matches.tsv"
                ),
                "latest_sensitivity": "latest_v7_arm_summary.tsv",
                "fit_grid": "fit_grid.tsv",
                "fit_selection": "selected_fit_rows.tsv",
                "timing": "timings.tsv",
                "equivalence": "representation_equivalence.tsv",
                "limitations": [
                    "sparse positives do not identify precision",
                    "v7 is candidate-assisted descriptive sensitivity",
                    "one recording does not establish population generalization",
                    "scientific-audit media are still pending",
                    "archived whole-review v5 fits are diagnostic only",
                ],
            },
        )
        _atomic_text(
            work / "REPORT.md",
            "\n".join(
                (
                    "# Protected matched-six-lag Gamma-LS ablation",
                    "",
                    "The metric stage completed for the fixed raw-domain multi-lag "
                    "difference, independently selected full-rank PCA-whitened total "
                    "energy, and fold-fitted CS-Parzen residual-subspace energy arms.",
                    "",
                    f"- Protected population: {len(v1)} occurrences across "
                    f"{len({row['canonical_roi_id'] for row in v1})} canonical identities.",
                    f"- Fold-fitted rotations: {len(packets)}; selected Gamma context "
                    f"role: `{context_role}`.",
                    f"- Label-sealed candidate rows: {len(candidate_rows)}.",
                    "- Primary outputs: `protected_v1_arm_summary.tsv`, "
                    "`protected_v1_clustered_bootstrap_contrasts.tsv`, "
                    "`representation_equivalence.tsv`, and `timings.tsv`.",
                    "",
                    "Unmatched proposals are unknown, not negatives; precision is not "
                    "identified. The 106-row v7 table is descriptive sensitivity only. "
                    "Scientific-audit media remain pending, so this artifact does not "
                    "by itself authorize a manuscript headline.",
                )
            ),
        )
        _atomic_json(
            work / "status.json",
            {
                "status": "complete_protected_metrics_scientific_audit_pending",
                "completed_at_utc": summary["completed_at_utc"],
                "validation_passed": True,
                "candidate_seal_sha256": _sha256(work / "candidate_seal.json"),
                "scientific_audit_complete": False,
                "paper_promotion_ready": False,
            },
        )
        heartbeat(
            "complete",
            status="complete_protected_metrics_scientific_audit_pending",
        )
        _atomic_json(work / "artifact_index.json", _artifact_index(work))
        if destination.exists():
            raise FileExistsError(
                f"matched-six-lag output appeared during execution: {destination}"
            )
        work.replace(destination)
        return summary
    except Exception as error:
        if work.exists():
            _atomic_json(
                work / "status.json",
                {
                    "status": "interrupted_or_failed_resumable",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": repr(error),
                    "safe_to_resume_with_identical_contract": True,
                },
            )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--context-selection-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-role", default=DEFAULT_CONTEXT_ROLE)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args(argv)
    config = GammaLSDifferenceConfig.load(arguments.config)
    result = run_multilag_protected_experiment(
        config,
        preflight_dir=arguments.preflight_dir,
        context_selection_dir=arguments.context_selection_dir,
        output_dir=arguments.output_dir,
        context_role=arguments.context_role,
        device=arguments.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIRMATION_SAMPLES",
    "CS_PARZEN_BANDWIDTHS",
    "DEFAULT_CONTEXT_ROLE",
    "FoldDelaySamples",
    "FoldDelayWhitening",
    "MULTILAG_ARMS",
    "MultilagProtectedUnavailable",
    "SAMPLE_SEEDS",
    "SCREEN_SAMPLES",
    "audit_historical_v5_surface",
    "eligible_delay_current_frames",
    "fit_cs_parzen_delay_embedding",
    "fit_full_rank_delay_whitening",
    "fit_from_dict",
    "guard_safe_selection_bursts",
    "main",
    "multilag_design",
    "run_multilag_protected_experiment",
    "sample_fold_delay_observations",
    "select_fold_contexts",
    "select_fold_models",
]
