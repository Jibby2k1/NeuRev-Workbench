"""Causal, annotation-file-free full-recording Gamma-LS proposal runner.

The runner freezes two deployment-calibration variants while making one
causal pass over the complete Spon Ca Burst acquisition:

``initial_100``
    Fit the local-scale floor and empirical thresholds from two nonoverlapping
    one-second blocks in UI frames 1--100.  The block duration is derived only
    from the declared 20-ms acquisition cadence.  These frames are not assumed
    event-free.  Frozen operating points apply only to UI frames 101--2359.

``declared_quiet``
    Fit from the human-declared UI interval 1800--1899, then apply only to UI
    frames 1900--2359.  This variant is explicitly burst-window-supervised.

The fixed representation arm and radial Gamma-LS context are command inputs;
this module never selects either one.  Dense causal preprocessing, the fixed
representation, Gamma moments, scale-floor scoring, and host transfer all use
bounded chunks with dense scoring on CUDA.  Candidate extraction calls the
maintained deterministic greedy Euclidean NMS on CPU.  It is exact for the
transferred float32 score maps, rather than the plateau approximation used by
the separate latency-only streaming benchmark.

No annotation table is opened.  The output is one row per frame-level
proposal.  It deliberately performs no temporal linking and reports no unique
biological event count.
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
import shutil
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid

import numpy as np

from neurobench.algorithms.gamma_local_standardization import (
    GammaReferenceSpec,
    gamma_local_standardization,
)

from . import gpu_representations
from .config import GammaLSDifferenceConfig
from .cuda_runtime import CudaRuntimeUnavailable, require_cuda_device
from .evaluation import (
    NMS_DISTANCE_PX,
    QUIET_NMS_PEAK_BURDENS,
    calibrate_training_quiet_thresholds,
    strict_separated_nms,
)


FULL_RECORDING_ARMS = (
    "raw",
    "difference_signed",
    "difference_energy_normalized",
)
TOTAL_FRAMES = 2359
FRAME_HEIGHT = 340
FRAME_WIDTH = 573
INITIAL_CALIBRATION_UI = (1, 100)
INITIAL_APPLICATION_UI = (101, TOTAL_FRAMES)
DECLARED_CALIBRATION_UI = (1800, 1899)
DECLARED_APPLICATION_UI = (1900, TOTAL_FRAMES)
ENERGY_EPSILON = 1e-8
GAMMA_EPSILON = 1e-6
DEFAULT_MAX_CANDIDATES_PER_FRAME = FRAME_HEIGHT * FRAME_WIDTH
INITIAL_BURDEN_UNIT = "nms_peaks_per_nonoverlapping_1s_initialization_block"
DECLARED_BURDEN_UNIT = "nms_peaks_per_duration_matched_pseudo_burst"


class FullRecordingProposalUnavailable(RuntimeError):
    """Raised before final-output mutation when a frozen gate does not pass."""


@dataclass(frozen=True)
class CalibrationVariant:
    """One frozen calibration/application boundary."""

    variant_id: str
    calibration_interval_ui: tuple[int, int]
    application_interval_ui: tuple[int, int]
    calibration_role: str
    calibration_frame_locations_use_annotation_content: bool
    burst_window_supervised: bool
    calibration_window_rule: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "calibration_interval_ui": list(self.calibration_interval_ui),
            "application_interval_ui": list(self.application_interval_ui),
            "calibration_role": self.calibration_role,
            "calibration_frame_locations_use_annotation_content": (
                self.calibration_frame_locations_use_annotation_content
            ),
            "calibration_window_rule": self.calibration_window_rule,
            "burst_window_supervised": self.burst_window_supervised,
        }


CALIBRATION_VARIANTS = (
    CalibrationVariant(
        variant_id="initial_100_annotation_file_and_location_free",
        calibration_interval_ui=INITIAL_CALIBRATION_UI,
        application_interval_ui=INITIAL_APPLICATION_UI,
        calibration_role="initialization_frames_not_assumed_event_free",
        calibration_frame_locations_use_annotation_content=False,
        burst_window_supervised=False,
        calibration_window_rule="two_nonoverlapping_one_second_blocks_from_frame_interval",
    ),
    CalibrationVariant(
        variant_id="declared_quiet_burst_window_supervised",
        calibration_interval_ui=DECLARED_CALIBRATION_UI,
        application_interval_ui=DECLARED_APPLICATION_UI,
        calibration_role="human_declared_quiet_interval",
        calibration_frame_locations_use_annotation_content=True,
        burst_window_supervised=True,
        calibration_window_rule="configured_burst_duration_matched_pseudo_bursts",
    ),
)


def _burden_unit_for_variant(variant_id: str) -> str:
    if variant_id == CALIBRATION_VARIANTS[0].variant_id:
        return INITIAL_BURDEN_UNIT
    if variant_id == CALIBRATION_VARIANTS[1].variant_id:
        return DECLARED_BURDEN_UNIT
    raise ValueError(f"unknown calibration variant: {variant_id!r}")


CANDIDATE_FIELDS = (
    "proposal_id",
    "variant_id",
    "representation",
    "context_id",
    "target_nms_peaks_per_calibration_unit",
    "calibration_burden_unit",
    "threshold_z",
    "scale_floor_percentile",
    "scale_floor",
    "source_frame_ui",
    "candidate_rank_within_frame",
    "score",
    "x_px",
    "y_px",
    "biological_status",
    "temporal_linking_applied",
)

FRAME_COUNT_FIELDS = (
    "variant_id",
    "representation",
    "context_id",
    "target_nms_peaks_per_calibration_unit",
    "calibration_burden_unit",
    "threshold_z",
    "source_frame_ui",
    "proposal_count",
)

THRESHOLD_FIELDS = (
    "variant_id",
    "calibration_interval_ui",
    "calibration_frame_first_ui",
    "calibration_frame_last_ui",
    "calibration_frame_count",
    "representation_alignment_note",
    "calibration_role",
    "calibration_frame_locations_use_annotation_content",
    "calibration_window_template_ui_frames",
    "calibration_window_template_source",
    "calibration_window_template_is_annotation_derived",
    "burst_window_supervised",
    "scale_floor_percentile",
    "scale_floor",
    "target_nms_peaks_per_calibration_unit",
    "calibration_burden_unit",
    "threshold_z",
    "achieved_nms_peaks_per_calibration_unit",
    "total_calibration_nms_peaks",
    "calibration_windows_source_ui",
    "calibration_window_overlap_frames",
    "nms_distance_px",
    "probability_model_claimed",
)


@dataclass(frozen=True)
class FullRecordingExecution:
    """In-memory tables returned by the bounded CUDA executor."""

    candidate_rows: tuple[Mapping[str, Any], ...]
    frame_count_rows: tuple[Mapping[str, Any], ...]
    threshold_rows: tuple[Mapping[str, Any], ...]
    execution_summary: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_tsv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    fields = list(fieldnames)
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("TSV field names must be nonempty and unique")
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f"table {path.name} has inconsistent fields")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _artifact_index(root: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and path.name != "artifact_index.json"
            and not path.name.endswith(".partial")
        ):
            artifacts.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return {"schema_version": 1, "artifacts": artifacts}


def verify_indexed_artifact(root: str | Path) -> dict[str, Any]:
    """Verify every file frozen by an artifact index."""

    directory = Path(root).expanduser().resolve()
    index_path = directory / "artifact_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("artifacts"), list
    ):
        raise FullRecordingProposalUnavailable("artifact index is invalid")
    seen: set[str] = set()
    for item in payload["artifacts"]:
        relative = str(item.get("path", ""))
        if not relative or relative in seen or Path(relative).is_absolute():
            raise FullRecordingProposalUnavailable("artifact index path is invalid")
        seen.add(relative)
        path = (directory / relative).resolve()
        if directory not in path.parents:
            raise FullRecordingProposalUnavailable(
                "artifact index path escapes its frozen root"
            )
        if not path.is_file():
            raise FullRecordingProposalUnavailable(
                f"indexed artifact is missing: {relative}"
            )
        if path.stat().st_size != int(item["size_bytes"]) or _sha256(path) != str(
            item["sha256"]
        ):
            raise FullRecordingProposalUnavailable(
                f"indexed artifact changed after freeze: {relative}"
            )
    return {
        "root": str(directory),
        "artifact_index_sha256": _sha256(index_path),
        "verified_artifact_count": len(seen),
    }


def _verify_preflight_without_annotation_reads(
    config: GammaLSDifferenceConfig,
    preflight_dir: str | Path,
) -> dict[str, Any]:
    """Verify a current GPU preflight while hashing only the movie source.

    The general experiment preflight freezes every source, including annotation
    sources.  This runner consumes only its metadata and the movie hash; it does
    not reopen any annotation source while verifying the run.
    """

    from .preflight import _implementation_status

    root = Path(preflight_dir).expanduser().resolve()
    indexed = verify_indexed_artifact(root)
    payload = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
    portable = json.loads(
        (root / "config.portable.json").read_text(encoding="utf-8")
    )
    if portable != config.portable_dict():
        raise FullRecordingProposalUnavailable("preflight config is stale")
    if not payload.get("data_ready") or not payload.get("gpu_run_ready"):
        raise FullRecordingProposalUnavailable("preflight is not GPU-run ready")
    current = _implementation_status(config)
    frozen_files = payload.get("implementation", {}).get("files")
    if not current.get("complete") or current.get("files") != frozen_files:
        raise FullRecordingProposalUnavailable(
            "implementation fingerprints changed after preflight"
        )
    executor_key = (
        "repo://neurobench/experiments/gamma_ls_difference/full_recording.py"
    )
    if executor_key not in frozen_files:
        raise FullRecordingProposalUnavailable(
            "preflight does not fingerprint the full-recording executor"
        )
    movie = config.source_paths["movie"]
    frozen_movie_hash = payload.get("source", {}).get("movie", {}).get("sha256")
    if not frozen_movie_hash or _sha256(movie) != frozen_movie_hash:
        raise FullRecordingProposalUnavailable(
            "movie fingerprint changed after preflight"
        )
    return {
        **indexed,
        "preflight_sha256": _sha256(root / "preflight.json"),
        "portable_config_sha256": _canonical_sha256(portable),
        "movie_sha256": frozen_movie_hash,
        "executor_sha256": frozen_files[executor_key]["sha256"],
        "annotation_sources_reopened": False,
    }


def _context_geometry(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    required = {
        "context_id",
        "half_width_px",
        "guard_radius_px",
        "shape",
        "mode_fraction_of_half_width",
        "mode_radius_px",
        "support",
        "padding",
        "eligible_primary",
    }
    if not required.issubset(payload):
        raise FullRecordingProposalUnavailable(
            "selected context lacks explicit radial geometry"
        )
    return (
        str(payload["context_id"]),
        int(payload["half_width_px"]),
        int(payload["guard_radius_px"]),
        float(payload["shape"]),
        float(payload["mode_fraction_of_half_width"]),
        float(payload["mode_radius_px"]),
        str(payload["support"]),
        str(payload["padding"]),
        bool(payload["eligible_primary"]),
    )


def _reference_from_context(payload: Mapping[str, Any]) -> GammaReferenceSpec:
    geometry = _context_geometry(payload)
    (
        context_id,
        half,
        guard,
        shape,
        mode_fraction,
        mode_radius,
        support,
        padding,
        eligible,
    ) = geometry
    if not eligible or support != "radial_disk":
        raise FullRecordingProposalUnavailable(
            "deployment context must be primary-eligible radial Gamma-LS"
        )
    if padding != "valid_renormalized_zero":
        raise FullRecordingProposalUnavailable(
            "deployment context changed the valid-reference border contract"
        )
    if not 0 <= guard < half or not math.isclose(
        mode_radius, half * mode_fraction
    ):
        raise FullRecordingProposalUnavailable("deployment context geometry is invalid")
    return GammaReferenceSpec.from_mode(
        context_id,
        support_width_px=2 * half + 1,
        shape_n=shape,
        mode_radius_px=mode_radius,
        guard_radius_px=guard,
        support_geometry="disk",
        boundary_mode="valid_renormalized_zero",
        epsilon=GAMMA_EPSILON,
        scale_floor=0.0,
    )


def _verify_support_context(
    support_dir: str | Path,
    context_id: str,
) -> tuple[GammaReferenceSpec, dict[str, Any]]:
    """Verify the immutable support screen and locate an explicit context."""

    root = Path(support_dir).expanduser().resolve()
    indexed = verify_indexed_artifact(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (root / "validation.json").read_text(encoding="utf-8")
    )
    contexts = json.loads(
        (root / "fold_contexts.json").read_text(encoding="utf-8")
    )
    if summary.get("status") != "complete_support_screen_only":
        raise FullRecordingProposalUnavailable("support screen is not complete")
    if validation.get("status") != (
        "passed_support_screen_artifact_contract_scientific_audit_pending"
    ):
        raise FullRecordingProposalUnavailable(
            "support screen artifact contract did not pass"
        )
    if contexts.get("selection_scope") != "outer_training_fold_only":
        raise FullRecordingProposalUnavailable("support context scope changed")
    if contexts.get("selection_uses_positive_coordinates") is not False or contexts.get(
        "selection_uses_positive_identities"
    ) is not False:
        raise FullRecordingProposalUnavailable(
            "support context artifact contains protected selection fields"
        )
    if contexts.get("burst_windows_used") is not True:
        raise FullRecordingProposalUnavailable(
            "support artifact must disclose burst-window supervision"
        )
    roles = (
        "original_screen_context",
        "support_candidate_context",
        "larger_support_comparator",
        "training_best_context",
        "max_support_endpoint",
    )
    matches: list[tuple[int, str, Mapping[str, Any]]] = []
    folds = contexts.get("folds")
    if not isinstance(folds, list) or len(folds) != 4:
        raise FullRecordingProposalUnavailable(
            "support context artifact must contain four folds"
        )
    for fold in folds:
        fold_id = int(fold["training_fold"])
        for role in roles:
            candidate = fold.get(role)
            if isinstance(candidate, Mapping) and str(candidate.get("context_id")) == str(
                context_id
            ):
                matches.append((fold_id, role, candidate))
    if not matches:
        raise FullRecordingProposalUnavailable(
            "explicit context is absent from the frozen support artifact"
        )
    geometry = _context_geometry(matches[0][2])
    if any(_context_geometry(row[2]) != geometry for row in matches[1:]):
        raise FullRecordingProposalUnavailable(
            "support artifact assigns conflicting geometry to the context id"
        )
    reference = _reference_from_context(matches[0][2])
    return reference, {
        **indexed,
        "summary_sha256": _sha256(root / "summary.json"),
        "validation_sha256": _sha256(root / "validation.json"),
        "fold_contexts_sha256": _sha256(root / "fold_contexts.json"),
        "context_id": context_id,
        "context_occurrences": [
            {"training_fold": fold, "role": role} for fold, role, _ in matches
        ],
        "selection_scope": "explicit_input_not_selected_by_full_recording_runner",
        "support_screen_burst_windows_used": True,
        "positive_coordinates_used_for_support_selection": False,
        "positive_identities_used_for_support_selection": False,
    }


def expected_calibration_frame_count(arm: str, variant: CalibrationVariant) -> int:
    """Return the exact number of aligned representation frames used to fit."""

    if arm not in FULL_RECORDING_ARMS:
        raise ValueError(f"unsupported full-recording arm: {arm!r}")
    start, stop = variant.calibration_interval_ui
    return stop - start + 1


def fit_empirical_calibration_thresholds(
    scores: np.ndarray,
    *,
    window_durations: Mapping[str, int],
    burden_unit: str,
) -> Any:
    """Fit the protected occupancy/NMS threshold semantics on explicit windows.

    Numeric burdens are never interpreted per frame.  The operational variant
    uses two one-second initialization blocks; the comparison variant uses the
    configured duration-matched pseudo-bursts.
    """

    if burden_unit not in {INITIAL_BURDEN_UNIT, DECLARED_BURDEN_UNIT}:
        raise ValueError("per-frame burden reinterpretation is not permitted")
    durations = {str(key): int(value) for key, value in window_durations.items()}
    if burden_unit == INITIAL_BURDEN_UNIT and durations != {
        "initial_1s_block_1": 50,
        "initial_1s_block_2": 50,
    }:
        raise ValueError("initial calibration requires exactly two 50-frame blocks")
    if not durations or any(value < 1 for value in durations.values()):
        raise ValueError("calibration window durations must be positive")
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError("calibration scores must be finite TYX")
    return calibrate_training_quiet_thresholds(
        values,
        np.ones(len(values), dtype=np.bool_),
        durations,
        target_peak_burdens=QUIET_NMS_PEAK_BURDENS,
        nms_distance_px=NMS_DISTANCE_PX,
    )


def extract_frame_proposals(
    score: np.ndarray,
    *,
    source_frame_ui: int,
    variant_id: str,
    representation: str,
    context_id: str,
    operating_points: Sequence[Mapping[str, Any]],
    scale_floor_percentile: float,
    scale_floor: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract exact frame-level proposals for every frozen operating point."""

    if tuple(
        float(row["target_nms_peaks_per_calibration_unit"])
        for row in operating_points
    ) != QUIET_NMS_PEAK_BURDENS:
        raise ValueError("operating points left the frozen burden grid")
    thresholds = [float(row["threshold_z"]) for row in operating_points]
    if not all(math.isfinite(value) for value in thresholds):
        raise ValueError("operating thresholds must be finite")
    peaks = strict_separated_nms(
        score,
        distance_px=NMS_DISTANCE_PX,
        threshold=min(thresholds),
        limit=DEFAULT_MAX_CANDIDATES_PER_FRAME,
    )
    proposals: list[dict[str, Any]] = []
    counts: list[dict[str, Any]] = []
    for operating in operating_points:
        target = float(operating["target_nms_peaks_per_calibration_unit"])
        threshold = float(operating["threshold_z"])
        burden_unit = str(operating["calibration_burden_unit"])
        if burden_unit != _burden_unit_for_variant(variant_id):
            raise ValueError("operating point uses an unsupported burden unit")
        retained = [peak for peak in peaks if float(peak[0]) > threshold]
        counts.append(
            {
                "variant_id": variant_id,
                "representation": representation,
                "context_id": context_id,
                "target_nms_peaks_per_calibration_unit": target,
                "calibration_burden_unit": burden_unit,
                "threshold_z": threshold,
                "source_frame_ui": int(source_frame_ui),
                "proposal_count": len(retained),
            }
        )
        target_token = str(target).replace(".", "p")
        for rank, (value, x_px, y_px) in enumerate(retained, start=1):
            proposals.append(
                {
                    "proposal_id": (
                        f"{variant_id}__b{target_token}__ui{int(source_frame_ui):04d}"
                        f"__r{rank:05d}"
                    ),
                    "variant_id": variant_id,
                    "representation": representation,
                    "context_id": context_id,
                    "target_nms_peaks_per_calibration_unit": target,
                    "calibration_burden_unit": burden_unit,
                    "threshold_z": threshold,
                    "scale_floor_percentile": float(scale_floor_percentile),
                    "scale_floor": float(scale_floor),
                    "source_frame_ui": int(source_frame_ui),
                    "candidate_rank_within_frame": rank,
                    "score": float(value),
                    "x_px": int(x_px),
                    "y_px": int(y_px),
                    "biological_status": "unknown_unreviewed_proposal",
                    "temporal_linking_applied": False,
                }
            )
    return proposals, counts


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    variant_order = {
        item.variant_id: index for index, item in enumerate(CALIBRATION_VARIANTS)
    }
    burden_order = {
        value: index for index, value in enumerate(QUIET_NMS_PEAK_BURDENS)
    }
    return (
        variant_order[str(row["variant_id"])],
        burden_order[float(row["target_nms_peaks_per_calibration_unit"])],
        int(row["source_frame_ui"]),
        int(row["candidate_rank_within_frame"]),
    )


def _frame_count_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    variant_order = {
        item.variant_id: index for index, item in enumerate(CALIBRATION_VARIANTS)
    }
    burden_order = {
        value: index for index, value in enumerate(QUIET_NMS_PEAK_BURDENS)
    }
    return (
        variant_order[str(row["variant_id"])],
        burden_order[float(row["target_nms_peaks_per_calibration_unit"])],
        int(row["source_frame_ui"]),
    )


def candidate_content_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash sorted proposal content independently of run timestamps."""

    normalized = [dict(row) for row in sorted(rows, key=_candidate_sort_key)]
    return _canonical_sha256(normalized)


def _aligned_chunks(
    total_frames: int,
    chunk_frames: int,
    *,
    boundary_stops_ui: Iterable[int],
) -> list[tuple[int, int]]:
    """Return zero-based chunks that end exactly at every protocol boundary."""

    if total_frames < 1 or chunk_frames < 1:
        raise ValueError("frame and chunk counts must be positive")
    boundaries = sorted(
        {int(value) for value in boundary_stops_ui if 0 < int(value) < total_frames}
    )
    result: list[tuple[int, int]] = []
    start = 0
    while start < total_frames:
        stop = min(start + chunk_frames, total_frames)
        inside = [value for value in boundaries if start < value < stop]
        if inside:
            stop = min(inside)
        result.append((start, stop))
        start = stop
    if result[0][0] != 0 or result[-1][1] != total_frames:
        raise AssertionError("aligned chunks did not cover the full acquisition")
    if any(left[1] != right[0] for left, right in zip(result, result[1:])):
        raise AssertionError("aligned chunks are not contiguous")
    return result


class _CausalCudaArm:
    """Stateful exact Gaussian/EMA and one fixed adjacent representation."""

    def __init__(
        self,
        *,
        arm: str,
        device: Any,
        frame_shape: tuple[int, int],
    ) -> None:
        import torch

        if arm not in FULL_RECORDING_ARMS:
            raise ValueError(f"unsupported full-recording arm: {arm!r}")
        self.torch = torch
        self.arm = arm
        self.device = torch.device(device)
        self.height, self.width = map(int, frame_shape)
        radius = int(
            gpu_representations.GAUSSIAN_TRUNCATE
            * gpu_representations.SPATIAL_SIGMA_PX
            + 0.5
        )
        coordinates = torch.arange(
            -radius, radius + 1, dtype=torch.float32, device=self.device
        )
        kernel = torch.exp(
            -0.5
            * (coordinates / gpu_representations.SPATIAL_SIGMA_PX).square()
        )
        self.gaussian = kernel / kernel.sum()
        self.y_indices = gpu_representations._scipy_reflect_indices(
            self.height, radius, device=self.device
        )
        self.x_indices = gpu_representations._scipy_reflect_indices(
            self.width, radius, device=self.device
        )
        self.ema_state = None
        self.previous_common = None

    def _spatial(self, values: Any) -> Any:
        import torch.nn.functional as functional

        frames = values.unsqueeze(1)
        vertical = functional.conv2d(
            self.torch.index_select(frames, -2, self.y_indices),
            self.gaussian.reshape(1, 1, -1, 1),
        )
        return functional.conv2d(
            self.torch.index_select(vertical, -1, self.x_indices),
            self.gaussian.reshape(1, 1, 1, -1),
        ).squeeze(1)

    def process(
        self,
        host_chunk: np.ndarray,
        *,
        first_source_frame_ui: int,
    ) -> tuple[Any, np.ndarray, dict[str, float]]:
        """Transfer/process a contiguous uint16 chunk and preserve causal state."""

        torch = self.torch
        source = np.asarray(host_chunk)
        if (
            source.ndim != 3
            or tuple(map(int, source.shape[1:])) != (self.height, self.width)
            or source.dtype != np.uint16
        ):
            raise ValueError("source chunk must be uint16 TYX at the frozen geometry")

        def timed(operation: Callable[[], Any]) -> tuple[Any, float]:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            value = operation()
            stop.record()
            stop.synchronize()
            return value, float(start.elapsed_time(stop))

        host_tensor = torch.from_numpy(np.ascontiguousarray(source))
        device_values, h2d_ms = timed(
            lambda: host_tensor.to(
                device=self.device, dtype=torch.float32, non_blocking=False
            )
        )

        def represent() -> tuple[Any, np.ndarray]:
            spatial = self._spatial(device_values)
            outputs = []
            source_ui = []
            for offset in range(int(spatial.shape[0])):
                ui = int(first_source_frame_ui) + offset
                if self.ema_state is None:
                    common = spatial[offset]
                else:
                    common = (
                        gpu_representations.EMA_ALPHA * spatial[offset]
                        + (1.0 - gpu_representations.EMA_ALPHA) * self.ema_state
                    )
                if self.arm == "raw":
                    output = common
                elif self.previous_common is None:
                    # The streaming deployment contract has no pre-acquisition
                    # predecessor.  Its deterministic cold start is a zero
                    # adjacent difference at UI frame 1.  Application begins
                    # only at UI frame 101, after genuine causal history.
                    output = torch.zeros_like(common)
                else:
                    difference = common - self.previous_common
                    if self.arm == "difference_signed":
                        output = difference
                    else:
                        output = difference / torch.sqrt(
                            common.square()
                            + self.previous_common.square()
                            + ENERGY_EPSILON
                        )
                self.ema_state = common
                self.previous_common = common
                if output is not None:
                    outputs.append(output)
                    source_ui.append(ui)
            if not outputs:
                return torch.empty(
                    (0, self.height, self.width),
                    dtype=torch.float32,
                    device=self.device,
                ), np.asarray([], dtype=np.int64)
            return torch.stack(outputs), np.asarray(source_ui, dtype=np.int64)

        (representation, source_ui), preprocessing_ms = timed(represent)
        return representation, source_ui, {
            "h2d_ms": h2d_ms,
            "preprocessing_and_representation_ms": preprocessing_ms,
        }


def _exact_positive_quantile_on_host(
    values: Any, percentile: float
) -> tuple[float, int]:
    """Compute the full-sample linear percentile beyond CUDA's size limit.

    CUDA ``torch.quantile`` rejects inputs above 2**24 elements.  A complete
    100-frame calibration tensor can exceed that limit, and the frozen contract
    does not permit sampling.  Transfer every value to the host and use NumPy's
    equivalent deterministic linear quantile instead.
    """

    requested = float(percentile)
    if not math.isfinite(requested) or not 0.0 <= requested <= 100.0:
        raise ValueError("scale floor percentile must be finite and in [0, 100]")
    host = values.detach().cpu().numpy()
    positive = host[host > 0]
    if positive.size < 1:
        raise RuntimeError("calibration local standard deviation has no positive values")
    floor = float(np.quantile(positive, requested / 100.0, method="linear"))
    if not math.isfinite(floor) or floor <= 0:
        raise RuntimeError("calibration scale floor is not finite and positive")
    return floor, int(positive.size)


def _finalize_calibration(
    *,
    variant: CalibrationVariant,
    arm: str,
    representation_parts: Sequence[Any],
    mean_parts: Sequence[Any],
    std_parts: Sequence[Any],
    source_ui_parts: Sequence[np.ndarray],
    scale_floor_percentile: float,
    frame_interval_ms: float,
    configured_burst_durations: Mapping[str, int],
) -> tuple[float, tuple[dict[str, Any], ...], dict[str, Any]]:
    import torch

    if not representation_parts or not mean_parts or not std_parts:
        raise RuntimeError(f"calibration {variant.variant_id} has no CUDA moments")
    representation = torch.cat(tuple(representation_parts), dim=0)
    local_mean = torch.cat(tuple(mean_parts), dim=0)
    local_std = torch.cat(tuple(std_parts), dim=0)
    source_ui = np.concatenate(tuple(source_ui_parts))
    expected = expected_calibration_frame_count(arm, variant)
    if len(source_ui) != expected or int(representation.shape[0]) != expected:
        raise RuntimeError(
            f"calibration {variant.variant_id} expected {expected} aligned frames, "
            f"received {len(source_ui)}"
        )
    floor, positive_count = _exact_positive_quantile_on_host(
        local_std, scale_floor_percentile
    )
    floor_tensor = torch.as_tensor(
        floor, dtype=local_std.dtype, device=local_std.device
    )
    score = (representation - local_mean) / (
        torch.maximum(local_std, floor_tensor) + GAMMA_EPSILON
    )
    score_host = score.detach().cpu().numpy().astype(np.float32, copy=False)
    torch.cuda.synchronize(representation.device)
    if variant.variant_id == CALIBRATION_VARIANTS[0].variant_id:
        frames_per_second = int(round(1000.0 / float(frame_interval_ms)))
        if not math.isclose(
            frames_per_second * float(frame_interval_ms),
            1000.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "frame interval does not define an integral one-second block"
            )
        window_durations = {
            "initial_1s_block_1": frames_per_second,
            "initial_1s_block_2": frames_per_second,
        }
        burden_unit = INITIAL_BURDEN_UNIT
        window_template_source = "declared_frame_interval_only"
        template_annotation_derived = False
    else:
        window_durations = {
            str(key): int(value)
            for key, value in configured_burst_durations.items()
        }
        burden_unit = DECLARED_BURDEN_UNIT
        window_template_source = "configured_burst_interval_durations"
        template_annotation_derived = True
    calibration = fit_empirical_calibration_thresholds(
        score_host,
        window_durations=window_durations,
        burden_unit=burden_unit,
    )
    windows_source_ui = {
        str(key): [int(source_ui[start]), int(source_ui[stop - 1])]
        for key, (start, stop) in calibration.pseudo_burst_windows.items()
    }
    alignment_note = (
        "UI 1 uses a zero-difference cold start; UI 2 onward uses the causal predecessor"
        if arm != "raw" and variant.calibration_interval_ui[0] == 1
        else "all frames in the declared calibration interval"
    )
    rows = tuple(
        {
            "variant_id": variant.variant_id,
            "calibration_interval_ui": json.dumps(
                list(variant.calibration_interval_ui), separators=(",", ":")
            ),
            "calibration_frame_first_ui": int(source_ui[0]),
            "calibration_frame_last_ui": int(source_ui[-1]),
            "calibration_frame_count": int(len(source_ui)),
            "representation_alignment_note": alignment_note,
            "calibration_role": variant.calibration_role,
            "calibration_frame_locations_use_annotation_content": (
                variant.calibration_frame_locations_use_annotation_content
            ),
            "calibration_window_template_ui_frames": json.dumps(
                window_durations, sort_keys=True, separators=(",", ":")
            ),
            "calibration_window_template_source": window_template_source,
            "calibration_window_template_is_annotation_derived": (
                template_annotation_derived
            ),
            "burst_window_supervised": variant.burst_window_supervised,
            "scale_floor_percentile": float(scale_floor_percentile),
            "scale_floor": floor,
            "target_nms_peaks_per_calibration_unit": float(
                row["target_nms_peaks_per_pseudo_burst"]
            ),
            "calibration_burden_unit": burden_unit,
            "threshold_z": float(row["threshold_z"]),
            "achieved_nms_peaks_per_calibration_unit": float(
                row["achieved_nms_peaks_per_pseudo_burst"]
            ),
            "total_calibration_nms_peaks": int(
                row["total_nms_peaks"]
            ),
            "calibration_windows_source_ui": json.dumps(
                windows_source_ui, sort_keys=True, separators=(",", ":")
            ),
            "calibration_window_overlap_frames": int(
                calibration.diagnostics["pseudo_burst_windows_overlap_frames"]
            ),
            "nms_distance_px": NMS_DISTANCE_PX,
            "probability_model_claimed": False,
        }
        for row in calibration.operating_points
    )
    return floor, rows, {
        "variant_id": variant.variant_id,
        "calibration_frame_count": len(source_ui),
        "calibration_frame_first_ui": int(source_ui[0]),
        "calibration_frame_last_ui": int(source_ui[-1]),
        "positive_local_std_sample_count": positive_count,
        "scale_floor_percentile": float(scale_floor_percentile),
        "scale_floor": floor,
        "calibration_burden_unit": burden_unit,
        "calibration_window_template_ui_frames": window_durations,
        "calibration_windows_source_ui": windows_source_ui,
        "calibration_window_overlap_frames": int(
            calibration.diagnostics["pseudo_burst_windows_overlap_frames"]
        ),
        "cpu_nms_exact_parity": "maintained_strict_separated_nms_on_transferred_float32_scores",
    }


def _execute_cuda_full_recording(
    *,
    movie_path: Path,
    arm: str,
    reference: GammaReferenceSpec,
    device: Any,
    chunk_frames: int,
    scale_floor_percentile: float,
    frame_interval_ms: float,
    configured_burst_durations: Mapping[str, int],
    progress: Callable[[Mapping[str, Any]], None] | None,
) -> FullRecordingExecution:
    """Execute the one-pass causal full-recording proposal program."""

    import torch

    movie = np.load(movie_path, mmap_mode="r", allow_pickle=False)
    if not isinstance(movie, np.memmap):
        raise ValueError("full-recording source must remain memory-mappable")
    if tuple(map(int, movie.shape)) != (TOTAL_FRAMES, FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(
            "full-recording source must remain 2359x340x573 in TYX order"
        )
    if movie.dtype != np.uint16:
        raise ValueError("full-recording source must preserve uint16 camera values")
    if arm not in FULL_RECORDING_ARMS:
        raise ValueError(f"unsupported full-recording arm: {arm!r}")
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be positive")

    processor = _CausalCudaArm(
        arm=arm,
        device=device,
        frame_shape=(FRAME_HEIGHT, FRAME_WIDTH),
    )
    zero_floor_reference = reference
    collectors: dict[str, dict[str, list[Any]]] = {
        variant.variant_id: {
            "representation": [],
            "mean": [],
            "std": [],
            "source_ui": [],
        }
        for variant in CALIBRATION_VARIANTS
    }
    floors: dict[str, float] = {}
    operating_by_variant: dict[str, tuple[dict[str, Any], ...]] = {}
    threshold_rows: list[dict[str, Any]] = []
    calibration_summaries: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    frame_count_rows: list[dict[str, Any]] = []
    timings = {
        "h2d_ms": 0.0,
        "preprocessing_and_representation_ms": 0.0,
        "gamma_moments_ms": 0.0,
        "score_and_d2h_ms": 0.0,
        "cpu_exact_nms_ms": 0.0,
    }
    chunks = _aligned_chunks(
        TOTAL_FRAMES,
        chunk_frames,
        boundary_stops_ui=(
            INITIAL_CALIBRATION_UI[1],
            DECLARED_CALIBRATION_UI[0] - 1,
            DECLARED_CALIBRATION_UI[1],
        ),
    )
    torch.cuda.reset_peak_memory_stats(device)
    free_before, total_vram = torch.cuda.mem_get_info(device)
    wall_started = time.perf_counter()

    def timed_cuda(operation: Callable[[], Any]) -> tuple[Any, float]:
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        value = operation()
        stop.record()
        stop.synchronize()
        return value, float(start.elapsed_time(stop))

    with torch.inference_mode():
        for chunk_index, (start_zero, stop_zero) in enumerate(chunks, start=1):
            host = np.array(
                movie[start_zero:stop_zero], dtype=np.uint16, order="C", copy=True
            )
            representation, source_ui, stage_timing = processor.process(
                host,
                first_source_frame_ui=start_zero + 1,
            )
            timings["h2d_ms"] += stage_timing["h2d_ms"]
            timings["preprocessing_and_representation_ms"] += stage_timing[
                "preprocessing_and_representation_ms"
            ]
            del host
            if int(representation.shape[0]) == 0:
                continue
            moments, gamma_ms = timed_cuda(
                lambda: gamma_local_standardization(
                    representation,
                    zero_floor_reference,
                    chunk_frames=chunk_frames,
                    return_statistics=True,
                )
            )
            timings["gamma_moments_ms"] += gamma_ms
            if moments.local_mean is None or moments.local_std is None:
                raise AssertionError("full-recording Gamma-LS moments are required")

            for variant in CALIBRATION_VARIANTS:
                calibration_start, calibration_stop = variant.calibration_interval_ui
                calibration_mask = (source_ui >= calibration_start) & (
                    source_ui <= calibration_stop
                )
                if np.any(calibration_mask):
                    mask_device = torch.as_tensor(
                        calibration_mask, dtype=torch.bool, device=representation.device
                    )
                    bucket = collectors[variant.variant_id]
                    bucket["representation"].append(
                        representation[mask_device].detach().clone()
                    )
                    bucket["mean"].append(
                        moments.local_mean[mask_device].detach().clone()
                    )
                    bucket["std"].append(
                        moments.local_std[mask_device].detach().clone()
                    )
                    bucket["source_ui"].append(source_ui[calibration_mask].copy())

            for variant in CALIBRATION_VARIANTS:
                if (
                    stop_zero == variant.calibration_interval_ui[1]
                    and variant.variant_id not in operating_by_variant
                ):
                    bucket = collectors[variant.variant_id]
                    floor, rows, summary = _finalize_calibration(
                        variant=variant,
                        arm=arm,
                        representation_parts=bucket["representation"],
                        mean_parts=bucket["mean"],
                        std_parts=bucket["std"],
                        source_ui_parts=bucket["source_ui"],
                        scale_floor_percentile=scale_floor_percentile,
                        frame_interval_ms=frame_interval_ms,
                        configured_burst_durations=configured_burst_durations,
                    )
                    floors[variant.variant_id] = floor
                    operating_by_variant[variant.variant_id] = rows
                    threshold_rows.extend(rows)
                    calibration_summaries.append(summary)
                    collectors[variant.variant_id] = {
                        "representation": [],
                        "mean": [],
                        "std": [],
                        "source_ui": [],
                    }

            for variant in CALIBRATION_VARIANTS:
                application_start, application_stop = variant.application_interval_ui
                application_mask = (source_ui >= application_start) & (
                    source_ui <= application_stop
                )
                if not np.any(application_mask):
                    continue
                if variant.variant_id not in operating_by_variant:
                    raise RuntimeError(
                        f"application began before {variant.variant_id} was frozen"
                    )
                mask_device = torch.as_tensor(
                    application_mask,
                    dtype=torch.bool,
                    device=representation.device,
                )

                def score_and_transfer() -> np.ndarray:
                    floor_tensor = torch.as_tensor(
                        floors[variant.variant_id],
                        dtype=representation.dtype,
                        device=representation.device,
                    )
                    score = (
                        representation[mask_device]
                        - moments.local_mean[mask_device]
                    ) / (
                        torch.maximum(moments.local_std[mask_device], floor_tensor)
                        + GAMMA_EPSILON
                    )
                    return score.detach().cpu().numpy().astype(np.float32, copy=False)

                score_host, transfer_ms = timed_cuda(score_and_transfer)
                timings["score_and_d2h_ms"] += transfer_ms
                selected_ui = source_ui[application_mask]
                cpu_started = time.perf_counter()
                for score_frame, frame_ui in zip(score_host, selected_ui):
                    proposals, counts = extract_frame_proposals(
                        score_frame,
                        source_frame_ui=int(frame_ui),
                        variant_id=variant.variant_id,
                        representation=arm,
                        context_id=reference.context_id,
                        operating_points=operating_by_variant[variant.variant_id],
                        scale_floor_percentile=scale_floor_percentile,
                        scale_floor=floors[variant.variant_id],
                    )
                    candidate_rows.extend(proposals)
                    frame_count_rows.extend(counts)
                timings["cpu_exact_nms_ms"] += (
                    time.perf_counter() - cpu_started
                ) * 1000.0
                del score_host
            del moments, representation
            if progress is not None:
                progress(
                    {
                        "stage": "full_recording_causal_pass",
                        "completed_chunks": chunk_index,
                        "total_chunks": len(chunks),
                        "completed_source_frames": stop_zero,
                        "total_source_frames": TOTAL_FRAMES,
                        "proposal_rows_so_far": len(candidate_rows),
                    }
                )

    if set(operating_by_variant) != {
        variant.variant_id for variant in CALIBRATION_VARIANTS
    }:
        raise RuntimeError("not every calibration variant froze an operating grid")
    free_after, _ = torch.cuda.mem_get_info(device)
    candidate_rows.sort(key=_candidate_sort_key)
    frame_count_rows.sort(key=_frame_count_sort_key)
    threshold_rows.sort(
        key=lambda row: (
            [item.variant_id for item in CALIBRATION_VARIANTS].index(
                str(row["variant_id"])
            ),
            QUIET_NMS_PEAK_BURDENS.index(
                float(row["target_nms_peaks_per_calibration_unit"])
            ),
        )
    )
    return FullRecordingExecution(
        candidate_rows=tuple(candidate_rows),
        frame_count_rows=tuple(frame_count_rows),
        threshold_rows=tuple(threshold_rows),
        execution_summary={
            "causal_source_passes": 1,
            "causal_history_processed_ui": [1, TOTAL_FRAMES],
            "source_frames_processed": TOTAL_FRAMES,
            "source_chunk_frames_maximum": int(chunk_frames),
            "source_chunk_count": len(chunks),
            "ema_state_carried_across_all_chunks": True,
            "dense_scoring_device": str(device),
            "candidate_extraction_device": "cpu",
            "candidate_extraction_parity": (
                "exact maintained deterministic greedy Euclidean NMS on each "
                "transferred float32 score frame"
            ),
            "nms_distance_px": NMS_DISTANCE_PX,
            "timings": timings,
            "wall_seconds": time.perf_counter() - wall_started,
            "peak_vram_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "peak_vram_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
            "free_vram_bytes_before": int(free_before),
            "free_vram_bytes_after": int(free_after),
            "total_vram_bytes": int(total_vram),
            "calibrations": calibration_summaries,
        },
    )


def _validate_execution(
    execution: FullRecordingExecution,
    *,
    arm: str,
    context_id: str,
) -> dict[str, bool]:
    candidates = list(execution.candidate_rows)
    counts = list(execution.frame_count_rows)
    thresholds = list(execution.threshold_rows)
    variant_by_id = {item.variant_id: item for item in CALIBRATION_VARIANTS}
    expected_frame_rows = sum(
        (
            variant.application_interval_ui[1]
            - variant.application_interval_ui[0]
            + 1
        )
        * len(QUIET_NMS_PEAK_BURDENS)
        for variant in CALIBRATION_VARIANTS
    )
    expected_threshold_rows = len(CALIBRATION_VARIANTS) * len(
        QUIET_NMS_PEAK_BURDENS
    )
    candidate_group_counts: dict[tuple[str, float, int], int] = {}
    proposal_ids: set[str] = set()
    candidate_fields_exact = True
    candidate_values_valid = True
    for row in candidates:
        candidate_fields_exact &= set(row) == set(CANDIDATE_FIELDS)
        candidate_variant = variant_by_id.get(str(row["variant_id"]))
        key = (
            str(row["variant_id"]),
            float(row["target_nms_peaks_per_calibration_unit"]),
            int(row["source_frame_ui"]),
        )
        candidate_group_counts[key] = candidate_group_counts.get(key, 0) + 1
        proposal_id = str(row["proposal_id"])
        candidate_values_valid &= (
            str(row["representation"]) == arm
            and str(row["context_id"]) == context_id
            and str(row["biological_status"]) == "unknown_unreviewed_proposal"
            and row["temporal_linking_applied"] is False
            and math.isfinite(float(row["score"]))
            and int(row["candidate_rank_within_frame"]) >= 1
            and candidate_variant is not None
            and candidate_variant.application_interval_ui[0]
            <= int(row["source_frame_ui"])
            <= candidate_variant.application_interval_ui[1]
            and int(row["source_frame_ui"])
            > candidate_variant.calibration_interval_ui[1]
            and float(row["target_nms_peaks_per_calibration_unit"])
            in QUIET_NMS_PEAK_BURDENS
            and str(row["calibration_burden_unit"])
            == _burden_unit_for_variant(str(row["variant_id"]))
        )
        proposal_ids.add(proposal_id)
    count_sum = 0
    frame_rows_valid = True
    frame_keys: set[tuple[str, float, int]] = set()
    for row in counts:
        key = (
            str(row["variant_id"]),
            float(row["target_nms_peaks_per_calibration_unit"]),
            int(row["source_frame_ui"]),
        )
        frame_keys.add(key)
        variant = variant_by_id.get(key[0])
        frame_rows_valid &= (
            set(row) == set(FRAME_COUNT_FIELDS)
            and variant is not None
            and variant.application_interval_ui[0]
            <= key[2]
            <= variant.application_interval_ui[1]
            and key[1] in QUIET_NMS_PEAK_BURDENS
            and str(row["calibration_burden_unit"])
            == _burden_unit_for_variant(key[0])
            and int(row["proposal_count"]) >= 0
        )
        count_sum += int(row["proposal_count"])
        frame_rows_valid &= int(row["proposal_count"]) == candidate_group_counts.get(
            key, 0
        )
    expected_threshold_keys = {
        (variant.variant_id, float(target))
        for variant in CALIBRATION_VARIANTS
        for target in QUIET_NMS_PEAK_BURDENS
    }
    threshold_keys = {
        (
            str(row["variant_id"]),
            float(row["target_nms_peaks_per_calibration_unit"]),
        )
        for row in thresholds
    }
    threshold_rows_valid = True
    for row in thresholds:
        threshold_variant = variant_by_id.get(str(row["variant_id"]))
        threshold_rows_valid &= (
            set(row) == set(THRESHOLD_FIELDS)
            and threshold_variant is not None
            and float(row["target_nms_peaks_per_calibration_unit"])
            in QUIET_NMS_PEAK_BURDENS
            and int(row["nms_distance_px"]) == NMS_DISTANCE_PX
            and math.isfinite(float(row["threshold_z"]))
            and float(row["scale_floor"]) > 0
            and str(row["calibration_burden_unit"])
            == _burden_unit_for_variant(str(row["variant_id"]))
            and json.loads(str(row["calibration_interval_ui"]))
            == list(threshold_variant.calibration_interval_ui)
            and int(row["calibration_frame_first_ui"])
            == threshold_variant.calibration_interval_ui[0]
            and int(row["calibration_frame_last_ui"])
            == threshold_variant.calibration_interval_ui[1]
            and int(row["calibration_frame_count"]) == 100
            and bool(row["calibration_frame_locations_use_annotation_content"])
            is threshold_variant.calibration_frame_locations_use_annotation_content
            and bool(row["burst_window_supervised"])
            is threshold_variant.burst_window_supervised
            and bool(row["calibration_window_template_is_annotation_derived"])
            is (
                str(row["variant_id"])
                == CALIBRATION_VARIANTS[1].variant_id
            )
            and int(row["calibration_window_overlap_frames"])
            == (
                0
                if str(row["variant_id"])
                == CALIBRATION_VARIANTS[0].variant_id
                else 23
            )
        )
    checks = {
        "candidate_fields_exact": bool(candidate_fields_exact),
        "candidate_values_valid": bool(candidate_values_valid),
        "proposal_ids_unique": len(proposal_ids) == len(candidates),
        "frame_count_row_count_exact": len(counts) == expected_frame_rows,
        "frame_count_keys_unique": len(frame_keys) == len(counts),
        "frame_rows_valid": bool(frame_rows_valid),
        "candidate_and_frame_counts_reconcile": count_sum == len(candidates),
        "threshold_row_count_exact": len(thresholds) == expected_threshold_rows,
        "threshold_grid_complete_and_unique": threshold_keys
        == expected_threshold_keys,
        "threshold_rows_valid": bool(threshold_rows_valid),
        "candidate_rows_sorted": candidates
        == sorted(candidates, key=_candidate_sort_key),
        "frame_count_rows_sorted": counts
        == sorted(counts, key=_frame_count_sort_key),
        "annotation_files_not_opened": True,
        "temporal_linking_absent": all(
            row["temporal_linking_applied"] is False for row in candidates
        ),
    }
    if not all(checks.values()):
        failed = ", ".join(key for key, value in checks.items() if not value)
        raise RuntimeError(f"full-recording artifact validation failed: {failed}")
    return checks


def _proposal_summary(
    rows: Sequence[Mapping[str, Any]],
    frame_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for variant in CALIBRATION_VARIANTS:
        for target in QUIET_NMS_PEAK_BURDENS:
            selected = [
                row
                for row in frame_rows
                if row["variant_id"] == variant.variant_id
                and float(row["target_nms_peaks_per_calibration_unit"])
                == float(target)
            ]
            total = sum(int(row["proposal_count"]) for row in selected)
            result.append(
                {
                    "variant_id": variant.variant_id,
                    "target_nms_peaks_per_calibration_unit": float(target),
                    "calibration_burden_unit": _burden_unit_for_variant(
                        variant.variant_id
                    ),
                    "eligible_frame_count": len(selected),
                    "proposal_row_count": total,
                    "proposals_per_eligible_frame": total / len(selected),
                    "frames_with_at_least_one_proposal": sum(
                        int(row["proposal_count"]) > 0 for row in selected
                    ),
                    "biological_interpretation": "unknown_until_separate_review",
                }
            )
    if sum(int(row["proposal_row_count"]) for row in result) != len(rows):
        raise AssertionError("summary proposal counts do not close")
    return result


def run_full_recording_proposals(
    config: GammaLSDifferenceConfig,
    *,
    preflight_dir: str | Path,
    support_dir: str | Path,
    output_dir: str | Path,
    arm: str,
    context_id: str,
    device: str = "cuda:0",
    chunk_frames: int | None = None,
    _executor: Callable[..., FullRecordingExecution] | None = None,
) -> dict[str, Any]:
    """Run and atomically commit the annotation-file-free proposal tables."""

    if not isinstance(config, GammaLSDifferenceConfig):
        raise TypeError("config must be a validated GammaLSDifferenceConfig")
    if arm not in FULL_RECORDING_ARMS:
        raise ValueError(f"arm must be one of {FULL_RECORDING_ARMS}")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"full-recording output exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"full-recording output parent is missing: {destination.parent}"
        )

    # Read-only gates precede any output mutation.
    preflight = _verify_preflight_without_annotation_reads(config, preflight_dir)
    reference, support = _verify_support_context(support_dir, context_id)
    requested_chunk = int(
        chunk_frames
        if chunk_frames is not None
        else max(config.payload["efficiency"]["frame_chunks"])
    )
    if requested_chunk not in tuple(
        int(value) for value in config.payload["efficiency"]["frame_chunks"]
    ):
        raise ValueError("chunk_frames must be one of the frozen efficiency chunks")
    scale_floor_percentile = float(
        config.payload["gamma_ls_grid"]["finalist_scale_floor_percentiles"][0]
    )
    if tuple(
        float(value)
        for value in config.payload["cfar"]["quiet_nms_peaks_per_pseudo_burst"]
    ) != QUIET_NMS_PEAK_BURDENS:
        raise ValueError("configured empirical burden grid changed")
    if int(config.payload["cfar"]["nms_distance_px"]) != NMS_DISTANCE_PX:
        raise ValueError("configured primary NMS distance changed")
    frame_interval_ms = float(config.payload["frames"]["frame_interval_ms"])
    frames_per_second = int(round(1000.0 / frame_interval_ms))
    if frames_per_second != 50 or not math.isclose(
        frame_interval_ms * frames_per_second,
        1000.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("initial calibration requires the frozen 20-ms cadence")
    configured_burst_durations = {
        str(key): int(bounds[1]) - int(bounds[0]) + 1
        for key, bounds in config.payload["frames"]["burst_intervals_ui"].items()
    }
    try:
        runtime = require_cuda_device(device)
    except CudaRuntimeUnavailable as error:
        raise FullRecordingProposalUnavailable(str(error)) from error

    disk = shutil.disk_usage(destination.parent)
    minimum_disk = int(
        float(config.payload["resources"]["minimum_free_disk_gib"]) * 2**30
    )
    if disk.free < minimum_disk:
        raise FullRecordingProposalUnavailable(
            f"free disk {disk.free} is below the frozen minimum {minimum_disk} bytes"
        )

    contract = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "run_type": "causal_full_recording_frame_level_proposals",
        "portable_config_sha256": _canonical_sha256(config.portable_dict()),
        "preflight": preflight,
        "support_artifact": support,
        "movie_sha256": preflight["movie_sha256"],
        "executor_sha256": _sha256(Path(__file__).resolve()),
        "representation": arm,
        "context_id": context_id,
        "context": {
            "support_width_px": int(reference.support_width_px),
            "guard_radius_px": float(reference.guard_radius_px),
            "shape_n": float(reference.shape_n),
            "mode_radius_px": reference.nominal_mode_radius_px,
            "support_geometry": reference.support_geometry,
            "boundary_mode": reference.boundary_mode,
        },
        "calibration_variants": [item.as_dict() for item in CALIBRATION_VARIANTS],
        "scale_floor_percentile": scale_floor_percentile,
        "empirical_burdens": list(QUIET_NMS_PEAK_BURDENS),
        "burden_units": {
            CALIBRATION_VARIANTS[0].variant_id: INITIAL_BURDEN_UNIT,
            CALIBRATION_VARIANTS[1].variant_id: DECLARED_BURDEN_UNIT,
        },
        "initial_calibration_window_durations": {
            "initial_1s_block_1": frames_per_second,
            "initial_1s_block_2": frames_per_second,
        },
        "declared_comparison_window_durations": configured_burst_durations,
        "nms": "deterministic_greedy_euclidean_separated",
        "nms_distance_px": NMS_DISTANCE_PX,
        "chunk_frames": requested_chunk,
        "device": str(runtime["resolved_device"]),
        "annotation_sources_opened": False,
        "temporal_linking_applied": False,
    }
    work = destination.parent / (
        f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    )
    work.mkdir()

    def heartbeat(stage: str, **details: Any) -> None:
        _atomic_json(
            work / "heartbeat.json",
            {
                "status": "running",
                "stage": stage,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                **details,
            },
        )

    try:
        _atomic_json(work / "run_contract.json", contract)
        heartbeat("full_recording_causal_pass")
        executor = _executor or _execute_cuda_full_recording
        execution = executor(
            movie_path=config.source_paths["movie"],
            arm=arm,
            reference=reference,
            device=runtime["resolved_device"],
            chunk_frames=requested_chunk,
            scale_floor_percentile=scale_floor_percentile,
            frame_interval_ms=frame_interval_ms,
            configured_burst_durations=configured_burst_durations,
            progress=lambda row: heartbeat(
                str(row.get("stage", "full_recording_causal_pass")),
                **{key: value for key, value in row.items() if key != "stage"},
            ),
        )
        peak_vram = int(
            execution.execution_summary.get("peak_vram_allocated_bytes", 0)
        )
        max_vram = int(
            float(config.payload["resources"].get("max_peak_vram_gib", 8.0))
            * 2**30
        )
        if peak_vram > max_vram:
            raise FullRecordingProposalUnavailable(
                f"full-recording peak VRAM {peak_vram} exceeds cap {max_vram}"
            )
        checks = _validate_execution(
            execution,
            arm=arm,
            context_id=context_id,
        )
        checks["peak_vram_within_frozen_cap"] = peak_vram <= max_vram
        candidate_rows = list(execution.candidate_rows)
        frame_rows = list(execution.frame_count_rows)
        threshold_rows = list(execution.threshold_rows)
        summaries = _proposal_summary(candidate_rows, frame_rows)
        _atomic_tsv(
            work / "frame_level_proposals_label_sealed.tsv",
            candidate_rows,
            fieldnames=CANDIDATE_FIELDS,
        )
        _atomic_tsv(
            work / "proposal_counts_by_frame.tsv",
            frame_rows,
            fieldnames=FRAME_COUNT_FIELDS,
        )
        _atomic_tsv(
            work / "empirical_thresholds.tsv",
            threshold_rows,
            fieldnames=THRESHOLD_FIELDS,
        )
        _atomic_tsv(
            work / "proposal_summary.tsv",
            summaries,
            fieldnames=tuple(summaries[0]),
        )
        content_hash = candidate_content_sha256(candidate_rows)
        seal = {
            "schema_version": 1,
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "annotation_sources_opened_before_seal": False,
            "annotation_sources_opened_after_seal": False,
            "annotation_join_performed": False,
            "candidate_table": {
                "path": "frame_level_proposals_label_sealed.tsv",
                "rows": len(candidate_rows),
                "file_sha256": _sha256(
                    work / "frame_level_proposals_label_sealed.tsv"
                ),
                "canonical_content_sha256": content_hash,
            },
            "frame_count_table": {
                "path": "proposal_counts_by_frame.tsv",
                "rows": len(frame_rows),
                "file_sha256": _sha256(work / "proposal_counts_by_frame.tsv"),
            },
            "threshold_table": {
                "path": "empirical_thresholds.tsv",
                "rows": len(threshold_rows),
                "file_sha256": _sha256(work / "empirical_thresholds.tsv"),
            },
            "biological_status_of_every_proposal": "unknown_until_separate_review",
            "temporal_linking_applied": False,
            "unique_event_count_reported": False,
        }
        _atomic_json(work / "candidate_seal.json", seal)
        claim_boundary = {
            "result_unit": "frame_level_automated_proposal_row",
            "automated_proposal_counts_identified": True,
            "biological_detection_count_identified": False,
            "precision_identified": False,
            "temporal_linking_applied": False,
            "unique_event_count_identified": False,
            "initial_100_calibration_uses_annotation_files": False,
            "initial_100_calibration_uses_annotated_frame_locations": False,
            "initial_100_calibration_window_duration_source": (
                "declared_frame_interval_only"
            ),
            "initial_100_result_role": "operational_deployment_count",
            "initial_100_frames_assumed_event_free": False,
            "declared_quiet_variant_uses_human_interval": True,
            "declared_quiet_variant_uses_configured_burst_durations": True,
            "declared_quiet_result_role": "paper_comparable_supervised_sensitivity",
            "fixed_context_support_screen_used_burst_windows": True,
            "end_to_end_hyperparameter_selection_is_annotation_free": False,
            "annotation_sources_opened_by_this_runner": False,
            "scientific_audit_complete": False,
            "paper_promotion_ready": False,
        }
        _atomic_json(work / "claim_boundary.json", claim_boundary)
        summary = {
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "run_type": "causal_full_recording_frame_level_proposals",
            "status": "complete_proposal_ledger_scientific_audit_pending",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "representation": arm,
            "context_id": context_id,
            "source_shape_tyx": [TOTAL_FRAMES, FRAME_HEIGHT, FRAME_WIDTH],
            "proposal_row_count": len(candidate_rows),
            "proposal_counts": summaries,
            "candidate_seal": seal,
            "execution": dict(execution.execution_summary),
            "runtime": runtime,
            "claim_boundary": claim_boundary,
            "scientific_audit": {
                "status": "pending",
                "expert_section": "not_applicable_for_annotation_file_free_run",
                "model_section": "required_candidate_surrogate_media_not_yet_rendered",
                "comparison_section": "not_applicable_without_annotation_join",
                "paper_promotion_allowed": False,
            },
        }
        _atomic_json(work / "summary.json", summary)
        validation = {
            "status": "passed_proposal_ledger_artifact_contract_scientific_audit_pending",
            "checks": checks,
            "all_checks_pass": all(checks.values()),
        }
        _atomic_json(work / "validation.json", validation)
        _atomic_json(
            work / "llm_context.json",
            {
                "entrypoint": "summary.json",
                "grain": (
                    "calibration variant x empirical burden x source UI frame x "
                    "frame-local proposal rank"
                ),
                "primary_tables": {
                    "proposals": "frame_level_proposals_label_sealed.tsv",
                    "per_frame_counts": "proposal_counts_by_frame.tsv",
                    "thresholds": "empirical_thresholds.tsv",
                    "summary": "proposal_summary.tsv",
                },
                "candidate_seal": "candidate_seal.json",
                "coordinate_convention": "x_column_y_row",
                "frame_convention": "UI_one_based_inclusive",
                "calibration_burden_units": {
                    CALIBRATION_VARIANTS[0].variant_id: INITIAL_BURDEN_UNIT,
                    CALIBRATION_VARIANTS[1].variant_id: DECLARED_BURDEN_UNIT,
                },
                "stage_sequence": [
                    "uint16 acquired frame",
                    "Gaussian sigma 1 reflect",
                    "causal EMA alpha 0.4",
                    arm,
                    "signed radial Gamma-LS",
                    "empirical strict score threshold",
                    "deterministic greedy Euclidean NMS radius 6",
                    "frame-level proposal row",
                ],
                "limitations": [
                    "proposal rows do not establish biological identity",
                    "no temporal linking or unique-event count was performed",
                    "the first 100 initialization frames were not assumed event-free",
                    "the fixed context originated in a burst-window-supervised support screen",
                    "scientific-audit candidate-surrogate media remain pending",
                ],
            },
        )
        report_lines = [
            "# Full-recording Gamma-LS automated proposal ledger",
            "",
            f"Status: `{summary['status']}`.",
            "",
            f"Fixed pipeline: `{arm}` -> `{context_id}` -> empirical threshold -> radius-6 NMS.",
            "",
            "The complete 2,359-frame acquisition was processed causally from UI frame 1. "
            "The first calibration variant used UI frames 1--100 without assuming that they "
            "were event-free and applied only from UI frame 101. The comparison variant used "
            "the human-declared UI 1800--1899 interval and applied only from UI frame 1900.",
            "The first variant is the operational deployment count: it calibrates on two "
            "nonoverlapping one-second blocks derived only from the 20-ms cadence. The second "
            "is the paper-comparable supervised sensitivity: it uses the configured burst-duration "
            "template, with any pseudo-window overlap reported in the threshold table.",
            "",
            "## Automated proposal counts",
            "",
            "| Variant | Calibration burden unit | Target burden | Eligible frames | "
            "Proposal rows | Proposals/frame |",
            "|---|---|---:|---:|---:|---:|",
        ]
        report_lines.extend(
            "| {variant_id} | {calibration_burden_unit} | "
            "{target_nms_peaks_per_calibration_unit:g} | "
            "{eligible_frame_count} | {proposal_row_count} | "
            "{proposals_per_eligible_frame:.6g} |".format(**row)
            for row in summaries
        )
        report_lines.extend(
            [
                "",
                "Every row is an automated, frame-local proposal with unknown biological "
                "status. No annotation table was opened, no temporal linking was applied, "
                "and no unique biological event count is reported.",
                "",
                "The proposal ledger passed its count/hash/boundary contract. Scientific-audit "
                "candidate-surrogate media are still required before paper promotion.",
                "",
            ]
        )
        _atomic_text(work / "REPORT.md", "\n".join(report_lines))
        _atomic_json(
            work / "status.json",
            {
                "status": summary["status"],
                "completed_at_utc": summary["completed_at_utc"],
                "candidate_seal_sha256": _sha256(work / "candidate_seal.json"),
                "scientific_audit_complete": False,
            },
        )
        _atomic_json(work / "artifact_index.json", _artifact_index(work))
        if destination.exists():
            raise FileExistsError(
                f"full-recording output appeared during execution: {destination}"
            )
        work.replace(destination)
        return summary
    except Exception as error:
        if work.exists():
            _atomic_json(
                work / "status.json",
                {
                    "status": "interrupted_or_failed_recoverable",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": repr(error),
                    "requested_output_absent": not destination.exists(),
                },
            )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the annotation-file-free full-recording Gamma-LS proposal ledger."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--support-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arm", required=True, choices=FULL_RECORDING_ARMS)
    parser.add_argument("--context-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-frames", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run_full_recording_proposals(
            GammaLSDifferenceConfig.load(args.config),
            preflight_dir=args.preflight_dir,
            support_dir=args.support_dir,
            output_dir=args.output_dir,
            arm=args.arm,
            context_id=args.context_id,
            device=args.device,
            chunk_frames=args.chunk_frames,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "blocked_full_recording_proposal_run",
                    "error": repr(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())


__all__ = [
    "CALIBRATION_VARIANTS",
    "CANDIDATE_FIELDS",
    "DECLARED_BURDEN_UNIT",
    "FRAME_COUNT_FIELDS",
    "FULL_RECORDING_ARMS",
    "FullRecordingExecution",
    "FullRecordingProposalUnavailable",
    "INITIAL_BURDEN_UNIT",
    "THRESHOLD_FIELDS",
    "candidate_content_sha256",
    "expected_calibration_frame_count",
    "extract_frame_proposals",
    "fit_empirical_calibration_thresholds",
    "run_full_recording_proposals",
    "verify_indexed_artifact",
]
