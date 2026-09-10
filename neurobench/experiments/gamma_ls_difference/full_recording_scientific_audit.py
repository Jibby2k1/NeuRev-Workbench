"""Exact model-only scientific audit for the frozen full-record q=1 ledger.

This module never changes the already sealed proposal metric.  It replays the
operational ``initial_100`` lane solely to persist synchronized causal stage
arrays, requires exact row-for-row reconciliation with the frozen ledger, and
then renders the candidate-surrogate media required by the repository's
scientific-audit standard.  No annotation source is opened.
"""
from __future__ import annotations

import argparse
import ast
import csv
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from neurobench.algorithms.gamma_local_standardization import (
    GammaReferenceSpec,
    gamma_local_standardization,
)

from . import gpu_representations
from .config import GammaLSDifferenceConfig
from .cuda_runtime import CudaRuntimeUnavailable, require_cuda_device
from .evaluation import NMS_DISTANCE_PX, QUIET_NMS_PEAK_BURDENS
from .full_recording import (
    CALIBRATION_VARIANTS,
    CANDIDATE_FIELDS,
    FRAME_COUNT_FIELDS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GAMMA_EPSILON,
    INITIAL_BURDEN_UNIT,
    THRESHOLD_FIELDS,
    TOTAL_FRAMES,
    _CausalCudaArm,
    _aligned_chunks,
    _canonical_sha256,
    _candidate_sort_key,
    _finalize_calibration,
    _frame_count_sort_key,
    extract_frame_proposals,
    verify_indexed_artifact,
)
from .scientific_audit import (
    ORANGE,
    _VideoWriter,
    _decode_marker_frame,
    _gray_signed,
    _gray_unsigned,
    _marker_counts,
    _probe_video,
)


EXPERIMENT_ID = "spon_ca_burst_gamma_ls_full_recording_q1_scientific_audit_v1"
LEDGER_EXPERIMENT_ID = "spon_ca_burst_gamma_ls_difference_ablation_v1"
VARIANT_ID = "initial_100_annotation_file_and_location_free"
REPRESENTATION = "difference_signed"
CONTEXT_ID = "support_support_a_h15_g7_n9_m0p5"
TARGET_BURDEN = 1.0
CALIBRATION_UI = (1, 100)
APPLICATION_UI = (101, 2359)
EXPECTED_ELIGIBLE_FRAMES = 2259
EXPECTED_PROPOSALS = 371
EXPECTED_PROPOSAL_FRAMES = 295
PINNED_LEDGER_ROOT_NAME = (
    "spon_ca_burst_gamma_ls_full_recording_signed_h15_v1_20260908_r2"
)
PINNED_LEDGER_HASHES = {
    "ledger_artifact_index_sha256": (
        "cc1cdfa70732204cf2c8e73e72f90b8cf5f06acf2193fa56218be6c4e9a6bced"
    ),
    "candidate_table_sha256": (
        "ad7654a25d682516ded876a85adbfdbf8a78cc5a09a0de7391ffb03ca18c9f29"
    ),
    "frame_count_table_sha256": (
        "76112b091256fdf56c88eec5a865d56a7fecdf8e7354fa03b62cc586264ae94b"
    ),
    "threshold_table_sha256": (
        "ba8d667ae246dd7b8339601011c4093b188976be394fdd443e5ba43076159ec8"
    ),
    "initial_candidates_content_sha256": (
        "3d61bc4429b81e7864d175706bdbca3d65aee3fd9ca5f86d8cb5773a57816f74"
    ),
    "q1_candidates_content_sha256": (
        "8d575ccc26d68032a55fb591bd8f296fc5dce43a5b7ec488103a47b3bfe7a19c"
    ),
    "initial_counts_content_sha256": (
        "fba1b58c1c9ae8f267d55e5f3fe00ac93c5f74cc439a3f71246a31c29fe69f85"
    ),
    "q1_counts_content_sha256": (
        "262f1fdf0a63f92a8890ffba57c4883a945aa64f6c8e4b8ce9664fe5e97825e4"
    ),
    "initial_thresholds_content_sha256": (
        "207f8f29636fa2bb027fd6c2edf27a29b8f04fd32fe1b7728da3661f26f4898d"
    ),
}
PINNED_LEDGER_VALIDATION_SHA256 = (
    "501e770087343f568e817495a917683b59a3231f2000344218e5cbc479918686"
)
PINNED_LEDGER_CONTRACT_SHA256 = (
    "b4b20c2c8acc2737a316293c4277b659eaddd020bfdbbea58e7021facf1deaa6"
)
PINNED_Q1_THRESHOLD = 7.080160140991211
PINNED_SCALE_FLOOR = 3.8241920471191406
EXPECTED_CONTEXT = {
    "support_width_px": 31,
    "guard_radius_px": 7.0,
    "shape_n": 9.0,
    "mode_radius_px": 7.5,
    "support_geometry": "disk",
    "boundary_mode": "valid_renormalized_zero",
}
DEFAULT_CHUNK_FRAMES = 64
DEFAULT_ANCHOR_LIMIT = 24
CONSOLIDATION_RADIUS_PX = 3.0
STAGE_FILES = {
    "conditioned": "stage_arrays/conditioned_current_frame.npy",
    "difference": "stage_arrays/difference_signed.npy",
    "gamma": "stage_arrays/gamma_initial100_scale.npy",
    "threshold": "stage_arrays/threshold_exceedance_q1_application.npy",
}
KNOWN_ALL_TRUE_GATE_RECOVERY = {
    "replay_output_name": (
        "spon_ca_burst_gamma_ls_full_recording_q1_stage_replay_v1_20260909_r1"
    ),
    "preflight_root_name": (
        "spon_ca_burst_gamma_ls_full_recording_q1_scientific_audit_preflight_"
        "v1_20260909_r1"
    ),
    "preflight_artifact_index_sha256": (
        "90e3c1b84345707cf1b8957616fd59d03229c0994b1d119bf37cde1c1389d1f8"
    ),
    "preflight_json_sha256": (
        "e0080a1e0ff9b2d80dc5498186a1d97b51271fe47ee54c8748b1e7d737991dfc"
    ),
    "implementation_sha256": (
        "35a05e1b2678522e148f26c6088fddc2ebbbbe2ebb5d857ca211909254741a41"
    ),
    "test_sha256": (
        "a68091b24f3d3bf095f188c5ea9b96f363b6dea659cdff0b8352244b8c4ac3f6"
    ),
    "run_contract_sha256": (
        "08860bea4d5015fd8e1968d15ebe53b777c1c0c633376b5c8dde7395b78018d6"
    ),
    "replay_checkpoint_sha256": (
        "c935cf4a5a24befdc2b0ba58be4fa27c581d41db3187e38eee9166ad18bd434d"
    ),
    "failure_sha256": (
        "e3775280d82b5f5c5d726d5011fb14c1a6746be3e3fa0363f5e70191034170f1"
    ),
}


class FullRecordScientificAuditUnavailable(RuntimeError):
    """Raised when any frozen input, replay, or media invariant fails."""


@dataclass
class _ReplayFailureRecorder:
    work: Path

    def __enter__(self) -> "_ReplayFailureRecorder":
        return self

    def __exit__(self, error_type: Any, error: Any, traceback: Any) -> bool:
        del traceback
        if error_type is not None:
            _atomic_json(
                self.work / "failure.json",
                {
                    "status": "replay_failed_checkpoint_retained",
                    "failed_at_utc": _utc_now(),
                    "error_type": error_type.__name__,
                    "error": str(error),
                    "resume_command_required": True,
                },
            )
            _atomic_json(
                self.work / "status.json",
                {
                    "status": "replay_failed_checkpoint_retained",
                    "scientific_audit_complete": False,
                },
            )
        return False


def _record_replay_failure(function: Any) -> Any:
    """Record failures occurring before, during, or after the CUDA loop."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            replay_output = kwargs.get("replay_output")
            if replay_output is not None:
                destination = Path(replay_output).expanduser().resolve()
                work = destination.parent / f".{destination.name}.replay-work"
                if work.is_dir():
                    checkpoint_exists = (work / "replay_checkpoint.json").is_file()
                    _atomic_json(
                        work / "failure.json",
                        {
                            "status": "replay_failed_checkpoint_retained",
                            "failed_at_utc": _utc_now(),
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "checkpoint_exists": checkpoint_exists,
                            "recovery": (
                                "resume_from_verified_checkpoint"
                                if checkpoint_exists
                                else "resume_restarts_short_precalibration_prefix_at_ui1"
                            ),
                        },
                    )
                    _atomic_json(
                        work / "status.json",
                        {
                            "status": "replay_failed_checkpoint_retained",
                            "scientific_audit_complete": False,
                            "checkpoint_exists": checkpoint_exists,
                        },
                    )
            raise

    wrapped.__name__ = str(getattr(function, "__name__", "wrapped"))
    wrapped.__doc__ = getattr(function, "__doc__", None)
    return wrapped


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FullRecordScientificAuditUnavailable(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _atomic_table(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
    delimiter: str,
) -> None:
    fields = list(fieldnames)
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("table fields must be nonempty and unique")
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f"table fields changed for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter=delimiter, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_tsv(
    path: Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str]
) -> None:
    _atomic_table(path, rows, fieldnames=fieldnames, delimiter="\t")


def _atomic_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str]
) -> None:
    _atomic_table(path, rows, fieldnames=fieldnames, delimiter=",")


def _artifact_index(root: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifacts": [
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.name != "artifact_index.json"
            and not path.name.endswith(".partial")
        ],
    }


def _verify_complete_index(root: str | Path) -> dict[str, Any]:
    directory = Path(root).expanduser().resolve()
    result = verify_indexed_artifact(directory)
    index = _read_json(directory / "artifact_index.json")
    indexed = {str(row["path"]) for row in index["artifacts"]}
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
        and path.name != "artifact_index.json"
        and not path.name.endswith(".partial")
    }
    if indexed != actual:
        raise FullRecordScientificAuditUnavailable(
            f"artifact index coverage changed: missing={sorted(indexed-actual)}, "
            f"unindexed={sorted(actual-indexed)}"
        )
    return {**result, "unindexed_file_count": 0}


def _bool(value: Any) -> bool:
    if value is True or str(value).strip().lower() == "true":
        return True
    if value is False or str(value).strip().lower() == "false":
        return False
    raise ValueError(f"not a boolean value: {value!r}")


def _normalize_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "proposal_id": str(row["proposal_id"]),
        "variant_id": str(row["variant_id"]),
        "representation": str(row["representation"]),
        "context_id": str(row["context_id"]),
        "target_nms_peaks_per_calibration_unit": float(
            row["target_nms_peaks_per_calibration_unit"]
        ),
        "calibration_burden_unit": str(row["calibration_burden_unit"]),
        "threshold_z": float(row["threshold_z"]),
        "scale_floor_percentile": float(row["scale_floor_percentile"]),
        "scale_floor": float(row["scale_floor"]),
        "source_frame_ui": int(row["source_frame_ui"]),
        "candidate_rank_within_frame": int(row["candidate_rank_within_frame"]),
        "score": float(row["score"]),
        "x_px": int(row["x_px"]),
        "y_px": int(row["y_px"]),
        "biological_status": str(row["biological_status"]),
        "temporal_linking_applied": _bool(row["temporal_linking_applied"]),
    }


def _normalize_count(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "variant_id": str(row["variant_id"]),
        "representation": str(row["representation"]),
        "context_id": str(row["context_id"]),
        "target_nms_peaks_per_calibration_unit": float(
            row["target_nms_peaks_per_calibration_unit"]
        ),
        "calibration_burden_unit": str(row["calibration_burden_unit"]),
        "threshold_z": float(row["threshold_z"]),
        "source_frame_ui": int(row["source_frame_ui"]),
        "proposal_count": int(row["proposal_count"]),
    }


def _normalize_threshold(row: Mapping[str, Any]) -> dict[str, Any]:
    def structured(value: Any) -> Any:
        if isinstance(value, (list, dict)):
            return value
        encoded = str(value)
        try:
            parsed = json.loads(encoded)
        except json.JSONDecodeError:
            # ``csv.DictWriter`` stringifies in-memory containers with Python
            # repr syntax.  Accept that exact sealed-replay representation
            # without permitting arbitrary evaluation or scalar values.
            parsed = ast.literal_eval(encoded)
        if not isinstance(parsed, (list, dict)):
            raise ValueError("structured threshold field must be a list or mapping")
        return parsed

    return {
        "variant_id": str(row["variant_id"]),
        "calibration_interval_ui": structured(row["calibration_interval_ui"]),
        "calibration_frame_first_ui": int(row["calibration_frame_first_ui"]),
        "calibration_frame_last_ui": int(row["calibration_frame_last_ui"]),
        "calibration_frame_count": int(row["calibration_frame_count"]),
        "representation_alignment_note": str(row["representation_alignment_note"]),
        "calibration_role": str(row["calibration_role"]),
        "calibration_frame_locations_use_annotation_content": _bool(
            row["calibration_frame_locations_use_annotation_content"]
        ),
        "calibration_window_template_ui_frames": structured(
            row["calibration_window_template_ui_frames"]
        ),
        "calibration_window_template_source": str(
            row["calibration_window_template_source"]
        ),
        "calibration_window_template_is_annotation_derived": _bool(
            row["calibration_window_template_is_annotation_derived"]
        ),
        "burst_window_supervised": _bool(row["burst_window_supervised"]),
        "scale_floor_percentile": float(row["scale_floor_percentile"]),
        "scale_floor": float(row["scale_floor"]),
        "target_nms_peaks_per_calibration_unit": float(
            row["target_nms_peaks_per_calibration_unit"]
        ),
        "calibration_burden_unit": str(row["calibration_burden_unit"]),
        "threshold_z": float(row["threshold_z"]),
        "achieved_nms_peaks_per_calibration_unit": float(
            row["achieved_nms_peaks_per_calibration_unit"]
        ),
        "total_calibration_nms_peaks": int(row["total_calibration_nms_peaks"]),
        "calibration_windows_source_ui": structured(
            row["calibration_windows_source_ui"]
        ),
        "calibration_window_overlap_frames": int(
            row["calibration_window_overlap_frames"]
        ),
        "nms_distance_px": int(row["nms_distance_px"]),
        "probability_model_claimed": _bool(row["probability_model_claimed"]),
    }


def _normalize_surrogate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_roi_id": str(row["model_roi_id"]),
        "x_px": int(row["x_px"]),
        "y_px": int(row["y_px"]),
        "proposal_occurrence_count": int(row["proposal_occurrence_count"]),
        "distinct_proposal_frame_count": int(row["distinct_proposal_frame_count"]),
        "first_source_frame_ui": int(row["first_source_frame_ui"]),
        "last_source_frame_ui": int(row["last_source_frame_ui"]),
        "best_frame_rank": int(row["best_frame_rank"]),
        "maximum_score": float(row["maximum_score"]),
        "selection_rule": str(row["selection_rule"]),
    }


def _normalize_assignment(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "proposal_id": str(row["proposal_id"]),
        "model_roi_id": str(row["model_roi_id"]),
        "selected_for_closeup": _bool(row["selected_for_closeup"]),
    }


def _subset_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_sha256([dict(row) for row in rows])


def _build_candidate_surrogates(
    q1_rows: Sequence[Mapping[str, Any]], *, limit: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Freeze a label-free recurrent spatial surrogate panel."""

    if limit < 1:
        raise ValueError("surrogate limit must be positive")
    ordered = sorted(
        (_normalize_candidate(row) for row in q1_rows),
        key=lambda row: (
            row["source_frame_ui"],
            row["candidate_rank_within_frame"],
            row["proposal_id"],
        ),
    )
    clusters: list[dict[str, Any]] = []
    assignment_by_id: dict[str, str] = {}
    squared_radius = CONSOLIDATION_RADIUS_PX**2
    for candidate in ordered:
        x_px, y_px = float(candidate["x_px"]), float(candidate["y_px"])
        cluster = next(
            (
                row
                for row in clusters
                if (float(row["x_px"]) - x_px) ** 2
                + (float(row["y_px"]) - y_px) ** 2
                <= squared_radius
            ),
            None,
        )
        if cluster is None:
            cluster = {
                "model_roi_id": f"model_roi_{len(clusters) + 1:04d}",
                "x_px": x_px,
                "y_px": y_px,
                "proposal_ids": [],
                "source_frames_ui": [],
                "best_frame_rank": int(candidate["candidate_rank_within_frame"]),
                "maximum_score": float(candidate["score"]),
            }
            clusters.append(cluster)
        cluster["proposal_ids"].append(candidate["proposal_id"])
        cluster["source_frames_ui"].append(int(candidate["source_frame_ui"]))
        cluster["best_frame_rank"] = min(
            int(cluster["best_frame_rank"]),
            int(candidate["candidate_rank_within_frame"]),
        )
        cluster["maximum_score"] = max(
            float(cluster["maximum_score"]), float(candidate["score"])
        )
        assignment_by_id[candidate["proposal_id"]] = cluster["model_roi_id"]
    ranked = sorted(
        clusters,
        key=lambda row: (
            -len(set(row["source_frames_ui"])),
            int(row["best_frame_rank"]),
            -float(row["maximum_score"]),
            str(row["model_roi_id"]),
        ),
    )
    selected = ranked[:limit]
    selected_ids = {str(row["model_roi_id"]) for row in selected}
    surrogate_rows = [
        {
            "model_roi_id": str(row["model_roi_id"]),
            "x_px": int(round(float(row["x_px"]))),
            "y_px": int(round(float(row["y_px"]))),
            "proposal_occurrence_count": len(row["proposal_ids"]),
            "distinct_proposal_frame_count": len(set(row["source_frames_ui"])),
            "first_source_frame_ui": min(row["source_frames_ui"]),
            "last_source_frame_ui": max(row["source_frames_ui"]),
            "best_frame_rank": int(row["best_frame_rank"]),
            "maximum_score": float(row["maximum_score"]),
            "selection_rule": (
                "greedy_3px_spatial_consolidation_then_recurrence_rank_score_id"
            ),
        }
        for row in selected
    ]
    assignments = [
        {
            "proposal_id": row["proposal_id"],
            "model_roi_id": assignment_by_id[row["proposal_id"]],
            "selected_for_closeup": assignment_by_id[row["proposal_id"]]
            in selected_ids,
        }
        for row in ordered
    ]
    return surrogate_rows, assignments, len(clusters)


def _load_operational_ledger(root: str | Path) -> dict[str, Any]:
    directory = Path(root).expanduser().resolve()
    if directory.name != PINNED_LEDGER_ROOT_NAME:
        raise FullRecordScientificAuditUnavailable(
            "ledger is not the authoritative sealed r2 artifact"
        )
    indexed = _verify_complete_index(directory)
    status = _read_json(directory / "status.json")
    validation = _read_json(directory / "validation.json")
    summary = _read_json(directory / "summary.json")
    contract = _read_json(directory / "run_contract.json")
    seal = _read_json(directory / "candidate_seal.json")
    if status.get("status") != "complete_proposal_ledger_scientific_audit_pending":
        raise FullRecordScientificAuditUnavailable("ledger status changed")
    if not validation.get("all_checks_pass"):
        raise FullRecordScientificAuditUnavailable("ledger validation is not passing")
    if (
        summary.get("experiment_id") != LEDGER_EXPERIMENT_ID
        or summary.get("representation") != REPRESENTATION
        or summary.get("context_id") != CONTEXT_ID
        or contract.get("representation") != REPRESENTATION
        or contract.get("context_id") != CONTEXT_ID
    ):
        raise FullRecordScientificAuditUnavailable("ledger pipeline identity changed")
    if contract.get("annotation_sources_opened") is not False:
        raise FullRecordScientificAuditUnavailable("ledger annotation boundary changed")
    if contract.get("temporal_linking_applied") is not False:
        raise FullRecordScientificAuditUnavailable("ledger temporal-link boundary changed")
    if contract.get("context") != EXPECTED_CONTEXT:
        raise FullRecordScientificAuditUnavailable("ledger Gamma-LS context changed")
    variants = {str(row["variant_id"]): row for row in contract["calibration_variants"]}
    initial = variants.get(VARIANT_ID)
    if not isinstance(initial, Mapping) or (
        list(initial.get("calibration_interval_ui", [])) != list(CALIBRATION_UI)
        or list(initial.get("application_interval_ui", [])) != list(APPLICATION_UI)
        or initial.get("calibration_role")
        != "initialization_frames_not_assumed_event_free"
        or initial.get("calibration_frame_locations_use_annotation_content") is not False
        or initial.get("burst_window_supervised") is not False
    ):
        raise FullRecordScientificAuditUnavailable(
            "initial-100 calibration/application contract changed"
        )
    candidate_rows = [
        _normalize_candidate(row)
        for row in _read_tsv(directory / "frame_level_proposals_label_sealed.tsv")
    ]
    frame_rows = [
        _normalize_count(row)
        for row in _read_tsv(directory / "proposal_counts_by_frame.tsv")
    ]
    threshold_rows = [
        _normalize_threshold(row)
        for row in _read_tsv(directory / "empirical_thresholds.tsv")
    ]
    initial_candidates = [row for row in candidate_rows if row["variant_id"] == VARIANT_ID]
    q1_candidates = [
        row
        for row in initial_candidates
        if row["target_nms_peaks_per_calibration_unit"] == TARGET_BURDEN
    ]
    initial_counts = [row for row in frame_rows if row["variant_id"] == VARIANT_ID]
    q1_counts = [
        row
        for row in initial_counts
        if row["target_nms_peaks_per_calibration_unit"] == TARGET_BURDEN
    ]
    initial_thresholds = [
        row for row in threshold_rows if row["variant_id"] == VARIANT_ID
    ]
    q1_thresholds = [
        row
        for row in initial_thresholds
        if row["target_nms_peaks_per_calibration_unit"] == TARGET_BURDEN
    ]
    proposal_frames = {
        int(row["source_frame_ui"]) for row in q1_candidates
    }
    checks = {
        "q1_proposal_rows_exact": len(q1_candidates) == EXPECTED_PROPOSALS,
        "q1_proposal_frames_exact": len(proposal_frames)
        == EXPECTED_PROPOSAL_FRAMES,
        "q1_count_rows_exact": len(q1_counts) == EXPECTED_ELIGIBLE_FRAMES,
        "q1_count_interval_exact": [
            min(int(row["source_frame_ui"]) for row in q1_counts),
            max(int(row["source_frame_ui"]) for row in q1_counts),
        ]
        == list(APPLICATION_UI),
        "q1_counts_reconcile": sum(int(row["proposal_count"]) for row in q1_counts)
        == EXPECTED_PROPOSALS,
        "q1_threshold_unique": len(q1_thresholds) == 1,
        "initial_threshold_grid_exact": [
            row["target_nms_peaks_per_calibration_unit"]
            for row in initial_thresholds
        ]
        == list(QUIET_NMS_PEAK_BURDENS),
        "q1_border_and_nms_exact": len(q1_thresholds) == 1
        and q1_thresholds[0]["nms_distance_px"] == NMS_DISTANCE_PX,
        "proposal_ids_unique": len({row["proposal_id"] for row in q1_candidates})
        == len(q1_candidates),
        "all_candidates_unknown": all(
            row["biological_status"] == "unknown_unreviewed_proposal"
            and row["temporal_linking_applied"] is False
            for row in q1_candidates
        ),
        "seal_has_no_annotation_join": seal.get("annotation_join_performed") is False,
    }
    if not all(checks.values()):
        raise FullRecordScientificAuditUnavailable(
            "operational ledger checks failed: "
            + ", ".join(key for key, value in checks.items() if not value)
        )
    q1_threshold = q1_thresholds[0]
    hashes = {
        "ledger_artifact_index_sha256": indexed["artifact_index_sha256"],
        "candidate_table_sha256": _sha256(
            directory / "frame_level_proposals_label_sealed.tsv"
        ),
        "frame_count_table_sha256": _sha256(
            directory / "proposal_counts_by_frame.tsv"
        ),
        "threshold_table_sha256": _sha256(directory / "empirical_thresholds.tsv"),
        "initial_candidates_content_sha256": _subset_hash(initial_candidates),
        "q1_candidates_content_sha256": _subset_hash(q1_candidates),
        "initial_counts_content_sha256": _subset_hash(initial_counts),
        "q1_counts_content_sha256": _subset_hash(q1_counts),
        "initial_thresholds_content_sha256": _subset_hash(initial_thresholds),
    }
    if hashes != PINNED_LEDGER_HASHES:
        raise FullRecordScientificAuditUnavailable(
            "authoritative sealed-r2 ledger hashes changed"
        )
    if _sha256(directory / "validation.json") != PINNED_LEDGER_VALIDATION_SHA256:
        raise FullRecordScientificAuditUnavailable("sealed-r2 validation hash changed")
    if _sha256(directory / "run_contract.json") != PINNED_LEDGER_CONTRACT_SHA256:
        raise FullRecordScientificAuditUnavailable("sealed-r2 contract hash changed")
    if (
        float(q1_threshold["threshold_z"]) != PINNED_Q1_THRESHOLD
        or float(q1_threshold["scale_floor"]) != PINNED_SCALE_FLOOR
    ):
        raise FullRecordScientificAuditUnavailable(
            "sealed-r2 q1 threshold or scale floor changed"
        )
    return {
        "root": directory,
        "indexed": indexed,
        "summary": summary,
        "contract": contract,
        "seal": seal,
        "initial_candidates": initial_candidates,
        "q1_candidates": q1_candidates,
        "initial_counts": initial_counts,
        "q1_counts": q1_counts,
        "initial_thresholds": initial_thresholds,
        "q1_threshold": q1_threshold,
        "checks": checks,
        "hashes": hashes,
    }


def _implementation_files(repository: Path) -> dict[str, dict[str, Any]]:
    relatives = (
        "neurobench/algorithms/gamma_local_standardization.py",
        "neurobench/metrics/sparse_detection.py",
        "neurobench/experiments/gamma_ls_difference/config.py",
        "neurobench/experiments/gamma_ls_difference/cuda_runtime.py",
        "neurobench/experiments/gamma_ls_difference/evaluation.py",
        "neurobench/experiments/gamma_ls_difference/gpu_representations.py",
        "neurobench/experiments/gamma_ls_difference/full_recording.py",
        "neurobench/experiments/gamma_ls_difference/scientific_audit.py",
        "neurobench/experiments/gamma_ls_difference/full_recording_scientific_audit.py",
        "tests/test_gamma_ls_full_recording_scientific_audit.py",
    )
    output: dict[str, dict[str, Any]] = {}
    for relative in relatives:
        path = repository / relative
        if not path.is_file():
            raise FullRecordScientificAuditUnavailable(
                f"implementation input missing: repo://{relative}"
            )
        output[f"repo://{relative}"] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return output


def _reference_from_contract(contract: Mapping[str, Any]) -> GammaReferenceSpec:
    context = contract["context"]
    if dict(context) != EXPECTED_CONTEXT:
        raise FullRecordScientificAuditUnavailable("frozen context geometry changed")
    return GammaReferenceSpec.from_mode(
        CONTEXT_ID,
        support_width_px=31,
        shape_n=9.0,
        mode_radius_px=7.5,
        guard_radius_px=7.0,
        support_geometry="disk",
        boundary_mode="valid_renormalized_zero",
        epsilon=GAMMA_EPSILON,
        scale_floor=0.0,
    )


def _git_state(repository: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = run("status", "--short")
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _require_distinct_nonoverlapping_roots(paths: Sequence[Path]) -> None:
    resolved = [path.expanduser().resolve() for path in paths]
    for index, first in enumerate(resolved):
        for second in resolved[index + 1 :]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError(
                    f"artifact roots must be distinct and non-overlapping: {first}, {second}"
                )


def _require_peak_vram_within_cap(
    allocated_bytes: int, config: GammaLSDifferenceConfig
) -> int:
    cap = int(float(config.payload["resources"]["max_peak_vram_gib"]) * 2**30)
    if int(allocated_bytes) > cap:
        raise FullRecordScientificAuditUnavailable(
            f"peak CUDA allocation {int(allocated_bytes)} exceeds cap {cap}"
        )
    return cap


def _require_free_vram_for_preflight(runtime: Mapping[str, Any], cap: int) -> None:
    if int(runtime["free_vram_bytes_before"]) < int(cap):
        raise FullRecordScientificAuditUnavailable(
            "free CUDA memory is below the frozen peak-VRAM guard"
        )


def _verify_recovery_checkpoint_dense_coverage(
    checkpoint: Mapping[str, Any], *, chunk_frames: int
) -> list[tuple[int, int]]:
    """Require the exact complete checkpoint schedule and all stage-hash keys."""

    expected_ranges = [(0, CALIBRATION_UI[1])] + [
        (start, stop)
        for start, stop in _aligned_chunks(
            TOTAL_FRAMES,
            int(chunk_frames),
            boundary_stops_ui=(CALIBRATION_UI[1],),
        )
        if start >= CALIBRATION_UI[1]
    ]
    raw_ranges = checkpoint.get("completed_ranges_zero_half_open")
    if not isinstance(raw_ranges, list) or any(
        not isinstance(row, list) or len(row) != 2 for row in raw_ranges
    ):
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery checkpoint is not the complete dense replay"
        )
    completed_ranges = [(int(row[0]), int(row[1])) for row in raw_ranges]
    hashes = checkpoint.get("chunk_stage_sha256")
    expected_hash_keys = {f"{start}:{stop}" for start, stop in expected_ranges}
    if (
        checkpoint.get("schema_version") != 2
        or checkpoint.get("status") != "checkpointed_replay_incomplete"
        or checkpoint.get("last_completed_source_frame_ui") != TOTAL_FRAMES
        or checkpoint.get("annotation_sources_opened") is not False
        or completed_ranges != expected_ranges
        or not isinstance(hashes, Mapping)
        or set(hashes) != expected_hash_keys
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes.values()
        )
    ):
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery checkpoint is not the complete dense replay"
        )
    return completed_ranges


def _verify_known_gate_recovery_source_for_preflight(
    work: Path,
    replay_destination: Path,
    *,
    ledger: Mapping[str, Any],
    device: str,
    chunk_frames: int,
    configured_vram_cap: int,
) -> dict[str, Any]:
    """Verify the one pinned completed replay without touching live CUDA.

    This path exists only to recover the historical post-compute all-true gate
    failure.  The original preflight's GPU identity and the completed dense
    checkpoint are evidence inputs; they are never refreshed or recomputed.
    """

    pins = KNOWN_ALL_TRUE_GATE_RECOVERY
    if replay_destination.name != pins["replay_output_name"]:
        raise FullRecordScientificAuditUnavailable(
            "known gate recovery is not authorized for this replay destination"
        )
    required_hashes = {
        "run_contract.json": pins["run_contract_sha256"],
        "replay_checkpoint.json": pins["replay_checkpoint_sha256"],
        "failure.json": pins["failure_sha256"],
    }
    for name, expected in required_hashes.items():
        path = work / name
        if not path.is_file() or _sha256(path) != expected:
            raise FullRecordScientificAuditUnavailable(
                f"known gate-recovery source changed: {name}"
            )
    contract = _read_json(work / "run_contract.json")
    old_preflight_root = Path(str(contract["preflight_root"])).resolve()
    if old_preflight_root.name != pins["preflight_root_name"]:
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery preflight identity changed"
        )
    old_index = _verify_complete_index(old_preflight_root)
    if old_index["artifact_index_sha256"] != pins[
        "preflight_artifact_index_sha256"
    ]:
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery preflight index changed"
        )
    old_preflight_path = old_preflight_root / "preflight.json"
    if _sha256(old_preflight_path) != pins["preflight_json_sha256"]:
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery preflight payload changed"
        )
    old_preflight = _read_json(old_preflight_path)
    old_science = _read_json(old_preflight_root / "science_contract.json")
    if (
        old_preflight.get("status") != "ready"
        or old_preflight.get("gpu_run_ready") is not True
        or _canonical_sha256(old_science)
        != str(old_preflight.get("science_contract_sha256", ""))
    ):
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery original preflight was not GPU-ready"
        )
    implementation = old_preflight.get("implementation", {}).get("files", {})
    if implementation.get(
        "repo://neurobench/experiments/gamma_ls_difference/"
        "full_recording_scientific_audit.py",
        {},
    ).get("sha256") != pins["implementation_sha256"]:
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery implementation identity changed"
        )
    if implementation.get(
        "repo://tests/test_gamma_ls_full_recording_scientific_audit.py", {}
    ).get("sha256") != pins["test_sha256"]:
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery test identity changed"
        )
    runtime = old_preflight.get("runtime")
    resources = old_preflight.get("resources")
    if not isinstance(runtime, Mapping) or not isinstance(resources, Mapping):
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery preflight lacks recorded GPU resources"
        )
    integer_runtime_fields = (
        "visible_device_count",
        "free_vram_bytes_before",
        "total_vram_bytes",
    )
    if any(
        isinstance(runtime.get(key), bool) or not isinstance(runtime.get(key), int)
        for key in integer_runtime_fields
    ):
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery recorded GPU identity is invalid"
        )
    compute_capability = runtime.get("compute_capability")
    if (
        runtime.get("cuda_available") is not True
        or str(runtime.get("requested_device")) != str(device)
        or str(runtime.get("resolved_device")) != str(device)
        or str(old_science.get("device")) != str(device)
        or int(runtime["visible_device_count"]) < 1
        or int(runtime["free_vram_bytes_before"]) <= 0
        or int(runtime["total_vram_bytes"])
        < int(runtime["free_vram_bytes_before"])
        or not str(runtime.get("device_name", "")).strip()
        or not str(runtime.get("torch_version", "")).strip()
        or not str(runtime.get("torch_cuda_build", "")).strip()
        or not isinstance(compute_capability, list)
        or len(compute_capability) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in compute_capability
        )
    ):
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery recorded GPU identity differs from the request"
        )
    if (
        isinstance(resources.get("configured_peak_vram_cap_bytes"), bool)
        or not isinstance(resources.get("configured_peak_vram_cap_bytes"), int)
        or int(resources["configured_peak_vram_cap_bytes"])
        != int(configured_vram_cap)
        or resources.get("free_vram_gate_passed") is not True
        or int(runtime["free_vram_bytes_before"]) < int(configured_vram_cap)
    ):
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery original GPU resource gate changed"
        )

    checkpoint_path = work / "replay_checkpoint.json"
    checkpoint = _read_json(checkpoint_path)
    _verify_recovery_checkpoint_dense_coverage(
        checkpoint, chunk_frames=int(chunk_frames)
    )
    arrays: dict[str, np.ndarray] = {}
    expected_arrays = {
        "conditioned": ((TOTAL_FRAMES, FRAME_HEIGHT, FRAME_WIDTH), "float32"),
        "difference": ((TOTAL_FRAMES, FRAME_HEIGHT, FRAME_WIDTH), "float32"),
        "gamma": ((TOTAL_FRAMES, FRAME_HEIGHT, FRAME_WIDTH), "float32"),
        "threshold": (
            (EXPECTED_ELIGIBLE_FRAMES, FRAME_HEIGHT, FRAME_WIDTH),
            "bool",
        ),
    }
    for key, (shape, dtype) in expected_arrays.items():
        path = work / STAGE_FILES[key]
        if not path.is_file():
            raise FullRecordScientificAuditUnavailable(
                f"known gate-recovery stage array is missing: {STAGE_FILES[key]}"
            )
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(values.shape) != shape or str(values.dtype) != dtype:
            raise FullRecordScientificAuditUnavailable(
                f"known gate-recovery stage array contract changed: {STAGE_FILES[key]}"
            )
        arrays[key] = values
    verified_checkpoint = _load_replay_checkpoint(
        work,
        arrays=arrays,
        ledger=ledger,
        chunk_frames=int(chunk_frames),
    )
    execution = verified_checkpoint["cumulative_execution"]
    peak_allocated = int(execution["peak_vram_allocated_bytes"])
    if peak_allocated > int(configured_vram_cap):
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery recorded peak VRAM exceeds the frozen cap"
        )
    return {
        "recovery_id": "known_all_true_gate_after_complete_exact_replay",
        "source_work_root": str(work),
        "source_preflight_root": str(old_preflight_root),
        "source_preflight_artifact_index_sha256": old_index[
            "artifact_index_sha256"
        ],
        "source_run_contract_sha256": required_hashes["run_contract.json"],
        "source_replay_checkpoint_sha256": required_hashes[
            "replay_checkpoint.json"
        ],
        "source_failure_sha256": required_hashes["failure.json"],
        "dense_stages_or_proposal_rows_recomputed": False,
        "execution_mode": "recovery_only_no_compute",
        "live_cuda_probe_performed": False,
        "compute_authorized": False,
        "source_preflight_gpu_identity_verified": True,
        "complete_checkpoint_stage_hashes_and_ledger_coverage_verified": True,
        "recorded_peak_vram_within_cap": True,
        "recorded_peak_vram_allocated_bytes": peak_allocated,
        "recorded_peak_vram_reserved_bytes": int(
            execution["peak_vram_reserved_bytes"]
        ),
        "recorded_peak_vram_cap_bytes": int(configured_vram_cap),
        "recorded_runtime_identity": dict(runtime),
    }


def _resolve_preflight_runtime_for_mode(
    *,
    device: str,
    configured_vram_cap: int,
    checkpoint_recovery_authorization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Use frozen GPU evidence for recovery; probe CUDA only for a fresh run."""

    if checkpoint_recovery_authorization is not None:
        if (
            checkpoint_recovery_authorization.get("execution_mode")
            != "recovery_only_no_compute"
            or checkpoint_recovery_authorization.get("live_cuda_probe_performed")
            is not False
            or checkpoint_recovery_authorization.get("compute_authorized")
            is not False
            or checkpoint_recovery_authorization.get(
                "source_preflight_gpu_identity_verified"
            )
            is not True
            or checkpoint_recovery_authorization.get(
                "complete_checkpoint_stage_hashes_and_ledger_coverage_verified"
            )
            is not True
            or checkpoint_recovery_authorization.get(
                "recorded_peak_vram_within_cap"
            )
            is not True
        ):
            raise FullRecordScientificAuditUnavailable(
                "checkpoint recovery authorization is incomplete"
            )
        runtime = checkpoint_recovery_authorization.get(
            "recorded_runtime_identity"
        )
        if not isinstance(runtime, Mapping):
            raise FullRecordScientificAuditUnavailable(
                "checkpoint recovery lacks its recorded GPU identity"
            )
        return dict(runtime)
    try:
        runtime = require_cuda_device(device)
    except CudaRuntimeUnavailable as error:
        raise FullRecordScientificAuditUnavailable(str(error)) from error
    _require_free_vram_for_preflight(runtime, configured_vram_cap)
    return runtime


def create_preflight(
    config: GammaLSDifferenceConfig,
    *,
    ledger_root: str | Path,
    preflight_root: str | Path,
    replay_output: str | Path,
    audit_output: str | Path,
    device: str,
    chunk_frames: int,
    anchor_limit: int,
    recover_known_all_true_gate: bool = False,
) -> dict[str, Any]:
    """Freeze code, source, ledger, resources, destinations, and surrogate panel."""

    ledger = _load_operational_ledger(ledger_root)
    repository = config.repository.resolve()
    destination = Path(preflight_root).expanduser().resolve()
    replay_destination = Path(replay_output).expanduser().resolve()
    audit_destination = Path(audit_output).expanduser().resolve()
    _require_distinct_nonoverlapping_roots(
        (destination, replay_destination, audit_destination, ledger["root"])
    )
    if destination.exists() or replay_destination.exists() or audit_destination.exists():
        raise FileExistsError("preflight, replay, and audit roots must be new")
    replay_work = replay_destination.parent / f".{replay_destination.name}.replay-work"
    audit_work = audit_destination.parent / f".{audit_destination.name}.audit-work"
    checkpoint_recovery_authorization: dict[str, Any] | None = None
    if audit_work.exists():
        raise FileExistsError(
            "pre-existing replay/audit work root would make the run ambiguous"
        )
    if replay_work.exists():
        if not recover_known_all_true_gate:
            raise FileExistsError(
                "pre-existing replay/audit work root would make the run ambiguous"
            )
    elif recover_known_all_true_gate:
        raise FileNotFoundError(
            "known all-true gate recovery was requested but its work root is absent"
        )
    for parent in (
        destination.parent,
        replay_destination.parent,
        audit_destination.parent,
    ):
        if not parent.is_dir():
            raise FileNotFoundError(parent)
    if int(chunk_frames) != DEFAULT_CHUNK_FRAMES:
        raise ValueError("full-record audit replay is frozen to chunk_frames=64")
    if int(anchor_limit) != DEFAULT_ANCHOR_LIMIT:
        raise ValueError("full-record audit is frozen to 24 candidate surrogates")
    movie_path = config.source_paths["movie"]
    if not movie_path.is_file():
        raise FileNotFoundError(movie_path)
    movie_sha256 = _sha256(movie_path)
    ledger_contract = ledger["contract"]
    if movie_sha256 != str(ledger_contract["movie_sha256"]):
        raise FullRecordScientificAuditUnavailable("source movie hash changed")
    movie = np.load(movie_path, mmap_mode="r", allow_pickle=False)
    if tuple(movie.shape) != (TOTAL_FRAMES, FRAME_HEIGHT, FRAME_WIDTH) or str(
        movie.dtype
    ) != "uint16":
        raise FullRecordScientificAuditUnavailable("source movie shape/dtype changed")
    if _canonical_sha256(config.portable_dict()) != str(
        ledger_contract["portable_config_sha256"]
    ):
        raise FullRecordScientificAuditUnavailable("portable config changed")
    full_recording_path = repository / (
        "neurobench/experiments/gamma_ls_difference/full_recording.py"
    )
    if _sha256(full_recording_path) != str(ledger_contract["executor_sha256"]):
        raise FullRecordScientificAuditUnavailable(
            "original full-record executor changed since the sealed ledger"
        )
    support_root = Path(str(ledger_contract["support_artifact"]["root"])).resolve()
    support_indexed = _verify_complete_index(support_root)
    if support_indexed["artifact_index_sha256"] != str(
        ledger_contract["support_artifact"]["artifact_index_sha256"]
    ):
        raise FullRecordScientificAuditUnavailable("support artifact hash changed")
    reference = _reference_from_contract(ledger_contract)
    del reference
    surrogates, assignments, cluster_count = _build_candidate_surrogates(
        ledger["q1_candidates"], limit=anchor_limit
    )
    if len(surrogates) != anchor_limit:
        raise FullRecordScientificAuditUnavailable(
            "frozen q1 ledger does not supply 24 review surrogates"
        )
    representative = min(
        ledger["q1_candidates"],
        key=lambda row: (
            -float(row["score"]),
            int(row["source_frame_ui"]),
            int(row["candidate_rank_within_frame"]),
            str(row["proposal_id"]),
        ),
    )
    configured_vram_cap = int(
        float(config.payload["resources"]["max_peak_vram_gib"]) * 2**30
    )
    if recover_known_all_true_gate:
        checkpoint_recovery_authorization = (
            _verify_known_gate_recovery_source_for_preflight(
                replay_work,
                replay_destination,
                ledger=ledger,
                device=device,
                chunk_frames=int(chunk_frames),
                configured_vram_cap=configured_vram_cap,
            )
        )
    runtime = _resolve_preflight_runtime_for_mode(
        device=device,
        configured_vram_cap=configured_vram_cap,
        checkpoint_recovery_authorization=checkpoint_recovery_authorization,
    )
    recovery_only = checkpoint_recovery_authorization is not None
    free_disk = shutil.disk_usage(destination.parent).free
    minimum_disk = max(
        int(float(config.payload["resources"]["minimum_free_disk_gib"]) * 2**30),
        12 * 2**30,
    )
    if free_disk < minimum_disk:
        raise FullRecordScientificAuditUnavailable(
            f"free disk {free_disk} below required {minimum_disk} bytes"
        )
    implementation = _implementation_files(repository)
    q1_threshold = ledger["q1_threshold"]
    science_contract = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "run_type": "exact_operational_replay_then_model_only_scientific_audit",
        "evidence_role": "full_record_operational_model_only_audit",
        "source_frames_processed_ui": [1, 2359],
        "calibration_interval_ui": [1, 100],
        "calibration_windows_ui_inclusive": [[1, 50], [51, 100]],
        "calibration_frames_assumed_event_free": False,
        "application_interval_ui": [101, 2359],
        "representation": REPRESENTATION,
        "context_id": CONTEXT_ID,
        "context": EXPECTED_CONTEXT,
        "scale_floor_percentile": 10.0,
        "scale_floor": float(q1_threshold["scale_floor"]),
        "target_nms_peaks_per_calibration_unit": TARGET_BURDEN,
        "calibration_burden_unit": INITIAL_BURDEN_UNIT,
        "threshold_z": float(q1_threshold["threshold_z"]),
        "threshold_rule": "strictly_greater_than",
        "nms": "deterministic_greedy_euclidean_separated",
        "nms_distance_px": NMS_DISTANCE_PX,
        "nms_border_exclusion_px": NMS_DISTANCE_PX,
        "expected_eligible_frames": EXPECTED_ELIGIBLE_FRAMES,
        "expected_proposal_rows": EXPECTED_PROPOSALS,
        "expected_frames_with_proposal": EXPECTED_PROPOSAL_FRAMES,
        "temporal_linking_applied": False,
        "annotation_sources_opened": False,
        "biological_identity_or_precision_claimed": False,
        "candidate_surrogate_rule": (
            "greedy_3px_spatial_consolidation_then_recurrence_rank_score_id"
        ),
        "candidate_surrogate_limit": anchor_limit,
        "candidate_cluster_count": cluster_count,
        "candidate_surrogates": surrogates,
        "candidate_assignment_content_sha256": _subset_hash(assignments),
        "representative_stage_proposal": representative,
        "representative_stage_selection_rule": (
            "maximum_q1_score_then_source_frame_rank_proposal_id"
        ),
        "stage_arrays": {
            "conditioned_current_frame": "float32 UI1-2359",
            "difference_signed": "float32 UI1-2359 with zero cold start at UI1",
            "gamma_initial100_scale": "float32 UI1-2359",
            "threshold_exceedance_q1_application": "bool UI101-2359",
            "nms_coordinates": "sparse exact q1 ledger UI101-2359",
        },
        "replay_output": str(replay_destination),
        "audit_output": str(audit_destination),
        "device": str(runtime["resolved_device"]),
        "chunk_frames": chunk_frames,
    }
    payload = {
        "schema_version": 1,
        "status": "ready",
        "gpu_run_ready": not recovery_only,
        **(
            {
                "checkpoint_recovery_ready": True,
                "execution_mode": "recovery_only_no_compute",
                "live_cuda_probe_performed": False,
                "compute_authorized": False,
            }
            if recovery_only
            else {}
        ),
        "created_at_utc": _utc_now(),
        "repository": str(repository),
        "git": _git_state(repository),
        "runtime": runtime,
        "resources": {
            "free_disk_bytes": free_disk,
            "minimum_disk_bytes": minimum_disk,
            "estimated_stage_array_bytes": (
                3 * TOTAL_FRAMES * FRAME_HEIGHT * FRAME_WIDTH * 4
                + EXPECTED_ELIGIBLE_FRAMES * FRAME_HEIGHT * FRAME_WIDTH
            ),
            "estimated_stage_array_gib": (
                3 * TOTAL_FRAMES * FRAME_HEIGHT * FRAME_WIDTH * 4
                + EXPECTED_ELIGIBLE_FRAMES * FRAME_HEIGHT * FRAME_WIDTH
            )
            / 2**30,
            "configured_peak_vram_cap_gib": float(
                config.payload["resources"]["max_peak_vram_gib"]
            ),
            "configured_peak_vram_cap_bytes": configured_vram_cap,
            "free_vram_gate_passed": int(runtime["free_vram_bytes_before"])
            >= configured_vram_cap,
            **(
                {
                    "free_vram_gate_scope": (
                        "hash_pinned_original_preflight_not_live_reprobed"
                    )
                }
                if recovery_only
                else {}
            ),
        },
        "source": {
            "movie_path": str(movie_path.resolve()),
            "movie_sha256": movie_sha256,
            "shape_tyx": list(movie.shape),
            "dtype": str(movie.dtype),
            "annotation_sources_opened": False,
        },
        "ledger": {
            "root": str(ledger["root"]),
            **ledger["hashes"],
            "checks": ledger["checks"],
        },
        "support": {
            "root": str(support_root),
            **support_indexed,
        },
        "portable_config_sha256": _canonical_sha256(config.portable_dict()),
        "science_contract_sha256": _canonical_sha256(science_contract),
        "implementation": {
            "complete": True,
            "files": implementation,
        },
        "forbidden_inputs": {
            "annotation_files": "not_opened_and_not_hashed",
            "protected_labels": "not_used",
            "latest_labels": "not_used",
        },
        "checkpoint_recovery_authorization": checkpoint_recovery_authorization,
    }
    work = destination.parent / (
        f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    )
    work.mkdir()
    try:
        _atomic_json(work / "science_contract.json", science_contract)
        _atomic_json(work / "preflight.json", payload)
        _atomic_json(
            work / "status.json",
            {
                "status": "ready",
                "gpu_run_ready": not recovery_only,
                **(
                    {
                        "checkpoint_recovery_ready": True,
                        "execution_mode": "recovery_only_no_compute",
                        "live_cuda_probe_performed": False,
                        "compute_authorized": False,
                    }
                    if recovery_only
                    else {}
                ),
                "annotation_sources_opened": False,
            },
        )
        report_body = (
            "Recovery-only/no-compute preflight. No live CUDA initialization, "
            "allocation, or probe was performed. The hash-pinned original GPU "
            "identity, complete dense-stage checkpoint hashes and ledger coverage, "
            "and recorded peak VRAM under the frozen cap were verified. This "
            "preflight authorizes CPU-only checkpoint finalization; it does not "
            "authorize dense replay or any new GPU computation."
            if recovery_only
            else (
                "Ready for one exact CUDA replay. The immutable 371-row operational "
                "ledger, source movie, original executor, h15 context, current replay "
                "implementation, two initial 50-frame calibration blocks, q=1 "
                "threshold, NMS6 border/separation contract, destinations, and 24 "
                "label-free review surrogates are hash-bound."
            )
        )
        _atomic_text(
            work / "REPORT.md",
            "# Full-record q=1 scientific-audit replay preflight\n\n"
            f"{report_body} No annotation source was opened.\n",
        )
        _atomic_json(work / "artifact_index.json", _artifact_index(work))
        work.replace(destination)
    except BaseException:
        raise
    return {
        "output": str(destination),
        "status": "ready",
        "gpu_run_ready": not recovery_only,
        **(
            {
                "checkpoint_recovery_ready": True,
                "execution_mode": "recovery_only_no_compute",
                "live_cuda_probe_performed": False,
                "compute_authorized": False,
            }
            if recovery_only
            else {}
        ),
        "science_contract_sha256": payload["science_contract_sha256"],
        "artifact_index_sha256": _sha256(destination / "artifact_index.json"),
        "implementation_file_count": len(implementation),
    }


def _resolve_replay_runtime(
    payload: Mapping[str, Any],
    *,
    device: str,
    require_live_cuda: bool,
) -> dict[str, Any]:
    if require_live_cuda:
        runtime = require_cuda_device(device)
        _require_free_vram_for_preflight(
            runtime, int(payload["resources"]["configured_peak_vram_cap_bytes"])
        )
        return {**runtime, "live_cuda_reprobe_performed": True}
    recorded_runtime = payload.get("runtime")
    if not isinstance(recorded_runtime, Mapping):
        raise FullRecordScientificAuditUnavailable(
            "preflight lacks its frozen CUDA runtime record"
        )
    if (
        recorded_runtime.get("cuda_available") is not True
        or str(recorded_runtime.get("resolved_device")) != str(device)
    ):
        raise FullRecordScientificAuditUnavailable(
            "recorded preflight CUDA runtime differs from the requested device"
        )
    return {
        **dict(recorded_runtime),
        "live_cuda_reprobe_performed": False,
        "runtime_use": "verified_complete_checkpoint_cpu_finalization_only",
    }


def _verify_preflight(
    config: GammaLSDifferenceConfig,
    *,
    preflight_root: str | Path,
    ledger_root: str | Path,
    replay_output: str | Path,
    audit_output: str | Path,
    device: str,
    chunk_frames: int,
    anchor_limit: int,
    require_live_cuda: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(preflight_root).expanduser().resolve()
    indexed = _verify_complete_index(root)
    payload = _read_json(root / "preflight.json")
    science = _read_json(root / "science_contract.json")
    recovery_only = (
        payload.get("execution_mode") == "recovery_only_no_compute"
        and payload.get("checkpoint_recovery_ready") is True
        and payload.get("gpu_run_ready") is False
        and payload.get("live_cuda_probe_performed") is False
        and payload.get("compute_authorized") is False
    )
    if payload.get("status") != "ready" or (
        payload.get("gpu_run_ready") is not True and not recovery_only
    ):
        raise FullRecordScientificAuditUnavailable("audit preflight is not ready")
    if recovery_only and require_live_cuda:
        raise FullRecordScientificAuditUnavailable(
            "recovery-only preflight does not authorize CUDA compute"
        )
    if _canonical_sha256(science) != str(payload["science_contract_sha256"]):
        raise FullRecordScientificAuditUnavailable("science contract hash changed")
    if (
        str(Path(science["replay_output"]).resolve())
        != str(Path(replay_output).expanduser().resolve())
        or str(Path(science["audit_output"]).resolve())
        != str(Path(audit_output).expanduser().resolve())
        or str(science["device"]) != str(device)
        or int(science["chunk_frames"]) != int(chunk_frames)
        or int(science["candidate_surrogate_limit"]) != int(anchor_limit)
    ):
        raise FullRecordScientificAuditUnavailable("run arguments differ from preflight")
    if _canonical_sha256(config.portable_dict()) != str(
        payload["portable_config_sha256"]
    ):
        raise FullRecordScientificAuditUnavailable("portable config changed")
    current_implementation = _implementation_files(config.repository.resolve())
    if current_implementation != payload["implementation"]["files"]:
        raise FullRecordScientificAuditUnavailable(
            "implementation fingerprints changed after preflight"
        )
    ledger = _load_operational_ledger(ledger_root)
    if str(ledger["root"]) != str(Path(payload["ledger"]["root"]).resolve()):
        raise FullRecordScientificAuditUnavailable("ledger root changed")
    for key, value in ledger["hashes"].items():
        if str(payload["ledger"].get(key)) != str(value):
            raise FullRecordScientificAuditUnavailable(f"ledger binding changed: {key}")
    movie_path = config.source_paths["movie"]
    if _sha256(movie_path) != str(payload["source"]["movie_sha256"]):
        raise FullRecordScientificAuditUnavailable("source movie hash changed")
    runtime = _resolve_replay_runtime(
        payload,
        device=device,
        require_live_cuda=require_live_cuda,
    )
    return (
        {
            **indexed,
            "preflight_sha256": _sha256(root / "preflight.json"),
            "science_contract_sha256": payload["science_contract_sha256"],
        },
        ledger,
        {"runtime": runtime, "science": science, "payload": payload},
    )


class _CausalStageProcessor(_CausalCudaArm):
    """Expose the exact conditioned frame and signed-difference replay stages.

    The production executor returns only the selected representation.  This
    subclass deliberately reuses its kernel, reflect-index, and causal-state
    initialization while exposing the synchronized conditioned frame needed
    for audit media.  Arithmetic and state-update order match
    :meth:`_CausalCudaArm.process` exactly.
    """

    def __init__(self, *, device: Any, frame_shape: tuple[int, int]) -> None:
        super().__init__(
            arm=REPRESENTATION,
            device=device,
            frame_shape=frame_shape,
        )

    def process_stages(
        self,
        host_chunk: np.ndarray,
        *,
        first_source_frame_ui: int,
    ) -> tuple[Any, Any, np.ndarray]:
        torch = self.torch
        source = np.asarray(host_chunk)
        if (
            source.ndim != 3
            or tuple(map(int, source.shape[1:])) != (self.height, self.width)
            or source.dtype != np.uint16
        ):
            raise ValueError("source chunk must be uint16 TYX at the frozen geometry")
        device_values = torch.from_numpy(np.ascontiguousarray(source)).to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=False,
        )
        spatial = self._spatial(device_values)
        conditioned = []
        differences = []
        source_ui = []
        for offset in range(int(spatial.shape[0])):
            if self.ema_state is None:
                common = spatial[offset]
            else:
                common = (
                    gpu_representations.EMA_ALPHA * spatial[offset]
                    + (1.0 - gpu_representations.EMA_ALPHA) * self.ema_state
                )
            if self.previous_common is None:
                difference = torch.zeros_like(common)
            else:
                difference = common - self.previous_common
            self.ema_state = common
            self.previous_common = common
            conditioned.append(common)
            differences.append(difference)
            source_ui.append(int(first_source_frame_ui) + offset)
        return (
            torch.stack(conditioned),
            torch.stack(differences),
            np.asarray(source_ui, dtype=np.int64),
        )


def _heartbeat(path: Path, stage: str, **details: Any) -> None:
    _atomic_json(
        path,
        {
            "status": "running",
            "stage": stage,
            "updated_at_utc": _utc_now(),
            **details,
        },
    )


def _archive_recovered_failure(work: Path) -> bool:
    failure = work / "failure.json"
    if not failure.is_file():
        return False
    history = work / "recovery_history"
    history.mkdir(parents=True, exist_ok=True)
    payload = _read_json(failure)
    target = history / "last_failure_before_success.json"
    _atomic_json(
        target,
        {
            **payload,
            "archived_at_success_utc": _utc_now(),
            "final_status_supersedes_failure": True,
        },
    )
    failure.unlink()
    return True


def _manifest_array(
    path: Path, *, relative_path: str, ui_interval: Sequence[int]
) -> dict[str, Any]:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "path": relative_path,
        "file_sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "shape_tyx": list(map(int, values.shape)),
        "dtype": str(values.dtype),
        "source_interval_ui": list(map(int, ui_interval)),
    }


def _normalized_rows_equal(
    first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]
) -> bool:
    return _canonical_sha256([dict(row) for row in first]) == _canonical_sha256(
        [dict(row) for row in second]
    )


def _stage_chunk_sha256(
    arrays: Mapping[str, np.ndarray], *, start_zero: int, stop_zero: int
) -> str:
    digest = hashlib.sha256()
    for key in ("conditioned", "difference", "gamma"):
        values = np.ascontiguousarray(arrays[key][start_zero:stop_zero])
        digest.update(key.encode("ascii"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.view(np.uint8))
    application_start = max(start_zero, APPLICATION_UI[0] - 1)
    if application_start < stop_zero:
        threshold_start = application_start - (APPLICATION_UI[0] - 1)
        threshold_stop = stop_zero - (APPLICATION_UI[0] - 1)
        values = np.ascontiguousarray(
            arrays["threshold"][threshold_start:threshold_stop]
        )
        digest.update(b"threshold")
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.view(np.uint8))
    return digest.hexdigest()


def _resume_chunk_ranges(last_completed_ui: int, chunk_frames: int) -> list[tuple[int, int]]:
    chunks = _aligned_chunks(
        TOTAL_FRAMES,
        int(chunk_frames),
        boundary_stops_ui=(CALIBRATION_UI[1],),
    )
    stops = {stop for _, stop in chunks}
    if last_completed_ui not in ({0} | stops):
        raise FullRecordScientificAuditUnavailable(
            "checkpoint does not end on a frozen chunk boundary"
        )
    return [(start, stop) for start, stop in chunks if start >= last_completed_ui]


def _validate_checkpoint_calibration_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FullRecordScientificAuditUnavailable(
            "checkpoint calibration summary must be an object"
        )
    summary = dict(value)
    expected_fields = {
        "variant_id",
        "calibration_frame_count",
        "calibration_frame_first_ui",
        "calibration_frame_last_ui",
        "positive_local_std_sample_count",
        "scale_floor_percentile",
        "scale_floor",
        "calibration_burden_unit",
        "calibration_window_template_ui_frames",
        "calibration_windows_source_ui",
        "calibration_window_overlap_frames",
        "cpu_nms_exact_parity",
    }
    if set(summary) != expected_fields:
        raise FullRecordScientificAuditUnavailable(
            "checkpoint calibration summary schema changed"
        )
    expected_values = {
        "variant_id": VARIANT_ID,
        "calibration_frame_count": 100,
        "calibration_frame_first_ui": 1,
        "calibration_frame_last_ui": 100,
        "scale_floor_percentile": 10.0,
        "scale_floor": PINNED_SCALE_FLOOR,
        "calibration_burden_unit": INITIAL_BURDEN_UNIT,
        "calibration_window_template_ui_frames": {
            "initial_1s_block_1": 50,
            "initial_1s_block_2": 50,
        },
        "calibration_windows_source_ui": {
            "initial_1s_block_1": [1, 50],
            "initial_1s_block_2": [51, 100],
        },
        "calibration_window_overlap_frames": 0,
        "cpu_nms_exact_parity": (
            "maintained_strict_separated_nms_on_transferred_float32_scores"
        ),
    }
    if any(summary[key] != expected for key, expected in expected_values.items()):
        raise FullRecordScientificAuditUnavailable(
            "checkpoint calibration summary differs from the frozen protocol"
        )
    positive_count = summary["positive_local_std_sample_count"]
    if isinstance(positive_count, bool) or not isinstance(positive_count, int):
        raise FullRecordScientificAuditUnavailable(
            "checkpoint calibration positive sample count must be an integer"
        )
    if positive_count <= 0 or positive_count > 100 * FRAME_HEIGHT * FRAME_WIDTH:
        raise FullRecordScientificAuditUnavailable(
            "checkpoint calibration positive sample count is out of range"
        )
    return summary


def _validate_checkpoint_execution(
    value: Any,
    *,
    last_completed_ui: int,
    completed_ranges: Sequence[tuple[int, int]],
    chunk_frames: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FullRecordScientificAuditUnavailable(
            "checkpoint cumulative execution must be an object"
        )
    execution = dict(value)
    expected_fields = {
        "cumulative_successful_stage_wall_seconds",
        "completed_source_frames",
        "completed_compute_chunk_count",
        "completed_checkpoint_range_count",
        "contributing_execution_segment_count",
        "peak_vram_allocated_bytes",
        "peak_vram_reserved_bytes",
        "wall_seconds_scope",
        "throughput_claim_eligible",
    }
    if set(execution) != expected_fields:
        raise FullRecordScientificAuditUnavailable(
            "checkpoint cumulative execution schema changed"
        )
    wall_seconds = float(execution["cumulative_successful_stage_wall_seconds"])
    if not math.isfinite(wall_seconds) or wall_seconds < 0.0:
        raise FullRecordScientificAuditUnavailable(
            "checkpoint cumulative wall time is invalid"
        )
    integer_fields = (
        "completed_source_frames",
        "completed_compute_chunk_count",
        "completed_checkpoint_range_count",
        "contributing_execution_segment_count",
        "peak_vram_allocated_bytes",
        "peak_vram_reserved_bytes",
    )
    if any(
        isinstance(execution[key], bool) or not isinstance(execution[key], int)
        for key in integer_fields
    ):
        raise FullRecordScientificAuditUnavailable(
            "checkpoint cumulative execution counts must be integers"
        )
    if execution["completed_source_frames"] != int(last_completed_ui):
        raise FullRecordScientificAuditUnavailable(
            "checkpoint execution frame count differs from stage coverage"
        )
    if execution["completed_checkpoint_range_count"] != len(completed_ranges):
        raise FullRecordScientificAuditUnavailable(
            "checkpoint execution range count differs from stage coverage"
        )
    expected_compute_chunks = sum(
        stop <= int(last_completed_ui)
        for _, stop in _aligned_chunks(
            TOTAL_FRAMES,
            int(chunk_frames),
            boundary_stops_ui=(CALIBRATION_UI[1],),
        )
    )
    if execution["completed_compute_chunk_count"] != expected_compute_chunks:
        raise FullRecordScientificAuditUnavailable(
            "checkpoint execution compute-chunk count differs from frozen schedule"
        )
    if execution["contributing_execution_segment_count"] < 1:
        raise FullRecordScientificAuditUnavailable(
            "checkpoint execution segment count is invalid"
        )
    if (
        execution["peak_vram_allocated_bytes"] < 0
        or execution["peak_vram_reserved_bytes"] < 0
        or execution["peak_vram_reserved_bytes"]
        < execution["peak_vram_allocated_bytes"]
    ):
        raise FullRecordScientificAuditUnavailable(
            "checkpoint peak VRAM accounting is invalid"
        )
    if execution["wall_seconds_scope"] != (
        "sum_of_successful_stage_compute_flush_and_hash_intervals_ending_before_"
        "each_atomic_checkpoint_write; failed_incomplete_chunks_are_excluded"
    ):
        raise FullRecordScientificAuditUnavailable(
            "checkpoint wall-time scope changed"
        )
    if execution["throughput_claim_eligible"] is not False:
        raise FullRecordScientificAuditUnavailable(
            "audit replay must not be promoted as a throughput benchmark"
        )
    execution["cumulative_successful_stage_wall_seconds"] = wall_seconds
    return execution


def _cumulative_checkpoint_execution(
    *,
    prior: Mapping[str, Any] | None,
    current_segment_wall_seconds: float,
    current_segment_peak_allocated_bytes: int,
    current_segment_peak_reserved_bytes: int,
    last_completed_ui: int,
    completed_ranges: Sequence[tuple[int, int]],
    chunk_frames: int,
) -> dict[str, Any]:
    prior_wall = (
        0.0
        if prior is None
        else float(prior["cumulative_successful_stage_wall_seconds"])
    )
    prior_segments = (
        0 if prior is None else int(prior["contributing_execution_segment_count"])
    )
    prior_allocated = 0 if prior is None else int(prior["peak_vram_allocated_bytes"])
    prior_reserved = 0 if prior is None else int(prior["peak_vram_reserved_bytes"])
    result = {
        "cumulative_successful_stage_wall_seconds": (
            prior_wall + float(current_segment_wall_seconds)
        ),
        "completed_source_frames": int(last_completed_ui),
        "completed_compute_chunk_count": sum(
            stop <= int(last_completed_ui)
            for _, stop in _aligned_chunks(
                TOTAL_FRAMES,
                int(chunk_frames),
                boundary_stops_ui=(CALIBRATION_UI[1],),
            )
        ),
        "completed_checkpoint_range_count": len(completed_ranges),
        "contributing_execution_segment_count": prior_segments + 1,
        "peak_vram_allocated_bytes": max(
            prior_allocated, int(current_segment_peak_allocated_bytes)
        ),
        "peak_vram_reserved_bytes": max(
            prior_reserved, int(current_segment_peak_reserved_bytes)
        ),
        "wall_seconds_scope": (
            "sum_of_successful_stage_compute_flush_and_hash_intervals_ending_before_"
            "each_atomic_checkpoint_write; failed_incomplete_chunks_are_excluded"
        ),
        "throughput_claim_eligible": False,
    }
    return _validate_checkpoint_execution(
        result,
        last_completed_ui=last_completed_ui,
        completed_ranges=completed_ranges,
        chunk_frames=chunk_frames,
    )


def _write_replay_checkpoint(
    work: Path,
    *,
    last_completed_ui: int,
    completed_ranges: Sequence[tuple[int, int]],
    chunk_hashes: Mapping[str, str],
    candidates: Sequence[Mapping[str, Any]],
    counts: Sequence[Mapping[str, Any]],
    threshold_rows: Sequence[Mapping[str, Any]],
    fitted_floor: float,
    calibration_summary: Mapping[str, Any],
    cumulative_execution: Mapping[str, Any],
    chunk_frames: int,
) -> None:
    calibration = _validate_checkpoint_calibration_summary(calibration_summary)
    execution = _validate_checkpoint_execution(
        cumulative_execution,
        last_completed_ui=last_completed_ui,
        completed_ranges=completed_ranges,
        chunk_frames=chunk_frames,
    )
    _atomic_json(
        work / "replay_checkpoint.json",
        {
            "schema_version": 2,
            "status": "checkpointed_replay_incomplete",
            "updated_at_utc": _utc_now(),
            "last_completed_source_frame_ui": int(last_completed_ui),
            "completed_ranges_zero_half_open": [list(row) for row in completed_ranges],
            "chunk_stage_sha256": dict(chunk_hashes),
            "replay_candidates": [dict(row) for row in candidates],
            "replay_counts": [dict(row) for row in counts],
            "threshold_rows": [dict(row) for row in threshold_rows],
            "fitted_floor": float(fitted_floor),
            "calibration_summary": calibration,
            "calibration_summary_sha256": _canonical_sha256(calibration),
            "cumulative_execution": execution,
            "cumulative_execution_sha256": _canonical_sha256(execution),
            "annotation_sources_opened": False,
            "resume_semantics": (
                "resume verifies every completed stage-chunk hash and exact sealed-"
                "ledger prefix, restores the causal EMA state from the last persisted "
                "conditioned frame, and continues at the next frozen chunk boundary"
            ),
        },
    )


def _load_replay_checkpoint(
    work: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    ledger: Mapping[str, Any],
    chunk_frames: int,
) -> dict[str, Any]:
    checkpoint = _read_json(work / "replay_checkpoint.json")
    if checkpoint.get("schema_version") != 2:
        raise FullRecordScientificAuditUnavailable(
            "checkpoint schema does not preserve resumable execution provenance"
        )
    last_completed = int(checkpoint["last_completed_source_frame_ui"])
    _resume_chunk_ranges(last_completed, chunk_frames)
    completed_ranges = [
        (int(row[0]), int(row[1]))
        for row in checkpoint["completed_ranges_zero_half_open"]
    ]
    if not completed_ranges or completed_ranges[-1][1] != last_completed:
        raise FullRecordScientificAuditUnavailable("checkpoint range coverage changed")
    if completed_ranges[0][0] != 0 or any(
        first[1] != second[0]
        for first, second in zip(completed_ranges, completed_ranges[1:])
    ):
        raise FullRecordScientificAuditUnavailable("checkpoint ranges are not contiguous")
    hashes = dict(checkpoint["chunk_stage_sha256"])
    for start, stop in completed_ranges:
        key = f"{start}:{stop}"
        if _stage_chunk_sha256(arrays, start_zero=start, stop_zero=stop) != hashes.get(
            key
        ):
            raise FullRecordScientificAuditUnavailable(
                f"checkpointed stage chunk changed: {key}"
            )
    candidates = [
        _normalize_candidate(row) for row in checkpoint["replay_candidates"]
    ]
    counts = [_normalize_count(row) for row in checkpoint["replay_counts"]]
    thresholds = [
        _normalize_threshold(row) for row in checkpoint["threshold_rows"]
    ]
    expected_candidates = [
        row
        for row in ledger["initial_candidates"]
        if int(row["source_frame_ui"]) <= last_completed
    ]
    expected_counts = [
        row
        for row in ledger["initial_counts"]
        if int(row["source_frame_ui"]) <= last_completed
    ]
    if not _normalized_rows_equal(candidates, expected_candidates):
        raise FullRecordScientificAuditUnavailable(
            "checkpoint candidate prefix differs from sealed ledger"
        )
    if not _normalized_rows_equal(counts, expected_counts):
        raise FullRecordScientificAuditUnavailable(
            "checkpoint frame-count prefix differs from sealed ledger"
        )
    if not _normalized_rows_equal(thresholds, ledger["initial_thresholds"]):
        raise FullRecordScientificAuditUnavailable(
            "checkpoint threshold grid differs from sealed ledger"
        )
    if float(checkpoint["fitted_floor"]) != PINNED_SCALE_FLOOR:
        raise FullRecordScientificAuditUnavailable("checkpoint scale floor changed")
    calibration_summary = _validate_checkpoint_calibration_summary(
        checkpoint.get("calibration_summary")
    )
    if checkpoint.get("calibration_summary_sha256") != _canonical_sha256(
        calibration_summary
    ):
        raise FullRecordScientificAuditUnavailable(
            "checkpoint calibration summary hash changed"
        )
    cumulative_execution = _validate_checkpoint_execution(
        checkpoint.get("cumulative_execution"),
        last_completed_ui=last_completed,
        completed_ranges=completed_ranges,
        chunk_frames=chunk_frames,
    )
    if checkpoint.get("cumulative_execution_sha256") != _canonical_sha256(
        cumulative_execution
    ):
        raise FullRecordScientificAuditUnavailable(
            "checkpoint cumulative execution hash changed"
        )
    return {
        **checkpoint,
        "last_completed_source_frame_ui": last_completed,
        "completed_ranges": completed_ranges,
        "chunk_stage_sha256": hashes,
        "replay_candidates": candidates,
        "replay_counts": counts,
        "threshold_rows": thresholds,
        "calibration_summary": calibration_summary,
        "cumulative_execution": cumulative_execution,
    }


def _contract_without_preflight_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in contract.items()
        if key not in {"preflight", "preflight_root"}
    }


def _verify_known_all_true_gate_recovery(
    work: Path,
    *,
    existing_contract: Mapping[str, Any],
    current_contract: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Authorize only the sealed post-replay all-true gate failure."""

    pins = KNOWN_ALL_TRUE_GATE_RECOVERY
    if destination.name != pins["replay_output_name"]:
        raise FullRecordScientificAuditUnavailable(
            "checkpoint run contract differs from current preflight"
        )
    if _sha256(work / "run_contract.json") != pins["run_contract_sha256"]:
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery run contract hash changed"
        )
    if _sha256(work / "replay_checkpoint.json") != pins[
        "replay_checkpoint_sha256"
    ]:
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery checkpoint hash changed"
        )
    if _sha256(work / "failure.json") != pins["failure_sha256"]:
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery failure evidence hash changed"
        )
    if _contract_without_preflight_identity(
        existing_contract
    ) != _contract_without_preflight_identity(current_contract):
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery scientific run contract changed"
        )
    old_preflight_root = Path(str(existing_contract["preflight_root"])).resolve()
    if old_preflight_root.name != pins["preflight_root_name"]:
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery preflight identity changed"
        )
    old_index = _verify_complete_index(old_preflight_root)
    if old_index["artifact_index_sha256"] != pins[
        "preflight_artifact_index_sha256"
    ]:
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery preflight index changed"
        )
    old_preflight = _read_json(old_preflight_root / "preflight.json")
    if _sha256(old_preflight_root / "preflight.json") != pins[
        "preflight_json_sha256"
    ]:
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery preflight payload changed"
        )
    implementation = old_preflight.get("implementation", {}).get("files", {})
    if implementation.get(
        "repo://neurobench/experiments/gamma_ls_difference/"
        "full_recording_scientific_audit.py",
        {},
    ).get("sha256") != pins["implementation_sha256"]:
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery implementation identity changed"
        )
    if implementation.get(
        "repo://tests/test_gamma_ls_full_recording_scientific_audit.py", {}
    ).get("sha256") != pins["test_sha256"]:
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery test identity changed"
        )
    checkpoint = _read_json(work / "replay_checkpoint.json")
    if (
        checkpoint.get("schema_version") != 2
        or checkpoint.get("status") != "checkpointed_replay_incomplete"
        or checkpoint.get("last_completed_source_frame_ui") != TOTAL_FRAMES
    ):
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery checkpoint is not the complete dense replay"
        )
    failure = _read_json(work / "failure.json")
    error = str(failure.get("error", ""))
    if (
        failure.get("error_type") != "FullRecordScientificAuditUnavailable"
        or failure.get("checkpoint_exists") is not True
        or not error.startswith("stage replay checks failed:")
        or "'annotation_sources_opened': False" not in error
        or "'biological_claims_made': False" not in error
        or "False" not in error
    ):
        raise FullRecordScientificAuditUnavailable(
            "known gate-recovery failure signature changed"
        )
    return {
        "schema_version": 1,
        "recovery_id": "known_all_true_gate_after_complete_exact_replay",
        "old_preflight_root": str(old_preflight_root),
        "old_preflight_artifact_index_sha256": old_index[
            "artifact_index_sha256"
        ],
        "old_run_contract_sha256": pins["run_contract_sha256"],
        "old_replay_checkpoint_sha256": pins["replay_checkpoint_sha256"],
        "old_failure_sha256": pins["failure_sha256"],
        "old_implementation_sha256": pins["implementation_sha256"],
        "old_test_sha256": pins["test_sha256"],
        "dense_stages_or_proposal_rows_recomputed": False,
        "metrics_or_selection_changed": False,
        "requires_full_checkpoint_and_ledger_verification_before_rebind": True,
    }


def _preserve_known_gate_recovery_inputs(
    work: Path, recovery: Mapping[str, Any]
) -> None:
    history = work / "recovery_history"
    history.mkdir(parents=True, exist_ok=True)
    copies = {
        "all_true_gate_run_contract.json": work / "run_contract.json",
        "all_true_gate_failure.json": work / "failure.json",
        "all_true_gate_checkpoint.json": work / "replay_checkpoint.json",
    }
    for name, source in copies.items():
        target = history / name
        if target.is_file():
            if _sha256(target) != _sha256(source):
                raise FullRecordScientificAuditUnavailable(
                    f"preserved gate-recovery input changed: {name}"
                )
        else:
            shutil.copy2(source, target)
    _atomic_json(history / "all_true_gate_recovery_authorization.json", recovery)


def _apply_known_gate_recovery(
    work: Path,
    *,
    recovery: Mapping[str, Any],
    current_contract: Mapping[str, Any],
) -> None:
    payload = {
        **dict(recovery),
        "applied_at_utc": _utc_now(),
        "new_preflight_root": str(current_contract["preflight_root"]),
        "new_preflight_artifact_index_sha256": str(
            current_contract["preflight"]["artifact_index_sha256"]
        ),
        "full_checkpoint_and_ledger_verification_passed": True,
    }
    _atomic_json(work / "checkpoint_recovery.json", payload)
    _atomic_json(work / "run_contract.json", current_contract)
    failure = work / "failure.json"
    if failure.is_file():
        failure.unlink()


def _absence_pass_checks(
    *, annotation_sources_opened: bool, biological_claims_made: bool
) -> dict[str, bool]:
    return {
        "annotation_sources_not_opened": annotation_sources_opened is False,
        "biological_claims_not_made": biological_claims_made is False,
    }


def _replay_completion_checks(
    *,
    calibration_summary: Mapping[str, Any] | None,
    calibration_frames_assumed_event_free: bool,
    fitted_floor: float | None,
    expected_floor: float,
    q1_proposal_rows: int,
    q1_distinct_proposal_frames: int,
    q1_count_rows: int,
    peak_vram_allocated: int,
    peak_vram_cap: int,
    annotation_sources_opened: bool,
    biological_claims_made: bool,
) -> dict[str, bool]:
    return {
        "preflight_verified": True,
        "source_hash_verified": True,
        "calibration_two_50_frame_blocks": calibration_summary is not None
        and calibration_summary["calibration_window_template_ui_frames"]
        == {"initial_1s_block_1": 50, "initial_1s_block_2": 50},
        "calibration_not_assumed_event_free": (
            calibration_frames_assumed_event_free is False
        ),
        "scale_floor_exact": fitted_floor == expected_floor,
        "threshold_grid_exact": True,
        "all_initial_candidates_exact": True,
        "all_initial_frame_counts_exact": True,
        "q1_371_rows_exact": q1_proposal_rows == EXPECTED_PROPOSALS,
        "q1_295_frames_exact": (
            q1_distinct_proposal_frames == EXPECTED_PROPOSAL_FRAMES
        ),
        "q1_2259_frame_counts_exact": q1_count_rows == EXPECTED_ELIGIBLE_FRAMES,
        "source_ui_complete": True,
        "peak_vram_within_frozen_cap": peak_vram_allocated <= peak_vram_cap,
        **_absence_pass_checks(
            annotation_sources_opened=annotation_sources_opened,
            biological_claims_made=biological_claims_made,
        ),
    }


REPLAY_COMPLETION_CHECK_KEYS = frozenset(
    _replay_completion_checks(
        calibration_summary={
            "calibration_window_template_ui_frames": {
                "initial_1s_block_1": 50,
                "initial_1s_block_2": 50,
            }
        },
        calibration_frames_assumed_event_free=False,
        fitted_floor=PINNED_SCALE_FLOOR,
        expected_floor=PINNED_SCALE_FLOOR,
        q1_proposal_rows=EXPECTED_PROPOSALS,
        q1_distinct_proposal_frames=EXPECTED_PROPOSAL_FRAMES,
        q1_count_rows=EXPECTED_ELIGIBLE_FRAMES,
        peak_vram_allocated=0,
        peak_vram_cap=1,
        annotation_sources_opened=False,
        biological_claims_made=False,
    )
)


def _checkpoint_declares_complete_dense_coverage(work: Path) -> bool:
    path = work / "replay_checkpoint.json"
    if not path.is_file():
        return False
    checkpoint = _read_json(path)
    return (
        checkpoint.get("schema_version") == 2
        and checkpoint.get("last_completed_source_frame_ui") == TOTAL_FRAMES
        and checkpoint.get("status")
        in {"checkpointed_replay_incomplete", "complete_exact_stage_replay"}
    )


def _prepare_replay_cuda_session(
    *,
    chunks: Sequence[tuple[int, int]],
    resolved_device: str,
    conditioned: np.ndarray,
    last_completed_ui: int,
) -> tuple[Any | None, Any | None, float | None]:
    """Allocate CUDA state only when at least one dense chunk remains."""

    if not chunks:
        return None, None, None
    import torch

    processor = _CausalStageProcessor(
        device=resolved_device,
        frame_shape=(FRAME_HEIGHT, FRAME_WIDTH),
    )
    if last_completed_ui > 0:
        causal_state = torch.as_tensor(
            np.array(conditioned[last_completed_ui - 1], copy=True),
            dtype=torch.float32,
            device=processor.device,
        )
        processor.ema_state = causal_state
        processor.previous_common = causal_state
    torch.cuda.reset_peak_memory_stats(resolved_device)
    return processor, torch, time.perf_counter()


@_record_replay_failure
def run_stage_replay(
    config: GammaLSDifferenceConfig,
    *,
    preflight_root: str | Path,
    ledger_root: str | Path,
    replay_output: str | Path,
    audit_output: str | Path,
    device: str,
    chunk_frames: int,
    anchor_limit: int,
    resume: bool = False,
) -> dict[str, Any]:
    """Replay the exact operational head and seal dense synchronized stages."""

    destination = Path(replay_output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    work = destination.parent / f".{destination.name}.replay-work"
    completed_checkpoint_cpu_finalize = bool(
        resume
        and work.is_dir()
        and _checkpoint_declares_complete_dense_coverage(work)
    )
    preflight, ledger, frozen = _verify_preflight(
        config,
        preflight_root=preflight_root,
        ledger_root=ledger_root,
        replay_output=destination,
        audit_output=audit_output,
        device=device,
        chunk_frames=chunk_frames,
        anchor_limit=anchor_limit,
        require_live_cuda=not completed_checkpoint_cpu_finalize,
    )
    science = frozen["science"]
    q1_threshold = float(science["threshold_z"])
    expected_threshold = float(ledger["q1_threshold"]["threshold_z"])
    expected_floor = float(ledger["q1_threshold"]["scale_floor"])
    if q1_threshold != expected_threshold:
        raise FullRecordScientificAuditUnavailable("q=1 threshold binding changed")
    reference = _reference_from_contract(ledger["contract"])
    initial_variant = CALIBRATION_VARIANTS[0]
    if initial_variant.variant_id != VARIANT_ID:
        raise FullRecordScientificAuditUnavailable("initial calibration identity changed")

    run_contract = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "run_type": "exact_operational_q1_dense_stage_replay",
            "preflight": preflight,
            "preflight_root": str(Path(preflight_root).expanduser().resolve()),
            "science_contract": science,
            "science_contract_sha256": preflight["science_contract_sha256"],
            "source_ledger_root": str(ledger["root"]),
            "source_ledger_hashes": ledger["hashes"],
            "source_movie_sha256": frozen["payload"]["source"]["movie_sha256"],
            "original_ledger_or_metrics_modified": False,
            "annotation_sources_opened": False,
            "checkpoint_work_root": str(work),
        }
    pending_gate_recovery: dict[str, Any] | None = None
    if resume:
        if not work.is_dir():
            raise FileNotFoundError(f"replay checkpoint root is missing: {work}")
        existing_contract = _read_json(work / "run_contract.json")
        if existing_contract != run_contract:
            pending_gate_recovery = _verify_known_all_true_gate_recovery(
                work,
                existing_contract=existing_contract,
                current_contract=run_contract,
                destination=destination,
            )
            authorization = frozen["payload"].get(
                "checkpoint_recovery_authorization"
            )
            if not isinstance(authorization, Mapping) or any(
                authorization.get(key) != expected
                for key, expected in {
                    "recovery_id": pending_gate_recovery["recovery_id"],
                    "source_work_root": str(work),
                    "source_run_contract_sha256": pending_gate_recovery[
                        "old_run_contract_sha256"
                    ],
                    "source_replay_checkpoint_sha256": pending_gate_recovery[
                        "old_replay_checkpoint_sha256"
                    ],
                    "source_failure_sha256": pending_gate_recovery[
                        "old_failure_sha256"
                    ],
                    "dense_stages_or_proposal_rows_recomputed": False,
                }.items()
            ):
                raise FullRecordScientificAuditUnavailable(
                    "current preflight does not authorize the pinned checkpoint recovery"
                )
            _preserve_known_gate_recovery_inputs(work, pending_gate_recovery)
    else:
        if work.exists():
            raise FileExistsError(
                f"replay checkpoint root exists; use --resume after inspection: {work}"
            )
        work.mkdir()
        (work / "stage_arrays").mkdir()
        _atomic_json(work / "run_contract.json", run_contract)
        _heartbeat(work / "heartbeat.json", "allocating_stage_arrays")
    array_mode = "r+" if resume else "w+"
    conditioned_map = np.lib.format.open_memmap(
        work / STAGE_FILES["conditioned"],
        mode=array_mode,
        dtype=np.float32,
        shape=(TOTAL_FRAMES, FRAME_HEIGHT, FRAME_WIDTH),
    )
    difference_map = np.lib.format.open_memmap(
        work / STAGE_FILES["difference"],
        mode=array_mode,
        dtype=np.float32,
        shape=(TOTAL_FRAMES, FRAME_HEIGHT, FRAME_WIDTH),
    )
    gamma_map = np.lib.format.open_memmap(
        work / STAGE_FILES["gamma"],
        mode=array_mode,
        dtype=np.float32,
        shape=(TOTAL_FRAMES, FRAME_HEIGHT, FRAME_WIDTH),
    )
    threshold_map = np.lib.format.open_memmap(
        work / STAGE_FILES["threshold"],
        mode=array_mode,
        dtype=np.bool_,
        shape=(EXPECTED_ELIGIBLE_FRAMES, FRAME_HEIGHT, FRAME_WIDTH),
    )
    movie = np.load(config.source_paths["movie"], mmap_mode="r", allow_pickle=False)
    if not isinstance(movie, np.memmap):
        raise FullRecordScientificAuditUnavailable("source movie is not memory-mappable")
    representation_parts: list[Any] = []
    mean_parts: list[Any] = []
    std_parts: list[Any] = []
    source_ui_parts: list[np.ndarray] = []
    operating_points: tuple[dict[str, Any], ...] | None = None
    replay_threshold_rows: list[dict[str, Any]] = []
    replay_candidates: list[dict[str, Any]] = []
    replay_counts: list[dict[str, Any]] = []
    calibration_summary: dict[str, Any] | None = None
    fitted_floor: float | None = None
    completed_ranges: list[tuple[int, int]] = []
    chunk_hashes: dict[str, str] = {}
    prior_cumulative_execution: dict[str, Any] | None = None
    latest_cumulative_execution: dict[str, Any] | None = None
    checkpoint_arrays = {
        "conditioned": conditioned_map,
        "difference": difference_map,
        "gamma": gamma_map,
        "threshold": threshold_map,
    }
    last_completed_ui = 0
    if resume and (work / "replay_checkpoint.json").is_file():
        checkpoint = _load_replay_checkpoint(
            work,
            arrays=checkpoint_arrays,
            ledger=ledger,
            chunk_frames=chunk_frames,
        )
        last_completed_ui = int(checkpoint["last_completed_source_frame_ui"])
        completed_ranges = list(checkpoint["completed_ranges"])
        chunk_hashes = dict(checkpoint["chunk_stage_sha256"])
        replay_candidates = list(checkpoint["replay_candidates"])
        replay_counts = list(checkpoint["replay_counts"])
        replay_threshold_rows = list(checkpoint["threshold_rows"])
        fitted_floor = float(checkpoint["fitted_floor"])
        operating_points = tuple(replay_threshold_rows)
        calibration_summary = dict(checkpoint["calibration_summary"])
        prior_cumulative_execution = dict(checkpoint["cumulative_execution"])
        latest_cumulative_execution = dict(prior_cumulative_execution)
        if pending_gate_recovery is not None:
            _apply_known_gate_recovery(
                work,
                recovery=pending_gate_recovery,
                current_contract=run_contract,
            )
    elif resume:
        # A failure before UI100 has no scientifically complete checkpoint.
        # Reuse the already allocated files but deterministically overwrite the
        # short pre-calibration prefix from UI1.
        _heartbeat(
            work / "heartbeat.json",
            "resume_before_first_checkpoint_restarts_ui1_to_ui100",
        )
    chunks = _resume_chunk_ranges(last_completed_ui, int(chunk_frames))
    processor, torch, wall_started = _prepare_replay_cuda_session(
        chunks=chunks,
        resolved_device=str(frozen["runtime"]["resolved_device"]),
        conditioned=conditioned_map,
        last_completed_ui=last_completed_ui,
    )
    inference_context = torch.inference_mode() if torch is not None else nullcontext()
    with _ReplayFailureRecorder(work), inference_context:
        for chunk_index, (start_zero, stop_zero) in enumerate(chunks, start=1):
            if processor is None or torch is None or wall_started is None:
                raise AssertionError("nonempty replay chunks require a CUDA session")
            host = np.array(
                movie[start_zero:stop_zero], dtype=np.uint16, order="C", copy=True
            )
            conditioned, difference, source_ui = processor.process_stages(
                host,
                first_source_frame_ui=start_zero + 1,
            )
            del host
            conditioned_host = conditioned.detach().cpu().numpy().astype(
                np.float32, copy=False
            )
            difference_host = difference.detach().cpu().numpy().astype(
                np.float32, copy=False
            )
            conditioned_map[start_zero:stop_zero] = conditioned_host
            difference_map[start_zero:stop_zero] = difference_host
            moments = gamma_local_standardization(
                difference,
                reference,
                chunk_frames=chunk_frames,
                return_statistics=True,
            )
            if moments.local_mean is None or moments.local_std is None:
                raise AssertionError("Gamma-LS moments are required")
            if stop_zero <= CALIBRATION_UI[1]:
                representation_parts.append(difference.detach().clone())
                mean_parts.append(moments.local_mean.detach().clone())
                std_parts.append(moments.local_std.detach().clone())
                source_ui_parts.append(source_ui.copy())
            if stop_zero == CALIBRATION_UI[1]:
                fitted_floor, rows, calibration_summary = _finalize_calibration(
                    variant=initial_variant,
                    arm=REPRESENTATION,
                    representation_parts=representation_parts,
                    mean_parts=mean_parts,
                    std_parts=std_parts,
                    source_ui_parts=source_ui_parts,
                    scale_floor_percentile=float(
                        science["scale_floor_percentile"]
                    ),
                    frame_interval_ms=float(
                        config.payload["frames"]["frame_interval_ms"]
                    ),
                    configured_burst_durations={},
                )
                operating_points = rows
                replay_threshold_rows = list(rows)
                if fitted_floor != expected_floor:
                    raise FullRecordScientificAuditUnavailable(
                        f"replayed scale floor {fitted_floor!r} differs from sealed "
                        f"{expected_floor!r}"
                    )
                normalized_thresholds = [
                    _normalize_threshold(row) for row in replay_threshold_rows
                ]
                if not _normalized_rows_equal(
                    normalized_thresholds, ledger["initial_thresholds"]
                ):
                    raise FullRecordScientificAuditUnavailable(
                        "replayed initial threshold grid differs from sealed ledger"
                    )
                floor_tensor = torch.as_tensor(
                    fitted_floor,
                    dtype=difference.dtype,
                    device=difference.device,
                )
                calibration_representation = torch.cat(representation_parts, dim=0)
                calibration_mean = torch.cat(mean_parts, dim=0)
                calibration_std = torch.cat(std_parts, dim=0)
                calibration_gamma = (
                    (calibration_representation - calibration_mean)
                    / (torch.maximum(calibration_std, floor_tensor) + GAMMA_EPSILON)
                ).detach().cpu().numpy().astype(np.float32, copy=False)
                gamma_map[: CALIBRATION_UI[1]] = calibration_gamma
                representation_parts.clear()
                mean_parts.clear()
                std_parts.clear()
                source_ui_parts.clear()
            if stop_zero > CALIBRATION_UI[1]:
                if fitted_floor is None or operating_points is None:
                    raise RuntimeError("application began before calibration froze")
                application_mask = source_ui >= APPLICATION_UI[0]
                mask_device = torch.as_tensor(
                    application_mask,
                    dtype=torch.bool,
                    device=difference.device,
                )
                floor_tensor = torch.as_tensor(
                    fitted_floor,
                    dtype=difference.dtype,
                    device=difference.device,
                )
                score = (
                    (difference[mask_device] - moments.local_mean[mask_device])
                    / (
                        torch.maximum(moments.local_std[mask_device], floor_tensor)
                        + GAMMA_EPSILON
                    )
                )
                score_host = score.detach().cpu().numpy().astype(np.float32, copy=False)
                selected_ui = source_ui[application_mask]
                destination_start = int(selected_ui[0]) - APPLICATION_UI[0]
                destination_stop = destination_start + len(selected_ui)
                gamma_map[
                    int(selected_ui[0]) - 1 : int(selected_ui[-1])
                ] = score_host
                threshold_map[destination_start:destination_stop] = (
                    score_host > q1_threshold
                )
                for score_frame, frame_ui in zip(score_host, selected_ui, strict=True):
                    proposals, counts = extract_frame_proposals(
                        score_frame,
                        source_frame_ui=int(frame_ui),
                        variant_id=VARIANT_ID,
                        representation=REPRESENTATION,
                        context_id=CONTEXT_ID,
                        operating_points=operating_points,
                        scale_floor_percentile=float(
                            science["scale_floor_percentile"]
                        ),
                        scale_floor=fitted_floor,
                    )
                    replay_candidates.extend(proposals)
                    replay_counts.extend(counts)
            if fitted_floor is not None:
                conditioned_map.flush()
                difference_map.flush()
                gamma_map.flush()
                threshold_map.flush()
                checkpoint_start = start_zero if completed_ranges else 0
                completed_ranges.append((checkpoint_start, stop_zero))
                chunk_hashes[f"{checkpoint_start}:{stop_zero}"] = _stage_chunk_sha256(
                    checkpoint_arrays,
                    start_zero=checkpoint_start,
                    stop_zero=stop_zero,
                )
                replay_candidates.sort(key=_candidate_sort_key)
                replay_counts.sort(key=_frame_count_sort_key)
                if calibration_summary is None:
                    raise AssertionError(
                        "checkpoint cannot precede exact calibration summary"
                    )
                current_peak_allocated = int(
                    torch.cuda.max_memory_allocated(
                        frozen["runtime"]["resolved_device"]
                    )
                )
                current_peak_reserved = int(
                    torch.cuda.max_memory_reserved(
                        frozen["runtime"]["resolved_device"]
                    )
                )
                latest_cumulative_execution = _cumulative_checkpoint_execution(
                    prior=prior_cumulative_execution,
                    current_segment_wall_seconds=time.perf_counter() - wall_started,
                    current_segment_peak_allocated_bytes=current_peak_allocated,
                    current_segment_peak_reserved_bytes=current_peak_reserved,
                    last_completed_ui=stop_zero,
                    completed_ranges=completed_ranges,
                    chunk_frames=chunk_frames,
                )
                _write_replay_checkpoint(
                    work,
                    last_completed_ui=stop_zero,
                    completed_ranges=completed_ranges,
                    chunk_hashes=chunk_hashes,
                    candidates=replay_candidates,
                    counts=replay_counts,
                    threshold_rows=replay_threshold_rows,
                    fitted_floor=fitted_floor,
                    calibration_summary=calibration_summary,
                    cumulative_execution=latest_cumulative_execution,
                    chunk_frames=chunk_frames,
                )
            del moments, difference, conditioned
            _heartbeat(
                work / "heartbeat.json",
                "cuda_dense_stage_replay",
                completed_chunks=chunk_index,
                total_chunks=len(chunks),
                completed_source_frames=stop_zero,
                total_source_frames=TOTAL_FRAMES,
                replay_candidate_rows=len(replay_candidates),
            )
    if latest_cumulative_execution is None:
        raise FullRecordScientificAuditUnavailable(
            "replay finished without a calibration-bound execution checkpoint"
        )
    conditioned_map.flush()
    difference_map.flush()
    gamma_map.flush()
    threshold_map.flush()
    replay_candidates.sort(key=_candidate_sort_key)
    replay_counts.sort(key=_frame_count_sort_key)
    normalized_candidates = [_normalize_candidate(row) for row in replay_candidates]
    normalized_counts = [_normalize_count(row) for row in replay_counts]
    if not _normalized_rows_equal(
        normalized_candidates, ledger["initial_candidates"]
    ):
        raise FullRecordScientificAuditUnavailable(
            "replayed candidate rows differ from sealed initial-100 ledger"
        )
    if not _normalized_rows_equal(normalized_counts, ledger["initial_counts"]):
        raise FullRecordScientificAuditUnavailable(
            "replayed frame-count rows differ from sealed initial-100 ledger"
        )
    q1_candidates = [
        row
        for row in normalized_candidates
        if row["target_nms_peaks_per_calibration_unit"] == TARGET_BURDEN
    ]
    q1_counts = [
        row
        for row in normalized_counts
        if row["target_nms_peaks_per_calibration_unit"] == TARGET_BURDEN
    ]
    if (
        len(q1_candidates) != EXPECTED_PROPOSALS
        or len(q1_counts) != EXPECTED_ELIGIBLE_FRAMES
        or len({row["source_frame_ui"] for row in q1_candidates})
        != EXPECTED_PROPOSAL_FRAMES
    ):
        raise FullRecordScientificAuditUnavailable("q=1 replay counts changed")
    surrogates, assignments, cluster_count = _build_candidate_surrogates(
        q1_candidates, limit=anchor_limit
    )
    if surrogates != science["candidate_surrogates"]:
        raise FullRecordScientificAuditUnavailable(
            "candidate surrogate panel differs from preflight"
        )
    if _subset_hash(assignments) != science["candidate_assignment_content_sha256"]:
        raise FullRecordScientificAuditUnavailable(
            "candidate assignments differ from preflight"
        )
    candidate_fields = tuple(q1_candidates[0])
    count_fields = tuple(q1_counts[0])
    threshold_fields = tuple(_normalize_threshold(replay_threshold_rows[0]))
    surrogate_fields = tuple(surrogates[0])
    assignment_fields = tuple(assignments[0])
    _atomic_tsv(
        work / "q1_frame_level_proposals_exact_replay.tsv",
        q1_candidates,
        fieldnames=candidate_fields,
    )
    _atomic_tsv(
        work / "q1_proposal_counts_by_frame_exact_replay.tsv",
        q1_counts,
        fieldnames=count_fields,
    )
    _atomic_tsv(
        work / "initial100_empirical_thresholds_exact_replay.tsv",
        [_normalize_threshold(row) for row in replay_threshold_rows],
        fieldnames=threshold_fields,
    )
    _atomic_tsv(
        work / "candidate_surrogates.tsv",
        surrogates,
        fieldnames=surrogate_fields,
    )
    _atomic_tsv(
        work / "candidate_surrogate_assignments.tsv",
        assignments,
        fieldnames=assignment_fields,
    )
    frame_index_rows = [
        {
            "stage_array_index_zero": frame_ui - 1,
            "source_frame_ui": frame_ui,
            "application_index_zero": (
                "" if frame_ui < APPLICATION_UI[0] else frame_ui - APPLICATION_UI[0]
            ),
        }
        for frame_ui in range(1, TOTAL_FRAMES + 1)
    ]
    _atomic_tsv(
        work / "stage_frame_index.tsv",
        frame_index_rows,
        fieldnames=tuple(frame_index_rows[0]),
    )
    del conditioned_map, difference_map, gamma_map, threshold_map
    stage_manifest = {
        "schema_version": 1,
        "source_movie": {
            "path": str(config.source_paths["movie"].resolve()),
            "file_sha256": frozen["payload"]["source"]["movie_sha256"],
            "shape_tyx": [TOTAL_FRAMES, FRAME_HEIGHT, FRAME_WIDTH],
            "dtype": "uint16",
            "source_interval_ui": [1, TOTAL_FRAMES],
        },
        "arrays": {
            "conditioned_current_frame": _manifest_array(
                work / STAGE_FILES["conditioned"],
                relative_path=STAGE_FILES["conditioned"],
                ui_interval=(1, TOTAL_FRAMES),
            ),
            "difference_signed": _manifest_array(
                work / STAGE_FILES["difference"],
                relative_path=STAGE_FILES["difference"],
                ui_interval=(1, TOTAL_FRAMES),
            ),
            "gamma_initial100_scale": _manifest_array(
                work / STAGE_FILES["gamma"],
                relative_path=STAGE_FILES["gamma"],
                ui_interval=(1, TOTAL_FRAMES),
            ),
            "threshold_exceedance_q1_application": _manifest_array(
                work / STAGE_FILES["threshold"],
                relative_path=STAGE_FILES["threshold"],
                ui_interval=APPLICATION_UI,
            ),
        },
        "alignment": {
            "all_float_stage_array_index_zero": "source_frame_ui_minus_1",
            "threshold_array_index_zero": "source_frame_ui_minus_101",
            "difference_cold_start_ui1": "exact_zero_frame",
            "conditioned_difference_identity": (
                "difference[ui-1]=conditioned[ui-1]-conditioned[ui-2] for ui>=2"
            ),
        },
    }
    _atomic_json(work / "stage_manifest.json", stage_manifest)
    reconciliation = {
        "schema_version": 1,
        "status": "exact_match",
        "ledger_root": str(ledger["root"]),
        "ledger_hashes": ledger["hashes"],
        "predicate": {
            "variant_id": VARIANT_ID,
            "target_nms_peaks_per_calibration_unit": TARGET_BURDEN,
        },
        "replayed_initial_candidate_rows": len(normalized_candidates),
        "replayed_initial_count_rows": len(normalized_counts),
        "q1_proposal_rows": len(q1_candidates),
        "q1_distinct_proposal_frames": len(
            {row["source_frame_ui"] for row in q1_candidates}
        ),
        "q1_eligible_frame_rows": len(q1_counts),
        "q1_sum_of_per_frame_counts": sum(row["proposal_count"] for row in q1_counts),
        "candidate_rows_canonical_sha256": _subset_hash(q1_candidates),
        "count_rows_canonical_sha256": _subset_hash(q1_counts),
        "initial_thresholds_canonical_sha256": _subset_hash(
            [_normalize_threshold(row) for row in replay_threshold_rows]
        ),
        "all_initial_operating_points_reconciled": True,
        "original_ledger_or_metrics_modified": False,
        "annotation_sources_opened": False,
    }
    _atomic_json(work / "ledger_reconciliation.json", reconciliation)
    peak_vram_allocated = int(
        latest_cumulative_execution["peak_vram_allocated_bytes"]
    )
    peak_vram_reserved = int(
        latest_cumulative_execution["peak_vram_reserved_bytes"]
    )
    peak_vram_cap = _require_peak_vram_within_cap(peak_vram_allocated, config)
    execution = {
        "wall_seconds": latest_cumulative_execution[
            "cumulative_successful_stage_wall_seconds"
        ],
        "wall_seconds_scope": latest_cumulative_execution["wall_seconds_scope"],
        "throughput_claim_eligible": latest_cumulative_execution[
            "throughput_claim_eligible"
        ],
        "source_frames_processed": latest_cumulative_execution[
            "completed_source_frames"
        ],
        "chunk_frames": chunk_frames,
        "chunk_count": latest_cumulative_execution[
            "completed_compute_chunk_count"
        ],
        "checkpoint_range_count": latest_cumulative_execution[
            "completed_checkpoint_range_count"
        ],
        "contributing_execution_segment_count": latest_cumulative_execution[
            "contributing_execution_segment_count"
        ],
        "causal_state_initialized_at_ui": 1,
        "causal_state_carried_across_chunks": True,
        "dense_stage_device": str(frozen["runtime"]["resolved_device"]),
        "nms_device": "cpu",
        "peak_vram_allocated_bytes": peak_vram_allocated,
        "peak_vram_reserved_bytes": peak_vram_reserved,
        "peak_vram_cap_bytes": peak_vram_cap,
        "calibration": calibration_summary,
        "checkpoint_recovery": (
            _read_json(work / "checkpoint_recovery.json")
            if (work / "checkpoint_recovery.json").is_file()
            else None
        ),
    }
    _atomic_json(work / "execution.json", execution)
    replay_checks = _replay_completion_checks(
        calibration_summary=calibration_summary,
        calibration_frames_assumed_event_free=bool(
            science["calibration_frames_assumed_event_free"]
        ),
        fitted_floor=fitted_floor,
        expected_floor=expected_floor,
        q1_proposal_rows=len(q1_candidates),
        q1_distinct_proposal_frames=len(
            {row["source_frame_ui"] for row in q1_candidates}
        ),
        q1_count_rows=len(q1_counts),
        peak_vram_allocated=peak_vram_allocated,
        peak_vram_cap=peak_vram_cap,
        annotation_sources_opened=False,
        biological_claims_made=False,
    )
    if not all(replay_checks.values()):
        raise FullRecordScientificAuditUnavailable(
            f"stage replay checks failed: {replay_checks}"
        )
    summary = {
        "schema_version": 1,
        "status": "complete_exact_stage_replay_model_media_pending",
        "experiment_id": EXPERIMENT_ID,
        "evidence_role": "full_record_operational_model_only_audit_input",
        "source_interval_ui": [1, TOTAL_FRAMES],
        "application_interval_ui": list(APPLICATION_UI),
        "proposal_row_count": EXPECTED_PROPOSALS,
        "distinct_proposal_frame_count": EXPECTED_PROPOSAL_FRAMES,
        "eligible_frame_count": EXPECTED_ELIGIBLE_FRAMES,
        "candidate_surrogate_count": len(surrogates),
        "candidate_cluster_count": cluster_count,
        "ledger_reconciliation": reconciliation,
        "execution": execution,
        "claim_boundary": (
            "frame-local automated proposals only; no label, biological identity, "
            "precision, specificity, false-positive-rate, or unique-event claim"
        ),
    }
    _atomic_json(work / "summary.json", summary)
    _atomic_json(
        work / "validation.json",
        {
            "status": "passed_exact_stage_replay_model_media_pending",
            "checks": replay_checks,
            "all_checks_pass": True,
            "scientific_audit_complete": False,
        },
    )
    _atomic_json(
        work / "llm_context.json",
        {
            "schema_version": 1,
            "entrypoint": "summary.json",
            "stage_manifest": "stage_manifest.json",
            "ledger_reconciliation": "ledger_reconciliation.json",
            "coordinate_convention": "x=column,y=row",
            "frame_convention": "UI one-based inclusive",
            "annotation_separation": "no annotation source opened",
            "claim_boundary": summary["claim_boundary"],
        },
    )
    _atomic_text(
        work / "REPORT.md",
        "# Exact full-record q=1 stage replay\n\n"
        "This hash-bound CUDA replay reproduced every initial-100 operating-point "
        "proposal and frame-count row before retaining the exact q=1 lane: 371 "
        "frame-local proposals on 295 of 2,259 eligible UI frames. It persists "
        "synchronized causal conditioned, signed-difference, Gamma-LS, strict-q1 "
        "threshold-exceedance, and exact sparse NMS provenance. No annotation file "
        "was opened and no biological or precision claim is made.\n",
    )
    _atomic_json(
        work / "status.json",
        {
            "status": "complete_exact_stage_replay_model_media_pending",
            "scientific_audit_complete": False,
            "annotation_sources_opened": False,
        },
    )
    recovered_failure_archived = _archive_recovered_failure(work)
    _atomic_json(
        work / "heartbeat.json",
        {
            "status": "complete_exact_stage_replay_model_media_pending",
            "stage": "replay_complete",
            "updated_at_utc": _utc_now(),
            "recovered_failure_archived": recovered_failure_archived,
        },
    )
    final_checkpoint = _read_json(work / "replay_checkpoint.json")
    final_checkpoint["status"] = "complete_exact_stage_replay"
    final_checkpoint["completed_at_utc"] = _utc_now()
    final_checkpoint["resume_required"] = False
    _atomic_json(work / "replay_checkpoint.json", final_checkpoint)
    _atomic_json(work / "artifact_index.json", _artifact_index(work))
    if destination.exists():
        raise FileExistsError(
            f"replay output appeared before atomic promotion: {destination}"
        )
    work.replace(destination)
    return {
        "output": str(destination),
        "status": summary["status"],
        "proposal_rows": EXPECTED_PROPOSALS,
        "distinct_proposal_frames": EXPECTED_PROPOSAL_FRAMES,
        "artifact_index_sha256": _sha256(destination / "artifact_index.json"),
    }


def _verify_stage_replay(
    replay_root: str | Path,
    *,
    expected_audit_output: str | Path | None = None,
    verify_dense_threshold: bool = True,
) -> dict[str, Any]:
    root = Path(replay_root).expanduser().resolve()
    indexed = _verify_complete_index(root)
    status = _read_json(root / "status.json")
    validation = _read_json(root / "validation.json")
    contract = _read_json(root / "run_contract.json")
    manifest = _read_json(root / "stage_manifest.json")
    reconciliation = _read_json(root / "ledger_reconciliation.json")
    if status.get("status") != "complete_exact_stage_replay_model_media_pending":
        raise FullRecordScientificAuditUnavailable("stage replay status changed")
    replay_check_values = validation.get("checks")
    if (
        not isinstance(replay_check_values, Mapping)
        or set(replay_check_values) != REPLAY_COMPLETION_CHECK_KEYS
        or any(value is not True for value in replay_check_values.values())
        or validation.get("all_checks_pass") is not True
    ):
        raise FullRecordScientificAuditUnavailable("stage replay validation failed")
    if reconciliation.get("status") != "exact_match":
        raise FullRecordScientificAuditUnavailable("ledger reconciliation is not exact")
    science = contract.get("science_contract")
    if not isinstance(science, Mapping) or _canonical_sha256(science) != str(
        contract.get("science_contract_sha256")
    ):
        raise FullRecordScientificAuditUnavailable("replay science contract changed")
    if expected_audit_output is not None and str(
        Path(str(science["audit_output"])).resolve()
    ) != str(Path(expected_audit_output).expanduser().resolve()):
        raise FullRecordScientificAuditUnavailable("audit destination differs from preflight")
    ledger = _load_operational_ledger(contract["source_ledger_root"])
    if dict(contract["source_ledger_hashes"]) != ledger["hashes"]:
        raise FullRecordScientificAuditUnavailable("source ledger binding changed")
    proposals = [
        _normalize_candidate(row)
        for row in _read_tsv(root / "q1_frame_level_proposals_exact_replay.tsv")
    ]
    counts = [
        _normalize_count(row)
        for row in _read_tsv(root / "q1_proposal_counts_by_frame_exact_replay.tsv")
    ]
    thresholds = [
        _normalize_threshold(row)
        for row in _read_tsv(root / "initial100_empirical_thresholds_exact_replay.tsv")
    ]
    surrogates = [
        _normalize_surrogate(row)
        for row in _read_tsv(root / "candidate_surrogates.tsv")
    ]
    assignments = [
        _normalize_assignment(row)
        for row in _read_tsv(root / "candidate_surrogate_assignments.tsv")
    ]
    if not _normalized_rows_equal(proposals, ledger["q1_candidates"]):
        raise FullRecordScientificAuditUnavailable("replay q1 proposal table changed")
    if not _normalized_rows_equal(counts, ledger["q1_counts"]):
        raise FullRecordScientificAuditUnavailable("replay q1 frame-count table changed")
    if not _normalized_rows_equal(thresholds, ledger["initial_thresholds"]):
        raise FullRecordScientificAuditUnavailable("replay threshold table changed")
    if surrogates != list(science["candidate_surrogates"]):
        raise FullRecordScientificAuditUnavailable("candidate surrogate panel changed")
    if _subset_hash(assignments) != str(
        science["candidate_assignment_content_sha256"]
    ):
        raise FullRecordScientificAuditUnavailable(
            "candidate surrogate assignments changed"
        )
    arrays = manifest.get("arrays")
    if not isinstance(arrays, Mapping) or set(arrays) != {
        "conditioned_current_frame",
        "difference_signed",
        "gamma_initial100_scale",
        "threshold_exceedance_q1_application",
    }:
        raise FullRecordScientificAuditUnavailable("stage manifest array inventory changed")
    expected_arrays = {
        "conditioned_current_frame": (
            STAGE_FILES["conditioned"],
            (TOTAL_FRAMES, FRAME_HEIGHT, FRAME_WIDTH),
            "float32",
        ),
        "difference_signed": (
            STAGE_FILES["difference"],
            (TOTAL_FRAMES, FRAME_HEIGHT, FRAME_WIDTH),
            "float32",
        ),
        "gamma_initial100_scale": (
            STAGE_FILES["gamma"],
            (TOTAL_FRAMES, FRAME_HEIGHT, FRAME_WIDTH),
            "float32",
        ),
        "threshold_exceedance_q1_application": (
            STAGE_FILES["threshold"],
            (EXPECTED_ELIGIBLE_FRAMES, FRAME_HEIGHT, FRAME_WIDTH),
            "bool",
        ),
    }
    loaded: dict[str, np.ndarray] = {}
    for key, (relative, shape, dtype) in expected_arrays.items():
        row = arrays[key]
        if (
            str(row.get("path")) != relative
            or tuple(row.get("shape_tyx", [])) != shape
            or str(row.get("dtype")) != dtype
        ):
            raise FullRecordScientificAuditUnavailable(
                f"stage array contract changed: {key}"
            )
        path = root / relative
        if _sha256(path) != str(row.get("file_sha256")):
            raise FullRecordScientificAuditUnavailable(
                f"stage manifest hash changed: {key}"
            )
        loaded[key] = np.load(path, mmap_mode="r", allow_pickle=False)
    difference = loaded["difference_signed"]
    conditioned = loaded["conditioned_current_frame"]
    if np.count_nonzero(difference[0]) != 0:
        raise FullRecordScientificAuditUnavailable("UI1 difference cold start changed")
    for index in (1, 99, 100, 1000, TOTAL_FRAMES - 1):
        if not np.array_equal(
            difference[index], conditioned[index] - conditioned[index - 1]
        ):
            raise FullRecordScientificAuditUnavailable(
                f"conditioned/difference alignment changed at UI {index + 1}"
            )
    gamma = loaded["gamma_initial100_scale"]
    threshold_map = loaded["threshold_exceedance_q1_application"]
    threshold_z = float(ledger["q1_threshold"]["threshold_z"])
    if verify_dense_threshold:
        for start in range(0, EXPECTED_ELIGIBLE_FRAMES, DEFAULT_CHUNK_FRAMES):
            stop = min(start + DEFAULT_CHUNK_FRAMES, EXPECTED_ELIGIBLE_FRAMES)
            if not np.array_equal(
                threshold_map[start:stop],
                gamma[
                    APPLICATION_UI[0] - 1 + start : APPLICATION_UI[0] - 1 + stop
                ]
                > threshold_z,
            ):
                raise FullRecordScientificAuditUnavailable(
                    "dense strict-threshold array differs from Gamma stage"
                )
    for row in proposals:
        ui = int(row["source_frame_ui"])
        if float(gamma[ui - 1, int(row["y_px"]), int(row["x_px"])]) != float(
            row["score"]
        ):
            raise FullRecordScientificAuditUnavailable(
                f"proposal score/Gamma alignment changed: {row['proposal_id']}"
            )
    return {
        "root": root,
        "indexed": indexed,
        "contract": contract,
        "science": dict(science),
        "manifest": manifest,
        "reconciliation": reconciliation,
        "ledger": ledger,
        "proposals": proposals,
        "counts": counts,
        "thresholds": thresholds,
        "surrogates": surrogates,
        "assignments": assignments,
        "arrays": loaded,
    }


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _crop_bounds(
    x_px: float,
    y_px: float,
    *,
    width: int,
    height: int,
    radius: int = 24,
) -> tuple[int, int, int, int]:
    x_center, y_center = int(round(x_px)), int(round(y_px))
    side_x = min(2 * radius + 1, width)
    side_y = min(2 * radius + 1, height)
    x0 = min(max(0, x_center - radius), width - side_x)
    y0 = min(max(0, y_center - radius), height - side_y)
    return x0, y0, x0 + side_x, y0 + side_y


def _draw_panel_markers(
    canvas: Image.Image,
    rows: Sequence[Mapping[str, Any]],
    *,
    panel_count: int,
    panel_size: tuple[int, int],
    header: int,
    source_bounds: tuple[int, int, int, int],
    radius: int,
    square: bool = False,
) -> None:
    x0, y0, x1, y1 = source_bounds
    panel_width, panel_height = panel_size
    draw = ImageDraw.Draw(canvas)
    for row in rows:
        x_px, y_px = float(row["x_px"]), float(row["y_px"])
        if not (x0 <= x_px < x1 and y0 <= y_px < y1):
            continue
        for panel in range(panel_count):
            center_x = panel * panel_width + (x_px - x0 + 0.5) * panel_width / (
                x1 - x0
            )
            center_y = header + (y_px - y0 + 0.5) * panel_height / (
                y1 - y0
            )
            box = (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            )
            if square:
                draw.rectangle(box, outline=ORANGE, width=2)
            else:
                draw.ellipse(box, outline=ORANGE, width=2)


def _full_record_panel(
    stages: Sequence[np.ndarray],
    *,
    raw_limits: tuple[float, float],
    conditioned_limits: tuple[float, float],
    difference_limit: float,
    gamma_limit: float,
    source_frame_ui: int,
    proposals: Sequence[Mapping[str, Any]],
) -> Image.Image:
    """Render one synchronized six-stage operational full-field frame."""

    if len(stages) != 6:
        raise ValueError("full-record panel requires exactly six stages")
    # 236:140 matches the 573:340 source aspect within 0.03%, preserving spatial
    # geometry and the visual meaning of the six-pixel NMS separation.
    panel_size, header = (236, 140), 46
    canvas = Image.new("RGB", (1416, 186), "black")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    titles = (
        "Acquired raw",
        "Causal conditioned",
        "Signed difference",
        "Radial Gamma-LS",
        "strict q1 exceedance",
        "framewise NMS6",
    )
    images = (
        _gray_unsigned(stages[0], raw_limits, panel_size),
        _gray_unsigned(stages[1], conditioned_limits, panel_size),
        _gray_signed(stages[2], difference_limit, panel_size),
        _gray_signed(stages[3], gamma_limit, panel_size),
        _gray_unsigned(stages[4], (0.0, 1.0), panel_size),
        _gray_signed(stages[5], gamma_limit, panel_size),
    )
    draw.text(
        (5, 5),
        f"operational full record | source UI {source_frame_ui}",
        fill="white",
        font=font,
    )
    for index, (stage_image, title) in enumerate(zip(images, titles, strict=True)):
        canvas.paste(stage_image, (index * panel_size[0], header))
        draw.text((index * panel_size[0] + 3, 26), title, fill="white", font=font)
    _draw_panel_markers(
        canvas,
        proposals,
        panel_count=6,
        panel_size=panel_size,
        header=header,
        source_bounds=(0, 0, FRAME_WIDTH, FRAME_HEIGHT),
        radius=4,
    )
    return canvas


def _full_record_close_panel(
    stages: Sequence[np.ndarray],
    *,
    raw_limits: tuple[float, float],
    conditioned_limits: tuple[float, float],
    difference_limit: float,
    gamma_limit: float,
    source_frame_ui: int,
    anchor: Mapping[str, Any],
    proposals: Sequence[Mapping[str, Any]],
    crop_radius: int = 24,
) -> Image.Image:
    """Render one model-only close-up with a fixed surrogate and current NMS."""

    if len(stages) != 6:
        raise ValueError("close panel requires exactly six stages")
    x0, y0, x1, y1 = _crop_bounds(
        float(anchor["x_px"]),
        float(anchor["y_px"]),
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        radius=crop_radius,
    )
    crops = [np.asarray(stage)[y0:y1, x0:x1] for stage in stages]
    panel_size, header = (96, 96), 36
    canvas = Image.new("RGB", (576, 132), "black")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    titles = ("Raw", "Conditioned", "Difference", "Gamma-LS", "q1", "NMS6")
    images = (
        _gray_unsigned(crops[0], raw_limits, panel_size),
        _gray_unsigned(crops[1], conditioned_limits, panel_size),
        _gray_signed(crops[2], difference_limit, panel_size),
        _gray_signed(crops[3], gamma_limit, panel_size),
        _gray_unsigned(crops[4], (0.0, 1.0), panel_size),
        _gray_signed(crops[5], gamma_limit, panel_size),
    )
    draw.text(
        (4, 3),
        f"{anchor['model_roi_id']} | source UI {source_frame_ui}",
        fill="white",
        font=font,
    )
    for index, (stage_image, title) in enumerate(zip(images, titles, strict=True)):
        canvas.paste(stage_image, (index * panel_size[0], header))
        draw.text((index * panel_size[0] + 3, 20), title, fill="white", font=font)
    _draw_panel_markers(
        canvas,
        [anchor],
        panel_count=6,
        panel_size=panel_size,
        header=header,
        source_bounds=(x0, y0, x1, y1),
        radius=6,
    )
    _draw_panel_markers(
        canvas,
        proposals,
        panel_count=6,
        panel_size=panel_size,
        header=header,
        source_bounds=(x0, y0, x1, y1),
        radius=3,
        square=True,
    )
    return canvas


def _write_representative_stage_packet(
    path: Path,
    metadata_path: Path,
    preview_path: Path,
    *,
    movie: np.ndarray,
    conditioned: np.ndarray,
    difference: np.ndarray,
    gamma: np.ndarray,
    threshold: np.ndarray,
    representative: Mapping[str, Any],
    frame_proposals: Sequence[Mapping[str, Any]],
    source_movie_sha256: str,
    threshold_z: float,
    crop_radius: int = 40,
) -> dict[str, Any]:
    """Persist a same-crop packet for the exact operational framewise head."""

    source_ui = int(representative["source_frame_ui"])
    if not APPLICATION_UI[0] <= source_ui <= APPLICATION_UI[1]:
        raise FullRecordScientificAuditUnavailable("representative is outside application")
    x_px, y_px = int(representative["x_px"]), int(representative["y_px"])
    x0, y0, x1, y1 = _crop_bounds(
        x_px,
        y_px,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        radius=crop_radius,
    )
    stage_index = source_ui - 1
    threshold_index = source_ui - APPLICATION_UI[0]
    nms_mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
    for row in frame_proposals:
        nms_mask[int(row["y_px"]), int(row["x_px"])] = 1
    arrays = {
        "raw": np.asarray(movie[stage_index, y0:y1, x0:x1], dtype=np.uint16),
        "conditioned_previous": np.asarray(
            conditioned[stage_index - 1, y0:y1, x0:x1], dtype=np.float32
        ),
        "conditioned_current": np.asarray(
            conditioned[stage_index, y0:y1, x0:x1], dtype=np.float32
        ),
        "difference": np.asarray(
            difference[stage_index, y0:y1, x0:x1], dtype=np.float32
        ),
        "gamma_ls": np.asarray(
            gamma[stage_index, y0:y1, x0:x1], dtype=np.float32
        ),
        "threshold_exceedance": np.asarray(
            threshold[threshold_index, y0:y1, x0:x1], dtype=np.uint8
        ),
        "proposals": np.asarray(nms_mask[y0:y1, x0:x1], dtype=np.uint8),
    }
    maximum_error = float(
        np.max(
            np.abs(
                arrays["conditioned_current"]
                - arrays["conditioned_previous"]
                - arrays["difference"]
            )
        )
    )
    if maximum_error != 0.0:
        raise FullRecordScientificAuditUnavailable(
            f"representative conditioned/difference alignment changed: {maximum_error}"
        )
    if not np.array_equal(
        arrays["threshold_exceedance"], arrays["gamma_ls"] > float(threshold_z)
    ):
        raise FullRecordScientificAuditUnavailable(
            "representative strict-threshold alignment changed"
        )
    if not bool(arrays["proposals"][y_px - y0, x_px - x0]):
        raise FullRecordScientificAuditUnavailable(
            "representative proposal is absent from exact NMS mask"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    raw_limits = tuple(map(float, np.percentile(arrays["raw"], [1.0, 99.0])))
    conditioned_limits = tuple(
        map(float, np.percentile(arrays["conditioned_current"], [1.0, 99.0]))
    )
    difference_limit = max(float(np.percentile(np.abs(arrays["difference"]), 99.0)), 1e-6)
    gamma_limit = max(float(np.percentile(np.abs(arrays["gamma_ls"]), 99.0)), 1e-6)
    local_proposals = [
        row
        for row in frame_proposals
        if x0 <= int(row["x_px"]) < x1 and y0 <= int(row["y_px"]) < y1
    ]
    # The close-panel helper consumes full-field coordinates; render the
    # already extracted packet directly to avoid a second coordinate system.
    panel_size, header = (128, 128), 44
    preview = Image.new("RGB", (768, 172), "black")
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    preview_images = (
        _gray_unsigned(arrays["raw"], raw_limits, panel_size),
        _gray_unsigned(arrays["conditioned_current"], conditioned_limits, panel_size),
        _gray_signed(arrays["difference"], difference_limit, panel_size),
        _gray_signed(arrays["gamma_ls"], gamma_limit, panel_size),
        _gray_unsigned(arrays["threshold_exceedance"], (0.0, 1.0), panel_size),
        _gray_signed(arrays["gamma_ls"], gamma_limit, panel_size),
    )
    titles = ("Raw", "Conditioned", "Difference", "Gamma-LS", "q1 threshold", "NMS6")
    draw.text(
        (4, 4),
        f"source UI {source_ui}; same crop; exact operational q1 head",
        fill="white",
        font=font,
    )
    for index, (stage_image, title) in enumerate(zip(preview_images, titles, strict=True)):
        preview.paste(stage_image, (index * panel_size[0], header))
        draw.text((index * panel_size[0] + 3, 25), title, fill="white", font=font)
    for row in local_proposals:
        cx = 5 * panel_size[0] + (int(row["x_px"]) - x0 + 0.5) * panel_size[0] / (x1 - x0)
        cy = header + (int(row["y_px"]) - y0 + 0.5) * panel_size[1] / (y1 - y0)
        draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), outline=ORANGE, width=3)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(preview_path)
    metadata = {
        "schema_version": 1,
        "selection_rule": "maximum_q1_score_then_source_frame_rank_proposal_id",
        "proposal_id": str(representative["proposal_id"]),
        "source_frame_ui": source_ui,
        "center_x_px": x_px,
        "center_y_px": y_px,
        "crop_xyxy_half_open": [x0, y0, x1, y1],
        "coordinate_convention": "x=column,y=row",
        "source_movie_sha256": source_movie_sha256,
        "raw_provenance": "exact uint16 acquired frame from hash-bound source movie",
        "conditioned_provenance": "persisted exact causal replay arrays",
        "difference_semantics": "conditioned_current_minus_conditioned_previous",
        "gamma_ls_semantics": "signed radial h15_g7_n9_mode7p5 with initial100 scale floor",
        "threshold_exceedance_semantics": "Gamma-LS strictly greater than exact q1 threshold",
        "proposals_semantics": (
            "all exact deterministic Euclidean NMS6 peaks from the full source frame, "
            "cropped only after NMS"
        ),
        "threshold_z": float(threshold_z),
        "nms_distance_px": NMS_DISTANCE_PX,
        "full_frame_nms_proposal_count": len(frame_proposals),
        "crop_nms_proposal_count": len(local_proposals),
        "difference_alignment_max_abs_error": maximum_error,
        "keys": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
        "npz": path.name,
        "preview": preview_path.name,
        "annotation_sources_opened": False,
        "claim_boundary": "frame-local model proposal only; no biological identity or precision",
    }
    _atomic_json(metadata_path, metadata)
    return metadata


def _display_limits(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.percentile(np.asarray(values, dtype=np.float32), [0.5, 99.8])
    return float(lower), max(float(upper), float(lower) + 1e-6)


def _write_model_trace(
    path: Path,
    *,
    source_ui: np.ndarray,
    traces: Mapping[str, np.ndarray],
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ("raw", "conditioned", "difference", "gamma", "threshold", "nms")
    labels = (
        "Acquired raw",
        "Causal conditioned",
        "Signed difference",
        "Radial Gamma-LS",
        "q1 exceedance",
        "NMS occurrence",
    )
    figure, axes = plt.subplots(
        6, 1, figsize=(9, 9.2), sharex=True, constrained_layout=True
    )
    for axis, key, label in zip(axes, keys, labels, strict=True):
        axis.plot(source_ui, traces[key], color="#ff9123", linewidth=0.65)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("source frame (UI one-based)")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def _probe_operational_video(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames,avg_frame_rate,duration:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise FullRecordScientificAuditUnavailable(result.stderr[-2000:])
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise FullRecordScientificAuditUnavailable(
            f"video stream count invalid: {path}"
        )
    stream = streams[0]
    numerator, denominator = str(stream.get("avg_frame_rate", "0/1")).split("/", 1)
    frame_rate = float(numerator) / float(denominator)
    duration = float(
        stream.get("duration") or payload.get("format", {}).get("duration") or 0.0
    )
    return {**stream, "frame_rate": frame_rate, "duration_seconds": duration}


def _video_is_complete(
    path: Path, *, width: int, height: int, expected_frames: int
) -> bool:
    if not path.is_file():
        return False
    try:
        probe = _probe_operational_video(path)
    except Exception:
        return False
    if (
        int(probe.get("width") or 0) != int(width)
        or int(probe.get("height") or 0) != int(height)
        or int(probe.get("nb_frames") or 0) != int(expected_frames)
        or str(probe.get("codec_name")) != "h264"
        or not math.isclose(float(probe["frame_rate"]), 50.0, abs_tol=1e-9)
        or not math.isclose(
            float(probe["duration_seconds"]),
            float(expected_frames) / 50.0,
            rel_tol=0.0,
            abs_tol=0.025,
        )
    ):
        return False
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
    )
    return decoded.returncode == 0


def _render_checkpoint(
    work: Path, *, audit_contract_sha256: str
) -> dict[str, Any]:
    path = work / "render_checkpoint.json"
    if not path.is_file():
        return {
            "schema_version": 1,
            "status": "render_incomplete",
            "audit_contract_sha256": audit_contract_sha256,
            "completed_videos": {},
        }
    checkpoint = _read_json(path)
    if checkpoint.get("audit_contract_sha256") != audit_contract_sha256:
        raise FullRecordScientificAuditUnavailable(
            "render checkpoint implementation/contract binding changed"
        )
    if not isinstance(checkpoint.get("completed_videos"), dict):
        raise FullRecordScientificAuditUnavailable("render checkpoint is invalid")
    return checkpoint


def _checkpoint_completed_video(
    work: Path,
    *,
    audit_contract_sha256: str,
    path: Path,
    width: int,
    height: int,
    expected_frames: int,
) -> None:
    if not _video_is_complete(
        path, width=width, height=height, expected_frames=expected_frames
    ):
        raise FullRecordScientificAuditUnavailable(
            f"completed video failed decode/geometry validation: {path}"
        )
    relative = path.relative_to(work).as_posix()
    probe = _probe_operational_video(path)
    checkpoint = _render_checkpoint(
        work, audit_contract_sha256=audit_contract_sha256
    )
    checkpoint["updated_at_utc"] = _utc_now()
    checkpoint["completed_videos"][relative] = {
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "width": int(width),
        "height": int(height),
        "expected_frames": int(expected_frames),
        "codec": str(probe["codec_name"]),
        "frame_rate": float(probe["frame_rate"]),
        "duration_seconds": float(probe["duration_seconds"]),
        "full_decode_passed_before_checkpoint": True,
    }
    _atomic_json(work / "render_checkpoint.json", checkpoint)


def _checkpointed_video_is_complete(
    work: Path,
    *,
    audit_contract_sha256: str,
    path: Path,
    width: int,
    height: int,
    expected_frames: int,
) -> bool:
    checkpoint = _render_checkpoint(
        work, audit_contract_sha256=audit_contract_sha256
    )
    relative = path.relative_to(work).as_posix()
    frozen = checkpoint["completed_videos"].get(relative)
    if not isinstance(frozen, Mapping) or not path.is_file():
        return False
    if (
        path.stat().st_size != int(frozen.get("size_bytes", -1))
        or _sha256(path) != str(frozen.get("sha256"))
        or int(frozen.get("width", -1)) != int(width)
        or int(frozen.get("height", -1)) != int(height)
        or int(frozen.get("expected_frames", -1)) != int(expected_frames)
    ):
        raise FullRecordScientificAuditUnavailable(
            f"checkpointed video changed before resume: {relative}"
        )
    return _video_is_complete(
        path, width=width, height=height, expected_frames=expected_frames
    )


def _section_scientific_media(root: Path, section: str) -> list[str]:
    suffixes = {".mp4", ".png", ".csv", ".tsv", ".npz"}
    directory = root / section
    if not directory.is_dir():
        raise FullRecordScientificAuditUnavailable(
            f"scientific-audit section is missing: {section}"
        )
    return [
        path.relative_to(root).as_posix()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in suffixes
    ]


def run_model_only_scientific_audit(
    replay_root: str | Path,
    audit_output: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Render the complete label-free Model section for the exact q=1 stream."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    destination = Path(audit_output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    replay = _verify_stage_replay(
        replay_root,
        expected_audit_output=destination,
        verify_dense_threshold=True,
    )
    science = replay["science"]
    movie_path = Path(str(replay["manifest"]["source_movie"]["path"]))
    if not movie_path.is_file() or _sha256(movie_path) != str(
        replay["manifest"]["source_movie"]["file_sha256"]
    ):
        raise FullRecordScientificAuditUnavailable("source movie hash changed")
    movie = np.load(movie_path, mmap_mode="r", allow_pickle=False)
    conditioned = replay["arrays"]["conditioned_current_frame"]
    difference = replay["arrays"]["difference_signed"]
    gamma = replay["arrays"]["gamma_initial100_scale"]
    threshold = replay["arrays"]["threshold_exceedance_q1_application"]
    proposals = replay["proposals"]
    surrogates = replay["surrogates"]
    assignments = replay["assignments"]
    if len(surrogates) != DEFAULT_ANCHOR_LIMIT:
        raise FullRecordScientificAuditUnavailable("model audit requires 24 surrogates")
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in proposals:
        by_frame.setdefault(int(row["source_frame_ui"]), []).append(row)
    assignment_by_proposal = {
        str(row["proposal_id"]): row for row in assignments
    }

    work = destination.parent / f".{destination.name}.audit-work"
    repository = Path(__file__).resolve().parents[3]
    renderer_test = repository / "tests/test_gamma_ls_full_recording_scientific_audit.py"
    audit_contract = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "run_type": "resumable_model_only_scientific_audit_renderer",
        "source_replay_root": str(replay["root"]),
        "source_replay_artifact_index_sha256": replay["indexed"][
            "artifact_index_sha256"
        ],
        "audit_output": str(destination),
        "candidate_surrogate_count": len(surrogates),
        "renderer_implementation": {
            "module_path": "repo://neurobench/experiments/gamma_ls_difference/full_recording_scientific_audit.py",
            "module_sha256": _sha256(Path(__file__).resolve()),
            "test_path": "repo://tests/test_gamma_ls_full_recording_scientific_audit.py",
            "test_sha256": _sha256(renderer_test),
        },
        "annotation_sources_opened": False,
    }
    audit_contract_sha256 = _canonical_sha256(audit_contract)
    if resume:
        if not work.is_dir():
            raise FileNotFoundError(f"audit work root is missing: {work}")
        if _read_json(work / "run_contract.json") != audit_contract:
            raise FullRecordScientificAuditUnavailable(
                "audit checkpoint contract differs from current replay"
            )
    else:
        if work.exists():
            raise FileExistsError(
                f"audit work root exists; use render --resume after inspection: {work}"
            )
        work.mkdir()
        _atomic_json(work / "run_contract.json", audit_contract)
        _atomic_json(
            work / "render_checkpoint.json",
            _render_checkpoint(
                work, audit_contract_sha256=audit_contract_sha256
            ),
        )
    os.environ.setdefault("MPLCONFIGDIR", str(work / ".matplotlib"))
    expert_root = work / "1_Expert_Annotations"
    model_root = work / "2_Model_Annotations"
    comparison_root = work / "3_Comparison"
    for path in (
        expert_root,
        model_root / "videos/closeups",
        model_root / "figures/traces",
        model_root / "metadata",
        model_root / "representative_stage",
        comparison_root,
        work / "projection_checks",
    ):
        path.mkdir(parents=True, exist_ok=True)
    try:
        _heartbeat(work / "heartbeat.json", "derive_label_free_display_scales")
        calibration_stride = (slice(0, CALIBRATION_UI[1]), slice(None, None, 4), slice(None, None, 4))
        raw_limits = _display_limits(movie[calibration_stride])
        conditioned_limits = _display_limits(conditioned[calibration_stride])
        difference_limit = max(
            float(np.percentile(np.abs(difference[calibration_stride]), 99.8)),
            1e-6,
        )
        gamma_limit = max(
            float(np.percentile(np.abs(gamma[calibration_stride]), 99.8)),
            1e-6,
        )
        display_scales = {
            "schema_version": 1,
            "population": "UI1-100 calibration frames spatially subsampled every 4 pixels",
            "labels_used": False,
            "raw": {"type": "fixed_linear", "vmin": raw_limits[0], "vmax": raw_limits[1]},
            "conditioned": {
                "type": "fixed_linear",
                "vmin": conditioned_limits[0],
                "vmax": conditioned_limits[1],
            },
            "difference_signed": {
                "type": "fixed_symmetric_zero_midgray",
                "absolute_limit": difference_limit,
            },
            "gamma_ls_signed": {
                "type": "fixed_symmetric_zero_midgray",
                "absolute_limit": gamma_limit,
            },
            "threshold_exceedance": {"type": "binary_black_white"},
            "nms_marker": {"color": "orange", "rgb": list(ORANGE)},
        }
        _atomic_json(work / "display_scales.json", display_scales)

        representative = _normalize_candidate(science["representative_stage_proposal"])
        representative_frame_rows = by_frame[int(representative["source_frame_ui"])]
        representative_metadata = _write_representative_stage_packet(
            model_root / "representative_stage/full_record_q1_stage_crop.npz",
            model_root / "representative_stage/full_record_q1_stage_crop.json",
            model_root / "representative_stage/full_record_q1_stage_crop.png",
            movie=movie,
            conditioned=conditioned,
            difference=difference,
            gamma=gamma,
            threshold=threshold,
            representative=representative,
            frame_proposals=representative_frame_rows,
            source_movie_sha256=str(
                replay["manifest"]["source_movie"]["file_sha256"]
            ),
            threshold_z=float(science["threshold_z"]),
        )

        _heartbeat(work / "heartbeat.json", "render_model_full_field")
        full_video_path = (
            model_root / "videos/model_q1_operational_sequential_full_field.mp4"
        )
        if not _checkpointed_video_is_complete(
            work,
            audit_contract_sha256=audit_contract_sha256,
            path=full_video_path,
            width=1416,
            height=186,
            expected_frames=EXPECTED_ELIGIBLE_FRAMES,
        ):
            full_video = _VideoWriter(full_video_path, 1416, 186, 50.0)
            for source_ui in range(APPLICATION_UI[0], APPLICATION_UI[1] + 1):
                stage_index = source_ui - 1
                threshold_index = source_ui - APPLICATION_UI[0]
                frame_rows = by_frame.get(source_ui, [])
                panel = _full_record_panel(
                    (
                        movie[stage_index],
                        conditioned[stage_index],
                        difference[stage_index],
                        gamma[stage_index],
                        threshold[threshold_index],
                        gamma[stage_index],
                    ),
                    raw_limits=raw_limits,
                    conditioned_limits=conditioned_limits,
                    difference_limit=difference_limit,
                    gamma_limit=gamma_limit,
                    source_frame_ui=source_ui,
                    proposals=frame_rows,
                )
                full_video.write(np.asarray(panel, dtype=np.uint8))
            full_video.close()
            _checkpoint_completed_video(
                work,
                audit_contract_sha256=audit_contract_sha256,
                path=full_video_path,
                width=1416,
                height=186,
                expected_frames=EXPECTED_ELIGIBLE_FRAMES,
            )

        video_manifest: list[dict[str, Any]] = [
            {
                "path": (
                    "2_Model_Annotations/videos/"
                    "model_q1_operational_sequential_full_field.mp4"
                ),
                "kind": "model_full_field_exact_operational_q1",
                "expected_frames": EXPECTED_ELIGIBLE_FRAMES,
                "expected_width": 1416,
                "expected_height": 186,
                "expected_frame_rate": 50.0,
                "expected_duration_seconds": EXPECTED_ELIGIBLE_FRAMES / 50.0,
                "expected_codec": "h264",
                "sample_encoded_frame": int(representative["source_frame_ui"])
                - APPLICATION_UI[0],
                "expected_marker": "orange",
            }
        ]
        _heartbeat(work / "heartbeat.json", "render_model_closeups")
        for batch_start in range(0, len(surrogates), 4):
            batch = surrogates[batch_start : batch_start + 4]
            pending_batch = [
                anchor
                for anchor in batch
                if not _checkpointed_video_is_complete(
                    work,
                    audit_contract_sha256=audit_contract_sha256,
                    path=model_root
                    / f"videos/closeups/{_safe_id(str(anchor['model_roi_id']))}.mp4",
                    width=576,
                    height=132,
                    expected_frames=EXPECTED_ELIGIBLE_FRAMES,
                )
            ]
            writers = {
                str(anchor["model_roi_id"]): _VideoWriter(
                    model_root
                    / f"videos/closeups/{_safe_id(str(anchor['model_roi_id']))}.mp4",
                    576,
                    132,
                    50.0,
                )
                for anchor in pending_batch
            }
            if pending_batch:
                for source_ui in range(APPLICATION_UI[0], APPLICATION_UI[1] + 1):
                    stage_index = source_ui - 1
                    threshold_index = source_ui - APPLICATION_UI[0]
                    frame_rows = by_frame.get(source_ui, [])
                    stages = (
                        movie[stage_index],
                        conditioned[stage_index],
                        difference[stage_index],
                        gamma[stage_index],
                        threshold[threshold_index],
                        gamma[stage_index],
                    )
                    for anchor in pending_batch:
                        panel = _full_record_close_panel(
                            stages,
                            raw_limits=raw_limits,
                            conditioned_limits=conditioned_limits,
                            difference_limit=difference_limit,
                            gamma_limit=gamma_limit,
                            source_frame_ui=source_ui,
                            anchor=anchor,
                            proposals=frame_rows,
                        )
                        writers[str(anchor["model_roi_id"])].write(
                            np.asarray(panel, dtype=np.uint8)
                        )
            for anchor in pending_batch:
                model_roi_id = str(anchor["model_roi_id"])
                writers[model_roi_id].close()
                _checkpoint_completed_video(
                    work,
                    audit_contract_sha256=audit_contract_sha256,
                    path=model_root
                    / f"videos/closeups/{_safe_id(model_roi_id)}.mp4",
                    width=576,
                    height=132,
                    expected_frames=EXPECTED_ELIGIBLE_FRAMES,
                )
            for anchor in batch:
                model_roi_id = str(anchor["model_roi_id"])
                video_manifest.append(
                    {
                        "path": (
                            "2_Model_Annotations/videos/closeups/"
                            f"{_safe_id(model_roi_id)}.mp4"
                        ),
                        "kind": "model_candidate_surrogate_closeup",
                        "model_roi_id": model_roi_id,
                        "expected_frames": EXPECTED_ELIGIBLE_FRAMES,
                        "expected_width": 576,
                        "expected_height": 132,
                        "expected_frame_rate": 50.0,
                        "expected_duration_seconds": EXPECTED_ELIGIBLE_FRAMES
                        / 50.0,
                        "expected_codec": "h264",
                        "sample_encoded_frame": 0,
                        "expected_marker": "orange",
                    }
                )
            _heartbeat(
                work / "heartbeat.json",
                "render_model_closeups",
                completed_surrogates=min(batch_start + 4, len(surrogates)),
                total_surrogates=len(surrogates),
            )

        _heartbeat(work / "heartbeat.json", "render_model_traces")
        source_ui_values = np.arange(
            APPLICATION_UI[0], APPLICATION_UI[1] + 1, dtype=np.int64
        )
        for anchor in surrogates:
            model_roi_id = str(anchor["model_roi_id"])
            x_px, y_px = int(anchor["x_px"]), int(anchor["y_px"])
            nms_trace = np.zeros(EXPECTED_ELIGIBLE_FRAMES, dtype=np.uint8)
            assigned_rows = []
            for row in proposals:
                assignment = assignment_by_proposal[str(row["proposal_id"])]
                if str(assignment["model_roi_id"]) == model_roi_id:
                    nms_trace[int(row["source_frame_ui"]) - APPLICATION_UI[0]] += 1
                    assigned_rows.append(row)
            traces = {
                "raw": np.asarray(movie[APPLICATION_UI[0] - 1 :, y_px, x_px]),
                "conditioned": np.asarray(
                    conditioned[APPLICATION_UI[0] - 1 :, y_px, x_px]
                ),
                "difference": np.asarray(
                    difference[APPLICATION_UI[0] - 1 :, y_px, x_px]
                ),
                "gamma": np.asarray(gamma[APPLICATION_UI[0] - 1 :, y_px, x_px]),
                "threshold": np.asarray(threshold[:, y_px, x_px], dtype=np.uint8),
                "nms": nms_trace,
            }
            _write_model_trace(
                model_root / f"figures/traces/{_safe_id(model_roi_id)}.png",
                source_ui=source_ui_values,
                traces=traces,
                title=(
                    f"{model_roi_id} at x={x_px}, y={y_px}; label-free review surrogate"
                ),
            )
            _atomic_json(
                model_root / f"metadata/{_safe_id(model_roi_id)}.json",
                {
                    "schema_version": 1,
                    **anchor,
                    "assigned_proposal_ids": [row["proposal_id"] for row in assigned_rows],
                    "trace_semantics": {
                        "raw_through_threshold": "exact pixel at fixed surrogate coordinate",
                        "nms": "count of exact q1 proposals assigned by frozen 3px surrogate consolidation",
                    },
                    "biological_identity_claimed": False,
                    "annotation_sources_opened": False,
                },
            )

        model_occurrences = [
            {
                **row,
                "model_roi_id": str(
                    assignment_by_proposal[str(row["proposal_id"])]["model_roi_id"]
                ),
                "selected_for_closeup": bool(
                    assignment_by_proposal[str(row["proposal_id"])][
                        "selected_for_closeup"
                    ]
                ),
            }
            for row in proposals
        ]
        _atomic_csv(
            model_root / "model_occurrences.csv",
            model_occurrences,
            fieldnames=tuple(model_occurrences[0]),
        )
        _atomic_csv(
            model_root / "candidate_surrogates.csv",
            surrogates,
            fieldnames=tuple(surrogates[0]),
        )

        _heartbeat(work / "heartbeat.json", "render_projection_and_timeline")
        raw_projection = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint16)
        gamma_projection = np.full(
            (FRAME_HEIGHT, FRAME_WIDTH), -np.inf, dtype=np.float32
        )
        for start in range(0, TOTAL_FRAMES, DEFAULT_CHUNK_FRAMES):
            stop = min(start + DEFAULT_CHUNK_FRAMES, TOTAL_FRAMES)
            raw_projection = np.maximum(
                raw_projection, np.max(movie[start:stop], axis=0)
            )
            gamma_start = max(start, APPLICATION_UI[0] - 1)
            if gamma_start < stop:
                gamma_projection = np.maximum(
                    gamma_projection, np.max(gamma[gamma_start:stop], axis=0)
                )
        proposal_density = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint16)
        for row in proposals:
            proposal_density[int(row["y_px"]), int(row["x_px"])] += 1
        projection_npz = work / "projection_checks/model_projection_arrays.npz"
        temporary_projection = projection_npz.with_name(
            f".{projection_npz.stem}.{uuid.uuid4().hex}.partial.npz"
        )
        np.savez_compressed(
            temporary_projection,
            raw_max=raw_projection,
            gamma_max=gamma_projection,
            proposal_density=proposal_density,
        )
        temporary_projection.replace(projection_npz)
        figure, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
        axes[0].imshow(raw_projection, cmap="gray", vmin=raw_limits[0], vmax=raw_limits[1])
        axes[0].set_title("Acquired raw max projection")
        axes[1].imshow(gamma_projection, cmap="gray", vmin=0, vmax=max(gamma_limit, 1e-6))
        axes[1].set_title("Operational Gamma-LS max; all q1 proposals")
        axes[2].imshow(proposal_density, cmap="gray", vmin=0, vmax=max(int(proposal_density.max()), 1))
        axes[2].set_title("Frame-local proposal density")
        for row in proposals:
            axes[1].add_patch(
                Circle(
                    (float(row["x_px"]), float(row["y_px"])),
                    2.0,
                    fill=False,
                    edgecolor="#ff9123",
                    linewidth=0.35,
                    alpha=0.55,
                )
            )
        for anchor in surrogates:
            for axis in (axes[1], axes[2]):
                axis.add_patch(
                    Circle(
                        (float(anchor["x_px"]), float(anchor["y_px"])),
                        6.0,
                        fill=False,
                        edgecolor="#ff9123",
                        linewidth=1.0,
                    )
                )
        for axis in axes:
            axis.set_xlim(-0.5, FRAME_WIDTH - 0.5)
            axis.set_ylim(FRAME_HEIGHT - 0.5, -0.5)
            axis.set_xlabel("x = column")
            axis.set_ylabel("y = row")
        figure.suptitle("Model-only coordinate/projection audit; no labels loaded")
        figure.savefig(
            work / "projection_checks/model_projection_coordinate_check.png",
            dpi=125,
        )
        plt.close(figure)
        count_by_frame = {
            int(row["source_frame_ui"]): int(row["proposal_count"])
            for row in replay["counts"]
        }
        count_values = np.asarray(
            [count_by_frame[ui] for ui in source_ui_values], dtype=np.int64
        )
        figure, axis = plt.subplots(figsize=(11, 3.6), constrained_layout=True)
        axis.plot(source_ui_values, count_values, color="#ff9123", linewidth=0.75)
        axis.set(
            title="Exact operational q1 frame-local proposal stream",
            xlabel="source frame (UI one-based)",
            ylabel="NMS6 proposals per frame",
        )
        axis.grid(alpha=0.2)
        figure.savefig(model_root / "figures/proposal_timeline.png", dpi=130)
        plt.close(figure)

        _atomic_text(
            expert_root / "README.md",
            "# Expert Annotations — not applicable\n\n"
            "This operational audit opened no annotation source. Expert-only media "
            "are intentionally absent.\n",
        )
        _atomic_text(
            model_root / "README.md",
            "# Model Annotations\n\n"
            "Orange marks the exact frozen q=1 frame-local NMS6 proposal stream or "
            "one of 24 deterministic label-free spatial review surrogates. The full-"
            "field and close-up videos synchronize acquired raw, causal conditioning, "
            "signed difference, radial Gamma-LS, strict threshold exceedance, and "
            "framewise Euclidean NMS6. Surrogates are not biological identities.\n",
        )
        _atomic_text(
            comparison_root / "README.md",
            "# Comparison — not applicable\n\n"
            "No expert/model comparison is possible because no label source was "
            "opened. Comparison figures, videos, and precision claims are intentionally "
            "absent.\n",
        )
        _atomic_json(work / "video_manifest.json", {"schema_version": 1, "videos": video_manifest})
        inventory = {
            "schema_version": 1,
            "status": "complete_model_only_inventory",
            "section_applicability": {
                "1_Expert_Annotations": "not_applicable_no_annotation_source_opened",
                "2_Model_Annotations": "required_and_complete",
                "3_Comparison": "not_applicable_no_annotation_join",
            },
            "expert_media_count": 0,
            "comparison_media_count": 0,
            "model_full_field_video_count": 1,
            "model_closeup_video_count": len(surrogates),
            "model_trace_figure_count": len(surrogates),
            "model_occurrence_rows": len(model_occurrences),
            "model_surrogate_rows": len(surrogates),
            "projection_check_count": 1,
            "representative_stage_packet_count": 1,
            "complete": True,
        }
        _atomic_json(work / "inventory.json", inventory)
        summary = {
            "schema_version": 1,
            "status": "rendered_pending_visual_inspection",
            "experiment_id": EXPERIMENT_ID,
            "evidence_role": "full_record_operational_model_only_scientific_audit",
            "source_replay_root": str(replay["root"]),
            "source_replay_artifact_index_sha256": replay["indexed"]["artifact_index_sha256"],
            "source_ledger_root": str(replay["ledger"]["root"]),
            "source_ledger_artifact_index_sha256": replay["ledger"]["indexed"]["artifact_index_sha256"],
            "calibration_interval_ui": list(CALIBRATION_UI),
            "calibration_windows_ui": [[1, 50], [51, 100]],
            "calibration_frames_assumed_event_free": False,
            "application_interval_ui": list(APPLICATION_UI),
            "representation": REPRESENTATION,
            "context_id": CONTEXT_ID,
            "target_nms_peaks_per_calibration_unit": TARGET_BURDEN,
            "threshold_z": float(science["threshold_z"]),
            "nms_distance_and_border_px": NMS_DISTANCE_PX,
            "proposal_row_count": len(proposals),
            "distinct_proposal_frame_count": len(by_frame),
            "eligible_frame_count": EXPECTED_ELIGIBLE_FRAMES,
            "candidate_cluster_count": int(science["candidate_cluster_count"]),
            "candidate_surrogate_count": len(surrogates),
            "temporal_linking_applied": False,
            "annotation_sources_opened": False,
            "expert_section_applicable": False,
            "comparison_section_applicable": False,
            "scientific_audit_complete": False,
            "representative_stage": representative_metadata,
            "claim_boundary": (
                "371 frame-local automated proposals on 295 of 2259 eligible frames; "
                "not unique biological detections and no precision, specificity, "
                "false-positive-rate, or biological-identity claim"
            ),
        }
        _atomic_json(work / "summary.json", summary)
        _atomic_json(
            work / "llm_context.json",
            {
                "schema_version": 1,
                "entrypoint": "summary.json",
                "annotation_separation": "strict_no_annotation_source_opened",
                "section_applicability": inventory["section_applicability"],
                "model_stage_sequence": [
                    "acquired raw",
                    "causal Gaussian and EMA conditioned frame",
                    "signed adjacent conditioned-frame difference",
                    "signed radial h15/g7/n9/mode7.5 Gamma-LS",
                    "strict q1 threshold exceedance",
                    "framewise 6px-border deterministic Euclidean NMS",
                ],
                "coordinate_convention": "x=column,y=row",
                "frame_convention": "UI one-based inclusive",
                "expected_and_observed_counts": summary,
                "ledger_reconciliation": "ledger_reconciliation.json",
                "primary_tables": [
                    "2_Model_Annotations/model_occurrences.csv",
                    "2_Model_Annotations/candidate_surrogates.csv",
                ],
                "representative_artifacts": [
                    "2_Model_Annotations/videos/model_q1_operational_sequential_full_field.mp4",
                    "projection_checks/model_projection_coordinate_check.png",
                    "2_Model_Annotations/figures/proposal_timeline.png",
                    "2_Model_Annotations/representative_stage/full_record_q1_stage_crop.npz",
                    "2_Model_Annotations/representative_stage/full_record_q1_stage_crop.json",
                    "2_Model_Annotations/representative_stage/full_record_q1_stage_crop.png",
                ],
                "limitations": [
                    "no labels were opened, so expert and comparison sections are not applicable",
                    "frame-local proposals are not unique biological detections",
                    "the 24 review surrogates are label-free media anchors, not biological identities",
                    "no precision, specificity, false-positive rate, or population generalization is identified",
                ],
            },
        )
        shutil.copy2(
            replay["root"] / "ledger_reconciliation.json",
            work / "ledger_reconciliation.json",
        )
        _atomic_text(
            work / "REPORT.md",
            "# Full-record operational q=1 model-only scientific audit\n\n"
            "The exact initial-100 calibration head emitted 371 frame-local proposals "
            "on 295 of 2,259 eligible UI frames. This packet exposes the synchronized "
            "acquired, conditioned, signed-difference, radial Gamma-LS, strict-q1, and "
            "framewise NMS6 stages without opening a label file. Expert and comparison "
            "sections are therefore explicitly not applicable. The proposal count is "
            "not a count of unique biological detections and does not identify precision.\n",
        )
        _atomic_json(
            work / "validation.json",
            {"status": "pending_visual_inspection", "scientific_audit_complete": False},
        )
        _atomic_json(
            work / "status.json",
            {
                "status": "rendered_pending_visual_inspection",
                "scientific_audit_complete": False,
                "inventory_complete": True,
            },
        )
        recovered_failure_archived = _archive_recovered_failure(work)
        render_checkpoint = _render_checkpoint(
            work, audit_contract_sha256=audit_contract_sha256
        )
        if len(render_checkpoint["completed_videos"]) != 1 + DEFAULT_ANCHOR_LIMIT:
            raise FullRecordScientificAuditUnavailable(
                "render checkpoint does not contain all 25 videos"
            )
        render_checkpoint["status"] = "render_complete_pending_visual_inspection"
        render_checkpoint["completed_at_utc"] = _utc_now()
        render_checkpoint["expected_completed_video_count"] = 25
        _atomic_json(work / "render_checkpoint.json", render_checkpoint)
        _atomic_json(
            work / "heartbeat.json",
            {
                "status": "rendered_pending_visual_inspection",
                "stage": "render_complete",
                "updated_at_utc": _utc_now(),
                "recovered_failure_archived": recovered_failure_archived,
            },
        )
        _atomic_json(work / "artifact_index.json", _artifact_index(work))
        if destination.exists():
            raise FileExistsError(
                f"audit output appeared before atomic promotion: {destination}"
            )
        work.replace(destination)
        return {
            "output": str(destination),
            "status": "rendered_pending_visual_inspection",
            "inventory": inventory,
            "artifact_index_sha256": _sha256(destination / "artifact_index.json"),
        }
    except BaseException as error:
        _atomic_json(
            work / "failure.json",
            {
                "status": "render_failed_partial_retained",
                "failed_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def _validation_transaction_paths(root: Path) -> tuple[Path, Path]:
    return (
        root.parent / f".{root.name}.validation-work",
        root.parent / f".{root.name}.prevalidation-backup",
    )


def _recover_validation_transaction(root: Path) -> None:
    staging, backup = _validation_transaction_paths(root)
    if not root.exists():
        if staging.is_dir():
            try:
                _verify_complete_index(staging)
            except Exception:
                if backup.is_dir():
                    staging_failed = staging.with_name(staging.name + ".failed")
                    if not staging_failed.exists():
                        staging.replace(staging_failed)
                    backup.replace(root)
                else:
                    raise
            else:
                staging.replace(root)
        elif backup.is_dir():
            backup.replace(root)
        else:
            raise FileNotFoundError(root)
    # A valid canonical root always wins. Sibling transaction directories are
    # recoverable implementation state, never part of the scientific artifact.
    _verify_complete_index(root)
    if backup.is_dir():
        shutil.rmtree(backup)
    if staging.is_dir():
        shutil.rmtree(staging)


def _transactionally_finalize_audit(
    root: Path,
    *,
    summary: Mapping[str, Any],
    llm_context: Mapping[str, Any],
    validation: Mapping[str, Any],
    status: Mapping[str, Any],
) -> dict[str, Any]:
    staging, backup = _validation_transaction_paths(root)
    if staging.exists() or backup.exists():
        raise FullRecordScientificAuditUnavailable(
            "validation transaction roots were not clean after recovery"
        )
    shutil.copytree(root, staging, copy_function=os.link)
    try:
        _atomic_json(staging / "summary.json", summary)
        _atomic_json(staging / "llm_context.json", llm_context)
        _atomic_json(staging / "validation.json", validation)
        _atomic_json(staging / "status.json", status)
        _atomic_json(staging / "artifact_index.json", _artifact_index(staging))
        staged_index = _verify_complete_index(staging)
        root.replace(backup)
        try:
            staging.replace(root)
        except BaseException:
            backup.replace(root)
            raise
        finalized_index = _verify_complete_index(root)
        if (
            finalized_index["artifact_index_sha256"]
            != staged_index["artifact_index_sha256"]
            or finalized_index["verified_artifact_count"]
            != staged_index["verified_artifact_count"]
        ):
            raise FullRecordScientificAuditUnavailable(
                "validation transaction index changed during promotion"
            )
        shutil.rmtree(backup)
        return finalized_index
    except BaseException:
        # Leave a complete staging tree and/or backup for deterministic recovery.
        raise


def validate_model_only_scientific_audit(
    audit_root: str | Path,
    *,
    visual_passed: bool,
    inspection_note: str,
) -> dict[str, Any]:
    """Fully decode the model-only media and record manual visual QA."""

    root = Path(audit_root).expanduser().resolve()
    _recover_validation_transaction(root)
    indexed_before = _verify_complete_index(root)
    summary = _read_json(root / "summary.json")
    inventory = _read_json(root / "inventory.json")
    existing_validation = _read_json(root / "validation.json")
    if (
        summary.get("status") == "complete"
        and existing_validation.get("scientific_audit_complete") is True
        and existing_validation.get("all_checks_pass") is True
    ):
        return {
            "output": str(root),
            "status": "complete",
            "scientific_audit_complete": True,
            "artifact_index_sha256": indexed_before["artifact_index_sha256"],
            "validation_sha256": _sha256(root / "validation.json"),
            "validation": existing_validation,
            "idempotent_revalidation": True,
        }
    if summary.get("status") not in {
        "rendered_pending_visual_inspection",
        "failed_media_validation",
    }:
        raise FullRecordScientificAuditUnavailable(
            "audit is not in a finalizable or re-finalizable state"
        )
    if inventory.get("complete") is not True:
        raise FullRecordScientificAuditUnavailable("model-only inventory is incomplete")
    replay = _verify_stage_replay(
        summary["source_replay_root"],
        expected_audit_output=root,
        verify_dense_threshold=False,
    )
    if replay["indexed"]["artifact_index_sha256"] != str(
        summary["source_replay_artifact_index_sha256"]
    ):
        raise FullRecordScientificAuditUnavailable("source replay index changed")
    model_occurrences = []
    with (root / "2_Model_Annotations/model_occurrences.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        model_occurrences = list(csv.DictReader(stream))
    with (root / "2_Model_Annotations/candidate_surrogates.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        surrogate_rows = list(csv.DictReader(stream))
    normalized_model_occurrences = [
        {
            **_normalize_candidate(row),
            "model_roi_id": str(row["model_roi_id"]),
            "selected_for_closeup": _bool(row["selected_for_closeup"]),
        }
        for row in model_occurrences
    ]
    replay_assignment_by_id = {
        str(row["proposal_id"]): row for row in replay["assignments"]
    }
    expected_model_occurrences = [
        {
            **row,
            "model_roi_id": str(
                replay_assignment_by_id[str(row["proposal_id"])]["model_roi_id"]
            ),
            "selected_for_closeup": bool(
                replay_assignment_by_id[str(row["proposal_id"])][
                    "selected_for_closeup"
                ]
            ),
        }
        for row in replay["proposals"]
    ]
    normalized_surrogate_rows = [
        _normalize_surrogate(row) for row in surrogate_rows
    ]
    expert_media = _section_scientific_media(root, "1_Expert_Annotations")
    comparison_media = _section_scientific_media(root, "3_Comparison")
    manifest = _read_json(root / "video_manifest.json").get("videos", [])
    manifest_paths = [str(row.get("path", "")) for row in manifest]
    manifest_paths_unique = (
        len(manifest_paths) == len(set(manifest_paths))
        and all(manifest_paths)
    )
    actual_full_field_videos = sorted(
        (root / "2_Model_Annotations/videos").glob("*full_field*.mp4")
    )
    actual_closeup_videos = sorted(
        (root / "2_Model_Annotations/videos/closeups").glob("*.mp4")
    )
    actual_trace_figures = sorted(
        (root / "2_Model_Annotations/figures/traces").glob("*.png")
    )
    actual_model_metadata = sorted(
        (root / "2_Model_Annotations/metadata").glob("*.json")
    )
    expected_surrogate_ids = [
        str(row["model_roi_id"]) for row in replay["surrogates"]
    ]
    expected_full_field_paths = {
        "2_Model_Annotations/videos/model_q1_operational_sequential_full_field.mp4"
    }
    expected_closeup_paths = {
        f"2_Model_Annotations/videos/closeups/{_safe_id(model_roi_id)}.mp4"
        for model_roi_id in expected_surrogate_ids
    }
    expected_trace_paths = {
        f"2_Model_Annotations/figures/traces/{_safe_id(model_roi_id)}.png"
        for model_roi_id in expected_surrogate_ids
    }
    expected_metadata_paths = {
        f"2_Model_Annotations/metadata/{_safe_id(model_roi_id)}.json"
        for model_roi_id in expected_surrogate_ids
    }
    actual_full_field_paths = {
        path.relative_to(root).as_posix() for path in actual_full_field_videos
    }
    actual_closeup_paths = {
        path.relative_to(root).as_posix() for path in actual_closeup_videos
    }
    actual_trace_paths = {
        path.relative_to(root).as_posix() for path in actual_trace_figures
    }
    actual_metadata_paths = {
        path.relative_to(root).as_posix() for path in actual_model_metadata
    }
    metadata_content_exact = True
    for anchor in replay["surrogates"]:
        model_roi_id = str(anchor["model_roi_id"])
        payload = _read_json(
            root / f"2_Model_Annotations/metadata/{_safe_id(model_roi_id)}.json"
        )
        assigned_expected = [
            row["proposal_id"]
            for row in replay["proposals"]
            if str(replay_assignment_by_id[str(row["proposal_id"])]["model_roi_id"])
            == model_roi_id
        ]
        metadata_content_exact &= (
            _normalize_surrogate(payload) == anchor
            and payload.get("assigned_proposal_ids") == assigned_expected
            and payload.get("biological_identity_claimed") is False
            and payload.get("annotation_sources_opened") is False
        )
    video_results: list[dict[str, Any]] = []
    video_failures: list[str] = []
    marker_failures: list[str] = []
    for row in manifest:
        relative = str(row["path"])
        path = root / relative
        probe = _probe_operational_video(path)
        decode = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
            capture_output=True,
        )
        observed_frames = int(probe.get("nb_frames") or 0)
        contract_pass = (
            observed_frames == int(row["expected_frames"])
            and int(probe["width"]) == int(row["expected_width"])
            and int(probe["height"]) == int(row["expected_height"])
            and str(probe["codec_name"]) == str(row["expected_codec"])
            and math.isclose(
                float(probe["frame_rate"]),
                float(row["expected_frame_rate"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(probe["duration_seconds"]),
                float(row["expected_duration_seconds"]),
                rel_tol=0.0,
                abs_tol=0.025,
            )
        )
        if decode.returncode or not contract_pass:
            video_failures.append(relative)
        frame = _decode_marker_frame(
            path,
            int(row["sample_encoded_frame"]),
            int(probe["width"]),
            int(probe["height"]),
        )
        marker_counts = _marker_counts(frame)
        marker_pass = (
            marker_counts["orange_pixels"] > 0
            and marker_counts["green_pixels"] == 0
        )
        if not marker_pass:
            marker_failures.append(relative)
        video_results.append(
            {
                "path": relative,
                "codec": probe.get("codec_name"),
                "width": int(probe["width"]),
                "height": int(probe["height"]),
                "frames": observed_frames,
                "frame_rate": float(probe["frame_rate"]),
                "duration_seconds": float(probe["duration_seconds"]),
                "contract_pass": contract_pass,
                "full_decode_pass": decode.returncode == 0,
                "encoded_marker_counts": marker_counts,
                "encoded_marker_separation_pass": marker_pass,
            }
        )
    png_failures: list[str] = []
    png_count = 0
    for path in sorted(root.rglob("*.png")):
        png_count += 1
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            png_failures.append(path.relative_to(root).as_posix())
    packet_path = (
        root
        / "2_Model_Annotations/representative_stage/full_record_q1_stage_crop.npz"
    )
    with np.load(packet_path, allow_pickle=False) as packet:
        packet_keys = sorted(packet.files)
        expected_packet_keys = sorted(
            [
                "raw",
                "conditioned_previous",
                "conditioned_current",
                "difference",
                "gamma_ls",
                "threshold_exceedance",
                "proposals",
            ]
        )
        packet_shapes = {key: list(packet[key].shape) for key in packet.files}
        packet_same_shape = len({tuple(value) for value in packet_shapes.values()}) == 1
        packet_values = {key: np.array(packet[key], copy=True) for key in packet.files}
    stage_metadata = _read_json(
        root
        / "2_Model_Annotations/representative_stage/full_record_q1_stage_crop.json"
    )
    representative = _normalize_candidate(
        replay["science"]["representative_stage_proposal"]
    )
    source_ui = int(representative["source_frame_ui"])
    crop = tuple(int(value) for value in stage_metadata["crop_xyxy_half_open"])
    x0, y0, x1, y1 = crop
    source_movie = np.load(
        replay["manifest"]["source_movie"]["path"],
        mmap_mode="r",
        allow_pickle=False,
    )
    full_nms = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
    for row in replay["proposals"]:
        if int(row["source_frame_ui"]) == source_ui:
            full_nms[int(row["y_px"]), int(row["x_px"])] = 1
    stage_index = source_ui - 1
    threshold_index = source_ui - APPLICATION_UI[0]
    expected_packet = {
        "raw": np.asarray(source_movie[stage_index, y0:y1, x0:x1], dtype=np.uint16),
        "conditioned_previous": np.asarray(
            replay["arrays"]["conditioned_current_frame"][
                stage_index - 1, y0:y1, x0:x1
            ],
            dtype=np.float32,
        ),
        "conditioned_current": np.asarray(
            replay["arrays"]["conditioned_current_frame"][stage_index, y0:y1, x0:x1],
            dtype=np.float32,
        ),
        "difference": np.asarray(
            replay["arrays"]["difference_signed"][stage_index, y0:y1, x0:x1],
            dtype=np.float32,
        ),
        "gamma_ls": np.asarray(
            replay["arrays"]["gamma_initial100_scale"][stage_index, y0:y1, x0:x1],
            dtype=np.float32,
        ),
        "threshold_exceedance": np.asarray(
            replay["arrays"]["threshold_exceedance_q1_application"][
                threshold_index, y0:y1, x0:x1
            ],
            dtype=np.uint8,
        ),
        "proposals": np.asarray(full_nms[y0:y1, x0:x1], dtype=np.uint8),
    }
    packet_exact_replay = packet_keys == expected_packet_keys and all(
        np.array_equal(packet_values[key], expected_packet[key])
        for key in expected_packet_keys
    )
    metadata_exact_representative = (
        str(stage_metadata.get("proposal_id")) == str(representative["proposal_id"])
        and int(stage_metadata.get("source_frame_ui", -1)) == source_ui
        and int(stage_metadata.get("center_x_px", -1)) == int(representative["x_px"])
        and int(stage_metadata.get("center_y_px", -1)) == int(representative["y_px"])
        and float(stage_metadata.get("threshold_z", math.nan)) == PINNED_Q1_THRESHOLD
        and int(stage_metadata.get("nms_distance_px", -1)) == NMS_DISTANCE_PX
    )
    checks = {
        "prevalidation_index_complete": indexed_before["unindexed_file_count"] == 0,
        "authoritative_r2_ledger_pinned": replay["ledger"]["hashes"]
        == PINNED_LEDGER_HASHES,
        "exact_replay_reconciled": replay["reconciliation"]["status"] == "exact_match",
        "exact_371_model_occurrences": len(model_occurrences) == EXPECTED_PROPOSALS,
        "exact_24_label_free_surrogates": len(surrogate_rows)
        == DEFAULT_ANCHOR_LIMIT,
        "model_occurrences_exact_replay_content": _normalized_rows_equal(
            normalized_model_occurrences, expected_model_occurrences
        ),
        "candidate_surrogates_exact_replay_content": normalized_surrogate_rows
        == replay["surrogates"],
        "model_occurrences_all_unknown": all(
            row["biological_status"] == "unknown_unreviewed_proposal"
            for row in model_occurrences
        ),
        "no_temporal_linking": all(
            _bool(row["temporal_linking_applied"]) is False
            for row in model_occurrences
        ),
        "expert_section_explicitly_not_applicable": not expert_media,
        "comparison_section_explicitly_not_applicable": not comparison_media,
        "exact_25_model_videos": len(manifest) == 1 + DEFAULT_ANCHOR_LIMIT,
        "video_manifest_paths_unique": manifest_paths_unique,
        "exact_actual_full_field_video_count": len(actual_full_field_videos) == 1,
        "exact_actual_closeup_video_count": len(actual_closeup_videos)
        == DEFAULT_ANCHOR_LIMIT,
        "exact_actual_trace_figure_count": len(actual_trace_figures)
        == DEFAULT_ANCHOR_LIMIT,
        "exact_actual_model_metadata_count": len(actual_model_metadata)
        == DEFAULT_ANCHOR_LIMIT,
        "full_field_video_paths_exact": actual_full_field_paths
        == expected_full_field_paths,
        "closeup_video_paths_exact": actual_closeup_paths == expected_closeup_paths,
        "trace_figure_paths_exact": actual_trace_paths == expected_trace_paths,
        "model_metadata_paths_exact": actual_metadata_paths
        == expected_metadata_paths,
        "model_metadata_content_exact_replay": metadata_content_exact,
        "video_manifest_paths_exact": set(manifest_paths)
        == expected_full_field_paths | expected_closeup_paths,
        "all_videos_full_decode_and_frame_count": not video_failures,
        "all_encoded_markers_orange_only": not marker_failures,
        "all_pngs_decode": not png_failures,
        "representative_packet_keys_exact": packet_keys == expected_packet_keys,
        "representative_packet_same_crop": packet_same_shape,
        "representative_packet_values_exact_replay": packet_exact_replay,
        "representative_metadata_exact_frozen_selection": metadata_exact_representative,
        "representative_packet_full_frame_nms_before_crop": stage_metadata.get(
            "proposals_semantics"
        )
        == (
            "all exact deterministic Euclidean NMS6 peaks from the full source frame, "
            "cropped only after NMS"
        ),
        "inventory_complete": inventory.get("complete") is True,
        "manual_visual_inspection_passed": bool(visual_passed),
        "no_annotations_opened": summary.get("annotation_sources_opened") is False,
    }
    passed = all(checks.values())
    validation = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "checks": checks,
        "all_checks_pass": passed,
        "scientific_audit_complete": passed,
        "video_count": len(manifest),
        "video_failures": sorted(set(video_failures)),
        "encoded_marker_separation_failures": marker_failures,
        "video_results": video_results,
        "png_count": png_count,
        "png_decode_failures": png_failures,
        "representative_packet": {
            "keys": packet_keys,
            "shapes": packet_shapes,
            "same_crop": packet_same_shape,
        },
        "visual_inspection": {
            "performed": True,
            "passed": bool(visual_passed),
            "note": str(inspection_note),
            "checks": [
                "fixed grayscale stage scales and signed zero semantics",
                "source aspect preserved in full-field panels",
                "orange-only model marker separation",
                "sequential-stage and source-frame synchronization",
                "projection, trace, threshold, and NMS legibility",
            ],
        },
        "claim_boundary": summary["claim_boundary"],
    }
    finalized_status = "complete" if passed else "failed_media_validation"
    summary["status"] = finalized_status
    summary["scientific_audit_complete"] = passed
    summary["visual_inspection"] = validation["visual_inspection"]
    llm_context = _read_json(root / "llm_context.json")
    llm_context["expected_and_observed_counts"] = summary
    llm_context["audit_finalization"] = {
        "status": finalized_status,
        "scientific_audit_complete": passed,
        "validation": "validation.json",
    }
    finalized_index = _transactionally_finalize_audit(
        root,
        summary=summary,
        llm_context=llm_context,
        validation=validation,
        status={
            "status": finalized_status,
            "scientific_audit_complete": passed,
            "inventory_complete": inventory.get("complete") is True,
        },
    )
    return {
        "output": str(root),
        "status": finalized_status,
        "scientific_audit_complete": passed,
        "artifact_index_sha256": finalized_index["artifact_index_sha256"],
        "validation_sha256": _sha256(root / "validation.json"),
        "validation": validation,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--ledger-root", required=True)
    preflight.add_argument("--preflight-root", required=True)
    preflight.add_argument("--replay-output", required=True)
    preflight.add_argument("--audit-output", required=True)
    preflight.add_argument("--device", default="cuda:0")
    preflight.add_argument("--chunk-frames", type=int, default=DEFAULT_CHUNK_FRAMES)
    preflight.add_argument("--anchor-limit", type=int, default=DEFAULT_ANCHOR_LIMIT)
    preflight.add_argument(
        "--recover-known-all-true-gate",
        action="store_true",
        help="authorize only the hard-pinned completed r1 checkpoint gate recovery",
    )

    replay = subparsers.add_parser("replay")
    replay.add_argument("--config", required=True)
    replay.add_argument("--preflight-root", required=True)
    replay.add_argument("--ledger-root", required=True)
    replay.add_argument("--replay-output", required=True)
    replay.add_argument("--audit-output", required=True)
    replay.add_argument("--device", default="cuda:0")
    replay.add_argument("--chunk-frames", type=int, default=DEFAULT_CHUNK_FRAMES)
    replay.add_argument("--anchor-limit", type=int, default=DEFAULT_ANCHOR_LIMIT)
    replay.add_argument("--resume", action="store_true")

    render = subparsers.add_parser("render")
    render.add_argument("--replay-root", required=True)
    render.add_argument("--audit-output", required=True)
    render.add_argument("--resume", action="store_true")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--audit-root", required=True)
    validate.add_argument("--visual-passed", action="store_true")
    validate.add_argument("--inspection-note", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "preflight":
        result = create_preflight(
            GammaLSDifferenceConfig.load(arguments.config),
            ledger_root=arguments.ledger_root,
            preflight_root=arguments.preflight_root,
            replay_output=arguments.replay_output,
            audit_output=arguments.audit_output,
            device=arguments.device,
            chunk_frames=arguments.chunk_frames,
            anchor_limit=arguments.anchor_limit,
            recover_known_all_true_gate=arguments.recover_known_all_true_gate,
        )
    elif arguments.command == "replay":
        result = run_stage_replay(
            GammaLSDifferenceConfig.load(arguments.config),
            preflight_root=arguments.preflight_root,
            ledger_root=arguments.ledger_root,
            replay_output=arguments.replay_output,
            audit_output=arguments.audit_output,
            device=arguments.device,
            chunk_frames=arguments.chunk_frames,
            anchor_limit=arguments.anchor_limit,
            resume=arguments.resume,
        )
    elif arguments.command == "render":
        result = run_model_only_scientific_audit(
            arguments.replay_root,
            arguments.audit_output,
            resume=arguments.resume,
        )
    else:
        result = validate_model_only_scientific_audit(
            arguments.audit_root,
            visual_passed=arguments.visual_passed,
            inspection_note=arguments.inspection_note,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
