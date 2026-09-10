"""Post-selection Gamma-LS confirmation on the eligible ``15 right`` movie.

The historical independent-recording eligibility preflight inspected annotation
content to decide which recording was usable.  This module therefore does not
describe the workflow as end-to-end label blind.  Instead, it enforces the
narrower and auditable contract that candidate construction and model scoring
do not open the annotation manifest after that eligibility preflight.  Candidate
rows, block score maps, and their hashes are frozen before the annotation file
is verified and joined.

The runner intentionally does not reuse the older ICA candidate universe.  A
new universe is constructed from the explicitly frozen representation and
radial Gamma-LS context, one 50-frame block at a time.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from neurobench.algorithms.gamma_local_standardization import (
    GammaReferenceSpec,
    gamma_local_standardization,
)
from neurobench.metrics.sparse_detection import (
    extract_separated_local_maxima,
    temporal_pool,
)

from . import gpu_representations
from .cuda_runtime import CudaRuntimeUnavailable, require_cuda_device


RECORDING_ID = "15 right"
EXPECTED_MOVIE_SHAPE = (1608, 512, 512)
EXPECTED_MOVIE_DTYPE = "uint16"
FRAME_RATE_HZ = 50.0
BLOCK_FRAMES = 50
COMPLETE_BLOCKS = 32
EVALUATED_SOURCE_FRAMES = BLOCK_FRAMES * COMPLETE_BLOCKS
DROPPED_REMAINDER_FRAMES = EXPECTED_MOVIE_SHAPE[0] - EVALUATED_SOURCE_FRAMES
REFERENCE_SOURCE_FRAMES = 100
WARMUP_SOURCE_FRAMES = 1
FIRST_SCORED_SOURCE_FRAME_ZERO = WARMUP_SOURCE_FRAMES
LAST_SCORED_SOURCE_FRAME_ZERO = EVALUATED_SOURCE_FRAMES - 1
SCORED_OUTPUT_FRAMES = EVALUATED_SOURCE_FRAMES - WARMUP_SOURCE_FRAMES
TEMPORAL_POOL_MODE = "lme0.25"
SCALE_FLOOR_PERCENTILE = 10.0
CALIBRATION_SPATIAL_STRIDE = 4
EMPIRICAL_THRESHOLD_QUANTILE = 0.9999
NMS_DISTANCE_PX = 6
CANDIDATES_PER_BLOCK = 200
EVALUATION_BUDGETS_PER_BLOCK = (10, 20, 58, 100, 200)
PRIMARY_BUDGET_PER_BLOCK = 58
MATCH_RADIUS_PX = 6.0
ENERGY_EPSILON = 1e-8
MINIMUM_FREE_DISK_GIB = 20.0
MINIMUM_AVAILABLE_RAM_GIB = 8.0
MINIMUM_FREE_VRAM_GIB = 8.0

SUPPORTED_REPRESENTATION_ARMS = (
    "raw",
    "difference_signed",
    "difference_energy_normalized",
    "pca_whitened_derivative",
    "cs_parzen_two_frame",
)
LEARNED_REPRESENTATION_ARMS = (
    "pca_whitened_derivative",
    "cs_parzen_two_frame",
)
MATCH_TABLE_FIELDS = (
    "budget_per_block",
    "positive_id",
    "candidate_id",
    "spatial_distance_px",
    "candidate_block_id",
    "candidate_block_rank",
    "candidate_peak_frame_zero",
    "candidate_peak_frame_ui",
    "candidate_score",
)


INDEPENDENT_GAMMA_PROTOCOL: dict[str, Any] = {
    "schema_version": 1,
    "recording_id": RECORDING_ID,
    "recording_frame_rate_hz": FRAME_RATE_HZ,
    "source_movie_shape_tyx": list(EXPECTED_MOVIE_SHAPE),
    "source_movie_dtype": EXPECTED_MOVIE_DTYPE,
    "source_block_frames": BLOCK_FRAMES,
    "complete_source_blocks": COMPLETE_BLOCKS,
    "evaluated_source_frames": EVALUATED_SOURCE_FRAMES,
    "dropped_remainder_frames": DROPPED_REMAINDER_FRAMES,
    "causal_warmup_source_frames": WARMUP_SOURCE_FRAMES,
    "scored_output_frames": SCORED_OUTPUT_FRAMES,
    "scored_source_frame_interval_zero_inclusive": [
        FIRST_SCORED_SOURCE_FRAME_ZERO,
        LAST_SCORED_SOURCE_FRAME_ZERO,
    ],
    "reference_source_frames": REFERENCE_SOURCE_FRAMES,
    "reference_semantics": "predeclared_initial_frames_not_asserted_event_free",
    "common_preprocessing": "gaussian_sigma1_reflect_then_causal_ema_alpha0p4",
    "scale_floor_percentile": SCALE_FLOOR_PERCENTILE,
    "calibration_spatial_stride": CALIBRATION_SPATIAL_STRIDE,
    "empirical_threshold_quantile": EMPIRICAL_THRESHOLD_QUANTILE,
    "empirical_threshold_role": (
        "descriptive_framewise_label_content_isolated_calibration_not_candidate_gate_not_pfa"
    ),
    "calibration_evaluation_overlap": (
        "source frames 1 through 99 zero-based are both calibration and scored evaluation; "
        "this is a within-recording external sensitivity, not a held-out independent test"
    ),
    "temporal_pool": TEMPORAL_POOL_MODE,
    "candidate_nms_distance_px": NMS_DISTANCE_PX,
    "candidate_limit_per_block": CANDIDATES_PER_BLOCK,
    "candidate_selection": "top_budget_ranked_block_lme_peaks_not_empirical_threshold_gated",
    "candidate_id_namespace": "first_12_hex_characters_of_frozen_selection_sha256",
    "evaluation_budgets_per_block": list(EVALUATION_BUDGETS_PER_BLOCK),
    "primary_budget_per_block": PRIMARY_BUDGET_PER_BLOCK,
    "spatial_match_radius_px": MATCH_RADIUS_PX,
    "temporal_match": "candidate_peak_zero_based_frame_inside_source_zero_based_inclusive_interval",
    "matching": "maximum_cardinality_one_to_one_then_distance_and_rank_tiebreak",
    "candidate_interpretation": "unmatched_candidates_are_unknown_not_negative",
    "claim_scope": "one_recording_sparse_positive_confirmation_no_precision",
    "resource_guard": {
        "minimum_free_disk_gib": MINIMUM_FREE_DISK_GIB,
        "minimum_available_ram_gib": MINIMUM_AVAILABLE_RAM_GIB,
        "minimum_free_vram_gib": MINIMUM_FREE_VRAM_GIB,
    },
    "annotation_access_boundary": (
        "eligibility_preflight_inspected_annotation_content; this runner opens the "
        "annotation manifest only after Gamma candidate and score hashes are frozen"
    ),
}


class IndependentGammaValidationError(RuntimeError):
    """Raised when a frozen source, selection, or artifact contract changes."""


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def independent_gamma_protocol_digest() -> str:
    """Return the digest that a post-protected selection must freeze."""

    return _canonical_json_sha256(INDEPENDENT_GAMMA_PROTOCOL)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    fieldnames: Sequence[str] | None = None,
) -> None:
    if rows:
        fields = list(rows[0])
    elif fieldnames:
        fields = list(fieldnames)
    else:
        raise ValueError(f"empty table {path.name} requires explicit fieldnames")
    if any(list(row) != fields for row in rows):
        raise ValueError(f"inconsistent fields in {path.name}")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".partial.npy")
    np.save(temporary, np.asarray(values), allow_pickle=False)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def _resolve_from(manifest: Path, value: str) -> Path:
    supplied = Path(value).expanduser()
    return supplied.resolve() if supplied.is_absolute() else (manifest.parent / supplied).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise IndependentGammaValidationError(f"expected a JSON object: {path}")
    return payload


def _exact_keys(payload: Mapping[str, Any], expected: set[str], scope: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise IndependentGammaValidationError(
            f"{scope} fields changed: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )


@dataclass(frozen=True)
class FrozenIndependentSelection:
    """Verified post-protected deployment choice for external confirmation."""

    path: Path
    sha256: str
    arm: str
    gamma_reference: GammaReferenceSpec
    representation_model: Mapping[str, Any] | None
    representation_model_path: Path | None
    representation_model_sha256: str | None
    protected_summary_path: Path
    protected_summary_sha256: str
    protected_validation_path: Path
    protected_validation_sha256: str
    rationale: str
    raw: Mapping[str, Any]


def _verified_hashed_json(
    selection_path: Path, entry: Mapping[str, Any], *, role: str
) -> tuple[Path, str, dict[str, Any]]:
    _exact_keys(entry, {"path", "sha256"}, role)
    path = _resolve_from(selection_path, str(entry["path"]))
    if not path.is_file():
        raise IndependentGammaValidationError(f"missing {role}: {path}")
    observed = _sha256(path)
    if observed != str(entry["sha256"]):
        raise IndependentGammaValidationError(f"{role} hash changed")
    return path, observed, _read_json(path)


def _validate_transferable_two_frame_model(
    arm: str, payload: Mapping[str, Any]
) -> None:
    fit = payload.get("fit", payload)
    if not isinstance(fit, Mapping):
        raise IndependentGammaValidationError("transferable model fit must be an object")
    mean = np.asarray(fit.get("mean"), dtype=np.float64)
    whitening = np.asarray(fit.get("whitening"), dtype=np.float64)
    if mean.shape != (2,) or whitening.shape != (2, 2):
        raise IndependentGammaValidationError("transferable model must contain two-frame whitening")
    if not np.isfinite(mean).all() or not np.isfinite(whitening).all() or np.linalg.matrix_rank(
        whitening
    ) != 2:
        raise IndependentGammaValidationError("transferable whitening must be finite and full rank")
    if arm == "cs_parzen_two_frame":
        demixing = np.asarray(fit.get("demixing"), dtype=np.float64)
        if demixing.shape != (2, 2) or not np.isfinite(demixing).all() or np.linalg.matrix_rank(
            demixing @ whitening
        ) != 2:
            raise IndependentGammaValidationError("transferable ICA demixing must be finite and full rank")
        if int(fit.get("activity_component", -1)) not in (0, 1):
            raise IndependentGammaValidationError("transferable ICA activity component is invalid")
        if float(fit.get("activity_sign", 0)) not in (-1.0, 1.0):
            raise IndependentGammaValidationError("transferable ICA activity sign is invalid")


def load_frozen_independent_selection(
    path: str | Path,
) -> FrozenIndependentSelection:
    """Load a strict selection made only after protected Spon evaluation.

    The selection is deliberately a separate, small artifact.  The protected
    summary and validation bytes are hash-bound, and learned representations
    must additionally provide a hash-bound transferable model JSON.
    """

    selection_path = Path(path).expanduser().resolve()
    raw = _read_json(selection_path)
    _exact_keys(
        raw,
        {
            "schema_version",
            "status",
            "selection_scope",
            "decision_made_at_utc",
            "decision_rationale",
            "representation",
            "gamma_context",
            "protected_evidence",
            "candidate_protocol_sha256",
            "external_annotation_content_used_for_selection",
            "scientific_audit",
        },
        "frozen selection",
    )
    if raw["schema_version"] != 1:
        raise IndependentGammaValidationError("selection schema_version must be 1")
    if raw["status"] != "frozen_for_external_sparse_positive_confirmation":
        raise IndependentGammaValidationError("selection is not frozen for external confirmation")
    if raw["selection_scope"] != "post_protected_primary_decision":
        raise IndependentGammaValidationError("selection must follow the protected primary decision")
    if not str(raw["decision_made_at_utc"]).strip() or not str(
        raw["decision_rationale"]
    ).strip():
        raise IndependentGammaValidationError("selection timestamp and rationale are required")
    if raw["candidate_protocol_sha256"] != independent_gamma_protocol_digest():
        raise IndependentGammaValidationError("independent Gamma protocol digest changed")
    if raw["external_annotation_content_used_for_selection"] is not False:
        raise IndependentGammaValidationError("external annotations cannot select the pipeline")
    if raw["scientific_audit"] != {"enabled": True}:
        raise IndependentGammaValidationError("scientific audit must remain enabled")

    evidence = raw["protected_evidence"]
    _exact_keys(evidence, {"summary", "validation"}, "protected evidence")
    summary_path, summary_sha, summary = _verified_hashed_json(
        selection_path, evidence["summary"], role="protected summary"
    )
    validation_path, validation_sha, validation = _verified_hashed_json(
        selection_path, evidence["validation"], role="protected validation"
    )
    if not str(summary.get("status", "")).startswith("complete_protected_metrics"):
        raise IndependentGammaValidationError("protected metrics are not complete")
    if validation.get("status") != "passed_protected_metric_artifact_contract":
        raise IndependentGammaValidationError("protected validation did not pass")
    checks = validation.get("checks")
    if validation.get("all_checks_pass") is not True or not isinstance(checks, dict) or not all(
        value is True for value in checks.values()
    ):
        raise IndependentGammaValidationError("protected validation checks are incomplete")

    representation = raw["representation"]
    _exact_keys(representation, {"arm", "model"}, "representation selection")
    arm = str(representation["arm"])
    if arm not in SUPPORTED_REPRESENTATION_ARMS:
        raise IndependentGammaValidationError(f"unsupported selected representation: {arm}")
    model_path: Path | None = None
    model_sha: str | None = None
    model_payload: Mapping[str, Any] | None = None
    if arm in LEARNED_REPRESENTATION_ARMS:
        if not isinstance(representation["model"], Mapping):
            raise IndependentGammaValidationError(
                f"selected learned arm {arm} requires a frozen transferable model"
            )
        model_path, model_sha, model_payload = _verified_hashed_json(
            selection_path, representation["model"], role="representation model"
        )
        _validate_transferable_two_frame_model(arm, model_payload)
    elif representation["model"] is not None:
        raise IndependentGammaValidationError("fixed representations must not attach a model")

    gamma = raw["gamma_context"]
    _exact_keys(
        gamma,
        {
            "context_id",
            "support_width_px",
            "shape_n",
            "mode_radius_px",
            "guard_radius_px",
            "support_geometry",
            "boundary_mode",
            "epsilon",
        },
        "Gamma context",
    )
    if gamma["support_geometry"] != "disk":
        raise IndependentGammaValidationError("external confirmation requires radial disk Gamma-LS")
    if gamma["boundary_mode"] != "valid_renormalized_zero":
        raise IndependentGammaValidationError("external Gamma boundary mode changed")
    if float(gamma["epsilon"]) != 1e-6:
        raise IndependentGammaValidationError("external Gamma epsilon changed")
    reference = GammaReferenceSpec.from_mode(
        str(gamma["context_id"]),
        support_width_px=int(gamma["support_width_px"]),
        shape_n=float(gamma["shape_n"]),
        mode_radius_px=float(gamma["mode_radius_px"]),
        guard_radius_px=float(gamma["guard_radius_px"]),
        support_geometry="disk",
        boundary_mode="valid_renormalized_zero",
        epsilon=float(gamma["epsilon"]),
        scale_floor=0.0,
    )
    protected_primary = summary.get("protected_v1", {})
    protected_arm_rows = protected_primary.get("arm_summary", [])
    if not any(
        isinstance(row, Mapping) and row.get("representation") == arm
        for row in protected_arm_rows
    ):
        raise IndependentGammaValidationError(
            "selected representation is absent from protected primary evidence"
        )
    context_rows = [
        row
        for rows in summary.get("context_roles", {}).values()
        if isinstance(rows, list)
        for row in rows
        if isinstance(row, Mapping)
        and row.get("context_id") == reference.context_id
    ]
    if not context_rows:
        raise IndependentGammaValidationError(
            "selected Gamma context is absent from protected primary evidence"
        )
    expected_context = {
        "support_width_px": int(reference.support_width_px),
        "guard_radius_px": float(reference.guard_radius_px),
        "shape": float(reference.shape_n),
        "mode_radius_px": float(reference.nominal_mode_radius_px),
        "support": "radial_disk",
        "padding": "valid_renormalized_zero",
    }
    if not any(
        all(
            row.get(key) == value
            for key, value in expected_context.items()
        )
        for row in context_rows
    ):
        raise IndependentGammaValidationError(
            "selected Gamma parameters do not match the protected context"
        )
    return FrozenIndependentSelection(
        path=selection_path,
        sha256=_sha256(selection_path),
        arm=arm,
        gamma_reference=reference,
        representation_model=model_payload,
        representation_model_path=model_path,
        representation_model_sha256=model_sha,
        protected_summary_path=summary_path,
        protected_summary_sha256=summary_sha,
        protected_validation_path=validation_path,
        protected_validation_sha256=validation_sha,
        rationale=str(raw["decision_rationale"]),
        raw=raw,
    )


@dataclass(frozen=True)
class IndependentSourceAuthority:
    """Hash-verified source information, excluding annotation-file access."""

    preflight_path: Path
    preflight_sha256: str
    contract_path: Path
    contract_sha256: str
    movie_path: Path
    movie_sha256: str
    movie_size_bytes: int
    annotation_path: Path
    annotation_sha256: str
    safe_manifest_hashes: Mapping[str, str]
    preflight: Mapping[str, Any]
    contract: Mapping[str, Any]


def verify_independent_source_authority(
    *, preflight_path: str | Path, contract_path: str | Path
) -> IndependentSourceAuthority:
    """Verify eligibility and non-annotation source bytes before scoring.

    This function deliberately does not open or hash the annotation manifest.
    Its expected hash and path are merely carried forward for the post-seal
    label join.
    """

    preflight_file = Path(preflight_path).expanduser().resolve()
    contract_file = Path(contract_path).expanduser().resolve()
    preflight = _read_json(preflight_file)
    contract = _read_json(contract_file)
    if preflight.get("status") != "ready_for_frozen_finalist":
        raise IndependentGammaValidationError("independent eligibility preflight is not ready")
    if preflight.get("eligible_recordings") != [RECORDING_ID]:
        raise IndependentGammaValidationError("eligible independent recording set changed")
    if contract.get("status") != "frozen_awaiting_within_recording_finalists":
        raise IndependentGammaValidationError("historical independent contract is not frozen")
    if contract.get("eligible_recordings") != [RECORDING_ID]:
        raise IndependentGammaValidationError("historical independent contract eligibility changed")
    historical_preflight = Path(str(contract.get("independent_preflight_path", ""))).resolve()
    if historical_preflight != preflight_file:
        raise IndependentGammaValidationError("contract points to a different eligibility preflight")
    candidate_contract = contract.get("candidate_contract")
    if not isinstance(candidate_contract, dict) or _canonical_json_sha256(candidate_contract) != contract.get(
        "candidate_contract_sha256"
    ):
        raise IndependentGammaValidationError("historical candidate contract hash changed")
    expected_shared = {
        "block_frames": BLOCK_FRAMES,
        "evaluation_budgets_per_block": list(EVALUATION_BUDGETS_PER_BLOCK),
        "reference_frame_count": REFERENCE_SOURCE_FRAMES,
        "spatial_match_radius_px": int(MATCH_RADIUS_PX),
    }
    for key, value in expected_shared.items():
        if candidate_contract.get(key) != value:
            raise IndependentGammaValidationError(
                f"shared historical independent contract field changed: {key}"
            )

    recording = next(
        (row for row in preflight.get("recordings", []) if row.get("video_id") == RECORDING_ID),
        None,
    )
    if not isinstance(recording, dict) or recording.get("eligible") is not True:
        raise IndependentGammaValidationError("15 right is no longer eligibility-approved")
    if tuple(recording.get("shape_tyx", ())) != EXPECTED_MOVIE_SHAPE:
        raise IndependentGammaValidationError("15 right shape changed")
    if float(recording.get("frame_rate_hz")) != FRAME_RATE_HZ:
        raise IndependentGammaValidationError("15 right frame rate changed")
    if int(recording.get("roi_count")) != 10 or int(recording.get("spike_interval_count")) != 51:
        raise IndependentGammaValidationError("eligibility-preflight annotation counts changed")
    movie_entry = recording.get("cropped_video", {})
    movie_path = Path(str(movie_entry.get("path", ""))).resolve()
    if not movie_path.is_file():
        raise IndependentGammaValidationError(f"independent movie is missing: {movie_path}")
    if movie_path.stat().st_size != int(movie_entry.get("size_bytes")):
        raise IndependentGammaValidationError("independent movie size changed")
    movie_sha = _sha256(movie_path)
    if movie_sha != movie_entry.get("sha256"):
        raise IndependentGammaValidationError("independent movie hash changed")

    source_contracts = preflight.get("source_contracts")
    if not isinstance(source_contracts, dict):
        raise IndependentGammaValidationError("eligibility source contracts are missing")
    safe_hashes: dict[str, str] = {}
    for role in ("crop_manifest", "video_manifest"):
        entry = source_contracts.get(role, {})
        path = Path(str(entry.get("path", ""))).resolve()
        if not path.is_file() or _sha256(path) != entry.get("sha256"):
            raise IndependentGammaValidationError(f"{role} hash changed")
        safe_hashes[role] = str(entry["sha256"])
    annotation = source_contracts.get("annotation_manifest", {})
    annotation_path = Path(str(annotation.get("path", ""))).resolve()
    expected_annotation_sha = str(annotation.get("sha256", ""))
    if not annotation_path.is_file() or len(expected_annotation_sha) != 64:
        raise IndependentGammaValidationError("annotation source contract is incomplete")
    # Do not hash/read annotation_path here.  The byte verification is post-seal.
    return IndependentSourceAuthority(
        preflight_path=preflight_file,
        preflight_sha256=_sha256(preflight_file),
        contract_path=contract_file,
        contract_sha256=_sha256(contract_file),
        movie_path=movie_path,
        movie_sha256=movie_sha,
        movie_size_bytes=movie_path.stat().st_size,
        annotation_path=annotation_path,
        annotation_sha256=expected_annotation_sha,
        safe_manifest_hashes=safe_hashes,
        preflight=preflight,
        contract=contract,
    )


def candidate_universe_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash the canonical ordered candidate records."""

    if not rows:
        raise ValueError("candidate universe cannot be empty")
    return _canonical_json_sha256(list(rows))


def candidate_score_digest(scores: Any) -> str:
    """Hash an ordered finite candidate score vector as little-endian float64."""

    values = np.asarray(scores, dtype="<f8")
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("candidate scores must be a non-empty finite vector")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def block_score_digest(scores: Any) -> str:
    """Hash the complete ordered block-LME maps as little-endian float32."""

    values = np.asarray(scores, dtype="<f4")
    if values.ndim != 3 or values.shape[0] < 1 or not np.isfinite(values).all():
        raise ValueError("block scores must be a non-empty finite BYX array")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def rank_gamma_block_candidates(
    scores_tyx: Any,
    source_frame_indices: Any,
    *,
    block_id: int,
    block_start_frame_zero: int,
    block_stop_frame_exclusive: int,
    empirical_threshold_z: float,
    candidate_namespace: str,
    distance_px: int = NMS_DISTANCE_PX,
    limit: int = CANDIDATES_PER_BLOCK,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Pool one block and construct its deterministic Gamma-LS candidates."""

    scores = np.asarray(scores_tyx, dtype=np.float32)
    frames = np.asarray(source_frame_indices, dtype=np.int64)
    if scores.ndim != 3 or scores.shape[0] != frames.size or scores.shape[0] < 1:
        raise ValueError("scores and source_frame_indices must align as non-empty TYX")
    if not np.isfinite(scores).all() or not np.all(np.diff(frames) == 1):
        raise ValueError("block scores must be finite with contiguous frame indices")
    if frames[0] < block_start_frame_zero or frames[-1] >= block_stop_frame_exclusive:
        raise ValueError("representation frames escape their source block")
    namespace = str(candidate_namespace)
    if (
        len(namespace) != 12
        or any(character not in "0123456789abcdef" for character in namespace)
    ):
        raise ValueError("candidate_namespace must be the 12-character selection hash prefix")
    if int(block_id) < 1 or int(distance_px) < 1 or int(limit) < 1:
        raise ValueError("block_id, distance_px, and limit must be positive")
    threshold = float(empirical_threshold_z)
    if not math.isfinite(threshold):
        raise ValueError("empirical_threshold_z must be finite")
    pooled = temporal_pool(scores, TEMPORAL_POOL_MODE)
    peaks = extract_separated_local_maxima(
        pooled, int(distance_px), limit=int(limit)
    )
    rows = []
    for rank, (pooled_score, x, y) in enumerate(peaks, start=1):
        local_peak = int(np.argmax(scores[:, y, x]))
        peak_score = float(scores[local_peak, y, x])
        peak_frame = int(frames[local_peak])
        rows.append(
            {
                "candidate_id": f"indg_{namespace}_b{int(block_id):03d}_r{rank:03d}",
                "block_id": int(block_id),
                "block_start_frame_zero": int(block_start_frame_zero),
                "block_stop_frame_exclusive": int(block_stop_frame_exclusive),
                "block_start_frame_ui": int(block_start_frame_zero) + 1,
                "block_stop_frame_ui_inclusive": int(block_stop_frame_exclusive),
                "block_rank": rank,
                "pooled_lme_score": float(pooled_score),
                "peak_score_z": peak_score,
                "peak_frame_zero": peak_frame,
                "peak_frame_ui": peak_frame + 1,
                "x_px": int(x),
                "y_px": int(y),
                "peak_frame_z_exceeds_descriptive_initial100_quantile": bool(
                    peak_score > threshold
                ),
                "candidate_selected_by_empirical_threshold": False,
                "interpretation_before_label_join": "unknown_candidate",
            }
        )
    return rows, pooled


def _annotation_positives(
    annotations: Sequence[Mapping[str, Any]], *, evaluated_frames: int
) -> list[dict[str, Any]]:
    positives: list[dict[str, Any]] = []
    for annotation in annotations:
        if annotation.get("video_id") != RECORDING_ID:
            continue
        x = float(annotation["crop_x"])
        y = float(annotation["crop_y"])
        for index, interval in enumerate(annotation.get("spike_intervals", []), start=1):
            start = int(interval["start_frame"])
            stop = int(interval["end_frame"])
            if not 0 <= start <= stop < EXPECTED_MOVIE_SHAPE[0]:
                raise IndependentGammaValidationError("manual positive interval is out of bounds")
            positives.append(
                {
                    "positive_id": f"{annotation['annotation_id']}:interval_{index:02d}",
                    "annotation_id": str(annotation["annotation_id"]),
                    "roi_id": str(annotation["roi_id"]),
                    "x_px": x,
                    "y_px": y,
                    "start_frame_zero_inclusive": start,
                    "stop_frame_zero_inclusive": stop,
                    "start_frame_ui": start + 1,
                    "stop_frame_ui_inclusive": stop + 1,
                    "overlaps_scored_source_interval": bool(
                        stop >= FIRST_SCORED_SOURCE_FRAME_ZERO
                        and start < evaluated_frames
                    ),
                }
            )
    identifiers = [row["positive_id"] for row in positives]
    if len(identifiers) != len(set(identifiers)):
        raise IndependentGammaValidationError("manual positive identifiers are not unique")
    return positives


def _candidate_positive_distance(
    candidate: Mapping[str, Any], positive: Mapping[str, Any]
) -> float | None:
    peak = int(candidate["peak_frame_zero"])
    if not (
        int(positive["start_frame_zero_inclusive"])
        <= peak
        <= int(positive["stop_frame_zero_inclusive"])
    ):
        return None
    return math.hypot(
        float(candidate["x_px"]) - float(positive["x_px"]),
        float(candidate["y_px"]) - float(positive["y_px"]),
    )


def _one_to_one_matches(
    candidates: Sequence[Mapping[str, Any]],
    positives: Sequence[Mapping[str, Any]],
    *,
    radius_px: float,
) -> list[dict[str, Any]]:
    """Maximum-cardinality assignment with deterministic distance/rank costs."""

    from scipy.optimize import linear_sum_assignment

    if not positives or not candidates:
        return []
    count_positive = len(positives)
    count_candidate = len(candidates)
    unmatched_cost = 1_000_000.0
    invalid_cost = 1_000_000_000.0
    costs = np.full(
        (count_positive, count_candidate + count_positive),
        invalid_cost,
        dtype=np.float64,
    )
    distances = np.full((count_positive, count_candidate), np.nan, dtype=np.float64)
    for positive_index, positive in enumerate(positives):
        costs[positive_index, count_candidate + positive_index] = unmatched_cost
        for candidate_index, candidate in enumerate(candidates):
            distance = _candidate_positive_distance(candidate, positive)
            if distance is None or distance > float(radius_px):
                continue
            distances[positive_index, candidate_index] = distance
            costs[positive_index, candidate_index] = (
                distance * 1000.0
                + float(candidate["block_rank"])
                + float(candidate["block_id"]) / 1000.0
                + candidate_index / 10_000_000.0
            )
    row_indices, column_indices = linear_sum_assignment(costs)
    matches = []
    for positive_index, candidate_index in zip(row_indices, column_indices):
        if candidate_index >= count_candidate or costs[positive_index, candidate_index] >= unmatched_cost:
            continue
        candidate = candidates[int(candidate_index)]
        positive = positives[int(positive_index)]
        matches.append(
            {
                "positive_id": str(positive["positive_id"]),
                "candidate_id": str(candidate["candidate_id"]),
                "spatial_distance_px": float(distances[positive_index, candidate_index]),
                "candidate_block_id": int(candidate["block_id"]),
                "candidate_block_rank": int(candidate["block_rank"]),
                "candidate_peak_frame_zero": int(candidate["peak_frame_zero"]),
                "candidate_peak_frame_ui": int(candidate["peak_frame_ui"]),
                "candidate_score": float(candidate["pooled_lme_score"]),
            }
        )
    return sorted(matches, key=lambda row: (row["positive_id"], row["candidate_id"]))


def _nearest_candidate_rows_for_audit(
    candidates: Sequence[Mapping[str, Any]],
    positives: Sequence[Mapping[str, Any]],
    one_to_one_matches: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Record nearest time-eligible candidates separately from metric assignment."""

    matched_pairs = {
        (str(row["positive_id"]), str(row["candidate_id"]))
        for row in one_to_one_matches
    }
    rows = []
    for positive in positives:
        choices = []
        for candidate in candidates:
            distance = _candidate_positive_distance(candidate, positive)
            if distance is not None:
                choices.append(
                    (
                        distance,
                        int(candidate["block_rank"]),
                        str(candidate["candidate_id"]),
                        candidate,
                    )
                )
        nearest = min(choices, default=None)
        candidate = None if nearest is None else nearest[3]
        candidate_id = "" if candidate is None else str(candidate["candidate_id"])
        rows.append(
            {
                "positive_id": str(positive["positive_id"]),
                "annotation_id": str(positive["annotation_id"]),
                "nearest_time_eligible_candidate_id": candidate_id,
                "nearest_spatial_distance_px": "" if nearest is None else float(nearest[0]),
                "nearest_candidate_block_rank": "" if candidate is None else int(candidate["block_rank"]),
                "nearest_candidate_score": "" if candidate is None else float(candidate["pooled_lme_score"]),
                "nearest_candidate_peak_frame_zero": "" if candidate is None else int(candidate["peak_frame_zero"]),
                "nearest_pair_is_primary_one_to_one_match": (
                    False
                    if candidate is None
                    else (str(positive["positive_id"]), candidate_id) in matched_pairs
                ),
                "nearest_identity_distinct_from_one_to_one_assignment": True,
            }
        )
    return rows


def evaluate_independent_sparse_positives(
    candidates: Sequence[Mapping[str, Any]],
    candidate_scores: Any,
    annotations: Sequence[Mapping[str, Any]],
    *,
    frozen_candidate_universe_sha256: str,
    frozen_candidate_score_sha256: str,
    budgets: Sequence[int] = EVALUATION_BUDGETS_PER_BLOCK,
    match_radius_px: float = MATCH_RADIUS_PX,
    expected_known_positive_count: int = 51,
    expected_roi_count: int = 10,
) -> dict[str, Any]:
    """Join sparse positives only after exact candidate and score hashes match."""

    candidate_rows = [dict(row) for row in candidates]
    scores = np.asarray(candidate_scores, dtype=np.float64)
    if candidate_universe_digest(candidate_rows) != frozen_candidate_universe_sha256:
        raise IndependentGammaValidationError("candidate universe changed before label join")
    if candidate_score_digest(scores) != frozen_candidate_score_sha256:
        raise IndependentGammaValidationError("candidate scores changed before label join")
    if len(candidate_rows) != scores.size:
        raise ValueError("candidate rows and score vector differ in length")
    identifiers = [str(row["candidate_id"]) for row in candidate_rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate IDs must be unique")
    for row, score in zip(candidate_rows, scores):
        if not math.isclose(
            float(row["pooled_lme_score"]), float(score), rel_tol=0.0, abs_tol=0.0
        ):
            raise IndependentGammaValidationError("candidate row score and frozen vector differ")
    positives = _annotation_positives(
        annotations, evaluated_frames=EVALUATED_SOURCE_FRAMES
    )
    if len(positives) != int(expected_known_positive_count) or len(
        {row["annotation_id"] for row in positives}
    ) != int(expected_roi_count):
        raise IndependentGammaValidationError("15 right sparse-positive population changed")
    evaluated_positives = [
        row for row in positives if row["overlaps_scored_source_interval"]
    ]
    metric_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    for budget_value in budgets:
        budget = int(budget_value)
        if budget < 1:
            raise ValueError("budgets must be positive")
        selected = [row for row in candidate_rows if int(row["block_rank"]) <= budget]
        matches = _one_to_one_matches(
            selected, evaluated_positives, radius_px=float(match_radius_px)
        )
        match_by_positive = {row["positive_id"]: row for row in matches}
        reciprocal_ranks = []
        for positive in evaluated_positives:
            eligible_ranks = [
                int(candidate["block_rank"])
                for candidate in selected
                if (distance := _candidate_positive_distance(candidate, positive)) is not None
                and distance <= float(match_radius_px)
            ]
            reciprocal = 0.0 if not eligible_ranks else 1.0 / min(eligible_ranks)
            reciprocal_ranks.append(reciprocal)
            matched = match_by_positive.get(positive["positive_id"])
            outcome_rows.append(
                {
                    "budget_per_block": budget,
                    "positive_id": positive["positive_id"],
                    "annotation_id": positive["annotation_id"],
                    "roi_id": positive["roi_id"],
                    "matched_one_to_one": matched is not None,
                    "matched_candidate_id": "" if matched is None else matched["candidate_id"],
                    "best_eligible_reciprocal_block_rank": reciprocal,
                }
            )
        for row in matches:
            match_rows.append({"budget_per_block": budget, **row})
        matched_count = len(matches)
        known_count = len(evaluated_positives)
        metric_rows.append(
            {
                "budget_per_block": budget,
                "selected_candidate_count": len(selected),
                "known_positive_count": known_count,
                "one_to_one_known_positive_matches": matched_count,
                "known_positive_recall": matched_count / known_count if known_count else 0.0,
                "mean_reciprocal_block_rank": float(np.mean(reciprocal_ranks)),
                "unmatched_candidate_interpretation": "unknown_not_negative",
                "precision_specificity_false_positive_rate": "not_identified",
            }
        )
    primary_matches = [
        row for row in match_rows if row["budget_per_block"] == PRIMARY_BUDGET_PER_BLOCK
    ]
    return {
        "known_positive_count_all_source": len(positives),
        "known_positive_count_evaluated": len(evaluated_positives),
        "known_positive_count_without_scored_frame_overlap": len(positives)
        - len(evaluated_positives),
        "candidate_universe_sha256_before_label_join": frozen_candidate_universe_sha256,
        "candidate_score_sha256_before_label_join": frozen_candidate_score_sha256,
        "metric_rows": metric_rows,
        "match_rows": match_rows,
        "positive_outcome_rows": outcome_rows,
        "primary_matches": primary_matches,
        "unmatched_candidates": "unknown_not_negative",
        "precision_specificity_and_false_positive_rate": "not_identified",
        "claim_scope": "one_recording_sparse_positive_confirmation_no_population_generalization",
    }


class _CausalRepresentationProcessor:
    """Bounded GPU processor retaining EMA and adjacent-frame state."""

    def __init__(
        self,
        *,
        arm: str,
        model: Mapping[str, Any] | None,
        device: Any,
    ) -> None:
        self.arm = arm
        self.model = model
        self.device = device
        self.previous_common = None
        self.next_source_frame = 0

    def process(self, frames: np.ndarray, *, source_start_frame: int) -> tuple[Any, Any]:
        import torch

        source = np.asarray(frames)
        if source.ndim != 3 or source.dtype != np.uint16 or source.shape[0] < 1:
            raise ValueError("source chunk must be non-empty uint16 TYX")
        if int(source_start_frame) != self.next_source_frame:
            raise ValueError("source chunks must be contiguous and start at frame zero")
        raw = torch.as_tensor(np.array(source, copy=True), device=self.device).to(torch.float32)
        spatial = gpu_representations._spatial_gaussian_reflect(raw)
        common_frames = []
        previous = self.previous_common
        for frame in spatial:
            current = frame if previous is None else (
                gpu_representations.EMA_ALPHA * frame
                + (1.0 - gpu_representations.EMA_ALPHA) * previous
            )
            common_frames.append(current)
            previous = current
        common = torch.stack(common_frames)
        prior = self.previous_common
        self.previous_common = common[-1].detach()
        self.next_source_frame += int(source.shape[0])
        current_indices = torch.arange(
            source_start_frame,
            source_start_frame + source.shape[0],
            dtype=torch.int64,
            device=self.device,
        )

        if prior is None:
            paired_common = common
            paired_indices = current_indices
        else:
            paired_common = torch.cat((prior[None], common), dim=0)
            paired_indices = torch.cat((current_indices[:1] - 1, current_indices))

        if self.arm == "raw":
            if prior is None:
                return common[1:], current_indices[1:]
            return common, current_indices
        if self.arm == "difference_signed":
            result = gpu_representations.signed_difference_representation(
                paired_common, source_frame_indices=paired_indices
            )
        elif self.arm == "difference_energy_normalized":
            result = gpu_representations.energy_normalized_difference_representation(
                paired_common,
                source_frame_indices=paired_indices,
                epsilon=ENERGY_EPSILON,
            )
        elif self.arm == "pca_whitened_derivative":
            if self.model is None:
                raise IndependentGammaValidationError("PCA arm is missing its frozen model")
            result = gpu_representations.pca_whitened_derivative_representation(
                paired_common, self.model, source_frame_indices=paired_indices
            )
        elif self.arm == "cs_parzen_two_frame":
            if self.model is None:
                raise IndependentGammaValidationError("CS-Parzen arm is missing its frozen model")
            result = gpu_representations.frozen_two_frame_cs_parzen_representation(
                paired_common, self.model, source_frame_indices=paired_indices
            )
        else:  # pragma: no cover - selection validation prevents this
            raise ValueError(f"unsupported representation arm {self.arm}")
        return result.values, result.source_frame_indices


def _fit_label_content_isolated_calibration(
    movie: Any,
    *,
    selection: FrozenIndependentSelection,
    device: Any,
) -> tuple[GammaReferenceSpec, float, dict[str, Any]]:
    """Fit the external floor/threshold from fixed initial frames, without labels."""

    import torch

    processor = _CausalRepresentationProcessor(
        arm=selection.arm,
        model=selection.representation_model,
        device=device,
    )
    representation, indices = processor.process(
        np.asarray(movie[:REFERENCE_SOURCE_FRAMES]), source_start_frame=0
    )
    zero_floor = replace(selection.gamma_reference, scale_floor=0.0)
    with torch.inference_mode():
        moments = gamma_local_standardization(
            representation,
            zero_floor,
            chunk_frames=16,
            return_statistics=True,
        )
        if moments.local_mean is None or moments.local_std is None:
            raise AssertionError("calibration requires Gamma local moments")
        std_sample = moments.local_std[
            :, ::CALIBRATION_SPATIAL_STRIDE, ::CALIBRATION_SPATIAL_STRIDE
        ].reshape(-1)
        positive_std = std_sample[std_sample > 0]
        if positive_std.numel() < 1:
            raise IndependentGammaValidationError("calibration local scale has no positive values")
        floor = torch.quantile(positive_std, SCALE_FLOOR_PERCENTILE / 100.0)
        score = (representation - moments.local_mean) / (
            torch.maximum(moments.local_std, floor) + float(zero_floor.epsilon)
        )
        score_sample = score[
            :, ::CALIBRATION_SPATIAL_STRIDE, ::CALIBRATION_SPATIAL_STRIDE
        ].reshape(-1)
        threshold = torch.quantile(score_sample, EMPIRICAL_THRESHOLD_QUANTILE)
    fitted = replace(zero_floor, scale_floor=float(floor.item()))
    diagnostics = {
        "source_frame_interval_zero_half_open": [0, REFERENCE_SOURCE_FRAMES],
        "scored_frame_interval_zero_inclusive": [int(indices[0].item()), int(indices[-1].item())],
        "reference_semantics": "predeclared_initial_frames_not_asserted_event_free",
        "scale_floor_percentile": SCALE_FLOOR_PERCENTILE,
        "scale_floor": float(floor.item()),
        "spatial_sample_stride": CALIBRATION_SPATIAL_STRIDE,
        "scale_sample_count": int(positive_std.numel()),
        "threshold_quantile": EMPIRICAL_THRESHOLD_QUANTILE,
        "threshold_z": float(threshold.item()),
        "threshold_role": (
            "descriptive_framewise_calibration_not_candidate_gate_not_probability_of_false_alarm"
        ),
        "calibration_frames_overlap_scored_evaluation": True,
        "overlap_scored_source_frame_interval_zero_inclusive": [1, 99],
        "independent_heldout_test_claimed": False,
        "annotation_manifest_opened": False,
        "sparse_positive_coordinates_used": False,
        "sparse_positive_identities_used": False,
    }
    return fitted, float(threshold.item()), diagnostics


def _load_verified_annotations_after_seal(
    authority: IndependentSourceAuthority,
    *,
    seal_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Perform the first annotation-manifest read only after a valid score seal."""

    seal_file = Path(seal_path).resolve()
    seal = _read_json(seal_file)
    if seal.get("status") != "frozen_before_annotation_manifest_open":
        raise IndependentGammaValidationError("candidate/score seal is not frozen")
    for key in (
        "candidate_universe_sha256",
        "candidate_score_sha256",
        "block_score_sha256",
    ):
        if len(str(seal.get(key, ""))) != 64:
            raise IndependentGammaValidationError(f"candidate seal is missing {key}")
    sealed_files = {
        "candidate_universe.tsv": "candidate_universe_tsv_sha256",
        "candidate_scores.npy": "candidate_scores_npy_sha256",
        "block_lme_score_maps.npy": "block_score_maps_npy_sha256",
    }
    for name, hash_key in sealed_files.items():
        path = seal_file.parent / name
        if not path.is_file() or _sha256(path) != seal.get(hash_key):
            raise IndependentGammaValidationError(
                f"sealed candidate artifact changed before annotation join: {name}"
            )
    sealed_scores = np.load(
        seal_file.parent / "candidate_scores.npy", allow_pickle=False
    )
    if candidate_score_digest(sealed_scores) != seal["candidate_score_sha256"]:
        raise IndependentGammaValidationError(
            "sealed candidate score content changed before annotation join"
        )
    sealed_block_scores = np.load(
        seal_file.parent / "block_lme_score_maps.npy", mmap_mode="r", allow_pickle=False
    )
    if block_score_digest(sealed_block_scores) != seal["block_score_sha256"]:
        raise IndependentGammaValidationError(
            "sealed block score content changed before annotation join"
        )
    observed = _sha256(authority.annotation_path)
    if observed != authority.annotation_sha256:
        raise IndependentGammaValidationError("annotation manifest hash changed after preflight")
    payload = _read_json(authority.annotation_path)
    annotations = [
        dict(row) for row in payload.get("annotations", []) if row.get("video_id") == RECORDING_ID
    ]
    if len(annotations) != 10 or sum(
        len(row.get("spike_intervals", [])) for row in annotations
    ) != 51:
        raise IndependentGammaValidationError("15 right annotation content changed")
    return annotations, {
        "path": str(authority.annotation_path),
        "sha256": observed,
        "roi_count": len(annotations),
        "spike_interval_count": sum(len(row["spike_intervals"]) for row in annotations),
        "first_opened_after_candidate_score_seal": True,
    }


def _artifact_index(root: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_index.json" and not path.name.endswith(".partial"):
            artifacts.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return {"schema_version": 1, "artifacts": artifacts}


def _available_ram_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        raise IndependentGammaValidationError("cannot verify available RAM")
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise IndependentGammaValidationError("MemAvailable is absent from /proc/meminfo")


def _render_projection_checks(
    audit_root: Path,
    *,
    raw_projection: np.ndarray,
    score_projection: np.ndarray,
    positives: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    primary_matches: Sequence[Mapping[str, Any]],
    arm: str,
) -> list[str]:
    """Render bounded coordinate checks; these are not the complete audit media."""

    os.environ.setdefault("MPLCONFIGDIR", str(audit_root / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    raw = np.asarray(raw_projection, dtype=np.float32)
    evidence = np.asarray(score_projection, dtype=np.float32)
    raw_limits = tuple(np.percentile(raw, [1.0, 99.8]))
    evidence_limits = tuple(np.percentile(evidence, [1.0, 99.8]))
    primary = [row for row in candidates if int(row["block_rank"]) <= PRIMARY_BUDGET_PER_BLOCK]
    unique_experts = {}
    for row in positives:
        unique_experts.setdefault(row["annotation_id"], row)

    def background(ax: Any, values: np.ndarray, limits: tuple[float, float], title: str) -> None:
        ax.imshow(values, cmap="gray", vmin=limits[0], vmax=limits[1], interpolation="nearest")
        ax.set_title(title)
        ax.set_axis_off()

    outputs = []
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    background(axis, raw, raw_limits, "Expert coordinate projection check")
    for row in unique_experts.values():
        axis.add_patch(Circle((row["x_px"], row["y_px"]), 6, fill=False, edgecolor="#25c46b", linewidth=1.3))
    path = audit_root / "expert_projection_coordinate_check.png"
    figure.savefig(path, dpi=160); plt.close(figure); outputs.append(path.name)

    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    background(axis, evidence, evidence_limits, "Model coordinate projection check")
    for row in primary:
        axis.add_patch(Circle((row["x_px"], row["y_px"]), 2.5, fill=False, edgecolor="#f28e2b", linewidth=0.35, alpha=0.5))
    path = audit_root / "model_projection_coordinate_check.png"
    figure.savefig(path, dpi=160); plt.close(figure); outputs.append(path.name)

    candidate_by_id = {str(row["candidate_id"]): row for row in primary}
    positive_by_id = {str(row["positive_id"]): row for row in positives}
    figure, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    background(axes[0], raw, raw_limits, "Raw matched comparison")
    background(axes[1], evidence, evidence_limits, f"{arm} + radial Gamma-LS matched comparison")
    for axis in axes:
        for row in unique_experts.values():
            axis.add_patch(Circle((row["x_px"], row["y_px"]), 6, fill=False, edgecolor="#25c46b", linewidth=1.1))
    for row in primary:
        axes[1].add_patch(Circle((row["x_px"], row["y_px"]), 2.5, fill=False, edgecolor="#f28e2b", linewidth=0.3, alpha=0.45))
    for match in primary_matches:
        positive = positive_by_id[match["positive_id"]]
        candidate = candidate_by_id[match["candidate_id"]]
        axes[1].plot(
            [positive["x_px"], candidate["x_px"]],
            [positive["y_px"], candidate["y_px"]],
            color="#f6e7a1",
            linewidth=0.8,
        )
    path = audit_root / "comparison_projection_coordinate_check.png"
    figure.savefig(path, dpi=160); plt.close(figure); outputs.append(path.name)
    shutil.rmtree(audit_root / ".matplotlib", ignore_errors=True)
    return outputs


def run_independent_gamma_validation(
    *,
    selection_path: str | Path,
    preflight_path: str | Path,
    contract_path: str | Path,
    artifact_dir: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run the post-selection one-recording Gamma-LS sparse-positive check."""

    destination = Path(artifact_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"independent Gamma output already exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"independent Gamma output parent is missing: {destination.parent}")
    selection = load_frozen_independent_selection(selection_path)
    authority = verify_independent_source_authority(
        preflight_path=preflight_path, contract_path=contract_path
    )
    if str(device) != "cuda":
        raise IndependentGammaValidationError("production independent confirmation requires CUDA")
    try:
        runtime = require_cuda_device(device)
    except CudaRuntimeUnavailable as error:
        raise IndependentGammaValidationError(str(error)) from error
    disk_free = shutil.disk_usage(destination.parent).free
    available_ram = _available_ram_bytes()
    minimum_disk = int(MINIMUM_FREE_DISK_GIB * 2**30)
    minimum_ram = int(MINIMUM_AVAILABLE_RAM_GIB * 2**30)
    minimum_vram = int(MINIMUM_FREE_VRAM_GIB * 2**30)
    if disk_free < minimum_disk:
        raise IndependentGammaValidationError(
            f"free disk {disk_free} is below the frozen minimum {minimum_disk} bytes"
        )
    if available_ram < minimum_ram:
        raise IndependentGammaValidationError(
            f"available RAM {available_ram} is below the frozen minimum {minimum_ram} bytes"
        )
    if int(runtime["free_vram_bytes_before"]) < minimum_vram:
        raise IndependentGammaValidationError(
            "free VRAM is below the frozen 8-GiB pre-run minimum"
        )
    import tifffile
    import torch

    movie = tifffile.memmap(authority.movie_path, mode="r")
    if tuple(map(int, movie.shape)) != EXPECTED_MOVIE_SHAPE or str(movie.dtype) != EXPECTED_MOVIE_DTYPE:
        raise IndependentGammaValidationError("memory-mapped independent movie contract changed")
    resolved_device = torch.device(str(runtime["resolved_device"]))
    work = destination.parent / f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    work.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    try:
        run_contract = {
            "schema_version": 1,
            "status": "running_label_content_isolated_scoring",
            "started_at_utc": started,
            "selection": {"path": str(selection.path), "sha256": selection.sha256},
            "protected_evidence": {
                "summary_path": str(selection.protected_summary_path),
                "summary_sha256": selection.protected_summary_sha256,
                "validation_path": str(selection.protected_validation_path),
                "validation_sha256": selection.protected_validation_sha256,
            },
            "source": {
                "preflight_path": str(authority.preflight_path),
                "preflight_sha256": authority.preflight_sha256,
                "contract_path": str(authority.contract_path),
                "contract_sha256": authority.contract_sha256,
                "movie_path": str(authority.movie_path),
                "movie_sha256": authority.movie_sha256,
                "movie_size_bytes": authority.movie_size_bytes,
                "safe_manifest_hashes_verified_before_scoring": dict(authority.safe_manifest_hashes),
                "annotation_path_carried_not_opened": str(authority.annotation_path),
                "annotation_expected_sha256_carried_not_verified": authority.annotation_sha256,
            },
            "protocol": INDEPENDENT_GAMMA_PROTOCOL,
            "protocol_sha256": independent_gamma_protocol_digest(),
            "representation_arm": selection.arm,
            "representation_model": None
            if selection.representation_model_path is None
            else {
                "path": str(selection.representation_model_path),
                "sha256": selection.representation_model_sha256,
            },
            "gamma_context": dict(selection.raw["gamma_context"]),
            "runtime": runtime,
            "resource_precheck": {
                "free_disk_bytes": int(disk_free),
                "available_ram_bytes": int(available_ram),
                "free_vram_bytes": int(runtime["free_vram_bytes_before"]),
            },
            "historical_ica_candidate_universe_reused": False,
        }
        _atomic_json(work / "run_contract.json", run_contract)
        _atomic_json(
            work / "heartbeat.json",
            {"status": "calibrating", "updated_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        fitted_reference, empirical_threshold, calibration = _fit_label_content_isolated_calibration(
            movie, selection=selection, device=resolved_device
        )
        _atomic_json(work / "calibration.json", calibration)

        processor = _CausalRepresentationProcessor(
            arm=selection.arm,
            model=selection.representation_model,
            device=resolved_device,
        )
        candidate_rows: list[dict[str, Any]] = []
        pooled_maps: list[np.ndarray] = []
        scored_frame_counts: list[int] = []
        raw_projection = np.zeros(EXPECTED_MOVIE_SHAPE[1:], dtype=np.uint16)
        for block_index in range(COMPLETE_BLOCKS):
            start = block_index * BLOCK_FRAMES
            stop = start + BLOCK_FRAMES
            source_block = np.asarray(movie[start:stop])
            raw_projection = np.maximum(raw_projection, np.max(source_block, axis=0))
            with torch.inference_mode():
                representation, frame_indices = processor.process(
                    source_block, source_start_frame=start
                )
                result = gamma_local_standardization(
                    representation,
                    fitted_reference,
                    chunk_frames=16,
                    return_statistics=False,
                )
                scores = result.values.detach().cpu().numpy().astype(np.float32, copy=False)
                source_indices = frame_indices.detach().cpu().numpy().astype(np.int64, copy=False)
            rows, pooled = rank_gamma_block_candidates(
                scores,
                source_indices,
                block_id=block_index + 1,
                block_start_frame_zero=start,
                block_stop_frame_exclusive=stop,
                empirical_threshold_z=empirical_threshold,
                candidate_namespace=selection.sha256[:12],
            )
            if len(rows) != CANDIDATES_PER_BLOCK:
                raise IndependentGammaValidationError(
                    f"block {block_index + 1} yielded {len(rows)} candidates; "
                    f"budget {CANDIDATES_PER_BLOCK} is not evaluable"
                )
            candidate_rows.extend(rows)
            pooled_maps.append(pooled)
            scored_frame_counts.append(int(scores.shape[0]))
            _atomic_json(
                work / "heartbeat.json",
                {
                    "status": "constructing_gamma_candidates_without_annotation_manifest",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_blocks": block_index + 1,
                    "expected_blocks": COMPLETE_BLOCKS,
                },
            )
        block_maps = np.stack(pooled_maps).astype(np.float32, copy=False)
        candidate_scores = np.asarray(
            [row["pooled_lme_score"] for row in candidate_rows], dtype=np.float64
        )
        universe_sha = candidate_universe_digest(candidate_rows)
        score_sha = candidate_score_digest(candidate_scores)
        maps_sha = block_score_digest(block_maps)
        _atomic_tsv(work / "candidate_universe.tsv", candidate_rows)
        _atomic_npy(work / "candidate_scores.npy", candidate_scores)
        _atomic_npy(work / "block_lme_score_maps.npy", block_maps)
        _atomic_npy(work / "raw_max_projection.npy", raw_projection)
        candidate_seal = {
            "schema_version": 1,
            "status": "frozen_before_annotation_manifest_open",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_count": len(candidate_rows),
            "candidate_id_namespace": selection.sha256[:12],
            "candidate_universe_sha256": universe_sha,
            "candidate_universe_tsv_sha256": _sha256(work / "candidate_universe.tsv"),
            "candidate_score_sha256": score_sha,
            "candidate_scores_npy_sha256": _sha256(work / "candidate_scores.npy"),
            "block_score_sha256": maps_sha,
            "block_score_maps_npy_sha256": _sha256(work / "block_lme_score_maps.npy"),
            "selection_sha256": selection.sha256,
            "movie_sha256": authority.movie_sha256,
            "annotation_manifest_opened_by_candidate_scoring": False,
            "external_sparse_positive_coordinates_used": False,
            "external_sparse_positive_identities_used": False,
            "historical_ica_candidates_reused": False,
        }
        _atomic_json(work / "candidate_score_seal.json", candidate_seal)

        # This is the first annotation-file read/hash in the runner.
        annotations, annotation_provenance = _load_verified_annotations_after_seal(
            authority, seal_path=work / "candidate_score_seal.json"
        )
        metrics = evaluate_independent_sparse_positives(
            candidate_rows,
            candidate_scores,
            annotations,
            frozen_candidate_universe_sha256=universe_sha,
            frozen_candidate_score_sha256=score_sha,
        )
        _atomic_tsv(work / "sparse_positive_metrics.tsv", metrics["metric_rows"])
        _atomic_tsv(
            work / "one_to_one_matches.tsv",
            metrics["match_rows"],
            fieldnames=MATCH_TABLE_FIELDS,
        )
        _atomic_tsv(work / "positive_outcomes.tsv", metrics["positive_outcome_rows"])

        positives = _annotation_positives(
            annotations, evaluated_frames=EVALUATED_SOURCE_FRAMES
        )
        audit_root = work / "scientific_audit_inputs"
        audit_root.mkdir()
        _atomic_tsv(audit_root / "expert_occurrences.tsv", positives)
        primary_candidates = [
            row for row in candidate_rows if int(row["block_rank"]) <= PRIMARY_BUDGET_PER_BLOCK
        ]
        _atomic_tsv(audit_root / "primary_model_candidates.tsv", primary_candidates)
        _atomic_tsv(
            audit_root / "primary_one_to_one_matches.tsv",
            metrics["primary_matches"],
            fieldnames=MATCH_TABLE_FIELDS,
        )
        nearest_candidate_rows = _nearest_candidate_rows_for_audit(
            primary_candidates, positives, metrics["primary_matches"]
        )
        _atomic_tsv(
            audit_root / "nearest_time_eligible_candidate_pairs.tsv",
            nearest_candidate_rows,
        )
        score_projection = np.max(block_maps, axis=0)
        _atomic_npy(audit_root / "gamma_lme_max_projection.npy", score_projection)
        projection_files = _render_projection_checks(
            audit_root,
            raw_projection=raw_projection,
            score_projection=score_projection,
            positives=positives,
            candidates=candidate_rows,
            primary_matches=metrics["primary_matches"],
            arm=selection.arm,
        )
        audit_plan = {
            "schema_version": 1,
            "status": "pending_full_media_generation_and_validation",
            "standard": "docs/workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md",
            "projection_coordinate_checks": projection_files,
            "stage_sequence": [
                "Raw acquired frame",
                selection.arm,
                "signed radial Gamma-LS",
                "per-block lme0.25 evidence",
                "deterministic spatial NMS candidates",
            ],
            "expert_inputs": ["expert_occurrences.tsv", "../raw_max_projection.npy"],
            "model_inputs": ["primary_model_candidates.tsv", "gamma_lme_max_projection.npy"],
            "comparison_inputs": [
                "primary_one_to_one_matches.tsv",
                "nearest_time_eligible_candidate_pairs.tsv",
            ],
            "replay_contract": "../run_contract.json",
            "required_next_outputs": {
                "expert": "section-pure full-field video, per-ROI closeups and full-duration traces",
                "model": "section-pure sequential full-field video and consolidated-model closeups/traces",
                "comparison": "two-panel grayscale figures, per-occurrence nearest traces, and match tables",
            },
            "coordinate_convention": "x=column, y=row",
            "candidate_frame_fields": "zero-based computational and one-based UI fields are both stored",
            "source_annotation_frame_fields": "source-manifest zero-based inclusive; UI fields add one",
            "unmatched_candidates": "unknown_not_negative",
            "projection_checks_are_complete_scientific_audit": False,
        }
        _atomic_json(audit_root / "audit_plan.json", audit_plan)

        primary_metric = next(
            row for row in metrics["metric_rows"] if row["budget_per_block"] == PRIMARY_BUDGET_PER_BLOCK
        )
        summary = {
            "schema_version": 1,
            "status": "complete_metrics_scientific_audit_pending",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "recording_id": RECORDING_ID,
            "representation_arm": selection.arm,
            "gamma_context_id": selection.gamma_reference.context_id,
            "candidate_count": len(candidate_rows),
            "candidate_count_per_block": CANDIDATES_PER_BLOCK,
            "complete_blocks": COMPLETE_BLOCKS,
            "complete_block_source_frame_count": EVALUATED_SOURCE_FRAMES,
            "scored_output_frame_count": sum(scored_frame_counts),
            "scored_output_frames_by_block": scored_frame_counts,
            "scored_source_frame_interval_zero_inclusive": [
                FIRST_SCORED_SOURCE_FRAME_ZERO,
                LAST_SCORED_SOURCE_FRAME_ZERO,
            ],
            "dropped_remainder_frames": DROPPED_REMAINDER_FRAMES,
            "known_positive_count": metrics["known_positive_count_evaluated"],
            "metrics": metrics["metric_rows"],
            "primary_budget_per_block": PRIMARY_BUDGET_PER_BLOCK,
            "primary_known_positive_recall": primary_metric["known_positive_recall"],
            "primary_mean_reciprocal_block_rank": primary_metric["mean_reciprocal_block_rank"],
            "candidate_score_seal": candidate_seal,
            "annotation_provenance": annotation_provenance,
            "annotation_access_boundary": INDEPENDENT_GAMMA_PROTOCOL["annotation_access_boundary"],
            "end_to_end_label_blind_claimed": False,
            "candidate_construction_and_scoring_accessed_annotation_manifest": False,
            "candidate_selection_used_empirical_threshold": False,
            "calibration_frames_overlap_scored_evaluation": True,
            "independent_heldout_test_claimed": False,
            "historical_ica_candidate_universe_reused": False,
            "unmatched_candidates": "unknown_not_negative",
            "precision_specificity_and_false_positive_rate": "not_identified",
            "claim_scope": "one_recording_sparse_positive_confirmation_no_population_generalization",
            "scientific_audit_status": "pending_full_media_generation_and_validation",
        }
        _atomic_json(work / "summary.json", summary)
        _atomic_json(
            work / "llm_context.json",
            {
                "entrypoint": "summary.json",
                "grain": "one frozen representation/context x 32 complete 50-frame blocks",
                "primary_tables": [
                    "sparse_positive_metrics.tsv",
                    "one_to_one_matches.tsv",
                    "positive_outcomes.tsv",
                ],
                "candidate_seal": "candidate_score_seal.json",
                "audit_inputs": "scientific_audit_inputs/audit_plan.json",
                "stage_sequence": audit_plan["stage_sequence"],
                "limitations": [
                    "the eligibility preflight inspected annotation content",
                    "this is one-recording sparse-positive confirmation only",
                    "unmatched candidates are unknown and precision is not identified",
                    "projection checks are audit hooks, not the complete scientific audit",
                ],
            },
        )
        checks = {
            "selection_is_post_protected_and_hash_verified": True,
            "source_movie_and_safe_manifests_hash_verified_before_scoring": True,
            "exact_32_complete_blocks": len(block_maps) == COMPLETE_BLOCKS,
            "exact_1599_scored_output_frames": sum(scored_frame_counts) == SCORED_OUTPUT_FRAMES,
            "first_block_49_then_31_blocks_50": scored_frame_counts
            == [49] + [50] * (COMPLETE_BLOCKS - 1),
            "exact_8_frame_remainder_dropped": DROPPED_REMAINDER_FRAMES == 8,
            "exact_candidate_count": len(candidate_rows) == COMPLETE_BLOCKS * CANDIDATES_PER_BLOCK,
            "candidate_universe_hash_recomputed": candidate_universe_digest(candidate_rows) == universe_sha,
            "candidate_score_hash_recomputed": candidate_score_digest(candidate_scores) == score_sha,
            "block_score_hash_recomputed": block_score_digest(block_maps) == maps_sha,
            "annotation_hash_verified_only_after_seal": annotation_provenance[
                "first_opened_after_candidate_score_seal"
            ],
            "exact_10_rois_and_51_intervals": len(annotations) == 10
            and metrics["known_positive_count_all_source"] == 51,
            "one_to_one_matching": True,
            "historical_ica_candidates_not_reused": True,
            "precision_not_claimed": True,
            "full_scientific_audit_pending": True,
        }
        _atomic_json(
            work / "validation.json",
            {
                "status": "passed_metric_contract_scientific_audit_pending",
                "checks": checks,
                "all_checks_pass": all(checks.values()),
            },
        )
        _atomic_json(
            work / "status.json",
            {
                "status": "complete_metrics_scientific_audit_pending",
                "metric_contract_passed": all(checks.values()),
                "scientific_audit_complete": False,
                "paper_promotion_ready_from_this_artifact_alone": False,
            },
        )
        report = "# Independent Gamma-LS sparse-positive confirmation\n\n"
        report += (
            f"The frozen `{selection.arm}` representation and radial Gamma-LS context "
            f"`{selection.gamma_reference.context_id}` produced {len(candidate_rows):,} "
            f"automated candidates across {COMPLETE_BLOCKS} complete one-second blocks. "
            f"At B{PRIMARY_BUDGET_PER_BLOCK} per block, one-to-one known-positive recall "
            f"was {primary_metric['known_positive_recall']:.4f} and mean reciprocal block "
            f"rank was {primary_metric['mean_reciprocal_block_rank']:.4f}.\n\n"
        )
        report += (
            "This is one-recording sparse-positive confirmation. Unmatched candidates are "
            "unknown, so precision, specificity, and false-positive rate are not identified. "
            "The eligibility preflight inspected annotation content; only the subsequent "
            "Gamma candidate construction and scoring phase was isolated from the annotation "
            "manifest. Projection checks are present, but the full scientific-audit media and "
            "validation remain pending.\n"
        )
        _atomic_text(work / "REPORT.md", report)
        _atomic_json(work / "heartbeat.json", {"status": "complete", "updated_at_utc": datetime.now(timezone.utc).isoformat()})
        _atomic_json(work / "artifact_index.json", _artifact_index(work))
        if destination.exists():
            raise FileExistsError("independent Gamma destination appeared during commit")
        work.replace(destination)
        return summary
    except Exception as error:
        if work.exists():
            _atomic_json(
                work / "status.json",
                {
                    "status": "failed_preserved_partial",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": repr(error),
                    "annotation_may_have_been_opened": (work / "candidate_score_seal.json").is_file(),
                },
            )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run post-protected one-recording radial Gamma-LS confirmation."
    )
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--independent-preflight", required=True, type=Path)
    parser.add_argument("--independent-contract", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_independent_gamma_validation(
        selection_path=args.selection,
        preflight_path=args.independent_preflight,
        contract_path=args.independent_contract,
        artifact_dir=args.artifact_dir,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANDIDATES_PER_BLOCK",
    "COMPLETE_BLOCKS",
    "EVALUATION_BUDGETS_PER_BLOCK",
    "INDEPENDENT_GAMMA_PROTOCOL",
    "IndependentGammaValidationError",
    "FrozenIndependentSelection",
    "candidate_score_digest",
    "candidate_universe_digest",
    "evaluate_independent_sparse_positives",
    "independent_gamma_protocol_digest",
    "load_frozen_independent_selection",
    "rank_gamma_block_candidates",
    "run_independent_gamma_validation",
    "verify_independent_source_authority",
]
