"""Protected sparse-positive comparison for the two executed simple controls.

The support-sufficiency screen executed a signed square-annulus local
standardizer and the maintained positive-clipped box CFAR, but deliberately
kept both controls outside radial-support selection and never opened sparse
positive coordinates.  That is not evidence that either control is an
inferior detector.  This separate executor therefore applies the already
specified controls to the three fixed representations, freezes their complete
candidate stream, and only then joins the protected-v1 and descriptive-v7
positive tables.

This module intentionally has its own entry point.  It does not extend the
shared command dispatcher or mutate the completed protected executor.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import GammaLSDifferenceConfig
from .cuda_runtime import CudaRuntimeUnavailable, require_cuda_device
from .evaluation import (
    CANDIDATE_BUDGETS_PER_BURST,
    MATCH_RADIUS_PX,
    NMS_DISTANCE_PX,
    QUIET_NMS_PEAK_BURDENS,
    calibrate_training_quiet_thresholds,
    duration_matched_quiet_windows,
    extract_burst_candidates,
)
from .preflight import verify_matching_preflight
from .protected import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FIXED_ARMS,
    QUIET_SWAPS,
    _read_sparse_positives,
    _summary_by_arm,
    aggregate_match_rows,
    observation_match_rows,
)
from .screen import (
    GAMMA_EPSILON,
    _elapsed,
    _positive_scale_floor,
    _representation,
    _stream_common_history_to_device,
    build_fold_contracts,
)
from .support_grid import CONTROLS
from .support_sufficiency import (
    _maintained_positive_box_score,
    _square_annulus_moments,
)


CONTROL_IDS = (
    "signed_square_annulus_ls_h11_g3",
    "maintained_positive_box_cfar_h11_g3",
)
NMS_DISTANCES_PX = (4, 6, 8)
NONINFERIORITY_MARGIN = 0.02
PROTOCOL_VERSION = 1


class ProtectedControlUnavailable(RuntimeError):
    """Raised when a fail-closed control-comparison gate is not satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _parse_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token == "true":
        return True
    if token == "false":
        return False
    raise ValueError(f"{field} must be an explicit true/false value, got {value!r}")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError(f"required table is empty: {path.name}")
    return rows


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_text(path: Path, payload: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(payload)
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


def _validate_index_rows(root: Path, payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if payload.get("schema_version") != 1 or not isinstance(payload.get("artifacts"), list):
        raise ValueError(f"invalid artifact index: {root}")
    indexed: dict[str, dict[str, Any]] = {}
    for raw in payload["artifacts"]:
        if not isinstance(raw, Mapping):
            raise ValueError("artifact-index rows must be mappings")
        relative = str(raw.get("path", ""))
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"artifact index escapes its root: {relative!r}") from error
        if not relative or relative in indexed or candidate == root / "artifact_index.json":
            raise ValueError(f"invalid or duplicate indexed path: {relative!r}")
        size = int(raw.get("size_bytes", -1))
        sha = str(raw.get("sha256", ""))
        if size < 0 or len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
            raise ValueError(f"invalid artifact fingerprint for {relative!r}")
        if not candidate.is_file() or candidate.stat().st_size != size:
            raise ValueError(f"indexed artifact is absent or changed size: {relative}")
        indexed[relative] = {"path": candidate, "size_bytes": size, "sha256": sha}
    return indexed


def _verify_artifact_index(root: Path) -> dict[str, Any]:
    index_path = root / "artifact_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    indexed = _validate_index_rows(root, payload)
    for relative, row in indexed.items():
        if _sha256(row["path"]) != row["sha256"]:
            raise ValueError(f"indexed artifact changed content: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_index.json"
    }
    if actual != set(indexed):
        missing = sorted(set(indexed) - actual)
        extra = sorted(actual - set(indexed))
        raise ValueError(f"artifact index inventory mismatch: missing={missing}, extra={extra}")
    return {
        "root": str(root),
        "artifact_index_sha256": _sha256(index_path),
        "artifact_count": len(indexed),
        "indexed": indexed,
    }


def audit_support_control_need(support_dir: str | Path) -> dict[str, Any]:
    """Verify the label-free support artifact and quantify why controls need recall.

    Ineligibility for radial support selection is structural, not evidence of
    worse detection.  Both modern controls are therefore required in the
    protected comparison when they were executed and protected recall is false.
    """

    root = Path(support_dir).expanduser().resolve()
    verified = _verify_artifact_index(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    boundary = json.loads((root / "claim_boundary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete_support_screen_only":
        raise ValueError("support artifact is not the completed screen-only result")
    if validation.get("status") != "passed_support_screen_artifact_contract_scientific_audit_pending":
        raise ValueError("support artifact did not pass its metric contract")
    if validation.get("checks", {}).get("three_named_controls_executed") is not True:
        raise ValueError("support artifact does not attest all named controls")
    if boundary.get("protected_recall_computed") is not False:
        raise ValueError("support artifact no longer represents a recall-pending control screen")
    if boundary.get("positive_coordinates_used") is not False or boundary.get(
        "positive_identities_used"
    ) is not False:
        raise ValueError("support control screen accessed protected fields")

    named = {str(row["control_id"]): row for row in summary.get("named_controls_executed", [])}
    if any(control_id not in named for control_id in CONTROL_IDS):
        raise ValueError("one or more required modern controls were not executed")
    if any(named[control_id].get("eligible_primary") is not False for control_id in CONTROL_IDS):
        raise ValueError("modern control unexpectedly entered radial support selection")

    swap_rows = _read_tsv(root / "control_quiet_swap_rows.tsv")
    latency_rows = _read_tsv(root / "repeated_single_frame_latency.tsv")
    control_evidence = []
    for control_id in CONTROL_IDS:
        rows = [row for row in swap_rows if row.get("context_id") == control_id]
        expected = len(FIXED_ARMS) * 4 * len(QUIET_SWAPS)
        if len(rows) != expected:
            raise ValueError(
                f"control {control_id} has {len(rows)} quiet-swap rows, expected {expected}"
            )
        if any(
            _parse_bool(row["positive_coordinates_used"], field="positive_coordinates_used")
            or _parse_bool(row["positive_identities_used"], field="positive_identities_used")
            for row in rows
        ):
            raise ValueError(f"control {control_id} quiet-swap evidence used protected fields")
        contrasts = np.asarray([float(row["positive_tail_contrast"]) for row in rows])
        latency = [row for row in latency_rows if row.get("context_id") == control_id]
        if len(latency) != 1:
            raise ValueError(f"control {control_id} lacks one repeated-latency row")
        control_evidence.append(
            {
                "control_id": control_id,
                "quiet_swap_cell_count": len(rows),
                "positive_tail_contrast_min": float(contrasts.min()),
                "positive_tail_contrast_mean": float(contrasts.mean()),
                "positive_tail_contrast_max": float(contrasts.max()),
                "positive_contrast_cell_count": int(np.count_nonzero(contrasts > 0)),
                "repeated_single_frame_p50_ms": float(latency[0]["p50_ms"]),
                "eligible_primary_in_support_selection": False,
            }
        )
    radial_h11 = [
        row
        for row in latency_rows
        if row.get("family") == "radial_gamma_ls" and int(row["half_width_px"]) == 11
    ]
    if len(radial_h11) != 1:
        raise ValueError("support artifact lacks the unique radial h11 latency reference")
    return {
        "decision": "protected_sparse_positive_comparison_required",
        "reason": (
            "Control ineligibility only protected radial-support selection. Both modern "
            "controls produced coordinate-free event/quiet contrast and one is materially "
            "faster, so detection recall must be compared before claiming the radial "
            "Gamma family is necessary or preferable."
        ),
        "support_root": str(root),
        "support_artifact_index_sha256": verified["artifact_index_sha256"],
        "support_summary_sha256": _sha256(root / "summary.json"),
        "fold_contexts_sha256": _sha256(root / "fold_contexts.json"),
        "radial_h11_repeated_single_frame_p50_ms": float(radial_h11[0]["p50_ms"]),
        "controls": control_evidence,
        "legacy_exact_control_in_scope": False,
        "protected_recall_already_computed": False,
    }


def _protected_index_metadata(
    protected_dir: str | Path,
    *,
    radial_context_role: str,
    support_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Read only completion/index and label-free contract metadata before sealing.

    The indexed observation-match and summary files are deliberately not opened
    here.  Their bytes and contents are verified only after the new control
    candidates have been sealed.
    """

    root = Path(protected_dir).expanduser().resolve()
    index_path = root / "artifact_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    indexed = _validate_index_rows(root, payload)
    required = {
        "run_contract.json",
        "candidate_seal.json",
        "candidates_label_sealed.tsv",
        "threshold_calibration.tsv",
        "protected_v1_observation_matches.tsv",
        "summary.json",
        "validation.json",
        "claim_boundary.json",
    }
    if not required.issubset(indexed):
        raise ValueError(
            "protected comparator index is incomplete: "
            + ",".join(sorted(required - set(indexed)))
        )
    contract = json.loads((root / "run_contract.json").read_text(encoding="utf-8"))
    if contract.get("run_type") != "protected_outer_fold_two_frame_representation":
        raise ValueError("protected comparator run contract has the wrong run type")
    provenance = contract.get("context_selection", {})
    if provenance.get("positive_coordinates_used") is not False or provenance.get(
        "positive_identities_used"
    ) is not False:
        raise ValueError("protected radial context selection used protected fields")
    by_fold = provenance.get("context_roles_by_fold", {})
    if set(map(str, by_fold)) != {"1", "2", "3", "4"}:
        raise ValueError("protected comparator contract lacks four context-plan folds")
    for fold, lanes in by_fold.items():
        matching = [row for row in lanes if row.get("role") == radial_context_role]
        if len(matching) != 1:
            raise ValueError(
                f"radial context role {radial_context_role!r} is not unique in fold {fold}"
            )
        lane = matching[0]
        if lane.get("support") != "radial_disk" or lane.get("padding") != "valid_renormalized_zero":
            raise ValueError("requested protected comparator role is not modern radial Gamma-LS")
    expected_context_hash = str(support_evidence["fold_contexts_sha256"])
    if provenance.get("source_sha256") != expected_context_hash:
        raise ValueError("protected comparator did not use the supplied support fold contexts")
    seal = json.loads((root / "candidate_seal.json").read_text(encoding="utf-8"))
    if seal.get("positive_coordinates_used") is not False or seal.get(
        "positive_identities_used"
    ) is not False:
        raise ValueError("protected radial candidates were not label-sealed")
    return {
        "root": str(root),
        "artifact_index_sha256": _sha256(index_path),
        "artifact_count": len(indexed),
        "run_contract_sha256": _sha256(root / "run_contract.json"),
        "candidate_seal_sha256": _sha256(root / "candidate_seal.json"),
        "movie_sha256": str(contract.get("movie_sha256", "")),
        "radial_context_role": radial_context_role,
        "label_derived_artifacts_opened_before_control_seal": False,
    }


def _active_gamma_processes() -> list[dict[str, Any]]:
    try:
        import psutil
    except ModuleNotFoundError:  # pragma: no cover - environment dependent
        return []
    excluded = {os.getpid(), os.getppid()}
    rows = []
    for process in psutil.process_iter(("pid", "name", "cmdline")):
        try:
            pid = int(process.info["pid"])
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if pid in excluded:
            continue
        if "gamma_ls_difference" in command.lower() and (process.info.get("name") or "").lower().startswith("python"):
            rows.append({"pid": pid, "command": command[:400]})
    return rows


def _source_and_resource_gates(
    config: GammaLSDifferenceConfig,
    *,
    destination: Path,
    device: str,
) -> tuple[dict[str, Any], np.memmap]:
    try:
        runtime = require_cuda_device(device)
    except CudaRuntimeUnavailable as error:
        raise ProtectedControlUnavailable(str(error)) from error
    minimum_disk = int(float(config.payload["resources"]["minimum_free_disk_gib"]) * 2**30)
    free_disk = shutil.disk_usage(destination.parent).free
    if free_disk < minimum_disk:
        raise ProtectedControlUnavailable(
            f"free disk {free_disk} is below frozen minimum {minimum_disk} bytes"
        )
    minimum_vram = int(float(config.payload["resources"]["max_peak_vram_gib"]) * 2**30)
    if int(runtime["free_vram_bytes_before"]) < minimum_vram:
        raise ProtectedControlUnavailable("free VRAM is below the frozen 8-GiB run cap")
    active = _active_gamma_processes()
    if active:
        raise ProtectedControlUnavailable(
            "another gamma_ls_difference Python process is active: "
            + json.dumps(active, sort_keys=True)
        )
    movie = np.load(config.source_paths["movie"], mmap_mode="r", allow_pickle=False)
    if not isinstance(movie, np.memmap) or movie.ndim != 3 or str(movie.dtype) != "uint16":
        raise ValueError("control source must remain the frozen uint16 TYX memory map")
    if tuple(map(int, movie.shape)) != (2359, 340, 573):
        raise ValueError(f"control source shape changed: {movie.shape}")
    return {**runtime, "free_disk_bytes_before": int(free_disk)}, movie


def _mask_interval(frame_ui: np.ndarray, interval: Sequence[int]) -> np.ndarray:
    start, stop = map(int, interval)
    return (frame_ui >= start) & (frame_ui <= stop)


def _ui_window(interval: Sequence[int], *, review_start_ui: int) -> tuple[int, int]:
    start, stop = map(int, interval)
    return start - review_start_ui, stop - review_start_ui + 1


def _duration_map(bursts: Mapping[str, Sequence[int]]) -> dict[str, int]:
    return {
        str(key): int(bounds[1]) - int(bounds[0]) + 1
        for key, bounds in sorted(bursts.items())
    }


def _score_to_host(score: Any) -> tuple[np.ndarray, float]:
    device = score.device
    started = time.perf_counter()
    host = score.detach().cpu().numpy().astype(np.float32, copy=False)
    if device.type == "cuda":
        __import__("torch").cuda.synchronize(device)
    return host, (time.perf_counter() - started) * 1000.0


def _candidate_rows_for_score(
    score_host: np.ndarray,
    *,
    control_id: str,
    representation_name: str,
    quiet_swap: str,
    scale_floor: float | str,
    frame_ui: np.ndarray,
    folds: Sequence[Any],
    bursts: Mapping[str, Sequence[int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the frozen empirical quiet calibration and heldout candidate ranking."""

    if quiet_swap == QUIET_SWAPS[0]:
        floor_interval = folds[0].quiet_half_a_ui
        test_interval = folds[0].quiet_half_b_ui
    elif quiet_swap == QUIET_SWAPS[1]:
        floor_interval = folds[0].quiet_half_b_ui
        test_interval = folds[0].quiet_half_a_ui
    else:
        raise ValueError(f"unexpected quiet swap: {quiet_swap}")
    if any(
        fold.quiet_half_a_ui != folds[0].quiet_half_a_ui
        or fold.quiet_half_b_ui != folds[0].quiet_half_b_ui
        for fold in folds
    ):
        raise ValueError("control calibration requires identical quiet halves across folds")
    floor_mask = _mask_interval(frame_ui, floor_interval)
    test_mask = _mask_interval(frame_ui, test_interval)
    durations = _duration_map(bursts)
    quiet_test_windows = duration_matched_quiet_windows(test_mask, durations)
    candidates: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    for nms_distance in NMS_DISTANCES_PX:
        calibration = calibrate_training_quiet_thresholds(
            score_host,
            floor_mask,
            durations,
            target_peak_burdens=QUIET_NMS_PEAK_BURDENS,
            nms_distance_px=nms_distance,
        )
        for operating in calibration.operating_points:
            target = float(operating["target_nms_peaks_per_pseudo_burst"])
            threshold = float(operating["threshold_z"])
            quiet_test = extract_burst_candidates(
                score_host,
                quiet_test_windows,
                threshold_z=threshold,
                nms_distance_px=nms_distance,
            )
            realized = sum(len(peaks) for peaks in quiet_test.peaks.values()) / len(
                quiet_test.peaks
            )
            for fold in folds:
                heldout_window = {
                    fold.heldout_burst: _ui_window(
                        bursts[fold.heldout_burst], review_start_ui=int(frame_ui[0])
                    )
                }
                heldout = extract_burst_candidates(
                    score_host,
                    heldout_window,
                    threshold_z=threshold,
                    nms_distance_px=nms_distance,
                )
                peaks = heldout.peaks[fold.heldout_burst]
                calibrations.append(
                    {
                        "training_fold": int(fold.training_fold),
                        "heldout_burst": int(fold.heldout_burst),
                        "context_role": control_id,
                        "context_id": control_id,
                        "representation": representation_name,
                        "quiet_swap": quiet_swap,
                        "nms_distance_px": nms_distance,
                        "nms_role": "primary" if nms_distance == NMS_DISTANCE_PX else "descriptive_sensitivity",
                        "floor_interval_ui": json.dumps(list(floor_interval)),
                        "test_interval_ui": json.dumps(list(test_interval)),
                        "scale_floor_percentile": 10.0 if control_id.startswith("signed_square") else "not_applicable",
                        "scale_floor": scale_floor,
                        "target_nms_peaks_per_pseudo_burst": target,
                        "threshold_z": threshold,
                        "calibration_achieved_peaks_per_pseudo_burst": float(
                            operating["achieved_nms_peaks_per_pseudo_burst"]
                        ),
                        "heldout_quiet_peaks_per_pseudo_burst": float(realized),
                        "heldout_burst_candidate_count": len(peaks),
                        "probability_of_false_alarm_claimed": False,
                        "positive_coordinates_used": False,
                        "positive_identities_used": False,
                    }
                )
                for rank, (occupancy, x_px, y_px) in enumerate(peaks, start=1):
                    candidates.append(
                        {
                            "training_fold": int(fold.training_fold),
                            "burst_id": int(fold.heldout_burst),
                            "context_role": control_id,
                            "context_id": control_id,
                            "representation": representation_name,
                            "quiet_swap": quiet_swap,
                            "nms_distance_px": nms_distance,
                            "nms_role": "primary" if nms_distance == NMS_DISTANCE_PX else "descriptive_sensitivity",
                            "target_nms_peaks_per_pseudo_burst": target,
                            "threshold_z": threshold,
                            "candidate_rank": rank,
                            "occupancy_score": float(occupancy),
                            "x_px": int(x_px),
                            "y_px": int(y_px),
                            "interpretation_before_label_join": "unknown_candidate",
                        }
                    )
    return candidates, calibrations


def build_control_candidates(
    representation: Any,
    *,
    representation_name: str,
    control_id: str,
    frame_ui: np.ndarray,
    folds: Sequence[Any],
    bursts: Mapping[str, Sequence[int]],
    scale_floor_percentile: float,
    chunk_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Compute one control fresh and return its label-sealed candidate universe."""

    import torch

    if control_id not in CONTROL_IDS:
        raise ValueError(f"unsupported protected control: {control_id}")
    if representation_name not in FIXED_ARMS:
        raise ValueError("protected controls are restricted to the three fixed arms")
    score_packets: list[tuple[str, Any, float | str, float]] = []
    operator_ms = 0.0
    if control_id == "signed_square_annulus_ls_h11_g3":
        moments, moment_ms = _elapsed(
            representation.device,
            lambda: _square_annulus_moments(
                representation, outer=11, guard=3, chunk_frames=chunk_frames
            ),
        )
        local_mean, local_std = moments
        operator_ms += moment_ms
        frame_ui_device = torch.as_tensor(frame_ui, device=representation.device)
        for swap, interval in (
            (QUIET_SWAPS[0], folds[0].quiet_half_a_ui),
            (QUIET_SWAPS[1], folds[0].quiet_half_b_ui),
        ):
            mask = (frame_ui_device >= int(interval[0])) & (frame_ui_device <= int(interval[1]))
            floor = _positive_scale_floor(local_std, mask, scale_floor_percentile)
            score, score_ms = _elapsed(
                representation.device,
                lambda floor=floor: (representation - local_mean)
                / (torch.maximum(local_std, floor) + GAMMA_EPSILON),
            )
            score_packets.append((swap, score, float(floor.item()), score_ms))
        del local_mean, local_std, moments
    else:
        score, score_ms = _elapsed(
            representation.device,
            lambda: _maintained_positive_box_score(
                representation, outer=11, guard=3, chunk_frames=chunk_frames
            ),
        )
        operator_ms += score_ms
        score_packets = [
            (QUIET_SWAPS[0], score, "not_applicable_fixed_positive_control", 0.0),
            (QUIET_SWAPS[1], score, "not_applicable_fixed_positive_control", 0.0),
        ]

    candidate_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    score_hashes: dict[str, str] = {}
    d2h_ms = 0.0
    unique_scores: dict[int, np.ndarray] = {}
    try:
        for swap, score, floor, score_ms in score_packets:
            identity = id(score)
            if identity not in unique_scores:
                unique_scores[identity], transfer_ms = _score_to_host(score)
                d2h_ms += transfer_ms
            host = unique_scores[identity]
            score_hashes[swap] = _array_sha256(host)
            rows, calibration = _candidate_rows_for_score(
                host,
                control_id=control_id,
                representation_name=representation_name,
                quiet_swap=swap,
                scale_floor=floor,
                frame_ui=frame_ui,
                folds=folds,
                bursts=bursts,
            )
            candidate_rows.extend(rows)
            calibration_rows.extend(calibration)
            operator_ms += score_ms
    finally:
        unique_scores.clear()
        # The positive control reuses the same tensor for two quiet swaps.
        for score in {id(packet[1]): packet[1] for packet in score_packets}.values():
            del score

    expected_calibrations = 4 * len(QUIET_SWAPS) * len(NMS_DISTANCES_PX) * len(
        QUIET_NMS_PEAK_BURDENS
    )
    if len(calibration_rows) != expected_calibrations:
        raise AssertionError(
            f"control calibration count changed: {len(calibration_rows)} != {expected_calibrations}"
        )
    if any(row["interpretation_before_label_join"] != "unknown_candidate" for row in candidate_rows):
        raise AssertionError("control candidate acquired a pre-label interpretation")
    return candidate_rows, calibration_rows, {
        "control_id": control_id,
        "representation": representation_name,
        "frame_count": len(representation),
        "operator_and_score_runtime_ms": float(operator_ms),
        "operator_and_score_runtime_ms_per_frame": float(operator_ms / len(representation)),
        "device_to_host_ms": float(d2h_ms),
        "score_hashes_by_quiet_swap": score_hashes,
        "candidate_row_count": len(candidate_rows),
        "calibration_row_count": len(calibration_rows),
        "positive_coordinates_used": False,
        "positive_identities_used": False,
    }


def _candidate_hashes(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    universe_fields = (
        "training_fold",
        "burst_id",
        "context_role",
        "representation",
        "quiet_swap",
        "nms_distance_px",
        "target_nms_peaks_per_pseudo_burst",
        "threshold_z",
        "candidate_rank",
        "x_px",
        "y_px",
    )
    score_fields = (*universe_fields, "occupancy_score")
    return {
        "candidate_universe_sha256": _canonical_sha256(
            [{field: row[field] for field in universe_fields} for row in rows]
        ),
        "candidate_score_stream_sha256": _canonical_sha256(
            [{field: row[field] for field in score_fields} for row in rows]
        ),
    }


def _verify_seal_files(root: Path, seal: Mapping[str, Any]) -> None:
    expected = {
        "control_candidates_label_sealed.tsv": seal["candidate_table_sha256"],
        "control_threshold_calibration.tsv": seal["threshold_table_sha256"],
        "control_timings.tsv": seal["timing_table_sha256"],
        "control_score_hashes.json": seal["score_hash_manifest_sha256"],
    }
    for relative, digest in expected.items():
        if _sha256(root / relative) != digest:
            raise ValueError(f"sealed control artifact changed before label join: {relative}")


def _typed_radial_match_rows(
    path: Path,
    *,
    radial_context_role: str,
) -> list[dict[str, Any]]:
    source = _read_tsv(path)
    rows: list[dict[str, Any]] = []
    for row in source:
        if row.get("cohort") != "protected_v1" or row.get("context_role") != radial_context_role:
            continue
        if row.get("representation") not in FIXED_ARMS:
            continue
        typed: dict[str, Any] = dict(row)
        typed.update(
            {
                "nms_distance_px": int(row["nms_distance_px"]),
                "target_nms_peaks_per_pseudo_burst": float(
                    row["target_nms_peaks_per_pseudo_burst"]
                ),
                "burst_id": int(row["burst_id"]),
                "candidate_budget": int(row["candidate_budget"]),
                "matched": _parse_bool(row["matched"], field="matched"),
            }
        )
        rows.append(typed)
    expected = (
        len(FIXED_ARMS)
        * len(QUIET_SWAPS)
        * len(NMS_DISTANCES_PX)
        * len(QUIET_NMS_PEAK_BURDENS)
        * len(CANDIDATE_BUDGETS_PER_BURST)
        * 79
    )
    if len(rows) != expected:
        raise ValueError(
            f"radial protected-v1 slice has {len(rows)} rows, expected {expected}"
        )
    dimensions: dict[tuple[Any, ...], int] = {}
    for row in rows:
        key = (
            row["representation"],
            row["quiet_swap"],
            row["nms_distance_px"],
            row["target_nms_peaks_per_pseudo_burst"],
            row["candidate_budget"],
        )
        dimensions[key] = dimensions.get(key, 0) + 1
    if len(dimensions) != len(FIXED_ARMS) * 2 * 3 * 5 * 5 or set(dimensions.values()) != {79}:
        raise ValueError("radial protected-v1 operating grid is incomplete or duplicated")
    return rows


def _typed_radial_calibration_rows(
    path: Path,
    *,
    radial_context_role: str,
) -> list[dict[str, Any]]:
    rows = []
    for row in _read_tsv(path):
        if row.get("context_role") != radial_context_role or row.get("representation") not in FIXED_ARMS:
            continue
        typed: dict[str, Any] = dict(row)
        typed.update(
            {
                "training_fold": int(row["training_fold"]),
                "heldout_burst": int(row["heldout_burst"]),
                "nms_distance_px": int(row["nms_distance_px"]),
                "target_nms_peaks_per_pseudo_burst": float(
                    row["target_nms_peaks_per_pseudo_burst"]
                ),
                "heldout_quiet_peaks_per_pseudo_burst": float(
                    row["heldout_quiet_peaks_per_pseudo_burst"]
                ),
            }
        )
        rows.append(typed)
    expected = len(FIXED_ARMS) * 4 * 2 * 3 * 5
    if len(rows) != expected:
        raise ValueError(f"radial calibration slice has {len(rows)} rows, expected {expected}")
    return rows


def quiet_burden_comparison_rows(
    radial_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    key_fields = (
        "training_fold",
        "heldout_burst",
        "representation",
        "quiet_swap",
        "nms_distance_px",
        "target_nms_peaks_per_pseudo_burst",
    )

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(row[field] for field in key_fields)

    radial = {key(row): row for row in radial_rows}
    if len(radial) != len(radial_rows):
        raise ValueError("radial quiet calibration contains duplicate dimensions")
    output = []
    for row in control_rows:
        match = radial.get(key(row))
        if match is None:
            raise ValueError(f"control quiet calibration lacks radial pair: {key(row)}")
        control_burden = float(row["heldout_quiet_peaks_per_pseudo_burst"])
        radial_burden = float(match["heldout_quiet_peaks_per_pseudo_burst"])
        output.append(
            {
                "control_id": row["context_role"],
                **{field: row[field] for field in key_fields},
                "radial_context_role": match["context_role"],
                "radial_heldout_quiet_peaks_per_pseudo_burst": radial_burden,
                "control_heldout_quiet_peaks_per_pseudo_burst": control_burden,
                "control_minus_radial_quiet_burden": control_burden - radial_burden,
                "empirical_quiet_burden_only_not_pfa": True,
            }
        )
    return output


def _match_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["representation"]),
        str(row["quiet_swap"]),
        int(row["nms_distance_px"]),
        float(row["target_nms_peaks_per_pseudo_burst"]),
        int(row["burst_id"]),
        int(row["candidate_budget"]),
        str(row["observation_id"]),
        str(row["canonical_roi_id"]),
    )


def _metric_arrays(
    rows: Sequence[Mapping[str, Any]],
    *,
    identities: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    identity_index = {identity: index for index, identity in enumerate(identities)}
    budget_index = {budget: index for index, budget in enumerate(CANDIDATE_BUDGETS_PER_BURST)}
    matched = np.zeros((len(identities), 4, len(budget_index)), dtype=np.float64)
    counts = np.zeros_like(matched)
    for row in rows:
        identity = str(row["canonical_roi_id"])
        burst = int(row["burst_id"]) - 1
        budget = budget_index[int(row["candidate_budget"])]
        if identity not in identity_index or not 0 <= burst < 4:
            raise ValueError("match row has an undeclared identity or burst")
        counts[identity_index[identity], burst, budget] += 1.0
        matched[identity_index[identity], burst, budget] += float(
            _parse_bool(row["matched"], field="matched")
        )
    return matched, counts


def _metrics_from_weights(
    matched: np.ndarray,
    counts: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numerators = np.einsum("ri,ibk->rbk", weights, matched, optimize=True)
    denominators = np.einsum("ri,ibk->rbk", weights, counts, optimize=True)
    recall = np.full_like(numerators, np.nan)
    np.divide(numerators, denominators, out=recall, where=denominators > 0)
    with np.errstate(invalid="ignore"):
        macro = np.nanmean(recall, axis=1)
    x = np.asarray(CANDIDATE_BUDGETS_PER_BURST, dtype=np.float64)
    auc = np.trapezoid(macro, x=x, axis=1) / (x[-1] - x[0])
    b58 = macro[:, CANDIDATE_BUDGETS_PER_BURST.index(58)]
    return auc, b58, recall[:, :, CANDIDATE_BUDGETS_PER_BURST.index(58)]


def paired_radial_control_bootstrap(
    radial_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
    expected_cluster_count: int = 26,
) -> list[dict[str, Any]]:
    """Identity-clustered paired v1 contrasts, radial minus simple control."""

    radial_lookup = {_match_key(row): row for row in radial_rows}
    control_lookup = {_match_key(row): row for row in control_rows}
    if len(radial_lookup) != len(radial_rows) or len(control_lookup) != len(control_rows):
        raise ValueError("paired match inputs contain duplicate observation dimensions")
    if set(radial_lookup) != set(control_lookup):
        raise ValueError("radial and control match rows do not share an exact paired universe")
    identities = sorted({str(row["canonical_roi_id"]) for row in radial_rows})
    if len(identities) != expected_cluster_count:
        raise ValueError(
            f"protected bootstrap requires {expected_cluster_count} identities, got {len(identities)}"
        )
    rng = np.random.default_rng(int(seed))
    samples = rng.integers(0, len(identities), size=(int(replicates), len(identities)))
    bootstrap_weights = np.zeros((int(replicates), len(identities)), dtype=np.float64)
    for replicate, sample in enumerate(samples):
        bootstrap_weights[replicate] = np.bincount(sample, minlength=len(identities))
    point_weights = np.ones((1, len(identities)), dtype=np.float64)

    dimensions = sorted(
        {
            (
                str(row["representation"]),
                str(row["quiet_swap"]),
                int(row["nms_distance_px"]),
                float(row["target_nms_peaks_per_pseudo_burst"]),
            )
            for row in radial_rows
        }
    )
    pooled = sorted(
        {
            (
                str(row["representation"]),
                "crossfit_average",
                int(row["nms_distance_px"]),
                float(row["target_nms_peaks_per_pseudo_burst"]),
            )
            for row in radial_rows
        }
    )
    control_ids = {str(row["context_role"]) for row in control_rows}
    radial_roles = {str(row["context_role"]) for row in radial_rows}
    if len(control_ids) != 1 or len(radial_roles) != 1:
        raise ValueError("each paired bootstrap call requires one control and one radial role")
    output = []
    for representation, swap, nms, target in dimensions + pooled:
        def selected(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
            return [
                row
                for row in rows
                if str(row["representation"]) == representation
                and (swap == "crossfit_average" or str(row["quiet_swap"]) == swap)
                and int(row["nms_distance_px"]) == nms
                and float(row["target_nms_peaks_per_pseudo_burst"]) == target
            ]

        radial_selected = selected(radial_rows)
        control_selected = selected(control_rows)
        radial_matched, radial_counts = _metric_arrays(radial_selected, identities=identities)
        control_matched, control_counts = _metric_arrays(control_selected, identities=identities)
        if not np.array_equal(radial_counts, control_counts):
            raise ValueError("paired pipelines do not have the same identity/burst/budget counts")
        r_auc, r_b58, r_bursts = _metrics_from_weights(
            radial_matched, radial_counts, point_weights
        )
        c_auc, c_b58, c_bursts = _metrics_from_weights(
            control_matched, control_counts, point_weights
        )
        rb_auc, rb_b58, _ = _metrics_from_weights(
            radial_matched, radial_counts, bootstrap_weights
        )
        cb_auc, cb_b58, _ = _metrics_from_weights(
            control_matched, control_counts, bootstrap_weights
        )
        auc_delta = float(r_auc[0] - c_auc[0])
        b58_delta = float(r_b58[0] - c_b58[0])
        boot_auc = rb_auc - cb_auc
        boot_b58 = rb_b58 - cb_b58
        auc_low, auc_high = np.nanpercentile(boot_auc, [2.5, 97.5])
        b58_low, b58_high = np.nanpercentile(boot_b58, [2.5, 97.5])
        output.append(
            {
                "radial_context_role": next(iter(radial_roles)),
                "control_id": next(iter(control_ids)),
                "representation": representation,
                "quiet_swap": swap,
                "nms_distance_px": nms,
                "nms_role": "primary" if nms == NMS_DISTANCE_PX else "descriptive_sensitivity",
                "target_nms_peaks_per_pseudo_burst": target,
                "radial_budget_auc": float(r_auc[0]),
                "control_budget_auc": float(c_auc[0]),
                "radial_minus_control_budget_auc": auc_delta,
                "budget_auc_delta_ci95_low": float(auc_low),
                "budget_auc_delta_ci95_high": float(auc_high),
                "radial_b58_macro_recall": float(r_b58[0]),
                "control_b58_macro_recall": float(c_b58[0]),
                "radial_minus_control_b58_macro_recall": b58_delta,
                "b58_delta_ci95_low": float(b58_low),
                "b58_delta_ci95_high": float(b58_high),
                "radial_b58_burst_wins": int(np.sum(r_bursts[0] > c_bursts[0])),
                "radial_b58_burst_ties": int(np.sum(r_bursts[0] == c_bursts[0])),
                "radial_superiority_established_descriptive_gate": bool(
                    auc_low > 0 and b58_low > 0
                ),
                "control_practically_noninferior_at_0p02_point_margin": bool(
                    auc_delta <= NONINFERIORITY_MARGIN
                    and b58_delta <= NONINFERIORITY_MARGIN
                ),
                "control_stronger_on_both_point_metrics": bool(
                    auc_delta < 0 and b58_delta < 0
                ),
                "cluster_field": "canonical_roi_id",
                "cluster_count": len(identities),
                "bootstrap_seed": int(seed),
                "bootstrap_replicates": int(replicates),
                "noninferiority_was_predeclared_in_parent_campaign": False,
                "automatic_pipeline_selection_authorized": False,
            }
        )
    return output


def _control_match_slice(
    rows: Sequence[Mapping[str, Any]], *, control_id: str
) -> list[Mapping[str, Any]]:
    selected = [row for row in rows if str(row["context_role"]) == control_id]
    if not selected:
        raise ValueError(f"control match table lacks {control_id}")
    return selected


def _verify_protected_after_control_seal(
    protected_dir: str | Path,
    *,
    metadata: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(protected_dir).expanduser().resolve()
    verified = _verify_artifact_index(root)
    if verified["artifact_index_sha256"] != metadata["artifact_index_sha256"]:
        raise ValueError("protected comparator index changed during control construction")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    boundary = json.loads((root / "claim_boundary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete_protected_metrics_scientific_audit_pending":
        raise ValueError("radial comparator is not complete protected metrics")
    if validation.get("status") != "passed_protected_metric_artifact_contract" or validation.get(
        "all_checks_pass"
    ) is not True:
        raise ValueError("radial comparator did not pass its protected metric contract")
    if boundary.get("candidate_artifacts_sealed_before_label_join") is not True:
        raise ValueError("radial comparator did not seal candidates before its label join")
    if boundary.get("precision_identified") is not False:
        raise ValueError("radial comparator violates sparse-positive interpretation")
    movie_hash = str(preflight["source"]["movie"]["sha256"])
    if metadata["movie_sha256"] != movie_hash:
        raise ValueError("protected comparator and current preflight movie hashes differ")
    return {
        "artifact_index_sha256": verified["artifact_index_sha256"],
        "summary_sha256": _sha256(root / "summary.json"),
        "validation_sha256": _sha256(root / "validation.json"),
        "protected_metric_contract_passed": True,
    }


def _audit_hook_payload(
    *,
    preflight_dir: Path,
    radial_context_role: str,
) -> dict[str, Any]:
    overlay = preflight_dir / "label_projection_overlay.png"
    if not overlay.is_file():
        raise FileNotFoundError("preflight label-projection overlay is missing")
    return {
        "schema_version": 1,
        "scientific_audit_status": "pending_renderer_and_inventory_validation",
        "projection_overlay": {
            "source": str(overlay),
            "sha256": _sha256(overlay),
            "role": "coordinate_orientation_preflight_only",
        },
        "required_stage_sequences": {
            CONTROL_IDS[0]: [
                "fixed representation",
                "signed square-annulus local standardization",
                "framewise empirical threshold decision",
                "burst threshold-occupancy map",
                "frozen candidate ranking",
            ],
            CONTROL_IDS[1]: [
                "fixed representation",
                "positive clipping",
                "replicate-boundary box CFAR score",
                "framewise empirical threshold decision",
                "burst threshold-occupancy map",
                "frozen candidate ranking",
            ],
            "radial_comparator": [
                "fixed representation",
                f"radial Gamma-LS role {radial_context_role}",
                "framewise empirical threshold decision",
                "burst threshold-occupancy map",
                "frozen candidate ranking",
            ],
        },
        "required_sections": [
            "1_Expert_Annotations",
            "2_Model_Annotations",
            "3_Comparison",
        ],
        "section_purity_required": True,
        "comparison_default": "figures_and_tables_only",
        "nearest_candidate_and_one_to_one_assignment_must_remain_distinct": True,
        "unmatched_candidates": "unknown_not_negative",
        "metric_artifact_may_complete_before_scientific_audit": True,
    }


def run_protected_control_comparison(
    config: GammaLSDifferenceConfig,
    *,
    preflight_dir: str | Path,
    support_dir: str | Path,
    protected_dir: str | Path,
    output_dir: str | Path,
    radial_context_role: str,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run the protected modern-control comparison; never infer precision."""

    if not isinstance(config, GammaLSDifferenceConfig):
        raise TypeError("config must be a validated GammaLSDifferenceConfig")
    if not radial_context_role.strip():
        raise ValueError("radial_context_role must be explicit and non-empty")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"protected control output exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"protected control output parent is missing: {destination.parent}")
    work = destination.parent / f".{destination.name}.protected-controls-work"
    if work.exists():
        raise FileExistsError(
            f"protected control work directory already exists and will not be reused: {work}"
        )

    # All readiness gates precede destination or temporary-work mutation.
    preflight = verify_matching_preflight(config, preflight_dir, require_gpu_ready=True)
    support_evidence = audit_support_control_need(support_dir)
    protected_metadata = _protected_index_metadata(
        protected_dir,
        radial_context_role=radial_context_role,
        support_evidence=support_evidence,
    )
    if protected_metadata["movie_sha256"] != preflight["source"]["movie"]["sha256"]:
        raise ValueError("protected comparator movie differs from the current preflight")
    runtime, movie = _source_and_resource_gates(
        config, destination=destination, device=device
    )

    contract = {
        "schema_version": PROTOCOL_VERSION,
        "experiment_id": config.experiment_id,
        "run_type": "protected_simple_control_comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "portable_config_sha256": _canonical_sha256(config.portable_dict()),
        "executor_sha256": _sha256(Path(__file__).resolve()),
        "preflight_sha256": _sha256(Path(preflight_dir).resolve() / "preflight.json"),
        "movie_sha256": preflight["source"]["movie"]["sha256"],
        "support_evidence": support_evidence,
        "protected_comparator_metadata": protected_metadata,
        "controls": list(CONTROL_IDS),
        "representations": list(FIXED_ARMS),
        "quiet_swaps": list(QUIET_SWAPS),
        "nms_distances_px": list(NMS_DISTANCES_PX),
        "quiet_burdens": list(QUIET_NMS_PEAK_BURDENS),
        "candidate_budgets_per_burst": list(CANDIDATE_BUDGETS_PER_BURST),
        "match_radius_px": MATCH_RADIUS_PX,
        "radial_context_role": radial_context_role,
        "annotation_content_access": (
            "Candidate construction/model scoring does not access annotation content "
            "after the existing eligibility preflight. Preflight had already inspected "
            "annotation content, so this is not described as end-to-end label blind."
        ),
    }
    work.mkdir()
    _atomic_json(work / "run_contract.json", contract)

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
        import torch

        heartbeat("causal_preprocessing")
        start_ui, stop_ui = map(int, config.payload["frames"]["review_interval_ui"])
        chunk_frames = max(int(value) for value in config.payload["efficiency"]["frame_chunks"])
        common, preprocessing_timing = _stream_common_history_to_device(
            config.source_paths["movie"],
            review_start_ui=start_ui,
            review_stop_ui=stop_ui,
            chunk_frames=chunk_frames,
            device=torch.device(str(runtime["resolved_device"])),
            heartbeat=lambda payload: heartbeat(
                "causal_preprocessing", upstream_progress=dict(payload)
            ),
        )
        frame_ui = np.arange(start_ui, stop_ui + 1, dtype=np.int64)
        folds = build_fold_contracts(config)
        bursts = config.payload["frames"]["burst_intervals_ui"]
        scale_floor_percentile = float(
            config.payload["gamma_ls_grid"]["finalist_scale_floor_percentiles"][0]
        )
        candidate_rows: list[dict[str, Any]] = []
        calibration_rows: list[dict[str, Any]] = []
        timing_rows: list[dict[str, Any]] = []
        completed = 0
        total = len(FIXED_ARMS) * len(CONTROL_IDS)
        max_vram = int(float(config.payload["resources"]["max_peak_vram_gib"]) * 2**30)
        for representation_name in FIXED_ARMS:
            representation = _representation(common, representation_name)
            for control_id in CONTROL_IDS:
                torch.cuda.reset_peak_memory_stats(representation.device)
                candidates, calibrations, timing = build_control_candidates(
                    representation,
                    representation_name=representation_name,
                    control_id=control_id,
                    frame_ui=frame_ui,
                    folds=folds,
                    bursts=bursts,
                    scale_floor_percentile=scale_floor_percentile,
                    chunk_frames=chunk_frames,
                )
                peak = int(torch.cuda.max_memory_allocated(representation.device))
                if peak > max_vram:
                    raise ProtectedControlUnavailable(
                        f"control {control_id}/{representation_name} exceeded VRAM cap"
                    )
                timing["max_memory_allocated_bytes"] = peak
                candidate_rows.extend(candidates)
                calibration_rows.extend(calibrations)
                timing_rows.append(timing)
                completed += 1
                heartbeat(
                    "control_candidate_freeze",
                    completed_control_representation_cells=completed,
                    total_control_representation_cells=total,
                )
            del representation
            torch.cuda.empty_cache()
        del common

        _atomic_tsv(work / "control_candidates_label_sealed.tsv", candidate_rows)
        _atomic_tsv(work / "control_threshold_calibration.tsv", calibration_rows)
        timing_table_rows = [
            {
                **row,
                "score_hashes_by_quiet_swap": json.dumps(
                    row["score_hashes_by_quiet_swap"], sort_keys=True
                ),
            }
            for row in timing_rows
        ]
        _atomic_tsv(work / "control_timings.tsv", timing_table_rows)
        score_hash_manifest = {
            "schema_version": 1,
            "algorithm": "sha256(dtype || shape || contiguous_float32_score_bytes)",
            "rows": [
                {
                    "control_id": row["control_id"],
                    "representation": row["representation"],
                    "score_hashes_by_quiet_swap": row["score_hashes_by_quiet_swap"],
                }
                for row in timing_rows
            ],
        }
        _atomic_json(work / "control_score_hashes.json", score_hash_manifest)
        candidate_hashes = _candidate_hashes(candidate_rows)
        candidate_seal = {
            "schema_version": 1,
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_table_sha256": _sha256(
                work / "control_candidates_label_sealed.tsv"
            ),
            "candidate_table_rows": len(candidate_rows),
            "threshold_table_sha256": _sha256(
                work / "control_threshold_calibration.tsv"
            ),
            "threshold_table_rows": len(calibration_rows),
            "timing_table_sha256": _sha256(work / "control_timings.tsv"),
            "score_hash_manifest_sha256": _sha256(
                work / "control_score_hashes.json"
            ),
            **candidate_hashes,
            "sparse_positive_fields_parsed_before_seal": False,
            "label_derived_radial_match_table_opened_before_seal": False,
            "annotation_content_accessed_by_candidate_construction": False,
            "preflight_previously_inspected_annotation_content": True,
            "end_to_end_label_blind_claimed": False,
            "unmatched_candidates": "unknown_not_negative",
        }
        _atomic_json(work / "candidate_seal.json", candidate_seal)
        _verify_seal_files(work, candidate_seal)

        # Only byte fingerprints were checked above. Annotation tables and the
        # radial label-derived match table are first parsed after this seal.
        heartbeat(
            "protected_label_join",
            candidate_seal_sha256=_sha256(work / "candidate_seal.json"),
        )
        for key in ("protected_labels_v1", "latest_labels_v7"):
            expected_hash = preflight["source"][key]["sha256"]
            if _sha256(config.source_paths[key]) != expected_hash:
                raise ValueError(f"annotation source changed after candidate seal: {key}")
        protected_verified = _verify_protected_after_control_seal(
            protected_dir,
            metadata=protected_metadata,
            preflight=preflight,
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
        v1_recall = aggregate_match_rows(v1_matches)
        v7_recall = aggregate_match_rows(v7_matches)
        v1_summary = _summary_by_arm(v1_matches)
        v7_summary = _summary_by_arm(v7_matches)

        protected_root = Path(protected_dir).expanduser().resolve()
        radial_matches = _typed_radial_match_rows(
            protected_root / "protected_v1_observation_matches.tsv",
            radial_context_role=radial_context_role,
        )
        contrasts: list[dict[str, Any]] = []
        for control_id in CONTROL_IDS:
            contrasts.extend(
                paired_radial_control_bootstrap(
                    radial_matches,
                    _control_match_slice(v1_matches, control_id=control_id),
                )
            )
        radial_calibration = _typed_radial_calibration_rows(
            protected_root / "threshold_calibration.tsv",
            radial_context_role=radial_context_role,
        )
        quiet_comparison = quiet_burden_comparison_rows(
            radial_calibration, calibration_rows
        )

        _atomic_tsv(work / "protected_v1_observation_matches.tsv", v1_matches)
        _atomic_tsv(work / "protected_v1_recall.tsv", v1_recall)
        _atomic_tsv(work / "protected_v1_control_summary.tsv", v1_summary)
        _atomic_tsv(work / "latest_v7_observation_matches.tsv", v7_matches)
        _atomic_tsv(work / "latest_v7_sensitivity.tsv", v7_recall)
        _atomic_tsv(work / "latest_v7_control_summary.tsv", v7_summary)
        _atomic_tsv(work / "radial_control_bootstrap_contrasts.tsv", contrasts)
        _atomic_tsv(work / "radial_control_quiet_burden_comparison.tsv", quiet_comparison)

        overlay_dir = work / "audit_hooks"
        overlay_dir.mkdir()
        preflight_overlay = Path(preflight_dir).resolve() / "label_projection_overlay.png"
        shutil.copyfile(
            preflight_overlay,
            overlay_dir / "protected_label_projection_overlay.png",
        )
        audit_hooks = _audit_hook_payload(
            preflight_dir=Path(preflight_dir).resolve(),
            radial_context_role=radial_context_role,
        )
        audit_hooks["copied_projection_overlay"] = {
            "path": "audit_hooks/protected_label_projection_overlay.png",
            "sha256": _sha256(overlay_dir / "protected_label_projection_overlay.png"),
        }
        _atomic_json(work / "scientific_audit_hooks.json", audit_hooks)
        _atomic_text(
            overlay_dir / "README.md",
            "# Protected control scientific-audit hook\n\n"
            "The metric comparison is complete only when the run finishes. The full "
            "three-section scientific-audit media set remains pending; use "
            "`../scientific_audit_hooks.json` and the frozen candidate/score hashes as "
            "renderer inputs. Unmatched candidates remain unknown, not negatives.\n",
        )

        primary_contrasts = [
            row
            for row in contrasts
            if row["quiet_swap"] == "crossfit_average"
            and row["nms_distance_px"] == NMS_DISTANCE_PX
            and math.isclose(
                float(row["target_nms_peaks_per_pseudo_burst"]), 1.0
            )
        ]
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
            "candidate_construction_and_scoring_accessed_annotation_content_after_eligibility_preflight": False,
            "eligibility_preflight_previously_inspected_annotation_content": True,
            "end_to_end_label_blind_claimed": False,
            "candidate_artifacts_sealed_before_label_join": True,
            "radial_label_derived_results_opened_after_control_seal": True,
            "unmatched_candidates": "unknown_not_negative",
            "precision_identified": False,
            "v7_inferential_claim": False,
            "noninferiority_margin_predeclared_in_parent_campaign": False,
            "automatic_pipeline_selection_authorized": False,
            "scientific_audit_complete": False,
            "paper_promotion_ready": False,
        }
        summary = {
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "run_type": "protected_simple_control_comparison",
            "status": "complete_protected_control_metrics_scientific_audit_pending",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime,
            "preprocessing_timing": preprocessing_timing,
            "support_control_need_audit": support_evidence,
            "protected_radial_comparator": {
                **protected_verified,
                "context_role": radial_context_role,
            },
            "candidate_row_count": len(candidate_rows),
            "candidate_seal": candidate_seal,
            "protected_v1": {
                "occurrences": len(v1),
                "canonical_identities": len(
                    {row["canonical_roi_id"] for row in v1}
                ),
                "control_summary": v1_summary,
                "paired_radial_control_primary_rows": primary_contrasts,
            },
            "latest_v7_sensitivity": {
                "occurrences": len(v7),
                "canonical_identities": len(
                    {row["canonical_roi_id"] for row in v7}
                ),
                "inferential_claim": False,
            },
            "claim_boundary": claim_boundary,
        }
        _atomic_json(work / "claim_boundary.json", claim_boundary)
        _atomic_json(work / "summary.json", summary)
        checks = {
            "support_artifact_verified": True,
            "two_modern_controls_executed": {
                row["control_id"] for row in timing_rows
            }
            == set(CONTROL_IDS),
            "three_fixed_representations_executed": {
                row["representation"] for row in timing_rows
            }
            == set(FIXED_ARMS),
            "four_outer_folds": len(folds) == 4,
            "candidate_seal_precedes_label_join": True,
            "radial_protected_artifact_verified_after_seal": True,
            "protected_v1_uses_79_inclusive": len(v1) == 79,
            "protected_v1_has_26_identity_clusters": len(
                {row["canonical_roi_id"] for row in v1}
            )
            == 26,
            "v7_uses_106_confirmed": len(v7) == 106,
            "precision_not_claimed": True,
            "scientific_audit_pending": True,
        }
        _atomic_json(
            work / "validation.json",
            {
                "status": "passed_protected_control_metric_artifact_contract",
                "checks": checks,
                "all_checks_pass": all(checks.values()),
            },
        )
        _atomic_json(
            work / "llm_context.json",
            {
                "entrypoint": "summary.json",
                "question": "Does modern radial Gamma-LS outperform either simpler executed control at matched empirical quiet burdens?",
                "candidate_seal": "candidate_seal.json",
                "protected_primary": "protected_v1_control_summary.tsv",
                "paired_comparison": "radial_control_bootstrap_contrasts.tsv",
                "quiet_calibration_check": "radial_control_quiet_burden_comparison.tsv",
                "latest_sensitivity": "latest_v7_control_summary.tsv",
                "scientific_audit_hooks": "scientific_audit_hooks.json",
                "limitations": [
                    "sparse positives do not identify precision",
                    "v7 is candidate-assisted descriptive sensitivity",
                    "the 0.02 practical margin was not predeclared for the parent campaign",
                    "this executor does not automatically select a deployment pipeline",
                    "scientific-audit media remain pending",
                ],
            },
        )
        _atomic_text(
            work / "REPORT.md",
            "# Protected simple-control comparison\n\n"
            "The completed metric artifact compares fresh signed square-annulus and "
            "maintained positive-clipped box-CFAR candidates against the explicitly "
            f"selected radial role `{radial_context_role}`. Read `summary.json`, "
            "`radial_control_bootstrap_contrasts.tsv`, and "
            "`radial_control_quiet_burden_comparison.tsv`.\n\n"
            "This is sparse-positive recall evidence only: unmatched candidates are "
            "unknown and precision is not identified. Candidate construction/scoring "
            "did not access annotation content after the pre-existing eligibility "
            "preflight; the overall workflow is not end-to-end label blind. Full "
            "scientific-audit media and owner interpretation remain pending.\n",
        )
        _atomic_json(
            work / "status.json",
            {
                "status": "complete_protected_control_metrics_scientific_audit_pending",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        _atomic_json(work / "artifact_index.json", _artifact_index(work))
        if destination.exists():
            raise FileExistsError(
                f"protected control output appeared during execution: {destination}"
            )
        work.replace(destination)
        return summary
    except Exception as error:
        if work.exists():
            _atomic_json(
                work / "status.json",
                {
                    "status": "interrupted_or_failed_not_promotable",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": repr(error),
                    "safe_to_resume": False,
                    "reason": "use a new noncolliding output after diagnosing the failure",
                },
            )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the sealed protected comparison of two simple Gamma-LS controls."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--support-screen", required=True)
    parser.add_argument("--protected-output", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--radial-context-role", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = GammaLSDifferenceConfig.load(args.config)
    summary = run_protected_control_comparison(
        config,
        preflight_dir=args.preflight,
        support_dir=args.support_screen,
        protected_dir=args.protected_output,
        output_dir=args.artifact_dir,
        radial_context_role=args.radial_context_role,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by exact run command
    raise SystemExit(main())


__all__ = [
    "CONTROL_IDS",
    "NONINFERIORITY_MARGIN",
    "ProtectedControlUnavailable",
    "audit_support_control_need",
    "build_control_candidates",
    "main",
    "paired_radial_control_bootstrap",
    "quiet_burden_comparison_rows",
    "run_protected_control_comparison",
]
