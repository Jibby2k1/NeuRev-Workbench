"""Strict CPU-only finalizer for a sealed matched-six-lag partial run.

The GPU executor may terminate after sealing its complete label-free candidate
universe.  This module never refits or rescores.  It verifies that exact
pre-label state, reconstructs only the sparse-positive joins and summaries,
uses the parity-tested vectorized 26-identity bootstrap, and atomically promotes
the work directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import GammaLSDifferenceConfig
from .evaluation import CANDIDATE_BUDGETS_PER_BURST, NMS_DISTANCE_PX
from . import multilag_protected as executor_module
from .multilag_protected import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIRMATION_SAMPLES,
    CS_PARZEN_BANDWIDTHS,
    MULTILAG_ARMS,
    NMS_DISTANCES_PX,
    SAMPLE_SEEDS,
    SCREEN_SAMPLES,
    _artifact_index,
    _atomic_json,
    _atomic_text,
    _atomic_tsv,
    _canonical_sha256,
    _sha256,
    multilag_design,
    select_fold_models,
)
from .protected import (
    QUIET_SWAPS,
    _read_sparse_positives,
    _summary_by_arm,
    aggregate_match_rows,
    observation_match_rows,
)
from .protected_finalize import (
    EXPECTED_QUIET_BURDENS,
    EXPECTED_V1_IDENTITIES,
    EXPECTED_V1_ROWS,
    EXPECTED_V7_IDENTITIES,
    EXPECTED_V7_ROWS,
    ProtectedFinalizeUnavailable,
    _read_tsv,
    _strict_bool,
    vectorized_clustered_bootstrap_contrasts,
)


EXPECTED_EXIT_CODE = 139
EXPECTED_CONTEXT_ROLE = "size_sufficient_candidate"
EXPECTED_FIT_MODELS = 24
EXPECTED_FIT_SELECTION_CELLS = 36
EXPECTED_FIT_ROWS = 24
EXPECTED_SELECTED_ROWS = 8
EXPECTED_CANDIDATE_CELLS = 12
EXPECTED_CANDIDATE_ROWS = 1559
EXPECTED_CALIBRATION_ROWS = 360
EXPECTED_TIMING_ROWS = 12
EXPECTED_EQUIVALENCE_ROWS = 14


@dataclass(frozen=True)
class VerifiedMultilagPartial:
    work: Path
    destination: Path
    contract: Mapping[str, Any]
    preflight: Mapping[str, Any]
    candidate_seal: Mapping[str, Any]
    original_state: Mapping[str, Any]
    fit_rows: Sequence[Mapping[str, str]]
    selected_rows: Sequence[Mapping[str, str]]
    candidate_rows: Sequence[Mapping[str, str]]
    calibration_rows: Sequence[Mapping[str, str]]
    timing_rows: Sequence[Mapping[str, str]]
    equivalence_rows: Sequence[Mapping[str, str]]
    fit_models: Sequence[Mapping[str, Any]]
    upstream_sha256: Mapping[str, str]


def _recorded_termination(
    work: Path, *, observed_exit_code: int
) -> dict[str, Any]:
    """Validate either the initial SIGSEGV state or a finalizer retry."""

    status_path = work / "status.json"
    if not status_path.exists():
        if int(observed_exit_code) != EXPECTED_EXIT_CODE:
            raise ProtectedFinalizeUnavailable(
                "status-less sealed partial requires operator-observed exit 139"
            )
        return {
            "original_status_file_present": False,
            "operator_observed_exit_code": EXPECTED_EXIT_CODE,
            "interpretation": "SIGSEGV_after_candidate_seal",
        }
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (
        status.get("status") != "finalizer_failed_resumable"
        or int(status.get("original_operator_observed_exit_code", -1))
        != EXPECTED_EXIT_CODE
        or status.get("original_status_file_present") is not False
    ):
        raise ProtectedFinalizeUnavailable(
            "existing multilag status is not a recognized finalizer retry"
        )
    return {
        "original_status_file_present": False,
        "operator_observed_exit_code": EXPECTED_EXIT_CODE,
        "interpretation": "SIGSEGV_after_candidate_seal",
        "prior_finalizer_error": status.get("error"),
    }


def _exact_values(
    rows: Sequence[Mapping[str, Any]], field: str, expected: Sequence[Any]
) -> None:
    observed = {str(row[field]) for row in rows}
    wanted = {str(value) for value in expected}
    if observed != wanted:
        raise ProtectedFinalizeUnavailable(
            f"{field} changed: observed={sorted(observed)}, expected={sorted(wanted)}"
        )


def _counter_from_json_rows(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> Counter[tuple[str, ...]]:
    return Counter(tuple(str(row[field]) for field in fields) for row in rows)


def _counter_from_tsv_rows(
    rows: Sequence[Mapping[str, str]], fields: Sequence[str]
) -> Counter[tuple[str, ...]]:
    return Counter(tuple(row[field] for field in fields) for row in rows)


def _expected_raw_source_sha256(
    config: GammaLSDifferenceConfig, movie_sha256: str
) -> str:
    review_start, review_stop = map(
        int, config.payload["frames"]["review_interval_ui"]
    )
    return _canonical_sha256(
        {
            "movie_sha256": movie_sha256,
            "input_domain": "acquisition_raw",
            "preprocessing": "none",
            "loaded_source_frame_ui": [review_start - 16, review_stop],
            "review_output_frame_ui": [review_start, review_stop],
            "lags": [0, 1, 2, 4, 8, 16],
        }
    )


def _typed_fit_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    integer_fields = {
        "training_fold",
        "heldout_burst",
        "sample_seed",
        "sweeps",
        "accepted_updates",
        "pairwise_cs_objective_calls",
        "numerical_clamps",
        "persistence_index",
        "innovation_index",
    }
    float_fields = {
        "bandwidth",
        "whitening_fit_seconds",
        "cs_parzen_rotation_fit_seconds",
        "fit_seconds",
        "fit_seconds_excluding_shared_whitening",
        "confirmation_objective",
        "confirmation_baseline_objective",
        "confirmation_gain_fraction",
        "whitening_condition_number",
        "full_rotation_total_energy_relative_max_error",
        "pca_mean_positive_tail_contrast",
        "pca_minimum_quiet_swap_positive_tail_contrast",
        "ica_mean_positive_tail_contrast",
        "ica_minimum_quiet_swap_positive_tail_contrast",
    }
    boolean_fields = {
        "whitening_reused_across_bandwidths",
        "converged",
        "fit_uses_only_outer_training_burst_windows",
        "selection_event_windows_are_outer_training_bursts_only",
        "selection_quiet_reference_is_predeclared_crossfit_quiet_halves",
        "positive_coordinates_used",
        "positive_identities_used",
        "archived_fit_loaded",
    }
    for row in rows:
        typed: dict[str, Any] = dict(row)
        for field in integer_fields:
            typed[field] = int(row[field])
        for field in float_fields:
            typed[field] = float(row[field])
        for field in boolean_fields:
            typed[field] = _strict_bool(row[field], field=field)
        result.append(typed)
    return result


def _audit_fit_models(
    config: GammaLSDifferenceConfig,
    contract: Mapping[str, Any],
    work: Path,
    fit_rows: Sequence[Mapping[str, str]],
    selected_rows: Sequence[Mapping[str, str]],
) -> tuple[list[Path], list[Mapping[str, Any]]]:
    paths = sorted((work / "fit_models").glob("*.json"))
    if len(paths) != EXPECTED_FIT_MODELS:
        raise ProtectedFinalizeUnavailable(
            f"expected 24 fit checkpoints, found {len(paths)}"
        )
    expected_keys = {
        (fold, seed, bandwidth)
        for fold in (1, 2, 3, 4)
        for seed in SAMPLE_SEEDS
        for bandwidth in CS_PARZEN_BANDWIDTHS
    }
    models: dict[tuple[int, int, float], Mapping[str, Any]] = {}
    shared: dict[tuple[int, int], Mapping[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (
            int(payload["training_fold"]),
            int(payload["sample_seed"]),
            float(payload["bandwidth"]),
        )
        if key not in expected_keys or key in models:
            raise ProtectedFinalizeUnavailable(
                f"fit checkpoint key is duplicate or unexpected: {key}"
            )
        manifest = payload["sample_manifest"]
        fit = payload["fit"]
        expected_training = {
            str(value)
            for value in config.payload["frames"]["burst_intervals_ui"]
            if int(value) != key[0]
        }
        if (
            manifest.get("input_domain") != "acquisition_raw"
            or manifest.get("lags") != [0, 1, 2, 4, 8, 16]
            or manifest.get("guard_rule")
            != "every_t_minus_lag_outside_inclusive_heldout_guard"
            or set(manifest.get("training_bursts", ())) != expected_training
            or str(manifest.get("heldout_burst")) != str(key[0])
            or int(manifest.get("screen_sample_count", -1)) != SCREEN_SAMPLES
            or int(manifest.get("confirmation_sample_count", -1))
            != CONFIRMATION_SAMPLES
            or manifest.get("screen_confirmation_disjoint") is not True
            or manifest.get("positive_coordinates_used") is not False
            or manifest.get("positive_identities_used") is not False
            or any(
                int(counts[split]) != 128
                for counts in manifest.get("counts_by_training_burst", {}).values()
                for split in ("screen_samples", "confirmation_samples")
            )
            or fit.get("formulation") != "delay_embedding"
            or fit.get("objective_family") != "cs_parzen"
            or fit.get("lags") != [0, 1, 2, 4, 8, 16]
            or fit.get("converged") is not True
            or fit["diagnostics"].get("backend") != "cuda"
            or fit["diagnostics"].get("numerical_clamps") != 0
            or fit["diagnostics"].get("positive_coordinates_used") is not False
            or fit["diagnostics"].get("positive_identities_used") is not False
            or fit["diagnostics"].get("component_rule")
            != (
                "persistence=max_abs_common_axis; innovation=max_abs_t_minus_t1_"
                "axis_excluding_persistence; residual=all_other_four_components"
            )
        ):
            raise ProtectedFinalizeUnavailable(
                f"fit checkpoint violates the raw/fold-local contract: {path.name}"
            )
        whitening = np.asarray(fit["whitening"], dtype=np.float64)
        rotation = np.asarray(fit["rotation"], dtype=np.float64)
        demixing = np.asarray(fit["demixing"], dtype=np.float64)
        if (
            whitening.shape != (6, 6)
            or rotation.shape != (6, 6)
            or demixing.shape != (6, 6)
            or np.linalg.matrix_rank(whitening) != 6
            or np.linalg.matrix_rank(rotation) != 6
            or not np.allclose(demixing, rotation @ whitening, rtol=1e-12, atol=1e-12)
            or float(
                fit["diagnostics"]["full_rotation_total_energy_relative_max_error"]
            )
            > 1e-10
        ):
            raise ProtectedFinalizeUnavailable(
                f"fit checkpoint violates full-rank rotation invariance: {path.name}"
            )
        normalized = demixing / np.maximum(
            np.linalg.norm(demixing, axis=1, keepdims=True), 1e-12
        )
        common = np.ones(6, dtype=np.float64) / math.sqrt(6.0)
        difference = np.asarray([1, -1, 0, 0, 0, 0], dtype=np.float64)
        difference /= np.linalg.norm(difference)
        persistence = int(np.argmax(np.abs(normalized @ common)))
        innovation = max(
            (index for index in range(6) if index != persistence),
            key=lambda index: abs(float(normalized[index] @ difference)),
        )
        residual = set(range(6)) - {persistence, innovation}
        if (
            int(fit["persistence_index"]) != persistence
            or int(fit["innovation_index"]) != innovation
            or set(map(int, fit["residual_indices"])) != residual
            or float(normalized[persistence] @ common) < -1e-12
            or float(normalized[innovation] @ difference) < -1e-12
        ):
            raise ProtectedFinalizeUnavailable(
                f"analytic persistence/difference-axis rule changed: {path.name}"
            )
        pair = key[:2]
        frozen_shared = {
            "sample_manifest": manifest,
            "center": fit["center"],
            "whitening": fit["whitening"],
        }
        if pair in shared and shared[pair] != frozen_shared:
            raise ProtectedFinalizeUnavailable(
                "bandwidth fits did not reuse one fold/seed whitening and sample"
            )
        shared[pair] = frozen_shared
        models[key] = payload
    if set(models) != expected_keys or len(shared) != 12:
        raise ProtectedFinalizeUnavailable("fit grid is incomplete")
    anatomy_counts = {
        int(payload["sample_manifest"]["anatomy_pixel_count"])
        for payload in models.values()
    }
    if len(anatomy_counts) != 1:
        raise ProtectedFinalizeUnavailable(
            "fit checkpoints disagree on the shared anatomy-mask population"
        )

    typed_rows = _typed_fit_rows(fit_rows)
    if {
        (row["training_fold"], row["sample_seed"], row["bandwidth"])
        for row in typed_rows
    } != expected_keys:
        raise ProtectedFinalizeUnavailable("fit-grid TSV keys changed")
    model_by_key = models
    for row in typed_rows:
        payload = model_by_key[
            (row["training_fold"], row["sample_seed"], row["bandwidth"])
        ]
        fit = payload["fit"]
        manifest = payload["sample_manifest"]
        whitening_sha = _canonical_sha256(
            {"center": fit["center"], "whitening": fit["whitening"]}
        )
        if (
            row["sample_identity_sha256"]
            != manifest["sample_identity_sha256"]
            or row["whitening_sha256"] != whitening_sha
            or row["converged"] is not True
            or row["whitening_reused_across_bandwidths"] is not True
            or row["fit_uses_only_outer_training_burst_windows"] is not True
            or row["selection_event_windows_are_outer_training_bursts_only"]
            is not True
            or row["selection_quiet_reference_is_predeclared_crossfit_quiet_halves"]
            is not True
            or row["positive_coordinates_used"] is not False
            or row["positive_identities_used"] is not False
            or row["archived_fit_loaded"] is not False
        ):
            raise ProtectedFinalizeUnavailable(
                "fit-grid TSV disagrees with a checkpoint or leakage contract"
            )

    selected_by_fold_arm = {
        (int(row["training_fold"]), row["selected_for_representation"]): row
        for row in selected_rows
    }
    if len(selected_by_fold_arm) != EXPECTED_SELECTED_ROWS:
        raise ProtectedFinalizeUnavailable("selected-fit rows are not unique")
    for fold in (1, 2, 3, 4):
        pca, ica = select_fold_models(typed_rows, fold=fold)
        for arm, winner in (
            ("pca_whitened_delay_total_energy", pca),
            ("cs_parzen_delay_residual", ica),
        ):
            selected = selected_by_fold_arm[(fold, arm)]
            if (
                int(selected["sample_seed"]) != int(winner["sample_seed"])
                or selected["sample_identity_sha256"]
                != winner["sample_identity_sha256"]
                or _strict_bool(
                    selected["positive_coordinates_used"],
                    field="positive_coordinates_used",
                )
                or _strict_bool(
                    selected["positive_identities_used"],
                    field="positive_identities_used",
                )
                or (
                    arm == "cs_parzen_delay_residual"
                    and float(selected["bandwidth"]) != float(winner["bandwidth"])
                )
                or (
                    arm == "pca_whitened_delay_total_energy"
                    and selected["bandwidth"] != "not_applicable_to_whitening"
                )
            ):
                raise ProtectedFinalizeUnavailable(
                    "selected-fit TSV does not reproduce the label-free rule"
                )
    return paths, list(models.values())


def _upstream_hashes(work: Path) -> dict[str, str]:
    paths = []
    for name in (
        "run_contract.json",
        "candidate_seal.json",
        "candidates_label_sealed.tsv",
        "threshold_calibration.tsv",
        "fit_grid.tsv",
        "selected_fit_rows.tsv",
        "timings.tsv",
        "representation_equivalence.tsv",
    ):
        paths.append(work / name)
    for directory in ("fit_models", "fit_selection_cells", "candidate_cells"):
        paths.extend(sorted((work / directory).glob("*.json")))
    return {
        path.relative_to(work).as_posix(): _sha256(path)
        for path in sorted(paths)
    }


def verify_multilag_partial(
    config: GammaLSDifferenceConfig,
    *,
    preflight_dir: str | Path,
    context_selection_dir: str | Path,
    destination: str | Path,
    expected_candidate_seal_sha256: str,
    observed_exit_code: int,
) -> VerifiedMultilagPartial:
    """Verify the exact stable partial before opening either label table."""

    destination_path = Path(destination).expanduser().resolve()
    work = destination_path.parent / f".{destination_path.name}.multilag-protected-work"
    if destination_path.exists():
        raise FileExistsError(f"matched-six-lag destination exists: {destination_path}")
    if not work.is_dir():
        raise FileNotFoundError(f"matched-six-lag partial is missing: {work}")
    if list(work.rglob("*.partial")):
        raise ProtectedFinalizeUnavailable("partial contains an unfinished atomic file")
    original_state = _recorded_termination(
        work, observed_exit_code=int(observed_exit_code)
    )
    required = (
        "run_contract.json",
        "candidate_seal.json",
        "candidates_label_sealed.tsv",
        "threshold_calibration.tsv",
        "fit_grid.tsv",
        "selected_fit_rows.tsv",
        "timings.tsv",
        "representation_equivalence.tsv",
        "heartbeat.json",
    )
    missing = [name for name in required if not (work / name).is_file()]
    if missing:
        raise ProtectedFinalizeUnavailable(f"partial lacks required files: {missing}")

    contract = json.loads((work / "run_contract.json").read_text(encoding="utf-8"))
    if (
        contract.get("run_type")
        != "protected_outer_fold_matched_six_lag_representation"
        or contract.get("portable_config_sha256")
        != _canonical_sha256(config.portable_dict())
        or contract.get("design") != multilag_design()
        or contract.get("selected_context_role") != EXPECTED_CONTEXT_ROLE
        or contract.get("sparse_positive_fields_available_to_fit_or_selection")
        is not False
        or contract.get("archived_fit_loaded_for_projection") is not False
        or int(contract.get("bootstrap", {}).get("seed", -1))
        != BOOTSTRAP_SEED
        or int(contract.get("bootstrap", {}).get("replicates", -1))
        != BOOTSTRAP_REPLICATES
    ):
        raise ProtectedFinalizeUnavailable("run contract changed")
    source_hashes = {
        "executor_sha256": Path(executor_module.__file__).resolve(),
        "protected_helpers_sha256": Path(executor_module.__file__).with_name(
            "protected.py"
        ),
        "gpu_representations_sha256": Path(executor_module.__file__).with_name(
            "gpu_representations.py"
        ),
        "multilag_algorithm_sha256": config.repository
        / "neurobench/algorithms/multilag_msica.py",
    }
    for field, path in source_hashes.items():
        if _sha256(path) != str(contract[field]):
            raise ProtectedFinalizeUnavailable(f"source implementation changed: {field}")

    preflight_path = Path(preflight_dir).expanduser().resolve() / "preflight.json"
    if _sha256(preflight_path) != str(contract["preflight_sha256"]):
        raise ProtectedFinalizeUnavailable("preflight bytes changed")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "ready"
        or preflight.get("gpu_run_ready") is not True
        or preflight["source"]["movie"]["sha256"] != contract["movie_sha256"]
        or _sha256(config.source_paths["movie"]) != contract["movie_sha256"]
    ):
        raise ProtectedFinalizeUnavailable("preflight/movie contract is not ready")
    for source_key in ("protected_labels_v1", "latest_labels_v7"):
        if _sha256(config.source_paths[source_key]) != str(
            preflight["source"][source_key]["sha256"]
        ):
            raise ProtectedFinalizeUnavailable(
                f"label source changed since preflight: {source_key}"
            )
    context_path = (
        Path(context_selection_dir).expanduser().resolve() / "fold_contexts.json"
    )
    if _sha256(context_path) != str(contract["context_selection_source_sha256"]):
        raise ProtectedFinalizeUnavailable("fold-context source changed")
    historical = contract["historical_grid_basis"]
    if (
        historical.get("fit_matrices_loaded_for_protected_projection") is not False
        or _sha256(Path(historical["surface_path"])) != historical["surface_sha256"]
    ):
        raise ProtectedFinalizeUnavailable("historical v5 diagnostic source changed")

    seal_path = work / "candidate_seal.json"
    if _sha256(seal_path) != str(expected_candidate_seal_sha256):
        raise ProtectedFinalizeUnavailable("candidate-seal hash differs from supplied hash")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    candidate_path = work / str(seal["candidate_table"]["path"])
    if (
        candidate_path != work / "candidates_label_sealed.tsv"
        or int(seal["candidate_table"]["rows"]) != EXPECTED_CANDIDATE_ROWS
        or _sha256(candidate_path) != seal["candidate_table"]["sha256"]
        or _sha256(work / "threshold_calibration.tsv")
        != seal["threshold_table_sha256"]
        or _sha256(work / "fit_grid.tsv") != seal["fit_grid_table_sha256"]
        or _sha256(work / "selected_fit_rows.tsv")
        != seal["selected_fit_table_sha256"]
        or _sha256(work / "representation_equivalence.tsv")
        != seal["equivalence_table_sha256"]
        or seal.get("sparse_positive_fields_parsed_before_seal") is not False
        or seal.get("positive_coordinates_used") is not False
        or seal.get("positive_identities_used") is not False
        or seal.get("unmatched_candidates") != "unknown_not_negative"
    ):
        raise ProtectedFinalizeUnavailable("candidate seal or sealed table changed")
    heartbeat = json.loads((work / "heartbeat.json").read_text(encoding="utf-8"))
    sealed_at = datetime.fromisoformat(str(seal["sealed_at_utc"]))
    joined_at = datetime.fromisoformat(str(heartbeat["updated_at_utc"]))
    if (
        heartbeat.get("stage") != "protected_label_join"
        or heartbeat.get("status") != "running"
        or heartbeat.get("candidate_seal_sha256")
        != expected_candidate_seal_sha256
        or sealed_at > joined_at
    ):
        raise ProtectedFinalizeUnavailable("seal-before-label heartbeat evidence changed")

    fit_rows = _read_tsv(work / "fit_grid.tsv")
    selected_rows = _read_tsv(work / "selected_fit_rows.tsv")
    candidate_rows = _read_tsv(work / "candidates_label_sealed.tsv")
    calibration_rows = _read_tsv(work / "threshold_calibration.tsv")
    timing_rows = _read_tsv(work / "timings.tsv")
    equivalence_rows = _read_tsv(work / "representation_equivalence.tsv")
    counts = (
        (fit_rows, EXPECTED_FIT_ROWS, "fit-grid"),
        (selected_rows, EXPECTED_SELECTED_ROWS, "selected-fit"),
        (candidate_rows, EXPECTED_CANDIDATE_ROWS, "candidate"),
        (calibration_rows, EXPECTED_CALIBRATION_ROWS, "calibration"),
        (timing_rows, EXPECTED_TIMING_ROWS, "timing"),
        (equivalence_rows, EXPECTED_EQUIVALENCE_ROWS, "equivalence"),
    )
    for rows, expected, label in counts:
        if len(rows) != expected:
            raise ProtectedFinalizeUnavailable(
                f"{label} row count changed: {len(rows)} != {expected}"
            )
    _exact_values(calibration_rows, "context_role", (EXPECTED_CONTEXT_ROLE,))
    _exact_values(calibration_rows, "representation", MULTILAG_ARMS)
    _exact_values(calibration_rows, "quiet_swap", QUIET_SWAPS)
    _exact_values(calibration_rows, "nms_distance_px", NMS_DISTANCES_PX)
    _exact_values(
        calibration_rows,
        "target_nms_peaks_per_pseudo_burst",
        EXPECTED_QUIET_BURDENS,
    )
    if any(
        row["interpretation_before_label_join"] != "unknown_candidate"
        for row in candidate_rows
    ) or any(
        field in candidate_rows[0]
        for field in ("observation_id", "canonical_roi_id", "matched")
    ):
        raise ProtectedFinalizeUnavailable("candidate table contains label-derived fields")
    for row in calibration_rows:
        if (
            _strict_bool(
                row["probability_of_false_alarm_claimed"],
                field="probability_of_false_alarm_claimed",
            )
            or _strict_bool(
                row["positive_coordinates_used"], field="positive_coordinates_used"
            )
            or _strict_bool(
                row["positive_identities_used"], field="positive_identities_used"
            )
        ):
            raise ProtectedFinalizeUnavailable("calibration table violates claim boundary")
    raw_sha = _expected_raw_source_sha256(config, contract["movie_sha256"])
    for row in timing_rows:
        if (
            not _strict_bool(
                row["source_domain_assertion_passed"],
                field="source_domain_assertion_passed",
            )
            or row["source_domain"] != "acquisition_raw"
            or row["raw_source_contract_sha256"] != raw_sha
            or _strict_bool(row["fit_time_included"], field="fit_time_included")
        ):
            raise ProtectedFinalizeUnavailable(
                "not every six-lag candidate cell asserts acquisition-raw input"
            )
    for row in equivalence_rows:
        if (
            _strict_bool(
                row["sparse_positive_coordinates_used"],
                field="sparse_positive_coordinates_used",
            )
            or _strict_bool(
                row["sparse_positive_identities_used"],
                field="sparse_positive_identities_used",
            )
        ):
            raise ProtectedFinalizeUnavailable("equivalence table used protected labels")

    _, fit_models = _audit_fit_models(
        config, contract, work, fit_rows, selected_rows
    )
    selection_paths = sorted((work / "fit_selection_cells").glob("*.json"))
    candidate_paths = sorted((work / "candidate_cells").glob("*.json"))
    if len(selection_paths) != EXPECTED_FIT_SELECTION_CELLS:
        raise ProtectedFinalizeUnavailable("expected exactly 36 fit-selection cells")
    if len(candidate_paths) != EXPECTED_CANDIDATE_CELLS:
        raise ProtectedFinalizeUnavailable("expected exactly 12 candidate cells")
    selection_cells = [json.loads(path.read_text(encoding="utf-8")) for path in selection_paths]
    if any(
        cell["contract"].get("input_domain") != "acquisition_raw"
        or cell["contract"].get("raw_source_contract_sha256") != raw_sha
        or cell["metric"].get("selection_uses_training_burst_windows") is not True
        or cell["metric"].get("positive_coordinates_used") is not False
        or cell["metric"].get("positive_identities_used") is not False
        for cell in selection_cells
    ):
        raise ProtectedFinalizeUnavailable("fit-selection checkpoint contract changed")
    candidate_cells = [json.loads(path.read_text(encoding="utf-8")) for path in candidate_paths]
    expected_cell_keys = {
        (fold, arm) for fold in (1, 2, 3, 4) for arm in MULTILAG_ARMS
    }
    observed_cell_keys = {
        (int(cell["training_fold"]), str(cell["representation"]))
        for cell in candidate_cells
    }
    if observed_cell_keys != expected_cell_keys:
        raise ProtectedFinalizeUnavailable("candidate-cell key set changed")
    json_candidates = [row for cell in candidate_cells for row in cell["candidates"]]
    json_calibrations = [row for cell in candidate_cells for row in cell["calibrations"]]
    json_timings = [cell["timing"] for cell in candidate_cells]
    for json_rows, tsv_rows, label in (
        (json_candidates, candidate_rows, "candidate"),
        (json_calibrations, calibration_rows, "calibration"),
        (json_timings, timing_rows, "timing"),
    ):
        fields = list(tsv_rows[0])
        if _counter_from_json_rows(json_rows, fields) != _counter_from_tsv_rows(
            tsv_rows, fields
        ):
            raise ProtectedFinalizeUnavailable(
                f"{label} table does not reproduce its completed candidate cells"
            )
    upstream = _upstream_hashes(work)
    return VerifiedMultilagPartial(
        work=work,
        destination=destination_path,
        contract=contract,
        preflight=preflight,
        candidate_seal=seal,
        original_state=original_state,
        fit_rows=fit_rows,
        selected_rows=selected_rows,
        candidate_rows=candidate_rows,
        calibration_rows=calibration_rows,
        timing_rows=timing_rows,
        equivalence_rows=equivalence_rows,
        fit_models=fit_models,
        upstream_sha256=upstream,
    )


def finalize_multilag_partial(
    config: GammaLSDifferenceConfig,
    *,
    preflight_dir: str | Path,
    context_selection_dir: str | Path,
    output_dir: str | Path,
    expected_candidate_seal_sha256: str,
    observed_exit_code: int,
) -> dict[str, Any]:
    """Join labels, summarize, validate, and atomically promote a sealed run."""

    verified = verify_multilag_partial(
        config,
        preflight_dir=preflight_dir,
        context_selection_dir=context_selection_dir,
        destination=output_dir,
        expected_candidate_seal_sha256=expected_candidate_seal_sha256,
        observed_exit_code=observed_exit_code,
    )
    work = verified.work
    try:
        movie_shape = tuple(
            int(value) for value in verified.preflight["source"]["movie"]["shape"][1:]
        )
        v1 = _read_sparse_positives(
            config.source_paths["protected_labels_v1"],
            selector="include_inclusive",
            expected_rows=EXPECTED_V1_ROWS,
            movie_shape_yx=movie_shape,
        )
        v7 = _read_sparse_positives(
            config.source_paths["latest_labels_v7"],
            selector="include_confirmed",
            expected_rows=EXPECTED_V7_ROWS,
            movie_shape_yx=movie_shape,
        )
        v1_ids = {str(row["canonical_roi_id"]) for row in v1}
        v7_ids = {str(row["canonical_roi_id"]) for row in v7}
        if len(v1_ids) != EXPECTED_V1_IDENTITIES or len(v7_ids) != EXPECTED_V7_IDENTITIES:
            raise ProtectedFinalizeUnavailable("protected identity denominator changed")
        v1_matches = observation_match_rows(
            verified.candidate_rows,
            v1,
            cohort="protected_v1",
            operating_rows=verified.calibration_rows,
        )
        v7_matches = observation_match_rows(
            verified.candidate_rows,
            v7,
            cohort="latest_v7_sensitivity",
            operating_rows=verified.calibration_rows,
        )
        v1_aggregate = aggregate_match_rows(v1_matches)
        v7_aggregate = aggregate_match_rows(v7_matches)
        contrasts = vectorized_clustered_bootstrap_contrasts(
            v1_matches,
            learned_arm="cs_parzen_delay_residual",
            controls=(
                "difference_multilag_energy_normalized",
                "pca_whitened_delay_total_energy",
            ),
            seed=BOOTSTRAP_SEED,
            replicates=BOOTSTRAP_REPLICATES,
        )
        v1_summary = _summary_by_arm(v1_matches)
        v7_summary = _summary_by_arm(v7_matches)
        base = (
            len(MULTILAG_ARMS)
            * len(QUIET_SWAPS)
            * len(NMS_DISTANCES_PX)
            * len(EXPECTED_QUIET_BURDENS)
            * len(CANDIDATE_BUDGETS_PER_BURST)
        )
        expected_counts = {
            "v1_match_rows": EXPECTED_V1_ROWS * base,
            "v7_match_rows": EXPECTED_V7_ROWS * base,
            "v1_aggregate_rows": len(MULTILAG_ARMS)
            * len(QUIET_SWAPS)
            * len(NMS_DISTANCES_PX)
            * len(EXPECTED_QUIET_BURDENS)
            * 4
            * len(CANDIDATE_BUDGETS_PER_BURST),
            "v7_aggregate_rows": len(MULTILAG_ARMS)
            * len(QUIET_SWAPS)
            * len(NMS_DISTANCES_PX)
            * len(EXPECTED_QUIET_BURDENS)
            * 4
            * len(CANDIDATE_BUDGETS_PER_BURST),
            "v1_summary_rows": len(MULTILAG_ARMS)
            * (len(QUIET_SWAPS) + 1)
            * len(NMS_DISTANCES_PX)
            * len(EXPECTED_QUIET_BURDENS)
            * len(CANDIDATE_BUDGETS_PER_BURST),
            "v7_summary_rows": len(MULTILAG_ARMS)
            * (len(QUIET_SWAPS) + 1)
            * len(NMS_DISTANCES_PX)
            * len(EXPECTED_QUIET_BURDENS)
            * len(CANDIDATE_BUDGETS_PER_BURST),
            "bootstrap_contrast_rows": (len(QUIET_SWAPS) + 1)
            * len(NMS_DISTANCES_PX)
            * len(EXPECTED_QUIET_BURDENS)
            * 2,
        }
        observed_counts = {
            "v1_match_rows": len(v1_matches),
            "v7_match_rows": len(v7_matches),
            "v1_aggregate_rows": len(v1_aggregate),
            "v7_aggregate_rows": len(v7_aggregate),
            "v1_summary_rows": len(v1_summary),
            "v7_summary_rows": len(v7_summary),
            "bootstrap_contrast_rows": len(contrasts),
        }
        if observed_counts != expected_counts:
            raise ProtectedFinalizeUnavailable(
                f"derived row grain changed: {observed_counts} != {expected_counts}"
            )
        _atomic_tsv(work / "protected_v1_observation_matches.tsv", v1_matches)
        _atomic_tsv(work / "protected_v1_recall.tsv", v1_aggregate)
        _atomic_tsv(
            work / "protected_v1_clustered_bootstrap_contrasts.tsv", contrasts
        )
        _atomic_tsv(work / "protected_v1_arm_summary.tsv", v1_summary)
        _atomic_tsv(work / "latest_v7_observation_matches.tsv", v7_matches)
        _atomic_tsv(work / "latest_v7_sensitivity.tsv", v7_aggregate)
        _atomic_tsv(work / "latest_v7_arm_summary.tsv", v7_summary)

        rotation_seconds = sum(
            float(payload["fit"]["diagnostics"]["fit_seconds"])
            for payload in verified.fit_models
        )
        one_per_whitening = {}
        for payload in verified.fit_models:
            one_per_whitening.setdefault(
                (int(payload["training_fold"]), int(payload["sample_seed"])),
                float(payload["fit"]["diagnostics"]["whitening_fit_seconds"]),
            )
        whitening_seconds = sum(one_per_whitening.values())
        peak_vram = max(
            int(row["peak_allocated_vram_bytes_end_to_end"])
            for row in verified.timing_rows
        )
        cap = int(float(config.payload["resources"]["max_peak_vram_gib"]) * 2**30)
        claim_boundary = {
            "protected_v1_population": {
                "selector": "include_inclusive",
                "occurrences": len(v1),
                "canonical_identity_count": len(v1_ids),
            },
            "latest_v7_population": {
                "selector": "include_confirmed",
                "occurrences": len(v7),
                "canonical_identity_count": len(v7_ids),
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
        finalized_at = datetime.now(timezone.utc).isoformat()
        finalization = {
            "mode": "salvaged_after_statusless_exit_139_post_seal",
            "original_state": verified.original_state,
            "executor_sha256": verified.contract["executor_sha256"],
            "finalizer_sha256": _sha256(Path(__file__).resolve()),
            "candidate_seal_sha256": expected_candidate_seal_sha256,
            "sealed_upstream_sha256": verified.upstream_sha256,
            "derived_row_counts": observed_counts,
            "vectorized_bootstrap": {
                "algorithm": "26_identity_integer_sufficient_statistics_v1",
                "rng_call_pattern": "one_default_rng_choice_call_per_replicate",
                "seed": BOOTSTRAP_SEED,
                "replicates": BOOTSTRAP_REPLICATES,
                "scalar_parity_test": (
                    "tests/test_gamma_ls_difference_protected_finalize.py"
                ),
            },
            "finalized_at_utc": finalized_at,
        }
        _atomic_json(work / "finalization_provenance.json", finalization)
        runtime_gpu = verified.preflight["resources"]["gpu"]
        summary = {
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "run_type": "protected_outer_fold_matched_six_lag_representation",
            "status": "complete_protected_metrics_scientific_audit_pending",
            "completed_at_utc": finalized_at,
            "runtime": {
                "source": "frozen_host_visible_preflight",
                "requested_device": runtime_gpu["requested_device"],
                "device_name": runtime_gpu["device_name"],
                "torch_version": runtime_gpu["torch_version"],
                "torch_cuda_build": runtime_gpu["torch_cuda_build"],
                "nvidia_smi": runtime_gpu["nvidia_smi"],
                "original_runtime_object_not_preserved_before_sigsegv": True,
            },
            "design": verified.contract["design"],
            "historical_grid_basis": verified.contract["historical_grid_basis"],
            "selected_context_role": verified.contract["selected_context_role"],
            "selected_contexts": verified.contract["selected_contexts"],
            "anatomy_sampling": {
                "anatomy_pixel_count": int(
                    verified.fit_models[0]["sample_manifest"]["anatomy_pixel_count"]
                ),
                "source": "consistent_fit_checkpoint_manifests",
                "full_original_summary_not_preserved": True,
            },
            "raw_load_timing": {
                "status": "not_preserved_before_sigsegv",
                "paper_runtime_claim_from_this_field_allowed": False,
            },
            "raw_source_contract_sha256": _expected_raw_source_sha256(
                config, verified.contract["movie_sha256"]
            ),
            "fit_timing_seconds": float(whitening_seconds + rotation_seconds),
            "whitening_fit_timing_seconds": float(whitening_seconds),
            "rotation_fit_timing_seconds": float(rotation_seconds),
            "fit_timing_aggregation": (
                "12_unique_whitenings_plus_24_bandwidth_specific_rotations"
            ),
            "fit_count": EXPECTED_FIT_MODELS,
            "selected_fit_row_count": len(verified.selected_rows),
            "candidate_row_count": len(verified.candidate_rows),
            "candidate_seal": verified.candidate_seal,
            "peak_allocated_vram_bytes": peak_vram,
            "protected_v1": {
                "occurrences": len(v1),
                "canonical_identities": len(v1_ids),
                "arm_summary": v1_summary,
                "clustered_bootstrap_contrasts": contrasts,
            },
            "latest_v7_sensitivity": {
                "occurrences": len(v7),
                "canonical_identities": len(v7_ids),
                "arm_summary": v7_summary,
                "inferential_claim": False,
            },
            "finalization": finalization,
            "claim_boundary": claim_boundary,
        }
        _atomic_json(work / "summary.json", summary)
        checks = {
            "original_exit_139_recorded": verified.original_state[
                "operator_observed_exit_code"
            ]
            == EXPECTED_EXIT_CODE,
            "sealed_upstream_hashes_unchanged": _upstream_hashes(work)
            == verified.upstream_sha256,
            "exact_24_cs_parzen_rotation_fits": len(verified.fit_models) == 24,
            "exact_12_unique_full_rank_whitening_models": len(one_per_whitening)
            == 12,
            "exact_36_fit_selection_cells": len(
                list((work / "fit_selection_cells").glob("*.json"))
            )
            == 36,
            "exact_12_candidate_cells": len(
                list((work / "candidate_cells").glob("*.json"))
            )
            == 12,
            "exact_1559_sealed_candidate_rows": len(verified.candidate_rows)
            == 1559,
            "exact_360_threshold_rows": len(verified.calibration_rows) == 360,
            "exact_12_timing_rows": len(verified.timing_rows) == 12,
            "exact_14_equivalence_rows": len(verified.equivalence_rows) == 14,
            "all_six_lag_arms_assert_acquisition_raw_source": all(
                row["source_domain"] == "acquisition_raw"
                and _strict_bool(
                    row["source_domain_assertion_passed"],
                    field="source_domain_assertion_passed",
                )
                for row in verified.timing_rows
            ),
            "candidate_seal_precedes_label_join": True,
            "protected_v1_uses_79_inclusive": len(v1) == EXPECTED_V1_ROWS,
            "protected_v1_has_26_identity_clusters": len(v1_ids)
            == EXPECTED_V1_IDENTITIES,
            "v7_uses_106_confirmed": len(v7) == EXPECTED_V7_ROWS,
            "v7_has_44_canonical_label_identities": len(v7_ids)
            == EXPECTED_V7_IDENTITIES,
            "derived_row_grain_exact": observed_counts == expected_counts,
            "vectorized_bootstrap_exact_90_rows": len(contrasts) == 90,
            "peak_vram_within_manifest_cap": peak_vram <= cap,
            "all_candidates_unknown_before_join": all(
                row["interpretation_before_label_join"] == "unknown_candidate"
                for row in verified.candidate_rows
            ),
            "precision_not_claimed": True,
            "scientific_audit_pending": True,
        }
        validation = {
            "status": (
                "passed_protected_metric_artifact_contract"
                if all(checks.values())
                else "failed_protected_metric_artifact_contract"
            ),
            "checks": checks,
            "all_checks_pass": all(checks.values()),
            "finalizer_sha256": finalization["finalizer_sha256"],
        }
        _atomic_json(work / "validation.json", validation)
        if not validation["all_checks_pass"]:
            raise ProtectedFinalizeUnavailable("multilag finalization failed closed")
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
                "paired_inference": "protected_v1_clustered_bootstrap_contrasts.tsv",
                "per_observation_recomputability": "protected_v1_observation_matches.tsv",
                "latest_sensitivity": "latest_v7_arm_summary.tsv",
                "fit_grid": "fit_grid.tsv",
                "fit_selection": "selected_fit_rows.tsv",
                "timing": "timings.tsv",
                "equivalence": "representation_equivalence.tsv",
                "finalization_provenance": "finalization_provenance.json",
                "limitations": [
                    "sparse positives do not identify precision",
                    "v7 is candidate-assisted descriptive sensitivity",
                    "one recording does not establish population generalization",
                    "scientific-audit media are still pending",
                    "archived whole-review v5 fits are diagnostic only",
                    "original raw-load timing was not preserved before SIGSEGV",
                ],
            },
        )
        _atomic_text(
            work / "REPORT.md",
            "\n".join(
                (
                    "# Protected matched-six-lag Gamma-LS ablation",
                    "",
                    "The sealed metric stage was finalized without refitting or rescoring. ",
                    f"It contains {len(v1)} protected occurrences across {len(v1_ids)} ",
                    f"canonical identities, {len(verified.fit_models)} fold-local CS-Parzen ",
                    f"fits, and {len(verified.candidate_rows)} label-free proposal rows.",
                    "",
                    "All three six-lag arms assert the acquisition-raw source domain. ",
                    "Unmatched proposals are unknown, precision is not identified, v7 is ",
                    "descriptive only, and scientific-audit media remain pending.",
                )
            ),
        )
        _atomic_json(
            work / "status.json",
            {
                "status": "complete_protected_metrics_scientific_audit_pending",
                "completed_at_utc": finalized_at,
                "validation_passed": True,
                "candidate_seal_sha256": expected_candidate_seal_sha256,
                "salvaged_from_exit_code": EXPECTED_EXIT_CODE,
                "scientific_audit_complete": False,
                "paper_promotion_ready": False,
            },
        )
        _atomic_json(
            work / "heartbeat.json",
            {
                "updated_at_utc": finalized_at,
                "stage": "complete",
                "status": "complete_protected_metrics_scientific_audit_pending",
                "finalized_by": "multilag_finalize.py",
            },
        )
        if _upstream_hashes(work) != verified.upstream_sha256:
            raise ProtectedFinalizeUnavailable("sealed upstream bytes changed during finalization")
        _atomic_json(work / "artifact_index.json", _artifact_index(work))
        index = json.loads((work / "artifact_index.json").read_text(encoding="utf-8"))
        for artifact in index["artifacts"]:
            path = work / artifact["path"]
            if (
                path.stat().st_size != int(artifact["size_bytes"])
                or _sha256(path) != artifact["sha256"]
            ):
                raise ProtectedFinalizeUnavailable(
                    f"artifact index verification failed: {artifact['path']}"
                )
        if verified.destination.exists():
            raise FileExistsError(
                f"matched-six-lag destination appeared: {verified.destination}"
            )
        work.replace(verified.destination)
        return summary
    except Exception as error:
        if work.exists():
            _atomic_json(
                work / "status.json",
                {
                    "status": "finalizer_failed_resumable",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": repr(error),
                    "original_status_file_present": False,
                    "original_operator_observed_exit_code": EXPECTED_EXIT_CODE,
                    "sealed_tables_must_not_be_modified": True,
                },
            )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--context-selection-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-candidate-seal-sha256", required=True)
    parser.add_argument("--observed-exit-code", required=True, type=int)
    arguments = parser.parse_args(argv)
    config = GammaLSDifferenceConfig.load(arguments.config)
    result = finalize_multilag_partial(
        config,
        preflight_dir=arguments.preflight_dir,
        context_selection_dir=arguments.context_selection_dir,
        output_dir=arguments.output_dir,
        expected_candidate_seal_sha256=arguments.expected_candidate_seal_sha256,
        observed_exit_code=arguments.observed_exit_code,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_EXIT_CODE",
    "VerifiedMultilagPartial",
    "_recorded_termination",
    "finalize_multilag_partial",
    "verify_multilag_partial",
]
