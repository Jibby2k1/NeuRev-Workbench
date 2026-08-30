"""Frozen Run-B source-on versus intervention rank-displacement diagnostic.

This is a derived, non-claim-bearing analysis of the completed EXP-0029 Run-B
screen.  It rehydrates the exact frozen raw / JEPA-residual / random-residual
score lanes and asks a deliberately narrower question: when an injected source
is recovered by the paired intervention map but missed by the deployment-like
source-on map, is its local response absent (attenuation) or displaced by a
stronger native/background maximum (competition)?  Native candidates are never
assigned negative labels.
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
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import torch

from .contracts import atomic_json, atomic_text, stable_hash
from .discovery import sha256_file
from .jepa_comparators import frozen_handcrafted_comparator
from .jepa_evaluation import (
    footprint_centers_yx,
    hierarchical_grouped_bootstrap,
    local_maxima_yx,
    match_injected_sources,
)
from . import jepa_background_residual as decoder_module
from . import jepa_residual_pilot_v1_1 as residual


DIAGNOSTIC_ID = "NREV-DIAG-EXP-0029-RANK-DISPLACEMENT-20260830-F"
DEFAULT_OUTPUT_ROOT = (
    "Outputs/NeuronIdentifiability/NREV-EXP-0029/diagnostics/"
    + DIAGNOSTIC_ID
)
RUN_B_ROOT = (
    "Outputs/NeuronIdentifiability/NREV-EXP-0029/runs/"
    "NREV-RUN-EXP-0029-SCREEN-20260830-B"
)
RUN_B_ARTIFACT_INDEX_SHA256 = "34bd983ffdf55cb0941bbd1443adf939f4cb86ac4ffe51a736b2d6d9383cfc1c"
CONFIG_PATH = "examples/conditional_background_residual_v1_1.example.json"
METHODS = (
    residual.RAW_METHOD,
    residual.JEPA_RESIDUAL_METHOD,
    residual.RANDOM_RESIDUAL_METHOD,
)
METHOD_SHORT = {
    residual.RAW_METHOD: "raw",
    residual.JEPA_RESIDUAL_METHOD: "jepa_residual",
    residual.RANDOM_RESIDUAL_METHOD: "random_residual",
}
NUMERICAL_DEPENDENCIES = (
    "neurobench/experiments/neuron_identifiability/jepa_rank_displacement.py",
    "neurobench/experiments/neuron_identifiability/jepa_comparators.py",
    "neurobench/experiments/neuron_identifiability/jepa_evaluation.py",
    "neurobench/experiments/neuron_identifiability/jepa_background_residual.py",
    "neurobench/experiments/neuron_identifiability/jepa_residual_pilot_v1_1.py",
    "neurobench/experiments/neuron_identifiability/contracts.py",
    "neurobench/experiments/neuron_identifiability/discovery.py",
)


class RankDisplacementError(ValueError):
    """Raised when the frozen Run-B diagnostic cannot preserve its contract."""


@dataclass(frozen=True)
class ScorePair:
    source_on_z: np.ndarray
    intervention_z: np.ndarray
    source_on_candidates: tuple[tuple[int, int, float], ...]
    intervention_candidates: tuple[tuple[int, int, float], ...]


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RankDisplacementError(f"could not read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RankDisplacementError(f"JSON root must be an object: {path}")
    return value


def _output_paths(output: Path) -> tuple[Path, Path]:
    resolved = Path(output).expanduser().resolve()
    return resolved, resolved.with_name(resolved.name + ".partial")


def _guard_output(output: Path) -> tuple[Path, Path]:
    final, partial = _output_paths(output)
    if final.exists() or partial.exists():
        raise FileExistsError(f"refusing existing output or partial root: {final}")
    return final, partial


def _atomic_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_text(path, "")
        return
    fields = list(rows[0])
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _artifact_index(root: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "artifact_index.json"):
        artifacts.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return {"schema_version": 1, "artifact_count": len(artifacts), "artifacts": artifacts}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _phase_timing(
    *,
    planned_at: str,
    preflight_started_at: str,
    started_at: str,
    ended_at: str,
) -> dict[str, Any]:
    """Return an explicit UTC phase contract with timestamp-derived durations."""

    timestamps = {
        name: datetime.fromisoformat(value)
        for name, value in (
            ("planned_at", planned_at),
            ("preflight_started_at", preflight_started_at),
            ("started_at", started_at),
            ("ended_at", ended_at),
        )
    }
    if any(value.tzinfo is None for value in timestamps.values()):
        raise RankDisplacementError("execution timing timestamps must be timezone-aware")
    ordered = [
        timestamps["planned_at"],
        timestamps["preflight_started_at"],
        timestamps["started_at"],
        timestamps["ended_at"],
    ]
    if ordered != sorted(ordered):
        raise RankDisplacementError("execution timing phases are not monotonic")

    def seconds(start: str, end: str) -> float:
        return float((timestamps[end] - timestamps[start]).total_seconds())

    return {
        "planned_at": planned_at,
        "preflight_started_at": preflight_started_at,
        "started_at": started_at,
        "ended_at": ended_at,
        "preflight_duration_seconds": seconds("preflight_started_at", "started_at"),
        "duration_seconds": seconds("started_at", "ended_at"),
        "planned_to_ended_duration_seconds": seconds("planned_at", "ended_at"),
        "primary_duration_field": "duration_seconds",
        "primary_duration_contract": (
            "duration_seconds equals ended_at minus started_at using timezone-aware "
            "UTC timestamp arithmetic"
        ),
        "phase_contract": {
            "planned": {
                "timestamp_field": "planned_at",
                "definition": "diagnostic invocation accepted before preflight",
            },
            "preflight": {
                "started_at_field": "preflight_started_at",
                "ended_at_field": "started_at",
                "duration_field": "preflight_duration_seconds",
            },
            "execution": {
                "started_at_field": "started_at",
                "ended_at_field": "ended_at",
                "duration_field": "duration_seconds",
            },
            "total": {
                "started_at_field": "planned_at",
                "ended_at_field": "ended_at",
                "duration_field": "planned_to_ended_duration_seconds",
            },
        },
    }


def _git_provenance(repository: Path) -> dict[str, Any]:
    """Capture a content-aware dirty digest without persisting absolute paths."""

    def git(*arguments: str) -> bytes:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repository,
            check=True,
            capture_output=True,
        )
        return completed.stdout

    try:
        commit = git("rev-parse", "HEAD").strip().decode("ascii")
        branch = git("branch", "--show-current").strip().decode("utf-8") or "detached"
        status = git("status", "--porcelain=v1", "-z")
        tracked_diff = git("diff", "--binary", "--no-ext-diff", "HEAD", "--")
        untracked_raw = git("ls-files", "--others", "--exclude-standard", "-z")
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        raise RankDisplacementError("Git execution provenance is unavailable") from exc
    untracked = sorted(
        value.decode("utf-8") for value in untracked_raw.split(b"\0") if value
    )
    untracked_rows = []
    untracked_digest = hashlib.sha256()
    for logical in untracked:
        candidate = (repository / logical).resolve()
        try:
            candidate.relative_to(repository)
        except ValueError as exc:
            raise RankDisplacementError("untracked Git path escapes repository") from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise RankDisplacementError(
                f"untracked provenance input is not a regular file: {logical}"
            )
        digest = sha256_file(candidate)
        size = int(candidate.stat().st_size)
        untracked_rows.append({"path": logical, "bytes": size, "sha256": digest})
        untracked_digest.update(logical.encode("utf-8"))
        untracked_digest.update(b"\0")
        untracked_digest.update(str(size).encode("ascii"))
        untracked_digest.update(b"\0")
        untracked_digest.update(digest.encode("ascii"))
        untracked_digest.update(b"\0")
    combined = hashlib.sha256()
    combined.update(b"tracked_diff_sha256\0")
    combined.update(hashlib.sha256(tracked_diff).hexdigest().encode("ascii"))
    combined.update(b"\0untracked_manifest_sha256\0")
    combined.update(untracked_digest.hexdigest().encode("ascii"))
    combined.update(b"\0status_sha256\0")
    combined.update(hashlib.sha256(status).hexdigest().encode("ascii"))
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "dirty_entry_count": int(status.count(b"\0")),
        "dirty_status_sha256": hashlib.sha256(status).hexdigest(),
        "tracked_binary_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "untracked_content_manifest_sha256": untracked_digest.hexdigest(),
        "content_aware_dirty_state_sha256": combined.hexdigest(),
        "content_aware_dirty_state_scope": (
            "sha256_of_status_sha256_plus_tracked_git_diff_binary_sha256_plus_"
            "repo_relative_untracked_path_size_content_sha256_manifest"
        ),
        "untracked_files": untracked_rows,
        "absolute_paths_persisted": False,
    }


def _dependency_hashes(repository: Path) -> list[dict[str, Any]]:
    rows = []
    for logical in NUMERICAL_DEPENDENCIES:
        path = (repository / logical).resolve()
        try:
            path.relative_to(repository)
        except ValueError as exc:
            raise RankDisplacementError("numerical dependency escapes repository") from exc
        if not path.is_file() or path.is_symlink():
            raise RankDisplacementError(f"numerical dependency is missing: {logical}")
        rows.append(
            {
                "path": logical,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _runtime_provenance(
    *,
    device: str,
    old_threads: int,
    configured_threads: int,
    prediction_amp: bool,
) -> dict[str, Any]:
    cuda_device = None
    if device.startswith("cuda"):
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        cuda_device = {
            "index": int(index),
            "name": str(properties.name),
            "capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": int(properties.total_memory),
            "bfloat16_supported": bool(torch.cuda.is_bf16_supported()),
        }
    return {
        "platform": platform.platform(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_basename": Path(sys.executable).name,
        },
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "torch": {
            "version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "device": {"requested": device, "resolved": device, "cuda": cuda_device},
        "dtypes": {
            "raw_fixture": "float32",
            "normalized_movie": "float32",
            "provider_and_decoder_parameters": "float32",
            "cuda_prediction_autocast": "bfloat16" if prediction_amp else "disabled",
            "score_maps_and_ranks": "float64",
        },
        "threads": {
            "torch_before": int(old_threads),
            "torch_configured_during_run": int(configured_threads),
            "torch_observed_during_run": int(torch.get_num_threads()),
            "torch_interop": int(torch.get_num_interop_threads()),
            "cpu_count": os.cpu_count(),
        },
    }


def _verify_output_tree(root: Path) -> dict[str, Any]:
    index = _json(root / "artifact_index.json")
    raw_rows = index.get("artifacts")
    if not isinstance(raw_rows, list):
        raise RankDisplacementError("artifact index rows are unavailable")
    indexed = {str(row["path"]): row for row in raw_rows}
    actual = {
        str(path.relative_to(root)): path
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_index.json"
    }
    if set(indexed) != set(actual):
        raise RankDisplacementError("artifact index does not cover the complete output tree")
    for logical, path in actual.items():
        row = indexed[logical]
        if (
            int(row["bytes"]) != int(path.stat().st_size)
            or str(row["sha256"]) != sha256_file(path)
        ):
            raise RankDisplacementError(f"artifact hash/size mismatch: {logical}")
    return {
        "artifact_count": len(actual),
        "all_existing_non_index_files_indexed": True,
        "all_indexed_hashes_and_sizes_exact": True,
        "artifact_index_sha256": sha256_file(root / "artifact_index.json"),
    }


def _assert_no_workstation_paths(root: Path, forbidden: Sequence[Path]) -> None:
    needles = [str(path.expanduser().resolve()).encode("utf-8") for path in forbidden]
    needles.append(str(Path.home().resolve()).encode("utf-8"))
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        if any(needle and needle in payload for needle in needles):
            raise RankDisplacementError(
                f"workstation-specific absolute path leaked into output: {path.name}"
            )


def _verify_run_b(root: Path) -> dict[str, Any]:
    integrity = residual.verify_parent_run_integrity(
        root, expected_artifact_index_sha256=RUN_B_ARTIFACT_INDEX_SHA256
    )
    required = (
        "paired_injection_results.json",
        "decoder_checkpoints_manifest.json",
        "checkpoints/decoder_seed_6201_final_non_scientific.pt",
        "prediction_coverage.json",
        "summary.json",
    )
    indexed = {str(row["path"]): row for row in _json(root / "artifact_index.json")["artifacts"]}
    missing = [name for name in required if name not in indexed]
    if missing:
        raise RankDisplacementError("Run-B artifact index lacks required inputs: " + ", ".join(missing))
    return {
        "run_b_artifact_index_sha256": sha256_file(root / "artifact_index.json"),
        "run_b_indexed_artifacts_verified": int(integrity["artifact_count"]),
        "run_b_required_input_artifacts": [
            {
                "path": name,
                "bytes": int(indexed[name]["bytes"]),
                "sha256": str(indexed[name]["sha256"]),
            }
            for name in required
        ],
        "run_b_integrity_snapshot_sha256": str(integrity["snapshot_sha256"]),
    }


def _score_pair(
    source_off: np.ndarray,
    source_on: np.ndarray,
    *,
    candidate_budget: int,
    minimum_distance_px: int,
    border_px: int,
) -> ScorePair:
    scorer = frozen_handcrafted_comparator.fit_source_off(np.asarray(source_off, dtype=np.float32))
    background = np.asarray(scorer(source_off), dtype=np.float64)
    on = np.asarray(scorer(source_on), dtype=np.float64)
    center = float(np.median(background))
    scale = float(1.4826 * np.median(np.abs(background - center)))
    if not math.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = float(np.std(background))
    scale = max(scale, 1e-8)
    source_on_z = (on - center) / scale
    intervention_z = source_on_z - (background - center) / scale
    if not (np.isfinite(source_on_z).all() and np.isfinite(intervention_z).all()):
        raise RankDisplacementError("reconstructed score maps are nonfinite")
    return ScorePair(
        source_on_z=source_on_z,
        intervention_z=intervention_z,
        source_on_candidates=local_maxima_yx(
            source_on_z,
            budget=int(candidate_budget),
            minimum_distance_px=int(minimum_distance_px),
            border_px=int(border_px),
        ),
        intervention_candidates=local_maxima_yx(
            intervention_z,
            budget=int(candidate_budget),
            minimum_distance_px=int(minimum_distance_px),
            border_px=int(border_px),
        ),
    )


def _rank_with_ties(values: np.ndarray, row: int, column: int) -> int:
    """One-based rank, breaking equal-score ties by row then column."""
    score = float(values[row, column])
    rows, columns = np.indices(values.shape)
    earlier_tie = (values == score) & ((rows < row) | ((rows == row) & (columns < column)))
    return int(np.count_nonzero(values > score) + np.count_nonzero(earlier_tie) + 1)


def _local_rank(values: np.ndarray, row: int, column: int, radius: int) -> tuple[int, int]:
    y0, y1 = max(0, row - radius), min(values.shape[0], row + radius + 1)
    x0, x1 = max(0, column - radius), min(values.shape[1], column + radius + 1)
    return _rank_with_ties(values[y0:y1, x0:x1], row - y0, column - x0), int((y1 - y0) * (x1 - x0))


def _top_competitor(values: np.ndarray, row: int, column: int, radius: float) -> tuple[float, float]:
    rows, columns = np.indices(values.shape)
    outside = np.hypot(rows - row, columns - column) > float(radius)
    if not np.any(outside):
        return float("nan"), float("nan")
    maximum = float(np.max(values[outside]))
    winner = np.argwhere(outside & (values == maximum))
    winner = winner[np.lexsort((winner[:, 1], winner[:, 0]))][0]
    return maximum, float(math.hypot(float(winner[0] - row), float(winner[1] - column)))


def classify_source(*, source_on_recovered: bool, intervention_recovered: bool) -> str:
    """Classify a source without assigning a biological label to competitors."""
    if source_on_recovered and intervention_recovered:
        return "recovered_source_on"
    if not source_on_recovered and intervention_recovered:
        return "native_background_competition"
    if not source_on_recovered and not intervention_recovered:
        return "attenuation_or_response_failure"
    return "source_on_only_inconsistent"


def _source_rows(
    *,
    fixture: Any,
    metadata: Mapping[str, Any],
    method: str,
    score_pair: ScorePair,
    match_radius_px: float,
    local_radius_px: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    centers = footprint_centers_yx(fixture.footprints)
    source_recovery = match_injected_sources(score_pair.source_on_candidates, centers, radius_px=match_radius_px)
    intervention_recovery = match_injected_sources(score_pair.intervention_candidates, centers, radius_px=match_radius_px)
    source_matched = set(int(value) for value in source_recovery.matched_source_indices)
    intervention_matched = set(int(value) for value in intervention_recovery.matched_source_indices)
    morphology = tuple(str(value) for value in fixture.metadata["morphology"])
    rows: list[dict[str, Any]] = []
    top4_cutoff = float(score_pair.source_on_candidates[-1][2]) if score_pair.source_on_candidates else float("nan")
    for source_index, (center_y, center_x) in enumerate(centers):
        row, column = int(round(center_y)), int(round(center_x))
        local_rank, local_count = _local_rank(score_pair.source_on_z, row, column, local_radius_px)
        intervention_local_rank, _ = _local_rank(score_pair.intervention_z, row, column, local_radius_px)
        competitor_score, competitor_distance = _top_competitor(
            score_pair.source_on_z, row, column, match_radius_px
        )
        top4_distances = [math.hypot(candidate[0] - center_y, candidate[1] - center_x) for candidate in score_pair.source_on_candidates]
        top4_min_distance = min(top4_distances) if top4_distances else float("inf")
        source_on_recovered = source_index in source_matched
        intervention_recovered = source_index in intervention_matched
        rows.append(
            {
                "fixture_id": str(fixture.fixture_id),
                "method": method,
                "method_short": METHOD_SHORT[method],
                "background_recording_id": str(metadata["background_recording_id"]),
                "background_window_id": str(metadata["background_window_id"]),
                "mad_stratum": str(metadata["background_window_id"]).split("_")[-2] + "_mad",
                "injection_seed": int(metadata["injection_seed"]),
                "source_count": int(metadata["source_count"]),
                "crowding_case": str(metadata["crowding_case"]),
                "source_index": int(source_index),
                "morphology": morphology[source_index],
                "truth_center_y": row,
                "truth_center_x": column,
                "source_on_truth_center_score_z": float(score_pair.source_on_z[row, column]),
                "intervention_truth_center_score_z": float(score_pair.intervention_z[row, column]),
                "source_on_native_full_pixel_rank": _rank_with_ties(score_pair.source_on_z, row, column),
                "source_on_native_full_pixel_count": int(score_pair.source_on_z.size),
                "source_on_local_pixel_rank": int(local_rank),
                "source_on_local_pixel_count": int(local_count),
                "intervention_local_pixel_rank": int(intervention_local_rank),
                "source_on_top4_cutoff_score_z": top4_cutoff,
                "source_on_top4_cutoff_margin_z": float(score_pair.source_on_z[row, column] - top4_cutoff),
                "source_on_strongest_outside_match_radius_score_z": competitor_score,
                "source_on_strongest_outside_match_radius_margin_z": float(score_pair.source_on_z[row, column] - competitor_score),
                "source_on_strongest_outside_match_radius_distance_px": competitor_distance,
                "source_on_top4_min_distance_to_truth_px": float(top4_min_distance),
                "source_on_top4_displaced": bool(top4_min_distance > match_radius_px),
                "source_on_recovered": bool(source_on_recovered),
                "intervention_recovered": bool(intervention_recovered),
                "classification": classify_source(
                    source_on_recovered=source_on_recovered,
                    intervention_recovered=intervention_recovered,
                ),
                "native_candidates_semantics": "unknown_not_negative",
            }
        )
    reconstructed = {
        "source_on_recovery": source_recovery.to_dict(),
        "intervention_recovery": intervention_recovery.to_dict(),
    }
    return rows, reconstructed


def _cluster_rows(rows: Sequence[Mapping[str, Any]], metric: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[float]] = {}
    for row in rows:
        key = (
            str(row["background_recording_id"]),
            str(row["background_window_id"]),
            int(row["injection_seed"]),
            str(row["method"]),
        )
        if metric == "competition_among_source_on_misses":
            if bool(row["source_on_recovered"]):
                continue
            value = float(row["classification"] == "native_background_competition")
        elif metric == "attenuation_among_source_on_misses":
            if bool(row["source_on_recovered"]):
                continue
            value = float(row["classification"] == "attenuation_or_response_failure")
        else:
            value = float(row[metric])
        grouped.setdefault(key, []).append(value)
    return [
        {
            "background_recording_id": recording,
            "background_window_id": window,
            "injection_seed": seed,
            "method": method,
            "value": float(np.mean(values)),
        }
        for (recording, window, seed, method), values in sorted(grouped.items())
    ]


def summarize_rank_displacement(rows: Sequence[Mapping[str, Any]], *, bootstrap_seed: int, draws: int) -> dict[str, Any]:
    metrics = (
        "source_on_recovered",
        "intervention_recovered",
        "source_on_top4_displaced",
        "competition_among_source_on_misses",
        "attenuation_among_source_on_misses",
    )
    output: dict[str, Any] = {}
    for metric_index, metric in enumerate(metrics):
        output[metric] = {}
        for method in METHODS:
            cluster_rows = [row for row in _cluster_rows(rows, metric) if row["method"] == method]
            if not cluster_rows:
                output[metric][method] = {"status": "not_estimable_no_applicable_sources"}
                continue
            output[metric][method] = {
                "status": "descriptive_grouped_uncertainty_only",
                **hierarchical_grouped_bootstrap(
                    cluster_rows,
                    value_key="value",
                    draws=int(draws),
                    seed=int(bootstrap_seed + metric_index * 100 + METHODS.index(method)),
                ),
            }
    return output


def _stratified_summary(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHODS:
        for value in sorted({str(row[key]) for row in rows}):
            selected = [row for row in rows if row["method"] == method and str(row[key]) == value]
            misses = [row for row in selected if not bool(row["source_on_recovered"])]
            output.append(
                {
                    "method": method,
                    "method_short": METHOD_SHORT[method],
                    key: value,
                    "source_row_count": len(selected),
                    "source_on_recovery_fraction": float(np.mean([bool(row["source_on_recovered"]) for row in selected])),
                    "intervention_recovery_fraction": float(np.mean([bool(row["intervention_recovered"]) for row in selected])),
                    "competition_fraction_of_source_on_misses": (
                        float(np.mean([row["classification"] == "native_background_competition" for row in misses])) if misses else None
                    ),
                    "median_top4_margin_z": float(np.median([float(row["source_on_top4_cutoff_margin_z"]) for row in selected])),
                    "median_local_rank": float(np.median([float(row["source_on_local_pixel_rank"]) for row in selected])),
                }
            )
    return output


def _recovery_totals(counts: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    """Make the asymmetric source-on-only category explicit in headline totals."""
    output: dict[str, dict[str, int]] = {}
    for method in METHODS:
        values = counts[method]
        recovered_both = int(values["recovered_source_on"])
        source_on_only = int(values["source_on_only_inconsistent"])
        output[method] = {
            "recovered_on_both_source_on_and_intervention": recovered_both,
            "recovered_source_on_only_intervention_inconsistent": source_on_only,
            "total_source_on_recovered": recovered_both + source_on_only,
            "total_intervention_recovered": recovered_both
            + int(values["native_background_competition"]),
        }
    return output


def _report(summary: Mapping[str, Any]) -> str:
    counts = summary["classification_counts"]
    totals = summary["recovery_totals"]
    lines = []
    for method in METHODS:
        values = counts[method]
        recovered = totals[method]
        lines.append(
            f"- `{METHOD_SHORT[method]}`: total source-on recovered `{recovered['total_source_on_recovered']}` "
            f"(`{recovered['recovered_on_both_source_on_and_intervention']}` also recovered on the intervention map; "
            f"`{recovered['recovered_source_on_only_intervention_inconsistent']}` source-on-only inconsistent), "
            f"competition `{values['native_background_competition']}`, attenuation/response failure `{values['attenuation_or_response_failure']}`."
        )
    return (
        "# Source-on versus intervention rank displacement\n\n"
        "This deterministic derived diagnostic rehydrated the frozen EXP-0029 Run-B raw, JEPA-residual, and random-residual score lanes over all 108 exact fixtures / 252 injected sources. It distinguishes paired intervention response from source-on top-four displacement; native competitors remain unknown rather than negatives.\n\n"
        "## Classification counts\n\n" + "\n".join(lines) + "\n\n"
        "A source-on miss with intervention recovery is classified as `native_background_competition`; a miss on both maps is `attenuation_or_response_failure`. These are score-map mechanism categories, not biological labels or precision estimates.\n\n"
        "## Boundary\n\n"
        "The source Run-B scientific audit and motion dependency remain incomplete. This package provides bounded exact-injection diagnostics only; it establishes no neuron identity, false-positive rate, denoising benefit, nuisance robustness, generalization, causal claim, or reinforcement-learning result.\n"
    )


def run_rank_displacement_diagnostic(
    *,
    repository_root: Path,
    data_root: Path,
    output_root: Path | None = None,
    device: str | None = None,
    grouped_bootstrap_draws: int = 1_000,
) -> dict[str, Any]:
    """Run the non-mutating-input, non-claim-bearing 108-fixture diagnostic."""
    planned_at = _utc_now()
    preflight_started_at = _utc_now()
    repository = Path(repository_root).expanduser().resolve()
    runtime_data = Path(data_root).expanduser().resolve()
    output = (repository / DEFAULT_OUTPUT_ROOT).resolve() if output_root is None else Path(output_root).expanduser().resolve()
    final, partial = _guard_output(output)
    if grouped_bootstrap_draws < 1_000:
        raise RankDisplacementError("grouped_bootstrap_draws must be at least 1000")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RankDisplacementError("CUDA was requested but is unavailable")
    config_file = (repository / CONFIG_PATH).resolve()
    config = residual.load_jepa_residual_config(config_file)
    run_b_root = (repository / RUN_B_ROOT).resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir(parents=False, exist_ok=False)
    old_threads = torch.get_num_threads()
    configured_threads = int(config["_resources"]["cpu_threads"])
    torch.set_num_threads(configured_threads)
    started_at = _utc_now()
    try:
        git_provenance = _git_provenance(repository)
        dependency_hashes = _dependency_hashes(repository)
        input_integrity = _verify_run_b(run_b_root)
        parent_root, paths = residual._pinned_parent_paths(config, repository)
        exp0028_integrity = residual.verify_parent_run_integrity(
            parent_root,
            expected_artifact_index_sha256=config["upstream_screen"][
                "artifact_index_sha256"
            ],
        )
        parent_integrity = residual._verify_pinned_parent_files(config, paths)
        backgrounds, _ = residual.reconstruct_frozen_backgrounds(
            config, repository_root=repository, runtime_data_root=runtime_data, paths=paths, verify_source_hashes=True
        )
        parent_config = residual._parent_registered_config(paths)
        fixtures, fixture_metadata, _, fixture_identity = residual.reconstruct_exact_fixtures(
            parent_config, backgrounds, residual._json_object(paths["paired_injection_manifest"])
        )
        if len(fixtures) != residual.STRICT_SCREEN_FIXTURES or sum(int(item.footprints.shape[0]) for item in fixtures) != residual.STRICT_SCREEN_SOURCES:
            raise RankDisplacementError("fixture coverage drifted from the frozen 108/252 contract")
        checkpoint = residual._load_strict_checkpoint_payload(paths["checkpoint"], expected_sha256=config["upstream_screen"]["checkpoint"]["sha256"])
        provider = decoder_module.load_frozen_jepa_checkpoint(paths["checkpoint"], expected_sha256=config["upstream_screen"]["checkpoint"]["sha256"], map_location="cpu")
        random_provider = decoder_module.load_frozen_random_checkpoint(paths["checkpoint"], expected_sha256=config["upstream_screen"]["checkpoint"]["sha256"], map_location="cpu")
        checkpoint_manifest = _json(run_b_root / "decoder_checkpoints_manifest.json")
        checkpoint_row = checkpoint_manifest["checkpoints"][0]
        decoder_path = run_b_root / str(checkpoint_row["path"])
        if sha256_file(decoder_path) != str(checkpoint_row["sha256"]):
            raise RankDisplacementError("Run-B decoder checkpoint hash mismatch")
        decoder_payload = torch.load(decoder_path, map_location="cpu", weights_only=True)
        model = decoder_module.FrozenLatentPixelBackgroundModel(provider.provider, decoder_initialization_seed=int(decoder_payload["decoder_seed"])).to(device)
        random_model = decoder_module.FrozenLatentPixelBackgroundModel(random_provider.provider, decoder_initialization_seed=int(decoder_payload["decoder_seed"])).to(device)
        residual._decoder_module_only(model).load_state_dict(decoder_payload["decoder_state_dict"], strict=True)
        residual._decoder_module_only(random_model).load_state_dict(decoder_payload["random_control_decoder_state_dict"], strict=True)
        model.eval(); random_model.eval()
        if decoder_module.module_state_sha256(residual._decoder_module_only(model)) != checkpoint_manifest["decoder_state_identity"]["jepa_arm_runtime_module_state_sha256"]:
            raise RankDisplacementError("rehydrated JEPA decoder state differs from Run-B manifest")
        if decoder_module.module_state_sha256(residual._decoder_module_only(random_model)) != checkpoint_manifest["decoder_state_identity"]["random_arm_runtime_module_state_sha256"]:
            raise RankDisplacementError("rehydrated random decoder state differs from Run-B manifest")
        normalization = residual.normalization_from_config(config)
        evaluation = config["_evaluation"]
        existing_rows = _json(run_b_root / "paired_injection_results.json")["rows"]
        existing = {(str(row["fixture_id"]), str(row["method"])): row for row in existing_rows}
        if len(existing) != 324:
            raise RankDisplacementError("Run-B recovery table must contain exactly 324 fixture-method rows")
        rows: list[dict[str, Any]] = []
        recovered_validation = []
        prediction_amp = bool(config["_decoder"]["amp"] and str(device).startswith("cuda"))
        for fixture in sorted(fixtures, key=lambda item: item.fixture_id):
            metadata = fixture_metadata[str(fixture.fixture_id)]
            raw_pair = _score_pair(fixture.native_background, fixture.observation, candidate_budget=int(evaluation["proposal_cap"]), minimum_distance_px=int(evaluation["minimum_distance_px"]), border_px=int(evaluation["candidate_border_px"]))
            normal_off = residual.normalize_raw_movie_once(fixture.native_background, normalization)
            normal_on = residual.normalize_raw_movie_once(fixture.observation, normalization)
            pred_off, _ = residual._predict(decoder_module, model, normal_off, device=device, target_batch_size=int(config["_decoder"]["target_batch_size"]), amp=prediction_amp)
            pred_on, _ = residual._predict(decoder_module, model, normal_on, device=device, target_batch_size=int(config["_decoder"]["target_batch_size"]), amp=prediction_amp)
            random_pred_off, _ = residual._predict(decoder_module, random_model, normal_off, device=device, target_batch_size=int(config["_decoder"]["target_batch_size"]), amp=prediction_amp)
            random_pred_on, _ = residual._predict(decoder_module, random_model, normal_on, device=device, target_batch_size=int(config["_decoder"]["target_batch_size"]), amp=prediction_amp)
            pairs = {
                residual.RAW_METHOD: raw_pair,
                residual.JEPA_RESIDUAL_METHOD: _score_pair(normal_off - pred_off, normal_on - pred_on, candidate_budget=int(evaluation["proposal_cap"]), minimum_distance_px=int(evaluation["minimum_distance_px"]), border_px=int(evaluation["candidate_border_px"])),
                residual.RANDOM_RESIDUAL_METHOD: _score_pair(normal_off - random_pred_off, normal_on - random_pred_on, candidate_budget=int(evaluation["proposal_cap"]), minimum_distance_px=int(evaluation["minimum_distance_px"]), border_px=int(evaluation["candidate_border_px"])),
            }
            for method, pair in pairs.items():
                source_rows, observed_recovery = _source_rows(fixture=fixture, metadata=metadata, method=method, score_pair=pair, match_radius_px=float(evaluation["match_radius_px"]), local_radius_px=int(evaluation["minimum_distance_px"]) + 1)
                expected = existing.get((str(fixture.fixture_id), method))
                if expected is None or any(observed_recovery[key] != expected[key] for key in observed_recovery):
                    raise RankDisplacementError(f"reconstructed recovery drifted from Run-B: {fixture.fixture_id}/{method}")
                rows.extend(source_rows)
                recovered_validation.append({"fixture_id": fixture.fixture_id, "method": method, "matches_run_b_recovery_objects": True})
        if len(rows) != 3 * residual.STRICT_SCREEN_SOURCES:
            raise RankDisplacementError("diagnostic must contain exactly 756 source-method rows")
        counts = {method: {name: sum(row["classification"] == name for row in rows if row["method"] == method) for name in ("recovered_source_on", "native_background_competition", "attenuation_or_response_failure", "source_on_only_inconsistent")} for method in METHODS}
        grouped = summarize_rank_displacement(rows, bootstrap_seed=int(config["_evaluation"]["grouped_bootstrap_seed"]), draws=int(grouped_bootstrap_draws))
        dependency_hashes_after = _dependency_hashes(repository)
        if dependency_hashes_after != dependency_hashes:
            raise RankDisplacementError("live numerical dependency changed during execution")
        input_integrity_after = _verify_run_b(run_b_root)
        if input_integrity_after != input_integrity:
            raise RankDisplacementError("Run-B input package changed during execution")
        exp0028_after = residual.assert_parent_unchanged(
            exp0028_integrity,
            parent_root,
            expected_artifact_index_sha256=config["upstream_screen"][
                "artifact_index_sha256"
            ],
        )
        input_provenance = {
            "schema_version": 1,
            "hash_algorithm": "sha256",
            "configuration": {
                "path": CONFIG_PATH,
                "sha256": sha256_file(config_file),
            },
            "numerical_dependencies": dependency_hashes,
            "run_b": input_integrity,
            "exp_0028_parent": {
                "artifact_index_sha256": str(
                    exp0028_integrity["artifact_index_sha256"]
                ),
                "artifact_count": int(exp0028_integrity["artifact_count"]),
                "integrity_snapshot_sha256": str(
                    exp0028_integrity["snapshot_sha256"]
                ),
                "unchanged_pre_post": exp0028_after,
                "pinned_files": list(parent_integrity["files"]),
            },
            "decoder_checkpoint": {
                "path": str(checkpoint_row["path"]),
                "sha256": sha256_file(decoder_path),
                "jepa_decoder_state_sha256": checkpoint_manifest[
                    "decoder_state_identity"
                ]["jepa_arm_runtime_module_state_sha256"],
                "random_decoder_state_sha256": checkpoint_manifest[
                    "decoder_state_identity"
                ]["random_arm_runtime_module_state_sha256"],
            },
            "upstream_provider_checkpoint": {
                "path_role": "EXP-0028 Screen-B checkpoint",
                "complete_file_sha256": sha256_file(paths["checkpoint"]),
                "jepa_tensor_mapping_sha256": checkpoint["_jepa_tensor_sha256"],
                "random_tensor_mapping_sha256": checkpoint[
                    "_random_tensor_sha256"
                ],
            },
            "all_live_numerical_dependencies_unchanged_during_execution": True,
            "all_frozen_input_packages_unchanged_during_execution": True,
        }
        input_provenance["frozen_input_contract_sha256"] = stable_hash(
            input_provenance
        )
        ended_at = _utc_now()
        timing = _phase_timing(
            planned_at=planned_at,
            preflight_started_at=preflight_started_at,
            started_at=started_at,
            ended_at=ended_at,
        )
        execution_provenance = {
            "schema_version": 2,
            "diagnostic_id": DIAGNOSTIC_ID,
            **timing,
            "portable_reproduction_command": (
                ".venv-neurobench/bin/python -m "
                "neurobench.experiments.neuron_identifiability."
                "jepa_rank_displacement --repository-root . "
                "--data-root-env NEUROBENCH_DATA_ROOT --device "
                f"{device} --grouped-bootstrap-draws {int(grouped_bootstrap_draws)}"
            ),
            "command_uses_environment_placeholder_not_workstation_path": True,
            "git": git_provenance,
            "runtime": _runtime_provenance(
                device=str(device),
                old_threads=old_threads,
                configured_threads=configured_threads,
                prediction_amp=prediction_amp,
            ),
            "scientific_execution": False,
            "scientific_completion": False,
            "claim_promotion_allowed": False,
        }
        execution_provenance["execution_provenance_sha256"] = stable_hash(
            execution_provenance
        )
        summary = {
            "schema_version": 1,
            "diagnostic_id": DIAGNOSTIC_ID,
            "status": "complete_non_claim_bearing_exact_injection_diagnostic",
            "inputs": {
                **input_integrity,
                "parent_exp_0028_pinned_files_verified": bool(parent_integrity["all_pinned_parent_files_match"]),
                "fixture_identity": fixture_identity,
                "config_sha256": sha256_file(config_file),
                "implementation_sha256": sha256_file(Path(__file__)),
                "decoder_checkpoint_sha256": sha256_file(decoder_path),
                "upstream_checkpoint_tensor_sha256": checkpoint["_jepa_tensor_sha256"],
                "random_upstream_checkpoint_tensor_sha256": checkpoint["_random_tensor_sha256"],
                "frozen_input_contract_sha256": input_provenance[
                    "frozen_input_contract_sha256"
                ],
            },
            "execution": {
                **timing,
                "execution_provenance_sha256": execution_provenance[
                    "execution_provenance_sha256"
                ],
            },
            "coverage": {"fixture_count": len(fixtures), "source_count": residual.STRICT_SCREEN_SOURCES, "method_count": len(METHODS), "source_method_rows": len(rows), "all_324_recovery_objects_exactly_reproduced": len(recovered_validation) == 324},
            "classification_definition": {"recovered_source_on": "exact source recovered on both source-on and paired intervention maps", "source_on_only_inconsistent": "exact source recovered on source-on but not the paired intervention map; retained separately rather than forced into a mechanism category", "native_background_competition": "intervention recovered but source-on top-four did not recover the exact source", "attenuation_or_response_failure": "neither paired intervention nor source-on top-four recovered the exact source", "native_candidates": "unknown_not_negative"},
            "classification_counts": counts,
            "recovery_totals": _recovery_totals(counts),
            "grouped_uncertainty": grouped,
            "strata": {"mad": _stratified_summary(rows, "mad_stratum"), "recording": _stratified_summary(rows, "background_recording_id"), "morphology": _stratified_summary(rows, "morphology"), "source_count": _stratified_summary(rows, "source_count")},
            "interpretation_limits": ["Exact injected sources only; native candidates are unknown, not negatives.", "Run-B motion dependency and scientific audit remain incomplete.", "This is not a precision, biological identity, denoising, or JEPA-specific-benefit result."],
        }
        summary["result_sha256"] = stable_hash(summary)
        _atomic_tsv(partial / "rank_displacement_sources.tsv", rows)
        atomic_json(partial / "rank_displacement_summary.json", summary)
        atomic_json(partial / "input_provenance.json", input_provenance)
        atomic_json(partial / "execution_provenance.json", execution_provenance)
        atomic_json(partial / "grouped_uncertainty.json", {"schema_version": 1, "grouped_uncertainty": grouped})
        atomic_json(partial / "recovery_reproduction.json", {"schema_version": 1, "expected_fixture_method_rows": 324, "observed_fixture_method_rows": len(recovered_validation), "all_exact": True, "rows": recovered_validation})
        audit = {"schema_version": 1, "enabled": True, "derived_from": "NREV-RUN-EXP-0029-SCREEN-20260830-B", "required_media_complete": False, "scientific_audit_complete": False, "scientific_promotion_allowed": False, "reason": "Derived score-map mechanism diagnostic; source Run-B audit remains incomplete and no new media were generated."}
        atomic_json(partial / "scientific_audit_status.json", audit)
        atomic_json(partial / "llm_context.json", {"schema_version": 1, "entrypoint": "rank_displacement_summary.json", "primary_tables": ["rank_displacement_sources.tsv"], "provenance": ["input_provenance.json", "execution_provenance.json"], "coordinate_contract": "y=row,x=column", "native_candidate_semantics": "unknown_not_negative", "classification_boundary": summary["classification_definition"], "scientific_audit": audit})
        validation = {"schema_version": 1, "status": "passed", "run_b_artifact_integrity": True, "exp_0028_artifact_integrity": True, "fixture_coverage_exact_108_252": True, "all_324_recovery_objects_exactly_reproduced": True, "source_method_rows_exact_756": True, "all_values_finite": bool(all(math.isfinite(float(row["source_on_truth_center_score_z"])) and math.isfinite(float(row["intervention_truth_center_score_z"])) for row in rows)), "all_numerical_dependencies_hashed": len(dependency_hashes) == len(NUMERICAL_DEPENDENCIES), "dependencies_and_inputs_unchanged_pre_post": True, "explicit_planned_preflight_execution_end_phase_contract": True, "primary_duration_exactly_started_at_to_ended_at": timing["duration_seconds"] == float((datetime.fromisoformat(ended_at) - datetime.fromisoformat(started_at)).total_seconds()), "scipy_runtime_version_persisted": execution_provenance["runtime"]["scipy_version"] == scipy.__version__, "full_tree_index_validation_is_atomic_promotion_gate": True, "workstation_paths_absent": True, "scientific_audit_complete": False, "claim_promotion_allowed": False}
        atomic_json(partial / "validation.json", validation)
        atomic_json(partial / "status.json", {"schema_version": 1, "status": "complete_non_claim_bearing_exact_injection_diagnostic", "scientific_completion": False, "claim_promotion_allowed": False})
        atomic_text(partial / "REPORT.md", _report(summary))
        _assert_no_workstation_paths(partial, (repository, runtime_data))
        atomic_json(partial / "artifact_index.json", _artifact_index(partial))
        tree_validation = _verify_output_tree(partial)
        if not (
            tree_validation["all_existing_non_index_files_indexed"]
            and tree_validation["all_indexed_hashes_and_sizes_exact"]
        ):
            raise RankDisplacementError("full output-tree validation failed")
        partial.replace(final)
        final_tree_validation = _verify_output_tree(final)
        if final_tree_validation != tree_validation:
            raise RankDisplacementError("output tree changed during atomic promotion")
        return summary
    except Exception:
        atomic_json(partial / "status.json", {"schema_version": 1, "status": "failed_nonresumable_partial", "diagnostic_id": DIAGNOSTIC_ID})
        raise
    finally:
        torch.set_num_threads(old_threads)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--data-root-env", default="NEUROBENCH_DATA_ROOT")
    parser.add_argument("--device", default=None)
    parser.add_argument("--grouped-bootstrap-draws", type=int, default=1_000)
    arguments = parser.parse_args(argv)
    data_root_value = os.environ.get(str(arguments.data_root_env))
    if not data_root_value:
        parser.error(f"environment variable {arguments.data_root_env!r} is not set")
    summary = run_rank_displacement_diagnostic(
        repository_root=arguments.repository_root,
        data_root=Path(data_root_value),
        device=arguments.device,
        grouped_bootstrap_draws=arguments.grouped_bootstrap_draws,
    )
    print(
        json.dumps(
            {
                "diagnostic_id": summary["diagnostic_id"],
                "status": summary["status"],
                "result_sha256": summary["result_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the real module run
    raise SystemExit(main())


__all__ = ["DIAGNOSTIC_ID", "DEFAULT_OUTPUT_ROOT", "RankDisplacementError", "ScorePair", "classify_source", "main", "run_rank_displacement_diagnostic", "summarize_rank_displacement"]
