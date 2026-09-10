"""Materialize the frozen Innovation Ranker V5 proposal census.

This module is deliberately narrower than an experiment runner.  It reuses the
already frozen map and candidate-table primitives, verifies their exact input
and code provenance, and writes an unlabeled row-level census for downstream
positive-unlabeled analyses.  It does not join labels, fit preprocessing, fit a
model, or choose a detection threshold.

Candidate feature values are the quiet-normalized values already sampled by
Innovation Ranker V5.  They are not calibrated biological probabilities.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from neurobench.algorithms.proposal_ranking import CandidateTable
from neurobench.experiments.hierarchical_parzen_ica.feature_utility_config import (
    FeatureUtilityConfig,
)
from neurobench.experiments.hierarchical_parzen_ica.innovation_grid import (
    _progress,
)
from neurobench.experiments.hierarchical_parzen_ica.innovation_ranker_config import (
    FEATURE_IDS,
    PROPOSAL_SOURCE_IDS,
    InnovationRankerConfig,
)
from neurobench.experiments.hierarchical_parzen_ica.innovation_ranker_program import (
    _candidate_tables,
    _generate_maps,
)
from neurobench.experiments.learnable_contrast import core as label_core
from neurobench.portable_paths import portable_path, portableize_paths

from .contracts import atomic_json, stable_hash


class CensusContractError(RuntimeError):
    """Raised when the frozen V5 census cannot be reproduced exactly."""


EXPECTED_EVENT_COUNTS = {1: 544, 2: 530, 3: 476, 4: 449}
EXPECTED_QUIET_COUNTS = (683, 748, 759, 748)

EXPECTED_PROPOSAL_SOURCE_IDS = (
    "carrier_signed",
    "local_psd_signal",
    "asymmetric_state",
    "spatial_coherence",
    "cross_scale_rank",
    "cross_scale_recall",
    "cfar_score",
    "signed_power_1p25",
    "signed_power_1p5",
    "signed_power_2",
    "shot_whitened_rho0p25",
    "shot_whitened_rho0p5",
    "shot_whitened_rho1",
    "cut_center_sigma1p5",
    "cut_center_sigma2p5",
    "cut_center_sigma3p5",
    "cut_ring_r3p0_t0p75",
    "cut_ring_r3p0_t1p25",
    "cut_ring_r4p5_t0p75",
    "cut_ring_r4p5_t1p25",
    "cut_ring_r6p0_t0p75",
    "cut_ring_r6p0_t1p25",
)

FROZEN_PROPOSAL_CONTRACT = {
    "experiment_id": "spon_ca_burst_innovation_ranker_v5",
    "source_ids": list(EXPECTED_PROPOSAL_SOURCE_IDS),
    "source_count": 22,
    "per_source_limit": 100,
    "nms_distance_px": 6,
    "dedupe_radius_px": 3.0,
    "candidate_labels_joined": False,
}

# These are the frozen scientific inputs recorded by the completed V5
# preflight.  The configuration itself is validated below with a portable
# semantic hash so an exact data-root relocation does not invalidate it.
EXPECTED_INPUT_SHA256 = {
    "feature_manifest": "e53c92dc9f58f6b5f2f9e37f207ffb05a908c04304e27218f632d6cc00295361",
    "source_video": "04dfbe2f7cb69d72ff75e23ad17c87b3fd406c96a4b872148de2285a9a44d449",
    "event_window_labels": "8f008fede5771d83e949d805094292aae7516cc30353dd383da7ff1cf20b077e",
    "feature_utility_run_state": "7a8eba3ef0bade5644eb7055bd85bfc4e5993589b5aebe4b6ee0af445da77c33",
    "feature_utility_config": "48fcb5e7714e6e64ab12bba7b4656498bae2d5fbc43e5c15fce0f82e80586e07",
}

EXPECTED_PORTABLE_CONFIG_SHA256 = (
    "4b86fbfce306ec5bf2a28d427f4f2cadf45afa2324c62834c7a816f93bbe9b75"
)

# Pin every checked-in scientific primitive called while constructing the
# maps and proposal tables.  Deliberate changes require a new census version,
# not an unnoticed continuation of the V5 claim.
EXPECTED_CODE_SHA256 = {
    "innovation_ranker_program": "1b4bd8a8f326c87b20e0f10c1b45fb4f2b1dfda92e54e4145366de212e804dcd",
    "innovation_ranker_config": "2ca2ed10b924af1ea0701e0475c0ba212614b0e2c14d604320b8f10c5deaa188",
    "proposal_ranking": "0407b541874e288bc4300861e9deeee495fa3b778e5b2e0f81c80fd763a6c54b",
    "feature_utility_program": "94bb6f6c47573d1b33ab6aeb9dbdab8af17b0f907355a13e09aa73541a0017f8",
    "feature_utility_config": "321eaecc90ecd4e3264eca7f8c67563ba7ff600c67b34eb11ff5a242c2b2b4cb",
    "sparse_detection": "d661201ec5193e47df2c9f9260a02c5dea329b5b30f6ba6df13a05599b582b46",
    "label_loader": "e321ed1953115ad2253d84748ae39ae53d8807a80b94cb3ae571e37351273a3e",
}

CODE_RELATIVE_PATHS = {
    "innovation_ranker_program": "neurobench/experiments/hierarchical_parzen_ica/innovation_ranker_program.py",
    "innovation_ranker_config": "neurobench/experiments/hierarchical_parzen_ica/innovation_ranker_config.py",
    "proposal_ranking": "neurobench/algorithms/proposal_ranking.py",
    "feature_utility_program": "neurobench/experiments/hierarchical_parzen_ica/feature_utility_program.py",
    "feature_utility_config": "neurobench/experiments/hierarchical_parzen_ica/feature_utility_config.py",
    "sparse_detection": "neurobench/metrics/sparse_detection.py",
    "label_loader": "neurobench/experiments/learnable_contrast/core.py",
}

CANDIDATE_BASE_COLUMNS = (
    "candidate_id",
    "partition",
    "partition_id",
    "proposal_order",
    "burst_id",
    "quiet_map_id",
    "x_px",
    "y_px",
    "source_count",
)


def stable_candidate_id(
    partition: str, partition_id: int, x_px: int, y_px: int
) -> str:
    """Return a coordinate-stable identifier independent of proposal order."""
    if partition not in {"event", "quiet"}:
        raise ValueError(f"unsupported partition: {partition!r}")
    values = (int(partition_id), int(x_px), int(y_px))
    if values[0] < 1 or min(values[1:]) < 0:
        raise ValueError("partition IDs must be positive and coordinates nonnegative")
    return (
        f"irv5_{partition}_p{values[0]:02d}_"
        f"x{values[1]:05d}_y{values[2]:05d}"
    )


def _validated_candidate_arrays(
    table: CandidateTable, feature_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    names = tuple(str(value) for value in feature_ids)
    if not names or len(names) != len(set(names)):
        raise CensusContractError("feature IDs must be non-empty and unique")

    positions = np.asarray(table.positions)
    features = np.asarray(table.features)
    source_count = np.asarray(table.source_count)
    if positions.ndim != 2 or positions.shape[1:] != (2,):
        raise CensusContractError("candidate positions must have shape (N, 2)")
    if features.shape != (len(positions), len(names)):
        raise CensusContractError(
            "candidate feature schema differs from the declared feature IDs"
        )
    if source_count.shape != (len(positions),):
        raise CensusContractError("candidate source_count does not align")
    if not np.issubdtype(positions.dtype, np.integer):
        raise CensusContractError("candidate coordinates must use an integer dtype")
    if not np.issubdtype(source_count.dtype, np.integer):
        raise CensusContractError("candidate source_count must use an integer dtype")
    if np.any(positions < 0):
        raise CensusContractError("candidate coordinates must be nonnegative")
    if len({(int(x), int(y)) for x, y in positions}) != len(positions):
        raise CensusContractError("candidate table contains duplicate coordinates")
    if np.any(source_count < 1) or np.any(
        source_count > len(EXPECTED_PROPOSAL_SOURCE_IDS)
    ):
        raise CensusContractError("candidate source_count is outside [1, 22]")
    if not np.issubdtype(features.dtype, np.number) or not np.isfinite(
        features
    ).all():
        raise CensusContractError("candidate features must be finite numeric values")
    return positions, features, source_count, names


def candidate_table_rows(
    table: CandidateTable,
    *,
    partition: str,
    partition_id: int,
    feature_ids: Sequence[str] = FEATURE_IDS,
) -> list[dict[str, Any]]:
    """Flatten one :class:`CandidateTable` into deterministic TSV-ready rows."""
    if partition not in {"event", "quiet"}:
        raise CensusContractError(f"unsupported partition: {partition!r}")
    partition_id = int(partition_id)
    if partition_id < 1:
        raise CensusContractError("partition_id must be positive")
    positions, features, source_count, names = _validated_candidate_arrays(
        table, feature_ids
    )
    rows: list[dict[str, Any]] = []
    for index, ((x_px, y_px), values, sources) in enumerate(
        zip(positions, features, source_count), start=1
    ):
        x_value, y_value = int(x_px), int(y_px)
        row: dict[str, Any] = {
            "candidate_id": stable_candidate_id(
                partition, partition_id, x_value, y_value
            ),
            "partition": partition,
            "partition_id": partition_id,
            "proposal_order": index,
            "burst_id": partition_id if partition == "event" else None,
            "quiet_map_id": partition_id if partition == "quiet" else None,
            "x_px": x_value,
            "y_px": y_value,
            "source_count": int(sources),
        }
        row.update(
            {
                f"feature__{feature_id}": float(value)
                for feature_id, value in zip(names, values)
            }
        )
        rows.append(row)
    return rows


def _require_counts(
    *,
    event_rows: Sequence[Mapping[str, Any]],
    quiet_rows: Sequence[Mapping[str, Any]],
    expected_event_counts: Mapping[int, int],
    expected_quiet_counts: Sequence[int],
) -> dict[str, Any]:
    observed_event_ids = {int(row["partition_id"]) for row in event_rows}
    observed_quiet_ids = {int(row["partition_id"]) for row in quiet_rows}
    expected_event_ids = {int(value) for value in expected_event_counts}
    expected_quiet_ids = set(range(1, len(expected_quiet_counts) + 1))
    if observed_event_ids != expected_event_ids or observed_quiet_ids != expected_quiet_ids:
        raise CensusContractError(
            "frozen candidate partition IDs differ; "
            f"event observed={sorted(observed_event_ids)} "
            f"expected={sorted(expected_event_ids)}; "
            f"quiet observed={sorted(observed_quiet_ids)} "
            f"expected={sorted(expected_quiet_ids)}"
        )
    observed_event = {
        burst: sum(int(row["partition_id"]) == burst for row in event_rows)
        for burst in sorted(expected_event_counts)
    }
    observed_quiet = tuple(
        sum(int(row["partition_id"]) == index for row in quiet_rows)
        for index in range(1, len(expected_quiet_counts) + 1)
    )
    expected_event = {int(key): int(value) for key, value in expected_event_counts.items()}
    expected_quiet = tuple(int(value) for value in expected_quiet_counts)
    if (
        observed_event != expected_event
        or observed_quiet != expected_quiet
        or len(event_rows) != sum(expected_event.values())
        or len(quiet_rows) != sum(expected_quiet)
    ):
        raise CensusContractError(
            "frozen candidate counts differ; "
            f"event observed={observed_event} expected={expected_event}; "
            f"quiet observed={observed_quiet} expected={expected_quiet}"
        )
    return {
        "event": {str(key): value for key, value in observed_event.items()},
        "quiet": list(observed_quiet),
        "event_total": len(event_rows),
        "quiet_total": len(quiet_rows),
    }


def materialize_candidate_rows(
    *,
    quiet_tables: Sequence[CandidateTable],
    event_tables: Mapping[int, CandidateTable],
    feature_ids: Sequence[str] = FEATURE_IDS,
    expected_event_counts: Mapping[int, int] | None = None,
    expected_quiet_counts: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Materialize event and quiet tables without labels or model decisions.

    Passing expected counts is optional for synthetic callers.  The live V5
    builder always supplies the frozen counts and therefore fails closed.
    """
    event_keys = tuple(sorted(int(key) for key in event_tables))
    if event_keys != tuple(sorted(event_tables)):
        raise CensusContractError("event-table keys must be integer burst IDs")
    event_rows = [
        row
        for burst in event_keys
        for row in candidate_table_rows(
            event_tables[burst],
            partition="event",
            partition_id=burst,
            feature_ids=feature_ids,
        )
    ]
    quiet_rows = [
        row
        for index, table in enumerate(quiet_tables, start=1)
        for row in candidate_table_rows(
            table,
            partition="quiet",
            partition_id=index,
            feature_ids=feature_ids,
        )
    ]
    all_ids = [row["candidate_id"] for row in (*event_rows, *quiet_rows)]
    if len(all_ids) != len(set(all_ids)):
        raise CensusContractError("candidate IDs are not globally unique")

    counts = {
        "event": {
            str(burst): sum(row["partition_id"] == burst for row in event_rows)
            for burst in event_keys
        },
        "quiet": [len(table.positions) for table in quiet_tables],
        "event_total": len(event_rows),
        "quiet_total": len(quiet_rows),
    }
    if (expected_event_counts is None) != (expected_quiet_counts is None):
        raise CensusContractError("event and quiet expected counts must be supplied together")
    if expected_event_counts is not None and expected_quiet_counts is not None:
        counts = _require_counts(
            event_rows=event_rows,
            quiet_rows=quiet_rows,
            expected_event_counts=expected_event_counts,
            expected_quiet_counts=expected_quiet_counts,
        )
    return {"event_rows": event_rows, "quiet_rows": quiet_rows, "counts": counts}


def census_feature_dictionary(
    feature_ids: Sequence[str] = FEATURE_IDS,
) -> dict[str, Any]:
    """Describe the unlabeled, normalized candidate-row schema."""
    names = tuple(str(value) for value in feature_ids)
    if not names or len(names) != len(set(names)):
        raise CensusContractError("feature IDs must be non-empty and unique")
    payload = {
        "schema_version": 1,
        "census_id": "innovation_ranker_v5_broad_proposal_census",
        "grain": "one NMS-deduplicated candidate in one event or quiet map",
        "coordinate_convention": "x=column, y=row, zero-based pixels",
        "candidate_id": (
            "coordinate-stable within partition and partition_id; proposal_order "
            "is not part of identity"
        ),
        "base_columns": {
            "partition": "event or quiet; not a biological class label",
            "partition_id": "one-based burst ID or quiet-map ID",
            "proposal_order": "one-based order returned by frozen V5 proposal merger",
            "source_count": "number of the 22 proposal sources merged at this coordinate",
        },
        "feature_order": list(names),
        "feature_columns": [f"feature__{value}" for value in names],
        "feature_semantics": {
            f"feature__{value}": {
                "dtype": "finite float64 serialization of sampled float32",
                "normalization": (
                    "Innovation Ranker V5 quiet-only robust map normalization "
                    "followed by monotone log1p compression"
                ),
            }
            for value in names
        },
        "candidate_label_state": "unlabeled",
        "prohibited_interpretation": (
            "quiet candidates are source-off controls, not verified non-neurons; "
            "unmatched event candidates remain unknown"
        ),
        "proposal_contract": FROZEN_PROPOSAL_CONTRACT,
    }
    payload["feature_dictionary_sha256"] = stable_hash(payload)
    return payload


def sha256_file(path: Path) -> str:
    """Hash one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_and_verify_files(
    files: Mapping[str, Path], expected: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    """Return exact file hashes and raise on a missing, extra, or changed role."""
    if set(files) != set(expected):
        raise CensusContractError(
            "hash role schema differs; "
            f"missing={sorted(set(expected) - set(files))}; "
            f"extra={sorted(set(files) - set(expected))}"
        )
    results: dict[str, dict[str, Any]] = {}
    for role in sorted(files):
        path = Path(files[role])
        if not path.is_file():
            raise CensusContractError(f"missing frozen {role}: {path}")
        observed = sha256_file(path)
        wanted = str(expected[role])
        if observed != wanted:
            raise CensusContractError(
                f"frozen {role} hash differs: observed={observed} expected={wanted}"
            )
        results[role] = {
            "bytes": path.stat().st_size,
            "sha256": observed,
        }
    return results


def validate_frozen_contract(config: InnovationRankerConfig) -> None:
    """Reject any configuration that is not the exact V5 proposal contract."""
    failures = []
    if config.experiment_id != FROZEN_PROPOSAL_CONTRACT["experiment_id"]:
        failures.append("experiment_id")
    if tuple(PROPOSAL_SOURCE_IDS) != EXPECTED_PROPOSAL_SOURCE_IDS:
        failures.append("checked-in proposal source IDs")
    proposals = config.proposals
    if tuple(proposals.get("source_ids", ())) != EXPECTED_PROPOSAL_SOURCE_IDS:
        failures.append("config proposal source IDs")
    if int(proposals.get("per_source_limit", -1)) != 100:
        failures.append("per_source_limit")
    if int(proposals.get("nms_distance_px", -1)) != 6:
        failures.append("nms_distance_px")
    if float(proposals.get("dedupe_radius_px", -1)) != 3.0:
        failures.append("dedupe_radius_px")
    if len(FEATURE_IDS) != 34:
        failures.append("feature count")
    if failures:
        raise CensusContractError(
            "configuration differs from frozen Innovation Ranker V5: "
            + ", ".join(failures)
        )


def _validate_feature_maps(
    feature_maps: Mapping[str, Mapping[str, Any]],
) -> tuple[int, int]:
    if set(feature_maps) != set(FEATURE_IDS):
        raise CensusContractError(
            "feature-map schema differs; "
            f"missing={sorted(set(FEATURE_IDS) - set(feature_maps))}; "
            f"extra={sorted(set(feature_maps) - set(FEATURE_IDS))}"
        )
    common_shape: tuple[int, int] | None = None
    for feature_id in FEATURE_IDS:
        maps = feature_maps[feature_id]
        quiet = maps.get("quiet")
        events = maps.get("events")
        if not isinstance(quiet, Sequence) or len(quiet) != 4:
            raise CensusContractError(f"{feature_id} does not have four quiet maps")
        if not isinstance(events, Mapping) or set(events) != set(EXPECTED_EVENT_COUNTS):
            raise CensusContractError(f"{feature_id} event-map schema differs")
        for value in (*quiet, *(events[burst] for burst in sorted(events))):
            array = np.asarray(value)
            if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
                raise CensusContractError(f"{feature_id} contains a nonnumeric map")
            if not np.isfinite(array).all():
                raise CensusContractError(f"{feature_id} contains non-finite map values")
            shape = (int(array.shape[0]), int(array.shape[1]))
            if common_shape is None:
                common_shape = shape
            elif shape != common_shape:
                raise CensusContractError(
                    f"{feature_id} map shape {shape} differs from {common_shape}"
                )
    if common_shape is None:
        raise CensusContractError("feature-map collection is empty")
    return common_shape


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise CensusContractError(f"refusing to write empty census table: {path.name}")
    fields = [*CANDIDATE_BASE_COLUMNS, *(f"feature__{value}" for value in FEATURE_IDS)]
    expected = set(fields)
    for row in rows:
        if set(row) != expected:
            raise CensusContractError(
                f"candidate row schema differs while writing {path.name}"
            )
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
    temporary.replace(path)


def _resolved(path: Path, repository_root: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (repository_root / value).resolve()


def _inside(path: Path, root: Path, role: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise CensusContractError(f"{role} is outside its declared authority root") from exc


def _preflight_context(
    *,
    repository_root: Path,
    data_root: Path,
    config_path: Path,
    output_root: Path,
    progress: Path | None = None,
) -> dict[str, Any]:
    repository = Path(repository_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    config_file = _resolved(Path(config_path), repository)
    output = _resolved(Path(output_root), repository)
    partial = Path(str(output) + ".partial")
    progress_path = (
        _resolved(Path(progress), repository)
        if progress is not None
        else output.with_name(output.name + ".progress.jsonl")
    )

    if not repository.is_dir() or not data.is_dir():
        raise CensusContractError("repository_root and data_root must exist")
    if not (repository / "AGENTS.md").is_file():
        raise CensusContractError("repository_root is not the NeuRev checkout")
    materializer_path = Path(__file__).resolve()
    _inside(materializer_path, repository, "loaded census materializer")
    if output.exists() or partial.exists():
        raise FileExistsError("completed or partial census output already exists")
    if progress_path.exists():
        raise FileExistsError("progress sidecar already exists")
    if progress_path == output or progress_path == partial:
        raise CensusContractError("progress must be separate from census output")

    config = InnovationRankerConfig.load(config_file)
    validate_frozen_contract(config)
    for role, path in {
        "feature_root": config.feature_root,
        "feature_manifest": config.feature_manifest,
        "source_video": config.source_video,
        "event_window_labels": config.labels_tsv,
    }.items():
        _inside(Path(path), data, role)

    portable_config = portableize_paths(
        config.to_dict(), repository=repository, data=data
    )
    portable_config_sha256 = stable_hash(portable_config)
    if portable_config_sha256 != EXPECTED_PORTABLE_CONFIG_SHA256:
        raise CensusContractError(
            "portable V5 config hash differs: "
            f"observed={portable_config_sha256} "
            f"expected={EXPECTED_PORTABLE_CONFIG_SHA256}"
        )

    input_files = {
        "feature_manifest": config.feature_manifest,
        "source_video": config.source_video,
        "event_window_labels": config.labels_tsv,
        "feature_utility_run_state": config.feature_root / "run_state.json",
        "feature_utility_config": config.feature_root / "config.resolved.json",
    }
    code_files = {
        role: repository / relative for role, relative in CODE_RELATIVE_PATHS.items()
    }
    input_hashes = hash_and_verify_files(input_files, EXPECTED_INPUT_SHA256)
    code_hashes = hash_and_verify_files(code_files, EXPECTED_CODE_SHA256)
    payload = {
        "schema_version": 1,
        "kind": "read_only_uncertainty_aware_census_preflight",
        "ready": True,
        "gates": {
            "repository_valid": True,
            "data_authority_valid": True,
            "output_absent": True,
            "partial_output_absent": True,
            "progress_absent": True,
            "proposal_contract_exact": True,
            "portable_config_hash_exact": True,
            "input_hashes_exact": True,
            "code_hashes_exact": True,
        },
        "paths": {
            "config": portable_path(config_file, repository=repository, data=data),
            "output": portable_path(output, repository=repository, data=data),
            "progress": portable_path(
                progress_path, repository=repository, data=data
            ),
        },
        "config": {
            "file_sha256": sha256_file(config_file),
            "portable_semantic_sha256": portable_config_sha256,
            "portable_resolved": portable_config,
        },
        "inputs": {
            role: {
                "path": portable_path(
                    Path(input_files[role]), repository=repository, data=data
                ),
                **input_hashes[role],
            }
            for role in sorted(input_files)
        },
        "code": {
            role: {
                "path": portable_path(
                    Path(code_files[role]), repository=repository, data=data
                ),
                **code_hashes[role],
            }
            for role in sorted(code_files)
        },
        "materializer_code": {
            "path": portable_path(
                materializer_path, repository=repository, data=data
            ),
            "bytes": materializer_path.stat().st_size,
            "sha256": sha256_file(materializer_path),
        },
        "proposal_contract": FROZEN_PROPOSAL_CONTRACT,
        "expected_counts": {
            "event": {
                str(key): value for key, value in EXPECTED_EVENT_COUNTS.items()
            },
            "quiet": list(EXPECTED_QUIET_COUNTS),
        },
        "scientific_scope": (
            "read-only provenance and collision check; does not generate maps, "
            "join candidate labels, fit preprocessing, or fit models"
        ),
    }
    return {
        "payload": payload,
        "repository": repository,
        "data": data,
        "config_file": config_file,
        "output": output,
        "partial": partial,
        "progress_path": progress_path,
        "config": config,
        "portable_config": portable_config,
        "portable_config_sha256": portable_config_sha256,
        "input_files": input_files,
        "input_hashes": input_hashes,
        "code_files": code_files,
        "code_hashes": code_hashes,
    }


def preflight_uncertainty_aware_census(
    *,
    repository_root: Path,
    data_root: Path,
    config_path: Path,
    output_root: Path,
    progress: Path | None = None,
) -> dict[str, Any]:
    """Validate the exact V5 inputs and collision contract without writing."""
    return _preflight_context(
        repository_root=repository_root,
        data_root=data_root,
        config_path=config_path,
        output_root=output_root,
        progress=progress,
    )["payload"]


def build_uncertainty_aware_census(
    *,
    repository_root: Path,
    data_root: Path,
    config_path: Path,
    output_root: Path,
    progress: Path | None = None,
) -> dict[str, Any]:
    """Build one collision-safe, unlabeled V5 proposal census.

    ``data_root`` may be the repository itself or a relocated frozen-data
    authority.  Serialized paths use ``repo://`` and ``data://`` identifiers.
    The progress JSONL is a live operational sidecar and is intentionally not
    part of the deterministic census hashes.
    """
    context = _preflight_context(
        repository_root=repository_root,
        data_root=data_root,
        config_path=config_path,
        output_root=output_root,
        progress=progress,
    )
    repository = context["repository"]
    data = context["data"]
    config_file = context["config_file"]
    output = context["output"]
    partial = context["partial"]
    progress_path = context["progress_path"]
    config = context["config"]
    portable_config = context["portable_config"]
    portable_config_sha256 = context["portable_config_sha256"]
    input_files = context["input_files"]
    input_hashes = context["input_hashes"]
    code_files = context["code_files"]
    code_hashes = context["code_hashes"]
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    _progress(progress_path, "census_preflight_passed")

    event_window_rows = label_core.load_labels(config.labels_tsv)
    base_config = FeatureUtilityConfig.load(
        config.feature_root / "config.resolved.json"
    )
    feature_maps, _calibration = _generate_maps(
        config, event_window_rows, base_config, progress_path
    )
    map_shape = _validate_feature_maps(feature_maps)
    _progress(progress_path, "census_feature_maps_validated", shape=list(map_shape))
    _normalizers, quiet_tables, event_tables = _candidate_tables(
        feature_maps, config
    )
    materialized = materialize_candidate_rows(
        quiet_tables=quiet_tables,
        event_tables=event_tables,
        feature_ids=FEATURE_IDS,
        expected_event_counts=EXPECTED_EVENT_COUNTS,
        expected_quiet_counts=EXPECTED_QUIET_COUNTS,
    )
    event_rows = materialized["event_rows"]
    quiet_rows = materialized["quiet_rows"]
    counts = materialized["counts"]
    _progress(progress_path, "census_candidate_counts_validated", **counts)

    dictionary = census_feature_dictionary()
    partial.mkdir(parents=True)
    event_path = partial / "event_candidates.tsv"
    quiet_path = partial / "quiet_candidates.tsv"
    _write_tsv(event_path, event_rows)
    _write_tsv(quiet_path, quiet_rows)
    atomic_json(partial / "feature_dictionary.json", dictionary)

    input_records = {
        role: {
            "path": portable_path(
                Path(input_files[role]), repository=repository, data=data
            ),
            **input_hashes[role],
        }
        for role in sorted(input_files)
    }
    code_records = {
        role: {
            "path": portable_path(
                Path(code_files[role]), repository=repository, data=data
            ),
            **code_hashes[role],
        }
        for role in sorted(code_files)
    }
    table_hashes = {
        "event_rows_sha256": stable_hash(event_rows),
        "quiet_rows_sha256": stable_hash(quiet_rows),
        "event_tsv_sha256": sha256_file(event_path),
        "quiet_tsv_sha256": sha256_file(quiet_path),
    }
    provenance = {
        "schema_version": 1,
        "census_id": "innovation_ranker_v5_broad_proposal_census",
        "candidate_label_state": "unlabeled",
        "proposal_contract": FROZEN_PROPOSAL_CONTRACT,
        "proposal_contract_sha256": stable_hash(FROZEN_PROPOSAL_CONTRACT),
        "config": {
            "path": portable_path(config_file, repository=repository, data=data),
            "file_sha256": sha256_file(config_file),
            "portable_semantic_sha256": portable_config_sha256,
            "portable_resolved": portable_config,
        },
        "inputs": input_records,
        "code": code_records,
        "materializer_code": context["payload"]["materializer_code"],
        "tables": table_hashes,
        "event_window_contract": {
            "source": "frozen 79-row sparse label sheet",
            "use": "derive the already frozen V5 event intervals only",
            "candidate_labels_joined": False,
        },
        "implementation": {
            "map_builder": (
                "neurobench.experiments.hierarchical_parzen_ica."
                "innovation_ranker_program._generate_maps"
            ),
            "candidate_builder": (
                "neurobench.experiments.hierarchical_parzen_ica."
                "innovation_ranker_program._candidate_tables"
            ),
        },
    }
    validation = {
        "schema_version": 1,
        "passed": True,
        "gates": {
            "portable_config_hash_exact": True,
            "input_hashes_exact": True,
            "code_hashes_exact": True,
            "proposal_contract_exact": True,
            "feature_map_schema_exact": True,
            "feature_map_values_finite": True,
            "candidate_schema_exact": True,
            "candidate_values_finite": True,
            "candidate_ids_unique": True,
            "candidate_counts_exact": True,
            "candidate_labels_absent": True,
        },
        "counts": counts,
        "map_shape_yx": list(map_shape),
        "feature_count": len(FEATURE_IDS),
        "proposal_source_count": len(PROPOSAL_SOURCE_IDS),
        "table_hashes": table_hashes,
    }
    summary = {
        "schema_version": 1,
        "census_id": provenance["census_id"],
        "status": "materialized_unlabeled_census",
        "counts": counts,
        "feature_count": len(FEATURE_IDS),
        "proposal_source_count": len(PROPOSAL_SOURCE_IDS),
        "paths": {
            "event_candidates": "event_candidates.tsv",
            "quiet_candidates": "quiet_candidates.tsv",
            "feature_dictionary": "feature_dictionary.json",
            "provenance": "provenance.json",
            "validation": "validation.json",
        },
        "scientific_scope": (
            "development census from one recording and frozen V5 proposal "
            "distribution; no candidate labels or model decisions"
        ),
    }
    atomic_json(partial / "provenance.json", provenance)
    atomic_json(partial / "validation.json", validation)
    atomic_json(partial / "summary.json", summary)
    partial.replace(output)
    _progress(
        progress_path,
        "census_completed",
        output=portable_path(output, repository=repository, data=data),
        event_total=counts["event_total"],
        quiet_total=counts["quiet_total"],
    )
    return summary


__all__ = [
    "CANDIDATE_BASE_COLUMNS",
    "CensusContractError",
    "EXPECTED_CODE_SHA256",
    "EXPECTED_EVENT_COUNTS",
    "EXPECTED_INPUT_SHA256",
    "EXPECTED_PORTABLE_CONFIG_SHA256",
    "EXPECTED_PROPOSAL_SOURCE_IDS",
    "EXPECTED_QUIET_COUNTS",
    "FROZEN_PROPOSAL_CONTRACT",
    "build_uncertainty_aware_census",
    "candidate_table_rows",
    "census_feature_dictionary",
    "hash_and_verify_files",
    "materialize_candidate_rows",
    "preflight_uncertainty_aware_census",
    "sha256_file",
    "stable_candidate_id",
    "validate_frozen_contract",
]
