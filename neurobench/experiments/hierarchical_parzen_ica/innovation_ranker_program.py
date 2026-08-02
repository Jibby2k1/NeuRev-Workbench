"""Nested proposal-generation and fine-tuning study for Spon Ca Burst."""
from __future__ import annotations

import csv
import itertools
import json
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import tifffile

from neurobench.algorithms.proposal_ranking import (
    CandidateTable,
    cut_morphology_basis,
    fit_bounded_pairwise_linear,
    fit_residual_mlp_ranker,
    merge_peak_proposals,
    normalize_map,
    robust_map_normalizer,
    sample_candidate_features,
    score_bounded_pairwise_linear,
    score_residual_mlp_ranker,
)
from neurobench.experiments.learnable_contrast import core as label_core
from neurobench.metrics.sparse_detection import (
    extract_local_maxima,
    match_peaks_one_to_one,
)

from .feature_utility_config import FeatureUtilityConfig
from .feature_utility_program import _pooled_maps
from .innovation_grid import (
    _atomic_json,
    _available_ram_mib,
    _progress,
    _sha256,
    _snapshots,
)
from .innovation_ranker_config import (
    EXISTING_FEATURE_IDS,
    FEATURE_IDS,
    FEATURE_SETS,
    GENERATED_FEATURE_IDS,
    NEGATIVE_EVIDENCE_IDS,
    PROPOSAL_SOURCE_IDS,
    InnovationRankerConfig,
)


BURSTS = (1, 2, 3, 4)


def _atomic_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def preflight(
    config: InnovationRankerConfig, *, write_artifacts: bool = True
) -> dict[str, Any]:
    inputs = (
        config.feature_manifest,
        config.source_video,
        config.labels_tsv,
        config.feature_root / "run_state.json",
        config.feature_root / "config.resolved.json",
    )
    missing = [str(path) for path in inputs if not path.is_file()]
    source_shape = None
    source_dtype = None
    labels: list[dict[str, Any]] = []
    feature_valid = labels_valid = bounds_valid = finite = False
    if not missing:
        source = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
        source_shape = list(source.shape)
        source_dtype = str(source.dtype)
        start = int(config.frames["review_start_ui"]) - 1
        stop = int(config.frames["review_end_ui"])
        bounds_valid = (
            source.ndim == 3
            and 0 <= start < start + int(config.frames["quiet_count"]) < stop
            <= len(source)
        )
        finite = bounds_valid and bool(
            np.isfinite(source[start:stop:32, ::16, ::16]).all()
        )
        labels = label_core.load_labels(config.labels_tsv)
        labels_valid = bool(
            len(labels) == 79
            and len({row["roi_identity"] for row in labels}) == 27
            and all(
                0 <= row["x_px"] < source.shape[2]
                and 0 <= row["y_px"] < source.shape[1]
                for row in labels
            )
        )
        manifest = json.loads(
            config.feature_manifest.read_text(encoding="utf-8")
        )
        state = json.loads(
            (config.feature_root / "run_state.json").read_text(encoding="utf-8")
        )
        available = {
            manifest["carrier"]["feature_id"],
            *[row["feature_id"] for row in manifest["features"]],
        }
        feature_valid = bool(
            manifest.get("feature_count") == 25
            and state.get("status") == "completed"
            and state.get("feature_count") == 25
            and set(EXISTING_FEATURE_IDS) <= available
            and all(
                (config.feature_root / "features" / f"{feature_id}.npy").is_file()
                for feature_id in EXISTING_FEATURE_IDS
            )
        )
    frames = (
        int(config.frames["review_end_ui"])
        - int(config.frames["review_start_ui"])
        + 1
    )
    pixels = (
        0 if source_shape is None else int(source_shape[1]) * int(source_shape[2])
    )
    dense_mib = frames * pixels * 4 / 2**20
    estimated_ram = 10.0 * dense_mib + 1024
    estimated_output = 512.0
    probe = config.output_dir.parent
    while not probe.exists():
        probe = probe.parent
    free_disk = shutil.disk_usage(probe).free / 2**20
    gates = {
        "inputs_exist": not missing,
        "source_bounds_valid": bounds_valid,
        "finite_source_sample": finite,
        "labels_valid": labels_valid,
        "completed_feature_bank_valid": feature_valid,
        "feature_contract_valid": len(FEATURE_IDS)
        == len(EXISTING_FEATURE_IDS) + len(GENERATED_FEATURE_IDS),
        "output_absent": not config.output_dir.exists(),
        "partial_output_absent": not Path(
            str(config.output_dir) + ".partial"
        ).exists(),
        "preflight_separate_from_output": config.preflight_dir
        != config.output_dir,
        "ram_cap_sufficient": estimated_ram
        <= int(config.resources["max_ram_mib"]),
        "available_ram_sufficient": estimated_ram <= _available_ram_mib(),
        "disk_headroom_sufficient": free_disk
        >= int(config.resources["min_free_disk_mib"]) + estimated_output,
        "output_cap_sufficient": estimated_output
        <= int(config.resources["max_output_mib"]),
    }
    payload = {
        "schema_version": 1,
        "kind": "read_only_spon_ca_nested_innovation_ranker_preflight",
        "experiment_id": config.experiment_id,
        "ready": all(gates.values()),
        "gates": gates,
        "source_shape": source_shape,
        "source_dtype": source_dtype,
        "label_rows": len(labels),
        "roi_identities": len({row["roi_identity"] for row in labels}),
        "design": {
            "existing_feature_count": len(EXISTING_FEATURE_IDS),
            "generated_feature_count": len(GENERATED_FEATURE_IDS),
            "total_feature_count": len(FEATURE_IDS),
            "proposal_source_count": len(PROPOSAL_SOURCE_IDS),
            "feature_set_count": len(FEATURE_SETS),
            "linear_config_count": config.linear_config_count,
            "mlp_config_count": config.mlp_config_count,
            "inner_fit_count": config.inner_fit_count,
            "outer_refit_count": config.outer_refit_count,
            "total_model_fit_count": config.inner_fit_count
            + config.outer_refit_count,
            "outer_folds": 4,
            "inner_folds_per_outer": 3,
        },
        "resources": {
            "estimated_peak_ram_mib": estimated_ram,
            "available_ram_mib": _available_ram_mib(),
            "estimated_output_mib": estimated_output,
            "free_disk_mib": free_disk,
            **config.resources,
        },
        "inputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in inputs
            if path.is_file()
        ],
        "system_snapshot": _snapshots(),
        "scientific_contract": (
            "Outer bursts are untouched during hyperparameter selection. "
            "Unmatched event candidates remain unknown. Fine-tuning begins "
            "from an exact carrier skip and uses known positives versus quiet "
            "hard negatives only."
        ),
    }
    if write_artifacts:
        config.preflight_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(config.preflight_dir / "preflight.json", payload)
        _atomic_json(
            config.preflight_dir / "config.resolved.json", config.to_dict()
        )
        if not missing and bounds_valid:
            label_core._write_overlay(
                np.load(config.source_video, mmap_mode="r", allow_pickle=False),
                labels,
                config.preflight_dir / "label_projection_overlay.png",
            )
    if not payload["ready"]:
        raise RuntimeError(f"innovation ranker preflight failed: {payload}")
    return payload


def _matching_preflight(config: InnovationRankerConfig) -> dict[str, Any]:
    audit = json.loads(
        (config.preflight_dir / "preflight.json").read_text(encoding="utf-8")
    )
    resolved = json.loads(
        (config.preflight_dir / "config.resolved.json").read_text(
            encoding="utf-8"
        )
    )
    if not audit.get("ready") or resolved != config.to_dict():
        raise RuntimeError("run requires a matching ready preflight")
    if config.output_dir.exists() or Path(
        str(config.output_dir) + ".partial"
    ).exists():
        raise FileExistsError("completed or partial output already exists")
    return audit


def _copy_maps(maps: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "quiet": [np.asarray(value, dtype=np.float32) for value in maps["quiet"]],
        "events": {
            int(key): np.asarray(value, dtype=np.float32)
            for key, value in maps["events"].items()
        },
    }


def _all_map_values(
    maps: Mapping[str, Any],
) -> list[tuple[str | int, int | None, np.ndarray]]:
    return [
        *[("quiet", index, value) for index, value in enumerate(maps["quiet"])],
        *[("event", int(burst), value) for burst, value in maps["events"].items()],
    ]


def _map_from_values(
    values: list[tuple[str | int, int | None, np.ndarray]],
) -> dict[str, Any]:
    quiet = [value for kind, _, value in values if kind == "quiet"]
    events = {
        int(key): value for kind, key, value in values if kind == "event"
    }
    return {"quiet": quiet, "events": events}


def _generate_maps(
    config: InnovationRankerConfig,
    labels: list[dict[str, Any]],
    base_config: FeatureUtilityConfig,
    progress: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    feature_maps: dict[str, dict[str, Any]] = {}
    for feature_id in EXISTING_FEATURE_IDS:
        values = np.load(
            config.feature_root / "features" / f"{feature_id}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        feature_maps[feature_id] = _pooled_maps(
            values, int(config.frames["quiet_count"]), labels, base_config
        )
        _progress(progress, "existing_feature_pooled", feature_id=feature_id)
        del values

    carrier = np.load(
        config.feature_root / "features" / "carrier_signed.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    clip = float(config.map_generation["normalization_clip"])
    for power in config.map_generation["signed_powers"]:
        feature_id = f"signed_power_{_token(power)}"
        bounded = np.clip(np.asarray(carrier, dtype=np.float32), -clip, clip)
        transformed = np.sign(bounded) * np.power(
            np.abs(bounded), float(power)
        )
        feature_maps[feature_id] = _pooled_maps(
            transformed,
            int(config.frames["quiet_count"]),
            labels,
            base_config,
        )
        del bounded, transformed
        _progress(progress, "generated_feature_pooled", feature_id=feature_id)

    structure = np.load(
        config.feature_root / "structure_unit.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    quiet_variance = np.var(
        np.asarray(carrier[: int(config.frames["quiet_count"])], dtype=np.float32),
        axis=0,
        ddof=1,
    )
    x = np.asarray(structure, dtype=np.float32).ravel()[::8]
    y = quiet_variance.ravel()[::8]
    edges = np.quantile(x, np.linspace(0, 1, 33))
    bx, by = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        selected = (x >= left) & (x <= right)
        if np.count_nonzero(selected) >= 32:
            bx.append(float(np.median(x[selected])))
            by.append(float(np.median(y[selected])))
    slope, intercept = np.polyfit(bx, by, 1)
    slope = max(float(slope), 0.0)
    intercept = max(float(intercept), 1e-6)
    predicted = slope * np.asarray(structure, dtype=np.float32) + intercept
    predicted /= max(float(np.median(predicted)), 1e-6)
    gain = np.clip(np.sqrt(predicted), 0.5, 2.0).astype(np.float32)
    calibration = {
        "quiet_variance_slope": slope,
        "quiet_variance_intercept": intercept,
        "binned_structure": bx,
        "binned_quiet_variance": by,
        "gain_minimum": float(gain.min()),
        "gain_median": float(np.median(gain)),
        "gain_maximum": float(gain.max()),
        "interpretation": (
            "Residual heteroscedasticity fitted after the accepted per-pixel "
            "quiet standardization; a near-zero slope means extra whitening "
            "is unnecessary."
        ),
    }
    for rho in config.map_generation["whitening_rhos"]:
        feature_id = f"shot_whitened_rho{_token(rho)}"
        blended_gain = (1.0 - float(rho)) + float(rho) * gain
        transformed = np.asarray(carrier, dtype=np.float32) / blended_gain[None]
        feature_maps[feature_id] = _pooled_maps(
            transformed,
            int(config.frames["quiet_count"]),
            labels,
            base_config,
        )
        del transformed
        _progress(progress, "generated_feature_pooled", feature_id=feature_id)

    positive = feature_maps["derivative_positive_lag1"]
    negative = feature_maps["derivative_negative_lag1"]
    onset_rows, energy_rows = [], []
    for (kind, key, pos), (_, _, neg) in zip(
        _all_map_values(positive), _all_map_values(negative)
    ):
        denominator = np.maximum(pos + neg, 1e-6)
        onset_rows.append((kind, key, (pos / denominator).astype(np.float32)))
        energy_rows.append((kind, key, (pos + neg).astype(np.float32)))
    feature_maps["onset_dominance"] = _map_from_values(onset_rows)
    feature_maps["derivative_energy"] = _map_from_values(energy_rows)

    artifact = feature_maps["persistent_artifact_score"]
    artifact_normalizer = robust_map_normalizer(artifact["quiet"])
    for authority in config.map_generation["artifact_authorities"]:
        feature_id = f"artifact_attenuated_{_token(authority)}"
        rows = []
        for (kind, key, raw), (_, _, context) in zip(
            _all_map_values(feature_maps["carrier_signed"]),
            _all_map_values(artifact),
        ):
            unit = normalize_map(context, artifact_normalizer, clip=1.0)
            floor = 1.0 - float(authority) * unit
            rows.append((kind, key, (raw * floor).astype(np.float32)))
        feature_maps[feature_id] = _map_from_values(rows)

    morphology: dict[str, list[tuple[str | int, int | None, np.ndarray]]] = {}
    for kind, key, score in _all_map_values(feature_maps["carrier_signed"]):
        basis = cut_morphology_basis(
            score,
            center_sigmas_px=config.map_generation["center_sigmas_px"],
            ring_specs=config.map_generation["ring_specs"],
            crowd_sigma_px=float(config.map_generation["crowd_sigma_px"]),
        )
        for feature_id, value in basis.items():
            morphology.setdefault(feature_id, []).append((kind, key, value))
    for feature_id, rows in morphology.items():
        feature_maps[feature_id] = _map_from_values(rows)
    structure_map = np.asarray(structure, dtype=np.float32)
    feature_maps["structure_context"] = {
        "quiet": [structure_map] * 4,
        "events": {burst: structure_map for burst in BURSTS},
    }
    if set(feature_maps) != set(FEATURE_IDS):
        raise RuntimeError(
            "feature map mismatch; missing="
            f"{sorted(set(FEATURE_IDS)-set(feature_maps))}; extra="
            f"{sorted(set(feature_maps)-set(FEATURE_IDS))}"
        )
    return feature_maps, calibration


def _candidate_tables(
    feature_maps: Mapping[str, Mapping[str, Any]],
    config: InnovationRankerConfig,
) -> tuple[
    dict[str, tuple[float, float]],
    list[CandidateTable],
    dict[int, CandidateTable],
]:
    normalizers = {
        feature_id: robust_map_normalizer(feature_maps[feature_id]["quiet"])
        for feature_id in FEATURE_IDS
    }
    quiet_tables = []
    for index in range(4):
        source_maps = {
            feature_id: feature_maps[feature_id]["quiet"][index]
            for feature_id in PROPOSAL_SOURCE_IDS
        }
        positions, source_count = merge_peak_proposals(
            source_maps,
            normalizers,
            nms_distance_px=int(config.proposals["nms_distance_px"]),
            per_source_limit=int(config.proposals["per_source_limit"]),
            dedupe_radius_px=float(config.proposals["dedupe_radius_px"]),
            clip=float(config.map_generation["normalization_clip"]),
        )
        features = sample_candidate_features(
            positions,
            {
                feature_id: feature_maps[feature_id]["quiet"][index]
                for feature_id in FEATURE_IDS
            },
            FEATURE_IDS,
            normalizers,
            clip=float(config.map_generation["normalization_clip"]),
        )
        quiet_tables.append(CandidateTable(positions, features, source_count))
    event_tables = {}
    for burst in BURSTS:
        source_maps = {
            feature_id: feature_maps[feature_id]["events"][burst]
            for feature_id in PROPOSAL_SOURCE_IDS
        }
        positions, source_count = merge_peak_proposals(
            source_maps,
            normalizers,
            nms_distance_px=int(config.proposals["nms_distance_px"]),
            per_source_limit=int(config.proposals["per_source_limit"]),
            dedupe_radius_px=float(config.proposals["dedupe_radius_px"]),
            clip=float(config.map_generation["normalization_clip"]),
        )
        features = sample_candidate_features(
            positions,
            {
                feature_id: feature_maps[feature_id]["events"][burst]
                for feature_id in FEATURE_IDS
            },
            FEATURE_IDS,
            normalizers,
            clip=float(config.map_generation["normalization_clip"]),
        )
        event_tables[burst] = CandidateTable(
            positions, features, source_count
        )
    return normalizers, quiet_tables, event_tables


def _quiet_threshold(
    quiet_scores: Sequence[np.ndarray], allowed_per_map: float
) -> float:
    ranked = sorted(
        (float(value) for score in quiet_scores for value in score),
        reverse=True,
    )
    allowed = max(1, int(round(float(allowed_per_map) * len(quiet_scores))))
    if len(ranked) <= allowed:
        raise RuntimeError("too few quiet proposal scores")
    return float(np.nextafter(ranked[allowed], np.inf))


def _peaks(
    scores: np.ndarray,
    table: CandidateTable,
    indices: np.ndarray,
) -> list[tuple[float, int, int]]:
    return [
        (
            float(scores[index]),
            int(table.positions[index, 0]),
            int(table.positions[index, 1]),
        )
        for index in indices
    ]


def _evaluate_candidate_scores(
    scores: np.ndarray,
    table: CandidateTable,
    rows: list[dict[str, Any]],
    threshold: float,
    config: InnovationRankerConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1]
    fixed_count = int(config.evaluation["fixed_candidates_per_burst"])
    fixed_indices = order[:fixed_count]
    threshold_indices = order[np.asarray(scores)[order] >= float(threshold)]
    fixed_peaks = _peaks(scores, table, fixed_indices)
    threshold_peaks = _peaks(scores, table, threshold_indices)
    fixed_matches, fixed_matched_indices = match_peaks_one_to_one(
        fixed_peaks, rows, float(config.evaluation["match_radius_px"])
    )
    threshold_matches, threshold_matched_indices = match_peaks_one_to_one(
        threshold_peaks, rows, float(config.evaluation["match_radius_px"])
    )
    metrics = {
        "labels": len(rows),
        "fixed_matched": len(fixed_matches),
        "fixed_recall": len(fixed_matches) / len(rows),
        "fixed_candidates": len(fixed_peaks),
        "matched": len(threshold_matches),
        "recall": len(threshold_matches) / len(rows),
        "candidates": len(threshold_peaks),
        "threshold": float(threshold),
    }
    detail = {
        "fixed_order_indices": fixed_indices.tolist(),
        "threshold_order_indices": threshold_indices.tolist(),
        "fixed_label_indices": sorted(match[0] for match in fixed_matches),
        "threshold_label_indices": sorted(
            match[0] for match in threshold_matches
        ),
        "fixed_matched_candidate_indices": sorted(
            int(fixed_indices[index]) for index in fixed_matched_indices
        ),
        "threshold_matched_candidate_indices": sorted(
            int(threshold_indices[index])
            for index in threshold_matched_indices
        ),
    }
    return metrics, detail


def _evaluate_candidate_budget(
    scores: np.ndarray,
    table: CandidateTable,
    rows: list[dict[str, Any]],
    budget: int,
    match_radius_px: float,
) -> dict[str, Any]:
    order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1][
        : int(budget)
    ]
    peaks = _peaks(scores, table, order)
    matches, matched_indices = match_peaks_one_to_one(
        peaks, rows, float(match_radius_px)
    )
    return {
        "budget": int(budget),
        "labels": len(rows),
        "candidates": len(peaks),
        "matched": len(matches),
        "recall": len(matches) / len(rows),
        "label_indices": sorted(int(match[0]) for match in matches),
        "matched_candidate_indices": sorted(
            int(order[index]) for index in matched_indices
        ),
    }


def _label_rows(
    labels: list[dict[str, Any]], burst: int
) -> list[dict[str, Any]]:
    return [row for row in labels if int(row["burst_id"]) == int(burst)]


def _positive_examples(
    table: CandidateTable,
    rows: list[dict[str, Any]],
    feature_columns: Sequence[int],
    radius: float,
) -> tuple[np.ndarray, set[int]]:
    selected = []
    covered = set()
    columns = np.asarray(feature_columns, dtype=np.int64)
    for label_index, row in enumerate(rows):
        distance_sq = (
            (table.positions[:, 0] - float(row["x_px"])) ** 2
            + (table.positions[:, 1] - float(row["y_px"])) ** 2
        )
        choices = np.flatnonzero(distance_sq <= float(radius) ** 2)
        if choices.size:
            local_score = np.max(table.features[choices][:, columns], axis=1)
            selected.append(int(choices[int(np.argmax(local_score))]))
            covered.add(label_index)
    if not selected:
        raise RuntimeError("no positive proposals available")
    return table.features[np.asarray(selected)], covered


def _training_arrays(
    bursts: Sequence[int],
    event_tables: Mapping[int, CandidateTable],
    quiet_tables: Sequence[CandidateTable],
    labels: list[dict[str, Any]],
    feature_columns: Sequence[int],
    config: InnovationRankerConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    positives = []
    covered = {}
    for burst in bursts:
        values, indices = _positive_examples(
            event_tables[int(burst)],
            _label_rows(labels, int(burst)),
            feature_columns,
            float(config.evaluation["match_radius_px"]),
        )
        positives.append(values)
        covered[int(burst)] = len(indices)
    negative = np.concatenate([table.features for table in quiet_tables])
    separation_columns = np.asarray(
        [FEATURE_IDS.index(value) for value in FEATURE_SETS["separation"]]
    )
    hardness = np.max(negative[:, separation_columns], axis=1)
    hard_count = min(512, len(negative))
    hard_indices = np.argpartition(hardness, -hard_count)[-hard_count:]
    return (
        np.concatenate(positives),
        negative[hard_indices],
        {"covered_positive_labels": covered, "quiet_hard_negatives": hard_count},
    )


def _feature_columns(feature_set: str) -> list[int]:
    return [FEATURE_IDS.index(value) for value in FEATURE_SETS[feature_set]]


def _linear_model(
    positive: np.ndarray,
    negative: np.ndarray,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    columns = _feature_columns(str(row["feature_set"]))
    carrier = FEATURE_IDS.index("carrier_signed")
    auxiliary = [column for column in columns if column != carrier]
    directions = [
        -1.0 if FEATURE_IDS[column] in NEGATIVE_EVIDENCE_IDS else 1.0
        for column in auxiliary
    ]
    model = fit_bounded_pairwise_linear(
        positive,
        negative,
        carrier_column=carrier,
        auxiliary_columns=auxiliary,
        auxiliary_directions=directions,
        learning_rate=float(row["learning_rate"]),
        epochs=int(row["epochs"]),
        l2=float(row["l2"]),
        maximum_total=float(row["maximum_total_weight"]),
    )
    model["feature_set"] = row["feature_set"]
    return model


def _mlp_model(
    positive: np.ndarray,
    negative: np.ndarray,
    row: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    model = fit_residual_mlp_ranker(
        positive,
        negative,
        carrier_column=FEATURE_IDS.index("carrier_signed"),
        input_columns=_feature_columns(str(row["feature_set"])),
        hidden_units=int(row["hidden_units"]),
        maximum_residual=float(row["maximum_residual"]),
        learning_rate=float(row["learning_rate"]),
        epochs=int(row["epochs"]),
        weight_decay=float(row["weight_decay"]),
        seed=int(seed),
    )
    model["feature_set"] = row["feature_set"]
    return model


def _linear_grid(config: InnovationRankerConfig) -> list[dict[str, Any]]:
    rows = []
    for index, values in enumerate(
        itertools.product(
            config.linear_grid["feature_sets"],
            config.linear_grid["learning_rates"],
            config.linear_grid["l2_values"],
            config.linear_grid["maximum_total_weights"],
        )
    ):
        feature_set, learning_rate, l2, maximum = values
        rows.append(
            {
                "config_id": f"linear__{index:03d}",
                "family": "linear",
                "feature_set": feature_set,
                "learning_rate": float(learning_rate),
                "l2": float(l2),
                "maximum_total_weight": float(maximum),
                "epochs": int(config.linear_grid["epochs"]),
            }
        )
    return rows


def _mlp_grid(config: InnovationRankerConfig) -> list[dict[str, Any]]:
    rows = []
    for index, values in enumerate(
        itertools.product(
            config.mlp_grid["feature_sets"],
            config.mlp_grid["learning_rates"],
            config.mlp_grid["weight_decays"],
            config.mlp_grid["hidden_units"],
            config.mlp_grid["maximum_residuals"],
        )
    ):
        feature_set, learning_rate, decay, hidden, maximum = values
        rows.append(
            {
                "config_id": f"mlp__{index:03d}",
                "family": "mlp",
                "feature_set": feature_set,
                "learning_rate": float(learning_rate),
                "weight_decay": float(decay),
                "hidden_units": int(hidden),
                "maximum_residual": float(maximum),
                "epochs": int(config.mlp_grid["epochs"]),
            }
        )
    return rows


def _score_model(
    table: CandidateTable,
    model: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> np.ndarray:
    if isinstance(model, Sequence) and not isinstance(model, Mapping):
        return np.mean([_score_model(table, value) for value in model], axis=0)
    assert isinstance(model, Mapping)
    if model["kind"] == "bounded_linear":
        return score_bounded_pairwise_linear(table.features, model)
    if model["kind"] == "residual_mlp":
        return score_residual_mlp_ranker(table.features, model)
    raise ValueError(f"unknown model kind {model['kind']}")


def _evaluate_model(
    model: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    validation_burst: int,
    event_tables: Mapping[int, CandidateTable],
    quiet_tables: Sequence[CandidateTable],
    labels: list[dict[str, Any]],
    config: InnovationRankerConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    quiet_scores = [_score_model(table, model) for table in quiet_tables]
    threshold = _quiet_threshold(
        quiet_scores,
        float(config.evaluation["quiet_false_candidates_per_map"]),
    )
    return _evaluate_candidate_scores(
        _score_model(event_tables[int(validation_burst)], model),
        event_tables[int(validation_burst)],
        _label_rows(labels, int(validation_burst)),
        threshold,
        config,
    )


def _screen_features(
    quiet_tables: Sequence[CandidateTable],
    event_tables: Mapping[int, CandidateTable],
    labels: list[dict[str, Any]],
    config: InnovationRankerConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    details = []
    for column, feature_id in enumerate(FEATURE_IDS):
        quiet_scores = [table.features[:, column] for table in quiet_tables]
        threshold = _quiet_threshold(
            quiet_scores,
            float(config.evaluation["quiet_false_candidates_per_map"]),
        )
        folds = []
        for burst in BURSTS:
            metrics, detail = _evaluate_candidate_scores(
                event_tables[burst].features[:, column],
                event_tables[burst],
                _label_rows(labels, burst),
                threshold,
                config,
            )
            folds.append({"burst_id": burst, **metrics})
            details.append(
                {
                    "feature_id": feature_id,
                    "burst_id": burst,
                    **detail,
                }
            )
        rows.append(
            {
                "feature_id": feature_id,
                "mean_recall": float(np.mean([fold["recall"] for fold in folds])),
                "fixed_budget_mean_recall": float(
                    np.mean([fold["fixed_recall"] for fold in folds])
                ),
                "event_candidates": sum(fold["candidates"] for fold in folds),
                "folds": folds,
            }
        )
    return rows, details


def _nested_feature_selection(
    screening: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    folds = []
    for held_out in BURSTS:
        def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
            training = [
                fold for fold in row["folds"]
                if int(fold["burst_id"]) != held_out
            ]
            return (
                float(np.mean([fold["fixed_recall"] for fold in training])),
                float(np.min([fold["fixed_recall"] for fold in training])),
                float(np.mean([fold["recall"] for fold in training])),
                -sum(fold["candidates"] for fold in training),
                str(row["feature_id"]),
            )

        selected = max(screening, key=key)
        held = next(
            fold for fold in selected["folds"]
            if int(fold["burst_id"]) == held_out
        )
        folds.append(
            {
                "held_out_burst": held_out,
                "selected_feature_id": selected["feature_id"],
                **held,
            }
        )
    return {
        "folds": folds,
        "mean_recall": float(np.mean([row["recall"] for row in folds])),
        "fixed_budget_mean_recall": float(
            np.mean([row["fixed_recall"] for row in folds])
        ),
        "event_candidates": sum(row["candidates"] for row in folds),
    }


def _oracle_coverage(
    feature_maps: Mapping[str, Mapping[str, Any]],
    normalizers: Mapping[str, tuple[float, float]],
    labels: list[dict[str, Any]],
    config: InnovationRankerConfig,
) -> tuple[list[dict[str, Any]], dict[int, set[int]]]:
    rows = []
    recovery_at_fixed: dict[int, set[int]] = {}
    fixed_budget = int(config.evaluation["fixed_candidates_per_burst"])
    for budget in config.evaluation["oracle_source_budgets"]:
        fold_coverages = []
        for burst in BURSTS:
            burst_labels = _label_rows(labels, burst)
            recovered: set[int] = set()
            source_counts = {}
            for feature_id in PROPOSAL_SOURCE_IDS:
                normalized = normalize_map(
                    feature_maps[feature_id]["events"][burst],
                    normalizers[feature_id],
                    clip=float(config.map_generation["normalization_clip"]),
                )
                peaks = extract_local_maxima(
                    normalized,
                    int(config.proposals["nms_distance_px"]),
                    limit=int(budget),
                )[: int(budget)]
                matches = match_peaks_one_to_one(
                    peaks,
                    burst_labels,
                    float(config.evaluation["match_radius_px"]),
                )[0]
                indices = {int(match[0]) for match in matches}
                recovered |= indices
                source_counts[feature_id] = len(indices)
            if int(budget) == fixed_budget:
                recovery_at_fixed[burst] = recovered
            fold_coverages.append(len(recovered) / len(burst_labels))
            rows.append(
                {
                    "source_budget": int(budget),
                    "burst_id": burst,
                    "labels": len(burst_labels),
                    "union_recovered": len(recovered),
                    "union_coverage": len(recovered) / len(burst_labels),
                    "best_source": max(
                        source_counts, key=lambda key: source_counts[key]
                    ),
                    "best_source_recovered": max(source_counts.values()),
                    "source_counts": source_counts,
                }
            )
    return rows, recovery_at_fixed


def _budget_curves(
    feature_maps: Mapping[str, Mapping[str, Any]],
    normalizers: Mapping[str, tuple[float, float]],
    event_tables: Mapping[int, CandidateTable],
    nested_feature: Mapping[str, Any],
    nested_rankers: Mapping[str, Any],
    model_details: Mapping[str, Any],
    labels: list[dict[str, Any]],
    config: InnovationRankerConfig,
) -> dict[str, Any]:
    budgets = sorted(
        {
            *[int(value) for value in config.evaluation["oracle_source_budgets"]],
            int(config.evaluation["fixed_candidates_per_burst"]),
        }
    )
    carrier_column = FEATURE_IDS.index("carrier_signed")
    methods: dict[str, list[dict[str, Any]]] = {
        "standardized_carrier_native": [],
        "union_carrier_score": [],
        "nested_single_feature": [],
        "nested_linear": [],
        "nested_mlp": [],
    }
    for budget in budgets:
        for burst in BURSTS:
            burst_labels = _label_rows(labels, burst)
            native_peaks = extract_local_maxima(
                feature_maps["carrier_signed"]["events"][burst],
                int(config.proposals["nms_distance_px"]),
                limit=int(budget),
            )[: int(budget)]
            native_matches = match_peaks_one_to_one(
                native_peaks,
                burst_labels,
                float(config.evaluation["match_radius_px"]),
            )[0]
            methods["standardized_carrier_native"].append(
                {
                    "budget": int(budget),
                    "burst_id": burst,
                    "labels": len(burst_labels),
                    "candidates": len(native_peaks),
                    "matched": len(native_matches),
                    "recall": len(native_matches) / len(burst_labels),
                    "label_indices": sorted(
                        int(match[0]) for match in native_matches
                    ),
                }
            )
            methods["union_carrier_score"].append(
                {
                    "burst_id": burst,
                    **_evaluate_candidate_budget(
                        event_tables[burst].features[:, carrier_column],
                        event_tables[burst],
                        burst_labels,
                        budget,
                        float(config.evaluation["match_radius_px"]),
                    ),
                }
            )
            selected_feature = next(
                row
                for row in nested_feature["folds"]
                if int(row["held_out_burst"]) == burst
            )["selected_feature_id"]
            methods["nested_single_feature"].append(
                {
                    "burst_id": burst,
                    "selected_feature_id": selected_feature,
                    **_evaluate_candidate_budget(
                        event_tables[burst].features[
                            :, FEATURE_IDS.index(str(selected_feature))
                        ],
                        event_tables[burst],
                        burst_labels,
                        budget,
                        float(config.evaluation["match_radius_px"]),
                    ),
                }
            )
            for family in ("linear", "mlp"):
                model = model_details[f"{family}__burst{burst}"]
                selected = next(
                    row
                    for row in nested_rankers[family]["folds"]
                    if int(row["burst_id"]) == burst
                )
                methods[f"nested_{family}"].append(
                    {
                        "burst_id": burst,
                        "selected_config_id": selected["selected_config_id"],
                        "selected_feature_set": selected[
                            "selected_feature_set"
                        ],
                        **_evaluate_candidate_budget(
                            _score_model(event_tables[burst], model),
                            event_tables[burst],
                            burst_labels,
                            budget,
                            float(config.evaluation["match_radius_px"]),
                        ),
                    }
                )
    summaries = {}
    for method, rows in methods.items():
        summaries[method] = [
            {
                "budget": budget,
                "mean_recall": float(
                    np.mean(
                        [
                            row["recall"]
                            for row in rows
                            if int(row["budget"]) == budget
                        ]
                    )
                ),
                "minimum_burst_recall": float(
                    np.min(
                        [
                            row["recall"]
                            for row in rows
                            if int(row["budget"]) == budget
                        ]
                    )
                ),
                "pooled_matched": int(
                    sum(
                        row["matched"]
                        for row in rows
                        if int(row["budget"]) == budget
                    )
                ),
                "pooled_labels": int(
                    sum(
                        row["labels"]
                        for row in rows
                        if int(row["budget"]) == budget
                    )
                ),
            }
            for budget in budgets
        ]
    return {
        "budgets": budgets,
        "methods": methods,
        "summaries": summaries,
        "interpretation": (
            "standardized_carrier_native evaluates the stored per-pixel "
            "quiet-standardized carrier on its own raw pooled map. The "
            "archived feature-bank baseline instead used the centered, "
            "unscaled residual and is retained separately. "
            "union_carrier_score holds the positive-evidence proposal union "
            "fixed and therefore isolates the gain from proposal generation. "
            "Learned and selected-feature rows use strictly nested outer-"
            "burst evaluation."
        ),
    }


def _inner_fine_tuning(
    config: InnovationRankerConfig,
    event_tables: Mapping[int, CandidateTable],
    quiet_tables: Sequence[CandidateTable],
    labels: list[dict[str, Any]],
    progress: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    linear_grid = _linear_grid(config)
    mlp_grid = _mlp_grid(config)
    registry = {
        row["config_id"]: row for row in [*linear_grid, *mlp_grid]
    }
    rows = []
    training_pairs = list(itertools.combinations(BURSTS, 2))
    for pair_index, pair in enumerate(training_pairs, start=1):
        validations = [burst for burst in BURSTS if burst not in pair]
        for grid_row in linear_grid:
            columns = _feature_columns(str(grid_row["feature_set"]))
            positive, negative, training = _training_arrays(
                pair,
                event_tables,
                quiet_tables,
                labels,
                columns,
                config,
            )
            model = _linear_model(positive, negative, grid_row)
            for validation in validations:
                metrics, _ = _evaluate_model(
                    model,
                    validation,
                    event_tables,
                    quiet_tables,
                    labels,
                    config,
                )
                rows.append(
                    {
                        **grid_row,
                        "training_bursts": list(pair),
                        "validation_burst": validation,
                        **training,
                        **metrics,
                        "loss_initial": model["loss_initial"],
                        "loss_final": model["loss_final"],
                    }
                )
        _progress(
            progress,
            "linear_training_pair_complete",
            pair_index=pair_index,
            pair_total=len(training_pairs),
            training_bursts=list(pair),
        )
        for grid_row in mlp_grid:
            columns = _feature_columns(str(grid_row["feature_set"]))
            positive, negative, training = _training_arrays(
                pair,
                event_tables,
                quiet_tables,
                labels,
                columns,
                config,
            )
            models = [
                _mlp_model(positive, negative, grid_row, int(seed))
                for seed in config.mlp_grid["inner_seeds"]
            ]
            for validation in validations:
                metrics, _ = _evaluate_model(
                    models,
                    validation,
                    event_tables,
                    quiet_tables,
                    labels,
                    config,
                )
                rows.append(
                    {
                        **grid_row,
                        "training_bursts": list(pair),
                        "validation_burst": validation,
                        **training,
                        **metrics,
                        "seed_count": len(models),
                        "loss_initial": float(
                            np.mean([model["loss_initial"] for model in models])
                        ),
                        "loss_final": float(
                            np.mean([model["loss_final"] for model in models])
                        ),
                    }
                )
        _progress(
            progress,
            "mlp_training_pair_complete",
            pair_index=pair_index,
            pair_total=len(training_pairs),
            training_bursts=list(pair),
        )
    return rows, registry


def _select_inner_config(
    rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Mapping[str, Any]],
    outer_burst: int,
    family: str,
    config: InnovationRankerConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    for config_id, definition in registry.items():
        if definition["family"] != family:
            continue
        relevant = [
            row
            for row in rows
            if row["config_id"] == config_id
            and int(row["validation_burst"]) != int(outer_burst)
            and int(outer_burst) not in row["training_bursts"]
        ]
        if len(relevant) != 3:
            raise RuntimeError("nested inner-fold count differs from three")
        fixed = np.asarray([row["fixed_recall"] for row in relevant])
        recalls = np.asarray([row["recall"] for row in relevant])
        candidates_count = np.asarray([row["candidates"] for row in relevant])
        score = (
            float(config.selection["mean_weight"]) * float(np.mean(fixed))
            + float(config.selection["minimum_weight"]) * float(np.min(fixed))
            + float(config.selection["threshold_weight"])
            * float(np.mean(recalls))
            - float(config.selection["candidate_penalty"])
            * float(np.mean(candidates_count))
            / float(config.evaluation["fixed_candidates_per_burst"])
        )
        candidates.append(
            {
                **definition,
                "outer_burst": int(outer_burst),
                "inner_selection_score": score,
                "inner_mean_fixed_recall": float(np.mean(fixed)),
                "inner_minimum_fixed_recall": float(np.min(fixed)),
                "inner_std_fixed_recall": float(np.std(fixed)),
                "inner_mean_recall": float(np.mean(recalls)),
                "inner_mean_candidates": float(np.mean(candidates_count)),
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            row["inner_selection_score"],
            row["inner_minimum_fixed_recall"],
            row["inner_mean_fixed_recall"],
            -row["inner_std_fixed_recall"],
            -row["inner_mean_candidates"],
            row["config_id"],
        ),
    )
    return selected, candidates


def _outer_folds(
    inner_rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Mapping[str, Any]],
    event_tables: Mapping[int, CandidateTable],
    quiet_tables: Sequence[CandidateTable],
    labels: list[dict[str, Any]],
    config: InnovationRankerConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    family_payloads = {}
    model_details = {}
    for family in ("linear", "mlp"):
        folds = []
        selections = []
        for held_out in BURSTS:
            selected, candidates = _select_inner_config(
                inner_rows, registry, held_out, family, config
            )
            selections.append(
                {
                    "outer_burst": held_out,
                    "selected": selected,
                    "candidate_count": len(candidates),
                    "top_five": sorted(
                        candidates,
                        key=lambda row: row["inner_selection_score"],
                        reverse=True,
                    )[:5],
                }
            )
            training_bursts = [
                burst for burst in BURSTS if burst != held_out
            ]
            columns = _feature_columns(str(selected["feature_set"]))
            positive, negative, training = _training_arrays(
                training_bursts,
                event_tables,
                quiet_tables,
                labels,
                columns,
                config,
            )
            if family == "linear":
                model: Mapping[str, Any] | Sequence[Mapping[str, Any]] = (
                    _linear_model(positive, negative, selected)
                )
                saved_model: Any = model
            else:
                models = [
                    _mlp_model(positive, negative, selected, int(seed))
                    for seed in config.mlp_grid["confirmation_seeds"]
                ]
                model = models
                saved_model = models
            metrics, detail = _evaluate_model(
                model,
                held_out,
                event_tables,
                quiet_tables,
                labels,
                config,
            )
            folds.append(
                {
                    "burst_id": held_out,
                    "training_bursts": training_bursts,
                    "selected_config_id": selected["config_id"],
                    "selected_feature_set": selected["feature_set"],
                    **training,
                    **metrics,
                    "detail": detail,
                }
            )
            model_details[f"{family}__burst{held_out}"] = saved_model
        family_payloads[family] = {
            "folds": folds,
            "selections": selections,
            "mean_recall": float(np.mean([row["recall"] for row in folds])),
            "fixed_budget_mean_recall": float(
                np.mean([row["fixed_recall"] for row in folds])
            ),
            "event_candidates": sum(row["candidates"] for row in folds),
            "minimum_fixed_recall": min(
                row["fixed_recall"] for row in folds
            ),
        }
    return family_payloads, model_details


def _score_map(
    scores: np.ndarray,
    table: CandidateTable,
    shape: tuple[int, int],
    sigma: float,
) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    result = np.zeros(shape, dtype=np.float32)
    for score, (x, y) in zip(scores, table.positions):
        result[int(y), int(x)] = max(result[int(y), int(x)], float(score))
    result -= float(np.min(result))
    return gaussian_filter(result, sigma=float(sigma), mode="reflect")


def _write_score_tiff(
    path: Path,
    maps: Sequence[np.ndarray],
    *,
    compression: str,
    description: Mapping[str, Any],
) -> dict[str, Any]:
    maximum = max(
        float(np.percentile(np.concatenate([value.ravel() for value in maps]), 99.9)),
        1e-6,
    )
    temporary = path.with_name(path.name + ".partial")
    with tifffile.TiffWriter(temporary) as writer:
        for index, value in enumerate(maps):
            page = np.rint(np.clip(value / maximum, 0, 1) * 65535).astype(
                np.uint16
            )
            writer.write(
                page,
                photometric="minisblack",
                compression=compression,
                metadata=None,
                description=json.dumps(description, sort_keys=True)
                if index == 0
                else None,
            )
    temporary.replace(path)
    with tifffile.TiffFile(path) as tiff:
        if len(tiff.pages) != len(maps):
            raise RuntimeError("score TIFF page verification failed")
    return {
        "path": f"diagnostics/{path.name}",
        "pages": len(maps),
        "shape": list(maps[0].shape),
        "bytes": path.stat().st_size,
    }


def _overlay_page(
    structure: np.ndarray,
    table: CandidateTable,
    scores: np.ndarray,
    detail: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    limit: int,
) -> np.ndarray:
    base = np.clip(np.asarray(structure, dtype=np.float32), 0, 1)
    rgb = np.repeat(np.rint(base[..., None] * 255), 3, axis=2).astype(np.uint8)

    def mark(x: int, y: int, color: tuple[int, int, int], radius: int) -> None:
        y0, y1 = max(0, y - radius), min(rgb.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(rgb.shape[1], x + radius + 1)
        rgb[y0:y1, x, :] = color
        rgb[y, x0:x1, :] = color

    order = np.argsort(scores)[::-1][: int(limit)]
    matched = set(detail["fixed_matched_candidate_indices"])
    fixed = set(detail["fixed_order_indices"])
    for index in order:
        if int(index) not in fixed:
            continue
        x, y = table.positions[int(index)]
        mark(
            int(x),
            int(y),
            (0, 255, 0) if int(index) in matched else (0, 220, 255),
            3,
        )
    recovered = set(detail["fixed_label_indices"])
    for label_index, row in enumerate(rows):
        if label_index not in recovered:
            mark(
                int(round(row["x_px"])),
                int(round(row["y_px"])),
                (255, 40, 40),
                4,
            )
    return rgb


def _write_overlay_tiff(
    path: Path,
    pages: Sequence[np.ndarray],
    compression: str,
) -> dict[str, Any]:
    temporary = path.with_name(path.name + ".partial")
    with tifffile.TiffWriter(temporary) as writer:
        for page in pages:
            writer.write(
                page,
                photometric="rgb",
                compression=compression,
                metadata=None,
            )
    temporary.replace(path)
    with tifffile.TiffFile(path) as tiff:
        if len(tiff.pages) != len(pages):
            raise RuntimeError("overlay TIFF page verification failed")
    return {
        "path": f"diagnostics/{path.name}",
        "pages": len(pages),
        "shape": list(pages[0].shape),
        "bytes": path.stat().st_size,
        "colors": {
            "green": "candidate matched to a known label",
            "cyan": "unmatched candidate; scientific status unknown",
            "red": "known label missed at the fixed budget",
        },
    }


def _per_neuron_audit(
    family_payloads: Mapping[str, Any],
    feature_screen: Sequence[Mapping[str, Any]],
    feature_details: Sequence[Mapping[str, Any]],
    nested_feature: Mapping[str, Any],
    oracle_fixed: Mapping[int, set[int]],
    labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    carrier = next(
        row for row in feature_screen if row["feature_id"] == "carrier_signed"
    )
    core_ids = (
        "local_psd_signal",
        "asymmetric_state",
        "spatial_coherence",
        "cross_scale_rank",
        "cfar_score",
    )
    rows = []
    for burst in BURSTS:
        burst_labels = _label_rows(labels, burst)
        carrier_detail = next(
            fold for fold in carrier["folds"] if fold["burst_id"] == burst
        )
        carrier_exact = next(
            row
            for row in feature_details
            if row["feature_id"] == "carrier_signed"
            and int(row["burst_id"]) == burst
        )
        selected_feature = next(
            row
            for row in nested_feature["folds"]
            if int(row["held_out_burst"]) == burst
        )["selected_feature_id"]
        selected_feature_exact = next(
            row
            for row in feature_details
            if row["feature_id"] == selected_feature
            and int(row["burst_id"]) == burst
        )
        linear = next(
            fold
            for fold in family_payloads["linear"]["folds"]
            if fold["burst_id"] == burst
        )
        mlp = next(
            fold
            for fold in family_payloads["mlp"]["folds"]
            if fold["burst_id"] == burst
        )
        carrier_recovered = set(carrier_exact["fixed_label_indices"])
        selected_feature_recovered = set(
            selected_feature_exact["fixed_label_indices"]
        )
        linear_recovered = set(linear["detail"]["fixed_label_indices"])
        mlp_recovered = set(mlp["detail"]["fixed_label_indices"])
        for index, label in enumerate(burst_labels):
            rows.append(
                {
                    "burst_id": burst,
                    "label_index_within_burst": index,
                    "roi_identity": label["roi_identity"],
                    "x_px": label["x_px"],
                    "y_px": label["y_px"],
                    "oracle_union_recoverable": index
                    in oracle_fixed.get(burst, set()),
                    "union_carrier_recovered": index in carrier_recovered,
                    "nested_single_feature_id": selected_feature,
                    "nested_single_feature_recovered": index
                    in selected_feature_recovered,
                    "linear_recovered": index in linear_recovered,
                    "mlp_recovered": index in mlp_recovered,
                    "both_rankers_recovered": index in linear_recovered
                    and index in mlp_recovered,
                    "neither_ranker_recovered": index not in linear_recovered
                    and index not in mlp_recovered,
                    "carrier_fold_fixed_recall": carrier_detail[
                        "fixed_recall"
                    ],
                    "reference_feature_ids": ",".join(core_ids),
                    "carrier_exact_recovery_available": True,
                }
            )
    return rows


def _report(path: Path, metrics: Mapping[str, Any]) -> None:
    baseline = metrics["feature_bank_carrier_baseline"]
    linear = metrics["nested_rankers"]["linear"]
    mlp = metrics["nested_rankers"]["mlp"]
    nested_feature = metrics["nested_feature_selection"]
    fixed_budget = metrics["fixed_budget"]
    union_carrier = metrics["same_union_carrier_baseline"]
    standardized_carrier = metrics["standardized_carrier_native_baseline"]
    oracle = metrics["oracle_summary"]
    lines = [
        f"# {metrics['experiment_id']}",
        "",
        "## Outcome",
        "",
        metrics["conclusion"],
        "",
        "| Method | Mean recall | Fixed-budget recall | Candidates | Minimum fold fixed |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Archived centered-residual carrier | {baseline['mean_recall']:.3f} | "
        f"{baseline['fixed_budget_mean_recall']:.3f} | "
        f"{baseline['event_candidates']} | — |",
        f"| Quiet-standardized carrier, native peaks | — | "
        f"{standardized_carrier['mean_recall']:.3f} | — | "
        f"{standardized_carrier['minimum_burst_recall']:.3f} |",
        f"| Carrier score on broad proposal union | — | "
        f"{union_carrier['mean_recall']:.3f} | — | "
        f"{union_carrier['minimum_burst_recall']:.3f} |",
        f"| Nested single-feature selection | "
        f"{nested_feature['mean_recall']:.3f} | "
        f"{nested_feature['fixed_budget_mean_recall']:.3f} | "
        f"{nested_feature['event_candidates']} | — |",
        f"| Nested linear ranker | {linear['mean_recall']:.3f} | "
        f"{linear['fixed_budget_mean_recall']:.3f} | "
        f"{linear['event_candidates']} | {linear['minimum_fixed_recall']:.3f} |",
        f"| Nested residual MLP | {mlp['mean_recall']:.3f} | "
        f"{mlp['fixed_budget_mean_recall']:.3f} | "
        f"{mlp['event_candidates']} | {mlp['minimum_fixed_recall']:.3f} |",
        "",
        f"The optimistic per-source union covers "
        f"{oracle['fixed_budget_mean_coverage']:.3f} of labels at "
        f"{fixed_budget} "
        "candidates per source. It is a headroom diagnostic, not a deployable "
        "fixed-total-budget score.",
        "",
        "Unmatched event candidates remain unknown because labels are sparse. "
        "Candidate burden is not precision.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config: InnovationRankerConfig) -> dict[str, Any]:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(config.resources["cpu_threads"])
    audit = _matching_preflight(config)
    partial = Path(str(config.output_dir) + ".partial")
    partial.mkdir(parents=True)
    evaluation_dir = partial / "evaluation"
    diagnostics_dir = partial / "diagnostics"
    models_dir = partial / "models"
    for directory in (evaluation_dir, diagnostics_dir, models_dir):
        directory.mkdir()
    _atomic_json(partial / "preflight.json", audit)
    _atomic_json(partial / "config.resolved.json", config.to_dict())
    progress = partial / "progress.jsonl"
    started = time.time()
    labels = label_core.load_labels(config.labels_tsv)
    base_config = FeatureUtilityConfig.load(
        config.feature_root / "config.resolved.json"
    )
    feature_maps, calibration = _generate_maps(
        config, labels, base_config, progress
    )
    _atomic_json(evaluation_dir / "noise_calibration.json", calibration)
    normalizers, quiet_tables, event_tables = _candidate_tables(
        feature_maps, config
    )
    _atomic_json(
        evaluation_dir / "candidate_inventory.json",
        {
            "feature_ids": list(FEATURE_IDS),
            "proposal_source_ids": list(PROPOSAL_SOURCE_IDS),
            "quiet_candidate_counts": [
                len(table.positions) for table in quiet_tables
            ],
            "event_candidate_counts": {
                str(burst): len(table.positions)
                for burst, table in event_tables.items()
            },
            "normalizers": {
                feature_id: {
                    "quiet_median": values[0],
                    "quiet_scale": values[1],
                }
                for feature_id, values in normalizers.items()
            },
        },
    )
    feature_screen, feature_details = _screen_features(
        quiet_tables, event_tables, labels, config
    )
    nested_feature = _nested_feature_selection(feature_screen)
    _atomic_json(
        evaluation_dir / "feature_screen.json",
        {
            "feature_count": len(feature_screen),
            "rows": feature_screen,
            "details": feature_details,
            "nested_selection": nested_feature,
        },
    )
    oracle_rows, oracle_fixed = _oracle_coverage(
        feature_maps, normalizers, labels, config
    )
    _atomic_json(
        evaluation_dir / "oracle_coverage.json",
        {"rows": oracle_rows},
    )
    fixed_oracle_rows = [
        row
        for row in oracle_rows
        if row["source_budget"]
        == int(config.evaluation["fixed_candidates_per_burst"])
    ]
    oracle_summary = {
        "fixed_budget_mean_coverage": float(
            np.mean([row["union_coverage"] for row in fixed_oracle_rows])
        ),
        "fixed_budget_folds": fixed_oracle_rows,
        "saturation_mean_coverage": float(
            np.mean(
                [
                    row["union_coverage"]
                    for row in oracle_rows
                    if row["source_budget"]
                    == max(config.evaluation["oracle_source_budgets"])
                ]
            )
        ),
    }
    inner_rows, registry = _inner_fine_tuning(
        config, event_tables, quiet_tables, labels, progress
    )
    _atomic_json(
        evaluation_dir / "inner_fine_tuning.json",
        {
            "evaluation_row_count": len(inner_rows),
            "model_fit_count": config.inner_fit_count,
            "rows": inner_rows,
        },
    )
    nested_rankers, model_details = _outer_folds(
        inner_rows,
        registry,
        event_tables,
        quiet_tables,
        labels,
        config,
    )
    _atomic_json(evaluation_dir / "nested_rankers.json", nested_rankers)
    for model_id, model in model_details.items():
        _atomic_json(models_dir / f"{model_id}.json", model)

    budget_curves = _budget_curves(
        feature_maps,
        normalizers,
        event_tables,
        nested_feature,
        nested_rankers,
        model_details,
        labels,
        config,
    )
    _atomic_json(evaluation_dir / "budget_curves.json", budget_curves)
    neuron_rows = _per_neuron_audit(
        nested_rankers,
        feature_screen,
        feature_details,
        nested_feature,
        oracle_fixed,
        labels,
    )
    _atomic_tsv(evaluation_dir / "per_neuron_audit.tsv", neuron_rows)
    common_misses = [
        row
        for row in neuron_rows
        if row["oracle_union_recoverable"] and row["neither_ranker_recovered"]
    ]
    unrecoverable = [
        row for row in neuron_rows if not row["oracle_union_recoverable"]
    ]
    _atomic_tsv(
        evaluation_dir / "recoverable_but_missed.tsv", common_misses
    )
    _atomic_tsv(
        evaluation_dir / "not_in_feature_union.tsv", unrecoverable
    )

    structure = np.load(
        config.feature_root / "structure_unit.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    diagnostics = []
    for family in ("linear", "mlp"):
        score_maps = []
        overlay_pages = []
        for burst in BURSTS:
            model = model_details[f"{family}__burst{burst}"]
            scores = _score_model(event_tables[burst], model)
            score_maps.append(
                _score_map(
                    scores,
                    event_tables[burst],
                    structure.shape,
                    float(config.visualization["score_sigma_px"]),
                )
            )
            fold = next(
                row
                for row in nested_rankers[family]["folds"]
                if row["burst_id"] == burst
            )
            overlay_pages.append(
                _overlay_page(
                    structure,
                    event_tables[burst],
                    scores,
                    fold["detail"],
                    _label_rows(labels, burst),
                    int(config.visualization["overlay_candidate_limit"]),
                )
            )
        diagnostics.append(
            {
                "family": family,
                "score_tiff": _write_score_tiff(
                    diagnostics_dir / f"{family}_candidate_scores.tif",
                    score_maps,
                    compression=str(config.visualization["compression"]),
                    description={
                        "family": family,
                        "pages": "bursts 1 through 4",
                        "semantics": "pooled proposal-ranking score",
                    },
                ),
                "overlay_tiff": _write_overlay_tiff(
                    diagnostics_dir / f"{family}_candidate_overlay.tif",
                    overlay_pages,
                    str(config.visualization["compression"]),
                ),
            }
        )
    morphology_pages = [
        feature_maps["cut_center_sigma2p5"]["events"][burst]
        for burst in BURSTS
    ] + [
        feature_maps["cut_ring_r4p5_t1p25"]["events"][burst]
        for burst in BURSTS
    ]
    diagnostics.append(
        {
            "family": "cut_morphology_basis",
            "score_tiff": _write_score_tiff(
                diagnostics_dir / "cut_morphology_basis.tif",
                morphology_pages,
                compression=str(config.visualization["compression"]),
                description={
                    "pages_1_to_4": "center sigma 2.5, bursts 1-4",
                    "pages_5_to_8": "ring radius 4.5 thickness 1.25, bursts 1-4",
                },
            ),
        }
    )

    carrier_baseline = json.loads(
        (config.feature_root / "metrics.json").read_text(encoding="utf-8")
    )["carrier_baseline"]
    best_family = max(
        ("linear", "mlp"),
        key=lambda family: (
            nested_rankers[family]["fixed_budget_mean_recall"],
            nested_rankers[family]["minimum_fixed_recall"],
            -nested_rankers[family]["event_candidates"],
            nested_rankers[family]["mean_recall"],
        ),
    )
    best = nested_rankers[best_family]
    fixed_budget = int(config.evaluation["fixed_candidates_per_burst"])
    same_union_carrier = next(
        row
        for row in budget_curves["summaries"]["union_carrier_score"]
        if int(row["budget"]) == fixed_budget
    )
    standardized_carrier_native = next(
        row
        for row in budget_curves["summaries"]["standardized_carrier_native"]
        if int(row["budget"]) == fixed_budget
    )
    standardization_gain = float(
        standardized_carrier_native["mean_recall"]
        - carrier_baseline["fixed_budget_mean_recall"]
    )
    proposal_union_gain = float(
        same_union_carrier["mean_recall"]
        - standardized_carrier_native["mean_recall"]
    )
    ranking_gain = float(
        best["fixed_budget_mean_recall"]
        - same_union_carrier["mean_recall"]
    )
    conclusion = (
        f"At budget {fixed_budget}, quiet per-pixel standardization raised "
        f"the archived centered-residual carrier from "
        f"{carrier_baseline['fixed_budget_mean_recall']:.3f} to "
        f"{standardized_carrier_native['mean_recall']:.3f}. Holding the "
        f"standardized carrier score fixed, the positive-evidence proposal "
        f"union reached {same_union_carrier['mean_recall']:.3f}. The "
        f"preferred nested "
        f"ranker was {best_family} at {best['fixed_budget_mean_recall']:.3f}; "
        f"nested single-feature selection reached "
        f"{nested_feature['fixed_budget_mean_recall']:.3f}. The decomposed "
        f"increments were {standardization_gain:+.3f} from standardization, "
        f"{proposal_union_gain:+.3f} from the proposal union, and "
        f"{ranking_gain:+.3f} from linear fine-tuning. The optimistic "
        f"{fixed_budget}-per-"
        f"source union coverage was "
        f"{oracle_summary['fixed_budget_mean_coverage']:.3f}."
    )
    metrics = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": "completed",
        "feature_bank_carrier_baseline": carrier_baseline,
        "candidate_feature_screen": feature_screen,
        "nested_feature_selection": nested_feature,
        "budget_curves": budget_curves,
        "fixed_budget": fixed_budget,
        "same_union_carrier_baseline": same_union_carrier,
        "standardized_carrier_native_baseline": standardized_carrier_native,
        "gain_decomposition": {
            "quiet_per_pixel_standardization": standardization_gain,
            "positive_evidence_proposal_union": proposal_union_gain,
            "preferred_nested_ranking": ranking_gain,
        },
        "oracle_summary": oracle_summary,
        "nested_rankers": nested_rankers,
        "best_nested_family": best_family,
        "noise_calibration": calibration,
        "per_neuron_summary": {
            "label_rows": len(neuron_rows),
            "recoverable_but_missed_by_both": len(common_misses),
            "not_in_58_per_source_union": len(unrecoverable),
        },
        "diagnostics": diagnostics,
        "conclusion": conclusion,
        "precision_contract": (
            "Sparse unmatched event candidates are unknown. Candidate burden "
            "does not establish real-data precision."
        ),
        "elapsed_seconds": time.time() - started,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    _atomic_json(partial / "metrics.json", metrics)
    _report(partial / "REPORT.md", metrics)
    _atomic_json(
        partial / "run_state.json",
        {
            "status": "completed",
            "elapsed_seconds": metrics["elapsed_seconds"],
            "max_rss_mib": metrics["max_rss_mib"],
            "feature_count": len(FEATURE_IDS),
            "proposal_source_count": len(PROPOSAL_SOURCE_IDS),
            "inner_fit_count": config.inner_fit_count,
            "outer_refit_count": config.outer_refit_count,
            "diagnostic_tiff_count": sum(
                int("score_tiff" in row) + int("overlay_tiff" in row)
                for row in diagnostics
            ),
        },
    )
    partial.replace(config.output_dir)
    return metrics
