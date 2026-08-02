"""Principe-aligned local patch-information experiment for Spon Ca Burst."""
from __future__ import annotations

import csv
import itertools
import json
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any, Mapping, Sequence

import numpy as np
import tifffile

from neurobench.algorithms.patch_information import (
    information_fields_tensor,
    local_histogram_tensor,
)
from neurobench.algorithms.proposal_ranking import (
    CandidateTable,
    fit_bounded_pairwise_linear,
    merge_peak_proposals,
    normalize_map,
    robust_map_normalizer,
    sample_candidate_features,
    score_bounded_pairwise_linear,
)
from neurobench.experiments.learnable_contrast import core as label_core
from neurobench.experiments.pairwise_separation.evaluation import (
    QUIET_DURATIONS,
    QUIET_STARTS,
    event_intervals,
)
from neurobench.metrics.sparse_detection import (
    extract_local_maxima,
    match_peaks_one_to_one,
    quiet_calibrated_threshold,
    temporal_pool,
)

from .feature_utility_config import FeatureUtilityConfig
from .innovation_grid import (
    _atomic_json,
    _available_ram_mib,
    _progress,
    _sha256,
    _snapshots,
)
from .innovation_ranker_config import (
    FEATURE_SETS as V5_FEATURE_SETS,
    PROPOSAL_SOURCE_IDS as V5_PROPOSAL_SOURCE_IDS,
    InnovationRankerConfig,
)
from .innovation_ranker_program import (
    _generate_maps as _generate_v5_maps,
    _overlay_page,
    _score_map,
    _write_overlay_tiff,
    _write_score_tiff,
)
from .patch_information_config import FAMILIES, PatchInformationConfig


BURSTS = (1, 2, 3, 4)
SEPARATION_FEATURES = tuple(V5_FEATURE_SETS["separation"])


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


def _feature_id(family: str, patch: int, bandwidth: float) -> str:
    aliases = {
        "renyi2_information_potential": "renyi2_ip",
        "cs_quiet_divergence": "cs_quiet",
        "local_correntropy": "correntropy",
    }
    return f"{aliases[family]}__p{int(patch)}__bw{_token(bandwidth)}"


def _itl_feature_ids(config: PatchInformationConfig) -> tuple[str, ...]:
    return tuple(
        _feature_id(family, int(patch), float(bandwidth))
        for patch in config.itl["patch_sizes_px"]
        for bandwidth in config.itl["kernel_bandwidths_z"]
        for family in FAMILIES
    )


def preflight(
    config: PatchInformationConfig, *, write_artifacts: bool = True
) -> dict[str, Any]:
    ranker = InnovationRankerConfig.load(config.source_ranker_config)
    inputs = (
        config.source_ranker_config,
        config.source_ranker_root / "metrics.json",
        config.source_ranker_root / "run_state.json",
        ranker.source_video,
        ranker.labels_tsv,
        ranker.feature_manifest,
        ranker.feature_root / "features" / "carrier_signed.npy",
        ranker.feature_root / "config.resolved.json",
    )
    missing = [str(path) for path in inputs if not path.is_file()]
    source_shape = carrier_shape = None
    labels: list[dict[str, Any]] = []
    source_valid = labels_valid = ranker_valid = finite = False
    if not missing:
        source = np.load(ranker.source_video, mmap_mode="r", allow_pickle=False)
        carrier = np.load(
            ranker.feature_root / "features" / "carrier_signed.npy",
            mmap_mode="r", allow_pickle=False,
        )
        source_shape, carrier_shape = list(source.shape), list(carrier.shape)
        expected_frames = (
            int(ranker.frames["review_end_ui"])
            - int(ranker.frames["review_start_ui"]) + 1
        )
        source_valid = bool(
            source.ndim == 3
            and carrier.shape == (expected_frames, *source.shape[1:])
            and int(ranker.frames["quiet_count"]) == 100
        )
        finite = source_valid and bool(
            np.isfinite(carrier[::32, ::16, ::16]).all()
        )
        labels = label_core.load_labels(ranker.labels_tsv)
        labels_valid = bool(
            len(labels) == 79
            and len({row["roi_identity"] for row in labels}) == 27
            and all(
                0 <= row["x_px"] < source.shape[2]
                and 0 <= row["y_px"] < source.shape[1]
                for row in labels
            )
        )
        completed = json.loads(
            (config.source_ranker_root / "metrics.json").read_text(
                encoding="utf-8"
            )
        )
        ranker_valid = bool(
            completed.get("status") == "completed"
            and completed.get("experiment_id") == ranker.experiment_id
        )
    gpu: dict[str, Any] = {"available": False}
    try:
        import torch
        gpu["available"] = bool(torch.cuda.is_available())
        if gpu["available"]:
            free, total = torch.cuda.mem_get_info()
            gpu.update(
                name=torch.cuda.get_device_name(0),
                free_mib=free / 2**20,
                total_mib=total / 2**20,
            )
    except ImportError:
        pass
    pixels = 0 if source_shape is None else int(source_shape[1]) * int(source_shape[2])
    batch = int(config.itl["frame_batch_size"])
    bins = len(config.itl["bin_centers_z"])
    estimated_gpu = 7 * batch * bins * pixels * 4 / 2**20 + 768
    estimated_ram = 4096.0
    estimated_output = 384.0
    probe = config.output_dir.parent
    while not probe.exists():
        probe = probe.parent
    free_disk = shutil.disk_usage(probe).free / 2**20
    cuda_requested = config.resources["device"] == "cuda"
    gates = {
        "inputs_exist": not missing,
        "source_and_carrier_valid": source_valid,
        "finite_carrier_sample": finite,
        "labels_valid": labels_valid,
        "authoritative_ranker_valid": ranker_valid,
        "feature_grid_count_valid": len(_itl_feature_ids(config)) == 27,
        "output_absent": not config.output_dir.exists(),
        "partial_output_absent": not Path(str(config.output_dir) + ".partial").exists(),
        "preflight_separate_from_output": config.preflight_dir != config.output_dir,
        "ram_cap_sufficient": estimated_ram <= int(config.resources["max_ram_mib"]),
        "available_ram_sufficient": estimated_ram <= _available_ram_mib(),
        "disk_headroom_sufficient": free_disk >= int(config.resources["min_free_disk_mib"]) + estimated_output,
        "output_cap_sufficient": estimated_output <= int(config.resources["max_output_mib"]),
        "requested_device_available": (not cuda_requested) or gpu["available"],
        "gpu_memory_cap_sufficient": estimated_gpu <= int(config.resources["max_gpu_memory_mib"]),
        "live_gpu_memory_sufficient": (not cuda_requested) or estimated_gpu <= gpu.get("free_mib", 0),
    }
    payload = {
        "schema_version": 1,
        "kind": "read_only_principe_patch_information_preflight",
        "experiment_id": config.experiment_id,
        "ready": all(gates.values()),
        "gates": gates,
        "source_shape": source_shape,
        "carrier_shape": carrier_shape,
        "label_rows": len(labels),
        "roi_identities": len({row["roi_identity"] for row in labels}),
        "design": {
            "feature_count": config.feature_count,
            "feature_ids": list(_itl_feature_ids(config)),
            "fixed_lane_count": config.fixed_lane_count,
            "linear_config_count": config.linear_config_count,
            "inner_fit_count": config.inner_fit_count,
            "outer_refit_count": config.outer_refit_count,
            "total_model_fit_count": config.inner_fit_count + config.outer_refit_count,
        },
        "resources": {
            "estimated_peak_ram_mib": estimated_ram,
            "available_ram_mib": _available_ram_mib(),
            "estimated_peak_gpu_memory_mib": estimated_gpu,
            "estimated_output_mib": estimated_output,
            "free_disk_mib": free_disk,
            "gpu": gpu,
            **config.resources,
        },
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in inputs if path.is_file()
        ],
        "system_snapshot": _snapshots(),
        "scientific_contract": (
            "Quadratic Renyi information potential, Parzen Cauchy-Schwarz "
            "quiet divergence, and correntropy are computed without labels. "
            "Labels enter only fixed evaluation and nested outer-burst model "
            "selection. Sparse unmatched candidates remain unknown."
        ),
    }
    if write_artifacts:
        config.preflight_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(config.preflight_dir / "preflight.json", payload)
        _atomic_json(config.preflight_dir / "config.resolved.json", config.to_dict())
        if not missing and source_valid:
            label_core._write_overlay(
                np.load(ranker.source_video, mmap_mode="r", allow_pickle=False),
                labels,
                config.preflight_dir / "label_projection_overlay.png",
            )
    if not payload["ready"]:
        raise RuntimeError(f"patch-information preflight failed: {payload}")
    return payload


def _matching_preflight(config: PatchInformationConfig) -> dict[str, Any]:
    audit = json.loads((config.preflight_dir / "preflight.json").read_text(encoding="utf-8"))
    resolved = json.loads((config.preflight_dir / "config.resolved.json").read_text(encoding="utf-8"))
    if not audit.get("ready") or resolved != config.to_dict():
        raise RuntimeError("run requires a matching ready preflight")
    if config.output_dir.exists() or Path(str(config.output_dir) + ".partial").exists():
        raise FileExistsError("completed or partial output already exists")
    return audit


def _pool_values(
    quiet: np.ndarray,
    events: Mapping[int, np.ndarray],
    tau: float,
) -> dict[str, Any]:
    baseline = np.median(quiet, axis=0).astype(np.float32)
    low, high = np.percentile(
        np.asarray(quiet[:, ::4, ::4], dtype=np.float32), [1.0, 99.9]
    )
    scale = max(float(high - low), 1e-6)

    def pool(values: np.ndarray) -> np.ndarray:
        normalized = np.maximum(
            (np.asarray(values, dtype=np.float32) - baseline[None]) / scale,
            0,
        )
        return temporal_pool(normalized, f"lme{float(tau)}")

    return {
        "quiet": [
            pool(quiet[start:start + duration])
            for start, duration in zip(QUIET_STARTS, QUIET_DURATIONS)
        ],
        "events": {int(burst): pool(values) for burst, values in events.items()},
        "normalization_scale": scale,
    }


def _generate_itl_maps(
    config: PatchInformationConfig,
    ranker: InnovationRankerConfig,
    labels: list[dict[str, Any]],
    base_config: FeatureUtilityConfig,
    progress: Path,
) -> dict[str, dict[str, Any]]:
    import torch

    device = torch.device(str(config.resources["device"]))
    carrier = np.load(
        ranker.feature_root / "features" / "carrier_signed.npy",
        mmap_mode="r", allow_pickle=False,
    )
    quiet_count = int(ranker.frames["quiet_count"])
    centers = tuple(float(value) for value in config.itl["bin_centers_z"])
    bandwidths = tuple(float(value) for value in config.itl["kernel_bandwidths_z"])
    batch_size = int(config.itl["frame_batch_size"])
    intervals = event_intervals(labels, int(ranker.frames["review_start_ui"]))
    height, width = carrier.shape[1:]
    tau = float(base_config.evaluation["temporal_pool_tau"])
    result: dict[str, dict[str, Any]] = {}
    for patch_index, patch in enumerate(config.itl["patch_sizes_px"], start=1):
        patch = int(patch)
        quiet_histogram = torch.zeros(
            (len(centers), height, width), dtype=torch.float32, device=device
        )
        with torch.inference_mode():
            for start in range(0, quiet_count, batch_size):
                stop = min(quiet_count, start + batch_size)
                frames = torch.as_tensor(
                    np.asarray(carrier[start:stop], dtype=np.float32), device=device
                )
                quiet_histogram += local_histogram_tensor(
                    frames, centers=centers, patch_size_px=patch
                ).sum(dim=0)
            quiet_histogram /= quiet_count
        lane_ids = [
            _feature_id(family, patch, bandwidth)
            for bandwidth in bandwidths for family in FAMILIES
        ]
        quiet_store = np.empty(
            (len(lane_ids), quiet_count, height, width), dtype=np.float16
        )
        event_store = {
            burst: np.empty(
                (len(lane_ids), stop - start, height, width), dtype=np.float16
            )
            for burst, (start, stop) in intervals.items()
        }

        def compute(source_start: int, source_stop: int, target: np.ndarray) -> None:
            for local_start in range(0, source_stop - source_start, batch_size):
                local_stop = min(source_stop - source_start, local_start + batch_size)
                start = source_start + local_start
                stop = source_start + local_stop
                frames = torch.as_tensor(
                    np.asarray(carrier[start:stop], dtype=np.float32), device=device
                )
                with torch.inference_mode():
                    histogram = local_histogram_tensor(
                        frames, centers=centers, patch_size_px=patch
                    )
                    lanes = []
                    for bandwidth in bandwidths:
                        fields = information_fields_tensor(
                            histogram, frames, quiet_histogram,
                            centers=centers, bandwidth=bandwidth,
                        )
                        lanes.extend(fields[family] for family in FAMILIES)
                    values = torch.stack(lanes, dim=0).to("cpu").numpy()
                target[:, local_start:local_stop] = values.astype(np.float16)
                del frames, histogram, values

        compute(0, quiet_count, quiet_store)
        for burst, (start, stop) in intervals.items():
            compute(start, stop, event_store[int(burst)])
        for lane_index, lane_id in enumerate(lane_ids):
            result[lane_id] = _pool_values(
                quiet_store[lane_index],
                {burst: values[lane_index] for burst, values in event_store.items()},
                tau,
            )
        del quiet_store, event_store, quiet_histogram
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _progress(
            progress, "itl_patch_complete", patch_size_px=patch,
            patch_index=patch_index,
            patch_total=len(config.itl["patch_sizes_px"]),
            feature_count=len(lane_ids),
        )
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if rss > float(config.resources["max_ram_mib"]):
            raise MemoryError(f"ITL patch generation exceeded RAM cap: {rss:.1f} MiB")
    if set(result) != set(_itl_feature_ids(config)):
        raise RuntimeError("ITL feature map inventory mismatch")
    return result


def _label_rows(labels: list[dict[str, Any]], burst: int) -> list[dict[str, Any]]:
    return [row for row in labels if int(row["burst_id"]) == int(burst)]


def _evaluate_maps(
    lane_id: str,
    maps: Mapping[str, Any],
    labels: list[dict[str, Any]],
    ranker: InnovationRankerConfig,
    budgets: Sequence[int],
) -> dict[str, Any]:
    threshold = quiet_calibrated_threshold(
        maps["quiet"], int(ranker.proposals["nms_distance_px"]),
        float(ranker.evaluation["quiet_false_candidates_per_map"]), limit=3000,
    )
    folds = []
    for burst in BURSTS:
        ranked = extract_local_maxima(
            maps["events"][burst], int(ranker.proposals["nms_distance_px"]),
            limit=max(500, max(int(value) for value in budgets)),
        )
        rows = _label_rows(labels, burst)
        selected = [peak for peak in ranked if peak[0] >= threshold]
        threshold_matches = match_peaks_one_to_one(
            selected, rows, float(ranker.evaluation["match_radius_px"])
        )[0]
        budget_rows = {}
        for budget in budgets:
            matches = match_peaks_one_to_one(
                ranked[: int(budget)], rows,
                float(ranker.evaluation["match_radius_px"]),
            )[0]
            budget_rows[str(int(budget))] = {
                "matched": len(matches), "labels": len(rows),
                "recall": len(matches) / len(rows),
                "label_indices": sorted(int(match[0]) for match in matches),
            }
        folds.append({
            "burst_id": burst, "labels": len(rows),
            "threshold_matched": len(threshold_matches),
            "threshold_recall": len(threshold_matches) / len(rows),
            "threshold_candidates": len(selected), "budgets": budget_rows,
        })
    return {
        "config_id": lane_id, "threshold": float(threshold), "folds": folds,
        "threshold_mean_recall": float(np.mean([row["threshold_recall"] for row in folds])),
        "threshold_event_candidates": sum(row["threshold_candidates"] for row in folds),
        "budget_mean_recall": {
            str(int(budget)): float(np.mean([
                row["budgets"][str(int(budget))]["recall"] for row in folds
            ])) for budget in budgets
        },
    }


def _unit_maps(maps: Mapping[str, Any]) -> dict[str, Any]:
    normalizer = robust_map_normalizer(maps["quiet"])
    return {
        "quiet": [normalize_map(value, normalizer, clip=8.0) for value in maps["quiet"]],
        "events": {burst: normalize_map(value, normalizer, clip=8.0) for burst, value in maps["events"].items()},
    }


def _combine_maps(
    carrier: Mapping[str, Any], feature: Mapping[str, Any],
    *, kind: str, value: float,
) -> dict[str, Any]:
    carrier_unit, feature_unit = _unit_maps(carrier), _unit_maps(feature)

    def combine(raw: np.ndarray, auxiliary: np.ndarray) -> np.ndarray:
        auxiliary = np.clip(auxiliary, 0, 1)
        if kind == "boost":
            return (raw + float(value) * auxiliary).astype(np.float32)
        if kind == "gate":
            return (raw * (float(value) + (1.0 - float(value)) * auxiliary)).astype(np.float32)
        raise ValueError("unknown fusion kind")

    return {
        "quiet": [combine(a, b) for a, b in zip(carrier_unit["quiet"], feature_unit["quiet"])],
        "events": {burst: combine(carrier_unit["events"][burst], feature_unit["events"][burst]) for burst in BURSTS},
    }


def _screen(
    config: PatchInformationConfig,
    ranker: InnovationRankerConfig,
    itl_maps: Mapping[str, Mapping[str, Any]],
    carrier_maps: Mapping[str, Any],
    labels: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    budgets = [int(value) for value in config.screen["budgets"]]
    baseline = _evaluate_maps("carrier_native", carrier_maps, labels, ranker, budgets)
    rows = []
    for feature_id, maps in itl_maps.items():
        standalone = _evaluate_maps(f"standalone__{feature_id}", maps, labels, ranker, budgets)
        rows.append({"kind": "standalone", "feature_id": feature_id, "value": None, **standalone})
        for weight in config.screen["carrier_boosts"]:
            lane_id = f"boost__{feature_id}__{_token(weight)}"
            result = _evaluate_maps(
                lane_id, _combine_maps(carrier_maps, maps, kind="boost", value=float(weight)),
                labels, ranker, budgets,
            )
            rows.append({"kind": "boost", "feature_id": feature_id, "value": float(weight), **result})
        for floor in config.screen["carrier_gate_floors"]:
            lane_id = f"gate__{feature_id}__{_token(floor)}"
            result = _evaluate_maps(
                lane_id, _combine_maps(carrier_maps, maps, kind="gate", value=float(floor)),
                labels, ranker, budgets,
            )
            rows.append({"kind": "gate", "feature_id": feature_id, "value": float(floor), **result})
    return baseline, rows


def _selection_key(row: Mapping[str, Any], held_out: int, config: PatchInformationConfig) -> tuple[Any, ...]:
    training = [fold for fold in row["folds"] if int(fold["burst_id"]) != int(held_out)]
    primary = [str(int(value)) for value in config.screen["primary_budgets"]]
    primary_values = [fold["budgets"][budget]["recall"] for fold in training for budget in primary]
    secondary = [fold["budgets"]["58"]["recall"] for fold in training]
    return (
        float(np.mean(primary_values)), float(np.min(primary_values)),
        float(np.mean(secondary)),
        float(np.mean([fold["threshold_recall"] for fold in training])),
        -sum(fold["threshold_candidates"] for fold in training),
        str(row["config_id"]),
    )


def _crossfit_screen(
    rows: Sequence[Mapping[str, Any]], kind: str, config: PatchInformationConfig
) -> dict[str, Any]:
    candidates = [row for row in rows if row["kind"] == kind]
    folds = []
    for held_out in BURSTS:
        selected = max(candidates, key=lambda row: _selection_key(row, held_out, config))
        held = next(fold for fold in selected["folds"] if int(fold["burst_id"]) == held_out)
        folds.append({
            "held_out_burst": held_out,
            "selected_config_id": selected["config_id"],
            "selected_feature_id": selected["feature_id"],
            "selected_value": selected["value"],
            **held,
        })
    return {
        "kind": kind, "folds": folds,
        "threshold_mean_recall": float(np.mean([row["threshold_recall"] for row in folds])),
        "threshold_event_candidates": sum(row["threshold_candidates"] for row in folds),
        "budget_mean_recall": {
            str(int(budget)): float(np.mean([
                row["budgets"][str(int(budget))]["recall"] for row in folds
            ])) for budget in config.screen["budgets"]
        },
    }


def _normalizers(feature_maps: Mapping[str, Mapping[str, Any]]) -> dict[str, tuple[float, float]]:
    return {feature_id: robust_map_normalizer(maps["quiet"]) for feature_id, maps in feature_maps.items()}


def _candidate_tables(
    feature_maps: Mapping[str, Mapping[str, Any]],
    feature_ids: Sequence[str], proposal_ids: Sequence[str],
    ranker: InnovationRankerConfig,
) -> tuple[dict[str, tuple[float, float]], list[CandidateTable], dict[int, CandidateTable]]:
    normalizer_ids = tuple(dict.fromkeys((*feature_ids, *proposal_ids)))
    normalizers = _normalizers(
        {feature_id: feature_maps[feature_id] for feature_id in normalizer_ids}
    )
    quiet_tables = []
    for index in range(4):
        positions, source_count = merge_peak_proposals(
            {feature_id: feature_maps[feature_id]["quiet"][index] for feature_id in proposal_ids},
            normalizers,
            nms_distance_px=int(ranker.proposals["nms_distance_px"]),
            per_source_limit=int(ranker.proposals["per_source_limit"]),
            dedupe_radius_px=float(ranker.proposals["dedupe_radius_px"]), clip=8.0,
        )
        matrix = sample_candidate_features(
            positions,
            {feature_id: feature_maps[feature_id]["quiet"][index] for feature_id in feature_ids},
            feature_ids, normalizers, clip=8.0,
        )
        quiet_tables.append(CandidateTable(positions, matrix, source_count))
    events = {}
    for burst in BURSTS:
        positions, source_count = merge_peak_proposals(
            {feature_id: feature_maps[feature_id]["events"][burst] for feature_id in proposal_ids},
            normalizers,
            nms_distance_px=int(ranker.proposals["nms_distance_px"]),
            per_source_limit=int(ranker.proposals["per_source_limit"]),
            dedupe_radius_px=float(ranker.proposals["dedupe_radius_px"]), clip=8.0,
        )
        matrix = sample_candidate_features(
            positions,
            {feature_id: feature_maps[feature_id]["events"][burst] for feature_id in feature_ids},
            feature_ids, normalizers, clip=8.0,
        )
        events[burst] = CandidateTable(positions, matrix, source_count)
    return normalizers, quiet_tables, events


def _evaluate_scores(
    scores: np.ndarray, table: CandidateTable, rows: list[dict[str, Any]],
    quiet_scores: Sequence[np.ndarray], ranker: InnovationRankerConfig,
    budgets: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    allowed = max(1, int(round(float(ranker.evaluation["quiet_false_candidates_per_map"]) * len(quiet_scores))))
    ranked_quiet = sorted((float(value) for score in quiet_scores for value in score), reverse=True)
    threshold = float(np.nextafter(ranked_quiet[allowed], np.inf))
    order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1]
    threshold_order = order[np.asarray(scores)[order] >= threshold]
    threshold_peaks = [(float(scores[i]), int(table.positions[i, 0]), int(table.positions[i, 1])) for i in threshold_order]
    threshold_matches = match_peaks_one_to_one(
        threshold_peaks, rows, float(ranker.evaluation["match_radius_px"])
    )[0]
    budget_rows, details = {}, {}
    for budget in budgets:
        selected = order[: int(budget)]
        peaks = [(float(scores[i]), int(table.positions[i, 0]), int(table.positions[i, 1])) for i in selected]
        matches, matched_peak_indices = match_peaks_one_to_one(
            peaks, rows, float(ranker.evaluation["match_radius_px"])
        )
        budget_rows[str(int(budget))] = {
            "matched": len(matches), "labels": len(rows),
            "recall": len(matches) / len(rows), "candidates": len(peaks),
        }
        details[str(int(budget))] = {
            "order_indices": selected.tolist(),
            "label_indices": sorted(int(match[0]) for match in matches),
            "matched_candidate_indices": sorted(int(selected[index]) for index in matched_peak_indices),
        }
    return {
        "labels": len(rows), "threshold": threshold,
        "threshold_matched": len(threshold_matches),
        "threshold_recall": len(threshold_matches) / len(rows),
        "threshold_candidates": len(threshold_order), "budgets": budget_rows,
    }, details


def _positive_negative(
    training_bursts: Sequence[int], feature_columns: Sequence[int],
    event_tables: Mapping[int, CandidateTable], quiet_tables: Sequence[CandidateTable],
    labels: list[dict[str, Any]], ranker: InnovationRankerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    columns = np.asarray(feature_columns, dtype=np.int64)
    positives = []
    for burst in training_bursts:
        table = event_tables[int(burst)]
        for row in _label_rows(labels, int(burst)):
            distance_sq = (
                (table.positions[:, 0] - float(row["x_px"])) ** 2
                + (table.positions[:, 1] - float(row["y_px"])) ** 2
            )
            choices = np.flatnonzero(distance_sq <= float(ranker.evaluation["match_radius_px"]) ** 2)
            if choices.size:
                local = np.max(table.features[choices][:, columns], axis=1)
                positives.append(table.features[int(choices[int(np.argmax(local))])])
    if not positives:
        raise RuntimeError("no positive proposals for ranker fit")
    negative = np.concatenate([table.features for table in quiet_tables])
    hardness = np.max(negative[:, columns], axis=1)
    count = min(512, len(negative))
    indices = np.argpartition(hardness, -count)[-count:]
    return np.asarray(positives), negative[indices]


def _feature_sets(feature_ids: Sequence[str], itl_ids: Sequence[str]) -> dict[str, tuple[str, ...]]:
    anchors = tuple(
        _feature_id(family, 11, 1.0) for family in FAMILIES
    )
    return {
        "separation": SEPARATION_FEATURES,
        "itl_anchor": ("carrier_signed", *anchors),
        "itl_all": ("carrier_signed", *itl_ids),
        "separation_itl_anchor": tuple(dict.fromkeys((*SEPARATION_FEATURES, *anchors))),
    }


def _ranker_grid(config: PatchInformationConfig) -> list[dict[str, Any]]:
    rows = []
    for feature_set, learning_rate, l2, maximum in itertools.product(
        config.ranker["feature_sets"], config.ranker["learning_rates"],
        config.ranker["l2_values"], config.ranker["maximum_total_weights"],
    ):
        rows.append({
            "feature_set": str(feature_set), "learning_rate": float(learning_rate),
            "l2": float(l2), "maximum_total_weight": float(maximum),
            "epochs": int(config.ranker["epochs"]),
        })
    for index, row in enumerate(rows):
        row["config_id"] = f"linear__{index:03d}"
    return rows


def _fit_model(
    definition: Mapping[str, Any], columns: Sequence[int], positive: np.ndarray,
    negative: np.ndarray,
) -> dict[str, Any]:
    auxiliary = [int(column) for column in columns if int(column) != 0]
    model = fit_bounded_pairwise_linear(
        positive, negative, carrier_column=0, auxiliary_columns=auxiliary,
        auxiliary_directions=[1.0] * len(auxiliary),
        learning_rate=float(definition["learning_rate"]),
        epochs=int(definition["epochs"]), l2=float(definition["l2"]),
        maximum_total=float(definition["maximum_total_weight"]),
    )
    model.update(config_id=definition["config_id"], feature_set=definition["feature_set"])
    return model


def _model_objective(rows: Sequence[Mapping[str, Any]], config: PatchInformationConfig) -> tuple[Any, ...]:
    primary = [str(int(value)) for value in config.screen["primary_budgets"]]
    values = [row["metrics"]["budgets"][budget]["recall"] for row in rows for budget in primary]
    return (
        float(np.mean(values)), float(np.min(values)),
        float(np.mean([row["metrics"]["budgets"]["58"]["recall"] for row in rows])),
        float(np.mean([row["metrics"]["threshold_recall"] for row in rows])),
        -sum(row["metrics"]["threshold_candidates"] for row in rows),
    )


def _nested_rankers(
    config: PatchInformationConfig, ranker: InnovationRankerConfig,
    feature_ids: Sequence[str], itl_ids: Sequence[str],
    quiet_tables: Sequence[CandidateTable], event_tables: Mapping[int, CandidateTable],
    labels: list[dict[str, Any]], progress: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    sets = _feature_sets(feature_ids, itl_ids)
    grid = _ranker_grid(config)
    budgets = [int(value) for value in config.screen["budgets"]]
    inner_rows, models, outer_by_set = [], {}, {name: [] for name in sets}
    for outer in BURSTS:
        training = [burst for burst in BURSTS if burst != outer]
        for feature_set, names in sets.items():
            columns = [feature_ids.index(name) for name in names]
            definitions = [row for row in grid if row["feature_set"] == feature_set]
            evaluations = []
            for definition in definitions:
                validation_rows = []
                for validation in training:
                    fit_bursts = [burst for burst in training if burst != validation]
                    positive, negative = _positive_negative(
                        fit_bursts, columns, event_tables, quiet_tables, labels, ranker
                    )
                    model = _fit_model(definition, columns, positive, negative)
                    quiet_scores = [score_bounded_pairwise_linear(table.features, model) for table in quiet_tables]
                    scores = score_bounded_pairwise_linear(event_tables[validation].features, model)
                    metrics, _ = _evaluate_scores(
                        scores, event_tables[validation], _label_rows(labels, validation),
                        quiet_scores, ranker, budgets,
                    )
                    row = {
                        "outer_burst": outer, "validation_burst": validation,
                        "training_bursts": fit_bursts, **definition, "metrics": metrics,
                        "loss_initial": model["loss_initial"], "loss_final": model["loss_final"],
                    }
                    inner_rows.append(row); validation_rows.append(row)
                evaluations.append((definition, validation_rows))
            selected, selected_rows = max(
                evaluations, key=lambda item: (*_model_objective(item[1], config), item[0]["config_id"])
            )
            positive, negative = _positive_negative(
                training, columns, event_tables, quiet_tables, labels, ranker
            )
            model = _fit_model(selected, columns, positive, negative)
            quiet_scores = [score_bounded_pairwise_linear(table.features, model) for table in quiet_tables]
            scores = score_bounded_pairwise_linear(event_tables[outer].features, model)
            metrics, detail = _evaluate_scores(
                scores, event_tables[outer], _label_rows(labels, outer),
                quiet_scores, ranker, budgets,
            )
            model_id = f"{feature_set}__burst{outer}"
            models[model_id] = model
            outer_by_set[feature_set].append({
                "held_out_burst": outer, "selected_config_id": selected["config_id"],
                "inner_objective": _model_objective(selected_rows, config),
                "metrics": metrics, "detail": detail,
            })
        _progress(progress, "nested_outer_complete", held_out_burst=outer)
    summaries = {}
    for feature_set, folds in outer_by_set.items():
        summaries[feature_set] = {
            "feature_set": feature_set, "folds": folds,
            "threshold_mean_recall": float(np.mean([row["metrics"]["threshold_recall"] for row in folds])),
            "threshold_event_candidates": sum(row["metrics"]["threshold_candidates"] for row in folds),
            "budget_mean_recall": {
                str(int(budget)): float(np.mean([
                    row["metrics"]["budgets"][str(int(budget))]["recall"] for row in folds
                ])) for budget in budgets
            },
        }
    return summaries, inner_rows, models


def _oracle_coverage(
    feature_maps: Mapping[str, Mapping[str, Any]], source_ids: Sequence[str],
    labels: list[dict[str, Any]], ranker: InnovationRankerConfig,
    budgets: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[int, set[int]]]:
    normalizers = _normalizers({feature_id: feature_maps[feature_id] for feature_id in source_ids})
    rows, fixed = [], {}
    for budget in budgets:
        for burst in BURSTS:
            burst_labels = _label_rows(labels, burst); recovered = set(); counts = {}
            for feature_id in source_ids:
                score = normalize_map(feature_maps[feature_id]["events"][burst], normalizers[feature_id], clip=8.0)
                peaks = extract_local_maxima(score, int(ranker.proposals["nms_distance_px"]), limit=int(budget))
                matches = match_peaks_one_to_one(peaks, burst_labels, float(ranker.evaluation["match_radius_px"]))[0]
                indices = {int(match[0]) for match in matches}; recovered |= indices; counts[feature_id] = len(indices)
            if int(budget) == 58:
                fixed[burst] = recovered
            rows.append({
                "source_budget": int(budget), "burst_id": burst, "labels": len(burst_labels),
                "union_recovered": len(recovered), "union_coverage": len(recovered) / len(burst_labels),
                "best_source": max(counts, key=counts.get), "best_source_recovered": max(counts.values()),
            })
    return rows, fixed


def _report(path: Path, metrics: Mapping[str, Any]) -> None:
    baseline = metrics["carrier_baseline"]["budget_mean_recall"]
    crossfit = metrics["crossfitted_fixed_lanes"]
    rankers = metrics["nested_rankers"]
    preferred = metrics["preferred_ranker"]
    lines = [
        f"# {metrics['experiment_id']}", "", "## Executive result", "",
        metrics["conclusion"], "", "## Fixed screens", "",
        "| Method | Budget 20 | Budget 40 | Budget 58 | Threshold recall | Candidates |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Standardized carrier | {baseline['20']:.3f} | {baseline['40']:.3f} | {baseline['58']:.3f} | {metrics['carrier_baseline']['threshold_mean_recall']:.3f} | {metrics['carrier_baseline']['threshold_event_candidates']} |",
    ]
    for kind in ("standalone", "boost", "gate"):
        row = crossfit[kind]
        lines.append(
            f"| Cross-fitted {kind} | {row['budget_mean_recall']['20']:.3f} | "
            f"{row['budget_mean_recall']['40']:.3f} | {row['budget_mean_recall']['58']:.3f} | "
            f"{row['threshold_mean_recall']:.3f} | {row['threshold_event_candidates']} |"
        )
    lines.extend(["", "## Nested candidate rankers", "", "| Feature set | Budget 20 | Budget 40 | Budget 58 | Threshold recall | Candidates |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for name, row in rankers.items():
        lines.append(
            f"| {name} | {row['budget_mean_recall']['20']:.3f} | {row['budget_mean_recall']['40']:.3f} | "
            f"{row['budget_mean_recall']['58']:.3f} | {row['threshold_mean_recall']:.3f} | {row['threshold_event_candidates']} |"
        )
    lines.extend([
        "", "## Interpretation", "",
        f"The preferred nested feature set is `{preferred}`. Selection prioritizes budgets 20 and 40; budget 58 is secondary.",
        "", "Quadratic information potential is the Gaussian-Parzen estimate underlying Renyi order-2 entropy. Cauchy--Schwarz divergence compares each current local density to its same-location quiet density. Local correntropy measures kernel similarity between the center pixel and its spatial patch.",
        "", "Unmatched event candidates remain scientifically unknown because the labels are sparse. Candidate burden is selectivity pressure, not precision.",
        "", "## Artifacts", "", "See `metrics.json`, `evaluation/`, `models/`, and `diagnostics/`.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config: PatchInformationConfig) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(config.resources["cpu_threads"])
    audit = _matching_preflight(config)
    ranker = InnovationRankerConfig.load(config.source_ranker_config)
    partial = Path(str(config.output_dir) + ".partial")
    partial.mkdir(parents=True)
    evaluation_dir, diagnostics_dir, models_dir = partial / "evaluation", partial / "diagnostics", partial / "models"
    for directory in (evaluation_dir, diagnostics_dir, models_dir):
        directory.mkdir()
    _atomic_json(partial / "preflight.json", audit)
    _atomic_json(partial / "config.resolved.json", config.to_dict())
    progress = partial / "progress.jsonl"
    started = time.time()
    labels = label_core.load_labels(ranker.labels_tsv)
    base_config = FeatureUtilityConfig.load(ranker.feature_root / "config.resolved.json")
    v5_maps, _ = _generate_v5_maps(ranker, labels, base_config, progress)
    itl_maps = _generate_itl_maps(config, ranker, labels, base_config, progress)
    carrier_maps = v5_maps["carrier_signed"]
    carrier_baseline, screen_rows = _screen(config, ranker, itl_maps, carrier_maps, labels)
    crossfitted = {kind: _crossfit_screen(screen_rows, kind, config) for kind in ("standalone", "boost", "gate")}
    _atomic_json(evaluation_dir / "fixed_screen.json", {"carrier_baseline": carrier_baseline, "rows": screen_rows, "crossfitted": crossfitted})

    itl_ids = _itl_feature_ids(config)
    all_maps = {**v5_maps, **itl_maps}
    feature_ids = tuple(dict.fromkeys((*SEPARATION_FEATURES, *itl_ids)))
    proposal_ids = tuple(dict.fromkeys((*V5_PROPOSAL_SOURCE_IDS, *itl_ids)))
    normalizers, quiet_tables, event_tables = _candidate_tables(all_maps, feature_ids, proposal_ids, ranker)
    _atomic_json(evaluation_dir / "candidate_inventory.json", {
        "feature_ids": list(feature_ids), "proposal_source_ids": list(proposal_ids),
        "quiet_candidate_counts": [len(table.positions) for table in quiet_tables],
        "event_candidate_counts": {str(burst): len(table.positions) for burst, table in event_tables.items()},
        "normalizers": {key: {"quiet_median": value[0], "quiet_scale": value[1]} for key, value in normalizers.items()},
    })
    oracle_rows, augmented_fixed = _oracle_coverage(
        all_maps, proposal_ids, labels, ranker,
        [int(value) for value in config.screen["oracle_source_budgets"]],
    )
    _atomic_json(evaluation_dir / "oracle_coverage.json", {"rows": oracle_rows})
    nested, inner_rows, models = _nested_rankers(
        config, ranker, feature_ids, itl_ids, quiet_tables, event_tables, labels, progress
    )
    _atomic_json(evaluation_dir / "inner_fine_tuning.json", {"model_fit_count": config.inner_fit_count, "rows": inner_rows})
    _atomic_json(evaluation_dir / "nested_rankers.json", nested)
    for model_id, model in models.items():
        _atomic_json(models_dir / f"{model_id}.json", model)
    preferred = max(
        nested,
        key=lambda name: (
            np.mean([nested[name]["budget_mean_recall"]["20"], nested[name]["budget_mean_recall"]["40"]]),
            min(nested[name]["budget_mean_recall"]["20"], nested[name]["budget_mean_recall"]["40"]),
            nested[name]["budget_mean_recall"]["58"],
            -nested[name]["threshold_event_candidates"], name,
        ),
    )
    neuron_rows = []
    for fold in nested[preferred]["folds"]:
        burst = int(fold["held_out_burst"]); rows = _label_rows(labels, burst)
        for index, row in enumerate(rows):
            neuron_rows.append({
                "burst_id": burst, "roi_identity": row["roi_identity"],
                "x_px": row["x_px"], "y_px": row["y_px"],
                **{f"recovered_budget_{budget}": index in fold["detail"][str(budget)]["label_indices"] for budget in config.screen["budgets"]},
                "augmented_oracle_recoverable_at_58_per_source": index in augmented_fixed[burst],
            })
    _atomic_tsv(evaluation_dir / "per_neuron_audit.tsv", neuron_rows)

    primary = [str(value) for value in config.screen["primary_budgets"]]
    posthoc = sorted(
        [row for row in screen_rows if row["kind"] == "standalone"],
        key=lambda row: (
            np.mean([row["budget_mean_recall"][value] for value in primary]),
            row["budget_mean_recall"]["58"], row["config_id"],
        ), reverse=True,
    )[: int(config.visualization["tiff_feature_count"])]
    pages = [itl_maps[row["feature_id"]]["events"][burst] for row in posthoc for burst in BURSTS]
    feature_tiff = _write_score_tiff(
        diagnostics_dir / "top_itl_feature_maps.tif", pages,
        compression=str(config.visualization["compression"]),
        description={"feature_ids": [row["feature_id"] for row in posthoc], "page_order": "feature-major; bursts 1-4"},
    )
    score_pages, overlay_pages = [], []
    structure = np.load(ranker.feature_root / "structure_unit.npy", mmap_mode="r", allow_pickle=False)
    for fold in nested[preferred]["folds"]:
        burst = int(fold["held_out_burst"]); model = models[f"{preferred}__burst{burst}"]
        scores = score_bounded_pairwise_linear(event_tables[burst].features, model)
        score_pages.append(_score_map(scores, event_tables[burst], structure.shape, float(ranker.visualization["score_sigma_px"])))
        detail58 = {
            "fixed_order_indices": fold["detail"]["58"]["order_indices"],
            "fixed_matched_candidate_indices": fold["detail"]["58"]["matched_candidate_indices"],
            "fixed_label_indices": fold["detail"]["58"]["label_indices"],
        }
        overlay_pages.append(_overlay_page(
            np.asarray(structure), event_tables[burst], scores, detail58,
            _label_rows(labels, burst), int(ranker.visualization["overlay_candidate_limit"]),
        ))
    ranker_score_tiff = _write_score_tiff(
        diagnostics_dir / "preferred_ranker_scores.tif", score_pages,
        compression=str(config.visualization["compression"]),
        description={"feature_set": preferred, "pages": "bursts 1-4"},
    )
    overlay_tiff = _write_overlay_tiff(
        diagnostics_dir / "preferred_ranker_overlay.tif", overlay_pages,
        str(config.visualization["compression"]),
    )
    v5_metrics = json.loads((config.source_ranker_root / "metrics.json").read_text(encoding="utf-8"))
    oracle58 = [row for row in oracle_rows if int(row["source_budget"]) == 58]
    oracle_mean = float(np.mean([row["union_coverage"] for row in oracle58]))
    conclusion = (
        f"Cross-fitted ITL screens and {config.inner_fit_count + config.outer_refit_count} "
        f"nested linear fits completed. The preferred ranker was {preferred}; "
        f"its budget-20/40/58 recalls were "
        f"{nested[preferred]['budget_mean_recall']['20']:.3f}, "
        f"{nested[preferred]['budget_mean_recall']['40']:.3f}, and "
        f"{nested[preferred]['budget_mean_recall']['58']:.3f}. The augmented "
        f"58-per-source oracle union covered {oracle_mean:.3f} of known labels."
    )
    metrics = {
        "schema_version": 1, "experiment_id": config.experiment_id, "status": "completed",
        "feature_count": config.feature_count, "fixed_lane_count": config.fixed_lane_count,
        "inner_fit_count": config.inner_fit_count, "outer_refit_count": config.outer_refit_count,
        "carrier_baseline": carrier_baseline, "crossfitted_fixed_lanes": crossfitted,
        "nested_rankers": nested, "preferred_ranker": preferred,
        "augmented_oracle_58_mean_coverage": oracle_mean,
        "v5_oracle_58_mean_coverage": v5_metrics["oracle_summary"]["fixed_budget_mean_coverage"],
        "diagnostics": {"top_itl_feature_maps": feature_tiff, "preferred_scores": ranker_score_tiff, "preferred_overlay": overlay_tiff},
        "conclusion": conclusion,
        "precision_contract": "Sparse unmatched event candidates are unknown; candidate burden is not precision.",
        "elapsed_seconds": time.time() - started,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    _atomic_json(partial / "metrics.json", metrics)
    _report(partial / "REPORT.md", metrics)
    _atomic_json(partial / "run_state.json", {
        "status": "completed", "elapsed_seconds": metrics["elapsed_seconds"],
        "max_rss_mib": metrics["max_rss_mib"], "feature_count": config.feature_count,
        "fixed_lane_count": config.fixed_lane_count,
        "model_fit_count": config.inner_fit_count + config.outer_refit_count,
        "diagnostic_tiff_count": 3,
    })
    partial.replace(config.output_dir)
    return metrics
