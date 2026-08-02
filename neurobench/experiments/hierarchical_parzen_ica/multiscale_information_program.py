"""Guarded multiscale Cauchy--Schwarz experiment on Spon Ca Burst."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any, Mapping, Sequence

import numpy as np

from neurobench.algorithms.patch_information import (
    cauchy_schwarz_divergence_tensor,
    local_center_annulus_histograms_tensor,
    local_histogram_pyramid_tensor,
    local_histogram_tensor,
)
from neurobench.algorithms.proposal_ranking import normalize_map, robust_map_normalizer
from neurobench.experiments.learnable_contrast import core as label_core
from neurobench.experiments.pairwise_separation.evaluation import (
    QUIET_DURATIONS,
    QUIET_STARTS,
    event_intervals,
)
from neurobench.metrics.sparse_detection import (
    extract_local_maxima,
    match_peaks_one_to_one,
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
    PROPOSAL_SOURCE_IDS as V5_PROPOSAL_SOURCE_IDS,
    InnovationRankerConfig,
)
from .innovation_ranker_program import (
    _generate_maps as _generate_v5_maps,
    _write_score_tiff,
)
from .multiscale_information_config import MultiscaleInformationConfig
from .patch_information_config import PatchInformationConfig
from .patch_information_program import (
    BURSTS,
    _candidate_tables,
    _combine_maps,
    _evaluate_maps,
    _evaluate_scores,
    _label_rows,
    _oracle_coverage,
    _pool_values,
    _unit_maps,
)


def _token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _base_id(patch: int, bandwidth: float) -> str:
    return f"cs_quiet__p{int(patch)}__bw{_token(bandwidth)}"


def _feature_inventory(
    config: MultiscaleInformationConfig,
) -> tuple[tuple[str, ...], dict[str, str]]:
    patches = [int(value) for value in config.multiscale["patch_sizes_px"]]
    bandwidths = [float(value) for value in config.multiscale["kernel_bandwidths_z"]]
    ids: list[str] = []
    families: dict[str, str] = {}

    def add(feature_id: str, family: str) -> None:
        ids.append(feature_id)
        families[feature_id] = family

    for bandwidth in bandwidths:
        for patch in patches:
            add(_base_id(patch, bandwidth), "single_scale")
        add(f"ms_max__bw{_token(bandwidth)}", "scale_max")
        for tau in config.fusions["softmax_temperatures"]:
            add(
                f"ms_lme__bw{_token(bandwidth)}__tau{_token(tau)}",
                "soft_scale_selection",
            )
        for first, second in zip(patches[:-1], patches[1:]):
            add(
                f"ms_agree__p{first}_p{second}__bw{_token(bandwidth)}",
                "adjacent_scale_agreement",
            )
        first, second = [int(value) for value in config.fusions["contrast_pair"]]
        for authority in config.fusions["contrast_authorities"]:
            add(
                f"ms_contrast__p{first}_p{second}__bw{_token(bandwidth)}"
                f"__a{_token(authority)}",
                "compact_broad_contrast",
            )
        for center, outer in config.multiscale["center_surround_pairs"]:
            add(
                f"center_annulus__p{int(center)}_p{int(outer)}"
                f"__bw{_token(bandwidth)}",
                "center_annulus",
            )
    return tuple(ids), families


def preflight(
    config: MultiscaleInformationConfig, *, write_artifacts: bool = True
) -> dict[str, Any]:
    patch = PatchInformationConfig.load(config.source_patch_config)
    ranker = InnovationRankerConfig.load(patch.source_ranker_config)
    carrier_path = ranker.feature_root / "features" / "carrier_signed.npy"
    inputs = (
        config.source_patch_config,
        config.source_patch_root / "metrics.json",
        patch.source_ranker_config,
        patch.source_ranker_root / "metrics.json",
        carrier_path,
        ranker.labels_tsv,
    )
    missing = [str(path) for path in inputs if not path.is_file()]
    carrier_shape = None
    labels: list[dict[str, Any]] = []
    source_valid = labels_valid = finite = False
    if not missing:
        source_metrics = json.loads(
            (config.source_patch_root / "metrics.json").read_text(encoding="utf-8")
        )
        carrier = np.load(carrier_path, mmap_mode="r", allow_pickle=False)
        carrier_shape = list(carrier.shape)
        expected = (
            int(ranker.frames["review_end_ui"])
            - int(ranker.frames["review_start_ui"]) + 1
        )
        source_valid = bool(
            source_metrics.get("status") == "completed"
            and carrier.ndim == 3
            and len(carrier) == expected
        )
        finite = bool(np.isfinite(carrier[::32, ::16, ::16]).all())
        labels = label_core.load_labels(ranker.labels_tsv)
        labels_valid = bool(
            len(labels) == 79
            and len({row["roi_identity"] for row in labels}) == 27
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
    feature_ids, _ = _feature_inventory(config)
    estimated_ram = 6144.0
    estimated_gpu = 3072.0
    estimated_output = 512.0
    probe = config.output_dir.parent
    while not probe.exists():
        probe = probe.parent
    free_disk = shutil.disk_usage(probe).free / 2**20
    gates = {
        "inputs_exist": not missing,
        "source_patch_completed": source_valid,
        "carrier_finite": finite,
        "labels_valid": labels_valid,
        "feature_inventory_valid": len(feature_ids) == config.feature_count == 42,
        "lane_count_valid": config.lane_count == 168,
        "output_absent": not config.output_dir.exists(),
        "partial_output_absent": not Path(str(config.output_dir) + ".partial").exists(),
        "preflight_separate_from_output": config.preflight_dir != config.output_dir,
        "ram_cap_sufficient": estimated_ram <= int(config.resources["max_ram_mib"]),
        "available_ram_sufficient": estimated_ram <= _available_ram_mib(),
        "gpu_available": gpu["available"],
        "gpu_cap_sufficient": estimated_gpu <= int(config.resources["max_gpu_memory_mib"]),
        "live_gpu_sufficient": estimated_gpu <= gpu.get("free_mib", 0),
        "output_cap_sufficient": estimated_output <= int(config.resources["max_output_mib"]),
        "disk_headroom_sufficient": free_disk >= estimated_output + int(config.resources["min_free_disk_mib"]),
    }
    payload = {
        "schema_version": 1,
        "kind": "read_only_multiscale_patch_information_preflight",
        "experiment_id": config.experiment_id,
        "ready": all(gates.values()),
        "gates": gates,
        "carrier_shape": carrier_shape,
        "label_rows": len(labels),
        "roi_identities": len({row["roi_identity"] for row in labels}),
        "design": {
            "base_feature_count": config.base_feature_count,
            "fused_feature_count": config.fused_feature_count,
            "feature_count": config.feature_count,
            "feature_ids": list(feature_ids),
            "native_lane_count": config.lane_count,
            "identical_proposal_lane_count": config.lane_count,
            "evaluated_lane_count": 2 * config.lane_count,
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
            "All multiscale maps are quiet-calibrated without labels. Native "
            "peak evaluation and identical-v5-proposal ranking are reported "
            "separately. Sparse unmatched candidates remain unknown."
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
        raise RuntimeError(f"multiscale preflight failed: {payload}")
    return payload


def _matching_preflight(config: MultiscaleInformationConfig) -> dict[str, Any]:
    audit = json.loads(
        (config.preflight_dir / "preflight.json").read_text(encoding="utf-8")
    )
    resolved = json.loads(
        (config.preflight_dir / "config.resolved.json").read_text(encoding="utf-8")
    )
    if not audit.get("ready") or resolved != config.to_dict():
        raise RuntimeError("run requires a matching ready preflight")
    if config.output_dir.exists() or Path(str(config.output_dir) + ".partial").exists():
        raise FileExistsError("completed or partial output already exists")
    return audit


def _empty_segments(
    quiet_count: int,
    intervals: Mapping[int, tuple[int, int]],
    shape: tuple[int, int],
) -> dict[str, Any]:
    return {
        "quiet": np.empty((quiet_count, *shape), dtype=np.float16),
        "events": {
            int(burst): np.empty((stop - start, *shape), dtype=np.float16)
            for burst, (start, stop) in intervals.items()
        },
    }


def _compute_raw_maps(
    config: MultiscaleInformationConfig,
    patch: PatchInformationConfig,
    ranker: InnovationRankerConfig,
    labels: list[dict[str, Any]],
    progress: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    import torch

    carrier = np.load(
        ranker.feature_root / "features" / "carrier_signed.npy",
        mmap_mode="r", allow_pickle=False,
    )
    device = torch.device(str(config.resources["device"]))
    centers = tuple(float(value) for value in patch.itl["bin_centers_z"])
    patches = tuple(int(value) for value in config.multiscale["patch_sizes_px"])
    bandwidths = tuple(float(value) for value in config.multiscale["kernel_bandwidths_z"])
    batch_size = int(config.multiscale["frame_batch_size"])
    quiet_count = int(ranker.frames["quiet_count"])
    intervals = event_intervals(labels, int(ranker.frames["review_start_ui"]))
    shape = tuple(int(value) for value in carrier.shape[1:])
    quiet_histograms = {}
    with torch.inference_mode():
        for patch_size in patches:
            total = torch.zeros(
                (len(centers), *shape), dtype=torch.float32, device=device
            )
            for start in range(0, quiet_count, batch_size):
                stop = min(quiet_count, start + batch_size)
                frames = torch.as_tensor(
                    np.asarray(carrier[start:stop], dtype=np.float32), device=device
                )
                total += local_histogram_tensor(
                    frames, centers=centers, patch_size_px=patch_size
                ).sum(dim=0)
            quiet_histograms[patch_size] = total / float(quiet_count)

    raw = {
        _base_id(patch_size, bandwidth): _empty_segments(
            quiet_count, intervals, shape
        )
        for bandwidth in bandwidths for patch_size in patches
    }
    for bandwidth in bandwidths:
        for center, outer in config.multiscale["center_surround_pairs"]:
            feature_id = (
                f"center_annulus__p{int(center)}_p{int(outer)}"
                f"__bw{_token(bandwidth)}"
            )
            raw[feature_id] = _empty_segments(quiet_count, intervals, shape)

    def ranges():
        yield "quiet", None, 0, quiet_count
        for burst, (start, stop) in intervals.items():
            yield "events", int(burst), start, stop

    compute_started = time.perf_counter()
    with torch.inference_mode():
        for destination, burst, source_start, source_stop in ranges():
            for local_start in range(0, source_stop - source_start, batch_size):
                local_stop = min(source_stop - source_start, local_start + batch_size)
                start = source_start + local_start
                stop = source_start + local_stop
                frames = torch.as_tensor(
                    np.asarray(carrier[start:stop], dtype=np.float32), device=device
                )
                pyramid = local_histogram_pyramid_tensor(
                    frames, centers=centers, patch_sizes_px=patches
                )
                for patch_size, histogram in pyramid.items():
                    quiet = quiet_histograms[patch_size][None].expand_as(histogram)
                    for bandwidth in bandwidths:
                        values = cauchy_schwarz_divergence_tensor(
                            histogram, quiet, centers=centers, bandwidth=bandwidth
                        ).to("cpu").numpy().astype(np.float16)
                        target = raw[_base_id(patch_size, bandwidth)][destination]
                        if burst is not None:
                            target = target[burst]
                        target[local_start:local_stop] = values
                for center, outer in config.multiscale["center_surround_pairs"]:
                    center_histogram, annulus_histogram = (
                        local_center_annulus_histograms_tensor(
                            frames,
                            centers=centers,
                            center_patch_px=int(center),
                            outer_patch_px=int(outer),
                        )
                    )
                    for bandwidth in bandwidths:
                        values = cauchy_schwarz_divergence_tensor(
                            center_histogram,
                            annulus_histogram,
                            centers=centers,
                            bandwidth=bandwidth,
                        ).to("cpu").numpy().astype(np.float16)
                        feature_id = (
                            f"center_annulus__p{int(center)}_p{int(outer)}"
                            f"__bw{_token(bandwidth)}"
                        )
                        target = raw[feature_id][destination]
                        if burst is not None:
                            target = target[burst]
                        target[local_start:local_stop] = values
            _progress(
                progress, "multiscale_raw_range_complete",
                range_kind=destination, burst_id=burst,
            )
    compute_seconds = time.perf_counter() - compute_started
    for feature_id, values in raw.items():
        sample = np.asarray(values["quiet"][::8, ::8, ::8], dtype=np.float32)
        if not np.isfinite(sample).all():
            raise RuntimeError(f"non-finite multiscale map: {feature_id}")
    benchmark = {
        "computed_frames": quiet_count + sum(stop - start for start, stop in intervals.values()),
        "full_bank_compute_seconds": compute_seconds,
        "full_bank_batched_frames_per_second": (
            quiet_count + sum(stop - start for start, stop in intervals.values())
        ) / compute_seconds,
        "frame_batch_size": batch_size,
        "real_time_status": (
            "offline_throughput_only; selected-map single-frame latency requires "
            "a separate deployment benchmark"
        ),
    }
    del quiet_histograms
    torch.cuda.empty_cache()
    return raw, benchmark


def _calibrated_frames(values: Mapping[str, Any]) -> dict[str, Any]:
    quiet = np.asarray(values["quiet"], dtype=np.float32)
    baseline = np.median(quiet, axis=0).astype(np.float32)
    low, high = np.percentile(quiet[:, ::4, ::4], [1.0, 99.9])
    scale = max(float(high - low), 1e-6)

    def apply(array: np.ndarray) -> np.ndarray:
        return np.maximum(
            (np.asarray(array, dtype=np.float32) - baseline[None]) / scale, 0
        )

    return {
        "quiet": apply(values["quiet"]),
        "events": {int(burst): apply(array) for burst, array in values["events"].items()},
    }


def _pool_evidence(values: Mapping[str, Any], tau: float) -> dict[str, Any]:
    return {
        "quiet": [
            temporal_pool(values["quiet"][start:start + duration], f"lme{tau}")
            for start, duration in zip(QUIET_STARTS, QUIET_DURATIONS)
        ],
        "events": {
            int(burst): temporal_pool(array, f"lme{tau}")
            for burst, array in values["events"].items()
        },
    }


def _binary_frames(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    operation,
) -> dict[str, Any]:
    return {
        "quiet": operation(first["quiet"], second["quiet"]),
        "events": {
            burst: operation(first["events"][burst], second["events"][burst])
            for burst in BURSTS
        },
    }


def _fuse_maps(
    config: MultiscaleInformationConfig,
    patch: PatchInformationConfig,
    ranker: InnovationRankerConfig,
    raw: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    base_config = FeatureUtilityConfig.load(ranker.feature_root / "config.resolved.json")
    tau_pool = float(base_config.evaluation["temporal_pool_tau"])
    patches = [int(value) for value in config.multiscale["patch_sizes_px"]]
    bandwidths = [float(value) for value in config.multiscale["kernel_bandwidths_z"]]
    result: dict[str, dict[str, Any]] = {}
    for bandwidth in bandwidths:
        evidence = {
            patch_size: _calibrated_frames(raw[_base_id(patch_size, bandwidth)])
            for patch_size in patches
        }
        for patch_size in patches:
            feature_id = _base_id(patch_size, bandwidth)
            result[feature_id] = _pool_values(
                raw[feature_id]["quiet"], raw[feature_id]["events"], tau_pool
            )
        maximum = evidence[patches[0]]
        for patch_size in patches[1:]:
            maximum = _binary_frames(maximum, evidence[patch_size], np.maximum)
        result[f"ms_max__bw{_token(bandwidth)}"] = _pool_evidence(maximum, tau_pool)
        for temperature in config.fusions["softmax_temperatures"]:
            value = {
                "quiet": evidence[patches[0]]["quiet"] / float(temperature),
                "events": {
                    burst: evidence[patches[0]]["events"][burst] / float(temperature)
                    for burst in BURSTS
                },
            }
            for patch_size in patches[1:]:
                value = _binary_frames(
                    value,
                    {
                        "quiet": evidence[patch_size]["quiet"] / float(temperature),
                        "events": {
                            burst: evidence[patch_size]["events"][burst] / float(temperature)
                            for burst in BURSTS
                        },
                    },
                    np.logaddexp,
                )
            offset = math.log(len(patches))
            value = {
                "quiet": float(temperature) * (value["quiet"] - offset),
                "events": {
                    burst: float(temperature) * (value["events"][burst] - offset)
                    for burst in BURSTS
                },
            }
            result[
                f"ms_lme__bw{_token(bandwidth)}__tau{_token(temperature)}"
            ] = _pool_evidence(value, tau_pool)
        for first, second in zip(patches[:-1], patches[1:]):
            agreement = _binary_frames(
                evidence[first], evidence[second],
                lambda a, b: np.sqrt(np.maximum(a * b, 0.0)),
            )
            result[
                f"ms_agree__p{first}_p{second}__bw{_token(bandwidth)}"
            ] = _pool_evidence(agreement, tau_pool)
        first, second = [int(value) for value in config.fusions["contrast_pair"]]
        for authority in config.fusions["contrast_authorities"]:
            contrast = _binary_frames(
                evidence[first], evidence[second],
                lambda a, b, weight=float(authority): np.maximum(a - weight * b, 0.0),
            )
            result[
                f"ms_contrast__p{first}_p{second}__bw{_token(bandwidth)}"
                f"__a{_token(authority)}"
            ] = _pool_evidence(contrast, tau_pool)
        del evidence, maximum
    for bandwidth in bandwidths:
        for center, outer in config.multiscale["center_surround_pairs"]:
            feature_id = (
                f"center_annulus__p{int(center)}_p{int(outer)}"
                f"__bw{_token(bandwidth)}"
            )
            result[feature_id] = _pool_values(
                raw[feature_id]["quiet"], raw[feature_id]["events"], tau_pool
            )
    feature_ids, _ = _feature_inventory(config)
    if set(result) != set(feature_ids):
        raise RuntimeError("multiscale fused feature inventory mismatch")
    return result


def _lane_rows(
    config: MultiscaleInformationConfig,
    maps: Mapping[str, Mapping[str, Any]],
    carrier: Mapping[str, Any],
    labels: list[dict[str, Any]],
    ranker: InnovationRankerConfig,
    families: Mapping[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    budgets = [int(value) for value in config.evaluation["budgets"]]
    baseline = _evaluate_maps("carrier_native", carrier, labels, ranker, budgets)
    rows = []
    for feature_id, values in maps.items():
        standalone = _evaluate_maps(
            f"standalone__{feature_id}", values, labels, ranker, budgets
        )
        rows.append({
            "kind": "standalone", "family": families[feature_id],
            "feature_id": feature_id, "value": None, **standalone,
        })
        for weight in config.fusions["carrier_boosts"]:
            result = _evaluate_maps(
                f"boost__{feature_id}__{_token(weight)}",
                _combine_maps(carrier, values, kind="boost", value=float(weight)),
                labels, ranker, budgets,
            )
            rows.append({
                "kind": "boost", "family": families[feature_id],
                "feature_id": feature_id, "value": float(weight), **result,
            })
    return baseline, rows


def _selection_key(
    row: Mapping[str, Any], held_out: int, config: MultiscaleInformationConfig
) -> tuple[Any, ...]:
    training = [fold for fold in row["folds"] if int(fold["burst_id"]) != held_out]
    primary = [str(int(value)) for value in config.evaluation["primary_budgets"]]
    values = [fold["budgets"][budget]["recall"] for fold in training for budget in primary]
    return (
        float(np.mean(values)), float(np.min(values)),
        float(np.mean([fold["budgets"]["58"]["recall"] for fold in training])),
        float(np.mean([fold["threshold_recall"] for fold in training])),
        -sum(int(fold["threshold_candidates"]) for fold in training),
        str(row["config_id"]),
    )


def _crossfit(
    rows: Sequence[Mapping[str, Any]],
    config: MultiscaleInformationConfig,
    *,
    kind: str | None = None,
    family: str | None = None,
) -> dict[str, Any]:
    candidates = [
        row for row in rows
        if (kind is None or row["kind"] == kind)
        and (family is None or row["family"] == family)
    ]
    folds = []
    for held_out in BURSTS:
        selected = max(candidates, key=lambda row: _selection_key(row, held_out, config))
        held = next(
            fold for fold in selected["folds"] if int(fold["burst_id"]) == held_out
        )
        folds.append({
            "held_out_burst": held_out,
            "selected_config_id": selected["config_id"],
            "selected_feature_id": selected["feature_id"],
            "selected_family": selected["family"],
            "selected_kind": selected["kind"],
            "selected_value": selected["value"],
            **held,
        })
    return {
        "kind": kind or "all",
        "family": family or "all",
        "folds": folds,
        "threshold_mean_recall": float(np.mean([row["threshold_recall"] for row in folds])),
        "threshold_event_candidates": sum(int(row["threshold_candidates"]) for row in folds),
        "budget_mean_recall": {
            str(int(budget)): float(np.mean([
                row["budgets"][str(int(budget))]["recall"] for row in folds
            ])) for budget in config.evaluation["budgets"]
        },
    }


def _identical_proposal_rows(
    config: MultiscaleInformationConfig,
    all_maps: Mapping[str, Mapping[str, Any]],
    feature_ids: Sequence[str],
    families: Mapping[str, str],
    labels: list[dict[str, Any]],
    ranker: InnovationRankerConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    table_features = ("carrier_signed", *feature_ids)
    normalizers, quiet_tables, event_tables = _candidate_tables(
        all_maps, table_features, V5_PROPOSAL_SOURCE_IDS, ranker
    )
    budgets = [int(value) for value in config.evaluation["budgets"]]

    def evaluate(config_id: str, scores_by_burst, quiet_scores):
        folds = []
        for burst in BURSTS:
            metrics, detail = _evaluate_scores(
                scores_by_burst[burst], event_tables[burst],
                _label_rows(labels, burst), quiet_scores, ranker, budgets,
            )
            folds.append({"burst_id": burst, **metrics, "detail": detail})
        return {
            "config_id": config_id,
            "folds": folds,
            "threshold_mean_recall": float(np.mean([row["threshold_recall"] for row in folds])),
            "threshold_event_candidates": sum(row["threshold_candidates"] for row in folds),
            "budget_mean_recall": {
                str(budget): float(np.mean([
                    row["budgets"][str(budget)]["recall"] for row in folds
                ])) for budget in budgets
            },
        }

    carrier_column = table_features.index("carrier_signed")
    carrier_quiet = [table.features[:, carrier_column] for table in quiet_tables]
    carrier_event = {
        burst: table.features[:, carrier_column] for burst, table in event_tables.items()
    }
    baseline = evaluate("carrier_same_v5_union", carrier_event, carrier_quiet)
    rows = []
    for feature_id in feature_ids:
        column = table_features.index(feature_id)
        feature_quiet = [table.features[:, column] for table in quiet_tables]
        feature_event = {
            burst: table.features[:, column] for burst, table in event_tables.items()
        }
        standalone = evaluate(
            f"standalone__{feature_id}", feature_event, feature_quiet
        )
        rows.append({
            "kind": "standalone", "family": families[feature_id],
            "feature_id": feature_id, "value": None, **standalone,
        })
        for weight in config.fusions["carrier_boosts"]:
            quiet_scores = [
                carrier + float(weight) * feature
                for carrier, feature in zip(carrier_quiet, feature_quiet)
            ]
            event_scores = {
                burst: carrier_event[burst] + float(weight) * feature_event[burst]
                for burst in BURSTS
            }
            result = evaluate(
                f"boost__{feature_id}__{_token(weight)}",
                event_scores, quiet_scores,
            )
            rows.append({
                "kind": "boost", "family": families[feature_id],
                "feature_id": feature_id, "value": float(weight), **result,
            })
    inventory = {
        "feature_ids": list(table_features),
        "proposal_source_ids": list(V5_PROPOSAL_SOURCE_IDS),
        "quiet_candidate_counts": [len(table.positions) for table in quiet_tables],
        "event_candidate_counts": {
            str(burst): len(table.positions) for burst, table in event_tables.items()
        },
        "normalizers": {
            key: {"quiet_median": value[0], "quiet_scale": value[1]}
            for key, value in normalizers.items()
        },
    }
    return baseline, rows, inventory


def _quota_peaks(
    carrier_map: np.ndarray,
    feature_map: np.ndarray,
    carrier_normalizer: tuple[float, float],
    feature_normalizer: tuple[float, float],
    *,
    budget: int,
    carrier_fraction: float,
    ranker: InnovationRankerConfig,
) -> list[tuple[float, int, int]]:
    carrier = normalize_map(carrier_map, carrier_normalizer, clip=8.0)
    feature = normalize_map(feature_map, feature_normalizer, clip=8.0)
    distance = int(ranker.proposals["nms_distance_px"])
    carrier_peaks = extract_local_maxima(carrier, distance, limit=500)
    feature_peaks = extract_local_maxima(feature, distance, limit=500)
    carrier_quota = int(round(int(budget) * float(carrier_fraction)))
    feature_quota = int(budget) - carrier_quota
    candidates = [
        *((score, x, y, "carrier") for score, x, y in carrier_peaks[:carrier_quota]),
        *((score, x, y, "feature") for score, x, y in feature_peaks[:feature_quota]),
    ]
    remainder = [
        *((score, x, y, "carrier") for score, x, y in carrier_peaks[carrier_quota:]),
        *((score, x, y, "feature") for score, x, y in feature_peaks[feature_quota:]),
    ]
    candidates.sort(reverse=True)
    remainder.sort(reverse=True)
    selected: list[tuple[float, int, int]] = []
    radius_sq = float(ranker.proposals["dedupe_radius_px"]) ** 2
    for pool in (candidates, remainder):
        for score, x, y, _ in pool:
            if all((x - sx) ** 2 + (y - sy) ** 2 > radius_sq for _, sx, sy in selected):
                selected.append((float(score), int(x), int(y)))
                if len(selected) == int(budget):
                    return selected
    return selected


def _quota_evaluation(
    config: MultiscaleInformationConfig,
    selected: Mapping[str, Any],
    maps: Mapping[str, Mapping[str, Any]],
    carrier: Mapping[str, Any],
    labels: list[dict[str, Any]],
    ranker: InnovationRankerConfig,
) -> dict[str, Any]:
    carrier_normalizer = robust_map_normalizer(carrier["quiet"])
    feature_normalizers = {
        feature_id: robust_map_normalizer(values["quiet"])
        for feature_id, values in maps.items()
    }
    folds = []
    for selection in selected["folds"]:
        burst = int(selection["held_out_burst"])
        feature_id = selection["selected_feature_id"]
        rows = _label_rows(labels, burst)
        budgets = {}
        for budget in config.evaluation["budgets"]:
            peaks = _quota_peaks(
                carrier["events"][burst], maps[feature_id]["events"][burst],
                carrier_normalizer, feature_normalizers[feature_id],
                budget=int(budget),
                carrier_fraction=float(config.evaluation["quota_carrier_fraction"]),
                ranker=ranker,
            )
            matches = match_peaks_one_to_one(
                peaks, rows, float(ranker.evaluation["match_radius_px"])
            )[0]
            budgets[str(int(budget))] = {
                "candidates": len(peaks), "matched": len(matches),
                "labels": len(rows), "recall": len(matches) / len(rows),
                "label_indices": sorted(int(match[0]) for match in matches),
            }
        folds.append({
            "held_out_burst": burst, "selected_feature_id": feature_id,
            "selected_family": selection["selected_family"], "budgets": budgets,
        })
    return {
        "folds": folds,
        "budget_mean_recall": {
            str(int(budget)): float(np.mean([
                fold["budgets"][str(int(budget))]["recall"] for fold in folds
            ])) for budget in config.evaluation["budgets"]
        },
    }


def _report(path: Path, metrics: Mapping[str, Any]) -> None:
    native = metrics["crossfitted_native_all"]
    identical = metrics["crossfitted_identical_all"]
    quota = metrics["quota_union"]
    carrier = metrics["carrier_native"]["budget_mean_recall"]
    lines = [
        f"# {metrics['experiment_id']}", "", "## Executive result", "",
        metrics["conclusion"], "", "## Leakage-safe comparisons", "",
        "| Method | Budget 20 | Budget 40 | Budget 58 |",
        "| --- | ---: | ---: | ---: |",
        f"| Native carrier | {carrier['20']:.3f} | {carrier['40']:.3f} | {carrier['58']:.3f} |",
        f"| Selected multiscale native peaks | {native['budget_mean_recall']['20']:.3f} | {native['budget_mean_recall']['40']:.3f} | {native['budget_mean_recall']['58']:.3f} |",
        f"| Selected multiscale on identical v5 proposals | {identical['budget_mean_recall']['20']:.3f} | {identical['budget_mean_recall']['40']:.3f} | {identical['budget_mean_recall']['58']:.3f} |",
        f"| 50/50 carrier-feature proposal quota | {quota['budget_mean_recall']['20']:.3f} | {quota['budget_mean_recall']['40']:.3f} | {quota['budget_mean_recall']['58']:.3f} |",
        "", "## Interpretation", "",
        "Native peaks, identical-proposal ranking, and proposal quota are distinct estimands. The optimistic per-source oracle is headroom only. Sparse unmatched candidates remain unknown, not false positives.",
        "", "## Artifacts", "",
        "See `metrics.json`, `evaluation/`, and `diagnostics/`.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config: MultiscaleInformationConfig) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(config.resources["cpu_threads"])
    audit = _matching_preflight(config)
    patch = PatchInformationConfig.load(config.source_patch_config)
    ranker = InnovationRankerConfig.load(patch.source_ranker_config)
    partial = Path(str(config.output_dir) + ".partial")
    partial.mkdir(parents=True)
    evaluation_dir = partial / "evaluation"
    diagnostics_dir = partial / "diagnostics"
    evaluation_dir.mkdir()
    diagnostics_dir.mkdir()
    _atomic_json(partial / "preflight.json", audit)
    _atomic_json(partial / "config.resolved.json", config.to_dict())
    progress = partial / "progress.jsonl"
    started = time.time()
    labels = label_core.load_labels(ranker.labels_tsv)
    feature_ids, families = _feature_inventory(config)
    raw, benchmark = _compute_raw_maps(config, patch, ranker, labels, progress)
    maps = _fuse_maps(config, patch, ranker, raw)
    del raw
    base_config = FeatureUtilityConfig.load(ranker.feature_root / "config.resolved.json")
    v5_maps, _ = _generate_v5_maps(ranker, labels, base_config, progress)
    carrier = v5_maps["carrier_signed"]
    native_baseline, native_rows = _lane_rows(
        config, maps, carrier, labels, ranker, families
    )
    native_crossfits = {
        family: _crossfit(native_rows, config, family=family)
        for family in sorted(set(families.values()))
    }
    native_all = _crossfit(native_rows, config)
    _atomic_json(evaluation_dir / "native_peak_screen.json", {
        "carrier_baseline": native_baseline, "rows": native_rows,
        "crossfitted_families": native_crossfits, "crossfitted_all": native_all,
    })
    all_maps = {**v5_maps, **maps}
    identical_baseline, identical_rows, inventory = _identical_proposal_rows(
        config, all_maps, feature_ids, families, labels, ranker
    )
    identical_crossfits = {
        family: _crossfit(identical_rows, config, family=family)
        for family in sorted(set(families.values()))
    }
    identical_all = _crossfit(identical_rows, config)
    _atomic_json(evaluation_dir / "identical_proposal_screen.json", {
        "carrier_baseline": identical_baseline, "rows": identical_rows,
        "crossfitted_families": identical_crossfits,
        "crossfitted_all": identical_all,
    })
    _atomic_json(evaluation_dir / "identical_proposal_inventory.json", inventory)
    selected_features = _crossfit(
        [row for row in native_rows if row["kind"] == "standalone"], config
    )
    quota = _quota_evaluation(
        config, selected_features, maps, carrier, labels, ranker
    )
    _atomic_json(evaluation_dir / "quota_union.json", quota)
    oracle_rows, oracle_fixed = _oracle_coverage(
        all_maps, tuple(dict.fromkeys((*V5_PROPOSAL_SOURCE_IDS, *feature_ids))),
        labels, ranker,
        [int(value) for value in config.evaluation["oracle_source_budgets"]],
    )
    _atomic_json(evaluation_dir / "oracle_coverage.json", {"rows": oracle_rows})
    posthoc = sorted(
        [row for row in native_rows if row["kind"] == "standalone"],
        key=lambda row: (
            np.mean([row["budget_mean_recall"]["20"], row["budget_mean_recall"]["40"]]),
            row["budget_mean_recall"]["58"], row["config_id"],
        ), reverse=True,
    )[: int(config.visualization["tiff_feature_count"])]
    diagnostic = _write_score_tiff(
        diagnostics_dir / "top_multiscale_maps.tif",
        [maps[row["feature_id"]]["events"][burst] for row in posthoc for burst in BURSTS],
        compression=str(config.visualization["compression"]),
        description={
            "feature_ids": [row["feature_id"] for row in posthoc],
            "page_order": "feature-major; bursts 1-4",
        },
    )
    oracle58 = [row for row in oracle_rows if int(row["source_budget"]) == 58]
    oracle_mean = float(np.mean([row["union_coverage"] for row in oracle58]))
    patch_metrics = json.loads(
        (config.source_patch_root / "metrics.json").read_text(encoding="utf-8")
    )
    conclusion = (
        f"Completed {config.feature_count} multiscale maps and "
        f"{2 * config.lane_count} native/identical-proposal lanes. "
        f"Leakage-safe native selection scored "
        f"{native_all['budget_mean_recall']['20']:.3f}/"
        f"{native_all['budget_mean_recall']['40']:.3f}/"
        f"{native_all['budget_mean_recall']['58']:.3f} at budgets 20/40/58; "
        f"identical-proposal selection scored "
        f"{identical_all['budget_mean_recall']['20']:.3f}/"
        f"{identical_all['budget_mean_recall']['40']:.3f}/"
        f"{identical_all['budget_mean_recall']['58']:.3f}."
    )
    metrics = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": "completed",
        "feature_count": config.feature_count,
        "native_lane_count": config.lane_count,
        "identical_proposal_lane_count": config.lane_count,
        "carrier_native": native_baseline,
        "carrier_same_v5_union": identical_baseline,
        "crossfitted_native_all": native_all,
        "crossfitted_native_families": native_crossfits,
        "crossfitted_identical_all": identical_all,
        "crossfitted_identical_families": identical_crossfits,
        "quota_union": quota,
        "augmented_oracle_58_mean_coverage": oracle_mean,
        "patch_v1_oracle_58_mean_coverage": patch_metrics[
            "augmented_oracle_58_mean_coverage"
        ],
        "benchmark": benchmark,
        "diagnostics": {"top_multiscale_maps": diagnostic},
        "precision_contract": "Sparse unmatched candidates are unknown; candidate burden is not precision.",
        "conclusion": conclusion,
        "elapsed_seconds": time.time() - started,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    _atomic_json(partial / "metrics.json", metrics)
    _report(partial / "REPORT.md", metrics)
    _atomic_json(partial / "run_state.json", {
        "status": "completed", "feature_count": config.feature_count,
        "evaluated_lane_count": 2 * config.lane_count,
        "elapsed_seconds": metrics["elapsed_seconds"],
        "max_rss_mib": metrics["max_rss_mib"],
    })
    partial.replace(config.output_dir)
    return metrics
