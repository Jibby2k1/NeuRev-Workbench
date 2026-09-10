"""Freeze a label-blind common proposal universe and real-data run contract."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from neurobench.experiments.frame_difference import _atomic_json, _available_ram_mib
from neurobench.experiments.learned_operator_selection.data import (
    canonical_event_intervals, load_and_validate_labels,
)
from neurobench.metrics.sparse_detection import (
    extract_separated_local_maxima, match_peaks_one_to_one, temporal_pool,
)

from .config import ICAWhiteningConfig
from .real_config import RealDataConfig


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_eligible(path: Path) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    total = 0
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if str(row["all_numerically_resolved"]).lower() != "true":
                continue
            total += 1
            counts[row["family"]] = counts.get(row["family"], 0) + 1
    return total, counts


def _normalized_review(source: np.ndarray, parent: ICAWhiteningConfig) -> np.ndarray:
    start, stop = parent.frames.review_start_ui - 1, parent.frames.review_end_ui
    values = np.asarray(source[start:stop], dtype=np.float32)
    quiet_count = parent.frames.quiet_end_ui - parent.frames.review_start_ui + 1
    baseline = np.median(values[:quiet_count], axis=0)
    low, high = np.percentile(values[:quiet_count, ::4, ::4], [1, 99.9])
    scale = max(float(high - low), 1.0)
    return ((values - baseline[None]) / scale).astype(np.float32)


def _lane_stacks(signed: np.ndarray) -> dict[str, np.ndarray]:
    difference = np.zeros_like(signed)
    difference[1:] = np.abs(np.diff(signed, axis=0))
    smooth = gaussian_filter(signed, sigma=(0, 2, 2), mode="reflect")
    return {
        "raw_activity": np.maximum(signed, 0),
        "signed_temporal_difference": difference,
        "spatial_highpass": np.abs(signed - smooth).astype(np.float32),
    }


def _proposal_universe(
    signed: np.ndarray, labels: list[dict[str, Any]], parent: ICAWhiteningConfig,
    config: RealDataConfig,
) -> tuple[list[dict[str, Any]], str]:
    intervals = canonical_event_intervals(labels, parent.frames.review_start_ui)
    lane_stacks = _lane_stacks(signed)
    lane_peaks: dict[tuple[int, str], list[tuple[float, int, int]]] = {}
    for burst, (start, stop) in sorted(intervals.items()):
        for lane in config.proposals.lanes:
            score = temporal_pool(lane_stacks[lane][start:stop], config.proposals.temporal_pool)
            lane_peaks[(burst, lane)] = extract_separated_local_maxima(
                score, config.proposals.proposal_nms_distance_px,
                limit=config.proposals.per_lane_per_burst,
            )

    rows: list[dict[str, Any]] = []
    separation2 = config.proposals.union_separation_px ** 2
    for burst in sorted(intervals):
        selected: list[tuple[int, int]] = []
        depth = 0
        while True:
            added = False
            for lane in config.proposals.lanes:
                values = lane_peaks[(burst, lane)]
                if depth >= len(values):
                    continue
                added = True
                score, x, y = values[depth]
                if any((x - old_x) ** 2 + (y - old_y) ** 2 <= separation2 for old_x, old_y in selected):
                    continue
                selected.append((x, y))
                start, stop = intervals[burst]
                peak_frame = int(start + np.argmax(lane_stacks[lane][start:stop, y, x]))
                rows.append({
                    "candidate_id": f"common_b{burst}_{len(selected):04d}",
                    "burst_id": burst, "x_px": x, "y_px": y,
                    "peak_frame_review_zero": peak_frame,
                    "origin_lane": lane, "origin_score": score,
                })
            if not added:
                break
            depth += 1
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return rows, _hash_bytes(payload)


def _coverage(rows: list[dict[str, Any]], labels: list[dict[str, Any]], radius: int) -> dict[str, Any]:
    folds = []
    for burst in sorted({int(row["burst_id"]) for row in labels}):
        candidates = [
            (float(-index), int(row["x_px"]), int(row["y_px"]))
            for index, row in enumerate(rows) if int(row["burst_id"]) == burst
        ]
        truth = [row for row in labels if int(row["burst_id"]) == burst]
        matches, _ = match_peaks_one_to_one(candidates, truth, radius)
        folds.append({
            "burst_id": burst, "candidate_count": len(candidates),
            "known_positive_count": len(truth), "covered_known_positives": len(matches),
            "coverage_ceiling": len(matches) / len(truth),
        })
    return {
        "folds": folds,
        "pooled_coverage_ceiling": sum(row["covered_known_positives"] for row in folds)
        / sum(row["known_positive_count"] for row in folds),
        "interpretation": "coverage of frozen label-blind proposal universe; unmatched candidates remain unknown",
    }


def preflight(config: RealDataConfig, *, artifact_dir: str | Path) -> dict[str, Any]:
    destination = Path(artifact_dir).expanduser().resolve()
    if destination.exists() or config.output_dir.exists():
        raise FileExistsError("real-data preflight or output root already exists")
    parent = ICAWhiteningConfig.from_json(config.parent_config)
    required = (parent.source_video, parent.labels_tsv, parent.label_summary,
                config.synthetic_registry, config.decision_amendment)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"real-data prerequisite missing: {missing}")
    amendment = json.loads(config.decision_amendment.read_text(encoding="utf-8"))
    if amendment.get("amended_downstream_decision") != "continue_real_data_with_synthetic_warning":
        raise RuntimeError("matching investigator amendment is required")
    source = np.load(parent.source_video, mmap_mode="r", allow_pickle=False)
    labels = load_and_validate_labels(parent.labels_tsv, parent.label_summary, tuple(source.shape[1:]))
    signed = _normalized_review(source, parent)
    proposals, proposal_digest = _proposal_universe(signed, labels, parent, config)
    eligible_count, family_counts = _load_eligible(config.synthetic_registry)
    review_mib = signed.nbytes / 2**20
    proposal_bytes = len(proposals) * 560 * 8 * 4
    estimated_output_mib = int(np.ceil((eligible_count * 4096 + proposal_bytes) / 2**20))
    disk_probe = destination.parent
    while not disk_probe.exists():
        disk_probe = disk_probe.parent
    free_mib = shutil.disk_usage(disk_probe).free // 2**20
    ready = bool(
        eligible_count > 0
        and review_mib * 4 < parent.resources.max_ram_mib
        and _available_ram_mib() >= parent.resources.max_ram_mib
        and free_mib >= parent.resources.min_free_disk_mib + estimated_output_mib
    )
    coverage = _coverage(proposals, labels, config.proposals.match_radius_px)
    intervals = canonical_event_intervals(labels, parent.frames.review_start_ui)
    payload = {
        "schema_version": 1, "experiment_id": config.experiment_id, "ready": ready,
        "synthetic_warning": {
            "non_blocking": True, "amendment_sha256": _hash_file(config.decision_amendment),
            "claim_restriction": "ICA components are not verified biological sources",
        },
        "source": {"sha256": _hash_file(parent.source_video), "shape": list(source.shape)},
        "labels": {
            "sha256": _hash_file(parent.labels_tsv), "rows": len(labels),
            "unique_identities": len({row["roi_identity"] for row in labels}),
            "use_before_proposal_freeze": "event-window boundaries only; no coordinates or identities",
            "use_after_proposal_freeze": "coverage ceiling and frozen sparse-positive metrics",
            "unmatched_candidates": "unknown_not_negative",
        },
        "proposal_universe": {
            "digest": proposal_digest, "candidate_count": len(proposals),
            "lanes": list(config.proposals.lanes), "coverage": coverage,
        },
        "eligible_fits": {
            "rule": config.fitting.eligibility, "count": eligible_count,
            "family_counts": family_counts,
            "synthetic_recovery_score_used_for_eligibility": False,
        },
        "folds": {
            "outer": "leave_one_burst_out", "burst_ids": [1, 2, 3, 4],
            "event_intervals_review_zero_half_open": {
                str(burst): [start, stop]
                for burst, (start, stop) in sorted(intervals.items())
            },
        },
        "controls": list(config.controls),
        "resources": {
            "review_array_mib": review_mib, "estimated_output_mib": estimated_output_mib,
            "disk_free_mib": free_mib, "ram_available_mib": _available_ram_mib(),
            "implementation_note": "common-universe scoring for every eligible fit; dense native maps only for nested finalists",
        },
        "claim_limits": [
            "single recording", "sparse positives do not identify precision",
            "no biological-source identity", "no cross-recording generalization",
        ],
    }
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "common_proposal_universe.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(proposals[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(proposals)
    _atomic_json(destination / "preflight.json", payload)
    _atomic_json(destination / "resolved_real_config.json", {
        "manifest": str(config.manifest_path), "parent_config": str(config.parent_config),
        "synthetic_registry": str(config.synthetic_registry), "output_dir": str(config.output_dir),
        "proposals": config.proposals.__dict__, "fitting": config.fitting.__dict__,
        "controls": list(config.controls), "synthetic_warning_non_blocking": True,
    })
    if not ready:
        raise RuntimeError(f"real-data preflight is not ready: {payload['resources']}")
    return payload
