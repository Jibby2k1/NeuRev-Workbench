"""Freeze eligibility and provenance for independent-recording confirmation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from neurobench.experiments.frame_difference import _atomic_json


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _video_contract(
    root: Path,
    video: dict[str, Any],
    crop_rows: dict[str, dict[str, Any]],
    annotations: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    dataset_rate_hz: float,
) -> dict[str, Any]:
    video_id = str(video["video_id"])
    rows = [row for row in annotations if row["video_id"] == video_id]
    crop = crop_rows.get(video_id)
    video_path = _resolved(root, str(video["path"]))
    source_files = sorted({str(row["source_file"]) for row in rows})
    sheets = sorted({str(row["source_sheet_title"]) for row in rows})
    relevant_warnings = [
        warning for warning in warnings
        if any(source in str(warning.get("file", "")) for source in source_files)
    ]
    identity_warning = any(row.get("kind") == "title_mismatch" for row in relevant_warnings)
    malformed_warning = any(
        row.get("kind") in {"reversed_frame_range", "unusually_long_frame_range"}
        for row in relevant_warnings
    )
    intervals = [interval for row in rows for interval in row["spike_intervals"]]
    in_bounds = all(
        0 <= int(interval["start_frame"]) <= int(interval["end_frame"]) < int(video["frame_count"])
        for interval in intervals
    )
    crop_matches = bool(
        crop
        and list(crop["output_shape"]) == [
            int(video["frame_count"]), int(video["height"]), int(video["width"])
        ]
        and _resolved(root, str(crop["output_path"])) == video_path
    )
    eligible = bool(
        rows and intervals and in_bounds and crop_matches and video_path.is_file()
        and not identity_warning and not malformed_warning
    )
    return {
        "video_id": video_id,
        "eligible": eligible,
        "decision": "eligible_for_frozen_finalist_confirmation" if eligible else "excluded",
        "exclusion_reasons": [
            reason for condition, reason in (
                (not rows, "no_manual_roi_labels"),
                (not intervals, "no_manual_spike_intervals"),
                (not in_bounds, "label_interval_out_of_bounds"),
                (not crop_matches, "crop_manifest_mismatch"),
                (not video_path.is_file(), "cropped_video_missing"),
                (identity_warning, "workbook_video_identity_mismatch"),
                (malformed_warning, "malformed_interval_warning"),
            ) if condition
        ],
        "shape_tyx": [int(video["frame_count"]), int(video["height"]), int(video["width"])],
        "frame_rate_hz": dataset_rate_hz,
        "frame_rate_source": "dataset_manifest_root_contract",
        "roi_count": len(rows),
        "spike_interval_count": len(intervals),
        "source_workbooks": source_files,
        "source_sheet_titles": sheets,
        "warnings": relevant_warnings,
        "cropped_video": {
            "path": str(video_path),
            "size_bytes": video_path.stat().st_size if video_path.is_file() else None,
            "sha256": _hash_file(video_path) if eligible else None,
        },
        "crop": crop,
    }


def run_independent_preflight(
    *, dataset_root: str | Path, dataset_dir: str | Path, destination: str | Path,
) -> dict[str, Any]:
    """Audit independent recordings without constructing or scoring candidates."""
    root = Path(dataset_root).expanduser().resolve()
    dataset = Path(dataset_dir).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise FileExistsError(target)
    manifest_path = dataset / "manifest" / "video_manifest.json"
    crop_path = dataset / "crop_manifest.json"
    labels_path = dataset / "annotations" / "manual_roi_spikes_v1" / "manual_roi_spike_annotations.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    crops = json.loads(crop_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    rate = float(manifest["frame_rate_hz"])
    if rate <= 0 or float(labels["frame_rate_hz"]) != rate:
        raise RuntimeError("frame-rate contracts disagree")
    crop_rows = {
        Path(str(row["output_path"])).stem: row for row in crops["videos"]
    }
    labeled_ids = set(labels["video_roi_counts"])
    videos = [row for row in manifest["videos"] if row["video_id"] in labeled_ids]
    contracts = [
        _video_contract(
            root, video, crop_rows, labels["annotations"], labels.get("warnings", []), rate,
        )
        for video in videos
    ]
    eligible = [row["video_id"] for row in contracts if row["eligible"]]
    payload = {
        "schema_version": 1,
        "status": "ready_for_frozen_finalist" if eligible else "blocked_no_eligible_recording",
        "eligible_recordings": eligible,
        "recordings": contracts,
        "source_contracts": {
            "video_manifest": {"path": str(manifest_path), "sha256": _hash_file(manifest_path)},
            "crop_manifest": {"path": str(crop_path), "sha256": _hash_file(crop_path)},
            "annotation_manifest": {"path": str(labels_path), "sha256": _hash_file(labels_path)},
        },
        "confirmation_guard": {
            "requires_frozen_within_recording_finalists": True,
            "candidate_generation": "label_blind_full_recording_before_label_join",
            "required_order": [
                "freeze_finalist_operator_hashes", "construct_candidate_universe",
                "freeze_candidate_score_hash", "join_sparse_positive_labels",
            ],
            "unmatched_candidates": "unknown_not_negative",
            "claim_limit": "one independent recording; no population-level generalization",
        },
    }
    target.mkdir(parents=True, exist_ok=False)
    _atomic_json(target / "independent_confirmation_preflight.json", payload)
    return payload
