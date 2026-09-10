from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import pytest

from neurobench.experiments.gamma_ls_difference import (
    full_recording_scientific_audit as audit,
)
from neurobench.experiments.gamma_ls_difference.full_recording import (
    FullRecordingProposalUnavailable,
)
from neurobench.experiments.gamma_ls_difference.scientific_audit import GREEN


def _candidate(
    proposal_id: str,
    *,
    source_frame_ui: int,
    rank: int,
    score: float,
    x_px: int,
    y_px: int,
) -> dict[str, Any]:
    return {
        "proposal_id": proposal_id,
        "variant_id": audit.VARIANT_ID,
        "representation": audit.REPRESENTATION,
        "context_id": audit.CONTEXT_ID,
        "target_nms_peaks_per_calibration_unit": "1.0",
        "calibration_burden_unit": audit.INITIAL_BURDEN_UNIT,
        "threshold_z": "7.080160140991211",
        "scale_floor_percentile": "10.0",
        "scale_floor": "3.8241920471191406",
        "source_frame_ui": str(source_frame_ui),
        "candidate_rank_within_frame": str(rank),
        "score": str(score),
        "x_px": str(x_px),
        "y_px": str(y_px),
        "biological_status": "unknown_unreviewed_proposal",
        "temporal_linking_applied": "false",
    }


def test_normalization_is_typed_exact_and_canonical_hash_is_key_order_stable() -> None:
    source = _candidate(
        "proposal-1",
        source_frame_ui=101,
        rank=2,
        score=9.25,
        x_px=17,
        y_px=23,
    )
    normalized = audit._normalize_candidate(source)

    assert normalized == {
        "proposal_id": "proposal-1",
        "variant_id": audit.VARIANT_ID,
        "representation": audit.REPRESENTATION,
        "context_id": audit.CONTEXT_ID,
        "target_nms_peaks_per_calibration_unit": 1.0,
        "calibration_burden_unit": audit.INITIAL_BURDEN_UNIT,
        "threshold_z": 7.080160140991211,
        "scale_floor_percentile": 10.0,
        "scale_floor": 3.8241920471191406,
        "source_frame_ui": 101,
        "candidate_rank_within_frame": 2,
        "score": 9.25,
        "x_px": 17,
        "y_px": 23,
        "biological_status": "unknown_unreviewed_proposal",
        "temporal_linking_applied": False,
    }
    encoded = json.dumps(
        [normalized],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    expected = hashlib.sha256(encoded).hexdigest()
    assert audit._subset_hash([normalized]) == expected
    assert audit._subset_hash([dict(reversed(tuple(normalized.items())))]) == expected

    second = audit._normalize_candidate(
        _candidate(
            "proposal-2",
            source_frame_ui=102,
            rank=1,
            score=8.0,
            x_px=19,
            y_px=24,
        )
    )
    assert audit._normalized_rows_equal(
        [normalized, second],
        [dict(reversed(tuple(normalized.items()))), second],
    )
    assert not audit._normalized_rows_equal(
        [normalized, second], [second, normalized]
    )


def test_normalization_rejects_ambiguous_booleans_and_parses_structured_fields() -> None:
    count = audit._normalize_count(
        {
            "variant_id": audit.VARIANT_ID,
            "representation": audit.REPRESENTATION,
            "context_id": audit.CONTEXT_ID,
            "target_nms_peaks_per_calibration_unit": "1",
            "calibration_burden_unit": audit.INITIAL_BURDEN_UNIT,
            "threshold_z": "7.5",
            "source_frame_ui": "101",
            "proposal_count": "3",
        }
    )
    assert count["target_nms_peaks_per_calibration_unit"] == 1.0
    assert count["source_frame_ui"] == 101
    assert count["proposal_count"] == 3

    threshold = audit._normalize_threshold(
        {
            "variant_id": audit.VARIANT_ID,
            "calibration_interval_ui": "[1, 100]",
            "calibration_frame_first_ui": "1",
            "calibration_frame_last_ui": "100",
            "calibration_frame_count": "100",
            "representation_alignment_note": "cold start retained",
            "calibration_role": "initialization_frames_not_assumed_event_free",
            "calibration_frame_locations_use_annotation_content": "false",
            "calibration_window_template_ui_frames": (
                '{"initial_1s_block_1": 50, "initial_1s_block_2": 50}'
            ),
            "calibration_window_template_source": "frozen protocol",
            "calibration_window_template_is_annotation_derived": "false",
            "burst_window_supervised": "false",
            "scale_floor_percentile": "10",
            "scale_floor": "3.5",
            "target_nms_peaks_per_calibration_unit": "1",
            "calibration_burden_unit": audit.INITIAL_BURDEN_UNIT,
            "threshold_z": "7.0",
            "achieved_nms_peaks_per_calibration_unit": "1.0",
            "total_calibration_nms_peaks": "2",
            "calibration_windows_source_ui": "[[1, 50], [51, 100]]",
            "calibration_window_overlap_frames": "0",
            "nms_distance_px": "6",
            "probability_model_claimed": "false",
        }
    )
    assert threshold["calibration_interval_ui"] == [1, 100]
    assert threshold["calibration_window_template_ui_frames"] == {
        "initial_1s_block_1": 50,
        "initial_1s_block_2": 50,
    }
    assert threshold["calibration_windows_source_ui"] == [[1, 50], [51, 100]]
    assert threshold["probability_model_claimed"] is False
    with pytest.raises(ValueError, match="not a boolean value"):
        audit._bool("0")


def test_complete_index_accepts_exact_nested_inventory_and_rejects_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sealed"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "summary.json").write_text('{"status":"complete"}\n', encoding="utf-8")
    (nested / "values.bin").write_bytes(b"frozen-values")
    (root / "ignored.partial").write_bytes(b"incomplete")
    (root / "artifact_index.json").write_text(
        json.dumps(audit._artifact_index(root), sort_keys=True), encoding="utf-8"
    )

    verified = audit._verify_complete_index(root)
    assert verified["verified_artifact_count"] == 2
    assert verified["unindexed_file_count"] == 0
    assert verified["artifact_index_sha256"] == audit._sha256(
        root / "artifact_index.json"
    )

    unindexed = root / "not-frozen.txt"
    unindexed.write_text("drift", encoding="utf-8")
    with pytest.raises(
        audit.FullRecordScientificAuditUnavailable,
        match="artifact index coverage changed",
    ):
        audit._verify_complete_index(root)
    unindexed.unlink()

    (nested / "values.bin").write_bytes(b"tampered")
    with pytest.raises(
        FullRecordingProposalUnavailable,
        match="indexed artifact changed after freeze",
    ):
        audit._verify_complete_index(root)


def test_candidate_surrogates_are_deterministic_spatial_and_label_free() -> None:
    rows = [
        _candidate(
            "singleton-low-score",
            source_frame_ui=101,
            rank=1,
            score=5.0,
            x_px=30,
            y_px=30,
        ),
        _candidate(
            "recurrent-a",
            source_frame_ui=101,
            rank=2,
            score=7.0,
            x_px=10,
            y_px=10,
        ),
        _candidate(
            "recurrent-b",
            source_frame_ui=102,
            rank=1,
            score=8.0,
            x_px=12,
            y_px=12,
        ),
        _candidate(
            "recurrent-c-radius-inclusive",
            source_frame_ui=103,
            rank=1,
            score=6.0,
            x_px=13,
            y_px=10,
        ),
        _candidate(
            "singleton-high-score-outside-radius",
            source_frame_ui=104,
            rank=1,
            score=9.0,
            x_px=13,
            y_px=11,
        ),
    ]

    selected, assignments, cluster_count = audit._build_candidate_surrogates(
        list(reversed(rows)), limit=2
    )
    repeated = audit._build_candidate_surrogates(rows, limit=2)
    assert (selected, assignments, cluster_count) == repeated
    assert cluster_count == 3
    assert [row["model_roi_id"] for row in selected] == [
        "model_roi_0002",
        "model_roi_0003",
    ]
    assert selected[0] == {
        "model_roi_id": "model_roi_0002",
        "x_px": 10,
        "y_px": 10,
        "proposal_occurrence_count": 3,
        "distinct_proposal_frame_count": 3,
        "first_source_frame_ui": 101,
        "last_source_frame_ui": 103,
        "best_frame_rank": 1,
        "maximum_score": 8.0,
        "selection_rule": (
            "greedy_3px_spatial_consolidation_then_recurrence_rank_score_id"
        ),
    }
    assignment_by_id = {row["proposal_id"]: row for row in assignments}
    assert assignment_by_id["recurrent-a"]["model_roi_id"] == "model_roi_0002"
    assert assignment_by_id["recurrent-b"]["model_roi_id"] == "model_roi_0002"
    assert assignment_by_id["recurrent-c-radius-inclusive"]["model_roi_id"] == (
        "model_roi_0002"
    )
    assert assignment_by_id["singleton-low-score"]["selected_for_closeup"] is False
    assert all(
        row["selected_for_closeup"] is True
        for proposal_id, row in assignment_by_id.items()
        if proposal_id != "singleton-low-score"
    )
    with pytest.raises(ValueError, match="surrogate limit must be positive"):
        audit._build_candidate_surrogates(rows, limit=0)


def test_causal_stage_processor_preserves_signed_difference_across_cpu_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    processor = audit._CausalStageProcessor(device="cpu", frame_shape=(2, 3))
    monkeypatch.setattr(processor, "_spatial", lambda values: values)

    first = np.stack(
        [
            np.full((2, 3), 10, dtype=np.uint16),
            np.full((2, 3), 20, dtype=np.uint16),
        ]
    )
    conditioned_1, difference_1, source_ui_1 = processor.process_stages(
        first, first_source_frame_ui=1
    )
    alpha = audit.gpu_representations.EMA_ALPHA
    expected_conditioned_1 = torch.stack(
        [
            torch.full((2, 3), 10.0),
            torch.full((2, 3), alpha * 20.0 + (1.0 - alpha) * 10.0),
        ]
    )
    assert torch.equal(conditioned_1, expected_conditioned_1)
    assert torch.count_nonzero(difference_1[0]).item() == 0
    assert torch.equal(difference_1[1], conditioned_1[1] - conditioned_1[0])
    np.testing.assert_array_equal(source_ui_1, np.asarray([1, 2], dtype=np.int64))

    second = np.full((1, 2, 3), 30, dtype=np.uint16)
    conditioned_2, difference_2, source_ui_2 = processor.process_stages(
        second, first_source_frame_ui=3
    )
    expected_current = alpha * torch.full((2, 3), 30.0) + (
        1.0 - alpha
    ) * conditioned_1[-1]
    assert torch.allclose(conditioned_2[0], expected_current, rtol=0, atol=1e-6)
    assert torch.allclose(
        difference_2[0], conditioned_2[0] - conditioned_1[-1], rtol=0, atol=1e-6
    )
    np.testing.assert_array_equal(source_ui_2, np.asarray([3], dtype=np.int64))
    with pytest.raises(ValueError, match="uint16 TYX"):
        processor.process_stages(
            second.astype(np.float32), first_source_frame_ui=4
        )


def test_marker_counting_separates_model_orange_from_expert_green() -> None:
    model_frame = np.zeros((12, 12, 3), dtype=np.uint8)
    model_frame[2:5, 3:7] = np.asarray(audit.ORANGE, dtype=np.uint8)
    model_counts = audit._marker_counts(model_frame)
    assert model_counts == {"green_pixels": 0, "orange_pixels": 12}

    expert_frame = np.zeros((12, 12, 3), dtype=np.uint8)
    expert_frame[6:9, 5:10] = np.asarray(GREEN, dtype=np.uint8)
    expert_counts = audit._marker_counts(expert_frame)
    assert expert_counts == {"green_pixels": 15, "orange_pixels": 0}


def test_gray_stage_helpers_keep_signed_zero_at_midgray() -> None:
    signed = audit._gray_signed(
        np.asarray([[-2.0, 0.0, 2.0]], dtype=np.float32),
        2.0,
        (3, 1),
    )
    unsigned = audit._gray_unsigned(
        np.asarray([[0.0, 5.0, 10.0]], dtype=np.float32),
        (0.0, 10.0),
        (3, 1),
    )
    np.testing.assert_array_equal(
        np.asarray(signed)[0, :, 0], np.asarray([0, 127, 255], dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        np.asarray(unsigned)[0, :, 0], np.asarray([0, 127, 255], dtype=np.uint8)
    )


def _checkpoint_threshold_row() -> dict[str, Any]:
    return audit._normalize_threshold(
        {
            "variant_id": audit.VARIANT_ID,
            "calibration_interval_ui": [1, 100],
            "calibration_frame_first_ui": 1,
            "calibration_frame_last_ui": 100,
            "calibration_frame_count": 100,
            "representation_alignment_note": "cold start retained",
            "calibration_role": "initialization_frames_not_assumed_event_free",
            "calibration_frame_locations_use_annotation_content": False,
            "calibration_window_template_ui_frames": {
                "initial_1s_block_1": 50,
                "initial_1s_block_2": 50,
            },
            "calibration_window_template_source": "declared_frame_interval_only",
            "calibration_window_template_is_annotation_derived": False,
            "burst_window_supervised": False,
            "scale_floor_percentile": 10.0,
            "scale_floor": audit.PINNED_SCALE_FLOOR,
            "target_nms_peaks_per_calibration_unit": 1.0,
            "calibration_burden_unit": audit.INITIAL_BURDEN_UNIT,
            "threshold_z": audit.PINNED_Q1_THRESHOLD,
            "achieved_nms_peaks_per_calibration_unit": 1.0,
            "total_calibration_nms_peaks": 2,
            "calibration_windows_source_ui": {
                "initial_1s_block_1": [1, 50],
                "initial_1s_block_2": [51, 100],
            },
            "calibration_window_overlap_frames": 0,
            "nms_distance_px": 6,
            "probability_model_claimed": False,
        }
    )


def test_normalize_threshold_accepts_sealed_replay_python_dict_repr() -> None:
    row = _checkpoint_threshold_row()
    expected_template = dict(row["calibration_window_template_ui_frames"])
    expected_windows = dict(row["calibration_windows_source_ui"])
    row["calibration_window_template_ui_frames"] = str(expected_template)
    row["calibration_windows_source_ui"] = str(expected_windows)

    normalized = audit._normalize_threshold(row)

    assert normalized["calibration_window_template_ui_frames"] == expected_template
    assert normalized["calibration_windows_source_ui"] == expected_windows


def _checkpoint_calibration_summary() -> dict[str, Any]:
    return {
        "variant_id": audit.VARIANT_ID,
        "calibration_frame_count": 100,
        "calibration_frame_first_ui": 1,
        "calibration_frame_last_ui": 100,
        "positive_local_std_sample_count": 19_000_000,
        "scale_floor_percentile": 10.0,
        "scale_floor": audit.PINNED_SCALE_FLOOR,
        "calibration_burden_unit": audit.INITIAL_BURDEN_UNIT,
        "calibration_window_template_ui_frames": {
            "initial_1s_block_1": 50,
            "initial_1s_block_2": 50,
        },
        "calibration_windows_source_ui": {
            "initial_1s_block_1": [1, 50],
            "initial_1s_block_2": [51, 100],
        },
        "calibration_window_overlap_frames": 0,
        "cpu_nms_exact_parity": (
            "maintained_strict_separated_nms_on_transferred_float32_scores"
        ),
    }


def test_authoritative_operational_ledger_is_hard_pinned() -> None:
    repository = Path(audit.__file__).resolve().parents[3]
    root = (
        repository
        / "Outputs/GammaLSDifference"
        / audit.PINNED_LEDGER_ROOT_NAME
    )
    ledger = audit._load_operational_ledger(root)
    assert ledger["hashes"] == audit.PINNED_LEDGER_HASHES
    assert ledger["q1_threshold"]["threshold_z"] == audit.PINNED_Q1_THRESHOLD
    assert ledger["q1_threshold"]["scale_floor"] == audit.PINNED_SCALE_FLOOR
    assert ledger["q1_threshold"]["calibration_burden_unit"] == (
        audit.INITIAL_BURDEN_UNIT
    )
    assert len(ledger["q1_candidates"]) == audit.EXPECTED_PROPOSALS


def test_self_consistent_reindexed_ledger_lookalike_is_rejected(
    tmp_path: Path,
) -> None:
    repository = Path(audit.__file__).resolve().parents[3]
    source = (
        repository
        / "Outputs/GammaLSDifference"
        / audit.PINNED_LEDGER_ROOT_NAME
    )
    lookalike = tmp_path / audit.PINNED_LEDGER_ROOT_NAME
    shutil.copytree(source, lookalike)
    with (lookalike / "REPORT.md").open("a", encoding="utf-8") as stream:
        stream.write("\nself-consistent lookalike mutation\n")
    (lookalike / "artifact_index.json").write_text(
        json.dumps(audit._artifact_index(lookalike), sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(
        audit.FullRecordScientificAuditUnavailable,
        match="sealed-r2 ledger hashes changed",
    ):
        audit._load_operational_ledger(lookalike)


def test_roots_must_be_pairwise_distinct_and_nonoverlapping(tmp_path: Path) -> None:
    audit._require_distinct_nonoverlapping_roots(
        [tmp_path / "one", tmp_path / "two", tmp_path / "three"]
    )
    with pytest.raises(ValueError, match="distinct and non-overlapping"):
        audit._require_distinct_nonoverlapping_roots(
            [tmp_path / "same", tmp_path / "same"]
        )
    with pytest.raises(ValueError, match="distinct and non-overlapping"):
        audit._require_distinct_nonoverlapping_roots(
            [tmp_path / "parent", tmp_path / "parent/child"]
        )


def test_checkpoint_resume_verifies_stage_hashes_and_ledger_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "TOTAL_FRAMES", 4)
    monkeypatch.setattr(audit, "CALIBRATION_UI", (1, 2))
    monkeypatch.setattr(audit, "APPLICATION_UI", (3, 4))
    work = tmp_path / "work"
    work.mkdir()
    arrays = {
        "conditioned": np.zeros((4, 2, 3), dtype=np.float32),
        "difference": np.zeros((4, 2, 3), dtype=np.float32),
        "gamma": np.zeros((4, 2, 3), dtype=np.float32),
        "threshold": np.zeros((2, 2, 3), dtype=np.bool_),
    }
    chunk_hash = audit._stage_chunk_sha256(
        arrays, start_zero=0, stop_zero=2
    )
    thresholds = [_checkpoint_threshold_row()]
    calibration_summary = _checkpoint_calibration_summary()
    cumulative_execution = audit._cumulative_checkpoint_execution(
        prior=None,
        current_segment_wall_seconds=2.5,
        current_segment_peak_allocated_bytes=100,
        current_segment_peak_reserved_bytes=200,
        last_completed_ui=2,
        completed_ranges=[(0, 2)],
        chunk_frames=2,
    )
    audit._write_replay_checkpoint(
        work,
        last_completed_ui=2,
        completed_ranges=[(0, 2)],
        chunk_hashes={"0:2": chunk_hash},
        candidates=[],
        counts=[],
        threshold_rows=thresholds,
        fitted_floor=audit.PINNED_SCALE_FLOOR,
        calibration_summary=calibration_summary,
        cumulative_execution=cumulative_execution,
        chunk_frames=2,
    )
    ledger = {
        "initial_candidates": [],
        "initial_counts": [],
        "initial_thresholds": thresholds,
    }
    restored = audit._load_replay_checkpoint(
        work, arrays=arrays, ledger=ledger, chunk_frames=2
    )
    assert restored["last_completed_source_frame_ui"] == 2
    assert restored["completed_ranges"] == [(0, 2)]
    assert restored["calibration_summary"] == calibration_summary
    assert restored["calibration_summary"][
        "positive_local_std_sample_count"
    ] == 19_000_000
    assert restored["cumulative_execution"] == cumulative_execution
    arrays["difference"][0, 0, 0] = 1.0
    with pytest.raises(
        audit.FullRecordScientificAuditUnavailable,
        match="checkpointed stage chunk changed",
    ):
        audit._load_replay_checkpoint(
            work, arrays=arrays, ledger=ledger, chunk_frames=2
        )


def test_checkpoint_hashes_calibration_and_accumulates_resume_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "TOTAL_FRAMES", 4)
    monkeypatch.setattr(audit, "CALIBRATION_UI", (1, 2))
    monkeypatch.setattr(audit, "APPLICATION_UI", (3, 4))
    work = tmp_path / "work"
    work.mkdir()
    arrays = {
        "conditioned": np.zeros((4, 2, 3), dtype=np.float32),
        "difference": np.zeros((4, 2, 3), dtype=np.float32),
        "gamma": np.zeros((4, 2, 3), dtype=np.float32),
        "threshold": np.zeros((2, 2, 3), dtype=np.bool_),
    }
    hashes = {
        "0:2": audit._stage_chunk_sha256(arrays, start_zero=0, stop_zero=2)
    }
    first_execution = audit._cumulative_checkpoint_execution(
        prior=None,
        current_segment_wall_seconds=2.5,
        current_segment_peak_allocated_bytes=100,
        current_segment_peak_reserved_bytes=200,
        last_completed_ui=2,
        completed_ranges=[(0, 2)],
        chunk_frames=2,
    )
    resumed_execution = audit._cumulative_checkpoint_execution(
        prior=first_execution,
        current_segment_wall_seconds=1.25,
        current_segment_peak_allocated_bytes=150,
        current_segment_peak_reserved_bytes=180,
        last_completed_ui=4,
        completed_ranges=[(0, 2), (2, 4)],
        chunk_frames=2,
    )
    assert resumed_execution == {
        "cumulative_successful_stage_wall_seconds": 3.75,
        "completed_source_frames": 4,
        "completed_compute_chunk_count": 2,
        "completed_checkpoint_range_count": 2,
        "contributing_execution_segment_count": 2,
        "peak_vram_allocated_bytes": 150,
        "peak_vram_reserved_bytes": 200,
        "wall_seconds_scope": (
            "sum_of_successful_stage_compute_flush_and_hash_intervals_ending_before_"
            "each_atomic_checkpoint_write; failed_incomplete_chunks_are_excluded"
        ),
        "throughput_claim_eligible": False,
    }

    thresholds = [_checkpoint_threshold_row()]
    summary = _checkpoint_calibration_summary()
    audit._write_replay_checkpoint(
        work,
        last_completed_ui=2,
        completed_ranges=[(0, 2)],
        chunk_hashes=hashes,
        candidates=[],
        counts=[],
        threshold_rows=thresholds,
        fitted_floor=audit.PINNED_SCALE_FLOOR,
        calibration_summary=summary,
        cumulative_execution=first_execution,
        chunk_frames=2,
    )
    ledger = {
        "initial_candidates": [],
        "initial_counts": [],
        "initial_thresholds": thresholds,
    }
    checkpoint_path = work / "replay_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["calibration_summary"]["positive_local_std_sample_count"] -= 1
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(
        audit.FullRecordScientificAuditUnavailable,
        match="calibration summary hash changed",
    ):
        audit._load_replay_checkpoint(
            work, arrays=arrays, ledger=ledger, chunk_frames=2
        )

    audit._write_replay_checkpoint(
        work,
        last_completed_ui=2,
        completed_ranges=[(0, 2)],
        chunk_hashes=hashes,
        candidates=[],
        counts=[],
        threshold_rows=thresholds,
        fitted_floor=audit.PINNED_SCALE_FLOOR,
        calibration_summary=summary,
        cumulative_execution=first_execution,
        chunk_frames=2,
    )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["cumulative_execution"][
        "cumulative_successful_stage_wall_seconds"
    ] += 1.0
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(
        audit.FullRecordScientificAuditUnavailable,
        match="cumulative execution hash changed",
    ):
        audit._load_replay_checkpoint(
            work, arrays=arrays, ledger=ledger, chunk_frames=2
        )


def test_checkpoint_counts_compute_chunks_across_64_100_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "TOTAL_FRAMES", 164)
    monkeypatch.setattr(audit, "CALIBRATION_UI", (1, 100))
    first = audit._cumulative_checkpoint_execution(
        prior=None,
        current_segment_wall_seconds=1.0,
        current_segment_peak_allocated_bytes=100,
        current_segment_peak_reserved_bytes=200,
        last_completed_ui=100,
        completed_ranges=[(0, 100)],
        chunk_frames=64,
    )
    assert first["completed_compute_chunk_count"] == 2
    assert first["completed_checkpoint_range_count"] == 1

    resumed = audit._cumulative_checkpoint_execution(
        prior=first,
        current_segment_wall_seconds=0.5,
        current_segment_peak_allocated_bytes=80,
        current_segment_peak_reserved_bytes=160,
        last_completed_ui=164,
        completed_ranges=[(0, 100), (100, 164)],
        chunk_frames=64,
    )
    assert resumed["completed_compute_chunk_count"] == 3
    assert resumed["completed_checkpoint_range_count"] == 2
    assert resumed["contributing_execution_segment_count"] == 2


def test_absence_facts_are_encoded_as_positive_pass_checks() -> None:
    checks = audit._absence_pass_checks(
        annotation_sources_opened=False,
        biological_claims_made=False,
    )
    assert checks == {
        "annotation_sources_not_opened": True,
        "biological_claims_not_made": True,
    }
    assert all(checks.values())
    assert not all(
        audit._absence_pass_checks(
            annotation_sources_opened=True,
            biological_claims_made=False,
        ).values()
    )


def test_full_successful_replay_completion_check_map_is_all_true() -> None:
    checks = audit._replay_completion_checks(
        calibration_summary=_checkpoint_calibration_summary(),
        calibration_frames_assumed_event_free=False,
        fitted_floor=audit.PINNED_SCALE_FLOOR,
        expected_floor=audit.PINNED_SCALE_FLOOR,
        q1_proposal_rows=audit.EXPECTED_PROPOSALS,
        q1_distinct_proposal_frames=audit.EXPECTED_PROPOSAL_FRAMES,
        q1_count_rows=audit.EXPECTED_ELIGIBLE_FRAMES,
        peak_vram_allocated=512,
        peak_vram_cap=1024,
        annotation_sources_opened=False,
        biological_claims_made=False,
    )
    assert set(checks) == audit.REPLAY_COMPLETION_CHECK_KEYS
    assert all(value is True for value in checks.values())


def test_complete_checkpoint_cpu_finalization_does_not_touch_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("CUDA entrypoint was called")

    monkeypatch.setattr(audit, "require_cuda_device", forbidden)
    monkeypatch.setattr(audit, "_require_free_vram_for_preflight", forbidden)
    monkeypatch.setattr(audit, "_CausalStageProcessor", forbidden)
    runtime = audit._resolve_replay_runtime(
        {
            "runtime": {
                "cuda_available": True,
                "resolved_device": "cuda:0",
                "device_name": "frozen preflight device",
            }
        },
        device="cuda:0",
        require_live_cuda=False,
    )
    assert runtime["live_cuda_reprobe_performed"] is False
    assert runtime["runtime_use"] == (
        "verified_complete_checkpoint_cpu_finalization_only"
    )
    processor, torch_module, wall_started = audit._prepare_replay_cuda_session(
        chunks=[],
        resolved_device="cuda:0",
        conditioned=np.empty((0,), dtype=np.float32),
        last_completed_ui=audit.TOTAL_FRAMES,
    )
    assert (processor, torch_module, wall_started) == (None, None, None)
    work = tmp_path / "work"
    work.mkdir()
    (work / "replay_checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "checkpointed_replay_incomplete",
                "last_completed_source_frame_ui": audit.TOTAL_FRAMES,
            }
        ),
        encoding="utf-8",
    )
    assert audit._checkpoint_declares_complete_dense_coverage(work)


def test_known_all_true_gate_recovery_is_hash_pinned_and_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = tmp_path / "old-preflight"
    preflight.mkdir()
    old_module_sha = "old-module"
    old_test_sha = "old-test"
    old_science = {"device": "cuda:0"}
    (preflight / "science_contract.json").write_text(
        json.dumps(old_science), encoding="utf-8"
    )
    recorded_runtime = {
        "compute_capability": [8, 9],
        "cuda_available": True,
        "device_name": "frozen test GPU",
        "free_vram_bytes_before": 4_000,
        "requested_device": "cuda:0",
        "resolved_device": "cuda:0",
        "torch_cuda_build": "test-build",
        "torch_version": "test-torch",
        "total_vram_bytes": 8_000,
        "visible_device_count": 1,
    }
    preflight_payload = {
        "status": "ready",
        "gpu_run_ready": True,
        "science_contract_sha256": audit._canonical_sha256(old_science),
        "runtime": recorded_runtime,
        "resources": {
            "configured_peak_vram_cap_bytes": 2_000,
            "free_vram_gate_passed": True,
        },
        "implementation": {
            "files": {
                "repo://neurobench/experiments/gamma_ls_difference/"
                "full_recording_scientific_audit.py": {
                    "sha256": old_module_sha
                },
                "repo://tests/test_gamma_ls_full_recording_scientific_audit.py": {
                    "sha256": old_test_sha
                },
            }
        }
    }
    (preflight / "preflight.json").write_text(
        json.dumps(preflight_payload), encoding="utf-8"
    )
    (preflight / "status.json").write_text("{}\n", encoding="utf-8")
    (preflight / "artifact_index.json").write_text(
        json.dumps(audit._artifact_index(preflight)), encoding="utf-8"
    )

    destination = tmp_path / "replay-output"
    work = tmp_path / ".replay-output.replay-work"
    work.mkdir()
    scientific_core = {
        "schema_version": 1,
        "science_contract": {"frozen": True},
        "source_ledger_hashes": {"ledger": "same"},
        "checkpoint_work_root": str(work),
    }
    existing_contract = {
        **scientific_core,
        "preflight": {"artifact_index_sha256": "old"},
        "preflight_root": str(preflight),
    }
    current_contract = {
        **scientific_core,
        "preflight": {"artifact_index_sha256": "new"},
        "preflight_root": str(tmp_path / "new-preflight"),
    }
    (work / "run_contract.json").write_text(
        json.dumps(existing_contract), encoding="utf-8"
    )
    expected_ranges = [(0, audit.CALIBRATION_UI[1])] + [
        (start, stop)
        for start, stop in audit._aligned_chunks(
            audit.TOTAL_FRAMES,
            audit.DEFAULT_CHUNK_FRAMES,
            boundary_stops_ui=(audit.CALIBRATION_UI[1],),
        )
        if start >= audit.CALIBRATION_UI[1]
    ]
    (work / "replay_checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "checkpointed_replay_incomplete",
                "last_completed_source_frame_ui": audit.TOTAL_FRAMES,
                "completed_ranges_zero_half_open": [
                    list(row) for row in expected_ranges
                ],
                "chunk_stage_sha256": {
                    f"{start}:{stop}": "0" * 64
                    for start, stop in expected_ranges
                },
                "annotation_sources_opened": False,
            }
        ),
        encoding="utf-8",
    )
    for relative in audit.STAGE_FILES.values():
        path = work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test-stage-placeholder")
    (work / "failure.json").write_text(
        json.dumps(
            {
                "error_type": "FullRecordScientificAuditUnavailable",
                "checkpoint_exists": True,
                "error": (
                    "stage replay checks failed: {'scientific': True, "
                    "'annotation_sources_opened': False, "
                    "'biological_claims_made': False}"
                ),
            }
        ),
        encoding="utf-8",
    )
    pins = {
        "replay_output_name": destination.name,
        "preflight_root_name": preflight.name,
        "preflight_artifact_index_sha256": audit._sha256(
            preflight / "artifact_index.json"
        ),
        "preflight_json_sha256": audit._sha256(preflight / "preflight.json"),
        "implementation_sha256": old_module_sha,
        "test_sha256": old_test_sha,
        "run_contract_sha256": audit._sha256(work / "run_contract.json"),
        "replay_checkpoint_sha256": audit._sha256(
            work / "replay_checkpoint.json"
        ),
        "failure_sha256": audit._sha256(work / "failure.json"),
    }
    monkeypatch.setattr(audit, "KNOWN_ALL_TRUE_GATE_RECOVERY", pins)
    expected_stage_contracts = {
        "conditioned_current_frame.npy": (
            (audit.TOTAL_FRAMES, audit.FRAME_HEIGHT, audit.FRAME_WIDTH),
            np.dtype("float32"),
        ),
        "difference_signed.npy": (
            (audit.TOTAL_FRAMES, audit.FRAME_HEIGHT, audit.FRAME_WIDTH),
            np.dtype("float32"),
        ),
        "gamma_initial100_scale.npy": (
            (audit.TOTAL_FRAMES, audit.FRAME_HEIGHT, audit.FRAME_WIDTH),
            np.dtype("float32"),
        ),
        "threshold_exceedance_q1_application.npy": (
            (
                audit.EXPECTED_ELIGIBLE_FRAMES,
                audit.FRAME_HEIGHT,
                audit.FRAME_WIDTH,
            ),
            np.dtype("bool"),
        ),
    }

    def fake_np_load(path: Path, **kwargs: Any) -> Any:
        del kwargs
        shape, dtype = expected_stage_contracts[Path(path).name]
        return SimpleNamespace(shape=shape, dtype=dtype)

    def fake_checkpoint_load(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {
            "cumulative_execution": {
                "peak_vram_allocated_bytes": 1_000,
                "peak_vram_reserved_bytes": 1_500,
            }
        }

    def forbidden_cuda(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("recovery preflight touched live CUDA")

    monkeypatch.setattr(audit.np, "load", fake_np_load)
    monkeypatch.setattr(audit, "_load_replay_checkpoint", fake_checkpoint_load)
    monkeypatch.setattr(audit, "require_cuda_device", forbidden_cuda)
    preflight_recovery = audit._verify_known_gate_recovery_source_for_preflight(
        work,
        destination,
        ledger={},
        device="cuda:0",
        chunk_frames=audit.DEFAULT_CHUNK_FRAMES,
        configured_vram_cap=2_000,
    )
    assert preflight_recovery["source_replay_checkpoint_sha256"] == pins[
        "replay_checkpoint_sha256"
    ]
    assert preflight_recovery["execution_mode"] == "recovery_only_no_compute"
    assert preflight_recovery["live_cuda_probe_performed"] is False
    assert preflight_recovery["compute_authorized"] is False
    assert preflight_recovery["source_preflight_gpu_identity_verified"] is True
    assert preflight_recovery[
        "complete_checkpoint_stage_hashes_and_ledger_coverage_verified"
    ] is True
    assert preflight_recovery["recorded_peak_vram_within_cap"] is True
    recovery = audit._verify_known_all_true_gate_recovery(
        work,
        existing_contract=existing_contract,
        current_contract=current_contract,
        destination=destination,
    )
    assert recovery["dense_stages_or_proposal_rows_recomputed"] is False
    audit._preserve_known_gate_recovery_inputs(work, recovery)
    assert audit._sha256(
        work / "recovery_history/all_true_gate_checkpoint.json"
    ) == pins["replay_checkpoint_sha256"]
    audit._apply_known_gate_recovery(
        work,
        recovery=recovery,
        current_contract=current_contract,
    )
    assert json.loads((work / "run_contract.json").read_text()) == current_contract
    assert not (work / "failure.json").exists()
    assert json.loads((work / "checkpoint_recovery.json").read_text())[
        "full_checkpoint_and_ledger_verification_passed"
    ] is True


def test_recovery_checkpoint_requires_complete_untampered_hash_coverage() -> None:
    ranges = [(0, audit.CALIBRATION_UI[1])] + [
        (start, stop)
        for start, stop in audit._aligned_chunks(
            audit.TOTAL_FRAMES,
            audit.DEFAULT_CHUNK_FRAMES,
            boundary_stops_ui=(audit.CALIBRATION_UI[1],),
        )
        if start >= audit.CALIBRATION_UI[1]
    ]
    checkpoint = {
        "schema_version": 2,
        "status": "checkpointed_replay_incomplete",
        "last_completed_source_frame_ui": audit.TOTAL_FRAMES,
        "completed_ranges_zero_half_open": [list(row) for row in ranges],
        "chunk_stage_sha256": {
            f"{start}:{stop}": "a" * 64 for start, stop in ranges
        },
        "annotation_sources_opened": False,
    }
    assert audit._verify_recovery_checkpoint_dense_coverage(
        checkpoint, chunk_frames=audit.DEFAULT_CHUNK_FRAMES
    ) == ranges

    incomplete = json.loads(json.dumps(checkpoint))
    incomplete["last_completed_source_frame_ui"] = audit.TOTAL_FRAMES - 1
    with pytest.raises(
        audit.FullRecordScientificAuditUnavailable,
        match="not the complete dense replay",
    ):
        audit._verify_recovery_checkpoint_dense_coverage(
            incomplete, chunk_frames=audit.DEFAULT_CHUNK_FRAMES
        )

    tampered = json.loads(json.dumps(checkpoint))
    first_hash_key = next(iter(tampered["chunk_stage_sha256"]))
    tampered["chunk_stage_sha256"][first_hash_key] = "tampered"
    with pytest.raises(
        audit.FullRecordScientificAuditUnavailable,
        match="not the complete dense replay",
    ):
        audit._verify_recovery_checkpoint_dense_coverage(
            tampered, chunk_frames=audit.DEFAULT_CHUNK_FRAMES
        )


def test_recovery_preflight_runtime_uses_frozen_identity_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_cuda(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("recovery preflight touched live CUDA")

    monkeypatch.setattr(audit, "require_cuda_device", forbidden_cuda)
    recorded_runtime = {
        "cuda_available": True,
        "requested_device": "cuda:0",
        "resolved_device": "cuda:0",
        "device_name": "hash-pinned GPU",
    }
    authorization = {
        "execution_mode": "recovery_only_no_compute",
        "live_cuda_probe_performed": False,
        "compute_authorized": False,
        "source_preflight_gpu_identity_verified": True,
        "complete_checkpoint_stage_hashes_and_ledger_coverage_verified": True,
        "recorded_peak_vram_within_cap": True,
        "recorded_runtime_identity": recorded_runtime,
    }
    assert audit._resolve_preflight_runtime_for_mode(
        device="cuda:0",
        configured_vram_cap=2_000,
        checkpoint_recovery_authorization=authorization,
    ) == recorded_runtime

    incomplete = dict(authorization)
    incomplete["complete_checkpoint_stage_hashes_and_ledger_coverage_verified"] = (
        False
    )
    with pytest.raises(
        audit.FullRecordScientificAuditUnavailable,
        match="authorization is incomplete",
    ):
        audit._resolve_preflight_runtime_for_mode(
            device="cuda:0",
            configured_vram_cap=2_000,
            checkpoint_recovery_authorization=incomplete,
        )


def test_failure_recorder_distinguishes_restart_and_resume(tmp_path: Path) -> None:
    destination = tmp_path / "result"
    work = tmp_path / ".result.replay-work"
    work.mkdir()

    @audit._record_replay_failure
    def fail(*, replay_output: Path) -> None:
        del replay_output
        raise RuntimeError("intentional")

    with pytest.raises(RuntimeError, match="intentional"):
        fail(replay_output=destination)
    failure = json.loads((work / "failure.json").read_text(encoding="utf-8"))
    assert failure["checkpoint_exists"] is False
    assert failure["recovery"].startswith("resume_restarts_short_precalibration")
    (work / "replay_checkpoint.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="intentional"):
        fail(replay_output=destination)
    failure = json.loads((work / "failure.json").read_text(encoding="utf-8"))
    assert failure["checkpoint_exists"] is True
    assert failure["recovery"] == "resume_from_verified_checkpoint"


def test_restored_causal_state_matches_uninterrupted_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    frames = np.stack(
        [np.full((2, 3), value, dtype=np.uint16) for value in (10, 20, 30)]
    )
    uninterrupted = audit._CausalStageProcessor(device="cpu", frame_shape=(2, 3))
    monkeypatch.setattr(uninterrupted, "_spatial", lambda values: values)
    conditioned_all, difference_all, _ = uninterrupted.process_stages(
        frames, first_source_frame_ui=1
    )

    before_failure = audit._CausalStageProcessor(device="cpu", frame_shape=(2, 3))
    monkeypatch.setattr(before_failure, "_spatial", lambda values: values)
    conditioned_prefix, _, _ = before_failure.process_stages(
        frames[:2], first_source_frame_ui=1
    )
    resumed = audit._CausalStageProcessor(device="cpu", frame_shape=(2, 3))
    monkeypatch.setattr(resumed, "_spatial", lambda values: values)
    restored_state = conditioned_prefix[-1].detach().clone()
    resumed.ema_state = restored_state
    resumed.previous_common = restored_state
    conditioned_suffix, difference_suffix, source_ui = resumed.process_stages(
        frames[2:], first_source_frame_ui=3
    )
    assert torch.equal(conditioned_suffix[0], conditioned_all[2])
    assert torch.equal(difference_suffix[0], difference_all[2])
    np.testing.assert_array_equal(source_ui, np.asarray([3], dtype=np.int64))


def test_peak_vram_cap_is_fail_closed() -> None:
    config = SimpleNamespace(
        payload={"resources": {"max_peak_vram_gib": 0.5}}
    )
    assert audit._require_peak_vram_within_cap(2**28, config) == 2**29
    with pytest.raises(
        audit.FullRecordScientificAuditUnavailable,
        match="exceeds cap",
    ):
        audit._require_peak_vram_within_cap(2**29 + 1, config)
    audit._require_free_vram_for_preflight(
        {"free_vram_bytes_before": 2**30}, 2**29
    )
    with pytest.raises(
        audit.FullRecordScientificAuditUnavailable,
        match="free CUDA memory",
    ):
        audit._require_free_vram_for_preflight(
            {"free_vram_bytes_before": 2**28}, 2**29
        )


def test_not_applicable_sections_reject_scientific_media(tmp_path: Path) -> None:
    for section in (
        "1_Expert_Annotations",
        "2_Model_Annotations",
        "3_Comparison",
    ):
        directory = tmp_path / section
        directory.mkdir()
        (directory / "README.md").write_text("N/A\n", encoding="utf-8")
    assert audit._section_scientific_media(tmp_path, "1_Expert_Annotations") == []
    (tmp_path / "1_Expert_Annotations/forbidden.png").write_bytes(b"not media")
    assert audit._section_scientific_media(tmp_path, "1_Expert_Annotations") == [
        "1_Expert_Annotations/forbidden.png"
    ]
    with pytest.raises(
        audit.FullRecordScientificAuditUnavailable,
        match="section is missing",
    ):
        audit._section_scientific_media(tmp_path, "missing")


def test_full_record_panel_preserves_aspect_and_orange_only_marker() -> None:
    stages = [np.zeros((340, 573), dtype=np.float32) for _ in range(6)]
    panel = audit._full_record_panel(
        stages,
        raw_limits=(0.0, 1.0),
        conditioned_limits=(0.0, 1.0),
        difference_limit=1.0,
        gamma_limit=1.0,
        source_frame_ui=101,
        proposals=[{"x_px": 200, "y_px": 100}],
    )
    assert panel.size == (1416, 186)
    assert abs((236 / 140) - (573 / 340)) < 0.002
    counts = audit._marker_counts(np.asarray(panel))
    assert counts["orange_pixels"] > 0
    assert counts["green_pixels"] == 0


def test_representative_packet_is_same_crop_and_exact_framewise_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "FRAME_HEIGHT", 7)
    monkeypatch.setattr(audit, "FRAME_WIDTH", 9)
    monkeypatch.setattr(audit, "APPLICATION_UI", (2, 4))
    movie = np.arange(4 * 7 * 9, dtype=np.uint16).reshape(4, 7, 9)
    conditioned = movie.astype(np.float32)
    difference = np.zeros_like(conditioned)
    difference[1:] = conditioned[1:] - conditioned[:-1]
    gamma = np.zeros_like(conditioned)
    gamma[1, 3, 4] = 2.0
    threshold = gamma[1:].copy() > 1.0
    representative = {
        "proposal_id": "q1-ui2-r1",
        "source_frame_ui": 2,
        "x_px": 4,
        "y_px": 3,
    }
    frame_proposals = [{**representative, "score": 2.0}]
    metadata = audit._write_representative_stage_packet(
        tmp_path / "packet.npz",
        tmp_path / "packet.json",
        tmp_path / "packet.png",
        movie=movie,
        conditioned=conditioned,
        difference=difference,
        gamma=gamma,
        threshold=threshold,
        representative=representative,
        frame_proposals=frame_proposals,
        source_movie_sha256="synthetic",
        threshold_z=1.0,
        crop_radius=2,
    )
    with np.load(tmp_path / "packet.npz", allow_pickle=False) as packet:
        assert set(packet.files) == {
            "raw",
            "conditioned_previous",
            "conditioned_current",
            "difference",
            "gamma_ls",
            "threshold_exceedance",
            "proposals",
        }
        assert {packet[key].shape for key in packet.files} == {(5, 5)}
        assert packet["proposals"].sum() == 1
    assert metadata["full_frame_nms_proposal_count"] == 1
    assert metadata["crop_nms_proposal_count"] == 1
    assert metadata["difference_alignment_max_abs_error"] == 0.0
    assert "cropped only after NMS" in metadata["proposals_semantics"]
    with Image.open(tmp_path / "packet.png") as preview:
        assert preview.size == (768, 172)


def test_cli_binds_recovery_and_resume_to_explicit_subcommands() -> None:
    preflight = audit._parser().parse_args(
        [
            "preflight",
            "--config",
            "config.json",
            "--ledger-root",
            "ledger",
            "--preflight-root",
            "preflight",
            "--replay-output",
            "replay",
            "--audit-output",
            "audit",
            "--recover-known-all-true-gate",
        ]
    )
    replay = audit._parser().parse_args(
        [
            "replay",
            "--config",
            "config.json",
            "--preflight-root",
            "preflight",
            "--ledger-root",
            "ledger",
            "--replay-output",
            "replay",
            "--audit-output",
            "audit",
            "--resume",
        ]
    )
    render = audit._parser().parse_args(
        [
            "render",
            "--replay-root",
            "replay",
            "--audit-output",
            "audit",
            "--resume",
        ]
    )
    assert preflight.recover_known_all_true_gate is True
    assert replay.resume is True
    assert render.resume is True


def test_video_resume_requires_checkpointed_hash_and_exact_contract(
    tmp_path: Path,
) -> None:
    work = tmp_path / "audit-work"
    path = work / "2_Model_Annotations/videos/closeups/model_roi_0001.mp4"
    writer = audit._VideoWriter(path, 16, 16, 50.0)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    frame[4:12, 4:12] = np.asarray(audit.ORANGE, dtype=np.uint8)
    for _ in range(4):
        writer.write(frame)
    writer.close()
    contract_sha = "frozen-renderer"
    audit._checkpoint_completed_video(
        work,
        audit_contract_sha256=contract_sha,
        path=path,
        width=16,
        height=16,
        expected_frames=4,
    )
    assert audit._checkpointed_video_is_complete(
        work,
        audit_contract_sha256=contract_sha,
        path=path,
        width=16,
        height=16,
        expected_frames=4,
    )
    values = bytearray(path.read_bytes())
    values[-1] ^= 1
    path.write_bytes(values)
    with pytest.raises(
        audit.FullRecordScientificAuditUnavailable,
        match="checkpointed video changed",
    ):
        audit._checkpointed_video_is_complete(
            work,
            audit_contract_sha256=contract_sha,
            path=path,
            width=16,
            height=16,
            expected_frames=4,
        )


def test_validation_directory_swap_is_recoverable_and_indexed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "audit"
    root.mkdir()
    for name in ("summary.json", "llm_context.json", "validation.json", "status.json"):
        (root / name).write_text('{"status":"pending"}\n', encoding="utf-8")
    (root / "artifact_index.json").write_text(
        json.dumps(audit._artifact_index(root)), encoding="utf-8"
    )
    result = audit._transactionally_finalize_audit(
        root,
        summary={"status": "complete"},
        llm_context={"status": "complete"},
        validation={"status": "passed"},
        status={"status": "complete"},
    )
    assert result["unindexed_file_count"] == 0
    assert json.loads((root / "summary.json").read_text())["status"] == "complete"

    staging, backup = audit._validation_transaction_paths(root)
    shutil.copytree(root, staging, copy_function=lambda source, target: Path(target).write_bytes(Path(source).read_bytes()))
    root.replace(backup)
    audit._recover_validation_transaction(root)
    assert root.is_dir()
    assert not backup.exists()
    assert audit._verify_complete_index(root)["unindexed_file_count"] == 0
