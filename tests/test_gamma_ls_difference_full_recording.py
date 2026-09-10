from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from neurobench.algorithms.gamma_local_standardization import GammaReferenceSpec
from neurobench.experiments.gamma_ls_difference.config import GammaLSDifferenceConfig
from neurobench.experiments.gamma_ls_difference.evaluation import (
    QUIET_NMS_PEAK_BURDENS,
)
from neurobench.experiments.gamma_ls_difference.full_recording import (
    CALIBRATION_VARIANTS,
    CANDIDATE_FIELDS,
    DECLARED_BURDEN_UNIT,
    FRAME_COUNT_FIELDS,
    INITIAL_BURDEN_UNIT,
    THRESHOLD_FIELDS,
    FullRecordingExecution,
    FullRecordingProposalUnavailable,
    _aligned_chunks,
    _exact_positive_quantile_on_host,
    _validate_execution,
    _verify_preflight_without_annotation_reads,
    _verify_support_context,
    candidate_content_sha256,
    expected_calibration_frame_count,
    extract_frame_proposals,
    fit_empirical_calibration_thresholds,
    run_full_recording_proposals,
    verify_indexed_artifact,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_index(root: Path) -> None:
    rows = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name != "artifact_index.json":
            rows.append(
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _hash(path),
                }
            )
    (root / "artifact_index.json").write_text(
        json.dumps({"schema_version": 1, "artifacts": rows}), encoding="utf-8"
    )


def _context_payload() -> dict[str, Any]:
    return {
        "context_id": "support_support_a_h19_g5_n9_m1",
        "half_width_px": 19,
        "guard_radius_px": 5,
        "shape": 9.0,
        "mode_fraction_of_half_width": 1.0,
        "mode_radius_px": 19.0,
        "support": "radial_disk",
        "padding": "valid_renormalized_zero",
        "eligible_primary": True,
    }


def _write_support(root: Path) -> None:
    root.mkdir()
    (root / "summary.json").write_text(
        json.dumps({"status": "complete_support_screen_only"}), encoding="utf-8"
    )
    (root / "validation.json").write_text(
        json.dumps(
            {
                "status": (
                    "passed_support_screen_artifact_contract_scientific_audit_pending"
                )
            }
        ),
        encoding="utf-8",
    )
    (root / "fold_contexts.json").write_text(
        json.dumps(
            {
                "selection_scope": "outer_training_fold_only",
                "selection_uses_positive_coordinates": False,
                "selection_uses_positive_identities": False,
                "burst_windows_used": True,
                "folds": [
                    {
                        "training_fold": fold,
                        "support_candidate_context": _context_payload(),
                    }
                    for fold in range(1, 5)
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_index(root)


def test_calibration_and_application_boundaries_are_exact() -> None:
    initial, declared = CALIBRATION_VARIANTS
    assert initial.calibration_interval_ui == (1, 100)
    assert initial.application_interval_ui == (101, 2359)
    assert initial.calibration_frame_locations_use_annotation_content is False
    assert initial.calibration_role == "initialization_frames_not_assumed_event_free"
    assert initial.burst_window_supervised is False
    assert declared.calibration_interval_ui == (1800, 1899)
    assert declared.application_interval_ui == (1900, 2359)
    assert declared.calibration_frame_locations_use_annotation_content is True
    assert expected_calibration_frame_count("raw", initial) == 100
    assert expected_calibration_frame_count("difference_signed", initial) == 100
    assert expected_calibration_frame_count(
        "difference_energy_normalized", initial
    ) == 100
    assert expected_calibration_frame_count("difference_signed", declared) == 100


def test_aligned_chunks_never_cross_freeze_boundaries() -> None:
    chunks = _aligned_chunks(
        2359, 64, boundary_stops_ui=(100, 1799, 1899)
    )
    assert chunks[0][0] == 0
    assert chunks[-1][1] == 2359
    assert {stop for _, stop in chunks}.issuperset({100, 1799, 1899})
    assert all(left[1] == right[0] for left, right in zip(chunks, chunks[1:]))


def test_empirical_thresholds_keep_block_burden_and_reject_per_frame_unit() -> None:
    rng = np.random.default_rng(11)
    scores = rng.normal(size=(100, 25, 25)).astype(np.float32)
    calibration = fit_empirical_calibration_thresholds(
        scores,
        window_durations={"initial_1s_block_1": 50, "initial_1s_block_2": 50},
        burden_unit=INITIAL_BURDEN_UNIT,
    )
    rows = calibration.operating_points
    assert tuple(
        row["target_nms_peaks_per_pseudo_burst"] for row in rows
    ) == QUIET_NMS_PEAK_BURDENS
    assert calibration.pseudo_burst_windows == {
        "initial_1s_block_1": (0, 50),
        "initial_1s_block_2": (50, 100),
    }
    assert calibration.diagnostics["pseudo_burst_windows_overlap_frames"] == 0
    assert all(
        row["achieved_nms_peaks_per_pseudo_burst"]
        <= row["target_nms_peaks_per_pseudo_burst"]
        for row in rows
    )
    with pytest.raises(ValueError, match="per-frame burden reinterpretation"):
        fit_empirical_calibration_thresholds(
            scores,
            window_durations={"initial_1s_block_1": 50, "initial_1s_block_2": 50},
            burden_unit="nms_peaks_per_calibration_frame",
        )


def test_scale_floor_uses_full_host_linear_quantile_not_torch_quantile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    values = torch.tensor(
        [[[-4.0, 0.0, 1.0], [2.0, 3.0, 20.0]]], dtype=torch.float32
    )

    def rejected_cuda_quantile(*_: Any, **__: Any) -> Any:
        raise RuntimeError("quantile() input tensor is too large")

    monkeypatch.setattr(torch, "quantile", rejected_cuda_quantile)
    floor, count = _exact_positive_quantile_on_host(values, 10.0)
    expected = np.quantile(
        np.asarray([1.0, 2.0, 3.0, 20.0], dtype=np.float32),
        0.1,
        method="linear",
    )
    assert floor == float(expected)
    assert count == 4


def test_declared_comparison_discloses_duration_window_overlap() -> None:
    rng = np.random.default_rng(12)
    scores = rng.normal(size=(100, 25, 25)).astype(np.float32)
    calibration = fit_empirical_calibration_thresholds(
        scores,
        window_durations={"1": 24, "2": 24, "3": 28, "4": 47},
        burden_unit=DECLARED_BURDEN_UNIT,
    )
    assert calibration.diagnostics["pseudo_burst_windows_overlap_frames"] > 0


def test_frame_proposals_use_strict_threshold_and_reconcile_counts() -> None:
    score = np.zeros((25, 25), dtype=np.float32)
    score[7, 7] = 5.0
    score[17, 17] = 4.0
    thresholds = [4.5, 3.5, 3.0, 2.5, 1.0]
    operating = [
        {
            "target_nms_peaks_per_calibration_unit": target,
            "calibration_burden_unit": INITIAL_BURDEN_UNIT,
            "threshold_z": threshold,
        }
        for target, threshold in zip(QUIET_NMS_PEAK_BURDENS, thresholds)
    ]
    proposals, counts = extract_frame_proposals(
        score,
        source_frame_ui=101,
        variant_id=CALIBRATION_VARIANTS[0].variant_id,
        representation="difference_signed",
        context_id="support_support_a_h19_g5_n9_m1",
        operating_points=operating,
        scale_floor_percentile=10.0,
        scale_floor=2.0,
    )
    assert [row["proposal_count"] for row in counts] == [1, 2, 2, 2, 2]
    assert sum(row["proposal_count"] for row in counts) == len(proposals)
    assert all(set(row) == set(CANDIDATE_FIELDS) for row in proposals)
    assert all(set(row) == set(FRAME_COUNT_FIELDS) for row in counts)
    assert all(row["biological_status"] == "unknown_unreviewed_proposal" for row in proposals)
    assert all(row["temporal_linking_applied"] is False for row in proposals)


def test_candidate_hash_is_order_invariant_after_canonical_sort() -> None:
    score = np.zeros((25, 25), dtype=np.float32)
    score[7, 7] = 5.0
    operating = [
        {
            "target_nms_peaks_per_calibration_unit": target,
            "calibration_burden_unit": INITIAL_BURDEN_UNIT,
            "threshold_z": 1.0,
        }
        for target in QUIET_NMS_PEAK_BURDENS
    ]
    rows, _ = extract_frame_proposals(
        score,
        source_frame_ui=101,
        variant_id=CALIBRATION_VARIANTS[0].variant_id,
        representation="raw",
        context_id="support_support_a_h19_g5_n9_m1",
        operating_points=operating,
        scale_floor_percentile=10.0,
        scale_floor=2.0,
    )
    assert candidate_content_sha256(rows) == candidate_content_sha256(list(reversed(rows)))


def test_index_verification_fails_after_artifact_mutation(tmp_path: Path) -> None:
    root = tmp_path / "indexed"
    root.mkdir()
    (root / "payload.json").write_text('{"value":1}', encoding="utf-8")
    _write_index(root)
    assert verify_indexed_artifact(root)["verified_artifact_count"] == 1
    (root / "payload.json").write_text('{"value":2}', encoding="utf-8")
    with pytest.raises(FullRecordingProposalUnavailable, match="changed after freeze"):
        verify_indexed_artifact(root)


def test_support_context_is_hash_verified_and_explicit(tmp_path: Path) -> None:
    root = tmp_path / "support"
    _write_support(root)
    reference, provenance = _verify_support_context(
        root, "support_support_a_h19_g5_n9_m1"
    )
    assert reference.support_width_px == 39
    assert reference.guard_radius_px == 5
    assert len(provenance["context_occurrences"]) == 4
    with pytest.raises(FullRecordingProposalUnavailable, match="absent"):
        _verify_support_context(root, "gamma_h11_g5_n9_m1")
    (root / "fold_contexts.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FullRecordingProposalUnavailable, match="changed after freeze"):
        _verify_support_context(root, "support_support_a_h19_g5_n9_m1")


class _MovieOnlySources(Mapping[str, Path]):
    def __init__(self, movie: Path) -> None:
        self.movie = movie
        self.requested: list[str] = []

    def __getitem__(self, key: str) -> Path:
        self.requested.append(key)
        if key != "movie":
            raise AssertionError(f"forbidden source access: {key}")
        return self.movie

    def __iter__(self):
        raise AssertionError("source mapping must not be iterated")

    def __len__(self) -> int:
        raise AssertionError("source mapping length must not be requested")


def _minimal_config(tmp_path: Path, sources: Mapping[str, Path]) -> GammaLSDifferenceConfig:
    payload = {
        "experiment_id": "spon_ca_burst_gamma_ls_difference_ablation_v1",
        "sources": {
            "movie": "data://movie.npy",
            "protected_labels_v1": "data://never-open-v1.tsv",
            "latest_labels_v7": "data://never-open-v7.tsv",
        },
        "efficiency": {"frame_chunks": [1, 8, 32, 64]},
        "frames": {
            "frame_interval_ms": 20.0,
            "burst_intervals_ui": {
                "1": [2003, 2026],
                "2": [2040, 2063],
                "3": [2122, 2149],
                "4": [2254, 2300],
            },
        },
        "gamma_ls_grid": {"finalist_scale_floor_percentiles": [10.0]},
        "cfar": {
            "quiet_nms_peaks_per_pseudo_burst": list(QUIET_NMS_PEAK_BURDENS),
            "nms_distance_px": 6,
        },
        "resources": {"minimum_free_disk_gib": 0.0},
    }
    return GammaLSDifferenceConfig(
        manifest_path=tmp_path / "manifest.json",
        repository=tmp_path,
        authority=tmp_path,
        payload=payload,
        source_paths=sources,  # type: ignore[arg-type]
        output_root=tmp_path,
    )


def test_preflight_verifier_hashes_movie_but_does_not_access_annotation_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neurobench.experiments.gamma_ls_difference import preflight as preflight_module

    movie = tmp_path / "movie.npy"
    movie.write_bytes(b"movie-bytes")
    sources = _MovieOnlySources(movie)
    config = _minimal_config(tmp_path, sources)
    root = tmp_path / "preflight"
    root.mkdir()
    executor_key = "repo://neurobench/experiments/gamma_ls_difference/full_recording.py"
    frozen_files = {
        executor_key: {"present": True, "sha256": "executor", "size_bytes": 1}
    }
    (root / "config.portable.json").write_text(
        json.dumps(config.portable_dict()), encoding="utf-8"
    )
    (root / "preflight.json").write_text(
        json.dumps(
            {
                "data_ready": True,
                "gpu_run_ready": True,
                "implementation": {"files": frozen_files},
                "source": {"movie": {"sha256": _hash(movie)}},
            }
        ),
        encoding="utf-8",
    )
    _write_index(root)
    monkeypatch.setattr(
        preflight_module,
        "_implementation_status",
        lambda _: {"complete": True, "files": frozen_files},
    )
    payload = _verify_preflight_without_annotation_reads(config, root)
    assert payload["annotation_sources_reopened"] is False
    assert sources.requested == ["movie"]


def _threshold_rows() -> tuple[dict[str, Any], ...]:
    rows = []
    for variant in CALIBRATION_VARIANTS:
        count = expected_calibration_frame_count("raw", variant)
        for target in QUIET_NMS_PEAK_BURDENS:
            rows.append(
                {
                    "variant_id": variant.variant_id,
                    "calibration_interval_ui": json.dumps(
                        list(variant.calibration_interval_ui), separators=(",", ":")
                    ),
                    "calibration_frame_first_ui": variant.calibration_interval_ui[0],
                    "calibration_frame_last_ui": variant.calibration_interval_ui[1],
                    "calibration_frame_count": count,
                    "representation_alignment_note": (
                        "all frames in the declared calibration interval"
                    ),
                    "calibration_role": variant.calibration_role,
                    "calibration_frame_locations_use_annotation_content": (
                        variant.calibration_frame_locations_use_annotation_content
                    ),
                    "calibration_window_template_ui_frames": (
                        '{"initial_1s_block_1":50,"initial_1s_block_2":50}'
                        if variant.variant_id == CALIBRATION_VARIANTS[0].variant_id
                        else '{"1":24,"2":24,"3":28,"4":47}'
                    ),
                    "calibration_window_template_source": (
                        "declared_frame_interval_only"
                        if variant.variant_id == CALIBRATION_VARIANTS[0].variant_id
                        else "configured_burst_interval_durations"
                    ),
                    "calibration_window_template_is_annotation_derived": (
                        variant.variant_id == CALIBRATION_VARIANTS[1].variant_id
                    ),
                    "burst_window_supervised": variant.burst_window_supervised,
                    "scale_floor_percentile": 10.0,
                    "scale_floor": 2.0,
                    "target_nms_peaks_per_calibration_unit": target,
                    "calibration_burden_unit": (
                        INITIAL_BURDEN_UNIT
                        if variant.variant_id == CALIBRATION_VARIANTS[0].variant_id
                        else DECLARED_BURDEN_UNIT
                    ),
                    "threshold_z": 3.0,
                    "achieved_nms_peaks_per_calibration_unit": 0.0,
                    "total_calibration_nms_peaks": 0,
                    "calibration_windows_source_ui": "{}",
                    "calibration_window_overlap_frames": (
                        0
                        if variant.variant_id == CALIBRATION_VARIANTS[0].variant_id
                        else 23
                    ),
                    "nms_distance_px": 6,
                    "probability_model_claimed": False,
                }
            )
    assert all(set(row) == set(THRESHOLD_FIELDS) for row in rows)
    return tuple(rows)


def _mock_execution(**_: Any) -> FullRecordingExecution:
    frame_rows: list[dict[str, Any]] = []
    for variant in CALIBRATION_VARIANTS:
        start, stop = variant.application_interval_ui
        for target in QUIET_NMS_PEAK_BURDENS:
            for frame_ui in range(start, stop + 1):
                frame_rows.append(
                    {
                        "variant_id": variant.variant_id,
                        "representation": "raw",
                        "context_id": "support_support_a_h19_g5_n9_m1",
                        "target_nms_peaks_per_calibration_unit": target,
                        "calibration_burden_unit": (
                            INITIAL_BURDEN_UNIT
                            if variant.variant_id == CALIBRATION_VARIANTS[0].variant_id
                            else DECLARED_BURDEN_UNIT
                        ),
                        "threshold_z": 3.0,
                        "source_frame_ui": frame_ui,
                        "proposal_count": 0,
                    }
                )
    first = {
        "proposal_id": "initial_100_annotation_file_and_location_free__b0p25__ui0101__r00001",
        "variant_id": CALIBRATION_VARIANTS[0].variant_id,
        "representation": "raw",
        "context_id": "support_support_a_h19_g5_n9_m1",
        "target_nms_peaks_per_calibration_unit": 0.25,
        "calibration_burden_unit": INITIAL_BURDEN_UNIT,
        "threshold_z": 3.0,
        "scale_floor_percentile": 10.0,
        "scale_floor": 2.0,
        "source_frame_ui": 101,
        "candidate_rank_within_frame": 1,
        "score": 4.0,
        "x_px": 20,
        "y_px": 30,
        "biological_status": "unknown_unreviewed_proposal",
        "temporal_linking_applied": False,
    }
    second = {
        **first,
        "proposal_id": "declared_quiet_burst_window_supervised__b5p0__ui1900__r00001",
        "variant_id": CALIBRATION_VARIANTS[1].variant_id,
        "target_nms_peaks_per_calibration_unit": 5.0,
        "calibration_burden_unit": DECLARED_BURDEN_UNIT,
        "source_frame_ui": 1900,
    }
    for row in frame_rows:
        if (
            row["variant_id"] == first["variant_id"]
            and row["target_nms_peaks_per_calibration_unit"] == 0.25
            and row["source_frame_ui"] == 101
        ) or (
            row["variant_id"] == second["variant_id"]
            and row["target_nms_peaks_per_calibration_unit"] == 5.0
            and row["source_frame_ui"] == 1900
        ):
            row["proposal_count"] = 1
    return FullRecordingExecution(
        candidate_rows=(first, second),
        frame_count_rows=tuple(frame_rows),
        threshold_rows=_threshold_rows(),
        execution_summary={
            "causal_source_passes": 1,
            "causal_history_processed_ui": [1, 2359],
        },
    )


def test_validation_rejects_preboundary_candidate_and_incomplete_burden_grid() -> None:
    execution = _mock_execution()
    candidate = dict(execution.candidate_rows[0])
    candidate["source_frame_ui"] = 100
    preboundary = FullRecordingExecution(
        candidate_rows=(candidate, execution.candidate_rows[1]),
        frame_count_rows=execution.frame_count_rows,
        threshold_rows=execution.threshold_rows,
        execution_summary=execution.execution_summary,
    )
    with pytest.raises(RuntimeError, match="candidate_values_valid"):
        _validate_execution(
            preboundary,
            arm="raw",
            context_id="support_support_a_h19_g5_n9_m1",
        )

    incomplete = FullRecordingExecution(
        candidate_rows=execution.candidate_rows,
        frame_count_rows=execution.frame_count_rows,
        threshold_rows=execution.threshold_rows[:-1],
        execution_summary=execution.execution_summary,
    )
    with pytest.raises(RuntimeError, match="threshold_row_count_exact"):
        _validate_execution(
            incomplete,
            arm="raw",
            context_id="support_support_a_h19_g5_n9_m1",
        )


def test_mock_run_writes_sealed_reconciled_annotation_file_free_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import neurobench.experiments.gamma_ls_difference.full_recording as module

    sources = _MovieOnlySources(tmp_path / "movie-never-opened.npy")
    config = _minimal_config(tmp_path, sources)
    reference = GammaReferenceSpec.from_mode(
        "support_support_a_h19_g5_n9_m1",
        support_width_px=39,
        shape_n=9.0,
        mode_radius_px=19.0,
        guard_radius_px=5,
    )
    monkeypatch.setattr(
        module,
        "_verify_preflight_without_annotation_reads",
        lambda *_: {
            "movie_sha256": "movie",
            "preflight_sha256": "preflight",
        },
    )
    monkeypatch.setattr(
        module,
        "_verify_support_context",
        lambda *_: (
            reference,
            {
                "artifact_index_sha256": "support",
                "support_screen_burst_windows_used": True,
            },
        ),
    )
    monkeypatch.setattr(
        module,
        "require_cuda_device",
        lambda _: {"resolved_device": "cuda:0", "ready": True},
    )
    output = tmp_path / "full-recording"
    summary = run_full_recording_proposals(
        config,
        preflight_dir=tmp_path / "unused-preflight",
        support_dir=tmp_path / "unused-support",
        output_dir=output,
        arm="raw",
        context_id="support_support_a_h19_g5_n9_m1",
        _executor=_mock_execution,
    )
    assert summary["status"] == "complete_proposal_ledger_scientific_audit_pending"
    assert summary["proposal_row_count"] == 2
    seal = json.loads((output / "candidate_seal.json").read_text(encoding="utf-8"))
    assert seal["annotation_sources_opened_before_seal"] is False
    assert seal["annotation_sources_opened_after_seal"] is False
    assert seal["temporal_linking_applied"] is False
    assert seal["candidate_table"]["rows"] == 2
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert validation["all_checks_pass"] is True
    assert sources.requested == ["movie"]


def test_module_has_no_annotation_reader_api_or_source_key_access() -> None:
    import neurobench.experiments.gamma_ls_difference.full_recording as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "_read_sparse_positives",
        "protected_labels_v1\"]",
        "latest_labels_v7\"]",
        "csv.DictReader",
    ):
        assert forbidden not in source
