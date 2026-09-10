from __future__ import annotations

import hashlib
import inspect
import json

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import independent_validation as module
from neurobench.experiments.gamma_ls_difference.independent_validation import (
    CANDIDATES_PER_BLOCK,
    COMPLETE_BLOCKS,
    EVALUATION_BUDGETS_PER_BLOCK,
    INDEPENDENT_GAMMA_PROTOCOL,
    IndependentGammaValidationError,
    candidate_score_digest,
    candidate_universe_digest,
    evaluate_independent_sparse_positives,
    independent_gamma_protocol_digest,
    load_frozen_independent_selection,
    rank_gamma_block_candidates,
)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selection(
    tmp_path,
    *,
    geometry: str = "disk",
    arm: str = "difference_signed",
    model_payload=None,
):
    summary = tmp_path / "protected_summary.json"
    validation = tmp_path / "protected_validation.json"
    summary.write_text(
        json.dumps(
            {
                "status": "complete_protected_metrics_scientific_audit_pending",
                "protected_v1": {
                    "occurrences": 79,
                    "arm_summary": [{"representation": arm}],
                },
                "context_roles": {
                    "1": [
                        {
                            "context_id": "gamma_h11_g5_n9_m1",
                            "support_width_px": 23,
                            "guard_radius_px": 5.0,
                            "shape": 9.0,
                            "mode_radius_px": 11.0,
                            "support": "radial_disk",
                            "padding": "valid_renormalized_zero",
                        }
                    ]
                },
            }
        )
    )
    validation.write_text(
        json.dumps(
            {
                "status": "passed_protected_metric_artifact_contract",
                "checks": {"candidate_seal_precedes_label_join": True},
                "all_checks_pass": True,
            }
        )
    )
    model_entry = None
    if model_payload is not None:
        model = tmp_path / "transferable_model.json"
        model.write_text(json.dumps(model_payload))
        model_entry = {"path": str(model), "sha256": _sha(model)}
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "frozen_for_external_sparse_positive_confirmation",
                "selection_scope": "post_protected_primary_decision",
                "decision_made_at_utc": "2026-09-08T00:00:00+00:00",
                "decision_rationale": "protected primary comparison selected signed difference",
                "representation": {"arm": arm, "model": model_entry},
                "gamma_context": {
                    "context_id": "gamma_h11_g5_n9_m1",
                    "support_width_px": 23,
                    "shape_n": 9.0,
                    "mode_radius_px": 11.0,
                    "guard_radius_px": 5.0,
                    "support_geometry": geometry,
                    "boundary_mode": "valid_renormalized_zero",
                    "epsilon": 1e-6,
                },
                "protected_evidence": {
                    "summary": {"path": str(summary), "sha256": _sha(summary)},
                    "validation": {"path": str(validation), "sha256": _sha(validation)},
                },
                "candidate_protocol_sha256": independent_gamma_protocol_digest(),
                "external_annotation_content_used_for_selection": False,
                "scientific_audit": {"enabled": True},
            }
        )
    )
    return selection, summary


def _candidate(
    candidate_id: str,
    *,
    block: int,
    rank: int,
    frame: int,
    x: int,
    y: int,
    score: float,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "block_id": block,
        "block_start_frame_zero": (block - 1) * 50,
        "block_stop_frame_exclusive": block * 50,
        "block_start_frame_ui": (block - 1) * 50 + 1,
        "block_stop_frame_ui_inclusive": block * 50,
        "block_rank": rank,
        "pooled_lme_score": score,
        "peak_score_z": score + 1,
        "peak_frame_zero": frame,
        "peak_frame_ui": frame + 1,
        "x_px": x,
        "y_px": y,
        "peak_frame_z_exceeds_descriptive_initial100_quantile": True,
        "candidate_selected_by_empirical_threshold": False,
        "interpretation_before_label_join": "unknown_candidate",
    }


def test_independent_gamma_protocol_freezes_exact_complete_blocks() -> None:
    assert COMPLETE_BLOCKS == 32
    assert INDEPENDENT_GAMMA_PROTOCOL["evaluated_source_frames"] == 1600
    assert INDEPENDENT_GAMMA_PROTOCOL["dropped_remainder_frames"] == 8
    assert INDEPENDENT_GAMMA_PROTOCOL["source_block_frames"] == 50
    assert INDEPENDENT_GAMMA_PROTOCOL["causal_warmup_source_frames"] == 1
    assert INDEPENDENT_GAMMA_PROTOCOL["scored_output_frames"] == 1599
    assert INDEPENDENT_GAMMA_PROTOCOL["scored_source_frame_interval_zero_inclusive"] == [1, 1599]
    assert INDEPENDENT_GAMMA_PROTOCOL["candidate_selection"].endswith(
        "not_empirical_threshold_gated"
    )
    assert "not a held-out independent test" in INDEPENDENT_GAMMA_PROTOCOL[
        "calibration_evaluation_overlap"
    ]
    assert tuple(INDEPENDENT_GAMMA_PROTOCOL["evaluation_budgets_per_block"]) == (
        10,
        20,
        58,
        100,
        200,
    )
    assert CANDIDATES_PER_BLOCK == 200
    assert len(independent_gamma_protocol_digest()) == 64
    assert "eligibility_preflight_inspected_annotation_content" in INDEPENDENT_GAMMA_PROTOCOL[
        "annotation_access_boundary"
    ]


def test_selection_requires_hash_bound_passed_protected_evidence_and_radial_gamma(
    tmp_path,
) -> None:
    selection, summary = _selection(tmp_path)
    loaded = load_frozen_independent_selection(selection)
    assert loaded.arm == "difference_signed"
    assert loaded.gamma_reference.support_geometry == "disk"
    assert loaded.gamma_reference.support_width_px == 23
    assert loaded.representation_model is None

    summary.write_text(json.dumps({"status": "changed"}))
    with pytest.raises(IndependentGammaValidationError, match="hash changed"):
        load_frozen_independent_selection(selection)

    square_dir = tmp_path / "square"
    square_dir.mkdir()
    square, _ = _selection(square_dir, geometry="square")
    with pytest.raises(IndependentGammaValidationError, match="radial disk"):
        load_frozen_independent_selection(square)


def test_selection_accepts_hash_bound_transferable_cs_parzen_model(tmp_path) -> None:
    selection, _ = _selection(
        tmp_path,
        arm="cs_parzen_two_frame",
        model_payload={
            "fit": {
                "mean": [0.0, 0.0],
                "whitening": [[1.0, 0.0], [0.0, 1.0]],
                "demixing": [[0.0, 1.0], [1.0, 0.0]],
                "activity_component": 1,
                "activity_sign": -1,
            }
        },
    )
    loaded = load_frozen_independent_selection(selection)
    assert loaded.arm == "cs_parzen_two_frame"
    assert loaded.representation_model["fit"]["activity_component"] == 1
    assert loaded.representation_model_sha256 is not None


def test_block_candidates_are_deterministic_and_preserve_peak_frames() -> None:
    scores = np.zeros((5, 20, 21), dtype=np.float32)
    scores[:, 6, 6] = np.asarray([0, 1, 5, 1, 0], dtype=np.float32)
    scores[:, 14, 14] = np.asarray([0, 2, 1, 0, 0], dtype=np.float32)
    frames = np.arange(50, 55, dtype=np.int64)
    first, pooled_first = rank_gamma_block_candidates(
        scores,
        frames,
        block_id=2,
        block_start_frame_zero=50,
        block_stop_frame_exclusive=100,
        empirical_threshold_z=1.5,
        candidate_namespace="0123456789ab",
        distance_px=2,
        limit=2,
    )
    second, pooled_second = rank_gamma_block_candidates(
        scores.copy(),
        frames.copy(),
        block_id=2,
        block_start_frame_zero=50,
        block_stop_frame_exclusive=100,
        empirical_threshold_z=1.5,
        candidate_namespace="0123456789ab",
        distance_px=2,
        limit=2,
    )
    assert first == second
    np.testing.assert_array_equal(pooled_first, pooled_second)
    assert first[0]["candidate_id"] == "indg_0123456789ab_b002_r001"
    assert first[0]["peak_frame_zero"] == 52
    assert first[0]["peak_frame_ui"] == 53
    assert first[0]["candidate_selected_by_empirical_threshold"] is False
    assert all(row["interpretation_before_label_join"] == "unknown_candidate" for row in first)

    with pytest.raises(ValueError, match="selection hash prefix"):
        rank_gamma_block_candidates(
            scores,
            frames,
            block_id=2,
            block_start_frame_zero=50,
            block_stop_frame_exclusive=100,
            empirical_threshold_z=1.5,
            candidate_namespace="gamma15r",
            distance_px=2,
            limit=2,
        )


def test_sparse_positive_metrics_hash_and_match_one_to_one() -> None:
    candidates = [
        _candidate("a", block=1, rank=1, frame=12, x=5, y=5, score=3.0),
        _candidate("b", block=2, rank=1, frame=70, x=8, y=9, score=2.0),
        _candidate("c", block=1, rank=2, frame=13, x=6, y=5, score=1.0),
    ]
    scores = np.asarray([row["pooled_lme_score"] for row in candidates], dtype=np.float64)
    annotations = [
        {
            "video_id": "15 right",
            "annotation_id": "15 right:roi_1",
            "roi_id": "1",
            "crop_x": 5.0,
            "crop_y": 5.0,
            "spike_intervals": [{"start_frame": 10, "end_frame": 15}],
        },
        {
            "video_id": "15 right",
            "annotation_id": "15 right:roi_2",
            "roi_id": "2",
            "crop_x": 8.0,
            "crop_y": 9.0,
            "spike_intervals": [{"start_frame": 68, "end_frame": 75}],
        },
    ]
    universe_sha = candidate_universe_digest(candidates)
    score_sha = candidate_score_digest(scores)
    result = evaluate_independent_sparse_positives(
        candidates,
        scores,
        annotations,
        frozen_candidate_universe_sha256=universe_sha,
        frozen_candidate_score_sha256=score_sha,
        budgets=(1, 2),
        expected_known_positive_count=2,
        expected_roi_count=2,
    )
    assert [row["known_positive_recall"] for row in result["metric_rows"]] == [1.0, 1.0]
    assert result["metric_rows"][0]["mean_reciprocal_block_rank"] == 1.0
    assert len(result["match_rows"]) == 4
    assert result["unmatched_candidates"] == "unknown_not_negative"
    assert result["precision_specificity_and_false_positive_rate"] == "not_identified"

    changed = scores.copy()
    changed[0] += 0.25
    with pytest.raises(IndependentGammaValidationError, match="scores changed"):
        evaluate_independent_sparse_positives(
            candidates,
            changed,
            annotations,
            frozen_candidate_universe_sha256=universe_sha,
            frozen_candidate_score_sha256=score_sha,
            budgets=(1,),
            expected_known_positive_count=2,
            expected_roi_count=2,
        )


def test_one_candidate_cannot_match_two_overlapping_positive_intervals() -> None:
    candidates = [_candidate("a", block=1, rank=1, frame=12, x=5, y=5, score=3.0)]
    scores = np.asarray([3.0])
    annotations = [
        {
            "video_id": "15 right",
            "annotation_id": f"15 right:roi_{index}",
            "roi_id": str(index),
            "crop_x": float(5 + index - 1),
            "crop_y": 5.0,
            "spike_intervals": [{"start_frame": 10, "end_frame": 15}],
        }
        for index in (1, 2)
    ]
    result = evaluate_independent_sparse_positives(
        candidates,
        scores,
        annotations,
        frozen_candidate_universe_sha256=candidate_universe_digest(candidates),
        frozen_candidate_score_sha256=candidate_score_digest(scores),
        budgets=(1,),
        expected_known_positive_count=2,
        expected_roi_count=2,
    )
    assert result["metric_rows"][0]["one_to_one_known_positive_matches"] == 1
    assert result["metric_rows"][0]["known_positive_recall"] == 0.5


def test_positive_denominator_requires_overlap_with_actual_scored_frames() -> None:
    candidates = [_candidate("a", block=1, rank=1, frame=1, x=5, y=5, score=3.0)]
    scores = np.asarray([3.0])
    annotations = [
        {
            "video_id": "15 right",
            "annotation_id": "15 right:roi_1",
            "roi_id": "1",
            "crop_x": 5.0,
            "crop_y": 5.0,
            "spike_intervals": [{"start_frame": 0, "end_frame": 0}],
        },
        {
            "video_id": "15 right",
            "annotation_id": "15 right:roi_2",
            "roi_id": "2",
            "crop_x": 5.0,
            "crop_y": 5.0,
            "spike_intervals": [{"start_frame": 1, "end_frame": 1}],
        },
    ]
    result = evaluate_independent_sparse_positives(
        candidates,
        scores,
        annotations,
        frozen_candidate_universe_sha256=candidate_universe_digest(candidates),
        frozen_candidate_score_sha256=candidate_score_digest(scores),
        budgets=(1,),
        expected_known_positive_count=2,
        expected_roi_count=2,
    )
    assert result["known_positive_count_all_source"] == 2
    assert result["known_positive_count_evaluated"] == 1
    assert result["known_positive_count_without_scored_frame_overlap"] == 1
    assert result["metric_rows"][0]["known_positive_recall"] == 1.0


def test_new_gamma_runner_does_not_import_or_reuse_legacy_ica_candidates() -> None:
    source = inspect.getsource(module)
    assert "build_independent_candidate_universe" not in source
    assert "S6_INDEPENDENT_CONFIRMATION/candidate_universe.tsv" not in source
    assert "historical_ica_candidate_universe_reused\": False" in source


def test_annotation_join_reverifies_sealed_score_files_first(tmp_path) -> None:
    annotations = []
    for roi_index in range(10):
        interval_count = 6 if roi_index == 0 else 5
        annotations.append(
            {
                "video_id": "15 right",
                "annotation_id": f"15 right:roi_{roi_index + 1}",
                "roi_id": str(roi_index + 1),
                "crop_x": float(roi_index + 2),
                "crop_y": float(roi_index + 3),
                "spike_intervals": [
                    {"start_frame": 10 + index, "end_frame": 10 + index}
                    for index in range(interval_count)
                ],
            }
        )
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps({"annotations": annotations}))
    candidate_path = tmp_path / "candidate_universe.tsv"
    candidate_path.write_text("candidate_id\nfixture\n")
    scores_path = tmp_path / "candidate_scores.npy"
    maps_path = tmp_path / "block_lme_score_maps.npy"
    np.save(scores_path, np.asarray([1.0], dtype=np.float64), allow_pickle=False)
    np.save(maps_path, np.ones((1, 2, 2), dtype=np.float32), allow_pickle=False)
    authority = module.IndependentSourceAuthority(
        preflight_path=tmp_path / "preflight.json",
        preflight_sha256="0" * 64,
        contract_path=tmp_path / "contract.json",
        contract_sha256="1" * 64,
        movie_path=tmp_path / "movie.tif",
        movie_sha256="2" * 64,
        movie_size_bytes=0,
        annotation_path=annotation_path,
        annotation_sha256=_sha(annotation_path),
        safe_manifest_hashes={},
        preflight={},
        contract={},
    )
    seal_path = tmp_path / "candidate_score_seal.json"
    seal_path.write_text(
        json.dumps(
            {
                "status": "frozen_before_annotation_manifest_open",
                "candidate_universe_sha256": "3" * 64,
                "candidate_score_sha256": candidate_score_digest(np.asarray([1.0])),
                "block_score_sha256": module.block_score_digest(
                    np.ones((1, 2, 2), dtype=np.float32)
                ),
                "candidate_universe_tsv_sha256": _sha(candidate_path),
                "candidate_scores_npy_sha256": _sha(scores_path),
                "block_score_maps_npy_sha256": _sha(maps_path),
            }
        )
    )
    loaded, provenance = module._load_verified_annotations_after_seal(
        authority, seal_path=seal_path
    )
    assert len(loaded) == 10
    assert provenance["spike_interval_count"] == 51

    np.save(scores_path, np.asarray([2.0], dtype=np.float64), allow_pickle=False)
    with pytest.raises(IndependentGammaValidationError, match="sealed candidate artifact"):
        module._load_verified_annotations_after_seal(authority, seal_path=seal_path)


def test_empty_match_table_keeps_auditable_headers(tmp_path) -> None:
    path = tmp_path / "matches.tsv"
    module._atomic_tsv(path, [], fieldnames=module.MATCH_TABLE_FIELDS)
    assert path.read_text().splitlines() == ["\t".join(module.MATCH_TABLE_FIELDS)]
