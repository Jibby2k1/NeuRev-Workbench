from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import (
    fixed_deployment_characterization as fixed,
)


def test_science_contract_is_single_scale_and_post_selection_only() -> None:
    contract = fixed._science_contract()

    assert contract["representation"] == "difference_signed"
    assert contract["context"]["context_id"] == (
        "support_support_a_h15_g7_n9_m0p5"
    )
    assert contract["context"]["support_width_px"] == 31
    assert contract["quiet_cross_fit_swaps"] == list(fixed.QUIET_SWAPS)
    assert contract["quiet_nms_peaks_per_pseudo_burst"] == list(
        fixed.QUIET_NMS_PEAK_BURDENS
    )
    assert contract["protected_selection_estimate"] is False
    assert contract["independent_confirmation"] is False
    assert contract["full_record_q1_proposal_stream_covered"] is False


def test_build_label_free_evidence_persists_peak_frames_without_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operating = [
        {
            "target_nms_peaks_per_pseudo_burst": burden,
            "threshold_z": burden,
            "achieved_nms_peaks_per_pseudo_burst": burden,
        }
        for burden in fixed.QUIET_NMS_PEAK_BURDENS
    ]
    monkeypatch.setattr(
        fixed,
        "calibrate_training_quiet_thresholds",
        lambda *args, **kwargs: SimpleNamespace(operating_points=operating),
    )
    monkeypatch.setattr(
        fixed,
        "duration_matched_quiet_windows",
        lambda mask, durations: {
            key: (index, index + value)
            for index, (key, value) in enumerate(durations.items())
        },
    )

    def fake_extract(scores, windows, *, threshold_z, nms_distance_px):
        shape = np.asarray(scores).shape[1:]
        maps = {key: np.full(shape, threshold_z, dtype=np.float32) for key in windows}
        peaks = {key: ((3.0, 1, 2),) for key in windows}
        return SimpleNamespace(occupancy_maps=maps, peaks=peaks)

    monkeypatch.setattr(fixed, "extract_burst_candidates", fake_extract)
    frame_ui = np.arange(1800, 2360, dtype=np.int64)
    scores = np.zeros((560, 8, 9), dtype=np.float32)
    # The deterministic peak for every configured burst is its second frame.
    for start_ui in (2003, 2040, 2122, 2254):
        scores[start_ui - 1800 + 1, 2, 1] = 9.0
    burst_intervals = {
        "1": (2003, 2026),
        "2": (2040, 2063),
        "3": (2122, 2149),
        "4": (2254, 2300),
    }
    candidates, calibrations, occupancy, index = (
        fixed.build_label_free_candidate_evidence(
            {fixed.QUIET_SWAPS[0]: scores, fixed.QUIET_SWAPS[1]: scores},
            frame_ui=frame_ui,
            burst_intervals_ui=burst_intervals,
            quiet_half_a_ui=(1800, 1849),
            quiet_half_b_ui=(1850, 1899),
            scale_floors={swap: 0.25 for swap in fixed.QUIET_SWAPS},
        )
    )

    assert len(calibrations) == 10
    assert len(index) == 40
    assert len(candidates) == 40
    assert occupancy.shape == (2, 5, 4, 8, 9)
    assert len({row["candidate_id"] for row in candidates}) == 40
    for row in candidates:
        start_ui = burst_intervals[str(row["burst_id"])][0]
        assert row["peak_source_frame_ui"] == start_ui + 1
        assert row["interpretation_before_label_join"] == "unknown_candidate"
    assert all(row["positive_coordinates_used"] is False for row in calibrations)


def test_crossfit_summary_keeps_swaps_and_unknown_candidate_boundary() -> None:
    observation_rows = []
    candidate_rows = []
    for swap in fixed.QUIET_SWAPS:
        for burden in fixed.QUIET_NMS_PEAK_BURDENS:
            for burst in (1, 2, 3, 4):
                candidate_rows.append(
                    {
                        "quiet_swap": swap,
                        "target_nms_peaks_per_pseudo_burst": burden,
                        "burst_id": burst,
                    }
                )
            for budget in fixed.CANDIDATE_BUDGETS_PER_BURST:
                for index in range(fixed.EXPECTED_PROTECTED_V1_ROWS):
                    observation_rows.append(
                        {
                            "quiet_swap": swap,
                            "target_nms_peaks_per_pseudo_burst": burden,
                            "candidate_budget": budget,
                            "matched": index == 0,
                        }
                    )

    summary = fixed._crossfit_summary(observation_rows, candidate_rows)

    assert len(summary) == 25
    assert all(
        row["paired_quiet_crossfit_mean_known_positive_recall"]
        == pytest.approx(1 / fixed.EXPECTED_PROTECTED_V1_ROWS)
        for row in summary
    )
    assert all(row["precision_identified"] is False for row in summary)
    assert all(row["unmatched_candidates"] == "unknown_not_negative" for row in summary)


def test_run_refuses_to_create_work_without_explicit_gpu_authorization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = SimpleNamespace()
    monkeypatch.setattr(fixed.GammaLSDifferenceConfig, "load", lambda path: config)
    monkeypatch.setattr(
        fixed,
        "_verify_replay_preflight",
        lambda cfg, path: {
            "replay_preflight": {"artifact_index_sha256": "a" * 64},
            "science_contract_sha256": "b" * 64,
        },
    )
    output = tmp_path / "result"

    with pytest.raises(
        fixed.FixedDeploymentCharacterizationUnavailable,
        match="explicit --gpu-authorized",
    ):
        fixed.run_fixed_deployment_characterization(
            tmp_path / "config.json",
            replay_preflight_dir=tmp_path / "preflight",
            output_dir=output,
            gpu_authorized=False,
        )

    assert not output.exists()
    assert not (tmp_path / f".{output.name}.fixed-deployment-work").exists()


def test_join_creates_audit_inputs_parent_before_atomic_tables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    label_path = tmp_path / "labels.tsv"
    label_path.write_text("sealed label fixture\n", encoding="utf-8")
    (tmp_path / "candidate_score_seal.json").write_text("{}\n", encoding="utf-8")
    positives = [
        {
            "observation_id": f"p{index:03d}",
            "burst_id": 1,
            "canonical_roi_id": f"roi{index % 3}",
            "x_px": 1.0,
            "y_px": 2.0,
        }
        for index in range(fixed.EXPECTED_PROTECTED_V1_ROWS)
    ]
    candidate = {field: "" for field in fixed.CANDIDATE_FIELDS}
    candidate.update(
        {
            "candidate_id": "candidate_1",
            "burst_id": 1,
            "quiet_swap": fixed.DEFAULT_AUDIT_SWAP,
            "target_nms_peaks_per_pseudo_burst": fixed.DEFAULT_AUDIT_BURDEN,
            "candidate_rank": 1,
        }
    )
    observations = [
        {
            "quiet_swap": fixed.DEFAULT_AUDIT_SWAP,
            "target_nms_peaks_per_pseudo_burst": fixed.DEFAULT_AUDIT_BURDEN,
            "candidate_budget": fixed.PRIMARY_AUDIT_BUDGET,
            "observation_id": row["observation_id"],
            "matched": False,
        }
        for row in positives
    ]
    monkeypatch.setattr(fixed, "_verify_candidate_seal", lambda work: {"ok": True})
    real_sha = fixed._sha256
    monkeypatch.setattr(
        fixed, "_sha256", lambda path: "label-hash" if Path(path) == label_path else real_sha(Path(path))
    )
    monkeypatch.setattr(fixed, "_read_sparse_positives", lambda *args, **kwargs: positives)

    def fake_read(path: Path):
        return [candidate] if Path(path).name == "candidates_label_sealed.tsv" else [{"quiet_swap": fixed.DEFAULT_AUDIT_SWAP}]

    monkeypatch.setattr(fixed, "_read_tsv", fake_read)
    monkeypatch.setattr(fixed, "observation_match_rows", lambda *args, **kwargs: observations)
    monkeypatch.setattr(fixed, "aggregate_match_rows", lambda rows: [{"rows": len(rows)}])
    monkeypatch.setattr(fixed, "_crossfit_summary", lambda *args: [{"rows": 1}])
    config = SimpleNamespace(
        source_paths={"protected_labels_v1": label_path},
        payload={
            "frames": {
                "burst_intervals_ui": {
                    "1": [2003, 2026],
                    "2": [2040, 2063],
                    "3": [2122, 2149],
                    "4": [2254, 2300],
                }
            },
            "sources": {"movie": "data://movie.npy"},
        },
    )
    preflight = {
        "inputs": {
            "base_preflight": {
                "protected_v1_sha256": "label-hash",
                "source_movie_sha256": "movie-hash",
            }
        },
        "science_contract": fixed._science_contract(),
    }

    fixed._write_join_and_audit_inputs(config, tmp_path, preflight)

    assert (tmp_path / "audit_inputs" / "expert_occurrences.tsv").is_file()
    assert (tmp_path / "audit_inputs" / "model_occurrences.tsv").is_file()
    assert (tmp_path / "audit_inputs" / "one_to_one_matches.tsv").is_file()
    assert (tmp_path / "audit_inputs" / "stage_sources.json").is_file()


def test_rebind_sealed_resume_allows_only_packaging_hotfix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    work = tmp_path / "work"
    old_root = tmp_path / "old_preflight"
    work.mkdir()
    old_root.mkdir()
    (work / "candidate_score_seal.json").write_text("candidate\n", encoding="utf-8")
    (work / "stage_array_manifest.json").write_text("stage\n", encoding="utf-8")
    science = fixed._science_contract()
    module_key = (
        "repo://neurobench/experiments/gamma_ls_difference/"
        "fixed_deployment_characterization.py"
    )
    test_key = "repo://tests/test_gamma_ls_fixed_deployment_characterization.py"
    unchanged_key = "repo://neurobench/algorithms/gamma_local_standardization.py"
    old_preflight = {
        "science_contract": science,
        "implementation": {
            module_key: {"sha256": "old-module"},
            test_key: {"sha256": "old-test"},
            unchanged_key: {"sha256": "same"},
        },
    }
    (old_root / "preflight.json").write_text(
        json.dumps(old_preflight), encoding="utf-8"
    )
    base_contract = {
        "schema_version": 1,
        "experiment_id": fixed.EXPERIMENT_ID,
        "destination": str(tmp_path / "result"),
        "science_contract_sha256": fixed._canonical_sha256(science),
        "device": "cuda",
        "chunk_frames": 64,
    }
    existing = {
        **base_contract,
        "replay_preflight_root": str(old_root),
        "replay_preflight_artifact_index_sha256": "old-index",
    }
    requested = {
        **base_contract,
        "replay_preflight_root": str(tmp_path / "new_preflight"),
        "replay_preflight_artifact_index_sha256": "new-index",
    }
    requested_preflight = {
        "science_contract": science,
        "implementation": {
            module_key: {"sha256": "new-module"},
            test_key: {"sha256": "new-test"},
            unchanged_key: {"sha256": "same"},
        },
    }
    monkeypatch.setattr(fixed, "_verify_stage_manifest", lambda root: {"ok": True})
    monkeypatch.setattr(
        fixed,
        "_verify_candidate_seal",
        lambda root: {"candidate_rows": 138},
    )
    monkeypatch.setattr(
        fixed,
        "verify_indexed_artifact",
        lambda root: {"artifact_index_sha256": "old-index"},
    )
    monkeypatch.setattr(fixed, "_heartbeat", lambda *args, **kwargs: None)

    fixed._rebind_sealed_resume_contract(
        work,
        existing_contract=existing,
        requested_contract=requested,
        requested_preflight=requested_preflight,
    )

    assert json.loads((work / "work_contract.json").read_text()) == requested
    rebind = json.loads((work / "preflight_rebind.json").read_text())
    assert rebind["cuda_or_scoring_reexecuted"] is False
    assert rebind["candidate_membership_changed"] is False
    assert set(rebind["changed_implementation_files"]) == {module_key, test_key}
