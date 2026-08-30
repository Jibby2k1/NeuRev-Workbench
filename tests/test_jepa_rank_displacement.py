from __future__ import annotations

import json

import numpy as np
import pytest
import scipy

from neurobench.experiments.neuron_identifiability import jepa_rank_displacement as rank
from neurobench.experiments.neuron_identifiability import jepa_residual_pilot_v1_1 as residual


def test_classification_keeps_native_competitors_unknown() -> None:
    assert rank.classify_source(source_on_recovered=False, intervention_recovered=True) == "native_background_competition"
    assert rank.classify_source(source_on_recovered=False, intervention_recovered=False) == "attenuation_or_response_failure"
    assert rank.classify_source(source_on_recovered=True, intervention_recovered=True) == "recovered_source_on"
    assert rank.classify_source(source_on_recovered=True, intervention_recovered=False) == "source_on_only_inconsistent"


def test_rank_breaks_score_ties_by_row_then_column() -> None:
    values = np.asarray([[2.0, 2.0], [3.0, 2.0]])
    assert rank._rank_with_ties(values, 1, 0) == 1
    assert rank._rank_with_ties(values, 0, 0) == 2
    assert rank._rank_with_ties(values, 0, 1) == 3
    assert rank._rank_with_ties(values, 1, 1) == 4


def test_score_pair_preserves_top_four_candidate_contract() -> None:
    off = np.zeros((32, 16, 16), dtype=np.float32)
    on = off.copy()
    on[:, 8, 8] = np.linspace(0.0, 5.0, 32, dtype=np.float32)
    pair = rank._score_pair(off, on, candidate_budget=4, minimum_distance_px=2, border_px=2)
    assert pair.source_on_z.shape == (16, 16)
    assert len(pair.source_on_candidates) <= 4
    assert np.isfinite(pair.source_on_z).all()
    assert np.isfinite(pair.intervention_z).all()


def test_grouped_summary_keeps_source_count_nested() -> None:
    rows = []
    for method in rank.METHODS:
        for recording in ("r1", "r2"):
            for window in ("w1", "w2"):
                for seed in (1, 2, 3):
                    for source_count in (1, 2, 4):
                        rows.append({
                            "method": method,
                            "background_recording_id": recording,
                            "background_window_id": window,
                            "injection_seed": seed,
                            "source_count": source_count,
                            "source_on_recovered": method == residual.RAW_METHOD,
                            "intervention_recovered": True,
                            "source_on_top4_displaced": method != residual.RAW_METHOD,
                            "classification": "recovered_source_on" if method == residual.RAW_METHOD else "native_background_competition",
                        })
    summary = rank.summarize_rank_displacement(rows, bootstrap_seed=6203, draws=1000)
    value = summary["competition_among_source_on_misses"][residual.JEPA_RESIDUAL_METHOD]
    assert value["cluster_count"] == 12
    assert value["nested_not_resampled_as_independent"] == ["source_count"]
    assert value["observed_mean"] == 1.0


def test_recovery_totals_include_source_on_only_inconsistent_rows() -> None:
    counts = {
        method: {
            "recovered_source_on": 10,
            "native_background_competition": 20,
            "attenuation_or_response_failure": 5,
            "source_on_only_inconsistent": 2,
        }
        for method in rank.METHODS
    }

    totals = rank._recovery_totals(counts)

    assert totals[residual.RAW_METHOD]["total_source_on_recovered"] == 12
    assert totals[residual.RAW_METHOD]["total_intervention_recovered"] == 30
    assert totals[residual.RAW_METHOD]["recovered_on_both_source_on_and_intervention"] == 10


def test_source_count_stratum_keeps_value_and_row_denominator_separate() -> None:
    rows = []
    for method in rank.METHODS:
        for source_index in range(3):
            rows.append(
                {
                    "method": method,
                    "source_count": 2,
                    "source_on_recovered": source_index == 0,
                    "intervention_recovered": source_index < 2,
                    "classification": (
                        "recovered_source_on"
                        if source_index == 0
                        else "native_background_competition"
                        if source_index == 1
                        else "attenuation_or_response_failure"
                    ),
                    "source_on_top4_cutoff_margin_z": -float(source_index),
                    "source_on_local_pixel_rank": source_index + 1,
                }
            )

    summary = rank._stratified_summary(rows, "source_count")
    raw = next(row for row in summary if row["method"] == residual.RAW_METHOD)

    assert raw["source_count"] == "2"
    assert raw["source_row_count"] == 3


def test_rank_f_pins_every_live_numerical_dependency() -> None:
    assert rank.DIAGNOSTIC_ID.endswith("-F")
    assert set(rank.NUMERICAL_DEPENDENCIES) == {
        "neurobench/experiments/neuron_identifiability/jepa_rank_displacement.py",
        "neurobench/experiments/neuron_identifiability/jepa_comparators.py",
        "neurobench/experiments/neuron_identifiability/jepa_evaluation.py",
        "neurobench/experiments/neuron_identifiability/jepa_background_residual.py",
        "neurobench/experiments/neuron_identifiability/jepa_residual_pilot_v1_1.py",
        "neurobench/experiments/neuron_identifiability/contracts.py",
        "neurobench/experiments/neuron_identifiability/discovery.py",
    }


def test_rank_f_timing_contract_separates_preflight_and_execution() -> None:
    timing = rank._phase_timing(
        planned_at="2026-08-30T10:00:00+00:00",
        preflight_started_at="2026-08-30T10:00:00.100000+00:00",
        started_at="2026-08-30T10:00:00.350000+00:00",
        ended_at="2026-08-30T10:00:12.600000+00:00",
    )

    assert timing["preflight_duration_seconds"] == 0.25
    assert timing["duration_seconds"] == 12.25
    assert timing["planned_to_ended_duration_seconds"] == 12.6
    assert timing["primary_duration_field"] == "duration_seconds"
    assert timing["phase_contract"]["execution"] == {
        "started_at_field": "started_at",
        "ended_at_field": "ended_at",
        "duration_field": "duration_seconds",
    }


def test_rank_f_timing_contract_rejects_out_of_order_phases() -> None:
    with pytest.raises(rank.RankDisplacementError, match="not monotonic"):
        rank._phase_timing(
            planned_at="2026-08-30T10:00:00+00:00",
            preflight_started_at="2026-08-30T10:00:01+00:00",
            started_at="2026-08-30T10:00:00.500000+00:00",
            ended_at="2026-08-30T10:00:02+00:00",
        )


def test_rank_f_runtime_provenance_records_scipy_version() -> None:
    runtime = rank._runtime_provenance(
        device="cpu",
        old_threads=4,
        configured_threads=4,
        prediction_amp=False,
    )

    assert runtime["scipy_version"] == scipy.__version__


def test_complete_tree_index_detects_hash_drift(tmp_path) -> None:
    (tmp_path / "result.json").write_text('{"value": 1}\n', encoding="utf-8")
    index = rank._artifact_index(tmp_path)
    (tmp_path / "artifact_index.json").write_text(
        json.dumps(index, sort_keys=True) + "\n", encoding="utf-8"
    )
    validation = rank._verify_output_tree(tmp_path)
    assert validation["all_existing_non_index_files_indexed"] is True
    assert validation["all_indexed_hashes_and_sizes_exact"] is True
    (tmp_path / "result.json").write_text('{"value": 2}\n', encoding="utf-8")
    with pytest.raises(rank.RankDisplacementError, match="hash/size mismatch"):
        rank._verify_output_tree(tmp_path)


def test_workstation_path_scan_fails_closed(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "bad.json").write_text(
        json.dumps({"path": str(tmp_path.resolve())}), encoding="utf-8"
    )
    with pytest.raises(rank.RankDisplacementError, match="absolute path leaked"):
        rank._assert_no_workstation_paths(output, (tmp_path,))
