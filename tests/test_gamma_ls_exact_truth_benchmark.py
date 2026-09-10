from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import exact_truth_benchmark as benchmark


REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples/gamma_ls_exact_truth_benchmark_v1.example.json"


@pytest.fixture(scope="module")
def config() -> benchmark.ExactTruthConfig:
    return benchmark.ExactTruthConfig.load(EXAMPLE)


def _changed_config(
    tmp_path: Path,
    payload: dict,
) -> Path:
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _metric_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "group": "a",
        "active_truth_count": 2,
        "proposal_count": 3,
        "tp": 1,
        "fp": 2,
        "fn": 1,
        "duplicate_false_positive_count": 1,
        "background_false_positive_count": 1,
        "localization_distance_sum_px": 0.5,
        "localized_match_count": 1,
    }
    row.update(changes)
    return row


def test_config_freezes_nine_framewise_pipelines_and_split_manifest(
    config: benchmark.ExactTruthConfig,
) -> None:
    assert benchmark.pipeline_ids() == tuple(
        f"{representation}__{operator}"
        for representation in benchmark.REPRESENTATIONS
        for operator in benchmark.OPERATORS
    )
    assert len(benchmark.pipeline_ids()) == 9
    assert config.payload["operators"]["radial_gamma"] == {
        "half_width_px": 15,
        "guard_radius_px": 7,
        "shape_n": 9.0,
        "mode_fraction_of_half_width": 0.5,
        "support_geometry": "disk",
        "boundary_mode": "valid_renormalized_zero",
    }
    assert config.payload["evaluation"]["temporal_max_pooling"] is False
    assert config.payload["evaluation"]["time_collapsed_site_score"] is False
    assert config.payload["evaluation"]["candidate_budget"] == "none_threshold_only"

    specs = benchmark.movie_specs(config)
    assert len(specs) == 130
    assert sum(row.split == "scale_floor_fit" for row in specs) == 9
    assert sum(row.split == "threshold_calibration" for row in specs) == 12
    assert sum(row.split == "evaluation" for row in specs) == 108
    assert sum(row.split == "latency" for row in specs) == 1
    assert {
        row.family for row in specs if row.split == "evaluation" and row.family_holdout
    } == set(benchmark.FAMILY_HOLDOUTS)


def test_simulator_emits_exact_dynamic_truth_with_deterministic_identities(
    config: benchmark.ExactTruthConfig,
) -> None:
    spec = benchmark.MovieSpec(
        split="evaluation",
        family="source_specific_nonrigid_motion",
        seed=1001,
        source_count=4,
        family_holdout=True,
    )
    first = benchmark.simulate_exact_truth_movie(config, spec)
    second = benchmark.simulate_exact_truth_movie(config, spec)

    assert first.values.shape == (96, 64, 64)
    assert first.centers_yx.shape == (96, 4, 2)
    assert first.traces.shape == first.events.shape == first.active.shape == (96, 4)
    assert np.array_equal(first.active, first.traces >= 0.12)
    assert np.all(np.ptp(first.centers_yx, axis=0) > 0.5)
    assert len(first.source_ids) == len(set(first.source_ids)) == 4
    assert all(source_id.startswith(spec.movie_id) for source_id in first.source_ids)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.centers_yx, second.centers_yx)
    np.testing.assert_array_equal(first.events, second.events)
    assert first.truth_sha256 == second.truth_sha256

    active_frame = int(np.flatnonzero(first.active.any(axis=1))[0])
    labels = first.active_labels(active_frame)
    assert labels
    assert {"source_id", "x_px", "y_px"} == set(labels[0])

    null_spec = next(
        row for row in benchmark.movie_specs(config) if row.split == "scale_floor_fit"
    )
    null_movie = benchmark.simulate_exact_truth_movie(config, null_spec)
    assert null_movie.centers_yx.shape == (96, 0, 2)
    assert null_movie.active_labels(1) == []
    assert len(null_movie.truth_sha256) == 64


@pytest.mark.parametrize("family", benchmark.FAMILIES)
def test_all_six_families_preserve_exhaustive_source_frame_truth(
    config: benchmark.ExactTruthConfig,
    family: str,
) -> None:
    movie = benchmark.simulate_exact_truth_movie(
        config,
        benchmark.MovieSpec(
            split="evaluation",
            family=family,
            seed=1002,
            source_count=2,
            family_holdout=family in benchmark.FAMILY_HOLDOUTS,
        ),
    )
    assert movie.centers_yx.shape[0] == len(movie.values)
    assert movie.active.shape == (len(movie.values), 2)
    assert np.isfinite(movie.values).all()
    assert all(len(movie.active_labels(frame)) == int(movie.active[frame].sum()) for frame in range(96))


def test_split_overlap_and_source_present_calibration_fail_closed(
    config: benchmark.ExactTruthConfig,
    tmp_path: Path,
) -> None:
    overlapping = copy.deepcopy(config.payload)
    overlapping["splits"]["threshold_calibration"]["seeds"][0] = overlapping[
        "splits"
    ]["evaluation"]["seeds"][0]
    with pytest.raises(benchmark.ExactTruthBenchmarkError, match="split seeds overlap"):
        benchmark.ExactTruthConfig.load(_changed_config(tmp_path, overlapping))

    source_present = copy.deepcopy(config.payload)
    source_present["splits"]["threshold_calibration"]["source_count"] = 2
    with pytest.raises(benchmark.ExactTruthBenchmarkError, match="zero-source"):
        benchmark.ExactTruthConfig.load(_changed_config(tmp_path, source_present))


def test_source_count_cannot_enter_threshold_or_candidate_budget_contract(
    config: benchmark.ExactTruthConfig,
    tmp_path: Path,
) -> None:
    assert "source_count" not in inspect.signature(
        benchmark.calibrate_empirical_thresholds
    ).parameters

    source_derived = copy.deepcopy(config.payload)
    source_derived["calibration"]["source_count_used"] = 4
    source_derived["calibration"]["source_count_derived_threshold_or_budget"] = True
    with pytest.raises(benchmark.ExactTruthBenchmarkError, match="calibration contract"):
        benchmark.ExactTruthConfig.load(_changed_config(tmp_path, source_derived))

    budgeted = copy.deepcopy(config.payload)
    budgeted["evaluation"]["candidate_budget"] = "source_count_times_two"
    with pytest.raises(benchmark.ExactTruthBenchmarkError, match="evaluation contract"):
        benchmark.ExactTruthConfig.load(_changed_config(tmp_path, budgeted))

    mislabeled_profile = copy.deepcopy(config.payload)
    mislabeled_profile["empirical_noise_profile"]["label_use"] = "coordinates"
    with pytest.raises(benchmark.ExactTruthBenchmarkError, match="profile changed"):
        benchmark.ExactTruthConfig.load(
            _changed_config(tmp_path, mislabeled_profile)
        )


def test_empirical_thresholds_respect_fixed_null_proposal_burdens() -> None:
    peak_frames = [
            np.asarray([10.0 - frame, 5.0 - frame / 10.0], dtype=np.float64)
            for frame in range(8)
        ]
    peaks = {pipeline: peak_frames for pipeline in benchmark.pipeline_ids()}
    thresholds, rows = benchmark.calibrate_empirical_thresholds(peaks)
    assert set(thresholds) == {
        (pipeline, burden)
        for pipeline in benchmark.pipeline_ids()
        for burden in benchmark.BURDENS
    }
    assert len(rows) == len(benchmark.pipeline_ids()) * len(benchmark.BURDENS)
    for row in rows:
        assert row["calibration_source_count"] == 0
        assert row["threshold_depends_on_source_count"] is False
        assert row["candidate_budget_used"] is False
        assert row["achieved_null_nms_proposals_per_frame"] <= row[
            "target_nms_proposals_per_frame"
        ]


def test_all_operators_retain_aligned_frame_axis_without_temporal_pooling() -> None:
    rng = np.random.default_rng(7)
    movie = rng.normal(size=(4, 64, 64)).astype(np.float32)
    representations = benchmark.aligned_representations(movie)
    assert all(tuple(value.shape) == (3, 64, 64) for value in representations.values())

    for representation_name in benchmark.REPRESENTATIONS:
        for operator in benchmark.OPERATORS:
            result = benchmark.score_representation(
                representations[representation_name],
                operator=operator,
                scale_floor=None if operator == benchmark.OPERATORS[2] else 0.1,
                chunk_frames=2,
            )
            assert tuple(result.shape) == (3, 64, 64)
            assert np.isfinite(result.detach().cpu().numpy()).all()


def test_framewise_nms_is_deterministic_and_strict_threshold_filter_is_equivalent() -> None:
    score = np.zeros((2, 32, 32), dtype=np.float32)
    score[:, 8, 8] = (4.0, 3.0)
    score[:, 20, 21] = (2.0, 5.0)
    score[:, 15, 15] = 1.0
    first = benchmark.frame_nms_candidates(score, nms_distance_px=3)
    second = benchmark.frame_nms_candidates(score.copy(), nms_distance_px=3)
    assert first == second

    threshold = 1.5
    filtered = [[peak for peak in peaks if peak[0] > threshold] for peaks in first]
    direct = [
        benchmark.extract_separated_local_maxima(
            frame,
            3,
            threshold=float(np.nextafter(threshold, np.inf)),
            limit=10_000,
        )
        for frame in score
    ]
    assert filtered == direct


def test_framewise_one_to_one_matching_reconciles_duplicates_and_localization() -> None:
    truth = [
        {"source_id": "source-a", "x_px": 5.0, "y_px": 5.0},
        {"source_id": "source-b", "x_px": 15.0, "y_px": 15.0},
    ]
    peaks = [
        (9.0, 5, 5),
        (8.0, 6, 5),
        (7.0, 14, 15),
        (6.0, 30, 30),
    ]
    metric, candidates = benchmark.framewise_truth_metrics(
        peaks, truth, match_radius_px=3.0
    )
    assert (metric["tp"], metric["fp"], metric["fn"]) == (2, 2, 0)
    assert metric["duplicate_false_positive_count"] == 1
    assert metric["background_false_positive_count"] == 1
    assert metric["precision"] == pytest.approx(0.5)
    assert metric["recall"] == pytest.approx(1.0)
    assert metric["f1"] == pytest.approx(2.0 / 3.0)
    assert metric["mean_localization_px"] == pytest.approx(0.5)
    assert [row["classification"] for row in candidates] == [
        "true_positive",
        "duplicate_false_positive",
        "true_positive",
        "background_false_positive",
    ]


def test_candidate_and_movie_identities_are_hash_stable() -> None:
    spec = benchmark.MovieSpec("evaluation", "baseline", 1001, 2, False)
    assert spec.movie_id == "etv1__evaluation__baseline__seed01001__sources02"
    assert spec.rng_seed == benchmark.MovieSpec(
        "evaluation", "baseline", 1001, 2, False
    ).rng_seed
    assert spec.rng_seed != benchmark.MovieSpec(
        "threshold_calibration", "baseline", 1001, 2, False
    ).rng_seed

    first = benchmark._candidate_id(spec.movie_id, benchmark.pipeline_ids()[0], 1.0, 3, 1, 5, 7)
    second = benchmark._candidate_id(spec.movie_id, benchmark.pipeline_ids()[0], 1.0, 3, 1, 5, 7)
    assert first == second
    assert first.startswith("etcand_")
    assert first != benchmark._candidate_id(
        spec.movie_id, benchmark.pipeline_ids()[0], 1.0, 4, 1, 5, 7
    )


def test_metric_aggregation_reconciles_exact_integer_counts() -> None:
    rows = [
        _metric_row(),
        _metric_row(
            active_truth_count=1,
            proposal_count=1,
            tp=1,
            fp=0,
            fn=0,
            duplicate_false_positive_count=0,
            background_false_positive_count=0,
            localization_distance_sum_px=1.5,
            localized_match_count=1,
        ),
    ]
    [summary] = benchmark.aggregate_frame_metrics(rows, group_fields=("group",))
    assert summary["frame_count"] == 2
    assert (summary["tp"], summary["fp"], summary["fn"]) == (2, 2, 1)
    assert summary["proposal_count"] == summary["tp"] + summary["fp"] == 4
    assert summary["active_truth_count"] == summary["tp"] + summary["fn"] == 3
    assert summary["duplicate_false_positive_count"] + summary[
        "background_false_positive_count"
    ] == summary["fp"]
    assert summary["precision"] == pytest.approx(0.5)
    assert summary["recall"] == pytest.approx(2.0 / 3.0)
    assert summary["f1"] == pytest.approx(4.0 / 7.0)
    assert summary["mean_localization_px"] == pytest.approx(1.0)
    assert summary["precision_scope"] == "exhaustive_synthetic_truth_only"

    with pytest.raises(
        benchmark.ExactTruthBenchmarkError, match="does not reconcile"
    ):
        benchmark.aggregate_frame_metrics(
            [_metric_row(proposal_count=99)], group_fields=("group",)
        )


def test_plan_freeze_is_hash_bound_noncolliding_and_tamper_evident(
    config: benchmark.ExactTruthConfig,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "plan"
    frozen = benchmark.freeze_plan(config, output_dir=destination)
    assert frozen["status"] == "frozen_before_result_execution"
    verified = benchmark.verify_frozen_plan(config, destination)
    assert verified["artifact_count"] == 4
    assert (destination / "artifact_index.json").is_file()

    with pytest.raises(FileExistsError, match="output exists"):
        benchmark.freeze_plan(config, output_dir=destination)

    with (destination / "plan.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(benchmark.ExactTruthBenchmarkError, match="artifact changed"):
        benchmark.verify_frozen_plan(config, destination)


def test_cpu_only_gate_precedes_plan_read_and_output_mutation(
    config: benchmark.ExactTruthConfig,
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-exist"
    with pytest.raises(benchmark.ExactTruthBenchmarkError, match="CPU-only"):
        benchmark.run_benchmark(
            config,
            plan_dir=tmp_path / "missing-plan",
            output_dir=output,
            device="cuda",
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".*exact-truth-work"))
