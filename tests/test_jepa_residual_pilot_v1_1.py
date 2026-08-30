from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from neurobench.experiments.neuron_identifiability import (
    jepa_residual_pilot_v1_1 as pilot,
)
from neurobench.experiments.neuron_identifiability import jepa_residual_pilot as legacy_pilot
from neurobench.experiments.neuron_identifiability.jepa_data import RobustNormalization


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parent(root: Path, payload: bytes = b"immutable") -> tuple[Path, str]:
    artifact = root / "artifact.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(payload)
    index = {
        "schema_version": 1,
        "artifact_count": 1,
        "artifacts": [
            {
                "path": "artifact.bin",
                "bytes": len(payload),
                "sha256": _sha(artifact),
            }
        ],
    }
    index_path = root / "artifact_index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return artifact, _sha(index_path)


def _normalization() -> RobustNormalization:
    return RobustNormalization(
        center=10.0,
        scale=2.0,
        unscaled_mad=1.0,
        scale_was_floored=False,
        fit_recording_ids=("train",),
        uniform_frame_indices=(),
        spatial_stride=1,
    )


def test_maintained_config_loads_with_strict_screen_budget() -> None:
    config = pilot.load_jepa_residual_config(
        Path("examples/conditional_background_residual_v1_1.example.json")
    )
    assert config["experiment_id"] == "NREV-EXP-0029"
    assert config["runner_version"] == "1.1"
    assert config["run_id"] == "NREV-RUN-EXP-0029-SCREEN-20260830-B"
    assert config["output_root"].endswith(
        "NREV-RUN-EXP-0029-SCREEN-20260830-B"
    )
    assert config["output_root"] == pilot.STRICT_OUTPUT_ROOT
    assert config["execution_modes"]["screen"]["decoder_steps"] == 500
    assert config["execution_modes"]["screen"]["fixture_cells_per_background"] == 9
    assert config["_decoder"]["seed"] == 6201
    assert config["_decoder"]["training_seed"] == 6202
    assert "prediction_coverage.json" in config["required_outputs"]
    raw_arm = next(arm for arm in config["arms"] if arm["id"] == "raw_hc")
    assert raw_arm["movie"] == "native_observed_video_float32_parent_anchor_domain"


def test_v1_1_loader_rejects_run_a_descriptor(tmp_path: Path) -> None:
    config = json.loads(
        Path("examples/conditional_background_residual_v1_1.example.json").read_text()
    )
    config["run_id"] = "NREV-RUN-EXP-0029-SCREEN-20260830-A"
    config["output_root"] = (
        "Outputs/NeuronIdentifiability/NREV-EXP-0029/runs/"
        "NREV-RUN-EXP-0029-SCREEN-20260830-A"
    )
    path = tmp_path / "run-a.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(pilot.JEPAResidualPilotError, match="Run-B"):
        pilot.load_jepa_residual_config(path)


def test_legacy_v1_loader_rejects_v1_1_run_b_descriptor() -> None:
    with pytest.raises(legacy_pilot.JEPAResidualPilotError, match="schema_version"):
        legacy_pilot.load_jepa_residual_config(
            Path("examples/conditional_background_residual_v1_1.example.json")
        )


def test_v1_1_loader_rejects_canonical_output_root_drift(tmp_path: Path) -> None:
    config = json.loads(
        Path("examples/conditional_background_residual_v1_1.example.json").read_text()
    )
    config["output_root"] = pilot.STRICT_OUTPUT_ROOT + "-redirected"
    path = tmp_path / "drifted-output.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(pilot.JEPAResidualPilotError, match="canonical Run-B output_root"):
        pilot.load_jepa_residual_config(path)


def test_screen_output_override_rejected_in_preflight_and_run_paths(
    tmp_path: Path,
) -> None:
    config = Path("examples/conditional_background_residual_v1_1.example.json")
    override = tmp_path / "redirected-screen"
    with pytest.raises(pilot.JEPAResidualPilotError, match="forbids output_root overrides"):
        pilot.preflight_jepa_residual_pilot(
            config,
            repository_root=Path.cwd(),
            mode="screen",
            output_root=override,
        )
    with pytest.raises(pilot.JEPAResidualPilotError, match="forbids output_root overrides"):
        pilot.run_jepa_residual_pilot(
            config,
            repository_root=Path.cwd(),
            mode="screen",
            output_root=override,
        )


def test_non_screen_output_overrides_remain_allowed_by_guard(tmp_path: Path) -> None:
    for mode in ("preflight", "smoke"):
        pilot._guard_output_override(mode, tmp_path / mode)


def test_screen_run_does_not_turn_canonical_root_into_internal_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def stopped_preflight(*args, **kwargs):
        captured["output_root"] = kwargs.get("output_root")
        return {"failed_checks": ["intentional_test_stop"]}

    monkeypatch.setattr(pilot, "preflight_jepa_residual_pilot", stopped_preflight)
    monkeypatch.setattr(
        pilot,
        "guard_output_collision",
        lambda requested: (
            Path(requested).resolve(),
            Path(requested).resolve().with_name(Path(requested).name + ".partial"),
        ),
    )
    with pytest.raises(pilot.JEPAResidualPilotError, match="intentional_test_stop"):
        pilot.run_jepa_residual_pilot(
            Path("examples/conditional_background_residual_v1_1.example.json"),
            repository_root=Path.cwd(),
            mode="screen",
        )
    assert captured["output_root"] is None


def test_run_preflight_mode_forwards_allowed_output_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "preflight-only"
    captured: dict[str, object] = {}

    def fake_preflight(*args, **kwargs):
        captured["output_root"] = kwargs.get("output_root")
        return {"status": "sentinel"}

    monkeypatch.setattr(pilot, "preflight_jepa_residual_pilot", fake_preflight)
    result = pilot.run_jepa_residual_pilot(
        Path("examples/conditional_background_residual_v1_1.example.json"),
        repository_root=Path.cwd(),
        mode="preflight",
        output_root=override,
    )
    assert result == {"status": "sentinel"}
    assert captured["output_root"] == override


def test_checkpoint_hash_failure_happens_before_torch_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    calls = 0

    def forbidden_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("torch.load must not run after a hash failure")

    monkeypatch.setattr(pilot.torch, "load", forbidden_load)
    with pytest.raises(pilot.JEPAResidualPilotError, match="hash mismatch"):
        pilot._load_strict_checkpoint_payload(checkpoint, expected_sha256="0" * 64)
    assert calls == 0


def test_parent_integrity_and_byte_identity_are_complete(tmp_path: Path) -> None:
    artifact, index_sha = _parent(tmp_path / "parent")
    before = pilot.verify_parent_run_integrity(
        artifact.parent, expected_artifact_index_sha256=index_sha
    )
    after = pilot.assert_parent_unchanged(
        before, artifact.parent, expected_artifact_index_sha256=index_sha
    )
    assert before["artifact_count"] == 1
    assert after["parent_root_byte_identical_pre_post"] is True
    artifact.write_bytes(b"mutated")
    with pytest.raises(pilot.JEPAResidualPilotError, match="mismatch"):
        pilot.assert_parent_unchanged(
            before, artifact.parent, expected_artifact_index_sha256=index_sha
        )


def test_normalized_cache_is_reused_without_double_normalization(tmp_path: Path) -> None:
    values = np.linspace(-2, 3, 2 * 4 * 5 * 6, dtype=np.float32).reshape(2, 4, 5, 6)
    path = tmp_path / "clips.npy"
    np.save(path, values, allow_pickle=False)
    loaded, manifest = pilot.load_normalized_parent_cache(
        path,
        expected_sha256=_sha(path),
        expected_shape=values.shape,
        expected_dtype="float32",
    )
    np.testing.assert_array_equal(loaded, values)
    assert manifest["normalization_applied_by_this_runner"] is False
    raw = np.array([[[10.0, 12.0]]], dtype=np.float32)
    np.testing.assert_array_equal(
        pilot.normalize_raw_movie_once(raw, _normalization()),
        np.array([[[0.0, 1.0]]], dtype=np.float32),
    )


def test_v1_1_raw_hc_forwards_native_float32_units_without_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, np.ndarray] = {}

    class SpyComparator:
        def fit_source_off(self, source_off: np.ndarray):
            captured["fit"] = np.asarray(source_off).copy()

            def score(movie: np.ndarray) -> np.ndarray:
                captured.setdefault("apply_first", np.asarray(movie).copy())
                captured["apply_last"] = np.asarray(movie).copy()
                return np.zeros(np.asarray(movie).shape[1:], dtype=np.float64)

            return score

    monkeypatch.setattr(pilot, "frozen_handcrafted_comparator", SpyComparator())
    monkeypatch.setattr(
        pilot,
        "normalize_raw_movie_once",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("native raw-HC anchor must not normalize")
        ),
    )
    source_off = (
        np.arange(20 * 8 * 8, dtype=np.float32).reshape(20, 8, 8) + 489.0
    )
    source_on = source_off.copy()
    source_on[4:9, 3, 5] += np.float32(37.25)
    scorer = pilot.FrozenNativeHandcraftedComparator()
    frozen = scorer.fit_source_off(source_off)
    frozen(source_off)
    frozen(source_on)
    np.testing.assert_array_equal(captured["fit"], source_off)
    np.testing.assert_array_equal(captured["apply_first"], source_off)
    np.testing.assert_array_equal(captured["apply_last"], source_on)
    assert captured["fit"].dtype == np.float32
    assert scorer.fit_calls == 1
    assert scorer.apply_calls == ["source_off", "source_on"]


def test_fixture_manifest_identity_and_ulp_gate() -> None:
    expected = {
        "fixtures": [
            {
                "fixture_id": "f1",
                "maximum_pair_closure_float32_ulp": 0.5,
            }
        ]
    }
    result = pilot.assert_fixture_manifest_identity(dict(expected), expected)
    assert result["exact_parent_fixture_manifest_identity"] is True
    changed = json.loads(json.dumps(expected))
    changed["fixtures"][0]["fixture_id"] = "drift"
    with pytest.raises(pilot.JEPAResidualPilotError, match="differs"):
        pilot.assert_fixture_manifest_identity(changed, expected)


def test_frozen_state_hash_stays_fixed_when_only_decoder_is_trained() -> None:
    provider = torch.nn.Linear(3, 2)
    provider.requires_grad_(False)
    decoder = torch.nn.Linear(2, 1)
    before = pilot.module_state_sha256(provider)
    optimizer = torch.optim.SGD(decoder.parameters(), lr=0.1)
    inputs = torch.ones(4, 3)
    loss = decoder(provider(inputs)).square().mean()
    loss.backward()
    optimizer.step()
    assert pilot.module_state_sha256(provider) == before
    assert all(parameter.grad is None for parameter in provider.parameters())


def test_residual_handcrafted_calibration_receives_source_off_only() -> None:
    rng = np.random.default_rng(19)
    source_off = rng.normal(size=(20, 16, 16)).astype(np.float32)
    source_on = source_off.copy()
    source_on[5:10, 8, 8] += 2.0
    residual_off = source_off * 0.2
    residual_on = residual_off.copy()
    residual_on[5:10, 8, 8] += 1.5
    scorer = pilot.PrecomputedResidualHandcraftedScorer(
        source_off, source_on, residual_off, residual_on
    )
    frozen = scorer.fit_source_off(source_off)
    off_map = frozen(source_off)
    on_map = frozen(source_on)
    assert scorer.fit_calls == 1
    assert scorer.apply_calls == ["source_off", "source_on"]
    assert off_map.shape == on_map.shape == (16, 16)
    with pytest.raises(pilot.JEPAResidualPilotError, match="source-off"):
        scorer.fit_source_off(source_on)


def test_retention_absorption_algebra_closes() -> None:
    rng = np.random.default_rng(71)
    off = rng.normal(size=(8, 5, 5))
    signal = np.zeros_like(off)
    signal[2:6, 2, 2] = [0.5, 1.0, 0.8, 0.3]
    on = off + signal
    pred_off = 0.8 * off
    pred_on = pred_off + 0.25 * signal
    result = pilot.paired_signal_algebra(off, on, pred_off, pred_on, signal)
    assert result["retained_gain_projection"] == pytest.approx(0.75)
    assert result["predictor_absorption_projection"] == pytest.approx(0.25)
    assert result["retention_plus_absorption_minus_one"] == pytest.approx(0.0)
    assert result["pair_closure_max_abs"] < 1e-12
    assert result["total_signal_error_ratio"] == pytest.approx(0.25)
    assert result["orthogonal_distortion_ratio"] == pytest.approx(0.0, abs=1e-12)


def test_orthogonal_distortion_removes_aligned_gain_component() -> None:
    signal = np.zeros((2, 2, 2), dtype=np.float64)
    signal[0, 0, 0] = 2.0
    orthogonal = np.zeros_like(signal)
    orthogonal[0, 0, 1] = 1.0
    off = np.zeros_like(signal)
    on = signal.copy()
    residual_delta = 0.6 * signal + orthogonal
    predicted_off = np.zeros_like(signal)
    predicted_on = on - residual_delta
    result = pilot.paired_signal_algebra(
        off, on, predicted_off, predicted_on, signal
    )
    assert result["retained_gain_projection"] == pytest.approx(0.6)
    assert result["orthogonal_distortion_ratio"] == pytest.approx(0.5)
    assert result["total_signal_error_ratio"] == pytest.approx(
        np.sqrt(0.4**2 + 0.5**2)
    )


def test_output_and_partial_collisions_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(FileExistsError):
        pilot.guard_output_collision(output)
    output.rmdir()
    partial = tmp_path / "run.partial"
    partial.mkdir()
    with pytest.raises(FileExistsError):
        pilot.guard_output_collision(output)


def test_prediction_numpy_emits_exact_shape_finite_and_coverage_evidence() -> None:
    shape = pilot.STRICT_PREDICTION_SHAPE_BCTHW
    background = torch.zeros(shape, dtype=torch.float32)
    coverage = torch.ones(shape, dtype=torch.int16)
    values, evidence = pilot._prediction_numpy(
        SimpleNamespace(background=background, coverage_counts=coverage, manifest={})
    )
    assert values.shape == shape[2:]
    assert evidence["expected_background_shape_bcthw"] == list(shape)
    assert evidence["observed_background_shape_bcthw"] == list(shape)
    assert evidence["expected_coverage_shape_bcthw"] == list(shape)
    assert evidence["observed_coverage_shape_bcthw"] == list(shape)
    assert evidence["background_shape_exact"] is True
    assert evidence["coverage_shape_exact"] is True
    assert evidence["background_coverage_shapes_exact_match"] is True
    assert evidence["background_all_finite"] is True
    assert evidence["coverage_exactly_once"] is True
    assert evidence["prediction_contract_passed"] is True
    assert evidence["minimum_coverage"] == evidence["maximum_coverage"] == 1


def test_prediction_numpy_rejects_wrong_background_shape() -> None:
    background = torch.zeros((1, 1, 31, 64, 64), dtype=torch.float32)
    coverage = torch.ones(pilot.STRICT_PREDICTION_SHAPE_BCTHW, dtype=torch.int16)
    with pytest.raises(pilot.JEPAResidualPilotError, match="background sweep shape mismatch"):
        pilot._prediction_numpy(
            SimpleNamespace(background=background, coverage_counts=coverage, manifest={})
        )


def test_prediction_numpy_rejects_mismatched_coverage_shape() -> None:
    background = torch.zeros(pilot.STRICT_PREDICTION_SHAPE_BCTHW, dtype=torch.float32)
    coverage = torch.ones((1, 1, 32, 64, 63), dtype=torch.int16)
    with pytest.raises(pilot.JEPAResidualPilotError, match="coverage sweep shape mismatch"):
        pilot._prediction_numpy(
            SimpleNamespace(background=background, coverage_counts=coverage, manifest={})
        )


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_prediction_numpy_rejects_nonfinite_background(nonfinite: float) -> None:
    background = torch.zeros(pilot.STRICT_PREDICTION_SHAPE_BCTHW, dtype=torch.float32)
    background[0, 0, 0, 0, 0] = nonfinite
    coverage = torch.ones(pilot.STRICT_PREDICTION_SHAPE_BCTHW, dtype=torch.int16)
    with pytest.raises(pilot.JEPAResidualPilotError, match="nonfinite"):
        pilot._prediction_numpy(
            SimpleNamespace(background=background, coverage_counts=coverage, manifest={})
        )


@pytest.mark.parametrize("bad_count", [0, 2])
def test_prediction_numpy_rejects_nonexact_coverage(bad_count: int) -> None:
    background = torch.zeros(pilot.STRICT_PREDICTION_SHAPE_BCTHW, dtype=torch.float32)
    coverage = torch.ones(pilot.STRICT_PREDICTION_SHAPE_BCTHW, dtype=torch.int16)
    coverage[0, 0, 0, 0, 0] = bad_count
    with pytest.raises(pilot.JEPAResidualPilotError, match="exactly once"):
        pilot._prediction_numpy(
            SimpleNamespace(background=background, coverage_counts=coverage, manifest={})
        )


def test_incomplete_scientific_audit_is_explicit(tmp_path: Path) -> None:
    status = pilot._audit_placeholders(tmp_path, mode="smoke", fixture_count=1)
    persisted = json.loads((tmp_path / "scientific_audit_status.json").read_text())
    assert status == persisted
    assert persisted["scientific_audit_complete"] is False
    assert persisted["scientific_completion"] is False
    assert persisted["scientific_promotion_allowed"] is False
    assert persisted["promotion_blocked"] is True


def test_footprint_halo_leakage_uses_peak_tile_and_float64() -> None:
    rows, columns = np.mgrid[:64, :64]
    footprint = np.exp(-0.5 * (((rows - 24.0) / 2.0) ** 2 + ((columns - 24.0) / 2.0) ** 2))
    fixture = SimpleNamespace(fixture_id="f", footprints=footprint[None].astype(np.float32))
    result = pilot.footprint_halo_leakage_diagnostics([fixture], patch_yx=(8, 8))
    assert result["source_count"] == 1
    assert result["algorithm"].startswith("float64")
    assert result["mass_fraction_outside_halo"]["maximum"] >= 0.0
    assert result["energy_fraction_outside_halo"]["maximum"] >= 0.0


def test_grouped_summary_nests_source_counts_and_reports_recording_effects() -> None:
    rows = []
    methods = {
        pilot.RAW_METHOD: 0.25,
        pilot.JEPA_RESIDUAL_METHOD: 0.50,
        pilot.RANDOM_RESIDUAL_METHOD: 0.125,
    }
    for recording in ("r1", "r2"):
        for source_count in (1, 2, 4):
            fixture_id = f"{recording}_w_seed7_source{source_count}"
            for method, recall in methods.items():
                rows.append(
                    {
                        "fixture_id": fixture_id,
                        "method": method,
                        "background_recording_id": recording,
                        "background_window_id": f"{recording}_w",
                        "injection_seed": 7,
                        "source_count": source_count,
                        "source_on_recovery": {"recall": recall},
                    }
                )
    result = pilot._grouped_primary_summary(
        rows,
        evaluation={"grouped_bootstrap_draws": 1000, "grouped_bootstrap_seed": 17},
    )
    clusters = result["source_count_aggregated_cluster_rows"]
    assert len(clusters) == 2
    assert all(row["source_counts_aggregated"] == 3 for row in clusters)
    assert set(result["recording_level_effects"]) == {"r1", "r2"}
    assert result["paired_residual_minus_raw_hierarchical_bootstrap"][
        "observed_mean"
    ] == pytest.approx(0.25)
    assert result["paired_random_minus_raw_hierarchical_bootstrap"][
        "observed_mean"
    ] == pytest.approx(-0.125)
    assert result["paired_jepa_minus_random_hierarchical_bootstrap"][
        "observed_mean"
    ] == pytest.approx(0.375)
    assert result["recording_level_effects_by_comparison"][
        "jepa_residual_minus_random_residual"
    ] == {"r1": pytest.approx(0.375), "r2": pytest.approx(0.375)}
    assert result["recording_level_directions_by_comparison"][
        "random_residual_minus_raw"
    ] == {"r1": "negative", "r2": "negative"}


def test_descriptive_panel_handles_unavailable_smoke_intervals() -> None:
    rows = [
        {
            "fixture_id": "f1",
            "method": method,
            "background_recording_id": "r1",
            "background_window_id": "w1",
            "injection_seed": 3101,
            "source_count": 1,
            "source_on_recovery": {"recall": recall},
        }
        for method, recall in (
            (pilot.RAW_METHOD, 0.0),
            (pilot.JEPA_RESIDUAL_METHOD, 1.0),
            (pilot.RANDOM_RESIDUAL_METHOD, 0.5),
        )
    ]
    grouped = pilot._grouped_primary_summary(
        rows,
        evaluation={"grouped_bootstrap_draws": 1000, "grouped_bootstrap_seed": 17},
        expected_source_counts_per_cluster=1,
        expected_source_count_values=(1,),
    )
    diagnostics = [
        {
            "method": pilot.JEPA_RESIDUAL_METHOD,
            "algebra": {
                "retained_gain_projection": 1.0,
                "predictor_absorption_projection": 0.0,
                "retention_plus_absorption_minus_one": 0.0,
                "pair_closure_max_abs": 0.0,
            },
            "background": {"background_rms_ratio": 0.5},
        }
    ]
    panel = pilot._descriptive_advancement_panel(
        grouped,
        diagnostics,
        {
            "jepa_residual_minus_raw_macro_recall_min": 0.02,
            "grouped_bootstrap_lower_bound_min_exclusive": 0.0,
            "minimum_median_aligned_retained_gain": 0.9,
            "maximum_median_predictor_absorption": 0.1,
            "maximum_median_background_rms_ratio": 0.9,
            "jepa_residual_minus_random_observed_mean_min_exclusive": 0.0,
            "jepa_residual_minus_random_grouped_interval_lower_bound_preferred_exclusive": 0.0,
        },
    )
    assert panel["observed"]["grouped_bootstrap_lower_bound"] is None
    assert panel["threshold_crossings"][
        "bootstrap_lower_bound_strictly_positive"
    ] is False
    assert panel["specificity_control"][
        "jepa_residual_minus_random_interval_lower_bound_status"
    ] == "not_evaluated_requires_two_recordings"


def test_smoke_report_separates_108_anchor_fixtures_from_joined_subset() -> None:
    bootstrap = {"observed_mean": 0.0, "confidence_interval_95": None}
    summary = {
        "execution": {"mode": "smoke"},
        "decoder": {"trainable_parameters": 16385},
        "paired_injection": {
            "observed_fixture_count": 1,
            "registered_exact_fixture_count": 108,
            "exact_injected_source_occurrences": 1,
            "registered_exact_injected_source_occurrences": 252,
        },
        "primary_evaluation": {
            "method_macro_recall": {
                pilot.RAW_METHOD: 0.0,
                pilot.JEPA_RESIDUAL_METHOD: 0.0,
                pilot.RANDOM_RESIDUAL_METHOD: 0.0,
            },
            "paired_residual_minus_raw_hierarchical_bootstrap": bootstrap,
            "paired_random_minus_raw_hierarchical_bootstrap": bootstrap,
            "paired_jepa_minus_random_hierarchical_bootstrap": bootstrap,
            "recording_level_directions_by_comparison": {},
        },
        "raw_hc_cross_run_regression": {"execution_fixture_count": 108},
        "diagnostics": {
            "mean_total_signal_error_ratio": 0.0,
            "mean_orthogonal_distortion_ratio": 0.0,
            "engineering_algebra_integrity": {
                "maximum_projection_closure_absolute_error": 0.0,
                "maximum_projection_closure_absolute_error_allowed": 1e-5,
                "maximum_voxelwise_pair_closure_absolute_error": 0.0,
                "maximum_voxelwise_pair_closure_absolute_error_allowed": 1e-5,
            },
        },
        "proposal_counts": {
            "proposal_cap_per_fixture_arm": 20,
            "by_method": {
                pilot.RAW_METHOD: {
                    "source_on_candidate_count_total": 1,
                    "intervention_candidate_count_total": 1,
                    "fixture_count": 1,
                }
            },
        },
        "hashes": {"implementation_sources": {}},
        "seeds": {},
        "scientific_audit": {"scientific_audit_complete": False},
        "motion_dependency": {"satisfied": False},
    }
    report = pilot._report(summary)
    assert "parent-matched handcrafted comparator evaluated in native raw units" in report
    assert "all `108` registered anchor fixtures" in report
    assert "joined residual evaluation used `1` fixtures" in report


def test_grouped_summary_rejects_duplicate_source_count_condition() -> None:
    rows = []
    for source_count in (1, 2, 4):
        reported_source_count = 2 if source_count == 4 else source_count
        for method in (
            pilot.RAW_METHOD,
            pilot.JEPA_RESIDUAL_METHOD,
            pilot.RANDOM_RESIDUAL_METHOD,
        ):
            rows.append(
                {
                    "fixture_id": f"f{source_count}",
                    "method": method,
                    "background_recording_id": "r1",
                    "background_window_id": "w1",
                    "injection_seed": 3101,
                    "source_count": reported_source_count,
                    "source_on_recovery": {"recall": 0.5},
                }
            )
    with pytest.raises(pilot.JEPAResidualPilotError, match="exactly nested"):
        pilot._grouped_primary_summary(
            rows,
            evaluation={"grouped_bootstrap_draws": 1000, "grouped_bootstrap_seed": 17},
        )


def test_smoke_fixture_summary_reports_only_executed_subset() -> None:
    fixture = SimpleNamespace(
        metadata={
            "background_recording_id": "r1",
            "background_window_id": "r1_w1",
            "source_count": 1,
            "injection_seed": 3101,
        },
        footprints=np.zeros((1, 8, 8), dtype=np.float32),
    )
    registered = {
        "observed_fixture_count": 108,
        "planned_fixture_count": 108,
        "coverage_fraction": 1.0,
        "full_registered_grid": True,
        "background_recording_ids": ["r1", "r2", "r3", "r4"],
        "source_counts": [1, 2, 4],
        "injection_seeds": [3101, 3102, 3103],
    }
    summary = pilot._execution_fixture_summary([fixture], registered)
    assert summary["observed_fixture_count"] == 1
    assert summary["planned_registered_fixture_count"] == 108
    assert summary["execution_coverage_fraction_of_registered_grid"] == pytest.approx(
        1 / 108
    )
    assert summary["background_recording_ids"] == ["r1"]
    assert summary["background_window_ids"] == ["r1_w1"]
    assert summary["source_counts"] == [1]
    assert summary["injection_seeds"] == [3101]
    assert summary["exact_injected_source_occurrences"] == 1
    assert summary["registered_grid"]["background_recording_ids"] == [
        "r1",
        "r2",
        "r3",
        "r4",
    ]
    assert summary["registered_exact_fixture_count"] == 108
    assert summary["registered_exact_injected_source_occurrences"] == 252


def test_all_64_target_tiles_include_every_edge_class() -> None:
    targets = pilot._production_invariance_targets((8, 8))
    assert len(targets) == 64
    assert len(set(targets)) == 64
    for target in ((0, 0), (0, 4), (7, 3), (3, 0), (3, 7), (1, 1), (3, 3)):
        assert target in targets


def test_geometry_failure_prevents_first_optimizer_step() -> None:
    calls = 0

    def train(*args, **kwargs):
        nonlocal calls
        calls += 1
        return object()

    decoder_module = SimpleNamespace(train_pixel_decoder=train)
    with pytest.raises(pilot.JEPAResidualPilotError, match="blocked"):
        pilot._guarded_train_decoder_arms(
            decoder_module,
            model=torch.nn.Linear(1, 1),
            random_model=torch.nn.Linear(1, 1),
            training_tensor=torch.zeros(1, 1, 1, 1, 1),
            training_contract=object(),
            seed=6202,
            device="cpu",
            geometry_validation={"passed_before_first_optimizer_step": False},
        )
    assert calls == 0


def test_native_raw_anchor_failure_prevents_first_optimizer_step() -> None:
    optimizer_calls = 0
    construction_calls = 0

    def train(*args, **kwargs):
        nonlocal optimizer_calls
        optimizer_calls += 1
        return object()

    def construct(*args, **kwargs):
        nonlocal construction_calls
        construction_calls += 1
        return torch.nn.Linear(1, 1)

    decoder_module = SimpleNamespace(
        FrozenLatentPixelBackgroundModel=construct,
        train_pixel_decoder=train,
    )
    failed_anchor = {
        "exact_recovery_result_and_recall_identity": False,
        "recovery_result_object_count": 216,
        "anchor_execution_phase": "after_fixture_reconstruction_before_decoder_construction_or_training",
    }
    with pytest.raises(pilot.JEPAResidualPilotError, match="raw-HC"):
        pilot._construct_decoder_models(
            decoder_module,
            provider=SimpleNamespace(provider=object()),
            random_provider=SimpleNamespace(provider=object()),
            decoder_seed=6201,
            device="cpu",
            raw_anchor_validation=failed_anchor,
        )
    with pytest.raises(pilot.JEPAResidualPilotError, match="raw-HC"):
        pilot._guarded_train_decoder_arms(
            decoder_module,
            model=torch.nn.Linear(1, 1),
            random_model=torch.nn.Linear(1, 1),
            training_tensor=torch.zeros(1, 1, 1, 1, 1),
            training_contract=object(),
            seed=6202,
            device="cpu",
            geometry_validation={"passed_before_first_optimizer_step": True},
            raw_anchor_validation=failed_anchor,
        )
    assert construction_calls == 0
    assert optimizer_calls == 0


def test_decoder_distortion_reference_must_agree_with_runner() -> None:
    algebra = {
        "total_signal_error_ratio": 0.42,
        "orthogonal_distortion_ratio": 0.17,
    }
    reference = {
        "signal": {
            "total_signal_error_ratio": 0.4200001,
            "orthogonal_distortion_ratio": 0.1700001,
        }
    }
    result = pilot._decoder_diagnostic_agreement(algebra, reference)
    assert result["all_match"] is True
    reference["signal"]["orthogonal_distortion_ratio"] = 0.20
    with pytest.raises(pilot.JEPAResidualPilotError, match="disagree"):
        pilot._decoder_diagnostic_agreement(algebra, reference)


def test_projection_and_voxelwise_closure_are_separate_integrity_checks() -> None:
    diagnostics = [
        {
            "method": pilot.JEPA_RESIDUAL_METHOD,
            "algebra": {
                "retention_plus_absorption_minus_one": -2e-6,
                "pair_closure_max_abs": 3e-6,
            },
        },
        {
            "method": pilot.RANDOM_RESIDUAL_METHOD,
            "algebra": {
                "retention_plus_absorption_minus_one": 4e-6,
                "pair_closure_max_abs": 5e-6,
            },
        },
    ]
    result = pilot._algebra_integrity_summary(
        diagnostics,
        {
            "maximum_projection_closure_absolute_error": 1e-5,
            "maximum_voxelwise_pair_closure_absolute_error": 1e-5,
        },
    )
    assert result["maximum_projection_closure_absolute_error"] == pytest.approx(4e-6)
    assert result["maximum_voxelwise_pair_closure_absolute_error"] == pytest.approx(
        5e-6
    )
    assert result["projection_closure_passed"] is True
    assert result["voxelwise_pair_closure_passed"] is True
    projection_failure = json.loads(json.dumps(diagnostics))
    projection_failure[0]["algebra"][
        "retention_plus_absorption_minus_one"
    ] = 2e-5
    projection_result = pilot._algebra_integrity_summary(
        projection_failure,
        {
            "maximum_projection_closure_absolute_error": 1e-5,
            "maximum_voxelwise_pair_closure_absolute_error": 1e-5,
        },
    )
    assert projection_result["projection_closure_passed"] is False
    assert projection_result["voxelwise_pair_closure_passed"] is True
    voxel_failure = json.loads(json.dumps(diagnostics))
    voxel_failure[1]["algebra"]["pair_closure_max_abs"] = 2e-5
    voxel_result = pilot._algebra_integrity_summary(
        voxel_failure,
        {
            "maximum_projection_closure_absolute_error": 1e-5,
            "maximum_voxelwise_pair_closure_absolute_error": 1e-5,
        },
    )
    assert voxel_result["projection_closure_passed"] is True
    assert voxel_result["voxelwise_pair_closure_passed"] is False


def _raw_anchor_parent_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(108):
        recovery = {
            "candidate_count": 1,
            "injected_sources": 1,
            "localization_errors_px": [0.0],
            "matched_candidate_indices": [0],
            "matched_source_indices": [0],
            "recall": 1.0,
            "recovered_sources": 1,
            "unmatched_candidate_count_unknown": 0,
        }
        rows.append(
            {
                "fixture_id": f"f{index:03d}",
                "method": pilot.PARENT_RAW_METHOD,
                "training_seed": 1001,
                "source_on_recovery": dict(recovery),
                "intervention_recovery": dict(recovery),
            }
        )
    return rows


def test_raw_hc_cross_run_regression_is_exact_and_fail_closed() -> None:
    parent_rows = _raw_anchor_parent_rows()
    current = {
        **parent_rows[0],
        "method": pilot.RAW_METHOD,
    }
    result = pilot.raw_hc_cross_run_regression(
        {"rows": parent_rows},
        [current],
        executed_fixture_ids=["f000"],
        parent_run_id="parent-run",
        current_run_id="current-run",
    )
    assert result["parent_registered_fixture_count"] == 108
    assert result["execution_fixture_count"] == 1
    assert result["exact_recovery_result_and_recall_identity"] is True
    drifted = json.loads(json.dumps(current))
    drifted["source_on_recovery"]["recall"] = 0.0
    mismatch = pilot.raw_hc_cross_run_regression(
        {"rows": parent_rows},
        [drifted],
        executed_fixture_ids=["f000"],
        parent_run_id="parent-run",
        current_run_id="current-run",
    )
    assert mismatch["exact_recovery_result_and_recall_identity"] is False
    assert mismatch["mismatches"] == [
        {
            "fixture_id": "f000",
            "field": "source_on_recovery",
            "parent_sha256": mismatch["mismatches"][0]["parent_sha256"],
            "current_sha256": mismatch["mismatches"][0]["current_sha256"],
        }
    ]


def test_v1_1_all_108_raw_hc_source_on_and_intervention_results_are_exact() -> None:
    parent_rows = _raw_anchor_parent_rows()
    current_rows = [
        {**json.loads(json.dumps(row)), "method": pilot.RAW_METHOD}
        for row in parent_rows
    ]
    fixture_ids = [str(row["fixture_id"]) for row in parent_rows]
    result = pilot.raw_hc_cross_run_regression(
        {"rows": parent_rows},
        current_rows,
        executed_fixture_ids=fixture_ids,
        parent_run_id="NREV-RUN-EXP-0028-SCREEN-20260829-B",
        current_run_id="NREV-RUN-EXP-0029-SCREEN-20260830-B",
    )
    assert result["parent_registered_fixture_count"] == 108
    assert result["execution_fixture_count"] == 108
    assert result["matched_fixture_count"] == 108
    assert result["compared_fields"] == [
        "source_on_recovery",
        "intervention_recovery",
    ]
    assert result["numeric_tolerance"] == 0.0
    assert result["mismatch_count"] == 0
    assert result["exact_recovery_result_and_recall_identity"] is True

    drifted = json.loads(json.dumps(current_rows))
    drifted[-1]["intervention_recovery"]["candidate_count"] = 2
    failed = pilot.raw_hc_cross_run_regression(
        {"rows": parent_rows},
        drifted,
        executed_fixture_ids=fixture_ids,
        parent_run_id="NREV-RUN-EXP-0028-SCREEN-20260829-B",
        current_run_id="NREV-RUN-EXP-0029-SCREEN-20260830-B",
    )
    assert failed["exact_recovery_result_and_recall_identity"] is False
    assert failed["mismatch_count"] == 1
    assert failed["mismatches"][0]["field"] == "intervention_recovery"


def test_required_outputs_must_exist_and_match_artifact_index(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "summary.json").write_text("{}", encoding="utf-8")
    (root / "status.json").write_text("{}", encoding="utf-8")
    (root / "artifact_index.json").write_text(
        json.dumps(pilot._artifact_index(root)), encoding="utf-8"
    )
    result = pilot._validate_required_outputs(
        root, ["summary.json", "status.json", "artifact_index.json"]
    )
    assert result["all_required_outputs_exist"] is True
    (root / "summary.json").write_text('{"drift":true}', encoding="utf-8")
    with pytest.raises(pilot.JEPAResidualPilotError, match="complete live output tree"):
        pilot._validate_required_outputs(
            root, ["summary.json", "status.json", "artifact_index.json"]
        )
    with pytest.raises(pilot.JEPAResidualPilotError, match="incomplete"):
        pilot._validate_required_outputs(root, ["missing.json"])
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / "required.json").write_text("{}", encoding="utf-8")
    (fresh / "artifact_index.json").write_text(
        json.dumps(pilot._artifact_index(fresh)), encoding="utf-8"
    )
    (fresh / "unindexed.json").write_text("{}", encoding="utf-8")
    with pytest.raises(pilot.JEPAResidualPilotError, match="complete live output tree"):
        pilot._validate_required_outputs(
            fresh, ["required.json", "artifact_index.json"]
        )
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "required.json").write_text("{}", encoding="utf-8")
    stale_extra = stale / "indexed-then-removed.json"
    stale_extra.write_text("{}", encoding="utf-8")
    (stale / "artifact_index.json").write_text(
        json.dumps(pilot._artifact_index(stale)), encoding="utf-8"
    )
    stale_extra.unlink()
    with pytest.raises(pilot.JEPAResidualPilotError, match="complete live output tree"):
        pilot._validate_required_outputs(
            stale, ["required.json", "artifact_index.json"]
        )


def test_decoder_checkpoint_identity_covers_both_providers_and_hash_scopes() -> None:
    values = {
        "jepa_checkpoint_tensor_sha256": "1" * 64,
        "jepa_runtime_provider_sha256": "2" * 64,
        "random_checkpoint_tensor_sha256": "3" * 64,
        "random_runtime_provider_sha256": "4" * 64,
        "jepa_decoder_state_sha256": "5" * 64,
        "random_decoder_state_sha256": "6" * 64,
    }
    identity = pilot._decoder_checkpoint_identity(**values)
    assert set(identity["hash_contract"]) == {
        "serialized_checkpoint_file",
        "parent_checkpoint_tensor_mapping",
        "runtime_module_state",
    }
    assert all(
        contract["algorithm"] == "sha256" and contract["scope"]
        for contract in identity["hash_contract"].values()
    )
    providers = identity["frozen_provider_identity"]
    assert providers == {
        "jepa_checkpoint_tensor_mapping_sha256": "1" * 64,
        "jepa_runtime_provider_module_state_sha256": "2" * 64,
        "random_checkpoint_tensor_mapping_sha256": "3" * 64,
        "random_runtime_provider_module_state_sha256": "4" * 64,
    }
    assert identity["decoder_state_identity"] == {
        "jepa_arm_runtime_module_state_sha256": "5" * 64,
        "random_arm_runtime_module_state_sha256": "6" * 64,
    }


def _patch_preflight_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        pilot,
        "_pinned_parent_paths",
        lambda config, repository: (tmp_path / "parent", {}),
    )
    monkeypatch.setattr(
        pilot,
        "verify_parent_run_integrity",
        lambda *args, **kwargs: {
            "all_parent_artifacts_verified": True,
            "artifact_index_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        pilot,
        "_verify_pinned_parent_files",
        lambda *args, **kwargs: {"all_pinned_parent_files_match": True},
    )
    monkeypatch.setattr(
        pilot,
        "_validate_implementation_hashes",
        lambda *args, **kwargs: {"all_match": True, "sources": []},
    )
    monkeypatch.setattr(
        pilot,
        "_resource_snapshot",
        lambda *args, **kwargs: {
            "disk_free_mib": 100_000,
            "available_ram_mib": 100_000,
            "cuda": {"available": False},
        },
    )
    monkeypatch.setattr(
        pilot,
        "_execution_provenance",
        lambda *args, **kwargs: {
            "git": {"status": "captured"},
            "runtime": {"python_version": "test", "torch_version": "test"},
        },
    )


@pytest.mark.parametrize(
    "field",
    [
        "execution_authorized",
        "scientific_execution_authorized",
        "claim_bearing_execution_authorized",
        "scientific_completion_allowed",
        "claim_bearing_execution",
    ],
)
def test_preflight_rejects_every_scientific_or_claim_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    _patch_preflight_dependencies(monkeypatch, tmp_path)
    config = json.loads(
        Path("examples/conditional_background_residual_v1_1.example.json").read_text()
    )
    config[field] = True
    config_path = tmp_path / f"{field}.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = pilot.preflight_jepa_residual_pilot(
        config_path,
        repository_root=Path.cwd(),
        mode="smoke",
        output_root=tmp_path / f"out-{field}",
    )
    assert result["status"] == "failed"
    assert "all_scientific_and_claim_bearing_authorizations_false" in result[
        "failed_checks"
    ]


def test_preflight_rejects_missing_engineering_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_preflight_dependencies(monkeypatch, tmp_path)
    config = json.loads(
        Path("examples/conditional_background_residual_v1_1.example.json").read_text()
    )
    config["engineering_screen_execution_authorized"] = False
    config_path = tmp_path / "unauthorized.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = pilot.preflight_jepa_residual_pilot(
        config_path,
        repository_root=Path.cwd(),
        mode="smoke",
        output_root=tmp_path / "unauthorized-out",
    )
    assert result["status"] == "failed"
    assert "engineering_screen_execution_authorized" in result["failed_checks"]
