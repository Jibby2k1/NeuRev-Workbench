from __future__ import annotations

import copy
from pathlib import Path
import subprocess

import numpy as np
import pytest

from neurobench.experiments.neuron_identifiability import motion_registration_confound_audit as audit
from neurobench.experiments.neuron_identifiability.contracts import atomic_json, stable_hash
from neurobench.experiments.neuron_identifiability.discovery import sha256_file
from neurobench.experiments.neuron_identifiability.motion_registration_confound_audit import (
    MotionRegistrationConfoundConfig,
    _aggregate_parent_endpoints,
    adjust_grouped_association_family,
    analyze_window,
    compute_pair_diagnostics,
    grouped_exact_association,
    validate_artifact_index,
    validate_frozen_window_manifest,
    validate_output_provenance_bundle,
    validate_run_provenance,
    validate_status_against_provenance,
)


def _config() -> MotionRegistrationConfoundConfig:
    return MotionRegistrationConfoundConfig(
        minimum_registration_extent=12,
        minimum_peak_to_median_ratio=2.0,
        maximum_translation_search_px=8.0,
    )


def _texture(seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.indices((64, 64))
    values = 1500.0 + 180.0 * np.sin(x / 4.1) + 130.0 * np.cos(y / 5.3)
    values += rng.normal(0.0, 40.0, size=values.shape)
    return np.clip(np.rint(values), 1, 65_534).astype(np.uint16)


def _manifest() -> dict[str, object]:
    recordings = (
        ("060126_10_rest", "060126", "data://Inputs/060126/10 rest.tif"),
        ("060126_12_left", "060126", "data://Inputs/060126/12 left.tif"),
        ("060126_15_right", "060126", "data://Inputs/060126/15 right.tif"),
        (
            "spon_ca_burst_3_hindbrain_to_tail_488_20ms",
            "spon_ca_burst",
            "data://Inputs/Spon Ca Burst/3 hindbrain to tail 488 20ms.tif",
        ),
    )
    windows = []
    for recording_id, dataset_id, uri in recordings:
        for index, stratum in enumerate(("low_mad", "median_mad", "high_mad"), start=1):
            start = 40 * (index - 1)
            windows.append(
                {
                    "background_dataset_id": dataset_id,
                    "background_mad_stratum": stratum,
                    "background_recording_id": recording_id,
                    "background_window_id": f"{recording_id}__window_{index}_{stratum}",
                    "height": 64,
                    "quiet_policy": "label_free",
                    "recording_id": recording_id,
                    "source_uri": uri,
                    "start_frame_zero": start,
                    "stop_frame_zero_exclusive": start + 32,
                    "temporal_mad": float(index),
                    "width": 64,
                    "x_zero": 0,
                    "y_zero": 0,
                }
            )
    return {"window_count": len(windows), "windows": windows}


def _run_provenance() -> dict[str, object]:
    config = {"expected_window_count": 12, "scientific_completion_allowed": False}
    records = [
        {
            "id": "input_a",
            "role": "frozen_input",
            "uri": "repo://research/input.json",
            "sha256": "a" * 64,
            "bytes": 7,
        },
        {
            "id": "registry_run_schema",
            "role": "registry_schema",
            "uri": "repo://research/schemas/run.schema.json",
            "sha256": "6" * 64,
            "bytes": 11,
        },
    ]
    return {
        "schema_version": 1,
        "record_type": "run_provenance",
        "experiment_id": "NREV-EXP-0025",
        "run_id": "NREV-RUN-EXP-0025-SCREEN-TEST-C",
        "lifecycle": "succeeded",
        "planned_at": "2026-08-30T12:00:00.000000Z",
        "started_at": "2026-08-30T12:00:01.000000Z",
        "ended_at": "2026-08-30T12:00:03.500000Z",
        "duration_seconds": 2.5,
        "execution": {
            "mode": audit.RUN_MODE,
            "command": [".venv-neurobench/bin/python", "-m", audit.RUNNER_MODULE],
        },
        "code": {
            "repository": "https://github.com/example/neurev",
            "commit": "1" * 40,
            "branch": "codex/test",
            "dirty": True,
            "diff_sha256": "5" * 64,
            "runner_module": audit.RUNNER_MODULE,
            "runner_sha256": sha256_file(Path(audit.__file__)),
            "registry_schema": {
                "uri": "repo://research/schemas/run.schema.json",
                "sha256": "6" * 64,
                "checks": {
                    "run_schema_code_subschema_valid": True,
                    "dirty_diff_conditional_satisfied": True,
                    "additional_properties_absent": True,
                },
            },
            "git": {
                "repository": "https://github.com/example/neurev",
                "commit": "1" * 40,
                "branch": "codex/test",
                "dirty": True,
                "diff_sha256": "5" * 64,
                "dirty_entry_count": 1,
                "dirty_status_sha256": "2" * 64,
                "diff_algorithm": "NEUREV-DIRTY-STATE-V1",
            },
        },
        "configuration": {
            "config": config,
            "config_sha256": stable_hash(config),
            "resolved_config_uri": "run://resolved_config.json",
            "resolved_config_sha256": "3" * 64,
        },
        "inputs": {"records": records, "input_set_sha256": stable_hash(records)},
        "environment": {
            "runtime": {
                "python_version": "3.12.0",
                "python_implementation": "CPython",
                "platform": "test-platform",
                "numpy_version": "2.0.0",
                "scipy_version": "1.0.0",
                "tifffile_version": "2026.1.1",
                "pyyaml_version": "6.0.0",
                "jsonschema_version": "4.26.0",
            }
        },
        "scientific_boundary": {
            "scientific_completion": False,
            "scientific_promotion_allowed": False,
            "claim_bearing": False,
        },
    }


def _status(provenance: dict[str, object], provenance_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": provenance["experiment_id"],
        "run_id": provenance["run_id"],
        "lifecycle": "succeeded",
        "status": "succeeded_engineering_screen",
        "planned_at": provenance["planned_at"],
        "started_at": provenance["started_at"],
        "ended_at": provenance["ended_at"],
        "duration_seconds": provenance["duration_seconds"],
        "execution_mode": provenance["execution"]["mode"],
        "command": provenance["execution"]["command"],
        "hashes": {
            "runner_sha256": provenance["code"]["runner_sha256"],
            "config_sha256": provenance["configuration"]["config_sha256"],
            "resolved_config_sha256": provenance["configuration"]["resolved_config_sha256"],
            "input_set_sha256": provenance["inputs"]["input_set_sha256"],
            "diff_sha256": provenance["code"]["diff_sha256"],
        },
        "git": provenance["code"]["git"],
        "runtime": provenance["environment"]["runtime"],
        "provenance": {"path": "run_provenance.json", "sha256": provenance_sha256},
        "scientific_completion": False,
        "scientific_promotion_allowed": False,
    }


def test_bidirectional_translation_closes_and_reduces_registered_difference() -> None:
    reference = _texture()
    moving = np.roll(np.roll(reference, 2, axis=0), -3, axis=1)

    result = compute_pair_diagnostics(reference, moving, config=_config())

    assert result["bidirectional_valid"] is True
    assert result["forward_shift_y_px"] == pytest.approx(-2.0, abs=0.35)
    assert result["forward_shift_x_px"] == pytest.approx(3.0, abs=0.35)
    assert result["forward_backward_closure_px"] < 0.1
    assert result["registered_over_raw_difference_mad_ratio"] < 0.1
    bounds = result["matched_support_bounds_yx_zero_half_open"]
    interior = (slice(bounds[0], bounds[1]), slice(bounds[2], bounds[3]))
    raw_difference = moving.astype(np.float64)[interior] - reference.astype(np.float64)[interior]
    manual_matched_raw_mad = 1.4826 * np.median(np.abs(raw_difference - np.median(raw_difference)))
    assert result["raw_difference_mad_matched_support"] == pytest.approx(manual_matched_raw_mad)
    assert result["raw_difference_mad"] != pytest.approx(result["raw_difference_mad_matched_support"])
    assert result["registered_over_raw_difference_mad_ratio"] == pytest.approx(
        result["registered_residual_mad"] / result["raw_difference_mad_matched_support"]
    )
    assert result["registered_raw_mad_support_contract"] == "exact_same_shift_valid_interior"
    assert result["registered_over_raw_difference_mad_ratio_denominator"] == (
        "raw_difference_mad_matched_support"
    )
    assert audit._pair_matched_support_contract_valid(result, config=_config())
    mismatched = copy.deepcopy(result)
    mismatched["raw_difference_mad_matched_support"] *= 2.0
    assert not audit._pair_matched_support_contract_valid(mismatched, config=_config())
    assert result["tile_count"] == 4


def test_window_audit_reproduces_temporal_mad_and_keeps_analog_rail_unresolved() -> None:
    base = _texture()
    video = np.repeat(base[None, :, :], 32, axis=0)
    video[:, 0, 0] = np.iinfo(np.uint16).max
    sampled = video[::2, ::4, ::4].astype(np.float32)
    sampled_center = np.median(sampled, axis=0, keepdims=True)
    temporal_mad = float(1.4826 * np.median(np.abs(sampled - sampled_center)))
    window = _manifest()["windows"][0]
    window = dict(window)
    window["start_frame_zero"] = 0
    window["stop_frame_zero_exclusive"] = 32
    window["temporal_mad"] = temporal_mad

    summary, pairs = analyze_window(window, video, config=_config())

    assert summary["temporal_mad_absolute_error"] == pytest.approx(0.0)
    assert summary["stored_high_code_fraction"] == pytest.approx(1.0 / (64 * 64))
    assert summary["analog_sensor_rail_status"] == "unresolved"
    assert "ADC" in summary["analog_sensor_rail_reason"]
    assert len(pairs) == 31


def test_frozen_manifest_requires_four_groups_three_strata_and_no_overlap() -> None:
    manifest = _manifest()
    rows = validate_frozen_window_manifest(manifest, config=_config())
    assert len(rows) == 12

    missing = copy.deepcopy(manifest)
    missing["windows"] = missing["windows"][:-1]
    missing["window_count"] = 11
    with pytest.raises(ValueError, match="exactly twelve"):
        validate_frozen_window_manifest(missing, config=_config())

    overlap = copy.deepcopy(manifest)
    overlap["windows"][1]["start_frame_zero"] = 10
    overlap["windows"][1]["stop_frame_zero_exclusive"] = 42
    with pytest.raises(ValueError, match="overlap"):
        validate_frozen_window_manifest(overlap, config=_config())


def test_grouped_exact_association_enumerates_all_1296_within_recording_orders() -> None:
    rows = []
    for recording in range(4):
        for rank in range(3):
            rows.append({"recording_id": f"rec_{recording}", "motion": float(rank), "endpoint": float(rank)})

    result = grouped_exact_association(rows, predictor="motion", outcome="endpoint")

    assert result["n_windows"] == 12
    assert result["n_recordings"] == 4
    assert result["within_recording_rank_correlation"] == pytest.approx(1.0)
    assert result["grouped_permutation_count"] == 6**4
    assert result["grouped_exact_two_sided_p"] == pytest.approx(2 / 1296)
    assert result["loro_sign_consistent"] is True

    ratio_result = grouped_exact_association(
        [
            {
                **row,
                "registered_over_raw_difference_mad_ratio": row["motion"],
                "jepa_background_rms_ratio": row["endpoint"],
            }
            for row in rows
        ],
        predictor="registered_over_raw_difference_mad_ratio",
        outcome="jepa_background_rms_ratio",
    )
    assert ratio_result["difference_mad_support_contract"] == (
        "ratio_uses_exact_same_shift_valid_interior_per_pair"
    )


def test_association_family_gets_monotone_bh_adjustment() -> None:
    rows = [
        {"grouped_exact_two_sided_p": 0.01},
        {"grouped_exact_two_sided_p": 0.03},
        {"grouped_exact_two_sided_p": 0.8},
    ]

    adjusted = adjust_grouped_association_family(rows)

    assert [row["grouped_exact_bh_q"] for row in adjusted] == pytest.approx([0.03, 0.045, 0.8])
    assert [row["passes_bh_0_05"] for row in adjusted] == [True, True, False]
    assert all(row["association_family_size"] == 3 for row in adjusted)


def test_parent_endpoint_join_preserves_exact_fixture_and_method_grids() -> None:
    window_ids = {f"window_{index:02d}" for index in range(12)}
    raw_rows = []
    residual_rows = []
    for window_id in sorted(window_ids):
        for source_count in (1, 2, 4):
            for seed in (3101, 3102, 3103):
                fixture_id = f"{window_id}__sources_{source_count}__seed_{seed}"
                raw_rows.append(
                    {
                        "fixture_id": fixture_id,
                        "background_window_id": window_id,
                        "method": "raw_frozen_handcrafted_stack",
                        "source_count": str(source_count),
                        "source_on_recovered": str(source_count),
                        "injected_sources": str(source_count),
                        "source_on_recall": "1.0",
                        "intervention_recall": "1.0",
                        "median_center_delta_background_mad": "2.5",
                    }
                )
                for method in (
                    "jepa_conditional_pixel_residual_frozen_handcrafted_stack",
                    "random_encoder_conditional_pixel_residual_frozen_handcrafted_stack",
                ):
                    residual_rows.append(
                        {
                            "fixture_id": fixture_id,
                            "background_window_id": window_id,
                            "method": method,
                            "background_rms_ratio": "1.5",
                            "dynamic_mad_ratio": "1.1",
                            "seam_to_interior_jump_ratio": "2.0",
                        }
                    )

    result = _aggregate_parent_endpoints(raw_rows, residual_rows, window_ids=window_ids)

    assert set(result) == window_ids
    assert result["window_00"]["raw_hc_source_on_micro_recall"] == 1.0
    assert result["window_00"]["raw_hc_source_on_seed_sd_mean"] == 0.0
    assert result["window_00"]["jepa_background_rms_ratio"] == 1.5
    assert result["window_00"]["random_seam_to_interior_jump_ratio"] == 2.0


def test_run_provenance_requires_complete_utc_lifecycle_and_exact_hashes() -> None:
    provenance = _run_provenance()

    checks = validate_run_provenance(provenance)

    assert checks["timestamps_complete_and_ordered"] is True
    assert checks["duration_matches_timestamps"] is True
    assert checks["runner_hash_matches_live_module"] is True
    assert checks["config_hash_matches_exact_config"] is True
    assert checks["registry_dirty_diff_schema_contract"] is True

    missing_end = copy.deepcopy(provenance)
    del missing_end["ended_at"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_run_provenance(missing_end)

    non_utc = copy.deepcopy(provenance)
    non_utc["started_at"] = "2026-08-30T12:00:01"
    with pytest.raises(ValueError, match="ending in Z"):
        validate_run_provenance(non_utc)

    wrong_duration = copy.deepcopy(provenance)
    wrong_duration["duration_seconds"] = 3.5
    with pytest.raises(ValueError, match="exactly match"):
        validate_run_provenance(wrong_duration)

    wrong_runner = copy.deepcopy(provenance)
    wrong_runner["code"]["runner_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="live runner module"):
        validate_run_provenance(wrong_runner)

    missing_dirty_diff = copy.deepcopy(provenance)
    del missing_dirty_diff["code"]["diff_sha256"]
    with pytest.raises(ValueError, match="code.diff_sha256"):
        validate_run_provenance(missing_dirty_diff)

    inconsistent_dirty_diff = copy.deepcopy(provenance)
    inconsistent_dirty_diff["code"]["git"]["diff_sha256"] = "6" * 64
    with pytest.raises(ValueError, match="disagrees with the canonical"):
        validate_run_provenance(inconsistent_dirty_diff)

    wrong_config = copy.deepcopy(provenance)
    wrong_config["configuration"]["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="persisted scientific config"):
        validate_run_provenance(wrong_config)

    wrong_input_set = copy.deepcopy(provenance)
    wrong_input_set["inputs"]["input_set_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact ordered input records"):
        validate_run_provenance(wrong_input_set)

    leaked_command = copy.deepcopy(provenance)
    leaked_command["execution"]["command"].append("/" + "home" + "/example/private-data")
    with pytest.raises(ValueError, match="workstation paths"):
        validate_run_provenance(leaked_command)


def test_status_fails_closed_on_any_provenance_hash_or_timestamp_disagreement() -> None:
    provenance = _run_provenance()
    provenance_sha256 = "4" * 64
    status = _status(provenance, provenance_sha256)

    checks = validate_status_against_provenance(
        status,
        provenance,
        provenance_sha256=provenance_sha256,
    )

    assert checks["timestamps_and_duration_match"] is True
    assert checks["command_mode_and_hashes_match"] is True

    wrong_runner = copy.deepcopy(status)
    wrong_runner["hashes"]["runner_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash anchors"):
        validate_status_against_provenance(
            wrong_runner,
            provenance,
            provenance_sha256=provenance_sha256,
        )

    wrong_diff = copy.deepcopy(status)
    wrong_diff["hashes"]["diff_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash anchors"):
        validate_status_against_provenance(
            wrong_diff,
            provenance,
            provenance_sha256=provenance_sha256,
        )

    wrong_end = copy.deepcopy(status)
    wrong_end["ended_at"] = "2026-08-30T12:00:04.500000Z"
    with pytest.raises(ValueError, match="ended_at disagrees"):
        validate_status_against_provenance(
            wrong_end,
            provenance,
            provenance_sha256=provenance_sha256,
        )


def test_written_provenance_bundle_and_artifact_index_are_exact(tmp_path: Path) -> None:
    provenance = _run_provenance()
    resolved_config = {
        "runner_module": provenance["code"]["runner_module"],
        "runner_sha256": provenance["code"]["runner_sha256"],
        "config": provenance["configuration"]["config"],
        "config_sha256": provenance["configuration"]["config_sha256"],
        "inputs": provenance["inputs"],
        "execution": {
            "mode": provenance["execution"]["mode"],
            "command": provenance["execution"]["command"],
            "planned_at": provenance["planned_at"],
            "started_at": provenance["started_at"],
            "ended_at": provenance["ended_at"],
            "duration_seconds": provenance["duration_seconds"],
            "git": provenance["code"]["git"],
            "runtime": provenance["environment"]["runtime"],
        },
    }
    atomic_json(tmp_path / "resolved_config.json", resolved_config)
    provenance["configuration"]["resolved_config_sha256"] = sha256_file(tmp_path / "resolved_config.json")
    atomic_json(tmp_path / "run_provenance.json", provenance)
    provenance_sha256 = sha256_file(tmp_path / "run_provenance.json")
    atomic_json(tmp_path / "status.json", _status(provenance, provenance_sha256))

    bundle_checks = validate_output_provenance_bundle(tmp_path)

    assert all(bundle_checks.values())
    atomic_json(tmp_path / "artifact_index.json", audit._artifact_index(tmp_path))
    assert all(validate_artifact_index(tmp_path).values())

    resolved_config["runner_sha256"] = "0" * 64
    atomic_json(tmp_path / "resolved_config.json", resolved_config)
    with pytest.raises(ValueError, match="resolved_config_sha256"):
        validate_output_provenance_bundle(tmp_path)
    with pytest.raises(ValueError, match="hash/size mismatch"):
        validate_artifact_index(tmp_path)


def test_git_dirty_diff_digest_is_content_aware_for_untracked_files(tmp_path: Path) -> None:
    def git(*arguments: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *arguments], check=True, capture_output=True)

    git("init", "--quiet")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "NeuRev Test")
    (tmp_path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "--quiet", "-m", "fixture")
    git("remote", "add", "origin", "git@github.com:example/neurev.git")
    untracked = tmp_path / "untracked.txt"
    untracked.write_text("alpha\n", encoding="utf-8")

    first = audit._git_provenance(tmp_path)
    untracked.write_text("bravo\n", encoding="utf-8")
    second = audit._git_provenance(tmp_path)

    assert first["repository"] == "https://github.com/example/neurev"
    assert first["dirty"] is True
    assert first["diff_algorithm"] == "NEUREV-DIRTY-STATE-V1"
    assert first["dirty_status_sha256"] == second["dirty_status_sha256"]
    assert first["diff_sha256"] != second["diff_sha256"]


def test_dirty_code_projection_validates_against_repository_run_schema() -> None:
    provenance = _run_provenance()
    repository_root = Path(audit.__file__).resolve().parents[3]
    schema = audit._read_json(repository_root / "research" / "schemas" / "run.schema.json")
    code_record = {
        key: provenance["code"][key]
        for key in ("repository", "commit", "branch", "dirty", "diff_sha256")
    }

    checks = audit.validate_registry_code_against_run_schema(code_record, schema)

    assert all(checks.values())
    del code_record["diff_sha256"]
    with pytest.raises(ValueError, match="violates run.schema.json"):
        audit.validate_registry_code_against_run_schema(code_record, schema)
