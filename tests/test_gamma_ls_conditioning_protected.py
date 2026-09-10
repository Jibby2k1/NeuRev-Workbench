from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobench.algorithms.gamma_local_standardization import GammaReferenceSpec
from neurobench.experiments.gamma_ls_difference import conditioning_protected as protected
from neurobench.experiments.gamma_ls_difference import conditioning_sensitivity as upstream


REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = (
    REPOSITORY
    / "examples/spon_ca_burst_gamma_ls_conditioning_protected_v1.example.json"
)


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch) -> protected.ConditioningProtectedConfig:
    monkeypatch.setenv("NEUROBENCH_DATA_ROOT", str(REPOSITORY))
    return protected.ConditioningProtectedConfig.load(EXAMPLE)


def test_manifest_freezes_protected_crossfit_nms_budget_and_bootstrap(config) -> None:
    design = config.payload["design"]
    assert design["conditioning_roles"] == [
        "historical_anchor",
        "training_pareto_selected",
    ]
    assert design["quiet_nms_peaks_per_pseudo_burst"] == [0.25, 0.5, 1.0, 2.0, 5.0]
    assert design["nms_distance_px"] == [4, 6, 8]
    assert design["candidate_budgets_per_burst"] == [20, 40, 58, 80, 100]
    assert config.payload["bootstrap"]["cluster_count"] == 26
    assert config.payload["bootstrap"]["replicates"] == 2000
    assert config.payload["label_join"]["latest_v7_inferential_claim"] is False
    assert config.payload["sources"] == {
        "conditioning_config": (
            "repo://examples/"
            "spon_ca_burst_gamma_ls_conditioning_sensitivity_v1.example.json"
        ),
        "conditioning_preflight_artifact_index_sha256": (
            "ef1a79c135ecfde5d6dc9b4d567d79c8d9192ef3c0f3acb6b9c0cb8038a9f494"
        ),
        "conditioning_preflight_sha256": (
            "740e51413736a0e80ffb5219fad973200a1dbd585c3dc620ab65ce80eff27711"
        ),
        "conditioning_screen_artifact_index_sha256": (
            "71b55a0b56f2c4e096730d1671532ded9a031ddbaa6fe55d7db39d8c528e8efb"
        ),
        "conditioning_selection_sha256": (
            "3011bdbc3ad47260e19401ec2f7acc60f16fea195542380cd2e08f544a83a816"
        ),
    }


def test_manifest_rejects_changed_materiality_rule(tmp_path: Path, monkeypatch) -> None:
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["materiality"]["budget_curve_auc_absolute_delta"] = 0.01
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("NEUROBENCH_DATA_ROOT", str(REPOSITORY))
    with pytest.raises(
        protected.ConditioningProtectedConfigError, match="materiality rule changed"
    ):
        protected.ConditioningProtectedConfig.load(changed)


def _candidate_context(fold: int):
    half = 11 + fold
    payload = {
        "context_id": f"ctx_fold_{fold}",
        "half_width_px": half,
        "guard_radius_px": 5,
        "shape": 5.0,
        "mode_fraction_of_half_width": 0.5,
        "mode_radius_px": half * 0.5,
        "support": "radial_disk",
        "padding": "valid_renormalized_zero",
        "eligible_primary": True,
    }
    reference = GammaReferenceSpec.from_mode(
        payload["context_id"],
        support_width_px=2 * half + 1,
        shape_n=5.0,
        mode_radius_px=half * 0.5,
        guard_radius_px=5,
        support_geometry="disk",
        boundary_mode="valid_renormalized_zero",
        epsilon=1e-6,
        scale_floor=0.0,
    )
    return reference, payload


def _selection_payload(*, selected_setting=None):
    anchor = upstream.ConditioningSetting(*upstream.ANCHOR)
    selected = selected_setting or anchor
    rows = []
    for fold in upstream.OUTER_FOLDS:
        for representation in upstream.REPRESENTATIONS:
            rows.append(
                {
                    "training_fold": fold,
                    "heldout_burst": str(fold),
                    "representation": representation,
                    "support_context_id": f"ctx_fold_{fold}",
                    "selection_scope": "outer_training_fold_only",
                    "selector_uses_burst_windows": True,
                    "selector_uses_positive_coordinates": False,
                    "selector_uses_positive_identities": False,
                    "pareto_front_setting_ids": [selected.setting_id],
                    "historical_anchor": {
                        "role": "historical_anchor",
                        **anchor.as_dict(),
                        "pareto_layer": 0,
                    },
                    "training_pareto_selected": {
                        "role": "training_pareto_selected",
                        **selected.as_dict(),
                        "pareto_layer": 0,
                    },
                }
            )
    payload = {
        "schema_version": 1,
        "experiment_id": upstream.EXPERIMENT_ID,
        "selection_scope": "independent_per_representation_within_outer_training_fold",
        "selection_method": "fixture",
        "objectives": [],
        "downstream_roles": list(protected.CONDITIONING_ROLES),
        "downstream_stage": "protected_B_and_NMS_evaluation",
        "positive_coordinates_used": False,
        "positive_identities_used": False,
        "burst_windows_used": True,
        "fold_representation_selections": rows,
        "selection_hash_scope": "canonical_json_of_payload_excluding_selection_sha256",
    }
    payload["selection_sha256"] = upstream._canonical_sha256(payload)
    return payload


def _write_upstream_screen(screen: Path, preflight: Path, *, wrong_binding=False):
    screen.mkdir()
    preflight.mkdir()
    (preflight / "preflight.json").write_text("{}\n", encoding="utf-8")
    selection = _selection_payload()
    selection_sha = selection["selection_sha256"]
    preflight_sha = upstream._sha256(preflight / "preflight.json")
    (screen / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete_conditioning_sensitivity_screen_only",
                "selection_sha256": selection_sha,
            }
        ),
        encoding="utf-8",
    )
    (screen / "validation.json").write_text(
        json.dumps(
            {
                "status": (
                    "passed_conditioning_sensitivity_screen_artifact_contract_"
                    "scientific_audit_pending"
                )
            }
        ),
        encoding="utf-8",
    )
    (screen / "fold_selected_settings.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    (screen / "run_contract.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": upstream.EXPERIMENT_ID,
                "annotation_sources_reopened": False,
                "positive_coordinates_used": False,
                "positive_identities_used": False,
                "burst_windows_used": True,
                "protected_evaluation_in_this_run": False,
                "dense_map_device": "cuda",
                "outer_folds": 4,
                "fold_context_role": "support_candidate_context",
                "selection_sha256": selection_sha,
                "selection_sealed_before_any_label_join": True,
                "label_join_api_present": False,
                "sensitivity_preflight_sha256": (
                    "0" * 64 if wrong_binding else preflight_sha
                ),
                "support_artifact_index_sha256": "s" * 64,
                "support_fold_contexts_sha256": "f" * 64,
                "base_preflight_artifact_index_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    (screen / "artifact_index.json").write_text(
        json.dumps(upstream._artifact_index(screen)), encoding="utf-8"
    )


def _fake_verified_preflight():
    return {
        "artifact_index_sha256": "p" * 64,
        "base_preflight_artifact_index_sha256": "b" * 64,
        "support_artifact_index_sha256": "s" * 64,
        "support_fold_contexts_sha256": "f" * 64,
        "runtime": {
            "requested_device": "cuda",
            "resolved_device": "cuda:0",
        },
        "historical_conditioning_implementation": {},
        "fold_contexts": {
            fold: _candidate_context(fold) for fold in upstream.OUTER_FOLDS
        },
    }


def _bind_fixture_screen(config, screen: Path) -> None:
    config.payload["sources"]["conditioning_screen_artifact_index_sha256"] = (
        upstream._sha256(screen / "artifact_index.json")
    )
    config.payload["sources"]["conditioning_selection_sha256"] = (
        _selection_payload()["selection_sha256"]
    )


def test_upstream_verifier_binds_exact_preflight_screen_selection_and_24_roles(
    config, tmp_path, monkeypatch
) -> None:
    screen = tmp_path / "screen"
    preflight = tmp_path / "preflight"
    _write_upstream_screen(screen, preflight)
    _bind_fixture_screen(config, screen)
    monkeypatch.setattr(
        protected,
        "_verify_completed_conditioning_preflight_snapshot",
        lambda *args, **kwargs: _fake_verified_preflight(),
    )
    roles, contexts, provenance = protected._verify_conditioning_screen(
        config,
        conditioning_preflight_dir=preflight,
        conditioning_screen_dir=screen,
    )
    assert len(roles) == 24
    assert set(contexts) == set(upstream.OUTER_FOLDS)
    assert {role.role for role in roles} == set(protected.CONDITIONING_ROLES)
    assert provenance["conditioning_selection_sha256"] == _selection_payload()[
        "selection_sha256"
    ]


def test_upstream_verifier_rejects_wrong_preflight_binding(
    config, tmp_path, monkeypatch
) -> None:
    screen = tmp_path / "screen"
    preflight = tmp_path / "preflight"
    _write_upstream_screen(screen, preflight, wrong_binding=True)
    _bind_fixture_screen(config, screen)
    monkeypatch.setattr(
        protected,
        "_verify_completed_conditioning_preflight_snapshot",
        lambda *args, **kwargs: _fake_verified_preflight(),
    )
    with pytest.raises(
        protected.ConditioningProtectedUnavailable, match="supplied preflight"
    ):
        protected._verify_conditioning_screen(
            config,
            conditioning_preflight_dir=preflight,
            conditioning_screen_dir=screen,
        )


def test_screen_verifier_rejects_an_unpinned_complete_screen(
    config, tmp_path, monkeypatch
) -> None:
    screen = tmp_path / "screen"
    preflight = tmp_path / "preflight"
    _write_upstream_screen(screen, preflight)
    monkeypatch.setattr(
        protected,
        "_verify_completed_conditioning_preflight_snapshot",
        lambda *args, **kwargs: _fake_verified_preflight(),
    )
    with pytest.raises(
        protected.ConditioningProtectedUnavailable, match="manifest-frozen artifact"
    ):
        protected._verify_conditioning_screen(
            config,
            conditioning_preflight_dir=preflight,
            conditioning_screen_dir=screen,
        )


def _cuda_runtime(*, free_vram_bytes_before: int = 10 * 2**30):
    return {
        "requested_device": "cuda",
        "resolved_device": "cuda:0",
        "torch_version": "2.test",
        "torch_cuda_build": "13.0",
        "device_name": "fixture gpu",
        "compute_capability": [8, 9],
        "visible_device_count": 1,
        "total_vram_bytes": 12 * 2**30,
        "free_vram_bytes_before": free_vram_bytes_before,
        "cuda_available": True,
    }


def test_current_cuda_runtime_is_reverified_separately(config, monkeypatch) -> None:
    frozen = _cuda_runtime()
    monkeypatch.setattr(upstream, "_require_cuda", lambda device: dict(frozen))
    assert protected._verify_current_cuda_runtime(config, frozen) == frozen
    changed = {**frozen, "device_name": "different gpu"}
    monkeypatch.setattr(upstream, "_require_cuda", lambda device: changed)
    with pytest.raises(
        protected.ConditioningProtectedUnavailable, match="runtime identity changed"
    ):
        protected._verify_current_cuda_runtime(config, frozen)
    low_memory = _cuda_runtime(free_vram_bytes_before=7 * 2**30)
    monkeypatch.setattr(upstream, "_require_cuda", lambda device: low_memory)
    with pytest.raises(
        protected.ConditioningProtectedUnavailable, match="below the frozen"
    ):
        protected._verify_current_cuda_runtime(config, frozen)


def _roles(selected_setting=None):
    selected = selected_setting or upstream.ConditioningSetting(*upstream.ANCHOR)
    anchor = upstream.ConditioningSetting(*upstream.ANCHOR)
    rows = []
    for fold in upstream.OUTER_FOLDS:
        for representation in upstream.REPRESENTATIONS:
            for role, setting in (
                ("historical_anchor", anchor),
                ("training_pareto_selected", selected),
            ):
                rows.append(
                    protected.FrozenRole(
                        training_fold=fold,
                        heldout_burst=str(fold),
                        representation=representation,
                        support_context_id=f"ctx_fold_{fold}",
                        role=role,
                        setting=setting,
                    )
                )
    return tuple(rows)


def _mock_candidate_execution(roles=None):
    roles = roles or _roles()
    calibrations = []
    candidates = []
    timings = []
    for frozen in roles:
        common = {
            "training_fold": frozen.training_fold,
            "representation": frozen.representation,
            "conditioning_role": frozen.role,
            "conditioning_setting_id": frozen.setting.setting_id,
            "context_id": frozen.support_context_id,
        }
        timings.append({**common, "runtime_ms": 1.0})
        for swap in ("a_train_b_test", "b_train_a_test"):
            for nms in protected.NMS_DISTANCES_PX:
                for burden in protected.QUIET_NMS_PEAK_BURDENS:
                    calibrations.append(
                        {
                            **common,
                            "quiet_swap": swap,
                            "nms_distance_px": nms,
                            "target_nms_peaks_per_pseudo_burst": burden,
                            "threshold_z": 2.0,
                            "heldout_burst_candidate_count": 1,
                        }
                    )
                    candidates.append(
                        {
                            **common,
                            "quiet_swap": swap,
                            "nms_distance_px": nms,
                            "target_nms_peaks_per_pseudo_burst": burden,
                            "burst_id": frozen.training_fold,
                            "threshold_z": 2.0,
                            "candidate_rank": 1,
                            "occupancy_score": 1.0,
                            "x_px": 1,
                            "y_px": 1,
                            "interpretation_before_label_join": "unknown_candidate",
                        }
                    )
    return protected.CandidateExecution(
        candidate_rows=tuple(candidates),
        calibration_rows=tuple(calibrations),
        timing_rows=tuple(timings),
        summary={"labels_opened": False},
    )


def test_candidate_execution_validation_covers_all_720_operating_points() -> None:
    roles = _roles()
    execution = _mock_candidate_execution(roles)
    checks = protected._validate_candidate_execution(execution, roles)
    assert len(execution.calibration_rows) == 720
    assert checks["nms_4_6_8_complete"] is True
    changed = list(execution.candidate_rows)
    changed[0] = {**changed[0], "interpretation_before_label_join": "classified"}
    with pytest.raises(ValueError, match="unknown before"):
        protected._validate_candidate_execution(
            protected.CandidateExecution(
                tuple(changed),
                execution.calibration_rows,
                execution.timing_rows,
                execution.summary,
            ),
            roles,
        )
    changed_calibration = list(execution.calibration_rows)
    changed_calibration[0] = {
        **changed_calibration[0],
        "heldout_burst_candidate_count": 2,
    }
    with pytest.raises(ValueError, match="does not reconcile"):
        protected._validate_candidate_execution(
            protected.CandidateExecution(
                execution.candidate_rows,
                tuple(changed_calibration),
                execution.timing_rows,
                execution.summary,
            ),
            roles,
        )


def _bootstrap_match_rows(*, equal_roles: bool = False):
    rows = []
    identities = [f"roi_{index:02d}" for index in range(26)]
    for role in protected.CONDITIONING_ROLES:
        for burst in upstream.OUTER_FOLDS:
            burst_identities = identities[burst - 1 :: 4]
            for budget in protected.CANDIDATE_BUDGETS_PER_BURST:
                for index, identity in enumerate(burst_identities):
                    anchor_match = index == 0
                    selected_match = anchor_match if equal_roles else index <= 2
                    rows.append(
                        {
                            "context_role": role,
                            "representation": "raw",
                            "quiet_swap": "a_train_b_test",
                            "nms_distance_px": 6,
                            "target_nms_peaks_per_pseudo_burst": 1.0,
                            "candidate_budget": budget,
                            "burst_id": burst,
                            "canonical_roi_id": identity,
                            "matched": (
                                anchor_match
                                if role == "historical_anchor"
                                else selected_match
                            ),
                        }
                    )
    return rows


def test_26_identity_paired_bootstrap_is_deterministic_and_role_paired() -> None:
    rows = _bootstrap_match_rows()
    first = protected.conditioning_clustered_bootstrap_contrasts(
        rows, seed=20260908, replicates=40
    )
    second = protected.conditioning_clustered_bootstrap_contrasts(
        rows, seed=20260908, replicates=40
    )
    assert first == second
    assert len(first) == 2  # one quiet swap plus its crossfit-average view
    assert all(row["cluster_count"] == 26 for row in first)
    assert all(row["selected_minus_anchor_budget_curve_auc"] > 0 for row in first)

    equal = protected.conditioning_clustered_bootstrap_contrasts(
        _bootstrap_match_rows(equal_roles=True), seed=20260908, replicates=20
    )
    assert all(row["selected_minus_anchor_budget_curve_auc"] == 0 for row in equal)
    assert all(row["b58_delta_ci95_low"] == 0 for row in equal)


def _synthetic_contrasts(direction="material_improvement"):
    rows = []
    for representation in upstream.REPRESENTATIONS:
        for quiet_scope in (
            "a_train_b_test",
            "b_train_a_test",
            "crossfit_average",
        ):
            for nms in protected.NMS_DISTANCES_PX:
                for burden in protected.QUIET_NMS_PEAK_BURDENS:
                    rows.append(
                        {
                            "representation": representation,
                            "quiet_scope": quiet_scope,
                            "nms_distance_px": nms,
                            "target_nms_peaks_per_pseudo_burst": burden,
                            "materiality_direction": direction,
                            "bootstrap_replicates": 2000,
                        }
                    )
    return rows


def test_materiality_requires_all_five_primary_burdens_same_direction() -> None:
    rows = _synthetic_contrasts()
    result = protected.protected_curve_materiality(rows)
    assert all(
        row["conclusion"] == "material_improvement_across_curve_family"
        for row in result["decisions"]
    )
    target = next(
        row
        for row in rows
        if row["representation"] == "raw"
        and row["quiet_scope"] == "crossfit_average"
        and row["nms_distance_px"] == 6
        and row["target_nms_peaks_per_pseudo_burst"] == 1.0
    )
    target["materiality_direction"] = "no_material_change_at_this_operating_point"
    result = protected.protected_curve_materiality(rows)
    raw = next(row for row in result["decisions"] if row["representation"] == "raw")
    assert raw["conclusion"] == "no_consistent_material_change_across_curve_family"


def _labels(count: int, identities: int):
    return [
        {
            "observation_id": f"obs_{index}",
            "burst_id": index % 4 + 1,
            "canonical_roi_id": f"roi_{index % identities}",
            "x_px": 1.0,
            "y_px": 1.0,
        }
        for index in range(count)
    ]


def test_mock_run_seals_candidates_before_label_join(config, tmp_path, monkeypatch) -> None:
    roles = _roles()
    execution = _mock_candidate_execution(roles)
    provenance = {
        "artifact_index_sha256": "a" * 64,
        "conditioning_preflight_artifact_index_sha256": "b" * 64,
        "conditioning_preflight_sha256": "c" * 64,
        "conditioning_selection_sha256": "d" * 64,
        "conditioning_preflight_runtime": _cuda_runtime(),
    }
    verification_count = 0

    def verified_screen(*args, **kwargs):
        nonlocal verification_count
        verification_count += 1
        return roles, {}, provenance

    monkeypatch.setattr(protected, "_verify_conditioning_screen", verified_screen)
    monkeypatch.setattr(
        protected,
        "_execute_label_free_candidates",
        lambda *args, **kwargs: execution,
    )
    monkeypatch.setattr(
        upstream, "_require_cuda", lambda device: _cuda_runtime()
    )
    join_observed = False

    def joined(config, *, work_dir, conditioning_preflight_dir):
        nonlocal join_observed
        seal = protected._verify_candidate_seal(work_dir)
        join_observed = seal["verified_before_label_join"]
        return _labels(79, 26), _labels(106, 53), {
            **seal,
            "label_fields_first_parsed_after_candidate_seal": True,
            "label_source_hashes": {"v1": "1" * 64, "v7": "2" * 64},
        }

    monkeypatch.setattr(protected, "_read_labels_after_candidate_seal", joined)
    monkeypatch.setattr(
        protected,
        "observation_match_rows",
        lambda *args, cohort, **kwargs: [{"cohort": cohort, "matched": True}],
    )
    monkeypatch.setattr(
        protected, "aggregate_match_rows", lambda rows: [{"metric": 1.0}]
    )
    monkeypatch.setattr(protected, "_summary_by_arm", lambda rows: [{"metric": 1.0}])
    contrasts = _synthetic_contrasts("no_material_change_at_this_operating_point")
    monkeypatch.setattr(
        protected,
        "conditioning_clustered_bootstrap_contrasts",
        lambda *args, **kwargs: contrasts,
    )
    output = tmp_path / "protected"
    summary = protected.run_conditioning_protected(
        config,
        conditioning_preflight_dir=tmp_path / "preflight",
        conditioning_screen_dir=tmp_path / "screen",
        output_dir=output,
    )
    assert join_observed is True
    assert verification_count == 2
    assert summary["candidate_seal"]["label_fields_parsed_before_seal"] is False
    assert summary["claim_boundary"]["unmatched_candidates"] == "unknown_not_negative"
    assert upstream._verify_indexed_artifact(output, role="test")[
        "verified_artifact_count"
    ] > 15


def test_stale_upstream_fails_before_output_mutation(config, tmp_path, monkeypatch) -> None:
    output = tmp_path / "output"

    def stale(*args, **kwargs):
        raise protected.ConditioningProtectedUnavailable("fixture stale screen")

    monkeypatch.setattr(protected, "_verify_conditioning_screen", stale)
    with pytest.raises(
        protected.ConditioningProtectedUnavailable, match="stale screen"
    ):
        protected.run_conditioning_protected(
            config,
            conditioning_preflight_dir=tmp_path / "preflight",
            conditioning_screen_dir=tmp_path / "screen",
            output_dir=output,
        )
    assert not output.exists()
