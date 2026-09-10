from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from neurobench.experiments.neuron_identifiability import uncertainty_aware_learning as runner


REPOSITORY = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _preflight_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    repository = tmp_path / "repo"
    data = tmp_path / "data"
    repository.mkdir(parents=True)
    data.mkdir(parents=True)
    _write(repository / "AGENTS.md", "synthetic test authority\n")
    _write(
        repository / runner.DEPENDENCY_DECLARATION_RELATIVE_PATH,
        "[project]\nname='synthetic'\n",
    )

    config = json.loads(
        (REPOSITORY / runner.CONFIG_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    config["outer_validation"]["scheme"] = (
        "five_positive_anchor_median_x_blocks_with_whole_leakage_group_closure"
    )
    protocol = repository / runner.PROTOCOL_RELATIVE_PATH
    _write(protocol, "frozen synthetic protocol\n")
    config["protocol_sha256"] = _sha(protocol)

    source_records = []
    for index, (source_id, locator) in enumerate(runner.EXPECTED_SOURCE_LOCATORS.items()):
        path = runner._source_path(locator, repository=repository, data=data)
        _write(path, f"synthetic source {index}\n")
        source_records.append(
            {
                "source_id": source_id,
                "role": "synthetic test source",
                "portable_path": locator,
                "sha256": _sha(path),
                "payload_committed": False,
            }
        )
    descriptor = {
        "schema_version": 1,
        "descriptor_id": runner.DESCRIPTOR_ID,
        "experiment_id": runner.EXPERIMENT_ID,
        "sources": source_records,
    }
    descriptor_path = repository / runner.DESCRIPTOR_RELATIVE_PATH
    _write(descriptor_path, json.dumps(descriptor, sort_keys=True) + "\n")
    config["input_descriptor_sha256"] = _sha(descriptor_path)

    implementation_by_path = {
        record["path"]: record
        for record in config["implementation_hash_contract"]["sources"]
    }
    for relative, role in runner.EXPECTED_IMPLEMENTATIONS.items():
        path = repository / relative
        _write(path, f"synthetic implementation {relative}\n")
        implementation_by_path[relative]["role"] = role
        implementation_by_path[relative]["sha256"] = _sha(path)
    runner_path = repository / runner.RUNNER_RELATIVE_PATH
    monkeypatch.setattr(runner, "_runner_path", lambda: runner_path.resolve())
    for module, relative in (
        (
            runner.census_core,
            "neurobench/experiments/neuron_identifiability/uncertainty_aware_census.py",
        ),
        (
            runner.data_core,
            "neurobench/experiments/neuron_identifiability/uncertainty_aware_data.py",
        ),
        (
            runner.model_core,
            "neurobench/experiments/neuron_identifiability/uncertainty_aware_models.py",
        ),
    ):
        monkeypatch.setattr(module, "__file__", str(repository / relative))
    for module_name, relative in runner.LOADED_UPSTREAM_MODULES.items():
        module = runner.sys.modules[module_name]
        monkeypatch.setattr(module, "__file__", str(repository / relative))
    monkeypatch.setattr(
        runner,
        "_validate_repository_authority",
        lambda *_args, **_kwargs: {
            "repository": "https://github.com/Jibby2k1/NeuRev-Workbench",
            "commit": "1" * 40,
            "branch": "codex/neuron-identifiability-paper-20260822",
            "dirty": False,
            "dirty_entry_count": 0,
            "dirty_status_sha256": "0" * 64,
            "dirty_paths_persisted": False,
            "diff_algorithm": "NEUREV-DIRTY-STATE-V1",
        },
    )
    monkeypatch.setattr(
        runner.census_core,
        "preflight_uncertainty_aware_census",
        lambda **_: {"schema_version": 1, "ready": True},
    )
    config_path = repository / runner.CONFIG_RELATIVE_PATH
    _write(config_path, json.dumps(config, indent=2, sort_keys=True) + "\n")
    return repository, data, config_path


def _fold_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidate = 0
    for block in range(5):
        for offset in range(10):
            x = float(block * 200 + offset * 20)
            identity = f"identity_{block}_{offset}"
            positive_group = f"positive_group_{block}_{offset}"
            rows.append(
                {
                    "candidate_id": f"candidate_{candidate:04d}",
                    "x_px": x,
                    "y_px": float(offset),
                    "identity_group_id": identity,
                    "spatial_group_id": positive_group,
                    "leakage_group_id": positive_group,
                    "training_state": "positive",
                    "primary_anchor_canonical_roi_id": identity,
                    "eligible_for_primary_fit": True,
                    "eligible_for_anchor_balance": True,
                }
            )
            candidate += 1
            # Exact-x overlap forces the two leakage intervals into one atomic
            # x component while retaining distinct leakage groups.
            unlabeled_group = f"unlabeled_group_{block}_{offset}"
            rows.append(
                {
                    "candidate_id": f"candidate_{candidate:04d}",
                    "x_px": x,
                    "y_px": float(offset + 50),
                    "identity_group_id": f"u_identity_{block}_{offset}",
                    "spatial_group_id": unlabeled_group,
                    "leakage_group_id": unlabeled_group,
                    "training_state": "unlabeled",
                    "primary_anchor_canonical_roi_id": None,
                    "eligible_for_primary_fit": True,
                    "eligible_for_anchor_balance": False,
                }
            )
            candidate += 1
    # A second occurrence in one near-boundary leakage group proves that the
    # training guard purges the complete group rather than a single row.
    rows.append(
        {
            **rows[39],
            "candidate_id": f"candidate_{candidate:04d}",
            "x_px": float(rows[39]["x_px"]) + 1.0,
        }
    )
    return rows


def _install_public_run_smoke_mocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_thread_cap: bool = False,
    raise_deadline_during_census: bool = False,
    fail_thread_restore: bool = False,
) -> tuple[Path, Path, Path]:
    repository, data, config_path = _preflight_fixture(tmp_path, monkeypatch)
    fold_rows = _fold_rows()
    real_fold_builder = runner.build_contiguous_x_block_folds
    full_rows = []
    model_rows = []
    for index, fold_row in enumerate(fold_rows):
        is_positive = fold_row["training_state"] == "positive"
        features = {
            feature: (1.0 if is_positive else 0.0) + feature_index * 1e-3
            for feature_index, feature in enumerate(runner.PRIMARY_FEATURES)
        }
        canonical_id = (
            str(fold_row["primary_anchor_canonical_roi_id"])
            if is_positive
            else None
        )
        full_rows.append(
            {
                **fold_row,
                "partition": "event",
                "partition_id": "p01",
                "source_count": 1,
                "features": features,
                "canonical_v7": {
                    "observation_id": f"obs_{index:04d}" if is_positive else None,
                    "canonical_roi_id": canonical_id,
                    "match_distance_px": 0.0 if is_positive else None,
                    "anchor_status": "primary_positive" if is_positive else "unmatched",
                    "inside_confirmed_exclusion_radius": is_positive,
                },
                "review_holdout": {
                    "reserved": False,
                    "nearest_review_site_id": None,
                    "review_label": None,
                },
            }
        )
        model_rows.append(
            {
                "row_id": fold_row["candidate_id"],
                "partition_id": "p01",
                "training_state": fold_row["training_state"],
                "leakage_group_id": fold_row["leakage_group_id"],
                **features,
            }
        )

    class FakeHarmonized:
        rows = full_rows
        feature_columns = runner.PRIMARY_FEATURES
        manifest = {"contract": "synthetic_public_runner_smoke"}
        validation = {
            "model_readiness": {"passed": True, "failures": []},
            "confirmed_canonical_labels": 106,
            "confirmed_positive_anchors": 105,
            "primary_positive_anchors_after_review_reserve": 78,
            "confirmed_anchors_reserved_for_review": 27,
            "confirmed_labels_without_anchor": 1,
            "review_holdout": {
                "canonical_v7_overlap": {
                    "reviewed_sites_with_confirmed_canonical_v7_match": 0,
                    "reviewed_sites_with_cross_source_disagreement": 0,
                }
            },
        }

        @staticmethod
        def assert_model_ready() -> None:
            return None

        @staticmethod
        def fold_assignment_rows() -> tuple[dict[str, object], ...]:
            return tuple(fold_rows)

        @staticmethod
        def model_feature_rows() -> tuple[dict[str, object], ...]:
            return tuple(model_rows)

        @staticmethod
        def validate_outer_fold_assignments(assignments: object) -> dict[str, object]:
            assert len(assignments) == len(fold_rows)
            return {"passed": True, "synthetic": True}

    def fake_census_build(**kwargs: object) -> dict[str, object]:
        output_root = Path(kwargs["output_root"])
        if raise_deadline_during_census:
            runner.signal.raise_signal(runner.signal.SIGALRM)
        runner._write_tsv(
            output_root / "event_candidates.tsv",
            [{"candidate_id": "event_1", "partition": "event"}],
        )
        runner._write_tsv(
            output_root / "quiet_candidates.tsv",
            [{"candidate_id": "quiet_1", "partition": "quiet"}],
        )
        runner._write_json(
            output_root / "provenance.json",
            {"schema_version": 1, "synthetic": True},
        )
        return {"ready": True}

    def fake_evaluation(
        _X: object,
        labels: list[str],
        _groups: list[str],
        **_kwargs: object,
    ) -> dict[str, object]:
        scores = [1.0 if label == "positive" else 0.0 for label in labels]

        def payload() -> dict[str, object]:
            return {
                "metrics": {"macro_fold_positive_vs_unlabeled_rank_auc": 1.0},
                "oof_scores": list(scores),
                "oof_scores_fold_percentile": list(scores),
                "oof_scores_raw_fold_specific": list(scores),
                "spatial_group_score_association_null": {"known_invalid": True},
            }

        def scalar_baseline_payload() -> dict[str, object]:
            result = payload()
            result["scores"] = result.pop("oof_scores")
            result["scores_raw_fold_specific"] = result.pop(
                "oof_scores_raw_fold_specific"
            )
            return result

        tiny = payload()
        tiny["seed_oof_scores"] = {
            str(seed): list(scores) for seed in range(3101, 3111)
        }
        tiny["seed_rank_stability"] = {
            "median_pairwise_spearman": 1.0,
            "minimum_pairwise_spearman": 1.0,
        }
        return {
            "estimand": "synthetic model-only spatial holdout",
            "baselines": {
                # Match the evaluator's real external-score schema.  Learned
                # model payloads use oof_scores; scalar baselines use scores.
                "carrier_signed": scalar_baseline_payload(),
                "cfar_score": scalar_baseline_payload(),
            },
            "models": {
                "equal_weight_feature_separation": payload(),
                "positive_reference_distance": payload(),
                "logistic_l2": payload(),
                "logistic_elastic": payload(),
                "bagged_pu_logistic": payload(),
                "tiny_mlp_4_tanh": tiny,
            },
        }

    def fake_representative_null(
        *_args: object, **_kwargs: object
    ) -> dict[str, object]:
        fold_support = ((9, 317), (3, 14), (3, 22), (2, 38), (4, 128))
        representatives = []
        for fold_id, (positive_count, unlabeled_count) in enumerate(
            fold_support, start=1
        ):
            for unit_index in range(positive_count):
                representatives.append(
                    {
                        "unit_id": f"positive_containing_leakage_component::f{fold_id}_p{unit_index}",
                        "unit_type": "positive_containing_leakage_component",
                        "source_group_id": f"f{fold_id}_p{unit_index}",
                        "outer_fold": fold_id,
                        "representative_candidate_id": f"p_rep_{fold_id}_{unit_index}",
                        "representative_input_index": 0,
                        "source_group_candidate_count": 1,
                        "is_positive_unit": True,
                        "score": 0.5,
                    }
                )
            for unit_index in range(unlabeled_count):
                representatives.append(
                    {
                        "unit_id": f"unlabeled_only_leakage_component::f{fold_id}_u{unit_index}",
                        "unit_type": "unlabeled_only_leakage_component",
                        "source_group_id": f"f{fold_id}_u{unit_index}",
                        "outer_fold": fold_id,
                        "representative_candidate_id": f"u_rep_{fold_id}_{unit_index}",
                        "representative_input_index": 0,
                        "source_group_candidate_count": 1,
                        "is_positive_unit": False,
                        "score": 0.5,
                    }
                )
        return {
            "null_semantics": "synthetic_corrected_representative_unit_null",
            "observed_macro_fold_representative_unit_spu_auc": 0.5,
            "observed_by_fold": [
                {
                    "fold_id": fold_id,
                    "positive_containing_leakage_component_count": support[0],
                    "unlabeled_only_leakage_component_count": support[1],
                    "positive_vs_unlabeled_rank_auc": 0.5,
                }
                for fold_id, support in enumerate(fold_support, start=1)
            ],
            "draws": 2000,
            "seed": 4201,
            "positive_containing_leakage_component_count": 21,
            "unlabeled_only_leakage_component_count": 519,
            "greater_or_equal_p_value": 1.0,
            "representatives": representatives,
        }

    def fake_macro_bootstrap(*_args: object, **_kwargs: object) -> dict[str, object]:
        supports = ((9, 317), (3, 14), (3, 22), (2, 38), (4, 128))
        mixed_u = (30, 30, 30, 24, 24)
        u_only_rows = (400, 50, 50, 200, 703)
        return {
            "observed_score_a_macro_fold_spu_auc": 1.0,
            "observed_score_b_macro_fold_spu_auc": 1.0,
            "observed_delta_a_minus_b": 0.0,
            "resampling_unit": "whole_leakage_group_with_all_positive_and_unlabeled_members_joint",
            "observed_by_fold": [
                {
                    "fold_id": fold_id,
                    "positive_containing_leakage_component_count": support[0],
                    "unlabeled_only_leakage_component_count": support[1],
                    "unlabeled_rows_inside_positive_containing_components": mixed_u[
                        fold_id - 1
                    ],
                    "unlabeled_rows_inside_unlabeled_only_components": u_only_rows[
                        fold_id - 1
                    ],
                    "score_a": 1.0,
                    "score_b": 1.0,
                }
                for fold_id, support in enumerate(supports, start=1)
            ],
            "bootstrap_delta_ci95_low": 0.0,
            "bootstrap_delta_ci95_high": 0.0,
        }

    def fake_misses(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "rows": [
                {
                    "observation_id": "reserved_miss",
                    "burst_id": 4,
                    "canonical_roi_id": "roi_027",
                    "x_px": 0.0,
                    "y_px": 0.0,
                    "inside_locked_review_reserve": True,
                    "nearest_review_site_id": "dsite_026",
                    "nearest_review_distance_px": 2.98,
                    "nearest_frozen_candidate_id": "candidate_nearest",
                    "nearest_frozen_candidate_distance_px": 8.6693335961,
                    "no_candidate_within_match_radius": True,
                    "candidate_match_radius_px": 6.0,
                    "primary_denominator_role": "excluded_review_reserve",
                }
            ],
            "total_unmatched_confirmed": 1,
            "reserved_unmatched_confirmed": 1,
            "unreserved_unmatched_confirmed": 0,
        }

    def fake_audits(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "review_rows": [
                {
                    "candidate_id": "review_candidate",
                    "partition_id": "p01",
                    "training_state": "heldout_review",
                    "review_site_id": "dsite_001",
                    "review_label": "uncertain",
                    "canonical_anchor_status": "none",
                    "inside_confirmed_exclusion_radius": False,
                    "tiny_mlp_final_fit_score": 0.5,
                    "audit_role": "locked_review_conflict_panel_descriptive_only",
                }
            ],
            "quiet_rows": [
                {
                    "candidate_id": "quiet_candidate",
                    "partition_id": "q01",
                    "training_state": "source_off_null",
                    "tiny_mlp_final_fit_score": 0.1,
                    "audit_role": "source_off_null_control_descriptive_only",
                }
            ],
            "final_fit": {
                "training_candidate_count": len(model_rows),
                "review_candidate_count": 1,
                "quiet_candidate_count": 1,
                "seed_count": 10,
                "panel_values_used_for_selection_or_tuning": False,
            },
        }

    monkeypatch.setattr(runner.census_core, "build_uncertainty_aware_census", fake_census_build)
    def synthetic_frozen_fold_builder(*args: object, **kwargs: object) -> dict[str, object]:
        plan = real_fold_builder(*args, **kwargs)
        plan["assignment_sha256"] = (
            "517fbb0ef37ea7b57917367f9e39efe931c01ea9f848a53acde097cc47b221ba"
        )
        plan["train_test_guard_membership_sha256"] = (
            "4425f9a12a8d367969edd50c2daf4192b450d4de27fc5776628881c5538e0a9c"
        )
        return plan

    monkeypatch.setattr(
        runner, "build_contiguous_x_block_folds", synthetic_frozen_fold_builder
    )
    monkeypatch.setattr(
        runner.data_core,
        "harmonize_innovation_candidate_census",
        lambda *_args, **_kwargs: FakeHarmonized(),
    )
    monkeypatch.setattr(runner.model_core, "nested_spatial_pu_evaluation", fake_evaluation)
    monkeypatch.setattr(
        runner.model_core,
        "grouped_paired_bootstrap_rank_delta",
        lambda *_args, **_kwargs: {"synthetic": True},
    )
    monkeypatch.setattr(
        runner,
        "grouped_paired_bootstrap_macro_fold_auc",
        fake_macro_bootstrap,
    )
    monkeypatch.setattr(
        runner,
        "fold_stratified_representative_unit_score_permutation_null",
        fake_representative_null,
    )
    monkeypatch.setattr(runner, "classify_proposal_stage_misses", fake_misses)
    monkeypatch.setattr(runner, "_score_locked_audits", fake_audits)
    monkeypatch.setattr(
        runner,
        "aggregate_review_site_audit",
        lambda *_args, **_kwargs: [
            {
                "review_site_id": f"dsite_{index:03d}",
                "canonical_v7_confirmed_overlap": False,
                "cross_source_disagreement": False,
                "panel_role": "synthetic_locked_site",
            }
            for index in range(1, 19)
        ],
    )
    monkeypatch.setattr(
        runner,
        "capture_git_provenance",
        lambda _repository: {
            "repository": "https://example.test/synthetic",
            "commit": "0" * 40,
            "branch": "codex/synthetic",
            "dirty": False,
            "dirty_entry_count": 0,
            "dirty_status_sha256": "0" * 64,
            "dirty_paths_persisted": False,
            "diff_algorithm": "NEUREV-DIRTY-STATE-V1",
        },
    )

    class DummyThreadLimiter:
        @staticmethod
        def restore_original_limits() -> None:
            if fail_thread_restore:
                raise RuntimeError("injected thread-limit restore failure")
            return None

    monkeypatch.setattr(
        runner,
        "threadpool_info",
        lambda: []
        if fail_thread_cap
        else [{"user_api": "blas", "internal_api": "synthetic", "num_threads": 1}],
    )
    monkeypatch.setattr(
        runner,
        "threadpool_limits",
        lambda **_kwargs: DummyThreadLimiter(),
    )
    return repository, data, config_path


def test_preflight_is_read_only_and_hash_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository, data, config_path = _preflight_fixture(tmp_path, monkeypatch)
    before = sorted(path.relative_to(repository) for path in repository.rglob("*") if path.is_file())
    result = runner.preflight_uncertainty_aware_learning(
        config_path, repository_root=repository, data_root=data
    )
    after = sorted(path.relative_to(repository) for path in repository.rglob("*") if path.is_file())
    assert result["ready"] is True
    assert result["gates"]["implementation_hashes_exact"] is True
    assert result["claim_boundary"] == {
        "scientific_audit_complete": False,
        "scientific_completion": False,
        "claim_promotion_allowed": False,
    }
    assert before == after
    assert not (repository / runner.OUTPUT_RELATIVE_PATH).exists()


@pytest.mark.parametrize("collision", ["final", "partial"])
def test_preflight_refuses_output_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, collision: str
) -> None:
    repository, data, config_path = _preflight_fixture(tmp_path, monkeypatch)
    output = repository / runner.OUTPUT_RELATIVE_PATH
    target = output if collision == "final" else output.with_name(output.name + ".partial")
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="output already exists"):
        runner.preflight_uncertainty_aware_learning(
            config_path, repository_root=repository, data_root=data
        )


def test_preflight_refuses_id_root_and_hash_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, data, config_path = _preflight_fixture(tmp_path, monkeypatch)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["experiment_id"] = "NREV-EXP-WRONG"
    _write(config_path, json.dumps(config) + "\n")
    with pytest.raises(runner.LearningContractError, match="experiment_id"):
        runner.preflight_uncertainty_aware_learning(
            config_path, repository_root=repository, data_root=data
        )

    repository, data, config_path = _preflight_fixture(tmp_path / "root", monkeypatch)
    monkeypatch.setattr(runner, "_runner_path", lambda: (tmp_path / "outside.py").resolve())
    with pytest.raises(runner.LearningContractError, match="loaded runner"):
        runner.preflight_uncertainty_aware_learning(
            config_path, repository_root=repository, data_root=data
        )

    repository, data, config_path = _preflight_fixture(tmp_path / "module", monkeypatch)
    monkeypatch.setattr(runner.data_core, "__file__", str(tmp_path / "outside_data.py"))
    with pytest.raises(runner.LearningContractError, match="loaded module authority"):
        runner.preflight_uncertainty_aware_learning(
            config_path, repository_root=repository, data_root=data
        )

    repository, data, config_path = _preflight_fixture(tmp_path / "hash", monkeypatch)
    source_path = runner._source_path(
        runner.EXPECTED_SOURCE_LOCATORS["canonical_v7_adjudication"],
        repository=repository,
        data=data,
    )
    _write(source_path, "mutated bytes\n")
    with pytest.raises(runner.LearningContractError, match="input hash differs"):
        runner.preflight_uncertainty_aware_learning(
            config_path, repository_root=repository, data_root=data
        )


def test_preflight_refuses_descriptor_and_unpinned_code_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, data, config_path = _preflight_fixture(tmp_path, monkeypatch)
    descriptor_path = repository / runner.DESCRIPTOR_RELATIVE_PATH
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["descriptor_id"] = "WRONG"
    _write(descriptor_path, json.dumps(descriptor) + "\n")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["input_descriptor_sha256"] = _sha(descriptor_path)
    _write(config_path, json.dumps(config) + "\n")
    with pytest.raises(runner.LearningContractError, match="descriptor ID"):
        runner.preflight_uncertainty_aware_learning(
            config_path, repository_root=repository, data_root=data
        )

    repository, data, config_path = _preflight_fixture(tmp_path / "pending", monkeypatch)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["implementation_hash_contract"]["sources"][-1]["sha256"] = "PENDING"
    _write(config_path, json.dumps(config) + "\n")
    with pytest.raises(runner.LearningContractError, match="not pinned"):
        runner.preflight_uncertainty_aware_learning(
            config_path, repository_root=repository, data_root=data
        )


def test_config_and_resolved_evaluator_contract_are_exact() -> None:
    config = json.loads(
        (REPOSITORY / runner.CONFIG_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    runner._validate_config(config)
    resolved = runner._evaluation_config(config)
    assert resolved.inner_splits == 2
    assert resolved.logistic_penalties == (0.01,)
    assert resolved.elastic_penalties == (0.01,)
    assert resolved.pu_penalties == (0.01,)
    assert resolved.logistic_tolerance == 1e-6
    assert resolved.max_logistic_iterations == 50_000
    assert resolved.permutation_draws == 0

    config["models"]["bagged_pu_linear"]["base_convergence_tolerance"] = 1e-7
    with pytest.raises(runner.LearningContractError, match="bagged tolerance"):
        runner._validate_config(config)


def test_run_b_contract_changes_only_version_and_implementation_bindings() -> None:
    run_a = json.loads(
        (
            REPOSITORY
            / "examples/uncertainty_aware_feature_learning_v1.example.json"
        ).read_text(encoding="utf-8")
    )
    run_b = json.loads(
        (REPOSITORY / runner.CONFIG_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    assert run_a["run_id"] == "NREV-RUN-EXP-0021-SCREEN-20260830-A"
    assert run_b["run_id"] == runner.RUN_ID == (
        "NREV-RUN-EXP-0021-SCREEN-20260830-B"
    )
    assert run_a["output_root"] != run_b["output_root"]
    assert run_b["output_root"] == runner.OUTPUT_RELATIVE_PATH.as_posix()
    assert run_a["protocol_path"] == (
        "docs/workflows/uncertainty_aware_feature_learning_v1.md"
    )
    assert run_b["protocol_path"] == runner.PROTOCOL_RELATIVE_PATH.as_posix()

    normalized_b = json.loads(json.dumps(run_b))
    for key in (
        "run_id",
        "planned_at_utc",
        "protocol_path",
        "protocol_sha256",
        "output_root",
    ):
        normalized_b[key] = run_a[key]
    runner_path = runner.RUNNER_RELATIVE_PATH.as_posix()
    run_a_sources = {
        record["path"]: record
        for record in run_a["implementation_hash_contract"]["sources"]
    }
    for record in normalized_b["implementation_hash_contract"]["sources"]:
        if record["path"] == runner_path:
            record["sha256"] = run_a_sources[runner_path]["sha256"]
    assert normalized_b == run_a


def test_canonical_score_accessor_accepts_exact_learned_and_scalar_schemas() -> None:
    learned = {
        "oof_scores_fold_percentile": [0.1, 0.9],
        "oof_scores": [0.1, 0.9],
        "oof_scores_raw_fold_specific": [-2.0, 3.0],
    }
    scalar = {
        "oof_scores_fold_percentile": [0.2, 0.8],
        "scores": [0.2, 0.8],
        "scores_raw_fold_specific": [-1.0, 4.0],
    }
    assert runner._method_score_vector(learned, expected_length=2).tolist() == [
        0.1,
        0.9,
    ]
    assert runner._method_score_vector(scalar, expected_length=2).tolist() == [
        0.2,
        0.8,
    ]
    assert runner._method_score_vector(
        learned, raw_fold_specific=True, expected_length=2
    ).tolist() == [-2.0, 3.0]
    assert runner._method_score_vector(
        scalar, raw_fold_specific=True, expected_length=2
    ).tolist() == [-1.0, 4.0]
    with pytest.raises(runner.LearningContractError, match="exactly one canonical"):
        runner._method_score_vector({**learned, "scores": [0.1, 0.9]})
    with pytest.raises(runner.LearningContractError, match="differ from schema alias"):
        runner._method_score_vector({**scalar, "scores": [0.2, 0.7]})
    with pytest.raises(runner.LearningContractError, match="raw score alias differs"):
        runner._method_score_vector(
            {
                **learned,
                "scores_raw_fold_specific": learned[
                    "oof_scores_raw_fold_specific"
                ],
            },
            raw_fold_specific=True,
        )

def test_contiguous_folds_cover_once_preserve_groups_and_whole_guard() -> None:
    rows = _fold_rows()
    plan = runner.build_contiguous_x_block_folds(rows)
    assert len(plan["folds"]) == 5
    assert len(plan["assignments"]) == len(rows)
    assert min(fold["test_positive_anchor_count"] for fold in plan["folds"]) == 10
    assert min(fold["test_unlabeled_spatial_group_count"] for fold in plan["folds"]) == 10
    assert [row["boundary_x_px"] for row in plan["boundaries"]] == sorted(
        row["boundary_x_px"] for row in plan["boundaries"]
    )
    assert all(
        left["nominal_maximum_x_px"] == right["nominal_minimum_x_px"]
        for left, right in zip(plan["folds"], plan["folds"][1:])
    )
    model_rows = [
        {
            "row_id": row["candidate_id"],
            "training_state": row["training_state"],
            "leakage_group_id": row["leakage_group_id"],
        }
        for row in rows
    ]
    frozen = runner.materialize_frozen_outer_folds(model_rows, plan)
    assert sorted(index for _, test in frozen for index in test) == list(range(len(rows)))
    by_id = {str(row["candidate_id"]): row for row in rows}
    for fold in plan["folds"]:
        train_groups = {by_id[value]["leakage_group_id"] for value in fold["train_candidate_ids"]}
        test_groups = {by_id[value]["leakage_group_id"] for value in fold["test_candidate_ids"]}
        assert train_groups.isdisjoint(test_groups)
        purged = set(fold["guard_purged_candidate_ids"])
        for group in fold["guard_purged_leakage_group_ids"]:
            expected = {str(row["candidate_id"]) for row in rows if row["leakage_group_id"] == group}
            assert expected <= purged


def test_fold_builder_fails_closed_on_identity_split_and_support() -> None:
    rows = _fold_rows()
    rows[2]["primary_anchor_canonical_roi_id"] = rows[0]["primary_anchor_canonical_roi_id"]
    with pytest.raises(runner.LearningContractError, match="identity spans leakage"):
        runner.build_contiguous_x_block_folds(rows)

    rows = _fold_rows()
    for row in rows:
        if row["training_state"] == "positive" and float(row["x_px"]) >= 800:
            row["eligible_for_primary_fit"] = False
    with pytest.raises(runner.LearningContractError, match="positive"):
        runner.build_contiguous_x_block_folds(rows)


def test_euclidean_guard_closes_whole_group_extension_leak() -> None:
    rows = _fold_rows()
    train_candidate = next(
        row
        for row in rows
        if row["leakage_group_id"] == "unlabeled_group_0_8"
    )
    extended_test_group = next(
        row
        for row in rows
        if row["leakage_group_id"] == "positive_group_1_0"
    )
    rows.append(
        {
            **extended_test_group,
            "candidate_id": "candidate_extension_guard",
            "x_px": float(train_candidate["x_px"]) + 10.0,
            "y_px": train_candidate["y_px"],
            "training_state": "unlabeled",
            "spatial_group_id": "extension_u_spatial",
            "primary_anchor_canonical_roi_id": None,
            "eligible_for_anchor_balance": False,
        }
    )
    plan = runner.build_contiguous_x_block_folds(rows)
    fold_two = plan["folds"][1]
    purged = {
        row["leakage_group_id"]: row
        for row in plan["guard_purge_rows"]
        if row["fold_id"] == 2
    }
    detail = purged["unlabeled_group_0_8"]
    assert detail["boundary_interval_guard_intersection"] is False
    assert detail["held_candidate_euclidean_guard_intersection"] is True
    assert detail["minimum_distance_to_held_test_candidate_px"] == 10.0
    assert fold_two["minimum_retained_train_test_candidate_distance_px"] > 12.0

def test_proposal_miss_denominator_excludes_review_reserved_miss() -> None:
    metrics = {"positive_count": 78, "positive_recovered_at_budget": 40}
    unchanged = runner.adjust_recall_for_proposal_misses(metrics, 0)
    adjusted = runner.adjust_recall_for_proposal_misses(metrics, 1)
    assert unchanged["known_positive_proposal_denominator"] == 78
    assert unchanged["known_positive_proposal_recall_at_budget"] == pytest.approx(40 / 78)
    assert adjusted["known_positive_proposal_denominator"] == 79
    assert adjusted["known_positive_proposal_recall_at_budget"] == pytest.approx(40 / 79)

    canonical = [
        {"observation_id": "matched", "include_confirmed": "true", "burst_id": "1", "canonical_roi_id": "roi_1", "x_px": "10", "y_px": "10"},
        {"observation_id": "reserved_miss", "include_confirmed": "true", "burst_id": "4", "canonical_roi_id": "roi_2", "x_px": "100", "y_px": "100"},
        {"observation_id": "unreserved_miss", "include_confirmed": "true", "burst_id": "2", "canonical_roi_id": "roi_3", "x_px": "300", "y_px": "300"},
    ]
    harmonized = [{"canonical_v7": {"observation_id": "matched"}}]
    review_sites = [{"detection_site_id": "dsite_1", "x_px": "103", "y_px": "102"}]
    misses = runner.classify_proposal_stage_misses(canonical, harmonized, review_sites)
    assert misses["total_unmatched_confirmed"] == 2
    assert misses["reserved_unmatched_confirmed"] == 1
    assert misses["unreserved_unmatched_confirmed"] == 1
    roles = {row["observation_id"]: row["primary_denominator_role"] for row in misses["rows"]}
    assert roles == {
        "reserved_miss": "excluded_review_reserve",
        "unreserved_miss": "unrecovered_unreserved_positive",
    }


def test_macro_fold_group_bootstrap_targets_primary_estimand() -> None:
    positive = [True, True, False, False, False] * 2
    folds = [1] * 5 + [2] * 5
    leakage = ["p1", "p2", "p1", "u1", "u2", "p3", "p4", "p3", "u3", "u4"]
    score_a = [0.9, 0.8, 0.2, 0.1, 0.05, 0.95, 0.75, 0.3, 0.2, 0.05]
    score_b = [0.1, 0.2, 0.8, 0.9, 0.95, 0.05, 0.3, 0.75, 0.8, 0.95]
    result = runner.grouped_paired_bootstrap_macro_fold_auc(
        positive,
        score_a,
        score_b,
        folds,
        leakage,
        draws=50,
        seed=17,
        score_a_name="good",
        score_b_name="bad",
    )
    assert result["observed_score_a_macro_fold_spu_auc"] == 1.0
    assert result["observed_score_b_macro_fold_spu_auc"] == 0.0
    assert result["observed_delta_a_minus_b"] == 1.0
    assert result["dependency_components_split"] is False
    assert result["resampling_unit"].startswith("whole_leakage_group")
    assert [
        (
            row["positive_containing_leakage_component_count"],
            row["unlabeled_only_leakage_component_count"],
            row["unlabeled_rows_inside_positive_containing_components"],
        )
        for row in result["observed_by_fold"]
    ] == [(2, 2, 1), (2, 2, 1)]


def test_representative_unit_null_is_size_safe_and_nonrepresentative_invariant() -> None:
    positive = [True, True, False, False, False, False] * 2
    sample_ids = ["a", "b", "c", "d", "e", "k", "f", "g", "h", "i", "j", "l"]
    folds = [1] * 6 + [2] * 6
    leakage = ["p1", "p2", "p1", "u1", "u2", "u2", "p3", "p4", "p3", "u3", "u4", "u4"]
    scores = [0.5] * 12
    tied = runner.fold_stratified_representative_unit_score_permutation_null(
        positive,
        scores,
        sample_ids,
        folds,
        leakage,
        draws=50,
        seed=23,
    )
    assert tied["observed_macro_fold_representative_unit_spu_auc"] == 0.5
    assert tied["greater_or_equal_p_value"] == 1.0
    assert tied["positive_containing_leakage_component_count"] == 4
    assert tied["unlabeled_only_leakage_component_count"] == 4
    assert tied["dependency_components_split"] is False

    changed = list(scores)
    for index in (2, 5, 8, 11):
        changed[index] = 1000.0 + index
    invariant = runner.fold_stratified_representative_unit_score_permutation_null(
        positive,
        changed,
        sample_ids,
        folds,
        leakage,
        draws=50,
        seed=23,
    )
    assert invariant["observed_macro_fold_representative_unit_spu_auc"] == tied[
        "observed_macro_fold_representative_unit_spu_auc"
    ]
    assert invariant["greater_or_equal_p_value"] == tied["greater_or_equal_p_value"]
    assert [row["score"] for row in invariant["representatives"]] == [
        row["score"] for row in tied["representatives"]
    ]


def test_review_audit_is_site_level_with_overlap_and_conflict() -> None:
    sites = [
        {"detection_site_id": "dsite_1", "x_px": "10", "y_px": "10"},
        {"detection_site_id": "dsite_2", "x_px": "50", "y_px": "50"},
    ]
    occurrences = [
        {"detection_site_id": "dsite_1", "burst_id": "1", "x_px": "10", "y_px": "10"},
        {"detection_site_id": "dsite_2", "burst_id": "1", "x_px": "50", "y_px": "50"},
    ]
    reviews = [
        {"detection_site_id": "dsite_1", "normalized_label": "artifact_or_noise", "confidence_1_to_5": "4"},
        {"detection_site_id": "dsite_2", "normalized_label": "definite_neuron", "confidence_1_to_5": "5"},
    ]
    canonical = [
        {"observation_id": "b01__roi_1", "burst_id": "1", "x_px": "11", "y_px": "10", "include_confirmed": "true"}
    ]
    candidates = [
        {"review_site_id": "dsite_1", "tiny_mlp_final_fit_score": 0.75},
        {"review_site_id": "dsite_1", "tiny_mlp_final_fit_score": 0.25},
    ]
    result = runner.aggregate_review_site_audit(
        sites, occurrences, reviews, canonical, candidates
    )
    by_site = {row["review_site_id"]: row for row in result}
    assert by_site["dsite_1"]["canonical_v7_confirmed_overlap"] is True
    assert by_site["dsite_1"]["cross_source_disagreement"] is True
    assert by_site["dsite_1"]["reserved_candidate_score_median"] == 0.5
    assert by_site["dsite_2"]["reserved_candidate_count"] == 0


def test_atomic_finalization_indexes_complete_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    partial = tmp_path / "run.partial"
    _write(partial / "summary.json", "{}\n")
    _write(partial / "tables" / "scores.tsv", "id\tscore\na\t1\n")
    _write(partial / "census" / "validation.json", "{}\n")
    index = runner.finalize_atomic_run_tree(partial, output)
    assert not partial.exists()
    assert output.is_dir()
    assert {row["path"] for row in index["artifacts"]} == {
        "summary.json",
        "tables/scores.tsv",
        "census/validation.json",
    }
    assert json.loads((output / "artifact_index.json").read_text(encoding="utf-8")) == runner._artifact_index(output)
    with pytest.raises(FileExistsError):
        runner.finalize_atomic_run_tree(output.with_name("run.partial"), output)

    failed_output = tmp_path / "failed_run"
    failed_partial = tmp_path / "failed_run.partial"
    _write(failed_partial / "summary.json", "{}\n")
    monkeypatch.setattr(
        runner,
        "write_and_validate_artifact_index",
        lambda _root: (_ for _ in ()).throw(
            runner.LearningContractError("injected pre-rename validation failure")
        ),
    )
    with pytest.raises(runner.LearningContractError, match="injected pre-rename"):
        runner.finalize_atomic_run_tree(failed_partial, failed_output)
    assert failed_partial.is_dir()
    assert not failed_output.exists()


def test_public_runner_promotes_only_after_all_engineering_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, data, config_path = _install_public_run_smoke_mocks(
        tmp_path / "success", monkeypatch
    )
    result = runner.run_uncertainty_aware_learning(
        config_path, repository_root=repository, data_root=data
    )
    output = repository / runner.OUTPUT_RELATIVE_PATH
    partial = output.with_name(output.name + ".partial")
    assert output.is_dir()
    assert not partial.exists()
    assert result["status"] == "complete_non_claim_bearing_engineering_screen"
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert validation["passed_engineering_execution"] is True
    assert validation["scientific_completion"] is False
    assert validation["claim_promotion_allowed"] is False
    assert (output / "tables" / "fold_summary.tsv").is_file()
    provenance = json.loads(
        (output / "execution_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["execution"]["command"] == [
        "python",
        "-m",
        runner.RUNNER_MODULE,
        "run",
        "--config",
        runner.CONFIG_RELATIVE_PATH.as_posix(),
        "--repository-root",
        ".",
        "--data-root",
        "$NEUROBENCH_DATA_ROOT",
    ]
    assert provenance["execution"]["shell_replay_command"] == (
        f"python -m {runner.RUNNER_MODULE} run "
        f"--config {runner.CONFIG_RELATIVE_PATH.as_posix()} "
        '--repository-root . --data-root "$NEUROBENCH_DATA_ROOT"'
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["tiny_mlp_cluster_bootstrap_contrasts"]) == {
        "linear_spu",
        "carrier_signed",
        "cfar_score",
        "expert_separation_equal_weight",
    }
    for comparator in (
        "carrier_signed",
        "cfar_score",
        "expert_separation_equal_weight",
    ):
        assert summary["tiny_mlp_cluster_bootstrap_contrasts"][comparator] == {
            "tiny_mlp_minus_comparator_observed_delta": 0.0,
            "cluster_bootstrap_ci95_low": 0.0,
            "cluster_bootstrap_ci95_high": 0.0,
            "resampling_unit": "whole_leakage_group_with_all_positive_and_unlabeled_members_joint",
            "interval_interpretation": "descriptive_cluster_bootstrap_interval_non_claim_bearing",
        }
    assert summary["positive_containing_leakage_components_by_fold"] == [9, 3, 3, 2, 4]
    assert summary["bootstrap_interval_support_caveat"] == (
        "descriptive_only_due_to_2_or_3_positive_containing_leakage_components_in_some_folds"
    )
    assert not any(
        "spatial_group_score_association_null" in path.read_text(encoding="utf-8")
        for path in output.rglob("*.json")
    )
    assert json.loads((output / "artifact_index.json").read_text(encoding="utf-8")) == runner._artifact_index(output)

    repository, data, config_path = _install_public_run_smoke_mocks(
        tmp_path / "failure", monkeypatch, fail_thread_cap=True
    )
    with pytest.raises(runner.LearningContractError, match="numeric_thread_cap_observed"):
        runner.run_uncertainty_aware_learning(
            config_path, repository_root=repository, data_root=data
        )
    output = repository / runner.OUTPUT_RELATIVE_PATH
    partial = output.with_name(output.name + ".partial")
    assert not output.exists()
    assert partial.is_dir()
    failed = json.loads(
        (partial / "validation.failed.json").read_text(encoding="utf-8")
    )
    status = json.loads((partial / "status.json").read_text(encoding="utf-8"))
    assert failed["passed_engineering_execution"] is False
    assert "numeric_thread_cap_observed" in failed["failed_engineering_checks"]
    assert status["status"] == "failed_incomplete_partial"
    assert status["scientific_completion"] is False
    assert status["claim_promotion_allowed"] is False

    repository, data, config_path = _install_public_run_smoke_mocks(
        tmp_path / "restore_failure",
        monkeypatch,
        fail_thread_restore=True,
    )
    with pytest.raises(RuntimeError, match="thread-limit restore failure"):
        runner.run_uncertainty_aware_learning(
            config_path, repository_root=repository, data_root=data
        )
    output = repository / runner.OUTPUT_RELATIVE_PATH
    partial = output.with_name(output.name + ".partial")
    assert not output.exists()
    assert partial.is_dir()
    status = json.loads((partial / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed_incomplete_partial"

    repository, data, config_path = _install_public_run_smoke_mocks(
        tmp_path / "deadline",
        monkeypatch,
        raise_deadline_during_census=True,
    )
    with pytest.raises(runner.LearningDeadlineExceeded, match="deadline exceeded"):
        runner.run_uncertainty_aware_learning(
            config_path, repository_root=repository, data_root=data
        )
    output = repository / runner.OUTPUT_RELATIVE_PATH
    partial = output.with_name(output.name + ".partial")
    assert not output.exists()
    assert partial.is_dir()
    status = json.loads((partial / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed_incomplete_partial"
    assert status["error_type"] == "LearningDeadlineExceeded"
