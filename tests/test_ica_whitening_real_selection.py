from neurobench.experiments.ica_whitening_evaluation.real_selection import (
    nested_leave_one_burst_out, paired_seed_groups, pareto_front,
)
from neurobench.experiments.ica_whitening_evaluation.real_finalist_confirmation import (
    confirmation_fit_ids_by_held_out_burst, select_confirmation_fit_ids,
    summarize_protected_refits,
)


def _row(fit_id, recalls, condition=2.0, seed=1, scale=1.0):
    return {
        "fit_id": fit_id, "family": "temporal", "seed": seed,
        "objective_scale": scale, "real_data_condition_number": condition, "rank": 2,
        "label_metrics": {"folds": [
            {"burst_id": burst, "budgets": [{"budget": 58, "recall": recall}]}
            for burst, recall in enumerate(recalls, start=1)
        ]},
    }


def test_pareto_front_removes_strictly_dominated_rows():
    rows = [
        {"fit_id": "a", "utility": .8, "cost": 2},
        {"fit_id": "b", "utility": .7, "cost": 3},
        {"fit_id": "c", "utility": .9, "cost": 5},
    ]
    assert [row["fit_id"] for row in pareto_front(
        rows, maximize=("utility",), minimize=("cost",)
    )] == ["a", "c"]


def test_nested_loo_never_selects_on_held_out_burst():
    rows = [
        _row("balanced", [.6, .6, .6, .6], condition=1),
        _row("burst4_only", [0, 0, 0, 1], condition=1),
    ]
    result = nested_leave_one_burst_out(rows, finalists_per_training_fold=1)
    fold4 = next(fold for fold in result["folds"] if fold["held_out_burst"] == 4)
    assert fold4["selected_fit_ids"] == ["balanced"]
    assert result["unmatched_candidates"] == "unknown_not_negative"


def test_nested_loo_pareto_uses_training_recall_and_label_independent_rank():
    high = _row("high_recall", [.8, .8, .8, 0]); high["rank"] = 5
    compact = _row("compact", [.7, .7, .7, 1]); compact["rank"] = 2
    result = nested_leave_one_burst_out(
        [high, compact], finalists_per_training_fold=2,
    )
    fold4 = next(fold for fold in result["folds"] if fold["held_out_burst"] == 4)
    assert fold4["selected_fit_ids"] == ["high_recall", "compact"]
    assert fold4["training_pareto_candidate_count"] == 2
    assert fold4["pareto_objectives"]["minimize"] == ["ICA_rank_label_independent_complexity"]


def test_seed_groups_require_same_nonseed_hyperparameters():
    rows = [
        _row("a", [.5] * 4, seed=1, scale=.5),
        _row("b", [.5] * 4, seed=2, scale=.5),
        _row("c", [.5] * 4, seed=3, scale=.7),
    ]
    groups = paired_seed_groups(rows)
    assert len(groups) == 1
    assert [row["fit_id"] for row in groups[0]] == ["a", "b"]


def test_confirmation_rule_prefers_repeated_loo_pareto_members():
    summary = {
        "preliminary_pareto_fit_ids": ["a", "b", "d"],
        "protected_leave_one_burst_out": {
            "selection_frequency": {"a": 3, "b": 1, "c": 4}
        },
    }
    assert select_confirmation_fit_ids(summary, maximum=3) == ["a", "c", "b"]


def test_confirmation_uses_only_fold_specific_protected_choices():
    summary = {"protected_leave_one_burst_out": {"folds": [
        {"held_out_burst": burst, "selected_fit_ids": [f"safe_{burst}", "shared"]}
        for burst in (1, 2, 3, 4)
    ]}}
    assert confirmation_fit_ids_by_held_out_burst(summary, maximum_per_fold=1) == {
        1: ["safe_1"], 2: ["safe_2"], 3: ["safe_3"], 4: ["safe_4"],
    }


def test_protected_refit_summary_groups_by_burst_not_seed():
    payloads = []
    for burst in (1, 2, 3, 4):
        for seed in (1, 2):
            recall = burst / 10 + seed / 100
            fold = {"burst_id": burst, "budgets": [{"budget": 58, "recall": recall}]}
            controls = [{
                "control_id": control_id,
                "label_metrics": {"folds": [fold]},
            } for control_id in ("rank_matched_pca", "rank_matched_random_rotation")]
            payloads.append({
                "source_fit_id": f"f{burst}", "confirmation_seed": seed,
                "held_out_burst": burst, "held_out_label_metrics": fold,
                "rank_matched_controls": {"controls": controls},
            })
    result = summarize_protected_refits(payloads, bootstrap_replicates=128)
    assert len(result["burst_rows"]) == 4
    assert result["burst_rows"][0]["robustness_run_count"] == 2
    assert result["grouping_unit"] == "held_out_burst"
    assert "not independent biological replicates" in result["robustness_run_semantics"]
