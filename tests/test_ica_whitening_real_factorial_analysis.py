from neurobench.experiments.ica_whitening_evaluation.real_factorial_analysis import (
    _render_learned_parameter_figures, summarize_factorial,
)


def _row(index):
    recall = .2 + .1 * index
    return {
        "fit_id": f"f{index}", "family": "temporal" if index % 2 else "spatial",
        "cell_id": f"cell_{index % 2}", "point_kind": "sobol",
        "whitening_geometry": "none", "covariance_scope": "global_quiet",
        "causality": "centered", "objective": "fastica_logcosh", "rank": 2,
        "spatial_width_px": 3, "temporal_width_frames": 3,
        "objective_scale": .1 + index / 10, "raw_preserving_blend": .5,
        "seed": index % 2, "real_data_fit_converged": True,
        "real_data_condition_number": 2 + index,
        "component_response": {
            "temporal_frequency_centroid_hz": 2.0 + index,
            "temporal_frequency_bandwidth_hz": 1.0,
            "spatial_frequency_centroid_cycles_per_px": .1,
            "spatial_frequency_bandwidth_cycles_per_px": .05,
            "separability_residual_fraction": .2,
            "best_rank1_separable_energy_fraction": .8,
            "kernel_l2": 1.5,
            "kernel_sum": .25,
        },
        "external_whitening_diagnostics": {
            "resolved": True, "maximum_condition_number": 4.0 + index,
            "minimum_effective_rank_fraction": .75,
        },
        "label_metrics": {
            "macro_known_positive_recall": recall,
            "mean_reciprocal_rank": recall / 2,
            "folds": [{"burst_id": burst, "budgets": [{"budget": 58, "recall": recall}]}
                      for burst in (1, 2, 3, 4)],
        },
    }


def test_factorial_summary_is_complete_and_claim_bounded():
    result = summarize_factorial([_row(index) for index in range(8)])
    assert result["fit_count"] == result["unique_fit_count"] == 8
    assert result["finalist_eligible_count"] == 8
    assert result["formal_sobol_indices_identified"] is False
    assert result["continuous_sensitivity"][0]["semantics"].endswith("not_formal_sobol_index")
    learned = result["learned_parameter_summary"]
    assert learned["response_by_family_and_rank"]
    assert learned["response_by_family_and_temporal_width"]
    assert learned["response_by_family_and_spatial_width"]
    assert learned["response_by_family_and_whitening_geometry"]
    assert learned["whitening_by_geometry_scope_support_and_blend"]
    temporal = learned["response_by_family_and_rank"][0]["metrics"][
        "temporal_frequency_centroid_hz"
    ]
    assert temporal["count"] == 4
    assert temporal["q05"] <= temporal["median"] <= temporal["q95"]
    assert result["conditional_factor_summaries"]["stratification"] == [
        "family", "whitening_geometry"
    ]
    assert result["design_audit"]["categorical_ordinal_policy"].startswith("exact_cartesian")
    assert result["design_audit"]["formal_sobol_indices_identified"] is False
    assert result["claim_scope"].endswith("unmatched_unknown")


def test_factorial_summary_rejects_duplicate_fit_ids():
    row = _row(0)
    try:
        summarize_factorial([row, row])
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate IDs must fail")


def test_learned_parameter_figures_render(tmp_path):
    paths = _render_learned_parameter_figures(
        tmp_path, [_row(index) for index in range(8)]
    )
    assert paths == [
        "figures/learned_parameters/frequency_response_vs_rank.png",
        "figures/learned_parameters/frequency_response_vs_support.png",
        "figures/learned_parameters/whitening_diagnostics_by_geometry.png",
    ]
    assert all((tmp_path / path).stat().st_size > 1000 for path in paths)
