from neurobench.experiments.ica_whitening_evaluation.analysis import analyze_rows


def test_analysis_marks_formal_sobol_indices_unavailable():
    rows = []
    for seed in (1, 2, 3):
        for index in range(8):
            rows.append({
                "fit_id": f"f{seed}-{index}", "cell_id": "c", "point_id": f"p{index}",
                "seed": seed, "family": "temporal", "objective": "logcosh",
                "whitening_geometry": "temporal", "covariance_scope": "global",
                "causality": "causal", "rank": 2, "spatial_width_px": None,
                "temporal_width_frames": 3, "point_kind": "sobol",
                "temporal_exponent": index / 8, "objective_scale": index + 1,
                "mean_truth_source_correlation": index / 10,
                "mean_truth_crosstalk": .2, "mean_trace_preservation": .9,
                "converged_fraction": 1., "unresolved_accuracy": 1., "cases": [],
            })
    result = analyze_rows(rows)
    assert result["fit_count"] == 24
    assert result["formal_sobol_indices"]["status"] == "not_identifiable_from_this_design"
    assert result["within_cell_continuous_associations"]
