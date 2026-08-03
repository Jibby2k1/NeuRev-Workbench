from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_population_evaluation import (
    evaluate_population_generated_matrix,
)


def test_population_gate_improves_attribution_but_blocks_subgroup_failure():
    result = evaluate_population_generated_matrix(seeds=(7,))
    assert result["gates"]["C1_numerical_reconstruction"] == "pass"
    assert result["gates"]["C2_generated_attribution"] == "pass"
    assert result["summary"]["relative_signal_leakage_improvement"] >= 0.05
    assert result["summary"]["C3_aggregate_pass"]
    assert not result["summary"]["C3_subgroup_pass"]
    assert result["gates"]["C3_signal_preservation"] == "fail"
    assert not result["advance_to_W6"]
