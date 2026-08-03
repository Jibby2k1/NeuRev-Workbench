from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_evaluation import (
    evaluate_generated_matrix,
)


def test_generated_w5_matrix_is_complete_and_gate_explicit():
    result = evaluate_generated_matrix(seeds=(7,))
    assert result["fixture_count"] == 15
    assert result["evaluation_count"] == 45
    assert result["gates"]["C1_numerical_reconstruction"] == "pass"
    assert result["gates"]["C4_residual_qualification"] == "not_qualified"
    assert isinstance(result["advance_to_real_scientific_run"], bool)
    for row in result["rows"]:
        assert set(row["lanes"]) == {
            "orthogonal_shared_private",
            "dependent_groups_only",
            "dependent_groups_joint_quiet",
        }
        assert all(lane["noise_name"] == "noise_candidate" for lane in row["lanes"].values())
