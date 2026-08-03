from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_confirmation_evaluation import (
    PRIMARY_LANE,
    evaluate_confirmation_generated_matrix,
)


def test_w5c_is_label_free_and_morphology_gate_is_authoritative():
    result = evaluate_confirmation_generated_matrix(seeds=(7,))
    assert result["quiet_calibration_only"]
    assert not result["labels_used_for_fit"]
    assert result["gates"]["C1_numerical_reconstruction"] == "pass"
    primary = result["lane_summaries"][PRIMARY_LANE]
    assert not primary["C3_subgroup_pass"]
    assert result["gates"]["C3_signal_preservation"] == "fail"
    assert not result["advance_to_W6"]
