from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]

def test_ica_suite_separates_temporal_spatial_and_tensor_claims():
    suite=yaml.safe_load((ROOT/"paper/ICA_STATISTICAL_TEST_SUITE.yaml").read_text())
    families=suite["representation_families"]
    assert set(families)=={"temporal_two_frame_ica","spatial_dense_fastica_wiener","tensor_factor_ica"}
    assert families["temporal_two_frame_ica"]["prohibited_interpretation"]=="independent_neural_source"
    assert families["spatial_dense_fastica_wiener"]["primary_role"]=="spatial_morphology_and_boundary_visibility"
    assert families["tensor_factor_ica"]["individual_factor_interpretation"]=="prohibited_due_lobo_instability"

def test_ica_morphology_gate_requires_panel_provenance_and_blinded_ablation():
    suite=yaml.safe_load((ROOT/"paper/ICA_STATISTICAL_TEST_SUITE.yaml").read_text())
    assert suite["gates"]["provenance"]["require_exact_displayed_panel_for_each_review"]
    assert suite["gates"]["provenance"]["do_not_relabel_positive_change_as_ica"]
    assert suite["gates"]["morphology_promotion"]["require_blinded_raw_ica_combined_ablation"]
    assert suite["gates"]["interpretation"]["visual_utility_does_not_require_native_signal_fidelity"]

def test_spatial_checkpoint_remains_exploratory_despite_useful_metrics():
    suite=yaml.safe_load((ROOT/"paper/ICA_STATISTICAL_TEST_SUITE.yaml").read_text())
    checkpoint=suite["existing_spatial_screen_checkpoint"]
    assert checkpoint["fixed_budget_recall"] > .65
    assert checkpoint["median_peak_retention"] > .95
    assert checkpoint["audit_pass"] is False
