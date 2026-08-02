from neurobench.experiments.information_source_separation.calibration_config import CalibrationConfig


def test_calibration_manifest_has_disjoint_cases_seeds_and_bounded_count() -> None:
    config = CalibrationConfig.load("examples/spon_ca_burst_identifiability_calibration_v1.example.json")
    assert config.split_count("calibration") == 30
    assert config.split_count("evaluation") == 45
    assert len(config.methods) == 4
    assert not set(config.calibration["case_ids"]) & set(config.evaluation["case_ids"])
