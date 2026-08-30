import json
from pathlib import Path

from neurobench.experiments.neuron_identifiability.targeted_automated_development import run_targeted_automated_development


def test_targeted_package_has_three_components(tmp_path:Path)->None:
    result=run_targeted_automated_development(tmp_path/"v8",empirical_profile={"simulator_read_sigma":.3},seeds=1)
    validation=json.loads((tmp_path/"v8"/"validation.json").read_text())
    assert validation["status"] == "passed"
    assert set(result["components"]) == {"parameter_consensus_stopping","close_source_separation","directional_kinetics"}
    assert validation["tables"] == 3
