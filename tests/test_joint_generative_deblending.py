import json
from pathlib import Path

from neurobench.experiments.neuron_identifiability.joint_generative_deblending import FROZEN_CONFIG,run_joint_generative_program


def test_joint_program_preserves_locked_boundary(tmp_path:Path)->None:
    result=run_joint_generative_program(tmp_path/"v9",empirical_profile={"simulator_read_sigma":.3},seeds=1)
    validation=json.loads((tmp_path/"v9"/"validation.json").read_text())
    assert validation["status"] == "passed"
    assert validation["locked_used_for_tuning"] is False
    assert result["frozen_config"]["components"] == FROZEN_CONFIG.components
    assert set(result["development"]) == {"spatial_context","constrained_generative"}
