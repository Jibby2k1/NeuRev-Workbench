import json
from pathlib import Path

from neurobench.experiments.neuron_identifiability.automated_challenge_suite import FAMILIES,run_automated_challenge_suite


def test_all_twelve_challenge_families_are_materialized(tmp_path: Path) -> None:
    result=run_automated_challenge_suite(tmp_path/"suite",empirical_profile={"simulator_read_sigma":.3},seeds=1)
    validation=json.loads((tmp_path/"suite"/"validation.json").read_text())
    assert validation["status"] == "passed"
    assert result["families"] == list(FAMILIES)
    assert validation["families"] == 12
    assert all((tmp_path/"suite"/f"{name}.tsv").is_file() for name in FAMILIES)
