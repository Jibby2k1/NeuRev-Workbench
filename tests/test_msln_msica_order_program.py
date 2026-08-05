import json
from pathlib import Path

import numpy as np

from neurobench.experiments.msln_msica.order_program import (
    REFERENCE_CONTEXTS,
    _objective_gain,
    _selection_score,
    _validate,
)


class _Fit:
    baseline_objective_value = 2.0
    objective_value = 1.5


def test_objective_gain_is_relative_to_analytic_baseline():
    assert _objective_gain(_Fit()) == 0.25


def test_selection_score_excludes_recall_fields():
    metrics = {"visual_stats": {"event_quiet_ratio_p999": 3.0}, "recall_guardrail": {"matched_by_budget": {"58": 79}}}
    synthetic = {"signal_to_nuisance_proxy": 2.0}
    expected = np.log1p(3.0) + np.log1p(2.0)
    assert np.isclose(_selection_score(metrics, synthetic), expected)


def test_example_freezes_reference_contexts_and_cuda_contract():
    path = Path("examples/spon_ca_burst_msln_msica_order_program_v3.example.json")
    config = json.loads(path.read_text())
    config["_config_path"] = str(path.resolve())
    _validate(config)
    assert tuple(config["sweep"]["forced_contexts"]) == REFERENCE_CONTEXTS
    assert config["compute"]["device"] == "cuda"
