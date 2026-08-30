from neurobench.experiments.neuron_identifiability.nms_sensitivity import sensitivity_gate


def _row(macro,bursts): return {"macro_recall":{"20":macro},"folds":[{"budgets":{"20":{"recall":x}}} for x in bursts]}


def test_nms_gate_requires_positive_aggregate_and_three_safe_bursts():
    result=sensitivity_gate(_row(.55,[.6,.5,.5,.6]),_row(.5,[.5,.5,.5,.5]))
    assert result["passed"] is True
    assert result["nonnegative_bursts"]==4
