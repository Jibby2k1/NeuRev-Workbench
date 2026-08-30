from neurobench.experiments.neuron_identifiability.representation_confirmation import classify_native


def _row(macro, bursts):
    return {"macro_recall": {"20": macro}, "folds": [{"budgets": {"20": {"recall": value}}} for value in bursts]}


def test_c3_candidate_requires_aggregate_and_burst_guards():
    result = classify_native(_row(.61, [.60, .55, .62, .65]), _row(.54, [.53, .45, .57, .61]))
    assert result["candidate_level"] == "C3_candidate_pending_nms"
    assert result["nonnegative_bursts"] == 4
    assert result["catastrophic_drop"] is False

    confirmed = classify_native(
        _row(.61, [.60, .55, .62, .65]),
        _row(.54, [.53, .45, .57, .61]),
        nms_confirmed=True,
    )
    assert confirmed["candidate_level"] == "C3_confirmed_compact_utility"


def test_catastrophic_burst_drop_prevents_c3_candidate():
    result = classify_native(_row(.58, [.70, .39, .70, .70]), _row(.54, [.53, .45, .57, .61]))
    assert result["candidate_level"] != "C3_candidate_pending_nms"
    assert result["catastrophic_drop"] is True
