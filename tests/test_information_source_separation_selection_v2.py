from neurobench.experiments.information_source_separation.selection_v2 import (
    select_finalists_resolvable_only,
)


def _row(configuration: str, case: str, correlation: float) -> dict:
    return {
        "method_id": "multilag_sobi",
        "configuration_json": configuration,
        "converged": True,
        "mean_absolute_correlation": correlation,
        "worst_absolute_correlation": correlation,
        "mean_absolute_crosstalk": 1.0 - correlation,
        "unresolved_expected": case == "unresolved",
        "reported_unresolved": case == "unresolved",
    }


def test_unresolved_recovery_cannot_change_resolvable_ranking() -> None:
    rows = [
        _row('{"rank": 4}', "isolated", 0.80),
        _row('{"rank": 4}', "unresolved", 0.01),
        _row('{"rank": 8}', "isolated", 0.70),
        _row('{"rank": 8}', "unresolved", 0.99),
    ]
    result = select_finalists_resolvable_only(
        rows, expected_fixture_count=2, finalists_per_new_method=1
    )
    selected = result["selected_finalists"][0]
    assert selected["configuration_json"] == '{"rank": 4}'
    assert selected["mean_absolute_correlation"] == 0.80
    assert selected["recovery_fit_count"] == 1
