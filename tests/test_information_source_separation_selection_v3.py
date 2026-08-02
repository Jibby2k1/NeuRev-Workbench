from neurobench.experiments.information_source_separation.selection_v3 import (
    select_finalists_with_equivalence_margin,
)


def _row(rank: int, case: str, correlation: float) -> dict:
    return {
        "method_id": "pca_reference",
        "configuration_json": f'{{"rank": {rank}}}',
        "converged": True,
        "mean_absolute_correlation": correlation,
        "worst_absolute_correlation": correlation,
        "mean_absolute_crosstalk": 1.0 - correlation,
        "unresolved_expected": case == "unresolved",
        "reported_unresolved": case == "unresolved",
    }


def test_equivalent_recovery_prefers_lower_rank() -> None:
    rows = [
        _row(4, "isolated", 0.795), _row(4, "unresolved", 0.1),
        _row(8, "isolated", 0.800), _row(8, "unresolved", 0.9),
    ]
    result = select_finalists_with_equivalence_margin(
        rows, expected_fixture_count=2, equivalence_margin=0.01
    )
    assert result["selected_finalists"][0]["complexity_rank"] == 4
