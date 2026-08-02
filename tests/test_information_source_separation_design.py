import json

from neurobench.experiments.information_source_separation.config import (
    InformationSeparationConfig,
)
from neurobench.experiments.information_source_separation.design import (
    method_screen,
    screen_fixtures,
    select_finalists,
    staged_fit_counts,
)


def _config() -> InformationSeparationConfig:
    return InformationSeparationConfig.load(
        "examples/spon_ca_burst_information_source_separation_v1.example.json"
    )


def test_staged_design_replaces_full_cartesian_grid() -> None:
    config = _config()
    assert len(method_screen(config)) == 48
    assert len(screen_fixtures(config)) == 14
    counts = staged_fit_counts(config)
    assert counts["screen_fit_count"] == 672
    assert counts["maximum_finalist_configuration_count"] == 7
    assert counts["maximum_confirmation_fit_count"] == 1365
    assert counts["maximum_staged_fit_count"] == 2037
    assert counts["full_cartesian_fit_count"] == 9360


def test_selection_requires_unresolved_success_before_recovery_ranking() -> None:
    rows = []
    for configuration, correlation, unresolved in (
        ({"rank": 4}, 0.95, False),
        ({"rank": 8}, 0.85, True),
    ):
        for case in ("isolated", "unresolved"):
            rows.append({
                "method_id": "multilag_sobi",
                "configuration_json": json.dumps(configuration, sort_keys=True),
                "converged": True,
                "mean_absolute_correlation": correlation,
                "worst_absolute_correlation": correlation - 0.1,
                "mean_absolute_crosstalk": 1 - correlation,
                "unresolved_expected": case == "unresolved",
                "reported_unresolved": unresolved if case == "unresolved" else False,
            })
    selection = select_finalists(rows, expected_fixture_count=2)
    assert selection["status"] == "finalists_selected"
    assert json.loads(selection["selected_finalists"][0]["configuration_json"])["rank"] == 8
