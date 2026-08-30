from __future__ import annotations

import copy

import pytest

from neurobench.experiments.neuron_identifiability.residual_safety import (
    JEPA_RESIDUAL_METHOD,
    RAW_METHOD,
    ResidualSafetyError,
    apply_residual_safety_policy,
    evaluate_residual_safety,
)


def _diagnostic(
    *,
    window: str,
    fixture: str,
    rms: float,
    dynamic: float,
    seam: float,
) -> dict[str, object]:
    return {
        "method": JEPA_RESIDUAL_METHOD,
        "background_recording_id": "recording_1",
        "background_window_id": window,
        "fixture_id": fixture,
        "background": {
            "background_rms_ratio": rms,
            "dynamic_mad_ratio": dynamic,
        },
        "seams": {"seam_to_interior_jump_ratio": seam},
    }


def _evaluation(*, fixture: str, window: str, method: str, recall: float) -> dict[str, object]:
    return {
        "fixture_id": fixture,
        "background_recording_id": "recording_1",
        "background_window_id": window,
        "method": method,
        "source_on_recovery": {
            "recall": recall,
            "recovered_sources": int(recall > 0),
            "injected_sources": 1,
        },
    }


def test_source_off_repetition_collapses_to_one_window_decision() -> None:
    rows = [
        _diagnostic(window="window_1", fixture="fixture_1", rms=0.8, dynamic=0.9, seam=1.1),
        _diagnostic(window="window_1", fixture="fixture_2", rms=0.8, dynamic=0.9, seam=1.1),
    ]

    result = evaluate_residual_safety(rows)

    assert len(result["decisions"]) == 1
    assert result["decisions"][0]["repeated_fixture_rows"] == 2
    assert result["decisions"][0]["safe_for_residual_use"] is True
    assert result["decisions"][0]["selected_endpoint"] == JEPA_RESIDUAL_METHOD
    assert result["decision_uses_source_truth_or_labels"] is False


def test_any_source_off_amplification_forces_raw_fallback() -> None:
    result = evaluate_residual_safety(
        [_diagnostic(window="window_1", fixture="fixture_1", rms=1.01, dynamic=0.9, seam=1.1)]
    )

    decision = result["decisions"][0]
    assert decision["safe_for_residual_use"] is False
    assert decision["selected_endpoint"] == RAW_METHOD
    assert decision["checks"]["background_rms_not_amplified"] is False
    assert result["methods"][JEPA_RESIDUAL_METHOD]["safe_window_fraction"] == 0.0


def test_repeated_source_off_drift_fails_closed() -> None:
    rows = [
        _diagnostic(window="window_1", fixture="fixture_1", rms=0.8, dynamic=0.9, seam=1.1),
        _diagnostic(window="window_1", fixture="fixture_2", rms=0.81, dynamic=0.9, seam=1.1),
    ]

    with pytest.raises(ResidualSafetyError, match="drift across repeated fixtures"):
        evaluate_residual_safety(rows)


def test_explicit_source_off_seam_is_used_instead_of_source_on_seam() -> None:
    first = _diagnostic(
        window="window_1", fixture="fixture_1", rms=0.8, dynamic=0.9, seam=4.0
    )
    second = _diagnostic(
        window="window_1", fixture="fixture_2", rms=0.8, dynamic=0.9, seam=5.0
    )
    for row in (first, second):
        row["decoder_reference"] = {
            "seams": {
                "signed_residual_off": {"boundary_to_within_jump_ratio": 1.1}
            }
        }

    result = evaluate_residual_safety([first, second])

    assert result["decisions"][0]["seam_to_interior_jump_ratio"] == 1.1
    assert result["decisions"][0]["safe_for_residual_use"] is True


def test_nonfinite_source_off_metric_fails_closed() -> None:
    row = _diagnostic(window="window_1", fixture="fixture_1", rms=0.8, dynamic=0.9, seam=1.1)
    row = copy.deepcopy(row)
    row["background"]["dynamic_mad_ratio"] = float("nan")

    with pytest.raises(ResidualSafetyError, match="must be finite"):
        evaluate_residual_safety([row])


def test_policy_uses_frozen_source_off_decision_before_recovery() -> None:
    diagnostics = [
        _diagnostic(window="safe_window", fixture="fixture_1", rms=0.8, dynamic=0.9, seam=1.1),
        _diagnostic(window="unsafe_window", fixture="fixture_2", rms=1.2, dynamic=0.9, seam=1.1),
    ]
    safety = evaluate_residual_safety(diagnostics)
    rows = [
        _evaluation(fixture="fixture_1", window="safe_window", method=RAW_METHOD, recall=0.0),
        _evaluation(fixture="fixture_1", window="safe_window", method=JEPA_RESIDUAL_METHOD, recall=1.0),
        _evaluation(fixture="fixture_2", window="unsafe_window", method=RAW_METHOD, recall=1.0),
        _evaluation(fixture="fixture_2", window="unsafe_window", method=JEPA_RESIDUAL_METHOD, recall=0.0),
    ]

    applied = apply_residual_safety_policy(
        rows,
        safety,
        residual_method=JEPA_RESIDUAL_METHOD,
    )

    assert applied["macro_source_on_recall"] == 1.0
    assert applied["micro_recovered_sources"] == 2
    assert applied["residual_admitted_fixture_count"] == 1
    assert applied["selection_uses_source_truth_or_labels"] is False


def test_incomplete_raw_residual_pair_fails_closed() -> None:
    safety = evaluate_residual_safety(
        [_diagnostic(window="window_1", fixture="fixture_1", rms=0.8, dynamic=0.9, seam=1.1)]
    )

    with pytest.raises(ResidualSafetyError, match="incomplete raw/residual pair"):
        apply_residual_safety_policy(
            [_evaluation(fixture="fixture_1", window="window_1", method=RAW_METHOD, recall=1.0)],
            safety,
            residual_method=JEPA_RESIDUAL_METHOD,
        )
