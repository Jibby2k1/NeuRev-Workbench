from __future__ import annotations

import pytest

from neurobench.metrics.truth_set import CoverageAuthorizationError, authorize_metric, exhaustive_event_metrics, exhaustive_event_pr_curve, exhaustive_object_metrics, exhaustive_object_pr_curve, known_positive_recall_at_k, review_efficiency


@pytest.mark.parametrize("coverage", ["candidate_assisted", "sparse_positive"])
@pytest.mark.parametrize("metric", ["object_precision", "object_ap", "event_precision", "event_ap", "false_events_per_field_minute"])
def test_precision_metrics_reject_non_exhaustive_coverage(coverage: str, metric: str) -> None:
    with pytest.raises(CoverageAuthorizationError, match="requires exhaustive"):
        authorize_metric(coverage, metric)


def test_known_positive_recall_at_k_remains_available_for_sparse_positive() -> None:
    result = known_positive_recall_at_k(["a", "b", "c"], ["b", "x", "a"], k=2, coverage_mode="sparse_positive")
    assert result["recall_at_k"] == pytest.approx(1 / 3)


def test_unresolved_objects_are_not_counted_as_negatives() -> None:
    truth = [{"id": "n", "disposition": "neuron", "x": 1, "y": 1}, {"id": "u", "disposition": "unresolved", "x": 8, "y": 8}]
    result = exhaustive_object_metrics(truth, [{"id": "p", "x": 1, "y": 1}], coverage_mode="exhaustive", centroid_tolerance_px=1)
    assert result["object_count_gt"] == 1
    assert result["unresolved_excluded_count"] == 1
    assert result["object_precision"] == 1.0


def test_unresolved_events_are_not_counted_as_negatives() -> None:
    truth = [{"event_id": "e", "object_id": "o", "frame": 3, "disposition": "event"}, {"event_id": "u", "object_id": "o", "frame": 8, "disposition": "unresolved"}]
    result = exhaustive_event_metrics(truth, [{"event_id": "p", "object_id": "o", "frame": 3}], coverage_mode="exhaustive")
    assert result["event_count_gt"] == 1 and result["unresolved_excluded_count"] == 1


def test_review_efficiency_reports_time_and_unresolved_fraction() -> None:
    result = review_efficiency(review_minutes=30, accepted_count=6, unresolved_count=2, total_count=10)
    assert result["review_minutes_per_accepted"] == 5
    assert result["unresolved_fraction"] == 0.2


def test_exhaustive_object_and_event_pr_curves_compute_ap() -> None:
    objects = [{"id": "n", "disposition": "neuron", "x": 1, "y": 1}]
    object_candidates = [{"id": "p", "x": 1, "y": 1, "score": 0.9}, {"id": "fp", "x": 9, "y": 9, "score": 0.1}]
    object_result = exhaustive_object_pr_curve(objects, object_candidates, coverage_mode="exhaustive", centroid_tolerance_px=1)
    events = [{"event_id": "e", "object_id": "o", "frame": 3, "disposition": "event"}]
    event_candidates = [{"event_id": "p", "object_id": "o", "frame": 3, "score": 0.8}]
    event_result = exhaustive_event_pr_curve(events, event_candidates, coverage_mode="exhaustive")
    assert object_result["average_precision"] == 1.0
    assert event_result["average_precision"] == 1.0
