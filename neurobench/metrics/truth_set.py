"""Coverage-authorized metrics for truth-set evaluation."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from neurobench.metrics.detection import object_matching_metrics
from neurobench.metrics.event_quality import event_timing_metrics


ALL_COVERAGE_METRICS = {
    "known_positive_recall", "recall_at_k", "candidate_efficiency", "reviewer_acceptance_at_k",
    "review_minutes_per_accepted", "unresolved_fraction", "reviewer_agreement", "candidate_stability",
}
EXHAUSTIVE_ONLY_METRICS = {
    "object_precision", "object_recall", "object_f1", "object_pr_curve", "object_ap",
    "event_precision", "event_recall", "event_ap", "precision_at_fixed_recall", "recall_at_fixed_precision",
    "false_events_per_field_minute", "centroid_distance", "footprint_iou", "onset_error", "peak_error",
    "duration_error", "duplicate_rate", "split_rate", "merge_rate", "calibration", "abstention",
}


class CoverageAuthorizationError(ValueError):
    """Raised when a precision-oriented metric lacks exhaustive coverage."""


def authorize_metric(coverage_mode: str, metric_name: str) -> None:
    if coverage_mode not in {"exhaustive", "candidate_assisted", "sparse_positive"}:
        raise ValueError(f"unknown coverage mode: {coverage_mode}")
    if metric_name in EXHAUSTIVE_ONLY_METRICS and coverage_mode != "exhaustive":
        raise CoverageAuthorizationError(
            f"metric '{metric_name}' requires exhaustive coverage; received {coverage_mode}"
        )
    if metric_name not in ALL_COVERAGE_METRICS | EXHAUSTIVE_ONLY_METRICS:
        raise ValueError(f"unknown truth-set metric: {metric_name}")


def known_positive_recall_at_k(
    known_positive_ids: Sequence[str], ranked_candidate_ids: Sequence[str], *, k: int, coverage_mode: str
) -> dict[str, Any]:
    authorize_metric(coverage_mode, "recall_at_k")
    positives = set(str(item) for item in known_positive_ids)
    selected = set(str(item) for item in ranked_candidate_ids[: int(k)])
    recovered = positives & selected
    return {"coverage_mode": coverage_mode, "k": int(k), "known_positive_count": len(positives), "recovered_count": len(recovered), "recall_at_k": len(recovered) / len(positives) if positives else 0.0}


def exhaustive_object_metrics(
    ground_truth: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]], *, coverage_mode: str, **kwargs: Any
) -> dict[str, Any]:
    authorize_metric(coverage_mode, "object_precision")
    truth = [item for item in ground_truth if item.get("disposition") == "neuron"]
    if any(item.get("disposition") == "unresolved" for item in ground_truth):
        unresolved = sum(item.get("disposition") == "unresolved" for item in ground_truth)
    else:
        unresolved = 0
    result = object_matching_metrics(truth, candidates, **kwargs)
    precision, recall = result["object_precision"], result["object_recall"]
    result.update({"coverage_mode": coverage_mode, "object_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "unresolved_excluded_count": unresolved})
    return result


def exhaustive_event_metrics(
    ground_truth: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]], *, coverage_mode: str, **kwargs: Any
) -> dict[str, Any]:
    authorize_metric(coverage_mode, "event_precision")
    truth = [item for item in ground_truth if item.get("disposition") == "event"]
    result = event_timing_metrics(truth, candidates, **kwargs)
    result.update({"coverage_mode": coverage_mode, "unresolved_excluded_count": sum(item.get("disposition") == "unresolved" for item in ground_truth)})
    return result


def review_efficiency(*, review_minutes: float, accepted_count: int, unresolved_count: int, total_count: int) -> dict[str, Any]:
    return {
        "review_minutes": float(review_minutes),
        "accepted_count": int(accepted_count),
        "review_minutes_per_accepted": float(review_minutes) / accepted_count if accepted_count else None,
        "unresolved_fraction": unresolved_count / total_count if total_count else 0.0,
    }


def _threshold_curve(
    ground_truth: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    evaluator: Any,
    precision_key: str,
    recall_key: str,
) -> dict[str, Any]:
    scores = sorted({float(item["score"]) for item in candidates}, reverse=True)
    points = []
    for threshold in scores:
        selected = [item for item in candidates if float(item["score"]) >= threshold]
        metrics = evaluator(ground_truth, selected)
        points.append({"threshold": threshold, "candidate_count": len(selected), "precision": metrics[precision_key], "recall": metrics[recall_key]})
    points.append({"threshold": None, "candidate_count": 0, "precision": 1.0, "recall": 0.0})
    ordered = sorted(points, key=lambda item: (item["recall"], -item["precision"]))
    ap = 0.0
    previous_recall = 0.0
    for point in ordered:
        recall = float(point["recall"])
        if recall > previous_recall:
            ap += (recall - previous_recall) * max(float(item["precision"]) for item in ordered if float(item["recall"]) >= recall)
            previous_recall = recall
    return {"points": points, "average_precision": ap}


def exhaustive_object_pr_curve(
    ground_truth: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *, coverage_mode: str,
    iou_threshold: float = 0.25,
    centroid_tolerance_px: float | None = None,
) -> dict[str, Any]:
    authorize_metric(coverage_mode, "object_ap")
    truth = [item for item in ground_truth if item.get("disposition") == "neuron"]
    result = _threshold_curve(
        truth,
        candidates,
        evaluator=lambda gt, pred: object_matching_metrics(gt, pred, iou_threshold=iou_threshold, centroid_tolerance_px=centroid_tolerance_px),
        precision_key="object_precision",
        recall_key="object_recall",
    )
    result.update({"coverage_mode": coverage_mode, "unresolved_excluded_count": sum(item.get("disposition") == "unresolved" for item in ground_truth)})
    return result


def exhaustive_event_pr_curve(
    ground_truth: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *, coverage_mode: str,
    onset_tolerance_frames: int = 2,
) -> dict[str, Any]:
    authorize_metric(coverage_mode, "event_ap")
    truth = [item for item in ground_truth if item.get("disposition") == "event"]
    result = _threshold_curve(
        truth,
        candidates,
        evaluator=lambda gt, pred: event_timing_metrics(gt, pred, onset_tolerance_frames=onset_tolerance_frames),
        precision_key="event_precision",
        recall_key="event_recall",
    )
    result.update({"coverage_mode": coverage_mode, "unresolved_excluded_count": sum(item.get("disposition") == "unresolved" for item in ground_truth)})
    return result
