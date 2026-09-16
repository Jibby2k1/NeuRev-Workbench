import json
import math

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference.spatiotemporal_metrics import (
    evaluate_framewise, evaluate_sparse_occurrences, evaluate_threshold_sweep,
    extract_nms_prefix, filter_nms_prefix, paired_summary, seal_candidates, threshold_grid,
)
from neurobench.experiments.gamma_ls_difference.two_stencil_evaluation import extract_frame_candidates


def proposal(identity, frame, score, x, y):
    return dict(proposal_id=identity, source_frame_ui=frame, score=score, x_px=x, y_px=y)


def truth(event_id="e1", roi="r1", start=1, stop=3, x=10, y=10):
    event = dict(event_id=event_id, observation_id=event_id, canonical_roi_id=roi,
                 source_start_ui=start, source_stop_ui=stop, x_px=x, y_px=y, burst_id=1)
    active = [{**event, "source_frame_ui": frame} for frame in range(start, stop + 1)]
    return event, active


def test_framewise_one_to_one_duplicates_event_delay_and_exposure():
    event, active = truth()
    rows = [proposal("p1", 2, 5, 10, 10), proposal("duplicate", 2, 4, 11, 10),
            proposal("outside", 2, 3, 25, 25), proposal("later", 3, 2, 10, 10)]
    sealed = seal_candidates(rows, source_frames_ui=[1, 2, 3, 4])
    result = evaluate_framewise(sealed, active, [event])
    summary = result["summary"]
    assert summary["true_positive_count"] == 2
    assert summary["false_positive_count"] == 2
    assert summary["false_negative_count"] == 1
    assert summary["precision"] == 0.5
    assert summary["framewise_sensitivity"] == 2 / 3
    assert summary["duplicate_near_active_region_count"] == 1
    assert summary["false_proposals_per_second"] == 25
    assert summary["event_window_coverage"] == 1
    assert result["event_rows"][0]["first_delay_ms"] == 20
    assert result["frame_rows"][0]["proposal_count"] == 0
    assert summary["fpr"] is None and summary["event_precision"] is None


def test_matching_can_use_second_nearest_unassigned_roi_and_respects_inclusive_disk():
    first, a = truth("a", "a", stop=1, x=10)
    second, b = truth("b", "b", stop=1, x=16)
    rows = [proposal("first", 1, 3, 10, 10), proposal("second", 1, 2, 10, 10)]
    result = evaluate_framewise(seal_candidates(rows, source_frames_ui=[1]), a + b, [first, second])
    assert result["summary"]["true_positive_count"] == 2
    assert result["proposal_rows"][1]["match_distance_px"] == 6


def test_sweep_prefix_assignments_match_fresh_join_and_empty_endpoint_is_undefined_precision():
    event, active = truth(stop=2)
    rows = [proposal("a", 1, 2, 10, 10), proposal("b", 1, 2, 11, 10), proposal("c", 2, 0.5, 10, 10)]
    sealed = seal_candidates(rows, source_frames_ui=[1, 2])
    sweep = evaluate_threshold_sweep(sealed, active, [event], thresholds=(0, 0.5, 2, math.inf))
    for tau, row in zip((0, 0.5, 2, math.inf), sweep["curve_rows"]):
        fresh = seal_candidates([r for r in rows if r["score"] > tau], source_frames_ui=[1, 2])
        recomputed = evaluate_framewise(fresh, active, [event])["summary"]
        for key in ("true_positive_count", "false_positive_count", "precision", "framewise_sensitivity"):
            assert row[key] == recomputed[key]
    endpoint = sweep["curve_rows"][-1]
    assert endpoint["threshold_z"] is None
    assert endpoint["threshold_label"] == "no_output"
    assert endpoint["precision"] is None
    assert endpoint["framewise_sensitivity"] == 0
    json.dumps(sweep, allow_nan=False)
    assert len(threshold_grid()) == 45


def test_maintained_nms_prefix_matches_fresh_nms_with_ties_and_signed_background():
    values = np.full((44, 48), -2, dtype=np.float32)
    values[10, 10] = values[10, 18] = 2
    values[28, 12] = 0.5
    values[28, 30] = 5
    values[0, 0] = 100  # Maintained border rule must still exclude this peak.
    prefix = extract_nms_prefix(values, source_frame_ui=7, cell_id="fixture")
    for tau in (0, 0.5, 1, 2, 5, 10):
        reused = filter_nms_prefix(prefix, tau)
        fresh = extract_frame_candidates(values, source_frame_ui=7, threshold_z=tau, cell_id="fixture")
        assert [(r["score"], r["x_px"], r["y_px"], r["proposal_id"]) for r in reused] == [
            (r["score"], r["x_px"], r["y_px"], r["proposal_id"]) for r in fresh]
    assert filter_nms_prefix(prefix, math.inf) == []


def test_nuisance_only_truth_counts_false_proposals_but_sparse_real_does_not():
    sealed = seal_candidates([proposal("a", 1, 2, 10, 10)], source_frames_ui=[1, 2])
    synthetic = evaluate_framewise(sealed, [], [])["summary"]
    sparse = evaluate_framewise(sealed, [], [], truth_mode="sparse_real")["summary"]
    assert synthetic["false_positive_count"] == 1
    assert synthetic["precision"] == 0
    assert synthetic["framewise_sensitivity"] is None
    assert sparse["false_positive_count"] is sparse["precision"] is sparse["fpr"] is None
    assert sparse["unmatched_unknown_count"] == 1
    event, active = truth(stop=2)
    with pytest.raises(ValueError, match="not active-frame truth"):
        evaluate_framewise(sealed, active, [event], truth_mode="sparse_real")
    sparse_window = evaluate_sparse_occurrences(sealed, [event], burst_intervals_ui={1: (1, 2)})
    assert sparse_window["claim_boundaries"]["precision"] is None


def test_seal_copies_source_rows_and_truth_validation_rejects_duplicates_or_missing_frames():
    rows = [proposal("a", 1, 2, 10, 10)]
    sealed = seal_candidates(rows, source_frames_ui=[1, 2])
    rows[0]["score"] = 100
    assert sealed.rows[0].score == 2
    event, active = truth(stop=2)
    with pytest.raises(ValueError, match="duplicate active"):
        evaluate_framewise(sealed, active + active[:1], [event])
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_framewise(sealed, active[:1], [event])
    with pytest.raises(ValueError, match="unique"):
        seal_candidates(rows + rows, source_frames_ui=[1, 2])


def test_paired_summary_preserves_truth_strata_and_uses_scene_seed_units():
    rows = []
    for mode in ("fully_synthetic", "sparse_real"):
        for spec, precision in (("base", 0.5), ("candidate", 0.75)):
            rows.append(dict(scene_id="s", template_id="t", seed=1, spec_id=spec,
                             truth_mode=mode, threshold_label="1",
                             precision=precision if mode == "fully_synthetic" else None))
    result = paired_summary(rows, baseline_spec_id="base")
    assert len(result["aggregate_rows"]) == 2
    a, b = result["aggregate_rows"]
    assert a["mean_delta_precision"] == 0.25
    assert b["mean_delta_precision"] is None
    assert a["paired_scene_seed_count"] == 1
    with pytest.raises(ValueError, match="baseline"):
        paired_summary(rows[1:], baseline_spec_id="base")
