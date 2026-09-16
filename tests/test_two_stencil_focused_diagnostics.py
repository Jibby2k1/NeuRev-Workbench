from __future__ import annotations

import json

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import two_stencil_focused_diagnostics as focused


@pytest.mark.parametrize("matched,maximum,tau,proposals,sites,category", (
    (True, 2, 1, 1, 1, "matched"),
    (False, 1, 1, 0, 0, "no_matching_disk_pixel_above_threshold"),
    (False, 2, 1, 0, 0, "above_threshold_without_nearby_frame_proposal"),
    (False, 2, 1, 1, 0, "nearby_frame_proposal_without_nearby_burst_representative"),
    (False, 2, 1, 1, 1, "nearby_burst_representative_unmatched"),
))
def test_categories_distinguish_threshold_emission_consolidation_and_assignment(matched, maximum, tau, proposals, sites, category):
    assert focused.classify_occurrence(matched=matched, max_local_z=maximum, threshold_z=tau,
        nearby_frame_proposal_count=proposals, nearby_representative_count=sites) == category


def test_matching_disk_preserves_fractional_center_and_clips_to_image():
    ys, xs = focused.matching_disk((9, 12), 1.5, .5, 2)
    expected = [(y, x) for y in range(9) for x in range(12) if (x-1.5)**2+(y-.5)**2 <= 4]
    assert list(zip(ys, xs)) == expected
    assert (2, 3) not in expected  # Included by a rounded (2,1) radius-2 disk.
    assert (0, 0) in expected
    ys, xs = focused.matching_disk((20, 20), 10, 10)
    assert (10, 16) in set(zip(ys, xs))  # The exact 6px boundary is included.


def test_signed_negative_peak_is_not_abs_peak_and_ui_window_is_inclusive():
    stages = {name: np.zeros((6, 10, 12), dtype=np.float32) for name in focused.STAGES}
    stages["Z"][:] = -20
    # floor(2.5+.5)=3, whereas ties-to-even would wrongly choose column 2.
    stages["Z"][:, 4, 3] = [-.1, -7, -2, -2, -8, 10]
    stages["Z"][4, 4, 4] = -.5  # Disk maximum at the inclusive final frame.
    stages["X"][:, 4, 3] = np.arange(6)
    stages["A"][:, 4, 3] = np.arange(6)+10
    stages["M"][:, 4, 3] = np.arange(6)+20
    stages["contrast"][:, 4, 3] = np.arange(6)+30
    stages["sigma"][:] = .2
    stages["sigma"][2, 4, 3] = .05
    record, packet = focused.occurrence_stage_observations(stages, x_px=2.5, y_px=3.5,
        source_start_ui=2, source_stop_ui=5, threshold_z=-1, scale_floor=.1)
    assert (record["trace_x_px"], record["trace_y_px"]) == (3, 4)
    assert record["rounded_pixel_peak_Z_source_frame_ui"] == 3
    assert record["Z_at_rounded_pixel_peak_Z"] == -2
    assert record["X_at_rounded_pixel_peak_Z"] == 2
    assert record["A_at_rounded_pixel_peak_Z"] == 12
    assert record["rounded_pixel_peak_Z_threshold_margin"] == -1
    assert record["sigma_floor_active_at_rounded_pixel_peak_Z"] is True
    assert record["max_local_disk_Z_source_frame_ui"] == 5
    assert record["max_local_disk_Z"] == -.5
    assert record["max_local_disk_Z_threshold_margin"] == .5
    assert record["suprathreshold_matching_disk_pixel_frame_count"] == 1
    np.testing.assert_array_equal(packet["source_frame_ui"], [2, 3, 4, 5])
    np.testing.assert_array_equal(packet["sigma_floor_active"], [False, True, False, False])


def test_trace_rounding_clips_only_at_final_pixel_and_disk_peak_ties_are_stable():
    stages = {name: np.zeros((2, 3, 4), dtype=np.float32) for name in focused.STAGES}
    record, _ = focused.occurrence_stage_observations(stages, x_px=3.8, y_px=2.9,
        source_start_ui=1, source_stop_ui=2, threshold_z=0, scale_floor=0)
    assert (record["trace_x_px"], record["trace_y_px"]) == (3, 2)
    assert record["max_local_disk_Z_source_frame_ui"] == 1
    assert (record["max_local_disk_Z_y_px"], record["max_local_disk_Z_x_px"]) == (0, 0)
    assert record["sigma_floor_active_at_rounded_pixel_peak_Z"] is False


def test_saved_proposals_use_inclusive_time_float_distance_and_burst_sites():
    occurrence = {"x_px": 10.5, "y_px": 10, "source_start_ui": 2, "source_stop_ui": 4, "burst_id": 1}
    candidates = [
        {"proposal_id": "before", "source_frame_ui": 1, "x_px": 10, "y_px": 10},
        {"proposal_id": "start", "source_frame_ui": 2, "x_px": 5, "y_px": 10},
        {"proposal_id": "stop", "source_frame_ui": 4, "x_px": 16, "y_px": 10},
        {"proposal_id": "outside_float_disk", "source_frame_ui": 3, "x_px": 17, "y_px": 10},
        {"proposal_id": "after", "source_frame_ui": 5, "x_px": 10, "y_px": 10},
    ]
    sites = [{"site_id": "outside_rep", "burst_id": 1, "x_px": 17, "y_px": 10},
             {"site_id": "other_burst", "burst_id": 2, "x_px": 10, "y_px": 10}]
    observed = focused.nearby_readout_observations(occurrence, candidates, sites)
    assert observed["nearby_frame_proposal_count"] == 2
    assert json.loads(observed["nearby_frame_proposal_ids"]) == ["start", "stop"]
    assert observed["nearby_representative_count"] == 0
    assert observed["nearest_burst_representative_distance_px"] == 6.5
    assert focused.classify_occurrence(matched=False, max_local_z=2, threshold_z=1,
        nearby_frame_proposal_count=observed["nearby_frame_proposal_count"],
        nearby_representative_count=observed["nearby_representative_count"]) == focused.CATEGORIES[3]


def test_false_csv_string_does_not_become_true_and_malformed_values_fail():
    assert focused._matched("False") is False
    assert focused._matched("True") is True
    with pytest.raises(ValueError):
        focused._matched("not-a-bool")
    with pytest.raises(ValueError):
        focused.classify_occurrence(matched=False, max_local_z=np.nan, threshold_z=1,
                                    nearby_frame_proposal_count=0, nearby_representative_count=0)


def test_sealed_file_mutation_is_rejected(tmp_path):
    path = tmp_path/"sealed.tsv"
    path.write_text("original\n")
    expected = focused._sha256(path)
    assert focused._verify_file(path, expected) == expected
    path.write_text("changed\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        focused._verify_file(path, expected)


def test_incomplete_matrix_is_rejected_before_diagnostic_output_creation(tmp_path):
    (tmp_path/"protocol.json").write_text(json.dumps({"cells": [{"cell_id": "one"}]}))
    (tmp_path/"preflight.json").write_text("{}")
    with pytest.raises(ValueError, match="25-cell"):
        focused.run_focused_diagnostics(tmp_path, render=False)
    assert not (tmp_path/"focused_diagnostics").exists()


def test_join_replay_detects_assignment_drift_even_when_total_matches_stays_fixed(tmp_path):
    from neurobench.experiments.gamma_ls_difference.two_stencil_evaluation import evaluate_occurrence_windows
    candidates = [{"proposal_id": "p1", "score": "3.0", "x_px": "10", "y_px": "10", "source_frame_ui": "2"}]
    labels = [
        {"observation_id": "a", "canonical_roi_id": "roi_a", "burst_id": "1",
         "x_px": "10.0", "y_px": "10.0", "source_start_ui": "2", "source_stop_ui": "4"},
        {"observation_id": "b", "canonical_roi_id": "roi_b", "burst_id": "1",
         "x_px": "11.0", "y_px": "10.0", "source_start_ui": "2", "source_stop_ui": "4"},
    ]
    expected = evaluate_occurrence_windows(candidates, labels, burst_intervals_ui={1: (2, 4)})
    for name in ("occurrence_rows", "site_rows", "membership_rows", "burst_summaries"):
        focused._write_tsv(tmp_path/f"{name}.tsv", [{"operator_cell_id": "cell", "readout": "Z",
            "target_proposals_per_frame": 1.0, **row} for row in expected[name]])
    (tmp_path/"summary.json").write_text(json.dumps(expected["summary"]))
    occurrences, sites = focused._verify_join_tables(tmp_path, "cell", candidates, labels)
    assert [row["matched"] for row in occurrences] == [True, False]
    assert len(sites) == 1
    rows = focused._read_tsv(tmp_path/"occurrence_rows.tsv")
    rows[0]["matched"], rows[1]["matched"] = "False", "True"
    focused._write_tsv(tmp_path/"occurrence_rows.tsv", rows)
    with pytest.raises(ValueError, match="differs from sealed candidate/label join"):
        focused._verify_join_tables(tmp_path, "cell", candidates, labels)
