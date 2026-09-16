import json
import random

import pytest

from neurobench.experiments.gamma_ls_difference import spatiotemporal_sparse_fast as fast
from neurobench.experiments.gamma_ls_difference import two_stencil_evaluation as old


def compare(candidates, labels, windows):
    expected = old.evaluate_occurrence_windows(candidates, labels, burst_intervals_ui=windows)
    actual = fast.evaluate_occurrence_windows(iter(candidates), labels, burst_intervals_ui=windows)
    assert actual == expected
    assert json.dumps(actual, separators=(",", ":")) == json.dumps(expected, separators=(",", ":"))
    return actual


@pytest.mark.parametrize("seed", range(5))
def test_random_complete_output_matches_original(seed):
    rng = random.Random(seed)
    candidates = [dict(proposal_id=f"p{i}", source_frame_ui=rng.randint(1, 15), score=rng.randint(-2, 8),
                       x_px=rng.randrange(60), y_px=rng.randrange(50), cell_id="same") for i in range(500)]
    labels = [dict(observation_id=f"e{i}", canonical_roi_id=f"r{i}", burst_id=1+i%2,
                   x_px=rng.random()*60, y_px=rng.random()*50) for i in range(30)]
    compare(candidates, labels, {1: (2, 6), 2: (8, 13)})


def test_first_representative_not_nearest_and_exact_bucket_radius_boundary():
    candidates = [dict(proposal_id=f"p{i}", source_frame_ui=1, score=20-i, x_px=x, y_px=y)
                  for i, (x, y) in enumerate(((0,0), (8,0), (6,0), (12,0), (0,6), (6,6), (6,12), (0,0)))]
    result = compare(candidates, [], {1: (1, 1)})
    memberships = {row["proposal_id"]: row["site_id"] for row in result["membership_rows"]}
    assert memberships["p2"] == memberships["p0"]
    assert memberships["p3"] == memberships["p1"]
    assert memberships["p4"] == memberships["p0"]
    assert memberships["p7"] == memberships["p0"]


def test_empty_and_equal_score_y_x_frame_id_ties_match():
    labels = [dict(observation_id="e", burst_id=1, canonical_roi_id="r", x_px=12.5, y_px=12.5)]
    compare([], labels, {1: (2, 3)})
    candidates = [dict(proposal_id=identity, source_frame_ui=frame, score=2, x_px=x, y_px=y)
                  for identity, frame, x, y in (("z",3,12,12),("a",2,12,12),("b",2,18,12),("c",2,12,18))]
    result = compare(candidates, labels, {1: (2, 3)})
    assert result["site_rows"][0]["representative_proposal_id"] == "a"


def test_invalid_stream_errors_are_preserved():
    row = dict(proposal_id="p", source_frame_ui=2, score=2, x_px=1, y_px=1)
    for evaluator in (fast.evaluate_occurrence_windows, old.evaluate_occurrence_windows):
        with pytest.raises(ValueError, match="unique"):
            evaluator([row,row], [], burst_intervals_ui={1:(1,3)})
        with pytest.raises(ValueError, match="overlap"):
            evaluator([], [], burst_intervals_ui={1:(1,3),2:(3,5)})


def test_bounded_real_prefix_reader_stops_before_third_frame(tmp_path):
    rows = [dict(proposal_id=f"p{i}", source_frame_ui=1800+i//2, score=3, x_px=10+i, y_px=12)
            for i in range(8)]
    path = tmp_path / "prefix.json"
    path.write_text(json.dumps(rows,indent=2))
    assert fast._first_candidate_frames(path) == rows[:4]


def test_resume_refuses_without_parity_and_complete_matrix(tmp_path):
    path=tmp_path/"parity.json"
    path.write_text(json.dumps({"status":"FAIL","checks":[]}))
    with pytest.raises(ValueError,match="parity"):
        fast.resume_evaluation(tmp_path,parity_path=path)
