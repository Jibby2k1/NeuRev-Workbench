import math

import numpy as np
import pytest
from scipy.ndimage import maximum_filter

from neurobench.experiments.gamma_ls_difference.followup_fixtures import (
    SEEDS, SEPARATIONS_PX, case_id, generate_crowding,
)
from neurobench.experiments.gamma_ls_difference.followup_selection import extract_candidates
from neurobench.experiments.gamma_ls_difference.spatiotemporal_fixtures import temporal_profile
from neurobench.experiments.gamma_ls_difference.two_stencil_evaluation import extract_frame_candidates


def _without_ids(rows):
    return [{key: value for key, value in row.items() if key != "proposal_id"} for row in rows]


def _brute(score, window):
    values = np.asarray(score, dtype=np.float64)
    keep = (values == maximum_filter(values, size=window, mode="nearest")) & (values > 0)
    keep[:6] = False; keep[-6:] = False; keep[:, :6] = False; keep[:, -6:] = False
    y, x = np.nonzero(keep)
    result = []
    for index in np.lexsort((x, y, -values[y, x])):
        px, py = int(x[index]), int(y[index])
        if all((px - old_x) ** 2 + (py - old_y) ** 2 > 36 for _, old_x, old_y in result):
            result.append((float(values[py, px]), px, py))
    return result


def test_twelve_fixed_unique_condition_ids_and_invalid_inputs():
    assert len({case_id(seed, distance) for seed in SEEDS for distance in SEPARATIONS_PX}) == 12
    assert case_id(20260916) == "crowding_alone__seed20260916"
    assert case_id(20260918, 12) == "crowding_sep12__seed20260918"
    for seed, distance in ((0, None), (True, None), (20260916, 6), (20260916, True)):
        with pytest.raises(ValueError):
            case_id(seed, distance)


@pytest.mark.parametrize("window", [3, 13])
def test_spatial_hash_matches_full_greedy_reference_on_random_ties_and_plateaus(window):
    rng = np.random.default_rng(20260916)
    for score in (rng.normal(size=(45, 51)), np.round(rng.normal(size=(45, 51)), 1),
                  np.ones((45, 51)), np.zeros((45, 51)), -np.ones((45, 51))):
        result = extract_candidates(score, source_frame_ui=185, window=window, cell_id="fixture")
        assert [(row["score"], row["x_px"], row["y_px"]) for row in result] == _brute(score, window)
        assert all(row["source_frame_ui"] == 185 and row["source_time_s"] == 3.68 for row in result)
        assert len({row["proposal_id"] for row in result}) == len(result)
        if window == 13:
            old = extract_frame_candidates(score, source_frame_ui=185, threshold_z=0., cell_id="fixture")
            assert _without_ids(result) == _without_ids(old)


@pytest.mark.parametrize("dx,dy,expected3,expected13", [(5, 5, 2, 1), (8, 0, 2, 2),
                                                        (6, 0, 1, 1), (6, 1, 2, 1)])
def test_prefilter_square_is_distinct_from_strict_euclidean_separation(dx, dy, expected3, expected13):
    score = np.full((45, 45), -1.)
    score[20, 20] = 3.
    score[20 + dy, 20 + dx] = 2.
    assert len(extract_candidates(score, source_frame_ui=1, window=3)) == expected3
    assert len(extract_candidates(score, source_frame_ui=1, window=13)) == expected13


@pytest.mark.parametrize("window", [3, 13])
def test_six_pixel_border_and_positive_threshold_are_unchanged(window):
    score = np.full((41, 43), -1.)
    score[5, 10] = 2.       # outside the fixed border
    score[20, 6] = 1.      # first eligible column
    score[30, 30] = 0.     # strict zero threshold excludes this maximum
    result = extract_candidates(score, source_frame_ui=1, window=window)
    assert [(r["x_px"], r["y_px"]) for r in result] == [(6, 20)]


@pytest.mark.parametrize("score,frame,window", [(np.zeros((12, 20)), 1, 3),
    (np.full((20, 20), np.nan), 1, 3), (np.zeros((20, 20)), 0, 3),
    (np.zeros((20, 20)), True, 13), (np.zeros((20, 20)), 1.5, 3),
    (np.zeros((20, 20)), 1, 5)])
def test_invalid_selector_geometry_or_contract_fails(score, frame, window):
    with pytest.raises(ValueError):
        extract_candidates(score, source_frame_ui=frame, window=window)


def test_fixed_movie_truth_setup_and_paired_noise_are_exact():
    alone = generate_crowding(20260916)
    profile = temporal_profile(100., 1000.)
    assert alone.raw.shape == (464, 226, 226) and alone.raw.dtype == np.float32
    assert alone.metadata["evaluation_box_yxyx"] == [49, 49, 177, 177]
    assert alone.metadata["setup_source_frames_ui"] == list(range(65, 165))
    assert alone.metadata["application_source_frames_ui"] == list(range(165, 465))
    weak = alone.event_rows[0]
    assert (weak["x_px"], weak["y_px"], weak["source_role"]) == (117., 113., "weak")
    assert weak["source_start_ui"] == 185 and weak["source_stop_ui"] == 184 + len(profile)
    assert weak["peak_amplitude"] == 18 and weak["sigma_px"] == 1
    assert len(alone.active_rows) == len(profile)
    # Independently reconstruct the base-noise arithmetic used in the preceding pilot.
    yy, xx = np.indices((226, 226), dtype=np.float32)
    background = 100 + .03 * (xx - 113) + .02 * (yy - 113)
    rng = np.random.default_rng(20260916)
    assert np.array_equal(alone.raw[0], background + 2.0 * rng.standard_normal((226, 226), dtype=np.float32))
    for separation in (8, 12, 16):
        paired = generate_crowding(20260916, separation)
        assert np.array_equal(paired.raw[:184], alone.raw[:184])
        a, b = paired.event_rows
        assert (a["source_start_ui"], a["source_stop_ui"], a["burst_id"]) == (
            b["source_start_ui"], b["source_stop_ui"], b["burst_id"])
        assert a["source_role"] == "weak" and b["source_role"] == "neighbor"
        assert b["x_px"] == 117 - separation and b["y_px"] == 113
        assert b["sigma_px"] == 2 and b["peak_amplitude"] == 24
        assert a["canonical_roi_id"] != b["canonical_roi_id"]
        assert len(paired.active_rows) == 2 * len(profile)
        assert len({(row["canonical_roi_id"], row["source_frame_ui"]) for row in paired.active_rows}) == len(paired.active_rows)
        outside_neighbor = (xx - b["x_px"]) ** 2 + (yy - b["y_px"]) ** 2 > 8 ** 2
        assert np.array_equal(paired.raw[:, outside_neighbor], alone.raw[:, outside_neighbor])
        # At the neighbor center the weak truncated footprint contributes zero.
        center = (int(b["y_px"]), int(b["x_px"]))
        observed = paired.raw[184:184 + len(profile), center[0], center[1]] - alone.raw[184:184 + len(profile), center[0], center[1]]
        np.testing.assert_allclose(observed, 24 * profile, rtol=0, atol=2e-5)
        assert paired.metadata["paired_noise_group"] == alone.metadata["paired_noise_group"]
        del paired


def test_seed_changes_base_noise_but_not_declared_source_geometry():
    first = generate_crowding(20260917)
    first_frame = first.raw[0].copy()
    geometry = [(r["x_px"], r["y_px"], r["sigma_px"], r["peak_amplitude"]) for r in first.event_rows]
    del first
    other = generate_crowding(20260918)
    assert not np.array_equal(first_frame, other.raw[0])
    assert geometry == [(r["x_px"], r["y_px"], r["sigma_px"], r["peak_amplitude"]) for r in other.event_rows]
