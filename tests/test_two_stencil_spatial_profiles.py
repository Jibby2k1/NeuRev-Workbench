from __future__ import annotations

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference.two_stencil_spatial_profiles import (
    _locations, extract_profiles, snapshot_frame_index,
)


def test_profiles_bind_source_ui_metadata_and_round_half_up_without_axis_swap():
    # Snapshot ordinal 1 means source UI30, not source UI2 or array ordinal30.
    source_frames = [10, 30]
    images = np.asarray([np.arange(30).reshape(5, 6), 100+np.arange(30).reshape(5, 6)])
    index = snapshot_frame_index(source_frames, 30)
    center, profiles = extract_profiles(images[index], x_px=2.5, y_px=1.5, radius_px=2)
    assert index == 1
    assert (center["trace_x_px"], center["trace_y_px"]) == (3, 2)
    np.testing.assert_array_equal(profiles["horizontal"]["values"], [113, 114, 115, 116, 117])
    np.testing.assert_array_equal(profiles["vertical"]["values"], [103, 109, 115, 121, 127])
    with pytest.raises(ValueError, match="absent"):
        snapshot_frame_index(source_frames, 2)
    with pytest.raises(ValueError, match="distinct"):
        snapshot_frame_index([10, 10], 10)


def test_edge_profiles_retain_explicit_unavailable_offsets_and_clip_final_center():
    center, profiles = extract_profiles(np.arange(12).reshape(3, 4), x_px=3.8, y_px=2.9, radius_px=2)
    assert (center["trace_x_px"], center["trace_y_px"]) == (3, 2)
    np.testing.assert_array_equal(profiles["horizontal"]["valid"], [True, True, True, False, False])
    np.testing.assert_array_equal(profiles["horizontal"]["values"][:3], [9, 10, 11])
    assert np.isnan(profiles["horizontal"]["values"][3:]).all()


def test_anchor_location_uses_same_burst_nearest_site_without_assignment_or_frame_filter():
    label = {"observation_id": "b01__roi_001", "canonical_roi_id": "roi_001", "burst_id": "1",
             "x_px": "10.5", "y_px": "20", "source_start_ui": "2003", "source_stop_ui": "2026"}
    sites = [
        {"burst_id": 2, "site_id": "other_burst", "site_rank": 1, "x_px": 10, "y_px": 20,
         "representative_proposal_id": "wrong", "source_frame_ui": 2050},
        {"burst_id": 1, "site_id": "far", "site_rank": 2, "x_px": 19, "y_px": 20,
         "representative_proposal_id": "far_proposal", "source_frame_ui": 2003},
        {"burst_id": 1, "site_id": "nearest", "site_rank": 3, "x_px": 18, "y_px": 20,
         "representative_proposal_id": "nearest_proposal", "source_frame_ui": 2020},
    ]
    locations = _locations([{"observation_id": label["observation_id"]}], [label], sites)
    assert locations[0]["source_frame_ui"] == locations[1]["source_frame_ui"] == 2014
    assert locations[1]["anchor_site_id"] == "nearest"
    assert locations[1]["distance_from_expert_px"] == 7.5
    assert locations[1]["anchor_representative_source_frame_ui"] == 2020
    assert locations[1]["location_is_one_to_one_assignment"] is False
    missing = _locations([{"observation_id": label["observation_id"]}], [label], [])
    assert missing[1]["available"] is False and missing[1]["x_px"] is None
