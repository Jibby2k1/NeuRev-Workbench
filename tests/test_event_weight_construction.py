import numpy as np
import pytest

from neurobench.experiments.event_weighted_cs_parzen.sample_weights import (
    PairSampleIndex,
    build_weighted_pair_batch,
    equal_event_weights,
    validate_split_integrity,
    weight_effective_sample_size,
)
from neurobench.experiments.event_weighted_cs_parzen.sampling import (
    build_fold_sample_pools,
)


def test_equal_event_mass_ignores_unequal_event_sample_counts():
    indices = [
        PairSampleIndex(10, 1, 1, 1, "event_roi"),
        PairSampleIndex(11, 1, 1, 2, "event_roi"),
        PairSampleIndex(12, 1, 1, 2, "event_roi"),
        PairSampleIndex(13, 1, 1, 2, "event_roi"),
    ]
    weights = equal_event_weights(indices)
    assert weights[0] == pytest.approx(0.5)
    assert weights[1:].sum() == pytest.approx(0.5)


def test_duplicate_identity_merge_and_ess_closed_forms():
    natural_indices = [
        PairSampleIndex(10, 1, 1),
        PairSampleIndex(11, 1, 1),
    ]
    event_indices = [
        PairSampleIndex(11, 1, 1, 1, "event_roi"),
        PairSampleIndex(12, 1, 1, 1, "event_roi"),
    ]
    natural = np.asarray([[1, 2], [2, 3]], dtype=float)
    events = np.asarray([[2, 3], [3, 5]], dtype=float)
    batch = build_weighted_pair_batch(
        natural, natural_indices, events, event_indices, alpha=0.2
    )
    assert len(batch.samples) == 3
    assert batch.weight_sum == pytest.approx(1)
    assert batch.per_event_mass == {1: pytest.approx(0.2)}
    assert weight_effective_sample_size(np.ones(4)) == pytest.approx(4)
    assert weight_effective_sample_size(np.asarray([1, 0, 0, 0])) == pytest.approx(1)


def test_split_first_pools_exclude_heldout_event_and_guard():
    mask = np.ones((9, 10), dtype=bool)
    intervals = {1: (20, 22), 2: (30, 32), 3: (40, 42), 4: (50, 52)}
    labels = [
        {
            "burst_id": event,
            "x_px": 5,
            "y_px": 4,
        }
        for event in intervals
    ]
    pools = build_fold_sample_pools(
        heldout_event_id=2,
        event_intervals_ui=intervals,
        review_interval_ui=(1, 60),
        anatomy_mask=mask,
        labels=labels,
        mode="roi_balanced",
        heldout_guard_frames=2,
        screen_samples=8,
        confirmation_samples=16,
        event_screen_max_samples_per_event=2,
        event_confirmation_max_samples_per_event=4,
        event_roi_radius_px=2,
        seed=7,
    )
    all_indices = (*pools.natural_confirm_indices, *pools.event_confirm_indices)
    assert all(not 28 <= row.frame_ui <= 34 for row in all_indices)
    assert all(row.event_id != 2 for row in pools.event_confirm_indices)
    assert set(row.event_id for row in pools.event_confirm_indices) == {1, 3, 4}


def test_frame_and_roi_event_support_are_distinct():
    mask = np.ones((10, 12), dtype=bool)
    intervals = {1: (20, 22), 2: (30, 32), 3: (40, 42), 4: (50, 52)}
    labels = [{"burst_id": event, "x_px": 6, "y_px": 5} for event in intervals]
    common = dict(
        heldout_event_id=1,
        event_intervals_ui=intervals,
        review_interval_ui=(1, 60),
        anatomy_mask=mask,
        labels=labels,
        heldout_guard_frames=1,
        screen_samples=8,
        confirmation_samples=16,
        event_screen_max_samples_per_event=2,
        event_confirmation_max_samples_per_event=6,
        event_roi_radius_px=1,
        seed=19,
    )
    frame = build_fold_sample_pools(mode="frame_balanced", **common)
    roi = build_fold_sample_pools(mode="roi_balanced", **common)
    assert {row.stratum for row in frame.event_confirm_indices} == {"event_frame"}
    assert {row.stratum for row in roi.event_confirm_indices} == {"event_roi"}
    assert {
        row.identity for row in frame.event_confirm_indices
    } != {row.identity for row in roi.event_confirm_indices}


def test_leakage_validator_rejects_guard_and_heldout_metadata():
    with pytest.raises(ValueError, match="guard"):
        validate_split_integrity(
            [PairSampleIndex(20, 1, 1)],
            heldout_interval_ui=(21, 23),
            guard_frames=1,
            heldout_event_id=2,
        )
    with pytest.raises(ValueError, match="held-out event"):
        validate_split_integrity(
            [PairSampleIndex(10, 1, 1, 2, "event_roi")],
            heldout_interval_ui=(21, 23),
            guard_frames=1,
            heldout_event_id=2,
        )
