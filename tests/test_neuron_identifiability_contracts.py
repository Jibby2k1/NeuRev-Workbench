from __future__ import annotations

import json

import pytest

from neurobench.experiments.neuron_identifiability.contracts import (
    ObservationRecord, atomic_json, disk_geometry_hash,
)


def record(**updates):
    values = dict(
        observation_id="b01__roi_010", burst_id=1, original_roi_id="roi_010",
        observation_site_id="roi_010", canonical_neuron_id="roi_010",
        x_px=10.0, y_px=12.0, geometry_kind="disk_r2",
        geometry_hash=disk_geometry_hash(10, 12, 2),
        original_start_frame_ui=21, original_end_frame_ui=25,
        event_onset_ui=None, event_peak_ui=None, event_end_ui=None,
        include_confirmed=True, include_inclusive=True,
        review_status="pending", disposition="unresolved",
    )
    values.update(updates)
    return ObservationRecord(**values)


def test_ui_numpy_interval_round_trip_and_trace_key() -> None:
    value = record()
    assert (value.start_zero, value.stop_zero_exclusive) == (20, 25)
    assert value.trace_key("video", "raw", "params")[:2] == ("roi_010", value.geometry_hash)


def test_partial_or_reversed_timing_is_rejected() -> None:
    with pytest.raises(ValueError): record(event_onset_ui=20)
    with pytest.raises(ValueError): record(event_onset_ui=23, event_peak_ui=22, event_end_ui=24)


def test_atomic_json_has_no_partial_final(tmp_path) -> None:
    path = tmp_path / "value.json"
    atomic_json(path, {"finite": 1.0})
    assert json.loads(path.read_text()) == {"finite": 1.0}
    assert not list(tmp_path.glob("*.partial*"))
