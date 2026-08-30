from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from neurobench.experiments.neuron_identifiability.contracts import ObservationRecord, disk_geometry_hash
from neurobench.experiments.neuron_identifiability.trace_extraction import frozen_two_frame_ica_traces, occurrence_metrics


def _frozen(tmp_path):
    payload = {
        "fit": {"effective_directions": [[1.0, 0.0], [1.0, -1.0]], "mean": [0.0, 0.0]},
        "selection": {"selected_component": 1},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    path = tmp_path / "frozen.json"; path.write_text(json.dumps(payload))
    return path


def test_frozen_ica_is_hash_verified_and_oriented_toward_difference(tmp_path) -> None:
    raw = np.asarray([[1.0, 3.0, 2.0, 6.0]])
    path = _frozen(tmp_path)
    values, fingerprint = frozen_two_frame_ica_traces(raw, path)
    assert fingerprint == json.loads(path.read_text())["sha256"]
    np.testing.assert_allclose(values[0, 1:], np.diff(raw[0]))
    damaged = json.loads(path.read_text()); damaged["fit"]["mean"] = [1, 1]; path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="fingerprint"):
        frozen_two_frame_ica_traces(raw, path)


def test_occurrence_metrics_keep_native_and_change_channels_separate() -> None:
    record = ObservationRecord(
        observation_id="b01__roi_001", burst_id=1, original_roi_id="roi_001",
        observation_site_id="roi_001", canonical_neuron_id="roi_001", x_px=10, y_px=10,
        geometry_kind="disk_r2", geometry_hash=disk_geometry_hash(10,10,2),
        original_start_frame_ui=4, original_end_frame_ui=6,
        event_onset_ui=None, event_peak_ui=None, event_end_ui=None,
        include_confirmed=True, include_inclusive=True, review_status="legacy_source", disposition="confirmed_neuron",
    )
    raw = np.asarray([[1.,1.,1.,2.,5.,3.,2.]])
    annulus = np.ones_like(raw)
    traces = {"raw":raw,"annulus":annulus,"residual":raw-annulus,
              "frozen_two_frame_ica":np.asarray([[np.nan,0,0,1,3,-2,-1.]])}
    row = occurrence_metrics(record,traces,0,pre_frames=3)
    assert row["signed_peak_amplitude"] == 4.0
    assert row["frozen_two_frame_ica_signed_peak"] == 3.0
    assert row["analysis_view"] == "original_site_original_timing"
