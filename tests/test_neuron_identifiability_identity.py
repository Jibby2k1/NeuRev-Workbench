from __future__ import annotations

import numpy as np

from neurobench.experiments.neuron_identifiability.contracts import ObservationRecord, disk_geometry_hash
from neurobench.experiments.neuron_identifiability.identity import identity_counts
from neurobench.experiments.neuron_identifiability.trace_extraction import Geometry, extract_site_traces


def _record(site: str, x: float) -> ObservationRecord:
    return ObservationRecord(
        observation_id=f"b01__{site}", burst_id=1, original_roi_id=site,
        observation_site_id=site, canonical_neuron_id="roi_010", x_px=x, y_px=22,
        geometry_kind="disk_r2", geometry_hash=disk_geometry_hash(x, 22, 2),
        original_start_frame_ui=5, original_end_frame_ui=8,
        event_onset_ui=None, event_peak_ui=None, event_end_ui=None,
        include_confirmed=True, include_inclusive=True,
        review_status="provisional_expert_note", disposition="confirmed_neuron",
    )


def test_duplicate_canonical_identity_preserves_two_geometries_and_traces() -> None:
    video = np.zeros((12, 50, 60), dtype=np.float32)
    video[:, 20:25, 18:23] = np.arange(12)[:, None, None]
    video[:, 20:25, 38:43] = (np.arange(12) ** 2)[:, None, None]
    records = [_record("roi_010", 20), _record("roi_015", 40)]
    sites, traces = extract_site_traces(video, records, Geometry())
    assert sites == ["roi_010", "roi_015"]
    assert traces["raw"].shape == (2, 12)
    assert not np.allclose(traces["raw"][0], traces["raw"][1])
    assert records[0].geometry_hash != records[1].geometry_hash
    assert identity_counts(records) == {"occurrences": 2, "original_sites": 2, "proposed_canonical_identities": 1}


def test_round_trip_preserves_identity_fields() -> None:
    value = _record("roi_015", 20)
    assert ObservationRecord.from_dict(value.to_dict()) == value
