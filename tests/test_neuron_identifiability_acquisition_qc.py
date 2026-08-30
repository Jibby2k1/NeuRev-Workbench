from __future__ import annotations

import numpy as np

from neurobench.experiments.neuron_identifiability.acquisition_qc import acquisition_diagnostics, quiet_indices
from neurobench.experiments.neuron_identifiability.contracts import ObservationRecord, disk_geometry_hash


def _record() -> ObservationRecord:
    return ObservationRecord("b01__roi_001",1,"roi_001","roi_001","roi_001",20,20,"disk_r2",disk_geometry_hash(20,20,2),80,90,None,None,None,True,True,"legacy_source","confirmed_neuron")


def test_quiet_indices_exclude_event_margin() -> None:
    values=quiet_indices(150,[_record()],margin=5)
    assert 73 in values and 74 not in values and 94 not in values and 95 in values


def test_acquisition_qc_detects_declared_column_jump() -> None:
    rng=np.random.default_rng(3); video=rng.poisson(50,size=(160,48,320)).astype(np.uint16)
    video[:,:,286:]+=100
    metrics,arrays=acquisition_diagnostics(video,[_record()],spatial_stride=2,maximum_quiet_frames=80)
    assert metrics["quiet_frame_count"]>40
    assert 286 in metrics["spatial"]["strongest_column_boundaries"][:3]
    assert arrays["mean_map"].shape==(48,320)
