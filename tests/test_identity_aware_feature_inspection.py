import numpy as np

from neurobench.experiments.neuron_identifiability.identity_aware_feature_inspection import (
    event_response,
    leakage_metrics,
    morphology,
)


def test_event_response_uses_pre_event_median_and_mad():
    trace = np.array([0, 1, 0, 1, 0, 5, 4], dtype=float)
    response, peak_mad = event_response(trace, 5, 7)
    assert response == 5.0
    assert peak_mad > 0


def test_morphology_flags_elongated_component():
    patch = np.zeros((13, 13), dtype=float)
    patch[6, 2:11] = 5
    metrics = morphology(patch)
    assert metrics["footprint_area_px"] >= 9
    assert metrics["eccentricity"] > 0.8


def test_morphology_solidity_detects_concavity():
    patch = np.zeros((13, 13), dtype=float)
    patch[3:10, 3] = 5
    patch[9, 3:10] = 5
    metrics = morphology(patch)
    assert metrics["solidity"] < 0.8
    assert metrics["local_peak_count"] >= 1


def test_neighbor_leakage_recovers_neighbor_weight():
    t = np.linspace(0, 4*np.pi, 200)
    original = np.sin(t)
    neighbor = np.cos(2*t)
    candidate = 0.2*original + 0.8*neighbor
    metrics = leakage_metrics(candidate, original, neighbor)
    assert metrics["nnls_neighbor_weight"] > metrics["nnls_original_weight"]
    assert metrics["nnls_two_trace_r2"] > 0.95
