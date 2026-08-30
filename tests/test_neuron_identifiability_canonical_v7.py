from neurobench.experiments.neuron_identifiability.canonical_v7_extension import _representatives


def test_v7_representatives_require_repeated_confirmed_neurons():
    rows = []
    for index in range(4):
        rows.append({
            "canonical_roi_id": f"roi_{index}", "confirmed_bursts": 3,
            "signed_peak_amplitude": 1.0 + index, "robust_peak_snr": 2.0 + index,
            "residual_peak_amplitude": 3.0 + index, "normalized_spatial_specificity": .1 + index,
            "recovery_score": index / 3,
        })
    selected = _representatives(rows)
    assert set(selected) == {"hero", "typical", "hard_case"}
    assert selected["hero"]["canonical_roi_id"] == "roi_3"
    assert selected["hard_case"]["canonical_roi_id"] == "roi_0"
