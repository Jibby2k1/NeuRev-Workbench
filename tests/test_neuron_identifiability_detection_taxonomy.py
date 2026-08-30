import numpy as np

from neurobench.experiments.neuron_identifiability.detection_profile_taxonomy import FEATURES, _matrix, _union_occurrences


def test_union_collapses_cross_lane_candidate_at_same_location():
    base={"burst_id":1,"start_ui":10,"end_ui":20,"peak_frame_ui":15,"score":2.0}
    rows=[{**base,"lane":"coherence_w15","rank":2,"x_px":10,"y_px":10},{**base,"lane":"propagation_lag2_w15","rank":3,"x_px":12,"y_px":11}]
    union=_union_occurrences(rows)
    assert len(union)==1 and union[0]["lane_agreement"]==1
    assert len(FEATURES)>=10


def test_profile_matrix_preserves_raw_metrics_and_caps_influence():
    row={"raw_peak":1e12,"residual_peak":2,"robust_snr":3,"spatial_specificity":.4,"annulus_correlation":.5,"event_area":-1e15,"time_to_peak_fraction":.2,"lane_agreement":1,"mean_rank_fraction":.3,"recurrence_fraction":.5,"quiet_intensity":900}
    rows=[{**row,"burst_id":i%4+1} for i in range(8)]; rows[-1]["event_area"]=1
    z,_=_matrix(rows)
    assert rows[0]["spatial_specificity"]==.4
    assert "z_spatial_specificity" in rows[0]
    assert np.max(np.abs(z))<=3
