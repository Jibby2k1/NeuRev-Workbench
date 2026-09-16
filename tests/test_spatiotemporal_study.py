import numpy as np

from neurobench.experiments.gamma_ls_difference.spatiotemporal_study import calibrate, frame_sets


def test_calibration_uses_only_declared_setup_and_obeys_area_budget():
    scores=np.zeros((5,32,32),np.float32)
    scores[0,10,10]=1; scores[1,10,10]=2; scores[2,10,10]=3
    first=calibrate(scores,[0,1,2],[65,66,67],194820/3)
    scores[3:]=10000
    second=calibrate(scores,[0,1,2],[65,66,67],194820/3)
    assert first==second
    assert first['threshold']==2
    assert first['setup_proposal_count']==first['setup_proposal_budget']==1


def test_calibration_all_zero_is_no_output_and_keeps_nonnegative_prefix():
    result=calibrate(np.zeros((2,20,20)),[0,1],[65,66],100)
    assert result['threshold']==0
    assert result['setup_proposal_count']==0


def test_frame_sets_have_warmup_before_setup_before_application():
    frames,setup,application=frame_sets(dict(source_frames_ui=list(range(1,465)),warmup_frames=64,setup_frames=100))
    assert setup==list(range(65,165))
    assert application==list(range(165,465))
    assert set(setup).isdisjoint(application)
