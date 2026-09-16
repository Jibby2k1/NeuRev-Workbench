import math
import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference.followup_calibration import (
    regions, allocate_budget, region_index, regional_plan, select)
from neurobench.experiments.gamma_ls_difference.followup_audit import _frozen_threshold, _threshold_at, _verify_candidate_score_binding
from neurobench.experiments.gamma_ls_difference.spatiotemporal_metrics import seal_candidates, evaluate_framewise


def test_exact_partition_and_integer_allocation():
    grid=regions((242,475))
    counts=[0]*6
    for y in range(6,236):
        for x in range(6,469):
            counts[region_index(x,y,grid)]+=1
    assert counts==[r['eligible_area_px'] for r in grid]
    assert sum(counts)==230*463
    for budget in (0,1,5,6,109,500):
        got=allocate_budget(budget,counts)
        assert sum(got)==budget
        assert all(abs(a-budget*b/sum(counts))<1 for a,b in zip(got,counts))
    assert allocate_budget(5,[1]*6)==[1,1,1,1,1,0]
    with pytest.raises(ValueError):region_index(5,20,grid)


def test_ties_never_exceed_shared_budget():
    grid=regions((128,128))
    peaks=[]
    for i,r in enumerate(grid):
        y,x,_,_=r['box_yxyx']
        peaks.extend(dict(x_px=x,y_px=y,score=10.+i) for _ in range(100))
    plan=regional_plan(peaks,grid,100)
    for point in plan[:8]:
        assert sum(r['setup_proposal_budget'] for r in point['regional_thresholds'])==point['setup_proposal_budget']
        assert point['setup_proposal_count']<=point['setup_proposal_budget']
        assert all(r['setup_proposal_count']<=r['setup_proposal_budget'] for r in point['regional_thresholds'])
    assert select(peaks,plan[-1])==[]
    assert len(select(peaks,plan[-2]))==len(peaks)
    assert plan[0]['threshold_id']=='q0'
    assert all(r['threshold'] is not None for r in plan[0]['regional_thresholds'])


def test_filter_keeps_original_rank_and_can_reassign_removed_higher_score():
    # Regional rejection of the higher score must free the truth assignment.
    prefix=[dict(proposal_id='a',source_frame_ui=1,x_px=19,y_px=20,score=10.,candidate_rank_within_frame=1),
            dict(proposal_id='b',source_frame_ui=1,x_px=26,y_px=20,score=8.,candidate_rank_within_frame=2)]
    setting=dict(regional_thresholds=[dict(box_yxyx=[6,6,34,24],threshold=11.),dict(box_yxyx=[6,24,34,34],threshold=7.)])
    kept=select(prefix,setting)
    assert [r['proposal_id'] for r in kept]==['b']
    assert kept[0]['candidate_rank_within_frame']==2 and kept[0]['score']==8.
    active=[dict(source_frame_ui=1,canonical_roi_id='roi',event_id='ev',x_px=22,y_px=20)]
    events=[dict(event_id='ev',canonical_roi_id='roi',source_start_ui=1,source_stop_ui=1,x_px=22,y_px=20)]
    result=evaluate_framewise(seal_candidates(kept,source_frames_ui=[1]),active,events,threshold_z=0.)
    assert result['summary']['true_positive_count']==1
    assert result['proposal_rows'][0]['proposal_id']=='b'


def test_regional_media_checks_native_score_and_local_cutoff():
    op=dict(regional_thresholds=[dict(box_yxyx=[0,0,4,2],threshold=10.),dict(box_yxyx=[0,2,4,4],threshold=2.)],
            application_source_start_ui=1,application_source_stop_ui=1)
    candidates=[dict(source_frame_ui=1,x_px=3,y_px=2,score=3.,threshold_z=2.)]
    threshold=_frozen_threshold(candidates,op)
    score=np.zeros((1,4,4));score[0,2,3]=3.
    _verify_candidate_score_binding(candidates,score,{1:0},threshold,op)
    assert _threshold_at(threshold,0,2)==10.
    with pytest.raises(ValueError):_frozen_threshold([dict(candidates[0],threshold_z=10.)],op)
    with pytest.raises(ValueError):_verify_candidate_score_binding([dict(candidates[0],score=4.)],score,{1:0},threshold,op)
    score[0,2,3]=1.
    with pytest.raises(ValueError):_verify_candidate_score_binding([dict(candidates[0],score=1.)],score,{1:0},threshold,op)
