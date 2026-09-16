from copy import deepcopy
import numpy as np
import pytest
from neurobench.experiments.gamma_ls_difference import background_integrity as i
from neurobench.experiments.gamma_ls_difference import background_study as s


def protocol():
    return dict(experiment='gamma_background_conditioned_variance_factorial',expected_cells=252,expected_datasets=84,
        epochs=i.EPOCHS,frame_rate_hz=50,pixel_size_um=.5,eligible_area_px=13456,
        references=[{'arm_id':a} for a in s.ARMS],cases=s.case_design(),logical_cases=s.logical_design())


def test_complete_pair_geometry():
    pairs=i.protocol_matrix(protocol())
    assert len(pairs)==252
    assert sum(c['V']==0 for c,a in pairs)==126


@pytest.mark.parametrize('change',['drop_case','wrong_alias','wrong_background','wrong_epoch'])
def test_matrix_mutations_rejected(change):
    p=protocol()
    if change=='drop_case':p['cases'].pop()
    elif change=='wrong_alias':p['logical_cases'][-1]['canonical_case_id']='wrong'
    elif change=='wrong_background':p['cases'][0]['background']='flat'
    else:p['epochs']={}
    with pytest.raises(ValueError):i.protocol_matrix(p)


def test_prefix_ignores_future_but_catches_past():
    a=np.ones((5,5,5),dtype=np.float32);b=a.copy();b[4]=9
    assert i.compare_prefix_arrays(a,b,4)['bitwise_equal']
    b[3,1,1]=2
    with pytest.raises(ValueError):i.compare_prefix_arrays(a,b,4)


def test_setup_floor_mismatch_rejected():
    plans=[[{'threshold':1.}]]*14
    assert i._calibration_group(plans,[.2]*14)==.2
    with pytest.raises(ValueError):i._calibration_group(plans,[.2]*13+[.21])


def test_candidate_identity_is_preserved():
    rows=[{'source_frame_ui':165,'proposal_id':'a','score':2.}]
    changed=[dict(rows[0],proposal_id='b')]
    with pytest.raises(ValueError):i.compare_prefix_rows(rows,changed,first_ui=165,last_ui=264,allowed_stop_ui=464)
