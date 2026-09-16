"""Tiny fail-closed checks for post-scoring integrity without outcome reads."""
import copy
import json

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import noise_integrity as n


def protocol():
    from neurobench.experiments.gamma_ls_difference.noise_study import case_design
    return dict(experiment='gamma_noise_variance_spatial_temporal_factorial',expected_cells=81,
        expected_datasets=27,epochs=copy.deepcopy(n.EPOCHS),frame_rate_hz=50,pixel_size_um=.5,
        eligible_area_px=13456,references=[dict(arm_id=a) for a in n.ARMS],cases=case_design())


def test_exact_production_matrix_contains_36_factorial_pairs():
    p=protocol();matrix=n.protocol_matrix(p)
    assert len(matrix)==81
    assert sum(c['kind']=='factorial' and c['V']==1 for c,a in matrix)==36


@pytest.mark.parametrize('change',['epoch','reference','duplicate','legacy'])
def test_changed_matrix_contract_is_rejected(change):
    p=protocol()
    if change=='epoch':p['epochs']['early']=[165,263]
    elif change=='reference':p['references'][0]['arm_id']='mean1_n3'
    elif change=='duplicate':p['cases'][-1]=p['cases'][0]
    else:p['cases'][-1]['V']=1
    with pytest.raises(ValueError):n.protocol_matrix(p)


def test_missing_seal_refuses_before_source_hashes_or_any_truth_read(tmp_path,monkeypatch):
    (tmp_path/'protocol.json').write_text(json.dumps(protocol()))
    class Guard(n.FileVerifier):
        def verify(self,*args,**kwargs):
            raise AssertionError('No provenance file should be traversed before the missing-seal gate')
        def read_json(self,path):
            assert path.name=='protocol.json'
            return super().read_json(path)
    monkeypatch.setattr(n,'FileVerifier',Guard)
    with pytest.raises(ValueError,match='Seal all 81'):n.check(tmp_path)
    assert not (tmp_path/'paired_variance_prefix_check.json').exists()


def test_existing_evaluation_is_observed_without_parsing_or_blocking(tmp_path,monkeypatch):
    p=protocol();(tmp_path/'protocol.json').write_text(json.dumps(p))
    for c,a in n.protocol_matrix(p):
        folder=tmp_path/'cells'/c['case_id']/a;folder.mkdir(parents=True)
        (folder/'sealed.json').write_text('{}')
    (tmp_path/'evaluation_complete.json').write_text('This outcome must not be parsed')
    (tmp_path/'computation_complete.json').write_text(json.dumps(dict(status='FAIL',cells=81)))
    class Guard(n.FileVerifier):
        def read_json(self,path):
            assert path.name!='evaluation_complete.json'
            return super().read_json(path)
    monkeypatch.setattr(n,'FileVerifier',Guard)
    with pytest.raises(ValueError,match='Complete all scoring'):n.check(tmp_path)


def test_numeric_comparison_never_reads_late_frames():
    class PrefixOnly:
        def __init__(self,values):self.values=values;self.shape=values.shape;self.dtype=values.dtype;self.reads=[]
        def __getitem__(self,key):
            assert isinstance(key,slice) and key.start>=0 and key.stop<=6
            self.reads.append((key.start,key.stop));return self.values[key]
    a=np.arange(90,dtype=np.float32).reshape(10,3,3);b=a.copy();b[6:]=np.nan
    left,right=PrefixOnly(a),PrefixOnly(b)
    result=n.compare_prefix_arrays(left,right,6,chunk_frames=4)
    assert result['bitwise_equal'] and left.reads==right.reads==[(0,4),(4,6)]


@pytest.mark.parametrize('change',['value','signed_zero','nan','dtype','shape'])
def test_prefix_corruption_or_incompatible_arrays_are_rejected(change):
    a=np.zeros((10,3,3),dtype=np.float32);b=a.copy()
    if change=='value':b[5,1,1]=1
    elif change=='signed_zero':b[0,0,0]=-0.
    elif change=='nan':a[1,1,1]=b[1,1,1]=np.nan
    elif change=='dtype':b=b.astype(np.float64)
    else:b=b[:5]
    with pytest.raises(ValueError):n.compare_prefix_arrays(a,b,6,chunk_frames=4)


def rows():
    return [dict(proposal_id=f'arm__ui{f}',cell_id='arm',source_frame_ui=f,score=2.,x_px=7,y_px=9)
            for f in (165,264,265)]


def test_candidate_prefix_is_exact_and_late_records_may_differ():
    left=rows();right=copy.deepcopy(left);right[-1]['score']=99
    result=n.compare_prefix_rows(left,right,first_ui=165,last_ui=264,allowed_stop_ui=464)
    assert result['record_count']==2 and result['fields_removed']==[]
    right[0]['cell_id']='another_case'
    with pytest.raises(ValueError,match='records differ'):
        n.compare_prefix_rows(left,right,first_ui=165,last_ui=264,allowed_stop_ui=464)


def test_bad_candidate_frame_and_rank_order_are_not_normalized():
    left=rows();right=copy.deepcopy(left);right[0]['source_frame_ui']=164
    with pytest.raises(ValueError,match='source frame'):
        n.compare_prefix_rows(left,right,first_ui=165,last_ui=264,allowed_stop_ui=464)
    with pytest.raises(ValueError,match='records differ'):
        n.compare_prefix_rows(left,list(reversed(left)),first_ui=165,last_ui=264,allowed_stop_ui=464)


@pytest.mark.parametrize('name',['experts.json','active.json','curves.json','operating_metrics_r2.json','evaluated.json'])
def test_truth_and_evaluation_json_cannot_be_parsed(name,tmp_path):
    class Forbidden:
        def read_json(self,path):raise AssertionError('Forbidden parse was attempted')
    with pytest.raises(ValueError,match='forbidden'):n._read_source_json(Forbidden(),tmp_path/name)


def test_calibration_group_requires_all_conditions_and_exact_plan_floor():
    plans=[[dict(threshold=2.)] for _ in range(9)];floors=[.3]*9
    assert n._calibration_group(plans,floors)==.3
    plans[-1][0]['threshold']=2.000001
    with pytest.raises(ValueError,match='plans or floors'):n._calibration_group(plans,floors)
    plans[-1][0]['threshold']=2.;floors[-1]=.300001
    with pytest.raises(ValueError,match='plans or floors'):n._calibration_group(plans,floors)
    with pytest.raises(ValueError,match='nine conditions'):n._calibration_group(plans[:-1],floors[:-1])
