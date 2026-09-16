"""Small generated records only; no campaign movies, stages or labels."""
import copy
import itertools

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import background_report as report


def protocol():
    cases={};logical=[]
    for background,normalization,v,s,t,seed in itertools.product(
            report.BACKGROUNDS,report.NORMALIZATIONS,(0,1),(0,1),(0,1),report.SEEDS):
        canonical='raw' if not (s or t) else normalization
        case=f'{background}_{canonical}_{v}{s}{t}_{seed}'
        row=dict(canonical_case_id=case,background=background,normalization=normalization,V=v,S=s,T=t,
                 seed=seed,alias=normalization!=canonical,reused_dataset=background=='sloped' and canonical=='raw')
        logical.append(row)
        if not row['alias']:
            cases[case]=dict(case_id=case,background=background,normalization=normalization,V=v,S=s,T=t,
                seed=seed,reused_dataset=row['reused_dataset'],kind='factorial')
    return dict(cases=list(cases.values()),logical_cases=logical,references=[dict(arm_id=a) for a in report.ARMS],
        expected_cells=252,expected_curve_rows=5040,frame_rate_hz=50,pixel_size_um=.5,code_bindings=[],kernel_bindings=[])


def test_physical_logical_inventory_and_only_identity_aliases():
    p=protocol();assert len(report._matrix(p))==252
    rows=[dict(case_id=c['case_id'],arm_id=a,y=float(i)) for i,c in enumerate(p['cases']) for a in report.ARMS]
    logical=report.expand_logical(rows,p)
    assert len(logical)==288 and sum(r['is_alias'] for r in logical)==36
    assert len({(r['canonical_case_id'],r['arm_id']) for r in logical})==252
    physical={(r['case_id'],r['arm_id']):r for r in rows}
    assert all(r['y']==physical[r['canonical_case_id'],r['arm_id']]['y'] for r in logical)
    assert sum(c['reused_dataset'] for c in p['cases'])==24


@pytest.mark.parametrize('mutation',('alias_factor','missing_logical','duplicate_logical'))
def test_invalid_logical_factorial_is_rejected(mutation):
    p=protocol()
    if mutation=='alias_factor':next(r for r in p['logical_cases'] if r['alias'])['canonical_case_id']=p['cases'][-1]['case_id']
    elif mutation=='missing_logical':p['logical_cases'].pop()
    else:p['logical_cases'][-1]=dict(p['logical_cases'][0])
    with pytest.raises(RuntimeError):report._matrix(p)


def test_missing_seal_gate_precedes_outcomes(tmp_path,monkeypatch):
    p=protocol();reads=[]
    class Source:
        def read(self,path):
            reads.append(path.name)
            if path.name=='protocol.json':return p
            if path.name=='preflight.json':return dict(status='PASS',protocol_sha256='p')
            raise AssertionError('Must not open outcomes before complete seals')
        def bind(self,path):return dict(sha256='p')
        def check(self,b):raise AssertionError('No fixture dependencies')
    monkeypatch.setattr(report,'Sources',Source)
    with pytest.raises(RuntimeError,match='All 252'):report._load(tmp_path)
    assert reads==['protocol.json','preflight.json']


@pytest.mark.parametrize('gradient',((0.,0.),(.03,.02)))
def test_known_background_second_moment_is_not_spatial_variance(gradient):
    yy,xx=np.mgrid[:18,:18];bg=dict(offset=100.,gradient_x_per_px=gradient[0],gradient_y_per_px=gradient[1])
    field=100+gradient[0]*(xx+49-113)+gradient[1]*(yy+49-113)
    checker=2*((xx+yy)%2)-1
    array=np.stack([field+3,field+checker,field+2*checker])
    rows=report.residual_diagnostics(array,bg)
    assert rows[0]['residual_second_moment']==pytest.approx(9)
    assert rows[0]['residual_variance']==pytest.approx(0)
    assert rows[1]['residual_second_moment']==pytest.approx(1)
    assert rows[1]['residual_variance']==pytest.approx(1)
    assert rows[2]['temporal_lag1_covariance']==pytest.approx(2)
    changed=array.copy();changed[2]+=np.arange(18)[None,:]*10
    assert report.residual_diagnostics(changed,bg)[:2]==rows[:2]


def test_spread_squared_contrast_variance_and_strict_floor():
    a=np.ones((1,16,16));a[0,6:10,6:10]=np.arange(16).reshape(4,4)
    spread=report.stage_diagnostics(a,stage='Spread',floor=1,threshold=1)[0]
    contrast=report.stage_diagnostics(a,stage='C',floor=1,threshold=1)[0]
    z=report.stage_diagnostics(a,stage='Z',floor=1,threshold=1)[0]
    assert spread['squared_mean']==np.mean(np.arange(16)**2)
    assert contrast['variance']==np.var(np.arange(16))
    assert spread['floor_active_fraction']==1/16
    assert z['exceedance_fraction']==14/16
    assert not any('precision' in k or 'sensitivity' in k for k in spread|contrast|z)


def base_row(frame,**changes):
    row=dict(case_id='c',arm_id=report.ARMS[0],seed=report.SEEDS[0],case_kind='factorial',background='flat',
             normalization='raw',V=0,S=0,T=0,source_frame_ui=frame,threshold=2.,scale_floor=.1,proposal_count=1,
             Raw_residual_second_moment=4.,Raw_residual_variance=3.9,Input_residual_second_moment=.1,
             Input_residual_variance=.09,C_variance=.05,Spread_squared_mean=.07,floor_active_fraction=.1,Z_q999=2.)
    row.update(changes);return row


def test_prescribed_windows_inclusive_exposure_and_transition_bins():
    rows=[base_row(f) for f in range(65,465)]
    result={r['window']:r for r in report.aggregate_windows(rows)}
    assert result['early']['exposure_seconds']==2 and result['late']['exposure_seconds']==4
    assert result['settled_early']['source_start_ui']==215 and result['settled_early']['frame_count']==50
    assert result['settled_late']['source_start_ui']==315 and result['settled_late']['frame_count']==150
    assert result['early'][report.RATE]==result['late'][report.RATE]==pytest.approx(10000*50/3364)
    bins=report.time_bins(rows)
    assert len(bins)==40 and sum(r['proposal_count'] for r in bins)==400
    assert min(r['source_start_ui'] for r in bins)==65 and max(r['source_stop_ui'] for r in bins)==464
    assert all(r['frame_count']==10 for r in bins)
    with pytest.raises(RuntimeError,match='Incomplete'):report.aggregate_windows(rows[1:])
    with pytest.raises(RuntimeError,match='Duplicate'):report.aggregate_windows(rows+[rows[0]])


def test_paired_background_normalization_differences_are_not_regression_betas():
    rows=[]
    for b,n in report.PANELS:
        i,j=int(b=='flat'),int(n=='conditioned')
        rows.append(dict(seed=1,arm_id='a',window='late',V=1,S=1,T=0,background=b,normalization=n,y=10+2*i+3*j+7*i*j))
    result={(r['contrast'],r['condition']):r['difference'] for r in report.paired_effects(rows,metrics=('y',))}
    assert result==pytest.approx({('flat_minus_sloped','raw'):2,('flat_minus_sloped','conditioned'):9,
        ('conditioned_minus_raw','sloped'):3,('conditioned_minus_raw','flat'):10,
        ('background_by_normalization','difference_of_differences'):7})
    with pytest.raises(RuntimeError,match='Incomplete'):report.paired_effects(rows[:3],metrics=('y',))
    with pytest.raises(RuntimeError,match='Duplicate'):report.paired_effects(rows+[rows[0]],metrics=('y',))


def test_stationary_matching_preserves_V_but_not_raw_variance():
    from neurobench.experiments.gamma_ls_difference.noise_fixtures import theory
    rows=[]
    for v,s,t,n in itertools.product((0,1),(0,1),(0,1),report.NORMALIZATIONS):
        rows.append(dict(base_row(315,V=v,S=s,T=t,normalization=n),canonical_case_id='c',is_alias=False,window='settled_late',source_start_ui=315))
    predicted=report.variance_theory_rows(rows,theory());white=theory()['gaussian']['white_noise_energy']*.25
    for r in predicted:
        sigma2=25. if r['V'] else 4.
        if r['normalization']=='conditioned':assert r['conditioned_stationary_pointwise_variance_theory']==pytest.approx(sigma2*white)
        if r['normalization']=='conditioned' and (r['S'] or r['T']):assert r['raw_pointwise_variance_theory']<sigma2
        assert r['stationary_not_exact_transition_prediction'] is True


def test_threshold_windows_strict_ties_endpoints_and_late_count():
    prefix=[dict(proposal_id=str(i),source_frame_ui=f,score=s,x_px=10,y_px=10)
            for i,(f,s) in enumerate([(165,2.),(264,3.),(265,2.),(464,4.)])]
    plan=[dict(threshold_id='q1',threshold=2.),dict(threshold_id='all_positive',threshold=0.),dict(threshold_id='no_output',threshold=None)]
    rows=report.threshold_windows(prefix,plan,case_id='c',arm_id='a');lookup={(r['threshold_id'],r['window']):r for r in rows}
    assert lookup['q1','application']['proposal_count']==2 and lookup['q1','late']['proposal_count']==1
    assert lookup['all_positive','application']['proposal_count']==4
    assert lookup['no_output','application']['proposal_count']==0
    assert lookup['q1','late'][report.RATE]==pytest.approx(10000/(3364*4))
    with pytest.raises(RuntimeError,match='duplicate'):report.threshold_windows(prefix+[prefix[0]],plan,case_id='c',arm_id='a')


def test_stationary_theory_rejects_window_spanning_variance_step():
    from neurobench.experiments.gamma_ls_difference.noise_fixtures import theory
    row=dict(base_row(165,V=1),canonical_case_id='c',is_alias=False,window='application',source_start_ui=165,source_stop_ui=464)
    with pytest.raises(RuntimeError,match='single prescribed phase'):report.variance_theory_rows([row],theory())


def test_early_V_check_does_not_compare_poststep_changes():
    rows=[base_row(f,V=v,Z_q999=2 if f<=264 else 5*v) for f in (200,264,265) for v in (0,1)]
    assert report.verify_early_pairs(rows)['physical_frame_pairs']==2
    rows[1]['C_variance']+=.01
    with pytest.raises(RuntimeError,match='before UI265'):report.verify_early_pairs(rows)


def test_QA_requires_current_full_figure_inventory(tmp_path):
    figures=[]
    for i in range(6):
        p=tmp_path/f'f{i}.png';p.write_bytes(bytes([i]));figures.append(dict(png=p.name,pdf=f'f{i}.pdf'))
    report._write(tmp_path/'visual_qa.json',dict(status='PASS',figures=[report._binding(tmp_path/f['png']) for f in figures]))
    assert report._qa(tmp_path,dict(figures=figures),report.Sources())['visual_qa_complete']
    (tmp_path/'f1.png').write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='Changed evidence'):report._qa(tmp_path,dict(figures=figures),report.Sources())


def test_live_runner_logical_design_contract_without_running():
    from neurobench.experiments.gamma_ls_difference import background_study
    p=protocol();p['cases']=background_study.case_design();p['logical_cases']=background_study.logical_design()
    assert len(report._matrix(p))==252


def test_six_plot_contracts_complete_factors_and_data_extents(tmp_path,monkeypatch):
    import matplotlib.pyplot as plt
    epochs=[];curves=[];bins=[]
    for arm,b,n,v,s,t,seed in itertools.product(report.ARMS,report.BACKGROUNDS,report.NORMALIZATIONS,
            (0,1),(0,1),(0,1),report.SEEDS):
        row=dict(arm_id=arm,background=b,normalization=n,V=v,S=s,T=t,seed=seed)
        value=1+v*50+s*5+t*9+int(b=='sloped')*3+int(n=='raw')*2
        for window in ('late','settled_late'):
            epochs.append(dict(row,window=window,Raw_residual_second_moment=float(4+21*v),
                               Input_residual_second_moment=value/100,**{report.RATE:float(value)}))
        for window,threshold in itertools.product(('application','late'),report.common.THRESHOLDS):
            curves.append(dict(row,window=window,threshold_id=threshold,
                threshold=None if threshold=='no_output' else (0. if threshold=='all_positive' else
                    float(12-report.common.THRESHOLDS.index(threshold)+(seed-report.SEEDS[0])*.01)),
                **{report.RATE:0. if threshold=='no_output' else float(value*(2 if threshold=='all_positive' else 1))}))
        for i in range(40):bins.append(dict(row,time_mid_seconds=-1.91+i*.2,**{report.RATE:float(value+i)}))
    saved=[]
    def inspect(fig,out,stem,caption,uses):
        fig.canvas.draw()
        for ax in fig.axes:
            lo,hi=ax.get_ylim()
            for line in ax.lines:
                if line.get_transform()!=ax.transData:continue  # Epoch guides use axes coordinates vertically.
                y=np.asarray(line.get_ydata(),dtype=float)
                assert np.isfinite(y).all() and (y>=lo).all() and (y<=hi).all(),stem
            for collection in ax.collections:
                if len(collection.get_offsets()):
                    y=np.asarray(collection.get_offsets())[:,1]
                    assert (y>=lo).all() and (y<=hi).all(),stem
        assert len(fig.axes) in (6,12)
        if stem=='thresholds_late':
            ax=fig.axes[0]
            assert len(ax.lines)==8*3
            expected=[]
            for f,seed in itertools.product(report.FACTORS,report.SEEDS):
                group=[r for r in curves if r['window']=='late' and r['arm_id']==report.ARMS[0] and
                    (r['background'],r['normalization'])==report.PANELS[0] and tuple(r[k] for k in 'VST')==f and r['seed']==seed]
                finite,q1,_=report._native_curve_rows(group)
                expected.append(([r['threshold'] for r in finite],[r[report.RATE] for r in finite]))
            for line,(x,y) in zip(ax.lines,expected):
                np.testing.assert_array_equal(line.get_xdata(),x)
                np.testing.assert_array_equal(line.get_ydata(),y)
            assert fig.axes[-1].get_xlabel()=='Native Z cutoff'
            assert 'never averaged' in caption and 'infinite/None' in caption
        saved.append(stem);plt.close(fig)
        return dict(id=stem,png=stem+'.png',pdf=stem+'.pdf',caption=caption,source_tables=uses)
    monkeypatch.setattr(report.common,'_save',inspect)
    figures=report._plot(tmp_path,dict(logical_epochs=epochs,logical_threshold_windows=curves,logical_time_bins=bins))
    assert len(figures)==len(set(saved))==6
    assert not list(tmp_path.iterdir())  # No publication figures are generated by this test.


def test_native_cutoff_curve_omits_infinite_endpoint_and_preserves_ties():
    rows=[dict(threshold_id=name,threshold=None if name=='no_output' else 0. if name=='all_positive' else 3.,
               **{report.RATE:float(i)}) for i,name in enumerate(report.common.THRESHOLDS)]
    finite,q1,all_positive=report._native_curve_rows(rows)
    assert len(finite)==8 and len({r['threshold'] for r in finite})==1
    assert {r['threshold_id'] for r in finite}==set(report.common.THRESHOLDS)-{'all_positive','no_output'}
    assert q1['threshold_id']=='q1' and all_positive['threshold']==0
    rows[-1]['threshold']=float('inf')
    with pytest.raises(RuntimeError,match='Invalid native cutoff'):report._native_curve_rows(rows)
