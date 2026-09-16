"""Bounded reporting contracts; no campaign outcomes, media or PDF authoring."""
import copy
import itertools
import json

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import persistence_report as report
from neurobench.experiments.gamma_ls_difference.proposal_persistence import analyze


def matrix():
    cells=[]
    for arm in report.REAL_ARMS:
        cells.append(dict(cell_id='real__'+arm,cohort='real',case_id='real',arm_id=arm,
                          first_ui=1800,last_ui=2359,fps=50,pixel_um=.5,input_proposal_count=0 if arm in ('level_X','level_C') else 3,
                          eligible_area_px=106490))
    for b,n,v,s,t,seed,arm in itertools.product(('flat','sloped'),('raw','conditioned'),(0,1),(0,1),(0,1),(1,2,3),report.REFS):
        if n=='conditioned' and s==t==0: continue
        cid=f'{b}_{n}_{v}{s}{t}_{seed}_{arm}'
        cells.append(dict(cell_id=cid,cohort='null',case_id=cid,arm_id=arm,first_ui=165,last_ui=464,fps=50,pixel_um=.5,
                          input_proposal_count=3,eligible_area_px=13456,background=b,normalization=n,V=v,S=s,T=t,seed=seed))
    rows=[]
    for cell,config in itertools.product(cells,report.CONFIGS):
        radius,gap=int(config[1]),int(config[-1]);n=cell['input_proposal_count']
        row=dict(cell,config_id=config,radius_px=radius,max_missing_frames=gap,proposal_count=n,
                 window_frames=cell['last_ui']-cell['first_ui']+1,site_count=int(n>0),episode_count=int(n>0),recurrent_site_count=0)
        row['window_seconds']=row['window_frames']/50
        for minimum in (3,5,10):
            row[f'persistent_proposal_count_{minimum}']=n if n>=minimum else 0
            row[f'persistent_site_count_{minimum}']=int(n>=minimum)
            row[f'persistent_episode_count_{minimum}']=int(n>=minimum)
        rows.append(row)
    return dict(cells=cells),rows


def test_complete_physical_matrix_and_zero_denominator():
    p,rows=matrix();result=report.validate_matrix(p,rows)
    assert len(result)==1560
    real=[r for r in result if r['cohort']=='real' and r['config_id']=='r4_g1']
    assert len(real)==8
    assert all(r['persistent_proposal_fraction_3'] is None for r in real if r['arm_id'] in ('level_X','level_C'))
    assert all(r['persistent_proposal_fraction_3']==1 for r in real if r['proposal_count'])
    assert report.fraction(0,0) is None
    assert report.fraction(0,2)==0


@pytest.mark.parametrize('mutation',('missing','duplicate','count','window','config','bad_fraction'))
def test_fail_closed_summary_contract(mutation):
    p,rows=matrix()
    if mutation=='missing': rows.pop()
    if mutation=='duplicate': rows[-1]=copy.deepcopy(rows[0])
    if mutation=='count': rows[-1]['proposal_count']+=1
    if mutation=='window': rows[-1]['window_seconds']=.02
    if mutation=='config': rows[-1]['radius_px']=2
    if mutation=='bad_fraction': rows[-1]['persistent_proposal_count_3']=4
    with pytest.raises(ValueError):report.validate_matrix(p,rows)


def test_completion_gate_precedes_any_result_reads_or_write(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'FileVerifier',lambda:pytest.fail('Do not read outcomes before gates'))
    with pytest.raises(ValueError,match='Complete analysis'): report._load(tmp_path)
    (tmp_path/'analysis_complete.json').write_text('{}')
    with pytest.raises(ValueError,match='inherited audit'): report._load(tmp_path)
    assert not (tmp_path/'report').exists()


def test_mean_uses_only_declared_setup_and_no_future():
    class Guarded:
        ndim=3;shape=(10,2,3)
        def __len__(self): return 10
        def __getitem__(self,s):
            assert 2<=s.start<s.stop<=6
            return np.stack([np.full((2,3),v) for v in range(s.start,s.stop)])
    mean=report.setup_mean(Guarded(),source_first_ui=100,setup_first_ui=102,setup_last_ui=105,chunk=3)
    assert np.all(mean==3.5)
    a=np.arange(60).reshape(10,2,3)
    first=report.setup_mean(a,source_first_ui=100,setup_first_ui=102,setup_last_ui=105)
    a[6:]=-100000
    assert np.array_equal(first,report.setup_mean(a,source_first_ui=100,setup_first_ui=102,setup_last_ui=105))
    with pytest.raises(ValueError,match='outside'):report.setup_mean(a,source_first_ui=100,setup_first_ui=99,setup_last_ui=105)


def site_fixture():
    rows=[dict(proposal_id=f'p{i}',source_frame_ui=f,x_px=20,y_px=20,score=3) for i,f in enumerate((10,12,14,19))]
    result=analyze(rows,first_ui=10,last_ui=20,radius_px=4,max_missing_frames=1,offset_xy=(49,49))
    cell=dict(cell_id='real__level_Z',arm_id='level_Z',input_proposal_count=4)
    sites=report.real_site_rows(cell, result['sites'],result['episodes'],result['memberships'])
    return cell,result,sites


def test_episodes_observations_spans_and_retrospective_counts():
    _,_,sites=site_fixture();site=sites[0]
    assert site['observed_frames']==4 and site['episode_count']==2 and site['recurrence_count']==1
    assert site['persistent_proposal_count_3']==3 and site['maximum_episode_observed_frames']==3
    assert site['maximum_episode_span_frames']==5 and site['maximum_episode_elapsed_ms']==80
    assert site['first_confirmation_ui']==14 and site['anchor_source_x_px']==69
    assert site['biological_status']=='unknown'


def test_missing_member_is_rejected():
    cell,result,_=site_fixture()
    with pytest.raises(ValueError,match='Incomplete membership'):report.real_site_rows(cell,result['sites'],result['episodes'],result['memberships'][:-1])


def test_html_includes_every_member_and_distinguishes_trace_pixel(tmp_path):
    cell,result,sites=site_fixture();links={}
    for row in result['memberships']:
        links[cell['cell_id'],row['proposal_id']]=dict(row,original_trace_x_px=21,original_trace_y_px=20,
            candidate_anchor_equals_original_trace_pixel=False,original_model_roi_id='old_roi',audit_root=str(tmp_path/'Original Audit'),
            trace_csv='trace.csv',trace_png='trace.png',closeup_thumbnail='close.png',closeup_video='close.mp4',
            model_fullfield_video='full.mp4',closeup_playback_time_s=.12,fullfield_frame_present=False,fullfield_playback_time_s=None)
    output=report.render_html(sites,links,tmp_path/'report')
    assert all(pid in output for pid in sites[0]['member_proposal_ids'])
    assert 'Original trace crop pixel is (21, 20)' in output and 'differs from' in output
    assert 'this UI decimated' in output and '#t=0.120000' in output
    assert 'Original%20Audit' in output and 'All 4 proposal members' in output
    changed=copy.deepcopy(sites);changed[0]['anchor_x_px']=999
    with pytest.raises(ValueError,match='anchor differs'):report.render_html(changed,links,tmp_path)


def test_report_retains_unknowns_and_no_precision_or_identity_promotion():
    p,rows=matrix();rows=report.validate_matrix(p,rows)
    text=report.report_text(rows,[])
    assert 'NA' in text and 'every previously emitted q1 proposal' in text
    assert 'exhaustive real-panel acceptance remains false' in text
    assert '40 ms elapsed time and a 60 ms inclusive span' in text
    assert 'reported' in text.lower() and 'retrospectively' in text
    assert '36 exact S=T=0 normalization aliases are excluded' in text


def test_four_in_memory_figures_without_PDF_writes():
    p,rows=matrix();rows=report.validate_matrix(p,rows)
    figures=report.build_figures(rows,[],np.arange(242*475,dtype=float).reshape(242,475))
    assert tuple(figures)==report.FIGURE_IDS
    assert len(figures['real_spatial_anchors'].axes)==8
    assert len(figures['null_persistence'].axes)==12
    assert len(figures['real_radius_gap'].axes)==4
    for fig in figures.values():
        fig.canvas.draw()
        width,height=fig.canvas.get_width_height()
        renderer=fig.canvas.get_renderer()
        for ax in fig.axes:
            xt=[tick.label1 for tick in ax.xaxis.get_major_ticks() if min(ax.get_xlim())<=tick.get_loc()<=max(ax.get_xlim())]
            yt=[tick.label1 for tick in ax.yaxis.get_major_ticks() if min(ax.get_ylim())<=tick.get_loc()<=max(ax.get_ylim())]
            for artist in [ax.title,ax.xaxis.label,ax.yaxis.label,*xt,*yt]:
                if artist.get_visible() and artist.get_text():
                    box=artist.get_window_extent(renderer)
                    assert box.x0>=-1 and box.y0>=-1 and box.x1<=width+1 and box.y1<=height+1
        report._plt().close(fig)
