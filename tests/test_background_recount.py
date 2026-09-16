"""Independent small-row test fixtures; never open a scientific output root."""
from collections import Counter
from copy import deepcopy
import csv
import itertools
import json
import math

import pytest

from neurobench.experiments.gamma_ls_difference import background_recount as r


def protocol_fixture():
    cases={};logical=[]
    for seed,bg,n,v,s,t in itertools.product(r.SEEDS,("sloped","flat"),("raw","conditioned"),(0,1),(0,1),(0,1)):
        canonical=r.physical_id(seed,bg,n,v,s,t);alias=n=="conditioned" and s==t==0
        row=dict(seed=seed,background=bg,normalization=n,V=v,S=s,T=t,
                 canonical_case_id=canonical,alias=alias,reused_dataset=bg=="sloped" and (n=="raw" or alias))
        logical.append(row)
        if not alias:
            cases[canonical]=dict(row,case_id=canonical,kind="factorial")
    return dict(experiment="gamma_background_conditioned_variance_factorial",expected_cells=252,
        expected_datasets=84,expected_curve_rows=5040,references=[dict(arm_id=a) for a in r.ARMS],
        frame_rate_hz=50,pixel_size_um=.5,eligible_area_px=13456,cases=list(cases.values()),logical_cases=logical,
        epochs=dict(setup=[65,164],early=[165,264],late=[265,464],application=[165,464]),
        settled_epochs=dict(early=[215,264],late=[315,464]))


def candidate(score,rank=1,frame=65):
    return dict(proposal_id=f"p{frame}_{rank}",source_frame_ui=frame,x_px=6+rank,y_px=7,
                candidate_rank_within_frame=rank,score=float(score),threshold_z=0.,target_proposals_per_frame=None)


def cutoff_fixture():
    setup=[candidate(s,i+1) for i,s in enumerate([9,8,7,7,7,6,6,5,4])]
    values=[row['score'] for row in setup];plan=[]
    for q in r.BUDGETS:
        budget=math.floor(q*13456*100/194820)
        tau=values[budget] if budget<len(values) else 0.
        plan.append(dict(threshold_id=f"q{q:g}",threshold=tau,setup_budget_per_reference_area_frame=q,
                         setup_proposal_budget=budget,setup_proposal_count=sum(x>tau for x in values)))
    plan.extend([dict(threshold_id="all_positive",threshold=0.,setup_budget_per_reference_area_frame=None,
                     setup_proposal_budget=None,setup_proposal_count=len(setup)),
                 dict(threshold_id="no_output",threshold=None,setup_budget_per_reference_area_frame=None,
                     setup_proposal_budget=0,setup_proposal_count=0)])
    return setup,plan


def test_exact_physical_and_logical_factorial_has_only_unit_aliases():
    cases,logical=r.validate_design(protocol_fixture())
    assert len(cases)==84 and len(logical)==96
    assert sum(row['alias'] for row in logical)==12
    assert sum(row['reused_dataset'] for row in cases.values())==24


@pytest.mark.parametrize('kind',['alias_target','alias_flag','physical_factor','missing_logical','epoch'])
def test_design_mutations_are_rejected(kind):
    p=protocol_fixture()
    if kind=='alias_target':p['logical_cases'][0]['canonical_case_id']=p['logical_cases'][-1]['canonical_case_id']
    elif kind=='alias_flag':p['logical_cases'][0]['alias']=True
    elif kind=='physical_factor':p['cases'][0]['background']='flat'
    elif kind=='missing_logical':p['logical_cases'].pop()
    else:p['settled_epochs']['late']=[314,464]
    with pytest.raises(ValueError):r.validate_design(p)


def test_setup_strict_ties_duplicate_cutoffs_and_endpoints():
    setup,plan=cutoff_fixture();q1=r.check_plan(plan,setup)
    assert q1['threshold']==6 and q1['setup_proposal_count']==5 and q1['setup_proposal_budget']==6
    scores=r.check_prefix(setup,65,164)
    assert r.above(scores,6)==5 and r.above(scores,None)==0 and r.above(scores,0)==9


@pytest.mark.parametrize('kind',['tiny_cutoff','count','budget','endpoint','missing'])
def test_setup_plan_changes_fail_even_if_the_count_stays_the_same(kind):
    setup,plan=cutoff_fixture()
    if kind=='tiny_cutoff':plan[3]['threshold']+=1e-13
    elif kind=='count':plan[3]['setup_proposal_count']+=1
    elif kind=='budget':plan[3]['setup_proposal_budget']+=1
    elif kind=='endpoint':plan[-1]['threshold']=999.
    else:plan.pop()
    with pytest.raises(ValueError):r.check_plan(plan,setup)


@pytest.mark.parametrize('kind',['rank','score_order','nonpositive','duplicate','border','future','fractional_pixel'])
def test_prefix_identity_geometry_and_order_are_not_normalized_away(kind):
    rows=[candidate(9,1),candidate(8,2)]
    if kind=='rank':rows[1]['candidate_rank_within_frame']=1
    elif kind=='score_order':rows[1]['score']=10.
    elif kind=='nonpositive':rows[1]['score']=0.
    elif kind=='duplicate':rows[1]['proposal_id']=rows[0]['proposal_id']
    elif kind=='border':rows[1]['x_px']=122
    elif kind=='future':rows[1]['source_frame_ui']=165
    else:rows[1]['x_px']=7.5
    with pytest.raises(ValueError):r.check_prefix(rows,65,164)


def test_q1_is_exact_strict_filtered_rows_and_source_frame_counter():
    setup,plan=cutoff_fixture();op=plan[3]
    prefix=[candidate(7,1,165),candidate(6,2,165),candidate(8,1,265)]
    expected=[dict(prefix[i],threshold_z=6.,target_proposals_per_frame=13456/194820,calibration_region_id='global') for i in (0,2)]
    counts=r.q1_counts(prefix,setup,expected,op)
    assert counts[165]==counts[265]==1 and counts[164]==0
    assert sum(counts[f] for f in range(65,165))==5
    changed=deepcopy(expected);changed[0]['proposal_id']='renamed'
    with pytest.raises(ValueError,match='identity'):r.q1_counts(prefix,setup,changed,op)


def metric(count,tau,radius):
    return dict(threshold_z=tau,truth_mode='fully_synthetic',match_radius_px=float(radius),
        application_frame_count=300,exposure_seconds=6.,proposal_count=count,proposals_per_frame=count/300,
        proposals_per_second=count/6.,false_proposals_per_second=count/6.,false_positive_count=count,
        true_positive_count=0,false_negative_count=0,active_region_frame_count=0,duplicate_near_active_region_count=0,
        event_count=0,recovered_event_count=0,first_delay_recovered_denominator=0,precision=0. if count else None,
        framewise_sensitivity=None,event_window_coverage=None,first_delay_ms_mean_among_recovered=None,
        first_delay_ms_median_among_recovered=None,unmatched_unknown_count=None,fpr=None,event_precision=None)


def test_all_twenty_curve_rows_use_native_cutoff_not_an_artificial_zero():
    _,plan=cutoff_fixture();prefix=[candidate(7,1,165),candidate(6,2,165),candidate(8,1,265)]
    curves=dict(event_rows=[],curve_rows=[])
    for setting,radius in itertools.product(plan,(2.,6.)):
        count=sum(setting['threshold'] is not None and p['score']>setting['threshold'] for p in prefix)
        curves['curve_rows'].append(dict(metric(count,setting['threshold'],radius),**setting,
                                        case_id='case',arm_id='arm',eligible_area_px=13456))
    actual=r.check_curves(curves,plan,prefix,'case','arm')
    assert actual['q1']==2 and actual['all_positive']==3 and actual['no_output']==0
    curves['curve_rows'][0]['threshold_z']=0.
    with pytest.raises(ValueError):r.check_curves(curves,plan,prefix,'case','arm')


def test_no_output_and_empty_truth_have_undefined_sensitivity():
    good=metric(0,None,2.);r.check_null_metric(good,0,None,2.)
    good['framewise_sensitivity']=0.
    with pytest.raises(ValueError,match='sensitivity'):r.check_null_metric(good,0,None,2.)


def report_fixture():
    cases,logical=r.validate_design(protocol_fixture());states={};physical=[];thresholds=[]
    for case,c in cases.items():
        for arm in r.ARMS:
            # Prescribed tiny counters expose background, normalization and V effects.
            base=1+int(c['background']=='sloped')+c['S']+2*c['T']
            base-=int(c['normalization']=='conditioned')
            frames=Counter({f:(base+c['V']*int(f>=265)) if f%17==0 else 0 for f in range(65,465)})
            plan=[dict(threshold_id=x,threshold=None if x=='no_output' else 2.) for x in r.THRESHOLDS]
            tc={(p['threshold_id'],w):0 if p['threshold'] is None else sum(frames[f] for f in range(lo,hi+1))
                for p in plan for w,(lo,hi) in r.THRESHOLD_WINDOWS.items()}
            states[case,arm]=dict(op=dict(threshold=2.,scale_floor=.1),plan=plan,frames=frames,threshold_counts=tc)
            factors={k:c[k] for k in r.FACTOR_FIELDS}
            for window,(lo,hi) in r.WINDOWS.items():
                count=sum(frames[f] for f in range(lo,hi+1));seconds=(hi-lo+1)*.02
                physical.append(dict(case_id=case,arm_id=arm,case_kind='factorial',**factors,window=window,epoch=window,
                    source_start_ui=lo,source_stop_ui=hi,frame_count=hi-lo+1,exposure_seconds=seconds,
                    eligible_area_px=13456,proposal_count=count,threshold=2.,scale_floor=.1,
                    **{r.RATE:count/(13456*.25*seconds)*10000}))
            for setting,(window,(lo,hi)) in itertools.product(plan,r.THRESHOLD_WINDOWS.items()):
                count=tc[setting['threshold_id'],window];seconds=(hi-lo+1)*.02
                thresholds.append(dict(case_id=case,arm_id=arm,**factors,**setting,window=window,
                    source_start_ui=lo,source_stop_ui=hi,frame_count=hi-lo+1,exposure_seconds=seconds,
                    eligible_area_px=13456,proposal_count=count,false_positive_count=count,
                    **{r.RATE:count/(13456*.25*seconds)*10000}))
    expanded=[];expanded_thresholds=[]
    for request in logical:
        canonical=request['canonical_case_id'];alias=request['alias']
        name=(f"logical__bg_{request['background']}__conditioned_v{request['V']}_s0_t0__seed{request['seed']}"
              if alias else canonical)
        extra=dict(case_id=name,canonical_case_id=canonical,normalization=request['normalization'],
                   is_alias=alias,source_reused=cases[canonical]['reused_dataset'])
        expanded.extend(dict(row,**extra) for row in physical if row['case_id']==canonical)
        expanded_thresholds.extend(dict(row,**extra) for row in thresholds if row['case_id']==canonical)
    byfactor={tuple(row[k] for k in ('seed','arm_id','window','V','S','T','background','normalization')):row[r.RATE]
              for row in expanded}
    paired=[]
    for seed,arm,window,v,s,t in itertools.product(r.SEEDS,r.ARMS,r.WINDOWS,(0,1),(0,1),(0,1)):
        key=(seed,arm,window,v,s,t);get=lambda bg,n:byfactor[key+(bg,n)]
        differences=[('flat_minus_sloped',n,get('flat',n)-get('sloped',n)) for n in ('raw','conditioned')]
        differences += [('conditioned_minus_raw',bg,get(bg,'conditioned')-get(bg,'raw')) for bg in ('sloped','flat')]
        differences += [('background_by_normalization','difference_of_differences',
                        (get('flat','conditioned')-get('flat','raw'))-(get('sloped','conditioned')-get('sloped','raw')))]
        for contrast,condition,value in differences:
            paired.append(dict(seed=seed,arm_id=arm,window=window,V=v,S=s,T=t,metric=r.RATE,
                contrast=contrast,condition=condition,difference=value,replicate_unit='paired seed',logical_aliases_are_not_replicates=True))
    return cases,logical,states,dict(physical_epochs=physical,logical_epochs=expanded,
        physical_threshold_windows=thresholds,logical_threshold_windows=expanded_thresholds,paired_effects=paired)


def test_complete_table_counts_denominators_paired_directions_and_aliases():
    cases,logical,states,tables=report_fixture()
    checks=r.verify_report_tables(tables,cases,logical,states)
    assert checks['physical_epoch_rows']==1260 and checks['logical_epoch_rows']==1440
    assert checks['physical_threshold_window_rows']==5040 and checks['logical_threshold_window_rows']==5760
    assert checks['paired_rate_contrasts']==1800 and checks['alias_logical_cells']==36
    assert len(checks['primary_late_counts'])==252


@pytest.mark.parametrize('kind',['count','denominator','alias','contrast_sign','missing_rate_pair','full_vs_late'])
def test_report_mutations_cannot_pass(kind):
    cases,logical,states,tables=report_fixture()
    if kind=='count':tables['physical_epochs'][0]['proposal_count']+=1
    elif kind=='denominator':tables['physical_epochs'][0][r.RATE]*=2
    elif kind=='alias':next(x for x in tables['logical_epochs'] if x['is_alias'])['canonical_case_id']='wrong'
    elif kind=='contrast_sign':next(x for x in tables['paired_effects'] if x['difference']!=0)['difference']*=-1
    elif kind=='missing_rate_pair':tables['paired_effects'].pop()
    else:next(x for x in tables['physical_threshold_windows'] if x['window']=='application')['exposure_seconds']=4.
    with pytest.raises(ValueError):r.verify_report_tables(tables,cases,logical,states)


def test_reader_verifies_both_json_and_tsv_and_rejects_tamper(tmp_path):
    rows=[dict(proposal_count=3,threshold=None,is_alias=True,rate=1.25)]
    (tmp_path/'tiny.json').write_text(json.dumps(rows))
    with (tmp_path/'tiny.tsv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    reader=r.Reader();bindings=[reader.bind(tmp_path/f'tiny.{ext}') for ext in ('json','tsv')]
    manifest=dict(artifacts=[dict(b,path=f'tiny.{ext}') for b,ext in zip(bindings,('json','tsv'))],table_row_counts=dict(tiny=1))
    assert reader.table(tmp_path,manifest,'tiny')==rows
    (tmp_path/'tiny.tsv').write_text('changed')
    with pytest.raises(ValueError):reader.table(tmp_path,manifest,'tiny')


@pytest.mark.parametrize('name',['Raw.npy','stages.npz','movie.mp4','experts.json','active.json'])
def test_reader_cannot_consume_arrays_media_or_activity_truth(tmp_path,name):
    path=tmp_path/name;path.write_bytes(b'not read')
    with pytest.raises(ValueError):r.Reader().bind(path)


def test_incomplete_evaluation_does_not_open_any_candidate_or_report(tmp_path):
    p=protocol_fixture();reader=r.Reader()
    (tmp_path/'protocol.json').write_text(json.dumps(p));protocol=reader.bind(tmp_path/'protocol.json')
    (tmp_path/'preflight.json').write_text(json.dumps(dict(status='PASS',protocol_sha256=protocol['sha256'])))
    (tmp_path/'computation_complete.json').write_text(json.dumps(dict(status='PASS',cells=252)))
    (tmp_path/'evaluation_complete.json').write_text(json.dumps(dict(status='RUNNING',cells=0)))
    with pytest.raises(ValueError,match='Evaluation is incomplete'):r._recount(tmp_path,reader)
    assert len(reader.bindings)==4


def test_completed_root_guard_never_rewrites_receipt(tmp_path):
    (tmp_path/'completion_manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='immutable'):r.recount(tmp_path)
    assert not (tmp_path/'validation').exists()
