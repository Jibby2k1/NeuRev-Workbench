"""Paired background/conditioned-variance report, without scoring or refitting.

Physical states are read once. Logical aliases are expanded only for paired
comparisons and never counted as additional replicates or acquisitions.
"""
from __future__ import annotations

import argparse
import itertools
import math
from collections import defaultdict
from pathlib import Path

from . import noise_report as noise
from . import reference_report as common

ROOT = Path(__file__).resolve().parents[3] / 'Outputs/GammaLSBackground/background_20260915_r1'
ARMS, SEEDS, FACTORS, RATE = noise.ARMS, noise.SEEDS, noise.FACTORS, noise.RATE
BACKGROUNDS, NORMALIZATIONS = ('sloped', 'flat'), ('raw', 'conditioned')
PANELS = tuple(itertools.product(BACKGROUNDS, NORMALIZATIONS))
WINDOWS = dict(setup=(65,164), early=(165,264), late=(265,464),
               settled_early=(215,264), settled_late=(315,464))
META = ('case_id','arm_id','seed','case_kind','background','normalization','V','S','T')
STATS = ('Raw_residual_second_moment','Raw_residual_variance',
         'Input_residual_second_moment','Input_residual_variance','C_variance',
         'Spread_squared_mean','floor_active_fraction','Z_q999')
require, Sources = common.require, common.Sources
_read, _write, _binding, _tsv = common._read, common._write, common._binding, common._tsv


def _logical_id(row):
    if 'case_id' in row:return row['case_id']
    if not row.get('alias',False):return row['canonical_case_id']
    return (f"logical__bg_{row['background']}__{row['normalization']}_"
            f"v{row['V']}_s{row['S']}_t{row['T']}__seed{row['seed']}")


def _matrix(protocol):
    cases = protocol['cases']; logical = protocol['logical_cases']
    require(len(cases)==len({c['case_id'] for c in cases})==84, 'Expected 84 unique physical cases')
    require(protocol['expected_cells']==252 and protocol['expected_curve_rows']==5040, 'Unexpected physical matrix')
    require(len(protocol['references'])==3 and {r['arm_id'] for r in protocol['references']}==set(ARMS), 'Reference inventory differs')
    require(len(logical)==len({_logical_id(c) for c in logical})==96, 'Expected 96 unique logical cases')
    physical = {c['case_id']:c for c in cases}
    expected = set(itertools.product(BACKGROUNDS,NORMALIZATIONS,(0,1),(0,1),(0,1),SEEDS))
    actual = {(c['background'],c['normalization'],c['V'],c['S'],c['T'],c['seed']) for c in logical}
    require(actual==expected, 'Incomplete logical background/normalization factorial')
    aliases=0
    for row in logical:
        require(row['canonical_case_id'] in physical, 'Unknown canonical physical case')
        canonical=physical[row['canonical_case_id']]
        for key in ('background','V','S','T','seed'):
            require(canonical[key]==row[key], 'Alias changed a scientific factor')
        if canonical['normalization']!=row['normalization']:
            require(row['normalization']=='conditioned' and canonical['normalization']=='raw' and
                    row['S']==row['T']==0, 'Only unit-gain normalization may alias raw')
            aliases+=1
        else: require(_logical_id(row)==row['canonical_case_id'], 'Non-identity redundant alias')
    require(aliases==12 and {r['canonical_case_id'] for r in logical}==set(physical), 'Alias count/coverage differs')
    for case in cases:
        require(Path(case['case_id']).name==case['case_id'] and case['case_id'] not in ('.','..'), 'Unsafe case ID')
    return [(c['case_id'],arm) for c in cases for arm in ARMS]


def expand_logical(rows, protocol):
    """Expand compact outcomes, retaining explicit canonical and reuse identity."""
    index=defaultdict(list)
    for row in rows: index[row['case_id']].append(row)
    physical={c['case_id']:c for c in protocol['cases']}; result=[]
    for logical in protocol['logical_cases']:
        canonical=logical['canonical_case_id']
        require(canonical in index, 'Missing physical outcome for logical row')
        for row in index[canonical]:
            result.append(dict(row,case_id=_logical_id(logical),canonical_case_id=canonical,
                background=logical['background'],normalization=logical['normalization'],
                is_alias=_logical_id(logical)!=canonical,
                source_reused=bool(physical[canonical].get('reused_dataset',False))))
    return result


def _load(root):
    sources=Sources(); p=sources.read(root/'protocol.json'); states=_matrix(p)
    pre=sources.read(root/'preflight.json')
    require(pre['status']=='PASS' and pre['protocol_sha256']==sources.bind(root/'protocol.json')['sha256'], 'Preflight binding differs')
    for b in p['code_bindings']+p['kernel_bindings']: sources.check(b)
    require(p['frame_rate_hz']==50 and p['pixel_size_um']==.5, 'Physical scale differs')
    require(all((root/'cells'/c/a/'sealed.json').is_file() for c,a in states), 'All 252 seals required before diagnostic/outcome reads')
    seals={key:sources.read(root/'cells'/key[0]/key[1]/'sealed.json') for key in states}
    require(all(s['status']=='SEALED_BEFORE_ACTIVITY_TRUTH_JOIN' for s in seals.values()), 'Incomplete seal')
    require(sources.read(root/'computation_complete.json')==dict(status='PASS',cells=252), 'Computation incomplete')
    inventory=sources.read(root/'all_scoring_seals.json')
    require(inventory['status']=='PASS' and inventory['cells']==252,'All-seal inventory incomplete')
    sources.check(inventory['protocol'],root/'protocol.json')
    expected_seals={(root/'cells'/c/a/'sealed.json').resolve() for c,a in states}
    require(len(inventory['seals'])==252 and {Path(b['path']).resolve() for b in inventory['seals']}==expected_seals,
            'All-seal inventory matrix differs')
    for b in inventory['seals']:sources.check(b)
    require(sources.read(root/'evaluation_complete.json')==dict(status='PASS',cells=252,curve_rows=5040), 'Evaluation incomplete')
    for key in ('baseline_completion','baseline_protocol','paper_protocol'):
        if key in p: sources.check(p[key])
    prep=sources.read(root/'datasets_complete.json')
    require(prep['status']=='PASS' and prep['datasets']==84, 'Physical preparation incomplete')
    cases={c['case_id']:c for c in p['cases']}; curves=[]; calibration=[]; operations={}; pairing={}; metadata={}
    for case,c in cases.items():
        require(len({tuple(seals[case,a]['stages'][k]['sha256'] for k in ('Raw','Input','A')) for a in ARMS})==1,
                'Input or target differs across references')
        files={Path(b['path']).name:b for b in seals[case,ARMS[0]]['dataset_bindings']}
        for name in ('metadata.json','experts.json','active.json'): sources.check(files[name])
        meta=sources.read(files['metadata.json']['path']); metadata[case]=meta
        require(meta['truth_mode']=='fully_synthetic' and meta['shape_tyx']==[464,226,226] and
                meta['source_frames_ui']==list(range(1,465)) and meta['setup_source_frames_ui']==list(range(65,165)) and
                meta['application_source_frames_ui']==list(range(165,465)) and meta['original_source_offset_xy']==[49,49],
                'Source geometry/chronology differs')
        background=meta['background']; gx,gy=(.03,.02) if c['background']=='sloped' else (0.,0.)
        require(background['offset']==100 and background['gradient_x_per_px']==gx and background['gradient_y_per_px']==gy,
                'Background metadata disagrees with factor')
        require(not sources.read(files['experts.json']['path']) and not sources.read(files['active.json']['path']), 'Expected empty synthetic truth')
    for (case,arm),seal in seals.items():
        folder=root/'cells'/case/arm; evaluated=sources.read(folder/'evaluated.json')
        require(evaluated['status']=='PASS','Cell evaluation incomplete');sources.check(evaluated['seal'],folder/'sealed.json')
        for b in evaluated['outputs']:sources.check(b)
        require((folder/'curves.json').resolve() in {Path(b['path']).resolve() for b in evaluated['outputs']},'Curves not bound')
        for key in ('calibration','threshold_plan'):sources.check(seal[key],folder/(key+'.json'))
        op=sources.read(folder/'calibration.json');plan=sources.read(folder/'threshold_plan.json')
        require(op['threshold_id']=='q1' and op['threshold_frozen_from_calibration_only'] and op['window']==3 and
                op['eligible_area_px']==13456 and op['setup_proposal_budget']==6, 'Operating point differs')
        require(len(plan)==10 and {r['threshold_id'] for r in plan}==set(common.THRESHOLDS),'Cutoff inventory differs')
        c=cases[case];key=(c['background'],c['seed'],arm);value=(plan,op['scale_floor'])
        require(key not in pairing or pairing[key]==value,'Setup plans/floors differ within background/seed/reference')
        pairing[key]=value; operations[case,arm]=op
        current=sources.read(folder/'curves.json');require(not current['event_rows'],'Null event outcomes exist')
        cr=current['curve_rows'];keys=[common._key(r) for r in cr]
        require(len(keys)==len(set(keys))==20 and set(keys)=={(case,arm,t,r) for t in common.THRESHOLDS for r in (2.,6.)},'Curve keys differ')
        settings={r['threshold_id']:r for r in plan}
        for row in cr:
            require(row['threshold_z']==settings[row['threshold_id']]['threshold'] and
                    row['event_count']==row['active_region_frame_count']==row['true_positive_count']==0 and
                    row['false_positive_count']==row['proposal_count'] and row['framewise_sensitivity'] is None and
                    row['event_window_coverage'] is None and row['eligible_area_px']==13456 and row['exposure_seconds']==6,
                    'Null curve partition/calibration/exposure differs')
        curves.extend(cr)
        calibration.extend(dict(r,case_id=case,arm_id=arm,scale_floor=op['scale_floor'],
                                **{k:c[k] for k in ('background','normalization','V','S','T','seed')}) for r in plan)
    aggregate=sources.read(root/'all_curves.json');sources.bind(root/'all_curves.tsv')
    require(len(aggregate)==len(curves)==5040 and {common._key(r):r for r in aggregate}=={common._key(r):r for r in curves},'Aggregate differs')
    require(len(pairing)==18,'Expected 18 background/seed/reference calibration groups')
    receipt=sources.read(root/'paired_setup_calibration_check.json')
    require(receipt['status']=='PASS' and len(receipt['groups'])==18 and
            {(r['background'],r['seed'],r['arm_id']) for r in receipt['groups']}==set(pairing) and
            all(r['cases']==14 and r['thresholds_equal'] and r['floors_equal'] for r in receipt['groups']),
            'Independent paired calibration receipt differs')
    replication=sources.read(root/'baseline_replication.json')
    reused={(c['case_id'],a) for c in cases.values() if c['reused_dataset'] for a in ARMS}
    require(replication['status']=='PASS' and len(reused)==len(replication['cells'])==72 and
            {(r['case_id'],r['arm_id']) for r in replication['cells']}==reused and
            all(all(r[k] for k in ('all_thresholds_equal','metrics_equal','stages_equal','candidates_equal')) for r in replication['cells']),
            'Exact reused baseline replication differs')
    return dict(protocol=p,sources=sources,seals=seals,cases=cases,metadata=metadata,operations=operations,
                curves=curves,calibration=calibration)


def residual_diagnostics(array, background, *, border=6,crop_origin=49,full_center=113):
    """Second moment around the known zero ensemble mean AND spatial variance."""
    import numpy as np
    require(array.ndim==3 and min(array.shape[1:])>2*border+1,'Invalid residual geometry')
    y,x=np.mgrid[border:array.shape[1]-border,border:array.shape[2]-border]
    bg=background['offset']+background['gradient_x_per_px']*(x+crop_origin-full_center)+background['gradient_y_per_px']*(y+crop_origin-full_center)
    result=[];previous=None
    for frame in array:
        u=np.asarray(frame[border:-border,border:-border],dtype=np.float64)-bg
        require(bool(np.isfinite(u).all()),'Nonfinite residual')
        h=noise._covariance(u[:,:-1],u[:,1:]);v=noise._covariance(u[:-1],u[1:]);temporal=noise._covariance(previous,u) if previous is not None else (None,None)
        result.append(dict(residual_mean=float(np.mean(u)),residual_second_moment=float(np.mean(u*u)),residual_variance=float(np.var(u)),
            spatial_lag1_covariance=(h[0]+v[0])/2,temporal_lag1_covariance=temporal[0],
            spatial_lag1_correlation=(h[1]+v[1])/2 if h[1] is not None and v[1] is not None else None,
            temporal_lag1_correlation=temporal[1]))
        previous=u
    return result


def stage_diagnostics(array, *, stage,floor,threshold,border=6):
    import numpy as np
    result=[]
    for frame in array:
        v=np.asarray(frame[border:-border,border:-border],dtype=np.float64).ravel()
        require(v.size and bool(np.isfinite(v).all()),'Invalid score stage')
        if stage=='C':row=dict(mean=float(v.mean()),second_moment=float(np.mean(v*v)),variance=float(np.var(v)))
        elif stage in ('Spread','Z'):
            q=np.quantile(v,[.5,.95,.99,.999]);row=dict(q50=float(q[0]),q95=float(q[1]),q99=float(q[2]),q999=float(q[3]),maximum=float(v.max()))
            if stage=='Spread':
                require(bool((v>=0).all()),'Negative spread')
                row.update(squared_mean=float(np.mean(v*v)),floor_active_fraction=float(np.mean(v<floor)))
            else:row['exceedance_fraction']=float(np.mean(v>threshold))
        else:raise ValueError(stage)
        result.append(row)
    return result


def threshold_windows(prefix,plan,*,case_id,arm_id):
    """Threshold sealed positive NMS rows; preserve no-output as an endpoint."""
    result=[]
    for setting in plan:
        tau=math.inf if setting['threshold'] is None else setting['threshold']
        counts=noise.proposal_frames(prefix,first=165,last=464,threshold=tau)
        for window,(lo,hi) in dict(application=(165,464),late=(265,464)).items():
            count=sum(counts[f] for f in range(lo,hi+1));exposure=(hi-lo+1)/50
            result.append(dict(case_id=case_id,arm_id=arm_id,threshold_id=setting['threshold_id'],
                threshold=setting['threshold'],window=window,source_start_ui=lo,source_stop_ui=hi,
                frame_count=hi-lo+1,proposal_count=count,false_positive_count=count,eligible_area_px=13456,
                exposure_seconds=exposure,**{RATE:10000*count/(3364*exposure)}))
    return result


def _per_frame(data):
    import numpy as np
    rows=[];curve_windows=[];shared={};sources=data['sources']
    expected={common._key(r):r for r in data['curves']}
    for number,((case,arm),seal) in enumerate(data['seals'].items(),1):
        c=data['cases'][case];op=data['operations'][case,arm];values={}
        for stage in ('Raw','Input','C','Spread','Z'):
            b=seal['stages'][stage];key=(b['path'],b['sha256'],stage,c['background'])
            if stage in ('Raw','Input') and key in shared:values[stage]=shared[key];continue
            sources.check(b);a=np.load(b['path'],mmap_mode='r',allow_pickle=False)
            require(a.shape==(464,128,128),'Diagnostic array shape differs')
            values[stage]=residual_diagnostics(a,data['metadata'][case]['background']) if stage in ('Raw','Input') else stage_diagnostics(
                a,stage=stage,floor=op['scale_floor'],threshold=op['threshold'])
            if stage in ('Raw','Input'):shared[key]=values[stage]
            del a
        for key in ('prefix','audit_candidates','setup_prefix'):sources.check(seal[key])
        prefix=sources.read(seal['prefix']['path']);audit=sources.read(seal['audit_candidates']['path'])
        app=noise.proposal_frames(audit,first=165,last=464,threshold=op['threshold'],already_filtered=True)
        require(app==noise.proposal_frames(prefix,first=165,last=464,threshold=op['threshold']),'Audit q1 prefix differs')
        setup=noise.proposal_frames(sources.read(seal['setup_prefix']['path']),first=65,last=164,threshold=op['threshold'])
        require(sum(app.values())==expected[case,arm,'q1',2.]['proposal_count'] and sum(setup.values())==op['setup_proposal_count'],'Q1 counts differ')
        plan=sources.read(seal['threshold_plan']['path'])
        window_rows=threshold_windows(prefix,plan,case_id=case,arm_id=arm)
        for r in window_rows:
            if r['window']=='application':require(r['proposal_count']==expected[case,arm,r['threshold_id'],2.]['proposal_count'],'Prefix threshold count differs from frozen curve')
            r.update({k:c[k] for k in ('background','normalization','V','S','T','seed')})
        curve_windows.extend(window_rows)
        for frame in range(65,465):
            row=dict(case_id=case,arm_id=arm,case_kind='factorial',**{k:c[k] for k in ('background','normalization','V','S','T','seed')},
                source_frame_ui=frame,application_relative_seconds=(frame-165)/50,
                threshold=op['threshold'],scale_floor=op['scale_floor'],proposal_count=(setup if frame<165 else app)[frame])
            for stage in values:row.update({stage+'_'+k:v for k,v in values[stage][frame-1].items()})
            row['floor_active_fraction']=row.pop('Spread_floor_active_fraction');rows.append(row)
        if number%3==0:print(dict(status='DIAGNOSTIC_CELLS_COMPLETE',cells=number,total=252),flush=True)
    return rows,curve_windows


def aggregate_windows(rows,windows=WINDOWS):
    groups=defaultdict(list)
    for row in rows:groups[row['case_id'],row['arm_id']].append(row)
    result=[]
    for _,group in sorted(groups.items()):
        byframe={r['source_frame_ui']:r for r in group}
        require(len(byframe)==len(group),'Duplicate frame statistic')
        for window,(lo,hi) in windows.items():
            require(all(f in byframe for f in range(lo,hi+1)),'Incomplete prescribed window')
            selected=[byframe[f] for f in range(lo,hi+1)];first=selected[0]
            row={k:first[k] for k in META};count=sum(r['proposal_count'] for r in selected);seconds=len(selected)/50
            row.update(window=window,epoch=window,source_start_ui=lo,source_stop_ui=hi,frame_count=len(selected),
                eligible_area_px=13456,exposure_seconds=seconds,proposal_count=count,threshold=first['threshold'],scale_floor=first['scale_floor'],
                **{RATE:10000*count/(3364*seconds)})
            for key in first:
                if key.startswith(('Raw_','Input_','C_','Spread_','Z_')) or key=='floor_active_fraction':
                    vals=[r[key] for r in selected if r[key] is not None];row[key]=sum(vals)/len(vals) if vals else None
            result.append(row)
    return result


def time_bins(rows):
    windows={str(i):(65+10*i,74+10*i) for i in range(40)}
    result=aggregate_windows(rows,windows)
    for r in result:r['time_mid_seconds']=((r['source_start_ui']+r['source_stop_ui'])/2-165)/50
    return result


def paired_effects(rows,metrics=(RATE,)+STATS):
    groups=defaultdict(dict)
    for r in rows:
        key=(r['seed'],r['arm_id'],r['window'],r['V'],r['S'],r['T']);panel=(r['background'],r['normalization'])
        require(panel not in groups[key],'Duplicate logical factor cell');groups[key][panel]=r
    result=[]
    for key,group in sorted(groups.items()):
        require(set(group)==set(PANELS),'Incomplete B/N paired factorial')
        seed,arm,window,v,s,t=key
        for metric in metrics:
            require(all(r[metric] is not None and math.isfinite(r[metric]) for r in group.values()),'Undefined paired statistic')
            contrasts=[('flat_minus_sloped',n,group['flat',n][metric]-group['sloped',n][metric]) for n in NORMALIZATIONS]
            contrasts += [('conditioned_minus_raw',b,group[b,'conditioned'][metric]-group[b,'raw'][metric]) for b in BACKGROUNDS]
            contrasts += [('background_by_normalization','difference_of_differences',
                (group['flat','conditioned'][metric]-group['flat','raw'][metric])-
                (group['sloped','conditioned'][metric]-group['sloped','raw'][metric]))]
            for contrast,at,difference in contrasts:result.append(dict(seed=seed,arm_id=arm,window=window,V=v,S=s,T=t,
                metric=metric,contrast=contrast,condition=at,difference=difference,replicate_unit='paired seed',logical_aliases_are_not_replicates=True))
    return result


def variance_theory_rows(rows,theory):
    """Stationary targets for single-phase windows, never a mixed six-second target."""
    g=theory['gaussian'];e=theory['ema'];result=[]
    for r in rows:
        lo,hi=r['source_start_ui'],r.get('source_stop_ui',r['source_start_ui'])
        require(lo<=hi and not any(lo<boundary<=hi for boundary in (165,265)),
                'Stationary targets require a single prescribed phase')
        sigma2=25. if r['V'] and r['source_start_ui']>=265 else 4.
        active=r['source_start_ui']>=165
        gain=(g['spatial_on_off_variance_ratio']**r['S'])*(e['temporal_on_off_variance_ratio']**r['T']) if active else 1.
        denominator=gain if active and r['normalization']=='conditioned' else 1.
        raw=sigma2/denominator;conditioned=raw*g['white_noise_energy']*e['white_noise_variance_factor']*gain
        result.append(dict(**{k:r[k] for k in ('case_id','canonical_case_id','arm_id','seed','background','normalization','V','S','T','window','is_alias')},
            raw_pointwise_variance_theory=raw,conditioned_stationary_pointwise_variance_theory=conditioned,
            measured_raw_second_moment=r['Raw_residual_second_moment'],measured_input_second_moment=r['Input_residual_second_moment'],
            measured_raw_spatial_variance=r['Raw_residual_variance'],measured_input_spatial_variance=r['Input_residual_variance'],
            stationary_not_exact_transition_prediction=True))
    return result


def verify_early_pairs(rows):
    groups=defaultdict(dict)
    for r in rows:
        if r['source_frame_ui']<=264:
            key=tuple(r[k] for k in ('background','normalization','seed','arm_id','S','T','source_frame_ui'))
            require(r['V'] not in groups[key],'Duplicate early V statistic');groups[key][r['V']]=r
    for pair in groups.values():
        require(set(pair)=={0,1},'Missing early V statistic')
        for key in pair[0]:
            if key.startswith(('Raw_','Input_','C_','Spread_','Z_')) or key in ('threshold','scale_floor','proposal_count','floor_active_fraction'):
                require(pair[0][key]==pair[1][key],'V changed before UI265')
    return dict(status='PASS',physical_frame_pairs=len(groups),scope='Exact consumed statistics/counts through UI264, not a new stage-array parity certificate')


def _audit_status(root,p,sources):
    if not (root/'audit_complete.json').exists():return dict(status='NUMERICAL_COMPLETE_AUDITS_PENDING',scientific_audit_complete=False)
    audit=sources.read(root/'audit_complete.json')
    require(audit['status']=='PASS' and audit['cells']==252 and audit['new_cells']==180 and audit['reused_cells']==72,'Audit counts differ')
    keys=[(r['case_id'],r['arm_id']) for r in audit['audits']]
    require(len(keys)==len(set(keys))==252 and set(keys)==set(_matrix(p)),'Physical audit matrix differs')
    for b in audit['validation_code']:sources.check(b)
    for key,name in (('protocol','protocol.json'),('evaluation','evaluation_complete.json'),('all_curves','all_curves.json'),
                     ('display_contract','display_contract.json'),('forecast','audit_forecast.json'),('baseline_replication','baseline_replication.json')):
        sources.check(audit[key],root/name)
    bindings=[]
    for row in audit['audits']:
        require(set(row['metadata_bindings'])==common.AUDIT_METADATA,'Audit metadata inventory differs')
        for name,b in row['metadata_bindings'].items():
            require(Path(b['path']).name==name,'Wrong metadata filename');bindings.append(sources.check(b))
        summary=sources.read(row['metadata_bindings']['summary.json']['path']);status=sources.read(row['metadata_bindings']['status.json']['path']);validation=sources.read(row['metadata_bindings']['validation.json']['path'])
        require(summary==row['summary'] and summary['scientific_audit_complete'] is True and status['status']=='complete' and
                status['scientific_audit_complete'] is True and validation['status']=='passed' and not validation['failures'],'Incomplete audit metadata')
    return dict(status='AUDITS_VERIFIED_VISUAL_QA_SEPARATE',scientific_audit_complete=True,aggregate=sources.bind(root/'audit_complete.json'),
                metadata_bindings=bindings,validation_code=audit['validation_code'])


def _native_curve_rows(rows):
    """One seed's frozen cutoff rows; never average cutoffs across seeds."""
    require(len(rows)==10 and {r['threshold_id'] for r in rows}==set(common.THRESHOLDS),
            'Native-cutoff curve inventory differs')
    by_id={r['threshold_id']:r for r in rows}
    for r in rows:
        require(r['threshold'] is None or (math.isfinite(r['threshold']) and r['threshold']>=0),
                'Invalid native cutoff')
    require(by_id['all_positive']['threshold']==0 and by_id['no_output']['threshold'] is None,
            'Native-cutoff endpoint semantics differ')
    finite=sorted((r for r in rows if r['threshold_id'] not in ('all_positive','no_output') and
                   r['threshold'] is not None),key=lambda r:(r['threshold'],r['threshold_id']))
    q1=by_id['q1'] if by_id['q1']['threshold'] is not None else None
    return finite,q1,by_id['all_positive']


def _plot(out,tables):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10})
    figures=[];colors=('#333333','#2471A3','#C57E12','#7E477F')
    epochs=tables['logical_epochs'];labels=[''.join(map(str,f)) for f in FACTORS]
    def save(fig,stem,caption,uses):
        fig.tight_layout(rect=(0,.025,1,.95));figures.append(common._save(fig,out,stem,caption,uses))
    def mean_seeds(group,key):
        require(len(group)==3 and {r['seed'] for r in group}==set(SEEDS),'Missing/duplicate plot seed')
        return [r[key] for r in sorted(group,key=lambda r:r['seed'])]
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey='row')
    for j,arm in enumerate(ARMS):
        for i,(key,label) in enumerate((('Raw_residual_second_moment','Raw residual second moment'),('Input_residual_second_moment','Conditioned residual second moment'))):
            ax=axes[i,j]
            for color,panel in zip(colors,PANELS):
                means=[]
                for x,f in enumerate(FACTORS):
                    g=[r for r in epochs if r['arm_id']==arm and r['window']=='settled_late' and (r['background'],r['normalization'])==panel and tuple(r[k] for k in 'VST')==f]
                    y=mean_seeds(g,key);means.append(np.mean(y));ax.scatter(x+np.linspace(-.06,.06,3),y,color=color,s=12,marker='x',alpha=.45)
                ax.plot(range(8),means,color=color,label=' / '.join(panel),marker='o',ms=3)
            if i==0:ax.set_title(noise.ARM_LABELS[arm])
            if j==0:ax.set_ylabel(label+'\n(native units²)')
            ax.set_xticks(range(8),labels);ax.set_xlabel('VST factors');ax.set_yscale('symlog',linthresh=.01)
    for i in range(2):axes[i,0].set_ylim(bottom=0)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=4);common._style(axes)
    save(fig,'variance_matching','UI315–464, prescribed settled-late diagnostic. Known background removed; second moment is about ensemble mean zero, not a spatially demeaned variance. Crosses are paired seeds. Stationary targets are theoretical; finite windows and float32 arithmetic need not match exactly.',['logical_epochs','variance_theory'])
    fig,axes=plt.subplots(4,3,figsize=(15,12),sharex=True,sharey=True)
    for i,panel in enumerate(PANELS):
        for j,arm in enumerate(ARMS):
            ax=axes[i,j];means=[]
            for x,f in enumerate(FACTORS):
                g=[r for r in epochs if r['arm_id']==arm and r['window']=='late' and (r['background'],r['normalization'])==panel and tuple(r[k] for k in 'VST')==f]
                y=mean_seeds(g,RATE);means.append(np.mean(y));ax.scatter(x+np.linspace(-.1,.1,3),y,color=colors[i],marker='x',s=20)
            ax.plot(range(8),means,color=colors[i],marker='o',ms=4);ax.set_xticks(range(8),labels);ax.set_yscale('symlog',linthresh=1)
            if i==0:ax.set_title(noise.ARM_LABELS[arm])
            if j==0:ax.set_ylabel(' / '.join(panel)+'\nProposals / 10⁴ µm² / s')
            if i==3:ax.set_xlabel('VST factors')
    axes[0,0].set_ylim(bottom=0)
    common._style(axes);save(fig,'late_false_burden','Primary frozen q1 late window UI265–464 (4 s). Dots/lines summarize three paired seeds; aliases are repeated factor conditions, not new replicates. Equal setup budgets are not equal application false-alarm rates.',['logical_epochs'])
    for window in ('application','late'):
        fig,axes=plt.subplots(4,3,figsize=(15,12),sharex=True,sharey=True)
        for i,panel in enumerate(PANELS):
            for j,arm in enumerate(ARMS):
                ax=axes[i,j]
                for f in FACTORS:
                    color=colors[2*f[1]+f[2]];style='-' if f[0] else '--'
                    group=[r for r in tables['logical_threshold_windows'] if r['arm_id']==arm and r['window']==window and
                        (r['background'],r['normalization'])==panel and tuple(r[k] for k in 'VST')==f]
                    if window=='application':
                        y=[np.mean(mean_seeds([r for r in group if r['threshold_id']==threshold],RATE))
                           for threshold in common.THRESHOLDS]
                        ax.plot(range(8),y[:8],color=color,ls=style,lw=1,label=''.join(map(str,f)))
                        ax.scatter([8,9],y[8:],color=color,s=15,marker='o' if f[0] else 'x')
                    else:
                        require(len(group)==30 and {r['seed'] for r in group}==set(SEEDS),'Native-cutoff plot seeds differ')
                        for seed_index,seed in enumerate(SEEDS):
                            finite,q1,all_positive=_native_curve_rows([r for r in group if r['seed']==seed])
                            ax.plot([r['threshold'] for r in finite],[r[RATE] for r in finite],color=color,
                                ls=style,lw=.65,alpha=.7,label=''.join(map(str,f)) if seed_index==0 else '_nolegend_')
                            if q1 is not None:
                                ax.scatter([q1['threshold']],[q1[RATE]],color=color,s=21,marker='D',zorder=4)
                            ax.scatter([0],[all_positive[RATE]],color=color,s=14,marker='x',alpha=.65)
                ax.set_yscale('symlog',linthresh=1)
                if window=='application':ax.set_xticks(range(10),('0','.25','.5','1','2','4','8','16','all+','none'),rotation=35)
                if i==0:ax.set_title(noise.ARM_LABELS[arm])
                if j==0:ax.set_ylabel(' / '.join(panel)+'\nProposals / 10⁴ µm² / s')
                if i==3:ax.set_xlabel('Setup q / distinct endpoints' if window=='application' else 'Native Z cutoff')
        axes[0,0].set_ylim(bottom=0)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=8,
            title='VST factors' if window=='application' else 'VST factors · each line is one seed · diamonds mark frozen q1');common._style(axes)
        caption=('All ten frozen plans over the full 6-s application. The x axis is setup q; native cutoffs differ across seeds, references and backgrounds. Three-seed mean rates. All-positive/no-output are separate endpoints, not points at finite q. No application cutoff selected.' if window=='application' else
                 'Late window UI265–464 (4 s), plotted against each seed\'s actual finite native Z cutoffs. Each thin line is one seed; colors encode S/T and line style V. Cutoffs are never averaged across seeds. Diamonds highlight the predeclared q1 setting wherever finite. All-positive cutoff zero is an isolated cross; infinite/None no-output endpoints are omitted from this finite x axis and remain in the source table. Only frozen setup-derived cutoffs are displayed; no application cutoff selected. Equal numerical Z cutoffs need not imply equal false-alarm probabilities.')
        save(fig,'thresholds_'+window,caption,['logical_threshold_windows'])
    for v in (0,1):
        fig,axes=plt.subplots(4,3,figsize=(15,12),sharex=True,sharey=True)
        for i,panel in enumerate(PANELS):
            for j,arm in enumerate(ARMS):
                ax=axes[i,j]
                for color,(s,t) in zip(colors,itertools.product((0,1),repeat=2)):
                    group=[r for r in tables['logical_time_bins'] if r['arm_id']==arm and (r['background'],r['normalization'])==panel and (r['V'],r['S'],r['T'])==(v,s,t)]
                    seq=[sorted([r for r in group if r['seed']==seed],key=lambda r:r['time_mid_seconds']) for seed in SEEDS]
                    require(all(len(g)==40 for g in seq),'Transient bin inventory differs');x=[r['time_mid_seconds'] for r in seq[0]];ys=np.array([[r[RATE] for r in g] for g in seq])
                    for y in ys:ax.plot(x,y,color=color,alpha=.15,lw=.6)
                    ax.plot(x,ys.mean(axis=0),color=color,lw=1.4,label=f'S{s} T{t}')
                ax.axvspan(-2,0,color='.9');ax.axvline(0,color='black',lw=.6);ax.axvline(2,color='black',lw=.6,ls=':');ax.set_yscale('symlog',linthresh=1)
                if i==0:ax.set_title(noise.ARM_LABELS[arm])
                if j==0:ax.set_ylabel(' / '.join(panel)+'\nProposals / 10⁴ µm² / s')
                if i==3:ax.set_xlabel('Time from application start (s)')
        axes[0,0].set_ylim(bottom=0)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=4,title=f'V={v}');common._style(axes)
        save(fig,f'transitions_v{v}','Fixed 200-ms bins, all three seed trajectories and their mean. S/T start at 0 s; V=1 changes amplitude at 2 s. Stationary variance matching does not remove switching transients.',['logical_time_bins'])
    return figures


def _text(tables,figures,audit):
    lines=['# Background and conditioned-variance controls','',
        '252 physical states (84 datasets × 3 references) are complete. The crossed design has 288 logical states; 36 are exact unit-gain aliases. Seventy-two physical states reuse the original sloped/raw factorial. These are paired development controls, not independent confirmations.','',
        'Scientific media: '+('complete with current metadata verified.' if audit['scientific_audit_complete'] else 'pending; numerical completion does not imply audit completion.'),'',
        'Primary comparison: q1 setup-frozen false proposals per 10,000 µm² per second during UI265–464. Each seed uses the same 3,364 µm² field. All clips are synthetic nulls; sensitivity/event recovery are undefined. No optimal reference, biological precision or control-loop claim follows.','',
        'Background is either 100 or the original 100+0.03x+0.02y with centered coordinates, throughout warmup/setup/application. The innovation stream and stationary AR initializer are paired. Setup UI65–164 stays white with raw standard deviation 2. Application noise alone is divided by sqrt(kS^S kT^T) in conditioned matching; kS=5.2140943812 and kT=2.8461538462. Neither the background nor any source amplitude is scaled. V still changes nominal standard deviation 2→5 at UI265. This matches stationary conditioned marginal variance. Spatial matching does not generally preserve the joint field law. At fixed S and noise amplitude, ideal stationary Gaussian/separable temporal matching also preserves the same-frame field law and the distribution of an identical detector with a shared fixed floor/cutoff. Transients, finite samples and float32 arithmetic qualify this ideal statement.','',
        'Floors/cutoffs are identical within background/seed/reference across normalization and VST. Flat and sloped backgrounds are calibrated separately; that comparison includes their setup adaptation. Floors remain active in the detector. No application statistics fit the normalization or thresholds.','',
        'Setup/early/late windows are UI65–164/165–264/265–464. Secondary settled windows UI215–264 and 315–464 discard 50 frames (1 s) after each switch. These fixed windows reduce transients; they do not assert exact stationarity. Ten-frame (200-ms) bins retain the transitions.','',
        'Raw/Input second moments subtract the known ensemble background but not each frame mean. Spatial variances additionally remove each residual frame mean (ddof0). Their difference is scientifically relevant under correlation. C variance is separately measured; Spread² is reference spatial variability and includes background structure. Z quantiles are means of per-frame quantiles, not pooled pixel quantiles. Float32 production arithmetic is retained.','',
        'Paired flat-minus-sloped and conditioned-minus-raw differences are computed within VST/reference/seed. The B×N difference of differences is descriptive. Three seeds are the replicates; aliases and adjacent frames are not extra replicates. Native thresholds and no-output/all-positive endpoints remain explicit.','',
        '[Physical epoch rows](physical_epochs.tsv) · [Logical paired rows](logical_epochs.tsv) · [Paired contrasts](paired_effects.tsv) · [All thresholds and windows](logical_threshold_windows.tsv) · [Stationary variance checks](variance_theory.tsv) · [Per-frame diagnostics](per_frame.tsv)','']
    for f in figures:lines.extend([f"![{f['id']}]({f['png']})",'',f['caption'],'',f"[PDF]({f['pdf']})",''])
    lines.extend(['[Protocol](../protocol.json) · [Evaluation](../evaluation_complete.json) · [Sources](sources_manifest.json) · [Manifest](manifest.json)',''])
    return '\n'.join(lines)


def _qa(out,manifest,sources):
    if not (out/'visual_qa.json').exists():return dict(visual_qa_complete=False,visual_qa_binding=None)
    qa=sources.read(out/'visual_qa.json');require(qa['status']=='PASS','Visual QA failed')
    expected={(out/f['png']).resolve() for f in manifest['figures']};actual=[Path(b['path']).resolve() for b in qa['figures']]
    require(len(actual)==len(set(actual))==len(expected) and set(actual)==expected,'QA figure inventory differs')
    for b in qa['figures']:sources.check(b)
    if 'pdf_companion_receipt' in qa:sources.check(qa['pdf_companion_receipt'])
    return dict(visual_qa_complete=True,visual_qa_binding=sources.bind(out/'visual_qa.json'))


def report(root=ROOT):
    root=common._mutable(root);out=root/'report';require(not out.exists() or not any(out.iterdir()),'Previous report must be preserved')
    data=_load(root);per_frame,thresholds=_per_frame(data);pairing=verify_early_pairs(per_frame)
    physical=aggregate_windows(per_frame);logical=expand_logical(physical,data['protocol'])
    bins=time_bins(per_frame)
    gain=data['protocol']['matching_gains'][0]
    theory=dict(gaussian=gain['gaussian'],ema=gain['ema'])
    tables=dict(curves=data['curves'],calibration=data['calibration'],per_frame=per_frame,physical_epochs=physical,logical_epochs=logical,
        paired_effects=paired_effects(logical),physical_threshold_windows=thresholds,logical_threshold_windows=expand_logical(thresholds,data['protocol']),
        physical_time_bins=bins,logical_time_bins=expand_logical(bins,data['protocol']),variance_theory=variance_theory_rows(logical,theory))
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():_write(out/(name+'.json'),rows);_tsv(out/(name+'.tsv'),rows)
    _write(out/'early_variance_pairing.json',pairing)
    figures=_plot(out,tables);inputs=data['sources'].finish();audit_sources=Sources();audit=_audit_status(root,data['protocol'],audit_sources);audit_sources.finish()
    dependencies=[_binding(Path(module.__file__)) for module in (noise,common)];dependencies.insert(0,_binding(Path(__file__)))
    _write(out/'sources_manifest.json',dict(status='VERIFIED_CONSUMED_SOURCES',inputs=inputs,reporting_code=dependencies,
        scope='Every consumed stage/prefix and metadata checked by SHA256. No scoring/refitting. Logical aliases never read duplicate data or create extra replicates.'))
    (out/'REPORT.md').write_text(_text(tables,figures,audit))
    manifest=dict(schema_version=1,status='GENERATED',numerical_complete=True,physical_cell_count=252,logical_cell_count=288,
        scientific_audit_complete=audit['scientific_audit_complete'],report_stage=audit['status'],audit_verification=audit,
        visual_qa_complete=False,visual_qa_binding=None,figure_count=len(figures),figures=figures,
        table_row_counts={k:len(v) for k,v in tables.items()},inputs=inputs,reporter=dependencies[0],reporting_code=dependencies,
        artifacts=[dict(_binding(p),path=p.name) for p in sorted(out.iterdir()) if p.is_file()])
    _write(out/'manifest.json',manifest);return manifest


def update(root=ROOT):
    root=common._mutable(root);out=root/'report';m=_read(out/'manifest.json');sources=Sources()
    sources.check(m['reporter'],Path(__file__))
    for b in m['reporting_code']+m['inputs']:sources.check(b)
    for b in m['artifacts']:sources.check(dict(b,path=str(out/b['path'])))
    p=sources.read(root/'protocol.json');audit=_audit_status(root,p,sources);qa=_qa(out,m,sources)
    tables={name:_read(out/(name+'.json')) for name in m['table_row_counts']};sources.finish()
    (out/'REPORT.md').write_text(_text(tables,m['figures'],audit))
    m.update(scientific_audit_complete=audit['scientific_audit_complete'],report_stage=audit['status'],audit_verification=audit,**qa)
    m['artifacts']=[dict(_binding(out/b['path']),path=b['path']) for b in m['artifacts']]
    _write(out/'manifest.json',m);return m


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('report','update'));parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();print(globals()[args.command](args.root)['status'],flush=True)


if __name__=='__main__':main()
