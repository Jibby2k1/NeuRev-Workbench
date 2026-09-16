"""Complete, resumable scientific media for the regional/crowding follow-up."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .followup_study import ROOT, BASE, load, cells, read, write_json, binding, verify, sha256, frame_sets, link
from .followup_audit import run_spatiotemporal_audit, audit_inventory_plan
from .two_stencil_audit import verify_completed_audit


def reused(c):
    return c['study']=='regional' and c['calibration_method']=='global'


def display(root):
    if (root/'display_contract.json').exists():return read(root/'display_contract.json')
    p=load(root);existing=read(BASE/'display_contract.json')['cases'];result={c:existing[c] for c in p['regional_cases']}
    raw_low=1e30;raw_high=-1e30;level_high=1.;c_high=1e-6
    for c in p['crowding_cases']:
        case=c['case_id'];raw=np.load(root/'datasets'/case/'Raw.npy',mmap_mode='r')
        lo,hi=map(float,np.percentile(raw[::5],[1,99.8]));raw_low=min(raw_low,lo);raw_high=max(raw_high,hi)
        st=read(root/'stages'/case/'level/complete.json')['stages']
        for key in ('X','A','C'):
            v=np.load(st[key]['path'],mmap_mode='r')[::5]
            if key=='C':c_high=max(c_high,float(np.percentile(np.abs(v),99.8)))
            else:level_high=max(level_high,float(np.percentile(v,99.8)))
    common=dict(Raw=[raw_low,raw_high],level=dict(X=[0.,level_high],A=[0.,level_high],C=[-c_high,c_high],Z=[-8.,8.]))
    for c in p['crowding_cases']:result[c['case_id']]=common
    contract=dict(cases=result,rule='Regional arms reuse baseline fixed native-unit ranges. All12crowding scenes share ranges; X/A share fluorescence units, C symmetric fluorescence units, Z symmetric[-8,8]. Display quantiles do not enter detection.',baseline_display=binding(BASE/'display_contract.json'))
    write_json(root/'display_contract.json',contract)
    return contract


def forecast(root):
    p=load(root);result=[]
    if read(root/'evaluation_complete.json')['cells']!=242:raise RuntimeError('Evaluate every cell first')
    display(root)
    for c in cells(p):
        out=root/'cells'/c['case_id']/c['arm_id'];folder=root/'datasets'/c['case_id']
        frames,_,_=frame_sets(read(folder/'metadata.json'))
        plan=audit_inventory_plan(read(out/'audit_candidates.json'),read(folder/'experts.json'),frames)
        result.append(dict(c,reused=reused(c),expected_videos=plan['expected_video_count'],expert_rois=plan['expert_roi_count'],model_rois=plan['model_roi_count'],expert_occurrences=plan['expert_occurrence_count']))
    value=dict(status='PASS',cells=result,new_cells=sum(not r['reused'] for r in result),new_videos=sum(r['expected_videos'] for r in result if not r['reused']),total_videos=sum(r['expected_videos'] for r in result))
    write_json(root/'audit_forecast.json',value)
    print({k:v for k,v in value.items() if k!='cells'},flush=True)


def media(root,worker=0,workers=3):
    p=load(root);limits=read(root/'display_contract.json')['cases'];count=0
    if not 0<=worker<workers<=3:raise ValueError('At most three disjoint media workers')
    for index,c in enumerate(cells(p)):
        if index%workers!=worker:continue
        case=c['case_id'];folder=root/'datasets'/case;out=root/'cells'/case/c['arm_id'];audit=root/'audits'/case/c['arm_id']
        if reused(c):
            old=BASE/'audits'/case/c['base_arm']
            audit.parent.mkdir(parents=True,exist_ok=True);link(old,audit)
            # Exact global score/threshold/candidate semantic replication is checked
            # numerically before this reuse, and all media hashes at closure.
            count+=1;continue
        seal=read(out/'sealed.json');mode=c['input_mode'];frames,_,_=frame_sets(read(folder/'metadata.json'))
        names=['Raw','Input']+(['A'] if c['readout'] in ('C','Z') else [])+['Score']
        stages={k:seal['stages'][k]['path'] for k in names}
        dl=dict(Raw=limits[case]['Raw'],Input=limits[case][mode]['X'],Score=limits[case][mode][c['readout']])
        if 'A' in names:dl['A']=limits[case][mode]['A']
        source=dict(stage_sha256={k:seal['stages'][k]['sha256'] for k in names},
            candidates_sha256=seal['audit_candidates']['sha256'],expert_occurrences_sha256=sha256(folder/'experts.json'),
            candidate_seal_sha256=sha256(out/'sealed.json'),display_contract_sha256=sha256(root/'display_contract.json'),
            numeric_stage_bindings={k:seal['stages'][k] for k in ('M','Spread','C','Z') if seal['stages'][k]['path'] not in stages.values()},
            original_source_offset_xy=read(folder/'metadata.json')['original_source_offset_xy'],truth_mode=read(folder/'metadata.json')['truth_mode'],
            cell=c,stage_semantics='Raw fluorescence; Input causal conditioned level/change; optional A centered target; Score original untransformed readout. Regional cutoffs applied AFTER unchanged NMS; no margin reranking. Model sites are review locations. Comparison cutoff line applies to primary expert trace; nearest cutoff recorded separately.')
        summary=run_spatiotemporal_audit(audit,stage_paths=stages,signed_stage_keys=[k for k in names if dl[k][0]<0],
            display_limits=dl,source_frames_ui=frames,source_binding=source,operating_point=read(out/'calibration.json'),
            numeric_threshold_paths={'curves':out/'curves.json','calibration':out/'calibration.json'},
            candidates_path=out/'audit_candidates.json',expert_occurrences=folder/'experts.json',fps=50.)
        count+=1
        print(dict(status='AUDIT_COMPLETE',worker=worker,case_id=case,arm_id=c['arm_id'],videos=summary['video_count']),flush=True)
    write_json(root/f'media_worker_{worker}.json',dict(status='PASS',worker=worker,workers=workers,cells=count))


def validate(root):
    p=load(root);forecast_rows={(r['case_id'],r['arm_id']):r for r in read(root/'audit_forecast.json')['cells']};result=[]
    replication=read(root/'baseline_replication.json')
    if replication['status']!='PASS' or len(replication['cells'])!=85:raise RuntimeError('Global replication incomplete')
    for c in cells(p):
        audit=root/'audits'/c['case_id']/c['arm_id'];verify_completed_audit(audit)
        summary=read(audit/'summary.json');validation=read(audit/'validation.json')
        if not summary['scientific_audit_complete'] or validation['status']!='passed':raise RuntimeError('Audit incomplete')
        expected=forecast_rows[(c['case_id'],c['arm_id'])]
        if summary['video_count']!=expected['expected_videos']:raise RuntimeError('Inventory forecast differs')
        if reused(c):
            new=root/'cells'/c['case_id']/c['arm_id'];old=BASE/'cells'/c['case_id']/c['base_arm']
            n=read(new/'sealed.json');o=read(old/'sealed.json')
            for k in ('Raw','Input','Score'):
                if n['stages'][k]['sha256']!=o['stages'][k]['sha256']:raise RuntimeError('Reused audit stage differs')
            fields=('proposal_id','source_frame_ui','x_px','y_px','score','threshold_z','candidate_rank_within_frame')
            nr=read(new/'audit_candidates.json');oldr=read(old/'audit_candidates.json')
            if len(nr)!=len(oldr) or any(any(a.get(k)!=b.get(k) for k in fields) for a,b in zip(nr,oldr)):raise RuntimeError('Reused candidates differ')
        result.append(dict(c,reused=reused(c),summary=summary,metadata_bindings={n:binding(audit/n) for n in ('summary.json','status.json','validation.json','inventory.json','artifact_index.json','source_manifest.json','run_contract.json','llm_context.json')}))
    write_json(root/'audit_complete.json',dict(status='PASS',cells=len(result),new_cells=157,reused_cells=85,audits=result))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['forecast','media','validate']);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    if a.command=='media':media(a.root.resolve(),a.worker,a.workers)
    else:globals()[a.command](a.root.resolve())


if __name__=='__main__':main()
