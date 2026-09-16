"""Mean-distance/order reference sensitivity, with an exactly fixed target."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import numpy as np

from neurobench.algorithms.gamma_spatiotemporal import GammaSTSpec, GammaSTKernels, build_kernels, iter_chunks
from .necessity_study import read, binding, verify, write_json, write_tsv, sha256, threshold_plan, deadline_rows, prepare_mode, emit, link
from .spatiotemporal_study import REPO, frame_sets, mmap, REFERENCE_AREA_PX
from .followup_study import verify_seal
from .followup_selection import extract_candidates
from .followup_calibration import select
from .spatiotemporal_metrics import seal_candidates, evaluate_threshold_sweep, evaluate_framewise

BASE = REPO / 'Outputs/GammaLSFollowup/followup_20260914_r1'
ROOT = REPO / 'Outputs/GammaLSReference/reference_20260915_r1'
PAPER = REPO.parent / 'Neural_Event_Extraction_Gamma_LS_Clarity_Revision_2026-09-12/editorial'
SEEDS = (20260916, 20260917, 20260918)
ANCHOR = 'mean1_n9'


def cells(p):
    return [dict(case_id=c['case_id'], arm_id=r['arm_id'], study='reference', input_mode='level',
                 readout='Z', window=3, calibration_method='global', baseline_arm='level_Z__w3',
                 reused_audit=c['kind']=='crowding' and r['arm_id']==ANCHOR,
                 case_kind=c['kind'], seed=c['seed'], separation_px=c.get('separation_px'))
            for c in p['cases'] for r in p['references']]


def guard_open(root):
    if (root/'completion_manifest.json').exists():
        raise RuntimeError('Preserve completed reference study')


def preflight(root):
    from .reference_geometry import reference_grid
    from .reference_fixtures import null_specs
    from .followup_fixtures import case_id
    if root.exists():raise FileExistsError('New noncolliding root required')
    completion=read(BASE/'completion_manifest.json')
    if completion['status']!='PASS':raise RuntimeError('Completed baseline required')
    old=read(BASE/'protocol.json')
    for b in old['code_bindings']:verify(b)
    authority={str(Path(b['path']).resolve()):b for b in completion['evidence_bindings']}
    crowding=[dict(case_id=case_id(seed,sep),seed=seed,separation_px=sep,kind='crowding')
              for seed in SEEDS for sep in (None,8,12,16)]
    baseline_sources=[]
    for c in crowding:
        case=c['case_id']
        for f in [BASE/'datasets'/case/n for n in ('metadata.json','experts.json','active.json','prepared.json')] + [BASE/'stages'/case/'level/complete.json',BASE/'cells'/case/'level_Z__w3/sealed.json']:
            # Per-cell stages are also transitively bound through the sealed record.
            b=authority.get(str(f.resolve()))
            if b is None and f.name=='complete.json':
                b=binding(f)
                for v in read(f)['stages'].values():
                    known=authority.get(str(Path(v['path']).resolve()))
                    if known is None:raise RuntimeError('Unbound baseline stage')
                    verify(known)
            elif b is None:raise RuntimeError(f'Baseline source outside completion: {f}')
            verify(b);baseline_sources.append(b)
    if shutil.disk_usage(REPO).free<120*1024**3:raise RuntimeError('Need120GiB free disk')
    kernels=reference_grid(anchor_reference=build_kernels(GammaSTSpec()).reference[0])
    refs=[dict(arm_id=k.spec.spec_id, **k.metadata) for k in kernels]
    nulls=[dict(case_id=c['case_id'],seed=c['seed'],kind='null',null_kind=c['kind']) for c in null_specs()]
    code=[Path(__file__),Path(__file__).with_name('reference_geometry.py'),Path(__file__).with_name('reference_fixtures.py')]
    code += [Path(b['path']) for b in old['code_bindings']]
    p=dict(schema_version=1,experiment='gamma_reference_mean_distance_order',
           baseline_completion=binding(BASE/'completion_manifest.json'),baseline_protocol=binding(BASE/'protocol.json'),
           baseline_sources=baseline_sources,code_bindings=[binding(f) for f in dict.fromkeys(code)],
           crowding_cases=crowding,null_cases=nulls,cases=crowding+nulls,references=refs,
           expected_cells=189,new_audits=177,reused_audits=12,expected_curve_rows=3780,
           git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
           initial_git_status=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True),
           geometry='3 discrete radial-mass means (2/3,1,4/3 of exact baseline) crossed with orders3,9,15; fixed finite support per solve; retention>=.995; no hard mask.',
           conditioning='Gaussian sigma1px truncate4 reflect, causal EMA alpha.4, level input, spatial-only direct reference moments.',
           target='Exact baseline target response reused across all9 references per dataset; never rebuilt on a reference-dependent support.',
           calibration='Per-kernel10th percentile positive setup spread floor, setup-only global NMS order statistics; strict score>cutoff; q0,.25,.5,1,2,4,8,16 plus all_positive/no_output.',
           selector='3x3 local maximum,6pxborder,score/y/x rank,greedy Euclidean separation strictly>6px.',
           primary='q1 weak recovery within100ms and inclusive2px, all events in denominator; active-frame coverage; neighbor recovery; null false proposals per area-time.',
           secondary='All budget/deadline curves;6px matching separately; synthetic precision, duplicates, localization and floor activation.',
           frame_rate_hz=50,pixel_size_um=.5,kinetics='Provisional100ms rise,1000ms decay, finite1%tail; not measured indicator kinetics.',
           scientific_audit=dict(enabled=True,operating_point='q1',policy='Every expert/modelfullfield,ROIvideo and trace,occurrencecomparison;12identical baselineaudits reused after exact replication;177newcompleteaudits.'),
           scope='Synthetic development mechanism study only. Three seeds are replicates. No independent validation, biological precision, optimized radius claim, constant application false-alarm guarantee or feedback-control claim.',
           resources=dict(cpu='one numerical thread,chunks8,lowpriority,CPUs6/7excluded',media='atmost3workers,onecodec thread each',free_disk_bytes=shutil.disk_usage(REPO).free))
    root.mkdir(parents=True);(root/'kernels').mkdir()
    for k in kernels:
        np.save(root/'kernels'/f'{k.spec.spec_id}.npy',k.weights)
    p['kernel_bindings']=[binding(f) for f in sorted((root/'kernels').glob('*.npy'))]
    write_json(root/'protocol.json',p);write_json(root/'preflight.json',dict(status='PASS',protocol_sha256=sha256(root/'protocol.json')))
    emit(root,status='PREFLIGHT_PASS',cells=len(cells(p)))


def load(root):
    p=read(root/'protocol.json')
    if sha256(root/'protocol.json')!=read(root/'preflight.json')['protocol_sha256']:raise RuntimeError('Frozen protocol changed')
    for b in p['code_bindings']+p['kernel_bindings']:verify(b)
    for b in [p['baseline_completion'],p['baseline_protocol']]+p['baseline_sources']:verify(b)
    return p


def prepare(root):
    from .reference_fixtures import build_null_dataset
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    guard_open(root);p=load(root)
    for c in p['crowding_cases']:
        case=c['case_id'];(root/'datasets').mkdir(exist_ok=True);(root/'stages'/case).mkdir(parents=True,exist_ok=True)
        link(BASE/'datasets'/case,root/'datasets'/case)
        link(BASE/'stages'/case/'level',root/'stages'/case/'level')
    for c in p['null_cases']:build_null_dataset(root,c['seed'],c['null_kind'])
    records=[];datasets={};pairs=[]
    for c in p['cases']:
        case=c['case_id'];folder=root/'datasets'/case
        names=('metadata.json','experts.json','active.json','prepared.json','level_prepared.json','raw_source.npy','Raw.npy','level_source.npy','level.npy')
        datasets[case]=[binding(folder/n) for n in names]
        raw=np.load(folder/'raw_source.npy',mmap_mode='r');anchor=np.load(BASE/'datasets'/f"crowding_alone__seed{c['seed']}"/'raw_source.npy',mmap_mode='r')
        if not np.array_equal(raw[:164],anchor[:164]):raise RuntimeError('Shared setup differs')
        pairs.append(dict(case_id=case,setup_byte_equal=True))
        meta=read(folder/'metadata.json');xy=sorted({(float(r['x_px']),float(r['y_px'])) for r in read(folder/'experts.json')})
        if any(not(6<=x<122 and 6<=y<122) for x,y in xy):raise RuntimeError('Source outside eligible field')
        path=root/'projection_preflight'/f'{case}.png';path.parent.mkdir(exist_ok=True)
        if not path.exists():
            fig,ax=plt.subplots(figsize=(6,5));ax.imshow(np.mean(raw[64:164,49:-49,49:-49],axis=0),cmap='gray',origin='upper')
            if xy:ax.scatter(*zip(*xy),facecolors='none',edgecolors='#46dc7d',s=60)
            ax.set_title(case+'\nSetup mean; geometry only',fontsize=9);fig.tight_layout();fig.savefig(path,dpi=110);plt.close(fig)
        records.append(dict(case_id=case,unique_centers=len(xy),coordinates_within_domain=True,figure=binding(path)))
    write_json(root/'datasets_complete.json',dict(status='PASS',datasets=21,paired_setup_byte_equal=True,setup_pairs=pairs,projection_records=records,dataset_bindings=datasets))
    emit(root,status='DATASETS_PREPARED',datasets=21)


def score_reference(root,case,arm):
    base=prepare_mode(root,case,'level')
    if arm==ANCHOR:return base
    out=root/'stages'/case/arm
    if (out/'complete.json').exists():return read(out/'complete.json')
    out.mkdir(parents=True,exist_ok=True);folder=root/'datasets'/case
    ref=np.load(root/'kernels'/f'{arm}.npy')[None]
    # Only reference outputs are consumed. The target A is copied from the fixed baseline.
    kernels=GammaSTKernels(GammaSTSpec(),ref,ref,{})
    source=np.load(folder/'level_source.npy',mmap_mode='r');inp=np.load(folder/'level.npy',mmap_mode='r')
    arrays={k:mmap(out/f'{k}.npy',inp.shape) for k in ('M','Spread')}
    for start,stop,values in iter_chunks(source,kernels,chunk_frames=8):
        arrays['M'][start:stop]=values['M'][:,49:-49,49:-49]
        arrays['Spread'][start:stop]=np.sqrt(values['variance'][:,49:-49,49:-49])
    for a in arrays.values():a.flush()
    population=np.asarray(arrays['Spread'][64:164,6:-6,6:-6]).ravel();positive=population[population>0]
    floor=max(1e-6,float(np.percentile(positive,10)) if len(positive) else 0.)
    del population,positive,arrays
    a=np.load(base['stages']['A']['path'],mmap_mode='r');m=np.load(out/'M.npy',mmap_mode='r');s=np.load(out/'Spread.npy',mmap_mode='r')
    contrast=mmap(out/'C.npy',inp.shape);z=mmap(out/'Z.npy',inp.shape)
    for start in range(0,len(inp),8):
        stop=min(len(inp),start+8);contrast[start:stop]=a[start:stop]-m[start:stop]
        z[start:stop]=contrast[start:stop]/np.maximum(s[start:stop],floor)
    contrast.flush();z.flush()
    stages={k:binding(out/f'{k}.npy') for k in ('M','Spread','C','Z')}
    stages.update({k:base['stages'][k] for k in ('A','X','Raw')})
    state=dict(status='PASS',stages=stages,scale_floor=floor,target_exactly_reused=True,reference_kernel=binding(root/'kernels'/f'{arm}.npy'))
    write_json(out/'complete.json',state);return state


def run(root):
    guard_open(root);p=load(root);prep=read(root/'datasets_complete.json')
    if prep['datasets']!=21:raise RuntimeError('Incomplete preparation')
    for bs in prep['dataset_bindings'].values():
        for b in bs:verify(b)
    for c in cells(p):
        case=c['case_id'];arm=c['arm_id'];out=root/'cells'/case/arm
        if (out/'sealed.json').exists():verify_seal(read(out/'sealed.json'));continue
        state=score_reference(root,case,arm);records=dict(state['stages']);records['Input']=records['X'];records['Score']=records['Z']
        for b in records.values():verify(b)
        frames,setup,application=frame_sets(read(root/'datasets'/case/'metadata.json'));lookup={f:i for i,f in enumerate(frames)}
        score=np.load(records['Z']['path'],mmap_mode='r');area=(score.shape[1]-12)*(score.shape[2]-12)
        out.mkdir(parents=True,exist_ok=True);cache=root/'prefix_cache'/case/arm;cache.mkdir(parents=True,exist_ok=True)
        if c['reused_audit']:
            previous=BASE/'cells'/case/'level_Z__w3';old=read(previous/'sealed.json');verify_seal(old)
            link(old['prefix']['path'],cache/'application.json');link(old['setup_prefix']['path'],cache/'setup.json')
            plan=read(previous/'threshold_plan.json')
            link(previous/'audit_candidates.json',out/'audit_candidates.json')
        else:
            if not (cache/'complete.json').exists():
                fn=lambda f:extract_candidates(score[lookup[f]],source_frame_ui=f,window=3,cell_id=arm)
                setup_rows=[r for f in setup for r in fn(f)];prefix=[r for f in application for r in fn(f)]
                seal_candidates(prefix,source_frames_ui=application)
                write_json(cache/'setup.json',setup_rows);write_json(cache/'application.json',prefix)
                write_json(cache/'complete.json',dict(status='PASS',bindings=[binding(cache/n) for n in ('setup.json','application.json')]))
            for b in read(cache/'complete.json')['bindings']:verify(b)
            plan=threshold_plan([r['score'] for r in read(cache/'setup.json')],area,len(setup))
        fixed=next(r for r in plan if r['threshold_id']=='q1')
        op=dict(fixed,threshold_z=fixed['threshold'],threshold_frozen_from_calibration_only=True,
                eligible_area_px=area,target_proposals_per_frame=area/REFERENCE_AREA_PX,reference_area_px=REFERENCE_AREA_PX,
                setup_source_frames_ui=setup,application_source_start_ui=application[0],application_source_stop_ui=application[-1],
                application_frame_count=len(application),scale_floor=state['scale_floor'],truth_mode='fully_synthetic',**c)
        if not c['reused_audit']:write_json(out/'audit_candidates.json',select(read(cache/'application.json'),fixed,area/REFERENCE_AREA_PX))
        write_json(out/'threshold_plan.json',plan);write_json(out/'calibration.json',op)
        write_json(out/'sealed.json',dict(status='SEALED_BEFORE_ACTIVITY_TRUTH_JOIN',stages=records,dataset_bindings=prep['dataset_bindings'][case],
                   prefix=binding(cache/'application.json'),setup_prefix=binding(cache/'setup.json'),audit_candidates=binding(out/'audit_candidates.json'),
                   threshold_plan=binding(out/'threshold_plan.json'),calibration=binding(out/'calibration.json')))
        emit(root,status='CELL_SEALED',**c,audit_proposals=len(read(out/'audit_candidates.json')))
    write_json(root/'computation_complete.json',dict(status='PASS',cells=len(cells(p))))


def evaluate(root):
    guard_open(root);p=load(root);allcells=cells(p)
    if not all((root/'cells'/c['case_id']/c['arm_id']/'sealed.json').exists() for c in allcells):raise RuntimeError('Seal all189cells before activity truth join')
    rows=[];replication=[]
    for c in allcells:
        out=root/'cells'/c['case_id']/c['arm_id'];folder=root/'datasets'/c['case_id'];seal=read(out/'sealed.json');verify_seal(seal)
        _,_,application=frame_sets(read(folder/'metadata.json'))
        if not (out/'evaluated.json').exists():
            prefix=read(seal['prefix']['path']);plan=read(out/'threshold_plan.json');op=read(out/'calibration.json')
            experts=read(folder/'experts.json');active=read(folder/'active.json');curves=[];events=[]
            # Global strict thresholds preserve the score-ranked per-frame prefix.
            stream=seal_candidates(prefix,source_frames_ui=application)
            for radius in (2.,6.):
                taus=sorted({float('inf') if s['threshold'] is None else s['threshold'] for s in plan})
                res=evaluate_threshold_sweep(stream,active,experts,thresholds=taus,match_radius_px=radius)
                by_tau={r['threshold_z']:r for r in res['curve_rows']}
                for setting in plan:
                    tau=float('inf') if setting['threshold'] is None else setting['threshold']
                    event_subset=[r for r in res['event_rows'] if r['threshold_z']==setting['threshold']]
                    row=dict(by_tau[setting['threshold']],**setting,**c,eligible_area_px=op['eligible_area_px'])
                    row['deadline_rows']=deadline_rows(event_subset);curves.append(row)
                    events.extend(dict(r,threshold_id=setting['threshold_id'],match_radius_px=radius) for r in event_subset)
                    if setting['threshold_id']=='q1':
                        fixed=evaluate_framewise(stream,active,experts,threshold_z=tau,match_radius_px=radius)
                        write_json(out/f'operating_metrics_r{radius:g}.json',fixed)
            write_json(out/'curves.json',dict(curve_rows=curves,event_rows=events))
            write_json(out/'evaluated.json',dict(status='PASS',seal=binding(out/'sealed.json'),outputs=[binding(f) for f in sorted(out.glob('*metrics*.json'))]+[binding(out/'curves.json')]))
        evaluated=read(out/'evaluated.json');verify(evaluated['seal'])
        for b in evaluated['outputs']:verify(b)
        current=read(out/'curves.json')['curve_rows'];rows.extend(current)
        if c['reused_audit']:
            old=BASE/'cells'/c['case_id']/'level_Z__w3';prior=read(old/'curves.json')['curve_rows']
            fields=('proposal_count','true_positive_count','false_positive_count','recovered_event_count','duplicate_near_active_region_count')
            prior={(r['threshold_id'],r['match_radius_px']):r for r in prior}
            if read(out/'threshold_plan.json')!=read(old/'threshold_plan.json'):raise RuntimeError('Baseline cutoff differs')
            for row in current:
                before=prior[(row['threshold_id'],row['match_radius_px'])]
                if any(row.get(k)!=before.get(k) for k in fields):raise RuntimeError('Baseline metrics differ')
            before=read(old/'sealed.json')
            if any(seal['stages'][k]['sha256']!=before['stages'][k]['sha256'] for k in ('A','M','Spread','C','Z','Raw','Input','Score')):raise RuntimeError('Baseline stage differs')
            if seal['audit_candidates']['sha256']!=before['audit_candidates']['sha256']:raise RuntimeError('Baseline candidates differ')
            replication.append(dict(case_id=c['case_id'],arm_id=c['arm_id'],all_thresholds_equal=True,metrics_equal=True,stages_equal=True,candidates_equal=True))
        emit(root,status='CELL_EVALUATED',**c)
    write_json(root/'all_curves.json',rows);write_tsv(root/'all_curves.tsv',rows)
    write_json(root/'baseline_replication.json',dict(status='PASS',cells=replication))
    write_json(root/'evaluation_complete.json',dict(status='PASS',cells=len(allcells),curve_rows=len(rows)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['preflight','prepare','run','evaluate']);parser.add_argument('--root',type=Path,default=ROOT)
    a=parser.parse_args();globals()[a.command](a.root.resolve())

if __name__=='__main__':main()
