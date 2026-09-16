"""Paired variance/spatial/temporal noise control with frozen reference choices."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import numpy as np

from .necessity_study import read, binding, verify, write_json, write_tsv, sha256, threshold_plan, deadline_rows, emit, link
from .spatiotemporal_study import REPO, frame_sets, REFERENCE_AREA_PX
from .followup_study import verify_seal
from .followup_selection import extract_candidates
from .followup_calibration import select
from .spatiotemporal_metrics import seal_candidates, evaluate_threshold_sweep, evaluate_framewise
from .reference_study import score_reference

BASE = REPO / 'Outputs/GammaLSReference/reference_20260915_r1'
ROOT = REPO / 'Outputs/GammaLSNoise/noise_20260915_r1'
PAPER = REPO.parent / 'Neural_Event_Extraction_Gamma_LS_Clarity_Revision_2026-09-12/editorial'
SEEDS = (20260916, 20260917, 20260918)
ARMS = ('mean2of3_n3', 'mean1_n9', 'mean4of3_n9')
TOTAL_CELLS = 81
DATA_NAMES = ('metadata.json','experts.json','active.json','prepared.json','level_prepared.json',
              'raw_source.npy','Raw.npy','input_source.npy','Input.npy','level_source.npy','level.npy')


def require(value, message):
    if not value: raise RuntimeError(message)


def guard_open(root):
    require(not (Path(root)/'completion_manifest.json').exists(), 'Preserve completed noise study')


def check_seal(seal):
    verify_seal(seal)
    for b in {v['path']:v for v in seal['stages'].values()}.values():verify(b)


def cells(p):
    return [dict(case_id=c['case_id'], arm_id=r['arm_id'], study='noise', input_mode='level',
                 readout='Z', window=3, calibration_method='global', reused_audit=c['reused_dataset'],
                 baseline_case_id=c.get('baseline_case_id'), baseline_arm=r['arm_id'] if c['reused_dataset'] else None,
                 case_kind=c['kind'], seed=c['seed'], V=c['V'], S=c['S'], T=c['T'])
            for c in p['cases'] for r in p['references']]


def case_design():
    cases=[]
    for seed in SEEDS:
        for v in (0,1):
            for s in (0,1):
                for t in (0,1):
                    reused=(v,s,t)==(0,0,0)
                    case=f'reference_null_stationary__seed{seed}' if reused else f'noise_v{v}_s{s}_t{t}__seed{seed}'
                    cases.append(dict(case_id=case,seed=seed,kind='factorial',V=v,S=s,T=t,
                                      reused_dataset=reused,baseline_case_id=case if reused else None))
        case=f'reference_null_variance_correlation__seed{seed}'
        cases.append(dict(case_id=case,seed=seed,kind='legacy',V=None,S=None,T=None,
                          reused_dataset=True,baseline_case_id=case))
    return cases


def preflight(root):
    from .noise_fixtures import theory
    root=Path(root).resolve()
    require(not root.exists(),'New noncolliding root required')
    done=read(BASE/'completion_manifest.json');old=read(BASE/'protocol.json')
    require(done['status']=='PASS' and done['counts']['cells']==189,'Completed reference study required')
    require(sha256(BASE/'completion_manifest.json')=='688fcf5e20956558c036598da1e58f5387ceedf0db83f430926adf35895672e9','Reference completion differs')
    authority={str(Path(b['path']).resolve()):b for b in done['evidence_bindings']}
    refs=[next(r for r in old['references'] if r['arm_id']==arm) for arm in ARMS]
    cases=case_design();baseline_sources=[]
    for c in cases:
        if not c['reused_dataset']:continue
        folder=BASE/'datasets'/c['case_id']
        paths=[folder/n for n in DATA_NAMES]
        paths += [BASE/'cells'/c['case_id']/arm/n for arm in ARMS for n in ('sealed.json','calibration.json','threshold_plan.json','curves.json')]
        for f in paths:
            key=str(f.resolve());known=authority.get(key)
            # Difference inputs are unused by this level study but prepared.json binds them.
            if known is None and f.name in ('input_source.npy','Input.npy'):
                prepared=read(folder/'prepared.json');verify(prepared['input'])
                if f.name=='input_source.npy':known=prepared['input']
                else:continue
            require(known is not None,f'Baseline source outside completion: {f}')
            verify(known);baseline_sources.append(known)
        for arm in ARMS:check_seal(read(BASE/'cells'/c['case_id']/arm/'sealed.json'))
    code=[Path(__file__),Path(__file__).with_name('noise_fixtures.py')]+[Path(b['path']) for b in old['code_bindings']]
    extra=Path(__file__).with_name('noise_theory.py')
    if extra.exists():code.append(extra)
    for b in old['code_bindings']:verify(b)
    free=shutil.disk_usage(REPO).free;require(free>=120*1024**3,'Need120GiB free disk')
    p=dict(schema_version=1,experiment='gamma_noise_variance_spatial_temporal_factorial',
           paper_protocol=binding(PAPER/'NOISE_FACTORIAL_PROTOCOL_2026-09-15.md'),
           baseline_completion=binding(BASE/'completion_manifest.json'),baseline_protocol=binding(BASE/'protocol.json'),
           baseline_sources=list({b['path']:b for b in baseline_sources}.values()),
           code_bindings=[binding(f) for f in dict.fromkeys(code)],cases=cases,references=refs,
           expected_cells=81,expected_datasets=27,new_audits=63,reused_audits=18,expected_curve_rows=1620,
           factorial='Complete2x2x2 V,S,T;3paired seeds;3fixed reference choices. Legacy compound excluded from factorial effects.',
           noise='Common main innovations; raw setup stationary sigma2. SpatialL2-normalized3x3 sum/3; temporalAR.8/.6 with independent stationary initial state. Sigma2 or5 multiplies AR output, not innovations.',
           changes='S/T start at application offset0 (UI165); V sigma2-to5 step at offset100 (UI265). No empirical frame normalization.',
           theory=theory(),conditioning=old['conditioning'],target=old['target'],calibration=old['calibration'],selector=old['selector'],
           primary='q1 null false proposals per10000um2/s;early100frames andlate200frames separately; paired factorial main effects and interactions on absolute rate scale.',
           secondary='All10setup-budget cutoffs,frame counts and trajectories,score/spread/floor diagnostics,raw and conditioned noise variance;legacy bridge as separate comparison.',
           frame_rate_hz=50,pixel_size_um=.5,eligible_area_px=13456,
           epochs=dict(setup=[65,164],early=[165,264],late=[265,464],application=[165,464]),
           scientific_audit=dict(enabled=True,operating_point='q1',policy='Every required model fullfield,ROIvideo and full trace; empty Expert/Comparison explicit.18exactreuses and63newfullaudits.'),
           scope='Synthetic source-free mechanism control. Every proposal is false only under this exhaustive null. Sensitivity undefined; no biological performance, optimal reference, independent validation or controller claim. Three seeds are replicates.',
           resources=dict(cpu='One numerical thread,chunk8,low priority,CPUs6/7excluded',media='At most3workers,onecodec threadeach',free_disk_bytes=free),
           git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
           initial_git_status=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True))
    require(len(cells(p))==81,'Matrix count differs')
    root.mkdir(parents=True);(root/'kernels').mkdir()
    for arm in ARMS:link(BASE/'kernels'/f'{arm}.npy',root/'kernels'/f'{arm}.npy')
    p['kernel_bindings']=[binding(root/'kernels'/f'{arm}.npy') for arm in ARMS]
    write_json(root/'protocol.json',p);write_json(root/'preflight.json',dict(status='PASS',protocol_sha256=sha256(root/'protocol.json')))
    emit(root,status='PREFLIGHT_PASS',cells=81,datasets=27)


def load(root):
    p=read(root/'protocol.json')
    require(sha256(root/'protocol.json')==read(root/'preflight.json')['protocol_sha256'],'Frozen protocol changed')
    for b in p['code_bindings']+p['kernel_bindings']+p['baseline_sources']+[p['baseline_completion'],p['baseline_protocol']]:verify(b)
    if 'paper_protocol' in p:verify(p['paper_protocol'])
    return p


def prepare(root):
    from .noise_fixtures import build_noise_dataset
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    guard_open(root);p=load(root);(root/'datasets').mkdir(exist_ok=True)
    for c in p['cases']:
        if c['reused_dataset']:link(BASE/'datasets'/c['case_id'],root/'datasets'/c['case_id'])
        else:build_noise_dataset(root,c['seed'],c['V'],c['S'],c['T'])
    records=[];datasets={};pairs=[]
    for c in p['cases']:
        case=c['case_id'];folder=root/'datasets'/case
        datasets[case]=[binding(folder/n) for n in DATA_NAMES]
        raw=np.load(folder/'raw_source.npy',mmap_mode='r')
        anchor=np.load(BASE/'datasets'/f"reference_null_stationary__seed{c['seed']}"/'raw_source.npy',mmap_mode='r')
        require(np.array_equal(raw[:164],anchor[:164]),'Paired setup bytes differ')
        require(read(folder/'metadata.json')['event_count']==0,'Non-null dataset')
        pairs.append(dict(case_id=case,setup_byte_equal=True))
        path=root/'projection_preflight'/f'{case}.png';path.parent.mkdir(exist_ok=True)
        if not path.exists():
            fig,ax=plt.subplots(figsize=(5.5,4.5));ax.imshow(np.mean(raw[64:164,49:-49,49:-49],axis=0),cmap='gray',origin='upper')
            ax.set_title(case+'\nSetup mean; declared source-free geometry',fontsize=8);fig.tight_layout();fig.savefig(path,dpi=110);plt.close(fig)
        records.append(dict(case_id=case,unique_centers=0,coordinates_within_domain=True,figure=binding(path)))
    for c in p['cases']:
        if c['kind']!='factorial' or c['V']!=1:continue
        peer=next(x for x in p['cases'] if x['kind']=='factorial' and (x['seed'],x['V'],x['S'],x['T'])==(c['seed'],0,c['S'],c['T']))
        left=np.load(root/'datasets'/c['case_id']/'raw_source.npy',mmap_mode='r')
        right=np.load(root/'datasets'/peer['case_id']/'raw_source.npy',mmap_mode='r')
        require(np.array_equal(left[:264],right[:264]),'V pair differs before its step')
    write_json(root/'datasets_complete.json',dict(status='PASS',datasets=27,paired_setup_byte_equal=True,variance_pairs_early_byte_equal=True,
               setup_pairs=pairs,projection_records=records,dataset_bindings=datasets))
    emit(root,status='DATASETS_PREPARED',datasets=27)


def run(root):
    guard_open(root);p=load(root);prep=read(root/'datasets_complete.json')
    require(prep['datasets']==27 and prep['variance_pairs_early_byte_equal'],'Incomplete preparation')
    require(read(root/'projection_visual_qa.json')['status']=='PASS','Projection visual QA required before scoring')
    for bs in prep['dataset_bindings'].values():
        for b in bs:verify(b)
    for c in cells(p):
        case,arm=c['case_id'],c['arm_id'];out=root/'cells'/case/arm
        if (out/'sealed.json').exists():check_seal(read(out/'sealed.json'));continue
        out.mkdir(parents=True,exist_ok=True);cache=root/'prefix_cache'/case/arm;cache.mkdir(parents=True,exist_ok=True)
        if c['reused_audit']:
            previous=BASE/'cells'/case/arm;old=read(previous/'sealed.json');check_seal(old)
            records=dict(old['stages']);floor=read(previous/'calibration.json')['scale_floor']
            link(old['prefix']['path'],cache/'application.json');link(old['setup_prefix']['path'],cache/'setup.json')
            plan=read(previous/'threshold_plan.json');link(previous/'audit_candidates.json',out/'audit_candidates.json')
        else:
            state=score_reference(root,case,arm);records=dict(state['stages']);records['Input']=records['X'];records['Score']=records['Z'];floor=state['scale_floor']
        for b in {v['path']:v for v in records.values()}.values():verify(b)
        frames,setup,application=frame_sets(read(root/'datasets'/case/'metadata.json'));lookup={f:i for i,f in enumerate(frames)}
        score=np.load(records['Z']['path'],mmap_mode='r');area=(score.shape[1]-12)*(score.shape[2]-12)
        if not c['reused_audit']:
            if not (cache/'complete.json').exists():
                fn=lambda f:extract_candidates(score[lookup[f]],source_frame_ui=f,window=3,cell_id=arm)
                setup_rows=[r for f in setup for r in fn(f)];prefix=[r for f in application for r in fn(f)]
                seal_candidates(prefix,source_frames_ui=application)
                write_json(cache/'setup.json',setup_rows);write_json(cache/'application.json',prefix)
                write_json(cache/'complete.json',dict(status='PASS',bindings=[binding(cache/n) for n in ('setup.json','application.json')]))
            for b in read(cache/'complete.json')['bindings']:verify(b)
            plan=threshold_plan([r['score'] for r in read(cache/'setup.json')],area,len(setup))
        fixed=next(r for r in plan if r['threshold_id']=='q1')
        op=dict(fixed,threshold_z=fixed['threshold'],threshold_frozen_from_calibration_only=True,eligible_area_px=area,
                target_proposals_per_frame=area/REFERENCE_AREA_PX,reference_area_px=REFERENCE_AREA_PX,setup_source_frames_ui=setup,
                application_source_start_ui=application[0],application_source_stop_ui=application[-1],application_frame_count=len(application),
                scale_floor=floor,truth_mode='fully_synthetic',**c)
        if not c['reused_audit']:write_json(out/'audit_candidates.json',select(read(cache/'application.json'),fixed,area/REFERENCE_AREA_PX))
        write_json(out/'threshold_plan.json',plan);write_json(out/'calibration.json',op)
        write_json(out/'sealed.json',dict(status='SEALED_BEFORE_ACTIVITY_TRUTH_JOIN',stages=records,dataset_bindings=prep['dataset_bindings'][case],
                   prefix=binding(cache/'application.json'),setup_prefix=binding(cache/'setup.json'),audit_candidates=binding(out/'audit_candidates.json'),
                   threshold_plan=binding(out/'threshold_plan.json'),calibration=binding(out/'calibration.json')))
        emit(root,status='CELL_SEALED',**c,audit_proposals=len(read(out/'audit_candidates.json')))
    groups=[]
    for seed in SEEDS:
        for arm in ARMS:
            paths=[root/'cells'/c['case_id']/arm for c in p['cases'] if c['seed']==seed]
            plans=[read(x/'threshold_plan.json') for x in paths];floors=[read(x/'calibration.json')['scale_floor'] for x in paths]
            require(all(x==plans[0] for x in plans) and all(x==floors[0] for x in floors),'Paired setup calibration differs')
            groups.append(dict(seed=seed,arm_id=arm,cases=len(paths),thresholds_equal=True,floors_equal=True))
    write_json(root/'paired_setup_calibration_check.json',dict(status='PASS',groups=groups))
    write_json(root/'computation_complete.json',dict(status='PASS',cells=81))


def evaluate(root):
    guard_open(root);p=load(root);allcells=cells(p)
    require(all((root/'cells'/c['case_id']/c['arm_id']/'sealed.json').exists() for c in allcells),'Seal all81cells before truth join')
    rows=[];replication=[]
    for c in allcells:
        out=root/'cells'/c['case_id']/c['arm_id'];folder=root/'datasets'/c['case_id'];seal=read(out/'sealed.json');check_seal(seal)
        _,_,application=frame_sets(read(folder/'metadata.json'))
        if not (out/'evaluated.json').exists():
            prefix=read(seal['prefix']['path']);plan=read(out/'threshold_plan.json');op=read(out/'calibration.json')
            experts=read(folder/'experts.json');active=read(folder/'active.json');require(experts==active==[],'Null truth must be empty')
            curves=[];stream=seal_candidates(prefix,source_frames_ui=application)
            for radius in (2.,6.):
                taus=sorted({float('inf') if s['threshold'] is None else s['threshold'] for s in plan})
                res=evaluate_threshold_sweep(stream,active,experts,thresholds=taus,match_radius_px=radius)
                by_tau={r['threshold_z']:r for r in res['curve_rows']}
                for setting in plan:
                    row=dict(by_tau[setting['threshold']],**setting,**c,eligible_area_px=op['eligible_area_px'],deadline_rows=deadline_rows([]));curves.append(row)
                    if setting['threshold_id']=='q1':
                        tau=float('inf') if setting['threshold'] is None else setting['threshold']
                        write_json(out/f'operating_metrics_r{radius:g}.json',evaluate_framewise(stream,active,experts,threshold_z=tau,match_radius_px=radius))
            write_json(out/'curves.json',dict(curve_rows=curves,event_rows=[]))
            write_json(out/'evaluated.json',dict(status='PASS',seal=binding(out/'sealed.json'),outputs=[binding(f) for f in sorted(out.glob('*metrics*.json'))]+[binding(out/'curves.json')]))
        evaluated=read(out/'evaluated.json');verify(evaluated['seal'])
        for b in evaluated['outputs']:verify(b)
        current=read(out/'curves.json')['curve_rows'];rows.extend(current)
        if c['reused_audit']:
            old=BASE/'cells'/c['case_id']/c['arm_id'];prior={(r['threshold_id'],r['match_radius_px']):r for r in read(old/'curves.json')['curve_rows']}
            require(read(out/'threshold_plan.json')==read(old/'threshold_plan.json'),'Reused cutoff differs')
            fields=('proposal_count','true_positive_count','false_positive_count','recovered_event_count','duplicate_near_active_region_count')
            require(all(all(r.get(k)==prior[(r['threshold_id'],r['match_radius_px'])].get(k) for k in fields) for r in current),'Reused metrics differ')
            before=read(old/'sealed.json');require(all(seal['stages'][k]['sha256']==before['stages'][k]['sha256'] for k in ('A','M','Spread','C','Z','Raw','Input','Score')),'Reused stage differs')
            require(seal['audit_candidates']['sha256']==before['audit_candidates']['sha256'],'Reused candidates differ')
            replication.append(dict(case_id=c['case_id'],arm_id=c['arm_id'],all_thresholds_equal=True,metrics_equal=True,stages_equal=True,candidates_equal=True))
        emit(root,status='CELL_EVALUATED',**c)
    write_json(root/'all_curves.json',rows);write_tsv(root/'all_curves.tsv',rows)
    write_json(root/'baseline_replication.json',dict(status='PASS',cells=replication))
    write_json(root/'evaluation_complete.json',dict(status='PASS',cells=81,curve_rows=1620))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['preflight','prepare','run','evaluate']);parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();globals()[args.command](args.root.resolve())

if __name__=='__main__':main()
