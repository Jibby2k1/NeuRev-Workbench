"""Background and stationary-conditioned-variance controls with frozen CFAR settings."""
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

BASE = REPO / 'Outputs/GammaLSNoise/noise_20260915_r1'
ROOT = REPO / 'Outputs/GammaLSBackground/background_20260915_r1'
PAPER = REPO.parent / 'Neural_Event_Extraction_Gamma_LS_Clarity_Revision_2026-09-12/editorial'
SEEDS = (20260916, 20260917, 20260918)
ARMS = ('mean2of3_n3', 'mean1_n9', 'mean4of3_n9')
TOTAL_CELLS = 252
DATA_NAMES = ('metadata.json','experts.json','active.json','prepared.json','level_prepared.json',
              'raw_source.npy','Raw.npy','input_source.npy','Input.npy','level_source.npy','level.npy')


def require(value, message):
    if not value: raise RuntimeError(message)


def guard_open(root):
    require(not (Path(root)/'completion_manifest.json').exists(), 'Preserve completed background study')


def check_seal(seal):
    verify_seal(seal)
    for b in {v['path']:v for v in seal['stages'].values()}.values():verify(b)


def cells(p):
    return [dict(case_id=c['case_id'], arm_id=r['arm_id'], study='background', input_mode='level',
                 readout='Z', window=3, calibration_method='global', reused_audit=c['reused_dataset'],
                 baseline_case_id=c.get('baseline_case_id'), baseline_arm=r['arm_id'] if c['reused_dataset'] else None,
                 case_kind=c['kind'], seed=c['seed'], V=c['V'], S=c['S'], T=c['T'], background=c['background'], normalization=c['normalization'])
            for c in p['cases'] for r in p['references']]


def logical_design():
    from .background_fixtures import case_id
    from .noise_fixtures import case_id as old_id
    rows=[]
    for seed in SEEDS:
        for background in ('sloped','flat'):
            for normalization in ('raw','conditioned'):
                for v in (0,1):
                    for s in (0,1):
                        for t in (0,1):
                            canonical_norm='raw' if not (s or t) else normalization
                            reused=background=='sloped' and canonical_norm=='raw'
                            case=old_id(seed,v,s,t) if reused else case_id(seed,v,s,t,background,canonical_norm)
                            rows.append(dict(canonical_case_id=case,seed=seed,background=background,normalization=normalization,
                                V=v,S=s,T=t,alias=(normalization!=canonical_norm),reused_dataset=reused))
    return rows


def case_design():
    cases={}
    for r in logical_design():
        if r['alias']:continue
        c=dict(r);case=c.pop('canonical_case_id');c.pop('alias')
        cases[case]=dict(c,case_id=case,kind='factorial',baseline_case_id=case if c['reused_dataset'] else None)
    return list(cases.values())


def preflight(root):
    from .background_fixtures import matching_parameters
    from .followup_validate import FileVerifier
    root=Path(root).resolve();require(not root.exists(),'New noncolliding root required')
    done=read(BASE/'completion_manifest.json')
    require(done['status']=='PASS' and done['counts']['cells']==81,'Completed noise study required')
    require(sha256(BASE/'completion_manifest.json')=='b9f696a9e1ae2ddc146107e2f07f010f3b1f0c3e2f060263dd1bc4d3c7d7d706','Noise completion differs')
    verifier=FileVerifier();authority={str(Path(b['path']).resolve()):b for b in done['evidence_bindings']}
    protocol_bound=authority.get(str((BASE/'protocol.json').resolve()))
    require(protocol_bound is not None,'Completed baseline protocol is unbound')
    verifier.verify(protocol_bound);old=read(BASE/'protocol.json')
    baseline_sources=[protocol_bound]
    for b in old['kernel_bindings']:
        known=authority.get(str(Path(b['path']).resolve()))
        require(known==b,'Completed baseline kernel binding differs')
        verifier.verify(known);baseline_sources.append(known)
    refs=old['references'];cases=case_design()
    for c in cases:
        if not c['reused_dataset']:continue
        paths=[BASE/'datasets'/c['case_id']/n for n in DATA_NAMES]
        paths += [BASE/'cells'/c['case_id']/arm/n for arm in ARMS for n in
            ('sealed.json','calibration.json','threshold_plan.json','curves.json','evaluated.json')]
        for f in paths:
            known=authority.get(str(f.resolve()))
            if known is None:
                # Unused signed-difference data remain bound through preparation.
                prep=read(BASE/'datasets'/c['case_id']/'prepared.json')
                require(f.name in ('input_source.npy','Input.npy'),'Baseline source outside completion: '+str(f))
                verifier.verify(prep['input'])
                if f.name=='Input.npy':
                    full=np.load(prep['input']['path'],mmap_mode='r');cropped=np.load(f,mmap_mode='r')
                    require(np.array_equal(full[:,49:-49,49:-49],cropped),'Unused signed-difference crop differs from bound full source')
                    del full,cropped
                known=verifier.binding(f)
            verifier.verify(known);baseline_sources.append(known)
        for arm in ARMS:
            seal=read(BASE/'cells'/c['case_id']/arm/'sealed.json')
            for b in list(seal['stages'].values())+[seal[k] for k in ('prefix','setup_prefix','audit_candidates','threshold_plan','calibration')]:
                verifier.verify(b);baseline_sources.append(b)
    code=[Path(__file__),Path(__file__).with_name('background_fixtures.py'),REPO/'tests/test_background_study.py',REPO/'tests/test_background_fixtures.py']+[Path(b['path']) for b in old['code_bindings']]
    for b in old['code_bindings']:verifier.verify(b)
    free=shutil.disk_usage(REPO).free;require(free>=240*1024**3,'Need240GiB free disk')
    p=dict(schema_version=1,experiment='gamma_background_conditioned_variance_factorial',
        paper_protocol=binding(PAPER/'BACKGROUND_VARIANCE_PROTOCOL_2026-09-15.md'),
        baseline_completion=binding(BASE/'completion_manifest.json'),baseline_protocol=binding(BASE/'protocol.json'),
        baseline_sources=list({b['path']:b for b in baseline_sources}.values()),
        code_bindings=[binding(f) for f in dict.fromkeys(code)],cases=cases,logical_cases=logical_design(),references=refs,
        expected_cells=252,expected_datasets=84,new_audits=180,reused_audits=72,expected_curve_rows=5040,
        logical_cells=288,logical_datasets=96,internal_alias_cells=36,
        factorial='B flat/sloped x N raw/conditioned-matched x V/S/T x3pairedseeds x3fixedreferences; ST00 has identity normalization and aliases raw.',
        matching_gains=[dict(S=s,T=t,**matching_parameters(s,t,'conditioned')) for s in (0,1) for t in (0,1)],
        noise='Exact old innovations and AR initializer; application noise only scaled by reciprocal square root of stationary conditioning gain in matched mode. V sigma applied after AR. No empirical normalization.',
        changes='S/T and normalization begin UI165; sigma2-to5 V step UI265. Flat or sloped background throughout warmup/setup/application.',
        conditioning=old['conditioning'],target=old['target'],calibration=old['calibration'],selector=old['selector'],
        primary='q1 late UI265-464 false proposals per10000um2/s; paired B/N absolute-rate contrasts for each VST/reference, seed as replicate.',
        secondary='Original early and full exposure, settled early UI215-264 and late UI315-464, fixed10frame bins after each switch, all10setupcutoffs, raw/input residual moments, C variance, spread/floor/Z tails.',
        frame_rate_hz=50,pixel_size_um=.5,eligible_area_px=13456,
        epochs=dict(setup=[65,164],early=[165,264],late=[265,464],application=[165,464]),
        settled_epochs=dict(early=[215,264],late=[315,464]),transition_bin_frames=10,
        scientific_audit=dict(enabled=True,operating_point='q1',policy='All252unique states audited;72exact prior reuses,180new complete null model audits.36logical identity aliases are not new evidence.'),
        scope='Synthetic source-free mechanism control. All proposals false under generator only; sensitivity undefined. Same three seeds reused, not independent biological or control validation. Stationary matching is a diagnostic generator control, not an estimated online normalization or exact transient match.',
        resources=dict(cpu='At most3 disjoint case workers, one numeric thread each, chunk8,lowpriority,CPUs6/7excluded',media='At most3workers,onecodec thread each',free_disk_bytes=free),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        initial_git_status=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True))
    require(len(cases)==84 and len(cells(p))==252 and len(p['logical_cases'])==96,'Matrix differs')
    verifier.assert_unchanged();root.mkdir(parents=True);(root/'kernels').mkdir()
    for arm in ARMS:link(BASE/'kernels'/f'{arm}.npy',root/'kernels'/f'{arm}.npy')
    p['kernel_bindings']=[binding(root/'kernels'/f'{arm}.npy') for arm in ARMS]
    write_json(root/'protocol.json',p);write_json(root/'preflight.json',dict(status='PASS',protocol_sha256=sha256(root/'protocol.json')))
    emit(root,status='PREFLIGHT_PASS',cells=252,datasets=84,logical_cells=288)


def load(root):
    p=read(root/'protocol.json')
    require(sha256(root/'protocol.json')==read(root/'preflight.json')['protocol_sha256'],'Frozen protocol changed')
    from .followup_validate import FileVerifier
    verifier=FileVerifier()
    for b in p['code_bindings']+p['kernel_bindings']+p['baseline_sources']+[p['baseline_completion'],p['baseline_protocol']]:verifier.verify(b)
    verifier.assert_unchanged()
    if 'paper_protocol' in p:verify(p['paper_protocol'])
    return p


def prepare(root):
    from .background_fixtures import build_background_dataset
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    guard_open(root);p=load(root);(root/'datasets').mkdir(exist_ok=True)
    for c in p['cases']:
        if c['reused_dataset']:link(BASE/'datasets'/c['case_id'],root/'datasets'/c['case_id'])
        else:build_background_dataset(root,c['seed'],c['V'],c['S'],c['T'],c['background'],c['normalization'])
        emit(root,status='DATASET_PREPARED',case_id=c['case_id'])
    records=[];datasets={};pairs=[]
    for c in p['cases']:
        case=c['case_id'];folder=root/'datasets'/case
        datasets[case]=[binding(folder/n) for n in DATA_NAMES]
        raw=np.load(folder/'raw_source.npy',mmap_mode='r')
        anchor_case=next(x['case_id'] for x in p['cases'] if x['seed']==c['seed'] and x['background']==c['background'])
        anchor=np.load(root/'datasets'/anchor_case/'raw_source.npy',mmap_mode='r')
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
        peer=next(x for x in p['cases'] if x['kind']=='factorial' and (x['seed'],x['background'],x['normalization'],x['V'],x['S'],x['T'])==(c['seed'],c['background'],c['normalization'],0,c['S'],c['T']))
        left=np.load(root/'datasets'/c['case_id']/'raw_source.npy',mmap_mode='r')
        right=np.load(root/'datasets'/peer['case_id']/'raw_source.npy',mmap_mode='r')
        require(np.array_equal(left[:264],right[:264]),'V pair differs before its step')
    write_json(root/'datasets_complete.json',dict(status='PASS',datasets=84,paired_setup_byte_equal=True,variance_pairs_early_byte_equal=True,
               setup_pairs=pairs,projection_records=records,dataset_bindings=datasets))
    emit(root,status='DATASETS_PREPARED',datasets=84)


def run(root, worker=0, workers=1):
    guard_open(root);p=load(root);prep=read(root/'datasets_complete.json')
    require(prep['datasets']==84 and prep['variance_pairs_early_byte_equal'],'Incomplete preparation')
    require(read(root/'projection_visual_qa.json')['status']=='PASS','Projection visual QA required before scoring')
    require(0<=worker<workers<=3,'Invalid numerical partition')
    owned={c['case_id'] for i,c in enumerate(p['cases']) if i%workers==worker}
    for case,bs in prep['dataset_bindings'].items():
        if case in owned:
            for b in bs:verify(b)
    for c in cells(p):
        if c['case_id'] not in owned:continue
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
    write_json(root/f'numeric_worker_{worker}.json',dict(status='PASS',worker=worker,workers=workers,cells=len(owned)*3))
    if workers==1:seal_all(root)


def validate_all_seals(root,p):
    records=[]
    for c in cells(p):
        path=root/'cells'/c['case_id']/c['arm_id']/'sealed.json'
        seal=read(path)
        require(seal.get('status')=='SEALED_BEFORE_ACTIVITY_TRUTH_JOIN','Invalid scoring seal')
        check_seal(seal)
        records.append(binding(path))
    return records


def seal_all(root):
    guard_open(root);p=load(root)
    require(all((root/'cells'/c['case_id']/c['arm_id']/'sealed.json').is_file() for c in cells(p)),'Missing physical cell seals')
    sealed_bindings=validate_all_seals(root,p)
    groups=[]
    for seed in SEEDS:
        for background in ('flat','sloped'):
            for arm in ARMS:
                paths=[root/'cells'/c['case_id']/arm for c in p['cases'] if c['seed']==seed and c['background']==background]
                plans=[read(x/'threshold_plan.json') for x in paths];floors=[read(x/'calibration.json')['scale_floor'] for x in paths]
                require(len(paths)==14 and all(x==plans[0] for x in plans) and all(x==floors[0] for x in floors),'Paired setup calibration differs')
                groups.append(dict(seed=seed,background=background,arm_id=arm,cases=len(paths),thresholds_equal=True,floors_equal=True))
    write_json(root/'paired_setup_calibration_check.json',dict(status='PASS',groups=groups))
    write_json(root/'all_scoring_seals.json',dict(status='PASS',cells=252,protocol=binding(root/'protocol.json'),seals=sealed_bindings))
    write_json(root/'computation_complete.json',dict(status='PASS',cells=252))


def evaluate(root):
    guard_open(root);p=load(root);allcells=cells(p)
    require(all((root/'cells'/c['case_id']/c['arm_id']/'sealed.json').exists() for c in allcells),'Seal all252cells before truth join')
    require(read(root/'computation_complete.json')==dict(status='PASS',cells=252),'Complete scoring first')
    inventory=read(root/'all_scoring_seals.json')
    require(inventory['status']=='PASS' and inventory['cells']==252,'Missing all-seal inventory')
    verify(inventory['protocol'])
    require(inventory['seals']==validate_all_seals(root,p),'All-seal inventory differs before outcome join')
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
    write_json(root/'evaluation_complete.json',dict(status='PASS',cells=252,curve_rows=5040))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['preflight','prepare','run','seal_all','evaluate']);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--worker',type=int,default=0);parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    if args.command=='run':run(args.root.resolve(),args.worker,args.workers)
    else:globals()[args.command](args.root.resolve())

if __name__=='__main__':main()
