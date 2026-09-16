"""Close the reference study after numerical, full-audit, and visual gates."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
from .reference_study import ROOT, PAPER, REPO, load, cells, write_json
from .followup_validate import FileVerifier


def require(value, message):
    if not value:raise RuntimeError(message)


def finalize(root, write=False):
    root=Path(root).resolve();require(not (root/'completion_manifest.json').exists(),'Preserve completed study')
    p=load(root);v=FileVerifier();records={}
    def bind(path):
        b=v.binding(path);records[b['path']]=b;return b
    def get(path):
        bind(path);return v.read_json(path)
    def check(b,base=None):
        value=v.verify(b,root=base);records[value['path']]=value
    def walk_bindings(value):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value:check(value)
            else:
                for child in value.values():walk_bindings(child)
        elif isinstance(value,list):
            for child in value:walk_bindings(child)
    for b in p['code_bindings']+p['kernel_bindings']+p['baseline_sources']:check(b)
    for name in ('baseline_completion','baseline_protocol'):check(p[name])
    for name in ('protocol.json','preflight.json','resource_preflight.json'):get(root/name)
    prep=get(root/'datasets_complete.json')
    require(prep['status']=='PASS' and prep['datasets']==21 and prep['paired_setup_byte_equal'],'Dataset gate failed')
    for bs in prep['dataset_bindings'].values():
        for b in bs:check(b)
    for c in p['null_cases']:
        capsule=get(root/'datasets'/c['case_id']/'null_prepared.json')
        walk_bindings(capsule)
    for r in prep['projection_records']:check(r['figure'])
    qa=get(root/'projection_visual_qa.json');require(qa['status']=='PASS','Projection QA failed')
    for b in qa['reviewed']:check(b)
    require(get(root/'computation_complete.json')==dict(status='PASS',cells=189),'Computation incomplete')
    require(get(root/'evaluation_complete.json')==dict(status='PASS',cells=189,curve_rows=3780),'Evaluation incomplete')
    require(len(get(root/'all_curves.json'))==3780,'Curve count differs');bind(root/'all_curves.tsv')
    replication=get(root/'baseline_replication.json')
    require(replication['status']=='PASS' and len(replication['cells'])==12,'Baseline replication incomplete')
    require(all(all(r[k] for k in ('all_thresholds_equal','metrics_equal','stages_equal','candidates_equal')) for r in replication['cells']),'Baseline replication failed')
    for c in cells(p):
        folder=root/'cells'/c['case_id']/c['arm_id'];seal=get(folder/'sealed.json')
        require(seal['status']=='SEALED_BEFORE_ACTIVITY_TRUTH_JOIN','Unsealed cell')
        for k in ('prefix','setup_prefix','audit_candidates','threshold_plan','calibration'):check(seal[k])
        for b in seal['stages'].values():check(b)
        ev=get(folder/'evaluated.json');require(ev['status']=='PASS','Cell evaluation failed');check(ev['seal'])
        for b in ev['outputs']:check(b)
    setup_check=get(root/'paired_setup_calibration_check.json')
    require(setup_check['status']=='PASS' and len(setup_check['groups'])==27,'Paired setup calibration check failed')
    for seed in (20260916,20260917,20260918):
        for reference in p['references']:
            folders=[root/'cells'/c['case_id']/reference['arm_id'] for c in p['cases'] if c['seed']==seed]
            plans=[get(f/'threshold_plan.json') for f in folders]
            floors=[get(f/'calibration.json')['scale_floor'] for f in folders]
            require(len(folders)==7 and all(x==plans[0] for x in plans) and all(x==floors[0] for x in floors),'Paired setup settings differ')
    audit=get(root/'audit_complete.json')
    require(audit['status']=='PASS' and audit['cells']==189 and audit['new_cells']==177 and audit['reused_cells']==12,'Full media gate failed')
    bind(root/'audit_validation.log')
    for b in audit['validation_code']:check(b)
    for row in audit['worker_receipts']:
        check(row['receipt']);check(row['log'])
    require({(c['case_id'],c['arm_id']) for c in cells(p)}=={(r['case_id'],r['arm_id']) for r in audit['audits']},'Audit matrix differs')
    videos=traces=comparisons=0
    for r in audit['audits']:
        summary=r['summary'];require(summary['scientific_audit_complete'],'Cell audit incomplete')
        videos+=summary['video_count'];traces+=summary['expert_roi_count']+summary['model_roi_count'];comparisons+=summary['expert_occurrence_count']
        for b in r['metadata_bindings'].values():check(b)
    forecast=get(root/'audit_forecast.json');require(videos==forecast['total_videos'],'Forecast total differs')
    get(root/'display_contract.json')
    report=get(root/'report/manifest.json')
    require(report['scientific_audit_complete'] and report['visual_qa_complete'],'Report does not reflect completed audit and visual QA')
    for b in report['artifacts']:check(b,root/'report')
    for b in report['inputs']:check(b)
    check(report['reporter'])
    for name,key in (('report_visual_qa.json','figures'),('representative_trace_visual_qa.json','reviewed')):
        qa=get(root/name);require(qa['status']=='PASS',f'Visual QA incomplete: {name}')
        require(bool(qa[key]),f'Visual QA has no reviewed artifacts: {name}')
        walk_bindings(qa)
    shape=get(root/'geometry_limits/manifest.json')
    require(shape['status']=='PASS','Analytic shape-limit capsule incomplete')
    walk_bindings(shape)
    repair=get(root/'report_layout_repair.json');walk_bindings(repair)
    for path in sorted((root/'validation').glob('*.json')):walk_bindings(get(path))
    bind(root/'independent_numeric_summary.json')
    verification=get(root/'verification.json')
    require(verification['status']=='PASS' and verification['test_count']>=75,'Focused tests incomplete')
    for b in verification['bindings']:check(b)
    links=get(root/'document_link_validation.json');require(links['status']=='PASS','Document links failed')
    for b in links['documents']:check(b)
    geometry=get(root/'geometry_design_receipt.json')
    require(geometry['status']=='PASS','Geometry design receipt missing')
    for key in ('protocol','table','paper_protocol'):check(geometry[key])
    bind(PAPER/'REFERENCE_RADIUS_AND_ORDER_FINDINGS_2026-09-15.md')
    bind(REPO/'docs/workflows/gamma_ls_reference.md')
    source_files=list(dict.fromkeys([Path(b['path']) for b in p['code_bindings']]+list(Path(__file__).parent.glob('reference_*.py'))+list((REPO/'tests').glob('test_reference_*.py'))))
    capsule=[]
    for source in source_files:
        original=bind(source);destination=root/'source_capsule'/source.relative_to(REPO)
        if write:
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():require(bind(destination)['sha256']==original['sha256'],'Source capsule changed')
            else:shutil.copyfile(source,destination)
            capsule.append(dict(source=original,copy=bind(destination)))
    if write:write_json(root/'source_capsule_manifest.json',dict(status='PASS',files=capsule));bind(root/'source_capsule_manifest.json')
    v.assert_unchanged()
    result=dict(status='PASS',completed_utc=datetime.now(timezone.utc).isoformat(),numerical_complete=True,
                scientific_artifact_audit_complete=True,report_visual_qa_complete=True,
                counts=dict(cells=189,datasets=21,kernels=9,curve_rows=3780,new_audits=177,reused_audits=12,videos=videos,roi_traces=traces,occurrence_comparisons=comparisons),
                claim_boundaries=p['scope'],validation_scope='Fresh code,data,numerical,report and audit-metadata hashes; complete per-video source/decode/marker evidence inherited by bound full aggregate validation.',
                evidence_bindings=sorted(records.values(),key=lambda b:b['path']))
    if write:write_json(root/'completion_manifest.json',result)
    print(dict(status='PASS' if write else 'READY',counts=result['counts'],evidence_files=len(records)),flush=True)
    return result


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,default=ROOT);a.add_argument('--write',action='store_true');args=a.parse_args();finalize(args.root,args.write)

if __name__=='__main__':main()
