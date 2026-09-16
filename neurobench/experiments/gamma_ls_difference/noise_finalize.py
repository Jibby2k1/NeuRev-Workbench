"""Seal the noise control only after numerical, media and visual gates pass."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from .noise_study import ROOT,PAPER,REPO,load,cells,write_json,require
from .followup_validate import FileVerifier


def finalize(root,write=False):
    root=Path(root).resolve();require(not(root/'completion_manifest.json').exists(),'Preserve completed noise study')
    p=load(root);v=FileVerifier();records={}
    def bind(path):
        b=v.binding(path);records[b['path']]=b;return b
    def get(path):
        bind(path);return v.read_json(path)
    def check(b,base=None):
        x=v.verify(b,root=base);records[x['path']]=x
    def walk(x):
        if isinstance(x,dict):
            if 'path' in x and 'sha256' in x:check(x)
            else:
                for y in x.values():walk(y)
        elif isinstance(x,list):
            for y in x:walk(y)
    for b in p['code_bindings']+p['kernel_bindings']+p['baseline_sources']:check(b)
    for name in ('baseline_completion','baseline_protocol','paper_protocol'):check(p[name])
    for name in ('protocol.json','preflight.json','resource_preflight.json','validation/runtime_environment.json'):get(root/name)
    prep=get(root/'datasets_complete.json')
    require(prep['status']=='PASS' and prep['datasets']==27 and prep['paired_setup_byte_equal'] and prep['variance_pairs_early_byte_equal'],'Dataset gate failed')
    for bs in prep['dataset_bindings'].values():walk(bs)
    for c in p['cases']:
        folder=root/'datasets'/c['case_id']
        require(get(folder/'experts.json')==get(folder/'active.json')==[],'Non-null truth')
        if not c['reused_dataset']:
            capsule=get(folder/'noise_prepared.json');require(capsule['status']=='PASS','Fixture capsule incomplete');walk(capsule)
    qa=get(root/'projection_visual_qa.json');require(qa['status']=='PASS' and len(qa['reviewed'])==27,'Projection QA failed');walk(qa)
    require(get(root/'computation_complete.json')==dict(status='PASS',cells=81),'Scoring incomplete')
    require(get(root/'evaluation_complete.json')==dict(status='PASS',cells=81,curve_rows=1620),'Evaluation incomplete')
    rows=get(root/'all_curves.json');require(len(rows)==1620,'Curve count differs');bind(root/'all_curves.tsv')
    require(all(r['true_positive_count']==0 and r['false_positive_count']==r['proposal_count'] and r['framewise_sensitivity'] is None for r in rows),'Null metric semantics differ')
    reuse=get(root/'baseline_replication.json');require(reuse['status']=='PASS' and len(reuse['cells'])==18,'Replication count differs')
    require(all(all(r[k] for k in ('all_thresholds_equal','metrics_equal','stages_equal','candidates_equal')) for r in reuse['cells']),'Replication failed')
    for c in cells(p):
        folder=root/'cells'/c['case_id']/c['arm_id'];seal=get(folder/'sealed.json')
        require(seal['status']=='SEALED_BEFORE_ACTIVITY_TRUTH_JOIN','Unsealed cell');walk(seal)
        ev=get(folder/'evaluated.json');require(ev['status']=='PASS','Unevaluated cell');walk(ev)
    paired=get(root/'paired_setup_calibration_check.json')
    require(paired['status']=='PASS' and len(paired['groups'])==9 and all(r['cases']==9 for r in paired['groups']),'Paired setup gate failed')
    integrity=get(root/'paired_variance_prefix_check.json');require(integrity['status']=='PASS','V-prefix integrity failed');walk(integrity)
    require([len(integrity[k]) for k in ('fixed_target_checks','calibration_groups','raw_pairs','stage_pairs')]==[27,9,12,36]
            and integrity['source_interval_ui']==[1,264] and integrity['numpy_interval']==[0,264]
            and integrity['compared_stage_names']==['Input','A','M','Spread','C','Z']
            and integrity['activity_truth_or_metrics_parsed'] is False,'Incomplete V-prefix integrity inventory')
    independent=get(root/'validation/independent_numeric_check.json');walk(independent)
    require(independent['status']=='PASS' and independent['curve_rows_recounted']==1620
            and independent['epoch_counts_checked']==162 and independent['paired_rate_effects_checked']==189,'Independent numeric recount failed')
    mechanism=get(root/'validation/noise_covariance_mechanism/manifest.json')
    require(mechanism['status']=='PASS' and mechanism['scope']['mathematical_diagnostic_only'] is True,'Mechanism diagnostic failed')
    walk(mechanism)
    for name in ('noise_integrity_run.json','reuse_metadata_precheck_durable.json','media_launch_resources.json'):
        walk(get(root/'validation'/name))
    for name in ('independent_numeric_check.log','independent_numeric_check.attempt1.log','independent_numeric_check.attempt2.log'):
        bind(root/'validation'/name)
    audit=get(root/'audit_complete.json');require(audit['status']=='PASS' and audit['cells']==81 and audit['new_cells']==63 and audit['reused_cells']==18,'Full audit incomplete')
    bind(root/'audit_validation.log')
    for b in audit['validation_code']:check(b)
    for row in audit['worker_receipts']:check(row['receipt']);check(row['log'])
    require({(c['case_id'],c['arm_id']) for c in cells(p)}=={(c['case_id'],c['arm_id']) for c in audit['audits']},'Audit matrix differs')
    videos=traces=0
    for row in audit['audits']:
        s=row['summary'];require(s['scientific_audit_complete'] and s['expert_roi_count']==s['expert_occurrence_count']==0,'Null audit semantics differ')
        videos+=s['video_count'];traces+=s['model_roi_count'];walk(row['metadata_bindings'])
    forecast=get(root/'audit_forecast.json');require(forecast['status']=='PASS' and videos==forecast['total_videos'],'Forecast differs');get(root/'display_contract.json')
    report=get(root/'report/manifest.json')
    require(report['numerical_complete'] and report['scientific_audit_complete'] and report['visual_qa_complete'],'Report not complete')
    for b in report['artifacts']:check(b,root/'report')
    for b in report['inputs']+report['reporting_code']:check(b)
    check(report['reporter'])
    check(report['visual_qa_binding']);report_qa=get(root/'report/visual_qa.json')
    require(report['visual_qa_binding']['sha256']==bind(root/'report/visual_qa.json')['sha256'] and report_qa['status']=='PASS'
            and {Path(b['path']).resolve() for b in report_qa['figures']}=={(root/'report'/f['png']).resolve() for f in report['figures']},'Report QA authority differs')
    walk(report_qa)
    for name,key in (('report_visual_qa.json','figures'),('representative_trace_visual_qa.json','reviewed')):
        qa=get(root/name);require(qa['status']=='PASS' and len(qa[key])==5,f'Visual QA failed: {name}');walk(qa)
    pdf=get(root/'validation/pdf_visual_qa.json');require(pdf['status']=='PASS','PDF QA failed');walk(pdf)
    supplement=get(root/'validation/threshold_sensitivity/manifest.json');walk(supplement)
    require(supplement['status']=='PASS' and supplement['visual_qa_complete'] is True
            and supplement['counts']['factorial_cutoff_rows']==720 and supplement['counts']['pooled_cutoff_rows']==240,'Supplemental threshold readout failed')
    supplement_qa=get(root/'validation/threshold_sensitivity/visual_qa.json');require(supplement_qa['status']=='PASS','Supplemental threshold QA failed');walk(supplement_qa)
    tests=get(root/'verification.json');require(tests['status']=='PASS','Tests incomplete');walk(tests)
    require({s['name'] for s in tests['suites']}=={'fixtures','study','media_validation','integrity','report'},'Test suites incomplete')
    count=0
    for suite in tests['suites']:
        x=ET.parse(suite['xml']['path']);n=len(x.findall('.//testcase'))
        require(n==suite['test_count'] and not x.findall('.//failure') and not x.findall('.//error') and not x.findall('.//skipped'),'Test XML failed')
        count+=n
    require(count==tests['test_count'] and count>=58,'Test count differs')
    links=get(root/'document_link_validation.json');require(links['status']=='PASS','Document links failed');walk(links)
    for path in (PAPER/'NOISE_FACTORIAL_FINDINGS_2026-09-15.md',REPO/'docs/workflows/gamma_ls_noise.md'):bind(path)
    for name in ('prepare.log','run.log','evaluate.log','report_update.log'):bind(root/name)
    originals=list(dict.fromkeys([Path(b['path']) for b in p['code_bindings']]+list(Path(__file__).parent.glob('noise_*.py'))+list((REPO/'tests').glob('test_noise_*.py'))))
    copies=[]
    for source in originals:
        original=bind(source);dest=root/'source_capsule'/source.relative_to(REPO)
        if write:
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():require(bind(dest)['sha256']==original['sha256'],'Changed source capsule')
            else:shutil.copyfile(source,dest)
            copies.append(dict(source=original,copy=bind(dest)))
    if write:write_json(root/'source_capsule_manifest.json',dict(status='PASS',files=copies));bind(root/'source_capsule_manifest.json')
    v.assert_unchanged()
    result=dict(status='PASS',completed_utc=datetime.now(timezone.utc).isoformat(),numerical_complete=True,scientific_artifact_audit_complete=True,report_visual_qa_complete=True,
                counts=dict(cells=81,datasets=27,references=3,factorial_cells=72,legacy_bridge_cells=9,curve_rows=1620,new_audits=63,reused_audits=18,
                            videos=videos,new_videos=forecast['new_videos'],reused_videos=videos-forecast['new_videos'],model_roi_traces=traces,
                            expert_occurrences=0,focused_tests=count,report_figure_pairs=5,supplemental_figure_pairs=1,manual_trace_reviews=5),
                claim_boundaries=p['scope'],validation_scope='Fresh source/numeric/report/audit-metadata hashes and complete hash-bound aggregate source/media validation. No biological validation.',
                evidence_bindings=sorted(records.values(),key=lambda b:b['path']))
    if write:write_json(root/'completion_manifest.json',result)
    print(dict(status='PASS' if write else 'READY',counts=result['counts'],evidence_files=len(records)),flush=True)
    return result


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,default=ROOT);a.add_argument('--write',action='store_true');args=a.parse_args();finalize(args.root,args.write)

if __name__=='__main__':main()
