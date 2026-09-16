"""Close the follow-up only after numerical, full media and visual QA gates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

from .followup_study import ROOT, PAPER, load, cells, write_json
from .followup_validate import FileVerifier


def require(value,message):
    if not value:raise RuntimeError(message)


def finalize(root,write=False):
    root=Path(root).resolve()
    require(not (root/'completion_manifest.json').exists(),'Preserve completed study')
    p=load(root);records={};verifier=FileVerifier()
    def bind(path):
        b=verifier.binding(path);records[b['path']]=b;return b
    def check(b,base=None):
        current=verifier.verify(b,root=base);records[current['path']]=current
    def get(path):
        bind(path);return verifier.read_json(path)
    for b in p['code_bindings']:check(b)
    for k in ('baseline_completion','baseline_protocol','baseline_dataset_completion','real_annotation_acceptance'):check(p[k])
    require(not get(Path(p['real_annotation_acceptance']['path']))['accepted'],'Unexpected accepted real annotation version')
    get(root/'protocol.json');get(root/'preflight.json')
    prep=get(root/'datasets_complete.json')
    require(prep['status']=='PASS' and prep['datasets']==29 and prep['paired_setup_byte_equal'],'Dataset preparation incomplete')
    for bs in prep['dataset_bindings'].values():
        for b in bs:check(b)
    for r in prep['projection_records']:check(r['figure'])
    projectionqa=get(root/'projection_visual_qa.json');require(projectionqa['status']=='PASS','Projection review incomplete')
    for b in projectionqa['reviewed']:check(b)
    require(get(root/'computation_complete.json')==dict(status='PASS',cells=242),'Scoring incomplete')
    require(get(root/'evaluation_complete.json')==dict(status='PASS',cells=242,curve_rows=3140),'Evaluation incomplete')
    require(len(get(root/'all_curves.json'))==3140,'Incomplete curve inventory');bind(root/'all_curves.tsv')
    replication=get(root/'baseline_replication.json')
    require(replication['status']=='PASS' and len(replication['cells'])==85,'Global controls did not replicate')
    require(all(r['all_thresholds_equal'] and r['metrics_equal'] for r in replication['cells']),'Replication failure')
    stage_records={}
    for c in cells(p):
        folder=root/'cells'/c['case_id']/c['arm_id'];seal=get(folder/'sealed.json')
        require(seal['status']=='SEALED_BEFORE_ACTIVITY_TRUTH_JOIN','Unsealed cell')
        for key in ('prefix','setup_prefix','audit_candidates','threshold_plan','calibration'):check(seal[key])
        for b in seal['dataset_bindings']:check(b)
        stage_records.update({b['path']:b for b in seal['stages'].values()})
        ev=get(folder/'evaluated.json');check(ev['seal'])
        for b in ev['outputs']:check(b)
    # Fresh score hashes are deduplicated by resolved source path.
    for b in stage_records.values():check(b)
    audit=get(root/'audit_complete.json')
    require(audit['status']=='PASS' and audit['cells']==242 and audit['new_cells']==157 and audit['reused_cells']==85,'Full media validation incomplete')
    check(audit['validator'])
    for row in audit['worker_receipts']:
        check(row['receipt']);check(row['log'])
    keys={(r['case_id'],r['arm_id']) for r in audit['audits']}
    require(len(audit['audits'])==242 and keys=={(c['case_id'],c['arm_id']) for c in cells(p)},'Missing or duplicate audit cells')
    videos=0;traces=0;comparisons=0
    for r in audit['audits']:
        require(r['summary']['scientific_audit_complete'],'Incomplete cell audit')
        for b in r['metadata_bindings'].values():check(b)
        summary=r['summary'];videos+=summary['video_count']
        traces+=summary.get('expert_roi_count',0)+summary.get('model_roi_count',0)
        comparisons+=summary.get('expert_occurrence_count',0)
    forecast=get(root/'audit_forecast.json')
    require(videos==forecast['total_videos'],'Forecast inventory differs')
    get(root/'display_contract.json')
    for worker in range(3):
        receipt=get(root/f'media_worker_{worker}.json');require(receipt['status']=='PASS' and receipt['cells'] in (80,81),'Incomplete media worker')
    report=get(root/'report/manifest.json')
    require(report['scientific_audit_complete'],'Report does not reflect complete media')
    require(Path(report['audit_verification']['source']['path']).resolve()==(root/'audit_complete.json').resolve(),'Report references another aggregate')
    check(report['audit_verification']['source'])
    for b in report['artifacts']:check(b,root/'report')
    for b in report['inputs']:check(b)
    check(report['reporter'])
    qa=get(root/'report_visual_qa.json');require(qa['status']=='PASS','Report visual review incomplete')
    for b in qa['figures']:check(b)
    tests=get(root/'verification.json');require(tests['status']=='PASS' and tests['total_test_count']==44,'Software/contract checks incomplete')
    for b in tests['tests']+[tests[k] for k in ('junit','log','report_test_source','report_junit')]:check(b)
    check(tests['validator_test_receipt'])
    validator_tests=get(Path(tests['validator_test_receipt']['path']))
    require(validator_tests['status']=='PASS' and validator_tests['counts']==dict(tests=13,errors=0,failures=0,skipped=0),'Aggregate validator tests incomplete')
    check(validator_tests['junit'])
    for b in validator_tests['source_bindings']:check(b)
    diag=get(root/'crowding_diagnostics/summary.json')
    require(diag['status']=='PASS' and diag['diagnostic_cells']==72 and diag['weak_active_frame_rows']==17712,'Mechanism diagnostics incomplete')
    for b in get(root/'crowding_diagnostics/artifact_index.json')['artifacts']:check(b)
    for b in get(root/'crowding_diagnostics/source_manifest.json')['sources']:check(b)
    layout=get(root/'crowding_diagnostics_layout_v2/manifest.json')
    require(layout['status']=='RENDER_PASS_PENDING_SEPARATE_VISUAL_QA' and layout['figure_count']==4,'Diagnostic figure layout incomplete')
    for b in layout['figures']+layout['original_artifacts']:check(b)
    for key in ('generator','original_generator','original_artifact_index'):check(layout[key])
    # Preserve the full diagnostic artifact set regardless of its manifest schema.
    for path in sorted((root/'crowding_diagnostics').rglob('*')):
        if path.is_file():bind(path)
    for path in sorted((root/'crowding_diagnostics_layout_v2').rglob('*')):
        if path.is_file():bind(path)
    diagqa=get(root/'diagnostic_visual_qa.json');require(diagqa['status']=='PASS','Diagnostic visual review incomplete')
    check(diagqa['receipt'])
    diagreview=get(Path(diagqa['receipt']['path']));require(diagreview['status']=='PASS','Separate diagnostic layout review incomplete')
    for key in ('manifest','generator','original_source_manifest'):check(diagreview[key])
    for b in diagqa['figures']:check(b)
    companion=PAPER/'REGIONAL_CALIBRATION_AND_CROWDING_FINDINGS_2026-09-14.md'
    bind(companion);bind(PAPER/'REGIONAL_AND_CROWDING_PROTOCOL_2026-09-14.md')
    links=get(root/'document_link_validation.json');require(links['status']=='PASS','Companion links were not checked')
    for b in links['documents']:check(b)
    traceqa=get(root/'representative_trace_visual_qa.json');require(traceqa['status']=='PASS','Representative trace review incomplete')
    for b in traceqa['reviewed']:check(b)
    for folder,summary_name,counts_key in ((root/'real_window_triage','summary.json','counts'),(root/'real_window_triage/final_links','link_status.json','original_counts')):
        triage=get(folder/summary_name);require(triage['status']=='PASS','Real-window review aid incomplete')
        require(triage[counts_key]==dict(eligible_occurrences=76,geometry_exclusions=3,lost=0,newly_reached=8,original_known_occurrences=79,remaining_misses=45,retained=23),'Real-window triage changed')
        for b in get(folder/'artifact_index.json')['artifacts']:check(b)
        for b in get(folder/'source_manifest.json')['sources']:check(b)
    require(triage['audit_links_ready']=={'global':True,'regional':True},'Real-window trace links incomplete')
    require(triage['ready_link_count']==304 and triage['occurrence_count']==76 and triage['numeric_rows_unchanged'],'Real-window link inventory differs')
    for path in sorted((root/'validation').glob('*')):
        if path.is_file():bind(path)
    bind(Path(__file__))
    source_files=list(dict.fromkeys([Path(b['path']) for b in p['code_bindings']]+list(Path(__file__).parent.glob('followup_*.py'))))
    capsule=[]
    for source in source_files:
        destination=root/'source_capsule'/source.relative_to(Path(__file__).resolve().parents[3])
        if write:
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():require(bind(destination)['sha256']==bind(source)['sha256'],'Changed source capsule')
            else:shutil.copyfile(source,destination)
            capsule.append(dict(source=bind(source),copy=bind(destination)))
    if write:write_json(root/'source_capsule_manifest.json',dict(status='PASS',files=capsule));bind(root/'source_capsule_manifest.json')
    verifier.assert_unchanged()
    value=dict(status='PASS',completed_utc=datetime.now(timezone.utc).isoformat(),numerical_complete=True,
        scientific_artifact_audit_complete=True,report_visual_qa_complete=True,
        counts=dict(cells=242,regional_cells=170,crowding_cells=72,curve_rows=3140,new_audits=157,reused_audits=85,videos=videos,roi_traces=traces,occurrence_comparisons=comparisons),
        claim_boundaries=['Development mechanism comparisons','Sparse real known-window coverage and unknown burden only','No accepted exhaustive real precision/sensitivity','No independent recording validation','Simulated source dimensions/kinetics are provisional','No feedback-control validation','Configured-source nuisance investigation remains subsequent work'],
        validation_scope='Fresh code, dataset metadata, score-array, numerical-table and report hashes. Full media decode and marker/source checks inherited from the completed aggregate validator and its bound per-cell records.',
        evidence_bindings=sorted(records.values(),key=lambda b:b['path']))
    if write:write_json(root/'completion_manifest.json',value)
    print(dict(status='PASS' if write else 'READY',counts=value['counts'],evidence_files=len(records)),flush=True)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--write',action='store_true');a=p.parse_args();finalize(a.root,a.write)


if __name__=='__main__':main()
