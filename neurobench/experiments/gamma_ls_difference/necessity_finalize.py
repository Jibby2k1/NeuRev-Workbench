"""Fail-closed metadata closure after the full necessity media validator."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .necessity_study import DEFAULT_ROOT, arms, read, binding, load_protocol


def require(value,message):
    if not value:raise RuntimeError(message)


def finalize(root,write=False):
    root=Path(root).resolve();destination=root/'completion_manifest.json'
    require(not destination.exists(),'Completed roots cannot be overwritten')
    p=load_protocol(root);records={}
    def bind(path):
        path=Path(path).resolve()
        if str(path) not in records:records[str(path)]=binding(path)
        return records[str(path)]
    def verify(record,base=None):
        path=Path(record['path'])
        if not path.is_absolute():path=base/path
        current=bind(path)
        require(all(current[k]==record[k] for k in ('sha256','size_bytes')),f'Changed evidence: {path}')
    def load(path):
        bind(path);return read(path)
    load(root/'preflight.json');load(root/'protocol.json')
    for record in p['code_bindings']:verify(record)
    verify(p['prior_completion']);verify(p['prior_protocol']);verify(p['real_annotation_acceptance'])
    require(load(root/'datasets_complete.json')==dict(status='PASS',datasets=17),'Dataset preparation incomplete')
    require(load(root/'computation_complete.json')==dict(status='PASS',cells=136),'Scoring incomplete')
    require(load(root/'evaluation_complete.json')==dict(status='PASS',cells=136,curve_rows=1360),'Evaluation incomplete')
    baseline=load(root/'baseline_replication.json')
    require(baseline.get('status')=='PASS' and len(baseline['cases'])==17 and all(r['q1_candidates_equal'] and r['threshold_equal'] for r in baseline['cases']),'Difference-Z replication failed')
    expected={(case,a['arm_id']) for case in p['cases'] for a in arms()}
    aggregate=load(root/'audit_complete.json')
    require(aggregate.get('status')=='PASS' and aggregate.get('cells')==136,'Full scientific media audit incomplete')
    keys=[(r['case_id'],r['arm_id']) for r in aggregate['audits']]
    require(len(keys)==len(set(keys))==136 and set(keys)==expected,'Missing or duplicate audits')
    count_videos=count_traces=count_comparisons=0
    for row in aggregate['audits']:
        case,arm=row['case_id'],row['arm_id'];audit=root/'audits'/case/arm;cell=root/'cells'/case/arm
        require(set(row['metadata_bindings'])=={'summary.json','status.json','validation.json','inventory.json','artifact_index.json','source_manifest.json','run_contract.json','llm_context.json'},'Need eight audit metadata bindings')
        for name,record in row['metadata_bindings'].items():
            require(Path(record['path']).resolve()==audit/name,'Incorrect metadata location');verify(record)
        summary=load(audit/'summary.json');validation=load(audit/'validation.json')
        require(summary==row['summary'] and summary.get('scientific_audit_complete') is True,'Summary mismatch')
        require(validation.get('status')=='passed' and validation.get('failures')==[],'Media validation failed')
        for field in ('all_frames_rgb_byte_exact','all_videos_full_decode','all_video_geometry_timing_checked','candidate_scores_equal_saved_score_array','encoded_marker_separation_pass','inventory_complete','scientific_audit_complete','source_hashes_computed'):
            require(validation.get(field) is True,f'Missing validation gate: {field}')
        require(load(audit/'inventory.json').get('complete') is True,'Incomplete inventory')
        seal=load(cell/'sealed.json');evaluated=load(cell/'evaluated.json')
        require(seal.get('status')=='SEALED_BEFORE_ACTIVITY_TRUTH_JOIN' and evaluated.get('status')=='PASS','Invalid scoring/evaluation status')
        verify(evaluated['seal'])
        for record in evaluated['outputs']:verify(record)
        for key in ('prefix','audit_candidates','calibration','threshold_plan','configured_calibration'):verify(seal[key])
        contract=load(audit/'run_contract.json')
        require(contract['source_binding']['candidate_seal_sha256']==bind(cell/'sealed.json')['sha256'],'Audit is not bound to scored cell')
        require(contract['renderer_sha256']==bind(Path(__file__).with_name('spatiotemporal_audit.py'))['sha256'],'Wrong renderer')
        require(contract['operating_point']==load(cell/'calibration.json'),'Operating point differs from calibration')
        count_videos+=summary['video_count'];count_traces+=summary['expert_roi_count']+summary['model_roi_count'];count_comparisons+=summary['comparison_trace_count']
    curves=load(root/'all_curves.json');ops=load(root/'operating_points.json');monitoring=load(root/'monitoring.json')
    require(len(curves)==1360 and len(ops)==len(monitoring)==136,'Aggregate metric count mismatch')
    require(all(r.get('precision') is None and r.get('framewise_sensitivity') is None and r.get('false_positive_count') is None for r in curves if r['case_id']=='real'),'Sparse real labels used as exhaustive truth')
    manifest=load(root/'report/manifest.json');qa=load(root/'report/visual_qa.json')
    require(manifest.get('visual_qa_complete') is True and qa.get('status')=='PASS' and qa.get('viewed_figures')==5,'Report visual QA incomplete')
    for record in manifest['artifacts']:verify(record,root/'report')
    for record in manifest['inputs']:verify(record)
    verify(manifest['reporter']);verify(manifest['visual_qa'],root/'report')
    tests=root/'focused_tests.xml';bind(tests);tree=ET.parse(tests).getroot()
    suites=list(tree.iter('testsuite'));test_count=sum(int(s.get('tests',0)) for s in suites)
    require(test_count==32 and all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped')),'Focused tests did not pass')
    bind(Path(__file__))
    capsule=load(root/'code_capsule/manifest.json')
    for record in capsule['files']:verify(record,root/'code_capsule')
    companion=load(root/'paper_companion_binding.json');verify(companion)
    result=dict(status='PASS',completed_utc=datetime.now(timezone.utc).isoformat(),
        counts=dict(cells=136,arms=8,datasets=17,threshold_rows=1360,videos=count_videos,roi_traces=count_traces,comparison_traces=count_comparisons,figures=5,focused_tests=test_count),
        numerical_complete=True,scientific_artifact_audit_complete=True,report_visual_qa_complete=True,
        scope='Fresh metadata, score-table, source-code and report hashes. Full movie/array hashes inherited from the completed aggregate validator, which ran after all per-cell full-decode and exact RGB/source checks. No fresh movie decoding in this closure.',
        claim_boundaries=dict(real_precision=None,real_exhaustive_sensitivity=None,real_annotation_review='pending',
            configured_monitoring='oracle-known synthetic source centers, separate thresholds and no NMS',
            independent_recording_validation=False,closed_loop_control_validation=False,universal_necessity_established=False,
            media_operating_point='q1 only; other thresholds have numerical evaluation'),
        evidence_bindings=sorted(records.values(),key=lambda r:r['path']))
    if write:
        with destination.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(status='PASS',written=write,counts=result['counts'])),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,default=DEFAULT_ROOT);parser.add_argument('--write',action='store_true')
    args=parser.parse_args();finalize(args.root,args.write)
