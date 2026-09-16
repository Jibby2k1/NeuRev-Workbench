"""Small file fixtures test inherited evidence contracts, not video decoding."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from neurobench.experiments.gamma_ls_difference import persistence_audit as audit


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2)+"\n")


def binding(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)


def table(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


@pytest.fixture
def source(tmp_path):
    """Two saved coordinates, one original trace pixel; broad/full source maps."""
    original=tmp_path/'original'; original.mkdir()
    target=original/'audits/case/arm'; target.mkdir(parents=True)
    cell=original/'cells/case/arm'; cell.mkdir(parents=True)
    data=original/'datasets/case'; data.mkdir(parents=True)
    ppath=original/'protocol.json'; completion=original/'completion_manifest.json'
    model=audit.MODEL
    frames=[1,2,3,4]; app=[2,3,4]
    rows=[dict(proposal_id='p1', source_frame_ui=2, x_px=3, y_px=3, score=5.0, threshold_z=1., candidate_rank_within_frame=1),
          dict(proposal_id='p2', source_frame_ui=3, x_px=5, y_px=3, score=3.0, threshold_z=1., candidate_rank_within_frame=1)]
    op={k:1 for k in audit.OP_FIELDS}
    op.update(threshold_id='q1', threshold=1.,threshold_z=1.,threshold_frozen_from_calibration_only=True,
              application_source_start_ui=2,application_source_stop_ui=4,application_frame_count=3,
              setup_source_frames_ui=[1],scale_floor=.2,eligible_area_px=64)
    stages={}
    for name in ('Raw','Input','A','M','Spread','C','Z','Score'):
        path=data/(name+'.npy');path.write_bytes(b'fixture stages, not a real NumPy movie: '+name.encode());stages[name]=binding(path)
    put(cell/'audit_candidates.json',rows); put(cell/'calibration.json',op)
    meta=dict(source_frames_ui=frames,application_source_frames_ui=app,original_source_offset_xy=[49,49],
              evaluation_box_yxyx=[49,49,59,59],truth_mode='sparse_real')
    put(data/'metadata.json',meta)
    put(ppath,dict(frame_rate_hz=50,pixel_size_um=.5))
    seal=dict(status='SEALED_BEFORE_ACTIVITY_TRUTH_JOIN',audit_candidates=binding(cell/'audit_candidates.json'),
              calibration=binding(cell/'calibration.json'),stages=stages)
    put(cell/'sealed.json',seal)
    srcbind=dict(candidates_sha256=seal['audit_candidates']['sha256'],candidate_seal_sha256=binding(cell/'sealed.json')['sha256'],
                 original_source_offset_xy=[49,49],stage_sha256={k:stages[k]['sha256'] for k in ('Raw','Input','A','Score')},
                 numeric_stage_bindings={k:stages[k] for k in ('M','Spread','C')})
    truth=dict(truth_mode='sparse_real',real_precision_identified=False)
    contract=dict(source_binding=srcbind,source_frames_ui=frames,source_fps=50.,operating_point=op,context_frames=1,
                  fullfield_fps=10.,stage_sequence=['Raw','Input','A','Score'],display_limits={'Score':[-8,8]},truth_semantics=truth)
    put(target/'run_contract.json',contract)
    put(target/'source_manifest.json',dict(source_binding=srcbind,sources=[dict(stage=k,**r) for k,r in stages.items()],numeric_state_sources=[]))
    assignments=[dict(**r,model_roi_id='review_site_00001',distance_to_representative_px=0. if i==0 else 2.,
                      is_representative=i==0) for i,r in enumerate(rows)]
    table(target/model/'model_occurrences.csv',assignments)
    site=dict(**rows[0],model_roi_id='review_site_00001',representative_proposal_id='p1',
              member_proposal_ids=['p1','p2'],member_proposal_count=2,member_source_frames_ui=[2,3])
    put(target/model/'review_sites.json',[site])
    details=dict(**site,exact_trace_pixel_xy=[3,3],all_source_samples_in_trace=True,original_source_offset_xy=[49,49],
                 closeup_source_frames_ui=frames,crop_xyxy_half_open=[0,0,10,10])
    put(target/model/'metadata/review_site_00001.json',details)
    put(target/'1_Expert_Annotations/metadata/roi_001.json',dict(exact_trace_pixel_xy=[2,2]))
    for section,roi in [(model,'review_site_00001'),('1_Expert_Annotations','roi_001')]:
        table(target/section/f'exact_pixel_traces/{roi}.csv',[dict(source_frame_ui=f,Raw=1.,Score=2.) for f in frames])
        q=target/section/f'figures/traces/{roi}.png';q.parent.mkdir(parents=True);q.write_bytes(b'fixture trace')
    comp=target/'3_Comparison/trace_comparisons/obs1.png';comp.parent.mkdir(parents=True);comp.write_bytes(b'fixture comparison')
    videos=[]
    for section,prefix,roi in [('1_Expert_Annotations','expert',None),(model,'model',None),
                               ('1_Expert_Annotations','expert','roi_001'),(model,'model','review_site_00001')]:
        rel=f'{section}/videos/'+(f'closeups/{roi}.mp4' if roi else f'{prefix}_sequential_full_field.mp4')
        path=target/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture inherited video '+rel.encode())
        path.with_suffix('.png').write_bytes(b'fixture thumbnail '+rel.encode())
        fs=frames if roi else [1,4]
        videos.append(dict(path=str(path.resolve()),sha256=binding(path)['sha256'],thumbnail=str(path.with_suffix('.png').resolve()),
                      thumbnail_sha256=binding(path.with_suffix('.png'))['sha256'],fps=50. if roi else 10.,frame_count=len(fs),
                      decoded_frame_count=len(fs),source_frames_ui=fs,annotation_section=prefix,
                      source_rgb_stream_sha256='a'*64,decoded_rgb_stream_sha256='a'*64,
                      source_crop_xyxy_half_open=[0,0,10,10],**dict.fromkeys(audit.VIDEO_FLAGS,True)))
    put(target/'video_manifest.json',dict(videos=videos))
    summary=dict(scientific_audit_complete=True,expert_roi_count=1,expert_occurrence_count=1,model_roi_count=1,
                 model_proposal_count=2,comparison_trace_count=1,video_count=4,all_trace_sample_count=4)
    put(target/'summary.json',summary);put(target/'status.json',dict(status='complete',scientific_audit_complete=True))
    put(target/'validation.json',dict(status='passed',scientific_audit_complete=True,failures=[]))
    put(target/'inventory.json',dict(complete=True,expert_applicable=True))
    put(target/'llm_context.json',dict(scope='tiny fixture proof records, not newly decoded media'))
    plan=dict(all_model_proposal_count=2,model_roi_count=1,expert_roi_count=1,expert_occurrence_count=1,
              expected_video_count=4,expected_trace_figure_count=3,source_frame_count=4,unique_experts={'roi_001':{}},
              model_sites=[site],model_closeup_source_frames_ui={'review_site_00001':frames},fullfield_source_frames_ui=[1,4])
    put(target/'inventory_plan.json',plan)
    put(target/'coverage_manifest.json',dict(all_model_rois_rendered=True,all_expert_occurrences_compared=True,
              closeups_full_rate_all_relevant_source_frames=True,model_sites_are_biological_identities=False,
              full_duration_exact_pixel_trace_sample_count=4))
    entry=dict(cell_id='real__case__arm',cohort='real',case_id='case',arm_id='arm',audit_root=str(target.resolve()),
               candidates=binding(cell/'audit_candidates.json'),calibration=binding(cell/'calibration.json'),
               seal=binding(cell/'sealed.json'),metadata=binding(data/'metadata.json'))
    def refresh():
        # Rebind the tiny fixture when testing a semantic error, so failures are
        # distinct from the separately tested stale-byte/hash failure.
        files=sorted(p for p in target.rglob('*') if p.is_file() and p.name!='artifact_index.json')
        put(target/'artifact_index.json',dict(artifacts=[dict(binding(p),path=str(p.relative_to(target))) for p in files]))
        put(original/'audit_complete.json',dict(status='PASS',cells=1,audits=[dict(case_id='case',arm_id='arm',
            summary=json.loads((target/'summary.json').read_text()),metadata_bindings=[binding(target/n) for n in audit.METADATA])]))
        evidence=[binding(p) for p in (ppath,original/'audit_complete.json',cell/'audit_candidates.json',cell/'calibration.json',cell/'sealed.json',data/'metadata.json')]
        put(completion,dict(status='PASS',numerical_complete=True,scientific_artifact_audit_complete=True,evidence_bindings=evidence))
        entry['completion']=binding(completion)
        for key,path in [('candidates',cell/'audit_candidates.json'),('seal',cell/'sealed.json'),('calibration',cell/'calibration.json'),('metadata',data/'metadata.json')]:
            entry[key]=binding(path)
    refresh()
    return dict(root=tmp_path/'new',original=original,audit=target,cell=cell,data=data,entry=entry,refresh=refresh)


def run(source):
    return audit.audit_sources(source['root'],dict(cells=[source['entry']],expected_cells=1))


def alter(path, fn):
    d=json.loads(path.read_text());fn(d);put(path,d)


def test_all_members_exact_original_pixel_and_time_mapping(source):
    result=run(source)
    assert result['status']=='PASS' and result['candidate_rows']==2 and result['inherited_video_count']==4
    rows=json.loads((source['root']/'candidate_media_links.json').read_text())['rows']
    assert rows[0]['candidate_anchor_equals_original_trace_pixel'] is True
    assert rows[1]['candidate_anchor_equals_original_trace_pixel'] is False
    assert rows[1]['x_px']==5 and rows[1]['original_trace_x_px']==3 and rows[1]['original_source_x_px']==54
    assert rows[1]['fullfield_frame_present'] is False and rows[1]['fullfield_frame_index_zero_based'] is None
    assert rows[1]['closeup_frame_index_zero_based']==2 and rows[1]['closeup_playback_time_s']==.04
    assert rows[1]['source_time_s']==.04 and rows[1]['biological_interpretation']=='unknown_not_negative'
    assert result['sources'][0]['application_exposure_s']==.06
    assert len(result['sources'][0]['review_resources'][0]['member_proposal_ids'])==2
    assert run(source)==result  # deterministic resume rechecks every old input


def test_media_byte_tamper_and_completed_root_refused(source):
    run(source)
    (source['audit']/audit.MODEL/'videos/closeups/review_site_00001.mp4').write_bytes(b'changed')
    with pytest.raises(ValueError,match='mismatch'):run(source)
    put(source['root']/'completion_manifest.json',{'status':'PASS'})
    with pytest.raises(ValueError,match='immutable'):run(source)


@pytest.mark.parametrize('change', ['coordinate','duplicate','membership'])
def test_candidate_membership_errors(source,change):
    path=source['audit']/audit.MODEL/'model_occurrences.csv'
    with path.open() as stream:rows=list(csv.DictReader(stream))
    if change=='coordinate':rows[1]['x_px']=6
    elif change=='duplicate':rows[1]['proposal_id']='p1'
    else:rows[1]['model_roi_id']='other_site'
    table(path,rows);source['refresh']()
    with pytest.raises(ValueError,match='candidate coordinate|membership|assignment'):run(source)


@pytest.mark.parametrize('change', ['pixel','crop','frames','members'])
def test_roi_metadata_errors(source,change):
    path=source['audit']/audit.MODEL/'metadata/review_site_00001.json'
    def edit(d):
        if change=='pixel':d['exact_trace_pixel_xy']=[5,3]
        elif change=='crop':d['crop_xyxy_half_open']=[0,0,4,4]
        elif change=='frames':d['closeup_source_frames_ui']=[1,2,4]
        else:d['member_proposal_ids']=['p1']
    alter(path,edit);source['refresh']()
    with pytest.raises(ValueError,match='pixel|crop|map|metadata'):run(source)


@pytest.mark.parametrize('change', ['palette','duration_count','rgb','missingvideo'])
def test_inherited_video_proof_errors(source,change):
    def edit(d):
        if change=='palette':d['videos'][-1]['all_frames_decoded_palette_pure']=False
        elif change=='duration_count':d['videos'][-1]['decoded_frame_count']=3
        elif change=='rgb':d['videos'][-1]['decoded_rgb_stream_sha256']='b'*64
        else:d['videos'].pop()
    alter(source['audit']/'video_manifest.json',edit);source['refresh']()
    with pytest.raises(ValueError,match='video|Video|RGB|validation'):run(source)


def test_missing_indexed_trace_fails_even_with_complete_summary(source):
    (source['audit']/audit.MODEL/'figures/traces/review_site_00001.png').unlink();source['refresh']()
    with pytest.raises(ValueError,match='missing from sealed index'):run(source)


def test_no_new_threshold_or_new_row_can_reuse(source):
    alter(source['cell']/'calibration.json',lambda d:d.update(threshold_id='q2'))
    alter(source['cell']/'sealed.json',lambda d:d.update(calibration=binding(source['cell']/'calibration.json')))
    source['refresh']()
    with pytest.raises(ValueError,match='q1'):run(source)


def test_wrong_completion_owner_fails(source):
    alter(source['original']/'audit_complete.json',lambda d:d['audits'][0].update(case_id='another'))
    alter(source['original']/'completion_manifest.json',lambda d:d['evidence_bindings'].__setitem__(1,binding(source['original']/'audit_complete.json')))
    source['entry']['completion']=binding(source['original']/'completion_manifest.json')
    with pytest.raises(ValueError,match='absent'):run(source)


def test_metadata_lineage_from_bound_prior_completion(source):
    ancestor=source['original']/'ancestor';ancestor.mkdir()
    put(ancestor/'protocol.json',dict(frame_rate_hz=50,pixel_size_um=.5))
    put(ancestor/'completion_manifest.json',dict(status='PASS',numerical_complete=True,scientific_artifact_audit_complete=True,
        evidence_bindings=[binding(ancestor/'protocol.json'),source['entry']['metadata']]))
    alter(source['original']/'protocol.json',lambda d:d.update(prior_completion=binding(ancestor/'completion_manifest.json')))
    source['refresh']()
    alter(source['original']/'completion_manifest.json',lambda d:d.update(evidence_bindings=[r for r in d['evidence_bindings'] if r['path']!=source['entry']['metadata']['path']]))
    source['entry']['completion']=binding(source['original']/'completion_manifest.json')
    assert len(run(source)['source_completion_lineage'])==2


def test_original_trace_stage_parity_uses_bytes_not_owner_path(source):
    # A copied, identical stage may have a different source owner path.
    other=source['original']/'copy.npy';old=Path(json.loads((source['cell']/'sealed.json').read_text())['stages']['Raw']['path'])
    other.write_bytes(old.read_bytes())
    alter(source['cell']/'sealed.json',lambda d:d['stages'].update(Raw=binding(other)))
    source['refresh']()
    assert run(source)['status']=='PASS'


def test_stage_mismatch_fails_before_promoting(source):
    other=source['original']/'wrong.npy';other.write_bytes(b'wrong stage')
    alter(source['cell']/'sealed.json',lambda d:d['stages'].update(Raw=binding(other)))
    source['refresh']()
    with pytest.raises(ValueError,match='stage differs'):run(source)
    assert not (source['root']/'audit_reuse.json').exists()


def test_zero_proposal_arm_retains_expert_and_two_fullfields(source):
    model=source['audit']/audit.MODEL
    put(source['cell']/'audit_candidates.json',[])
    alter(source['cell']/'sealed.json',lambda d:d.update(audit_candidates=binding(source['cell']/'audit_candidates.json')))
    empty_sha=binding(source['cell']/'audit_candidates.json')['sha256']
    for name in ('run_contract.json','source_manifest.json'):
        alter(source['audit']/name,lambda d:d['source_binding'].update(candidates_sha256=empty_sha))
    table(model/'model_occurrences.csv',[],['proposal_id'])
    put(model/'review_sites.json',[])
    for directory in ('metadata','exact_pixel_traces','figures/traces','videos/closeups'):
        for p in (model/directory).glob('*'):p.unlink()
    alter(source['audit']/'video_manifest.json',lambda d:d.update(videos=d['videos'][:-1]))
    alter(source['audit']/'summary.json',lambda d:d.update(model_proposal_count=0,model_roi_count=0,video_count=3))
    alter(source['audit']/'inventory_plan.json',lambda d:d.update(all_model_proposal_count=0,model_roi_count=0,expected_video_count=3,
          expected_trace_figure_count=2,model_sites=[],model_closeup_source_frames_ui={}))
    source['refresh']()
    r=run(source)
    assert r['candidate_rows']==0 and r['inherited_video_count']==3 and r['inherited_expert_trace_count']==1


def test_no_output_overwrite_on_protocol_or_ledger_change(source):
    run(source)
    (source['root']/'candidate_media_links.tsv').write_text('different bytes')
    with pytest.raises(ValueError,match='overwrite'):run(source)


def test_declared_scope_and_unsafe_ids_fail(source):
    with pytest.raises(ValueError,match='Frozen real candidate total'):
        audit.audit_sources(source['root'],dict(cells=[source['entry']],expected_candidate_rows_by_cohort={'real':3}))
    source['entry']['cell_id']='../unsafe'
    with pytest.raises(ValueError,match='Unsafe'):run(source)


def test_compact_verification_does_not_read_large_media_or_stages(source,monkeypatch):
    result=run(source)
    old=audit.FileVerifier.binding
    def guarded(self,path):
        assert Path(path).suffix not in ('.mp4','.npy','.png') and 'exact_pixel_traces' not in str(path), 'Compact closure rehashed large/trace evidence'
        return old(self,path)
    monkeypatch.setattr(audit.FileVerifier,'binding',guarded)
    verified=audit.verify_reuse_receipt(source['root'],dict(cells=[source['entry']],expected_cells=1))
    assert verified['status']=='PASS' and verified['candidate_rows']==2
    assert verified['full_media_and_stage_file_validation'].startswith('inherited')


def test_compact_rejects_changed_roi_metadata(source):
    run(source)
    alter(source['audit']/audit.MODEL/'metadata/review_site_00001.json',lambda d:d.update(exact_trace_pixel_xy=[5,3]))
    with pytest.raises(ValueError,match='mismatch'):
        audit.verify_reuse_receipt(source['root'],dict(cells=[source['entry']],expected_cells=1))


def test_compact_rejects_extra_or_changed_link(source):
    run(source)
    path=source['root']/'candidate_media_links.json'
    alter(path,lambda d:d['rows'][1].update(original_trace_x_px=5))
    with pytest.raises(ValueError,match='mismatch'):
        audit.verify_reuse_receipt(source['root'],dict(cells=[source['entry']],expected_cells=1))


def test_null_empty_expert_inventory_keeps_model_fullfields(source):
    a=source['audit'];e=a/'1_Expert_Annotations'
    source['entry']['cohort']='null';source['entry']['cell_id']='null__case__arm'
    alter(source['data']/'metadata.json',lambda d:d.update(truth_mode='fully_synthetic'))
    alter(a/'run_contract.json',lambda d:d.update(truth_semantics=dict(truth_mode='fully_synthetic')))
    for directory in ('metadata','exact_pixel_traces','figures/traces','videos/closeups'):
        for p in (e/directory).glob('*'):p.unlink()
    (a/'3_Comparison/trace_comparisons/obs1.png').unlink()
    alter(a/'video_manifest.json',lambda d:d.update(videos=[r for r in d['videos'] if 'roi_001.mp4' not in r['path']]))
    alter(a/'summary.json',lambda d:d.update(expert_roi_count=0,expert_occurrence_count=0,comparison_trace_count=0,video_count=3))
    alter(a/'inventory.json',lambda d:d.update(expert_applicable=False))
    alter(a/'inventory_plan.json',lambda d:d.update(expert_roi_count=0,expert_occurrence_count=0,unique_experts={},
                                                  expected_video_count=3,expected_trace_figure_count=1))
    source['refresh']()
    result=run(source)
    assert result['inherited_expert_trace_count']==0 and result['inherited_video_count']==3
    assert result['cohort_counts']['null']==dict(cells=1,candidate_rows=2)


@pytest.mark.parametrize('schema,accepted', [('gamma_st_sensitivity_completion_v1',True),('unknown_schema',False)])
def test_exact_historical_completion_numeric_flag_spelling(source,schema,accepted):
    path=source['original']/'completion_manifest.json'
    def old(d):
        d.pop('numerical_complete');d.update(schema=schema,numeric_complete=True)
    alter(path,old);source['entry']['completion']=binding(path)
    if accepted:
        assert run(source)['status']=='PASS'
    else:
        with pytest.raises(ValueError,match='not complete'):run(source)


@pytest.mark.parametrize('wrong_name',[False,True])
def test_completed_audit_named_metadata_bindings(source,wrong_name):
    path=source['original']/'audit_complete.json'
    def named(d):
        row=d['audits'][0];row['metadata_bindings']={Path(r['path']).name:r for r in row['metadata_bindings']}
        if wrong_name:
            row['metadata_bindings']['wrong.json']=row['metadata_bindings'].pop('summary.json')
    alter(path,named)
    alter(source['original']/'completion_manifest.json',lambda d:d['evidence_bindings'].__setitem__(1,binding(path)))
    source['entry']['completion']=binding(source['original']/'completion_manifest.json')
    if wrong_name:
        with pytest.raises(ValueError,match='named metadata inventory'):run(source)
    else:
        assert run(source)['status']=='PASS'
        assert audit.verify_reuse_receipt(source['root'],dict(cells=[source['entry']],expected_cells=1))['status']=='PASS'
