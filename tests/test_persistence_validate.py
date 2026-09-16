import csv
import json
from pathlib import Path

import pytest

from neurobench.experiments.gamma_ls_difference import persistence_validate as v
from neurobench.experiments.gamma_ls_difference.proposal_persistence import analyze


def fixture(rows=None, *, gap=1, radius=4):
    if rows is None:
        rows = [dict(proposal_id=f'p{t}', source_frame_ui=t, x_px=10, y_px=20, score=5)
                for t in (1, 2, 4, 8, 9, 10)]
    cell = dict(first_ui=1, last_ui=12, fps=50., pixel_um=.5, offset_xy=[49, 49])
    config = dict(radius_px=radius, max_missing_frames=gap)
    result = analyze(rows, **cell, **config)
    return cell, config, rows, result


@pytest.mark.parametrize('gap', [0, 1])
@pytest.mark.parametrize('radius', [2, 4, 6])
def test_independent_reconciliation_accepts_all_six_configs(gap, radius):
    args = fixture(gap=gap, radius=radius)
    expected = v.check_analysis(*args)
    assert expected['proposal_count'] == 6
    assert expected['site_count'] == 1


def test_empty_results_and_zero_denominators():
    c, cfg, rows, result = fixture([])
    expected = v.check_analysis(c, cfg, rows, result)
    assert expected['proposal_count'] == expected['episode_count'] == expected['site_count'] == 0


@pytest.mark.parametrize('kind', ['drop', 'duplicate', 'score', 'frame', 'offset', 'order'])
def test_original_rows_cannot_be_lost_repeated_or_changed(kind):
    c, cfg, rows, result = fixture()
    if kind == 'drop': result['memberships'].pop()
    elif kind == 'duplicate': result['memberships'].append(result['memberships'][0].copy())
    elif kind == 'score': result['memberships'][0]['score'] += 1
    elif kind == 'frame': result['memberships'][0]['source_frame_ui'] += 1
    elif kind == 'offset': result['memberships'][0]['source_x_px'] += 1
    elif kind == 'order': result['memberships'].reverse()
    with pytest.raises(ValueError): v.check_analysis(c, cfg, rows, result)


@pytest.mark.parametrize('table,key,value', [
    ('sites', 'anchor_x_px', 11), ('sites', 'recurrence_count', 99),
    ('sites', 'inter_episode_missing_frames', [0]),
    ('episodes', 'span_frames', 88), ('episodes', 'observed_frames', 99),
    ('episodes', 'elapsed_ms', 99), ('episodes', 'occupancy', .8),
    ('episodes', 'max_consecutive_run', 88), ('episodes', 'left_censored', False),
    ('episodes', 'right_censored', True), ('episodes', 'confirmed_at_ui', 3),
    ('episodes', 'closed_at_ui', 99), ('episodes', 'qualifies_5_observations', True),
    ('memberships', 'episode_confirmed_so_far', True),
    ('memberships', 'eligible_existing_anchor_count', 1),
    ('memberships', 'created_despite_eligible_anchor_collision', True),
])
def test_arithmetic_censoring_and_causal_metadata_tamper(table, key, value):
    c, cfg, rows, result = fixture()
    result[table][0][key] = value
    with pytest.raises(ValueError): v.check_analysis(c, cfg, rows, result)


def test_in_radius_wrong_tie_rejected_by_greedy_certificate():
    rows = [dict(proposal_id=p, source_frame_ui=f, x_px=x, y_px=0, score=s)
            for p, f, x, s in [('a', 1, 0, 10), ('b', 1, 8, 5), ('c', 2, 4, 5)]]
    c, cfg, rows, result = fixture(rows)
    result['memberships'][-1]['site_id'] = 'site_000002'
    with pytest.raises(ValueError, match='greedy edge order'):
        v.check_analysis(c, cfg, rows, result)


def test_one_site_per_frame_rejected_before_summary_check():
    rows = [dict(proposal_id=p, source_frame_ui=f, x_px=x, y_px=0, score=5)
            for p, f, x in [('a', 1, 0), ('b', 2, 1), ('c', 2, -1)]]
    c, cfg, rows, result = fixture(rows)
    result['memberships'][-1]['site_id'] = result['memberships'][-2]['site_id']
    with pytest.raises(ValueError, match='one proposal'):
        v.check_analysis(c, cfg, rows, result)


@pytest.mark.parametrize('key', ['proposal_count', 'site_count', 'episode_count', 'persistent_site_count_3',
                                 'persistent_episode_count_5', 'persistent_proposal_count_10'])
def test_summary_counts_independently_recomputed(key):
    c, cfg, rows, result = fixture()
    result['summary'][key] += 1
    with pytest.raises(ValueError, match='summary'):
        v.check_analysis(c, cfg, rows, result)


def test_no_transitive_drift_and_no_cross_episode_merging():
    rows = [dict(proposal_id=p, source_frame_ui=f, x_px=x, y_px=0, score=5)
            for p, f, x in [('a', 1, 0), ('b', 2, 4), ('c', 3, 8), ('d', 8, 0)]]
    c, cfg, rows, result = fixture(rows)
    v.check_analysis(c, cfg, rows, result)
    result['memberships'][-1]['episode_id'] = result['memberships'][0]['episode_id']
    with pytest.raises(ValueError): v.check_analysis(c, cfg, rows, result)


def dump_tsv(path, rows, compact=True):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields or ['empty_table'], delimiter='\t')
        writer.writeheader()
        writer.writerows({k: json.dumps(x, separators=(',', ':') if compact else None) if isinstance(x, (dict, list)) else x
                         for k, x in row.items()} for row in rows)


@pytest.mark.parametrize('compact', [False, True])
def test_tsv_scalar_boolean_none_nested_and_tamper(tmp_path, compact):
    p = tmp_path / 'table.tsv'
    rows = [dict(flag=True, no=False, nothing=None, items=[1, 2], mapping={'x': 3}, n=5, f=5.0)]
    dump_tsv(p, rows, compact)
    v.check_tsv(p, rows, compact_nested=compact)
    p.write_text(p.read_text().replace('True', 'False'))
    with pytest.raises(ValueError, match='values'): v.check_tsv(p, rows, compact_nested=compact)


def test_empty_tsv_requires_exact_empty_header(tmp_path):
    p = tmp_path / 'empty.tsv'
    p.write_text('empty_table\n')
    v.check_tsv(p, [])
    p.write_text('empty_table\nwrong\n')
    with pytest.raises(ValueError): v.check_tsv(p, [])


def _tests_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(v, 'REPO', tmp_path / 'repo')
    e, scopes, frozen = v.Evidence(), [], {}
    for component in v.COMPONENTS:
        source = v.REPO / 'neurobench/experiments/gamma_ls_difference' / f'{component}.py'
        test = v.REPO / 'tests' / f'test_{component}.py'
        source.parent.mkdir(parents=True, exist_ok=True)
        test.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('# tiny source\n')
        test.write_text('# tiny test\n')
        junit = tmp_path / f'{component}.xml'
        junit.write_text('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="one"/></testsuite></testsuites>')
        bs, bt = e.bind(source), e.bind(test)
        frozen.update({bs['path']: bs, bt['path']: bt})
        scopes.append(dict(component=component, status='PASS', source=bs, test=bt, junit=e.bind(junit), test_count=1))
    return dict(status='PASS', scopes=scopes, total_test_count=5), e, frozen


def test_actual_five_junit_scopes_and_sources(tmp_path, monkeypatch):
    receipt, evidence, frozen = _tests_fixture(tmp_path, monkeypatch)
    assert v.verify_tests(receipt, evidence, frozen) == 5


@pytest.mark.parametrize('kind', ['count', 'duplicate', 'source', 'skipped'])
def test_receipts_cannot_invent_counts_reuse_xml_or_hide_skips(tmp_path, monkeypatch, kind):
    receipt, evidence, frozen = _tests_fixture(tmp_path, monkeypatch)
    if kind == 'count': receipt['total_test_count'] = 100
    elif kind == 'duplicate': receipt['scopes'][1]['junit'] = receipt['scopes'][0]['junit']
    elif kind == 'source': receipt['scopes'][0]['source'] = receipt['scopes'][1]['source']
    elif kind == 'skipped':
        p = Path(receipt['scopes'][0]['junit']['path'])
        p.write_text('<testsuite tests="1" skipped="1"><testcase name="one"><skipped/></testcase></testsuite>')
        receipt['scopes'][0]['junit'] = v.Evidence().bind(p)
        evidence = v.Evidence()
    with pytest.raises(ValueError): v.verify_tests(receipt, evidence, frozen)


def test_capsule_read_only_then_exclusive_copy_and_tamper(tmp_path, monkeypatch):
    _, evidence, _ = _tests_fixture(tmp_path, monkeypatch)
    root = tmp_path / 'out'
    root.mkdir()
    assert v.source_capsule(root, evidence, write=False) == 10
    assert list(root.iterdir()) == []
    assert v.source_capsule(root, evidence, write=True) == 10
    assert len(list((root / 'source_capsule').rglob('*.py'))) == 10
    assert v.source_capsule(root, v.Evidence(), write=False) == 10
    next((root / 'source_capsule').rglob('*.py')).write_text('changed')
    with pytest.raises(ValueError, match='capsule differs'):
        v.source_capsule(root, v.Evidence(), write=False)


def test_exclusive_completion_never_overwrites(tmp_path):
    p = tmp_path / 'completion_manifest.json'
    v.exclusive_json(p, {'status': 'PASS'})
    with pytest.raises(FileExistsError): v.exclusive_json(p, {'status': 'FAIL'})
    with pytest.raises(ValueError, match='Preserve'):
        v.finalize(tmp_path, write=True)
    assert json.loads(p.read_text()) == {'status': 'PASS'}


def test_invalid_preflight_is_read_only(tmp_path):
    p = tmp_path / 'preflight.json'
    p.write_text('{"status":"FAIL"}')
    before = list(tmp_path.iterdir())
    with pytest.raises(ValueError, match='Preflight'): v.finalize(tmp_path)
    assert list(tmp_path.iterdir()) == before


def test_html_decodes_ids_and_counts_member_rows_only():
    parser = v.ReviewHTML()
    parser.feed('<a href="table.tsv">table</a><section class="site" id="c__s"><p>anchor a</p>'
                '<table><tr><th>proposal</th></tr><tr><td>a&amp;b</td><td>1</td></tr>'
                '<tr><td>c</td><td><a href="x.mp4#t=1">video</a></td></tr></table></section>')
    assert parser.section_ids == ['c__s']
    assert parser.members == [('c__s', 'a&b'), ('c__s', 'c')]
    assert parser.links == ['table.tsv', 'x.mp4#t=1']


def test_real_acceptance_is_completed_authority_binding_not_boolean(tmp_path):
    p = tmp_path / 'annotation_acceptance.json'
    p.write_text('{"accepted":false}')
    b = v.Evidence().bind(p)
    assert v.verify_unaccepted_review(b, {b['path']: b}, v.Evidence()) == b
    with pytest.raises(ValueError, match='bound record'):
        v.verify_unaccepted_review(False, {}, v.Evidence())
    with pytest.raises(ValueError, match='authority'):
        v.verify_unaccepted_review(b, {}, v.Evidence())
    p.write_text('{"accepted":true}')
    b = v.Evidence().bind(p)
    with pytest.raises(ValueError, match='acceptance changed'):
        v.verify_unaccepted_review(b, {b['path']: b}, v.Evidence())


def _amendment_fixture(tmp_path, monkeypatch):
    _, _, frozen = _tests_fixture(tmp_path, monkeypatch)
    root = tmp_path / 'campaign'
    root.mkdir()
    protocol = {'code_bindings': list(frozen.values())}
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return v.Evidence().bind(path)
    protocol_b = write(root / 'protocol.json', protocol)
    old_verification = write(root / 'verification.json', {'status': 'PASS', 'historical': True})
    new_verification = write(root / 'verification_amended.json', {'status': 'PASS', 'historical': False})
    outputs = {key: write(root / path, {'status': 'PASS'}) for key, path in
               [('analysis_complete', 'analysis_complete.json'), ('audit_reuse', 'audit_reuse.json'), ('report_manifest', 'report/manifest.json')]}
    pair = {'source': v.REPO / 'neurobench/experiments/gamma_ls_difference/persistence_validate.py',
            'test': v.REPO / 'tests/test_persistence_validate.py'}
    files, replacements = [], []
    for role, source in pair.items():
        old = frozen[str(source.resolve())]
        copy = root / 'validation/initial_validator_source' / source.relative_to(v.REPO)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(source.read_bytes())
        preserved = v.Evidence().bind(copy)
        files.append(dict(original=old, preserved=preserved))
        source.write_text('# amended ' + role + '\n')
        replacements.append(dict(role=role, original=old, preserved=preserved, amended=v.Evidence().bind(source)))
    record = write(root / 'validation/initial_validator_source.json', dict(status='PRESERVED', files=files))
    amendment = dict(schema_version=1, status='FINALIZATION_ONLY', reason='Fix the historical acceptance binding check.',
        protocol=protocol_b, preservation_record=record, unchanged_outputs=outputs,
        original_verification=old_verification, verification=new_verification, replacements=replacements)
    write(root / 'validation_amendment.json', amendment)
    return root, protocol, amendment


def test_amendment_changes_only_validator_pair_and_selects_new_receipt(tmp_path, monkeypatch):
    root, protocol, amendment = _amendment_fixture(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    effective, result, verification = v.effective_code_bindings(root, protocol, v.Evidence())
    assert result['applied'] and verification == root / 'verification_amended.json'
    assert len(effective) == 10
    amended = {r['amended']['path']: r['amended'] for r in amendment['replacements']}
    for old in protocol['code_bindings']:
        assert effective[old['path']] == amended.get(old['path'], old)
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('kind', ['configuration', 'numeric_sources', 'extra_role', 'report_source',
                                 'other_code_mutation', 'numeric_mutation', 'protocol_mutation',
                                 'preserved_mutation', 'receipt_replacement'])
def test_amendment_cannot_expand_scope_or_rewrite_protected_inputs(tmp_path, monkeypatch, kind):
    root, protocol, amendment = _amendment_fixture(tmp_path, monkeypatch)
    if kind == 'configuration': amendment['configurations'] = [{'radius_px': 9}]
    elif kind == 'numeric_sources': amendment['source_bindings'] = []
    elif kind == 'extra_role': amendment['replacements'].append(dict(amendment['replacements'][0], role='report'))
    elif kind == 'report_source':
        amendment['replacements'][0]['amended'] = next(b for b in protocol['code_bindings'] if b['path'].endswith('/persistence_report.py'))
    elif kind == 'other_code_mutation':
        Path(next(b['path'] for b in protocol['code_bindings'] if b['path'].endswith('/persistence_report.py'))).write_text('changed')
    elif kind == 'numeric_mutation': (root / 'analysis_complete.json').write_text('{"status":"changed"}')
    elif kind == 'protocol_mutation': (root / 'protocol.json').write_text('{"configurations":"changed"}')
    elif kind == 'preserved_mutation': Path(amendment['replacements'][0]['preserved']['path']).write_text('changed')
    elif kind == 'receipt_replacement': amendment['original_verification'] = amendment['verification']
    (root / 'validation_amendment.json').write_text(json.dumps(amendment))
    with pytest.raises(ValueError):
        v.effective_code_bindings(root, protocol, v.Evidence())
