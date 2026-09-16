"""Source lineage and output guards for the saved-proposal study."""
import json
from pathlib import Path

import pytest

from neurobench.experiments.gamma_ls_difference import persistence_study as p


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return p.binding(path)


def source(tmp_path, cohort='real', rows=None):
    folder = tmp_path / 'cells' / 'case' / 'arm'
    rows = rows if rows is not None else [dict(proposal_id='p1', source_frame_ui=5, score=3., x_px=7, y_px=8)]
    b = save(folder / 'audit_candidates.json', rows)
    c = save(folder / 'calibration.json', dict(threshold_id='q1', threshold=2.,
        application_source_start_ui=5, application_source_stop_ui=8, eligible_area_px=100))
    s = save(folder / 'sealed.json', dict(status='SEALED_BEFORE_ACTIVITY_TRUTH_JOIN',
        audit_candidates=b, calibration=c))
    m = save(tmp_path / 'datasets/case/metadata.json', dict(source_frames_ui=list(range(1, 9)),
        setup_source_frames_ui=[2, 3, 4], application_source_frames_ui=[5, 6, 7, 8],
        original_source_offset_xy=[49, 49], truth_mode='sparse_real' if cohort == 'real' else 'fully_synthetic',
        neural_sources_generated=False))
    authority = {x['path']: x for x in [b, c, s, m]}
    return authority, b, c, s, m


@pytest.mark.parametrize('frames', [[], [2, 1], [1, 1], [1, 3], [1., 2.], [True, 2]])
def test_interval_rejects_nonconsecutive_or_noninteger(frames):
    with pytest.raises(ValueError):
        p.interval(frames, 'fixture')


def test_interval_accepts_explicit_ui_sequence():
    assert p.interval([1800, 1801, 1802], 'real') == (1800, 1802)


def test_authority_rejects_silent_replacement(tmp_path):
    a = {'path': str(tmp_path / 'f'), 'sha256': 'old', 'size_bytes': 1}
    d = {}
    p.merge_authority(d, [a])
    with pytest.raises(ValueError, match='Conflicting'):
        p.merge_authority(d, [dict(a, sha256='new')])


def test_source_uses_all_frozen_rows_and_source_offset(tmp_path):
    a, *_ = source(tmp_path)
    c = p.make_cell('real', tmp_path, 'case', 'arm', a, p.FileVerifier(), {}, {})
    assert (c['input_proposal_count'], c['first_ui'], c['last_ui'], c['offset_xy']) == (1, 5, 8, [49, 49])
    assert c['cohort'] == 'real'


def test_zero_proposal_arm_is_preserved(tmp_path):
    a, *_ = source(tmp_path, rows=[])
    c = p.make_cell('real', tmp_path, 'case', 'arm', a, p.FileVerifier(), {}, {})
    assert c['input_proposal_count'] == 0


def test_missing_metadata_authority_fails_closed(tmp_path):
    a, *_, m = source(tmp_path)
    del a[m['path']]
    with pytest.raises(ValueError, match='no completed-source authority'):
        p.make_cell('real', tmp_path, 'case', 'arm', a, p.FileVerifier(), {}, {})


def test_changed_candidates_fail_before_analysis(tmp_path):
    a, b, *_ = source(tmp_path)
    Path(b['path']).write_text('[]')
    with pytest.raises(ValueError):
        p.make_cell('real', tmp_path, 'case', 'arm', a, p.FileVerifier(), {}, {})


def test_cutoff_ties_are_not_emitted(tmp_path):
    a, *_ = source(tmp_path, rows=[dict(proposal_id='p1', source_frame_ui=5, score=2., x_px=7, y_px=8)])
    with pytest.raises(ValueError, match='cutoff'):
        p.make_cell('real', tmp_path, 'case', 'arm', a, p.FileVerifier(), {}, {})


def test_new_neural_simulation_not_accepted_as_null(tmp_path):
    a, *_, m = source(tmp_path, 'null')
    x=p.read(m['path']);x['neural_sources_generated']=True
    a[m['path']]=save(Path(m['path']),x)
    with pytest.raises(ValueError, match='source-free'):
        p.make_cell('null', tmp_path, 'case', 'arm', a, p.FileVerifier(), {}, {})


def test_existing_results_cannot_be_overwritten(tmp_path):
    p.write(tmp_path/'result.json', {'status':'PASS'})
    with pytest.raises(FileExistsError):
        p.write(tmp_path/'result.json', {'status':'REPLACED'})
    assert p.read(tmp_path/'result.json') == {'status':'PASS'}


def test_analysis_guard_precedes_loading_or_writes(tmp_path):
    p.write(tmp_path/'analysis_complete.json', {'status':'PASS'})
    with pytest.raises(ValueError, match='already complete'):
        p.run(tmp_path)


def test_review_acceptance_is_read_from_its_completed_binding(tmp_path):
    b = save(tmp_path / 'acceptance.json', {'accepted': False})
    assert p.unaccepted_review(b, {b['path']: b}, p.FileVerifier()) == b


@pytest.mark.parametrize('accepted', [True, None, 0])
def test_review_acceptance_must_remain_explicitly_false(tmp_path, accepted):
    b = save(tmp_path / 'acceptance.json', {'accepted': accepted})
    with pytest.raises(ValueError, match='acceptance changed'):
        p.unaccepted_review(b, {b['path']: b}, p.FileVerifier())
