import copy
import json
import math
import random

import pytest

from neurobench.experiments.gamma_ls_difference.proposal_persistence import analyze


def row(pid, frame, x=10, y=10, score=5, **extra):
    return dict(proposal_id=pid, source_frame_ui=frame, x_px=x, y_px=y, score=score, **extra)


def run(rows, **kwargs):
    options = dict(first_ui=1, last_ui=20, radius_px=4, max_missing_frames=1)
    options.update(kwargs)
    return analyze(rows, **options)


def test_fixed_anchor_prevents_transitive_spatial_drift():
    result = run([row('a', 1, 0), row('b', 2, 4), row('c', 3, 8)])
    assert [m['site_id'] for m in result['memberships']] == ['site_000001', 'site_000001', 'site_000002']
    assert [s['anchor_x_px'] for s in result['sites']] == [0, 8]
    assert result['sites'][0]['maximum_anchor_distance_px'] == 4


def test_one_proposal_per_anchor_per_frame_and_collision_is_visible():
    result = run([row('origin', 1), row('a', 2, 9, score=1), row('z', 2, 11, score=99)])
    by_id = {m['proposal_id']: m for m in result['memberships']}
    assert by_id['a']['site_id'] == 'site_000001'  # distance ties use proposal ID, not score
    assert by_id['z']['created_despite_eligible_anchor_collision']
    assert by_id['z']['eligible_existing_anchor_count'] == 1
    assert result['summary']['new_sites_due_to_anchor_collision'] == 1
    keys = [(m['site_id'], m['source_frame_ui']) for m in result['memberships']]
    assert len(keys) == len(set(keys)) == 3


def test_same_frame_births_never_associate_to_each_other():
    result = run([row('a', 1), row('b', 1)])
    assert result['summary']['site_count'] == 2
    assert all(m['eligible_existing_anchor_count'] == 0 for m in result['memberships'])


def test_greedy_not_globally_optimal_convention_is_explicit():
    result = run([row('site_a', 1, 10, score=9), row('site_b', 1, 17),
                  row('near_a', 2, 13), row('only_a', 2, 6)])
    by_id = {m['proposal_id']: m for m in result['memberships']}
    assert by_id['near_a']['site_id'] == 'site_000001'
    assert by_id['near_a']['eligible_existing_anchor_count'] == 2
    assert by_id['only_a']['created_despite_eligible_anchor_collision']
    assert result['summary']['site_count'] == 3
    assert not result['summary']['global_assignment_optimality_claimed']


def test_site_creation_order_breaks_equal_distance_tie():
    result = run([row('right', 1, 14, score=9), row('left', 1, 6), row('middle', 2, 10)])
    assert result['memberships'][-1]['site_id'] == 'site_000001'
    assert result['memberships'][-1]['eligible_existing_anchor_count'] == 2


@pytest.mark.parametrize('gap,expected', [(0, 3), (1, 2)])
def test_gap_off_by_one_and_recurrence(gap, expected):
    result = run([row('a', 2), row('b', 4), row('c', 7)], max_missing_frames=gap)
    assert result['summary']['site_count'] == 1
    assert result['summary']['episode_count'] == expected
    assert result['sites'][0]['recurrence_count'] == expected - 1
    assert result['sites'][0]['inter_episode_missing_frames'] == ([1, 2] if gap == 0 else [2])


def test_gap_changes_episode_segmentation_but_not_spatial_assignment():
    source = [row('a', 1, 0), row('b', 3, 3), row('c', 6, 4), row('d', 7, 8)]
    a, b = [run(source, max_missing_frames=g) for g in (0, 1)]
    fields = ('proposal_id', 'site_id', 'distance_to_anchor_px', 'eligible_existing_anchor_count')
    assert [tuple(m[k] for k in fields) for m in a['memberships']] == [tuple(m[k] for k in fields) for m in b['memberships']]


def test_span_observation_count_elapsed_and_confirmation_are_distinct():
    result = run([row('a', 3), row('b', 5), row('c', 6), row('d', 7)])
    e = result['episodes'][0]
    assert e['observed_frames'] == 4
    assert e['span_frames'] == 5
    assert e['elapsed_ms'] == 80
    assert e['span_ms'] == 100
    assert e['occupancy'] == 0.8
    assert e['missing_frames_within_span'] == 1
    assert e['max_consecutive_run'] == 3
    assert e['max_missing_frames_between_observations'] == 1
    assert e['confirmed_at_ui'] == 6
    assert e['confirmation_elapsed_ms'] == 60
    assert [m['episode_confirmed_so_far'] for m in result['memberships']] == [False, False, True, True]


@pytest.mark.parametrize('gap,frame,left,right,closed', [
    (1, 1, True, False, 3), (1, 2, True, False, 4), (1, 3, False, False, 5),
    (1, 8, False, False, 10), (1, 9, False, True, None), (1, 10, False, True, None),
    (0, 1, True, False, 2), (0, 2, False, False, 3),
    (0, 9, False, False, 10), (0, 10, False, True, None),
])
def test_boundary_censoring_and_causal_closure(gap, frame, left, right, closed):
    e = run([row('a', frame)], last_ui=10, max_missing_frames=gap)['episodes'][0]
    assert (e['left_censored'], e['right_censored'], e['closed_at_ui']) == (left, right, closed)


def test_input_order_invariance_and_no_input_mutation():
    source = [row(f'p{i}', 1 + i // 4, (i % 4) * 6, (i // 4) % 3,
                  score=i % 5, candidate_rank_within_frame=1 + i % 4) for i in range(40)]
    original = copy.deepcopy(source)
    expected = run(source)
    rng = random.Random(731)
    for _ in range(5):
        rng.shuffle(source)
        assert run(source) == expected
    assert sorted(source, key=lambda x: x['proposal_id']) == sorted(original, key=lambda x: x['proposal_id'])


def test_memberships_and_anchor_creation_are_prefix_causal():
    source = [row(f'p{i}', i + 1, (i % 5) * 2, score=i % 3) for i in range(15)]
    full = run(source)
    anchor_fields = ('site_id', 'creation_index', 'anchor_proposal_id', 'anchor_frame_ui',
                     'anchor_x_px', 'anchor_y_px', 'anchor_source_x_px', 'anchor_source_y_px')
    for cut in (1, 4, 8, 12):
        prefix = run([r for r in source if r['source_frame_ui'] <= cut], last_ui=cut)
        assert prefix['memberships'] == [m for m in full['memberships'] if m['source_frame_ui'] <= cut]
        assert [tuple(s[k] for k in anchor_fields) for s in prefix['sites']] == [
            tuple(s[k] for k in anchor_fields) for s in full['sites'] if s['anchor_frame_ui'] <= cut]
    assert run(source[:1], last_ui=1)['episodes'][0]['right_censored']


def test_empty_input_is_complete_zero_ledger():
    result = run([])
    assert result['sites'] == result['episodes'] == result['memberships'] == []
    assert result['summary']['proposal_count'] == result['summary']['episode_count'] == 0
    assert result['summary']['episodes_with_at_least_observations'] == {'3': 0, '5': 0, '10': 0}
    assert result['summary']['window_seconds'] == 0.4
    json.dumps(result, allow_nan=False)


def test_dense_collisions_preserve_every_row_and_one_per_site_frame():
    source = [row(f'p{frame:02d}_{i:03d}', frame, 0, 0, score=i) for frame in range(1, 11) for i in range(50)]
    result = run(source)
    assert result['summary']['proposal_count'] == 500
    assert result['summary']['site_count'] == 50
    assert result['summary']['episode_count'] == 50
    assert result['summary']['episodes_with_at_least_observations']['10'] == 50
    keys = {(m['site_id'], m['source_frame_ui']) for m in result['memberships']}
    assert len(keys) == 500
    assert {p for e in result['episodes'] for p in e['proposal_ids']} == {r['proposal_id'] for r in source}


@pytest.mark.parametrize('radius', [2, 4, 6])
def test_inclusive_radius_and_grid_boundaries(radius):
    source = [row('a', 1, -0.25, 0), row('b', 2, radius - 0.25, 0),
              row('c', 3, radius - 0.25 + 1e-9, 0)]
    result = run(source, radius_px=radius)
    assert result['memberships'][1]['site_id'] == 'site_000001'
    assert result['memberships'][1]['distance_to_anchor_px'] == radius
    assert result['memberships'][2]['site_id'] == 'site_000002'


def test_one_float_step_outside_radius_is_not_included():
    result = run([row('a', 1, 0, 0), row('b', 2, math.nextafter(4.0, math.inf), 0)])
    assert result['summary']['site_count'] == 2


def test_persistence_counts_require_one_qualifying_episode_not_total_site_visits():
    source = [row(f'a{t}', t, 0) for t in (1, 2, 10, 11)]
    source += [row(f'b{t}', t, 20) for t in (3, 4, 5)]
    source += [row(f'c{t}', t, 40) for t in range(1, 11)]
    s = run(source)['summary']
    assert s['recurrent_site_count'] == 1
    for n, expected in [(3, (2, 2, 13)), (5, (1, 1, 10)), (10, (1, 1, 10))]:
        assert tuple(s[f'persistent_{kind}_count_{n}'] for kind in ('episode', 'site', 'proposal')) == expected


def test_numeric_csv_values_and_crop_source_mapping():
    result = run([row('a', '2', '1.25', '-2.5', score='3.5', candidate_rank_within_frame='2'),
                  row('b', 3, 2.25, -2.5)], offset_xy=(49, 71), pixel_um=0.5)
    assert result['sites'][0]['anchor_source_x_px'] == 50.25
    assert result['sites'][0]['anchor_source_y_px'] == 68.5
    assert result['memberships'][1]['source_x_px'] == 51.25
    assert result['memberships'][1]['distance_to_anchor_um'] == 0.5
    assert result['memberships'][0]['candidate_rank_within_frame'] == 2
    assert result['summary']['radius_um'] == 2
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('field,value', [
    ('x_px', float('nan')), ('y_px', float('inf')), ('score', float('-inf')),
    ('x_px', True), ('source_frame_ui', 1.5), ('source_frame_ui', 0),
    ('source_frame_ui', 21), ('source_frame_ui', True), ('score', 'not a number'),
    ('proposal_id', ''), ('proposal_id', 1), ('candidate_rank_within_frame', 0),
])
def test_invalid_rows_fail_without_silent_drops(field, value):
    source = row('p', 1)
    source[field] = value
    with pytest.raises(ValueError):
        run([source])


def test_duplicate_ids_and_missing_keys_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        run([row('a', 1), row('a', 2)])
    with pytest.raises(ValueError, match='missing keys'):
        run([{'proposal_id': 'a'}])


@pytest.mark.parametrize('kwargs', [
    {'radius_px': 0}, {'radius_px': float('inf')}, {'radius_px': 1e200},
    {'max_missing_frames': -1}, {'max_missing_frames': 0.5}, {'fps': 0},
    {'pixel_um': -1}, {'first_ui': 0}, {'last_ui': 0}, {'offset_xy': (0,)},
])
def test_invalid_analysis_parameters(kwargs):
    with pytest.raises(ValueError):
        run([], **kwargs)
