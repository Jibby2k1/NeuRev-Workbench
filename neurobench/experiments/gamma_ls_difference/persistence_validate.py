"""Independent ledger reconciliation and fail-closed persistence finalization.

Default execution is read-only. --write exclusively adds a source capsule and
completion manifest after every gate passes. The association implementation is
never imported or executed. Old media bytes are not rehashed here: their full
validation is inherited through the exact completed-source/audit ledger chain.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.parse import unquote, urlparse

from .followup_validate import FileVerifier

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'Outputs/GammaLSPersistence/persistence_20260915_r1'
COMPONENTS = ('proposal_persistence', 'persistence_study', 'persistence_audit',
              'persistence_report', 'persistence_validate')
CONFIGS = {(r, g): f'r{r}_g{g}' for r in (2, 4, 6) for g in (0, 1)}
SOURCE_SHA = {'real': '85b78c42f7f8a3884c73534f6161a14c9f63ae0735eafa4ce85e8abefd88e194',
              'null': '913528d46912ded236e2049380781f47793f0f6b31cf76c3662fafbe7a14617f'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Evidence:
    def __init__(self):
        self.files = FileVerifier()
        self.records = {}

    def bind(self, path):
        b = self.files.binding(path)
        self.records[b['path']] = b
        return b

    def check(self, binding, base=None):
        b = self.files.verify(binding, root=base)
        self.records[b['path']] = b
        return b

    def get(self, path):
        self.bind(path)
        return self.files.read_json(path)

    def same(self, binding, path, base=None):
        require(self.check(binding, base) == self.bind(path), f'Binding owner differs: {path}')

    def walk(self, value, base=None):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                self.check(value, base)
            else:
                for child in value.values():
                    self.walk(child, base)
        elif isinstance(value, list):
            for child in value:
                self.walk(child, base)


def equal(actual, expected, label):
    if isinstance(expected, float):
        require(not isinstance(actual, bool) and isinstance(actual, (int, float))
                and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label)
    elif isinstance(expected, bool):
        require(actual is expected, label)
    elif isinstance(expected, int):
        require(type(actual) is int and actual == expected, label)
    else:
        require(actual == expected, label)


def selected(actual, expected, label):
    for key, value in expected.items():
        require(key in actual, f'{label}: missing {key}')
        equal(actual[key], value, f'{label}: {key} differs')


def unique(rows, key, label):
    require(isinstance(rows, list), f'{label} must be a list')
    result = {}
    for row in rows:
        value = row[key]
        require(value not in result, f'{label}: duplicate {value}')
        result[value] = row
    return result


def check_tsv(path, rows, evidence=None, *, compact_nested=True):
    if evidence is not None:
        evidence.bind(path)
    with Path(path).open(newline='') as stream:
        parsed = list(csv.reader(stream, delimiter='\t'))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        require(parsed == [['empty_table']], f'Empty TSV differs: {path}')
        return
    require(parsed and parsed[0] == fields and len(parsed) == len(rows) + 1,
            f'TSV header/row count differs: {path}')
    for actual, row in zip(parsed[1:], rows):
        expected = []
        for key in fields:
            value = row.get(key)
            expected.append(json.dumps(value, separators=(',', ':') if compact_nested else None) if isinstance(value, (dict, list))
                            else '' if value is None else str(value))
        require(actual == expected, f'TSV values differ: {path}')


def check_analysis(cell, config, source, result):
    """Check saved outputs directly, including the ordered-edge certificate."""
    radius, gap = config['radius_px'], config['max_missing_frames']
    first, last, fps, pixel = cell['first_ui'], cell['last_ui'], cell['fps'], cell['pixel_um']
    ox, oy = cell['offset_xy']
    original = unique(source, 'proposal_id', 'source proposals')
    ordered = sorted(source, key=lambda row: (row['source_frame_ui'], -row['score'],
                                             row['y_px'], row['x_px'], row['proposal_id']))
    members = result['memberships']
    membership = unique(members, 'proposal_id', 'memberships')
    sites = unique(result['sites'], 'site_id', 'sites')
    episodes = unique(result['episodes'], 'episode_id', 'episodes')
    require(set(membership) == set(original), 'Every original proposal must occur exactly once')
    require([m['proposal_id'] for m in members] == [r['proposal_id'] for r in ordered], 'Membership order differs')
    require(list(sites) == [f'site_{i:06d}' for i in range(1, len(sites) + 1)], 'Site creation IDs differ')
    require(list(episodes) == [f'episode_{i:06d}' for i in range(1, len(episodes) + 1)], 'Episode creation IDs differ')
    frames, by_site, by_episode = defaultdict(list), defaultdict(list), defaultdict(list)
    site_frames = set()
    for row in members:
        src = original[row['proposal_id']]
        require(type(src['source_frame_ui']) is int and first <= src['source_frame_ui'] <= last, 'Invalid source frame')
        for key in ('x_px', 'y_px', 'score'):
            require(not isinstance(src[key], bool) and math.isfinite(float(src[key])), 'Invalid source numeric value')
            require(row[key] == float(src[key]), f'Original proposal {key} changed')
        selected(row, dict(source_frame_ui=src['source_frame_ui'],
                          candidate_rank_within_frame=src.get('candidate_rank_within_frame'),
                          source_x_px=float(src['x_px']) + ox, source_y_px=float(src['y_px']) + oy), 'membership source mapping')
        require(row['site_id'] in sites and row['episode_id'] in episodes, 'Unknown site/episode membership')
        site = sites[row['site_id']]
        key = (row['site_id'], row['source_frame_ui'])
        require(key not in site_frames, 'More than one proposal per site/frame')
        site_frames.add(key)
        d2 = (row['x_px'] - site['anchor_x_px']) ** 2 + (row['y_px'] - site['anchor_y_px']) ** 2
        require(d2 <= radius ** 2, 'Proposal lies outside immutable anchor radius')
        selected(row, dict(distance_to_anchor_px=math.sqrt(d2), distance_to_anchor_um=math.sqrt(d2) * pixel), 'anchor distance')
        frames[row['source_frame_ui']].append(row)
        by_site[row['site_id']].append(row)
        by_episode[row['episode_id']].append(row)
    require(set(by_site) == set(sites) and set(by_episode) == set(episodes), 'Unused site or episode')

    # Each supplied assignment must satisfy the greedy edge certificate. This
    # checker never invokes analyze or constructs a replacement result ledger.
    spatial = defaultdict(list)
    next_site = 1
    for frame in sorted(frames):
        eligible, edges = {}, []
        for row in frames[frame]:
            pid = row['proposal_id']
            eligible[pid] = 0
            bx, by = math.floor(row['x_px'] / radius), math.floor(row['y_px'] / radius)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for sid in spatial.get((bx + dx, by + dy), ()):
                        s = sites[sid]
                        d2 = (row['x_px'] - s['anchor_x_px']) ** 2 + (row['y_px'] - s['anchor_y_px']) ** 2
                        if d2 <= radius ** 2:
                            eligible[pid] += 1
                            edges.append((d2, s['creation_index'], pid, sid))
        claimed_sites, claimed_proposals = set(), set()
        for _, _, pid, sid in sorted(edges):
            if sid not in claimed_sites and pid not in claimed_proposals:
                require(membership[pid]['site_id'] == sid and membership[pid]['created_new_site'] is False,
                        'Saved association violates greedy edge order')
                claimed_sites.add(sid)
                claimed_proposals.add(pid)
        for row in frames[frame]:
            pid, sid = row['proposal_id'], row['site_id']
            created = pid not in claimed_proposals
            selected(row, dict(eligible_existing_anchor_count=eligible[pid], created_new_site=created,
                               created_despite_eligible_anchor_collision=created and eligible[pid] > 0), 'association ambiguity')
            if created:
                require(sid == f'site_{next_site:06d}', 'New anchor creation order differs')
                selected(sites[sid], dict(creation_index=next_site, anchor_proposal_id=pid, anchor_frame_ui=frame,
                    anchor_x_px=row['x_px'], anchor_y_px=row['y_px'], anchor_source_x_px=row['source_x_px'],
                    anchor_source_y_px=row['source_y_px']), 'immutable anchor')
                spatial[(math.floor(row['x_px'] / radius), math.floor(row['y_px'] / radius))].append(sid)
                next_site += 1

    first_episode_order = list(dict.fromkeys(row['episode_id'] for row in members))
    require(first_episode_order == list(episodes), 'Episode creation order is not causal')
    for sid, rows in by_site.items():
        s = sites[sid]
        observed_episode_ids = []
        prev = None
        for row in rows:
            if prev is None or row['source_frame_ui'] - prev['source_frame_ui'] > gap + 1:
                require(row['episode_id'] not in observed_episode_ids, 'Separate visits merged into one episode')
                observed_episode_ids.append(row['episode_id'])
            else:
                require(row['episode_id'] == prev['episode_id'], 'Allowed-gap episode was split')
            prev = row
        intervening = [episodes[b]['first_ui'] - episodes[a]['last_ui'] - 1
                       for a, b in zip(observed_episode_ids, observed_episode_ids[1:])]
        selected(s, dict(first_ui=rows[0]['source_frame_ui'], last_ui=rows[-1]['source_frame_ui'],
                         observed_frames=len(rows), episode_ids=observed_episode_ids,
                         episode_count=len(observed_episode_ids), recurrence_count=len(observed_episode_ids) - 1,
                         inter_episode_missing_frames=intervening,
                         maximum_anchor_distance_px=max(row['distance_to_anchor_px'] for row in rows)), 'site totals')
    for eid, rows in by_episode.items():
        e = episodes[eid]
        fs = [row['source_frame_ui'] for row in rows]
        sid = rows[0]['site_id']
        require(all(row['site_id'] == sid for row in rows), 'Episode crosses sites')
        gaps = [b - a - 1 for a, b in zip(fs, fs[1:])]
        require(all(0 <= g <= gap for g in gaps), 'Episode exceeds allowed missing-frame gap')
        consecutive, maximum = 1, 1
        for g in gaps:
            consecutive = consecutive + 1 if g == 0 else 1
            maximum = max(maximum, consecutive)
        span = fs[-1] - fs[0] + 1
        third = fs[2] if len(fs) >= 3 else None
        expected = dict(site_id=sid, episode_index_within_site=sites[sid]['episode_ids'].index(eid) + 1,
            first_ui=fs[0], last_ui=fs[-1], observed_frames=len(fs), source_frames_ui=fs,
            proposal_ids=[row['proposal_id'] for row in rows], span_frames=span, elapsed_ms=(span - 1) * 1000 / fps,
            span_ms=span * 1000 / fps, occupancy=len(fs) / span, missing_frames_within_span=span - len(fs),
            max_consecutive_run=maximum, max_missing_frames_between_observations=max(gaps, default=0),
            confirmed_at_ui=third, confirmation_elapsed_ms=(third - fs[0]) * 1000 / fps if third is not None else None,
            left_censored=fs[0] <= first + gap, right_censored=fs[-1] + gap + 1 > last,
            closed_at_ui=fs[-1] + gap + 1 if fs[-1] + gap + 1 <= last else None)
        expected.update({key: sites[sid][key] for key in ('anchor_x_px', 'anchor_y_px', 'anchor_source_x_px', 'anchor_source_y_px')})
        expected.update({f'qualifies_{n}_observations': len(fs) >= n for n in (3, 5, 10)})
        selected(e, expected, 'episode arithmetic')
        for index, row in enumerate(rows, 1):
            selected(row, dict(episode_observations_so_far=index, episode_confirmed_so_far=index >= 3), 'causal confirmation')
    expected_summary = dict(first_ui=first, last_ui=last, window_frames=last - first + 1,
        window_seconds=(last - first + 1) / fps, radius_px=float(radius), radius_um=radius * pixel,
        max_missing_frames=gap, fps=float(fps), pixel_um=float(pixel), offset_xy=[float(ox), float(oy)],
        proposal_count=len(source), frames_with_proposals=len(frames), site_count=len(sites), episode_count=len(episodes),
        recurrent_site_count=sum(s['episode_count'] >= 2 for s in sites.values()),
        ambiguous_proposal_count=sum(m['eligible_existing_anchor_count'] > 1 for m in members),
        new_sites_due_to_anchor_collision=sum(m['created_despite_eligible_anchor_collision'] for m in members),
        biological_identity_established=False, global_assignment_optimality_claimed=False,
        independent_frame_inference_claimed=False)
    for n in (3, 5, 10):
        es = [e for e in episodes.values() if e['observed_frames'] >= n]
        expected_summary.update({f'persistent_episode_count_{n}': len(es),
            f'persistent_site_count_{n}': len({e['site_id'] for e in es}),
            f'persistent_proposal_count_{n}': sum(e['observed_frames'] for e in es)})
    expected_summary['episodes_with_at_least_observations'] = {str(n): expected_summary[f'persistent_episode_count_{n}'] for n in (3, 5, 10)}
    expected_summary['proposals_in_episodes_with_at_least_observations'] = {str(n): expected_summary[f'persistent_proposal_count_{n}'] for n in (3, 5, 10)}
    selected(result['summary'], expected_summary, 'analysis summary')
    return expected_summary


def code_paths():
    return [REPO / 'neurobench/experiments/gamma_ls_difference' / f'{name}.py' for name in COMPONENTS] + [
        REPO / 'tests' / f'test_{name}.py' for name in COMPONENTS]


def effective_code_bindings(root, protocol, evidence):
    """Permit only an explicitly bound validator/test finalization amendment."""
    originals = protocol['code_bindings']
    declared = {str(Path(b['path']).resolve()): b for b in originals}
    require(len(originals) == len(declared) == 10
            and set(declared) == {str(p.resolve()) for p in code_paths()},
            'Protocol must bind all five components and tests')
    path = root / 'validation_amendment.json'
    if not path.exists():
        return {k: evidence.check(b) for k, b in declared.items()}, dict(applied=False), root / 'verification.json'
    amendment = evidence.get(path)
    allowed_keys = {'schema_version', 'status', 'reason', 'protocol', 'preservation_record',
                    'unchanged_outputs', 'original_verification', 'verification', 'replacements'}
    require(set(amendment) == allowed_keys, 'Amendment contains missing or unauthorized scope fields')
    require(amendment['schema_version'] == 1 and amendment['status'] == 'FINALIZATION_ONLY'
            and isinstance(amendment['reason'], str) and amendment['reason'].strip(), 'Invalid finalization amendment')
    evidence.same(amendment['protocol'], root / 'protocol.json')
    outputs = {'analysis_complete': root / 'analysis_complete.json', 'audit_reuse': root / 'audit_reuse.json',
               'report_manifest': root / 'report/manifest.json'}
    require(set(amendment['unchanged_outputs']) == set(outputs), 'Amendment cannot replace scientific outputs')
    for key, filename in outputs.items():
        evidence.same(amendment['unchanged_outputs'][key], filename)
    evidence.same(amendment['original_verification'], root / 'verification.json')
    evidence.same(amendment['verification'], root / 'verification_amended.json')
    record_path = root / 'validation/initial_validator_source.json'
    evidence.same(amendment['preservation_record'], record_path)
    preservation = evidence.get(record_path)
    require(preservation['status'] == 'PRESERVED' and len(preservation['files']) == 2,
            'Original validator preservation record differs')
    if 'failed_log' in preservation:
        evidence.check(preservation['failed_log'])
    preserved = {str(Path(r['original']['path']).resolve()): r for r in preservation['files']}
    pair = {'source': REPO / 'neurobench/experiments/gamma_ls_difference/persistence_validate.py',
            'test': REPO / 'tests/test_persistence_validate.py'}
    require(set(preserved) == {str(p.resolve()) for p in pair.values()}, 'Preservation extends beyond validator pair')
    replacements = unique(amendment['replacements'], 'role', 'validator replacements')
    require(set(replacements) == set(pair), 'Exactly the validator source and test may be amended')
    effective = {}
    for role, current in pair.items():
        replacement = replacements[role]
        require(set(replacement) == {'role', 'original', 'preserved', 'amended'}, 'Unauthorized replacement fields')
        key = str(current.resolve())
        original = declared[key]
        saved = preserved[key]
        require(replacement['original'] == original == saved['original'], 'Amendment original differs from frozen protocol')
        require(replacement['preserved'] == saved['preserved'], 'Amendment preservation differs')
        preserved_path = root / 'validation/initial_validator_source' / current.relative_to(REPO)
        evidence.same(saved['preserved'], preserved_path)
        require(saved['preserved']['sha256'] == original['sha256']
                and saved['preserved']['size_bytes'] == original['size_bytes'], 'Preserved original is not byte-identical')
        evidence.same(replacement['amended'], current)
        effective[key] = evidence.check(replacement['amended'])
    for key, binding in declared.items():
        if key not in effective:
            effective[key] = evidence.check(binding)
    return effective, dict(applied=True, manifest=evidence.bind(path), reason=amendment['reason'],
        replacements=amendment['replacements'], unchanged_outputs=amendment['unchanged_outputs'],
        original_verification=amendment['original_verification'], effective_verification=amendment['verification']), root / 'verification_amended.json'


def verify_tests(receipt, evidence, frozen_code):
    require(receipt['status'] == 'PASS', 'Test receipt is incomplete')
    scopes = unique(receipt['scopes'], 'component', 'test scopes')
    require(set(scopes) == set(COMPONENTS), 'Expected exactly five tested components')
    seen_xml, total = set(), 0
    for component, record in scopes.items():
        require(record['status'] == 'PASS', 'Test component is not PASS')
        source = REPO / 'neurobench/experiments/gamma_ls_difference' / f'{component}.py'
        test = REPO / 'tests' / f'test_{component}.py'
        for name, path in [('source', source), ('test', test)]:
            evidence.same(record[name], path)
            require(evidence.check(record[name]) == frozen_code[str(path.resolve())], 'Tested code differs from protocol')
        xml = evidence.check(record['junit'])['path']
        require(xml not in seen_xml, 'One JUnit receipt cannot count as multiple scopes')
        seen_xml.add(xml)
        tree = ET.parse(xml)
        cases = list(tree.iter('testcase'))
        require(cases and not any(list(tree.iter(tag)) for tag in ('failure', 'error', 'skipped')), 'JUnit contains failures, skips, or no cases')
        for node in tree.iter('testsuite'):
            require(all(int(node.get(k, '0')) == 0 for k in ('failures', 'errors', 'skipped')), 'JUnit summary is not PASS')
        require(len(cases) == record['test_count'], 'Actual JUnit test count differs')
        total += len(cases)
    require(total == receipt['total_test_count'], 'Total test count differs from actual JUnit cases')
    return total


def _source_authority(protocol, evidence):
    authority = {}
    lineage = protocol['source_lineage']
    for b in lineage:
        evidence.check(b)
        if Path(b['path']).name == 'completion_manifest.json':
            completed = evidence.get(b['path'])
            require(completed['status'] == 'PASS', 'Original source completion is not PASS')
            for old in completed['evidence_bindings']:
                path = str(Path(old['path']).resolve())
                if path in authority:
                    require(authority[path]['sha256'] == old['sha256'], 'Conflicting source completion authority')
                authority[path] = old
    return authority


def verify_unaccepted_review(binding, authority, evidence):
    """The prior protocol stores a review-record binding, not a Boolean."""
    require(isinstance(binding, dict) and {'path', 'sha256', 'size_bytes'} <= set(binding),
            'Real annotation acceptance must be an exact bound record')
    actual = evidence.check(binding)
    require(actual['path'] in authority, 'Annotation acceptance lacks completed-source authority')
    require(actual == evidence.check(authority[actual['path']]), 'Annotation acceptance authority differs')
    require(evidence.get(actual['path']).get('accepted') is False, 'Real exhaustive annotation acceptance changed')
    return actual


def _check_source_cell(cell, authority, evidence, output_root):
    source_root = Path(cell['source_root']).resolve()
    require(not source_root.is_relative_to(output_root) and not output_root.is_relative_to(source_root),
            'Persistence output overlaps an immutable input root')
    require(cell['cohort'] in SOURCE_SHA and cell['completion']['sha256'] == SOURCE_SHA[cell['cohort']], 'Completed source version differs')
    evidence.same(cell['completion'], source_root / 'completion_manifest.json')
    evidence.same(cell['seal'], source_root / 'cells' / cell['case_id'] / cell['arm_id'] / 'sealed.json')
    evidence.same(cell['metadata'], source_root / 'datasets' / cell['case_id'] / 'metadata.json')
    complete = evidence.get(cell['completion']['path'])
    require(complete['status'] == 'PASS' and complete['scientific_artifact_audit_complete'] is True, 'Inherited source audit incomplete')
    for name in ('candidates', 'seal', 'calibration', 'metadata'):
        b = evidence.check(cell[name])
        require(b['path'] in authority and b['sha256'] == authority[b['path']]['sha256'], 'Input lacks completed-source authority')
    seal, meta, calibration = [evidence.get(cell[k]['path']) for k in ('seal', 'metadata', 'calibration')]
    require(seal['status'] == 'SEALED_BEFORE_ACTIVITY_TRUTH_JOIN', 'Original candidate seal differs')
    require(seal['audit_candidates'] == cell['candidates'] and seal['calibration'] == cell['calibration'], 'Source seal binding differs')
    app, setup, frames = [meta[k] for k in ('application_source_frames_ui', 'setup_source_frames_ui', 'source_frames_ui')]
    for values in (app, setup, frames):
        require(values and all(type(v) is int for v in values) and values == list(range(values[0], values[-1] + 1)), 'Source frame inventory is not contiguous')
    selected(cell, dict(first_ui=app[0], last_ui=app[-1], source_first_ui=frames[0], source_last_ui=frames[-1],
                        setup_first_ui=setup[0], setup_last_ui=setup[-1], fps=50., pixel_um=.5,
                        offset_xy=meta['original_source_offset_xy'], eligible_area_px=calibration['eligible_area_px']), 'source cell metadata')
    require(frames[0] <= setup[0] <= setup[-1] < app[0] <= app[-1] <= frames[-1], 'Source chronology differs')
    require(len(setup) == (200 if cell['cohort'] == 'real' else 100), 'Expected real four-second/null two-second setup')
    if cell['cohort'] == 'real':
        require(meta['truth_mode'] == 'sparse_real', 'Real truth interpretation differs')
    else:
        require(meta['truth_mode'] == 'fully_synthetic' and meta['neural_sources_generated'] is False, 'Null source interpretation differs')
    tau = calibration.get('threshold', calibration.get('threshold_z'))
    require(cell['threshold'] == tau and calibration.get('threshold_id', 'q1') == 'q1', 'Original cutoff changed')
    selected(calibration, dict(application_source_start_ui=app[0], application_source_stop_ui=app[-1]), 'calibration chronology')
    source = evidence.get(cell['candidates']['path'])
    require(len(source) == cell['input_proposal_count'] and len(unique(source, 'proposal_id', 'input rows')) == len(source), 'Source candidate count differs')
    require(all(app[0] <= r['source_frame_ui'] <= app[-1] and tau is not None and r['score'] > tau for r in source), 'Original emitted row violates native cutoff/window')
    return source


def verify_numeric(root, protocol, evidence):
    selected(protocol, dict(expected_cells=260, expected_analyses=1560, frame_rate_hz=50., pixel_size_um=.5,
                             input_proposal_totals={'real': 4061, 'null': 99691}, internal_aliases_excluded=36), 'protocol inventory')
    selected(protocol['primary'], dict(radius_px=4, max_missing_frames=1, minimum_observations=3), 'primary configuration')
    for key in ('new_simulated_calcium_events', 'detector_rerun', 'threshold_refit', 'deployment'):
        require(protocol['scope'][key] is False, f'Unauthorized scope change: {key}')
    require(protocol['scientific_audit']['enabled'] is True and protocol['scientific_audit']['no_opt_out'] is True, 'Audit contract differs')
    cells = unique(protocol['cells'], 'cell_id', 'source matrix')
    require(len(cells) == 260 and sum(c['cohort'] == 'real' for c in cells.values()) == 8
            and sum(c['cohort'] == 'null' for c in cells.values()) == 252, 'Expected eight real and252 physical null states')
    configurations = unique(protocol['configurations'], 'config_id', 'configurations')
    require({(c['radius_px'], c['max_missing_frames']): key for key, c in configurations.items()} == CONFIGS,
            'Radius/gap configuration inventory differs')
    complete = evidence.get(root / 'analysis_complete.json')
    selected(complete, dict(status='PASS', cells=260, analyses=1560, input_proposal_totals=protocol['input_proposal_totals']), 'numerical completion')
    evidence.same(complete['protocol'], root / 'protocol.json')
    expected_artifacts = {str((root / 'summary.json').resolve()), str((root / 'summary.tsv').resolve())}
    for c in cells:
        require(c not in ('.', '..') and '/' not in c and '\\' not in c, 'Unsafe cell ID')
        expected_artifacts.update(str((root / 'cells' / c / cfg / f'{name}{ext}').resolve())
            for cfg in configurations for name in ('sites', 'episodes', 'memberships', 'summary') for ext in ('.json', '.tsv'))
    actual = [evidence.check(b)['path'] for b in complete['artifacts']]
    require(len(actual) == len(set(actual)) and set(actual) == expected_artifacts, 'Numerical artifact matrix differs')
    require(all(Path(p).is_relative_to(root) for p in actual), 'New numerical artifact aliases an input root')
    require(complete['source_bindings'] == [c['candidates'] for c in protocol['cells']], 'Numerical source matrix differs')
    authority = _source_authority(protocol, evidence)
    actual_matrix = [(c['cohort'], c['case_id'], c['arm_id']) for c in cells.values()]
    require(len(set(actual_matrix)) == 260 and all(c['cell_id'] == f"{c['cohort']}__{c['case_id']}__{c['arm_id']}" for c in cells.values()), 'Source cell identities duplicated or renamed')
    expected_matrix = set()
    for cohort in ('real', 'null'):
        roots = {c['source_root'] for c in cells.values() if c['cohort'] == cohort}
        require(len(roots) == 1, 'Unexpected additional source root')
        source_protocol = Path(next(iter(roots))) / 'protocol.json'
        b = evidence.bind(source_protocol)
        require(b['path'] in authority and b['sha256'] == authority[b['path']]['sha256'], 'Original source protocol lacks completion authority')
        old = evidence.get(source_protocol)
        if cohort == 'real':
            verify_unaccepted_review(old['real_annotation_acceptance'], authority, evidence)
            expected_matrix.update((cohort, 'real', a['arm_id']) for a in old['arms'])
        else:
            expected_matrix.update((cohort, c['case_id'], a['arm_id']) for c in old['cases'] for a in old['references'])
    require(set(actual_matrix) == expected_matrix, 'Persistence matrix differs from exact completed input states')
    summaries, totals, originals, primary = [], defaultdict(int), {}, {}
    membership_count = 0
    for index, cell in enumerate(protocol['cells'], 1):
        source = _check_source_cell(cell, authority, evidence, root)
        originals[cell['cell_id']] = {r['proposal_id']: r for r in source}
        totals[cell['cohort']] += len(source)
        spatial_memberships = {}
        for config in protocol['configurations']:
            folder = root / 'cells' / cell['cell_id'] / config['config_id']
            result = {name: evidence.get(folder / f'{name}.json') for name in ('sites', 'episodes', 'memberships', 'summary')}
            for name, rows in result.items():
                check_tsv(folder / f'{name}.tsv', [rows] if name == 'summary' else rows, evidence)
            check_analysis(cell, config, source, result)
            selected(result['summary'], {**cell['context'], **config, **{k: cell[k] for k in
                ('cell_id', 'cohort', 'case_id', 'arm_id', 'eligible_area_px')}}, 'merged summary context')
            fields = ('proposal_id', 'site_id', 'created_new_site', 'distance_to_anchor_px', 'eligible_existing_anchor_count')
            association = [tuple(m[k] for k in fields) for m in result['memberships']]
            radius = config['radius_px']
            if radius in spatial_memberships:
                require(spatial_memberships[radius] == association, 'Changing gap changed spatial association')
            spatial_memberships[radius] = association
            summaries.append(result['summary'])
            membership_count += len(result['memberships'])
            if config['config_id'] == 'r4_g1' and cell['cohort'] == 'real':
                primary[cell['cell_id']] = result
        if index % 20 == 0 or index == len(cells):
            print(json.dumps(dict(status='INDEPENDENT_PERSISTENCE_CHECK', cells=index, total_cells=260)), flush=True)
    require(dict(totals) == protocol['input_proposal_totals'] and membership_count == 622512, 'Original/expanded proposal counts differ')
    aggregate = evidence.get(root / 'summary.json')
    require(aggregate == summaries, 'Combined summary differs from per-analysis results')
    check_tsv(root / 'summary.tsv', aggregate, evidence)
    return dict(cells=cells, originals=originals, primary=primary, summaries=summaries,
                membership_rows=membership_count, input_proposal_count=sum(totals.values()))


class ReviewHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.section_ids, self.members = [], [], []
        self.section = None
        self.first_td = False
        self.column = 0
        self.text = ''

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        for key in ('href', 'src'):
            if a.get(key):
                self.links.append(a[key])
        if tag == 'section' and a.get('class') == 'site':
            self.section = a['id']
            self.section_ids.append(self.section)
        elif tag == 'tr':
            self.column = 0
        elif tag == 'td' and self.section:
            self.column += 1
            self.first_td = self.column == 1
            self.text = ''

    def handle_data(self, data):
        if self.first_td:
            self.text += data

    def handle_endtag(self, tag):
        if tag == 'td' and self.first_td:
            self.members.append((self.section, self.text))
            self.first_td = False
        elif tag == 'section':
            self.section = None


def verify_report(root, protocol, numeric, audit, evidence):
    folder = root / 'report'
    manifest = evidence.get(folder / 'manifest.json')
    selected(manifest, dict(schema_version=1, status='GENERATED_PENDING_VISUAL_QA', numerical_complete=True,
                            scientific_audit_complete=True, visual_qa_complete=False, figure_count=4), 'report generation record')
    # The generation status is preserved. Separate visual QA supplies approval;
    # the finalizer must not rewrite the reporter's historical metadata.
    evidence.same(manifest['reporter'], Path(__file__).with_name('persistence_report.py'))
    evidence.walk(manifest['inputs'])
    needed_inputs = {str((root / name).resolve()) for name in
                     ('protocol.json', 'analysis_complete.json', 'audit_reuse.json', 'candidate_media_links.json', 'summary.json')}
    needed_inputs.update(str((root / 'cells' / cell_id / 'r4_g1' / f'{name}.json').resolve())
                         for cell_id in numeric['primary'] for name in ('sites', 'episodes', 'memberships'))
    require(needed_inputs <= {str(Path(b['path']).resolve()) for b in manifest['inputs']}, 'Report omits required source bindings')
    figure_ids = {'real_counts', 'real_spatial_anchors', 'real_radius_gap', 'null_persistence'}
    figures = unique(manifest['figures'], 'figure_id', 'report figures')
    require(set(figures) == figure_ids, 'Expected exactly four scientific figure pairs')
    expected_artifacts = {str((folder / name).resolve()) for name in ('REPORT.md', 'index.html', 'setup_raw_mean.npy')}
    expected_artifacts.update(str((folder / f'{name}.{ext}').resolve()) for name in
                             ('primary_real_sites', 'primary_real_summary', 'null_primary_summary') for ext in ('json', 'tsv'))
    expected_artifacts.update(str((folder / 'figures' / f'{name}.{ext}').resolve()) for name in figure_ids for ext in ('png', 'pdf'))
    artifacts = [evidence.check(b, folder) for b in manifest['artifacts']]
    require(len(artifacts) == len(expected_artifacts) and {b['path'] for b in artifacts} == expected_artifacts,
            'Report artifact inventory differs')
    require(all(Path(b['path']).is_relative_to(folder) for b in artifacts), 'Report output aliases an input path')
    for name, figure in figures.items():
        require(figure['png'] == f'figures/{name}.png' and figure['pdf'] == f'figures/{name}.pdf', 'Figure filename/ID differs')
    for cohort, name in [('real', 'primary_real_summary'), ('null', 'null_primary_summary')]:
        rows = evidence.get(folder / f'{name}.json')
        check_tsv(folder / f'{name}.tsv', rows, evidence, compact_nested=False)
        source = {r['cell_id']: r for r in numeric['summaries'] if r['cohort'] == cohort and r['config_id'] == 'r4_g1'}
        declared = unique(rows, 'cell_id', name)
        require(set(declared) == set(source), 'Report summary omits physical states')
        for cell_id, row in declared.items():
            src = source[cell_id]
            selected(row, src, 'report source summary')
            for n in (3, 5, 10):
                for unit in ('proposal', 'site'):
                    den = src[f'{unit}_count']
                    expected = src[f'persistent_{unit}_count_{n}'] / den if den else None
                    equal(row[f'persistent_{unit}_fraction_{n}'], expected, 'Report persistence fraction differs')
            selected(row, dict(proposals_per_second=src['proposal_count'] / src['window_seconds'],
                proposals_per_10000_um2_second=src['proposal_count'] * 10000 / (src['window_seconds'] * src['eligible_area_px'] * src['pixel_um'] ** 2)), 'report burden units')
    site_rows = evidence.get(folder / 'primary_real_sites.json')
    check_tsv(folder / 'primary_real_sites.tsv', site_rows, evidence, compact_nested=False)
    expected_sites = {(cell_id, s['site_id']): s for cell_id, result in numeric['primary'].items() for s in result['sites']}
    actual_sites = {(r['cell_id'], r['site_id']): r for r in site_rows}
    require(len(actual_sites) == len(site_rows) and set(actual_sites) == set(expected_sites), 'Review must include every primary real site')
    expected_html_members = []
    for key, row in actual_sites.items():
        cell_id, sid = key
        result = numeric['primary'][cell_id]
        selected(row, expected_sites[key], 'review site')
        members = sorted((m for m in result['memberships'] if m['site_id'] == sid), key=lambda m: (m['source_frame_ui'], m['proposal_id']))
        require(row['member_proposal_ids'] == [m['proposal_id'] for m in members] and row['biological_status'] == 'unknown',
                'Review member list/status differs')
        parts = [e for e in result['episodes'] if e['site_id'] == sid]
        expected = dict(arm_id=numeric['cells'][cell_id]['arm_id'],
            maximum_episode_observed_frames=max(e['observed_frames'] for e in parts),
            maximum_episode_span_frames=max(e['span_frames'] for e in parts),
            maximum_episode_elapsed_ms=max(e['elapsed_ms'] for e in parts),
            maximum_episode_occupancy=max(e['occupancy'] for e in parts),
            left_censored_episode_count=sum(e['left_censored'] for e in parts),
            right_censored_episode_count=sum(e['right_censored'] for e in parts),
            first_confirmation_ui=min((e['confirmed_at_ui'] for e in parts if e['confirmed_at_ui'] is not None), default=None))
        for n in (3, 5, 10):
            qualifying = [e for e in parts if e['observed_frames'] >= n]
            expected[f'persistent_episode_count_{n}'] = len(qualifying)
            expected[f'persistent_proposal_count_{n}'] = sum(e['observed_frames'] for e in qualifying)
        selected(row, expected, 'review episode metrics')
        expected_html_members.extend((cell_id + '__' + sid, m['proposal_id']) for m in members)
    page = ReviewHTML()
    page.feed((folder / 'index.html').read_text())
    expected_sections = {c + '__' + s for c, s in expected_sites}
    require(len(page.section_ids) == len(set(page.section_ids)) and set(page.section_ids) == expected_sections,
            'HTML site inventory differs')
    require(len(page.members) == 4061 and sorted(page.members) == sorted(expected_html_members), 'HTML member inventory differs')
    allowed = {b['path']: b for b in artifacts}
    for b in audit['artifacts']:
        allowed[str(Path(b['path']).resolve())] = b
    for source in audit['sources']:
        b = source['fullfield_video']
        allowed[str(Path(b['path']).resolve())] = b
        for resource in source['review_resources']:
            for b in resource['artifacts'].values():
                allowed[str(Path(b['path']).resolve())] = b
    for href in page.links:
        parsed = urlparse(href)
        require(not parsed.scheme and not parsed.netloc, 'Unexpected external report link')
        if not parsed.path:
            continue
        path = (folder / unquote(parsed.path)).resolve()
        require(str(path) in allowed and path.is_file() and path.stat().st_size == allowed[str(path)]['size_bytes'],
                f'HTML link missing or outside bound source inventory: {href}')
    qa = evidence.get(folder / 'visual_qa.json')
    require(qa['status'] == 'PASS', 'Report visual QA is not PASS')
    reviewed = unique(qa['figures'], 'figure_id', 'visual figure QA')
    require(set(reviewed) == figure_ids, 'Visual QA omits a figure')
    for name, record in reviewed.items():
        require(record['status'] == 'PASS', 'Figure visual QA failed')
        evidence.same(record['png'], folder / 'figures' / f'{name}.png')
        evidence.same(record['pdf'], folder / 'figures' / f'{name}.pdf')
        require(Path(record['render']['path']).resolve().is_relative_to(root), 'Inspected PDF render must be retained in the campaign')
        evidence.check(record['render'])
    evidence.same(qa['html'], folder / 'index.html')
    if 'html_screenshot' in qa:
        evidence.check(qa['html_screenshot'])
    selected(manifest['scope'], dict(real_precision=None, real_biological_status='unknown', new_calcium_events=False,
        new_detector_outputs=False, persistence_groups_are_neurons=False, feedback_control_claim=False), 'report scientific scope')
    selected(manifest['counts'], dict(physical_cells=260, analyses=1560, real_proposals=4061, null_proposals=99691,
                                      primary_real_groups=len(site_rows)), 'report displayed inventory')
    return dict(figure_pairs=4, primary_real_groups=len(site_rows), html_members=4061, local_html_links=len(page.links))


def source_capsule(root, evidence, *, write):
    plans = []
    for source in code_paths():
        original = evidence.bind(source)
        dest = root / 'source_capsule' / source.relative_to(REPO)
        require(not dest.is_symlink() and dest.resolve().is_relative_to(root / 'source_capsule'), 'Source capsule cannot alias an original')
        if dest.exists():
            copied = evidence.bind(dest)
            require(copied['sha256'] == original['sha256'] and copied['size_bytes'] == original['size_bytes'], 'Existing capsule differs')
        plans.append((source, dest, original))
    path = root / 'source_capsule_manifest.json'
    if path.exists():
        saved = evidence.get(path)
        require(saved['status'] == 'PASS' and len(saved['files']) == 10, 'Existing source capsule manifest differs')
        expected = {str(dest.resolve()): original for _, dest, original in plans}
        require({row['copy']['path'] for row in saved['files']} == set(expected), 'Capsule inventory differs')
        for row in saved['files']:
            require(row['source'] == expected[row['copy']['path']], 'Capsule original binding differs')
            evidence.check(row['copy'])
    if write:
        copied_rows = []
        for source, dest, original in plans:
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with source.open('rb') as src, dest.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
            copied = evidence.bind(dest)
            require((copied['sha256'], copied['size_bytes']) == (original['sha256'], original['size_bytes']), 'Source copy differs')
            copied_rows.append(dict(source=original, copy=copied))
        if not path.exists():
            exclusive_json(path, dict(status='PASS', files=copied_rows))
        evidence.bind(path)
    return 10


def exclusive_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def finalize(root=ROOT, *, write=False):
    started = time.monotonic()
    root = Path(root).resolve(strict=True)
    require(not write or not (root / 'completion_manifest.json').exists(), 'Preserve the completed study')
    evidence = Evidence()
    preflight = evidence.get(root / 'preflight.json')
    require(preflight['status'] == 'PASS', 'Preflight is incomplete')
    evidence.same(preflight['protocol'], root / 'protocol.json')
    protocol = evidence.get(root / 'protocol.json')
    frozen, amendment, verification_path = effective_code_bindings(root, protocol, evidence)
    evidence.check(protocol['workflow'])
    # The audit module only verifies its compact saved certificate here. Its
    # full audit pass precedes this finalizer and binds all inherited media.
    from .persistence_audit import verify_reuse_receipt
    inherited = verify_reuse_receipt(root, protocol)
    audit = evidence.get(root / 'audit_reuse.json')
    selected(audit, dict(status='PASS', cells=260, candidate_rows=103752, inherited_scientific_audit_complete=True,
                         all_input_proposals_covered=True, new_media_rendered=False), 'inherited audit receipt')
    evidence.same(audit['protocol'], root / 'protocol.json')
    for b in audit['validation_code'] + audit['source_completion_lineage'] + audit['artifacts']:
        evidence.check(b)
    numeric = verify_numeric(root, protocol, evidence)
    links = evidence.get(root / 'candidate_media_links.json')['rows']
    link_keys = [(r['cell_id'], r['proposal_id']) for r in links]
    expected_keys = {(cid, pid) for cid, rows in numeric['originals'].items() for pid in rows}
    require(len(links) == len(set(link_keys)) == 103752 and set(link_keys) == expected_keys, 'Original media ledger omits proposals')
    for row in links:
        cell = numeric['cells'][row['cell_id']]
        src = numeric['originals'][row['cell_id']][row['proposal_id']]
        selected(row, dict(cohort=cell['cohort'], case_id=cell['case_id'], arm_id=cell['arm_id'],
            source_frame_ui=src['source_frame_ui'], x_px=src['x_px'], y_px=src['y_px'], score=float(src['score']),
            original_source_x_px=src['x_px'] + cell['offset_xy'][0], original_source_y_px=src['y_px'] + cell['offset_xy'][1]), 'candidate-to-media source mapping')
    report = verify_report(root, protocol, numeric, audit, evidence)
    tests = verify_tests(evidence.get(verification_path), evidence, frozen)
    evidence.files.assert_unchanged()
    capsule_count = source_capsule(root, evidence, write=write)
    evidence.files.assert_unchanged()
    real = [r for r in numeric['summaries'] if r['cohort'] == 'real' and r['config_id'] == 'r4_g1' and r['arm_id'] in ('level_Z', 'difference_Z')]
    result = dict(schema_version=1, status='PASS', checked_utc=datetime.now(timezone.utc).isoformat(),
        validation_mode='final_write' if write else 'read_only', numerical_reconciliation_complete=True,
        scientific_artifact_audit_complete=True, inherited_audit_scope='Existing source/artifact byte validation and encoded-media proofs are inherited through the bound audit_reuse receipt; finalization verifies its compact certificate, current input ledgers and new artifacts.',
        report_visual_qa_complete=True, source_capsule_complete=bool(write or (root / 'source_capsule_manifest.json').exists()),
        counts=dict(physical_cells=260, configurations=6, analyses=1560, input_proposals=103752,
                    reconciled_memberships=622512, test_count=tests, source_capsule_files=capsule_count, **report),
        real_primary_Z=real, inherited_audit_validation=inherited, validation_amendment=amendment,
        scientific_scope=dict(new_simulated_calcium_events=False, new_detector_outputs=False, threshold_refit=False,
            real_groups='unknown biological identity; review candidates only', null_groups='algorithmic recurrence under existing synthetic null generators',
            setup_seconds={'real': 4, 'null': 2}, sensitivity_estimated=False, precision_estimated=False,
            first_confirmation='third observation, not a controller trigger', independent_frames_assumed=False),
        validator=evidence.bind(Path(__file__)), validation_dependency=evidence.bind(Path(__file__).with_name('followup_validate.py')),
        evidence_bindings=list(evidence.records.values()), runtime_seconds=time.monotonic() - started)
    if write:
        exclusive_json(root / 'completion_manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    result = finalize(args.root, write=args.write)
    print(json.dumps({k: result[k] for k in ('status', 'validation_mode', 'counts', 'real_primary_Z')}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
