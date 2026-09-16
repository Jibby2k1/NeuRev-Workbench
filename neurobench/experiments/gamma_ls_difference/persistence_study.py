"""Frozen, additive persistence analysis of previously emitted q1 proposals.

No new images, calcium events, scores, thresholds or biological labels are
generated. All old experiment roots remain immutable.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

from .followup_validate import FileVerifier

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'Outputs/GammaLSPersistence/persistence_20260915_r1'
SOURCES = (
    ('real', REPO / 'Outputs/GammaLSNecessity/necessity_20260914_r1',
     '85b78c42f7f8a3884c73534f6161a14c9f63ae0735eafa4ce85e8abefd88e194'),
    ('null', REPO / 'Outputs/GammaLSBackground/background_20260915_r1',
     '913528d46912ded236e2049380781f47793f0f6b31cf76c3662fafbe7a14617f'),
)
COMPONENTS = ('proposal_persistence', 'persistence_study', 'persistence_audit', 'persistence_report', 'persistence_validate')
CONFIGS = tuple((r, g) for r in (2, 4, 6) for g in (0, 1))


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    return FileVerifier().binding(path)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def tsv(path, rows):
    path = Path(path)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as stream:
        if not fields:
            stream.write('empty_table\n')
            return
        writer = csv.DictWriter(stream, fields, delimiter='\t')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(',', ':')) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def code_paths():
    return [REPO / 'neurobench/experiments/gamma_ls_difference' / (name + '.py') for name in COMPONENTS] + [
        REPO / 'tests' / ('test_' + name + '.py') for name in COMPONENTS]


def interval(values, name):
    require(isinstance(values, list) and values, f'Missing {name} frame interval')
    require(all(type(v) is int for v in values) and values == list(range(values[0], values[-1] + 1)),
            f'{name} must be consecutive, unique integer UI frames')
    return values[0], values[-1]


def merge_authority(authority, records):
    for b in records:
        key = str(Path(b['path']).resolve())
        if key in authority:
            require(authority[key] == b, f'Conflicting immutable authority: {key}')
        authority[key] = b


def bound_file(path, authority, verifier):
    key = str(Path(path).resolve())
    require(key in authority, f'File has no completed-source authority: {path}')
    return verifier.verify(authority[key])


def unaccepted_review(record, authority, verifier):
    verified = bound_file(record['path'], authority, verifier)
    require(verified == record, 'Review acceptance binding differs')
    require(read(verified['path'])['accepted'] is False, 'Review label acceptance changed')
    return verified


def make_cell(cohort, source, case, arm, authority, verifier, completion, context):
    folder = source / 'cells' / case / arm
    seal_b = bound_file(folder / 'sealed.json', authority, verifier)
    seal = read(seal_b['path'])
    require(seal['status'] == 'SEALED_BEFORE_ACTIVITY_TRUTH_JOIN', 'Unsealed input proposals')
    candidate_b = bound_file(seal['audit_candidates']['path'], authority, verifier)
    calibration_b = bound_file(seal['calibration']['path'], authority, verifier)
    require(candidate_b == seal['audit_candidates'] and calibration_b == seal['calibration'],
            'Completed candidate/calibration differs from seal')
    metadata_b = bound_file(source / 'datasets' / case / 'metadata.json', authority, verifier)
    meta, calibration = read(metadata_b['path']), read(calibration_b['path'])
    first, last = interval(meta['application_source_frames_ui'], 'application')
    setup_first, setup_last = interval(meta['setup_source_frames_ui'], 'setup')
    source_first, source_last = interval(meta['source_frames_ui'], 'source')
    require(source_first <= setup_first <= setup_last < first <= last <= source_last,
            'Setup/application/source chronology differs')
    require(calibration.get('threshold_id', 'q1') == 'q1', 'Only existing emitted q1 rows are authorized')
    require((calibration['application_source_start_ui'], calibration['application_source_stop_ui']) == (first, last),
            'Calibration and source frame windows differ')
    if cohort == 'real':
        require(meta['truth_mode'] == 'sparse_real', 'Real data identity differs')
    else:
        require(meta['truth_mode'] == 'fully_synthetic' and meta['neural_sources_generated'] is False,
                'Only already-generated source-free controls are included')
    offset = meta['original_source_offset_xy']
    require(len(offset) == 2 and all(type(v) is int for v in offset), 'Source coordinate offset differs')
    rows = read(candidate_b['path'])
    tau = calibration.get('threshold', calibration.get('threshold_z'))
    if tau is None:
        require(not rows, 'No-output threshold has emitted rows')
    else:
        require(math.isfinite(tau), 'Nonfinite native threshold')
    for row in rows:
        require(first <= row['source_frame_ui'] <= last and row['score'] > tau,
                'Candidate falls outside frozen application or cutoff')
    cell = dict(cell_id=f'{cohort}__{case}__{arm}', cohort=cohort, case_id=case, arm_id=arm,
                candidates=candidate_b, seal=seal_b, calibration=calibration_b, metadata=metadata_b,
                completion=completion, audit_root=str(source / 'audits' / case / arm),
                source_root=str(source), first_ui=first, last_ui=last,
                source_first_ui=source_first, source_last_ui=source_last,
                setup_first_ui=setup_first, setup_last_ui=setup_last,
                fps=50., pixel_um=.5, offset_xy=offset,
                eligible_area_px=calibration['eligible_area_px'],
                input_proposal_count=len(rows), threshold=tau, context=context)
    if cohort == 'real' and 'stages' in seal:
        cell['raw'] = seal['stages']['Raw']
    return cell


def preflight(root):
    root = Path(root).resolve()
    require(not root.exists(), 'Use a new noncolliding output root')
    verifier = FileVerifier()
    cells, lineage = [], []
    for cohort, source, expected_sha in SOURCES:
        completion_b = verifier.binding(source / 'completion_manifest.json')
        require(completion_b['sha256'] == expected_sha, 'Completed source version differs')
        completion = read(completion_b['path'])
        require(completion['status'] == 'PASS' and completion['scientific_artifact_audit_complete'] is True,
                'Source must have completed its full audit')
        authority = {}
        merge_authority(authority, completion['evidence_bindings'])
        p_b = bound_file(source / 'protocol.json', authority, verifier)
        p = read(p_b['path'])
        lineage.extend([completion_b, p_b])
        require(p['frame_rate_hz'] == 50 and p['pixel_size_um'] == .5, 'Acquisition calibration differs')
        if cohort == 'real':
            parent_b = verifier.verify(p['prior_completion'])
            parent = read(parent_b['path'])
            require(parent['status'] == 'PASS', 'Real-source ancestor is incomplete')
            merge_authority(authority, parent['evidence_bindings'])
            lineage.append(parent_b)
            lineage.append(unaccepted_review(p['real_annotation_acceptance'], authority, verifier))
            for arm in p['arms']:
                cells.append(make_cell(cohort, source, 'real', arm['arm_id'], authority, verifier,
                                       completion_b, arm))
        else:
            for case, arm in itertools.product(p['cases'], p['references']):
                context = {k: case[k] for k in ('seed', 'background', 'normalization', 'V', 'S', 'T')}
                cells.append(make_cell(cohort, source, case['case_id'], arm['arm_id'], authority,
                                       verifier, completion_b, context))
    require(len(cells) == len({c['cell_id'] for c in cells}) == 260, 'Physical source matrix differs')
    totals = {cohort: sum(c['input_proposal_count'] for c in cells if c['cohort'] == cohort)
              for cohort in ('real', 'null')}
    require(totals == dict(real=4061, null=99691), 'Frozen q1 proposal totals differ')
    code = [verifier.binding(f) for f in code_paths()]
    protocol = dict(schema_version=1, experiment='saved_q1_proposal_persistence',
        created_utc=datetime.now(timezone.utc).isoformat(), cells=cells,
        source_lineage=lineage, code_bindings=code,
        workflow=verifier.binding(REPO / 'docs/workflows/gamma_ls_persistence.md'),
        configurations=[dict(config_id=f'r{r}_g{g}', radius_px=r, max_missing_frames=g) for r, g in CONFIGS],
        primary=dict(radius_px=4, max_missing_frames=1, minimum_observations=3),
        frame_rate_hz=50., pixel_size_um=.5, input_proposal_totals=totals,
        expected_cells=260, expected_analyses=1560, internal_aliases_excluded=36,
        scope=dict(new_simulated_calcium_events=False, detector_rerun=False, threshold_refit=False,
                   real_biological_status='unknown; persistent groups require independent review',
                   null_status='false only under the existing exhaustive noise-only generator',
                   anchor_meaning='immutable first proposal location; algorithmic grouping, not neuronal identity',
                   study_type='within-source descriptive analysis; no new independent recording validation',
                   deployment=False),
        scientific_audit=dict(enabled=True, mode='exact inherited q1 artifacts plus complete candidate-to-media ledger',
                              no_opt_out=True, anchors_are_not_relabelled_original_trace_pixels=True),
        algorithm='Causal immutable anchors, at most one candidate per site/frame, deterministic nearest greedy association; gap-based episodes with all candidates retained.',
        review_plan='All real primary groups appear in maps and the review table; persistence alone assigns no neuron/artifact label.',
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())
    verifier.assert_unchanged()
    root.mkdir(parents=True)
    write(root / 'protocol.json', protocol)
    write(root / 'preflight.json', dict(status='PASS', protocol=binding(root / 'protocol.json'),
        physical_cells=260, analyses=1560, proposal_totals=totals,
        disk_free_bytes=shutil.disk_usage(root).free, cpu_affinity=sorted(os.sched_getaffinity(0)),
        numeric_threads=1, new_calcium_simulation=False, new_detector_evaluation=False))
    print(dict(status='PREFLIGHT_PASS', **totals), flush=True)


def load(root):
    verifier = FileVerifier()
    verifier.verify(read(root / 'preflight.json')['protocol'])
    protocol = read(root / 'protocol.json')
    for b in protocol['code_bindings'] + protocol['source_lineage']:
        verifier.verify(b)
    return protocol, verifier


def run(root):
    from .proposal_persistence import analyze
    require(not (root / 'analysis_complete.json').exists(), 'Analysis already complete; preserve outputs')
    protocol, verifier = load(root)
    summaries, artifacts = [], []
    for index, cell in enumerate(protocol['cells']):
        verifier.verify(cell['candidates'])
        rows = read(cell['candidates']['path'])
        for config in protocol['configurations']:
            folder = root / 'cells' / cell['cell_id'] / config['config_id']
            require(not folder.exists(), 'Partial results require explicit reconciliation, not overwrite')
            result = analyze(rows, first_ui=cell['first_ui'], last_ui=cell['last_ui'],
                radius_px=config['radius_px'], max_missing_frames=config['max_missing_frames'],
                fps=cell['fps'], pixel_um=cell['pixel_um'], offset_xy=cell['offset_xy'])
            require(len(result['memberships']) == len(rows), 'Persistence discarded input proposals')
            summary = dict(result['summary'])
            summary.update(cell['context'])
            summary.update(config)
            summary.update(cell_id=cell['cell_id'], cohort=cell['cohort'], case_id=cell['case_id'], arm_id=cell['arm_id'],
                           eligible_area_px=cell['eligible_area_px'])
            result['summary'] = summary
            for name in ('sites', 'episodes', 'memberships', 'summary'):
                write(folder / (name + '.json'), result[name])
                tsv(folder / (name + '.tsv'), [result[name]] if name == 'summary' else result[name])
                artifacts.extend([binding(folder / (name + suffix)) for suffix in ('.json', '.tsv')])
            summaries.append(summary)
        if (index + 1) % 10 == 0 or index + 1 == len(protocol['cells']):
            print(dict(status='PERSISTENCE_CELLS_COMPLETE', cells=index + 1, total=len(protocol['cells'])), flush=True)
    write(root / 'summary.json', summaries)
    tsv(root / 'summary.tsv', summaries)
    artifacts.extend([binding(root / 'summary.json'), binding(root / 'summary.tsv')])
    verifier.assert_unchanged()
    write(root / 'analysis_complete.json', dict(status='PASS', cells=260, analyses=len(summaries),
        input_proposal_totals=protocol['input_proposal_totals'], protocol=binding(root / 'protocol.json'),
        artifacts=artifacts, source_bindings=[c['candidates'] for c in protocol['cells']]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'run'))
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    globals()[args.command](args.root.resolve())


if __name__ == '__main__':
    main()
