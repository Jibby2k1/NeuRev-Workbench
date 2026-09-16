"""Static report of bound, already-emitted q1 proposal persistence.

No score, cutoff, association, trace, movie, or biological label is recomputed.
Only the common real setup Raw mean is read for a grayscale spatial backdrop.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np
from .followup_validate import FileVerifier

ROOT = Path(__file__).resolve().parents[3] / 'Outputs/GammaLSPersistence/persistence_20260915_r1'
REAL_ARMS = tuple(f'{frontend}_{stage}' for frontend in ('level', 'difference') for stage in ('X', 'A', 'C', 'Z'))
REFS = ('mean2of3_n3', 'mean1_n9', 'mean4of3_n9')
CONFIGS = tuple(f'r{r}_g{g}' for r in (2, 4, 6) for g in (0, 1))
FIGURE_IDS = ('real_counts', 'real_spatial_anchors', 'real_radius_gap', 'null_persistence')
COLORS = dict(X='#426d91', A='#cb841f', C='#6a6b44', Z='#995d83')
PRIMARY = 'r4_g1'
CAPTIONS = {
    'real_counts': 'Real recording, source UI1800–2359 (560 scored frames; 11.2 s). Radius 4 px (2 µm), at most one missing scored frame, at least three observed frames per qualifying episode. The fraction is qualifying-episode proposals / all emitted q1 proposals; an empty denominator is NA, not zero. All members of a qualifying episode are counted retrospectively.',
    'real_spatial_anchors': 'Every primary real spatial group, including one-observation groups, is shown at its immutable first-proposal anchor. Orange filled circles have at least one episode with three observations; open orange circles are other groups. The identical grayscale backdrop is the setup-only Raw mean at UI1600–1799; crop-local coordinates add (49,49) to reach source pixels. Marks are algorithmic groups, not identified neurons or artifact labels. Overlapping markers can hide one another; every group remains in the review table.',
    'real_radius_gap': 'All six prespecified radius/gap settings on the same emitted q1 rows. Gap is the maximum number of intervening scored frames without an assignment. Changing gap segments episodes but does not change the underlying fixed-anchor association. Fractions use all proposals in episodes with at least three observed frames, divided by all emitted proposals. Empty arms remain NA.',
    'null_persistence': 'Existing source-free noise controls only, application UI165–464 (300 frames; 6 s), primary radius 4 px / gap 1. Each marker is one of three paired seeds, not an independent frame. V, S, T denote variance-step, spatial-correlation and temporal-correlation factors. The share is qualifying-episode proposals / all emitted q1 proposals. The 36 conditioned S=T=0 aliases were excluded from the physical matrix; blank 000/100 conditioned positions refer to the identical raw-normalization records. These controls do not establish real-recording precision or neuronal sensitivity.'
}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def tsv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fields or ['empty_table'], delimiter='\t')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def fraction(numerator, denominator):
    require(0 <= numerator <= denominator, 'Invalid fraction numerator/denominator')
    return numerator / denominator if denominator else None


def decorate(row):
    result = dict(row)
    for n in (3, 5, 10):
        result[f'persistent_proposal_fraction_{n}'] = fraction(row[f'persistent_proposal_count_{n}'], row['proposal_count'])
        result[f'persistent_site_fraction_{n}'] = fraction(row[f'persistent_site_count_{n}'], row['site_count'])
    result['proposals_per_second'] = row['proposal_count'] / row['window_seconds']
    result['proposals_per_10000_um2_second'] = row['proposal_count'] * 10000 / (row['window_seconds'] * row['eligible_area_px'] * row['pixel_um']**2)
    return result


def validate_matrix(protocol, rows):
    cells = protocol['cells']
    expected = {(c['cell_id'], config) for c in cells for config in CONFIGS}
    actual = [(r['cell_id'], r['config_id']) for r in rows]
    require(len(actual) == len(set(actual)) and set(actual) == expected, 'Missing or duplicate persistence summaries')
    require(len(cells) == 260 and len(rows) == 1560, 'Expected 260 physical cells and 1560 analyses')
    require({c['arm_id'] for c in cells if c['cohort'] == 'real'} == set(REAL_ARMS), 'Real eight-arm matrix differs')
    require(Counter(c['cohort'] for c in cells) == dict(real=8, null=252), 'Real/null matrix differs')
    lookup = {c['cell_id']: c for c in cells}
    for row in rows:
        cell = lookup[row['cell_id']]
        require(row['cohort'] == cell['cohort'] and row['arm_id'] == cell['arm_id'] and row['case_id'] == cell['case_id'], 'Summary cell identity differs')
        require(row['proposal_count'] == cell['input_proposal_count'], 'A summary discarded or added proposals')
        require(row['first_ui'] == cell['first_ui'] and row['last_ui'] == cell['last_ui'], 'Summary frame window differs')
        require(row['window_frames'] == cell['last_ui'] - cell['first_ui'] + 1 and row['window_seconds'] == row['window_frames']/cell['fps'], 'Exposure denominator differs')
        require(row['config_id'] == f"r{row['radius_px']:g}_g{row['max_missing_frames']}", 'Configuration fields differ')
        require(row['site_count'] <= row['proposal_count'] and row['episode_count'] <= row['proposal_count'], 'Impossible group counts')
        for n in (3, 5, 10):
            fraction(row[f'persistent_proposal_count_{n}'], row['proposal_count'])
            fraction(row[f'persistent_site_count_{n}'], row['site_count'])
    return [decorate(row) for row in rows]


def _load(root):
    # Completion gates precede result reads and output creation.
    require((root / 'analysis_complete.json').is_file(), 'Complete analysis required before report')
    require((root / 'audit_reuse.json').is_file(), 'Complete inherited audit required before report')
    verifier = FileVerifier()
    analysis = verifier.read_json(root / 'analysis_complete.json')
    audit = verifier.read_json(root / 'audit_reuse.json')
    require(analysis.get('status') == 'PASS' and analysis.get('cells') == 260 and analysis.get('analyses') == 1560,
            'Incomplete analysis matrix')
    require(audit.get('status') == 'PASS', 'Inherited audit did not pass')
    protocol = verifier.read_json(root / 'protocol.json')
    require(protocol['primary'] == dict(radius_px=4,max_missing_frames=1,minimum_observations=3), 'Primary persistence definition differs')
    require(protocol['input_proposal_totals'] == dict(real=4061,null=99691), 'Input cohort totals differ')
    verifier.verify(analysis['protocol'])
    for binding in protocol['code_bindings'] + protocol['source_lineage']:
        verifier.verify(binding)
    authority = {str(Path(b['path']).resolve()): b for b in analysis['artifacts']}
    expected = {str((root / 'cells' / c['cell_id'] / config / (name + suffix)).resolve())
                for c in protocol['cells'] for config in CONFIGS
                for name in ('sites', 'episodes', 'memberships', 'summary') for suffix in ('.json', '.tsv')}
    expected.update(str((root / ('summary' + suffix)).resolve()) for suffix in ('.json', '.tsv'))
    require(set(authority) == expected and len(authority) == len(analysis['artifacts']), 'Analysis artifact inventory differs')
    for binding in authority.values():
        verifier.verify(binding)
    rows = validate_matrix(protocol, verifier.read_json(root / 'summary.json'))
    return protocol, rows, analysis, audit, verifier


def setup_mean(raw, *, source_first_ui, setup_first_ui, setup_last_ui, chunk=8):
    require(raw.ndim == 3 and chunk >= 1, 'Raw must be TYX and chunk positive')
    first, stop = setup_first_ui - source_first_ui, setup_last_ui - source_first_ui + 1
    require(0 <= first < stop <= len(raw), 'Setup interval outside Raw source')
    total = np.zeros(raw.shape[1:], dtype=np.float64)
    for start in range(first, stop, chunk):
        block = np.asarray(raw[start:min(start+chunk, stop)], dtype=np.float64)
        require(np.isfinite(block).all(), 'Nonfinite setup Raw values')
        total += block.sum(axis=0)
    return total / (stop-first)


def real_site_rows(cell, sites, episodes, memberships):
    groups = defaultdict(list)
    ep = {row['episode_id']: row for row in episodes}
    require(len(ep) == len(episodes), 'Duplicate episode IDs')
    for member in memberships:
        groups[member['site_id']].append(member)
    require(len({m['proposal_id'] for m in memberships}) == len(memberships) == cell['input_proposal_count'], 'Incomplete membership ledger')
    require(set(groups) == {s['site_id'] for s in sites}, 'Site/membership inventory differs')
    result = []
    for site in sites:
        parts = [ep[e] for e in site['episode_ids']]
        members = sorted(groups[site['site_id']], key=lambda m: (m['source_frame_ui'], m['proposal_id']))
        require(len(members) == site['observed_frames'] == sum(e['observed_frames'] for e in parts), 'Site proposal counts disagree')
        require(site['anchor_proposal_id'] in {m['proposal_id'] for m in members}, 'Missing anchor proposal')
        row = dict(site, cell_id=cell['cell_id'], arm_id=cell['arm_id'],
                   member_proposal_ids=[m['proposal_id'] for m in members],
                   maximum_episode_observed_frames=max(e['observed_frames'] for e in parts),
                   maximum_episode_span_frames=max(e['span_frames'] for e in parts),
                   maximum_episode_elapsed_ms=max(e['elapsed_ms'] for e in parts),
                   maximum_episode_occupancy=max(e['occupancy'] for e in parts),
                   left_censored_episode_count=sum(e['left_censored'] for e in parts),
                   right_censored_episode_count=sum(e['right_censored'] for e in parts),
                   first_confirmation_ui=min((e['confirmed_at_ui'] for e in parts if e['confirmed_at_ui'] is not None), default=None),
                   biological_status='unknown')
        for n in (3, 5, 10):
            qualifying = [e for e in parts if e['observed_frames'] >= n]
            row[f'persistent_episode_count_{n}'] = len(qualifying)
            row[f'persistent_proposal_count_{n}'] = sum(e['observed_frames'] for e in qualifying)
        result.append(row)
    return result


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                         'font.family': 'DejaVu Sans', 'savefig.facecolor': 'white'})
    return plt


def _percent(ax):
    from matplotlib.ticker import PercentFormatter
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0,.25,.5,.75,1])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y', color='.9', zorder=0)
    ax.spines[['top', 'right']].set_visible(False)


def build_figures(rows, site_rows, background):
    """Construct four figures in memory; caller alone writes PNG/PDF files."""
    plt = _plt()
    primary = {r['arm_id']: r for r in rows if r['cohort'] == 'real' and r['config_id'] == PRIMARY}
    require(set(primary) == set(REAL_ARMS), 'Incomplete primary real rows')
    figs = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={'width_ratios': [1.25, 1]})
    y = np.arange(8)
    axes[0].barh(y, [primary[a]['proposal_count'] for a in REAL_ARMS], color='.8', label='All emitted q1 proposals')
    axes[0].barh(y, [primary[a]['persistent_proposal_count_3'] for a in REAL_ARMS], color='#cb841f', label='In ≥3-observation episodes')
    axes[0].set_yticks(y, REAL_ARMS); axes[0].invert_yaxis(); axes[0].set_xlabel('Proposal count')
    axes[0].set_title('Counts at radius 4 px / gap 1')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2)
    for i, arm in enumerate(REAL_ARMS):
        row = primary[arm]; value = row['persistent_proposal_fraction_3']
        if value is None:
            axes[1].text(.01, i, 'NA — no emitted proposals', va='center', fontsize=9)
        else:
            axes[1].barh(i, value, color='#cb841f')
            axes[1].text(min(value+.02, .79), i, f"{row['persistent_proposal_count_3']}/{row['proposal_count']} ({value:.1%})", va='center', fontsize=9)
    from matplotlib.ticker import PercentFormatter
    axes[1].set_yticks(y, REAL_ARMS); axes[1].invert_yaxis(); axes[1].set_xlim(0, 1.05); axes[1].set_xticks([0,.25,.5,.75,1])
    axes[1].xaxis.set_major_formatter(PercentFormatter(1)); axes[1].set_xlabel('Share of emitted proposals')
    axes[1].set_title('Retrospective persistence fraction')
    for ax in axes: ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Real q1 proposals: 11.2 s, same recording and fixed setup cutoffs')
    fig.tight_layout(rect=(0, .065, 1, .94)); figs['real_counts'] = fig

    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex=True, sharey=True)
    lo, hi = np.quantile(background, [.005, .995])
    if hi <= lo: hi = lo+1
    spatial_order = tuple(f'{frontend}_{stage}' for stage in ('X','A','C','Z') for frontend in ('level','difference'))
    for ax, arm in zip(axes.ravel(), spatial_order):
        points = [r for r in site_rows if r['arm_id'] == arm]
        ax.imshow(background, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
        for persistent in (False, True):
            selected = [r for r in points if bool(r['persistent_episode_count_3']) == persistent]
            ax.scatter([r['anchor_x_px'] for r in selected], [r['anchor_y_px'] for r in selected],
                       s=24, linewidths=.85, marker='o', edgecolors='#ef992b',
                       facecolors='#ef992b' if persistent else 'none')
        ax.set_title(f'{arm} · {len(points)} groups'); ax.set_xlabel('Crop x (px)'); ax.set_ylabel('Crop y (px)')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker='o', linestyle='', color='#ef992b', label='Has ≥3-observation episode'),
                        Line2D([], [], marker='o', linestyle='', color='#ef992b', markerfacecolor='none', label='Other group')],
               loc='lower center', ncol=2, bbox_to_anchor=(.5, .005))
    fig.suptitle('All primary real anchors on the same setup Raw mean · groups are not neuronal identities')
    fig.tight_layout(rect=(0, .035, 1, .97)); figs['real_spatial_anchors'] = fig

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for i, frontend in enumerate(('level', 'difference')):
        for j, gap in enumerate((0, 1)):
            ax = axes[i, j]
            for stage in ('X', 'A', 'C', 'Z'):
                selected = sorted((r for r in rows if r['cohort']=='real' and r['arm_id']==f'{frontend}_{stage}' and r['max_missing_frames']==gap), key=lambda r:r['radius_px'])
                require(len(selected) == 3, 'Radius sensitivity rows missing')
                ax.plot([r['radius_px'] for r in selected], [np.nan if r['persistent_proposal_fraction_3'] is None else r['persistent_proposal_fraction_3'] for r in selected],
                        marker={'X':'o','A':'s','C':'^','Z':'D'}[stage], color=COLORS[stage], label=stage)
            ax.set_title(f'{frontend} · gap {gap}'); ax.set_xticks([2,4,6]); ax.set_xlabel('Association radius (px)')
            ax.set_ylabel('Share in ≥3-observation episodes'); _percent(ax)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=4, loc='lower center')
    fig.suptitle('Radius/gap sensitivity on unchanged real proposals; empty arms are NA')
    fig.tight_layout(rect=(0, .05, 1, .95)); figs['real_radius_gap'] = fig

    fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=True, sharey=True)
    panels = [('flat','raw'),('flat','conditioned'),('sloped','raw'),('sloped','conditioned')]
    null = [r for r in rows if r['cohort']=='null' and r['config_id']==PRIMARY]
    seeds = sorted({r['seed'] for r in null})
    require(len(null)==252 and len(seeds)==3, 'Null physical matrix differs')
    markers = ('o','s','^')
    for i, (bg, norm) in enumerate(panels):
        for j, ref in enumerate(REFS):
            ax=axes[i,j]
            for k, seed in enumerate(seeds):
                selected = [r for r in null if (r['background'],r['normalization'],r['arm_id'],r['seed'])==(bg,norm,ref,seed)]
                for row in selected:
                    x = row['V']*4+row['S']*2+row['T']; value=row['persistent_proposal_fraction_3']
                    if value is not None: ax.scatter(x+(k-1)*.15, value, s=24, marker=markers[k], color='#426d91', label=str(seed) if x==0 else None)
                    else: ax.text(x+(k-1)*.15, .035+.03*k, '∅', ha='center', fontsize=8)
            ax.set_title(f'{bg} / {norm}\n{ref}'); ax.set_xticks(range(8), [f'{x:03b}' for x in range(8)], rotation=0)
            ax.set_xlabel('VST'); ax.set_ylabel('Persistent proposal share'); ax.set_xlim(-.5,7.5); _percent(ax)
    fig.legend(handles=[Line2D([],[],marker=m,linestyle='',color='#426d91',label=str(s)) for s,m in zip(seeds,markers)], loc='lower center',ncol=3,bbox_to_anchor=(.5,.003))
    fig.suptitle('Noise-only controls: 6 s · radius 4 px / gap 1 · three seed points per physical condition')
    fig.tight_layout(rect=(0,.04,1,.955)); figs['null_persistence']=fig
    return figs


def relative_link(path, output_dir):
    return quote(os.path.relpath(Path(path), output_dir), safe='/')


def pct(value):
    return 'NA' if value is None else f'{100*value:.1f}%'


def report_text(rows, site_rows):
    real = {r['arm_id']:r for r in rows if r['cohort']=='real' and r['config_id']==PRIMARY}
    lines=['# Saved q1 proposal persistence', '',
        'This is a descriptive association of every previously emitted q1 proposal. It creates no detector scores, cutoff fits, calcium events, or biological labels. The same eight real arms contribute 4,061 proposals; existing noise-only controls contribute 99,691. Counts across arms are repeated analyses of shared sources, not independent detections.', '',
        'Primary grouping uses a 4 px (2 µm) radius around the first proposal, at most one missing scored frame between observations, and at least three observed frames per persistent episode. Every proposal is retained. Association is causal and greedily deterministic, with at most one proposal assigned to a group in a frame; it is not globally optimal object tracking. The anchor never moves, and matching is gap-independent. A later gap-separated episode at the same anchor is called recurrence. These groups do not establish neuronal identity.', '',
        'Observed frames count samples with an assigned proposal. Inclusive span counts intervening scored frames too; elapsed time is (last UI − first UI)/50 s. Three consecutive observations have 40 ms elapsed time and a 60 ms inclusive span. Confirmation is available on the third observation. Reported qualifying-episode proposal totals include earlier members retrospectively and must not be read as causal controller outputs. Window-edge episodes can be censored.', '',
        '## Real recording: one 11.2 s window', '',
        'The source window is UI1800–2359 (560 frames at 50 Hz), after setup UI1600–1799 (4 s; at most 109 allowed setup proposals, with strict-tie underfill retained). The common crop is 242×475 px; source coordinates add (49,49), and the six-pixel proposal border leaves 106,490 eligible pixels. Readouts are conditioned input X, centered target A, local contrast C=A−M, and full Z=C/max(Spread,floor), with level or signed adjacent-difference frontends. Each arm keeps its original setup-only q1 cutoff. These cutoff values have different units across stages; the realized application burdens differ. Empty arms remain part of the matrix.', '',
        '| Arm | All q1 proposals | Groups | Episodes | Recurrent groups | Persistent groups ≥3 | Proposals (≥3 obs.) | Share | Proposals (≥5 obs.) | Proposals (≥10 obs.) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in REAL_ARMS:
        r=real[arm]; lines.append(f"| {arm} | {r['proposal_count']} | {r['site_count']} | {r['episode_count']} | {r['recurrent_site_count']} | {r['persistent_site_count_3']} | {r['persistent_proposal_count_3']} | {pct(r['persistent_proposal_fraction_3'])} | {r['persistent_proposal_count_5']} | {r['persistent_proposal_count_10']} |")
    lines += ['', f'All {len(site_rows)} primary real groups and all 4,061 membership IDs are exposed in [the review page](index.html), including nonpersistent groups. The denominator of the share is all emitted q1 proposals, not groups and not the positive pre-cutoff NMS prefix. A zero denominator is undefined (NA). Full machine-readable counts, episode spans, censoring and membership assignments remain available in the bound source tables.', '',
        'Real proposal status remains unknown. Sparse annotations are incomplete, the exhaustive real-panel acceptance remains false, and persistence alone cannot distinguish a neuron from a stationary artifact, brightness structure, motion-related proposal or correlated noise. No real precision, false-positive rate, onset accuracy, cell count or feedback-control performance is inferred. Reviewing source media can generate hypotheses; it does not silently adjudicate identity.', '',
        '## Existing noise-only controls', '',
        'The 252 physical null states contain three fixed references, three paired seeds, flat/sloped backgrounds, raw/conditioned noise normalization and V/S/T factors. The 36 exact S=T=0 normalization aliases are excluded, not counted again. Each null cutoff used setup UI65–164 (2 s; at most six allowed setup proposals). Only the original 300 application frames (6 s) and original q1 proposals are analyzed. Proposals are false under these explicit source-free generators; this does not estimate false-positive probabilities in the real movie. Frames and correlated episodes are not independent replicates. Three seed points show finite-clip variation without a population confidence claim.', '',
        'The real and null windows differ in duration, field geometry, score families and generative assumptions; their raw counts are not a matched biological comparison. Temporal smoothing and correlated inputs can themselves induce persistence. Radius/gap sensitivity is descriptive, not a search for a validated neuronal operating point.', '']
    for key in FIGURE_IDS: lines += [f'![{key}](figures/{key}.png)', '', CAPTIONS[key], '']
    lines += ['## Provenance and review scope', '',
        'Generation requires complete bound analysis and inherited-audit reuse validation. Original detector artifacts, membership-to-media mappings and full-field/close-up/trace products are reused without relabeling. A new persistence anchor and the old media review-site trace pixel can differ; the review page displays both rather than pretending that the original exact-pixel trace was sampled at the new anchor. Report visual review is recorded separately; successful generation alone is not visual QA.', '']
    return '\n'.join(lines)


# The candidate-to-original-media schema is handled below after audit linkage validation.

def validate_links(root, protocol, audit, verifier):
    import hashlib
    require(audit.get('cells') == 260 and audit.get('candidate_rows') == 103752
            and audit.get('all_input_proposals_covered') is True
            and audit.get('inherited_scientific_audit_complete') is True,
            'Incomplete inherited candidate-to-media coverage')
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(audit['protocol_canonical_sha256'] == digest, 'Audit protocol differs')
    verifier.verify(audit['protocol'])
    authority = {str(Path(b['path']).resolve()): b for b in audit['artifacts']}
    target = (root/'candidate_media_links.json').resolve()
    require(str(target) in authority, 'Audit does not bind candidate media ledger')
    for b in audit['artifacts']: verifier.verify(b)
    payload = verifier.read_json(target)
    links = payload['rows']
    require(payload['protocol_canonical_sha256'] == digest and payload['row_count'] == len(links) == 103752, 'Media ledger version/count differs')
    keys = [(r['cell_id'], r['proposal_id']) for r in links]
    require(len(set(keys)) == len(keys), 'Duplicate candidate media key')
    bycell = Counter(r['cell_id'] for r in links)
    require(bycell == Counter({c['cell_id']:c['input_proposal_count'] for c in protocol['cells']}), 'Media ledger cell coverage differs')
    sources = {s['cell_id']:s for s in audit['sources']}
    require(len(sources) == len(audit['sources']) == 260 and set(sources) == {c['cell_id'] for c in protocol['cells']}, 'Audit source matrix differs')
    resources = {r['review_resource_id']:r for s in sources.values() for r in s['review_resources']}
    real = {}
    compact = {}
    for row in links:
        if row['cohort'] != 'real': continue
        source = sources[row['cell_id']]
        resource = resources[row['review_resource_id']]
        require(row['proposal_id'] in resource['member_proposal_ids'], 'Link points to a different original review site')
        require([row['original_trace_x_px'],row['original_trace_y_px']] == resource['exact_trace_pixel_xy'], 'Original trace pixel differs')
        audit_root = Path(source['audit_root'])
        for key in ('trace_csv','trace_png','closeup_video','closeup_thumbnail'):
            b=resource['artifacts'][key]
            require((audit_root/row[key]).resolve() == Path(b['path']).resolve(), 'Media path differs from its indexed binding')
            if key != 'closeup_video': compact[str(Path(b['path']).resolve())]=b
        require((audit_root/row['model_fullfield_video']).resolve()==Path(source['fullfield_video']['path']).resolve(), 'Full-field link differs')
        real[row['cell_id'],row['proposal_id']]=dict(row, audit_root=str(audit_root))
    for b in compact.values(): verifier.verify(b)
    return real, list(compact.values())


def _anchor_member_links(site, links):
    members = [links[site['cell_id'], pid] for pid in site['member_proposal_ids']]
    anchor = links[site['cell_id'],site['anchor_proposal_id']]
    require(anchor['x_px']==site['anchor_x_px'] and anchor['y_px']==site['anchor_y_px'], 'Persistence anchor differs from proposal media key')
    return anchor, members


def _resource_html(row, output_dir):
    def link(key, label, seek=None):
        url = relative_link(Path(row['audit_root'])/row[key], output_dir)
        if seek is not None: url += f'#t={seek:.6f}'
        return f'<a href="{html.escape(url,quote=True)}">{html.escape(label)}</a>'
    fields = [link('trace_png','trace'),link('trace_csv','CSV'),link('closeup_thumbnail','thumbnail'),
              link('closeup_video','close-up',row['closeup_playback_time_s'])]
    if row['fullfield_frame_present']:
        fields.append(link('model_fullfield_video','full field',row['fullfield_playback_time_s']))
    else:
        fields.append(link('model_fullfield_video','full field (this UI decimated)'))
    return ' · '.join(fields)


def render_html(site_rows, links, output_dir):
    entries=[]
    for site in site_rows:
        anchor,members = _anchor_member_links(site, links)
        detail=[]
        for row in members:
            detail.append('<tr><td>'+html.escape(row['proposal_id'])+'</td>'
                f"<td>{row['source_frame_ui']}</td><td>({row['x_px']}, {row['y_px']})</td>"
                f"<td>({row['original_trace_x_px']}, {row['original_trace_y_px']})</td>"
                f'<td>{_resource_html(row,output_dir)}</td></tr>')
        label=f"{site['arm_id']} {site['site_id']} {site['anchor_proposal_id']}"
        cellid=html.escape(site['cell_id']+'__'+site['site_id'],quote=True)
        entries.append(f'<section class="site" id="{cellid}" data-search="{html.escape(label,quote=True)}">'
            f"<h2>{html.escape(site['arm_id'])} / {html.escape(site['site_id'])}</h2>"
            f"<p><b>Unknown biological status.</b> Anchor crop ({site['anchor_x_px']}, {site['anchor_y_px']}); source ({site['anchor_source_x_px']}, {site['anchor_source_y_px']}); first UI {site['anchor_frame_ui']}. "
            f"{site['observed_frames']} observed frames across {site['episode_count']} episodes; {site['recurrence_count']} recurrences. "
            f"Episode counts with ≥3 / ≥5 / ≥10 observations: {site['persistent_episode_count_3']} / {site['persistent_episode_count_5']} / {site['persistent_episode_count_10']}.</p>"
            f"<p>Longest episode by observations: {site['maximum_episode_observed_frames']}; maximum inclusive span: {site['maximum_episode_span_frames']} frames; maximum elapsed time: {site['maximum_episode_elapsed_ms']:g} ms. "
            f"First third-observation confirmation UI: {site['first_confirmation_ui'] if site['first_confirmation_ui'] is not None else 'not reached'}. Left/right-censored episode counts: {site['left_censored_episode_count']}/{site['right_censored_episode_count']}.</p>"
            f"<p>Anchor proposal <code>{html.escape(site['anchor_proposal_id'])}</code>: {_resource_html(anchor,output_dir)}. "
            f"Original trace crop pixel is ({anchor['original_trace_x_px']}, {anchor['original_trace_y_px']}); "
            f"{'matches' if anchor['candidate_anchor_equals_original_trace_pixel'] else 'differs from'} the anchor pixel. Original review group: {html.escape(anchor['original_model_roi_id'])}.</p>"
            f'<details><summary>All {len(members)} proposal members and original media</summary><div class="scroll"><table><thead><tr><th>Proposal ID</th><th>Source UI</th><th>Proposal crop pixel</th><th>Original trace crop pixel</th><th>Original media</th></tr></thead><tbody>'
            +''.join(detail)+'</tbody></table></div></details></section>')
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>All primary real proposal groups</title><style>
body{font:16px/1.5 system-ui,sans-serif;color:#26313a;background:#fff;max-width:1160px;margin:32px auto;padding:0 20px}h1{font-size:28px}h2{font-size:20px}a{color:#235e8b}code{overflow-wrap:anywhere;font-size:13px}.site{border-top:1px solid #d8dde2;padding:16px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}td:first-child{overflow-wrap:anywhere;min-width:220px}input{width:min(95%,640px);padding:10px;font:inherit}summary{cursor:pointer;font-weight:600}img{max-width:100%;height:auto}.hidden{display:none}</style></head><body>
<h1>All primary real proposal groups</h1><p><a href="REPORT.md">Report</a> · <a href="primary_real_sites.tsv">All group rows</a> · <a href="primary_real_summary.tsv">Eight-arm summary</a> · <a href="../candidate_media_links.tsv">Complete original-media ledger</a></p>
<p>All emitted q1 proposals, radius 4 px (2 µm), gap at most one scored frame; persistence means at least three observed frames in an episode. These immutable-anchor groups are review candidates, not neurons or accepted artifact labels. No new calcium events or detector outputs were created. The exhaustive real annotations remain unaccepted. The 11.2 s window cannot establish long-term stability.</p>
<p>Every group's members are listed, including one-observation groups. Source UI is not video playback time. Close-ups compact some between-event gaps; seek links use their bound playback frame maps. Full-field videos are decimated, and absent source frames are marked explicitly. Original traces retain their original review-site pixel; a different persistence anchor has not acquired a new exact-pixel trace. Original overlays show all old proposals, not persistence filtering.</p>
<img src="figures/real_spatial_anchors.png" alt="Every primary real fixed anchor on the setup Raw mean">
<label for="filter">Search arm, group, anchor or member proposal ID</label><br><input id="filter" type="search" placeholder="e.g. difference_Z or a proposal ID"><p id="count"></p>
''' + ''.join(entries) + '''<script>const sites=[...document.querySelectorAll('.site')];const field=document.querySelector('#filter');function filter(){const q=field.value.trim().toLowerCase();let n=0;sites.forEach(s=>{const show=s.textContent.toLowerCase().includes(q);s.classList.toggle('hidden',!show);n+=show;});document.querySelector('#count').textContent=n+' of '+sites.length+' groups shown';}field.addEventListener('input',filter);filter();</script></body></html>'''


def generate(root=ROOT):
    root=Path(root).resolve()
    protocol, rows, analysis, audit, verifier = _load(root)
    require(not (root/'report').exists(), 'Preserve existing report; output already exists')
    links, compact_bindings=validate_links(root, protocol, audit, verifier)
    real_cells=[c for c in protocol['cells'] if c['cohort']=='real']
    require(len({(c['raw']['sha256'],c['raw']['size_bytes'],c['source_first_ui'],c['setup_first_ui'],c['setup_last_ui']) for c in real_cells})==1, 'Real arms do not share setup Raw bytes/window')
    raw_binding=verifier.verify(real_cells[0]['raw'])
    cell=real_cells[0]
    raw=np.load(raw_binding['path'], mmap_mode='r',allow_pickle=False)
    require(raw.shape==(cell['source_last_ui']-cell['source_first_ui']+1,242,475), 'Bound real Raw geometry differs')
    background=setup_mean(raw, source_first_ui=cell['source_first_ui'], setup_first_ui=cell['setup_first_ui'], setup_last_ui=cell['setup_last_ui'])
    del raw
    sites=[]; inputs=[verifier.binding(root/name) for name in ('protocol.json','analysis_complete.json','audit_reuse.json','candidate_media_links.json','summary.json')]
    inputs.append(raw_binding); inputs.extend(compact_bindings)
    for cell in real_cells:
        tables={}
        for name in ('sites','episodes','memberships'):
            path=root/'cells'/cell['cell_id']/PRIMARY/(name+'.json')
            tables[name]=verifier.read_json(path); inputs.append(verifier.binding(path))
        site_rows=real_site_rows(cell,**tables)
        require({(cell['cell_id'],m['proposal_id']) for m in tables['memberships']}=={key for key in links if key[0]==cell['cell_id']}, 'Review page omits emitted proposals')
        sites.extend(site_rows)
    sites.sort(key=lambda r:(REAL_ARMS.index(r['arm_id']),r['creation_index']))
    real_summary=[r for r in rows if r['cohort']=='real' and r['config_id']==PRIMARY]
    null_summary=[r for r in rows if r['cohort']=='null' and r['config_id']==PRIMARY]
    require(sum(r['observed_frames'] for r in sites)==4061, 'Review does not contain every real q1 proposal')
    output=root/'report';(output/'figures').mkdir(parents=True)
    for name,table in [('primary_real_sites',sites),('primary_real_summary',real_summary),('null_primary_summary',null_summary)]:
        write(output/(name+'.json'),table);tsv(output/(name+'.tsv'),table)
    np.save(output/'setup_raw_mean.npy',background,allow_pickle=False)
    (output/'REPORT.md').write_text(report_text(rows,sites))
    (output/'index.html').write_text(render_html(sites,links,output))
    figures=build_figures(rows,sites,background)
    require(tuple(figures)==FIGURE_IDS,'Figure inventory differs')
    inventory=[]
    plt=_plt()
    for key,fig in figures.items():
        fig.savefig(output/'figures'/(key+'.png'),dpi=150)
        fig.savefig(output/'figures'/(key+'.pdf'))
        inventory.append(dict(figure_id=key,png=f'figures/{key}.png',pdf=f'figures/{key}.pdf',caption=CAPTIONS[key]))
        plt.close(fig)
    verifier.assert_unchanged()
    artifacts=[]
    for path in sorted(output.rglob('*')):
        if path.is_file():
            b=verifier.binding(path);b['path']=str(path.relative_to(output));artifacts.append(b)
    manifest=dict(schema_version=1,status='GENERATED_PENDING_VISUAL_QA',created_utc=datetime.now(timezone.utc).isoformat(),
        numerical_complete=True,scientific_audit_complete=True,visual_qa_complete=False,
        figure_count=4,figures=inventory,artifacts=artifacts,inputs=inputs,
        reporter=verifier.binding(Path(__file__)),counts=dict(physical_cells=260,analyses=1560,real_proposals=4061,null_proposals=99691,primary_real_groups=len(sites)),
        geometry=dict(raw_mean_source=raw_binding,setup_first_ui=cell['setup_first_ui'],setup_last_ui=cell['setup_last_ui'],mean_dtype='float64',display_quantiles=[.005,.995],coordinate_frame='crop-local; original source offset (49,49)'),
        verification_scope='Bound analysis and audit reuse metadata, full analysis tables and linked compact real artifacts freshly hashed. Original media decode and video hashes inherited from audit_reuse; no new video or trace generated.',
        scope=dict(real_precision=None,real_biological_status='unknown',new_calcium_events=False,new_detector_outputs=False,persistence_groups_are_neurons=False,feedback_control_claim=False),
        visual_qa_note='Root-owned direct PNG/PDF/HTML inspection is required separately; this flag is not self-promoted by generation.')
    write(output/'manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args(); result=generate(args.root)
    print(json.dumps(dict(status=result['status'],figure_count=result['figure_count'],counts=result['counts'])))


if __name__=='__main__':
    main()
