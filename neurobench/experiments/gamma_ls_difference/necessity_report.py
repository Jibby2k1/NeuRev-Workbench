"""Small, source-bound report for the frozen eight-arm necessity comparison."""
from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from .necessity_study import DEFAULT_ROOT, PRIOR, arms, read, binding, write_json, write_tsv, DEADLINES


def ratio(a,b):return a/b if b else None


def pool(rows):
    tp=sum(r['true_positive_count'] for r in rows)
    fp=sum(r['false_positive_count'] for r in rows)
    active=sum(r['active_region_frame_count'] for r in rows)
    events=sum(r['event_count'] for r in rows)
    proposals=sum(r['proposal_count'] for r in rows)
    recovered=sum(r['recovered_event_count'] for r in rows)
    deadlines=[dict(deadline_ms=d,event_count=events,recovered_by_deadline=sum(
        next(v['recovered_by_deadline'] for v in r['deadline_rows'] if v['deadline_ms']==d) for r in rows)) for d in DEADLINES]
    area_time=sum(r['eligible_area_px']*.25*r['exposure_seconds'] for r in rows)
    return dict(case_count=len(rows),true_positive_count=tp,false_positive_count=fp,active_region_frame_count=active,
        event_count=events,proposal_count=proposals,recovered_event_count=recovered,
        precision=ratio(tp,proposals),sensitivity=ratio(tp,active),coverage=ratio(recovered,events),
        false_proposals_per_10000um2_second=ratio(fp*10000,area_time),deadline_rows=deadlines)


def monitoring_pool(rows):
    total={k:sum(r[k] for r in rows) for k in ('true_positive_count','false_positive_count','false_negative_count','true_negative_count','roi_frame_count')}
    tp,fp,fn,tn=[total[k] for k in ('true_positive_count','false_positive_count','false_negative_count','true_negative_count')]
    total.update(precision=ratio(tp,tp+fp),sensitivity=ratio(tp,tp+fn),inactive_roi_frame_exceedance_rate=ratio(fp,fp+tn))
    total['deadline_rows']=[dict(deadline_ms=d,event_count=sum(next(v['event_count'] for v in r['deadline_rows'] if v['deadline_ms']==d) for r in rows),
        recovered_by_deadline=sum(next(v['recovered_by_deadline'] for v in r['deadline_rows'] if v['deadline_ms']==d) for r in rows)) for d in DEADLINES]
    return total


def tables(root):
    rows=read(root/'all_curves.json')
    if len(rows)!=1360:raise RuntimeError('Expected136cells x10threshold identities')
    grouped=defaultdict(list)
    for r in rows:
        if r['truth_mode']=='fully_synthetic':grouped[(r['arm_id'],r['threshold_id'])].append(r)
    pooled=[dict(arm_id=arm,threshold_id=threshold,setup_budget_per_reference_area_frame=group[0]['setup_budget_per_reference_area_frame'],**pool(group)) for (arm,threshold),group in grouped.items()]
    if any(r['case_count']!=16 for r in pooled):raise RuntimeError('Incomplete paired synthetic comparison')
    monitoring=read(root/'monitoring.json');mon=[]
    for arm in arms():
        group=[r['metrics'] for r in monitoring if r['arm_id']==arm['arm_id'] and r['metrics']['truth_mode']=='fully_synthetic']
        mon.append(dict(arm,**monitoring_pool(group)))
    return rows,pooled,mon


def pct(v):return 'undefined' if v is None else f'{100*v:.1f}%'


def generate(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root).resolve();out=root/'report'
    if (out/'manifest.json').exists():raise FileExistsError('Preserve existing report; use a new report version for changes')
    out.mkdir(parents=True,exist_ok=True)
    rows,pooled,mon=tables(root)
    write_json(out/'pooled_discovery.json',pooled);write_json(out/'pooled_monitoring.json',mon)
    fixed={r['arm_id']:r for r in pooled if r['threshold_id']=='q1'}
    real={r['arm_id']:r for r in rows if r['case_id']=='real' and r['threshold_id']=='q1'}
    ids=[a['arm_id'] for a in arms()];labels={a:('Level: ' if a.startswith('level_') else 'Change: ')+{'X':'input','A':'target','C':'contrast','Z':'Gamma-LS'}[a.rsplit('_',1)[1]] for a in ids};colors={a:plt.get_cmap('tab10')(i) for i,a in enumerate(ids)}
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'figure.dpi':130})
    figures=[]
    def save(fig,name,caption):
        fig.text(.05,.035,caption,ha='left',va='bottom',fontsize=9,wrap=True)
        fig.savefig(out/name,dpi=130,facecolor='white');plt.close(fig);figures.append(name)
    def setup(fig,title):
        fig.suptitle(title,fontsize=15,y=.98);fig.subplots_adjust(left=.08,right=.97,top=.91,bottom=.25,hspace=.35,wspace=.27)

    fig,axes=plt.subplots(2,2,figsize=(12,9));setup(fig,'Synthetic discovery across setup-calibrated proposal budgets')
    for arm in ids:
        rs=sorted((r for r in pooled if r['arm_id']==arm and r['setup_budget_per_reference_area_frame'] is not None),key=lambda r:r['setup_budget_per_reference_area_frame'])
        x=[r['setup_budget_per_reference_area_frame'] for r in rs]
        ys=[[r['precision'] for r in rs],[r['sensitivity'] for r in rs],
            [ratio(next(v['recovered_by_deadline'] for v in r['deadline_rows'] if v['deadline_ms']==200),r['event_count']) for r in rs],
            [r['false_proposals_per_10000um2_second'] for r in rs]]
        for ax,y in zip(axes.flat,ys):ax.plot(x,[np.nan if v is None else v for v in y],marker='o',ms=3,label=labels[arm],color=colors[arm])
    for ax,title in zip(axes.flat,['Precision of frame/location proposals','Active-frame sensitivity','Events recovered within 200 ms','False proposals /10,000 µm² /second']):
        ax.set_title(title);ax.set_xlabel('Target setup proposals /194,820 px /frame');ax.set_xscale('symlog',linthresh=.5);ax.grid(alpha=.2)
    for ax in list(axes.flat)[:3]:ax.set_ylim(-.03,1.03)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.105),ncol=4,frameon=False)
    save(fig,'discovery_budget_curves.png','All 16 paired simulation clips. Thresholds use setup frames only. Application burden is measured, not guaranteed.\nPrecision is undefined when no proposals are emitted; missed events remain in the recovery denominator.')

    fig,ax=plt.subplots(figsize=(11,6.5));setup(fig,'Prompt event recovery at the frozen q=1 operating point')
    for arm in ids:
        r=fixed[arm];ax.plot(DEADLINES,[ratio(v['recovered_by_deadline'],v['event_count']) for v in r['deadline_rows']],marker='o',label=labels[arm],color=colors[arm])
    ax.set_xlabel('Allowed delay from simulated fluorescence onset (ms)');ax.set_ylabel('Fraction of all simulated events recovered');ax.set_ylim(-.03,1.03);ax.grid(alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.105),ncol=4,frameon=False)
    save(fig,'prompt_recovery.png','All simulated events remain in the denominator. These are sampled signal-onset delays; computation and actuation are excluded.')

    fig,axes=plt.subplots(1,3,figsize=(13,6));setup(fig,'Configured monitoring at known simulated source centers')
    for ax,key,title in zip(axes,['precision','sensitivity','inactive_roi_frame_exceedance_rate'],['Precision','Active ROI/frame sensitivity','Inactive ROI/frame exceedance rate']):
        ax.bar(range(8),[np.nan if r[key] is None else r[key] for r in mon],color=[colors[a] for a in ids]);ax.set_xticks(range(8),[labels[a] for a in ids],rotation=60,ha='right');ax.set_ylim(0,.05 if key=='inactive_roi_frame_exceedance_rate' else 1.03);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.subplots_adjust(bottom=.35,top=.85)
    save(fig,'configured_monitoring.png','Oracle-known source coordinates; no NMS. Each center has a separate setup cutoff with a 1% exceedance budget.\nNuisance-only clips contain no configured source; their false proposals are evaluated in the discovery analysis.')

    fig,axes=plt.subplots(1,2,figsize=(12,6));setup(fig,'Recording: known-window coverage and downstream proposal burden')
    for ax,key,title in zip(axes,['matched_known_positive_count','proposal_count'],['Known broad windows reached /76','Frame/location proposals over 11.2 seconds']):
        ax.bar(range(8),[real[a][key] for a in ids],color=[colors[a] for a in ids]);ax.set_xticks(range(8),[labels[a] for a in ids],rotation=60,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylim(0,76);fig.subplots_adjust(bottom=.35,top=.85)
    save(fig,'real_coverage_burden.png','q=1 setup-calibrated operating points. Sparse positive windows do not identify real precision or exhaustive sensitivity.\nUnmatched proposals remain unknown. The same 76 windows and common spatial interior are used by every arm.')

    # Native cutoff plots are case-specific: pooling thresholds with different calibrations is invalid.
    case='isolated_intermediate__seed20260914'
    fig,axes=plt.subplots(2,4,figsize=(14,8));setup(fig,'Native score cutoffs: one fixed intermediate synthetic clip')
    for ax,arm in zip(axes.flat,ids):
        rs=sorted((r for r in rows if r['case_id']==case and r['arm_id']==arm and r['threshold'] is not None),key=lambda r:r['threshold'])
        ax.plot([r['threshold'] for r in rs],[np.nan if r['precision'] is None else r['precision'] for r in rs],marker='o',ms=3,label='Precision')
        ax.plot([r['threshold'] for r in rs],[r['framewise_sensitivity'] for r in rs],marker='o',ms=3,label='Sensitivity')
        ax.set_title(labels[arm]);ax.set_ylim(-.03,1.03);ax.set_xlabel('Native score cutoff');ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.11),ncol=2,frameon=False)
    save(fig,'native_threshold_example.png','Fixed geometry/seed example chosen by protocol role. X/A/C use fluorescence units; Z is dimensionless.\nDifferent axes are intentional. Use the budget curves for the paired 16-clip comparison.')

    paired=[]
    for case_id in sorted({r['case_id'] for r in rows}):
        selected={r['arm_id']:r for r in rows if r['case_id']==case_id and r['threshold_id']=='q1'}
        for mode in ('level','difference'):
            for left,right in (('X','A'),('A','C'),('C','Z')):
                a,b=selected[f'{mode}_{left}'],selected[f'{mode}_{right}']
                paired.append(dict(case_id=case_id,input_mode=mode,comparison=f'{left}_to_{right}',
                    proposal_count_change=b['proposal_count']-a['proposal_count'],
                    true_positive_count_change=None if a.get('true_positive_count') is None else b['true_positive_count']-a['true_positive_count'],
                    false_positive_count_change=None if a.get('false_positive_count') is None else b['false_positive_count']-a['false_positive_count'],
                    known_window_change=None if case_id!='real' else b['matched_known_positive_count']-a['matched_known_positive_count']))
    write_json(out/'paired_component_changes.json',paired);write_tsv(out/'paired_component_changes.tsv',paired)
    lines=['# Gamma-LS necessity pilot','',
        'Eight fixed arms compare activity level and signed change at X (input), A (target), C (local contrast), and Z (full Gamma-LS). All use the same spatial kernel and paired data. These are development results, with no selected test optimum or independent validation.','',
        '## Discovery at the frozen q=1 setup target','',
        '| Arm | Synthetic precision | Active-frame sensitivity | Events reached | Events by 200 ms | False proposals | Real known windows | Real proposals |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ids:
        r=fixed[arm];d=next(v for v in r['deadline_rows'] if v['deadline_ms']==200)
        lines.append(f"| {arm} | {pct(r['precision'])} | {pct(r['sensitivity'])} | {r['recovered_event_count']}/{r['event_count']} | {d['recovered_by_deadline']}/{d['event_count']} | {r['false_positive_count']} | {real[arm]['matched_known_positive_count']}/76 | {real[arm]['proposal_count']} |")
    lines += ['', 'Synthetic precision counts matched frame/location proposals. Active-frame sensitivity covers the whole declared fluorescent support, including tails. Event recovery counts at least one matched proposal and is separate from prompt recovery. All-positive and no-output endpoints remain in the numerical tables. Equal setup budgets do not fix application burden or false-alarm probability.','',
        '## Configured monitoring','',
        '| Arm | Precision | Active ROI/frame sensitivity | Inactive ROI/frame exceedance rate |','|---|---:|---:|---:|']
    for r in mon:lines.append(f"| {r['arm_id']} | {pct(r['precision'])} | {pct(r['sensitivity'])} | {pct(r['inactive_roi_frame_exceedance_rate'])} |")
    lines += ['', 'Monitoring supplies the exact simulated source centers as an oracle configuration and uses separate per-center setup cutoffs with a 1% exceedance budget. It does not use NMS. Its ROI/frame rates are not interchangeable with discovery proposal metrics. Nuisance-only clips provide no configured source opportunities.','',
        '## Figures and evidence','']
    lines += [f'- [{name}]({name})' for name in figures]
    lines += ['', '- [Paired component changes](paired_component_changes.tsv)',
        '- [All case-specific threshold rows](../all_curves.tsv)',
        '- [Frozen protocol](../protocol.json)',
        '- [Prepared raw recording review](../../../GammaLSST/sensitivity_20260914_r2/real_review/raw_media_v1/index.html)','',
        '## Interpretation boundaries','',
        'There are two paired seeds per synthetic template and one real recording. Real labels are sparse broad windows: unmatched proposals remain unknown, and real precision, exhaustive sensitivity and exact onset latency are unavailable. The known-window comparison uses the same 76 eligible annotations; three excluded border windows are geometry exclusions. No learning, temporal-kernel search, universal necessity claim, neural spike-timing claim, or closed-loop control claim follows from this pilot.','',
        'Configured source locations are supplied, not discovered during calibration. The same Gaussian/EMA frontend is held fixed except for level versus signed change. Future work can alter conditioning after this comparison; this experiment does not isolate Gaussian and EMA contributions.','',
        'Numerical evaluation is complete. Scientific media completion is recorded separately in the audit aggregate and final completion manifest; generated figures require visual review before report completion.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    inputs=[root/n for n in ('protocol.json','all_curves.json','monitoring.json','operating_points.json','evaluation_complete.json')]
    manifest=dict(status='GENERATED',figure_count=len(figures),visual_qa_complete=False,
        reporter=binding(Path(__file__)),inputs=[binding(p) for p in inputs],
        artifacts=[dict(binding(p),path=str(p.relative_to(out))) for p in sorted(out.iterdir()) if p.is_file()])
    write_json(out/'manifest.json',manifest)
    print(dict(status='GENERATED',figures=len(figures)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    generate(parser.parse_args().root)
