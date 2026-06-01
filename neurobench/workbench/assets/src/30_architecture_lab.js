function renderArchitectureLab(){
  const root = document.getElementById('architectureRuns');
  if(!root) return;
  const runs = data.architectureRuns?.runs || [];
  populateRunSelectors(runs);
  renderArchitecturePresets();
  renderComponentLibrary();
  renderRunComparison();
  renderPipelineBuilder();
  root.innerHTML = '';
  if(!runs.length){
    root.innerHTML = renderRunSummaryCards(null) + renderArchitectureLibrary() + '<p class="hint">No pipeline runs are attached yet. Use tools/build_architecture_run.py to create architecture_runs.json.</p>';
    for(const btn of root.querySelectorAll('[data-load-template]')) btn.onclick = () => loadTemplateIntoBuilder(btn.dataset.loadTemplate);
    for(const btn of root.querySelectorAll('[data-template-experiment]')) btn.onclick = () => { experimentDraft.baseTemplateId = btn.dataset.templateExperiment; location.hash = '#experiments'; };
    for(const btn of root.querySelectorAll('[data-rename-template]')) btn.onclick = () => renameArchitectureTemplate(btn.dataset.renameTemplate);
    for(const btn of root.querySelectorAll('[data-delete-template]')) btn.onclick = () => deleteArchitectureTemplate(btn.dataset.deleteTemplate);
    return;
  }
  root.innerHTML = renderRunSummaryCards(activeRun() || runs[0]) + renderArchitectureLibrary() + renderParameterExperiments(runs);
  for(const run of runs){
    const card = document.createElement('div');
    const status = run.execution?.status || 'completed';
    card.className = `archCard runStatus-${status}${run.run_id === activeRunId() ? ' activeRunCard' : ''}`;
    const evidence = (run.artifacts?.evidence_maps || []).map(m => `<span>${m.label || m.id || 'map'}</span>`).join('');
    const pipeline = (run.pipeline || []).map(stage => {
      const def = stageDef(stage);
      return `<span title="${escapeHtml(paramSummary(stage).replace(/<[^>]+>/g, ' '))}">${escapeHtml(def?.label || stage.label || stage.name || stageOp(stage) || stage.id || 'stage')}</span>`;
    }).join('');
    const sweep = run.sweep?.parameters ? run.sweep.parameters.map(item => `<span>${escapeHtml((item.stage || item.stage_id) + '.' + item.param)}=${escapeHtml(item.value ?? (item.values || []).join(','))}</span>`).join('') : '';
    const ann = run.annotation_summary || {};
    const provenance = parameterProvenanceSummary(run);
    const readiness = runReadiness(run);
    card.innerHTML = `
      <div class="runCardHeader"><h3>${run.label || run.run_id}</h3><span class="stageStatus ${readiness.className}">${escapeHtml(readiness.label)}</span></div>
      <div class="miniChipRow runReadinessBadges">${runReadinessBadges(run)}</div>
      <p class="hint">${run.run_id} | ${run.dataset_id}${run.sweep?.total_runs ? ` | sweep ${Number(run.sweep.index || 0) + 1}/${run.sweep.total_runs}` : ''}</p>
      <div class="archMeta">
        <div><b>${run.summary?.roi_count ?? 'n/a'}</b><span>ROIs</span></div>
        <div><b>${run.summary?.event_count ?? 'n/a'}</b><span>events</span></div>
        <div><b>${run.summary?.suggestion_count ?? 'n/a'}</b><span>suggestions</span></div>
        <div><b>${run.summary?.frame_count ?? data.video.frames}</b><span>frames</span></div>
        <div><b>${ann.roi_states?.accepted ?? 'n/a'}</b><span>accepted ROIs</span></div>
        <div><b>${ann.control_ready?.yes ?? 'n/a'}</b><span>control-ready</span></div>
      </div>
      <table class="smallTable"><tr><th>Artifact</th><th>Path</th></tr>
        <tr><td>review data</td><td>${run.artifacts?.review_data || ''}</td></tr>
        <tr><td>ROI summary</td><td>${run.artifacts?.roi_summary_tsv || ''}</td></tr>
      </table>
      <div class="archEvidence">${pipeline}</div>
      <div class="archEvidence">${sweep}</div>
      <div class="archEvidence">${evidence}</div>
      <div class="parameterProvenance ${provenance.className}">
        <b>${escapeHtml(provenance.label)}</b>
        <span>${escapeHtml(provenance.detail)}</span>
      </div>
      <div class="buttonRow">
        <button type="button" data-activate-run="${escapeHtml(run.run_id)}">Use In Review/Data</button>
        <button type="button" data-load-review-run="${escapeHtml(run.run_id)}" ${runGenerated(run) ? '' : 'disabled'}>Load Review</button>
      </div>`;
    root.appendChild(card);
  }
  for(const btn of root.querySelectorAll('[data-activate-run]')) btn.onclick = () => selectActiveRun(btn.dataset.activateRun, {loadReview:false});
  for(const btn of root.querySelectorAll('[data-load-review-run]')) btn.onclick = () => selectActiveRun(btn.dataset.loadReviewRun, {loadReview:true});
  for(const btn of root.querySelectorAll('[data-exp-label]')) btn.onclick = () => setExperimentLabel(btn.dataset.runId, btn.dataset.expLabel);
  for(const btn of root.querySelectorAll('[data-load-template]')) btn.onclick = () => loadTemplateIntoBuilder(btn.dataset.loadTemplate);
  for(const btn of root.querySelectorAll('[data-template-experiment]')) btn.onclick = () => { experimentDraft.baseTemplateId = btn.dataset.templateExperiment; location.hash = '#experiments'; };
  for(const btn of root.querySelectorAll('[data-rename-template]')) btn.onclick = () => renameArchitectureTemplate(btn.dataset.renameTemplate);
  for(const btn of root.querySelectorAll('[data-delete-template]')) btn.onclick = () => deleteArchitectureTemplate(btn.dataset.deleteTemplate);
}

function parameterProvenanceSummary(run){
  const params = run?.parameters || {};
  const entries = Object.entries(params);
  const pipeline = run?.pipeline || [];
  const mismatches = [];
  for(const [key, value] of entries){
    const parts = key.split('.');
    if(parts.length < 2) continue;
    const stageKey = parts[0];
    const param = parts.slice(1).join('.');
    const stage = pipeline.find(item => item.id === stageKey || stageOp(item) === stageKey);
    if(!stage) {
      mismatches.push(key);
      continue;
    }
    const stageValue = stage.params?.[param];
    if(stageValue !== undefined && String(stageValue) !== String(value)) mismatches.push(key);
  }
  const pipelineRun = Boolean(run?.artifacts?.pipeline_run || run?.execution?.pipeline_run);
  if(mismatches.length) return {
    className:'warn',
    label:'parameter provenance warning',
    detail:`${mismatches.slice(0, 3).join(', ')} ${mismatches.length > 3 ? `+${mismatches.length - 3} more ` : ''}do not match a visible stage parameter.`
  };
  if(entries.length) return {
    className:'ok',
    label:'parameters aligned',
    detail:`${entries.length} run parameter${entries.length === 1 ? '' : 's'} match visible stage IDs${pipelineRun ? '; CLI pipeline run linked.' : '.'}`
  };
  return {
    className:pipelineRun ? 'ok' : 'off',
    label:pipelineRun ? 'CLI run linked' : 'no run parameter overrides',
    detail:pipelineRun ? 'Pipeline execution metadata is attached.' : 'No sweep/CLI parameter override metadata is attached for this run.'
  };
}

function runLabel(run){
  if(!run) return '';
  if(isGammaCfarRun(run)) {
    const params = run.parameters || {};
    const summary = run.summary || {};
    const sweep = String(run.run_id).split('__').pop()?.replace('sweep_', '') || '';
    const pfa = params['cfar_small_ref.pfa'] ?? summary['cfar_small_ref.pfa'];
    const ref = params['cfar_large_ref.training_radius_px'] ?? summary['cfar_large_ref.training_radius_px'];
    const support = params['components.support_min_frames'] ?? summary['components.support_min_frames'] ?? summary.component_support_min_frames;
    const rois = summary.roi_count ?? 'n/a';
    return `Gamma CFAR ${sweep} · ${rois} ROIs · pfa ${pfa ?? 'n/a'} · ref ${ref ?? 'n/a'} · support ${support ?? 'n/a'}`;
  }
  return `${run.label || run.run_id}`;
}

function runMetric(run, path, fallback=0){
  let cur = run;
  for(const part of path.split('.')){
    if(cur === undefined || cur === null) return fallback;
    cur = cur[part];
  }
  return cur === undefined || cur === null ? fallback : cur;
}

function experimentLabels(){
  annotations.settings.experimentLabels = annotations.settings.experimentLabels || {};
  return annotations.settings.experimentLabels;
}

function experimentLabel(runId){
  return experimentLabels()[runId] || '';
}

function setExperimentLabel(runId, label){
  experimentLabels()[runId] = label;
  queueSave();
  renderArchitectureLab();
  renderExperimentLab();
}

function experimentNotes(){
  annotations.settings.experimentNotes = annotations.settings.experimentNotes || {};
  return annotations.settings.experimentNotes;
}

function experimentNote(runId){
  return experimentNotes()[runId] || '';
}

function setExperimentNote(runId, note){
  if(!runId) return;
  const notes = experimentNotes();
  if(String(note || '').trim()) notes[runId] = String(note || '').trim();
  else delete notes[runId];
  queueSave();
}

function experimentLabelChoices(){
  return ['shortlist','looks best','baseline candidate','too noisy','too strict','artifact heavy','needs review'];
}

function experimentProposalStates(){
  annotations.settings.experimentProposalStates = annotations.settings.experimentProposalStates || {};
  return annotations.settings.experimentProposalStates;
}

function proposalLifecycleKey(proposalSetId, proposalId){
  return `${proposalSetId || 'pasted'}::${proposalId || 'proposal'}`;
}

function experimentProposalState(proposalSetId, proposalId){
  return experimentProposalStates()[proposalLifecycleKey(proposalSetId, proposalId)]?.state || '';
}

function setExperimentProposalState(proposalSetId, proposalId, state){
  const key = proposalLifecycleKey(proposalSetId, proposalId);
  if(!state) delete experimentProposalStates()[key];
  else experimentProposalStates()[key] = {state, updatedAt:new Date().toISOString()};
  queueSave();
  renderExperimentLab();
}

function experimentProposalStateChoices(){
  return ['try next','imported','generated','promising','reject','needs repair','discussed'];
}

function shortlistedRuns(){
  const useful = new Set(['shortlist', 'looks best', 'baseline candidate']);
  return architectureRuns().filter(run => useful.has(experimentLabel(run.run_id)));
}

function runExperimentMetrics(run){
  const summary = run?.summary || {};
  const ann = run?.annotation_summary || {};
  const artifactLike = runMetric(run, 'annotation_summary.triage_queue_counts.artifact_like', null);
  const missed = runMetric(run, 'annotation_summary.triage_queue_counts.possible_missed_neuron', null);
  const medianArea = runMetric(run, 'qc.roiAreaStats.median', null);
  return {
    rois: summary.roi_count ?? 'n/a',
    events: summary.event_count ?? 'n/a',
    suggestions: summary.suggestion_count ?? 'n/a',
    accepted: ann.roi_states?.accepted ?? 'n/a',
    artifacts: artifactLike ?? 'n/a',
    missed: missed ?? 'n/a',
    median_area: medianArea ?? 'n/a',
    status: run?.execution?.status || (runGenerated(run) ? 'completed' : 'planned')
  };
}

function runSummaryCardData(run=activeRun()){
  const s = annotationSummary();
  const summary = run?.summary || {};
  const ann = run?.annotation_summary || {};
  const rois = summary.roi_count ?? s.roi_count ?? data.rois.length;
  const events = summary.event_count ?? s.event_count ?? (data.events || []).length;
  const suggestions = summary.suggestion_count ?? s.suggestion_count ?? (data.discovery?.suggestions || []).length;
  const accepted = ann.roi_states?.accepted ?? s.roi_states.accepted ?? 0;
  const area = runMetric(run, 'qc.roiAreaStats.median', null);
  const pixelSize = Number(data.dataset?.pixel_size_microns);
  const diamPx = Number.isFinite(area) && area > 0 ? 2 * Math.sqrt(Number(area) / Math.PI) : null;
  const diamText = Number.isFinite(diamPx)
    ? (Number.isFinite(pixelSize) ? `${fmt(diamPx * pixelSize, 1)} um` : `${fmt(diamPx, 1)} px`)
    : 'n/a';
  return [
    {label:'active run', value:run ? runLabel(run) : 'none'},
    {label:'status', value:runStatusLabel(run)},
    {label:'candidate ROIs', value:rois},
    {label:'candidate events', value:events},
    {label:'suggestions', value:suggestions},
    {label:'accepted ROIs', value:accepted},
    {label:'median diameter', value:diamText},
    {label:'review ready', value:s.review_progress.tuning_ready ? 'yes' : 'not yet'}
  ];
}

function renderRunSummaryCards(run=activeRun()){
  const cards = runSummaryCardData(run).map(item => `
    <div class="summaryCard">
      <b>${escapeHtml(item.value)}</b>
      <span>${escapeHtml(item.label)}</span>
    </div>`).join('');
  return `<section class="sharedRunSummary">${cards}</section>`;
}

function nextBestAction(){
  const s = annotationSummary();
  const run = activeRun() || architectureRuns()[0] || null;
  if(run && runHasIntermediates(run) && !runGenerated(run)) {
    return {
      eyebrow:'Next best action',
      title:'Inspect generated stage outputs',
      detail:'This run has intermediate videos ready. Check raw and processed stages together before deciding whether it should become a Review dataset.',
      href:'#data',
      action:'Open Data'
    };
  }
  if(!s.review_progress.tuning_ready) {
    const roiNeed = Math.max(0, s.review_progress.tuning_ready_targets.reviewed_rois - s.review_progress.reviewed_rois);
    const eventNeed = Math.max(0, s.review_progress.tuning_ready_targets.reviewed_events - s.review_progress.reviewed_events);
    return {
      eyebrow:'Next best action',
      title:'Finish the first review seed',
      detail:`Review ${roiNeed} more ROI label${roiNeed === 1 ? '' : 's'} and ${eventNeed} more event label${eventNeed === 1 ? '' : 's'} before treating parameter comparisons as evidence.`,
      href:'#review',
      action:'Open Review'
    };
  }
  if(run && runGenerated(run)) {
    return {
      eyebrow:'Next best action',
      title:'Compare and document the active run',
      detail:'The selected run has review data attached. Use Progress and Report to capture what is ready, uncertain, and worth testing next.',
      href:'#progress',
      action:'Open Progress'
    };
  }
  return {
    eyebrow:'Next best action',
    title:'Choose or generate a pipeline run',
    detail:'Start from Pipelines if you want to adjust the stack, or Experiment Lab if you want to plan a parameter search.',
    href:'#pipelines',
    action:'Open Pipelines'
  };
}

function renderNextBestActionPanel(){
  const action = nextBestAction();
  return `
    <section class="nextBestActionCard">
      <div>
        <span>${escapeHtml(action.eyebrow)}</span>
        <h2>${escapeHtml(action.title)}</h2>
        <p>${escapeHtml(action.detail)}</p>
      </div>
      <a class="textButton" href="${escapeHtml(action.href)}">${escapeHtml(action.action)}</a>
    </section>`;
}

function renderNextBestActions(){
  for(const root of document.querySelectorAll('[data-next-action]')) {
    root.innerHTML = renderNextBestActionPanel();
  }
}

function renderParameterExperiments(runs){
  const rows = runs.map(run => {
    const m = runExperimentMetrics(run);
    const label = experimentLabel(run.run_id);
    const labelButtons = experimentLabelChoices().map(value =>
      `<button type="button" class="${label === value ? 'active' : ''}" data-run-id="${escapeHtml(run.run_id)}" data-exp-label="${escapeHtml(value)}">${escapeHtml(value)}</button>`
    ).join('');
    return `
      <tr class="${run.run_id === activeRunId() ? 'activeRunRow' : ''}">
        <td><b>${escapeHtml(runLabel(run))}</b><br><span class="hint">${escapeHtml(run.run_id)}</span></td>
        <td>${escapeHtml(m.status)}</td>
        <td>${escapeHtml(m.rois)}</td>
        <td>${escapeHtml(m.events)}</td>
        <td>${escapeHtml(m.suggestions)}</td>
        <td>${escapeHtml(m.accepted)}</td>
        <td>${escapeHtml(m.artifacts)}</td>
        <td>${escapeHtml(m.missed)}</td>
        <td>${escapeHtml(m.median_area)}</td>
        <td><div class="buttonRow">${labelButtons}</div></td>
        <td><button type="button" data-activate-run="${escapeHtml(run.run_id)}">Use</button> <a href="#data">Data</a></td>
      </tr>`;
  }).join('');
  return `
    <section class="archCard experimentBoard">
      <div class="runCardHeader">
        <h3>Parameter Experiments</h3>
        <span class="runStatus">${runs.length} run${runs.length === 1 ? '' : 's'}</span>
      </div>
      <p class="hint">Label sweep outputs and compare review burden before committing to a detector setting.</p>
      <table class="smallTable compareTable">
        <tr><th>Run</th><th>Status</th><th>ROIs</th><th>Events</th><th>Suggestions</th><th>Accepted</th><th>Artifact-like</th><th>Missed candidates</th><th>Median area</th><th>Label</th><th>Open</th></tr>
        ${rows}
      </table>
    </section>`;
}

function statusClass(status){
  const value = String(status || '').toLowerCase();
  if(value === 'completed' || value === 'valid' || value === 'ready') return 'ok';
  if(value === 'failed' || value === 'invalid') return 'bad';
  if(value === 'planned' || value === 'running' || value === 'exported') return 'warn';
  return 'off';
}

function completedArchitectureRuns(){
  return architectureRuns().filter(run => runGenerated(run) || String(run?.execution?.status || '').toLowerCase() === 'completed');
}

function experimentBaselineRun(){
  const runs = completedArchitectureRuns();
  return runs.find(run => run.run_id === 'current_review_pipeline') || runs[0] || architectureRuns()[0] || null;
}

function numericRunMetric(run, path, fallback=0){
  const value = Number(runMetric(run, path, fallback));
  return Number.isFinite(value) ? value : fallback;
}

function sweepCombinationCountForRun(run){
  const axes = sweepFactors(run);
  if(!axes.length) return 1;
  return axes.reduce((total, axis) => total * Math.max(1, Array.isArray(axis.values) ? axis.values.length : 0), 1);
}

function sweepBudgetStatus(count){
  if(count > 4096) return {className:'bad', label:'too large', text:'This is likely too expensive for a first pass. Narrow axes or use Optuna seeds.'};
  if(count > 512) return {className:'warn', label:'large', text:'Consider previewing a smaller seed set before generating the full sweep.'};
  if(count > 64) return {className:'warn', label:'moderate', text:'Good for batch testing, but label a preview before committing.'};
  return {className:'ok', label:'focused', text:'Small enough for readable comparison.'};
}

function changedParametersForRun(run){
  if(run?.sweep?.parameters?.length) return run.sweep.parameters.map(p => `${p.stage || p.stage_id}.${p.param}=${p.value ?? (p.values || []).join(',')}`);
  if(run?.experiment?.override) {
    const o = run.experiment.override;
    return [`${o.stage || o.stage_id}.${o.param}=${o.value ?? ''}`];
  }
  return [];
}

function pipelineChangeSummary(run){
  const changed = changedParametersForRun(run);
  if(changed.length) return changed.slice(0, 4).join(', ') + (changed.length > 4 ? `, +${changed.length - 4} more` : '');
  const ops = (run?.pipeline || []).map(stage => stageDef(stage)?.label || stageOp(stage) || stage.id).filter(Boolean);
  return ops.slice(0, 5).join(' -> ') + (ops.length > 5 ? ` -> +${ops.length - 5} more` : '');
}

function runReadiness(run){
  if(!run) return {className:'off', label:'no run', text:'No architecture run is selected.'};
  const validation = validatePipeline(normalizePipelineDraft(JSON.parse(JSON.stringify(run))));
  if(validation.errors.length) return {className:'bad', label:'invalid', text:validation.errors[0]};
  if(!runGenerated(run)) return {className:'warn', label:'planned', text:'Generate a preview before using this run for Review or Data.'};
  if(!run.annotation_summary) return {className:'warn', label:'needs labels', text:'Generated outputs exist, but annotation summaries are not attached yet.'};
  return {className:'ok', label:'reviewable', text:'Generated artifacts and annotation summary are available.'};
}

function runReadinessBadges(run){
  const pipeline = normalizedRunPipeline(run);
  const local = pipeline.length && pipeline.every(stage => {
    const def = stageDef(stage);
    return def?.runner_available || def?.locally_runnable || def?.availability === 'implemented';
  });
  const hasReviewData = runGenerated(run);
  const hasRois = runHasCandidateRois(run);
  const hasIntermediates = runHasIntermediates(run);
  const hasEvidence = availableEvidenceMapsForRun(run).length > 0 || (data.discovery?.evidenceMaps || []).length > 0;
  const stencil = savedStencilPoints().length >= 3;
  const rois = runReviewRoisFromCache(run);
  const metrics = stencil && rois.length ? stencilMetricsForRois(rois) : null;
  const stencilText = !stencil ? 'no stencil' : metrics ? `${metrics.inside + metrics.edge}/${metrics.total} stencil` : 'stencil ready';
  const items = [
    {ok:local, label:local ? 'local executable' : 'not fully local'},
    {ok:hasReviewData, label:hasReviewData ? 'review data' : 'no review data'},
    {ok:hasRois, label:hasRois ? 'ROI candidates' : 'no ROI candidates'},
    {ok:hasIntermediates, label:hasIntermediates ? 'intermediates' : 'no intermediates'},
    {ok:hasEvidence, label:hasEvidence ? 'projection diagnostics' : 'no projections'},
    {ok:stencil && (!metrics || metrics.inside + metrics.edge > 0), label:stencilText}
  ];
  return items.map(item => `<span class="stageStatus ${item.ok ? 'ok' : 'warn'}">${escapeHtml(item.label)}</span>`).join('');
}

function runUtilityScore(run){
  const readiness = runReadiness(run);
  const ann = run?.annotation_summary;
  const label = experimentLabel(run?.run_id || '');
  const generated = runGenerated(run);
  if(!run) return {score:null, status:'missing', className:'off', components:[], warnings:['No run selected.']};
  const accepted = numericRunMetric(run, 'annotation_summary.roi_states.accepted', 0);
  const controlReady = numericRunMetric(run, 'annotation_summary.control_ready.yes', 0);
  const artifacts = numericRunMetric(run, 'annotation_summary.triage_queue_counts.artifact_like', 0);
  const missed = numericRunMetric(run, 'annotation_summary.triage_queue_counts.possible_missed_neuron', 0);
  const rejected = numericRunMetric(run, 'annotation_summary.roi_states.rejected', 0);
  const suggestions = numericRunMetric(run, 'summary.suggestion_count', 0);
  const rois = numericRunMetric(run, 'summary.roi_count', 0);
  const warnings = [];
  if(!generated) warnings.push('Generate this run before scoring it.');
  if(!ann) warnings.push('Add or attach annotation summaries for a meaningful score.');
  if(label === 'too noisy' || label === 'artifact heavy') warnings.push(`Human label: ${label}.`);
  if(suggestions > Math.max(12, rois * 0.4)) warnings.push('Many uncovered suggestions remain; inspect recall before tightening.');
  const rawScore = 45 + accepted * 1.8 + controlReady * 3.2 - artifacts * 1.4 - missed * 1.2 - rejected * 0.35;
  const score = generated && ann ? Math.max(0, Math.min(100, Math.round(rawScore))) : null;
  return {
    score,
    status: score === null ? readiness.label : score >= 72 ? 'promising' : score >= 50 ? 'usable' : 'needs review',
    className: score === null ? readiness.className : score >= 72 ? 'ok' : score >= 50 ? 'warn' : 'bad',
    components: [
      {label:'accepted ROIs', value:accepted, detail:'more is usually better after artifact review'},
      {label:'control-ready', value:controlReady, detail:'best proxy for downstream inverse-dynamics usefulness'},
      {label:'artifact-like', value:artifacts, detail:'review burden and false-positive risk'},
      {label:'missed candidates', value:missed, detail:'recall risk from suggestions/labels'}
    ],
    warnings
  };
}

function runDeltaSummary(run, baseline=experimentBaselineRun()){
  const rows = [
    ['ROIs', 'summary.roi_count', false],
    ['events', 'summary.event_count', false],
    ['suggestions', 'summary.suggestion_count', false],
    ['accepted', 'annotation_summary.roi_states.accepted', true],
    ['control-ready', 'annotation_summary.control_ready.yes', true],
    ['artifact-like', 'annotation_summary.triage_queue_counts.artifact_like', false],
    ['missed', 'annotation_summary.triage_queue_counts.possible_missed_neuron', false]
  ];
  if(!run || !baseline || run.run_id === baseline.run_id) return [];
  return rows.map(([label, path, higherGood]) => {
    const value = numericRunMetric(run, path, 0);
    const base = numericRunMetric(baseline, path, 0);
    const delta = value - base;
    const className = Math.abs(delta) < 1e-9 ? 'deltaNeutral' : (higherGood ? delta > 0 : delta < 0) ? 'deltaGood' : 'deltaBad';
    return {label, value, base, delta, className};
  });
}

function renderUtilityScorecard(run){
  const score = runUtilityScore(run);
  const warnings = score.warnings.map(w => `<div class="pipelineWarning">${escapeHtml(w)}</div>`).join('');
  return `
    <article class="experimentScorecard">
      <div class="runCardHeader">
        <h3>${escapeHtml(runLabel(run) || 'No run')}</h3>
        <span class="stageStatus ${score.className}">${escapeHtml(score.status)}</span>
      </div>
      <div class="utilityScore ${score.className}">${score.score === null ? 'n/a' : score.score}<span>utility</span></div>
      <div class="scoreComponents">
        ${score.components.map(item => `<div><b>${escapeHtml(item.value)}</b><span title="${escapeHtml(item.detail)}">${escapeHtml(item.label)}</span></div>`).join('')}
      </div>
      ${warnings}
    </article>`;
}

function renderRunDifferenceCard(run, baseline=experimentBaselineRun()){
  const deltas = runDeltaSummary(run, baseline);
  if(!deltas.length) return `
    <article class="experimentScorecard muted">
      <div class="runCardHeader"><h3>Compare Against Baseline</h3><span class="stageStatus off">baseline</span></div>
      <p class="hint">This run is the current baseline or no completed baseline is available yet.</p>
    </article>`;
  return `
    <article class="experimentScorecard">
      <div class="runCardHeader">
        <h3>Compare Against Baseline</h3>
        <span class="runStatus">${escapeHtml(runLabel(baseline))}</span>
      </div>
      <div class="deltaGrid">
        ${deltas.map(row => `<div><span>${escapeHtml(row.label)}</span><b class="${row.className}">${row.delta >= 0 ? '+' : ''}${fmt(row.delta, Number.isInteger(row.delta) ? 0 : 1)}</b></div>`).join('')}
      </div>
    </article>`;
}

function llmProposalSets(){
  const sets = Array.isArray(data.architectureRuns?.llm_proposal_sets) ? data.architectureRuns.llm_proposal_sets : [];
  const byId = new Map(sets.map(set => [set.id || set.proposal_set_id || set.label, Object.assign({templates:[], runs:[]}, set)]));
  for(const template of savedPipelineTemplates().filter(item => item.source === 'llm_architecture_proposal')) {
    const id = template.proposal_set_id || template.llm_proposal_set_id || 'imported_llm_proposals';
    if(!byId.has(id)) byId.set(id, {id, label:'Imported LLM proposals', templates:[], runs:[]});
    byId.get(id).templates.push(template);
  }
  for(const run of architectureRuns().filter(item => item.artifacts?.proposal_set_id || item.experiment?.proposal_set_id)) {
    const id = run.artifacts?.proposal_set_id || run.experiment?.proposal_set_id;
    if(!byId.has(id)) byId.set(id, {id, label:id, templates:[], runs:[]});
    byId.get(id).runs.push(run);
  }
  return [...byId.values()];
}

function recommendationsFromAnnotations(){
  const active = activeRun() || experimentBaselineRun();
  const s = annotationSummary();
  const label = experimentLabel(active?.run_id || '');
  const recs = [];
  if(!s.review_progress.tuning_ready) recs.push({
    title:'Finish a small review seed before optimizing',
    text:`Need ${Math.max(0, s.review_progress.tuning_ready_targets.reviewed_rois - s.review_progress.reviewed_rois)} more ROI labels and ${Math.max(0, s.review_progress.tuning_ready_targets.reviewed_events - s.review_progress.reviewed_events)} more event labels for a useful tuning signal.`,
    action:'Open Review', href:'#review', status:'review first'
  });
  if(s.triage_queue_counts.artifact_like > 0 || label === 'artifact heavy' || label === 'too noisy') recs.push({
    title:'Try an artifact-suppression architecture',
    text:'Artifact-like labels or noisy-run labels suggest adding despiking, heterogeneity cues, and stricter component filtering before expanding sweeps.',
    preset:'artifact_suppression', status:'noise'
  });
  if(s.triage_queue_counts.possible_missed_neuron > 0 || s.suggestion_states.missed > 0 || s.suggestion_states.unlabeled > 10) recs.push({
    title:'Run a high-recall discovery pass',
    text:'Missed-neuron or unlabeled suggestion burden means recall should be tested before selecting strict thresholds.',
    preset:'high_recall_discovery', status:'recall'
  });
  if(!architectureRuns().some(runGenerated)) recs.push({
    title:'Generate a first preview',
    text:'The dashboard can plan many runs, but only generated previews can be inspected in Review and Data.',
    action:'Generate Preview', generate:true, status:'preview'
  });
  if(!recs.length) recs.push({
    title:'Compare the best two generated runs',
    text:'Use A/B Review and utility deltas to choose one candidate architecture before widening the search.',
    action:'Open Pipelines', href:'#pipelines', status:'compare'
  });
  return recs.slice(0, 4);
}

function experimentObjective(){
  return annotations.settings.experimentObjective || 'Improve recall while suppressing impulse noise and artifact-like clusters.';
}

function setExperimentObjective(value){
  annotations.settings.experimentObjective = String(value || '').trim();
  queueSave();
  renderExperimentLab();
}

function experimentSessionRecipe(manifest=experimentManifest(), run=activeRun() || experimentBaselineRun()){
  const actions = experimentActionQueue(manifest, run, 'open');
  const followUps = experimentFollowUpSuggestions().slice(0, 4);
  const coverage = experimentCoverageRows().filter(row => row.status.label !== 'covered').slice(0, 4);
  const recommendations = recommendationsFromAnnotations();
  const budget = Math.max(sweepCombinationCountForRun(plannedRun()), manifest.runs.length);
  return {
    objective: experimentObjective(),
    focus_mode: experimentPanelMode(),
    active_run_id: run?.run_id || '',
    active_run_label: run ? runLabel(run) : '',
    next_action: actions[0] ? {
      id: actions[0].id,
      title: actions[0].title,
      source: actions[0].source,
      priority: actions[0].priority,
      detail: actions[0].detail,
      action: actions[0].actionLabel
    } : null,
    run_recipe: manifest.runs.slice(0, 8).map(item => ({
      run_id: item.run_id,
      label: item.label || item.run_id,
      changed_parameters: changedParametersForRun(item),
      readiness: runReadiness(item).label
    })),
    recommended_architectures: recommendations.map(rec => ({
      title: rec.title,
      status: rec.status,
      preset: rec.preset || '',
      reason: rec.text
    })),
    parameter_moves: [
      ...followUps.map(item => ({
        source: 'follow_up',
        parameter: item.key,
        values: item.values,
        reason: item.reason
      })),
      ...coverage.map(row => ({
        source: 'coverage_gap',
        parameter: row.key,
        values: row.probes,
        reason: row.status.text
      }))
    ].slice(0, 8),
    safeguards: experimentReadinessItems(manifest, run).filter(item => !item.ok).map(item => ({
      title: item.title,
      detail: item.text
    })),
    sweep_budget: budget,
    validation: validatePipeline(pipelineDraft).status
  };
}

function experimentSessionRecipeMarkdown(recipe=experimentSessionRecipe()){
  const lines = [
    `# Neurobench Session Recipe: ${datasetId}`,
    '',
    `Objective: ${recipe.objective || 'not specified'}`,
    `Focus mode: ${recipe.focus_mode}`,
    `Active run: ${recipe.active_run_label || recipe.active_run_id || 'none'}`,
    `Sweep budget: ${recipe.sweep_budget}`,
    `Validation: ${recipe.validation}`,
    '',
    '## Next Action',
    ''
  ];
  if(recipe.next_action) {
    lines.push(`- ${recipe.next_action.title} (${recipe.next_action.source}, priority ${recipe.next_action.priority})`);
    lines.push(`- Why: ${recipe.next_action.detail}`);
    lines.push(`- Dashboard action: ${recipe.next_action.action}`);
  } else {
    lines.push('- No open blocking action.');
  }
  lines.push('', '## Runs To Try', '');
  if(recipe.run_recipe.length) {
    for(const run of recipe.run_recipe) {
      lines.push(`- ${run.label} (${run.run_id}): ${run.changed_parameters.length ? run.changed_parameters.join(', ') : 'base stack'}; ${run.readiness}`);
    }
  } else {
    lines.push('- No planned runs yet.');
  }
  lines.push('', '## Parameter Moves', '');
  if(recipe.parameter_moves.length) {
    for(const move of recipe.parameter_moves) lines.push(`- ${move.parameter}: try ${move.values.join(', ') || 'n/a'} (${move.source}). ${move.reason}`);
  } else {
    lines.push('- No parameter moves are suggested yet.');
  }
  lines.push('', '## Architecture Prompts', '');
  if(recipe.recommended_architectures.length) {
    for(const rec of recipe.recommended_architectures) lines.push(`- ${rec.title}${rec.preset ? ` [${rec.preset}]` : ''}: ${rec.reason}`);
  } else {
    lines.push('- No architecture recommendation is active.');
  }
  lines.push('', '## Safeguards', '');
  if(recipe.safeguards.length) {
    for(const item of recipe.safeguards) lines.push(`- ${item.title}: ${item.detail}`);
  } else {
    lines.push('- All checklist safeguards are currently ready.');
  }
  return lines.join('\n') + '\n';
}

function experimentLlmPromptMode(){
  return annotations.settings.experimentLlmPromptMode || 'architecture_feedback';
}

function setExperimentLlmPromptMode(value){
  annotations.settings.experimentLlmPromptMode = value || 'architecture_feedback';
  queueSave();
  renderExperimentLab();
}

function experimentLlmConstraints(){
  return annotations.settings.experimentLlmConstraints || 'Prefer high recall with explicit artifact controls. Keep proposals inspectable, CPU-first when possible, and compatible with local generation. Do not treat candidate ROIs or event markers as ground truth.';
}

function setExperimentLlmConstraints(value){
  annotations.settings.experimentLlmConstraints = String(value || '').trim();
  queueSave();
  renderExperimentLab();
}

function experimentLlmPromptSpec(manifest=experimentManifest(), run=activeRun() || experimentBaselineRun()){
  const recipe = experimentSessionRecipe(manifest, run);
  const mode = experimentLlmPromptMode();
  const modeText = {
    architecture_feedback: 'Propose 3-5 architecture variants that directly address the objective.',
    parameter_search: 'Propose a compact parameter-search plan using named sets or small sweeps.',
    noise_artifact: 'Focus on impulse-noise attenuation, artifact rejection, and preserving sparse neural events.',
    realtime_100hz: 'Focus on stages that can plausibly operate on 100 Hz video or have a clear offline-only role.'
  }[mode] || 'Propose 3-5 architecture variants that directly address the objective.';
  const currentStack = (pipelineDraft.pipeline || []).map(stage => ({
    id: stage.id,
    stage_id: stageOp(stage),
    label: stageDef(stage)?.label || stage.label || stage.id,
    params: stage.params || {}
  }));
  const availableStages = STAGE_CATALOG.slice(0, 24).map(stage => ({
    stage_id: stage.op,
    label: stage.label,
    description: stage.description || stage.why_use_it || '',
    real_time_mode: stage.real_time_profile?.mode || 'unknown',
    params: Object.keys(stage.params || {})
  }));
  return {
    dataset_id: datasetId,
    mode,
    mode_instruction: modeText,
    objective: recipe.objective,
    constraints: experimentLlmConstraints(),
    active_run_id: recipe.active_run_id,
    current_stack: currentStack,
    session_recipe: recipe,
    available_stage_catalog: availableStages,
    response_schema: {
      schema_version: 1,
      proposal_set_id: `${slugify(datasetId)}_llm_proposals`,
      dataset_id: datasetId,
      objective: recipe.objective,
      proposals: [{
        id: 'short_unique_id',
        label: 'Human readable architecture name',
        rationale: 'Why this should help',
        hypothesis: 'What should improve and what may get worse',
        priority: 1,
        expected_tradeoffs: 'Review burden, runtime, recall/noise tradeoffs',
        pipeline: [{id: 'stage_step_id', stage_id: 'known_stage_id', params: {}}],
        sweep: {parameters: [{stage: 'stage_step_id', stage_id: 'known_stage_id', param: 'param_name', values: []}]}
      }]
    }
  };
}

function experimentLlmPromptMarkdown(spec=experimentLlmPromptSpec()){
  const lines = [
    `# LLM Architecture Request: ${spec.dataset_id}`,
    '',
    'You are helping design inspectable calcium-imaging detection architectures for the Neurobench workbench.',
    'Return only JSON matching `schemas/llm_architecture_proposal.schema.json`.',
    '',
    '## Task',
    '',
    spec.mode_instruction,
    '',
    '## Objective',
    '',
    spec.objective || 'Not specified.',
    '',
    '## Constraints',
    '',
    spec.constraints || 'No extra constraints.',
    '',
    '## Current Stack',
    ''
  ];
  if(spec.current_stack.length) {
    for(const stage of spec.current_stack) lines.push(`- ${stage.id}: ${stage.label} (${stage.stage_id}) params=${JSON.stringify(stage.params || {})}`);
  } else {
    lines.push('- No current stack is available.');
  }
  lines.push('', '## Session Recipe', '', experimentSessionRecipeMarkdown(spec.session_recipe).trim(), '', '## Available Stage IDs', '');
  for(const stage of spec.available_stage_catalog) {
    lines.push(`- ${stage.stage_id}: ${stage.label}; realtime=${stage.real_time_mode}; params=${stage.params.join(', ') || 'none'}`);
  }
  lines.push(
    '',
    '## Response Requirements',
    '',
    '- Use concrete stage step IDs in every sweep axis.',
    '- Keep each proposal small enough for local preview generation.',
    '- Explain expected tradeoffs, especially false positives, missed neurons, runtime, and review burden.',
    '- Include at least one conservative baseline-style proposal and at least one high-recall proposal.',
    '- If suggesting multi-stage CFAR, include separate step IDs for each CFAR stage.',
    '',
    '## JSON Shape',
    '',
    '```json',
    JSON.stringify(spec.response_schema, null, 2),
    '```'
  );
  return lines.join('\n') + '\n';
}

function experimentLlmResponseText(){
  return annotations.settings.experimentLlmResponseText || '';
}

function setExperimentLlmResponseText(value){
  annotations.settings.experimentLlmResponseText = String(value || '');
  queueSave();
  renderExperimentLab();
}

function experimentLlmImportCommand(proposalName='llm_proposals.json'){
  const proposalPath = `Outputs/ArchitectureRuns/${datasetId}/${proposalName}`;
  const architecturePath = `Outputs/NeuronReview/${datasetId}/app/architecture_runs.json`;
  const reportStem = String(proposalName || 'llm_proposals.json').replace(/\.json$/i, '');
  const reportPath = `Outputs/ArchitectureRuns/${datasetId}/${reportStem}_validation_report.json`;
  return `python3 tools/import_llm_architecture_proposals.py --proposal ${proposalPath} --architecture-runs ${architecturePath} --out ${architecturePath} --validation-report ${reportPath}`;
}

function experimentLlmServerImportEndpoint(){
  return apiUrl('llm-proposals/import');
}

function experimentLlmImportCommands(){
  return {
    full: experimentLlmImportCommand('llm_proposals.json'),
    candidate: experimentLlmImportCommand(`${datasetId}_llm_candidate_proposals.json`),
    serverEndpoint: experimentLlmServerImportEndpoint()
  };
}

async function importLlmProposalSetViaServer({candidateOnly=false}={}){
  const intake = experimentLlmProposalIntake();
  if(candidateOnly && !intake.candidate_count) {
    setSaveState('no candidate proposals to import', 'bad');
    return;
  }
  if(!candidateOnly && (!intake.parsed || intake.errors.length)) {
    setSaveState('fix proposal errors before full import', 'bad');
    return;
  }
  const proposal = candidateOnly ? experimentLlmCandidateProposalPack(intake) : intake.parsed;
  if(!proposal?.proposals?.length) {
    setSaveState('proposal pack is empty', 'bad');
    return;
  }
  setSaveState(`importing ${proposal.proposals.length} LLM proposal${proposal.proposals.length === 1 ? '' : 's'}...`, '');
  try {
    const result = await fetchJson(experimentLlmServerImportEndpoint(), {
      method: 'POST',
      headers: generationHeaders(),
      body: JSON.stringify({proposal})
    });
    annotations.settings.lastLlmImportResult = {
      importedAt: new Date().toISOString(),
      proposal_set_id: result.proposal_set_id,
      run_count: result.run_ids?.length || 0,
      saved_pipeline_count: result.saved_pipeline_ids?.length || 0,
      candidateOnly
    };
    queueSave();
    await refreshArchitectureRuns();
    renderExperimentLab();
    setSaveState(`imported ${result.run_ids?.length || 0} planned run${(result.run_ids?.length || 0) === 1 ? '' : 's'} from LLM proposals`, 'ok');
  } catch (err) {
    setSaveState(err.message || 'LLM proposal import failed', 'bad');
  }
}

function experimentLlmProposalIntake(text=experimentLlmResponseText()){
  const raw = String(text || '').trim();
  const result = {
    status: 'empty',
    className: 'off',
    label: 'empty',
    errors: [],
    warnings: [],
    proposal_set_id: '',
    objective: '',
    proposal_count: 0,
    candidate_count: 0,
    total_combinations: 0,
    proposals: [],
    parsed: null,
    import_command: experimentLlmImportCommand(),
    import_commands: experimentLlmImportCommands()
  };
  if(!raw) {
    result.warnings.push('Paste proposal JSON returned by an external LLM to validate it before import.');
    return result;
  }
  try {
    result.parsed = JSON.parse(raw);
  } catch(err) {
    result.status = 'invalid';
    result.className = 'bad';
    result.label = 'invalid JSON';
    result.errors.push(`JSON parse error: ${err.message}`);
    return result;
  }
  const payload = result.parsed || {};
  const knownStages = new Set(STAGE_CATALOG.map(stage => stage.op));
  result.proposal_set_id = payload.proposal_set_id || '';
  result.objective = payload.objective || '';
  const proposals = Array.isArray(payload.proposals) ? payload.proposals : [];
  result.proposal_count = proposals.length;
  if(payload.schema_version !== 1) result.errors.push('schema_version must be 1.');
  if(!payload.proposal_set_id) result.errors.push('proposal_set_id is required.');
  if(payload.dataset_id && payload.dataset_id !== datasetId) result.warnings.push(`dataset_id is ${payload.dataset_id}, but this dashboard is ${datasetId}.`);
  if(!payload.dataset_id) result.errors.push('dataset_id is required.');
  if(!payload.objective) result.errors.push('objective is required.');
  if(!proposals.length) result.errors.push('proposals must contain at least one proposal.');
  for(const [index, proposal] of proposals.entries()) {
    const prefix = proposal?.id || `proposal_${index + 1}`;
    const pipeline = Array.isArray(proposal?.pipeline) ? proposal.pipeline : [];
    const stepIds = new Set();
    const duplicateSteps = [];
    const unknownStages = [];
    const missingFields = [];
    const proposalIssues = [];
    const stageLabels = [];
    for(const field of ['id', 'label', 'rationale', 'hypothesis']) {
      if(!proposal?.[field]) missingFields.push(field);
    }
    if(!pipeline.length) missingFields.push('pipeline');
    for(const step of pipeline) {
      const stepId = step?.id || '';
      const stageId = step?.stage_id || step?.stage || '';
      const def = stageDef(stageId);
      if(stageId) stageLabels.push(def?.label || stageId);
      if(!stepId) missingFields.push('pipeline[].id');
      else if(stepIds.has(stepId)) duplicateSteps.push(stepId);
      else stepIds.add(stepId);
      if(!stageId) missingFields.push(`pipeline[${stepId || '?'}].stage_id`);
      else if(!knownStages.has(stageId)) unknownStages.push(stageId);
    }
    let combinations = 1;
    const sweepParams = Array.isArray(proposal?.sweep?.parameters) ? proposal.sweep.parameters : [];
    for(const axis of sweepParams) {
      const values = Array.isArray(axis?.values) ? axis.values : [];
      const axisStage = axis?.stage || '';
      const axisStageId = axis?.stage_id || '';
      if(!axisStage || !stepIds.has(axisStage)) {
        result.errors.push(`${prefix}: sweep axis references unknown step "${axisStage || 'missing'}".`);
        proposalIssues.push('broken sweep reference');
      }
      if(axisStageId && !knownStages.has(axisStageId)) {
        result.errors.push(`${prefix}: sweep axis uses unknown stage_id "${axisStageId}".`);
        proposalIssues.push('unknown sweep stage');
      }
      if(!axis?.param) {
        result.errors.push(`${prefix}: sweep axis is missing param.`);
        proposalIssues.push('missing sweep param');
      }
      if(!values.length) {
        result.errors.push(`${prefix}: sweep axis ${axisStage}.${axis?.param || 'param'} has no values.`);
        proposalIssues.push('empty sweep values');
      }
      combinations *= Math.max(1, values.length);
    }
    if(!sweepParams.length) combinations = 1;
    result.total_combinations += combinations;
    if(missingFields.length) {
      result.errors.push(`${prefix}: missing ${[...new Set(missingFields)].join(', ')}.`);
      proposalIssues.push('missing required fields');
    }
    if(duplicateSteps.length) {
      result.errors.push(`${prefix}: duplicate step IDs ${[...new Set(duplicateSteps)].join(', ')}.`);
      proposalIssues.push('duplicate step IDs');
    }
    if(unknownStages.length) {
      result.errors.push(`${prefix}: unknown stage IDs ${[...new Set(unknownStages)].join(', ')}.`);
      proposalIssues.push('unknown stages');
    }
    if(combinations > 4096) {
      result.warnings.push(`${prefix}: ${combinations} sweep combinations exceeds the default importer budget.`);
      proposalIssues.push('large sweep budget');
    }
    if(!proposal?.expected_tradeoffs) proposalIssues.push('missing tradeoff note');
    result.proposals.push({
      id: prefix,
      label: proposal?.label || prefix,
      priority: Number.isFinite(Number(proposal?.priority)) ? Number(proposal.priority) : index + 1,
      rationale: proposal?.rationale || '',
      hypothesis: proposal?.hypothesis || '',
      expected_tradeoffs: proposal?.expected_tradeoffs || '',
      pipeline_count: pipeline.length,
      stage_labels: [...new Set(stageLabels)].slice(0, 6),
      sweep_axes: sweepParams.length,
      combinations,
      unknown_stages: [...new Set(unknownStages)],
      issues: [...new Set(proposalIssues)],
      decision: proposalIssues.some(issue => !['missing tradeoff note', 'large sweep budget'].includes(issue)) ? 'repair first' : combinations > 512 ? 'budget review' : 'candidate'
    });
  }
  if(result.total_combinations > 4096) result.warnings.push(`Total planned combinations are ${result.total_combinations}; consider asking for a smaller proposal pack.`);
  result.status = result.errors.length ? 'invalid' : result.warnings.length ? 'warning' : 'ready';
  result.className = result.errors.length ? 'bad' : result.warnings.length ? 'warn' : 'ok';
  result.label = result.errors.length ? 'needs fixes' : result.warnings.length ? 'review warnings' : 'import-ready';
  result.candidate_count = result.proposals.filter(item => item.decision === 'candidate').length;
  return result;
}

function experimentLlmImportReadiness(intake=experimentLlmProposalIntake()){
  const items = [
    {
      label: 'Valid JSON parsed',
      ok: Boolean(intake.parsed),
      text: intake.parsed ? 'Proposal payload can be parsed.' : 'Paste valid JSON before import.'
    },
    {
      label: 'Required schema fields',
      ok: Boolean(intake.parsed) && !intake.errors.some(err => /schema_version|proposal_set_id|dataset_id|objective|proposals/.test(err)),
      text: 'Top-level schema fields are present for importer validation.'
    },
    {
      label: 'No blocking proposal errors',
      ok: intake.errors.length === 0,
      text: intake.errors.length ? `${intake.errors.length} blocking issue${intake.errors.length === 1 ? '' : 's'} should be fixed first.` : 'No browser-detected blocking proposal issues.'
    },
    {
      label: 'Candidate pack available',
      ok: intake.candidate_count > 0,
      text: intake.candidate_count ? `${intake.candidate_count} candidate proposal${intake.candidate_count === 1 ? '' : 's'} can be exported for the first import pass.` : 'No candidate proposals are available yet.'
    },
    {
      label: 'Sweep budget acceptable',
      ok: intake.total_combinations > 0 && intake.total_combinations <= 4096,
      text: intake.total_combinations ? `${intake.total_combinations} total combinations in the returned pack.` : 'No sweep combinations available yet.'
    }
  ];
  const recommended = intake.candidate_count ? 'candidate' : intake.errors.length ? 'repair' : intake.parsed ? 'full' : 'paste';
  return {
    status: items.every(item => item.ok) ? 'ready' : items.some(item => item.ok) ? 'partial' : 'blocked',
    recommended,
    items
  };
}

function experimentLlmExecutionCommand(proposalName='llm_proposals.json'){
  const proposalPath = `Outputs/ArchitectureRuns/${datasetId}/${proposalName}`;
  const runStem = String(proposalName || 'llm_proposals.json').replace(/\.json$/i, '');
  const runRoot = `Outputs/ArchitectureRuns/${datasetId}/${runStem}_runs`;
  return `python3 tools/run_llm_architecture_experiments.py --proposal ${proposalPath} --run-root ${runRoot}`;
}

function experimentLlmPostImportPlan(intake=experimentLlmProposalIntake()){
  const readiness = experimentLlmImportReadiness(intake);
  const useCandidate = readiness.recommended === 'candidate';
  const proposalName = useCandidate ? `${datasetId}_llm_candidate_proposals.json` : 'llm_proposals.json';
  const commands = intake.import_commands || experimentLlmImportCommands();
  const importCommand = useCandidate ? commands.candidate : commands.full;
  const executionCommand = experimentLlmExecutionCommand(proposalName);
  const steps = [
    {
      label: 'Save proposal JSON',
      status: useCandidate ? 'candidate pack' : intake.parsed ? 'full pack' : 'waiting',
      text: useCandidate ? `Download Candidate Pack and place it at Outputs/ArchitectureRuns/${datasetId}/${proposalName}.` : `Download Parsed Proposal JSON and place it at Outputs/ArchitectureRuns/${datasetId}/${proposalName}.`
    },
    {
      label: 'Import proposals',
      status: readiness.recommended === 'repair' || readiness.recommended === 'paste' ? 'blocked' : 'ready',
      text: readiness.recommended === 'repair' ? 'Repair the proposal JSON before importing.' : 'Run the importer command to add templates and planned runs to architecture_runs.json.'
    },
    {
      label: 'Reload dashboard',
      status: 'manual',
      text: 'Reload this dashboard after import so Proposal Inbox and Pipelines pick up the new templates and runs.'
    },
    {
      label: 'Generate preview',
      status: 'next',
      text: 'Use Experiment Lab or Pipelines to select the first imported run and generate a preview for Review and Data.'
    },
    {
      label: 'Optional local smoke run',
      status: intake.candidate_count || intake.parsed ? 'available' : 'waiting',
      text: 'For executable pipeline stages, run the local proposal experiment command to produce a summary report before deeper review.'
    }
  ];
  return {
    recommended_path: readiness.recommended,
    proposal_name: proposalName,
    import_command: importCommand,
    execution_command: executionCommand,
    steps
  };
}

function experimentLlmPostImportMarkdown(plan=experimentLlmPostImportPlan()){
  const lines = [
    `# LLM Proposal Post-Import Plan: ${datasetId}`,
    '',
    `Recommended path: ${plan.recommended_path}`,
    `Proposal file: ${plan.proposal_name}`,
    '',
    '## Steps',
    ''
  ];
  for(const [index, step] of plan.steps.entries()) {
    lines.push(`${index + 1}. ${step.label} (${step.status})`);
    lines.push(`   ${step.text}`);
  }
  lines.push('', '## Import Command', '', '```bash', plan.import_command, '```');
  lines.push('', '## Optional Local Experiment Command', '', '```bash', plan.execution_command, '```');
  return lines.join('\n') + '\n';
}

function experimentProposalLifecycleRows(intake=experimentLlmProposalIntake()){
  const rows = new Map();
  const add = row => {
    const key = proposalLifecycleKey(row.proposal_set_id, row.proposal_id);
    const previous = rows.get(key) || {};
    rows.set(key, Object.assign({}, previous, row, {
      run_ids: [...new Set([...(previous.run_ids || []), ...(row.run_ids || [])].filter(Boolean))],
      template_ids: [...new Set([...(previous.template_ids || []), ...(row.template_ids || [])].filter(Boolean))]
    }));
  };
  const pastedSet = intake.proposal_set_id || 'pasted_response';
  for(const proposal of intake.proposals || []) {
    add({
      source: 'pasted',
      proposal_set_id: pastedSet,
      proposal_id: proposal.id,
      label: proposal.label,
      priority: proposal.priority,
      triage_decision: proposal.decision,
      combinations: proposal.combinations,
      issues: proposal.issues || [],
      stage_labels: proposal.stage_labels || [],
      template_ids: [],
      run_ids: []
    });
  }
  for(const template of savedPipelineTemplates().filter(item => item.source === 'llm_architecture_proposal')) {
    add({
      source: 'imported',
      proposal_set_id: template.proposal_set_id || template.llm_proposal_set_id || 'imported_llm_proposals',
      proposal_id: template.proposal_id || template.id,
      label: template.label || template.id,
      priority: template.metadata?.priority ?? '',
      triage_decision: 'imported',
      combinations: sweepCombinationCountForRun({sweep: template.sweep}),
      issues: [],
      stage_labels: (template.pipeline || []).map(stage => stageDef(stage)?.label || stageOp(stage)).filter(Boolean).slice(0, 6),
      template_ids: [template.id],
      run_ids: []
    });
  }
  for(const run of architectureRuns().filter(item => item.artifacts?.proposal_id || item.experiment?.proposal_id)) {
    const proposalSetId = run.artifacts?.proposal_set_id || run.experiment?.proposal_set_id || 'imported_llm_proposals';
    const proposalId = run.artifacts?.proposal_id || run.experiment?.proposal_id || run.run_id;
    add({
      source: runGenerated(run) ? 'generated' : 'planned',
      proposal_set_id: proposalSetId,
      proposal_id: proposalId,
      label: run.label || proposalId,
      triage_decision: runGenerated(run) ? 'generated' : 'planned',
      combinations: 1,
      issues: [],
      stage_labels: (run.pipeline || []).map(stage => stageDef(stage)?.label || stageOp(stage)).filter(Boolean).slice(0, 6),
      template_ids: [run.parameters?.template_id || ''].filter(Boolean),
      run_ids: [run.run_id]
    });
  }
  return [...rows.values()].map(row => {
    const generatedRuns = (row.run_ids || []).map(runById).filter(runGenerated);
    const bestRun = generatedRuns.map(run => ({run, utility:runUtilityScore(run)})).sort((a,b) => (b.utility.score ?? -1) - (a.utility.score ?? -1))[0];
    const state = experimentProposalState(row.proposal_set_id, row.proposal_id);
    return Object.assign({}, row, {
      state,
      generated_count: generatedRuns.length,
      best_run_id: bestRun?.run?.run_id || '',
      best_score: bestRun?.utility?.score ?? null,
      human_label: bestRun?.run ? experimentLabel(bestRun.run.run_id) : '',
      artifact_count: bestRun?.run ? numericRunMetric(bestRun.run, 'annotation_summary.triage_queue_counts.artifact_like', 0) : null,
      missed_count: bestRun?.run ? numericRunMetric(bestRun.run, 'annotation_summary.triage_queue_counts.possible_missed_neuron', 0) : null,
      recommended_action: state === 'reject' ? 'Leave out of next pass' :
        state === 'needs repair' || row.triage_decision === 'repair first' ? 'Repair proposal' :
        generatedRuns.length ? 'Review outcome' :
        (row.run_ids || []).length ? 'Generate preview' :
        (row.template_ids || []).length ? 'Plan run' : 'Import proposal'
    });
  }).sort((a,b) => {
    const stateRank = {'try next':0, promising:1, imported:2, generated:3, 'needs repair':4, discussed:5, reject:6};
    return (stateRank[a.state] ?? 3) - (stateRank[b.state] ?? 3) || (a.priority || 99) - (b.priority || 99) || String(a.label).localeCompare(String(b.label));
  });
}

function experimentProposalOutcomeSummary(rows=experimentProposalLifecycleRows()){
  const generated = rows.filter(row => row.generated_count > 0);
  const best = generated.filter(row => row.best_score !== null).sort((a,b) => b.best_score - a.best_score)[0] || null;
  const repair = rows.filter(row => row.recommended_action === 'Repair proposal').length;
  const generate = rows.filter(row => row.recommended_action === 'Generate preview').length;
  return {
    total: rows.length,
    generated: generated.length,
    best,
    repair,
    generate,
    promising: rows.filter(row => ['try next','promising'].includes(row.state)).length
  };
}

function experimentFailureSignals(){
  const s = annotationSummary();
  const labels = architectureRuns().reduce((acc, run) => {
    const label = experimentLabel(run.run_id);
    if(label) acc[label] = (acc[label] || 0) + 1;
    return acc;
  }, {});
  return {
    artifact_like: s.triage_queue_counts.artifact_like || 0,
    possible_missed_neuron: s.triage_queue_counts.possible_missed_neuron || 0,
    weak_trace: s.triage_queue_counts.weak_trace || 0,
    needs_event_review: s.triage_queue_counts.needs_event_review || 0,
    suggestion_unlabeled: s.suggestion_states.unlabeled || 0,
    labels
  };
}

function experimentLlmFollowUpPromptMarkdown(rows=experimentProposalLifecycleRows()){
  const signals = experimentFailureSignals();
  const outcome = experimentProposalOutcomeSummary(rows);
  const lines = [
    `# Follow-up LLM Architecture Request: ${datasetId}`,
    '',
    'Use the review outcomes and proposal lifecycle notes below to propose the next compact architecture iteration.',
    'Return only JSON matching `schemas/llm_architecture_proposal.schema.json`.',
    '',
    '## Current Objective',
    '',
    experimentObjective(),
    '',
    '## Failure Signals',
    '',
    `- Artifact-like review burden: ${signals.artifact_like}`,
    `- Possible missed neurons: ${signals.possible_missed_neuron}`,
    `- Weak/problem traces: ${signals.weak_trace}`,
    `- Events needing review: ${signals.needs_event_review}`,
    `- Unlabeled discovery suggestions: ${signals.suggestion_unlabeled}`,
    `- Run labels: ${JSON.stringify(signals.labels)}`,
    '',
    '## Proposal Outcomes',
    ''
  ];
  if(rows.length) {
    for(const row of rows.slice(0, 12)) {
      lines.push(`- ${row.label} (${row.proposal_id}): state=${row.state || 'unlabeled'}, action=${row.recommended_action}, generated=${row.generated_count}, best_score=${row.best_score ?? 'n/a'}, issues=${(row.issues || []).join(', ') || 'none'}`);
    }
  } else {
    lines.push('- No proposal lifecycle rows yet.');
  }
  lines.push(
    '',
    '## Request',
    '',
    '- If artifacts dominate, propose conservative artifact suppression or multi-stage CFAR variants.',
    '- If missed neurons dominate, propose high-recall variants with explicit review-burden controls.',
    '- If traces are weak or noisy, propose denoising/background-correction changes that preserve sparse transients.',
    '- Keep the proposal pack compact enough for local preview generation.',
    '',
    '## Best Current Proposal',
    '',
    outcome.best ? `- ${outcome.best.label}: utility ${outcome.best.best_score}, best run ${outcome.best.best_run_id}` : '- No generated proposal outcome yet.'
  );
  return lines.join('\n') + '\n';
}

function experimentLlmCandidateProposalPack(intake=experimentLlmProposalIntake()){
  const source = intake.parsed || {};
  const proposals = Array.isArray(source.proposals) ? source.proposals : [];
  const candidateById = new Map(intake.proposals.filter(item => item.decision === 'candidate').map(item => [item.id, item]));
  const kept = proposals.filter((proposal, index) => candidateById.has(proposal?.id || `proposal_${index + 1}`));
  return {
    schema_version: 1,
    proposal_set_id: `${slugify(source.proposal_set_id || `${datasetId}_llm_proposals`)}_candidates`,
    dataset_id: datasetId,
    objective: source.objective || experimentObjective(),
    max_combinations_per_architecture: source.max_combinations_per_architecture || 4096,
    proposals: kept,
    neurobench_triage: {
      generated_at: new Date().toISOString(),
      source_proposal_set_id: source.proposal_set_id || '',
      source_status: intake.status,
      kept_count: kept.length,
      dropped_count: Math.max(0, proposals.length - kept.length),
      kept_ids: kept.map((proposal, index) => proposal?.id || `proposal_${index + 1}`),
      dropped: intake.proposals
        .filter(item => item.decision !== 'candidate')
        .map(item => ({id:item.id, label:item.label, decision:item.decision, issues:item.issues, combinations:item.combinations}))
    }
  };
}

function experimentLlmCandidatePackMarkdown(pack=experimentLlmCandidateProposalPack()){
  const lines = [
    `# Candidate-Only LLM Proposal Pack: ${datasetId}`,
    '',
    `Proposal set: ${pack.proposal_set_id}`,
    `Objective: ${pack.objective || 'n/a'}`,
    `Kept proposals: ${pack.proposals.length}`,
    `Dropped proposals: ${pack.neurobench_triage?.dropped_count ?? 0}`,
    '',
    '## Kept Proposals',
    ''
  ];
  if(pack.proposals.length) {
    for(const proposal of pack.proposals) {
      const sweepAxes = Array.isArray(proposal?.sweep?.parameters) ? proposal.sweep.parameters.length : 0;
      lines.push(`- ${proposal.label || proposal.id}: ${proposal.pipeline?.length || 0} stages, ${sweepAxes} sweep axes.`);
      if(proposal.hypothesis) lines.push(`  Hypothesis: ${proposal.hypothesis}`);
    }
  } else {
    lines.push('- No candidate proposals are available.');
  }
  lines.push('', '## Dropped Proposals', '');
  if(pack.neurobench_triage?.dropped?.length) {
    for(const item of pack.neurobench_triage.dropped) lines.push(`- ${item.label} (${item.id}): ${item.decision}; ${item.issues.join(', ') || 'no issue note'}`);
  } else {
    lines.push('- None.');
  }
  lines.push('', '## Import Command', '', '```bash', experimentLlmImportCommands().candidate, '```');
  return lines.join('\n') + '\n';
}

function experimentLlmProposalTriageMarkdown(intake=experimentLlmProposalIntake()){
  const lines = [
    `# LLM Proposal Triage: ${datasetId}`,
    '',
    `Proposal set: ${intake.proposal_set_id || 'n/a'}`,
    `Objective: ${intake.objective || experimentObjective() || 'n/a'}`,
    `Status: ${intake.label}`,
    `Total combinations: ${intake.total_combinations}`,
    '',
    '## Recommended Review Order',
    ''
  ];
  const rows = [...(intake.proposals || [])].sort((a,b) => {
    const decisionRank = {candidate: 0, 'budget review': 1, 'repair first': 2};
    return (decisionRank[a.decision] ?? 9) - (decisionRank[b.decision] ?? 9) || a.priority - b.priority || a.combinations - b.combinations;
  });
  if(!rows.length) lines.push('- No parsed proposals yet.');
  for(const row of rows) {
    lines.push(`- ${row.label} (${row.id}): ${row.decision}; priority ${row.priority}; ${row.pipeline_count} stages; ${row.sweep_axes} sweep axes; ${row.combinations} combinations.`);
    if(row.stage_labels?.length) lines.push(`  Stages: ${row.stage_labels.join(' -> ')}`);
    if(row.issues?.length) lines.push(`  Issues: ${row.issues.join(', ')}`);
    if(row.hypothesis) lines.push(`  Hypothesis: ${row.hypothesis}`);
    if(row.expected_tradeoffs) lines.push(`  Tradeoffs: ${row.expected_tradeoffs}`);
  }
  lines.push('', '## Import Command', '', '```bash', intake.import_commands?.full || intake.import_command, '```');
  return lines.join('\n') + '\n';
}

function experimentLlmRepairPromptMarkdown(intake=experimentLlmProposalIntake()){
  const spec = experimentLlmPromptSpec();
  const sourceText = experimentLlmResponseText().trim();
  const lines = [
    `# Repair Neurobench LLM Proposal JSON: ${datasetId}`,
    '',
    'Please repair the proposal JSON below so it can be imported into Neurobench.',
    'Return only JSON matching `schemas/llm_architecture_proposal.schema.json`.',
    '',
    '## Objective',
    '',
    spec.objective || 'Not specified.',
    '',
    '## Errors To Fix',
    ''
  ];
  if(intake.errors.length) for(const err of intake.errors) lines.push(`- ${err}`);
  else lines.push('- No blocking browser-side errors were found.');
  lines.push('', '## Warnings To Consider', '');
  if(intake.warnings.length) for(const warning of intake.warnings) lines.push(`- ${warning}`);
  else lines.push('- No warnings.');
  lines.push(
    '',
    '## Repair Rules',
    '',
    '- Keep the same scientific objective unless a correction is needed.',
    '- Use only known stage IDs from the list below.',
    '- Every pipeline step needs a unique `id` and known `stage_id`.',
    '- Every sweep axis must reference a concrete pipeline step `id`, not just a stage type.',
    '- Keep total combinations small enough for local preview generation; prefer compact named searches over harsh grids.',
    '- Preserve the rationale, hypothesis, expected tradeoffs, and priority for each proposal.',
    '',
    '## Known Stage IDs',
    ''
  );
  for(const stage of spec.available_stage_catalog) {
    lines.push(`- ${stage.stage_id}: ${stage.label}; params=${stage.params.join(', ') || 'none'}`);
  }
  lines.push(
    '',
    '## Expected JSON Shape',
    '',
    '```json',
    JSON.stringify(spec.response_schema, null, 2),
    '```',
    '',
    '## Proposal JSON To Repair',
    '',
    '```json',
    sourceText || '{}',
    '```'
  );
  return lines.join('\n') + '\n';
}

function experimentHandoffContext(manifest=experimentManifest(), run=activeRun() || experimentBaselineRun()){
  const baseline = experimentBaselineRun();
  const utility = runUtilityScore(run);
  return {
    schema_version: 1,
    generated_at: new Date().toISOString(),
    dataset: {
      id: datasetId,
      video_name: data.video?.name || data.dataset?.paths?.raw_video || '',
      frame_count: data.video?.frames,
      frame_size: [data.video?.width, data.video?.height],
      frame_rate_hz: data.dataset?.frame_rate_hz || data.video?.frameRateHz || null,
      pixel_size_microns: data.dataset?.pixel_size_microns || null
    },
    active_run: run ? {
      run_id: run.run_id,
      label: runLabel(run),
      status: run.execution?.status || (runGenerated(run) ? 'completed' : 'planned'),
      human_label: experimentLabel(run.run_id),
      note: experimentNote(run.run_id),
      readiness: runReadiness(run),
      utility_score: utility,
      changed_parameters: changedParametersForRun(run),
      pipeline: (run.pipeline || []).map(stage => ({
        id: stage.id,
        op: stageOp(stage),
        label: stageDef(stage)?.label || stage.label || stage.id,
        params: stage.params || {}
      })),
      summary: run.summary || {},
      annotation_summary: run.annotation_summary || null
    } : null,
    baseline_run_id: baseline?.run_id || '',
    baseline_deltas: runDeltaSummary(run, baseline).map(row => ({metric: row.label, value: row.value, baseline: row.base, delta: row.delta})),
    annotation_summary: annotationSummary(),
    session_recipe: experimentSessionRecipe(manifest, run),
    llm_prompt_request: experimentLlmPromptSpec(manifest, run),
    llm_proposal_intake: (() => {
      const intake = experimentLlmProposalIntake();
      const lifecycleRows = experimentProposalLifecycleRows(intake);
      return {
        status: intake.status,
        proposal_set_id: intake.proposal_set_id,
        proposal_count: intake.proposal_count,
        candidate_count: intake.candidate_count,
        total_combinations: intake.total_combinations,
        candidate_pack_available: intake.candidate_count > 0,
        proposal_triage: intake.proposals.map(item => ({
          id: item.id,
          label: item.label,
          priority: item.priority,
          decision: item.decision,
          combinations: item.combinations,
          issues: item.issues
        })),
        errors: intake.errors,
        warnings: intake.warnings,
        repair_prompt_available: Boolean(experimentLlmResponseText().trim()),
        import_command: intake.import_command,
        import_commands: intake.import_commands,
        import_readiness: experimentLlmImportReadiness(intake),
        post_import_plan: experimentLlmPostImportPlan(intake),
        lifecycle_summary: experimentProposalOutcomeSummary(lifecycleRows),
        lifecycle_rows: lifecycleRows.slice(0, 32).map(row => ({
          proposal_set_id: row.proposal_set_id,
          proposal_id: row.proposal_id,
          label: row.label,
          state: row.state,
          recommended_action: row.recommended_action,
          generated_count: row.generated_count,
          best_run_id: row.best_run_id,
          best_score: row.best_score,
          issues: row.issues
        }))
      };
    })(),
    recommendations: recommendationsFromAnnotations().map(rec => ({
      title: rec.title,
      status: rec.status,
      text: rec.text,
      preset: rec.preset || '',
      action: rec.action || ''
    })),
    action_queue: experimentActionQueue(manifest, run).map((action, index) => ({
      rank: index + 1,
      priority: action.priority,
      state: action.state || 'open',
      title: action.title,
      source: action.source,
      detail: action.detail,
      action: action.actionLabel
    })),
    action_history: experimentActionHistory().slice(0, 20),
    planned_experiment: {
      mode: experimentDraft.mode || 'sweep',
      run_count: manifest.runs.length,
      sweep_budget: Math.max(sweepCombinationCountForRun(plannedRun()), manifest.runs.length),
      validation: validatePipeline(pipelineDraft),
      runs: manifest.runs.slice(0, 32).map(item => ({
        run_id: item.run_id,
        label: item.label,
        changed_parameters: changedParametersForRun(item),
        status: item.validation?.status || validatePipeline(item).status
      }))
    },
    shortlist: shortlistedRuns().map(item => ({
      run_id: item.run_id,
      label: runLabel(item),
      human_label: experimentLabel(item.run_id),
      utility_score: runUtilityScore(item).score,
      note: experimentNote(item.run_id)
    })),
    sensitivity: experimentSensitivityRows().map(row => ({
      parameter: row.key,
      values: row.values,
      run_count: row.run_count,
      generated_count: row.generated_count,
      average_score: row.average_score,
      score_spread: row.score_spread,
      best_run_id: row.best_run?.run_id || '',
      best_score: row.best_score,
      recommendation: row.recommendation
    })),
    follow_up_suggestions: experimentFollowUpSuggestions().map(item => ({
      parameter: item.key,
      best_run_id: item.run_id,
      best_value: item.best_value,
      best_score: item.best_score,
      proposed_values: item.values,
      reason: item.reason
    })),
    coverage_gaps: experimentCoverageRows().filter(row => row.status.label !== 'covered').slice(0, 24).map(row => ({
      parameter: row.key,
      stage: row.stage_label,
      current_value: row.current_value,
      tested_values: row.values,
      status: row.status.label,
      probe_values: row.probes,
      recommendation: row.status.text
    })),
    llm_proposal_sets: llmProposalSets().map(set => ({
      id: set.id || set.proposal_set_id || '',
      label: set.label || '',
      objective: set.objective || '',
      template_count: set.templates?.length || 0,
      run_count: set.runs?.length || 0
    }))
  };
}

function experimentBriefMarkdown(manifest=experimentManifest(), run=activeRun() || experimentBaselineRun()){
  const ctx = experimentHandoffContext(manifest, run);
  const active = ctx.active_run;
  const lines = [
    `# Experiment Brief: ${datasetId}`,
    '',
    '## Dataset',
    '',
    `- Video: ${ctx.dataset.video_name || 'n/a'}`,
    `- Frames: ${ctx.dataset.frame_count || 'n/a'}`,
    `- Frame size: ${(ctx.dataset.frame_size || []).filter(v => v !== undefined).join(' x ') || 'n/a'}`,
    `- Frame rate: ${ctx.dataset.frame_rate_hz ?? 'n/a'} Hz`,
    `- Pixel size: ${ctx.dataset.pixel_size_microns ?? 'n/a'} microns/px`,
    '',
    '## Active Run',
    ''
  ];
  if(active) {
    lines.push(
      `- Run: ${active.label} (${active.run_id})`,
      `- Status: ${active.status}`,
      `- Human label: ${active.human_label || 'unlabeled'}`,
      `- Readiness: ${active.readiness.label} - ${active.readiness.text}`,
      `- Utility score: ${active.utility_score.score ?? 'n/a'} (${active.utility_score.status})`,
      `- Note: ${active.note || 'none'}`,
      `- Changed parameters: ${active.changed_parameters.length ? active.changed_parameters.join(', ') : 'base stack'}`
    );
  } else {
    lines.push('- No active run selected.');
  }
  lines.push('', '## Baseline Delta', '');
  if(ctx.baseline_deltas.length) for(const row of ctx.baseline_deltas) lines.push(`- ${row.metric}: ${row.delta >= 0 ? '+' : ''}${fmt(row.delta, Number.isInteger(row.delta) ? 0 : 2)} vs ${ctx.baseline_run_id}`);
  else lines.push('- Active run is the baseline or no completed baseline is available.');
  lines.push('', '## Session Recipe', '');
  lines.push(`- Objective: ${ctx.session_recipe.objective || 'not specified'}`);
  if(ctx.session_recipe.next_action) {
    lines.push(`- Next action: ${ctx.session_recipe.next_action.title} (${ctx.session_recipe.next_action.source})`);
    lines.push(`- Why: ${ctx.session_recipe.next_action.detail}`);
  } else {
    lines.push('- Next action: no open blocking action.');
  }
  lines.push('', '## Recommendations', '');
  for(const rec of ctx.recommendations) lines.push(`- ${rec.title}: ${rec.text}`);
  lines.push('', '## Prioritized Action Queue', '');
  if(ctx.action_queue.length) for(const action of ctx.action_queue.slice(0, 8)) lines.push(`- ${action.rank}. ${action.title} (${action.source}, ${action.state}, priority ${action.priority}): ${action.detail}`);
  else lines.push('- No blocking actions found.');
  lines.push('', '## Action History', '');
  if(ctx.action_history.length) for(const row of ctx.action_history.slice(0, 8)) lines.push(`- ${row.updatedAt}: ${row.state} - ${row.title}`);
  else lines.push('- No action state changes recorded yet.');
  lines.push('', '## Planned Experiment', '');
  lines.push(`- Mode: ${ctx.planned_experiment.mode}`);
  lines.push(`- Planned runs: ${ctx.planned_experiment.run_count}`);
  lines.push(`- Sweep budget: ${ctx.planned_experiment.sweep_budget}`);
  lines.push(`- Validation: ${ctx.planned_experiment.validation.status}`);
  lines.push('', '## Shortlist', '');
  if(ctx.shortlist.length) for(const item of ctx.shortlist) lines.push(`- ${item.label} (${item.run_id}): ${item.human_label}, utility ${item.utility_score ?? 'n/a'}${item.note ? `, note: ${item.note}` : ''}`);
  else lines.push('- No shortlisted runs yet.');
  lines.push('', '## Parameter Sensitivity', '');
  if(ctx.sensitivity.length) for(const row of ctx.sensitivity.slice(0, 8)) lines.push(`- ${row.parameter}: avg utility ${row.average_score === null ? 'n/a' : fmt(row.average_score, 1)}, spread ${fmt(row.score_spread, 1)}, best ${row.best_run_id || 'n/a'}; ${row.recommendation}`);
  else lines.push('- No explicit sweep/named-set parameter changes are available yet.');
  lines.push('', '## Follow-up Suggestions', '');
  if(ctx.follow_up_suggestions.length) for(const item of ctx.follow_up_suggestions.slice(0, 6)) lines.push(`- ${item.parameter}: test ${item.proposed_values.join(', ') || 'n/a'} around ${item.best_value || 'n/a'} from ${item.best_run_id}; ${item.reason}`);
  else lines.push('- No follow-up suggestions are available yet.');
  lines.push('', '## Coverage Gaps', '');
  if(ctx.coverage_gaps.length) for(const row of ctx.coverage_gaps.slice(0, 8)) lines.push(`- ${row.parameter}: ${row.status}; probe ${row.probe_values.join(', ') || 'n/a'}. ${row.recommendation}`);
  else lines.push('- No coverage gaps in the current numeric pipeline parameters.');
  lines.push('', '## Pipeline', '');
  if(active?.pipeline?.length) for(const stage of active.pipeline) lines.push(`- ${stage.id}: ${stage.label} (${stage.op}) ${Object.keys(stage.params || {}).length ? JSON.stringify(stage.params) : ''}`);
  else lines.push('- No pipeline attached.');
  return lines.join('\n') + '\n';
}

function renderExperimentSharePanel(manifest, validation, active){
  const shortlist = shortlistedRuns();
  const activeNote = active ? experimentNote(active.run_id) : '';
  const chips = shortlist.slice(0, 6).map(run => `
    <button type="button" class="shortlistChip" data-load-shortlist-run="${escapeHtml(run.run_id)}">
      ${escapeHtml(runLabel(run))}
      <span>${escapeHtml(experimentLabel(run.run_id))}</span>
    </button>`).join('');
  return `
    <section class="archCard experimentSharePanel supportPanel">
      <div class="runCardHeader">
        <h3>Lab Share And LLM Handoff</h3>
        <span class="runStatus">${shortlist.length} shortlisted</span>
      </div>
      <p class="hint">Capture why a run matters, then export a concise brief or machine-readable context for lab discussion or external LLM architecture feedback.</p>
      <label>Active run note
        <textarea id="experimentActiveRunNote" rows="4" ${active ? '' : 'disabled'} placeholder="Why is this run worth keeping, changing, or rejecting?">${escapeHtml(activeNote)}</textarea>
      </label>
      <div class="buttonRow">
        <button type="button" id="experimentShortlistActiveBtn" ${active ? '' : 'disabled'}>Shortlist Active Run</button>
        <button type="button" id="experimentDownloadBriefBtn">Download Brief</button>
        <button type="button" id="experimentDownloadHandoffBtn">Download LLM Context</button>
      </div>
      <div class="shortlistGrid">${chips || '<p class="hint">No shortlisted runs yet. Mark promising runs as shortlist, looks best, or baseline candidate.</p>'}</div>
      <details>
        <summary>Brief Preview</summary>
        <pre class="briefPreview">${escapeHtml(experimentBriefMarkdown(manifest, active))}</pre>
      </details>
    </section>`;
}

function renderExperimentSessionRecipePanel(manifest, active){
  const recipe = experimentSessionRecipe(manifest, active);
  const moves = recipe.parameter_moves.slice(0, 4).map(move => `
    <li><b>${escapeHtml(move.parameter)}</b>: ${escapeHtml(move.values.join(', ') || 'n/a')}<br><span class="hint">${escapeHtml(move.reason)}</span></li>`).join('');
  const runs = recipe.run_recipe.slice(0, 4).map(run => `
    <li><b>${escapeHtml(run.label)}</b><br><span class="hint">${escapeHtml(run.changed_parameters.join(', ') || 'base stack')} · ${escapeHtml(run.readiness)}</span></li>`).join('');
  return `
    <section class="archCard experimentSessionRecipe primaryPanel">
      <div class="runCardHeader">
        <div>
          <h3>Session Recipe</h3>
          <p class="hint">A compact, copyable plan for the current objective, next action, run list, and parameter moves.</p>
        </div>
        <span class="runStatus">${recipe.next_action ? 'next action' : 'clear'}</span>
      </div>
      <label>Objective
        <textarea id="experimentObjectiveInput" rows="2" placeholder="What should the next experiment improve?">${escapeHtml(recipe.objective)}</textarea>
      </label>
      <div class="recipeNextAction">
        <span class="priorityBadge">${recipe.next_action?.priority ?? 'ok'}</span>
        <div>
          <h3>${escapeHtml(recipe.next_action?.title || 'No open blocking action')}</h3>
          <p>${escapeHtml(recipe.next_action?.detail || 'Compare shortlisted generated runs or broaden the experiment space.')}</p>
        </div>
      </div>
      <div class="recipeMiniGrid">
        <div>
          <h3>Runs</h3>
          <ol>${runs || '<li>No planned runs yet.</li>'}</ol>
        </div>
        <div>
          <h3>Parameters</h3>
          <ol>${moves || '<li>No suggested parameter moves yet.</li>'}</ol>
        </div>
      </div>
      <div class="buttonRow">
        <button type="button" id="experimentRunTopRecipeActionBtn" ${recipe.next_action ? '' : 'disabled'}>Do Next Action</button>
        <button type="button" id="experimentCopyRecipeBtn">Copy Recipe</button>
        <button type="button" id="experimentDownloadRecipeBtn">Download Recipe</button>
        <button type="button" id="experimentDownloadRecipeJsonBtn">Download Recipe JSON</button>
      </div>
    </section>`;
}

function renderExperimentLlmPromptPanel(manifest, active){
  const spec = experimentLlmPromptSpec(manifest, active);
  const modes = [
    ['architecture_feedback', 'Architecture feedback'],
    ['parameter_search', 'Parameter search'],
    ['noise_artifact', 'Noise/artifact control'],
    ['realtime_100hz', '100 Hz real-time']
  ].map(([value, label]) => `<option value="${value}" ${spec.mode === value ? 'selected' : ''}>${label}</option>`).join('');
  return `
    <section class="archCard experimentLlmPromptPanel primaryPanel">
      <div class="runCardHeader">
        <div>
          <h3>LLM Architecture Request</h3>
          <p class="hint">Build a provider-neutral prompt from the current recipe, stack, constraints, and import schema.</p>
        </div>
        <span class="runStatus">${spec.available_stage_catalog.length} stages</span>
      </div>
      <div class="llmPromptControls">
        <label>Request type
          <select id="experimentLlmPromptMode">${modes}</select>
        </label>
        <label>Extra constraints
          <textarea id="experimentLlmConstraintsInput" rows="3" placeholder="What should an external model preserve or avoid?">${escapeHtml(spec.constraints)}</textarea>
        </label>
      </div>
      <div class="llmPromptSummary">
        <div><b>Objective</b><span>${escapeHtml(spec.objective || 'not specified')}</span></div>
        <div><b>Current stack</b><span>${spec.current_stack.length} stages</span></div>
        <div><b>Response</b><span>proposal JSON for importer</span></div>
      </div>
      <div class="buttonRow">
        <button type="button" id="experimentCopyLlmPromptBtn">Copy Prompt</button>
        <button type="button" id="experimentDownloadLlmPromptBtn">Download Prompt</button>
        <button type="button" id="experimentDownloadLlmPromptJsonBtn">Download Prompt JSON</button>
      </div>
      <details>
        <summary>Prompt Preview</summary>
        <pre class="briefPreview">${escapeHtml(experimentLlmPromptMarkdown(spec))}</pre>
      </details>
    </section>`;
}

function renderExperimentLlmIntakePanel(){
  const text = experimentLlmResponseText();
  const intake = experimentLlmProposalIntake(text);
  const readiness = experimentLlmImportReadiness(intake);
  const postImport = experimentLlmPostImportPlan(intake);
  const commands = intake.import_commands || experimentLlmImportCommands();
  const proposalRows = intake.proposals.slice(0, 8).map(item => `
    <tr>
      <td><b>${escapeHtml(item.label)}</b><br><span class="hint">${escapeHtml(item.id)}</span></td>
      <td>${item.pipeline_count}</td>
      <td>${item.sweep_axes}</td>
      <td>${item.combinations}</td>
      <td>${escapeHtml(item.unknown_stages.join(', ') || 'none')}</td>
    </tr>`).join('');
  const triageRows = [...intake.proposals].sort((a,b) => {
    const decisionRank = {candidate: 0, 'budget review': 1, 'repair first': 2};
    return (decisionRank[a.decision] ?? 9) - (decisionRank[b.decision] ?? 9) || a.priority - b.priority || a.combinations - b.combinations;
  }).slice(0, 8).map(item => `
    <tr>
      <td><span class="stageStatus ${item.decision === 'candidate' ? 'ok' : item.decision === 'budget review' ? 'warn' : 'bad'}">${escapeHtml(item.decision)}</span></td>
      <td><b>${escapeHtml(item.label)}</b><br><span class="hint">${escapeHtml(item.stage_labels.join(' -> ') || 'no stages')}</span></td>
      <td>${escapeHtml(item.priority)}</td>
      <td>${escapeHtml(item.combinations)}</td>
      <td>${escapeHtml(item.issues.join(', ') || 'none')}</td>
    </tr>`).join('');
  const issueList = [...intake.errors.map(text => ['bad', text]), ...intake.warnings.map(text => ['warn', text])].slice(0, 8)
    .map(([cls, text]) => `<li><span class="stageStatus ${cls}">${cls === 'bad' ? 'fix' : 'check'}</span> ${escapeHtml(text)}</li>`)
    .join('');
  const readinessRows = readiness.items.map(item => `
    <li class="${item.ok ? 'ok' : 'warn'}">
      <span class="stageStatus ${item.ok ? 'ok' : 'warn'}">${item.ok ? 'ready' : 'todo'}</span>
      <b>${escapeHtml(item.label)}</b>
      <span>${escapeHtml(item.text)}</span>
    </li>`).join('');
  return `
    <section class="archCard experimentLlmIntakePanel supportPanel">
      <div class="runCardHeader">
        <div>
          <h3>LLM Proposal Intake</h3>
          <p class="hint">Paste returned proposal JSON here for lightweight checks before running the local importer.</p>
        </div>
        <span class="stageStatus ${intake.className}">${escapeHtml(intake.label)}</span>
      </div>
      <label>Returned proposal JSON
        <textarea id="experimentLlmResponseInput" rows="8" placeholder='{"schema_version":1,"proposal_set_id":"...","dataset_id":"...","objective":"...","proposals":[...]}'>${escapeHtml(text)}</textarea>
      </label>
      <div class="llmIntakeSummary">
        <div><b>${intake.proposal_count}</b><span>proposals</span></div>
        <div><b>${intake.candidate_count}</b><span>candidates</span></div>
        <div><b>${intake.total_combinations}</b><span>combinations</span></div>
        <div><b>${intake.errors.length}</b><span>errors</span></div>
        <div><b>${intake.warnings.length}</b><span>warnings</span></div>
      </div>
      ${issueList ? `<ul class="llmIntakeIssues">${issueList}</ul>` : '<p class="hint">No import-blocking issues found by the browser checks. The CLI importer still performs authoritative validation.</p>'}
      <table class="smallTable compareTable">
        <tr><th>Proposal</th><th>Stages</th><th>Sweep axes</th><th>Combos</th><th>Unknown stages</th></tr>
        ${proposalRows || '<tr><td colspan="5">No parsed proposals yet.</td></tr>'}
      </table>
      <div class="llmTriageBlock">
        <div class="runCardHeader">
          <h3>Proposal Triage</h3>
          <span class="runStatus">${intake.proposals.filter(item => item.decision === 'candidate').length} candidates</span>
        </div>
        <table class="smallTable compareTable">
          <tr><th>Decision</th><th>Proposal</th><th>Priority</th><th>Combos</th><th>Issues</th></tr>
          ${triageRows || '<tr><td colspan="5">Paste proposal JSON to see a review order.</td></tr>'}
        </table>
      </div>
      <label>Local CLI import command
        <textarea id="experimentLlmImportCommand" rows="2" readonly>${escapeHtml(readiness.recommended === 'candidate' ? commands.candidate : commands.full)}</textarea>
      </label>
      <label>Local server import endpoint
        <textarea id="experimentLlmServerImportEndpoint" rows="1" readonly>${escapeHtml(commands.serverEndpoint)}</textarea>
      </label>
      <div class="llmImportReadiness">
        <div class="runCardHeader">
          <h3>Import Readiness</h3>
          <span class="stageStatus ${readiness.status === 'ready' ? 'ok' : readiness.status === 'partial' ? 'warn' : 'bad'}">${escapeHtml(readiness.recommended)}</span>
        </div>
        <ul>${readinessRows}</ul>
        <div class="buttonRow">
          <button type="button" id="experimentServerImportFullBtn" ${intake.parsed && !intake.errors.length ? '' : 'disabled'}>Import Full To Dashboard</button>
          <button type="button" id="experimentServerImportCandidateBtn" ${intake.candidate_count ? '' : 'disabled'}>Import Candidates To Dashboard</button>
          <button type="button" id="experimentCopyFullImportCommandBtn">Copy Full Import</button>
          <button type="button" id="experimentCopyCandidateImportCommandBtn" ${intake.candidate_count ? '' : 'disabled'}>Copy Candidate Import</button>
        </div>
      </div>
      <div class="llmPostImportPlan">
        <div class="runCardHeader">
          <h3>Post-Import Plan</h3>
          <span class="runStatus">${escapeHtml(postImport.recommended_path)}</span>
        </div>
        <ol>
          ${postImport.steps.map(step => `<li><b>${escapeHtml(step.label)}</b><span>${escapeHtml(step.text)}</span></li>`).join('')}
        </ol>
        <label>Optional local experiment command
          <textarea id="experimentLlmExecutionCommand" rows="2" readonly>${escapeHtml(postImport.execution_command)}</textarea>
        </label>
        <div class="buttonRow">
          <button type="button" id="experimentCopyPostImportPlanBtn">Copy Post-Import Plan</button>
          <button type="button" id="experimentDownloadPostImportPlanBtn">Download Post-Import Plan</button>
          <button type="button" id="experimentCopyExecutionCommandBtn">Copy Local Experiment Command</button>
        </div>
      </div>
      <div class="buttonRow">
        <button type="button" id="experimentValidateLlmResponseBtn">Validate Response</button>
        <button type="button" id="experimentCopyImportCommandBtn">Copy Import Command</button>
        <button type="button" id="experimentCopyRepairPromptBtn" ${text.trim() ? '' : 'disabled'}>Copy Repair Prompt</button>
        <button type="button" id="experimentDownloadRepairPromptBtn" ${text.trim() ? '' : 'disabled'}>Download Repair Prompt</button>
        <button type="button" id="experimentDownloadProposalTriageBtn" ${intake.proposals.length ? '' : 'disabled'}>Download Triage Note</button>
        <button type="button" id="experimentDownloadCandidatePackBtn" ${intake.candidate_count ? '' : 'disabled'}>Download Candidate Pack</button>
        <button type="button" id="experimentDownloadCandidatePackNoteBtn" ${intake.candidate_count ? '' : 'disabled'}>Download Candidate Note</button>
        <button type="button" id="experimentDownloadParsedProposalBtn" ${intake.parsed && !intake.errors.length ? '' : 'disabled'}>Download Parsed Proposal JSON</button>
      </div>
      ${text.trim() ? `
        <details class="llmRepairPromptPreview">
          <summary>Repair Prompt Preview</summary>
          <pre class="briefPreview">${escapeHtml(experimentLlmRepairPromptMarkdown(intake))}</pre>
        </details>` : ''}
    </section>`;
}

function renderExperimentProposalLifecyclePanel(){
  const rows = experimentProposalLifecycleRows();
  const outcome = experimentProposalOutcomeSummary(rows);
  const stateOptions = [''].concat(experimentProposalStateChoices()).map(choice => choice ? `<option value="${escapeHtml(choice)}">${escapeHtml(choice)}</option>` : '<option value="">unlabeled</option>').join('');
  const body = rows.slice(0, 14).map(row => `
    <tr>
      <td>
        <select data-proposal-state="${escapeHtml(row.proposal_set_id)}::${escapeHtml(row.proposal_id)}">
          ${stateOptions.replace(`value="${escapeHtml(row.state)}"`, `value="${escapeHtml(row.state)}" selected`)}
        </select>
      </td>
      <td><b>${escapeHtml(row.label || row.proposal_id)}</b><br><span class="hint">${escapeHtml(row.proposal_set_id)} / ${escapeHtml(row.proposal_id)}</span></td>
      <td><span class="stageStatus ${row.recommended_action === 'Repair proposal' ? 'bad' : row.recommended_action === 'Generate preview' ? 'warn' : 'ok'}">${escapeHtml(row.recommended_action)}</span></td>
      <td>${escapeHtml(row.generated_count)}</td>
      <td>${escapeHtml(row.best_score ?? 'n/a')}</td>
      <td>${escapeHtml(row.human_label || 'unlabeled')}</td>
      <td>${row.artifact_count === null ? 'n/a' : escapeHtml(row.artifact_count)} / ${row.missed_count === null ? 'n/a' : escapeHtml(row.missed_count)}</td>
      <td>${escapeHtml((row.issues || []).join(', ') || 'none')}</td>
    </tr>`).join('');
  return `
    <section class="archCard experimentProposalLifecyclePanel supportPanel">
      <details class="llmWorkflowSection" open>
        <summary>
          <span>LLM Proposal Lifecycle</span>
          <span class="runStatus">${outcome.total} proposals</span>
        </summary>
        <p class="hint">Track proposal state from pasted idea to import, generated preview, review outcome, and follow-up prompt.</p>
        <div class="llmLifecycleSummary">
          <div><b>${outcome.generated}</b><span>generated</span></div>
          <div><b>${outcome.promising}</b><span>promising/next</span></div>
          <div><b>${outcome.repair}</b><span>repair-first</span></div>
          <div><b>${outcome.generate}</b><span>need preview</span></div>
        </div>
        <table class="smallTable compareTable">
          <tr><th>State</th><th>Proposal</th><th>Next</th><th>Generated</th><th>Best utility</th><th>Run label</th><th>Artifacts / missed</th><th>Issues</th></tr>
          ${body || '<tr><td colspan="8">No LLM proposals are available yet. Paste a returned proposal JSON or import a proposal set.</td></tr>'}
        </table>
        <div class="buttonRow">
          <button type="button" id="experimentCopyFollowUpPromptBtn" ${rows.length ? '' : 'disabled'}>Copy Follow-up Prompt</button>
          <button type="button" id="experimentDownloadFollowUpPromptBtn" ${rows.length ? '' : 'disabled'}>Download Follow-up Prompt</button>
        </div>
        <details>
          <summary>Outcome Feedback Prompt Preview</summary>
          <pre class="briefPreview">${escapeHtml(experimentLlmFollowUpPromptMarkdown(rows))}</pre>
        </details>
      </details>
    </section>`;
}

function experimentDecisionFilter(){
  return annotations.settings.experimentDecisionFilter || 'all';
}

function setExperimentDecisionFilter(value){
  annotations.settings.experimentDecisionFilter = value || 'all';
  queueSave();
  renderExperimentLab();
}

function runDecisionAction(run){
  const readiness = runReadiness(run);
  const label = experimentLabel(run?.run_id || '');
  if(readiness.className === 'bad') return 'Fix pipeline validation';
  if(!runGenerated(run)) return 'Generate preview';
  if(label === 'too noisy' || label === 'artifact heavy') return 'Try artifact suppression';
  if(label === 'too strict') return 'Try high-recall pass';
  if(!run?.annotation_summary) return 'Attach or build annotation summary';
  if(!experimentNote(run.run_id)) return 'Add decision note';
  return 'Compare or shortlist';
}

function experimentDecisionRows(filter=experimentDecisionFilter()){
  const positive = new Set(['shortlist', 'looks best', 'baseline candidate']);
  const noisy = new Set(['too noisy', 'artifact heavy', 'too strict']);
  return architectureRuns().filter(run => {
    const label = experimentLabel(run.run_id);
    if(filter === 'generated') return runGenerated(run);
    if(filter === 'planned') return !runGenerated(run);
    if(filter === 'shortlisted') return positive.has(label);
    if(filter === 'needs_review') return !label || label === 'needs review';
    if(filter === 'noisy_or_artifact') return noisy.has(label);
    return true;
  }).map(run => {
    const utility = runUtilityScore(run);
    const readiness = runReadiness(run);
    const label = experimentLabel(run.run_id);
    const score = utility.score === null ? -1 : utility.score;
    return {run, utility, readiness, label, score, action: runDecisionAction(run)};
  }).sort((a,b) => {
    const positiveA = ['shortlist','looks best','baseline candidate'].includes(a.label) ? 1 : 0;
    const positiveB = ['shortlist','looks best','baseline candidate'].includes(b.label) ? 1 : 0;
    return positiveB - positiveA || b.score - a.score || Number(runGenerated(b.run)) - Number(runGenerated(a.run)) || runLabel(a.run).localeCompare(runLabel(b.run));
  });
}

function cleanTsv(value){
  return String(value ?? '').replace(/\t/g, ' ').replace(/\r?\n/g, ' ').trim();
}

function experimentDecisionMatrixTsv(rows=experimentDecisionRows()){
  const header = ['rank','run_id','label','status','readiness','utility_score','human_label','next_action','changed_parameters','note'];
  const body = rows.map((row, index) => [
    index + 1,
    row.run.run_id,
    runLabel(row.run),
    row.run.execution?.status || (runGenerated(row.run) ? 'completed' : 'planned'),
    row.readiness.label,
    row.utility.score ?? '',
    row.label || '',
    row.action,
    changedParametersForRun(row.run).join(', ') || pipelineChangeSummary(row.run),
    experimentNote(row.run.run_id)
  ].map(cleanTsv).join('\t'));
  return [header.join('\t'), ...body].join('\n') + '\n';
}

function changedParameterEntriesForRun(run){
  const entries = [];
  for(const p of run?.sweep?.parameters || []){
    if(p?.param) entries.push({
      stage: p.stage || p.stage_id || '',
      stage_id: p.stage_id || p.stage || '',
      param: p.param,
      value: p.value ?? (Array.isArray(p.values) ? p.values.join(',') : ''),
      key: `${p.stage || p.stage_id || ''}.${p.param}`
    });
  }
  const override = run?.experiment?.override;
  if(override?.param) entries.push({
    stage: override.stage || override.stage_id || '',
    stage_id: override.stage_id || override.stage || '',
    param: override.param,
    value: override.value ?? '',
    key: `${override.stage || override.stage_id || ''}.${override.param}`
  });
  return entries.filter(entry => entry.param && entry.key !== `.${entry.param}`);
}

function sensitivityRecommendation(row){
  if(!row.generated_count) return 'Generate previews before judging this parameter.';
  if(row.scored_count < 2) return 'Test at least two generated values for this parameter.';
  if(row.noisy_count > 0) return 'Review noisy/artifact labels before widening this axis.';
  if(row.score_spread >= 15) return 'This parameter appears influential; refine around the best value.';
  if(row.best_score !== null && row.best_score >= 72) return 'Shortlist the best run and compare against baseline.';
  return 'Keep as a secondary axis until more labels are available.';
}

function experimentSensitivityRows(){
  const groups = new Map();
  for(const run of architectureRuns()){
    const entries = changedParameterEntriesForRun(run);
    if(!entries.length) continue;
    const utility = runUtilityScore(run);
    const score = utility.score;
    const label = experimentLabel(run.run_id);
    for(const entry of entries){
      const key = entry.key;
      if(!groups.has(key)) groups.set(key, {
        key,
        stage: entry.stage,
        param: entry.param,
        values: new Set(),
        runs: [],
        scores: [],
        generated_count: 0,
        noisy_count: 0,
        positive_count: 0,
        best_run: null,
        best_score: null
      });
      const group = groups.get(key);
      group.values.add(String(entry.value));
      group.runs.push({run, entry, score, label});
      if(runGenerated(run)) group.generated_count++;
      if(['too noisy','artifact heavy','too strict'].includes(label)) group.noisy_count++;
      if(['shortlist','looks best','baseline candidate'].includes(label)) group.positive_count++;
      if(score !== null && score !== undefined) {
        group.scores.push(score);
        if(group.best_score === null || score > group.best_score) {
          group.best_score = score;
          group.best_run = run;
        }
      }
    }
  }
  return [...groups.values()].map(group => {
    const scoreMin = group.scores.length ? Math.min(...group.scores) : null;
    const scoreMax = group.scores.length ? Math.max(...group.scores) : null;
    const row = {
      key: group.key,
      stage: group.stage,
      param: group.param,
      values: [...group.values].filter(v => v !== '').slice(0, 10),
      run_count: group.runs.length,
      generated_count: group.generated_count,
      scored_count: group.scores.length,
      average_score: group.scores.length ? group.scores.reduce((a,b) => a + b, 0) / group.scores.length : null,
      score_spread: scoreMin === null || scoreMax === null ? 0 : scoreMax - scoreMin,
      best_run: group.best_run,
      best_score: group.best_score,
      noisy_count: group.noisy_count,
      positive_count: group.positive_count
    };
    row.recommendation = sensitivityRecommendation(row);
    return row;
  }).sort((a,b) => (b.average_score ?? -1) - (a.average_score ?? -1) || b.score_spread - a.score_spread || b.run_count - a.run_count || a.key.localeCompare(b.key));
}

function experimentSensitivityTsv(rows=experimentSensitivityRows()){
  const header = ['parameter','values','runs','generated_runs','scored_runs','average_utility','score_spread','best_run_id','best_score','positive_labels','noisy_labels','recommendation'];
  const body = rows.map(row => [
    row.key,
    row.values.join(', '),
    row.run_count,
    row.generated_count,
    row.scored_count,
    row.average_score === null ? '' : fmt(row.average_score, 1),
    fmt(row.score_spread, 1),
    row.best_run?.run_id || '',
    row.best_score ?? '',
    row.positive_count,
    row.noisy_count,
    row.recommendation
  ].map(cleanTsv).join('\t'));
  return [header.join('\t'), ...body].join('\n') + '\n';
}

function renderExperimentSensitivityPanel(){
  const rows = experimentSensitivityRows();
  const body = rows.slice(0, 12).map(row => `
    <tr>
      <td><b>${escapeHtml(row.key)}</b><br><span class="hint">${escapeHtml(row.values.join(', ') || 'values not recorded')}</span></td>
      <td>${row.run_count}</td>
      <td>${row.generated_count}</td>
      <td>${row.average_score === null ? 'n/a' : fmt(row.average_score, 1)}</td>
      <td>${fmt(row.score_spread, 1)}</td>
      <td>${row.best_run ? `<button type="button" data-sensitivity-use-run="${escapeHtml(row.best_run.run_id)}">${escapeHtml(runLabel(row.best_run))}</button><br><span class="hint">score ${escapeHtml(row.best_score ?? 'n/a')}</span>` : '<span class="hint">not scored</span>'}</td>
      <td>${escapeHtml(row.recommendation)}</td>
    </tr>`).join('');
  return `
    <section class="archCard experimentSensitivityPanel diagnosticPanel">
      <div class="runCardHeader">
        <div>
          <h3>Parameter Sensitivity</h3>
          <p class="hint">Groups explicit sweep/named-set changes to show which knobs have useful signal so far.</p>
        </div>
        <span class="runStatus">${rows.length} parameter${rows.length === 1 ? '' : 's'}</span>
      </div>
      <table class="smallTable compareTable">
        <tr><th>Parameter</th><th>Runs</th><th>Generated</th><th>Avg utility</th><th>Spread</th><th>Best run</th><th>Suggested next move</th></tr>
        ${body || '<tr><td colspan="7">No explicit parameter changes found yet. Create a sweep or named set to populate this panel.</td></tr>'}
      </table>
      ${rows.length > 12 ? `<p class="hint">Showing first 12 of ${rows.length} changed parameters.</p>` : ''}
      <div class="buttonRow">
        <button type="button" id="experimentDownloadSensitivityBtn">Download Sensitivity TSV</button>
      </div>
    </section>`;
}

function clampNumber(value, min, max){
  let out = value;
  if(Number.isFinite(Number(min))) out = Math.max(Number(min), out);
  if(Number.isFinite(Number(max))) out = Math.min(Number(max), out);
  return out;
}

function paramSpecForRunKey(run, key){
  const [stageKey, param] = String(key || '').split('.');
  const stage = (run?.pipeline || []).find(s => s.id === stageKey) || (run?.pipeline || []).find(s => stageOp(s) === stageKey);
  const def = stageDef(stage);
  return {stage, def, param, spec: def?.params?.[param] || null};
}

function followUpValuesForSensitivity(row){
  if(!row?.best_run) return [];
  const bestEntry = changedParameterEntriesForRun(row.best_run).find(entry => entry.key === row.key);
  const bestValue = Number(bestEntry?.value);
  if(!Number.isFinite(bestValue)) return [];
  const {spec} = paramSpecForRunKey(row.best_run, row.key);
  const tested = row.values.map(Number).filter(Number.isFinite).sort((a,b) => a-b);
  const nearestGap = tested.length > 1 ? Math.min(...tested.slice(1).map((value, index) => Math.abs(value - tested[index])).filter(v => v > 0)) : null;
  const step = Number(spec?.step);
  const delta = Number.isFinite(nearestGap) ? nearestGap / 2 : Number.isFinite(step) && step > 0 ? step : Math.max(Math.abs(bestValue) * 0.15, 0.1);
  const candidates = [bestValue - delta, bestValue, bestValue + delta].map(value => {
    const clamped = clampNumber(value, spec?.min, spec?.max);
    return Number(clamped.toFixed(6));
  });
  return [...new Set(candidates)].filter(value => Number.isFinite(value));
}

function followUpReason(row){
  if(!row.generated_count) return 'No generated output exists yet for this parameter; start with a preview.';
  if(row.scored_count < 2) return 'Only one scored value exists; add nearby values before interpreting sensitivity.';
  if(row.score_spread >= 15) return 'Utility spread is large enough to justify a local refinement around the best value.';
  if(row.best_score !== null && row.best_score >= 72) return 'Best score is promising; confirm it with nearby values or a replicate.';
  return 'Signal is weak so far; keep the follow-up small.';
}

function experimentFollowUpSuggestions(){
  return experimentSensitivityRows().filter(row => row.best_run).slice(0, 6).map((row, index) => {
    const bestEntry = changedParameterEntriesForRun(row.best_run).find(entry => entry.key === row.key);
    const values = followUpValuesForSensitivity(row);
    const {stage, param} = paramSpecForRunKey(row.best_run, row.key);
    return {
      id: slugify(`followup_${row.key}_${row.best_run?.run_id || index}`),
      key: row.key,
      stage: stage?.id || bestEntry?.stage || '',
      stage_id: stageOp(stage) || bestEntry?.stage_id || '',
      param: param || bestEntry?.param || row.param,
      run_id: row.best_run.run_id,
      run_label: runLabel(row.best_run),
      best_value: bestEntry?.value ?? '',
      best_score: row.best_score,
      values,
      score_spread: row.score_spread,
      reason: followUpReason(row)
    };
  });
}

function experimentFollowUpTsv(rows=experimentFollowUpSuggestions()){
  const header = ['parameter','best_run_id','best_value','best_score','proposed_values','score_spread','reason'];
  const body = rows.map(row => [
    row.key,
    row.run_id,
    row.best_value,
    row.best_score ?? '',
    row.values.join(', '),
    fmt(row.score_spread, 1),
    row.reason
  ].map(cleanTsv).join('\t'));
  return [header.join('\t'), ...body].join('\n') + '\n';
}

function ensureFollowUpBase(suggestion){
  const stageExists = pipelineDraft.pipeline?.some(stage => stage.id === suggestion.stage);
  if(stageExists) return;
  const run = runById(suggestion.run_id);
  if(!run) return;
  pipelineDraft = normalizePipelineDraft(JSON.parse(JSON.stringify(Object.assign({}, run, {execution:{status:'planned'}}))));
  pipelineDraft.run_id = `planned_followup_${slugify(run.run_id)}_${Date.now().toString(36)}`;
  pipelineDraft.label = `Follow-up from ${runLabel(run)}`;
  selectedPipelineStageId = pipelineDraft.pipeline?.[0]?.id || null;
  experimentDraft.baseTemplateId = '';
}

function applyFollowUpSuggestion(id, mode='sets'){
  const suggestion = experimentFollowUpSuggestions().find(item => item.id === id);
  if(!suggestion || !suggestion.values.length) {
    setSaveState('follow-up needs a numeric best value before it can be added', 'bad');
    return;
  }
  ensureFollowUpBase(suggestion);
  const stage = pipelineDraft.pipeline?.find(item => item.id === suggestion.stage);
  if(!stage) {
    setSaveState(`could not find stage ${suggestion.stage} in the current pipeline`, 'bad');
    return;
  }
  if(mode === 'sweep') {
    const axes = sweepFactors(pipelineDraft).filter(axis => !(axis.stage === suggestion.stage && axis.param === suggestion.param));
    axes.push({
      stage: suggestion.stage,
      stage_id: suggestion.stage_id || stageOp(stage),
      param: suggestion.param,
      values: suggestion.values,
      label: `${suggestion.key} follow-up`
    });
    setSweepFactors(axes);
    experimentDraft.mode = 'sweep';
    setSaveState(`added follow-up sweep for ${suggestion.key}`, 'ok');
  } else {
    const stamp = Date.now().toString(36);
    for(const [index, value] of suggestion.values.entries()){
      experimentDraft.setRows.push({
        run_id: `${pipelineDraft.run_id}__followup_${slugify(suggestion.param)}_${index + 1}_${stamp}`,
        label: `${pipelineDraft.label || pipelineDraft.run_id} | ${suggestion.key}=${value}`,
        stage: suggestion.stage,
        stage_id: suggestion.stage_id || stageOp(stage),
        param: suggestion.param,
        value,
        experiment: {source:'followup_planner', parent_run_id:suggestion.run_id, reason:suggestion.reason}
      });
    }
    experimentDraft.mode = 'sets';
    setSaveState(`added ${suggestion.values.length} follow-up named sets for ${suggestion.key}`, 'ok');
  }
  renderExperimentLab();
  renderPipelineBuilder();
}

function renderExperimentFollowUpPlanner(){
  const suggestions = experimentFollowUpSuggestions();
  const cards = suggestions.map(item => `
    <article class="followUpCard">
      <div class="runCardHeader">
        <h3>${escapeHtml(item.key)}</h3>
        <span class="utilityPill ${item.best_score !== null && item.best_score >= 72 ? 'ok' : item.best_score !== null ? 'warn' : 'off'}">${escapeHtml(item.best_score ?? 'n/a')}</span>
      </div>
      <p class="hint">Best so far: ${escapeHtml(item.run_label)} at ${escapeHtml(item.best_value || 'n/a')}</p>
      <p>${escapeHtml(item.reason)}</p>
      <div class="miniChipRow">${item.values.length ? item.values.map(value => `<span>${escapeHtml(item.param)}=${escapeHtml(value)}</span>`).join('') : '<span>non-numeric value</span>'}</div>
      <div class="buttonRow">
        <button type="button" data-followup-use-run="${escapeHtml(item.run_id)}">Use Best Run</button>
        <button type="button" data-followup-add-sets="${escapeHtml(item.id)}" ${item.values.length ? '' : 'disabled'}>Add Named Sets</button>
        <button type="button" data-followup-add-sweep="${escapeHtml(item.id)}" ${item.values.length ? '' : 'disabled'}>Add Sweep Axis</button>
      </div>
    </article>`).join('');
  return `
    <section class="archCard experimentFollowUpPlanner primaryPanel">
      <div class="runCardHeader">
        <div>
          <h3>Follow-up Planner</h3>
          <p class="hint">Convert sensitivity signals into small local refinements around the best observed values.</p>
        </div>
        <span class="runStatus">${suggestions.length} suggestion${suggestions.length === 1 ? '' : 's'}</span>
      </div>
      <div class="followUpGrid">${cards || '<p class="hint">Run or import explicit parameter variants before follow-up suggestions are available.</p>'}</div>
      <div class="buttonRow">
        <button type="button" id="experimentDownloadFollowUpBtn">Download Follow-up TSV</button>
      </div>
    </section>`;
}

function coverageProbeValues(option){
  const current = Number(option.value);
  const fallback = Number(option.spec?.default);
  const center = Number.isFinite(current) ? current : (Number.isFinite(fallback) ? fallback : 0);
  const step = Number(option.spec?.step);
  const delta = Number.isFinite(step) && step > 0 ? step : Math.max(Math.abs(center) * 0.15, 0.1);
  return [...new Set([center - delta, center, center + delta].map(value => Number(clampNumber(value, option.spec?.min, option.spec?.max).toFixed(6))))].filter(Number.isFinite);
}

function coverageStatus(row){
  if(row.generated_count === 0) return {className:'warn', label:'untested', text:'Add a tiny probe before judging this knob.'};
  if(row.value_count < 2) return {className:'warn', label:'thin', text:'Only one value is represented; add nearby values.'};
  if(row.scored_count < 2) return {className:'warn', label:'needs scores', text:'Generated variants need annotation summaries or labels.'};
  return {className:'ok', label:'covered', text:'Enough first-pass coverage exists for comparison.'};
}

function experimentCoverageRows(){
  const tested = new Map();
  for(const run of architectureRuns()){
    for(const entry of changedParameterEntriesForRun(run)){
      if(!tested.has(entry.key)) tested.set(entry.key, {values:new Set(), generated_count:0, scored_count:0, runs:[]});
      const bucket = tested.get(entry.key);
      bucket.values.add(String(entry.value));
      bucket.runs.push(run);
      if(runGenerated(run)) bucket.generated_count++;
      if(runUtilityScore(run).score !== null) bucket.scored_count++;
    }
  }
  return experimentParamOptions().map(option => {
    const key = `${option.stage.id}.${option.name}`;
    const bucket = tested.get(key) || {values:new Set(), generated_count:0, scored_count:0, runs:[]};
    const row = {
      key,
      stage: option.stage.id,
      stage_id: stageOp(option.stage),
      param: option.name,
      label: `${option.stage.id}.${option.name}`,
      stage_label: option.def?.label || stageOp(option.stage),
      current_value: option.value,
      values: [...bucket.values].filter(Boolean).slice(0, 12),
      value_count: bucket.values.size,
      run_count: bucket.runs.length,
      generated_count: bucket.generated_count,
      scored_count: bucket.scored_count,
      probes: coverageProbeValues(option)
    };
    row.status = coverageStatus(row);
    return row;
  }).sort((a,b) => {
    const rank = {untested:0, thin:1, 'needs scores':2, covered:3};
    return (rank[a.status.label] ?? 9) - (rank[b.status.label] ?? 9) || a.stage_label.localeCompare(b.stage_label) || a.param.localeCompare(b.param);
  });
}

function experimentCoverageTsv(rows=experimentCoverageRows()){
  const header = ['parameter','stage','current_value','tested_values','runs','generated_runs','scored_runs','coverage_status','probe_values','recommendation'];
  const body = rows.map(row => [
    row.key,
    row.stage_label,
    row.current_value,
    row.values.join(', '),
    row.run_count,
    row.generated_count,
    row.scored_count,
    row.status.label,
    row.probes.join(', '),
    row.status.text
  ].map(cleanTsv).join('\t'));
  return [header.join('\t'), ...body].join('\n') + '\n';
}

function applyCoverageProbe(key, mode='sets'){
  const option = experimentParamOptions().find(item => `${item.stage.id}.${item.name}` === key);
  if(!option) {
    setSaveState(`could not find parameter ${key} in the current pipeline`, 'bad');
    return;
  }
  const values = coverageProbeValues(option);
  if(!values.length) {
    setSaveState(`could not create numeric probes for ${key}`, 'bad');
    return;
  }
  if(mode === 'sweep') {
    const axes = sweepFactors(pipelineDraft).filter(axis => !(axis.stage === option.stage.id && axis.param === option.name));
    axes.push({
      stage: option.stage.id,
      stage_id: stageOp(option.stage),
      param: option.name,
      values,
      label: `${key} coverage probe`
    });
    setSweepFactors(axes);
    experimentDraft.mode = 'sweep';
    setSaveState(`added coverage sweep for ${key}`, 'ok');
  } else {
    const stamp = Date.now().toString(36);
    for(const [index, value] of values.entries()){
      experimentDraft.setRows.push({
        run_id: `${pipelineDraft.run_id}__coverage_${slugify(option.name)}_${index + 1}_${stamp}`,
        label: `${pipelineDraft.label || pipelineDraft.run_id} | ${key}=${value}`,
        stage: option.stage.id,
        stage_id: stageOp(option.stage),
        param: option.name,
        value,
        experiment: {source:'coverage_map', reason:'first-pass parameter coverage'}
      });
    }
    experimentDraft.mode = 'sets';
    setSaveState(`added ${values.length} coverage probe sets for ${key}`, 'ok');
  }
  renderExperimentLab();
  renderPipelineBuilder();
}

function renderExperimentCoverageMap(){
  const rows = experimentCoverageRows();
  const body = rows.slice(0, 14).map(row => `
    <tr>
      <td><b>${escapeHtml(row.key)}</b><br><span class="hint">${escapeHtml(row.stage_label)}</span></td>
      <td>${escapeHtml(row.current_value)}</td>
      <td>${escapeHtml(row.values.join(', ') || 'none')}</td>
      <td><span class="stageStatus ${row.status.className}">${escapeHtml(row.status.label)}</span></td>
      <td>${escapeHtml(row.probes.join(', '))}</td>
      <td>${escapeHtml(row.status.text)}</td>
      <td>
        <button type="button" data-coverage-add-sets="${escapeHtml(row.key)}">Add Probe Sets</button>
        <button type="button" data-coverage-add-sweep="${escapeHtml(row.key)}">Add Sweep</button>
      </td>
    </tr>`).join('');
  return `
    <section class="archCard experimentCoverageMap diagnosticPanel">
      <div class="runCardHeader">
        <div>
          <h3>Parameter Coverage Map</h3>
          <p class="hint">Find tunable numeric parameters in the current stack that have not been explored yet.</p>
        </div>
        <span class="runStatus">${rows.filter(row => row.status.label !== 'covered').length} gaps</span>
      </div>
      <table class="smallTable compareTable">
        <tr><th>Parameter</th><th>Current</th><th>Tested</th><th>Status</th><th>Probe values</th><th>Suggested next move</th><th></th></tr>
        ${body || '<tr><td colspan="7">No numeric parameters are available in the current pipeline.</td></tr>'}
      </table>
      ${rows.length > 14 ? `<p class="hint">Showing first 14 of ${rows.length} numeric parameters.</p>` : ''}
      <div class="buttonRow">
        <button type="button" id="experimentDownloadCoverageBtn">Download Coverage TSV</button>
      </div>
    </section>`;
}

function experimentActionStates(){
  annotations.settings.experimentActionStates = annotations.settings.experimentActionStates || {};
  return annotations.settings.experimentActionStates;
}

function experimentActionState(actionId){
  return experimentActionStates()[actionId]?.state || 'open';
}

function experimentActionHistory(){
  annotations.settings.experimentActionHistory = Array.isArray(annotations.settings.experimentActionHistory) ? annotations.settings.experimentActionHistory : [];
  return annotations.settings.experimentActionHistory;
}

function recordExperimentActionHistory(actionId, state){
  const action = experimentActionQueue(experimentManifest(), activeRun() || experimentBaselineRun(), 'all').find(item => item.id === actionId);
  experimentActionHistory().unshift({
    action_id: actionId,
    state: state || 'open',
    title: action?.title || actionId,
    source: action?.source || '',
    priority: action?.priority ?? '',
    updatedAt: new Date().toISOString()
  });
  annotations.settings.experimentActionHistory = experimentActionHistory().slice(0, 80);
}

function setExperimentActionState(actionId, state='open'){
  const states = experimentActionStates();
  if(!actionId) return;
  if(!state || state === 'open') delete states[actionId];
  else states[actionId] = {state, updatedAt:new Date().toISOString()};
  recordExperimentActionHistory(actionId, state || 'open');
  queueSave();
  renderExperimentLab();
}
