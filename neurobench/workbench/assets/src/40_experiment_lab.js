function experimentActionFilter(){
  return annotations.settings.experimentActionFilter || 'open';
}

const EXPERIMENT_PANEL_MODE_CLASSES = {
  guided: 'panelMode-guided',
  diagnostics: 'panelMode-diagnostics',
  all: 'panelMode-all'
};

function experimentPanelMode(){
  return annotations.settings.experimentPanelMode || 'guided';
}

function setExperimentPanelMode(value){
  annotations.settings.experimentPanelMode = value || 'guided';
  queueSave();
  renderExperimentLab();
}

function setExperimentActionFilter(value){
  annotations.settings.experimentActionFilter = value || 'open';
  queueSave();
  renderExperimentLab();
}

function experimentActionQueue(manifest=experimentManifest(), active=activeRun() || experimentBaselineRun(), filter=experimentActionFilter()){
  const actions = [];
  const add = action => actions.push(Object.assign({priority:50, status:'next', detail:'', actionLabel:'Open'}, action));
  const s = annotationSummary();
  if(!s.review_progress.tuning_ready) add({
    id:'review_seed',
    priority:98,
    title:'Finish review seed labels',
    source:'Evaluation Checklist',
    detail:`Need ${Math.max(0, s.review_progress.tuning_ready_targets.reviewed_rois - s.review_progress.reviewed_rois)} ROI and ${Math.max(0, s.review_progress.tuning_ready_targets.reviewed_events - s.review_progress.reviewed_events)} event labels before tuning has a stable target.`,
    href:'#review',
    actionLabel:'Open Review'
  });
  if(!architectureRuns().some(runGenerated)) add({
    id:'generate_preview',
    priority:94,
    title:'Generate the first preview',
    source:'Generation',
    detail:'Planned runs need generated review data before Review, Data, or utility scores can judge them.',
    action:'generate_preview',
    actionLabel:'Generate Preview'
  });
  if(active && runGenerated(active) && !experimentNote(active.run_id)) add({
    id:'decision_note',
    priority:82,
    title:'Add a decision note to the active run',
    source:'Lab Share',
    detail:'A short note keeps the reason for keeping, changing, or rejecting this run visible in briefs and exports.',
    focus:'experimentActiveRunNote',
    actionLabel:'Write Note'
  });
  if(!shortlistedRuns().length && active) add({
    id:'shortlist_active',
    priority:78,
    title:'Shortlist one candidate run',
    source:'Decision Matrix',
    detail:'Mark at least one promising or baseline candidate before exporting a lab brief.',
    action:'shortlist_active',
    actionLabel:'Shortlist Active'
  });
  for(const item of experimentFollowUpSuggestions().slice(0, 3)) add({
    id:`followup_${item.id}`,
    priority:item.best_score !== null && item.best_score >= 72 ? 76 : 68,
    title:`Refine ${item.key}`,
    source:'Follow-up Planner',
    detail:`Best observed value is ${item.best_value || 'n/a'} in ${item.run_label}; proposed values: ${item.values.join(', ') || 'n/a'}.`,
    followupId:item.id,
    action:'followup_sets',
    actionLabel:'Add Named Sets'
  });
  for(const row of experimentCoverageRows().filter(row => row.status.label !== 'covered').slice(0, 3)) add({
    id:`coverage_${row.key}`,
    priority:row.status.label === 'untested' ? 72 : 60,
    title:`Probe ${row.key}`,
    source:'Coverage Map',
    detail:`${row.status.text} Suggested values: ${row.probes.join(', ') || 'n/a'}.`,
    coverageKey:row.key,
    action:'coverage_sets',
    actionLabel:'Add Probe Sets'
  });
  for(const rec of recommendationsFromAnnotations().slice(0, 2)) add({
    id:`recommend_${slugify(rec.title)}`,
    priority:rec.status === 'noise' || rec.status === 'recall' ? 70 : 55,
    title:rec.title,
    source:'Recommendations',
    detail:rec.text,
    preset:rec.preset || '',
    href:rec.href || '',
    action:rec.preset ? 'recommend_experiment' : rec.generate ? 'generate_preview' : '',
    actionLabel:rec.preset ? 'Plan Experiment' : rec.generate ? 'Generate Preview' : (rec.action || 'Open')
  });
  if(active && runGenerated(active)) {
    const missingOutputs = qcStageTiles(active).filter(tile => tile.status === 'missing').length;
    if(missingOutputs) add({
      id:'process_outputs',
      priority:52,
      title:'Inspect missing Data outputs',
      source:'Data',
      detail:`${missingOutputs} declared stage output${missingOutputs === 1 ? '' : 's'} are missing or not browser-readable for the active run.`,
      href:'#data',
      actionLabel:'Open Data'
    });
  }
  const seen = new Set();
  const ranked = actions
    .filter(action => {
      if(seen.has(action.id)) return false;
      seen.add(action.id);
      return true;
    })
    .map(action => Object.assign({}, action, {state:experimentActionState(action.id)}))
    .sort((a,b) => b.priority - a.priority || a.title.localeCompare(b.title))
    .slice(0, 16);
  return ranked.filter(action => {
    if(filter === 'all') return true;
    if(filter === 'done') return action.state === 'done';
    if(filter === 'snoozed') return action.state === 'snoozed';
    return action.state === 'open';
  }).slice(0, 10);
}

function experimentActionQueueTsv(actions=experimentActionQueue()){
  const header = ['rank','priority','state','title','source','detail','action'];
  const body = actions.map((action, index) => [
    index + 1,
    action.priority,
    action.state || 'open',
    action.title,
    action.source,
    action.detail,
    action.actionLabel
  ].map(cleanTsv).join('\t'));
  return [header.join('\t'), ...body].join('\n') + '\n';
}

function experimentActionHistoryTsv(rows=experimentActionHistory()){
  const header = ['updated_at','state','priority','title','source','action_id'];
  const body = rows.map(row => [
    row.updatedAt,
    row.state,
    row.priority,
    row.title,
    row.source,
    row.action_id
  ].map(cleanTsv).join('\t'));
  return [header.join('\t'), ...body].join('\n') + '\n';
}

function runExperimentAction(id){
  const action = experimentActionQueue().find(item => item.id === id);
  if(!action) return;
  if(action.action === 'generate_preview') generateExperimentPreview();
  else if(action.action === 'shortlist_active') {
    const run = activeRun();
    if(run) setExperimentLabel(run.run_id, 'shortlist');
  } else if(action.action === 'followup_sets') applyFollowUpSuggestion(action.followupId, 'sets');
  else if(action.action === 'coverage_sets') applyCoverageProbe(action.coverageKey, 'sets');
  else if(action.action === 'recommend_experiment' && action.preset) {
    pipelineDraft = makePresetPipeline(action.preset);
    selectedPipelineStageId = pipelineDraft.pipeline?.[0]?.id || null;
    experimentDraft.baseTemplateId = '';
    renderExperimentLab();
    renderPipelineBuilder();
  } else if(action.href) {
    location.hash = action.href;
  } else if(action.focus) {
    document.getElementById(action.focus)?.focus();
  }
}

function renderExperimentActionQueue(manifest, active){
  const filter = experimentActionFilter();
  const actions = experimentActionQueue(manifest, active, filter);
  const allActions = experimentActionQueue(manifest, active, 'all');
  const counts = allActions.reduce((acc, action) => {
    acc[action.state || 'open'] = (acc[action.state || 'open'] || 0) + 1;
    return acc;
  }, {open:0, done:0, snoozed:0});
  const filterOptions = [
    ['open', `Open (${counts.open || 0})`],
    ['all', `All (${allActions.length})`],
    ['done', `Done (${counts.done || 0})`],
    ['snoozed', `Snoozed (${counts.snoozed || 0})`]
  ].map(([value, label]) => `<option value="${value}" ${filter === value ? 'selected' : ''}>${label}</option>`).join('');
  const rows = actions.map((action, index) => `
    <tr class="actionState-${escapeHtml(action.state || 'open')}">
      <td>${index + 1}</td>
      <td><span class="priorityBadge">${action.priority}</span></td>
      <td><b>${escapeHtml(action.title)}</b><br><span class="hint">${escapeHtml(action.source)}</span></td>
      <td>${escapeHtml(action.detail)}</td>
      <td>
        ${action.href && !action.action ? `<a class="textButton" href="${escapeHtml(action.href)}">${escapeHtml(action.actionLabel)}</a>` : `<button type="button" data-action-queue-run="${escapeHtml(action.id)}">${escapeHtml(action.actionLabel)}</button>`}
        <button type="button" data-action-state="${escapeHtml(action.id)}" data-action-state-value="done">Done</button>
        <button type="button" data-action-state="${escapeHtml(action.id)}" data-action-state-value="snoozed">Snooze</button>
        ${action.state !== 'open' ? `<button type="button" data-action-state="${escapeHtml(action.id)}" data-action-state-value="open">Reopen</button>` : ''}
      </td>
    </tr>`).join('');
  return `
    <section class="archCard experimentActionQueue primaryPanel">
      <div class="runCardHeader">
        <div>
          <h3>Prioritized Action Queue</h3>
          <p class="hint">Merged next steps from readiness, recommendations, sensitivity, follow-ups, and coverage.</p>
        </div>
        <label>View <select id="experimentActionFilter">${filterOptions}</select></label>
      </div>
      <table class="smallTable compareTable">
        <tr><th>#</th><th>Priority</th><th>Action</th><th>Why</th><th></th></tr>
        ${rows || '<tr><td colspan="5">No blocking actions found. Compare shortlisted generated runs or broaden experiments.</td></tr>'}
      </table>
      <div class="buttonRow">
        <button type="button" id="experimentDownloadActionQueueBtn">Download Action Queue TSV</button>
      </div>
    </section>`;
}

function renderExperimentActionHistory(){
  const rows = experimentActionHistory().slice(0, 8).map(row => `
    <tr>
      <td>${escapeHtml(row.updatedAt || '')}</td>
      <td><span class="stageStatus ${row.state === 'done' ? 'ok' : row.state === 'snoozed' ? 'warn' : 'off'}">${escapeHtml(row.state || 'open')}</span></td>
      <td><b>${escapeHtml(row.title || row.action_id)}</b><br><span class="hint">${escapeHtml(row.source || row.action_id)}</span></td>
      <td>${escapeHtml(row.priority ?? '')}</td>
    </tr>`).join('');
  return `
    <section class="archCard experimentActionHistory supportPanel">
      <div class="runCardHeader">
        <div>
          <h3>Action History</h3>
          <p class="hint">Recent queue decisions for handoff and session continuity.</p>
        </div>
        <span class="runStatus">${experimentActionHistory().length} events</span>
      </div>
      <table class="smallTable compareTable">
        <tr><th>Updated</th><th>State</th><th>Action</th><th>Priority</th></tr>
        ${rows || '<tr><td colspan="4">No action state changes recorded yet.</td></tr>'}
      </table>
      <div class="buttonRow">
        <button type="button" id="experimentDownloadActionHistoryBtn">Download Action History TSV</button>
      </div>
    </section>`;
}

function experimentReadinessItems(manifest, active){
  const s = annotationSummary();
  const generatedCount = architectureRuns().filter(runGenerated).length;
  const shortlistCount = shortlistedRuns().length;
  const budget = Math.max(sweepCombinationCountForRun(plannedRun()), manifest.runs.length);
  const budgetState = sweepBudgetStatus(budget);
  const missingOutputs = active ? qcStageTiles(active).filter(tile => tile.status === 'missing').length : 0;
  return [
    {
      title:'Review seed',
      ok:s.review_progress.tuning_ready,
      text:s.review_progress.tuning_ready ? 'Enough ROI/event labels exist for first-pass tuning.' : `Need ${Math.max(0, s.review_progress.tuning_ready_targets.reviewed_rois - s.review_progress.reviewed_rois)} ROI and ${Math.max(0, s.review_progress.tuning_ready_targets.reviewed_events - s.review_progress.reviewed_events)} event labels.`,
      href:'#review',
      action:'Review'
    },
    {
      title:'Generated preview',
      ok:generatedCount > 0,
      text:generatedCount ? `${generatedCount} generated run${generatedCount === 1 ? '' : 's'} available for inspection.` : 'Generate at least one preview before comparing architecture quality.',
      generate:true,
      action:'Generate'
    },
    {
      title:'Shortlist',
      ok:shortlistCount > 0,
      text:shortlistCount ? `${shortlistCount} run${shortlistCount === 1 ? '' : 's'} shortlisted for lab discussion.` : 'Mark at least one promising or baseline candidate run.',
      action:'Shortlist active',
      shortlist:true
    },
    {
      title:'Decision note',
      ok:Boolean(active && experimentNote(active.run_id)),
      text:active && experimentNote(active.run_id) ? 'Active run has a decision note.' : 'Add a short note explaining why the active run matters.',
      action:'Write note'
    },
    {
      title:'Process outputs',
      ok:Boolean(active && runGenerated(active) && missingOutputs === 0),
      text:active ? (missingOutputs ? `${missingOutputs} declared stage output${missingOutputs === 1 ? '' : 's'} missing in Data.` : 'Data has the declared stage outputs for the active run.') : 'No active run selected.',
      href:'#data',
      action:'Data'
    },
    {
      title:'Sweep budget',
      ok:budgetState.className !== 'bad',
      text:`${budget} planned combination${budget === 1 ? '' : 's'}: ${budgetState.text}`,
      action:'Adjust budget'
    }
  ];
}

function renderExperimentReadinessChecklist(manifest, active){
  const items = experimentReadinessItems(manifest, active);
  return `
    <section class="archCard experimentReadinessPanel primaryPanel">
      <div class="runCardHeader">
        <h3>Evaluation Checklist</h3>
        <span class="runStatus">${items.filter(item => item.ok).length}/${items.length} ready</span>
      </div>
      <div class="checklistGrid">
        ${items.map(item => `
          <article class="checklistItem ${item.ok ? 'ok' : 'warn'}">
            <span class="stageStatus ${item.ok ? 'ok' : 'warn'}">${item.ok ? 'ready' : 'todo'}</span>
            <h3>${escapeHtml(item.title)}</h3>
            <p>${escapeHtml(item.text)}</p>
            ${item.href ? `<a class="textButton" href="${escapeHtml(item.href)}">${escapeHtml(item.action)}</a>` : item.generate ? '<button type="button" data-checklist-generate>Generate</button>' : item.shortlist ? '<button type="button" data-checklist-shortlist>Shortlist Active</button>' : ''}
          </article>`).join('')}
      </div>
    </section>`;
}

function renderExperimentDecisionMatrix(){
  const filter = experimentDecisionFilter();
  const rows = experimentDecisionRows(filter);
  const filterOptions = [
    ['all', 'All runs'],
    ['generated', 'Generated'],
    ['planned', 'Planned'],
    ['shortlisted', 'Shortlisted'],
    ['needs_review', 'Needs review'],
    ['noisy_or_artifact', 'Noisy/artifact']
  ].map(([value, label]) => `<option value="${value}" ${filter === value ? 'selected' : ''}>${label}</option>`).join('');
  const body = rows.slice(0, 20).map((row, index) => `
    <tr class="${row.run.run_id === activeRunId() ? 'activeRunRow' : ''}">
      <td>${index + 1}</td>
      <td><b>${escapeHtml(runLabel(row.run))}</b><br><span class="hint">${escapeHtml(row.run.run_id)}</span></td>
      <td><span class="utilityPill ${row.utility.className}">${escapeHtml(row.utility.score ?? 'n/a')}</span></td>
      <td><span class="stageStatus ${row.readiness.className}">${escapeHtml(row.readiness.label)}</span></td>
      <td>
        <select data-decision-label="${escapeHtml(row.run.run_id)}">
          <option value="">unlabeled</option>
          ${experimentLabelChoices().map(choice => `<option value="${escapeHtml(choice)}" ${row.label === choice ? 'selected' : ''}>${escapeHtml(choice)}</option>`).join('')}
        </select>
      </td>
      <td>${escapeHtml(row.action)}</td>
      <td class="decisionNoteCell">${escapeHtml(experimentNote(row.run.run_id) || 'no note')}</td>
      <td>
        <button type="button" data-decision-use-run="${escapeHtml(row.run.run_id)}">Use</button>
        <button type="button" data-decision-load-run="${escapeHtml(row.run.run_id)}" ${runGenerated(row.run) ? '' : 'disabled'}>Review</button>
      </td>
    </tr>`).join('');
  return `
    <section class="archCard experimentDecisionMatrix supportPanel">
      <div class="runCardHeader">
        <div>
          <h3>Decision Matrix</h3>
          <p class="hint">Rank runs by shortlist status, utility, and generated readiness, then label candidates directly from the table.</p>
        </div>
        <label>View <select id="experimentDecisionFilter">${filterOptions}</select></label>
      </div>
      <table class="smallTable compareTable">
        <tr><th>#</th><th>Run</th><th>Utility</th><th>Ready</th><th>Decision</th><th>Next action</th><th>Note</th><th></th></tr>
        ${body || '<tr><td colspan="8">No runs match this decision filter.</td></tr>'}
      </table>
      ${rows.length > 20 ? `<p class="hint">Showing first 20 of ${rows.length} matching runs.</p>` : ''}
      <div class="buttonRow">
        <button type="button" id="experimentDownloadDecisionMatrixBtn">Download Decision Matrix TSV</button>
      </div>
    </section>`;
}

function renderExperimentCommandCenter(manifest, validation){
  const runs = architectureRuns();
  const active = activeRun() || runs[0] || null;
  const baseline = experimentBaselineRun();
  const panelMode = experimentPanelMode();
  const panelClass = EXPERIMENT_PANEL_MODE_CLASSES[panelMode] || EXPERIMENT_PANEL_MODE_CLASSES.guided;
  const plannedCount = runs.filter(run => !runGenerated(run)).length;
  const generatedCount = runs.filter(runGenerated).length;
  const failedCount = runs.filter(run => String(run.execution?.status || '').toLowerCase() === 'failed').length;
  const budget = sweepCombinationCountForRun(plannedRun());
  const budgetState = sweepBudgetStatus(Math.max(budget, manifest.runs.length));
  const proposalSets = llmProposalSets();
  const queueRows = runs.slice(0, 8).map(run => {
    const readiness = runReadiness(run);
    return `
      <tr class="${run.run_id === activeRunId() ? 'activeRunRow' : ''}">
        <td><b>${escapeHtml(runLabel(run))}</b><br><span class="hint">${escapeHtml(run.run_id)}</span></td>
        <td><span class="stageStatus ${readiness.className}">${escapeHtml(readiness.label)}</span></td>
        <td>${escapeHtml(pipelineChangeSummary(run) || 'base stack')}</td>
        <td>${escapeHtml(experimentLabel(run.run_id) || 'unlabeled')}</td>
        <td>
          <button type="button" data-experiment-use-run="${escapeHtml(run.run_id)}">Use</button>
          <button type="button" data-experiment-load-run="${escapeHtml(run.run_id)}" ${runGenerated(run) ? '' : 'disabled'}>Review</button>
        </td>
      </tr>`;
  }).join('');
  const proposalCards = proposalSets.map(set => `
    <article class="proposalCard">
      <div class="runCardHeader">
        <h3>${escapeHtml(set.label || set.id || 'LLM proposal set')}</h3>
        <span class="runStatus">${escapeHtml((set.templates?.length || 0) + (set.runs?.length || 0))} items</span>
      </div>
      <p class="hint">${escapeHtml(set.objective || set.summary || set.description || 'Imported architectures and planned runs from external LLM planning.')}</p>
      <div class="miniChipRow">
        ${(set.templates || []).slice(0, 4).map(t => `<span>${escapeHtml(t.label || t.id)}</span>`).join('')}
        ${(set.runs || []).slice(0, 4).map(r => `<span>${escapeHtml(r.label || r.run_id)}</span>`).join('')}
      </div>
      <div class="buttonRow">
        ${(set.templates || []).slice(0, 2).map(t => `<button type="button" data-load-llm-template="${escapeHtml(t.id)}">Load ${escapeHtml(t.label || t.id)}</button>`).join('')}
      </div>
    </article>`).join('');
  const recommendationCards = recommendationsFromAnnotations().map(rec => `
    <article class="recommendationCard">
      <div class="runCardHeader"><h3>${escapeHtml(rec.title)}</h3><span class="stageStatus warn">${escapeHtml(rec.status)}</span></div>
      <p>${escapeHtml(rec.text)}</p>
      <div class="buttonRow">
        ${rec.preset ? `<button type="button" data-recommend-preset="${escapeHtml(rec.preset)}">Load Architecture</button><button type="button" data-recommend-experiment="${escapeHtml(rec.preset)}">Plan Experiment</button>` : ''}
        ${rec.generate ? '<button type="button" data-recommend-generate>Generate Preview</button>' : ''}
        ${rec.href ? `<a class="textButton" href="${escapeHtml(rec.href)}">${escapeHtml(rec.action || 'Open')}</a>` : ''}
      </div>
    </article>`).join('');
  return `
    <section class="experimentCommandCenter ${escapeHtml(panelClass)}">
      <div class="runCardHeader">
        <div>
          <h2>Experiment Command Center</h2>
          <p class="hint">Use review labels, proposal imports, run readiness, and sweep budget before launching more compute.</p>
        </div>
        <div class="commandCenterControls">
          <label>Focus
            <select id="experimentPanelMode">
              <option value="guided" ${panelMode === 'guided' ? 'selected' : ''}>Guided</option>
              <option value="diagnostics" ${panelMode === 'diagnostics' ? 'selected' : ''}>Diagnostics</option>
              <option value="all" ${panelMode === 'all' ? 'selected' : ''}>All</option>
            </select>
          </label>
          <span class="stageStatus ${statusClass(validation.status)}">${escapeHtml(validation.status)}</span>
        </div>
      </div>
      <div class="metricGrid">
        <div class="metric"><b>${generatedCount}</b><span>generated runs</span></div>
        <div class="metric"><b>${plannedCount}</b><span>planned runs</span></div>
        <div class="metric"><b>${failedCount}</b><span>failed runs</span></div>
        <div class="metric"><b>${manifest.runs.length}</b><span>draft runs</span></div>
        <div class="metric"><b>${Math.max(budget, manifest.runs.length)}</b><span>sweep budget</span></div>
        <div class="metric"><b>${proposalSets.length}</b><span>LLM proposal sets</span></div>
      </div>
      <div class="pipelineWarning sweepBudget ${budgetState.className}">
        Sweep budget: <b>${escapeHtml(budgetState.label)}</b>. ${escapeHtml(budgetState.text)}
      </div>
      <div class="commandCenterGrid">
        <section class="archCard routinePanel">
          <div class="runCardHeader"><h3>Experiment Queue</h3><span class="runStatus">${runs.length} total</span></div>
          <table class="smallTable compareTable">
            <tr><th>Run</th><th>Ready</th><th>Change</th><th>Label</th><th></th></tr>
            ${queueRows || '<tr><td colspan="5">No saved or planned runs yet.</td></tr>'}
          </table>
        </section>
        <section class="archCard recommendationPanel primaryPanel">
          <div class="runCardHeader"><h3>Recommendations</h3><span class="runStatus">annotation-aware</span></div>
          <div class="recommendationGrid">${recommendationCards}</div>
        </section>
        ${renderExperimentSessionRecipePanel(manifest, active)}
        ${renderExperimentLlmPromptPanel(manifest, active)}
        ${renderExperimentLlmIntakePanel()}
        ${renderExperimentProposalLifecyclePanel()}
        <section class="archCard proposalInbox diagnosticPanel">
          <div class="runCardHeader"><h3>Proposal Inbox</h3><span class="runStatus">${proposalSets.length} set${proposalSets.length === 1 ? '' : 's'}</span></div>
          ${proposalCards || '<p class="hint">No LLM proposal sets have been imported yet. Imported proposal packs will appear here as reusable architectures.</p>'}
        </section>
        ${renderExperimentActionQueue(manifest, active)}
        ${renderExperimentActionHistory()}
        ${renderExperimentReadinessChecklist(manifest, active)}
        ${renderExperimentDecisionMatrix()}
        ${renderExperimentSensitivityPanel()}
        ${renderExperimentFollowUpPlanner()}
        ${renderExperimentCoverageMap()}
        ${renderExperimentSharePanel(manifest, validation, active)}
        <section class="scorecardGrid primaryPanel">
          ${renderUtilityScorecard(active)}
          ${renderRunDifferenceCard(active, baseline)}
        </section>
      </div>
    </section>`;
}

function experimentParamOptions(){
  const options = [];
  for(const stage of pipelineDraft.pipeline || []){
    const def = stageDef(stage);
    for(const [name, spec] of Object.entries(def?.params || {})){
      if(spec.type === 'number') {
        options.push({stage, def, name, spec, value: stage.params?.[name] ?? spec.default ?? ''});
      }
    }
  }
  return options;
}

function experimentBaseOptions(){
  const templateOptions = savedPipelineTemplates().map(item => ({kind: 'template', id: item.id, label: `Saved: ${item.label || item.id}`}));
  const presetOptions = ARCHITECTURE_PRESETS.map(item => ({kind: 'preset', id: item.id, label: `Preset: ${item.label}`}));
  const runOptions = architectureRuns().map(run => ({kind: 'run', id: run.run_id, label: `Run: ${run.label || run.run_id}`}));
  return [...templateOptions, ...presetOptions, ...runOptions];
}

function applyExperimentBase(value){
  const [kind, ...rest] = String(value || '').split(':');
  const id = rest.join(':');
  if(kind === 'template') {
    const template = savedPipelineTemplates().find(item => item.id === id);
    if(template) pipelineDraft = normalizeTemplateForDraft(template);
    experimentDraft.baseTemplateId = id;
  } else if(kind === 'run') {
    const run = runById(id);
    if(run) pipelineDraft = normalizePipelineDraft(JSON.parse(JSON.stringify(Object.assign({}, run, {execution:{status:'planned'}}))));
    experimentDraft.baseTemplateId = '';
  } else {
    pipelineDraft = makePresetPipeline(id || 'current_review_pipeline');
    pipelineDraft.template_id = slugify(pipelineDraft.run_id);
    experimentDraft.baseTemplateId = '';
  }
  selectedPipelineStageId = pipelineDraft.pipeline?.[0]?.id || null;
  experimentDraft.setRows = [];
  experimentDraft.optunaRows = [];
  renderExperimentLab();
}

function experimentRunWithOverride(baseRun, override, index=0){
  const run = JSON.parse(JSON.stringify(baseRun));
  const stage = run.pipeline?.find(s => s.id === override.stage);
  if(stage) {
    stage.params = stage.params || {};
    stage.params[override.param] = override.value;
  }
  run.run_id = override.run_id || `${baseRun.run_id}__set_${String(index + 1).padStart(3, '0')}`;
  run.label = override.label || `${baseRun.label || baseRun.run_id} set ${index + 1}`;
  run.execution = {status: 'planned'};
  run.experiment = Object.assign({
    source: 'experiment_lab',
    mode: 'set',
    index,
    override: {stage: override.stage, stage_id: override.stage_id, param: override.param, value: override.value}
  }, override.experiment || {});
  return run;
}

function experimentManifest(){
  const base = plannedRun();
  base.experiment = Object.assign({}, base.experiment || {}, {source: 'experiment_lab', mode: experimentDraft.mode || 'sweep'});
  if(experimentDraft.mode === 'optuna') {
    const study = {
      id: slugify(`optuna_${pipelineDraft.run_id || pipelineDraft.label}`),
      source: 'experiment_lab',
      mode: 'optuna_plan',
      createdAt: new Date().toISOString(),
      base_pipeline_id: pipelineDraft.template_id || '',
      base_run_id: base.run_id,
      label: `${pipelineDraft.label || base.run_id} Optuna plan`,
      direction: experimentDraft.optuna?.direction || 'maximize',
      objective: experimentDraft.optuna?.objective || 'accepted_control_ready_rois',
      trials: Math.max(1, Number(experimentDraft.optuna?.trials) || 40),
      sampler: experimentDraft.optuna?.sampler || 'tpe',
      pruner: experimentDraft.optuna?.pruner || 'median',
      search_space: experimentDraft.optunaRows || []
    };
    return {
      schema_version: 1,
      dataset_id: datasetId,
      experiment: {source: 'experiment_lab', mode: 'optuna_plan', generatedAt: new Date().toISOString()},
      optimization_studies: [study],
      runs: [Object.assign({}, base, {optimization_study_id: study.id})]
    };
  }
  if(experimentDraft.mode === 'sets' && experimentDraft.setRows.length) {
    return {
      schema_version: 1,
      dataset_id: datasetId,
      experiment: {source: 'experiment_lab', mode: 'sets', generatedAt: new Date().toISOString()},
      runs: experimentDraft.setRows.map((row, index) => experimentRunWithOverride(base, row, index))
    };
  }
  const manifest = plannedManifest();
  manifest.experiment = {source: 'experiment_lab', mode: 'sweep', generatedAt: new Date().toISOString()};
  manifest.runs = manifest.runs.map((run, index) => Object.assign({}, run, {
    experiment: Object.assign({}, run.experiment || {}, {source: 'experiment_lab', mode: 'sweep', index})
  }));
  return manifest;
}

function addExperimentSetRow(){
  const optionValue = document.getElementById('experimentSetParamSelect')?.value || '';
  const opt = experimentParamOptions().find(item => `${item.stage.id}.${item.name}` === optionValue) || experimentParamOptions()[0];
  if(!opt) return;
  const raw = document.getElementById('experimentSetValueInput')?.value;
  const value = raw === '' || raw === undefined ? opt.value : (opt.spec.type === 'number' ? Number(raw) : raw);
  const suffix = Date.now().toString(36);
  experimentDraft.setRows.push({
    run_id: `${pipelineDraft.run_id}__set_${String(experimentDraft.setRows.length + 1).padStart(3, '0')}_${suffix}`,
    label: `${pipelineDraft.label || pipelineDraft.run_id} | ${opt.name}=${value}`,
    stage: opt.stage.id,
    stage_id: stageOp(opt.stage),
    param: opt.name,
    value
  });
  renderExperimentLab();
}

function addOptunaSearchRow(){
  const optionValue = document.getElementById('optunaParamSelect')?.value || '';
  const opt = experimentParamOptions().find(item => `${item.stage.id}.${item.name}` === optionValue) || experimentParamOptions()[0];
  if(!opt) return;
  const min = Number(document.getElementById('optunaMinInput')?.value);
  const max = Number(document.getElementById('optunaMaxInput')?.value);
  experimentDraft.optunaRows.push({
    stage: opt.stage.id,
    stage_id: stageOp(opt.stage),
    param: opt.name,
    type: 'float',
    low: Number.isFinite(min) ? min : Number(opt.spec.min ?? 0),
    high: Number.isFinite(max) ? max : Number(opt.spec.max ?? 1)
  });
  renderExperimentLab();
}

function optunaRowSeedValues(row){
  const low = Number(row.low);
  const high = Number(row.high);
  if(!Number.isFinite(low) || !Number.isFinite(high)) return [];
  const a = Math.min(low, high);
  const b = Math.max(low, high);
  const mid = (a + b) / 2;
  return [...new Set([a, mid, b].map(value => Number(value.toFixed(6))))];
}

function convertOptunaPlanToSweepSeeds(){
  const rows = experimentDraft.optunaRows || [];
  const axes = rows.map(row => ({
    stage: row.stage,
    stage_id: row.stage_id,
    param: row.param,
    values: optunaRowSeedValues(row),
    label: `${row.stage}.${row.param}`
  })).filter(axis => axis.values.length);
  if(!axes.length) {
    setSaveState('add Optuna search-space bounds before converting to sweep seeds', 'bad');
    return;
  }
  setSweepFactors(axes);
  experimentDraft.mode = 'sweep';
  setSaveState(`converted ${axes.length} Optuna parameter${axes.length === 1 ? '' : 's'} to sweep seeds`, 'ok');
  renderExperimentLab();
  renderPipelineBuilder();
}

function duplicateOptunaPlan(){
  const stamp = Date.now().toString(36);
  experimentDraft.optuna = Object.assign({}, experimentDraft.optuna, {
    objective: `${experimentDraft.optuna?.objective || 'objective'}_${stamp}`
  });
  experimentDraft.optunaRows = JSON.parse(JSON.stringify(experimentDraft.optunaRows || []));
  experimentDraft.mode = 'optuna';
  setSaveState('duplicated Optuna plan draft', 'ok');
  renderExperimentLab();
}

function loadExperimentPreset(){
  const preset = document.getElementById('experimentPresetSelect')?.value || 'current_review_pipeline';
  pipelineDraft = makePresetPipeline(preset);
  selectedPipelineStageId = pipelineDraft.pipeline[0]?.id || null;
  experimentDraft.setRows = [];
  renderExperimentLab();
  renderPipelineBuilder();
}

function cloneActiveRunToExperiment(){
  const run = activeRun();
  if(!run) return;
  pipelineDraft = normalizePipelineDraft(JSON.parse(JSON.stringify(Object.assign({}, run, {execution:{status:'planned'}}))));
  pipelineDraft.run_id = `planned_experiment_${Date.now().toString(36)}`;
  pipelineDraft.label = `Experiment from ${run.label || run.run_id}`;
  selectedPipelineStageId = pipelineDraft.pipeline?.[0]?.id || null;
  experimentDraft.setRows = [];
  renderExperimentLab();
  renderPipelineBuilder();
}

async function saveExperimentPlan({activateFirst=false}={}){
  const manifestPatch = experimentManifest();
  const manifest = Object.assign({}, data.architectureRuns || {schema_version: 1, dataset_id: datasetId, runs: []});
  const ids = new Set(manifestPatch.runs.map(r => r.run_id));
  manifest.runs = [...(manifest.runs || []).filter(r => !ids.has(r.run_id)), ...manifestPatch.runs];
  manifest.experiments = manifest.experiments || [];
  manifest.experiments.push({
    id: `experiment_${Date.now().toString(36)}`,
    source: 'experiment_lab',
    mode: experimentDraft.mode || 'sweep',
    createdAt: new Date().toISOString(),
    run_ids: manifestPatch.runs.map(r => r.run_id)
  });
  if(manifestPatch.optimization_studies?.length) {
    const studyIds = new Set(manifestPatch.optimization_studies.map(study => study.id));
    manifest.optimization_studies = [
      ...(manifest.optimization_studies || []).filter(study => !studyIds.has(study.id)),
      ...manifestPatch.optimization_studies
    ];
  }
  await persistArchitectureRuns(manifest, 'saved experiment plan', `${datasetId}_experiment_plan.json`);
  if(activateFirst && manifestPatch.runs?.[0]?.run_id) annotations.settings.activeRunId = manifestPatch.runs[0].run_id;
  renderExperimentLab();
  renderArchitectureLab();
  renderRunSyncControls();
  return manifestPatch;
}

async function generateExperimentPreview(){
  const manifest = await saveExperimentPlan({activateFirst:true});
  if(manifest?.runs?.[0]?.run_id) await selectActiveRun(manifest.runs[0].run_id, {loadReview:false});
  await startGenerationJob({preview:true});
}

function renderWorkflowHome(){
  const root = document.getElementById('workflowHome');
  if(!root) return;
  const s = annotationSummary();
  const run = activeRun() || architectureRuns()[0] || null;
  const primary = nextBestAction();
  const reviewed = `${s.review_progress.reviewed_rois}/${s.review_progress.tuning_ready_targets.reviewed_rois} ROI labels`;
  const workflow = [
    {title:'Inspect Data', href:'#data', status:runHasIntermediates(run) || runGenerated(run) ? 'ready' : 'needs outputs', detail:'Raw video, stage previews, missing-output diagnostics, and artifact context.', cta:'Open Data'},
    {title:'Build Pipelines', href:'#pipelines', status:`${architectureRuns().length} runs`, detail:'Compare runs, inspect stage parameters, and create reusable pipeline stacks.', cta:'Open Pipelines'},
    {title:'Review Neurons', href:'#review', status:reviewed, detail:'Validate candidates, mark artifacts, add missed neurons, and inspect traces.', cta:'Open Review'},
    {title:'Plan Experiment', href:'#experiments', status:'wizard', detail:'Choose a base pipeline, vary parameters, preview runs, and generate locally.', cta:'Open Experiment Lab'}
  ];
  root.innerHTML = `
    <section class="homeLaunchpad">
      <div class="homeIntro">
        <span class="homeEyebrow">Neuron workflow</span>
        <h2>Choose the next review action</h2>
        <p>Inspect data, compare pipelines, validate candidates, or plan a small parameter experiment from one place.</p>
      </div>
      <article class="homeNextAction">
        <span>${escapeHtml(primary.eyebrow)}</span>
        <h3>${escapeHtml(primary.title)}</h3>
        <p>${escapeHtml(primary.detail)}</p>
        <a class="primaryActionButton" href="${escapeHtml(primary.href)}">${escapeHtml(primary.action)}</a>
      </article>
    </section>
    <section class="workflowGrid homeWorkflowGrid">
      ${workflow.map((item, index) => `
        <a class="workflowCard homeWorkflowCard ${index === 0 ? 'primaryWorkflow' : ''}" href="${item.href}">
          <div>
            <span class="workflowStep">${index + 1}</span>
            <h3>${escapeHtml(item.title)}</h3>
            <p>${escapeHtml(item.detail)}</p>
          </div>
          <div class="workflowCardFooter">
            <span class="runStatus">${escapeHtml(item.status)}</span>
            <b>${escapeHtml(item.cta)}</b>
          </div>
        </a>`).join('')}
    </section>
    <details class="homeRunDetails">
      <summary>Run details</summary>
      ${renderRunSummaryCards(run)}
    </details>
    <section class="homeSecondaryLinks">
      <a class="textButton" href="#progress">Check readiness</a>
      <a class="textButton" href="#report">Prepare report</a>
    </section>`;
}

function renderExperimentLab(){
  const root = document.getElementById('experimentLab');
  if(!root) return;
  const params = experimentParamOptions();
  const paramOptions = params.map(item => `<option value="${escapeHtml(item.stage.id + '.' + item.name)}">${escapeHtml(item.stage.id)}.${escapeHtml(item.name)} (${escapeHtml(item.def?.label || stageOp(item.stage))})</option>`).join('');
  const baseOptions = experimentBaseOptions();
  const baseValue = experimentDraft.baseTemplateId ? `template:${experimentDraft.baseTemplateId}` : '';
  const manifest = experimentManifest();
  const validation = validatePipeline(pipelineDraft);
  const previewRows = manifest.runs.slice(0, 24).map(run => {
    const sweepChanged = run.sweep?.parameters?.map(p => `${p.stage}.${p.param}=${p.value}`).join(', ');
    const override = run.experiment?.override;
    const changed = sweepChanged || (override ? `${override.stage || ''}.${override.param || ''}=${override.value ?? ''}` : 'base stack');
    const readiness = runReadiness(run);
    return `<tr><td>${escapeHtml(run.label || run.run_id)}</td><td>${escapeHtml(run.run_id)}</td><td>${escapeHtml(changed)}</td><td><span class="stageStatus ${readiness.className}">${escapeHtml(run.validation?.status || validation.status)}</span></td><td>${escapeHtml(readiness.text)}</td></tr>`;
  }).join('');
  const setRows = experimentDraft.setRows.map((row, index) => `
    <tr>
      <td><input data-experiment-set-label="${index}" value="${escapeHtml(row.label)}"></td>
      <td>${escapeHtml(row.stage)}.${escapeHtml(row.param)}</td>
      <td><input data-experiment-set-value="${index}" value="${escapeHtml(row.value)}"></td>
      <td><button type="button" data-remove-experiment-set="${index}">Remove</button></td>
    </tr>`).join('');
  const optunaRows = (experimentDraft.optunaRows || []).map((row, index) => `
    <tr>
      <td>${escapeHtml(row.stage)}.${escapeHtml(row.param)}</td>
      <td>${escapeHtml(row.type || 'float')}</td>
      <td><input data-optuna-low="${index}" value="${escapeHtml(row.low)}"></td>
      <td><input data-optuna-high="${index}" value="${escapeHtml(row.high)}"></td>
      <td><button type="button" data-remove-optuna-row="${index}">Remove</button></td>
    </tr>`).join('');
  root.innerHTML = `
    <section class="experimentHero experimentWizardHero">
      <div>
        <span class="homeEyebrow">Experiment wizard</span>
        <h2>Plan one concrete parameter test</h2>
        <p class="hint">Choose a base pipeline, pick how parameters should vary, preview the planned runs, then save or generate the first preview locally.</p>
      </div>
      <div class="buttonRow">
        <a class="textButton" href="#pipelines">Edit stack</a>
        <button type="button" id="experimentSaveBtn">Save Plan</button>
        <button type="button" id="experimentDownloadBtn">Download Plan</button>
        <button type="button" id="experimentPreviewBtn">Generate First Preview</button>
      </div>
    </section>
    <section class="experimentWizard">
      <article class="archCard wizardStep">
        <div class="runCardHeader"><h3>1. Choose base pipeline</h3><span class="runStatus">${validation.status}</span></div>
        <label>Model instance
          <select id="experimentBaseSelect">
            <option value="">Current Build Pipeline stack</option>
            ${baseOptions.map(item => `<option value="${escapeHtml(item.kind + ':' + item.id)}" ${baseValue === item.kind + ':' + item.id ? 'selected' : ''}>${escapeHtml(item.label)}</option>`).join('')}
          </select>
        </label>
        <label>Preset
          <select id="experimentPresetSelect">
            ${ARCHITECTURE_PRESETS.map(p => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.label)}</option>`).join('')}
          </select>
        </label>
        <div class="buttonRow">
          <button type="button" id="experimentLoadPresetBtn">Load Preset</button>
          <button type="button" id="experimentCloneActiveBtn">Clone Active Run</button>
        </div>
        <div class="archEvidence">${(pipelineDraft.pipeline || []).map(stage => `<span>${escapeHtml(stageDef(stage)?.label || stageOp(stage))}</span>`).join('')}</div>
        ${validation.errors.map(e => `<div class="qcWarning">${escapeHtml(e)}</div>`).join('')}
        ${validation.warnings.map(w => `<div class="pipelineWarning">${escapeHtml(w)}</div>`).join('')}
      </article>
      <article class="archCard wizardStep">
        <div class="runCardHeader"><h3>2. Choose strategy</h3><span class="runStatus">${manifest.runs.length} planned</span></div>
        <label>Mode
          <select id="experimentModeSelect">
            <option value="sweep" ${experimentDraft.mode === 'sweep' ? 'selected' : ''}>Sweep axes</option>
            <option value="sets" ${experimentDraft.mode === 'sets' ? 'selected' : ''}>Named sets</option>
            <option value="optuna" ${experimentDraft.mode === 'optuna' ? 'selected' : ''}>Optuna plan</option>
          </select>
        </label>
        <p class="hint">Sweep mode uses stack axes. Set mode saves named variants. Optuna plan mode stores study/search-space metadata only; it does not run optimization in the browser.</p>
      </article>
      <article class="archCard wizardStep">
        <div class="runCardHeader"><h3>3. Add named parameter sets</h3><span class="runStatus">${experimentDraft.setRows.length} sets</span></div>
        <label>Parameter <select id="experimentSetParamSelect">${paramOptions}</select></label>
        <label>Value <input id="experimentSetValueInput" placeholder="new value"></label>
        <button type="button" id="experimentAddSetBtn" ${params.length ? '' : 'disabled'}>Add Set</button>
        <table class="smallTable"><tr><th>Label</th><th>Parameter</th><th>Value</th><th></th></tr>${setRows || '<tr><td colspan="4">No named sets yet.</td></tr>'}</table>
      </article>
      <article class="archCard wizardStep">
        <div class="runCardHeader"><h3>4. Preview planned runs</h3><span class="runStatus">${manifest.runs.length} run${manifest.runs.length === 1 ? '' : 's'}</span></div>
      <table class="smallTable compareTable">
        <tr><th>Label</th><th>Run ID</th><th>Changed parameters</th><th>Status</th><th>Readiness</th></tr>
        ${previewRows || '<tr><td colspan="5">No planned runs.</td></tr>'}
      </table>
      ${manifest.runs.length > 24 ? `<p class="hint">Showing first 24 of ${manifest.runs.length} planned runs.</p>` : ''}
      </article>
    </section>
    <details class="archCard experimentAdvancedPlan">
      <summary>Optuna and advanced planner options</summary>
      <div class="experimentGrid compact">
        <section>
          <div class="runCardHeader"><h3>Optuna Plan</h3><span class="runStatus">${(experimentDraft.optunaRows || []).length} params</span></div>
          <label>Study direction
            <select id="optunaDirectionSelect">
              <option value="maximize" ${experimentDraft.optuna.direction === 'maximize' ? 'selected' : ''}>Maximize</option>
              <option value="minimize" ${experimentDraft.optuna.direction === 'minimize' ? 'selected' : ''}>Minimize</option>
            </select>
          </label>
          <label>Objective <input id="optunaObjectiveInput" value="${escapeHtml(experimentDraft.optuna.objective || '')}"></label>
          <label>Trial budget <input id="optunaTrialsInput" type="number" min="1" value="${escapeHtml(experimentDraft.optuna.trials || 40)}"></label>
          <label>Parameter <select id="optunaParamSelect">${paramOptions}</select></label>
          <label>Low <input id="optunaMinInput" placeholder="min"></label>
          <label>High <input id="optunaMaxInput" placeholder="max"></label>
          <button type="button" id="optunaAddParamBtn" ${params.length ? '' : 'disabled'}>Add Optuna Parameter</button>
          <div class="buttonRow">
            <button type="button" id="optunaDuplicateBtn">Duplicate Optuna Plan</button>
            <button type="button" id="optunaConvertSweepBtn">Convert To Sweep Seeds</button>
          </div>
        </section>
        <section>
          <table class="smallTable"><tr><th>Parameter</th><th>Type</th><th>Low</th><th>High</th><th></th></tr>${optunaRows || '<tr><td colspan="5">No Optuna search space yet.</td></tr>'}</table>
        </section>
      </div>
    </details>
    <details class="archCard experimentAdvisorPanel expertOnly">
      <summary>Advisor, LLM handoff, and diagnostics</summary>
      ${renderRunSummaryCards(activeRun() || experimentBaselineRun())}
      ${renderExperimentCommandCenter(manifest, validation)}
    </details>
    <section class="archCard experimentManifestShell expertOnly">
      <details>
        <summary>Experiment Manifest JSON</summary>
        <pre id="experimentManifestPreview">${escapeHtml(JSON.stringify(manifest, null, 2))}</pre>
      </details>
    </section>`;
  document.getElementById('experimentBaseSelect').onchange = e => applyExperimentBase(e.target.value);
  document.getElementById('experimentModeSelect').onchange = e => { experimentDraft.mode = e.target.value; renderExperimentLab(); };
  document.getElementById('experimentLoadPresetBtn').onclick = loadExperimentPreset;
  document.getElementById('experimentCloneActiveBtn').onclick = cloneActiveRunToExperiment;
  document.getElementById('experimentAddSetBtn').onclick = addExperimentSetRow;
  document.getElementById('experimentSaveBtn').onclick = () => saveExperimentPlan();
  document.getElementById('experimentDownloadBtn').onclick = () => downloadJson(`${datasetId}_experiment_plan.json`, experimentManifest());
  document.getElementById('experimentPreviewBtn').onclick = generateExperimentPreview;
  document.getElementById('optunaDirectionSelect').onchange = e => { experimentDraft.optuna.direction = e.target.value; renderExperimentLab(); };
  document.getElementById('optunaObjectiveInput').onchange = e => { experimentDraft.optuna.objective = e.target.value; renderExperimentLab(); };
  document.getElementById('optunaTrialsInput').onchange = e => { experimentDraft.optuna.trials = Math.max(1, Number(e.target.value) || 40); renderExperimentLab(); };
  document.getElementById('optunaAddParamBtn').onclick = addOptunaSearchRow;
  document.getElementById('optunaDuplicateBtn').onclick = duplicateOptunaPlan;
  document.getElementById('optunaConvertSweepBtn').onclick = convertOptunaPlanToSweepSeeds;
  const presetSelect = document.getElementById('experimentPresetSelect');
  const matchingPreset = ARCHITECTURE_PRESETS.find(p => pipelineDraft.label?.toLowerCase().includes(p.label.toLowerCase().split(' ')[0]));
  if(presetSelect && matchingPreset) presetSelect.value = matchingPreset.id;
  for(const input of root.querySelectorAll('[data-experiment-set-label]')) input.onchange = e => {
    const row = experimentDraft.setRows[Number(e.target.dataset.experimentSetLabel)];
    if(row) row.label = e.target.value;
    renderExperimentLab();
  };
  for(const input of root.querySelectorAll('[data-experiment-set-value]')) input.onchange = e => {
    const row = experimentDraft.setRows[Number(e.target.dataset.experimentSetValue)];
    if(row) {
      const opt = params.find(item => item.stage.id === row.stage && item.name === row.param);
      row.value = opt?.spec?.type === 'number' && Number.isFinite(Number(e.target.value)) ? Number(e.target.value) : e.target.value;
    }
    renderExperimentLab();
  };
  for(const btn of root.querySelectorAll('[data-remove-experiment-set]')) btn.onclick = () => {
    experimentDraft.setRows.splice(Number(btn.dataset.removeExperimentSet), 1);
    renderExperimentLab();
  };
  for(const input of root.querySelectorAll('[data-optuna-low]')) input.onchange = e => {
    const row = experimentDraft.optunaRows[Number(e.target.dataset.optunaLow)];
    if(row && Number.isFinite(Number(e.target.value))) row.low = Number(e.target.value);
    renderExperimentLab();
  };
  for(const input of root.querySelectorAll('[data-optuna-high]')) input.onchange = e => {
    const row = experimentDraft.optunaRows[Number(e.target.dataset.optunaHigh)];
    if(row && Number.isFinite(Number(e.target.value))) row.high = Number(e.target.value);
    renderExperimentLab();
  };
  for(const btn of root.querySelectorAll('[data-remove-optuna-row]')) btn.onclick = () => {
    experimentDraft.optunaRows.splice(Number(btn.dataset.removeOptunaRow), 1);
    renderExperimentLab();
  };
  for(const btn of root.querySelectorAll('[data-experiment-use-run]')) btn.onclick = () => selectActiveRun(btn.dataset.experimentUseRun, {loadReview:false});
  for(const btn of root.querySelectorAll('[data-experiment-load-run]')) btn.onclick = () => selectActiveRun(btn.dataset.experimentLoadRun, {loadReview:true});
  for(const btn of root.querySelectorAll('[data-load-llm-template]')) btn.onclick = () => {
    loadTemplateIntoBuilder(btn.dataset.loadLlmTemplate);
    experimentDraft.baseTemplateId = btn.dataset.loadLlmTemplate;
    renderExperimentLab();
  };
  for(const btn of root.querySelectorAll('[data-recommend-preset]')) btn.onclick = () => {
    pipelineDraft = makePresetPipeline(btn.dataset.recommendPreset);
    selectedPipelineStageId = pipelineDraft.pipeline?.[0]?.id || null;
    location.hash = '#pipelines';
    renderPipelineBuilder();
  };
  for(const btn of root.querySelectorAll('[data-recommend-experiment]')) btn.onclick = () => {
    pipelineDraft = makePresetPipeline(btn.dataset.recommendExperiment);
    selectedPipelineStageId = pipelineDraft.pipeline?.[0]?.id || null;
    experimentDraft.baseTemplateId = '';
    renderExperimentLab();
    renderPipelineBuilder();
  };
  for(const btn of root.querySelectorAll('[data-recommend-generate]')) btn.onclick = generateExperimentPreview;
  document.getElementById('experimentPanelMode')?.addEventListener('change', e => setExperimentPanelMode(e.target.value));
  document.getElementById('experimentObjectiveInput')?.addEventListener('change', e => setExperimentObjective(e.target.value));
  document.getElementById('experimentRunTopRecipeActionBtn')?.addEventListener('click', () => {
    const action = experimentSessionRecipe(manifest, activeRun() || active).next_action;
    if(action?.id) runExperimentAction(action.id);
  });
  document.getElementById('experimentCopyRecipeBtn')?.addEventListener('click', () => {
    copyTextToClipboard(experimentSessionRecipeMarkdown(experimentSessionRecipe(manifest, activeRun() || active)), 'copied session recipe');
  });
  document.getElementById('experimentDownloadRecipeBtn')?.addEventListener('click', () => {
    downloadText(`${datasetId}_session_recipe.md`, experimentSessionRecipeMarkdown(experimentSessionRecipe(manifest, activeRun() || active)), 'text/markdown');
  });
  document.getElementById('experimentDownloadRecipeJsonBtn')?.addEventListener('click', () => {
    downloadJson(`${datasetId}_session_recipe.json`, experimentSessionRecipe(manifest, activeRun() || active));
  });
  document.getElementById('experimentLlmPromptMode')?.addEventListener('change', e => setExperimentLlmPromptMode(e.target.value));
  document.getElementById('experimentLlmConstraintsInput')?.addEventListener('change', e => setExperimentLlmConstraints(e.target.value));
  document.getElementById('experimentCopyLlmPromptBtn')?.addEventListener('click', () => {
    copyTextToClipboard(experimentLlmPromptMarkdown(experimentLlmPromptSpec(manifest, activeRun() || active)), 'copied LLM architecture prompt');
  });
  document.getElementById('experimentDownloadLlmPromptBtn')?.addEventListener('click', () => {
    downloadText(`${datasetId}_llm_architecture_prompt.md`, experimentLlmPromptMarkdown(experimentLlmPromptSpec(manifest, activeRun() || active)), 'text/markdown');
  });
  document.getElementById('experimentDownloadLlmPromptJsonBtn')?.addEventListener('click', () => {
    downloadJson(`${datasetId}_llm_architecture_prompt.json`, experimentLlmPromptSpec(manifest, activeRun() || active));
  });
  document.getElementById('experimentLlmResponseInput')?.addEventListener('change', e => setExperimentLlmResponseText(e.target.value));
  document.getElementById('experimentValidateLlmResponseBtn')?.addEventListener('click', () => {
    const input = document.getElementById('experimentLlmResponseInput');
    setExperimentLlmResponseText(input?.value || '');
  });
  document.getElementById('experimentCopyImportCommandBtn')?.addEventListener('click', () => {
    const intake = experimentLlmProposalIntake();
    const readiness = experimentLlmImportReadiness(intake);
    copyTextToClipboard(readiness.recommended === 'candidate' ? intake.import_commands.candidate : intake.import_commands.full, 'copied recommended LLM import command');
  });
  document.getElementById('experimentCopyFullImportCommandBtn')?.addEventListener('click', () => {
    copyTextToClipboard(experimentLlmImportCommands().full, 'copied full proposal import command');
  });
  document.getElementById('experimentCopyCandidateImportCommandBtn')?.addEventListener('click', () => {
    copyTextToClipboard(experimentLlmImportCommands().candidate, 'copied candidate proposal import command');
  });
  document.getElementById('experimentServerImportFullBtn')?.addEventListener('click', () => {
    importLlmProposalSetViaServer({candidateOnly:false});
  });
  document.getElementById('experimentServerImportCandidateBtn')?.addEventListener('click', () => {
    importLlmProposalSetViaServer({candidateOnly:true});
  });
  document.getElementById('experimentCopyPostImportPlanBtn')?.addEventListener('click', () => {
    copyTextToClipboard(experimentLlmPostImportMarkdown(experimentLlmPostImportPlan(experimentLlmProposalIntake())), 'copied post-import plan');
  });
  document.getElementById('experimentDownloadPostImportPlanBtn')?.addEventListener('click', () => {
    downloadText(`${datasetId}_llm_post_import_plan.md`, experimentLlmPostImportMarkdown(experimentLlmPostImportPlan(experimentLlmProposalIntake())), 'text/markdown');
  });
  document.getElementById('experimentCopyExecutionCommandBtn')?.addEventListener('click', () => {
    copyTextToClipboard(experimentLlmPostImportPlan(experimentLlmProposalIntake()).execution_command, 'copied local experiment command');
  });
  document.getElementById('experimentCopyRepairPromptBtn')?.addEventListener('click', () => {
    copyTextToClipboard(experimentLlmRepairPromptMarkdown(experimentLlmProposalIntake()), 'copied LLM proposal repair prompt');
  });
  document.getElementById('experimentDownloadRepairPromptBtn')?.addEventListener('click', () => {
    downloadText(`${datasetId}_llm_proposal_repair_prompt.md`, experimentLlmRepairPromptMarkdown(experimentLlmProposalIntake()), 'text/markdown');
  });
  document.getElementById('experimentDownloadProposalTriageBtn')?.addEventListener('click', () => {
    downloadText(`${datasetId}_llm_proposal_triage.md`, experimentLlmProposalTriageMarkdown(experimentLlmProposalIntake()), 'text/markdown');
  });
  document.getElementById('experimentDownloadCandidatePackBtn')?.addEventListener('click', () => {
    downloadJson(`${datasetId}_llm_candidate_proposals.json`, experimentLlmCandidateProposalPack(experimentLlmProposalIntake()));
  });
  document.getElementById('experimentDownloadCandidatePackNoteBtn')?.addEventListener('click', () => {
    downloadText(`${datasetId}_llm_candidate_pack_note.md`, experimentLlmCandidatePackMarkdown(experimentLlmCandidateProposalPack(experimentLlmProposalIntake())), 'text/markdown');
  });
  document.getElementById('experimentDownloadParsedProposalBtn')?.addEventListener('click', () => {
    const intake = experimentLlmProposalIntake();
    if(intake.parsed && !intake.errors.length) downloadJson(`${datasetId}_llm_proposals.json`, intake.parsed);
  });
  for(const select of root.querySelectorAll('[data-proposal-state]')) select.onchange = e => {
    const [proposalSetId, proposalId] = String(e.target.dataset.proposalState || '').split('::');
    setExperimentProposalState(proposalSetId, proposalId, e.target.value);
  };
  document.getElementById('experimentCopyFollowUpPromptBtn')?.addEventListener('click', () => {
    copyTextToClipboard(experimentLlmFollowUpPromptMarkdown(experimentProposalLifecycleRows()), 'copied proposal follow-up prompt');
  });
  document.getElementById('experimentDownloadFollowUpPromptBtn')?.addEventListener('click', () => {
    downloadText(`${datasetId}_llm_followup_prompt.md`, experimentLlmFollowUpPromptMarkdown(experimentProposalLifecycleRows()), 'text/markdown');
  });
  document.getElementById('experimentDecisionFilter')?.addEventListener('change', e => setExperimentDecisionFilter(e.target.value));
  for(const select of root.querySelectorAll('[data-decision-label]')) select.onchange = e => setExperimentLabel(e.target.dataset.decisionLabel, e.target.value);
  for(const btn of root.querySelectorAll('[data-decision-use-run]')) btn.onclick = () => selectActiveRun(btn.dataset.decisionUseRun, {loadReview:false});
  for(const btn of root.querySelectorAll('[data-decision-load-run]')) btn.onclick = () => selectActiveRun(btn.dataset.decisionLoadRun, {loadReview:true});
  document.getElementById('experimentDownloadDecisionMatrixBtn')?.addEventListener('click', () => {
    downloadText(`${datasetId}_experiment_decision_matrix.tsv`, experimentDecisionMatrixTsv(), 'text/tab-separated-values');
  });
  document.getElementById('experimentDownloadSensitivityBtn')?.addEventListener('click', () => {
    downloadText(`${datasetId}_experiment_sensitivity.tsv`, experimentSensitivityTsv(), 'text/tab-separated-values');
  });
  for(const btn of root.querySelectorAll('[data-sensitivity-use-run]')) btn.onclick = () => selectActiveRun(btn.dataset.sensitivityUseRun, {loadReview:false});
  document.getElementById('experimentDownloadFollowUpBtn')?.addEventListener('click', () => {
    downloadText(`${datasetId}_experiment_followups.tsv`, experimentFollowUpTsv(), 'text/tab-separated-values');
  });
  for(const btn of root.querySelectorAll('[data-followup-use-run]')) btn.onclick = () => selectActiveRun(btn.dataset.followupUseRun, {loadReview:false});
  for(const btn of root.querySelectorAll('[data-followup-add-sets]')) btn.onclick = () => applyFollowUpSuggestion(btn.dataset.followupAddSets, 'sets');
  for(const btn of root.querySelectorAll('[data-followup-add-sweep]')) btn.onclick = () => applyFollowUpSuggestion(btn.dataset.followupAddSweep, 'sweep');
  document.getElementById('experimentDownloadCoverageBtn')?.addEventListener('click', () => {
    downloadText(`${datasetId}_experiment_parameter_coverage.tsv`, experimentCoverageTsv(), 'text/tab-separated-values');
  });
  for(const btn of root.querySelectorAll('[data-coverage-add-sets]')) btn.onclick = () => applyCoverageProbe(btn.dataset.coverageAddSets, 'sets');
  for(const btn of root.querySelectorAll('[data-coverage-add-sweep]')) btn.onclick = () => applyCoverageProbe(btn.dataset.coverageAddSweep, 'sweep');
  document.getElementById('experimentActionFilter')?.addEventListener('change', e => setExperimentActionFilter(e.target.value));
  document.getElementById('experimentDownloadActionQueueBtn')?.addEventListener('click', () => {
    downloadText(`${datasetId}_experiment_action_queue.tsv`, experimentActionQueueTsv(), 'text/tab-separated-values');
  });
  document.getElementById('experimentDownloadActionHistoryBtn')?.addEventListener('click', () => {
    downloadText(`${datasetId}_experiment_action_history.tsv`, experimentActionHistoryTsv(), 'text/tab-separated-values');
  });
  for(const btn of root.querySelectorAll('[data-action-queue-run]')) btn.onclick = () => runExperimentAction(btn.dataset.actionQueueRun);
  for(const btn of root.querySelectorAll('[data-action-state]')) btn.onclick = () => setExperimentActionState(btn.dataset.actionState, btn.dataset.actionStateValue);
  for(const btn of root.querySelectorAll('[data-checklist-generate]')) btn.onclick = generateExperimentPreview;
  for(const btn of root.querySelectorAll('[data-checklist-shortlist]')) btn.onclick = () => {
    const run = activeRun();
    if(run) setExperimentLabel(run.run_id, 'shortlist');
  };
  const noteInput = document.getElementById('experimentActiveRunNote');
  if(noteInput) noteInput.onchange = e => {
    const run = activeRun();
    if(run) {
      setExperimentNote(run.run_id, e.target.value);
      renderExperimentLab();
    }
  };
  document.getElementById('experimentShortlistActiveBtn')?.addEventListener('click', () => {
    const run = activeRun();
    if(run) setExperimentLabel(run.run_id, 'shortlist');
  });
  document.getElementById('experimentDownloadBriefBtn')?.addEventListener('click', () => {
    downloadText(`${datasetId}_experiment_brief.md`, experimentBriefMarkdown(experimentManifest(), activeRun()), 'text/markdown');
  });
  document.getElementById('experimentDownloadHandoffBtn')?.addEventListener('click', () => {
    downloadJson(`${datasetId}_llm_handoff_context.json`, experimentHandoffContext(experimentManifest(), activeRun()));
  });
  for(const btn of root.querySelectorAll('[data-load-shortlist-run]')) btn.onclick = () => selectActiveRun(btn.dataset.loadShortlistRun, {loadReview:false});
}

const FALLBACK_STAGE_CATALOG = [
  {type:'temporal_smoothing', op:'temporal_highpass_gaussian', label:'Temporal high-pass Gaussian', input:'raw_video', output:'highpass_video', params:{sigma_frames:{type:'number', min:1, max:20, step:1, default:6}}},
  {type:'spatial_smoothing', op:'spatial_gaussian', label:'Spatial Gaussian smoothing', input:'highpass_video', output:'smoothed_video', params:{sigma_px:{type:'number', min:0, max:4, step:0.1, default:0.8}}},
  {type:'background_correction', op:'local_background_ring', label:'Local background ring', input:'roi_candidates', output:'roi_traces', params:{outer_radius_px:{type:'number', min:4, max:40, step:1, default:15}, neuropil_weight:{type:'number', min:0, max:1.5, step:0.05, default:0.7}}},
  {type:'motion_correction', op:'rigid_shift_estimate', label:'Rigid drift estimate', input:'raw_video', output:'registered_video', params:{max_shift_px:{type:'number', min:1, max:12, step:1, default:4}}},
  {type:'filtering', op:'robust_positive_local_z', label:'Robust positive local-z', input:'highpass_video', output:'z_stack', params:{local_radius_px:{type:'number', min:3, max:31, step:2, default:11}, epsilon:{type:'number', min:0, max:10, step:0.5, default:1}}},
  {type:'filtering', op:'gamma_cfar', label:'Gamma CFAR', input:'smoothed_video', output:'candidate_mask', params:{pfa:{type:'number', min:0.000001, max:0.1, step:0.0001, default:0.001}, guard_px:{type:'number', min:0, max:12, step:1, default:2}}},
  {type:'trace_extraction', op:'component_filter', label:'Component extraction', input:'z_stack', output:'roi_candidates', params:{seed_z:{type:'number', min:0.5, max:8, step:0.1, default:2.0}, grow_z:{type:'number', min:0.2, max:5, step:0.1, default:1.1}, min_area_px:{type:'number', min:1, max:100, step:1, default:4}, max_area_px:{type:'number', min:20, max:800, step:10, default:260}}},
  {type:'event_model', op:'robust_kalman_positive_innovation', label:'Kalman positive innovation events', input:'roi_traces', output:'candidate_events', params:{event_threshold_z:{type:'number', min:0.5, max:8, step:0.1, default:2.4}, kalman_gain:{type:'number', min:0.001, max:0.3, step:0.005, default:0.06}, spike_gain:{type:'number', min:0, max:0.08, step:0.002, default:0.008}}},
  {type:'event_model', op:'oasis_deconvolution_import', label:'OASIS trace import', input:'roi_traces', output:'deconvolved_events', params:{array_key:{type:'text', default:'spikes'}}},
  {type:'candidate_ranking', op:'heuristic_priority_v1', label:'Heuristic priority ranking', input:'roi_candidates', output:'ranked_candidates', params:{local_correlation_weight:{type:'number', min:-1, max:1, step:0.05, default:0.2}, event_support_weight:{type:'number', min:-1, max:1, step:0.05, default:0.2}, artifact_weight:{type:'number', min:-1, max:1, step:0.05, default:-0.15}}},
  {type:'import', op:'pmd_denoised_video_import', label:'PMD denoised video import', input:'raw_video', output:'highpass_video', params:{denoised_video:{type:'text', default:''}}},
  {type:'import', op:'suite2p_import', label:'Suite2p import', input:'raw_video', output:'roi_candidates', params:{suite2p_dir:{type:'text', default:''}}}
];

function paramStep(name, spec){
  if(name === 'pfa') return 0.0001;
  if(name.includes('weight') || name.includes('gain')) return 0.01;
  if(name.includes('threshold') || name.endsWith('_z') || name.includes('sigma')) return 0.1;
  return 1;
}

function catalogParamSpec(name, stage){
  const docs = stage.parameter_docs?.[name] || {};
  const ranges = stage.param_ranges?.[name] || docs.range || {};
  const defaultValue = Object.prototype.hasOwnProperty.call(stage.default_params || {}, name) ? stage.default_params[name] : '';
  const isNumber = ranges.minimum !== undefined || ranges.maximum !== undefined || typeof defaultValue === 'number';
  return {
    type: isNumber ? 'number' : 'text',
    min: ranges.minimum,
    max: ranges.maximum,
    step: isNumber ? paramStep(name, docs) : undefined,
    default: defaultValue ?? '',
    doc: docs.meaning || '',
    why: docs.why || '',
    required: Boolean(docs.required || (stage.required_params || []).includes(name))
  };
}
