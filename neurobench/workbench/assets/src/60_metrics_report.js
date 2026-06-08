function renderRunComparison(){
  const root = document.getElementById('runComparison');
  if(!root) return;
  const runs = data.architectureRuns?.runs || [];
  if(runs.length < 2){
    root.innerHTML = '<p class="hint">Add a second architecture run manifest to compare methods side-by-side. The current run is still shown below.</p>';
    return;
  }
  const a = runs.find(r => r.run_id === document.getElementById('archRunA').value) || runs[0];
  const b = runs.find(r => r.run_id === document.getElementById('archRunB').value) || runs[1];
  const rows = [
    ['Candidate ROIs', 'summary.roi_count', false],
    ['Candidate events', 'summary.event_count', false],
    ['Discovery suggestions', 'summary.suggestion_count', false],
    ['Accepted ROIs', 'annotation_summary.roi_states.accepted', true],
    ['Rejected ROIs', 'annotation_summary.roi_states.rejected', false],
    ['Control-ready ROIs', 'annotation_summary.control_ready.yes', true],
    ['Accepted events', 'annotation_summary.event_states.accepted', true],
    ['Candidates per accepted ROI', 'annotation_summary.review_burden.candidate_rois_per_accepted_roi', false],
    ['Events per accepted event', 'annotation_summary.review_burden.candidate_events_per_accepted_event', false]
  ];
  let html = `<table class="smallTable compareTable"><tr><th>Metric</th><th>${runLabel(a)}</th><th>${runLabel(b)}</th><th>Delta B-A</th></tr>`;
  for(const [label, path, higherGood] of rows){
    const av = Number(runMetric(a, path, 0));
    const bv = Number(runMetric(b, path, 0));
    const delta = bv - av;
    const cls = Math.abs(delta) < 1e-9 ? 'deltaNeutral' : (higherGood ? delta > 0 : delta < 0) ? 'deltaGood' : 'deltaBad';
    html += `<tr><td>${label}</td><td>${fmt(av, Number.isInteger(av) ? 0 : 2)}</td><td>${fmt(bv, Number.isInteger(bv) ? 0 : 2)}</td><td class="${cls}">${delta >= 0 ? '+' : ''}${fmt(delta, Number.isInteger(delta) ? 0 : 2)}</td></tr>`;
  }
  root.innerHTML = html + `</table>
    <section class="abReviewShell">
      <div class="runCardHeader">
        <h3>Synchronized A/B Review</h3>
        <span id="reviewComparisonStatus" class="hint">not loaded</span>
      </div>
      <p class="hint">Compare generated review frames side-by-side at the same frame. This viewer is read-only; use the explicit buttons below to switch the main Review/Data context.</p>
      <div class="buttonRow">
        <button type="button" id="loadReviewComparisonBtn" ${runGenerated(a) && runGenerated(b) ? '' : 'disabled'}>Load A/B Review</button>
        <button type="button" id="prevReviewDiffBtn">Prev Difference</button>
        <button type="button" id="nextReviewDiffBtn">Next Difference</button>
        <button type="button" id="useRunAReviewBtn">Use A In Review/Data</button>
        <button type="button" id="useRunBReviewBtn">Use B In Review/Data</button>
      </div>
      <div id="reviewComparisonViewer"></div>
    </section>`;
  const compare = reviewCompareSettings();
  compare.runAId = a.run_id;
  compare.runBId = b.run_id;
  document.getElementById('loadReviewComparisonBtn').onclick = loadReviewComparison;
  document.getElementById('prevReviewDiffBtn').onclick = () => nextReviewComparisonDifference(-1);
  document.getElementById('nextReviewDiffBtn').onclick = () => nextReviewComparisonDifference(1);
  document.getElementById('useRunAReviewBtn').onclick = () => selectActiveRun(a.run_id, {loadReview:true});
  document.getElementById('useRunBReviewBtn').onclick = () => selectActiveRun(b.run_id, {loadReview:true});
  renderReviewComparisonViewer();
}

function annotationSummary(){
  const roiStates = {accepted:0, rejected:0, unsure:0, unlabeled:0};
  const eventStates = {accepted:0, rejected:0, unsure:0, unlabeled:0};
  const suggestionStates = {promoted:0, missed:0, artifact:0, unsure:0, unlabeled:0};
  const traceQuality = {good:0, weak:0, noisy:0, unusable:0, unlabeled:0};
  const controlReady = {yes:0, maybe:0, no:0, unlabeled:0};
  const triageQueues = {strong_neuron:0, possible_missed_neuron:0, artifact_like:0, merged_cluster:0, weak_trace:0, needs_event_review:0, standard_review:0};
  const reviewerCounts = {};
  const reviewerMissing = {rois:0, virtual_rois:0, events:0, suggestions:0, split_merge_decisions:0};
  const bumpReviewer = ann => {
    const reviewer = String(ann?.reviewer_id || '').trim() || 'unassigned';
    reviewerCounts[reviewer] = (reviewerCounts[reviewer] || 0) + 1;
  };
  for(const roi of data.rois){
    const ann = roiAnn(roi.id);
    const rs = ann.cell_state || (ann.state === 'accept' ? 'accepted' : ann.state === 'reject' ? 'rejected' : ann.state === 'unsure' ? 'unsure' : 'unlabeled');
    roiStates[roiStates[rs] === undefined ? 'unlabeled' : rs]++;
    if(rs !== 'unlabeled') {
      bumpReviewer(ann);
      if(!roiReviewerId(roi)) reviewerMissing.rois++;
    }
    const tq = ann.trace_quality || 'unlabeled';
    traceQuality[traceQuality[tq] === undefined ? 'unlabeled' : tq]++;
    const cr = ann.control_ready || 'unlabeled';
    controlReady[controlReady[cr] === undefined ? 'unlabeled' : cr]++;
    triageQueues[roiTriageCategory(roi)]++;
    for(const ev of eventsForRoi(roi)){
      const eann = eventAnn(roi.id, ev.frame);
      const es = eann.event_state || (eann.state === 'accept' ? 'accepted' : eann.state === 'reject' ? 'rejected' : eann.state === 'unsure' ? 'unsure' : 'unlabeled');
      eventStates[eventStates[es] === undefined ? 'unlabeled' : es]++;
      if(es !== 'unlabeled') {
        bumpReviewer(eann);
        if(!String(eann.reviewer_id || '').trim()) reviewerMissing.events++;
      }
    }
  }
  for(const virtual of Object.values(annotations.virtualRois || {})){
    const rs = virtual.cell_state || (virtual.state === 'accept' ? 'accepted' : virtual.state === 'reject' ? 'rejected' : virtual.state === 'unsure' ? 'unsure' : '');
    if(rs) {
      bumpReviewer(virtual);
      if(!String(virtual.reviewer_id || '').trim()) reviewerMissing.virtual_rois++;
    }
  }
  for(const s of data.discovery?.suggestions || []){
    const ann = suggestionAnn(s.id);
    const ss = annotations.promotedRois[s.id] ? 'promoted' : ann.state || 'unlabeled';
    suggestionStates[suggestionStates[ss] === undefined ? 'unlabeled' : ss]++;
    if(ss !== 'unlabeled') {
      bumpReviewer(ann);
      if(!String(ann.reviewer_id || '').trim()) reviewerMissing.suggestions++;
    }
    if(ss === 'promoted' || ss === 'missed') triageQueues.possible_missed_neuron++;
    if(ss === 'artifact' || (ann.artifact_class || ann.artifactClass) || (s.artifactCue && s.artifactCue !== 'none') || scoreValue(s, 'artifactScore') >= 0.4) triageQueues.artifact_like++;
  }
  for(const decision of Object.values(annotations.splitMergeDecisions || {})){
    if(decision.decision_state) {
      bumpReviewer(decision);
      if(!String(decision.reviewer_id || '').trim()) reviewerMissing.split_merge_decisions++;
    }
  }
  const eventCount = Object.values(eventStates).reduce((a,b) => a+b, 0);
  const reviewedRois = roiStates.accepted + roiStates.rejected + roiStates.unsure;
  const reviewedEvents = eventStates.accepted + eventStates.rejected + eventStates.unsure;
  const reviewedSuggestions = suggestionStates.promoted + suggestionStates.missed + suggestionStates.artifact + suggestionStates.unsure;
  return {
    roi_count: data.rois.length,
    event_count: eventCount,
    suggestion_count: data.discovery?.suggestions?.length || 0,
    roi_states: roiStates,
    event_states: eventStates,
    suggestion_states: suggestionStates,
    trace_quality: traceQuality,
    control_ready: controlReady,
    reviewer_counts: reviewerCounts,
    reviewer_missing: reviewerMissing,
    triage_categories: triageQueues,
    triage_queue_counts: triageQueues,
    review_burden: {
      candidate_rois_per_accepted_roi: data.rois.length / Math.max(1, roiStates.accepted),
      candidate_events_per_accepted_event: eventCount / Math.max(1, eventStates.accepted)
    },
    review_progress: {
      reviewed_rois: reviewedRois,
      reviewed_events: reviewedEvents,
      reviewed_suggestions: reviewedSuggestions,
      roi_review_fraction: reviewedRois / Math.max(1, data.rois.length),
      event_review_fraction: reviewedEvents / Math.max(1, eventCount),
      suggestion_review_fraction: reviewedSuggestions / Math.max(1, data.discovery?.suggestions?.length || 0),
      tuning_ready: reviewedRois >= 20 && reviewedEvents >= 20,
      tuning_ready_targets: {reviewed_rois: 20, reviewed_events: 20}
    }
  };
}

function auditRows(title, counts){
  const total = Object.values(counts).reduce((a,b) => a+b, 0) || 1;
  let html = `<h2>${title}</h2><div class="auditBars">`;
  for(const [name, count] of Object.entries(counts)){
    const pct = Math.round(100 * count / total);
    html += `<div class="auditRow"><span>${name}</span><div class="auditBar"><div class="auditFill" style="width:${pct}%"></div></div><b>${count}</b></div>`;
  }
  return html + '</div>';
}


function renderTemplateGridProgressGates(){
  const payload = typeof templateGridPayload === 'function' ? templateGridPayload() : {};
  const text = JSON.stringify(payload || {}).toLowerCase();
  const gates = [
    ['TG-005', 'Dataset video manifest parsed', ['video_manifest', 'videomanifest', 'label_counts']],
    ['TG-016', 'Template projection built', ['template_spec', 'template projection', 'template_projection']],
    ['TG-026', 'Per-video rigid registration completed', ['registration', 'registered_videos', 'registered video']],
    ['TG-034', '32x32 grid specification generated', ['grid_spec', 'grid 32', '32x32']],
    ['TG-043', 'Grid states extracted', ['grid_state', 'grid_states']],
    ['TG-050', 'Video-level dynamics dataset split', ['split_unit', 'split manifest', 'split_manifest']],
    ['TG-058', 'Persistence baseline evaluated', ['persistence', 'baseline']],
    ['TG-064', 'Autoencoder reconstruction trained', ['autoencoder', 'reconstruction']],
    ['TG-072', 'Latent RNN prediction trained', ['latent_rnn', 'latent rnn', 'gru']],
    ['TG-078', 'Latent classifier evaluated', ['latent_classifier', 'latent classifier', 'confusion']],
    ['TG-081', 'Dashboard/report artifacts available', ['report', 'templategridpanel', 'template / registration / grid']]
  ];
  return `
    <section class="archCard templateGridGates" id="templateGridProgressGates">
      <div class="runCardHeader"><h3>Template Grid Dynamics Gates</h3><span class="runStatus">goal.md</span></div>
      <p class="hint">Progress gates for the template-aligned 32x32 grid workflow. These gates exclude inverse control/stimulation and transformer modeling.</p>
      <div class="templateGridGateList">
        ${gates.map(([id, label, needles]) => {
          const done = needles.some(needle => text.includes(needle));
          return `<div class="templateGridGate ${done ? 'done' : 'pending'}"><b>${escapeHtml(id)}</b><span>${escapeHtml(label)}</span><i>${done ? 'detected' : 'not recorded'}</i></div>`;
        }).join('')}
      </div>
    </section>`;
}

function renderMetricsAudit(){
  const root = document.getElementById('metricsAudit');
  if(!root) return;
  const s = annotationSummary();
  const actionCount = Object.values(annotations.reviewStats?.actions || {}).reduce((a,b) => a + b, 0);
  const batch = nextAnnotationBatch();
  const batchRows = batch.rois.slice(0, 10).map(item => `
    <tr><td>${item.roi_id}</td><td>${fmt(item.score, 2)}</td><td>${item.event_count}</td><td>${escapeHtml(item.reasons.join(', '))}</td></tr>
  `).join('');
  const eventRows = batch.events.slice(0, 8).map(item => `
    <tr><td>${item.roi_id}</td><td>${item.frame}</td><td>${fmt(item.score, 2)}</td><td>${fmt(item.z, 2)}</td></tr>
  `).join('');
  const suggestionRows = batch.suggestions.slice(0, 8).map(item => `
    <tr><td>${item.suggestion_id}</td><td>${fmt(item.score, 2)}</td><td>${escapeHtml(item.reasons.join(', '))}</td></tr>
  `).join('');
  root.innerHTML = `
    ${renderRunSummaryCards(activeRun())}
    <section class="progressHero">
      <div>
        <span class="homeEyebrow">Review readiness</span>
        <h2>${s.review_progress.tuning_ready ? 'Ready for first tuning comparison' : 'More seed labels needed'}</h2>
        <p class="hint">Progress focuses on what blocks confidence: label coverage, reviewer provenance, artifact burden, and representative examples.</p>
      </div>
      <div class="progressActions">
        <a class="primaryActionButton" href="#review">Continue Review</a>
        <a class="textButton" href="#report">Open Report</a>
      </div>
    </section>
    <div class="progressReadinessGrid">
      <div class="metric"><b>${s.review_progress.reviewed_rois}/${s.review_progress.tuning_ready_targets.reviewed_rois}</b><span>ROI seed labels</span></div>
      <div class="metric"><b>${s.review_progress.reviewed_events}/${s.review_progress.tuning_ready_targets.reviewed_events}</b><span>event seed labels</span></div>
      <div class="metric"><b>${Object.values(s.reviewer_missing || {}).reduce((a,b) => a + b, 0)}</b><span>missing reviewer IDs</span></div>
      <div class="metric"><b>${s.triage_queue_counts.artifact_like || 0}</b><span>artifact-like queue</span></div>
    </div>
    ${renderTemplateGridProgressGates()}
    ${typeof renderTemplateGridSweepPanel === 'function' ? renderTemplateGridSweepPanel(activeRun(), {standalone:true}) : ''}
    <details class="progressMetrics">
      <summary>Detailed metrics</summary>
      <div class="metricGrid">
      <div class="metric"><b>${s.roi_count}</b><span>candidate ROIs</span></div>
      <div class="metric"><b>${s.roi_states.accepted}</b><span>accepted ROIs</span></div>
      <div class="metric"><b>${s.event_count}</b><span>candidate events</span></div>
      <div class="metric"><b>${s.event_states.accepted}</b><span>accepted events</span></div>
      <div class="metric"><b>${s.suggestion_count}</b><span>discovery suggestions</span></div>
      <div class="metric"><b>${s.suggestion_states.promoted}</b><span>promoted suggestions</span></div>
      <div class="metric"><b>${s.review_burden.candidate_rois_per_accepted_roi.toFixed(1)}</b><span>ROIs per accepted ROI</span></div>
      <div class="metric"><b>${s.review_burden.candidate_events_per_accepted_event.toFixed(1)}</b><span>events per accepted event</span></div>
      <div class="metric"><b>${actionCount}</b><span>review actions</span></div>
      <div class="metric"><b>${annotations.reviewStats?.lastActionAt ? 'yes' : 'no'}</b><span>active session</span></div>
      <div class="metric"><b>${Math.round(100 * s.review_progress.roi_review_fraction)}%</b><span>ROI review progress</span></div>
      <div class="metric"><b>${s.review_progress.tuning_ready ? 'yes' : 'no'}</b><span>tuning-ready labels</span></div>
      </div>
    </details>
    <div class="archCard annotationBatchCard">
      <div class="runCardHeader"><h3>Recommended Next Annotation Batch</h3><span class="runStatus">${s.review_progress.reviewed_rois}/${s.review_progress.tuning_ready_targets.reviewed_rois} ROI labels</span></div>
      <p class="hint">Use the Review queue option “Next annotation batch” to work through these candidates first. The first tuning milestone is 20 reviewed ROIs and 20 reviewed events.</p>
      <div class="batchGrid">
        <div>
          <h2>ROIs</h2>
          <table class="smallTable"><tr><th>ID</th><th>Score</th><th>Events</th><th>Why</th></tr>${batchRows || '<tr><td colspan="4">No ROI batch items.</td></tr>'}</table>
        </div>
        <div>
          <h2>Events</h2>
          <table class="smallTable"><tr><th>ROI</th><th>Frame</th><th>Score</th><th>z</th></tr>${eventRows || '<tr><td colspan="4">No event batch items.</td></tr>'}</table>
        </div>
        <div>
          <h2>Suggestions</h2>
          <table class="smallTable"><tr><th>ID</th><th>Score</th><th>Why</th></tr>${suggestionRows || '<tr><td colspan="3">No suggestion batch items.</td></tr>'}</table>
        </div>
      </div>
    </div>
    <div class="auditSplit">
      <div class="archCard">${auditRows('ROI states', s.roi_states)}</div>
      <div class="archCard">${auditRows('Event states', s.event_states)}</div>
      <div class="archCard">${auditRows('Trace quality', s.trace_quality)}</div>
      <div class="archCard">${auditRows('Control readiness', s.control_ready)}</div>
      <div class="archCard">${auditRows('Triage queues', s.triage_queue_counts)}</div>
      <div class="archCard">${auditRows('Discovery suggestions', s.suggestion_states)}</div>
    </div>
    ${renderRobustnessExampleGallery()}
    ${renderValidationBenchmarkPanel()}
    ${renderAdjudicationPanel()}`;
  bindMetricsActionPanels();
}

function exampleCard(kind, title, detail, roi=null, suggestion=null){
  if(!roi && !suggestion) {
    return `<article class="exampleCard muted"><h3>${escapeHtml(title)}</h3><p class="hint">No matching example yet.</p></article>`;
  }
  const target = roi ? `ROI ${roi.id}` : `Suggestion ${suggestion.id}`;
  const attrs = roi ? `data-example-roi="${escapeHtml(roi.id)}"` : `data-example-suggestion="${escapeHtml(suggestion.id)}"`;
  return `
    <article class="exampleCard">
      <div class="runCardHeader"><h3>${escapeHtml(title)}</h3><span class="stageStatus ${kind}">${escapeHtml(target)}</span></div>
      <p class="hint">${escapeHtml(detail)}</p>
      <button type="button" ${attrs}>Open In Review</button>
    </article>`;
}

function renderRobustnessExampleGallery(){
  const accepted = reviewRois().find(roi => roiAnn(roi.id).state === 'accept' || roiAnn(roi.id).cell_state === 'accepted') || reviewRois().find(roiStrongNeuronLike);
  const uncertain = reviewRois().find(roi => roiAnn(roi.id).state === 'unsure' || roiAnn(roi.id).cell_state === 'unsure') || reviewRois().find(roiWeakTraceLike);
  const artifact = reviewRois().find(roiArtifactLike);
  const merged = reviewRois().find(roiMergedClusterLike);
  const activeEvent = reviewRois().find(roi => eventsForRoi(roi).length);
  const suggestion = (data.discovery?.suggestions || []).find(s => !suggestionAnn(s.id).state && !annotations.promotedRois[s.id]) || (data.discovery?.suggestions || [])[0];
  return `
    <section class="archCard robustnessGallery" id="robustnessExampleGallery">
      <div class="runCardHeader">
        <h3>Robustness Example Gallery</h3>
        <span class="runStatus">jump targets</span>
      </div>
      <p class="hint">Use these as a quick sanity set while tuning parameters: strong neuron, uncertain trace, artifact-like ROI, merged cluster, active event, and missed-neuron suggestion.</p>
      <div class="exampleGrid">
        ${exampleCard('ok', 'Accepted / Strong Neuron', accepted ? `${eventsForRoi(accepted).length} events, area ${accepted.area}, score ${fmt(roiQualityScore(accepted), 2)}` : '', accepted)}
        ${exampleCard('warn', 'Uncertain / Weak Trace', uncertain ? `Trace or label uncertainty; SNR ${fmt(scoreValue(uncertain, 'traceSnr', null), 2)}` : '', uncertain)}
        ${exampleCard('bad', 'Artifact-Like ROI', artifact ? artifactReasonsForRoi(artifact).join(', ') || 'artifact cue' : '', artifact)}
        ${exampleCard('warn', 'Merged / Large Cluster', merged ? `Area ${merged.area}; may need split/merge review` : '', merged)}
        ${exampleCard('ok', 'Event-Supported ROI', activeEvent ? `First candidate event at frame ${eventsForRoi(activeEvent)[0]?.frame}` : '', activeEvent)}
        ${exampleCard('warn', 'Missed-Neuron Suggestion', suggestion ? `Area ${suggestion.area}; score ${fmt(scoreValue(suggestion, 'priorityScore', suggestion.discoveryScore), 2)}` : '', null, suggestion)}
      </div>
    </section>`;
}

function renderValidationBenchmarkPanel(){
  const run = activeRun() || plannedRun();
  const normalized = normalizePipelineDraft(JSON.parse(JSON.stringify(run || pipelineDraft)));
  const validation = validatePipeline(normalized);
  const realtime = pipelineRealtimeSummary(normalized);
  const readiness = backendReadiness();
  const warnings = [...validation.warnings, ...realtime.warnings];
  const command = `.venv-neurobench/bin/python tools/benchmark_pipeline_stage.py --frames 300 --height 128 --width 128 --out Outputs/Benchmarks/${datasetId}_stage_latency.json`;
  return `
    <section class="archCard validationBenchmarkPanel" id="validationBenchmarkPanel">
      <div class="runCardHeader">
        <h3>Validation And Real-Time Readiness</h3>
        <span class="stageStatus ${validation.status === 'valid' && !warnings.length ? 'ok' : 'warn'}">${escapeHtml(validation.status)}</span>
      </div>
      <div class="metricGrid">
        <div class="metric"><b>${fmt(realtime.frame_rate_hz, 1)}</b><span>Hz target</span></div>
        <div class="metric"><b>${realtime.frame_budget_ms ? fmt(realtime.frame_budget_ms, 1) : 'n/a'}</b><span>ms/frame budget</span></div>
        <div class="metric"><b>${fmt(realtime.estimated_ms, 1)}</b><span>estimated ms/frame</span></div>
        <div class="metric"><b>${realtime.gpu.length}</b><span>GPU-sensitive stages</span></div>
      </div>
      <p class="hint">${escapeHtml(readiness.text)}</p>
      ${validation.errors.map(e => `<div class="qcWarning">${escapeHtml(e)}</div>`).join('')}
      ${warnings.map(w => `<div class="pipelineWarning">${escapeHtml(w)}</div>`).join('') || '<div class="stageStatus ok">No real-time warnings recorded for this stack.</div>'}
      <details>
        <summary>Synthetic latency smoke test</summary>
        <pre>${escapeHtml(command)}</pre>
      </details>
      <div class="buttonRow">
        <button type="button" id="metricsGeneratePreviewBtn" ${readiness.ok ? '' : 'disabled'}>Generate Active Preview</button>
        <button type="button" id="metricsDownloadValidationBtn">Download Validation Summary</button>
      </div>
    </section>`;
}

function renderAdjudicationPanel(){
  return `
    <section class="archCard adjudicationPanel" id="adjudicationPanel">
      <div class="runCardHeader"><h3>Adjudication Comparator</h3><span class="runStatus">two-file review</span></div>
      <p class="hint">Load two annotation JSON files to find disagreements that need a final lab decision. The comparison is local in the browser.</p>
      <div class="adjudicationInputs">
        <label>Reviewer A <input type="file" id="adjudicationFileA" accept=".json,application/json"></label>
        <label>Reviewer B <input type="file" id="adjudicationFileB" accept=".json,application/json"></label>
        <button type="button" id="runAdjudicationCompareBtn">Compare</button>
      </div>
      <div id="adjudicationResults"><p class="hint">No comparison loaded.</p></div>
    </section>`;
}

function annotationLabelForGroup(group, item){
  if(!item) return '';
  if(group === 'events') return item.event_state || item.state || '';
  if(group === 'suggestions') return item.state || '';
  return item.cell_state || (item.state === 'accept' ? 'accepted' : item.state === 'reject' ? 'rejected' : item.state || '');
}

function clientAgreementReport(annA, annB){
  const groups = ['rois', 'events', 'suggestions'];
  const rows = [];
  for(const group of groups){
    const a = annA[group] || {};
    const b = annB[group] || {};
    for(const id of [...new Set([...Object.keys(a), ...Object.keys(b)])].sort((x,y) => String(x).localeCompare(String(y), undefined, {numeric:true}))){
      const labelA = annotationLabelForGroup(group, a[id]);
      const labelB = annotationLabelForGroup(group, b[id]);
      const both = Boolean(labelA && labelB);
      if(!both || labelA !== labelB) rows.push({group, id, labelA, labelB, reviewerA: a[id]?.reviewer_id || '', reviewerB: b[id]?.reviewer_id || ''});
    }
  }
  const labeled = rows.filter(row => row.labelA || row.labelB).length;
  return {generatedAt: new Date().toISOString(), disagreement_count: rows.length, labeled_disagreement_count: labeled, rows};
}

function readJsonFile(input){
  return new Promise((resolve, reject) => {
    const file = input?.files?.[0];
    if(!file) reject(new Error('missing file'));
    const reader = new FileReader();
    reader.onload = () => {
      try { resolve(JSON.parse(reader.result)); }
      catch (err) { reject(err); }
    };
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

async function runAdjudicationCompare(){
  const root = document.getElementById('adjudicationResults');
  try {
    const annA = await readJsonFile(document.getElementById('adjudicationFileA'));
    const annB = await readJsonFile(document.getElementById('adjudicationFileB'));
    const report = clientAgreementReport(annA, annB);
    root.innerHTML = renderAdjudicationResults(report);
    for(const btn of root.querySelectorAll('[data-adjudicate-item]')) btn.onclick = () => openAdjudicationItem(btn.dataset.adjudicateGroup, btn.dataset.adjudicateId);
    document.getElementById('downloadAdjudicationReportBtn').onclick = () => downloadJson(`${datasetId}_adjudication_report.json`, report);
  } catch (err) {
    root.innerHTML = `<div class="qcWarning">Could not compare files: ${escapeHtml(err.message || err)}</div>`;
  }
}

function renderAdjudicationResults(report){
  const rows = report.rows.slice(0, 40).map(row => `
    <tr>
      <td>${escapeHtml(row.group)}</td>
      <td>${escapeHtml(row.id)}</td>
      <td>${escapeHtml(row.labelA || 'unlabeled')}</td>
      <td>${escapeHtml(row.labelB || 'unlabeled')}</td>
      <td><button type="button" data-adjudicate-item data-adjudicate-group="${escapeHtml(row.group)}" data-adjudicate-id="${escapeHtml(row.id)}">Open</button></td>
    </tr>`).join('');
  return `
    <div class="metricGrid">
      <div class="metric"><b>${report.disagreement_count}</b><span>disagreement items</span></div>
      <div class="metric"><b>${report.labeled_disagreement_count}</b><span>labeled conflicts/missing labels</span></div>
    </div>
    <table class="smallTable"><tr><th>Group</th><th>ID</th><th>A</th><th>B</th><th></th></tr>${rows || '<tr><td colspan="5">No disagreements found.</td></tr>'}</table>
    ${report.rows.length > 40 ? `<p class="hint">Showing first 40 of ${report.rows.length} disagreement items.</p>` : ''}
    <button type="button" id="downloadAdjudicationReportBtn">Download Comparison JSON</button>`;
}

function openAdjudicationItem(group, id){
  location.hash = '#review';
  if(group === 'events') {
    const [roiId, frame] = String(id).split(':');
    selectRoi(roiId);
    if(frame) {
      selectedEventFrame = Number(frame);
      setFrame(Number(frame));
    }
  } else if(group === 'suggestions') {
    selectSuggestion(id);
  } else {
    selectRoi(id);
  }
}

function bindMetricsActionPanels(){
  for(const btn of document.querySelectorAll('[data-example-roi]')) btn.onclick = () => {
    location.hash = '#review';
    selectRoi(btn.dataset.exampleRoi);
  };
  for(const btn of document.querySelectorAll('[data-example-suggestion]')) btn.onclick = () => {
    location.hash = '#review';
    selectSuggestion(btn.dataset.exampleSuggestion);
  };
  document.getElementById('metricsGeneratePreviewBtn')?.addEventListener('click', () => startGenerationJob({preview:true}));
  document.getElementById('metricsDownloadValidationBtn')?.addEventListener('click', () => {
    const run = normalizePipelineDraft(JSON.parse(JSON.stringify(activeRun() || pipelineDraft)));
    downloadJson(`${datasetId}_validation_summary.json`, {
      dataset_id: datasetId,
      active_run_id: activeRunId(),
      validation: validatePipeline(run),
      realtime: pipelineRealtimeSummary(run),
      backend: backendReadiness()
    });
  });
  document.getElementById('runAdjudicationCompareBtn')?.addEventListener('click', runAdjudicationCompare);
}

function quantile(values, q){
  const arr = values.filter(v => Number.isFinite(v)).sort((a,b) => a-b);
  if(!arr.length) return null;
  const idx = Math.max(0, Math.min(arr.length - 1, Math.round((arr.length - 1) * q)));
  return arr[idx];
}

function fmt(v, digits=2){
  return v === null || v === undefined || Number.isNaN(v) ? 'n/a' : Number(v).toFixed(digits);
}

function availableEvidenceMapsForRun(run){
  const runMaps = run?.artifacts?.evidence_maps || [];
  const maps = runMaps.length ? runMaps : (data.discovery?.evidenceMaps || []);
  return maps.filter(m => m && (m.file || m.path));
}

function selectedQcRun(){
  const runs = data.architectureRuns?.runs || [];
  const selected = activeRunId();
  return runs.find(r => r.run_id === selected) || runs[0] || null;
}

function normalizedRunPipeline(run){
  return (run?.pipeline || []).map((stage, index) => {
    const def = stageDef(stage);
    if(def) return normalizeStageForBuilder(stage, index);
    return Object.assign({id: `legacy_stage_${index + 1}`, enabled: true}, stage, {stage_id: stageOp(stage), op: stageOp(stage), type: stage.type || 'legacy'});
  });
}

function qcOutputAvailable(output, run){
  const key = String(output || '').toLowerCase();
  if(!key) return false;
  if(key.includes('frame') && data.video?.framePattern) return true;
  if(key.includes('drift') && data.qc?.driftStats) return true;
  if(key.includes('noise') && data.qc?.noiseSigmaStats) return true;
  if(key.includes('roi') && data.rois?.length) return true;
  if(key.includes('event') && data.rois?.some(r => (r.events || []).length)) return true;
  if(key.includes('suggestion') && data.discovery?.suggestions?.length) return true;
  if(key.includes('map') && availableEvidenceMapsForRun(run).length) return true;
  if(key.includes('trace') && data.rois?.length) return true;
  return false;
}

function renderQcStageTimeline(run){
  const root = document.getElementById('qcPipelineTimeline');
  if(!root) return;
  const pipeline = normalizedRunPipeline(run);
  if(!pipeline.length){
    root.innerHTML = '<p class="hint">No pipeline is attached to this run yet.</p>';
    return;
  }
  root.innerHTML = pipeline.map((stage, index) => {
    const def = stageDef(stage);
    const expected = def?.expected_qc_outputs || [];
    const outputs = expected.length ? expected.map(item => `<span class="${qcOutputAvailable(item, run) ? 'available' : ''}">${escapeHtml(item)}</span>`).join('') : '<span>no declared QC outputs</span>';
    const status = stage.enabled === false ? 'disabled' : (run?.execution?.status || def?.availability || 'available');
    return `
      <div class="qcPipelineStep">
        <span class="stageIndex">${index + 1}</span>
        <div>
          <div class="componentTitle">
            <h4>${escapeHtml(def?.label || stage.label || stage.name || stageOp(stage) || stage.id)}</h4>
            <span class="stageStatus ${status === 'completed' || status === 'implemented' ? 'ok' : status === 'planned' ? 'warn' : 'off'}">${escapeHtml(String(status).replace(/_/g, ' '))}</span>
          </div>
          <p>${escapeHtml(def?.description || 'Legacy architecture-run step.')}</p>
          <div class="artifactFlow"><i>${escapeHtml(stage.input || def?.input || 'input')}</i><strong>-></strong><i>${escapeHtml(stage.output || def?.output || 'output')}</i></div>
          <div class="miniChipRow qcChips">${outputs}</div>
        </div>
      </div>`;
  }).join('');
}

function intermediateArtifactsForRun(run){
  return Array.isArray(run?.artifacts?.intermediates) ? run.artifacts.intermediates : [];
}
function runHasIntermediates(run){
  return intermediateArtifactsForRun(run).length > 0;
}
function isGammaCfarRun(run){
  const runId = String(run?.run_id || '');
  return runId.startsWith('gamma_cfar_cascade_grid_') && runId.includes('__sweep_');
}
function processRunOptionsHtml(runs, selectedRun){
  const selectedId = selectedRun?.run_id || '';
  const recommended = runs.filter(run => isGammaCfarRun(run) && runHasIntermediates(run));
  const otherGamma = runs.filter(run => isGammaCfarRun(run) && !runHasIntermediates(run));
  const other = runs.filter(run => !isGammaCfarRun(run));
  const option = run => `<option value="${escapeHtml(run.run_id)}" ${selectedId === run.run_id ? 'selected' : ''}>${escapeHtml(runLabel(run))}</option>`;
  const groups = [];
  if(recommended.length) groups.push(`<optgroup label="Gamma CFAR results">${recommended.map(option).join('')}</optgroup>`);
  if(otherGamma.length) groups.push(`<optgroup label="Gamma CFAR runs without previews">${otherGamma.map(option).join('')}</optgroup>`);
  if(other.length) groups.push(`<optgroup label="Current review / other runs">${other.map(option).join('')}</optgroup>`);
  return groups.join('');
}
function gammaCfarQuickPickHtml(runs, selectedRun){
  const recommended = runs.filter(run => isGammaCfarRun(run) && runHasIntermediates(run));
  if(!recommended.length) return '';
  return `
    <section class="archCard gammaQuickPick">
      <div class="runCardHeader">
        <div>
          <h3>Gamma CFAR Results</h3>
          <p class="hint">These runs have synchronized intermediate videos attached for immediate inspection.</p>
        </div>
        <span class="runStatus">${recommended.length} ready</span>
      </div>
      <div class="buttonRow">
        ${recommended.map(run => `<button type="button" class="${selectedRun?.run_id === run.run_id ? 'active' : ''}" data-qc-run-shortcut="${escapeHtml(run.run_id)}">${escapeHtml(runLabel(run))}</button>`).join('')}
      </div>
    </section>`;
}
function findIntermediateForStage(stage, run){
  const op = stageOp(stage);
  const id = stage.id || '';
  const artifacts = intermediateArtifactsForRun(run);
  const exact = artifacts.find(item =>
    (id && (item.step_id === id || item.stage === id || item.id === id)) ||
    (!id && op && item.id === op)
  );
  if(exact) return exact;
  const opMatches = artifacts.filter(item => op && (item.stage_id === op || item.id === op));
  return opMatches.length === 1 ? opMatches[0] : null;
}

function summaryTileForStage(stage, run){
  const op = stageOp(stage);
  const params = Object.assign({}, stage.params || {}, run?.parameters || {});
  if(op === 'source_video_import') return {
    status: 'available',
    frame_pattern: data.video?.framePattern,
    description: 'Raw source frames for this pipeline run.'
  };
  if(op === 'component_filter') return {
    status: 'summary',
    description: 'Component filtering affects candidate count and shape constraints; no per-frame component overlay is attached for this run.',
    summaryRows: [
      {label:'candidate ROIs', value:run?.summary?.roi_count ?? data.rois.length ?? 'n/a'},
      {label:'support', value:frameDurationLabel(params['components.support_min_frames'] ?? params.support_min_frames ?? stage.params?.support_min_frames)},
      {label:'min area', value:params['components.min_area_px'] ?? stage.params?.min_area_px ?? 'n/a'}
    ]
  };
  if(op === 'local_background_ring') return {
    status: 'summary',
    description: 'Background-ring correction is trace-level context; frame output is optional unless exported by the runner.',
    summaryRows: [
      {label:'neuropil weight', value:stage.params?.neuropil_weight ?? params['background.neuropil_weight'] ?? 'n/a'},
      {label:'ring radius', value:stage.params?.ring_radius_px ?? params['background.ring_radius_px'] ?? 'n/a'},
      {label:'review traces', value:run?.summary?.roi_count ?? data.rois.length ?? 'n/a'}
    ]
  };
  if(op === 'robust_kalman_positive_innovation') return {
    status: 'summary',
    description: 'Kalman event scoring is trace/event metadata; inspect selected ROI traces in Review for frame-level timing.',
    summaryRows: [
      {label:'candidate events', value:run?.summary?.event_count ?? data.events?.length ?? 'n/a'},
      {label:'threshold', value:stage.params?.event_threshold ?? params['events.event_threshold'] ?? 'n/a'},
      {label:'gain', value:stage.params?.kalman_gain ?? params['events.kalman_gain'] ?? 'n/a'}
    ]
  };
  if(op === 'heuristic_priority_v1') return {
    status: 'summary',
    description: 'Priority ranking orders review targets; it does not create a new frame video.',
    summaryRows: [
      {label:'ranked ROIs', value:run?.summary?.roi_count ?? data.rois.length ?? 'n/a'},
      {label:'suggestions', value:run?.summary?.suggestion_count ?? data.discovery?.suggestions?.length ?? 'n/a'},
      {label:'review ready', value:runGenerated(run) ? 'yes' : 'partial'}
    ]
  };
  if(op === 'review_data_export') return {
    status: runGenerated(run) ? 'summary' : 'missing',
    description: 'Final browser review bundle generated from the selected pipeline run.',
    summaryRows: runGenerated(run) ? [
      {label:'review data', value:'attached'},
      {label:'ROIs', value:run?.summary?.roi_count ?? 'n/a'},
      {label:'events', value:run?.summary?.event_count ?? 'n/a'}
    ] : []
  };
  return null;
}

function qcTileImageHtml(tile){
  if(tile.frame_pattern) return `<img class="qcStageMedia" data-frame-pattern="${escapeHtml(tile.frame_pattern)}" data-stage-id="${escapeHtml(tile.id || tile.stage_id || tile.label)}" data-missing-text="${escapeHtml(tile.label)} frame did not load" onerror="handleQcImageError(this)" alt="${escapeHtml(tile.label)}">`;
  if(tile.file || tile.path) return `<img class="qcStageMedia" src="${escapeHtml(artifactUrl(tile.file || tile.path))}" data-missing-text="${escapeHtml(tile.label)} artifact did not load" onerror="handleQcImageError(this)" alt="${escapeHtml(tile.label)}">`;
  if(tile.summaryRows?.length) return `<div class="qcStageSummary">${tile.summaryRows.map(row => `<div><b>${escapeHtml(row.value)}</b><span>${escapeHtml(row.label)}</span></div>`).join('')}</div>`;
  return `<div class="qcStageMissing">${escapeHtml(tile.missing || 'Output not generated yet')}</div>`;
}
function handleQcImageError(imgEl){
  const msg = imgEl?.dataset?.missingText || 'Artifact did not load';
  const div = document.createElement('div');
  div.className = 'qcStageMissing';
  div.textContent = msg;
  imgEl.closest('.qcStageTile')?.classList.add('missing');
  imgEl.replaceWith(div);
}
function qcStageTiles(run){
  const tiles = [{
    id: 'raw_video',
    label: 'Raw video',
    stage_id: 'source_video_import',
    status: 'available',
    frame_pattern: data.video?.framePattern,
    description: 'Source frames used by the current review data.'
  }];
  const pipeline = normalizedRunPipeline(run);
  for(const stage of pipeline){
    const def = stageDef(stage);
    const artifact = findIntermediateForStage(stage, run);
    const summary = artifact ? null : summaryTileForStage(stage, run);
    tiles.push({
      id: stage.id || stageOp(stage),
      label: artifact?.label || def?.label || stage.label || stage.name || stageOp(stage),
      stage_id: stageOp(stage),
      status: artifact ? 'available' : (summary?.status || 'missing'),
      frame_pattern: artifact?.frame_pattern || artifact?.framePattern || summary?.frame_pattern,
      file: artifact?.file,
      path: artifact?.path,
      description: artifact?.description || summary?.description || def?.description || 'Pipeline stage output.',
      summaryRows: summary?.summaryRows || [],
      missing: artifact || summary?.status === 'summary' || summary?.status === 'available' ? '' : 'Intermediate frames not attached yet.'
    });
  }
  for(const map of availableEvidenceMapsForRun(run)) tiles.push({
    id: map.id || map.label,
    label: map.label || map.id || 'Evidence map',
    stage_id: 'evidence_map',
    status: 'available',
    file: map.file || map.path,
    description: 'Static evidence map from the selected run.'
  });
  return tiles;
}
function renderQcStageGrid(run){
  const root = document.getElementById('qcStageGrid');
  if(!root) return;
  const size = setting('qcTileSize') || document.getElementById('qcTileSize')?.value || 'medium';
  const missingOnly = Boolean(document.getElementById('qcMissingOnly')?.checked);
  const tiles = qcStageTiles(run).filter(tile => !missingOnly || tile.status === 'missing');
  root.className = `qcStageGrid ${escapeHtml(size)}`;
  root.innerHTML = tiles.map(tile => `
    <article class="qcStageTile ${tile.status === 'missing' ? 'missing' : ''} ${tile.status === 'summary' ? 'summaryOnly' : ''}">
      <div class="componentTitle">
        <h4>${escapeHtml(tile.label)}</h4>
        <span class="stageStatus ${tile.status === 'available' ? 'ok' : 'warn'}">${escapeHtml(tile.status)}</span>
      </div>
      <div class="qcStageFrame">${qcTileImageHtml(tile)}</div>
      <p>${escapeHtml(tile.description || '')}</p>
      <div class="miniChipRow"><span>${escapeHtml(tile.stage_id || 'stage')}</span></div>
    </article>`).join('');
  updateQcFrameView();
}

function updateQcFrameView(){
  const qcSlider = document.getElementById('qcFrameSlider');
  const qcLabel = document.getElementById('qcFrameLabel');
  if(qcSlider) qcSlider.value = currentFrame;
  if(qcLabel) qcLabel.textContent = `${frameLabelText(currentFrame)} / ${data.video.frames}`;
  for(const imgEl of document.querySelectorAll('[data-frame-pattern]')){
    const url = framePatternPath(imgEl.dataset.framePattern, currentFrame);
    imgEl.src = withMediaCacheKey(url, `${activeRunId()}:${imgEl.dataset.stageId || ''}:${currentFrame}`);
  }
  if(typeof updateDataCompareFrameView === 'function') updateDataCompareFrameView();
}

function toggleQcPlay(){
  const btn = document.getElementById('qcPlayBtn');
  if(qcTimer) {
    clearInterval(qcTimer);
    qcTimer = null;
    if(btn) btn.textContent = 'Play';
    return;
  }
  if(btn) btn.textContent = 'Pause';
  qcTimer = setInterval(() => setFrame(currentFrame >= data.video.frames ? 1 : currentFrame + 1), 120);
}

function wireDatasetQcControls(){
  const runSelect = document.getElementById('qcRunSelect');
  if(runSelect) runSelect.onchange = async e => {
    await selectActiveRun(e.target.value, {loadReview:false});
    renderDatasetQc();
  };
  const mapSelect = document.getElementById('qcEvidenceSelect');
  if(mapSelect) mapSelect.onchange = e => {
    setSetting('qcEvidenceMap', e.target.value);
    updateQcFrameView();
  };
  const frameSlider = document.getElementById('qcFrameSlider');
  if(frameSlider) frameSlider.oninput = e => setFrame(Number(e.target.value));
  const tileSize = document.getElementById('qcTileSize');
  const missingOnly = document.getElementById('qcMissingOnly');
  if(tileSize) tileSize.onchange = e => {
    setSetting('qcTileSize', e.target.value);
    renderQcStageGrid(selectedQcRun());
  };
  if(missingOnly) missingOnly.onchange = () => renderQcStageGrid(selectedQcRun());
  const prev = document.getElementById('qcPrevFrameBtn');
  const next = document.getElementById('qcNextFrameBtn');
  const play = document.getElementById('qcPlayBtn');
  if(prev) prev.onclick = () => setFrame(currentFrame - 1);
  if(next) next.onclick = () => setFrame(currentFrame + 1);
  if(play) play.onclick = toggleQcPlay;
  for(const btn of document.querySelectorAll('[data-qc-run-shortcut]')) btn.onclick = async () => {
    await selectActiveRun(btn.dataset.qcRunShortcut, {loadReview:false});
    renderDatasetQc();
  };
}

function renderProcessDecisionSupport(run){
  const readiness = runReadiness(run);
  const utility = runUtilityScore(run);
  const missingTiles = qcStageTiles(run).filter(tile => tile.status === 'missing');
  const changed = pipelineChangeSummary(run);
  const recs = recommendationsFromAnnotations().slice(0, 3);
  return `
    <section class="archCard processDecisionSupport">
      <div class="runCardHeader">
        <div>
          <h3>Data Decision Support</h3>
          <p class="hint">Data mirrors the active pipeline run and highlights what blocks interpretation.</p>
        </div>
        <span class="stageStatus ${readiness.className}">${escapeHtml(readiness.label)}</span>
      </div>
      <div class="metricGrid">
        <div class="metric"><b>${utility.score === null ? 'n/a' : utility.score}</b><span>utility score</span></div>
        <div class="metric"><b>${missingTiles.length}</b><span>missing stage outputs</span></div>
        <div class="metric"><b>${normalizedRunPipeline(run).length}</b><span>pipeline stages</span></div>
        <div class="metric"><b>${runGenerated(run) ? 'yes' : 'no'}</b><span>generated review data</span></div>
      </div>
      <p class="hint">${escapeHtml(changed || readiness.text)}</p>
      ${missingTiles.length ? `<div class="pipelineWarning">Missing outputs: ${escapeHtml(missingTiles.slice(0, 5).map(tile => tile.label).join(', '))}${missingTiles.length > 5 ? `, +${missingTiles.length - 5} more` : ''}.</div>` : '<div class="stageStatus ok">All declared browser-readable outputs are present for this run.</div>'}
      <div class="recommendationGrid">
        ${recs.map(rec => `<article class="recommendationCard compact"><h3>${escapeHtml(rec.title)}</h3><p>${escapeHtml(rec.text)}</p></article>`).join('')}
      </div>
    </section>`;
}

function processInsightPanel(run){
  const analysis = proposalAnalysisForRun(run);
  const proposalRows = analysis?.missed_neuron_proposals?.rows || null;
  const artifactRows = analysis?.artifact_classifier?.rows || null;
  const missed = proposalRows ? proposalRows.slice(0, 8)
    .map(s => `<tr><td>${escapeHtml(s.suggestion_id)}</td><td>${fmt(s.proposal_score, 2)}</td><td>${fmt(s.event_support, 2)}</td><td>${escapeHtml(s.reasons?.join(', ') || s.artifact_cue || 'none')}</td></tr>`)
    .join('') : [...(data.discovery?.suggestions || [])]
      .sort((a,b) => scoreValue(b, 'priorityScore', scoreValue(b, 'discoveryScore')) - scoreValue(a, 'priorityScore', scoreValue(a, 'discoveryScore')))
      .slice(0, 8)
      .map(s => `<tr><td>${escapeHtml(s.id)}</td><td>${fmt(scoreValue(s, 'priorityScore', scoreValue(s, 'discoveryScore')), 2)}</td><td>${fmt(scoreValue(s, 'eventSupport', null), 2)}</td><td>${escapeHtml(s.artifactCue || 'none')}</td></tr>`)
      .join('');
  const artifacts = artifactRows ? artifactRows.slice(0, 8)
    .map(row => `<tr><td>${escapeHtml(row.roi_id)}</td><td>${fmt(row.artifact_risk, 2)}</td><td>${escapeHtml(row.area)}</td><td>${escapeHtml(row.reasons?.join(', ') || 'none')}</td></tr>`)
    .join('') : [...data.rois]
      .map(roi => ({roi, reasons: artifactReasonsForRoi(roi)}))
      .filter(item => item.reasons.length)
      .sort((a,b) => scoreValue(b.roi, 'artifactScore') - scoreValue(a.roi, 'artifactScore'))
      .slice(0, 8)
      .map(item => `<tr><td>${escapeHtml(item.roi.id)}</td><td>${fmt(scoreValue(item.roi, 'artifactScore', null), 2)}</td><td>${escapeHtml(item.roi.area)}</td><td>${escapeHtml(item.reasons.join(', '))}</td></tr>`)
      .join('');
  const loading = analysis?.status === 'loading' ? '<span class="stageStatus warn">loading generated analysis</span>' : '';
  const error = analysis?.status === 'error' ? `<div class="qcWarning">${escapeHtml(analysis.error)}</div>` : '';
  const artifactsLink = proposalAnalysisUrl(run) ? `<a href="${escapeHtml(proposalAnalysisUrl(run))}" target="_blank" rel="noreferrer">proposal_analysis.json</a>` : '';
  const proposalSummary = analysis?.missed_neuron_proposals?.summary;
  const classifierSummary = analysis?.artifact_classifier;
  return `
    <section class="archCard processInsightPanel">
      <div class="runCardHeader">
        <h3>Discovery And Artifact Triage</h3>
        <span class="runStatus">active run: ${escapeHtml(runLabel(run))}</span>
      </div>
      <div class="miniChipRow">
        ${loading}
        ${artifactsLink ? `<span>${artifactsLink}</span>` : '<span>using embedded review data</span>'}
        ${proposalSummary ? `<span>${proposalSummary.high_confidence_count} high-confidence missed-neuron proposals</span>` : ''}
        ${classifierSummary ? `<span>${classifierSummary.high_risk_count} artifact-risk ROI cues</span>` : ''}
      </div>
      ${error}
      <div class="batchGrid">
        <div>
          <h2>Missed-neuron candidates</h2>
          <table class="smallTable"><tr><th>ID</th><th>Score</th><th>Event support</th><th>Why it matters</th></tr>${missed || '<tr><td colspan="4">No suggestions available.</td></tr>'}</table>
        </div>
        <div>
          <h2>Artifact-risk ROIs</h2>
          <table class="smallTable"><tr><th>ROI</th><th>Risk</th><th>Area</th><th>Reasons</th></tr>${artifacts || '<tr><td colspan="4">No artifact-risk ROIs flagged.</td></tr>'}</table>
        </div>
      </div>
    </section>`;
}
