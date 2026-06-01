function buildStageCatalog(rawCatalog){
  if(!rawCatalog || typeof rawCatalog !== 'object') return FALLBACK_STAGE_CATALOG;
  return Object.values(rawCatalog).sort((a,b) => (a.order || 0) - (b.order || 0)).map(stage => {
    const names = new Set([...(stage.required_params || []), ...Object.keys(stage.default_params || {}), ...Object.keys(stage.param_ranges || {}), ...Object.keys(stage.parameter_docs || {})]);
    const params = {};
    for(const name of names) params[name] = catalogParamSpec(name, stage);
    return {
      type: stage.type || 'stage',
      op: stage.stage_id,
      label: stage.label || stage.stage_id,
      input: stage.input || '',
      output: stage.output || '',
      params,
      description: stage.description || '',
      why_use_it: stage.why_use_it || '',
      real_time_profile: stage.real_time_profile || {},
      parameter_docs: stage.parameter_docs || {},
      availability: stage.availability || 'implemented',
      ui_group: stage.ui_group || stage.type || 'stage',
      expected_qc_outputs: stage.expected_qc_outputs || []
    };
  });
}

const STAGE_CATALOG = buildStageCatalog(data.pipelineCatalog);

const ARCHITECTURE_PRESETS = [
  {
    id: 'current_review_pipeline',
    label: 'Current local-z review',
    summary: 'Baseline proposal workflow used by the current dashboard.',
    best_for: 'Reviewing the present resting crop and comparing future changes against a known baseline.'
  },
  {
    id: 'adaptive_cfar',
    label: 'Adaptive CFAR detector',
    summary: 'Adds local robust scoring plus adaptive Gamma CFAR for nonuniform background.',
    best_for: 'Bright local clusters, uneven background, and planned 100 Hz streaming tests.'
  },
  {
    id: 'multi_stage_cfar',
    label: 'Multi-stage CFAR cascade',
    summary: 'Tests small-reference CFAR followed by large-reference CFAR to suppress clutter while retaining compact events.',
    best_for: 'Professor-suggested CFAR cascades where local bright clusters need both fine and broad background checks.'
  },
  {
    id: 'artifact_suppression',
    label: 'Artifact suppression pass',
    summary: 'Front-loads despiking, heterogeneity maps, artifact classification, and active-learning ranking.',
    best_for: 'Impulse noise, vessels/static blobs, borders, and false positives that burden review.'
  },
  {
    id: 'high_recall_discovery',
    label: 'High-recall discovery',
    summary: 'Combines local-z candidates with correlation and event-triggered footprint evidence.',
    best_for: 'Finding missed neurons before tightening thresholds.'
  },
  {
    id: 'motion_aware',
    label: 'Motion-aware QC',
    summary: 'Tracks drift and motion sensitivity before scoring candidates.',
    best_for: 'Datasets where weak candidates may be explained by frame-to-frame movement.'
  },
  {
    id: 'pmd_import',
    label: 'PMD denoised local-z',
    summary: 'Uses an external PMD-denoised stack as the input to the local-z detector.',
    best_for: 'Offline denoising comparisons and low-SNR recordings.'
  },
  {
    id: 'suite2p_import',
    label: 'Suite2p import',
    summary: 'Imports Suite2p ROI proposals for review and ranking in this dashboard.',
    best_for: 'Benchmarking against a common calcium-imaging segmentation pipeline.'
  },
  {
    id: 'oasis_import',
    label: 'OASIS event model',
    summary: 'Keeps current ROI proposals but swaps event scoring toward deconvolved traces.',
    best_for: 'Comparing calcium-transient calls against a standard event/deconvolution model.'
  }
];

let pipelineDraft = makePresetPipeline('current_review_pipeline');
let selectedPipelineStageId = pipelineDraft.pipeline[0]?.id || null;
let experimentDraft = {mode: 'sweep', setRows: [], optunaRows: [], optuna: {direction: 'maximize', objective: 'accepted_control_ready_rois', trials: 40, sampler: 'tpe', pruner: 'median'}};

function escapeHtml(value){
  return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

function stageOp(stageOrOp){
  if(typeof stageOrOp === 'string') return stageOrOp;
  return stageOrOp?.stage_id || stageOrOp?.stage || stageOrOp?.op || stageOrOp?.name || '';
}

function stageDef(stageOrOp){ return STAGE_CATALOG.find(s => s.op === stageOp(stageOrOp)); }

function datasetFrameRateHz(){
  return Number(data.dataset?.online?.target_frame_rate_hz || data.dataset?.frame_rate_hz || data.video?.frameRateHz || data.video?.frame_rate_hz || 5);
}

function formatSeconds(seconds){
  const value = Number(seconds);
  if(!Number.isFinite(value)) return 'n/a';
  if(Math.abs(value) >= 60) return `${(value / 60).toFixed(value >= 600 ? 1 : 2)} min`;
  if(Math.abs(value) >= 10) return `${value.toFixed(1)} s`;
  return `${value.toFixed(2)} s`;
}

function frameTimeSec(frameOneBased){
  const rate = datasetFrameRateHz();
  if(!Number.isFinite(rate) || rate <= 0) return null;
  return Math.max(0, Number(frameOneBased || 1) - 1) / rate;
}

function frameLabelText(frameOneBased){
  const sec = frameTimeSec(frameOneBased);
  return sec === null ? String(frameOneBased) : `${frameOneBased} (${formatSeconds(sec)})`;
}

function frameRangeLabel(startFrame, endFrame){
  return `frames ${Math.round(startFrame)}-${Math.round(endFrame)} (${formatSeconds(Math.max(0, Number(endFrame) - Number(startFrame) + 1) / Math.max(1, datasetFrameRateHz()))})`;
}

function frameDurationLabel(frameCount){
  const frames = Number(frameCount);
  if(!Number.isFinite(frames)) return 'n/a';
  return `${frames} frame${frames === 1 ? '' : 's'} (${formatSeconds(frames / Math.max(1, datasetFrameRateHz()))})`;
}

function frameBudgetMs(){
  const rate = datasetFrameRateHz();
  return Number.isFinite(rate) && rate > 0 ? 1000 / rate : null;
}

function realtimeBadges(def){
  const rt = def?.real_time_profile || {};
  const badges = [];
  badges.push(def?.runner_available || def?.locally_runnable ? '<span class="rtBadge implemented">local runner</span>' : '<span class="rtBadge planned">no local runner</span>');
  if(rt.mode) badges.push(`<span class="rtBadge ${escapeHtml(rt.mode)}">${escapeHtml(rt.mode)}</span>`);
  if(rt.adaptive) badges.push('<span class="rtBadge adaptive">adaptive</span>');
  if(rt.stateful) badges.push('<span class="rtBadge stateful">stateful</span>');
  if(rt.requires_gpu) badges.push('<span class="rtBadge gpu">GPU</span>');
  if(rt.closed_loop_candidate) badges.push('<span class="rtBadge loop">closed-loop candidate</span>');
  return badges.join('');
}

function pipelineRealtimeSummary(run){
  const budget = frameBudgetMs();
  let total = 0;
  const offline = [];
  const unknown = [];
  const gpu = [];
  for(const stage of run.pipeline || []){
    if(stage.enabled === false) continue;
    const def = stageDef(stage);
    const rt = def?.real_time_profile || {};
    if(rt.mode === 'offline' || rt.mode === 'batch') offline.push(def?.label || stage.id);
    if(rt.mode === 'unknown') unknown.push(def?.label || stage.id);
    if(rt.requires_gpu) gpu.push(def?.label || stage.id);
    if(Number.isFinite(Number(rt.latency_budget_ms))) total += Number(rt.latency_budget_ms);
  }
  const warnings = [];
  if(offline.length) warnings.push(`${offline.length} offline/batch stage${offline.length === 1 ? '' : 's'} in a plan intended for streaming.`);
  if(unknown.length) warnings.push(`${unknown.length} stage${unknown.length === 1 ? '' : 's'} have unknown latency metadata.`);
  if(budget !== null && total > budget) warnings.push(`Estimated ${total.toFixed(1)} ms/frame exceeds ${budget.toFixed(1)} ms/frame at ${datasetFrameRateHz().toFixed(1)} Hz.`);
  return {frame_rate_hz: datasetFrameRateHz(), frame_budget_ms: budget, estimated_ms: total, offline, unknown, gpu, warnings};
}

function defaultParams(def){
  const params = {};
  for(const [name, spec] of Object.entries(def?.params || {})) params[name] = spec.default ?? '';
  return params;
}

function makeStage(op, index=0){
  const def = stageDef(op) || STAGE_CATALOG[0];
  const base = def.op.replace(/[^a-z0-9]+/gi, '_').replace(/^_|_$/g, '').toLowerCase();
  return {
    id: `${base}_${index + 1}`,
    stage_id: def.op,
    type: def.type,
    op: def.op,
    enabled: true,
    input: def.input,
    output: def.output,
    params: defaultParams(def)
  };
}

function normalizeStageForBuilder(stage, index=0){
  const def = stageDef(stage) || STAGE_CATALOG[0];
  const normalized = Object.assign(makeStage(def.op, index), stage || {});
  normalized.stage_id = def.op;
  normalized.op = def.op;
  normalized.type = normalized.type || def.type;
  normalized.input = normalized.input || def.input;
  normalized.output = normalized.output || def.output;
  normalized.params = Object.assign(defaultParams(def), normalized.params || {});
  if(!normalized.id) normalized.id = makeStage(def.op, index).id;
  return normalized;
}

function normalizePipelineDraft(run){
  const draft = Object.assign(makePresetPipeline('current_review_pipeline'), run || {});
  draft.pipeline = (draft.pipeline || []).map((stage, index) => normalizeStageForBuilder(stage, index));
  if(draft.sweep && Array.isArray(draft.sweep.parameters)) {
    draft.sweep = Object.assign({}, draft.sweep, {parameters: draft.sweep.parameters.map(axis => normalizeSweepAxis(axis, draft.pipeline)).filter(Boolean)});
  } else if(Array.isArray(draft.sweep_axes) && draft.sweep_axes.length) {
    draft.sweep = {id: `${draft.run_id || 'planned'}_sweep`, label: 'Dashboard sweep', parameters: draft.sweep_axes.map(axis => normalizeSweepAxis(axis, draft.pipeline)).filter(Boolean)};
  } else {
    delete draft.sweep;
  }
  delete draft.sweep_axes;
  return draft;
}

function makePresetPipeline(name){
  const presetOps = {
    current_review_pipeline: ['temporal_highpass_gaussian', 'robust_positive_local_z', 'component_filter', 'local_background_ring', 'robust_kalman_positive_innovation', 'heuristic_priority_v1'],
    adaptive_cfar: ['temporal_highpass_gaussian', 'spatial_gaussian', 'robust_positive_local_z', 'adaptive_gamma_cfar', 'component_filter', 'local_background_ring', 'robust_kalman_positive_innovation', 'heuristic_priority_v1'],
    multi_stage_cfar: ['temporal_highpass_gaussian', 'spatial_gaussian', 'gamma_cfar', 'gamma_cfar', 'component_filter', 'local_background_ring', 'robust_kalman_positive_innovation', 'heuristic_priority_v1'],
    artifact_suppression: ['temporal_highpass_gaussian', 'temporal_hampel', 'robust_positive_local_z', 'background_heterogeneity_map', 'saturation_blob_map', 'component_filter', 'artifact_classifier_v1', 'active_learning_ranker'],
    high_recall_discovery: ['temporal_highpass_gaussian', 'robust_positive_local_z', 'component_filter', 'local_background_ring', 'robust_kalman_positive_innovation', 'local_temporal_correlation', 'event_triggered_footprint', 'ensemble_union', 'heuristic_priority_v1'],
    motion_aware: ['temporal_highpass_gaussian', 'rigid_shift_estimate', 'motion_sensitivity_map', 'robust_positive_local_z', 'component_filter', 'local_background_ring', 'robust_kalman_positive_innovation', 'heuristic_priority_v1'],
    pmd_import: ['pmd_denoised_video_import', 'robust_positive_local_z', 'component_filter', 'robust_kalman_positive_innovation', 'heuristic_priority_v1'],
    suite2p_import: ['suite2p_import', 'heuristic_priority_v1'],
    oasis_import: ['temporal_highpass_gaussian', 'robust_positive_local_z', 'component_filter', 'local_background_ring', 'oasis_deconvolution_import', 'heuristic_priority_v1']
  };
  const ops = presetOps[name] || presetOps.current_review_pipeline;
  const pipeline = ops.map((op, i) => makeStage(op, i));
  if(name === 'multi_stage_cfar') {
    const cfarStages = pipeline.filter(stage => stage.stage_id === 'gamma_cfar');
    if(cfarStages[0]) {
      cfarStages[0].id = 'cfar_small_ref';
      cfarStages[0].params = Object.assign({}, cfarStages[0].params, {guard_px: 1, training_radius_px: 5, pfa: 0.01});
      cfarStages[0].metadata = Object.assign({}, cfarStages[0].metadata, {cfar_role: 'small_reference'});
    }
    if(cfarStages[1]) {
      cfarStages[1].id = 'cfar_large_ref';
      cfarStages[1].params = Object.assign({}, cfarStages[1].params, {guard_px: 2, training_radius_px: 17, pfa: 0.001});
      cfarStages[1].metadata = Object.assign({}, cfarStages[1].metadata, {
        cfar_role: 'large_reference',
        previous_mask_step: 'cfar_small_ref',
        combine_mode: 'intersection'
      });
    }
  }
  const runId = `planned_${name}_${Date.now().toString(36)}`;
  const preset = ARCHITECTURE_PRESETS.find(p => p.id === name);
  return {
    schema_version: 1,
    run_id: runId,
    dataset_id: datasetId,
    label: preset ? `Planned ${preset.label}` : `Planned ${name.replace(/_/g, ' ')}`,
    method_family: 'architecture_lab_pipeline',
    purpose: 'candidate_proposal',
    pipeline,
    summary: {roi_count: 0, event_count: 0, suggestion_count: 0, frame_count: data.video.frames},
    artifacts: {source_video: data.dataset?.paths?.raw_video || data.video?.name || '', intermediates: []},
    provenance: {source: 'architecture_lab_builder', source_script: null, git_commit: null, software_versions: {}},
    execution: {status: 'planned'},
    validation: {status: 'unchecked', errors: [], warnings: []}
  };
}

function sweepFactors(run=pipelineDraft){
  return Array.isArray(run.sweep?.parameters) ? run.sweep.parameters : [];
}

function setSweepFactors(factors){
  const cleaned = factors.filter(Boolean);
  if(cleaned.length) {
    pipelineDraft.sweep = Object.assign({
      id: `${pipelineDraft.run_id || 'planned'}_sweep`,
      label: `${pipelineDraft.label || pipelineDraft.run_id || 'Planned'} sweep`
    }, pipelineDraft.sweep || {}, {parameters: cleaned});
  } else {
    delete pipelineDraft.sweep;
  }
}

function normalizeSweepAxis(axis, pipeline=pipelineDraft.pipeline || []){
  const stageKey = axis?.stage || axis?.step_id || axis?.stage_id;
  const stage = pipeline.find(s => s.id === stageKey) || pipeline.find(s => stageOp(s) === stageKey);
  if(!stage || !axis?.param) return null;
  return {
    stage: stage.id,
    stage_id: stageOp(stage),
    param: axis.param,
    values: Array.isArray(axis.values) ? axis.values : [],
    label: axis.label || `${stage.id}.${axis.param}`
  };
}

function validatePipeline(run){
  const errors = [], warnings = [], stageIssues = {};
  const addIssue = (stageId, kind, message) => {
    (kind === 'error' ? errors : warnings).push(message);
    if(stageId) {
      stageIssues[stageId] = stageIssues[stageId] || {errors: [], warnings: []};
      stageIssues[stageId][kind === 'error' ? 'errors' : 'warnings'].push(message);
    }
  };
  const seenIds = new Set();
  const available = new Set(['raw_video']);
  for(const [index, stage] of (run.pipeline || []).entries()){
    const op = stageOp(stage);
    const stageId = stage.id || `stage_${index + 1}`;
    if(!stage.id) addIssue(stageId, 'error', `Stage ${index + 1} is missing an id.`);
    if(seenIds.has(stage.id)) addIssue(stage.id, 'error', `Duplicate stage id: ${stage.id}`);
    seenIds.add(stage.id);
    const def = stageDef(op);
    if(!def) {
      addIssue(stageId, 'error', `Unknown operation: ${op || '(blank)'}`);
      continue;
    }
    if(stage.enabled === false) continue;
    if(stage.input && !available.has(stage.input)) addIssue(stageId, 'error', `${stageId} needs input ${stage.input}, but no earlier enabled stage produces it.`);
    for(const [param, spec] of Object.entries(def.params || {})){
      const value = stage.params?.[param];
      if(value === undefined || value === '') addIssue(stageId, 'error', `${stageId} is missing ${param}.`);
      if(spec.type === 'number' && value !== undefined && value !== ''){
        const numeric = Number(value);
        if(!Number.isFinite(numeric)) addIssue(stageId, 'error', `${stageId}.${param} must be numeric.`);
        else {
          if(spec.min !== undefined && numeric < spec.min) addIssue(stageId, 'error', `${stageId}.${param} is below ${spec.min}.`);
          if(spec.max !== undefined && numeric > spec.max) addIssue(stageId, 'error', `${stageId}.${param} is above ${spec.max}.`);
        }
      }
    }
    if(stage.output) available.add(stage.output);
  }
  for(const axis of sweepFactors(run)){
    const stage = (run.pipeline || []).find(s => s.id === axis.stage);
    if(!stage) {
      addIssue(axis.stage, 'error', `Sweep axis references unknown stage ${axis.stage}.`);
      continue;
    }
    const def = stageDef(stage);
    const spec = def?.params?.[axis.param];
    if(!spec) {
      addIssue(axis.stage, 'error', `Sweep axis references unknown parameter ${axis.stage}.${axis.param}.`);
      continue;
    }
    const values = Array.isArray(axis.values) ? axis.values : [];
    if(!values.length) addIssue(axis.stage, 'error', `Sweep axis ${axis.stage}.${axis.param} has no values.`);
    for(const value of values){
      if(spec.type === 'number'){
        const numeric = Number(value);
        if(!Number.isFinite(numeric)) addIssue(axis.stage, 'error', `Sweep value ${axis.stage}.${axis.param}=${value} must be numeric.`);
        else {
          if(spec.min !== undefined && numeric < spec.min) addIssue(axis.stage, 'error', `Sweep value ${axis.stage}.${axis.param}=${value} is below ${spec.min}.`);
          if(spec.max !== undefined && numeric > spec.max) addIssue(axis.stage, 'error', `Sweep value ${axis.stage}.${axis.param}=${value} is above ${spec.max}.`);
        }
      }
    }
  }
  if(!(run.pipeline || []).some(s => s.enabled !== false && s.type === 'candidate_ranking')) warnings.push('No candidate-ranking stage is enabled.');
  if(!(run.pipeline || []).some(s => s.enabled !== false && s.type === 'event_model')) warnings.push('No event model is enabled; Pipelines will compare ROI candidates only.');
  for(const warning of pipelineRealtimeSummary(run).warnings) warnings.push(warning);
  return {status: errors.length ? 'invalid' : 'valid', errors, warnings, stageIssues};
}

function plannedRun(){
  const validation = validatePipeline(pipelineDraft);
  const run = Object.assign({}, pipelineDraft, {validation});
  if(!sweepFactors(run).length) delete run.sweep;
  return run;
}

function sweepAxisLabel(axis){
  return `${axis.stage || axis.stage_id}.${axis.param}`;
}

function parseSweepValues(raw, spec){
  return String(raw || '').split(',').map(v => v.trim()).filter(Boolean).map(v => spec?.type === 'number' ? Number(v) : v);
}

function axisCombinations(axes){
  const active = (axes || []).filter(axis => Array.isArray(axis.values) && axis.values.length);
  if(!active.length) return [[]];
  return active.reduce((combos, axis) => {
    const next = [];
    for(const combo of combos) for(const value of axis.values) next.push([...combo, {axis, value}]);
    return next;
  }, [[]]);
}

function expandPlannedRuns(run){
  const axes = sweepFactors(run);
  const combos = axisCombinations(axes);
  if(combos.length === 1 && combos[0].length === 0) return [run];
  const totalRuns = combos.length;
  const sweepBase = Object.assign({}, run.sweep || {}, {parameters: axes});
  return combos.map((combo, index) => {
    const child = JSON.parse(JSON.stringify(run));
    child.run_id = `${run.run_id}__sweep_${String(index + 1).padStart(3, '0')}`;
    child.label = `${run.label || run.run_id} sweep ${index + 1}`;
    child.sweep = Object.assign({}, sweepBase, {index, total_runs: totalRuns, parameters: []});
    for(const item of combo){
      const stage = child.pipeline.find(s => s.id === item.axis.stage);
      if(stage) {
        stage.params = stage.params || {};
        stage.params[item.axis.param] = item.value;
      }
      child.sweep.parameters.push({stage: item.axis.stage, stage_id: item.axis.stage_id, param: item.axis.param, value: item.value});
    }
    return child;
  });
}

function plannedManifest(){
  const run = plannedRun();
  const manifest = {schema_version: 1, dataset_id: datasetId, runs: expandPlannedRuns(run)};
  const axes = sweepFactors(run);
  if(axes.length) manifest.sweep = Object.assign({}, run.sweep || {}, {parameters: axes, total_runs: manifest.runs.length});
  return manifest;
}

function paramSummary(stage){
  const def = stageDef(stage);
  const entries = Object.entries(def?.params || {}).slice(0, 4);
  if(!entries.length) return '<span class="pipelineParam muted">no params</span>';
  return entries.map(([name]) => `<span class="pipelineParam">${escapeHtml(name)}=${escapeHtml(stage.params?.[name] ?? '')}</span>`).join('');
}

function parameterHelp(name, spec){
  const parts = [];
  if(spec.doc) parts.push(escapeHtml(spec.doc));
  const bounds = [];
  if(spec.min !== undefined) bounds.push(`min ${spec.min}`);
  if(spec.max !== undefined) bounds.push(`max ${spec.max}`);
  if(bounds.length) parts.push(`Range: ${bounds.join(', ')}.`);
  if(spec.why) parts.push(escapeHtml(spec.why));
  return parts.join(' ');
}

function stageIssueBadge(stage, validation){
  const issues = validation.stageIssues?.[stage.id] || {errors: [], warnings: []};
  if(stage.enabled === false) return '<span class="stageStatus off">off</span>';
  if(issues.errors?.length) return `<span class="stageStatus bad">${issues.errors.length} issue${issues.errors.length === 1 ? '' : 's'}</span>`;
  if(issues.warnings?.length) return `<span class="stageStatus warn">${issues.warnings.length} warning${issues.warnings.length === 1 ? '' : 's'}</span>`;
  return '<span class="stageStatus ok">valid</span>';
}

function downloadJson(name, payload){
  const blob = new Blob([JSON.stringify(payload, null, 2) + '\n'], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

function downloadText(name, text, type='text/plain'){
  const blob = new Blob([String(text || '')], {type});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

async function copyTextToClipboard(text, okMessage='copied'){
  const value = String(text || '');
  try {
    if(navigator.clipboard?.writeText) await navigator.clipboard.writeText(value);
    else {
      const area = document.createElement('textarea');
      area.value = value;
      area.setAttribute('readonly', '');
      area.style.position = 'fixed';
      area.style.left = '-9999px';
      document.body.appendChild(area);
      area.select();
      document.execCommand('copy');
      area.remove();
    }
    setSaveState(okMessage, 'ok');
  } catch(err) {
    setSaveState(`copy failed: ${err.message}`, 'bad');
  }
}

function renderPipelineIdentityPanel(){
  const root = document.getElementById('pipelineIdentityPanel');
  if(!root) return;
  const template = pipelineTemplateFromDraft();
  const saved = savedPipelineTemplates().some(item => item.id === template.id);
  root.innerHTML = `
    <div class="pipelineIdentityGrid">
      <label>Architecture name <input id="pipelineLabelInput" value="${escapeHtml(pipelineDraft.label || '')}" placeholder="Adaptive CFAR hindbrain v1"></label>
      <label>Local ID <input id="pipelineRunIdInput" value="${escapeHtml(pipelineDraft.run_id || '')}" placeholder="planned_adaptive_cfar_v1"></label>
      <label class="wide">Description <input id="pipelineDescriptionInput" value="${escapeHtml(pipelineDraft.description || '')}" placeholder="What this architecture is meant to test"></label>
    </div>
    <p class="hint">${saved ? 'This architecture name/ID matches a saved local template.' : 'Save Architecture stores this stack as a reusable local template for Experiment Lab.'}</p>`;
  document.getElementById('pipelineLabelInput').onchange = e => { pipelineDraft.label = e.target.value; renderPipelineBuilder(); };
  document.getElementById('pipelineRunIdInput').onchange = e => {
    pipelineDraft.run_id = slugify(e.target.value);
    pipelineDraft.template_id = slugify(e.target.value);
    renderPipelineBuilder();
  };
  document.getElementById('pipelineDescriptionInput').onchange = e => { pipelineDraft.description = e.target.value; renderPipelineBuilder(); };
}

function renderPipelineBuilder(){
  renderPipelineIdentityPanel();
  const palette = document.getElementById('pipelineStagePalette');
  const stack = document.getElementById('pipelineStack');
  const inspector = document.getElementById('pipelineInspector');
  const validationRoot = document.getElementById('pipelineValidation');
  const preview = document.getElementById('pipelineJsonPreview');
  if(!palette || !stack || !inspector || !validationRoot || !preview) return;
  const groups = {};
  for(const def of STAGE_CATALOG) (groups[def.type] = groups[def.type] || []).push(def);
  palette.innerHTML = Object.entries(groups).map(([type, defs]) => `
    <details open><summary>${escapeHtml(type.replace(/_/g, ' '))}</summary>
      <div class="stagePaletteGroup">${defs.map(def => `<button type="button" data-stage-op="${escapeHtml(def.op)}">${escapeHtml(def.label)}</button>`).join('')}</div>
    </details>`).join('');
  for(const btn of palette.querySelectorAll('[data-stage-op]')) btn.onclick = () => {
    pipelineDraft.pipeline.push(makeStage(btn.dataset.stageOp, pipelineDraft.pipeline.length));
    selectedPipelineStageId = pipelineDraft.pipeline[pipelineDraft.pipeline.length - 1].id;
    renderPipelineBuilder();
  };
  const validation = validatePipeline(pipelineDraft);
  stack.innerHTML = pipelineDraft.pipeline.map((stage, index) => {
    const def = stageDef(stage);
    return `
    <div class="pipelineStage ${stage.id === selectedPipelineStageId ? 'sel' : ''} ${stage.enabled === false ? 'disabled' : ''}">
      <button type="button" class="pipelineStageMain" data-select-stage="${escapeHtml(stage.id)}">
        <span class="stageIndex">${index + 1}</span>
        <span class="stageBody">
          <b>${escapeHtml(def?.label || stageOp(stage) || stage.id)}</b>
          <span class="stageMeta"><span class="stageTypeChip">${escapeHtml((def?.type || stage.type || 'stage').replace(/_/g, ' '))}</span>${stageIssueBadge(stage, validation)}${realtimeBadges(def)}</span>
          <span class="stageDescription">${escapeHtml(def?.description || '')}</span>
          <span class="artifactFlow"><i>${escapeHtml(stage.input || 'input')}</i><strong>-></strong><i>${escapeHtml(stage.output || 'output')}</i></span>
          <span class="pipelineParamRow">${paramSummary(stage)}</span>
        </span>
      </button>
      <div class="buttonRow">
        <button type="button" data-move-stage="${escapeHtml(stage.id)}" data-dir="-1">Up</button>
        <button type="button" data-move-stage="${escapeHtml(stage.id)}" data-dir="1">Down</button>
        <button type="button" data-duplicate-stage="${escapeHtml(stage.id)}">Duplicate</button>
        <button type="button" data-toggle-stage="${escapeHtml(stage.id)}">${stage.enabled === false ? 'Enable' : 'Disable'}</button>
        <button type="button" data-delete-stage="${escapeHtml(stage.id)}">Delete</button>
      </div>
    </div>`;
  }).join('');
  for(const btn of stack.querySelectorAll('[data-select-stage]')) btn.onclick = () => { selectedPipelineStageId = btn.dataset.selectStage; renderPipelineBuilder(); };
  for(const btn of stack.querySelectorAll('[data-toggle-stage]')) btn.onclick = () => {
    const stage = pipelineDraft.pipeline.find(s => s.id === btn.dataset.toggleStage);
    if(stage) stage.enabled = stage.enabled === false;
    renderPipelineBuilder();
  };
  for(const btn of stack.querySelectorAll('[data-delete-stage]')) btn.onclick = () => {
    pipelineDraft.pipeline = pipelineDraft.pipeline.filter(s => s.id !== btn.dataset.deleteStage);
    setSweepFactors(sweepFactors().filter(axis => axis.stage !== btn.dataset.deleteStage));
    selectedPipelineStageId = pipelineDraft.pipeline[0]?.id || null;
    renderPipelineBuilder();
  };
  for(const btn of stack.querySelectorAll('[data-duplicate-stage]')) btn.onclick = () => {
    const idx = pipelineDraft.pipeline.findIndex(s => s.id === btn.dataset.duplicateStage);
    if(idx >= 0){
      const stage = JSON.parse(JSON.stringify(pipelineDraft.pipeline[idx]));
      stage.id = `${stage.id}_copy`;
      pipelineDraft.pipeline.splice(idx + 1, 0, stage);
      selectedPipelineStageId = stage.id;
    }
    renderPipelineBuilder();
  };
  for(const btn of stack.querySelectorAll('[data-move-stage]')) btn.onclick = () => {
    const idx = pipelineDraft.pipeline.findIndex(s => s.id === btn.dataset.moveStage);
    const next = idx + Number(btn.dataset.dir);
    if(idx >= 0 && next >= 0 && next < pipelineDraft.pipeline.length){
      const [stage] = pipelineDraft.pipeline.splice(idx, 1);
      pipelineDraft.pipeline.splice(next, 0, stage);
    }
    renderPipelineBuilder();
  };
  const selected = pipelineDraft.pipeline.find(s => s.id === selectedPipelineStageId) || pipelineDraft.pipeline[0];
  if(selected) {
    const def = stageDef(selected);
    const numericParams = Object.entries(def?.params || {}).filter(([, spec]) => spec.type === 'number');
    const sweepParamOptions = numericParams.map(([name]) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join('');
    inspector.innerHTML = `
      <h3>${escapeHtml(def?.label || selected.op)}</h3>
      <div class="stageExplain">
        <p>${escapeHtml(def?.description || '')}</p>
        <p><b>Why:</b> ${escapeHtml(def?.why_use_it || '')}</p>
        <div class="stageMeta">${realtimeBadges(def)}</div>
      </div>
      <label>Stage ID <input id="stageIdInput" value="${escapeHtml(selected.id)}"></label>
      ${(Object.entries(def?.params || {}).map(([name, spec]) => `
        <label>${escapeHtml(name)}
          <input data-stage-param="${escapeHtml(name)}" inputmode="${spec.type === 'number' ? 'decimal' : 'text'}" type="text" value="${escapeHtml(selected.params?.[name] ?? spec.default ?? '')}" ${spec.type === 'number' ? `data-min="${spec.min}" data-max="${spec.max}"` : ''}>
          <span class="paramHelp">${parameterHelp(name, spec)}</span>
        </label>`).join(''))}`;
    inspector.innerHTML += `
      <section class="sweepEditor">
        <h3>Parameter sweep</h3>
        <p class="hint">Comma-separated values create planned runs only; no browser-side execution happens.</p>
        <label>Sweep parameter <select id="sweepParamSelect">${sweepParamOptions}</select></label>
        <label>Values <input id="sweepValuesInput" placeholder="1.8, 2.2, 2.6"></label>
        <button type="button" id="addSweepAxisBtn" ${numericParams.length ? '' : 'disabled'}>Add sweep axis</button>
      </section>`;
    document.getElementById('stageIdInput').onchange = e => { selected.id = e.target.value.trim(); selectedPipelineStageId = selected.id; renderPipelineBuilder(); };
    for(const input of inspector.querySelectorAll('[data-stage-param]')) input.oninput = e => {
      const spec = def.params[e.target.dataset.stageParam];
      selected.params[e.target.dataset.stageParam] = spec.type === 'number' && e.target.value !== '' && Number.isFinite(Number(e.target.value)) ? Number(e.target.value) : e.target.value;
    };
    for(const input of inspector.querySelectorAll('[data-stage-param]')) input.onchange = () => renderPipelineBuilder();
    const addSweepBtn = document.getElementById('addSweepAxisBtn');
    if(addSweepBtn) addSweepBtn.onclick = () => {
      const param = document.getElementById('sweepParamSelect').value;
      const spec = def.params[param];
      const values = parseSweepValues(document.getElementById('sweepValuesInput').value, spec);
      if(param && values.length) {
        setSweepFactors([...sweepFactors().filter(axis => !(axis.stage === selected.id && axis.param === param)), {stage: selected.id, stage_id: stageOp(selected), param, values, label: `${selected.id}.${param}`}]);
      }
      renderPipelineBuilder();
    };
  } else {
    inspector.innerHTML = '<p class="hint">Add a stage to configure parameters.</p>';
  }
  const run = plannedRun();
  const manifest = plannedManifest();
  const realtime = pipelineRealtimeSummary(run);
  validationRoot.innerHTML = `
    <div class="validationState ${run.validation.status}">${run.validation.status}</div>
    <div class="realtimeSummary">
      <b>100 Hz readiness</b>
      <span>${fmt(realtime.estimated_ms, 1)} ms estimated / ${realtime.frame_budget_ms ? fmt(realtime.frame_budget_ms, 1) : 'n/a'} ms budget at ${fmt(realtime.frame_rate_hz, 1)} Hz</span>
      <span>${realtime.offline.length} offline, ${realtime.unknown.length} unknown-latency, ${realtime.gpu.length} GPU-sensitive stages</span>
    </div>
    <div class="sweepAxisList">
      ${sweepFactors(run).map((axis, index) => `<div class="sweepAxis"><b>${escapeHtml(sweepAxisLabel(axis))}</b><span>${escapeHtml((axis.values || []).join(', '))}</span><button type="button" data-remove-sweep="${index}">Remove</button></div>`).join('') || '<p class="hint">No sweep axes configured.</p>'}
    </div>
    <div class="pipelineWarning">${manifest.runs.length} planned run${manifest.runs.length === 1 ? '' : 's'} will be saved/exported.</div>
    ${run.validation.errors.map(e => `<div class="qcWarning">${escapeHtml(e)}</div>`).join('')}
    ${run.validation.warnings.map(w => `<div class="pipelineWarning">${escapeHtml(w)}</div>`).join('')}`;
  for(const btn of validationRoot.querySelectorAll('[data-remove-sweep]')) btn.onclick = () => {
    const factors = sweepFactors();
    factors.splice(Number(btn.dataset.removeSweep), 1);
    setSweepFactors(factors);
    renderPipelineBuilder();
  };
  preview.textContent = JSON.stringify(manifest, null, 2);
}

async function savePlannedRun(){
  const planned = plannedManifest();
  const manifest = Object.assign({}, data.architectureRuns || {schema_version: 1, dataset_id: datasetId, runs: []});
  const plannedIds = new Set(planned.runs.map(r => r.run_id));
  manifest.runs = [...(manifest.runs || []).filter(r => !plannedIds.has(r.run_id)), ...planned.runs];
  await persistArchitectureRuns(manifest, 'saved planned run', 'planned_architecture_run.json');
  if(planned.runs?.[0]?.run_id) annotations.settings.activeRunId = planned.runs[0].run_id;
  renderArchitectureLab();
  renderRunSyncControls();
}

function setArchitectureMode(mode){
  const build = mode === 'build';
  document.getElementById('architectureComparePanel')?.classList.toggle('hidden', build);
  document.getElementById('architectureBuildPanel')?.classList.toggle('hidden', !build);
  document.getElementById('archCompareModeBtn')?.classList.toggle('active', !build);
  document.getElementById('archBuildModeBtn')?.classList.toggle('active', build);
  for(const el of document.querySelectorAll('.compareRunControl')) el.classList.toggle('hidden', build);
  if(build) renderPipelineBuilder();
}

function populateRunSelectors(runs){
  const compare = reviewCompareSettings();
  for(const [id, defaultIndex] of [['archRunA', 0], ['archRunB', Math.min(1, Math.max(0, runs.length - 1))]]){
    const select = document.getElementById(id);
    if(!select) continue;
    const previous = select.value;
    select.innerHTML = '';
    for(const run of runs){
      const opt = document.createElement('option');
      opt.value = run.run_id;
      opt.textContent = runLabel(run);
      select.appendChild(opt);
    }
    const preferred = id === 'archRunA' ? compare.runAId : compare.runBId;
    if(runs.some(r => r.run_id === preferred)) select.value = preferred;
    else if(runs.some(r => r.run_id === previous)) select.value = previous;
    else if(runs[defaultIndex]) select.value = runs[defaultIndex].run_id;
  }
}

function reviewDataForRunFromCache(run){
  const key = reviewDataCacheKey(run);
  return key ? reviewDataCache.get(key)?.data || null : null;
}

function reviewDataStatusForRun(run){
  if(!runGenerated(run)) return 'not generated';
  const key = reviewDataCacheKey(run);
  const cached = key ? reviewDataCache.get(key) : null;
  if(cached?.status === 'ready') return 'loaded';
  if(cached?.status === 'loading') return 'loading';
  if(cached?.status === 'error') return 'error';
  return 'not loaded';
}

function reviewFramePath(reviewData, frame){
  const frames = Math.max(1, Number(reviewData?.video?.frames) || 1);
  return framePatternPath(reviewData?.video?.framePattern, Math.max(1, Math.min(frames, frame)));
}

function reviewEventsForRoi(roi){
  return (roi?.events || []).map(ev => ({
    frame: Number(ev.frame ?? ev[0]),
    z: Number(ev.z ?? ev.score ?? ev[1] ?? 0)
  })).filter(ev => Number.isFinite(ev.frame));
}

function reviewEventNearFrame(roi, frame){
  return reviewEventsForRoi(roi).some(ev => Math.abs(ev.frame - frame) <= 1);
}

function reviewDataSummary(reviewData){
  const rois = reviewData?.rois || [];
  const suggestions = reviewData?.discovery?.suggestions || [];
  const events = rois.reduce((sum, roi) => sum + reviewEventsForRoi(roi).length, 0);
  const eventNow = rois.filter(roi => reviewEventNearFrame(roi, currentFrame)).length;
  return {rois: rois.length, suggestions: suggestions.length, events, eventNow};
}

function reviewFrameEventCounts(reviewData){
  if(!reviewData) return [];
  if(Array.isArray(reviewData._frameEventCounts)) return reviewData._frameEventCounts;
  const frames = Math.max(1, Number(reviewData.video?.frames) || data.video.frames || 1);
  const counts = Array(frames + 1).fill(0);
  for(const roi of reviewData.rois || []) for(const ev of reviewEventsForRoi(roi)){
    const frame = Math.max(1, Math.min(frames, Number(ev.frame) || 1));
    counts[frame] += 1;
  }
  reviewData._frameEventCounts = counts;
  return counts;
}

function nextReviewComparisonDifference(direction=1){
  const compare = reviewCompareSettings();
  const runA = runById(compare.runAId) || architectureRuns()[0];
  const runB = runById(compare.runBId) || architectureRuns()[1] || runA;
  const dataA = reviewDataForRunFromCache(runA);
  const dataB = reviewDataForRunFromCache(runB);
  if(!dataA || !dataB) {
    setSaveState('load A/B Review before jumping to differences', 'bad');
    return;
  }
  const aCounts = reviewFrameEventCounts(dataA);
  const bCounts = reviewFrameEventCounts(dataB);
  const frames = Math.max(1, Math.min(aCounts.length - 1, bCounts.length - 1));
  const step = direction >= 0 ? 1 : -1;
  for(let offset = 1; offset <= frames; offset++){
    const frame = ((currentFrame - 1 + step * offset + frames) % frames) + 1;
    if((aCounts[frame] || 0) !== (bCounts[frame] || 0)) {
      setFrame(frame);
      setSaveState(`A/B event-count difference at frame ${frame}`, 'ok');
      return;
    }
  }
  setSaveState('no A/B event-count differences found', 'ok');
}

function reviewComparisonPaneHtml(label, run, reviewData){
  if(!reviewData) {
    return `
      <article class="abReviewPane missing">
        <div class="runCardHeader">
          <h3>${escapeHtml(label)}: ${escapeHtml(runLabel(run) || 'no run')}</h3>
          <span class="stageStatus warn">${escapeHtml(reviewDataStatusForRun(run))}</span>
        </div>
        <div class="abReviewMissing">Load generated review data to inspect this run here.</div>
      </article>`;
  }
  const video = reviewData.video || {};
  const width = Number(video.width) || data.video.width || 1;
  const height = Number(video.height) || data.video.height || 1;
  const frame = Math.max(1, Math.min(Number(video.frames) || data.video.frames || 1, currentFrame));
  const summary = reviewDataSummary(reviewData);
  const rois = [...(reviewData.rois || [])]
    .sort((a,b) => Number(b.priorityScore || b.peakScore || 0) - Number(a.priorityScore || a.peakScore || 0))
    .slice(0, 320);
  const roiCircles = rois.map(roi => {
    const eventNow = reviewEventNearFrame(roi, frame);
    const color = eventNow ? '#facc15' : '#38bdf8';
    const r = Math.max(3, Math.min(18, Math.sqrt(Number(roi.area || 12) / Math.PI) + 2));
    return `<circle class="${eventNow ? 'eventNow' : ''}" cx="${Number(roi.centroidX || 0)}" cy="${Number(roi.centroidY || 0)}" r="${r}" fill="none" stroke="${color}" stroke-width="${eventNow ? 2.4 : 1.2}"><title>ROI ${escapeHtml(roi.id)}${eventNow ? ' event near this frame' : ''}</title></circle>`;
  }).join('');
  return `
    <article class="abReviewPane">
      <div class="runCardHeader">
        <h3>${escapeHtml(label)}: ${escapeHtml(runLabel(run))}</h3>
        <span class="stageStatus ok">loaded</span>
      </div>
      <div class="miniChipRow">
        <span>${summary.rois} ROIs</span>
        <span>${summary.events} events</span>
        <span>${summary.eventNow} active near frame</span>
        <span>${summary.suggestions} suggestions</span>
      </div>
      <div class="abFrame" style="aspect-ratio:${width}/${height}">
        <img src="${escapeHtml(reviewFramePath(reviewData, frame))}" alt="${escapeHtml(runLabel(run))} frame ${frame}">
        <svg class="abOverlay" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true">${roiCircles}</svg>
      </div>
    </article>`;
}

function renderReviewComparisonViewer(){
  const root = document.getElementById('reviewComparisonViewer');
  const status = document.getElementById('reviewComparisonStatus');
  if(!root) return;
  const compare = reviewCompareSettings();
  const runA = runById(compare.runAId) || runById(document.getElementById('archRunA')?.value) || architectureRuns()[0];
  const runB = runById(compare.runBId) || runById(document.getElementById('archRunB')?.value) || architectureRuns()[1] || runA;
  const dataA = reviewDataForRunFromCache(runA);
  const dataB = reviewDataForRunFromCache(runB);
  if(status) status.textContent = `Frame ${currentFrame}: A ${reviewDataStatusForRun(runA)}, B ${reviewDataStatusForRun(runB)}`;
  if(!compare.enabled && !dataA && !dataB) {
    root.innerHTML = '<p class="hint">Load A/B Review to compare generated run frames and ROI overlays without switching the main Review page.</p>';
    return;
  }
  root.innerHTML = `
    <div class="abReviewGrid">
      ${reviewComparisonPaneHtml('A', runA, dataA)}
      ${reviewComparisonPaneHtml('B', runB, dataB)}
    </div>`;
}

async function loadReviewComparison(){
  const runA = runById(document.getElementById('archRunA')?.value) || architectureRuns()[0];
  const runB = runById(document.getElementById('archRunB')?.value) || architectureRuns()[1] || runA;
  const compare = reviewCompareSettings();
  compare.enabled = true;
  compare.runAId = runA?.run_id || '';
  compare.runBId = runB?.run_id || '';
  queueSave();
  const status = document.getElementById('reviewComparisonStatus');
  if(status) status.textContent = 'Loading generated review data for A/B comparison...';
  renderReviewComparisonViewer();
  try {
    await Promise.all([fetchReviewDataForRun(runA), fetchReviewDataForRun(runB)]);
    setSaveState('loaded A/B Review comparison', 'ok');
  } catch (err) {
    setSaveState(err.message || 'A/B Review comparison failed to load', 'bad');
  }
  renderReviewComparisonViewer();
}
