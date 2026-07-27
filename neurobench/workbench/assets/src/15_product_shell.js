const ANNOTATION_TASK_DEFS = {
  neuron_validation: {
    label:'Validate neuron',
    description:'Decide whether the selected candidate is a neuron. The decision is saved only when you choose it.',
    queue:'annotationBatch',
    decisionTarget:'roi',
    decisions:[['accept','Neuron'],['reject','Not neuron'],['unsure','Unsure']],
    actions:[['roi-select','ROI selection tools']]
  },
  missed_neuron_search: {
    label:'Find missed neuron',
    description:'Search the full field, draw a free-form ROI, and add event windows without treating unlabeled pixels as negative.',
    queue:'all',
    decisionTarget:null,
    decisions:[],
    actions:[['roi-lasso','Draw free-form ROI'],['roi-circle','Draw circular ROI'],['roi-select','Select or deselect ROI']]
  },
  event_validation: {
    label:'Validate event',
    description:'Decide whether the selected temporal event is present, absent, or uncertain.',
    eventQueue:'unlabeled',
    decisionTarget:'event',
    decisions:[['accept','Event'],['reject','No event'],['unsure','Unsure']],
    actions:[['next-event','Next event']]
  },
  artifact_resolution: {
    label:'Resolve artifact',
    description:'Classify the selected candidate and refine its shape when the footprint is wrong.',
    queue:'artifactLike',
    decisionTarget:'roi',
    decisions:[['reject','Artifact'],['accept','Clean'],['unsure','Unsure']],
    actions:[['roi-add','Add to ROI'],['roi-erase','Remove from ROI'],['roi-lasso','Redraw free-form ROI']]
  },
  exhaustive_tile: {
    label:'Exhaustive tile',
    description:'Review every candidate in the current queue. Unreviewed space remains unknown.',
    queue:'all',
    decisionTarget:'roi',
    decisions:[['accept','Neuron'],['reject','Not neuron'],['unsure','Unsure']],
    actions:[['roi-select','Select or deselect ROI'],['roi-lasso','Add free-form ROI']]
  },
  signal_background: {
    label:'Signal / background',
    description:'Mark CFAR foreground and background with brush or bounded flood selection. This task does not assign neuron identity.',
    queue:'all',
    decisionTarget:null,
    decisions:[],
    actions:[
      ['foreground-brush','Foreground brush'],
      ['background-brush','Background brush'],
      ['foreground-flood','Flood select'],
      ['background-flood-erase','Flood deselect']
    ]
  }
};

let datasetPageRequest = 0;
let resultsPageRequest = 0;
let annotationTaskDisclosureTask = null;

function annotationTask(){
  const task = String(setting?.('annotationTask') || 'neuron_validation');
  return ANNOTATION_TASK_DEFS[task] ? task : 'neuron_validation';
}

function researchToolsEnabled(){
  return Boolean(setting?.('researchToolsEnabled'));
}

function setAnnotationTask(task){
  const def = ANNOTATION_TASK_DEFS[task];
  if(!def) return;
  annotations.settings.annotationTask = task;
  annotations.settings.reviewWorkflowPreset = 'custom';
  if(def.queue) annotations.settings.queue = def.queue;
  if(def.eventQueue) annotations.settings.eventQueue = def.eventQueue;
  if(typeof setAnnotationToolModes === 'function') {
    setAnnotationToolModes({manualRoiMode:'select', roiEditMode:'off', cfarMaskTool:'off'}, {render:false});
  } else queueSave();
  renderAll();
}

function taskDecisionState(def){
  if(def.decisionTarget === 'event'){
    const roi = selectedRoi?.();
    return roi && selectedEventFrame ? String(eventAnn(roi.id, selectedEventFrame).state || '') : '';
  }
  if(def.decisionTarget === 'roi'){
    const roi = selectedRoi?.();
    return roi ? String(roiAnn(roi.id).state || '') : '';
  }
  return '';
}

function applyTaskDecision(def, decision){
  if(def.decisionTarget === 'event' && typeof setEventState === 'function') setEventState(decision);
  else if(def.decisionTarget === 'roi' && typeof setRoiState === 'function') setRoiState(decision);
}

function saveAndAdvanceTask(def){
  if(typeof queueSave === 'function') queueSave();
  if(def.decisionTarget === 'event' && typeof nextEventQueue === 'function') nextEventQueue(1);
  else if(typeof nextRoi === 'function') nextRoi(1);
}

function normalQueueChoices(def){
  if(def.decisionTarget === 'event') return [
    [def.eventQueue || 'unlabeled','Next'],
    ['unsure','Needs attention'],
    ['reviewed','Reviewed'],
    ['all','All']
  ];
  const nextQueue = def.queue && def.queue !== 'all' ? def.queue : 'unlabeled';
  return [
    [nextQueue,'Next'],
    ['needsAction','Needs attention'],
    ['reviewed','Reviewed'],
    ['all','All']
  ];
}

function setNormalQueue(def, value){
  setSetting(def.decisionTarget === 'event' ? 'eventQueue' : 'queue', value);
  if(typeof renderAll === 'function') renderAll();
}

function openToolPanel(id){
  const panel = document.getElementById(id);
  if(!panel) return;
  panel.classList.remove('hidden');
  panel.open = true;
  panel.scrollIntoView?.({block:'nearest', behavior:'smooth'});
}

function launchTaskTool(action){
  if(action === 'next-event'){
    if(typeof nextEventQueue === 'function') nextEventQueue(1);
    return;
  }
  if(typeof activateAnnotationTool === 'function') activateAnnotationTool(action);
}

function setTaskContextVisibility(task, def){
  for(const id of Object.keys(ANNOTATION_TASK_DEFS)) appRoot.classList.toggle(`task-${id}`, id === task);
  appRoot.classList.toggle('research-context-enabled', researchToolsEnabled());
  const taskChanged = annotationTaskDisclosureTask !== task;

  const roiPanel = document.getElementById('roiAnnotationPanel');
  if(roiPanel){
    roiPanel.classList.remove('hidden');
    if(taskChanged) roiPanel.open = ['missed_neuron_search','artifact_resolution','exhaustive_tile','signal_background'].includes(task);
  }
  const cfarPanel = document.getElementById('cfarMaskAnnotationPanel');
  if(cfarPanel){
    cfarPanel.classList.toggle('hidden', task !== 'signal_background' && !researchToolsEnabled());
    if(taskChanged) cfarPanel.open = task === 'signal_background';
  }

  const roiRail = document.getElementById('roiReviewRail');
  const eventRail = document.getElementById('eventReviewRail');
  const queueRail = document.getElementById('reviewQueueRail');
  if(roiRail){
    roiRail.classList.toggle('hidden', task === 'event_validation');
    if(taskChanged && task !== 'event_validation') roiRail.open = true;
  }
  if(eventRail){
    const showEvents = task === 'event_validation' || task === 'missed_neuron_search';
    eventRail.classList.toggle('hidden', !showEvents);
    if(taskChanged) eventRail.open = showEvents;
  }
  if(queueRail) queueRail.classList.remove('hidden');
  annotationTaskDisclosureTask = task;
}

function renderAnnotationTaskShell(){
  const root = document.getElementById('annotationTaskShell');
  if(!root) return;
  const hash = (location.hash || '#datasets').replace(/^#\/?/, '');
  const visible = ['annotate','review','review-stencil','candidate-overlay','review-triage'].includes(hash);
  root.classList.toggle('hidden', !visible);
  if(!visible) return;

  const task = annotationTask();
  const def = ANNOTATION_TASK_DEFS[task];
  const state = taskDecisionState(def);
  const queueKey = def.decisionTarget === 'event' ? 'eventQueue' : 'queue';
  const queueChoices = normalQueueChoices(def);
  const configuredQueueValue = String(setting(queueKey) || '');
  const queueValue = queueChoices.some(([value]) => value === configuredQueueValue)
    ? configuredQueueValue
    : queueChoices[0][0];
  const decisions = def.decisions.length ? `
    <div class="annotationDecisionRow" role="group" aria-label="${escapeHtml(def.label)} decision">
      ${def.decisions.map(([value, label]) => `<button type="button" data-task-decision="${value}" aria-pressed="${state === value}">${escapeHtml(label)}</button>`).join('')}
      <button type="button" class="decisionSaveNext" data-task-save-next>Save + Next</button>
    </div>` : '';
  const actions = def.actions.length ? `<div class="taskActionRow">${def.actions.map(([value, label]) => `<button type="button" data-task-tool="${value}">${escapeHtml(label)}</button>`).join('')}</div>` : '';
  const context = task === 'signal_background'
    ? 'Foreground/background masks are stored separately from neuron identity labels.'
    : researchToolsEnabled()
      ? 'Contextual research controls are enabled; the original advanced workspace remains under Research Tools.'
      : 'Detector and pipeline controls remain under Research Tools.';

  root.innerHTML = `
    <div class="annotationTaskHeader">
      <div><span class="eyebrow">Current annotation task</span><h2>${escapeHtml(def.label)}</h2><p class="hint">${escapeHtml(def.description)}</p></div>
      <label class="taskSelectLabel">Task
        <select id="annotationTaskSelect">${Object.entries(ANNOTATION_TASK_DEFS).map(([id, item]) => `<option value="${id}"${id === task ? ' selected' : ''}>${escapeHtml(item.label)}</option>`).join('')}</select>
      </label>
      <label class="taskSelectLabel">Queue
        <select id="normalQueueSelect">${queueChoices.map(([value, label]) => `<option value="${value}"${value === queueValue ? ' selected' : ''}>${escapeHtml(label)}</option>`).join('')}</select>
      </label>
    </div>
    ${decisions}${actions}
    <div class="taskContextLine"><span>${escapeHtml(def.eventQueue || def.queue || 'manual')} queue</span><span>${escapeHtml(context)}</span></div>`;

  root.querySelector('#annotationTaskSelect').onchange = event => setAnnotationTask(event.target.value);
  root.querySelector('#normalQueueSelect').onchange = event => setNormalQueue(def, event.target.value);
  for(const button of root.querySelectorAll('[data-task-decision]')) button.onclick = () => applyTaskDecision(def, button.dataset.taskDecision);
  root.querySelector('[data-task-save-next]')?.addEventListener('click', () => saveAndAdvanceTask(def));
  for(const button of root.querySelectorAll('[data-task-tool]')) button.onclick = () => launchTaskTool(button.dataset.taskTool);
  setTaskContextVisibility(task, def);
}

function productUrl(value){
  const text = String(value || '').trim();
  if(!text) return '';
  if(/^(?:https?:)?\/\//i.test(text) || text.startsWith('/') || text.startsWith('file:')) return text;
  return `/${text.replace(/^\.\//, '')}`;
}

function safeDatasetId(value, fallback='dataset'){
  const normalized = String(value || '').trim().toLowerCase().replace(/[^a-z0-9._-]+/g, '_').replace(/^[_\-.]+|[_\-.]+$/g, '');
  return normalized || fallback;
}

function datasetApiBase(dataset){
  return productUrl(dataset?.links?.api_base || `api/datasets/${encodeURIComponent(dataset?.dataset_id || datasetId)}`);
}

function datasetImportActionUrl(dataset, importIdValue, action){
  const template = dataset?.links?.import_action_template;
  if(template) return productUrl(template.replace('{import_id}', encodeURIComponent(importIdValue)).replace('{action}', encodeURIComponent(action)));
  return `${datasetApiBase(dataset)}/imports/${encodeURIComponent(importIdValue)}/${encodeURIComponent(action)}`;
}

function datasetLabelActionUrl(dataset, action){
  return `${productUrl(dataset?.links?.labels || `${datasetApiBase(dataset)}/labels`)}/${encodeURIComponent(action)}`;
}

function datasetNeuRevActionUrl(dataset, action){
  return `${productUrl(dataset?.links?.neurev || `${datasetApiBase(dataset)}/neurev`)}/${encodeURIComponent(action)}`;
}

async function productRequest(url, {method='POST', json=null, binary=null}={}){
  const headers = generationHeaders();
  let body = null;
  if(binary !== null){
    headers['Content-Type'] = 'application/octet-stream';
    body = binary;
  } else if(json !== null) body = JSON.stringify(json);
  const response = await fetch(productUrl(url), {method, headers, body, cache:'no-store'});
  const text = await response.text();
  let payload = {};
  try { payload = text ? JSON.parse(text) : {}; } catch (_) { payload = {error:text}; }
  if(!response.ok) throw new Error(payload.error || `${method} ${url} failed`);
  return payload;
}

function updateOwnerAuthControls(){
  const required = Boolean(generationEnvironment?.owner_token_required);
  for(const button of document.querySelectorAll('.ownerAuthControl')){
    button.classList.toggle('hidden', !required);
    button.textContent = generationOwnerToken ? 'Changes unlocked' : 'Unlock changes';
  }
}

function promptForOwnerToken(){
  const token = prompt('Owner token for local NeuRev changes');
  if(token === null) return false;
  generationOwnerToken = token.trim();
  if(generationOwnerToken) localStorage.setItem(ownerTokenKey, generationOwnerToken);
  else localStorage.removeItem(ownerTokenKey);
  updateOwnerAuthControls();
  if(typeof renderRunSyncControls === 'function') renderRunSyncControls();
  return Boolean(generationOwnerToken);
}

function datasetDisplayState(dataset){
  return dataset?.lifecycle?.state || (dataset?.readiness?.review_ready ? 'ready' : dataset?.readiness?.video_ready ? 'import_only' : 'unavailable');
}

function datasetCard(dataset){
  const video = dataset?.video || {};
  const states = dataset?.capability_states || {};
  const rate = video.frame_rate_hz ?? video.frameRateHz;
  const metadata = [
    video.frames ? `${video.frames} frames` : 'frame count unknown',
    video.width && video.height ? `${video.width} × ${video.height}` : 'dimensions unknown',
    rate ? `${rate} Hz` : 'frame rate unknown'
  ];
  const imports = Array.isArray(dataset?.imports) ? dataset.imports : [];
  const latest = imports.length ? imports[imports.length - 1] : null;
  const latestOfKind = kind => [...imports].reverse().find(item => item?.kind === kind) || null;
  const latestVideo = latestOfKind('video');
  const latestLabels = latestOfKind('label_table');
  const latestNeuRev = latestOfKind('neurev_json');
  const videoImport = latestVideo || (latest?.kind === 'video' ? latest : null);
  const latestId = videoImport?.import_id || '';
  const hasQc = Boolean(videoImport?.has_qc);
  const isPrimaryVideo = Boolean(videoImport?.is_primary_video || videoImport?.source_role === 'primary_video');
  let actions = '';
  if(videoImport && !isPrimaryVideo && !['processing','ready','complete'].includes(videoImport.state)) actions += `<button type="button" data-import-action="promote" data-import-id="${escapeHtml(latestId)}">Use as primary video</button>`;
  if(videoImport && isPrimaryVideo && !hasQc && !['processing','ready','complete'].includes(videoImport.state)) actions += `<button type="button" data-import-action="qc" data-import-id="${escapeHtml(latestId)}">Run bounded QC</button>`;
  if(videoImport && isPrimaryVideo && hasQc && !dataset?.readiness?.review_ready && !['processing','ready','complete'].includes(videoImport.state)) actions += `<button type="button" data-import-action="process" data-import-id="${escapeHtml(latestId)}">Prepare manual annotation</button>`;
  if(latestLabels?.state === 'qc_ready') actions += `<button type="button" data-label-action="preview" data-import-id="${escapeHtml(latestLabels.import_id || '')}">Preview and map labels</button>`;
  if(latestNeuRev?.state === 'qc_ready') actions += `<button type="button" data-neurev-action="preview" data-import-id="${escapeHtml(latestNeuRev.import_id || '')}">Preview NeuRev JSON</button>`;
  const metadataForm = videoImport?.state === 'metadata_needed' ? `
    <details class="datasetMetadataForm"><summary>Resolve acquisition metadata</summary><div class="toolbar">
      <label>Frame rate Hz <input data-meta="frame_rate_hz" type="number" min="0" step="any"></label>
      <label>Pixel size µm <input data-meta="pixel_size_microns" type="number" min="0" step="any"></label>
      <label>Modality <input data-meta="modality" type="text"></label>
      <label>Indicator <input data-meta="indicator" type="text"></label>
      <button type="button" data-import-action="metadata" data-import-id="${escapeHtml(latestId)}">Save supplied metadata</button>
    </div><p class="hint">Leave unknown scientific fields blank; NeuRev will not invent them.</p></details>` : '';
  const annotateUrl = productUrl(dataset?.links?.annotate || `/_datasets/${encodeURIComponent(dataset?.dataset_id || '')}/#annotate`);
  const openButton = `<button type="button" data-open-dataset="${escapeHtml(dataset?.dataset_id || '')}" data-annotate-url="${escapeHtml(annotateUrl)}" ${states.annotate !== 'ready' ? 'disabled' : ''}>Open Annotate</button>`;
  const neurevStatus = latestNeuRev ? `<p class="hint">NeuRev JSON · ${escapeHtml(String(latestNeuRev.payload_kind || 'unknown').replace(/_/g, ' '))} · ${latestNeuRev.state === 'complete' ? 'confirmed external copy' : 'confirmation required'}</p>` : '';
  return `<article class="datasetCard" data-dataset-id="${escapeHtml(dataset?.dataset_id || '')}">
    <div class="datasetCardHeader"><div><h2>${escapeHtml(dataset?.name || dataset?.dataset_id || 'Dataset')}</h2><p class="hint"><code>${escapeHtml(dataset?.dataset_id || '')}</code></p></div><span class="lifecycleBadge lifecycle-${escapeHtml(datasetDisplayState(dataset))}">${escapeHtml(datasetDisplayState(dataset).replace(/_/g, ' '))}</span></div>
    <p>${metadata.map(escapeHtml).join(' · ')}</p>
    <div class="capabilityChips"><span>Annotate: ${escapeHtml(states.annotate || 'unavailable')}</span><span>Results: ${escapeHtml(states.results || 'unavailable')}</span><span>Raw video: ${escapeHtml(states.raw_video || 'unavailable')}</span></div>
    ${metadataForm}${neurevStatus}<div class="buttonRow">${openButton}${actions}</div>
  </article>`;
}

async function postImportAction(dataset, action, importIdValue, payload={}){
  return productRequest(datasetImportActionUrl(dataset, importIdValue, action), {json:payload});
}

async function postLabelAction(dataset, action, importIdValue, payload={}){
  return productRequest(datasetLabelActionUrl(dataset, action), {json:{import_id:importIdValue, ...payload}});
}

async function postNeuRevAction(dataset, action, importIdValue, payload={}){
  return productRequest(datasetNeuRevActionUrl(dataset, action), {json:{import_id:importIdValue, ...payload}});
}

async function waitForDatasetJob(job, status, dataset=null){
  const jobIdValue = typeof job === 'string' ? job : job?.job_id;
  const fallback = dataset
    ? `${datasetApiBase(dataset)}/durable-jobs/${encodeURIComponent(jobIdValue || '')}`
    : `api/durable-jobs/${encodeURIComponent(jobIdValue || '')}`;
  const statusUrl = productUrl(job?.links?.self || job?.status_url || fallback);
  for(let attempt = 0; attempt < 120; attempt += 1){
    const response = await fetch(statusUrl, {cache:'no-store'});
    if(response.ok){
      const record = await response.json();
      status.textContent = `${record.kind || 'job'}: ${record.stage || record.status} (${Math.round((record.progress || 0) * 100)}%)`;
      if(['completed','failed','stopped'].includes(record.status)){
        if(record.status !== 'completed') throw new Error(record.error || `job ${record.status}`);
        return record;
      }
    }
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  throw new Error('Job did not finish within the local wait window. It remains durable and can be checked after refresh.');
}

function labelPreviewTable(rows, columns){
  if(!rows.length) return '<p class="hint">The table has no preview rows.</p>';
  return `<div class="tableScroll"><table class="smallTable"><thead><tr>${columns.map(column => `<th>${escapeHtml(column)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${columns.map(column => `<td>${escapeHtml(row?.[column] ?? '')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function renderLabelPreview(dataset, importIdValue, preview){
  const root = document.getElementById('labelPreviewPanel');
  if(!root) return;
  const rows = Array.isArray(preview?.sample_rows) ? preview.sample_rows : [];
  const columns = Array.isArray(preview?.columns) && preview.columns.length ? preview.columns : [...new Set(rows.flatMap(row => Object.keys(row || {})))];
  const inferred = preview?.label_mapping || {};
  const fields = [
    ['roi_id','ROI identity'],['x','X / column'],['y','Y / row'],
    ['start_frame','Start frame (UI, one-based)'],['end_frame','End frame (UI, inclusive)'],
    ['label','Label'],['confidence','Confidence']
  ];
  const optionHtml = (field) => `<option value="">Not mapped</option>${columns.map(column => `<option value="${escapeHtml(column)}"${inferred[field] === column || inferred[`${field}_ui`] === column ? ' selected' : ''}>${escapeHtml(column)}</option>`).join('')}`;
  root.innerHTML = `<section class="datasetImportCard labelPreviewPanel">
    <div class="sectionTitle"><div><span class="eyebrow">Confirmation required</span><h2>Map label columns</h2><p class="hint">${escapeHtml(preview.row_count ?? rows.length)} source rows. Previewing does not change annotations.</p></div><button type="button" data-close-label-preview>Close</button></div>
    <div class="labelMappingGrid">${fields.map(([field,label]) => `<label>${escapeHtml(label)}<select data-label-field="${field}">${optionHtml(field)}</select></label>`).join('')}</div>
    ${labelPreviewTable(rows, columns)}
    <p class="hint">Coordinates use x=column, y=row. UI frames are one-based and inclusive. Unmatched rows remain distinct and are not converted to negatives.</p>
    <button type="button" data-confirm-label-import>Confirm mapping and import</button>
  </section>`;
  root.querySelector('[data-close-label-preview]').onclick = () => { root.innerHTML = ''; };
  root.querySelector('[data-confirm-label-import]').onclick = async () => {
    const status = document.getElementById('datasetImportStatus');
    const labelMapping = {};
    for(const select of root.querySelectorAll('[data-label-field]')) if(select.value) labelMapping[select.dataset.labelField] = select.value;
    status.textContent = 'Importing confirmed label mapping…';
    try {
      const result = await postLabelAction(dataset, 'import', importIdValue, {label_mapping:labelMapping, confirmed:true});
      if(result.job) await waitForDatasetJob(result.job, status, dataset);
      status.textContent = 'Labels imported as a separate reconciliation artifact; native annotations were not replaced.';
      root.innerHTML = '';
      renderDatasetsPage();
    } catch(error) { status.textContent = error.message; }
  };
}

function renderNeuRevPreview(dataset, importIdValue, preview){
  const root = document.getElementById('labelPreviewPanel');
  if(!root) return;
  const counts = Object.entries(preview?.counts || {});
  const countMarkup = counts.length
    ? `<dl class="datasetMetadataList">${counts.map(([key,value]) => `<div><dt>${escapeHtml(key.replace(/_/g, ' '))}</dt><dd>${escapeHtml(value)}</dd></div>`).join('')}</dl>`
    : '<p class="hint">This document type has no count summary.</p>';
  root.innerHTML = `<section class="datasetImportCard labelPreviewPanel">
    <div class="sectionTitle"><div><span class="eyebrow">Confirmation required</span><h2>Review NeuRev JSON</h2><p class="hint">${escapeHtml(preview?.source?.original_name || 'NeuRev JSON')} was recognized as ${escapeHtml(String(preview?.payload_kind || '').replace(/_/g, ' '))}.</p></div><button type="button" data-close-neurev-preview>Close</button></div>
    ${countMarkup}
    <p class="hint">Declared dataset: ${escapeHtml(preview?.declared_dataset_id || 'not declared')} · SHA-256: <code>${escapeHtml(preview?.source?.checksum?.sha256 || 'unavailable')}</code></p>
    <p class="hint">Confirmation publishes an exact external copy under this dataset. Native review data, annotations, and run manifests are not merged or replaced, and declared paths are not followed.</p>
    <button type="button" data-confirm-neurev-import>Confirm external NeuRev JSON</button>
  </section>`;
  root.querySelector('[data-close-neurev-preview]').onclick = () => { root.innerHTML = ''; };
  root.querySelector('[data-confirm-neurev-import]').onclick = async () => {
    const status = document.getElementById('datasetImportStatus');
    status.textContent = 'Publishing the confirmed lossless NeuRev JSON copy…';
    try {
      const result = await postNeuRevAction(dataset, 'import', importIdValue, {confirmed:true});
      if(result.job) await waitForDatasetJob(result.job, status, dataset);
      status.textContent = 'NeuRev JSON confirmed as a separate lossless artifact; native app state was not replaced.';
      root.innerHTML = '';
      renderDatasetsPage();
    } catch(error) { status.textContent = error.message; }
  };
}

function bindDatasetActions(grid, datasets){
  const status = document.getElementById('datasetImportStatus');
  const byId = new Map(datasets.map(dataset => [String(dataset.dataset_id), dataset]));
  for(const button of grid.querySelectorAll('[data-open-dataset]')) button.onclick = () => location.assign(button.dataset.annotateUrl);
  for(const button of grid.querySelectorAll('[data-import-action]')) button.onclick = async () => {
    const dataset = byId.get(String(button.closest('[data-dataset-id]')?.dataset.datasetId || ''));
    if(!dataset) return;
    try {
      const action = button.dataset.importAction;
      const importIdValue = button.dataset.importId;
      const payload = {};
      if(action === 'metadata'){
        for(const field of button.closest('.datasetCard').querySelectorAll('[data-meta]')){
          const value = field.value.trim();
          if(value) payload[field.dataset.meta] = ['frame_rate_hz','pixel_size_microns'].includes(field.dataset.meta) ? Number(value) : value;
        }
      }
      status.textContent = `${action}: starting…`;
      const result = await postImportAction(dataset, action, importIdValue, payload);
      if(result.job) await waitForDatasetJob(result.job, status, dataset);
      status.textContent = action === 'process' ? 'Manual annotation app prepared.' : action === 'promote' ? 'Primary video selected; run bounded QC next.' : `${action}: complete.`;
      renderDatasetsPage();
    } catch(error) { status.textContent = error.message; }
  };
  for(const button of grid.querySelectorAll('[data-label-action]')) button.onclick = async () => {
    const dataset = byId.get(String(button.closest('[data-dataset-id]')?.dataset.datasetId || ''));
    if(!dataset) return;
    try {
      status.textContent = 'Reading a bounded label preview…';
      const result = await postLabelAction(dataset, 'preview', button.dataset.importId);
      status.textContent = `Previewed ${result.sample_rows?.length || 0} of ${result.row_count || 0} rows. Confirm the mapping below.`;
      renderLabelPreview(dataset, button.dataset.importId, result);
    } catch(error) { status.textContent = error.message; }
  };
  for(const button of grid.querySelectorAll('[data-neurev-action]')) button.onclick = async () => {
    const dataset = byId.get(String(button.closest('[data-dataset-id]')?.dataset.datasetId || ''));
    if(!dataset) return;
    try {
      status.textContent = 'Reading a bounded NeuRev JSON preview…';
      const result = await postNeuRevAction(dataset, 'preview', button.dataset.importId);
      status.textContent = `Recognized ${String(result.payload_kind || '').replace(/_/g, ' ')}. Confirm the external copy below.`;
      renderNeuRevPreview(dataset, button.dataset.importId, result);
    } catch(error) { status.textContent = error.message; }
  };
}

function renderDatasetsPage(){
  const root = document.getElementById('datasetsPageBody');
  if(!root) return;
  const request = ++datasetPageRequest;
  root.innerHTML = `<div class="productIntro"><div><span class="eyebrow">Normal workflow</span><h1>Datasets</h1><p>Add a source, resolve only trusted metadata, run bounded QC, then prepare manual annotation.</p></div><div class="buttonRow"><button type="button" id="unlockDatasetChangesBtn" class="ownerAuthControl hidden">Unlock changes</button><span class="stageStatus ok">local-first</span></div></div>
    <section class="datasetImportCard"><h2>Add data</h2>
      <div class="datasetIntakeSteps"><div class="datasetIntakeStep"><b>1 · Source</b><span>Register an existing local file or upload a bounded file.</span></div><div class="datasetIntakeStep"><b>2 · Inspect</b><span>Confirm unknown metadata and run sampled QC.</span></div><div class="datasetIntakeStep"><b>3 · Prepare</b><span>Render a manual annotation app without claiming detector results.</span></div></div>
      <div class="toolbar"><label>Dataset ID <input id="newDatasetId" class="datasetIdField" type="text" placeholder="optional; derived from filename"></label><label>Local file <input id="localRegisterPath" type="text" placeholder="Inputs/my_dataset/movie.npy"></label><button type="button" id="registerLocalBtn">Register local file</button><label class="fileButton">Upload <input id="uploadDatasetInput" type="file" accept=".npy,.tif,.tiff,.csv,.tsv,.xlsx,.json"></label></div>
      <p id="datasetImportStatus" class="hint">Use an existing dataset ID when attaching a label table or native NeuRev JSON. JSON requires a bounded preview and explicit confirmation. Scientific metadata remains unknown until supplied or read from trusted metadata.</p>
    </section><div id="labelPreviewPanel"></div>
    <section><div class="sectionTitle"><h2>Dataset catalog</h2><button type="button" id="refreshDatasetsBtn">Refresh</button></div><div id="datasetCatalogGrid" class="datasetCatalogGrid"><p class="hint">Loading catalog…</p></div></section>`;
  const fallback = [{dataset_id:datasetId, name:data.dataset?.name || data.video?.name || datasetId, video:{frames:data.video?.frames, width:data.video?.width, height:data.video?.height, frame_rate_hz:data.video?.frameRateHz}, readiness:{review_ready:true, scientific_results_ready:false}, capability_states:{annotate:'ready',results:'unavailable',raw_video:'ready'}, links:{annotate:'#annotate'}}];
  const load = serverBacked ? fetch(productUrl('api/datasets'), {cache:'no-store'}).then(response => response.ok ? response.json() : ({datasets:fallback})).catch(() => ({datasets:fallback})) : Promise.resolve({datasets:fallback});
  load.then(payload => {
    if(request !== datasetPageRequest) return;
    const grid = document.getElementById('datasetCatalogGrid');
    if(!grid) return;
    const datasets = Array.isArray(payload?.datasets) && payload.datasets.length ? payload.datasets : fallback;
    grid.innerHTML = datasets.map(datasetCard).join('');
    bindDatasetActions(grid, datasets);
  });
  document.getElementById('refreshDatasetsBtn').onclick = renderDatasetsPage;
  document.getElementById('unlockDatasetChangesBtn').onclick = promptForOwnerToken;
  updateOwnerAuthControls();
  document.getElementById('registerLocalBtn').onclick = async () => {
    const field = document.getElementById('localRegisterPath');
    const status = document.getElementById('datasetImportStatus');
    if(!serverBacked) { status.textContent = 'Run the local workbench server to register files safely.'; return; }
    if(!field.value.trim()) { status.textContent = 'Enter a path inside Inputs/ or Outputs/.'; return; }
    const inferred = field.value.trim().split(/[\\/]/).pop().replace(/\.[^.]+$/, '');
    const targetId = safeDatasetId(document.getElementById('newDatasetId').value, safeDatasetId(inferred));
    status.textContent = 'Registering and inspecting…';
    try {
      const payload = await productRequest(`api/datasets/${encodeURIComponent(targetId)}/imports/register`, {json:{dataset_id:targetId, source_path:field.value.trim()}});
      status.textContent = `Registered ${payload.import.original_name}; next state: ${payload.import.state}.`;
      renderDatasetsPage();
    } catch(error) { status.textContent = error.message; }
  };
  document.getElementById('uploadDatasetInput').onchange = async event => {
    const file = event.target.files?.[0];
    const status = document.getElementById('datasetImportStatus');
    if(!file) return;
    if(!serverBacked) { status.textContent = 'Run the local workbench server to stream uploads safely.'; return; }
    const inferred = file.name.replace(/\.[^.]+$/, '');
    const targetId = safeDatasetId(document.getElementById('newDatasetId').value, safeDatasetId(inferred));
    status.textContent = `Uploading ${file.name}…`;
    try {
      const endpoint = `api/datasets/${encodeURIComponent(targetId)}/imports/upload?filename=${encodeURIComponent(file.name)}`;
      const payload = await productRequest(endpoint, {binary:file});
      status.textContent = `Uploaded ${file.name}; next state: ${payload.import.state}.`;
      renderDatasetsPage();
    } catch(error) { status.textContent = error.message; }
  };
}

function resultExport(action){
  if(action === 'roi') exportRows('roi');
  else if(action === 'event') exportRows('event');
  else if(action === 'suggestion') exportRows('suggestion');
  else if(action === 'split-merge') exportRows('splitMerge');
  else if(action === 'annotations') exportJson();
  else if(action === 'provenance') exportReviewerProvenanceAudit();
  else if(action === 'handoff-json') downloadJson(`${datasetId}_review_handoff.json`, reviewSessionHandoff());
  else if(action === 'handoff-md') downloadText(`${datasetId}_review_handoff.md`, reviewSessionHandoffMarkdown(), 'text/markdown');
  else if(action === 'report-md') downloadText(`${datasetId}_review_report.md`, reviewReportMarkdown(), 'text/markdown');
}

function resultsMarkup(record, jobsPayload){
  const summary = annotationSummary();
  const reviewedRois = summary.review_progress?.reviewed_rois || 0;
  const reviewedEvents = summary.review_progress?.reviewed_events || 0;
  const roiTotal = summary.roi_count || 0;
  const eventTotal = summary.event_count || 0;
  const imports = Array.isArray(record?.imports) ? record.imports : [];
  const durableJobs = Array.isArray(jobsPayload?.durable_jobs) ? jobsPayload.durable_jobs : [];
  const latestDurableJob = durableJobs.length ? durableJobs[durableJobs.length - 1] : null;
  const runs = Array.isArray(data.architectureRuns?.runs) ? data.architectureRuns.runs : [];
  const completedRuns = runs.filter(run => run?.execution?.status === 'completed').length;
  const scientificReady = Boolean(record?.readiness?.scientific_results_ready);
  return `<div class="productIntro"><div><span class="eyebrow">Normal workflow</span><h1>Results</h1><p>Annotation progress, import provenance, durable jobs, and scientific-run availability in one place.</p></div><span class="stageStatus ${scientificReady ? 'ok' : 'warn'}">${scientificReady ? 'scientific artifacts available' : 'annotation results only'}</span></div>
    <div class="reportExportSummary"><div class="metric"><b>${reviewedRois} / ${roiTotal}</b><span>reviewed ROIs</span></div><div class="metric"><b>${reviewedEvents} / ${eventTotal}</b><span>reviewed events</span></div><div class="metric"><b>${summary.roi_states?.accepted || 0}</b><span>accepted neurons</span></div><div class="metric"><b>${summary.event_states?.accepted || 0}</b><span>accepted events</span></div></div>
    <div class="resultsGrid">
      <section class="resultsCard"><h2>Annotation state</h2>${reviewedRois || reviewedEvents ? `<p>${reviewedRois} ROI and ${reviewedEvents} event decisions are saved. Unlabeled candidates remain unknown.</p>` : '<p class="resultsEmpty">No annotation decisions have been saved yet.</p>'}<a href="#annotate">Continue annotation</a></section>
      <section class="resultsCard"><h2>Imports and jobs</h2><p>${imports.length} import record${imports.length === 1 ? '' : 's'} · ${durableJobs.length} durable job${durableJobs.length === 1 ? '' : 's'}</p>${latestDurableJob ? `<p class="hint">Latest job: ${escapeHtml(latestDurableJob.kind || 'job')} · ${escapeHtml(latestDurableJob.status || 'unknown')}</p>` : '<p class="resultsEmpty">No dataset preparation jobs are recorded for this app.</p>'}</section>
      <section class="resultsCard"><h2>Scientific runs</h2><p>${runs.length} attached run record${runs.length === 1 ? '' : 's'} · ${completedRuns} marked completed</p>${scientificReady ? '<p>At least one completed run has a declared scientific result artifact.</p>' : '<p class="resultsEmpty">No verified scientific result artifact is available. Frame preparation is not reported as detector success.</p>'}<a href="#research">Open Research Tools</a></section>
      <section class="resultsCard"><h2>Export</h2><div class="resultsExportRow"><label>Artifact<select id="resultsExportSelect"><option value="roi">ROI annotations TSV</option><option value="event">Event annotations TSV</option><option value="suggestion">Discovery suggestions TSV</option><option value="split-merge">Split / merge decisions TSV</option><option value="annotations">Annotations JSON</option><option value="provenance">Reviewer provenance JSON</option><option value="handoff-json">Review handoff JSON</option><option value="handoff-md">Review handoff Markdown</option><option value="report-md">Review report Markdown</option></select></label><button type="button" id="resultsExportBtn">Download</button></div><p class="hint">Exports preserve unknowns and keep imported-label reconciliation separate from native annotation decisions.</p></section>
    </div>`;
}

function renderResultsPage(record=null, jobsPayload=null){
  const root = document.getElementById('reportPageBody');
  if(!root) return;
  root.innerHTML = resultsMarkup(record || data.dataset || {}, jobsPayload || {});
  document.getElementById('resultsExportBtn').onclick = () => resultExport(document.getElementById('resultsExportSelect').value);
  if(record !== null || !serverBacked) return;
  const request = ++resultsPageRequest;
  const apiBase = productUrl(data.dataset?.links?.api_base || `api/datasets/${encodeURIComponent(datasetId)}`);
  Promise.all([
    fetch(apiBase, {cache:'no-store'}).then(response => response.ok ? response.json() : data.dataset || {}).catch(() => data.dataset || {}),
    fetch(`${apiBase}/jobs`, {cache:'no-store'}).then(response => response.ok ? response.json() : {}).catch(() => ({}))
  ]).then(([datasetRecord, jobs]) => {
    if(request !== resultsPageRequest || (location.hash || '').replace(/^#\/?/, '') !== 'results') return;
    renderResultsPage(datasetRecord, jobs);
  });
}

function renderResearchToolsPage(){
  const root = document.getElementById('researchPageBody');
  if(!root) return;
  root.innerHTML = `<div class="productIntro"><div><span class="eyebrow">Advanced workspace</span><h1>Research Tools</h1><p>All prior detector development, pipeline comparison, experiments, raw parameters, LLM planning, QC masks, and reports remain available here.</p></div><label class="researchToggle"><input id="researchToolsEnabledToggle" type="checkbox" ${researchToolsEnabled() ? 'checked' : ''}> Enable contextual research controls in Annotate</label></div><div class="researchToolGrid"><a href="#review-stencil" class="researchToolCard"><h2>Stencil Review</h2><p>Inspect anatomy stencil coverage and alignment.</p></a><a href="#candidate-overlay" class="researchToolCard"><h2>Candidate Overlay</h2><p>Compare candidate footprints and overlap diagnostics.</p></a><a href="#review-triage" class="researchToolCard"><h2>Review Triage</h2><p>Open the preserved advanced review queue and triage utilities.</p></a><a href="#data" class="researchToolCard"><h2>Data & QC</h2><p>Inspect projections, chunks, and process warnings.</p></a><a href="#pipelines" class="researchToolCard"><h2>Pipelines</h2><p>Compare and plan detector architectures.</p></a><a href="#experiments" class="researchToolCard"><h2>Experiment Lab</h2><p>Run bounded, auditable research jobs.</p></a><a href="#progress" class="researchToolCard"><h2>Progress & adjudication</h2><p>Review metrics, reviewer agreement, benchmark coverage, and adjudication.</p></a><a href="#report" class="researchToolCard"><h2>Detailed report</h2><p>Open the original scientific report and audit views.</p></a></div>`;
  document.getElementById('researchToolsEnabledToggle').onchange = event => {
    setSetting('researchToolsEnabled', Boolean(event.target.checked));
    appRoot.classList.toggle('research-context-enabled', Boolean(event.target.checked));
    renderAnnotationTaskShell();
  };
}
