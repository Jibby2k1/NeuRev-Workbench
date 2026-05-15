const embedded = document.getElementById('review-data');
let data = JSON.parse(embedded.textContent);
const appRoot = document.getElementById('appRoot');
const img = document.getElementById('frameImg');
const evidenceImg = document.getElementById('evidenceImg');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const slider = document.getElementById('frameSlider');
const frameLabel = document.getElementById('frameLabel');
const statusEl = document.getElementById('statusText');
const selectionText = document.getElementById('selectionText');
const saveStateEl = document.getElementById('saveState');
const traceCanvas = document.getElementById('traceCanvas');
const traceCtx = traceCanvas.getContext('2d');
const eventTimelineCanvas = document.getElementById('eventTimelineCanvas');
const eventTimelineCtx = eventTimelineCanvas?.getContext('2d');
const cropCanvas = document.getElementById('roiCropCanvas');
const cropCtx = cropCanvas?.getContext('2d');
const roiNotes = document.getElementById('roiNotes');
const eventNotes = document.getElementById('eventNotes');
const viewerScroll = document.getElementById('viewerScroll');
const viewerWrap = document.getElementById('viewerWrap');
const datasetId = data.dataset?.dataset_id || data.video?.name || 'calcium-video';
const storeKey = `neuron-review-workbench-v3-${datasetId}`;
const recoveryStoreKey = `${storeKey}-recovery-history`;
const traceCache = new Map();
const traceEventCache = new Map();
const TRACE_CACHE_LIMIT = 512;
const traceCacheStats = {traceHits:0, traceMisses:0, eventHits:0, eventMisses:0, clears:0, lastClearReason:''};
const TRACE_PAD = 30;
const annotationUndoStack = [];
const REASON_TAG_OPTIONS = ['compact', 'event_supported', 'clear_trace', 'low_snr', 'artifact_risk', 'duplicate', 'manual', 'needs_second_review'];
const OVERLAY_PRESETS = {
  validate: {
    label: 'Validate firing',
    selectedOverlayMode: 'outline',
    selectedFillOpacity: 0.10,
    selectedOutlineWidth: 2.5,
    overlayOpacity: 0.38,
    showLabels: true,
    showEvents: true,
    showSuggestions: true,
    showEvidence: false
  },
  dense: {
    label: 'Dense triage',
    selectedOverlayMode: 'soft',
    selectedFillOpacity: 0.32,
    selectedOutlineWidth: 2.0,
    overlayOpacity: 0.72,
    showLabels: true,
    showEvents: true,
    showSuggestions: true,
    showEvidence: false
  },
  discovery: {
    label: 'Discovery',
    selectedOverlayMode: 'event',
    selectedFillOpacity: 0.14,
    selectedOutlineWidth: 2.5,
    overlayOpacity: 0.28,
    showLabels: false,
    showEvents: true,
    showSuggestions: true,
    showEvidence: true
  }
};
const REVIEW_WORKFLOW_PRESETS = {
  fast_triage: {
    label: 'Fast triage',
    queue: 'annotationBatch',
    discoveryQueue: 'all',
    overlayPreset: 'validate',
    roiFocusMode: 'all',
    reviewMode: 'guided',
    showEvidence: false,
    showSuggestions: true,
    showLabels: true,
    showEvents: true,
    selectedOverlayMode: 'outline'
  },
  event_validation: {
    label: 'Event validation',
    queue: 'needsEventReview',
    discoveryQueue: 'all',
    overlayPreset: 'validate',
    roiFocusMode: 'neighbors',
    reviewMode: 'explore',
    showEvidence: false,
    showSuggestions: false,
    showLabels: true,
    showEvents: true,
    selectedOverlayMode: 'event'
  },
  missed_neuron_search: {
    label: 'Missed neuron search',
    queue: 'all',
    discoveryQueue: 'unlabeled',
    overlayPreset: 'discovery',
    roiFocusMode: 'all',
    reviewMode: 'explore',
    showEvidence: true,
    showSuggestions: true,
    showLabels: false,
    showEvents: true,
    selectedOverlayMode: 'outline'
  },
  artifact_cleanup: {
    label: 'Artifact cleanup',
    queue: 'artifactLike',
    discoveryQueue: 'artifactSuspects',
    overlayPreset: 'dense',
    roiFocusMode: 'all',
    reviewMode: 'explore',
    showEvidence: false,
    showSuggestions: true,
    showLabels: true,
    showEvents: true,
    selectedOverlayMode: 'soft'
  },
  mask_editing: {
    label: 'Mask editing',
    queue: 'needsAction',
    discoveryQueue: 'all',
    overlayPreset: 'validate',
    roiFocusMode: 'solo',
    reviewMode: 'explore',
    showEvidence: false,
    showSuggestions: false,
    showLabels: true,
    showEvents: false,
    selectedOverlayMode: 'outline',
    uiMode: 'advanced'
  }
};

let currentFrame = 1;
let selectedId = data.rois.length ? data.rois[0].id : null;
let selectedRoiIds = new Set(selectedId ? [String(selectedId)] : []);
let selectedEventFrame = null;
let selectedSuggestionId = data.discovery?.suggestions?.[0]?.id || null;
let playing = false;
let timer = null;
let qcTimer = null;
let saveTimer = null;
let serverBacked = location.protocol.startsWith('http');
let saveStatus = {text: 'loading', className: '', updatedAt: null};
const ownerTokenKey = `${storeKey}-owner-token`;
let generationOwnerToken = localStorage.getItem(ownerTokenKey) || '';
let annotations = defaultAnnotations();
let lastRecoverySnapshotAt = 0;
let generationEnvironment = null;
let currentGenerationJob = null;
let generationPollTimer = null;
const proposalAnalysisCache = new Map();
const reviewDataCache = new Map();
let traceView = {start: 1, end: Math.max(1, Number(data.video?.frames) || 1), dragging: false};
let manualRoiState = {drawing:false, start:null, points:[], preview:null, suppressClick:false};
let roiEditState = {drawing:false, editedId:null};

function architectureRuns(){ return data.architectureRuns?.runs || []; }
function baselineRunId(){ return architectureRuns()[0]?.run_id || 'current_review_pipeline'; }

function defaultAnnotations() {
  return {
    version: 3,
    schema_version: 3,
    updatedAt: new Date().toISOString(),
    rois: {},
    events: {},
    suggestions: {},
    promotedRois: {},
    virtualRois: {},
    splitMergeDecisions: {},
    bookmarks: [],
    runs: {},
    reviewStats: {
      sessionStartedAt: new Date().toISOString(),
      lastActionAt: null,
      actions: {}
    },
    settings: {
      eventThreshold: 2.4,
      kalmanGain: 0.06,
      spikeGain: 0.008,
      zoom: 3.0,
      brightness: 1,
      contrast: 1.08,
      overlayOpacity: 0.72,
      overlayPreset: 'validate',
      selectedOverlayMode: 'outline',
      selectedFillOpacity: 0.10,
      selectedOutlineWidth: 2.5,
      roiFocusMode: 'all',
      neighborRadiusPx: 36,
      uiMode: 'basic',
      theme: 'system',
      reviewerId: '',
      manualRoiMode: 'select',
      manualRoiRadius: 6,
      roiEditMode: 'off',
      roiEditBrushRadius: 4,
      reviewWorkflowPreset: 'custom',
      activeSnapshotId: '',
      parameterSnapshots: [],
      queue: 'unlabeled',
      eventQueue: 'all',
      discoveryQueue: 'all',
      evidenceMap: data.discovery?.evidenceMaps?.[0]?.id || '',
      showEvidence: false,
      showSuggestions: true,
      minArea: 0,
      minEvents: 0,
      reviewMode: 'explore',
      guidedTaskIndex: 0,
      targetRois: 30,
      targetEvents: 30,
      targetSuggestions: 15,
      activeRunId: baselineRunId(),
      reviewCompare: {
        enabled: false,
        runAId: baselineRunId(),
        runBId: architectureRuns()[1]?.run_id || baselineRunId()
      },
      qcRunId: data.architectureRuns?.runs?.[0]?.run_id || '',
      experimentLabels: {},
      qcEvidenceMap: data.discovery?.evidenceMaps?.[0]?.id || ''
    }
  };
}

function mergeAnnotations(incoming) {
  annotations = Object.assign(defaultAnnotations(), incoming || {});
  annotations.version = 3;
  annotations.schema_version = 3;
  annotations.rois = {};
  for(const [id, ann] of Object.entries(incoming?.rois || {})) annotations.rois[id] = migrateRoiAnn(ann);
  annotations.events = {};
  for(const [id, ann] of Object.entries(incoming?.events || {})) annotations.events[id] = migrateEventAnn(ann);
  annotations.suggestions = {};
  for(const [id, ann] of Object.entries(incoming?.suggestions || {})) annotations.suggestions[id] = migrateSuggestionAnn(ann);
  annotations.promotedRois = Object.assign({}, incoming?.promotedRois || {});
  annotations.virtualRois = Object.assign({}, incoming?.virtualRois || {});
  annotations.bookmarks = Array.isArray(incoming?.bookmarks) ? incoming.bookmarks.map(migrateBookmark).filter(Boolean) : [];
  annotations.splitMergeDecisions = {};
  for(const [id, ann] of Object.entries(incoming?.splitMergeDecisions || {})) annotations.splitMergeDecisions[id] = migrateSplitMergeDecision(ann);
  annotations.runs = {};
  for(const [runId, bucket] of Object.entries(incoming?.runs || {})) annotations.runs[runId] = migrateRunBucket(bucket);
  annotations.reviewStats = Object.assign(defaultAnnotations().reviewStats, incoming?.reviewStats || {});
  annotations.reviewStats.actions = Object.assign({}, incoming?.reviewStats?.actions || {});
  annotations.settings = Object.assign(defaultAnnotations().settings, incoming?.settings || {});
}

function migrateRunBucket(bucket) {
  const out = {
    rois: {},
    events: {},
    suggestions: {},
    promotedRois: Object.assign({}, bucket?.promotedRois || {}),
    virtualRois: Object.assign({}, bucket?.virtualRois || {}),
    splitMergeDecisions: {}
  };
  for(const [id, ann] of Object.entries(bucket?.rois || {})) out.rois[id] = migrateRoiAnn(ann);
  for(const [id, ann] of Object.entries(bucket?.events || {})) out.events[id] = migrateEventAnn(ann);
  for(const [id, ann] of Object.entries(bucket?.suggestions || {})) out.suggestions[id] = migrateSuggestionAnn(ann);
  for(const [id, ann] of Object.entries(bucket?.splitMergeDecisions || {})) out.splitMergeDecisions[id] = migrateSplitMergeDecision(ann);
  return out;
}

function migrateRoiAnn(ann) {
  const out = Object.assign({state:'', notes:'', deleted:false}, ann || {});
  if(!out.cell_state) out.cell_state = out.state === 'accept' ? 'accepted' : out.state === 'reject' ? 'rejected' : out.state === 'unsure' ? 'unsure' : '';
  if(out.cell_state && !out.state) out.state = out.cell_state === 'accepted' ? 'accept' : out.cell_state === 'rejected' ? 'reject' : out.cell_state === 'unsure' ? 'unsure' : '';
  out.trace_quality = out.trace_quality || '';
  out.control_ready = out.control_ready || '';
  out.artifact_class = out.artifact_class || out.artifactClass || '';
  out.identity_group = out.identity_group || '';
  out.needs_action = out.needs_action || '';
  out.reason_tags = normalizeIdList(out.reason_tags || out.reason_codes);
  out.confidence = ['low','medium','high'].includes(String(out.confidence || '').toLowerCase()) ? String(out.confidence).toLowerCase() : '';
  return out;
}

function migrateSuggestionAnn(ann) {
  const out = Object.assign({state:'', artifactClass:'', artifact_class:'', notes:''}, ann || {});
  out.artifact_class = out.artifact_class || out.artifactClass || '';
  out.reason_tags = normalizeIdList(out.reason_tags || out.reason_codes);
  out.confidence = ['low','medium','high'].includes(String(out.confidence || '').toLowerCase()) ? String(out.confidence).toLowerCase() : '';
  return out;
}

function migrateSplitMergeDecision(ann) {
  const out = Object.assign({id:'', decision_type:'', decision_state:'', source_roi_ids:[], target_roi_ids:[], virtual_roi_id:'', identity_group:'', needs_action:'', reason_tags:[], confidence:'', notes:''}, ann || {});
  out.decision_type = ['split','merge'].includes(String(out.decision_type || out.type || '').toLowerCase()) ? String(out.decision_type || out.type).toLowerCase() : '';
  out.decision_state = ['proposed','accepted','rejected','unsure'].includes(String(out.decision_state || out.state || '').toLowerCase()) ? String(out.decision_state || out.state).toLowerCase() : '';
  out.source_roi_ids = normalizeIdList(out.source_roi_ids || out.source_rois);
  out.target_roi_ids = normalizeIdList(out.target_roi_ids || out.target_rois);
  out.reason_tags = normalizeIdList(out.reason_tags || out.reason_codes);
  out.confidence = ['low','medium','high'].includes(String(out.confidence || '').toLowerCase()) ? String(out.confidence).toLowerCase() : '';
  return out;
}

function normalizeIdList(value) {
  if(value == null || value === '') return [];
  if(Array.isArray(value)) return value.map(v => String(v).trim()).filter(Boolean);
  return String(value).split(/[;,]/).map(v => v.trim()).filter(Boolean);
}

function migrateEventAnn(ann) {
  const out = Object.assign({state:'', notes:''}, ann || {});
  if(!out.event_state) out.event_state = out.state === 'accept' ? 'accepted' : out.state === 'reject' ? 'rejected' : out.state === 'unsure' ? 'unsure' : '';
  if(out.event_state && !out.state) out.state = out.event_state === 'accepted' ? 'accept' : out.event_state === 'rejected' ? 'reject' : out.event_state === 'unsure' ? 'unsure' : '';
  out.event_type = out.event_type || '';
  out.timing_quality = out.timing_quality || '';
  out.reason_tags = normalizeIdList(out.reason_tags || out.reason_codes);
  out.confidence = ['low','medium','high'].includes(String(out.confidence || '').toLowerCase()) ? String(out.confidence).toLowerCase() : '';
  return out;
}

function migrateBookmark(bookmark) {
  if(!bookmark || typeof bookmark !== 'object') return null;
  const out = Object.assign({
    id: `mark_${Date.now().toString(36)}`,
    label: '',
    createdAt: new Date().toISOString(),
    runId: '',
    frame: 1,
    roiId: '',
    eventFrame: null,
    suggestionId: ''
  }, bookmark);
  out.id = String(out.id || `mark_${Date.now().toString(36)}`);
  out.label = String(out.label || 'Review bookmark');
  out.runId = String(out.runId || '');
  out.roiId = out.roiId === null || out.roiId === undefined ? '' : String(out.roiId);
  out.suggestionId = out.suggestionId === null || out.suggestionId === undefined ? '' : String(out.suggestionId);
  out.frame = Math.max(1, Number(out.frame) || 1);
  out.eventFrame = out.eventFrame === null || out.eventFrame === undefined || out.eventFrame === '' ? null : Math.max(1, Number(out.eventFrame) || 1);
  return out;
}

function activeRunId(){ return setting('activeRunId') || baselineRunId(); }
function activeRun(){ return architectureRuns().find(r => r.run_id === activeRunId()) || architectureRuns()[0] || null; }
function reviewCompareSettings(){
  const runs = architectureRuns();
  annotations.settings.reviewCompare = Object.assign({
    enabled: false,
    runAId: activeRunId(),
    runBId: runs[1]?.run_id || runs[0]?.run_id || baselineRunId()
  }, annotations.settings.reviewCompare || {});
  return annotations.settings.reviewCompare;
}
function runAnnotationSnapshot(){
  return {
    rois: Object.assign({}, annotations.rois || {}),
    events: Object.assign({}, annotations.events || {}),
    suggestions: Object.assign({}, annotations.suggestions || {}),
    promotedRois: Object.assign({}, annotations.promotedRois || {}),
    virtualRois: Object.assign({}, annotations.virtualRois || {}),
    splitMergeDecisions: Object.assign({}, annotations.splitMergeDecisions || {})
  };
}
function captureActiveRunAnnotations(){
  annotations.runs = annotations.runs || {};
  annotations.runs[activeRunId()] = migrateRunBucket(runAnnotationSnapshot());
}
function materializeRunAnnotations(runId){
  annotations.runs = annotations.runs || {};
  const hasLegacy = Object.keys(annotations.rois || {}).length || Object.keys(annotations.events || {}).length || Object.keys(annotations.suggestions || {}).length || Object.keys(annotations.promotedRois || {}).length || Object.keys(annotations.virtualRois || {}).length || Object.keys(annotations.splitMergeDecisions || {}).length;
  if(!annotations.runs[runId] && hasLegacy && runId === baselineRunId()) annotations.runs[runId] = migrateRunBucket(runAnnotationSnapshot());
  const bucket = annotations.runs[runId] || {rois:{}, events:{}, suggestions:{}, promotedRois:{}, virtualRois:{}, splitMergeDecisions:{}};
  annotations.rois = Object.assign({}, bucket.rois || {});
  annotations.events = Object.assign({}, bucket.events || {});
  annotations.suggestions = Object.assign({}, bucket.suggestions || {});
  annotations.promotedRois = Object.assign({}, bucket.promotedRois || {});
  annotations.virtualRois = Object.assign({}, bucket.virtualRois || {});
  annotations.splitMergeDecisions = Object.assign({}, bucket.splitMergeDecisions || {});
}
function ensureRunAnnotationScope(){
  if(!setting('activeRunId')) annotations.settings.activeRunId = baselineRunId();
  materializeRunAnnotations(activeRunId());
}

async function loadAnnotations() {
  const local = localStorage.getItem(storeKey);
  if (local) mergeAnnotations(JSON.parse(local));
  if (serverBacked) {
    try {
      const res = await fetch('annotations.json', {cache: 'no-store'});
      if (res.ok) {
        mergeAnnotations(await res.json());
        setSaveState('autosave ready', 'ok');
      }
    } catch (_) {
      setSaveState('local browser save only', 'bad');
    }
  } else {
    setSaveState('static mode: export to save files', '');
  }
  ensureRunAnnotationScope();
  applySettingsToControls();
}

function setSaveState(text, cls) {
  saveStatus = {text, className: cls || '', updatedAt: new Date().toISOString(), serverBacked};
  saveStateEl.textContent = text;
  saveStateEl.className = 'saveState ' + (cls || '');
  renderReviewSessionPanel();
}

function recoveryHistory(){
  try {
    const parsed = JSON.parse(localStorage.getItem(recoveryStoreKey) || '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch (_) {
    return [];
  }
}

function recoverySummary(snapshot){
  const ann = snapshot?.annotations || {};
  const roiCount = Object.keys(ann.rois || {}).length;
  const virtualCount = Object.keys(ann.virtualRois || {}).length;
  const eventCount = Object.keys(ann.events || {}).length;
  return `${snapshot.createdAt || 'unknown'} | ${snapshot.reason || 'autosave'} | ${roiCount} ROI labels, ${virtualCount} virtual ROIs, ${eventCount} event labels`;
}

function pushRecoverySnapshot(reason='autosave', {force=false}={}){
  const now = Date.now();
  if(!force && now - lastRecoverySnapshotAt < 60_000) return;
  lastRecoverySnapshotAt = now;
  try {
    const history = recoveryHistory();
    history.unshift({
      id: `recovery_${now.toString(36)}`,
      createdAt: new Date(now).toISOString(),
      reason,
      activeRunId: activeRunId(),
      annotations: JSON.parse(JSON.stringify(annotations))
    });
    localStorage.setItem(recoveryStoreKey, JSON.stringify(history.slice(0, 12)));
    renderRecoveryControls();
  } catch (_) {
    setSaveState('could not write recovery snapshot', 'bad');
  }
}

function renderRecoveryControls(){
  const select = document.getElementById('recoverySnapshotSelect');
  const summary = document.getElementById('recoverySnapshotSummary');
  if(!select) return;
  const history = recoveryHistory();
  const previous = select.value;
  select.innerHTML = history.map(snap => `<option value="${escapeHtml(snap.id)}">${escapeHtml(recoverySummary(snap))}</option>`).join('');
  if(history.some(snap => snap.id === previous)) select.value = previous;
  else if(history[0]) select.value = history[0].id;
  if(summary) summary.textContent = history.length ? `${history.length} local recovery point${history.length === 1 ? '' : 's'} available.` : 'No recovery points yet.';
}

function restoreRecoverySnapshot(){
  const select = document.getElementById('recoverySnapshotSelect');
  const snapshot = recoveryHistory().find(item => item.id === select?.value);
  if(!snapshot?.annotations) {
    setSaveState('no recovery snapshot selected', 'bad');
    return;
  }
  pushRecoverySnapshot('before recovery restore', {force:true});
  mergeAnnotations(snapshot.annotations);
  ensureRunAnnotationScope();
  localStorage.setItem(storeKey, JSON.stringify(annotations));
  clearTraceCaches('recovery-restore');
  applySettingsToControls();
  renderAll();
  queueSave();
  setSaveState('restored recovery snapshot', 'ok');
}

function downloadRecoverySnapshot(){
  const select = document.getElementById('recoverySnapshotSelect');
  const snapshot = recoveryHistory().find(item => item.id === select?.value);
  if(snapshot) downloadJson(`${snapshot.id}.json`, snapshot);
}

function bookmarkSummary(bookmark){
  const bits = [];
  if(bookmark.roiId) bits.push(`ROI ${bookmark.roiId}`);
  if(bookmark.eventFrame) bits.push(`event f${bookmark.eventFrame}`);
  if(bookmark.suggestionId) bits.push(`suggestion ${bookmark.suggestionId}`);
  bits.push(`f${bookmark.frame}`);
  return `${bookmark.label || bits.join(' | ')} (${bits.join(', ')})`;
}

function renderBookmarkControls(){
  const select = document.getElementById('bookmarkSelect');
  if(!select) return;
  const previous = select.value;
  const bookmarks = Array.isArray(annotations.bookmarks) ? annotations.bookmarks : [];
  select.innerHTML = bookmarks.length
    ? bookmarks.map(mark => `<option value="${escapeHtml(mark.id)}">${escapeHtml(bookmarkSummary(mark))}</option>`).join('')
    : '<option value="">No bookmarks</option>';
  if(bookmarks.some(mark => mark.id === previous)) select.value = previous;
}

function addReviewBookmark(){
  annotations.bookmarks = Array.isArray(annotations.bookmarks) ? annotations.bookmarks : [];
  const roi = selectedRoi();
  const suggestion = selectedSuggestion();
  const roiId = roi ? String(roi.id) : '';
  const eventFrame = selectedEventFrame || null;
  const suggestionId = suggestion && !roiId ? String(suggestion.id) : '';
  const label = roiId ? `ROI ${roiId}${eventFrame ? ` event f${eventFrame}` : ''}` : suggestionId ? `Suggestion ${suggestionId}` : `Frame ${currentFrame}`;
  const mark = migrateBookmark({
    id: `mark_${Date.now().toString(36)}`,
    label,
    createdAt: new Date().toISOString(),
    runId: activeRunId(),
    frame: currentFrame,
    roiId,
    eventFrame,
    suggestionId
  });
  annotations.bookmarks.unshift(mark);
  annotations.bookmarks = annotations.bookmarks.slice(0, 80);
  recordAction('review_bookmark_add');
  queueSave();
  renderBookmarkControls();
  setSaveState(`bookmarked ${label}`, 'ok');
}

async function goToReviewBookmark(){
  const id = document.getElementById('bookmarkSelect')?.value;
  const mark = (annotations.bookmarks || []).find(item => item.id === id);
  if(!mark) {
    setSaveState('no bookmark selected', 'bad');
    return;
  }
  if(mark.runId && mark.runId !== activeRunId() && runById(mark.runId)) await selectActiveRun(mark.runId, {loadReview:false});
  if(mark.roiId && roiById(mark.roiId)) {
    selectRoi(mark.roiId);
    if(mark.eventFrame) {
      selectedEventFrame = Number(mark.eventFrame);
      eventNotes.value = eventAnn(mark.roiId, selectedEventFrame).notes || '';
    }
  } else if(mark.suggestionId) {
    selectSuggestion(mark.suggestionId);
  }
  setFrame(mark.frame || mark.eventFrame || currentFrame);
  renderAll();
  setSaveState(`opened bookmark: ${mark.label}`, 'ok');
}

function deleteReviewBookmark(){
  const id = document.getElementById('bookmarkSelect')?.value;
  const before = annotations.bookmarks?.length || 0;
  annotations.bookmarks = (annotations.bookmarks || []).filter(item => item.id !== id);
  if((annotations.bookmarks?.length || 0) !== before) {
    recordAction('review_bookmark_delete');
    queueSave();
    renderBookmarkControls();
    setSaveState('deleted bookmark', 'ok');
  }
}

function saveAnnotationsNow() {
  captureActiveRunAnnotations();
  annotations.updatedAt = new Date().toISOString();
  pushRecoverySnapshot('autosave');
  localStorage.setItem(storeKey, JSON.stringify(annotations));
  if (!serverBacked) {
    setSaveState('saved in browser', 'ok');
    return;
  }
  fetch('annotations.json', {
    method: 'PUT',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(annotations, null, 2)
  }).then(res => {
    setSaveState(res.ok ? 'autosaved to annotations.json' : 'autosave failed', res.ok ? 'ok' : 'bad');
  }).catch(() => setSaveState('autosave failed', 'bad'));
}

function queueSave() {
  setSaveState('saving...', '');
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveAnnotationsNow, 220);
}

function framePath(frame){ return data.video.framePattern.replace('%03d', String(frame).padStart(3, '0')); }
function virtualRoisArray(){ return Object.values(annotations.virtualRois || {}).filter(v => v && v.points?.length); }
function reviewRois(){ return (data.rois || []).concat(virtualRoisArray()); }
function selectedRoi(){ return reviewRois().find(r => String(r.id) === String(selectedId)) || data.rois[0] || virtualRoisArray()[0] || null; }
function roiAnn(id){ return annotations.rois[id] || migrateRoiAnn({}); }
function selectedRoiIdList(){ return [...selectedRoiIds].map(id => String(id)).filter(Boolean); }
function selectedRois(){ return selectedRoiIdList().map(roiById).filter(Boolean); }
function scoreValue(item, key, fallback=0){ const v = Number(item?.[key]); return Number.isFinite(v) ? v : fallback; }
function eventKey(roiId, frame){ return `${roiId}:${frame}`; }
function eventAnn(roiId, frame){ return annotations.events[eventKey(roiId, frame)] || migrateEventAnn({}); }
function suggestionAnn(id){ return annotations.suggestions[id] || migrateSuggestionAnn({}); }
function setting(name){ return annotations.settings[name]; }
function setSetting(name, value){ annotations.settings[name] = value; queueSave(); }
function recordAction(kind){
  annotations.reviewStats = annotations.reviewStats || {sessionStartedAt: new Date().toISOString(), lastActionAt: null, actions: {}};
  annotations.reviewStats.actions = annotations.reviewStats.actions || {};
  annotations.reviewStats.actions[kind] = (annotations.reviewStats.actions[kind] || 0) + 1;
  annotations.reviewStats.lastActionAt = new Date().toISOString();
}
function currentReviewerId(){
  return String(setting('reviewerId') || '').trim();
}
function stampAnnotation(item){
  item.updatedAt = new Date().toISOString();
  const reviewer = currentReviewerId();
  if(reviewer) item.reviewer_id = reviewer;
  return item;
}
function annotationIsReviewed(group, id, item){
  const ann = item || {};
  if(group === 'rois' || group === 'virtualRois') return Boolean(ann.state || ann.cell_state);
  if(group === 'events') return Boolean(ann.state || ann.event_state);
  if(group === 'suggestions') return Boolean(ann.state || annotations.promotedRois?.[id]);
  if(group === 'splitMergeDecisions') return Boolean(ann.decision_state);
  return false;
}
function requireCurrentReviewer(){
  if(currentReviewerId()) return true;
  setSaveState('set Reviewer before stamping reviewer IDs', 'bad');
  document.getElementById('reviewerIdInput')?.focus();
  return false;
}
function stampAnnotationRecord(group, id, snapshots){
  const bucket = annotationBucket(group);
  const item = bucket[id];
  if(!item || item.reviewer_id || !annotationIsReviewed(group, id, item)) return false;
  snapshots.push(annotationSnapshot(group, id));
  stampAnnotation(item);
  return true;
}
function stampSelectedReviewer(){
  if(!requireCurrentReviewer()) return;
  const snapshots = [];
  let count = 0;
  for(const id of selectedRoiIdList()){
    if(stampAnnotationRecord('rois', id, snapshots)) count++;
    if(stampAnnotationRecord('virtualRois', id, snapshots)) count++;
  }
  const roi = selectedRoi();
  if(roi && selectedEventFrame && stampAnnotationRecord('events', eventKey(roi.id, selectedEventFrame), snapshots)) count++;
  const s = selectedSuggestion();
  if(s && stampAnnotationRecord('suggestions', s.id, snapshots)) count++;
  if(!count) {
    setSaveState('selected reviewed labels already have reviewer IDs', 'ok');
    return;
  }
  pushAnnotationUndo(`reviewer stamp on ${count} selected label${count === 1 ? '' : 's'}`, snapshots);
  recordAction('reviewer_stamp_selected');
  queueSave();
  renderAll();
  setSaveState(`stamped ${count} selected label${count === 1 ? '' : 's'}`, 'ok');
}
function stampMissingReviewerLabels(){
  if(!requireCurrentReviewer()) return;
  const snapshots = [];
  let count = 0;
  for(const group of ['rois', 'events', 'suggestions', 'virtualRois', 'splitMergeDecisions']){
    const bucket = annotationBucket(group);
    for(const id of Object.keys(bucket)){
      if(stampAnnotationRecord(group, id, snapshots)) count++;
    }
  }
  if(!count) {
    setSaveState('no reviewed labels missing reviewer IDs', 'ok');
    return;
  }
  pushAnnotationUndo(`reviewer stamp on ${count} missing label${count === 1 ? '' : 's'}`, snapshots);
  recordAction('reviewer_stamp_missing');
  queueSave();
  renderAll();
  setSaveState(`stamped ${count} reviewed label${count === 1 ? '' : 's'}`, 'ok');
}
function cloneAnnotationValue(value){
  return value === undefined ? undefined : JSON.parse(JSON.stringify(value));
}
function annotationBucket(group){
  annotations[group] = annotations[group] || {};
  return annotations[group];
}
function annotationSnapshot(group, id){
  const bucket = annotationBucket(group);
  return {group, id:String(id), existed:Object.prototype.hasOwnProperty.call(bucket, id), value:cloneAnnotationValue(bucket[id])};
}
function pushAnnotationUndo(label, snapshots){
  const records = (snapshots || []).filter(Boolean);
  if(!records.length) return;
  annotationUndoStack.push({label, records});
  while(annotationUndoStack.length > 40) annotationUndoStack.shift();
  updateUndoButton();
}
function restoreAnnotationSnapshot(snapshot){
  const bucket = annotationBucket(snapshot.group);
  if(snapshot.existed) bucket[snapshot.id] = cloneAnnotationValue(snapshot.value);
  else delete bucket[snapshot.id];
}
function undoLastAnnotationChange(){
  const item = annotationUndoStack.pop();
  if(!item) {
    setSaveState('nothing to undo', 'bad');
    updateUndoButton();
    return;
  }
  for(const snapshot of item.records) restoreAnnotationSnapshot(snapshot);
  recordAction('annotation_undo');
  queueSave();
  renderAll();
  setSaveState(`undid ${item.label}`, 'ok');
}
function updateUndoButton(){
  const btn = document.getElementById('undoAnnotationBtn');
  if(!btn) return;
  const last = annotationUndoStack[annotationUndoStack.length - 1];
  btn.disabled = !last;
  btn.title = last ? `Undo ${last.label}` : 'No label changes to undo in this session';
}
function threshold(){ return Number(setting('eventThreshold')); }
function kalmanGain(){ return Number(setting('kalmanGain')); }
function spikeGain(){ return Number(setting('spikeGain')); }
function minAreaFilter(){ return Number(setting('minArea')); }
function minEventsFilter(){ return Number(setting('minEvents')); }
function targetCounts(){
  return {
    rois: Number(setting('targetRois')) || 30,
    events: Number(setting('targetEvents')) || 30,
    suggestions: Number(setting('targetSuggestions')) || 15
  };
}

function runById(runId){ return architectureRuns().find(r => r.run_id === runId) || null; }
function runGenerated(run){ return Boolean(run?.artifacts?.review_data) && run?.execution?.status !== 'planned'; }
function runAppUrl(run){ return run?.artifacts?.app_url || run?.artifacts?.app || ''; }
function reviewDataCacheKey(run){
  const url = artifactUrl(run?.artifacts?.review_data);
  return run && url ? `${run.run_id}:${url}` : '';
}
function artifactUrl(path){
  if(!path) return '';
  const value = String(path);
  if(/^https?:\/\//.test(value)) return value;
  const match = value.match(/Outputs\/NeuronReview\/([^/]+)\/app\/(.+)$/);
  if(match){
    const dataset = match[1], rest = match[2];
    const currentDataset = data.dataset?.dataset_id || datasetId;
    if(dataset === currentDataset) return rest;
    return location.pathname.includes('/app/') ? `../../${dataset}/app/${rest}` : '';
  }
  if(!value.startsWith('/')) return value;
  const generated = value.match(/generated_runs\/(.+)$/);
  if(generated) return `generated_runs/${generated[1]}`;
  const evidence = value.match(/evidence\/(.+)$/);
  if(evidence) return `evidence/${evidence[1]}`;
  const frames = value.match(/frames\/(.+)$/);
  if(frames) return `frames/${frames[1]}`;
  return value.startsWith('/') ? '' : value;
}
function framePatternPath(pattern, frame){
  if(!pattern) return '';
  const frameText = String(frame).padStart(3, '0');
  return artifactUrl(String(pattern).replace('%03d', frameText).replace('{frame}', String(frame)).replace('{frame03}', frameText));
}
function rebaseRelativeAsset(value, base){
  if(!value || !base) return value;
  const text = String(value);
  if(/^https?:\/\//.test(text) || text.startsWith('../') || text.startsWith(base)) return text;
  if(text.includes('Outputs/NeuronReview/')) return artifactUrl(text);
  if(text.startsWith('/')) return text;
  return `${base.replace(/\/$/, '')}/${text}`;
}
function rebaseReviewDataAssets(reviewData, reviewUrl){
  const slash = String(reviewUrl || '').lastIndexOf('/');
  const base = slash >= 0 ? String(reviewUrl).slice(0, slash) : '';
  if(!base) return reviewData;
  if(reviewData.video?.framePattern) reviewData.video.framePattern = rebaseRelativeAsset(reviewData.video.framePattern, base);
  for(const map of reviewData.discovery?.evidenceMaps || []){
    if(map.file) map.file = rebaseRelativeAsset(map.file, base);
    if(map.path) map.path = rebaseRelativeAsset(map.path, base);
  }
  return reviewData;
}
async function fetchReviewDataForRun(run){
  if(!runGenerated(run)) throw new Error('Run does not have generated review data.');
  const url = artifactUrl(run.artifacts?.review_data);
  if(!url) throw new Error('Review data is not reachable from this app.');
  const key = reviewDataCacheKey(run);
  const cached = reviewDataCache.get(key);
  if(cached?.status === 'ready') return cached.data;
  if(cached?.status === 'loading') return cached.promise;
  const promise = fetch(url, {cache:'no-store'}).then(async res => {
    if(!res.ok) throw new Error(await res.text());
    const reviewData = rebaseReviewDataAssets(await res.json(), url);
    reviewData.architectureRuns = data.architectureRuns;
    reviewData.pipelineCatalog = data.pipelineCatalog;
    reviewDataCache.set(key, {status:'ready', data:reviewData});
    return reviewData;
  }).catch(err => {
    reviewDataCache.set(key, {status:'error', error:err.message || 'review data did not load'});
    throw err;
  });
  reviewDataCache.set(key, {status:'loading', promise});
  return promise;
}
function generationCommandForRun(run){
  const manifestPath = data.dataset?.paths?.dataset_manifest || data.dataset?.manifest || `Outputs/Manifests/${datasetId}.json`;
  const outPath = `Outputs/ArchitectureRuns/${datasetId}/${run?.run_id || 'planned_run'}.json`;
  return [
    `python3 tools/build_pipeline_run.py --spec planned_architecture_run.json --out ${outPath}`,
    `python3 tools/run_neuron_review_pipeline.py --dataset-manifest ${manifestPath} --architecture-runs ${data.dataset?.paths?.architecture_runs || 'Outputs/NeuronReview/' + datasetId + '/app/architecture_runs.json'} --run-id ${run?.run_id || 'planned_run'} --stages all`
  ].join('\n');
}
function runStatusLabel(run){
  if(!run) return 'no run selected';
  if(runGenerated(run)) return 'generated review view available';
  if(runAppUrl(run)) return 'generated app link available';
  if(run.execution?.status === 'planned') return 'planned, not generated yet';
  return 'metadata only';
}
function apiUrl(path){ return `api/${path.replace(/^\/+/, '')}`; }
function generationHeaders(){
  const headers = {'Content-Type':'application/json'};
  if(generationOwnerToken) headers['X-Neurobench-Owner-Token'] = generationOwnerToken;
  return headers;
}
async function fetchJson(url, options={}){
  const res = await fetch(url, Object.assign({cache:'no-store'}, options));
  const text = await res.text();
  let payload = {};
  try { payload = text ? JSON.parse(text) : {}; } catch (_) { payload = {error:text}; }
  if(!res.ok) {
    const err = new Error(payload.error || res.statusText);
    err.payload = payload;
    throw err;
  }
  return payload;
}
function proposalAnalysisUrl(run){
  return artifactUrl(run?.artifacts?.proposal_analysis || '');
}
function proposalAnalysisForRun(run){
  const url = proposalAnalysisUrl(run);
  if(!url) return null;
  const cached = proposalAnalysisCache.get(url);
  if(cached?.status === 'ready') return cached.data;
  if(cached?.status === 'error') return cached;
  if(!cached){
    proposalAnalysisCache.set(url, {status:'loading'});
    fetchJson(url).then(payload => {
      proposalAnalysisCache.set(url, {status:'ready', data:payload});
      const hash = (location.hash || '#review').replace(/^#\/?/, '');
      if(['process','process-lab','qc','dataset-qc'].includes(hash)) renderDatasetQc();
    }).catch(err => {
      proposalAnalysisCache.set(url, {status:'error', error:err.message || 'proposal analysis did not load'});
      const hash = (location.hash || '#review').replace(/^#\/?/, '');
      if(['process','process-lab','qc','dataset-qc'].includes(hash)) renderDatasetQc();
    });
  }
  return {status:'loading'};
}
async function loadGenerationEnvironment(){
  if(!serverBacked) return null;
  try {
    generationEnvironment = await fetchJson(apiUrl('environment'));
  } catch (_) {
    generationEnvironment = null;
  }
  renderRunSyncControls();
  return generationEnvironment;
}
function backendReadiness(){
  const backend = document.getElementById('generationBackend')?.value || 'auto';
  if(!serverBacked) return {ok:false, text:'Generation requires the local workbench server.'};
  if(!generationEnvironment) return {ok:false, text:'Checking generation environment.'};
  if(generationEnvironment.owner_token_required && !generationOwnerToken) return {ok:false, text:'Owner token required to start local processing jobs.'};
  if(backend === 'python_gpu') {
    const cuda = Boolean(generationEnvironment.gpu?.cuda);
    return {ok:cuda, text: cuda ? `CUDA ready (${generationEnvironment.gpu?.cuda_device_count || 1} device)` : 'Python GPU selected, but Torch CUDA is unavailable.'};
  }
  const fijiOk = Boolean(generationEnvironment.fiji_available);
  return {ok:fijiOk, text: fijiOk ? 'Fiji/Groovy backend ready.' : 'Fiji executable was not found by the local server.'};
}

function median(arr){ const a = [...arr].sort((x,y)=>x-y); const m = Math.floor(a.length/2); return a.length % 2 ? a[m] : 0.5*(a[m-1]+a[m]); }
function madSigma(arr, center){ return Math.max(1e-6, 1.4826 * median(arr.map(v => Math.abs(v - center)))); }
function modeledTrace(roi){
  const gain = kalmanGain(), sgain = spikeGain();
  const center = median(roi.dffTrace);
  const sigma = madSigma(roi.dffTrace, center);
  let baseline = center;
  const baselineTrace = [], eventTrace = [], zTrace = [];
  for(const v of roi.dffTrace){
    const residual = v - baseline;
    let k = gain;
    if(residual > 2.5 * sigma) k = sgain;
    if(residual < -1.0 * sigma) k = Math.min(0.18, gain * 1.8);
    baseline += k * residual;
    baselineTrace.push(baseline);
    const ev = Math.max(0, v - baseline);
    eventTrace.push(ev);
    zTrace.push(ev / sigma);
  }
  return {baselineTrace, eventTrace, zTrace, sigma};
}
function cacheSetBounded(cache, key, value, limit=TRACE_CACHE_LIMIT){
  if(cache.size >= limit) {
    const firstKey = cache.keys().next().value;
    if(firstKey !== undefined) cache.delete(firstKey);
  }
  cache.set(key, value);
  return value;
}
function clearTraceCaches(reason='manual'){
  traceCache.clear();
  traceEventCache.clear();
  traceCacheStats.clears++;
  traceCacheStats.lastClearReason = reason;
}
function clearTraceEventCache(reason='event-threshold'){
  traceEventCache.clear();
  traceCacheStats.clears++;
  traceCacheStats.lastClearReason = reason;
}
function traceCacheKey(roi){
  return `${roi.id}|${roi.dffTrace?.length || 0}|${Number(kalmanGain()).toFixed(4)}|${Number(spikeGain()).toFixed(4)}`;
}
function modeledTraceCached(roi){
  const key = traceCacheKey(roi);
  if(traceCache.has(key)) {
    traceCacheStats.traceHits++;
    return traceCache.get(key);
  }
  traceCacheStats.traceMisses++;
  cacheSetBounded(traceCache, key, modeledTrace(roi));
  return traceCache.get(key);
}
function eventCacheKey(roi){
  return `${traceCacheKey(roi)}|${Number(threshold()).toFixed(4)}`;
}
function eventsForRoi(roi){
  if(!roi || !Array.isArray(roi.dffTrace) || roi.dffTrace.length < 3) return [];
  const key = eventCacheKey(roi);
  if(traceEventCache.has(key)) {
    traceCacheStats.eventHits++;
    return traceEventCache.get(key);
  }
  traceCacheStats.eventMisses++;
  const model = modeledTraceCached(roi);
  const zt = model.zTrace;
  const th = threshold();
  const out = [];
  for(let i=1;i<zt.length-1;i++){
    if(zt[i] >= th && zt[i] >= zt[i-1] && zt[i] >= zt[i+1]){
      out.push({frame:i+1, z:zt[i], amplitude:model.eventTrace[i]});
    }
  }
  return cacheSetBounded(traceEventCache, key, out);
}
function eventFrames(roi){ return eventsForRoi(roi).map(e => e.frame); }
function eventNearFrame(roi, frame){ return eventFrames(roi).some(f => Math.abs(f - frame) <= 1); }
function roiById(id){ return reviewRois().find(r => String(r.id) === String(id)) || null; }
function distance(a, b){
  if(!a || !b) return Infinity;
  const ax = Number(a.centroidX ?? a.x), ay = Number(a.centroidY ?? a.y);
  const bx = Number(b.centroidX ?? b.x), by = Number(b.centroidY ?? b.y);
  const dx = ax - bx;
  const dy = ay - by;
  return Math.sqrt(dx * dx + dy * dy);
}
function nearestRoiForPoint(point, excludeId=null){
  let best = null, bestDistance = Infinity;
  for(const roi of data.rois || []){
    if(String(roi.id) === String(excludeId)) continue;
    const d = distance(point, roi);
    if(d < bestDistance) {
      best = roi;
      bestDistance = d;
    }
  }
  return best ? {roi: best, distance: bestDistance} : null;
}
function nearestRoiForSuggestion(s){ return nearestRoiForPoint(s); }
function roiFocusMode(){ return setting('roiFocusMode') || 'all'; }
function roiInFocus(roi){
  const mode = roiFocusMode();
  if(mode === 'all') return true;
  const selected = selectedRoi();
  if(!selected) return true;
  if(String(roi.id) === String(selected.id) || selectedRoiIds.has(String(roi.id))) return true;
  if(mode === 'solo') return false;
  if(mode === 'neighbors') return distance(roi, selected) <= (Number(setting('neighborRadiusPx')) || 36);
  return true;
}
function visibleOverlayRois(){ return visibleRois().filter(roiInFocus); }

function roiQualityScore(roi) {
  return scoreValue(roi, 'priorityScore', scoreValue(roi, 'peakScore', 0) / Math.max(0.04, scoreValue(roi, 'noiseSigma', 0.04)) + eventsForRoi(roi).length * 0.4);
}
function roiUncertaintyScore(roi) {
  const ev = eventsForRoi(roi).length;
  const ann = roiAnn(roi.id);
  return (ann.state ? 0 : 20) + scoreValue(roi, 'noiseSigma', 0) * 12 + Math.abs(scoreValue(roi, 'area', 0) - 65) / 50 - ev * 0.15;
}
function roiArtifactLike(roi) {
  const ann = roiAnn(roi.id);
  const artifactClass = ann.artifact_class || ann.artifactClass || '';
  return artifactReasonsForRoi(roi).length > 0 || Boolean(artifactClass && artifactClass !== 'none');
}
function artifactReasonsForRoi(roi){
  const ann = roiAnn(roi.id);
  const reasons = [];
  if(scoreValue(roi, 'artifactScore') >= 0.4) reasons.push('artifact score');
  if(scoreValue(roi, 'backgroundCorrelation') >= 0.55) reasons.push('background correlated');
  if(scoreValue(roi, 'localCorrelationMean') > 0 && scoreValue(roi, 'localCorrelationMean') < 0.35) reasons.push('low local coherence');
  if(roi.area < 8) reasons.push('too small');
  if(roi.area >= 180) reasons.push('large or merged');
  const bbox = roi.bbox || [];
  if(bbox.length === 4) {
    const w = Math.max(1, bbox[2] - bbox[0] + 1);
    const h = Math.max(1, bbox[3] - bbox[1] + 1);
    if(Math.max(w / h, h / w) >= 5) reasons.push('elongated');
    if(bbox[0] <= 1 || bbox[1] <= 1 || bbox[2] >= data.video.width - 2 || bbox[3] >= data.video.height - 2) reasons.push('near border');
  }
  const artifactClass = ann.artifact_class || ann.artifactClass || '';
  if(artifactClass && artifactClass !== 'none') reasons.push(artifactClass.replace(/_/g, ' '));
  return [...new Set(reasons)];
}
function roiMergedClusterLike(roi) {
  const ann = roiAnn(roi.id);
  return ann.needs_action === 'merge_needed' || ann.artifact_class === 'merge_needed' || ann.artifactClass === 'merge_needed' || roi.area >= 180;
}
function roiWeakTraceLike(roi) {
  const ann = roiAnn(roi.id);
  return ['weak','noisy','unusable'].includes(ann.trace_quality) || scoreValue(roi, 'traceSnr', 99) < 1.5;
}
function roiNeedsEventReview(roi) {
  return eventsForRoi(roi).some(ev => !eventAnn(roi.id, ev.frame).state && !eventAnn(roi.id, ev.frame).event_state);
}
function roiStrongNeuronLike(roi) {
  const ann = roiAnn(roi.id);
  return ann.state === 'accept' || ann.cell_state === 'accepted' ||
    (roiQualityScore(roi) >= 4 && eventsForRoi(roi).length >= 1 && !roiArtifactLike(roi) && !roiWeakTraceLike(roi));
}
function roiReviewed(roi){
  const ann = roiAnn(roi.id);
  return Boolean(ann.state || ann.cell_state);
}
function roiReviewerId(roi){
  return String(roiAnn(roi.id).reviewer_id || '').trim();
}
function eventReviewState(roiId, frame){
  const ann = eventAnn(roiId, frame);
  return ann.event_state || (ann.state === 'accept' ? 'accepted' : ann.state === 'reject' ? 'rejected' : ann.state === 'unsure' ? 'unsure' : '');
}
function eventReviewed(roiId, frame){
  return Boolean(eventReviewState(roiId, frame));
}
function eventReviewerId(roiId, frame){
  return String(eventAnn(roiId, frame).reviewer_id || '').trim();
}
function eventMatchesQueue(roi, ev, queue=setting('eventQueue') || 'all'){
  const ann = eventAnn(roi.id, ev.frame);
  const state = eventReviewState(roi.id, ev.frame);
  const reviewer = String(ann.reviewer_id || '').trim();
  if(queue === 'unlabeled') return !state;
  if(queue === 'accepted') return state === 'accepted';
  if(queue === 'rejected') return state === 'rejected';
  if(queue === 'unsure') return state === 'unsure';
  if(queue === 'missingReviewer') return Boolean(state) && !reviewer;
  if(queue === 'reviewedByMe') return currentReviewerId() && reviewer === currentReviewerId();
  if(queue === 'reviewedByOther') return currentReviewerId() && Boolean(state) && reviewer && reviewer !== currentReviewerId();
  if(queue === 'highZ') return scoreValue(ev, 'z') >= Math.max(2, Number(setting('eventThreshold')) || 2.4);
  return true;
}
function eventQueueItems(){
  const queue = setting('eventQueue') || 'all';
  const items = [];
  for(const roi of reviewRois()){
    if(roiAnn(roi.id).deleted) continue;
    for(const ev of eventsForRoi(roi)){
      if(eventMatchesQueue(roi, ev, queue)) items.push({roi, ev, key: eventKey(roi.id, ev.frame)});
    }
  }
  return items.sort((a,b) => Number(a.ev.frame) - Number(b.ev.frame) || String(a.roi.id).localeCompare(String(b.roi.id), undefined, {numeric:true}));
}
function visibleEventsForRoi(roi){
  return eventsForRoi(roi).filter(ev => eventMatchesQueue(roi, ev));
}
function roiTriageCategory(roi) {
  if(roiArtifactLike(roi)) return 'artifact_like';
  if(roiMergedClusterLike(roi)) return 'merged_cluster';
  if(roiWeakTraceLike(roi)) return 'weak_trace';
  if(roiNeedsEventReview(roi)) return 'needs_event_review';
  if(roiStrongNeuronLike(roi)) return 'strong_neuron';
  return 'standard_review';
}
function visibleRois(){
  const queue = setting('queue');
  const batchIds = queue === 'annotationBatch' ? new Set(nextAnnotationBatch().rois.map(r => String(r.roi_id))) : null;
  let rows = reviewRois().filter(r => scoreValue(r, 'area', 0) >= minAreaFilter() && eventsForRoi(r).length >= minEventsFilter());
  rows = rows.filter(r => {
    const ann = roiAnn(r.id);
    if (queue !== 'deleted' && ann.deleted) return false;
    if (queue === 'annotationBatch') return batchIds.has(String(r.id));
    if (queue === 'unlabeled') return !ann.state;
    if (queue === 'accepted') return ann.state === 'accept';
    if (queue === 'rejected') return ann.state === 'reject';
    if (queue === 'unsure') return ann.state === 'unsure';
    if (queue === 'missingReviewer') return roiReviewed(r) && !roiReviewerId(r);
    if (queue === 'reviewedByMe') return currentReviewerId() && roiReviewerId(r) === currentReviewerId();
    if (queue === 'reviewedByOther') return currentReviewerId() && roiReviewed(r) && roiReviewerId(r) && roiReviewerId(r) !== currentReviewerId();
    if (queue === 'deleted') return ann.deleted;
    if (queue === 'needsAction') return Boolean(ann.needs_action);
    if (queue === 'controlReady') return ann.control_ready === 'yes' || ann.control_ready === 'maybe';
    if (queue === 'problemTrace') return ann.trace_quality === 'noisy' || ann.trace_quality === 'unusable';
    if (queue === 'artifactRisk') return scoreValue(r, 'artifactScore') >= 0.4;
    if (queue === 'strongNeuron') return roiStrongNeuronLike(r);
    if (queue === 'artifactLike') return roiArtifactLike(r);
    if (queue === 'mergedCluster') return roiMergedClusterLike(r);
    if (queue === 'weakTrace') return roiWeakTraceLike(r);
    if (queue === 'needsEventReview') return roiNeedsEventReview(r);
    return true;
  });
  if (queue === 'highNoise') rows.sort((a,b) => scoreValue(b, 'noiseSigma', 0) - scoreValue(a, 'noiseSigma', 0));
  else if (queue === 'highEvents') rows.sort((a,b) => eventsForRoi(b).length - eventsForRoi(a).length);
  else if (queue === 'priority') rows.sort((a,b) => roiQualityScore(b) - roiQualityScore(a));
  else if (queue === 'localCorrelation') rows.sort((a,b) => scoreValue(b, 'localCorrelationMean') - scoreValue(a, 'localCorrelationMean'));
  else if (queue === 'eventSupport') rows.sort((a,b) => scoreValue(b, 'eventSupport') - scoreValue(a, 'eventSupport'));
  else if (queue === 'traceSnr') rows.sort((a,b) => scoreValue(b, 'traceSnr') - scoreValue(a, 'traceSnr'));
  else if (queue === 'artifactRisk') rows.sort((a,b) => scoreValue(b, 'artifactScore') - scoreValue(a, 'artifactScore'));
  else if (queue === 'strongNeuron') rows.sort((a,b) => roiQualityScore(b) - roiQualityScore(a));
  else if (queue === 'artifactLike') rows.sort((a,b) => scoreValue(b, 'artifactScore') - scoreValue(a, 'artifactScore'));
  else if (queue === 'mergedCluster') rows.sort((a,b) => scoreValue(b, 'area', 0) - scoreValue(a, 'area', 0));
  else if (queue === 'weakTrace') rows.sort((a,b) => scoreValue(a, 'traceSnr', 99) - scoreValue(b, 'traceSnr', 99));
  else if (queue === 'needsEventReview') rows.sort((a,b) => eventsForRoi(b).length - eventsForRoi(a).length);
  else if (queue === 'missingReviewer' || queue === 'reviewedByMe' || queue === 'reviewedByOther') rows.sort((a,b) => String(roiReviewerId(a)).localeCompare(String(roiReviewerId(b))) || roiQualityScore(b) - roiQualityScore(a));
  else if (queue === 'uncertain') rows.sort((a,b) => roiUncertaintyScore(b) - roiUncertaintyScore(a));
  else if (queue === 'annotationBatch') rows.sort((a,b) => batchIds.has(String(b.id)) - batchIds.has(String(a.id)) || roiReviewPriority(b).score - roiReviewPriority(a).score);
  else rows.sort((a,b) => roiQualityScore(b) - roiQualityScore(a));
  return rows;
}

function selectedSuggestion(){
  const suggestions = data.discovery?.suggestions || [];
  return suggestions.find(s => s.id === selectedSuggestionId) || suggestions[0] || null;
}
function visibleSuggestions(){
  const queue = setting('discoveryQueue') || 'all';
  let rows = [...(data.discovery?.suggestions || [])];
  rows = rows.filter(s => {
    const ann = suggestionAnn(s.id);
    const reviewer = String(ann.reviewer_id || '').trim();
    const reviewed = Boolean(ann.state || annotations.promotedRois[s.id]);
    if (queue === 'unlabeled') return !ann.state;
    if (queue === 'promoted') return ann.state === 'promoted' || Boolean(annotations.promotedRois[s.id]);
    if (queue === 'missed') return ann.state === 'missed';
    if (queue === 'artifact') return ann.state === 'artifact';
    if (queue === 'artifactSuspects') return s.artifactCue && s.artifactCue !== 'none';
    if (queue === 'missingReviewer') return reviewed && !reviewer;
    if (queue === 'reviewedByMe') return currentReviewerId() && reviewer === currentReviewerId();
    if (queue === 'reviewedByOther') return currentReviewerId() && reviewed && reviewer && reviewer !== currentReviewerId();
    return true;
  });
  rows.sort((a,b) => scoreValue(b, 'priorityScore', b.discoveryScore || 0) - scoreValue(a, 'priorityScore', a.discoveryScore || 0));
  return rows;
}

function roiReviewPriority(roi) {
  const ann = roiAnn(roi.id);
  const eventCount = eventsForRoi(roi).length;
  const traceSnr = scoreValue(roi, 'traceSnr');
  const localCorr = scoreValue(roi, 'localCorrelationMean');
  const eventSupport = scoreValue(roi, 'eventSupport');
  const artifact = scoreValue(roi, 'artifactScore');
  let score = scoreValue(roi, 'priorityScore') + Math.min(eventCount, 8) * 0.45;
  score += Math.min(Math.max(traceSnr, 0), 6) * 0.25;
  score += Math.min(Math.max(localCorr, 0), 1) * 1.2;
  score += Math.min(Math.max(eventSupport, 0), 1) * 1.1;
  score -= Math.min(Math.max(artifact, 0), 1) * 1.6;
  if(!ann.state && !ann.cell_state) score += 2.0;
  if(ann.needs_action) score += 0.6;
  if(roi.area >= 20 && roi.area <= 180) score += 0.35;
  const reasons = [];
  if(!ann.state && !ann.cell_state) reasons.push('unlabeled ROI');
  if(eventCount) reasons.push(`${eventCount} events`);
  if(traceSnr >= 1.5) reasons.push('usable SNR');
  else if(traceSnr > 0) reasons.push('weak SNR');
  if(localCorr >= 0.4) reasons.push('coherent');
  else if(localCorr > 0) reasons.push('low coherence');
  if(eventSupport >= 0.35) reasons.push('event support');
  if(artifact >= 0.4) reasons.push('artifact check');
  return {score, reasons};
}

function suggestionReviewPriority(s) {
  const ann = suggestionAnn(s.id);
  let score = scoreValue(s, 'priorityScore', scoreValue(s, 'discoveryScore'));
  score += Math.min(Math.max(scoreValue(s, 'localCorrelationMean'), 0), 1) * 0.8;
  score += Math.min(Math.max(scoreValue(s, 'eventSupport'), 0), 1) * 0.8;
  if(!ann.state && !annotations.promotedRois[s.id]) score += 1.5;
  if((s.artifactCue && s.artifactCue !== 'none') || scoreValue(s, 'artifactScore') >= 0.4) score += 0.7;
  const reasons = [];
  if(!ann.state && !annotations.promotedRois[s.id]) reasons.push('unlabeled suggestion');
  if((s.artifactCue && s.artifactCue !== 'none') || scoreValue(s, 'artifactScore') >= 0.4) reasons.push('artifact check');
  if(scoreValue(s, 'localCorrelationMean') >= 0.4) reasons.push('coherent');
  if(scoreValue(s, 'eventSupport') >= 0.35) reasons.push('event support');
  return {score, reasons};
}

function nextAnnotationBatch(targets={rois:30, events:30, suggestions:15}) {
  const rois = data.rois
    .filter(roi => {
      const ann = roiAnn(roi.id);
      return !ann.state && !ann.cell_state || Boolean(ann.needs_action);
    })
    .map(roi => {
      const priority = roiReviewPriority(roi);
      return {
        roi_id: roi.id,
        score: priority.score,
        event_count: eventsForRoi(roi).length,
        area: roi.area,
        reasons: priority.reasons
      };
    })
    .sort((a,b) => b.score - a.score || Number(a.roi_id) - Number(b.roi_id))
    .slice(0, targets.rois);
  const selected = new Set(rois.map(r => String(r.roi_id)));
  const events = [];
  for(const roi of data.rois){
    for(const ev of eventsForRoi(roi)){
      const ann = eventAnn(roi.id, ev.frame);
      if(ann.state || ann.event_state) continue;
      events.push({
        roi_id: roi.id,
        frame: ev.frame,
        score: roiReviewPriority(roi).score + Number(ev.z || 0) * 0.4 + (selected.has(String(roi.id)) ? 1 : 0),
        z: ev.z,
        amplitude: ev.amplitude,
        reasons: selected.has(String(roi.id)) ? ['unlabeled event', 'selected ROI'] : ['unlabeled event']
      });
    }
  }
  events.sort((a,b) => b.score - a.score || Number(a.roi_id) - Number(b.roi_id) || Number(a.frame) - Number(b.frame));
  const suggestions = (data.discovery?.suggestions || [])
    .filter(s => !suggestionAnn(s.id).state && !annotations.promotedRois[s.id])
    .map(s => {
      const priority = suggestionReviewPriority(s);
      return {suggestion_id: s.id, score: priority.score, area: s.area, reasons: priority.reasons};
    })
    .sort((a,b) => b.score - a.score || String(a.suggestion_id).localeCompare(String(b.suggestion_id)))
    .slice(0, targets.suggestions);
  return {rois, events: events.slice(0, targets.events), suggestions};
}

function guidedTasks(){
  const batch = nextAnnotationBatch(targetCounts());
  const tasks = [];
  for(const item of batch.rois) tasks.push({
    task_id: `roi:${item.roi_id}`,
    task_type: 'roi',
    subject_id: String(item.roi_id),
    priority_score: item.score,
    prompt: `Decide whether ROI ${item.roi_id} is a neuron, artifact, or unsure case.`,
    reasons: item.reasons,
    recommended_context: ['video', 'crop', 'trace', 'event frames']
  });
  for(const item of batch.events) tasks.push({
    task_id: `event:${item.roi_id}:${item.frame}`,
    task_type: 'event',
    subject_id: `${item.roi_id}:${item.frame}`,
    roi_id: String(item.roi_id),
    frame: item.frame,
    priority_score: item.score,
    prompt: `Review ROI ${item.roi_id} event at frame ${item.frame}.`,
    reasons: item.reasons,
    recommended_context: ['video', 'trace', 'event frames']
  });
  for(const item of batch.suggestions) tasks.push({
    task_id: `suggestion:${item.suggestion_id}`,
    task_type: 'suggestion',
    subject_id: String(item.suggestion_id),
    priority_score: item.score,
    prompt: `Check whether suggestion ${item.suggestion_id} is a missed neuron or artifact.`,
    reasons: item.reasons,
    recommended_context: ['video', 'evidence map', 'suggestion overlay']
  });
  tasks.sort((a,b) => Number(b.priority_score) - Number(a.priority_score) || a.task_id.localeCompare(b.task_id));
  return tasks;
}

function currentGuidedTask(){
  const tasks = guidedTasks();
  if(!tasks.length) return null;
  const idx = Math.max(0, Math.min(tasks.length - 1, Number(setting('guidedTaskIndex')) || 0));
  return tasks[idx];
}

function selectGuidedTask(task=currentGuidedTask()){
  if(!task) return;
  if(task.task_type === 'roi') selectRoi(Number(task.subject_id));
  else if(task.task_type === 'event') {
    selectRoi(Number(task.roi_id));
    selectedEventFrame = Number(task.frame);
    eventNotes.value = eventAnn(task.roi_id, task.frame).notes || '';
    setFrame(Number(task.frame));
  } else if(task.task_type === 'suggestion') {
    selectSuggestion(task.subject_id);
  }
}

function guidedActionButtons(task){
  if(!task) return '';
  const actions = {
    roi: [
      ['accept', 'Accept neuron'],
      ['reject', 'Reject artifact'],
      ['unsure', 'Mark unsure']
    ],
    event: [
      ['accept', 'Accept event'],
      ['reject', 'Reject event'],
      ['unsure', 'Mark unsure']
    ],
    suggestion: [
      ['missed', 'Missed neuron'],
      ['artifact', 'Artifact'],
      ['unsure', 'Mark unsure'],
      ['promote', 'Promote']
    ]
  }[task.task_type] || [];
  return `
    <div class="guidedQuickActions" aria-label="Guided task decisions">
      ${actions.map(([action, label]) => `<button type="button" data-guided-action="${escapeHtml(action)}">${escapeHtml(label)}</button>`).join('')}
    </div>`;
}

function advanceGuidedAfterDecision(){
  const tasks = guidedTasks();
  if(!tasks.length) {
    setSetting('guidedTaskIndex', 0);
    renderAll();
    return;
  }
  const idx = Math.max(0, Math.min(tasks.length - 1, Number(setting('guidedTaskIndex')) || 0));
  setSetting('guidedTaskIndex', idx);
  selectGuidedTask(tasks[idx]);
}

function applyGuidedAction(action){
  const task = currentGuidedTask();
  if(!task) return;
  selectGuidedTask(task);
  if(task.task_type === 'roi' && ['accept','reject','unsure'].includes(action)) setRoiState(action);
  else if(task.task_type === 'event' && ['accept','reject','unsure'].includes(action)) setEventState(action);
  else if(task.task_type === 'suggestion') {
    if(['missed','artifact','unsure'].includes(action)) setSuggestionState(action);
    else if(action === 'promote') promoteSuggestion();
  }
  recordAction(`guided_quick_${task.task_type}_${action}`);
  advanceGuidedAfterDecision();
}

function applySettingsToControls() {
  const pairs = [
    ['eventThreshold', 'eventThresholdLabel', 1],
    ['kalmanGain', 'kalmanGainLabel', 3],
    ['spikeGain', 'spikeGainLabel', 3],
    ['zoom', 'zoomLabel', 2],
    ['brightness', 'brightnessLabel', 2],
    ['contrast', 'contrastLabel', 2],
    ['overlayOpacity', 'overlayOpacityLabel', 2],
    ['selectedFillOpacity', 'selectedFillOpacityLabel', 2],
    ['selectedOutlineWidth', 'selectedOutlineWidthLabel', 1],
    ['neighborRadiusPx', 'neighborRadiusPxLabel', 0],
    ['manualRoiRadius', 'manualRoiRadiusLabel', 0],
    ['roiEditBrushRadius', 'roiEditBrushRadiusLabel', 0],
    ['minArea', 'minAreaLabel', 0],
    ['minEvents', 'minEventsLabel', 0]
  ];
  for (const [id, label, digits] of pairs) {
    const el = document.getElementById(id);
    if (!el) continue;
    el.value = setting(id);
    const labelEl = document.getElementById(label);
    if(labelEl) labelEl.textContent = Number(setting(id)).toFixed(digits);
  }
  document.getElementById('queueSelect').value = setting('queue');
  const eventQueueSelect = document.getElementById('eventQueueSelect');
  if(eventQueueSelect) eventQueueSelect.value = setting('eventQueue') || 'all';
  document.getElementById('discoveryQueueSelect').value = setting('discoveryQueue') || 'all';
  document.getElementById('evidenceSelect').value = setting('evidenceMap') || '';
  document.getElementById('showEvidence').checked = Boolean(setting('showEvidence'));
  document.getElementById('showSuggestions').checked = Boolean(setting('showSuggestions'));
  const overlayPresetSelect = document.getElementById('overlayPresetSelect');
  if(overlayPresetSelect) overlayPresetSelect.value = setting('overlayPreset') || 'validate';
  const selectedOverlayMode = document.getElementById('selectedOverlayMode');
  if(selectedOverlayMode) selectedOverlayMode.value = setting('selectedOverlayMode') || 'outline';
  const roiFocusMode = document.getElementById('roiFocusMode');
  if(roiFocusMode) roiFocusMode.value = setting('roiFocusMode') || 'all';
  const uiMode = document.getElementById('uiMode');
  if(uiMode) uiMode.value = setting('uiMode') || 'basic';
  for(const select of document.querySelectorAll('.themeSelect')) select.value = setting('theme') || 'system';
  const reviewerIdInput = document.getElementById('reviewerIdInput');
  if(reviewerIdInput) reviewerIdInput.value = setting('reviewerId') || '';
  const manualRoiMode = document.getElementById('manualRoiMode');
  if(manualRoiMode) manualRoiMode.value = setting('manualRoiMode') || 'select';
  const roiEditMode = document.getElementById('roiEditMode');
  if(roiEditMode) roiEditMode.value = setting('roiEditMode') || 'off';
  const workflowPreset = document.getElementById('reviewWorkflowPreset');
  if(workflowPreset) workflowPreset.value = setting('reviewWorkflowPreset') || 'custom';
  renderBookmarkControls();
  renderSnapshotControls();
  renderRecoveryControls();
  applyUiMode();
  applyTheme();
  applyDisplaySettings();
}

function applyUiMode(){
  const mode = setting('uiMode') === 'advanced' ? 'advanced' : 'basic';
  appRoot.classList.toggle('basic-ui', mode === 'basic');
  appRoot.classList.toggle('advanced-ui', mode === 'advanced');
}

function resolvedTheme(){
  const theme = setting('theme') || 'system';
  if(theme === 'dark' || theme === 'light') return theme;
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function applyTheme(){
  const theme = setting('theme') || 'system';
  const resolved = resolvedTheme();
  document.documentElement.dataset.theme = resolved;
  appRoot.dataset.themePreference = theme;
  for(const select of document.querySelectorAll('.themeSelect')) select.value = theme;
}

function populateEvidenceSelect(){
  const select = document.getElementById('evidenceSelect');
  if(!select) return;
  select.innerHTML = '';
  for(const m of data.discovery?.evidenceMaps || []){
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.label;
    select.appendChild(opt);
  }
}

function applyDisplaySettings() {
  img.style.width = `${data.video.width * Number(setting('zoom'))}px`;
  img.style.filter = `brightness(${setting('brightness')}) contrast(${setting('contrast')})`;
  evidenceImg.style.width = img.style.width;
  const evidenceMap = (data.discovery?.evidenceMaps || []).find(m => m.id === setting('evidenceMap'));
  evidenceImg.src = evidenceMap ? evidenceMap.file : '';
  evidenceImg.style.opacity = setting('showEvidence') ? '0.58' : '0';
  ctx.globalAlpha = Number(setting('overlayOpacity'));
  resizeOverlay();
}

function refreshReviewAfterDataChange(){
  clearTraceCaches('review-data-change');
  slider.max = data.video.frames;
  traceView = {start: 1, end: Math.max(1, Number(data.video?.frames) || 1), dragging: false};
  if(currentFrame > data.video.frames) currentFrame = data.video.frames;
  selectedId = data.rois?.[0]?.id || null;
  selectedRoiIds = new Set(selectedId ? [String(selectedId)] : []);
  selectedEventFrame = selectedId ? eventsForRoi(selectedRoi())?.[0]?.frame || null : null;
  selectedSuggestionId = data.discovery?.suggestions?.[0]?.id || null;
  populateEvidenceSelect();
  if(!(data.discovery?.evidenceMaps || []).some(m => m.id === setting('evidenceMap'))) annotations.settings.evidenceMap = data.discovery?.evidenceMaps?.[0]?.id || '';
  applySettingsToControls();
  renderParams();
  setFrame(currentFrame || 1);
  renderAll();
  renderRunSyncControls();
}

function renderRunSyncControls(){
  const select = document.getElementById('activeRunSelect');
  const status = document.getElementById('activeRunStatus');
  const panel = document.getElementById('runGeneratePanel');
  const loadBtn = document.getElementById('loadRunReviewBtn');
  const openBtn = document.getElementById('openRunViewBtn');
  const previewBtn = document.getElementById('previewRunViewBtn');
  const generateBtn = document.getElementById('generateRunViewBtn');
  const unlockBtn = document.getElementById('unlockGenerationBtn');
  const refreshBtn = document.getElementById('refreshRunBtn');
  if(!select || !status || !panel) return;
  const runs = architectureRuns();
  const activeId = activeRunId();
  select.innerHTML = runs.map(run => `<option value="${escapeHtml(run.run_id)}" ${run.run_id === activeId ? 'selected' : ''}>${escapeHtml(runLabel(run))}</option>`).join('');
  const run = runById(activeId) || runs[0] || null;
  status.textContent = runStatusLabel(run);
  const canLoad = runGenerated(run) && Boolean(artifactUrl(run.artifacts?.review_data));
  const canOpen = Boolean(runAppUrl(run));
  if(loadBtn) loadBtn.disabled = !canLoad;
  if(openBtn) openBtn.disabled = !canOpen;
  const readiness = backendReadiness();
  const jobActive = currentGenerationJob && ['queued','running'].includes(currentGenerationJob.status);
  if(previewBtn) {
    previewBtn.disabled = !run || jobActive || !readiness.ok;
    previewBtn.textContent = jobActive && currentGenerationJob?.preview ? 'Previewing...' : 'Generate Preview';
  }
  if(generateBtn) {
    generateBtn.disabled = !run || runGenerated(run) || jobActive || !readiness.ok;
    generateBtn.textContent = jobActive ? 'Generating...' : 'Generate View';
  }
  if(unlockBtn) {
    const needsToken = Boolean(generationEnvironment?.owner_token_required);
    unlockBtn.classList.toggle('hidden', !needsToken);
    unlockBtn.textContent = generationOwnerToken ? 'Generation Unlocked' : 'Unlock Generation';
  }
  if(refreshBtn) refreshBtn.disabled = !serverBacked;
  if(run && (!runGenerated(run) || currentGenerationJob)){
    panel.classList.remove('hidden');
    const jobHtml = currentGenerationJob ? generationJobHtml(currentGenerationJob) : '';
    panel.innerHTML = `
      <div>
        <b>${escapeHtml(runLabel(run))}</b>
        <p class="hint">This run is configured, but the Review/QC frame outputs have not been generated or attached yet.</p>
        <p class="hint">${escapeHtml(readiness.text)}</p>
      </div>
      <div>
        ${jobHtml}
        <details ${currentGenerationJob ? 'open' : ''}>
          <summary>Fallback command</summary>
          <pre>${escapeHtml(generationCommandForRun(run))}</pre>
        </details>
      </div>`;
  } else {
    panel.classList.add('hidden');
    panel.innerHTML = '';
  }
}

function generationJobHtml(job){
  const logs = (job.log_tail || []).slice(-20).join('\n');
  const cls = job.status === 'completed' ? 'ok' : job.status === 'failed' || job.status === 'blocked' ? 'bad' : 'warn';
  return `
    <div class="jobStatusBox ${cls}">
      <div class="componentTitle">
        <h4>Generation job ${escapeHtml(job.job_id || '')}</h4>
        <span class="stageStatus ${job.status === 'completed' ? 'ok' : job.status === 'failed' || job.status === 'blocked' ? 'bad' : 'warn'}">${escapeHtml(job.status || 'unknown')}</span>
      </div>
      <p class="hint">Stage: ${escapeHtml(job.stage || 'n/a')} | Backend: ${escapeHtml(job.backend || 'auto')}</p>
      ${job.error ? `<div class="qcWarning">${escapeHtml(job.error)}</div>` : ''}
      <pre>${escapeHtml(logs || 'Waiting for logs...')}</pre>
    </div>`;
}

async function startGenerationJob({preview=false}={}){
  const run = activeRun();
  if(!run || !serverBacked) return;
  const backend = document.getElementById('generationBackend')?.value || 'auto';
  const readiness = backendReadiness();
  if(!readiness.ok) {
    setSaveState(readiness.text, 'bad');
    renderRunSyncControls();
    return;
  }
  try {
    const job = await fetchJson(apiUrl(preview ? 'jobs/generate-preview' : 'jobs/generate-view'), {
      method:'POST',
      headers:generationHeaders(),
      body:JSON.stringify({
        run_id: run.run_id,
        dataset_id: run.dataset_id || datasetId,
        backend,
        stages: preview ? 'high-pass,event-denoise,candidates,temporal-scoring,review-data,proposal-analysis,workbench' : 'all',
        generate_intermediates: true,
        preview,
        force: false
      })
    });
    currentGenerationJob = job;
    setSaveState(preview ? 'preview generation started' : 'generation started', 'ok');
    renderRunSyncControls();
    pollGenerationJob(job.job_id);
  } catch (err) {
    currentGenerationJob = err.payload?.job || null;
    setSaveState(err.message || 'generation failed to start', 'bad');
    renderRunSyncControls();
    if(currentGenerationJob?.job_id) pollGenerationJob(currentGenerationJob.job_id);
  }
}

async function pollGenerationJob(jobId){
  clearTimeout(generationPollTimer);
  if(!jobId) return;
  try {
    currentGenerationJob = await fetchJson(apiUrl(`jobs/${jobId}`));
    renderRunSyncControls();
    if(['queued','running'].includes(currentGenerationJob.status)) {
      generationPollTimer = setTimeout(() => pollGenerationJob(jobId), 1500);
    } else {
      await refreshArchitectureRuns();
      if(currentGenerationJob.status === 'completed') {
        const run = activeRun();
        if(runGenerated(run)) await loadReviewForRun(run);
      }
      renderRunSyncControls();
    }
  } catch (_) {
    generationPollTimer = setTimeout(() => pollGenerationJob(jobId), 3000);
  }
}

async function loadReviewForRun(run){
  if(!runGenerated(run)) {
    renderRunSyncControls();
    return;
  }
  try {
    data = await fetchReviewDataForRun(run);
    setSaveState(`loaded ${runLabel(run)}`, 'ok');
    refreshReviewAfterDataChange();
  } catch (_) {
    setSaveState('could not load generated review data', 'bad');
    renderRunSyncControls();
  }
}

async function selectActiveRun(runId, {loadReview=false}={}){
  const run = runById(runId);
  captureActiveRunAnnotations();
  annotations.settings.activeRunId = runId || baselineRunId();
  annotations.settings.qcRunId = annotations.settings.activeRunId;
  materializeRunAnnotations(activeRunId());
  if(loadReview && runGenerated(run)) await loadReviewForRun(run);
  else {
    renderRunSyncControls();
    renderAll();
    updateQcFrameView();
  }
  queueSave();
}

async function refreshArchitectureRuns(){
  if(!serverBacked) return;
  try {
    const res = await fetch('architecture_runs.json', {cache:'no-store'});
    if(!res.ok) throw new Error(await res.text());
    data.architectureRuns = await res.json();
    renderRunSyncControls();
    renderArchitectureLab();
    renderDatasetQc();
    setSaveState('refreshed architecture runs', 'ok');
  } catch (_) {
    setSaveState('could not refresh architecture runs', 'bad');
  }
}

function resizeOverlay(){
  const rect = img.getBoundingClientRect();
  overlay.width = data.video.width;
  overlay.height = data.video.height;
  overlay.style.width = rect.width + 'px';
  overlay.style.height = rect.height + 'px';
  drawOverlay();
}

function selectedOverlayFillAlpha(isEvent=false){
  const mode = setting('selectedOverlayMode') || 'outline';
  const fill = Math.max(0, Math.min(1, Number(setting('selectedFillOpacity')) || 0));
  if(mode === 'outline') return isEvent ? Math.max(0.08, fill * 0.7) : Math.min(0.04, fill * 0.35);
  if(mode === 'event') return isEvent ? Math.max(0.42, fill) : Math.min(0.05, fill * 0.4);
  return fill;
}

function selectedOverlayStrokeColor(color, isEvent=false, isMultiSel=false){
  if(isEvent) return '#facc15';
  if(isMultiSel) return '#22c55e';
  return color === '#38bdf8' ? '#e0f2fe' : color;
}

function applyOverlayPreset(name){
  const preset = OVERLAY_PRESETS[name];
  if(!preset) return;
  setSetting('overlayPreset', name);
  for(const key of ['selectedOverlayMode','selectedFillOpacity','selectedOutlineWidth','overlayOpacity','showEvidence','showSuggestions']) {
    if(Object.prototype.hasOwnProperty.call(preset, key)) setSetting(key, preset[key]);
  }
  for(const [id, value] of [['showLabels', preset.showLabels], ['showEvents', preset.showEvents], ['showSuggestions', preset.showSuggestions], ['showEvidence', preset.showEvidence]]) {
    const el = document.getElementById(id);
    if(el && value !== undefined) el.checked = Boolean(value);
  }
  applySettingsToControls();
  renderAll();
}

function setCheckbox(id, value){
  const el = document.getElementById(id);
  if(el) el.checked = Boolean(value);
}

function applyReviewWorkflowPreset(name){
  const preset = REVIEW_WORKFLOW_PRESETS[name];
  if(!preset) {
    setSetting('reviewWorkflowPreset', 'custom');
    applySettingsToControls();
    return;
  }
  setSetting('reviewWorkflowPreset', name);
  for(const key of ['queue','discoveryQueue','roiFocusMode','reviewMode','selectedOverlayMode','showEvidence','showSuggestions','uiMode']) {
    if(Object.prototype.hasOwnProperty.call(preset, key)) setSetting(key, preset[key]);
  }
  if(preset.overlayPreset) applyOverlayPreset(preset.overlayPreset);
  setCheckbox('showLabels', preset.showLabels);
  setCheckbox('showEvents', preset.showEvents);
  setCheckbox('showSuggestions', preset.showSuggestions);
  setCheckbox('showEvidence', preset.showEvidence);
  if(name === 'missed_neuron_search') {
    const details = document.getElementById('discoveryDetails');
    if(details) details.open = true;
  }
  if(name === 'mask_editing') setSetting('manualRoiMode', 'select');
  const first = visibleRois()[0];
  if(first && !selectedRoi()) selectedId = first.id;
  recordAction(`workflow_preset_${name}`);
  applySettingsToControls();
  renderAll();
  setSaveState(`workflow preset: ${preset.label}`, 'ok');
}

function toggleShortcutHelp(force=null){
  const overlayEl = document.getElementById('shortcutOverlay');
  if(!overlayEl) return;
  const shouldOpen = force === null ? overlayEl.classList.contains('hidden') : Boolean(force);
  overlayEl.classList.toggle('hidden', !shouldOpen);
  if(shouldOpen) document.getElementById('shortcutCloseBtn')?.focus();
}

function drawOverlay(){
  ctx.clearRect(0,0,overlay.width,overlay.height);
  const showRois = document.getElementById('showRois').checked;
  const showLabels = document.getElementById('showLabels').checked;
  const showEvents = document.getElementById('showEvents').checked;
  const showSuggestions = document.getElementById('showSuggestions').checked;
  if(!showRois && !showSuggestions) return;
  const opacity = Number(setting('overlayOpacity'));
  if(showRois) for(const roi of visibleOverlayRois()){
    const ann = roiAnn(roi.id);
    const isSel = roi.id === selectedId;
    const isMultiSel = selectedRoiIds.has(String(roi.id));
    const isEvent = showEvents && eventNearFrame(roi, currentFrame);
    let color = ann.state === 'accept' ? '#16a34a' : ann.state === 'reject' ? '#dc2626' : ann.state === 'unsure' ? '#9333ea' : '#38bdf8';
    if(isEvent) color = '#facc15';
    const fillAlpha = isSel || isMultiSel ? selectedOverlayFillAlpha(isEvent) : opacity;
    ctx.globalAlpha = fillAlpha;
    ctx.fillStyle = color;
    if(fillAlpha > 0.005) for(const p of roi.points){ ctx.fillRect(p[0], p[1], 1, 1); }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = isSel || isMultiSel ? selectedOverlayStrokeColor(color, isEvent, isMultiSel) : color;
    ctx.lineWidth = isSel || isMultiSel ? Number(setting('selectedOutlineWidth')) || 2.5 : 1;
    const r = Math.max(4, Math.sqrt(roi.area / Math.PI) + 2);
    ctx.beginPath(); ctx.arc(roi.centroidX, roi.centroidY, r, 0, Math.PI*2); ctx.stroke();
    if(showLabels){
      ctx.font = '10px Arial';
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = '#111827';
      ctx.lineWidth = 3;
      ctx.strokeText(String(roi.id), roi.centroidX + 5, roi.centroidY - 5);
      ctx.fillText(String(roi.id), roi.centroidX + 5, roi.centroidY - 5);
    }
  }
  if(showRois) drawSplitMergeGuides(showLabels, opacity);
  if(showSuggestions){
    for(const s of visibleSuggestions()){
      const ann = suggestionAnn(s.id);
      const isSel = s.id === selectedSuggestionId;
      let color = ann.state === 'promoted' || annotations.promotedRois[s.id] ? '#16a34a' :
        ann.state === 'artifact' ? '#dc2626' :
        ann.state === 'missed' ? '#facc15' :
        ann.state === 'unsure' ? '#9333ea' : '#fb7185';
      ctx.globalAlpha = isSel ? Math.max(0.12, selectedOverlayFillAlpha(false)) : Math.max(0.38, opacity * 0.82);
      ctx.fillStyle = color;
      for(const p of s.points || []) ctx.fillRect(p[0], p[1], 1, 1);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = isSel ? '#ffffff' : color;
      ctx.lineWidth = isSel ? 2 : 1;
      const r = Math.max(5, Math.sqrt((s.area || 20) / Math.PI) + 3);
      ctx.beginPath(); ctx.arc(s.centroidX, s.centroidY, r, 0, Math.PI*2); ctx.stroke();
      if(showLabels){
        ctx.font = '10px Arial';
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#111827';
        ctx.lineWidth = 3;
        ctx.strokeText(String(s.id), s.centroidX + 5, s.centroidY - 5);
        ctx.fillText(String(s.id), s.centroidX + 5, s.centroidY - 5);
      }
    }
  }
  drawManualPreview();
}

function drawSplitMergeGuides(showLabels, opacity){
  const virtuals = Object.values(annotations.virtualRois || {});
  const decisions = Object.values(annotations.splitMergeDecisions || {});
  for(const virtual of virtuals){
    if((virtual.roi_kind || '').startsWith('manual_') || virtual.roi_kind === 'virtual_merge') continue;
    if(!virtual.points?.length) continue;
    ctx.globalAlpha = Math.max(0.28, opacity * 0.55);
    ctx.fillStyle = '#14b8a6';
    for(const p of virtual.points) ctx.fillRect(p[0], p[1], 1, 1);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = '#0f766e';
    ctx.lineWidth = 2;
    const r = Math.max(6, Math.sqrt((virtual.area || virtual.points.length) / Math.PI) + 4);
    ctx.beginPath(); ctx.arc(virtual.centroidX, virtual.centroidY, r, 0, Math.PI*2); ctx.stroke();
    if(showLabels) drawOverlayLabel(virtual.id || 'merge', virtual.centroidX + 5, virtual.centroidY + 9, '#ccfbf1');
  }
  ctx.save();
  ctx.setLineDash([4, 3]);
  for(const decision of decisions){
    const color = decision.decision_type === 'split' ? '#f97316' : '#14b8a6';
    const sourceRois = (decision.source_roi_ids || []).map(roiById).filter(Boolean);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    for(const roi of sourceRois){
      const r = Math.max(7, Math.sqrt(roi.area / Math.PI) + 5);
      ctx.beginPath(); ctx.arc(roi.centroidX, roi.centroidY, r, 0, Math.PI*2); ctx.stroke();
      if(showLabels) drawOverlayLabel(decision.decision_type || 'edit', roi.centroidX + 6, roi.centroidY + 12, color);
    }
  }
  ctx.restore();
}

function drawOverlayLabel(label, x, y, color){
  ctx.font = '10px Arial';
  ctx.fillStyle = '#ffffff';
  ctx.strokeStyle = '#111827';
  ctx.lineWidth = 3;
  ctx.strokeText(String(label), x, y);
  ctx.fillStyle = color || '#ffffff';
  ctx.fillText(String(label), x, y);
}

function overlayPointFromEvent(e){
  const rect = overlay.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(data.video.width - 1, (e.clientX - rect.left) * data.video.width / rect.width)),
    y: Math.max(0, Math.min(data.video.height - 1, (e.clientY - rect.top) * data.video.height / rect.height))
  };
}

function circlePoints(cx, cy, radius){
  const r = Math.max(1, Math.round(radius));
  const points = [];
  const x0 = Math.max(0, Math.floor(cx - r));
  const x1 = Math.min(data.video.width - 1, Math.ceil(cx + r));
  const y0 = Math.max(0, Math.floor(cy - r));
  const y1 = Math.min(data.video.height - 1, Math.ceil(cy + r));
  for(let y=y0;y<=y1;y++) for(let x=x0;x<=x1;x++){
    const dx = x - cx, dy = y - cy;
    if(dx * dx + dy * dy <= r * r) points.push([x, y]);
  }
  return points;
}

function pointInPolygon(x, y, polygon){
  let inside = false;
  for(let i=0, j=polygon.length - 1; i<polygon.length; j=i++){
    const xi = polygon[i].x, yi = polygon[i].y;
    const xj = polygon[j].x, yj = polygon[j].y;
    const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / Math.max(1e-6, yj - yi) + xi);
    if(intersect) inside = !inside;
  }
  return inside;
}

function lassoPoints(path){
  if(!path || path.length < 3) return [];
  const xs = path.map(p => p.x), ys = path.map(p => p.y);
  const x0 = Math.max(0, Math.floor(Math.min(...xs)));
  const x1 = Math.min(data.video.width - 1, Math.ceil(Math.max(...xs)));
  const y0 = Math.max(0, Math.floor(Math.min(...ys)));
  const y1 = Math.min(data.video.height - 1, Math.ceil(Math.max(...ys)));
  const points = [];
  for(let y=y0;y<=y1;y++) for(let x=x0;x<=x1;x++) if(pointInPolygon(x + 0.5, y + 0.5, path)) points.push([x, y]);
  return points;
}

function geometrySummary(points){
  const unique = new Map();
  for(const p of points || []) {
    const x = Math.max(0, Math.min(data.video.width - 1, Math.round(p[0])));
    const y = Math.max(0, Math.min(data.video.height - 1, Math.round(p[1])));
    unique.set(`${x},${y}`, [x, y]);
  }
  const out = [...unique.values()];
  if(!out.length) return null;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity, sumX = 0, sumY = 0;
  for(const [x, y] of out){
    minX = Math.min(minX, x); minY = Math.min(minY, y);
    maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
    sumX += x; sumY += y;
  }
  return {points: out, bbox: [minX, minY, maxX, maxY], area: out.length, centroidX: Number((sumX / out.length).toFixed(1)), centroidY: Number((sumY / out.length).toFixed(1))};
}

function createManualRoi(kind, points, label='Manual ROI'){
  const summary = geometrySummary(points);
  if(!summary) return null;
  const id = `MR_${Date.now().toString(36)}`;
  const item = Object.assign({
    id,
    roi_kind: kind,
    source_roi_ids: [],
    provenance: 'manual_overlay',
    createdAt: new Date().toISOString(),
    cell_state: 'accepted',
    trace_quality: '',
    control_ready: '',
    artifact_class: '',
    identity_group: '',
    needs_action: '',
    reason_tags: ['manual'],
    confidence: 'medium',
    notes: label
  }, summary);
  annotations.virtualRois[id] = stampAnnotation(item);
  annotations.rois[id] = stampAnnotation(migrateRoiAnn({state:'accept', cell_state:'accepted', reason_tags:['manual'], confidence:'medium', notes:label}));
  selectedId = id;
  selectedRoiIds = new Set([String(id)]);
  recordAction(`manual_roi_${kind}`);
  queueSave();
  renderAll();
  return item;
}

function pointMap(points){
  const map = new Map();
  for(const p of points || []) {
    const x = Math.max(0, Math.min(data.video.width - 1, Math.round(p[0])));
    const y = Math.max(0, Math.min(data.video.height - 1, Math.round(p[1])));
    map.set(`${x},${y}`, [x, y]);
  }
  return map;
}

function roiGeometrySnapshot(roi, reason='edit'){
  return {
    reason,
    createdAt: new Date().toISOString(),
    points: (roi.points || []).map(p => [Number(p[0]), Number(p[1])]),
    bbox: Array.isArray(roi.bbox) ? [...roi.bbox] : [],
    area: roi.area,
    centroidX: roi.centroidX,
    centroidY: roi.centroidY
  };
}

function pushRoiEditHistory(roi, reason='brush'){
  if(!roi || !annotations.virtualRois[roi.id]) return;
  const history = Array.isArray(roi.edit_history) ? roi.edit_history : [];
  const previous = history[history.length - 1];
  const snapshot = roiGeometrySnapshot(roi, reason);
  if(previous && JSON.stringify(previous.points || []) === JSON.stringify(snapshot.points || [])) return;
  roi.edit_history = [...history, snapshot].slice(-20);
}

function clearMaterializedTraceFields(roi){
  for(const key of ['rawTrace','backgroundTrace','dffTrace','baselineTrace','eventTrace','zTrace','events','noiseSigma','traceSnr','backgroundCorrelation','eventSupport','trace_materialized','trace_materialized_at','trace_materialization']){
    delete roi[key];
  }
}

function ensureEditableRoi(roi){
  if(!roi || !roi.points?.length) return null;
  if(annotations.virtualRois[roi.id]) return annotations.virtualRois[roi.id];
  const summary = geometrySummary(roi.points);
  if(!summary) return null;
  const sourceId = String(roi.id);
  const id = `EDIT_${sourceId}_${Date.now().toString(36)}`;
  const sourceAnn = roiAnn(sourceId);
  const item = Object.assign({
    id,
    roi_kind: 'manual_edit',
    source_roi_ids: [sourceId],
    provenance: 'roi_brush_edit',
    createdAt: new Date().toISOString(),
    cell_state: sourceAnn.cell_state || '',
    trace_quality: sourceAnn.trace_quality || '',
    control_ready: sourceAnn.control_ready || '',
    artifact_class: sourceAnn.artifact_class || '',
    identity_group: sourceAnn.identity_group || '',
    needs_action: sourceAnn.needs_action || 'mask_refined',
    reason_tags: [...new Set([...(sourceAnn.reason_tags || []), 'manual'])],
    confidence: sourceAnn.confidence || 'medium',
    notes: sourceAnn.notes || `Edited mask copied from ROI ${sourceId}`
  }, summary);
  annotations.virtualRois[id] = stampAnnotation(item);
  annotations.rois[id] = stampAnnotation(migrateRoiAnn(Object.assign({}, sourceAnn, {
    notes: item.notes,
    needs_action: item.needs_action,
    reason_tags: item.reason_tags,
    confidence: item.confidence
  })));
  selectedId = id;
  selectedRoiIds = new Set([String(id)]);
  return item;
}

function updateVirtualRoiGeometry(id, points){
  const summary = geometrySummary(points);
  if(!summary || summary.area < 2) return null;
  const roi = annotations.virtualRois[id];
  if(!roi) return null;
  clearMaterializedTraceFields(roi);
  Object.assign(roi, summary, {updatedAt: new Date().toISOString(), roi_kind: roi.roi_kind || 'manual_edit'});
  annotations.rois[id] = migrateRoiAnn(Object.assign({}, annotations.rois[id] || {}, {
    needs_action: roi.needs_action || 'mask_refined',
    reason_tags: [...new Set([...(roi.reason_tags || []), 'manual'])],
    confidence: roi.confidence || 'medium'
  }));
  return roi;
}

function restoreRoiGeometry(id, snapshot, reason='restore'){
  const roi = annotations.virtualRois[id];
  if(!roi || !snapshot?.points?.length) return null;
  clearMaterializedTraceFields(roi);
  Object.assign(roi, {
    points: snapshot.points.map(p => [Number(p[0]), Number(p[1])]),
    bbox: Array.isArray(snapshot.bbox) && snapshot.bbox.length === 4 ? [...snapshot.bbox] : geometrySummary(snapshot.points)?.bbox,
    area: snapshot.area,
    centroidX: snapshot.centroidX,
    centroidY: snapshot.centroidY,
    updatedAt: new Date().toISOString(),
    needs_action: roi.needs_action || 'mask_refined'
  });
  if(reason) roi.last_edit_reason = reason;
  selectedId = id;
  selectedRoiIds = new Set([String(id)]);
  selectedEventFrame = null;
  queueSave();
  renderAll();
  return roi;
}

function undoRoiEdit(){
  const roi = selectedRoi();
  const virtual = roi ? annotations.virtualRois[roi.id] : null;
  if(!virtual?.edit_history?.length) {
    setSaveState('no mask edit history for selected ROI', 'bad');
    return;
  }
  const snapshot = virtual.edit_history.pop();
  const restored = restoreRoiGeometry(virtual.id, snapshot, 'undo');
  if(restored) {
    recordAction('roi_edit_undo');
    setSaveState(`restored previous mask for ROI ${restored.id}`, 'ok');
  }
}

function revertEditedRoiToSource(){
  const roi = selectedRoi();
  const virtual = roi ? annotations.virtualRois[roi.id] : null;
  const sourceId = virtual?.source_roi_ids?.[0];
  const source = sourceId ? data.rois.find(item => String(item.id) === String(sourceId)) : null;
  if(!virtual || !source?.points?.length) {
    setSaveState('selected ROI has no source mask to revert to', 'bad');
    return;
  }
  pushRoiEditHistory(virtual, 'before source revert');
  const restored = restoreRoiGeometry(virtual.id, roiGeometrySnapshot(source, 'source'), 'revert_to_source');
  if(restored) {
    recordAction('roi_edit_revert_to_source');
    setSaveState(`reverted ROI ${restored.id} to source ${sourceId}`, 'ok');
  }
}

function applyRoiBrush(point, editableOverride=null){
  const mode = setting('roiEditMode') || 'off';
  if(!['brush_add','brush_erase'].includes(mode)) return null;
  const selected = selectedRoi();
  const editable = editableOverride || ensureEditableRoi(selected);
  if(!editable) return null;
  const radius = Number(setting('roiEditBrushRadius')) || 4;
  const brush = circlePoints(point.x, point.y, radius);
  const map = pointMap(editable.points);
  if(mode === 'brush_add') {
    for(const p of brush) map.set(`${p[0]},${p[1]}`, p);
  } else {
    for(const p of brush) map.delete(`${p[0]},${p[1]}`);
  }
  const updated = updateVirtualRoiGeometry(editable.id, [...map.values()]);
  if(updated) {
    selectedId = updated.id;
    selectedRoiIds = new Set([String(updated.id)]);
    selectedEventFrame = null;
    queueSave();
    renderAll();
    statusEl.textContent = `Edited ROI ${updated.id} (${updated.area} px)`;
  }
  return updated;
}

function drawManualPreview(){
  if(!manualRoiState.preview && !manualRoiState.points.length) return;
  ctx.save();
  ctx.globalAlpha = 1;
  ctx.strokeStyle = '#f97316';
  ctx.fillStyle = 'rgba(249, 115, 22, 0.16)';
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 3]);
  const preview = manualRoiState.preview;
  if(preview?.type === 'circle') {
    ctx.beginPath();
    ctx.arc(preview.x, preview.y, preview.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  } else if(manualRoiState.points.length) {
    ctx.beginPath();
    manualRoiState.points.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    ctx.stroke();
  }
  ctx.restore();
}

function cancelManualRoi(){
  manualRoiState = {drawing:false, start:null, points:[], preview:null, suppressClick:false};
  setSetting('manualRoiMode', 'select');
  applySettingsToControls();
  drawOverlay();
}

async function materializeManualTraces(){
  if(!serverBacked) {
    setSaveState('trace materialization requires local server mode', 'bad');
    return;
  }
  const ids = Object.values(annotations.virtualRois || {})
    .filter(roi => roi?.points?.length && !Array.isArray(roi.dffTrace))
    .map(roi => String(roi.id))
    .filter(Boolean);
  if(!ids.length) {
    setSaveState('no unmaterialized manual ROI traces', 'ok');
    return;
  }
  captureActiveRunAnnotations();
  setSaveState(`materializing ${ids.length} manual ROI trace${ids.length === 1 ? '' : 's'}...`, '');
  try {
    const payload = await fetchJson(apiUrl('materialize-traces'), {
      method:'POST',
      headers:generationHeaders(),
      body:JSON.stringify({
        run_id: activeRunId(),
        roi_ids: ids,
        annotations,
        outer_radius_px: 15,
        neuropil_weight: 0.7,
        event_threshold_z: threshold(),
        kalman_gain: kalmanGain(),
        spike_gain: spikeGain(),
        negative_gain: 0.11
      })
    });
    mergeAnnotations(payload.annotations);
    ensureRunAnnotationScope();
    clearTraceCaches('manual-trace-materialization');
    localStorage.setItem(storeKey, JSON.stringify(annotations));
    applySettingsToControls();
    renderAll();
    setSaveState(`materialized ${payload.materialized_ids?.length || 0} manual ROI trace${(payload.materialized_ids?.length || 0) === 1 ? '' : 's'}`, 'ok');
  } catch (err) {
    setSaveState(err.message || 'manual ROI trace materialization failed', 'bad');
  }
}

function traceBounds(){
  const frames = Math.max(1, Number(data.video?.frames) || 1);
  let start = Number(traceView.start);
  let end = Number(traceView.end);
  if(!Number.isFinite(start) || !Number.isFinite(end)) {
    start = 1;
    end = frames;
  }
  if(start > end) [start, end] = [end, start];
  start = Math.max(1, Math.min(frames, start));
  end = Math.max(1, Math.min(frames, end));
  if(frames > 1 && end - start < 1) end = Math.min(frames, start + 1);
  traceView.start = start;
  traceView.end = end;
  return {start, end};
}

function setTraceWindow(start, end){
  const frames = Math.max(1, Number(data.video?.frames) || 1);
  if(frames <= 1) {
    traceView.start = 1;
    traceView.end = 1;
    return;
  }
  const minSpan = Math.min(7, frames - 1);
  let span = Math.max(minSpan, end - start);
  span = Math.min(span, frames - 1);
  let nextStart = start;
  let nextEnd = start + span;
  if(nextStart < 1) {
    nextStart = 1;
    nextEnd = nextStart + span;
  }
  if(nextEnd > frames) {
    nextEnd = frames;
    nextStart = nextEnd - span;
  }
  traceView.start = Math.max(1, nextStart);
  traceView.end = Math.min(frames, nextEnd);
}

function resetTraceZoom(){
  setTraceWindow(1, Math.max(1, Number(data.video?.frames) || 1));
  drawTrace();
}

function ensureTraceFrameVisible(frame){
  const frames = Math.max(1, Number(data.video?.frames) || 1);
  const bounds = traceBounds();
  if(frame >= bounds.start && frame <= bounds.end) return;
  const span = Math.max(1, bounds.end - bounds.start);
  const nextStart = Math.max(1, Math.min(frames - span, frame - span / 2));
  setTraceWindow(nextStart, nextStart + span);
}

function traceXForFrame(frame, width=traceCanvas.width, pad=TRACE_PAD){
  const bounds = traceBounds();
  if(bounds.end <= bounds.start) return pad;
  return pad + (frame - bounds.start) * (width - 2 * pad) / (bounds.end - bounds.start);
}

function traceFrameFromX(x, width=traceCanvas.width, pad=TRACE_PAD){
  const bounds = traceBounds();
  const plotW = Math.max(1, width - 2 * pad);
  const ratio = Math.max(0, Math.min(1, (x - pad) / plotW));
  return Math.max(1, Math.min(data.video.frames, Math.round(bounds.start + ratio * (bounds.end - bounds.start))));
}

function traceCanvasPoint(e){
  const rect = traceCanvas.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) * traceCanvas.width / rect.width,
    y: (e.clientY - rect.top) * traceCanvas.height / rect.height
  };
}

function updateTraceWindowText(){
  const el = document.getElementById('traceWindowText');
  if(!el) return;
  const bounds = traceBounds();
  el.textContent = `frames ${Math.round(bounds.start)}-${Math.round(bounds.end)}`;
}

function traceEventAtPoint(point, roi){
  if(!roi || point.y > TRACE_PAD + 24) return null;
  const bounds = traceBounds();
  let best = null;
  let bestD = Infinity;
  for(const ev of eventsForRoi(roi)){
    if(ev.frame < bounds.start || ev.frame > bounds.end) continue;
    const dx = point.x - traceXForFrame(ev.frame);
    const dy = point.y - (TRACE_PAD + 8);
    const d = dx * dx + dy * dy;
    if(d < bestD && d <= 144) {
      bestD = d;
      best = ev;
    }
  }
  return best;
}

function selectTraceEvent(ev, roi){
  if(!ev || !roi) return;
  selectedEventFrame = ev.frame;
  eventNotes.value = eventAnn(roi.id, selectedEventFrame).notes || '';
  setFrame(selectedEventFrame);
  renderAll();
}

function timelineEventCounts(){
  const frames = Math.max(1, Number(data.video?.frames) || 1);
  const counts = new Array(frames).fill(0);
  for(const roi of visibleRois()){
    if(roiAnn(roi.id).deleted) continue;
    for(const ev of eventsForRoi(roi)) if(ev.frame >= 1 && ev.frame <= frames) counts[ev.frame - 1]++;
  }
  return counts;
}

function timelineXForFrame(frame, width=eventTimelineCanvas?.width || 1, pad=TRACE_PAD){
  const frames = Math.max(1, Number(data.video?.frames) || 1);
  if(frames <= 1) return pad;
  return pad + (frame - 1) * (width - 2 * pad) / (frames - 1);
}

function timelineFrameFromX(x, width=eventTimelineCanvas?.width || 1, pad=TRACE_PAD){
  const frames = Math.max(1, Number(data.video?.frames) || 1);
  const plotW = Math.max(1, width - 2 * pad);
  const ratio = Math.max(0, Math.min(1, (x - pad) / plotW));
  return Math.max(1, Math.min(frames, Math.round(1 + ratio * (frames - 1))));
}

function drawEventTimeline(){
  if(!eventTimelineCanvas || !eventTimelineCtx) return;
  const w = eventTimelineCanvas.width, h = eventTimelineCanvas.height, pad = TRACE_PAD;
  const counts = timelineEventCounts();
  const maxCount = Math.max(1, ...counts);
  eventTimelineCtx.clearRect(0,0,w,h);
  eventTimelineCtx.fillStyle = '#ffffff';
  eventTimelineCtx.fillRect(0,0,w,h);
  eventTimelineCtx.strokeStyle = '#e2e8f0';
  eventTimelineCtx.beginPath();
  eventTimelineCtx.moveTo(pad, h - 16);
  eventTimelineCtx.lineTo(w - pad, h - 16);
  eventTimelineCtx.stroke();
  const barW = Math.max(1, (w - 2 * pad) / Math.max(1, counts.length));
  counts.forEach((count, i) => {
    if(!count) return;
    const x = timelineXForFrame(i + 1, w, pad);
    const barH = Math.max(2, (h - 28) * count / maxCount);
    eventTimelineCtx.fillStyle = count >= maxCount ? '#0284c7' : '#7dd3fc';
    eventTimelineCtx.fillRect(x, h - 16 - barH, barW, barH);
  });
  const xf = timelineXForFrame(currentFrame, w, pad);
  eventTimelineCtx.strokeStyle = '#ef4444';
  eventTimelineCtx.lineWidth = 1;
  eventTimelineCtx.beginPath();
  eventTimelineCtx.moveTo(xf, 8);
  eventTimelineCtx.lineTo(xf, h - 10);
  eventTimelineCtx.stroke();
  eventTimelineCtx.fillStyle = '#475569';
  eventTimelineCtx.font = '12px Arial';
  eventTimelineCtx.fillText(`${counts.reduce((sum, v) => sum + v, 0)} visible events`, pad, 13);
}

function drawTrace(){
  const roi = selectedRoi();
  const w = traceCanvas.width, h = traceCanvas.height;
  traceCtx.clearRect(0,0,w,h);
  traceCtx.fillStyle = '#fff'; traceCtx.fillRect(0,0,w,h);
  updateTraceWindowText();
  if(!roi) return;
  const pad = TRACE_PAD;
  if(!Array.isArray(roi.dffTrace) || roi.dffTrace.length < 3){
    traceCtx.fillStyle = '#0f172a'; traceCtx.font = '13px Arial';
    traceCtx.fillText(`ROI ${roi.id} | manual/virtual footprint | trace not materialized`, pad, 22);
    traceCtx.fillStyle = '#64748b'; traceCtx.font = '12px Arial';
    traceCtx.fillText('Manual ROIs are saved for review/export. Re-run materialization to extract fluorescence traces.', pad, 44);
    return;
  }
  const bounds = traceBounds();
  const startIdx = Math.max(0, Math.floor(bounds.start) - 1);
  const endIdx = Math.min(data.video.frames - 1, Math.ceil(bounds.end) - 1);
  const model = modeledTraceCached(roi);
  const zScaled = model.zTrace.map(v => v * 0.05);
  const vals = [roi.dffTrace.slice(startIdx, endIdx + 1), model.baselineTrace.slice(startIdx, endIdx + 1), zScaled.slice(startIdx, endIdx + 1)].flat();
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if(hi - lo < 1e-6){ hi = lo + 1; }
  function x(i){ return traceXForFrame(i + 1, w, pad); }
  function y(v){ return h - pad - (v - lo) * (h - 2*pad) / (hi - lo); }
  traceCtx.strokeStyle = '#e2e8f0'; traceCtx.lineWidth = 1;
  for(let k=0;k<5;k++){ const yy = pad + k*(h-2*pad)/4; traceCtx.beginPath(); traceCtx.moveTo(pad,yy); traceCtx.lineTo(w-pad,yy); traceCtx.stroke(); }
  const drawLine = (arr, color, width=1.6) => {
    traceCtx.strokeStyle=color; traceCtx.lineWidth=width; traceCtx.beginPath();
    for(let i=startIdx;i<=endIdx;i++){
      const v = arr[i];
      if(i===startIdx) traceCtx.moveTo(x(i),y(v));
      else traceCtx.lineTo(x(i),y(v));
    }
    traceCtx.stroke();
  };
  drawLine(roi.dffTrace, '#2563eb');
  drawLine(model.baselineTrace, '#64748b');
  drawLine(zScaled, '#f59e0b');
  traceCtx.strokeStyle = '#ef4444'; traceCtx.lineWidth = 1;
  const xf = traceXForFrame(currentFrame, w, pad); traceCtx.beginPath(); traceCtx.moveTo(xf,pad); traceCtx.lineTo(xf,h-pad); traceCtx.stroke();
  for(const ev of eventsForRoi(roi)){
    if(ev.frame < bounds.start || ev.frame > bounds.end) continue;
    const ann = eventAnn(roi.id, ev.frame);
    traceCtx.fillStyle = ann.state === 'accept' ? '#16a34a' : ann.state === 'reject' ? '#dc2626' : ann.state === 'unsure' ? '#9333ea' : '#facc15';
    traceCtx.beginPath(); traceCtx.arc(traceXForFrame(ev.frame, w, pad), pad + 8, ev.frame === selectedEventFrame ? 5 : 3, 0, Math.PI*2); traceCtx.fill();
  }
  traceCanvas.setAttribute('aria-label', `ROI ${roi.id} trace with ${eventsForRoi(roi).length} called events. Current frame ${currentFrame}.`);
  traceCtx.fillStyle = '#0f172a'; traceCtx.font = '13px Arial';
  traceCtx.fillText(`ROI ${roi.id} | area ${roi.area} | noise sigma ${model.sigma.toFixed(5)} | events ${eventsForRoi(roi).length}`, pad, 18);
}

function roiCropBounds(roi, pad=18){
  if(!roi) return null;
  const bbox = roi.bbox || [
    Math.floor(roi.centroidX - 8), Math.floor(roi.centroidY - 8),
    Math.ceil(roi.centroidX + 8), Math.ceil(roi.centroidY + 8)
  ];
  const x0 = Math.max(0, Math.floor(bbox[0] - pad));
  const y0 = Math.max(0, Math.floor(bbox[1] - pad));
  const x1 = Math.min(data.video.width - 1, Math.ceil(bbox[2] + pad));
  const y1 = Math.min(data.video.height - 1, Math.ceil(bbox[3] + pad));
  return {x0, y0, x1, y1, w: Math.max(1, x1 - x0 + 1), h: Math.max(1, y1 - y0 + 1)};
}

function drawCrop(){
  if(!cropCanvas || !cropCtx) return;
  const roi = selectedRoi();
  cropCtx.clearRect(0,0,cropCanvas.width,cropCanvas.height);
  cropCtx.fillStyle = '#08111f';
  cropCtx.fillRect(0,0,cropCanvas.width,cropCanvas.height);
  if(!roi || !img.complete || !img.naturalWidth) return;
  const b = roiCropBounds(roi);
  const scale = Math.min(cropCanvas.width / b.w, cropCanvas.height / b.h);
  const dw = b.w * scale, dh = b.h * scale;
  const ox = (cropCanvas.width - dw) / 2, oy = (cropCanvas.height - dh) / 2;
  cropCtx.imageSmoothingEnabled = false;
  cropCtx.drawImage(img, b.x0, b.y0, b.w, b.h, ox, oy, dw, dh);
  const cropFill = selectedOverlayFillAlpha(selectedEventFrame && eventNearFrame(roi, currentFrame));
  if(cropFill > 0.005) {
    cropCtx.fillStyle = `rgba(56, 189, 248, ${cropFill.toFixed(3)})`;
    for(const p of roi.points || []){
      const x = ox + (p[0] - b.x0) * scale;
      const y = oy + (p[1] - b.y0) * scale;
      cropCtx.fillRect(x, y, Math.max(1, scale), Math.max(1, scale));
    }
  }
  cropCtx.strokeStyle = selectedEventFrame && eventNearFrame(roi, currentFrame) ? '#facc15' : '#ffffff';
  cropCtx.lineWidth = Math.max(2, Number(setting('selectedOutlineWidth')) || 2.5);
  cropCtx.beginPath();
  cropCtx.arc(ox + (roi.centroidX - b.x0) * scale, oy + (roi.centroidY - b.y0) * scale, Math.max(5, Math.sqrt(roi.area / Math.PI) * scale), 0, Math.PI * 2);
  cropCtx.stroke();
  cropCanvas.setAttribute('aria-label', `Crop around ROI ${roi.id}, area ${roi.area} pixels, centered at x ${Number(roi.centroidX).toFixed(1)}, y ${Number(roi.centroidY).toFixed(1)}.`);
}

function renderRoiContext(){
  const roi = selectedRoi();
  const card = document.getElementById('roiEvidenceCard');
  const strip = document.getElementById('eventFilmstrip');
  drawCrop();
  if(!roi){
    if(card) card.innerHTML = '';
    if(strip) strip.innerHTML = '';
    return;
  }
  const events = eventsForRoi(roi);
  const diameterPx = 2 * Math.sqrt(roi.area / Math.PI);
  const pixelSize = Number(data.dataset?.pixel_size_microns);
  const diameterUm = Number.isFinite(pixelSize) ? `${(diameterPx * pixelSize).toFixed(1)} um` : 'n/a';
  const warnings = artifactReasonsForRoi(roi);
  if(scoreValue(roi, 'artifactScore') >= 0.45 && !warnings.includes('artifact-risk')) warnings.push('artifact-risk');
  if(scoreValue(roi, 'backgroundCorrelation') >= 0.55) warnings.push('background-correlated');
  if(scoreValue(roi, 'localCorrelationMean') > 0 && scoreValue(roi, 'localCorrelationMean') < 0.40) warnings.push('low local correlation');
  if(scoreValue(roi, 'eventSupport') > 0 && scoreValue(roi, 'eventSupport') < 0.35) warnings.push('weak event support');
  const warningHtml = warnings.length ? `<tr><td>warnings</td><td>${warnings.map(w => `<span class="riskPill">${w}</span>`).join(' ')}</td></tr>` : '';
  if(card) card.innerHTML = `
    <table class="smallTable">
      <tr><th>Field</th><th>Value</th></tr>
      <tr><td>ROI</td><td>${roi.id}</td></tr>
      <tr><td>priority score</td><td>${fmt(scoreValue(roi, 'priorityScore', null), 3)}</td></tr>
      <tr><td>area</td><td>${roi.area} px</td></tr>
      <tr><td>equiv. diameter</td><td>${diameterPx.toFixed(1)} px / ${diameterUm}</td></tr>
      <tr><td>kind</td><td>${escapeHtml(roi.roi_kind || 'source')}</td></tr>
      <tr><td>peak score</td><td>${fmt(scoreValue(roi, 'peakScore', null), 2)}</td></tr>
      <tr><td>noise sigma</td><td>${fmt(scoreValue(roi, 'noiseSigma', null), 5)}</td></tr>
      <tr><td>trace SNR</td><td>${fmt(scoreValue(roi, 'traceSnr', null), 2)}</td></tr>
      <tr><td>local correlation</td><td>${fmt(scoreValue(roi, 'localCorrelationMean', null), 3)}</td></tr>
      <tr><td>background corr.</td><td>${fmt(scoreValue(roi, 'backgroundCorrelation', null), 3)}</td></tr>
      <tr><td>event support</td><td>${fmt(scoreValue(roi, 'eventSupport', null), 3)}</td></tr>
      <tr><td>artifact risk</td><td>${fmt(scoreValue(roi, 'artifactScore', null), 3)}</td></tr>
      <tr><td>events</td><td>${events.length}</td></tr>
      ${warningHtml}
    </table>`;
  if(!strip) return;
  const center = selectedEventFrame || events[0]?.frame || currentFrame;
  const b = roiCropBounds(roi, 24);
  const thumb = 52;
  const scale = thumb / Math.max(b.w, b.h);
  strip.innerHTML = '';
  for(let frame = Math.max(1, center - 5); frame <= Math.min(data.video.frames, center + 10); frame++){
    const cell = document.createElement('button');
    cell.type = 'button';
    cell.className = 'filmFrame' + (frame === currentFrame ? ' active' : '');
    cell.setAttribute('aria-label', `Show frame ${frame} near ROI ${roi.id}`);
    if(frame === currentFrame) cell.setAttribute('aria-current', 'true');
    cell.style.backgroundImage = `url("${framePath(frame)}")`;
    cell.style.backgroundSize = `${data.video.width * scale}px ${data.video.height * scale}px`;
    cell.style.backgroundPosition = `${-b.x0 * scale}px ${-b.y0 * scale}px`;
    cell.innerHTML = `<span>${frame}</span>`;
    cell.onclick = () => setFrame(frame);
    strip.appendChild(cell);
  }
}

function setFrame(frame){
  currentFrame = Math.max(1, Math.min(data.video.frames, frame));
  ensureTraceFrameVisible(currentFrame);
  slider.value = currentFrame;
  frameLabel.textContent = currentFrame;
  img.src = framePath(currentFrame);
  statusEl.textContent = `Frame ${currentFrame} / ${data.video.frames}`;
  const roi = selectedRoi();
  selectionText.textContent = roi ? `ROI ${roi.id}${selectedEventFrame ? `, event f${selectedEventFrame}` : ''}` : '';
  drawTrace();
  drawEventTimeline();
  renderRoiContext();
  updateQcFrameView();
  renderReviewComparisonViewer();
}

function quickJump(value){
  const raw = String(value || '').trim();
  if(!raw) return;
  const frameMatch = raw.match(/^f(?:rame)?\s*:?\s*(\d+)$/i);
  const roiMatch = raw.match(/^r(?:oi)?\s*:?\s*(.+)$/i);
  if(frameMatch) {
    setFrame(Number(frameMatch[1]));
    setSaveState(`jumped to frame ${currentFrame}`, 'ok');
    return;
  }
  const roiText = roiMatch ? roiMatch[1].trim() : raw;
  const roi = reviewRois().find(item => String(item.id).toLowerCase() === roiText.toLowerCase());
  if(roi) {
    selectRoi(roi.id);
    setSaveState(`jumped to ROI ${roi.id}`, 'ok');
    return;
  }
  if(/^\d+$/.test(raw)) {
    setFrame(Number(raw));
    setSaveState(`jumped to frame ${currentFrame}`, 'ok');
    return;
  }
  setSaveState(`no ROI or frame matched "${raw}"`, 'bad');
}

function selectRoi(id, additive=false){
  selectedId = id;
  if(additive) {
    const key = String(id);
    if(selectedRoiIds.has(key)) selectedRoiIds.delete(key);
    else selectedRoiIds.add(key);
    if(!selectedRoiIds.size) selectedRoiIds.add(key);
  } else {
    selectedRoiIds = new Set([String(id)]);
  }
  const roi = selectedRoi();
  selectedEventFrame = visibleEventsForRoi(roi)[0]?.frame || eventsForRoi(roi)[0]?.frame || null;
  roiNotes.value = roiAnn(id).notes || '';
  eventNotes.value = selectedEventFrame ? eventAnn(id, selectedEventFrame).notes || '' : '';
  setFrame(selectedEventFrame || currentFrame);
  renderAll();
}

function selectSuggestion(id){
  selectedSuggestionId = id;
  const s = selectedSuggestion();
  document.getElementById('suggestionNotes').value = s ? suggestionAnn(s.id).notes || '' : '';
  document.getElementById('artifactClass').value = s ? suggestionAnn(s.id).artifact_class || suggestionAnn(s.id).artifactClass || '' : '';
  if(s) {
    selectedEventFrame = null;
    currentFrame = Math.max(1, Math.min(data.video.frames, currentFrame));
    selectionText.textContent = `Suggestion ${s.id}`;
  }
  renderAll();
}

function renderRoiList(){
  const root = document.getElementById('roiList');
  root.innerHTML = '';
  const rows = visibleRois();
  document.getElementById('visibleCount').textContent = rows.length;
  const status = document.getElementById('queueStatusText');
  if(status) {
    const idx = rows.findIndex(r => String(r.id) === String(selectedId));
    status.textContent = rows.length ? `${idx >= 0 ? idx + 1 : 0} of ${rows.length} queued` : '0 queued';
  }
  const multiCount = document.getElementById('multiSelectCount');
  if(multiCount) multiCount.textContent = String(selectedRoiIds.size);
  for(const roi of rows){
    const ann = roiAnn(roi.id);
    const row = document.createElement('div');
    row.className = 'roiRow' + (roi.id === selectedId ? ' sel' : '') + (selectedRoiIds.has(String(roi.id)) ? ' multiSel' : '') + (ann.deleted ? ' deleted' : '');
    const state = ann.deleted ? 'deleted' : ann.state || 'new';
    const triage = roiTriageCategory(roi).replace(/_/g, ' ');
    row.innerHTML = `<b>#${roi.id}</b><span>${eventsForRoi(roi).length} events, area ${roi.area}, priority ${fmt(scoreValue(roi, 'priorityScore', null), 2)} <i class="triageChip">${triage}</i></span><span class="badge ${ann.state || ''}">${state}</span>`;
    row.onclick = e => selectRoi(roi.id, e.shiftKey || e.ctrlKey || e.metaKey);
    root.appendChild(row);
  }
}

function renderEventList(){
  const roi = selectedRoi();
  const root = document.getElementById('eventList');
  root.innerHTML = '';
  const status = document.getElementById('eventQueueStatusText');
  if(!roi) {
    if(status) status.textContent = '0 events';
    return;
  }
  const rows = visibleEventsForRoi(roi);
  const globalRows = eventQueueItems();
  if(status) {
    const idx = globalRows.findIndex(item => String(item.roi.id) === String(roi.id) && item.ev.frame === selectedEventFrame);
    status.textContent = globalRows.length ? `${idx >= 0 ? idx + 1 : 0} of ${globalRows.length} event queue, ${rows.length} in ROI` : '0 events';
  }
  for(const ev of rows){
    const ann = eventAnn(roi.id, ev.frame);
    const row = document.createElement('div');
    row.className = 'eventRow' + (ev.frame === selectedEventFrame ? ' sel' : '');
    const reviewer = String(ann.reviewer_id || '').trim();
    const reviewerText = reviewer ? `, ${reviewer}` : (eventReviewed(roi.id, ev.frame) ? ', no reviewer' : '');
    row.innerHTML = `<b>f${ev.frame}</b><span>z ${ev.z.toFixed(2)}, amp ${ev.amplitude.toFixed(4)}${reviewerText}</span><span class="badge ${ann.state || ''}">${ann.state || 'new'}</span>`;
    row.onclick = () => { selectedEventFrame = ev.frame; eventNotes.value = eventAnn(roi.id, ev.frame).notes || ''; setFrame(ev.frame); renderAll(); };
    root.appendChild(row);
  }
  if(!rows.length) root.innerHTML = '<p class="hint">No events match the current event queue for this ROI.</p>';
}

function renderSuggestionList(){
  const root = document.getElementById('suggestionList');
  if(!root) return;
  root.innerHTML = '';
  const rows = visibleSuggestions();
  document.getElementById('suggestionVisibleCount').textContent = rows.length;
  const status = document.getElementById('suggestionQueueStatusText');
  if(status) {
    const idx = rows.findIndex(s => String(s.id) === String(selectedSuggestionId));
    status.textContent = rows.length ? `${idx >= 0 ? idx + 1 : 0} of ${rows.length} suggestions` : '0 suggestions';
  }
  for(const s of rows){
    const ann = suggestionAnn(s.id);
    const row = document.createElement('div');
    row.className = 'suggestionRow' + (s.id === selectedSuggestionId ? ' sel' : '');
    const state = annotations.promotedRois[s.id] ? 'promoted' : ann.state || 'new';
    const cue = s.artifactCue && s.artifactCue !== 'none' ? `, ${s.artifactCue}` : '';
    const reviewer = String(ann.reviewer_id || '').trim();
    const reviewerText = reviewer ? `, ${reviewer}` : ((ann.state || annotations.promotedRois[s.id]) ? ', no reviewer' : '');
    row.innerHTML = `<b>${s.id}</b><span>priority ${fmt(scoreValue(s, 'priorityScore', s.discoveryScore), 3)}, area ${s.area}${cue}${reviewerText}</span><span class="badge ${ann.state || ''}">${state}</span>`;
    row.onclick = () => selectSuggestion(s.id);
    root.appendChild(row);
  }
  if(!rows.length) root.innerHTML = '<p class="hint">No discovery suggestions match the current filter.</p>';
}

function updateCounts(){
  const allEvents = reviewRois().reduce((sum, r) => sum + eventsForRoi(r).length, 0);
  let acc = 0, rej = 0, unsure = 0, eventAccepted = 0;
  let promoted = 0, missed = 0, artifacts = 0;
  for(const r of reviewRois()){
    const st = roiAnn(r.id).state;
    if(st === 'accept') acc++;
    if(st === 'reject') rej++;
    if(st === 'unsure') unsure++;
    for(const ev of eventsForRoi(r)) if(eventAnn(r.id, ev.frame).state === 'accept') eventAccepted++;
  }
  for(const s of data.discovery?.suggestions || []){
    const ann = suggestionAnn(s.id);
    if(annotations.promotedRois[s.id] || ann.state === 'promoted') promoted++;
    if(ann.state === 'missed') missed++;
    if(ann.state === 'artifact') artifacts++;
  }
  document.getElementById('roiCount').textContent = reviewRois().length;
  document.getElementById('eventCount').textContent = allEvents;
  document.getElementById('acceptedCount').textContent = acc;
  document.getElementById('rejectedCount').textContent = rej;
  document.getElementById('unsureCount').textContent = unsure;
  document.getElementById('eventAcceptedCount').textContent = eventAccepted;
  document.getElementById('suggestionCount').textContent = data.discovery?.suggestions?.length || 0;
  document.getElementById('promotedCount').textContent = promoted;
  document.getElementById('missedCount').textContent = missed;
  document.getElementById('artifactCount').textContent = artifacts;
}

function setRoiState(state){
  const roi = selectedRoi(); if(!roi) return;
  pushAnnotationUndo(`ROI ${roi.id} label`, [
    annotationSnapshot('rois', roi.id),
    annotations.virtualRois[roi.id] ? annotationSnapshot('virtualRois', roi.id) : null
  ]);
  const cellState = state === 'accept' ? 'accepted' : state === 'reject' ? 'rejected' : state === 'unsure' ? 'unsure' : '';
  annotations.rois[roi.id] = stampAnnotation(Object.assign(roiAnn(roi.id), {state, cell_state: cellState}));
  if(annotations.virtualRois[roi.id]) stampAnnotation(Object.assign(annotations.virtualRois[roi.id], {cell_state: cellState}));
  recordAction(`roi_${state || 'clear'}`);
  queueSave();
  renderAll();
}

function setRoiStateAndNext(state){
  setRoiState(state);
  nextRoi(1);
}

function markRoiStrongAndNext(){
  const roi = selectedRoi(); if(!roi) return;
  pushAnnotationUndo(`ROI ${roi.id} strong neuron preset`, [
    annotationSnapshot('rois', roi.id),
    annotations.virtualRois[roi.id] ? annotationSnapshot('virtualRois', roi.id) : null
  ]);
  const fields = {
    state: 'accept',
    cell_state: 'accepted',
    trace_quality: 'good',
    control_ready: 'yes',
    artifact_class: 'none',
    confidence: 'high',
    reason_tags: [...new Set([...(roiAnn(roi.id).reason_tags || []), 'event_supported', 'clear_trace'])]
  };
  annotations.rois[roi.id] = stampAnnotation(Object.assign(roiAnn(roi.id), fields));
  if(annotations.virtualRois[roi.id]) stampAnnotation(Object.assign(annotations.virtualRois[roi.id], fields));
  recordAction('roi_strong_neuron_preset');
  queueSave();
  renderAll();
  nextRoi(1);
}

function markRoiArtifactAndNext(){
  const roi = selectedRoi(); if(!roi) return;
  pushAnnotationUndo(`ROI ${roi.id} artifact preset`, [
    annotationSnapshot('rois', roi.id),
    annotations.virtualRois[roi.id] ? annotationSnapshot('virtualRois', roi.id) : null
  ]);
  const fields = {
    state: 'reject',
    cell_state: 'rejected',
    trace_quality: roiAnn(roi.id).trace_quality || 'unusable',
    control_ready: 'no',
    artifact_class: roiAnn(roi.id).artifact_class && roiAnn(roi.id).artifact_class !== 'none' ? roiAnn(roi.id).artifact_class : 'uncertain_artifact',
    confidence: roiAnn(roi.id).confidence || 'medium',
    reason_tags: [...new Set([...(roiAnn(roi.id).reason_tags || []), 'artifact_risk'])]
  };
  annotations.rois[roi.id] = stampAnnotation(Object.assign(roiAnn(roi.id), fields));
  if(annotations.virtualRois[roi.id]) stampAnnotation(Object.assign(annotations.virtualRois[roi.id], fields));
  recordAction('roi_artifact_preset');
  queueSave();
  renderAll();
  nextRoi(1);
}

function applyToSelectedRois(fields, actionName){
  const ids = selectedRoiIdList();
  if(!ids.length) return;
  const snapshots = [];
  for(const id of ids){
    snapshots.push(annotationSnapshot('rois', id));
    if(annotations.virtualRois[id]) snapshots.push(annotationSnapshot('virtualRois', id));
  }
  pushAnnotationUndo(`${ids.length} selected ROI edit`, snapshots);
  for(const id of ids){
    annotations.rois[id] = stampAnnotation(Object.assign(roiAnn(id), fields));
    if(annotations.virtualRois[id]) stampAnnotation(Object.assign(annotations.virtualRois[id], fields));
  }
  recordAction(actionName || 'roi_bulk_edit');
  queueSave();
  renderAll();
}

function setSelectedRoisState(state){
  const cellState = state === 'accept' ? 'accepted' : state === 'reject' ? 'rejected' : state === 'unsure' ? 'unsure' : '';
  if(!cellState) return;
  const actionNames = {accept: 'roi_bulk_accept', reject: 'roi_bulk_reject', unsure: 'roi_bulk_unsure'};
  applyToSelectedRois({state, cell_state: cellState}, actionNames[state] || 'roi_bulk_label');
}

function assignSelectedIdentity(){
  const ids = selectedRoiIdList();
  if(ids.length < 2) return;
  const value = document.getElementById('bulkIdentityGroup').value.trim() || `group_${ids.join('_')}`;
  applyToSelectedRois({identity_group: value, needs_action: 'merge_needed'}, 'roi_bulk_identity_group');
}

function markSelectedAction(){
  const value = document.getElementById('bulkNeedsAction').value;
  if(!value) return;
  applyToSelectedRois({needs_action: value}, 'roi_bulk_needs_action');
}

function splitMergeDecisionId(prefix, ids){
  return `SM_${prefix}_${ids.map(v => String(v).replace(/[^A-Za-z0-9_-]/g, '')).join('_')}`;
}

function recordSplitMergeDecision(decision, actionName){
  annotations.splitMergeDecisions = annotations.splitMergeDecisions || {};
  const item = migrateSplitMergeDecision(decision);
  item.id = item.id || splitMergeDecisionId(item.decision_type || 'edit', item.source_roi_ids);
  item.createdAt = item.createdAt || new Date().toISOString();
  annotations.splitMergeDecisions[item.id] = stampAnnotation(item);
  recordAction(actionName || `roi_${item.decision_type || 'split_merge'}_decision`);
}

function createVirtualMerge(){
  const rois = selectedRois();
  if(rois.length < 2) return;
  const ids = rois.map(r => r.id);
  const pixels = new Map();
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity, sumX = 0, sumY = 0;
  for(const roi of rois){
    for(const p of roi.points || []){
      const key = `${p[0]},${p[1]}`;
      if(!pixels.has(key)) {
        pixels.set(key, p);
        sumX += p[0];
        sumY += p[1];
        minX = Math.min(minX, p[0]);
        minY = Math.min(minY, p[1]);
        maxX = Math.max(maxX, p[0]);
        maxY = Math.max(maxY, p[1]);
      }
    }
  }
  const points = [...pixels.values()];
  const id = `VM_${ids.join('_')}`;
  annotations.virtualRois[id] = stampAnnotation({
    id,
    roi_kind: 'virtual_merge',
    source_roi_ids: ids,
    identity_group: document.getElementById('bulkIdentityGroup').value.trim() || `group_${ids.join('_')}`,
    centroidX: points.length ? Number((sumX / points.length).toFixed(1)) : null,
    centroidY: points.length ? Number((sumY / points.length).toFixed(1)) : null,
    area: points.length,
    bbox: [minX, minY, maxX, maxY],
    points,
    cell_state: 'unsure',
    trace_quality: '',
    control_ready: '',
    artifact_class: '',
    reason_tags: ['merge'],
    confidence: '',
    notes: ''
  });
  recordSplitMergeDecision({
    id: splitMergeDecisionId('merge', ids),
    decision_type: 'merge',
    decision_state: 'accepted',
    source_roi_ids: ids,
    virtual_roi_id: id,
    identity_group: annotations.virtualRois[id].identity_group,
    needs_action: 'merge_needed',
    reason_tags: ['merge']
  }, 'roi_virtual_merge_decision');
  applyToSelectedRois({identity_group: annotations.virtualRois[id].identity_group, needs_action: 'merge_needed'}, 'roi_virtual_merge');
}

function createVisualSplitDecision(){
  const roi = selectedRoi();
  if(!roi) return;
  const targetText = prompt('Target ROI IDs after split, comma-separated', '');
  if(targetText === null) return;
  const targets = normalizeIdList(targetText);
  const id = splitMergeDecisionId('split', [roi.id].concat(targets.length ? targets : [Date.now().toString(36)]));
  recordSplitMergeDecision({
    id,
    decision_type: 'split',
    decision_state: 'accepted',
    source_roi_ids: [roi.id],
    target_roi_ids: targets,
    needs_action: 'split_needed',
    reason_tags: ['split'],
    notes: targets.length ? `Split into ${targets.join(',')}` : 'Split requested from visual review'
  }, 'roi_visual_split_decision');
  annotations.rois[roi.id] = Object.assign(roiAnn(roi.id), {needs_action: 'split_needed'});
  queueSave();
  renderAll();
}

function clearMultiSelection(){
  if(selectedId) selectedRoiIds = new Set([String(selectedId)]);
  renderAll();
}

function toggleDeleted(){
  const roi = selectedRoi(); if(!roi) return;
  pushAnnotationUndo(`ROI ${roi.id} visibility`, [
    annotationSnapshot('rois', roi.id),
    annotations.virtualRois[roi.id] ? annotationSnapshot('virtualRois', roi.id) : null
  ]);
  const ann = stampAnnotation(Object.assign(roiAnn(roi.id), {deleted: !roiAnn(roi.id).deleted}));
  annotations.rois[roi.id] = ann;
  if(annotations.virtualRois[roi.id]) stampAnnotation(Object.assign(annotations.virtualRois[roi.id], {deleted: ann.deleted}));
  recordAction(ann.deleted ? 'roi_hide' : 'roi_restore');
  queueSave();
  renderAll();
}
function setEventState(state){
  const roi = selectedRoi(); if(!roi || !selectedEventFrame) return;
  pushAnnotationUndo(`ROI ${roi.id} frame ${selectedEventFrame} event label`, [
    annotationSnapshot('events', eventKey(roi.id, selectedEventFrame))
  ]);
  const eventState = state === 'accept' ? 'accepted' : state === 'reject' ? 'rejected' : state === 'unsure' ? 'unsure' : '';
  annotations.events[eventKey(roi.id, selectedEventFrame)] = stampAnnotation(Object.assign(eventAnn(roi.id, selectedEventFrame), {state, event_state: eventState}));
  recordAction(`event_${state || 'clear'}`);
  queueSave();
  renderAll();
}

function setEventStateAndNext(state){
  const roi = selectedRoi();
  const currentKey = roi && selectedEventFrame ? eventKey(roi.id, selectedEventFrame) : '';
  const rows = eventQueueItems();
  setEventState(state);
  advanceEventFromRows(rows, currentKey, 1);
}

function markEventArtifactAndNext(){
  const roi = selectedRoi(); if(!roi || !selectedEventFrame) return;
  const currentKey = eventKey(roi.id, selectedEventFrame);
  const rows = eventQueueItems();
  pushAnnotationUndo(`ROI ${roi.id} frame ${selectedEventFrame} event artifact preset`, [
    annotationSnapshot('events', currentKey)
  ]);
  annotations.events[currentKey] = stampAnnotation(Object.assign(eventAnn(roi.id, selectedEventFrame), {
    state: 'reject',
    event_state: 'rejected',
    event_type: 'artifact',
    timing_quality: 'ambiguous',
    confidence: 'medium',
    reason_tags: [...new Set([...(eventAnn(roi.id, selectedEventFrame).reason_tags || []), 'artifact_risk'])]
  }));
  recordAction('event_artifact_preset');
  queueSave();
  renderAll();
  advanceEventFromRows(rows, currentKey, 1);
}

function advanceEventFromRows(rows, currentKey, delta=1){
  const queueRows = rows || [];
  if(!queueRows.length) {
    renderAll();
    return;
  }
  const idx = queueRows.findIndex(item => item.key === currentKey);
  const base = idx >= 0 ? idx : delta > 0 ? -1 : 0;
  for(let offset = 1; offset <= queueRows.length; offset++){
    const item = queueRows[(base + delta * offset + queueRows.length) % queueRows.length];
    if(item && item.key !== currentKey) {
      selectEventQueueItem(item);
      return;
    }
  }
  renderAll();
}

function setSuggestionState(state){
  const s = selectedSuggestion(); if(!s) return;
  pushAnnotationUndo(`suggestion ${s.id} label`, [annotationSnapshot('suggestions', s.id)]);
  annotations.suggestions[s.id] = stampAnnotation(Object.assign(suggestionAnn(s.id), {state}));
  recordAction(`suggestion_${state || 'clear'}`);
  queueSave();
  renderAll();
}
function nextSuggestion(delta=1){
  const rows = visibleSuggestions();
  if(!rows.length) {
    setSaveState('no suggestions match the current filter', 'bad');
    renderSuggestionList();
    return;
  }
  const idx = rows.findIndex(s => String(s.id) === String(selectedSuggestionId));
  const base = idx >= 0 ? idx : delta > 0 ? -1 : 0;
  selectSuggestion(rows[(base + delta + rows.length) % rows.length].id);
}
function advanceSuggestionFromRows(rows, currentId, delta=1){
  const candidates = (rows || []).filter(s => String(s.id) !== String(currentId));
  if(candidates.length) {
    const idx = rows.findIndex(s => String(s.id) === String(currentId));
    const next = rows[(idx + delta + rows.length) % rows.length];
    if(next && String(next.id) !== String(currentId)) {
      selectSuggestion(next.id);
      return;
    }
    selectSuggestion(candidates[0].id);
    return;
  }
  renderAll();
}
function setSuggestionStateAndNext(state){
  const rows = visibleSuggestions();
  const currentId = selectedSuggestion()?.id;
  setSuggestionState(state);
  advanceSuggestionFromRows(rows, currentId, 1);
}
function promoteSuggestionAndNext(){
  const rows = visibleSuggestions();
  const currentId = selectedSuggestion()?.id;
  promoteSuggestion();
  advanceSuggestionFromRows(rows, currentId, 1);
}
function promoteSuggestion(){
  const s = selectedSuggestion(); if(!s) return;
  pushAnnotationUndo(`suggestion ${s.id} promotion`, [
    annotationSnapshot('suggestions', s.id),
    annotationSnapshot('promotedRois', s.id)
  ]);
  annotations.suggestions[s.id] = stampAnnotation(Object.assign(suggestionAnn(s.id), {state:'promoted'}));
  annotations.promotedRois[s.id] = {
    sourceSuggestion: s.id,
    provenance: s.provenance || 'discovery',
    centroidX: s.centroidX,
    centroidY: s.centroidY,
    area: s.area,
    bbox: s.bbox,
    points: s.points || [],
    promotedAt: new Date().toISOString(),
    reviewer_id: currentReviewerId()
  };
  recordAction('suggestion_promote');
  queueSave();
  renderAll();
}

function renderButtons(){
  const roi = selectedRoi();
  const ann = roi ? roiAnn(roi.id) : {};
  for (const [id, state] of [['acceptBtn','accept'],['rejectBtn','reject'],['unsureBtn','unsure']]) {
    document.getElementById(id).classList.toggle('active', ann.state === state);
  }
  document.getElementById('deleteBtn').textContent = ann.deleted ? 'Restore ROI' : 'Hide ROI';
  const eann = roi && selectedEventFrame ? eventAnn(roi.id, selectedEventFrame) : {};
  for (const [id, state] of [['eventAcceptBtn','accept'],['eventRejectBtn','reject'],['eventUnsureBtn','unsure']]) {
    document.getElementById(id).classList.toggle('active', eann.state === state);
  }
  for (const [id, field] of [['traceQuality','trace_quality'],['controlReady','control_ready'],['roiArtifactClass','artifact_class'],['needsAction','needs_action']]) {
    const el = document.getElementById(id);
    if(el) el.value = ann[field] || '';
  }
  const identity = document.getElementById('identityGroup');
  if(identity) identity.value = ann.identity_group || '';
  const roiConfidence = document.getElementById('roiConfidence');
  if(roiConfidence) roiConfidence.value = ann.confidence || '';
  const roiReasonTags = document.getElementById('roiReasonTags');
  if(roiReasonTags) roiReasonTags.value = (ann.reason_tags || []).join(',');
  for (const [id, field] of [['eventType','event_type'],['timingQuality','timing_quality']]) {
    const el = document.getElementById(id);
    if(el) el.value = eann[field] || '';
  }
  const eventConfidence = document.getElementById('eventConfidence');
  if(eventConfidence) eventConfidence.value = eann.confidence || '';
  const eventReasonTags = document.getElementById('eventReasonTags');
  if(eventReasonTags) eventReasonTags.value = (eann.reason_tags || []).join(',');
  const sann = selectedSuggestion() ? suggestionAnn(selectedSuggestion().id) : {};
  for (const [id, state] of [['suggestionMissedBtn','missed'],['suggestionArtifactBtn','artifact'],['suggestionUnsureBtn','unsure']]) {
    document.getElementById(id).classList.toggle('active', sann.state === state);
  }
  const suggestionConfidence = document.getElementById('suggestionConfidence');
  if(suggestionConfidence) suggestionConfidence.value = sann.confidence || '';
  const suggestionReasonTags = document.getElementById('suggestionReasonTags');
  if(suggestionReasonTags) suggestionReasonTags.value = (sann.reason_tags || []).join(',');
}

function renderParams(){
  const rows = Object.entries(data.parameters || {}).map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
  document.getElementById('paramTable').innerHTML = '<tr><th>Parameter</th><th>Value</th></tr>' + rows;
}

function renderAll(){
  updateCounts();
  updateUndoButton();
  renderButtons();
  renderReviewSessionPanel();
  renderGuidedPanel();
  renderFocusSummary();
  renderSnapshotControls();
  renderSuggestionContext();
  renderRoiList();
  renderEventList();
  renderSuggestionList();
  drawOverlay();
  drawTrace();
  drawEventTimeline();
  renderRoiContext();
}

function progressPercent(done, total){
  if(!Number.isFinite(Number(total)) || Number(total) <= 0) return 0;
  return Math.max(0, Math.min(100, Math.round(100 * Number(done || 0) / Number(total))));
}

function queuePosition(rows, predicate){
  const idx = rows.findIndex(predicate);
  return {index: idx, current: idx >= 0 ? idx + 1 : 0, total: rows.length};
}

function sessionChecklistState(){
  const summary = annotationSummary();
  const provenance = reviewerProvenanceAudit();
  const run = activeRun();
  const roiRows = visibleRois();
  const eventRows = eventQueueItems();
  const suggestionRows = visibleSuggestions();
  const eventKeyCurrent = selectedId && selectedEventFrame !== null ? eventKey(selectedId, selectedEventFrame) : '';
  const batch = nextAnnotationBatch();
  const roiTarget = summary.review_progress.tuning_ready_targets.reviewed_rois;
  const eventTarget = summary.review_progress.tuning_ready_targets.reviewed_events;
  const acceptedControlReady = summary.control_ready.yes + summary.control_ready.maybe;
  const reviewedTotal = summary.review_progress.reviewed_rois + summary.review_progress.reviewed_events + summary.review_progress.reviewed_suggestions;
  return {
    reviewer_id: currentReviewerId(),
    save: saveStatus,
    run,
    run_label: runLabel(run) || activeRunId(),
    queues: {
      roi: Object.assign({name: setting('queue') || 'all'}, queuePosition(roiRows, roi => String(roi.id) === String(selectedId))),
      event: Object.assign({name: setting('eventQueue') || 'all'}, queuePosition(eventRows, item => item.key === eventKeyCurrent)),
      suggestion: Object.assign({name: setting('discoveryQueue') || 'all'}, queuePosition(suggestionRows, item => String(item.id) === String(selectedSuggestionId)))
    },
    remaining: {
      rois: summary.roi_states.unlabeled,
      events: summary.event_states.unlabeled,
      suggestions: summary.suggestion_states.unlabeled
    },
    progress: {
      reviewed_rois: summary.review_progress.reviewed_rois,
      reviewed_events: summary.review_progress.reviewed_events,
      reviewed_suggestions: summary.review_progress.reviewed_suggestions,
      roi_target: roiTarget,
      event_target: eventTarget,
      roi_target_remaining: Math.max(0, roiTarget - summary.review_progress.reviewed_rois),
      event_target_remaining: Math.max(0, eventTarget - summary.review_progress.reviewed_events),
      tuning_ready: summary.review_progress.tuning_ready,
      next_batch: {
        rois: batch.rois.length,
        events: batch.events.length,
        suggestions: batch.suggestions.length
      }
    },
    provenance,
    export_ready: {
      accepted_rois: summary.roi_states.accepted,
      accepted_events: summary.event_states.accepted,
      control_ready_rois: acceptedControlReady,
      reviewed_total: reviewedTotal,
      ready: summary.roi_states.accepted > 0 && summary.event_states.accepted > 0 && provenance.totals.missing === 0
    },
    summary
  };
}

function sessionStatusChip(label, value, kind){
  return `<div class="sessionStatus ${kind || ''}"><span>${escapeHtml(label)}</span><b>${escapeHtml(value)}</b></div>`;
}

function sessionProgressBar(label, done, total, detail=''){
  const pct = progressPercent(done, total);
  return `
    <div class="sessionProgress">
      <div class="sessionProgressLabel"><span>${escapeHtml(label)}</span><b>${escapeHtml(done)} / ${escapeHtml(total)}</b></div>
      <div class="sessionProgressBar"><div class="sessionProgressFill" style="width:${pct}%"></div></div>
      ${detail ? `<p class="hint">${escapeHtml(detail)}</p>` : ''}
    </div>`;
}

function reviewSessionHandoff(){
  const state = sessionChecklistState();
  return {
    schema_version: 1,
    dataset_id: datasetId,
    generatedAt: new Date().toISOString(),
    active_run_id: activeRunId(),
    active_run_label: state.run_label,
    reviewer_id: state.reviewer_id,
    save_status: state.save,
    queues: state.queues,
    remaining: state.remaining,
    progress: state.progress,
    export_ready: state.export_ready,
    reviewer_provenance: state.provenance.totals,
    reviewer_missing_by_group: Object.fromEntries(Object.entries(state.provenance.by_group || {}).map(([group, item]) => [group, item.missing || 0])),
    recommended_next_batch: nextAnnotationBatch()
  };
}

function reviewSessionHandoffMarkdown(){
  const h = reviewSessionHandoff();
  const missing = h.reviewer_provenance.missing || 0;
  const lines = [
    `# Review Session Handoff - ${datasetId}`,
    '',
    `Generated: ${h.generatedAt}`,
    `Active run: ${h.active_run_label} (${h.active_run_id})`,
    `Reviewer: ${h.reviewer_id || 'not set'}`,
    `Save mode: ${h.save_status.serverBacked ? 'local server autosave' : 'static/browser local'} - ${h.save_status.text}`,
    '',
    '## Current Queues',
    `- ROI queue: ${h.queues.roi.name}, ${h.queues.roi.current || 0}/${h.queues.roi.total}`,
    `- Event queue: ${h.queues.event.name}, ${h.queues.event.current || 0}/${h.queues.event.total}`,
    `- Suggestion queue: ${h.queues.suggestion.name}, ${h.queues.suggestion.current || 0}/${h.queues.suggestion.total}`,
    '',
    '## Review Progress',
    `- Reviewed ROIs: ${h.progress.reviewed_rois}/${h.progress.roi_target}`,
    `- Reviewed events: ${h.progress.reviewed_events}/${h.progress.event_target}`,
    `- Reviewed suggestions: ${h.progress.reviewed_suggestions}`,
    `- Remaining unlabeled: ${h.remaining.rois} ROIs, ${h.remaining.events} events, ${h.remaining.suggestions} suggestions`,
    `- Tuning-ready labels: ${h.progress.tuning_ready ? 'yes' : 'not yet'}`,
    '',
    '## Export Readiness',
    `- Accepted ROIs: ${h.export_ready.accepted_rois}`,
    `- Accepted events: ${h.export_ready.accepted_events}`,
    `- Control-ready ROIs: ${h.export_ready.control_ready_rois}`,
    `- Labels missing reviewer ID: ${missing}`,
    `- Ready for clean export: ${h.export_ready.ready ? 'yes' : 'not yet'}`,
    '',
    '## Suggested Next Work',
    `- Next batch contains ${h.progress.next_batch.rois} ROIs, ${h.progress.next_batch.events} events, and ${h.progress.next_batch.suggestions} missed-neuron suggestions.`,
    missing ? '- Stamp missing reviewer IDs before sharing final exports.' : '- Provenance is complete for currently reviewed labels.'
  ];
  return lines.join('\n') + '\n';
}

function renderReviewSessionPanel(){
  const root = document.getElementById('reviewSessionPanel');
  if(!root) return;
  const state = sessionChecklistState();
  const reviewerKind = state.reviewer_id ? 'ok' : 'warn';
  const saveKind = state.save.className === 'ok' ? 'ok' : state.save.className === 'bad' ? 'bad' : 'warn';
  const provenanceKind = state.provenance.totals.missing ? 'warn' : 'ok';
  const exportKind = state.export_ready.ready ? 'ok' : 'warn';
  root.innerHTML = `
    <div class="sessionHeader">
      <div>
        <h2>Review Session</h2>
        <p class="hint">${escapeHtml(state.run_label)} | ${escapeHtml(activeRunId())}</p>
      </div>
      <span class="stageStatus ${state.progress.tuning_ready ? 'ok' : 'warn'}">${state.progress.tuning_ready ? 'tuning ready' : 'needs labels'}</span>
    </div>
    <div class="sessionStatusGrid">
      ${sessionStatusChip('Reviewer', state.reviewer_id || 'not set', reviewerKind)}
      ${sessionStatusChip('Save', state.save.text || 'loading', saveKind)}
      ${sessionStatusChip('Provenance', `${state.provenance.totals.missing || 0} missing`, provenanceKind)}
      ${sessionStatusChip('Export', state.export_ready.ready ? 'ready' : 'review', exportKind)}
    </div>
    ${sessionProgressBar('ROI tuning labels', state.progress.reviewed_rois, state.progress.roi_target, `${state.remaining.rois} unlabeled in full ROI set`)}
    ${sessionProgressBar('Event tuning labels', state.progress.reviewed_events, state.progress.event_target, `${state.remaining.events} unlabeled events`)}
    <div class="sessionQueueGrid">
      <span><b>ROI</b> ${escapeHtml(state.queues.roi.name)} ${state.queues.roi.current || 0}/${state.queues.roi.total}</span>
      <span><b>Event</b> ${escapeHtml(state.queues.event.name)} ${state.queues.event.current || 0}/${state.queues.event.total}</span>
      <span><b>Suggest</b> ${escapeHtml(state.queues.suggestion.name)} ${state.queues.suggestion.current || 0}/${state.queues.suggestion.total}</span>
    </div>
    <div class="sessionChecklistActions">
      <button type="button" id="sessionHandoffMarkdownBtn">Handoff Markdown</button>
      <button type="button" id="sessionHandoffJsonBtn">Handoff JSON</button>
      <button type="button" id="sessionOpenMissingReviewerBtn" ${state.provenance.totals.missing ? '' : 'disabled'}>Next Missing</button>
    </div>`;
  document.getElementById('sessionHandoffMarkdownBtn').onclick = () => {
    downloadText(`${datasetId}_review_handoff.md`, reviewSessionHandoffMarkdown(), 'text/markdown');
    recordAction('export_session_handoff_markdown');
  };
  document.getElementById('sessionHandoffJsonBtn').onclick = () => {
    downloadJson(`${datasetId}_review_handoff.json`, reviewSessionHandoff());
    recordAction('export_session_handoff_json');
  };
  document.getElementById('sessionOpenMissingReviewerBtn').onclick = nextMissingReviewerLabel;
}

function renderGuidedPanel(){
  const root = document.getElementById('guidedPanel');
  if(!root) return;
  const tasks = guidedTasks();
  const task = currentGuidedTask();
  const s = annotationSummary();
  document.getElementById('reviewModeToggle')?.classList.toggle('guidedActive', setting('reviewMode') === 'guided');
  if(!task){
    root.innerHTML = '<p class="hint">No guided tasks remain for the current targets.</p>';
    return;
  }
  const idx = Math.max(0, Math.min(tasks.length - 1, Number(setting('guidedTaskIndex')) || 0));
  root.innerHTML = `
    <div class="guidedHero">
      <span class="runStatus">${escapeHtml(task.task_type)}</span>
      <h3>${escapeHtml(task.prompt)}</h3>
      <div class="reasonPills">${(task.reasons || []).map(r => `<span>${escapeHtml(r)}</span>`).join('')}</div>
      <p class="hint">${idx + 1} of ${tasks.length} guided tasks. Context: ${(task.recommended_context || []).join(', ')}.</p>
      ${guidedActionButtons(task)}
    </div>
    <div class="goalGrid">
      <div><b>${s.review_progress.reviewed_rois}/${targetCounts().rois}</b><span>ROI goal</span></div>
      <div><b>${s.review_progress.reviewed_events}/${targetCounts().events}</b><span>event goal</span></div>
      <div><b>${s.review_progress.reviewed_suggestions}/${targetCounts().suggestions}</b><span>suggestion goal</span></div>
    </div>
    <div class="buttonRow">
      <button id="guidedPrevBtn">Previous Task</button>
      <button id="guidedOpenBtn">Open Task</button>
      <button id="guidedNextBtn">Next Task</button>
    </div>`;
  for(const btn of root.querySelectorAll('[data-guided-action]')){
    btn.onclick = () => applyGuidedAction(btn.dataset.guidedAction);
  }
  document.getElementById('guidedPrevBtn').onclick = () => {
    setSetting('guidedTaskIndex', Math.max(0, idx - 1));
    selectGuidedTask();
    renderAll();
  };
  document.getElementById('guidedNextBtn').onclick = () => {
    setSetting('guidedTaskIndex', Math.min(tasks.length - 1, idx + 1));
    selectGuidedTask();
    renderAll();
  };
  document.getElementById('guidedOpenBtn').onclick = () => selectGuidedTask(task);
}

function exportRows(type) {
  const newline = String.fromCharCode(10);
  let rows = [];
  if (type === 'roi') {
    rows.push('roi_id\troi_kind\tsource_roi_ids\tstate\tcell_state\ttrace_quality\tcontrol_ready\tartifact_class\tidentity_group\tneeds_action\tconfidence\treason_tags\treviewer_id\tupdatedAt\tdeleted\tnotes\tcentroid_x\tcentroid_y\tarea\tpeak_score\tevent_count\tnoise_sigma\tpriority_score\tlocal_correlation_mean\tbackground_correlation\ttrace_snr\tevent_support\tartifact_score');
    for(const roi of data.rois){
      const ann = roiAnn(roi.id);
      const notes = (ann.notes || '').split(String.fromCharCode(9)).join(' ').split(newline).join(' ');
      rows.push([roi.id, 'source', '', ann.state || '', ann.cell_state || '', ann.trace_quality || '', ann.control_ready || '', ann.artifact_class || '', ann.identity_group || '', ann.needs_action || '', ann.confidence || '', (ann.reason_tags || []).join(','), ann.reviewer_id || '', ann.updatedAt || '', ann.deleted ? 1 : 0, notes, roi.centroidX, roi.centroidY, roi.area, roi.peakScore, eventsForRoi(roi).length, roi.noiseSigma, roi.priorityScore || '', roi.localCorrelationMean || '', roi.backgroundCorrelation || '', roi.traceSnr || '', roi.eventSupport || '', roi.artifactScore || ''].join('\t'));
    }
    for(const virtual of Object.values(annotations.virtualRois || {})){
      const ann = Object.assign({}, virtual, roiAnn(virtual.id));
      const notes = (ann.notes || '').split(String.fromCharCode(9)).join(' ').split(newline).join(' ');
      rows.push([virtual.id, virtual.roi_kind || 'virtual', (virtual.source_roi_ids || []).join(','), ann.state || '', ann.cell_state || '', ann.trace_quality || '', ann.control_ready || '', ann.artifact_class || '', ann.identity_group || '', ann.needs_action || '', ann.confidence || '', (ann.reason_tags || []).join(','), ann.reviewer_id || '', ann.updatedAt || '', ann.deleted ? 1 : 0, notes, virtual.centroidX || '', virtual.centroidY || '', virtual.area || '', '', '', '', '', '', '', '', '', ''].join('\t'));
    }
  } else if (type === 'event') {
    rows.push('roi_id\tframe\tstate\tevent_state\tevent_type\ttiming_quality\tconfidence\treason_tags\treviewer_id\tupdatedAt\tnotes\tz\tamplitude\troi_state');
    for(const roi of data.rois){
      for(const ev of eventsForRoi(roi)){
        const ann = eventAnn(roi.id, ev.frame);
        const notes = (ann.notes || '').split(String.fromCharCode(9)).join(' ').split(newline).join(' ');
        rows.push([roi.id, ev.frame, ann.state || '', ann.event_state || '', ann.event_type || '', ann.timing_quality || '', ann.confidence || '', (ann.reason_tags || []).join(','), ann.reviewer_id || '', ann.updatedAt || '', notes, ev.z.toFixed(4), ev.amplitude.toFixed(6), roiAnn(roi.id).state || ''].join('\t'));
      }
    }
  } else if (type === 'splitMerge') {
    rows.push('decision_id\tdecision_type\tdecision_state\tsource_roi_ids\ttarget_roi_ids\tvirtual_roi_id\tidentity_group\tneeds_action\tconfidence\treason_tags\treviewer_id\tupdatedAt\tnotes');
    for(const [decisionId, decision] of Object.entries(annotations.splitMergeDecisions || {})){
      const notes = (decision.notes || '').split(String.fromCharCode(9)).join(' ').split(newline).join(' ');
      rows.push([decision.id || decisionId, decision.decision_type || '', decision.decision_state || '', (decision.source_roi_ids || []).join(','), (decision.target_roi_ids || []).join(','), decision.virtual_roi_id || '', decision.identity_group || '', decision.needs_action || '', decision.confidence || '', (decision.reason_tags || []).join(','), decision.reviewer_id || '', decision.updatedAt || '', notes].join('\t'));
    }
  } else {
    rows.push('suggestion_id\tstate\tartifact_class\tconfidence\treason_tags\treviewer_id\tupdatedAt\tnotes\tpromoted\tcentroid_x\tcentroid_y\tarea\tdiscovery_score\tpriority_score\tlocal_correlation_mean\tevent_support\tartifact_score\tmax_z\tactive_frames\tartifact_cue\tprovenance');
    for(const s of data.discovery?.suggestions || []){
      const ann = suggestionAnn(s.id);
      const notes = (ann.notes || '').split(String.fromCharCode(9)).join(' ').split(newline).join(' ');
      rows.push([s.id, ann.state || '', ann.artifact_class || ann.artifactClass || '', ann.confidence || '', (ann.reason_tags || []).join(','), ann.reviewer_id || '', ann.updatedAt || '', notes, annotations.promotedRois[s.id] ? 1 : 0, s.centroidX, s.centroidY, s.area, s.discoveryScore, s.priorityScore || '', s.localCorrelationMean || '', s.eventSupport || '', s.artifactScore || '', s.maxZ, s.activeFrames, s.artifactCue || '', s.provenance || ''].join('\t'));
    }
  }
  const blob = new Blob([rows.join(newline) + newline], {type:'text/tab-separated-values'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = type === 'roi' ? 'neuron_roi_annotations.tsv' : type === 'event' ? 'neuron_event_annotations.tsv' : type === 'splitMerge' ? 'neuron_split_merge_decisions.tsv' : 'neuron_discovery_suggestions.tsv';
  a.click();
  URL.revokeObjectURL(a.href);
}

function downloadTsv(name, rows){
  const newline = String.fromCharCode(10);
  const blob = new Blob([rows.join(newline) + newline], {type:'text/tab-separated-values'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

function cleanTsv(value){
  return String(value ?? '').split(String.fromCharCode(9)).join(' ').split(String.fromCharCode(10)).join(' ');
}

function exportActiveQueue(type){
  const rows = [];
  if(type === 'roi'){
    rows.push('rank\tqueue\troi_id\tstate\tcell_state\ttrace_quality\tcontrol_ready\tartifact_class\tconfidence\treason_tags\treviewer_id\tupdatedAt\tarea\tevent_count\tpriority_score\ttrace_snr\tartifact_score\tneeds_action');
    visibleRois().forEach((roi, idx) => {
      const ann = roiAnn(roi.id);
      rows.push([idx + 1, setting('queue') || 'all', roi.id, ann.state || '', ann.cell_state || '', ann.trace_quality || '', ann.control_ready || '', ann.artifact_class || '', ann.confidence || '', (ann.reason_tags || []).join(','), ann.reviewer_id || '', ann.updatedAt || '', roi.area, eventsForRoi(roi).length, roi.priorityScore || '', roi.traceSnr || '', roi.artifactScore || '', ann.needs_action || ''].map(cleanTsv).join('\t'));
    });
    downloadTsv(`${datasetId}_active_roi_queue.tsv`, rows);
  } else if(type === 'event'){
    rows.push('rank\tevent_queue\troi_id\tframe\tstate\tevent_state\tevent_type\ttiming_quality\tconfidence\treason_tags\treviewer_id\tupdatedAt\tz\tamplitude\troi_state');
    eventQueueItems().forEach((item, idx) => {
      const ann = eventAnn(item.roi.id, item.ev.frame);
      rows.push([idx + 1, setting('eventQueue') || 'all', item.roi.id, item.ev.frame, ann.state || '', ann.event_state || '', ann.event_type || '', ann.timing_quality || '', ann.confidence || '', (ann.reason_tags || []).join(','), ann.reviewer_id || '', ann.updatedAt || '', fmt(item.ev.z, 4), fmt(item.ev.amplitude, 6), roiAnn(item.roi.id).state || ''].map(cleanTsv).join('\t'));
    });
    downloadTsv(`${datasetId}_active_event_queue.tsv`, rows);
  } else {
    rows.push('rank\tdiscovery_queue\tsuggestion_id\tstate\tartifact_class\tconfidence\treason_tags\treviewer_id\tupdatedAt\tpromoted\tarea\tdiscovery_score\tpriority_score\tevent_support\tartifact_score\tartifact_cue\tprovenance');
    visibleSuggestions().forEach((s, idx) => {
      const ann = suggestionAnn(s.id);
      rows.push([idx + 1, setting('discoveryQueue') || 'all', s.id, ann.state || '', ann.artifact_class || ann.artifactClass || '', ann.confidence || '', (ann.reason_tags || []).join(','), ann.reviewer_id || '', ann.updatedAt || '', annotations.promotedRois[s.id] ? 1 : 0, s.area, s.discoveryScore || '', s.priorityScore || '', s.eventSupport || '', s.artifactScore || '', s.artifactCue || '', s.provenance || ''].map(cleanTsv).join('\t'));
    });
    downloadTsv(`${datasetId}_active_suggestion_queue.tsv`, rows);
  }
  recordAction(`export_active_${type}_queue`);
}

function exportJson() {
  annotations.updatedAt = new Date().toISOString();
  const blob = new Blob([JSON.stringify(annotations, null, 2) + String.fromCharCode(10)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'annotations_v3.json';
  a.click();
  URL.revokeObjectURL(a.href);
}

function reviewedAnnotationState(group, id, ann){
  if(group === 'events') return ann.event_state || ann.state || '';
  if(group === 'suggestions') return annotations.promotedRois?.[id] ? 'promoted' : ann.state || '';
  if(group === 'split_merge_decisions') return ann.decision_state || '';
  return ann.cell_state || ann.state || '';
}

function reviewerProvenanceAudit(){
  const audit = {
    schema_version: 1,
    dataset_id: datasetId,
    generatedAt: new Date().toISOString(),
    current_reviewer_id: currentReviewerId(),
    totals: {reviewed: 0, stamped: 0, missing: 0},
    by_group: {},
    by_reviewer: {},
    missing_items: []
  };
  const note = (group, id, ann, meta={}) => {
    const state = reviewedAnnotationState(group, id, ann);
    if(!state) return;
    const reviewer = String(ann.reviewer_id || '').trim();
    const updatedAt = ann.updatedAt || '';
    const bucket = audit.by_group[group] || {reviewed: 0, stamped: 0, missing: 0};
    bucket.reviewed++;
    audit.totals.reviewed++;
    if(reviewer) {
      bucket.stamped++;
      audit.totals.stamped++;
      audit.by_reviewer[reviewer] = (audit.by_reviewer[reviewer] || 0) + 1;
    } else {
      bucket.missing++;
      audit.totals.missing++;
      audit.missing_items.push(Object.assign({group, id, state, updatedAt}, meta));
    }
    audit.by_group[group] = bucket;
  };
  for(const roi of data.rois) {
    note('rois', String(roi.id), roiAnn(roi.id), {area: roi.area, event_count: eventsForRoi(roi).length});
    for(const ev of eventsForRoi(roi)) note('events', eventKey(roi.id, ev.frame), eventAnn(roi.id, ev.frame), {roi_id: roi.id, frame: ev.frame});
  }
  for(const [id, roi] of Object.entries(annotations.virtualRois || {})) {
    note('virtual_rois', String(id), roi, {source_roi_ids: roi.source_roi_ids || []});
  }
  for(const s of data.discovery?.suggestions || []) {
    note('suggestions', String(s.id), suggestionAnn(s.id), {promoted: Boolean(annotations.promotedRois?.[s.id]), area: s.area});
  }
  for(const [id, decision] of Object.entries(annotations.splitMergeDecisions || {})) {
    note('split_merge_decisions', String(id), decision, {
      decision_type: decision.decision_type || '',
      source_roi_ids: decision.source_roi_ids || [],
      target_roi_ids: decision.target_roi_ids || []
    });
  }
  audit.coverage_fraction = audit.totals.reviewed ? audit.totals.stamped / audit.totals.reviewed : 1;
  return audit;
}

function exportReviewerProvenanceAudit(){
  downloadJson(`${datasetId}_reviewer_provenance_audit.json`, reviewerProvenanceAudit());
  recordAction('export_reviewer_provenance_audit');
}

function provenanceItemKey(item){
  return `${item.group}:${item.id}`;
}

function currentProvenanceItemKey(){
  const roi = selectedRoi();
  if(roi && selectedEventFrame) {
    const eAnn = eventAnn(roi.id, selectedEventFrame);
    if((eAnn.state || eAnn.event_state) && !String(eAnn.reviewer_id || '').trim()) return `events:${eventKey(roi.id, selectedEventFrame)}`;
  }
  if(roi) {
    const rAnn = roiAnn(roi.id);
    if((rAnn.state || rAnn.cell_state) && !String(rAnn.reviewer_id || '').trim()) return `${annotations.virtualRois?.[roi.id] ? 'virtual_rois' : 'rois'}:${roi.id}`;
  }
  if(selectedSuggestionId) {
    const sAnn = suggestionAnn(selectedSuggestionId);
    if((sAnn.state || annotations.promotedRois?.[selectedSuggestionId]) && !String(sAnn.reviewer_id || '').trim()) {
      return `suggestions:${selectedSuggestionId}`;
    }
  }
  return '';
}

function openProvenanceAuditItem(item){
  if(!item) return;
  if(item.group === 'rois' || item.group === 'virtual_rois') selectRoi(item.id);
  else if(item.group === 'events') {
    selectRoi(item.roi_id);
    selectedEventFrame = Number(item.frame);
    eventNotes.value = eventAnn(item.roi_id, selectedEventFrame).notes || '';
    setFrame(selectedEventFrame);
    renderAll();
  } else if(item.group === 'suggestions') {
    selectSuggestion(item.id);
  } else {
    setSaveState(`missing reviewer on ${item.group} ${item.id}`, 'bad');
  }
}

function nextMissingReviewerLabel(){
  const missing = reviewerProvenanceAudit().missing_items || [];
  if(!missing.length) {
    setSaveState('no reviewed labels missing reviewer IDs', 'ok');
    return;
  }
  const currentKey = currentProvenanceItemKey();
  const idx = missing.findIndex(item => provenanceItemKey(item) === currentKey);
  const next = missing[(idx + 1 + missing.length) % missing.length];
  openProvenanceAuditItem(next);
  setSaveState(`opened ${next.group} ${next.id} missing reviewer ID`, 'bad');
}

function nextRoiMatching(predicate, delta=1){
  const rows = visibleRois().filter(predicate);
  if(!rows.length) return;
  const currentIndex = rows.findIndex(r => String(r.id) === String(selectedId));
  const base = currentIndex >= 0 ? currentIndex : (delta > 0 ? -1 : 0);
  const next = rows[(base + delta + rows.length) % rows.length];
  selectRoi(next.id);
}

function eventfulFrames(){
  return [...new Set(visibleRois().flatMap(roi => eventsForRoi(roi).map(ev => ev.frame)))].sort((a,b) => a - b);
}

function nextActiveFrame(delta=1){
  const frames = eventfulFrames();
  if(!frames.length) return;
  let next = frames[0];
  if(delta > 0) next = frames.find(f => f > currentFrame) || frames[0];
  else next = [...frames].reverse().find(f => f < currentFrame) || frames[frames.length - 1];
  setFrame(next);
}

function applyTracePreset(kind){
  const frames = Math.max(1, Number(data.video?.frames) || 1);
  if(kind === 'full') {
    resetTraceZoom();
    return;
  }
  const roi = selectedRoi();
  const hz = Math.max(1, datasetFrameRateHz());
  const seconds = kind === 'event5s' ? 5 : 2;
  const halfWindow = Math.max(2, Math.round(seconds * hz));
  let center = selectedEventFrame || currentFrame;
  if(kind.startsWith('event') && roi) {
    const events = eventsForRoi(roi);
    if(events.length && !events.some(ev => ev.frame === center)) {
      center = events.reduce((best, ev) => Math.abs(ev.frame - currentFrame) < Math.abs(best.frame - currentFrame) ? ev : best, events[0]).frame;
      selectedEventFrame = center;
    }
  }
  setTraceWindow(Math.max(1, center - halfWindow), Math.min(frames, center + halfWindow));
  setFrame(center);
  drawTrace();
}

function renderFocusSummary(){
  const root = document.getElementById('focusSummary');
  if(!root) return;
  const mode = roiFocusMode();
  const visible = visibleRois().length;
  const overlayed = visibleOverlayRois().length;
  const selected = selectedRoi();
  const radius = Number(setting('neighborRadiusPx')) || 36;
  root.textContent = mode === 'all'
    ? `${visible} queue ROIs shown`
    : mode === 'solo'
      ? `solo ROI ${selected?.id || ''}`
      : `${overlayed}/${visible} ROIs within ${Math.round(radius)} px`;
}

function renderSuggestionContext(){
  const root = document.getElementById('suggestionContextCard');
  if(!root) return;
  const s = selectedSuggestion();
  if(!s) {
    root.innerHTML = '<p class="hint">No discovery suggestion selected.</p>';
    return;
  }
  const nearest = nearestRoiForSuggestion(s);
  const ann = suggestionAnn(s.id);
  const duplicateRisk = nearest && nearest.distance <= Math.max(8, Number(setting('neighborRadiusPx')) || 36);
  root.innerHTML = `
    <table class="smallTable">
      <tr><th>Field</th><th>Value</th></tr>
      <tr><td>Suggestion</td><td>${escapeHtml(s.id)}</td></tr>
      <tr><td>state</td><td>${escapeHtml(annotations.promotedRois[s.id] ? 'promoted' : ann.state || 'new')}</td></tr>
      <tr><td>priority</td><td>${fmt(scoreValue(s, 'priorityScore', s.discoveryScore), 3)}</td></tr>
      <tr><td>nearest ROI</td><td>${nearest ? `#${nearest.roi.id} (${fmt(nearest.distance, 1)} px)` : 'n/a'}</td></tr>
      <tr><td>duplicate risk</td><td>${duplicateRisk ? 'possible duplicate/merge' : 'low'}</td></tr>
    </table>`;
}

function markSuggestionDuplicate(){
  const s = selectedSuggestion(); if(!s) return;
  const nearest = nearestRoiForSuggestion(s);
  pushAnnotationUndo(`suggestion ${s.id} duplicate`, [annotationSnapshot('suggestions', s.id)]);
  annotations.suggestions[s.id] = stampAnnotation(Object.assign(suggestionAnn(s.id), {
    state: 'artifact',
    artifactClass: 'duplicate_existing_roi',
    artifact_class: 'duplicate_existing_roi',
    notes: `${suggestionAnn(s.id).notes || ''}${suggestionAnn(s.id).notes ? '\n' : ''}Possible duplicate of ROI ${nearest?.roi?.id || 'unknown'}.`
  }));
  recordAction('suggestion_duplicate');
  queueSave();
  renderAll();
}
function markSuggestionDuplicateAndNext(){
  const rows = visibleSuggestions();
  const currentId = selectedSuggestion()?.id;
  markSuggestionDuplicate();
  advanceSuggestionFromRows(rows, currentId, 1);
}

function activateMissedNeuronMode(){
  const details = document.getElementById('discoveryDetails');
  if(details) details.open = true;
  document.getElementById('showSuggestions').checked = true;
  document.getElementById('showEvidence').checked = true;
  setSetting('showSuggestions', true);
  setSetting('showEvidence', true);
  setSetting('discoveryQueue', 'unlabeled');
  applyOverlayPreset('discovery');
  const rows = visibleSuggestions();
  if(rows.length) selectSuggestion(rows[0].id);
  else renderAll();
}

function snapshotFields(){
  return ['eventThreshold','kalmanGain','spikeGain','overlayOpacity','overlayPreset','selectedOverlayMode','selectedFillOpacity','selectedOutlineWidth','roiFocusMode','neighborRadiusPx','queue','eventQueue','discoveryQueue','evidenceMap','showEvidence','showSuggestions','minArea','minEvents','activeRunId'];
}

function parameterSnapshots(){
  const items = annotations.settings.parameterSnapshots;
  if(Array.isArray(items)) return items;
  annotations.settings.parameterSnapshots = [];
  return annotations.settings.parameterSnapshots;
}

function currentSnapshotPayload(){
  const settings = {};
  for(const key of snapshotFields()) settings[key] = setting(key);
  return {
    settings,
    frame: currentFrame,
    selectedId,
    selectedEventFrame,
    traceView: {start: traceView.start, end: traceView.end}
  };
}

function saveParameterSnapshot(){
  const name = prompt('Snapshot name', `snapshot_${parameterSnapshots().length + 1}`);
  if(!name) return;
  const id = `snapshot_${Date.now().toString(36)}`;
  parameterSnapshots().push({id, name: name.trim(), createdAt: new Date().toISOString(), payload: currentSnapshotPayload()});
  setSetting('activeSnapshotId', id);
  recordAction('parameter_snapshot_save');
  queueSave();
  renderSnapshotControls();
}

function restoreParameterSnapshot(id){
  const snap = parameterSnapshots().find(s => s.id === id);
  if(!snap) return;
  for(const [key, value] of Object.entries(snap.payload?.settings || {})) annotations.settings[key] = value;
  annotations.settings.activeSnapshotId = id;
  traceView = Object.assign(traceView, snap.payload?.traceView || {});
  applySettingsToControls();
  if(snap.payload?.selectedId) selectedId = snap.payload.selectedId;
  if(snap.payload?.selectedEventFrame) selectedEventFrame = snap.payload.selectedEventFrame;
  setFrame(snap.payload?.frame || currentFrame);
  recordAction('parameter_snapshot_restore');
  queueSave();
  renderAll();
}

function deleteParameterSnapshot(){
  const select = document.getElementById('parameterSnapshotSelect');
  const id = select?.value;
  if(!id) return;
  annotations.settings.parameterSnapshots = parameterSnapshots().filter(s => s.id !== id);
  if(setting('activeSnapshotId') === id) annotations.settings.activeSnapshotId = '';
  recordAction('parameter_snapshot_delete');
  queueSave();
  renderSnapshotControls();
}

function renderSnapshotControls(){
  const select = document.getElementById('parameterSnapshotSelect');
  const summary = document.getElementById('snapshotSummary');
  if(!select) return;
  const snaps = parameterSnapshots();
  const active = setting('activeSnapshotId') || '';
  select.innerHTML = '<option value="">No snapshot selected</option>' + snaps.map(s => `<option value="${escapeHtml(s.id)}">${escapeHtml(s.name || s.id)}</option>`).join('');
  select.value = snaps.some(s => s.id === active) ? active : '';
  if(summary) {
    const snap = snaps.find(s => s.id === select.value);
    summary.textContent = snap ? `saved ${new Date(snap.createdAt).toLocaleString()}` : `${snaps.length} saved`;
  }
}

function exportCurrentViewPng(){
  const scale = 1;
  const metaH = 68;
  const gap = 12;
  const w = Math.max(data.video.width, traceCanvas.width);
  const h = metaH + data.video.height + gap + traceCanvas.height;
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(w * scale);
  canvas.height = Math.round(h * scale);
  const out = canvas.getContext('2d');
  out.fillStyle = '#ffffff';
  out.fillRect(0, 0, canvas.width, canvas.height);
  out.fillStyle = '#0f172a';
  out.font = '18px Arial';
  out.fillText(`${datasetId} | frame ${currentFrame} | ROI ${selectedRoi()?.id || 'n/a'}`, 12, 25);
  out.font = '13px Arial';
  out.fillStyle = '#475569';
  out.fillText(`event ${selectedEventFrame || 'n/a'} | overlay ${setting('overlayPreset') || 'custom'} | queue ${setting('queue')}`, 12, 48);
  out.drawImage(img, 0, metaH, data.video.width, data.video.height);
  out.drawImage(overlay, 0, metaH, data.video.width, data.video.height);
  out.drawImage(traceCanvas, 0, metaH + data.video.height + gap, traceCanvas.width, traceCanvas.height);
  const a = document.createElement('a');
  a.href = canvas.toDataURL('image/png');
  a.download = `${datasetId}_frame_${String(currentFrame).padStart(3, '0')}_review.png`;
  a.click();
}

function nextRoi(delta){
  const rows = visibleRois();
  if(!rows.length) return;
  const idx = Math.max(0, rows.findIndex(r => r.id === selectedId));
  const next = rows[(idx + delta + rows.length) % rows.length];
  selectRoi(next.id);
}
function nextEvent(delta){
  const roi = selectedRoi();
  if(!roi) return;
  const events = visibleEventsForRoi(roi);
  if(!events.length) return;
  const idx = Math.max(0, events.findIndex(e => e.frame === selectedEventFrame));
  selectedEventFrame = events[(idx + delta + events.length) % events.length].frame;
  eventNotes.value = eventAnn(roi.id, selectedEventFrame).notes || '';
  setFrame(selectedEventFrame);
  renderAll();
}
function selectEventQueueItem(item){
  if(!item) return;
  selectedId = item.roi.id;
  selectedRoiIds = new Set([String(item.roi.id)]);
  selectedEventFrame = item.ev.frame;
  roiNotes.value = roiAnn(item.roi.id).notes || '';
  eventNotes.value = eventAnn(item.roi.id, item.ev.frame).notes || '';
  setFrame(item.ev.frame);
  renderAll();
}
function nextEventQueue(delta=1){
  const rows = eventQueueItems();
  if(!rows.length) {
    setSaveState('no events match the current event queue', 'bad');
    renderEventList();
    return;
  }
  const roi = selectedRoi();
  const idx = rows.findIndex(item => roi && String(item.roi.id) === String(roi.id) && item.ev.frame === selectedEventFrame);
  const base = idx >= 0 ? idx : delta > 0 ? -1 : 0;
  selectEventQueueItem(rows[(base + delta + rows.length) % rows.length]);
}
function togglePlay(){
  playing = !playing;
  document.getElementById('playBtn').textContent = playing ? 'Pause' : 'Play';
  if(playing) timer = setInterval(() => setFrame(currentFrame >= data.video.frames ? 1 : currentFrame + 1), 120);
  else clearInterval(timer);
}
function fitWidth(){
  const width = Math.max(1, viewerScroll.clientWidth - 34);
  setSetting('zoom', Math.max(0.5, width / data.video.width));
  applySettingsToControls();
}

function initControls(){
  slider.max = data.video.frames;
  slider.oninput = () => setFrame(Number(slider.value));
  document.getElementById('playBtn').onclick = togglePlay;
  document.getElementById('fitBtn').onclick = fitWidth;
  document.getElementById('fullscreenBtn').onclick = () => viewerScroll.requestFullscreen?.();
  document.getElementById('prevActiveFrameBtn').onclick = () => nextActiveFrame(-1);
  document.getElementById('nextActiveFrameBtn').onclick = () => nextActiveFrame(1);
  document.getElementById('exportScreenshotBtn').onclick = exportCurrentViewPng;
  document.getElementById('eventWindowPrevBtn').onclick = () => nextEvent(-1);
  document.getElementById('eventWindowNextBtn').onclick = () => nextEvent(1);
  document.getElementById('activeRunSelect').onchange = e => selectActiveRun(e.target.value, {loadReview:false});
  document.getElementById('loadRunReviewBtn').onclick = () => selectActiveRun(document.getElementById('activeRunSelect').value, {loadReview:true});
  document.getElementById('openRunViewBtn').onclick = () => {
    const run = runById(activeRunId());
    const url = artifactUrl(runAppUrl(run));
    if(url) location.href = url;
  };
  document.getElementById('previewRunViewBtn').onclick = () => startGenerationJob({preview:true});
  document.getElementById('generateRunViewBtn').onclick = () => startGenerationJob({preview:false});
  document.getElementById('unlockGenerationBtn').onclick = () => {
    const token = prompt('Owner token for local generation jobs');
    if(token !== null) {
      generationOwnerToken = token.trim();
      if(generationOwnerToken) localStorage.setItem(ownerTokenKey, generationOwnerToken);
      else localStorage.removeItem(ownerTokenKey);
      renderRunSyncControls();
    }
  };
  document.getElementById('refreshRunBtn').onclick = refreshArchitectureRuns;
  document.getElementById('generationBackend').onchange = renderRunSyncControls;
  document.getElementById('archRunA').onchange = e => {
    reviewCompareSettings().runAId = e.target.value;
    queueSave();
    renderRunComparison();
  };
  document.getElementById('archRunB').onchange = e => {
    reviewCompareSettings().runBId = e.target.value;
    queueSave();
    renderRunComparison();
  };
  document.getElementById('archCompareModeBtn').onclick = () => setArchitectureMode('compare');
  document.getElementById('archBuildModeBtn').onclick = () => setArchitectureMode('build');
  document.getElementById('pipelineNewArchitectureBtn').onclick = () => {
    pipelineDraft = makePresetPipeline('current_review_pipeline');
    pipelineDraft.label = 'Untitled architecture';
    pipelineDraft.run_id = `planned_architecture_${Date.now().toString(36)}`;
    pipelineDraft.template_id = slugify(pipelineDraft.run_id);
    pipelineDraft.description = '';
    selectedPipelineStageId = pipelineDraft.pipeline[0]?.id || null;
    setArchitectureMode('build');
    renderPipelineBuilder();
  };
  document.getElementById('pipelineNewBtn').onclick = () => {
    pipelineDraft = makePresetPipeline(document.getElementById('pipelinePresetSelect').value);
    pipelineDraft.template_id = slugify(pipelineDraft.run_id);
    selectedPipelineStageId = pipelineDraft.pipeline[0]?.id || null;
    renderPipelineBuilder();
  };
  document.getElementById('pipelineCloneRunBtn').onclick = () => {
    const runs = data.architectureRuns?.runs || [];
    const selected = runs.find(r => r.run_id === document.getElementById('archRunA').value) || runs[0];
    if(selected) {
      pipelineDraft = normalizePipelineDraft(JSON.parse(JSON.stringify(Object.assign({}, selected, {execution:{status:'planned'}}))));
      pipelineDraft.run_id = `planned_clone_${Date.now().toString(36)}`;
      pipelineDraft.label = `Planned clone of ${selected.label || selected.run_id}`;
      pipelineDraft.template_id = slugify(pipelineDraft.run_id);
      selectedPipelineStageId = pipelineDraft.pipeline?.[0]?.id || null;
      renderPipelineBuilder();
    }
  };
  document.getElementById('pipelineSaveTemplateBtn').onclick = saveArchitectureTemplate;
  document.getElementById('pipelineSaveBtn').onclick = savePlannedRun;
  document.getElementById('pipelineUseExperimentBtn').onclick = useCurrentArchitectureInExperiment;
  document.getElementById('pipelineDownloadBtn').onclick = () => downloadJson('planned_architecture_run.json', plannedManifest());
  document.getElementById('uiMode').onchange = e => {
    setSetting('uiMode', e.target.value);
    applyUiMode();
  };
  for(const select of document.querySelectorAll('.themeSelect')) select.onchange = e => {
    setSetting('theme', e.target.value);
    applyTheme();
  };
  window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener?.('change', () => {
    if((setting('theme') || 'system') === 'system') applyTheme();
  });
  document.getElementById('reviewerIdInput').onchange = e => {
    setSetting('reviewerId', e.target.value.trim());
    recordAction('reviewer_id_set');
    renderAll();
  };
  document.getElementById('nextMissingReviewerBtn').onclick = nextMissingReviewerLabel;
  document.getElementById('stampSelectedReviewerBtn').onclick = stampSelectedReviewer;
  document.getElementById('stampMissingReviewerBtn').onclick = stampMissingReviewerLabels;
  document.getElementById('reviewModeToggle').onclick = () => {
    const next = setting('reviewMode') === 'guided' ? 'explore' : 'guided';
    setSetting('reviewMode', next);
    setSetting('reviewWorkflowPreset', 'custom');
    if(next === 'guided') {
      setSetting('queue', 'annotationBatch');
      selectGuidedTask();
    }
    applySettingsToControls();
    renderAll();
  };
  document.getElementById('reviewWorkflowPreset').onchange = e => applyReviewWorkflowPreset(e.target.value);
  document.getElementById('shortcutHelpBtn').onclick = () => toggleShortcutHelp(true);
  document.getElementById('shortcutCloseBtn').onclick = () => toggleShortcutHelp(false);
  document.getElementById('shortcutOverlay').addEventListener('click', e => {
    if(e.target.id === 'shortcutOverlay') toggleShortcutHelp(false);
  });
  document.getElementById('quickJumpBtn').onclick = () => quickJump(document.getElementById('quickJumpInput').value);
  document.getElementById('quickJumpInput').addEventListener('keydown', e => {
    if(e.key === 'Enter') {
      e.preventDefault();
      quickJump(e.target.value);
    }
  });
  document.getElementById('undoAnnotationBtn').onclick = undoLastAnnotationChange;
  document.getElementById('bookmarkAddBtn').onclick = addReviewBookmark;
  document.getElementById('bookmarkGoBtn').onclick = goToReviewBookmark;
  document.getElementById('bookmarkDeleteBtn').onclick = deleteReviewBookmark;
  for(const id of ['showRois','showLabels','showEvents']) document.getElementById(id).onchange = drawOverlay;
  document.getElementById('showSuggestions').onchange = e => { setSetting('showSuggestions', e.target.checked); drawOverlay(); };
  document.getElementById('showEvidence').onchange = e => { setSetting('showEvidence', e.target.checked); applyDisplaySettings(); };
  document.getElementById('evidenceSelect').onchange = e => { setSetting('evidenceMap', e.target.value); applyDisplaySettings(); };
  const overlayPresetSelect = document.getElementById('overlayPresetSelect');
  if(overlayPresetSelect) overlayPresetSelect.onchange = e => applyOverlayPreset(e.target.value);
  const selectedOverlayMode = document.getElementById('selectedOverlayMode');
  if(selectedOverlayMode) selectedOverlayMode.onchange = e => {
    setSetting('selectedOverlayMode', e.target.value);
    setSetting('overlayPreset', 'custom');
    applySettingsToControls();
    drawOverlay();
  };
  const roiFocusSelect = document.getElementById('roiFocusMode');
  if(roiFocusSelect) roiFocusSelect.onchange = e => {
    setSetting('roiFocusMode', e.target.value);
    applySettingsToControls();
    renderAll();
  };
  const traceResetZoomBtn = document.getElementById('traceResetZoomBtn');
  if(traceResetZoomBtn) traceResetZoomBtn.onclick = resetTraceZoom;
  document.getElementById('manualRoiMode').onchange = e => {
    setSetting('manualRoiMode', e.target.value);
    manualRoiState = {drawing:false, start:null, points:[], preview:null, suppressClick:false};
    if(e.target.value !== 'select') setSetting('roiEditMode', 'off');
    applySettingsToControls();
    drawOverlay();
  };
  document.getElementById('manualRoiCancelBtn').onclick = cancelManualRoi;
  document.getElementById('roiEditMode').onchange = e => {
    setSetting('roiEditMode', e.target.value);
    roiEditState = {drawing:false, editedId:null};
    if(e.target.value !== 'off') setSetting('manualRoiMode', 'select');
    applySettingsToControls();
    drawOverlay();
  };
  document.getElementById('roiEditDoneBtn').onclick = () => {
    setSetting('roiEditMode', 'off');
    roiEditState = {drawing:false, editedId:null};
    applySettingsToControls();
    drawOverlay();
  };
  document.getElementById('roiEditUndoBtn').onclick = undoRoiEdit;
  document.getElementById('roiEditRevertBtn').onclick = revertEditedRoiToSource;
  document.getElementById('materializeManualTracesBtn').onclick = materializeManualTraces;
  document.getElementById('traceFullBtn').onclick = () => applyTracePreset('full');
  document.getElementById('traceEvent2sBtn').onclick = () => applyTracePreset('event2s');
  document.getElementById('traceEvent5sBtn').onclick = () => applyTracePreset('event5s');
  for(const id of ['eventThreshold','kalmanGain','spikeGain','zoom','brightness','contrast','overlayOpacity','selectedFillOpacity','selectedOutlineWidth','neighborRadiusPx','manualRoiRadius','roiEditBrushRadius','minArea','minEvents']) {
    const control = document.getElementById(id);
    if(!control) continue;
    control.oninput = e => {
      const value = Number(e.target.value);
      setSetting(id, value);
      if(id === 'overlayOpacity' || id === 'selectedFillOpacity' || id === 'selectedOutlineWidth') setSetting('overlayPreset', 'custom');
      if(id === 'kalmanGain' || id === 'spikeGain') clearTraceCaches(id);
      if(id === 'eventThreshold') clearTraceEventCache(id);
      applySettingsToControls();
      renderAll();
    };
  }
  document.getElementById('queueSelect').onchange = e => { setSetting('queue', e.target.value); setSetting('reviewWorkflowPreset', 'custom'); renderAll(); };
  document.getElementById('queuePrevBtn').onclick = () => nextRoi(-1);
  document.getElementById('queueNextBtn').onclick = () => nextRoi(1);
  document.getElementById('nextActiveRoiBtn').onclick = () => nextRoiMatching(roi => eventsForRoi(roi).length > 0, 1);
  document.getElementById('nextUncertainRoiBtn').onclick = () => nextRoiMatching(roi => !roiAnn(roi.id).state || roiAnn(roi.id).state === 'unsure', 1);
  document.getElementById('nextArtifactRiskBtn').onclick = () => nextRoiMatching(roiArtifactLike, 1);
  document.getElementById('missedNeuronModeBtn').onclick = activateMissedNeuronMode;
  document.getElementById('bulkAcceptBtn').onclick = () => setSelectedRoisState('accept');
  document.getElementById('bulkRejectBtn').onclick = () => setSelectedRoisState('reject');
  document.getElementById('bulkUnsureBtn').onclick = () => setSelectedRoisState('unsure');
  document.getElementById('bulkIdentityBtn').onclick = assignSelectedIdentity;
  document.getElementById('bulkNeedsActionBtn').onclick = markSelectedAction;
  document.getElementById('virtualMergeBtn').onclick = createVirtualMerge;
  document.getElementById('visualSplitBtn').onclick = createVisualSplitDecision;
  document.getElementById('clearMultiSelectBtn').onclick = clearMultiSelection;
  document.getElementById('acceptBtn').onclick = () => setRoiState('accept');
  document.getElementById('rejectBtn').onclick = () => setRoiState('reject');
  document.getElementById('unsureBtn').onclick = () => setRoiState('unsure');
  document.getElementById('acceptNextBtn').onclick = () => setRoiStateAndNext('accept');
  document.getElementById('rejectNextBtn').onclick = () => setRoiStateAndNext('reject');
  document.getElementById('unsureNextBtn').onclick = () => setRoiStateAndNext('unsure');
  document.getElementById('strongNeuronNextBtn').onclick = markRoiStrongAndNext;
  document.getElementById('artifactRoiNextBtn').onclick = markRoiArtifactAndNext;
  document.getElementById('clearBtn').onclick = () => setRoiState('');
  document.getElementById('deleteBtn').onclick = toggleDeleted;
  document.getElementById('eventAcceptBtn').onclick = () => setEventState('accept');
  document.getElementById('eventRejectBtn').onclick = () => setEventState('reject');
  document.getElementById('eventUnsureBtn').onclick = () => setEventState('unsure');
  document.getElementById('eventAcceptNextBtn').onclick = () => setEventStateAndNext('accept');
  document.getElementById('eventRejectNextBtn').onclick = () => setEventStateAndNext('reject');
  document.getElementById('eventUnsureNextBtn').onclick = () => setEventStateAndNext('unsure');
  document.getElementById('eventArtifactNextBtn').onclick = markEventArtifactAndNext;
  document.getElementById('eventClearBtn').onclick = () => setEventState('');
  document.getElementById('eventQueueSelect').onchange = e => {
    setSetting('eventQueue', e.target.value);
    setSetting('reviewWorkflowPreset', 'custom');
    renderAll();
  };
  document.getElementById('eventQueuePrevBtn').onclick = () => nextEventQueue(-1);
  document.getElementById('eventQueueNextBtn').onclick = () => nextEventQueue(1);
  document.getElementById('suggestionPromoteBtn').onclick = promoteSuggestion;
  document.getElementById('suggestionPromoteNextBtn').onclick = promoteSuggestionAndNext;
  document.getElementById('suggestionMissedBtn').onclick = () => setSuggestionState('missed');
  document.getElementById('suggestionMissedNextBtn').onclick = () => setSuggestionStateAndNext('missed');
  document.getElementById('suggestionDuplicateBtn').onclick = markSuggestionDuplicate;
  document.getElementById('suggestionDuplicateNextBtn').onclick = markSuggestionDuplicateAndNext;
  document.getElementById('suggestionArtifactBtn').onclick = () => setSuggestionState('artifact');
  document.getElementById('suggestionArtifactNextBtn').onclick = () => setSuggestionStateAndNext('artifact');
  document.getElementById('suggestionUnsureBtn').onclick = () => setSuggestionState('unsure');
  document.getElementById('suggestionUnsureNextBtn').onclick = () => setSuggestionStateAndNext('unsure');
  document.getElementById('suggestionClearBtn').onclick = () => setSuggestionState('');
  document.getElementById('artifactClass').onchange = e => {
    const s = selectedSuggestion(); if(!s) return;
    annotations.suggestions[s.id] = Object.assign(suggestionAnn(s.id), {artifactClass:e.target.value, artifact_class:e.target.value});
    queueSave();
    renderAll();
  };
  document.getElementById('discoveryQueueSelect').onchange = e => { setSetting('discoveryQueue', e.target.value); renderAll(); };
  document.getElementById('suggestionQueuePrevBtn').onclick = () => nextSuggestion(-1);
  document.getElementById('suggestionQueueNextBtn').onclick = () => nextSuggestion(1);
  document.getElementById('exportRoiBtn').onclick = () => exportRows('roi');
  document.getElementById('exportEventBtn').onclick = () => exportRows('event');
  document.getElementById('exportSuggestionBtn').onclick = () => exportRows('suggestion');
  document.getElementById('exportSplitMergeBtn').onclick = () => exportRows('splitMerge');
  document.getElementById('exportActiveRoiQueueBtn').onclick = () => exportActiveQueue('roi');
  document.getElementById('exportActiveEventQueueBtn').onclick = () => exportActiveQueue('event');
  document.getElementById('exportActiveSuggestionQueueBtn').onclick = () => exportActiveQueue('suggestion');
  document.getElementById('exportJsonBtn').onclick = exportJson;
  document.getElementById('exportProvenanceAuditBtn').onclick = exportReviewerProvenanceAudit;
  document.getElementById('snapshotSaveBtn').onclick = saveParameterSnapshot;
  document.getElementById('snapshotRestoreBtn').onclick = () => restoreParameterSnapshot(document.getElementById('parameterSnapshotSelect').value);
  document.getElementById('snapshotDeleteBtn').onclick = deleteParameterSnapshot;
  document.getElementById('parameterSnapshotSelect').onchange = e => setSetting('activeSnapshotId', e.target.value);
  document.getElementById('recoveryRestoreBtn').onclick = restoreRecoverySnapshot;
  document.getElementById('recoveryDownloadBtn').onclick = downloadRecoverySnapshot;
  for (const [id, field] of [['traceQuality','trace_quality'],['controlReady','control_ready'],['roiArtifactClass','artifact_class'],['needsAction','needs_action'],['roiConfidence','confidence']]) {
    document.getElementById(id).onchange = e => {
      const roi = selectedRoi(); if(!roi) return;
      annotations.rois[roi.id] = stampAnnotation(Object.assign(roiAnn(roi.id), {[field]: e.target.value}));
      if(annotations.virtualRois[roi.id]) stampAnnotation(Object.assign(annotations.virtualRois[roi.id], {[field]: e.target.value}));
      recordAction(`roi_${field}`);
      queueSave();
      renderAll();
    };
  }
  document.getElementById('roiReasonTags').onchange = e => {
    const roi = selectedRoi(); if(!roi) return;
    const tags = normalizeIdList(e.target.value);
    annotations.rois[roi.id] = stampAnnotation(Object.assign(roiAnn(roi.id), {reason_tags: tags}));
    if(annotations.virtualRois[roi.id]) stampAnnotation(Object.assign(annotations.virtualRois[roi.id], {reason_tags: tags}));
    recordAction('roi_reason_tags');
    queueSave();
    renderAll();
  };
  document.getElementById('identityGroup').oninput = e => {
    const roi = selectedRoi(); if(!roi) return;
    annotations.rois[roi.id] = stampAnnotation(Object.assign(roiAnn(roi.id), {identity_group:e.target.value}));
    if(annotations.virtualRois[roi.id]) stampAnnotation(Object.assign(annotations.virtualRois[roi.id], {identity_group:e.target.value}));
    recordAction('roi_identity_group');
    queueSave();
  };
  for (const [id, field] of [['eventType','event_type'],['timingQuality','timing_quality'],['eventConfidence','confidence']]) {
    document.getElementById(id).onchange = e => {
      const roi = selectedRoi(); if(!roi || !selectedEventFrame) return;
      annotations.events[eventKey(roi.id, selectedEventFrame)] = stampAnnotation(Object.assign(eventAnn(roi.id, selectedEventFrame), {[field]: e.target.value}));
      recordAction(`event_${field}`);
      queueSave();
      renderAll();
    };
  }
  document.getElementById('eventReasonTags').onchange = e => {
    const roi = selectedRoi(); if(!roi || !selectedEventFrame) return;
    annotations.events[eventKey(roi.id, selectedEventFrame)] = stampAnnotation(Object.assign(eventAnn(roi.id, selectedEventFrame), {reason_tags: normalizeIdList(e.target.value)}));
    recordAction('event_reason_tags');
    queueSave();
    renderAll();
  };
  roiNotes.oninput = e => {
    const roi = selectedRoi(); if(!roi) return;
    annotations.rois[roi.id] = stampAnnotation(Object.assign(roiAnn(roi.id), {notes:e.target.value}));
    if(annotations.virtualRois[roi.id]) stampAnnotation(Object.assign(annotations.virtualRois[roi.id], {notes:e.target.value}));
    queueSave();
  };
  eventNotes.oninput = e => {
    const roi = selectedRoi(); if(!roi || !selectedEventFrame) return;
    annotations.events[eventKey(roi.id, selectedEventFrame)] = stampAnnotation(Object.assign(eventAnn(roi.id, selectedEventFrame), {notes:e.target.value}));
    queueSave();
  };
  document.getElementById('suggestionNotes').oninput = e => {
    const s = selectedSuggestion(); if(!s) return;
    annotations.suggestions[s.id] = stampAnnotation(Object.assign(suggestionAnn(s.id), {notes:e.target.value}));
    queueSave();
  };
  document.getElementById('suggestionConfidence').onchange = e => {
    const s = selectedSuggestion(); if(!s) return;
    annotations.suggestions[s.id] = stampAnnotation(Object.assign(suggestionAnn(s.id), {confidence:e.target.value}));
    recordAction('suggestion_confidence');
    queueSave();
    renderAll();
  };
  document.getElementById('suggestionReasonTags').onchange = e => {
    const s = selectedSuggestion(); if(!s) return;
    annotations.suggestions[s.id] = stampAnnotation(Object.assign(suggestionAnn(s.id), {reason_tags: normalizeIdList(e.target.value)}));
    recordAction('suggestion_reason_tags');
    queueSave();
    renderAll();
  };
  traceCanvas.addEventListener('pointerdown', e => {
    const point = traceCanvasPoint(e);
    const roi = selectedRoi();
    const ev = traceEventAtPoint(point, roi);
    if(ev) {
      selectTraceEvent(ev, roi);
      return;
    }
    traceView.dragging = true;
    traceCanvas.setPointerCapture?.(e.pointerId);
    setFrame(traceFrameFromX(point.x));
  });
  traceCanvas.addEventListener('pointermove', e => {
    if(!traceView.dragging) return;
    setFrame(traceFrameFromX(traceCanvasPoint(e).x));
  });
  traceCanvas.addEventListener('pointerup', e => {
    traceView.dragging = false;
    traceCanvas.releasePointerCapture?.(e.pointerId);
  });
  traceCanvas.addEventListener('pointercancel', () => { traceView.dragging = false; });
  traceCanvas.addEventListener('wheel', e => {
    e.preventDefault();
    const point = traceCanvasPoint(e);
    const bounds = traceBounds();
    const pointerFrame = traceFrameFromX(point.x);
    const plotW = Math.max(1, traceCanvas.width - 2 * TRACE_PAD);
    const ratio = Math.max(0, Math.min(1, (point.x - TRACE_PAD) / plotW));
    const factor = e.deltaY > 0 ? 1.2 : 0.82;
    const span = Math.max(1, (bounds.end - bounds.start) * factor);
    setTraceWindow(pointerFrame - span * ratio, pointerFrame + span * (1 - ratio));
    drawTrace();
  }, {passive:false});
  traceCanvas.addEventListener('dblclick', e => {
    e.preventDefault();
    resetTraceZoom();
  });
  if(eventTimelineCanvas) {
    eventTimelineCanvas.addEventListener('pointerdown', e => {
      const rect = eventTimelineCanvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * eventTimelineCanvas.width / rect.width;
      setFrame(timelineFrameFromX(x));
      eventTimelineCanvas.setPointerCapture?.(e.pointerId);
      eventTimelineCanvas.dataset.dragging = '1';
    });
    eventTimelineCanvas.addEventListener('pointermove', e => {
      if(eventTimelineCanvas.dataset.dragging !== '1') return;
      const rect = eventTimelineCanvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * eventTimelineCanvas.width / rect.width;
      setFrame(timelineFrameFromX(x));
    });
    eventTimelineCanvas.addEventListener('pointerup', e => {
      eventTimelineCanvas.dataset.dragging = '';
      eventTimelineCanvas.releasePointerCapture?.(e.pointerId);
    });
    eventTimelineCanvas.addEventListener('pointercancel', () => { eventTimelineCanvas.dataset.dragging = ''; });
  }
  overlay.addEventListener('pointerdown', e => {
    const mode = setting('manualRoiMode') || 'select';
    const editMode = setting('roiEditMode') || 'off';
    if(mode === 'select' && editMode !== 'off') {
      e.preventDefault();
      const p = overlayPointFromEvent(e);
      const editable = ensureEditableRoi(selectedRoi());
      if(!editable) {
        setSaveState('select an ROI mask before brush editing', 'bad');
        return;
      }
      pushRoiEditHistory(editable, editMode);
      roiEditState = {drawing:true, editedId:String(editable.id)};
      overlay.setPointerCapture?.(e.pointerId);
      applyRoiBrush(p, editable);
      return;
    }
    if(mode === 'select') return;
    e.preventDefault();
    const p = overlayPointFromEvent(e);
    manualRoiState = {drawing:true, start:p, points:[p], preview:null, suppressClick:true};
    overlay.setPointerCapture?.(e.pointerId);
    if(mode === 'center') {
      createManualRoi('manual_center', circlePoints(p.x, p.y, Number(setting('manualRoiRadius')) || 6), 'Manual center ROI');
      manualRoiState = {drawing:false, start:null, points:[], preview:null, suppressClick:true};
      setTimeout(() => { manualRoiState.suppressClick = false; }, 0);
    }
  });
  overlay.addEventListener('pointermove', e => {
    if(roiEditState.drawing) {
      e.preventDefault();
      applyRoiBrush(overlayPointFromEvent(e), annotations.virtualRois[roiEditState.editedId]);
      return;
    }
    const mode = setting('manualRoiMode') || 'select';
    if(mode === 'select' || !manualRoiState.drawing) return;
    const p = overlayPointFromEvent(e);
    if(mode === 'circle') {
      const dx = p.x - manualRoiState.start.x, dy = p.y - manualRoiState.start.y;
      manualRoiState.preview = {type:'circle', x:manualRoiState.start.x, y:manualRoiState.start.y, radius:Math.max(1, Math.sqrt(dx*dx + dy*dy))};
    } else if(mode === 'lasso') {
      const last = manualRoiState.points[manualRoiState.points.length - 1];
      if(!last || distance(last, p) >= 1.5) manualRoiState.points.push(p);
    }
    drawOverlay();
  });
  overlay.addEventListener('pointerup', e => {
    if(roiEditState.drawing) {
      e.preventDefault();
      roiEditState = {drawing:false, editedId:null};
      overlay.releasePointerCapture?.(e.pointerId);
      recordAction('roi_brush_edit');
      queueSave();
      return;
    }
    const mode = setting('manualRoiMode') || 'select';
    if(mode === 'select' || !manualRoiState.drawing) return;
    e.preventDefault();
    const p = overlayPointFromEvent(e);
    if(mode === 'circle') {
      const dx = p.x - manualRoiState.start.x, dy = p.y - manualRoiState.start.y;
      const radius = Math.max(1, Math.sqrt(dx*dx + dy*dy));
      createManualRoi('manual_circle', circlePoints(manualRoiState.start.x, manualRoiState.start.y, radius), 'Manual circle ROI');
    } else if(mode === 'lasso') {
      manualRoiState.points.push(p);
      createManualRoi('manual_lasso', lassoPoints(manualRoiState.points), 'Manual lasso ROI');
    }
    overlay.releasePointerCapture?.(e.pointerId);
    manualRoiState = {drawing:false, start:null, points:[], preview:null, suppressClick:true};
    setTimeout(() => { manualRoiState.suppressClick = false; }, 0);
  });
  overlay.addEventListener('pointercancel', () => {
    roiEditState = {drawing:false, editedId:null};
    manualRoiState = {drawing:false, start:null, points:[], preview:null, suppressClick:false};
    drawOverlay();
  });
  overlay.onclick = e => {
    if(manualRoiState.suppressClick || (setting('manualRoiMode') || 'select') !== 'select' || (setting('roiEditMode') || 'off') !== 'off') return;
    const rect = overlay.getBoundingClientRect();
    const x = (e.clientX - rect.left) * data.video.width / rect.width;
    const y = (e.clientY - rect.top) * data.video.height / rect.height;
    let best = null, bestD = Infinity, bestType = 'roi';
    for(const roi of visibleRois()){
      const dx = x - roi.centroidX, dy = y - roi.centroidY, d = dx*dx + dy*dy;
      if(d < bestD){ bestD = d; best = roi; bestType = 'roi'; }
    }
    if(document.getElementById('showSuggestions').checked) for(const s of visibleSuggestions()){
      const dx = x - s.centroidX, dy = y - s.centroidY, d = dx*dx + dy*dy;
      if(d < bestD){ bestD = d; best = s; bestType = 'suggestion'; }
    }
    if(bestType === 'suggestion') selectSuggestion(best.id);
    else if(best) selectRoi(best.id, e.shiftKey || e.ctrlKey || e.metaKey);
  };
  document.addEventListener('keydown', e => {
    if(e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
    if((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      undoLastAnnotationChange();
      return;
    }
    if(e.key === 'Escape') toggleShortcutHelp(false);
    else if(e.key === '?'){ e.preventDefault(); toggleShortcutHelp(); }
    else if(e.code === 'Space'){ e.preventDefault(); togglePlay(); }
    else if(e.key === 'ArrowRight') setFrame(currentFrame + 1);
    else if(e.key === 'ArrowLeft') setFrame(currentFrame - 1);
    else if(e.key === 'j') nextRoi(1);
    else if(e.key === 'k') nextRoi(-1);
    else if(e.key === 'N') nextEventQueue(1);
    else if(e.key === 'P') nextEventQueue(-1);
    else if(e.key === 'n') nextEvent(1);
    else if(e.key === 'p') nextEvent(-1);
    else if(e.key === '.') nextSuggestion(1);
    else if(e.key === ',') nextSuggestion(-1);
    else if(e.key === '0') resetTraceZoom();
    else if(e.key === 'v') nextActiveFrame(1);
    else if(e.key === 'V') nextActiveFrame(-1);
    else if(e.key === 'a') setRoiState('accept');
    else if(e.key === 'r') setRoiState('reject');
    else if(e.key === 'u') setRoiState('unsure');
    else if(e.key === 'e') setEventState('accept');
    else if(e.key === 'x') setEventState('reject');
    else if(e.key === 'f') viewerScroll.requestFullscreen?.();
    else if(e.key === 'M') setSuggestionStateAndNext('missed');
    else if(e.key === 'G') promoteSuggestionAndNext();
    else if(e.key === 'm') setSuggestionState('missed');
    else if(e.key === 'g') promoteSuggestion();
    else if(e.key === ']') {
      const tasks = guidedTasks();
      setSetting('guidedTaskIndex', Math.min(Math.max(0, tasks.length - 1), Number(setting('guidedTaskIndex') || 0) + 1));
      selectGuidedTask();
    }
    else if(e.key === '[') {
      setSetting('guidedTaskIndex', Math.max(0, Number(setting('guidedTaskIndex') || 0) - 1));
      selectGuidedTask();
    }
  });
  img.onload = () => { resizeOverlay(); drawCrop(); };
  window.onresize = resizeOverlay;
  window.addEventListener('hashchange', routePage);
}

function availabilityBadge(def){
  const value = def?.availability || 'implemented';
  const label = value === 'external_import' ? 'external' : value;
  return `<span class="stageStatus ${value === 'implemented' ? 'ok' : value === 'planned' ? 'warn' : 'off'}">${escapeHtml(label.replace(/_/g, ' '))}</span>`;
}

function pipelinePresetSummary(preset){
  const run = makePresetPipeline(preset.id);
  const realtime = pipelineRealtimeSummary(run);
  const ops = run.pipeline.map(stage => stageDef(stage)).filter(Boolean);
  const chips = ops.slice(0, 7).map(def => `<span>${escapeHtml(def.label)}</span>`).join('');
  return `
    <article class="presetCard">
      <div class="presetHeader">
        <h3>${escapeHtml(preset.label)}</h3>
        <span class="stageStatus ${realtime.warnings.length ? 'warn' : 'ok'}">${realtime.warnings.length ? 'review' : 'ready'}</span>
      </div>
      <p>${escapeHtml(preset.summary)}</p>
      <p class="hint">${escapeHtml(preset.best_for)}</p>
      <div class="archEvidence">${chips}${ops.length > 7 ? `<span>+${ops.length - 7} more</span>` : ''}</div>
      <div class="buttonRow">
        <button type="button" data-load-preset="${escapeHtml(preset.id)}">Use preset</button>
      </div>
    </article>`;
}

function renderArchitecturePresets(){
  const root = document.getElementById('architecturePresetGallery');
  if(!root) return;
  root.innerHTML = ARCHITECTURE_PRESETS.map(pipelinePresetSummary).join('');
  for(const btn of root.querySelectorAll('[data-load-preset]')) btn.onclick = () => {
    pipelineDraft = makePresetPipeline(btn.dataset.loadPreset);
    selectedPipelineStageId = pipelineDraft.pipeline[0]?.id || null;
    const select = document.getElementById('pipelinePresetSelect');
    if(select) select.value = btn.dataset.loadPreset;
    setArchitectureMode('build');
    renderPipelineBuilder();
  };
}

function renderComponentLibrary(){
  const root = document.getElementById('componentLibrary');
  if(!root) return;
  const groups = {};
  for(const def of STAGE_CATALOG) (groups[def.ui_group || def.type || 'stage'] = groups[def.ui_group || def.type || 'stage'] || []).push(def);
  root.innerHTML = Object.entries(groups).map(([group, defs]) => `
    <section class="componentGroup">
      <div class="componentGroupHeader">
        <h3>${escapeHtml(group.replace(/_/g, ' '))}</h3>
        <span class="hint">${defs.length} component${defs.length === 1 ? '' : 's'}</span>
      </div>
      <div class="componentGrid">
        ${defs.map(def => {
          const qc = (def.expected_qc_outputs || []).slice(0, 4).map(item => `<span>${escapeHtml(item)}</span>`).join('');
          const params = Object.keys(def.params || {}).slice(0, 4).map(name => `<span>${escapeHtml(name)}</span>`).join('');
          return `
          <article class="componentCard">
            <div class="componentTitle">
              <h4>${escapeHtml(def.label)}</h4>
              ${availabilityBadge(def)}
            </div>
            <p>${escapeHtml(def.description || 'Pipeline component.')}</p>
            <p class="hint">${escapeHtml(def.why_use_it || '')}</p>
            <div class="stageMeta">${realtimeBadges(def)}</div>
            <div class="artifactFlow"><i>${escapeHtml(def.input || 'input')}</i><strong>-></strong><i>${escapeHtml(def.output || 'output')}</i></div>
            <div class="miniChipRow">${params || '<span>no tunable params</span>'}</div>
            <div class="miniChipRow qcChips">${qc || '<span>QC pending</span>'}</div>
            <button type="button" data-add-component="${escapeHtml(def.op)}">Add to stack</button>
          </article>`;
        }).join('')}
      </div>
    </section>`).join('');
  for(const btn of root.querySelectorAll('[data-add-component]')) btn.onclick = () => {
    pipelineDraft.pipeline.push(makeStage(btn.dataset.addComponent, pipelineDraft.pipeline.length));
    selectedPipelineStageId = pipelineDraft.pipeline[pipelineDraft.pipeline.length - 1].id;
    setArchitectureMode('build');
    renderPipelineBuilder();
  };
}

function slugify(value){
  return String(value || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '') || `pipeline_${Date.now().toString(36)}`;
}

function savedPipelineTemplates(){
  data.architectureRuns = data.architectureRuns || {schema_version: 1, dataset_id: datasetId, runs: []};
  data.architectureRuns.saved_pipelines = Array.isArray(data.architectureRuns.saved_pipelines) ? data.architectureRuns.saved_pipelines : [];
  return data.architectureRuns.saved_pipelines;
}

function pipelineTemplateFromDraft(){
  return {
    id: slugify(pipelineDraft.template_id || pipelineDraft.run_id || pipelineDraft.label),
    label: pipelineDraft.label || 'Untitled architecture',
    description: pipelineDraft.description || '',
    dataset_id: datasetId,
    createdAt: pipelineDraft.createdAt || new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    source: 'architecture_lab',
    pipeline: JSON.parse(JSON.stringify(pipelineDraft.pipeline || [])),
    sweep: pipelineDraft.sweep ? JSON.parse(JSON.stringify(pipelineDraft.sweep)) : undefined,
    artifacts: JSON.parse(JSON.stringify(pipelineDraft.artifacts || {})),
    summary: JSON.parse(JSON.stringify(pipelineDraft.summary || {}))
  };
}

function normalizeTemplateForDraft(template){
  const draft = normalizePipelineDraft({
    schema_version: 1,
    run_id: `planned_${slugify(template.id || template.label)}_${Date.now().toString(36)}`,
    label: template.label || 'Untitled architecture',
    description: template.description || '',
    dataset_id: datasetId,
    pipeline: template.pipeline || [],
    sweep: template.sweep,
    artifacts: template.artifacts || {source_video: data.dataset?.paths?.raw_video || data.video?.name || '', intermediates: []},
    summary: template.summary || {roi_count: 0, event_count: 0, suggestion_count: 0, frame_count: data.video.frames},
    execution: {status: 'planned'}
  });
  draft.template_id = template.id || '';
  return draft;
}

async function persistArchitectureRuns(manifest, successText, fallbackName='architecture_runs.json'){
  data.architectureRuns = manifest;
  if(serverBacked){
    try {
      const res = await fetch('architecture_runs.json', {method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify(manifest, null, 2)});
      if(!res.ok) throw new Error(await res.text());
      setSaveState(successText, 'ok');
      return true;
    } catch (_) {
      downloadJson(fallbackName, manifest);
      setSaveState(`downloaded ${fallbackName}`, 'ok');
      return false;
    }
  }
  downloadJson(fallbackName, manifest);
  setSaveState(`downloaded ${fallbackName}`, 'ok');
  return false;
}

async function saveArchitectureTemplate(){
  const manifest = Object.assign({schema_version: 1, dataset_id: datasetId, runs: []}, data.architectureRuns || {});
  const template = pipelineTemplateFromDraft();
  const templates = savedPipelineTemplates().filter(item => item.id !== template.id);
  const previous = savedPipelineTemplates().find(item => item.id === template.id);
  if(previous?.createdAt) template.createdAt = previous.createdAt;
  manifest.saved_pipelines = [...templates, template].sort((a,b) => String(a.label || a.id).localeCompare(String(b.label || b.id)));
  await persistArchitectureRuns(manifest, `saved architecture ${template.label}`, `${datasetId}_architecture_library.json`);
  pipelineDraft.template_id = template.id;
  renderArchitectureLab();
  renderExperimentLab();
}

async function renameArchitectureTemplate(templateId){
  const template = savedPipelineTemplates().find(item => item.id === templateId);
  if(!template) return;
  const nextLabel = prompt('Architecture name', template.label || template.id);
  if(nextLabel === null) return;
  const nextDescription = prompt('Architecture description', template.description || '');
  if(nextDescription === null) return;
  const manifest = Object.assign({schema_version: 1, dataset_id: datasetId, runs: []}, data.architectureRuns || {});
  manifest.saved_pipelines = savedPipelineTemplates().map(item => item.id === templateId ? Object.assign({}, item, {
    label: nextLabel.trim() || item.label || item.id,
    description: nextDescription.trim(),
    updatedAt: new Date().toISOString()
  }) : item);
  await persistArchitectureRuns(manifest, 'renamed saved architecture', `${datasetId}_architecture_library.json`);
  renderArchitectureLab();
  renderExperimentLab();
}

async function deleteArchitectureTemplate(templateId){
  const template = savedPipelineTemplates().find(item => item.id === templateId);
  if(!template) return;
  if(!confirm(`Delete saved architecture "${template.label || template.id}"? Planned/generated runs will not be deleted.`)) return;
  const manifest = Object.assign({schema_version: 1, dataset_id: datasetId, runs: []}, data.architectureRuns || {});
  manifest.saved_pipelines = savedPipelineTemplates().filter(item => item.id !== templateId);
  if(experimentDraft.baseTemplateId === templateId) experimentDraft.baseTemplateId = '';
  await persistArchitectureRuns(manifest, 'deleted saved architecture', `${datasetId}_architecture_library.json`);
  renderArchitectureLab();
  renderExperimentLab();
}

function loadTemplateIntoBuilder(templateId){
  const template = savedPipelineTemplates().find(item => item.id === templateId);
  if(!template) return;
  pipelineDraft = normalizeTemplateForDraft(template);
  selectedPipelineStageId = pipelineDraft.pipeline?.[0]?.id || null;
  setArchitectureMode('build');
  renderPipelineBuilder();
}

async function useCurrentArchitectureInExperiment(){
  const template = pipelineTemplateFromDraft();
  await saveArchitectureTemplate();
  experimentDraft.baseTemplateId = template.id;
  location.hash = '#experiments';
  renderExperimentLab();
}

function renderArchitectureLibrary(){
  const templates = savedPipelineTemplates();
  if(!templates.length) return `
    <section class="archCard savedArchitectureLibrary">
      <div class="runCardHeader"><h3>Saved Architectures</h3><span class="runStatus">0 saved</span></div>
      <p class="hint">Save a named architecture from Build Pipeline to reuse it in Experiment Lab.</p>
    </section>`;
  const cards = templates.map(template => `
    <article class="savedArchitectureCard">
      <div class="runCardHeader">
        <h3>${escapeHtml(template.label || template.id)}</h3>
        <span class="runStatus">${escapeHtml((template.pipeline || []).length)} stages</span>
      </div>
      <p class="hint">${escapeHtml(template.description || template.id || '')}</p>
      <div class="archEvidence">${(template.pipeline || []).map(stage => `<span>${escapeHtml(stageDef(stage)?.label || stageOp(stage) || stage.id || 'stage')}</span>`).join('')}</div>
      <div class="buttonRow">
        <button type="button" data-load-template="${escapeHtml(template.id)}">Edit</button>
        <button type="button" data-template-experiment="${escapeHtml(template.id)}">Experiment</button>
        <button type="button" data-rename-template="${escapeHtml(template.id)}">Rename</button>
        <button type="button" data-delete-template="${escapeHtml(template.id)}">Delete</button>
      </div>
    </article>`).join('');
  return `
    <section class="archCard savedArchitectureLibrary">
      <div class="runCardHeader"><h3>Saved Architectures</h3><span class="runStatus">${templates.length} saved</span></div>
      <div class="savedArchitectureGrid">${cards}</div>
    </section>`;
}

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
    root.innerHTML = renderArchitectureLibrary() + '<p class="hint">No architecture runs are attached yet. Use tools/build_architecture_run.py to create architecture_runs.json.</p>';
    for(const btn of root.querySelectorAll('[data-load-template]')) btn.onclick = () => loadTemplateIntoBuilder(btn.dataset.loadTemplate);
    for(const btn of root.querySelectorAll('[data-template-experiment]')) btn.onclick = () => { experimentDraft.baseTemplateId = btn.dataset.templateExperiment; location.hash = '#experiments'; };
    for(const btn of root.querySelectorAll('[data-rename-template]')) btn.onclick = () => renameArchitectureTemplate(btn.dataset.renameTemplate);
    for(const btn of root.querySelectorAll('[data-delete-template]')) btn.onclick = () => deleteArchitectureTemplate(btn.dataset.deleteTemplate);
    return;
  }
  root.innerHTML = renderArchitectureLibrary() + renderParameterExperiments(runs);
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
    card.innerHTML = `
      <div class="runCardHeader"><h3>${run.label || run.run_id}</h3><span class="runStatus">${escapeHtml(status)}</span></div>
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
      <div class="buttonRow">
        <button type="button" data-activate-run="${escapeHtml(run.run_id)}">Use In Review/QC</button>
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

function runLabel(run){
  return run ? `${run.label || run.run_id}` : '';
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

function renderParameterExperiments(runs){
  const rows = runs.map(run => {
    const m = runExperimentMetrics(run);
    const label = experimentLabel(run.run_id);
    const labelButtons = ['looks best','too noisy','too strict','artifact heavy','needs review'].map(value =>
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
        <td><button type="button" data-activate-run="${escapeHtml(run.run_id)}">Use</button> <a href="#process">Process</a></td>
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
    return `<tr><td>${escapeHtml(run.label || run.run_id)}</td><td>${escapeHtml(run.run_id)}</td><td>${escapeHtml(changed)}</td><td>${escapeHtml(run.validation?.status || validation.status)}</td></tr>`;
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
    <section class="experimentHero">
      <div>
        <h2>Experiment Lab</h2>
        <p class="hint">Design parameter sweeps or named hand-picked sets from the current pipeline stack, then save them as planned architecture runs for local generation.</p>
      </div>
      <div class="buttonRow">
        <a class="textButton" href="#architecture">Edit stack</a>
        <button type="button" id="experimentSaveBtn">Save Plan</button>
        <button type="button" id="experimentDownloadBtn">Download Plan</button>
        <button type="button" id="experimentPreviewBtn">Generate First Preview</button>
      </div>
    </section>
    <div class="experimentGrid">
      <section class="archCard">
        <div class="runCardHeader"><h3>Base Pipeline</h3><span class="runStatus">${validation.status}</span></div>
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
      </section>
      <section class="archCard">
        <div class="runCardHeader"><h3>Experiment Mode</h3><span class="runStatus">${manifest.runs.length} planned</span></div>
        <label>Mode
          <select id="experimentModeSelect">
            <option value="sweep" ${experimentDraft.mode === 'sweep' ? 'selected' : ''}>Sweep axes</option>
            <option value="sets" ${experimentDraft.mode === 'sets' ? 'selected' : ''}>Named sets</option>
            <option value="optuna" ${experimentDraft.mode === 'optuna' ? 'selected' : ''}>Optuna plan</option>
          </select>
        </label>
        <p class="hint">Sweep mode uses stack axes. Set mode saves named variants. Optuna plan mode stores study/search-space metadata only; it does not run optimization in the browser.</p>
      </section>
      <section class="archCard">
        <div class="runCardHeader"><h3>Named Set Builder</h3><span class="runStatus">${experimentDraft.setRows.length} sets</span></div>
        <label>Parameter <select id="experimentSetParamSelect">${paramOptions}</select></label>
        <label>Value <input id="experimentSetValueInput" placeholder="new value"></label>
        <button type="button" id="experimentAddSetBtn" ${params.length ? '' : 'disabled'}>Add Set</button>
        <table class="smallTable"><tr><th>Label</th><th>Parameter</th><th>Value</th><th></th></tr>${setRows || '<tr><td colspan="4">No named sets yet.</td></tr>'}</table>
      </section>
      <section class="archCard">
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
        <table class="smallTable"><tr><th>Parameter</th><th>Type</th><th>Low</th><th>High</th><th></th></tr>${optunaRows || '<tr><td colspan="5">No Optuna search space yet.</td></tr>'}</table>
      </section>
    </div>
    <section class="archCard">
      <div class="runCardHeader"><h3>Planned Runs Preview</h3><span class="runStatus">${manifest.runs.length} run${manifest.runs.length === 1 ? '' : 's'}</span></div>
      <table class="smallTable compareTable">
        <tr><th>Label</th><th>Run ID</th><th>Changed parameters</th><th>Status</th></tr>
        ${previewRows || '<tr><td colspan="4">No planned runs.</td></tr>'}
      </table>
      ${manifest.runs.length > 24 ? `<p class="hint">Showing first 24 of ${manifest.runs.length} planned runs.</p>` : ''}
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
}

const FALLBACK_STAGE_CATALOG = [
  {type:'temporal_smoothing', op:'temporal_highpass_gaussian', label:'Temporal high-pass Gaussian', input:'raw_video', output:'highpass_video', params:{sigma_frames:{type:'number', min:1, max:20, step:1, default:6}}},
  {type:'spatial_smoothing', op:'spatial_gaussian', label:'Spatial Gaussian smoothing', input:'highpass_video', output:'smoothed_video', params:{sigma_px:{type:'number', min:0, max:4, step:0.1, default:0.8}}},
  {type:'background_correction', op:'local_background_ring', label:'Local background ring', input:'roi_candidates', output:'roi_traces', params:{outer_radius_px:{type:'number', min:4, max:40, step:1, default:15}, neuropil_weight:{type:'number', min:0, max:1.5, step:0.05, default:0.7}}},
  {type:'motion_correction', op:'rigid_shift_estimate', label:'Rigid drift estimate', input:'raw_video', output:'registered_video', params:{max_shift_px:{type:'number', min:1, max:12, step:1, default:4}}},
  {type:'filtering', op:'robust_positive_local_z', label:'Robust positive local-z', input:'highpass_video', output:'z_stack', params:{local_radius_px:{type:'number', min:3, max:31, step:2, default:11}, epsilon:{type:'number', min:0, max:10, step:0.5, default:1}}},
  {type:'filtering', op:'gamma_cfar', label:'Gamma CFAR', input:'smoothed_video', output:'candidate_mask', params:{pfa:{type:'number', min:0.000001, max:0.1, step:0.0001, default:0.001}, guard_px:{type:'number', min:0, max:12, step:1, default:2}}},
  {type:'trace_extraction', op:'component_filter', label:'Component extraction', input:'z_stack', output:'roi_candidates', params:{seed_z:{type:'number', min:0.5, max:8, step:0.1, default:2.0}, grow_z:{type:'number', min:0.2, max:5, step:0.1, default:1.1}, min_area_px:{type:'number', min:1, max:100, step:1, default:8}, max_area_px:{type:'number', min:20, max:800, step:10, default:260}}},
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

function frameBudgetMs(){
  const rate = datasetFrameRateHz();
  return Number.isFinite(rate) && rate > 0 ? 1000 / rate : null;
}

function realtimeBadges(def){
  const rt = def?.real_time_profile || {};
  const badges = [];
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
    artifact_suppression: ['temporal_highpass_gaussian', 'temporal_hampel', 'robust_positive_local_z', 'background_heterogeneity_map', 'saturation_blob_map', 'component_filter', 'artifact_classifier_v1', 'active_learning_ranker'],
    high_recall_discovery: ['temporal_highpass_gaussian', 'robust_positive_local_z', 'component_filter', 'local_background_ring', 'robust_kalman_positive_innovation', 'local_temporal_correlation', 'event_triggered_footprint', 'ensemble_union', 'heuristic_priority_v1'],
    motion_aware: ['temporal_highpass_gaussian', 'rigid_shift_estimate', 'motion_sensitivity_map', 'robust_positive_local_z', 'component_filter', 'local_background_ring', 'robust_kalman_positive_innovation', 'heuristic_priority_v1'],
    pmd_import: ['pmd_denoised_video_import', 'robust_positive_local_z', 'component_filter', 'robust_kalman_positive_innovation', 'heuristic_priority_v1'],
    suite2p_import: ['suite2p_import', 'heuristic_priority_v1'],
    oasis_import: ['temporal_highpass_gaussian', 'robust_positive_local_z', 'component_filter', 'local_background_ring', 'oasis_deconvolution_import', 'heuristic_priority_v1']
  };
  const ops = presetOps[name] || presetOps.current_review_pipeline;
  const pipeline = ops.map((op, i) => makeStage(op, i));
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
  if(!(run.pipeline || []).some(s => s.enabled !== false && s.type === 'event_model')) warnings.push('No event model is enabled; Architecture Lab will compare ROI candidates only.');
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
      <p class="hint">Compare generated review frames side-by-side at the same frame. This viewer is read-only; use the explicit buttons below to switch the main Review/QC context.</p>
      <div class="buttonRow">
        <button type="button" id="loadReviewComparisonBtn" ${runGenerated(a) && runGenerated(b) ? '' : 'disabled'}>Load A/B Review</button>
        <button type="button" id="prevReviewDiffBtn">Prev Difference</button>
        <button type="button" id="nextReviewDiffBtn">Next Difference</button>
        <button type="button" id="useRunAReviewBtn">Use A In Review/QC</button>
        <button type="button" id="useRunBReviewBtn">Use B In Review/QC</button>
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
function findIntermediateForStage(stage, run){
  const op = stageOp(stage);
  const id = stage.id || '';
  return intermediateArtifactsForRun(run).find(item =>
    item.step_id === id || item.stage_id === op || item.stage === id || item.id === id || item.id === op
  );
}
function qcTileImageHtml(tile){
  if(tile.frame_pattern) return `<img class="qcStageMedia" data-frame-pattern="${escapeHtml(tile.frame_pattern)}" data-missing-text="${escapeHtml(tile.label)} frame did not load" onerror="handleQcImageError(this)" alt="${escapeHtml(tile.label)}">`;
  if(tile.file || tile.path) return `<img class="qcStageMedia" src="${escapeHtml(artifactUrl(tile.file || tile.path))}" data-missing-text="${escapeHtml(tile.label)} artifact did not load" onerror="handleQcImageError(this)" alt="${escapeHtml(tile.label)}">`;
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
    tiles.push({
      id: stage.id || stageOp(stage),
      label: artifact?.label || def?.label || stage.label || stage.name || stageOp(stage),
      stage_id: stageOp(stage),
      status: artifact ? 'available' : 'missing',
      frame_pattern: artifact?.frame_pattern || artifact?.framePattern,
      file: artifact?.file,
      path: artifact?.path,
      description: artifact?.description || def?.description || 'Pipeline stage output.',
      missing: artifact ? '' : 'Intermediate frames not attached yet.'
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
  const size = document.getElementById('qcTileSize')?.value || 'medium';
  const missingOnly = Boolean(document.getElementById('qcMissingOnly')?.checked);
  const tiles = qcStageTiles(run).filter(tile => !missingOnly || tile.status === 'missing');
  root.className = `qcStageGrid ${escapeHtml(size)}`;
  root.innerHTML = tiles.map(tile => `
    <article class="qcStageTile ${tile.status === 'missing' ? 'missing' : ''}">
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
  if(qcLabel) qcLabel.textContent = `${currentFrame} / ${data.video.frames}`;
  for(const imgEl of document.querySelectorAll('[data-frame-pattern]')){
    imgEl.src = framePatternPath(imgEl.dataset.framePattern, currentFrame);
  }
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
  if(tileSize) tileSize.onchange = () => renderQcStageGrid(selectedQcRun());
  if(missingOnly) missingOnly.onchange = () => renderQcStageGrid(selectedQcRun());
  const prev = document.getElementById('qcPrevFrameBtn');
  const next = document.getElementById('qcNextFrameBtn');
  const play = document.getElementById('qcPlayBtn');
  if(prev) prev.onclick = () => setFrame(currentFrame - 1);
  if(next) next.onclick = () => setFrame(currentFrame + 1);
  if(play) play.onclick = toggleQcPlay;
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

function renderDatasetQc(){
  const root = document.getElementById('datasetQc');
  if(!root) return;
  const runs = data.architectureRuns?.runs || [];
  const run = selectedQcRun();
  const mapsForRun = availableEvidenceMapsForRun(run);
  if(mapsForRun.length && !mapsForRun.some(m => (m.id || m.label) === setting('qcEvidenceMap'))) annotations.settings.qcEvidenceMap = mapsForRun[0].id || mapsForRun[0].label || '';
  const areas = data.rois.map(r => Number(r.area));
  const diamPx = areas.map(a => 2 * Math.sqrt(a / Math.PI));
  const pixelSize = Number(data.dataset?.pixel_size_microns);
  const diamUm = Number.isFinite(pixelSize) ? diamPx.map(v => v * pixelSize) : [];
  const noise = data.rois.map(r => Number(r.noiseSigma));
  const eventCounts = data.rois.map(r => eventsForRoi(r).length);
  const peakScores = data.rois.map(r => Number(r.peakScore));
  const priorityScores = data.rois.map(r => Number(r.priorityScore));
  const suggestions = data.discovery?.suggestions || [];
  const artifactCueCount = suggestions.filter(s => s.artifactCue && s.artifactCue !== 'none').length;
  const driftMax = Number(data.qc?.driftStats?.maxMagnitudePx);
  const satMax = Number(data.qc?.saturationStats?.maxFraction);
  const warnings = [];
  if(data.rois.length && quantile(diamPx, 0.5) < 8) warnings.push('Median ROI footprint is small in pixels; the detector may be capturing active cores or fragments rather than full somata.');
  if(Number.isFinite(pixelSize) && quantile(diamUm, 0.5) < 5) warnings.push('Median equivalent ROI diameter is below 5 microns with the configured pixel size.');
  if(suggestions.length > data.rois.length) warnings.push('Discovery suggestions outnumber current ROIs; review missed-neuron coverage before tightening thresholds.');
  if(artifactCueCount > suggestions.length * 0.25) warnings.push('Many discovery suggestions have artifact cues; inspect evidence maps for vessels, borders, or bright static structures.');
  if(Number.isFinite(driftMax) && driftMax >= 2) warnings.push('Estimated rigid drift exceeds 2 px; compare raw candidates against motion-sensitive evidence before accepting weak traces.');
  if(Number.isFinite(satMax) && satMax > 0.001) warnings.push('Saturation-like bright pixels appear in the frame stack; inspect raw max and artifact-risk candidates.');
  if(!Number.isFinite(pixelSize)) warnings.push('Pixel size is not set in the dataset manifest, so physical-size QC is disabled.');
  const qcWarnings = warnings.map(w => `<div class="qcWarning">${w}</div>`).join('') || '<div class="qcWarning">No QC warnings from the current lightweight checks.</div>';
  const maps = (data.discovery?.evidenceMaps || []).map(m => `
    <div class="qcMap">
      <img src="${m.file}" alt="${m.label}">
      <p class="hint">${m.label}</p>
    </div>`).join('');
  const runOptions = runs.map(r => `<option value="${escapeHtml(r.run_id)}" ${run?.run_id === r.run_id ? 'selected' : ''}>${escapeHtml(runLabel(r))}</option>`).join('');
  const evidenceOptions = mapsForRun.map(m => `<option value="${escapeHtml(m.id || m.label)}" ${(setting('qcEvidenceMap') || '') === (m.id || m.label) ? 'selected' : ''}>${escapeHtml(m.label || m.id || 'evidence map')}</option>`).join('');
  root.innerHTML = `
    <div class="qcWorkbench">
      <section class="qcViewerPanel">
        <div class="toolbar">
          <button id="qcPlayBtn">Play</button>
          <button id="qcPrevFrameBtn">Prev</button>
          <button id="qcNextFrameBtn">Next</button>
          <label>Frame <input id="qcFrameSlider" type="range" min="1" max="${data.video.frames}" value="${currentFrame}"></label>
          <b id="qcFrameLabel">${currentFrame} / ${data.video.frames}</b>
          <label>Tile size
            <select id="qcTileSize">
              <option value="medium">Medium</option>
              <option value="large">Large</option>
              <option value="compact">Compact</option>
            </select>
          </label>
          <label><input id="qcMissingOnly" type="checkbox"> missing outputs only</label>
        </div>
        <div id="qcStageGrid" class="qcStageGrid medium"></div>
      </section>
      <section class="qcPipelinePanel">
        <div class="componentGroupHeader">
          <h3>Pipeline Context</h3>
          <label>Run <select id="qcRunSelect">${runOptions}</select></label>
        </div>
        <p class="hint">Process Lab follows the active Architecture Lab run, so raw frames, intermediate outputs, and warnings stay in pipeline order.</p>
        <div id="qcPipelineTimeline"></div>
      </section>
    </div>
    ${processInsightPanel(run)}
    <div class="metricGrid">
      <div class="metric"><b>${data.video.width} x ${data.video.height}</b><span>frame size</span></div>
      <div class="metric"><b>${data.video.frames}</b><span>frames</span></div>
      <div class="metric"><b>${data.rois.length}</b><span>candidate ROIs</span></div>
      <div class="metric"><b>${suggestions.length}</b><span>discovery suggestions</span></div>
      <div class="metric"><b>${fmt(quantile(areas, 0.5), 0)}</b><span>median ROI area px</span></div>
      <div class="metric"><b>${fmt(quantile(diamPx, 0.5), 1)}</b><span>median ROI diameter px</span></div>
      <div class="metric"><b>${diamUm.length ? fmt(quantile(diamUm, 0.5), 1) : 'n/a'}</b><span>median ROI diameter microns</span></div>
      <div class="metric"><b>${fmt(quantile(noise, 0.5), 4)}</b><span>median trace noise sigma</span></div>
      <div class="metric"><b>${fmt(quantile(eventCounts, 0.5), 0)}</b><span>median events per ROI</span></div>
      <div class="metric"><b>${fmt(quantile(peakScores, 0.5), 2)}</b><span>median peak score</span></div>
      <div class="metric"><b>${fmt(quantile(priorityScores, 0.5), 2)}</b><span>median priority score</span></div>
      <div class="metric"><b>${Number.isFinite(driftMax) ? fmt(driftMax, 2) : 'n/a'}</b><span>max drift px</span></div>
      <div class="metric"><b>${Number.isFinite(satMax) ? fmt(100 * satMax, 3) + '%' : 'n/a'}</b><span>max saturated fraction</span></div>
    </div>
    <div class="qcWarnings">${qcWarnings}</div>
    <div class="auditSplit">
      <div class="archCard">${auditRows('ROI area px', {min: Math.min(...areas), median: quantile(areas, 0.5), max: Math.max(...areas)})}</div>
      <div class="archCard">${auditRows('Events per ROI', {zero: eventCounts.filter(v => v === 0).length, one_to_three: eventCounts.filter(v => v >= 1 && v <= 3).length, four_plus: eventCounts.filter(v => v >= 4).length})}</div>
      <div class="archCard">${auditRows('Discovery artifact cues', {with_cue: artifactCueCount, no_cue: Math.max(0, suggestions.length - artifactCueCount)})}</div>
    </div>
    <h2>Evidence Maps</h2>
    <div class="qcMapGrid">${maps}</div>`;
  wireDatasetQcControls();
  renderQcStageTimeline(run);
  renderQcStageGrid(run);
}

function reviewReportMarkdown(){
  const s = annotationSummary();
  const batch = nextAnnotationBatch();
  const lines = [
    `# Neuron Workbench Review Report: ${datasetId}`,
    '',
    '## Review Status',
    '',
    `- Candidate ROIs: ${s.roi_count}`,
    `- Candidate events: ${s.event_count}`,
    `- Discovery suggestions: ${s.suggestion_count}`,
    `- Reviewed ROIs: ${s.review_progress.reviewed_rois} (${Math.round(100 * s.review_progress.roi_review_fraction)}%)`,
    `- Reviewed events: ${s.review_progress.reviewed_events} (${Math.round(100 * s.review_progress.event_review_fraction)}%)`,
    `- Tuning-ready: ${s.review_progress.tuning_ready ? 'yes' : 'no'}`,
    '',
    '## Accepted Outputs',
    '',
    `- Accepted ROIs: ${s.roi_states.accepted}`,
    `- Accepted events: ${s.event_states.accepted}`,
    `- Control-ready yes/maybe: ${s.control_ready.yes} / ${s.control_ready.maybe}`,
    '',
    '## Reviewer Contributions',
    ''
  ];
  for(const [reviewer, count] of Object.entries(s.reviewer_counts || {}).sort((a,b) => b[1] - a[1] || a[0].localeCompare(b[0]))) {
    lines.push(`- ${reviewer}: ${count} reviewed labels`);
  }
  if(!Object.keys(s.reviewer_counts || {}).length) lines.push('- No reviewer-stamped labels yet.');
  const missingReviewerLabels = Object.values(s.reviewer_missing || {}).reduce((a,b) => a + b, 0);
  if(missingReviewerLabels) lines.push(`- Missing reviewer IDs: ${missingReviewerLabels} reviewed labels`);
  for(const [group, count] of Object.entries(s.reviewer_missing || {})) {
    if(count) lines.push(`  - ${group}: ${count}`);
  }
  lines.push(
    '',
    '## Recommended Next Review',
    ''
  );
  for(const roi of batch.rois.slice(0, 5)) lines.push(`- ROI ${roi.roi_id}: score ${fmt(roi.score, 2)}, ${(roi.reasons || []).join(', ')}`);
  lines.push('', '## Recommendations', '');
  if(!s.review_progress.tuning_ready) lines.push('- Complete the first guided annotation target before treating parameter comparisons as tuning evidence.');
  if(Object.values(s.reviewer_missing || {}).reduce((a,b) => a + b, 0)) lines.push('- Backfill missing reviewer IDs before using inter-rater comparison outputs for adjudication.');
  if(s.suggestion_states.unlabeled) lines.push('- Audit missed-neuron suggestions to estimate recall gaps.');
  if(s.roi_states.accepted && !s.control_ready.yes) lines.push('- Mark trace quality and control readiness for accepted ROIs before inverse-dynamics export.');
  if(s.review_progress.tuning_ready) lines.push('- Generate a review sweep pack and compare candidate stability across presets.');
  return lines.join('\n') + '\n';
}

function renderReviewReport(){
  const root = document.getElementById('reportPageBody');
  if(!root) return;
  const s = annotationSummary();
  const markdown = reviewReportMarkdown();
  const audit = reviewerProvenanceAudit();
  root.innerHTML = `
    <div class="reportHero">
      <div>
        <h2>Review Summary</h2>
        <p class="hint">A shareable snapshot of annotation progress, accepted outputs, and recommended next work.</p>
      </div>
      <div class="buttonRow">
        <button id="downloadReportBtn">Download Markdown</button>
        <button id="downloadProvenanceAuditBtn">Download Provenance Audit</button>
      </div>
    </div>
    <div class="metricGrid">
      <div class="metric"><b>${s.roi_states.accepted}</b><span>accepted ROIs</span></div>
      <div class="metric"><b>${s.event_states.accepted}</b><span>accepted events</span></div>
      <div class="metric"><b>${s.control_ready.yes + s.control_ready.maybe}</b><span>control-ready yes/maybe</span></div>
      <div class="metric"><b>${s.review_progress.tuning_ready ? 'yes' : 'no'}</b><span>tuning ready</span></div>
      <div class="metric"><b>${Object.values(s.reviewer_missing || {}).reduce((a,b) => a + b, 0)}</b><span>labels missing reviewer</span></div>
      <div class="metric"><b>${Math.round(100 * audit.coverage_fraction)}%</b><span>reviewer coverage</span></div>
    </div>
    <section class="archCard">${auditRows('Reviewer contributions', Object.keys(s.reviewer_counts || {}).length ? s.reviewer_counts : {unassigned: 0})}</section>
    <section class="archCard">${auditRows('Missing reviewer IDs', s.reviewer_missing || {none: 0})}</section>
    <pre class="reportPreview">${escapeHtml(markdown)}</pre>`;
  document.getElementById('downloadReportBtn').onclick = () => {
    const blob = new Blob([markdown], {type:'text/markdown'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${datasetId}_review_report.md`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  document.getElementById('downloadProvenanceAuditBtn').onclick = exportReviewerProvenanceAudit;
}

function routePage(){
  const hash = (location.hash || '#architecture').replace(/^#\/?/, '');
  const page = hash === 'architecture' || hash === 'architecture-lab' ? 'architecture' : hash === 'experiments' || hash === 'experiment-lab' ? 'experiments' : hash === 'metrics' || hash === 'audit' ? 'metrics' : hash === 'process' || hash === 'process-lab' || hash === 'qc' || hash === 'dataset-qc' ? 'qc' : hash === 'report' ? 'report' : 'review';
  for(const id of ['reviewTab','reviewTabArch','reviewTabExperiments','reviewTabMetrics','reviewTabQc','reviewTabReport']) document.getElementById(id)?.classList.toggle('active', page === 'review');
  for(const id of ['architectureTab','architectureTabArch','architectureTabExperiments','architectureTabMetrics','architectureTabQc','architectureTabReport']) document.getElementById(id)?.classList.toggle('active', page === 'architecture');
  for(const id of ['experimentsTab','experimentsTabArch','experimentsTabExperiments','experimentsTabMetrics','experimentsTabQc','experimentsTabReport']) document.getElementById(id)?.classList.toggle('active', page === 'experiments');
  for(const id of ['metricsTab','metricsTabArch','metricsTabExperiments','metricsTabMetrics','metricsTabQc','metricsTabReport']) document.getElementById(id)?.classList.toggle('active', page === 'metrics');
  for(const id of ['qcTab','qcTabArch','qcTabExperiments','qcTabMetrics','qcTabQc','qcTabReport']) document.getElementById(id)?.classList.toggle('active', page === 'qc');
  for(const id of ['reportTab','reportTabArch','reportTabExperiments','reportTabMetrics','reportTabQc','reportTabReport']) document.getElementById(id)?.classList.toggle('active', page === 'report');
  document.getElementById('architecturePage').classList.toggle('hidden', page !== 'architecture');
  document.getElementById('experimentsPage').classList.toggle('hidden', page !== 'experiments');
  document.getElementById('metricsPage').classList.toggle('hidden', page !== 'metrics');
  document.getElementById('qcPage').classList.toggle('hidden', page !== 'qc');
  document.getElementById('reportPage').classList.toggle('hidden', page !== 'report');
  appRoot.classList.toggle('arch-mode', page === 'architecture');
  appRoot.classList.toggle('lab-mode', page === 'metrics' || page === 'report' || page === 'experiments');
  appRoot.classList.toggle('qc-mode', page === 'qc');
  if(page === 'architecture') renderArchitectureLab();
  else if(page === 'experiments') renderExperimentLab();
  else if(page === 'metrics') renderMetricsAudit();
  else if(page === 'qc') renderDatasetQc();
  else if(page === 'report') renderReviewReport();
  else resizeOverlay();
  appRoot.classList.remove('booting');
  if(!routePage.lastPage || routePage.lastPage !== page) {
    window.requestAnimationFrame(() => window.scrollTo({top: 0, left: 0, behavior: 'auto'}));
  }
  routePage.lastPage = page;
}

async function boot(){
  populateEvidenceSelect();
  await loadAnnotations();
  initControls();
  renderParams();
  const first = visibleRois()[0] || data.rois[0];
  selectedId = first?.id || null;
  selectedRoiIds = new Set(selectedId ? [String(selectedId)] : []);
  if(selectedId) {
    selectedEventFrame = eventsForRoi(selectedRoi())[0]?.frame || null;
    roiNotes.value = roiAnn(selectedId).notes || '';
    eventNotes.value = selectedEventFrame ? eventAnn(selectedId, selectedEventFrame).notes || '' : '';
  }
  if(selectedSuggestionId) {
    document.getElementById('suggestionNotes').value = suggestionAnn(selectedSuggestionId).notes || '';
    document.getElementById('artifactClass').value = suggestionAnn(selectedSuggestionId).artifact_class || suggestionAnn(selectedSuggestionId).artifactClass || '';
  }
  renderRunSyncControls();
  loadGenerationEnvironment();
  setFrame(1);
  routePage();
  renderAll();
}
boot();
