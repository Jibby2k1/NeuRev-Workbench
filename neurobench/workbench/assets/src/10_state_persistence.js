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
    uiMode: 'expert'
  },
  validate_neurons: {
    label: 'Validate neurons',
    queue: 'annotationBatch',
    discoveryQueue: 'all',
    overlayPreset: 'validate',
    roiFocusMode: 'neighbors',
    reviewMode: 'guided',
    showEvidence: false,
    showSuggestions: true,
    showLabels: true,
    showEvents: true,
    selectedOverlayMode: 'outline',
    uiMode: 'guided'
  },
  find_missed_neurons: {
    label: 'Find missed neurons',
    queue: 'all',
    discoveryQueue: 'unlabeled',
    overlayPreset: 'discovery',
    roiFocusMode: 'all',
    reviewMode: 'guided',
    showEvidence: true,
    showSuggestions: true,
    showLabels: false,
    showEvents: true,
    selectedOverlayMode: 'outline',
    uiMode: 'guided'
  },
  clean_artifacts: {
    label: 'Clean artifacts',
    queue: 'artifactLike',
    discoveryQueue: 'artifactSuspects',
    overlayPreset: 'dense',
    roiFocusMode: 'all',
    reviewMode: 'guided',
    showEvidence: false,
    showSuggestions: true,
    showLabels: true,
    showEvents: true,
    selectedOverlayMode: 'soft',
    uiMode: 'guided'
  },
  tune_parameters: {
    label: 'Tune parameters',
    queue: 'annotationBatch',
    discoveryQueue: 'all',
    overlayPreset: 'validate',
    roiFocusMode: 'all',
    reviewMode: 'guided',
    showEvidence: false,
    showSuggestions: true,
    showLabels: true,
    showEvents: true,
    selectedOverlayMode: 'event',
    uiMode: 'guided'
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
const reviewRoisFileCache = new Map();
const traceShardCache = new Map();
const stencilGapReportCache = new Map();
const sweepEvidenceReportCache = new Map();
let traceView = {start: 1, end: Math.max(1, Number(data.video?.frames) || 1), dragging: false};
let manualRoiState = {drawing:false, start:null, points:[], preview:null, suppressClick:false};
let roiEditState = {drawing:false, editedId:null};
let selectedOverlayFlashUntil = 0;
let selectedOverlayFlashTimer = null;
const REVIEW_OVERLAP_COLORS = ['#ff4646', '#50dc64', '#4ba0ff', '#ffc83c', '#c084fc', '#22d3ee'];
let stencilState = {image:null, imageName:'', points:[], initialized:false, wired:false};
let overlapState = {image:null, imageName:'', wired:false, loading:false, runs:[], selected:null};
let reviewFocusBox = null;

function architectureRuns(){ return data.architectureRuns?.runs || []; }
function runHasCandidateRois(run){
  if(!run) return false;
  const artifactRois = run.artifacts?.review_rois;
  if(Array.isArray(artifactRois) && artifactRois.length) return true;
  if(run.artifacts?.review_rois_summary_file || run.artifacts?.review_rois_file) return true;
  return Number(run.summary?.roi_count || 0) > 0;
}
function preferredReviewRun(){
  const runs = architectureRuns();
  if(!runs.length) return null;
  const embeddedHasRois = Array.isArray(data.rois) && data.rois.length > 0;
  if(embeddedHasRois) return runs[0];
  return runs.find(runHasCandidateRois) || runs[0];
}
function isExternalTestDataset(){
  return datasetId === 'external_test'
    || data.dataset?.dataset_id === 'external_test'
    || String(data.video?.name || '').includes('zebrafish_test')
    || String(data.dataset?.raw_video || data.dataset?.paths?.raw_video || '').includes('external_test')
    || architectureRuns().some(run => String(run.run_id || '').includes('gamma_cfar_cascade_grid_high_recall_v1'))
    || architectureRuns().some(run => String(run.run_id || '').includes('green_excess_single_cfar_v1'));
}
function defaultDataCompareRunId(){
  if(isExternalTestDataset()) {
    const greenRun = architectureRuns().find(run => String(run.run_id || '').startsWith('green_excess_single_cfar_v1__sweep_') && runHasCandidateRois(run));
    if(greenRun) return greenRun.run_id;
    if(runById('gamma_cfar_cascade_grid_high_recall_v1__sweep_009')) return 'gamma_cfar_cascade_grid_high_recall_v1__sweep_009';
  }
  return baselineRunId();
}
function defaultDataComparePreset(runId=defaultDataCompareRunId()){
  return isExternalTestDataset() && (String(runId || '').startsWith('green_excess_single_cfar_v1__sweep_') || runId === 'gamma_cfar_cascade_grid_high_recall_v1__sweep_009') ? 'focused_diagnostic' : 'raw_artifact';
}
function baselineRunId(){ return preferredReviewRun()?.run_id || 'current_review_pipeline'; }

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
      roiLabelMode: 'all',
      selectedOverlayMode: 'outline',
      selectedFillOpacity: 0.10,
      selectedOutlineWidth: 2.5,
      roiFocusMode: 'all',
      neighborRadiusPx: 36,
      uiMode: 'guided',
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
      showStencilOverlay: true,
      showPotentialRois: true,
      showAnnotatedNeuronRois: true,
      showAnnotatedNonNeuronRois: true,
      overlayScope: 'all',
      minArea: 0,
      minEvents: 0,
      reviewMode: 'explore',
      guidedTaskIndex: 0,
      targetRois: 30,
      targetEvents: 30,
      targetSuggestions: 15,
      activeRunId: defaultDataCompareRunId(),
      reviewCompare: {
        enabled: false,
        runAId: defaultDataCompareRunId(),
        runBId: architectureRuns()[1]?.run_id || defaultDataCompareRunId()
      },
      qcRunId: defaultDataCompareRunId(),
      experimentLabels: {},
      qcTileSize: 'medium',
      qcEvidenceMap: data.discovery?.evidenceMaps?.[0]?.id || '',
      dataComparePreset: defaultDataComparePreset(),
      dataCompareMode: 'side_by_side',
      dataCompareOpacity: 0.55,
      dataCompareShowInactiveRois: String(defaultDataCompareRunId() || '').startsWith('green_excess_single_cfar_v1__sweep_') || !isExternalTestDataset(),
      dataCompareDiagnosticFrame: isExternalTestDataset() ? 132 : 1
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

function activeRunHasNoRenderableRois(run=activeRun()){
  const embeddedEmpty = !Array.isArray(data.rois) || data.rois.length === 0;
  return embeddedEmpty && !runHasCandidateRois(run);
}

function booleanSetting(name, fallback=true){
  const value = setting(name);
  return value === undefined || value === null ? fallback : Boolean(value);
}

function roiAnnotationClass(roi){
  const ann = roiAnn(roi.id);
  const cellState = String(ann.cell_state || '').toLowerCase();
  if(ann.state === 'accept' || cellState === 'accepted') return 'annotated_neuron';
  if(ann.state === 'reject' || cellState === 'rejected') return 'annotated_non_neuron';
  return 'potential';
}

function repairEmptyActiveRunSelection(){
  const run = activeRun();
  if(!activeRunHasNoRenderableRois(run)) return false;
  const preferred = preferredReviewRun();
  if(!preferred || preferred.run_id === run?.run_id) return false;
  annotations.settings.activeRunId = preferred.run_id;
  annotations.settings.qcRunId = preferred.run_id;
  materializeRunAnnotations(preferred.run_id);
  return true;
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

function preferredRunPreviewArtifact(run){
  const artifacts = intermediateArtifactsForRun(run);
  if(!artifacts.length) return null;
  const preferredIds = ['cfar_large_ref', 'candidate_mask', 'components', 'score', 'smooth', 'highpass'];
  for(const id of preferredIds){
    const match = artifacts.find(item => item.id === id || item.step_id === id || item.stage === id);
    if(match?.frame_pattern || match?.framePattern) return match;
  }
  return artifacts.find(item => item.frame_pattern || item.framePattern) || null;
}
function reviewPreviewArtifact(){
  const run = activeRun();
  if(!run || runGenerated(run) || !runHasIntermediates(run)) return null;
  return preferredRunPreviewArtifact(run);
}
function withMediaCacheKey(url, key){
  if(!url || !key) return url || '';
  const sep = String(url).includes('?') ? '&' : '?';
  return `${url}${sep}nb=${encodeURIComponent(key)}`;
}
function framePath(frame){
  return framePatternPath(data.video.framePattern, frame);
}
function virtualRoisArray(){ return Object.values(annotations.virtualRois || {}).filter(v => v && v.points?.length); }
function activeRunReviewRois(){
  const run = activeRun();
  if(!run || runGenerated(run)) return null;
  const rois = run.artifacts?.review_rois;
  return Array.isArray(rois) && rois.length ? rois : null;
}
function reviewRoisFileUrl(run){
  if(!run) return '';
  return artifactUrl(run.artifacts?.review_rois_summary_file) || artifactUrl(run.artifacts?.review_rois_file);
}
function reviewRoisCacheKey(run){
  const url = reviewRoisFileUrl(run);
  return run && url ? `${run.run_id}:${url}` : '';
}
function activeRunReviewRoisFile(){
  const run = activeRun();
  if(!run || runGenerated(run)) return null;
  const key = reviewRoisCacheKey(run);
  const cached = key ? reviewRoisFileCache.get(key) : null;
  return Array.isArray(cached?.rois) && cached.rois.length ? cached.rois : null;
}
function reviewRois(){ return (activeRunReviewRois() || activeRunReviewRoisFile() || data.rois || []).concat(virtualRoisArray()); }
function selectedRoi(){ return reviewRois().find(r => String(r.id) === String(selectedId)) || reviewRois()[0] || null; }
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
function rebaseReviewRoiSummaryAssets(payload, summaryUrl){
  const slash = String(summaryUrl || '').lastIndexOf('/');
  const base = slash >= 0 ? String(summaryUrl).slice(0, slash) : '';
  const rois = Array.isArray(payload) ? payload : payload.review_rois || payload.rois || [];
  return rois.map(roi => {
    if(!roi || typeof roi !== 'object') return roi;
    if(roi.trace_file) roi._traceFileUrl = rebaseRelativeAsset(roi.trace_file, base);
    if(payload?.payload_kind === 'review_rois_summary') roi._summaryOnly = true;
    return roi;
  });
}
async function ensureReviewRoisForRun(run){
  if(!run || runGenerated(run) || Array.isArray(run.artifacts?.review_rois)) return;
  const url = reviewRoisFileUrl(run);
  if(!url) return;
  const key = reviewRoisCacheKey(run);
  const cached = key ? reviewRoisFileCache.get(key) : null;
  if(cached?.status === 'ready') return;
  if(cached?.status === 'loading') return cached.promise;
  const promise = fetch(url, {cache:'no-store'}).then(async res => {
    if(!res.ok) throw new Error(await res.text());
    const payload = await res.json();
    const rois = rebaseReviewRoiSummaryAssets(payload, url);
    reviewRoisFileCache.set(key, {status:'ready', rois, source: payload?.payload_kind === 'review_rois_summary' ? 'summary' : 'full', url});
    return rois;
  }).catch(err => {
    reviewRoisFileCache.delete(key);
    throw err;
  });
  reviewRoisFileCache.set(key, {status:'loading', promise});
  await promise;
}
function stencilGapReportUrl(run){ return artifactUrl(run?.artifacts?.stencil_gap_report_file); }
function stencilGapReportCacheKey(run){
  const url = stencilGapReportUrl(run);
  return run && url ? `${run.run_id}:${url}` : '';
}
function stencilGapReportFromCache(run=activeRun()){
  const key = stencilGapReportCacheKey(run);
  return key ? stencilGapReportCache.get(key)?.data || null : null;
}
async function ensureStencilGapReportForRun(run=activeRun()){
  const url = stencilGapReportUrl(run);
  if(!run || !url) return null;
  const key = stencilGapReportCacheKey(run);
  const cached = stencilGapReportCache.get(key);
  if(cached?.status === 'ready') return cached.data;
  if(cached?.status === 'loading') return cached.promise;
  const promise = fetch(url, {cache:'no-store'}).then(async res => {
    if(!res.ok) throw new Error(await res.text());
    const payload = await res.json();
    stencilGapReportCache.set(key, {status:'ready', data:payload});
    return payload;
  }).catch(err => {
    stencilGapReportCache.set(key, {status:'error', error:err.message || 'stencil gap report did not load'});
    throw err;
  });
  stencilGapReportCache.set(key, {status:'loading', promise});
  return promise;
}
function sweepEvidenceReportUrl(){
  const artifacts = data.architectureRuns?.artifacts || {};
  return artifactUrl(artifacts.sweep_evidence_report || data.architectureRuns?.sweep_evidence_report || '');
}
function sweepEvidenceReportFromCache(){
  const url = sweepEvidenceReportUrl();
  return url ? sweepEvidenceReportCache.get(url)?.data || null : null;
}
async function ensureSweepEvidenceReport(){
  const url = sweepEvidenceReportUrl();
  if(!url) return null;
  const cached = sweepEvidenceReportCache.get(url);
  if(cached?.status === 'ready') return cached.data;
  if(cached?.status === 'loading') return cached.promise;
  const promise = fetch(url, {cache:'no-store'}).then(async res => {
    if(!res.ok) throw new Error(await res.text());
    const payload = await res.json();
    sweepEvidenceReportCache.set(url, {status:'ready', data:payload});
    return payload;
  }).catch(err => {
    sweepEvidenceReportCache.set(url, {status:'error', error:err.message || 'sweep evidence report did not load'});
    throw err;
  });
  sweepEvidenceReportCache.set(url, {status:'loading', promise});
  return promise;
}
function traceShardCacheKey(roi, run=activeRun()){
  const url = roi?._traceFileUrl || artifactUrl(roi?.trace_file);
  return run && roi && url ? `${run.run_id}:${roi.id}:${url}` : '';
}
async function ensureRoiTraceLoaded(roi, {render=false}={}){
  if(!roi || Array.isArray(roi.dffTrace)) return roi;
  const url = roi._traceFileUrl || artifactUrl(roi.trace_file);
  if(!url) return roi;
  const key = traceShardCacheKey(roi);
  const cached = key ? traceShardCache.get(key) : null;
  if(cached?.status === 'ready') { mergeRoiTracePayload(roi, cached.data); return roi; }
  if(cached?.status === 'loading') return cached.promise;
  roi._traceLoading = true;
  const promise = fetch(url, {cache:'no-store'}).then(async res => {
    if(!res.ok) throw new Error(await res.text());
    const payload = await res.json();
    traceShardCache.set(key, {status:'ready', data:payload});
    mergeRoiTracePayload(roi, payload);
    return roi;
  }).catch(err => {
    roi._traceLoading = false;
    traceShardCache.set(key, {status:'error', error:err.message || 'trace shard did not load'});
    setSaveState(err.message || 'trace shard did not load', 'bad');
    throw err;
  }).finally(() => {
    roi._traceLoading = false;
    if(render) renderAll();
  });
  if(key) traceShardCache.set(key, {status:'loading', promise});
  if(render) drawTrace();
  return promise;
}
function mergeRoiTracePayload(roi, payload){
  if(!roi || !payload) return roi;
  for(const key of ['rawTrace','backgroundTrace','dffTrace','baselineTrace','eventTrace','zTrace','noiseSigma','traceSnr','backgroundCorrelation','eventSupport','trace_materialization']){
    if(Object.prototype.hasOwnProperty.call(payload, key)) roi[key] = payload[key];
  }
  if(Array.isArray(payload.events)) roi.events = payload.events;
  roi._summaryOnly = false;
  roi.trace_status = 'loaded';
  clearTraceCaches('trace-shard-loaded');
  return roi;
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
  if(runHasIntermediates(run)) return 'Data preview available; no Review dataset attached yet';
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
      if(['data','process','process-lab','qc','dataset-qc'].includes(hash)) renderDatasetQc();
    }).catch(err => {
      proposalAnalysisCache.set(url, {status:'error', error:err.message || 'proposal analysis did not load'});
      const hash = (location.hash || '#review').replace(/^#\/?/, '');
      if(['data','process','process-lab','qc','dataset-qc'].includes(hash)) renderDatasetQc();
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
  return `${activeRunId()}|${roi.id}|${roi.dffTrace?.length || 0}|${Number(kalmanGain()).toFixed(4)}|${Number(spikeGain()).toFixed(4)}`;
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
function summaryEventsForRoi(roi){
  if(!Array.isArray(roi?.events)) return [];
  return roi.events.map(ev => ({
    frame: Number(ev.frame),
    z: Number(ev.z ?? ev.score ?? ev.peak_z ?? 0),
    amplitude: Number(ev.amplitude ?? ev.event_amplitude ?? 0),
    mode: ev.mode || 'summary'
  })).filter(ev => Number.isFinite(ev.frame) && ev.frame >= 1).sort((a,b) => a.frame - b.frame);
}
function eventsForRoi(roi){
  if(!roi) return [];
  if(!Array.isArray(roi.dffTrace) || roi.dffTrace.length < 3) return summaryEventsForRoi(roi);
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
