/* Slice 3 draft editing: append-only expert corrections with local/API autosave. */

const annotationCorrectionReadOnlyModel = annotationCorrectionModel;
let annotationCorrectionOperationSequence = 0;

Object.assign(annotationCorrectionState, {
  draftReady: false,
  draftProjections: {},
  draftOperations: [],
  undoStack: [],
  redoStack: [],
  revisionToken: 0,
  saveStatus: 'Draft ready',
  saving: false,
  conflict: false,
  apiLoaded: false,
  apiLoading: false,
  apiAvailable: false
});

function annotationCorrectionDraftKey(model){
  const revisionId = model?.revision?.revisionId || 'unpublished';
  return 'neurobench-correction-draft-' + String(datasetId || 'dataset') + '-' + revisionId;
}
function annotationCorrectionProjection(item){
  return {
    ...(item.revisionProjection
      ? JSON.parse(JSON.stringify(item.revisionProjection)) : {}),
    id: item.id,
    annotation_correction_kind: 'expert',
    source_xy: item.sourceXy.map(Number),
    ui_frame: Number(item.uiFrame || 1),
    events: (item.events || []).map(Number),
    event_intervals: (item.eventIntervals || (item.events || []).map(frame => [Number(frame), Number(frame)])).map(interval => interval.map(Number)),
    geometry: JSON.parse(JSON.stringify(item.geometry || {kind:'center'})),
    linked_model_id: item.linkedModelId || '',
    promoted_from_model_id: item.promotedFromModelId || '',
    review_state: item.reviewState || '',
    notes: item.notes || '',
    deleted: Boolean(item.deleted)
  };
}
function annotationCorrectionLoadDraft(model){
  if(annotationCorrectionState.draftReady) return;
  annotationCorrectionState.revisionToken = Number(model.revision?.revisionToken || 0);
  for(const item of model.experts) annotationCorrectionState.draftProjections[item.id] = annotationCorrectionProjection(item);
  if(typeof localStorage !== 'undefined'){
    try {
      const saved = JSON.parse(localStorage.getItem(annotationCorrectionDraftKey(model)) || 'null');
      if(saved && saved.revisionId === model.revision?.revisionId){
        annotationCorrectionState.draftProjections = saved.projections || annotationCorrectionState.draftProjections;
        annotationCorrectionState.draftOperations = saved.operations || [];
        annotationCorrectionState.undoStack = saved.undoStack || [];
        annotationCorrectionState.redoStack = saved.redoStack || [];
        annotationCorrectionState.revisionToken = Number(saved.revisionToken || annotationCorrectionState.revisionToken);
        annotationCorrectionState.saveStatus = 'Recovered browser draft';
      }
    } catch (_) {
      annotationCorrectionState.saveStatus = 'Draft recovery unavailable';
    }
  }
  annotationCorrectionState.draftReady = true;
}
function annotationCorrectionApplyProjection(item, projection){
  if(!projection) return item;
  item.revisionProjection = JSON.parse(JSON.stringify(projection));
  item.sourceXy = (projection.source_xy || item.sourceXy).map(Number);
  item.uiFrame = Number(projection.ui_frame || item.uiFrame || 1);
  item.events = (projection.events || item.events || []).map(Number);
  item.eventIntervals = (projection.event_intervals || item.events.map(frame => [frame, frame])).map(interval => interval.map(Number));
  item.geometry = JSON.parse(JSON.stringify(projection.geometry || item.geometry || {kind:'center'}));
  item.linkedModelId = String(projection.linked_model_id || '');
  item.promotedFromModelId = String(projection.promoted_from_model_id || '');
  item.reviewState = String(projection.review_state || '');
  item.notes = String(projection.notes || '');
  item.deleted = Boolean(projection.deleted);
  return item;
}
annotationCorrectionModel = function(){
  const model = annotationCorrectionReadOnlyModel();
  if(!model) return model;
  annotationCorrectionLoadDraft(model);
  for(const projection of Object.values(annotationCorrectionState.draftProjections)){
    if(
      projection?.annotation_correction_kind === 'expert' &&
      !model.experts.some(item => item.id === String(projection.id))
    ){
      model.experts.push(annotationCorrectionNormalizeItem(projection, 'expert'));
    }
  }
  for(const item of model.experts){
    annotationCorrectionApplyProjection(item, annotationCorrectionState.draftProjections[item.id]);
  }
  for(const item of model.models) item.linkedExpertId = '';
  for(const expert of model.experts){
    const linked = model.models.find(item => item.id === expert.linkedModelId);
    if(linked) linked.linkedExpertId = expert.id;
  }
  return model;
};

function annotationCorrectionPersistDraft(model){
  if(typeof localStorage === 'undefined') return false;
  try {
    localStorage.setItem(annotationCorrectionDraftKey(model), JSON.stringify({
      schema_version: 1,
      revisionId: model.revision?.revisionId || '',
      revisionToken: annotationCorrectionState.revisionToken,
      projections: annotationCorrectionState.draftProjections,
      operations: annotationCorrectionState.draftOperations,
      undoStack: annotationCorrectionState.undoStack,
      redoStack: annotationCorrectionState.redoStack,
      savedAt: new Date().toISOString()
    }));
    return true;
  } catch (_) {
    return false;
  }
}
function annotationCorrectionOperation(model, selected, type, before, after){
  annotationCorrectionOperationSequence += 1;
  const timestamp = new Date().toISOString();
  return {
    schema_version: 1,
    operationId: 'op_' + Date.now().toString(36) + '_' + annotationCorrectionOperationSequence.toString(36),
    operationType: type,
    targetId: selected.id,
    before,
    after,
    evidenceViewId: annotationCorrectionState.processedViewId || 'raw',
    uiFrame: annotationCorrectionState.frame,
    sourceXy: (after?.source_xy || before?.source_xy || selected.sourceXy).map(Number),
    reviewerId: String(model.revision?.reviewerId || 'local_reviewer'),
    timestamp,
    expectedRevisionToken: annotationCorrectionState.revisionToken
  };
}
async function annotationCorrectionCommit(type, after, {history=null, clearRedo=true}={}){
  const model = annotationCorrectionModel();
  const selected = annotationCorrectionSelected(model);
  if(!selected || selected.kind !== 'expert' || model.readOnly || annotationCorrectionState.saving) return;
  const before = annotationCorrectionProjection(selected);
  const operation = annotationCorrectionOperation(model, selected, type, before, after);
  annotationCorrectionState.saving = true;
  annotationCorrectionState.saveStatus = 'Saving draft…';
  annotationCorrectionState.conflict = false;
  annotationCorrectionState.draftProjections[selected.id] = type === 'tombstone'
    ? {...before, deleted:true, tombstonedByOperationId:operation.operationId}
    : JSON.parse(JSON.stringify(after));
  annotationCorrectionState.draftProjections[selected.id].review_state = 'recently_edited';
  annotationCorrectionState.draftOperations.push(operation);
  annotationCorrectionState.revisionToken += 1;
  if(history) annotationCorrectionState.undoStack.push(history);
  if(clearRedo) annotationCorrectionState.redoStack = [];
  annotationCorrectionPersistDraft(model);
  renderAnnotationCorrection();
  try {
    if(annotationCorrectionState.apiAvailable && typeof productRequest === 'function'){
      const revisionId = encodeURIComponent(model.revision.revisionId);
      const snapshot = await productRequest('api/annotation-revisions/' + revisionId + '/operations', {
        method:'POST',
        json:{operation}
      });
      annotationCorrectionState.revisionToken = Number(snapshot.revision?.revisionToken || annotationCorrectionState.revisionToken);
      const projection = snapshot.annotations?.rois?.[selected.id];
      if(projection) annotationCorrectionState.draftProjections[selected.id] = projection;
      annotationCorrectionState.saveStatus = 'Autosaved · token ' + annotationCorrectionState.revisionToken;
    } else {
      annotationCorrectionState.saveStatus = annotationCorrectionPersistDraft(model)
        ? 'Saved in browser · token ' + annotationCorrectionState.revisionToken
        : 'Browser storage unavailable · export now';
    }
  } catch (error) {
    annotationCorrectionState.conflict = true;
    annotationCorrectionState.saveStatus = 'Conflict · local edit preserved';
  } finally {
    annotationCorrectionState.saving = false;
    annotationCorrectionPersistDraft(model);
    renderAnnotationCorrection();
  }
}
function annotationCorrectionInputNumber(id, fallback){
  const value = Number(document.getElementById(id)?.value);
  return Number.isFinite(value) ? value : fallback;
}
function annotationCorrectionApplyCenter(){
  const model = annotationCorrectionModel();
  const selected = annotationCorrectionSelected(model);
  if(!selected || selected.kind !== 'expert') return;
  const before = annotationCorrectionProjection(selected);
  const after = {...before, source_xy:[
    annotationCorrectionInputNumber('correctionEditX', before.source_xy[0]),
    annotationCorrectionInputNumber('correctionEditY', before.source_xy[1])
  ]};
  annotationCorrectionCommit('move', after, {history:{type:'move', targetId:selected.id, before, after}});
}
function annotationCorrectionApplyRadius(){
  const model = annotationCorrectionModel();
  const selected = annotationCorrectionSelected(model);
  if(!selected || selected.kind !== 'expert') return;
  const before = annotationCorrectionProjection(selected);
  const radius = Math.max(0.25, annotationCorrectionInputNumber('correctionEditRadius', Number(before.geometry?.radius_px || before.geometry?.radiusPx || 0.75)));
  const after = {...before, geometry:{kind:'circle', radius_px:radius}};
  annotationCorrectionCommit('resize', after, {history:{type:'resize', targetId:selected.id, before, after}});
}
function annotationCorrectionSaveNotes(){
  const model = annotationCorrectionModel();
  const selected = annotationCorrectionSelected(model);
  if(!selected || selected.kind !== 'expert') return;
  const before = annotationCorrectionProjection(selected);
  const after = {...before, notes:String(document.getElementById('correctionEditNotes')?.value || '')};
  annotationCorrectionCommit('edit-notes', after, {history:{type:'edit-notes', targetId:selected.id, before, after}});
}
function annotationCorrectionToggleTombstone(){
  const model = annotationCorrectionModel();
  const selected = annotationCorrectionSelected(model);
  if(!selected || selected.kind !== 'expert') return;
  const before = annotationCorrectionProjection(selected);
  const type = before.deleted ? 'restore' : 'tombstone';
  const after = before.deleted ? {...before, deleted:false} : null;
  annotationCorrectionCommit(type, after, {history:{type, targetId:selected.id, before, after}});
}
function annotationCorrectionSelectExpert(model, targetId){
  annotationCorrectionState.queue = 'all_expert';
  annotationCorrectionState.selectedKey = 'expert:' + targetId;
}
function annotationCorrectionUndo(){
  const history = annotationCorrectionState.undoStack.pop();
  if(!history || annotationCorrectionState.saving) return;
  const model = annotationCorrectionModel();
  annotationCorrectionSelectExpert(model, history.targetId);
  const type = history.type === 'tombstone' ? 'restore'
    : history.type === 'restore' ? 'tombstone'
    : history.type === 'link' ? 'unlink'
    : history.type === 'unlink' ? 'link'
    : history.type === 'promote' ? 'tombstone'
    : history.type;
  const after = type === 'tombstone' ? null : history.before;
  annotationCorrectionState.redoStack.push(history);
  annotationCorrectionCommit(type, after, {clearRedo:false});
}
function annotationCorrectionRedo(){
  const history = annotationCorrectionState.redoStack.pop();
  if(!history || annotationCorrectionState.saving) return;
  const model = annotationCorrectionModel();
  annotationCorrectionSelectExpert(model, history.targetId);
  const type = history.type === 'promote' ? 'restore' : history.type;
  const after = type === 'tombstone' ? null : history.after;
  annotationCorrectionCommit(type, after, {history, clearRedo:false});
}
function annotationCorrectionExport(){
  const model = annotationCorrectionModel();
  const payload = {
    schema_version: 1,
    kind: 'annotation_revision_draft_export',
    revision: {...model.revision, revisionToken:annotationCorrectionState.revisionToken, operationCount:annotationCorrectionState.draftOperations.length},
    expert_projections: annotationCorrectionState.draftProjections,
    operations: annotationCorrectionState.draftOperations,
    exportedAt: new Date().toISOString()
  };
  const blob = new Blob([JSON.stringify(payload, null, 2) + '\n'], {type:'application/json'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = String(model.revision?.revisionId || 'annotation_draft') + '_draft.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 0);
}
function annotationCorrectionEditControlsHtml(model, selected){
  if(!selected) return '';
  if(selected.kind !== 'expert'){
    return '<p class="correctionReadOnlyNote">Model proposal geometry is frozen. Promotion will be an explicit expert-label operation in the next control set.</p>';
  }
  if(model.readOnly){
    return '<p class="correctionReadOnlyNote">This revision is read-only. Fork a draft before editing.</p>';
  }
  const projection = annotationCorrectionProjection(selected);
  const radius = Number(projection.geometry?.radius_px || projection.geometry?.radiusPx || 0.75);
  const disabled = annotationCorrectionState.saving ? ' disabled' : '';
  return [
    '<div class="correctionEditPanel">',
      '<div class="correctionEditHeading"><h3>Draft correction</h3><span class="correctionSaveStatus' + (annotationCorrectionState.conflict ? ' conflict' : '') + '">' + escapeHtml(annotationCorrectionState.saveStatus) + '</span></div>',
      '<div class="correctionEditGrid">',
        '<label>x <input id="correctionEditX" type="number" step="0.1" value="' + projection.source_xy[0] + '"' + disabled + '></label>',
        '<label>y <input id="correctionEditY" type="number" step="0.1" value="' + projection.source_xy[1] + '"' + disabled + '></label>',
        '<button id="correctionApplyCenterBtn" type="button"' + disabled + '>Apply center</button>',
        '<label>Radius <input id="correctionEditRadius" type="number" min="0.25" step="0.25" value="' + radius + '"' + disabled + '></label>',
        '<button id="correctionApplyRadiusBtn" type="button"' + disabled + '>Apply radius</button>',
      '</div>',
      '<label class="correctionNotesLabel">Notes<textarea id="correctionEditNotes" rows="2"' + disabled + '>' + escapeHtml(projection.notes || '') + '</textarea></label>',
      '<div class="correctionEditActions">',
        '<button id="correctionSaveNotesBtn" type="button"' + disabled + '>Save notes</button>',
        '<button id="correctionTombstoneBtn" type="button"' + disabled + '>' + (projection.deleted ? 'Restore ROI' : 'Tombstone ROI') + '</button>',
        '<button id="correctionUndoBtn" type="button"' + (!annotationCorrectionState.undoStack.length || annotationCorrectionState.saving ? ' disabled' : '') + '>Undo</button>',
        '<button id="correctionRedoBtn" type="button"' + (!annotationCorrectionState.redoStack.length || annotationCorrectionState.saving ? ' disabled' : '') + '>Redo</button>',
        '<button id="correctionExportBtn" type="button">Export draft</button>',
      '</div>',
    '</div>'
  ].join('');
}

const annotationCorrectionReadOnlyInspectorHtml = annotationCorrectionInspectorHtml;
annotationCorrectionInspectorHtml = function(model){
  const selected = annotationCorrectionSelected(model);
  const summary = annotationCorrectionReadOnlyInspectorHtml(model).replace(
    '<p class="correctionReadOnlyNote">Slice 2 is inspection-only. No label or geometry mutation is available.</p>',
    ''
  );
  return summary + annotationCorrectionEditControlsHtml(model, selected);
};

const annotationCorrectionReadOnlyWire = annotationCorrectionWire;
annotationCorrectionWire = function(model){
  annotationCorrectionReadOnlyWire(model);
  document.getElementById('correctionApplyCenterBtn')?.addEventListener('click', annotationCorrectionApplyCenter);
  document.getElementById('correctionApplyRadiusBtn')?.addEventListener('click', annotationCorrectionApplyRadius);
  document.getElementById('correctionSaveNotesBtn')?.addEventListener('click', annotationCorrectionSaveNotes);
  document.getElementById('correctionTombstoneBtn')?.addEventListener('click', annotationCorrectionToggleTombstone);
  document.getElementById('correctionUndoBtn')?.addEventListener('click', annotationCorrectionUndo);
  document.getElementById('correctionRedoBtn')?.addEventListener('click', annotationCorrectionRedo);
  document.getElementById('correctionExportBtn')?.addEventListener('click', annotationCorrectionExport);
};


async function annotationCorrectionLoadRevisionApi(model){
  if(annotationCorrectionState.apiLoaded || annotationCorrectionState.apiLoading) return;
  annotationCorrectionState.apiLoading = true;
  annotationCorrectionState.saving = true;
  annotationCorrectionState.saveStatus = 'Loading revision…';
  try {
    const revisionId = encodeURIComponent(model.revision?.revisionId || '');
    const snapshot = await productRequest('api/annotation-revisions/' + revisionId, {method:'GET'});
    const serverToken = Number(snapshot.revision?.revisionToken || 0);
    const hasLocalHistory = annotationCorrectionState.draftOperations.length > 0;
    if(hasLocalHistory && annotationCorrectionState.revisionToken !== serverToken){
      annotationCorrectionState.conflict = true;
      annotationCorrectionState.apiAvailable = false;
      annotationCorrectionState.saveStatus = 'Conflict · browser draft preserved';
    } else {
      const baseExpertIds = new Set(model.experts.map(item => item.id));
      for(const [targetId, projection] of Object.entries(snapshot.annotations?.rois || {})){
        if(baseExpertIds.has(targetId) || projection?.annotation_correction_kind === 'expert'){
          annotationCorrectionState.draftProjections[targetId] = projection;
        }
      }
      annotationCorrectionState.draftOperations = snapshot.operations || [];
      annotationCorrectionState.revisionToken = serverToken;
      annotationCorrectionState.undoStack = annotationCorrectionState.draftOperations.map(operation => ({
        type:operation.operationType,
        targetId:operation.targetId,
        before:operation.before,
        after:operation.after
      }));
      annotationCorrectionState.redoStack = [];
      annotationCorrectionState.apiAvailable = true;
      annotationCorrectionState.saveStatus = 'Autosave ready · token ' + serverToken;
      annotationCorrectionPersistDraft(model);
    }
  } catch (_) {
    annotationCorrectionState.apiAvailable = false;
    annotationCorrectionState.saveStatus = 'Server revision unavailable · browser draft';
  } finally {
    annotationCorrectionState.apiLoaded = true;
    annotationCorrectionState.apiLoading = false;
    annotationCorrectionState.saving = false;
    annotationCorrectionEditableRender();
  }
}

const annotationCorrectionEditableRender = renderAnnotationCorrection;
renderAnnotationCorrection = function(){
  annotationCorrectionEditableRender();
  const model = annotationCorrectionModel();
  if(
    model?.payload?.revision_api_enabled === true &&
    typeof productRequest === 'function' &&
    !annotationCorrectionState.apiLoaded &&
    !annotationCorrectionState.apiLoading
  ){
    annotationCorrectionLoadRevisionApi(model);
  }
};
