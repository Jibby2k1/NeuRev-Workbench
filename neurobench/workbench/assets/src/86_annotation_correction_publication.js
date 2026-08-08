/* Slice 4: attributable review, server draft forks, and immutable publication. */
function annotationCorrectionSafeRevisionId(value){
  return String(value || '').trim().replace(/[^A-Za-z0-9_.-]+/g, '_').replace(/^[^A-Za-z0-9]+/, '');
}
function annotationCorrectionOperationLabel(type){
  return ({promote:'Promoted proposal',move:'Moved center',resize:'Resized ROI',tombstone:'Tombstoned ROI',restore:'Restored ROI',link:'Linked model',unlink:'Unlinked model','edit-notes':'Edited notes','edit-event-interval':'Edited events'})[type] || String(type || 'Changed ROI');
}
function annotationCorrectionOperationDetail(operation){
  const before = operation.before || {}, after = operation.after || {};
  if(operation.operationType === 'move') return (before.source_xy || []).join(', ') + ' → ' + (after.source_xy || []).join(', ');
  if(operation.operationType === 'resize') return 'radius ' + Number(before.geometry?.radius_px || 0).toFixed(2) + ' → ' + Number(after.geometry?.radius_px || 0).toFixed(2) + ' px';
  if(['link','unlink'].includes(operation.operationType)) return String(after.linked_model_id || 'no model link');
  if(operation.operationType === 'edit-event-interval') return String((after.event_intervals || []).length) + ' inclusive interval(s)';
  if(operation.operationType === 'promote') return 'from ' + String(before.proposal_id || after.promoted_from_model_id || 'model proposal');
  return 'evidence ' + String(operation.evidenceViewId || 'raw') + ' · UI frame ' + Number(operation.uiFrame || 1);
}
function annotationCorrectionDefaultChildId(revision, suffix){
  return annotationCorrectionSafeRevisionId((revision?.revisionId || 'annotation_revision') + '_' + suffix);
}
function annotationCorrectionChangeReviewHtml(model){
  const operations = annotationCorrectionState.draftOperations || [], counts = {};
  for(const operation of operations){
    const label = annotationCorrectionOperationLabel(operation.operationType);
    counts[label] = (counts[label] || 0) + 1;
  }
  const chips = Object.entries(counts).length ? Object.entries(counts).map(([label,count]) =>
    '<span class="correctionChangeChip"><b>' + count + '</b> ' + escapeHtml(label) + '</span>'
  ).join('') : '<span class="hint">No operations in this revision yet.</span>';
  const rows = operations.length ? operations.slice(-6).reverse().map(operation => [
    '<li><div><b>' + escapeHtml(annotationCorrectionOperationLabel(operation.operationType)) + '</b><span>' + escapeHtml(operation.targetId) + '</span></div>',
    '<p>' + escapeHtml(annotationCorrectionOperationDetail(operation)) + '</p>',
    '<small>' + escapeHtml(operation.evidenceViewId || 'raw') + ' · UI ' + Number(operation.uiFrame || 1) + ' · ' + escapeHtml(operation.reviewerId || '') + '</small></li>'
  ].join('')).join('') : '<li class="correctionEmptyChanges">The child preserves the current projection and provenance.</li>';
  const apiReady = annotationCorrectionState.apiAvailable && typeof productRequest === 'function';
  const published = model.revision?.state === 'published' || model.readOnly;
  const disabled = !apiReady || annotationCorrectionState.saving ? ' disabled' : '';
  const publishDisabled = published || !apiReady || annotationCorrectionState.saving ? ' disabled' : '';
  return [
    '<div class="correctionChangeReview"><div class="correctionEditHeading"><h3>Review changes</h3><span class="correctionAuditLabel">' + operations.length + ' operation' + (operations.length === 1 ? '' : 's') + '</span></div>',
    '<div class="correctionChangeChips">' + chips + '</div><ol class="correctionChangeList">' + rows + '</ol>',
    '<p class="correctionPublicationNote">' + (apiReady ? 'Server revision connected. Child IDs are collision-safe.' : 'Live revision API required; browser draft and export remain available.') + '</p>',
    '<div class="correctionPublicationGrid"><label>Child revision ID<input id="correctionChildRevisionId" type="text" value="' + escapeHtml(annotationCorrectionDefaultChildId(model.revision, published ? 'draft_v1' : 'published_v1')) + '"' + disabled + '></label>',
    '<label>Reviewer<input id="correctionChildReviewerId" type="text" value="' + escapeHtml(model.revision?.reviewerId || 'local_reviewer') + '"' + disabled + '></label></div>',
    '<div class="correctionPublicationActions"><button id="correctionForkRevisionBtn" type="button"' + disabled + '>Fork editable draft</button>',
    '<button id="correctionPublishRevisionBtn" class="primary" type="button"' + publishDisabled + '>Publish immutable child</button></div>',
    '<p class="hint">Publishing never rewrites this draft. Reevaluation and scientific audits remain separate jobs.</p></div>'
  ].join('');
}
const annotationCorrectionRelationshipInspectorHtml = annotationCorrectionInspectorHtml;
annotationCorrectionInspectorHtml = model => annotationCorrectionRelationshipInspectorHtml(model) + annotationCorrectionChangeReviewHtml(model);

function annotationCorrectionAdoptRevisionSnapshot(snapshot, status){
  const payload = annotationCorrectionPayload();
  if(!payload || !snapshot?.revision) return;
  const previousKey = annotationCorrectionDraftKey(annotationCorrectionModel());
  payload.revision = JSON.parse(JSON.stringify(snapshot.revision));
  payload.read_only = snapshot.revision.state !== 'draft';
  annotationCorrectionState.draftProjections = Object.fromEntries(Object.entries(snapshot.annotations?.rois || {}).map(([id,item]) => [id,JSON.parse(JSON.stringify(item))]));
  annotationCorrectionState.draftOperations = (snapshot.operations || []).map(item => JSON.parse(JSON.stringify(item)));
  annotationCorrectionState.revisionToken = Number(snapshot.revision.revisionToken || 0);
  annotationCorrectionState.undoStack = annotationCorrectionState.draftOperations.map(operation => ({type:operation.operationType,targetId:operation.targetId,before:operation.before,after:operation.after}));
  annotationCorrectionState.redoStack = [];
  Object.assign(annotationCorrectionState,{conflict:false,saving:false,apiLoaded:true,apiLoading:false,apiAvailable:true,saveStatus:status,selectedKey:''});
  if(typeof localStorage !== 'undefined'){ try { localStorage.removeItem(previousKey); } catch (_) {} }
  annotationCorrectionPersistDraft(annotationCorrectionModel());
}
function annotationCorrectionPublicationInput(){
  return {
    revisionId:annotationCorrectionSafeRevisionId(document.getElementById('correctionChildRevisionId')?.value),
    reviewerId:String(document.getElementById('correctionChildReviewerId')?.value || '').trim()
  };
}
function annotationCorrectionPublicationError(message){
  Object.assign(annotationCorrectionState,{conflict:true,saving:false,saveStatus:message});
  renderAnnotationCorrection();
}
async function annotationCorrectionForkRevision(){
  const model = annotationCorrectionModel(), input = annotationCorrectionPublicationInput();
  if(!input.revisionId || !input.reviewerId) return annotationCorrectionPublicationError('Fork requires child revision ID and reviewer');
  Object.assign(annotationCorrectionState,{saving:true,saveStatus:'Forking draft…'});
  renderAnnotationCorrection();
  try {
    const snapshot = await productRequest('api/annotation-revisions/' + encodeURIComponent(model.revision.revisionId) + '/fork',{method:'POST',json:{revisionId:input.revisionId,reviewerId:input.reviewerId}});
    annotationCorrectionAdoptRevisionSnapshot(snapshot,'Editable fork ready · token 0');
  } catch (_) { return annotationCorrectionPublicationError('Fork failed · current local draft preserved'); }
  renderAnnotationCorrection();
}
async function annotationCorrectionPublishRevision(){
  const model = annotationCorrectionModel(), input = annotationCorrectionPublicationInput();
  if(!input.revisionId) return annotationCorrectionPublicationError('Publish requires a child revision ID');
  Object.assign(annotationCorrectionState,{saving:true,saveStatus:'Validating publication…'});
  renderAnnotationCorrection();
  try {
    const snapshot = await productRequest('api/annotation-revisions/' + encodeURIComponent(model.revision.revisionId) + '/publish',{method:'POST',json:{revisionId:input.revisionId,expectedRevisionToken:annotationCorrectionState.revisionToken}});
    annotationCorrectionAdoptRevisionSnapshot(snapshot,'Published immutable child');
  } catch (_) { return annotationCorrectionPublicationError('Publish conflict or collision · draft preserved'); }
  renderAnnotationCorrection();
}
const annotationCorrectionRelationshipWire = annotationCorrectionWire;
annotationCorrectionWire = function(model){
  annotationCorrectionRelationshipWire(model);
  document.getElementById('correctionForkRevisionBtn')?.addEventListener('click',annotationCorrectionForkRevision);
  document.getElementById('correctionPublishRevisionBtn')?.addEventListener('click',annotationCorrectionPublishRevision);
};
