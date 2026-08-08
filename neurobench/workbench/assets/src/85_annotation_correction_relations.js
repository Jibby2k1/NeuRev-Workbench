/* Slice 3 control set 2: explicit links, promotions, and event intervals. */

function annotationCorrectionSafeExpertId(value){
  return String(value || '').trim().replace(/[^A-Za-z0-9_.-]+/g, '_').replace(/^[^A-Za-z0-9]+/, '');
}
function annotationCorrectionModelLinkOptions(model, selected){
  const options = ['<option value="">No model link</option>'];
  for(const item of model.models){
    if(item.linkedExpertId && item.linkedExpertId !== selected.id) continue;
    options.push(
      '<option value="' + escapeHtml(item.id) + '"' +
      (item.id === selected.linkedModelId ? ' selected' : '') + '>' +
      escapeHtml(item.id + (item.status ? ' · ' + item.status : '')) + '</option>'
    );
  }
  return options.join('');
}
function annotationCorrectionIntervals(projection){
  const intervals = projection.event_intervals || (projection.events || []).map(frame => [frame, frame]);
  return intervals.map(interval => [Number(interval[0]), Number(interval[1])]);
}
function annotationCorrectionLinkEventControlsHtml(model, selected){
  const projection = annotationCorrectionProjection(selected);
  const disabled = annotationCorrectionState.saving || projection.deleted ? ' disabled' : '';
  const intervals = annotationCorrectionIntervals(projection);
  const intervalRows = intervals.length
    ? intervals.map((interval, index) =>
        '<span class="correctionIntervalChip">UI ' + interval[0] + '–' + interval[1] +
        '<button type="button" data-correction-remove-interval="' + index + '"' + disabled + ' aria-label="Remove event interval ' + (index + 1) + '">Remove</button></span>'
      ).join('')
    : '<span class="hint">No expert event intervals.</span>';
  return [
    '<div class="correctionEditPanel correctionRelationshipPanel">',
      '<div class="correctionEditHeading"><h3>Model correspondence</h3><span class="correctionAuditLabel">explicit operation</span></div>',
      '<div class="correctionRelationshipRow">',
        '<label>Linked model<select id="correctionLinkedModelSelect"' + disabled + '>' + annotationCorrectionModelLinkOptions(model, selected) + '</select></label>',
        '<button id="correctionApplyLinkBtn" type="button"' + disabled + '>' + (projection.linked_model_id ? 'Apply / unlink' : 'Link model') + '</button>',
      '</div>',
    '</div>',
    '<div class="correctionEditPanel correctionEventPanel">',
      '<div class="correctionEditHeading"><h3>Event intervals</h3><span class="correctionAuditLabel">UI frames · inclusive</span></div>',
      '<div class="correctionIntervalList">' + intervalRows + '</div>',
      '<div class="correctionEventInputs">',
        '<label>Start <input id="correctionEventStart" type="number" min="1" step="1" value="' + annotationCorrectionState.frame + '"' + disabled + '></label>',
        '<label>End <input id="correctionEventEnd" type="number" min="1" step="1" value="' + annotationCorrectionState.frame + '"' + disabled + '></label>',
        '<button id="correctionAddIntervalBtn" type="button"' + disabled + '>Add interval</button>',
      '</div>',
    '</div>'
  ].join('');
}
function annotationCorrectionPromotionControlsHtml(model, selected){
  if(model.readOnly) return '<p class="correctionReadOnlyNote">Fork a draft before promoting a proposal.</p>';
  if(selected.linkedExpertId){
    return [
      '<div class="correctionEditPanel">',
        '<div class="correctionEditHeading"><h3>Model correspondence</h3><span class="correctionAuditLabel">already linked</span></div>',
        '<p class="hint">This frozen proposal is linked to expert ' + escapeHtml(selected.linkedExpertId) + '.</p>',
        '<button id="correctionOpenLinkedExpertBtn" type="button">Open linked expert</button>',
      '</div>'
    ].join('');
  }
  const proposedId = annotationCorrectionSafeExpertId('E_promoted_' + selected.id);
  const disabled = annotationCorrectionState.saving ? ' disabled' : '';
  return [
    '<div class="correctionEditPanel correctionPromotionPanel">',
      '<div class="correctionEditHeading"><h3>Promote model proposal</h3><span class="correctionAuditLabel">creates expert ROI</span></div>',
      '<p class="hint">The model square remains frozen. Promotion creates a separately attributable expert circle at the same canonical center.</p>',
      '<div class="correctionPromotionInputs">',
        '<label>Expert ID <input id="correctionPromotionId" type="text" value="' + escapeHtml(proposedId) + '"' + disabled + '></label>',
        '<label>Radius <input id="correctionPromotionRadius" type="number" min="0.25" step="0.25" value="0.75"' + disabled + '></label>',
        '<button id="correctionPromoteBtn" type="button"' + disabled + '>Promote proposal</button>',
      '</div>',
    '</div>'
  ].join('');
}

const annotationCorrectionCoreEditControlsHtml = annotationCorrectionEditControlsHtml;
annotationCorrectionEditControlsHtml = function(model, selected){
  if(!selected) return '';
  if(selected.kind === 'model') return annotationCorrectionPromotionControlsHtml(model, selected);
  return annotationCorrectionCoreEditControlsHtml(model, selected) +
    annotationCorrectionLinkEventControlsHtml(model, selected);
};

function annotationCorrectionApplyLink(){
  const model = annotationCorrectionModel();
  const selected = annotationCorrectionSelected(model);
  if(!selected || selected.kind !== 'expert') return;
  const before = annotationCorrectionProjection(selected);
  const linkedModelId = String(document.getElementById('correctionLinkedModelSelect')?.value || '');
  if(linkedModelId === before.linked_model_id) return;
  const after = {...before, linked_model_id:linkedModelId};
  const type = linkedModelId ? 'link' : 'unlink';
  annotationCorrectionCommit(type, after, {history:{type, targetId:selected.id, before, after}});
}
function annotationCorrectionEventBounds(model){
  const raw = annotationCorrectionContract(model, 'raw') || model.contracts[0];
  return annotationCorrectionShape(raw)[0];
}
function annotationCorrectionSetIntervalStatus(message){
  annotationCorrectionState.conflict = false;
  annotationCorrectionState.saveStatus = message;
  renderAnnotationCorrection();
}
function annotationCorrectionAddInterval(){
  const model = annotationCorrectionModel();
  const selected = annotationCorrectionSelected(model);
  if(!selected || selected.kind !== 'expert') return;
  const start = Number(document.getElementById('correctionEventStart')?.value);
  const end = Number(document.getElementById('correctionEventEnd')?.value);
  const maxFrame = annotationCorrectionEventBounds(model);
  if(!Number.isInteger(start) || !Number.isInteger(end) || start < 1 || end < start || end > maxFrame){
    annotationCorrectionSetIntervalStatus('Invalid interval · use inclusive UI frames 1–' + maxFrame);
    return;
  }
  const before = annotationCorrectionProjection(selected);
  const intervals = annotationCorrectionIntervals(before);
  if(intervals.some(interval => start <= interval[1] && end >= interval[0])){
    annotationCorrectionSetIntervalStatus('Invalid interval · overlaps existing event');
    return;
  }
  const updated = [...intervals, [start, end]].sort((left, right) => left[0] - right[0]);
  const after = {...before, event_intervals:updated, events:updated.map(interval => interval[0])};
  annotationCorrectionCommit('edit-event-interval', after, {
    history:{type:'edit-event-interval', targetId:selected.id, before, after}
  });
}
function annotationCorrectionRemoveInterval(index){
  const model = annotationCorrectionModel();
  const selected = annotationCorrectionSelected(model);
  if(!selected || selected.kind !== 'expert') return;
  const before = annotationCorrectionProjection(selected);
  const intervals = annotationCorrectionIntervals(before);
  if(index < 0 || index >= intervals.length) return;
  const updated = intervals.filter((_, rowIndex) => rowIndex !== index);
  const after = {...before, event_intervals:updated, events:updated.map(interval => interval[0])};
  annotationCorrectionCommit('edit-event-interval', after, {
    history:{type:'edit-event-interval', targetId:selected.id, before, after}
  });
}

async function annotationCorrectionCommitPromotion(){
  const model = annotationCorrectionModel();
  const selected = annotationCorrectionSelected(model);
  if(!selected || selected.kind !== 'model' || selected.linkedExpertId || model.readOnly || annotationCorrectionState.saving) return;
  const targetId = annotationCorrectionSafeExpertId(document.getElementById('correctionPromotionId')?.value);
  if(!targetId){
    annotationCorrectionSetIntervalStatus('Promotion requires a valid expert ID');
    return;
  }
  if(model.experts.some(item => item.id === targetId) || annotationCorrectionState.draftProjections[targetId]){
    annotationCorrectionSetIntervalStatus('Promotion ID already exists');
    return;
  }
  const radius = Number(document.getElementById('correctionPromotionRadius')?.value);
  if(!Number.isFinite(radius) || radius < 0.25){
    annotationCorrectionSetIntervalStatus('Promotion radius must be at least 0.25 px');
    return;
  }
  const before = {
    proposal_id:selected.id,
    source_xy:selected.sourceXy.map(Number),
    ui_frame:selected.uiFrame,
    events:(selected.events || []).map(Number),
    geometry:JSON.parse(JSON.stringify(selected.geometry || {kind:'center'})),
    status:selected.status || 'unknown'
  };
  const after = {
    id:targetId,
    annotation_correction_kind:'expert',
    source_xy:selected.sourceXy.map(Number),
    ui_frame:selected.uiFrame,
    events:(selected.events || []).map(Number),
    event_intervals:(selected.events || []).map(frame => [Number(frame), Number(frame)]),
    geometry:{kind:'circle', radius_px:radius},
    linked_model_id:selected.id,
    promoted_from_model_id:selected.id,
    review_state:'recently_edited',
    notes:'',
    deleted:false
  };
  const operation = annotationCorrectionOperation(
    model,
    {id:targetId, sourceXy:selected.sourceXy},
    'promote',
    before,
    after
  );
  annotationCorrectionState.saving = true;
  annotationCorrectionState.saveStatus = 'Saving promotion…';
  annotationCorrectionState.conflict = false;
  annotationCorrectionState.draftProjections[targetId] = after;
  annotationCorrectionState.draftOperations.push(operation);
  annotationCorrectionState.revisionToken += 1;
  annotationCorrectionState.undoStack.push({type:'promote', targetId, before, after});
  annotationCorrectionState.redoStack = [];
  annotationCorrectionSelectExpert(model, targetId);
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
      const projection = snapshot.annotations?.rois?.[targetId];
      if(projection) annotationCorrectionState.draftProjections[targetId] = projection;
      annotationCorrectionState.saveStatus = 'Promotion autosaved · token ' + annotationCorrectionState.revisionToken;
    } else {
      annotationCorrectionState.saveStatus = annotationCorrectionPersistDraft(model)
        ? 'Promotion saved in browser · token ' + annotationCorrectionState.revisionToken
        : 'Browser storage unavailable · export now';
    }
  } catch (_) {
    annotationCorrectionState.conflict = true;
    annotationCorrectionState.saveStatus = 'Conflict · local promotion preserved';
  } finally {
    annotationCorrectionState.saving = false;
    annotationCorrectionPersistDraft(model);
    renderAnnotationCorrection();
  }
}
function annotationCorrectionOpenLinkedExpert(){
  const model = annotationCorrectionModel();
  const selected = annotationCorrectionSelected(model);
  if(!selected?.linkedExpertId) return;
  annotationCorrectionSelectExpert(model, selected.linkedExpertId);
  renderAnnotationCorrection();
}

const annotationCorrectionCoreWireWithDraft = annotationCorrectionWire;
annotationCorrectionWire = function(model){
  annotationCorrectionCoreWireWithDraft(model);
  document.getElementById('correctionApplyLinkBtn')?.addEventListener('click', annotationCorrectionApplyLink);
  document.getElementById('correctionAddIntervalBtn')?.addEventListener('click', annotationCorrectionAddInterval);
  for(const button of document.querySelectorAll('[data-correction-remove-interval]')){
    button.addEventListener('click', () => annotationCorrectionRemoveInterval(Number(button.dataset.correctionRemoveInterval)));
  }
  document.getElementById('correctionPromoteBtn')?.addEventListener('click', annotationCorrectionCommitPromotion);
  document.getElementById('correctionOpenLinkedExpertBtn')?.addEventListener('click', annotationCorrectionOpenLinkedExpert);
};
