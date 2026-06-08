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
  for(const roi of reviewRois()){
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
function visibleOverlayRois(){
  const rows = (setting('overlayScope') || 'all') === 'focus'
    ? visibleRois().filter(roiInFocus)
    : visibleRois();
  if(rows.length) return rows;
  const selected = selectedRoi();
  return selected ? [selected] : [];
}

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
    if (queue === 'insideStencil') return ['inside', 'edge-near'].includes(roiStencilStatus(r).status);
    if (queue === 'outsideStencil') return roiStencilStatus(r).status === 'outside';
    if (queue === 'stencilEdge') return roiStencilStatus(r).status === 'edge-near';
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
  else if (queue === 'insideStencil' || queue === 'outsideStencil' || queue === 'stencilEdge') rows.sort((a,b) => roiQualityScore(b) - roiQualityScore(a));
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
  const sourceRois = reviewRois();
  const rois = sourceRois
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
    .sort((a,b) => b.score - a.score || String(a.roi_id).localeCompare(String(b.roi_id), undefined, {numeric:true}))
    .slice(0, targets.rois);
  const selected = new Set(rois.map(r => String(r.roi_id)));
  const events = [];
  for(const roi of sourceRois){
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
  events.sort((a,b) => b.score - a.score || String(a.roi_id).localeCompare(String(b.roi_id), undefined, {numeric:true}) || Number(a.frame) - Number(b.frame));
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
  if(task.task_type === 'roi') selectRoi(task.subject_id);
  else if(task.task_type === 'event') {
    selectRoi(task.roi_id);
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

function normalizeRoiLabelMode(value){
  return value === 'selected' || value === 'off' ? value : 'all';
}
function currentRoiLabelMode(){
  return normalizeRoiLabelMode(setting('roiLabelMode') || (document.getElementById('showLabels')?.checked === false ? 'off' : 'all'));
}
function shouldDrawRoiLabel(roi){
  const mode = currentRoiLabelMode();
  if(mode === 'off') return false;
  if(mode === 'selected') return String(roi?.id) === String(selectedId) || selectedRoiIds.has(String(roi?.id));
  return true;
}
function shouldDrawSuggestionLabel(suggestion){
  const mode = currentRoiLabelMode();
  if(mode === 'off') return false;
  if(mode === 'selected') return String(suggestion?.id) === String(selectedSuggestionId);
  return true;
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
  const labelMode = currentRoiLabelMode();
  const roiLabelModeSelect = document.getElementById('roiLabelMode');
  if(roiLabelModeSelect) roiLabelModeSelect.value = labelMode;
  const showLabelsCheckbox = document.getElementById('showLabels');
  if(showLabelsCheckbox) showLabelsCheckbox.checked = labelMode !== 'off';
  const showStencilOverlay = document.getElementById('showStencilOverlay');
  if(showStencilOverlay) showStencilOverlay.checked = setting('showStencilOverlay') !== false;
  for(const id of ['showTemplateOverlay','showRegisteredProjectionOverlay','showGridOverlay','showGridIntensityOverlay','showPredictionErrorOverlay']) {
    const el = document.getElementById(id);
    if(el) el.checked = Boolean(setting(id));
  }
  updateGridCellStatus();
  const overlayPresetSelect = document.getElementById('overlayPresetSelect');
  if(overlayPresetSelect) overlayPresetSelect.value = setting('overlayPreset') || 'validate';
  const selectedOverlayMode = document.getElementById('selectedOverlayMode');
  if(selectedOverlayMode) selectedOverlayMode.value = setting('selectedOverlayMode') || 'outline';
  const roiFocusMode = document.getElementById('roiFocusMode');
  if(roiFocusMode) roiFocusMode.value = setting('roiFocusMode') || 'all';
  for(const select of document.querySelectorAll('.uiModeSelect')) select.value = normalizeUiMode(setting('uiMode'));
  for(const select of document.querySelectorAll('.themeSelect')) select.value = setting('theme') || 'system';
  const reviewerIdInput = document.getElementById('reviewerIdInput');
  if(reviewerIdInput) reviewerIdInput.value = setting('reviewerId') || '';
  const manualRoiMode = document.getElementById('manualRoiMode');
  if(manualRoiMode) manualRoiMode.value = setting('manualRoiMode') || 'select';
  const roiEditMode = document.getElementById('roiEditMode');
  if(roiEditMode) roiEditMode.value = setting('roiEditMode') || 'off';
  const workflowPreset = document.getElementById('reviewWorkflowPreset');
  if(workflowPreset) workflowPreset.value = setting('reviewWorkflowPreset') || 'custom';
  updateOverlayViewButtons();
  renderBookmarkControls();
  renderSnapshotControls();
  renderRecoveryControls();
  applyUiMode();
  applyTheme();
  applyDisplaySettings();
}

function normalizeUiMode(value){
  const mode = String(value || '').toLowerCase();
  if(mode === 'advanced') return 'expert';
  if(mode === 'basic') return 'guided';
  if(['guided', 'standard', 'expert'].includes(mode)) return mode;
  return 'guided';
}

function applyUiMode(){
  const mode = normalizeUiMode(setting('uiMode'));
  annotations.settings.uiMode = mode;
  appRoot.classList.toggle('guided-ui', mode === 'guided');
  appRoot.classList.toggle('standard-ui', mode === 'standard');
  appRoot.classList.toggle('expert-ui', mode === 'expert');
  appRoot.classList.toggle('basic-ui', mode === 'guided');
  appRoot.classList.toggle('advanced-ui', mode === 'expert');
  for(const select of document.querySelectorAll('.uiModeSelect')) select.value = mode;
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
  const detail = document.getElementById('activeRunDetail');
  if(!select || !status || !panel) return;
  const runs = architectureRuns();
  const activeId = activeRunId();
  const run = runById(activeId) || runs[0] || null;
  select.innerHTML = processRunOptionsHtml(runs, run) || runs.map(item => `<option value="${escapeHtml(item.run_id)}" ${item.run_id === activeId ? 'selected' : ''}>${escapeHtml(runLabel(item))}</option>`).join('');
  if(run && select.value !== run.run_id) select.value = run.run_id;
  status.textContent = runStatusLabel(run);
  if(detail) detail.textContent = runDetailText(run);
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
    if(runHasIntermediates(run) && !runGenerated(run)) {
      panel.innerHTML = `
        <div>
          <b>${escapeHtml(runLabel(run))}</b>
          <p class="hint">This run has browser-ready intermediate videos and candidate overlays, but it has not been converted into a standalone Review-tab dataset yet.</p>
          <p class="hint">The Review viewer keeps the raw video visible and overlays the selected run's candidate locations. Data shows processed stages side-by-side.</p>
        </div>
        <div>
          ${jobHtml}
          <div class="buttonRow">
            <button type="button" data-open-process-lab>Inspect In Data</button>
          </div>
          <details ${currentGenerationJob ? 'open' : ''}>
            <summary>Fallback command</summary>
            <pre>${escapeHtml(generationCommandForRun(run))}</pre>
          </details>
        </div>`;
      panel.querySelector('[data-open-process-lab]')?.addEventListener('click', () => {
        location.hash = '#data';
        renderDatasetQc();
      });
    } else {
      panel.innerHTML = `
        <div>
          <b>${escapeHtml(runLabel(run))}</b>
          <p class="hint">This run is configured, but the Review/Data frame outputs have not been generated or attached yet.</p>
          <p class="hint">${escapeHtml(readiness.text)}</p>
        </div>
        <div>
          ${jobHtml}
          <details ${currentGenerationJob ? 'open' : ''}>
            <summary>Fallback command</summary>
            <pre>${escapeHtml(generationCommandForRun(run))}</pre>
          </details>
        </div>`;
    }
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
    try {
      await ensureReviewRoisForRun(run);
    } catch (_) {
      setSaveState('could not load run overlays', 'bad');
    }
    renderRunSyncControls();
    const rows = reviewRois();
    if(!rows.some(roi => String(roi.id) === String(selectedId))) {
      selectedId = rows[0]?.id || null;
      selectedRoiIds = new Set(selectedId ? [String(selectedId)] : []);
    }
    clearTraceCaches('active-run-change');
    setFrame(currentFrame);
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
  return color === '#38bdf8' ? '#0ea5e9' : color;
}

function roiStateColor(ann){
  if(ann.state === 'accept') return '#22c55e';
  if(ann.state === 'reject') return '#ef4444';
  if(ann.state === 'unsure') return '#a855f7';
  return '#38bdf8';
}

function roiOverlayGroupVisible(roi){
  const group = roiAnnotationClass(roi);
  if(group === 'annotated_neuron') return booleanSetting('showAnnotatedNeuronRois', true);
  if(group === 'annotated_non_neuron') return booleanSetting('showAnnotatedNonNeuronRois', true);
  return booleanSetting('showPotentialRois', true);
}

function overlayGroupCounts(){
  const counts = {potential:0, annotated_neuron:0, annotated_non_neuron:0};
  for(const roi of visibleRois()) counts[roiAnnotationClass(roi)] += 1;
  return counts;
}

function updateOverlayToggleButton(id, settingName, visibleText, hiddenText, count=0){
  const btn = document.getElementById(id);
  if(!btn) return;
  const isVisible = booleanSetting(settingName, true);
  btn.textContent = `${isVisible ? visibleText : hiddenText} (${count})`;
  btn.classList.toggle('active', isVisible);
  btn.setAttribute('aria-pressed', String(isVisible));
}

function updateOverlayViewButtons(){
  const counts = overlayGroupCounts();
  updateOverlayToggleButton('togglePotentialRoisBtn', 'showPotentialRois', 'Hide Potential Neurons', 'Show Potential Neurons', counts.potential);
  updateOverlayToggleButton('toggleAnnotatedNeuronRoisBtn', 'showAnnotatedNeuronRois', 'Hide Annotated Neurons', 'Show Annotated Neurons', counts.annotated_neuron);
  updateOverlayToggleButton('toggleAnnotatedNonNeuronRoisBtn', 'showAnnotatedNonNeuronRois', 'Hide Annotated Non-Neurons', 'Show Annotated Non-Neurons', counts.annotated_non_neuron);
  const status = document.getElementById('overlayViewStatus');
  if(status) {
    const total = visibleRois().length;
    const drawRows = visibleOverlayRois().filter(roi => roiOverlayGroupVisible(roi) || String(roi.id) === String(selectedId));
    const scope = (setting('overlayScope') || 'all') === 'focus' ? `focused ${roiFocusMode()}` : 'all ROIs';
    const showRois = document.getElementById('showRois')?.checked !== false;
    status.textContent = showRois ? `${drawRows.length}/${total} ROI overlays visible · ${scope}` : `ROI overlays hidden · ${total} in queue`;
  }
}

function toggleOverlayRoiGroup(settingName){
  setSetting(settingName, !booleanSetting(settingName, true));
  setSetting('overlayPreset', 'custom');
  setSetting('overlayScope', 'all');
  setSetting('roiFocusMode', 'all');
  applySettingsToControls();
  updateOverlayViewButtons();
  drawOverlay();
}

function showAllOverlayRoiGroups(){
  for(const name of ['showPotentialRois','showAnnotatedNeuronRois','showAnnotatedNonNeuronRois']) setSetting(name, true);
  setSetting('overlayPreset', 'custom');
  setSetting('overlayScope', 'all');
  setSetting('roiFocusMode', 'all');
  applySettingsToControls();
  updateOverlayViewButtons();
  drawOverlay();
}

function applyOverlayPreset(name){
  const preset = OVERLAY_PRESETS[name];
  if(!preset) return;
  setSetting('overlayPreset', name);
  for(const key of ['selectedOverlayMode','selectedFillOpacity','selectedOutlineWidth','overlayOpacity','showEvidence','showSuggestions']) {
    if(Object.prototype.hasOwnProperty.call(preset, key)) setSetting(key, preset[key]);
  }
  if(Object.prototype.hasOwnProperty.call(preset, 'showLabels')) setSetting('roiLabelMode', preset.showLabels ? 'all' : 'off');
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
  if(Object.prototype.hasOwnProperty.call(preset, 'showLabels')) setSetting('roiLabelMode', preset.showLabels ? 'all' : 'off');
  setCheckbox('showLabels', preset.showLabels);
  setCheckbox('showEvents', preset.showEvents);
  setCheckbox('showSuggestions', preset.showSuggestions);
  setCheckbox('showEvidence', preset.showEvidence);
  if(name === 'missed_neuron_search' || name === 'find_missed_neurons') {
    const details = document.getElementById('discoveryDetails');
    if(details) details.open = true;
  }
  if(name === 'mask_editing') setSetting('manualRoiMode', 'select');
  setSetting('overlayScope', preset.roiFocusMode && preset.roiFocusMode !== 'all' ? 'focus' : 'all');
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
  const showLabels = currentRoiLabelMode() !== 'off';
  const showEvents = document.getElementById('showEvents').checked;
  const showSuggestions = document.getElementById('showSuggestions').checked;
  const showStencil = setting('showStencilOverlay') !== false;
  if(showStencil) drawReviewStencilOverlay();
  if(setting('showTemplateOverlay') || setting('showRegisteredProjectionOverlay')) drawTemplateReferenceOverlay();
  if(setting('showGridOverlay') || setting('showGridIntensityOverlay') || setting('showPredictionErrorOverlay')) drawTemplateGridOverlay();
  if(!showRois && !showSuggestions) {
    drawReviewFocusBox();
    updateOverlayViewButtons();
    return;
  }
  const opacity = Number(setting('overlayOpacity'));
  if(showRois) for(const roi of visibleOverlayRois()){
    if(!roiOverlayGroupVisible(roi) && String(roi.id) !== String(selectedId)) continue;
    const ann = roiAnn(roi.id);
    const isSel = String(roi.id) === String(selectedId);
    const isMultiSel = selectedRoiIds.has(String(roi.id));
    const isEvent = showEvents && eventNearFrame(roi, currentFrame);
    const isFlash = isSel && Date.now() < selectedOverlayFlashUntil;
    let color = roiStateColor(ann);
    if(isEvent) color = '#facc15';
    const fillAlpha = isSel || isMultiSel ? Math.max(selectedOverlayFillAlpha(isEvent), isFlash ? 0.18 : 0) : (isEvent ? Math.min(0.20, opacity * 0.35) : Math.min(0.035, opacity * 0.08));
    ctx.globalAlpha = fillAlpha;
    ctx.fillStyle = color;
    if(fillAlpha > 0.005) for(const p of roi.points || []){ ctx.fillRect(p[0], p[1], 1, 1); }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = isSel || isMultiSel ? selectedOverlayStrokeColor(color, isEvent, isMultiSel) : color;
    ctx.lineWidth = isSel || isMultiSel ? Math.max(Number(setting('selectedOutlineWidth')) || 2.5, isFlash ? 4.5 : 0) : 1;
    const r = Math.max(4, Math.sqrt(roi.area / Math.PI) + 2);
    ctx.globalAlpha = isSel || isMultiSel ? 1 : (isEvent ? 0.92 : 0.52);
    ctx.beginPath(); ctx.arc(roi.centroidX, roi.centroidY, r, 0, Math.PI*2); ctx.stroke();
    if(isFlash) {
      ctx.save();
      ctx.globalAlpha = 0.95;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.arc(roi.centroidX, roi.centroidY, r + 5, 0, Math.PI*2); ctx.stroke();
      ctx.restore();
    }
    ctx.globalAlpha = 1;
    if(shouldDrawRoiLabel(roi)){
      ctx.font = '10px Arial';
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = '#111827';
      ctx.lineWidth = 3;
      ctx.strokeText(String(roi.id), roi.centroidX + 5, roi.centroidY - 5);
      ctx.fillText(String(roi.id), roi.centroidX + 5, roi.centroidY - 5);
    }
  }
  if(showRois) drawSplitMergeGuides(currentRoiLabelMode() === 'all', opacity);
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
      if(shouldDrawSuggestionLabel(s)){
        ctx.font = '10px Arial';
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#111827';
        ctx.lineWidth = 3;
        ctx.strokeText(String(s.id), s.centroidX + 5, s.centroidY - 5);
        ctx.fillText(String(s.id), s.centroidX + 5, s.centroidY - 5);
      }
    }
  }
  drawReviewFocusBox();
  drawManualPreview();
  updateOverlayViewButtons();
}

function drawReviewStencilOverlay(){
  const points = typeof savedStencilPoints === 'function' ? savedStencilPoints() : [];
  if(!points.length || points.length < 3) return;
  if(typeof drawStencilPolygon === 'function') drawStencilPolygon(ctx, points, {fill:'rgba(250, 204, 21, 0.08)', stroke:'#facc15', pointsVisible:false});
}


function templateGridPayload(){
  return data.templateGrid || data.template_grid || data.gridDynamics || data.grid_dynamics || data.architectureRuns?.templateGrid || data.architectureRuns?.template_grid || {};
}
function templateGridSpec(){
  const payload = templateGridPayload() || {};
  return payload.grid_spec || payload.gridSpec || payload.grid || payload.grid_32x32 || payload.spec || {};
}
function templateGridRegions(){
  const spec = templateGridSpec() || {};
  const regions = spec.regions || spec.cells || spec.region_specs || spec.regionSpecs || [];
  if(Array.isArray(regions)) return regions;
  if(regions && typeof regions === 'object') return Object.values(regions);
  return [];
}
function templateGridDimensions(){
  const spec = templateGridSpec() || {};
  const shape = spec.shape || spec.grid_shape || spec.gridShape || [];
  const imageShape = spec.image_shape || spec.imageShape || spec.template_shape || spec.templateShape || [];
  const rows = Math.max(1, Number(spec.rows ?? spec.grid_rows ?? spec.gridRows ?? spec.n_rows ?? shape[0] ?? 32) || 32);
  const cols = Math.max(1, Number(spec.cols ?? spec.columns ?? spec.grid_cols ?? spec.gridCols ?? spec.n_cols ?? shape[1] ?? 32) || 32);
  const width = Math.max(1, Number(spec.width ?? spec.image_width ?? spec.imageWidth ?? spec.template_width ?? spec.templateWidth ?? imageShape[1] ?? data.video?.width ?? overlay?.width ?? 1) || 1);
  const height = Math.max(1, Number(spec.height ?? spec.image_height ?? spec.imageHeight ?? spec.template_height ?? spec.templateHeight ?? imageShape[0] ?? data.video?.height ?? overlay?.height ?? 1) || 1);
  return {rows, cols, width, height};
}
function templateGridRegion(row, col){
  return templateGridRegions().find(region => {
    const r = Number(region.row ?? region.grid_row ?? region.gridRow ?? region.i);
    const c = Number(region.col ?? region.column ?? region.grid_col ?? region.gridCol ?? region.j);
    return (r === row || r === row + 1) && (c === col || c === col + 1);
  }) || null;
}
function templateGridRegionId(row, col){
  const region = templateGridRegion(row, col);
  return region?.region_id || region?.regionId || region?.id || `R${String(row).padStart(2, '0')}C${String(col).padStart(2, '0')}`;
}
function templateGridCellBbox(row, col){
  const dims = templateGridDimensions();
  const region = templateGridRegion(row, col);
  const bbox = region?.bbox || region?.bounding_box || region?.boundingBox;
  if(Array.isArray(bbox) && bbox.length >= 4) return bbox.slice(0, 4).map(Number);
  if(bbox && typeof bbox === 'object') {
    const x0 = Number(bbox.x0 ?? bbox.left ?? bbox.x ?? 0);
    const y0 = Number(bbox.y0 ?? bbox.top ?? bbox.y ?? 0);
    const x1 = Number(bbox.x1 ?? bbox.right ?? (x0 + Number(bbox.width ?? 0)));
    const y1 = Number(bbox.y1 ?? bbox.bottom ?? (y0 + Number(bbox.height ?? 0)));
    if([x0, y0, x1, y1].every(Number.isFinite)) return [x0, y0, x1, y1];
  }
  const x0 = col * dims.width / dims.cols;
  const y0 = row * dims.height / dims.rows;
  const x1 = (col + 1) * dims.width / dims.cols;
  const y1 = (row + 1) * dims.height / dims.rows;
  return [x0, y0, x1, y1];
}
function templateGridCellFromPoint(x, y){
  const dims = templateGridDimensions();
  if(!Number.isFinite(x) || !Number.isFinite(y) || dims.width <= 0 || dims.height <= 0) return null;
  const col = Math.max(0, Math.min(dims.cols - 1, Math.floor(x / dims.width * dims.cols)));
  const row = Math.max(0, Math.min(dims.rows - 1, Math.floor(y / dims.height * dims.rows)));
  return {row, col, region_id: templateGridRegionId(row, col), bbox: templateGridCellBbox(row, col)};
}
function drawTemplateReferenceOverlay(){
  const dims = templateGridDimensions();
  if(!ctx || !dims.width || !dims.height) return;
  ctx.save();
  if(setting('showTemplateOverlay')) {
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.setLineDash([7, 5]);
    ctx.strokeRect(0.5, 0.5, Math.max(1, dims.width - 1), Math.max(1, dims.height - 1));
    drawOverlayLabel('template', 8, 14, '#bae6fd');
  }
  if(setting('showRegisteredProjectionOverlay')) {
    ctx.strokeStyle = '#a78bfa';
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 4]);
    ctx.strokeRect(4.5, 4.5, Math.max(1, dims.width - 9), Math.max(1, dims.height - 9));
    drawOverlayLabel('registered projection', 8, 30, '#ddd6fe');
  }
  ctx.restore();
}
function drawTemplateGridOverlay(){
  const dims = templateGridDimensions();
  if(!ctx || !dims.width || !dims.height) return;
  const selected = setting('selectedGridCell') || null;
  ctx.save();
  if(setting('showGridIntensityOverlay') || setting('showPredictionErrorOverlay')) {
    const base = setting('showPredictionErrorOverlay') ? [244, 63, 94] : [14, 165, 233];
    for(let row = 0; row < dims.rows; row++) for(let col = 0; col < dims.cols; col++) {
      const [x0, y0, x1, y1] = templateGridCellBbox(row, col);
      const isSelected = selected && Number(selected.row) === row && Number(selected.col) === col;
      const alpha = isSelected ? 0.28 : 0.035 + ((row + col) % 2) * 0.018;
      ctx.fillStyle = `rgba(${base[0]}, ${base[1]}, ${base[2]}, ${alpha})`;
      ctx.fillRect(x0, y0, Math.max(0.5, x1 - x0), Math.max(0.5, y1 - y0));
    }
  }
  ctx.strokeStyle = setting('showPredictionErrorOverlay') ? 'rgba(244, 63, 94, 0.70)' : 'rgba(14, 165, 233, 0.68)';
  ctx.lineWidth = 0.75;
  ctx.setLineDash([]);
  for(let r = 0; r <= dims.rows; r++) {
    const y = r * dims.height / dims.rows;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(dims.width, y); ctx.stroke();
  }
  for(let c = 0; c <= dims.cols; c++) {
    const x = c * dims.width / dims.cols;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, dims.height); ctx.stroke();
  }
  if(selected && Number.isFinite(Number(selected.row)) && Number.isFinite(Number(selected.col))) {
    const [x0, y0, x1, y1] = selected.bbox || templateGridCellBbox(Number(selected.row), Number(selected.col));
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#facc15';
    ctx.fillStyle = 'rgba(250, 204, 21, 0.18)';
    ctx.fillRect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
    ctx.strokeRect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
    drawOverlayLabel(selected.region_id || templateGridRegionId(Number(selected.row), Number(selected.col)), x0 + 4, Math.max(12, y0 - 5), '#fde68a');
  }
  ctx.restore();
}
function handleGridCellClick(x, y){
  if(!setting('showGridOverlay') && !setting('showGridIntensityOverlay') && !setting('showPredictionErrorOverlay')) return false;
  const cell = templateGridCellFromPoint(x, y);
  if(!cell) return false;
  setSetting('selectedGridCell', cell);
  updateGridCellStatus();
  drawOverlay();
  return true;
}
function updateGridCellStatus(){
  const el = document.getElementById('gridCellStatus');
  if(!el) return;
  const cell = setting('selectedGridCell');
  if(!cell || cell.row === undefined || cell.col === undefined) {
    el.textContent = 'Grid cell: none';
    return;
  }
  const label = cell.region_id || templateGridRegionId(Number(cell.row), Number(cell.col));
  el.textContent = `Grid cell: ${label} (r${Number(cell.row) + 1}, c${Number(cell.col) + 1})`;
}

function drawReviewFocusBox(){
  if(!reviewFocusBox?.bbox) return;
  const [x0, y0, x1, y1] = reviewFocusBox.bbox.map(Number);
  if(![x0, y0, x1, y1].every(Number.isFinite)) return;
  ctx.save();
  ctx.setLineDash([6, 4]);
  ctx.strokeStyle = '#22d3ee';
  ctx.lineWidth = 2.5;
  ctx.strokeRect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
  ctx.fillStyle = 'rgba(34, 211, 238, 0.10)';
  ctx.fillRect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
  if(reviewFocusBox.label) drawOverlayLabel(reviewFocusBox.label, x0 + 4, Math.max(12, y0 - 5), '#a5f3fc');
  ctx.restore();
}

function focusReviewGapBox(gap){
  if(!gap?.bbox) return;
  reviewFocusBox = {bbox: gap.bbox, label: gap.id || 'stencil gap'};
  setSetting('showStencilOverlay', true);
  applySettingsToControls();
  location.hash = '#review';
  window.requestAnimationFrame(() => {
    const [x0, y0, x1, y1] = reviewFocusBox.bbox.map(Number);
    const centerX = 0.5 * (x0 + x1);
    const centerY = 0.5 * (y0 + y1);
    const zoom = Number(setting('zoom')) || 1;
    viewerScroll.scrollTo({left: Math.max(0, centerX * zoom - viewerScroll.clientWidth / 2), top: Math.max(0, centerY * zoom - viewerScroll.clientHeight / 2), behavior: 'smooth'});
    drawOverlay();
  });
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

function centerViewerOnRoi(roi, {animate=false}={}){
  if(!roi || !viewerScroll || !img) return;
  window.requestAnimationFrame(() => {
    const scale = Math.max(0.01, img.getBoundingClientRect().width / Math.max(1, data.video.width));
    const targetLeft = viewerWrap.offsetLeft + roi.centroidX * scale - viewerScroll.clientWidth / 2;
    const targetTop = viewerWrap.offsetTop + roi.centroidY * scale - viewerScroll.clientHeight / 2;
    viewerScroll.scrollTo({
      left: Math.max(0, targetLeft),
      top: Math.max(0, targetTop),
      behavior: animate ? 'smooth' : 'auto'
    });
  });
}

function flashSelectedRoi(){
  selectedOverlayFlashUntil = Date.now() + 900;
  clearTimeout(selectedOverlayFlashTimer);
  drawOverlay();
  selectedOverlayFlashTimer = setTimeout(drawOverlay, 950);
}

function focusSelectedRoi({center=true, flash=true, animate=false}={}){
  const roi = selectedRoi();
  if(!roi) return;
  if(center) centerViewerOnRoi(roi, {animate});
  if(flash) flashSelectedRoi();
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

function selectedEventWindowRecord(){
  const roi = selectedRoi();
  if(!roi) return null;
  const id = String(roi.id);
  if(annotations.virtualRois[id]) {
    annotations.virtualRois[id].event_windows = Array.isArray(annotations.virtualRois[id].event_windows) ? annotations.virtualRois[id].event_windows : [];
    return annotations.virtualRois[id];
  }
  annotations.rois[id] = roiAnn(id);
  annotations.rois[id].event_windows = Array.isArray(annotations.rois[id].event_windows) ? annotations.rois[id].event_windows : [];
  return annotations.rois[id];
}

function eventWindowsForSelectedRoi(){
  const roi = selectedRoi();
  if(!roi) return [];
  const record = annotations.virtualRois[String(roi.id)] || annotations.rois[String(roi.id)] || {};
  return Array.isArray(record.event_windows) ? record.event_windows : [];
}

function renderManualEventWindowPanel(){
  const root = document.getElementById('manualEventWindowList');
  if(!root) return;
  const roi = selectedRoi();
  if(!roi) {
    root.innerHTML = '<p class="hint">Select or draw an ROI before marking event windows.</p>';
    return;
  }
  const windows = eventWindowsForSelectedRoi();
  root.innerHTML = windows.length ? windows.map(win => `
    <div class="manualEventWindowRow">
      <b>${escapeHtml(win.start_frame)}-${escapeHtml(win.end_frame)}</b>
      <span>${escapeHtml(win.precision || 'rough')} · ${escapeHtml(win.state || 'candidate')}</span>
      <button type="button" data-delete-event-window="${escapeHtml(win.id)}">Delete</button>
    </div>`).join('') : '<p class="hint">No manual event windows for the selected ROI yet.</p>';
  for(const btn of root.querySelectorAll('[data-delete-event-window]')) btn.onclick = () => deleteManualEventWindow(btn.dataset.deleteEventWindow);
}

function setManualEventWindowFrame(which){
  const input = document.getElementById(which === 'start' ? 'manualEventStart' : 'manualEventEnd');
  if(input) input.value = currentFrame;
}

function addManualEventWindow(){
  const roi = selectedRoi();
  if(!roi) {
    setSaveState('select or draw an ROI before adding an event window', 'bad');
    return;
  }
  const start = Math.max(1, Math.min(data.video.frames, Number(document.getElementById('manualEventStart')?.value || currentFrame)));
  const end = Math.max(1, Math.min(data.video.frames, Number(document.getElementById('manualEventEnd')?.value || currentFrame)));
  const record = selectedEventWindowRecord();
  if(!record) return;
  const windowItem = stampAnnotation({
    id: `ew_${Date.now().toString(36)}`,
    start_frame: Math.min(start, end),
    end_frame: Math.max(start, end),
    precision: document.getElementById('manualEventPrecision')?.value || 'rough',
    state: document.getElementById('manualEventState')?.value || 'candidate',
    notes: ''
  });
  pushAnnotationUndo(`ROI ${roi.id} manual event window`, [
    annotations.virtualRois[roi.id] ? annotationSnapshot('virtualRois', roi.id) : annotationSnapshot('rois', roi.id)
  ]);
  record.event_windows.push(windowItem);
  if(annotations.virtualRois[roi.id]) annotations.virtualRois[roi.id] = stampAnnotation(record);
  else annotations.rois[roi.id] = stampAnnotation(record);
  selectedEventFrame = windowItem.start_frame;
  recordAction('manual_event_window_add');
  queueSave();
  renderAll();
  setSaveState(`saved event window for ROI ${roi.id}`, 'ok');
}

function deleteManualEventWindow(id){
  const roi = selectedRoi();
  const record = selectedEventWindowRecord();
  if(!roi || !record) return;
  pushAnnotationUndo(`ROI ${roi.id} delete event window`, [
    annotations.virtualRois[roi.id] ? annotationSnapshot('virtualRois', roi.id) : annotationSnapshot('rois', roi.id)
  ]);
  record.event_windows = (record.event_windows || []).filter(item => String(item.id) !== String(id));
  if(annotations.virtualRois[roi.id]) annotations.virtualRois[roi.id] = stampAnnotation(record);
  else annotations.rois[roi.id] = stampAnnotation(record);
  recordAction('manual_event_window_delete');
  queueSave();
  renderAll();
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
  el.textContent = frameRangeLabel(bounds.start, bounds.end);
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
    if(roi.trace_file || roi._traceFileUrl){
      ensureRoiTraceLoaded(roi, {render:true}).catch(() => {});
      traceCtx.fillStyle = '#0f172a'; traceCtx.font = '13px Arial';
      traceCtx.fillText(`ROI ${roi.id} | trace ${roi._traceLoading ? 'loading' : 'not loaded'}`, pad, 22);
      traceCtx.fillStyle = '#64748b'; traceCtx.font = '12px Arial';
      traceCtx.fillText('This sweep uses lightweight ROI summaries. The selected ROI trace loads on demand.', pad, 44);
      return;
    }
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
  const stencilStatus = roiStencilStatus(roi);
  const stencilHtml = `<tr><td>anatomy stencil</td><td>${escapeHtml(stencilStatusLabel(stencilStatus))}${Number.isFinite(stencilStatus.distance_px) ? ` (${fmt(stencilStatus.distance_px, 1)} px edge distance)` : ''}</td></tr>`;
  const candidateSourceHtml = roi.candidateSource ? `<tr><td>candidate source</td><td>${escapeHtml(roi.candidateSource)}</td></tr>` : '';
  const projectionScore = Number(roi.projectionScore);
  const projectionHtml = Number.isFinite(projectionScore) && projectionScore > 0 ? `<tr><td>projection score</td><td>${fmt(projectionScore, 3)}</td></tr>` : '';
  const supportFrames = Number(roi.supportFrames);
  const supportHtml = Number.isFinite(supportFrames) && supportFrames > 0 ? `<tr><td>support frames</td><td>${Math.round(supportFrames)}</td></tr>` : '';
  const warningHtml = warnings.length ? `<tr><td>warnings</td><td>${warnings.map(w => `<span class="riskPill">${w}</span>`).join(' ')}</td></tr>` : '';
  if(card) card.innerHTML = `
    <table class="smallTable">
      <tr><th>Field</th><th>Value</th></tr>
      <tr><td>ROI</td><td>${roi.id}</td></tr>
      <tr><td>priority score</td><td>${fmt(scoreValue(roi, 'priorityScore', null), 3)}</td></tr>
      <tr><td>area</td><td>${roi.area} px</td></tr>
      <tr><td>equiv. diameter</td><td>${diameterPx.toFixed(1)} px / ${diameterUm}</td></tr>
      <tr><td>kind</td><td>${escapeHtml(roi.roi_kind || 'source')}</td></tr>
      ${candidateSourceHtml}
      ${projectionHtml}
      ${supportHtml}
      <tr><td>peak score</td><td>${fmt(scoreValue(roi, 'peakScore', null), 2)}</td></tr>
      <tr><td>noise sigma</td><td>${fmt(scoreValue(roi, 'noiseSigma', null), 5)}</td></tr>
      <tr><td>trace SNR</td><td>${fmt(scoreValue(roi, 'traceSnr', null), 2)}</td></tr>
      <tr><td>local correlation</td><td>${fmt(scoreValue(roi, 'localCorrelationMean', null), 3)}</td></tr>
      <tr><td>background corr.</td><td>${fmt(scoreValue(roi, 'backgroundCorrelation', null), 3)}</td></tr>
      <tr><td>event support</td><td>${fmt(scoreValue(roi, 'eventSupport', null), 3)}</td></tr>
      <tr><td>artifact risk</td><td>${fmt(scoreValue(roi, 'artifactScore', null), 3)}</td></tr>
      <tr><td>events</td><td>${events.length}</td></tr>
      ${stencilHtml}
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
  frameLabel.textContent = frameLabelText(currentFrame);
  img.src = framePath(currentFrame);
  const runOverlayText = activeRunReviewRois() ? ` | overlay: ${runLabel(activeRun())}` : '';
  statusEl.textContent = `Frame ${frameLabelText(currentFrame)} / ${data.video.frames} (${formatSeconds(data.video.frames / Math.max(1, datasetFrameRateHz()))} total)${runOverlayText}`;
  const roi = selectedRoi();
  selectionText.textContent = roi ? `ROI ${roi.id}${selectedEventFrame ? `, event f${frameLabelText(selectedEventFrame)}` : ''}` : '';
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

function selectRoi(id, additive=false, options={}){
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
  const requestedFrame = options.frame ? Number(options.frame) : null;
  selectedEventFrame = options.eventFrame !== undefined
    ? options.eventFrame
    : requestedFrame
      ? eventsForRoi(roi).find(ev => Number(ev.frame) === requestedFrame)?.frame || visibleEventsForRoi(roi)[0]?.frame || eventsForRoi(roi)[0]?.frame || null
      : visibleEventsForRoi(roi)[0]?.frame || eventsForRoi(roi)[0]?.frame || null;
  roiNotes.value = roiAnn(id).notes || '';
  eventNotes.value = selectedEventFrame ? eventAnn(id, selectedEventFrame).notes || '' : '';
  setFrame(requestedFrame || selectedEventFrame || currentFrame);
  renderAll();
  ensureRoiTraceLoaded(roi, {render:true}).catch(() => {});
  focusSelectedRoi({center: options.center !== false, flash: options.flash !== false, animate: Boolean(options.animate)});
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
  renderManualEventWindowPanel();
  drawOverlay();
  drawTrace();
  drawEventTimeline();
  renderRoiContext();
  renderNextBestActions();
  if((location.hash || '').replace(/^#\/?/, '') === 'home') renderWorkflowHome();
}

let renderAllScheduled = false;
function scheduleRenderAll(){
  if(renderAllScheduled) return;
  renderAllScheduled = true;
  window.requestAnimationFrame(() => {
    renderAllScheduled = false;
    renderAll();
  });
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
    debug: reviewVisibilityDebugState(batch),
    summary
  };
}

function activeRunRoiSourceLabel(){
  const run = activeRun();
  if(!run) return 'no active run';
  if(Array.isArray(run.artifacts?.review_rois) && run.artifacts.review_rois.length) return 'active run embedded review_rois';
  const fileUrl = artifactUrl(run.artifacts?.review_rois_file);
  if(fileUrl) {
    const cached = reviewRoisFileCache.get(`${run.run_id}:${fileUrl}`);
    if(cached?.status === 'ready') return `active run review_rois_file loaded (${fileUrl})`;
    if(cached?.status === 'loading') return `active run review_rois_file loading (${fileUrl})`;
    return `active run review_rois_file not loaded yet (${fileUrl})`;
  }
  if(Array.isArray(data.rois) && data.rois.length) return 'embedded review_data rois';
  return 'no ROI source with candidates';
}

function reviewVisibilityDebugState(batch=nextAnnotationBatch()){
  const all = reviewRois();
  const visible = visibleRois();
  const overlayBase = visibleOverlayRois();
  const drawn = overlayBase.filter(roi => roiOverlayGroupVisible(roi) || String(roi.id) === String(selectedId));
  const counts = overlayGroupCounts();
  const notes = [];
  if(!Array.isArray(data.rois) || !data.rois.length) notes.push('Embedded baseline review_data has 0 ROIs; active run ROI files are required.');
  if((setting('queue') || '') === 'annotationBatch' && !batch.rois.length && all.length) notes.push('Annotation batch is empty; switch ROI queue to All or reset Review visibility.');
  if(visible.length <= 1 && all.length > 1) notes.push('The ROI queue/filter is reducing the loaded ROI set to one or zero candidates.');
  if((setting('overlayScope') || 'all') === 'focus' && overlayBase.length <= 1 && visible.length > 1) notes.push('Overlay scope is focused; use Overlay views / Show All to draw all queued ROIs.');
  if(!booleanSetting('showPotentialRois', true) || !booleanSetting('showAnnotatedNeuronRois', true) || !booleanSetting('showAnnotatedNonNeuronRois', true)) notes.push('One or more overlay groups are hidden.');
  if(minAreaFilter() > 0 || minEventsFilter() > 0) notes.push('Minimum area/event filters are active.');
  if(document.getElementById('showRois')?.checked === false) notes.push('The ROIs checkbox is off.');
  return {
    active_run_id: activeRunId(),
    active_run_label: runLabel(activeRun()) || activeRunId(),
    roi_source: activeRunRoiSourceLabel(),
    embedded_rois: data.rois?.length || 0,
    loaded_rois: all.length,
    visible_rois: visible.length,
    overlay_base_rois: overlayBase.length,
    drawn_overlay_rois: drawn.length,
    selected_roi: selectedId || '',
    queue: setting('queue') || 'all',
    annotation_batch_rois: batch.rois.length,
    overlay_scope: setting('overlayScope') || 'all',
    focus_mode: roiFocusMode(),
    min_area: minAreaFilter(),
    min_events: minEventsFilter(),
    groups: counts,
    group_visibility: {
      potential: booleanSetting('showPotentialRois', true),
      annotated_neuron: booleanSetting('showAnnotatedNeuronRois', true),
      annotated_non_neuron: booleanSetting('showAnnotatedNonNeuronRois', true)
    },
    notes
  };
}

function reviewDebugRows(debug){
  const rows = [
    ['active run', `${debug.active_run_label} | ${debug.active_run_id}`],
    ['ROI source', debug.roi_source],
    ['loaded / visible / drawn', `${debug.loaded_rois} / ${debug.visible_rois} / ${debug.drawn_overlay_rois}`],
    ['embedded baseline ROIs', debug.embedded_rois],
    ['queue', `${debug.queue} (${debug.annotation_batch_rois} batch ROIs)`],
    ['overlay scope / focus', `${debug.overlay_scope} / ${debug.focus_mode}`],
    ['selected ROI', debug.selected_roi || 'none'],
    ['min filters', `area >= ${debug.min_area}, events >= ${debug.min_events}`],
    ['group counts', `potential ${debug.groups.potential}, neurons ${debug.groups.annotated_neuron}, non-neurons ${debug.groups.annotated_non_neuron}`],
    ['group visibility', `potential ${debug.group_visibility.potential ? 'on' : 'off'}, neurons ${debug.group_visibility.annotated_neuron ? 'on' : 'off'}, non-neurons ${debug.group_visibility.annotated_non_neuron ? 'on' : 'off'}`]
  ];
  return rows.map(([k, v]) => `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(v)}</td></tr>`).join('');
}

function renderReviewDiagnostics(debug){
  return `
    <details class="reviewDebugPanel" open>
      <summary>Visibility Diagnostics</summary>
      <table class="smallTable">${reviewDebugRows(debug)}</table>
      ${debug.notes.length ? `<div class="debugNotes">${debug.notes.map(note => `<p>${escapeHtml(note)}</p>`).join('')}</div>` : '<p class="hint">No obvious visibility blockers detected.</p>'}
      <div class="buttonRow">
        <button type="button" id="resetReviewVisibilityBtn">Reset Review Visibility</button>
        <button type="button" id="copyReviewDebugBtn">Copy Debug JSON</button>
      </div>
    </details>`;
}

function resetReviewVisibility(){
  setSetting('queue', 'all');
  setSetting('eventQueue', 'all');
  setSetting('discoveryQueue', 'all');
  setSetting('minArea', 0);
  setSetting('minEvents', 0);
  setSetting('roiFocusMode', 'all');
  setSetting('overlayScope', 'all');
  setSetting('showPotentialRois', true);
  setSetting('showAnnotatedNeuronRois', true);
  setSetting('showAnnotatedNonNeuronRois', true);
  setSetting('showSuggestions', true);
  setSetting('roiLabelMode', 'all');
  setSetting('reviewWorkflowPreset', 'custom');
  for(const id of ['showRois','showLabels','showEvents','showSuggestions']){
    const el = document.getElementById(id);
    if(el) el.checked = true;
  }
  const first = visibleRois()[0] || reviewRois()[0];
  if(first) {
    selectedId = first.id;
    selectedRoiIds = new Set([String(first.id)]);
  }
  applySettingsToControls();
  renderAll();
  focusSelectedRoi({animate:true});
  setSaveState('review visibility reset', 'ok');
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
    visibility_debug: state.debug,
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
    ${renderReviewDiagnostics(state.debug)}
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
  document.getElementById('resetReviewVisibilityBtn').onclick = resetReviewVisibility;
  document.getElementById('copyReviewDebugBtn').onclick = () => {
    const payload = JSON.stringify(reviewVisibilityDebugState(), null, 2);
    if(navigator.clipboard?.writeText) navigator.clipboard.writeText(payload).then(() => setSaveState('copied review debug JSON', 'ok')).catch(() => downloadText(`${datasetId}_review_visibility_debug.json`, payload, 'application/json'));
    else downloadText(`${datasetId}_review_visibility_debug.json`, payload, 'application/json');
    recordAction('copy_review_visibility_debug');
  };
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
  if(!rows.length) {
    setSaveState('no ROI matches this navigation target', 'bad');
    return;
  }
  const currentIndex = rows.findIndex(r => String(r.id) === String(selectedId));
  const base = currentIndex >= 0 ? currentIndex : (delta > 0 ? -1 : 0);
  const next = rows[(base + delta + rows.length) % rows.length];
  selectRoi(next.id, false, {animate:true});
  setSaveState(`selected ROI ${next.id}`, 'ok');
}

function eventfulFrames(){
  return [...new Set(visibleRois().flatMap(roi => eventsForRoi(roi).map(ev => ev.frame)))].sort((a,b) => a - b);
}

function nextActiveFrame(delta=1){
  const frames = eventfulFrames();
  if(!frames.length) {
    setSaveState('no active frames in the current ROI queue', 'bad');
    return;
  }
  let next = frames[0];
  if(delta > 0) next = frames.find(f => f > currentFrame) || frames[0];
  else next = [...frames].reverse().find(f => f < currentFrame) || frames[frames.length - 1];
  const activeRoi = visibleRois().find(roi => eventsForRoi(roi).some(ev => Number(ev.frame) === Number(next))) || selectedRoi();
  if(activeRoi) {
    selectedId = activeRoi.id;
    selectedRoiIds = new Set([String(activeRoi.id)]);
    selectedEventFrame = next;
    roiNotes.value = roiAnn(activeRoi.id).notes || '';
    eventNotes.value = eventAnn(activeRoi.id, selectedEventFrame).notes || '';
  }
  setFrame(next);
  renderAll();
  focusSelectedRoi({animate:true});
  setSaveState(`frame ${next}${activeRoi ? `, ROI ${activeRoi.id}` : ''}`, 'ok');
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
  if((setting('overlayScope') || 'all') !== 'focus') {
    root.textContent = `${overlayed}/${visible} ROI overlays shown`;
    return;
  }
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
  return ['eventThreshold','kalmanGain','spikeGain','overlayOpacity','overlayPreset','selectedOverlayMode','selectedFillOpacity','selectedOutlineWidth','roiFocusMode','overlayScope','neighborRadiusPx','queue','eventQueue','discoveryQueue','evidenceMap','showEvidence','showSuggestions','showStencilOverlay','showTemplateOverlay','showRegisteredProjectionOverlay','showGridOverlay','showGridIntensityOverlay','showPredictionErrorOverlay','selectedGridCell','showPotentialRois','showAnnotatedNeuronRois','showAnnotatedNonNeuronRois','minArea','minEvents','activeRunId'];
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
  if(!rows.length) {
    setSaveState('no visible ROIs in the current queue', 'bad');
    return;
  }
  const idx = Math.max(0, rows.findIndex(r => String(r.id) === String(selectedId)));
  const next = rows[(idx + delta + rows.length) % rows.length];
  selectRoi(next.id, false, {animate:true});
  setSaveState(`selected ROI ${next.id}`, 'ok');
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
  applyDisplaySettings();
}
function fitHeight(){
  const height = Math.max(1, viewerScroll.clientHeight - 34);
  setSetting('zoom', Math.max(0.5, height / data.video.height));
  applySettingsToControls();
  applyDisplaySettings();
}
