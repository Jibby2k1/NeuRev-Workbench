/* Slice 2 visual layout: spatial review left, scientific review right. */

function annotationCorrectionDrawCloseupMarkers(context, model, contract, selected, x0, y0, cellWidth, cellHeight){
  const items = [selected];
  if(selected.kind === 'expert' && selected.linkedModelId){
    const linked = model.models.find(item => item.id === selected.linkedModelId);
    if(linked) items.push(linked);
  }
  if(selected.kind === 'model' && selected.linkedExpertId){
    const linked = model.experts.find(item => item.id === selected.linkedExpertId);
    if(linked) items.push(linked);
  }
  for(const item of items){
    const viewXy = annotationCorrectionSourceToView(contract, item.sourceXy);
    const px = (viewXy[0] - x0 + 0.5) * cellWidth;
    const py = (viewXy[1] - y0 + 0.5) * cellHeight;
    const markerRadius = Math.max(8, Math.min(cellWidth, cellHeight) * 0.38);
    context.strokeStyle = item.key === selected.key ? '#facc15' : item.kind === 'expert' ? '#38d47a' : '#f59e0b';
    context.setLineDash(item.deleted ? [6, 4] : []);
    context.lineWidth = item.key === selected.key ? 4 : 2.5;
    context.beginPath();
    if(item.kind === 'expert') context.arc(px, py, markerRadius, 0, Math.PI * 2);
    else context.rect(px - markerRadius, py - markerRadius, markerRadius * 2, markerRadius * 2);
    context.stroke();
    context.setLineDash([]);
  }
}

function annotationCorrectionDrawCloseup(canvas, model, viewId){
  if(!canvas || !model) return;
  const selected = annotationCorrectionSelected(model);
  const contract = annotationCorrectionContract(model, viewId);
  const frames = model.arrays[viewId] || [];
  const shape = annotationCorrectionShape(contract);
  canvas.width = 320;
  canvas.height = 190;
  const context = canvas.getContext('2d');
  context.fillStyle = '#0a0f12';
  context.fillRect(0, 0, canvas.width, canvas.height);
  if(!selected){
    context.fillStyle = '#d7e0e5';
    context.font = '13px sans-serif';
    context.fillText('Select an ROI to inspect its neighborhood.', 14, 28);
    return;
  }
  const selectedViewXy = annotationCorrectionSourceToView(contract, selected.sourceXy);
  const radius = Math.max(8, Math.ceil(Number(selected.geometry?.radius_px || selected.geometry?.radiusPx || 1) * 4));
  const x0 = Math.max(0, Math.floor(selectedViewXy[0]) - radius);
  const x1 = Math.min(shape[2] - 1, Math.ceil(selectedViewXy[0]) + radius);
  const y0 = Math.max(0, Math.floor(selectedViewXy[1]) - radius);
  const y1 = Math.min(shape[1] - 1, Math.ceil(selectedViewXy[1]) + radius);
  const columns = Math.max(1, x1 - x0 + 1);
  const rows = Math.max(1, y1 - y0 + 1);
  const cellWidth = canvas.width / columns;
  const cellHeight = canvas.height / rows;
  const frame = frames[annotationCorrectionFrameIndex(contract, annotationCorrectionState.frame)];
  if(frame){
    const semantics = contract?.intensity_semantics || contract?.intensitySemantics || '';
    const range = annotationCorrectionRange(frames, semantics);
    for(let y = y0; y <= y1; y++) for(let x = x0; x <= x1; x++){
      const value = Number(frame[y]?.[x] || 0);
      const gray = Math.max(0, Math.min(255, Math.round(255 * (value - range[0]) / (range[1] - range[0]))));
      context.fillStyle = 'rgb(' + gray + ',' + gray + ',' + gray + ')';
      context.fillRect((x - x0) * cellWidth, (y - y0) * cellHeight, cellWidth + 0.5, cellHeight + 0.5);
    }
    annotationCorrectionDrawCloseupMarkers(context, model, contract, selected, x0, y0, cellWidth, cellHeight);
    return;
  }
  const url = annotationCorrectionFrameUrl(contract, annotationCorrectionState.frame);
  if(url && typeof Image !== 'undefined'){
    canvas.dataset.annotationFrameUrl = url;
    const image = new Image();
    image.onload = () => {
      if(canvas.dataset.annotationFrameUrl !== url) return;
      context.drawImage(image, x0, y0, columns, rows, 0, 0, canvas.width, canvas.height);
      annotationCorrectionDrawCloseupMarkers(context, model, contract, selected, x0, y0, cellWidth, cellHeight);
    };
    image.onerror = () => {
      if(canvas.dataset.annotationFrameUrl !== url) return;
      context.fillStyle = '#d7e0e5';
      context.font = '13px sans-serif';
      context.fillText('ROI frame unavailable.', 14, 28);
    };
    image.src = url;
    return;
  }
  context.fillStyle = '#d7e0e5';
  context.font = '13px sans-serif';
  context.fillText('ROI frame source unavailable.', 14, 28);
}
function annotationCorrectionCanvasClick(event, viewId, model){
  const canvas = event.currentTarget;
  const contract = annotationCorrectionContract(model, viewId);
  const shape = annotationCorrectionShape(contract);
  const bounds = canvas.getBoundingClientRect();
  const viewXy = [
    (event.clientX - bounds.left) * shape[2] / bounds.width,
    (event.clientY - bounds.top) * shape[1] / bounds.height
  ];
  const sourceXy = annotationCorrectionViewToSource(contract, viewXy);
  if(!sourceXy) return;
  if(annotationCorrectionState.toolMode === 'highlight'){
    annotationCorrectionState.probeSourceXy = sourceXy;
  } else {
    const nearest = annotationCorrectionNearest(model, sourceXy);
    if(nearest){
      annotationCorrectionState.selectedKey = nearest.key;
      annotationCorrectionState.probeSourceXy = nearest.sourceXy.slice();
      annotationCorrectionState.frame = nearest.uiFrame;
    } else annotationCorrectionState.probeSourceXy = sourceXy;
  }
  renderAnnotationCorrection();
}
function annotationCorrectionProcessedOptions(model){
  return model.contracts.filter(item => annotationCorrectionViewId(item) !== 'raw').map(item => {
    const id = annotationCorrectionViewId(item);
    return '<option value="' + escapeHtml(id) + '"' + (id === annotationCorrectionState.processedViewId ? ' selected' : '') + '>' + escapeHtml(id.toUpperCase()) + '</option>';
  }).join('');
}

function annotationCorrectionQueueOptions(model){
  const modelOnly = annotationCorrectionIsModelOnly(model);
  const queues = modelOnly
    ? ANNOTATION_CORRECTION_QUEUES.filter(item => ['model_unknown', 'all_model', 'recently_edited'].includes(item[0]) || (item[0] === 'all_expert' && model.experts.length))
    : ANNOTATION_CORRECTION_QUEUES;
  return queues.map(item => {
    const count = annotationCorrectionQueueRows(model, item[0]).length;
    const label = modelOnly && item[0] === 'model_unknown' ? 'Model proposals' : item[1];
    return '<option value="' + item[0] + '"' + (item[0] === annotationCorrectionState.queue ? ' selected' : '') + '>' + escapeHtml(label) + ' (' + count + ')</option>';
  }).join('');
}

function annotationCorrectionSpatialToolbarHtml(model, minFrame, maxFrame){
  return [
    '<div class="correctionViewerToolbar" aria-label="Spatial review tools">',
      '<span class="correctionFitBadge" title="Both 50% columns fit the visible browser height and width">Auto-fit screen</span>',
      '<label>Tool <select id="correctionToolSelect"><option value="select"' + (annotationCorrectionState.toolMode === 'select' ? ' selected' : '') + '>Select ROI</option><option value="highlight"' + (annotationCorrectionState.toolMode === 'highlight' ? ' selected' : '') + '>Highlight pixel</option></select></label>',
      '<button id="correctionPlayBtn" type="button">' + (annotationCorrectionState.playing ? 'Stop' : 'Play') + '</button>',
      '<button id="correctionPrevBtn" type="button">Previous</button>',
      '<button id="correctionNextBtn" type="button">Next</button>',
      '<label class="correctionFrameControl">Frame <input id="correctionFrameSlider" type="range" min="' + minFrame + '" max="' + maxFrame + '" value="' + annotationCorrectionState.frame + '"></label>',
      '<b id="correctionFrameReadout">UI ' + annotationCorrectionState.frame + ' / index ' + (annotationCorrectionState.frame - 1) + '</b>',
      '<label>Processed <select id="correctionProcessedSelect">' + annotationCorrectionProcessedOptions(model) + '</select></label>',
      '<label>Overlay <select id="correctionOverlaySelect">',
        '<option value="selected_pair"' + (annotationCorrectionState.overlayMode === 'selected_pair' ? ' selected' : '') + '>Selected pair</option>',
        '<option value="selected_expert"' + (annotationCorrectionState.overlayMode === 'selected_expert' ? ' selected' : '') + '>Selected expert only</option>',
        '<option value="selected_model"' + (annotationCorrectionState.overlayMode === 'selected_model' ? ' selected' : '') + '>Selected model only</option>',
        '<option value="none"' + (annotationCorrectionState.overlayMode === 'none' ? ' selected' : '') + '>No annotations</option>',
        '<option value="all_experts"' + (annotationCorrectionState.overlayMode === 'all_experts' ? ' selected' : '') + '>All expert annotations</option>',
        '<option value="all_models"' + (annotationCorrectionState.overlayMode === 'all_models' ? ' selected' : '') + '>All model annotations</option>',
        '<option value="all_annotations"' + (annotationCorrectionState.overlayMode === 'all_annotations' ? ' selected' : '') + '>All annotations</option>',
      '</select></label>',
    '</div>'
  ].join('');
}
function annotationCorrectionReviewPanelHtml(model, selected){
  return [
    '<aside class="correctionReviewPanel" aria-label="Review process">',
      '<div class="correctionReviewHeading"><div><span class="eyebrow">Review process</span><h2>' + escapeHtml(selected ? selected.id : 'No ROI selected') + '</h2></div><span class="stageStatus">' + escapeHtml(selected?.kind || 'none') + '</span></div>',
      '<section class="correctionQueuePanel"><label>Review queue<select id="correctionQueueSelect">' + annotationCorrectionQueueOptions(model) + '</select></label><div id="correctionQueueList">' + annotationCorrectionQueueHtml(model) + '</div></section>',
      '<section class="correctionInspectorPanel"><h3>Selection summary</h3><div id="correctionInspector">' + annotationCorrectionInspectorHtml(model) + '</div></section>',
      '<section class="correctionCloseupPanel">',
        '<div><h3>ROI neighborhood</h3><p>Zoomed out from the ROI and synchronized to the current frame.</p></div>',
        '<div class="correctionCloseupGrid">',
          '<figure><canvas id="correctionRawCloseupCanvas" aria-label="Raw ROI neighborhood"></canvas><figcaption>Raw close-up</figcaption></figure>',
          '<figure><canvas id="correctionProcessedCloseupCanvas" aria-label="Processed ROI neighborhood"></canvas><figcaption>' + escapeHtml(annotationCorrectionState.processedViewId.toUpperCase()) + ' close-up</figcaption></figure>',
        '</div>',
      '</section>',
      '<section class="correctionTraceDock"><div class="correctionTraceHeader"><div><h3>Raw vs processed time series</h3><p id="correctionTraceSemantics">Hover for values; click or drag to select a frame; wheel to zoom horizontally.</p><p id="correctionTraceReadout" class="correctionTraceReadout">UI frame ' + annotationCorrectionState.frame + '</p></div><button id="correctionTraceResetBtn" type="button">Reset zoom</button></div><canvas id="correctionTraceCanvas" role="img" aria-label="Interactive Raw and processed selected-pixel and ROI traces"></canvas></section>',
    '</aside>'
  ].join('');
}
function annotationCorrectionWorkspaceHtml(model){
  const selected = annotationCorrectionSelected(model);
  const rawContract = annotationCorrectionContract(model, 'raw') || model.contracts[0];
  const [minFrame, maxFrame] = annotationCorrectionFrameBounds(rawContract);
  const revision = model.revision;
  const modelOnly = annotationCorrectionIsModelOnly(model);
  const revisionStatus = model.readOnly ? 'Read only' : 'Draft · token ' + Number(annotationCorrectionState.revisionToken || revision.revisionToken || 0);
  const eyebrow = modelOnly ? 'Single-reviewer proposal review' : 'Single-reviewer correction';
  const heading = modelOnly ? 'Review frozen model proposals without expert labels' : 'Inspect and correct synchronized evidence';
  const context = (model.sourceVideoId || datasetId) + ' · frozen run ' + (revision.frozenRunId || 'unspecified') + (modelOnly ? ' · expert labels pending' : '');
  return [
    '<div class="correctionContextBar">',
      '<div><span class="eyebrow">' + escapeHtml(eyebrow) + '</span><h2>' + escapeHtml(heading) + '</h2><p>' + escapeHtml(context) + '</p></div>',
      '<div class="correctionContextMeta"><span class="stageStatus' + (model.readOnly ? ' warn' : '') + '">' + escapeHtml(revisionStatus) + '</span><b>' + escapeHtml(revision.revisionId || 'unpublished fixture') + '</b><span>' + escapeHtml(revision.state || 'fixture') + '</span></div>',
    '</div>',
    '<div class="correctionWorkspaceGrid correctionWorkspaceSplit correction-fit-screen">',
      '<main class="correctionViewerPanel">',
        annotationCorrectionSpatialToolbarHtml(model, minFrame, maxFrame),
        '<div class="correctionCanvasStack">',
          '<figure><canvas id="correctionRawCanvas" aria-label="Raw annotation correction view"></canvas><figcaption>Raw · source coordinates</figcaption></figure>',
          '<figure><canvas id="correctionProcessedCanvas" aria-label="Processed annotation correction view"></canvas><figcaption>' + escapeHtml(annotationCorrectionState.processedViewId.toUpperCase()) + ' · synchronized processed view</figcaption></figure>',
        '</div>',
        '<p class="correctionLegend"><span class="expertShape">Expert circle</span><span class="modelShape">Model square</span><span class="selectedShape">Selected yellow</span><span>Fixed grayscale per view.</span></p>',
      '</main>',
      annotationCorrectionReviewPanelHtml(model, selected),
    '</div>'
  ].join('');
}
function annotationCorrectionPreloadFrames(model, frame){
  if(typeof Image === 'undefined') return;
  for(const contract of model.contracts){
    const [firstFrame, lastFrame] = annotationCorrectionFrameBounds(contract);
    for(const offset of [-2, -1, 1, 2]){
      const candidate = Math.max(firstFrame, Math.min(lastFrame, Number(frame) + offset));
      const url = annotationCorrectionFrameUrl(contract, candidate);
      if(url){ const image = new Image(); image.src = url; }
    }
  }
}

function annotationCorrectionRenderFrameViews(model, {preload=true}={}){
  const slider = document.getElementById('correctionFrameSlider');
  if(slider) slider.value = String(annotationCorrectionState.frame);
  const readout = document.getElementById('correctionFrameReadout');
  if(readout) readout.textContent = 'UI ' + annotationCorrectionState.frame + ' / index ' + (annotationCorrectionState.frame - 1);
  const play = document.getElementById('correctionPlayBtn');
  if(play) play.textContent = annotationCorrectionState.playing ? 'Stop' : 'Play';
  annotationCorrectionDrawFrame(document.getElementById('correctionRawCanvas'), model, 'raw');
  annotationCorrectionDrawFrame(document.getElementById('correctionProcessedCanvas'), model, annotationCorrectionState.processedViewId);
  annotationCorrectionDrawCloseup(document.getElementById('correctionRawCloseupCanvas'), model, 'raw');
  annotationCorrectionDrawCloseup(document.getElementById('correctionProcessedCloseupCanvas'), model, annotationCorrectionState.processedViewId);
  annotationCorrectionDrawTraces(document.getElementById('correctionTraceCanvas'), model);
  if(preload) annotationCorrectionPreloadFrames(model, annotationCorrectionState.frame);
}

function annotationCorrectionSetFrame(frame, model, {preload=true}={}){
  const raw = annotationCorrectionContract(model, 'raw') || model.contracts[0];
  const [firstFrame, lastFrame] = annotationCorrectionFrameBounds(raw);
  annotationCorrectionState.frame = Math.max(firstFrame, Math.min(lastFrame, Math.round(Number(frame))));
  annotationCorrectionRenderFrameViews(model, {preload});
}

function annotationCorrectionQueueFrame(frame, model){
  annotationCorrectionState.pendingFrame = Number(frame);
  if(annotationCorrectionState.scrubRaf !== null) return;
  const schedule = typeof requestAnimationFrame === 'function' ? requestAnimationFrame : callback => setTimeout(callback, 16);
  annotationCorrectionState.scrubRaf = schedule(() => {
    annotationCorrectionState.scrubRaf = null;
    annotationCorrectionSetFrame(annotationCorrectionState.pendingFrame, model, {preload:false});
  });
}

function annotationCorrectionZoomTrace(event, model){
  event.preventDefault();
  const raw = annotationCorrectionContract(model, 'raw') || model.contracts[0];
  const [firstFrame, lastFrame] = annotationCorrectionFrameBounds(raw);
  const [start, end] = annotationCorrectionTraceWindow(model);
  const anchor = annotationCorrectionTraceFrameAtEvent(event, model);
  const span = end - start + 1;
  const nextSpan = Math.max(20, Math.min(lastFrame - firstFrame + 1, Math.round(span * (event.deltaY > 0 ? 1.25 : 0.8))));
  const ratio = span > 1 ? (anchor - start) / (span - 1) : 0.5;
  let nextStart = Math.round(anchor - ratio * (nextSpan - 1));
  nextStart = Math.max(firstFrame, Math.min(lastFrame - nextSpan + 1, nextStart));
  annotationCorrectionState.traceStart = nextStart;
  annotationCorrectionState.traceEnd = nextStart + nextSpan - 1;
  annotationCorrectionDrawTraces(event.currentTarget, model);
}

function annotationCorrectionWire(model){
  document.getElementById('correctionQueueSelect')?.addEventListener('change', event => {
    annotationCorrectionState.queue = event.target.value;
    annotationCorrectionState.selectedKey = '';
    renderAnnotationCorrection();
  });
  for(const button of document.querySelectorAll('[data-correction-key]')) button.addEventListener('click', () => {
    annotationCorrectionState.selectedKey = button.dataset.correctionKey;
    const selected = annotationCorrectionSelected(model);
    if(selected){
      annotationCorrectionState.frame = selected.uiFrame;
      annotationCorrectionState.probeSourceXy = selected.sourceXy.slice();
    }
    renderAnnotationCorrection();
  });
  document.getElementById('correctionToolSelect')?.addEventListener('change', event => {
    annotationCorrectionState.toolMode = event.target.value;
  });
  document.getElementById('correctionProcessedSelect')?.addEventListener('change', event => {
    annotationCorrectionState.processedViewId = event.target.value;
    renderAnnotationCorrection();
  });
  document.getElementById('correctionOverlaySelect')?.addEventListener('change', event => {
    annotationCorrectionState.overlayMode = event.target.value;
    annotationCorrectionRenderFrameViews(model, {preload:false});
  });
  document.getElementById('correctionFrameSlider')?.addEventListener('input', event => {
    annotationCorrectionQueueFrame(event.target.value, model);
  });
  document.getElementById('correctionFrameSlider')?.addEventListener('change', event => {
    annotationCorrectionSetFrame(event.target.value, model);
  });
  document.getElementById('correctionPrevBtn')?.addEventListener('click', () => annotationCorrectionSetFrame(annotationCorrectionState.frame - 1, model));
  document.getElementById('correctionNextBtn')?.addEventListener('click', () => annotationCorrectionSetFrame(annotationCorrectionState.frame + 1, model));
  document.getElementById('correctionPlayBtn')?.addEventListener('click', () => {
    annotationCorrectionState.playing = !annotationCorrectionState.playing;
    if(annotationCorrectionState.timer){
      clearInterval(annotationCorrectionState.timer);
      annotationCorrectionState.timer = null;
    }
    if(annotationCorrectionState.playing){
      annotationCorrectionState.timer = setInterval(() => {
        const currentModel = annotationCorrectionModel();
        const raw = annotationCorrectionContract(currentModel, 'raw') || currentModel.contracts[0];
        const [firstFrame, lastFrame] = annotationCorrectionFrameBounds(raw);
        const nextFrame = annotationCorrectionState.frame >= lastFrame ? firstFrame : annotationCorrectionState.frame + 1;
        annotationCorrectionSetFrame(nextFrame, currentModel);
      }, 100);
    }
    annotationCorrectionRenderFrameViews(model, {preload:true});
  });
  document.getElementById('correctionRawCanvas')?.addEventListener('click', event => annotationCorrectionCanvasClick(event, 'raw', model));
  document.getElementById('correctionProcessedCanvas')?.addEventListener('click', event => annotationCorrectionCanvasClick(event, annotationCorrectionState.processedViewId, model));
  const trace = document.getElementById('correctionTraceCanvas');
  trace?.addEventListener('pointermove', event => {
    annotationCorrectionState.traceHoverFrame = annotationCorrectionTraceFrameAtEvent(event, model);
    if(annotationCorrectionState.traceDragging) annotationCorrectionQueueFrame(annotationCorrectionState.traceHoverFrame, model);
    else annotationCorrectionDrawTraces(trace, model);
  });
  trace?.addEventListener('pointerdown', event => {
    annotationCorrectionState.traceDragging = true;
    trace.setPointerCapture?.(event.pointerId);
    annotationCorrectionSetFrame(annotationCorrectionTraceFrameAtEvent(event, model), model, {preload:false});
  });
  trace?.addEventListener('pointerup', event => {
    annotationCorrectionState.traceDragging = false;
    trace.releasePointerCapture?.(event.pointerId);
    annotationCorrectionPreloadFrames(model, annotationCorrectionState.frame);
  });
  trace?.addEventListener('pointerleave', () => {
    if(!annotationCorrectionState.traceDragging){
      annotationCorrectionState.traceHoverFrame = null;
      annotationCorrectionDrawTraces(trace, model);
    }
  });
  trace?.addEventListener('wheel', event => annotationCorrectionZoomTrace(event, model), {passive:false});
  trace?.addEventListener('dblclick', () => {
    annotationCorrectionState.traceStart = null;
    annotationCorrectionState.traceEnd = null;
    annotationCorrectionDrawTraces(trace, model);
  });
  document.getElementById('correctionTraceResetBtn')?.addEventListener('click', () => {
    annotationCorrectionState.traceStart = null;
    annotationCorrectionState.traceEnd = null;
    annotationCorrectionDrawTraces(trace, model);
  });
}

function renderAnnotationCorrection(){
  const root = document.getElementById('annotationCorrectionWorkspace');
  if(!root) return;
  const model = annotationCorrectionModel();
  if(!model){
    root.innerHTML = '<section class="correctionEmptyState"><span class="stageStatus warn">Read only · Slice 2</span><h2>No correction evidence attached</h2><p>Attach an annotationCorrection payload with view contracts, expert/model ROIs, matches, and Raw/processed frames to inspect this workspace.</p></section>';
    return;
  }
  annotationCorrectionEnsureSelection(model);
  root.innerHTML = annotationCorrectionWorkspaceHtml(model);
  annotationCorrectionWire(model);
  annotationCorrectionRenderFrameViews(model);
}
