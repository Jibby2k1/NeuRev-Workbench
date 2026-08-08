const ANNOTATION_CORRECTION_QUEUES = Object.freeze([
  ['matched_expert', 'Matched expert'],
  ['missed_expert', 'Missed expert'],
  ['model_unknown', 'Model-only unknown'],
  ['all_expert', 'All expert'],
  ['all_model', 'All model'],
  ['recently_edited', 'Recently edited']
]);

let annotationCorrectionState = {
  queue: 'matched_expert',
  processedViewId: 'msica',
  overlayMode: 'selected_pair',
  toolMode: 'select',
  selectedKey: '',
  frame: 1,
  playing: false,
  timer: null,
  scrubRaf: null,
  pendingFrame: null,
  traceStart: null,
  traceEnd: null,
  traceHoverFrame: null,
  traceDragging: false,
  probeSourceXy: null
};

function annotationCorrectionPayload(){
  return data.annotationCorrection || data.annotation_correction || null;
}

function annotationCorrectionNormalizeItem(item, kind){
  const sourceXy = item.source_xy || item.sourceXy || [
    item.centroidX ?? item.x ?? 0,
    item.centroidY ?? item.y ?? 0
  ];
  return {
    ...item,
    kind,
    id: String(item.id),
    key: kind + ':' + String(item.id),
    sourceXy: [Number(sourceXy[0]), Number(sourceXy[1])],
    uiFrame: Number(item.ui_frame || item.uiFrame || 1),
    events: (item.events || []).map(Number).filter(Number.isFinite),
    linkedModelId: item.linked_model_id ? String(item.linked_model_id) : '',
    linkedExpertId: item.linked_expert_id ? String(item.linked_expert_id) : '',
    reviewState: String(item.review_state || item.reviewState || ''),
    status: String(item.status || '')
  };
}

function annotationCorrectionModel(){
  const payload = annotationCorrectionPayload();
  if(!payload) return null;
  const experts = (payload.expert_rois || payload.expertRois || []).map(
    item => annotationCorrectionNormalizeItem(item, 'expert')
  );
  const models = (payload.model_rois || payload.modelRois || []).map(
    item => annotationCorrectionNormalizeItem(item, 'model')
  );
  for(const match of payload.matches || []){
    const expert = experts.find(item => item.id === String(match.expert_id || match.expertId));
    const model = models.find(item => item.id === String(match.model_id || match.modelId));
    if(expert && model){
      expert.linkedModelId = model.id;
      model.linkedExpertId = expert.id;
    }
  }
  const contracts = payload.view_contracts || payload.viewContracts || [];
  const arrays = payload.arrays_tyx || payload.arraysTyx || {};
  return {
    payload,
    experts,
    models,
    contracts,
    arrays,
    revision: payload.revision || {},
    sourceVideoId: String(payload.source_video_id || payload.sourceVideoId || ''),
    readOnly: payload.read_only !== false
  };
}

function annotationCorrectionContract(model, viewId){
  return model?.contracts.find(item => String(item.view_id || item.viewId) === String(viewId)) || null;
}

function annotationCorrectionViewId(contract){
  return String(contract?.view_id || contract?.viewId || '');
}

function annotationCorrectionShape(contract){
  const shape = contract?.shape_tyx || contract?.shapeTyx || [1, 1, 1];
  return [Number(shape[0]) || 1, Number(shape[1]) || 1, Number(shape[2]) || 1];
}

function annotationCorrectionFrameBounds(contract){
  const mapping = contract?.frame_mapping || contract?.frameMapping || {kind:'identity', offset:0};
  const start = Number(mapping.offset || 0) + 1;
  return [start, start + annotationCorrectionShape(contract)[0] - 1];
}

function annotationCorrectionFrameIndex(contract, uiFrame){
  return Number(uiFrame) - annotationCorrectionFrameBounds(contract)[0];
}

function annotationCorrectionFramePattern(contract){
  return String(contract?.frame_pattern || contract?.framePattern || '');
}

function annotationCorrectionFrameUrl(contract, uiFrame){
  const pattern = annotationCorrectionFramePattern(contract);
  return pattern && typeof framePatternPath === 'function' ? framePatternPath(pattern, uiFrame) : '';
}

function annotationCorrectionSourceToView(contract, sourceXy){
  const transform = contract?.source_to_view || contract?.sourceToView || {kind:'identity'};
  if(transform.kind !== 'affine') return [Number(sourceXy[0]), Number(sourceXy[1])];
  const matrix = transform.matrix_3x3 || transform.matrix3x3;
  return [
    Number(matrix[0][0]) * sourceXy[0] + Number(matrix[0][1]) * sourceXy[1] + Number(matrix[0][2]),
    Number(matrix[1][0]) * sourceXy[0] + Number(matrix[1][1]) * sourceXy[1] + Number(matrix[1][2])
  ];
}

function annotationCorrectionViewToSource(contract, viewXy){
  const transform = contract?.source_to_view || contract?.sourceToView || {kind:'identity'};
  if(transform.kind !== 'affine') return [Number(viewXy[0]), Number(viewXy[1])];
  const matrix = transform.matrix_3x3 || transform.matrix3x3;
  const a = Number(matrix[0][0]), b = Number(matrix[0][1]), tx = Number(matrix[0][2]);
  const c = Number(matrix[1][0]), d = Number(matrix[1][1]), ty = Number(matrix[1][2]);
  const determinant = a * d - b * c;
  if(!Number.isFinite(determinant) || Math.abs(determinant) <= 1e-12) return null;
  const x = Number(viewXy[0]) - tx, y = Number(viewXy[1]) - ty;
  return [(d * x - b * y) / determinant, (-c * x + a * y) / determinant];
}

function annotationCorrectionQueueRows(model, queueId){
  if(!model) return [];
  if(queueId === 'matched_expert') return model.experts.filter(item => item.linkedModelId);
  if(queueId === 'missed_expert') return model.experts.filter(item => !item.linkedModelId);
  if(queueId === 'model_unknown') return model.models.filter(item => !item.linkedExpertId && (item.status || 'unknown') === 'unknown');
  if(queueId === 'all_expert') return model.experts.slice();
  if(queueId === 'all_model') return model.models.slice();
  if(queueId === 'recently_edited') return [...model.experts, ...model.models].filter(item => item.reviewState === 'recently_edited');
  return [];
}

function annotationCorrectionSelected(model){
  return [...(model?.experts || []), ...(model?.models || [])].find(
    item => item.key === annotationCorrectionState.selectedKey
  ) || null;
}

function annotationCorrectionIsModelOnly(model){
  const state = String(model?.payload?.expert_annotation_state || model?.payload?.expertAnnotationState || '');
  return model?.payload?.mode === 'model_only' || (!model?.experts?.length && state.startsWith('not_applicable'));
}

function annotationCorrectionEnsureSelection(model){
  if(annotationCorrectionIsModelOnly(model)){
    if(['matched_expert', 'missed_expert', 'all_expert'].includes(annotationCorrectionState.queue)){
      annotationCorrectionState.queue = annotationCorrectionQueueRows(model, 'model_unknown').length ? 'model_unknown' : 'all_model';
    }
    if(annotationCorrectionState.overlayMode === 'selected_pair'){
      annotationCorrectionState.overlayMode = 'selected_model';
    }
  }
  const rows = annotationCorrectionQueueRows(model, annotationCorrectionState.queue);
  let selectionChanged = false;
  if(!annotationCorrectionSelected(model) || !rows.some(item => item.key === annotationCorrectionState.selectedKey)){
    annotationCorrectionState.selectedKey = rows[0]?.key || model.experts[0]?.key || model.models[0]?.key || '';
    selectionChanged = true;
  }
  if(!annotationCorrectionContract(model, annotationCorrectionState.processedViewId)){
    annotationCorrectionState.processedViewId = model.contracts.find(
      item => annotationCorrectionViewId(item) !== 'raw'
    ) ? annotationCorrectionViewId(model.contracts.find(item => annotationCorrectionViewId(item) !== 'raw')) : 'raw';
  }
  const raw = annotationCorrectionContract(model, 'raw') || model.contracts[0];
  const [minFrame, maxFrame] = annotationCorrectionFrameBounds(raw);
  if(selectionChanged){
    const selected = annotationCorrectionSelected(model);
    if(selected) annotationCorrectionState.frame = selected.uiFrame;
  }
  annotationCorrectionState.frame = Math.max(minFrame, Math.min(maxFrame, annotationCorrectionState.frame));
}

function annotationCorrectionRange(frames, semantics){
  const values = [];
  for(const frame of frames || []) for(const row of frame || []) for(const value of row || []){
    if(Number.isFinite(Number(value))) values.push(Number(value));
  }
  if(!values.length) return [0, 1];
  if(String(semantics).includes('signed')){
    const bound = Math.max(1e-12, ...values.map(value => Math.abs(value)));
    return [-bound, bound];
  }
  const low = Math.min(...values), high = Math.max(...values);
  return high > low ? [low, high] : [low, low + 1];
}

function annotationCorrectionLinkedExpert(model, selected){
  if(selected?.kind === 'expert') return selected;
  return model.experts.find(item => item.id === selected?.linkedExpertId) || null;
}

function annotationCorrectionLinkedModel(model, selected){
  if(selected?.kind === 'model') return selected;
  return model.models.find(item => item.id === selected?.linkedModelId) || null;
}

function annotationCorrectionOverlayItems(model){
  const selected = annotationCorrectionSelected(model);
  const expert = annotationCorrectionLinkedExpert(model, selected);
  const modelItem = annotationCorrectionLinkedModel(model, selected);
  if(annotationCorrectionState.overlayMode === 'none') return [];
  if(annotationCorrectionState.overlayMode === 'selected_expert') return expert ? [expert] : [];
  if(annotationCorrectionState.overlayMode === 'selected_model') return modelItem ? [modelItem] : [];
  if(annotationCorrectionState.overlayMode === 'all_experts') return model.experts;
  if(annotationCorrectionState.overlayMode === 'all_models') return model.models;
  if(annotationCorrectionState.overlayMode === 'all_annotations') return [...model.experts, ...model.models];
  return [expert, modelItem].filter((item, index, rows) => item && rows.findIndex(row => row.key === item.key) === index);
}

function annotationCorrectionDrawOverlays(context, model, contract, scale){
  const selected = annotationCorrectionSelected(model);
  const items = annotationCorrectionOverlayItems(model);

  for(const item of items){
    const viewXy = annotationCorrectionSourceToView(contract, item.sourceXy);
    const px = (viewXy[0] + 0.5) * scale, py = (viewXy[1] + 0.5) * scale;
    const isSelected = selected?.key === item.key;
    context.strokeStyle = isSelected ? '#facc15' : item.kind === 'expert' ? '#38d47a' : '#f59e0b';
    context.setLineDash(item.deleted ? [6, 4] : []);
    context.lineWidth = isSelected ? 4 : 2.5;
    const radius = Math.max(7, scale * 0.42);
    context.beginPath();
    if(item.kind === 'expert') context.arc(px, py, radius, 0, Math.PI * 2);
    else context.rect(px - radius, py - radius, radius * 2, radius * 2);
    context.stroke();
    context.fillStyle = context.strokeStyle;
    context.font = 'bold 12px sans-serif';
    context.fillText(item.id, px + radius + 3, py - radius);
    context.setLineDash([]);
  }

  if(annotationCorrectionState.probeSourceXy){
    const viewXy = annotationCorrectionSourceToView(contract, annotationCorrectionState.probeSourceXy);
    const px = (viewXy[0] + 0.5) * scale, py = (viewXy[1] + 0.5) * scale;
    context.strokeStyle = '#e2e8f0';
    context.lineWidth = 1.5;
    context.beginPath();
    context.moveTo(px - 7, py); context.lineTo(px + 7, py);
    context.moveTo(px, py - 7); context.lineTo(px, py + 7);
    context.stroke();
  }
}

function annotationCorrectionDrawFrame(canvas, model, viewId){
  if(!canvas || !model) return;
  const contract = annotationCorrectionContract(model, viewId);
  const frames = model.arrays[viewId] || [];
  const shape = annotationCorrectionShape(contract);
  const scale = Math.max(1, Math.floor(360 / Math.max(shape[1], shape[2])));
  canvas.width = shape[2] * scale;
  canvas.height = shape[1] * scale;
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = '#0a0f12';
  context.fillRect(0, 0, canvas.width, canvas.height);
  const frame = frames[annotationCorrectionFrameIndex(contract, annotationCorrectionState.frame)];
  if(frame){
    const semantics = contract?.intensity_semantics || contract?.intensitySemantics || '';
    const range = annotationCorrectionRange(frames, semantics);
    for(let y = 0; y < shape[1]; y++) for(let x = 0; x < shape[2]; x++){
      const value = Number(frame[y]?.[x] || 0);
      const gray = Math.max(0, Math.min(255, Math.round(255 * (value - range[0]) / (range[1] - range[0]))));
      context.fillStyle = 'rgb(' + gray + ',' + gray + ',' + gray + ')';
      context.fillRect(x * scale, y * scale, scale, scale);
    }
    annotationCorrectionDrawOverlays(context, model, contract, scale);
    return;
  }
  const url = annotationCorrectionFrameUrl(contract, annotationCorrectionState.frame);
  if(url && typeof Image !== 'undefined'){
    canvas.dataset.annotationFrameUrl = url;
    const image = new Image();
    image.onload = () => {
      if(canvas.dataset.annotationFrameUrl !== url) return;
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      annotationCorrectionDrawOverlays(context, model, contract, scale);
    };
    image.onerror = () => {
      if(canvas.dataset.annotationFrameUrl !== url) return;
      context.fillStyle = '#d7e0e5';
      context.font = '14px sans-serif';
      context.fillText('Frame unavailable for ' + viewId, 16, 28);
    };
    image.src = url;
    return;
  }
  context.fillStyle = '#d7e0e5';
  context.font = '14px sans-serif';
  context.fillText('No frame source for ' + viewId, 16, 28);
  annotationCorrectionDrawOverlays(context, model, contract, scale);
}

function annotationCorrectionSeries(model, viewId, sourceXy){
  const contract = annotationCorrectionContract(model, viewId);
  const frames = model.arrays[viewId] || [];
  const viewXy = annotationCorrectionSourceToView(contract, sourceXy);
  const x = Math.round(viewXy[0]), y = Math.round(viewXy[1]);
  return frames.map(frame => Number(frame[y]?.[x] || 0));
}


function annotationCorrectionProvidedSeries(item, viewId, kind){
  const traces = item?.traces || item?.trace_series || item?.traceSeries || {};
  const view = traces[viewId] || {};
  const aliases = kind === 'pixel' ? ['pixel', 'pixel_trace', 'pixelTrace'] : ['roi_mean', 'roiMean', 'mean'];
  for(const alias of aliases) if(Array.isArray(view[alias])) return view[alias].map(Number);
  return [];
}

function annotationCorrectionRoiSeries(model, viewId, item){
  if(!item) return [];
  const provided = annotationCorrectionProvidedSeries(item, viewId, 'roi_mean');
  if(provided.length) return provided;
  const contract = annotationCorrectionContract(model, viewId);
  const frames = model.arrays[viewId] || [];
  const geometry = item.geometry || {};
  const radius = geometry.kind === 'circle' ? Math.max(0, Number(geometry.radius_px || geometry.radiusPx || 0)) : 0;
  const sourcePoints = [];
  const minX = Math.floor(item.sourceXy[0] - radius), maxX = Math.ceil(item.sourceXy[0] + radius);
  const minY = Math.floor(item.sourceXy[1] - radius), maxY = Math.ceil(item.sourceXy[1] + radius);
  for(let sourceY = minY; sourceY <= maxY; sourceY++) for(let sourceX = minX; sourceX <= maxX; sourceX++){
    if(radius === 0 || Math.hypot(sourceX - item.sourceXy[0], sourceY - item.sourceXy[1]) <= radius){
      sourcePoints.push([sourceX, sourceY]);
    }
  }
  if(!sourcePoints.length) sourcePoints.push(item.sourceXy);
  return frames.map(frame => {
    const values = sourcePoints.map(sourceXy => {
      const viewXy = annotationCorrectionSourceToView(contract, sourceXy);
      return Number(frame[Math.round(viewXy[1])]?.[Math.round(viewXy[0])]);
    }).filter(Number.isFinite);
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
  });
}

function annotationCorrectionTraceWindow(model){
  const raw = annotationCorrectionContract(model, 'raw') || model.contracts[0];
  const [firstFrame, lastFrame] = annotationCorrectionFrameBounds(raw);
  let start = Number(annotationCorrectionState.traceStart);
  let end = Number(annotationCorrectionState.traceEnd);
  if(!Number.isFinite(start) || !Number.isFinite(end) || start < firstFrame || end > lastFrame || end <= start){
    start = firstFrame;
    end = lastFrame;
  }
  annotationCorrectionState.traceStart = start;
  annotationCorrectionState.traceEnd = end;
  return [start, end];
}

function annotationCorrectionTraceFrameAtEvent(event, model){
  const canvas = event.currentTarget;
  const bounds = canvas.getBoundingClientRect();
  const [start, end] = annotationCorrectionTraceWindow(model);
  const left = 110 / canvas.width * bounds.width;
  const right = 20 / canvas.width * bounds.width;
  const ratio = Math.max(0, Math.min(1, (event.clientX - bounds.left - left) / Math.max(1, bounds.width - left - right)));
  return Math.round(start + ratio * (end - start));
}

function annotationCorrectionTraceSeriesAtFrame(series, contract, uiFrame){
  return Number(series[annotationCorrectionFrameIndex(contract, uiFrame)]);
}

function annotationCorrectionUpdateTraceReadout(model, frame=annotationCorrectionState.traceHoverFrame){
  const element = document.getElementById('correctionTraceReadout');
  if(!element) return;
  const selected = annotationCorrectionSelected(model);
  const hoverIsSet = frame !== null && frame !== undefined && frame !== '';
  const uiFrame = hoverIsSet && Number.isFinite(Number(frame)) ? Number(frame) : annotationCorrectionState.frame;
  const parts = ['UI frame ' + uiFrame];
  for(const viewId of ['raw', annotationCorrectionState.processedViewId]){
    const contract = annotationCorrectionContract(model, viewId);
    const values = annotationCorrectionProvidedSeries(selected, viewId, 'pixel');
    const value = annotationCorrectionTraceSeriesAtFrame(values, contract, uiFrame);
    if(Number.isFinite(value)) parts.push(viewId.toUpperCase() + ' pixel ' + value.toFixed(3));
  }
  element.textContent = parts.join(' · ');
}

function annotationCorrectionDrawTraces(canvas, model){
  if(!canvas || !model) return;
  canvas.width = Math.max(720, Math.round(canvas.clientWidth || 720));
  canvas.height = Math.max(120, Math.round(canvas.clientHeight || 180));
  const context = canvas.getContext('2d');
  context.fillStyle = '#0f171b';
  context.fillRect(0, 0, canvas.width, canvas.height);
  const selected = annotationCorrectionSelected(model);
  const sourceXy = selected?.sourceXy || annotationCorrectionState.probeSourceXy || [0, 0];
  const expert = annotationCorrectionLinkedExpert(model, selected);
  const modelItem = annotationCorrectionLinkedModel(model, selected);
  const viewIds = ['raw', annotationCorrectionState.processedViewId].filter((value, index, rows) => rows.indexOf(value) === index);
  const lanes = viewIds.map(viewId => {
    const providedPixel = annotationCorrectionProvidedSeries(selected, viewId, 'pixel');
    return {
      viewId,
      contract: annotationCorrectionContract(model, viewId),
      series: [
        {label:'pixel', color:'#d7e0e5', values:providedPixel.length ? providedPixel : annotationCorrectionSeries(model, viewId, sourceXy)},
        ...(expert ? [{label:'expert mean', color:'#38d47a', values:annotationCorrectionRoiSeries(model, viewId, expert)}] : []),
        ...(modelItem ? [{label:'model mean', color:'#f59e0b', values:annotationCorrectionRoiSeries(model, viewId, modelItem)}] : [])
      ]
    };
  });
  const [windowStart, windowEnd] = annotationCorrectionTraceWindow(model);
  const left = 110, right = 20, top = 8, laneGap = 12;
  const laneHeight = Math.max(60, Math.floor((canvas.height - top * 2 - laneGap * Math.max(0, lanes.length - 1)) / Math.max(1, lanes.length)));
  for(let laneIndex = 0; laneIndex < lanes.length; laneIndex++){
    const lane = lanes[laneIndex];
    const fullValues = lane.series[0]?.values || [];
    const frameStart = annotationCorrectionFrameBounds(lane.contract)[0];
    const startIndex = Math.max(0, windowStart - frameStart);
    const endIndex = Math.min(fullValues.length - 1, windowEnd - frameStart);
    const visibleSeries = lane.series.map(item => ({...item, visible:item.values.slice(startIndex, endIndex + 1)}));
    const yTop = top + laneIndex * (laneHeight + laneGap);
    const range = annotationCorrectionRange([[visibleSeries.flatMap(item => item.visible)]], lane.contract?.intensity_semantics || '');
    context.strokeStyle = '#314149';
    context.strokeRect(left, yTop, canvas.width - left - right, laneHeight);
    context.fillStyle = '#d7e0e5';
    context.font = '12px sans-serif';
    context.fillText(lane.viewId.toUpperCase(), 12, yTop + 17);
    context.fillStyle = '#8fa1aa';
    context.font = '10px sans-serif';
    const semantics = String(lane.contract?.intensity_semantics || '');
    context.fillText(semantics.slice(0, 17), 12, yTop + 32);
    visibleSeries.forEach((series, index) => {
      context.fillStyle = series.color;
      context.fillRect(12, yTop + 36 + index * 10, 8, 2);
      context.fillText(series.label, 25, yTop + 40 + index * 10);
    });
    const intervals = selected?.eventIntervals || (selected?.events || []).map(frame => [frame, frame]);
    for(const interval of intervals){
      const clippedStart = Math.max(windowStart, Number(interval[0]));
      const clippedEnd = Math.min(windowEnd, Number(interval[1]));
      if(clippedEnd < clippedStart) continue;
      const x0 = left + (canvas.width - left - right) * (clippedStart - windowStart) / Math.max(1, windowEnd - windowStart);
      const x1 = left + (canvas.width - left - right) * (clippedEnd - windowStart) / Math.max(1, windowEnd - windowStart);
      context.fillStyle = 'rgba(56, 212, 122, 0.16)';
      context.fillRect(x0 - 3, yTop, Math.max(6, x1 - x0 + 6), laneHeight);
    }
    for(const series of visibleSeries){
      if(!series.visible.length) continue;
      context.strokeStyle = series.color;
      context.lineWidth = series.label === 'pixel' ? 1.25 : 2;
      context.beginPath();
      series.visible.forEach((value, index) => {
        const x = left + (canvas.width - left - right) * index / Math.max(1, series.visible.length - 1);
        const y = yTop + laneHeight - 7 - (laneHeight - 14) * (value - range[0]) / Math.max(1e-12, range[1] - range[0]);
        if(index === 0) context.moveTo(x, y); else context.lineTo(x, y);
      });
      context.stroke();
    }
    const cursorX = left + (canvas.width - left - right) * (annotationCorrectionState.frame - windowStart) / Math.max(1, windowEnd - windowStart);
    context.strokeStyle = '#facc15';
    context.lineWidth = 1.5;
    context.beginPath();
    context.moveTo(cursorX, yTop); context.lineTo(cursorX, yTop + laneHeight);
    context.stroke();
    const hoverFrame = Number(annotationCorrectionState.traceHoverFrame);
    if(Number.isFinite(hoverFrame) && hoverFrame >= windowStart && hoverFrame <= windowEnd){
      const hoverX = left + (canvas.width - left - right) * (hoverFrame - windowStart) / Math.max(1, windowEnd - windowStart);
      context.strokeStyle = 'rgba(226,232,240,0.72)';
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(hoverX, yTop); context.lineTo(hoverX, yTop + laneHeight);
      context.stroke();
    }
  }
  annotationCorrectionUpdateTraceReadout(model);
}

function annotationCorrectionNearest(model, sourceXy){
  const items = [...model.experts, ...model.models];
  let best = null, bestDistance = Infinity;
  for(const item of items){
    const distance = Math.hypot(item.sourceXy[0] - sourceXy[0], item.sourceXy[1] - sourceXy[1]);
    if(distance < bestDistance){ best = item; bestDistance = distance; }
  }
  return bestDistance <= 2.5 ? best : null;
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
  const nearest = annotationCorrectionNearest(model, sourceXy);
  if(nearest){
    annotationCorrectionState.selectedKey = nearest.key;
    annotationCorrectionState.probeSourceXy = nearest.sourceXy.slice();
  } else annotationCorrectionState.probeSourceXy = sourceXy;
  renderAnnotationCorrection();
}

function annotationCorrectionInspectorHtml(model){
  const selected = annotationCorrectionSelected(model);
  if(!selected) return '<p class="hint">Select an expert or model ROI to inspect it.</p>';
  const linkedId = selected.kind === 'expert' ? selected.linkedModelId : selected.linkedExpertId;
  const status = selected.kind === 'model' && !linkedId ? 'unknown (not negative)' : linkedId ? 'matched' : 'missed expert';
  return [
    '<dl class="correctionInspectorList">',
    '<div><dt>Selection</dt><dd>' + escapeHtml(selected.kind + ' ' + selected.id) + '</dd></div>',
    '<div><dt>Status</dt><dd>' + escapeHtml(status) + '</dd></div>',
    '<div><dt>Canonical x, y</dt><dd>' + selected.sourceXy.map(value => value.toFixed(2)).join(', ') + '</dd></div>',
    '<div><dt>Frame</dt><dd>UI ' + selected.uiFrame + ' / index ' + (selected.uiFrame - 1) + '</dd></div>',
    '<div><dt>Linked ROI</dt><dd>' + escapeHtml(linkedId || 'none') + '</dd></div>',
    '<div><dt>Events</dt><dd>' + escapeHtml(selected.events.join(', ') || 'none') + '</dd></div>',
    '<div><dt>Geometry</dt><dd>' + escapeHtml(selected.geometry?.kind || 'center') + '</dd></div>',
    '<div><dt>Trace source</dt><dd>exact pixel + ROI means</dd></div>',
    '</dl>',
    '<p class="correctionReadOnlyNote">Slice 2 is inspection-only. No label or geometry mutation is available.</p>'
  ].join('');
}

function annotationCorrectionQueueHtml(model){
  const rows = annotationCorrectionQueueRows(model, annotationCorrectionState.queue);
  if(!rows.length) return '<p class="hint correctionEmptyQueue">No items in this queue.</p>';
  return rows.map(item => {
    const selected = item.key === annotationCorrectionState.selectedKey ? ' active' : '';
    const baseStatus = item.kind === 'model' && !item.linkedExpertId ? 'unknown' : item.linkedModelId || item.linkedExpertId ? 'matched' : 'missed';
    const status = item.deleted ? 'tombstoned' : baseStatus + (item.reviewState === 'recently_edited' ? ' · edited' : '');
    const tombstoned = item.deleted ? ' tombstoned' : '';
    return '<button type="button" class="correctionQueueItem' + selected + tombstoned + '" data-correction-key="' + escapeHtml(item.key) + '"><span><b>' + escapeHtml(item.id) + '</b><small>' + escapeHtml(item.kind) + '</small></span><span class="correctionQueueStatus">' + escapeHtml(status) + '</span></button>';
  }).join('');
}

function annotationCorrectionWorkspaceHtml(model){
  const selected = annotationCorrectionSelected(model);
  const rawContract = annotationCorrectionContract(model, 'raw') || model.contracts[0];
  const maxFrame = annotationCorrectionShape(rawContract)[0];
  const revision = model.revision;
  const processedOptions = model.contracts.filter(item => annotationCorrectionViewId(item) !== 'raw').map(item => {
    const id = annotationCorrectionViewId(item);
    return '<option value="' + escapeHtml(id) + '"' + (id === annotationCorrectionState.processedViewId ? ' selected' : '') + '>' + escapeHtml(id.toUpperCase()) + '</option>';
  }).join('');
  const queueOptions = ANNOTATION_CORRECTION_QUEUES.map(item => {
    const count = annotationCorrectionQueueRows(model, item[0]).length;
    return '<option value="' + item[0] + '"' + (item[0] === annotationCorrectionState.queue ? ' selected' : '') + '>' + escapeHtml(item[1]) + ' (' + count + ')</option>';
  }).join('');
  return [
    '<div class="correctionContextBar">',
      '<div><span class="eyebrow">Single-reviewer correction</span><h2>Inspect synchronized evidence</h2><p>' + escapeHtml(model.sourceVideoId || datasetId) + ' · frozen run ' + escapeHtml(revision.frozenRunId || 'unspecified') + '</p></div>',
      '<div class="correctionContextMeta"><span class="stageStatus warn">Read only · Slice 2</span><b>' + escapeHtml(revision.revisionId || 'unpublished fixture') + '</b><span>' + escapeHtml(revision.state || 'fixture') + '</span></div>',
    '</div>',
    '<div class="correctionWorkspaceGrid">',
      '<aside class="correctionQueuePanel"><label>Review queue<select id="correctionQueueSelect">' + queueOptions + '</select></label><div id="correctionQueueList">' + annotationCorrectionQueueHtml(model) + '</div></aside>',
      '<main class="correctionViewerPanel">',
        '<div class="correctionViewerToolbar">',
          '<button id="correctionPlayBtn" type="button">' + (annotationCorrectionState.playing ? 'Pause' : 'Play') + '</button>',
          '<button id="correctionPrevBtn" type="button">Previous</button>',
          '<button id="correctionNextBtn" type="button">Next</button>',
          '<label>Frame <input id="correctionFrameSlider" type="range" min="1" max="' + maxFrame + '" value="' + annotationCorrectionState.frame + '"></label>',
          '<b>UI ' + annotationCorrectionState.frame + ' / index ' + (annotationCorrectionState.frame - 1) + '</b>',
          '<label>Processed <select id="correctionProcessedSelect">' + processedOptions + '</select></label>',
          '<label>Overlays <select id="correctionOverlaySelect">',
            '<option value="selected"' + (annotationCorrectionState.overlayMode === 'selected' ? ' selected' : '') + '>Selected + linked</option>',
            '<option value="expert"' + (annotationCorrectionState.overlayMode === 'expert' ? ' selected' : '') + '>Expert only</option>',
            '<option value="model"' + (annotationCorrectionState.overlayMode === 'model' ? ' selected' : '') + '>Model only</option>',
            '<option value="both"' + (annotationCorrectionState.overlayMode === 'both' ? ' selected' : '') + '>Both</option>',
          '</select></label>',
        '</div>',
        '<div class="correctionCanvasGrid">',
          '<figure><canvas id="correctionRawCanvas" aria-label="Raw annotation correction view"></canvas><figcaption>Raw · source coordinates</figcaption></figure>',
          '<figure><canvas id="correctionProcessedCanvas" aria-label="Processed annotation correction view"></canvas><figcaption>' + escapeHtml(annotationCorrectionState.processedViewId.toUpperCase()) + ' · synchronized</figcaption></figure>',
        '</div>',
        '<p class="correctionLegend"><span class="expertShape">Expert circle</span><span class="modelShape">Model square</span><span class="selectedShape">Selected yellow</span><span>Image content uses fixed grayscale per view.</span></p>',
      '</main>',
      '<aside class="correctionInspectorPanel"><h3>ROI inspector</h3><div id="correctionInspector">' + annotationCorrectionInspectorHtml(model) + '</div></aside>',
    '</div>',
    '<section class="correctionTraceDock"><div><h3>Pixel and ROI evidence</h3><p id="correctionTraceSemantics">Exact pixel at ' + escapeHtml(selected ? selected.sourceXy.join(', ') : 'probe') + '; expert and linked-model ROI means; fixed range per stage.</p></div><canvas id="correctionTraceCanvas" role="img" aria-label="Raw and processed selected-pixel traces"></canvas></section>'
  ].join('');
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
  document.getElementById('correctionProcessedSelect')?.addEventListener('change', event => {
    annotationCorrectionState.processedViewId = event.target.value;
    renderAnnotationCorrection();
  });
  document.getElementById('correctionOverlaySelect')?.addEventListener('change', event => {
    annotationCorrectionState.overlayMode = event.target.value;
    renderAnnotationCorrection();
  });
  document.getElementById('correctionFrameSlider')?.addEventListener('input', event => {
    annotationCorrectionState.frame = Number(event.target.value);
    renderAnnotationCorrection();
  });
  document.getElementById('correctionPrevBtn')?.addEventListener('click', () => {
    annotationCorrectionState.frame = Math.max(1, annotationCorrectionState.frame - 1);
    renderAnnotationCorrection();
  });
  document.getElementById('correctionNextBtn')?.addEventListener('click', () => {
    const raw = annotationCorrectionContract(model, 'raw') || model.contracts[0];
    annotationCorrectionState.frame = Math.min(annotationCorrectionShape(raw)[0], annotationCorrectionState.frame + 1);
    renderAnnotationCorrection();
  });
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
        const count = annotationCorrectionShape(raw)[0];
        annotationCorrectionState.frame = annotationCorrectionState.frame >= count ? 1 : annotationCorrectionState.frame + 1;
        renderAnnotationCorrection();
      }, 650);
    }
    renderAnnotationCorrection();
  });
  document.getElementById('correctionRawCanvas')?.addEventListener('click', event => annotationCorrectionCanvasClick(event, 'raw', model));
  document.getElementById('correctionProcessedCanvas')?.addEventListener('click', event => annotationCorrectionCanvasClick(event, annotationCorrectionState.processedViewId, model));
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
  annotationCorrectionDrawFrame(document.getElementById('correctionRawCanvas'), model, 'raw');
  annotationCorrectionDrawFrame(document.getElementById('correctionProcessedCanvas'), model, annotationCorrectionState.processedViewId);
  annotationCorrectionDrawTraces(document.getElementById('correctionTraceCanvas'), model);
}
