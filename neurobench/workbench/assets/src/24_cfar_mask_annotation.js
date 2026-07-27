const CFAR_MASK_DEFAULTS = {
  cfarMaskTarget: 'foreground',
  cfarMaskTool: 'off',
  cfarMaskBrushRadius: 4,
  cfarFloodTolerance: 12,
  cfarFloodBound: 'roi_box',
  cfarFloodPadding: 12,
  cfarFloodRadius: 32,
  showCfarMasks: true
};
const CFAR_FLOOD_PIXEL_LIMIT = 50000;
const CFAR_MASK_HISTORY_LIMIT = 12;
const CFAR_MASK_COLORS = {
  foreground: {fill:'rgba(244, 63, 94, 0.46)', crop:'rgba(244, 63, 94, 0.52)', text:'#fda4af'},
  background: {fill:'rgba(14, 165, 233, 0.34)', crop:'rgba(14, 165, 233, 0.42)', text:'#7dd3fc'}
};
let cfarMaskState = {drawing:false, roiId:null, pointerId:null};
let cfarFramePixelCache = {src:'', width:0, height:0, imageData:null};
const cfarMaskUndoHistory = new Map();
const cfarFrameCanvas = document.createElement('canvas');

function cfarMaskSetting(name){
  const value = setting(name);
  return value === undefined || value === null || value === '' ? CFAR_MASK_DEFAULTS[name] : value;
}

function normalizedCfarPoints(points){
  const map = new Map();
  for(const point of points || []){
    if(!Array.isArray(point) || point.length < 2) continue;
    const x = Math.max(0, Math.min(data.video.width - 1, Math.round(Number(point[0]))));
    const y = Math.max(0, Math.min(data.video.height - 1, Math.round(Number(point[1]))));
    if(Number.isFinite(x) && Number.isFinite(y)) map.set(`${x},${y}`, [x, y]);
  }
  return [...map.values()].sort((a, b) => a[1] - b[1] || a[0] - b[0]);
}

function cfarRegionsForRoi(roi, {create=false}={}){
  if(!roi) return null;
  const id = String(roi.id);
  let ann = annotations.rois[id];
  if(!ann && create){
    ann = migrateRoiAnn({});
    annotations.rois[id] = ann;
  }
  if(!ann) return null;
  let regions = ann.cfar_regions;
  if((!regions || typeof regions !== 'object' || Array.isArray(regions)) && create){
    regions = {
      schema_version: 1,
      foreground_points: [],
      background_points: [],
      reference_frames: [],
      provenance: 'manual_cfar_feature_annotation'
    };
    ann.cfar_regions = regions;
  }
  if(!regions || typeof regions !== 'object' || Array.isArray(regions)) return null;
  if(create){
    regions.schema_version = 1;
    regions.foreground_points = normalizedCfarPoints(regions.foreground_points);
    regions.background_points = normalizedCfarPoints(regions.background_points);
    regions.reference_frames = [...new Set((regions.reference_frames || []).map(Number).filter(Number.isFinite))].sort((a,b) => a-b);
    regions.provenance = regions.provenance || 'manual_cfar_feature_annotation';
  }
  // Undo is session state, not scientific annotation data. Remove snapshots
  // written by early app builds so autosave size scales with the current mask.
  if(Object.prototype.hasOwnProperty.call(regions, 'edit_history')) delete regions.edit_history;
  return regions;
}

function cfarMaskCountsForRoi(roi){
  const regions = cfarRegionsForRoi(roi);
  return {
    foreground: Array.isArray(regions?.foreground_points) ? regions.foreground_points.length : 0,
    background: Array.isArray(regions?.background_points) ? regions.background_points.length : 0,
    frames: Array.isArray(regions?.reference_frames) ? regions.reference_frames : []
  };
}

function cfarMaskSnapshot(regions, reason){
  return {
    reason: String(reason || 'edit'),
    createdAt: new Date().toISOString(),
    foreground_bits: cfarMaskBitset(regions?.foreground_points),
    background_bits: cfarMaskBitset(regions?.background_points),
    reference_frames: [...(regions?.reference_frames || [])]
  };
}

function cfarMaskBitset(points){
  const width = Math.max(1, Number(data.video?.width) || 1);
  const height = Math.max(1, Number(data.video?.height) || 1);
  const bits = new Uint8Array(Math.ceil(width * height / 8));
  for(const [x, y] of normalizedCfarPoints(points)){
    const index = y * width + x;
    bits[index >> 3] |= 1 << (index & 7);
  }
  return bits;
}

function cfarPointsFromBitset(bits){
  const width = Math.max(1, Number(data.video?.width) || 1);
  const height = Math.max(1, Number(data.video?.height) || 1);
  const points = [];
  for(let index=0; index<width * height; index++){
    if(bits?.[index >> 3] & (1 << (index & 7))) points.push([index % width, Math.floor(index / width)]);
  }
  return points;
}

function cfarMaskHistoryKey(roi){ return `${activeRunId()}::${String(roi?.id || '')}`; }
function cfarMaskHistoryForRoi(roi, {create=false}={}){
  const key = cfarMaskHistoryKey(roi);
  if(!cfarMaskUndoHistory.has(key) && create) cfarMaskUndoHistory.set(key, []);
  return cfarMaskUndoHistory.get(key) || [];
}
function clearCfarMaskUndoHistory(){ cfarMaskUndoHistory.clear(); }

function pushCfarMaskHistory(roi, reason){
  const regions = cfarRegionsForRoi(roi, {create:true});
  if(!regions) return;
  const history = cfarMaskHistoryForRoi(roi, {create:true});
  history.push(cfarMaskSnapshot(regions, reason));
  if(history.length > CFAR_MASK_HISTORY_LIMIT) history.splice(0, history.length - CFAR_MASK_HISTORY_LIMIT);
}

function commitCfarMaskPoints(roi, target, points, metadata={}){
  const regions = cfarRegionsForRoi(roi, {create:true});
  if(!regions || !['foreground','background'].includes(target)) return null;
  const other = target === 'foreground' ? 'background' : 'foreground';
  const targetKey = `${target}_points`;
  const otherKey = `${other}_points`;
  const next = normalizedCfarPoints(points);
  const selectedKeys = new Set(next.map(point => `${point[0]},${point[1]}`));
  regions[targetKey] = next;
  regions[otherKey] = normalizedCfarPoints(regions[otherKey]).filter(point => !selectedKeys.has(`${point[0]},${point[1]}`));
  regions.reference_frames = [...new Set([...(regions.reference_frames || []), currentFrame])].sort((a,b) => a-b);
  regions.updatedAt = new Date().toISOString();
  regions.last_edit = Object.assign({
    frame: currentFrame,
    target,
    tool: cfarMaskSetting('cfarMaskTool'),
    reviewer_id: currentReviewerId() || ''
  }, metadata);
  const activeView = typeof activeLogicalVideoView === 'function' ? activeLogicalVideoView() : null;
  if(activeView) regions.last_edit.view_id = activeView.view_id;
  const ann = annotations.rois[String(roi.id)] || migrateRoiAnn({});
  ann.cfar_regions = regions;
  annotations.rois[String(roi.id)] = stampAnnotation(ann);
  queueSave();
  return regions;
}

function applyCfarMaskBrush(point){
  const roi = selectedRoi();
  const tool = cfarMaskSetting('cfarMaskTool');
  if(!roi || !['brush_add','brush_erase'].includes(tool)) return null;
  const target = cfarMaskSetting('cfarMaskTarget');
  const regions = cfarRegionsForRoi(roi, {create:true});
  const key = `${target}_points`;
  const map = pointMap(regions[key]);
  const brush = circlePoints(point.x, point.y, Number(cfarMaskSetting('cfarMaskBrushRadius')) || 4);
  if(tool === 'brush_add') for(const pixel of brush) map.set(`${pixel[0]},${pixel[1]}`, pixel);
  else for(const pixel of brush) map.delete(`${pixel[0]},${pixel[1]}`);
  const updated = commitCfarMaskPoints(roi, target, [...map.values()], {brush_radius_px:Number(cfarMaskSetting('cfarMaskBrushRadius')) || 4});
  if(updated){
    drawOverlay();
    renderCfarMaskStatus();
  }
  return updated;
}

function cfarFrameImageData(){
  const width = Math.max(1, Number(data.video?.width) || 1);
  const height = Math.max(1, Number(data.video?.height) || 1);
  const src = String(img.currentSrc || img.src || '');
  if(cfarFramePixelCache.imageData && cfarFramePixelCache.src === src && cfarFramePixelCache.width === width && cfarFramePixelCache.height === height){
    return cfarFramePixelCache.imageData;
  }
  if(!img.complete || !img.naturalWidth) throw new Error('current frame is still loading');
  cfarFrameCanvas.width = width;
  cfarFrameCanvas.height = height;
  const frameCtx = cfarFrameCanvas.getContext('2d', {willReadFrequently:true});
  frameCtx.clearRect(0, 0, width, height);
  frameCtx.drawImage(img, 0, 0, width, height);
  const imageData = frameCtx.getImageData(0, 0, width, height);
  cfarFramePixelCache = {src, width, height, imageData};
  return imageData;
}

function cfarFloodBounds(roi, seed){
  const mode = cfarMaskSetting('cfarFloodBound');
  const activeView = typeof activeLogicalVideoView === 'function' ? activeLogicalVideoView() : null;
  const viewBounds = activeView?.bounds || null;
  const inView = (x, y) => !viewBounds || (
    x >= viewBounds.x && x < viewBounds.x + viewBounds.width &&
    y >= viewBounds.y && y < viewBounds.y + viewBounds.height
  );
  if(mode === 'frame') return {mode:activeView ? 'logical_view' : mode, view_id:activeView?.view_id || '', contains:inView};
  if(mode === 'radius'){
    const radius = Math.max(1, Number(cfarMaskSetting('cfarFloodRadius')) || 32);
    const r2 = radius * radius;
    return {mode, radius, view_id:activeView?.view_id || '', contains:(x,y) => inView(x,y) && (x-seed.x)*(x-seed.x) + (y-seed.y)*(y-seed.y) <= r2};
  }
  const padding = Math.max(0, Number(cfarMaskSetting('cfarFloodPadding')) || 0);
  const bbox = Array.isArray(roi?.bbox) && roi.bbox.length >= 4 ? roi.bbox.map(Number) : geometrySummary(roi?.points || [])?.bbox;
  if(!bbox) return {mode:'radius', radius:32, view_id:activeView?.view_id || '', contains:(x,y) => inView(x,y) && (x-seed.x)*(x-seed.x) + (y-seed.y)*(y-seed.y) <= 32*32};
  const viewX1 = viewBounds ? viewBounds.x + viewBounds.width - 1 : data.video.width - 1;
  const viewY1 = viewBounds ? viewBounds.y + viewBounds.height - 1 : data.video.height - 1;
  const x0 = Math.max(viewBounds?.x || 0, Math.floor(bbox[0] - padding));
  const y0 = Math.max(viewBounds?.y || 0, Math.floor(bbox[1] - padding));
  const x1 = Math.min(viewX1, Math.ceil(bbox[2] + padding));
  const y1 = Math.min(viewY1, Math.ceil(bbox[3] + padding));
  return {mode:'roi_box', bbox:[x0,y0,x1,y1], padding, view_id:activeView?.view_id || '', contains:(x,y) => inView(x,y) && x >= x0 && x <= x1 && y >= y0 && y <= y1};
}

function connectedCfarFlood(seed, roi){
  const imageData = cfarFrameImageData();
  const width = imageData.width;
  const height = imageData.height;
  const sx = Math.max(0, Math.min(width - 1, Math.round(seed.x)));
  const sy = Math.max(0, Math.min(height - 1, Math.round(seed.y)));
  const bounds = cfarFloodBounds(roi, {x:sx,y:sy});
  if(!bounds.contains(sx, sy)) return {points:[], truncated:false, bounds};
  const tolerance = Math.max(0, Number(cfarMaskSetting('cfarFloodTolerance')) || 0);
  const pixels = imageData.data;
  const lumaAt = index => 0.2126 * pixels[index * 4] + 0.7152 * pixels[index * 4 + 1] + 0.0722 * pixels[index * 4 + 2];
  const seedIndex = sy * width + sx;
  const seedLuma = lumaAt(seedIndex);
  const total = width * height;
  const queueLimit = Math.min(total, CFAR_FLOOD_PIXEL_LIMIT);
  const queue = new Int32Array(queueLimit);
  const visited = new Uint8Array(total);
  let head = 0, tail = 0, truncated = false;
  queue[tail++] = seedIndex;
  visited[seedIndex] = 1;
  const points = [];
  const enqueue = (x, y) => {
    if(x < 0 || y < 0 || x >= width || y >= height || !bounds.contains(x,y)) return;
    const index = y * width + x;
    if(visited[index]) return;
    visited[index] = 1;
    if(tail >= queue.length){ truncated = true; return; }
    queue[tail++] = index;
  };
  while(head < tail && points.length < CFAR_FLOOD_PIXEL_LIMIT){
    const index = queue[head++];
    if(Math.abs(lumaAt(index) - seedLuma) > tolerance) continue;
    const x = index % width;
    const y = Math.floor(index / width);
    points.push([x,y]);
    enqueue(x-1,y); enqueue(x+1,y); enqueue(x,y-1); enqueue(x,y+1);
  }
  if(points.length >= CFAR_FLOOD_PIXEL_LIMIT) truncated = true;
  return {points, truncated, bounds, tolerance, seed_luma:Number(seedLuma.toFixed(3))};
}

function applyCfarMaskFlood(point){
  const roi = selectedRoi();
  const tool = cfarMaskSetting('cfarMaskTool');
  if(!roi || !['flood_add','flood_erase'].includes(tool)) return null;
  let result;
  try {
    result = connectedCfarFlood(point, roi);
  } catch (error) {
    setSaveState(`flood fill unavailable: ${error.message || error}`, 'bad');
    return null;
  }
  if(!result.points.length){
    setSaveState('flood seed did not match pixels inside the active bound', 'bad');
    return null;
  }
  pushCfarMaskHistory(roi, `before ${tool}`);
  const target = cfarMaskSetting('cfarMaskTarget');
  const regions = cfarRegionsForRoi(roi, {create:true});
  const key = `${target}_points`;
  const map = pointMap(regions[key]);
  if(tool === 'flood_add') for(const pixel of result.points) map.set(`${pixel[0]},${pixel[1]}`, pixel);
  else for(const pixel of result.points) map.delete(`${pixel[0]},${pixel[1]}`);
  const updated = commitCfarMaskPoints(roi, target, [...map.values()], {
    flood_tolerance: result.tolerance,
    flood_bound: result.bounds.mode,
    flood_seed: [Math.round(point.x), Math.round(point.y)],
    flood_seed_luma: result.seed_luma,
    flood_pixels: result.points.length,
    flood_truncated: result.truncated
  });
  if(updated){
    recordAction(`cfar_${target}_${tool}`);
    renderAll();
    setSaveState(`${tool === 'flood_add' ? 'selected' : 'deselected'} ${result.points.length} ${target} pixels${result.truncated ? ' (pixel cap reached)' : ''}`, result.truncated ? 'bad' : 'ok');
  }
  return updated;
}

function undoCfarMaskEdit(){
  const roi = selectedRoi();
  const regions = cfarRegionsForRoi(roi, {create:false});
  const history = cfarMaskHistoryForRoi(roi);
  if(!roi || !regions || !history.length){
    setSaveState('no CFAR mask history for selected ROI', 'bad');
    return;
  }
  const snapshot = history.pop();
  regions.foreground_points = cfarPointsFromBitset(snapshot.foreground_bits);
  regions.background_points = cfarPointsFromBitset(snapshot.background_bits);
  regions.reference_frames = [...(snapshot.reference_frames || [])];
  regions.updatedAt = new Date().toISOString();
  regions.last_edit = {frame:currentFrame, tool:'undo', target:cfarMaskSetting('cfarMaskTarget'), reviewer_id:currentReviewerId() || ''};
  annotations.rois[String(roi.id)] = stampAnnotation(Object.assign(roiAnn(roi.id), {cfar_regions:regions}));
  recordAction('cfar_mask_undo');
  queueSave();
  renderAll();
  setSaveState(`restored previous CFAR masks for ROI ${roi.id}`, 'ok');
}

function clearActiveCfarMask(){
  const roi = selectedRoi();
  if(!roi) return;
  const target = cfarMaskSetting('cfarMaskTarget');
  const regions = cfarRegionsForRoi(roi, {create:true});
  if(!(regions[`${target}_points`] || []).length){
    setSaveState(`${target} mask is already empty`, 'bad');
    return;
  }
  pushCfarMaskHistory(roi, `before clear ${target}`);
  commitCfarMaskPoints(roi, target, [], {tool:'clear'});
  recordAction(`cfar_${target}_clear`);
  renderAll();
  setSaveState(`cleared ${target} mask for ROI ${roi.id}; Undo CFAR can restore it`, 'ok');
}

function drawSelectedCfarMasks(){
  const roi = selectedRoi();
  const regions = cfarRegionsForRoi(roi);
  renderCfarMaskStatus();
  if(cfarMaskSetting('showCfarMasks') === false || !roi || !regions) return;
  ctx.save();
  for(const target of ['background','foreground']){
    const points = regions[`${target}_points`] || [];
    if(!points.length) continue;
    ctx.fillStyle = CFAR_MASK_COLORS[target].fill;
    for(const point of points) ctx.fillRect(point[0], point[1], 1, 1);
  }
  const counts = cfarMaskCountsForRoi(roi);
  if(counts.foreground || counts.background){
    drawOverlayLabel(`FG ${counts.foreground} · BG ${counts.background}`, roi.centroidX + 5, roi.centroidY + 18, '#ffffff');
  }
  ctx.restore();
  renderCfarMaskStatus();
}

function drawCfarMasksOnCrop(cropContext, roi, bounds, offsetX, offsetY, scale){
  if(cfarMaskSetting('showCfarMasks') === false) return;
  const regions = cfarRegionsForRoi(roi);
  if(!regions) return;
  cropContext.save();
  for(const target of ['background','foreground']){
    cropContext.fillStyle = CFAR_MASK_COLORS[target].crop;
    for(const point of regions[`${target}_points`] || []){
      if(point[0] < bounds.x0 || point[0] > bounds.x1 || point[1] < bounds.y0 || point[1] > bounds.y1) continue;
      cropContext.fillRect(offsetX + (point[0] - bounds.x0) * scale, offsetY + (point[1] - bounds.y0) * scale, Math.max(1, scale), Math.max(1, scale));
    }
  }
  cropContext.restore();
}

function renderCfarMaskStatus(){
  const status = document.getElementById('cfarMaskStatus');
  if(!status) return;
  const roi = selectedRoi();
  if(!roi){
    status.textContent = 'Select an ROI before annotating CFAR regions.';
    return;
  }
  const counts = cfarMaskCountsForRoi(roi);
  status.textContent = `ROI ${roi.id}: foreground ${counts.foreground} px · background ${counts.background} px${counts.frames.length ? ` · frames ${counts.frames.join(', ')}` : ''}`;
}

function syncCfarMaskControls(){
  for(const id of ['cfarMaskTarget','cfarMaskTool','cfarFloodBound']){
    const el = document.getElementById(id);
    if(el) el.value = cfarMaskSetting(id);
  }
  for(const [id, digits] of [['cfarMaskBrushRadius',0],['cfarFloodTolerance',0],['cfarFloodPadding',0],['cfarFloodRadius',0]]){
    const el = document.getElementById(id);
    if(el) el.value = cfarMaskSetting(id);
    const label = document.getElementById(`${id}Label`);
    if(label) label.textContent = Number(cfarMaskSetting(id)).toFixed(digits);
  }
  const visible = document.getElementById('showCfarMasks');
  if(visible) visible.checked = cfarMaskSetting('showCfarMasks') !== false;
  const bound = cfarMaskSetting('cfarFloodBound');
  document.getElementById('cfarFloodPaddingWrap')?.classList.toggle('hidden', bound !== 'roi_box');
  document.getElementById('cfarFloodRadiusWrap')?.classList.toggle('hidden', bound !== 'radius');
  renderCfarMaskStatus();
}

function stopCfarPointerEvent(event){
  event.preventDefault();
  event.stopImmediatePropagation();
}

function cfarMaskPointerDown(event){
  const tool = cfarMaskSetting('cfarMaskTool');
  if(tool === 'off') return;
  stopCfarPointerEvent(event);
  const roi = selectedRoi();
  if(!roi){
    setSaveState('select an ROI before annotating CFAR foreground/background', 'bad');
    return;
  }
  const point = overlayPointFromEvent(event);
  if(tool === 'brush_add' || tool === 'brush_erase'){
    pushCfarMaskHistory(roi, `before ${tool}`);
    cfarMaskState = {drawing:true, roiId:String(roi.id), pointerId:event.pointerId};
    overlay.setPointerCapture?.(event.pointerId);
    applyCfarMaskBrush(point);
  } else {
    applyCfarMaskFlood(point);
  }
}

function cfarMaskPointerMove(event){
  if(!cfarMaskState.drawing || cfarMaskSetting('cfarMaskTool') === 'off') return;
  stopCfarPointerEvent(event);
  if(String(selectedRoi()?.id) !== cfarMaskState.roiId) return;
  applyCfarMaskBrush(overlayPointFromEvent(event));
}

function cfarMaskPointerUp(event){
  if(!cfarMaskState.drawing) return;
  stopCfarPointerEvent(event);
  overlay.releasePointerCapture?.(event.pointerId);
  const target = cfarMaskSetting('cfarMaskTarget');
  const tool = cfarMaskSetting('cfarMaskTool');
  cfarMaskState = {drawing:false, roiId:null, pointerId:null};
  recordAction(`cfar_${target}_${tool}`);
  queueSave();
  renderAll();
}

function initCfarMaskAnnotation(){
  const required = ['cfarMaskTarget','cfarMaskTool','cfarMaskBrushRadius','cfarFloodTolerance','cfarFloodBound','cfarFloodPadding','cfarFloodRadius','showCfarMasks','cfarMaskUndoBtn','cfarMaskClearBtn','cfarMaskDoneBtn'];
  if(required.some(id => !document.getElementById(id))) return;
  document.getElementById('cfarMaskTarget').onchange = event => { setSetting('cfarMaskTarget', event.target.value); syncCfarMaskControls(); drawOverlay(); };
  document.getElementById('cfarMaskTool').onchange = event => {
    setAnnotationToolModes({
      manualRoiMode:'select',
      roiEditMode:'off',
      cfarMaskTool:event.target.value,
      cfarMaskTarget:cfarMaskSetting('cfarMaskTarget')
    }, {render:false, forceSingle:event.target.value !== 'off'});
    drawOverlay();
  };
  document.getElementById('cfarFloodBound').onchange = event => { setSetting('cfarFloodBound', event.target.value); syncCfarMaskControls(); };
  for(const id of ['cfarMaskBrushRadius','cfarFloodTolerance','cfarFloodPadding','cfarFloodRadius']){
    document.getElementById(id).oninput = event => { setSetting(id, Number(event.target.value)); syncCfarMaskControls(); };
  }
  document.getElementById('showCfarMasks').onchange = event => { setSetting('showCfarMasks', event.target.checked); drawOverlay(); drawCrop(); };
  document.getElementById('cfarMaskUndoBtn').onclick = undoCfarMaskEdit;
  document.getElementById('cfarMaskClearBtn').onclick = clearActiveCfarMask;
  document.getElementById('cfarMaskDoneBtn').onclick = () => {
    setAnnotationToolModes({manualRoiMode:'select', roiEditMode:'off', cfarMaskTool:'off'}, {render:false});
    drawOverlay();
  };
  overlay.addEventListener('pointerdown', cfarMaskPointerDown, true);
  overlay.addEventListener('pointermove', cfarMaskPointerMove, true);
  overlay.addEventListener('pointerup', cfarMaskPointerUp, true);
  overlay.addEventListener('pointercancel', cfarMaskPointerUp, true);
  overlay.addEventListener('click', event => {
    if(cfarMaskSetting('cfarMaskTool') !== 'off') stopCfarPointerEvent(event);
  }, true);
  syncCfarMaskControls();
}
