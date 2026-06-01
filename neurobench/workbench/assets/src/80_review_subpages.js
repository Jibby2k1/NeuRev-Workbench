function renderReviewStencil(){
  wireReviewStencilControls();
  if(!stencilState.initialized) {
    stencilState.initialized = true;
    loadSavedStencilIntoEditor();
  } else drawStencilCanvas();
}
function runSweepValue(run, key){
  const summary = run?.summary || {};
  if(summary[key] !== undefined) return summary[key];
  const [stage, param] = key.split('.');
  const item = (run?.sweep?.parameters || []).find(row => row.stage === stage && row.param === param);
  return item?.value;
}
function runDetailText(run){
  if(!run) return '';
  const bits = [];
  if(run.summary?.roi_count !== undefined) bits.push(`${run.summary.roi_count} ROIs`);
  if(run.summary?.event_count !== undefined) bits.push(`${run.summary.event_count} events`);
  const pfa = runSweepValue(run, 'cfar_small_ref.pfa');
  const ref = runSweepValue(run, 'cfar_large_ref.training_radius_px');
  const support = runSweepValue(run, 'components.support_min_frames');
  if(pfa !== undefined) bits.push(`pfa ${pfa}`);
  if(ref !== undefined) bits.push(`ref ${ref}`);
  if(support !== undefined) bits.push(`support ${frameDurationLabel(support)}`);
  if(run.summary?.median_equivalent_diameter_um !== undefined) bits.push(`median ${fmt(run.summary.median_equivalent_diameter_um, 1)} um`);
  return bits.join(' · ');
}
function runReviewRoisFromCache(run){
  if(!run) return [];
  if(Array.isArray(run.artifacts?.review_rois)) return run.artifacts.review_rois;
  if(runGenerated(run)) return reviewDataForRunFromCache(run)?.rois || [];
  const key = reviewRoisCacheKey(run);
  const cached = key ? reviewRoisFileCache.get(key) : null;
  return Array.isArray(cached?.rois) ? cached.rois : [];
}
function roiCenter(roi){
  return {
    x:Number(roi?.centroidX ?? roi?.x ?? roi?.center_x ?? roi?.centerX ?? 0),
    y:Number(roi?.centroidY ?? roi?.y ?? roi?.center_y ?? roi?.centerY ?? 0)
  };
}
function roiPeakZ(roi){ return Number(roi?.peakScore ?? roi?.z ?? roi?.score ?? roi?.eventSupport ?? 0); }
function roiEventSupport(roi){ return Array.isArray(roi?.events) ? roi.events.length : Number(roi?.eventSupport ?? 0); }
function gammaReviewRuns(){ return architectureRuns().filter(run => isGammaCfarRun(run) && runHasCandidateRois(run)); }
function defaultOverlapRunIds(){
  const preferred = ['gamma_cfar_cascade_grid_50hz_v2__sweep_036', 'gamma_cfar_cascade_grid_50hz_v2__sweep_033', 'gamma_cfar_cascade_grid_50hz_v2__sweep_017', 'gamma_cfar_cascade_grid_50hz_v2__sweep_020'];
  const runs = gammaReviewRuns();
  const exact = preferred.filter(id => runs.some(run => run.run_id === id));
  if(exact.length) return exact;
  return runs.slice().sort((a,b) => Number(b.summary?.roi_count || 0) - Number(a.summary?.roi_count || 0)).slice(0, 4).map(run => run.run_id);
}
function selectedOverlapRunIds(){
  const select = document.getElementById('overlapRunSelect');
  const selected = select ? [...select.selectedOptions].map(opt => opt.value) : [];
  return selected.length ? selected : (setting('overlapRunIds') || defaultOverlapRunIds());
}
function populateOverlapRunSelect(){
  const select = document.getElementById('overlapRunSelect');
  if(!select) return;
  const selected = new Set(setting('overlapRunIds') || defaultOverlapRunIds());
  select.innerHTML = gammaReviewRuns().map(run => `<option value="${escapeHtml(run.run_id)}" ${selected.has(run.run_id) ? 'selected' : ''}>${escapeHtml(runLabel(run))} · ${escapeHtml(runDetailText(run))}</option>`).join('');
  const sort = document.getElementById('overlapSortSelect');
  if(sort) sort.value = setting('overlapSortKey') || 'roi_count';
}
function loadOverlapProjection(name){
  const canvas = document.getElementById('overlapCanvas');
  if(!canvas) return;
  const image = new Image();
  overlapState.image = image;
  overlapState.imageName = name;
  image.onload = () => {
    canvas.width = image.naturalWidth || canvas.width;
    canvas.height = image.naturalHeight || canvas.height;
    drawOverlapCanvas();
  };
  image.onerror = drawOverlapCanvas;
  image.src = `${diagnosticAssetUrl(name)}?v=${Date.now()}`;
}
async function loadOverlapRuns(){
  if(overlapState.loading) return;
  overlapState.loading = true;
  const status = document.getElementById('overlapStatus');
  const ids = selectedOverlapRunIds();
  setSetting('overlapRunIds', ids);
  if(status) status.textContent = 'loading sweeps';
  const runs = ids.map(id => runById(id)).filter(Boolean);
  try {
    for(const run of runs){
      if(runGenerated(run) && !reviewDataForRunFromCache(run)) await fetchReviewDataForRun(run);
      else await ensureReviewRoisForRun(run);
    }
    overlapState.runs = runs.map((run, index) => ({run, color: REVIEW_OVERLAP_COLORS[index % REVIEW_OVERLAP_COLORS.length], rois: runReviewRoisFromCache(run)}));
    if(status) status.textContent = `${overlapState.runs.length} sweeps loaded`;
    drawOverlapCanvas();
  } catch(err) {
    if(status) status.textContent = 'could not load sweep ROIs';
    setSaveState(err.message || 'could not load sweep ROIs', 'bad');
  } finally {
    overlapState.loading = false;
  }
}
function drawOverlapCanvas(){
  const canvas = document.getElementById('overlapCanvas');
  if(!canvas) return;
  const ctx2 = canvas.getContext('2d');
  ctx2.clearRect(0, 0, canvas.width, canvas.height);
  if(overlapState.image?.complete && overlapState.image.naturalWidth) ctx2.drawImage(overlapState.image, 0, 0, canvas.width, canvas.height);
  else {
    ctx2.fillStyle = '#07111f';
    ctx2.fillRect(0, 0, canvas.width, canvas.height);
  }
  const stencilPoints = savedStencilPoints();
  if(document.getElementById('overlapShowStencil')?.checked && stencilPoints.length >= 3) drawStencilPolygon(ctx2, stencilPoints, {fill:'rgba(250, 204, 21, 0.10)', stroke:'#facc15', pointsVisible:false});
  for(const row of overlapState.runs){
    ctx2.strokeStyle = row.color;
    ctx2.fillStyle = row.color;
    for(const roi of row.rois){
      const c = roiCenter(roi);
      const radius = Math.max(3, Math.min(12, Math.sqrt(Number(roi.area || 12) / Math.PI) + 1));
      const selected = overlapState.selected?.runId === row.run.run_id && String(overlapState.selected?.roi?.id) === String(roi.id);
      ctx2.globalAlpha = selected ? 1 : 0.78;
      ctx2.lineWidth = selected ? 3 : 1.5;
      ctx2.beginPath();
      ctx2.arc(c.x, c.y, selected ? radius + 3 : radius, 0, Math.PI * 2);
      ctx2.stroke();
      if(selected){
        ctx2.globalAlpha = 0.24;
        ctx2.fill();
      }
    }
  }
  ctx2.globalAlpha = 1;
  renderOverlapCoverageSummary();
}
function overlapRowMetrics(row){
  const stencil = stencilMetricsForRois(row.rois);
  return {
    run: row.run,
    color: row.color,
    roi_count: row.rois.length,
    event_count: Number(row.run.summary?.event_count || 0),
    inside_stencil: stencil.inside + stencil.edge,
    outside_stencil: stencil.outside,
    plausible_size_fraction: Number(row.run.summary?.plausible_size_fraction || 0),
    pfa: Number(runSweepValue(row.run, 'cfar_small_ref.pfa') ?? NaN),
    ref: runSweepValue(row.run, 'cfar_large_ref.training_radius_px') ?? 'n/a',
    support: Number(runSweepValue(row.run, 'components.support_min_frames') ?? NaN)
  };
}
function sortedOverlapMetrics(){
  const key = setting('overlapSortKey') || 'roi_count';
  const rows = overlapState.runs.map(overlapRowMetrics);
  const direction = key === 'pfa' || key === 'support' ? 1 : -1;
  return rows.sort((a,b) => {
    const av = Number(a[key]);
    const bv = Number(b[key]);
    if(Number.isFinite(av) && Number.isFinite(bv) && av !== bv) return direction * (av - bv);
    return String(runLabel(a.run)).localeCompare(String(runLabel(b.run)), undefined, {numeric:true});
  });
}
function renderOverlapCoverageSummary(){
  const root = document.getElementById('overlapCoverageSummary');
  if(!root) return;
  const rows = sortedOverlapMetrics();
  if(!rows.length) {
    root.innerHTML = '<p class="hint">No sweeps loaded.</p>';
    return;
  }
  root.innerHTML = `
    <table class="smallTable overlapRunTable">
      <tr><th>Run</th><th>ROIs</th><th>Events</th><th>Stencil</th><th>Plausible</th><th>pfa</th><th>ref</th><th>support</th><th></th></tr>
      ${rows.map(row => `<tr class="${row.run.run_id === activeRunId() ? 'activeRunRow' : ''}">
        <td><span class="overlapLegendSwatch" style="background:${row.color}"></span>${escapeHtml(runLabel(row.run))}<br><span class="hint">${escapeHtml(row.run.run_id)}</span></td>
        <td>${row.roi_count}</td>
        <td>${row.event_count}</td>
        <td>${row.inside_stencil}/${row.roi_count} in/near<br><span class="hint">${row.outside_stencil} outside</span></td>
        <td>${fmt(row.plausible_size_fraction * 100, 1)}%</td>
        <td>${Number.isFinite(row.pfa) ? row.pfa : 'n/a'}</td>
        <td>${escapeHtml(row.ref)}</td>
        <td>${Number.isFinite(row.support) ? escapeHtml(frameDurationLabel(row.support)) : 'n/a'}</td>
        <td><button type="button" data-overlap-review-run="${escapeHtml(row.run.run_id)}">Review</button></td>
      </tr>`).join('')}
    </table>`;
  for(const btn of root.querySelectorAll('[data-overlap-review-run]')) btn.onclick = async () => {
    await selectActiveRun(btn.dataset.overlapReviewRun, {loadReview:false});
    location.hash = '#review';
    renderAll();
  };
}
function nearestOverlapRoi(point){
  let best = null;
  for(const row of overlapState.runs){
    for(const roi of row.rois){
      const c = roiCenter(roi);
      const dx = c.x - point.x, dy = c.y - point.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const limit = Math.max(8, Math.min(16, Math.sqrt(Number(roi.area || 12) / Math.PI) + 5));
      if(dist <= limit && (!best || dist < best.dist)) best = {run: row.run, color: row.color, roi, dist};
    }
  }
  return best;
}
function showOverlapSelection(item){
  overlapState.selected = item ? {runId:item.run.run_id, run:item.run, roi:item.roi, color:item.color} : null;
  const detail = document.getElementById('overlapSelectedDetails');
  const btn = document.getElementById('overlapOpenInReviewBtn');
  if(!detail || !btn) return;
  if(!item){
    detail.textContent = 'Click an ROI marker to inspect sweep, x, y, z, and parameters.';
    btn.disabled = true;
    drawOverlapCanvas();
    return;
  }
  const c = roiCenter(item.roi);
  const rows = [
    ['sweep', runLabel(item.run)],
    ['roi', item.roi.id],
    ['x', fmt(c.x, 1)],
    ['y', fmt(c.y, 1)],
    ['z', fmt(roiPeakZ(item.roi), 2)],
    ['area', item.roi.area ?? 'n/a'],
    ['rank', item.roi.rank ?? 'n/a'],
    ['events', roiEventSupport(item.roi)],
    ['stencil', stencilStatusLabel(roiStencilStatus(item.roi))],
    ['pfa', runSweepValue(item.run, 'cfar_small_ref.pfa') ?? 'n/a'],
    ['ref', runSweepValue(item.run, 'cfar_large_ref.training_radius_px') ?? 'n/a'],
    ['support', frameDurationLabel(runSweepValue(item.run, 'components.support_min_frames'))]
  ];
  detail.innerHTML = `<table class="overlapDetailTable">${rows.map(([k,v]) => `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(v)}</td></tr>`).join('')}</table>`;
  btn.disabled = false;
  drawOverlapCanvas();
}
function wireReviewOverlapControls(){
  if(overlapState.wired) return;
  overlapState.wired = true;
  const canvas = document.getElementById('overlapCanvas');
  const projection = document.getElementById('overlapProjectionSelect');
  projection?.addEventListener('change', () => loadOverlapProjection(projection.value));
  document.getElementById('overlapShowStencil')?.addEventListener('change', drawOverlapCanvas);
  document.getElementById('overlapSortSelect')?.addEventListener('change', ev => { setSetting('overlapSortKey', ev.target.value); renderOverlapCoverageSummary(); });
  document.getElementById('overlapLoadBtn')?.addEventListener('click', loadOverlapRuns);
  document.getElementById('overlapUseActiveBtn')?.addEventListener('click', () => {
    const select = document.getElementById('overlapRunSelect');
    if(!select) return;
    for(const opt of select.options) opt.selected = opt.value === activeRunId();
    loadOverlapRuns();
  });
  document.getElementById('overlapOpenInReviewBtn')?.addEventListener('click', async () => {
    const selected = overlapState.selected;
    if(!selected) return;
    await selectActiveRun(selected.runId, {loadReview:false});
    if(roiById(selected.roi.id)) selectRoi(selected.roi.id);
    location.hash = '#review';
    renderAll();
  });
  canvas?.addEventListener('click', ev => showOverlapSelection(nearestOverlapRoi(canvasPointFromEvent(canvas, ev))));
}
function renderReviewOverlap(){
  wireReviewOverlapControls();
  populateOverlapRunSelect();
  const projection = document.getElementById('overlapProjectionSelect');
  if(!overlapState.imageName) loadOverlapProjection(projection?.value || 'recommended_candidate_overlay.png');
  if(!overlapState.runs.length) loadOverlapRuns();
  else drawOverlapCanvas();
}

function renderReviewTriage(){
  const run = activeRun();
  const status = document.getElementById('triageStatus');
  const summaryRoot = document.getElementById('triageRunSummary');
  const gapRoot = document.getElementById('triageStencilGapPanel');
  const queueRoot = document.getElementById('triageQueuePreview');
  if(!summaryRoot || !gapRoot || !queueRoot) return;
  const loadBtn = document.getElementById('triageLoadActiveBtn');
  if(loadBtn) loadBtn.onclick = async () => {
    await ensureReviewRoisForRun(activeRun());
    try { await ensureStencilGapReportForRun(activeRun()); } catch (_) {}
    renderReviewTriage();
  };
  const insideBtn = document.getElementById('triageInsideStencilBtn');
  if(insideBtn) insideBtn.onclick = () => { setSetting('queue', 'insideStencil'); location.hash = '#review'; renderAll(); };
  const outsideBtn = document.getElementById('triageOutsideStencilBtn');
  if(outsideBtn) outsideBtn.onclick = () => { setSetting('queue', 'outsideStencil'); location.hash = '#review'; renderAll(); };
  const overlapBtn = document.getElementById('triageOpenOverlapBtn');
  if(overlapBtn) overlapBtn.onclick = () => { location.hash = '#candidate-overlay'; };
  if(!run) {
    if(status) status.textContent = 'no active sweep';
    summaryRoot.innerHTML = '<p class="hint">No active sweep is selected.</p>';
    gapRoot.innerHTML = '<p class="hint">No stencil gap report available.</p>';
    queueRoot.innerHTML = '<p class="hint">No queue to preview.</p>';
    return;
  }
  const cachedRois = runReviewRoisFromCache(run);
  if(!cachedRois.length && runHasCandidateRois(run)) {
    if(status) status.textContent = 'loading lightweight ROI summary';
    ensureReviewRoisForRun(run).then(() => {
      if(reviewSubPageFromHash() === 'triage') renderReviewTriage();
    }).catch(err => setSaveState(err.message || 'could not load ROI summary', 'bad'));
  }
  const report = stencilGapReportFromCache(run);
  if(stencilGapReportUrl(run) && !report) {
    ensureStencilGapReportForRun(run).then(() => {
      if(reviewSubPageFromHash() === 'triage') renderReviewTriage();
    }).catch(() => {});
  }
  const rois = cachedRois.length ? cachedRois : reviewRois();
  const stencil = stencilMetricsForRois(rois);
  if(status) status.textContent = `${rois.length} ROI summaries${report ? ' · stencil gaps ready' : ''}`;
  const sidecar = run.artifacts?.review_roi_summary || {};
  summaryRoot.innerHTML = `
    <div class="metricGrid">
      <div class="metric"><b>${rois.length || Number(run.summary?.roi_count || 0)}</b><span>candidate ROIs</span></div>
      <div class="metric"><b>${stencil.inside + stencil.edge}</b><span>in/near stencil</span></div>
      <div class="metric"><b>${stencil.outside}</b><span>outside stencil</span></div>
      <div class="metric"><b>${sidecar.trace_shard_count ?? 'n/a'}</b><span>lazy trace shards</span></div>
    </div>
    <p class="hint">${escapeHtml(runDetailText(run) || runLabel(run))}</p>`;
  const gaps = (report?.gaps || []).slice(0, 8);
  if(!gaps.length) {
    gapRoot.innerHTML = `<p class="hint">${report?.stencil_available === false ? 'No saved stencil was available when this sidecar was built.' : 'No low-coverage stencil boxes are available for this run.'}</p>`;
  } else {
    gapRoot.innerHTML = `
      <table class="smallTable">
        <tr><th>Gap</th><th>Center</th><th>ROIs</th><th></th></tr>
        ${gaps.map((gap, index) => `<tr>
          <td>${escapeHtml(gap.id || index + 1)}</td>
          <td>${escapeHtml((gap.center || []).map(v => fmt(v, 1)).join(', '))}</td>
          <td>${escapeHtml(gap.roi_count ?? 0)}</td>
          <td><button type="button" data-focus-gap="${index}">Inspect</button></td>
        </tr>`).join('')}
      </table>`;
    for(const btn of gapRoot.querySelectorAll('[data-focus-gap]')) btn.onclick = () => focusReviewGapBox(gaps[Number(btn.dataset.focusGap)]);
  }
  const queueRows = rois.slice().sort((a,b) => scoreValue(b, 'priorityScore', 0) - scoreValue(a, 'priorityScore', 0)).slice(0, 12);
  if(!queueRows.length) {
    queueRoot.innerHTML = '<p class="hint">No ROI summaries are loaded yet.</p>';
  } else {
    queueRoot.innerHTML = `
      <table class="smallTable">
        <tr><th>ROI</th><th>Events</th><th>Priority</th><th>Trace</th><th></th></tr>
        ${queueRows.map(roi => `<tr>
          <td>${escapeHtml(roi.id)}</td>
          <td>${eventsForRoi(roi).length}</td>
          <td>${fmt(scoreValue(roi, 'priorityScore', 0), 2)}</td>
          <td>${Array.isArray(roi.dffTrace) ? 'loaded' : roi.trace_file ? 'lazy' : 'none'}</td>
          <td><button type="button" data-open-triage-roi="${escapeHtml(roi.id)}">Review</button></td>
        </tr>`).join('')}
      </table>`;
    for(const btn of queueRoot.querySelectorAll('[data-open-triage-roi]')) btn.onclick = () => {
      location.hash = '#review';
      selectRoi(btn.dataset.openTriageRoi, false, {center:true, flash:true});
    };
  }
}
