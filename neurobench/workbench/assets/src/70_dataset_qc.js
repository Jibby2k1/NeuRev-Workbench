let dataCompareFocusedFrameApplied = false;

function sweepEvidenceRunFor(run){
  const report = sweepEvidenceReportFromCache();
  if(!report || !run) return null;
  return (report.runs || []).find(row => row.run_id === run.run_id) || null;
}
function sweepEvidenceRecommendedRows(report){
  return (report?.recommended_runs || []).slice(0, 5).map((row, index) => `
    <tr class="${row.run_id === activeRunId() ? 'activeRunRow' : ''}">
      <td>${index + 1}</td>
      <td>${escapeHtml(row.run_id || '')}</td>
      <td>${fmt(row.evidence_score, 1)}</td>
      <td>${row.roi_count ?? 0}</td>
      <td>${fmt(row.stencil_coverage_fraction, 3)}</td>
      <td>${fmt(row.stable_roi_fraction, 3)}</td>
    </tr>`).join('');
}
function scheduleSweepEvidenceRerender(){
  const subpage = dataSubPageFromHash();
  if(subpage === 'compare') renderDataCompare();
  else renderDatasetQc();
}
function renderSweepEvidencePanel(run){
  const url = sweepEvidenceReportUrl();
  if(!url) return `
    <section class="archCard sweepEvidencePanel">
      <div class="runCardHeader"><h3>Sweep Evidence Report</h3><span class="stageStatus off">not built</span></div>
      <p class="hint">Run tools/build_sweep_evidence_report.py for automated sweep ranking, stencil coverage, stability, and failure diagnostics.</p>
    </section>`;
  const report = sweepEvidenceReportFromCache();
  if(!report){
    ensureSweepEvidenceReport().then(scheduleSweepEvidenceRerender).catch(err => setSaveState(err.message || 'sweep evidence report did not load', 'bad'));
    return `
      <section class="archCard sweepEvidencePanel">
        <div class="runCardHeader"><h3>Sweep Evidence Report</h3><span class="stageStatus warn">loading</span></div>
        <p class="hint">Loading automated sweep evidence from ${escapeHtml(url)}.</p>
      </section>`;
  }
  const row = sweepEvidenceRunFor(run);
  const summary = report.summary || {};
  const recRows = sweepEvidenceRecommendedRows(report);
  const diagnostics = (row?.diagnostics || []).map(item => `<span class="stageIssueBadge">${escapeHtml(item.code || item)}</span>`).join('') || '<span class="stageStatus ok">no automated warnings</span>';
  const markdown = data.architectureRuns?.artifacts?.sweep_evidence_markdown ? artifactUrl(data.architectureRuns.artifacts.sweep_evidence_markdown) : '';
  const runMetrics = row ? `
    <div class="metricGrid">
      <div class="metric"><b>${fmt(row.evidence_score, 1)}</b><span>evidence score</span></div>
      <div class="metric"><b>${row.roi_count ?? 0}</b><span>candidate ROIs</span></div>
      <div class="metric"><b>${fmt(row.stencil_coverage_fraction, 3)}</b><span>stencil coverage</span></div>
      <div class="metric"><b>${fmt(row.stable_roi_fraction, 3)}</b><span>stable ROI fraction</span></div>
      <div class="metric"><b>${row.stencil_gap_report?.zero_roi_gap_count ?? 'n/a'}</b><span>zero-ROI gaps</span></div>
      <div class="metric"><b>${row.contrast_maps?.length || 0}</b><span>contrast maps</span></div>
    </div>` : `<p class="hint">The active run is not present in the report; use the recommendation table for sweep-level context.</p>`;
  return `
    <section class="archCard sweepEvidencePanel">
      <div class="runCardHeader">
        <div>
          <h3>Sweep Evidence Report</h3>
          <p class="hint">Automated ranking from stencil coverage, cross-sweep stability, events, contrast-map availability, and gap reports.</p>
        </div>
        <span class="stageStatus ok">${summary.analyzed_run_count || 0} runs</span>
      </div>
      ${runMetrics}
      <div class="miniChipRow">${diagnostics}</div>
      <details>
        <summary>Top recommended sweeps</summary>
        <table class="smallTable"><tr><th>#</th><th>Run</th><th>Score</th><th>ROIs</th><th>Stencil</th><th>Stable</th></tr>${recRows || '<tr><td colspan="6">No recommendations available.</td></tr>'}</table>
      </details>
      <div class="buttonRow">
        ${markdown ? `<a class="buttonLink" href="${escapeHtml(markdown)}" target="_blank" rel="noreferrer">Open Markdown Report</a>` : ''}
        <a class="buttonLink" href="${escapeHtml(url)}" target="_blank" rel="noreferrer">Open JSON Report</a>
      </div>
    </section>`;
}


function templateGridArtifactsForRun(run){
  const payload = typeof templateGridPayload === 'function' ? templateGridPayload() : {};
  const groups = [run?.artifacts, run?.outputs, run?.intermediates, run?.intermediate_artifacts, payload.artifacts, payload.outputs, payload.files, data.architectureRuns?.artifacts];
  return groups.flatMap(group => Array.isArray(group) ? group : []).filter(Boolean);
}
function findTemplateGridArtifact(run, needles){
  const wanted = needles.map(value => String(value).toLowerCase());
  return templateGridArtifactsForRun(run).find(item => {
    const text = [item.artifact_kind, item.kind, item.type, item.id, item.stage_id, item.step_id, item.label, item.name, item.file, item.path].map(v => String(v || '').toLowerCase()).join(' ');
    return wanted.some(needle => text.includes(needle));
  }) || null;
}
function templateGridArtifactPath(item){
  if(!item) return '';
  const raw = item.file || item.path || item.url || item.href || '';
  return raw && typeof artifactUrl === 'function' ? artifactUrl(raw) : raw;
}
function templateGridCountsHtml(counts){
  const entries = Object.entries(counts || {});
  if(!entries.length) return '<span class="hint">label counts not recorded</span>';
  return entries.map(([label, count]) => `<span class="utilityPill">${escapeHtml(label)}: ${escapeHtml(count)}</span>`).join('');
}
function templateGridMetric(source, keys){
  if(!source) return 'n/a';
  for(const key of keys) {
    const value = source[key] ?? source.metrics?.[key] ?? source.summary?.[key];
    if(value !== undefined && value !== null && value !== '') return typeof value === 'number' ? fmt(value, Math.abs(value) >= 10 ? 2 : 5) : escapeHtml(String(value));
  }
  return 'n/a';
}
function templateGridPreviewCard(label, artifact){
  const path = templateGridArtifactPath(artifact);
  if(!path) return `<article><div class="qcStageMissing">${escapeHtml(label)} not linked</div><span>${escapeHtml(label)}</span></article>`;
  const options = artifact?.input_options || artifact?.inputOptions || artifact?.video_options || artifact?.videoOptions || [];
  const optionCount = Array.isArray(options) ? options.length : 0;
  const isSelector = optionCount > 0;
  const isImage = /\.(png|jpg|jpeg|gif|webp|svg)(\?|$)/i.test(path);
  const action = isSelector ? 'Open input selector' : 'Open artifact';
  const meta = isSelector ? `<small>${escapeHtml(optionCount)} inputs</small>` : '';
  return `<article>${isImage ? `<img src="${escapeHtml(path)}" alt="${escapeHtml(label)}">` : `<a class="buttonLink" href="${escapeHtml(path)}" target="_blank" rel="noreferrer">${action}</a>${meta}`}<span>${escapeHtml(label)}</span></article>`;
}

function templateGridSweepPayload(){
  const payload = typeof templateGridPayload === 'function' ? templateGridPayload() : {};
  return payload.overnight_sweep || payload.overnightSweep || payload.sweep_visuals || payload.sweepVisuals || payload.dynamics_sweep || payload.dynamicsSweep || {};
}
function renderTemplateGridSweepPanel(run, options={}){
  const sweep = templateGridSweepPayload();
  const artifacts = Array.isArray(sweep.artifacts) ? sweep.artifacts : [];
  if(!sweep.experiment_count && !artifacts.length) return '';
  const best = sweep.best_validation_and_test?.experiment_id ? sweep.best_validation_and_test : sweep.best_validation || {};
  const topRows = (sweep.top_experiments || []).slice(0, options.compact ? 5 : 8).map(row => `
    <tr>
      <td>${escapeHtml(row.rank ?? '')}</td>
      <td>${escapeHtml(row.dataset_key || '')}</td>
      <td>${escapeHtml(row.kind || '')}</td>
      <td>${escapeHtml(row.seed ?? '')}</td>
      <td>${fmt(Number(row.val_improvement_over_persistence_mse || 0), 7)}</td>
      <td>${fmt(Number(row.test_improvement_over_persistence_mse || 0), 7)}</td>
    </tr>`).join('');
  const chartCards = artifacts.map(item => templateGridPreviewCard(item.label || item.id || 'sweep chart', item)).join('');
  const content = `
      <div class="runCardHeader">
        <h3>Overnight Dynamics Sweep</h3>
        <span class="runStatus">${escapeHtml(sweep.experiment_count || 0)} experiments</span>
      </div>
      <p class="hint">Cross-validated dynamics sweep against split-aware persistence. Positive improvement means lower MSE than persistence on that split.</p>
      <div class="metricGrid">
        <div class="metric"><b>${escapeHtml(sweep.positive_validation_count ?? 'n/a')}</b><span>positive validation</span></div>
        <div class="metric"><b>${escapeHtml(sweep.positive_test_count ?? 'n/a')}</b><span>positive test</span></div>
        <div class="metric"><b>${escapeHtml(sweep.positive_validation_and_test_count ?? 'n/a')}</b><span>positive both</span></div>
        <div class="metric"><b>${escapeHtml(best.dataset_key || 'n/a')}</b><span>best dataset</span></div>
        <div class="metric"><b>${fmt(Number(best.val_improvement_over_persistence_mse || 0), 7)}</b><span>best val improvement</span></div>
        <div class="metric"><b>${fmt(Number(best.test_improvement_over_persistence_mse || 0), 7)}</b><span>best test improvement</span></div>
      </div>
      <div class="templateGridPreviewStrip sweepVisualStrip">${chartCards || '<p class="hint">No sweep visual artifacts linked.</p>'}</div>
      <details class="sweepTopTable">
        <summary>Top sweep experiments</summary>
        <table class="smallTable"><tr><th>Rank</th><th>Dataset</th><th>Kind</th><th>Seed</th><th>Val improvement</th><th>Test improvement</th></tr>${topRows || '<tr><td colspan="6">No sweep rows available.</td></tr>'}</table>
      </details>`;
  return options.standalone ? `<section class="archCard templateGridSweepPanel">${content}</section>` : `<div class="templateGridSweepPanel">${content}</div>`;
}

function renderTemplateGridPanel(run){
  const payload = typeof templateGridPayload === 'function' ? templateGridPayload() : {};
  const spec = typeof templateGridSpec === 'function' ? templateGridSpec() : {};
  const dims = typeof templateGridDimensions === 'function' ? templateGridDimensions() : {rows: 32, cols: 32};
  const manifest = payload.video_manifest || payload.videoManifest || payload.manifest || {};
  const labelCounts = manifest.label_counts || manifest.labelCounts || payload.label_counts || payload.labelCounts || {};
  const registrationWarnings = payload.registration_warnings || payload.registrationWarnings || payload.registration?.warnings || [];
  const baseline = payload.persistence_baseline || payload.persistenceBaseline || payload.baseline_metrics || payload.baseline || {};
  const autoencoder = payload.autoencoder_run || payload.autoencoderRun || payload.autoencoder || payload.ae || {};
  const rnn = payload.latent_rnn_run || payload.latentRnnRun || payload.latent_rnn || payload.rnn || {};
  const classifier = payload.latent_classifier_run || payload.latentClassifierRun || payload.latent_classifier || payload.classifier || {};
  const dataset = payload.dynamics_dataset || payload.dynamicsDataset || {};
  const splitUnit = dataset.split_unit || dataset.splitUnit || payload.split_unit || payload.splitUnit || 'video';
  const stateCount = payload.grid_state_count ?? payload.gridStateCount ?? payload.grid_states?.length ?? payload.gridStates?.length ?? 'n/a';
  const templateProjection = findTemplateGridArtifact(run, ['template_projection', 'template projection', 'template_preview']);
  const registeredProjection = findTemplateGridArtifact(run, ['registered_projection', 'registration_overlay', 'registered preview']);
  const gridPreview = findTemplateGridArtifact(run, ['grid_preview', 'grid overlay', 'grid_spec']);
  return `
    <section class="archCard templateGridPanel" id="templateGridPanel">
      <div class="runCardHeader">
        <h3>Template / Registration / Grid</h3>
        <span class="runStatus">${escapeHtml(dims.rows)}x${escapeHtml(dims.cols)} grid</span>
      </div>
      <p class="hint">Template-aligned grid dynamics workflow: video manifest, per-video rigid registration, grid states, video-level splits, autoencoder, latent RNN, and latent classifier.</p>
      <div class="metricGrid">
        <div class="metric"><b>${escapeHtml(manifest.video_count ?? manifest.videos?.length ?? 'n/a')}</b><span>video manifest</span></div>
        <div class="metric"><b>${escapeHtml(Object.values(labelCounts || {}).reduce((a,b) => a + Number(b || 0), 0) || 'n/a')}</b><span>label counts</span></div>
        <div class="metric"><b>${escapeHtml(registrationWarnings.length || 0)}</b><span>registration warnings</span></div>
        <div class="metric"><b>${escapeHtml(stateCount)}</b><span>grid states</span></div>
        <div class="metric"><b>${escapeHtml(splitUnit)}</b><span>split unit: video</span></div>
        <div class="metric"><b>${templateGridMetric(baseline, ['mse','test_mse','mean_squared_error'])}</b><span>Persistence baseline</span></div>
        <div class="metric"><b>${templateGridMetric(autoencoder, ['reconstruction_mse','valid_reconstruction_mse','valid_loss','test_mse'])}</b><span>Autoencoder</span></div>
        <div class="metric"><b>${templateGridMetric(rnn, ['prediction_mse','valid_mse','test_mse','baseline_ratio'])}</b><span>Latent RNN</span></div>
        <div class="metric"><b>${templateGridMetric(classifier, ['accuracy','test_accuracy','balanced_accuracy'])}</b><span>Latent classifier</span></div>
      </div>
      <div class="miniChipRow">${templateGridCountsHtml(labelCounts)}</div>
      <div class="templateGridPreviewStrip">
        ${templateGridPreviewCard('template projection', templateProjection)}
        ${templateGridPreviewCard('registered projection', registeredProjection)}
        ${templateGridPreviewCard('grid specification', gridPreview)}
      </div>
      ${renderTemplateGridSweepPanel(run)}
    </section>`;
}
function templateGridReportRows(){
  const payload = typeof templateGridPayload === 'function' ? templateGridPayload() : {};
  const spec = typeof templateGridSpec === 'function' ? templateGridSpec() : {};
  const dims = typeof templateGridDimensions === 'function' ? templateGridDimensions() : {rows: spec.rows || 32, cols: spec.cols || 32};
  const manifest = payload.video_manifest || payload.videoManifest || payload.manifest || {};
  const dataset = payload.dynamics_dataset || payload.dynamicsDataset || {};
  const baseline = payload.persistence_baseline || payload.persistenceBaseline || payload.baseline_metrics || payload.baseline || {};
  const autoencoder = payload.autoencoder_run || payload.autoencoderRun || payload.autoencoder || {};
  const rnn = payload.latent_rnn_run || payload.latentRnnRun || payload.latent_rnn || {};
  const classifier = payload.latent_classifier_run || payload.latentClassifierRun || payload.latent_classifier || {};
  return [
    ['Dataset manifest', `${manifest.video_count ?? manifest.videos?.length ?? 'n/a'} videos; labels ${Object.keys(manifest.label_counts || manifest.labelCounts || payload.label_counts || {}).join(', ') || 'n/a'}`],
    ['Template construction', `${dims.rows || 32}x${dims.cols || 32} grid aligned to a reference projection`],
    ['Registration summary', `${(payload.registration_warnings || payload.registrationWarnings || []).length || 0} warnings recorded`],
    ['Grid extraction summary', `${payload.grid_state_count ?? payload.gridStateCount ?? payload.grid_states?.length ?? payload.gridStates?.length ?? 'n/a'} grid-state artifacts`],
    ['Autoencoder reconstruction metrics', templateGridMetric(autoencoder, ['reconstruction_mse','valid_reconstruction_mse','valid_loss','test_mse'])],
    ['Latent RNN prediction metrics', templateGridMetric(rnn, ['prediction_mse','valid_mse','test_mse','baseline_ratio'])],
    ['Persistence baseline comparison', templateGridMetric(baseline, ['mse','test_mse','mean_squared_error'])],
    ['Latent classifier metrics', templateGridMetric(classifier, ['accuracy','test_accuracy','balanced_accuracy'])],
    ['Known limitations', `Template-aligned grid dynamics only; split unit: ${dataset.split_unit || dataset.splitUnit || payload.split_unit || 'video'}; no inverse control/stimulation or transformer modeling.`]
  ];
}
function renderTemplateGridReportSummary(){
  const rows = templateGridReportRows();
  return `
    <section class="archCard templateGridReportSummary" id="templateGridReportSummary">
      <div class="runCardHeader"><h3>Template / Registration / Grid Summary</h3><span class="runStatus">experiment handoff</span></div>
      <table class="smallTable"><tbody>${rows.map(([label, value]) => `<tr><td>${escapeHtml(label)}</td><td>${escapeHtml(value)}</td></tr>`).join('')}</tbody></table>
    </section>`;
}
function templateGridReportMarkdownLines(){
  const lines = ['', '## Template / Registration / Grid Experiment', ''];
  for(const [label, value] of templateGridReportRows()) lines.push(`- ${label}: ${value}`);
  return lines;
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
  const runRoisForStencil = runReviewRoisFromCache(run).length ? runReviewRoisFromCache(run) : data.rois;
  const stencilMetrics = stencilMetricsForRois(runRoisForStencil);
  const driftMax = Number(data.qc?.driftStats?.maxMagnitudePx);
  const satMax = Number(data.qc?.saturationStats?.maxFraction);
  const warnings = [];
  if(data.rois.length && quantile(diamPx, 0.5) < 8) warnings.push('Median ROI footprint is small in pixels; the detector may be capturing active cores or fragments rather than full somata.');
  if(Number.isFinite(pixelSize) && quantile(diamUm, 0.5) < 5) warnings.push('Median equivalent ROI diameter is below 5 microns with the configured pixel size.');
  if(suggestions.length > data.rois.length) warnings.push('Discovery suggestions outnumber current ROIs; review missed-neuron coverage before tightening thresholds.');
  if(artifactCueCount > suggestions.length * 0.25) warnings.push('Many discovery suggestions have artifact cues; inspect evidence maps for vessels, borders, or bright static structures.');
  if(Number.isFinite(driftMax) && driftMax >= 2) warnings.push('Estimated rigid drift exceeds 2 px; compare raw candidates against motion-sensitive evidence before accepting weak traces.');
  if(Number.isFinite(satMax) && satMax > 0.001) warnings.push('Saturation-like bright pixels appear in the frame stack; inspect raw max and artifact-risk candidates.');
  if(savedStencilPoints().length < 3) warnings.push('No anatomy stencil is saved yet; draw a rough hindbrain region in Review > Stencil before using stencil-aware coverage metrics.');
  else if(stencilMetrics.total && stencilMetrics.inside + stencilMetrics.edge === 0) warnings.push('No loaded candidate ROI centers fall inside or near the anatomy stencil; inspect Review > Overlap before trusting this sweep.');
  if(!Number.isFinite(pixelSize)) warnings.push('Pixel size is not set in the dataset manifest, so physical-size QC is disabled.');
  const qcWarnings = warnings.map(w => `<div class="qcWarning">${w}</div>`).join('') || '<div class="qcWarning">No QC warnings from the current lightweight checks.</div>';
  const maps = (data.discovery?.evidenceMaps || []).map(m => `
    <div class="qcMap">
      <img src="${m.file}" alt="${m.label}">
      <p class="hint">${m.label}</p>
    </div>`).join('');
  const runOptions = processRunOptionsHtml(runs, run);
  const evidenceOptions = mapsForRun.map(m => `<option value="${escapeHtml(m.id || m.label)}" ${(setting('qcEvidenceMap') || '') === (m.id || m.label) ? 'selected' : ''}>${escapeHtml(m.label || m.id || 'evidence map')}</option>`).join('');
  root.innerHTML = `
    ${renderRunSummaryCards(run)}
    ${gammaCfarQuickPickHtml(runs, run)}
    ${renderSweepEvidencePanel(run)}
    ${renderTemplateGridPanel(run)}
    <div class="qcWorkbench">
      <section class="qcViewerPanel">
        <div class="toolbar">
          <button id="qcPlayBtn">Play</button>
          <button id="qcPrevFrameBtn">Prev</button>
          <button id="qcNextFrameBtn">Next</button>
          <label>Frame <input id="qcFrameSlider" type="range" min="1" max="${data.video.frames}" value="${currentFrame}"></label>
          <b id="qcFrameLabel">${frameLabelText(currentFrame)} / ${data.video.frames}</b>
          <label>Tile size
            <select id="qcTileSize">
              <option value="medium" ${(setting('qcTileSize') || 'medium') === 'medium' ? 'selected' : ''}>Medium</option>
              <option value="large" ${setting('qcTileSize') === 'large' ? 'selected' : ''}>Large</option>
              <option value="xlarge" ${setting('qcTileSize') === 'xlarge' ? 'selected' : ''}>X-Large</option>
              <option value="compact" ${setting('qcTileSize') === 'compact' ? 'selected' : ''}>Compact</option>
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
        <p class="hint">Data follows the active pipeline run, so raw frames, intermediate outputs, and warnings stay in pipeline order.</p>
        <div id="qcPipelineTimeline"></div>
      </section>
    </div>
    ${processInsightPanel(run)}
    ${renderProcessDecisionSupport(run)}
    <div class="metricGrid">
      <div class="metric"><b>${data.video.width} x ${data.video.height}</b><span>frame size</span></div>
      <div class="metric"><b>${data.video.frames}</b><span>frames (${formatSeconds(data.video.frames / Math.max(1, datasetFrameRateHz()))})</span></div>
      <div class="metric"><b>${fmt(datasetFrameRateHz(), 1)} Hz</b><span>frame rate</span></div>
      <div class="metric"><b>${data.rois.length}</b><span>baseline candidate ROIs</span></div>
      <div class="metric"><b>${stencilMetrics.inside + stencilMetrics.edge}/${stencilMetrics.total}</b><span>ROIs in/near stencil</span></div>
      <div class="metric"><b>${stencilMetrics.outside}</b><span>ROIs outside stencil</span></div>
      <div class="metric"><b>${stencilMetrics.events_inside + stencilMetrics.events_edge}</b><span>events in/near stencil</span></div>
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

function dataSubPageFromHash(hashText=location.hash){
  const hash = (hashText || '#data').replace(/^#\/?/, '');
  if(hash === 'data-compare' || hash === 'compare' || hash === 'process-compare') return 'compare';
  return 'inspect';
}
function dataPageLabel(subpage){ return subpage === 'compare' ? 'Data Compare' : 'Data Inspect'; }
function updateDataSubnav(subpage){
  document.getElementById('dataInspectSubtab')?.classList.toggle('active', subpage === 'inspect');
  document.getElementById('dataCompareSubtab')?.classList.toggle('active', subpage === 'compare');
  document.getElementById('datasetQc')?.classList.toggle('hidden', subpage !== 'inspect');
  document.getElementById('datasetCompare')?.classList.toggle('hidden', subpage !== 'compare');
  const hint = document.getElementById('dataPageHint');
  if(hint) hint.textContent = subpage === 'compare'
    ? 'Compare the raw video against a selected synchronized pipeline output for the active sweep.'
    : 'This page inspects the active pipeline run in pipeline order, including raw frames, generated intermediates, artifact states, and lightweight process warnings.';
  const context = document.querySelector('#qcPage .pageContext');
  if(context) context.textContent = `${dataPageLabel(subpage)} · ${datasetId}`;
}
function artifactFramePattern(artifact){ return artifact?.frame_pattern || artifact?.framePattern || ''; }
function compareArtifactKey(artifact){ return artifact?.id || artifact?.step_id || artifact?.label || artifactFramePattern(artifact); }
function compareFrameArtifactsForRun(run){
  const artifacts = intermediateArtifactsForRun(run).filter(item => artifactFramePattern(item));
  const rank = item => {
    const kind = String(item.artifact_kind || item.kind || '').toLowerCase();
    const id = String(item.id || item.step_id || '').toLowerCase();
    const label = String(item.label || '').toLowerCase();
    if(kind === 'cfar_contrast_map' && id.includes('green_single_cfar')) return 0;
    if(kind === 'cfar_contrast_map' && id.includes('large')) return 1;
    if(kind === 'cfar_contrast_map') return 2;
    if(label.includes('contrast') && id.includes('green_single_cfar')) return 3;
    if(label.includes('contrast') && id.includes('large')) return 4;
    if(label.includes('contrast')) return 5;
    return 10;
  };
  return artifacts.slice().sort((a,b) => rank(a) - rank(b) || String(a.label || a.id).localeCompare(String(b.label || b.id)));
}
function compareContrastArtifactsForRun(run){
  return compareFrameArtifactsForRun(run).filter(item => {
    const kind = String(item.artifact_kind || item.kind || '').toLowerCase();
    const label = String(item.label || '').toLowerCase();
    return kind === 'cfar_contrast_map' || label.includes('contrast');
  });
}

function isGreenExcessRun(run){
  if(!run) return false;
  const id = String(run.run_id || '').toLowerCase();
  if(id.includes('green_excess_single_cfar')) return true;
  return intermediateArtifactsForRun(run).some(item => String(item.id || item.step_id || item.label || '').toLowerCase().includes('green'));
}
function selectedCompareInputArtifact(run){
  return intermediateArtifactsForRun(run).find(item => {
    const id = String(item.id || item.step_id || '').toLowerCase();
    return id === 'green_input' || id.includes('green_input');
  }) || null;
}
function selectedCompareArtifact(run){
  const artifacts = compareFrameArtifactsForRun(run);
  if(!artifacts.length) return null;
  const saved = setting('dataCompareArtifact');
  const savedMatch = artifacts.find(item => compareArtifactKey(item) === saved);
  if(savedMatch) return savedMatch;
  return artifacts.find(item => String(item.id || item.step_id || '').includes('cfar_large_ref') && item.artifact_kind === 'cfar_contrast_map')
    || artifacts.find(item => item.artifact_kind === 'cfar_contrast_map')
    || artifacts[0];
}
function selectedCompareContrastArtifact(run){
  const artifacts = compareContrastArtifactsForRun(run);
  if(!artifacts.length) return null;
  const saved = setting('dataCompareArtifact');
  const savedMatch = artifacts.find(item => compareArtifactKey(item) === saved);
  if(savedMatch) return savedMatch;
  if(isGreenExcessRun(run)) return artifacts.find(item => String(item.id || item.step_id || '').includes('green_single_cfar')) || artifacts[0];
  return artifacts.find(item => String(item.id || item.step_id || '').includes('cfar_large_ref')) || artifacts[0];
}
function compareSummaryValue(value){
  if(value === null || value === undefined || value === '') return 'n/a';
  if(typeof value === 'number') return fmt(value, Math.abs(value) >= 10 ? 2 : 4);
  if(Array.isArray(value)) return value.join(' x ');
  if(typeof value === 'object') return JSON.stringify(value);
  return String(value);
}
function compareArtifactSummaryHtml(artifact, run=null){
  if(!artifact) {
    const rois = dataCompareRoisForRun(run);
    if(rois.length) return `<div class="compareSummaryRows"><div><b>view</b><span>ROI overlay</span></div><div><b>ROIs</b><span>${rois.length}</span></div><div><b>frames</b><span>${data.video.frames}</span></div></div>`;
    return '<p class="hint">No derived frame sequence is selected.</p>';
  }
  const summary = artifact.summary || {};
  const norm = summary.normalization || {};
  const rows = [
    ['view', artifact.label || artifact.id || 'derived output'],
    ['stage', artifact.step_id || artifact.stage_id || 'n/a'],
    ['artifact kind', artifact.artifact_kind || 'frame sequence'],
    ['frames', artifact.frame_count || summary.frame_count || data.video.frames],
    ['guard px', summary.guard_px],
    ['training radius px', summary.training_radius_px],
    ['normalization hi', norm.hi],
    ['sample stride', norm.sample_stride]
  ];
  return `<div class="compareSummaryRows">${rows.map(([label, value]) => `<div><b>${escapeHtml(label)}</b><span>${escapeHtml(compareSummaryValue(value))}</span></div>`).join('')}</div>`;
}
function dataComparePreset(){
  const preset = setting('dataComparePreset') || defaultDataComparePreset(activeRunId());
  return ['focused_diagnostic','true_raw_roi','color_raw_roi','exact_raw_roi','raw_roi','raw_contrast','raw_contrast_roi','contrast_roi','raw_artifact'].includes(preset) ? preset : 'raw_artifact';
}
function dataCompareMode(){
  const mode = setting('dataCompareMode') || 'side_by_side';
  return mode === 'overlay' ? 'overlay' : 'side_by_side';
}
function dataCompareOpacity(){ return Math.max(0.05, Math.min(1, Number(setting('dataCompareOpacity') || 0.55))); }
function dataCompareIsFocusedDiagnostic(preset=dataComparePreset()){ return preset === 'focused_diagnostic'; }
function dataCompareNeedsArtifactControl(preset=dataComparePreset()){ return ['focused_diagnostic','raw_artifact','raw_contrast','raw_contrast_roi','contrast_roi'].includes(preset); }
function dataCompareShowInactiveRois(){
  const saved = setting('dataCompareShowInactiveRois');
  if(saved === undefined || saved === null) return isGreenExcessRun(selectedQcRun()) || !dataCompareIsFocusedDiagnostic();
  return saved === true || saved === 'true' || saved === 1 || saved === '1';
}
function dataCompareDiagnosticFrameForRun(run){
  const saved = Number(setting('dataCompareDiagnosticFrame'));
  if(Number.isFinite(saved) && saved >= 1) return Math.max(1, Math.min(data.video.frames, Math.round(saved)));
  const rois = dataCompareRoisForRun(run);
  if(!rois.length) return currentFrame;
  const counts = new Map();
  for(const roi of rois){
    for(const event of eventFrames(roi)){
      for(let offset=-1; offset<=1; offset++) {
        const frame = Math.max(1, Math.min(data.video.frames, event + offset));
        counts.set(frame, (counts.get(frame) || 0) + 1);
      }
    }
  }
  let bestFrame = currentFrame;
  let bestCount = -1;
  for(const [frame, count] of counts.entries()) {
    if(count > bestCount || (count === bestCount && frame < bestFrame)) {
      bestFrame = frame;
      bestCount = count;
    }
  }
  return bestCount > 0 ? bestFrame : currentFrame;
}
function applyDataCompareDiagnosticFrame(run){
  if(dataCompareFocusedFrameApplied || currentFrame !== 1) return;
  const frame = dataCompareDiagnosticFrameForRun(run);
  dataCompareFocusedFrameApplied = true;
  if(frame !== currentFrame) currentFrame = frame;
}
function dataCompareImageHtml({pattern, stageId, alt, missingText, className=''}){
  return `<img class="${escapeHtml(className)}" data-frame-pattern="${escapeHtml(pattern || '')}" data-stage-id="${escapeHtml(stageId || '')}" data-missing-text="${escapeHtml(missingText || 'Frame did not load')}" onerror="handleQcImageError(this)" alt="${escapeHtml(alt || 'Comparison frame')}">`;
}
function trueOriginalFramePattern(){
  const explicit = data.dataset?.true_original_frame_pattern || data.dataset?.trueOriginalFramePattern || data.video?.trueOriginalFramePattern;
  if(explicit) return artifactUrl(explicit);
  if(isExternalTestDataset()) return 'source_color/frame_%03d.png';
  return '';
}
function colorOriginalFramePattern(){
  const explicit = data.dataset?.color_original_frame_pattern || data.dataset?.colorOriginalFramePattern || data.video?.colorOriginalFramePattern;
  if(explicit) return artifactUrl(explicit);
  if(isExternalTestDataset()) return 'source_color_display/frame_%03d.png';
  return trueOriginalFramePattern();
}
function dataCompareRoisForRun(run){
  if(!run) return [];
  if(Array.isArray(run.artifacts?.review_rois) && run.artifacts.review_rois.length) return run.artifacts.review_rois;
  const key = reviewRoisCacheKey(run);
  const cached = key ? reviewRoisFileCache.get(key) : null;
  return Array.isArray(cached?.rois) ? cached.rois : [];
}
function dataCompareNeedsRois(preset){ return ['focused_diagnostic','true_raw_roi','color_raw_roi','exact_raw_roi','raw_roi','raw_contrast_roi','contrast_roi'].includes(preset); }
function dataCompareNeedsContrast(preset){ return ['focused_diagnostic','raw_contrast','raw_contrast_roi','contrast_roi'].includes(preset); }
function dataComparePanelHtml(panel, ratio, run){
  const title = escapeHtml(panel.title || 'Comparison');
  const stage = escapeHtml(panel.stageId || panel.kind || title);
  if(panel.kind === 'roi' || panel.kind === 'true_roi') {
    const pattern = panel.kind === 'true_roi' ? panel.pattern : data.video?.framePattern || '';
    const media = dataCompareImageHtml({
      pattern,
      stageId: `${stage}_raw`,
      alt: panel.title || 'Raw video frame with ROI overlay',
      missingText: 'Raw frame did not load'
    });
    return `<figure class="dataCompareFrameCard dataCompareRoiFrame">
      <div class="dataCompareImageShell" style="aspect-ratio:${ratio};">
        ${media}
        <canvas class="dataCompareRoiCanvas" data-run-id="${escapeHtml(run?.run_id || '')}" aria-label="Passive ROI overlay"></canvas>
      </div>
      <figcaption>${title}</figcaption>
    </figure>`;
  }
  const img = dataCompareImageHtml({
    pattern: panel.pattern,
    stageId: stage,
    alt: panel.title,
    missingText: panel.missingText || 'Comparison frame did not load',
    className: panel.className || ''
  });
  return `<figure class="dataCompareFrameCard">
    <div class="dataCompareImageShell" style="aspect-ratio:${ratio};">${img}</div>
    <figcaption>${title}</figcaption>
  </figure>`;
}
function dataComparePanelsForPreset({run, artifact, contrastArtifact, artifactKey, mode}){
  const colorPattern = colorOriginalFramePattern();
  const colorRaw = colorPattern ? {kind:'image', title:'Color original video', pattern:colorPattern, stageId:'color_original', missingText:'Color original frame did not load'} : null;
  const colorRoi = colorPattern ? {kind:'true_roi', title:'Color original + ROI overlay', pattern:colorPattern, stageId:'color_roi_overlay'} : null;
  const focusedColorRaw = colorPattern ? {kind:'image', title:'Original color frame', pattern:colorPattern, stageId:'focused_color_original', missingText:'Color original frame did not load'} : null;
  const focusedDetection = colorPattern ? {kind:'true_roi', title:'Detections', pattern:colorPattern, stageId:'focused_detection'} : null;
  const truePattern = trueOriginalFramePattern();
  const trueRaw = truePattern ? {kind:'image', title:'Exact MP4 frame', pattern:truePattern, stageId:'true_original', missingText:'True original frame did not load'} : null;
  const trueRoi = truePattern ? {kind:'true_roi', title:'Exact MP4 frame + ROI overlay', pattern:truePattern, stageId:'true_roi_overlay'} : null;
  const raw = {kind:'image', title:'Original video', pattern:data.video?.framePattern || '', stageId:'raw_compare', missingText:'Raw frame did not load'};
  const inputArtifact = selectedCompareInputArtifact(run);
  const grayInput = inputArtifact
    ? {kind:'image', title:inputArtifact.label || 'Green-excess input', pattern:artifactFramePattern(inputArtifact), stageId:compareArtifactKey(inputArtifact), missingText:'Green-excess input frame did not load'}
    : {kind:'image', title:'Grayscale input', pattern:data.video?.framePattern || '', stageId:'focused_gray_input', missingText:'Grayscale input frame did not load'};
  const selected = artifact ? {kind:'image', title:artifact.label || 'Selected output', pattern:artifactFramePattern(artifact), stageId:artifactKey, missingText:'Derived comparison frame did not load'} : null;
  const contrastKey = contrastArtifact ? compareArtifactKey(contrastArtifact) : '';
  const contrast = contrastArtifact ? {kind:'image', title:contrastArtifact.label || 'Gamma CFAR contrast map', pattern:artifactFramePattern(contrastArtifact), stageId:contrastKey, missingText:'Contrast-map frame did not load'} : null;
  const focusedContrast = contrastArtifact ? {kind:'image', title:'Gamma CFAR contrast', pattern:artifactFramePattern(contrastArtifact), stageId:`focused_${contrastKey}`, missingText:'Contrast-map frame did not load'} : null;
  const roi = {kind:'roi', title:'Original + ROI overlay', stageId:'roi_overlay'};
  const focusedRoi = focusedDetection || {kind:'roi', title:'Detections', stageId:'focused_detection'};
  const preset = dataComparePreset();
  if(preset === 'focused_diagnostic') return [focusedColorRaw || raw, grayInput, focusedContrast, focusedRoi].filter(Boolean);
  if(preset === 'true_raw_roi' || preset === 'color_raw_roi') return colorRaw && colorRoi ? [colorRaw, colorRoi] : [raw, roi];
  if(preset === 'exact_raw_roi') return trueRaw && trueRoi ? [trueRaw, trueRoi] : [raw, roi];
  if(preset === 'raw_artifact') {
    if(mode === 'overlay' && selected) return null;
    return selected ? [raw, selected] : [];
  }
  if(preset === 'raw_contrast') return contrast ? [raw, contrast] : [];
  if(preset === 'raw_contrast_roi') return contrast ? [raw, contrast, roi] : [];
  if(preset === 'contrast_roi') return contrast ? [contrast, roi] : [];
  return [raw, roi];
}
function dataCompareViewerHtml({run, artifact, artifactKey, contrastArtifact, mode}){
  const preset = dataComparePreset();
  if(preset === 'raw_artifact' && !artifact) return '<div class="compareMissing">No synchronized derived frame sequence is attached to this run yet.</div>';
  if(dataCompareNeedsContrast(preset) && !contrastArtifact) return '<div class="compareMissing">No Gamma CFAR contrast-map frame sequence is attached to this run yet.</div>';
  const ratio = `${Number(data.video.width) || 1} / ${Number(data.video.height) || 1}`;
  const panels = dataComparePanelsForPreset({run, artifact, contrastArtifact, artifactKey, mode});
  if(preset === 'raw_artifact' && mode === 'overlay' && artifact) {
    const rawImg = dataCompareImageHtml({pattern:data.video?.framePattern || '', stageId:'raw_compare', alt:'Raw video frame', missingText:'Raw frame did not load'});
    const derivedImg = dataCompareImageHtml({pattern:artifactFramePattern(artifact), stageId:artifactKey, alt:artifact.label || 'Derived frame', missingText:'Derived comparison frame did not load', className:'compareOverlayDerived'});
    return `
      <div id="dataCompareOverlayShell" class="compareOverlapShell" style="aspect-ratio:${ratio}; --overlay-opacity:${dataCompareOpacity()};">
        ${rawImg}
        ${derivedImg}
        <span class="compareLabel raw">Raw</span>
        <span class="compareLabel derived">${escapeHtml(artifact.label || 'Derived')}</span>
      </div>`;
  }
  if(!panels?.length) return '<div class="compareMissing">The selected comparison preset has no available frames yet.</div>';
  return `<div class="dataCompareColumns dataCompareColumns-${panels.length}">${panels.map(panel => dataComparePanelHtml(panel, ratio, run)).join('')}</div>`;
}
function drawDataCompareRoiOverlays(){
  for(const canvas of document.querySelectorAll('.dataCompareRoiCanvas')){
    const run = runById(canvas.dataset.runId) || selectedQcRun();
    const rois = dataCompareRoisForRun(run).filter(roi => !roiAnn(roi.id).deleted);
    canvas.width = Number(data.video.width) || 1;
    canvas.height = Number(data.video.height) || 1;
    const showInactive = dataCompareShowInactiveRois();
    const c = canvas.getContext('2d');
    c.clearRect(0, 0, canvas.width, canvas.height);
    c.lineCap = 'round';
    c.lineJoin = 'round';
    for(const roi of rois){
      const cx = Number(roi.centroidX ?? roi.x);
      const cy = Number(roi.centroidY ?? roi.y);
      if(!Number.isFinite(cx) || !Number.isFinite(cy)) continue;
      const active = eventNearFrame(roi, currentFrame);
      if(!active && !showInactive) continue;
      const color = active ? '#f59e0b' : '#22d3ee';
      c.globalAlpha = active ? 0.20 : 0.08;
      c.fillStyle = color;
      for(const p of roi.points || []) c.fillRect(Number(p[0]) || 0, Number(p[1]) || 0, 1, 1);
      c.globalAlpha = active ? 0.94 : 0.72;
      c.strokeStyle = color;
      c.lineWidth = active ? 2.2 : 1.35;
      const area = Math.max(1, Number(roi.area) || (roi.points || []).length || 16);
      const radius = Math.max(4, Math.sqrt(area / Math.PI) + 2);
      c.beginPath();
      c.arc(cx, cy, radius, 0, Math.PI * 2);
      c.stroke();
      if(active) {
        c.save();
        c.globalAlpha = 0.9;
        c.strokeStyle = '#ffffff';
        c.lineWidth = 1.2;
        c.beginPath();
        c.arc(cx, cy, radius + 3, 0, Math.PI * 2);
        c.stroke();
        c.restore();
      }
    }
    c.globalAlpha = 1;
  }
}
function renderDataCompare(){
  const root = document.getElementById('datasetCompare');
  if(!root) return;
  const runs = data.architectureRuns?.runs || [];
  const run = selectedQcRun();
  if(!run){
    root.innerHTML = '<p class="hint">No pipeline run is available for comparison.</p>';
    return;
  }
  const preset = dataComparePreset();
  if(dataCompareNeedsRois(preset) && !dataCompareRoisForRun(run).length && reviewRoisFileUrl(run)) {
    root.innerHTML = '<p class="hint">Loading ROI overlay for comparison...</p>';
    ensureReviewRoisForRun(run).then(() => {
      if(selectedQcRun()?.run_id === run.run_id) renderDataCompare();
    }).catch(() => setSaveState('could not load Data Compare ROI overlays', 'bad'));
    return;
  }
  if(dataCompareIsFocusedDiagnostic(preset)) applyDataCompareDiagnosticFrame(run);
  const artifacts = compareFrameArtifactsForRun(run);
  const artifact = selectedCompareArtifact(run);
  const contrastArtifact = selectedCompareContrastArtifact(run);
  const artifactKey = artifact ? compareArtifactKey(artifact) : '';
  const mode = dataCompareMode();
  const runOptions = processRunOptionsHtml(runs, run) || runs.map(item => `<option value="${escapeHtml(item.run_id)}" ${item.run_id === run.run_id ? 'selected' : ''}>${escapeHtml(runLabel(item))}</option>`).join('');
  const focused = dataCompareIsFocusedDiagnostic(preset);
  const selectedArtifactKey = dataCompareNeedsContrast(preset) && contrastArtifact ? compareArtifactKey(contrastArtifact) : artifactKey;
  const artifactOptions = artifacts.map(item => {
    const key = compareArtifactKey(item);
    const label = item.label || item.id || key;
    return `<option value="${escapeHtml(key)}" ${key === selectedArtifactKey ? 'selected' : ''}>${escapeHtml(label)}</option>`;
  }).join('');
  const presetOptions = [
    ['focused_diagnostic', 'Focused diagnostic frame'],
    ['true_raw_roi', 'True original + ROI overlay'],
    ['exact_raw_roi', 'Exact MP4 frame + ROI overlay'],
    ['raw_roi', 'Grayscale original + ROI overlay'],
    ['raw_contrast', 'Original + CFAR contrast'],
    ['raw_contrast_roi', 'Original + contrast + ROI'],
    ['contrast_roi', 'Contrast + ROI overlay'],
    ['raw_artifact', 'Original + selected output']
  ].map(([value, label]) => `<option value="${value}" ${preset === value ? 'selected' : ''}>${escapeHtml(label)}</option>`).join('');
  const viewer = dataCompareViewerHtml({run, artifact, artifactKey, contrastArtifact, mode});
  const artifactControlHidden = dataCompareNeedsArtifactControl(preset) ? '' : 'hidden';
  const modeControlHidden = preset === 'raw_artifact' ? '' : 'hidden';
  const inactiveControlHidden = dataCompareNeedsRois(preset) ? '' : 'hidden';
  const playControl = focused ? '' : '<button id="dataComparePlayBtn" type="button">Play</button>';
  const focusedSourceControls = focused ? `
          <label>Run <select id="dataCompareRunSelect">${runOptions}</select></label>
          <label class="${artifactControlHidden}">CFAR map <select id="dataCompareArtifactSelect" ${artifacts.length ? '' : 'disabled'}>${artifactOptions}</select></label>` : '';
  const sidePanel = focused ? '' : `
      <aside class="dataCompareSide">
        <section class="archCard">
          <h3>Comparison Source</h3>
          <label>Run <select id="dataCompareRunSelect">${runOptions}</select></label>
          <label class="${artifactControlHidden}">Output <select id="dataCompareArtifactSelect" ${artifacts.length ? '' : 'disabled'}>${artifactOptions}</select></label>
        </section>
        <section class="archCard">
          <h3>Output Details</h3>
          ${compareArtifactSummaryHtml(dataCompareNeedsContrast(preset) ? contrastArtifact : preset === 'raw_artifact' ? artifact : null, run)}
        </section>
        ${renderSweepEvidencePanel(run)}
      </aside>`;
  root.innerHTML = `
    ${focused ? '' : renderRunSummaryCards(run)}
    <div class="dataCompareLayout ${focused ? 'dataCompareLayoutFocused' : ''}">
      <section class="dataComparePanel ${focused ? 'dataCompareFocusedPanel' : ''}">
        <div class="toolbar compactToolGroup">
          ${playControl}
          <button id="dataComparePrevFrameBtn" type="button">Prev</button>
          <button id="dataCompareNextFrameBtn" type="button">Next</button>
          <label>Frame <input id="dataCompareFrameSlider" type="range" min="1" max="${data.video.frames}" value="${currentFrame}"></label>
          <b id="dataCompareFrameLabel">${frameLabelText(currentFrame)} / ${data.video.frames}</b>
          <label>Preset
            <select id="dataComparePresetSelect">${presetOptions}</select>
          </label>
          ${focusedSourceControls}
          <label class="dataCompareInactiveControl ${inactiveControlHidden}"><input id="dataCompareShowInactiveRois" type="checkbox" ${dataCompareShowInactiveRois() ? 'checked' : ''}> Show inactive</label>
          <label class="compareModeControl ${modeControlHidden}">View mode
            <select id="dataCompareModeSelect">
              <option value="side_by_side" ${mode === 'side_by_side' ? 'selected' : ''}>Side by side</option>
              <option value="overlay" ${mode === 'overlay' ? 'selected' : ''}>Overlap</option>
            </select>
          </label>
          <label class="compareOverlayControl ${preset === 'raw_artifact' && mode === 'overlay' ? '' : 'hidden'}">Opacity <input id="dataCompareOpacitySlider" type="range" min="0.05" max="1" step="0.05" value="${dataCompareOpacity()}"></label>
        </div>
        ${viewer}
      </section>
      ${sidePanel}
    </div>`;
  wireDataCompareControls();
  updateQcFrameView();
  drawDataCompareRoiOverlays();
}
function updateDataCompareFrameView(){
  const slider = document.getElementById('dataCompareFrameSlider');
  const label = document.getElementById('dataCompareFrameLabel');
  if(slider) slider.value = currentFrame;
  if(label) label.textContent = `${frameLabelText(currentFrame)} / ${data.video.frames}`;
  updateDataCompareOverlay();
  drawDataCompareRoiOverlays();
}
function updateDataCompareOverlay(){
  const shell = document.getElementById('dataCompareOverlayShell');
  const slider = document.getElementById('dataCompareOpacitySlider');
  if(!shell) return;
  const value = Math.max(0.05, Math.min(1, Number(slider?.value || setting('dataCompareOpacity') || 0.55)));
  shell.style.setProperty('--overlay-opacity', value);
}
function toggleDataComparePlay(){
  const btn = document.getElementById('dataComparePlayBtn');
  if(qcTimer) {
    clearInterval(qcTimer);
    qcTimer = null;
    if(btn) btn.textContent = 'Play';
    return;
  }
  if(btn) btn.textContent = 'Pause';
  qcTimer = setInterval(() => setFrame(currentFrame >= data.video.frames ? 1 : currentFrame + 1), 120);
}
function wireDataCompareControls(){
  const runSelect = document.getElementById('dataCompareRunSelect');
  if(runSelect) runSelect.onchange = async e => {
    await selectActiveRun(e.target.value, {loadReview:false});
    if(setting('dataComparePreset') === undefined || setting('dataComparePreset') === null) annotations.settings.dataComparePreset = defaultDataComparePreset(e.target.value);
    renderDataCompare();
  };
  const presetSelect = document.getElementById('dataComparePresetSelect');
  if(presetSelect) presetSelect.onchange = e => {
    setSetting('dataComparePreset', e.target.value);
    renderDataCompare();
  };
  const artifactSelect = document.getElementById('dataCompareArtifactSelect');
  if(artifactSelect) artifactSelect.onchange = e => {
    setSetting('dataCompareArtifact', e.target.value);
    renderDataCompare();
  };
  const modeSelect = document.getElementById('dataCompareModeSelect');
  if(modeSelect) modeSelect.onchange = e => {
    setSetting('dataCompareMode', e.target.value === 'overlay' ? 'overlay' : 'side_by_side');
    renderDataCompare();
  };
  const frameSlider = document.getElementById('dataCompareFrameSlider');
  if(frameSlider) frameSlider.oninput = e => setFrame(Number(e.target.value));
  const inactiveToggle = document.getElementById('dataCompareShowInactiveRois');
  if(inactiveToggle) inactiveToggle.onchange = e => {
    setSetting('dataCompareShowInactiveRois', e.target.checked);
    drawDataCompareRoiOverlays();
  };
  const opacitySlider = document.getElementById('dataCompareOpacitySlider');
  if(opacitySlider) {
    opacitySlider.oninput = e => {
      annotations.settings.dataCompareOpacity = Number(e.target.value);
      updateDataCompareOverlay();
    };
    opacitySlider.onchange = e => setSetting('dataCompareOpacity', Number(e.target.value));
  }
  const prev = document.getElementById('dataComparePrevFrameBtn');
  const next = document.getElementById('dataCompareNextFrameBtn');
  const play = document.getElementById('dataComparePlayBtn');
  if(prev) prev.onclick = () => setFrame(currentFrame - 1);
  if(next) next.onclick = () => setFrame(currentFrame + 1);
  if(play) play.onclick = toggleDataComparePlay;
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
    ...templateGridReportMarkdownLines(),
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
        <span class="homeEyebrow">Shareable output</span>
        <h2>Review Report</h2>
        <p class="hint">Use this page to export a concise lab handoff. Progress is for working status; Report is for sharing and audit downloads.</p>
      </div>
      <div class="buttonRow">
        <button id="downloadReportBtn">Download Markdown</button>
        <button id="downloadProvenanceAuditBtn">Download Provenance Audit</button>
      </div>
    </div>
    <div class="reportExportSummary">
      <div class="metric"><b>${s.roi_states.accepted}</b><span>accepted ROIs</span></div>
      <div class="metric"><b>${s.event_states.accepted}</b><span>accepted events</span></div>
      <div class="metric"><b>${s.control_ready.yes + s.control_ready.maybe}</b><span>control-ready yes/maybe</span></div>
      <div class="metric"><b>${s.review_progress.tuning_ready ? 'yes' : 'no'}</b><span>tuning ready</span></div>
      <div class="metric"><b>${Object.values(s.reviewer_missing || {}).reduce((a,b) => a + b, 0)}</b><span>labels missing reviewer</span></div>
      <div class="metric"><b>${Math.round(100 * audit.coverage_fraction)}%</b><span>reviewer coverage</span></div>
    </div>
    ${renderTemplateGridReportSummary()}
    ${renderTemplateGridSweepPanel(activeRun(), {standalone:true, compact:true})}
    <details class="archCard">
      <summary>Reviewer audit details</summary>
      ${auditRows('Reviewer contributions', Object.keys(s.reviewer_counts || {}).length ? s.reviewer_counts : {unassigned: 0})}
      ${auditRows('Missing reviewer IDs', s.reviewer_missing || {none: 0})}
    </details>
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


function reviewSubPageFromHash(hashText=location.hash){
  const hash = (hashText || '#review').replace(/^#\/?/, '');
  if(hash === 'review-stencil' || hash === 'stencil' || hash === 'anatomy-stencil') return 'stencil';
  if(hash === 'review-overlap' || hash === 'overlap' || hash === 'sweep-overlap' || hash === 'candidate-overlay' || hash === 'review-candidate-overlay') return 'overlap';
  if(hash === 'review-triage' || hash === 'triage' || hash === 'review-queue') return 'triage';
  return 'inspect';
}
function reviewPageLabel(subpage){
  if(subpage === 'stencil') return 'Review Stencil';
  if(subpage === 'overlap') return 'Candidate Overlay';
  if(subpage === 'triage') return 'Review Triage';
  return 'Review Inspect';
}
function updateReviewSubnav(subpage){
  document.getElementById('reviewInspectSubtab')?.classList.toggle('active', subpage === 'inspect');
  document.getElementById('reviewStencilSubtab')?.classList.toggle('active', subpage === 'stencil');
  document.getElementById('reviewOverlapSubtab')?.classList.toggle('active', subpage === 'overlap');
  document.getElementById('reviewTriageSubtab')?.classList.toggle('active', subpage === 'triage');
  const context = document.querySelector('.stage.reviewOnly .pageContext');
  if(context) context.textContent = `${reviewPageLabel(subpage)} · ${datasetId}`;
  document.getElementById('reviewStencilPage')?.classList.toggle('hidden', subpage !== 'stencil');
  document.getElementById('reviewOverlapPage')?.classList.toggle('hidden', subpage !== 'overlap');
  document.getElementById('reviewTriagePage')?.classList.toggle('hidden', subpage !== 'triage');
  appRoot.classList.toggle('review-stencil-mode', subpage === 'stencil');
  appRoot.classList.toggle('review-overlap-mode', subpage === 'overlap');
  appRoot.classList.toggle('review-triage-mode', subpage === 'triage');
}
function diagnosticAssetUrl(file){ return `diagnostics/${file}`; }
function polygonArea(points){
  if(!Array.isArray(points) || points.length < 3) return 0;
  let sum = 0;
  for(let i = 0; i < points.length; i++){
    const a = points[i], b = points[(i + 1) % points.length];
    sum += Number(a.x || 0) * Number(b.y || 0) - Number(b.x || 0) * Number(a.y || 0);
  }
  return Math.abs(sum / 2);
}
function polygonBounds(points){
  if(!Array.isArray(points) || !points.length) return null;
  const xs = points.map(p => Number(p.x)).filter(Number.isFinite);
  const ys = points.map(p => Number(p.y)).filter(Number.isFinite);
  if(!xs.length || !ys.length) return null;
  return {x0:Math.min(...xs), y0:Math.min(...ys), x1:Math.max(...xs), y1:Math.max(...ys)};
}
function pointInPolygon(point, polygon){
  if(!point || !Array.isArray(polygon) || polygon.length < 3) return false;
  let inside = false;
  const x = Number(point.x), y = Number(point.y);
  for(let i = 0, j = polygon.length - 1; i < polygon.length; j = i++){
    const xi = Number(polygon[i].x), yi = Number(polygon[i].y);
    const xj = Number(polygon[j].x), yj = Number(polygon[j].y);
    const intersect = ((yi > y) !== (yj > y)) && x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-9) + xi;
    if(intersect) inside = !inside;
  }
  return inside;
}

function pointSegmentDistance(point, a, b){
  const px = Number(point.x), py = Number(point.y);
  const ax = Number(a.x), ay = Number(a.y), bx = Number(b.x), by = Number(b.y);
  const dx = bx - ax, dy = by - ay;
  const denom = dx * dx + dy * dy;
  const t = denom > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / denom)) : 0;
  const x = ax + t * dx, y = ay + t * dy;
  return Math.sqrt((px - x) * (px - x) + (py - y) * (py - y));
}
function distanceToPolygonEdge(point, polygon){
  if(!point || !Array.isArray(polygon) || polygon.length < 2) return Infinity;
  let best = Infinity;
  for(let i = 0; i < polygon.length; i++) best = Math.min(best, pointSegmentDistance(point, polygon[i], polygon[(i + 1) % polygon.length]));
  return best;
}
function stencilPointStatus(point, marginPx=12){
  const polygon = savedStencilPoints();
  if(polygon.length < 3) return {status:'unknown', inside:false, edge:false, distance_px:null};
  const inside = pointInPolygon(point, polygon);
  const distance = distanceToPolygonEdge(point, polygon);
  const edge = Number.isFinite(distance) && distance <= marginPx;
  return {status: edge ? 'edge-near' : inside ? 'inside' : 'outside', inside, edge, distance_px:distance};
}
function roiStencilStatus(roi){ return stencilPointStatus(roiCenter(roi)); }
function stencilStatusLabel(status){
  const value = typeof status === 'string' ? status : status?.status;
  if(value === 'inside') return 'inside stencil';
  if(value === 'edge-near') return 'near stencil edge';
  if(value === 'outside') return 'outside stencil';
  return 'no stencil';
}
function stencilMetricsForRois(rois){
  const rows = Array.isArray(rois) ? rois : [];
  const metrics = {total:rows.length, inside:0, edge:0, outside:0, unknown:0, events_inside:0, events_edge:0, events_outside:0};
  for(const roi of rows){
    const status = roiStencilStatus(roi).status;
    const events = roiEventSupport(roi);
    if(status === 'inside') { metrics.inside++; metrics.events_inside += events; }
    else if(status === 'edge-near') { metrics.edge++; metrics.events_edge += events; }
    else if(status === 'outside') { metrics.outside++; metrics.events_outside += events; }
    else metrics.unknown++;
  }
  return metrics;
}
function savedStencilPoints(){
  const poly = annotations.settings?.anatomyStencil?.polygon || [];
  return Array.isArray(poly) ? poly.map(p => Array.isArray(p) ? {x:Number(p[0]), y:Number(p[1])} : {x:Number(p.x), y:Number(p.y)}).filter(p => Number.isFinite(p.x) && Number.isFinite(p.y)) : [];
}
function stencilPayload(){
  const canvas = document.getElementById('stencilCanvas');
  const points = stencilState.points || [];
  const b = polygonBounds(points);
  return {
    schema_version: 1,
    id: 'hindbrain_rough_stencil_v1',
    label: 'Rough hindbrain stencil',
    coordinate_space: 'image_pixels',
    image_width: canvas?.width || data.video?.width || 0,
    image_height: canvas?.height || data.video?.height || 0,
    polygon: points.map(p => [Number(p.x.toFixed(3)), Number(p.y.toFixed(3))]),
    bounds: b ? [Math.round(b.x0), Math.round(b.y0), Math.round(b.x1), Math.round(b.y1)] : null,
    area_px: Number(polygonArea(points).toFixed(3)),
    source_projection: document.getElementById('stencilProjectionSelect')?.value || stencilState.imageName || '',
    updated_at: new Date().toISOString(),
    notes: 'Rough anatomy prior for hindbrain-focused candidate reranking; not a ground-truth segmentation.'
  };
}
function canvasPointFromEvent(canvas, ev){
  const rect = canvas.getBoundingClientRect();
  return {
    x:(ev.clientX - rect.left) * canvas.width / Math.max(1, rect.width),
    y:(ev.clientY - rect.top) * canvas.height / Math.max(1, rect.height)
  };
}
function updateStencilMetrics(){
  const points = stencilState.points || [];
  const b = polygonBounds(points);
  const area = polygonArea(points);
  const pointCount = document.getElementById('stencilPointCount');
  const areaEl = document.getElementById('stencilAreaPx');
  const boundsEl = document.getElementById('stencilBounds');
  const preview = document.getElementById('stencilJsonPreview');
  if(pointCount) pointCount.textContent = points.length;
  if(areaEl) areaEl.textContent = `${Math.round(area)} px`;
  if(boundsEl) boundsEl.textContent = b ? `${Math.round(b.x0)},${Math.round(b.y0)} to ${Math.round(b.x1)},${Math.round(b.y1)}` : '-';
  if(preview) preview.textContent = JSON.stringify(stencilPayload(), null, 2);
}
function drawStencilPolygon(ctx, points, {fill='rgba(250, 204, 21, 0.18)', stroke='#facc15', pointsVisible=true}={}){
  if(!ctx || !points?.length) return;
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = stroke;
  ctx.fillStyle = fill;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for(const p of points.slice(1)) ctx.lineTo(p.x, p.y);
  if(points.length >= 3) ctx.closePath();
  if(points.length >= 3) ctx.fill();
  ctx.stroke();
  if(pointsVisible) for(const p of points){
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = stroke;
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#111827';
    ctx.stroke();
  }
  ctx.restore();
}
function drawStencilCanvas(){
  const canvas = document.getElementById('stencilCanvas');
  if(!canvas) return;
  const ctx2 = canvas.getContext('2d');
  ctx2.clearRect(0, 0, canvas.width, canvas.height);
  if(stencilState.image?.complete && stencilState.image.naturalWidth) ctx2.drawImage(stencilState.image, 0, 0, canvas.width, canvas.height);
  drawStencilPolygon(ctx2, stencilState.points);
  updateStencilMetrics();
}
function loadStencilProjection(name){
  const canvas = document.getElementById('stencilCanvas');
  const status = document.getElementById('stencilStatus');
  if(!canvas) return;
  const image = new Image();
  stencilState.image = image;
  stencilState.imageName = name;
  if(status) status.textContent = 'loading projection';
  image.onload = () => {
    canvas.width = image.naturalWidth || canvas.width;
    canvas.height = image.naturalHeight || canvas.height;
    if(status) status.textContent = `${name} · ${canvas.width}x${canvas.height}`;
    drawStencilCanvas();
  };
  image.onerror = () => {
    if(status) status.textContent = 'projection unavailable';
    drawStencilCanvas();
  };
  image.src = `${diagnosticAssetUrl(name)}?v=${Date.now()}`;
}
function loadSavedStencilIntoEditor(){
  stencilState.points = savedStencilPoints();
  const source = annotations.settings?.anatomyStencil?.source_projection;
  const select = document.getElementById('stencilProjectionSelect');
  if(select && source && [...select.options].some(opt => opt.value === source)) select.value = source;
  loadStencilProjection(select?.value || source || 'raw_max.png');
}
function wireReviewStencilControls(){
  if(stencilState.wired) return;
  stencilState.wired = true;
  const canvas = document.getElementById('stencilCanvas');
  const select = document.getElementById('stencilProjectionSelect');
  canvas?.addEventListener('click', ev => {
    stencilState.points.push(canvasPointFromEvent(canvas, ev));
    drawStencilCanvas();
  });
  select?.addEventListener('change', () => loadStencilProjection(select.value));
  document.getElementById('stencilUndoBtn')?.addEventListener('click', () => { stencilState.points.pop(); drawStencilCanvas(); });
  document.getElementById('stencilClearBtn')?.addEventListener('click', () => { stencilState.points = []; drawStencilCanvas(); });
  document.getElementById('stencilReloadBtn')?.addEventListener('click', loadSavedStencilIntoEditor);
  document.getElementById('stencilSaveBtn')?.addEventListener('click', () => {
    if(stencilState.points.length < 3) {
      setSaveState('draw at least 3 stencil points', 'bad');
      return;
    }
    annotations.settings.anatomyStencil = stencilPayload();
    queueSave();
    setSaveState('saved anatomy stencil', 'ok');
    drawOverlapCanvas();
  });
}
