const PAGE_CHROME = {
  datasets: {label:'Datasets', page:'datasets'},
  annotate: {label:'Annotate', page:'annotate'},
  results: {label:'Results', page:'results'},
  research: {label:'Research Tools', page:'research'},
  home: {label:'Research Tools', page:'home'},
  architecture: {label:'Pipelines', page:'architecture'},
  experiments: {label:'Experiment Lab', page:'experiments'},
  metrics: {label:'Progress', page:'metrics'},
  qc: {label:'Data', page:'qc'},
  report: {label:'Detailed Report', page:'report'}
};

const PAGE_NAV_ITEMS = [
  {page:'datasets', href:'#datasets', label:'Datasets'},
  {page:'annotate', href:'#annotate', label:'Annotate'},
  {page:'results', href:'#results', label:'Results'},
  {page:'research', href:'#research', label:'Research Tools'}
];

const ROUTE_PAGE_IDS = [
  'homePage','datasetsPage','researchPage','architecturePage',
  'experimentsPage','metricsPage','qcPage','reportPage'
];

const ROUTE_MODE_CLASSES = [
  'product-mode','research-mode','home-mode','arch-mode','lab-mode',
  'qc-mode','normal-annotation-mode','research-context-enabled'
];

function pageNavHtml(){
  return `<nav class="navTabs">${PAGE_NAV_ITEMS.map(item => `<a data-nav-page="${item.page}" href="${item.href}">${escapeHtml(item.label)}</a>`).join('')}</nav>`;
}

function modeSelectHtml(){
  return `<label class="modeToggle researchModeControl">Mode
    <select class="uiModeSelect" aria-label="Interface mode">
      <option value="guided">Guided</option>
      <option value="standard">Standard</option>
      <option value="expert">Expert</option>
    </select>
  </label>`;
}

function themeSelectHtml(){
  return `<label class="modeToggle">Theme
    <select class="themeSelect" aria-label="Theme">
      <option value="system">System</option>
      <option value="light">Light</option>
      <option value="dark">Dark</option>
    </select>
  </label>`;
}

function renderSharedPageChrome(){
  for(const root of document.querySelectorAll('[data-page-chrome]')){
    const config = PAGE_CHROME[root.dataset.pageChrome] || PAGE_CHROME.home;
    root.innerHTML = `<h1>NeuRev</h1><span class="pageContext">${escapeHtml(config.label)} · ${escapeHtml(datasetId)}</span>${pageNavHtml()}${modeSelectHtml()}${themeSelectHtml()}`;
  }
}

function resetRouteState(){
  for(const id of ROUTE_PAGE_IDS) document.getElementById(id)?.classList.add('hidden');
  for(const className of ROUTE_MODE_CLASSES) appRoot.classList.remove(className);
  for(const task of Object.keys(ANNOTATION_TASK_DEFS || {})) appRoot.classList.remove(`task-${task}`);
  for(const link of document.querySelectorAll('[data-nav-page]')) link.classList.remove('active');
  updateReviewSubnav('inspect');
  updateDataSubnav('inspect');
}

function setActiveProductNav(page){
  for(const link of document.querySelectorAll('[data-nav-page]')) link.classList.toggle('active', link.dataset.navPage === page);
}

function setPageContext(pageId, label){
  const context = document.querySelector(`#${pageId} .pageContext`);
  if(context) context.textContent = `${label} · ${datasetId}`;
}

function finishRoute(routeKey){
  appRoot.classList.remove('booting');
  if(routePage.lastRoute !== routeKey) window.requestAnimationFrame(() => window.scrollTo({top:0, left:0, behavior:'auto'}));
  routePage.lastRoute = routeKey;
}

function routeProductLanding(page){
  resetRouteState();
  appRoot.classList.add('product-mode');
  const pageId = page === 'datasets' ? 'datasetsPage' : page === 'results' ? 'reportPage' : 'researchPage';
  document.getElementById(pageId)?.classList.remove('hidden');
  setActiveProductNav(page);
  setPageContext(pageId, PAGE_CHROME[page]?.label || page);
  if(page === 'datasets') renderDatasetsPage();
  else if(page === 'results') renderResultsPage();
  else renderResearchToolsPage();
  finishRoute(`product:${page}`);
}

function routeReview(hash){
  resetRouteState();
  const reviewSubpage = reviewSubPageFromHash(`#${hash}`);
  const normal = hash === 'annotate' || hash === 'review' || reviewSubpage === 'correction';
  if(normal){
    appRoot.classList.add('normal-annotation-mode');
    appRoot.classList.toggle('research-context-enabled', researchToolsEnabled());
    setActiveProductNav('annotate');
  } else setActiveProductNav('research');
  updateReviewSubnav(reviewSubpage);
  if(normal){
    const context = document.querySelector('.stage.reviewOnly .pageContext');
    if(context) context.textContent = `Annotate · ${datasetId}`;
  }
  if(reviewSubpage === 'stencil') renderReviewStencil();
  else if(reviewSubpage === 'overlap') renderReviewOverlap();
  else if(reviewSubpage === 'triage') renderReviewTriage();
  else if(reviewSubpage === 'correction') renderAnnotationCorrection();
  else resizeOverlay();
  renderAnnotationTaskShell();
  renderNextBestActions();
  finishRoute(`review:${hash}:${reviewSubpage}`);
}

function routeAdvanced(page, hash){
  resetRouteState();
  setActiveProductNav('research');
  const pageId = page === 'home' ? 'homePage' : page === 'architecture' ? 'architecturePage' : page === 'experiments' ? 'experimentsPage' : page === 'metrics' ? 'metricsPage' : page === 'qc' ? 'qcPage' : 'reportPage';
  document.getElementById(pageId)?.classList.remove('hidden');
  appRoot.classList.toggle('home-mode', page === 'home');
  appRoot.classList.toggle('arch-mode', page === 'architecture');
  appRoot.classList.toggle('lab-mode', ['experiments','metrics','report'].includes(page));
  appRoot.classList.toggle('qc-mode', page === 'qc');
  const dataSubpage = page === 'qc' ? dataSubPageFromHash(`#${hash}`) : 'inspect';
  if(page === 'qc') updateDataSubnav(dataSubpage);
  if(page === 'home') renderWorkflowHome();
  else if(page === 'architecture') renderArchitectureLab();
  else if(page === 'experiments') renderExperimentLab();
  else if(page === 'metrics') renderMetricsAudit();
  else if(page === 'qc' && dataSubpage === 'compare') renderDataCompare();
  else if(page === 'qc') renderDatasetQc();
  else renderReviewReport();
  renderNextBestActions();
  finishRoute(`advanced:${page}:${hash}`);
}

function routePage(){
  const hash = (location.hash || '#datasets').replace(/^#\/?/, '');
  if(['datasets','dataset','imports'].includes(hash)) return routeProductLanding('datasets');
  if(hash === 'annotate' || hash === 'review') return routeReview(hash);
  if(hash === 'results') return routeProductLanding('results');
  if(hash === 'research') return routeProductLanding('research');
  if(['review-stencil','stencil','anatomy-stencil','review-overlap','overlap','sweep-overlap','candidate-overlay','review-candidate-overlay','review-triage','triage','review-queue','annotation-correction','review-correction','correction'].includes(hash)) return routeReview(hash);

  const page = ['home','workflow'].includes(hash) ? 'home'
    : ['pipelines','architecture','architecture-lab'].includes(hash) ? 'architecture'
      : ['experiments','experiment-lab'].includes(hash) ? 'experiments'
        : ['progress','metrics','audit'].includes(hash) ? 'metrics'
          : ['data','data-compare','process','process-lab','qc','dataset-qc'].includes(hash) ? 'qc'
            : hash === 'report' ? 'report' : null;
  if(page) return routeAdvanced(page, hash);
  routeProductLanding('datasets');
}

async function boot(){
  renderSharedPageChrome();
  populateEvidenceSelect();
  await loadAnnotations();
  populateVideoViewControls();
  if(serverBacked){
    try {
      const res = await fetch('architecture_runs.json', {cache:'no-store'});
      if(res.ok) data.architectureRuns = await res.json();
    } catch (_) {}
  }
  repairEmptyActiveRunSelection();
  try {
    await ensureReviewRoisForRun(activeRun());
  } catch (err) {
    console.warn('Could not load active run ROI overlays during startup:', err);
  }
  initControls();
  initCfarMaskAnnotation();
  renderParams();
  const first = visibleRois()[0] || reviewRois()[0];
  selectedId = first?.id || null;
  selectedRoiIds = new Set(selectedId ? [String(selectedId)] : []);
  if(selectedId){
    selectedEventFrame = eventsForRoi(selectedRoi())[0]?.frame || null;
    roiNotes.value = roiAnn(selectedId).notes || '';
    eventNotes.value = selectedEventFrame ? eventAnn(selectedId, selectedEventFrame).notes || '' : '';
  }
  if(selectedSuggestionId){
    document.getElementById('suggestionNotes').value = suggestionAnn(selectedSuggestionId).notes || '';
    document.getElementById('artifactClass').value = suggestionAnn(selectedSuggestionId).artifact_class || suggestionAnn(selectedSuggestionId).artifactClass || '';
  }
  renderRunSyncControls();
  loadGenerationEnvironment();
  setFrame(1);
  routePage();
  renderAll();
}

boot();
