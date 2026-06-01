const PAGE_CHROME = {
  home: {label:'Home', page:'home'},
  architecture: {label:'Pipelines', page:'architecture'},
  experiments: {label:'Experiment Lab', page:'experiments'},
  metrics: {label:'Progress', page:'metrics'},
  qc: {label:'Data', page:'qc'},
  report: {label:'Report', page:'report'}
};
const PAGE_NAV_ITEMS = [
  {page:'home', href:'#home', label:'Home'},
  {page:'qc', href:'#data', label:'Data'},
  {page:'architecture', href:'#pipelines', label:'Pipelines'},
  {page:'experiments', href:'#experiments', label:'Experiment Lab'},
  {page:'review', href:'#review', label:'Review'},
  {page:'metrics', href:'#progress', label:'Progress'},
  {page:'report', href:'#report', label:'Report'}
];
function pageNavHtml(){
  return `<nav class="navTabs">${PAGE_NAV_ITEMS.map(item => `<a data-nav-page="${item.page}" href="${item.href}">${escapeHtml(item.label)}</a>`).join('')}</nav>`;
}
function modeSelectHtml(){
  return `<label class="modeToggle">Mode
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
    root.innerHTML = `<h1>Neurobench</h1><span class="pageContext">${escapeHtml(config.label)} · ${escapeHtml(datasetId)}</span>${pageNavHtml()}${modeSelectHtml()}${themeSelectHtml()}`;
  }
}

function routePage(){
  const hash = (location.hash || '#home').replace(/^#\/?/, '');
  const page = hash === 'home' || hash === 'workflow' ? 'home' : hash === 'pipelines' || hash === 'architecture' || hash === 'architecture-lab' ? 'architecture' : hash === 'experiments' || hash === 'experiment-lab' ? 'experiments' : hash === 'progress' || hash === 'metrics' || hash === 'audit' ? 'metrics' : hash === 'data' || hash === 'data-compare' || hash === 'process' || hash === 'process-lab' || hash === 'qc' || hash === 'dataset-qc' ? 'qc' : hash === 'report' ? 'report' : 'review';
  const reviewSubpage = page === 'review' ? reviewSubPageFromHash(location.hash) : 'inspect';
  const dataSubpage = page === 'qc' ? dataSubPageFromHash(location.hash) : 'inspect';
  for(const link of document.querySelectorAll('[data-nav-page]')) link.classList.toggle('active', link.dataset.navPage === page);
  document.getElementById('homePage')?.classList.toggle('hidden', page !== 'home');
  document.getElementById('architecturePage').classList.toggle('hidden', page !== 'architecture');
  document.getElementById('experimentsPage').classList.toggle('hidden', page !== 'experiments');
  document.getElementById('metricsPage').classList.toggle('hidden', page !== 'metrics');
  document.getElementById('qcPage').classList.toggle('hidden', page !== 'qc');
  document.getElementById('reportPage').classList.toggle('hidden', page !== 'report');
  appRoot.classList.toggle('home-mode', page === 'home');
  appRoot.classList.toggle('arch-mode', page === 'architecture');
  appRoot.classList.toggle('lab-mode', page === 'metrics' || page === 'report' || page === 'experiments');
  appRoot.classList.toggle('qc-mode', page === 'qc');
  updateReviewSubnav(reviewSubpage);
  if(page !== 'review') updateReviewSubnav('inspect');
  updateDataSubnav(dataSubpage);
  if(page !== 'qc') updateDataSubnav('inspect');
  if(page === 'home') renderWorkflowHome();
  else if(page === 'architecture') renderArchitectureLab();
  else if(page === 'experiments') renderExperimentLab();
  else if(page === 'metrics') renderMetricsAudit();
  else if(page === 'qc' && dataSubpage === 'compare') renderDataCompare();
  else if(page === 'qc') renderDatasetQc();
  else if(page === 'report') renderReviewReport();
  else if(reviewSubpage === 'stencil') renderReviewStencil();
  else if(reviewSubpage === 'overlap') renderReviewOverlap();
  else if(reviewSubpage === 'triage') renderReviewTriage();
  else resizeOverlay();
  renderNextBestActions();
  appRoot.classList.remove('booting');
  if(!routePage.lastPage || routePage.lastPage !== page || routePage.lastReviewSubpage !== reviewSubpage || routePage.lastDataSubpage !== dataSubpage) {
    window.requestAnimationFrame(() => window.scrollTo({top: 0, left: 0, behavior: 'auto'}));
  }
  routePage.lastPage = page;
  routePage.lastReviewSubpage = reviewSubpage;
  routePage.lastDataSubpage = dataSubpage;
}

async function boot(){
  renderSharedPageChrome();
  populateEvidenceSelect();
  await loadAnnotations();
  if(serverBacked) {
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
  renderParams();
  const first = visibleRois()[0] || reviewRois()[0];
  selectedId = first?.id || null;
  selectedRoiIds = new Set(selectedId ? [String(selectedId)] : []);
  if(selectedId) {
    selectedEventFrame = eventsForRoi(selectedRoi())[0]?.frame || null;
    roiNotes.value = roiAnn(selectedId).notes || '';
    eventNotes.value = selectedEventFrame ? eventAnn(selectedId, selectedEventFrame).notes || '' : '';
  }
  if(selectedSuggestionId) {
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
