(() => {
  const C = window.REVIEW_CONFIG;
  const $ = id => document.getElementById(id);
  let current = C.clips[0].clip_id;
  let selectedMark = null;
  let state = {marks: [], coverage: {}, full_region_certified: false, locked: false};

  function reviewer() { return $('reviewerId').value.trim(); }
  function storageKey() { return `${C.package_id}:${C.phase}:${reviewer() || 'unassigned'}`; }
  function load() {
    try { state = {...state, ...JSON.parse(localStorage.getItem(storageKey()) || '{}')}; } catch (_) {}
    render();
  }
  function save(message='Draft saved locally') {
    if (state.locked) return;
    localStorage.setItem(storageKey(), JSON.stringify(state));
    $('saveState').textContent = message;
    renderProgress();
  }
  function download(name, payload) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {type:'application/json'}));
    a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  function payload(locked) {
    return {schema_version:C.schema_version, package_id:C.package_id, phase:C.phase,
      reviewer_id:reviewer(), locked, exported_at:new Date().toISOString(), marks:state.marks,
      coverage:state.coverage, full_region_certified:state.full_region_certified};
  }
  function currentClip() { return C.clips.find(x => x.clip_id === current); }
  function marks() { return state.marks.filter(x => x.clip_id === current); }
  function nextMarkId() {
    const n = 1 + Math.max(0, ...state.marks.map(x => Number(String(x.mark_id).replace(/\D/g,'')) || 0));
    return `M${String(n).padStart(3,'0')}`;
  }
  function addMark(x, y, frame) {
    if (state.locked) return;
    const clip = currentClip();
    const g = C.display_geometry, r = C.region;
    if (y < g.header_px) { $('saveState').textContent = 'Header clicks are ignored'; return; }
    const mark = {mark_id:nextMarkId(), clip_id:current,
      x_px_global:Number((r.x0 + x/g.crop_scale).toFixed(2)),
      y_px_global:Number((r.y0 + (y-g.header_px)/g.crop_scale).toFixed(2)),
      identity_id:'', class:'unresolved', visibility_confidence:3,
      onset_ui:'', peak_ui:frame, end_ui:'', source_overlap_ids:'', notes:''};
    state.marks.push(mark); selectedMark = mark.mark_id; save('Source mark added'); renderEditor(); drawOverlay();
  }
  function renderMedia() {
    const clip = currentClip();
    $('currentItem').textContent = `${clip.clip_id} · Raw-only clip`;
    $('mediaHint').textContent = 'Pause at the clearest frame, then click the source center. Header clicks are ignored. Mean and maximum projections are reference views only.';
    $('mediaHost').innerHTML = `<div><div class="rawWrap"><video id="reviewVideo" class="reviewVideo" controls preload="metadata" src="${clip.video}"></video><canvas id="overlay"></canvas></div><div class="projectionGrid"><figure><img src="${clip.mean_projection}" alt="Mean projection"><figcaption class="hint">Mean projection</figcaption></figure><figure><img src="${clip.max_projection}" alt="Maximum projection"><figcaption class="hint">Maximum projection</figcaption></figure></div></div>`;
    const video = $('reviewVideo'), canvas = $('overlay');
    const sync = () => {
      canvas.width = video.videoWidth || C.display_geometry.intrinsic_width;
      canvas.height = video.videoHeight || C.display_geometry.intrinsic_height;
      drawOverlay(); updateFrame();
    };
    video.addEventListener('loadedmetadata', sync); video.addEventListener('timeupdate', updateFrame);
    canvas.addEventListener('click', event => {
      const rect = canvas.getBoundingClientRect();
      const x = (event.clientX-rect.left)*canvas.width/rect.width;
      const y = (event.clientY-rect.top)*canvas.height/rect.height;
      addMark(x, y, frameNow());
    });
    sync();
  }
  function frameNow() {
    const clip=currentClip(), video=$('reviewVideo');
    return Math.min(clip.last_frame_ui, clip.first_frame_ui + Math.floor((video?.currentTime || 0)*clip.fps));
  }
  function updateFrame() { $('frameReadout').textContent = `UI frame ${frameNow()}`; }
  function drawOverlay() {
    const canvas=$('overlay'); if (!canvas) return;
    const ctx=canvas.getContext('2d'), g=C.display_geometry, r=C.region;
    ctx.clearRect(0,0,canvas.width,canvas.height); ctx.font='bold 16px sans-serif';
    marks().forEach(mark => {
      const x=(mark.x_px_global-r.x0)*g.crop_scale, y=(mark.y_px_global-r.y0)*g.crop_scale+g.header_px;
      ctx.strokeStyle = mark.mark_id===selectedMark ? '#f3bd59' : '#61d4de'; ctx.lineWidth=2;
      ctx.beginPath(); ctx.arc(x,y,8,0,Math.PI*2); ctx.stroke();
      ctx.fillStyle=ctx.strokeStyle; ctx.fillText(mark.mark_id,x+10,y-8);
    });
  }
  function field(label, key, type='text') {
    const mark=state.marks.find(x=>x.mark_id===selectedMark); if(!mark) return '';
    return `<label>${label}<input data-key="${key}" type="${type}" value="${mark[key] ?? ''}"></label>`;
  }
  function renderEditor() {
    const list=marks();
    if (!list.some(x=>x.mark_id===selectedMark)) selectedMark=list[0]?.mark_id || null;
    const mark=state.marks.find(x=>x.mark_id===selectedMark);
    $('editor').innerHTML = `<div class="markList">${list.map(x=>`<div class="markRow"><button data-select="${x.mark_id}">${x.mark_id} · (${x.x_px_global}, ${x.y_px_global}) · ${x.class}</button></div>`).join('') || '<p class="hint">No sources marked in this clip.</p>'}</div>` + (mark ? `
      <div class="fieldGrid two">${field('Identity across clips','identity_id')}
      <label>Source class<select data-key="class">${C.classes.map(x=>`<option ${x===mark.class?'selected':''}>${x}</option>`).join('')}</select></label>
      ${field('Confidence 1–5','visibility_confidence','number')}${field('Overlapping mark IDs','source_overlap_ids')}</div>
      <div class="fieldGrid two">${field('Onset UI frame','onset_ui','number')}${field('Peak UI frame','peak_ui','number')}${field('End UI frame','end_ui','number')}</div>
      <label>Notes<textarea data-key="notes">${mark.notes || ''}</textarea></label>
      <div class="toolbar"><button id="useFrame">Use current frame as peak</button><button id="deleteMark" class="danger">Delete selected mark</button></div>` : '');
    $('editor').querySelectorAll('[data-select]').forEach(btn=>btn.onclick=()=>{selectedMark=btn.dataset.select;renderEditor();drawOverlay();});
    $('editor').querySelectorAll('[data-key]').forEach(el=>el.onchange=()=>{const m=state.marks.find(x=>x.mark_id===selectedMark);m[el.dataset.key]=el.type==='number'&&el.value!==''?Number(el.value):el.value;save();renderEditor();drawOverlay();});
    if ($('useFrame')) $('useFrame').onclick=()=>{mark.peak_ui=frameNow();save();renderEditor();};
    if ($('deleteMark')) $('deleteMark').onclick=()=>{if(confirm('Delete this source mark?')){state.marks=state.marks.filter(x=>x.mark_id!==selectedMark);selectedMark=null;save();renderEditor();drawOverlay();}};
    if(state.locked) $('editor').querySelectorAll('input,select,textarea,button').forEach(x=>x.disabled=true);
  }
  function renderRail() {
    $('itemList').innerHTML=C.clips.map(x=>`<button class="itemButton ${x.clip_id===current?'active':''} ${state.coverage[x.clip_id]?'complete':''}" data-id="${x.clip_id}">${x.clip_id}</button>`).join('');
    $('itemList').querySelectorAll('button').forEach(btn=>btn.onclick=()=>{current=btn.dataset.id;selectedMark=null;renderRail();renderMedia();renderEditor();});
  }
  function renderCompletion() {
    $('completionChecks').innerHTML=`<div class="completionList">${C.clips.map(x=>`<label><input type="checkbox" data-cover="${x.clip_id}" ${state.coverage[x.clip_id]?'checked':''} ${state.locked?'disabled':''}> ${x.clip_id}: entire field reviewed</label>`).join('')}</div>`;
    $('completionChecks').querySelectorAll('[data-cover]').forEach(x=>x.onchange=()=>{state.coverage[x.dataset.cover]=x.checked;save();renderRail();});
    $('certifyText').textContent='I reviewed the entire region in every clip independently and did not view assisted evidence first.';
    $('certify').checked=state.full_region_certified; $('certify').disabled=state.locked;
    $('certify').onchange=()=>{state.full_region_certified=$('certify').checked;save();};
  }
  function complete() { return C.clips.every(x=>state.coverage[x.clip_id]) && state.full_region_certified && reviewer(); }
  function renderProgress() {
    const n=C.clips.filter(x=>state.coverage[x.clip_id]).length;
    $('progressText').textContent=`${n} / ${C.clips.length} clips signed off · ${state.marks.length} source marks`;
    $('progressBar').style.width=`${100*n/C.clips.length}%`;
    $('finalBtn').disabled=!complete()||state.locked;
  }
  function render() {
    $('appTitle').textContent=C.title; $('appInstructions').textContent=C.instructions;
    renderRail(); renderMedia(); renderEditor(); renderCompletion(); renderProgress();
    if(state.locked) $('saveState').textContent='Final submission locked';
  }
  $('reviewerId').onchange=()=>load();
  $('backupBtn').onclick=()=>download(`${reviewer()||'unassigned'}_phase_A_DRAFT.json`,payload(false));
  $('finalBtn').onclick=()=>{if(!complete())return;if(!confirm('Lock this Phase A submission? Do not edit it after sending.'))return;state.locked=true;localStorage.setItem(storageKey(),JSON.stringify(state));download(`${reviewer()}_phase_A_LOCKED.json`,payload(true));render();};
  load();
})();
