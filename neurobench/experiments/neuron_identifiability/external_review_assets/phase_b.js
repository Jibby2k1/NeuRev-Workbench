(() => {
  const C=window.REVIEW_CONFIG, $=id=>document.getElementById(id);
  let current=C.items[0].blind_id;
  let state={ratings:{},phase_order_certified:false,locked:false};
  const blank=()=>({neuron_call:'',identity_relation:'unresolved',confidence:3,raw_visibility:'uncertain',ica_visibility:'uncertain',ls_visibility:'uncertain',morphology_flags:[],limitation_flags:[],offset_dx_px:0,offset_dy_px:0,notes:''});
  const reviewer=()=>$('reviewerId').value.trim();
  const storageKey=()=>`${C.package_id}:${C.phase}:${reviewer()||'unassigned'}`;
  function load(){try{state={...state,...JSON.parse(localStorage.getItem(storageKey())||'{}')};}catch(_){};render();}
  function save(msg='Draft saved locally'){if(state.locked)return;localStorage.setItem(storageKey(),JSON.stringify(state));$('saveState').textContent=msg;renderProgress();renderRail();}
  function rating(){if(!state.ratings[current])state.ratings[current]=blank();return state.ratings[current];}
  function download(name,payload){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
  function payload(locked){return{schema_version:C.schema_version,package_id:C.package_id,phase:C.phase,reviewer_id:reviewer(),locked,exported_at:new Date().toISOString(),ratings:state.ratings,phase_order_certified:state.phase_order_certified};}
  function completeRating(r){return C.calls.includes(r?.neuron_call)&&C.identity_relations.includes(r?.identity_relation)&&C.stage_visibility.includes(r?.raw_visibility)&&C.stage_visibility.includes(r?.ica_visibility)&&C.stage_visibility.includes(r?.ls_visibility);}
  function renderRail(){$('itemList').innerHTML=C.items.map(x=>`<button class="itemButton ${x.blind_id===current?'active':''} ${completeRating(state.ratings[x.blind_id])?'complete':''}" data-id="${x.blind_id}">${x.blind_id}</button>`).join('');$('itemList').querySelectorAll('button').forEach(b=>b.onclick=()=>{current=b.dataset.id;renderRail();renderMedia();renderEditor();});}
  function renderMedia(){const item=C.items.find(x=>x.blind_id===current);$('currentItem').textContent=`${item.blind_id} · synchronized six-panel review`;$('frameReadout').textContent='Complete recording';$('mediaHint').textContent='Left: Raw, ICA, and local-standardized close-ups. Right: exact-center complete traces. Yellow cross indicates the review center.';$('mediaHost').innerHTML=`<video class="reviewVideo" controls preload="metadata" src="${item.video}"></video>`;}
  const options=(values,selected)=>values.map(x=>`<option value="${x}" ${x===selected?'selected':''}>${x}</option>`).join('');
  const checks=(values,selected,key)=>values.map(x=>`<label><input type="checkbox" data-array="${key}" value="${x}" ${(selected||[]).includes(x)?'checked':''}>${x}</label>`).join('');
  function renderEditor(){const r=rating();$('editor').innerHTML=`
    <div class="fieldGrid two"><label>Neuron call<select data-key="neuron_call"><option value="">Choose…</option>${options(C.calls,r.neuron_call)}</select></label><label>Confidence<select data-key="confidence">${options(C.confidence_values.map(String),String(r.confidence))}</select></label></div>
    <label>Identity relation<select data-key="identity_relation">${options(C.identity_relations,r.identity_relation)}</select></label>
    <h3>Stage visibility</h3><div class="fieldGrid two"><label>Raw<select data-key="raw_visibility">${options(C.stage_visibility,r.raw_visibility)}</select></label><label>ICA<select data-key="ica_visibility">${options(C.stage_visibility,r.ica_visibility)}</select></label><label>Local standardized<select data-key="ls_visibility">${options(C.stage_visibility,r.ls_visibility)}</select></label></div>
    <h3>Morphology</h3><div class="checkGrid">${checks(C.morphology_flags,r.morphology_flags,'morphology_flags')}</div>
    <h3>Limitations</h3><div class="checkGrid">${checks(C.limitation_flags,r.limitation_flags,'limitation_flags')}</div>
    <div class="fieldGrid two"><label>Suggested center offset dx (pixels)<input type="number" step="1" data-key="offset_dx_px" value="${r.offset_dx_px}"></label><label>Suggested center offset dy (pixels)<input type="number" step="1" data-key="offset_dy_px" value="${r.offset_dy_px}"></label></div>
    <label>Notes<textarea data-key="notes">${r.notes||''}</textarea></label>`;
    $('editor').querySelectorAll('[data-key]').forEach(el=>el.onchange=()=>{r[el.dataset.key]=el.type==='number'||el.dataset.key==='confidence'?Number(el.value):el.value;save();});
    $('editor').querySelectorAll('[data-array]').forEach(el=>el.onchange=()=>{r[el.dataset.array]=[...$('editor').querySelectorAll(`[data-array="${el.dataset.array}"]:checked`)].map(x=>x.value);save();});
    if(state.locked)$('editor').querySelectorAll('input,select,textarea').forEach(x=>x.disabled=true);
  }
  function renderCompletion(){$('completionChecks').innerHTML='<p class="hint">Every item must have a neuron call, identity relation, and visibility judgment for all three stages.</p>';$('certifyText').textContent='I completed and submitted Phase A before opening this assisted package, and I reviewed these items independently.';$('certify').checked=state.phase_order_certified;$('certify').disabled=state.locked;$('certify').onchange=()=>{state.phase_order_certified=$('certify').checked;save();};}
  function complete(){return reviewer()&&state.phase_order_certified&&C.items.every(x=>completeRating(state.ratings[x.blind_id]));}
  function renderProgress(){const n=C.items.filter(x=>completeRating(state.ratings[x.blind_id])).length;$('progressText').textContent=`${n} / ${C.items.length} candidates complete`;$('progressBar').style.width=`${100*n/C.items.length}%`;$('finalBtn').disabled=!complete()||state.locked;}
  function render(){$('appTitle').textContent=C.title;$('appInstructions').textContent=C.instructions;renderRail();renderMedia();renderEditor();renderCompletion();renderProgress();if(state.locked)$('saveState').textContent='Final submission locked';}
  $('reviewerId').onchange=()=>load();
  $('backupBtn').onclick=()=>download(`${reviewer()||'unassigned'}_phase_B_DRAFT.json`,payload(false));
  $('finalBtn').onclick=()=>{if(!complete())return;if(!confirm('Lock this Phase B submission?'))return;state.locked=true;localStorage.setItem(storageKey(),JSON.stringify(state));download(`${reviewer()}_phase_B_LOCKED.json`,payload(true));render();};
  load();
})();
