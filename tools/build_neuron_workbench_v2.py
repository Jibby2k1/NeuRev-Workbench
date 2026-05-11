#!/usr/bin/env python3
"""Build the v2 interactive neuron annotation workbench.

This is intentionally stdlib-only. It consumes the Fiji/Groovy-generated
review_data.json and writes a larger, keyboard-first browser workbench around
the existing frame PNGs, ROI footprints, traces, and event candidates.
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path("/home/jibby2k1/CNEL/State Analysis (Fish)/Separable-Gamma-CFAR")
APP_DIR = PROJECT_ROOT / "Outputs/NeuronReview/calcium_video_2/app"
DATA_PATH = APP_DIR / "review_data.json"


CSS = r"""
:root {
  color-scheme: light;
  --bg: #eef3f8;
  --panel: #ffffff;
  --ink: #111827;
  --muted: #64748b;
  --line: #cbd5e1;
  --soft: #f1f5f9;
  --accent: #0284c7;
  --accent-soft: #e0f2fe;
  --event: #facc15;
  --ok: #16a34a;
  --bad: #dc2626;
  --unsure: #9333ea;
  --viewer: #07111f;
  --shadow: 0 14px 32px rgba(15, 23, 42, 0.10);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Arial, Helvetica, sans-serif;
}
.app {
  display: grid;
  grid-template-columns: minmax(900px, 1fr) 470px;
  height: 100vh;
  overflow: hidden;
}
.stage {
  padding: 14px;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.side {
  border-left: 1px solid var(--line);
  background: var(--panel);
  padding: 14px;
  overflow: auto;
  box-shadow: -10px 0 28px rgba(15, 23, 42, 0.06);
}
h1 {
  font-size: 19px;
  letter-spacing: 0;
  margin: 0;
}
h2 {
  font-size: 13px;
  margin: 16px 0 8px;
  color: #334155;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.topbar,
.toolbar,
.buttonRow {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 7px;
}
.topbar {
  justify-content: space-between;
  min-height: 32px;
}
.toolbar {
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 7px 8px;
  box-shadow: 0 1px 0 rgba(255, 255, 255, 0.7) inset;
}
button,
select {
  height: 31px;
  border: 1px solid var(--line);
  background: #ffffff;
  border-radius: 6px;
  padding: 0 9px;
  color: var(--ink);
  font-weight: 600;
  font-size: 12px;
  box-shadow: 0 1px 1px rgba(15, 23, 42, 0.04);
}
button.active,
button:hover {
  border-color: #93c5fd;
  background: #eff6ff;
}
button.accept.active { background: #dcfce7; border-color: #86efac; }
button.reject.active { background: #fee2e2; border-color: #fca5a5; }
button.unsure.active { background: #f3e8ff; border-color: #d8b4fe; }
label {
  font-size: 12px;
  color: var(--muted);
  white-space: nowrap;
}
input[type="checkbox"] {
  accent-color: var(--accent);
}
input[type="range"] {
  vertical-align: middle;
  accent-color: var(--accent);
}
.viewerScroll {
  flex: 1;
  min-height: 500px;
  overflow: auto;
  background: var(--viewer);
  border: 1px solid #94a3b8;
  border-radius: 9px;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  box-shadow: var(--shadow);
}
.viewerWrap {
  position: relative;
  display: inline-block;
  margin: 14px;
  line-height: 0;
  box-shadow: 0 10px 26px rgba(0, 0, 0, 0.36);
}
#frameImg {
  display: block;
  width: 753px;
  height: auto;
  image-rendering: auto;
  user-select: none;
}
#overlay {
  position: absolute;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  cursor: crosshair;
}
.status {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  color: var(--muted);
  font-size: 12px;
  min-height: 16px;
}
.traceBox {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 9px;
  padding: 8px;
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
}
#traceCanvas {
  display: block;
  width: 100%;
  height: 240px;
}
.legend {
  display: flex;
  gap: 12px;
  color: var(--muted);
  font-size: 12px;
}
.dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 4px;
}
.metricGrid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.metric {
  border: 1px solid var(--line);
  background: linear-gradient(#ffffff, #f8fafc);
  border-radius: 8px;
  padding: 8px;
}
.metric b {
  display: block;
  font-size: 18px;
  color: #0f172a;
}
.metric span {
  color: var(--muted);
  font-size: 12px;
}
.roiList {
  max-height: 245px;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #ffffff;
}
.roiRow {
  display: grid;
  grid-template-columns: 44px 1fr 68px;
  gap: 6px;
  padding: 8px 9px;
  border-bottom: 1px solid #edf2f7;
  font-size: 12px;
  cursor: pointer;
}
.roiRow:hover,
.roiRow.sel {
  background: var(--accent-soft);
}
.roiRow.sel {
  box-shadow: inset 3px 0 0 var(--accent);
}
.roiRow.deleted {
  opacity: 0.48;
  text-decoration: line-through;
}
.badge {
  align-self: center;
  justify-self: end;
  border-radius: 5px;
  background: #e2e8f0;
  color: #334155;
  padding: 2px 6px;
  font-size: 11px;
}
.badge.accept { background: #dcfce7; color: #166534; }
.badge.reject { background: #fee2e2; color: #991b1b; }
.badge.unsure { background: #f3e8ff; color: #6b21a8; }
.eventList {
  max-height: 180px;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #ffffff;
}
.eventRow {
  display: grid;
  grid-template-columns: 58px 1fr 70px;
  padding: 6px 8px;
  border-bottom: 1px solid #edf2f7;
  font-size: 12px;
  cursor: pointer;
}
.eventRow:hover,
.eventRow.sel {
  background: #fef9c3;
}
textarea {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 7px;
  resize: vertical;
  font-family: Arial, Helvetica, sans-serif;
}
.smallTable {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
.smallTable th,
.smallTable td {
  border: 1px solid var(--line);
  padding: 5px;
  text-align: left;
}
.smallTable th { background: #f1f5f9; }
.hint {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.35;
}
.saveState {
  font-size: 12px;
  color: var(--muted);
  border: 1px solid var(--line);
  border-radius: 999px;
  background: #ffffff;
  padding: 5px 9px;
}
.saveState.ok { color: var(--ok); }
.saveState.bad { color: var(--bad); }
@media (max-width: 1180px) {
  .app { grid-template-columns: 1fr; height: auto; overflow: visible; }
  .side { border-left: 0; border-top: 1px solid var(--line); }
  .viewerScroll { min-height: 560px; }
}
"""


JS = r"""
const embedded = document.getElementById('review-data');
const data = JSON.parse(embedded.textContent);
const img = document.getElementById('frameImg');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const slider = document.getElementById('frameSlider');
const frameLabel = document.getElementById('frameLabel');
const statusEl = document.getElementById('statusText');
const selectionText = document.getElementById('selectionText');
const saveStateEl = document.getElementById('saveState');
const traceCanvas = document.getElementById('traceCanvas');
const traceCtx = traceCanvas.getContext('2d');
const roiNotes = document.getElementById('roiNotes');
const eventNotes = document.getElementById('eventNotes');
const viewerScroll = document.getElementById('viewerScroll');
const viewerWrap = document.getElementById('viewerWrap');
const storeKey = 'neuron-review-workbench-v2-calcium-video-2';

let currentFrame = 1;
let selectedId = data.rois.length ? data.rois[0].id : null;
let selectedEventFrame = null;
let playing = false;
let timer = null;
let saveTimer = null;
let serverBacked = location.protocol.startsWith('http');
let annotations = defaultAnnotations();

function defaultAnnotations() {
  return {
    version: 2,
    updatedAt: new Date().toISOString(),
    rois: {},
    events: {},
    settings: {
      eventThreshold: 2.4,
      kalmanGain: 0.06,
      spikeGain: 0.008,
      zoom: 3.0,
      brightness: 1,
      contrast: 1.08,
      overlayOpacity: 0.72,
      queue: 'unlabeled',
      minArea: 0,
      minEvents: 0
    }
  };
}

function mergeAnnotations(incoming) {
  annotations = Object.assign(defaultAnnotations(), incoming || {});
  annotations.rois = Object.assign({}, incoming?.rois || {});
  annotations.events = Object.assign({}, incoming?.events || {});
  annotations.settings = Object.assign(defaultAnnotations().settings, incoming?.settings || {});
}

async function loadAnnotations() {
  const local = localStorage.getItem(storeKey);
  if (local) mergeAnnotations(JSON.parse(local));
  if (serverBacked) {
    try {
      const res = await fetch('annotations.json', {cache: 'no-store'});
      if (res.ok) {
        mergeAnnotations(await res.json());
        setSaveState('autosave ready', 'ok');
      }
    } catch (_) {
      setSaveState('local browser save only', 'bad');
    }
  } else {
    setSaveState('static mode: export to save files', '');
  }
  applySettingsToControls();
}

function setSaveState(text, cls) {
  saveStateEl.textContent = text;
  saveStateEl.className = 'saveState ' + (cls || '');
}

function saveAnnotationsNow() {
  annotations.updatedAt = new Date().toISOString();
  localStorage.setItem(storeKey, JSON.stringify(annotations));
  if (!serverBacked) {
    setSaveState('saved in browser', 'ok');
    return;
  }
  fetch('annotations.json', {
    method: 'PUT',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(annotations, null, 2)
  }).then(res => {
    setSaveState(res.ok ? 'autosaved to annotations.json' : 'autosave failed', res.ok ? 'ok' : 'bad');
  }).catch(() => setSaveState('autosave failed', 'bad'));
}

function queueSave() {
  setSaveState('saving...', '');
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveAnnotationsNow, 220);
}

function framePath(frame){ return data.video.framePattern.replace('%03d', String(frame).padStart(3, '0')); }
function selectedRoi(){ return data.rois.find(r => r.id === selectedId) || data.rois[0] || null; }
function roiAnn(id){ return annotations.rois[id] || {state:'', notes:'', deleted:false}; }
function eventKey(roiId, frame){ return `${roiId}:${frame}`; }
function eventAnn(roiId, frame){ return annotations.events[eventKey(roiId, frame)] || {state:'', notes:''}; }
function setting(name){ return annotations.settings[name]; }
function setSetting(name, value){ annotations.settings[name] = value; queueSave(); }
function threshold(){ return Number(setting('eventThreshold')); }
function kalmanGain(){ return Number(setting('kalmanGain')); }
function spikeGain(){ return Number(setting('spikeGain')); }
function minAreaFilter(){ return Number(setting('minArea')); }
function minEventsFilter(){ return Number(setting('minEvents')); }

function median(arr){ const a = [...arr].sort((x,y)=>x-y); const m = Math.floor(a.length/2); return a.length % 2 ? a[m] : 0.5*(a[m-1]+a[m]); }
function madSigma(arr, center){ return Math.max(1e-6, 1.4826 * median(arr.map(v => Math.abs(v - center)))); }
function modeledTrace(roi){
  const gain = kalmanGain(), sgain = spikeGain();
  const center = median(roi.dffTrace);
  const sigma = madSigma(roi.dffTrace, center);
  let baseline = center;
  const baselineTrace = [], eventTrace = [], zTrace = [];
  for(const v of roi.dffTrace){
    const residual = v - baseline;
    let k = gain;
    if(residual > 2.5 * sigma) k = sgain;
    if(residual < -1.0 * sigma) k = Math.min(0.18, gain * 1.8);
    baseline += k * residual;
    baselineTrace.push(baseline);
    const ev = Math.max(0, v - baseline);
    eventTrace.push(ev);
    zTrace.push(ev / sigma);
  }
  return {baselineTrace, eventTrace, zTrace, sigma};
}
function eventsForRoi(roi){
  const zt = modeledTrace(roi).zTrace;
  const th = threshold();
  const out = [];
  for(let i=1;i<zt.length-1;i++){
    if(zt[i] >= th && zt[i] >= zt[i-1] && zt[i] >= zt[i+1]){
      out.push({frame:i+1, z:zt[i], amplitude:modeledTrace(roi).eventTrace[i]});
    }
  }
  return out;
}
function eventFrames(roi){ return eventsForRoi(roi).map(e => e.frame); }
function eventNearFrame(roi, frame){ return eventFrames(roi).some(f => Math.abs(f - frame) <= 1); }

function roiQualityScore(roi) {
  return roi.peakScore / Math.max(0.04, roi.noiseSigma) + eventsForRoi(roi).length * 0.4;
}
function roiUncertaintyScore(roi) {
  const ev = eventsForRoi(roi).length;
  const ann = roiAnn(roi.id);
  return (ann.state ? 0 : 20) + roi.noiseSigma * 12 + Math.abs(roi.area - 65) / 50 - ev * 0.15;
}
function visibleRois(){
  const queue = setting('queue');
  let rows = data.rois.filter(r => r.area >= minAreaFilter() && eventsForRoi(r).length >= minEventsFilter());
  rows = rows.filter(r => {
    const ann = roiAnn(r.id);
    if (queue !== 'deleted' && ann.deleted) return false;
    if (queue === 'unlabeled') return !ann.state;
    if (queue === 'accepted') return ann.state === 'accept';
    if (queue === 'rejected') return ann.state === 'reject';
    if (queue === 'unsure') return ann.state === 'unsure';
    if (queue === 'deleted') return ann.deleted;
    return true;
  });
  if (queue === 'highNoise') rows.sort((a,b) => b.noiseSigma - a.noiseSigma);
  else if (queue === 'highEvents') rows.sort((a,b) => eventsForRoi(b).length - eventsForRoi(a).length);
  else if (queue === 'uncertain') rows.sort((a,b) => roiUncertaintyScore(b) - roiUncertaintyScore(a));
  else rows.sort((a,b) => roiQualityScore(b) - roiQualityScore(a));
  return rows;
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
    ['minArea', 'minAreaLabel', 0],
    ['minEvents', 'minEventsLabel', 0]
  ];
  for (const [id, label, digits] of pairs) {
    const el = document.getElementById(id);
    if (!el) continue;
    el.value = setting(id);
    document.getElementById(label).textContent = Number(setting(id)).toFixed(digits);
  }
  document.getElementById('queueSelect').value = setting('queue');
  applyDisplaySettings();
}

function applyDisplaySettings() {
  img.style.width = `${data.video.width * Number(setting('zoom'))}px`;
  img.style.filter = `brightness(${setting('brightness')}) contrast(${setting('contrast')})`;
  ctx.globalAlpha = Number(setting('overlayOpacity'));
  resizeOverlay();
}

function resizeOverlay(){
  const rect = img.getBoundingClientRect();
  overlay.width = data.video.width;
  overlay.height = data.video.height;
  overlay.style.width = rect.width + 'px';
  overlay.style.height = rect.height + 'px';
  drawOverlay();
}

function drawOverlay(){
  ctx.clearRect(0,0,overlay.width,overlay.height);
  if(!document.getElementById('showRois').checked) return;
  const showLabels = document.getElementById('showLabels').checked;
  const showEvents = document.getElementById('showEvents').checked;
  const opacity = Number(setting('overlayOpacity'));
  for(const roi of visibleRois()){
    const ann = roiAnn(roi.id);
    const isSel = roi.id === selectedId;
    const isEvent = showEvents && eventNearFrame(roi, currentFrame);
    let color = ann.state === 'accept' ? '#16a34a' : ann.state === 'reject' ? '#dc2626' : ann.state === 'unsure' ? '#9333ea' : '#38bdf8';
    if(isEvent) color = '#facc15';
    ctx.globalAlpha = isSel ? 0.96 : opacity;
    ctx.fillStyle = color;
    for(const p of roi.points){ ctx.fillRect(p[0], p[1], 1, 1); }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = isSel ? '#ffffff' : color;
    ctx.lineWidth = isSel ? 2 : 1;
    const r = Math.max(4, Math.sqrt(roi.area / Math.PI) + 2);
    ctx.beginPath(); ctx.arc(roi.centroidX, roi.centroidY, r, 0, Math.PI*2); ctx.stroke();
    if(showLabels){
      ctx.font = '10px Arial';
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = '#111827';
      ctx.lineWidth = 3;
      ctx.strokeText(String(roi.id), roi.centroidX + 5, roi.centroidY - 5);
      ctx.fillText(String(roi.id), roi.centroidX + 5, roi.centroidY - 5);
    }
  }
}

function drawTrace(){
  const roi = selectedRoi();
  const w = traceCanvas.width, h = traceCanvas.height;
  traceCtx.clearRect(0,0,w,h);
  traceCtx.fillStyle = '#fff'; traceCtx.fillRect(0,0,w,h);
  if(!roi) return;
  const pad = 30;
  const model = modeledTrace(roi);
  const zScaled = model.zTrace.map(v => v * 0.05);
  const vals = [roi.dffTrace, model.baselineTrace, zScaled].flat();
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if(hi - lo < 1e-6){ hi = lo + 1; }
  function x(i){ return pad + i * (w - 2*pad) / (data.video.frames - 1); }
  function y(v){ return h - pad - (v - lo) * (h - 2*pad) / (hi - lo); }
  traceCtx.strokeStyle = '#e2e8f0'; traceCtx.lineWidth = 1;
  for(let k=0;k<5;k++){ const yy = pad + k*(h-2*pad)/4; traceCtx.beginPath(); traceCtx.moveTo(pad,yy); traceCtx.lineTo(w-pad,yy); traceCtx.stroke(); }
  const drawLine = (arr, color, width=1.6) => {
    traceCtx.strokeStyle=color; traceCtx.lineWidth=width; traceCtx.beginPath();
    arr.forEach((v,i)=>{ if(i===0) traceCtx.moveTo(x(i),y(v)); else traceCtx.lineTo(x(i),y(v)); });
    traceCtx.stroke();
  };
  drawLine(roi.dffTrace, '#2563eb');
  drawLine(model.baselineTrace, '#64748b');
  drawLine(zScaled, '#f59e0b');
  traceCtx.strokeStyle = '#ef4444'; traceCtx.lineWidth = 1;
  const xf = x(currentFrame - 1); traceCtx.beginPath(); traceCtx.moveTo(xf,pad); traceCtx.lineTo(xf,h-pad); traceCtx.stroke();
  for(const ev of eventsForRoi(roi)){
    const ann = eventAnn(roi.id, ev.frame);
    traceCtx.fillStyle = ann.state === 'accept' ? '#16a34a' : ann.state === 'reject' ? '#dc2626' : ann.state === 'unsure' ? '#9333ea' : '#facc15';
    traceCtx.beginPath(); traceCtx.arc(x(ev.frame-1), pad + 8, ev.frame === selectedEventFrame ? 5 : 3, 0, Math.PI*2); traceCtx.fill();
  }
  traceCtx.fillStyle = '#0f172a'; traceCtx.font = '13px Arial';
  traceCtx.fillText(`ROI ${roi.id} | area ${roi.area} | noise sigma ${model.sigma.toFixed(5)} | events ${eventsForRoi(roi).length}`, pad, 18);
}

function setFrame(frame){
  currentFrame = Math.max(1, Math.min(data.video.frames, frame));
  slider.value = currentFrame;
  frameLabel.textContent = currentFrame;
  img.src = framePath(currentFrame);
  statusEl.textContent = `Frame ${currentFrame} / ${data.video.frames}`;
  const roi = selectedRoi();
  selectionText.textContent = roi ? `ROI ${roi.id}${selectedEventFrame ? `, event f${selectedEventFrame}` : ''}` : '';
  drawTrace();
}

function selectRoi(id){
  selectedId = id;
  const roi = selectedRoi();
  selectedEventFrame = eventsForRoi(roi)[0]?.frame || null;
  roiNotes.value = roiAnn(id).notes || '';
  eventNotes.value = selectedEventFrame ? eventAnn(id, selectedEventFrame).notes || '' : '';
  setFrame(selectedEventFrame || currentFrame);
  renderAll();
}

function renderRoiList(){
  const root = document.getElementById('roiList');
  root.innerHTML = '';
  const rows = visibleRois();
  document.getElementById('visibleCount').textContent = rows.length;
  for(const roi of rows){
    const ann = roiAnn(roi.id);
    const row = document.createElement('div');
    row.className = 'roiRow' + (roi.id === selectedId ? ' sel' : '') + (ann.deleted ? ' deleted' : '');
    const state = ann.deleted ? 'deleted' : ann.state || 'new';
    row.innerHTML = `<b>#${roi.id}</b><span>${eventsForRoi(roi).length} events, area ${roi.area}, noise ${roi.noiseSigma}</span><span class="badge ${ann.state || ''}">${state}</span>`;
    row.onclick = () => selectRoi(roi.id);
    root.appendChild(row);
  }
}

function renderEventList(){
  const roi = selectedRoi();
  const root = document.getElementById('eventList');
  root.innerHTML = '';
  if(!roi) return;
  for(const ev of eventsForRoi(roi)){
    const ann = eventAnn(roi.id, ev.frame);
    const row = document.createElement('div');
    row.className = 'eventRow' + (ev.frame === selectedEventFrame ? ' sel' : '');
    row.innerHTML = `<b>f${ev.frame}</b><span>z ${ev.z.toFixed(2)}, amp ${ev.amplitude.toFixed(4)}</span><span class="badge ${ann.state || ''}">${ann.state || 'new'}</span>`;
    row.onclick = () => { selectedEventFrame = ev.frame; eventNotes.value = eventAnn(roi.id, ev.frame).notes || ''; setFrame(ev.frame); renderAll(); };
    root.appendChild(row);
  }
}

function updateCounts(){
  const allEvents = data.rois.reduce((sum, r) => sum + eventsForRoi(r).length, 0);
  let acc = 0, rej = 0, unsure = 0, eventAccepted = 0;
  for(const r of data.rois){
    const st = roiAnn(r.id).state;
    if(st === 'accept') acc++;
    if(st === 'reject') rej++;
    if(st === 'unsure') unsure++;
    for(const ev of eventsForRoi(r)) if(eventAnn(r.id, ev.frame).state === 'accept') eventAccepted++;
  }
  document.getElementById('roiCount').textContent = data.rois.length;
  document.getElementById('eventCount').textContent = allEvents;
  document.getElementById('acceptedCount').textContent = acc;
  document.getElementById('rejectedCount').textContent = rej;
  document.getElementById('unsureCount').textContent = unsure;
  document.getElementById('eventAcceptedCount').textContent = eventAccepted;
}

function setRoiState(state){
  const roi = selectedRoi(); if(!roi) return;
  annotations.rois[roi.id] = Object.assign(roiAnn(roi.id), {state});
  queueSave();
  renderAll();
}
function toggleDeleted(){
  const roi = selectedRoi(); if(!roi) return;
  const ann = Object.assign(roiAnn(roi.id), {deleted: !roiAnn(roi.id).deleted});
  annotations.rois[roi.id] = ann;
  queueSave();
  renderAll();
}
function setEventState(state){
  const roi = selectedRoi(); if(!roi || !selectedEventFrame) return;
  annotations.events[eventKey(roi.id, selectedEventFrame)] = Object.assign(eventAnn(roi.id, selectedEventFrame), {state});
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
}

function renderParams(){
  const rows = Object.entries(data.parameters).map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
  document.getElementById('paramTable').innerHTML = '<tr><th>Parameter</th><th>Value</th></tr>' + rows;
}

function renderAll(){
  updateCounts();
  renderButtons();
  renderRoiList();
  renderEventList();
  drawOverlay();
  drawTrace();
}

function exportRows(type) {
  const newline = String.fromCharCode(10);
  let rows = [];
  if (type === 'roi') {
    rows.push('roi_id\tstate\tdeleted\tnotes\tcentroid_x\tcentroid_y\tarea\tpeak_score\tevent_count\tnoise_sigma');
    for(const roi of data.rois){
      const ann = roiAnn(roi.id);
      const notes = (ann.notes || '').split(String.fromCharCode(9)).join(' ').split(newline).join(' ');
      rows.push([roi.id, ann.state || '', ann.deleted ? 1 : 0, notes, roi.centroidX, roi.centroidY, roi.area, roi.peakScore, eventsForRoi(roi).length, roi.noiseSigma].join('\t'));
    }
  } else {
    rows.push('roi_id\tframe\tstate\tnotes\tz\tamplitude\troi_state');
    for(const roi of data.rois){
      for(const ev of eventsForRoi(roi)){
        const ann = eventAnn(roi.id, ev.frame);
        const notes = (ann.notes || '').split(String.fromCharCode(9)).join(' ').split(newline).join(' ');
        rows.push([roi.id, ev.frame, ann.state || '', notes, ev.z.toFixed(4), ev.amplitude.toFixed(6), roiAnn(roi.id).state || ''].join('\t'));
      }
    }
  }
  const blob = new Blob([rows.join(newline) + newline], {type:'text/tab-separated-values'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = type === 'roi' ? 'neuron_roi_annotations.tsv' : 'neuron_event_annotations.tsv';
  a.click();
  URL.revokeObjectURL(a.href);
}

function nextRoi(delta){
  const rows = visibleRois();
  if(!rows.length) return;
  const idx = Math.max(0, rows.findIndex(r => r.id === selectedId));
  const next = rows[(idx + delta + rows.length) % rows.length];
  selectRoi(next.id);
}
function nextEvent(delta){
  const roi = selectedRoi();
  if(!roi) return;
  const events = eventsForRoi(roi);
  if(!events.length) return;
  const idx = Math.max(0, events.findIndex(e => e.frame === selectedEventFrame));
  selectedEventFrame = events[(idx + delta + events.length) % events.length].frame;
  eventNotes.value = eventAnn(roi.id, selectedEventFrame).notes || '';
  setFrame(selectedEventFrame);
  renderAll();
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
}

function initControls(){
  slider.max = data.video.frames;
  slider.oninput = () => setFrame(Number(slider.value));
  document.getElementById('playBtn').onclick = togglePlay;
  document.getElementById('fitBtn').onclick = fitWidth;
  document.getElementById('fullscreenBtn').onclick = () => viewerScroll.requestFullscreen?.();
  for(const id of ['showRois','showLabels','showEvents']) document.getElementById(id).onchange = drawOverlay;
  for(const id of ['eventThreshold','kalmanGain','spikeGain','zoom','brightness','contrast','overlayOpacity','minArea','minEvents']) {
    document.getElementById(id).oninput = e => {
      const value = Number(e.target.value);
      setSetting(id, value);
      applySettingsToControls();
      renderAll();
    };
  }
  document.getElementById('queueSelect').onchange = e => { setSetting('queue', e.target.value); renderAll(); };
  document.getElementById('acceptBtn').onclick = () => setRoiState('accept');
  document.getElementById('rejectBtn').onclick = () => setRoiState('reject');
  document.getElementById('unsureBtn').onclick = () => setRoiState('unsure');
  document.getElementById('clearBtn').onclick = () => setRoiState('');
  document.getElementById('deleteBtn').onclick = toggleDeleted;
  document.getElementById('eventAcceptBtn').onclick = () => setEventState('accept');
  document.getElementById('eventRejectBtn').onclick = () => setEventState('reject');
  document.getElementById('eventUnsureBtn').onclick = () => setEventState('unsure');
  document.getElementById('eventClearBtn').onclick = () => setEventState('');
  document.getElementById('exportRoiBtn').onclick = () => exportRows('roi');
  document.getElementById('exportEventBtn').onclick = () => exportRows('event');
  roiNotes.oninput = e => {
    const roi = selectedRoi(); if(!roi) return;
    annotations.rois[roi.id] = Object.assign(roiAnn(roi.id), {notes:e.target.value});
    queueSave();
  };
  eventNotes.oninput = e => {
    const roi = selectedRoi(); if(!roi || !selectedEventFrame) return;
    annotations.events[eventKey(roi.id, selectedEventFrame)] = Object.assign(eventAnn(roi.id, selectedEventFrame), {notes:e.target.value});
    queueSave();
  };
  overlay.onclick = e => {
    const rect = overlay.getBoundingClientRect();
    const x = (e.clientX - rect.left) * data.video.width / rect.width;
    const y = (e.clientY - rect.top) * data.video.height / rect.height;
    let best = null, bestD = Infinity;
    for(const roi of visibleRois()){
      const dx = x - roi.centroidX, dy = y - roi.centroidY, d = dx*dx + dy*dy;
      if(d < bestD){ bestD = d; best = roi; }
    }
    if(best) selectRoi(best.id);
  };
  document.addEventListener('keydown', e => {
    if(e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
    if(e.code === 'Space'){ e.preventDefault(); togglePlay(); }
    else if(e.key === 'ArrowRight') setFrame(currentFrame + 1);
    else if(e.key === 'ArrowLeft') setFrame(currentFrame - 1);
    else if(e.key === 'j') nextRoi(1);
    else if(e.key === 'k') nextRoi(-1);
    else if(e.key === 'n') nextEvent(1);
    else if(e.key === 'p') nextEvent(-1);
    else if(e.key === 'a') setRoiState('accept');
    else if(e.key === 'r') setRoiState('reject');
    else if(e.key === 'u') setRoiState('unsure');
    else if(e.key === 'e') setEventState('accept');
    else if(e.key === 'x') setEventState('reject');
    else if(e.key === 'f') viewerScroll.requestFullscreen?.();
  });
  img.onload = resizeOverlay;
  window.onresize = resizeOverlay;
}

async function boot(){
  await loadAnnotations();
  initControls();
  renderParams();
  const first = visibleRois()[0] || data.rois[0];
  selectedId = first?.id || null;
  if(selectedId) {
    selectedEventFrame = eventsForRoi(selectedRoi())[0]?.frame || null;
    roiNotes.value = roiAnn(selectedId).notes || '';
    eventNotes.value = selectedEventFrame ? eventAnn(selectedId, selectedEventFrame).notes || '' : '';
  }
  setFrame(1);
  renderAll();
}
boot();
"""


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Neuron Annotation Workbench - calcium_video_2</title>
<link rel="stylesheet" href="workbench.css">
</head>
<body>
<div class="app">
  <main class="stage">
    <div class="topbar">
      <h1>Neuron Annotation Workbench: calcium_video_2</h1>
      <span id="saveState" class="saveState">loading</span>
    </div>
    <div class="toolbar">
      <button id="playBtn">Play</button>
      <button id="fitBtn">Fit Width</button>
      <button id="fullscreenBtn">Fullscreen</button>
      <label>Frame <input id="frameSlider" type="range" min="1" max="{frames}" value="1"></label>
      <b id="frameLabel">1</b>
      <label>Zoom <input id="zoom" type="range" min="0.75" max="5" step="0.05"> <span id="zoomLabel">3.00</span></label>
      <label>Brightness <input id="brightness" type="range" min="0.4" max="2.5" step="0.02"> <span id="brightnessLabel">1.00</span></label>
      <label>Contrast <input id="contrast" type="range" min="0.5" max="3" step="0.02"> <span id="contrastLabel">1.08</span></label>
    </div>
    <div class="toolbar">
      <label><input id="showRois" type="checkbox" checked> ROIs</label>
      <label><input id="showLabels" type="checkbox" checked> IDs</label>
      <label><input id="showEvents" type="checkbox" checked> event frames</label>
      <label>Overlay <input id="overlayOpacity" type="range" min="0.1" max="1" step="0.02"> <span id="overlayOpacityLabel">0.72</span></label>
      <label>event z <input id="eventThreshold" type="range" min="1.2" max="5" step="0.1"> <span id="eventThresholdLabel">2.4</span></label>
      <label>Kalman gain <input id="kalmanGain" type="range" min="0.01" max="0.18" step="0.005"> <span id="kalmanGainLabel">0.060</span></label>
      <label>spike gain <input id="spikeGain" type="range" min="0" max="0.05" step="0.002"> <span id="spikeGainLabel">0.008</span></label>
    </div>
    <div class="viewerScroll" id="viewerScroll">
      <div class="viewerWrap" id="viewerWrap">
        <img id="frameImg" alt="video frame">
        <canvas id="overlay"></canvas>
      </div>
    </div>
    <div class="status">
      <span id="statusText"></span>
      <span id="selectionText"></span>
    </div>
    <div class="traceBox">
      <canvas id="traceCanvas" width="1000" height="260"></canvas>
      <div class="legend">
        <span><i class="dot" style="background:#2563eb"></i>dF/F</span>
        <span><i class="dot" style="background:#64748b"></i>Kalman baseline</span>
        <span><i class="dot" style="background:#f59e0b"></i>event z</span>
        <span><i class="dot" style="background:#facc15"></i>called event</span>
      </div>
    </div>
  </main>
  <aside class="side">
    <h2>Dataset</h2>
    <div class="metricGrid">
      <div class="metric"><b id="roiCount"></b><span>candidate ROIs</span></div>
      <div class="metric"><b id="visibleCount"></b><span>visible in queue</span></div>
      <div class="metric"><b id="eventCount"></b><span>events at threshold</span></div>
      <div class="metric"><b id="eventAcceptedCount"></b><span>accepted events</span></div>
      <div class="metric"><b id="acceptedCount"></b><span>accepted ROIs</span></div>
      <div class="metric"><b id="rejectedCount"></b><span>rejected ROIs</span></div>
      <div class="metric"><b id="unsureCount"></b><span>unsure ROIs</span></div>
      <div class="metric"><b>{frames}</b><span>frames</span></div>
    </div>
    <h2>ROI Review</h2>
    <div class="buttonRow">
      <button id="acceptBtn" class="accept">Accept</button>
      <button id="rejectBtn" class="reject">Reject</button>
      <button id="unsureBtn" class="unsure">Unsure</button>
      <button id="clearBtn">Clear</button>
      <button id="deleteBtn">Hide ROI</button>
    </div>
    <textarea id="roiNotes" rows="3" placeholder="Notes for selected ROI"></textarea>
    <h2>Event Review</h2>
    <div class="buttonRow">
      <button id="eventAcceptBtn" class="accept">Accept Event</button>
      <button id="eventRejectBtn" class="reject">Reject Event</button>
      <button id="eventUnsureBtn" class="unsure">Unsure</button>
      <button id="eventClearBtn">Clear</button>
    </div>
    <textarea id="eventNotes" rows="2" placeholder="Notes for selected event"></textarea>
    <div class="eventList" id="eventList"></div>
    <h2>Review Queue</h2>
    <div class="toolbar">
      <select id="queueSelect">
        <option value="unlabeled">Unlabeled first</option>
        <option value="uncertain">Most uncertain</option>
        <option value="highEvents">High event count</option>
        <option value="highNoise">High noise</option>
        <option value="accepted">Accepted</option>
        <option value="rejected">Rejected</option>
        <option value="unsure">Unsure</option>
        <option value="deleted">Hidden</option>
        <option value="all">All active</option>
      </select>
      <label>area >= <input id="minArea" type="range" min="0" max="260" step="1"> <span id="minAreaLabel">0</span></label>
      <label>events >= <input id="minEvents" type="range" min="0" max="12" step="1"> <span id="minEventsLabel">0</span></label>
    </div>
    <div class="roiList" id="roiList"></div>
    <h2>Export</h2>
    <div class="buttonRow">
      <button id="exportRoiBtn">Export ROI TSV</button>
      <button id="exportEventBtn">Export Event TSV</button>
    </div>
    <h2>Parameters</h2>
    <table class="smallTable" id="paramTable"></table>
    <p class="hint">Run through the local server for autosave to <code>annotations.json</code>. Static file mode still keeps a browser-local backup and supports TSV export.</p>
  </aside>
</div>
<script id="review-data" type="application/json">{data_json}</script>
<script src="workbench.js"></script>
</body>
</html>
"""


def main() -> None:
    data = json.loads(DATA_PATH.read_text())
    APP_DIR.mkdir(parents=True, exist_ok=True)
    (APP_DIR / "workbench.css").write_text(CSS.strip() + "\n")
    (APP_DIR / "workbench.js").write_text(JS.strip() + "\n")
    html = HTML_TEMPLATE.format(
        frames=data["video"]["frames"],
        data_json=json.dumps(data, separators=(",", ":")).replace("</script>", "<\\/script>"),
    )
    (APP_DIR / "index.html").write_text(html)
    annotations_path = APP_DIR / "annotations.json"
    if not annotations_path.exists():
        annotations_path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "updatedAt": None,
                    "rois": {},
                    "events": {},
                    "settings": {},
                },
                indent=2,
            )
            + "\n"
        )
    print(f"Wrote workbench to {APP_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
