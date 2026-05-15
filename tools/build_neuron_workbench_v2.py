#!/usr/bin/env python3
"""Build the v2 interactive neuron annotation workbench.

This is intentionally stdlib-only. It consumes the Fiji/Groovy-generated
review_data.json and writes a larger, keyboard-first browser workbench around
the existing frame PNGs, ROI footprints, traces, and event candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurobench.workbench.builder import build_workbench, resolve_build_inputs


DEFAULT_APP_DIR = PROJECT_ROOT / "Outputs/NeuronReview/calcium_video_2/app"
DEFAULT_DATA_PATH = DEFAULT_APP_DIR / "review_data.json"
ASSET_DIR = PROJECT_ROOT / "neurobench/workbench/assets"


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
.app.arch-mode {
  display: block;
  overflow: auto;
}
.app.lab-mode {
  display: block;
  overflow: auto;
}
.app.qc-mode {
  display: block;
  overflow: auto;
}
.app.arch-mode .reviewOnly,
.app.lab-mode .reviewOnly,
.app.qc-mode .reviewOnly {
  display: none;
}
.hidden {
  display: none !important;
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
.navTabs {
  display: flex;
  gap: 6px;
  align-items: center;
}
.navTabs a {
  color: var(--ink);
  text-decoration: none;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: #fff;
  padding: 6px 10px;
  font-size: 12px;
  font-weight: 700;
}
.navTabs a.active {
  background: var(--accent-soft);
  border-color: #7dd3fc;
  color: #075985;
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
#evidenceImg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: fill;
  opacity: 0;
  pointer-events: none;
  mix-blend-mode: screen;
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
.contextPanel {
  border: 1px solid var(--line);
  background: #ffffff;
  border-radius: 8px;
  padding: 8px;
}
#roiCropCanvas {
  display: block;
  width: 100%;
  max-width: 260px;
  aspect-ratio: 1;
  background: #08111f;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
}
.filmstrip {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 6px;
  margin-top: 8px;
}
.filmFrame {
  height: 52px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background-color: #08111f;
  background-repeat: no-repeat;
  cursor: pointer;
  position: relative;
  overflow: hidden;
}
.filmFrame.active {
  border-color: #facc15;
  box-shadow: 0 0 0 2px rgba(250, 204, 21, 0.45);
}
.filmFrame span {
  position: absolute;
  right: 3px;
  bottom: 2px;
  background: rgba(15, 23, 42, 0.78);
  color: #fff;
  font-size: 10px;
  border-radius: 4px;
  padding: 1px 4px;
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
.eventList,
.suggestionList {
  max-height: 180px;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #ffffff;
}
.suggestionList {
  max-height: 210px;
}
.eventRow,
.suggestionRow {
  display: grid;
  grid-template-columns: 58px 1fr 70px;
  padding: 6px 8px;
  border-bottom: 1px solid #edf2f7;
  font-size: 12px;
  cursor: pointer;
}
.eventRow:hover,
.eventRow.sel,
.suggestionRow:hover,
.suggestionRow.sel {
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
.architecturePage {
  padding: 16px;
  height: 100vh;
  overflow: auto;
}
.archGrid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 12px;
}
.archCard {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
}
.archCard h3 {
  margin: 0 0 8px;
  font-size: 15px;
}
.archMeta {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin: 8px 0;
}
.archMeta div {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 7px;
}
.archMeta b {
  display: block;
  font-size: 16px;
}
.archEvidence {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}
.archEvidence span {
  background: #e0f2fe;
  color: #075985;
  border-radius: 999px;
  padding: 3px 7px;
  font-size: 11px;
}
.compareTable td.deltaGood {
  color: #166534;
  font-weight: 700;
}
.compareTable td.deltaBad {
  color: #991b1b;
  font-weight: 700;
}
.compareTable td.deltaNeutral {
  color: #334155;
  font-weight: 700;
}
.auditBars {
  display: grid;
  gap: 10px;
  margin: 12px 0;
}
.auditRow {
  display: grid;
  grid-template-columns: 130px 1fr 54px;
  gap: 8px;
  align-items: center;
  font-size: 12px;
}
.auditBar {
  height: 12px;
  background: #e2e8f0;
  border-radius: 999px;
  overflow: hidden;
}
.auditFill {
  height: 100%;
  background: linear-gradient(90deg, #0284c7, #16a34a);
}
.auditSplit {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
}
.qcMapGrid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin-top: 12px;
}
.qcMap {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 8px;
}
.qcMap img {
  width: 100%;
  image-rendering: auto;
  border-radius: 5px;
  background: #0b1220;
}
.qcWarnings {
  display: grid;
  gap: 8px;
  margin: 12px 0;
}
.qcWarning {
  border: 1px solid #fde68a;
  background: #fffbeb;
  color: #92400e;
  border-radius: 8px;
  padding: 9px;
  font-size: 12px;
}
@media (max-width: 1180px) {
  .app { grid-template-columns: 1fr; height: auto; overflow: visible; }
  .side { border-left: 0; border-top: 1px solid var(--line); }
  .viewerScroll { min-height: 560px; }
}
"""


JS = r"""
const embedded = document.getElementById('review-data');
const data = JSON.parse(embedded.textContent);
const appRoot = document.getElementById('appRoot');
const img = document.getElementById('frameImg');
const evidenceImg = document.getElementById('evidenceImg');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const slider = document.getElementById('frameSlider');
const frameLabel = document.getElementById('frameLabel');
const statusEl = document.getElementById('statusText');
const selectionText = document.getElementById('selectionText');
const saveStateEl = document.getElementById('saveState');
const traceCanvas = document.getElementById('traceCanvas');
const traceCtx = traceCanvas.getContext('2d');
const cropCanvas = document.getElementById('roiCropCanvas');
const cropCtx = cropCanvas?.getContext('2d');
const roiNotes = document.getElementById('roiNotes');
const eventNotes = document.getElementById('eventNotes');
const viewerScroll = document.getElementById('viewerScroll');
const viewerWrap = document.getElementById('viewerWrap');
const datasetId = data.dataset?.dataset_id || data.video?.name || 'calcium-video';
const storeKey = `neuron-review-workbench-v3-${datasetId}`;
const traceCache = new Map();
const traceEventCache = new Map();
const TRACE_CACHE_LIMIT = 512;
const traceCacheStats = {traceHits:0, traceMisses:0, eventHits:0, eventMisses:0, clears:0, lastClearReason:''};

let currentFrame = 1;
let selectedId = data.rois.length ? data.rois[0].id : null;
let selectedEventFrame = null;
let selectedSuggestionId = data.discovery?.suggestions?.[0]?.id || null;
let playing = false;
let timer = null;
let saveTimer = null;
let serverBacked = location.protocol.startsWith('http');
const ownerTokenKey = `${storeKey}-owner-token`;
let generationOwnerToken = localStorage.getItem(ownerTokenKey) || '';
let annotations = defaultAnnotations();

function defaultAnnotations() {
  return {
    version: 3,
    schema_version: 3,
    updatedAt: new Date().toISOString(),
    rois: {},
    events: {},
    suggestions: {},
    promotedRois: {},
    reviewStats: {
      sessionStartedAt: new Date().toISOString(),
      lastActionAt: null,
      actions: {}
    },
    settings: {
      eventThreshold: 2.4,
      kalmanGain: 0.06,
      spikeGain: 0.008,
      zoom: 3.0,
      brightness: 1,
      contrast: 1.08,
      overlayOpacity: 0.72,
      queue: 'unlabeled',
      discoveryQueue: 'all',
      evidenceMap: data.discovery?.evidenceMaps?.[0]?.id || '',
      showEvidence: false,
      showSuggestions: true,
      minArea: 0,
      minEvents: 0
    }
  };
}

function mergeAnnotations(incoming) {
  annotations = Object.assign(defaultAnnotations(), incoming || {});
  annotations.version = 3;
  annotations.schema_version = 3;
  annotations.rois = {};
  for(const [id, ann] of Object.entries(incoming?.rois || {})) annotations.rois[id] = migrateRoiAnn(ann);
  annotations.events = {};
  for(const [id, ann] of Object.entries(incoming?.events || {})) annotations.events[id] = migrateEventAnn(ann);
  annotations.suggestions = Object.assign({}, incoming?.suggestions || {});
  annotations.promotedRois = Object.assign({}, incoming?.promotedRois || {});
  annotations.reviewStats = Object.assign(defaultAnnotations().reviewStats, incoming?.reviewStats || {});
  annotations.reviewStats.actions = Object.assign({}, incoming?.reviewStats?.actions || {});
  annotations.settings = Object.assign(defaultAnnotations().settings, incoming?.settings || {});
}

function migrateRoiAnn(ann) {
  const out = Object.assign({state:'', notes:'', deleted:false}, ann || {});
  if(!out.cell_state) out.cell_state = out.state === 'accept' ? 'accepted' : out.state === 'reject' ? 'rejected' : out.state === 'unsure' ? 'unsure' : '';
  if(out.cell_state && !out.state) out.state = out.cell_state === 'accepted' ? 'accept' : out.cell_state === 'rejected' ? 'reject' : out.cell_state === 'unsure' ? 'unsure' : '';
  out.trace_quality = out.trace_quality || '';
  out.control_ready = out.control_ready || '';
  out.artifact_class = out.artifact_class || out.artifactClass || '';
  out.identity_group = out.identity_group || '';
  out.needs_action = out.needs_action || '';
  out.reason_tags = Array.isArray(out.reason_tags) ? out.reason_tags : [];
  out.confidence = ['low', 'medium', 'high'].includes(out.confidence) ? out.confidence : '';
  return out;
}

function migrateEventAnn(ann) {
  const out = Object.assign({state:'', notes:''}, ann || {});
  if(!out.event_state) out.event_state = out.state === 'accept' ? 'accepted' : out.state === 'reject' ? 'rejected' : out.state === 'unsure' ? 'unsure' : '';
  if(out.event_state && !out.state) out.state = out.event_state === 'accepted' ? 'accept' : out.event_state === 'rejected' ? 'reject' : out.event_state === 'unsure' ? 'unsure' : '';
  out.event_type = out.event_type || '';
  out.timing_quality = out.timing_quality || '';
  out.reason_tags = Array.isArray(out.reason_tags) ? out.reason_tags : [];
  out.confidence = ['low', 'medium', 'high'].includes(out.confidence) ? out.confidence : '';
  return out;
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
function roiAnn(id){ return annotations.rois[id] || migrateRoiAnn({}); }
function eventKey(roiId, frame){ return `${roiId}:${frame}`; }
function eventAnn(roiId, frame){ return annotations.events[eventKey(roiId, frame)] || migrateEventAnn({}); }
function suggestionAnn(id){ return annotations.suggestions[id] || {state:'', artifactClass:'', artifact_class:'', notes:''}; }
function setting(name){ return annotations.settings[name]; }
function setSetting(name, value){ annotations.settings[name] = value; queueSave(); }
function recordAction(kind){
  annotations.reviewStats = annotations.reviewStats || {sessionStartedAt: new Date().toISOString(), lastActionAt: null, actions: {}};
  annotations.reviewStats.actions = annotations.reviewStats.actions || {};
  annotations.reviewStats.actions[kind] = (annotations.reviewStats.actions[kind] || 0) + 1;
  annotations.reviewStats.lastActionAt = new Date().toISOString();
}
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
function cacheSetBounded(cache, key, value, limit=TRACE_CACHE_LIMIT){
  if(cache.size >= limit) {
    const firstKey = cache.keys().next().value;
    if(firstKey !== undefined) cache.delete(firstKey);
  }
  cache.set(key, value);
  return value;
}
function clearTraceCaches(reason='manual'){
  traceCache.clear();
  traceEventCache.clear();
  traceCacheStats.clears++;
  traceCacheStats.lastClearReason = reason;
}
function clearTraceEventCache(reason='event-threshold'){
  traceEventCache.clear();
  traceCacheStats.clears++;
  traceCacheStats.lastClearReason = reason;
}
function traceCacheKey(roi){
  return `${roi.id}|${roi.dffTrace?.length || 0}|${Number(kalmanGain()).toFixed(4)}|${Number(spikeGain()).toFixed(4)}`;
}
function modeledTraceCached(roi){
  const key = traceCacheKey(roi);
  if(traceCache.has(key)) {
    traceCacheStats.traceHits++;
    return traceCache.get(key);
  }
  traceCacheStats.traceMisses++;
  cacheSetBounded(traceCache, key, modeledTrace(roi));
  return traceCache.get(key);
}
function eventCacheKey(roi){
  return `${traceCacheKey(roi)}|${Number(threshold()).toFixed(4)}`;
}
function eventsForRoi(roi){
  if(!roi) return [];
  const key = eventCacheKey(roi);
  if(traceEventCache.has(key)) {
    traceCacheStats.eventHits++;
    return traceEventCache.get(key);
  }
  traceCacheStats.eventMisses++;
  const model = modeledTraceCached(roi);
  const zt = model.zTrace;
  const th = threshold();
  const out = [];
  for(let i=1;i<zt.length-1;i++){
    if(zt[i] >= th && zt[i] >= zt[i-1] && zt[i] >= zt[i+1]){
      out.push({frame:i+1, z:zt[i], amplitude:model.eventTrace[i]});
    }
  }
  return cacheSetBounded(traceEventCache, key, out);
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
    if (queue === 'needsAction') return Boolean(ann.needs_action);
    if (queue === 'controlReady') return ann.control_ready === 'yes' || ann.control_ready === 'maybe';
    if (queue === 'problemTrace') return ann.trace_quality === 'noisy' || ann.trace_quality === 'unusable';
    return true;
  });
  if (queue === 'highNoise') rows.sort((a,b) => b.noiseSigma - a.noiseSigma);
  else if (queue === 'highEvents') rows.sort((a,b) => eventsForRoi(b).length - eventsForRoi(a).length);
  else if (queue === 'uncertain') rows.sort((a,b) => roiUncertaintyScore(b) - roiUncertaintyScore(a));
  else rows.sort((a,b) => roiQualityScore(b) - roiQualityScore(a));
  return rows;
}

function selectedSuggestion(){
  const suggestions = data.discovery?.suggestions || [];
  return suggestions.find(s => s.id === selectedSuggestionId) || suggestions[0] || null;
}
function visibleSuggestions(){
  const queue = setting('discoveryQueue') || 'all';
  let rows = [...(data.discovery?.suggestions || [])];
  rows = rows.filter(s => {
    const ann = suggestionAnn(s.id);
    if (queue === 'unlabeled') return !ann.state;
    if (queue === 'promoted') return ann.state === 'promoted' || Boolean(annotations.promotedRois[s.id]);
    if (queue === 'missed') return ann.state === 'missed';
    if (queue === 'artifact') return ann.state === 'artifact';
    if (queue === 'artifactSuspects') return s.artifactCue && s.artifactCue !== 'none';
    return true;
  });
  rows.sort((a,b) => (b.discoveryScore || 0) - (a.discoveryScore || 0));
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
  document.getElementById('discoveryQueueSelect').value = setting('discoveryQueue') || 'all';
  document.getElementById('evidenceSelect').value = setting('evidenceMap') || '';
  document.getElementById('showEvidence').checked = Boolean(setting('showEvidence'));
  document.getElementById('showSuggestions').checked = Boolean(setting('showSuggestions'));
  applyDisplaySettings();
}

function populateEvidenceSelect(){
  const select = document.getElementById('evidenceSelect');
  if(!select) return;
  select.innerHTML = '';
  for(const m of data.discovery?.evidenceMaps || []){
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.label;
    select.appendChild(opt);
  }
}

function applyDisplaySettings() {
  img.style.width = `${data.video.width * Number(setting('zoom'))}px`;
  img.style.filter = `brightness(${setting('brightness')}) contrast(${setting('contrast')})`;
  evidenceImg.style.width = img.style.width;
  const evidenceMap = (data.discovery?.evidenceMaps || []).find(m => m.id === setting('evidenceMap'));
  evidenceImg.src = evidenceMap ? evidenceMap.file : '';
  evidenceImg.style.opacity = setting('showEvidence') ? '0.58' : '0';
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
  const showRois = document.getElementById('showRois').checked;
  const showLabels = document.getElementById('showLabels').checked;
  const showEvents = document.getElementById('showEvents').checked;
  const showSuggestions = document.getElementById('showSuggestions').checked;
  if(!showRois && !showSuggestions) return;
  const opacity = Number(setting('overlayOpacity'));
  if(showRois) for(const roi of visibleRois()){
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
  if(showSuggestions){
    for(const s of visibleSuggestions()){
      const ann = suggestionAnn(s.id);
      const isSel = s.id === selectedSuggestionId;
      let color = ann.state === 'promoted' || annotations.promotedRois[s.id] ? '#16a34a' :
        ann.state === 'artifact' ? '#dc2626' :
        ann.state === 'missed' ? '#facc15' :
        ann.state === 'unsure' ? '#9333ea' : '#fb7185';
      ctx.globalAlpha = isSel ? 0.96 : Math.max(0.38, opacity * 0.82);
      ctx.fillStyle = color;
      for(const p of s.points || []) ctx.fillRect(p[0], p[1], 1, 1);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = isSel ? '#ffffff' : color;
      ctx.lineWidth = isSel ? 2 : 1;
      const r = Math.max(5, Math.sqrt((s.area || 20) / Math.PI) + 3);
      ctx.beginPath(); ctx.arc(s.centroidX, s.centroidY, r, 0, Math.PI*2); ctx.stroke();
      if(showLabels){
        ctx.font = '10px Arial';
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#111827';
        ctx.lineWidth = 3;
        ctx.strokeText(String(s.id), s.centroidX + 5, s.centroidY - 5);
        ctx.fillText(String(s.id), s.centroidX + 5, s.centroidY - 5);
      }
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
  const model = modeledTraceCached(roi);
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

function roiCropBounds(roi, pad=18){
  if(!roi) return null;
  const bbox = roi.bbox || [
    Math.floor(roi.centroidX - 8), Math.floor(roi.centroidY - 8),
    Math.ceil(roi.centroidX + 8), Math.ceil(roi.centroidY + 8)
  ];
  const x0 = Math.max(0, Math.floor(bbox[0] - pad));
  const y0 = Math.max(0, Math.floor(bbox[1] - pad));
  const x1 = Math.min(data.video.width - 1, Math.ceil(bbox[2] + pad));
  const y1 = Math.min(data.video.height - 1, Math.ceil(bbox[3] + pad));
  return {x0, y0, x1, y1, w: Math.max(1, x1 - x0 + 1), h: Math.max(1, y1 - y0 + 1)};
}

function drawCrop(){
  if(!cropCanvas || !cropCtx) return;
  const roi = selectedRoi();
  cropCtx.clearRect(0,0,cropCanvas.width,cropCanvas.height);
  cropCtx.fillStyle = '#08111f';
  cropCtx.fillRect(0,0,cropCanvas.width,cropCanvas.height);
  if(!roi || !img.complete || !img.naturalWidth) return;
  const b = roiCropBounds(roi);
  const scale = Math.min(cropCanvas.width / b.w, cropCanvas.height / b.h);
  const dw = b.w * scale, dh = b.h * scale;
  const ox = (cropCanvas.width - dw) / 2, oy = (cropCanvas.height - dh) / 2;
  cropCtx.imageSmoothingEnabled = false;
  cropCtx.drawImage(img, b.x0, b.y0, b.w, b.h, ox, oy, dw, dh);
  cropCtx.fillStyle = 'rgba(56, 189, 248, 0.72)';
  for(const p of roi.points || []){
    const x = ox + (p[0] - b.x0) * scale;
    const y = oy + (p[1] - b.y0) * scale;
    cropCtx.fillRect(x, y, Math.max(1, scale), Math.max(1, scale));
  }
  cropCtx.strokeStyle = selectedEventFrame && eventNearFrame(roi, currentFrame) ? '#facc15' : '#ffffff';
  cropCtx.lineWidth = 2;
  cropCtx.beginPath();
  cropCtx.arc(ox + (roi.centroidX - b.x0) * scale, oy + (roi.centroidY - b.y0) * scale, Math.max(5, Math.sqrt(roi.area / Math.PI) * scale), 0, Math.PI * 2);
  cropCtx.stroke();
}

function renderRoiContext(){
  const roi = selectedRoi();
  const card = document.getElementById('roiEvidenceCard');
  const strip = document.getElementById('eventFilmstrip');
  drawCrop();
  if(!roi){
    if(card) card.innerHTML = '';
    if(strip) strip.innerHTML = '';
    return;
  }
  const events = eventsForRoi(roi);
  const diameterPx = 2 * Math.sqrt(roi.area / Math.PI);
  const pixelSize = Number(data.dataset?.pixel_size_microns);
  const diameterUm = Number.isFinite(pixelSize) ? `${(diameterPx * pixelSize).toFixed(1)} um` : 'n/a';
  if(card) card.innerHTML = `
    <table class="smallTable">
      <tr><th>Field</th><th>Value</th></tr>
      <tr><td>ROI</td><td>${roi.id}</td></tr>
      <tr><td>area</td><td>${roi.area} px</td></tr>
      <tr><td>equiv. diameter</td><td>${diameterPx.toFixed(1)} px / ${diameterUm}</td></tr>
      <tr><td>peak score</td><td>${Number(roi.peakScore).toFixed(2)}</td></tr>
      <tr><td>noise sigma</td><td>${Number(roi.noiseSigma).toFixed(5)}</td></tr>
      <tr><td>events</td><td>${events.length}</td></tr>
    </table>`;
  if(!strip) return;
  const center = selectedEventFrame || events[0]?.frame || currentFrame;
  const b = roiCropBounds(roi, 24);
  const thumb = 52;
  const scale = thumb / Math.max(b.w, b.h);
  strip.innerHTML = '';
  for(let frame = Math.max(1, center - 5); frame <= Math.min(data.video.frames, center + 10); frame++){
    const cell = document.createElement('div');
    cell.className = 'filmFrame' + (frame === currentFrame ? ' active' : '');
    cell.style.backgroundImage = `url("${framePath(frame)}")`;
    cell.style.backgroundSize = `${data.video.width * scale}px ${data.video.height * scale}px`;
    cell.style.backgroundPosition = `${-b.x0 * scale}px ${-b.y0 * scale}px`;
    cell.innerHTML = `<span>${frame}</span>`;
    cell.onclick = () => setFrame(frame);
    strip.appendChild(cell);
  }
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
  renderRoiContext();
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

function selectSuggestion(id){
  selectedSuggestionId = id;
  const s = selectedSuggestion();
  document.getElementById('suggestionNotes').value = s ? suggestionAnn(s.id).notes || '' : '';
  document.getElementById('artifactClass').value = s ? suggestionAnn(s.id).artifact_class || suggestionAnn(s.id).artifactClass || '' : '';
  if(s) {
    selectedEventFrame = null;
    currentFrame = Math.max(1, Math.min(data.video.frames, currentFrame));
    selectionText.textContent = `Suggestion ${s.id}`;
  }
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

function renderSuggestionList(){
  const root = document.getElementById('suggestionList');
  if(!root) return;
  root.innerHTML = '';
  const rows = visibleSuggestions();
  document.getElementById('suggestionVisibleCount').textContent = rows.length;
  for(const s of rows){
    const ann = suggestionAnn(s.id);
    const row = document.createElement('div');
    row.className = 'suggestionRow' + (s.id === selectedSuggestionId ? ' sel' : '');
    const state = annotations.promotedRois[s.id] ? 'promoted' : ann.state || 'new';
    const cue = s.artifactCue && s.artifactCue !== 'none' ? `, ${s.artifactCue}` : '';
    row.innerHTML = `<b>${s.id}</b><span>score ${Number(s.discoveryScore).toFixed(3)}, area ${s.area}${cue}</span><span class="badge ${ann.state || ''}">${state}</span>`;
    row.onclick = () => selectSuggestion(s.id);
    root.appendChild(row);
  }
}

function updateCounts(){
  const allEvents = data.rois.reduce((sum, r) => sum + eventsForRoi(r).length, 0);
  let acc = 0, rej = 0, unsure = 0, eventAccepted = 0;
  let promoted = 0, missed = 0, artifacts = 0;
  for(const r of data.rois){
    const st = roiAnn(r.id).state;
    if(st === 'accept') acc++;
    if(st === 'reject') rej++;
    if(st === 'unsure') unsure++;
    for(const ev of eventsForRoi(r)) if(eventAnn(r.id, ev.frame).state === 'accept') eventAccepted++;
  }
  for(const s of data.discovery?.suggestions || []){
    const ann = suggestionAnn(s.id);
    if(annotations.promotedRois[s.id] || ann.state === 'promoted') promoted++;
    if(ann.state === 'missed') missed++;
    if(ann.state === 'artifact') artifacts++;
  }
  document.getElementById('roiCount').textContent = data.rois.length;
  document.getElementById('eventCount').textContent = allEvents;
  document.getElementById('acceptedCount').textContent = acc;
  document.getElementById('rejectedCount').textContent = rej;
  document.getElementById('unsureCount').textContent = unsure;
  document.getElementById('eventAcceptedCount').textContent = eventAccepted;
  document.getElementById('suggestionCount').textContent = data.discovery?.suggestions?.length || 0;
  document.getElementById('promotedCount').textContent = promoted;
  document.getElementById('missedCount').textContent = missed;
  document.getElementById('artifactCount').textContent = artifacts;
}

function setRoiState(state){
  const roi = selectedRoi(); if(!roi) return;
  const cellState = state === 'accept' ? 'accepted' : state === 'reject' ? 'rejected' : state === 'unsure' ? 'unsure' : '';
  annotations.rois[roi.id] = Object.assign(roiAnn(roi.id), {state, cell_state: cellState});
  recordAction(`roi_${state || 'clear'}`);
  queueSave();
  renderAll();
}
function toggleDeleted(){
  const roi = selectedRoi(); if(!roi) return;
  const ann = Object.assign(roiAnn(roi.id), {deleted: !roiAnn(roi.id).deleted});
  annotations.rois[roi.id] = ann;
  recordAction(ann.deleted ? 'roi_hide' : 'roi_restore');
  queueSave();
  renderAll();
}
function setEventState(state){
  const roi = selectedRoi(); if(!roi || !selectedEventFrame) return;
  const eventState = state === 'accept' ? 'accepted' : state === 'reject' ? 'rejected' : state === 'unsure' ? 'unsure' : '';
  annotations.events[eventKey(roi.id, selectedEventFrame)] = Object.assign(eventAnn(roi.id, selectedEventFrame), {state, event_state: eventState});
  recordAction(`event_${state || 'clear'}`);
  queueSave();
  renderAll();
}

function setSuggestionState(state){
  const s = selectedSuggestion(); if(!s) return;
  annotations.suggestions[s.id] = Object.assign(suggestionAnn(s.id), {state});
  recordAction(`suggestion_${state || 'clear'}`);
  queueSave();
  renderAll();
}
function promoteSuggestion(){
  const s = selectedSuggestion(); if(!s) return;
  annotations.suggestions[s.id] = Object.assign(suggestionAnn(s.id), {state:'promoted'});
  annotations.promotedRois[s.id] = {
    sourceSuggestion: s.id,
    provenance: s.provenance || 'discovery',
    centroidX: s.centroidX,
    centroidY: s.centroidY,
    area: s.area,
    bbox: s.bbox,
    points: s.points || [],
    promotedAt: new Date().toISOString()
  };
  recordAction('suggestion_promote');
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
  for (const [id, field] of [['traceQuality','trace_quality'],['controlReady','control_ready'],['roiArtifactClass','artifact_class'],['needsAction','needs_action']]) {
    const el = document.getElementById(id);
    if(el) el.value = ann[field] || '';
  }
  const identity = document.getElementById('identityGroup');
  if(identity) identity.value = ann.identity_group || '';
  for (const [id, field] of [['eventType','event_type'],['timingQuality','timing_quality']]) {
    const el = document.getElementById(id);
    if(el) el.value = eann[field] || '';
  }
  const sann = selectedSuggestion() ? suggestionAnn(selectedSuggestion().id) : {};
  for (const [id, state] of [['suggestionMissedBtn','missed'],['suggestionArtifactBtn','artifact'],['suggestionUnsureBtn','unsure']]) {
    document.getElementById(id).classList.toggle('active', sann.state === state);
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
  renderSuggestionList();
  drawOverlay();
  drawTrace();
  renderRoiContext();
}

function exportRows(type) {
  const newline = String.fromCharCode(10);
  let rows = [];
  if (type === 'roi') {
    rows.push('roi_id\tstate\tcell_state\ttrace_quality\tcontrol_ready\tartifact_class\tidentity_group\tneeds_action\tdeleted\tnotes\tcentroid_x\tcentroid_y\tarea\tpeak_score\tevent_count\tnoise_sigma');
    for(const roi of data.rois){
      const ann = roiAnn(roi.id);
      const notes = (ann.notes || '').split(String.fromCharCode(9)).join(' ').split(newline).join(' ');
      rows.push([roi.id, ann.state || '', ann.cell_state || '', ann.trace_quality || '', ann.control_ready || '', ann.artifact_class || '', ann.identity_group || '', ann.needs_action || '', ann.deleted ? 1 : 0, notes, roi.centroidX, roi.centroidY, roi.area, roi.peakScore, eventsForRoi(roi).length, roi.noiseSigma].join('\t'));
    }
  } else if (type === 'event') {
    rows.push('roi_id\tframe\tstate\tevent_state\tevent_type\ttiming_quality\tnotes\tz\tamplitude\troi_state');
    for(const roi of data.rois){
      for(const ev of eventsForRoi(roi)){
        const ann = eventAnn(roi.id, ev.frame);
        const notes = (ann.notes || '').split(String.fromCharCode(9)).join(' ').split(newline).join(' ');
        rows.push([roi.id, ev.frame, ann.state || '', ann.event_state || '', ann.event_type || '', ann.timing_quality || '', notes, ev.z.toFixed(4), ev.amplitude.toFixed(6), roiAnn(roi.id).state || ''].join('\t'));
      }
    }
  } else {
    rows.push('suggestion_id\tstate\tartifact_class\tnotes\tpromoted\tcentroid_x\tcentroid_y\tarea\tdiscovery_score\tmax_z\tactive_frames\tartifact_cue\tprovenance');
    for(const s of data.discovery?.suggestions || []){
      const ann = suggestionAnn(s.id);
      const notes = (ann.notes || '').split(String.fromCharCode(9)).join(' ').split(newline).join(' ');
      rows.push([s.id, ann.state || '', ann.artifact_class || ann.artifactClass || '', notes, annotations.promotedRois[s.id] ? 1 : 0, s.centroidX, s.centroidY, s.area, s.discoveryScore, s.maxZ, s.activeFrames, s.artifactCue || '', s.provenance || ''].join('\t'));
    }
  }
  const blob = new Blob([rows.join(newline) + newline], {type:'text/tab-separated-values'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = type === 'roi' ? 'neuron_roi_annotations.tsv' : type === 'event' ? 'neuron_event_annotations.tsv' : 'neuron_discovery_suggestions.tsv';
  a.click();
  URL.revokeObjectURL(a.href);
}

function exportJson() {
  annotations.updatedAt = new Date().toISOString();
  const blob = new Blob([JSON.stringify(annotations, null, 2) + String.fromCharCode(10)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'annotations_v3.json';
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
  document.getElementById('eventWindowPrevBtn').onclick = () => nextEvent(-1);
  document.getElementById('eventWindowNextBtn').onclick = () => nextEvent(1);
  document.getElementById('archRunA').onchange = renderRunComparison;
  document.getElementById('archRunB').onchange = renderRunComparison;
  for(const id of ['showRois','showLabels','showEvents']) document.getElementById(id).onchange = drawOverlay;
  document.getElementById('showSuggestions').onchange = e => { setSetting('showSuggestions', e.target.checked); drawOverlay(); };
  document.getElementById('showEvidence').onchange = e => { setSetting('showEvidence', e.target.checked); applyDisplaySettings(); };
  document.getElementById('evidenceSelect').onchange = e => { setSetting('evidenceMap', e.target.value); applyDisplaySettings(); };
  for(const id of ['eventThreshold','kalmanGain','spikeGain','zoom','brightness','contrast','overlayOpacity','minArea','minEvents']) {
    document.getElementById(id).oninput = e => {
      const value = Number(e.target.value);
      setSetting(id, value);
      if(id === 'kalmanGain' || id === 'spikeGain') clearTraceCaches(id);
      if(id === 'eventThreshold') clearTraceEventCache(id);
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
  document.getElementById('suggestionPromoteBtn').onclick = promoteSuggestion;
  document.getElementById('suggestionMissedBtn').onclick = () => setSuggestionState('missed');
  document.getElementById('suggestionArtifactBtn').onclick = () => setSuggestionState('artifact');
  document.getElementById('suggestionUnsureBtn').onclick = () => setSuggestionState('unsure');
  document.getElementById('suggestionClearBtn').onclick = () => setSuggestionState('');
  document.getElementById('artifactClass').onchange = e => {
    const s = selectedSuggestion(); if(!s) return;
    annotations.suggestions[s.id] = Object.assign(suggestionAnn(s.id), {artifactClass:e.target.value, artifact_class:e.target.value});
    queueSave();
    renderAll();
  };
  document.getElementById('discoveryQueueSelect').onchange = e => { setSetting('discoveryQueue', e.target.value); renderAll(); };
  document.getElementById('exportRoiBtn').onclick = () => exportRows('roi');
  document.getElementById('exportEventBtn').onclick = () => exportRows('event');
  document.getElementById('exportSuggestionBtn').onclick = () => exportRows('suggestion');
  document.getElementById('exportJsonBtn').onclick = exportJson;
  for (const [id, field] of [['traceQuality','trace_quality'],['controlReady','control_ready'],['roiArtifactClass','artifact_class'],['needsAction','needs_action']]) {
    document.getElementById(id).onchange = e => {
      const roi = selectedRoi(); if(!roi) return;
      annotations.rois[roi.id] = Object.assign(roiAnn(roi.id), {[field]: e.target.value});
      recordAction(`roi_${field}`);
      queueSave();
      renderAll();
    };
  }
  document.getElementById('identityGroup').oninput = e => {
    const roi = selectedRoi(); if(!roi) return;
    annotations.rois[roi.id] = Object.assign(roiAnn(roi.id), {identity_group:e.target.value});
    recordAction('roi_identity_group');
    queueSave();
  };
  for (const [id, field] of [['eventType','event_type'],['timingQuality','timing_quality']]) {
    document.getElementById(id).onchange = e => {
      const roi = selectedRoi(); if(!roi || !selectedEventFrame) return;
      annotations.events[eventKey(roi.id, selectedEventFrame)] = Object.assign(eventAnn(roi.id, selectedEventFrame), {[field]: e.target.value});
      recordAction(`event_${field}`);
      queueSave();
      renderAll();
    };
  }
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
  document.getElementById('suggestionNotes').oninput = e => {
    const s = selectedSuggestion(); if(!s) return;
    annotations.suggestions[s.id] = Object.assign(suggestionAnn(s.id), {notes:e.target.value});
    queueSave();
  };
  overlay.onclick = e => {
    const rect = overlay.getBoundingClientRect();
    const x = (e.clientX - rect.left) * data.video.width / rect.width;
    const y = (e.clientY - rect.top) * data.video.height / rect.height;
    let best = null, bestD = Infinity, bestType = 'roi';
    for(const roi of visibleRois()){
      const dx = x - roi.centroidX, dy = y - roi.centroidY, d = dx*dx + dy*dy;
      if(d < bestD){ bestD = d; best = roi; bestType = 'roi'; }
    }
    if(document.getElementById('showSuggestions').checked) for(const s of visibleSuggestions()){
      const dx = x - s.centroidX, dy = y - s.centroidY, d = dx*dx + dy*dy;
      if(d < bestD){ bestD = d; best = s; bestType = 'suggestion'; }
    }
    if(bestType === 'suggestion') selectSuggestion(best.id);
    else if(best) selectRoi(best.id);
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
    else if(e.key === 'm') setSuggestionState('missed');
    else if(e.key === 'g') promoteSuggestion();
  });
  img.onload = () => { resizeOverlay(); drawCrop(); };
  window.onresize = resizeOverlay;
  window.addEventListener('hashchange', routePage);
}

function renderArchitectureLab(){
  const root = document.getElementById('architectureRuns');
  if(!root) return;
  const runs = data.architectureRuns?.runs || [];
  populateRunSelectors(runs);
  renderRunComparison();
  root.innerHTML = '';
  if(!runs.length){
    root.innerHTML = '<p class="hint">No architecture runs are attached yet. Use tools/build_architecture_run.py to create architecture_runs.json.</p>';
    return;
  }
  for(const run of runs){
    const card = document.createElement('div');
    card.className = 'archCard';
    const evidence = (run.artifacts?.evidence_maps || []).map(m => `<span>${m.label || m.id || 'map'}</span>`).join('');
    const ann = run.annotation_summary || {};
    card.innerHTML = `
      <h3>${run.label || run.run_id}</h3>
      <p class="hint">${run.run_id} | ${run.dataset_id}</p>
      <div class="archMeta">
        <div><b>${run.summary?.roi_count ?? 'n/a'}</b><span>ROIs</span></div>
        <div><b>${run.summary?.event_count ?? 'n/a'}</b><span>events</span></div>
        <div><b>${run.summary?.suggestion_count ?? 'n/a'}</b><span>suggestions</span></div>
        <div><b>${run.summary?.frame_count ?? data.video.frames}</b><span>frames</span></div>
        <div><b>${ann.roi_states?.accepted ?? 'n/a'}</b><span>accepted ROIs</span></div>
        <div><b>${ann.control_ready?.yes ?? 'n/a'}</b><span>control-ready</span></div>
      </div>
      <table class="smallTable"><tr><th>Artifact</th><th>Path</th></tr>
        <tr><td>review data</td><td>${run.artifacts?.review_data || ''}</td></tr>
        <tr><td>ROI summary</td><td>${run.artifacts?.roi_summary_tsv || ''}</td></tr>
      </table>
      <div class="archEvidence">${evidence}</div>`;
    root.appendChild(card);
  }
}

function runLabel(run){
  return run ? `${run.label || run.run_id}` : '';
}

function runMetric(run, path, fallback=0){
  let cur = run;
  for(const part of path.split('.')){
    if(cur === undefined || cur === null) return fallback;
    cur = cur[part];
  }
  return cur === undefined || cur === null ? fallback : cur;
}

function populateRunSelectors(runs){
  for(const [id, defaultIndex] of [['archRunA', 0], ['archRunB', Math.min(1, Math.max(0, runs.length - 1))]]){
    const select = document.getElementById(id);
    if(!select) continue;
    const previous = select.value;
    select.innerHTML = '';
    for(const run of runs){
      const opt = document.createElement('option');
      opt.value = run.run_id;
      opt.textContent = runLabel(run);
      select.appendChild(opt);
    }
    if(runs.some(r => r.run_id === previous)) select.value = previous;
    else if(runs[defaultIndex]) select.value = runs[defaultIndex].run_id;
  }
}

function renderRunComparison(){
  const root = document.getElementById('runComparison');
  if(!root) return;
  const runs = data.architectureRuns?.runs || [];
  if(runs.length < 2){
    root.innerHTML = '<p class="hint">Add a second architecture run manifest to compare methods side-by-side. The current run is still shown below.</p>';
    return;
  }
  const a = runs.find(r => r.run_id === document.getElementById('archRunA').value) || runs[0];
  const b = runs.find(r => r.run_id === document.getElementById('archRunB').value) || runs[1];
  const rows = [
    ['Candidate ROIs', 'summary.roi_count', false],
    ['Candidate events', 'summary.event_count', false],
    ['Discovery suggestions', 'summary.suggestion_count', false],
    ['Accepted ROIs', 'annotation_summary.roi_states.accepted', true],
    ['Rejected ROIs', 'annotation_summary.roi_states.rejected', false],
    ['Control-ready ROIs', 'annotation_summary.control_ready.yes', true],
    ['Accepted events', 'annotation_summary.event_states.accepted', true],
    ['Candidates per accepted ROI', 'annotation_summary.review_burden.candidate_rois_per_accepted_roi', false],
    ['Events per accepted event', 'annotation_summary.review_burden.candidate_events_per_accepted_event', false]
  ];
  let html = `<table class="smallTable compareTable"><tr><th>Metric</th><th>${runLabel(a)}</th><th>${runLabel(b)}</th><th>Delta B-A</th></tr>`;
  for(const [label, path, higherGood] of rows){
    const av = Number(runMetric(a, path, 0));
    const bv = Number(runMetric(b, path, 0));
    const delta = bv - av;
    const cls = Math.abs(delta) < 1e-9 ? 'deltaNeutral' : (higherGood ? delta > 0 : delta < 0) ? 'deltaGood' : 'deltaBad';
    html += `<tr><td>${label}</td><td>${fmt(av, Number.isInteger(av) ? 0 : 2)}</td><td>${fmt(bv, Number.isInteger(bv) ? 0 : 2)}</td><td class="${cls}">${delta >= 0 ? '+' : ''}${fmt(delta, Number.isInteger(delta) ? 0 : 2)}</td></tr>`;
  }
  root.innerHTML = html + '</table>';
}

function annotationSummary(){
  const roiStates = {accepted:0, rejected:0, unsure:0, unlabeled:0};
  const eventStates = {accepted:0, rejected:0, unsure:0, unlabeled:0};
  const suggestionStates = {promoted:0, missed:0, artifact:0, unsure:0, unlabeled:0};
  const traceQuality = {good:0, weak:0, noisy:0, unusable:0, unlabeled:0};
  const controlReady = {yes:0, maybe:0, no:0, unlabeled:0};
  for(const roi of data.rois){
    const ann = roiAnn(roi.id);
    const rs = ann.cell_state || (ann.state === 'accept' ? 'accepted' : ann.state === 'reject' ? 'rejected' : ann.state === 'unsure' ? 'unsure' : 'unlabeled');
    roiStates[roiStates[rs] === undefined ? 'unlabeled' : rs]++;
    const tq = ann.trace_quality || 'unlabeled';
    traceQuality[traceQuality[tq] === undefined ? 'unlabeled' : tq]++;
    const cr = ann.control_ready || 'unlabeled';
    controlReady[controlReady[cr] === undefined ? 'unlabeled' : cr]++;
    for(const ev of eventsForRoi(roi)){
      const eann = eventAnn(roi.id, ev.frame);
      const es = eann.event_state || (eann.state === 'accept' ? 'accepted' : eann.state === 'reject' ? 'rejected' : eann.state === 'unsure' ? 'unsure' : 'unlabeled');
      eventStates[eventStates[es] === undefined ? 'unlabeled' : es]++;
    }
  }
  for(const s of data.discovery?.suggestions || []){
    const ann = suggestionAnn(s.id);
    const ss = annotations.promotedRois[s.id] ? 'promoted' : ann.state || 'unlabeled';
    suggestionStates[suggestionStates[ss] === undefined ? 'unlabeled' : ss]++;
  }
  const eventCount = Object.values(eventStates).reduce((a,b) => a+b, 0);
  return {
    roi_count: data.rois.length,
    event_count: eventCount,
    suggestion_count: data.discovery?.suggestions?.length || 0,
    roi_states: roiStates,
    event_states: eventStates,
    suggestion_states: suggestionStates,
    trace_quality: traceQuality,
    control_ready: controlReady,
    review_burden: {
      candidate_rois_per_accepted_roi: data.rois.length / Math.max(1, roiStates.accepted),
      candidate_events_per_accepted_event: eventCount / Math.max(1, eventStates.accepted)
    }
  };
}

function auditRows(title, counts){
  const total = Object.values(counts).reduce((a,b) => a+b, 0) || 1;
  let html = `<h2>${title}</h2><div class="auditBars">`;
  for(const [name, count] of Object.entries(counts)){
    const pct = Math.round(100 * count / total);
    html += `<div class="auditRow"><span>${name}</span><div class="auditBar"><div class="auditFill" style="width:${pct}%"></div></div><b>${count}</b></div>`;
  }
  return html + '</div>';
}

function renderMetricsAudit(){
  const root = document.getElementById('metricsAudit');
  if(!root) return;
  const s = annotationSummary();
  const actionCount = Object.values(annotations.reviewStats?.actions || {}).reduce((a,b) => a + b, 0);
  root.innerHTML = `
    <div class="metricGrid">
      <div class="metric"><b>${s.roi_count}</b><span>candidate ROIs</span></div>
      <div class="metric"><b>${s.roi_states.accepted}</b><span>accepted ROIs</span></div>
      <div class="metric"><b>${s.event_count}</b><span>candidate events</span></div>
      <div class="metric"><b>${s.event_states.accepted}</b><span>accepted events</span></div>
      <div class="metric"><b>${s.suggestion_count}</b><span>discovery suggestions</span></div>
      <div class="metric"><b>${s.suggestion_states.promoted}</b><span>promoted suggestions</span></div>
      <div class="metric"><b>${s.review_burden.candidate_rois_per_accepted_roi.toFixed(1)}</b><span>ROIs per accepted ROI</span></div>
      <div class="metric"><b>${s.review_burden.candidate_events_per_accepted_event.toFixed(1)}</b><span>events per accepted event</span></div>
      <div class="metric"><b>${actionCount}</b><span>review actions</span></div>
      <div class="metric"><b>${annotations.reviewStats?.lastActionAt ? 'yes' : 'no'}</b><span>active session</span></div>
    </div>
    <div class="auditSplit">
      <div class="archCard">${auditRows('ROI states', s.roi_states)}</div>
      <div class="archCard">${auditRows('Event states', s.event_states)}</div>
      <div class="archCard">${auditRows('Trace quality', s.trace_quality)}</div>
      <div class="archCard">${auditRows('Control readiness', s.control_ready)}</div>
      <div class="archCard">${auditRows('Discovery suggestions', s.suggestion_states)}</div>
    </div>`;
}

function quantile(values, q){
  const arr = values.filter(v => Number.isFinite(v)).sort((a,b) => a-b);
  if(!arr.length) return null;
  const idx = Math.max(0, Math.min(arr.length - 1, Math.round((arr.length - 1) * q)));
  return arr[idx];
}

function fmt(v, digits=2){
  return v === null || v === undefined || Number.isNaN(v) ? 'n/a' : Number(v).toFixed(digits);
}

function renderDatasetQc(){
  const root = document.getElementById('datasetQc');
  if(!root) return;
  const areas = data.rois.map(r => Number(r.area));
  const diamPx = areas.map(a => 2 * Math.sqrt(a / Math.PI));
  const pixelSize = Number(data.dataset?.pixel_size_microns);
  const diamUm = Number.isFinite(pixelSize) ? diamPx.map(v => v * pixelSize) : [];
  const noise = data.rois.map(r => Number(r.noiseSigma));
  const eventCounts = data.rois.map(r => eventsForRoi(r).length);
  const peakScores = data.rois.map(r => Number(r.peakScore));
  const suggestions = data.discovery?.suggestions || [];
  const artifactCueCount = suggestions.filter(s => s.artifactCue && s.artifactCue !== 'none').length;
  const warnings = [];
  if(data.rois.length && quantile(diamPx, 0.5) < 8) warnings.push('Median ROI footprint is small in pixels; the detector may be capturing active cores or fragments rather than full somata.');
  if(Number.isFinite(pixelSize) && quantile(diamUm, 0.5) < 5) warnings.push('Median equivalent ROI diameter is below 5 microns with the configured pixel size.');
  if(suggestions.length > data.rois.length) warnings.push('Discovery suggestions outnumber current ROIs; review missed-neuron coverage before tightening thresholds.');
  if(artifactCueCount > suggestions.length * 0.25) warnings.push('Many discovery suggestions have artifact cues; inspect evidence maps for vessels, borders, or bright static structures.');
  if(!Number.isFinite(pixelSize)) warnings.push('Pixel size is not set in the dataset manifest, so physical-size QC is disabled.');
  const qcWarnings = warnings.map(w => `<div class="qcWarning">${w}</div>`).join('') || '<div class="qcWarning">No QC warnings from the current lightweight checks.</div>';
  const maps = (data.discovery?.evidenceMaps || []).map(m => `
    <div class="qcMap">
      <img src="${m.file}" alt="${m.label}">
      <p class="hint">${m.label}</p>
    </div>`).join('');
  root.innerHTML = `
    <div class="metricGrid">
      <div class="metric"><b>${data.video.width} x ${data.video.height}</b><span>frame size</span></div>
      <div class="metric"><b>${data.video.frames}</b><span>frames</span></div>
      <div class="metric"><b>${data.rois.length}</b><span>candidate ROIs</span></div>
      <div class="metric"><b>${suggestions.length}</b><span>discovery suggestions</span></div>
      <div class="metric"><b>${fmt(quantile(areas, 0.5), 0)}</b><span>median ROI area px</span></div>
      <div class="metric"><b>${fmt(quantile(diamPx, 0.5), 1)}</b><span>median ROI diameter px</span></div>
      <div class="metric"><b>${diamUm.length ? fmt(quantile(diamUm, 0.5), 1) : 'n/a'}</b><span>median ROI diameter microns</span></div>
      <div class="metric"><b>${fmt(quantile(noise, 0.5), 4)}</b><span>median trace noise sigma</span></div>
      <div class="metric"><b>${fmt(quantile(eventCounts, 0.5), 0)}</b><span>median events per ROI</span></div>
      <div class="metric"><b>${fmt(quantile(peakScores, 0.5), 2)}</b><span>median peak score</span></div>
    </div>
    <div class="qcWarnings">${qcWarnings}</div>
    <div class="auditSplit">
      <div class="archCard">${auditRows('ROI area px', {min: Math.min(...areas), median: quantile(areas, 0.5), max: Math.max(...areas)})}</div>
      <div class="archCard">${auditRows('Events per ROI', {zero: eventCounts.filter(v => v === 0).length, one_to_three: eventCounts.filter(v => v >= 1 && v <= 3).length, four_plus: eventCounts.filter(v => v >= 4).length})}</div>
      <div class="archCard">${auditRows('Discovery artifact cues', {with_cue: artifactCueCount, no_cue: Math.max(0, suggestions.length - artifactCueCount)})}</div>
    </div>
    <h2>Evidence Maps</h2>
    <div class="qcMapGrid">${maps}</div>`;
}

function routePage(){
  const page = location.hash === '#architecture' ? 'architecture' : location.hash === '#metrics' ? 'metrics' : (location.hash === '#process' || location.hash === '#qc') ? 'qc' : 'review';
  for(const id of ['reviewTab','reviewTabArch','reviewTabMetrics','reviewTabQc']) document.getElementById(id)?.classList.toggle('active', page === 'review');
  for(const id of ['architectureTab','architectureTabArch','architectureTabMetrics','architectureTabQc']) document.getElementById(id)?.classList.toggle('active', page === 'architecture');
  for(const id of ['metricsTab','metricsTabArch','metricsTabMetrics','metricsTabQc']) document.getElementById(id)?.classList.toggle('active', page === 'metrics');
  for(const id of ['qcTab','qcTabArch','qcTabMetrics','qcTabQc']) document.getElementById(id)?.classList.toggle('active', page === 'qc');
  document.getElementById('architecturePage').classList.toggle('hidden', page !== 'architecture');
  document.getElementById('metricsPage').classList.toggle('hidden', page !== 'metrics');
  document.getElementById('qcPage').classList.toggle('hidden', page !== 'qc');
  appRoot.classList.toggle('arch-mode', page === 'architecture');
  appRoot.classList.toggle('lab-mode', page === 'metrics');
  appRoot.classList.toggle('qc-mode', page === 'qc');
  if(page === 'architecture') renderArchitectureLab();
  else if(page === 'metrics') renderMetricsAudit();
  else if(page === 'qc') renderDatasetQc();
  else resizeOverlay();
}

async function boot(){
  populateEvidenceSelect();
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
  if(selectedSuggestionId) {
    document.getElementById('suggestionNotes').value = suggestionAnn(selectedSuggestionId).notes || '';
    document.getElementById('artifactClass').value = suggestionAnn(selectedSuggestionId).artifact_class || suggestionAnn(selectedSuggestionId).artifactClass || '';
  }
  setFrame(1);
  routePage();
  renderAll();
}
boot();
"""


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Neuron Annotation Workbench - {dataset_id}</title>
<link rel="stylesheet" href="workbench.css">
</head>
<body>
<div class="app booting" id="appRoot">
  <main class="stage reviewOnly">
    <div class="topbar">
      <h1>Neuron Annotation Workbench: {dataset_id}</h1>
      <nav class="navTabs">
        <a id="architectureTab" href="#architecture">Architecture Lab</a>
        <a id="experimentsTab" href="#experiments">Experiment Lab</a>
        <a id="qcTab" href="#process">Process Lab</a>
        <a id="reviewTab" href="#review">Review</a>
        <a id="metricsTab" href="#metrics">Metrics</a>
        <a id="reportTab" href="#report">Report</a>
      </nav>
      <label class="modeToggle">Mode
        <select id="uiMode">
          <option value="basic">Basic</option>
          <option value="advanced">Advanced</option>
        </select>
      </label>
      <label class="modeToggle">Theme
        <select class="themeSelect" aria-label="Theme">
          <option value="system">System</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </label>
      <label class="reviewerField">Reviewer <input id="reviewerIdInput" type="text" placeholder="initials"></label>
      <button id="nextMissingReviewerBtn" class="advancedOnly" type="button">Next Missing Reviewer</button>
      <button id="stampSelectedReviewerBtn" class="advancedOnly" type="button">Stamp Selected</button>
      <button id="stampMissingReviewerBtn" class="advancedOnly" type="button">Stamp Missing</button>
      <span id="saveState" class="saveState">loading</span>
    </div>
    <div class="toolbar runSyncBar advancedOnly">
      <label>Review run <select id="activeRunSelect"></select></label>
      <span id="activeRunStatus" class="runSyncStatus">loading run status</span>
      <label>Backend
        <select id="generationBackend">
          <option value="auto">Auto</option>
          <option value="fiji_groovy">Fiji/Groovy</option>
          <option value="python_gpu">Python GPU</option>
        </select>
      </label>
      <button id="loadRunReviewBtn">Load Review</button>
      <button id="openRunViewBtn">Open View</button>
      <button id="previewRunViewBtn">Generate Preview</button>
      <button id="generateRunViewBtn">Generate View</button>
      <button id="unlockGenerationBtn">Unlock Generation</button>
      <button id="refreshRunBtn">Refresh</button>
    </div>
    <div id="runGeneratePanel" class="runGeneratePanel hidden advancedOnly"></div>
    <div class="toolbar primaryReviewToolbar">
      <button id="playBtn">Play</button>
      <label class="frameControl">Frame <input id="frameSlider" type="range" min="1" max="{frames}" value="1"></label>
      <b id="frameLabel">1</b>
      <button id="fitBtn">Fit Width</button>
      <button id="fullscreenBtn">Fullscreen</button>
      <button id="exportScreenshotBtn">Screenshot</button>
    </div>
    <details class="toolbarDisclosure">
      <summary>Review tools</summary>
      <div class="toolbar compactToolGroup">
        <button id="prevActiveFrameBtn">Prev Active</button>
        <button id="nextActiveFrameBtn">Next Active</button>
      <label>Workflow
        <select id="reviewWorkflowPreset">
          <option value="custom">Custom</option>
          <option value="fast_triage">Fast triage</option>
          <option value="event_validation">Event validation</option>
          <option value="missed_neuron_search">Missed neuron search</option>
          <option value="artifact_cleanup">Artifact cleanup</option>
          <option value="mask_editing">Mask editing</option>
        </select>
      </label>
      <button id="shortcutHelpBtn" type="button">Shortcuts</button>
      <label>Jump <input id="quickJumpInput" type="search" placeholder="ROI 12 or f120"></label>
      <button id="quickJumpBtn" type="button">Go</button>
      <button id="undoAnnotationBtn" type="button">Undo Label</button>
      <button id="bookmarkAddBtn" type="button">Bookmark</button>
      <select id="bookmarkSelect" aria-label="Review bookmarks"></select>
      <button id="bookmarkGoBtn" type="button">Open Mark</button>
      <button id="bookmarkDeleteBtn" type="button">Delete Mark</button>
      </div>
    </details>
    <details class="toolbarDisclosure advancedOnly">
      <summary>Display controls</summary>
      <div class="toolbar compactToolGroup">
        <label>Zoom <input id="zoom" type="range" min="0.75" max="5" step="0.05"> <span id="zoomLabel">3.00</span></label>
        <label>Brightness <input id="brightness" type="range" min="0.4" max="2.5" step="0.02"> <span id="brightnessLabel">1.00</span></label>
        <label>Contrast <input id="contrast" type="range" min="0.5" max="3" step="0.02"> <span id="contrastLabel">1.08</span></label>
      </div>
    </details>
    <div class="toolbar">
      <label><input id="showRois" type="checkbox" checked> ROIs</label>
      <label><input id="showLabels" type="checkbox" checked> IDs</label>
      <label><input id="showEvents" type="checkbox" checked> event frames</label>
      <label><input id="showSuggestions" type="checkbox" checked> suggestions</label>
      <label><input id="showEvidence" type="checkbox"> evidence map</label>
      <select id="evidenceSelect"></select>
      <label class="advancedOnly">Overlay <input id="overlayOpacity" type="range" min="0.1" max="1" step="0.02"> <span id="overlayOpacityLabel">0.72</span></label>
      <label class="advancedOnly">Preset
        <select id="overlayPresetSelect">
          <option value="validate">Validate firing</option>
          <option value="dense">Dense triage</option>
          <option value="discovery">Discovery</option>
          <option value="custom">Custom</option>
        </select>
      </label>
      <label class="advancedOnly">Selected
        <select id="selectedOverlayMode">
          <option value="outline">Outline</option>
          <option value="soft">Soft fill</option>
          <option value="event">Event-aware</option>
        </select>
      </label>
      <label class="advancedOnly">Selected fill <input id="selectedFillOpacity" type="range" min="0" max="0.85" step="0.01"> <span id="selectedFillOpacityLabel">0.10</span></label>
      <label class="advancedOnly">Outline <input id="selectedOutlineWidth" type="range" min="1" max="5" step="0.5"> <span id="selectedOutlineWidthLabel">2.5</span></label>
      <label class="advancedOnly">Focus
        <select id="roiFocusMode">
          <option value="all">All ROIs</option>
          <option value="solo">Selected only</option>
          <option value="neighbors">Nearby</option>
        </select>
      </label>
      <label class="advancedOnly">Radius <input id="neighborRadiusPx" type="range" min="8" max="120" step="2"> <span id="neighborRadiusPxLabel">36</span></label>
      <span id="focusSummary" class="hint"></span>
      <label class="advancedOnly">event z <input id="eventThreshold" type="range" min="1.2" max="5" step="0.1"> <span id="eventThresholdLabel">2.4</span></label>
      <label class="advancedOnly">Kalman gain <input id="kalmanGain" type="range" min="0.01" max="0.18" step="0.005"> <span id="kalmanGainLabel">0.060</span></label>
      <label class="advancedOnly">spike gain <input id="spikeGain" type="range" min="0" max="0.05" step="0.002"> <span id="spikeGainLabel">0.008</span></label>
      <label class="advancedOnly">Manual ROI
        <select id="manualRoiMode">
          <option value="select">Select</option>
          <option value="center">Center</option>
          <option value="circle">Circle</option>
          <option value="lasso">Lasso</option>
        </select>
      </label>
      <label class="advancedOnly">Manual radius <input id="manualRoiRadius" type="range" min="2" max="30" step="1"> <span id="manualRoiRadiusLabel">6</span></label>
      <button id="manualRoiCancelBtn" class="advancedOnly" type="button">Cancel Manual</button>
      <label class="advancedOnly">Edit ROI
        <select id="roiEditMode">
          <option value="off">Off</option>
          <option value="brush_add">Brush add</option>
          <option value="brush_erase">Brush erase</option>
        </select>
      </label>
      <label class="advancedOnly">Brush <input id="roiEditBrushRadius" type="range" min="1" max="18" step="1"> <span id="roiEditBrushRadiusLabel">4</span></label>
      <button id="roiEditDoneBtn" class="advancedOnly" type="button">Finish Edit</button>
      <button id="roiEditUndoBtn" class="advancedOnly" type="button">Undo Mask</button>
      <button id="roiEditRevertBtn" class="advancedOnly" type="button">Revert To Source</button>
      <button id="materializeManualTracesBtn" class="advancedOnly" type="button">Materialize Traces</button>
    </div>
    <div class="viewerScroll" id="viewerScroll">
      <div class="viewerWrap" id="viewerWrap">
        <img id="frameImg" alt="video frame">
        <img id="evidenceImg" alt="evidence map">
        <canvas id="overlay"></canvas>
      </div>
    </div>
    <div class="status">
      <span id="statusText" aria-live="polite"></span>
      <span id="selectionText" aria-live="polite"></span>
    </div>
    <div id="shortcutOverlay" class="shortcutOverlay hidden" role="dialog" aria-modal="true" aria-labelledby="shortcutOverlayTitle">
      <div class="shortcutPanel">
        <div class="runCardHeader">
          <h2 id="shortcutOverlayTitle">Keyboard Shortcuts</h2>
          <button id="shortcutCloseBtn" type="button">Close</button>
        </div>
        <div class="shortcutGrid">
          <div><b>Space</b><span>Play / pause</span></div>
          <div><b>Left / Right</b><span>Previous / next frame</span></div>
          <div><b>j / k</b><span>Next / previous ROI</span></div>
          <div><b>n / p</b><span>Next / previous event</span></div>
          <div><b>N / P</b><span>Next / previous event queue item</span></div>
          <div><b>. / ,</b><span>Next / previous suggestion</span></div>
          <div><b>v / V</b><span>Next / previous active frame</span></div>
          <div><b>a / r / u</b><span>Accept / reject / unsure ROI</span></div>
          <div><b>e / x</b><span>Accept / reject event</span></div>
          <div><b>g / G</b><span>Promote suggestion / promote and next</span></div>
          <div><b>m / M</b><span>Mark missed / mark missed and next</span></div>
          <div><b>Ctrl/Cmd Z</b><span>Undo last label</span></div>
          <div><b>0</b><span>Reset trace zoom</span></div>
          <div><b>?</b><span>Show / hide this panel</span></div>
        </div>
      </div>
    </div>
    <div class="reviewEvidenceGrid">
      <div class="traceBox">
        <div class="traceControls">
          <span id="traceWindowText">frames 1-{frames}</span>
          <button id="traceFullBtn" type="button">Full</button>
          <button id="traceEvent2sBtn" type="button">±2s</button>
          <button id="traceEvent5sBtn" type="button">±5s</button>
          <button id="traceResetZoomBtn" type="button">Reset zoom</button>
        </div>
        <canvas id="traceCanvas" width="1000" height="260" role="img" aria-label="Selected ROI trace"></canvas>
        <canvas id="eventTimelineCanvas" width="1000" height="72" role="img" aria-label="Visible event density timeline"></canvas>
        <div class="legend">
          <span><i class="dot" style="background:#2563eb"></i>dF/F</span>
          <span><i class="dot" style="background:#64748b"></i>Kalman baseline</span>
          <span><i class="dot" style="background:#f59e0b"></i>event z</span>
          <span><i class="dot" style="background:#facc15"></i>called event</span>
        </div>
      </div>
      <section class="contextPanel selectedRoiPanel" aria-labelledby="selectedRoiContextHeading">
        <div class="topbar">
          <h2 id="selectedRoiContextHeading">Selected ROI Context</h2>
          <div class="buttonRow">
            <button id="eventWindowPrevBtn">Prev Event</button>
            <button id="eventWindowNextBtn">Next Event</button>
          </div>
        </div>
        <canvas id="roiCropCanvas" width="260" height="260" role="img" aria-label="Selected ROI crop"></canvas>
        <div id="roiEvidenceCard"></div>
        <div class="filmstrip" id="eventFilmstrip" aria-label="Event-centered frames"></div>
      </section>
    </div>
  </main>
  <aside class="side reviewOnly">
    <div class="metricGrid summaryGrid">
      <div class="metric"><b id="roiCount"></b><span>candidate ROIs</span></div>
      <div class="metric"><b id="visibleCount"></b><span>visible in queue</span></div>
      <div class="metric"><b id="eventAcceptedCount"></b><span>accepted events</span></div>
      <div class="metric"><b id="acceptedCount"></b><span>accepted ROIs</span></div>
    </div>
    <details>
      <summary>Dataset Summary</summary>
      <div class="metricGrid">
      <div class="metric"><b id="suggestionCount"></b><span>missed suggestions</span></div>
      <div class="metric"><b id="suggestionVisibleCount"></b><span>visible suggestions</span></div>
      <div class="metric"><b id="eventCount"></b><span>events at threshold</span></div>
      <div class="metric"><b id="rejectedCount"></b><span>rejected ROIs</span></div>
      <div class="metric"><b id="unsureCount"></b><span>unsure ROIs</span></div>
      <div class="metric"><b id="promotedCount"></b><span>promoted missed</span></div>
      <div class="metric"><b id="missedCount"></b><span>marked missed</span></div>
      <div class="metric"><b id="artifactCount"></b><span>artifact suggestions</span></div>
      <div class="metric"><b>{frames}</b><span>frames</span></div>
      </div>
    </details>
    <h2>ROI Review</h2>
    <section class="reviewSessionPanel" id="reviewSessionPanel" aria-label="Review session checklist"></section>
    <section class="guidedPanelShell">
      <div class="topbar">
        <h2>Guided Review</h2>
        <button id="reviewModeToggle">Explore / Guided</button>
      </div>
      <div id="guidedPanel"></div>
    </section>
    <div class="buttonRow">
      <button id="acceptBtn" class="accept">Accept</button>
      <button id="rejectBtn" class="reject">Reject</button>
      <button id="unsureBtn" class="unsure">Unsure</button>
      <button id="acceptNextBtn" class="accept">Accept + Next</button>
      <button id="rejectNextBtn" class="reject">Reject + Next</button>
      <button id="unsureNextBtn" class="unsure">Unsure + Next</button>
      <button id="strongNeuronNextBtn" class="accept">Strong Neuron + Next</button>
      <button id="artifactRoiNextBtn" class="reject">Artifact ROI + Next</button>
      <button id="clearBtn">Clear</button>
      <button id="deleteBtn">Hide ROI</button>
    </div>
    <div class="toolbar">
      <button id="nextActiveRoiBtn">Next Active ROI</button>
      <button id="nextUncertainRoiBtn">Next Uncertain</button>
      <button id="nextArtifactRiskBtn">Next Artifact Risk</button>
      <button id="missedNeuronModeBtn">Missed Mode</button>
    </div>
    <textarea id="roiNotes" rows="3" placeholder="Notes for selected ROI"></textarea>
    <div class="toolbar advancedOnly">
      <select id="traceQuality">
        <option value="">Trace quality</option>
        <option value="good">Good trace</option>
        <option value="weak">Weak trace</option>
        <option value="noisy">Noisy trace</option>
        <option value="unusable">Unusable trace</option>
      </select>
      <select id="controlReady">
        <option value="">Control ready?</option>
        <option value="yes">Control ready</option>
        <option value="maybe">Maybe</option>
        <option value="no">Not ready</option>
      </select>
      <select id="roiArtifactClass">
        <option value="">ROI artifact class</option>
        <option value="none">None</option>
        <option value="motion">Motion</option>
        <option value="vessel_static_structure">Vessel/static</option>
        <option value="motion_artifact">Motion artifact</option>
        <option value="background_fluctuation">Background fluctuation</option>
        <option value="impulse_noise">Impulse noise</option>
        <option value="split_needed">Split needed</option>
        <option value="merge_needed">Merge needed</option>
        <option value="redraw_needed">Redraw needed</option>
        <option value="low_snr_uncertain">Low-SNR uncertain</option>
      </select>
      <select id="needsAction">
        <option value="">Needs action?</option>
        <option value="review_trace">Review trace</option>
        <option value="split_needed">Split needed</option>
        <option value="merge_needed">Merge needed</option>
        <option value="redraw_needed">Redraw needed</option>
      </select>
      <input id="identityGroup" placeholder="Identity group" style="height:31px;border:1px solid var(--line);border-radius:6px;padding:0 8px;max-width:130px">
      <select id="roiConfidence">
        <option value="">Confidence</option>
        <option value="high">High</option>
        <option value="medium">Medium</option>
        <option value="low">Low</option>
      </select>
      <input id="roiReasonTags" placeholder="Reason tags" style="height:31px;border:1px solid var(--line);border-radius:6px;padding:0 8px;max-width:150px">
    </div>
    <div class="toolbar advancedOnly">
      <span class="hint"><b id="multiSelectCount">1</b> selected</span>
      <button id="bulkAcceptBtn">Accept Selected</button>
      <button id="bulkRejectBtn">Reject Selected</button>
      <button id="bulkUnsureBtn">Unsure Selected</button>
      <input id="bulkIdentityGroup" placeholder="Group selected" style="height:31px;border:1px solid var(--line);border-radius:6px;padding:0 8px;max-width:135px">
      <button id="bulkIdentityBtn">Set Group</button>
      <select id="bulkNeedsAction">
        <option value="">Bulk action</option>
        <option value="merge_needed">Merge needed</option>
        <option value="split_needed">Split needed</option>
        <option value="redraw_needed">Redraw needed</option>
        <option value="review_trace">Review trace</option>
      </select>
      <button id="bulkNeedsActionBtn">Apply</button>
      <button id="virtualMergeBtn">Virtual Merge</button>
      <button id="visualSplitBtn">Visual Split</button>
      <button id="clearMultiSelectBtn">Clear Multi</button>
    </div>
    <h2>Event Review</h2>
    <div class="buttonRow">
      <button id="eventAcceptBtn" class="accept">Accept Event</button>
      <button id="eventRejectBtn" class="reject">Reject Event</button>
      <button id="eventUnsureBtn" class="unsure">Unsure</button>
      <button id="eventAcceptNextBtn" class="accept">Accept + Next</button>
      <button id="eventRejectNextBtn" class="reject">Reject + Next</button>
      <button id="eventUnsureNextBtn" class="unsure">Unsure + Next</button>
      <button id="eventArtifactNextBtn" class="reject">Artifact + Next</button>
      <button id="eventClearBtn">Clear</button>
    </div>
    <div class="toolbar">
      <select id="eventQueueSelect">
        <option value="all">All events</option>
        <option value="unlabeled">Unlabeled events</option>
        <option value="accepted">Accepted events</option>
        <option value="rejected">Rejected events</option>
        <option value="unsure">Unsure events</option>
        <option value="highZ">High-z events</option>
        <option value="missingReviewer">Missing reviewer ID</option>
        <option value="reviewedByMe">Reviewed by me</option>
        <option value="reviewedByOther">Reviewed by other</option>
      </select>
      <button id="eventQueuePrevBtn">Prev Event Queue</button>
      <button id="eventQueueNextBtn">Next Event Queue</button>
      <span id="eventQueueStatusText" class="hint">0 events</span>
    </div>
    <div class="toolbar advancedOnly">
      <select id="eventType">
        <option value="">Event type</option>
        <option value="clear_transient">Clear transient</option>
        <option value="weak_transient">Weak transient</option>
        <option value="slow_transient">Slow transient</option>
        <option value="artifact">Artifact</option>
      </select>
      <select id="timingQuality">
        <option value="">Timing quality</option>
        <option value="clear_frame">Clear frame</option>
        <option value="ambiguous">Ambiguous</option>
        <option value="slow_transient">Slow transient</option>
      </select>
      <select id="eventConfidence">
        <option value="">Confidence</option>
        <option value="high">High</option>
        <option value="medium">Medium</option>
        <option value="low">Low</option>
      </select>
      <input id="eventReasonTags" placeholder="Reason tags" style="height:31px;border:1px solid var(--line);border-radius:6px;padding:0 8px;max-width:150px">
    </div>
    <textarea id="eventNotes" rows="2" placeholder="Notes for selected event"></textarea>
    <div class="eventList" id="eventList"></div>
    <details id="discoveryDetails" class="advancedOnly">
      <summary>Discovery Suggestions</summary>
      <div class="buttonRow">
        <button id="suggestionPromoteBtn" class="accept">Promote</button>
        <button id="suggestionPromoteNextBtn" class="accept">Promote + Next</button>
        <button id="suggestionMissedBtn">Missed Neuron</button>
        <button id="suggestionMissedNextBtn">Missed + Next</button>
        <button id="suggestionDuplicateBtn">Duplicate</button>
        <button id="suggestionDuplicateNextBtn">Duplicate + Next</button>
        <button id="suggestionArtifactBtn" class="reject">Artifact</button>
        <button id="suggestionArtifactNextBtn" class="reject">Artifact + Next</button>
        <button id="suggestionUnsureBtn" class="unsure">Unsure</button>
        <button id="suggestionUnsureNextBtn" class="unsure">Unsure + Next</button>
        <button id="suggestionClearBtn">Clear</button>
      </div>
      <div id="suggestionContextCard"></div>
      <select id="artifactClass" style="width:100%;margin:6px 0;height:30px">
        <option value="">Artifact class</option>
        <option value="vessel_static_structure">Vessel/static structure</option>
        <option value="motion_artifact">Motion artifact</option>
        <option value="background_fluctuation">Background fluctuation</option>
        <option value="impulse_noise">Impulse noise</option>
        <option value="border_artifact">Border artifact</option>
        <option value="saturation_bright_blob">Saturation/bright blob</option>
        <option value="uncertain_artifact">Uncertain artifact</option>
      </select>
      <textarea id="suggestionNotes" rows="2" placeholder="Notes for selected discovery suggestion"></textarea>
      <div class="toolbar">
        <select id="suggestionConfidence">
          <option value="">Confidence</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
        </select>
        <input id="suggestionReasonTags" placeholder="Reason tags" style="height:31px;border:1px solid var(--line);border-radius:6px;padding:0 8px;max-width:160px">
      </div>
      <div class="toolbar">
        <select id="discoveryQueueSelect">
          <option value="all">All suggestions</option>
          <option value="unlabeled">Unlabeled</option>
          <option value="promoted">Promoted</option>
          <option value="missed">Marked missed</option>
          <option value="artifact">Artifacts</option>
          <option value="artifactSuspects">Artifact suspects</option>
          <option value="missingReviewer">Missing reviewer ID</option>
          <option value="reviewedByMe">Reviewed by me</option>
          <option value="reviewedByOther">Reviewed by other</option>
        </select>
        <button id="suggestionQueuePrevBtn">Prev Suggestion</button>
        <button id="suggestionQueueNextBtn">Next Suggestion</button>
        <span id="suggestionQueueStatusText" class="hint">0 suggestions</span>
      </div>
      <div class="suggestionList" id="suggestionList"></div>
    </details>
    <h2>Review Queue</h2>
    <div class="toolbar advancedOnly">
      <select id="queueSelect">
        <option value="unlabeled">Unlabeled first</option>
        <option value="annotationBatch">Next annotation batch</option>
        <option value="priority">Priority score</option>
        <option value="strongNeuron">Strong neuron candidates</option>
        <option value="needsEventReview">Event review needed</option>
        <option value="weakTrace">Weak/noisy traces</option>
        <option value="mergedCluster">Merged/large clusters</option>
        <option value="artifactLike">Artifact-like</option>
        <option value="uncertain">Most uncertain</option>
        <option value="localCorrelation">Local correlation</option>
        <option value="eventSupport">Event support</option>
        <option value="traceSnr">Trace SNR</option>
        <option value="artifactRisk">Artifact risk</option>
        <option value="highEvents">High event count</option>
        <option value="highNoise">High noise</option>
        <option value="accepted">Accepted</option>
        <option value="rejected">Rejected</option>
        <option value="unsure">Unsure</option>
        <option value="missingReviewer">Missing reviewer ID</option>
        <option value="reviewedByMe">Reviewed by me</option>
        <option value="reviewedByOther">Reviewed by other</option>
        <option value="deleted">Hidden</option>
        <option value="needsAction">Needs action</option>
        <option value="controlReady">Control-ready</option>
        <option value="problemTrace">Problem traces</option>
        <option value="all">All active</option>
      </select>
      <label>area >= <input id="minArea" type="range" min="0" max="260" step="1"> <span id="minAreaLabel">0</span></label>
      <label>events >= <input id="minEvents" type="range" min="0" max="12" step="1"> <span id="minEventsLabel">0</span></label>
      <button id="queuePrevBtn">Prev Queue</button>
      <button id="queueNextBtn">Next Queue</button>
      <span id="queueStatusText" class="hint">0 queued</span>
    </div>
    <div class="roiList" id="roiList"></div>
    <details class="advancedOnly">
      <summary>Parameter Snapshots</summary>
      <div class="toolbar">
        <select id="parameterSnapshotSelect"></select>
        <button id="snapshotSaveBtn">Save Snapshot</button>
        <button id="snapshotRestoreBtn">Restore</button>
        <button id="snapshotDeleteBtn">Delete</button>
      </div>
      <p id="snapshotSummary" class="hint"></p>
    </details>
    <details class="advancedOnly">
      <summary>Autosave Recovery</summary>
      <div class="toolbar">
        <select id="recoverySnapshotSelect"></select>
        <button id="recoveryRestoreBtn">Restore</button>
        <button id="recoveryDownloadBtn">Download</button>
      </div>
      <p id="recoverySnapshotSummary" class="hint"></p>
    </details>
    <details class="advancedOnly">
      <summary>Export</summary>
      <div class="buttonRow">
        <button id="exportRoiBtn">Export ROI TSV</button>
        <button id="exportEventBtn">Export Event TSV</button>
        <button id="exportSuggestionBtn">Export Discovery TSV</button>
        <button id="exportSplitMergeBtn">Export Split/Merge TSV</button>
        <button id="exportActiveRoiQueueBtn">Export ROI Queue</button>
        <button id="exportActiveEventQueueBtn">Export Event Queue</button>
        <button id="exportActiveSuggestionQueueBtn">Export Suggestion Queue</button>
        <button id="exportJsonBtn">Export JSON</button>
        <button id="exportProvenanceAuditBtn">Export Provenance Audit</button>
      </div>
    </details>
    <details class="advancedOnly">
      <summary>Parameters</summary>
      <table class="smallTable" id="paramTable"></table>
    </details>
    <p class="hint">Run through the local server for autosave to <code>annotations.json</code>. Static file mode still keeps a browser-local backup and supports TSV export.</p>
  </aside>
  <section id="architecturePage" class="architecturePage hidden">
    <div class="topbar">
      <h1>Architecture Lab: {dataset_id}</h1>
      <nav class="navTabs">
        <a id="architectureTabArch" href="#architecture">Architecture Lab</a>
        <a id="experimentsTabArch" href="#experiments">Experiment Lab</a>
        <a id="qcTabArch" href="#process">Process Lab</a>
        <a id="reviewTabArch" href="#review">Review</a>
        <a id="metricsTabArch" href="#metrics">Metrics</a>
        <a id="reportTabArch" href="#report">Report</a>
      </nav>
      <label class="modeToggle">Theme
        <select class="themeSelect" aria-label="Theme">
          <option value="system">System</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </label>
    </div>
    <p class="hint">This page compares standardized architecture-run artifacts and builds planned pipeline manifests. Build mode configures and exports plans; it does not execute Fiji or Python pipelines in the browser.</p>
    <div class="toolbar">
      <button id="archCompareModeBtn">Compare</button>
      <button id="archBuildModeBtn">Build Pipeline</button>
      <label>Run A <select id="archRunA"></select></label>
      <label>Run B <select id="archRunB"></select></label>
    </div>
    <div id="architectureComparePanel">
      <div class="contextPanel" id="runComparison"></div>
      <div class="archGrid" id="architectureRuns"></div>
    </div>
    <div id="architectureBuildPanel" class="hidden">
      <section class="archCard pipelineIdentityCard">
        <h2>Pipeline Identity</h2>
        <div id="pipelineIdentityPanel"></div>
      </section>
      <div class="toolbar">
        <label>Preset
          <select id="pipelinePresetSelect">
            <option value="current_review_pipeline">Current-style local-z pipeline</option>
            <option value="adaptive_cfar">Adaptive CFAR detector</option>
            <option value="artifact_suppression">Artifact suppression pass</option>
            <option value="high_recall_discovery">High-recall discovery</option>
            <option value="motion_aware">Motion-aware QC</option>
            <option value="pmd_import">PMD denoised local-z</option>
            <option value="suite2p_import">Suite2p import</option>
            <option value="oasis_import">OASIS event model</option>
          </select>
        </label>
        <button id="pipelineNewArchitectureBtn">New Architecture</button>
        <button id="pipelineNewBtn">Load Preset</button>
        <button id="pipelineCloneRunBtn">Clone Run A</button>
        <button id="pipelineSaveTemplateBtn">Save Architecture</button>
        <button id="pipelineSaveBtn">Save Planned Run</button>
        <button id="pipelineUseExperimentBtn">Use In Experiment Lab</button>
        <button id="pipelineDownloadBtn">Download JSON</button>
      </div>
      <div class="pipelineBuilderGrid">
        <section class="archCard">
          <h2>Stage Palette</h2>
          <div id="pipelineStagePalette"></div>
        </section>
        <section class="archCard">
          <h2>Pipeline Stack</h2>
          <div id="pipelineStack"></div>
        </section>
        <section class="archCard">
          <h2>Parameters</h2>
          <div id="pipelineInspector"></div>
        </section>
      </div>
      <section class="archCard architecturePresets">
        <h2>Recommended Architectures</h2>
        <div id="architecturePresetGallery"></div>
      </section>
      <section class="archCard componentLibraryShell">
        <h2>Component Library</h2>
        <div id="componentLibrary"></div>
      </section>
      <div class="pipelineBuilderGrid bottom">
        <section class="archCard">
          <h2>Validation</h2>
          <div id="pipelineValidation"></div>
        </section>
        <section class="archCard wide">
          <details open>
            <summary>Advanced Manifest Preview</summary>
            <pre id="pipelineJsonPreview"></pre>
          </details>
        </section>
      </div>
    </div>
  </section>
  <section id="experimentsPage" class="architecturePage hidden">
    <div class="topbar">
      <h1>Experiment Lab: {dataset_id}</h1>
      <nav class="navTabs">
        <a id="architectureTabExperiments" href="#architecture">Architecture Lab</a>
        <a id="experimentsTabExperiments" href="#experiments">Experiment Lab</a>
        <a id="qcTabExperiments" href="#process">Process Lab</a>
        <a id="reviewTabExperiments" href="#review">Review</a>
        <a id="metricsTabExperiments" href="#metrics">Metrics</a>
        <a id="reportTabExperiments" href="#report">Report</a>
      </nav>
      <label class="modeToggle">Theme
        <select class="themeSelect" aria-label="Theme">
          <option value="system">System</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </label>
    </div>
    <div id="experimentLab"></div>
  </section>
  <section id="metricsPage" class="architecturePage hidden">
    <div class="topbar">
      <h1>Metrics/Audit: {dataset_id}</h1>
      <nav class="navTabs">
        <a id="architectureTabMetrics" href="#architecture">Architecture Lab</a>
        <a id="experimentsTabMetrics" href="#experiments">Experiment Lab</a>
        <a id="qcTabMetrics" href="#process">Process Lab</a>
        <a id="reviewTabMetrics" href="#review">Review</a>
        <a id="metricsTabMetrics" href="#metrics">Metrics</a>
        <a id="reportTabMetrics" href="#report">Report</a>
      </nav>
      <label class="modeToggle">Theme
        <select class="themeSelect" aria-label="Theme">
          <option value="system">System</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </label>
    </div>
    <p class="hint">This page summarizes the current annotation state from the live autosave data. It is useful for tracking review progress, burden, accepted events, control-ready ROIs, and discovery outcomes.</p>
    <div id="metricsAudit"></div>
  </section>
  <section id="qcPage" class="architecturePage hidden">
    <div class="topbar">
      <h1>Process Lab: {dataset_id}</h1>
      <nav class="navTabs">
        <a id="architectureTabQc" href="#architecture">Architecture Lab</a>
        <a id="experimentsTabQc" href="#experiments">Experiment Lab</a>
        <a id="qcTabQc" href="#process">Process Lab</a>
        <a id="reviewTabQc" href="#review">Review</a>
        <a id="metricsTabQc" href="#metrics">Metrics</a>
        <a id="reportTabQc" href="#report">Report</a>
      </nav>
      <label class="modeToggle">Theme
        <select class="themeSelect" aria-label="Theme">
          <option value="system">System</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </label>
    </div>
    <p class="hint">This page inspects the active architecture run in pipeline order, including raw frames, generated intermediates, artifact states, and lightweight process warnings.</p>
    <div id="datasetQc"></div>
  </section>
  <section id="reportPage" class="architecturePage hidden">
    <div class="topbar">
      <h1>Review Report: {dataset_id}</h1>
      <nav class="navTabs">
        <a id="architectureTabReport" href="#architecture">Architecture Lab</a>
        <a id="experimentsTabReport" href="#experiments">Experiment Lab</a>
        <a id="qcTabReport" href="#process">Process Lab</a>
        <a id="reviewTabReport" href="#review">Review</a>
        <a id="metricsTabReport" href="#metrics">Metrics</a>
        <a id="reportTabReport" href="#report">Report</a>
      </nav>
      <label class="modeToggle">Theme
        <select class="themeSelect" aria-label="Theme">
          <option value="system">System</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </label>
    </div>
    <div id="reportPageBody"></div>
  </section>
</div>
<script id="review-data" type="application/json">{data_json}</script>
<script src="workbench.js"></script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the interactive neuron annotation workbench.")
    parser.add_argument("--app-dir", type=Path, default=None)
    parser.add_argument("--review-data", type=Path, default=None)
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    parser.add_argument("--architecture-runs", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = resolve_build_inputs(
        app_dir=args.app_dir,
        review_data=args.review_data,
        dataset_manifest=args.dataset_manifest,
        architecture_runs=args.architecture_runs,
        default_app_dir=DEFAULT_APP_DIR,
        default_review_data=DEFAULT_DATA_PATH,
        default_dataset_id="calcium_video_2",
    )
    paths = build_workbench(
        app_dir=inputs["app_dir"],
        review_data_path=inputs["review_data_path"],
        dataset_id=inputs["dataset_id"],
        html_template=HTML_TEMPLATE,
        dataset_manifest=inputs["dataset_manifest"],
        architecture_runs_path=inputs["architecture_runs_path"],
        css_fallback=CSS,
        js_fallback=JS,
    )
    print(f"Wrote workbench to {paths['index']}")


if __name__ == "__main__":
    main()
