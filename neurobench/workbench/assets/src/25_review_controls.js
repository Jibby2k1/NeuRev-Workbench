function initControls(){
  slider.max = data.video.frames;
  slider.oninput = () => setFrame(Number(slider.value));
  document.getElementById('playBtn').onclick = togglePlay;
  document.getElementById('fitBtn').onclick = fitWidth;
  document.getElementById('fitHeightBtn').onclick = fitHeight;
  document.getElementById('fullscreenBtn').onclick = () => viewerScroll.requestFullscreen?.();
  document.getElementById('prevActiveFrameBtn').onclick = () => nextActiveFrame(-1);
  document.getElementById('nextActiveFrameBtn').onclick = () => nextActiveFrame(1);
  document.getElementById('exportScreenshotBtn').onclick = exportCurrentViewPng;
  document.getElementById('eventWindowPrevBtn').onclick = () => nextEvent(-1);
  document.getElementById('eventWindowNextBtn').onclick = () => nextEvent(1);
  document.getElementById('activeRunSelect').onchange = e => selectActiveRun(e.target.value, {loadReview:true});
  document.getElementById('loadRunReviewBtn').onclick = () => selectActiveRun(document.getElementById('activeRunSelect').value, {loadReview:true});
  document.getElementById('openRunViewBtn').onclick = () => {
    const run = runById(activeRunId());
    const url = artifactUrl(runAppUrl(run));
    if(url) location.href = url;
  };
  document.getElementById('previewRunViewBtn').onclick = () => startGenerationJob({preview:true});
  document.getElementById('generateRunViewBtn').onclick = () => startGenerationJob({preview:false});
  document.getElementById('unlockGenerationBtn').onclick = () => {
    const token = prompt('Owner token for local generation jobs');
    if(token !== null) {
      generationOwnerToken = token.trim();
      if(generationOwnerToken) localStorage.setItem(ownerTokenKey, generationOwnerToken);
      else localStorage.removeItem(ownerTokenKey);
      renderRunSyncControls();
    }
  };
  document.getElementById('refreshRunBtn').onclick = refreshArchitectureRuns;
  document.getElementById('generationBackend').onchange = renderRunSyncControls;
  document.getElementById('archRunA').onchange = e => {
    reviewCompareSettings().runAId = e.target.value;
    queueSave();
    renderRunComparison();
  };
  document.getElementById('archRunB').onchange = e => {
    reviewCompareSettings().runBId = e.target.value;
    queueSave();
    renderRunComparison();
  };
  document.getElementById('archCompareModeBtn').onclick = () => setArchitectureMode('compare');
  document.getElementById('archBuildModeBtn').onclick = () => setArchitectureMode('build');
  document.getElementById('pipelineNewArchitectureBtn').onclick = () => {
    pipelineDraft = makePresetPipeline('current_review_pipeline');
    pipelineDraft.label = 'Untitled architecture';
    pipelineDraft.run_id = `planned_architecture_${Date.now().toString(36)}`;
    pipelineDraft.template_id = slugify(pipelineDraft.run_id);
    pipelineDraft.description = '';
    selectedPipelineStageId = pipelineDraft.pipeline[0]?.id || null;
    setArchitectureMode('build');
    renderPipelineBuilder();
  };
  document.getElementById('pipelineNewBtn').onclick = () => {
    pipelineDraft = makePresetPipeline(document.getElementById('pipelinePresetSelect').value);
    pipelineDraft.template_id = slugify(pipelineDraft.run_id);
    selectedPipelineStageId = pipelineDraft.pipeline[0]?.id || null;
    renderPipelineBuilder();
  };
  document.getElementById('pipelinePresetSelect').onchange = e => {
    pipelineDraft = makePresetPipeline(e.target.value);
    pipelineDraft.template_id = slugify(pipelineDraft.run_id);
    selectedPipelineStageId = pipelineDraft.pipeline[0]?.id || null;
    renderPipelineBuilder();
  };
  document.getElementById('pipelineCloneRunBtn').onclick = () => {
    const runs = data.architectureRuns?.runs || [];
    const selected = runs.find(r => r.run_id === document.getElementById('archRunA').value) || runs[0];
    if(selected) {
      pipelineDraft = normalizePipelineDraft(JSON.parse(JSON.stringify(Object.assign({}, selected, {execution:{status:'planned'}}))));
      pipelineDraft.run_id = `planned_clone_${Date.now().toString(36)}`;
      pipelineDraft.label = `Planned clone of ${selected.label || selected.run_id}`;
      pipelineDraft.template_id = slugify(pipelineDraft.run_id);
      selectedPipelineStageId = pipelineDraft.pipeline?.[0]?.id || null;
      renderPipelineBuilder();
    }
  };
  document.getElementById('pipelineSaveTemplateBtn').onclick = saveArchitectureTemplate;
  document.getElementById('pipelineSaveBtn').onclick = savePlannedRun;
  document.getElementById('pipelineUseExperimentBtn').onclick = useCurrentArchitectureInExperiment;
  document.getElementById('pipelineDownloadBtn').onclick = () => downloadJson('planned_architecture_run.json', plannedManifest());
  for(const select of document.querySelectorAll('.uiModeSelect')) select.onchange = e => {
    setSetting('uiMode', normalizeUiMode(e.target.value));
    applySettingsToControls();
  };
  for(const select of document.querySelectorAll('.themeSelect')) select.onchange = e => {
    setSetting('theme', e.target.value);
    applyTheme();
  };
  window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener?.('change', () => {
    if((setting('theme') || 'system') === 'system') applyTheme();
  });
  document.getElementById('reviewerIdInput').onchange = e => {
    setSetting('reviewerId', e.target.value.trim());
    recordAction('reviewer_id_set');
    renderAll();
  };
  document.getElementById('nextMissingReviewerBtn').onclick = nextMissingReviewerLabel;
  document.getElementById('stampSelectedReviewerBtn').onclick = stampSelectedReviewer;
  document.getElementById('stampMissingReviewerBtn').onclick = stampMissingReviewerLabels;
  document.getElementById('reviewModeToggle').onclick = () => {
    const next = setting('reviewMode') === 'guided' ? 'explore' : 'guided';
    setSetting('reviewMode', next);
    setSetting('reviewWorkflowPreset', 'custom');
    if(next === 'guided') {
      setSetting('queue', 'annotationBatch');
      selectGuidedTask();
    }
    applySettingsToControls();
    renderAll();
  };
  document.getElementById('reviewWorkflowPreset').onchange = e => applyReviewWorkflowPreset(e.target.value);
  document.getElementById('shortcutHelpBtn').onclick = () => toggleShortcutHelp(true);
  document.getElementById('shortcutCloseBtn').onclick = () => toggleShortcutHelp(false);
  document.getElementById('shortcutOverlay').addEventListener('click', e => {
    if(e.target.id === 'shortcutOverlay') toggleShortcutHelp(false);
  });
  document.getElementById('quickJumpBtn').onclick = () => quickJump(document.getElementById('quickJumpInput').value);
  document.getElementById('quickJumpInput').addEventListener('keydown', e => {
    if(e.key === 'Enter') {
      e.preventDefault();
      quickJump(e.target.value);
    }
  });
  document.getElementById('undoAnnotationBtn').onclick = undoLastAnnotationChange;
  document.getElementById('bookmarkAddBtn').onclick = addReviewBookmark;
  document.getElementById('bookmarkGoBtn').onclick = goToReviewBookmark;
  document.getElementById('bookmarkDeleteBtn').onclick = deleteReviewBookmark;
  for(const id of ['showRois','showEvents']) document.getElementById(id).onchange = drawOverlay;
  document.getElementById('showLabels').onchange = e => { setSetting('roiLabelMode', e.target.checked ? 'all' : 'off'); applySettingsToControls(); drawOverlay(); };
  document.getElementById('roiLabelMode')?.addEventListener('change', e => { setSetting('roiLabelMode', normalizeRoiLabelMode(e.target.value)); applySettingsToControls(); drawOverlay(); });
  document.getElementById('showSuggestions').onchange = e => { setSetting('showSuggestions', e.target.checked); drawOverlay(); };
  document.getElementById('showStencilOverlay')?.addEventListener('change', e => { setSetting('showStencilOverlay', e.target.checked); drawOverlay(); });
  for(const id of ['showTemplateOverlay','showRegisteredProjectionOverlay','showGridOverlay','showGridIntensityOverlay','showPredictionErrorOverlay']) {
    document.getElementById(id)?.addEventListener('change', e => { setSetting(id, e.target.checked); updateGridCellStatus(); drawOverlay(); });
  }
  document.getElementById('showEvidence').onchange = e => { setSetting('showEvidence', e.target.checked); applyDisplaySettings(); };
  document.getElementById('evidenceSelect').onchange = e => { setSetting('evidenceMap', e.target.value); applyDisplaySettings(); };
  document.getElementById('togglePotentialRoisBtn').onclick = () => toggleOverlayRoiGroup('showPotentialRois');
  document.getElementById('toggleAnnotatedNeuronRoisBtn').onclick = () => toggleOverlayRoiGroup('showAnnotatedNeuronRois');
  document.getElementById('toggleAnnotatedNonNeuronRoisBtn').onclick = () => toggleOverlayRoiGroup('showAnnotatedNonNeuronRois');
  document.getElementById('showAllRoiGroupsBtn').onclick = showAllOverlayRoiGroups;
  const overlayPresetSelect = document.getElementById('overlayPresetSelect');
  if(overlayPresetSelect) overlayPresetSelect.onchange = e => applyOverlayPreset(e.target.value);
  const selectedOverlayMode = document.getElementById('selectedOverlayMode');
  if(selectedOverlayMode) selectedOverlayMode.onchange = e => {
    setSetting('selectedOverlayMode', e.target.value);
    setSetting('overlayPreset', 'custom');
    applySettingsToControls();
    drawOverlay();
  };
  const roiFocusSelect = document.getElementById('roiFocusMode');
  if(roiFocusSelect) roiFocusSelect.onchange = e => {
    setSetting('roiFocusMode', e.target.value);
    setSetting('overlayScope', 'focus');
    applySettingsToControls();
    renderAll();
  };
  const traceResetZoomBtn = document.getElementById('traceResetZoomBtn');
  if(traceResetZoomBtn) traceResetZoomBtn.onclick = resetTraceZoom;
  document.getElementById('manualRoiMode').onchange = e => {
    setSetting('manualRoiMode', e.target.value);
    manualRoiState = {drawing:false, start:null, points:[], preview:null, suppressClick:false};
    if(e.target.value !== 'select') setSetting('roiEditMode', 'off');
    applySettingsToControls();
    drawOverlay();
  };
  document.getElementById('manualRoiCancelBtn').onclick = cancelManualRoi;
  document.getElementById('startManualNeuronBtn').onclick = () => {
    setSetting('manualRoiMode', 'center');
    setSetting('roiEditMode', 'off');
    applySettingsToControls();
    setSaveState('click a neuron center in the video to add a missed-neuron ROI', 'ok');
  };
  document.getElementById('markEventStartBtn').onclick = () => setManualEventWindowFrame('start');
  document.getElementById('markEventEndBtn').onclick = () => setManualEventWindowFrame('end');
  document.getElementById('saveManualEventWindowBtn').onclick = addManualEventWindow;
  document.getElementById('roiEditMode').onchange = e => {
    setSetting('roiEditMode', e.target.value);
    roiEditState = {drawing:false, editedId:null};
    if(e.target.value !== 'off') setSetting('manualRoiMode', 'select');
    applySettingsToControls();
    drawOverlay();
  };
  document.getElementById('roiEditDoneBtn').onclick = () => {
    setSetting('roiEditMode', 'off');
    roiEditState = {drawing:false, editedId:null};
    applySettingsToControls();
    drawOverlay();
  };
  document.getElementById('roiEditUndoBtn').onclick = undoRoiEdit;
  document.getElementById('roiEditRevertBtn').onclick = revertEditedRoiToSource;
  document.getElementById('materializeManualTracesBtn').onclick = materializeManualTraces;
  document.getElementById('traceFullBtn').onclick = () => applyTracePreset('full');
  document.getElementById('traceEvent2sBtn').onclick = () => applyTracePreset('event2s');
  document.getElementById('traceEvent5sBtn').onclick = () => applyTracePreset('event5s');
  for(const id of ['eventThreshold','kalmanGain','spikeGain','zoom','brightness','contrast','overlayOpacity','selectedFillOpacity','selectedOutlineWidth','neighborRadiusPx','manualRoiRadius','roiEditBrushRadius','minArea','minEvents']) {
    const control = document.getElementById(id);
    if(!control) continue;
    control.oninput = e => {
      const value = Number(e.target.value);
      setSetting(id, value);
      if(id === 'overlayOpacity' || id === 'selectedFillOpacity' || id === 'selectedOutlineWidth') setSetting('overlayPreset', 'custom');
      if(id === 'kalmanGain' || id === 'spikeGain') clearTraceCaches(id);
      if(id === 'eventThreshold') clearTraceEventCache(id);
      applySettingsToControls();
      scheduleRenderAll();
    };
  }
  document.getElementById('queueSelect').onchange = e => { setSetting('queue', e.target.value); setSetting('reviewWorkflowPreset', 'custom'); renderAll(); };
  document.getElementById('queuePrevBtn').onclick = () => nextRoi(-1);
  document.getElementById('queueNextBtn').onclick = () => nextRoi(1);
  document.getElementById('nextActiveRoiBtn').onclick = () => nextRoiMatching(roi => eventsForRoi(roi).length > 0, 1);
  document.getElementById('nextUncertainRoiBtn').onclick = () => nextRoiMatching(roi => !roiAnn(roi.id).state || roiAnn(roi.id).state === 'unsure', 1);
  document.getElementById('nextArtifactRiskBtn').onclick = () => nextRoiMatching(roiArtifactLike, 1);
  document.getElementById('missedNeuronModeBtn').onclick = activateMissedNeuronMode;
  document.getElementById('bulkAcceptBtn').onclick = () => setSelectedRoisState('accept');
  document.getElementById('bulkRejectBtn').onclick = () => setSelectedRoisState('reject');
  document.getElementById('bulkUnsureBtn').onclick = () => setSelectedRoisState('unsure');
  document.getElementById('bulkIdentityBtn').onclick = assignSelectedIdentity;
  document.getElementById('bulkNeedsActionBtn').onclick = markSelectedAction;
  document.getElementById('virtualMergeBtn').onclick = createVirtualMerge;
  document.getElementById('visualSplitBtn').onclick = createVisualSplitDecision;
  document.getElementById('clearMultiSelectBtn').onclick = clearMultiSelection;
  document.getElementById('acceptBtn').onclick = () => setRoiState('accept');
  document.getElementById('rejectBtn').onclick = () => setRoiState('reject');
  document.getElementById('unsureBtn').onclick = () => setRoiState('unsure');
  document.getElementById('acceptNextBtn').onclick = () => setRoiStateAndNext('accept');
  document.getElementById('rejectNextBtn').onclick = () => setRoiStateAndNext('reject');
  document.getElementById('unsureNextBtn').onclick = () => setRoiStateAndNext('unsure');
  document.getElementById('strongNeuronNextBtn').onclick = markRoiStrongAndNext;
  document.getElementById('artifactRoiNextBtn').onclick = markRoiArtifactAndNext;
  document.getElementById('clearBtn').onclick = () => setRoiState('');
  document.getElementById('deleteBtn').onclick = toggleDeleted;
  document.getElementById('eventAcceptBtn').onclick = () => setEventState('accept');
  document.getElementById('eventRejectBtn').onclick = () => setEventState('reject');
  document.getElementById('eventUnsureBtn').onclick = () => setEventState('unsure');
  document.getElementById('eventAcceptNextBtn').onclick = () => setEventStateAndNext('accept');
  document.getElementById('eventRejectNextBtn').onclick = () => setEventStateAndNext('reject');
  document.getElementById('eventUnsureNextBtn').onclick = () => setEventStateAndNext('unsure');
  document.getElementById('eventArtifactNextBtn').onclick = markEventArtifactAndNext;
  document.getElementById('eventClearBtn').onclick = () => setEventState('');
  document.getElementById('eventQueueSelect').onchange = e => {
    setSetting('eventQueue', e.target.value);
    setSetting('reviewWorkflowPreset', 'custom');
    renderAll();
  };
  document.getElementById('eventQueuePrevBtn').onclick = () => nextEventQueue(-1);
  document.getElementById('eventQueueNextBtn').onclick = () => nextEventQueue(1);
  document.getElementById('suggestionPromoteBtn').onclick = promoteSuggestion;
  document.getElementById('suggestionPromoteNextBtn').onclick = promoteSuggestionAndNext;
  document.getElementById('suggestionMissedBtn').onclick = () => setSuggestionState('missed');
  document.getElementById('suggestionMissedNextBtn').onclick = () => setSuggestionStateAndNext('missed');
  document.getElementById('suggestionDuplicateBtn').onclick = markSuggestionDuplicate;
  document.getElementById('suggestionDuplicateNextBtn').onclick = markSuggestionDuplicateAndNext;
  document.getElementById('suggestionArtifactBtn').onclick = () => setSuggestionState('artifact');
  document.getElementById('suggestionArtifactNextBtn').onclick = () => setSuggestionStateAndNext('artifact');
  document.getElementById('suggestionUnsureBtn').onclick = () => setSuggestionState('unsure');
  document.getElementById('suggestionUnsureNextBtn').onclick = () => setSuggestionStateAndNext('unsure');
  document.getElementById('suggestionClearBtn').onclick = () => setSuggestionState('');
  document.getElementById('artifactClass').onchange = e => {
    const s = selectedSuggestion(); if(!s) return;
    annotations.suggestions[s.id] = Object.assign(suggestionAnn(s.id), {artifactClass:e.target.value, artifact_class:e.target.value});
    queueSave();
    renderAll();
  };
  document.getElementById('discoveryQueueSelect').onchange = e => { setSetting('discoveryQueue', e.target.value); renderAll(); };
  document.getElementById('suggestionQueuePrevBtn').onclick = () => nextSuggestion(-1);
  document.getElementById('suggestionQueueNextBtn').onclick = () => nextSuggestion(1);
  document.getElementById('exportRoiBtn').onclick = () => exportRows('roi');
  document.getElementById('exportEventBtn').onclick = () => exportRows('event');
  document.getElementById('exportSuggestionBtn').onclick = () => exportRows('suggestion');
  document.getElementById('exportSplitMergeBtn').onclick = () => exportRows('splitMerge');
  document.getElementById('exportActiveRoiQueueBtn').onclick = () => exportActiveQueue('roi');
  document.getElementById('exportActiveEventQueueBtn').onclick = () => exportActiveQueue('event');
  document.getElementById('exportActiveSuggestionQueueBtn').onclick = () => exportActiveQueue('suggestion');
  document.getElementById('exportJsonBtn').onclick = exportJson;
  document.getElementById('exportProvenanceAuditBtn').onclick = exportReviewerProvenanceAudit;
  document.getElementById('snapshotSaveBtn').onclick = saveParameterSnapshot;
  document.getElementById('snapshotRestoreBtn').onclick = () => restoreParameterSnapshot(document.getElementById('parameterSnapshotSelect').value);
  document.getElementById('snapshotDeleteBtn').onclick = deleteParameterSnapshot;
  document.getElementById('parameterSnapshotSelect').onchange = e => setSetting('activeSnapshotId', e.target.value);
  document.getElementById('recoveryRestoreBtn').onclick = restoreRecoverySnapshot;
  document.getElementById('recoveryDownloadBtn').onclick = downloadRecoverySnapshot;
  for (const [id, field] of [['traceQuality','trace_quality'],['controlReady','control_ready'],['roiArtifactClass','artifact_class'],['needsAction','needs_action'],['roiConfidence','confidence']]) {
    document.getElementById(id).onchange = e => {
      const roi = selectedRoi(); if(!roi) return;
      annotations.rois[roi.id] = stampAnnotation(Object.assign(roiAnn(roi.id), {[field]: e.target.value}));
      if(annotations.virtualRois[roi.id]) stampAnnotation(Object.assign(annotations.virtualRois[roi.id], {[field]: e.target.value}));
      recordAction(`roi_${field}`);
      queueSave();
      renderAll();
    };
  }
  document.getElementById('roiReasonTags').onchange = e => {
    const roi = selectedRoi(); if(!roi) return;
    const tags = normalizeIdList(e.target.value);
    annotations.rois[roi.id] = stampAnnotation(Object.assign(roiAnn(roi.id), {reason_tags: tags}));
    if(annotations.virtualRois[roi.id]) stampAnnotation(Object.assign(annotations.virtualRois[roi.id], {reason_tags: tags}));
    recordAction('roi_reason_tags');
    queueSave();
    renderAll();
  };
  document.getElementById('identityGroup').oninput = e => {
    const roi = selectedRoi(); if(!roi) return;
    annotations.rois[roi.id] = stampAnnotation(Object.assign(roiAnn(roi.id), {identity_group:e.target.value}));
    if(annotations.virtualRois[roi.id]) stampAnnotation(Object.assign(annotations.virtualRois[roi.id], {identity_group:e.target.value}));
    recordAction('roi_identity_group');
    queueSave();
  };
  for (const [id, field] of [['eventType','event_type'],['timingQuality','timing_quality'],['eventConfidence','confidence']]) {
    document.getElementById(id).onchange = e => {
      const roi = selectedRoi(); if(!roi || !selectedEventFrame) return;
      annotations.events[eventKey(roi.id, selectedEventFrame)] = stampAnnotation(Object.assign(eventAnn(roi.id, selectedEventFrame), {[field]: e.target.value}));
      recordAction(`event_${field}`);
      queueSave();
      renderAll();
    };
  }
  document.getElementById('eventReasonTags').onchange = e => {
    const roi = selectedRoi(); if(!roi || !selectedEventFrame) return;
    annotations.events[eventKey(roi.id, selectedEventFrame)] = stampAnnotation(Object.assign(eventAnn(roi.id, selectedEventFrame), {reason_tags: normalizeIdList(e.target.value)}));
    recordAction('event_reason_tags');
    queueSave();
    renderAll();
  };
  roiNotes.oninput = e => {
    const roi = selectedRoi(); if(!roi) return;
    annotations.rois[roi.id] = stampAnnotation(Object.assign(roiAnn(roi.id), {notes:e.target.value}));
    if(annotations.virtualRois[roi.id]) stampAnnotation(Object.assign(annotations.virtualRois[roi.id], {notes:e.target.value}));
    queueSave();
  };
  eventNotes.oninput = e => {
    const roi = selectedRoi(); if(!roi || !selectedEventFrame) return;
    annotations.events[eventKey(roi.id, selectedEventFrame)] = stampAnnotation(Object.assign(eventAnn(roi.id, selectedEventFrame), {notes:e.target.value}));
    queueSave();
  };
  document.getElementById('suggestionNotes').oninput = e => {
    const s = selectedSuggestion(); if(!s) return;
    annotations.suggestions[s.id] = stampAnnotation(Object.assign(suggestionAnn(s.id), {notes:e.target.value}));
    queueSave();
  };
  document.getElementById('suggestionConfidence').onchange = e => {
    const s = selectedSuggestion(); if(!s) return;
    annotations.suggestions[s.id] = stampAnnotation(Object.assign(suggestionAnn(s.id), {confidence:e.target.value}));
    recordAction('suggestion_confidence');
    queueSave();
    renderAll();
  };
  document.getElementById('suggestionReasonTags').onchange = e => {
    const s = selectedSuggestion(); if(!s) return;
    annotations.suggestions[s.id] = stampAnnotation(Object.assign(suggestionAnn(s.id), {reason_tags: normalizeIdList(e.target.value)}));
    recordAction('suggestion_reason_tags');
    queueSave();
    renderAll();
  };
  traceCanvas.addEventListener('pointerdown', e => {
    const point = traceCanvasPoint(e);
    const roi = selectedRoi();
    const ev = traceEventAtPoint(point, roi);
    if(ev) {
      selectTraceEvent(ev, roi);
      return;
    }
    traceView.dragging = true;
    traceCanvas.setPointerCapture?.(e.pointerId);
    setFrame(traceFrameFromX(point.x));
  });
  traceCanvas.addEventListener('pointermove', e => {
    if(!traceView.dragging) return;
    setFrame(traceFrameFromX(traceCanvasPoint(e).x));
  });
  traceCanvas.addEventListener('pointerup', e => {
    traceView.dragging = false;
    traceCanvas.releasePointerCapture?.(e.pointerId);
  });
  traceCanvas.addEventListener('pointercancel', () => { traceView.dragging = false; });
  traceCanvas.addEventListener('wheel', e => {
    e.preventDefault();
    const point = traceCanvasPoint(e);
    const bounds = traceBounds();
    const pointerFrame = traceFrameFromX(point.x);
    const plotW = Math.max(1, traceCanvas.width - 2 * TRACE_PAD);
    const ratio = Math.max(0, Math.min(1, (point.x - TRACE_PAD) / plotW));
    const factor = e.deltaY > 0 ? 1.2 : 0.82;
    const span = Math.max(1, (bounds.end - bounds.start) * factor);
    setTraceWindow(pointerFrame - span * ratio, pointerFrame + span * (1 - ratio));
    drawTrace();
  }, {passive:false});
  traceCanvas.addEventListener('dblclick', e => {
    e.preventDefault();
    resetTraceZoom();
  });
  if(eventTimelineCanvas) {
    eventTimelineCanvas.addEventListener('pointerdown', e => {
      const rect = eventTimelineCanvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * eventTimelineCanvas.width / rect.width;
      setFrame(timelineFrameFromX(x));
      eventTimelineCanvas.setPointerCapture?.(e.pointerId);
      eventTimelineCanvas.dataset.dragging = '1';
    });
    eventTimelineCanvas.addEventListener('pointermove', e => {
      if(eventTimelineCanvas.dataset.dragging !== '1') return;
      const rect = eventTimelineCanvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * eventTimelineCanvas.width / rect.width;
      setFrame(timelineFrameFromX(x));
    });
    eventTimelineCanvas.addEventListener('pointerup', e => {
      eventTimelineCanvas.dataset.dragging = '';
      eventTimelineCanvas.releasePointerCapture?.(e.pointerId);
    });
    eventTimelineCanvas.addEventListener('pointercancel', () => { eventTimelineCanvas.dataset.dragging = ''; });
  }
  overlay.addEventListener('pointerdown', e => {
    const mode = setting('manualRoiMode') || 'select';
    const editMode = setting('roiEditMode') || 'off';
    if(mode === 'select' && editMode !== 'off') {
      e.preventDefault();
      const p = overlayPointFromEvent(e);
      const editable = ensureEditableRoi(selectedRoi());
      if(!editable) {
        setSaveState('select an ROI mask before brush editing', 'bad');
        return;
      }
      pushRoiEditHistory(editable, editMode);
      roiEditState = {drawing:true, editedId:String(editable.id)};
      overlay.setPointerCapture?.(e.pointerId);
      applyRoiBrush(p, editable);
      return;
    }
    if(mode === 'select') return;
    e.preventDefault();
    const p = overlayPointFromEvent(e);
    manualRoiState = {drawing:true, start:p, points:[p], preview:null, suppressClick:true};
    overlay.setPointerCapture?.(e.pointerId);
    if(mode === 'center') {
      createManualRoi('manual_center', circlePoints(p.x, p.y, Number(setting('manualRoiRadius')) || 6), 'Manual center ROI');
      manualRoiState = {drawing:false, start:null, points:[], preview:null, suppressClick:true};
      setTimeout(() => { manualRoiState.suppressClick = false; }, 0);
    }
  });
  overlay.addEventListener('pointermove', e => {
    if(roiEditState.drawing) {
      e.preventDefault();
      applyRoiBrush(overlayPointFromEvent(e), annotations.virtualRois[roiEditState.editedId]);
      return;
    }
    const mode = setting('manualRoiMode') || 'select';
    if(mode === 'select' || !manualRoiState.drawing) return;
    const p = overlayPointFromEvent(e);
    if(mode === 'circle') {
      const dx = p.x - manualRoiState.start.x, dy = p.y - manualRoiState.start.y;
      manualRoiState.preview = {type:'circle', x:manualRoiState.start.x, y:manualRoiState.start.y, radius:Math.max(1, Math.sqrt(dx*dx + dy*dy))};
    } else if(mode === 'lasso') {
      const last = manualRoiState.points[manualRoiState.points.length - 1];
      if(!last || distance(last, p) >= 1.5) manualRoiState.points.push(p);
    }
    drawOverlay();
  });
  overlay.addEventListener('pointerup', e => {
    if(roiEditState.drawing) {
      e.preventDefault();
      roiEditState = {drawing:false, editedId:null};
      overlay.releasePointerCapture?.(e.pointerId);
      recordAction('roi_brush_edit');
      queueSave();
      return;
    }
    const mode = setting('manualRoiMode') || 'select';
    if(mode === 'select' || !manualRoiState.drawing) return;
    e.preventDefault();
    const p = overlayPointFromEvent(e);
    if(mode === 'circle') {
      const dx = p.x - manualRoiState.start.x, dy = p.y - manualRoiState.start.y;
      const radius = Math.max(1, Math.sqrt(dx*dx + dy*dy));
      createManualRoi('manual_circle', circlePoints(manualRoiState.start.x, manualRoiState.start.y, radius), 'Manual circle ROI');
    } else if(mode === 'lasso') {
      manualRoiState.points.push(p);
      createManualRoi('manual_lasso', lassoPoints(manualRoiState.points), 'Manual lasso ROI');
    }
    overlay.releasePointerCapture?.(e.pointerId);
    manualRoiState = {drawing:false, start:null, points:[], preview:null, suppressClick:true};
    setTimeout(() => { manualRoiState.suppressClick = false; }, 0);
  });
  overlay.addEventListener('pointercancel', () => {
    roiEditState = {drawing:false, editedId:null};
    manualRoiState = {drawing:false, start:null, points:[], preview:null, suppressClick:false};
    drawOverlay();
  });
  overlay.onclick = e => {
    if(manualRoiState.suppressClick || (setting('manualRoiMode') || 'select') !== 'select' || (setting('roiEditMode') || 'off') !== 'off') return;
    const rect = overlay.getBoundingClientRect();
    const x = (e.clientX - rect.left) * data.video.width / rect.width;
    const y = (e.clientY - rect.top) * data.video.height / rect.height;
    if(handleGridCellClick(x, y)) return;
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
    else if(best) selectRoi(best.id, e.shiftKey || e.ctrlKey || e.metaKey);
  };
  document.addEventListener('keydown', e => {
    if(e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
    if((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      undoLastAnnotationChange();
      return;
    }
    if(e.key === 'Escape') toggleShortcutHelp(false);
    else if(e.key === '?'){ e.preventDefault(); toggleShortcutHelp(); }
    else if(e.code === 'Space'){ e.preventDefault(); togglePlay(); }
    else if(e.key === 'ArrowRight') setFrame(currentFrame + 1);
    else if(e.key === 'ArrowLeft') setFrame(currentFrame - 1);
    else if(e.key === 'j') nextRoi(1);
    else if(e.key === 'k') nextRoi(-1);
    else if(e.key === 'N') nextEventQueue(1);
    else if(e.key === 'P') nextEventQueue(-1);
    else if(e.key === 'n') nextEvent(1);
    else if(e.key === 'p') nextEvent(-1);
    else if(e.key === '.') nextSuggestion(1);
    else if(e.key === ',') nextSuggestion(-1);
    else if(e.key === '0') resetTraceZoom();
    else if(e.key === 'v') nextActiveFrame(1);
    else if(e.key === 'V') nextActiveFrame(-1);
    else if(e.key === 'a') setRoiState('accept');
    else if(e.key === 'r') setRoiState('reject');
    else if(e.key === 'u') setRoiState('unsure');
    else if(e.key === 'e') setEventState('accept');
    else if(e.key === 'x') setEventState('reject');
    else if(e.key === 'f') viewerScroll.requestFullscreen?.();
    else if(e.key === 'M') setSuggestionStateAndNext('missed');
    else if(e.key === 'G') promoteSuggestionAndNext();
    else if(e.key === 'm') setSuggestionState('missed');
    else if(e.key === 'g') promoteSuggestion();
    else if(e.key === ']') {
      const tasks = guidedTasks();
      setSetting('guidedTaskIndex', Math.min(Math.max(0, tasks.length - 1), Number(setting('guidedTaskIndex') || 0) + 1));
      selectGuidedTask();
    }
    else if(e.key === '[') {
      setSetting('guidedTaskIndex', Math.max(0, Number(setting('guidedTaskIndex') || 0) - 1));
      selectGuidedTask();
    }
  });
  img.onload = () => { resizeOverlay(); drawCrop(); };
  window.onresize = resizeOverlay;
  window.addEventListener('hashchange', routePage);
}

function availabilityBadge(def){
  const value = def?.availability || 'implemented';
  const runnerAvailable = Boolean(def?.runner_available || def?.locally_runnable);
  const label = value === 'implemented' && !runnerAvailable ? 'metadata only' : (value === 'external_import' ? 'external' : value);
  const cls = value === 'implemented' && runnerAvailable ? 'ok' : value === 'planned' || value === 'implemented' ? 'warn' : 'off';
  return `<span class="stageStatus ${cls}">${escapeHtml(label.replace(/_/g, ' '))}</span>`;
}

function pipelinePresetSummary(preset){
  const run = makePresetPipeline(preset.id);
  const realtime = pipelineRealtimeSummary(run);
  const ops = run.pipeline.map(stage => stageDef(stage)).filter(Boolean);
  const chips = ops.slice(0, 7).map(def => `<span>${escapeHtml(def.label)}</span>`).join('');
  return `
    <article class="presetCard">
      <div class="presetHeader">
        <h3>${escapeHtml(preset.label)}</h3>
        <span class="stageStatus ${realtime.warnings.length ? 'warn' : 'ok'}">${realtime.warnings.length ? 'review' : 'ready'}</span>
      </div>
      <p>${escapeHtml(preset.summary)}</p>
      <p class="hint">${escapeHtml(preset.best_for)}</p>
      <div class="archEvidence">${chips}${ops.length > 7 ? `<span>+${ops.length - 7} more</span>` : ''}</div>
      <div class="buttonRow">
        <button type="button" data-load-preset="${escapeHtml(preset.id)}">Use preset</button>
      </div>
    </article>`;
}

function renderArchitecturePresets(){
  const root = document.getElementById('architecturePresetGallery');
  if(!root) return;
  root.innerHTML = ARCHITECTURE_PRESETS.map(pipelinePresetSummary).join('');
  for(const btn of root.querySelectorAll('[data-load-preset]')) btn.onclick = () => {
    pipelineDraft = makePresetPipeline(btn.dataset.loadPreset);
    selectedPipelineStageId = pipelineDraft.pipeline[0]?.id || null;
    const select = document.getElementById('pipelinePresetSelect');
    if(select) select.value = btn.dataset.loadPreset;
    setArchitectureMode('build');
    renderPipelineBuilder();
  };
}

function renderComponentLibrary(){
  const root = document.getElementById('componentLibrary');
  if(!root) return;
  const groups = {};
  for(const def of STAGE_CATALOG) (groups[def.ui_group || def.type || 'stage'] = groups[def.ui_group || def.type || 'stage'] || []).push(def);
  root.innerHTML = Object.entries(groups).map(([group, defs]) => `
    <section class="componentGroup">
      <div class="componentGroupHeader">
        <h3>${escapeHtml(group.replace(/_/g, ' '))}</h3>
        <span class="hint">${defs.length} component${defs.length === 1 ? '' : 's'}</span>
      </div>
      <div class="componentGrid">
        ${defs.map(def => {
          const qc = (def.expected_qc_outputs || []).slice(0, 4).map(item => `<span>${escapeHtml(item)}</span>`).join('');
          const params = Object.keys(def.params || {}).slice(0, 4).map(name => `<span>${escapeHtml(name)}</span>`).join('');
          return `
          <article class="componentCard">
            <div class="componentTitle">
              <h4>${escapeHtml(def.label)}</h4>
              ${availabilityBadge(def)}
            </div>
            <p>${escapeHtml(def.description || 'Pipeline component.')}</p>
            <p class="hint">${escapeHtml(def.why_use_it || '')}</p>
            <div class="stageMeta">${realtimeBadges(def)}</div>
            <div class="artifactFlow"><i>${escapeHtml(def.input || 'input')}</i><strong>-></strong><i>${escapeHtml(def.output || 'output')}</i></div>
            <div class="miniChipRow">${params || '<span>no tunable params</span>'}</div>
            <div class="miniChipRow qcChips">${qc || '<span>QC pending</span>'}</div>
            <button type="button" data-add-component="${escapeHtml(def.op)}">Add to stack</button>
          </article>`;
        }).join('')}
      </div>
    </section>`).join('');
  for(const btn of root.querySelectorAll('[data-add-component]')) btn.onclick = () => {
    pipelineDraft.pipeline.push(makeStage(btn.dataset.addComponent, pipelineDraft.pipeline.length));
    selectedPipelineStageId = pipelineDraft.pipeline[pipelineDraft.pipeline.length - 1].id;
    setArchitectureMode('build');
    renderPipelineBuilder();
  };
}

function slugify(value){
  return String(value || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '') || `pipeline_${Date.now().toString(36)}`;
}

function savedPipelineTemplates(){
  data.architectureRuns = data.architectureRuns || {schema_version: 1, dataset_id: datasetId, runs: []};
  data.architectureRuns.saved_pipelines = Array.isArray(data.architectureRuns.saved_pipelines) ? data.architectureRuns.saved_pipelines : [];
  return data.architectureRuns.saved_pipelines;
}

function pipelineTemplateFromDraft(){
  return {
    id: slugify(pipelineDraft.template_id || pipelineDraft.run_id || pipelineDraft.label),
    label: pipelineDraft.label || 'Untitled architecture',
    description: pipelineDraft.description || '',
    dataset_id: datasetId,
    createdAt: pipelineDraft.createdAt || new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    source: 'architecture_lab',
    pipeline: JSON.parse(JSON.stringify(pipelineDraft.pipeline || [])),
    sweep: pipelineDraft.sweep ? JSON.parse(JSON.stringify(pipelineDraft.sweep)) : undefined,
    artifacts: JSON.parse(JSON.stringify(pipelineDraft.artifacts || {})),
    summary: JSON.parse(JSON.stringify(pipelineDraft.summary || {}))
  };
}

function normalizeTemplateForDraft(template){
  const draft = normalizePipelineDraft({
    schema_version: 1,
    run_id: `planned_${slugify(template.id || template.label)}_${Date.now().toString(36)}`,
    label: template.label || 'Untitled architecture',
    description: template.description || '',
    dataset_id: datasetId,
    pipeline: template.pipeline || [],
    sweep: template.sweep,
    artifacts: template.artifacts || {source_video: data.dataset?.paths?.raw_video || data.video?.name || '', intermediates: []},
    summary: template.summary || {roi_count: 0, event_count: 0, suggestion_count: 0, frame_count: data.video.frames},
    execution: {status: 'planned'}
  });
  draft.template_id = template.id || '';
  return draft;
}

async function persistArchitectureRuns(manifest, successText, fallbackName='architecture_runs.json'){
  data.architectureRuns = manifest;
  if(serverBacked){
    try {
      const res = await fetch('architecture_runs.json', {method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify(manifest, null, 2)});
      if(!res.ok) throw new Error(await res.text());
      setSaveState(successText, 'ok');
      return true;
    } catch (_) {
      downloadJson(fallbackName, manifest);
      setSaveState(`downloaded ${fallbackName}`, 'ok');
      return false;
    }
  }
  downloadJson(fallbackName, manifest);
  setSaveState(`downloaded ${fallbackName}`, 'ok');
  return false;
}

async function saveArchitectureTemplate(){
  const manifest = Object.assign({schema_version: 1, dataset_id: datasetId, runs: []}, data.architectureRuns || {});
  const template = pipelineTemplateFromDraft();
  const templates = savedPipelineTemplates().filter(item => item.id !== template.id);
  const previous = savedPipelineTemplates().find(item => item.id === template.id);
  if(previous?.createdAt) template.createdAt = previous.createdAt;
  manifest.saved_pipelines = [...templates, template].sort((a,b) => String(a.label || a.id).localeCompare(String(b.label || b.id)));
  await persistArchitectureRuns(manifest, `saved architecture ${template.label}`, `${datasetId}_architecture_library.json`);
  pipelineDraft.template_id = template.id;
  renderArchitectureLab();
  renderExperimentLab();
}

async function renameArchitectureTemplate(templateId){
  const template = savedPipelineTemplates().find(item => item.id === templateId);
  if(!template) return;
  const nextLabel = prompt('Architecture name', template.label || template.id);
  if(nextLabel === null) return;
  const nextDescription = prompt('Architecture description', template.description || '');
  if(nextDescription === null) return;
  const manifest = Object.assign({schema_version: 1, dataset_id: datasetId, runs: []}, data.architectureRuns || {});
  manifest.saved_pipelines = savedPipelineTemplates().map(item => item.id === templateId ? Object.assign({}, item, {
    label: nextLabel.trim() || item.label || item.id,
    description: nextDescription.trim(),
    updatedAt: new Date().toISOString()
  }) : item);
  await persistArchitectureRuns(manifest, 'renamed saved architecture', `${datasetId}_architecture_library.json`);
  renderArchitectureLab();
  renderExperimentLab();
}

async function deleteArchitectureTemplate(templateId){
  const template = savedPipelineTemplates().find(item => item.id === templateId);
  if(!template) return;
  if(!confirm(`Delete saved architecture "${template.label || template.id}"? Planned/generated runs will not be deleted.`)) return;
  const manifest = Object.assign({schema_version: 1, dataset_id: datasetId, runs: []}, data.architectureRuns || {});
  manifest.saved_pipelines = savedPipelineTemplates().filter(item => item.id !== templateId);
  if(experimentDraft.baseTemplateId === templateId) experimentDraft.baseTemplateId = '';
  await persistArchitectureRuns(manifest, 'deleted saved architecture', `${datasetId}_architecture_library.json`);
  renderArchitectureLab();
  renderExperimentLab();
}

function loadTemplateIntoBuilder(templateId){
  const template = savedPipelineTemplates().find(item => item.id === templateId);
  if(!template) return;
  pipelineDraft = normalizeTemplateForDraft(template);
  selectedPipelineStageId = pipelineDraft.pipeline?.[0]?.id || null;
  setArchitectureMode('build');
  renderPipelineBuilder();
}

async function useCurrentArchitectureInExperiment(){
  const template = pipelineTemplateFromDraft();
  await saveArchitectureTemplate();
  experimentDraft.baseTemplateId = template.id;
  location.hash = '#experiments';
  renderExperimentLab();
}

function renderArchitectureLibrary(){
  const templates = savedPipelineTemplates();
  if(!templates.length) return `
    <section class="archCard savedArchitectureLibrary">
      <div class="runCardHeader"><h3>Saved Architectures</h3><span class="runStatus">0 saved</span></div>
      <p class="hint">Save a named architecture from Build Pipeline to reuse it in Experiment Lab.</p>
    </section>`;
  const cards = templates.map(template => `
    <article class="savedArchitectureCard">
      <div class="runCardHeader">
        <h3>${escapeHtml(template.label || template.id)}</h3>
        <span class="runStatus">${escapeHtml((template.pipeline || []).length)} stages</span>
      </div>
      <p class="hint">${escapeHtml(template.description || template.id || '')}</p>
      <div class="archEvidence">${(template.pipeline || []).map(stage => `<span>${escapeHtml(stageDef(stage)?.label || stageOp(stage) || stage.id || 'stage')}</span>`).join('')}</div>
      <div class="buttonRow">
        <button type="button" data-load-template="${escapeHtml(template.id)}">Edit</button>
        <button type="button" data-template-experiment="${escapeHtml(template.id)}">Experiment</button>
        <button type="button" data-rename-template="${escapeHtml(template.id)}">Rename</button>
        <button type="button" data-delete-template="${escapeHtml(template.id)}">Delete</button>
      </div>
    </article>`).join('');
  return `
    <section class="archCard savedArchitectureLibrary">
      <div class="runCardHeader"><h3>Saved Architectures</h3><span class="runStatus">${templates.length} saved</span></div>
      <div class="savedArchitectureGrid">${cards}</div>
    </section>`;
}
