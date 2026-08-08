from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ROOT / "neurobench" / "workbench" / "assets" / "src" / "82_annotation_correction.js",
    ROOT / "neurobench" / "workbench" / "assets" / "src" / "83_annotation_correction_review_layout.js",
    ROOT / "neurobench" / "workbench" / "assets" / "src" / "84_annotation_correction_editing.js",
    ROOT / "neurobench" / "workbench" / "assets" / "src" / "85_annotation_correction_relations.js",
    ROOT / "neurobench" / "workbench" / "assets" / "src" / "86_annotation_correction_publication.js",
]
FIXTURE = ROOT / "examples" / "annotation_correction_slice2.example.json"


def node_json(script: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for correction workspace runtime checks")
    result = subprocess.run([node, "-e", script], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def runtime_script(body: str) -> str:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    source = "\n".join(path.read_text(encoding="utf-8") for path in SOURCES)
    return "\n".join(
        [
            "const data = " + json.dumps(fixture) + ";",
            "const datasetId = data.dataset.dataset_id;",
            "const escapeHtml = value => String(value ?? '').replace(/[&<>]/g, '');",
            source,
            body,
        ]
    )


def test_correction_queue_semantics_keep_model_only_candidates_unknown() -> None:
    payload = node_json(
        runtime_script(
            """
const model = annotationCorrectionModel();
const counts = Object.fromEntries(
  ANNOTATION_CORRECTION_QUEUES.map(item => [item[0], annotationCorrectionQueueRows(model, item[0]).length])
);
const unknown = annotationCorrectionQueueRows(model, 'model_unknown')[0];
process.stdout.write(JSON.stringify({counts, status:unknown.status, key:unknown.key}));
"""
        )
    )

    assert payload["counts"] == {
        "matched_expert": 1,
        "missed_expert": 1,
        "model_unknown": 1,
        "all_expert": 2,
        "all_model": 2,
        "recently_edited": 0,
    }
    assert payload["status"] == "unknown"
    assert payload["key"] == "model:M2"




def test_model_only_contract_defaults_to_proposal_queue_and_selected_model_overlay() -> None:
    payload = node_json(
        runtime_script(
            """
data.annotationCorrection.mode = 'model_only';
data.annotationCorrection.expert_annotation_state = 'not_applicable_pending_labels';
data.annotationCorrection.expert_rois = [];
data.annotationCorrection.matches = [];
data.annotationCorrection.model_rois.forEach(item => {
  item.linked_expert_id = '';
  item.status = 'unknown';
});
annotationCorrectionState.queue = 'matched_expert';
annotationCorrectionState.overlayMode = 'selected_pair';
annotationCorrectionState.selectedKey = '';
const model = annotationCorrectionModel();
annotationCorrectionEnsureSelection(model);
const html = annotationCorrectionWorkspaceHtml(model);
process.stdout.write(JSON.stringify({
  modelOnly:annotationCorrectionIsModelOnly(model),
  queue:annotationCorrectionState.queue,
  overlay:annotationCorrectionState.overlayMode,
  selected:annotationCorrectionSelected(model).key,
  proposalLabel:html.includes('Model proposals (2)'),
  pendingLabel:html.includes('expert labels pending'),
  modelHeading:html.includes('Review frozen model proposals without expert labels'),
  matchedQueueAbsent:!html.includes('Matched expert (')
}));
"""
        )
    )

    assert payload == {
        "modelOnly": True,
        "queue": "model_unknown",
        "overlay": "selected_model",
        "selected": "model:M1",
        "proposalLabel": True,
        "pendingLabel": True,
        "modelHeading": True,
        "matchedQueueAbsent": True,
    }


def test_correction_affine_mapping_and_frame_contract_round_trip() -> None:
    payload = node_json(
        runtime_script(
            """
const contract = {
  source_to_view:{kind:'affine', matrix_3x3:[[2,0,-4],[0,0.5,3],[0,0,1]]},
  shape_tyx:[3,4,5]
};
const view = annotationCorrectionSourceToView(contract, [5,8]);
const source = annotationCorrectionViewToSource(contract, view);
const singular = annotationCorrectionViewToSource({source_to_view:{kind:'affine', matrix_3x3:[[1,2,0],[2,4,0],[0,0,1]]}}, [1,1]);
process.stdout.write(JSON.stringify({view, source, singular, shape:annotationCorrectionShape(contract)}));
"""
        )
    )

    assert payload["view"] == [6, 7]
    assert payload["source"] == [5, 8]
    assert payload["singular"] is None
    assert payload["shape"] == [3, 4, 5]


def test_real_data_contract_supports_absolute_frames_lazy_media_and_compact_traces() -> None:
    payload = node_json(
        runtime_script(
            """
framePatternPath = (pattern, frame) => pattern.replace('%04d', String(frame).padStart(4, '0'));
const contract = {
  shape_tyx:[560,340,573],
  frame_mapping:{kind:'identity',offset:1799},
  frame_pattern:'frames/msica/frame_%04d.png'
};
const item = {traces:{msica:{pixel:[1,2,3],roi_mean:[4,5,6]}}};
process.stdout.write(JSON.stringify({
  bounds:annotationCorrectionFrameBounds(contract),
  index:annotationCorrectionFrameIndex(contract, 2003),
  url:annotationCorrectionFrameUrl(contract, 2003),
  pixel:annotationCorrectionProvidedSeries(item, 'msica', 'pixel'),
  mean:annotationCorrectionProvidedSeries(item, 'msica', 'roi_mean')
}));
"""
        )
    )

    assert payload == {
        "bounds": [1800, 2359],
        "index": 203,
        "url": "frames/msica/frame_2003.png",
        "pixel": [1, 2, 3],
        "mean": [4, 5, 6],
    }


def test_correction_selected_overlay_modes_are_distinct_from_all_roi_modes() -> None:
    payload = node_json(
        runtime_script(
            """
const model = annotationCorrectionModel();
annotationCorrectionEnsureSelection(model);
const html = annotationCorrectionWorkspaceHtml(model);
const result = {defaultMode:annotationCorrectionState.overlayMode};
for(const mode of ['selected_pair','selected_expert','selected_model','all_experts','all_models','all_annotations']){
  annotationCorrectionState.overlayMode = mode;
  result[mode] = annotationCorrectionOverlayItems(model).map(item => item.key);
}
result.labels = ['Selected pair','Selected expert only','Selected model only','All expert annotations','All model annotations','All annotations'].every(label => html.includes(label));
process.stdout.write(JSON.stringify(result));
"""
        )
    )

    assert payload == {
        "defaultMode": "selected_pair",
        "selected_pair": ["expert:E1", "model:M1"],
        "selected_expert": ["expert:E1"],
        "selected_model": ["model:M1"],
        "all_experts": ["expert:E1", "expert:E2"],
        "all_models": ["model:M1", "model:M2"],
        "all_annotations": ["expert:E1", "expert:E2", "model:M1", "model:M2"],
        "labels": True,
    }


def test_frame_scrubbing_updates_views_without_rebuilding_workspace() -> None:
    payload = node_json(
        runtime_script(
            """
const model = annotationCorrectionModel();
annotationCorrectionEnsureSelection(model);
const elements = {
  correctionFrameSlider:{value:''}, correctionFrameReadout:{textContent:''}, correctionPlayBtn:{textContent:''},
  correctionRawCanvas:{}, correctionProcessedCanvas:{}, correctionRawCloseupCanvas:{}, correctionProcessedCloseupCanvas:{}, correctionTraceCanvas:{}
};
document = {getElementById:id => elements[id] || null};
let draws = [];
annotationCorrectionDrawFrame = (_canvas,_model,view) => draws.push('frame:'+view);
annotationCorrectionDrawCloseup = (_canvas,_model,view) => draws.push('closeup:'+view);
annotationCorrectionDrawTraces = () => draws.push('traces');
let rebuilds = 0;
renderAnnotationCorrection = () => rebuilds++;
annotationCorrectionSetFrame(3, model, {preload:false});
process.stdout.write(JSON.stringify({frame:annotationCorrectionState.frame,slider:elements.correctionFrameSlider.value,readout:elements.correctionFrameReadout.textContent,draws,rebuilds}));
"""
        )
    )

    assert payload == {
        "frame": 3,
        "slider": "3",
        "readout": "UI 3 / index 2",
        "draws": ["frame:raw", "frame:msica", "closeup:raw", "closeup:msica", "traces"],
        "rebuilds": 0,
    }


def test_trace_interaction_contract_exposes_hover_scrub_and_zoom_controls() -> None:
    source = (ROOT / "neurobench" / "workbench" / "assets" / "src" / "83_annotation_correction_review_layout.js").read_text(encoding="utf-8")
    css = (ROOT / "neurobench" / "workbench" / "assets" / "workbench.css").read_text(encoding="utf-8")

    assert "pointermove" in source and "pointerdown" in source and "wheel" in source and "dblclick" in source
    assert "correctionTraceResetBtn" in source
    renderer = (ROOT / "neurobench" / "workbench" / "assets" / "src" / "82_annotation_correction.js").read_text(encoding="utf-8")
    assert "canvas.clientHeight || 180" in renderer
    assert "max-height: 180px" in css
    assert "@media (max-height: 960px)" in css




def test_trace_readout_uses_current_frame_when_hover_is_inactive() -> None:
    payload = node_json(
        runtime_script(
            """
const model = annotationCorrectionModel();
annotationCorrectionEnsureSelection(model);
annotationCorrectionState.frame = 2;
annotationCorrectionState.traceHoverFrame = null;
const readout = {textContent:''};
document = {getElementById:id => id === 'correctionTraceReadout' ? readout : null};
annotationCorrectionUpdateTraceReadout(model);
process.stdout.write(JSON.stringify({text:readout.textContent}));
"""
        )
    )

    assert payload["text"].startswith("UI frame 2")


def test_correction_workspace_html_exposes_editable_synchronized_evidence() -> None:
    payload = node_json(
        runtime_script(
            """
const model = annotationCorrectionModel();
annotationCorrectionEnsureSelection(model);
const html = annotationCorrectionWorkspaceHtml(model);
process.stdout.write(JSON.stringify({
  selected:annotationCorrectionSelected(model).key,
  hasDraft:html.includes('Draft · token 0'),
  hasEditControls:html.includes('Apply center') && html.includes('Apply radius') && html.includes('Export draft'),
  hasRelationControls:html.includes('Model correspondence') && html.includes('Event intervals') && html.includes('UI frames · inclusive'),
  hasRaw:html.includes('Raw · source coordinates'),
  hasProcessed:html.includes('MSICA · synchronized'),
  hasVerticalStack:html.includes('correctionCanvasStack'),
  hasReviewPanel:html.includes('correctionReviewPanel'),
  hasInspector:html.includes('Selection summary'),
  hasTrace:html.includes('Raw vs processed time series'),
  hasRawCloseup:html.includes('correctionRawCloseupCanvas'),
  hasProcessedCloseup:html.includes('correctionProcessedCloseupCanvas'),
  hasTools:html.includes('Select ROI') && html.includes('Highlight pixel'),
  hasScreenFit:html.includes('correction-fit-screen') && html.includes('Auto-fit screen'),
  hasUnknown:html.includes('Model-only unknown (1)'),
  frame:annotationCorrectionState.frame,
  expertMean:annotationCorrectionRoiSeries(model, 'raw', model.experts[0]),
  modelMean:annotationCorrectionRoiSeries(model, 'raw', model.models[0])
}));
"""
        )
    )

    assert payload == {
        "selected": "expert:E1",
        "hasDraft": True,
        "hasEditControls": True,
        "hasRelationControls": True,
        "hasRaw": True,
        "hasProcessed": True,
        "hasVerticalStack": True,
        "hasReviewPanel": True,
        "hasInspector": True,
        "hasTrace": True,
        "hasRawCloseup": True,
        "hasProcessedCloseup": True,
        "hasTools": True,
        "hasScreenFit": True,
        "hasUnknown": True,
        "frame": 2,
        "expertMean": [2, 9, 3],
        "modelMean": [2, 9, 3],
    }


def test_workbench_routes_and_templates_include_correction_subpage() -> None:
    html = (ROOT / "neurobench" / "workbench" / "assets" / "workbench.html").read_text(encoding="utf-8")
    css = (ROOT / "neurobench" / "workbench" / "assets" / "workbench.css").read_text(encoding="utf-8")
    routing = (ROOT / "neurobench" / "workbench" / "assets" / "src" / "70_dataset_qc.js").read_text(encoding="utf-8")
    boot = (ROOT / "neurobench" / "workbench" / "assets" / "src" / "90_boot.js").read_text(encoding="utf-8")

    assert 'id="reviewCorrectionSubtab"' in html
    assert 'id="annotationCorrectionWorkspace"' in html
    assert ".app.review-correction-mode" in css
    assert "height: 100dvh" in css
    assert ".correctionWorkspaceSplit.correction-fit-screen" in css
    assert "return 'correction'" in routing
    assert "renderAnnotationCorrection()" in boot

def test_highlight_moves_probe_without_replacing_roi_selection() -> None:
    payload = node_json(
        runtime_script(
            """
const model = annotationCorrectionModel();
annotationCorrectionEnsureSelection(model);
renderAnnotationCorrection = () => {};
const canvas = {getBoundingClientRect:() => ({left:0, top:0, width:5, height:4})};
const selectedBefore = annotationCorrectionSelected(model).key;
annotationCorrectionState.toolMode = 'highlight';
annotationCorrectionCanvasClick({currentTarget:canvas, clientX:4, clientY:3}, 'raw', model);
const afterHighlight = {
  selected:annotationCorrectionSelected(model).key,
  probe:annotationCorrectionState.probeSourceXy
};
annotationCorrectionState.toolMode = 'select';
annotationCorrectionCanvasClick({currentTarget:canvas, clientX:3, clientY:2}, 'raw', model);
process.stdout.write(JSON.stringify({
  selectedBefore,
  afterHighlight,
  afterSelect:annotationCorrectionSelected(model).key
}));
"""
        )
    )

    assert payload == {
        "selectedBefore": "expert:E1",
        "afterHighlight": {"selected": "expert:E1", "probe": [4, 3]},
        "afterSelect": "model:M2",
    }


def test_edit_commit_undo_redo_preserve_append_only_history() -> None:
    payload = node_json(
        runtime_script(
            """
renderAnnotationCorrection = () => {};
const model = annotationCorrectionModel();
annotationCorrectionEnsureSelection(model);
const selected = annotationCorrectionSelected(model);
const before = annotationCorrectionProjection(selected);
const after = {...before, source_xy:[3, 2]};
(async () => {
  await annotationCorrectionCommit('move', after, {history:{type:'move', targetId:selected.id, before, after}});
  const afterCommit = annotationCorrectionProjection(annotationCorrectionSelected(annotationCorrectionModel()));
  annotationCorrectionUndo();
  const afterUndo = annotationCorrectionProjection(annotationCorrectionSelected(annotationCorrectionModel()));
  annotationCorrectionRedo();
  const afterRedo = annotationCorrectionProjection(annotationCorrectionSelected(annotationCorrectionModel()));
  process.stdout.write(JSON.stringify({
    afterCommit:afterCommit.source_xy,
    afterUndo:afterUndo.source_xy,
    afterRedo:afterRedo.source_xy,
    token:annotationCorrectionState.revisionToken,
    operationTypes:annotationCorrectionState.draftOperations.map(item => item.operationType),
    undoDepth:annotationCorrectionState.undoStack.length,
    redoDepth:annotationCorrectionState.redoStack.length
  }));
})().catch(error => { throw error; });
"""
        )
    )

    assert payload == {
        "afterCommit": [3, 2],
        "afterUndo": [2, 1],
        "afterRedo": [3, 2],
        "token": 3,
        "operationTypes": ["move", "move", "move"],
        "undoDepth": 1,
        "redoDepth": 0,
    }


def test_link_unlink_and_event_intervals_are_separate_operations() -> None:
    payload = node_json(
        runtime_script(
            """
renderAnnotationCorrection = () => {};
const controls = {
  correctionLinkedModelSelect:{value:'M2'},
  correctionEventStart:{value:'1'},
  correctionEventEnd:{value:'1'}
};
global.document = {
  getElementById:id => controls[id] || null,
  querySelectorAll:() => []
};
let model = annotationCorrectionModel();
annotationCorrectionState.queue = 'all_expert';
annotationCorrectionState.selectedKey = 'expert:E2';
annotationCorrectionApplyLink();
annotationCorrectionAddInterval();
controls.correctionLinkedModelSelect.value = '';
annotationCorrectionApplyLink();
model = annotationCorrectionModel();
const expert = model.experts.find(item => item.id === 'E2');
process.stdout.write(JSON.stringify({
  linkedModelId:expert.linkedModelId,
  eventIntervals:expert.eventIntervals,
  operationTypes:annotationCorrectionState.draftOperations.map(item => item.operationType),
  token:annotationCorrectionState.revisionToken
}));
"""
        )
    )

    assert payload == {
        "linkedModelId": "",
        "eventIntervals": [[1, 1], [2, 2]],
        "operationTypes": ["link", "edit-event-interval", "unlink"],
        "token": 3,
    }


def test_promote_keeps_model_frozen_and_undo_redo_compensate() -> None:
    payload = node_json(
        runtime_script(
            """
renderAnnotationCorrection = () => {};
const controls = {
  correctionPromotionId:{value:'E_promoted_M2'},
  correctionPromotionRadius:{value:'1.25'}
};
global.document = {
  getElementById:id => controls[id] || null,
  querySelectorAll:() => []
};
annotationCorrectionState.queue = 'all_model';
annotationCorrectionState.selectedKey = 'model:M2';
(async () => {
  await annotationCorrectionCommitPromotion();
  let model = annotationCorrectionModel();
  const promoted = model.experts.find(item => item.id === 'E_promoted_M2');
  const frozenModel = model.models.find(item => item.id === 'M2');
  const afterPromotion = {
    expertSource:promoted.sourceXy,
    expertRadius:promoted.geometry.radius_px,
    modelSource:frozenModel.sourceXy,
    linkedExpertId:frozenModel.linkedExpertId
  };
  annotationCorrectionUndo();
  const afterUndo = annotationCorrectionProjection(
    annotationCorrectionModel().experts.find(item => item.id === 'E_promoted_M2')
  ).deleted;
  annotationCorrectionRedo();
  const afterRedo = annotationCorrectionProjection(
    annotationCorrectionModel().experts.find(item => item.id === 'E_promoted_M2')
  ).deleted;
  process.stdout.write(JSON.stringify({
    afterPromotion,
    afterUndo,
    afterRedo,
    operationTypes:annotationCorrectionState.draftOperations.map(item => item.operationType),
    token:annotationCorrectionState.revisionToken
  }));
})().catch(error => { throw error; });
"""
        )
    )

    assert payload == {
        "afterPromotion": {
            "expertSource": [3, 2],
            "expertRadius": 1.25,
            "modelSource": [3, 2],
            "linkedExpertId": "E_promoted_M2",
        },
        "afterUndo": True,
        "afterRedo": False,
        "operationTypes": ["promote", "tombstone", "restore"],
        "token": 3,
    }


def test_change_review_summarizes_operations_and_adopts_published_child() -> None:
    payload = node_json(
        runtime_script(
            """
global.productRequest = async () => ({});
const model = annotationCorrectionModel();
annotationCorrectionState.apiAvailable = true;
const selected = model.experts[0];
const before = annotationCorrectionProjection(selected);
const operation = annotationCorrectionOperation(
  model,
  selected,
  'move',
  before,
  {...before, source_xy:[3, 2]}
);
annotationCorrectionState.draftOperations = [operation];
const html = annotationCorrectionChangeReviewHtml(model);
const snapshot = {
  revision:{...model.revision,revisionId:'ann_published',parentRevisionId:model.revision.revisionId,state:'published',revisionToken:1,operationCount:1},
  annotations:{rois:{E1:{...before,source_xy:[3,2]}}},
  operations:[operation]
};
annotationCorrectionAdoptRevisionSnapshot(snapshot, 'Published immutable child');
const adopted = annotationCorrectionModel();
process.stdout.write(JSON.stringify({
  hasReview:html.includes('Review changes'),
  hasMove:html.includes('Moved center'),
  hasPublish:html.includes('Publish immutable child'),
  revisionId:adopted.revision.revisionId,
  state:adopted.revision.state,
  readOnly:adopted.readOnly,
  token:annotationCorrectionState.revisionToken,
  status:annotationCorrectionState.saveStatus
}));
"""
        )
    )

    assert payload == {
        "hasReview": True,
        "hasMove": True,
        "hasPublish": True,
        "revisionId": "ann_published",
        "state": "published",
        "readOnly": True,
        "token": 1,
        "status": "Published immutable child",
    }
