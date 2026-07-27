from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


ASSETS = Path("neurobench/workbench/assets")


def _node_json(script: str):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for Workbench behavior checks")
    result = subprocess.run(
        [node, "-e", script],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _source_slice(path: Path, start: str, end: str) -> str:
    source = path.read_text(encoding="utf-8")
    first = source.index(start)
    last = source.index(end, first)
    return source[first:last]


def test_task_decision_and_save_next_are_separate_actions():
    product = ASSETS / "src" / "15_product_shell.js"
    source = _source_slice(
        product,
        "const ANNOTATION_TASK_DEFS",
        "\nfunction openToolPanel",
    )
    script = "\n".join(
        [
            "let calls = [];",
            "const currentSettings = {annotationTask:'neuron_validation', queue:'annotationBatch', eventQueue:'unlabeled'};",
            "const setting = key => currentSettings[key];",
            "const setSetting = (key, value) => { currentSettings[key] = value; };",
            "const selectedRoi = () => ({id:'roi-1'});",
            "const roiAnn = () => ({state:''});",
            "const eventAnn = () => ({state:''});",
            "let selectedEventFrame = 7;",
            "const setRoiState = value => calls.push(['roi-state', value]);",
            "const setEventState = value => calls.push(['event-state', value]);",
            "const queueSave = () => calls.push(['save']);",
            "const nextRoi = value => calls.push(['next-roi', value]);",
            "const nextEventQueue = value => calls.push(['next-event', value]);",
            "const renderAll = () => calls.push(['render']);",
            source,
            "const neuron = ANNOTATION_TASK_DEFS.neuron_validation;",
            "applyTaskDecision(neuron, 'accept');",
            "const decisionCalls = calls.slice();",
            "calls = [];",
            "saveAndAdvanceTask(neuron);",
            "const saveCalls = calls.slice();",
            "calls = [];",
            "applyTaskDecision(ANNOTATION_TASK_DEFS.event_validation, 'reject');",
            "const eventCalls = calls.slice();",
            "process.stdout.write(JSON.stringify({",
            "  taskIds:Object.keys(ANNOTATION_TASK_DEFS),",
            "  neuronLabels:neuron.decisions.map(item => item[1]),",
            "  eventLabels:ANNOTATION_TASK_DEFS.event_validation.decisions.map(item => item[1]),",
            "  artifactLabels:ANNOTATION_TASK_DEFS.artifact_resolution.decisions.map(item => item[1]),",
            "  signalDecisionCount:ANNOTATION_TASK_DEFS.signal_background.decisions.length,",
            "  decisionCalls, saveCalls, eventCalls,",
            "  normalQueues:normalQueueChoices(neuron).map(item => item[1])",
            "}));",
        ]
    )

    payload = _node_json(script)

    assert payload["taskIds"] == [
        "neuron_validation",
        "missed_neuron_search",
        "event_validation",
        "artifact_resolution",
        "exhaustive_tile",
        "signal_background",
    ]
    assert payload["neuronLabels"] == ["Neuron", "Not neuron", "Unsure"]
    assert payload["eventLabels"] == ["Event", "No event", "Unsure"]
    assert payload["artifactLabels"] == ["Artifact", "Clean", "Unsure"]
    assert payload["signalDecisionCount"] == 0
    assert payload["decisionCalls"] == [["roi-state", "accept"]]
    assert payload["saveCalls"] == [["save"], ["next-roi", 1]]
    assert payload["eventCalls"] == [["event-state", "reject"]]
    assert payload["normalQueues"] == ["Next", "Needs attention", "Reviewed", "All"]


def test_dataset_card_actions_are_dataset_qualified_and_require_ready_annotator():
    product = ASSETS / "src" / "15_product_shell.js"
    source = _source_slice(product, "function productUrl", "\nasync function waitForDatasetJob")
    script = "\n".join(
        [
            "const escapeHtml = value => String(value ?? '');",
            "const generationHeaders = () => ({'Content-Type':'application/json'});",
            "const fetch = () => { throw new Error('not called'); };",
            "const document = {querySelectorAll:() => []};",
            "const prompt = () => null;",
            "const localStorage = {setItem(){}, removeItem(){}};",
            "let generationEnvironment = null;",
            "let generationOwnerToken = '';",
            "const ownerTokenKey = 'owner';",
            "const datasetId = 'alpha';",
            source,
            "const dataset = {",
            "  dataset_id:'beta', name:'Beta', video:{frames:10,width:8,height:6},",
            "  readiness:{review_ready:false},",
            "  capability_states:{annotate:'import_only',results:'planned',raw_video:'ready'},",
            "  lifecycle:{state:'metadata_needed'},",
            "  links:{",
            "    annotate:'/_datasets/beta/#annotate',",
            "    api_base:'api/datasets/beta',",
            "    import_action_template:'api/datasets/beta/imports/{import_id}/{action}',",
            "    neurev:'api/datasets/beta/neurev'",
            "  },",
            "  imports:[",
            "    {import_id:'imp-beta',kind:'video',state:'metadata_needed',has_qc:false,is_primary_video:false},",
            "    {import_id:'imp-json',kind:'neurev_json',payload_kind:'annotations',state:'qc_ready'}",
            "  ]",
            "};",
            "process.stdout.write(JSON.stringify({",
            "  actionUrl:datasetImportActionUrl(dataset, 'imp-beta', 'promote'),",
            "  neurevUrl:datasetNeuRevActionUrl(dataset, 'preview'),",
            "  card:datasetCard(dataset)",
            "}));",
        ]
    )

    payload = _node_json(script)

    assert payload["actionUrl"] == "/api/datasets/beta/imports/imp-beta/promote"
    assert payload["neurevUrl"] == "/api/datasets/beta/neurev/preview"
    assert "Use as primary video" in payload["card"]
    assert "Preview NeuRev JSON" in payload["card"]
    assert "annotations · confirmation required" in payload["card"]
    assert 'data-annotate-url="/_datasets/beta/#annotate" disabled' in payload["card"]
    assert "api/datasets/alpha" not in payload["actionUrl"]


def test_default_and_annotation_routes_reset_product_state():
    boot_path = ASSETS / "src" / "90_boot.js"
    source = _source_slice(boot_path, "const PAGE_CHROME", "\nasync function boot")
    script = "\n".join(
        [
            "class Classes {",
            "  constructor(initial=[]) { this.values = new Set(initial); }",
            "  add(...items) { items.forEach(item => this.values.add(item)); }",
            "  remove(...items) { items.forEach(item => this.values.delete(item)); }",
            "  toggle(item, force) { const on = force === undefined ? !this.values.has(item) : Boolean(force); on ? this.values.add(item) : this.values.delete(item); return on; }",
            "  contains(item) { return this.values.has(item); }",
            "}",
            "const pageIds = ['homePage','datasetsPage','researchPage','architecturePage','experimentsPage','metricsPage','qcPage','reportPage'];",
            "const pages = Object.fromEntries(pageIds.map(id => [id, {id, classList:new Classes(['hidden'])}]));",
            "const navs = ['datasets','annotate','results','research'].map(page => ({dataset:{navPage:page}, classList:new Classes()}));",
            "const contexts = {};",
            "const document = {",
            "  getElementById:id => pages[id] || null,",
            "  querySelectorAll:selector => selector === '[data-nav-page]' ? navs : [],",
            "  querySelector:selector => contexts[selector] ||= {textContent:''}",
            "};",
            "const appRoot = {classList:new Classes(['product-mode','normal-annotation-mode','task-neuron_validation'])};",
            "const ANNOTATION_TASK_DEFS = {neuron_validation:{}};",
            "const datasetId = 'demo';",
            "const escapeHtml = value => String(value);",
            "const location = {hash:''};",
            "const window = {requestAnimationFrame:callback => callback(), scrollTo(){}};",
            "const calls = [];",
            "const updateReviewSubnav = value => calls.push(['review-subnav', value]);",
            "const updateDataSubnav = value => calls.push(['data-subnav', value]);",
            "const reviewSubPageFromHash = () => 'inspect';",
            "const dataSubPageFromHash = () => 'inspect';",
            "const researchToolsEnabled = () => false;",
            "const renderDatasetsPage = () => calls.push(['render','datasets']);",
            "const renderResultsPage = () => calls.push(['render','results']);",
            "const renderResearchToolsPage = () => calls.push(['render','research']);",
            "const renderAnnotationTaskShell = () => calls.push(['render','task-shell']);",
            "const renderNextBestActions = () => {};",
            "const resizeOverlay = () => {};",
            "const renderReviewStencil = () => {};",
            "const renderReviewOverlap = () => {};",
            "const renderReviewTriage = () => {};",
            "const renderWorkflowHome = () => {};",
            "const renderArchitectureLab = () => {};",
            "const renderExperimentLab = () => {};",
            "const renderMetricsAudit = () => {};",
            "const renderDataCompare = () => {};",
            "const renderDatasetQc = () => {};",
            "const renderReviewReport = () => {};",
            source,
            "routePage();",
            "const initial = {",
            "  datasetsVisible:!pages.datasetsPage.classList.contains('hidden'),",
            "  product:appRoot.classList.contains('product-mode'),",
            "  active:navs.find(item => item.classList.contains('active'))?.dataset.navPage",
            "};",
            "location.hash = '#annotate'; routePage();",
            "const annotate = {",
            "  datasetsHidden:pages.datasetsPage.classList.contains('hidden'),",
            "  product:appRoot.classList.contains('product-mode'),",
            "  normal:appRoot.classList.contains('normal-annotation-mode'),",
            "  active:navs.find(item => item.classList.contains('active'))?.dataset.navPage",
            "};",
            "location.hash = '#results'; routePage();",
            "const results = {",
            "  reportVisible:!pages.reportPage.classList.contains('hidden'),",
            "  normal:appRoot.classList.contains('normal-annotation-mode'),",
            "  active:navs.find(item => item.classList.contains('active'))?.dataset.navPage",
            "};",
            "process.stdout.write(JSON.stringify({initial, annotate, results}));",
        ]
    )

    payload = _node_json(script)

    assert payload["initial"] == {
        "datasetsVisible": True,
        "product": True,
        "active": "datasets",
    }
    assert payload["annotate"] == {
        "datasetsHidden": True,
        "product": False,
        "normal": True,
        "active": "annotate",
    }
    assert payload["results"] == {
        "reportVisible": True,
        "normal": False,
        "active": "results",
    }


def test_browser_run_bucket_migration_preserves_unknown_scientific_fields():
    state = ASSETS / "src" / "10_state_persistence.js"
    source = _source_slice(state, "function migrateRunBucket", "\nfunction migrateRoiAnn")
    script = "\n".join(
        [
            "const migrateRoiAnn = value => value;",
            "const migrateEventAnn = value => value;",
            "const migrateSuggestionAnn = value => value;",
            "const migrateSplitMergeDecision = value => value;",
            source,
            "const migrated = migrateRunBucket({",
            "  rois:{r1:{state:'accept'}}, events:{}, suggestions:{},",
            "  scientific_checkpoint:{fold:3,held_out:true},",
            "  provenance:{source:'historical'}",
            "});",
            "process.stdout.write(JSON.stringify(migrated));",
        ]
    )

    payload = _node_json(script)

    assert payload["scientific_checkpoint"] == {"fold": 3, "held_out": True}
    assert payload["provenance"] == {"source": "historical"}
    assert payload["rois"]["r1"]["state"] == "accept"


def test_capture_active_run_preserves_unknown_scientific_fields():
    state = ASSETS / "src" / "10_state_persistence.js"
    migration_source = _source_slice(
        state,
        "function migrateRunBucket",
        "\nfunction migrateRoiAnn",
    )
    capture_source = _source_slice(
        state,
        "function runAnnotationSnapshot",
        "\nfunction materializeRunAnnotations",
    )
    script = "\n".join(
        [
            "const migrateRoiAnn = value => Object.assign({}, value);",
            "const migrateEventAnn = value => Object.assign({}, value);",
            "const migrateSuggestionAnn = value => Object.assign({}, value);",
            "const migrateSplitMergeDecision = value => Object.assign({}, value);",
            migration_source,
            "let annotations = {",
            "  rois:{current:{state:'accept'}}, events:{}, suggestions:{},",
            "  promotedRois:{}, virtualRois:{}, splitMergeDecisions:{},",
            "  runs:{alpha:{",
            "    rois:{stale:{state:'reject'}}, events:{}, suggestions:{},",
            "    promotedRois:{}, virtualRois:{}, splitMergeDecisions:{},",
            "    scientific_checkpoint:{fold:3,held_out:true},",
            "    provenance:{source:'historical'}, custom_evidence:['a','b']",
            "  }}",
            "};",
            "const activeRunId = () => 'alpha';",
            capture_source,
            "captureActiveRunAnnotations();",
            "process.stdout.write(JSON.stringify(annotations.runs.alpha));",
        ]
    )

    payload = _node_json(script)

    assert payload["scientific_checkpoint"] == {"fold": 3, "held_out": True}
    assert payload["provenance"] == {"source": "historical"}
    assert payload["custom_evidence"] == ["a", "b"]
    assert payload["rois"] == {"current": {"state": "accept"}}


def test_review_shortcuts_are_route_and_control_scoped():
    controls = ASSETS / "src" / "25_review_controls.js"
    source = _source_slice(
        controls,
        "function reviewShortcutRouteActive",
        "\nfunction initControls",
    )
    script = "\n".join(
        [
            "let calls = [];",
            "const location = {hash:'#datasets'};",
            "const currentFrame = 4;",
            "const viewerScroll = {requestFullscreen:() => calls.push(['fullscreen'])};",
            "const toggleShortcutHelp = value => calls.push(['help', value === undefined ? 'toggle' : value]);",
            "const undoLastAnnotationChange = () => calls.push(['undo']);",
            "const togglePlay = () => calls.push(['play']);",
            "const setFrame = value => calls.push(['frame', value]);",
            "const nextRoi = value => calls.push(['next-roi', value]);",
            "const nextEventQueue = value => calls.push(['next-event-queue', value]);",
            "const nextEvent = value => calls.push(['next-event', value]);",
            "const nextSuggestion = value => calls.push(['next-suggestion', value]);",
            "const resetTraceZoom = () => calls.push(['trace-reset']);",
            "const nextActiveFrame = value => calls.push(['active-frame', value]);",
            "const setRoiState = value => calls.push(['roi-state', value]);",
            "const setEventState = value => calls.push(['event-state', value]);",
            "const setSuggestionStateAndNext = value => calls.push(['suggestion-next', value]);",
            "const promoteSuggestionAndNext = () => calls.push(['promote-next']);",
            "const setSuggestionState = value => calls.push(['suggestion-state', value]);",
            "const promoteSuggestion = () => calls.push(['promote']);",
            "const guidedTasks = () => [];",
            "const setSetting = () => {};",
            "const setting = () => 0;",
            "const selectGuidedTask = () => {};",
            source,
            "const keyEvent = (key, tagName='DIV') => ({",
            "  key, code:key === ' ' ? 'Space' : '', ctrlKey:false, metaKey:false,",
            "  target:{tagName, isContentEditable:false, closest:() => null},",
            "  preventDefault:() => calls.push(['prevent'])",
            "});",
            "const snapshot = () => { const out = calls.slice(); calls = []; return out; };",
            "handleReviewShortcutKeydown(keyEvent('a')); const datasets = snapshot();",
            "location.hash = '#results'; handleReviewShortcutKeydown(keyEvent('e')); const results = snapshot();",
            "location.hash = '#candidate-overlay'; handleReviewShortcutKeydown(keyEvent('a')); const utility = snapshot();",
            "location.hash = '#annotate'; handleReviewShortcutKeydown(keyEvent('a', 'SELECT')); const selectControl = snapshot();",
            "handleReviewShortcutKeydown(keyEvent('a')); const annotate = snapshot();",
            "location.hash = '#review'; handleReviewShortcutKeydown(keyEvent('e')); const review = snapshot();",
            "location.hash = '#datasets'; handleReviewShortcutKeydown(keyEvent('Escape')); const escape = snapshot();",
            "handleReviewShortcutKeydown(keyEvent('?')); const offRouteHelp = snapshot();",
            "location.hash = '#annotate'; handleReviewShortcutKeydown(keyEvent('?')); const reviewHelp = snapshot();",
            "process.stdout.write(JSON.stringify({datasets,results,utility,selectControl,annotate,review,escape,offRouteHelp,reviewHelp}));",
        ]
    )

    payload = _node_json(script)

    assert payload["datasets"] == []
    assert payload["results"] == []
    assert payload["utility"] == []
    assert payload["selectControl"] == []
    assert payload["annotate"] == [["roi-state", "accept"]]
    assert payload["review"] == [["event-state", "accept"]]
    assert payload["escape"] == [["help", False]]
    assert payload["offRouteHelp"] == []
    assert payload["reviewHelp"] == [["prevent"], ["help", "toggle"]]


def test_annotation_tool_controller_is_atomic_exclusive_and_single_view():
    core = ASSETS / "src" / "20_review_core.js"
    source = _source_slice(
        core,
        "const ANNOTATION_TOOL_ACTIONS",
        "\nfunction refreshReviewAfterDataChange",
    )
    script = "\n".join(
        [
            "let calls = [];",
            "let annotations = {settings:{manualRoiMode:'lasso',roiEditMode:'brush_add',cfarMaskTool:'flood_add',cfarMaskTarget:'background',reviewSideBySide:true}};",
            "let manualRoiState = {drawing:true,start:{x:1,y:2},points:[{x:1,y:2}],preview:{},suppressClick:true};",
            "let roiEditState = {drawing:true,editedId:'r1'};",
            "let cfarMaskState = {drawing:true,roiId:'r1',pointerId:4};",
            "const setting = key => annotations.settings[key];",
            "const applyReviewViewerMode = () => calls.push('viewer');",
            "const applySettingsToControls = () => calls.push('controls');",
            "const syncCfarMaskControls = () => calls.push('cfar-controls');",
            "const openToolPanel = id => calls.push(`open:${id}`);",
            "const queueSave = () => calls.push('save');",
            "const renderAll = () => calls.push('render');",
            source,
            "const snapshot = () => ({settings:{...annotations.settings},manual:{...manualRoiState},roi:{...roiEditState},cfar:{...cfarMaskState},calls:[...calls]});",
            "activateAnnotationTool('roi-lasso'); const lasso = snapshot(); calls = [];",
            "annotations.settings.reviewSideBySide = true; manualRoiState.drawing = true; roiEditState.drawing = true; cfarMaskState.drawing = true;",
            "activateAnnotationTool('foreground-brush'); const foreground = snapshot(); calls = [];",
            "annotations.settings.reviewSideBySide = true; manualRoiState.drawing = true; roiEditState.drawing = true; cfarMaskState.drawing = true;",
            "activateAnnotationTool('roi-add'); const roiAdd = snapshot();",
            "process.stdout.write(JSON.stringify({lasso,foreground,roiAdd}));",
        ]
    )

    payload = _node_json(script)

    lasso_expected = {
        "manualRoiMode": "lasso",
        "roiEditMode": "off",
        "cfarMaskTool": "off",
        "reviewSideBySide": False,
    }
    foreground_expected = {
        "manualRoiMode": "select",
        "roiEditMode": "off",
        "cfarMaskTool": "brush_add",
        "cfarMaskTarget": "foreground",
        "reviewSideBySide": False,
    }
    roi_add_expected = {
        "manualRoiMode": "select",
        "roiEditMode": "brush_add",
        "cfarMaskTool": "off",
        "reviewSideBySide": False,
    }
    assert all(payload["lasso"]["settings"].get(key) == value for key, value in lasso_expected.items())
    assert all(payload["foreground"]["settings"].get(key) == value for key, value in foreground_expected.items())
    assert all(payload["roiAdd"]["settings"].get(key) == value for key, value in roi_add_expected.items())
    for key in ("lasso", "foreground", "roiAdd"):
        state = payload[key]
        assert state["manual"]["drawing"] is False
        assert state["roi"]["drawing"] is False
        assert state["cfar"]["drawing"] is False
        assert state["calls"].count("save") == 1
        assert state["calls"].count("render") == 1
    assert "open:roiAnnotationPanel" in payload["lasso"]["calls"]
    assert "open:cfarMaskAnnotationPanel" in payload["foreground"]["calls"]


def test_task_disclosures_preserve_user_open_state_until_task_changes():
    product = ASSETS / "src" / "15_product_shell.js"
    source = _source_slice(
        product,
        "function setTaskContextVisibility",
        "\nfunction renderAnnotationTaskShell",
    )
    script = "\n".join(
        [
            "class Classes { remove(){} toggle(){} }",
            "const makePanel = () => ({open:false,classList:new Classes()});",
            "const panels = {roiAnnotationPanel:makePanel(),cfarMaskAnnotationPanel:makePanel(),roiReviewRail:makePanel(),eventReviewRail:makePanel(),reviewQueueRail:makePanel()};",
            "const document = {getElementById:id => panels[id] || null};",
            "const appRoot = {classList:new Classes()};",
            "const ANNOTATION_TASK_DEFS = {neuron_validation:{},missed_neuron_search:{},signal_background:{}};",
            "const researchToolsEnabled = () => false;",
            "let annotationTaskDisclosureTask = null;",
            source,
            "setTaskContextVisibility('missed_neuron_search', {});",
            "const defaults = {roi:panels.roiAnnotationPanel.open,events:panels.eventReviewRail.open};",
            "panels.roiAnnotationPanel.open = false; panels.eventReviewRail.open = false;",
            "setTaskContextVisibility('missed_neuron_search', {});",
            "const preserved = {roi:panels.roiAnnotationPanel.open,events:panels.eventReviewRail.open};",
            "setTaskContextVisibility('signal_background', {});",
            "const changed = {roi:panels.roiAnnotationPanel.open,cfar:panels.cfarMaskAnnotationPanel.open,events:panels.eventReviewRail.open};",
            "process.stdout.write(JSON.stringify({defaults,preserved,changed}));",
        ]
    )

    payload = _node_json(script)

    assert payload["defaults"] == {"roi": True, "events": True}
    assert payload["preserved"] == {"roi": False, "events": False}
    assert payload["changed"] == {"roi": True, "cfar": True, "events": False}


def test_task_shell_render_is_pure_and_exhaustive_copy_matches_implemented_scope():
    product = ASSETS / "src" / "15_product_shell.js"
    source = _source_slice(product, "const ANNOTATION_TASK_DEFS", "\nfunction productUrl")
    script = "\n".join(
        [
            "class Classes { remove(){} toggle(){} }",
            "const listeners = {onchange:null,onclick:null,addEventListener(){}};",
            "const root = {innerHTML:'',classList:new Classes(),querySelector:() => ({...listeners}),querySelectorAll:() => []};",
            "const document = {getElementById:id => id === 'annotationTaskShell' ? root : null};",
            "const appRoot = {classList:new Classes()};",
            "const location = {hash:'#annotate'};",
            "const settings = {annotationTask:'neuron_validation',queue:'obsolete_queue',eventQueue:'obsolete_event_queue',researchToolsEnabled:false};",
            "const setting = key => settings[key];",
            "const writes = [];",
            "const setSetting = (key,value) => writes.push([key,value]);",
            "const selectedRoi = () => null;",
            "const roiAnn = () => ({});",
            "const eventAnn = () => ({});",
            "let selectedEventFrame = null;",
            "const escapeHtml = value => String(value ?? '');",
            source,
            "renderAnnotationTaskShell();",
            "process.stdout.write(JSON.stringify({writes,markup:root.innerHTML,description:ANNOTATION_TASK_DEFS.exhaustive_tile.description}));",
        ]
    )

    payload = _node_json(script)

    assert payload["writes"] == []
    assert 'value="annotationBatch" selected' in payload["markup"]
    assert payload["description"] == "Review every candidate in the current queue. Unreviewed space remains unknown."
    assert "bounded region" not in payload["description"]


def test_save_next_starts_at_first_remaining_roi_after_queue_shrinks():
    core = ASSETS / "src" / "20_review_core.js"
    source = _source_slice(core, "function nextRoi(delta)", "\nfunction nextEvent(delta)")
    script = "\n".join(
        [
            "let selectedId = 'removed-roi';",
            "const visibleRois = () => [{id:'first-remaining'},{id:'second-remaining'}];",
            "const selected = [];",
            "const selectRoi = id => { selected.push(id); selectedId = id; };",
            "const setSaveState = () => {};",
            source,
            "nextRoi(1); selectedId = 'removed-roi'; nextRoi(-1);",
            "process.stdout.write(JSON.stringify(selected));",
        ]
    )

    assert _node_json(script) == ["first-remaining", "second-remaining"]


def test_legacy_workflow_adapter_updates_canonical_task_only_when_mapped():
    state = ASSETS / "src" / "10_state_persistence.js"
    core = ASSETS / "src" / "20_review_core.js"
    mapping_source = _source_slice(
        state,
        "const LEGACY_WORKFLOW_ANNOTATION_TASK",
        "\nlet currentFrame",
    )
    workflow_source = _source_slice(
        core,
        "function applyReviewWorkflowPreset",
        "\nfunction toggleShortcutHelp",
    )
    script = "\n".join(
        [
            mapping_source,
            "const REVIEW_WORKFLOW_PRESETS = {clean_artifacts:{label:'Clean',queue:'artifactLike',roiFocusMode:'all'},tune_parameters:{label:'Tune',queue:'annotationBatch',roiFocusMode:'all'}};",
            "let annotations = {settings:{annotationTask:'neuron_validation'}};",
            "const setSetting = (key,value) => { annotations.settings[key] = value; };",
            "const applyOverlayPreset = () => {};",
            "const setCheckbox = () => {};",
            "const document = {getElementById:() => null};",
            "const visibleRois = () => [];",
            "const selectedRoi = () => null;",
            "let selectedId = null;",
            "const recordAction = () => {};",
            "const applySettingsToControls = () => {};",
            "const renderAll = () => {};",
            "const setSaveState = () => {};",
            "const toolResets = [];",
            "const setAnnotationToolModes = value => toolResets.push(value);",
            workflow_source,
            "applyReviewWorkflowPreset('clean_artifacts'); const mapped = {...annotations.settings};",
            "applyReviewWorkflowPreset('tune_parameters'); const unmapped = {...annotations.settings};",
            "process.stdout.write(JSON.stringify({mapped,unmapped,toolResets}));",
        ]
    )

    payload = _node_json(script)

    assert payload["mapped"]["annotationTask"] == "artifact_resolution"
    assert payload["unmapped"]["annotationTask"] == "artifact_resolution"
    assert payload["unmapped"]["reviewWorkflowPreset"] == "tune_parameters"
    assert len(payload["toolResets"]) == 1


def test_overlay_visibility_survives_preset_persist_reload_and_apply():
    core = ASSETS / "src" / "20_review_core.js"
    visibility_source = _source_slice(
        core,
        "function applyOverlayVisibilityToControls",
        "\nfunction applySettingsToControls",
    )
    preset_source = _source_slice(
        core,
        "function applyOverlayPreset",
        "\nfunction setCheckbox",
    )
    script = "\n".join(
        [
            "let settings = {showRois:false,showEvents:false};",
            "const setting = key => settings[key];",
            "const controls = {showRois:{checked:true},showEvents:{checked:true}};",
            "const document = {getElementById:id => controls[id] || null};",
            "const setSetting = (key,value) => { settings[key] = value; };",
            "const OVERLAY_PRESETS = {validate:{showRois:true,showEvents:true}};",
            "const renderAll = () => {};",
            visibility_source,
            "const applySettingsToControls = () => applyOverlayVisibilityToControls();",
            preset_source,
            "applyOverlayVisibilityToControls(); const before = {rois:controls.showRois.checked,events:controls.showEvents.checked};",
            "applyOverlayPreset('validate'); const persisted = JSON.stringify(settings);",
            "settings = JSON.parse(persisted); controls.showRois.checked = false; controls.showEvents.checked = false; applyOverlayVisibilityToControls();",
            "process.stdout.write(JSON.stringify({before,after:{rois:controls.showRois.checked,events:controls.showEvents.checked},settings}));",
        ]
    )

    payload = _node_json(script)

    assert payload["before"] == {"rois": False, "events": False}
    assert payload["after"] == {"rois": True, "events": True}
    assert payload["settings"]["showRois"] is True
    assert payload["settings"]["showEvents"] is True


def test_dataset_preview_actions_require_exact_qc_ready_state():
    product = ASSETS / "src" / "15_product_shell.js"
    source = _source_slice(product, "function productUrl", "\nasync function waitForDatasetJob")
    script = "\n".join(
        [
            "const escapeHtml = value => String(value ?? '');",
            "const generationHeaders = () => ({'Content-Type':'application/json'});",
            "const fetch = () => { throw new Error('not called'); };",
            "const document = {querySelectorAll:() => []};",
            "const prompt = () => null;",
            "const localStorage = {setItem(){},removeItem(){}};",
            "let generationEnvironment = null; let generationOwnerToken = '';",
            "const ownerTokenKey = 'owner'; const datasetId = 'alpha';",
            source,
            "const cardFor = (labelState,jsonState) => datasetCard({dataset_id:'beta',name:'Beta',video:{frames:2,width:3,height:4},readiness:{},capability_states:{annotate:'import_only'},imports:[{import_id:'labels',kind:'label_table',state:labelState},{import_id:'json',kind:'neurev_json',payload_kind:'annotations',state:jsonState}]});",
            "const result = {};",
            "for(const state of ['qc_ready','processing','failed','complete']) { const card = cardFor(state,state); result[state] = {label:card.includes('data-label-action=\"preview\"'),json:card.includes('data-neurev-action=\"preview\"')}; }",
            "process.stdout.write(JSON.stringify(result));",
        ]
    )

    payload = _node_json(script)

    assert payload["qc_ready"] == {"label": True, "json": True}
    for state in ("processing", "failed", "complete"):
        assert payload[state] == {"label": False, "json": False}


def test_research_tools_page_links_to_preserved_review_utilities():
    product = ASSETS / "src" / "15_product_shell.js"
    text = product.read_text(encoding="utf-8")
    source = text[text.index("function renderResearchToolsPage") :]
    script = "\n".join(
        [
            "const root = {innerHTML:''}; const toggle = {};",
            "const document = {getElementById:id => id === 'researchPageBody' ? root : toggle};",
            "const researchToolsEnabled = () => false;",
            "const setSetting = () => {};",
            "const appRoot = {classList:{toggle(){}}};",
            "const renderAnnotationTaskShell = () => {};",
            source,
            "renderResearchToolsPage(); process.stdout.write(JSON.stringify(root.innerHTML));",
        ]
    )

    markup = _node_json(script)

    assert 'href="#review-stencil"' in markup
    assert 'href="#candidate-overlay"' in markup
    assert 'href="#review-triage"' in markup


def test_normal_annotation_mode_hides_legacy_workflow_selector():
    html = (ASSETS / "workbench.html").read_text(encoding="utf-8")
    css = (ASSETS / "workbench.css").read_text(encoding="utf-8")

    assert '<label class="legacyWorkflowControl">Workflow' in html
    assert ".app.normal-annotation-mode .legacyWorkflowControl" in css


def test_all_browser_mutations_use_owner_aware_headers():
    state = (ASSETS / "src" / "10_state_persistence.js").read_text(encoding="utf-8")
    controls = (ASSETS / "src" / "25_review_controls.js").read_text(encoding="utf-8")
    product = (ASSETS / "src" / "15_product_shell.js").read_text(encoding="utf-8")

    assert "method: 'PUT',\n    headers: generationHeaders()" in state
    assert "method:'PUT', headers:generationHeaders()" in controls
    assert "const headers = generationHeaders();" in product
    assert "X-Neurobench-Owner-Token" in state
    assert 'accept=".npy,.tif,.tiff,.csv,.tsv,.xlsx,.json"' in product
    assert "Confirm external NeuRev JSON" in product
    assert "native app state was not replaced" in product
