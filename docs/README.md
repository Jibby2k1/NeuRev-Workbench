# NeuRev Workbench documentation

Use this page as a role-based index. For the smallest possible orientation,
start with the [repository guide](REPOSITORY_GUIDE.md).

## First five links

1. [Generated research story](../research/generated/PROJECT_STORY.md) — what the
   repository currently shows, does not show, and plans next.
2. [Current neuron-identifiability state](../paper/overleaf_jnm/CURRENT_RESEARCH_STATE.md)
   — the flagship study in manuscript language.
3. [Scientific audit standard](workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md) —
   the required contract for new experiments.
4. [Neuron Workbench](NEURON_WORKBENCH.md) — local review UI, annotations,
   autosave, and exports.
5. [Codebase navigation](CODEBASE_NAVIGATION.md) — maintained packages,
   ownership, entry points, and compatibility code.

The machine-readable companions are [`navigation.json`](navigation.json),
[`../llms.txt`](../llms.txt), and the generated
[`llm_context.json`](../research/generated/llm_context.json).

## Research and evidence

| Document | Use it for |
| --- | --- |
| [Claim ledger](../research/generated/CLAIM_LEDGER.md) | Atomic claims, registered scope, limitations, state, and linked experiments |
| [Experiment timeline](../research/generated/EXPERIMENT_TIMELINE.md) | Question-to-decision chronology, including negative and planned work |
| [Research registry guide](../research/README.md) | Canonical records, IDs, lifecycle fields, capsules, and generation workflow |
| [Neuron-identifiability result index](research/README.md) | Detailed result reports and interpretation notes |
| [Uncertainty-aware learning plan](research/SPON_CA_BURST_UNCERTAINTY_AWARE_LEARNING_PLAN_V1.md) | Prespecified learning, leakage, robustness, and label-sensitivity work |
| [External bounded-review package](research/SPON_CA_BURST_EXTERNAL_BOUNDED_REVIEW_V2_PACKAGE.md) | What is ready for independent review and what remains human-gated |
| [New candidate review](research/SPON_CA_BURST_NEW_CANDIDATE_REVIEW_V1_RESULTS.md) | Provisional single-reviewer labels and ascertainment boundary |
| [Publication boundary](PUBLICATION_BOUNDARY.md) | Public/private/generated/release artifact rules |

Detailed experiment reports live under [`research/`](research/). Their status
is historical evidence; current claim state comes from the root
[`research/registry/`](../research/registry/).

## Run a workflow

- [Workflow index](workflows/README.md)
- [Raw video to report](workflows/raw_video_to_report.md)
- [Neuron-identifiability paper workflow](workflows/spon_ca_burst_neuron_identifiability_paper.md)
- [Full-trace feature panel](workflows/spon_ca_burst_full_trace_feature_panel.md)
- [Automated feature validation](workflows/spon_ca_burst_automated_feature_validation.md)
- [Feature deep dives](workflows/spon_ca_burst_feature_deep_dives.md)
- [External blinded bounded review](workflows/external_blinded_bounded_review_v2.md)
- [Unsupervised two-frame ICA evaluation](workflows/unsupervised_two_frame_ica_eval.md)

All scientific workflows inherit the
[audit output standard](workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md). A run
must use a new output root and must not overwrite prior evidence.

## Review, annotation, and datasets

- [How to use the dashboard](HOW_TO_USE_DASHBOARD.md)
- [Neuron Workbench](NEURON_WORKBENCH.md)
- [Annotation schema](ANNOTATION_SCHEMA.md)
- [Dataset intake and QC](DATASET_QC.md)
- [Metrics audit](METRICS_AUDIT.md)
- [Inverse-dynamics export](INVERSE_DYNAMICS_EXPORT.md)
- [Resting-video algorithm brief](RESTING_VIDEO_ALGORITHM_BRIEF.md)

## Dynamics and downstream programs

- [Template grid workflow](TEMPLATE_GRID_WORKFLOW.md)
- [Grid and latent dynamics](GRID_LATENT_DYNAMICS.md)
- [Grid32 real-data pilot](case_studies/grid32_real_data_pilot.md)
- [Fish intent and inverse-control program](programs/fish_inverse_control/README.md)
- [Fish inverse-control roadmap](research/FISH_INVERSE_CONTROL_ROADMAP.md)

Passive forecasting, causal intent, action-conditioned dynamics, and safe
control are separate gates. Do not infer the later stages from upstream
measurement performance.

## Developer map

| Document | Scope |
| --- | --- |
| [Codebase navigation](CODEBASE_NAVIGATION.md) | Maintained packages and task routes |
| [API reference](API_REFERENCE.md) | Generated module, class, and function index |
| [Adding a pipeline stage](developer/adding_pipeline_stage.md) | Catalog, executor, artifacts, tests, and realtime metadata |
| [Architecture Lab](ARCHITECTURE_LAB.md) | Pipeline composition and comparison surface |
| [Workbench video/catalog refactor](developer/WORKBENCH_VIDEO_CATALOG_REFACTOR.md) | Current UI and media-routing architecture |
| [Dashboard revamp handoff](developer/DASHBOARD_REVAMP_HANDOFF_2026_08.md) | Recent correction-workspace direction and remaining work |
| [Developer archive index](developer/README.md) | Specialized implementation briefs and checkpoints |
| [Repository decisions](decisions/README.md) | Durable architecture and process choices |
| [Visual system](VISUAL_SYSTEM.md) | Brand tokens, reserved scientific colors, diagrams, and accessibility |
| [Documentation archive](archive/README.md) | Preserved plans and journals that are no longer current authority |

## Generated versus curated documents

Generated files say so in their header. Correct their canonical source and
rebuild them; do not patch the rendered output. In particular:

```bash
.venv-neurobench/bin/python -m neurobench.research.registry build
make -C paper/overleaf_jnm story
```

Curated result and interpretation documents may contain more detail than the
registry, but they may not widen a registered claim. If a curated page and a
generated claim state conflict, inspect the capsule and canonical record.

## Recommended reading paths

### Reviewer

1. [Research story](../research/generated/PROJECT_STORY.md)
2. [Claim ledger](../research/generated/CLAIM_LEDGER.md)
3. [Current manuscript state](../paper/overleaf_jnm/CURRENT_RESEARCH_STATE.md)
4. Relevant evidence capsules and result reports
5. [Publication boundary](PUBLICATION_BOUNDARY.md)

### New researcher

1. [Repository guide](REPOSITORY_GUIDE.md)
2. [Resting-video algorithm brief](RESTING_VIDEO_ALGORITHM_BRIEF.md)
3. [Neuron Workbench](NEURON_WORKBENCH.md)
4. [Scientific audit standard](workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md)
5. One bounded workflow and its focused tests

### Developer or coding agent

1. [`../AGENTS.md`](../AGENTS.md)
2. [`../llms.txt`](../llms.txt)
3. [Codebase navigation](CODEBASE_NAVIGATION.md)
4. [`navigation.json`](navigation.json)
5. The smallest relevant package and focused tests
