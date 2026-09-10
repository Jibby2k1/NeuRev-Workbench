# NeuRev Workbench documentation

Use this page as a role-based index. For the smallest possible orientation,
start with the [repository guide](REPOSITORY_GUIDE.md).

## Start here

1. [Current work](research/CURRENT_WORK.md) — the latest repository evidence,
   implementation routes, and remaining gates across research threads.
2. [Generated research story](../research/generated/PROJECT_STORY.md) — registered
   claim states and the question-to-decision history.
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
| [Current work](research/CURRENT_WORK.md) | Cross-program synthesis that keeps dated snapshots and finalized results distinct |
| [Gamma-LS final campaign, September 9](research/SPON_CA_BURST_GAMMA_LS_PAPER_SUCCESS_RESULTS_2026_09_09.md) | Compact causal architecture, protected ablations, full-record proposal audit, independent sparse-positive readout, and failed timing gates |
| [ICA/whitening final real-data results](research/ICA_WHITENING_REAL_DATA_V1_FINAL_RESULTS.md) | Concluded factorial, protected finalist and independent evaluation, with its own readout and claim limits |
| [Claim ledger](../research/generated/CLAIM_LEDGER.md) | Atomic claims, registered scope, limitations, state, and linked experiments |
| [Experiment timeline](../research/generated/EXPERIMENT_TIMELINE.md) | Question-to-decision chronology, including negative and planned work |
| [Research registry guide](../research/README.md) | Canonical records, IDs, lifecycle fields, capsules, and generation workflow |
| [Research result index](research/README.md) | Gamma-LS, ICA/whitening, PC-MITL, contextual envelopes, Feature Atlas, learning, and historical studies |
| [Analytical report artifact index](reports/README.md) | Source-backed analytical artifacts and generated-reader status |
| [Uncertainty-aware learning v1.1 results](research/UNCERTAINTY_AWARE_FEATURE_LEARNING_V1_1_RESULTS.md) | Completed engineering screen and the unresolved nonlinear-versus-linear contrast |
| [Current neuron-identifiability state](../paper/overleaf_jnm/CURRENT_RESEARCH_STATE.md) | The flagship study in manuscript language |
| [External bounded-review package](research/SPON_CA_BURST_EXTERNAL_BOUNDED_REVIEW_V2_PACKAGE.md) | What is ready for independent review and what remains human-gated |
| [New candidate review](research/SPON_CA_BURST_NEW_CANDIDATE_REVIEW_V1_RESULTS.md) | Provisional single-reviewer labels and ascertainment boundary |
| [Publication boundary](PUBLICATION_BOUNDARY.md) | Public/private/generated/release artifact rules |

Detailed experiment reports live under [`research/`](research/). Their status
is historical evidence; current claim state comes from the root
[`research/registry/`](../research/registry/).

## Run a workflow

- [Workflow index](workflows/README.md)
- [Gamma-LS difference/ICA ablation](workflows/spon_ca_burst_gamma_ls_difference_ablation.md)
- [ICA and whitening evaluation](workflows/spon_ca_burst_ica_whitening_evaluation.md)
- [Feature Atlas](workflows/spon_ca_burst_feature_atlas_v1.md)
- [Uncertainty-aware learning v1.1](workflows/uncertainty_aware_feature_learning_v1_1.md)
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

1. [Current work](research/CURRENT_WORK.md) and [research story](../research/generated/PROJECT_STORY.md)
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
