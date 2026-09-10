# Workflow index

Use this index for runnable scientific workflows. Program strategy and research
interpretation belong under `docs/programs/` and `docs/research/`.

Start with [Current work](../research/CURRENT_WORK.md) to identify the latest
result before following a run command. Every scientific workflow inherits the
[scientific audit standard](SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md); artifact
validation and scientific advancement are separate gates.

## Recent research workflows

| Workflow contract | Scope | Read alongside |
| --- | --- | --- |
| [Gamma-LS difference/ICA ablation](spon_ca_burst_gamma_ls_difference_ablation.md) | Guarded screen, protected comparisons, causal state, and streaming benchmark | [Final campaign results](../research/SPON_CA_BURST_GAMMA_LS_PAPER_SUCCESS_RESULTS_2026_09_09.md) |
| [Gamma-LS simple controls](gamma_ls_protected_simple_controls.md) | Matched radial, square-annulus, and box comparisons | Final campaign control and superiority boundary |
| [Gamma-LS exact-truth benchmark](gamma_ls_exact_truth_benchmark_v1.md) | Frozen simulator and framewise exhaustive truth | Simulator precision does not transfer to biological precision |
| [Gamma-LS independent sparse-positive confirmation](gamma_ls_independent_sparse_positive_confirmation.md) | Frozen score-to-label join and blockwise readout | One-recording sensitivity; unmatched proposals remain unknown |
| [ICA and whitening evaluation](spon_ca_burst_ica_whitening_evaluation.md) | Synthetic factorial, real-data selection, diagnostics, audit, and independent gates | [Final real-data results](../research/ICA_WHITENING_REAL_DATA_V1_FINAL_RESULTS.md) and [synthetic screen](../research/ICA_WHITENING_HYPERPARAMETER_EVALUATION_RESULTS.md) |
| [Unsupervised two-frame ICA](unsupervised_two_frame_ica_eval.md) | Label-free matched representations and controls | [PC-MITL specialized stop decision](../research/PC_MITL_ICA_SPECIALIZED_CONFIRMATION_RESULTS.md) |
| [Contextual-envelope pilot](contextual_envelope_features_v1.md) | Frozen feature forms, temporal windows, and spatial supports | [Pilot results](../research/CONTEXTUAL_ENVELOPE_FEATURES_V1_RESULTS.md) |
| [Contextual-envelope retrieval](contextual_envelope_retrieval_v1.md) | Known-center peak localization and matched misalignment controls | [Retrieval results](../research/CONTEXTUAL_ENVELOPE_RETRIEVAL_V1_RESULTS.md) |
| [Contextual-envelope morphology](contextual_envelope_morphology_v1.md) | Aligned full-trace shape, peak, shoulder, and duration comparisons | [Morphology results](../research/CONTEXTUAL_ENVELOPE_MORPHOLOGY_V1_RESULTS.md) |
| [Feature Atlas v1](spon_ca_burst_feature_atlas_v1.md) | Frozen-candidate reranking and feature-family comparisons | [Atlas results](../research/SPON_CA_BURST_FEATURE_ATLAS_V1_RESULTS.md) |
| [Uncertainty-aware learning v1.1](uncertainty_aware_feature_learning_v1_1.md) | Bounded Run-B implementation repair and matched linear/nonlinear fusion | [v1.1 results](../research/UNCERTAINTY_AWARE_FEATURE_LEARNING_V1_1_RESULTS.md); [immutable v1 contract](uncertainty_aware_feature_learning_v1.md) |

The Gamma-LS campaign's protected burst-occupancy head, independent blockwise
head, and operational framewise head have distinct calibration and counting
units. Preserve those definitions when reproducing figures or comparing runs.

## Established workflows and program routes

| Workflow | Start manifest | CLI or entry point | Device | Output family |
|---|---|---|---|---|
| Raw video to review/report | `examples/pipeline_spec.example.json` | `neurobench run` and `neurobench report` | CPU | `Outputs/NeuronReview` |
| Spon soma excitation transfer | `examples/spon_ca_burst_soma_excitation.example.json` | `neurobench experiment soma-excitation` | bounded CPU | `Outputs/SomaExcitation` |
| Learnable contrast/direct tuning | `examples/spon_ca_burst_learnable_contrast.example.json` | `neurobench experiment learnable-contrast` | CUDA, explicit run | `Outputs/LearnableContrast` |
| Template grid preprocessing | `examples/template_grid_32x32_pipeline.example.json` | `neurobench template` / `neurobench grid` | CPU/GPU by stage | `Outputs/GridModel` |
| Grid latent dynamics | `examples/grid_latent_dynamics_pipeline.example.json` | `neurobench dynamics` | mixed; long runs gated | `Outputs/GridModel` |
| Fish-control program audit | `examples/fish_control_program.example.json` | `neurobench program fish-control audit` | read-only CPU | committed audit or explicit output |
| Neuron-identifiability paper program | experiment-specific manifests and frozen inputs | `python -m neurobench.experiments.neuron_identifiability` or focused modules | mixed; output-root gated | `Outputs/NeuronIdentifiability` |
| Compact spatiotemporal JEPA pilot | `examples/spatiotemporal_jepa_representation_v1.example.json` | `python -m neurobench.experiments.neuron_identifiability jepa-pilot`; claim-bearing execution remains unauthorized | bounded CUDA, explicit run | `Outputs/NeuronIdentifiability/NREV-EXP-0028` |
| Conditional-background residual screen v1 (failed, immutable) | `examples/conditional_background_residual_v1.example.json` | historical v1 runner only; run A failed closed on raw-HC domain continuity | no new execution | `Outputs/NeuronIdentifiability/NREV-EXP-0029` |
| Conditional-background residual screen v1.1 (completed, on hold) | `examples/conditional_background_residual_v1_1.example.json` | schema-2 v1.1 runner; Run B completed the bounded screen and remains non-claim-bearing | completed CUDA screen; no unchanged rerun | `Outputs/NeuronIdentifiability/NREV-EXP-0029` |
| Motion and registration-confound audit v2 (completed, fields unreliable) | frozen EXP-0028/EXP-0029 records and source descriptor | versioned matched-support phase-correlation audit; Run E is canonical | bounded CPU screen; numeric audit incomplete scientifically | `Outputs/NeuronIdentifiability/NREV-EXP-0025` |
| Source-off predictor feasibility v1 (completed, none admitted) | `examples/source_off_predictor_feasibility_v1.example.json` | versioned source-off predictor runner; Run B completed all eight methods and the 11-check gate | bounded CPU screen; no detector launch | `Outputs/NeuronIdentifiability/NREV-EXP-0030` |
| External bounded-field review v2 | frozen Raw and assisted review inputs | portable browser ZIPs; private scoring wrapper | CPU/browser, human-gated | `Outputs/NeuronIdentifiability/external_blinded_bounded_review_v2` |

## Detailed guides and preserved checkpoints

- [Raw video to report](raw_video_to_report.md)
- [Spon soma excitation](spon_ca_burst_soma_excitation.md)
- [Learnable contrast](spon_ca_burst_learnable_contrast.md)
- [Template grid workflow](../TEMPLATE_GRID_WORKFLOW.md)
- [Grid latent dynamics](../GRID_LATENT_DYNAMICS.md)
- [Fish intent and inverse-control program](../programs/fish_inverse_control/README.md)
- [Neuron-identifiability paper workflow](spon_ca_burst_neuron_identifiability_paper.md)
- [Compact spatiotemporal JEPA representation pilot](spatiotemporal_jepa_representation_v1.md)
- [Compact spatiotemporal JEPA bounded-screen results](../research/SPATIOTEMPORAL_JEPA_SCREEN_V1_RESULTS.md)
- [Conditional-background residual screen](conditional_background_residual_v1.md)
- [Conditional-background residual screen v1.1 amendment](conditional_background_residual_v1_1.md)
- [Conditional-background residual screen v1.1 results](../research/CONDITIONAL_BACKGROUND_RESIDUAL_V1_1_RESULTS.md)
- [Motion and registration-confound audit v2](motion_registration_confound_audit_v2.md)
- [Motion and registration-confound audit v2 results](../research/MOTION_REGISTRATION_CONFOUND_V2_RESULTS.md)
- [Source-off predictor feasibility v1](source_off_predictor_feasibility_v1.md)
- [Source-off predictor feasibility v1 results](../research/SOURCE_OFF_PREDICTOR_FEASIBILITY_V1_RESULTS.md)
- [Original uncertainty-aware learning specification](../research/SPON_CA_BURST_UNCERTAINTY_AWARE_LEARNING_PLAN_V1.md)
- [External blinded bounded review v2](external_blinded_bounded_review_v2.md)

Do not infer launch readiness from a manifest's presence. Read `AGENTS.md` and
the workflow's current handoff first.
