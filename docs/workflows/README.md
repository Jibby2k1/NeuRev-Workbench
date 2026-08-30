# Workflow Index

Use this index for runnable scientific workflows. Program strategy and research
interpretation belong under `docs/programs/` and `docs/research/`.

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

Detailed guides:

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
- [Next uncertainty-aware learning specification](../research/SPON_CA_BURST_UNCERTAINTY_AWARE_LEARNING_PLAN_V1.md)
- [External blinded bounded review v2](external_blinded_bounded_review_v2.md)

Do not infer launch readiness from a manifest's presence. Read `AGENTS.md` and
the workflow's current handoff first.
