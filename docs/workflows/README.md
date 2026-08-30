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
| External bounded-field review v2 | frozen Raw and assisted review inputs | portable browser ZIPs; private scoring wrapper | CPU/browser, human-gated | `Outputs/NeuronIdentifiability/external_blinded_bounded_review_v2` |

Detailed guides:

- [Raw video to report](raw_video_to_report.md)
- [Spon soma excitation](spon_ca_burst_soma_excitation.md)
- [Learnable contrast](spon_ca_burst_learnable_contrast.md)
- [Template grid workflow](../TEMPLATE_GRID_WORKFLOW.md)
- [Grid latent dynamics](../GRID_LATENT_DYNAMICS.md)
- [Fish intent and inverse-control program](../programs/fish_inverse_control/README.md)
- [Neuron-identifiability paper workflow](spon_ca_burst_neuron_identifiability_paper.md)
- [Next uncertainty-aware learning specification](../research/SPON_CA_BURST_UNCERTAINTY_AWARE_LEARNING_PLAN_V1.md)
- [External blinded bounded review v2](external_blinded_bounded_review_v2.md)

Do not infer launch readiness from a manifest's presence. Read `AGENTS.md` and
the workflow's current handoff first.
