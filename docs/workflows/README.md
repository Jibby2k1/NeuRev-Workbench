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

Detailed guides:

- [Raw video to report](raw_video_to_report.md)
- [Spon soma excitation](spon_ca_burst_soma_excitation.md)
- [Learnable contrast](spon_ca_burst_learnable_contrast.md)
- [Template grid workflow](../TEMPLATE_GRID_WORKFLOW.md)
- [Grid latent dynamics](../GRID_LATENT_DYNAMICS.md)
- [Fish intent and inverse-control program](../programs/fish_inverse_control/README.md)

Do not infer launch readiness from a manifest's presence. Read `AGENTS.md` and
the workflow's current handoff first.

