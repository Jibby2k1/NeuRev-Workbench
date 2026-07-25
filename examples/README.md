# Example Manifest Index

Examples remain flat because many paths are relative to this directory. Moving
them into subdirectories would silently change input/output resolution.

## Program and dataset contracts

| Example | Purpose | Validation or audit |
|---|---|---|
| `fish_control_program.example.json` | eight-experiment activation-to-control portfolio | `neurobench program fish-control audit` |
| `dataset_manifest.example.json` | imaging, behavior, online, and path metadata | `neurobench dataset validate` |
| `video_manifest.example.json` | source-video contract | `neurobench validate` |
| `architecture_runs.example.json` | architecture-result collection | schema validation |

## Focused experiments

| Example | Workflow | Device |
|---|---|---|
| `spon_ca_burst_soma_excitation.example.json` | dark-soma excitation transfer | bounded CPU |
| `spon_ca_burst_learnable_contrast.example.json` | guarded learnable contrast v1 | CUDA |
| `spon_ca_burst_learnable_contrast_spatiotemporal_diagnostic.example.json` | 2×2×2 v2 diagnostic | CUDA |
| `spon_ca_burst_learnable_direct_tuning.example.json` | direct-initialized v3 tuning | CUDA |
| `gamma_cfar_cascade_sweep.example.json` | Gamma-CFAR cascade sweep | mixed |

## Pipelines and grid dynamics

| Example | Purpose |
|---|---|
| `pipeline_spec.example.json` | minimal runnable pipeline |
| `template_grid_32x32_pipeline.example.json` | 32×32 registered grid extraction |
| `template_grid_128x128_pipeline.example.json` | 128×128 registered grid extraction |
| `grid_latent_dynamics_pipeline.example.json` | latent dynamics workflow |
| `template_spec.example.json` | template construction |
| `grid_spec_32x32.example.json` | grid-state contract |
| `registration_result.example.json` | registration result contract |

Run preflight/audit commands before any long execution. Example presence does
not authorize a stopped sweep, Stage B, or a GPU experiment.

