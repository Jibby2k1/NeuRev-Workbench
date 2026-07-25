# Focused Experiment Packages

`neurobench/experiments/` contains bounded, manifest-driven scientific
workflows. Generic program planning lives in `neurobench/programs/`.

| Package | Scientific job | CLI | Resource contract |
|---|---|---|---|
| `soma_excitation/` | frozen transfer to dark-soma excitation zones | `neurobench experiment soma-excitation` | CPU-only, explicit RAM/chunk/batch caps |
| `learnable_contrast/` | weakly supervised contrast, spatiotemporal diagnostic, direct tuning | `neurobench experiment learnable-contrast` | CUDA, explicit manifests and stage gates |

Package expectations:

- validated configuration;
- read-only preflight;
- collision-safe or explicitly resumable output;
- atomic status/metrics;
- exact frame and coordinate conventions;
- resource caps before scientific imports;
- focused tests and a matching workflow document.

Do not put action-conditioned control code here until its public state, action,
outcome, timing, and safety schemas are frozen.

