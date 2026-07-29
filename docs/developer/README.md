# Developer Guide Index

| Task | Read first | Main implementation |
|---|---|---|
| Navigate the package | [Codebase navigation](../CODEBASE_NAVIGATION.md) | `neurobench/` |
| Add a pipeline stage | [Adding a pipeline stage](adding_pipeline_stage.md) | `neurobench/pipeline_catalog.py`, `neurobench/pipelines/` |
| Work on stopped grid128 experiments | [Grid128 handoff](GRID128_EXPERIMENT_HANDOFF.md) | `neurobench/dynamics/` |
| Implement fish intent/control tools | [Fish control tooling roadmap](FISH_CONTROL_TOOLING_ROADMAP.md) | `neurobench/programs/`, future intent/control packages |
| Run or extend stable AR latent denoising | [Latent-dynamics workflow](../workflows/spon_ca_burst_latent_dynamics.md), [implementation brief](LATENT_DYNAMICS_DENOISING_IMPLEMENTATION_BRIEF.md) | `neurobench/algorithms/latent_dynamics.py`, `neurobench/experiments/latent_dynamics/` |
| Plan hierarchical Parzen/noisy ICA decomposition | [Research note](../research/HIERARCHICAL_PARZEN_NOISY_ICA.md), [Codex package](hierarchical_parzen_noisy_ica/README.md) | planned `neurobench/algorithms/hierarchical_parzen.py`, `neurobench/algorithms/noisy_parzen_ica.py`, and `neurobench/experiments/hierarchical_parzen_noisy_ica/` |
| Run PCA, spatial ICA, and autoencoder representation benchmarks | [Representation benchmark](../workflows/spon_ca_burst_representation_benchmark.md) | `neurobench/experiments/representation_benchmark/` |
| Run or implement pairwise binary difference, ICA, CS-divergence, or constrained NMF | [Pairwise workflow](../workflows/spon_ca_burst_pairwise_separation.md), [implementation brief](PAIRWISE_SOURCE_SEPARATION_IMPLEMENTATION_BRIEF.md) | `neurobench/algorithms/pairwise_separation.py`, `neurobench/experiments/pairwise_separation/`, `neurobench/metrics/sparse_detection.py` |
| Change workbench UI | [Neuron Workbench](../NEURON_WORKBENCH.md) | `neurobench/workbench/assets/src/` |
| Query/build video review apps | [Workbench video/catalog refactor](WORKBENCH_VIDEO_CATALOG_REFACTOR.md) | `neurobench/data/catalog.py`, `neurobench/workbench/` |
| Implement the NeuRev first release | [First-release handoff](NEUREV_FIRST_RELEASE_HANDOFF.md) | `neurobench/data/imports.py`, `neurobench/workbench/baseline.py`, `neurobench/workbench/server.py`, `neurobench/workbench/assets/src/` |
| Review architecture debt | [Codebase audit](../CODEBASE_AUDIT.md) | use the current package map before acting |

Public commands should use thin modules under `neurobench/cli/`. Reusable
science belongs in `neurobench/`, not in one-off scripts. New schemas require
examples and focused validation tests.
