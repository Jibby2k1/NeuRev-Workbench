# Developer Guide Index

| Task | Read first | Main implementation |
|---|---|---|
| Navigate the package | [Codebase navigation](../CODEBASE_NAVIGATION.md) | `neurobench/` |
| Add a pipeline stage | [Adding a pipeline stage](adding_pipeline_stage.md) | `neurobench/pipeline_catalog.py`, `neurobench/pipelines/` |
| Work on stopped grid128 experiments | [Grid128 handoff](GRID128_EXPERIMENT_HANDOFF.md) | `neurobench/dynamics/` |
| Implement fish intent/control tools | [Fish control tooling roadmap](FISH_CONTROL_TOOLING_ROADMAP.md) | `neurobench/programs/`, future intent/control packages |
| Change workbench UI | [Neuron Workbench](../NEURON_WORKBENCH.md) | `neurobench/workbench/assets/src/` |
| Query/build video review apps | [Workbench video/catalog refactor](WORKBENCH_VIDEO_CATALOG_REFACTOR.md) | `neurobench/data/catalog.py`, `neurobench/workbench/` |
| Review architecture debt | [Codebase audit](../CODEBASE_AUDIT.md) | use the current package map before acting |

Public commands should use thin modules under `neurobench/cli/`. Reusable
science belongs in `neurobench/`, not in one-off scripts. New schemas require
examples and focused validation tests.
