# Workspace Organization

## Implemented organization

The repository remains role-oriented rather than moving large or historically
referenced trees:

- `docs/programs/fish_inverse_control/` is the one context entry for this
  research program;
- `docs/research/` retains scientific method documents;
- `docs/developer/` retains implementation handoffs;
- `docs/reports/` retains generated reader artifacts and their canonical source;
- `examples/` retains flat manifests so existing relative paths do not change;
- `neurobench/programs/` owns reusable program auditing;
- `schemas/` owns the public program contract;
- `Outputs/` and `Inputs/` are never moved.

This avoids copying 708+ GiB of local data or breaking external/Dropbox paths.

## Search hygiene

`.rgignore` excludes:

- large ignored data and environments;
- generated workbench JavaScript;
- generated HTML reports;
- generated API reference;
- generic historical plan files.

Each excluded file remains directly readable by path or with ripgrep's
`--no-ignore` option. `.gitattributes` marks generated assets for code-review
tools.

## Navigation indexes

The following indexes make file discovery explicit:

- `docs/workflows/README.md`
- `docs/research/README.md`
- `docs/developer/README.md`
- `docs/reports/README.md`
- `examples/README.md`
- `neurobench/experiments/README.md`
- `tests/README.md`

`README.md` and `docs/CODEBASE_NAVIGATION.md` route into these indexes.

## Preserved historical plans

The following old, generic plan files are preserved under
`docs/archive/plans/`:

- `docs/archive/plans/goal.md`
- `docs/archive/plans/plan.md`
- `docs/archive/plans/codex_neurobench_plan.md`
- `docs/plan.md`

The three generic root snapshots were moved intact and their known incoming
references were updated. `docs/plan.md` remains the curated long-term
documentation roadmap rather than a root-level project entry point.

Also deferred:

- moving any `Inputs/` or `Outputs/` family;
- archiving `core/`, `evaluation/`, or `reporting/` while tests import them;
- reorganizing flat examples whose relative data paths would change;
- splitting the stopped overnight sweep runner;
- changing the source-tree dependency in
  `neurobench/dynamics/manual_annotations.py` without a dedicated regression
  task.
