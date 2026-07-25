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

## Intentionally deferred moves

The following old, generic plan files remain in place:

- `goal.md`
- `plan.md`
- `codex_neurobench_plan.md`
- `docs/plan.md`

They are large historical snapshots with possible external references. Moving
them during an already dirty research batch would create compatibility risk.
They are classified as historical in navigation and ignored by default search.
A future clean commit may move them under `docs/archive/plans/` while leaving
short compatibility stubs.

Also deferred:

- moving any `Inputs/` or `Outputs/` family;
- archiving `core/`, `evaluation/`, or `reporting/` while tests import them;
- reorganizing flat examples whose relative data paths would change;
- splitting the stopped overnight sweep runner;
- changing the source-tree dependency in
  `neurobench/dynamics/manual_annotations.py` without a dedicated regression
  task.

