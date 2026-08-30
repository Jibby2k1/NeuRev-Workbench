# NeuRev Workbench repository guide

This is the shortest reliable map for researchers, reviewers, developers, and
coding agents. NeuRev is broad, but its authority order is deliberately small.

## Start with the question you have

| Goal | Read first | Canonical authority | Validation |
| --- | --- | --- | --- |
| Understand the current research story | [Generated project story](https://github.com/Jibby2k1/NeuRev-Workbench/blob/main/research/generated/PROJECT_STORY.md) | [`research/registry/`](https://github.com/Jibby2k1/NeuRev-Workbench/tree/main/research/registry) | `python -m neurobench.research.registry check` |
| Inspect claim strength and limitations | [Claim ledger](https://github.com/Jibby2k1/NeuRev-Workbench/blob/main/research/generated/CLAIM_LEDGER.md) | [`research/registry/claims/`](https://github.com/Jibby2k1/NeuRev-Workbench/tree/main/research/registry/claims) plus [evidence capsules](https://github.com/Jibby2k1/NeuRev-Workbench/tree/main/research/evidence) | Registry check |
| Follow experiment history and decisions | [Experiment timeline](https://github.com/Jibby2k1/NeuRev-Workbench/blob/main/research/generated/EXPERIMENT_TIMELINE.md) | Experiment, decision, and capsule records | Registry check |
| Inspect the neuron-identifiability paper | [Current research state](https://github.com/Jibby2k1/NeuRev-Workbench/blob/main/paper/overleaf_jnm/CURRENT_RESEARCH_STATE.md) | Canonical [`research/registry/`](https://github.com/Jibby2k1/NeuRev-Workbench/tree/main/research/registry) records | `make -C paper/overleaf_jnm story-check` |
| Run or review a scientific experiment | [Scientific audit standard](workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md) | Frozen protocol, run record, validation, and capsule | Experiment-specific tests and gates |
| Use the local candidate-review UI | [Neuron Workbench](NEURON_WORKBENCH.md) | `neurobench/workbench/` | Workbench-focused tests |
| Implement a maintained method | [Codebase navigation](CODEBASE_NAVIGATION.md) | `neurobench/` | Focused package tests |
| Work on grid or latent dynamics | [Grid latent dynamics](GRID_LATENT_DYNAMICS.md) | `neurobench/dynamics/` | Dynamics tests and run manifests |
| Work on intent or inverse control | [Program hub](programs/fish_inverse_control/README.md) | Program gates and registered evidence | Read-only program audit |
| Prepare a public or reviewer release | [Publication boundary](PUBLICATION_BOUNDARY.md) | Git candidate plus sanitized capsules | Publication audit |

## Authority order

When two files appear to disagree, use this order:

1. immutable manifests, hashes, registered run records, and validation outputs;
2. sanitized evidence capsules under [`research/evidence/`](https://github.com/Jibby2k1/NeuRev-Workbench/tree/main/research/evidence);
3. canonical experiment, decision, and claim records under
   [`research/registry/`](https://github.com/Jibby2k1/NeuRev-Workbench/tree/main/research/registry);
4. generated research, documentation, navigation, and Overleaf views;
5. curated interpretation notes;
6. meeting material, historical plans, and local output prose.

A passing unit test proves a software contract. A completed process proves an
execution state. Neither alone proves a scientific claim.

<!-- BEGIN GENERATED FLAGSHIP BOUNDARY -->

## Current flagship boundary

An intensive within-recording methods study of 106 confirmed occurrences at 50 immutable sites in one calcium-imaging recording.

The registered unresolved boundaries are:

- Full-field precision is unresolved.
- One-to-one biological source identity is unresolved.
- Generalization to an independent recording is unresolved.

The exact claim states and next-experiment queue are generated from canonical
records; use the [claim ledger](https://github.com/Jibby2k1/NeuRev-Workbench/blob/main/research/generated/CLAIM_LEDGER.md) rather
than copying status prose into another document.

<!-- END GENERATED FLAGSHIP BOUNDARY -->

## Repository layers

| Layer | Role | Editing rule |
| --- | --- | --- |
| `neurobench/` | Maintained package, CLI, algorithms, workbench, experiments, dynamics | Primary implementation surface |
| `research/registry/` | Programs, claims, experiments, decisions, and runs | Canonical; schema validated |
| `research/evidence/` | Small public evidence capsules | Canonical; sanitized and checksum backed |
| `research/generated/` | Project story, claim ledger, timeline, machine context | Generated; never hand edit |
| `docs/workflows/` | Runnable workflow and audit contracts | Operational authority |
| `docs/research/` | Detailed results and interpretation | Check linked evidence and date |
| `paper/overleaf_jnm/` | Manuscript and synchronized story views | Build from registry, then build paper |
| `tests/` | Software and scientific-contract checks | Validation, not scientific authority |
| `Inputs/` | Raw or private local data | Ignored; never publish |
| `Outputs/` | Complete local runs, videos, arrays, reviewer material | Ignored; publish capsules instead |
| `core/`, `evaluation/`, `reporting/` | Compatibility code | Prefer maintained equivalents |

## Efficient evidence inspection

For a local output root, read small artifacts before opening large media:

1. `llm_context.json`;
2. `summary.json`;
3. `validation.json`;
4. `artifact_index.json` or manifest;
5. narrow TSV or CSV tables;
6. representative figures and stills;
7. full videos only for visual adjudication.

For public review, start with the evidence capsule. `metadata_only` means the
capsule identifies a local artifact by role, path, size, and hash without
publishing its payload.

## Common commands

```bash
# Portable clean-clone checks
.venv-neurobench/bin/python -m neurobench.research.registry check
make -C paper/overleaf_jnm story-check
.venv-neurobench/bin/python tools/audit_publication_boundary.py

# Research-workstation reconciliation
.venv-neurobench/bin/python -m neurobench.research.registry verify-live --check

# Focused tests
.venv-neurobench/bin/python -m pytest -q <test paths>
```

After changing a canonical record, regenerate views with:

```bash
.venv-neurobench/bin/python -m neurobench.research.registry build
make -C paper/overleaf_jnm story
```

Never overwrite a completed output root. Create a new run ID, preserve the old
result, and record supersession or invalidation explicitly.

## Machine-readable companions

- [`llms.txt`](https://github.com/Jibby2k1/NeuRev-Workbench/blob/main/llms.txt) is the smallest coding-agent entry point.
- [`navigation.json`](navigation.json) contains task routes and authority paths.
- [`research/generated/llm_context.json`](https://github.com/Jibby2k1/NeuRev-Workbench/blob/main/research/generated/llm_context.json)
  contains the bounded research snapshot.
- [`research/generated/canonical.json`](https://github.com/Jibby2k1/NeuRev-Workbench/blob/main/research/generated/canonical.json)
  is the compiled registry view.

These are generated navigation aids. The federated registry remains canonical.
