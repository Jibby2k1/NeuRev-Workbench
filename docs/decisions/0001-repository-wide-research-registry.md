# 0001 — Repository-wide research registry and generated story

- **Status:** accepted
- **Date:** 2026-08-29
- **Scope:** repository organization, experiment provenance, public evidence,
  generated narratives, and GitHub presentation

## Context

NeuRev Workbench now spans maintained software, multiple scientific programs,
hundreds of tests, dozens of workflow and result documents, a manuscript, local
large artifacts, and human-review packages. The neuron-identifiability paper
already has a valuable YAML claim and experiment ledger, but that ledger lives
inside one manuscript package and does not represent the complete repository.

Several human-facing pages repeat scientific facts manually. Large local
outputs are ignored by Git but are referenced by story validation, so checks
that pass on the research workstation are not necessarily portable to a clean
clone. Execution state, scientific outcome, evidence tier, review state, and
the resulting decision are also encoded together in free-form status strings.

The public repository must become easier to understand without weakening its
scientific boundaries or publishing raw, private, or oversized artifacts.

## Decision

NeuRev will use a repository-level, federated research registry with one record
per program, experiment, run, decision, claim, and compact evidence capsule.
The registry—not a manuscript or a manually maintained summary—will be the
authority for repeated research-story facts.

The system follows these rules:

1. **One record, many views.** GitHub summaries, documentation indexes,
   experiment timelines, diagrams, machine navigation, and Overleaf fragments
   are generated from the same canonical records.
2. **Large evidence stays external to Git.** Complete videos, arrays, private
   reviewer material, and full run directories remain under ignored output
   roots or immutable external archives.
3. **Small evidence capsules are versioned.** A capsule records logical artifact
   roles, hashes, validation state, headline results, limitations, and the
   permitted claim scope without embedding workstation-specific absolute paths.
4. **Scientific and operational states are separate.** Lifecycle, outcome,
   evidence tier, review state, claim state, and decision action use controlled
   vocabularies rather than compound free-form statuses.
5. **Completed history is durable.** Negative, partial, rejected, and
   superseded experiments remain discoverable. Amendments and supersession are
   explicit; completed records are not silently repurposed.
6. **Migration does not change conclusions.** Existing claims, findings,
   limitations, decisions, order, and manuscript views must survive migration
   semantically unchanged. Missing historical provenance is marked missing
   rather than reconstructed from assumption.
7. **A clean clone is a validation target.** Default CI may use only committed
   records and portable evidence. A stricter local verification mode may also
   reconcile ignored live artifacts and their hashes.
8. **Public presentation follows evidence.** NeuRev Workbench is the repository
   identity, `neurobench` remains the Python package and CLI, and Neuron
   Identifiability is the flagship current research story. Downstream dynamics
   and control programs retain separate causal gates.

## Intended lifecycle

An experiment proceeds through draft, preregistration, execution, computation,
scientific-audit validation, review, decision, claim update, story regeneration,
and archival. A completed run alone cannot promote a claim.

## Alternatives considered

### Keep the paper YAML as repository authority

Rejected because the repository contains programs that are not part of that
paper, and a manuscript-specific location makes repository navigation and
future paper reuse awkward.

### Commit complete output directories

Rejected because the current output corpus contains large videos, arrays,
review payloads, and workstation-specific provenance. Git is not the correct
transport for the full scientific artifact store.

### Maintain README, documentation, and manuscript stories independently

Rejected because current-state and next-experiment text has already drifted
between otherwise well-maintained documents.

### Use one permanently growing monolithic registry file

Rejected as the long-term form because it increases merge conflicts, obscures
ownership, and makes append-only experiment history difficult to audit. A
compiled canonical view may still be monolithic for efficient readers.

## Consequences

- The initial migration is larger than a cosmetic README edit.
- CI must distinguish portable evidence validation from optional live-artifact
  reconciliation.
- Existing documents can remain at stable paths while generated indexes and
  compatibility views are introduced.
- Repository visuals can be generated from real program and experiment state
  rather than maintained as decorative snapshots.
- Future experiments incur a small up-front registration cost in exchange for
  reproducible provenance and automatic publication updates.

## Falsifiers and revisit criteria

Revisit this decision if any of the following occurs:

- the migrated registry cannot reproduce the existing claim and experiment
  story without widening, narrowing, or reordering conclusions;
- evidence capsules cannot support meaningful clean-clone validation;
- normal experiment registration requires duplicating the same fact in more
  than one canonical source;
- generated GitHub, documentation, or manuscript views become harder to review
  than their present equivalents;
- a durable external research-object system replaces the local artifact model
  and provides stronger immutable identity and access guarantees.

## Verification

The decision is implemented only when:

- registry schemas and lifecycle validation pass;
- all existing claims and experiments are migrated and reconciled;
- clean-clone story and navigation checks do not require ignored outputs;
- local strict verification detects artifact drift when live outputs exist;
- generated repository and Overleaf views are current;
- publication-boundary checks find no private payloads, absolute workstation
  paths, or unapproved large artifacts in the public source set.
