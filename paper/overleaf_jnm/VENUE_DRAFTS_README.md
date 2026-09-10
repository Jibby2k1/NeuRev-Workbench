# Venue-specific manuscript drafts

These two entry points share the bounded biological case-study methods/results,
declarations, source-bound facts, bibliography, and five common figures. Their
introductions and discussions are venue-specific; each also has a distinct
Figure 1, and the Neuroinformatics draft has venue-specific system, software
validation, and conclusion sections. The new body lives under `venue_drafts/`
so it does not inherit older pre-rerun cohort language still present in the
canonical manuscript sections.

## Journal of Neuroscience Methods

- Entry point: `main_jnm_identity_safe_draft.tex`
- Highlights: `highlights_jnm_identity_safe.txt`
- Emphasis: measurement validity, sparse-positive inference, identity-aware
  failure decomposition, and competition-aware proposal resolution.
- Recommended first-submission version on the current evidence.

## Neuroinformatics

- Entry point: `main_neuroinformatics_identity_safe_draft.tex`
- Highlights: `highlights_neuroinformatics_identity_safe.txt`
- Emphasis: immutable identity and geometry contracts, positive--unlabeled
  estimands, run/evidence provenance, blinded review tooling, and reproducible
  scientific-promotion gates.
- Uses the official Springer Nature `sn-jnl` class, template version 3.1
  (December 2024), with the author--year math/physical-sciences bibliography
  style. The upload package vendors the required class and style files.

## Shared scientific boundary

Neither version claims that the 13 definite/probable calls are 13 independently
validated new neurons or that 13/18 estimates detector precision. Neither
version claims that the Feature Atlas is a confirmed improvement: its favorable
point estimate did not pass the grouped uncertainty gate. Full-field precision,
independent-recording generalization, and final software/data release remain
open submission gates.

These are visually complete working drafts, not submission-ready manuscripts.
Both contain six integrated main figures, an evidence-status/estimand table, a
compact quantitative result ledger, source-hashed result and figure manifests,
and a provisional appendix. The Neuroinformatics version adds a
software-validation matrix. Author metadata,
declarations, independent candidate adjudication, release identifiers, and a
targeted reference verification pass remain open.

## Rebuild and upload

- Refresh repeated facts:
  `.venv-neurobench/bin/python paper/overleaf_jnm/scripts/build_venue_result_facts.py`
- Refresh the figures:
  `.venv-neurobench/bin/python paper/overleaf_jnm/scripts/build_venue_manuscript_figures.py`
- Build deterministic flat packages:
  `.venv-neurobench/bin/python paper/overleaf_jnm/scripts/build_venue_overleaf_packages.py`
- Current upload directory: `../overleaf_uploads/2026-09-07/`

The package checker rejects unresolved inputs, placeholder figures, stale
79-occurrence/27-site language, excluded Feature Atlas values, and ambiguous
“58-candidate” wording. Figure 2 intentionally labels its image panels as
stage-specific response-maximizing frames; they are not same-frame views.

## Canonical-source rule

These are venue drafts, not independent evidence authorities. Registered claim
state remains controlled by `../../research/registry/`; repeated venue facts are
generated from compact source artifacts into `macros/venue_results.tex`, with
paths and hashes recorded in `macros/venue_results_manifest.json`. If a number
or status changes, update the canonical registry or source analysis first,
regenerate the facts and figures, run the story check, and revise both abstracts
together.
