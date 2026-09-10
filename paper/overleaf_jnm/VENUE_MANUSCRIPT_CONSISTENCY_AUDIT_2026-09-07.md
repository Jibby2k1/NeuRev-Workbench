# Venue manuscript consistency audit

Date: 2026-09-07

Scope: Journal of Neuroscience Methods and Neuroinformatics working drafts,
including narrative, repeated quantitative facts, figures, tables, flat
Overleaf packages, local PDF renders, and focused consistency tests.

## Outcome

Both papers now form complete, internally consistent **working drafts**. Each
has a venue-specific opening, six integrated source-backed main figures, an
explicit estimand/evidence-status table, a compact quantitative result ledger,
a provenance-bound quantitative core, and a provisional appendix. The
Neuroinformatics version additionally has a
distinct software architecture, validation matrix, discussion, and conclusion.

They are not submission-ready. Independent candidate adjudication, author and
ethics metadata, public release identifiers, reference verification, removal or
regeneration of the provisional two-frame control, and the remaining canonical
registry failures are still open gates.

## High-severity corrections made

1. The 94/106 endpoint is now described as an any-lane union over three
   independently evaluated per-burst B58 lists. It is never described as one
   combined global top-58 list.
2. The 93/102 canonical-collapsed analysis is labeled as a separate sensitivity
   view rather than a replacement cohort.
3. Per-burst B100 candidate availability is kept distinct from both B58
   sensitivity and any global reranking endpoint.
4. Feature Atlas numerical results were removed from both abstracts, the main
   Results, and the six-figure packages. The local run is explicitly excluded
   because it is not canonically registered and its model-annotation media audit
   is incomplete.
5. The legacy two-frame ICA correlation is confined to the Appendix, carries a
   visible provisional marker, and is separated from the registered
   six-component long-delay inversion audit.
6. The candidate-review result is consistently described as single-reviewer,
   detector-selected yield—not precision, a cohort update, or 13 validated new
   biological identities.
7. Exact-truth simulations replace Feature Atlas as the sixth main figure and
   are explicitly bounded as computational mechanism evidence.
8. Figure 2 now states that its image fields are stage-specific
   response-maximizing frames and adds the otherwise unsupported 38-site/12-site
   exploratory trace-phenotype panel.
9. Figure 4 now includes the blinded review flow and the priority-score AUC
   interpretation. Figure 5 names per-burst B100 and shows proposal absence
   alongside the identity decomposition.
10. Stale 79-occurrence/27-site figures are excluded by the venue-only package
    builder and cannot enter either ZIP.

## Narrative and visual spine

| Position | Journal of Neuroscience Methods | Neuroinformatics | Visual evidence |
| --- | --- | --- | --- |
| 1 | Sparse-positive inference and identity error motivate the measurement method | Identity-safe schema and evidence lifecycle motivate the software | Venue-specific Figure 1 |
| 2 | Define cohort, representation, budgets, controls, review, and identity audit | Define the shared case study plus software contracts and promotion states | Figure 2; evidence-status Table 1 |
| 3 | Quantify known-positive recovery and localized structure | Demonstrate the case study without treating it as broad software validation | Figure 3 |
| 4 | Show what selected unmatched candidates contain and retain review uncertainty | Demonstrate the review lifecycle and P/U state preservation | Figure 4 |
| 5 | Explain why apparent response gains can be identity failures | Validate identity invariants and failure-stage reporting | Figure 5; software-validation Table 2 in Neuroinformatics |
| 6 | Use exact truth to resolve feature roles and failure boundaries | Use exact truth as functional validation while separating evidence tiers | Figure 6 |
| Close | Conclude on valid inference under incomplete annotation | Conclude on reusable provenance architecture and unresolved release gates | Explicit limitations and declarations |

The complete result placement and experiment inclusion decisions are recorded
in `EXPERIMENT_INCLUSION_MATRIX_2026-09-07.md`.

## Provenance controls

- `macros/venue_results.tex` is generated from compact result artifacts; its
  manifest records every source path and SHA-256 hash.
- `figures/venue/venue_figure_manifest.json` hashes all source and output assets.
- The package builder flattens paths for Overleaf, includes only the six selected
  PNGs, and writes per-package `SHA256SUMS.txt` files.
- Package validation rejects unresolved inputs, figure placeholders, obsolete
  cohort counts, Feature Atlas point estimates, and ambiguous B58 wording.

## Verification record

- Venue fact freshness: passed; 52 generated facts reconcile against 12 hashed
  source artifacts.
- Venue package tests: 5 passed.
- Canonical story freshness: passed; 20 claims and 25 paper-story experiments.
- Scientific/identity/review focused tests: 19 passed.
- Registry/story/schema focused tests: 36 passed and 2 failed at already-known
  gates: NREV-EXP-0033 lacks a portable evidence capsule, and native draft
  ordering does not place an unprioritized injected experiment last.
- A monolithic invocation of the same focused files encountered a collection-time
  Python/SciPy segmentation fault. Splitting the files into fresh processes
  produced the deterministic results above; the crash is recorded as an
  environment event, not converted into a pass.
- The broader local-suite snapshot was 1,522 passed, 18 skipped, and 14 failed.
  Eleven failures were socket/environment dependent; three were effective
  repository gates, including generated API-reference drift. The drafts do not
  describe this state as a clean or exhaustive test pass.
- JNM PDF: compiled to 21 letter-sized pages and all pages were rendered and
  inspected. There are no figure/table overflows; Tectonic reports a 2.611-point
  Elsevier frontmatter/output-box warning plus ordinary underfull paragraphs.
- Neuroinformatics PDF: compiled to 19 A4 pages and all pages were rendered and
  inspected. There are no overfull boxes; the Springer class reports ordinary
  underfull page/paragraph warnings.
- Both package `SHA256SUMS.txt` manifests validate, and both ZIP archives pass
  complete `unzip -t` integrity checks.

## Remaining submission gates

### Scientific

- Complete independent second review and adjudication for all 18 selected
  unmatched candidates.
- Obtain bounded exhaustive truth before reporting precision, specificity,
  false-positive rate, average precision, or calibrated neuron probability.
- Test the frozen rules on an independent recording before claiming
  generalization.

### Repository and release

- Repair the missing portable evidence capsule for NREV-EXP-0033.
- Repair native-draft registry ordering and the remaining generated API
  reference drift before claiming a clean full-suite pass.
- Keep Feature Atlas excluded unless it is registered, evidence-bound, and
  media-audited.
- Publish a versioned software/data release with a clean isolated install,
  portable example data, environment lock, persistent identifiers, and a
  documented regeneration command.

### Author and journal

- Confirm all affiliations and corresponding-author information.
- Complete ethics, animal protocol, funding, acknowledgements, conflicts, and
  CRediT roles.
- Verify every citation, author name, year, title, journal field, and DOI.
- Approve the final data/code availability statement and generative-AI
  disclosure.
- Disable provisional mode and remove every red author-decision field for the
  release build.
