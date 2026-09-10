# Experiment inclusion and result map

Date: 2026-09-07

Status: author-facing map for the two identity-safe venue drafts. This document
organizes the evidence; it does not supersede the research registry, source
artifacts, or `macros/venue_results_manifest.json`.

## Main-paper result ledger

| Evidence block | Unit and population | Endpoint or protocol | Current result | Evidence status | Main presentation |
| --- | --- | --- | --- | --- | --- |
| Frozen cohort | One recording; four bursts; 106 occurrence geometries at 50 immutable sites | Identity-preserving occurrence/site contract | 106 occurrences; 50 sites | Current-recording cohort | Figure 1; Table 1; Methods |
| Known-positive recovery | 106 confirmed occurrences | Any-lane union across three independently truncated per-burst B58 lists | 94/106 (88.7%); 12 misses | Current-recording sensitivity; not precision | Abstract; Figure 3A; Results |
| Canonical-collapsed sensitivity | 102 occurrence rows after the prespecified ROI-010/015 grouping view | Separate sensitivity analysis at the same per-lane B58 operating point | 93/102 (91.2%) | Secondary analysis; does not replace the primary geometry | Abstract; Figure 3A; Results |
| Temporal retrieval | Confirmed sites and event frames versus guarded quiet-reference frames | Site-weighted frame ROC AUC with site bootstrap | Raw 0.931, 95% interval 0.906--0.952; carrier 0.920; coherence 0.915; lag-2 recurrence 0.905 | Same-recording retrieval; quiet is not biological negative truth | Figure 3B; Results |
| Spatial displacement | Six compact features at confirmed centers versus baseline-matched displaced tissue | Target-minus-displaced frame-AUC with site bootstrap | All six intervals are positive | Localized measurement structure; not biological specificity | Figure 3C; Results |
| Trace morphology | 50 immutable sites, burst-averaged within site | Exploratory, shape-only joint taxonomy | T1: 38 sites; T2: 12 sites | Provisional; phenotypes are not neuron types | Figure 2C; Results |
| Candidate review | 18 detector-selected previously unmatched sites | One blinded reviewer; four normalized call classes | 9 definite, 4 probable, 4 uncertain, 1 unlikely/artifact-or-noise; 13/18 likely-neuron calls | Provisional selected-batch yield; not precision or 13 validated identities | Abstract; Figure 4; Results |
| Review priority | Same 18 selected sites | Priority score against the single-reviewer provisional dichotomy | Descriptive AUC 0.308 | Review-value score, not neuron probability | Figure 4 note; Results |
| Identity-aware miss audit | 12 all-lane per-burst B58 misses | Six-pixel identity guard after local response search | 2 same-canonical-identity collisions; 6 other-labeled-identity collisions; 4 identity-clear hypotheses | Reviewed current-recording interpretation | Abstract; Figure 5A; Results |
| Candidate availability | Four identity-clear hypotheses | Any audited lane within eight pixels through per-burst B100 | 0/4 available | Proposal absence in frozen outputs; not biological absence | Figure 5C; Results |
| Identity diagnostics | Eight collision and four identity-clear cases | Cross-stage shift cosine and neighboring-trace association | Direction cosine 0.972 versus 0.448; multi-neighbor variance explained 0.523 versus 0.088 | Small descriptive groups; permutation results non-small | Figure 5B--C; Results |
| Long-delay transform audit | 50 audited sites | Six-component embedding inversion | Maximum sitewise RMS 5.68 x 10^-12 | Numerical implementation support; not source separation | Results; Table 1 |
| Local-standardization audit | 50 centers over 560 frames | Reproduction of saved center traces | RMSE 6.96 x 10^-7; maximum error 7.63 x 10^-6; scale floor inactive at audited centers | Numerical implementation support; not full-field behavior | Results; Table 1 |
| Factorial exact truth | 432 simulated movies | One-to-one identity-matched F1 at budget 4 | Combined stack leads the compact-budget comparison | Computational mechanism evidence; not biological transfer | Figure 6A; Results |
| Held-family exact truth | 108 simulated movies across six held generator families | One-to-one identity-matched F1 | Spatial context leads the equal-weight combined stack in all six families | Computational mechanism-shift evidence | Figure 6B; Results |
| Challenge suite | Twelve exact-truth intervention families | Morphology, stopping, close-neighbor, noise, motion, stability, and negative controls | Supports complementary feature roles; exposes weak-neighbor and deployable-stopping limits | Computational mechanism evidence | Figure 6C--D; Results |

The B58 and B100 labels above are per burst and per lane. They must never be
described as one combined global list. The Feature Atlas used a different
candidate universe and a different global reranking endpoint; because it is not
canonically registered and its media audit is incomplete, its numerical values
are deliberately absent from this ledger and both manuscripts.

## Experiment placement

| Bucket | Registry experiments or program | Treatment in the two drafts |
| --- | --- | --- |
| Main current-recording evidence | NREV-EXP-0001--0008, NREV-EXP-0016, NREV-EXP-0018, NREV-EXP-0019 | Headline NREV-EXP-0005 recovery, NREV-EXP-0007 identity audit, and NREV-EXP-0016 candidate review. Use NREV-EXP-0001--0003 for measurement context. Condense the remaining diagnostic analyses. |
| Exact-truth simulations | NREV-EXP-0009--0015 | Synthesize the role and failure-boundary results in Figure 6. Preserve simulation-versus-biological boundaries and include the stopped/negative branches as evidence against overfitting the story. |
| Implementation and review tooling | NREV-EXP-0017 plus the identity/geometry, artifact, registry, and story checks | Reproducibility support in the Journal of Neuroscience Methods draft; main software-validation evidence in the Neuroinformatics draft. Implementation readiness is not completed human review. |
| Exploratory engineering | NREV-EXP-0021, NREV-EXP-0025, NREV-EXP-0028--0033, plus the unregistered Feature Atlas run | Exclude from confirmed main Results. Retain only a bounded decision-history note. Feature Atlas requires registration, portable evidence, and media audit before reconsideration. |
| Planned validation | NREV-EXP-0020, NREV-EXP-0022--0024, NREV-EXP-0026--0027 | Limit to limitations and future work. Do not present a plan, staged package, or submitted job as an empirical result. |
| Companion representation paper | ICA/whitening factorial and real-data programs; PC-MITL E01/E02 and specialized confirmation | Keep separate because the protocols, truth conditions, and evaluation units differ from the identity-safe paper. |

## Authority and refresh order

1. Change scientific state only in the canonical registry or originating
   analysis artifact.
2. Regenerate the repository story and require `make story-check` to pass.
3. Regenerate `macros/venue_results.tex` and its source-hash manifest.
4. Regenerate the venue figures and their source/output-hash manifest.
5. Rebuild both deterministic Overleaf packages and rerun the venue package
   tests before editing either abstract.

