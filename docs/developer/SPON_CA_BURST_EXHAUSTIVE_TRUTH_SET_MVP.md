# Spon Ca Burst Exhaustive Truth-Set MVP

Status: implementation contract. This is not a detector-performance report and does not authorize a real-data or GPU run.

## Decision

NeuRev uses the existing annotation-revision and Review infrastructure. A separate truth-set manifest owns region selection, coverage, blinding, frozen candidate provenance, protected locks, and A0 authorization. Published annotation revisions remain the label source; archived annotation schemas are not migrated silently.

Raw-first discovery is always completed and locked before an opaque, deterministically randomized union of candidates becomes browser-readable. The source-lane key stays in `private/` and is never part of the Review payload. Unmatched and unresolved records remain unknown; precision-oriented metrics require explicit exhaustive coverage.

## Requirement matrix

| Roadmap requirement | Existing owner | Current support | Missing behavior | Planned change | Test |
|---|---|---|---|---|---|
| Bounded calibration and protected regions | dataset/view contracts | Source dimensions and coordinates | Region roles, masks, non-overlap | Truth-set region schema/model | schema, bounds, overlap tests |
| Detector-independent raw-first review | Review correction surface | Raw views, traces, manual ROI creation | Pass state and enforced lock | Raw-first payload with no candidates; lock gate | raw-first missed-object and lock tests |
| Detector-blinded candidate completeness | model proposal packages | Unknown proposals and expert-field stripping | Multi-lane opaque union | Deterministic union and private source key | union and leakage tests |
| Deterministic candidate randomization | review batches | Stable queue ordering | Seeded opaque order | Frozen randomization seed | repeatability test |
| Separate object and event labels | annotation schema | ROI/event maps | Truth-set-specific records | Object/event validators and separate TSVs | record and relationship tests |
| Explicit exhaustive coverage | robustness plan | Documentation only | Machine authorization | Region `coverage_mode`; metric gate | non-exhaustive rejection test |
| Unresolved-state retention | agreement/review queues | `unsure` and disagreement | Evaluation-safe unresolved state | Dedicated disposition and TSV | unresolved exclusion test |
| Second review and adjudication | `neurobench.review.agreement` | Agreement and disagreement queue | Frozen 20% sampling | Stratified deterministic sampler; forced unresolved/disagreements | sampling/agreement test |
| Protected-region locking | experiment manifests | General frozen-run conventions | Machine lock/unseal chain | Frozen lane manifest and immutable unseal record | precondition and tamper tests |
| Frozen-lane provenance | architecture/audit manifests | Run IDs and artifacts | Exact candidate-policy fingerprint | Lane, score, policy, mask and code checksums | checksum test |
| Coverage-aware metric authorization | detection/event metrics | Object/event matching | Coverage gate | `neurobench.metrics.truth_set` | precision/AP rejection and Recall@K tests |
| Reviewer time and efficiency | review stats | Action counts | Minutes per accepted item | Review-efficiency output | calculation test |
| Immutable publication and revision provenance | annotation revisions | Collision refusal, append-only operations, published children | Truth-set references and A0 validation | Revision references and immutable package root | collision and revision-reference tests |

## State and ownership

The manifest and region schemas live in `schemas/`. `neurobench.models.truth_set` validates frame/coordinate invariants. `neurobench.review.truth_set` owns deterministic union, blinding audit, and second-review sampling. `neurobench.workbench.truth_set` owns collision-safe packaging and A0. `neurobench.metrics.truth_set` is the coverage authorization boundary. The existing server exposes a read-only truth-set Review payload.

The initial MVP is server-readable and deliberately stops short of a partial parallel UI. A later Review subpage may consume the payload while reusing annotation operations, autosave, traces, reviewer provenance, and immutable publication.

## A0 meaning

`advance` requires explicit coverage, locked raw-first discovery, complete candidate dispositions, retained unresolved records, completed second review/adjudication, a freeze preceding unseal, valid references/checksums, and no blinding leak. It authorizes protected evaluation only. It never means a detector passed.

## Human-required production steps

The committed example is synthetic. A human must choose production calibration and label-free protected regions, confirm stored lane artifact contracts, review raw-first objects/events, finish blinded dispositions, provide an independent second review, adjudicate disagreements, and explicitly unseal the protected set. No production coordinates are inferred here.
