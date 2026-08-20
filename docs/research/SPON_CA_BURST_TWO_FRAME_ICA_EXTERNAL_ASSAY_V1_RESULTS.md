# Spon Ca Burst two-frame ICA external assay v1

## Outcome

The hash-frozen, label-free two-frame representation was evaluated without
refitting on the 79 inclusive occurrences in the canonical hard-ROI
adjudication table (26 canonical ROIs after identity merging). The discovery
fit used frames `[0,560)`; labeled bursts occur later in the recording.

Both the frozen transient ICA coordinate and ordinary signed adjacent-frame
difference were enriched at the labeled intervals relative to 199 shifted-label
controls:

| Lane | Mean true-interval score | Mean shifted score | Empirical upper p |
|---|---:|---:|---:|
| Frozen two-frame ICA | 0.8848 | 0.4048 | 0.005 |
| Signed difference | 64.1235 | 31.3531 | 0.005 |

ICA and difference occurrence scores correlated at `0.9988`. Together with the
Stage A operator fit (`R2=0.9938` against signed difference), this supports a
Level 1 sanity-check result, not a novel Level 2 or Level 3 representation:
ICA recovered event-associated temporal change, but did not add information
beyond the analytic difference baseline.

The amplitude-normalized Raw windows showed strong descriptive consistency:
leave-one-ROI-out template correlation was `0.8715`, and the first temporal PCA
mode explained `0.8739` of between-ROI template variance. This is a candidate
signature worth testing, not yet S2 evidence. Matched no-event windows,
same-ROI temporal shifts, uncertainty intervals, and a held-out template test
must pass before assigning a signature evidence level.

## Boundaries

- Labels were external to fitting and frozen model selection.
- Sparse-positive labels do not identify precision, false-positive rate, or
  specificity. Unmatched activations remain unknown.
- This ROI-centered assay does not yet contain a frozen full-field candidate
  panel or the complete scientific-audit video/close-up inventory.
- Multi-lag, spatial, and spatiotemporal ICA were not run.

## Artifact locations

- Stage A: `Outputs/UnsupervisedICAEval/two_frame_stage_a_v1/`
- External assay: `Outputs/UnsupervisedICAEval/two_frame_external_assay_v1/`
- Experiment manifests: `experiments/unsupervised_ica_eval/`
- Maintained code: `neurobench/experiments/unsupervised_ica_eval/`

## Decision

Keep two-frame ICA as an interpretable control and use standardized signed
difference as the simpler equivalent baseline. The next justified experiment
is a bounded, preregistered Raw candidate-signature null/generalization assay.
Only if that signature generalizes beyond matched same-ROI null windows should
the program test a small multi-time-step temporal ICA model.

## Follow-up gate result

The follow-up Raw signature null/generalization assay is complete and assigned
S0. Event windows strongly exceeded same-ROI shifted windows, but did not exceed
matched local-annulus event windows. Multi-time-step ICA is therefore not
authorized by this signature gate. See
`SPON_CA_BURST_RAW_SIGNATURE_NULL_GENERALIZATION_V1_RESULTS.md`.
