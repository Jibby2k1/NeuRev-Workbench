# ICA/Whitening Real-Data v1 Final Results

**Status:** complete computational and scientific-audit gates, reconciled from
the saved small artifacts on 2026-09-09. The final independent-recording endpoint
is **20/51 known-positive occurrences (0.392157) at budget 58 per block**, for each
of three refit seeds on `15 right`. This is sparse-positive retrieval evidence
from one independent recording. It does not establish precision, biological
source identity, or population-level generalization.

The [historical progress record](ICA_WHITENING_REAL_DATA_V1_PROGRESS.md) preserves
the earlier execution history. The
[evaluation plan](ICA_WHITENING_HYPERPARAMETER_EVALUATION_PLAN.md) and
[workflow](../workflows/spon_ca_burst_ica_whitening_evaluation.md) describe the
design and operational contract. The earlier
[hyperparameter evaluation results](ICA_WHITENING_HYPERPARAMETER_EVALUATION_RESULTS.md)
cover the preceding synthetic and numerical work.

## Final evidence and gates

The local artifact root is
`Outputs/ICAWhiteningEvaluation/spon_ca_burst_ica_whitening_evaluation_v1/real_data_v1`.
Paths in the source column below are relative to that root; standalone basenames
on a row share the first path's directory. These generated
outputs are local evidence references and are not expected to be available in
a source-only checkout.

| Readout | Final saved result | Source artifact |
| --- | --- | --- |
| Completion | `status: complete`; `completion_claim_allowed: true`; all eight gates true; no incomplete gates | `completion_audit_latest.json`, `CONCLUDING_REPORT.md` |
| S3 factorial coverage | 30,891 unique eligible fits; no missing or extra fits; 28,928 converged; 26,333 finalist eligible | `stages/S3_COMPLETE_FACTORIAL_ANALYSIS/summary.json`, `validation.json` |
| S4 protected confirmation | One selected configuration, 12/12 refits across four held-out bursts and three seeds | `stages/S4_FINALIST_CONFIRMATION/summary.json`, `validation.json` |
| S4 protected B58 retrieval | Macro held-out-burst recall 0.300319; burst-bootstrap 95% interval [0.091667, 0.508972] | `stages/S4_FINALIST_CONFIRMATION/summary.json` |
| S5 scientific audit | One of one finalist packages passed inventory, decode, and recorded visual inspection; `promotion_allowed_by_audit: true` | `stages/S5_SCIENTIFIC_AUDIT/validation.json` |
| S6 independent confirmation | Exact coverage for one finalist and three seeds; 18,689 candidates; all scores frozen before the label join | `stages/S6_INDEPENDENT_CONFIRMATION/summary.json`, `validation.json`, `candidate_manifest.json` |
| S6 B58 retrieval | 20/51 = 0.392157 for every seed; mean reciprocal rank across seeds 0.113933 | `stages/S6_INDEPENDENT_CONFIRMATION/summary.json`, `sparse_positive_metrics/icaw_bf39743bcf6a7d4f__seed_{7,104736,209766}.json` |

The completion gates cover the real-data contract, amended non-blocking
synthetic failure, exact factorial coverage, exclusion of superseded mixed
spatial artifacts, conditional analysis and interpretability, protected
confirmation, all-finalist visual audit, and independent confirmation coverage.
This is completion of the specified evaluation and audit. It is not a general
detector-superiority or manuscript-submission gate.

Some intermediate snapshots intentionally remain unchanged. S4's saved
`promotion_status` still says `pending_scientific_audit`, and S5's `summary.json`
still says `rendered_pending_visual_inspection`. The subsequent S5
`validation.json`, final completion audit, and concluding report record the
passed audit. Use those final validation artifacts for the current gate state.

## Protected within-recording result

Configuration `icaw_bf39743bcf6a7d4f` was selected in every held-out-burst fold
using the other bursts. Held-out event samples were excluded from fitting under
the S4 run manifest, and each burst's labels were used only after its candidate
score hash was frozen. Its macro recall of 0.300319 averages the burst-level
means from three seed refits; the seeds are robustness checks, not independent
biological replicates. The wide interval is based on only four burst units in
the development recording.

The instantaneous-response screen was superseded by the event-standardized
scorer. Likewise, the mixed-implementation spatial run is excluded from
scientific use; only the clean rerun contributes to the final factorial.
Those historical outputs remain preserved. S3 factor associations are
descriptive and stratified by ICA family and whitening geometry. The scrambled
Sobol point design does not identify formal Sobol sensitivity indices or causal
factor effects.

## Independent recording and exact endpoint

The frozen independent contract uses `15 right` at 50 Hz, with 51 annotated
positive intervals from 10 ROIs. It reuses the S4-selected hyperparameters and
refits whitening and ICA on the independent recording without labels. It does
not transfer component identities. The eligible-recording preflight and
candidate contract are stored next to `real_data_v1` under
`independent_confirmation_preflight_v1/independent_confirmation_preflight.json`
and `independent_confirmation_contract_v1/contract.json`.
The eligibility preflight enumerated ROI and interval metadata; label isolation
here refers to candidate construction, model fitting, and score freezing after
that eligibility review.

Candidates came from a frozen union of Raw activity, signed temporal
difference, and spatial high-pass lanes in one-second blocks. The contract uses
a 21-frame score window centered on each label-blind lane peak, LME temperature
0.25, a six-pixel spatial match radius, and a candidate peak frame inside the
manual positive interval. Its initial 100 reference frames are not asserted to
be event-free. The candidate manifest records 18,689 candidates with digest
`cb2947778be95f7330e5055cd6969a59a09496f9e9cec508e44e7c7401765393`.

For seeds 7, 104736, and 209766, each metric artifact records exactly 20 matches
among 51 known-positive occurrences at **B58 per block**. Thus the mean pooled
recall is also 20/51, not 60 independent recovered occurrences. Macro recall
over the 17 blocks containing positives is 0.470588; it uses a different
denominator and should not replace the pooled 0.392157 endpoint.

The score-freeze hashes in the three sparse-positive metric files agree with
the corresponding `label_blind_scores` metadata, and S6 validation reports no
missing or extra model, score, or metric stems. This records the saved
label-access order; the present documentation reconciliation did not rerun
fitting or independently repeat the earlier media inspection.

## Keep the Gamma-LS result separate

The [Gamma-LS final results](SPON_CA_BURST_GAMMA_LS_PAPER_SUCCESS_RESULTS_2026_09_09.md)
report **6/51 (0.117647) at B58 per block** on `15 right`. Its independent metric
root is
`Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_independent_15_right_v1_20260908_r1`.
That run uses a frozen signed-difference representation with radial Gamma-LS,
6,400 native candidates across 32 complete one-second blocks, and its own
calibration and scoring procedure. Its eligibility preflight inspected
annotations before the subsequent label-isolated candidate-scoring phase.

**ICA 20/51 versus Gamma-LS 6/51 is not a head-to-head comparison.** The common
recording name, positive count, and nominal B58 budget do not make the proposal
universes, scoring windows, fitting/calibration procedures, or evaluation
contracts identical. These endpoints establish results under their respective
frozen protocols. A method-superiority claim requires a matched, predeclared
comparison and cannot be inferred from their difference.

## Interpretation limits

- Unmatched candidates remain `unknown_not_negative`. Precision, specificity,
  false-positive rate, and exhaustive neuron recovery are unidentified.
- S4 did not pass the individual-component stability gate in every fold:
  held-out burst 1 had worst aligned component cosine 0.221459. Stable retained
  subspaces do not establish stable individual sources. S6 seed alignment
  passed for this independent refit, but that does not establish biological
  identity or component identity across recordings.
- S5 validation records review of 182 PNGs and representative frames from all
  105 MP4s, with no decode failures. Fixed robust display limits visibly
  saturated during some large transients. Those panels support localization
  and stage inspection; quantitative amplitude and fine morphology require
  the source statistics and traces.
- Synthetic truth recovery did not validate biological source identity. Its
  failure remains an explicitly amended, non-blocking warning for the
  real-data evaluation, not a successful recovery result.
- One independent recording and repeated numerical seeds do not establish
  cross-animal, acquisition-wide, or population-level generalization.
- The centered candidate-score window is an offline retrieval readout. Causal
  whitening alone does not make this evaluation an online detector benchmark.

The supported conclusion is a completed, auditable evaluation with protected
within-recording retrieval and a frozen one-recording independent retrieval
endpoint. Biological identity, exhaustive detection accuracy, and superiority
over a different pipeline remain outside that conclusion.
