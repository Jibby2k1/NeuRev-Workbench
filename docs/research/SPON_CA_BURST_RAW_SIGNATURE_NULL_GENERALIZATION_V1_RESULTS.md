# Spon Ca Burst Raw signature null/generalization v1

## Decision

The candidate Raw activation signature received **S0 — no reproducible
ROI-specific signature**. Event timing clearly identifies a recurring waveform,
but the same waveform is present in the matched local annulus. The evidence is
therefore consistent with a spatially shared burst/global signal and does not
support a neuron-specific morphology claim.

## Frozen assay

- 79 canonical inclusive occurrences and 26 canonical ROIs.
- Five deterministic ROI-held-out folds.
- Fixed 87-frame windows: 20 pre-event, the maximum declared 47-frame event
  duration, and 20 post-event frames.
- Baseline-median centering and pre-event MAD scaling only; no peak alignment or
  peak-amplitude normalization.
- 199 same-ROI shifted windows outside annotated burst neighborhoods.
- Matched 3–6 px spatial annulus at the true event time.
- 5,000 ROI bootstrap draws and 9,999 sign-flip draws.

## Results

| Measurement | Result |
|---|---:|
| Mean ROI-held-out event-template correlation | 0.8256 |
| Median ROI-held-out event-template correlation | 0.9330 |
| Median within-ROI pairwise event correlation | 0.3720 |
| Event minus same-ROI shift | 0.8249 |
| Shift-control 95% ROI-bootstrap CI | [0.7267, 0.9045] |
| Shift-control sign-flip p | 0.0001 |
| Event minus matched annulus | -0.0131 |
| Annulus-control 95% ROI-bootstrap CI | [-0.0277, 0.0008] |
| Annulus-control sign-flip p | 0.9523 |

The temporal null passed strongly: annotated intervals differ from ordinary
same-ROI times. The spatial-specificity gate failed: annulus traces predict the
held-out event template just as well as the ROI traces. Within-ROI repeatability
also remained below the preregistered `0.5` median-correlation threshold.

## Interpretation

The high event-aligned average from the previous assay was real as a
recording-level phenomenon, but it was not localized to the candidate neuron
support. This is exactly why an attractive population heatmap was insufficient.
No S1, S2, or S3 neuron-signature language is justified.

This result does not mean the labeled ROIs contain no neural activity. It means
that this fixed patch-average Raw trace and event alignment cannot separate a
neuron-specific temporal morphology from local shared fluorescence, motion, or
other common-mode structure.

## Gate outcome

Do not launch multi-time-step ICA from this signature claim. The next bounded
diagnostic, if selected, should test a preregistered amplitude-preserving
ROI-minus-annulus residual against the same nulls. It must remain an analytic
diagnostic, not a tuned detector. If spatial specificity still fails, stop the
signature branch.

## Final residual follow-up

The fixed ROI-minus-annulus residual also failed spatial specificity and wrote
`stop_signature_branch`. Although event-time residual amplitude exceeded
shifted windows, residual morphology did not outperform the matched
annulus-minus-outer-ring control. See
`SPON_CA_BURST_ROI_MINUS_ANNULUS_SIGNATURE_V1_RESULTS.md`.

## Artifacts

- `Outputs/UnsupervisedICAEval/raw_signature_null_generalization_v1/summary.json`
- `Outputs/UnsupervisedICAEval/raw_signature_null_generalization_v1/tables/roi_generalization_metrics.csv`
- `Outputs/UnsupervisedICAEval/raw_signature_null_generalization_v1/figures/signature_generalization.png`
- `Outputs/UnsupervisedICAEval/raw_signature_null_generalization_v1/REPORT.md`
