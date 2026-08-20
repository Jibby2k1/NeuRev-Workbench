# Spon Ca Burst ROI-minus-annulus signature v1

## Final decision

The final guarded signature diagnostic assigned **S0 — no reproducible
ROI-specific residual signature** and wrote `stop_signature_branch`.
Multi-time-step ICA is not authorized by this branch.

## Method

The amplitude-preserving signal was the 2 px-radius ROI mean minus the fixed
3–6 px local-annulus mean. Shape comparisons used only pre-event median
centering and MAD scaling; peak amplitudes remained in original intensity
units. Five deterministic ROI-held-out folds were compared with:

1. 199 same-ROI windows shifted outside every labeled burst neighborhood; and
2. a matched spatial residual formed by the 3–6 px annulus minus a 7–10 px
   outer ring at the true event time.

Uncertainty used 5,000 ROI bootstrap draws and 9,999 sign-flip draws.

## Results

| Measurement | Result |
|---|---:|
| Mean held-out residual-template correlation | 0.5307 |
| Median held-out residual-template correlation | 0.6024 |
| Median within-ROI repeatability | 0.3144 |
| Correlation delta versus same-ROI shifts | 0.5302 |
| Temporal-null 95% CI | [0.3819, 0.6602] |
| Temporal-null p | 0.0001 |
| Correlation delta versus spatial residual | -0.0953 |
| Spatial-control 95% CI | [-0.2560, 0.0472] |
| Spatial-control p | 0.8789 |
| Event peak residual-amplitude delta | +91.33 intensity units |
| Amplitude-delta 95% CI | [50.64, 140.75] |
| Amplitude-delta p | 0.0001 |

The residual is genuinely larger during labeled intervals than during ordinary
times. However, its temporal morphology does not generalize better than the
neighboring annulus-minus-outer-ring control, and repeated events within the
same ROI are not sufficiently consistent. Larger event-time amplitude is not
evidence for a neuron-specific waveform.

## Interpretation

Local background subtraction reduced the shared Raw waveform but did not
isolate a reproducible candidate-neuron signature. The remaining event-time
structure is still spatially shared or heterogeneous at this support scale.
This may include broad neural recruitment, motion, neuropil fluorescence, or
other nuisance structure; this experiment does not distinguish those causes.

Two-frame ICA remains a Level 1 derivative-like sanity check. Neither the Raw
trace nor the fixed local residual supports S1–S3 signature language. The
sequential branch therefore stops before multi-time-step, spatial, or
spatiotemporal ICA expansion.

## Boundaries

- One recording and four shared burst intervals.
- Labels define evaluation windows but did not fit the frozen two-frame ICA.
- Sparse positives do not establish precision or false-positive rate.
- Patch-minus-annulus subtraction may suppress spatially broad neural activity.
- The full scientific-audit media inventory remains incomplete because no
  full-field candidate detector was promoted.

## Artifacts

- `Outputs/UnsupervisedICAEval/roi_minus_annulus_signature_v1/summary.json`
- `Outputs/UnsupervisedICAEval/roi_minus_annulus_signature_v1/tables/roi_residual_generalization_metrics.csv`
- `Outputs/UnsupervisedICAEval/roi_minus_annulus_signature_v1/figures/residual_signature_generalization.png`
- `Outputs/UnsupervisedICAEval/roi_minus_annulus_signature_v1/REPORT.md`
