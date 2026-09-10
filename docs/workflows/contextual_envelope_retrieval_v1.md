# Contextual-envelope retrieval comparison v1

- Registry experiment: `NREV-EXP-0032`
- Status before execution: frozen bounded within-recording comparison
- Parent semantics pilot: `NREV-EXP-0031`

## Question

Does agreement-attenuated evidence improve confirmed-event localization over
instantaneous evidence or its amplitude-square control, and is any improvement
specific to an aligned causal envelope rather than an amplitude-matched
temporally shuffled or spatially displaced envelope?

## Frozen sources and calibration

Evaluate three aligned sources over UI 1800--2359:

1. Raw fluorescence;
2. current `ICA -> LS` evidence;
3. `TMax5 -> LS -> ICA` temporal-envelope evidence.

For each source, reconstruct the exact global quiet calibration from the v1
pilot: subtract the first-100-frame sampled median, divide by its positive
99.5th-percentile span, and clip to `[0,1]`. No pixelwise calibration is used.

## Frozen feature arms

For positive instantaneous evidence `A`, evaluate:

- `A`;
- `A2 = A^2`, the amplitude-only control;
- `H_t3 = A^2/U_3,1`, 60-ms temporal agreement attenuation;
- `H_t5 = A^2/U_5,1`, 100-ms temporal agreement attenuation;
- `H_st5k3 = A^2/U_5,3`, 100-ms plus 3-by-3 spatial attenuation;
- `H_t5_shuffle137 = A^2/max(A,roll_t(U_5,1,137))`, a deterministic
  marginal-preserving temporal misalignment control;
- `H_st5k3_displace11x17 = A^2/max(A,roll_yx(U_5,3,11,17))`, a deterministic
  displaced-envelope control.

The `max(A, control envelope)` construction preserves the attenuation bound
`0 <= H <= A` even for intentionally misaligned controls.

## Population, estimand, and grouping

- Population: all 106 canonical-v7 confirmed occurrences at 50 immutable sites.
- Primary estimand: each event-window maximum's percentile among all
  same-duration guarded quiet-window maxima at the same site.
- Quiet windows exclude all four burst intervals plus a 15-frame guard.
- Bootstrap unit: immutable site, 5,000 deterministic resamples.
- Secondary summaries: event-to-quiet robust effect, event-energy fraction,
  burst medians, and cross-burst site-rank repeatability.
- Quiet windows are temporal references, not verified biological negatives.

## Frozen decision gates

Feature advancement is source-specific and requires all of:

1. site-bootstrap mean localization percentile lower bound above 0.5 and every
   burst median at least 0.75;
2. paired delta versus `A2` has a 95% lower bound above zero;
3. paired delta versus its matched shuffled/displaced control has a 95% lower
   bound above zero;
4. no burst median is more than 0.02 below `A2`;
5. finite exact coverage of 106 occurrences and 50 sites.

`H_t3` and `H_t5` compare with the temporal shuffled control. `H_st5k3`
compares with the spatially displaced control. Failure holds that arm; it does
not reject contextual-envelope features in general.

## Scientific audit and claim boundary

This comparison creates no candidate detector or new coordinates. It reuses
the validated canonical-v7 expert media and generates one trace comparison per
confirmed occurrence plus figure/table comparisons. Model annotations are not
applicable. Unmatched activity remains unknown. Even a passed feature gate is
within-recording exploratory evidence, not precision, specificity, biological
identity, or independent-recording validation.
