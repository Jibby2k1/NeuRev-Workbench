# Spon Ca Burst full-trace feature panel

## Purpose

This analysis asks whether compact, interpretable features concentrate confirmed
canonical-v7 burst occurrences above same-duration quiet windows when evaluated
over the complete 560-frame aligned recording. It complements, rather than
replaces, the earlier event-window diagnostic audit and protected candidate-
ranking audit.

The run is exploratory. The recording and its four bursts have informed prior
method development, so the analysis cannot serve as independent confirmation.
Unmatched or off-window activity is not treated as a verified biological
negative.

## Immutable data contract

- Input population: canonical-v7 `include_confirmed=true` observation rows.
- Observation grain: one confirmed burst occurrence at one immutable spatial
  geometry.
- Resampling grain: `original_roi_id + rounded x/y center`; canonical identity
  alone is never a sampling key.
- Time: 560 frames at 10 fps, UI frames 1800--2359.
- Burst intervals are one-based inclusive and are copied from the validated
  canonical-v7 media manifest.
- Quiet frames exclude every burst interval plus a 15-frame guard on each side.

## Frozen feature panel

No feature or parameter is selected from the results of this run.

1. `raw_center`: robustly quiet-standardized Raw center trace.
2. `ica_center`: robustly quiet-standardized recovery CS-Parzen ICA center trace.
3. `ls_center`: robustly quiet-standardized recovery local-standardization trace.
4. `raw_center_annulus`: Raw center minus the 3--6 px annulus mean, robustly
   quiet-standardized.
5. `vst_center_annulus`: generalized-Anscombe Raw center minus transformed
   annulus mean, using the previously frozen descriptive noise fit.
6. `matched_filter_ls`: fixed one-second exponential matched filter applied to
   the positive quiet-standardized LS trace.
7. `carrier_signed`: frozen amplitude-preserving detector carrier.
8. `coherence_w15`: exact deterministic 15-frame causal local coherence,
   reconstructed from the frozen carrier.
9. `propagation_lag2_w15`: exact deterministic 15-frame, two-frame-lag local
   recurrence feature, reconstructed from the frozen carrier.
10. `representation_consensus`: minimum of the Raw, ICA, and LS empirical
    quiet-CDF scores at each frame; a conservative agreement feature.
11. `multiscale_persistence`: geometric mean of causal 3-, 7-, and 15-frame
    moving averages of positive quiet-standardized LS.

Features 1--9 are established signal-processing or previously frozen NeuRev
features. Features 10--11 are prespecified interpretable extensions. They are
not optimized in this run.

## Estimands

For each occurrence and feature:

- `event_localization_percentile`: percentile of the event-window maximum among
  all valid overlapping quiet-window maxima of the same duration;
- `event_quiet_effect`: robust standardized difference between the event maximum
  and quiet-window maxima;
- `event_energy_fraction`: positive feature energy inside the union of all four
  burst intervals divided by positive energy over the complete recording.

Feature summaries report immutable-site bootstrap intervals, burst-specific
medians, and paired site-bootstrap differences from `carrier_signed`. Scalar
repeatability is summarized by the median pairwise Spearman correlation across
burst pairs with at least five shared immutable sites.

## Interpretation gates

- `supported_full_trace_localization`: site-bootstrap mean event-localization
  percentile has a 95% interval wholly above 0.5 and every burst median is at
  least 0.75.
- `incremental_over_carrier`: paired site-bootstrap mean percentile difference
  from `carrier_signed` has a 95% interval wholly above zero.
- `repeatable_site_ordering`: median eligible burst-pair Spearman correlation is
  at least 0.30.

These gates organize exploratory evidence; they do not convert the run into an
independent confirmatory experiment.

## Required outputs

The collision-safe run root contains occurrence metrics, feature summaries,
burst-pair repeatability, bootstrap draws, a comparison figure, small JSON
indexes, source hashes, an analysis report, and validation. Scientific-audit
media are satisfied by immutable references to the already validated v7
full-trace and close-up media; the feature panel creates no new detections or
model annotations.
