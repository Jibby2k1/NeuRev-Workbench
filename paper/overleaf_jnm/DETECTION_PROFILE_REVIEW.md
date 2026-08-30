# Detection-profile result: author review guide

## What was analyzed

The strict, object-separated budget-20 union from the coherence and lag lanes
contains 86 detection occurrences. Repeated spatial locations were consolidated
into 36 detection sites. An occurrence is one detected event in one burst; a site
is a recurring image location and must not yet be called a neuron.

## What the model used

Each occurrence has 11 label-free measurements: native peak, residual peak,
robust signal-to-noise ratio, spatial specificity, annulus coupling, signed event
area, relative peak time, lane agreement, candidate rank, cross-burst recurrence,
and quiet intensity. Expert labels were not used to fit or choose the classes.

## Frozen result

Three classes were selected from candidate solutions with two through six
classes. The solution has silhouette 0.284 and bootstrap mean adjusted Rand index
0.724. Every class occurs in every one of the four bursts.

1. **Localized, high-signal, recurrent:** 35 occurrences at 12 sites; median
   native peak 618.7, residual peak 280.4, and spatial specificity 0.319.
2. **Broad, low-signal, early/episodic:** 8 occurrences at 6 sites; median native
   peak 269.5, residual peak 94.0, and spatial specificity 0.170.
3. **Broad, lower-signal, recurrent:** 43 occurrences at 24 sites; median native
   peak 271.6, residual peak 97.2, and spatial specificity 0.192.

## What to visually validate

- In `figures/final/detection_class_overview.png`, confirm that the three classes
  are visually distinguishable and that the spatial map is understandable.
- In `figures/final/detection_class_profiles.png`, confirm that the class names
  agree with the standardized feature patterns.
- In `figures/final/detection_class_representatives.png`, confirm that the crop
  and trace examples look like plausible representatives of the stated classes.
- In the Results subsection, confirm that “detection-event archetypes” is the
  right biological level of language.

## Interpretation boundary

These classes are not cell types, and the current data do not prove that every
site is a neuron. Sparse known-positive proximity was calculated only after the
classes were frozen. Therefore 28/35, 3/8, and 23/43 are descriptive overlaps,
not class precision, specificity, or false-positive rates.

## Certainty and stability extension

- Frozen-v1 coverage: 10/19 confirmed observations matched, versus 0/5
  identity-uncertain observations. This is suggestive but small ($p=0.053$).
- Canonical-v7 coverage: 34/54 confirmed versus 10/22 identity-uncertain
  observations matched. This difference was not established ($p=0.203$).
- Among matched v7 observations, uncertain identities had 0 Class 1, 4 Class 2,
  and 6 Class 3 occurrences; confirmed identities had 8, 2, and 24.
- Within repeated sites, 42/50 ordered class transitions remained in the same
  class. This supports recurring measurement regimes, not immutable neuron types.

Please review `figures/final/detection_class_extensions.png` and decide whether
“identity certainty” is the clearest author-facing term for the adjudication
distinction.

## Advanced extensions

- Class-assignment margin did not differ between confirmed and identity-uncertain
  matched observations. Uncertainty therefore does not appear to be merely a
  cluster-boundary effect.
- Ambiguous or non-neuronal reviewer morphology was absent from Class 1 and was
  concentrated in Classes 2 and 3.
- The provisional field comparison is site-level: only 3/36 sites were left of
  $x=286$. Two were Class 2, so the spatial association is interesting but fragile.
- Burst composition did not materially drift. Eight canonical-v7 identities had
  at least two matched bursts; most retained one dominant class.
- `generated_analysis/ambiguous_class_review.tsv` contains the 22 lowest-margin
  occurrences for optional visual review. Reviewing these does not change the
  frozen taxonomy unless a new explicitly versioned analysis is authorized.
