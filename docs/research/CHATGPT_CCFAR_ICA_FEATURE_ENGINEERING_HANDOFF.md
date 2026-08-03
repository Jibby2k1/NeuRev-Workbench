# ChatGPT handoff: continuous-CFAR ICA feature engineering

Date: 2026-08-03

## Purpose

This note packages a completed event-weighted two-frame CS-Parzen ICA study
and a proposed follow-up representation for discussion with ChatGPT. The
follow-up preserves raw fluorescence while using continuous local
spatiotemporal CFAR and ICA to construct an activity-dependent modulation map.

The central proposed representation is

```text
Raw -> continuous spatiotemporal CFAR -> ICA -> nonnegative phi -> multiply Raw
```

or, for pixel `p` and frame `t`,

```text
C_t(p) = ContinuousCFAR_ST(R)_t(p)
S_t(p) = ICA(C)_t(p)
Y_t(p) = R_t(p) * phi(S_t(p))
```

Here, `R` retains amplitude and anatomy, continuous CFAR supplies local
normalization, ICA extracts structured dynamics, and `phi` resolves ICA's
arbitrary sign and scale into a stable nonnegative gate.

## Relevant completed evidence

The completed standard event-balanced CS-Parzen study evaluated 76 fits over
four leave-one-burst-out folds:

- natural weighting: 4 fits;
- frame-balanced weighting: 28 fits;
- ROI-balanced weighting: 28 fits;
- ROI-balanced weighting with weighted whitening: 16 fits.

All fits converged, the immutable unweighted baseline parity check passed, and
held-out evaluation retained natural prevalence. Sparse labels identify known
label recall and candidate count, but not precision; unmatched candidates are
unknown.

| Configuration | Known matches | Candidates | Interpretation |
| --- | ---: | ---: | --- |
| Natural reference | 15/79 | 84 | Baseline for this diagnostic |
| Frame-balanced, alpha 0.05 | 16/79 | 89 | Small fold-local change |
| ROI-balanced, alpha 0.20 | 5/79 | 16 | Consistent angle movement but recall collapse |
| Weighted whitening, alpha 0.10 | 33/79 | 235 | Strongest exploratory result; 2.8x burden |
| Weighted whitening, alpha 0.20 | 31/79 | 236 | Similar but slightly weaker |

The alpha 0.10 weighted-whitening result was uneven: 7/15, 0/20, 16/21, and
10/23 known matches across folds, with 48, 0, 132, and 55 candidates. It is an
interesting feature-engineering clue, not evidence of improved precision or
validated source separation.

Formal Gate C failed. No moderate ROI-balanced alpha satisfied every required
angle, held-out recall, candidate-burden, effective-sample-size, fold, and
independent-seed criterion. The standard profile also contained only one
sampling seed. No spatial ICA extension was launched.

## Why change direction

Increasing event mass usually did not produce a large, stable departure from
the derivative-like global two-frame ICA solution. Weighted whitening changed
held-out activation behavior more strongly, but inconsistently and with many
more candidates. This suggests studying explicit representations instead of
asking one global ICA angle to absorb amplitude, local noise, anatomy, and
dynamics simultaneously.

There is also relevant prior negative evidence: strict multiplication by
derivative energy was too selective for slowly evolving calcium activity. In a
previous activity-gate benchmark, Raw Direct recovered 49/79 known labels,
whereas strict/floored derivative-energy gates recovered 4/79 or 16/79. The
follow-up must therefore preserve a direct-amplitude path and test a nonzero
gate floor.

## Proposed continuous-CFAR ICA representation

### 1. Raw amplitude

Retain `R` as an explicit branch. Compare literal raw fluorescence with a
positive residual above a frozen quiet baseline so that static brightness is
not silently mistaken for activity.

### 2. Continuous local spatiotemporal CFAR

Use the continuous score, not a thresholded CFAR mask:

```text
C_t(p) = (R_t(p) - mu_ref(t,p)) / (sigma_ref(t,p) + epsilon)
```

where the reference set is a local space-time neighborhood excluding a guard
region around `(t,p)`. A causal version uses only past reference frames; a
centered offline version must be identified separately. Reference contamination
by the event can self-normalize slow or spatially broad activity, so temporal
extent, spatial radius, guard geometry, and causality are primary variables.

The maintained local CFAR currently computes positive spatial contrast
independently in each frame; its temporal filter size is one. Calling the new
feature spatiotemporal therefore requires an explicit temporal reference
extension rather than relabeling the existing statistic.

### 3. ICA on continuous CFAR

Fit ICA to `C`, with the observation axis declared explicitly. Candidate forms
include adjacent-frame `[C_(t-1), C_t]` ICA and a short causal temporal window.
The recent global two-frame result makes `C` alone an essential baseline: ICA
must demonstrate incremental value beyond CFAR normalization.

### 4. Map ICA output through `phi`

ICA sign and scale are non-identifiable, so do not multiply a signed component
directly by raw fluorescence. Initial mappings should include:

```text
energy:          phi(s) = s^2
bounded energy:  phi(s) = s^2 / (tau^2 + s^2)
floored bounded: phi(s) = beta + (1-beta) * s^2 / (tau^2 + s^2)
```

The floored bounded mapping is the primary candidate. `beta > 0` preserves
slow/broad signals, while saturation limits amplification by rare ICA tails.
Scale and `tau` must be estimated from training/quiet data only.

## Minimum informative study

Use the existing four complete burst intervals as leave-one-burst-out folds,
apply the temporal guard before calibration or fitting, and evaluate at natural
prevalence. Compare:

1. `R`: raw or quiet-residual amplitude;
2. `C`: continuous spatiotemporal CFAR;
3. `S = ICA(C)`;
4. `R * phi(C)`;
5. `R * phi(ICA(C))`;
6. separate channels `[R, C, ICA(C)]`, avoiding irreversible multiplication.

For each representation, report known-label recall, candidate count, fixed-
budget recall, fold consistency, quiet false-alarm behavior, and sensitivity to
CFAR radius, temporal reference length, guard size, `beta`, and `tau`.

Before optimizing, characterize normalized distributions for quiet full-field,
event full-field, event ROI, sparse-positive coordinates, and declared artifact
regions. Useful diagnostics include quantiles, log-tail/CCDF plots, fraction of
total energy in the upper tail, spatial concentration, and stability across
bursts. Do not interpret unlabeled event pixels as negatives.

Entropy can describe these distributions, but maximizing entropy alone is not
a sufficient objective: noise and rare outliers can increase it. Any optimized
objective should be constrained by quiet false-alarm behavior, candidate
burden, held-out recall, and fold stability.

## Questions for ChatGPT

1. What is the most principled causal continuous spatiotemporal CFAR statistic
   for this representation, including reference and guard geometry?
2. Should CFAR operate on raw amplitude, quiet-baseline residual, a dynamics
   residual, or several parallel inputs?
3. What ICA observation construction best complements CFAR without simply
   recovering a derivative direction?
4. What properties should `phi` have for scale invariance, robustness to rare
   noise, preservation of slow signals, and transfer to other recordings?
5. Is multiplication preferable to retaining `[R, C, ICA(C)]` as separate
   channels, and what ablations can decide this cleanly?
6. Which information-theoretic or energy-based objectives are defensible under
   sparse-positive labels and non-exhaustive negatives?

## Prompt for ChatGPT

I want to return to the idea of feature engineering. Specifically, I am
interested in a feature of the form
`Y(c,r) = X_raw(c,r) * phi(X_dynamics(c,r))` and how we could potentially use
this information. Something of a random thought is that we could consider
optimizing that equation for some conditions, perhaps entropy, while
understanding what the `phi` function does in this context. Energy-based
scaling seems useful: since signal often tends to be larger than noise, a
simple function such as `x^2` may be sufficient. The primary concern is
low-probability noise, which we could initially tolerate or study through the
normalized signal and full-frame value distributions. Overall, I want to study
energy maps extensively while remaining considerate of generality.

To clarify the intended architecture, I am proposing:

1. Raw fluorescence as a preserved amplitude branch.
2. Continuous CFAR as local spatiotemporal normalization.
3. ICA applied to the continuous-CFAR representation.
4. A nonnegative `phi` applied to the ICA output and multiplied with Raw.

Please assess this architecture using the evidence, constraints, ablations,
and open questions in this handoff. I am particularly interested in a
principled definition of continuous spatiotemporal CFAR, the role of ICA after
CFAR, appropriate energy-based choices for `phi`, and how to evaluate whether
the multiplicative representation adds information beyond its component
features.
