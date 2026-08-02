# Spon Ca Burst Feature Utility v1 Results

## Executive result

The 25-channel study completed successfully. It evaluated 176 fixed lanes, 100
leave-one-burst-out scalar fits, and 4 constrained multi-feature fits against
the accepted Parzen Innovation carrier.

The carrier reproduced the prior baseline exactly:

| Method | Mean recall | Fixed-budget recall | Event candidates |
| --- | ---: | ---: | ---: |
| Parzen carrier | 0.330 | 0.641 | 70 |

The main result is not that one derivative transform solved detection. The
credible utility is concentrated in spatial/separation features. These features
improve the ordering of labeled neurons, but their quiet-calibrated operating
points admit more candidates:

| Standalone feature | Mean recall | Fixed-budget recall | Candidates | Synthetic correlation | Median peak error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local PSD signal | 0.726 | 0.703 | 440 | 0.656 | 3 frames |
| Asymmetric state | 0.705 | 0.699 | 238 | 0.560 | 4 frames |
| Spatial coherence | 0.622 | 0.693 | 145 | 0.692 | 10 frames |
| Cross-scale rank | 0.655 | 0.671 | 170 | 0.479 | 3 frames |
| Cross-scale recall | 0.609 | 0.626 | 151 | 0.404 | 3 frames |
| Continuous CFAR score | 0.374 | 0.552 | 58 | 0.694 | 3 frames |

Local PSD, asymmetric state, spatial coherence, and cross-scale consensus are
therefore useful candidate-generation or ranking views. None yet demonstrates a
better calibrated final detector.

## Why the nominal derivative winner is not the scientific winner

The learned scalar summary names `derivative_negative_lag1` because its mean
fixed-budget recall is 0.674 versus 0.641 for the carrier. That gain is fragile:

- Burst 1 improves from 0.600 to 0.733.
- Bursts 2, 3, and 4 are unchanged.
- Mean quiet-calibrated recall is unchanged at 0.330.
- Candidate count is essentially unchanged, 69 versus 70.
- Synthetic feature correlation is -0.036 and median peak error is 37.5 frames.

This feature is detecting decay/off-transition evidence and can alter spatial
ranking, but it is not a faithful positive-activation carrier. It may still be
useful as signed contextual evidence in a temporal model. It should not be
presented as the best activation detector.

Absolute, power-1.5, Huber, log-square, and square derivative channels are also
highly redundant. For example, absolute versus Huber correlation is 0.998.
Their apparent differences are mostly calibration effects, not distinct
information sources.

## Fusion results

The constrained six-feature fusion reached:

| Method | Mean recall | Fixed-budget recall | Candidates |
| --- | ---: | ---: | ---: |
| Parzen carrier | 0.330 | 0.641 | 70 |
| Learned multi-feature | 0.492 | 0.657 | 113 |

It selected nearly the same six channels in every fold: crowded-center,
crowded-membrane, spatial coherence, both cross-scale scores, and asymmetric
state. This consistency is useful evidence that spatial context matters.
However, the fixed-budget gain is only 0.017 and candidate burden rises 61%.
The present pointwise logistic objective rewards positive-versus-quiet
separation but does not optimize peak ranking or candidate budget directly.

The cross-fitted fixed-lane selector reached 0.670 fixed-budget recall and 0.560
mean recall, but emitted 300 candidates. It often selected standalone local PSD,
so it is a high-recall candidate generator rather than a selective detector.

## Failed or weak feature families

- Pure derivative-energy features are visually selective but poor standalone
  detectors and have large synthetic timing errors.
- The causal persistence activity gate had fixed-budget recall 0.011 and median
  synthetic peak error 69 frames. Its recurrence is too slow for the current
  decision role.
- Isolated center/membrane morphology experts had zero fixed-budget recall. The
  crowded experts produced very large candidate sets. Their hand-designed
  calibration is not usable as implemented.
- Continuous CFAR is selective and synthetically faithful, but it loses real
  labeled neurons. It remains useful as a localization/context statistic, not
  as the only detector.

## Interpretation and next experiment

The result supports a two-stage architecture:

1. Use local PSD, asymmetric state, spatial coherence, and cross-scale
   consensus as complementary proposal/ranking channels.
2. Preserve the signed Parzen carrier and positive/negative temporal evidence
   for timing and polarity.
3. Learn a ranker on proposed peaks or ROI traces rather than multiplying dense
   maps and optimizing pointwise positive-versus-quiet loss.
4. Optimize a nested objective that directly penalizes candidates or maximizes
   recall at a fixed candidate budget.
5. Add explicit per-neuron miss tables to distinguish the common hard neurons
   from method-specific recoveries.

Because the same four bursts helped select several incumbent hyperparameters,
these results are developmental evidence rather than an independent test-set
claim. The next high-value annotation is exhaustive foreground/background
review inside a bounded spatial region; that would permit real precision and
precision-recall measurements instead of candidate burden as a proxy.

## Artifacts

The completed root is:

```text
Outputs/HierarchicalParzenICA/spon_ca_burst_feature_utility_v1
```

Start with `REPORT.md`, `metrics.json`, `feature_summary.tsv`,
`evaluation/learned_scalar.json`, `evaluation/learned_multifeature.json`,
`evaluation/feature_redundancy.tsv`, and the ten full-resolution TIFFs under
`feature_tiffs/`.
