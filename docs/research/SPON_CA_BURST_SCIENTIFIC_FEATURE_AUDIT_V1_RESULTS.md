# Spon Ca Burst scientific feature audit v1 results

## Executive conclusion

The audit completed successfully and changes the next research decision. The
clearest result is not a new global mixture: it is a compact causal
neighborhood feature. A 15-frame local-coherence map improved budget-20
known-positive recall in all four held-out bursts and changed macro recall from
`0.5409` to `0.6053`. The lagged-propagation family was non-worse at budget 20
in every burst and improved the mean to `0.5886`. Radial Parzen information also
helped, especially the shell statistic, but was less consistent.

The acquisition audit found strong intensity-dependent noise, the generative
audit found evidence that ring-like observations are harder for the carrier,
and the identical-proposal audit showed some ranking utility. None of these
results establishes biological precision because unmatched candidates remain
unknown.

## Scope and protected design

The frozen program evaluated:

- 6 spatially indexed radial Cauchy--Schwarz maps;
- 4 generative z-cut morphology maps;
- 3 causal local-coherence maps;
- 3 causal lagged-recurrence maps;
- standalone scores and carrier boosts of 0.25, 0.5, and 1.0;
- full-field native, annotated-right native, and identical-proposal regimes.

This is 16 maps, 64 lanes per regime, and 192 scored lanes total. Selection was
leave-one-burst-out over four bursts. Budgets 20 and 40 were primary; budget 58
was secondary. Post-hoc maxima below are explicitly descriptive.

## Protected full-field result

| Estimand | B20 | B40 | B58 | B80 | B100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen carrier | 0.5409 | 0.6572 | 0.7031 | 0.7156 | 0.7264 |
| Cross-fitted all-family selection | 0.5836 | 0.6589 | 0.6947 | 0.7067 | 0.7067 |
| Delta | +0.0427 | +0.0016 | -0.0083 | -0.0089 | -0.0198 |

The protected selector recovered 46/79 labeled observations at budget 20
versus 43/79 for the carrier. At budgets 40 and 58 both recovered the same
total count as the carrier (52 and 55), but exchanged which observations were
found; macro recall therefore changed slightly because burst sizes differ.

The all-family selector chose `radial_cs_shell` for held-out bursts 1 and 4 and
`propagation_lag2_w15` for bursts 2 and 3. Its B20 change by burst was `+0.067`,
`+0.100`, `+0.048`, and `-0.043`. Choosing among families with only three
training bursts is therefore less stable than freezing the strongest compact
family.

## Family audit

| Cross-fitted family | B20 | B40 | B58 | B20 burst outcome |
| --- | ---: | ---: | ---: | --- |
| Local coherence | **0.6053** | 0.6764 | **0.7223** | 4 wins / 0 ties / 0 losses |
| Lagged recurrence | 0.5886 | **0.6806** | 0.7165 | 3 wins / 1 tie / 0 losses |
| Radial information | 0.5949 | 0.6702 | 0.6930 | 3 wins / 0 ties / 1 loss |
| Generative z-cut | 0.5647 | 0.6422 | 0.6589 | 1 win / 3 ties / 0 losses |
| Frozen carrier | 0.5409 | 0.6572 | 0.7031 | reference |

The local-coherence family selected the exact same `coherence_w15` lane in all
four folds. It recovered 48/79 known positives at budget 20, five more than the
carrier. It also recovered 54/79 at budget 40 and 57/79 at budget 58, compared
with 52 and 55 for the carrier. This repeat selection and four-burst B20
consistency make it the strongest compact confirmation candidate.

The lagged family selected `propagation_lag2_w15` in three folds and
`propagation_lag4_w31` in burst 1. Lag-2/15 was also the descriptive top native
lane at budgets 20 and 58 (`0.6220` and `0.7498`), but those post-hoc values are
not protected estimates. Lagged correlation provides temporal-order evidence;
it must not be described as causal biological propagation.

The radial family mainly selected the standalone shell statistic. Its spatial
organization is useful, but its burst-4 loss and budget-58 tradeoff argue
against promoting it alone without confirmation.

## Identical-proposal ranking result

| Estimand | B20 | B40 | B58 | B80 | B100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Carrier on frozen v5 union | 0.4590 | 0.5912 | 0.7086 | 0.7669 | 0.7794 |
| Cross-fitted scientific feature | 0.4590 | 0.6281 | 0.7352 | 0.8153 | 0.8153 |
| Delta | 0.0000 | +0.0369 | +0.0266 | +0.0484 | +0.0359 |

This controls candidate coordinates and shows that the features can add
ranking information at moderate budgets. The gain is not uniformly replicated:
burst 2 supplies most of the B40/B58 gain, burst 3 adds one B40 match, and burst
4 loses one B58 match. It is evidence for utility, not a finished ranker.

## Acquisition and noise audit

Adjacent quiet-frame pairs support a descriptive intensity-dependent noise
model in both apparent fields:

| Field | Variance slope | Weighted R-squared | Saturated sample fraction |
| --- | ---: | ---: | ---: |
| Full | 3.6322 | 0.9591 | 0.00797 |
| Left | 3.6590 | 0.9555 | 0.01369 |
| Right/annotated | 3.6314 | 0.9546 | 0.00226 |

The fitted nonnegative intercept is zero in all three fits, so the measured
range appears shot-noise-like rather than dominated by a constant additive
floor. This does not identify the physical sensor: pair differences contain
fast biology and the 12-bit ceiling clips bright samples.

The provisional `x=286` boundary has a mean absolute jump of 89.09 raw units,
versus 59.43 for internal adjacent columns. Left/right global traces correlate
only `0.557`, and saturation differs sixfold. The two fields must therefore be
audited separately until their acquisition meaning is documented.

Cropping to the annotated right field changes the carrier only slightly:
`0.5409 -> 0.5528` at B20 and not at B40/B58. The unannotated left field is a
real nuisance source, but it is not the main performance ceiling.

The variance-stabilized residual TIFF is a diagnostic. Its detector utility was
not tested in this run and should be evaluated only as an auxiliary feature,
not substituted for the amplitude-preserving carrier.

## Generative morphology audit

The 48-template z-cut bank assigns 67 observations a center-like maximum and
12 a membrane-ring maximum. These assignments are uncertain: 55/79 observations
have a top-two phenotype margin at or below 0.05, and 34/79 at or below 0.01.
They are fitted hypotheses, not morphology labels.

Even with that limitation, ring-like fits concentrate among difficult cases.
At carrier budget 58, 8/24 missed observations are ring-like versus 4/55
matched observations. Because ROI identities recur across bursts, an ordinary
independent-sample significance test would be misleading. This is a targeted
annotation hypothesis: manually verify the ring/center phenotype and z-cut
confidence for recurring misses before fitting a conditional expert.

Carrier-matched labels also have much larger mean template score and crowd
context than misses. That can mean true signal visibility, surrounding
coactivation, or both; it does not show that crowd is intrinsically beneficial.

## Runtime and artifacts

The guarded run completed in 232.5 seconds with 4.4 GiB peak RSS. Radial maps
ran at 5.03 complete frames/s in four-frame CUDA batches. Each recurrence map
required about 3 seconds for all 560 frames in the offline vectorized
implementation. A true 50 Hz claim still requires one-frame streaming latency
and state-update profiling.

Primary artifacts:

- `videos/radial_cs_center.tif`
- `videos/radial_cs_shell.tif`
- `videos/radial_cs_morph_max.tif`
- `videos/coherence_w15.tif`
- `videos/propagation_lag2_w15.tif`
- `videos/noise_vst_residual.tif`
- `diagnostics/generative_zcut_maps.tif`
- `diagnostics/noise_physics_maps.tif`
- `diagnostics/per_label_zcut_audit.tsv`
- `metrics.json`

The completed root is
`Outputs/HierarchicalParzenICA/spon_ca_burst_scientific_feature_audit_v1`.

## Decision and next checkpoint

Do not widen the grid. Freeze a compact confirmation panel:

1. carrier reference;
2. standalone `coherence_w15`;
3. standalone `propagation_lag2_w15`;
4. standalone `radial_cs_shell`;
5. one variance-stabilized auxiliary-carrier lane.

Run it on native and identical proposals with ROI-identity-aware resampling and
small NMS sensitivity checks. In parallel, exhaustively annotate one bounded
right-field calibration region for neuron, artifact, background, unresolved,
center/ring, isolated/crowded, and visibility confidence. That annotation is
required to decide whether the apparent selectivity translates into precision.

Only after confirmation should the three scientific features enter a small
morphology-conditional or monotone ranker. The promoted recurrence operator
must then be implemented as a bounded streaming state update and profiled at
the 20 ms frame period with p50/p95/p99 latency.
