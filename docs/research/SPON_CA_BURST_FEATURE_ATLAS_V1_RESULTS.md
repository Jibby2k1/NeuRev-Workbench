# Spon Ca Burst Feature Atlas v1 results

## Outcome first

Feature Atlas v1 found a small, directionally favorable but not confirmed
incremental signal. On the same 1,619-candidate union and the same five frozen
spatial folds, the full augmented linear model increased macro held-fold
SPU-AUC from `0.979734` to `0.984392` and recovered `35/78` known-positive
anchors at budget 58 versus `34/78` for the retrained existing model. The
paired global SPU-AUC delta was `+0.004784`, but its 2,000-draw grouped interval
crossed zero: `[-0.004081, +0.015125]`.

The evidence is therefore useful for prioritizing feature engineering, but it
does not justify replacing the current model or making a scientific detection
claim. Validation status is **share with caveats**. Scientific promotion is
false, and the full model-annotation audit for the new rankings is incomplete.

## Frozen evaluation

- Population: 78 canonical-v7 positive anchors and 1,541 unlabeled candidates.
- Unlabeled candidates remain unknown, not negative.
- Candidate coordinates, event windows, five outer folds, candidate budget,
  labels, existing features, penalty, and preprocessing were frozen.
- Primary metric: macro held-fold positive-versus-unlabeled rank AUC.
- Secondary metrics: global known-positive recall at budget 58, grouped paired
  bootstrap, and leave-one-burst-out SPU-AUC.
- The comparison is candidate-reranking only. It does not measure end-to-end
  proposal recall, precision, specificity, or calibrated neuron probability.

## Model comparison

| Model | Macro held-fold SPU-AUC | Known positives at 58 |
| --- | ---: | ---: |
| Carrier only | 0.933518 | 29/78 |
| Frozen Run-B linear | 0.974096 | 33/78 |
| Retrained existing 12-feature linear | 0.979734 | 34/78 |
| New Atlas features only | 0.973738 | 35/78 |
| Existing + all Atlas features | 0.984392 | 35/78 |
| Existing + nuisance/competition only | **0.985227** | **37/78** |

The new-feature-only model is nearly as strong as the existing rich model but
does not exceed it in macro AUC. This means the Atlas mostly recovers an
already available neuron-likeness ordering rather than adding a clearly new
axis of information.

The best family point estimate was nuisance/competition: global SPU-AUC delta
`+0.005641`, grouped interval `[-0.004037, +0.016485]`. This interval also
crosses zero, and the family is the best of four inspected families, so the
result is a candidate for confirmation rather than a promoted winner.

## What each feature family taught us

### 1. Temporal contextual envelope: diagnostic, not additive

The 3- and 5-frame `H=A^2/U` peaks were almost duplicates (Spearman `0.999796`)
and had standalone macro SPU-AUCs `0.891315` and `0.891275`. Adding the temporal
family changed global SPU-AUC by only `-0.000050`, with grouped interval
`[-0.001338, +0.001614]`, and recovered no additional positive at budget 58.

This confirms the earlier estimand explanation: because the causal envelope
includes the current sample, `H=A` at a new local maximum. Peak-rank metrics
therefore erase most of the temporal-context effect. The area-ratio feature was
weaker (`0.798097`), while high core concentration was strongly reversed
(`0.114159`), consistent with sharp isolated spikes being unlike the broader
known calcium events. Keep one envelope peak only as an implementation check
and retain `H` for morphology/shoulder diagnostics, not as a ranking-expansion
direction.

### 2. Soma morphology: useful alone, redundant after the existing panel

The 2.5-pixel scale-normalized LoG response was the strongest morphology
feature alone (`0.893672`, 27/78 at budget 58). Yet adding the full soma family
slightly reduced global SPU-AUC by `0.001314`; the grouped interval was
`[-0.004687, +0.002448]`, with no budget gain. The multiscale LoG minimum was
highly correlated with its component scales, and the center-minus-ring feature
was also strongly correlated with the 2.5-pixel response.

The current feature panel already contains centered and annular structure.
More fixed LoG radii are unlikely to help without a genuinely new estimand,
such as morphology-conditioned proposal formation or scale-selected NMS.

### 3. Map/source consistency: excellent QC signal, no incremental ranking gain

Alternating-frame carrier-map cosine was the strongest new standalone feature
(`0.970948`, 32/78 at budget 58). Frozen-ICA map stability reached `0.948309`,
and held-out prediction of the center trace from the surrounding spatial map
reached `0.918090`. Carrier and ICA stability were correlated (`rho=0.777423`).

Despite strong standalone performance, adding the family reduced global
SPU-AUC by `0.001464`; the grouped interval was
`[-0.004392, +0.000460]`. These features therefore look most valuable as
quality-control, abstention, or provenance fields: a candidate with an
unstable spatial footprint can be flagged for review even if stability does
not improve ranking over the existing panel.

The ICA feature measures temporal split-half stability of the already frozen
ICA representation. It does not refit ICA independently in each half, so it
must not be described as estimator-level ICA reproducibility.

### 4. Nuisance and local competition: the only promising incremental direction

Local proposal isolation was informative alone (`0.852748`) and had a positive
standardized coefficient in all five folds when added to the existing model
(median `+1.5485`). Carrier margin to neighboring proposals was negative in all
five folds (median `-0.0821`). This suggests a concrete detector failure mode:
a known anchor can lie near a stronger proposal, and scalar score ranking does
not resolve whether the pair is a duplicate, a split footprint, or two nearby
cells.

Global-nuisance independence was below chance alone (`0.454038`) but received a
positive conditional coefficient in all folds. That sign reversal is a
suppression/confounding warning, not evidence that global coupling is useful
by itself. The next implementation should emphasize explicit local candidate
clusters and one-to-one resolution, while treating global alignment as a QC or
stratification variable.

## Burst sensitivity

The augmented model improved leave-one-burst-out SPU-AUC in all four bursts:

| Held-out burst | Existing | Augmented | Delta |
| --- | ---: | ---: | ---: |
| 1 | 0.987528 | 0.997732 | +0.010204 |
| 2 | 0.991013 | 0.999183 | +0.008170 |
| 3 | 0.975190 | 0.988395 | +0.013205 |
| 4 | 0.983522 | 0.995821 | +0.012299 |

This consistency argues against one burst solely producing the point estimate.
It is still same-recording evidence conditioned on historically label-informed
event windows, and the grouped intervals above remain the primary uncertainty
check.

## Conclusion and next feature-engineering step

Do not expand dense temporal-envelope, LoG-radius, or generic feature-fusion
sweeps. The most defensible next feature program is **competition-aware,
identity-safe proposal resolution**:

1. Form local candidate clusters using distance plus footprint overlap.
2. Add cluster size, within-cluster score entropy, peak separation, footprint
   overlap, trace partial correlation, and split-versus-two-source evidence.
3. Compare independent retention, merge, and abstain decisions at a fixed
   cluster-review budget.
4. Measure known-positive cluster recall, duplicates per recovered identity,
   one-to-one assignment stability, and abstention coverage. Precision or
   specificity requires exhaustive bounded-field labels.
5. Freeze that resolver before applying it to an independent recording or a
   newly exhaustively reviewed field.

Map/source split-half consistency should accompany this resolver as a QC and
abstention feature. One of the nearly identical envelope peaks should be
removed before any future learned model.

## Engineering and audit record

- Final immutable output:
  `Outputs/NeuronIdentifiability/spon_ca_burst_feature_atlas_v1_20260905_e`
- The final run contains the 1,619-row feature table, OOF scores, model and
  univariate metrics, leave-one-burst-out results, coefficient summaries,
  feature correlations, five grouped family comparisons, three inspected
  figures, small LLM/audit indexes, and a complete hash index.
- `..._20260905.partial` stopped before metrics because baseline subtraction
  attempted to mutate a read-only float32 ICA memmap view.
- `..._20260905_b.partial` stopped before metrics because the full collinear
  augmented model did not meet the inherited 50,000-iteration convergence
  ceiling. The ceiling was raised to 100,000 without changing the loss,
  penalty, features, folds, or any inspected outcome.
- Run C passed numerically but its burst labels overlapped. Run D fixed those
  labels and added family intervals/redundancy diagnostics; its lowest focused
  y-axis tick clipped during rasterization. Run E changes only that display
  bound and is the handoff artifact.
- Full model-annotation videos and close-ups for the new rankings have not been
  rendered. Scientific-audit completion and scientific promotion remain false.
