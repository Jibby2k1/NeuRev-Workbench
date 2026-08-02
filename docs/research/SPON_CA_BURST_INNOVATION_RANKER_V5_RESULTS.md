# Spon Ca Burst innovation ranker v5 results

## Executive result

The authoritative v5 study completed 2,266 nested model fits over 34 feature
channels, 22 proposal sources, five feature sets, and four held-out bursts. It
also evaluated budgets 20, 40, 58, 80, and 100, wrote exact per-neuron recovery
tables, and verified five diagnostic TIFFs.

At the primary budget of 58 candidates per burst:

| Stage | Mean burst recall | Pooled matches | Increment |
| --- | ---: | ---: | ---: |
| Archived centered-residual carrier | 0.641 | 51/79 | — |
| Quiet-standardized carrier, native peaks | 0.703 | 55/79 | +0.062 |
| Standardized carrier score on proposal union | 0.709 | 56/79 | +0.006 |
| Nested bounded-linear ranker | 0.725 | 57/79 | +0.017 |
| Nested residual MLP | 0.725 | 57/79 | +0.017 |
| Nested single-feature selection | **0.734** | **58/79** | +0.025 versus union carrier |
| Optimistic 58-per-source union | 0.902 | 71/79 | headroom only |

The primary architectural conclusion is that preprocessing and feature choice
matter more than model depth. Quiet per-pixel standardization produced most of
the measured gain. The broad proposal union added one label, and learned
ranking added one more. The residual MLP did not beat the bounded linear model,
while leakage-safe selection of one feature was slightly stronger than both.

These are known-positive recall results under sparse annotation. Candidate
burden is not precision, and the optimistic source union is not a deployable
58-total-candidate detector.

## Exact experiment design

The study consumed the completed feature-utility bank and added:

- signed powers 1.25, 1.5, and 2.0;
- residual shot-noise whitening at authorities 0.25, 0.5, and 1.0;
- onset dominance and derivative energy;
- artifact attenuation at authorities 0.5 and 1.0;
- center responses at sigmas 1.5, 2.5, and 3.5 pixels;
- annular responses at radii 3.0, 4.5, and 6.0 pixels, each at thicknesses
  0.75 and 1.25 pixels;
- crowd and structural context.

The proposal union used 22 positive-evidence sources. Zero-evidence plateaus
were explicitly excluded before NMS. Every feature was normalized from quiet
maps only, and unmatched event candidates remained unknown.

The bounded-linear grid contained 135 configurations:

- five feature sets;
- learning rates 0.003, 0.01, and 0.03;
- L2 values 0.01, 0.1, and 1.0;
- auxiliary-weight authorities 0.25, 0.5, and 1.0;
- 300 epochs.

The residual-MLP grid contained 120 configurations:

- five feature sets;
- learning rates 0.0003, 0.001, and 0.003;
- weight decays 0.001 and 0.01;
- widths 8 and 16;
- residual authorities 0.25 and 0.5;
- 150 epochs;
- two inner seeds and three outer confirmation seeds.

For each outer burst, hyperparameters were selected using only the other three
bursts. Each inner fit trained on two bursts and validated on the third. The
selected model was then refit on all three training bursts and evaluated once
on the untouched outer burst. This produced 2,250 inner fits and 16 outer
refits.

## Budget behavior

Mean burst recall across the five candidate budgets was:

| Method | 20 | 40 | 58 | 80 | 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quiet-standardized carrier, native peaks | **0.541** | **0.657** | 0.703 | 0.716 | 0.726 |
| Carrier score on proposal union | 0.459 | 0.591 | 0.709 | 0.767 | 0.779 |
| Nested linear | 0.470 | 0.620 | 0.725 | 0.750 | 0.779 |
| Nested MLP | 0.470 | 0.591 | 0.725 | 0.763 | 0.775 |
| Nested single feature | 0.405 | 0.588 | **0.734** | **0.799** | **0.852** |

This changes the interpretation materially. At the tighter budgets of 20 and
40, the native standardized carrier is still best. The learned methods improve
only after the budget reaches 58. Therefore, the current fine-tuning does not
yet demonstrate a precision-oriented improvement; it rearranges the middle of
a larger candidate list more effectively than its very top.

## What the linear model learned

All four outer folds selected the `separation` feature set. Its immutable skip
was the standardized carrier, with auxiliary inputs:

1. local PSD signal;
2. asymmetric state;
3. spatial coherence;
4. cross-scale rank;
5. cross-scale recall;
6. continuous CFAR score.

Three folds used nearly the full auxiliary authority, with approximate weights
`0.30, 0.31, 0.09, 0.15, 0.15, 0.00`. Burst 2 selected stronger regularization
and used total authority about 0.32. The continuous CFAR score received exactly
zero weight in every outer refit.

This is consistent with earlier studies: local subspace evidence, asymmetric
dynamics, and cross-scale spatial agreement are useful, while the current CFAR
statistic is too selective to improve this ranking objective. It remains a
diagnostic localization statistic, not a promoted detector.

## Feature and morphology findings

On the common proposal table, the strongest post-hoc feature was
`cross_scale_rank` at fixed-budget recall 0.779. Center and small-annulus
responses also ranked highly, reaching approximately 0.761–0.767. Those
post-hoc values are optimistic because the same labels identify the feature;
they are not held-out selection estimates.

The leakage-safe nested single-feature procedure selected:

- burst 1: asymmetric state, fixed recall 0.733;
- burst 2: cross-scale rank, 0.700;
- burst 3: center sigma 2.5, 0.762;
- burst 4: cross-scale rank, 0.739.

The selected feature varies by outer fold, but the pattern is coherent:
spatial scale agreement and cut-aware center morphology contribute more than
another global nonlinearity of the carrier.

Residual shot-noise correction was effectively inert. After existing quiet
per-pixel standardization, the fitted variance-versus-structure slope was
0.102, and the whitening gain ranged only from 0.993 to 1.044. Stronger global
shot-noise whitening is not justified.

## Per-neuron failure structure

The 79 observations include several recurring hard identities:

| ROI | Observations | Missed by both learned rankers | Outside source union | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `roi_007` | 4 | 4 | 3 | predominantly proposal/representation failure |
| `roi_014` | 4 | 4 | 0 | proposal exists; persistent ranking failure |
| `roi_019` | 3 | 3 | 0 | proposal exists; persistent ranking failure |
| `roi_015` | 4 | 2 | 1 | mixed proposal and ranking failure |
| `roi_023` | 2 | 2 | 2 | proposal/representation failure |

At budget 58 per source, eight observations were absent from the entire source
union. Another 14 were recoverable somewhere in the union but missed by both
learned rankers. This creates two distinct development targets:

1. new or better spatial evidence for union-missing cells;
2. listwise or morphology-conditional ranking for recoverable cells.

Treating both groups as one generic model error would waste experiments.

## Evaluator audit trail

The completed v1–v3 roots and failed v4 partial are preserved as audit records.
They are not authoritative:

- v1 hard-clipped quiet-normalized maps before NMS, creating upper plateaus;
- v2 replaced the hard ceiling with monotone log compression;
- v3 added budget and per-neuron audits, which exposed lower zero-evidence
  plateau proposals and a carrier-semantics mismatch;
- v4 excluded zero-evidence proposals and deliberately stopped when an exact
  baseline guard revealed that the archived feature-bank carrier used the
  centered residual, whereas `carrier_signed.npy` is quiet-standardized;
- v5 reports both carriers separately and decomposes the gain correctly.

This audit is scientifically important: the apparent improvement initially
attributed to proposal diversity was mostly the effect of quiet per-pixel
standardization.

## Decision

Use the bounded linear ranker as the preferred learned reference because it
ties the MLP at the primary fixed budget, uses fewer thresholded candidates,
and is easier to inspect. Do not call it the final detector: it loses to the
native carrier at budgets 20 and 40 and does not establish precision.

The next experiment should not widen the MLP grid. It should:

1. obtain exhaustive foreground/background annotation in one bounded field;
2. label center/membrane and isolated/crowded morphology for the recurring hard
   ROIs;
3. evaluate proposal generation, ranking, NMS, and threshold calibration as
   separate stages;
4. train a small morphology-conditional or listwise ranker on manually
   reviewed hard negatives;
5. require improvement at budgets 20 and 40, not only 58;
6. retain the standardized carrier as an immutable skip and preserve exact
   timing/amplitude outputs separately from ranking scores.

## Artifacts

The authoritative root is:

```text
Outputs/HierarchicalParzenICA/spon_ca_burst_innovation_ranker_v5
```

Start with `REPORT.md`, `metrics.json`,
`evaluation/budget_curves.json`, `evaluation/per_neuron_audit.tsv`,
`evaluation/recoverable_but_missed.tsv`, and the five files under
`diagnostics/`.
