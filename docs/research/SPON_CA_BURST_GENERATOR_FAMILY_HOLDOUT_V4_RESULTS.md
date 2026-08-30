# Generator-family holdout benchmark v4

## Design

This automated package tests mechanism-level distribution shift rather than merely new random seeds. It contains 108 exact-truth movies across six generator families—baseline, dense neuropil, bleaching, source-specific nonrigid motion, empirical noise, and compound shift—with six seeds per family and 2, 4, or 8 sources per movie.

The empirical-noise family uses no labels. A robust frame-difference noise estimate was extracted from 48 sparsely sampled frames in untouched `060126/15 right.tif`. The input TIFF hash, sampled frames, crop, raw estimate, and bounded mapping into simulator read noise are stored in `summary.json`.

Carrier, spatial-context, kinetic, and equal-weight combined scores were kept frozen. Proposals use the existing five-pixel local-maximum rule and three-pixel one-to-one matching. The diagnostic proposal budget is twice the exact source count.

## Aggregate operating results

| Stack | Precision | Recall | F1 | Localization (px) | Duplicates/movie |
|---|---:|---:|---:|---:|---:|
| Carrier | 0.259 | 0.519 | 0.346 | 1.434 | 0.361 |
| Spatial context | **0.387** | **0.773** | **0.515** | **1.202** | **0.083** |
| Kinetic | 0.262 | 0.524 | 0.350 | 1.415 | 0.435 |
| Combined | 0.350 | 0.699 | 0.466 | 1.349 | 0.343 |

Spatial context had the highest F1 in every generator family. The compound-shift family was hardest: spatial-context F1 was 0.338, combined F1 0.319, and carrier F1 0.213.

## Leave-one-family-out calibration

For each held-out generator family, calibration was fitted using candidates from the other five families only. Averaged over the six held-out evaluations:

| Stack | ROC AUC | Average precision | Brier | Log loss | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Carrier | 0.771 | 0.531 | 0.131 | 0.429 | 0.500 |
| Spatial context | **0.910** | **0.824** | 0.135 | 0.436 | 0.500 |
| Kinetic | 0.789 | 0.562 | 0.125 | 0.410 | 0.579 |
| Combined | 0.870 | 0.738 | **0.094** | **0.327** | **0.722** |

Spatial context is therefore the strongest ranking and proposal backbone under mechanism shift. Combined remains the strongest probability model at a fixed threshold, but naive equal-weight fusion is less robust at the proposal operating point.

## Scientific boundary and decision

This is exact-truth mechanistic evidence, not biological validation. The empirical recording contributes a scalar label-free read-noise calibration rather than its full spatial and temporal noise distribution. The truth-scaled proposal budget is diagnostic and cannot be used in deployment.

The next automated package should therefore avoid tuning on these evaluated families. It should define development-only simulator families, fit a gated fusion there, and compare it against frozen spatial context on untouched families using label-free false-alarm or score-threshold controls rather than a budget derived from source count.
