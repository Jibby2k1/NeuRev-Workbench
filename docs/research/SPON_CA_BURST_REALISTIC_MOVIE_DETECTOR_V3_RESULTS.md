# Realistic movie detector benchmark v3

## Design

The exact-truth movie simulator was extended from pixel discrimination to complete proposal evaluation. The benchmark contains 432 movies: 12 seeds across 36 factorial cells varying source separation, weak-to-strong amplitude ratio, temporal correlation, and elliptical versus crescent weak-source morphology. Movies also include structured background, subpixel periodic motion, signal-dependent noise, read noise, and a compact impulse artifact.

Four frozen score stacks were compared: carrier variance, spatial context, a fixed calcium-kinetic matched score, and their robust-standardized sum. Proposals use a fixed five-pixel local-maximum window. Budgets 2, 4, and 8 are evaluated with three-pixel Hungarian one-to-one identity matching.

## Primary budget-four results

| Stack | Precision | Recall | F1 | Mean localization (px) | Duplicates/movie |
|---|---:|---:|---:|---:|---:|
| Carrier | 0.329 | 0.659 | 0.439 | 1.207 | 0.164 |
| Spatial context | 0.335 | 0.670 | 0.447 | **0.835** | **0.016** |
| Kinetic | 0.319 | 0.638 | 0.425 | 1.446 | 0.153 |
| Combined | **0.354** | **0.707** | **0.471** | 1.170 | 0.056 |

At the strict two-proposal budget, combined precision, recall, and F1 are all 0.655, compared with 0.571 for carrier. Increasing the budget to eight raises carrier recall to 0.738 but lowers its precision to 0.185; the same general budget tradeoff appears for every stack.

## Calibration and abstention

Calibration models were fitted only on seeds 0--5 and evaluated on seeds 6--11. Candidate prevalence among the top 16 proposals was only 9.2--11.7%, so raw classification error is not interpreted alone. Spatial context had the best held-seed ranking (average precision 0.896; ROC AUC 0.949). Combined had the best probability quality and thresholded balance (Brier 0.0286, log loss 0.119, ECE10 0.0094, balanced accuracy 0.862). Combined candidate classification error decreased from 3.18% at full coverage to 0.69% at 50% coverage, supporting confidence-based abstention within this simulator while retaining the prevalence caveat.

## Interpretation

The results sharpen the feature-role story:

- carrier provides strong general event evidence;
- spatial context is especially valuable for localization and duplicate suppression;
- the fixed kinetic score contributes calibrated temporal evidence but is not the strongest standalone proposer;
- combination gives the best compact-budget identity recovery and calibration.

These are mechanistic conclusions under exact simulated truth, not evidence of independent-animal generalization. The next major simulator test should hold out entire generator families and add variable source counts, neuropil contamination, bleaching, nonrigid motion, and empirical noise fitted without labels from untouched recordings.
