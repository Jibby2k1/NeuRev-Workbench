# PC-MITL-ICA phase-1 findings

## Decision

**Continue only as a bounded research candidate. Do not replace CS-Parzen and
do not advance to the process-conditional or real-video stages yet.**

## Verified evidence

The reference RBF Gram, trace normalization, matrix Rényi entropy, matrix total
correlation, factorized CS-Parzen reference, differentiable orthogonal rotation,
and two/four-component smoke fitter are implemented. The focused T00-T07 suite
passes on CPU; the CUDA parity check is skipped when CUDA is unavailable.

The frozen two-source synthetic run used a contiguous 58/19/19
train/validation/test split and train-only mean/covariance whitening. Repeating
the matrix-TC fit with the same seed produced a maximum rotation difference of
exactly `0.0`; orthogonality error was `1.57e-16`. Across three distinct
initialization seeds, held-out recovery was `0.961499 +/- 0.000001` (population
standard deviation), indicating no catastrophic seed instability on this
fixture.

On the small held-out test block, matched mean absolute source correlation was:

| Method | Mean absolute correlation |
| --- | ---: |
| Maintained CS-Parzen | 0.9830 |
| Matrix-TC, alpha 2 | 0.9615 |

This single smoke fixture establishes numerical viability, not superiority.
CS-Parzen was modestly better here. Matrix-TC's finite-sample estimate was
`-0.0125`; small negative estimates are possible for this empirical entropy
combination and must not be interpreted as literal negative population
dependence.

## Scientific interpretation

The mathematical bridge in the supplied prospectus is implementation-ready:
at `alpha = 2`, matrix entropy uses the exact squared-Gram information
potential identity, while matrix total correlation remains a different loss
from CS divergence. The current result supports proceeding to broader
synthetic sample-size, bandwidth, mixture-conditioning, and multi-seed gates.
It does not support conditional-history claims, neuronal identifiability,
artifact robustness, causality, or real-video improvement.

## Artifacts and limitations

Run artifacts are in
`Outputs/UnsupervisedICAEval/pc_mitl_ica_phase1_v2/`: `summary.json`, the
resolved configuration, and `phase1_comparison.png`. The fingerprint binds the
configuration, synthetic data identifier, and source Git commit.

This phase uses only 96 synthetic samples, two source distributions, one
mixture, and one primary seed. It does not yet measure multi-seed stability,
sample-size sensitivity, bandwidth sensitivity, four-source recovery, or the
full real-video scientific-audit media contract. The exact next experiment is
the proposal's E01/E02 controlled synthetic grid, paired by generated mixture
and seed, with CS-Parzen retained as the canonical baseline.
