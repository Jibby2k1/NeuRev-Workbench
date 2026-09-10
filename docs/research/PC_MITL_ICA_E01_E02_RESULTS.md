# PC-MITL-ICA E01/E02 paired synthetic results

## Decision

**Retain matrix-TC only for specialized synthetic confirmation.** It did not
pass the broad-advance gate and must not replace CS-Parzen or advance to real
video or process-conditional objectives.

## Frozen design

The development grid paired CS-Parzen and matrix-TC `alpha=2` on the same 80
fixtures: eight declared regimes, each with ten seeds. Regimes crossed two or
four sources, independent Laplace or sparse calcium-like traces, and easy or
hard mixing/noise settings. Both objectives used the same train-only whitening,
three bandwidth candidates, validation-only bandwidth selection, and untouched
contiguous test blocks. Hungarian matching resolved sign and permutation.

The four-component CS comparator is the new differentiable factorized
CS-Parzen reference under the same matrix-exponential orthogonal optimizer. It
does not replace the maintained two-component angle-search implementation.

## Results

Across all 80 pairs, matrix-TC minus CS-Parzen mean held-out absolute source
correlation was `+0.00193`, with bootstrap 95% CI `[-0.01120, +0.01287]` and a
`0.6125` win fraction. Mean recovery was `0.82270` for matrix-TC and `0.82077`
for CS-Parzen. Selected-fit runtime averaged `0.111 s` versus `0.064 s`, making
matrix-TC about 1.7 times slower in this small reference implementation.

The only clear specialized signal was four-source, easy-mixing Laplace data:
matrix-TC improved mean recovery by `+0.03734`, won `10/10` seeds, and had a
bootstrap interval `[+0.02201, +0.05344]`. Four-source hard Laplace averaged
`+0.02119`, but its interval crossed zero. Calcium-like easy regimes favored
CS-Parzen on average (`-0.03147` for two sources and `-0.03224` for four), with
wide intervals driven by seed sensitivity. Other regimes were inconclusive.

## Interpretation

The result is heterogeneous, not a universal matrix-TC win. It suggests the
spectral objective may help higher-dimensional non-Gaussian instantaneous
mixtures, while offering no reliable advantage for the temporally persistent,
sparse calcium-like sources most relevant to NeuRev. That distinction argues
against escalating method complexity in the real-video pipeline.

The exact next permitted experiment is a narrow confirmation of the
four-source Laplace signal with more samples, 20 fresh seeds, additional mixing
conditions, and frozen bandwidth rules. Sparse-calcium confirmation remains a
guardrail. Failure to reproduce the specialized gain stops the matrix-TC
branch; reproduction would justify trajectory matrix-TC simulations, not
immediate real-video use.

That confirmation is now complete. It narrowly failed its locked effect-size
gate, so the branch is stopped; see
`docs/research/PC_MITL_ICA_SPECIALIZED_CONFIRMATION_RESULTS.md`.

## Provenance and audit

The validated run is
`Outputs/UnsupervisedICAEval/pc_mitl_ica_e01_e02_v3`. It contains the resolved
configuration, 160 method rows, 80 paired rows, summary, LLM context, artifact
index, validation record, report, and primary comparison figure. Expert
annotations are explicitly not applicable because exact synthetic truth was
generated; no biological labels or real video were used.
