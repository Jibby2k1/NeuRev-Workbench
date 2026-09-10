# PC-MITL-ICA specialized confirmation results

## Decision

**Stop the matrix-TC extension and retain CS-Parzen.** The locked primary gate
failed narrowly; no trajectory matrix-TC, conditional-history objective, or
real-video study is authorized by this result.

## Confirmatory contract

The confirmation used four-component mixtures, 384 samples, two Laplace
mixing/noise conditions, two sparse-calcium guardrails, and 20 fresh seeds. The
CS-Parzen bandwidth was locked at `1.0` and matrix-TC at `0.5`, based on the
modal selections in the promoted four-source easy-Laplace development cells.
There was no confirmation-stage bandwidth search. Both methods used identical
fixtures, train-only whitening, optimizer settings, and untouched contiguous
test blocks.

Advancement required all of the following: primary mean improvement at least
`+0.020`, cluster-bootstrap lower bound above zero, primary win fraction at
least `0.60`, and sparse-calcium mean loss no worse than `-0.020`.

## Results

Across 40 primary paired cells, matrix-TC improved held-out mean absolute source
correlation by `+0.019481`. It won `35/40` cells (`0.875`), and the seed-cluster
bootstrap 95% interval was `[+0.011270, +0.029030]`. The two primary conditions
were:

- condition number 2: `+0.020474`, 17/20 wins;
- condition number 4: `+0.018489`, 18/20 wins.

Across 40 sparse-calcium guardrail cells, the mean difference was `-0.002331`,
with 29/40 matrix-TC wins and interval `[-0.012699, +0.004622]`. The guardrail
therefore passed, but the primary mean missed its locked `+0.020` threshold by
`0.000519`.

## Interpretation

The positive interval and high win fraction support a real specialized
four-source Laplace effect in this simulator. They do not change the gate:
effect magnitude was explicitly part of the advancement requirement, and the
observed improvement fell below it. Relaxing or rounding the threshold after
seeing the result would invalidate the confirmation.

The scientifically safe conclusion is that matrix-TC can modestly improve
some higher-dimensional non-Gaussian instantaneous mixtures, but the gain is
not large enough under the predeclared rule to justify further PC-MITL-ICA
complexity for NeuRev. CS-Parzen remains the canonical method. The
process-conditional and real-video branches remain stopped.

## Provenance

The validated run is
`Outputs/UnsupervisedICAEval/pc_mitl_ica_specialized_confirmation_v3`. It
contains 160 method rows, 80 paired rows, locked configuration, implementation
hashes, summary, validation, LLM context, audit index, report, and comparison
figure. No real video or biological labels were used. Earlier `v1` and `v2`
directories are incomplete collision-protected attempts that exceeded the
interactive execution window and must not be interpreted as results.
