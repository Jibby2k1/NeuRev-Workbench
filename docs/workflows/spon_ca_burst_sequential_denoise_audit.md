# Spon Ca Burst sequential denoising audit

## Purpose

This workflow evaluates denoisers on the completed Parzen Innovation residual
without treating sparse unlabeled pixels as negatives. It answers three
separate questions:

1. Does a method attenuate quiet residual energy?
2. Does it preserve the amplitude, waveform, and timing of known neural events?
3. Does it improve known-label detection at a fixed candidate budget?

The methods are evaluated sequentially so every family produces independently
auditable signal and remainder videos before the next family is interpreted.

## Frozen input

The only source dataset is Spon Ca Burst. UI frames 1800–2359 are evaluated;
frames 1800–1899 are the quiet calibration interval. The input is the accepted
10-second-reference Parzen Innovation lane with correction fraction `0.1` and
correction clip `4` quiet MAD.

For raw frame \(X_t\), causal reference state \(E_t\), accepted affine
teacher-forced reconstruction \(F(X_{t-1},X_t)\), correction bias \(b\),
correction limit \(L\), and correction fraction \(\epsilon=0.1\):

\[
E_t=(1-\alpha)E_{t-1}+\alpha X_t,
\]

\[
B_t=E_t+\epsilon\,
\operatorname{clip}\!\left(F(X_{t-1},X_t)-E_t-b,-L,L\right),
\qquad
R_t=X_t-B_t.
\]

Quiet frames define per-pixel median \(\mu_q\) and robust scale
\(\sigma_q\). Denoisers receive the signed centered residual
\(R_t-\mu_q\); no positive clipping occurs before processing.

## Frozen methods

The manifest contains seven families and eleven variants:

- pointwise: frame min-max gamma, robust-quantile gamma, quiet Wiener;
- spatial evidence gate;
- causal temporal evidence gate;
- temporal-only: Savitzky–Golay, undecimated Haar-like shrinkage, Kalman;
- overlapping local rank-4 PCA;
- quiet-noise-normalized local rank-4 PCA;
- local PCA, batch FastICA, and noisy-Parzen posterior shrinkage.

Local methods use 16×16 patches, stride 8, overlap-add reconstruction, rank 4,
and two randomized oversampling dimensions. The component method uses 20
FastICA iterations, a 32-center half-zero dictionary, bandwidth `0.5`, and
standardized noise variance `1`.

## Evaluation contract

Every method uses the same:

- 79 label rows and 27 ROI identities;
- four held event intervals;
- LME temporal pooling with \(\tau=0.25\);
- 6-pixel non-maximum suppression and match radius;
- one quiet false peak per map threshold;
- 58-candidate-per-burst fixed-budget audit;
- positive TIFF display scale.

Sparse labels support recall, known-label matches, and candidate burden. They
do not identify precision because unmatched candidates are scientifically
unknown.

A separate fixture injects centered, annular, crowded-centered, and
crowded-annular sources into a real quiet residual crop at four declared
amplitudes. It gives exact signal/noise truth, but it is deliberately harder
than an isolated Gaussian-noise simulation.

For each method:

\[
R = S + (R-S)
\]

is checked exactly. `signal_positive.tif` shows \(S^+\) with a common scale;
`remainder_detail.tif` shows \(R-S\) with a method-specific symmetric detail
scale recorded in TIFF metadata.

## Commands

```bash
.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment parzen-denoise-audit preflight \
  --config examples/spon_ca_burst_sequential_denoise_audit.example.json

.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment parzen-denoise-audit run \
  --config examples/spon_ca_burst_sequential_denoise_audit.example.json
```

The run requires a matching green preflight and refuses completed or partial
output collisions.

## Resource and artifact contract

The completed run used two CPU threads and bounded 12-patch CUDA batches on an
RTX 4070 SUPER. Preflight estimated 2.90 GiB peak RAM and capped RAM, GPU
memory, and output at 8 GiB. Actual peak RSS was 3.62 GiB.

Completed root:

```text
Outputs/HierarchicalParzenICA/spon_ca_burst_sequential_denoise_audit_v1
```

It contains 22 verified 560-page, 340×573, uint16 BigTIFFs, `metrics.json`,
`comparison.tsv`, the resolved configuration, preflight, checkpoints, and the
machine-generated report.

The interpretation report is
`docs/research/SPON_CA_BURST_DENOISING_AUDIT_RESULTS.md`.
