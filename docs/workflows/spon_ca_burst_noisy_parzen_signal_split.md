# Spon Ca Burst noisy-Parzen signal/noise split

## Scope

This workflow applies the noise-convolved Parzen posterior from the proposed
Noisy ICA stage to the signed Parzen Innovation residual:

```text
Parzen Innovation residual -> posterior signal + residual noise
```

It isolates posterior denoising so its visual effect can be audited before the
much larger patchwise ICA stage. It does not yet implement patchwise
noise-corrected subspace selection, ICA demixing, overlap-add reconstruction,
or neural-versus-artifact component qualification. Consequently, the residual
video is a `noise candidate`, not established pure measurement noise.

## Method

The input is the completed 10-second reference, 0.1 correction-fraction,
4-MAD Parzen Innovation lane. No positive clipping occurs before the split.

For residual \(R_t(p)\), quiet calibration gives a per-pixel median
\(\mu_q(p)\) and robust scale \(\sigma_q(p)\):

\[
z_t(p)=\frac{R_t(p)-\mu_q(p)}{\sigma_q(p)}.
\]

A 64-center label-free spike-and-slab dictionary is fitted from post-quiet
samples. Half of its centers are fixed at zero and half are empirical signed
active-sample quantiles. For dictionary centers \(c_m\), bandwidth \(h\), and
standardized noise variance \(\nu^2\), the clean posterior is

\[
\widehat s(z)=\sum_m \alpha_m(z)
\frac{\nu^2c_m+h^2z}{h^2+\nu^2}.
\]

The exact split is

\[
S_t(p)=\sigma_q(p)\widehat s(z_t(p)),
\qquad
N_t(p)=R_t(p)-S_t(p).
\]

Thus \(R=S+N\) closes exactly. The implementation evaluates 16 combinations:
bandwidths `0.25`, `0.5`, `1`, and `2` crossed with standardized noise
variance multipliers `0.5`, `1`, `2`, and `4`. The posterior is evaluated as a
65,536-point monotone lookup curve for efficient full-field inference.

## Commands

```bash
.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment parzen-signal-split preflight \
  --config examples/spon_ca_burst_noisy_parzen_signal_split_balanced.example.json

.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment parzen-signal-split run \
  --config examples/spon_ca_burst_noisy_parzen_signal_split_balanced.example.json
```

The runner requires a matching read-only preflight and refuses completed or
partial output collisions.

## Completed results

Two completed visual audits are preserved:

```text
Outputs/HierarchicalParzenICA/spon_ca_burst_noisy_parzen_signal_split_v1
Outputs/HierarchicalParzenICA/spon_ca_burst_noisy_parzen_signal_split_balanced_v1
```

The first is conservative (`h=2`, `nu2=0.5`): median labeled peak retention
`0.9360`, area retention `0.9358`, and quiet signal RMS ratio `0.9350`.

The balanced audit selected `h=0.5`, `nu2=1`. It retained median labeled peak
`0.7726`, area `0.7638`, late activity `0.7770`, and waveform correlation
`0.9970`, while reducing quiet signal RMS to `0.7261` of the input. Its
candidate-noise remainder contains `12.07%` of the non-orthogonal split energy.
This is stronger attenuation, but it also removes roughly 23% of labeled
amplitude. Selection used all real labels and is exploratory, not an unbiased
performance estimate.

Both runs completed in under 37 seconds with approximately 1.10 GiB peak RAM
and exact arithmetic closure.

## Videos

Each output root contains:

- `parzen_signal.tif`: signed posterior signal; mid-gray is zero;
- `parzen_noise.tif`: signed residual remainder; mid-gray is zero; and
- `parzen_signal_positive.tif`: positive posterior signal; black is zero.

The signed signal and noise TIFFs use the same fixed scale across all frames,
so their relative magnitude is visually meaningful. The positive video is
provided for easier neural-activity review.

## Audit questions

1. Are recognizable neurons or propagating activity visible in
   `parzen_noise.tif`? If so, the posterior is over-shrinking signal.
2. Does `parzen_signal_positive.tif` suppress spatial speckle without erasing
   weak or membrane-like activations?
3. Does the split flicker in quiet frames?
4. Are motion edges, saturated artifacts, or persistent anatomy concentrated
   in either output?
5. Is the balanced split preferable to the conservative split, or is the
   approximately 23% amplitude reduction too costly?

Only after this audit should the project implement local noisy ICA and attempt
to call the structured remainder measurement noise.
