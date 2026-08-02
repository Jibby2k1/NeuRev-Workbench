# Convolutional Infomax ICA for neural imaging

## Short answer

Evaluating Infomax ICA on image patches is a valid first experiment. It gives
ICA access to local spatial morphology rather than isolated pixel amplitude.
However, this is initially **patch ICA**, not yet convolutional ICA.

A convolutional ICA model learns one shared bank of spatial filters and applies
those filters at every image location. This is preferable for NeuRev because:

- a neuron should produce a similar response after a small translation;
- shared filters avoid arbitrary patch boundaries;
- the parameter count is much smaller than a dense ICA transform over the
  entire image;
- the filters can encode filled, annular, crowded-filled, and crowded-annular
  observations.

The complete proposal is formalized in:

```text
docs/research/overleaf/convolutional_infomax_ica_neural_imaging.tex
```

The Overleaf entrypoint is:

```text
docs/research/overleaf/convolutional_infomax_ica_neural_imaging_main.tex
```

## The central distinction

For a spatial patch \(x_{t,r}\), ordinary patch ICA is

\[
y_{t,r}=WQ(x_{t,r}-\mu).
\]

The samples indexed by time \(t\) and location \(r\) are used to estimate one
dense unmixing matrix \(W\). Independence is imposed among the entries of
\(y\), not among neighboring patches.

Convolutional ICA instead produces component maps:

\[
Y_{k,t}(r)=(b_k * Z_t)(r),
\]

where the same filter \(b_k\) is applied at every location. A tight-frame
constraint prevents collapse:

\[
\sum_k |\widetilde b_k(\omega)|^2 \approx 1.
\]

The model is therefore translation-equivariant and admits a declared synthesis
operator.

## Noise-aware Infomax

For component observation \(y=a+\eta\), use a Gaussian Parzen dictionary for
the clean coefficient \(a\) and explicitly convolve it with projected quiet
noise:

\[
\widehat p_y(y)
=
\frac{1}{M}\sum_m
\mathcal N(y;c_m,h^2+\nu^2).
\]

This gives both:

- a noise-aware marginal score for the Infomax update; and
- an analytic posterior mean for estimating the clean component coefficient.

The reconstructed structured signal must retain an exact unresolved remainder.
That remainder is not automatically measurement noise.

## Temporal extension

The direct causal extension is

\[
Y_{k,t}(r)
=
\sum_{\tau=0}^{L-1}
\sum_u b_k(\tau,u)Z_{t-\tau}(r-u).
\]

The recommended first temporal model factorizes the kernel:

\[
b_k(\tau,u)=h_k(\tau)g_k(u).
\]

This keeps the spatial morphology filter \(g_k\) and short temporal filter
\(h_k\) separately interpretable. Full three-dimensional kernels should be
tested only if the factorized model is stable.

## Important limitation

The Spon Ca Burst movie is single-channel. Patch pixels and temporal lags can
serve as representation coordinates or pseudo-channels, but they are not
multiple physical sensors. Consequently, the learned components are not
automatically physical neurons.

Neurons may also coactivate, violating strict temporal independence. The paper
therefore proposes groups of related filled, annular, neighbor, and artifact
filters, with independence encouraged between group energies rather than
between every individual response.

## Recommended experiment order

1. Patch FastICA as the literal spatial-Infomax baseline.
2. Spatial convolutional FastICA to test weight sharing.
3. Spatial convolutional Parzen-Infomax.
4. Grouped morphology filters.
5. Factorized causal space-time filters.
6. Full three-dimensional convolutional ICA only after earlier gates pass.

Raw Direct and unfiltered Parzen Innovation remain frozen amplitude carriers.
The convolutional model is initially an auxiliary feature branch.

## Initial bounded settings

- signed Parzen Innovation input;
- 13×13 spatial support;
- 16 filters;
- stride 1;
- morphology warm start plus three random seeds;
- 32 Parzen centers, half fixed at zero;
- starting bandwidth \(h=0.5\) and standardized noise variance \(1\);
- no temporal support initially, then 3- and 5-frame causal filters;
- natural-gradient learning-rate screen from \(10^{-5}\) to
  \(3\times10^{-4}\).

## What would constitute success

The model must improve held-burst detection or morphology-specific
semi-synthetic recovery while preserving:

- event peak and area;
- waveform shape;
- onset and peak timing;
- analysis/synthesis closure;
- filter stability across seeds and held bursts;
- bounded causal inference latency.

An improved Infomax objective alone is not evidence of improved neuron
detection.
