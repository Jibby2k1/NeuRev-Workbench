# Denoise first, then differentiate

## One-sentence objective

NeuRev should first estimate a stable latent fluorescence trajectory from the
noisy movie, then treat ordinary differences and model-aware temporal residuals
as features of that trajectory rather than as complete denoisers.

## Why this follows from the existing experiments

For an adjacent pair,

```text
x_t = [I(t-1), I(t)]^T,
```

persistent image structure lies near `[1, 1]`, while the background-null
direction is `[-1, 1]`. NeuRev's InfoMax and CS-Parzen fits recovered directions
almost exactly collinear with `[-1, 1]`. Pairwise ICA therefore behaved as a
learned temporal derivative, not as a general separation of neural sources.

That derivative is useful because it highlights change. It is incomplete
because it also:

- amplifies independent frame noise;
- suppresses slowly evolving fluorescence;
- reacts strongly to unregistered motion edges;
- removes absolute intensity and anatomy;
- and cannot decide whether a change is neural or artifactual.

The completed pairwise-fusion experiment confirmed the practical consequence:
the derivative/ICA channels did not improve Raw Direct under the tested additive
or gating rules. They remain auxiliary timing and confidence evidence.

## What is currently called “Kalman” in NeuRev

`kalman_positive_residual_stack` is a useful historical baseline, but it is not
a fitted Kalman filter or smoother. It maintains an asymmetric EMA-like
baseline, preserves positive changes, and updates more quickly for negative
changes.

The failed “Kalman spatiotemporal” learned-contrast lane subsequently combined
that positive residual with Gaussian smoothing, quiet-MAD whitening, and a local
contrast detector. Its failure rejects that composite lane. It does not answer
whether an explicit latent state-space model can recover a useful denoised
trajectory.

New reports should retain the historical artifact and API but describe the lane
as `legacy_asymmetric_ema`.

## The four signals that must not be confused

Let the calibrated observation be

\[
r_t = s_t + \epsilon_t,
\]

where `s_t` is latent fluorescence and `epsilon_t` is observation noise.

A stable first-order reference model is

\[
s_t = \gamma s_{t-1}+u_t,\qquad 0\leq\gamma<1.
\]

This produces four distinct quantities.

### 1. Latent state

\[
\widehat s_t.
\]

This is the denoised fluorescence trajectory. It should preserve amplitude,
duration, and slow evolution.

### 2. Ordinary state difference

\[
\Delta_k\widehat s_t
=
\widehat s_t-\widehat s_{t-k}.
\]

This is a change feature. Lag 1 measures 20 ms in the Spon data; lag 4 measures
80 ms.

### 3. Model-aware dynamic drive

\[
\widehat u_t
=
\widehat s_t-\gamma\widehat s_{t-1}.
\]

This is the part of the latent state not predicted by its fitted persistence.
It generalizes subtraction from `[-1, 1]` to `[-gamma, 1]`.

Ordinary differencing is the limiting `gamma=1` case. When `gamma<1`, ordinary
differencing mixes new drive with the expected decay of the previous state,
whereas the model-aware drive attempts to remove that predictable persistence.

### 4. Observation residual

\[
e_t=r_t-\widehat s_t.
\]

This is what the model did not explain. It is a diagnostic, not automatically
noise. Structured residuals can reveal motion, model mismatch, or erased neural
activity.

In standard Kalman terminology, the filter innovation is the one-step
measurement prediction error. NeuRev should use `dynamic_drive` for
\(\widehat s_t-\gamma\widehat s_{t-1}\) to avoid ambiguity.

## Why denoising must precede differencing

If raw frame noise is independent with variance \(\sigma^2\), then

\[
\operatorname{Var}(\epsilon_t-\epsilon_{t-1})=2\sigma^2.
\]

Raw differencing therefore increases high-frequency noise variance even while
canceling persistent background. A valid denoiser should reduce the observation
noise first, after which differencing can expose latent changes rather than
camera noise.

The order is:

```text
observation
  -> baseline/gain calibration
  -> stable latent filter or smoother
  -> latent amplitude
  -> lagged latent differences
  -> model-aware dynamic drive
  -> candidate scoring and feature extraction
```

## Recommended first model

Begin with a shared, quiet-normalized AR(1) state-space reference:

\[
s_t=\gamma s_{t-1}+u_t,\qquad
r_t=s_t+\epsilon_t.
\]

Use:

- a causal Kalman filter for real-time-compatible estimates;
- an offline Rauch--Tung--Striebel smoother for the best full-recording
  estimate;
- a strict stable parameterization for `gamma`;
- bounded process and observation variances;
- quiet-only noise calibration;
- label-free parameter fitting;
- posterior uncertainty and residual diagnostics.

This is intentionally a reference, not the final biological model. It is useful
because every assumption is visible and falsifiable.

If the reference passes, the strongest integrated extension is a local dynamic
factor model:

\[
\mathbf x_t=C\mathbf z_t+\boldsymbol\epsilon_t,\qquad
\mathbf z_t=A\mathbf z_{t-1}+\mathbf u_t,
\]

fitted on overlapping image patches with stable `A`. That extension can use
spatial redundancy to denoise while retaining a compact temporal state. It is
more expressive, but it should not precede a trustworthy scalar/tilewise
baseline.

## Where other methods belong

- **PMD or self-supervised video denoisers:** external denoising comparisons.
  They may improve the movie but still require signal-preservation and
  hallucination checks.
- **CNMF:** joint spatial demixing and temporal estimation when the objective
  becomes cell/footprint extraction.
- **OASIS:** sparse nonnegative calcium-drive inference after a trace or spatial
  component has been defined.
- **InfoMax, CS-Parzen, correntropy, or HSIC:** optional dependence criteria on
  fitted dynamic drives or residuals, not substitutes for the observation
  model.
- **CFAR and morphology:** downstream feature selection and candidate ranking.

No method may infer information absent from the observation without relying on
a prior. Prior-conditioned completion must remain distinguishable from
data-supported recovery.

## What counts as success

A denoiser is useful only if it simultaneously:

1. improves reconstruction on semi-synthetic clean/noisy data;
2. reduces quiet noise and leaves residuals plausibly noise-like;
3. preserves event amplitude, onset, duration, and spatial localization;
4. does not hallucinate persistent activity in pure-noise or quiet controls;
5. is stable across seeds, temporal blocks, and modest acquisition
   perturbations;
6. reports uncertainty where the observation is weak;
7. and produces latent-derived features that can be compared fairly with Raw
   Direct.

A smoother-looking video is not sufficient.

## Repository route

Implementation instructions:
`docs/developer/LATENT_DYNAMICS_DENOISING_IMPLEMENTATION_BRIEF.md`.

Canonical mathematical explanation:
`docs/research/overleaf/neurev_denoise_then_difference.tex`.

The implementation must preserve Raw Direct, completed outputs, sparse-label
semantics, explicit preflight, and the rule that a full Spon or GPU run requires
user selection.
