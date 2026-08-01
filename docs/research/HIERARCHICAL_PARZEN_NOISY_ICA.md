# Hierarchical Parzen ICA followed by noisy Parzen ICA

## Objective

NeuRev should test whether a reversible two-stage information-theoretic
factorization can separate the Spon Ca Burst movie into three scientifically
interpretable channels:

\[
X_t = \widehat B_t + \widehat S_t + \widehat N_t,
\]

where:

- \(\widehat B_t\) is reconstructed persistent or slowly varying background;
- \(\widehat S_t\) is reconstructed structured dynamic fluorescence;
- \(\widehat N_t\) is the unexplained residual, which is only called noise after
  residual-validity checks.

The hierarchy is:

```text
Stage 1: temporal Parzen ICA
X -> reconstructed background B_hat + amplitude residual R

Stage 2: local noisy Parzen ICA
R -> reconstructed structured signal S_hat + residual N_hat
```

This is not ordinary ICA applied twice. Stage 1 assigns aggregate temporal
components using explicit background criteria. Stage 2 treats noise as additive
observation uncertainty rather than as a uniquely identifiable ICA component.

## Why this follows from the existing NeuRev evidence

The adjacent-frame pairwise experiment used

\[
\mathbf x_t(p)=
\begin{bmatrix}
I_{t-1}(p)\\
I_t(p)
\end{bmatrix}.
\]

The background-like direction is approximately \([1,1]\), while its orthogonal
change direction is \([-1,1]\). NeuRev's InfoMax and CS-Parzen fits became nearly
collinear with subtraction. That is useful evidence that pairwise ICA finds a
temporal derivative, but it also means the non-background component still
contains both neural change and frame noise.

Subsequent results sharpen the conclusion:

- latent smoother **amplitude** improved known-label recovery;
- raw and latent differences, positive dynamic drive, and filter innovation
  underperformed;
- amplitude PCA rank 8 showed a small fixed-budget gain;
- higher-rank ICA became unstable and rank-64 ICA failed to converge reliably.

The new experiment should therefore use temporal ICA to estimate and reconstruct
background, but must pass the amplitude-preserving residual

\[
R_t = X_t-\widehat B_t
\]

to Stage 2. It must not equate the derivative component with the complete dynamic
movie.

## Stage 1: temporal Parzen ICA for background reconstruction

### Embeddings

The mandatory embedding is the adjacent pair

\[
\mathbf x_t^{(2)}(p)=
[I_{t-1}(p),I_t(p)]^\top.
\]

A gated extension uses three frames:

\[
\mathbf x_t^{(3)}(p)=
[I_{t-2}(p),I_{t-1}(p),I_t(p)]^\top.
\]

The three-frame reference geometry contains:

\[
[1,1,1] \quad \text{common level},
\]

\[
[-1,0,1] \quad \text{slope/change},
\]

\[
[1,-2,1] \quad \text{curvature}.
\]

ICA may rotate these directions, but the embedding dimension still limits the
number of aggregate components. These components are not automatically physical
sources.

### Parzen independence criterion

After robust centering and whitening, fit a demixing matrix \(W_1\). The first
reference is the existing bounded CS-Parzen criterion. A stochastic extension
may use a bounded dictionary or mini-batch estimate of the information gradient.

The implementation must report:

- objective history;
- orthogonality/decorrelation error;
- convergence status;
- restart or seed stability;
- effective temporal direction in observation coordinates;
- covariance condition number;
- kernel bandwidth and dictionary occupancy.

### Background assignment

Do not choose the background component by component index or by the mean of its
derivative. Zero-mean noise also has approximately zero mean derivative.

For component time course \(z_j(t)\), compute robust normalized energies:

\[
E_{1,j}=\frac{\operatorname{median}_t(\Delta z_j(t))^2}
{\operatorname{Var}(z_j)+\epsilon},
\]

\[
E_{2,j}=\frac{\operatorname{median}_t(\Delta^2 z_j(t))^2}
{\operatorname{Var}(z_j)+\epsilon}.
\]

Combine these with:

- correlation with global frame intensity;
- low-spatial-frequency mass;
- spatial support breadth;
- event-versus-quiet modulation;
- motion-edge correlation;
- sign/permutation continuity from the previous fitted window.

The assignment must return a confidence margin and may return `unresolved`.
When unresolved, the run must preserve the previous valid assignment or fall back
to a declared baseline; it must not force an arbitrary component to background.

### Reconstruction

If the fitted model is

\[
\mathbf x_i=A_1\mathbf y_i+\boldsymbol\epsilon_i,
\]

select the background-like component set \(\mathcal B\), reconstruct its
contribution to the **current-frame coordinate**, and define

\[
\widehat B_t(p)=
\mathbf e_{\mathrm{current}}^\top
A_{1,\mathcal B}\mathbf y_{i,\mathcal B},
\]

\[
R_t(p)=I_t(p)-\widehat B_t(p).
\]

Always preserve \(I_t\), \(\widehat B_t\), and \(R_t\). Stage 1 is reversible at
the artifact level and its closure error is measured explicitly.

## Stage 2: local noisy Parzen ICA

### Generative model

For each overlapping image patch, model the Stage-1 residual as

\[
\mathbf r_t=A_2\mathbf s_t+\mathbf n_t,
\]

where \(\mathbf s_t\) contains structured latent components and \(\mathbf n_t\)
is additive observation noise plus model error.

Noise is not another ordinary ICA source. Multiple Gaussian noise directions are
rotationally ambiguous, and structured artifacts may not be independent of the
signal. The goal is to identify reproducible structured sources under an
explicit noise model and validate the residual.

### Quiet noise model

Using quiet residual frames only, estimate:

- per-pixel robust center and scale;
- diagonal noise variance as the mandatory reference;
- optional diagonal-plus-low-rank covariance as a gated extension;
- temporal autocorrelation;
- spatial autocorrelation;
- intensity-conditioned variance.

The first implementation may use a diagonal Gaussian approximation after quiet
normalization, but it must report mismatch to that model.

### Noise-corrected subspace

For patch covariance \(\widehat\Sigma_r\) and estimated noise covariance
\(\widehat\Sigma_n\), form

\[
\widehat\Sigma_s=
\operatorname{PSD}\!\left(
\widehat\Sigma_r-\widehat\Sigma_n
\right).
\]

Retain only positive modes that pass eigenvalue, stability, and rank limits.
Whiten in the estimated signal subspace. Ordinary PCA whitening remains an
ablation because it treats noise energy as signal energy.

### Noise-convolved Parzen density

For latent scalar source \(s_j\), use the Gaussian Parzen prior

\[
\widehat p_{s_j}(s)=
\frac1M\sum_{m=1}^{M}
\mathcal N(s;c_{jm},h_j^2).
\]

If the projected additive noise is

\[
\eta_j\sim\mathcal N(0,\sigma_j^2),
\]

then the observed output density is analytically

\[
\widehat p_{y_j}(y)=
\frac1M\sum_{m=1}^{M}
\mathcal N\!\left(
 y;c_{jm},h_j^2+\sigma_j^2
\right).
\]

The Parzen bandwidth \(h_j\) and physical noise variance \(\sigma_j^2\) are
separate quantities and must be recorded separately.

The posterior kernel responsibility is

\[
\alpha_{jm}(y)=
\frac{\mathcal N(y;c_{jm},h_j^2+\sigma_j^2)}
{\sum_{\ell}\mathcal N(y;c_{j\ell},h_j^2+\sigma_j^2)}.
\]

The posterior mean inside kernel \(m\) is

\[
\mu_{jm}(y)=
\frac{\sigma_j^2c_{jm}+h_j^2y}
{h_j^2+\sigma_j^2}.
\]

The denoised source estimate is

\[
\widehat s_j(y)=
\sum_m\alpha_{jm}(y)\mu_{jm}(y).
\]

Posterior variance must also be emitted so the method does not hide uncertainty.

### Stochastic demixer update

The negative score of the noisy Parzen mixture is

\[
\psi_j(y)=
\frac{y-\overline c_j(y)}{h_j^2+\sigma_j^2},
\qquad
\overline c_j(y)=\sum_m\alpha_{jm}(y)c_{jm}.
\]

A reference natural-gradient update is

\[
W_2\leftarrow W_2+
\eta\left[I-\bm\psi(\mathbf y_t)\mathbf y_t^\top\right]W_2,
\]

followed by symmetric decorrelation. This is a starting implementation, not an
assumed convergence theorem. The code must support deterministic batch,
mini-batch, and bounded stochastic modes and expose all update diagnostics.

Dictionary centers should be updated from posterior clean-source estimates, not
blindly copied from noisy outputs. The first implementation may freeze the
centers after calibration; online dictionary adaptation is gated behind stable
batch results.

### Reconstruction

For accepted structured components,

\[
\widehat S_t=A_2\widehat{\mathbf s}_t,
\]

and

\[
\widehat N_t=R_t-\widehat S_t.
\]

Call the latter `noise_candidate` until it passes residual checks. A coherent
annulus, event-locked trace, motion edge, or global drift in \(\widehat N_t\) is
model failure, not measurement noise.

## Reversible refinement

A hard Stage-1 decision can remove weak neural signal before Stage 2. Preserve
all channels and optionally perform one bounded refinement:

\[
\widehat B\leftarrow
\arg\min_B\|X-B-\widehat S\|^2+R_B(B),
\]

then refit Stage 2 on \(X-\widehat B\). The update must be trust-region bounded
relative to the original Stage-1 background and must be reported as an explicit
ablation.

## Required visual results

Every dataset-scale run must produce fixed-scale, frame-aligned visual evidence.
At minimum:

1. raw frame;
2. reconstructed background;
3. Stage-1 amplitude residual;
4. reconstructed structured signal;
5. noise/artifact candidate;
6. numerical closure error;
7. Stage-1 component maps/traces and assignment scores;
8. Stage-2 source maps/traces, posterior means, and uncertainty;
9. representative raw/background/signal/noise traces at known and control ROIs;
10. sparse-label detection overlays with known TP, known FN, and unmatched
    candidates kept semantically distinct.

Use one fixed display scale within every comparison family. Do not independently
auto-scale frames in a way that makes noise suppression appear as signal gain.

## Required quantitative results

The overarching goal is correct attribution, not merely a smooth video.

### Decomposition accounting

\[
e_t^{\mathrm{closure}}=
X_t-\widehat B_t-\widehat S_t-\widehat N_t.
\]

Report maximum, median, p95, and p99 normalized closure error by frame and patch.

### Semi-synthetic leakage matrix

For known true channels \(B,S,N\) and estimated channels
\(\widehat B,\widehat S,\widehat N\), report the normalized attribution matrix

\[
L_{ij}=\frac{\langle C_i,\widehat C_j\rangle_F^2}
{\|C_i\|_F^2\|\widehat C_j\|_F^2+\epsilon}.
\]

Also report direct energy leakage and reconstruction NMSE. A successful result is
diagonally dominant and stable across seeds and perturbations.

### Signal preservation

At injected and known neural locations, report:

- amplitude ratio and bias;
- temporal-area ratio;
- onset, peak-time, and duration error;
- decay-time bias;
- spatial centroid error;
- footprint IoU and annular overlap;
- event energy leaked to background;
- event energy discarded to the noise candidate.

### Noise validity

Report:

- temporal residual ACF and integrated nonzero-lag ACF energy;
- residual PSD versus quiet noise PSD;
- spatial autocorrelation by radius;
- intensity-conditioned variance calibration;
- residual event-locking at known neurons;
- residual motion-edge correlation;
- normality/heavy-tail diagnostics as descriptive evidence.

### Detection utility

Under the existing NeuRev contract, compare:

- Raw Direct;
- current latent smoother amplitude;
- amplitude PCA rank 8;
- Stage-1 residual amplitude;
- Stage-2 structured-signal amplitude;
- posterior source probability/uncertainty-aware scores;
- bounded fusions initialized exactly at Raw Direct.

Report both quiet-calibrated and fixed-candidate-budget results, fold-wise known
recall, total candidates, localization error, shared misses, and candidate-set
stability. Unmatched candidates remain unknown.

### Optimization and stability

Report:

- objective and gradient histories;
- seed/restart convergence;
- demixer subspace angles and matched component correlations;
- background assignment margin;
- dictionary occupancy and center drift;
- bandwidth/noise-variance sensitivity;
- patch-overlap disagreement;
- candidate Jaccard across seeds and nearby hyperparameters;
- failure rate, not only the best run.

### Real-time metrics

For causal/frozen inference, report p50, p95, p99, and maximum latency for:

- Stage-1 projection and reconstruction;
- Stage-2 patch projection;
- posterior Parzen denoising;
- overlap-add reconstruction;
- feature/detection processing;
- total frame processing.

Separate per-frame inference from slow demixer or dictionary adaptation. At
50 Hz, p95 total software time must remain below 20 ms before the method is
considered real-time eligible.

## Interpretation boundaries

- Stage 1 estimates a background-like aggregate component; it does not prove a
  unique physical background.
- Stage 2 estimates structured sources under an additive-noise prior; it does not
  make Gaussian noise an identifiable ICA component.
- A visually clean residual is not evidence of correct separation.
- A high-entropy output is not evidence of neural recovery.
- A residual is noise only if its structure is consistent with the declared
  noise model and it contains no material event-locked or spatially coherent
  signal.
- Missing information is not recovered without a prior; prior-conditioned
  completion must be labeled accordingly.

## Repository routes

Authoritative Codex package:

`docs/developer/hierarchical_parzen_noisy_ica/`

Canonical Overleaf manuscript:

`docs/research/overleaf/hierarchical_parzen_noisy_ica/main.tex`

The implementation must preserve all completed experiment evidence and must not
launch a full Spon or GPU run without explicit selection and a new output root.
