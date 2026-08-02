# Hierarchical Parzen ICA: visual and metric contract

## Purpose

The experiment must make the decomposition inspectable. A lower objective or a
smoother video is not enough. Every report must show where the measured energy
went, which components were accepted, which failures are shared, and whether the
final residual behaves like noise.

The required channels are:

```text
observation
background_like
stage1_residual
structured_neural_signal
structured_artifact
measurement_noise
reconstruction_closure_error
```

## Mandatory visual artifacts

### Figure V1 — End-to-end decomposition sheet

For at least:

- one quiet frame;
- one representative frame from each labeled burst;
- one known shared-miss frame;
- one motion/artifact-heavy frame;

render a common-scale panel:

```text
Raw | Background | Stage-1 residual | Neural signal
Artifact | Noise | Sum reconstruction | Closure error
```

Requirements:

- fixed display scale within each physical channel across all selected frames;
- signed channels use a symmetric scale and explicit zero code;
- no per-frame auto-contrast;
- label circles and candidate markers are overlays, not burned into the
  scientific arrays;
- caption reports stage IDs, model IDs, causal status, and output units.

### Figure V2 — Stage-1 geometry and component identity

Show:

1. whitened two-observation sample cloud;
2. common direction `[1, alpha_gain]`;
3. derivative direction `[-alpha_gain, 1]`;
4. fitted demixing axes;
5. selected background component and confidence margin.

A companion time plot must show:

- demixing angle;
- sign/permutation tracking events;
- staticness scores for both components;
- covariance condition number;
- update acceptance/fallback events.

### Figure V3 — Staticness classification plot

Scatter every Stage-1 component/block using:

- normalized first-derivative energy on x;
- normalized second-derivative energy on y;
- point size = spatial broadness;
- point color = common-direction cosine or classification;
- marker edge = resolved/unresolved.

Show declared decision boundaries and the score margin.

### Figure V4 — Stage-1 leakage and residual preservation

For semi-synthetic fixtures, show:

- true background vs estimated background;
- true signal amplitude before/after Stage 1;
- signal leakage into background by SNR and event type;
- background leakage into residual;
- residual quiet variance.

For real data, show raw and Stage-1 residual traces at:

- known ROI centers;
- membrane annuli;
- matched nearby quiet controls;
- high-gradient motion edges.

### Figure V5 — Stage-2 noise spectrum and rank selection

For representative patches, plot:

- eigenvalues of residual covariance;
- estimated noise eigenvalues;
- positive eigenvalues of `Sigma_r - Sigma_n`;
- selected rank and uncertainty band;
- rank stability across temporal blocks.

### Figure V6 — Parzen dictionaries and noisy density

For selected Stage-2 components, show:

- posterior source samples;
- dictionary centers;
- source bandwidth `h`;
- projected noise standard deviation `nu`;
- clean Parzen density;
- noise-convolved observed density;
- posterior mean shrinkage curve `y -> E[s|y]`.

This figure must distinguish source smoothing from observation-noise broadening.

### Figure V7 — Local components

For every accepted component in a bounded representative set, show:

- spatial map;
- annularity/localization metrics;
- noisy output trace;
- posterior clean trace;
- dynamic drive or first difference;
- quiet and event windows;
- motion-edge correlation;
- acceptance/rejection reason.

Rejected structured components must have a separate artifact gallery.

### Figure V8 — B/S/A/N leakage matrix

On synthetic and semi-synthetic data, plot the normalized leakage matrix:

```text
                 estimated B  estimated S  estimated A  estimated N
true B
true S
true A
true N
```

Each row must sum to approximately one after closure normalization. Include
confidence intervals across seeds and conditions.

### Figure V9 — Residual-noise diagnostics

Show:

- temporal autocorrelation by lag;
- spatial correlation versus distance;
- residual power spectral density;
- Q-Q or tail plot against the chosen noise family;
- residual variance versus predicted intensity;
- event-triggered average residual at known ROIs;
- residual structure at motion edges.

A residual may be labeled `measurement_noise` only if these diagnostics pass the
declared thresholds.

### Figure V10 — Event preservation

For each known ROI/burst observation and for semi-synthetic injected events,
show summary distributions for:

- amplitude ratio;
- temporal-area ratio;
- onset shift;
- peak shift;
- duration ratio;
- localization shift;
- footprint overlap;
- signal energy moved into background, artifact, and noise.

Include per-ROI paired plots for shared failures.

### Figure V11 — Detection performance

Provide both:

1. recall versus quiet-calibrated candidate burden;
2. recall at a fixed candidate budget.

Include:

- Raw Direct;
- existing pairwise/difference references;
- latent filter and smoother amplitude;
- amplitude PCA rank 8;
- Stage-1 residual amplitude;
- Stage-2 neural reconstruction;
- accepted-component evidence;
- optional bounded fusion only after single-lane gates pass.

Unmatched candidates are `unknown`; do not label the x-axis false positives.

### Figure V12 — Shared-failure matrix

Rows are burst-specific known-label observations. Columns are complementary
methods. Display:

- detected;
- missed despite local pre-NMS evidence;
- below threshold;
- suppressed by NMS;
- localization mismatch;
- no visible evidence;
- unresolved/manual review.

This must distinguish representation failure from detector and matching failure.

### Figure V13 — Stability plots

Show across seeds, temporal blocks, patch offsets, bandwidths, dictionary sizes,
and nearby ranks:

- Stage-1 angle variation;
- background-map correlation;
- Stage-2 demixer alignment;
- accepted-component Jaccard;
- candidate-list Jaccard;
- known-event preservation variation;
- leakage-matrix variation.

### Figure V14 — Real-time profile

Show per-frame or per-chunk:

- Stage-1 inference latency;
- Stage-2 projection latency;
- posterior Parzen denoising latency;
- overlap-add latency;
- total p50/p95/p99 latency;
- adaptation latency and frequency;
- CPU/GPU memory;
- dropped/fallback update count.

Draw the 20 ms Spon deadline as a vertical reference.

## Mandatory quantitative metrics

### 1. Reconstruction closure

\[
E_{\mathrm{closure}}=
\frac{
\|Y-\widehat B-\widehat S-\widehat A-\widehat N\|_F^2
}{
\|Y\|_F^2+\epsilon
}.
\]

Also report maximum per-frame closure error and 99th percentile absolute error.

### 2. Semi-synthetic channel recovery

For true channel \(C_i\) and estimated channel \(\widehat C_j\), report
projection/leakage energy:

\[
L_{ij}=
\frac{
\|\operatorname{Proj}_{\widehat C_j}(C_i)\|_F^2
}{
\|C_i\|_F^2+\epsilon
}.
\]

Because projection definitions can favor high-rank channels, also report direct
component NMSE and aligned correlation.

### 3. Stage-1 metrics

- background NMSE/correlation on synthetic data;
- signal leakage into background;
- background leakage into residual;
- background common-direction cosine;
- first/second derivative energies;
- classification margin and unresolved rate;
- demixing angle drift;
- whitening condition number;
- event-amplitude and area preservation after Stage 1.

### 4. Stage-2 source metrics

- aligned source correlation or SI-SDR;
- structured-signal NMSE;
- component spatial IoU;
- annular-footprint score;
- temporal-trace correlation;
- posterior denoising gain;
- accepted/rejected component stability;
- noise-corrected rank selection stability;
- demixer orthogonality error;
- Parzen dictionary occupancy, replacement rate, and effective sample size.

### 5. Noise metrics

- quiet variance reduction;
- temporal autocorrelation norm over declared lags;
- spatial correlation integral beyond the optical scale;
- event-triggered residual energy;
- residual correlation with raw intensity and gradients;
- intensity-conditioned variance calibration error;
- standardized residual mean, variance, and tail rate;
- whiteness-test statistics as descriptive diagnostics, not sole pass criteria.

### 6. Event preservation

For every known or injected event:

- peak amplitude ratio;
- temporal-area ratio;
- onset error;
- peak-time error;
- duration error;
- centroid error;
- footprint IoU/annular overlap;
- background leakage fraction;
- noise leakage fraction;
- artifact leakage fraction.

### 7. Downstream detection

- macro and pooled known-label recall;
- burst-specific recall;
- fixed-budget recall;
- quiet-calibrated candidate burden;
- candidate count at equal recall;
- match-radius sensitivity at 4/6/8/10 px;
- annular/footprint-aware matching;
- pre-NMS versus post-NMS recovery;
- shared-miss frequency and method-set Jaccard;
- manual acceptance rate for a stratified candidate panel.

Ordinary precision, specificity, false-positive rate, and true-negative rate are
unidentified until exhaustive or reviewed negatives exist.

### 8. Stability

- parameter coefficient of variation;
- demixer subspace angles;
- component-map matched correlations;
- candidate-list Jaccard;
- known-event recovery consistency;
- decomposition-channel correlation;
- leakage-matrix confidence intervals;
- fallback/unresolved rates.

### 9. Runtime

- fitting time by stage;
- inference p50/p95/p99;
- adaptation p50/p95/p99;
- throughput in frames/s;
- peak CPU RSS;
- peak GPU allocated/reserved memory;
- disk and artifact sizes;
- dictionary and patch counts;
- real-time deadline margin.

## Primary scorecard

The report front page should contain:

| Goal | Primary metric | Guardrail |
|---|---|---|
| Separate background | neural leakage into background | unresolved rate and event preservation |
| Recover structured signal | semi-synthetic signal NMSE / event preservation | artifact contamination and component stability |
| Isolate measurement noise | residual event-locking and correlation norms | closure and variance calibration |
| Improve NeuRev evidence | fixed-budget known-label recall | candidate burden and sparse-positive semantics |
| Support real time | total inference p95 | 20 ms deadline, fallback rate, no online smoother |

## Advancement gates

### V0 — Visual integrity

Every mandatory array and figure has correct axes, alignment, fixed scales,
metadata, and no incomplete/partial artifact.

### V1 — Stage-1 background separation

Advance only if semi-synthetic neural leakage into background is below the
preregistered bound, background reconstruction improves over fixed subtraction,
and unresolved classification is surfaced rather than forced.

### V2 — Stage-2 noise separation

Advance only if structured-signal recovery improves over noise-agnostic ICA and
the final residual has less temporal/spatial/event structure than the Stage-1
residual.

### V3 — End-to-end decomposition

Advance only if closure, leakage, event preservation, and stability pass together.
A smooth video or low entropy score cannot compensate for signal loss.

### V4 — Real-data evidence

Advance only if known event amplitude/area/timing remain within declared bounds
and complementary detection metrics improve consistently.

### V5 — Real-time candidacy

Advance only if frozen causal inference meets the frame deadline at p95 and slow
adaptation has safe freeze/rollback behavior.
