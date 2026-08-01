# Experiment, Metrics, and Visualization Contract

This document defines the evidence required to decide whether hierarchical
Parzen ICA followed by noisy Parzen ICA advances NeuRev's overarching goal:

> Separate persistent background, biologically useful dynamic fluorescence, and
> measurement noise/artifact without erasing neural events, hallucinating
> structure, or producing an unstable decomposition.

Visual results are mandatory but never sufficient. Every visual panel must be
paired with relevant attribution, preservation, residual, stability, detection,
and latency metrics.

## 1. Evaluation order

Run and interpret experiments in this order:

1. exact numerical tests;
2. fully synthetic scalar/vector mixtures;
3. semi-synthetic movies using real quiet backgrounds;
4. tiny real-data smoke application;
5. Stage-1 real-data diagnostic, only after gates 1-3;
6. Stage-2 real-data diagnostic, only after Stage 1 passes;
7. frozen single-lane detection comparison;
8. optional bounded fusion;
9. real-time profiling;
10. full Spon run only after explicit selection.

Do not use known labels to fit either separation stage. Labels enter only after
all denoiser parameters and component-selection rules are frozen.

## 2. Baselines

Every comparison must include the following when available:

### Identity and simple temporal anchors

- `identity_observation`;
- `quiet_median_residual` / Raw Direct input;
- `fixed_common_mode_subtraction`;
- `fixed_difference_lag1`;
- `adaptive_gain_difference`.

### Existing information-theoretic anchors

- pairwise InfoMax;
- pairwise batch CS-Parzen;
- three-frame fixed common/slope/curvature transform;
- ordinary batch ICA for Stage 2 without explicit noise correction.

### Current NeuRev performance anchors

- Raw Direct detector;
- offline latent smoother amplitude;
- causal latent filter amplitude;
- amplitude PCA rank 8;
- selected morphology/CFAR anchor if its existing output is available.

### Non-ICA denoising controls

At least one of:

- local low-rank/PCA reconstruction;
- PMD import or bounded local PMD implementation;
- constrained NMF/CNMF-style local factorization;
- Wiener/linear Gaussian reference on synthetic data.

The purpose is not to prove ICA always wins. It is to identify the assumptions
under which the hierarchy adds value.

## 3. Synthetic suite

### 3.1 Stage-1 temporal mixtures

Create deterministic mixtures with known component contributions:

1. static background + white noise;
2. static background + transient signal + white noise;
3. linear background drift;
4. exponential photobleaching;
5. slow neural ramp that resembles background;
6. long calcium plateau;
7. two dynamic sources with different time scales;
8. gain change between adjacent frames;
9. impulsive frame artifact;
10. translated spatial edge.

For pair and triple embeddings, preserve ground-truth current-coordinate
background and dynamic contributions.

### 3.2 Stage-2 noisy source mixtures

Use local source ranks 1-8 and observation dimensions larger than the source
rank. Include:

- Laplace, logistic, generalized-Gaussian, and multimodal sources;
- sparse positive drives convolved with exponential kinetics;
- correlated but not independent neural sources;
- white Gaussian noise;
- colored Gaussian noise;
- diagonal heteroscedastic noise;
- diagonal-plus-low-rank noise;
- Poisson-Gaussian approximations;
- impulsive outliers;
- varying SNR;
- incorrect rank;
- incorrect noise covariance;
- incorrect Parzen bandwidth;
- changing source distribution.

### 3.3 End-to-end synthetic movies

Generate

\[
X=B+S+N+O,
\]

with:

- static anatomy-like texture;
- broad low-rank illumination/background dynamics;
- multiple localized annular neural footprints;
- overlapping footprints;
- variable onset, rise, plateau, decay, and recurrence;
- intensity-dependent measurement noise;
- motion edges and saturation artifacts.

Retain every true component separately so the full attribution matrix can be
computed.

## 4. Semi-synthetic NeuRev suite

Use real quiet frames from the declared Spon interval as the background/noise
substrate. Inject neural events with known clean ground truth.

### Spatial injection families

- filled disk;
- annulus/ring;
- asymmetric membrane arc;
- overlapping annuli;
- edge-adjacent soma;
- border soma;
- diffuse excitation zone.

### Temporal injection families

- one-frame impulse;
- rapid rise/slow decay;
- slow rise/slow decay;
- sustained plateau;
- two closely spaced events;
- weak recurrent event across all four burst-length windows;
- source synchronized with a second source;
- source synchronized with global illumination drift.

### SNR and nuisance grid

Cross a bounded set of:

- amplitude/SNR;
- footprint size;
- decay time;
- local background brightness;
- gain drift;
- translation magnitude;
- noise heteroscedasticity;
- saturation proximity.

Use deterministic seeds and write an injection manifest.

## 5. Decomposition metrics

### 5.1 Closure

Define

\[
E_{\mathrm{closure}}
=X-\widehat B-\widehat S-\widehat N.
\]

Report:

```text
relative Frobenius closure error
per-frame median/p95/p99/max relative error
per-patch median/p95/p99/max relative error
largest absolute pixel error
nonfinite count
```

Gate: numerical closure p99 must be at or below `1e-5` relative to the local
observation scale, unless a stricter implementation tolerance is declared.

### 5.2 Channel reconstruction

For known true channels:

\[
\operatorname{NMSE}(C,\widehat C)
=\frac{\|C-\widehat C\|_F^2}{\|C\|_F^2+\epsilon}.
\]

Report NMSE, correlation, scale-aligned NMSE, and structural similarity for
background, signal, and noise/artifact separately.

### 5.3 Attribution leakage matrix

Write both correlation-like and energy-attribution matrices.

Correlation-like:

\[
L^{\mathrm{corr}}_{ij}
=\frac{\langle C_i,\widehat C_j\rangle_F^2}
{(\|C_i\|_F^2+\epsilon)(\|\widehat C_j\|_F^2+\epsilon)}.
\]

Energy attribution should be computed through projection or regression of each
true channel onto the estimated channels and must sum approximately to one after
including unexplained error.

Report at least:

- true signal -> estimated background;
- true signal -> estimated noise;
- true background -> estimated signal;
- true noise/artifact -> estimated signal;
- diagonal dominance;
- matrix variation across seeds and perturbations.

Primary goal metric:

```text
signal_preserved_in_signal_channel
minus max(signal_leakage_to_background, signal_leakage_to_noise)
```

### 5.4 Stage-1 specific attribution

Report:

- background reconstruction NMSE;
- signal energy absorbed into Stage-1 background;
- background energy left in Stage-1 residual;
- Stage-1 assignment accuracy on synthetic mixtures;
- ambiguous/fallback assignment rate;
- selected temporal direction angle to common and derivative references;
- background score margin.

### 5.5 Stage-2 source recovery

Report:

- Amari index when the square identifiable case applies;
- subspace principal angles in undercomplete cases;
- matched source-map correlation;
- matched time-course correlation;
- posterior source NMSE;
- source split/merge count;
- accepted/rejected source accuracy on synthetic data;
- posterior interval coverage.

## 6. Neural-signal preservation metrics

For every injected or known ROI/event:

### Amplitude and energy

- peak amplitude ratio;
- peak amplitude bias;
- temporal-area ratio;
- total event-energy ratio;
- signal-to-quiet contrast before/after.

### Timing

- onset error in frames and milliseconds;
- peak-time error;
- rise-time bias;
- duration bias;
- decay-time bias;
- any acausal pre-echo or backward extension.

### Spatial localization

- centroid error;
- point-radius recovery at 4/6/8/10 px;
- footprint IoU;
- annular overlap;
- membrane-ring distance;
- footprint fragmentation and merging;
- border loss.

### Leakage

- event energy in Stage-1 background;
- event energy in Stage-2 noise candidate;
- event-locked residual z-score;
- maximum local coherent residual around the known footprint.

No method passes signal preservation merely because detection recall rises.

## 7. Noise/residual validity metrics

The final residual is called `noise_candidate` until these tests are reported.

### Temporal structure

For lags 1 through a declared maximum:

\[
E_{\mathrm{ACF}}
=\sum_{\tau=1}^{L}w_\tau\rho(\tau)^2.
\]

Report median and upper quantiles over pixels/patches, plus Ljung-Box or a
comparable descriptive test when assumptions are stated.

### Spectral structure

Report:

- median residual PSD;
- ratio to held-out quiet PSD by frequency band;
- low-frequency excess;
- line-frequency/narrowband excess;
- high-frequency excess caused by differencing or artifacts.

### Spatial structure

Report radial spatial autocorrelation and correlation length. Include residual
correlation with:

- raw spatial gradient magnitude;
- motion-edge templates;
- Stage-1 background;
- accepted signal footprints.

### Intensity-dependent calibration

Bin pixels by predicted/observed intensity and compare empirical residual
variance with the declared noise model. Report calibration slope, intercept,
relative error, and worst-bin error.

### Event locking

Compute residual energy during known/injected events versus matched quiet
windows. Significant or practically large event-locking is evidence that signal
was discarded as noise.

### Tail behavior

Report robust kurtosis, tail probabilities, and outlier maps. Do not reject a
method solely for non-Gaussian residuals; use these diagnostics to decide whether
the Gaussian noise model is inadequate.

## 8. Information-theoretic and optimization metrics

### Stage 1

- CS-divergence/objective by iteration or angle;
- stochastic objective estimate by update;
- gradient norm;
- demixer change;
- orthogonality error;
- effective sample/dictionary count;
- bandwidth sensitivity;
- component assignment score terms;
- assignment continuity over windows.

### Stage 2

- ordinary versus noisy Parzen objective;
- log-likelihood under the noise-convolved Parzen density;
- responsibilities entropy;
- effective number of active centers;
- posterior shrinkage as a function of noise level;
- source bandwidth and projected noise variance separately;
- demixer change, condition, and decorrelation error;
- component acceptance score;
- rank-zero and unresolved-patch rates.

Optimization convergence may not be equated with scientific separation.

## 9. Stability metrics

### Seed/restart stability

Match components using maximum-correlation assignment and report:

- mean/median matched absolute correlation;
- fraction above 0.9;
- subspace angle;
- assignment consistency;
- accepted-source count variation.

### Temporal-block stability

Fit or calibrate on contiguous blocks and report variation in:

- Stage-1 directions and assignments;
- noise covariance;
- Stage-2 subspace and sources;
- leakage metrics;
- candidate sets.

### Hyperparameter perturbation

Perturb one factor at a time around the selected configuration:

- Stage-1 bandwidth;
- Stage-1 dictionary size;
- assignment weights/margin;
- Stage-2 patch size and overlap;
- rank;
- source bandwidth;
- noise-variance multiplier;
- dictionary size;
- learning rate/update interval.

Report local sensitivity, not only a grid winner.

### Output/candidate stability

- background/signal/noise channel correlation;
- candidate Jaccard;
- known-label match retention;
- source-map overlap;
- signal-preservation variation.

Suggested gate: candidate Jaccard at least 0.70 across accepted nearby settings,
with no material known-event loss.

## 10. Downstream NeuRev detection metrics

Apply the same frozen detector/evaluation contract to every eligible single
feature before fusion.

### Required policies

1. quiet-calibrated threshold;
2. exactly 58 candidates per burst, matching the existing capacity reference;
3. optional full FROC curve over declared quiet peak rates.

### Required outputs

- per-burst known recall;
- pooled and macro known recall;
- total candidates;
- localization error;
- known-label candidate fraction lower bound, clearly named as such;
- shared and method-specific misses;
- unmatched candidates as unknown;
- candidate-set overlap with Raw Direct and current best lanes;
- recall-candidate tradeoff.

### Fusion

Fusion is a later gate. It must:

- initialize exactly at Raw Direct;
- initialize all hierarchical auxiliary weights at zero;
- use nonnegative bounded weights;
- include a trust-region/L2 pull to the Raw anchor;
- use leakage-safe outer folds;
- report whether any gain is caused only by additional candidates.

Suggested advancement gate:

- at least `+0.03` macro recall with wins in at least 3/4 bursts at fixed budget;
  or
- at least 20% candidate reduction with no known-label recall loss.

## 11. Real-time metrics

Profile separately:

```text
input/calibration
Stage-1 embedding
Stage-1 projection/reconstruction
Stage-2 patch extraction
Stage-2 projection
Parzen posterior inference
overlap-add
feature pooling/detection
total per-frame inference
slow Stage-1 update
slow Stage-2 demixer update
slow dictionary update
```

Report p50, p95, p99, maximum, throughput, warmup, memory, GPU allocation, and
CPU threads. Use native image dimensions and realistic component/dictionary
counts.

Real-time gate for the 50 Hz Spon recording:

- frozen/causal total p95 below 20 ms;
- no p99 deadline failure in the measured sequence, or an explicitly bounded
  fallback policy;
- slow updates scheduled without violating the frame deadline;
- stable component sign/permutation tracking;
- fallback latency included.

The offline result may be scientifically useful even when this gate fails.

## 12. Mandatory figures

The report builder must emit the following exact filenames.

### 12.1 Method and accounting

- `hierarchy_overview.png`: schematic generated from the resolved config.
- `fixed_scale_channel_montage.png`: raw, background, Stage-1 residual,
  structured signal, noise candidate, closure.
- `closure_by_frame.png`: closure quantiles and worst-frame index.
- `energy_accounting_by_frame.png`: energy in each channel over time.

### 12.2 Component interpretation

- `stage1_component_diagnostics.png`: maps/traces, derivative energies, spatial
  metrics, assignment score/margin.
- `stage1_direction_geometry.png`: fitted temporal directions against common,
  slope, and curvature references.
- `stage2_component_diagnostics.png`: accepted/rejected maps/traces, posterior
  SNR, uncertainty, and stability.
- `parzen_density_posteriors.png`: latent Parzen prior, noise-convolved observed
  density, responsibilities, posterior mean, and variance for representative
  sources/noise levels.
- `patch_rank_and_acceptance.png`: selected ranks and accepted-source counts.

### 12.3 Overarching separation goal

- `leakage_matrix.png`: median semi-synthetic background/signal/noise attribution.
- `leakage_matrix_by_seed.png`: variability or confidence interval.
- `signal_preservation.png`: amplitude, area, onset, duration, footprint metrics.
- `residual_validity.png`: temporal ACF, spatial ACF, PSD ratio, and event-locking.
- `intensity_noise_calibration.png`: empirical versus modeled residual variance.
- `failure_case_gallery.png`: worst leakage, missed event, motion, drift, and
  ambiguous Stage-1 assignment examples.

### 12.4 Downstream utility and stability

- `recall_candidate_tradeoff.png`;
- `fixed_budget_recall.png`;
- `per_burst_recall.png`;
- `candidate_jaccard.png`;
- `parameter_sensitivity.png`;
- `component_stability.png`;
- `latency_breakdown.png`.

Every figure must have a machine-readable source TSV/JSON/NPZ recorded in a
figure manifest.

## 13. Mandatory TIFF/video review artifacts

Use fixed scales and embedded metadata. Preregister a bounded set:

1. `stage1_background.tif`;
2. `stage1_residual.tif`;
3. `stage2_structured_signal.tif`;
4. `stage2_noise_candidate.tif`;
5. `six_channel_fixed_scale_diagnostic.tif` or a synchronized video;
6. sparse-label overlay for the structured-signal lane;
7. failure-case clips for the frozen shared-miss panel.

Do not emit every grid condition as a multi-gigabyte TIFF. Full dense arrays and
display TIFFs are separate artifacts.

## 14. Mandatory tables

- `stage1_fit_summary.tsv`;
- `stage1_component_scores.tsv`;
- `stage2_fit_summary.tsv`;
- `stage2_component_summary.tsv`;
- `decomposition_metrics.tsv`;
- `leakage_metrics.tsv`;
- `signal_preservation.tsv`;
- `residual_validity.tsv`;
- `stability.tsv`;
- `detection_lane_summary.tsv`;
- `per_burst_detection.tsv`;
- `latency.tsv`;
- `failure_cases.tsv`.

## 15. Advancement gates

### G0: implementation integrity

- exact shapes/axes/frame contracts;
- deterministic tiny results;
- no nonfinite values;
- collision-safe artifacts;
- Raw Direct anchor exact;
- all focused tests pass.

### G1: Stage-1 validity

On synthetic and semi-synthetic data:

- background NMSE improves over fixed common-mode subtraction in the declared
  nuisance cases;
- median neural energy leaked to background below 10%;
- event amplitude and area retained within 10% median error;
- background assignment resolved and correct in at least 90% of nonambiguous
  fixtures;
- ambiguous fixtures are reported rather than forced;
- stable direction/assignment across seeds and temporal blocks.

### G2: Stage-2 noisy-source validity

- noisy Parzen posterior source NMSE improves over ordinary Parzen ICA in a
  majority of predefined noisy regimes;
- posterior interval coverage is reasonably calibrated under matched cases;
- signal leakage to noise below 10% median in passed regimes;
- noise/artifact leakage to signal is lower than ordinary ICA;
- accepted component maps/traces have matched correlation at least 0.9 across
  accepted seeds, or the condition is explicitly rejected.

### G3: end-to-end attribution

- closure gate passes;
- median leakage matrix is diagonally dominant;
- true signal is primarily assigned to the signal channel;
- true background is primarily assigned to the background channel;
- residual noise/artifact is not primarily reconstructed as signal;
- gate holds across declared perturbations, not only one seed.

### G4: real signal preservation

- no material attenuation or timing distortion at known events;
- no systematic event-locked residual;
- spatial localization remains within declared tolerance;
- common failure cases are reviewed and categorized.

### G5: downstream utility

Pass the fixed-budget recall or candidate-reduction rule described above.
Detection value cannot override failed G1-G4 attribution/preservation gates.

### G6: real-time eligibility

Pass the latency and streaming continuity rules. Failure leaves the method as an
offline analysis lane.

## 16. Report narrative

The generated report must follow this order:

1. question and current NeuRev anchors;
2. exact hierarchical model and assumptions;
3. implementation integrity;
4. synthetic attribution;
5. semi-synthetic signal preservation;
6. residual validity;
7. real visual channels and component interpretation;
8. stability and failure cases;
9. downstream detection;
10. latency;
11. gates passed/failed/not run;
12. next justified action.

Do not lead with the prettiest TIFF or the best recall. Lead with whether the
method actually achieved a stable background--signal--noise attribution.
