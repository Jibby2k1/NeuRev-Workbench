# Dependent multiscale demixing: methods and meeting brief

Prepared 2026-08-02 for discussion on 2026-08-03.

## One-minute summary

We tested whether 5-, 7-, and 15-pixel spatial views of the same Spon Ca Burst
movie could be partitioned into background, neural signal, structured artifact,
and residual channels while preserving exact reconstruction. The implementation
is numerically reversible and produces useful failure diagnostics, but it is not
yet a validated biological source separation.

The main failure is scientifically informative: broad coordinated fluorescence
can resemble broad background. The original diagnostic assigned much of the
central burst to background. A population-preserving revision improved generated
source attribution, and a later coherence/carrier-confirmed revision reduced
overall leakage further, but neither could simultaneously preserve broad neural
events and reject drift/motion across all required morphologies. Therefore the
accepted carrier remains the scientific trace, the residual remains a
`noise_candidate`, and semi-synthetic or real scientific promotion is blocked.

## Artifact to open during the meeting

`Outputs/HierarchicalParzenICA/spon_ca_burst_dependent_multiscale_grayscale_review_v1/grayscale_decomposition_review.mp4`

- 560 frames, corresponding to UI frames 1800--2359 inclusive;
- 10 frames/s playback, 56 seconds total;
- H.264, 1146 by 570 pixels;
- SHA-256:
  `053ad79f3e28a582ef037454164582a1e29beffdbcdec984efd505a63ce430f9`.

The playback rate is for review and does not claim the acquisition frame rate.
The source filename contains `20ms`, but the frame period is not embedded
acquisition metadata.

## Data and coordinate contract

The displayed raw observation is the review interval from:

`Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy`

The decomposition input and accepted scientific carrier are:

`Outputs/HierarchicalParzenICA/spon_ca_burst_feature_utility_v1/features/carrier_signed.npy`

The raw movie is used only for visual anatomical context in this video. The
demixing proxy operates on the signed, quiet-standardized carrier. This
distinction is important: the decomposition is not a direct physical unmixing
of raw photon counts.

UI frames are one-based and inclusive. NumPy intervals are zero-based and
half-open. Coordinates use `x=column`, `y=row`. The 79 rings are sparse-positive
evaluation labels. Pixels without rings are unknown, not negative.

## Original decomposition method behind the grayscale video

### 1. Deterministic multiscale views

The same signed carrier movie, denoted (X), is spatially averaged with
reflection padding at three odd supports:

\[
Y_5 = R_5X,\qquad Y_7 = R_7X,\qquad Y_{15} = R_{15}X.
\]

Here (R_s) is a normalized (s\times s) box operator applied independently
to each frame. These are views of one observation, not separate biological
sources. No additional normalization is applied because the carrier was
already quiet-standardized.

### 2. Full-frame linear diagnostic proxy

The real-data diagnostic uses a scalable full-frame proxy rather than the full
patchwise model proposed in the implementation brief. Its initial channels are:

\[
B_0 = \operatorname{MA}_{31}(Y_{15}),
\]

where the moving average is along time with nearest-end padding;

\[
S_0 = \frac{Y_5+Y_7}{2}-B_0,
\]

\[
A_0 = \frac{Y_5-Y_7}{2},
\]

and

\[
N_0 = X-B_0-S_0-A_0.
\]

Thus broad, temporally persistent structure enters background; compact
5/7-scale agreement relative to that background enters neural signal; compact
scale disagreement enters artifact; and the exact remainder enters
`noise_candidate`.

### 3. Group-dependence refinement

Two label-free nuisance columns are constructed per frame:

1. global mean carrier intensity;
2. a linear slow-drift coordinate from -1 to +1.

For each candidate group, rank-four temporal coordinates are estimated from a
bounded sample of spatial columns. The background and artifact temporal bases
are residualized against the nuisance matrix and orthonormalized. With fixed
authority 0.75, the component of neural signal explained by the background
basis is moved from signal to background; the remaining signal component
explained by the artifact basis is moved from signal to artifact:

\[
C_B=0.75\,P_BS_0,
\]

\[
C_A=0.75\,P_A(S_0-C_B),
\]

\[
\widehat S=S_0-C_B-C_A,
\quad
\widehat B=B_0+C_B,
\quad
\widehat A=A_0+C_A.
\]

The residual remains (widehat N=N_0), and numerical closure is recomputed:

\[
E_{\mathrm{closure}}=X-\widehat B-\widehat S-\widehat A-\widehat N.
\]

Matrix-based Rényi group-dependence values are reported diagnostically after
this bounded projection. This real-data proxy is not a fully optimized
matrix-Rényi patchwise model.

### 4. Why the broad burst moved into background

The group rule treats neural temporal structure predictable from a broad
background basis as inappropriate between-group dependence. A coordinated
population burst is itself broad and shared, so this rule can remove legitimate
neural activity. The UI-2005 frame makes that failure visible: the central burst
is bright in background, while the positive neural panel emphasizes local and
edge-like structure.

This is the principal reason the video is diagnostic rather than a promoted
scientific result.

## Exact grayscale rendering method

The video is rendered from the immutable dense channels in:

`Outputs/HierarchicalParzenICA/spon_ca_burst_dependent_multiscale_real_v1/reconstruction`

It does not recompute or modify the decomposition. Each panel uses one fixed
range for all 560 frames so apparent changes are temporal changes, not
frame-wise contrast rescaling.

The six panels are:

1. **Raw observation** -- anatomical/intensity context from the original movie.
2. **Accepted carrier** -- the signed quiet-standardized scientific trace that
   remains authoritative.
3. **Neural signal: positive only** -- `max(structured_signal, 0)`; negative
   values are intentionally omitted from this panel for easier event review.
4. **Background** -- the signed reconstructed broad/persistent channel.
5. **Structured artifact** -- signed 5/7-scale disagreement after refinement.
6. **Noise candidate** -- the unqualified residual; it is not measurement noise.

For raw observation, the fixed display limits are sampled 0.5th and 99.5th
percentiles. For signed panels, the limits are symmetric about zero using the
sampled 99.5th percentile of absolute magnitude. For the positive-only neural
panel, the lower limit is zero and the upper limit is its sampled 99.5th
percentile.

Signed grayscale has the persistent interpretation:

- black = negative;
- mid-gray = zero;
- white = positive.

Raw and positive-only panels use black = low and white = high. Numeric limits
are printed above every panel. Sparse-positive labels are black-backed white
rings to remain visible without introducing a semantic color palette. Frames
are resized bilinearly into a 3-by-2 layout. Encoding uses `libx264`, CRF 18,
two threads, and an atomic temporary MP4 that is renamed only after successful
completion.

## Validation and resource history

The decomposition closes numerically to approximately (10^{-7}) normalized
maximum error. The first dense v1 run produced the arrays used for this video
but exceeded its declared 12 GiB peak-memory cap. It is retained unchanged as
an engineering diagnostic. A later v3 implementation eliminated avoidable
whole-movie copies and completed at 10,679.7 MiB peak RSS under the 12,288 MiB
cap. v3 confirmed the same scientific failure but did not write dense channels;
therefore the grayscale re-render uses the equivalent completed v1 arrays.

## Population-preserving W5b method

W5b introduced 15-pixel patches, 10-pixel stride, floored-Hann overlap-add, and
an explicit population hierarchy. A transient broad component was moved from
background to neural signal, and 25% of residual structure synchronous with
the protected neural temporal subspace was returned from `noise_candidate` to
signal. Positive-trace amplification was bounded.

Across 15 exact-truth fixtures and three seeds, W5b reduced median neural
leakage by 14.79% and improved diagonality. Aggregate peak and temporal-area
ratios were 1.012 and 1.062, but subgroup preservation failed: broad neural
activity was attenuated while broad drift and motion-crossing cases were
amplified. W6 was therefore blocked.

## Coherence/carrier-confirmed W5c method

W5c tested whether independent confirmation could control population authority.
All thresholds were derived from the first eight quiet frames of each generated
fixture; labels and exact truth were used only for evaluation.

### Confirmation features

**Coherence authority.** The existing causal local-correlation operator computes
rolling 15-frame correlation between each pixel and a Gaussian neighborhood
with spatial sigma 1.5 pixels. Only current and past frames enter each value.
Correlation is activity-qualified and calibrated against the quiet 99.5th
percentile, then mapped into [0,1].

**Carrier authority.** A quadratic temporal trend is removed from the 5-pixel
view. Positive innovation above the quiet null is mapped into [0,1]. This is a
constraint signal, not a replacement output.

**Motion suppression.** Absolute disagreement between 5- and 7-pixel views is
quiet-calibrated into [0,1] and divides confirmation authority.

### Frozen lanes

The experiment contains a geometry-matched orthogonal baseline plus four
candidate lanes:

1. W5b population reference;
2. coherence-confirmed authority;
3. carrier-constrained authority;
4. coherence-plus-carrier primary lane.

The primary combined authority is:

\[
q(p,t)=
\frac{\sqrt{q_{\mathrm{coh}}(p,t)q_{\mathrm{carrier}}(p,t)}}
{1+2q_{\mathrm{motion}}(p,t)}.
\]

W5b transfers are interpolated from the safe patchwise baseline using (q).
Where confirmed positive carrier innovation exceeds reconstructed positive
signal, a bounded correction is moved from `noise_candidate` to signal while
preserving closure. Authority at or above 0.5 is reported as confirmed, at or
below 0.1 as nuisance-like, and intermediate values as unresolved. These states
are diagnostics, not biological labels.

### W5c result

The completed artifact is:

`Outputs/HierarchicalParzenICA/dependent_multiscale_confirmation_w5c_generated_v1`

For the primary combined lane:

- median leakage fell from `0.40269` to `0.16162` (`59.86%` overall);
- median diagonality changed from `0.47095` to `0.47470`;
- aggregate peak and area ratios were `0.91490` and `0.91840`;
- numerical closure passed;
- required broad-neural attribution failed: leakage worsened by `80.43%`;
- broad-neural peak/area were `0.695/0.513`;
- broad-drift peak/area were `1.131/1.861`;
- motion-crossing peak/area were `1.103/2.438`;
- C1 passed, C2 failed, C3 failed, C4 was not qualified, and C5 remained
  diagnostic only.

Coherence-only produced the largest overall leakage reduction (`66.68%`) and
passed aggregate preservation, but it also worsened broad-neural attribution
and failed subgroup preservation. Carrier-only was more conservative but
attenuated multiple neural morphologies. No ablation passed the full gate.

## What can and cannot be concluded

### Supported conclusions

- The implementation is reversible and exposes source-assignment failures.
- Compact/coherent confirmation contains useful attribution information.
- Overall or median performance can hide severe morphology-specific errors.
- Broad neural activity, slow drift, and motion require more explicit measured
  context than multiscale dependence alone provides.

### Unsupported conclusions

- The displayed channels are not validated physical sources.
- The residual is not qualified measurement noise.
- Sparse labels do not establish precision or make unlabeled pixels negative.
- The method has not passed semi-synthetic W6 or scientific real-data W7.
- The accepted carrier has not been replaced.
- Coherence is association, not causal biological propagation.

## Suggested questions for the meeting

1. Should the scientific objective remain physical source decomposition, or
   should these maps be treated as proposal/ranking diagnostics around the
   accepted carrier?
2. Is broad coordinated fluorescence expected to be biologically meaningful
   population activity in this preparation, and what temporal/morphological
   priors would be defensible?
3. Are motion estimates, illumination measurements, microscope metadata, or
   acquisition-channel semantics available to disambiguate nuisance structure?
4. Would an exhaustively annotated bounded right-field region be feasible for
   neuron/artifact/background/unresolved and center/ring/crowding labels?
5. Is using the accepted carrier as a preservation constraint scientifically
   acceptable, or too circular for the intended claim?
6. Should the next experiment focus on the already successful `coherence_w15`
   ranking/confirmation panel instead of forcing a physical decomposition?
7. What amplitude and temporal-area tolerances are scientifically meaningful
   for broad, annular, crowded, and motion-overlap cases?

## Reproducibility pointers

- Original real diagnostic:
  `neurobench/experiments/hierarchical_parzen_ica/dependent_multiscale_real.py`
- Group refinement:
  `neurobench/experiments/hierarchical_parzen_ica/dependent_multiscale_information.py`
- Grayscale renderer:
  `neurobench/experiments/hierarchical_parzen_ica/dependent_multiscale_grayscale.py`
- Population-preserving W5b:
  `neurobench/experiments/hierarchical_parzen_ica/dependent_multiscale_population.py`
- W5c confirmation authority:
  `neurobench/experiments/hierarchical_parzen_ica/dependent_multiscale_confirmation.py`
- W5c generated evaluation:
  `neurobench/experiments/hierarchical_parzen_ica/dependent_multiscale_confirmation_evaluation.py`
- Frozen local-coherence operator:
  `neurobench/algorithms/scientific_feature_audit.py`

The complete repository test suite must pass before these methods are shared as
reproducible software evidence.
