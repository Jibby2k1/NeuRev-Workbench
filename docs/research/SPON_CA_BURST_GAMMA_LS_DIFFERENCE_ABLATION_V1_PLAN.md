# Spon Ca Burst Gamma-LS difference/ICA ablation v1

## Decision question

Does a learned temporal ICA transform add useful detection information beyond a
fixed temporal difference or an energy-normalized temporal difference when every
representation is passed through the same radial Gamma-weighted local
standardization and CFAR decision rule?

This experiment is also the naming gate for the paper. If the learned arms do
not improve the protected retrieval curve enough to justify their fit and
runtime, the main architecture will use **temporal differencing** rather than
ICA. ICA will remain an ablation and historical explanation.

The existing two-frame CS-Parzen effective direction is nearly derivative-like,
but its reported activity component and sign were chosen using correlation to
an analytic derivative reference. That result motivates this experiment; it is
not independent proof that ICA discovered differencing. The fixed-versus-learned
comparison below freezes every downstream operation before deciding whether the
learned model contributes anything beyond that reference.

## Terminology fixed before the run

- **Gamma-LS** is the continuous signed local-standardization score. Its
  reference moments use a radial Gamma-family kernel and an explicit guard
  region around the test pixel.
- **CFAR** is the one-sided threshold and spatial non-maximum-suppression
  decision applied to Gamma-LS. Gamma-LS and CFAR are not two independent
  filters.
- **Gamma-LS context search** means that multiple kernels are evaluated and one
  is selected. It is not called multiscale inference.
- **Gamma-LS multi-context fusion** is reserved for a future arm that actually
  consumes multiple scale-specific Gamma-LS maps at inference. Version 1 does
  not run such an arm and therefore makes no multiscale-LS claim.
- The current six-lag CS-Parzen lane is called a **multi-lag delay-embedding
  representation followed by selected-context local standardization**. The
  selected LS lane does not apply all searched contexts simultaneously.
- Event-window threshold occupancy is an evaluation reduction over a known
  burst interval. Version 1 uses no temporal maximum or max-pooling stage.

The maintained `robust_local_cfar` and current MSLN implementation use square
outer-minus-guard supports. They are controls only and cannot be relabeled as
Gamma-LS. The legacy voltage operator is retained as an exact anchor, including
its unusual 23-by-23, shape-9, nominal-mode-35 kernel and lack of an explicit
guard band.

## Canonical order

```text
Raw frames
  -> representation arm
  -> signed radial Gamma-LS
  -> one-sided CFAR threshold
  -> strict spatial NMS
  -> frame-level candidates
```

The protected occurrence assay reduces each burst to a two-dimensional
threshold-occupancy map only for matched evaluation. It does not use temporal
max pooling. The later full-recording pass keeps the frame-level candidate
stream without using annotated burst windows. Version 1 reports proposal rows
and proposals per frame; it does not claim a unique biological-event count
because a temporal-linking rule has not yet been frozen.

## Representation arms

The strict adjacent-frame comparison applies its five arms to the same causal
preprocessed movie
\(P_t=\operatorname{EMA}_{0.4}(G_{\sigma=1}*I_t)\). Scientific folds carry the
EMA state from acquisition frame 1. The archived pairwise fit used the same
operator but reset the EMA at UI frame 1800, so it is reproduced only in a
separate parity check and is not applied as the held-out model. All scientific
arms are aligned to the same output frames and use identical downstream
Gamma-LS/CFAR settings.

1. `raw`: the current preprocessed frame \(P_t\).
2. `difference_signed`: \(D_t=P_t-P_{t-1}\).
3. `difference_energy_normalized`:
   \(D_t/\sqrt{P_{t-1}^2+P_t^2+\epsilon}\), matching the existing analytic
   control.
4. `pca_whitened_derivative`: the full-rank covariance-whitened two-frame axis
   closest to the analytic derivative. This is the PCA control; no dimension is
   discarded and no independence rotation is fitted.
5. `cs_parzen_two_frame`: the two-frame CS-Parzen activity coordinate.
6. `difference_multilag_energy_normalized`: on the acquisition-raw domain used
   to fit v5, the root-sum-square of fixed pair-energy-normalized differences
   from the current frame to lags 1, 2, 4, 8, and 16.
7. `pca_whitened_delay_total_energy`: the root-sum-square of all six full-rank
   whitened acquisition-raw delay coordinates before any independence
   rotation. This is a whitening-only total-energy control, not a residual
   subspace.
8. `cs_parzen_delay_residual`: the frozen long-delay CS-Parzen residual-subspace
   energy on its original acquisition-raw input domain using lags 0, 1, 2, 4,
   8, and 16. These last three arms form a separate matched-support level rather
   than being used to invalidate or validate the adjacent-frame comparison.

For protected results, the two-frame whitening and CS-Parzen rotation are fit
separately inside every outer fold after excluding the held-out burst and its
10-frame guard. The archived whole-review fit is used only for implementation
parity and a clearly labeled transductive diagnostic; it cannot support a
held-out learned-model claim.

The comparisons `cs_parzen_two_frame` versus `pca_whitened_derivative` and
`difference_signed` isolate the independence rotation. The second,
matched-support level compares the long-delay CS-Parzen residual energy with
fixed normalized lag-difference energy and full-rank PCA-whitened total energy
using the same lags. The latter is deliberately not called a residual
subspace, because the residual coordinates are defined only after the learned
rotation. This level is required before making a conclusion about ICA broadly:
a two-frame ablation can reject pairwise ICA, but not the six-lag
representation. Because a full orthogonal ICA rotation preserves total
whitened energy, any difference between the PCA total-energy arm and the
four-component CS-Parzen residual arm tests the ICA rotation plus the analytic
post-fit persistence/difference-axis subspace rule, not an otherwise hidden
total-energy gain from rotation alone. The archived v5 fit also saw the whole
review interval and is diagnostic only. Any protected six-lag claim requires
fold-specific v5 refitting.

## Gamma-LS/CFAR search

The Gamma search uses successive halving. Two diagnostic controls are specified
for a later finalist comparison but are not run by the current G1/G2 screen:

- G1 crosses kernel half-width 7, 11, or 15 pixels with circular guard radius
  1, 3, or 5 pixels at shape 5 and mode radius 0.75 times the half-width;
- two G1 radius/guard pairs are retained inside each training fold;
- G2 crosses shape 2, 5, or 9 with mode-radius fraction 0.5, 0.75, or 1.0 for
  those pairs (at most 18 cells);
- the quiet-scale floor percentile remains fixed at 10 and is fitted separately
  for each context, representation, outer fold, and quiet-half swap;
- planned historical anchor: 23-by-23 square support, shape 9, nominal mode 35,
  center-only exclusion, reflect padding, and additive epsilon 64.
- planned square outer-minus-guard CFAR is a named diagnostic control and is never
  eligible as the primary Gamma-LS result.

Modern Gamma contexts use zero padding with per-pixel valid-weight
renormalization at borders; reflection is retained only for the exact legacy
anchor. Screening uses the 0.999 quantile across all spatial pixels. It averages
the three outer-training-burst quantiles and compares them with a held-out quiet
half, then reverses the two contiguous quiet halves and averages the two
contrasts. It records this burst-window-supervised contrast, held-out quiet
candidate burden, numerical health, kernel mass distribution, and synchronized
GPU cost.
To avoid letting a learned representation select its own downstream geometry,
G1 and G2 use only the three fixed arms (`raw`, `difference_signed`, and
`difference_energy_normalized`). Selection uses the predeclared burst windows
but never sparse-positive coordinates or identities. Within each outer fold,
the held-out burst is excluded, two G1 pairs and then one G2 context are chosen
from the other three bursts, and that fold-local context is applied unchanged
to every learned and analytic arm on the held-out burst. No context pooled
across all four bursts is assigned a protected score. The maximum screen is
108 G1 cells plus 216 G2 cells, followed by 32 fold-local common-context
representation evaluations. For the adjacent learned arm, three bandwidths by
three paired sample seeds are refit independently in each of four outer folds
(36 small fits). Their selection uses only training bursts, and fit time is
reported separately from inference. The six-lag refit stage is deferred until
the adjacent or fixed multi-lag gate warrants its much larger search.

It retains one context per outer fold, common to all representation arms in
that fold. A separate deployment context may be selected using all four bursts
only after the protected cross-validation summary is frozen; it cannot be
reported as a protected result on those same bursts.

Only retained contexts are run over the complete 560-frame review interval.
CFAR thresholds are cross-fitted: one contiguous quiet half calibrates 0.25,
0.5, 1, 2, or 5 NMS peaks per duration-matched pseudo-burst and the other half
measures realized burden, then the halves swap. This avoids assigning an
unvalidated Gaussian meaning to a nominal PFA or reporting calibration burden
as if it were held out. Candidate budgets 20, 40, 58, 80, and 100 are
evaluated after the continuous score is frozen. Strict Euclidean NMS uses six
pixels; matching uses a separate six-pixel one-to-one rule, with 4/8-pixel NMS
sensitivities.

## Population and leakage boundary

- Primary protected evaluation: the original frozen 79-inclusive-occurrence,
  26-canonical-identity adjudication table. Gamma-context selection is
  sparse-coordinate- and identity-free but uses the predeclared temporal burst
  windows. A held-out burst never selects its own context.
- Latest canonical-v7 sensitivity: 106 confirmed occurrence rows with 44
  canonical label identities; the downstream identity-safe feature audit
  partitions those rows into 50 immutable coordinate-defined sites. This cohort
  contains candidate-assisted additions, so it is internal-consistency evidence
  and cannot replace the protected result.
- Unmatched candidates remain unknown. Precision, specificity, and false
  positive rate are not identified by either sparse-positive table.
- Representation fits, Gamma kernels, thresholds, NMS, candidate budgets, and
  comparison rules are fingerprinted before protected labels are joined.

## Primary endpoints

1. Known-positive recall versus candidate budget, summarized as the area under
   the five-point budget curve and recall at B58 per burst.
2. Known-positive recall versus held-out quiet candidate burden over 0.25--5
   peaks per duration-matched pseudo-burst.
3. Candidate burden at each frozen CFAR threshold, reported as candidates per
   frame and per eligible block—not as false positives.
4. Representation equivalence: Pearson/Spearman correlation, normalized RMS
   difference, and top-1-percent candidate Jaccard between learned and analytic
   derivative arms.
5. Runtime: fit time is separated from frozen inference. Offline batch and
   causal one-frame streaming results report H2D, representation, Gamma moments,
   CFAR/NMS, D2H, p50/p95/p99/max latency, sustained frames/s, over-1-ms fraction,
   missed-deadline fraction, and queue/backlog growth.

Site-grouped bootstrap intervals are used for paired representation contrasts.
The latest-label sensitivity remains descriptive because its added candidates
were not independently surfaced.

## Advancement rules

The learned ICA stage stays in the compact main-paper architecture only if a
frozen learned arm:

- improves protected budget-curve area with a site-grouped 95% interval above
  zero;
- improves B58 macro recall by at least 0.02, wins at least two of four bursts,
  and does not reduce the partial quiet-burden curve;
- is not practically equivalent to the matched analytic derivative control;
- and either meets the 1-kHz streaming target or provides enough added recovery
  to justify a separately reported slower path.

If the two-frame ICA arm has absolute correlation at least 0.995 with fixed
difference and differs by at most one B58 match, it is treated as a learned
rediscovery of differencing. If no learned arm passes the gate, the paper uses
`Raw -> temporal difference -> Gamma-LS/CFAR`; ICA moves to the ablation.

The 1-kHz readiness claim requires sustained streaming with p99 below 1 ms and
no backlog under the frozen input dimensions. Batch throughput alone is not a
streaming result. Calcium-video results do not by themselves establish voltage
imaging performance.

## Execution gates and artifacts

1. Read-only preflight: hashes, shapes, label bounds, timeline alignment,
   non-colliding output root, RAM/disk, and GPU/driver state.
2. CPU analytic tests and CPU/CUDA parity on constant, impulse, border, signed,
   and legacy-exact fixtures.
3. Bounded GPU smoke on real frames with no label coordinates.
4. Label-sealed successive-halving screen and finalist freeze.
5. Full review-interval protected evaluation.
6. Full-recording frame-level count for the frozen winning arm.
7. Scientific-audit media and inventory validation.

The run is not scientifically complete until the required expert-only,
model-only, and matched-comparison audit set passes. The paper's oversized
Evaluation Contracts table will be removed; this versioned workflow and its
machine-readable artifacts remain the detailed contract.
