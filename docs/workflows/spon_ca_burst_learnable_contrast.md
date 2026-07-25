# Spon Ca Burst: Learnable Guarded-Contrast ROI Discovery Experiment

## Objective

Use the limited point-and-burst labels for `3 hindbrain to tail 488 20ms` to learn a translation-equivariant local contrast operator and an operating threshold, then apply the frozen detector to rank additional candidate ROIs for workbench review.

The experiment is deliberately **weakly supervised**. The spreadsheet supplies positive ROI centers within burst windows, but not complete ROI masks, exact onset frames, or exhaustive negatives. Unlabeled event pixels must therefore remain *unknown*, not negative.

## Data and label contract

- Source cache: the existing memory-mapped NumPy video used by the Spon Ca Burst workflow.
- Video: 2,359 frames, 340 x 573 pixels, 50 Hz.
- Quiet control: UI frames 1800--1899.
- Scored interval: UI frames 1900--2359.
- Labeled burst windows:

| Burst | UI frames, inclusive | Duration | Positive centers |
|---:|---:|---:|---:|
| 1 | 2003--2026 | 24 frames / 0.48 s | 15 |
| 2 | 2040--2063 | 24 frames / 0.48 s | 20 |
| 3 | 2122--2149 | 28 frames / 0.56 s | 21 |
| 4 | 2254--2300 | 47 frames / 0.94 s | 23 |

The sheet contains 79 point-window labels at 27 unique coordinates: 14 coordinates appear in all four bursts, 3 in three bursts, 4 in two bursts, and 6 in one burst.

**Preflight assumption to verify:** `x` is image column and `y` is image row in native pixel coordinates. The preflight must write a projection overlay containing every point before any training starts.

UI frame intervals are one-based and inclusive. Stored NumPy intervals are zero-based and half-open; for example, `2003--2026` becomes `[2002, 2026)`.

## Primary hypotheses

1. A fully learnable, guard-constrained pair of test and reference kernels improves held-out-burst ROI-center recall at a fixed quiet-control false-candidate rate relative to the current fixed box-CFAR detector.
2. A simple amplitude skip lane improves recovery of spatially broad excitation that a purely local contrast statistic can suppress.
3. Candidate peaks that are stable across cross-validation models and recur across bursts are enriched for true, previously unlabeled ROIs.

## Input representation

Estimate a frozen quiet baseline `B` and normalization scale using only UI frames 1800--1899. For each scored frame,

\[
r_t = \left[\operatorname{norm}(I_t)-\operatorname{norm}(B)\right]_+.
\]

No event frame may affect baseline fitting, quiet-null calibration, or coordinate priors.

## Learnable operator

Use a fixed 21 x 21 maximum support, chosen to cover the current approximately 10-pixel perisomatic radius while keeping the parameter count small. Every coefficient is learnable.

\[
K_i(u)=\frac{\operatorname{softplus}(A_i(u))}
{\sum_v \operatorname{softplus}(A_i(v))},
\qquad i\in\{T,R\}.
\]

Thus both kernels remain nonnegative and unit mass, but may learn arbitrary asymmetric or disconnected geometry.

For each frame,

\[
t_t=K_T*r_t,\qquad \mu_t=K_R*r_t,
\]

\[
v_t=K_R*(r_t^2)-\mu_t^2,
\qquad b_R=1-\lVert K_R\rVert_2^2,
\]

\[
q_\theta=\lVert K_T-K_R\rVert_2^2,
\]

\[
z_t=\frac{t_t-\mu_t}
{\sqrt{q_\theta\,\operatorname{spos}(v_t)/(b_R+\epsilon_b)+\epsilon_x}},
\qquad c_t=[z_t]_+^2.
\]

The kernel-shape correction prevents the score scale from changing solely because a learned kernel becomes sharper or broader.

Use soft penalties for:

- guard-distance violation between `K_T` and `K_R`;
- minimum reference effective sample size;
- centered and aligned kernel centroids;
- reference extent larger than test extent;
- weak spatial total variation.

Initialize from the current compact-test / guarded-reference geometry, but release every coefficient for optimization.

## Multiple-instance supervision

A label states that activity occurs near a coordinate sometime within a burst window. It does not identify one frame or one exact pixel. Define a positive bag score using normalized log-mean-exp pooling over time and a 4-pixel spatial tolerance disk:

\[
s^+_{b,i}=\operatorname{LME}_{t\in W_b,\;p\in D_4(p_i)} F_t(p).
\]

Use two model arms:

1. **Learned contrast:** `F_t = c_t`.
2. **Learned contrast + amplitude:**
   \[
   F_t=c_t+\lambda\log(1+t_t),
   \]
   where `lambda` is selected only on inner validation folds.

Pair each positive event bag with quiet-control bags of the same duration, preferably at the same coordinate plus additional random quiet locations. Optimize a positive-versus-quiet ranking loss:

\[
\mathcal L_{rank}=\operatorname{softplus}(m-s^+ + s^{quiet}).
\]

Do **not** classify unlabeled locations in event windows as negatives. This preserves the ability to discover new ROIs.

## Cross-validation and discovery simulation

### A. Temporal generalization

Use four outer leave-one-burst-out folds. The held-out burst is used only for final fold evaluation. Within the three remaining bursts, rotate one burst as inner validation and use the other two for gradient updates. Select regularization, `lambda`, and stopping epoch from mean inner-fold performance, then refit on all three outer-training bursts.

### B. Masked-ROI discovery

Within each outer fold, repeat a second experiment over at least 10 deterministic seeds:

- hide 25% of ROI identities from all training labels, stratified by recurrence count;
- leave their event pixels unlabeled rather than negative;
- train on the remaining identities;
- test whether hidden identities appear among the top-ranked candidate peaks.

This is the closest available estimate of the intended unknown-ROI discovery task.

### Quiet-control cross-fitting

Divide the 100 quiet frames into contiguous blocks. A quiet block used to set a threshold may not also supply the reported null false-candidate rate. The final production model may use all quiet frames after model selection is complete.

## Threshold and candidate formation

Thresholds are calibrated after kernel fitting, not learned as an unconstrained classifier bias.

For each validation model:

1. Build quiet pseudo-burst maps with durations matching the four labeled windows.
2. Apply non-maximum suppression with a 6-pixel minimum distance.
3. Sweep the score threshold.
4. Report FROC versus quiet peaks per burst map.
5. Select the primary operating point at no more than one quiet peak per burst map; retain 0.25, 0.5, 2, and 5 as secondary operating points.

For event data, unmatched peaks are called **unmatched candidates**, not false positives, because the labels are not exhaustive.

A burst-level peak matches a manual center through one-to-one assignment within 6 pixels. Report sensitivity analyses at 4 and 8 pixels.

Cluster peaks across bursts within 6 pixels to form candidate identities. Each candidate record should include:

- centroid and provisional footprint;
- scores and peak frames by burst;
- number of bursts observed;
- nearest known-label distance;
- outer-fold and random-seed agreement;
- quiet-control score and threshold margin;
- direct-amplitude and learned-contrast components;
- local temporal coherence and artifact cues.

The learned kernel is a detector, not automatically a cell mask. Initial footprints should come from local temporal-correlation region growing or a small provisional circle and remain reviewable/editable in the workbench.

## Baselines and ablations

Evaluate all methods under identical splits, matching, NMS, and threshold calibration:

1. direct positive residual only;
2. current fixed box-CFAR;
3. fixed Gamma/guarded initialization without learning;
4. fully learnable guarded contrast;
5. fully learnable guarded contrast plus amplitude lane;
6. optional unconstrained-kernel control to expose degeneracy;
7. optional quiet-dark-zone gating versus full-field search.

## Quantitative endpoints

Primary:

- held-out-burst center recall at one quiet peak per burst map;
- FROC across quiet false-candidate operating points;
- masked-ROI Recall@10, Recall@20, and Recall@50.

Secondary:

- localization error in pixels;
- recall by recurrence class and by burst;
- area under the truncated FROC curve;
- threshold variation across folds;
- kernel guard violation, effective sample size, centroid offset, and noise gain;
- kernel stability across folds and seeds;
- count of novel candidates observed in at least two bursts;
- workbench acceptance rate among the top 10, 20, and 50 novel candidates.

Use paired bootstrap intervals over ROI identities. With only four bursts, emphasize effect sizes and fold consistency rather than asymptotic significance tests.

## Final discovery run

After all design choices are frozen:

1. fit a 5-seed ensemble using all 79 labels;
2. calibrate the final threshold from the full quiet interval using the preselected operating rule;
3. score the four labeled burst windows and the full 1900--2359 interval in bounded chunks;
4. retain 2-D maps, sparse peaks, candidate traces, and review clips rather than a dense full-video score stack;
5. export known matches separately from novel candidates;
6. rank novel candidates by ensemble agreement, recurrence, threshold margin, local coherence, and low artifact risk;
7. attach them to `review_data.json` as discovery suggestions and review them through the existing missed-neuron queue.

Novel candidates become scientific positives only after manual review.

## Recommended repository layout

```text
neurobench/algorithms/learnable_contrast.py
neurobench/experiments/spon_learnable_contrast/
  config.py
  labels.py
  dataset.py
  model.py
  trainer.py
  evaluate.py
  discover.py
  report.py
neurobench/cli/experiment.py
examples/spon_ca_burst_learnable_contrast.example.json
docs/workflows/spon_ca_burst_learnable_contrast.md
tests/test_learnable_contrast.py
tests/test_spon_learnable_contrast_experiment.py
```

Suggested output root:

```text
Outputs/LearnableContrast/spon_ca_burst_v1/
```

Required durable artifacts:

- `resolved_config.json`;
- `labels_normalized.tsv` and `splits.json`;
- fold/seed kernel checkpoints and kernel images;
- `metrics.json`, `froc.tsv`, and a concise report;
- `known_recovery.tsv`;
- `novel_candidate_rois.tsv`;
- workbench-compatible discovery JSON;
- resource/preflight and run-state files.

## Initial go/no-go rule

Advance the operator if, at the same quiet-control peak rate, it improves mean leave-one-burst-out recall over the best fixed baseline, the improvement appears in at least three of four outer folds, and masked-ROI Recall@20 also improves. Advance the discovery list to broader review only if the top-ranked novel candidates remain stable across seeds and the first manual-review batch shows useful enrichment over the existing discovery ranking.

## Repository implementation and first guarded run

The implemented entry point is `experiment learnable-contrast`; it lazy-loads
the CUDA stack only after applying bounded CPU thread settings.

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment \
  learnable-contrast preflight \
  --config examples/spon_ca_burst_learnable_contrast.example.json \
  --artifact-dir Outputs/LearnableContrast/spon_ca_burst_v1_cuda_guarded_preflight

.venv-neurobench/bin/python -m neurobench.cli.main experiment \
  learnable-contrast run \
  --config examples/spon_ca_burst_learnable_contrast.example.json
```

The first completed run is under
`Outputs/LearnableContrast/spon_ca_burst_v1_cuda_guarded`. Its authoritative
quick read is `experiment_summary.json`; detailed evidence is in `metrics.json`
and `report.md`. The design gate returned `do_not_advance`: learned contrast
tied the fixed guarded initialization on mean leave-one-burst-out recall and was
well below direct positive residual, despite improving masked-ROI Recall@20 over
the direct lane. The one unmatched final-ensemble peak remains an unreviewed
candidate and is exported to `workbench_discovery_review_data.json` only for
inspection.

## Spatiotemporal factorial diagnostic v2

A separate v2 diagnostic preserves the completed v1 artifacts while isolating
three factors:

1. input: raw quiet-median residual versus causal Kalman spatiotemporal residual;
2. objective: legacy raw score versus stabilized log score;
3. initialization: fixed guarded kernels versus genuinely jittered guarded kernels.

The spatiotemporal lane uses the existing causal positive-innovation stack,
spatial Gaussian smoothing (`sigma=1 px`), temporal Gaussian smoothing
(`sigma=1.25 frames`), and quiet-only per-pixel median/MAD whitening. The
unsmoothed amplitude lane is whitened separately. Jittered cells use seeds 7,
13, and 19; fixed cells use seed 101. The matrix contains 8 factor combinations,
32 fold-level conditions, and 64 learned fits.

Run command:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment \
  learnable-contrast diagnostic \
  --config examples/spon_ca_burst_learnable_contrast_spatiotemporal_diagnostic.example.json
```

Completed output:

```text
Outputs/LearnableContrast/spon_ca_burst_spatiotemporal_factorial_v2
```

Key results:

| Input | Objective | Initialization | Mean held-out recall | Matched direct baseline |
| --- | --- | --- | ---: | ---: |
| Raw | Legacy | Fixed | 0.1328 | 0.6056 |
| Raw | Legacy | Jittered | 0.1328 | 0.6056 |
| Raw | Stabilized | Fixed | 0.1972 | 0.6056 |
| Raw | Stabilized | Jittered | **0.2051** | 0.6056 |
| Kalman spatiotemporal | Legacy | Fixed or jittered | 0.0000 | 0.2937 |
| Kalman spatiotemporal | Stabilized | Fixed or jittered | 0.0000 | 0.2937 |

Interpretation:

- stabilized score scaling produced the substantive optimization improvement
  (`+0.0644` absolute recall with fixed initialization);
- true initialization jitter added only `+0.0079` on the stabilized raw lane;
- the requested Kalman/Gaussian/whitened preprocessing reduced direct recall by
  `0.3120` and did not produce label-aligned learned peaks;
- none of the learned cells beat its matched direct baseline in any held-out
  burst, so the gate returned `do_not_advance`;
- masked-ROI and final-ensemble stages were deliberately not run after the gate
  failed. This is a planned stop, not an incomplete run.

Use `experiment_summary.json` for the compact result,
`combination_summary.tsv` for the eight-cell comparison, and `metrics.json` for
all 64 fit histories. The main implication is that objective scaling was a real
problem, while initialization was secondary and this particular
spatiotemporal preprocessing formulation was harmful. Future work should keep
the direct amplitude lane primary and test temporal/coherence features without
forcing them through the local spatial-contrast statistic.

## Learnable raw-direct tuning v3

The v3 experiment tests whether the strong raw-direct detector can be improved
without asking a new model to rediscover it. It preserves raw quiet-median
positive residual input, temporal log-mean-exp pooling, fixed NMS, and
quiet-control threshold calibration. Three cumulative variants soften only a
small neighborhood around that baseline:

1. learnable amplitude gain/nonlinearity and temporal-pooling temperature;
2. the first variant plus a bounded 5-by-5 spatial residual filter;
3. the second variant plus lightly learnable guarded kernels and low-weight
   contrast and temporal-coherence terms.

The temporal/amplitude and spatial variants reproduce the direct score exactly
at initialization. Guarded auxiliary weights start at `0.001`. Parameter
transforms bound the possible drift, an explicit trust-region penalty pulls
toward initialization, and the spatial/kernel learning rate is 30% of the
nominal rate.

The screen crosses the three variants with learning rates `3e-5`, `1e-4`, and
`3e-4`, then evaluates all four leave-one-burst-out folds: 9 combinations and
36 learned fits. A 12-fit, three-seed confirmation stage is conditional on a
screen winner exceeding direct recall and winning at least three of four
bursts. Masked and final stages remain behind later gates.

Run command:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment \
  learnable-contrast direct-tuning \
  --config examples/spon_ca_burst_learnable_direct_tuning.example.json
```

Completed output:

```text
Outputs/LearnableContrast/spon_ca_burst_learnable_direct_tuning_v3
```

All 36 fits completed in about 260 seconds. All nine combinations produced the
same mean held-out recall as frozen raw-direct:

| Variant | Learning rates | Mean held-out recall | Fold wins vs direct |
| --- | --- | ---: | ---: |
| Temporal + amplitude | `3e-5`, `1e-4`, `3e-4` | 0.6056 | 0/4 |
| + bounded spatial residual | `3e-5`, `1e-4`, `3e-4` | 0.6056 | 0/4 |
| + guarded contrast/coherence | `3e-5`, `1e-4`, `3e-4` | 0.6056 | 0/4 |

The direct and learned fold recalls were `0.4667`, `0.5500`, `0.6667`, and
`0.7391`, corresponding to 49 matches among 79 burst-label rows. This metric is
the unweighted mean of the four burst recalls; the pooled label-row recall is
`49/79 = 0.6203`.

Training was numerically active but conservative:

- stabilized rank loss decreased in every fit, by `0.0014` to `0.0151`;
- learned gain and amplitude exponent remained between `1.0015` and `1.0150`;
- temporal temperature moved from `0.25` to between `0.2439` and `0.2493`;
- spatial-filter L1 drift stayed at or below `0.0031`;
- guarded contrast weight stayed between `0.00101` and `0.00106`;
- event-peak counts changed by between `-1` and `+4`, but labeled recall did
  not change.

Peak GPU allocation was about 593 MiB, peak reserved memory about 696 MiB, and
peak process RSS about 1.64 GiB. During guarded fits the RTX 4070 SUPER reached
100% compute utilization near 198 W while remaining around 67 C. There were no
OOMs or resource failures.

The result is a valid negative result, not evidence that a direct-initialized
pipeline is intrinsically unhelpful. It shows that conservative local tuning
preserves the good baseline but does not change the labeled detections under
the current objective. The main remaining mismatch is methodological:
training contrasts event labels with quiet samples at those same coordinates,
