# Information Source Separation Conclusive Batch v1 Plan

## Objective

Run one collision-safe, resumable, internally gated program that explores the
remaining source-separation and identifiability solution space, evaluates
scientifically valid survivors on generated and real-background semi-synthetic
truth, runs bounded CaImAn CNMF references, evaluates frozen survivors on Spon
Ca Burst, produces diagnostic videos, and writes one final decision report.

The program is one batch operationally, but it is not one undifferentiated
Cartesian sweep. Later stages consume only artifacts that pass earlier frozen
gates. A failed gate is a terminal, reportable result.

Proposed root:

```text
Outputs/InformationSourceSeparation/conclusive_batch_v1
```

No existing output root is modified.

## Hardware envelope

Observed 2026-08-01:

- CPU: Intel Core i9-14900K, 24 physical cores / 32 logical CPUs;
- RAM: 78 GiB total, approximately 60 GiB available;
- GPU: NVIDIA RTX 4070 SUPER, 12,282 MiB total, approximately 9.6 GiB free;
- disk: approximately 1.9 TiB free;
- CaImAn: 1.13.1 in an isolated Python 3.11 environment (7.7 GiB).

Revised conservative recovery limits after the first run caused host
instability. These reductions do not change scientific method parameters:

- one GPU worker, 4 GiB allocation cap, 6 GiB minimum free-memory launch gate;
- two general CPU workers, at most one CaImAn process;
- BLAS/OpenMP threads set to one inside workers;
- 24 GiB aggregate RSS soft cap and 32 GiB hard stop;
- 40 GiB output cap and 200 GiB minimum free-disk gate;
- 15-second heartbeat, per-fit atomic JSON, resumable fit directories;
- GPU temperature warning at 74 C and stop/retry at 80 C;
- no simultaneous full CaImAn fit and high-utilization CUDA fit.

The preflight includes timing probes and may reduce worker counts. It may not
increase these caps automatically.

## Method panel

### Anchors and established references

- raw-direct downstream detector anchor;
- amplitude PCA, ranks 4/8/12;
- multi-lag SOBI, bounded rank/lag/shrinkage panel;
- existing spatial FastICA/Wiener adapter.

### Information-theoretic candidates

- CUDA normalized-HSIC pairwise rotation;
- kNN-MI pairwise rotation;
- grouped-energy HSIC ISA;
- spatial noisy-Parzen Infomax, only after numerical smoke gates;
- multi-start and perturbation-consensus variants of HSIC/MI.

### External reference

- CaImAn CNMF/CNMF-E adapter using exact CaImAn 1.13.1;
- two-photon CNMF and one-photon/background-aware CNMF-E are separate native-
  best tracks, not interchangeable configurations;
- ordinary NMF is not substituted.

Every method records `controlled_input` or `native_best`. Cross-track values
are not presented as algorithm-only ablations.

## Stage 0 — implementation and numerical qualification

Implement missing adapters, structured identifiability diagnostics, nested
selection, output contracts, and video rendering. Add deterministic fixtures
and focused tests.

Required checks:

- exact closure and finite outputs;
- deterministic replay;
- scale/sign/permutation matching;
- collapsed/duplicate/explosive component detection;
- CaImAn import, tiny CNMF fit, save/load replay, and process cleanup;
- CUDA/CPU HSIC parity;
- diagnostic MP4 frame/hash verification.

No long run starts until every preflight gate passes.

## Stage 1 — identifiability continuum and bounded method screen

Replace binary ambiguity with a continuous, truth-known design spanning:

- footprint overlap and mixing condition number;
- temporal correlation and exact/near temporal rank deficiency;
- source amplitude ratio;
- SNR 4/8/16;
- background persistence/alias strength;
- illumination drift, motion edge, clipping, and heteroscedastic noise;
- pure-noise and no-source controls.

Use a space-filling bounded design rather than a full factorial grid.

Provisional allocation:

- 72 development fixtures for the broad configuration screen;
- at most 36 method configurations;
- ceiling: 2,592 base fits.

Selection first requires numerical integrity. Recovery metrics from
unidentifiable/no-source controls are never included in source-recovery
ranking.

## Stage 2 — confidence and selective-risk program

Compare frozen confidence families:

- transparent evidence thresholds;
- singular-gap, mixing-condition, and temporal-rank diagnostics;
- multi-start component consensus;
- bootstrap/subsample stability;
- perturbation stability at multiple realistic scales;
- regularized logistic and monotone tree models;
- split-conformal abstention / risk-control thresholds.

Data partitions are disjoint by case family and seed:

- separator development;
- confidence training;
- confidence calibration;
- untouched generated evaluation.

Primary metrics:

- false-resolution rate and Wilson interval;
- identifiable coverage and false-abstention rate;
- selective recovery and crosstalk;
- risk-coverage curve and area;
- calibration error and Brier score;
- worst case/seed/SNR;
- configuration and multi-start stability.

Advancement requires zero observed false resolutions on the frozen generated
evaluation, at least 80% identifiable coverage, at least 95% convergence, and
no family with catastrophic recovery. Confidence thresholds cannot be retuned
after evaluation.

## Stage 3 — frozen generated confirmation

At most two configurations per surviving family enter approximately 312
additional generated fixtures. The adaptive ceiling is 12 configurations and
3,744 fits. It includes unused seeds, SNRs, near-boundary identifiability, and
critique-sensitive artifact families.

Primary selection is lexicographic:

1. numerical and identifiability gate;
2. amplitude/timing/footprint fidelity;
3. source recovery and crosstalk;
4. simpler model within the 0.01 equivalence margin;
5. runtime/output burden.

## Stage 4 — real-background semi-synthetic Spon

Inject known sources into quiet Spon crops using disjoint development and held
crop/seed partitions.

Bounded design:

- three spatial crops;
- overlap, synchronous, persistence-confounded, and amplitude-imbalanced
  morphologies;
- amplitudes 0.5/1/2;
- disjoint seeds;
- approximately 135 fixtures;
- at most six survivor configurations plus bounded CaImAn references.

Report source recovery, footprint IoU/centroid error, peak and area retention,
onset and peak-frame error, native-background leakage, residual dependence,
abstention, runtime, and memory. No method advances on detector score alone.

## Stage 5 — CaImAn reference program

Run in the isolated environment with at most 12 processes and explicit cluster
shutdown. First tune bounded native parameters on generated/semi-synthetic
development data, then freeze one CNMF and one CNMF-E configuration.

Artifacts include spatial components, calcium traces, deconvolved activity,
background components, residual, accepted/rejected component diagnostics,
runtime/RSS, exact package list, and videos. CaImAn does not receive final Spon
labels during fitting or component selection.

## Stage 6 — frozen Spon Ca Burst evaluation

Only G0-G2 survivors enter. Configuration, rank, component count, confidence,
and preprocessing are frozen before Spon evaluation.

Two tracks are reported:

- scientific reconstruction fidelity/diagnostics;
- downstream sparse-positive neuron-detection utility.

Use leave-one-burst-out threshold/ranking calibration, grouped ROI identity,
the exact Raw Direct anchor, and fixed candidate budgets 10/20/40/58/80/100.
Report known-positive recall, per-burst/per-neuron consistency, candidates,
timing, localization, and decomposed failure classes. Unmatched candidates
remain unknown.

Precision is reported only for an exhaustively adjudicated bounded field. If
that annotation is unavailable, the automatic final conclusion is explicitly
limited to known-positive recall and candidate burden, and the batch produces
the review queue/videos required for human adjudication.

## Stage 7 — diagnostic video package

Videos are required artifacts, not optional illustrations.

Generated/semi-synthetic videos show:

- observation;
- ground-truth neural contribution;
- recovered neural reconstruction;
- background/artifact reconstruction;
- residual;
- component traces and confidence/decision;
- match, leakage, amplitude, and timing summaries.

Spon videos show, for every burst and finalist:

- raw and pseudo-color intensity;
- positive baseline change and frame derivative;
- method reconstruction and residual;
- component traces;
- sparse-positive ROI IDs;
- candidates, known matches, and unmatched-as-unknown legend;
- original detection window plus sufficient pre-window context.

Also create side-by-side finalist montages, failure videos for ROI 007/008/014/
017/019/020, and an explicit ROI 010/015 overlap/adjudication clip. Every MP4
is probed and hashed.

## Stage 8 — final conclusive report

The batch finishes in one of three terminal states:

1. `scientific_candidate_validated`: at least one method passes generated,
   semi-synthetic, and frozen Spon gates;
2. `no_candidate_survived`: every method has a documented failure mechanism;
3. `precision_pending_adjudication`: model comparison is complete for
   known-positive recall, but precision awaits exhaustive human labels.

The report includes all attempted configurations, selection decisions,
confidence intervals, case-family wins/ties/losses, resource usage, video
index, limitations, and a single recommendation. Completion is never called a
scientific win by itself.

## Estimated workload

Adaptive ceiling before Spon evaluation:

- broad screen: 2,592 base fits;
- stability/confidence perturbations: approximately 2,000-3,000 fits;
- generated confirmation: at most 3,744 fits;
- semi-synthetic and CaImAn: approximately 800-1,200 fits;
- total expected numerical fits: approximately 9,000-10,500.

Expected wall time is approximately 12-30 hours after implementation and
microbenchmarking, depending mainly on CaImAn and HSIC confirmation survivors.
The run is resumable and safe across interruption.

Expected generated artifacts are 10-40 GiB; the hard cap is 80 GiB.

## Launch contract

A single top-level manifest and command will create a read-only preflight,
freeze exact counts and hashes, then execute the program. It will not overwrite
completed outputs or bypass scientific gates. Human adjudication, if required
for precision, is the only planned external pause.
