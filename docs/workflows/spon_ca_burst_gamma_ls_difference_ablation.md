# Spon Ca Burst Gamma-LS difference/ICA ablation

## Purpose

This workflow decides whether learned CS-Parzen ICA contributes useful
information beyond fixed temporal differencing when every representation uses
the same downstream Gamma-weighted local standardization and CFAR decision.
It also defines the first synchronized GPU-assisted timing contract for the
proposed voltage-imaging path. Dense representation and Gamma-LS operations
remain on CUDA; any host transfer and sparse candidate extraction must be timed
inside the reported end-to-end boundary rather than omitted.

The scientific design is frozen in
`docs/research/SPON_CA_BURST_GAMMA_LS_DIFFERENCE_ABLATION_V1_PLAN.md`; the
machine-readable example is
`examples/spon_ca_burst_gamma_ls_difference_ablation_v1.example.json`.

## Exact stage order

```text
acquired frames
  -> declared representation input
       adjacent level: causal spatial Gaussian + temporal EMA
       matched six-lag level: acquisition raw
  -> one fixed or fold-fitted representation arm
  -> signed radial Gamma-LS
  -> calibrated one-sided CFAR threshold
  -> deterministic spatial NMS
  -> frame-level candidate stream
```

`Raw` in result tables means the common causally preprocessed input, not the
unprocessed camera array. PCA is not a separately deployed stage: the ICA fits
use full-rank covariance eigendecomposition to center and whiten their temporal
observations, after which the independence rotation is applied. No principal
component is discarded. The explicit PCA-only arm is a control that isolates
the value of the subsequent independence rotation.

The six-lag delay-embedding arm genuinely consumes lags 0, 1, 2, 4, 8, and 16
at inference. Its frozen v5 fit and both matched-support controls operate on
the acquisition-raw domain used by that fit; the adjacent-frame level uses the
separately declared Gaussian-plus-EMA input. Gamma-LS itself is not multiscale:
many contexts are searched, then one is selected inside each outer training
fold. Version 1 has no multi-context fusion arm.

The archived two-frame and v5 fit packets both saw their complete review
intervals without sparse-positive coordinates. They are transductive parity
anchors, not outer-fold learned-model estimates. Protected two-frame ICA/PCA
arms must be refit inside each outer training fold. A protected six-lag claim
likewise requires fold-local v5 refitting; until that later stage exists, the
frozen v5 comparison is diagnostic only.

The archived two-frame preprocessor also reset its EMA state at UI frame 1800.
Scientific folds instead carry causal state from acquisition frame 1. The
archived model is therefore reproduced only under its reset-history parity
contract; it is never substituted for the required full-history fold refit.

## Gamma-LS versus historical implementations

`neurobench.algorithms.gamma_local_standardization` is the maintained
Gamma-weighted operator for this workflow. Modern contexts use a radial disk,
an explicit circular guard, and valid-reference renormalization at image
borders. The archived 23-by-23 shape-9/mode-35 reflect-padded operator is a
named legacy control. The maintained box-CFAR and square-annulus MSLN operators
are planned controls and are never labeled Gamma-LS. The current G1/G2 screen
runs only modern guarded radial contexts; neither control has yet been compared
empirically in this experiment.

Gamma-LS is the continuous signed standardized score. CFAR means the frozen
one-sided threshold plus spatial NMS. Thresholds are calibrated to observed
quiet candidate burden; their names do not claim a theoretical probability of
false alarm.

## Leakage and evidence boundary

The read-only preflight inspects annotation files only to validate their
eligibility, coordinates, counts, and hashes. Gamma-context screening uses the
predeclared temporal burst windows, so it is burst-window-aware rather than
fully label-free; it does not receive sparse-positive coordinates or
identities. Threshold calibration and candidate construction likewise do not
receive those sparse-positive fields. They enter only through the explicit
protected evaluation function after score/candidate artifacts are frozen.

The protected v1 endpoint contains 79 inclusive activity occurrences across
26 canonical identities and four bursts. The v7 endpoint contains 106
confirmed occurrences, but includes candidate-assisted additions and is a
descriptive sensitivity only. Unmatched candidates are unknown, not false
positives; therefore precision, specificity, and false-positive rate are not
identified.

## Preflight

Use the repository runtime and an external data authority:

```bash
NEUROBENCH_DATA_ROOT=/path/to/NeuRev-Workbench \
  .venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference \
  preflight \
  --config examples/spon_ca_burst_gamma_ls_difference_ablation_v1.example.json \
  --artifact-dir Outputs/GammaLSDifference/unique_preflight_id
```

The artifact directory and configured program output must not already exist.
Preflight writes source hashes, label-population checks, the frozen G1 context
table, resource probes, and a status file. `data_ready=true` does not authorize
a run when `gpu_run_ready=false`.

After a ready preflight, run the bounded device-residency smoke through the
same manifest entry point:

```bash
NEUROBENCH_DATA_ROOT=/path/to/NeuRev-Workbench \
  .venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference \
  smoke \
  --config examples/spon_ca_burst_gamma_ls_difference_ablation_v1.example.json \
  --preflight-dir Outputs/GammaLSDifference/ready_preflight_id \
  --output-dir Outputs/GammaLSDifference/unique_gpu_smoke_id
```

The command refuses a stale config, source, implementation fingerprint, or
blocked GPU preflight and leaves the requested output path absent on failure.

After the smoke passes, run the burst-window-supervised, sparse-coordinate-free
fold-local context screen:

```bash
NEUROBENCH_DATA_ROOT=/path/to/NeuRev-Workbench \
  .venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference \
  screen \
  --config examples/spon_ca_burst_gamma_ls_difference_ablation_v1.example.json \
  --preflight-dir Outputs/GammaLSDifference/ready_preflight_id \
  --output-dir Outputs/GammaLSDifference/unique_gpu_screen_id
```

This stage selects four fold-local Gamma contexts only. It does not evaluate
sparse-positive recall, fit the protected ICA models, choose a deployment
context, generate the full-recording proposal count, or establish streaming
speed.

## Protected adjacent-frame representation experiment

The protected executor accepts either the original screen directory or the
support-sufficiency directory containing `fold_contexts.json`. Contexts remain
fold-local and are never selected with protected coordinates or identities.
For every outer fold, adjacent training pairs are excluded when either source
frame touches the held-out burst plus its inclusive ten-frame guard.

The executor fits exactly 36 CS-Parzen packets: three frozen bandwidths by
three frozen paired-sample seeds by four outer folds. The archived whole-review
fit is not loaded. CS-Parzen is selected separately for each fold/context from
all nine rotation fits using training-window contrast. PCA is selected fairly
from the three unique seed-specific full-rank whitening fits after collapsing
its bandwidth-independent duplicate rows. A secondary PCA lane reuses the
packet selected by CS-Parzen so that its paired ICA-versus-PCA result isolates
only the learned rotation; this diagnostic cannot replace either independently
selected primary arm.

Quiet half A calibrates thresholds evaluated on quiet half B and the roles are
then reversed. Candidate streams are materialized and hashed before either
sparse-positive table is parsed. Spatial NMS distance 6 px is primary; 4 and
8 px are sealed descriptive sensitivity lanes and cannot select a model or
Gamma context. The protected v1 endpoint reports one-to-one recall at candidate
budgets 20, 40, 58, 80, and 100 and paired 95% intervals from a frozen
2,000-replicate bootstrap clustered by its 26 canonical identities. The 106-row
confirmed v7 endpoint remains descriptive only.

After all protected and shared implementation files are frozen, create a fresh
host-visible preflight and run:

```bash
NEUROBENCH_DATA_ROOT=/path/to/NeuRev-Workbench \
  .venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference \
  protected \
  --config examples/spon_ca_burst_gamma_ls_difference_ablation_v1.example.json \
  --preflight-dir Outputs/GammaLSDifference/unique_ready_preflight \
  --context-selection-dir Outputs/GammaLSDifference/unique_support_sufficiency_run \
  --output-dir Outputs/GammaLSDifference/unique_protected_run
```

Metric completion is intentionally recorded as
`complete_protected_metrics_scientific_audit_pending`. It does not authorize a
paper claim until the required scientific-audit media and consistency checks
are complete.

The host CUDA repair was verified on 2026-09-08 with NVIDIA driver 580.173.02
and an RTX 4070 SUPER. Ordinary sandboxed processes may hide `/dev/nvidia*`;
that is not evidence that the host driver regressed. Every real run must still
pass a fresh host-visible preflight in a new directory. Never edit a prior
preflight artifact to change its status.

## Sustained 1-kHz timing benchmark

The streaming executor measures three fixed, causal lanes independently:
`Raw`, signed adjacent difference, and energy-normalized adjacent difference.
Every lane receives one 340-by-573 uint16 frame on a declared one-millisecond
schedule for at least 60 seconds after 50 warm-up iterations. The source is a
64-frame pinned-memory ring populated from the real movie; camera acquisition
and disk I/O are explicitly outside the timing boundary. A separate logical
arrival queue holds at most eight frames and drops the newest arrival when it
is full, making overload visible rather than allowing unbounded backlog.

The timed boundary is:

```text
pinned uint16 host frame
  -> H2D
  -> Gaussian sigma-1 + stateful causal EMA alpha-0.4
  -> selected fixed representation
  -> cached guarded radial Gamma-LS
  -> strict threshold + 13x13 spatial local-maximum NMS
  -> bounded host decision packet + synchronization
```

The decision packet contains the retained-candidate count and the score,
`x=column`, and `y=row` of the highest-scoring retained candidate. No-candidate
frames use score `NaN` and coordinates `-1,-1`. This is materially more useful
for inverse-control latency than a scalar detection flag, while remaining
bounded. Dense frames and score maps remain device-resident. The local-maximum
operator matches the maintained square maximum-filter stage for continuous
scores; unlike offline `strict_separated_nms`, it does not run a subsequent
greedy cleanup for exactly tied plateaus. That difference is recorded in the
artifact and prevents an unqualified end-to-end detector-equivalence claim.

Primary p50/p95/p99/max latency uses wall time from H2D start through decision
packet transfer and terminal synchronization. CUDA-event stage timings,
response latency from the scheduled arrival, queue wait, missed deadlines,
drops, backlog, RAM, and VRAM are reported alongside it. A lane passes the
timing gate only when the duration and cadence contracts pass, p99 service
latency is strictly below one millisecond, there are zero missed processed
deadlines and zero drops, maximum queue depth never exceeds one, and backlog
after dequeue is always zero. A one-percent miss-rate diagnostic is reported
but cannot pass the readiness gate. This supports only the measured timing
boundary, not camera, control, or biological-detection readiness.

Scale-floor and quiet-tail threshold calibration are timed separately.
Fixed-difference lanes correctly report learned-model fit time as zero. The
quiet-tail threshold exists only to induce a representative bounded NMS load;
it is not the protected scientific operating point and makes no PFA claim.

Batch sizes 1, 8, 32, and 64 are measured after the paced experiment and saved
in `batch_throughput_frontier.tsv`. Those amortized rates are throughput only;
they cannot be substituted for the paced single-frame latency result.

After all implementation changes are frozen, create a new host-visible
preflight and run:

```bash
NEUROBENCH_DATA_ROOT=/path/to/NeuRev-Workbench \
  .venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference \
  streaming-benchmark \
  --config examples/spon_ca_burst_gamma_ls_difference_ablation_v1.example.json \
  --preflight-dir Outputs/GammaLSDifference/unique_ready_preflight \
  --screen-dir Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_difference_ablation_v1_gpu_screen_20260908_r1 \
  --output-dir Outputs/GammaLSDifference/unique_streaming_benchmark \
  --context-id gamma_h11_g5_n9_m1 \
  --arms raw difference_signed difference_energy_normalized \
  --duration-seconds 60
```

The timing artifact opens no labels and creates no candidate ranking or new
biological result. It therefore does not reproduce the paper experiment's
annotation media during the timed path and explicitly records the overall
scientific audit as incomplete. The protected detector winner still requires
the full scientific-audit evidence set before paper promotion.

## Run sequence

1. Run unit tests and require CPU fixture parity. CUDA parity remains a failure
   gate, not an optional skip, for a scientific run.
2. Run the sparse-coordinate-free GPU smoke on a short real-data interval.
   Confirm device residency from representation through Gamma-LS and measure
   explicit transfers.
3. For each held-out burst, run G1 on the other three bursts, retain two
   radius/guard pairs, then run G2 only for those pairs.
4. Freeze one fold-local Gamma context common to every representation arm and
   score only that fold's held-out burst. Do not pool folds to select a context
   before protected scoring.
5. Fit two-frame whitening and CS-Parzen independently within each outer
   training fold across the frozen three-bandwidth, three-seed grid. Select
   PCA from its three unique seed-specific whitening fits and CS-Parzen from
   its nine seed-by-bandwidth fits; also retain the matched-packet rotation-only
   diagnostic. The archived whole-review fit is parity-only.
6. Cross-fit score thresholds between the two contiguous quiet halves: fit on
   one half, measure burden on the other, then reverse. Materialize and hash
   candidate streams for primary NMS 6 px and descriptive NMS 4/8 px before
   joining sparse-positive coordinates or identities.
7. Evaluate protected leave-one-burst-out recovery, then run the v7 sensitivity.
8. Run the frozen winner over the full recording without annotated burst
   windows and report the automated frame-level proposal-row count and
   proposals per frame at every frozen threshold. Do not convert this to a
   unique event/site count until a temporal-linking contract is frozen, and do
   not call proposals biological detections without a separate exhaustive truth
   audit.
9. Run the streaming benchmark for at least 60 seconds after warm-up, with one
   340-by-573 frame submitted every millisecond. Report latency percentiles,
   missed deadlines, backlog, transfers, VRAM, and fit time separately.
10. Complete the scientific-audit inventory and media checks before manuscript
   promotion.

## Manuscript decision

If learned ICA does not clear the frozen improvement and runtime gates, the
compact paper architecture becomes `causal temporal difference -> Gamma-LS ->
CFAR/NMS`, and ICA appears only as the experiment that motivated and tested the
fixed difference. If it clears them, the architecture becomes `multi-lag
CS-Parzen representation -> selected-context Gamma-LS -> CFAR/NMS`; full-rank
PCA whitening is explained as part of the fit, not drawn as a separate online
block.

The large manuscript Evaluation Contracts table has been removed. Its essential
population and claim boundaries remain in prose; full operational detail lives
in versioned workflow artifacts.
