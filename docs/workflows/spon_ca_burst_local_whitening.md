# Spon Ca Burst local covariance whitening

## Question and scope

This stage-gated CPU-first program asks whether the three frozen signed causal
joint-MSLN channels retain reproducible off-diagonal covariance, and whether a
quiet-fitted local ZCA transform improves held-out covariance calibration over
identity, diagonal, and global controls without damaging synthetic event
morphology or temporal behavior.

The primary feature order is frozen as:

1. `joint_s5_g1_t15_g1`;
2. `joint_s15_g3_t23_g1`;
3. `joint_s15_g3_t31_g1`.

The maintained implementation uses signed scientific MSLN arrays. Display
arrays, squared evidence, spatial labels, and event frames are excluded from
covariance fitting and label-free lane selection. A whitened coordinate is a
statistical coordinate, not a biological source or cleaned movie.

## Implementation plan

The implementation is deliberately split into independently testable owners:

- `neurobench.metrics.whiteness` defines covariance, effective-rank,
  autocorrelation, and tile-seam diagnostics with analytical tests.
- `neurobench.algorithms.local_covariance_whitening` owns feature-bank
  validation, deterministic quiet-sample gathering, diagonal/fixed-ridge/OAS
  fitting, explicit ZCA construction, tile enumeration, overlap blending,
  fallback provenance, Mahalanobis energy, and empirical quiet-tail
  calibration.
- `neurobench.experiments.msln_msica.local_whitening_program` owns exact
  manifest validation, contiguous quiet partitions, source fingerprints,
  collision-safe stage transitions, generated fixtures, real-quiet audit
  authorization, full-Spon authorization, label-free freeze ordering, and
  protected-evaluation boundaries.
- `neurobench.reports.local_whitening` writes compact JSON/CSV/Markdown indices
  and fixed-scale preview figures. Full real-data execution remains incomplete
  until the standard three-section scientific audit validates successfully.

The implementation does not add adaptive covariance, patch ZCA, temporal AR
prewhitening, a new ICA objective, or an automatic maintained-detector change.

## Numerical contract

For feature vectors `phi[r,t]`, a covariance fit uses only its declared
contiguous quiet-fit block. OAS is primary. The fixed-ridge control shrinks the
sample covariance toward `trace(Sigma) / D * I`; the diagonal control preserves
only per-channel variances. Eigenvectors come from `numpy.linalg.eigh`, and ZCA
is constructed explicitly as `U diag(lambda**-0.5) U.T` after a recorded
eigenvalue floor.

Local transforms are fit before application. Overlapping transformed feature
vectors are blended with positive deterministic weights and normalized by the
accumulated weight. Energy is computed only after feature blending. An
unresolved local full fit falls back to the matching global full fit, then a
resolved local diagonal fit, then an explicitly recorded identity transform.

Empirical quiet surprise is calibrated on a contiguous quiet block disjoint
from covariance fitting and held-out assessment. Unmatched model candidates
remain unknown.

## Stages and authorization

```bash
.venv-neurobench/bin/python -m neurobench.experiments.msln_msica.local_whitening_program preflight \
  --config examples/spon_ca_burst_local_whitening_v1.example.json

.venv-neurobench/bin/python -m neurobench.experiments.msln_msica.local_whitening_program synthetic \
  --config examples/spon_ca_burst_local_whitening_v1.example.json

.venv-neurobench/bin/python -m neurobench.experiments.msln_msica.local_whitening_program covariance-audit \
  --config examples/spon_ca_burst_local_whitening_v1.example.json
```

`covariance-audit` defaults to generated data. Reading the declared Spon quiet
source requires `--use-real-quiet-source`. A full recording run additionally
requires an exact matching preflight and `--authorize-full-spon`.

### CUDA application backend

CUDA is an application backend, not a different covariance estimator. The
float64 CPU fit remains authoritative; frozen means and whitening matrices are
applied with CuPy float32 kernels in bounded frame chunks. Overlap blending and
Mahalanobis energy stay on the GPU until each chunk is copied back. Empirical
tail calibration and report generation remain on CPU. This separation ensures
that CPU and CUDA compare the same scientific model.

CUDA execution is refused unless a matching parity preflight passes and the
requested cap is no larger than the manifest's frozen 4 GiB VRAM limit:

```bash
.venv-neurobench/bin/python -m neurobench.experiments.msln_msica.local_whitening_program gpu-preflight \
  --config examples/spon_ca_burst_local_whitening_v1.example.json \
  --max-vram-gb 4

.venv-neurobench/bin/python -m neurobench.experiments.msln_msica.local_whitening_program covariance-audit \
  --config examples/spon_ca_burst_local_whitening_v1.example.json \
  --use-real-quiet-source --compute-backend cuda --max-vram-gb 4
```

The parity limits are `2e-5` maximum absolute error for whitened channels,
`1e-4` for energy, and `2e-3` for empirical surprise. Surprise has a separate
bound because the empirical rank-tail map is discontinuous at near-ties; it
does not relax the scientific-channel or energy comparison. The backend
records estimated and observed peak VRAM, device/runtime metadata, and refuses
an undersized cap before allocating the scientific outputs. CPU remains a
supported fallback.

## Gates and interpretation

G0 tests whether residual off-diagonal covariance is reproducible and whether
full or local modeling improves held-out whiteness. G1 tests synthetic event
preservation and conditioning. G2 tests real quiet-block generalization. G3
freezes one primary and at most two diagnostic lanes without spatial labels.
Only then may G4 inspect sparse known positives. ICA comparison is a later G5
family and requires a passed representation gate plus explicit full-data user
selection.

The bootstrap-complete quiet-only audit completed on 2026-08-15 in the
collision-safe root
`Outputs/LocalWhitening/spon_ca_burst_local_whitening_v1_bootstrap_complete_20260815`.
It passed G0 but failed G2, producing the terminal outcome
`global_full_only`: global full ZCA improved held-out covariance calibration in
both quiet blocks and both 95% frame-bootstrap improvement intervals remained
strictly positive. Local full ZCA was worse than global full in both blocks and
75.48% of its image support required fallback. One of two global split-half
stability refits was unresolved under the conditioning cap, so global full is
the best supported tested control, not a promoted production method.
Consequently the program stopped before spatial-label loading, full-interval
representation generation, protected evaluation, or ICA. This is an
informative negative result for localization, not a failure of the numerical
implementation.

Every report must preserve the current repository truth: scalar joint MSLN is
not multivariate whitening; the v2 `joint_s15_g3_t31_g1` 58/79 result and the
49/79 Raw Direct anchor used non-identical protocols; prior broad ICA directions
were unstable; the present recording is development data; sparse annotations
do not define negatives; and computational completion is not scientific
success.

## Scientific-audit contract

A full real-data run is not complete until
`docs/workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md` passes. Its model sequence
is Raw -> signed MSLN channels -> ZCA channels/declared summary -> Mahalanobis
energy -> empirical quiet surprise -> temporally pooled detection. Expert and
model markers remain section-pure, and display-normalized arrays never enter
detection.

## Comparison contract

The whitening comparison is factorial and fit-matched:

- identity (no additional whitening);
- global diagonal standardization;
- local diagonal standardization;
- global full ZCA;
- local full ZCA.

All lanes use the same signed three-context MSLN feature bank, contiguous quiet
fit/calibration/held-out blocks, empirical-tail definition, and application
precision. Backend parity is evaluated separately from scientific utility.
The label-blind quiet audit compares held-out identity error, maximum absolute
correlation, off-diagonal energy, tail stability, tile seams, unresolved
support, fit stability, runtime, and memory. Sparse-positive scoring and ICA
are permitted only if local full ZCA passes G2; a failed G2 is not bypassed to
obtain more favorable downstream metrics.

### Completed CUDA comparison (2026-08-15)

The guarded CuPy preflight passed on an NVIDIA GeForce RTX 4070 SUPER. Maximum
CPU/CUDA differences were `9.54e-7` for whitened features, `2.29e-5` for
Mahalanobis energy, and `0.001755` for rank-tail surprise. The preflight
observed 12 MiB of process GPU allocation under the 4 GiB cap. During the real
quiet audit, live process use was approximately 196 MiB GPU memory and 1.36 GiB
host RSS under the hard caps.

The matched CUDA audit completed in 202.1 seconds, compared with 209.6 seconds
for the guarded CPU audit. This end-to-end difference is not a kernel-only
speedup: CPU MSLN construction, covariance fitting, bootstraps, fit serialization,
and plotting dominate this quiet-audit runtime. The CUDA backend specifically
accelerates full-field transform application and energy calculation.

CPU and CUDA produced the same `global_full_only` outcome and their maximum
absolute difference across reported held-out metrics was `1.94e-7`. Ranking by
mean held-out covariance identity error was:

1. `global_full_zca`;
2. `local_full_zca`;
3. `identity`;
4. `local_diagonal`;
5. `global_diagonal`.

This ranking does not promote local full ZCA: it remained worse than global
full in both held-out blocks, left 75.48% of support unresolved, and failed G2.
The comparison therefore stops before sparse-positive evaluation and ICA. The
machine-readable comparison is in
`Outputs/LocalWhitening/spon_ca_burst_local_whitening_cpu_cuda_comparison_20260815`.
