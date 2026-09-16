# Gamma-LS direct and serial two-stencil development study

This workflow implements the user-approved comparison of centered Gamma target
pooling with direct or serial higher-order Gamma reference statistics. It is a
within-recording development experiment. It does not replace the frozen detector
or establish external generalization, precision, or true onset latency.

The detailed design is in the workspace's
`GAMMA_LS_TWO_STENCIL_EXPERIMENT_DESIGN_2026-09-10.md`.

## Frozen comparison

The square-support factorial has 24 cells: point/direct/serial target-reference
construction, Gamma/uniform reference weights, guard 0/guard 7, and conditioned
current-frame/signed adjacent-difference input. A separate point/Gamma/guard 7/disk
cell preserves the deployed spatial operator. Upstream Gaussian and causal
temporal conditioning remain fixed. The centered Gamma target is additional
pooling in this experiment, not a simultaneous replacement of the upstream filter.

The target is an n=1 exponential on a 31x 31 lattice with its discrete radial
second moment matched to Gaussian sigma 1/truncate 4. The reference has n=9 and
mode radius 7.5 on the same 31x 31 lattice. Both kernels are normalized by actual
finite sums; every convolution renormalizes valid support at image boundaries.
Guard 0 imposes no center exclusion. Uniform references then include the center,
while n>1 Gamma references have a naturally zero central weight.

With target response A, direct reference moments act on input X and serial
moments act on A. Both return `(A-M)/(max(sigma, floor)+epsilon)`. The serial
effective reference footprint is wider, and a guard on samples of A does not
exclude all contributions of the corresponding source pixels. Sigma represents
local field variability, not the sampling error of A-M or a standard-normal null.

## Source, calibration, and endpoint

The run reuses the hash-verified 2359x 340x 573 source replay arrays. Calibration
uses UI2..100 (99 scores), excluding the cold-start difference at UI1. The
10 th-percentile positive reference standard deviation supplies each operator's
floor. Separately fitted thresholds target 0.25/0.5/1/2/5 proposals per calibration
score frame. Initialization is not assumed event-free. These q values have a
different unit from the older proposals-per-one-second-block protocol.

Application is UI101..2359. All arms use the maintained 13x 13 square local-maximum
prefilter,6-pixel border exclusion, score/y/x ordering, and greedy Euclidean
separation strictly greater than 6 pixels. One exact NMS prefix at the most
permissive threshold supports every operating point; selected calibration counts
are exact. Saturated lower-bound counts cannot be accepted during calibration.

The first run exposed underfilling by the sampled-pixel threshold grid: an
initialization-only check found an exact99-proposal q=1 cutoff where the grid
had selected zero. The corrected development replay selects cutoffs from exact
NMS peak order statistics, using the largest attainable count at or below
`floor(99*q)`; equal-score ties remain indivisible. This correction applies to
every cell and readout with cached stages and floors unchanged. The first run
is preserved as an incomplete calibration diagnostic, not promoted as an input
or operator comparison. The correction occurred after its outcomes had been
computed, so the corrected run remains development evidence.

All 25 cells seal their stages, calibration records, and complete proposal tables
before the first outcome join. Within each annotated inclusive burst, actual
proposals are grouped around score-ranked spatial representatives within 6 pixels,
using deterministic nontransitive first-representative assignment. Representatives
are matched one-to-one to the nearest unassigned expert within 6 pixels, with
observation-ID ties. Report emitted rows and spatial representatives separately.
There are 79 known occurrences from 26 identities; unlabelled proposals are unknown.

The fixed readout controls also evaluate native-amplitude A and A-M for direct
and serial Gamma guard 0 on both inputs, plus the deployed anchor. Independent
empirical thresholds allow comparison with dimensionless Z. All 175 operating
point results,200 paired factorial contrasts, and 75 readout contrasts are retained.

## Execution

Use a unique non-colliding output root, the repository virtual environment, one
explicit CPU core, and one bounded workload at a time. The verified host run uses
CPU31, threads 1, nice 19, ionice 3, and external GPU chunks of 8 frames. The operator's
internal chunk option alone does not bound memory for an entire returned movie.

```bash
env MPLCONFIGDIR=/tmp/neurobench-mpl-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 31 ionice -c 3 nice -n 19 .venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.two_stencil_campaign preflight --output Outputs/GammaLSTwoStencil/UNIQUE_RUN

env MPLCONFIGDIR=/tmp/neurobench-mpl-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 taskset -c 31 ionice -c 3 nice -n 19 .venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.two_stencil_campaign run --output Outputs/GammaLSTwoStencil/UNIQUE_RUN --device cuda --chunk-frames 8

env MPLCONFIGDIR=/tmp/neurobench-mpl-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 31 ionice -c 3 nice -n 19 .venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.two_stencil_postprocess report --output Outputs/GammaLSTwoStencil/UNIQUE_RUN

env MPLCONFIGDIR=/tmp/neurobench-mpl-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 31 ionice -c 3 nice -n 19 .venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.two_stencil_postprocess audit-preflight --output Outputs/GammaLSTwoStencil/UNIQUE_RUN

env MPLCONFIGDIR=/tmp/neurobench-mpl-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 31 ionice -c 3 nice -n 19 .venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.two_stencil_postprocess audit --output Outputs/GammaLSTwoStencil/UNIQUE_RUN
```

Before a long run, recheck GPU/process ownership, disk and RAM headroom, and kernel
errors. Preflight binds inputs/code by SHA256 and renders a geometry-only expert
overlay. Resume verifies completed source/stage/candidate hashes and preserves
the original campaign seal. Partial output directories never imply completion.

For the current study, the authoritative numerical replay is
`Outputs/GammaLSTwoStencil/two_stencil_factorial_v2_exact_calibration_20260910_r1`.
Its sealed stage source is the preserved v1 root. The uniform correction was run
with the following command after the initialization-only diagnostic was saved:

```bash
env MPLCONFIGDIR=/tmp/neurobench-mpl-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 31 ionice -c 3 nice -n 19 .venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.two_stencil_recalibration --source Outputs/GammaLSTwoStencil/two_stencil_factorial_v1_20260910_r1 --output Outputs/GammaLSTwoStencil/two_stencil_factorial_v2_exact_calibration_20260910_r1
```

Use the corrected root for report, focused diagnostics, and audit commands.
Its stages are symlinks to verified parent arrays; preserve both roots together.

The current media revision is `encoding_v2`. Resume that revision with:

```bash
env MPLCONFIGDIR=/tmp/neurobench-mpl-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 31 ionice -c 3 nice -n 19 .venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.two_stencil_postprocess audit --output Outputs/GammaLSTwoStencil/two_stencil_factorial_v2_exact_calibration_20260910_r1 --audit-revision encoding_v2
```

Add `--cell deployed_signed_point_gamma_g7_disk` for the anchor-only runtime
pilot. The optional `--audit-revision NAME` flag also applies to
`audit-preflight`; it places configurations under `audit_configs/NAME/`, media
under `scientific_audits/NAME/`, and the inventory in
`audit_inventory_plan_NAME.json`. Omitting the flag uses the original directory
layout and must not be used to overwrite or resume the preserved failed attempt.
The old anchor media, configurations, and renderer/postprocessor source copies
remain in the original namespace. `audit_completion.json` records which revision
was verified, and a revision-specific completion file is retained too.

## Scientific audit and focused review

The scientific audit is enabled. Every cell's predeclared q=1 state receives
expert-only/model-only full-field movies, every unique expert ROI and consolidated
model review site's close-up and exact-pixel full-duration trace, and all 79
occurrence comparisons. Other q states retain complete numerical tables and
candidates and are explicitly outside the q 1 media scope. There is no ROI cap.

The seven displayed stages are Raw, X, A, M, sigma, A-M, and Z, with fixed
grayscale scales shared across all cells of each input representation. Full-field
movies sample every fifth source frame plus the final frame; metadata records the
exact map. Close-ups retain every occurrence/proposal frame plus five frames of
context. Per-ROI traces retain all 2359 samples. Occurrence comparison plots show
the corresponding burst plus context, while correlations and descriptive lags
use the exact burst only. No onset-truth claim follows from these lags.

Expert coordinate variants are preserved per occurrence. A deterministic actual
occurrence supplies each identity's fixed trace anchor; it is not an averaged
coordinate. Nearest-candidate traces and one-to-one assignments remain distinct.
The renderer verifies inventories, full video decode, encoded annotation-color
separation, PNG validity, and source/output hashes before reporting completion.

The first lossy video attempt introduced green-classified pixels into a
grayscale-and-orange model composite. The bounded reproduction and encoder
comparison are preserved in `audit_encoding_diagnostic_v1/` under the corrected
campaign root. The `encoding_v2` scientific masters use `libx264rgb` CRF 0 and
verify every decoded RGB frame byte-for-byte against its source composite, with
exact grayscale/section-color palette and per-frame marker-presence checks.
Per-frame hashes are temporary, while source/decoded stream hashes and checked
frame counts are persisted. H.264 High 4:4:4 Predictive/gbrp may require a
software-capable player; PNG previews are retained. No unchecked lossy proxy is
a validated scientific master.

RGB losslessness concerns the displayed source composite. The underlying
floating-point stage arrays still use the shared fixed grayscale display limits,
clipping, spatial resampling, and the declared source-frame selection. The
movies do not preserve the original floating-point values; the hash-bound arrays
and full-duration exact-pixel numerical traces retain that evidence.

After all 25 `encoding_v2` states complete, run the independent metadata check:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 31 ionice -c 3 nice -n 19 .venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.two_stencil_media_geometry --output Outputs/GammaLSTwoStencil/two_stencil_factorial_v2_exact_calibration_20260910_r1
```

The mandatory final artifact is `audit_media_geometry_validation.json`. It
requires all 25 states and all 8,176 expected videos, verifies exact full-field
and ROI frame maps, checks the sealed probe's canvas dimensions, rational FPS,
frame count, and duration, and binds the manifests, artifact indexes, contracts,
protocol, and numerical results by SHA-256. Duration is displayed sample count
divided by playback FPS, including the final sample duration; gaps between source
windows are not added to the clip duration. Actual MP4 hash verification is
explicitly inherited from the bound completed-audit checks; this metadata pass
matches each video's digest against its artifact-index digest and checks current
existence/size without rereading large media or source arrays. A partial campaign
or validation failure cannot produce a complete final geometry certificate.

The 14-card pilot uses all 6 successes and 8 deterministically chosen misses from
the earlier fixed occupancy audit. It is outcome-enriched review material, not a
performance sample. New q 1 stage-of-loss diagnostics retain all 79 occurrences
per cell and highlight those same 14 IDs. Threshold, spatial/readout, and matching
loss categories describe pipeline behavior; motion, biology, and causal mechanisms
still require evidence and reviewer interpretation.

## Promotion boundary

Computation, numerical validation, full q 1 media audit, manuscript readiness, and
external scientific confirmation are separate states. The matrix does not select
a winner automatically. A prospective untouched recording and renewed same-head
ICA/PCA comparison are subsequent confirmation work after the operator and task
have been fixed. Original protected results retain their original operators,
calibration units, folds, and readouts.
