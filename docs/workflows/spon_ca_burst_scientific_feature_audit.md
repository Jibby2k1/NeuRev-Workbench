# Spon Ca Burst scientific feature audit

## Purpose

This workflow tests three linked hypotheses in canonical scientific order:

1. acquisition and measurement noise may be limiting the current features;
2. a neuron observed through one z-plane may resemble a center, ring, or cap;
3. spatially indexed information and causal neighborhood recurrence may add
   evidence that a pointwise carrier cannot express.

The audit emphasizes noise-model identification, radial Parzen information,
and causal propagation features. It does not treat unlabeled pixels as
negatives and therefore reports known-positive fixed-budget recall rather than
precision.

## Frozen input and geometry

The source is the 560-frame review interval (UI frames 1800--2359) of the Spon
Ca Burst memory-mapped video. The first 100 review frames form the frozen quiet
reference. The input appears to contain two side-by-side fields with a
provisional boundary at `x=286`; all 79 labeled observations lie in the right
field. Full-field and annotated-right-field evaluations are both required.

The 20 ms frame period is inferred from the filename and is not embedded
acquisition metadata. Microscope model, indicator, objective NA, pixel size,
z-section thickness, and the semantic relationship between the two fields are
unknown and must remain explicit limitations.

## Stage 1: acquisition and noise

Adjacent quiet-frame pairs estimate the descriptive model

```text
Var(X[t+1] - X[t]) / 2 = intercept + slope * mean(X[t+1], X[t]).
```

The fit is computed independently for the full frame, left field, and right
field over 24 intensity bins. The audit also measures saturation occupancy,
temporal drift, left/right global-trace correlation, and the boundary jump.
Generalized-Anscombe residual video and fixed-pattern diagnostics are visual
checks, not evidence that the physical detector exactly follows this model.

## Stage 2: generative z-cut morphology

A bank of blurred 2D cuts through cytosolic spheres and membrane shells tests
center, ring, and cap observations. The frozen bank uses radii 3, 4.5, and 6
pixels; normalized z offsets 0, 0.5, 0.8, and 0.93; membrane thickness 1.25
pixels; and Gaussian PSF widths 0.75 and 1.25 pixels. Normalized correlation is
evaluated on pooled quiet and event carrier maps, both at labels and densely.

These templates are mechanistic hypotheses, not fitted biological dimensions.
They are intended to determine whether missed labels cluster by observable
morphology and whether phenotype responses improve ranking.

## Stage 3: structured information and recurrence

Radial Cauchy--Schwarz divergences compare the quiet and current local
intensity distributions in center (`r<=2`), shell (`2<r<=4`), and outer
(`4<r<=7`) zones. Gaussian Parzen bins span -6 to +6 quiet-standardized units
with bandwidth 0.5. Center/shell contrasts and a morphology maximum preserve
spatial organization that a pooled patch histogram discards.

Causal recurrence features measure rolling correlation between each pixel and
its Gaussian neighborhood. Zero-lag windows of 7, 15, and 31 frames measure
local coherence. Lagged pairs `(1,15)`, `(2,15)`, and `(4,31)` measure directed
temporal ordering. All windows use current and past frames only. Lagged
correlation is predictive association; it is not causal biological propagation.

## Evaluation contract

The bank contains 16 maps: six radial-information maps, four z-cut morphology
maps, three coherence maps, and three lagged recurrence maps. Each is evaluated
standalone and as a carrier boost of 0.25, 0.5, or 1.0. This produces 64 lanes
in each of three regimes:

- native full-field proposals;
- native annotated-right-field proposals;
- the identical frozen v5 proposal union.

The total is 192 scored lanes at budgets 20, 40, 58, 80, and 100. Budgets 20
and 40 are primary. Leave-one-burst-out selection across four labeled bursts is
the protected estimate; post-hoc lane maxima are descriptive only. The
identical-proposal regime separates ranking utility from proposal generation.

## Commands

```bash
.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment patch-information scientific-audit-preflight \
  --config examples/spon_ca_burst_scientific_feature_audit_v1.example.json

.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment patch-information scientific-audit-run \
  --config examples/spon_ca_burst_scientific_feature_audit_v1.example.json
```

The run requires an exact ready preflight, bounded CUDA batches, two CPU
threads, sufficient RAM/GPU/disk headroom, and collision-free output roots. It
writes a label projection overlay before any label-driven computation and
refuses to overwrite completed or partial outputs.

See
`docs/research/SPON_CA_BURST_SCIENTIFIC_FEATURE_AUDIT_V1_RESULTS.md` for the
completed v1 result.
