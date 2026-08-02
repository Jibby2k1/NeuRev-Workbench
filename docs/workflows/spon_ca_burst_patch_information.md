# Spon Ca Burst Principe-aligned patch information

## Purpose

This workflow tests whether local information-theoretic structure adds useful
neuron evidence beyond the accepted quiet-standardized Parzen Innovation
carrier. It does not treat low entropy as a detector by definition and never
replaces the scientific activity trace with an entropy score.

The three maintained criteria are:

1. quadratic Renyi information potential from a Gaussian-Parzen estimate;
2. Parzen Cauchy--Schwarz divergence between a current patch density and its
   frozen same-location quiet density;
3. local correntropy between the center observation and its spatial patch.

For local quantized probabilities \(p\), quiet probabilities \(q\), and the
Gaussian interaction matrix \(K_\sigma\), the first two criteria are

\[
V_2(p)=p^T K_\sigma p,
\qquad
H_2(p)=-\log V_2(p),
\]

and

\[
D_{CS}(p,q)=-\log
\frac{(p^T K_\sigma q)^2}
{(p^T K_\sigma p)(q^T K_\sigma q)}.
\]

The common Gaussian normalization cancels in \(D_{CS}\). The implementation
retains information potential rather than negating entropy so that greater
local concentration is positive evidence.

## Frozen experiment

- 13 quiet-standardized intensity centers from -6 to +6 z;
- patch sizes 7, 11, and 15 pixels;
- Parzen bandwidths 0.5, 1.0, and 2.0 z;
- 27 feature variants;
- standalone, four carrier boosts, and three bounded carrier gates per feature;
- 216 fixed lanes;
- budgets 20, 40, 58, 80, and 100, with 20/40 primary;
- leakage-safe leave-one-burst fixed-lane selection;
- the complete v5 proposal union augmented with all ITL proposal sources;
- four bounded linear feature sets, 72 configurations, 864 inner fits, and 16
  outer refits.

All image statistics are computed without labels. Labels enter fixed
evaluation and nested model selection only. The carrier remains an immutable
unit skip in every learned ranker.

## Commands

```bash
.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment patch-information preflight \
  --config examples/spon_ca_burst_patch_information_v1.example.json

.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment patch-information run \
  --config examples/spon_ca_burst_patch_information_v1.example.json

.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment patch-information video \
  --config examples/spon_ca_burst_patch_information_v1.example.json \
  --feature-id cs_quiet__p7__bw0p5 \
  --output-dir Outputs/HierarchicalParzenICA/spon_ca_burst_patch_information_video_v1
```

The run requires its exact ready preflight and refuses completed or partial
output collisions. CUDA feature generation uses bounded frame batches; model
fitting uses explicitly bounded CPU threads.

The `video` command preserves every framewise feature map as float16 NPY and
writes a globally normalized uint16 TIFF with one page per review frame. The
global display limits are fixed for the stack, so brightness changes are real
feature changes rather than per-frame autoscaling. The map is causal after the
same-location quiet histogram has been frozen; its manifest reports measured
single-frame GPU latency separately from offline TIFF-writing time.

The completed v1 video contains all 560 accepted carrier frames (inclusive UI
frames 1800--2359), not raw frames 1--1799 that were outside the experiment's
carrier contract. Its measured RTX 4070 SUPER compute latency was 0.83 ms
median and 1.29 ms p95 per frame; causal online evaluation begins after the
100-frame quiet calibration at UI frame 1900.

## Outputs

```text
Outputs/HierarchicalParzenICA/spon_ca_burst_patch_information_v1/
  REPORT.md
  metrics.json
  run_state.json
  evaluation/
    fixed_screen.json
    candidate_inventory.json
    oracle_coverage.json
    inner_fine_tuning.json
    nested_rankers.json
    per_neuron_audit.tsv
  models/
  diagnostics/
    top_itl_feature_maps.tif
    preferred_ranker_scores.tif
    preferred_ranker_overlay.tif
```

Unmatched candidates remain unknown under sparse annotation. Candidate count
is selectivity pressure, not biological precision.

## Completed v1 result

The guarded run completed all 27 ITL features, 216 fixed lanes, and 880 nested
fits. Leakage-safe standalone Cauchy--Schwarz quiet divergence improved mean
recall from 0.541 to 0.572 at budget 20 and from 0.657 to 0.696 at budget 40.
All four folds selected a 7-pixel Cauchy--Schwarz expert. The broad augmented
union increased the 58-per-source oracle ceiling from 0.902 to 0.919 but did
not improve the learned ranker, so it is not promoted. See
`docs/research/SPON_CA_BURST_PATCH_INFORMATION_V1_RESULTS.md` for the exact
interpretation and next checkpoint.
