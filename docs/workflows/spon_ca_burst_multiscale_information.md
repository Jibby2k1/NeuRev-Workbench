# Spon Ca Burst multiscale patch information

## Purpose

This workflow tests whether spatial scale adds useful information to the
quiet-relative Parzen Cauchy--Schwarz map. It separates three questions:

1. which support is best when used alone;
2. whether small and large supports should be fused;
3. whether a feature improves proposal generation, ranking on an identical
   proposal union, or a protected carrier/feature proposal quota.

The input is the accepted quiet-standardized Parzen Innovation carrier from
the feature-utility workflow. Labels are used only for evaluation and
leave-one-burst model selection. Unmatched event candidates remain unknown.

## Frozen experiment

The preregistered grid contains:

- patch widths 5, 7, 9, 11, 13, and 15 pixels;
- Gaussian-Parzen bandwidths 0.5 and 1.0 z;
- 12 single-scale Cauchy--Schwarz maps;
- two scale maxima;
- six log-mean-exp soft scale selections at temperatures 0.25, 0.5, and 1.0;
- ten adjacent-scale geometric agreements;
- six compact-minus-broad contrasts using 7- and 15-pixel supports and broad
  penalties 0.25, 0.5, and 1.0;
- six exact center-versus-annulus divergences for 5/13, 7/15, and 9/15;
- standalone scores and carrier boosts 0.25, 0.5, and 1.0;
- fixed proposal budgets 20, 40, 58, 80, and 100, with 20/40 primary;
- leave-one-burst-out selection across four labeled bursts.

This is 42 feature maps, 168 native lanes, and 168 identical-proposal lanes:
336 scored lanes total. The 50/50 carrier-feature quota and optimistic oracle
are separate estimands and are not counted as lanes.

All single-scale quiet-reference densities are estimated from the frozen
100-frame quiet interval. Fusions are calibrated from quiet data only. Native
maps generate their own local maxima. Identical-proposal experiments instead
score the exact frozen v5 proposal union, preventing proposal count from being
mistaken for ranking quality.

## Commands

```bash
.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment patch-information multiscale-preflight \
  --config examples/spon_ca_burst_multiscale_information_v1.example.json

.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment patch-information multiscale-run \
  --config examples/spon_ca_burst_multiscale_information_v1.example.json
```

The run requires an exact ready preflight and refuses completed or partial
output collisions. CUDA work uses bounded four-frame batches and CPU work is
capped at two threads.

## Interpretation contract

- Fixed-budget recall is known-positive coverage at a fixed candidate count.
- Quiet-threshold candidate burden is selectivity pressure, not precision.
- Post-hoc best configurations are descriptive; only outer-fold selections
  are leakage-safe.
- Oracle union coverage is a proposal ceiling, not deployable performance.
- The batched full-bank timing is an offline throughput measurement. It does
  not replace a single-frame p50/p95/p99 deployment benchmark.

See
`docs/research/SPON_CA_BURST_MULTISCALE_INFORMATION_V1_RESULTS.md` for the
completed v1 result.
