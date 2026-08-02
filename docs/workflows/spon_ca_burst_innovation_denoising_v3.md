# Spon Ca Burst innovation denoising v3

## Purpose

This workflow tests the next eight bounded denoising architectures suggested by
the corrected advanced-denoising v2 result, then constructs mixtures only from
a diverse multiobjective Pareto subset. The accepted Parzen Innovation
residual remains the immutable carrier in every lane.

The eight independently tuned families are:

1. overlap-add local noise-PSD Wiener filtering;
2. four-case morphology-conditioned shrinkage for isolated/crowded center and
   membrane observations;
3. NMF with explicit spatial-concentration and temporal-dynamics component
   selection;
4. ICA-component fast-rise/slow-decay innovation-gated dynamics;
5. activity-tempered, correction-limited Parzen posterior authority;
6. quiet-anatomy-guided spatial graph diffusion;
7. cross-scale sign/evidence consensus;
8. self-supervised linear blind-spot prediction with bounded correction.

The ninth algorithmic direction is a nonnegative bounded mixture of the family
corrections. The tenth direction is the multiobjective Pareto procedure used to
choose four diverse sources for those mixtures.

## Frozen design

The executable manifest is:

```text
examples/spon_ca_burst_innovation_denoising_v3.example.json
```

It declares:

- 96 Stage A breadth settings: 12 per family;
- 16 Stage B full-field semifinals: two per family;
- eight family finalists: one per family;
- four diverse Pareto sources selected from those finalists;
- eight full-field mixtures: all six source pairs and two four-source
  authority levels;
- ten TIFF finalist pairs: eight family finalists and the two best mixtures;
- at most six actual seed refits: two promoted candidates by three seeds.

The screening total is 120 evaluations before conditional confirmation:
96 crop evaluations, 16 full-field semifinals, and eight mixtures. TIFF
recomputation and identity-carrier measurement are not counted as tuned
combinations.

## Carrier and correction contract

Let the quiet-standardized Parzen Innovation carrier be \(R\). Every family
returns an estimate \(D_m(R)\), but the deployed candidate is a bounded
correction:

\[
\widehat S_m = R + \alpha_m \,
  \operatorname{clip}\!\left(D_m(R)-R,-c_m,c_m\right).
\]

Zero authority is therefore the exact identity carrier. A mixture uses:

\[
\widehat S_{\mathrm{mix}} =
R + \operatorname{clip}\!\left(
\sum_m w_m(D_m(R)-R),-c,c\right),
\quad w_m\ge0,\quad\sum_mw_m\le1.
\]

This prevents cancellation through negative weights and prevents a mixture
from acquiring more than unit correction authority.

## Evaluation and gate

The identity carrier is measured on the same crop, full field, real labels,
and four-morphology synthetic fixture. Each candidate reports:

- sparse-known-label threshold recall;
- fixed-budget recall;
- candidate burden;
- ROI waveform, peak, area, onset, and timing preservation;
- quiet signal RMS;
- exact synthetic noise attenuation and localized matched-trace metrics for
  all four morphologies.

Sparse real labels are not exhaustive. Unmatched candidates remain unknown,
not false positives. Candidate burden is a precision-pressure proxy only.

A candidate advances only if it simultaneously has:

- median real peak retention at least 0.85;
- median real area retention at least 0.85;
- median real peak error at most one frame;
- localized synthetic correlation at least 0.70;
- fixed-budget recall no worse than the identity carrier; and
- candidate burden no more than twice the identity carrier.

Only candidates passing that joint gate can trigger actual three-seed refits.

## Pareto selection

The family finalists are compared without collapsing all scientific goals into
one scalar. The Pareto objectives maximize fixed-budget recall, threshold
recall, peak retention, area retention, and synthetic correlation while
minimizing candidate burden, quiet RMS, and peak timing error.

If the Pareto front has more than four candidates, deterministic normalized
farthest-point selection retains a diverse subset. If it has fewer than four,
the remaining slots use the declared scalar screening score.

## Real-time interpretation

The component dynamics lane is causal. Morphology, graph, cross-scale, fitted
blind-spot, and fixed-transfer mixtures are framewise after calibration. The
screening implementation of local PSD-Wiener estimates total spectra over the
review interval, and the NMF lane uses overlapping temporal windows; those two
are offline scientific screens rather than real-time implementations.

A real-time PSD revision must freeze the transfer function after calibration
or update it with a causal bounded EMA. A real-time NMF revision requires
online warm-started factors or a distilled framewise correction. Runtime
metrics in the completed result determine which lanes are plausible at the
50 Hz source rate; scientific quality alone does not establish real-time
readiness.

## Resource and output contract

The run uses the 560-frame interval 1800--2359 and the same one-based UI,
zero-based NumPy, and x-column/y-row coordinate conventions as prior Spon Ca
Burst work. It refuses completed or partial output collisions.

Expected output:

```text
Outputs/HierarchicalParzenICA/spon_ca_burst_innovation_denoising_v3
```

Each finalist directory contains:

- `signal_positive.tif`;
- `remainder_detail.tif`.

Every pair satisfies exact input = signal + remainder closure before display
normalization. Machine metrics, the Pareto source decision, progression,
conditional seed records, and a concise report are written atomically.

## Commands

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment \
  innovation-denoising-v3 preflight \
  --config examples/spon_ca_burst_innovation_denoising_v3.example.json

.venv-neurobench/bin/python -m neurobench.cli.main experiment \
  innovation-denoising-v3 run \
  --config examples/spon_ca_burst_innovation_denoising_v3.example.json
```

## Completed run

The v3 root completed on 2026-07-30 in 397.6 seconds with 11,861 MiB peak
RSS. All 20 TIFFs passed page-count and geometry verification, and no partial
artifact remains. No candidate reached the joint advancement gate, so actual
seed refits did not run. The authoritative interpretation is in
docs/research/SPON_CA_BURST_INNOVATION_DENOISING_V3_RESULTS.md.
