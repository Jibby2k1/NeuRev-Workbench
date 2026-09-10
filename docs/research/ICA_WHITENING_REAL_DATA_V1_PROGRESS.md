# ICA/Whitening Real-Data Evaluation Progress (Historical)

**Status:** historical progress snapshot; superseded by the completed evaluation  
**Synthetic recovery:** retained as a non-blocking warning

For current results and claim limits, read
[the final real-data report](ICA_WHITENING_REAL_DATA_V1_FINAL_RESULTS.md).
The final completion audit records all required gates passed, with 30,891 exact
factorial fits and independent-recording ICA retrieval of 20/51 known-positive
occurrences at budget 58 per block. The separate Gamma-LS result of 6/51 uses a
different protocol and is not a head-to-head comparison.

The text below preserves the 2026-08-31 progress record. Its active, queued,
pending, and future-tense statements describe that snapshot, not current work.

## Frozen real-data contract

The live preflight passed against the source movie hash
`04dfbe2f7cb69d72ff75e23ad17c87b3fd406c96a4b872148de2285a9a44d449`.
Synthetic recovery scores are not used for real-data eligibility; only the
recorded numerical-resolution flag is used. This leaves 30,891 fits: 6,397
temporal, 9,157 spatial, and 15,337 joint spatiotemporal.

A label-blind common proposal universe was frozen from Raw activity, signed
temporal difference, and spatial high-pass lanes. It contains 4,258 candidates
and has digest
`9a47cf004defcc8928c08a7f455de143d402b6b8a9c512209e6e2621f3ae9f70`.
Only after freezing, sparse-label coverage was measured: 77/79 known positives
lie within six pixels of at least one candidate. Unmatched candidates remain
unknown.

## Completed no-whitening lanes

All 11,544 numerically eligible no-external-whitening fits completed and merged
exactly across four shards. This lane used one frozen, label-blind peak frame
per candidate and ranked the maximum absolute ICA component response.

The matched controls show that instantaneous scoring is not an adequate final
screen:

- Raw activity macro known-positive recall at 58: 0.000;
- signed temporal difference: 0.012;
- spatial high-pass: 0.209;
- best temporal ICA: 0.035;
- best spatial and joint ICA: 0.000.

Because the proposal ceiling is 0.975, this is a ranking/scoring failure rather
than proposal absence. The lane is retained as an instant-response diagnostic;
it is not eligible for finalist selection.

The corrected frozen scorer aggregates quiet-MAD-standardized maximum absolute
component evidence over each full event interval with LME temperature 0.25.
All 11,544 no-whitening fits completed under this scorer with exact coverage.
The best macro known-positive recall at budget 58 was 0.262 for temporal ICA,
0.139 for spatial ICA, and 0.175 for joint spatiotemporal ICA. The corresponding
best temporal pooled recall was 0.278 and its mean reciprocal rank was 0.060.
For context, the matched spatial-high-pass proposal score achieved 0.209 macro
recall. These values establish an exploratory within-recording baseline, not a
selected finalist.

Exact sparse external whitening has been verified against the dense reference
for spatial, temporal, and joint geometries under both global and regional/local-
block covariance scopes. Regional scope is not a claim of per-location adaptive
whitening.

The first spatial-whitening pass was scientifically superseded after conflicting
duplicate fit IDs revealed that two implementation revisions had appended to the
same shards. The original files and hashes are retained under
`real_data_v1/superseded/mixed_implementation_spatial_20260831`, with scientific
use prohibited. A clean four-shard spatial rerun is queued after the memory-heavy
geometries finish.

As of the 2026-08-31 18:38 EDT snapshot, temporal whitening is active in four
balanced shards at approximately 1,280 of 2,700 fits per shard. A fresh rerun of
fit `icaw_000841336a732c14` exactly matched its stored score hash, label metrics,
response summaries, whitening diagnostics, calibration, objective, condition,
explained fraction, and iterations. Joint and both separable whitening orders
will follow automatically, then the clean spatial rerun.

The downstream protocol is now frozen as follows:

- S3 requires exact 30,891-fit coverage, reports conditional family-by-whitening
  summaries, and treats scrambled Sobol associations as descriptive rather than
  formal Sobol indices.
- S4 forms a training-burst-only Pareto front using known-positive recall and
  label-independent ICA rank. Each held-out burst is evaluated only for fits
  selected without that burst's labels, across three confirmation seeds.
- S4 exports reconstruction integrity, approximate SNR, coherence/correlation,
  learned responses, whitening operators, PCA/random-rotation controls, and
  seed/subspace stability.
- S5 renders a separate three-section scientific audit for every finalist and
  cannot pass until every package is decoded and visually inspected.
- S6 is preregistered for the eligible independent recording `15 right` (10 ROIs,
  51 positive intervals, 50 Hz). Candidates and all model scores are frozen and
  hashed before labels are joined. `6 left` is excluded because its workbook
  identity conflicts with its filename and it contains malformed-range warnings.
- A fail-closed completion audit currently prohibits any completion claim until
  factorial, interpretability, protected-confirmation, all-finalist visual, and
  independent-confirmation gates all pass.

## Claim limits

These are within-recording sparse-positive measurements. They do not identify
precision, biological-source identity, or cross-recording generalization.
