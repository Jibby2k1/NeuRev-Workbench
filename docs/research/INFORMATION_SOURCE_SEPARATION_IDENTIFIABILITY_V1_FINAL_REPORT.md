# Information Source Separation Identifiability v1 — Final Report

## Outcome

The repaired identifiability program is complete. No source-separation method
passed the held-family advancement gate, so generated confirmation,
semi-synthetic Spon fitting, CaImAn CNMF fitting, and the full Spon benchmark
were stopped by design. A 24-video diagnostic package was completed and
integrity-checked.

This is a scientifically informative negative gate, not a runtime failure.

## Why the earlier screen could not advance

The earlier `similar_persistence` and `unresolved` fixtures were byte-identical
for matched seeds and SNR. Only their metadata labels differed. For seed 7,
both observations had SHA-256
`37c460205e36ae63fbcca3c72ff01a802d8db0aac31c7bdc6662fa61cb8bf378`;
for seed 13, both had
`a405dc0ce72752079ae8a3fce9fb65c206b00261f71b8078c35c4c734a157a65`.

No label-free method could resolve one and abstain on the other. The earlier
selector also did not penalize false abstention on identifiable fixtures. Its
qualification verdict is therefore invalid, although its truth-aligned source
recovery values remain useful exploratory evidence.

## Repaired design

The replacement uses actual numerical mechanisms:

- identifiable calibration families: isolated, overlap, motion edge;
- unidentifiable calibration families: spatial rank deficiency, exact duplicate
  sources;
- identifiable evaluation families: synchronous, saturation, similar
  persistence;
- unidentifiable evaluation families: temporal rank deficiency, pure noise.

Calibration and evaluation use disjoint case families and seeds. Each family is
evaluated at SNR 4, 8, and 16. Four frozen configurations were tested:

- PCA rank 4;
- SOBI rank 4, lags 1/2/4, shrinkage 0.02;
- CUDA normalized-HSIC rotation rank 4, bandwidth 2.0;
- kNN-MI rotation rank 6, k=5.

Confidence features were label-free: neural-evidence score/margin, temporal and
spatial stability under two small perturbations, source correlation, mixing
coherence, and reconstruction residual. Logistic regularization and thresholds
were frozen from grouped calibration predictions. Evaluation labels were
applied once afterward.

The run completed 300 method/fixture rows and 900 numerical fits.

## Held-family gate results

| Method | Identifiable resolved | False resolutions | Convergence | Gate |
|---|---:|---:|---:|---|
| PCA rank 4 | 0/27 | 0/18 | 100% | fail |
| SOBI rank 4 | 9/27 | 1/18 | 100% | fail |
| CUDA HSIC rank 4 | 0/27 | 0/18 | 100% | fail |
| kNN-MI rank 6 | 0/27 | 0/18 | 100% | fail |

The frozen requirements were zero false resolutions, at least 80% resolution
of identifiable mixtures, and at least 95% convergence. No method passed.

For methods with zero observed false resolutions, the 95% Wilson upper bound is
still `0.1759` because only 18 unidentifiable evaluation fixtures were present.
Zero observed errors must not be called proof of zero risk.

PCA, HSIC, and kNN-MI obtained safety only by abstaining universally. SOBI
retained some coverage, primarily on saturation, but violated the zero-false-
resolution gate and resolved only one-third of identifiable cases. The
confidence features did not generalize from spatial/duplicate degeneracy to
synchronous, persistence-confounded, temporal-rank-deficient, and pure-noise
families.

## Diagnostic videos

The completed root is:

```text
Outputs/InformationSourceSeparation/diagnostic_videos_v1
```

Its `manifest.json` records hashes, byte sizes, codecs, frame counts, frame
rates, dimensions, truth labels, confidence probabilities, thresholds, and
decisions for all videos.

### Generated truth-known diagnostics

Twenty videos cover four methods by five held-out families at seed 137 and SNR
8. Each video shows:

1. observed mixture;
2. true neural contribution;
3. top recovered component;
4. full-model residual;
5. true identifiability, calibrated probability, frozen threshold, decision,
   and correctness.

These videos make universal abstention and the SOBI coverage/error tradeoff
directly reviewable.

### Spon detector-blinded diagnostics

Four videos cover every labeled burst. Each frame shows raw intensity,
pseudo-color intensity, positive change from a pre-window baseline, and positive
frame derivative. Cyan circles and numeric IDs are sparse known-positive ROIs;
no detector result is shown.

| Burst | Video frames (UI, inclusive) | Original labeled window |
|---|---|---|
| 1 | 1988–2031 | 2003–2026 |
| 2 | 2025–2068 | 2040–2063 |
| 3 | 2107–2154 | 2122–2149 |
| 4 | 2239–2305 | 2254–2300 |

Burst 2 deliberately includes frame 2031 and the interval before frame 2040,
addressing the expert observation that the burst begins before the recorded
detection window. ROI 007, 008, 010, 014, 015, 017, 019, and 020 are explicitly
labeled where present. ROI 010/015 remains an annotation/adjudication issue;
the video does not silently merge them.

## Stage dispositions

- Repaired identifiability calibration: complete.
- Diagnostic videos: complete.
- Generated confirmation: gated, not run.
- Semi-synthetic Spon: gated, not run.
- CaImAn CNMF: installed and import-verified, fit gated and not run.
- Full Spon benchmark: gated, not run.

Running the dependent stages anyway would violate the preregistered G1-to-G2
progression and turn an unsafe confidence layer into a benchmark candidate.

## Critical interpretation

The generated recovery advantage previously seen for HSIC and kNN-MI remains a
reason to continue research, but it is not sufficient. Stability under tiny
perturbations, neural morphology, temporal burstiness, and residual closure do
not by themselves certify identifiability under unseen degeneracy mechanisms.

A next-generation confidence program would need a broader continuum of mixing
condition number, temporal collinearity, source amplitude ratio, and background
aliasing, with far more unidentifiable evaluation samples. Structural tests of
effective mixing/temporal rank and selective-risk control should be evaluated
before another confirmation request. The present program has reached its
declared terminal gate and should not be widened posthoc.
