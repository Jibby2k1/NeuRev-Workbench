# Spon Ca Burst signed frame derivatives

## Purpose

Create visually interpretable signed temporal differences from the original
Spon Ca Burst intensity stack. These are diagnostic videos, not activation
labels or CFAR detections.

## Transform and display encoding

For lag `k`, the signed difference is:

```text
D[t] = I[t] - I[t-k]
```

The recording period is 20 ms. Lag 1 therefore measures a 20 ms change and lag
4 an 80 ms change. Each stack uses its own exact global 99.5th percentile of
`abs(D)` as a fixed symmetric display scale. Values are encoded as uint16:

```text
32768 = zero change
below 32768 = negative / dimmer than the earlier frame
above 32768 = positive / brighter than the earlier frame
```

Normalization is never performed independently per frame. The first `k`
frames are neutral gray because their differences are undefined. Output frame
`t` remains aligned to source frame `t`.

## Completed outputs

Root: `Outputs/FrameDifference/spon_ca_burst_derivatives_v1`

| File | Lag | Global scale | Negative clipped | Positive clipped |
| --- | ---: | ---: | ---: | ---: |
| `spon_ca_burst_derivative_lag1.tif` | 20 ms | 245 | 0.2435% | 0.2446% |
| `spon_ca_burst_derivative_lag4.tif` | 80 ms | 251 | 0.2393% | 0.2534% |

Both are uncompressed uint16 BigTIFF stacks with shape `2359 × 340 × 573` and
embedded JSON metadata. Random-frame checks exactly reproduced the declared
arithmetic, sampled source TIFF frames matched the memory-mapped cache, and no
partial files remained. Use Fiji/ImageJ or another BigTIFF-capable stack viewer
and avoid per-frame auto-contrast when comparing temporal magnitudes.

## Reproduction

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment frame-difference preflight \
  --config examples/spon_ca_burst_derivatives.example.json

.venv-neurobench/bin/python -m neurobench.cli.main experiment frame-difference run \
  --config examples/spon_ca_burst_derivatives.example.json
```

The output root is immutable: the command refuses an existing destination.

## Smoothed diagnostic set

The completed smoothed set is rooted at
`Outputs/FrameDifference/spon_ca_burst_smoothed_derivatives_v1`. The original
intensity stack was spatially Gaussian-smoothed (`sigma=1 px`) and then
temporally smoothed with a centered 7-frame, order-2 Savitzky-Golay filter
before differencing. No motion correction was applied. Centered temporal
smoothing is appropriate for offline visualization but is noncausal and must
not be silently reused in an online controller.

Four uint16 BigTIFFs were generated:

- `spon_ca_burst_smoothed_derivative_lag1_global.tif`
- `spon_ca_burst_smoothed_derivative_lag4_global.tif`
- `spon_ca_burst_smoothed_derivative_lag1_quiet_mad.tif`
- `spon_ca_burst_smoothed_derivative_lag4_quiet_mad.tif`

Global scales are 27.6411 raw units for lag 1 and 76.4534 for lag 4. Quiet-MAD
views divide each pixel by its quiet-period robust derivative scale, clip at
`+/-5 z`, and set `abs(z)<2.5` to neutral gray for visualization only. Using
the exact annotated intervals, the non-neutral fractions were:

| View | Quiet | Burst 1 | Burst 2 | Burst 3 | Burst 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lag 1 quiet-MAD | 1.450% | 2.462% | 2.272% | 4.641% | 4.247% |
| Lag 4 quiet-MAD | 1.482% | 4.683% | 4.162% | 9.592% | 8.047% |

Lag 4 therefore provides the clearest event-versus-quiet visual separation in
this diagnostic. This does not prove that every surviving pixel is neural:
unregistered motion can still create paired positive/negative edges.

Reproduce with:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment smoothed-frame-difference preflight \
  --config examples/spon_ca_burst_smoothed_derivatives.example.json

.venv-neurobench/bin/python -m neurobench.cli.main experiment smoothed-frame-difference run \
  --config examples/spon_ca_burst_smoothed_derivatives.example.json
```

## Activity-gated intensity review set

The completed bounded comparison is rooted at
`Outputs/FrameDifference/spon_ca_burst_activity_gate_v1`. It covers UI frames
1800--2359 only and preserves alignment with the source stack. This is a
visual preprocessing comparison, not a detector evaluation. All intensity and
noise scales are fixed across frames; no per-frame normalization is used.

The shared activity gate uses the lag-1 derivative of the same Gaussian/Savitzky-
Golay smoothed video. The derivative is centered and divided by a per-pixel
quiet-period MAD, squared, and accumulated with a causal four-frame EMA:

```text
G[t] = 1 - exp(-EMA(z[t]^2) / (2 * 2.5^2))
```

The centered Savitzky-Golay input still makes the complete transform noncausal.
Four 560-frame uint16 BigTIFFs were generated:

- `spon_ca_burst_strict_gate.tif`: compressed intensity times `G`;
- `spon_ca_burst_floored_gate.tif`: compressed intensity times `0.2 + 0.8G`;
- `spon_ca_burst_artifact_gate.tif`: the floored gate times `1 - 0.7A`, where
  `A` is a quiet-only high-brightness, low-variability/saturation score;
- `spon_ca_burst_baseline_residual.tif`: 15% quiet anatomy plus positive
  residual above the quiet median.

The artifact score marked 0.552% of pixels at `A >= 0.5`. Within that mask,
the quiet median fell from 0.2050 in the floored view to 0.0669 in the
artifact-aware view, a 67.4% reduction. The measured mid-brightness anatomy
median remained 0.0950, and the aggregate event-contrast metrics across the 79
point labels matched the floored view. The strict gate
suppressed the artifact more strongly (median 0.00639), but also reduced the
anatomy median to 0.02394. The baseline-residual view produced the largest
pointwise event-minus-quiet contrast (median 0.9465), but retained more static
artifact (median 0.15).

These contrasts use each labeled coordinate's maximum within its annotated
window and are descriptive, not precision/recall. The labels are sparse and
unlabeled pixels remain unknown. Visual review should therefore compare the
artifact-aware and baseline-residual views first; only a later frozen detector
and full-field hard-negative annotation can determine whether either improves
precision or recall.

Reproduce with:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment activity-gate preflight \
  --config examples/spon_ca_burst_activity_gate.example.json

.venv-neurobench/bin/python -m neurobench.cli.main experiment activity-gate run \
  --config examples/spon_ca_burst_activity_gate.example.json
```

## Offline versus causal detection benchmark

The completed paired benchmark is rooted at
`Outputs/FrameDifference/spon_ca_burst_activity_gate_benchmark_v2`. It uses the
same Raw Direct temporal log-mean-exp pooling, six-pixel NMS, one quiet peak per
pseudo-burst-map calibration, and four labeled burst windows for every lane.
Raw Direct reproduced its historical mean recall exactly (`0.605615942`), which
is the validity check for comparison with the earlier experiments.

The displayed artifact gate itself is not real-time: its centered seven-frame
Savitzky-Golay filter has three frames, or 60 ms, of look-ahead at 50 Hz. The
causal alternatives replace that filter with a temporal EMA (span four frames).
They require a fixed 100-frame/2-second initial calibration but thereafter keep
only the previous temporal state and, when used, the previous energy state.

Primary results at the quiet-calibrated threshold were:

| Lane | Mean recall | Matches | Event candidates | Held-out quiet peaks/map |
| --- | ---: | ---: | ---: | ---: |
| Raw Direct | 0.6056 | 49/79 | 232 | 1.375 |
| Raw Direct + static artifact attenuation | 0.6056 | 49/79 | 251 | 1.250 |
| Offline derivative-energy artifact gate | 0.0472 | 4/79 | 6 | 3.000 |
| Causal artifact-only | **0.7342** | **58/79** | 745 | **1.000** |
| Causal derivative gate, floor 0.2 | 0.0472 | 4/79 | 6 | 2.625 |
| Causal derivative gate, floor 0.4 | 0.1982 | 16/79 | 20 | 2.625 |

This isolates two effects. Static attenuation alone does not improve Raw Direct,
because quiet-median subtraction already removes most persistent intensity.
Causal spatial/temporal smoothing plus artifact attenuation improves labeled
ranking, while multiplying intensity by derivative energy is far too selective
and removes slowly evolving calcium activity.

The causal artifact-only lane is not a demonstrated precision improvement at
its raw quiet threshold: it produces many more event candidates, and unmatched
event candidates remain unknown. A secondary capacity-matched analysis caps it
at 58 peaks per burst map, the Raw Direct average burden (`232 / 4`) and a value
chosen without consulting labels. At exactly 232 total candidates it recovers
58/79 labels versus Raw Direct's 49/79. The known-label candidate fraction rises
from 21.1% to 25.0%, but remains only a lower bound on precision. A paired
10,000-sample bootstrap over the 27 ROI identities estimates a `+0.1139` pooled
recall difference (95% percentile interval `[0.0119, 0.2254]`; 98.03% of
resamples above zero; 11 discordant gains and 2 losses). Treat this secondary
analysis as promising evidence requiring confirmation, not a final model claim.

A stricter spatial-consensus option keeps causal candidates only when Raw Direct
has a peak within eight pixels. It yields 48/79 known matches among 179
candidates (26.8% known-label fraction). This is a useful selectivity mode, but
it does not preserve the causal lane's recall gain.

### Real-time timing on the development PC

Frame-by-frame timing used the native 340x573 field on this PC:

- causal smoothing, compression, artifact attenuation, and quiet residual:
  median 2.76 ms, p95 3.17 ms, maximum 3.54 ms over 440 timed frames;
- the same preprocessing plus a causal 47-frame rolling log-mean-exp map and
  full-field six-pixel NMS: median 5.96 ms, p95 6.41 ms, p99 6.60 ms, maximum
  8.62 ms over 393 timed frames.

The 50 Hz compute deadline is 20 ms, leaving 13.6 ms p95 headroom for the full
tested software path. This demonstrates local compute feasibility, not total
camera, transfer, decision, and stimulation latency. A production path must use
a ring buffer, fixed calibration state, a quiet-threshold floor plus the
58-candidate cap, timing telemetry, and drift/recalibration guards.

### Precision checkpoint

The next valid test is manual review of a stratified candidate batch, including
Raw-only, causal-only, Raw/causal consensus, and held-out quiet candidates.
Until those candidates receive foreground/background labels, report unmatched
event peaks as unknown and do not call the known-label fraction precision. The
derivative-energy signal should remain available as an auxiliary ranking or
artifact feature rather than a multiplicative gate.

Reproduce the v2 comparison with:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment activity-gate-benchmark preflight \
  --config examples/spon_ca_burst_activity_gate_benchmark.example.json

.venv-neurobench/bin/python -m neurobench.cli.main experiment activity-gate-benchmark run \
  --config examples/spon_ca_burst_activity_gate_benchmark.example.json
```

## Causal proposal breadth/depth program

The preregistered overnight program is frozen in
`examples/spon_ca_burst_causal_proposal_overnight.example.json`. It is restricted
to the Spon Ca Burst review interval and starts with an exact C0 reproduction of
Raw Direct (`0.605615942` mean recall) and causal artifact-only (`0.734187371`).
Failure of either check stops the program before the wider search.

The maximum program contains 72 breadth methods under nine operating policies
(648 evaluations), then conditionally 8 finalists under 12 bounded fusions and
nine policies (864 evaluations), and 12 finalists under 31 calibration,
acquisition-perturbation, and early-horizon conditions (372 evaluations). The
maximum is therefore 1,884 logical evaluations and 7,536 fold-condition scores.
The 72 methods comprise four anchors, 20 one-factor-at-a-time tests, and a frozen
48-run space-filling fractional design. The fixed CFAR anchor is
`center_r8_sector_censored_causal_coherence` from v6.

All methods share the same quiet calibration, NMS, match radius, candidate-cap,
and sparse-label interpretation. C1 requires at least +0.03 nested mean recall
and wins in at least three of four bursts at the 58-candidate cap. C2 permits a
fusion only for at least +0.02 mean recall, or at least 20% candidate reduction
without recall loss. C3 requires robust median recall above Raw Direct and its
lower quartile no more than 0.05 below Raw Direct. These gates select proposals;
they do not establish precision because ordinary unmatched event candidates are
unknown.

The run writes atomic configuration and status records, JSONL progress/resource
heartbeats, compressed method-map checkpoints, stage summaries, a 240-row
stratified candidate-review queue, and `morning_report.md`. The review queue
includes causal-only, Raw-only, consensus, CFAR-only, quiet hard-negative, and
detector-independent samples. Its coverage mode is explicitly candidate review,
not exhaustive truth.

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment causal-proposal-program preflight \
  --config examples/spon_ca_burst_causal_proposal_overnight.example.json

.venv-neurobench/bin/python -m neurobench.cli.main experiment causal-proposal-program run \
  --config examples/spon_ca_burst_causal_proposal_overnight.example.json
```

### Completed result and temporary sideline, July 27

`Outputs/FrameDifference/spon_ca_burst_causal_proposal_overnight_v1` completed
all 1,884 logical evaluations in 1,213.9 seconds. C0 reproduced Raw Direct at
`0.605615942` and causal artifact-only at `0.734187371`. C1 passed: the nested
comparison was `0.705020` versus `0.594746` for Raw Direct and won all four
bursts.

The nominal best fractional method tied causal artifact-only at 58/79 known
labels. A more useful Pareto result, `fractional_494f8eee07`, also recovered
58/79 while producing 488 natural event candidates instead of 745 for the
causal reference. It used spatial sigma 1, causal EMA span 1, artifact
attenuation 0, asinh scale 5, a slow clipped EMA baseline, and log-mean-exp
pooling 0.25. This supports adaptive baseline subtraction as a candidate-yield
control, not as an established precision improvement.

Fixed CFAR recovered 34/79 with 65 candidates. Adding 10% CFAR fusion did not
improve recall and reduced candidates by only 1.34%, so C2 correctly stopped
further fusion stages. Robustness median and lower quartile were both
`0.734187`; the worst tested photobleach condition reached `0.6464`. The run
produced a 206-row review queue.

This program is now **sidelined**: do not widen or restart it. The next valid
action on this branch is manual foreground/background review of its fixed queue.
Current implementation work instead follows the pairwise source-separation
workflow, using synthetic/tiny validation only until a full Spon run is
explicitly selected.
