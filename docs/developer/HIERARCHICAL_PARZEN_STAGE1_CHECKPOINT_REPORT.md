# Hierarchical Parzen ICA Stage-1 checkpoint report

Last updated: 2026-07-29.

> Historical note: this report describes the initial unguarded checkpoint.
> The guarded 240-combination follow-up is reported in
> `docs/developer/HIERARCHICAL_PARZEN_STAGE1_GUARDED_SYNTHETIC_REPORT.md`.
> The original batch failure remains useful as the reason the guard was added.

## Bottom line

This checkpoint implemented and smoke-tested four Stage-1 method lanes. It did
not run the Spon Ca Burst dataset, use its labels, train a production model,
produce TIFFs, or execute a GPU experiment.

The scientific result is mixed:

- the fixed reference preserves a synthetic sustained event;
- the adaptive reference also handles a simple global multiplicative drift;
- the batch CS-Parzen fit is catastrophically unstable when placed in the
  recursive background pipeline;
- the short stochastic Parzen smoke fit does not converge and absorbs much of
  the synthetic signal into its background estimate; and
- a one-pixel translation remains a strong false dynamic signal.

Therefore C1 Stage-1 identifiability remains **partial**, not passed. Stage 2
and real-data execution are not justified yet.

## What was actually executed

The focused Stage-1 suite contains eight tests and invokes a Stage-1 lane eight
times:

| Test purpose | Lane executions |
| --- | ---: |
| Sustained-event amplitude | 1 fixed |
| Multiplicative illumination drift | 1 fixed + 1 adaptive |
| Rank-degenerate fallback | 1 fixed |
| Learned-lane bounded smoke | 1 batch + 1 stochastic |
| Slow-ramp leakage metrics | 1 fixed |
| Motion-edge counterexample | 1 fixed |
| Pair construction and staticness tie | No fitted lane |

There was no hyperparameter grid. The direct four-lane comparison below is one
deterministic fixture evaluated once per method. The complete focused
hierarchical/pairwise regression command executed 39 software tests and all 39
passed. A passing software test means the declared behavior occurred; it does
not mean every method achieved acceptable separation.

## Canonical deterministic fixture

The four-lane comparison uses:

| Property | Value |
| --- | --- |
| Shape | 15 frames by 10 rows by 11 columns |
| Static background | Linear spatial field from 0.2 to 1.0 |
| Additive noise | Independent Gaussian, standard deviation 0.002 |
| Calibration prefix | Source frames 0 through 6; 7 frames |
| Injected event | Rows 2:5 and columns 4:8; 12 pixels |
| Event frames | Source frames 7 through 14; 8 frames |
| Event amplitude | Linear ramp from 0.1 to 0.8 |
| Pair lag | 1 frame |
| Fit samples | 256 pixels sampled from quiet aligned pairs |
| Sample seed | 20260729 |
| Covariance | Ordinary |
| Labels used for fitting | None |

The staticness confidence reported below is a score margin, not a calibrated
probability.

## Per-lane results

Lower NMSE and leakage are better; zero is ideal.

| Method | Fit status | Selected background | Feedback `(a, b)` | Signal residual NMSE | Signal leakage into background | Background NMSE | Saved-output closure max |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Fixed reference | Resolved | 0 | `(1.0000, ~0)` | 0.000625 | 0.000625 | 0.0000240 | `5.27e-08` |
| Adaptive-gain reference | Resolved | 0 | `(0.999804, ~0)` | 0.000632 | 0.000632 | 0.0000242 | `4.51e-08` |
| Batch CS-Parzen | Resolved | 1 | `(6.12445, -5.12995)` | `2.77e17` | `2.77e17` | `1.06e16` | `7.54` |
| Stochastic Parzen smoke | Resolved, optimizer not converged | 0 | `(0.518605, 0.481377)` | 0.6776 | 0.6776 | 0.0260 | `6.04e-08` |

The recursive background equation is locally affine:

```text
background(t) = a * background(t-1) + b * observation(t) + offset
```

This makes the learned-lane failures interpretable:

- The references have `b` approximately zero, so a new event remains in the
  residual rather than being copied into background.
- The batch fit has `a = 6.12445`. Recurrent feedback therefore explodes even
  though the one-shot demixer and its objective are finite. Residual RMS reaches
  about `6.64e7`. The pre-cast arithmetic closure is exact, but storing two
  enormous float32 channels causes catastrophic cancellation; recomputed
  closure is therefore also unacceptable.
- The stochastic smoke fit sends about 48% of each current observation directly
  into background through `b = 0.481377`. It consequently absorbs a persistent
  event. It ran only three optimizer iterations, accepted 12 updates, ended
  with gradient norm 2.288, and did not converge. This run is sufficient to
  reject the current unguarded smoke result, but not to establish that a
  properly constrained stochastic architecture can never work.

## Dedicated adaptive-gain check

This fixture has 12 frames of size 14 by 13 and follows
`I(t) = 1.035 * I(t-1) + Gaussian noise` with noise standard deviation 0.001.
Nine frames are used for calibration.

| Measurement | Result |
| --- | ---: |
| True gain | 1.035000 |
| Estimated gain | 1.035047 |
| Fixed-reference residual RMS | 0.06344 |
| Adaptive-reference residual RMS | 0.002809 |
| Adaptive/fixed RMS ratio | 0.0443 |

This demonstrates the intended behavior for a simple global gain change. It
does not test spatially varying illumination, motion, bleaching, or neurons.

## Demonstrated counterexample

The motion fixture translates a textured 18-by-20 background by one pixel after
a quiet calibration prefix.

| Measurement | Result |
| --- | ---: |
| Quiet residual RMS | 0.001418 |
| First translated-frame RMS | 0.3745 |
| Motion/quiet ratio | 264.2 |

The fixed reference confidently classifies its background component but still
reports the translation edge as a large dynamic residual. Staticness confidence
therefore must not be interpreted as neural-signal confidence.

## Metric definitions

For known synthetic background `B`, known signal `S`, estimated background
`B_hat`, residual `R`, and observation `X = B + S`:

```text
background NMSE = ||B - B_hat||^2 / ||B||^2
signal residual NMSE = ||S - R||^2 / ||S||^2
signal leakage into background = ||B_hat - B||^2 / ||S||^2
background leakage into residual = ||R - S||^2 / ||B||^2
closure error = X - B_hat - R
```

These are reconstruction metrics. They are not neuron-detection recall,
precision, F1, or localization metrics.

## What the passing tests establish

The tests establish that:

- quiet calibration is separated from later event frames;
- all four method routes execute deterministically on a tiny bounded input;
- method IDs remain distinct;
- unresolved and rank-degenerate cases avoid subtraction;
- reference reconstruction preserves a slow synthetic ramp;
- output closure is checked; and
- known failure cases are reproducible.

They do not establish:

- performance on Spon Ca Burst;
- neuron recall or precision;
- robustness across seeds, SNR, kinetics, overlap, morphology, or bleaching;
- batch or stochastic learned-lane stability;
- real-time latency;
- detector improvement; or
- readiness for Stage 2.

## Required next checkpoint

Before additional learned-lane experiments:

1. derive and record `(a, b, offset)` for every fitted demixer;
2. reject recursive models with unsafe feedback gain or excessive direct
   observation leakage;
3. add bounded-state and output-scale guards before float32 conversion;
4. constrain or parameterize the learned demixer around the stable reference;
5. run multi-seed synthetic fixtures for slow ramps, impulses, gain drift,
   bleaching, translations, and overlapping sources;
6. report leakage distributions, not just one seed; and
7. integrate accepted-model sign/permutation tracking.

Only after those checks should a semi-synthetic Spon quiet-frame injection be
considered. A full Spon or GPU run remains unauthorized.
