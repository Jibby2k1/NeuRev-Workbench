# Source-off conditional-background predictor feasibility v1

## Outcome first

No tested predictor passed the complete source-off-only feasibility gate. The appropriate next action is not a new residual-detector experiment; first resolve the failed scale, dynamics, whiteness, seam, or recording-consistency checks.

This benchmark used no source truth, injected sources, ROI labels, detector scores, or recovery outcomes. Simple models were fitted on the frozen 512-clip training bank only. Evaluation used all 96 recording-held-out clips plus the exact 12 registered source-off/quiet windows. Native activity inside unlabeled clips remains unknown.

## Aggregate results

### All 96 held-recording clips

| Method | Centered RMS ratio | Dynamic MAD ratio | Seam ratio | Spectral flatness | |lag-1 ACF| |
| --- | ---: | ---: | ---: | ---: | ---: |
| No subtraction | 1.0000 | 1.0000 | 0.9994 | 0.9987 | 0.0376 |
| Last frame / ZOH | 1.4247 | 1.7273 | 0.9996 | 0.7042 | 0.4969 |
| Causal EMA (a=0.25) | 1.0839 | 1.1323 | 0.9999 | 0.9862 | 0.1187 |
| Causal median (w=7) | 1.1138 | 1.1030 | 0.9993 | 0.9884 | 0.0153 |
| Per-pixel AR(1) | 1.4210 | 1.7224 | 0.9996 | 0.7042 | 0.4969 |
| Low-rank AR(1), r=8 | 0.9922 | 1.0024 | 0.9994 | 0.9998 | 0.0425 |
| Frozen JEPA decoder | 1.6541 | 1.1169 | 1.2551 | 0.7141 | 0.5007 |
| Frozen random decoder | 1.0854 | 1.0401 | 1.0894 | 0.9788 | 0.0738 |

### Exact 12 registered source-off windows

| Method | Centered RMS ratio | Dynamic MAD ratio | Seam ratio | Spectral flatness | |lag-1 ACF| | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| No subtraction | 1.0000 | 1.0000 | 0.9995 | 0.9958 | 0.0426 | fail (9/11) |
| Last frame / ZOH | 1.4071 | 1.7262 | 1.0015 | 0.7107 | 0.4945 | fail (2/11) |
| Causal EMA (a=0.25) | 1.0807 | 1.1339 | 1.0016 | 0.9865 | 0.1135 | fail (2/11) |
| Causal median (w=7) | 1.1169 | 1.1102 | 1.0020 | 0.9838 | 0.0410 | fail (4/11) |
| Per-pixel AR(1) | 1.4035 | 1.7230 | 1.0016 | 0.7108 | 0.4944 | fail (2/11) |
| Low-rank AR(1), r=8 | 0.9824 | 1.0057 | 0.9998 | 0.9996 | 0.0406 | fail (5/11) |
| Frozen JEPA decoder | 1.5142 | 1.1039 | 1.2718 | 0.7542 | 0.4657 | fail (0/11) |
| Frozen random decoder | 1.0653 | 1.0309 | 1.0959 | 0.9583 | 0.1218 | fail (2/11) |

## Frozen design

- Common metric support is zero-based frames 8-31 (24 frames) for every method.
- No-subtraction uses a zero background prediction, so its signed residual is exactly the normalized observation.
- Causal EMA alpha is `0.25` and causal median history is `7` frames; neither was validation-selected.
- Per-pixel AR(1) and rank-8 spatial-basis/latent-AR(1) parameters were fit only on the 512 training clips.
- Frozen JEPA and random-provider decoders are byte-verified final-step Run-B heads; there was no refit or checkpoint selection.
- Centered RMS/MAD use per-pixel temporal-median centering. Dynamic MAD uses first differences. Spectral flatness and autocorrelations are residual-whiteness diagnostics. Positive prediction lag means prediction trails observation.
- Seam ratio compares absolute jumps at the frozen 8-pixel decoder lattice with all other adjacent-pixel jumps.

## Compute

- `no_subtraction_identity`: fit `0.000` s; inference `0.013` s across `108` clips.
- `causal_zero_order_hold`: fit `0.000` s; inference `0.010` s across `108` clips.
- `causal_ema_alpha_0p25`: fit `0.000` s; inference `0.027` s across `108` clips.
- `causal_temporal_median_w7`: fit `0.000` s; inference `1.017` s across `108` clips.
- `per_pixel_ar1_train512`: fit `0.173` s; inference `0.020` s across `108` clips.
- `low_rank_ar1_rank8_train512`: fit `0.639` s; inference `0.036` s across `108` clips.
- `frozen_jepa_decoder_run_b`: fit `0.000` s; inference `46.180` s across `108` clips.
- `frozen_random_decoder_run_b`: fit `0.000` s; inference `41.722` s across `108` clips.

## Interpretation boundary

A source-off feasibility pass would only establish that a predictor is numerically safe enough to enter a new detector screen. It would not establish neuron preservation, denoising, specificity, biological identity, independent-animal generalization, or improved detection. This run remains non-claim-bearing and creates no evidence capsule.
