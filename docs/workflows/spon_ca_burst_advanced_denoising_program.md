# Spon Ca Burst advanced denoising program

## Purpose

This workflow compares ten carrier-preserving denoising and source-separation
families on the accepted Parzen Innovation residual. It uses successive halving
so broad parameter coverage does not create a full-field TIFF for every setting.

The ten families are:

1. skip-connected shared-Parzen ICA;
2. per-component adaptive-Parzen ICA;
3. multiscale convolutional ICA group shrinkage;
4. bounded ICA-estimated noise subtraction;
5. quiet-noise PSD Wiener filtering;
6. short-window robust low-rank plus sparse decomposition;
7. short-window nonnegative factorization;
8. spatial nonlocal patch denoising;
9. component-space causal Kalman filtering; and
10. undecimated spatial wavelet group shrinkage.

Every method receives a signed, quiet-standardized residual. Every output keeps
the same TYX shape, writes an exact signal/remainder closure for finalists, and
is evaluated without treating sparse unlabeled pixels as negatives.

## Frozen staged design

The authoritative corrected manifest is:

```text
examples/spon_ca_burst_advanced_denoising_program_v2.example.json
```

The design contains:

- 69 unique Stage A hyperparameter settings across all ten families;
- two Stage A selections per family, giving 20 full-field Stage B semifinals;
- one Stage B selection per family, giving 10 finalist TIFF pairs; and
- 120 declared seed-by-held-burst confirmation strata.

Stage A uses the complete 560-frame event timeline in a 32-pixel-margin crop
enclosing every known label, plus the exact four-morphology semi-synthetic
fixture. Stage B runs the best two settings per family on the full 340-by-573
field. Only one finalist per family receives full TIFF output.

Confirmation refits are conditional. They must not run when no family finalist
passes the frozen preservation gate.

## Evaluation correction

Version 1 used the inherited full-frame summed synthetic trace. Real quiet
residual outside an injected source could therefore determine the reported peak
frame, even for a framewise denoiser that cannot shift time.

Version 2 replaces that diagnostic with one matched spatial projection per
injected morphology quadrant. It reports median morphology-specific trace
correlation, timing, amplitude, and area, along with global exact-truth NMSE and
noise attenuation. The anchors are:

- exact truth: correlation 1 and peak error 0;
- unprocessed noisy fixture: median correlation 0.645 and peak error 2 frames.

The completed v1 output is preserved for audit but is not authoritative for
synthetic interpretation or family selection.

## Promotion gate

A finalist must simultaneously satisfy:

- median real-event peak retention at least 0.85;
- median real-event area retention at least 0.85;
- median real-event peak error at most one frame; and
- localized synthetic correlation at least 0.75.

This gate is intentionally stronger than merely improving one detection metric.
If no finalist passes, seed/held-burst refits stop by design.

## Commands

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment \
  advanced-denoising-program preflight \
  --config examples/spon_ca_burst_advanced_denoising_program_v2.example.json

.venv-neurobench/bin/python -m neurobench.cli.main experiment \
  advanced-denoising-program run \
  --config examples/spon_ca_burst_advanced_denoising_program_v2.example.json
```

The run requires a matching ready preflight and refuses completed or partial
output collisions.

## Resource contract

The v2 preflight estimated 5.07 GiB peak RAM and 2.16 GiB peak CUDA memory,
under explicit 12 GiB and 9 GiB caps. The completed run used 9.02 GiB peak RSS,
reached high CUDA utilization during dense ICA, finished in 291 seconds, and
wrote approximately 3.4 GiB.

The authoritative result root is:

```text
Outputs/HierarchicalParzenICA/spon_ca_burst_advanced_denoising_program_v2
```

Each directory under `finalists/` contains `signal_positive.tif` and
`remainder_detail.tif`. Signal TIFFs use a shared display scale; remainders use
a variant-specific symmetric scale recorded in TIFF metadata.
