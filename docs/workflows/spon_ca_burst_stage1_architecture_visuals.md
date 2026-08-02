# Spon Ca Burst Stage-1 stochastic architecture visuals

## Purpose

This bounded CPU workflow compares four inference architectures around one
stochastic-Parzen ICA fit. It isolates state propagation from ICA estimation.
It writes exactly two method outputs per architecture:

1. estimated background;
2. signed dynamics plus noise, defined as `observation - background`.

The run is a visual and rollout diagnostic. It does not use labels, execute
Stage 2, detect neurons, or establish recall or precision.

## Data and frame contract

The source is the memory-mapped Spon Ca Burst cache:

```text
Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/
  spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy
```

The review interval is UI frames 1800--2359. UI frames 1800--1899 form the
quiet calibration prefix. With lag one, TIFF page 1 corresponds to UI frame
1801 and page 559 corresponds to UI frame 2359.

## Shared stochastic-Parzen fit

One raw stochastic-Parzen demixer is fitted on quiet aligned pairs. The same
fitted affine current-coordinate reconstruction is used by every architecture:

```text
P(t) = a * previous + b * observation(t) + c
```

The raw fit must be resolved, converged, feedback-safe, and retain learned
fraction 1.0. A fallback or partial reference anchor fails the run. This makes
the visual comparison a test of the stochastic solution rather than the
previous 90%-reference interpolation.

## Architectures

### 1. Teacher-forced stochastic

```text
B(t) = P(I(t-1), I(t))
```

The real previous observation is used at every frame. No estimated background
is fed back. This is the closest lane to instantaneous pairwise online ICA with
a frozen demixer.

### 2. Raw stochastic recurrence

```text
B(t) = P(B(t-1), I(t))
```

This exposes the raw stochastic demixer's closed-loop behavior without the
10% reference anchor.

### 3. Quiet fixed-point recurrence

For frozen per-pixel quiet median `Bq`:

```text
B(t) = Bq + a * (B(t-1) - Bq) + b * (I(t) - Bq)
```

The free affine offset is removed. A constant quiet scene is an exact fixed
point by construction.

### 4. Stable reference plus Parzen innovation

First update a stable reference with a declared half-life:

```text
Bref(t) = (1-rho) * Bref(t-1) + rho * I(t)
```

Then form the teacher-forced Parzen correction, subtract its per-pixel quiet
bias, clip it using a quiet-only robust MAD limit, and apply a declared learned
fraction:

```text
delta(t) = clip(P(t) - Bref(t) - quiet_bias, -limit, limit)
B(t) = Bref(t) + learned_fraction * delta(t)
```

The initial manifest uses a 10-second reference half-life, learned fraction
0.1, and a four-MAD correction limit.

## Display contract

Every architecture directory contains:

```text
background.tif
dynamics_noise.tif
```

All backgrounds share one linear scale. All dynamics/noise TIFFs share one
symmetric scale, with code 32768 as zero, darker values negative, and brighter
values positive. Scales are fixed across frames and estimated from a bounded
deterministic sample of the actual architecture outputs. The TIFFs are display
artifacts; exact scale metadata is embedded in the first page and manifest.

## Commands

Read-only preflight:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment parzen-architecture-visuals preflight \
  --config examples/spon_ca_burst_stage1_architecture_visuals.example.json
```

Guarded run:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment parzen-architecture-visuals run \
  --config examples/spon_ca_burst_stage1_architecture_visuals.example.json
```

The runner refuses completed or partial output collisions, limits CPU numerical
threads before scientific imports, checks disk/RAM/process/GPU state during
preflight, writes atomic TIFFs and metadata, records progress heartbeats, and
enforces live RAM/output caps.

## Interpretation

Compare background persistence, negative-background fraction, first-to-last
spatial contrast, quiet dynamics RMS, and post-quiet dynamics RMS. Lower
dynamics RMS is not automatically better: a background model can suppress
neural activity. Visual review should focus on whether sustained neural
structure moves into background and whether motion or saturation remains in
dynamics/noise.

The quantitative follow-up is now implemented and completed. See
`docs/workflows/spon_ca_burst_stochastic_architecture_grid.md` for the
185-effective-operator screen, leave-one-burst-out promotion, full-field
detection, identity bootstrap, and selected background/dynamics TIFFs.
