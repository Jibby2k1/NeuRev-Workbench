# Spon Ca Burst Soma-Excitation Experiment

This workflow is a CPU-first frozen-transfer case study for
Inputs/Spon Ca Burst/3 hindbrain to tail 488 20ms.tif. It does not restart the
stopped Stage A sweep, launch Stage B, train new weights, or write dense
full-video detection stacks.

Use the exact NumPy cache already present at:

~~~text
Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/
  spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy
~~~

The cache supports memory-mapped reads. Its shape and dtype were verified
against the TIFF, and sampled TIFF pages were verified byte-identical to the
corresponding cached frames. This was a sampled integrity check, not an
exhaustive frame-for-frame comparison. The source contains 2,359 frames of
shape 340x573, stored as uint16.

## Scientific Interpretation

The existing CFAR implementation is a one-sided positive local-excursion
detector. A dark soma core remaining background is therefore expected behavior,
not a dark-soma segmentation failure. It is also a CFAR-style normal-tail
threshold, not a fitted Gamma-distribution likelihood model.

The experiment keeps three related signals separate:

1. **Dark-core anatomy** comes from negative local contrast in a quiet
   pre-event projection. These are provisional soma-zone anchors.
2. **Broad positive excitation evidence** comes from the half-wave positive
   residual above the frozen quiet baseline. This direct amplitude lane is the
   primary lane for broad excitation because local-background CFAR can suppress
   a spatially broad change. It is summarized globally, in dark cores, in
   perisomatic rings, and per zone.
3. **Positive local-excursion evidence** comes from CFAR on normalized intensity
   and on change relative to the frozen quiet baseline. CFAR remains useful as
   a complementary local-contrast lane, but it is not the sole onset test.

CFAR pixels must not be described as soma bodies or as manual ground truth.

## Frame Contract

All user-facing frame numbers are one-based. Array indices are zero-based and
all stored array intervals are half-open.

| Purpose | Human frames | Zero-based interval | Scored |
| --- | ---: | ---: | --- |
| Quiet calibration/control | 1800-1899 | [1799, 1899) | No |
| Reported event interval | 1900-2359 | [1899, 2359) | Yes |

Normalization, dark-core geometry, and detector thresholds are frozen using
only the quiet interval. No event frame can leak into calibration.

## Frozen Model Case Study

The initial manifest evaluates the current manual-ROI Temporal CNN leaders:

- h5 (100 ms): g128_manual_roi_tcnn_w8_s1_h5_residual_mse_hc64_l4_lr1em04_rs0p1000_e50_s7
- h2 (40 ms): g128_manual_roi_tcnn_w8_s1_h2_residual_mse_hc64_l4_lr1em04_rs0p1000_e50_s13

Each checkpoint is loaded and evaluated sequentially with batch size 1 and
torch.inference_mode() on CPU. Persistence is mandatory as the comparison
baseline.

The source field of view is not registered to the training template. Results
are therefore explicitly out-of-domain and exploratory. Two spatial arms expose
the geometry sensitivity instead of hiding it:

- full-field adaptive max pooling to 128x128;
- fixed 4x4 max pooling to 85x143, preserving the original pooling footprint and
  native aspect ratio.

A transfer conclusion is spatially robust only when both arms agree in sign and
broad magnitude. Neither arm proves anatomical equivalence or external
generalization.

## Resource Guards

The supplied manifest fixes:

- device: CPU;
- workers: 1;
- CPU threads: 2;
- detector chunk: 8 frames;
- model batch: 1;
- RAM cap: 1,024 MiB;
- output cap: 256 MiB;
- checkpoints: one at a time;
- outputs: 2-D maps, scalar traces, sparse zone metadata, and a small review
  selection only.

Preflight resolves 560 analysis frames and 460 scored frames. Its peak-RAM
estimate includes an empirically observed 640 MiB scientific-runtime overhead
for NumPy/SciPy/PyTorch and 32 MiB of enforcement headroom in addition to the
workflow arrays. The completed v2 preflight estimated 780.3 MiB at chunk 8
without reduction. Do not infer the estimate from array sizes alone.

The runner also samples Linux `VmRSS` and `VmHWM` after the detector, after
the transfer stage, and before completion. It records the samples and fails the
run if the 1,024 MiB cap is exceeded. If `/proc` telemetry is unavailable, the
run records that limitation instead of pretending the live check occurred.
The completed v2 run peaked at 757.8 MiB, passed the live guard, and wrote
7.3 MiB under the 256 MiB output cap. Its durable result is
`Outputs/SomaExcitation/spon_ca_burst_transfer_v2_cpu_guarded/report.md`.

The top-level CLI now lazy-loads command groups. For this experiment command it
sets the OpenMP/BLAS thread limits and hides CUDA before importing the scientific
stack. This ordering is a resource guard: the first v1 diagnostic exposed that
eager CLI imports could initialize scientific libraries before the two-thread
limit was applied, causing CPU and RAM oversubscription. Treat that v1 output as
diagnostic evidence, not as the resource-validated v2 run. The current example
manifest targets a fresh v2 output directory.

## Commands

Run the read-only preflight first:

~~~bash
.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment soma-excitation preflight \
  --config examples/spon_ca_burst_soma_excitation.example.json
~~~

Run the bounded CPU experiment:

~~~bash
.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment soma-excitation run \
  --config examples/spon_ca_burst_soma_excitation.example.json
~~~

The runner refuses an existing output directory. Change output_dir for a new
experiment rather than overwriting evidence.

## Output Contract

The output root contains:

- resolved_config.json: immutable resolved paths and scientific parameters;
- preflight.json: frame, RAM, disk, and checkpoint checks;
- run_state.json: running/completed/failed state;
- detector_summary.json: direct positive-residual, CFAR, and zone-onset
  summaries;
- detector_arrays.npz: 2-D count maps and small frame/zone traces, not dense
  video masks;
- transfer_results.json: per-checkpoint, per-spatial-arm online metrics;
- review_frames.npz: a small top-evidence raw-frame selection;
- experiment_summary.json: the compact machine-readable result;
- report.md: interpretation-first human report.

Every time record includes both the zero-based source index and one-based UI
frame number.

## Metrics And Decision Rules

Report separately:

- model and persistence MSE/MAE;
- improvement over persistence;
- high-change improvement;
- positive-change correlation;
- dark-core and perisomatic-ring improvement;
- control versus event half-wave positive baseline residual;
- direct-signal zone activations and onset;
- control versus event CFAR fraction;
- perisomatic enrichment;
- CFAR zone activations and onset;
- agreement between h2/h5 and the two spatial arms.

The current workbench annotations for this video contain no accepted ROI/event
ground truth. Consequently, this run can establish forecast utility and
exploratory concordance only. A soma-excitation detection claim requires a
small manual gate: roughly 20 dark-core zones plus matched background zones,
event/non-event frames, and thresholds fixed before frame 1900. Precision,
recall, AUROC/AUPRC, and h2/h5 agreement should then be reported against that
manual set.

## Avoid These Paths

For this experiment, do not use the current full-stack dataset builder,
all-windows checkpoint backfill, or pixel trainer. Those paths materialize
overlapping windows or upload complete arrays. Do not seed dark-core anatomy
from soma_projection_candidates_v1; it uses full-video, bright-oriented
evidence and would leak event frames into the prior.
