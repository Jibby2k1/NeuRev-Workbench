# Gamma-LS framewise exact-truth benchmark v1

## Purpose and claim boundary

This standalone benchmark asks whether the frozen radial Gamma-LS deployment
context and two simpler spatial controls recover simulated fluorescent sources
under exact, frame-resolved truth. It is a mechanistic stress test. It does not
establish biological realism, in-vivo precision, independent-animal
generalization, or voltage-imaging readiness. Precision, recall, F1,
localization error, and duplicate counts from this workflow are valid only
inside the generated movies, where the source list is exhaustive.

The implementation is intentionally separate from the paper campaign CLI and
preflight code. It is CPU-only in v1; running it does not launch a GPU job.

## Frozen comparison

Each movie has 96 source frames on a 64 by 64 grid. The common causal input is
Gaussian smoothing followed by an EMA. All three representation arms are
aligned to current source frames 1 through 95 (zero based), so every arm has 95
scored output frames:

- raw common input;
- signed one-frame difference;
- energy-normalized one-frame difference.

Each representation is crossed with all three spatial operators:

- radial Gamma-LS at the already frozen context `h15/g7/n9/mode0.5`
  (`31 px` support width, `7 px` guard, `7.5 px` mode radius);
- signed square-annulus local standardization at `h11/g3`;
- the maintained positive-clipped box-CFAR control at `h11/g3`.

This is a nine-pipeline fixed comparison. There is no representation search,
temporal max pooling, time-collapsed site score, or evaluation candidate
capacity. Spatial NMS and matching operate independently on each scored frame.

## Simulator truth and six families

The generator writes every source identity, event impulse, latent fluorescence
trace, activity flag, and y/x center for every source frame. The frozen activity
definition is `latent_trace >= 0.12`; it is fluorescent-contribution truth, not
spike-time truth. Centers include common subpixel motion in every family and
source-specific trajectories in the two nonrigid families. Simulated neuropil,
bleaching, and noise fields are deliberately defined nuisance processes rather
than omitted target sources; truth is exhaustive for the benchmark's discrete
target-source class.

The evaluation grid uses source counts 2, 4, and 8 and six mechanisms:

1. baseline;
2. dense neuropil;
3. bleaching;
4. source-specific nonrigid motion;
5. empirically scaled, spatially and temporally correlated read noise;
6. compound shift combining dense neuropil, bleaching, nonrigid motion, and
   empirically scaled noise.

The empirical-noise scale is a frozen label-free statistic from a 48-frame
central crop of the source-authority `15 right` recording. The recording's
labels, coordinates, and identities are not used, and its noise statistic does
not make the simulator biologically validated.

## Isolation and empirical operating points

The source-controlled config freezes four disjoint seed sets:

- `scale_floor_fit`: nine zero-source movies from baseline, dense-neuropil, and
  bleaching development families;
- `threshold_calibration`: twelve different zero-source development movies;
- `evaluation`: 108 untouched source-present movies, including unseen seeds for
  the three development families and complete mechanism-family holdouts for
  nonrigid motion, empirical noise, and compound shift;
- `latency`: one separate zero-source baseline fixture.

For every pipeline, empirical thresholds are calibrated on the source-free
threshold split at fixed null NMS burdens of 0.25, 0.5, 1, 2, and 5 proposals
per scored frame. Threshold keys contain only pipeline and burden. Source count,
evaluation truth, and source-derived candidate budgets are absent from
calibration. These are empirical proposal-burden operating points, not analytic
CFAR probabilities.

Candidates are score-ranked by deterministic spatial NMS. Within each source
frame, candidates are greedily matched one-to-one to the nearest remaining
active source inside the frozen 6-pixel radius. Additional candidates near an
already matched source are duplicate false positives; all other unmatched
candidates are background false positives. Aggregate metrics are reconciled
from integer frame-level TP, FP, and FN counts. Family-specific tables must be
read before the overall summary. Any highest-F1 ordering in the generated
report is descriptive after evaluation; it is not model selection or an
independently confirmed winner.

## Mandatory two-step execution

Freeze the plan before generating any result:

```bash
.venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.exact_truth_benchmark freeze \
  --config examples/gamma_ls_exact_truth_benchmark_v1.example.json \
  --artifact-dir Outputs/GammaLSDifference/gamma_ls_exact_truth_framewise_v1_plan_20260908
```

Then execute only that immutable plan into a fresh, non-colliding directory:

```bash
.venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.exact_truth_benchmark run \
  --config examples/gamma_ls_exact_truth_benchmark_v1.example.json \
  --plan-dir Outputs/GammaLSDifference/gamma_ls_exact_truth_framewise_v1_plan_20260908 \
  --artifact-dir Outputs/GammaLSDifference/gamma_ls_exact_truth_framewise_v1_results_20260908 \
  --device cpu
```

The plan hashes the config, this protocol, and every implementation dependency.
Any change after freezing fails closed. Both plan and result directories must be
new; partial failures remain visibly non-promotable in a work directory.

## Required interpretation artifacts

The result contains exact source truth, the complete candidate ledger,
frame-level metrics, seed-, family-, and overall-level summaries, empirical
thresholds, scale floors, score hashes, runtime diagnostics, and a separate
latency table. Latency is measured on the isolated CPU fixture and is not used
to select a quality winner. It excludes preprocessing and NMS, and therefore
is an amortized 95-output-frame batch-throughput diagnostic rather than
single-frame response latency. It cannot support a GPU, streaming, or 1-kHz
claim.

The metric runner deliberately leaves the scientific audit pending. Projection
media, exact-truth/candidate overlays, artifact-inventory rendering, and final
paper promotion require a separate audited renderer and must not be inferred
from a successful metric run.
