# VISIONS

This document records ambitious project directions for the NeuRev grid-dynamics
work. It is meant to be reused by future agents and by the project owner when
choosing what to build while long GPU experiments are running.

## Project North Star

The project should become a defensible experimental workbench for calcium-video
state forecasting. The goal is not only to train models that achieve lower MSE,
but to explain where, when, and why a model improves over simple persistence and
biologically motivated temporal baselines.

The strongest final result would show:

- A robust 128x128 max-pooled representation derived from cropped 512x512
  calcium videos.
- Forecasting horizons tied to the 50 Hz acquisition rate and plausible calcium
  reaction timescales.
- Models that beat persistence, moving averages, and calcium-kinetics baselines
  on held-out videos.
- Error analysis that shows improvement on meaningful active regions, not only
  background pixels.
- Visual evidence that is easy for a professor, peer, or committee member to
  inspect without reading code.

## Current Experimental Thread

The current main experiment is the 128x128 max-pooled grid-dynamics sweep:

- Source videos: `Inputs/060126`.
- Crop source: existing crop512 registration artifacts under
  `Outputs/GridModel/060126_crop512_grid32_v1`.
- Current output root:
  `Outputs/GridModel/060126_crop512_grid128_max_v1`.
- Dataset keys:
  - `w8_s1_h2`: 8-frame window, stride 1, 2-frame forecast horizon.
  - `w8_s1_h5`: 8-frame window, stride 1, 5-frame forecast horizon.
- Sampling rate: 50 Hz, so each raw frame is 20 ms.
- Horizons:
  - `h2`: about 40 ms.
  - `h5`: about 100 ms.
- Autoencoder:
  `models/autoencoder128_s1_ld64_bc16_e60_lr0p0010_v1`.
- Sweep profile:
  `grid128_sequence_1day`.
- Intended sweep size: 972 experiment specs.
- Last Stage A recovery runtime setting: `batch_size=2` before the sweep stopped at `477 / 972`.

Stopped run status as of 2026-06-13 00:39 EDT:

- Former PID: `2235445` (no longer running).
- Latest `sweep_active.json` marker: index `477 / 972`,
  `g128_convgru_w8_s1_h2_residual_mse_hc64_l1_lr3em04_rs0p0500_e50_s13`, status `completed`.
- Stopped-run review:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_stage_a_stop_review_v1/stage_a_stop_review.md`.
  It recommends the refreshed Stage B manifest as the default next GPU job, with
  Stage A resume from index `478` only as an explicit alternative.
- Stage B launch readiness:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_stage_b_launch_readiness_v1/stage_b_launch_readiness.md`.
  It was generated at `2026-06-13T05:51:26.697392+00:00` and records that the Stage B manifest is
  ready pending explicit user approval, with 57 planned experiments, a
  matching 57-experiment dry run, and a pre-launch checklist.
- The run is a patched-code batch-size-2 retry after a batch-size-4 ConvGRU
  hidden-channel-64 OOM cluster and a stale batch-size-2 metric-export
  `NameError`.
- Several retried h64 configs have now completed successfully. The best current
  learned/global test row is index `456`,
  `g128_temporal_cnn_w8_s1_h2_residual_mse_hc64_l6_lr1em04_rs0p1000_e50_s7`, with
  test decoded MSE `0.0011072`, persistence MSE `0.0020035`, and improvement
  `0.0008963`.
- The best current active-cell row is index `438`,
  `g128_temporal_cnn_w8_s1_h2_motion_weighted_huber_hc32_l4_lr3em04_rs0p1000_e50_s7`,
  with test active-cell improvement about `0.001456`. The best top-activity row is index `412`,
  and the best high-change row is index `432`.
- Index `323` completed with test decoded MSE `0.0012198`, test improvement
  `0.0007837`, and active-cell improvement `0.0013224`. Index `324` completed
  with test decoded MSE `0.0011898`, test improvement `0.0008137`, and
  active-cell improvement `0.0014375`, becoming the current active-cell leader.
  Index `325` completed with test decoded MSE `0.0011599`, test improvement
  `0.0008436`, and active-cell improvement `0.0009206`; it did not replace the
  current leaders. Index `326` completed with test decoded MSE `0.0011146`, test
  improvement `0.0008889`, active-cell improvement `0.0009338`, and high-change
  improvement `0.005889`; it is now the rank-4 best-test row and rank-8
  active-cell row, but it did not replace the global or active-cell leaders.
  Index `328` completed with test decoded MSE `0.0011667`, test improvement
  `0.0008368`, active-cell improvement `0.0009096`, and high-change improvement
  `0.005177`; it did not replace the current leaders. Index `328` completed
  with test decoded MSE `0.0011179`, test improvement `0.0008856`, active-cell
  improvement `0.0009321`, and high-change improvement `0.005872`; it is now
  rank 5 in the best-test review and rank 6 by active-cell improvement. Index
  `329` completed with test decoded MSE `0.0011531`, test improvement
  `0.0008504`, active-cell improvement `0.0009379`, and high-change
  improvement `0.005268`; it entered the active-cell review as card 5 after
  lightweight prediction-example backfill. Index `330` completed with test
  decoded MSE `0.0011117`, test improvement `0.0008918`, active-cell
  improvement `0.0009547`, and high-change improvement `0.005925`; after
  CPU-only prediction-example backfill it is the learned/global leader, card 1
  in the best-test review, and card 5 in the active-cell review.
- Index `331` completed with test decoded MSE `0.0011566`, test improvement
  `0.0008469`, active-cell improvement `0.0009205`, and high-change
  improvement `0.005249`; it did not replace either current leader or enter the
  refreshed top-five visual review selections. Index `332` completed with test
  decoded MSE `0.0011133`, test improvement `0.0008902`, active-cell
  improvement `0.0009363`, and high-change improvement `0.005886`; after
  CPU-only prediction-example backfill it is the rank-2 best-test row and card
  2 in the refreshed best-test review. Indices `333`-`336` failed quickly on
  hidden-channel-64 L2 residual ConvGRU configs. Index `337` completed with
  test decoded MSE `0.0012131`, test improvement `0.0007904`, active-cell
  improvement `0.0013247`, top-activity improvement `0.0006699`, and
  high-change improvement `0.005193`; after CPU-only prediction-example
  backfill it is rank 4 by active-cell improvement and card 4 in the refreshed
  active-cell review. Index `338` completed with test decoded MSE `0.0011908`,
  test improvement `0.0008127`, active-cell improvement `0.0014241`,
  top-activity improvement `0.0006781`, and high-change improvement `0.005786`;
  after CPU-only prediction-example backfill it is rank 3 by active-cell
  improvement and card 3 in the refreshed active-cell review. Index `339`
  completed with test decoded MSE `0.0012346`, test improvement `0.0007689`,
  active-cell improvement `0.0013045`, top-activity improvement `0.0006975`,
  and high-change improvement `0.005063`; it is rank 8 by active-cell
  improvement after index `340` and rank 3 by top-activity improvement, so it
  did not enter the maintained active-cell top-five review. Index `340`
  completed with test decoded MSE `0.0011962`, test improvement `0.0008073`,
  active-cell improvement `0.0014230`, top-activity improvement `0.0007059`,
  and high-change improvement `0.005805`; after CPU-only prediction-example
  backfill it is rank 4 by active-cell improvement, rank 1 by top-activity
  improvement, and card 4 in the refreshed active-cell review. Indices
  `341`-`348` then failed quickly with CUDA OOM on h32/h64 L2 motion-weighted
  Huber ConvGRU configs. Index `349` completed with test decoded MSE
  `0.0011677`, test improvement `0.0008358`, active-cell improvement
  `0.0008964`, top-activity improvement `0.0004671`, and high-change
  improvement `0.005196`, adding a ConvLSTM pixel scout to Stage B. Index `350`
  completed with test decoded MSE `0.0011262`, test improvement `0.0008773`,
  active-cell improvement `0.0008930`, top-activity improvement `0.0004133`,
  and high-change improvement `0.005826`; it is rank 10 by global test
  improvement and rank 9 by high-change improvement, but remains outside the
  maintained visual-review selections. Index `351` completed with test decoded
  MSE `0.0011832`, test improvement `0.0008203`, active-cell improvement
  `0.0008863`, top-activity improvement `0.0004764`, and high-change
  improvement `0.005083`; it is rank 23 by global and active-cell improvement
  and does not affect the maintained review selections. Index `352` completed with test decoded MSE `0.0011313`,
  test improvement `0.0008722`, active-cell improvement `0.0008826`,
  top-activity improvement `0.0003935`, and high-change improvement
  `0.005777`; it is rank 11 by global improvement and rank 12 by high-change
  improvement, but remains outside maintained visual-review selections. Index
  `353` completed with test decoded MSE `0.0011628`, test improvement
  `0.0008407`, active-cell improvement `0.0009138`, top-activity improvement
  `0.0004876`, and high-change improvement `0.005224`; it is rank 19 by
  global, active-cell, and high-change improvement and rank 13 by top-activity
  improvement, remaining outside maintained visual-review selections. Index
  `354` completed with test decoded MSE `0.0011188`, test improvement
  `0.0008847`, active-cell improvement `0.0009166`, top-activity improvement
  `0.0004550`, and high-change improvement `0.005913`; it is rank 8 by global
  improvement and rank 3 by high-change improvement, but rank 19 by active-cell
  improvement and remains outside maintained visual-review selections. Index
  `355` completed with test decoded MSE `0.0011738`, test improvement
  `0.0008297`, active-cell improvement `0.0008668`, top-activity improvement
  `0.0004556`, and high-change improvement `0.005139`; it ranks 25 by global,
  27 by active-cell, 22 by top-activity, and 26 by high-change improvement, so
  it remains outside maintained visual-review selections. Index `356` completed
  with test decoded MSE `0.0011234`, test improvement `0.0008801`, active-cell
  improvement `0.0008803`, top-activity improvement `0.0004216`, and high-change
  improvement `0.005841`; it ranks 10 by global improvement, 27 by active-cell
  improvement, 27 by top-activity improvement, and 10 by high-change improvement,
  so it remains outside maintained visual-review selections. Index `357` completed
  with test decoded MSE `0.0011602`, test improvement `0.0008433`, active-cell
  improvement `0.0009116`, top-activity improvement `0.0004751`, and high-change
  improvement `0.005239`; it ranks 20 by global improvement and remains outside
  maintained visual-review selections. Index `358` completed with test decoded
  MSE `0.0011164`, test improvement `0.0008871`, active-cell improvement
  `0.0009235`, top-activity improvement `0.0004425`, and high-change improvement
  `0.005902`; it ranks 6 by global improvement, 16 by active-cell improvement,
  27 by top-activity improvement, and 5 by high-change improvement, but remains
  outside maintained visual-review selections. Index `359` completed with test
  decoded MSE `0.0011620`, test improvement `0.0008414`, active-cell
  improvement `0.0009009`, top-activity improvement `0.0004705`, and
  high-change improvement `0.005225`; it ranks 22 by global improvement, 25 by
  active-cell improvement, 19 by top-activity improvement, and 23 by high-change
  improvement, but remains outside maintained visual-review selections. Index
  `360` completed with test decoded MSE `0.0011178`, test improvement
  `0.0008857`, active-cell improvement `0.0009230`, top-activity improvement
  `0.0004600`, and high-change improvement `0.005886`; it ranks 8 by global
  improvement, 17 by active-cell improvement, 24 by top-activity improvement,
  and 8 by high-change improvement, but remains outside maintained visual-review
  selections. Index `361` completed with test decoded MSE `0.0012276`, test
  improvement `0.0007759`, active-cell improvement `0.0013096`, top-activity
  improvement `0.0006639`, and high-change improvement `0.005094`; it ranks 45
  by global improvement, 7 by active-cell improvement, 8 by top-activity
  improvement, and 32 by high-change improvement, but remains outside maintained
  visual-review selections. Index `362` completed with test decoded MSE
  `0.0012010`, test improvement `0.0008025`, active-cell improvement
  `0.0014266`, top-activity improvement `0.0006790`, and high-change
  improvement `0.005799`; it ranks 38 by global improvement, 3 by active-cell
  improvement, 5 by top-activity improvement, and 15 by high-change improvement,
  but remains outside maintained visual-review selections. Index `363` completed
  with test decoded MSE `0.0012368`, test improvement `0.0007667`, active-cell
  improvement `0.0012837`, top-activity improvement `0.0006742`, and high-change
  improvement `0.005052`; it ranks 49 by global improvement, 11 by active-cell
  improvement, 7 by top-activity improvement, and 36 by high-change improvement,
  but remains outside maintained visual-review selections. Index `364` completed
  with test decoded MSE `0.0012028`, test improvement `0.0008007`, active-cell
  improvement `0.0014036`, top-activity improvement `0.0007025`, and high-change
  improvement `0.005761`; it ranks 40 by global improvement, 6 by active-cell
  improvement, 3 by top-activity improvement, and 18 by high-change improvement,
  but remains outside maintained visual-review selections. Index `365` completed
  with test decoded MSE `0.0012229`, test improvement `0.0007806`, active-cell
  improvement `0.0013150`, top-activity improvement `0.0006524`, and high-change
  improvement `0.005146`; it ranks 46 by global improvement, 9 by active-cell
  improvement, 12 by top-activity improvement, and 33 by high-change improvement,
  but remains outside maintained visual-review selections. Index `366` completed with test decoded MSE `0.0012003`, test improvement `0.0008032`, active-cell improvement `0.0014207`, top-activity improvement `0.0006451`, and high-change improvement `0.005811`; it ranks 38 by global improvement, 6 by active-cell improvement, 13 by top-activity improvement, and 14 by high-change improvement, but remains outside maintained visual-review selections. Index `367` completed with test decoded MSE `0.0012325`, test improvement `0.0007710`, active-cell improvement `0.0012965`, top-activity improvement `0.0006630`, and high-change improvement `0.005068`; it ranks 50 by global improvement, 14 by active-cell improvement, 12 by top-activity improvement, and 38 by high-change improvement, but remains outside maintained visual-review selections. Index `368` completed with test decoded MSE `0.0011991`, test improvement `0.0008044`, active-cell improvement `0.0014182`, top-activity improvement `0.0006589`, and high-change improvement `0.005796`; it ranks 38 by global improvement, 7 by active-cell improvement, 13 by top-activity improvement, and 17 by high-change improvement, but remains outside maintained visual-review selections. Index `369` completed with test decoded MSE `0.0012249`, test improvement `0.0007786`, active-cell improvement `0.0013194`, top-activity improvement `0.0006235`, and high-change improvement `0.005163`; it ranks 49 by global improvement, 11 by active-cell improvement, 17 by top-activity improvement, and 34 by high-change improvement, but remains outside maintained visual-review selections. Index `370` completed with test decoded MSE `0.0011910`, test improvement `0.0008125`, active-cell improvement `0.0014146`, top-activity improvement `0.0006591`, and high-change improvement `0.005856`; it ranks 36 by global improvement and remains outside maintained visual-review selections. Index `371` completed with test decoded MSE `0.0012227`, test improvement `0.0007808`, active-cell improvement `0.0013184`, top-activity improvement `0.0006496`, and high-change improvement `0.005155`; it ranks 49 by global improvement, 13 by active-cell improvement, 16 by top-activity improvement, and 37 by high-change improvement, so it remains outside maintained visual-review selections. Index `372` completed with test decoded MSE `0.0011930`, test improvement `0.0008105`, active-cell improvement `0.0014174`, top-activity improvement `0.0006461`, and high-change improvement `0.005832`; it ranks 38 by global improvement, 8 by active-cell improvement, 17 by top-activity improvement, and 14 by high-change improvement, so it remains outside maintained visual-review selections. Index `373` completed with test decoded MSE `0.0011611`, test improvement `0.0008424`, active-cell improvement `0.0009236`, top-activity improvement `0.0005096`, and high-change improvement `0.005234`; it ranks 23 by global improvement, 28 by active-cell improvement, 22 by top-activity improvement, and 29 by high-change improvement, so it remains outside maintained visual-review selections. Index `374` completed with test decoded MSE `0.0011186`, test improvement `0.0008848`, active-cell improvement `0.0009450`, top-activity improvement `0.0004790`, and high-change improvement `0.005867`; it ranks 10 by global improvement, 22 by active-cell improvement, 29 by top-activity improvement, and 11 by high-change improvement, so it remains outside maintained visual-review selections. Index `375` completed with test decoded MSE `0.0011725`, test improvement `0.0008310`, active-cell improvement `0.0008877`, top-activity improvement `0.0004623`, and high-change improvement `0.005148`; it ranks 32 by global improvement, 43 by active-cell improvement, 38 by top-activity improvement, and 41 by high-change improvement, so it remains outside maintained visual-review selections. Index `376` completed with test decoded MSE `0.0011266`, test improvement `0.0008769`, active-cell improvement `0.0008966`, top-activity improvement `0.0003972`, and high-change improvement `0.005788`; it ranks 16 by global improvement, 41 by active-cell improvement, 49 by top-activity improvement, and 21 by high-change improvement, so it remains outside maintained visual-review selections. Index `377` completed with test decoded MSE `0.0011564`, test improvement `0.0008471`, active-cell improvement `0.0009236`, top-activity improvement `0.0004988`, and high-change improvement `0.005256`; it ranks 20 by global improvement, 29 by active-cell improvement, 24 by top-activity improvement, and 27 by high-change improvement, so it remains outside maintained visual-review selections. Later progress reached index `477 / 972` and the sweep stopped; see the stopped-run status above for current leaders and next-GPU guidance.
- The missing structured-error helper imports in `concept_tests.py` were fixed,
  a regression test was added, and focused dynamics tests passed.
- At the latest liveness check, `sweep-live-status` reported
  `active_status_completed`; PID `2235445` is no longer running. The latest completed
  record is index `477`, and `sweep_active.json` also shows index `477` completed.
  Progress is `477 / 972` with `1110` progress records: `642` skipped, `439`
  completed, and `29` failed.
- Future monitoring should use `sweep_active.json`, `sweep_progress.jsonl`, the
  process table, and GPU utilization together. Raw progress counts include
  duplicate skip rows after resume.

## Guiding Principles

1. Always compare to persistence.
   A learned predictor is not useful unless it beats split-aware persistence or
   the report clearly states where it does not.

2. Prefer evidence that survives inspection.
   Rankings should be backed by per-video metrics, visual clips, and failure
   analysis.

3. Treat failures as data.
   OOMs, unstable losses, poor persistence improvements, and slow runs should be
   summarized by architecture and hyperparameter group.

4. Make runs resumable.
   Long sweeps must write progress incrementally and should skip completed metric
   files when restarted.

5. Separate biological claims from engineering claims.
   The code may evaluate 40 ms and 100 ms horizons, but reports should say these
   are operational horizons motivated by 50 Hz sampling and plausible 10-30 Hz
   calcium-timescale interpretations.

6. Prefer reusable utilities over one-off notebooks.
   If a script will be useful for another sweep, put it in `tools/` or
   `neurobench/dynamics/` and cover the risky parts with tests.

## Ambitious Utility Goals

### 1. Sweep Supervisor

Build a process-level supervisor for long dynamics sweeps.

Responsibilities:

- Watch `sweep_progress.jsonl`.
- Detect repeated failure modes:
  - CUDA OOM.
  - missing artifacts.
  - invalid shape.
  - NaN or exploding loss.
  - stalled process with no progress for a configured timeout.
- Summarize failure rates by:
  - model family.
  - architecture.
  - hidden size.
  - batch size.
  - dataset horizon.
  - seed.
- Automatically recommend or launch recovery actions:
  - lower batch size.
  - skip known-too-large configurations.
  - retry on CPU only for small baselines.
  - archive failed progress logs before a clean resume.
- Write a human-readable health report:
  `sweep_health_report.md`.

Acceptance criteria:

- Given a progress file containing many OOMs, it reports the first failing index,
  failure count, affected families, and suggested batch size.
- It never deletes completed metrics.
- It can create a new detached resume script with a safer batch size.

Initial implementation:

- `neurobench.dynamics.supervisor` provides progress parsing, failure
  classification, archived OOM analysis, health-report rendering, and detached
  resume-script generation.
- CLI entrypoint:
  `python -m neurobench.cli.main dynamics sweep-health --sweep-dir <sweep_dir>`.
- Current stopped-sweep health report:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1/sweep_health_report.md`.
- `sweep-live-status` now writes a compact live report that merges active spec,
  progress tail, process/GPU evidence, and metrics-artifact detection into
  `sweep_live_status.md`; it distinguishes high-GPU `active_training` from
  CPU-active low-GPU phases that still own GPU memory.

### 2. Results Intelligence Dashboard

Extend the comparison dashboard beyond a ranked table.

New views:

- Best model per family.
- Best model per horizon.
- Best `delta` vs `absolute` target comparison.
- Improvement distribution over persistence.
- Failure-rate heatmap.
- Runtime and memory estimates when available.
- Top five and bottom five videos per model.
- Per-family leaderboard:
  - array baselines.
  - linear latent baselines.
  - latent GRU.
  - latent Transformer.
  - ConvGRU pixel.
  - ConvLSTM pixel.
  - temporal CNN pixel.

Acceptance criteria:

- A reviewer can answer which family is best without reading TSVs.
- Hyperparameters are visible in the UI.
- Failed configurations are represented, not silently hidden.

Initial implementation:

- `neurobench.dynamics.comparison` now computes a `Results Intelligence`
  payload while building `comparison_dashboard.html`.
- Outputs:
  - `comparison_manifest.json` includes an `intelligence` object.
  - `results_intelligence.json` stores machine-readable leaderboards, grouped
    summaries, improvement distributions, and failure records.
  - `results_intelligence.md` provides a compact meeting-readable report.
- The static dashboard exposes top models, family winners, horizon winners,
  delta-vs-absolute target comparisons, and a failure heatmap with archived
  failed configurations.
- Future metric rows now support the requested top-five and bottom-five video
  evidence per selected model: prediction metric writers attach
  `split_metrics.<split>.per_video`, comparison rows summarize those diagnostics
  as `video_error_summary`, and the selected-model dashboard panel renders them.
  The dashboard and report now also aggregate per-video metrics by inferred
  video label (`left`, `right`, `neutral`, or `unknown`) so label-specific
  evidence is available once new/restarted runs write the richer metrics.
  `dynamics backfill-concept-examples --backfill-metrics` can now selectively
  recompute full split, structured, and per-video diagnostics from an existing
  spatial pixel checkpoint, giving high-value completed rows a CPU-side path to
  `video_error_summary` without retraining. Use `--dry-run` first to validate
  checkpoint/dataset metadata, estimate array size, preview selected example
  indices, report batch-count estimates for the requested batch size, show
  intended write/update targets, report split window/video counts, show label
  counts, and show top videos without writing artifacts; add `--json` to emit
  the same preflight as machine-readable JSON, or `--markdown-out <path>` to
  write a reusable Markdown note from the same summary. Dry-run checkpoint
  loading is CPU-only even if
  `--device cuda` is supplied.
  The refreshed current active-cell leader dry run for index `324` reported
  `17,565` `w8_s1_h2` windows,
  split windows `test=4841`, `train=8034`, `val=4690`, `1` example batch and
  `1098` metric batches at batch size `16`, intended write/update targets
  `prediction_examples.json`, `prediction_examples_backfill.json`, and
  `concept_metrics.json`, split videos `test=3`, `train=5`,
  `val=3`, test labels `neutral=1682`, `left=1583`, `right=1576`, train
  labels `neutral=3252`, `left=3183`, `right=1599`, val labels
  `left=1599`, `right=1575`, `neutral=1516`, top test videos `7 rest=1682`,
  `8 left=1583`, `5 right=1576`, about `10.737 GiB` of array payload, and
  example preview `11125:test:5 right, 11126:test:5 right, 11127:test:5 right`,
  so full metric backfill should only run when
  CPU/RAM headroom is intentional on grid128 datasets. The validated JSON
  preflight artifact is `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_backfill_preflight_v1/current_active_cell_leader_metric_backfill_preflight.json`; the
  human-readable summary, now regenerated through `--markdown-out`, is `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_backfill_preflight_v1/current_active_cell_leader_metric_backfill_preflight.md`. Existing completed rows
  have not all been retroactively backfilled, so the current regenerated
  manifest still has `0` rows with video summaries until selected checkpoints
  are backfilled or new runs write the richer metrics.
- Runtime summaries are now implemented when progress logs provide
  `elapsed_seconds`. The dashboard intelligence stores timed-row count,
  total/median/max runtime, per-family runtime summaries, and slowest completed
  rows. The current refreshed artifacts have `310` timed progress rows, median runtime
  about `163.6` seconds, max runtime about `3823.9` seconds, and total timed
  runtime about `96235.6` seconds.
- Current partial dashboard:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/comparison_grid128_sequence_1day_v1/comparison_dashboard.html`.

### 3. Video Error Review Tool

Generate inspection clips for selected model runs.

Each clip should show:

- Target future frame.
- Model prediction.
- Persistence prediction.
- Absolute model error.
- Absolute persistence error.
- Difference map: model error minus persistence error.
- Optional motion/activity overlay.

Selection modes:

- Best validation model.
- Best test model.
- Best per family.
- Best active-cell improvement.
- Worst over persistence.
- Most improved video.
- Least improved video.
- Held-out-first representative video.

Acceptance criteria:

- A single HTML page can compare 3-5 selected models on the same input clip.
- Clips are aligned to forecast horizon and include metadata for horizon, split,
  video ID, and improvement score.

Initial implementation:

- `neurobench.dynamics.video_review` builds a static HTML review from saved
  `prediction_examples.json` artifacts and now prefers
  `prediction_clip_examples.json` temporal clips when a selected run provides
  them.
- CLI entrypoint:
  `python -m neurobench.cli.main dynamics review-video-errors --comparison-dir <comparison_dir> --out-dir <review_dir>`.
- Each model card renders a six-panel PNG: target future frame, model
  prediction, persistence prediction, model absolute error, persistence absolute
  error, and model-minus-persistence error.
- Selection modes include `best_by_family`, `best_test`, `best_val`,
  `best_active_cell`, `most_improved_video`, `least_improved_video`,
  `heldout_first`, and `worst_over_persistence`; hyperparameters and the
  selected split score are shown in the HTML. The per-video modes use
  `video_error_summary` best/worst video diagnostics when completed metric rows
  provide `split_metrics.<split>.per_video`; if saved examples or clips contain
  the selected video ID, the visual card is aligned to that video instead of the
  generic `--example-index` sample. `heldout_first` ranks rows by test
  improvement but displays the first saved artifact from the requested split
  when split metadata is available.
- Future dataset builds now persist `window_start_indices`,
  `window_end_indices`, and `target_frame_indices` in `arrays.npz`, allowing
  downstream tools to map each sample back to absolute frame positions within a
  video.
- Future latent GRU, Transformer, and linear latent `prediction_examples.json`
  files now store video ID, split, horizon, frame-rate, and windowing metadata
  when available.
- New shared-GRU and shared-Transformer runs additionally write per-horizon
  `prediction_clip_examples.json` files and review-ready
  `per_horizon_metrics_for_review.json` files. These group adjacent examples by video
  and split, with each frame carrying input-last, target-next, predicted-next,
  persistence-next, and decoded error summaries. The video review renders these
  as temporal six-panel strips and falls back to single-frame panels for older
  artifacts. Existing completed runs are not retroactively upgraded for temporal
  clips, but selected spatial pixel checkpoints can now be backfilled with
  single-frame examples plus full split/per-video diagnostics using
  `dynamics backfill-concept-examples --backfill-metrics`.
- Current real best-test review:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/reviews/grid128_video_error_review_best_test_v1/video_error_review.html`.
- Current real active-cell review:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/reviews/grid128_active_cell_review_v1/video_error_review.html`.
- Limitation: already completed learned runs only stored representative frames
  without video IDs or split labels, and did not persist full prediction clips.
  The current page is therefore a frame-level review until future runs persist
  richer prediction sequences. The latest best-test and active-cell reviews use
  backfilled ConvGRU prediction examples and have zero missing visuals among the
  requested top five rows. Future pixel concept runs now write
  `prediction_examples.json`, but the stopped Stage A rows predate
  that code change; refreshed reviews can show Stage B or explicitly resumed
  ConvGRU rows once those artifacts exist. The comparison manifest now carries explicit
  `prediction_examples_path` and `prediction_clip_examples_path` values from
  metrics when present, and the review utility uses those paths before falling
  back to sibling files next to `metrics_path`. Review generation removes stale
  generated `model_*_example_*.png` and `model_*_clip_*.png` files before
  writing current panels so rerun directories do not retain old selections.

### 4. Per-Region and Per-Time Error Analysis

Global MSE can hide whether the model improves biologically meaningful signal.
Add analysis utilities that break error down by spatial and temporal structure.

Metrics:

- Active-cell MSE.
- Inactive-cell MSE.
- Top-percentile activity MSE.
- Error during high-motion/high-change windows.
- Error by video label: left, right, neutral.
- Error by held-out video.
- Error by time within video.
- Persistence improvement on active cells only.

Acceptance criteria:

- A model that only improves background should be flagged.
- Reports should state whether improvement is concentrated in active grid cells.

Initial implementation:

- `neurobench.dynamics.error_analysis` computes split-aware structured error
  diagnostics for active cells, inactive cells, top-activity cells, and
  high-change cells.
- Metric writers for latent RNN, latent Transformer, linear latent baselines,
  pixel concept models, array baselines, and kinetics baselines now attach
  `structured_error_metrics` plus promoted fields such as
  `test_active_cell_improvement_over_persistence_mse`.
- The partial report includes an `Active-Cell Error Check` section when included
  metric rows contain structured diagnostics.
- Current regenerated kinetics metrics populate this section. Existing learned
  rows from the stopped Stage A process do not retroactively gain structured
  diagnostics because full prediction arrays were not stored; Stage B or
  explicitly resumed runs through the patched code path will include them.

### 5. Kinetics-Aware Baselines

Add baselines that encode calcium-signal assumptions.

Candidate baselines:

- Exponential decay predictor:
  `x_hat[t+h] = baseline + alpha^h * (x[t] - baseline)`.
- Per-cell autoregressive AR(1) baseline.
- Low-pass temporal predictor using timescales near 10 Hz and 30 Hz.
- Moving average with biologically motivated window sizes.
- Linear extrapolation with decay clipping.

Why this matters:

- Persistence is strong but naive.
- A kinetics-aware baseline is harder to beat and easier to justify to a
  professor.
- If GRU/RNN models outperform these, the case for learned temporal state is
  stronger.

Acceptance criteria:

- Baselines are evaluated through the same split-aware metrics as learned
  models.
- Reports include horizon-specific comparisons.

Initial implementation:

- `neurobench.dynamics.baselines` supports kinetics-aware pixel predictors:
  `exponential_decay_10hz`, `exponential_decay_30hz`, `lowpass_10hz`,
  `lowpass_30hz`, and `ar1_per_cell`.
- `neurobench.dynamics.kinetics_baselines` writes sweep-compatible
  `array_baseline_metrics.json` files with split-aware persistence-improvement
  metrics, frame-rate metadata, forecast-horizon metadata, and reaction-rate
  metadata.
- CLI entrypoint:
  `python -m neurobench.cli.main dynamics evaluate-kinetics-baselines --dataset <dataset_json> --out-dir <out_dir>`.
- Current real 128x128 kinetics baseline sweep:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_kinetics_baselines_v1`.
- The comparison dashboard has been regenerated with this sweep included, so
  `results_intelligence.json` now includes the `kinetics_baseline` family.

### 6. Adaptive Hyperparameter Planner

The first 972-run grid should not be the final search strategy. Build a planner
that reads partial results and proposes the next sweep.

Inputs:

- Current `sweep_manifest.json`.
- Current `sweep_summary.tsv`.
- Current `sweep_progress.jsonl`.
- Failure archives.

Outputs:

- `next_sweep_plan.md`.
- `next_sweep_manifest.json`.
- Suggested command to run.

Decision rules:

- Drop families that do not beat persistence.
- Shrink learning-rate ranges around successful values.
- Reduce batch size for heavy models.
- Expand promising hidden sizes or layer counts.
- Keep at least one baseline and one conservative GRU in every stage.

Acceptance criteria:

- The planner can produce a smaller second-stage search from a partial first
  stage.
- It preserves enough diversity to avoid overfitting to one lucky run.

Initial implementation:

- `neurobench.dynamics.planner` builds an adaptive second-stage planning
  manifest from `sweep_manifest.json`, `sweep_progress.jsonl`, archived
  progress logs, and `results_intelligence.json`.
- CLI entrypoint:
  `python -m neurobench.cli.main dynamics plan-next-sweep --sweep-dir <sweep_dir> --comparison-dir <comparison_dir> --out-dir <plan_dir>`.
- Current real Stage B plan:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_sequence_stage_b_v1/next_sweep_plan.md`.
- Current plan size: `56` selected specs from the `972`-spec source sweep; latest source progress at regeneration was `371 / 972`.
- Selected families: persistence/moving-average controls, linear latent delta
  controls, conservative delta GRUs, conservative delta Transformers, one
  current-best ConvGRU pixel scout, and one ConvLSTM pixel scout.
- Deferred families: broad ConvGRU pixel, ConvLSTM pixel, and temporal CNN
  pixel grids remain deferred because archived Stage A evidence is dominated by
  CUDA OOMs; the single ConvGRU selection is a scout retained from positive
  completed evidence.
- `neurobench.dynamics.overnight_sweep` now accepts `--manifest`, so the
  generated `next_sweep_manifest.json` is directly executable through the same
  progress, resume, metric, and summary paths as profile-based sweeps. A real
  dry-run of the refreshed Stage B manifest produced
  `plans/grid128_sequence_stage_b_v1/stage_b_sweep/sweep_manifest.json` with
  `57` experiments at source progress `477 / 972`. The refreshed plan was
  generated at `2026-06-13T04:39:55.073420+00:00`, and the dry-run manifest
  was regenerated after that plan.

### 7. Multi-Horizon Forecasting

Train one model to predict multiple horizons at once.

Motivation:

- The experiment already uses `h2` and `h5`.
- Multi-horizon training may produce a better temporal representation.
- It directly connects to the 50 Hz sampling-rate discussion.

Model options:

- Latent GRU with multiple output heads.
- Temporal CNN with horizon-specific heads.
- Shared encoder with horizon embedding.
- Sequence-to-sequence latent predictor.

Metrics:

- Per-horizon MSE.
- Per-horizon persistence improvement.
- Cross-horizon consistency.
- Short-horizon vs longer-horizon degradation.

Acceptance criteria:

- One run reports both `h2` and `h5` metrics.
- The report compares against separately trained single-horizon models.

Initial implementation:

- `neurobench.dynamics.multi_horizon` compares completed single-horizon rows
  that share the same architecture and hyperparameters across `w8_s1_h2` and
  `w8_s1_h5`.
- CLI entrypoint:
  `python -m neurobench.cli.main dynamics compare-horizons --comparison-dir <comparison_dir> --out-dir <report_dir>`.
- Outputs include `multi_horizon_report.md`, `multi_horizon_report.json`, and
  `multi_horizon_plan_manifest.json` with recommended shared-horizon candidate
  configs.
- Current real grid128 report:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/reports/grid128_multi_horizon_comparison_v1/multi_horizon_report.md`.
- Current evidence from the refreshed partial sweep: `83` paired h2/h5-style
  groups were found from `342` comparison rows. The strongest learned paired
  candidate remains a delta latent Transformer (`md=64`, `heads=2`, `layers=2`,
  `lr=1e-4`) with minimum test improvement about `0.000120` across the paired
  horizons. The new h64 ConvGRU index `309` is h2-only so far, so it strengthens
  active-cell evidence but does not yet form an h2/h5 paired group.

First true shared-horizon baseline:

- `neurobench.dynamics.multi_horizon_linear` fits one horizon-conditioned
  ridge model across multiple latent-window datasets. The shared features are
  latent-window features plus normalized-horizon interactions, so one fitted
  model emits per-horizon metrics for `h2` and `h5`.
- CLI entrypoint:
  `python -m neurobench.cli.main dynamics evaluate-shared-linear-horizons --dataset <h2_json> --dataset <h5_json> --autoencoder-run <autoencoder_run> --out-dir <out_dir>`.
- Current real grid128 shared-linear artifact:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_multi_horizon_linear_h2_h5_v1/multi_horizon_linear_metrics.json`.
- Current real result: overall decoded MSE `0.001527` vs persistence MSE
  `0.002065`, improvement `0.000537` over `35097` windows. Test improvement is
  modest but positive on both horizons: about `0.0000973` for `h2` and
  `0.0001067` for `h5`.
- Important caveat: this shared linear model is worse than persistence on test
  active-cell and top-activity masks, even though it improves high-change masks.
  Treat it as a useful shared-horizon control, not as the final biological
  result.

Neural shared-horizon implementation path:

- `neurobench.dynamics.multi_horizon_neural` now provides a true shared latent
  GRU trainer. It uses one recurrent encoder over standardized latent windows
  and a normalized-horizon-conditioned output head, so one checkpoint can emit
  per-horizon metrics for `h2` and `h5`.
- CLI entrypoint:
  `python -m neurobench.cli.main dynamics train-shared-gru-horizons --dataset <h2_json> --dataset <h5_json> --autoencoder-run <autoencoder_run> --out-dir <out_dir>`.
- Current verification: focused synthetic tests and a tiny CLI smoke run pass,
  proving the encode-train-decode-metrics path, per-horizon prediction examples,
  and checkpoint writing.
- Current real grid128 shared-GRU artifact:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_multi_horizon_gru_h2_h5_v1/multi_horizon_gru_metrics.json`.
- Current real result: overall decoded MSE `0.001660` vs persistence MSE
  `0.002065`, improvement `0.000404` over `35097` windows. Test improvement is
  mixed: about `-0.0000534` for `h2` and `0.0000315` for `h5`.
- Important caveat: the shared GRU is worse than persistence on test active-cell
  masks for both horizons, while it strongly improves high-change masks. Treat
  this as evidence that the neural shared-horizon path works end to end, not as
  the final biological-region result.
- Operational improvement: shared-GRU training now writes
  `multi_horizon_gru_progress.jsonl` plus
  `multi_horizon_gru_progress_latest.json`, and the CLI prints heartbeat lines
  by default. This closes the first silent-run gap observed during the 17-minute
  CPU baseline.
- Memory improvement: shared-GRU decoded evaluation now streams decoded
  predictions in chunks (`decoded_evaluation_mode=chunked`) and exposes
  `--evaluation-batch-size`, avoiding the previous full decoded prediction
  array during metric export.
- Temporal review improvement: new shared-GRU and shared-Transformer runs now
  write per-horizon `prediction_clip_examples.json` artifacts alongside
  `prediction_examples.json`, using stored frame indices when present and
  inferred window positions for older compatible datasets.
- Shared Transformer implementation: `train-shared-transformer-horizons` now
  trains a horizon-conditioned latent Transformer with the same progress,
  chunked-evaluation, per-horizon metric path, and clip-example export as the
  shared GRU.
- Shared temporal review input is now reproducible with
  `build-shared-horizon-review-input`, which writes a review-compatible
  `comparison_manifest.json` from shared-GRU/Transformer metrics before
  `review-video-errors` builds the temporal clip page. The current shared review
  has `8` selected temporal-clip models and zero missing visuals.
- Follow-up grid planning: `plan-shared-horizon-neural-grid` now writes an
  inspectable shared-horizon neural plan with executable shared-GRU and
  shared-Transformer commands.
- Current real grid128 follow-up plan:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/shared_horizon_neural_grid_v1/shared_horizon_neural_grid_plan.md`.
- Current plan size: `16` executable CPU-safe shared-GRU configs plus `4`
  executable shared-Transformer configs.
- First planned shared-GRU grid entry completed:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shgru_h2_h5_delta_hd32_l1_lr3em4_s7/multi_horizon_gru_metrics.json`.
  It achieved overall decoded MSE `0.001612` vs persistence `0.002065`, an
  improvement of `0.000453`; test improvement is positive for both h2 and h5,
  high-change regions improve strongly, and active-cell regions remain worse
  than persistence.
- Seed-13 planned shared-GRU companion completed:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shgru_h2_h5_delta_hd32_l1_lr3em4_s13/multi_horizon_gru_metrics.json`.
  It achieved overall decoded MSE `0.001563` vs persistence `0.002065`, an
  improvement of `0.000502`; h2/h5 test improvement improved over the seed-7
  GRU, high-change regions remain positive, and active-cell regions remain
  worse than persistence.
- First shared-Transformer rescue candidate completed:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shxfmr_h2_h5_delta_md64_h2_l1_lr1em4_s7/multi_horizon_transformer_metrics.json`.
  It achieved overall decoded MSE `0.001592` vs persistence `0.002065`, an
  improvement of `0.000472`; h2 and h5 test improvement are both positive, and
  h5 now beats the shared-linear control on horizon-specific test improvement.
  Active-cell regions still remain worse than persistence, so this is a global
  and high-change improvement, not yet a biological active-region win.
- Second shared-Transformer rescue candidate completed:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shxfmr_h2_h5_delta_md64_h4_l1_lr3em4_s7/multi_horizon_transformer_metrics.json`.
  It achieved overall decoded MSE `0.001573` vs persistence `0.002065`, an
  improvement of `0.000491`, making it the best completed shared-neural run by
  global MSE. However, h2 test improvement is negative, h5 test improvement is
  only slightly positive, and active-cell performance is worse than the earlier
  Transformer on both horizons. Treat it as useful architecture evidence, not
  as an active-region rescue.
- First real clip-enabled review from the planned shared-neural grid:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/reviews/shared_horizon_neural_clip_review_v1/video_error_review.html`.
  The review now contains eight temporal clip panels: planned GRU seed-7 h2/h5,
  planned GRU seed-13 h2/h5, first rescue Transformer h2/h5, and second rescue
  Transformer h2/h5.
- Shared-horizon baseline comparison report:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_baseline_comparison_v1/shared_horizon_baseline_comparison.md`.
  Current result: the shared linear control remains best overall and still wins
  h2 test improvement, while the first rescue Transformer still wins h5 test
  improvement. The seed-13 planned GRU improves over the seed-7 GRU globally
  and on both horizon test splits, but active-cell metrics remain negative. The
  second rescue Transformer ranks second overall but regresses on active-cell
  metrics. All compared shared runs remain negative on test active-cell
  improvement.
- Active-cell rescue plan:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/active_cell_rescue_v1/active_cell_rescue_plan.md`.
  It now recommends
  `shxfmr_h2_h5_delta_md64_h2_l2_lr1em4_s7` as the next architecture-diverse
  test after the second Transformer rescue run completed. The latest refresh at
  `2026-06-11T01:45:17Z` reports `12` active-cell warnings, `4` completed
  shared-neural entries, and `16` pending entries. It explicitly discourages
  launching another same-objective GRU only to chase global MSE and now includes
  the exact next-candidate CPU command in the Markdown artifact.
- Status reporting: `status-shared-horizon-neural-grid` now writes a Markdown
  and JSON summary of pending, started, incomplete, and completed planned runs,
  including best completed metrics plus per-horizon test active-cell and
  high-change improvement diagnostics when available. The latest refresh at
  `2026-06-11T01:45:04Z` still reports `4` completed and `16` pending entries;
  every completed shared-neural entry remains `0/2` on positive test active-cell
  horizons.
- Current real status report:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/shared_horizon_neural_grid_v1/shared_horizon_neural_grid_status.md`.
- Still open: investigate the current ConvGRU OOM cluster, then run additional
  shared-GRU or shared-Transformer entries only when CPU/RAM resources are
  intentionally allocated.

### 8. Latent State Interpretation Utilities

The model should not remain a black box if latent states can be interpreted.

Possible utilities:

- Latent trajectory PCA/UMAP by video label.
- Nearest-neighbor latent state retrieval.
- Latent velocity magnitude over time.
- Correlation between latent dimensions and grid activity regions.
- Label separability from latent summary statistics.

Acceptance criteria:

- Reports show whether left/right/neutral videos occupy different latent
  regions.
- Latent trajectories can be related back to video clips.

Initial implementation:

- `neurobench.dynamics.latent_interpretation` builds JSON, Markdown, HTML, and
  PNG artifacts from an autoencoder `latent_codes.npz` file.
- CLI entrypoint:
  `python -m neurobench.cli.main dynamics interpret-latents --autoencoder-run <autoencoder_run.json> --out-dir <report_dir>`.
- The report includes frame-level PCA previews, video-level mean-latent
  embeddings, label summaries, latent-velocity summaries, leave-one-video
  nearest-centroid accuracy, nearest-video neighbors, and top label-separating
  latent dimensions.
- Current real grid128 report:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/reports/latent_interpretation_autoencoder128_v1/latent_interpretation_report.html`.
- Current evidence: the 128x128 autoencoder latent space contains `17664`
  frames across `11` videos with PCA variance ratios about `0.377`, `0.201`,
  `0.144`, and `0.078` for the first four components, but video-label
  separability is weak so far (`0.1818` leave-one-video nearest-centroid
  accuracy; between/within centroid distance ratio about `0.642`). This argues
  against overclaiming that the current autoencoder latent space cleanly encodes
  left/right/neutral state without more modeling or supervision.
- Current real latent objective follow-up plan:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/latent_objective_plan_v1/latent_objective_plan.md`.
  It diagnoses `weak_label_separability`, recommends a held-out-video supervised
  latent-head smoke test first, and sets gates for label accuracy above chance,
  between/within ratio above `1.0`, reconstruction/forecasting metrics, and
  active-cell/top-activity/high-change reporting.
- Current real latent-head smoke test:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/latent_head_smoke_v1/latent_classifier_report.md`.
  The NumPy `ridge_linear` video-level head with leave-one-video-out evaluation
  reached accuracy `0.3636`, balanced accuracy `0.3611`, and macro F1 `0.3545`
  across `11` folds. It only weakly beats chance (`0.3333`) and exactly matches
  the majority-class baseline (`0.3636`), so it is not strong evidence of
  behavior-state separation. The JSON includes machine-readable gate fields, and
  the Markdown report includes the confusion matrix plus all per-video
  predictions.

### 9. Robust Experiment Reports

Create a one-command report generator for completed sweeps.

Sections:

- Executive summary.
- Dataset and timing summary.
- Completed/failed/skipped counts.
- Best models by validation and test metrics.
- Best model per family.
- Persistence and kinetics baseline comparison.
- Failure analysis.
- Hyperparameter findings.
- Visual examples.
- Recommended next sweep.
- Artifact integrity audit.

Acceptance criteria:

- The report is suitable as a meeting handout.
- It can be regenerated as new results arrive.

Initial implementation:

- `neurobench.dynamics.report` builds JSON and Markdown reports from partial or
  completed dynamics sweeps.
- CLI entrypoint:
  `python -m neurobench.cli.main dynamics report-sweep --sweep-dir <sweep_dir> --out-dir <report_dir>`.
- The report combines sweep health, active sweep liveness, dataset timing, dashboard intelligence,
  best models, best model per family, runtime summaries when available,
  hyperparameter findings, per-video evidence when available, discovered visual
  review artifacts, artifact-audit status, recommended next-sweep and active-cell
  rescue plan artifacts, persistence/kinetics comparisons, failure analysis,
  recommendations, and artifact links.
- `neurobench.dynamics.artifact_audit` now provides a lightweight artifact
  integrity pass for the grid128 report stack. CLI entrypoint:
  `python -m neurobench.cli.main dynamics audit-grid128-artifacts --root <grid128_root> --out-dir <audit_dir>`.
- `neurobench.dynamics.launch_readiness` now provides a reusable Stage B
  launch-readiness handoff builder. CLI entrypoint:
  `python -m neurobench.cli.main dynamics stage-b-launch-readiness --root <grid128_root> --out-dir <readiness_dir>`.
  It checks key sweep-status, comparison, report, review, planning, latent, and
  preflight artifacts for existence, parses JSON artifacts, and validates referenced
  review panel files so handoffs can distinguish missing/invalid evidence from
  stale but valid evidence. Use `--fail-on-issues` when scripts should return a
  nonzero exit for missing, invalid, or missing-reference evidence.
- Current partial report:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/reports/grid128_sequence_1day_partial_report_v1/dynamics_experiment_report.md`.
  Latest report JSON generated at `2026-06-13T05:28:20.708766+00:00` (2026-06-13 00:39 EDT): `467` completed dashboard
  rows, stopped sweep progress `477 / 972`, and index `456`
  is now the best learned/global row. Temporal-CNN index `438` is now the best active-cell row, index `412` is the best top-activity row, and index `432` is the best high-change row. Index `477` completed with test decoded MSE `0.0011562`, test improvement `0.0008473`, active-cell improvement `0.0009217`, top-activity improvement `0.0004893`, and high-change improvement `0.005257`; it ranks 46 by global improvement, 107 by active-cell improvement, 93 by top-activity improvement, and 83 by high-change improvement. The sweep is stopped at index `477`.
  The `Active Sweep Liveness`, `Hyperparameter Findings`, `Runtime Summary`,
  `Visual Examples`, `Artifact Integrity Audit`, and `Recommended Next Sweep`
  sections are populated; the `Per-Video Evidence` section is present but
  currently reports that no completed rows include per-video prediction
  diagnostics yet. The recommended next-sweep section still surfaces the
  `57`-experiment Stage B manifest command and the active-cell rescue candidate
  `shxfmr_h2_h5_delta_md64_h2_l2_lr1em4_s7`.
- Current artifact audit:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_artifact_audit_v1/grid128_artifact_audit.md`.
  The latest standalone audit, refreshed at `2026-06-13T05:53:29.360323+00:00`, checked `37` key
  artifacts and reported `{ok: 37}` with valid JSON for every JSON artifact.
  It verifies current sweep live/health Markdown progress `477 / 972`, the
  partial report active-sweep summary, the report-embedded artifact-audit
  summary, Stage B plan/manifest agreement, Stage B source progress index
  `477` with `1110` records, the Stage B dry-run manifest with `57` dry-run
  experiments, the Stage A stopped-run review artifacts, and the Stage B
  launch-readiness artifacts, including consistency against current progress and
  Stage B state. The current comparison manifest has `467` metric-file references
  plus `19` prediction/example references (`486/0` missing), and review/preflight
  references remain present with `0` missing references. The refreshed Stage B
  plan was generated at `2026-06-13T04:39:55.073420+00:00`, kept `57` planned
  experiments, and the dry-run manifest was regenerated after that plan.

## Near-Term Priorities After The Sweep Stopped

Best CPU-side work to do immediately:

1. Review the stopped Stage A batch-size-2 resume before choosing the next GPU job.
   The run stopped at progress `477 / 972` after several retried h64 configs completed successfully; the h64 L2 residual configs
   at indices `333`-`336` and h32/h64 L2 motion-weighted Huber configs at
   indices `341`-`348` then failed quickly, index `354` completed as a strong
   global/high-change ConvLSTM row, index `356` completed without changing
   maintained reviews, index `357` completed without changing maintained
   reviews, the sweep later stopped cleanly at progress `477 / 972` with index `477` completed. Do not resume or launch the next GPU job until this stopped state is reviewed.
   Future overnight sweeps now write `sweep_active.json`; `sweep-health`
   renders it when present and otherwise infers the next manifest spec, so long
   single-config retries are inspectable while `run_one` is still executing.
2. Run the next active-cell rescue candidate only when CPU/RAM headroom is
   intentionally available: `shxfmr_h2_h5_delta_md64_h2_l2_lr1em4_s7`.
3. Continue the planned shared-horizon neural follow-up grid selectively:
   `14` GRU configs and `2` Transformer configs remain pending after completing
   `shgru_h2_h5_delta_hd32_l1_lr3em4_s13`.
4. Use the completed latent-head smoke-test result before a full auxiliary
   objective sweep; it weakly beats chance but does not beat the majority
   baseline, so stronger label-state evidence is still needed.
5. Keep regenerating dashboard, reports, Stage B plans, latent reports, and
   video reviews as future approved runs or CPU backfills add evidence. Use
   `dynamics backfill-concept-examples` when high-ranked checkpointed pixel rows
   need single-frame `prediction_examples.json` artifacts for visual review; add
   `--backfill-metrics` when a selected row also needs full split, structured,
   and per-video diagnostics before dashboard/report regeneration; use
   `--dry-run` first, which stays CPU-only even when `--device cuda` is supplied
   and previews selected example indices, split counts, label counts, and top
   videos; add `--json` when that preflight should be captured by a script, or
   `--markdown-out <path>` when a durable Markdown planning note is needed.
   Schedule full-dataset CPU/RAM work deliberately on grid128 arrays. Use
   `review-video-errors --selection-mode best_active_cell`
   when active-region
   evidence is the priority.

These do not require interrupting the active GPU run and will make the current
overnight sweep more useful as soon as results accumulate.
