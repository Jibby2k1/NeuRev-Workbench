# AGENTS

This file gives future coding agents operational context for the NeuRev
Workbench repository, especially the stopped Stage A 128x128 overnight grid-dynamics
sweep and the pending Stage B decision. Read this before touching sweep code, restarting runs, or interpreting
results.

## Repository Context

- Project root:
  `/home/jibby2k1/UF Dropbox/CNEL/State Analysis (Fish)/NeuRev-Workbench`
- Use the project virtual environment for Python commands:
  `.venv-neurobench/bin/python`
- Current main experiment root:
  `Outputs/GridModel/060126_crop512_grid128_max_v1`
- Stage A sweep directory:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1`
- Current comparison dashboard directory:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/comparison_grid128_sequence_1day_v1`

The workspace has repeatedly required escalated shell commands because normal
sandboxed reads/writes can fail. When using shell tools, prefer scoped commands
and avoid destructive actions unless explicitly requested.

## Current Overnight Run Status

Last updated: 2026-06-13 00:39 EDT.

The Stage A 128x128 sweep stopped after a batch-size-2 resume on
patched code. Several hidden-channel-64 ConvGRU retries completed successfully
after the batch-size-4 OOM cluster and stale-code metric-export failure, but
the later hidden-channel-64 L2 residual and motion-weighted Huber ConvGRU
configs failed quickly. The runner has completed the first four h32 ConvLSTM
residual-MSE configs plus the lr3e-4 residual-scale-0.0500 and 0.1000 h64
ConvLSTM companions, and has completed the first four h32 motion-weighted Huber ConvLSTM companions plus the first four h64 motion-weighted Huber ConvLSTM companions, and later progressed through the seed-13 temporal-CNN and ConvGRU blocks before stopping at index `477`.

- Former process PID: `2235445` (no longer running).
- Sweep active-marker file:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1/sweep_active.json`.
- Latest `sweep_active.json` marker at latest check:
  index `477 / 972`,
  `g128_convgru_w8_s1_h2_residual_mse_hc64_l1_lr3em04_rs0p0500_e50_s13`, status `completed`.
- Stopped-run review artifact:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_stage_a_stop_review_v1/stage_a_stop_review.md`.
  It records the row-477 stop state, current leaders, Stage B readiness, and the
  recommendation to use the refreshed Stage B manifest as the default next GPU
  job unless the user explicitly asks to resume Stage A from index `478`.
- Stage B launch-readiness artifact:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_stage_b_launch_readiness_v1/stage_b_launch_readiness.md`.
  It was generated at `2026-06-13T05:51:26.697392+00:00` and records the default Stage B GPU
  launch decision, required user approval, validated 57-experiment manifest and
  dry-run counts, a pre-launch checklist, and guardrails for not overwriting stopped Stage A outputs.
- Best current learned/global test row: index `456`,
  `g128_temporal_cnn_w8_s1_h2_residual_mse_hc64_l6_lr1em04_rs0p1000_e50_s7`, with
  test decoded MSE `0.0011072`, test persistence MSE `0.0020035`, and test
  improvement `0.0008963`.
- Best current active-cell row: index `438`,
  `g128_temporal_cnn_w8_s1_h2_motion_weighted_huber_hc32_l4_lr3em04_rs0p1000_e50_s7`,
  with test active-cell improvement `0.001456`; best top-activity row is index `412`
  with top-activity improvement `0.0007576`, and best high-change row is index `432`
  with high-change improvement `0.005970`.
- Index `323` completed with test decoded MSE `0.0012198`, test improvement
  `0.0007837`, and active-cell improvement `0.0013224`. Index `324` then
  completed with test decoded MSE `0.0011898`, test improvement `0.0008137`,
  and active-cell improvement `0.0014375`, becoming the current active-cell
  leader while not replacing the global learned leader. Index `325` completed
  with test decoded MSE `0.0011599`, test improvement `0.0008436`, and
  active-cell improvement `0.0009206`; it did not replace either leader. Index
  `326` completed with test decoded MSE `0.0011146`, test improvement
  `0.0008889`, active-cell improvement `0.0009338`, and high-change
  improvement `0.005889`; it is now the rank-4 best-test row and rank-8
  active-cell row, but did not replace either leader. Index `328` completed
  with test decoded MSE `0.0011667`, test improvement `0.0008368`, active-cell
  improvement `0.0009096`, and high-change improvement `0.005177`; it did not
  replace the current leaders. Index `328` completed with test decoded MSE
  `0.0011179`, test improvement `0.0008856`, active-cell improvement
  `0.0009321`, and high-change improvement `0.005872`; it is now rank 5 in
  the best-test review and rank 6 by active-cell improvement, but did not
  replace either leader. Index `329` completed with test decoded MSE
  `0.0011531`, test improvement `0.0008504`, active-cell improvement
  `0.0009379`, and high-change improvement `0.005268`; it did not replace
  either leader, but it entered the active-cell top five. Index `330` completed
  with test decoded MSE `0.0011117`, test improvement `0.0008918`, active-cell
  improvement `0.0009547`, and high-change improvement `0.005925`; it is now
  the learned/global leader and rank 5 by active-cell improvement. Index `331`
  completed with test decoded MSE `0.0011566`, test improvement `0.0008469`,
  active-cell improvement `0.0009205`, and high-change improvement
  `0.005249`; it did not replace either leader. Index `332` completed with
  test decoded MSE `0.0011133`, test improvement `0.0008902`, active-cell
  improvement `0.0009363`, and high-change improvement `0.005886`; it became
  the rank-2 global best-test row, but did not replace index `330` or enter the
  top-five active-cell selection. Indices `333`-`336` failed quickly on
  hidden-channel-64 L2 residual ConvGRU configs. Index `337` completed with
  test decoded MSE `0.0012131`, test improvement `0.0007904`, active-cell
  improvement `0.0013247`, top-activity improvement `0.0006699`, and
  high-change improvement `0.005193`; it did not affect the global best-test
  ranking but became the rank-4 active-cell row after index `338` completed.
  Index `338` completed with test decoded MSE `0.0011908`, test improvement
  `0.0008127`, active-cell improvement `0.0014241`, top-activity improvement
  `0.0006781`, and high-change improvement `0.005786`; it did not affect the
  global best-test ranking but became the rank-3 active-cell row. Index `339`
  completed with test decoded MSE `0.0012346`, test improvement `0.0007689`,
  active-cell improvement `0.0013045`, top-activity improvement `0.0006975`,
  and high-change improvement `0.005063`; it did not affect the global
  best-test ranking and is rank 8 by active-cell improvement after index `340`,
  but it remains rank 3 by top-activity improvement. Index `341` completed with
  test decoded MSE `0.0011962`, test improvement `0.0008073`, active-cell
  improvement `0.0014230`, top-activity improvement `0.0007059`, and
  high-change improvement `0.005805`; it became rank 4 by active-cell
  improvement and rank 1 by top-activity improvement. Indices `341`-`348` then
  failed quickly with CUDA OOM on h32/h64 L2 motion-weighted Huber ConvGRU
  configs. Index `349` completed with test decoded MSE `0.0011677`, test
  improvement `0.0008358`, active-cell improvement `0.0008964`, top-activity
  improvement `0.0004671`, and high-change improvement `0.005196`; it did not
  enter the maintained global or active-cell top reviews, but it caused the
  Stage B plan to add a ConvLSTM pixel scout. Index `350` completed with test
  decoded MSE `0.0011262`, test improvement `0.0008773`, active-cell
  improvement `0.0008930`, top-activity improvement `0.0004133`, and
  high-change improvement `0.005826`; it is rank 10 by global test improvement
  and rank 9 by high-change improvement, but did not enter the maintained
  global best-test or active-cell visual review selections. Index `351`
  completed with test decoded MSE `0.0011832`, test improvement `0.0008203`,
  active-cell improvement `0.0008863`, top-activity improvement `0.0004764`,
  and high-change improvement `0.005083`; it is rank 23 by global and
  active-cell improvement and does not affect the maintained review selections.
  Index `352` completed with test decoded MSE `0.0011313`, test improvement
  `0.0008722`, active-cell improvement `0.0008826`, top-activity improvement
  `0.0003935`, and high-change improvement `0.005777`; it is rank 11 by
  global improvement and rank 12 by high-change improvement, but remains
  outside maintained visual-review selections. Index `353` completed with test
  decoded MSE `0.0011628`, test improvement `0.0008407`, active-cell
  improvement `0.0009138`, top-activity improvement `0.0004876`, and
  high-change improvement `0.005224`; it is rank 19 by global, active-cell,
  and high-change improvement and rank 13 by top-activity improvement,
  remaining outside maintained visual-review selections. Index `354` completed
  with test decoded MSE `0.0011188`, test improvement `0.0008847`, active-cell
  improvement `0.0009166`, top-activity improvement `0.0004550`, and
  high-change improvement `0.005913`; it is rank 8 by global improvement and
  rank 3 by high-change improvement, but rank 19 by active-cell improvement and
  remains outside maintained visual-review selections. Index `355` completed
  with test decoded MSE `0.0011738`, test improvement `0.0008297`, active-cell
  improvement `0.0008668`, top-activity improvement `0.0004556`, and
  high-change improvement `0.005139`; it ranks 25 by global improvement, 27 by
  active-cell improvement, 22 by top-activity improvement, and 26 by
  high-change improvement, so it remains outside maintained visual-review
  selections. Index `356` completed with test decoded MSE `0.0011234`, test
  improvement `0.0008801`, active-cell improvement `0.0008803`, top-activity
  improvement `0.0004216`, and high-change improvement `0.005841`; it ranks 10
  by global improvement, 27 by active-cell improvement, 27 by top-activity
  improvement, and 10 by high-change improvement, so it improves the global
  positive-test count but remains outside maintained visual-review selections.
  Index `357` completed with test decoded MSE `0.0011602`, test improvement
  `0.0008433`, active-cell improvement `0.0009116`, top-activity improvement
  `0.0004751`, and high-change improvement `0.005239`; it ranks 20 by global
  improvement and remains outside maintained visual-review selections. Index
  `358` completed with test decoded MSE `0.0011164`, test improvement
  `0.0008871`, active-cell improvement `0.0009235`, top-activity improvement
  `0.0004425`, and high-change improvement `0.005902`; it ranks 6 by global
  improvement, 16 by active-cell improvement, 27 by top-activity improvement,
  and 5 by high-change improvement, but remains outside maintained
  visual-review selections. Index `359` completed with test decoded MSE
  `0.0011620`, test improvement `0.0008414`, active-cell improvement
  `0.0009009`, top-activity improvement `0.0004705`, and high-change
  improvement `0.005225`; it ranks 22 by global improvement, 25 by active-cell
  improvement, 19 by top-activity improvement, and 23 by high-change
  improvement, so it remains outside maintained visual-review selections. Index
  `360` completed with test decoded MSE `0.0011178`, test improvement
  `0.0008857`, active-cell improvement `0.0009230`, top-activity improvement
  `0.0004600`, and high-change improvement `0.005886`; it ranks 8 by global
  improvement, 17 by active-cell improvement, 24 by top-activity improvement,
  and 8 by high-change improvement, so it improves the positive-test count but
  remains outside maintained visual-review selections. Index `361` completed
  with test decoded MSE `0.0012276`, test improvement `0.0007759`, active-cell
  improvement `0.0013096`, top-activity improvement `0.0006639`, and high-change
  improvement `0.005094`; it ranks 45 by global improvement, 7 by active-cell
  improvement, 8 by top-activity improvement, and 32 by high-change improvement,
  so it strengthens active-cell evidence but remains outside maintained
  visual-review selections. Index `362` completed with test decoded MSE
  `0.0012010`, test improvement `0.0008025`, active-cell improvement
  `0.0014266`, top-activity improvement `0.0006790`, and high-change
  improvement `0.005799`; it ranks 38 by global improvement, 3 by active-cell
  improvement, 5 by top-activity improvement, and 15 by high-change improvement,
  so it becomes a top-three active-cell row but remains outside maintained
  visual-review selections. Index `363` completed with test decoded MSE
  `0.0012368`, test improvement `0.0007667`, active-cell improvement
  `0.0012837`, top-activity improvement `0.0006742`, and high-change
  improvement `0.005052`; it ranks 49 by global improvement, 11 by active-cell
  improvement, 7 by top-activity improvement, and 36 by high-change improvement,
  so it is positive but remains outside maintained visual-review selections.
  Index `364` completed with test decoded MSE `0.0012028`, test improvement
  `0.0008007`, active-cell improvement `0.0014036`, top-activity improvement
  `0.0007025`, and high-change improvement `0.005761`; it ranks 40 by global
  improvement, 6 by active-cell improvement, 3 by top-activity improvement,
  and 18 by high-change improvement, so it is a strong top-activity row but
  remains outside maintained visual-review selections. Indices `365`-`369` then
  completed as positive h64 motion-weighted Huber ConvLSTM rows without changing
  the maintained reviews. Index `370` completed with test decoded MSE
  `0.0011910`, test improvement `0.0008125`, active-cell improvement
  `0.0014146`, top-activity improvement `0.0006591`, and high-change
  improvement `0.005856`; it ranks 36 by global improvement and remains outside
  maintained visual-review selections while index `371` trains.
- Prediction examples were backfilled from checkpoints for the top five
  completed best-test ConvGRU rows, for index `329`, for index `330`, for
  index `332`, for index `337`, and for index `338`. The best-test visual review now renders five single-frame cards
  with zero missing visuals and includes index `332` as card 2. A dedicated active-cell visual
  review now renders the refreshed top five active-cell ConvGRU rows, including
  index `338` as card 3, index `340` as card 4, and index `337` as card 5; index `330` remains outside that top-five review. The partial report's Visual
  Examples table now lists each review's selection mode and top model IDs. The
  Active-Cell Error Check now includes a top-row leaderboard that shows the
  tradeoff between active-cell improvement and global test improvement; the
  current active-cell leader gains about `0.0004828` active-cell improvement
  over the global best learned row at about `0.0000781` lower global
  improvement.
- Active command:

```bash
.venv-neurobench/bin/python -m neurobench.dynamics.overnight_sweep \
  --profile grid128_sequence_1day \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1 \
  --device cuda \
  --epochs 50 \
  --batch-size 2 \
  --seeds 7,13 \
  --time-limit-hours 48.0
```

Why this run was relaunched:

- The original batch-size-4 runner hit a trailing ConvGRU CUDA OOM cluster on
  hidden-channel-64 configs. The batch-4 runner/child PIDs `1524371` and
  `1524373` were stopped after archiving progress.
- The first batch-size-2 retry, PID `2144778`, avoided immediate OOM but failed
  index `309` at metric export with:
  `NameError("name 'structured_prediction_error_metrics' is not defined")`.
- Root cause: `neurobench/dynamics/concept_tests.py` called
  `structured_prediction_error_metrics` and `promote_structured_error_metrics`
  without importing them from `neurobench.dynamics.error_analysis`.
- Fix: import those helpers in `concept_tests.py` and add
  `tests/test_dynamics_concept_tests.py` as a focused regression test.
- Verification after the fix:

```bash
.venv-neurobench/bin/python -m pytest \
  tests/test_dynamics_concept_tests.py \
  tests/test_dynamics_supervisor.py \
  tests/test_overnight_sweep.py \
  tests/test_dynamics_error_analysis.py
# 24 passed, 2 warnings
```

Current state at the latest liveness check:

- PID `2144778` was terminated with SIGTERM to avoid repeating the stale-code
  NameError on index `310` or later.
- PID `2235445` was started with
  `Outputs/GridModel/060126_crop512_grid128_max_v1/run_grid128_sequence_1day_resume_batch2.sh`.
- At 2026-06-13 00:39 EDT, PID `2235445` was no longer present in `ps`. `sweep-live-status` reported live state `active_status_completed`, GPU utilization about `23%`, and no process GPU memory for the former run.
- Progress file contained `1110` records: `642 skipped`, `439 completed`,
  `29 failed`. The high skip count includes duplicate resume skip records and
  should not be read as unique experiment completion.
- Live status generated with `sweep-live-status`:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1/sweep_live_status.md`.
- Health report refreshed with `--stale-minutes 15`:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1/sweep_health_report.md`.
- Health report summary at latest check: progress `477 / 972`; index
  `477` is the latest completed spec and no sweep process is running.

Important archived evidence:

- Archived batch-4 progress log:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1/sweep_progress_batch4_convgru_hc64_oom_20260610_095246.jsonl`.
- Older archived failed progress logs:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1/sweep_progress_batch64_oom.jsonl` and
  `Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1/sweep_progress_batch8_partial_oom.jsonl`.
- Do not delete archived progress logs unless the user explicitly asks; they are
  useful for failure-mode summaries and supervisor validation.

How to monitor the active run:

Prefer the compact live-status command first:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dynamics sweep-live-status --sweep-dir Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1 --pid 2235445
.venv-neurobench/bin/python -m neurobench.cli.main dynamics sweep-health --sweep-dir Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1 --stale-minutes 15
```

Manual fallback commands:

```bash
ps -o pid,ppid,stat,etime,%cpu,%mem,rss,vsz,cmd -p 2235445
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
.venv-neurobench/bin/python -m json.tool Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1/sweep_active.json
.venv-neurobench/bin/python -c "import json; from pathlib import Path; from collections import Counter; p=Path('Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1/sweep_progress.jsonl'); rows=[json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []; print('records', len(rows)); print('counts', dict(Counter(r.get('status') for r in rows))); print('last', rows[-1] if rows else None)"
.venv-neurobench/bin/python -m neurobench.cli.main dynamics sweep-health --sweep-dir Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1 --stale-minutes 15
```

Interpretation rules:

- `sweep_progress.jsonl` is append-only and can contain duplicate skip records
  after resume. Use metric-file existence and manifest index, not raw status
  counts alone, when interpreting unique progress.
- `sweep_progress.jsonl` is written only after a spec is skipped, completed, or
  failed. During a long ConvGRU trial, `sweep_active.json`, `ps`, and
  `nvidia-smi` are better liveness indicators. `sweep-live-status` reports
  `active_training` for high GPU-utilization samples and `active_training`
  when the process is CPU-active while still holding GPU memory.
- If index `309` or nearby hidden-channel-64 ConvGRU rows now fail with CUDA
  OOM, keep the fixed code and relaunch at `batch_size=1`. If they fail with a
  Python exception, inspect the target experiment directory and traceback before
  changing batch size.

## Experiment Specification

The current sweep profile is `grid128_sequence_1day`.

Expected experiment count: `972`.

Datasets:

- `w8_s1_h2`
  - Window frames: `8`.
  - Temporal stride: `1`.
  - Prediction horizon: `2` frames.
  - Approximate horizon at 50 Hz: `40 ms`.
  - Dataset path:
    `Outputs/GridModel/060126_crop512_grid128_max_v1/datasets/w8_s1_h2/dynamics_dataset.json`

- `w8_s1_h5`
  - Window frames: `8`.
  - Temporal stride: `1`.
  - Prediction horizon: `5` frames.
  - Approximate horizon at 50 Hz: `100 ms`.
  - Dataset path:
    `Outputs/GridModel/060126_crop512_grid128_max_v1/datasets/w8_s1_h5/dynamics_dataset.json`

Grid representation:

- Source videos are cropped 512x512 videos.
- Grid output is 128x128.
- Pooling/statistic feature: `max_intensity`.
- The current workflow reuses existing crop512/template/registration artifacts
  from `Outputs/GridModel/060126_crop512_grid32_v1` and regenerates grid states
  at 128x128.

Autoencoder:

- Path:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/models/autoencoder128_s1_ld64_bc16_e60_lr0p0010_v1/autoencoder_run.json`
- Latent dimension: `64`.
- Base channels: `16`.
- Epochs: `60`.
- Learning rate: `0.001`.
- Seed: `7`.

Model families in the profile:

- Array baselines.
- Linear latent baselines.
- Latent GRU.
- Latent Transformer.
- ConvGRU pixel.
- ConvLSTM pixel.
- Temporal CNN pixel.

Naming:

- Experiment IDs should start with `g128_`.
- Configs include `grid_size=128`, `grid_pooling=max_intensity`, and
  `hyperparameter_summary`.

## Important Files

Live progress:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1/sweep_progress.jsonl
```

Sweep manifest:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1/sweep_manifest.json
```

Sweep summaries:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1/sweep_summary.tsv
Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1/sweep_summary.md
```

Historical batch-size-2 resume script; do not launch it unless the user explicitly chooses to resume Stage A from index `478`:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/run_grid128_sequence_1day_resume_batch2.sh
```

Earlier batch-size-4 runner log:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/logs/grid128_sequence_1day_resume_batch4.log
```

Archived failed progress logs:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1/sweep_progress_batch64_oom.jsonl
Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1/sweep_progress_batch8_partial_oom.jsonl
```

These archives are important evidence. Do not delete them unless the user asks.

## How To Check Progress

Use this compact progress summary:

```bash
.venv-neurobench/bin/python - <<'PY'
import json
from collections import Counter
from pathlib import Path

p = Path("Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1/sweep_progress.jsonl")
rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []
counts = Counter(r.get("status") for r in rows)
print("records", len(rows))
print("counts", dict(counts))
if rows:
    last = rows[-1]
    print("last_index", last.get("index"), "of", last.get("experiment_count"))
    print("last_status", last.get("status"))
    print("last_experiment", last.get("experiment_id"))
    if last.get("status") == "failed":
        print("last_error", str(last.get("error"))[:1000])
PY
```

Check whether the process is alive:

```bash
ps -f -u jibby2k1 | rg 'resume_batch4|overnight_sweep|grid128_sequence_1day|neurobench.dynamics.comparison'
```

Check GPU memory:

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
```

Tail the live progress:

```bash
tail -f Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1/sweep_progress.jsonl
```

Generate or refresh the sweep health report:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dynamics sweep-health \
  --sweep-dir Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1
```

The report is written to:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1/sweep_health_report.md
```

To only generate a safer resume script without launching it, add:

```bash
--resume-script Outputs/GridModel/060126_crop512_grid128_max_v1/run_grid128_sequence_1day_resume_batch2.sh \
--resume-batch-size 2
```

## Failure History And Recovery Rules

Initial run:

- Batch size: `64`.
- Result: widespread CUDA OOM after baselines/linear runs.
- Archived progress:
  `sweep_progress_batch64_oom.jsonl`.

First resume:

- Batch size: `8`.
- Result: mostly better, but one GRU still hit CUDA OOM.
- Archived progress:
  `sweep_progress_batch8_partial_oom.jsonl`.

Current resume:

- Batch size: `2`.
- The first h64 ConvGRU retry at index `309` completed successfully after the
  batch-size-4 OOM cluster and stale-code metric-export failure.
- Index `324` completed successfully and became the current active-cell leader;
  index `325` completed without replacing either leader, index `326` completed as
  a high-ranking best-test and active-cell row, indices `327`-`329` have
  completed without replacing the leaders, index `330` became the current learned/global
  leader, index `331` completed without replacing either leader, index `332`
  became the rank-2 best-test row after CPU-only example backfill, indices
  `333`-`336` failed quickly, index `337` became a top-five active-cell row after CPU-only example backfill,
  index `338` became the rank-3 active-cell row after CPU-only example backfill,
  index `350` entered the global top 10 without changing visual reviews, index
  `351` completed without changing leaderboards, index `352` completed as rank
  11 by global improvement and rank 12 by high-change improvement, index `353`
  completed as rank 19 by global, active-cell, and high-change improvement,
  index `354` completed as rank 8 by global improvement and rank 3 by
  high-change improvement, index `355` completed without changing maintained
  reviews, index `356` completed as rank 10 by global and high-change
  improvement without changing maintained reviews, index `357` completed as
  rank 20 by global and high-change improvement without changing maintained
  reviews, index `359` completed without changing maintained reviews, index
  `360` completed as rank 8 by global and high-change improvement without
  changing maintained reviews, index `361` completed as rank 7 by active-cell
  improvement and rank 8 by top-activity improvement without changing
  maintained reviews, index `362` completed as rank 3 by active-cell improvement
  and rank 5 by top-activity improvement without changing maintained reviews,
  index `363` completed as rank 7 by top-activity improvement without changing
  maintained reviews, index `364` completed as rank 3 by top-activity
  improvement without changing maintained reviews, index `365` completed as rank 9 by active-cell
  improvement without changing maintained reviews, index `366` completed as rank 38 by global improvement, rank 6 by active-cell improvement, rank 13 by top-activity improvement, and rank 14 by high-change improvement without changing maintained reviews, index `367` completed as rank 50 by global improvement, rank 14 by active-cell improvement, rank 12 by top-activity improvement, and rank 38 by high-change improvement without changing maintained reviews, index `368` completed as rank 38 by global improvement, rank 7 by active-cell improvement, rank 13 by top-activity improvement, and rank 17 by high-change improvement without changing maintained reviews, and index `369` completed as rank 49 by global improvement, rank 11 by active-cell improvement, rank 17 by top-activity improvement, and rank 34 by high-change improvement without changing maintained reviews, and index `370` completed as rank 36 by global improvement without changing maintained reviews, index `371` completed as rank 49 by global improvement, rank 13 by active-cell improvement, rank 16 by top-activity improvement, and rank 37 by high-change improvement without changing maintained reviews, index `372` completed as rank 38 by global improvement, rank 8 by active-cell improvement, rank 17 by top-activity improvement, and rank 14 by high-change improvement without changing maintained reviews, and index `373` completed as rank 23 by global improvement, rank 28 by active-cell improvement, rank 22 by top-activity improvement, and rank 29 by high-change improvement without changing maintained reviews, and index `374` completed as rank 10 by global improvement, rank 22 by active-cell improvement, rank 29 by top-activity improvement, and rank 11 by high-change improvement without changing maintained reviews, and index `375` completed as rank 32 by global improvement, rank 43 by active-cell improvement, rank 38 by top-activity improvement, and rank 41 by high-change improvement without changing maintained reviews, index `376` completed as rank 16 by global improvement, rank 41 by active-cell improvement, rank 49 by top-activity improvement, and rank 21 by high-change improvement without changing maintained reviews, index `377` completed as rank 20 by global improvement, rank 29 by active-cell improvement, rank 24 by top-activity improvement, and rank 27 by high-change improvement without changing maintained reviews, and index `378` completed as rank 1 by global improvement and rank 1 by high-change improvement, replacing the then-current learned/global leader while remaining outside maintained reviews. Later progress reached index `477 / 972` and the sweep stopped; see the current stopped-run status above.
- Do not increase the batch size for this sweep unless explicitly requested.
- If new ConvGRU/ConvLSTM OOMs accumulate at batch size 2, prefer a
  batch-size-1 resume rather than discarding the current completed metrics.

If new OOMs appear:

1. Do not delete metric directories.
2. Stop only the active sweep child process, not unrelated processes.
3. Archive the current `sweep_progress.jsonl` with a descriptive name.
4. Relaunch with a smaller batch size, likely `--batch-size 2`.
5. Preserve the same `--out-dir` so completed metrics are skipped.
6. Report the exact failing experiment IDs and error text.

Example batch-2 fallback command:

```bash
.venv-neurobench/bin/python -m neurobench.dynamics.overnight_sweep \
  --profile grid128_sequence_1day \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1 \
  --device cuda \
  --epochs 50 \
  --batch-size 2 \
  --seeds 7,13 \
  --time-limit-hours 48
```

## Completion Criteria

The overnight run is complete when:

- There is no active `overnight_sweep` process for this sweep.
- The progress file has reached index `972 / 972`, or the runner reports
  `status=finished` in its terminal output/log.
- The comparison dashboard has been built:
  - `comparison_dashboard.html`
  - `comparison_summary.json`
- `sweep_summary.tsv` exists and includes completed metric rows.

The dashboard is expected to be built after the sweep command returns. It can
also be regenerated safely from partial or stopped sweep results:

```bash
.venv-neurobench/bin/python -m neurobench.dynamics.comparison \
  --sweep-dir Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1 \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/comparison_grid128_sequence_1day_v1
```

Current dashboard intelligence outputs:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/comparison_grid128_sequence_1day_v1/comparison_dashboard.html
Outputs/GridModel/060126_crop512_grid128_max_v1/comparison_grid128_sequence_1day_v1/results_intelligence.json
Outputs/GridModel/060126_crop512_grid128_max_v1/comparison_grid128_sequence_1day_v1/results_intelligence.md
```

Use `results_intelligence.md` for quick meeting review and the HTML dashboard
for inspecting hyperparameters, family winners, horizon winners, target-mode
comparisons, and archived failure patterns. Regenerating these files does not
interrupt the active GPU sweep.

Forward-compatible per-video diagnostics were added after the current completed
rows were generated. Future latent, linear, kinetics/baseline, and pixel concept
metric writers now attach `split_metrics.<split>.per_video` maps with decoded
MSE, persistence MSE, and improvement over persistence. The comparison manifest
summarizes those as `video_error_summary` with top-five and bottom-five videos
per split. The dashboard selected-model panel renders them when present and now
also summarizes inferred video labels (`left`, `right`, `neutral`, or `unknown`)
from video IDs. The refreshed current manifest has `0` rows with video summaries
because the existing completed metrics predate this schema. The robust experiment
report now includes a `Per-Video Evidence` section sourced from
`video_error_summary`; for rows with per-video evidence it also carries the same
label summary, while the current partial report records the schema-age limitation
instead of silently omitting the section.

Runtime evidence is now propagated from completed `sweep_progress.jsonl` rows
into comparison rows as `elapsed_seconds`. `results_intelligence.json` and the
HTML dashboard include a `runtime_summary` with timed-row count, total/median/max
runtime, per-family runtime summaries, and slowest completed rows. The current
refreshed dashboard/report have `310` timed progress rows, median runtime about
`163.6` seconds, max runtime about `3823.9` seconds, and total timed runtime
about `96235.6` seconds.

The robust experiment report now includes an `Active Sweep Liveness` section and a `Hyperparameter Findings` section.
It groups completed rows by model family, hyperparameter group, target/baseline,
loss mode, learning rate, hidden channels/dimensions, model dimension, layer
count, and residual scale. Current findings are based on `342` completed rows;
by mean test improvement, pixel ConvGRU is the strongest model family in the
current partial comparison, residual MSE remains the strongest pixel loss mode
among completed pixel rows, and hidden-channel `64` now includes the completed
residual and motion-weighted h64 retries.

The robust report now includes a `Visual Examples` section. It discovers
`video_error_review.json` files under the experiment root `reviews/` directory
and links their HTML pages. The current refreshed report lists `4` reviews:
shared-horizon neural temporal clips (`8` models, `8` clip-enabled), the
best-test grid128 review (`5` visual cards, zero missing visuals), the
active-cell grid128 review (`5` visual cards, zero missing visuals), and the
earlier family-level grid128 review (`2` cards). Review generation now removes
stale generated `model_*_example_*.png` and `model_*_clip_*.png` files before
writing the current panels, so rerun directories do not retain panels from old
selections.

Kinetics-aware baseline artifacts have also been generated separately so the
active 972-run sweep manifest remains stable:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_kinetics_baselines_v1
```

To regenerate them:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv-neurobench/bin/python -m neurobench.cli.main dynamics evaluate-kinetics-baselines \
  --dataset Outputs/GridModel/060126_crop512_grid128_max_v1/datasets/w8_s1_h2/dynamics_dataset.json \
  --dataset Outputs/GridModel/060126_crop512_grid128_max_v1/datasets/w8_s1_h5/dynamics_dataset.json \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_kinetics_baselines_v1
```

When rebuilding the comparison dashboard, include both sweep directories:

```bash
.venv-neurobench/bin/python -m neurobench.dynamics.comparison \
  --sweep-dir Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1 \
  --sweep-dir Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_kinetics_baselines_v1 \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/comparison_grid128_sequence_1day_v1
```

Current partial meeting report:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/grid128_sequence_1day_partial_report_v1/dynamics_experiment_report.md
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/grid128_sequence_1day_partial_report_v1/dynamics_experiment_report.json
```

Latest report JSON generated at `2026-06-13T05:28:20.708766+00:00` (2026-06-13 00:39 EDT): `467` completed dashboard rows, stopped sweep progress `477 / 972`, and index `456` is now the best current learned/global row while index `438` is the active-cell leader. Index `477` completed with test decoded MSE `0.0011562`, test improvement `0.0008473`, active-cell improvement `0.0009217`, top-activity improvement `0.0004893`, and high-change improvement `0.005257`; it ranks 46 by global improvement, 107 by active-cell improvement, 93 by top-activity improvement, and 83 by high-change improvement. The report JSON has `active_sweep_summary.available=true`, showing index `477` (`g128_convgru_w8_s1_h2_residual_mse_hc64_l1_lr3em04_rs0p0500_e50_s13`), status `completed`, progress `477 / 972`, and links to `sweep_health_report.md` and `sweep_live_status.md`. It reports `467` completed metric rows, `215` positive-test rows, and failure count `594`, with current progress counts `642` skipped, `439` completed, and `29` failed. The report-embedded artifact audit summary is clean and the Artifact Integrity table shows current consistency checks passing, including progress `477 / 972`, Stage B source progress index `477`, and comparison references `486/486` (`467` metric files plus `19` prediction/example files) with `0` missing.

To regenerate the report and refresh dashboard intelligence first:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv-neurobench/bin/python -m neurobench.cli.main dynamics report-sweep \
  --sweep-dir Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1 \
  --sweep-dir Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_kinetics_baselines_v1 \
  --comparison-dir Outputs/GridModel/060126_crop512_grid128_max_v1/comparison_grid128_sequence_1day_v1 \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/reports/grid128_sequence_1day_partial_report_v1 \
  --title "Grid128 Sequence 1-Day Partial Report" \
  --refresh-dashboard
```

To audit the current report/review/plan evidence stack after refreshing
artifacts:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dynamics audit-grid128-artifacts \
  --root Outputs/GridModel/060126_crop512_grid128_max_v1 \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_artifact_audit_v1 \
  --title "Grid128 Artifact Audit" \
  --fail-on-issues
```

Current artifact audit:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_artifact_audit_v1/grid128_artifact_audit.md
Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_artifact_audit_v1/grid128_artifact_audit.json
```

Latest standalone audit summary, refreshed at `2026-06-13T05:53:29.360323+00:00`: `37` key artifacts checked, status counts `{'ok': 37}`. The audit verifies current sweep live/health Markdown progress `477 / 972`, the partial report active-sweep summary, the report-embedded artifact-audit summary, Stage B plan/manifest agreement, Stage B source progress index `477` with `1110` progress records, the Stage B dry-run manifest with `57` dry-run experiments, the Stage A stopped-run review artifacts, and the Stage B launch-readiness artifacts, including consistency against current progress and Stage B state. Current comparison-manifest reference counts are `486/0` missing (`467` metric files and `19` prediction/example files), while best-test, active-cell, and shared-horizon review references remain present with `0` missing references. The refreshed Stage B plan was generated at `2026-06-13T04:39:55.073420+00:00` with `57` planned experiments and selection counts `{'array_baseline': 4, 'convgru_pixel': 1, 'convlstm_pixel': 1, 'latent_gru': 16, 'latent_transformer': 32, 'linear_latent': 2, 'temporal_cnn_pixel': 1}`; the dry-run manifest was regenerated after that plan and validated `57` experiments.
It covers sweep live/health reports, comparison/dashboard outputs, the partial
report, Stage B plan/manifest, best-test and active-cell reviews, shared-horizon
clip review, shared-horizon status, active-cell rescue plan, latent objective
and smoke-test artifacts, plus the current active-cell leader, learned-leader,
and active-cell challenger backfill preflights with explicit input-reference
counts. The current active-cell challenger preflight now points at index `338`
and was regenerated at `2026-06-11T04:35:33.465508+00:00`; it estimates `1098`
metric batches, about `10.737` GiB payload, split windows `test=4841`,
`train=8034`, `val=4690`, and previews examples `11125`-`11127` from test video
`5 right`.

Structured active-cell diagnostics:

- `neurobench.dynamics.error_analysis` now computes active-cell, inactive-cell,
  top-activity, and high-change error metrics.
- Regenerated kinetics baseline metrics already include these fields, so the
  current report has an `Active-Cell Error Check` section.
- Older learned rows do not retroactively include structured metrics because
  full prediction arrays were not persisted. New or resumed runs through the
  patched metric writers include promoted fields such as
  `test_active_cell_improvement_over_persistence_mse`; the completed h64 ConvGRU
  retry at index `309` is the first important resumed pixel row with these
  diagnostics in the partial report.

## Video Error Review

The video error review utility has been implemented as a static artifact
generator. It reads the comparison manifest and saved `prediction_examples.json`
or `prediction_clip_examples.json` files, then writes an HTML page plus generated
six-panel PNGs. The panels are target future frame, model prediction,
persistence prediction, model absolute error, persistence absolute error, and
model-minus-persistence error. Review generation removes stale generated
`model_*_example_*.png` and `model_*_clip_*.png` files before writing the
current panels, so rerun directories do not retain old selections.

Current real review artifacts:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/reviews/grid128_video_error_review_best_test_v1/video_error_review.html
Outputs/GridModel/060126_crop512_grid128_max_v1/reviews/grid128_video_error_review_best_test_v1/video_error_review.json
Outputs/GridModel/060126_crop512_grid128_max_v1/reviews/grid128_active_cell_review_v1/video_error_review.html
Outputs/GridModel/060126_crop512_grid128_max_v1/reviews/grid128_active_cell_review_v1/video_error_review.json
Outputs/GridModel/060126_crop512_grid128_max_v1/reviews/shared_horizon_neural_clip_review_v1/video_error_review.html
Outputs/GridModel/060126_crop512_grid128_max_v1/reviews/shared_horizon_neural_clip_review_v1/video_error_review.json
```

Current review selection at latest regeneration:

- Best-test grid128 review: mode `best_test`, split `test`, `5` visual cards,
  zero missing visuals after checkpoint backfill of the current top ConvGRU
  rows.
- Active-cell grid128 review: mode `best_active_cell`, split `test`, `5` visual
  cards, zero missing visuals, and HTML score labels show active-cell
  improvement.
- Shared-horizon neural review: `8` selected models and `8` temporal-clip models
  from the generated `comparison_input` manifest, with zero missing visuals. The
  builder writes both current clip panels and current single-frame panels for
  those clip-capable rows.
- The comparison manifest preserves explicit `prediction_examples_path` and
  `prediction_clip_examples_path` fields from metric JSON, and the review
  utility prefers those paths before falling back to files beside `metrics_path`.
- `review-video-errors` also supports `most_improved_video`,
  `least_improved_video`, and `heldout_first` selection modes. The per-video
  extremes require rows with `video_error_summary.<split>.best_videos` or
  `worst_videos`, so they are most useful after new runs write per-video
  prediction diagnostics or after a selected checkpoint is backfilled with
  `dynamics backfill-concept-examples --backfill-metrics`. Use
  `--dry-run` first on grid128 rows to validate checkpoint/dataset metadata and
  estimate array size, preview selected example indices, report batch-count
  estimates for the requested batch size, report intended write/update targets,
  and report split window/video counts, label counts, and top videos without
  writing artifacts. Add `--json` to emit the same preflight as machine-readable
  JSON for scripts, or `--markdown-out <path>` to write a reusable Markdown
  note from the same summary.
  Dry-run checkpoint loading is CPU-only even if the requested real backfill
  device is CUDA. When the saved examples or clips include the selected video
  ID, the review card uses that
  matching artifact instead of the generic `--example-index` sample.
  `heldout_first` ranks rows by test improvement but uses the first saved
  example/clip from the requested split when split metadata is available.

Regenerate the best-test and active-cell grid128 reviews with:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dynamics review-video-errors \
  --comparison-dir Outputs/GridModel/060126_crop512_grid128_max_v1/comparison_grid128_sequence_1day_v1 \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/reviews/grid128_video_error_review_best_test_v1 \
  --selection-mode best_test \
  --split test \
  --max-models 5 \
  --example-index 0 \
  --dataset-key w8_s1_h2 \
  --title "Grid128 Best-Test Video Error Review"

.venv-neurobench/bin/python -m neurobench.cli.main dynamics review-video-errors \
  --comparison-dir Outputs/GridModel/060126_crop512_grid128_max_v1/comparison_grid128_sequence_1day_v1 \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/reviews/grid128_active_cell_review_v1 \
  --selection-mode best_active_cell \
  --split test \
  --max-models 5 \
  --example-index 0 \
  --dataset-key w8_s1_h2 \
  --title "Grid128 Active-Cell Video Error Review"
```

Regenerate the shared-horizon review-local input with the reproducible builder,
then regenerate the clip review from that input instead of the main grid128
comparison manifest:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dynamics build-shared-horizon-review-input \
  --run planned_gru_hd32=Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shgru_h2_h5_delta_hd32_l1_lr3em4_s7/multi_horizon_gru_metrics.json \
  --run planned_gru_hd32_s13=Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shgru_h2_h5_delta_hd32_l1_lr3em4_s13/multi_horizon_gru_metrics.json \
  --run rescue_transformer_md64=Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shxfmr_h2_h5_delta_md64_h2_l1_lr1em4_s7/multi_horizon_transformer_metrics.json \
  --run rescue_transformer_h4_lr3e4=Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shxfmr_h2_h5_delta_md64_h4_l1_lr3em4_s7/multi_horizon_transformer_metrics.json \
  --comparison-dir Outputs/GridModel/060126_crop512_grid128_max_v1/comparison_grid128_sequence_1day_v1 \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/reviews/shared_horizon_neural_clip_review_v1/comparison_input \
  --title "Shared-Horizon Neural Review Input"

.venv-neurobench/bin/python -m neurobench.cli.main dynamics review-video-errors \
  --comparison-dir Outputs/GridModel/060126_crop512_grid128_max_v1/reviews/shared_horizon_neural_clip_review_v1/comparison_input \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/reviews/shared_horizon_neural_clip_review_v1 \
  --selection-mode best_test \
  --split test \
  --max-models 8 \
  --example-index 0 \
  --title "Shared-Horizon Neural Clip Review"
```

A family-level review was also generated here:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/reviews/grid128_video_error_review_v1/video_error_review.html
```

Limitations to preserve in any interpretation:

- Older completed runs saved representative frames but not video IDs or split
  labels in `prediction_examples.json`.
- Future dataset builds persist `window_start_indices`, `window_end_indices`,
  and `target_frame_indices` in `arrays.npz`; future example writers use those
  fields when available and infer positions for older compatible datasets.
- Future latent GRU, Transformer, and linear latent example writers now add
  video ID, split, horizon, frame-rate, and windowing metadata when available.
- New shared-GRU and shared-Transformer runs write per-horizon
  `prediction_clip_examples.json` files that group adjacent examples by video
  and split for true temporal review.
- `review-video-errors` consumes those clip artifacts, records
  `artifact_mode=temporal_clip`, writes stacked temporal PNGs for selected clip
  models, and falls back to single-frame panels for older runs.
- The current grid128 ConvGRU reviews remain frame-level because those completed
  model runs did not persist full prediction sequences; the shared-horizon
  review is temporal-clip enabled.

## Multi-Horizon Comparison

The multi-horizon comparison utility has been implemented as the CPU-side
precursor to a true shared-horizon model. It pairs completed single-horizon rows
with matching hyperparameters across `w8_s1_h2` and `w8_s1_h5`, reports per
horizon improvement, and writes a planning manifest for shared `h2+h5` candidate
configs.

Current real artifacts:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/grid128_multi_horizon_comparison_v1/multi_horizon_report.md
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/grid128_multi_horizon_comparison_v1/multi_horizon_report.json
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/grid128_multi_horizon_comparison_v1/multi_horizon_plan_manifest.json
```

Current grid128 multi-horizon findings at latest regeneration:

- Source comparison rows: `342`.
- Paired h2/h5-style groups: `83`.
- Planned shared-horizon configs: `8`.
- Best learned paired candidate remains a latent Transformer, delta target,
  `md=64`, `heads=2`, `layers=2`, `lr=1e-4`, batch `4`.
- Best learned candidate minimum test improvement across paired horizons:
  about `0.000120`.
- Kinetics baselines remain stronger than learned paired candidates on this
  partial snapshot, with best minimum improvement about `0.000612`.
- The current best ConvGRU rows through index `332` are h2-only so far, so they
  improve the partial report and active-cell evidence but do not yet form h2/h5 paired groups.

Regenerate the multi-horizon comparison with:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dynamics compare-horizons \
  --comparison-dir Outputs/GridModel/060126_crop512_grid128_max_v1/comparison_grid128_sequence_1day_v1 \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/reports/grid128_multi_horizon_comparison_v1 \
  --split test \
  --max-candidates 20 \
  --title "Grid128 Multi-Horizon Forecasting Report"
```

Interpretation rule: this report compares separately trained single-horizon
runs. Do not describe it as evidence from a true multi-output model until a
shared-horizon trainer emits both h2 and h5 metrics from one run.

## Shared Multi-Horizon Linear Baseline

A first true shared-horizon baseline has been implemented. It fits one
horizon-conditioned ridge model across the `w8_s1_h2` and `w8_s1_h5` latent
window datasets using the grid128 autoencoder, then writes one metrics file with
per-horizon results.

Implementation and CLI:

```bash
neurobench/dynamics/multi_horizon_linear.py
.venv-neurobench/bin/python -m neurobench.cli.main dynamics evaluate-shared-linear-horizons --help
```

Current real artifacts:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_multi_horizon_linear_h2_h5_v1/multi_horizon_linear_run.json
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_multi_horizon_linear_h2_h5_v1/multi_horizon_linear_metrics.json
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_multi_horizon_linear_h2_h5_v1/multi_horizon_linear_weights.npz
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_multi_horizon_linear_h2_h5_v1/w8_s1_h2/prediction_examples.png
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_multi_horizon_linear_h2_h5_v1/w8_s1_h5/prediction_examples.png
```

Run command used, CPU-only so it did not compete with the CUDA sweep:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dynamics evaluate-shared-linear-horizons \
  --dataset Outputs/GridModel/060126_crop512_grid128_max_v1/datasets/w8_s1_h2/dynamics_dataset.json \
  --dataset Outputs/GridModel/060126_crop512_grid128_max_v1/datasets/w8_s1_h5/dynamics_dataset.json \
  --autoencoder-run Outputs/GridModel/060126_crop512_grid128_max_v1/models/autoencoder128_s1_ld64_bc16_e60_lr0p0010_v1/autoencoder_run.json \
  --prediction-target delta \
  --alphas 0,0.00001,0.0001,0.001,0.01,0.1,1 \
  --batch-size 128 \
  --device cpu \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_multi_horizon_linear_h2_h5_v1
```

Current result summary:

- Shared horizons: `[2, 5]`.
- Evaluation windows: `35097`.
- Best alpha: `1.0`.
- Overall decoded MSE: `0.001527`.
- Overall persistence MSE: `0.002065`.
- Overall improvement over persistence: `0.000537`.
- h2 test improvement over persistence: about `0.0000973`.
- h5 test improvement over persistence: about `0.0001067`.
- h2 test active-cell improvement: about `-0.00131`; h5: about `-0.00124`.
- h2 test high-change improvement: about `0.00510`; h5: about `0.00552`.

Interpretation rule: this is a real one-run shared-horizon control, but it is
not the final neural shared-horizon model. It globally beats persistence and is
strong on high-change masks, but it is worse than persistence on active-cell and
top-activity masks, so do not overstate biological-region performance.

## Shared Multi-Horizon Latent GRU

A true neural shared-horizon trainer has been added. It trains one latent GRU
across multiple horizon datasets using a normalized-horizon-conditioned output
head. It writes one checkpoint and one metrics file with per-horizon decoded,
latent, split, active-cell, and high-change diagnostics.

Implementation and CLI:

```bash
neurobench/dynamics/multi_horizon_neural.py
.venv-neurobench/bin/python -m neurobench.cli.main dynamics train-shared-gru-horizons --help
```

Durable real grid128 artifacts have been generated for the first neural shared
horizon trainer. The implementation has also been verified by synthetic unit
tests plus a temporary CLI smoke run under `/tmp/neurev_shared_gru_cli_smoke`.

Current real artifacts:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_multi_horizon_gru_h2_h5_v1/multi_horizon_gru_run.json
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_multi_horizon_gru_h2_h5_v1/multi_horizon_gru_metrics.json
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_multi_horizon_gru_h2_h5_v1/multi_horizon_gru_checkpoint.pt
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_multi_horizon_gru_h2_h5_v1/w8_s1_h2/prediction_examples.png
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_multi_horizon_gru_h2_h5_v1/w8_s1_h5/prediction_examples.png
```

New shared-GRU and shared-Transformer runs also emit review-ready per-horizon `per_horizon_metrics_for_review.json` files beside their examples and clips, plus operational progress artifacts and use chunked
decoded evaluation:

```bash
<out-dir>/multi_horizon_gru_progress.jsonl
<out-dir>/multi_horizon_gru_progress_latest.json
<out-dir>/<dataset-key>/prediction_examples.json
<out-dir>/<dataset-key>/prediction_clip_examples.json
```

Metrics from new runs include `decoded_evaluation_mode=chunked`,
`prediction_clip_examples_path`, and
`evaluation_batch_size`. The existing real `shared_multi_horizon_gru_h2_h5_v1`
run predates the logging/chunked-evaluation/clip-export patches, so it has final
metrics but not the progress JSONL stream, chunked-evaluation metadata, or
`prediction_clip_examples.json`.

Run command used, CPU-only so it did not use the CUDA device while the Stage A
overnight sweep was active:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dynamics train-shared-gru-horizons \
  --dataset Outputs/GridModel/060126_crop512_grid128_max_v1/datasets/w8_s1_h2/dynamics_dataset.json \
  --dataset Outputs/GridModel/060126_crop512_grid128_max_v1/datasets/w8_s1_h5/dynamics_dataset.json \
  --autoencoder-run Outputs/GridModel/060126_crop512_grid128_max_v1/models/autoencoder128_s1_ld64_bc16_e60_lr0p0010_v1/autoencoder_run.json \
  --hidden-dim 64 \
  --num-layers 1 \
  --epochs 25 \
  --batch-size 64 \
  --evaluation-batch-size 16 \
  --learning-rate 0.001 \
  --prediction-target delta \
  --device cpu \
  --seed 7 \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_multi_horizon_gru_h2_h5_v1
```

For new comparable runs, keep the default heartbeat printing or add
`--progress-interval-epochs 1` explicitly. Use `--evaluation-batch-size 16` or
smaller on memory-constrained CPUs. Use `--quiet-progress` only when a wrapper
script should suppress stdout; the JSON progress files are still written.

Current result summary:

- Shared horizons: `[2, 5]`.
- Training windows: `16053`.
- Evaluation windows: `35097`.
- Selection latent-code MSE: `0.134261`.
- Overall decoded MSE: `0.001660`.
- Overall persistence MSE: `0.002065`.
- Overall improvement over persistence: `0.000404`.
- h2 test improvement over persistence: about `-0.0000534`.
- h5 test improvement over persistence: about `0.0000315`.
- h2 test active-cell improvement: about `-0.00126`; h5: about `-0.00115`.
- h2 test high-change improvement: about `0.00494`; h5: about `0.00528`.

Interpretation rule: this is a real one-run neural shared-horizon baseline, and
it proves the shared-GRU path works end to end. It is not yet stronger than the
shared linear baseline and should not be described as improving active-cell
regions. The next neural version should test a small GRU/Transformer follow-up
grid; heartbeat/progress logging and chunked decoded evaluation are now
implemented for new shared-GRU runs.

Resource note: the completed CPU run took about 17 minutes and emitted no
progress because it predates the heartbeat patch. New runs write progress at
start, per dataset encode, train start, selected train epochs, per-horizon
evaluation, and completion. New runs also decode evaluation predictions in
chunks, controlled by `--evaluation-batch-size`. Do not switch this command to
CUDA while the active overnight sweep is using the GPU unless the user
explicitly asks.

## Shared Multi-Horizon Latent Transformer

A true neural shared-horizon Transformer trainer has been added. It trains one
latent Transformer across multiple horizon datasets using a normalized-horizon
conditioned output head and writes one checkpoint plus per-horizon decoded,
latent, split, active-cell, and high-change diagnostics.

Implementation and CLI:

```bash
neurobench/dynamics/multi_horizon_neural.py
.venv-neurobench/bin/python -m neurobench.cli.main dynamics train-shared-transformer-horizons --help
```

New shared-Transformer runs emit:

```bash
<out-dir>/multi_horizon_transformer_run.json
<out-dir>/multi_horizon_transformer_metrics.json
<out-dir>/multi_horizon_transformer_checkpoint.pt
<out-dir>/multi_horizon_transformer_progress.jsonl
<out-dir>/multi_horizon_transformer_progress_latest.json
<out-dir>/<dataset-key>/prediction_examples.json
<out-dir>/<dataset-key>/prediction_clip_examples.json
```

A first durable real grid128 shared-Transformer rescue run has completed:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shxfmr_h2_h5_delta_md64_h2_l1_lr1em4_s7/multi_horizon_transformer_run.json
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shxfmr_h2_h5_delta_md64_h2_l1_lr1em4_s7/multi_horizon_transformer_metrics.json
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shxfmr_h2_h5_delta_md64_h2_l1_lr1em4_s7/multi_horizon_transformer_checkpoint.pt
```

Result summary for `shxfmr_h2_h5_delta_md64_h2_l1_lr1em4_s7`:

- Overall decoded MSE: `0.001592`.
- Overall persistence MSE: `0.002065`.
- Overall improvement over persistence: `0.000472`.
- h2 test improvement: about `0.0000765`; h5 test improvement: about `0.0001149`.
- h2 test active-cell improvement: about `-0.00131`; h5: about `-0.00120`.
- h2 test high-change improvement: about `0.00503`; h5: about `0.00528`.
- Interpretation: this is the strongest completed shared-neural run so far and
  wins h5 among compared shared-horizon runs, but it still does not solve
  active-cell forecasting.

The current follow-up grid now contains `2` pending ready CPU Transformer
commands for this trainer after the seed-13 GRU companion completed.

## Shared-Horizon Neural Follow-Up Grid

A small follow-up grid plan has been generated, and the first planned CPU
shared-GRU seed pair plus two rescue shared-Transformer entries have completed. The
plan is intended to keep the next shared-horizon step inspectable while running
one CPU command at a time. Both shared-GRU and shared-Transformer commands are
executable.

Current real artifacts:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/plans/shared_horizon_neural_grid_v1/shared_horizon_neural_grid_plan.md
Outputs/GridModel/060126_crop512_grid128_max_v1/plans/shared_horizon_neural_grid_v1/shared_horizon_neural_grid_manifest.json
Outputs/GridModel/060126_crop512_grid128_max_v1/plans/shared_horizon_neural_grid_v1/run_shared_horizon_neural_grid.sh
Outputs/GridModel/060126_crop512_grid128_max_v1/plans/shared_horizon_neural_grid_v1/shared_horizon_neural_grid_status.md
Outputs/GridModel/060126_crop512_grid128_max_v1/plans/shared_horizon_neural_grid_v1/shared_horizon_neural_grid_status.json
```

Current plan summary:

- Planned configs: `20`.
- Directly executable shared-GRU configs: `16`.
- Directly executable shared-Transformer configs: `4`.
- Current status report, refreshed at `2026-06-11T01:45:04Z`: `16 pending`, `4 completed`, `0 started`, `0 incomplete`.
- The status report now includes per-horizon test active-cell and high-change
  diagnostics. The four completed shared-neural entries are still `0/2` on
  positive active-cell horizons, while their minimum high-change improvements
  are positive. The best completed shared-neural row remains
  `shgru_h2_h5_delta_hd32_l1_lr3em4_s13` with global improvement `0.0005019`
  and minimum test active-cell improvement `-0.001421`.
- Completed configs: `shgru_h2_h5_delta_hd32_l1_lr3em4_s7`,
  `shgru_h2_h5_delta_hd32_l1_lr3em4_s13`,
  `shxfmr_h2_h5_delta_md64_h2_l1_lr1em4_s7`, and
  `shxfmr_h2_h5_delta_md64_h4_l1_lr3em4_s7`.
- Pending configs: `14` GRU entries and `2` Transformer entries.
- GRU command defaults: CPU, `epochs=25`, `batch_size=64`,
  `evaluation_batch_size=16`, seeds `7,13`, target `delta`.
- Transformer command defaults: CPU, `epochs=25`, `batch_size=64`,
  `evaluation_batch_size=16`, target `delta`, normalized-horizon head
  conditioning.
- Run root for planned outputs:
  `Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1`.

Completed shared-neural grid artifacts:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shgru_h2_h5_delta_hd32_l1_lr3em4_s7/multi_horizon_gru_metrics.json
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shgru_h2_h5_delta_hd32_l1_lr3em4_s13/multi_horizon_gru_metrics.json
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shxfmr_h2_h5_delta_md64_h2_l1_lr1em4_s7/multi_horizon_transformer_metrics.json
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shxfmr_h2_h5_delta_md64_h4_l1_lr3em4_s7/multi_horizon_transformer_metrics.json
```

Result summary for `shgru_h2_h5_delta_hd32_l1_lr3em4_s7`:

- Overall decoded MSE: `0.001612`.
- Overall persistence MSE: `0.002065`.
- Overall improvement over persistence: `0.000453`.
- h2 test improvement: about `0.0000293`; h5 test improvement: about `0.0000841`.
- h2 test active-cell improvement: about `-0.00145`; h5: about `-0.00134`.
- h2 test high-change improvement: about `0.00495`; h5: about `0.00523`.
- Interpretation: this is a stronger planned-grid GRU than the first ad hoc
  shared-GRU baseline on global/test decoded MSE, but it still should not be
  claimed to solve active-cell forecasting.

Result summary for `shgru_h2_h5_delta_hd32_l1_lr3em4_s13`:

- Overall decoded MSE: `0.001563`.
- Overall persistence MSE: `0.002065`.
- Overall improvement over persistence: `0.000502`.
- h2 test improvement: about `0.000102`; h5 test improvement: about `0.000141`.
- h2 test active-cell improvement: about `-0.00142`; h5: about `-0.00135`.
- h2 test high-change improvement: about `0.00500`; h5: about `0.00521`.
- Interpretation: this seed-13 companion is the strongest completed shared-GRU
  and beats the seed-7 GRU globally and on both horizon test splits, but it
  still does not solve active-cell forecasting.

Result summary for `shxfmr_h2_h5_delta_md64_h2_l1_lr1em4_s7`:

- Overall decoded MSE: `0.001592`.
- Overall persistence MSE: `0.002065`.
- Overall improvement over persistence: `0.000472`.
- h2 test improvement: about `0.0000765`; h5 test improvement: about `0.0001149`.
- h2 test active-cell improvement: about `-0.00131`; h5: about `-0.00120`.
- h2 test high-change improvement: about `0.00503`; h5: about `0.00528`.
- Interpretation: this is the best completed shared-neural candidate and wins
  h5 among compared shared-horizon runs, but active-cell improvement is still
  negative.

Result summary for `shxfmr_h2_h5_delta_md64_h4_l1_lr3em4_s7`:

- Overall decoded MSE: `0.001573`.
- Overall persistence MSE: `0.002065`.
- Overall improvement over persistence: `0.000491`.
- h2 test improvement: about `-0.0000677`; h5 test improvement: about `0.0000153`.
- h2 test active-cell improvement: about `-0.00218`; h5: about `-0.00208`.
- h2 test high-change improvement: about `0.00471`; h5: about `0.00510`.
- Interpretation: this is the best completed shared-neural candidate globally,
  but it is worse than the first rescue Transformer on per-horizon test
  improvement and active-cell metrics. It is not an active-cell rescue.

Real clip-enabled review for these completed entries:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/reviews/shared_horizon_neural_clip_review_v1/video_error_review.html
Outputs/GridModel/060126_crop512_grid128_max_v1/reviews/shared_horizon_neural_clip_review_v1/video_error_review.json
```

The review-specific manifest is here:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/reviews/shared_horizon_neural_clip_review_v1/comparison_input/comparison_manifest.json
```

Regenerate that manifest with:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dynamics build-shared-horizon-review-input \
  --run planned_gru_hd32=Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shgru_h2_h5_delta_hd32_l1_lr3em4_s7/multi_horizon_gru_metrics.json \
  --run planned_gru_hd32_s13=Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shgru_h2_h5_delta_hd32_l1_lr3em4_s13/multi_horizon_gru_metrics.json \
  --run rescue_transformer_md64=Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shxfmr_h2_h5_delta_md64_h2_l1_lr1em4_s7/multi_horizon_transformer_metrics.json \
  --run rescue_transformer_h4_lr3e4=Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shxfmr_h2_h5_delta_md64_h4_l1_lr3em4_s7/multi_horizon_transformer_metrics.json \
  --comparison-dir Outputs/GridModel/060126_crop512_grid128_max_v1/comparison_grid128_sequence_1day_v1 \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/reviews/shared_horizon_neural_clip_review_v1/comparison_input \
  --title "Shared-Horizon Neural Review Input"
```

Current review summary: `8` selected models, all using `artifact_mode=temporal_clip`, with zero missing visuals: planned GRU seed-7 h2/h5, planned GRU seed-13 h2/h5, first rescue Transformer h2/h5, and second rescue Transformer h2/h5. Each temporal strip currently contains `8` adjacent frames from the saved clip artifact.

Shared-horizon baseline comparison report:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_baseline_comparison_v1/shared_horizon_baseline_comparison.md
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_baseline_comparison_v1/shared_horizon_baseline_comparison.json
```

Comparison result: `shared_linear` is still best overall and wins h2
per-horizon test improvement. `rescue_transformer_md64` still wins h5
per-horizon test improvement. The new `planned_gru_hd32_s13` improves over
the seed-7 GRU globally and on both horizon test splits, but still has negative
active-cell improvement on both horizons. `rescue_transformer_h4_lr3e4` ranks
second overall but has negative h2 test improvement and worse active-cell
metrics. All compared shared-horizon runs remain negative on test active-cell
improvement.

Regenerate that comparison with:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dynamics compare-shared-horizon-runs \
  --run shared_linear=Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_multi_horizon_linear_h2_h5_v1/multi_horizon_linear_metrics.json \
  --run first_shared_gru=Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_multi_horizon_gru_h2_h5_v1/multi_horizon_gru_metrics.json \
  --run planned_gru_hd32=Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shgru_h2_h5_delta_hd32_l1_lr3em4_s7/multi_horizon_gru_metrics.json \
  --run planned_gru_hd32_s13=Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shgru_h2_h5_delta_hd32_l1_lr3em4_s13/multi_horizon_gru_metrics.json \
  --run rescue_transformer_md64=Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shxfmr_h2_h5_delta_md64_h2_l1_lr1em4_s7/multi_horizon_transformer_metrics.json \
  --run rescue_transformer_h4_lr3e4=Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1/shxfmr_h2_h5_delta_md64_h4_l1_lr3em4_s7/multi_horizon_transformer_metrics.json \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_baseline_comparison_v1
```

Active-cell rescue plan:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/plans/active_cell_rescue_v1/active_cell_rescue_plan.md
Outputs/GridModel/060126_crop512_grid128_max_v1/plans/active_cell_rescue_v1/active_cell_rescue_plan.json
```

Current rescue recommendation, refreshed at `2026-06-11T01:45:17Z`: run
`shxfmr_h2_h5_delta_md64_h2_l2_lr1em4_s7` next, but only when CPU/RAM
headroom is intentionally available. The rationale
is another architecture-diverse Transformer test after shared linear, GRU, and
both completed rescue Transformers all remain negative on active-cell
improvement. The refreshed plan reports `12` active-cell warnings, `4`
completed shared-neural entries, and `16` pending shared-neural entries. The
Markdown plan includes a `Next Candidate Command` section with the exact CPU
command and expected output directory for the first recommended candidate.

Regenerate the rescue plan with:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dynamics plan-active-cell-rescue \
  --comparison-report Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_baseline_comparison_v1/shared_horizon_baseline_comparison.json \
  --grid-status Outputs/GridModel/060126_crop512_grid128_max_v1/plans/shared_horizon_neural_grid_v1/shared_horizon_neural_grid_status.json \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/plans/active_cell_rescue_v1 \
  --title "Grid128 Active-Cell Rescue Plan"
```

Refresh the status report with:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dynamics status-shared-horizon-neural-grid \
  --manifest Outputs/GridModel/060126_crop512_grid128_max_v1/plans/shared_horizon_neural_grid_v1/shared_horizon_neural_grid_manifest.json \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/plans/shared_horizon_neural_grid_v1 \
  --title "Grid128 Shared-Horizon Neural Grid Status"
```

Regenerate the plan with:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dynamics plan-shared-horizon-neural-grid \
  --dataset Outputs/GridModel/060126_crop512_grid128_max_v1/datasets/w8_s1_h2/dynamics_dataset.json \
  --dataset Outputs/GridModel/060126_crop512_grid128_max_v1/datasets/w8_s1_h5/dynamics_dataset.json \
  --autoencoder-run Outputs/GridModel/060126_crop512_grid128_max_v1/models/autoencoder128_s1_ld64_bc16_e60_lr0p0010_v1/autoencoder_run.json \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/plans/shared_horizon_neural_grid_v1 \
  --run-root Outputs/GridModel/060126_crop512_grid128_max_v1/reports/shared_horizon_neural_grid_v1 \
  --device cpu \
  --epochs 25 \
  --batch-size 64 \
  --evaluation-batch-size 16 \
  --seeds 7,13 \
  --max-gru-configs 16 \
  --title "Grid128 Shared-Horizon Neural Follow-Up Grid"
```

Interpretation rule: the shell script is executable but should not be launched
blindly. Run one neural command at a time unless CPU/RAM headroom is confirmed
and no higher-priority GPU decision is pending. Transformer rows now call the tested
`train-shared-transformer-horizons` trainer.

## Latent State Interpretation

The latent interpretation utility has been implemented for the grid128
autoencoder. It reads `latent_codes.npz` from `autoencoder_run.json` and writes
JSON, Markdown, HTML, and PNG summaries for PCA structure, video-level
neighbors, latent velocity, and label separability.

Current real artifacts:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/latent_interpretation_autoencoder128_v1/latent_interpretation_report.html
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/latent_interpretation_autoencoder128_v1/latent_interpretation_report.md
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/latent_interpretation_autoencoder128_v1/latent_interpretation_report.json
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/latent_interpretation_autoencoder128_v1/latent_video_embedding.png
Outputs/GridModel/060126_crop512_grid128_max_v1/reports/latent_interpretation_autoencoder128_v1/latent_trajectory_preview.png
```

Current grid128 latent interpretation findings at generation time:

- Frames: `17664`.
- Videos: `11`.
- Latent dimension: `64`.
- Label frame counts: `left=6401`, `neutral=6486`, `right=4777`.
- PCA explained variance ratios for the first four components: about `0.377`,
  `0.201`, `0.144`, `0.078`.
- Leave-one-video nearest-centroid label accuracy: `0.1818`.
- Between/within centroid distance ratio: about `0.642`.

Interpretation rule: do not claim that the current autoencoder latent space
cleanly separates left/right/neutral behavior. The current evidence suggests
weak label separability, which is useful motivation for supervised, contrastive,
or multi-horizon latent objectives.

A latent objective follow-up plan now turns that evidence into concrete next
steps and acceptance gates:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/plans/latent_objective_plan_v1/latent_objective_plan.md
Outputs/GridModel/060126_crop512_grid128_max_v1/plans/latent_objective_plan_v1/latent_objective_plan.json
```

Current plan diagnosis: `weak_label_separability`. The first recommended check
is a held-out-video supervised latent-head smoke test on frozen latent summaries
before spending GPU time on a full auxiliary-objective sweep. Other candidates
are a supervised contrastive regularizer, multi-task reconstruction plus h2/h5
prediction, and a targeted audit of latent dimensions `6`, `12`, `50`, `62`,
and `36`.

The first smoke test has now been run with a NumPy `ridge_linear` video-level
head and leave-one-video-out evaluation:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/plans/latent_head_smoke_v1/latent_classifier_report.md
Outputs/GridModel/060126_crop512_grid128_max_v1/plans/latent_head_smoke_v1/latent_classifier_run.json
Outputs/GridModel/060126_crop512_grid128_max_v1/plans/latent_head_smoke_v1/per_video_predictions.tsv
```

Result: accuracy `0.3636`, balanced accuracy `0.3611`, macro F1 `0.3545`,
chance accuracy `0.3333`, and majority-class accuracy `0.3636` over `11`
leave-one-video folds. The JSON now includes `metrics.gate_summary` with
`passes_chance_gate=true`, `passes_majority_gate=false`,
`accuracy_minus_chance=0.0303`, `accuracy_minus_majority=0`, and interpretation
`weak_chance_only_signal`. The Markdown report now includes the confusion matrix
and all per-video predictions. This is not strong evidence of behavior-state
separation.

Regenerate the latent-head smoke test with:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dynamics train-latent-classifier \
  --dataset Outputs/GridModel/060126_crop512_grid128_max_v1/datasets/w8_s1_h2/dynamics_dataset.json \
  --autoencoder-run Outputs/GridModel/060126_crop512_grid128_max_v1/models/autoencoder128_s1_ld64_bc16_e60_lr0p0010_v1/autoencoder_run.json \
  --evaluation leave_one_video_out \
  --classifier ridge_linear \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/plans/latent_head_smoke_v1
```

Regenerate the latent objective plan with:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dynamics plan-latent-objectives \
  --interpretation-report Outputs/GridModel/060126_crop512_grid128_max_v1/reports/latent_interpretation_autoencoder128_v1/latent_interpretation_report.json \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/plans/latent_objective_plan_v1 \
  --title "Grid128 Latent Objective Follow-Up Plan"
```

Regenerate the latent interpretation report with:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dynamics interpret-latents \
  --autoencoder-run Outputs/GridModel/060126_crop512_grid128_max_v1/models/autoencoder128_s1_ld64_bc16_e60_lr0p0010_v1/autoencoder_run.json \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/reports/latent_interpretation_autoencoder128_v1 \
  --max-frame-points 5000 \
  --nearest-neighbors 3 \
  --title "Grid128 Latent State Interpretation"
```

## Adaptive Stage B Planning

The adaptive planner has been implemented for deciding the next smaller search
after enough Stage A evidence has accumulated. It did not stop or modify the
Stage A sweep. Its `next_sweep_manifest.json` can now be executed directly by
the manifest-aware overnight sweep runner.

Current Stage B plan artifacts:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_sequence_stage_b_v1/next_sweep_plan.md
Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_sequence_stage_b_v1/next_sweep_plan.json
Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_sequence_stage_b_v1/next_sweep_manifest.json
```

Current plan summary:

- Source progress at planning time: `477 / 972`.
- Source progress records at planning time: `1110` records, with `439` completed,
  `29` failed, and `642` skipped progress rows.
- Selected experiments: `57`.
- Selected families: `4` array baselines, `2` linear latent deltas, `16` latent
  GRUs, `32` latent Transformers, `1` ConvGRU pixel scout, `1` ConvLSTM
  pixel scout, and `1` temporal-CNN pixel scout.
- Dataset balance and target balance are recorded in `next_sweep_plan.json` for the refreshed `57`-experiment plan.
- Target balance: `50` delta learned/baseline specs, `2` persistence controls,
  `2` moving-average controls, and `2` pixel scouts with no latent target.
- The partial experiment report now has a `Recommended Next Sweep` section that
  surfaces this Stage B manifest command and the active-cell rescue candidate
  directly from nearby `plans/` artifacts. The latest refreshed plan was
  generated at `2026-06-13T04:39:55.073420+00:00` from source progress
  `477 / 972` and selected `57` experiments; its manifest dry-run was
  regenerated after that plan and succeeded with `57`
  experiments. The artifact audit confirms this refreshed plan matches the
  source sweep progress log.
- Deferred specs: `192` ConvGRU pixel, `192` ConvLSTM pixel, and `288` temporal
  CNN pixel specs remain deferred as broad families because archived failures are
  dominated by CUDA OOMs. The single selected ConvGRU row is the current best
  completed pixel scout, not a broad pixel-family relaunch.

Regenerate the Stage B plan with:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dynamics plan-next-sweep \
  --sweep-dir Outputs/GridModel/060126_crop512_grid128_max_v1/sweeps/grid128_sequence_1day_v1 \
  --comparison-dir Outputs/GridModel/060126_crop512_grid128_max_v1/comparison_grid128_sequence_1day_v1 \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_sequence_stage_b_v1 \
  --max-experiments 80 \
  --suggested-batch-size 4
```

Important: `next_sweep_manifest.json` is now executable with the
manifest-aware overnight sweep runner. The current real dry-run succeeded with
`57` experiments and wrote:

```bash
Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_sequence_stage_b_v1/stage_b_sweep/sweep_manifest.json
```

Dry-run command used:

```bash
.venv-neurobench/bin/python -m neurobench.dynamics.overnight_sweep \
  --manifest Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_sequence_stage_b_v1/next_sweep_manifest.json \
  --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_sequence_stage_b_v1/stage_b_sweep \
  --device cuda \
  --batch-size 4 \
  --seeds 7,13 \
  --time-limit-hours 48 \
  --dry-run
```

Do not launch this Stage B sweep until the stopped Stage A state is reviewed and the user explicitly chooses the next GPU job. The latest dry run validated `57` experiments at source
progress `477 / 972` and wrote
`plans/grid128_sequence_stage_b_v1/stage_b_sweep/sweep_manifest.json`.

## Interpretation Rules

Do not claim a model is useful only because it completed. Prefer this hierarchy:

1. Beats split-aware persistence on validation.
2. Beats split-aware persistence on test.
3. Improves active-cell or high-change regions, not only background.
4. Performs consistently across `h2` and `h5`.
5. Has visual evidence in forecast clips.

Treat negative persistence improvement as a warning, even if raw prediction MSE
looks small.

## Recommended Next Utilities

The Stage A GPU sweep is stopped at `477 / 972`; do not describe it as active
or launch another GPU job without an explicit user choice. Until the next GPU
job is chosen, prioritize CPU/file utilities:

1. Use the Stage B launch-readiness artifact before any GPU command:
   `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_stage_b_launch_readiness_v1/stage_b_launch_readiness.md`.
   Refresh it with:

   ```bash
   .venv-neurobench/bin/python -m neurobench.cli.main dynamics stage-b-launch-readiness \
     --root Outputs/GridModel/060126_crop512_grid128_max_v1 \
     --out-dir Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_stage_b_launch_readiness_v1
   ```

   The default next GPU job is the validated 57-experiment Stage B manifest;
   resume Stage A from index `478` only if the user explicitly chooses that
   alternative.
2. Use the active-cell rescue plan for the next neural run; current first
   candidate is `shxfmr_h2_h5_delta_md64_h2_l2_lr1em4_s7`.
3. Run one additional planned shared-GRU or shared-Transformer entry only when
   CPU/RAM headroom is intentionally available. The latest completed CPU-side
   companion is `shgru_h2_h5_delta_hd32_l1_lr3em4_s13`; it improved global MSE
   over the seed-7 GRU but remained negative on active cells.
4. Use the completed latent-head smoke-test result before a full auxiliary
   objective sweep; it only weakly beats chance and does not beat the majority
   baseline, so stronger evidence is still needed before claiming label-state
   encoding.
5. Regeneration of dashboard, reports, Stage B plans, multi-horizon reports,
   latent reports, and video reviews as future approved runs or CPU backfills add evidence. The active-cell
   review can be regenerated with `--selection-mode best_active_cell`; the report
   Visual Examples table now shows selection mode and top model IDs for quick
   review, and the Active-Cell Error Check includes the active-vs-global tradeoff.
   For high-value checkpointed pixel rows, `dynamics backfill-concept-examples
   --backfill-metrics` can selectively recompute full split, structured, and
   per-video diagnostics on CPU before regenerating the dashboard/report. Run
   the same command with `--dry-run` first to estimate work without writing
   files. Add `--json` to emit a machine-readable summary, and
   `--markdown-out <path>` to write the same preflight as a reusable Markdown
   note; the real active-cell
   leader JSON preflight was validated with `python3 -m json.tool` and its
   Markdown companion was regenerated through the CLI. Dry-run
   checkpoint loading is CPU-only even if `--device cuda` is supplied, and
   dry-run output previews the selected example indices. The refreshed current active-cell leader dry run for index `324` reported `17,565` `w8_s1_h2` windows, split
   windows `test=4841`, `train=8034`, `val=4690`, `1` example batch and
   `1098` metric batches at batch size `16`, intended write/update targets
   `prediction_examples.json`, `prediction_examples_backfill.json`, and
   `concept_metrics.json`, split videos `test=3`,
   `train=5`, `val=3`, test labels `neutral=1682`, `left=1583`,
   `right=1576`, train labels `neutral=3252`, `left=3183`, `right=1599`,
   val labels `left=1599`, `right=1575`, `neutral=1516`, top test videos
   `7 rest=1682`, `8 left=1583`, `5 right=1576`, about `10.737 GiB` of array
   payload, and example preview
   `11125:test:5 right, 11126:test:5 right, 11127:test:5 right`, so full metric
   backfill should be a
   planned-headroom job on grid128 h2/h5 datasets. The validated active-cell JSON preflight
   artifact is:
   `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_backfill_preflight_v1/current_active_cell_leader_metric_backfill_preflight.json`.
   Human-readable active-cell summary, regenerated through `--markdown-out`:
   `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_backfill_preflight_v1/current_active_cell_leader_metric_backfill_preflight.md`.
   The current learned/global leader at index `330` now has the same dry-run
   metric-backfill planning artifacts, generated at
   `2026-06-11T02:24:12.885488+00:00`: `17,565` windows, `1098` estimated
   metric batches at batch size `16`, about `10.737 GiB` uncompressed array
   payload, and example preview
   `11125:test:5 right, 11126:test:5 right, 11127:test:5 right`. JSON:
   `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_backfill_preflight_v1/current_learned_leader_metric_backfill_preflight.json`.
   Markdown:
   `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_backfill_preflight_v1/current_learned_leader_metric_backfill_preflight.md`.

Refer to `VISIONS.md` for the ambitious roadmap and acceptance criteria.

## Safety Notes For Future Agents

- Never use `git reset --hard` or revert user changes without explicit user
  approval.
- Do not delete archived failed progress logs.
- Do not restart preprocessing from raw TIFFs unless artifacts are missing or the
  user asks. The faster path reuses crop512 registration artifacts.
- Before launching another long run, verify:
  - input datasets exist;
  - autoencoder run exists;
  - no duplicate sweep process is already active;
  - GPU memory is available.
- If asked for current status, inspect live files and processes rather than
  relying only on handoff text.
