# Spon Ca Burst MSLN/MS-ICA workflow

## Purpose and interpretation

This workflow implements the generated-only and guarded real-data architecture from `MSLN_MSICA_CODEX_IMPLEMENTATION_AND_EXPERIMENT_SPEC.md`. It asks whether signed multi-scale local normalization (MSLN), two-frame per-context ICA, quiet-calibrated energy, and fixed cross-context routing provide useful auxiliary activity evidence.

The output contract is:

```text
(raw amplitude reference, activity evidence, dominant context, latent diagnostics)
```

Raw data remain immutable and authoritative. MSLN maps are signed standardized evidence. ICA components are statistical coordinates, not named biological sources without downstream evidence. Squared and tail-calibrated maps are nonnegative activity evidence. `raw * gate(activity)` is display/feature interaction only and is never a reconstruction or cleaned movie. Sparse known-positive labels are used only for protected evaluation, never fitting in the primary fixed-unsupervised track.

## Frozen standard design

The standard manifest is [spon_ca_burst_msln_msica_v1.example.json](../../examples/spon_ca_burst_msln_msica_v1.example.json). Its eight contexts are ordered deterministically:

1. spatial 5, 7, and 15 pixel square-annulus mean/std contexts;
2. causal temporal 5, 15, and 31 frame contexts;
3. temporal-15-then-spatial-5 and temporal-15-then-spatial-7 contexts.

Every per-context model compares direct common/difference coordinates, sklearn FastICA, and bounded CS-Parzen ICA using identical sample IDs. CS-Parzen is the primary objective. Component order and sign are canonicalized against persistence `[1, 1]` and innovation `[-1, 1]`; contiguous-block bootstrap artifacts expose unstable angles or swaps.

Cross-context identity, PCA, FastICA, and predeclared group energy are bounded alternatives. True ISA and label-trained routing are disabled. Energy outputs include standardized signed coordinates, raw square, bounded square, empirical quiet-tail surprise, and quiet-tail group energy. Fixed routing includes max, compact agreement, compact-minus-broad, and softmax, with a categorical dominant-context map.

## Commands

Preflight is read-only with respect to sources but creates the new configured output root and writes fingerprints, resource estimates, a label projection overlay, and tiny numerical smoke results:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment msln-msica preflight \
  --config examples/spon_ca_burst_msln_msica_v1.example.json
```

The run requires the exact matching preflight. The real Spon dataset is additionally blocked unless the user explicitly selects it:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment msln-msica run \
  --config examples/spon_ca_burst_msln_msica_v1.example.json \
  --authorize-full-spon
```

Do not use that authorization flag without an explicit user request. A completed root can be inspected without refitting:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment msln-msica summarize \
  --output-root Outputs/HierarchicalParzenICA/spon_ca_burst_msln_msica_v1
```

`--dry-run` checks the matching preflight. `--device` can only confirm, not override, the frozen manifest device. `--fold`, `--stage`, `--resume`, and `--no-video` are recorded execution controls; changing scientific parameters requires a new manifest and output root.

## Safety and resource rules

- Preflight refuses an existing output root and fingerprints the movie, labels, baseline evidence, and resolved config.
- A run refuses changed inputs or config and writes partial status atomically on failure.
- Context maps are float32 `.npy` files with support, guard, validity, scale-floor, runtime, and checksum sidecars.
- Map generation and routing are context-wise/chunked. The standard profile is CPU, four threads, one worker, one fold at a time, one context batch, at most 24 GiB RAM and 8 GiB VRAM.
- Full real-data execution requires explicit selection. Preflight is not authorization.
- Existing completed roots are never overwritten. Use a new experiment ID/root for any confirmation or widened study.

## Stage-gate reading

A completed process is not scientific success. The tiny run validates implementation and artifact contracts only. Its stage gate therefore leaves real-data MSLN utility, incremental ICA value, and morphology preservation unestablished. Real-data advancement requires held-out evidence, fold/seed consistency, active-region metrics, candidate-budget curves, and visual review. Known matches, unmatched candidates, and manually accepted scientific positives remain separate because unlabeled pixels are unknown rather than negative.

The historically strong compact `coherence_w15` feature is an external reference, while prior strict derivative-energy and morphology-aware results are explicit failure controls. Interesting candidates may be retained diagnostically even when a gate fails, but a failed gate stops automatic widening.

## Causal joint residual sweep v2

The v2 follow-up is a distinct, collision-safe study rather than a reinterpretation
of the completed v1 root. It adds a true causal joint reference volume:

```text
Zst(c,r,t) = [Raw(c,r,t) - mean(prior-frame spatial annulus)] /
             max(std(prior-frame spatial annulus), scale floor)
```

The current frame, the spatial guard square, and the most recent temporal guard
frame are excluded. This preserves the interior of a newly coherent activation
instead of subtracting current-frame active neighbors as the sequential
`temporal -> spatial` composition can do.

The staged manifest is
`examples/spon_ca_burst_joint_msln_residual_sweep_v2.example.json`:

1. screen 30 joint contexts from six spatial outer/guard pairs and five temporal
   windows, retaining compact preview/event maps rather than 30 full movies;
2. sweep the 3 x 4 bounded gate
   `beta + (1-beta) Zst^2/(kappa^2+Zst^2)` on six diverse contexts;
3. apply adjacent-frame CS-Parzen ICA separately to `Zst` and
   `Raw * gate(Zst)` for three finalists, while retaining persistence and
   innovation as separate coordinates;
4. render full-review layer-journey and finalist-comparison videos.

Visual and morphology review is the primary winner criterion. Sparse-positive
known-label recall is a quantitative guardrail, with the exact historical Raw
Direct 49/79 result recorded as an external anchor. Fixed-budget v2 proposals
are not claimed to be protocol-identical to the historical quiet-threshold
proposal count, and unmatched candidates remain unknown.

```bash
.venv-neurobench/bin/python -m neurobench.experiments.msln_msica.joint_sweep \
  preflight --config examples/spon_ca_burst_joint_msln_residual_sweep_v2.example.json

.venv-neurobench/bin/python -m neurobench.experiments.msln_msica.joint_sweep \
  gpu-preflight \
  --config examples/spon_ca_burst_joint_msln_residual_sweep_v2.example.json \
  --max-vram-gb 8

.venv-neurobench/bin/python -m neurobench.experiments.msln_msica.joint_sweep \
  run --config examples/spon_ca_burst_joint_msln_residual_sweep_v2.example.json \
  --authorize-full-spon --compute-backend cuda --max-vram-gb 8
```

The CUDA backend keeps joint normalization, gate/energy transforms, burst
pooling, sampled-pair gathering, CS-Parzen fitting and bootstrap, and full-field
ICA projection on the GPU. Sparse peak matching, metadata/figures, video
encoding, and chunked atomic artifact writes remain on CPU. GPU preflight
requires CPU/GPU numerical parity, a full-context benchmark, and an observed
peak below the 8 GiB VRAM cap. A CPU fallback remains available. The run uses
one context and one worker at a time, checkpoints Stage-A rows after each
context, and never overwrites a completed output root.

## Completed v2 result

The completed root is:

```text
Outputs/HierarchicalParzenICA/spon_ca_burst_joint_msln_residual_sweep_v2
```

The CUDA redeployment completed all 30 Stage-A contexts, 72 Stage-B gate fits,
six Stage-C ICA fits, three finalist layer banks, and four synchronized review
videos. The resumed portion took 487.9 seconds. GPU preflight measured a 12.0x
full-context normalization speedup, a 6.35 GiB observed peak allocation under
the 8 GiB cap, maximum joint-map error of `1.43e-6`, and Parzen-objective error
of `1.81e-8` relative to CPU references. The final focused suite passed 33/33.

The review-leading lane was `joint_s15_g3_t31_g1` `Zst` persistence, with
58/79 known-label matches at fixed per-burst budget 58 and 63/79 at budget 100.
The corresponding `T23` lane reached 56/79 and 58/79; the compact
`joint_s5_g1_t15_g1` alternative reached 47/79 and 52/79. The historical Raw
Direct anchor is 49/79 with 232 quiet-threshold proposals. The v2 budget-58
comparison also totals 232 candidates, but its equal per-burst allocation is
not protocol-identical and is therefore a guardrail rather than a superiority
claim.

Visual review supports retaining the broad persistence lanes and the compact
alternative for communication and confirmation. It does not resolve the broad
ICA bootstrap-angle instability, establish precision from sparse-positive
labels, or validate a biological source interpretation. The durable status is
therefore `complete / awaiting_visual_review`, not a declared scientific
winner. Use
[`docs/research/msln_msica_joint_residual_v2_package/`](../research/msln_msica_joint_residual_v2_package/README.md)
for the paper and presentation handoff; generated videos and figures remain
ignored and are represented there by explicit placeholders.

## Block-order and cascade program v3

The paired GPU block-order study is rooted at
`Outputs/HierarchicalParzenICA/spon_ca_burst_msln_msica_block_switch_v3`.
It compared `Raw -> MSLN -> MSICA` with `Raw -> MSICA -> MSLN` under the
same 30 contexts, candidate budgets, NMS/matching radii, and CUDA contract.
The broad `joint_s15_g3_t31_g1` persistence lane retained 58/79 known
positives in the control order and 57/79 in the switched order at budget 58
per burst. Raw MSICA selected Parzen bandwidth 0.2 but was unstable across 16
blocked bootstraps (40.10 degree circular SD; 0.3125 component-swap fraction).

The cascade root is
`Outputs/HierarchicalParzenICA/spon_ca_burst_msln_msica_cascade_v3`. Its
preregistered raw-MSICA stability gate failed, so the planned
`MSICA -> MSLN -> MSICA` cascade was not executed. This stopped outcome is a
scientific result rather than an incomplete computation. Retain the existing
broad `Raw -> MSLN -> MSICA` persistence representation; do not deepen the
cascade without a redesigned or regularized first-stage separation.

The cross-experiment conclusion is
`Outputs/HierarchicalParzenICA/spon_ca_burst_msln_msica_order_program_v3_summary/CONCLUSIVE_REPORT.md`.
Reproduce the program with:

```bash
.venv-neurobench/bin/python -m neurobench.experiments.msln_msica.order_program preflight \
  --config examples/spon_ca_burst_msln_msica_order_program_v3.example.json
.venv-neurobench/bin/python -m neurobench.experiments.msln_msica.order_program gpu-preflight \
  --config examples/spon_ca_burst_msln_msica_order_program_v3.example.json
.venv-neurobench/bin/python -m neurobench.experiments.msln_msica.order_program run-block-switch \
  --config examples/spon_ca_burst_msln_msica_order_program_v3.example.json --authorize-full-spon
.venv-neurobench/bin/python -m neurobench.experiments.msln_msica.order_program run-cascade \
  --config examples/spon_ca_burst_msln_msica_order_program_v3.example.json --authorize-full-spon
.venv-neurobench/bin/python -m neurobench.experiments.msln_msica.order_program conclude \
  --config examples/spon_ca_burst_msln_msica_order_program_v3.example.json
```

## Broad cascade program v4

The user-authorized v4 program deliberately overrode the v3 early-stop for a
broad diagnostic test. Its completed root is:

```text
Outputs/HierarchicalParzenICA/spon_ca_burst_msln_msica_cascade_program_v4
```

The label-free screen evaluated every one of the 30 first-stage contexts, all
900 ordered first/second-context pairs, both persistence and innovation
branches, and five MSICA2 bandwidths. It also evaluated switched-order,
five-seed ensemble, cross-branch ICA, and parallel-fusion controls over all 30
contexts. The original-order factorial alone contained 1,800 branch
combinations and 9,000 MSICA2 fits. All screening preceded label access.

Three finalists per experiment were frozen by event/quiet contrast, then
recomputed with 4,096/16,384 sample fits, FastICA sensitivity checks, and 16
blocked-bootstrap replicates for each supported ICA block. Each experiment has
a concise report, three full float32 maps, and a synchronized 560-frame video.

At the fixed budget-58 guardrail, the strictly label-free rank-1 results were
39/79 (original shallow), 41/79 (original deep), 47/79 (switched deep), 52/79
(five-seed ensemble), 46/79 (cross-branch), and 51/79 (parallel mean fusion),
versus the external Raw Direct anchor of 49/79. The primary defensible result is
therefore the ensemble's provisional 52/79, a three-match gain. Selecting the
best protected finalist after opening labels gives exploratory ceilings of
42, 41, 57, 59, 46, and 53 matches respectively; these are not unbiased winner
estimates and must not replace the label-free rank-1 comparison.

Individual ensemble ICA fits remained angle/swap unstable, but the three
finalist deep-energy ensembles had mean cross-seed map correlations of 0.978,
0.919, and 0.889. This supports energy aggregation as a robustness mechanism,
not stable biological source identity. The original `MSLN -> MSICA -> MSLN`
cascades did not beat Raw Direct; adding another MSICA also did not rescue them.
Cross-branch ICA lost three matches. Parallel fusion gained two matches for its
label-free winner. The 52/79 ensemble result remains provisional because labels
are sparse-positive, the comparison is a fixed-budget guardrail, and no
independent recording has confirmed the selector.

The primary corrected conclusion is in
`summary/CONCLUSIVE_REPORT_LABEL_FREE_PRIMARY.md` under the completed v4 root.
The original generated `CONCLUSIVE_REPORT.md` is retained unchanged for audit but
reports protected-best ceilings as its headline. Reproduce the
stages with the schema-v4 manifest
`examples/spon_ca_burst_msln_msica_broad_cascade_v4.example.json` and the
`broad_cascade` / `broad_finalize` modules. A new confirmation recording or
output root requires explicit selection; never overwrite this completed root.

## Label-free proposal review packages

Completed frozen candidates can be repackaged for review before expert labels
arrive with `neurobench workbench model-proposal-package`. The reusable
contract and proof package are documented in
[unlabeled_model_proposal_review.md](unlabeled_model_proposal_review.md). This
packaging path performs no refitting: it removes expert-derived fields, creates
an empty annotation revision, defaults the dashboard to model proposals, and
writes both blinded and model-assisted XLSX workbooks. A future new recording
still requires an explicit, collision-safe inference run to produce the frozen
candidate-bearing source app.
