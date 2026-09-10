# Spon Ca Burst Gamma-LS radial-support sufficiency v1

## Question and claim boundary

The completed Gamma-LS G1/G2 screen searched half-widths 7, 11, and 15 pixels.
It did **not** simply choose the largest border: all four fold-local finalists
used half-width 11 and guard radius 5. Three folds used shape 9/mode fraction
1.0; fold 3 used shape 5/mode fraction 0.5. That result does not prove that
11 pixels is sufficient because supports larger than 15 were absent.

This extension asks whether a larger radial reference produces a material
training-window contrast or protected known-positive recovery gain that
justifies its latency. It remains burst-window-supervised and never uses
sparse-positive coordinates or identities for context selection. Unmatched
candidates remain unknown. It does not claim precision, specificity, a false
positive rate, or multiscale inference.

## Frozen radial search

Stage A evaluates these `(half-width, guard-radius)` pairs:

```text
(11,5), (15,5), (15,7), (19,5), (19,7),
(23,5), (23,9), (31,7), (31,11)
```

Every pair crosses Gamma shape `2, 5, 9` and mode fraction `0.5, 0.75, 1.0`.
This gives 81 modern radial contexts, 243 context/representation maps, 972
fold metric cells, and 1,944 quiet-role-swap rows. Only the fixed `raw`,
`difference_signed`, and `difference_energy_normalized` representations enter
selection. Floors remain fit per context, representation, fold, and reversed
quiet-half role.

Half-widths through 23 may become efficient support candidates. Half-width 31
is a boundary probe. Contexts remain fold-local; no across-fold context is
selected before protected evaluation.

## Adaptive upper-bound rule

Training near-optimality is frozen as a contrast deficit no larger than

```text
max(0.005, 0.05 * abs(best fold-local mean contrast)).
```

If the best half-width-31 context is best or within that tolerance in **any**
outer training fold, Stage B automatically evaluates:

```text
(39,9), (39,15), (47,11), (47,19)
```

with the same shapes and mode fractions. Stage B adds 36 radial contexts and
432 fold metric cells. If the best half-width-47 context remains near-optimal
in any fold, the upper boundary is unresolved and no sufficiency claim is
allowed. If the endpoint is clearly inferior in every fold, the tested upper
boundary is closed.

This adaptive rule is based only on the sealed training-window statistic. It
does not inspect protected labels.

## Actually executed controls

Controls are run through the same fixed representations and fold windows but
carry `eligible_primary=false` and cannot enter any radial selector:

1. `legacy_exact_n9_mode35_w23_eps64`: archived 23-by-23 square-truncated
   Gamma reference, center-only exclusion, reflect padding, additive epsilon
   64.
2. `signed_square_annulus_ls_h11_g3`: signed uniform square outer-minus-square
   guard moments with valid-reference border renormalization and the same
   fold-local scale-floor procedure as modern Gamma-LS. This is the direct
   support-geometry control.
3. `maintained_positive_box_cfar_h11_g3`: the maintained positive-clipped,
   replicate-boundary box-CFAR scoring semantics, implemented device-resident
   and checked against the maintained NumPy implementation.

The third control distinguishes an implementation-family comparison from the
fairer signed geometry comparison. Neither square result may be described as
Gamma-LS.

## Candidate and comparator contract

`fold_contexts.json` exposes, separately for each held-out burst:

- the original h11 G2 context;
- the smallest h<=23 context within the training near-optimal tolerance and
  above the `-0.005` worst-representation contrast floor, or an explicit
  best-h<=23 fallback;
- the best larger-support comparator;
- the unrestricted training-best context; and
- the predeclared maximum-support endpoint.

All are explicit radial specs. Protected evaluation may deduplicate identical
context IDs within a fold, but it must not replace fold-local selection with an
across-fold pooled winner.

## Latency and final sufficiency rule

For every executed half-width and control, measure one-frame Gamma-stage
latency after 50 warm-ups over 200 synchronized repetitions. Report p50, p95,
p99, maximum, mean, and peak allocated VRAM. Also retain the full 560-frame
batched-map runtime recorded by the contrast screen.

The stage-only latency gate requires p99 no greater than 1 ms and either p50 no
more than 1.25 times h11 or an absolute p50 increase no greater than 0.10 ms.
Passing this necessary Gamma-stage gate is not a whole-pipeline 1-kHz result.

No support is declared sufficient from contrast or latency alone. The final
claim requires all three evidence classes:

1. training-window contrast under the rule above;
2. protected v1 sensitivity where the smaller candidate loses at most one B58
   matched occurrence in total, at most one in any burst, and no more than
   0.02 budget-curve AUC versus the larger comparator; and
3. the repeated latency gate.

Protected recall results are joined only after `fold_contexts.json` is frozen
and hashed. Latest-v7 labels remain descriptive sensitivity.

## Scientific-audit status

This screen produces tables and frozen context specifications but no candidate
stream. Its artifact status must remain `complete_support_screen_only` and its
scientific audit remains pending. The downstream protected candidate run must
produce the complete expert-only, model-only, and matched-comparison artifact
set required by `SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md` before paper promotion.

## Execution

Use the repository virtual environment and new, non-colliding output roots:

```bash
.venv-neurobench/bin/python -m \
  neurobench.experiments.gamma_ls_difference.support_sufficiency preflight \
  --config examples/spon_ca_burst_gamma_ls_support_sufficiency_v1.example.json \
  --base-preflight-dir Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_difference_ablation_v1_preflight_20260908_r7 \
  --artifact-dir Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_support_sufficiency_v1_preflight_20260908_r1

.venv-neurobench/bin/python -m \
  neurobench.experiments.gamma_ls_difference.support_sufficiency run \
  --config examples/spon_ca_burst_gamma_ls_support_sufficiency_v1.example.json \
  --base-preflight-dir Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_difference_ablation_v1_preflight_20260908_r7 \
  --support-preflight-dir Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_support_sufficiency_v1_preflight_20260908_r1 \
  --output-dir Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_support_sufficiency_v1_gpu_screen_20260908_r1
```

The commands verify the completed base screen, artifact hashes, code hashes,
movie hash, CUDA device, VRAM, disk, cardinalities, and collision state. A
failed gate leaves the requested output absent.
