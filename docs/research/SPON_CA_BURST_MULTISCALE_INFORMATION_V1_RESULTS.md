# Spon Ca Burst multiscale patch-information v1 results

## Answer

Multiscale context is useful, but larger kernels are not better standalone
detectors. The best use of broad support is to qualify or suppress a compact
response. Leakage-safe multiscale native selection improved mean recall over
the quiet-standardized carrier from 0.541 to 0.583 at budget 20 and from 0.657
to 0.677 at budget 40. At budget 58 it tied within one tenth of a percentage
point (0.702 versus 0.703), and at budget 100 it was worse (0.702 versus
0.726). This is a tight-budget ordering improvement, not a universal detector
replacement.

On the identical frozen v5 proposal union, the selected multiscale score
improved recall from 0.459/0.591/0.709 to 0.471/0.639/0.759 at budgets
20/40/58. Most folds tied the carrier and the gains were concentrated in one
or two bursts, so this is encouraging but not strong replication.

## Exact scope

The guarded run completed all 42 maps and 336 scored lanes:

| Component | Count |
| --- | ---: |
| Single-scale maps | 12 |
| Scale maximum | 2 |
| Soft log-mean-exp selection | 6 |
| Adjacent-scale agreement | 10 |
| Compact-minus-broad contrast | 6 |
| Center-versus-annulus divergence | 6 |
| Native lanes | 168 |
| Identical-proposal lanes | 168 |

Every lane was evaluated at budgets 20, 40, 58, 80, and 100. Selection used
three bursts and evaluation used the held-out fourth burst. The 50/50 quota
and the source-union oracle were additional audits.

## Primary held-out results

| Estimand | Budget 20 | Budget 40 | Budget 58 | Budget 100 |
| --- | ---: | ---: | ---: | ---: |
| Native carrier | 0.541 | 0.657 | 0.703 | 0.726 |
| Selected native multiscale lane | **0.583** | **0.677** | 0.702 | 0.702 |
| Carrier on frozen v5 union | 0.459 | 0.591 | 0.709 | 0.779 |
| Selected multiscale score on same union | **0.471** | **0.639** | **0.759** | **0.816** |
| 50/50 carrier-feature proposal quota | 0.542 | 0.668 | 0.722 | 0.751 |

Native selection beat the carrier in three of four bursts at budget 20. At
budget 40 it won two, tied one, and lost one. At budget 58 it won two, tied
one, and lost one; the large loss on burst 1 removed the average gain. On the
identical proposal union, budget-40 gains occurred in two folds and budget-58
gain occurred in only one fold. These fold patterns matter more than the
rounded means on a dataset with only four bursts.

Quiet-threshold recall decreased for selected native and identical-proposal
lanes. The learned score therefore did not establish a threshold-level
selectivity improvement. Sparse labels do not permit ordinary precision,
because an unmatched candidate can be real unannotated activity.

## What patch size did

Standalone native maps show a clear decline beyond the compact supports:

| Patch / bandwidth | Budget 20 | Budget 40 | Budget 58 |
| --- | ---: | ---: | ---: |
| 5 / 0.5 | **0.617** | 0.705 | 0.745 |
| 7 / 0.5 | 0.605 | 0.707 | 0.718 |
| 7 / 1.0 | 0.580 | **0.708** | 0.731 |
| 9 / 0.5 | 0.610 | 0.696 | 0.719 |
| 11 / 1.0 | 0.548 | 0.634 | 0.634 |
| 13 / 1.0 | 0.518 | 0.601 | 0.601 |
| 15 / 1.0 | 0.438 | 0.510 | 0.510 |

Thus the earlier 7-pixel support was near the useful compact range, but 5
pixels deserves equal status. Eleven to fifteen pixels blur local evidence
when used directly. They remain useful as contextual references: the strongest
post-hoc native configurations were 7-minus-15 contrasts, and the most stable
held-out adjacent agreement used 5 and 7 pixels.

## Family-level held-out result

These rows restrict selection to standalone members of one family:

| Family | Budget 20 | Budget 40 | Budget 58 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Single scale | 0.585 | 0.683 | 0.723 | compact 5-pixel map usually selected |
| Scale maximum | **0.608** | 0.712 | 0.712 | strongest tightest-budget family |
| Soft scale selection | **0.608** | 0.679 | 0.690 | similar top-20 behavior, weaker later |
| Adjacent agreement | 0.595 | 0.700 | **0.739** | best broad-budget stability |
| Compact-minus-broad contrast | 0.566 | **0.723** | 0.735 | best budget-40 family |
| Center-annulus | 0.480 | 0.612 | 0.682 | not competitive as a native detector |

There is no single universal winner. Max/soft selection helps place a compact
response in the first 20 positions; contrast and adjacent agreement retain
more labeled events as the budget expands. Exact center-annulus divergence is
scientifically sensible but did not survive this evaluation as a standalone
proposal score.

Carrier boosts were not reliably superior. On the identical proposal union,
the best all-family cross-fit selected a mixture of standalone and boosted
lanes, but its fold choices were unstable. A future compact confirmation should
freeze one or two feature formulas rather than widen boost weights.

## Proposal ceiling and resource result

Adding the multiscale proposal sources raised the mean per-burst budget-58
oracle ceiling from 0.9186 to 0.9311, a 0.0125 absolute gain. The protected
50/50 proposal quota improved the native carrier by only 0.0016, 0.0103, and
0.0186 at budgets 20, 40, and 58. This says the new sources contain a little
unique coverage, but reserving half the budget for them is too blunt.

The run took 66.65 seconds and peaked at 4.57 GiB host RAM. The complete raw
map bank processed 223 calibration/event frames at 32.18 frames/s with a batch
size of four. This demonstrates bounded offline throughput on the RTX 4070
SUPER; it does not yet prove single-frame causal latency for the fused score.

## Scientific conclusion

The entropy/ITL direction remains valuable when formulated as quiet-relative
distributional change. The new result refines the architecture:

```text
compact quiet-relative CS response (5--7 px)
  + broad contextual CS response (approximately 15 px)
  -> agreement or compact-minus-broad score
  -> bounded proposal/ranking lane
  -> immutable activity carrier retained for the scientific trace
```

Do not replace the carrier with a large-support information map. Do not infer
precision from lower candidate counts. The next justified experiment is a
small preregistered confirmation of compact 5-pixel CS, 5/7 agreement, and one
7-minus-15 contrast on a bounded exhaustively annotated field. That field is
needed to determine whether broad-context subtraction removes noise/artifact
or suppresses real but unlabeled neurons.

## Artifacts

- Output root:
  `Outputs/HierarchicalParzenICA/spon_ca_burst_multiscale_information_v1`
- Full metrics: `metrics.json`
- Native and identical-proposal screens: `evaluation/`
- Sixteen-page feature/burst diagnostic TIFF:
  `diagnostics/top_multiscale_maps.tif`
- Frozen manifest:
  `examples/spon_ca_burst_multiscale_information_v1.example.json`
- Workflow contract:
  `docs/workflows/spon_ca_burst_multiscale_information.md`

The diagnostic TIFF contains four selected pooled feature maps across four
bursts; it is not a 560-frame feature video.
