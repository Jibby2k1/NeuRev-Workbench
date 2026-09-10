# Spon Ca Burst Gamma-LS conditioning protected comparison v1

## Question

Does fold-local conditioning/scale-floor selection materially change protected
candidate-budget recall curves relative to the historical
`sigma=1, alpha=0.4, percentile=10` anchor?

## Frozen comparison

The protected manifest pins the completed conditioning preflight artifact index
(`ef1a79c...f494`), preflight payload (`740e5141...711`), completed screen
artifact index (`71b55a0b...efb`), and canonical selection seal
(`3011bdbc...816`). A different internally consistent screen/preflight pair is
therefore not eligible for this comparison.

The only compared roles are `historical_anchor` and
`training_pareto_selected` from the completed conditioning screen. They are
evaluated separately for raw, signed adjacent difference, and energy-normalized
adjacent difference in each of four outer held-out bursts. Each role retains
the fold's already frozen radial support candidate. If the selected setting is
the anchor, dense computation may be reused but both semantic roles remain in
the paired output with an exact zero contrast.

The quiet calibration is reversed-half crossfit. Empirical burdens are
`{0.25, 0.5, 1, 2, 5}` NMS peaks per duration-matched pseudo-burst. Spatial NMS
distances are `{4, 6, 8}` pixels, with 6 primary. Candidate budgets are
`{20, 40, 58, 80, 100}` per burst, with B58 retained as the named point.
Temporal aggregation is threshold occupancy, not temporal max pooling.

## Seal and label boundary

The executor first verifies the complete indexed conditioning screen, its
selection seal, and the exact sensitivity preflight hash recorded by that
screen. The completed preflight's historical source-code hashes are retained
as provenance, while current downstream code and current CUDA identity are
checked separately; ordinary source maintenance cannot retroactively alter a
valid completed screen. It then produces and writes every candidate and threshold row. A
candidate seal binds those tables, upstream hashes, the executor, and the
frozen settings before either label table is reopened, rehashed, or parsed by
this executor. (The earlier base preflight hash metadata is available but the
protected executor does not reopen those source bytes before sealing.) The v1
and v7 tables are opened only after the on-disk seal is reverified. All proposals are
`unknown_candidate` before the join; candidates outside the sparse positive
set remain `unknown_not_negative`, so precision is not identified.

The v1 comparison uses a paired 2,000-replicate bootstrap over exactly 26
canonical identity clusters. V7 is a descriptive sensitivity analysis only.
For each representation, a material curve improvement requires the crossfit-
average NMS-6 budget-curve AUC delta to be at least 0.02 with its 95% paired
cluster-bootstrap interval above zero at all five quiet burdens. The mirrored
rule defines material degradation; otherwise the conclusion is no consistent
material change. Individual burden/NMS rows remain available and are not
silently pooled.

## Claim boundary

This protected stage estimates sensitivity to known sparse positives. It does
not identify proposal precision, exhaustive event counts, or neuron identity
for unmatched candidates. A completed metric artifact remains scientifically
provisional until the candidate-surrogate montage, expert ledger, and failure
taxonomy required by the audit standard are completed.

## Execution

No protected GPU result is implied by this plan. Using the manifest-pinned
conditioning preflight and completed screen:

```bash
.venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.conditioning_protected \
  --config examples/spon_ca_burst_gamma_ls_conditioning_protected_v1.example.json \
  --conditioning-preflight Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_conditioning_sensitivity_v1_preflight_20260908_r1 \
  --conditioning-screen Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_conditioning_sensitivity_v1_gpu_20260908_r1 \
  --output Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_conditioning_protected_v1_gpu_20260908_r1
```
