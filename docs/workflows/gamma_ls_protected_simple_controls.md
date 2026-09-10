# Protected Gamma-LS Simple-Control Comparison

## Why this comparison is required

The completed support-sufficiency screen executed two modern, simpler controls:

- `signed_square_annulus_ls_h11_g3`;
- `maintained_positive_box_cfar_h11_g3`.

Their `eligible_primary=false` status only kept them out of the radial-support
selector. It did not establish lower detection performance. All 24
representation-by-fold-by-quiet-swap contrast cells were positive for each
control in the completed support artifact. The repeated single-frame p50 was
approximately 1.658 ms for the signed square control and 0.214 ms for the
maintained positive box control, versus 1.805 ms for the radial h11 reference.
The box control is therefore a particularly important speed/recall baseline.

The legacy epsilon-64 anchor remains outside this follow-up: its support-screen
contrast was not competitive, whereas the two modern controls have a concrete
label-free signal and/or latency rationale for protected evaluation.

## Frozen comparison

The executor is
`neurobench.experiments.gamma_ls_difference.protected_controls`. It must run
only after the adjacent-frame protected radial artifact is complete and
indexed. It requires an explicit radial `context_role`; for the current support
artifact this is `size_sufficient_candidate`. This prevents an accidental
comparison against an original or larger-support sensitivity lane.

For each of Raw, signed adjacent difference, and energy-normalized adjacent
difference, the executor freshly computes both controls. It uses the same four
leave-one-burst-out folds, reversed 50-frame quiet splits, empirical quiet NMS
burdens 0.25/0.5/1/2/5, NMS distances 4/6/8 px, burst threshold-occupancy
ranking, candidate budgets 20/40/58/80/100, 6-px one-to-one matching, protected
v1 79-occurrence primary cohort, and descriptive v7 106-occurrence cohort as
the radial protected run.

The full control score arrays are SHA-256-bound by representation and quiet
swap. Candidate-universe, candidate-score-stream, threshold, timing, and TSV
hashes are frozen in `candidate_seal.json` before either sparse-positive table
or the radial label-derived match table is parsed.

## Label and claim boundary

The existing eligibility preflight inspected annotation content and produced a
coordinate projection. Accordingly, this workflow must not be described as
end-to-end label blind. The narrower, accurate statement is:

> Control candidate construction and model scoring did not access annotation
> content after the eligibility preflight; candidates were sealed before the
> protected join.

Protected v1 supports known-positive recall and paired, canonical-identity
clustered bootstrap contrasts. Unmatched candidates are unknown, not false
positives; precision is not identified. V7 is descriptive sensitivity only.
The reported 0.02 point-margin diagnostic was not predeclared in the parent
campaign and cannot by itself select a deployment pipeline. Owner review must
consider recall, held-out quiet burden, and latency together.

The metric executor copies the preflight projection overlay and writes a
scientific-audit renderer contract, but marks the full three-section media set
pending. Metric completion is not scientific-audit completion or paper
promotion.

## Exact command

Wait until the protected radial destination (not its hidden work directory)
contains a completed `artifact_index.json`, then run from the repository root:

```bash
.venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.protected_controls \
  --config examples/spon_ca_burst_gamma_ls_difference_ablation_v1.example.json \
  --preflight Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_difference_ablation_v1_preflight_20260908_r9 \
  --support-screen Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_support_sufficiency_v1_gpu_screen_20260908_r2 \
  --protected-output Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_difference_ablation_v1_protected_20260908_r2 \
  --artifact-dir Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_protected_simple_controls_v1_20260908_r1 \
  --radial-context-role size_sufficient_candidate \
  --device cuda:0
```

The executor fails closed on an existing output, a missing/partial protected
artifact, config/source/implementation drift, support or protected artifact
hash drift, an incorrect radial role, an active competing Gamma-LS process,
insufficient disk/VRAM, and any candidate-seal mismatch. A failed work
directory is not resumable or promotable; diagnose it and use a new output
root.
