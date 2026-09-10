# Spon Ca Burst Gamma-LS deployment-context freeze

**Frozen:** 2026-09-08, after the coordinate-free support screen completed and
before the protected runner joined either sparse-positive table.

## Purpose

The outer-fold support experiment legitimately uses a different context in
each held-out fold. Full-recording application, sustained streaming, and the
independent-recording sensitivity check instead require one fixed radial
Gamma-LS context. This note freezes that context without using sparse-positive
coordinates, identities, recovery metrics, or the independent recording.

## Frozen selection rule

1. Read each fold's `training_best_context` from the completed support screen.
2. Vote on the exact tuple `(half width, guard, shape, mode fraction, support,
   padding)`; each outer training fold has one vote.
3. Select the plurality tuple. Resolve a vote tie by lower repeated p50 Gamma
   latency, then smaller support width, then lexicographic context ID.
4. Do not revisit the context after protected or independent labels are read.

The four fold-local winners are:

- fold 1: `h15, g7, n9, mode_fraction=0.5`;
- fold 2: `h15, g7, n9, mode_fraction=0.75`;
- fold 3: `h15, g7, n9, mode_fraction=0.5`;
- fold 4: `h15, g7, n9, mode_fraction=0.5`.

The frozen deployment context is therefore:

```text
context_id: support_support_a_h15_g7_n9_m0p5
support: radial disk, 31 x 31 pixels (half width 15)
guard radius: 7 pixels
Gamma shape: 9
nominal mode radius: 7.5 pixels
boundary: zero outside the field with valid-reference renormalization
epsilon: 1e-6
```

The context is a single selected scale, not multiscale inference. Its scale
floor remains training-quiet fitted by the consuming experiment.

## Bound inputs

- `fold_contexts.json` SHA-256:
  `2bef8770c5b3dcb9cd0d344441d3569f9b0397e711edc73c03802a5cca55d243`
- `validation.json` SHA-256:
  `47a7eaf52c893d5a3e926fea5478f6a41b83748bbdc353e1b194985a925247dd`
- `artifact_index.json` SHA-256:
  `ae0337597ccd7b21a7e92f910f568809a57895b68df752f73940b7b830db7e83`

All three files are under
`Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_support_sufficiency_v1_gpu_screen_20260908_r2/`.

## Claim boundary

This freeze selects an application context only. It does not establish
protected recall, sufficiency, precision, 1-kHz readiness, cross-recording
generalization, or superiority over the square-reference controls. Those are
separate downstream tests.
