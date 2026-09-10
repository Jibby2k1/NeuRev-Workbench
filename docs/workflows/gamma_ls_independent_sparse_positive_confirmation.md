# Gamma-LS independent sparse-positive confirmation

This optional experiment is run only after the protected Spon Ca Burst analysis
has selected one deployment representation and one radial Gamma-LS context. It
does not select or tune that pipeline on `15 right`.

## Claim and access boundary

The prior independent-recording eligibility preflight inspected annotation
content to determine that `15 right` was usable and that `6 left` was not. This
workflow is therefore **not end-to-end label blind**. Its narrower guarantee is:

1. verify the hash-bound post-protected selection, eligibility preflight,
   historical independent contract, cropped movie, crop manifest, and video
   manifest;
2. calibrate from the predeclared first 100 source frames, which are not asserted
   event-free;
3. construct and hash a new Gamma-LS candidate universe and all candidate/block
   scores without opening the annotation manifest;
4. freeze `candidate_score_seal.json`;
5. only then hash, open, and join the annotation manifest.

The older ICA `S6_INDEPENDENT_CONFIRMATION/candidate_universe.tsv` is never used
as Gamma-LS evidence.

## Frozen evaluation

- Recording: `15 right`, 1608 by 512 by 512 uint16, 50 Hz.
- Blocks: 32 complete, non-overlapping 50-frame source blocks (1600 source
  frames); the final eight source frames are dropped. Source frame zero is
  causal warm-up, leaving 1599 scored outputs over zero-based source frames
  1--1599. The first block contains 49 scored representation frames and all
  later blocks contain 50. Only positive intervals overlapping that actual
  scored domain enter the denominator.
- Preprocessing: Gaussian sigma 1 pixel with reflect boundaries, then causal EMA
  alpha 0.4.
- Representation: exactly one post-protected frozen arm. Learned PCA/ICA arms
  additionally require a hash-bound transferable model JSON.
- Spatial operator: exactly one radial-disk Gamma-LS context with
  valid-renormalized-zero boundaries.
- External calibration: 10th-percentile positive local scale and descriptive
  0.9999 score quantile from the first 100 source frames, sampled every four
  pixels spatially. This framewise threshold has no probability-of-false-alarm
  meaning and does not gate the block-LME candidate ranking.
- Calibration overlap: zero-based source frames 1--99 occur in both calibration
  and scored evaluation. This is an explicitly disclosed within-recording
  sensitivity analysis, not an independent held-out test of calibration.
- Per-block evidence: signed Gamma-LS pooled with LME temperature 0.25.
- Candidates: deterministic 6-pixel separated NMS, exactly 200 top-ranked
  block-LME peaks per block. IDs carry the first 12 hexadecimal characters of
  the frozen selection hash; they do not hard-code a method or recording alias.
- Budgets: 10, 20, 58, 100, and 200 candidates per block.
- Match: candidate peak frame inside the imported zero-based inclusive positive
  interval and Euclidean distance at most 6 pixels, with maximum-cardinality
  one-to-one assignment.
- Metrics: known-positive recall and mean reciprocal within-block rank.
- Resource precheck: at least 20 GiB free disk, 8 GiB available RAM, and 8 GiB
  free VRAM before the non-colliding output work directory is created.

Unmatched candidates are unknown, not false positives. This single recording
does not identify precision, specificity, population generalization, or a
unique biological-event count.

## Required post-protected selection file

The runner accepts a small JSON artifact with these fields. Every referenced
file is hash checked. The `candidate_protocol_sha256` value is obtained with
`independent_gamma_protocol_digest()` after the implementation is frozen.

```json
{
  "schema_version": 1,
  "status": "frozen_for_external_sparse_positive_confirmation",
  "selection_scope": "post_protected_primary_decision",
  "decision_made_at_utc": "YYYY-MM-DDTHH:MM:SS+00:00",
  "decision_rationale": "Concise protected-result rationale",
  "representation": {
    "arm": "difference_signed",
    "model": null
  },
  "gamma_context": {
    "context_id": "gamma_h11_g5_n9_m1",
    "support_width_px": 23,
    "shape_n": 9.0,
    "mode_radius_px": 11.0,
    "guard_radius_px": 5.0,
    "support_geometry": "disk",
    "boundary_mode": "valid_renormalized_zero",
    "epsilon": 0.000001
  },
  "protected_evidence": {
    "summary": {"path": "/absolute/protected/summary.json", "sha256": "..."},
    "validation": {"path": "/absolute/protected/validation.json", "sha256": "..."}
  },
  "candidate_protocol_sha256": "...",
  "external_annotation_content_used_for_selection": false,
  "scientific_audit": {"enabled": true}
}
```

For a learned arm, replace `model: null` with a `{path, sha256}` object pointing
to the final transferable model. A fold-specific model must not be silently
promoted as the deployment model.

## Run command

Use a new output directory and the host CUDA runtime:

```bash
NEUROBENCH_DATA_ROOT="${NEUROBENCH_DATA_ROOT:-.}" \
.venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.independent_validation \
  --selection /absolute/path/to/frozen_gamma_external_selection.json \
  --independent-preflight Outputs/ICAWhiteningEvaluation/spon_ca_burst_ica_whitening_evaluation_v1/independent_confirmation_preflight_v1/independent_confirmation_preflight.json \
  --independent-contract Outputs/ICAWhiteningEvaluation/spon_ca_burst_ica_whitening_evaluation_v1/independent_confirmation_contract_v1/contract.json \
  --artifact-dir Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_independent_15_right_v1 \
  --device cuda
```

The metric run emits projection coordinate checks and deterministic audit-input
tables, including a nearest time-eligible candidate row per expert occurrence
kept distinct from the one-to-one metric assignment. Those hooks do not satisfy
the complete media inventory in
`SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md`; the run remains audit-pending until the
expert-only, model-only, and comparison media pass the standard validator.
