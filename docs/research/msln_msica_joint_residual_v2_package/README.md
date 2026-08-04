# MSLN/MS-ICA joint-residual v2 paper and presentation package

## One-sentence result

A causal joint spatiotemporal normalization produced visually coherent activity
maps, and broad-context CS-Parzen persistence retained 58/79 sparse known
positives at a fixed 232-candidate guardrail, but bootstrap instability and
non-exhaustive labels keep the result exploratory.

## Recommended interpretation

The most useful contribution is the representation and its visual behavior:

- a causal reference excludes the current frame, the protected spatial core,
  and the most recent temporal guard frame;
- signed `Zst` preserves activation direction, while persistence and innovation
  remain separate rather than being collapsed into a single score;
- the broad `S15/G3/T31` persistence lane is the review-leading result;
- `S15/G3/T23` is a close broad alternative and `S5/G1/T15` is a useful compact
  contrast;
- the videos show morphology and temporal behavior that scalar recall cannot.

Do not describe the output as cleaned fluorescence, recovered biological
sources, improved precision, or a confirmed replacement for Raw Direct. Sparse
labels are positives only, the fixed-budget comparison is not protocol-identical
to Raw Direct, and the broad ICA angles were unstable across block bootstraps.

## Package contents

- [PAPER_AND_SLIDES.md](PAPER_AND_SLIDES.md): concise manuscript and slide plan.
- [ASSET_MANIFEST.md](ASSET_MANIFEST.md): exact video/figure placeholders,
  source locations, and ready-to-use captions.
- [assets/README.md](assets/README.md): manual asset-placement instructions.

The generated source root is intentionally not versioned:

```text
Outputs/HierarchicalParzenICA/spon_ca_burst_joint_msln_residual_sweep_v2
```

Before circulation, replace every `PLACEHOLDER` entry in the asset manifest and
copy the selected binaries into `assets/` using the specified filenames. Keep a
portable copy of the four videos outside Git if the remote rejects their size.

## Publication readiness

Ready now: method diagram, visual-results narrative, qualitative comparison,
CUDA/resource implementation note, exploratory sparse-positive guardrails, and
limitations.

Still needed for a confirmatory claim: exhaustive bounded-field review or
labels, fold/seed confirmation, a protocol-identical operating-point comparison,
and a stability analysis that either resolves or bypasses the broad ICA angle
variability.
