# ICA Morphology Test Suite

## Why this was added

The previous ICA tensor result tested stability of temporal site factors. It did
not test the spatial boundary and neuronal-shape information used during visual
review. The statistical suite now treats these as separate questions.

## Three distinct ICA claims

1. **Two-frame temporal ICA:** a change detector nearly equivalent to signed
   temporal difference. Useful for timing and onset evidence; not an independent
   source.
2. **Spatial/dense ICA:** a morphology and boundary-visibility representation.
   This is the appropriate target for the reviewer's visual observation.
3. **ICA tensor factors:** a population-complexity representation. Rank 2 was
   stable, but its individual site factors were not stable enough to name.

## Existing spatial checkpoint

The dense convolutional FastICA plus Wiener lane previously achieved:

- fixed-budget known-positive recall: 0.671;
- median peak retention: 0.975;
- median area retention: 0.952;
- median peak timing error: 0 frames.

It remains exploratory because the complete preservation audit failed,
especially exact semi-synthetic preservation. That failure does not answer the
reviewer-utility question.

## New required spatial tests

- blinded within-observation comparison of Raw only, spatial ICA only, and
  Raw plus spatial ICA;
- boundary resolvability, neuronal-shape resolvability, reviewer confidence,
  and review time;
- center-to-surround contrast, compactness, footprint radius, and translated
  spatial controls;
- results separated by localized-center, membrane/ring, overlapping,
  weak-signal, and no-boundary morphology;
- uncertainty calculated with sites—not frames or repeated observations—as the
  resampling unit.

## Provenance warning

The original hard-ROI review clips showed Raw intensity, fixed pseudocolor, and
positive temporal change. They did not contain a panel explicitly identified
as ICA. Later v7 decisions mention neurons revealed by "pipeline stages," but
the exact displayed panel must be recovered before assigning that benefit to
ICA. Positive temporal change must not be retrospectively relabeled as ICA.

The complete machine-readable contract is
`paper/ICA_STATISTICAL_TEST_SUITE.yaml`.
