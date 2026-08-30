# Pipeline diagnostic audit

The table contains 318 rows at occurrence x stage grain (106 occurrences x 3 stages), with no duplicate composite keys.

## Available now

- Raw/ICA/LS center traces: **observed** — saved aligned movies
- Native amplitude, noise, timing, area: **derived** — computed from center traces
- Annulus coupling and center-annulus contrast: **derived** — 3-6 px annulus
- Spatial footprint radius and peak offset: **derived** — 21x21 event-minus-baseline crop
- Raw saturation: **derived** — uint16 ceiling 4095
- Candidate match/rank: **partial** — available for original 27-ROI diagnostic set only
- Local-standardization denominator: **derived** — exact bounded reproduction from frozen fit, causal pre-roll, and saved floor
- ICA component activations/contributions: **observed** — exported from frozen fit at all 50 sites
- Mixing/unmixing matrices: **observed** — exported directly from frozen temporal fit
- Embedding inversion residual: **derived** — numerical transform audit; not denoising error

## Missing model diagnostics

- Motion/registration residual: no motion field saved with this output
- Independent-recording validation: single recording in current analysis

## Interpretation boundary

The table supports signal, timing, neighborhood, spatial-footprint, stage-transition, component-attribution, and true LS-denominator diagnostics within this recording. The embedding residual tests numerical inversion only; motion correction and independent-recording validation remain unavailable.
