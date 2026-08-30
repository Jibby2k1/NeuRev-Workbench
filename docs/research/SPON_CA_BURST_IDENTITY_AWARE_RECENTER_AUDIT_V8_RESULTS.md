# Spon Ca Burst canonical-v8 identity-aware recentering audit

## Version boundary

Canonical v8 is an interpretation update over the immutable canonical-v7
measurement cohort. It does not replace the v7 adjudication, alter ROI centers,
or promote automatically selected coordinates to biological labels. The v8
addition is a visually reviewed, identity-aware audit of the 12 occurrences
missed by all three quantitative lanes at the frozen B58 operating point.

## Audit design

For each miss, a fixed plus-or-minus six-pixel neighborhood was searched using
the median within-patch response percentile across Raw fluorescence, CS-Parzen
ICA, and CS-Parzen ICA followed by local standardization. Response was defined
as the event maximum minus the median of the preceding 15 frames. The original
center remained immutable. Each consensus candidate was then compared with all
other labeled geometries within six pixels and classified as identity-clear, a
collision with the same canonical identity, or a collision with another
canonical identity. Six-panel videos preserved synchronized pipeline-stage
images and exact-center traces. All 12 outputs passed automated media checks
and visual inspection.

## Result

Of the 12 all-lane misses, two consensus candidates collided with another
geometry belonging to the same canonical identity, six collided with a
different canonical identity, and four were identity-clear within six pixels.
Thus, eight of 12 apparent recentering improvements cannot be interpreted as
safe coordinate corrections: they move the measurement toward an existing
labeled geometry.

The two reviewer-suggested shifts were spatially supported but both crossed an
identity boundary. The ROI 23 suggestion approached ROI 26, and the ROI 12
consensus landed exactly on `roi_new_002` while the suggested coordinate was
2.24 pixels from that geometry. ROI 10 and ROI 15 were confirmed as spatially
adjacent representations of the same canonical identity. All three missed ROI
11 occurrences shifted toward `roi_new_001`, consistent with a real temporal
signal embedded in an overlapping or ambiguously assigned footprint.

ROI 19, ROI 22, ROI 25, and ROI 27 were identity-clear under the six-pixel
guard. This does not make them validated coordinate corrections. ROI 22
remained noise-dominated; ROI 19 remained plausible but noise-limited; ROI 27
retained a credible transient in a crowded region consistent with non-maximum
suppression; and ROI 25 remained a noisy ranking case.

## Scientific interpretation

The primary failure decomposition is therefore identity-aware rather than
purely localization-based. A stronger trace after a small spatial shift is
insufficient evidence of a better center because crowded fields can transfer
the measurement to a neighboring labeled source. Recentered-response gain must
be paired with an identity-collision guard, footprint evidence, and bounded
review. The audit strengthens the case for separation of proposal, ranking,
non-maximum suppression, and identity-resolution errors and weakens any claim
that most all-lane misses could be repaired by local coordinate optimization.

## Evidence and validation

- Reviewer artifact root:
  `external://v7_b58_identity_aware_recenter_audit_v1`
- Itemized results: `recenter_summary.tsv`
- Audit report: `REPORT.md`
- Validation: `validation.json` (`status: passed`)
- Reproducible renderer:
  `external://render_v7_identity_aware_recenter_audit.py`

Here, `external://` denotes the project-adjacent review bundle supplied outside
the Git repository; its local root is intentionally not part of the record.

No authoritative coordinate, canonical identity, or inclusion label was
changed by this audit.

## Updated metric boundaries

- Strict frozen B58 recall remains `94/106 = 88.7%`.
- Canonical-collapsed sensitivity remains `93/102 = 91.2%`.
- Excluding the eight identity-collision occurrences yields an
  audit-conditioned `94/98 = 95.9%` sensitivity among currently
  identity-resolved occurrences. This is not replacement detector recall.
- If all four identity-clear hypotheses were independently validated and
  recovered, the strict-recall ceiling would be `98/106 = 92.5%`.

## Prioritized feature inspections

1. Replace the binary recovery target with recovered, identity-clear miss, and
   identity-ambiguous collision outcomes.
2. Measure displacement-vector consistency across Raw, ICA, local
   standardization, and bursts.
3. Compare exact-center, fixed-disk, and footprint-weighted trace extraction.
4. Quantify footprint eccentricity, solidity, convexity deficit, radial
   asymmetry, boundary sharpness, and multi-lobe or crescent structure.
5. Measure local target-to-neighbor peak ratio, saddle depth, NMS suppression
   margin, and response-field topology.
6. Regress original and candidate traces against nearby labeled traces and test
   whether nonnegative mixtures explain the apparent recentering gain.
7. Evaluate whether these features stratify the four identity-clear cases
   without tuning a new detector on four observations.

## Automated feature-inspection result

The validated v2 panel is stored at
`Outputs/NeuronIdentifiability/spon_ca_burst_identity_aware_feature_inspection_v8_v2`.
It fits no classifier and reuses the validated 12-case visual audit.

- Median cross-stage offset-direction cosine was 0.972 for the eight collision
  misses and 0.448 for the four identity-clear misses.
- Median offset-vector spread was 1.13 versus 1.83 pixels.
- In local-standardized traces, median candidate-neighbor partial correlation
  given the original trace was 0.474 versus 0.182; median nonnegative neighbor
  weight was 0.460 versus 0.187.
- Median eccentricity was 0.774 versus 0.810 and median solidity was 0.781
  versus 0.714, so morphology alone did not cleanly separate the groups.
- Response-weighted extraction increased median local-standardized peak/MAD
  from 4.06 to 22.99 in collision cases and from 3.97 to 12.67 in clear cases.
  This is same-event extraction sensitivity, not prospective detector recovery.
- ROI 19 is the cleanest identity-clear localization hypothesis by offset
  consistency and low leakage. ROI 25 is directionally consistent but
  morphologically irregular. ROI 22 and ROI 27 have inconsistent stage offsets.

## Comprehensive frozen validation suite

The follow-up suite is stored at
`Outputs/NeuronIdentifiability/spon_ca_burst_identity_aware_validation_suite_v2`.
It covers cross-burst transfer, leave-one-stage-out localization, pre-event
footprint freezing, multi-neighbor residualization, lag structure,
identity-exclusion localization, sector/topology features, recovered and quiet
negative controls, frozen budget/radius counterfactuals, and exact
occurrence-grain permutation tests.

- Leave-one-stage-out candidate gain was 1.54 for collision cases and 1.31 for
  clear misses; the exact descriptive permutation value was 0.371.
- Multi-neighbor fits explained median 52.3% of collision-candidate variance
  versus 8.8% for clear misses. Residual peak/MAD was 4.02 versus 5.48.
- Median best-neighbor lag was zero for collisions and two frames for clear
  misses, with median correlations 0.593 versus 0.235.
- Pre-event-frozen footprints retained median peak/MAD 11.98 in collisions and
  9.51 in clear misses, avoiding target-event footprint selection.
- Recovered sites moved a median 2.24 pixels under unconstrained response
  maximization, whereas quiet pseudo-events moved 5 pixels. Recentring is
  intrinsically biased and displacement alone is not a failure diagnosis.
- No identity-clear miss had a frozen candidate within eight pixels by budget
  100 in any of the three lanes. These four cases are proposal-absence cases at
  the audited frozen outputs, not merely B58 ranking failures.
- Cross-burst transfer was estimable for 34 collision source-target pairs but
  only one clear pair, so it cannot compare the groups.
- None of the three exact eight-versus-four descriptive tests was small
  ($p=0.371$, $0.169$, and $0.308$); the directional patterns remain
  exploratory.
