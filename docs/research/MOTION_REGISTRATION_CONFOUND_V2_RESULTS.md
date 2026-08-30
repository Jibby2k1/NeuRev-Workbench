# Motion and registration-confound audit v2 results

## Outcome first

The corrected matched-support screen does not provide a usable local motion
field. All `12 / 12` frozen windows require reliability review, and none of the
14 grouped associations survives Benjamini–Hochberg correction. These results
block motion-based attribution of the current waveform or residual behavior;
they do not show that motion is absent or irrelevant.

This is a label-free, non-claim-bearing engineering screen. Translation-like
phase estimates cannot identify biological motion, and the run does not apply
motion correction to a scientific endpoint. `NREV-EXP-0025` remains `draft`,
`not_evaluated`, and evidence tier `none`, with no evidence capsule.

## Canonical run and correction history

- Canonical run: `NREV-RUN-EXP-0025-SCREEN-20260830-E`
- Runner SHA-256:
  `bfb7d0439a4f51c8e4b1caab91a820d2ba1c06c26789a8fd1190ea31ab11ae1a`
- Run-start dirty-content digest:
  `0ce57730ee8cb63da41178b6f43ee83cee2bad52a5b41c0a78a47e2026e901f3`
- Resolved-config SHA-256:
  `23636ab6fc632f45cb92b4e33fada2b543133160a9dae0fdd56a4dd4c5111403`
- Run-provenance SHA-256:
  `fe000ca6d4a54d8415c406decf06df5193949a36642a17a0a883b0e288cbd28f`
- Artifact-index SHA-256:
  `bf7337aec92996ca0ed9179f8063397b0a0673f34d7493b78a1a2d170f94d8e2`
- Artifact-set SHA-256:
  `a7983ee8666293f485940cdfc77198ccb4c5680102ff7633799ffd2432840082`

Only Run E is registered. A had implementation drift, B lacked authoritative
time/command provenance, C lacked the schema-required content digest, and D
used mismatched spatial support for raw versus registered MAD. Those roots are
retained locally as provisional engineering history but are not copied into
the canonical provenance tree.

Run E was planned at `2026-08-30T14:30:42.965054Z`, started at
`2026-08-30T14:30:42.965253Z`, and recorded numeric/report completion at
`2026-08-30T14:30:51.308353Z` after `8.3431` seconds. The timestamp semantics
do not claim that final index serialization was included.

## Exact matched-support validation

The primary panel contains 12 windows, four recording groups, and 372
adjacent-frame pairs. For each of the 308 valid estimated shifts, raw and
registered MAD/RMS were recomputed on the exact same conservative
shift-dependent interior. All saved values, bounds, pixel counts, fractions,
ratios, and reductions reproduced with maximum absolute error `0.0`. The 64
invalid shifts correctly had empty matched-support fields.

The median retained interior was `0.67310` of the 64-by-64 window. Median
matched-support raw MAD divided by full-frame raw MAD was `1.0000`, which shows
no aggregate support-induced scale shift at the median but does not make the
supports interchangeable pair by pair.

## Reliability result

| Diagnostic | Result |
| --- | ---: |
| Windows requiring local reliability review | `12 / 12` |
| Large tile/global disagreement | `12 / 12` |
| Less than 10% matched-support reduction | `11 / 12` |
| Low valid-pair fraction | `4 / 12` |
| Search-boundary concentration | `4 / 12` |
| Median registered/raw difference-MAD ratio | `0.95209` |
| Ratio range | `[0.89942, 1.00169]` |
| Median two-step cycle-closure p95 | `11.48113 px` |
| Median tile/global disagreement p95 | `12.25522 px` |
| Median local/full-field p95 ratio, shared `060126` windows | `19.41884` |

The median ratio corresponds to only about `4.79%` registered-difference
reduction. One window crossed the 10% reduction trigger, but it still failed
other reliability checks. A median local translation summary of `3.38383`
native pixels and a maximum window p95 of `9.11106` pixels are retained as
translation-like diagnostics only; the disagreement and cycle-closure results
make them unsuitable as motion fields.

The package retains one reciprocal forward/back symmetry-trigger field for
traceability. Because reciprocal phase estimates are conjugate by
construction, that field is not interpreted as independent reliability
evidence and is not used to rescue or condemn the local field.

## Association result

All 14 fixed grouped association rows were complete, with `1,296` exact
within-recording permutations where supported. Zero passed BH at `0.05`.

The smallest raw p-value was `0.0138889` for tile/global disagreement versus
JEPA background RMS ratio, with within-recording rank correlation `0.875`; its
BH-adjusted q-value was `0.194444`. The corrected matched-support
registered/raw ratio versus raw-HC source-on micro recall had within-recording
correlation `-0.80178`, raw p-value `0.0308642`, and q-value `0.216049`.
Neither is multiplicity-supported, and four recording groups do not establish
generalization.

The prior EXP-0028 acquisition screen covered 11 `060126` recordings and
flagged one for review, but the endpoint association panel here contains only
three `060126` recordings plus Spon. No endpoint values are imputed for the
other eight recordings, and the two spatial/temporal sampling schemes are not
treated as equivalent.

## Feature and sensor findings

The requested relationship to validated carrier/coherence/recurrence features
could not be estimated without changing the support. Spon feature arrays begin
after every frozen Spon window, and the `060126` recordings have no matched
exported feature traces. This is an explicit non-join result, not missing-data
imputation and not evidence of no relationship.

Maximum occupancy of the stored `uint16` high-code rail was zero, but two
windows reached stored code 4095. Detector and ADC metadata are unavailable,
so neither observation resolves the analog rail. The result must not be stated
as “no saturation.”

## Scientific decision boundary

The screen establishes three bounded facts:

1. identical-support registration accounting is now exact;
2. the local translation-like estimates fail the frozen reliability review in
   every window; and
3. no tested motion/reliability association survives multiplicity correction.

It does not establish absence of motion, causal irrelevance of motion,
biological motion magnitude, nuisance-robust representations, neuron identity,
precision, or generalization. Required audit media are missing, so scientific
completion and promotion remain false.

The practical consequence is to keep the current residual program on hold,
not to regress the present data with these unreliable fields. A future motion
program needs a different estimator or acquisition metadata plus disjoint
validation before it can support nuisance adjustment.

## Portable provenance

Small exact Run-E configuration, provenance, status, summary, validation, and
primary tables are retained under:

```text
research/run-provenance/NREV-RUN-EXP-0025-SCREEN-20260830-E/
```

Raw recordings and the large pair-level table remain outside Git. No claim or
evidence capsule is created.
