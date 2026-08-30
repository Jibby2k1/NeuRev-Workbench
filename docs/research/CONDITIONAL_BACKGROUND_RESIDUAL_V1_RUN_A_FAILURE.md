# Conditional-background residual v1 run-A failure

## Outcome first

`NREV-RUN-EXP-0029-SCREEN-20260830-A` is a preserved failed engineering run,
not a residual-method result. Its exact raw-HC continuity guard detected that
the v1 runner supplied normalized pixels to a parent endpoint originally
evaluated in native raw `float32` units. The run stopped before any JEPA or
random-residual endpoint comparison was accepted.

The failure does not change `NREV-EXP-0029` from `draft`, `not_evaluated`, and
evidence tier `none`. It supports no claim, produces no evidence capsule, and
does not resolve a scientific or descriptive benefit threshold.

## What passed before the stop

The bounded run passed its registered preflight, parent/input and implementation
hashes, resource checks, blind-tube geometry checks, and matched 500-step
decoder training. These facts establish where the failure occurred; they do
not convert the partial package into endpoint evidence.

The raw-HC regression then matched 104 of 108 fixtures at complete serialized
`source_on_recovery` plus `intervention_recovery` identity and reported four
intervention mismatches. The source-on results were exact for all 108 fixtures.

## Exact mismatch diagnosis

Independent reconstruction showed that intervention recall, recovered-source
identity, matched indices, and localization-error arrays were unchanged for all
108 fixtures. Candidate coordinate lists differed in 53 fixtures, while only
four changed cardinality:

| Fixture | Parent candidates / unmatched unknown | Normalized-v1 candidates / unmatched unknown | Intervention recall | Localization error |
| --- | ---: | ---: | ---: | ---: |
| `060126_10_rest__window_2_median_mad__sources_1__seed_3102` | `2 / 1` | `3 / 2` | `1.0` in both | `1.0 px` in both |
| `060126_12_left__window_1_low_mad__sources_1__seed_3102` | `4 / 3` | `3 / 2` | `1.0` in both | `1.0 px` in both |
| `060126_15_right__window_3_high_mad__sources_1__seed_3102` | `3 / 2` | `4 / 3` | `1.0` in both | `0.0 px` in both |
| `spon_ca_burst_3_hindbrain_to_tail_488_20ms__window_1_low_mad__sources_1__seed_3103` | `4 / 3` | `3 / 2` | `1.0` in both | `1.0 px` in both |

For each of these four fixtures, source-on recall was `0.0` and source-on
candidate count was four in both domains. A native-unit reconstruction matched
both parent recovery objects for every fixture: 216 of 216 exact.

The discrepancy was therefore an endpoint-domain and exact-serialization
continuity failure, not evidence that recall or localization changed.

## Corrective action

The versioned
[v1.1 protocol](../workflows/conditional_background_residual_v1_1.md) assigns
the raw-HC comparator to native raw `float32` pixels and leaves both learned
conditional-residual arms in frozen-training-normalized pixels. It registers a
new run and output root:

```text
NREV-RUN-EXP-0029-SCREEN-20260830-B
Outputs/NeuronIdentifiability/NREV-EXP-0029/runs/
  NREV-RUN-EXP-0029-SCREEN-20260830-B
```

The correction does not relax the guard. Run B still requires complete exact
identity for 108 fixtures and 216 source-on/intervention `RecoveryResult`
objects at numeric tolerance zero before residual interpretation. Candidate
coordinate, cardinality, or unmatched-count exceptions remain forbidden.

## Provenance boundary

Small sanitized byte-exact artifacts from the failed partial run are retained
under
`research/run-provenance/NREV-RUN-EXP-0029-SCREEN-20260830-A/`, including the
status, preflight, resolved configuration, decoder metrics, blind-tube geometry
validation, and raw-HC regression record. Checkpoints, raw video, caches, and
the full partial output are not committed.

This record is an engineering failure diagnosis. Run B remains planned and has
not been executed by this amendment.
