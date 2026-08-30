# Source-off predictor feasibility v1 results

## Outcome first

None of the eight tested methods passed the complete source-off-only
feasibility gate; because no subtraction is a reference rather than a
predictor, zero of seven subtractors were eligible. The fixed consequence is
to **not launch a new residual detector experiment from these predictors**.
The closest learned baseline,
rank-eight low-rank AR(1), passed only `5 / 11` checks; the frozen JEPA decoder
passed `0 / 11`.

This is a numerically complete, non-claim-bearing engineering benchmark. It
does not evaluate injected-source or biological-signal preservation and does
not support a denoising, neuron-identity, detection, causality, or
generalization claim. `NREV-EXP-0030` remains `draft`, `not_evaluated`, and
evidence tier `none`, with no evidence capsule.

## Run identity and coverage

- Experiment: `NREV-EXP-0030`
- Canonical run: `NREV-RUN-EXP-0030-SCREEN-20260830-B`
- Provisional unregistered predecessor:
  `NREV-RUN-EXP-0030-SCREEN-20260830-A`
- Runner SHA-256:
  `9192d3bf30b0806027035bfe30bc6609aefa5d3149219a3eaca5b3f364efa4f8`
- Descriptor SHA-256:
  `1d914bf39e6f6b0b14e782ac8d1df482cc9f5e1730d33b33292e8a6d4dbd9214`
- Resolved-config SHA-256:
  `46089c2ea7ce92eae1060743065fd05de1ceefb93ce51396055be3fcbbeed5a2`
- Run-start dirty-content digest:
  `280243c4224546e406798dc928658e2975da8b33dfad425eed1372de13f3415e`
- Artifact-index SHA-256:
  `fafb1067f36f75b498a383cc244117b54cea7b8f504aa41f562a05e4a56f615b`

Run B evaluated all `8` methods on `96` recording-held validation clips and
the exact `12` registered source-off windows after fitting eligible models on
the frozen `512`-clip training bank. All `864` per-clip rows, all prediction
coverage records, `89 / 89` indexed artifacts, and `65,361,962` indexed bytes
reconciled exactly. All 24 audit videos decoded fully with exactly 32 frames.
Frozen numerical and media bytes, hashes, and conclusions are
integrity-checked, but the persisted runtime omits SciPy for the low-rank
randomized SVD, Matplotlib and Pillow for figures/frames, and the ffmpeg/ffprobe
build for MP4 generation and validation. Exact low-rank numerical and media
renderer replay environments are therefore incomplete.

The run was planned for `2026-08-30T14:08:04Z`, started at
`2026-08-30T14:12:32.426827Z`, and recorded the end of numeric, media, and
upstream-integrity work at `2026-08-30T14:15:27.882898Z` after `175.4561`
seconds. Report, provenance, status, index, and package-promotion assembly
followed; the final index was serialized approximately 38 ms later. The end
event is not a report-completion or atomic-promotion timestamp.

## Feasibility-gate result

| Method | Checks passed | Complete gate |
| --- | ---: | --- |
| No subtraction | `9 / 11` | fail |
| Last frame / ZOH | `2 / 11` | fail |
| Causal EMA, `alpha=0.25` | `2 / 11` | fail |
| Causal median, history `7` | `4 / 11` | fail |
| Per-pixel AR(1) | `2 / 11` | fail |
| Low-rank AR(1), rank `8` | `5 / 11` | fail |
| Frozen JEPA decoder | `0 / 11` | fail |
| Frozen random-provider decoder | `2 / 11` | fail |

The no-subtraction identity reference failed only the two required
`median centered RMS ratio <= 0.90` checks. This is expected: it preserves
scale exactly and serves as the reference, but it does not suppress it.

Low-rank AR(1) was the least adverse learned method. It kept median centered
RMS near one (`0.99225` held and `0.98239` registered source-off), but did not
reach the required `0.90` suppression, slightly amplified median dynamic MAD
(`1.00238` and `1.00570`), and failed the every-recording RMS check. Its
`5 / 11` score is therefore a useful design lead, not a gate pass.

## Aggregate diagnostics

### All 96 recording-held clips

| Method | Centered RMS | Dynamic MAD | Residual seam | Spectral flatness | Absolute lag-1 ACF |
| --- | ---: | ---: | ---: | ---: | ---: |
| No subtraction | `1.0000` | `1.0000` | `0.9994` | `0.9987` | `0.0376` |
| Last frame / ZOH | `1.4247` | `1.7273` | `0.9996` | `0.7042` | `0.4969` |
| Causal EMA | `1.0839` | `1.1323` | `0.9999` | `0.9862` | `0.1187` |
| Causal median | `1.1138` | `1.1030` | `0.9993` | `0.9884` | `0.0153` |
| Per-pixel AR(1) | `1.4210` | `1.7224` | `0.9996` | `0.7042` | `0.4969` |
| Low-rank AR(1) | `0.9922` | `1.0024` | `0.9994` | `0.9998` | `0.0425` |
| Frozen JEPA decoder | `1.6541` | `1.1169` | `1.2551` | `0.7141` | `0.5007` |
| Frozen random decoder | `1.0854` | `1.0401` | `1.0894` | `0.9788` | `0.0738` |

### Exact 12 registered source-off windows

| Method | Centered RMS | Dynamic MAD | Residual seam | Spectral flatness | Absolute lag-1 ACF |
| --- | ---: | ---: | ---: | ---: | ---: |
| No subtraction | `1.0000` | `1.0000` | `0.9995` | `0.9958` | `0.0426` |
| Last frame / ZOH | `1.4071` | `1.7262` | `1.0015` | `0.7107` | `0.4945` |
| Causal EMA | `1.0807` | `1.1339` | `1.0016` | `0.9865` | `0.1135` |
| Causal median | `1.1169` | `1.1102` | `1.0020` | `0.9838` | `0.0410` |
| Per-pixel AR(1) | `1.4035` | `1.7230` | `1.0016` | `0.7108` | `0.4944` |
| Low-rank AR(1) | `0.9824` | `1.0057` | `0.9998` | `0.9996` | `0.0406` |
| Frozen JEPA decoder | `1.5142` | `1.1039` | `1.2718` | `0.7542` | `0.4657` |
| Frozen random decoder | `1.0653` | `1.0309` | `1.0959` | `0.9583` | `0.1218` |

The learned JEPA residual is adverse on this source-off contract: it has the
largest centered RMS ratio, exceeds the seam limit on both panels, reduces
spectral flatness, and greatly increases absolute lag-1 autocorrelation. The
random-provider decoder is less adverse than JEPA on these diagnostics but
still fails nine of 11 checks. Neither result identifies the represented
content or explains the mechanism biologically.

## Integrity and selection boundaries

- EXP-0028 and EXP-0029 Run-B upstream artifact snapshots were exact and
  unchanged before and after the benchmark.
- The simple learned predictors fit only the 512 training clips.
- Validation clips and source-off windows did not tune or pre-gate-select the
  candidate panel, hyperparameters, checkpoints, thresholds, or a favorable
  reported subset. They were used only by the predeclared eligibility gate,
  which selected zero subtractors.
- Exact evaluation samples are disjoint, but the panels are not independent
  by recording: nine of 12 registered windows share the three `060126_10`,
  `060126_12`, or `060126_15` recording identities with held validation.
- No ROI labels, injected sources, detector outcomes, or recovery results were
  accessed.
- Native activity inside unlabeled validation and quiet/source-off clips is
  unknown, not assumed absent.
- All applicable model-surrogate audit media were produced and decoded, but
  this media audit does not supply the missing biological-signal endpoint or
  exact renderer replay because Matplotlib, Pillow, and ffmpeg/ffprobe build
  versions were not persisted.
- SciPy was not persisted despite use of randomized SVD in the low-rank lane;
  frozen low-rank outputs and conclusions are integrity-checked, but exact
  numeric environment replay for that lane is incomplete.

Run A is not registered because its persisted command provenance contained a
workstation path. Run B used a portable module command with repository root
`.` and a redacted runtime data root, and its run-start dirty state is bound by
a content-aware digest rather than a path-only status hash.

## Decision boundary

Do not use any of the seven tested subtractors to launch a new residual
detector screen. The current automated evidence supports two major directions
instead:

1. replace the completed but unreliable motion/registration sentinel with a
   new estimator or acquisition contract and disjoint reliability validation
   before attributing waveform variation to motion; and
2. redesign the predictor objective or representation before testing source
   preservation, using the low-rank result only as a bounded engineering clue.

Any future predictor must pass a newly frozen source-off gate on disjoint
development/evaluation data before it sees injected-source outcomes. A later
signal-preservation experiment would still require exact source recovery,
multi-seed stability, independent recordings, and the complete scientific
audit.

## Portable provenance

Small, exact, path-sanitized audit artifacts are retained under:

```text
research/run-provenance/NREV-RUN-EXP-0030-SCREEN-20260830-B/
```

The copied subset excludes videos, model state, per-clip tables, raw data, and
the local input manifest. The complete 65 MB package remains under the local
`Outputs/` run root. Frozen bytes and conclusions are integrity-checked, but
the exact low-rank numerical environment and media renderer environment remain
incompletely specified because SciPy, Matplotlib, Pillow, and ffmpeg/ffprobe
build versions were not serialized. No claim or evidence capsule is created.
