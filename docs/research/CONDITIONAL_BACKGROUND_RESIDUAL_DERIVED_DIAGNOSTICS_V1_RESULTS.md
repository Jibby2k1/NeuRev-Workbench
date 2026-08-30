# Conditional-background residual derived diagnostics v1

## Outcome first

Two frozen, integrity-checked post-screen diagnostic artifacts sharpen the
EXP-0029 failure story without changing its scientific status.

First, the exact source-on/intervention reconstruction shows that the JEPA
residual's misses are not confined to the competition-consistent
intervention-recovered/source-on-missed pattern. Among source-on misses,
the grouped point fractions were `0.51058` for that pattern and `0.48942` for
the both-maps-missed pattern. Raw and the random-provider residual were much
more dominated by the first operational category at `0.82183` and `0.83770`
respectively.

Second, a retrospective source-off-only guardrail audit admitted none of the
12 JEPA and none of the 12 random-residual background windows. Its fixed unsafe
action was to use the frozen raw endpoint, so both policies selected raw for
all 108 fixtures and reproduced raw macro recall of `0.1875`.

These are derived engineering diagnostics, not preregistered EXP-0029 gates.
They create no claim or evidence capsule, do not complete the scientific audit,
and do not establish denoising, precision, biological identity, nuisance
robustness, or generalization.

## Parent and diagnostic identity

- Parent experiment: `NREV-EXP-0029`
- Parent run: `NREV-RUN-EXP-0029-SCREEN-20260830-B`
- Parent artifact-index SHA-256:
  `34bd983ffdf55cb0941bbd1443adf939f4cb86ac4ffe51a736b2d6d9383cfc1c`
- Rank diagnostic:
  `NREV-DIAG-EXP-0029-RANK-DISPLACEMENT-20260830-F`
- Rank implementation SHA-256:
  `8407eb2926b55304ec4d1e409e9b5049e3db08b720ba3dc046c9461ece62ddac`
- Rank focused-test SHA-256:
  `bd065f4f546d3906ab14c64083bd50837990bc9703875e5b9d8c16ec0c9ec725`
- Rank artifact-index SHA-256:
  `ca1933513b224b735882de1196a234f2d1246649211855ac81c6d4cb3c1fdaf5`
- Rank result SHA-256:
  `4cc984641e17ab9f3eb5fd0a653b0051d16fe3f449ba45666366f5372444db2d`
- Rank summary-file SHA-256:
  `d8e1201a5c2657060ec93ee998b09dc619b68c80d8b8985779c9308915e6402b`
- Rank execution-provenance file SHA-256:
  `af42a4ff6314d100908fda54c22f7e1ad9549888af584363b43ef1d24b0400b9`
- Rank input-provenance SHA-256:
  `d8bc6e956456b6223fb7c28304e4b70c3379b217d04bbe3b362bdf6fc16f4445`
- Rank run-start dirty-content digest:
  `c70a694ec1ba069db42833cae730c9ed7db52cbb20ecd59cf7bbde1bc1fce4a4`
- Stable safety package: `source_off_residual_safety_v1_1`
- Safety implementation SHAs:
  `047d988b6158d67f89ac195a5fd2f7433a44ca17d43b13862fa870464c0559d0`
  for the policy core and
  `135144627dccb567e654211699105c23691a988dbef7df9ea784170306906274`
  for the audit runner
- Safety artifact-index SHA-256:
  `a7d3045c34e53d571c0cb277a5f8edd43b75345c94393b7efe675538fec4fe74`
- Safety result SHA-256:
  `fcc739667ba6e80056402ae88e31e4a7c766dc8a94db8a226c8d053a8c1a9a80`

The rank package contains 11 indexed artifacts totaling 491,492 bytes. The
safety package contains 11 indexed artifacts totaling 174,785 bytes. Both
indexes independently re-hashed without discrepancy. The complete small,
path-sanitized Rank-F artifact set and compact exact safety summaries,
validation records, and indexes are retained in the portable Run-B provenance
tree.

The earlier `source_off_residual_safety_v1` directory is a preliminary local
output from before implementation provenance was serialized. It is not used,
copied, or registered here.

These packages are not registered run executions. Rank F records a portable
command, run-start Git/content digest, runtime, timestamps, frozen inputs, and
all seven live numerical dependencies, including SciPy 1.17.1; its inputs and
dependencies remained unchanged through execution, and its primary duration is
exactly start-to-end with separate preflight and total phase durations. Safety
v1.1 is a fixed derived policy artifact with non-run status. Rank F is therefore
provenance-complete as a derived
diagnostic, while neither artifact is a scientific run or evidence capsule.
The reported counts remain descriptive derived results.

## Rank-displacement diagnostic

The diagnostic rehydrated all three Run-B score lanes over 108 fixtures and
252 exact injected sources. It reproduced all 324 method-by-fixture source-on
and intervention recovery objects exactly and emitted 756 finite
source-by-method rows.

Each source belongs to one of four exhaustive score-map categories:

| Method | Recovered on both maps | Source-on-only recovery | Intervention recovery but source-on miss | Miss on both maps | Total source-on recovered |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native raw HC | `43` | `2` | `167` | `40` | `45` |
| JEPA-residual HC | `18` | `1` | `114` | `119` | `19` |
| Random-residual HC | `16` | `2` | `196` | `38` | `18` |

The third category is named `native_background_competition` in the diagnostic,
but its evidence supports only a competition-consistent recovery pattern: the
exact source was recovered on its paired intervention map but
was not one-to-one recovered by the source-on top four. This may reflect
another injected source, assignment effects, or native structure; competitor
identity is unknown. The fourth is named
`attenuation_or_response_failure`: neither map recovered the source. The
source-on-only category is retained separately rather than silently folded
into either miss category. These are operational score-map categories, not
biological labels.

The final F package explicitly defines `43 / 18 / 16` as recovered on both
maps and reports complete source-on recovery totals of `45 / 19 / 18` after
retaining the `2 / 1 / 2` source-on-only cases.

Grouped uncertainty preserves recording, background-window, and injection-seed
clusters and keeps source count nested:

| Method | Competition-consistent fraction among source-on misses | Grouped 95% interval | Both-maps-missed fraction | Grouped 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Native raw HC | `0.82183` | `[0.67602, 0.94057]` | `0.17817` | `[0.06433, 0.35769]` |
| JEPA-residual HC | `0.51058` | `[0.31349, 0.67077]` | `0.48942` | `[0.30218, 0.68393]` |
| Random-residual HC | `0.83770` | `[0.68931, 0.94261]` | `0.16230` | `[0.05684, 0.32863]` |

The point estimates therefore show that the competition-consistent category
does not exhaust JEPA's source-on misses. That recovery pattern remains common,
but the JEPA residual also has a much larger both-maps-missed component than raw
or the random-provider residual. The latter means only that the source was not
recovered in either top-four map; it does not prove attenuation or response
failure. The grouped intervals are descriptive, not a formal gate.

### Median-MAD failure mode

The median-MAD stratum is the clearest local warning. JEPA recovered `0 / 84`
sources on source-on maps and `22 / 84` on intervention maps. Raw recovered
`13 / 84` and `70 / 84` respectively. JEPA's median local source rank was
`30.5`, compared with `9.5` for raw. This co-occurs with the Run-B seam and
background-amplification diagnostics and motivates separate tests of predictor
quality, assembly seams, and acquisition/registration reliability before any
larger residual run; it does not identify their causal contributions.

Earlier rank-diagnostic roots remain provisional and unregistered. C is
preliminary because its report label, standalone-status inventory, and
source-count-stratum serialization were incomplete. D corrected those fields
but did not pin every imported numerical module or carry complete execution
provenance. E added those dependencies but its primary duration semantics were
not explicit and the SciPy runtime version was absent despite SciPy numerical
calls. Only F is used here. F retains the corrected category definitions and
source-count levels `1`, `2`, and `4`, pins all numerical dependencies, and
records explicit phase durations and SciPy 1.17.1; none of those engineering
corrections changes the four-category counts or grouped results.
Scientific-audit media remain absent, so the diagnostic is still
non-claim-bearing and scientifically incomplete.

## Source-off residual guardrail audit

The safety audit used only source-off residual diagnostics to make one frozen
decision per method and background window. Injected truth and biological labels
did not enter selection. The post-screen thresholds were:

- background RMS ratio at most `1.0`;
- dynamic-MAD ratio at most `1.0`; and
- source-off seam-to-interior jump ratio at most `1.25`.

All three conditions were required. The thresholds were defined after Run B
for future guardrail development, so their application to Run B is an audit,
not a prospective gate.

| Residual arm | Safe windows | RMS-safe windows | Dynamic-MAD-safe windows | Seam-safe windows |
| --- | ---: | ---: | ---: | ---: |
| JEPA residual | `0 / 12` | `1 / 12` | `0 / 12` | `6 / 12` |
| Random residual | `0 / 12` | `1 / 12` | `0 / 12` | `12 / 12` |

Because no residual window was admitted, both policies used raw for every one
of the 108 fixtures. Each reproduced raw macro recall `0.1875` and raw micro
recovery `45 / 252`. This validates deterministic abstention and fallback
identity; it does not show that the residual improved detection.

A future prospective safety gate would need thresholds frozen on disjoint
source-off development data before any source-on evaluation. It should be
treated as a fail-closed deployment guardrail, not as a way to select a
favorable subgroup after observing injected-source outcomes.

## Updated engineering decision

Keep the current nonoverlapping tiled residual design on hold. The combined
evidence now distinguishes three issues:

1. Run B's near-unity aligned injected amplitude co-occurred with background
   amplification, orthogonal error, seams, and lower recovery than raw.
2. The JEPA-specific miss pattern includes a substantial both-maps-missed
   category in addition to the competition-consistent
   intervention-recovered/source-on-missed pattern.
3. A retrospective source-off-only guardrail rejects every residual window and
   abstains to raw.

The two upstream sentinels are now resolved negatively for their current
implementations. Motion Run E found all 12 local translation-like fields
unreliable and zero of 14 associations multiplicity-supported. EXP-0030 Run B
admitted zero of seven subtractors through its source-off gate. The next work
therefore requires a new motion estimator or acquisition contract and a new
predictor design on disjoint development data; seam-safe residual inference is
not justified until source-off feasibility is first established. Repeating the
same residual screen with more steps or seeds is not the next informative test.

## Scientific and publication boundary

- `NREV-EXP-0029` remains `draft`, `not_evaluated`, and evidence tier `none`.
- Native candidates remain `unknown_not_negative`; neither diagnostic estimates
  specificity or false-positive rate.
- Full-field videos, close-ups, full-duration traces, matched figures, and
  decode-validation media remain absent.
- The motion dependency remains unsatisfied because Run E produced no usable
  local field; multi-seed stability and independent-recording generalization
  also remain unresolved.
- `scientific_completion=false` and
  `scientific_promotion_allowed=false` remain in force.
- No claim or evidence capsule is created.

## Portable provenance

Exact sanitized diagnostic artifacts are retained beneath:

```text
research/run-provenance/
  NREV-RUN-EXP-0029-SCREEN-20260830-B/derived/
```

The rank summary and safety summary file SHA-256 values are respectively
`d8e1201a5c2657060ec93ee998b09dc619b68c80d8b8985779c9308915e6402b`
and
`1d5132a99851d59606488c48c9d15d0e40a12fe08c5df8113fb74f9841520134`.
Rank F's full 11-artifact portable set, including its 756-row source table, is
copied into the public research tree. Any future large/local diagnostic media
remain under `Outputs/` unless separately sanitized and registered.
