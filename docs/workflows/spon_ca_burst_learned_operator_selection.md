# Spon Ca Burst learned operator selection

This additive, sequential program tests learned whitening/operator selection and
the marginal utility of ICA. The authoritative design is
`docs/archive/plans/NEUREV_LEARNED_WHITENING_ICA_SEQUENTIAL_CODEX_PLAN.md`.

The current implementation contains Milestones 1–3 and the deterministic-design
portion of Milestone 4: strict configuration, state/dependency validation,
guarded preflight, the S0 Raw Direct evaluator freeze, fractional whitening
primitives, S1A synthetic validation, and frozen nested Sobol master designs.
The biological S1 response-screen runner, learned mixtures, adaptive gating,
and ICA ablations are not yet implemented. No run-all command exists.

## Commands

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment learned-operator preflight \
  --config examples/spon_ca_burst_learned_operator_selection.example.json \
  --artifact-dir Outputs/LearnedOperatorSelection/spon_ca_burst_learned_operator_selection_v1_preflight

.venv-neurobench/bin/python -m neurobench.cli.main experiment learned-operator run-stage \
  --config examples/spon_ca_burst_learned_operator_selection.example.json \
  --preflight-dir Outputs/LearnedOperatorSelection/spon_ca_burst_learned_operator_selection_v1_preflight \
  --stage S0_BASELINE

.venv-neurobench/bin/python -m neurobench.cli.main experiment learned-operator status \
  --program-dir Outputs/LearnedOperatorSelection/spon_ca_burst_learned_operator_selection_v1
```

Preflight is collision-safe, fingerprints the source and labels, writes the
required label projection, checks live resource headroom, and records the exact
audit exemption. S0 calls the maintained Raw Direct construction and detection
primitives, but freezes deterministic score/y/x tie ordering. It independently
validates Macro-KPR@58 and the Q1 operating point, writes only compact metrics,
and stops the program if the evaluator cannot be recovered.

Screening uses the user-authorized essential-metrics-only exemption recorded in
the manifest. A promoted finalist must re-enable and pass
`SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md`. Unmatched candidates remain unknown.
All conclusions from this recording are within-video only.

S1A passed its six numerical gates. The three 64-row scrambled Sobol master
designs are frozen under the program output's `designs/` directory before any
biological operator evaluation. This is implementation evidence only and does
not establish whitening utility on Spon Ca Burst.

## S1 eight-point result (2026-08-20)

The first screen was preserved but invalidated before promotion because it used
a per-pixel MAD base rather than the frozen global-scale Raw Direct protocol.
The single permitted calibration rescue reran the identical 24 frozen cells
with the exact maintained signed Raw Direct base. All cells were finite and
well conditioned. The authoritative leakage-safe results were:

| Family | Fixed-Select Macro-KPR@58 | Matches | Delta vs Raw Direct | Outcome |
|---|---:|---:|---:|---|
| Spatial | 0.6400 | 51/79 | -0.0173 | weak/dead |
| Temporal | 0.6416 | 51/79 | -0.0156 | weak/dead |
| Separable spatiotemporal | 0.6535 | 52/79 | -0.0037 | weak/dead |

Raw Direct remained `0.6572` with 52/79 matches. No family had a material
oracle gap, so S1 wrote `stop_branch`; S2 learned mixtures are not justified.
The sequential plan permits only the minimal matched rank-8/16 ICA utility
confirmation next. This negative result is limited to one recording and the
eight-point prefixes. Sparse positives do not identify precision.
