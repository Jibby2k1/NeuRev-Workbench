# Gamma-LS background and conditioned-noise controls

The prospective paper protocol (local workspace: `Neural_Event_Extraction_Gamma_LS_Clarity_Revision_2026-09-12/editorial/BACKGROUND_VARIANCE_PROTOCOL_2026-09-15.md`)
crosses flat/sloped background with raw/conditioned variance matching and the
original V/S/T interventions. There are 288 logical cells: 252 unique scored
states and 36 identity aliases. Seventy-two states reuse the completed noise
study; 180 require new full scientific audits. The old noise root stays frozen.

Run in `Outputs/GammaLSBackground/background_20260915_r1` only after a noncolliding
preflight. Use the `background_study` module commands `preflight`, `prepare`,
`run`, `seal_all`, and `evaluate`. Numerical `run` supports up to three disjoint
case workers with `--worker N --workers 3`. All cell seals and setup calibration
checks precede evaluation. Run `background_integrity --root ROOT` after
`seal_all`; it compares all paired prefixes without reading activity outcomes.

Background is fixed across warmup/setup/application, and only application noise
receives analytic normalization. No observed frame variance or application
score fits that scale. Flat/sloped setups are calibrated separately. S=T=0
normalization aliases raw and is not a replicate. The finite-filter stationary
matching gain is not an exact finite-realization or transition match.

Use `background_media forecast` before three complete `media` worker runs.
Its deterministic balanced assignments, original reuse ownership, display
contracts and per-cell source hashes are frozen before rendering. The
`background_validate` aggregate verifies the entire physical matrix. This
implements the full [Scientific Audit Output Standard](SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md):
all required model videos, close-ups, full traces, stage arrays, detection
metadata, LLM indexes and comparison records, with source-free Expert and
Comparison applicability made explicit. No audit opt-out is active.

Report generation comes from `background_report`. Preserve primary late rates,
early/full rates, predeclared settled windows and fixed transition bins. The
three seeds are the replicate units. Source-free sensitivity is undefined;
matched marginal variance does not imply matched score or proposal distributions.
The control does not establish signal preservation or real-time feedback utility.

Respect at most three single-thread CPU workers, low priority, CPUs6/7 excluded,
and verify host resources before each long phase. Never overwrite completed
roots or earlier scientific source files. Completion requires the numerical,
full media, figure/PDF and representative-trace QA, independent recount and
source-capsule gates, not just process exit.
