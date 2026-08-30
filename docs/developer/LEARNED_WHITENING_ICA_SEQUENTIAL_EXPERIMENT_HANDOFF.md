# Learned whitening and ICA sequential experiment handoff

The authoritative 31-section program is
`docs/archive/plans/NEUREV_LEARNED_WHITENING_ICA_SEQUENTIAL_CODEX_PLAN.md`. The maintained entry
point is `neurobench.experiments.learned_operator_selection` and the user-facing
workflow is `docs/workflows/spon_ca_burst_learned_operator_selection.md`.

S0 passed and Milestone 3 plus the frozen-design portion of Milestone 4 are now
implemented. `S1_OPERATOR_SCREEN --smoke` runs only S1A synthetic validation;
the runner still refuses a biological S1 response screen. The next integration
checkpoint is a sequential, non-caching 8-point-per-family runner over the
already frozen master designs, with leakage-safe Fixed-Select and compact
metrics. Do not regenerate or reorder those designs.

That checkpoint is now complete. The authoritative calibration-rescue root is
`stages/S1_OPERATOR_SCREEN/prefix_8_calibration_rescue`; the unsuffixed prefix
is preserved but superseded because its base normalization was mismatched.
All three families stopped without a useful Fixed-Select or oracle gap. Do not
implement S2. The next allowed implementation is the minimal matched S5 ICA
confirmation at ranks 8 and 16.

LLM entry order is `llm_context.json`, `summary.json`, `artifact_index.json`,
then `validation.json`. These files identify the valid and superseded roots
without requiring recursive scans or media decoding.

The working tree contained unrelated user changes before this work. Preserve
them. Do not overwrite an existing preflight or program root. Screening outputs
must remain compact; no videos, TIFF stacks, dense score arrays, or full
candidate tables are written by S0.
