# PC-MITL-ICA phase-1 evaluation

## Decision and scope

The supplied PC-MITL-ICA proposal is technically plausible as a staged research
program, but it is not evidence that the combined method improves neuronal
separation. The first justified slice is the proposal's minimal spectral
continuation: matrix-based Rényi total correlation at `alpha = 2`, compared
with the maintained CS-Parzen implementation on controlled synthetic mixtures.

This phase deliberately excludes trajectory/history objectives, conditional
Cauchy-Schwarz penalties, multiscale aggregation, robust whitening, nonlinear
encoders, real-video tuning, and publication claims. Those additions remain
blocked until the reference mathematics, gradients, orthogonal optimizer, and
synthetic recovery are reliable.

## Implementation

- Reference code: `neurobench/experiments/unsupervised_ica_eval/matrix_itl.py`
- Frozen baseline: `experiments/unsupervised_ica_eval/baseline_cs_parzen_canonical.yaml`
- Focused tests: `tests/test_pc_mitl_ica_phase1.py`
- Bounded runner: `scripts/run_pc_mitl_ica_phase1.py`
- Repository audit: `docs/research/reports/ica_repo_audit.md`
- Findings: `docs/research/reports/phase1_findings.md`

The baseline fits centering, covariance, and whitening on the contiguous
training block only. Validation and test frames are not used for fitting. Every
run binds the resolved configuration, synthetic data identifier, and Git commit
into a SHA-256 baseline fingerprint and refuses output-directory collisions.

## Scientific-audit status

This is an unlabeled, two-source synthetic numerical check. Its frozen sources
serve as the candidate-surrogate panel; `summary.json`, resolved configuration,
and a comparison figure provide the phase-1 evidence. It does not satisfy the
real-video expert/model/comparison media contract and cannot be promoted as a
completed neuronal experiment. Any later real-video study must implement the
full `SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md` artifact set.

The next E01/E02 gate has now been executed and is reported in
`docs/research/PC_MITL_ICA_E01_E02_RESULTS.md`.

## Run

```bash
env PYTHONPATH=. MPLCONFIGDIR=/tmp/pc_mitl_matplotlib \
  .venv-neurobench/bin/python scripts/run_pc_mitl_ica_phase1.py \
  --config experiments/unsupervised_ica_eval/baseline_cs_parzen_canonical.yaml \
  --output-dir Outputs/UnsupervisedICAEval/pc_mitl_ica_phase1_v2
```
