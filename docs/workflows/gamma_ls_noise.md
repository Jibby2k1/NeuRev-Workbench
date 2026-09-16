# Gamma-LS noise variance/covariance control

The paired source-free control uses eight V/S/T combinations, three seeds and
three exact references from the completed reference study. V changes raw noise
standard deviation 2→5 at application offset 100; S applies a unit-variance 3×3
spatial filter; T applies stationary causal AR(1) noise with coefficient 0.8.
S/T begin at application offset 0. Independent latent initialization preserves
unit application variance without consuming the shared main innovation stream.

All 64 warmup and 100 setup frames are identical within each seed. All variance
pairs are identical through the first 100 application frames. Apply σ after AR,
with no empirical normalization. Fixed Gaussian σ=1 and EMA α=0.4 conditioning can
amplify the retained variance of correlated noise; raw variance and conditioned
variance are separate diagnostics. The compound nuisance from the preceding
study remains a separate reused comparison, not factorial 111.

The new root is `Outputs/GammaLSNoise/noise_20260915_r1`. There are 27 datasets,
81 states, 1,620 threshold/radius rows, 18 exact audit reuses and 63 new audits.
Use `noise_study {preflight,prepare,run,evaluate}` in
`neurobench.experiments.gamma_ls_difference`; all commands take `--root`.
Review the source-free setup projections before `run`. All candidate streams
must be sealed before empty activity truth is joined. Keep setup-fitted floors,
native cutoff plans, direct reference statistics and target responses fixed.

Use `noise_media forecast`, then its three disjoint `media --worker N --workers 3`
jobs, and `noise_validate` for the complete source/artifact aggregate. Reports
come from `noise_report`; completion requires numerical, full media, focused
test, figure/PDF review and provenance gates. Do not overwrite completed roots
or edit frozen earlier source modules. Respect single-thread CPU limits, low
priority and excluded CPUs 6/7; perform the repository resource preflight.

The post-scoring `noise_integrity --root ROOT` check verifies exact fixed target
responses, paired setup calibration, and every pre-step stage/candidate prefix.
It does not parse truth or evaluation outcomes. Run it after all 81 seals exist.
The final `noise_finalize --root ROOT --write` command closes the experiment
only after numerical, full media, visual, document-link and focused-test gates
pass. Its completion manifest and copied source capsule preserve the evidence.

The full Scientific Audit Output Standard is enabled at q=1, including every
model ROI video and full-duration trace, both full-field section records,
explicitly empty Expert/Comparison applicability, indexes, full decoding and
marker checks. Reused source/media must match exact frozen hashes; declare any
unavoidable display-range exception. Model sites are spatial review locations.

Use early/late false-proposal rates with area-time denominators, per-seed paired
factorial effects/interactions and trajectories. Do not invent sensitivity for
a source-free simulation, count frames as independent replicates, treat the
legacy bridge as a factorial treatment, or infer biological/controller success.

Read the numerical report and figures (`Outputs/GammaLSNoise/noise_20260915_r1/report/REPORT.md`)
and the paper findings companion (local workspace: `Neural_Event_Extraction_Gamma_LS_Clarity_Revision_2026-09-12/editorial/NOISE_FACTORIAL_FINDINGS_2026-09-15.md`).
The actual-weight covariance calculation (`Outputs/GammaLSNoise/noise_20260915_r1/validation/noise_covariance_mechanism/README.md`)
explains why fixed background gradients and noise covariance can affect the
score; its ratio of moments is not a prediction of the score's variance or tail.
The supplemental threshold plot (`Outputs/GammaLSNoise/noise_20260915_r1/validation/threshold_sensitivity/README.md`)
shows all ten existing cutoffs using the full six-second application exposure.
Use the completion manifest, once present, as the authority for full closure.
