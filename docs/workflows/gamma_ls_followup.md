# Gamma-LS regional calibration and crowding follow-up

This is the first two-part investigation from the paper incorporation plan.
It tests two distinct possible failures: a bright part of the field setting an
unhelpful global cutoff, and a spatial peak selector suppressing a weaker
nearby source. Configured-source nuisance robustness remains a subsequent
experiment. These development tests do not establish biological identity,
independent-recording performance, or feedback control.

The frozen protocol is
`Outputs/GammaLSFollowup/followup_20260914_r1/protocol.json`. It binds the
completed necessity study and the earlier source/annotation completion record.
Existing completed roots and their code are preserved.

## Regional calibration

Seventeen existing datasets receive five score readouts (level input, target,
contrast and Gamma-LS, plus signed-change Gamma-LS) and two calibration schemes:
one global cutoff or a fixed 2-by-3 grid over the eligible field. This gives
170 cells. Kernels, conditioning, full positive NMS lists, original score units,
ranking, image borders, and setup/application intervals remain unchanged.
There are no new masks in the target or reference kernel.

Fit thresholds from earlier setup peaks only. First compute the same total
integer setup budget as the baseline, then distribute it by eligible area using
largest remainders with row-major tie breaking. Every proposal is tested against
its region's cutoff after NMS. A rejected peak does not restore its previously
suppressed neighbors. Strict score exceedance can underfill a budget when scores
tie. It cannot exceed the setup budget. Application burden is measured separately.

Retain all eight q targets and both all-positive/no-output endpoints. The media
operating point is q=1. Regional filtering does not preserve a global score
prefix, so matching is recomputed after each filtered subset is constructed.
All thresholds and proposal lists are sealed before joining activity truth.

## Crowded sources

Three new paired seeds (20260916, 20260917, 20260918) each generate the weaker
source alone and with a stronger neighbor 8, 12, or 16 pixels to its left.
The weaker center stays at evaluation coordinates (68,64). Its Gaussian sigma
is 1 pixel and peak amplitude is 18; the neighbor has sigma 2 and amplitude 24.
Both use the existing provisional 100ms rise and 1000ms decay profile, with
sampled onset at source frame 185. Noise is identical within seed; setup frames
are byte-identical. These are technical stress parameters, not measured kinetics.

Twelve clips cross level target, contrast and Gamma-LS with 13-by-13 or 3-by-3
local-maximum windows. Both selectors keep the 6px border, score/y/x tie order,
and greedy Euclidean separation strictly greater than 6px. Each has its own
setup calibration. This gives 72 cells. The primary separation diagnostic uses
inclusive 2px matching, whose disks do not overlap; the legacy 6px matching
remains a separate comparison. Report weak-source recovery by deadline, matched
active frames, localization error, false proposals and duplicate burden.

The total is 242 cells and 3,140 numerical curve rows, since crowding retains
two matching radii. Existing global controls have 85 equivalent audit states;
the other 157 states require new media.

## Execution and audit

Use `.venv-neurobench/bin/python`, one numerical thread, low priority and a
CPU affinity excluding CPUs 6/7. Check resources first; new scoring is CPU only.
The study module `neurobench.experiments.gamma_ls_difference.followup_study`
provides `preflight`, `prepare`, `run`, and `evaluate`. Preparation saves a
geometry-only projection preflight. `followup_media` provides `forecast`,
`media --worker 0 --workers 3` (also workers 1/2). Once all three workers have
written their PASS receipts, run `followup_validate --root <study-root>` for
aggregate validation. This checks every indexed artifact and source while
hashing shared resolved files once per pass, with file-change guards. It retains
the per-video full decode evidence. The original `followup_media validate`
remains available, but repeatedly hashes the same shared score arrays.

After aggregate PASS, use `followup_report update --root <study-root>`, refresh
the paper companion's completion statement and links, and run
`followup_finalize --root <study-root> --write`. Closure requires the numerical,
media, test and visual-review receipts and saves a source capsule. It never
reopens an already completed study.

The complete Scientific Audit Output Standard is enabled for every q=1 cell:
separate expert/model full fields, every applicable ROI close-up and exact-pixel
trace, every occurrence comparison, tables, context metadata, and full lossless
decode/marker validation. The versioned regional renderer keeps the original
Score and shows the frozen cutoff at each trace pixel. Comparison figures label
the cutoff as belonging to the primary expert trace; other local cutoffs are
recorded separately. No normalized margin is substituted for the score.

Sparse real annotations still support only known-window coverage and unknown
proposal burden. Their acceptance is false; raw-panel rendering does not imply
annotation completion. Numerical completion is separate from media completion.
The report and completion record must state both explicitly.
