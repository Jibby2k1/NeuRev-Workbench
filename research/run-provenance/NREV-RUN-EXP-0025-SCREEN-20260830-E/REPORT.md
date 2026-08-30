# NREV-EXP-0025 bounded motion and registration-confound audit

## Outcome

The bounded engineering screen completed on the exact twelve frozen NREV-EXP-0029 background windows. It is label-free and non-claim-bearing. It does not motion-correct the recordings or identify biological motion.

- Windows: 12 across 4 recording groups.
- Adjacent-frame pairs: 372.
- Median global translation across windows: 3.38383 native pixels.
- Maximum window p95 global translation: 9.11106 native pixels.
- Median registered/raw difference-MAD ratio on identical shift-valid pair interiors: 0.95209.
- Median retained matched-support pixel fraction: 0.673096.
- Median matched/full-frame raw difference-MAD ratio: 1.
- Windows with forward/back review triggers: 1 / 12.
- Windows requiring local-translation reliability review: 12 / 12.
- Median local-window/full-field p95 translation ratio on shared 060126 recordings: 19.4188.
- Smallest grouped exact p / BH q: 0.0138889 / 0.194444; BH-passing associations: 0.
- Stored uint16 high-code occupancy: maximum 0; analog rail status remains unresolved.

## Frozen endpoint linkage

Raw-HC and JEPA/random residual values are read from the hash-verified NREV-EXP-0029 Run-B tables without recomputation or tuning. Associations use twelve window-level observations, retain recording groups, enumerate all 6^4 = 1,296 within-recording permutations when complete, and report leave-one-recording-out ranges. They remain descriptive with only four recording groups.

Reciprocal forward/back phase estimates are conjugate by construction and therefore validate implementation symmetry rather than supply independent biological evidence. Independent two-step temporal cycle closure, search-boundary concentration, tile/global disagreement, and registered-difference reduction are used as the local reliability diagnostics. The local 64x64 estimates are contextualized against, but not equated with, the differently sampled prior full-field 11-recording screen.

For every valid pair, the raw and registered difference MADs entering the reduction ratio use the exact same conservative shift-valid interior. The separately reported full-frame raw difference MAD remains a standalone change diagnostic and is never used as that ratio's denominator.

Current validated carrier/coherence/recurrence traces were not joined: their finite Spon support is UI 1800-2359, while every frozen Spon background window ends earlier; 060126 has no matched exported feature traces. This prevents an estimand-changing or NaN-based comparison.

## Sensor and registration boundary

Stored integer dtype rails are exact digital-code diagnostics only. ADC bit depth, detector rail, black level, gain, and upstream clipping metadata are absent, so analog saturation cannot be ruled in or out. Phase correlation measures translation-like change and cannot distinguish specimen motion, deformation, scan effects, neural activity, gain drift, or structured noise. Registered residuals also include interpolation error and non-translational change.

## Scientific audit status

The small numeric package is validated, but the required full-field videos, candidate-surrogate close-ups/traces, matched figures, and media decoding checks were not produced. Therefore scientific audit completion and scientific promotion are false.
