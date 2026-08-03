# Spon Ca Burst dependent multiscale real diagnostic

## Decision

**Do not advance the current dependent-group proxy to the W7 scientific
patchwise run and do not replace the accepted scientific carrier.** Generated
gates C2 and C3 failed. The real-data application was explicitly selected for
failure analysis and remains diagnostic only.

## Generated gate

The W5 matrix evaluated 15 exact-truth fixtures, three seeds, and three lanes
(135 evaluations). Numerical closure passed, but the dependent-group lane did
not improve attribution over the orthogonal shared/private reference:

- median signal leakage changed from `0.4014791` to `0.4015899`
  (`-0.0276%` relative improvement);
- median diagonality changed from `0.4721517` to `0.4632796`;
- median neural peak-amplitude ratio was `0.5522204`;
- median temporal-area ratio was `0.6723938`;
- C1 passed, C2 failed, C3 failed, C4 was not qualified, and C5 remained
  diagnostic only.

## Real-data diagnostic

The final bounded verification is
`Outputs/HierarchicalParzenICA/spon_ca_burst_dependent_multiscale_real_v3`.
It used 560 real frames (UI 1800--2359), geometry 340 by 573, CPU only, two
threads, and no dense channel artifacts.

- status: `diagnostic_only_do_not_advance`;
- elapsed time: `19.82 s`;
- maximum normalized closure: `6.37e-08`;
- peak resident memory: `10679.7 MiB` against a `12288 MiB` cap (pass);
- output size: approximately `37 MiB`;
- MP4: H.264, 1430 by 384, 560 frames, 10 fps, 56 seconds;
- MP4 SHA-256:
  `be1f825e0366917682ac518f36d41116fdf864b45225ba26b53cb3a1367e1ea7`.

The fixed-scale diagnostic video contains raw observation, accepted carrier,
background, structured signal, structured artifact, noise candidate, closure,
and 5/7/15-pixel views. Cyan rings show sparse-positive labels; other pixels
remain unknown.

Visual review at UI frame 2005 shows the central broad burst primarily in the
background channel while structured signal emphasizes local/edge contrast.
This agrees with the generated preservation failure: the current proxy is not
safe as an amplitude-preserving neural reconstruction. The residual remains
`noise_candidate`, not qualified measurement noise.

## Earlier engineering diagnostics

The completed v1 and v2 roots are retained unchanged. Both produced valid
videos but exceeded the declared 12 GiB memory cap. The v3 implementation
removed whole-movie float64 promotion, aliases identical observation/carrier
inputs, and uses opt-in in-place conservation-preserving reassignment. v3 is the
first resource-compliant real diagnostic and is the preferred artifact.

## Next justified work

Further work should change the source-attribution model before another full
movie run. A bounded semi-synthetic injection lane (W6) and patchwise local
model are appropriate only after a revised method passes generated attribution
and preservation gates. The accepted carrier remains external authority.
