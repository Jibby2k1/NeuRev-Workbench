# Why Pairwise ICA Became a Temporal Derivative

For one pixel at adjacent times, the experiment supplies only two observations:

```text
x = [I(t-1), I(t)]^T.
```

If persistent anatomy/background contributes almost equally to both frames, its
direction is approximately `[1,1]`. The orthogonal background-null direction is
therefore `[-1,1]`, giving:

```text
[-1,1] x = I(t) - I(t-1).
```

Two-observation ICA can rotate/scale these aggregate directions, but it cannot
identify individual neurons or create additional independent information. In
the first Spon run, the selected effective directions after centering,
whitening, demixing, component selection, and sign orientation were:

```text
InfoMax:   [-0.698722,  0.715393]
CS-Parzen: [-0.707394,  0.706819]
```

Their absolute cosine similarities to normalized subtraction were `0.9999305`
and `0.9999999`; similarities to the common/background direction were only
`0.01179` and `0.00041`. Thus both fits empirically rediscovered the temporal
derivative. InfoMax's covariance condition number was about `22,990` and it hit
the 500-iteration cap, while CS-Parzen converged.

## Consequence

The useful information is onset/change evidence: it attenuates temporally
persistent brightness and can support propagation timing. It also removes the
absolute fluorescence and structural context that made Raw Direct stronger.
Therefore the continuous positive derivative/ICA value should be used as:

- an auxiliary channel;
- a bounded additive candidate score;
- or a soft gate with a nonzero structural floor.

The binary mask is a visualization/decision artifact, not the preferred fusion
feature. A learned fusion should initialize the Raw weight at one and the
derivative weight at zero, tune the latter narrowly, and retain Raw Direct when
held-out evidence does not improve.
