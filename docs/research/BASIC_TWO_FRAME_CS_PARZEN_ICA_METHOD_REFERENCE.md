# Basic two-frame CS-Parzen ICA: method and interpretation reference

## Technical summary

This package studies a two-observation ICA model applied to adjacent, causally preprocessed fluorescence frames. The fit is numerically resolved and converged, but the recovered activity direction is almost exactly the temporal derivative. Its absolute cosine similarity to `[-1, 1] / sqrt(2)` is `0.999999917`. The method is therefore best interpreted as a nonparametrically fitted temporal-change diagnostic, not as validated neural/background source separation.

## Data and analysis interval

- Source movie: Spon Ca Burst `3 hindbrain to tail 488 20ms.tif` and its memory-mapped uint16 cache.
- Full source shape: `2359 × 340 × 573` in `T,Y,X` order.
- Review interval: UI frames `1800–2359`, inclusive (`560` frames).
- Quiet calibration: UI frames `1800–1899`, inclusive (`100` frames).
- Source cadence: `20 ms/frame` (`50 Hz`). The video is encoded at `10 fps`, so playback is five times slower than acquisition.
- Labels: `79` burst-specific sparse-positive annotations. Unlabeled pixels are unknown, not negative.
- Coordinates: `x=column`, `y=row`; UI indices are one-based and inclusive.

## Step 1 — causal preprocessing

Let `R_t(x,y)` be the original uint16 fluorescence frame. The fitted method operated on

```text
P_t = EMA_a(G_sigma * R_t),    sigma = 1 pixel,    a = 2/(4+1) = 0.4.
```

`G_sigma` is a spatial Gaussian filter with reflect padding and truncation at `4 sigma`. The exponential moving average is causal:

```text
P_0 = G_sigma * R_0
P_t = 0.4 (G_sigma * R_t) + 0.6 P_(t-1).
```

No motion correction was applied.

## Step 2 — the two-observation ICA model

At every sampled pixel and time, the observation is

```text
x_t = [P_(t-1), P_t]^T.
```

The samples are centered by `mu` and whitened by `Q`:

```text
z_t = Q (x_t - mu).
```

The fitted rotation `W` produces two components:

```text
y_t = W z_t.
```

This model has only two observation dimensions. It cannot independently identify neural signal, background, motion, and measurement noise as four physical sources.

## Step 3 — the Parzen independence objective

For candidate outputs `(y_1,y_2)`, Gaussian Parzen kernels estimate their joint density and marginal densities. Independence is measured with the Cauchy–Schwarz divergence between the joint density `p(y_1,y_2)` and the product of marginals `p(y_1)p(y_2)`:

```text
D_CS(p,q) = -log( integral(p q) / sqrt(integral(p^2) integral(q^2)) ).
```

The implementation searches a bounded two-dimensional rotation angle and minimizes this divergence. It used bandwidth `0.35`, `1024` screen samples, `4096` confirmation samples, `3 degree` coarse steps, and `0.25 degree` refinement steps. Kernel computations were blocked in `256`-row chunks rather than constructing an all-pairs full-movie kernel matrix.

## Step 4 — selecting and orienting the activity component

The recovered component was selected using correlation with the fixed derivative, positive skewness, upper-tail occupancy, and low correlation with the common direction. The selected component is `k=1` with sign `s=-1`:

```text
Y_t = s e_k^T W Q ([P_(t-1),P_t]^T - mu).
```

The effective observation-space direction is

```text
[-0.707394214, +0.706819231].
```

For comparison:

```text
common direction       = [ 0.707106781,  0.707106781]
temporal derivative    = [-0.707106781,  0.707106781].
```

The learned direction has absolute cosine similarity `0.999999917` to the derivative and `0.000406574` to the common direction.

## Step 5 — quiet-standardized positive activity

For every pixel, the quiet interval supplies a robust center and scale:

```text
c(x,y) = median_Q Y_t(x,y)
s(x,y) = max(1.4826 median_Q |Y_t-c|, 0.454600135)
Z_t+(x,y) = max(0, (Y_t(x,y)-c(x,y))/s(x,y)).
```

The binary decision used `Z_t+ >= 3`. This z-like score is not a probability.

## What the real-data result establishes

- CS-Parzen converged in `58` objective evaluations with covariance condition number `22990.422`.
- Full-stack sampled correlation between CS-Parzen activity and the fixed derivative is `0.997828107`.
- After quiet-fitted scale alignment, the residual `E_t = Y_t - beta D_t` has normalized RMS `0.0689458033` with `beta=0.127960907`.
- CS-Parzen known-label mean recall was `0.1333` (`10/79` matches, `24` event candidates), versus Raw Direct `0.6056` (`49/79`, `232` candidates).

These measurements establish that the fitted output is essentially change evidence. They do not establish that detected changes are neural, that unmatched candidates are false, or that the residual is measurement noise.

## Why the derivative appears

Persistent intensity lies near `[1,1]`: it changes little between adjacent frames. The orthogonal direction is `[-1,1]`, which evaluates to `P_t-P_(t-1)`. With only two strongly correlated adjacent observations, whitening and independence optimization naturally expose a common-level coordinate and a change coordinate. The result is therefore a consequence of the observation geometry as well as the Parzen objective.

## Video panel dictionary

The full diagnostic video contains six formula-defined panels. All display limits are fixed for the full 560-frame interval.

1. `R_t`: original raw observation.
2. `P_t`: Gaussian-smoothed causal EMA input used by ICA.
3. `D_t=P_t-P_(t-1)`: explicit fixed derivative baseline.
4. `Y_t`: selected, oriented CS-Parzen component.
5. `Z_t+`: positive quiet-standardized activity; threshold `3` is shown but the panel remains continuous.
6. `E_t=Y_t-beta D_t`: what remains after fitting the derivative's scale on quiet frames.

Raw and preprocessed panels use black for low and white for high. Signed panels use black for negative, mid-gray for zero, and white for positive. Black-backed white rings appear only during the corresponding labeled burst and indicate sparse-positive known neurons; all other pixels remain unknown.

## Interpretation boundaries

The method is defensible as an interpretable temporal-change estimator and as evidence about identifiability. It is not defensible as a validated physical source decomposition or as a replacement for Raw Direct. Motion, illumination changes, neural onsets, and frame noise can all contribute to `D_t` and therefore to `Y_t`.

## Next questions justified by this package

1. Does the small non-derivative residual contain repeatable biological structure or only numerical/model mismatch?
2. Would longer temporal embeddings provide genuinely new identifiable directions, rather than reparameterizing derivatives?
3. Can measured motion or illumination covariates explain derivative energy before neural interpretation?
4. Can the continuous Parzen score help timing or ranking while Raw Direct retains amplitude and structure?
5. Which conclusions survive bounded, exhaustively annotated spatial review?
