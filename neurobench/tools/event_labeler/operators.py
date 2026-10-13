"""Standalone detector operators for the event labeler (numpy + scipy only, float64).

Ports of the NeuRev three-part prescreener operators, so the labeler runs without the research stack:

  P1Phys       arm A5 (P1-phys base chain N0/B0/K0): S = Gaussian blur (sigma 3.24, reflect) of the frame;
               P = EMA(S, 0.4); B = EMA of P's strict past (0.141); D = P - B; z = D / sqrt(c_t * g^2 (*) var_pix),
               var_pix = max(a * mu + b, 1), mu = raw EMA (rate 0.02, updated with the current frame before use,
               mu_0 = first frame), c_t = sum of squared impulse-response taps of D (800 taps).
  GammaLS      arm A1 (deployed Gamma-LS): D = one-frame change of EMA(0.4) levels of a sigma-1 Gaussian blur;
               M, V = mean and variance of D over a Gamma ring (guard 7, half-width 15, shape 9, mode 7.5; 37 px
               box), out-of-image pixels omitted and the ring mass renormalized; z = (D - M) / (max(sqrt V, floor)
               + 1e-6).
  frame_candidates  the sealed per-frame extractor: 13 x 13 local maxima (mode 'nearest'), 6 px border, greedy
               > 6 px separation in (-score, y, x) order, strict > 0.
  photon_line  'Spon method' photon-transfer fit: per-pixel var(Y_t - Y_{t-1}) / 2 against the per-pixel mean over
               the setup frames, 40 equal-count bins, OLS on the bin means.

Parity with the sealed implementation is checked by the analysis machine (see README, "Detector parity").
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import correlate1d, gaussian_filter, maximum_filter
from scipy.signal import fftconvolve

SIGMA_SOMA_PX = 3.24
A_F, A_S = 0.4, 0.141
MU_RATE = 0.02
CT_TAPS = 800
NMS_DISTANCE_PX = 6
GAMMA = dict(sigma=1.0, ema=0.4, guard=7, half_width=15, shape=9, mode=7.5, box=37)
GAMMA_EPSILON = 1e-6
SAT_DN, SAT_WINDOW = 4095, 37


def gauss1d(sigma):
    r = int(np.ceil(4 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-x * x / (2 * sigma * sigma))
    return g / g.sum()


def blur(frame, g):
    return correlate1d(correlate1d(frame, g, axis=0, mode="reflect"), g, axis=1, mode="reflect")


def temporal_factor(a_f, a_s, n=CT_TAPS):
    P = B = 0.0
    taps = []
    for t in range(n):
        s = 1.0 if t == 0 else 0.0
        Pprev = P
        P = P + a_f * (s - P)
        B = B + a_s * (Pprev - B) if t > 0 else 0.0
        taps.append(P - B)
    return float(np.sum(np.square(taps)))


class P1Phys:
    """Streaming P1-phys score (one frame in, one z map out)."""

    def __init__(self, photon_a, photon_b):
        self.pa, self.pb = float(photon_a), float(photon_b)
        self.g = gauss1d(SIGMA_SOMA_PX)
        self.g2 = self.g * self.g
        self.ct = temporal_factor(A_F, A_S)
        self.P = self.B = self.mu = None

    def step(self, frame):
        Y = np.asarray(frame, dtype=np.float64)
        self.mu = Y.copy() if self.mu is None else self.mu + MU_RATE * (Y - self.mu)
        S = blur(Y, self.g)
        sd = np.sqrt(self.ct * blur(np.maximum(self.pa * self.mu + self.pb, 1.0), self.g2))
        if self.P is None:
            self.P = S.copy()
            self.B = self.P.copy()
            return np.zeros_like(S) / sd
        Pprev = self.P
        self.P = Pprev + A_F * (S - Pprev)
        self.B = self.B + A_S * (Pprev - self.B)
        return (self.P - self.B) / sd


def ring_kernel(guard=GAMMA["guard"], half_width=GAMMA["half_width"], shape=GAMMA["shape"], mode=GAMMA["mode"],
                box=GAMMA["box"]):
    h = int(half_width)
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    r = np.hypot(yy, xx)
    k = shape - 1
    w = r ** k * np.exp(-(k / mode) * r)
    w[(r <= guard) | (r > h)] = 0
    w /= w.sum()
    p = (box - w.shape[0]) // 2
    return np.pad(w, p)


class GammaLS:
    """Streaming Gamma-LS: call step(frame) -> (D, M, V); z(D, M, V) once the floor is known."""

    def __init__(self, floor=None):
        self.ring = ring_kernel()
        self.floor = floor
        self.L = None
        self._mass = None

    def _conv(self, x):
        return fftconvolve(x, self.ring, mode="same")

    def step(self, frame):
        S = gaussian_filter(np.asarray(frame, dtype=np.float64), sigma=GAMMA["sigma"], truncate=4, mode="reflect")
        if self.L is None:
            self.L = S
            D = np.zeros_like(S)
        else:
            L = GAMMA["ema"] * S + (1 - GAMMA["ema"]) * self.L
            D = L - self.L
            self.L = L
        if self._mass is None:
            self._mass = self._conv(np.ones_like(D))
        M = self._conv(D) / self._mass
        V = np.maximum(0.0, self._conv(D * D) / self._mass - M * M)
        return D, M, V

    def z(self, D, M, V):
        if self.floor is None:
            raise ValueError("Gamma-LS floor not set")
        return (D - M) / (np.maximum(np.sqrt(V), self.floor) + GAMMA_EPSILON)


def frame_candidates(score, distance=NMS_DISTANCE_PX):
    """[(score, y, x), ...] for one frame's score map (region already cropped)."""
    v = np.asarray(score, dtype=np.float64)
    cutoff = float(np.nextafter(0.0, 1.0))
    keep = (v == maximum_filter(v, size=2 * distance + 1, mode="nearest")) & (v >= cutoff)
    keep[:distance] = False
    keep[-distance:] = False
    keep[:, :distance] = False
    keep[:, -distance:] = False
    y, x = np.nonzero(keep)
    s = v[y, x]
    order = np.lexsort((x, y, -s))
    sel = []
    d2 = float(distance * distance)
    for i in order:
        px, py = int(x[i]), int(y[i])
        if all((px - ox) ** 2 + (py - oy) ** 2 > d2 for _, oy, ox in sel):
            sel.append((float(s[i]), py, px))
    return sel


def sat_mask(frame):
    """True where the 37 x 37 window around a pixel holds a saturated (4095) pixel; None if none saturated."""
    sat = np.asarray(frame) >= SAT_DN
    if not sat.any():
        return None
    return maximum_filter(sat.astype(np.uint8), size=SAT_WINDOW, mode="constant", cval=0) > 0


def photon_line(frames, region_mask=None, nbins=40):
    """(a, b, record) from an iterable of setup frames ('Spon method')."""
    n = 0
    s1 = s2 = d1 = d2 = None
    prev = None
    sat_any = None
    for fr in frames:
        Y = np.asarray(fr, dtype=np.float64)
        sat_any = (Y >= SAT_DN) if sat_any is None else (sat_any | (Y >= SAT_DN))
        if s1 is None:
            s1, s2 = np.zeros_like(Y), np.zeros_like(Y)
            d1, d2 = np.zeros_like(Y), np.zeros_like(Y)
        s1 += Y
        if prev is not None:
            d = Y - prev
            d1 += d
            d2 += d * d
            n += 1
        prev = Y
    if n < 10:
        raise ValueError("photon-line fit needs at least 11 setup frames")
    mean = s1 / (n + 1)
    dvar = (d2 / n - (d1 / n) ** 2) * n / (n - 1) / 2.0
    ok = ~sat_any if region_mask is None else (region_mask & ~sat_any)
    m, v = mean[ok], dvar[ok]
    order = np.argsort(m, kind="stable")
    bins = np.array_split(order, nbins)
    bm = np.array([m[b].mean() for b in bins])
    bv = np.array([v[b].mean() for b in bins])
    a, b = np.polyfit(bm, bv, 1)
    r2 = 1 - np.sum((bv - (a * bm + b)) ** 2) / np.sum((bv - bv.mean()) ** 2)
    return float(a), float(b), dict(method="var(Y_t - Y_{t-1})/2 vs mean; 40 equal-count bins; OLS on bin means",
                                    a=float(a), b=float(b), r2_binned=float(r2), frames=n + 1,
                                    pixels=int(ok.sum()))
