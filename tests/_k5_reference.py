"""The K.5 reference estimator: the scratch implementation that ran the K.5(2) study of the
calibration-speed work (2026-10-04; 24 paired calibrations at 8·10⁵ particles against the sorted
path), committed as the reference the library's binned estimator is compared with.

It is "variant C" of that study, written before ``volsto/calibration/binned.py`` and by another
route: the two quantile ranks and the window-edge ranks each come from one ``np.partition`` over
**all** the particles (the library uses an exact histogram selection and a partition of the two
tail subsets), and the tail particles of the floored nodes are selected by a boolean mask.

What was done to the scratch file (``k5_binned.py`` and ``k5_hybrid.py`` of the study), and
nothing else: the statements of variant C are kept as they ran, one per line in the repository's
style; the other window variants of the slice check (K.5(1): symmetric and bin-weighted
windows), the tap counters and the stage timers are left out; the post-processing, which the
scratch file took from the source text of ``conditional_variance_estimate``, is the library
function ``_finish_estimate``, called directly.  ``deflate`` is the K.5(2) deflation arm.

Test-only: nothing in ``volsto`` imports it.  Do not edit.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from volsto._numba import njit
from volsto.calibration import particle
from volsto.config import ParticleConfig

FloatArray = NDArray[np.float64]

BIN_FRACTION = 4


@njit(cache=True)
def minmax(k):  # pragma: no cover - numba
    lo = k[0]
    hi = k[0]
    for i in range(k.shape[0]):
        x = k[i]
        if x < lo:
            lo = x
        elif x > hi:
            hi = x
    return lo, hi


@njit(cache=True)
def bin5(k, v, a, inv_d, w0, w1, w2, y0, y1):  # pragma: no cover - numba
    for i in range(k.shape[0]):
        x = (k[i] - a) * inv_d
        j = int(x)
        f = x - j
        g = 1.0 - f
        e0 = f / inv_d
        e1 = -g / inv_d
        vi = v[i]
        w0[j] += g
        w0[j + 1] += f
        w1[j] += g * e0
        w1[j + 1] += f * e1
        w2[j] += g * e0 * e0
        w2[j + 1] += f * e1 * e1
        y0[j] += g * vi
        y0[j + 1] += f * vi
        y1[j] += g * e0 * vi
        y1[j + 1] += f * e1 * vi


@njit(cache=True)
def count_less(k, T, lo, inv_cell, lut):  # pragma: no cover - numba
    """number of particles strictly below each sorted threshold T[j] (exact).  lut[c] = index of the
    first threshold above the left edge of cell c."""
    nT = T.shape[0]
    hist = np.zeros(nT + 1, dtype=np.int64)
    nc = lut.shape[0]
    for i in range(k.shape[0]):
        x = k[i]
        c = int((x - lo) * inv_cell)
        if c < 0:
            c = 0
        elif c >= nc:
            c = nc - 1
        b = lut[c]
        while b > 0 and T[b - 1] > x:
            b -= 1  # guard against rounding at the cell edge
        while b < nT and T[b] <= x:
            b += 1  # b = number of thresholds <= x
        hist[b] += 1
    out = np.empty(nT, dtype=np.int64)
    s = 0
    for j in range(nT):
        s += hist[j]
        out[j] = s  # particles with fewer than j+1 thresholds <= x, i.e. x < T[j]
    return out


@njit(cache=True)
def node_sums(w0, w1, w2, y0, y1, a, d, grid, hg, x_lo, x_hi):  # pragma: no cover - numba
    ng = grid.shape[0]
    M = w0.shape[0]
    out = np.empty(ng)
    slope = np.zeros(ng)
    for gi in range(ng):
        g = grid[gi]
        h = hg[gi]
        jl = max(0, int(np.ceil((x_lo[gi] - a) / d)))
        jh = min(M - 1, int(np.floor((x_hi[gi] - a) / d)))
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        t0 = 0.0
        t1 = 0.0
        for j in range(jl, jh + 1):
            c = a + j * d
            x = c - g
            u = x / h
            kq = np.exp(-0.5 * u * u)
            s0 += kq * w0[j]
            s1 += kq * (w0[j] * x + w1[j])
            s2 += kq * (w0[j] * x * x + 2.0 * x * w1[j] + w2[j])
            t0 += kq * y0[j]
            t1 += kq * (y0[j] * x + y1[j])
        if s0 <= 0.0:
            out[gi] = np.nan
        else:
            det = s0 * s2 - s1 * s1
            if det > 1e-12 * s0 * s2:
                out[gi] = (s2 * t0 - s1 * t1) / det
                slope[gi] = (s0 * t1 - s1 * t0) / det
            else:
                out[gi] = t0 / s0
    return out, slope


@njit(cache=True)
def exact_nodes(kt, vt, grid, hg, x_lo, x_hi, idx, out, slope):  # pragma: no cover - numba
    """floored nodes summed over the particles themselves (the library's formulas): kt, vt are the
    particles of one tail; the rank window is exactly the particles in [x_lo, x_hi]"""
    for q in range(idx.shape[0]):
        gi = idx[q]
        g = grid[gi]
        h = hg[gi]
        a = x_lo[gi]
        b = x_hi[gi]
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        t0 = 0.0
        t1 = 0.0
        for i in range(kt.shape[0]):
            x = kt[i]
            if x >= a and x <= b:
                u = (x - g) / h
                wt = np.exp(-0.5 * u * u)
                s0 += wt
                s1 += wt * u
                s2 += wt * u * u
                t0 += wt * vt[i]
                t1 += wt * u * vt[i]
        if s0 <= 0.0:
            out[gi] = np.nan
        else:
            det = s0 * s2 - s1 * s1
            if det > 1e-12 * s0 * s2:
                out[gi] = (s2 * t0 - s1 * t1) / det
                slope[gi] = (s0 * t1 - s1 * t0) / det / h
            else:
                out[gi] = t0 / s0


def regression(k, v, h, cfg, deflate=False):
    """nodes, node values, slopes and the per-node bandwidths; None if the cloud is degenerate"""
    n = k.size
    i_lo = min(int(cfg.quantile_clip * n), n - 1)
    i_hi = max(int((1.0 - cfg.quantile_clip) * n) - 1, 0)
    part = np.partition(k, [i_lo, i_hi])
    q_lo = float(part[i_lo])
    q_hi = float(part[i_hi])
    if not q_hi > q_lo:
        return None
    kreg = np.linspace(q_lo, q_hi, cfg.n_regression_points)
    ng = kreg.size
    mw = max(cfg.min_window, int(cfg.min_window_fraction * n))
    w = 4.0
    thr = np.concatenate([kreg - w * h, kreg + w * h, kreg])
    order = np.argsort(thr, kind="stable")
    T = thr[order]
    lo, hi = minmax(k)
    nc = 4096
    inv_cell = nc / max(hi - lo, 1e-300)
    lut = np.searchsorted(T, lo + np.arange(nc) / inv_cell, side="right").astype(np.int64)
    cl = np.empty(3 * ng, dtype=np.int64)
    cl[order] = count_less(k, T, lo, inv_cell, lut)
    c_lo, c_hi, c_mid = cl[:ng], cl[ng : 2 * ng], cl[2 * ng :]
    floored = ((c_hi - c_lo) < mw) & (n >= mw)
    hg = np.full(ng, h)
    x_lo = kreg - w * h
    x_hi = kreg + w * h
    if floored.any():
        rl = np.maximum(0, c_mid[floored] - mw // 2)
        rh = np.minimum(n, rl + mw)
        rl = np.maximum(0, rh - mw)  # particle.py:85-87
        ranks = np.unique(np.concatenate([rl, rh - 1]))
        p2 = np.partition(k, ranks)  # exact order statistics
        e_lo = p2[rl]
        e_hi = p2[rh - 1]
        g = kreg[floored]
        span = np.maximum(np.abs(e_lo - g), np.abs(e_hi - g))
        hgf = np.maximum(h, span / w)  # particle.py:88-89
        hg[floored] = hgf
        x_lo[floored] = e_lo
        x_hi[floored] = e_hi
    d = h / BIN_FRACTION
    a = lo - d
    M = int(np.ceil((hi - lo) / d)) + 4
    A = np.zeros((5, M))
    bin5(k, v, a, 1.0 / d, A[0], A[1], A[2], A[3], A[4])
    hb = np.sqrt(hg * hg - d * d / 6.0) if deflate else hg
    m, slope = node_sums(A[0], A[1], A[2], A[3], A[4], a, d, kreg, hb, x_lo, x_hi)
    if floored.any():
        idx = np.where(floored)[0]
        for side in (idx[idx < ng // 2], idx[idx >= ng // 2]):
            if side.size:
                sel = (k >= x_lo[side].min()) & (k <= x_hi[side].max())
                kt = np.ascontiguousarray(k[sel])
                vt = np.ascontiguousarray(v[sel])
                exact_nodes(kt, vt, kreg, hg, x_lo, x_hi, side, m, slope)
    return kreg, q_lo, q_hi, m, slope, hg


def conditional_variance_estimate(
    k: FloatArray,
    v: FloatArray,
    grid: FloatArray,
    h: float,
    cfg: ParticleConfig,
    tail_slope: float | None = None,
    *,
    deflate: bool = False,
) -> FloatArray:
    """Drop-in for ``volsto.calibration.particle.conditional_variance_estimate``."""
    r = regression(k, v, h, cfg, deflate)
    if r is None:
        return np.full(grid.shape, float(v.mean()))
    kreg, q_lo, q_hi, m, slope, _ = r
    return particle._finish_estimate(m, slope, kreg, q_lo, q_hi, grid, h, cfg, v, tail_slope)
