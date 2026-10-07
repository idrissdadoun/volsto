"""Frozen copy of the m6 regression estimator: ``kernel_regression`` and
``conditional_variance_estimate`` exactly as they stood in ``volsto/calibration/particle.py`` at
calibration code tag ``m6`` before the function was split into a regression stage and
``_finish_estimate`` (the binned-estimator PR).  The text below the imports is that source,
unedited.

Used by ``tests/test_lsv.py`` to prove that the default path of the library is still this
estimator, bit for bit.  Test-only: nothing in ``volsto`` imports it.  Do not edit; a deliberate
change of the default estimator replaces this file together with a bump of the code tag.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from volsto._numba import njit, prange
from volsto.config import ParticleConfig

FloatArray = NDArray[np.float64]


@njit(parallel=True, cache=True)
def kernel_regression(
    ks: FloatArray,
    vs: FloatArray,
    grid: FloatArray,
    h: float,
    gaussian: bool,
    local_linear: bool,
    min_window: int,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64]]:  # pragma: no cover - numba
    """``E[v | k]`` on ``grid`` from particles sorted by ``k`` with a truncated window (``4h``
    Gaussian, ``h`` quartic): Nadaraya–Watson or local-linear (weighted least squares with a
    slope, which cancels the design bias ``h² m' f'/f``).  Returns the estimate (``nan`` where the
    window is empty or degenerate), its slope ``dm/dk`` (local-linear only, else 0) and the
    window counts."""
    ng = grid.shape[0]
    out = np.empty(ng)
    slope = np.zeros(ng)
    cnt = np.empty(ng, dtype=np.int64)
    w = 4.0 if gaussian else 1.0
    n = ks.shape[0]
    for g in prange(ng):
        hg = h
        lo = np.searchsorted(ks, grid[g] - w * hg)
        hi = np.searchsorted(ks, grid[g] + w * hg)
        if hi - lo < min_window and n >= min_window:
            # k-NN floor: widen to the nearest min_window particles and rescale the bandwidth
            c = np.searchsorted(ks, grid[g])
            lo = max(0, c - min_window // 2)
            hi = min(n, lo + min_window)
            lo = max(0, hi - min_window)
            span = max(abs(ks[lo] - grid[g]), abs(ks[hi - 1] - grid[g]))
            hg = max(h, span / w)
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        t0 = 0.0
        t1 = 0.0
        for i in range(lo, hi):
            u = (ks[i] - grid[g]) / hg
            if gaussian:
                wt = np.exp(-0.5 * u * u)
            else:
                d = 1.0 - u * u
                wt = d * d if d > 0.0 else 0.0
            s0 += wt
            s1 += wt * u
            s2 += wt * u * u
            t0 += wt * vs[i]
            t1 += wt * u * vs[i]
        cnt[g] = hi - lo
        if s0 <= 0.0:
            out[g] = np.nan
        elif local_linear:
            det = s0 * s2 - s1 * s1
            if det > 1e-12 * s0 * s2:
                out[g] = (s2 * t0 - s1 * t1) / det
                slope[g] = (s0 * t1 - s1 * t0) / det / hg
            else:
                out[g] = t0 / s0
        else:
            out[g] = t0 / s0
    return out, slope, cnt


def conditional_variance_estimate(
    k: FloatArray,
    v: FloatArray,
    grid: FloatArray,
    h: float,
    cfg: ParticleConfig,
    tail_slope: float | None = None,
) -> FloatArray:
    """``E[v | k]`` on the output ``grid`` (the leverage ``k`` grid).

    The regression itself runs on a dense grid of ``cfg.n_regression_points`` spanning the
    cloud's trusted ``[q, 1−q]`` quantile range (M4b: a fixed grid over the whole leverage range
    put 0.025 between regression points while the cloud at the first slices is 0.006 wide and
    ``E[ξ|k]`` varies like ``exp(ω ρ k / (σ√t))``; linear interpolation of that exponential between
    far-apart nodes biased the short-end leverage low by ~1% in variance at ω = 3 — 0.09 vol
    points on the 1m ATM vol — independently of the step size); values inside the range are
    interpolated onto ``grid``, the tails are extended flat or log-linearly / log-quadratically
    (SPEC §4.1 and :class:`~volsto.config.ParticleConfig`).
    """
    order = np.argsort(k, kind="stable")
    ks = np.ascontiguousarray(k[order])
    vs = np.ascontiguousarray(v[order])
    n = ks.size
    q_lo = float(ks[min(int(cfg.quantile_clip * n), n - 1)])
    q_hi = float(ks[max(int((1.0 - cfg.quantile_clip) * n) - 1, 0)])
    if not q_hi > q_lo:
        # degenerate cloud (e.g. all particles at one point): use the plain mean everywhere
        return np.full(grid.shape, float(v.mean()))
    kreg = np.linspace(q_lo, q_hi, cfg.n_regression_points)
    window = max(cfg.min_window, int(cfg.min_window_fraction * n))
    m, slope, _ = kernel_regression(
        ks, vs, kreg, h, cfg.kernel == "gaussian", cfg.regression == "local_linear", window
    )
    good = np.isfinite(m) & (m > 0)
    if not np.any(good):
        return np.full(grid.shape, float(v.mean()))
    if not np.all(good):
        m = np.interp(kreg, kreg[good], m[good])
    if cfg.bias_correction and cfg.regression == "local_linear" and kreg.size >= 5:
        # plug-in correction of the local-linear bias 1/2 h^2 kappa_2 m'' (kappa_2 = 1 Gaussian,
        # 1/7 quartic); m'' by central second differences on a stencil of width ~h (the
        # regression grid is much finer than the bandwidth: differencing neighbouring points
        # would amplify the regression noise by (h/d)^2 before the one-sided clip below)
        kappa2 = 1.0 if cfg.kernel == "gaussian" else 1.0 / 7.0
        d = kreg[1] - kreg[0]
        st = int(np.clip(round(h / d), 1, (kreg.size - 1) // 2))
        m2 = np.zeros_like(m)
        m2[st:-st] = (m[2 * st :] - 2.0 * m[st:-st] + m[: -2 * st]) / (st * d) ** 2
        m2[:st], m2[-st:] = m2[st], m2[-st - 1]
        m = np.maximum(m - 0.5 * h * h * kappa2 * m2, 0.5 * m)
    out = np.interp(grid, kreg, m)
    lo = grid < q_lo
    hi = grid > q_hi
    if cfg.tail_extrapolation in ("sv_slope", "cloud_slope"):
        # model-consistent tail: ln E[V|S] continued with the kernel's conditional slope (the
        # analytic pure-SV slope or the slope fitted over the whole cloud)
        if tail_slope is None:
            raise ValueError("tail_extrapolation='sv_slope'/'cloud_slope' needs tail_slope")
        for edge, mask in ((0, lo), (kreg.size - 1, hi)):
            if np.any(mask):
                expo = tail_slope * (grid[mask] - kreg[edge])
                out[mask] = m[edge] * np.exp(np.clip(expo, -3.0, 3.0))
    elif cfg.tail_extrapolation in ("log_linear", "log_quadratic", "adaptive"):
        # ln m over the outer 10% of the trusted points (at least 5): linear fit for the edge
        # slope, or a quadratic whose slope is only allowed to decay towards zero (then flat)
        n_fit = max(5, kreg.size // 10)
        for edge, mask, pts, sign in (
            (0, lo, slice(0, min(n_fit, kreg.size)), -1.0),
            (kreg.size - 1, hi, slice(max(kreg.size - n_fit, 0), kreg.size), 1.0),
        ):
            if not np.any(mask):
                continue
            x = kreg[pts] - kreg[edge]
            y = np.log(m[pts])
            d = sign * (grid[mask] - kreg[edge])  # distance into the tail (>= 0)
            if x.size >= 3 and np.ptp(x) > 0:
                b = float(np.polyfit(x, y, 1)[0])  # d ln m / dk at the edge
            else:
                b = float(slope[edge] / m[edge])
            b = float(np.clip(b, -50.0, 50.0))
            expo = sign * b * d
            # adaptive: saturating quadratic only where ln m falls into the tail
            quadratic = cfg.tail_extrapolation == "log_quadratic" or (
                cfg.tail_extrapolation == "adaptive" and sign * b < 0
            )
            if quadratic and x.size >= 4 and np.ptp(x) > 0:
                c2, b2, _ = np.polyfit(x, y, 2)
                b2 = float(np.clip(b2, -50.0, 50.0))
                curv = float(c2)
                # keep the quadratic only if it makes |slope| decay into the tail
                if sign * b2 * curv < 0 and abs(b2) > 0:
                    d_flat = abs(b2) / (2.0 * abs(curv))  # slope vanishes here
                    dd = np.minimum(d, d_flat)
                    expo = sign * b2 * dd + curv * dd * dd
                else:
                    expo = sign * b2 * d
            out[mask] = m[edge] * np.exp(np.clip(expo, -3.0, 3.0))
    else:
        out[lo] = m[0]
        out[hi] = m[-1]
    return np.asarray(out, dtype=np.float64)
