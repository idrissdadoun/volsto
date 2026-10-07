"""Dispersion study: the skew-consistent copula, "model S" (spec §3.7; notes App. A, model D).

Latent ``L_i = √r·M + √(1−r)·E_i`` with ``r = min(max(c − s·M, 0.02), 0.98)``: the correlation
of a scenario rises when the common factor falls (``s > 0``).  ``L_i`` is then not standard
normal: its distribution ``F_L(x) = E_M[Φ((x − √r(M)·M)/√(1 − r(M)))]`` is tabulated by
Gauss–Hermite quadrature and the uniforms are ``F_L(L_i)``, so the marginals stay exactly the
smiles'.  ``(c, s)`` are calibrated to the basket's at-the-money straddle and its 90 % put on
the random numbers of the date.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from volsto.studies.disp_copula import Draws
from volsto.studies.disp_smile import Z_MAX, Z_POINTS

FloatArray = NDArray[np.float64]

R_BOUNDS: Final = (0.02, 0.98)
C_BOUNDS: Final = (0.03, 0.97)
S_BOUNDS: Final = (-0.30, 0.60)
X_MAX: Final = 10.0
X_POINTS: Final = 4001
_NODES, _WEIGHTS = np.polynomial.hermite.hermgauss(64)
_M: Final = np.sqrt(2.0) * _NODES
_W: Final = _WEIGHTS / np.sqrt(np.pi)
_X: Final = np.linspace(-X_MAX, X_MAX, X_POINTS)


def scenario_correlation(c: float, s: float, m: FloatArray) -> FloatArray:
    return np.clip(c - s * m, *R_BOUNDS)


def latent_to_normal(c: float, s: float) -> FloatArray:
    """``Φ⁻¹(F_L(x))`` on the grid of :data:`X_POINTS` points over ``±X_MAX``: the map from the
    latent variable to a standard normal one with the same rank."""
    r = scenario_correlation(c, s, _M)
    cdf = ndtr((_X[:, None] - np.sqrt(r)[None, :] * _M[None, :]) / np.sqrt(1.0 - r)[None, :]) @ _W
    return np.asarray(ndtri(np.clip(cdf, 1e-15, 1.0 - 1e-15)), dtype=np.float64)


def lookup(tab: FloatArray, z: FloatArray) -> FloatArray:
    """Performances from standard normal latents ``z`` (``N × n``) by the quantile tables."""
    n = tab.shape[0]
    pos = (z + Z_MAX) * ((Z_POINTS - 1) / (2.0 * Z_MAX))
    np.clip(pos, 0.0, Z_POINTS - 1.000001, out=pos)
    i0 = pos.astype(np.int64)
    frac = pos - i0
    i0 += np.arange(n, dtype=np.int64)[None, :] * Z_POINTS
    flat = tab.ravel()
    lo = flat[i0]
    return np.asarray(lo + frac * (flat[i0 + 1] - lo), dtype=np.float64)


def simulate_s(tab: FloatArray, draws: Draws, c: float, s: float) -> FloatArray:
    """Performances ``X`` (``N × n``) under model S."""
    n = tab.shape[0]
    r = scenario_correlation(c, s, draws.m)
    latent = np.sqrt(r)[:, None] * draws.m[:, None] + np.sqrt(1.0 - r)[:, None] * draws.e[:, :n]
    g = latent_to_normal(c, s)
    pos = (latent + X_MAX) * ((X_POINTS - 1) / (2.0 * X_MAX))
    np.clip(pos, 0.0, X_POINTS - 1.000001, out=pos)
    i0 = pos.astype(np.int64)
    z = g[i0] + (pos - i0) * (g[i0 + 1] - g[i0])
    return lookup(tab, z)


def level_for(tab: FloatArray, draws: Draws, w: FloatArray, s: float, straddle: float) -> float:
    """The ``c`` at which model S with slope ``s`` reprices the basket straddle (clipped to
    :data:`C_BOUNDS`: at a bound the straddle is not repriced, which the caller checks)."""

    def gap(c: float) -> float:
        return float(np.mean(np.abs(simulate_s(tab, draws, c, s) @ w - 1.0))) - straddle

    lo, hi = C_BOUNDS
    if gap(lo) >= 0:
        return lo
    if gap(hi) <= 0:
        return hi
    return float(brentq(gap, lo, hi, xtol=1e-6))


def calibrate_s(
    tab: FloatArray, draws: Draws, w: FloatArray, straddle: float, put_90: float
) -> tuple[float, float, str]:
    """``(c, s, flag)``: the level and the slope at which model S reprices the basket's
    at-the-money straddle and its 90 % put (both undiscounted, per unit of the basket's spot).
    ``flag`` is ``""`` or ``"slope at its bound"`` when no slope in :data:`S_BOUNDS` reprices
    the put (the nearer bound is then used)."""

    def gap(s: float) -> float:
        c = level_for(tab, draws, w, s, straddle)
        b = simulate_s(tab, draws, c, s) @ w
        return float(np.mean(np.maximum(0.9 - b, 0.0))) - put_90

    lo, hi = S_BOUNDS
    g_lo, g_hi = gap(lo), gap(hi)
    if g_lo * g_hi > 0:
        s = lo if abs(g_lo) < abs(g_hi) else hi
        return level_for(tab, draws, w, s, straddle), s, "slope at its bound"
    s = float(brentq(gap, lo, hi, xtol=1e-4))
    return level_for(tab, draws, w, s, straddle), s, ""
