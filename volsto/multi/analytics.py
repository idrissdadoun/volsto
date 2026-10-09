"""Gaussian (Bachelier) closed forms for the dispersion products — the intuition layer of the
dispersion study (SPEC §8.5), checked against the Monte Carlo in ``tests/test_multi.py``.

With ``r_i ≈ σ_i W_i(T)`` jointly normal, ``σ_B² = wᵀ Σ w`` is the basket variance and
``σ_{i−B}² = σ_i² + σ_B² − 2 (Σ w)_i`` the variance of ``r_i − r_B``:

* ``E|X| = σ √(2T/π)`` for ``X ~ N(0, σ² T)`` gives the **palladium forward** ``E[D] =
  √(2T/π) Σ_i w_i σ_{i−B}`` and the **straddle dispersion** ``E[Σ w_i |r_i| − |r_B|] =
  √(2T/π) (Σ w_i σ_i − σ_B)``;
* for the call on dispersion, ``D`` is approximated by a normal with the exact first two
  moments of ``Σ w_i |X_i|`` — ``E|X_i||X_j| = (2 σ_i σ_j T/π) (√(1 − ρ_ij²) + ρ_ij arcsin ρ_ij)``
  for a bivariate normal pair (folded-normal product moment) — and priced with Bachelier
  (:func:`gaussian_palladium_call`).

Equal vols ``σ`` and a constant correlation ``ρ`` make the intuition explicit: ``σ_B ≈ σ √ρ``
and ``σ_{i−B} ≈ σ √(1 − ρ)`` for large baskets, so the palladium forward is ``σ √(1−ρ)
√(2T/π)`` against ``σ (1 − √ρ) √(2T/π)`` for the straddle dispersion: the palladium pays
``√(1−ρ)/(1−√ρ)`` times more (2.4× at ρ = 0.5, 3.7× at ρ = 0.7) and its correlation
sensitivity ``−σ/(2√(1−ρ))`` is the smaller one in absolute terms at high correlation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

from volsto.multi.draws import check_correlation

if TYPE_CHECKING:
    from volsto.market.surface import ImpliedSurface

FloatArray = NDArray[np.float64]


def _inputs(
    vols: ArrayLike, correlation: ArrayLike, weights: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    s = np.asarray(vols, dtype=np.float64).ravel()
    c = check_correlation(correlation)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if not (s.size == w.size == c.shape[0]):
        raise ValueError("vols, weights and correlation must agree in size")
    if np.any(s <= 0) or not np.all(np.isfinite(s)):
        raise ValueError("vols must be positive")
    return s, c, w


def basket_vol(vols: ArrayLike, correlation: ArrayLike, weights: ArrayLike) -> float:
    """``σ_B = sqrt(wᵀ Σ w)`` with ``Σ_ij = ρ_ij σ_i σ_j``."""
    s, c, w = _inputs(vols, correlation, weights)
    cov = c * np.outer(s, s)
    return float(np.sqrt(w @ cov @ w))


def pairwise_mean_correlation(correlation: ArrayLike, weights: ArrayLike | None = None) -> float:
    """The weighted mean of the off-diagonal correlations, ``Σ_{i≠j} w_i w_j ρ_ij / Σ_{i≠j} w_i
    w_j`` (equal weights by default) — the desk's "average correlation"."""
    c = check_correlation(correlation)
    n = c.shape[0]
    w = np.ones(n) / n if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    ww = np.outer(w, w)
    np.fill_diagonal(ww, 0.0)
    return float(np.sum(ww * c) / np.sum(ww))


def implied_correlation(basket_vol_: float, vols: ArrayLike, weights: ArrayLike) -> float:
    """The constant correlation reproducing ``σ_B``: ``ρ̄ = (σ_B² − Σ w_i² σ_i²) / Σ_{i≠j} w_i w_j
    σ_i σ_j`` (the index-implied correlation of the desk)."""
    s = np.asarray(vols, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    own = float(np.sum(w * w * s * s))
    cross = float(np.sum(np.outer(w * s, w * s))) - own
    if cross <= 0:
        raise ValueError("need at least two names with positive weights")
    return (float(basket_vol_) ** 2 - own) / cross


def _sigma_i_minus_b(s: FloatArray, c: FloatArray, w: FloatArray) -> FloatArray:
    cov = c * np.outer(s, s)
    sb2 = float(w @ cov @ w)
    cov_ib = cov @ w
    return np.asarray(np.sqrt(np.maximum(s * s + sb2 - 2.0 * cov_ib, 0.0)), dtype=np.float64)


def gaussian_palladium_forward(
    vols: ArrayLike, correlation: ArrayLike, weights: ArrayLike, maturity: float
) -> float:
    """``E[D] = √(2T/π) Σ_i w_i σ_{i−B}``."""
    s, c, w = _inputs(vols, correlation, weights)
    return float(np.sqrt(2.0 * maturity / np.pi) * np.sum(w * _sigma_i_minus_b(s, c, w)))


def gaussian_straddle_dispersion(
    vols: ArrayLike,
    correlation: ArrayLike,
    weights: ArrayLike,
    maturity: float,
    basket_scale: float = 1.0,
) -> float:
    """``E[Σ w_i |r_i| − basket_scale |r_B|] = √(2T/π) (Σ w_i σ_i − basket_scale σ_B)``."""
    s, c, w = _inputs(vols, correlation, weights)
    sb = basket_vol(s, c, w)
    return float(np.sqrt(2.0 * maturity / np.pi) * (np.sum(w * s) - basket_scale * sb))


def gaussian_dispersion_moments(
    vols: ArrayLike, correlation: ArrayLike, weights: ArrayLike, maturity: float
) -> tuple[float, float]:
    """Mean and standard deviation of ``D = Σ w_i |X_i|``, ``X_i = r_i − r_B`` jointly normal
    (the folded-normal product moment of the module docstring)."""
    s, c, w = _inputs(vols, correlation, weights)
    cov = c * np.outer(s, s)
    # covariance of the X_i = r_i − r_B: Σ − Σw 1ᵀ − 1 wᵀΣ + σ_B² 1 1ᵀ
    cw = cov @ w
    sb2 = float(w @ cw)
    cx = cov - np.outer(cw, np.ones_like(w)) - np.outer(np.ones_like(w), cw) + sb2
    sx = np.sqrt(np.maximum(np.diag(cx), 1e-300))
    rho = np.clip(cx / np.outer(sx, sx), -1.0, 1.0)
    T = float(maturity)
    mean = float(np.sqrt(2.0 * T / np.pi) * np.sum(w * sx))
    e_abs_prod = (
        (2.0 * T / np.pi)
        * np.outer(sx, sx)
        * (np.sqrt(np.maximum(1.0 - rho * rho, 0.0)) + rho * np.arcsin(rho))
    )
    second = float(w @ e_abs_prod @ w)
    var = max(second - mean * mean, 0.0)
    return mean, float(np.sqrt(var))


def gaussian_palladium_call(
    vols: ArrayLike, correlation: ArrayLike, weights: ArrayLike, maturity: float, strike: float
) -> float:
    """Bachelier price (undiscounted) of ``(D − K)⁺`` with ``D`` normal at the exact first two
    moments of :func:`gaussian_dispersion_moments` — an approximation (``D ≥ 0`` is a sum of
    folded normals), within a few percent of the Monte Carlo for diversified baskets."""
    from scipy.stats import norm

    m, sd = gaussian_dispersion_moments(vols, correlation, weights, maturity)
    if sd <= 0:
        return max(m - strike, 0.0)
    d = (m - strike) / sd
    return float((m - strike) * norm.cdf(d) + sd * norm.pdf(d))


def margrabe_exchange(sigma_a: float, sigma_e: float, rho: float, T: float) -> float:
    """Margrabe's exchange option on two lognormal assets of unit forwards (recalled; derived by
    change of numeraire): ``E[(X_a − X_e)⁺] = N(½σ√T) − N(−½σ√T)`` with ``σ² = σ_a² + σ_e² −
    2ρσ_aσ_e``.  Checked against :class:`volsto.multi.products.OutperformanceOption`."""
    var = sigma_a * sigma_a + sigma_e * sigma_e - 2.0 * rho * sigma_a * sigma_e
    if var < 0 or T <= 0:
        raise ValueError("need a non-negative spread variance and a positive maturity")
    half = 0.5 * float(np.sqrt(var * T))
    return float(ndtr(half) - ndtr(-half))


def pair_means(x: ArrayLike, antithetic: bool = True) -> FloatArray:
    """The independent samples behind a standard error: the means of antithetic pairs (paths
    ``2i`` and ``2i + 1``), or the paths themselves."""
    v = np.asarray(x, dtype=np.float64)
    if not antithetic:
        return v
    if v.shape[0] % 2:
        raise ValueError("antithetic paths come in pairs")
    return np.asarray(0.5 * (v[0::2] + v[1::2]), dtype=np.float64)


def mean_se(x: ArrayLike, antithetic: bool = True) -> tuple[float, float]:
    """``(mean, standard error)`` on pair means."""
    y = pair_means(x, antithetic)
    return float(y.mean()), float(y.std(ddof=1) / np.sqrt(y.shape[0]))


def ratio_se(a: ArrayLike, b: ArrayLike, antithetic: bool = True) -> tuple[float, float]:
    """``(E[a]/E[b], its standard error)`` by the delta method on pair means — ``a`` and ``b``
    on the same paths: with ``r = ā/b̄``, the standard error of the mean of ``(a − r·b)/b̄``
    (derived; the reference implementation's ``ratio_pm``)."""
    ya, yb = pair_means(a, antithetic), pair_means(b, antithetic)
    r = float(ya.mean() / yb.mean())
    u = (ya - r * yb) / yb.mean()
    return r, float(u.std(ddof=1) / np.sqrt(u.shape[0]))


def kappa_se(d: ArrayLike, v: ArrayLike, antithetic: bool = True) -> tuple[float, float]:
    """``κ = E[d]/√E[v]`` and its delta-method standard error on pair means (``d`` the
    dispersion, ``v`` the dispersion variance, the same paths): the standard error of the mean of
    ``(d − d̄)/√v̄ − ½κ(v − v̄)/v̄`` (derived)."""
    yd, yv = pair_means(d, antithetic), pair_means(v, antithetic)
    md, mv = float(yd.mean()), float(yv.mean())
    kappa = md / float(np.sqrt(mv))
    u = (yd - md) / np.sqrt(mv) - 0.5 * kappa * (yv - mv) / mv
    return kappa, float(u.std(ddof=1) / np.sqrt(u.shape[0]))


def _simpson(y: FloatArray, h: float) -> float:
    """Simpson's rule on an odd number of equally spaced points."""
    return float(h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum()))


def strip_second_moment(
    surface: ImpliedSurface,
    T: float,
    *,
    splits: Sequence[float] | None = None,
    n_half: int = 2000,
) -> float | tuple[float, FloatArray]:
    """``E[R²]`` of ``R = S_T/S_0 − 1`` from the vanilla strip of an implied surface (SPEC §8.7):

        E[R²] = (f − 1)² + 2·∫_0^f P(K) dK + 2·∫_f^∞ C(K) dK,        f = F(T)/S_0,

    in units of the spot and undiscounted.  Simpson's rule in ``k = ln(K/F)`` (``dK = K dk``) on
    ``n_half`` intervals each side of the forward over ``±8a`` with ``a = max(σ_ATM√T, 0.1)``,
    wide enough for ``K/S_0`` in ``[0.2, 3]`` — the grid of ``disp_smile.build_marginal`` — and
    beyond it the lognormal tails at the edge vols.

    ``splits``: increasing log-moneyness levels; the function then also returns the strip part
    of ``E[R²]`` (everything but ``(f − 1)²``) by region, ``len(splits) + 1`` numbers: below the
    first split, between consecutive ones, above the last (trapezoids on the same grid, summing
    to the Simpson value within its quadrature error; the lognormal tails go to the outer
    regions).

    Lognormal check: a flat vol ``σ`` gives ``f²·e^{σ²T} − 2f + 1`` (derived; tested)."""
    if n_half < 2 or n_half % 2:
        raise ValueError("n_half must be even and at least 2")
    f = float(surface.forward_curve.forward(T)) / float(surface.forward_curve.spot)
    a = max(float(surface.atm_vol(T)) * np.sqrt(T), 0.1)
    k_lo = min(-8.0 * a, float(np.log(0.2 / f)))
    k_hi = max(8.0 * a, float(np.log(3.0 / f)))
    k = np.concatenate([np.linspace(k_lo, 0.0, n_half + 1), np.linspace(0.0, k_hi, n_half + 1)[1:]])
    s = np.sqrt(np.maximum(np.asarray(surface.total_variance(k, T), dtype=np.float64), 1e-12))
    strike = f * np.exp(k)
    d1 = -k / s + 0.5 * s
    call = f * ndtr(d1) - strike * ndtr(d1 - s)
    put = call - (f - strike)
    put_side = _simpson(put[: n_half + 1] * strike[: n_half + 1], -k_lo / n_half)
    call_side = _simpson(call[n_half:] * strike[n_half:], k_hi / n_half)

    def tail(edge: int, upper: bool) -> float:
        """``E[((X − K)⁺)²]`` above the grid or ``E[((K − X)⁺)²]`` below it, lognormal."""
        K, v = float(strike[edge]), float(s[edge])
        e1 = (np.log(f / K) + 0.5 * v * v) / v
        e2, e0 = e1 - v, e1 + v
        if upper:
            return float(f * f * np.exp(v * v) * ndtr(e0) - 2 * K * f * ndtr(e1) + K * K * ndtr(e2))
        return float(K * K * ndtr(-e2) - 2 * K * f * ndtr(-e1) + f * f * np.exp(v * v) * ndtr(-e0))

    up_tail, dn_tail = tail(-1, True), tail(0, False)
    total = 2.0 * (put_side + call_side) + up_tail + dn_tail + (f - 1.0) ** 2
    if splits is None:
        return float(total)
    cuts = np.asarray(splits, dtype=np.float64)
    if cuts.ndim != 1 or np.any(np.diff(cuts) <= 0):
        raise ValueError("splits must be increasing")
    otm = np.where(k < 0.0, put, call) * strike
    cells = (otm[1:] + otm[:-1]) * np.diff(k)  # twice the trapezoid: the factor 2 of the strip
    region = np.searchsorted(cuts, 0.5 * (k[1:] + k[:-1]))
    parts = np.bincount(region, weights=cells, minlength=cuts.size + 1)
    parts[0] += dn_tail
    parts[-1] += up_tail
    return float(total), np.asarray(parts, dtype=np.float64)


__all__ = [
    "basket_vol",
    "gaussian_dispersion_moments",
    "gaussian_palladium_call",
    "gaussian_palladium_forward",
    "gaussian_straddle_dispersion",
    "implied_correlation",
    "kappa_se",
    "margrabe_exchange",
    "mean_se",
    "pair_means",
    "pairwise_mean_correlation",
    "ratio_se",
    "strip_second_moment",
]
