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

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.multi.draws import check_correlation

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


__all__ = [
    "basket_vol",
    "gaussian_dispersion_moments",
    "gaussian_palladium_call",
    "gaussian_palladium_forward",
    "gaussian_straddle_dispersion",
    "implied_correlation",
    "pairwise_mean_correlation",
]
