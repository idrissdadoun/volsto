"""Black–Scholes / Black-76 prices, Greeks and a robust vectorised implied-vol inversion.

All functions broadcast over numpy arrays.  Core formulas are in forward terms (Black-76,
``black_*``); spot-based helpers (``bs_*``) take ``(S, K, T, vol, r, q)`` with flat continuously
compounded rates.  Checked by ``tests/test_bs.py`` (put–call parity, implied-vol round trip,
Greeks vs finite differences).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri

FloatArray = NDArray[np.float64]

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def _norm_pdf(x: FloatArray) -> FloatArray:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _arrays(*xs: ArrayLike) -> tuple[FloatArray, ...]:
    return tuple(np.asarray(x, dtype=np.float64) for x in xs)


def _d1_d2(F: FloatArray, K: FloatArray, s: FloatArray) -> tuple[FloatArray, FloatArray]:
    """``d1,2 = (ln(F/K) ± s²/2)/s`` with ``s = σ√T``; ``s = 0`` handled by the caller."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lm = np.log(F / K)
        d1 = lm / s + 0.5 * s
        d2 = d1 - s
    return d1, d2


def black_price(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, vol: ArrayLike, cp: ArrayLike, df: ArrayLike = 1.0
) -> FloatArray:
    """Black-76 price ``df · cp · [F N(cp d1) − K N(cp d2)]``.

    ``cp = +1`` call, ``−1`` put.  Zero total variance returns discounted intrinsic value.
    Source: Black (1976); checked by ``tests/test_bs.py::test_put_call_parity``.
    """
    F_, K_, T_, v_, cp_, df_ = _arrays(F, K, T, vol, cp, df)
    s = v_ * np.sqrt(np.maximum(T_, 0.0))
    d1, d2 = _d1_d2(F_, K_, s)
    intrinsic = np.maximum(cp_ * (F_ - K_), 0.0)
    live = cp_ * (F_ * ndtr(cp_ * d1) - K_ * ndtr(cp_ * d2))
    price = np.where(s > 0, live, intrinsic)
    return np.asarray(df_ * price, dtype=np.float64)


def black_vega(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, vol: ArrayLike, df: ArrayLike = 1.0
) -> FloatArray:
    """``∂Price/∂σ = df · F φ(d1) √T`` (same for calls and puts)."""
    F_, K_, T_, v_, df_ = _arrays(F, K, T, vol, df)
    sqT = np.sqrt(np.maximum(T_, 0.0))
    s = v_ * sqT
    d1, _ = _d1_d2(F_, K_, s)
    out = np.where(s > 0, df_ * F_ * _norm_pdf(d1) * sqT, 0.0)
    return np.asarray(out, dtype=np.float64)


def black_delta_forward(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, vol: ArrayLike, cp: ArrayLike, df: ArrayLike = 1.0
) -> FloatArray:
    """``∂Price/∂F = df · cp · N(cp d1)``."""
    F_, K_, T_, v_, cp_, df_ = _arrays(F, K, T, vol, cp, df)
    s = v_ * np.sqrt(np.maximum(T_, 0.0))
    d1, _ = _d1_d2(F_, K_, s)
    live = df_ * cp_ * ndtr(cp_ * d1)
    intrinsic = df_ * cp_ * (cp_ * (F_ - K_) > 0)
    return np.asarray(np.where(s > 0, live, intrinsic), dtype=np.float64)


# --------------------------------------------------------------------------------------------
# Spot-based Black–Scholes with flat r, q
# --------------------------------------------------------------------------------------------


def _spot_inputs(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, vol: ArrayLike, r: ArrayLike, q: ArrayLike
) -> tuple[FloatArray, ...]:
    S_, K_, T_, v_, r_, q_ = _arrays(S, K, T, vol, r, q)
    F = S_ * np.exp((r_ - q_) * T_)
    df = np.exp(-r_ * T_)
    return S_, K_, T_, v_, r_, q_, F, df


def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    vol: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    cp: ArrayLike,
) -> FloatArray:
    """Black–Scholes price with continuous dividend yield ``q``."""
    _, K_, T_, v_, _, _, F, df = _spot_inputs(S, K, T, vol, r, q)
    return black_price(F, K_, T_, v_, cp, df)


def bs_delta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    vol: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    cp: ArrayLike,
) -> FloatArray:
    """``Δ = cp e^{-qT} N(cp d1)``."""
    _, K_, T_, v_, _, q_, F, _ = _spot_inputs(S, K, T, vol, r, q)
    cp_ = np.asarray(cp, dtype=np.float64)
    s = v_ * np.sqrt(T_)
    d1, _ = _d1_d2(F, K_, s)
    return np.asarray(cp_ * np.exp(-q_ * T_) * ndtr(cp_ * d1), dtype=np.float64)


def bs_gamma(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, vol: ArrayLike, r: ArrayLike, q: ArrayLike
) -> FloatArray:
    """``Γ = e^{-qT} φ(d1) / (S σ √T)``."""
    S_, K_, T_, v_, _, q_, F, _ = _spot_inputs(S, K, T, vol, r, q)
    s = v_ * np.sqrt(T_)
    d1, _ = _d1_d2(F, K_, s)
    return np.asarray(np.exp(-q_ * T_) * _norm_pdf(d1) / (S_ * s), dtype=np.float64)


def bs_vega(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, vol: ArrayLike, r: ArrayLike, q: ArrayLike
) -> FloatArray:
    """``∂P/∂σ = S e^{-qT} φ(d1) √T``."""
    _, K_, T_, v_, _, _, F, df = _spot_inputs(S, K, T, vol, r, q)
    return black_vega(F, K_, T_, v_, df)


def bs_theta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    vol: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    cp: ArrayLike,
) -> FloatArray:
    """Calendar theta ``∂P/∂t = −∂P/∂T``:

    ``θ = −S e^{-qT} φ(d1) σ/(2√T) − cp r K e^{-rT} N(cp d2) + cp q S e^{-qT} N(cp d1)``.
    """
    S_, K_, T_, v_, r_, q_, F, df = _spot_inputs(S, K, T, vol, r, q)
    cp_ = np.asarray(cp, dtype=np.float64)
    sqT = np.sqrt(T_)
    d1, d2 = _d1_d2(F, K_, v_ * sqT)
    eq = np.exp(-q_ * T_)
    out = (
        -S_ * eq * _norm_pdf(d1) * v_ / (2.0 * sqT)
        - cp_ * r_ * K_ * df * ndtr(cp_ * d2)
        + cp_ * q_ * S_ * eq * ndtr(cp_ * d1)
    )
    return np.asarray(out, dtype=np.float64)


def bs_rho(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    vol: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    cp: ArrayLike,
) -> FloatArray:
    """``ρ = cp K T e^{-rT} N(cp d2)``."""
    _, K_, T_, v_, _, _, F, df = _spot_inputs(S, K, T, vol, r, q)
    cp_ = np.asarray(cp, dtype=np.float64)
    _, d2 = _d1_d2(F, K_, v_ * np.sqrt(T_))
    return np.asarray(cp_ * K_ * T_ * df * ndtr(cp_ * d2), dtype=np.float64)


def bs_vanna(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, vol: ArrayLike, r: ArrayLike, q: ArrayLike
) -> FloatArray:
    """``∂²P/∂S∂σ = −e^{-qT} φ(d1) d2/σ``."""
    _, K_, T_, v_, _, q_, F, _ = _spot_inputs(S, K, T, vol, r, q)
    d1, d2 = _d1_d2(F, K_, v_ * np.sqrt(T_))
    return np.asarray(-np.exp(-q_ * T_) * _norm_pdf(d1) * d2 / v_, dtype=np.float64)


def bs_volga(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, vol: ArrayLike, r: ArrayLike, q: ArrayLike
) -> FloatArray:
    """``∂²P/∂σ² = vega · d1 d2 / σ``."""
    S_, K_, T_, v_, _, q_, F, _ = _spot_inputs(S, K, T, vol, r, q)
    sqT = np.sqrt(T_)
    d1, d2 = _d1_d2(F, K_, v_ * sqT)
    vega = S_ * np.exp(-q_ * T_) * _norm_pdf(d1) * sqT
    return np.asarray(vega * d1 * d2 / v_, dtype=np.float64)


# --------------------------------------------------------------------------------------------
# Implied volatility
# --------------------------------------------------------------------------------------------


def implied_vol(
    price: ArrayLike,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    cp: ArrayLike,
    df: ArrayLike = 1.0,
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> FloatArray:
    """Invert Black-76 for σ, vectorised, safeguarded Newton on ``s = σ√T``.

    The option is mapped to the out-of-the-money one by put–call parity, so the objective is a
    small positive time value.  Newton steps that leave the bracket ``[s_lo, s_hi]`` fall back to
    bisection.  Prices outside ``[intrinsic, upper bound]`` (or ``T ≤ 0``) return ``nan``.
    Checked by ``tests/test_bs.py::test_implied_vol_round_trip``.
    """
    p_, F_, K_, T_, cp_, df_ = np.broadcast_arrays(*_arrays(price, F, K, T, cp, df))
    shape = p_.shape
    p = p_.ravel() / df_.ravel()
    F1 = F_.ravel()
    K1 = K_.ravel()
    T1 = T_.ravel()
    cp1 = cp_.ravel()
    x = np.log(K1 / F1)  # log-moneyness
    # map to OTM option: call if x >= 0 else put; parity: C - P = F - K
    otm_is_call = x >= 0.0
    target = np.where(
        otm_is_call,
        np.where(cp1 > 0, p, p + (F1 - K1)),
        np.where(cp1 > 0, p - (F1 - K1), p),
    )
    upper = np.where(otm_is_call, F1, K1)  # OTM price as s -> inf
    valid = (T1 > 0) & (target >= -1e-14 * F1) & (target < upper) & np.isfinite(target)
    out = np.full(p.shape, np.nan)
    # exact zero time value -> zero vol
    zero_tv = valid & (target <= 0.0)
    out[zero_tv] = 0.0
    idx = np.flatnonzero(valid & ~zero_tv)
    if idx.size == 0:
        return out.reshape(shape)
    Fv, Kv, xv, tv, cpv = (a[idx] for a in (F1, K1, x, target, np.where(otm_is_call, 1.0, -1.0)))

    def f_and_vega(s: FloatArray) -> tuple[FloatArray, FloatArray]:
        d1 = -xv / s + 0.5 * s  # ln(F/K)/s + s/2
        d2 = d1 - s
        pr = cpv * (Fv * ndtr(cpv * d1) - Kv * ndtr(cpv * d2))
        return pr - tv, Fv * _norm_pdf(d1)

    # bracket: price(s) increasing in s; s_hi where price >= target (start generous, expand)
    s_lo = np.zeros(idx.size)
    s_hi = np.full(idx.size, 2.0 * np.sqrt(2.0 * np.abs(xv) + 1.0))
    for _ in range(60):
        fh, _ = f_and_vega(s_hi)
        need = fh < 0
        if not np.any(need):
            break
        s_hi = np.where(need, 2.0 * s_hi, s_hi)
    # initial guess: OTM asymptotics / Brenner–Subrahmanyam blend, clipped into the bracket
    s = np.sqrt(2.0 * np.abs(xv)) + _SQRT_2PI * tv / np.where(otm_is_call[idx], Fv, Kv)
    s = np.clip(s, 1e-8, s_hi)
    active = np.ones(idx.size, dtype=bool)
    for _ in range(max_iter):
        fval, vega = f_and_vega(s)
        s_lo = np.where(fval < 0, np.maximum(s_lo, s), s_lo)
        s_hi = np.where(fval > 0, np.minimum(s_hi, s), s_hi)
        converged = (np.abs(fval) <= tol * np.maximum(tv, 1e-300) + 1e-16 * Fv) | (
            s_hi - s_lo <= tol
        )
        active &= ~converged
        if not np.any(active):
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = s - fval / vega
        bad = ~np.isfinite(newton) | (newton <= s_lo) | (newton >= s_hi)
        s_new = np.where(bad, 0.5 * (s_lo + s_hi), newton)
        s = np.where(active, s_new, s)
    out[idx] = s / np.sqrt(T1[idx])
    return out.reshape(shape)


def strike_from_delta(
    delta: ArrayLike, F: ArrayLike, T: ArrayLike, vol: ArrayLike, q: ArrayLike = 0.0
) -> FloatArray:
    """Strike of a call (``delta > 0``) or put (``delta < 0``) quoted by spot delta.

    ``d1 = N⁻¹(Δ_c e^{qT})``, ``Δ_c = 1 + Δ_p`` (forward-adjusted), ``K = F exp(−d1 σ√T + σ²T/2)``.
    """
    d_, F_, T_, v_, q_ = _arrays(delta, F, T, vol, q)
    call_delta = np.where(d_ > 0, d_, 1.0 * np.exp(-q_ * T_) + d_)
    d1 = ndtri(call_delta * np.exp(q_ * T_))
    s = v_ * np.sqrt(T_)
    return np.asarray(F_ * np.exp(-d1 * s + 0.5 * s * s), dtype=np.float64)


def norm_cdf(x: ArrayLike) -> FloatArray:
    return np.asarray(ndtr(np.asarray(x, dtype=np.float64)), dtype=np.float64)


def norm_pdf(x: ArrayLike) -> FloatArray:
    return _norm_pdf(np.asarray(x, dtype=np.float64))
